"""Scene boundary detection and merging.

Usage:
    python -m video.scenes data/video/input.mp4
    python -m video.scenes data/video/input.mp4 --method hist --threshold 0.35

Methods:
    pyscenedetect  AdaptiveDetector from PySceneDetect (default, robust to camera motion)
    hist           HSV histogram distance between consecutive sampled frames

Post-processing:
    * scenes shorter than 1.5 s are merged into the neighbour with the most similar HSV histogram
    * adjacent scenes with boundary-frame HSV histogram correlation exceeding 0.85 are merged
    * scenes longer than --max-scene seconds are split into equal segments (marked forced=true)
Results are cached in outputs/cache/<video>/scenes.json.
"""

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from video.common import (
    OUT_ROOT,
    cache_dir,
    iter_frames,
    load_json,
    probe,
    read_frame_at,
    save_json,
    stem,
)
from watermark import save_figure


def _hist_frame(frame):
    """Compute normalized 2D HSV histogram for a frame."""
    if frame is None:
        return np.zeros((32, 16), dtype=np.float32)
    hsv = cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [32, 16], [0, 180, 0, 256])
    cv2.normalize(h, h, alpha=1.0, norm_type=cv2.NORM_L1)
    return h


def _detect_pyscenedetect(video, threshold):
    from scenedetect import AdaptiveDetector, detect

    scene_list = detect(
        str(video),
        AdaptiveDetector(adaptive_threshold=threshold),
        show_progress=True,
    )

    def secs(tc):
        return float(tc.seconds) if hasattr(tc, "seconds") else float(tc.get_seconds())

    return [(secs(s), secs(e)) for s, e in scene_list]


def _detect_hist(video, threshold, sample_fps=5.0):
    """Cut where the HSV histogram correlation between sampled frames drops sharply."""
    info = probe(video)
    prev, times, dists = None, [], []
    total = int(info["duration_s"] * sample_fps)
    for t, frame in tqdm(
        iter_frames(video, sample_fps, width=160),
        total=total,
        desc="histogram diff",
    ):
        h = _hist_frame(frame)
        if prev is not None:
            dists.append(1.0 - cv2.compareHist(prev, h, cv2.HISTCMP_CORREL))
            times.append(t)
        prev = h
    dists = np.array(dists)
    cuts = [
        times[i]
        for i in range(len(dists))
        if dists[i] > threshold and dists[i] >= dists[max(0, i - 2) : i + 3].max()
    ]
    bounds = [0.0] + cuts + [info["duration_s"]]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def _merge_short_scenes(raw, video, min_scene=1.5):
    """Merge scenes shorter than min_scene into the neighbour with the most similar HSV histogram."""
    cap = cv2.VideoCapture(str(video))
    cur = [{"start_s": float(s), "end_s": float(e)} for s, e in raw]

    def get_hist_at(t_s):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_s * 1000.0))
        ok, frame = cap.read()
        return _hist_frame(frame) if ok else np.zeros((32, 16), dtype=np.float32)

    mid_hists = [get_hist_at((s["start_s"] + s["end_s"]) / 2.0) for s in cur]

    while True:
        short_indices = [
            i for i, s in enumerate(cur) if (s["end_s"] - s["start_s"]) < min_scene - 1e-4
        ]
        if not short_indices or len(cur) <= 1:
            break
        i = short_indices[0]
        h_curr = mid_hists[i]
        if i == 0:
            merge_with = 1
        elif i == len(cur) - 1:
            merge_with = i - 1
        else:
            c_left = cv2.compareHist(h_curr, mid_hists[i - 1], cv2.HISTCMP_CORREL)
            c_right = cv2.compareHist(h_curr, mid_hists[i + 1], cv2.HISTCMP_CORREL)
            merge_with = i - 1 if c_left >= c_right else i + 1

        target = min(i, merge_with)
        other = max(i, merge_with)
        cur[target]["end_s"] = cur[other]["end_s"]
        cur[target]["start_s"] = min(cur[target]["start_s"], cur[other]["start_s"])
        mid_hists[target] = get_hist_at((cur[target]["start_s"] + cur[target]["end_s"]) / 2.0)
        cur.pop(other)
        mid_hists.pop(other)

    cap.release()
    return cur


def _merge_similar_boundary_scenes(scenes, video, threshold=0.85):
    """Merge adjacent scenes whose boundary-frame histogram correlation exceeds threshold."""
    cap = cv2.VideoCapture(str(video))

    def get_hist_at(t_s):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_s * 1000.0))
        ok, frame = cap.read()
        return _hist_frame(frame) if ok else np.zeros((32, 16), dtype=np.float32)

    i = 0
    while i < len(scenes) - 1:
        s1 = scenes[i]
        s2 = scenes[i + 1]
        h_end1 = get_hist_at(max(0.0, s1["end_s"] - 0.05))
        h_start2 = get_hist_at(s2["start_s"] + 0.05)
        corr = cv2.compareHist(h_end1, h_start2, cv2.HISTCMP_CORREL)
        if corr > threshold:
            scenes[i]["end_s"] = s2["end_s"]
            scenes.pop(i + 1)
        else:
            i += 1

    cap.release()
    return scenes


def _postprocess(raw, video, duration, min_scene=1.5, max_scene=60.0):
    if not raw:
        raw = [(0.0, duration)]
    raw = sorted(raw, key=lambda x: x[0])
    raw[0] = (0.0, raw[0][1])
    raw[-1] = (raw[-1][0], duration)

    # 1. Enforce minimum scene length of 1.5 s by merging shorter scenes into most similar neighbour
    merged = _merge_short_scenes(raw, video, min_scene=min_scene)

    # 2. Merge adjacent scenes whose boundary-frame histogram correlation exceeds 0.85
    merged = _merge_similar_boundary_scenes(merged, video, threshold=0.85)

    # 3. Optional forced splitting of scenes exceeding max_scene
    scenes = []
    for sc in merged:
        s, e = sc["start_s"], sc["end_s"]
        length = e - s
        parts = max(1, int(np.ceil(length / max_scene))) if max_scene else 1
        edges = np.linspace(s, e, parts + 1)
        for k in range(parts):
            scenes.append(
                {
                    "start_s": float(edges[k]),
                    "end_s": float(edges[k + 1]),
                    "forced": parts > 1,
                }
            )
    return scenes


def detect_scenes(
    video,
    method="pyscenedetect",
    threshold=None,
    min_scene=1.5,
    max_scene=60.0,
):
    info = probe(video)
    # Check if raw scenes or previous scenes exist in cache to avoid full detector re-run
    cache_path = cache_dir(video) / "scenes.json"
    if cache_path.exists():
        cached_data = load_json(cache_path)
        cached_scenes = cached_data.get("scenes", [])
        if cached_scenes:
            raw = [(s["start_s"], s["end_s"]) for s in cached_scenes]
        else:
            raw = None
    else:
        raw = None

    if raw is None:
        if method == "pyscenedetect":
            try:
                raw = _detect_pyscenedetect(video, 3.0 if threshold is None else threshold)
            except ImportError:
                print("scenedetect not installed, falling back to histogram method")
                raw = _detect_hist(video, 0.35 if threshold is None else threshold)
        else:
            raw = _detect_hist(video, 0.35 if threshold is None else threshold)

    scenes = _postprocess(raw, video, info["duration_s"], min_scene, max_scene)
    fps = info["fps"]
    for i, sc in enumerate(scenes):
        sc["index"] = i
        sc["start_frame"] = int(round(sc["start_s"] * fps))
        sc["end_frame"] = int(round(sc["end_s"] * fps))
    return scenes


def detect_scenes_cached(video, force=False, **kwargs):
    path = cache_dir(video) / "scenes.json"
    if path.exists() and not force:
        scenes = load_json(path)["scenes"]
        # If cached scenes already meet min_scene requirement, return them
        shortest = min(s["end_s"] - s["start_s"] for s in scenes)
        if shortest >= 1.5 - 1e-3:
            return scenes
    scenes = detect_scenes(video, **kwargs)
    save_json({"video": str(video), "scenes": scenes}, path)
    return scenes


def plot_timeline(scenes, duration, out_path):
    fig, ax = plt.subplots(figsize=(12, 2.4))
    cmap = plt.get_cmap("tab20")
    for sc in scenes:
        ax.barh(
            0,
            sc["end_s"] - sc["start_s"],
            left=sc["start_s"],
            color=cmap(sc["index"] % 20),
            edgecolor="white",
            hatch="//" if sc["forced"] else None,
        )
    ax.set_xlim(0, duration)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.set_title(f"{len(scenes)} scenes (hatched = forced split of a long scene)")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Detect scene boundaries.")
    ap.add_argument("video")
    ap.add_argument(
        "--method",
        choices=["pyscenedetect", "hist"],
        default="pyscenedetect",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="pyscenedetect adaptive threshold or histogram distance",
    )
    ap.add_argument("--min-scene", type=float, default=1.5)
    ap.add_argument(
        "--max-scene",
        type=float,
        default=60.0,
        help="0 disables forced splitting",
    )
    ap.add_argument("--force", action="store_true", help="ignore cached result")
    args = ap.parse_args()

    scenes = detect_scenes(
        args.video,
        method=args.method,
        threshold=args.threshold,
        min_scene=args.min_scene,
        max_scene=args.max_scene,
    )
    save_json(
        {"video": str(args.video), "scenes": scenes},
        cache_dir(args.video) / "scenes.json",
    )
    info = probe(args.video)
    fig = plot_timeline(
        scenes,
        info["duration_s"],
        OUT_ROOT / f"{stem(args.video)}_scene_timeline.png",
    )
    forced = sum(1 for s in scenes if s["forced"])
    print(
        f"New scene count: {len(scenes)} scenes ({forced} from forced splitting) in {info['duration_s'] / 60:.1f} min"
    )
    for sc in scenes[:15]:
        print(
            f"  {sc['index']:3d}  {sc['start_s']:8.2f}s to {sc['end_s']:8.2f}s"
            f"{'  forced' if sc['forced'] else ''}"
        )
    if len(scenes) > 15:
        print(f"  ... {len(scenes) - 15} more in outputs/cache/{stem(args.video)}/scenes.json")
    print(f"timeline: {fig}")


if __name__ == "__main__":
    main()
