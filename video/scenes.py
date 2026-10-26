"""Scene boundary detection.

Usage:
    python -m video.scenes data/video/input.mp4
    python -m video.scenes data/video/input.mp4 --method hist --threshold 0.35

Methods:
    pyscenedetect  AdaptiveDetector from PySceneDetect (default, robust to camera motion)
    hist           HSV histogram distance between consecutive sampled frames (no extra dependency)

Post-processing:
    * scenes shorter than --min-scene seconds are merged into the previous scene
    * scenes longer than --max-scene seconds are split into equal segments and marked forced=true,
      so that a single-shot video (lecture, surveillance) still gives usable segments downstream
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

from video.common import OUT_ROOT, cache_dir, iter_frames, load_json, probe, save_json, stem
from watermark import save_figure


def _detect_pyscenedetect(video, threshold):
    from scenedetect import AdaptiveDetector, detect

    scene_list = detect(str(video), AdaptiveDetector(adaptive_threshold=threshold), show_progress=True)
    def secs(tc):
        return float(tc.seconds) if hasattr(tc, "seconds") else float(tc.get_seconds())

    return [(secs(s), secs(e)) for s, e in scene_list]


def _detect_hist(video, threshold, sample_fps=5.0):
    """Cut where the HSV histogram correlation between sampled frames drops sharply."""
    info = probe(video)
    prev, times, dists = None, [], []
    total = int(info["duration_s"] * sample_fps)
    for t, frame in tqdm(iter_frames(video, sample_fps, width=160), total=total, desc="histogram diff"):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [32, 16], [0, 180, 0, 256])
        cv2.normalize(h, h, alpha=1.0, norm_type=cv2.NORM_L1)
        if prev is not None:
            dists.append(1.0 - cv2.compareHist(prev, h, cv2.HISTCMP_CORREL))
            times.append(t)
        prev = h
    dists = np.array(dists)
    cuts = [times[i] for i in range(len(dists))
            if dists[i] > threshold and dists[i] >= dists[max(0, i - 2):i + 3].max()]
    bounds = [0.0] + cuts + [info["duration_s"]]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def _postprocess(raw, duration, min_scene, max_scene):
    if not raw:
        raw = [(0.0, duration)]
    raw = sorted(raw)
    raw[0] = (0.0, raw[0][1])
    raw[-1] = (raw[-1][0], duration)

    merged = []
    for s, e in raw:
        if merged and (e - s) < min_scene:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    if len(merged) > 1 and (merged[0][1] - merged[0][0]) < min_scene:
        merged[1] = (merged[0][0], merged[1][1])
        merged.pop(0)

    scenes = []
    for s, e in merged:
        length = e - s
        parts = max(1, int(np.ceil(length / max_scene))) if max_scene else 1
        edges = np.linspace(s, e, parts + 1)
        for k in range(parts):
            scenes.append({"start_s": float(edges[k]), "end_s": float(edges[k + 1]),
                           "forced": parts > 1})
    return scenes


def detect_scenes(video, method="pyscenedetect", threshold=None, min_scene=1.0, max_scene=60.0):
    info = probe(video)
    if method == "pyscenedetect":
        try:
            raw = _detect_pyscenedetect(video, 3.0 if threshold is None else threshold)
        except ImportError:
            print("scenedetect not installed, falling back to histogram method")
            raw = _detect_hist(video, 0.35 if threshold is None else threshold)
    else:
        raw = _detect_hist(video, 0.35 if threshold is None else threshold)
    scenes = _postprocess(raw, info["duration_s"], min_scene, max_scene)
    fps = info["fps"]
    for i, sc in enumerate(scenes):
        sc["index"] = i
        sc["start_frame"] = int(round(sc["start_s"] * fps))
        sc["end_frame"] = int(round(sc["end_s"] * fps))
    return scenes


def detect_scenes_cached(video, force=False, **kwargs):
    path = cache_dir(video) / "scenes.json"
    if path.exists() and not force:
        return load_json(path)["scenes"]
    scenes = detect_scenes(video, **kwargs)
    save_json({"video": str(video), "scenes": scenes}, path)
    return scenes


def plot_timeline(scenes, duration, out_path):
    fig, ax = plt.subplots(figsize=(12, 2.2 + 0.0 * len(scenes)))
    cmap = plt.get_cmap("tab20")
    for sc in scenes:
        ax.barh(0, sc["end_s"] - sc["start_s"], left=sc["start_s"], color=cmap(sc["index"] % 20),
                edgecolor="white", hatch="//" if sc["forced"] else None)
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
    ap.add_argument("--method", choices=["pyscenedetect", "hist"], default="pyscenedetect")
    ap.add_argument("--threshold", type=float, default=None,
                    help="pyscenedetect adaptive threshold (default 3.0) or histogram distance (default 0.35)")
    ap.add_argument("--min-scene", type=float, default=1.0)
    ap.add_argument("--max-scene", type=float, default=60.0, help="0 disables forced splitting")
    ap.add_argument("--force", action="store_true", help="ignore cached result")
    args = ap.parse_args()

    scenes = detect_scenes_cached(args.video, force=args.force, method=args.method,
                                  threshold=args.threshold, min_scene=args.min_scene,
                                  max_scene=args.max_scene)
    info = probe(args.video)
    fig = plot_timeline(scenes, info["duration_s"], OUT_ROOT / f"{stem(args.video)}_scene_timeline.png")
    forced = sum(1 for s in scenes if s["forced"])
    print(f"{len(scenes)} scenes ({forced} from forced splitting) in {info['duration_s'] / 60:.1f} min")
    for sc in scenes[:15]:
        print(f"  {sc['index']:3d}  {sc['start_s']:8.2f}s to {sc['end_s']:8.2f}s{'  forced' if sc['forced'] else ''}")
    if len(scenes) > 15:
        print(f"  ... {len(scenes) - 15} more in outputs/cache/{stem(args.video)}/scenes.json")
    print(f"timeline: {fig}")
    if len(scenes) - forced <= 2:
        print("NOTE: very few natural cuts found. If the video is one continuous shot this is expected;"
              " otherwise lower --threshold.")


if __name__ == "__main__":
    main()
