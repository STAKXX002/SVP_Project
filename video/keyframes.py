"""Key frame selection: one sharp, well-exposed frame per scene, ranked by importance.

Usage:
    python -m video.keyframes data/video/input.mp4
    python -m video.keyframes data/video/input.mp4 --samples 9 --top 12

For every scene, --samples frames are sampled and the sharpest well-exposed one wins
(variance of the Laplacian, penalised when underexposed or saturated).
Scene importance is 0.6 x normalised motion + 0.4 x normalised duration when motion has
been computed, otherwise duration only. Top key frames are selected greedily with
diversity constraints: a candidate is skipped if its HSV histogram correlation with
any already selected frame exceeds 0.8 or if it occurs within 30 s of one.

Outputs:
    outputs/keyframes/<video>/scene_XXX.jpg   key frame per scene
    outputs/<video>_keyframes.png             contact sheet of top-12 key frames
    outputs/cache/<video>/keyframes.json
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
    load_json,
    probe,
    resize_to_width,
    save_json,
    stem,
)
from video.motion import scene_scores, suppress_cuts
from video.scenes import detect_scenes_cached
from watermark import save_figure


def sharpness(frame):
    """Estimate image sharpness using variance of Laplacian with exposure penalty."""
    gray = cv2.cvtColor(resize_to_width(frame, 480), cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean = float(gray.mean())
    if mean < 25 or mean > 235:  # nearly black or saturated
        lap *= 0.3
    return lap


def _hist(frame):
    """Normalized 2D HSV histogram for duplicate and diversity testing."""
    hsv = cv2.cvtColor(resize_to_width(frame, 160), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [32, 16], [0, 180, 0, 256])
    cv2.normalize(h, h, alpha=1.0, norm_type=cv2.NORM_L1)
    return h


def best_frame(cap, fps, scene, n_samples):
    """Select the sharpest candidate frame within the central 80 percent of a scene."""
    length = scene["end_s"] - scene["start_s"]
    lo, hi = scene["start_s"] + 0.1 * length, scene["end_s"] - 0.1 * length
    times = (
        np.linspace(lo, hi, n_samples)
        if length > 0.5
        else [scene["start_s"] + length / 2]
    )
    best, best_t, best_score = None, None, -1.0
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        sc = sharpness(frame)
        if sc > best_score:
            best, best_t, best_score = frame, float(t), sc
    return best, best_t, best_score


def select_keyframes(video, samples=9, top=12, dup=0.8, min_time_sep=30.0):
    """Select keyframes per scene and filter top keyframes greedily with diversity constraints."""
    scenes = detect_scenes_cached(video)
    info = probe(video)
    out_dir = OUT_ROOT / "keyframes" / stem(video)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    rows = []
    hists = []

    for sc in tqdm(scenes, desc="key frames"):
        path = out_dir / f"scene_{sc['index']:03d}.jpg"
        # If already cached on disk, reuse to avoid decoding overhead
        if path.exists():
            frame = cv2.imread(str(path))
            t = (sc["start_s"] + sc["end_s"]) / 2.0
            score = sharpness(frame)
        else:
            frame, t, score = best_frame(cap, info["fps"], sc, samples)
            if frame is None:
                continue
            cv2.imwrite(str(path), frame)

        rows.append(
            {
                "scene": sc["index"],
                "time_s": t,
                "sharpness": score,
                "path": str(path),
                "start_s": sc["start_s"],
                "end_s": sc["end_s"],
            }
        )
        hists.append(_hist(frame))

    cap.release()

    if not rows:
        return [], False, []

    dur = np.array([r["end_s"] - r["start_s"] for r in rows])
    importance = dur / max(dur.max(), 1e-6)
    mpath = cache_dir(video) / "motion.json"
    used_motion = False

    if mpath.exists():
        m = load_json(mpath)
        vals = suppress_cuts(m["times"], m["values"], scenes, m["fps"])
        scores = np.array(
            scene_scores(m["times"], vals, [scenes[r["scene"]] for r in rows])
        )
        if scores.max() > 0:
            importance = 0.6 * scores / scores.max() + 0.4 * dur / max(dur.max(), 1e-6)
            used_motion = True

    for i, r in enumerate(rows):
        r["importance"] = float(importance[i])

    # Rank all candidates by importance descending
    candidates = sorted(rows, key=lambda r: -r["importance"])

    # Greedy selection with diversity:
    # Skip a candidate if HSV histogram correlation with any already selected frame > 0.8
    # or if within 30 s of one.
    selected = []
    for c in candidates:
        c_hist = _hist(cv2.imread(c["path"]))
        # Diversity check 1: time separation >= 30 s
        if any(abs(c["time_s"] - s["time_s"]) < min_time_sep for s in selected):
            continue
        # Diversity check 2: HSV histogram correlation <= 0.8
        if any(
            cv2.compareHist(c_hist, s["_hist"], cv2.HISTCMP_CORREL) > dup
            for s in selected
        ):
            continue
        c["_hist"] = c_hist
        selected.append(c)
        if len(selected) == top:
            break

    # Clean up temporary histogram object
    for s in selected:
        s.pop("_hist", None)

    return rows, used_motion, selected


def contact_sheet(chosen, out_path, title):
    """Render top keyframes chronologically in a contact sheet grid."""
    chosen_sorted = sorted(chosen, key=lambda r: r["time_s"])
    cols = min(4, max(1, len(chosen_sorted)))
    nrows = int(np.ceil(len(chosen_sorted) / cols))
    fig, axes = plt.subplots(
        nrows, cols, figsize=(3.4 * cols, 2.6 * nrows), squeeze=False
    )
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), chosen_sorted):
        img = cv2.cvtColor(cv2.imread(r["path"]), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        mm, ss = divmod(int(r["time_s"]), 60)
        ax.set_title(f"scene {r['scene']}  {mm:02d}:{ss:02d}", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, dpi=130)
    plt.close(fig)
    return out_path, len(chosen_sorted)


def main():
    ap = argparse.ArgumentParser(description="Select key frames per scene.")
    ap.add_argument("video")
    ap.add_argument(
        "--samples",
        type=int,
        default=9,
        help="candidate frames per scene",
    )
    ap.add_argument("--top", type=int, default=12, help="frames on contact sheet")
    ap.add_argument(
        "--dup",
        type=float,
        default=0.80,
        help="histogram correlation diversity threshold (skip if > dup)",
    )
    ap.add_argument(
        "--time-sep",
        type=float,
        default=30.0,
        help="minimum seconds between selected keyframes",
    )
    args = ap.parse_args()

    rows, used_motion, selected = select_keyframes(
        args.video,
        samples=args.samples,
        top=args.top,
        dup=args.dup,
        min_time_sep=args.time_sep,
    )
    save_json(
        {
            "video": args.video,
            "keyframes": rows,
            "selected_top12": selected,
            "importance_uses_motion": used_motion,
        },
        cache_dir(args.video) / "keyframes.json",
    )
    label = (
        "motion + duration"
        if used_motion
        else "duration only (run video.motion first for motion)"
    )
    sheet, n = contact_sheet(
        selected,
        OUT_ROOT / f"{stem(args.video)}_keyframes.png",
        f"Top {len(selected)} diverse key frames by importance ({label})",
    )
    print(
        f"{len(rows)} key frames saved to outputs/keyframes/{stem(args.video)}/ (selected top {len(selected)} with diversity)"
    )
    print(f"importance: {label}")
    print(f"contact sheet ({n} frames): {sheet}")


if __name__ == "__main__":
    main()
