"""Key frame selection: one sharp, well-exposed frame per scene, ranked by importance.

Usage:
    python -m video.keyframes data/video/input.mp4
    python -m video.keyframes data/video/input.mp4 --samples 9 --top 12

For every scene, --samples frames are taken evenly from the middle 80 percent of the scene and the
sharpest well-exposed one wins (variance of the Laplacian, penalised when almost black or white).
Scene importance is 0.6 x normalised motion + 0.4 x normalised duration when motion has been
computed (python -m video.motion), otherwise duration only. Near-duplicate key frames
(histogram correlation above --dup) are marked as duplicates and left out of the top list.

Outputs:
    outputs/keyframes/<video>/scene_XXX.jpg   every key frame
    outputs/<video>_keyframes.png             contact sheet of the --top most important key frames
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

from video.common import OUT_ROOT, cache_dir, load_json, probe, resize_to_width, save_json, stem
from video.motion import scene_scores, suppress_cuts
from video.scenes import detect_scenes_cached
from watermark import save_figure


def sharpness(frame):
    gray = cv2.cvtColor(resize_to_width(frame, 480), cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean = float(gray.mean())
    if mean < 25 or mean > 235:  # nearly black or blown out
        lap *= 0.3
    return lap


def _hist(frame):
    hsv = cv2.cvtColor(resize_to_width(frame, 160), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [32, 16], [0, 180, 0, 256])
    cv2.normalize(h, h, alpha=1.0, norm_type=cv2.NORM_L1)
    return h


def best_frame(cap, fps, scene, n_samples):
    length = scene["end_s"] - scene["start_s"]
    lo, hi = scene["start_s"] + 0.1 * length, scene["end_s"] - 0.1 * length
    times = np.linspace(lo, hi, n_samples) if length > 0.5 else [scene["start_s"] + length / 2]
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


def select_keyframes(video, samples=9, dup=0.97):
    scenes = detect_scenes_cached(video)
    info = probe(video)
    out_dir = OUT_ROOT / "keyframes" / stem(video)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    rows, hists = [], []
    for sc in tqdm(scenes, desc="key frames"):
        frame, t, score = best_frame(cap, info["fps"], sc, samples)
        if frame is None:
            continue
        path = out_dir / f"scene_{sc['index']:03d}.jpg"
        cv2.imwrite(str(path), frame)
        rows.append({"scene": sc["index"], "time_s": t, "sharpness": score, "path": str(path),
                     "start_s": sc["start_s"], "end_s": sc["end_s"]})
        hists.append(_hist(frame))
    cap.release()

    dur = np.array([r["end_s"] - r["start_s"] for r in rows])
    importance = dur / dur.max()
    mpath = cache_dir(video) / "motion.json"
    used_motion = False
    if mpath.exists():
        m = load_json(mpath)
        vals = suppress_cuts(m["times"], m["values"], scenes, m["fps"])
        scores = np.array(scene_scores(m["times"], vals, [scenes[r["scene"]] for r in rows]))
        if scores.max() > 0:
            importance = 0.6 * scores / scores.max() + 0.4 * dur / dur.max()
            used_motion = True

    order = np.argsort(importance)[::-1]
    kept = []
    for i in order:
        rows[i]["importance"] = float(importance[i])
        rows[i]["duplicate"] = any(
            cv2.compareHist(hists[i], hists[j], cv2.HISTCMP_CORREL) > dup for j in kept)
        if not rows[i]["duplicate"]:
            kept.append(i)
    return rows, used_motion


def contact_sheet(rows, top, out_path, title):
    chosen = [r for r in sorted(rows, key=lambda r: -r["importance"]) if not r["duplicate"]][:top]
    chosen.sort(key=lambda r: r["time_s"])
    cols = min(4, max(1, len(chosen)))
    nrows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(3.4 * cols, 2.6 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), chosen):
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
    return out_path, len(chosen)


def main():
    ap = argparse.ArgumentParser(description="Select key frames per scene.")
    ap.add_argument("video")
    ap.add_argument("--samples", type=int, default=9, help="candidate frames per scene")
    ap.add_argument("--top", type=int, default=12, help="frames on the contact sheet")
    ap.add_argument("--dup", type=float, default=0.97, help="histogram correlation marking duplicates")
    args = ap.parse_args()

    rows, used_motion = select_keyframes(args.video, args.samples, args.dup)
    save_json({"video": args.video, "keyframes": rows, "importance_uses_motion": used_motion},
              cache_dir(args.video) / "keyframes.json")
    label = "motion + duration" if used_motion else "duration only (run video.motion first for motion)"
    sheet, n = contact_sheet(rows, args.top, OUT_ROOT / f"{stem(args.video)}_keyframes.png",
                             f"Top {args.top} key frames by importance ({label})")
    dups = sum(r["duplicate"] for r in rows)
    print(f"{len(rows)} key frames saved to outputs/keyframes/{stem(args.video)}/ ({dups} near-duplicates)")
    print(f"importance: {label}")
    print(f"contact sheet ({n} frames): {sheet}")


if __name__ == "__main__":
    main()
