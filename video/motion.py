"""Dense optical-flow motion analysis: accumulated motion heatmap and motion-over-time curve.

Usage:
    python -m video.motion data/video/input.mp4
    python -m video.motion data/video/input.mp4 --fps 5 --width 320 --no-compensate

Method:
    Farneback optical flow between consecutive sampled frames (default 5 fps, 320 px wide).
    With --compensate (default) the median flow vector of each frame pair is subtracted first,
    which removes most of the camera pan and shake so that the map shows moving objects rather than
    camera movement. It cannot remove zoom or strong parallax.
Outputs:
    outputs/<video>_motion_map.png      heatmap over a background estimate, and motion vs time
    outputs/cache/<video>/motion.json   per-sample motion values (used by keyframes and summarize)
    outputs/cache/<video>/heat.npy      accumulated heatmap
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

FARNEBACK = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


def compute_motion(video, fps=5.0, width=320, start_s=0.0, end_s=None, compensate=True):
    info = probe(video)
    end = info["duration_s"] if end_s is None else min(end_s, info["duration_s"])
    total = int((end - start_s) * fps)

    prev_gray = None
    heat = None
    times, values = [], []
    bg_samples = []
    for k, (t, frame) in enumerate(tqdm(iter_frames(video, fps, start_s, end, width),
                                        total=total, desc="optical flow")):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if k % max(1, int(fps * 20)) == 0:  # a background candidate every ~20 s
            bg_samples.append(frame)
        if prev_gray is not None and prev_gray.shape == gray.shape:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **FARNEBACK)
            if compensate:
                flow = flow - np.median(flow.reshape(-1, 2), axis=0)
            mag = np.linalg.norm(flow, axis=2)
            if heat is None:
                heat = np.zeros_like(mag, dtype=np.float64)
            heat += mag
            times.append(float(t))
            values.append(float(mag.mean()))
        prev_gray = gray
    if heat is None:
        raise SystemExit("No frames were read. Check the video path and codec support in OpenCV.")
    background = np.median(np.stack(bg_samples), axis=0).astype(np.uint8)
    return {"times": times, "values": values, "heat": heat / max(len(times), 1),
            "background": background, "fps": fps, "compensate": compensate}


def suppress_cuts(times, values, scenes, fps):
    """Replace the flow sample that straddles a scene cut, which is a spurious spike.

    Optical flow between the last frame of one shot and the first frame of the next measures
    the cut, not motion. Those samples are overwritten with the smaller neighbouring value.
    """
    t = np.asarray(times)
    v = np.asarray(values, dtype=float).copy()
    step = 1.0 / fps
    for sc in scenes[1:]:
        idx = np.where((t >= sc["start_s"] - 1e-6) & (t < sc["start_s"] + 1.2 * step))[0]
        for i in idx:
            left = v[i - 1] if i > 0 else v[min(i + 1, len(v) - 1)]
            right = v[i + 1] if i + 1 < len(v) else left
            v[i] = min(left, right)
    return v


def scene_scores(times, values, scenes):
    """Mean motion inside each scene, from the motion curve."""
    t = np.asarray(times)
    v = np.asarray(values)
    scores = []
    for sc in scenes:
        m = (t >= sc["start_s"]) & (t < sc["end_s"])
        scores.append(float(v[m].mean()) if m.any() else 0.0)
    return scores


def plot_motion(result, scenes, out_path, title=""):
    heat = result["heat"]
    norm = np.clip(heat / (np.percentile(heat, 99.5) + 1e-9), 0, 1) ** 0.6
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    bg = result["background"]
    color = cv2.resize(color, (bg.shape[1], bg.shape[0]))
    overlay = cv2.addWeighted(bg, 0.45, color, 0.55, 0)

    fig = plt.figure(figsize=(13, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.4])
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax1.set_title("Accumulated motion map")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[1])
    t = np.asarray(result["times"])
    v = np.asarray(result["values"])
    ax2.plot(t / 60.0, v, linewidth=0.7, color="tab:blue")
    ax2.set_ylim(bottom=0)
    if scenes:
        for sc in scenes:
            ax2.axvline(sc["start_s"] / 60.0, color="gray", linewidth=0.4, alpha=0.6)
    ax2.set_xlabel("time (min)")
    ax2.set_ylabel("mean flow magnitude (px per sample)")
    ax2.set_title("Motion over time (vertical lines = scene starts)")
    fig.suptitle(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def motion_cached(video, force=False, **kwargs):
    d = cache_dir(video)
    jpath, hpath = d / "motion.json", d / "heat.npy"
    if jpath.exists() and hpath.exists() and (d / "motion_bg.png").exists() and not force:
        data = load_json(jpath)
        return {"times": data["times"], "values": data["values"], "heat": np.load(hpath),
                "background": cv2.imread(str(d / "motion_bg.png")), "fps": data["fps"],
                "compensate": data["compensate"]}
    res = compute_motion(video, **kwargs)
    save_json({"times": res["times"], "values": res["values"], "fps": res["fps"],
               "compensate": res["compensate"]}, jpath)
    np.save(hpath, res["heat"])
    cv2.imwrite(str(d / "motion_bg.png"), res["background"])
    return res


def main():
    ap = argparse.ArgumentParser(description="Optical-flow motion map.")
    ap.add_argument("video")
    ap.add_argument("--fps", type=float, default=5.0, help="analysis sampling rate")
    ap.add_argument("--width", type=int, default=320, help="analysis width in pixels")
    ap.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    ap.add_argument("--end", type=float, default=None, help="end time in seconds")
    ap.add_argument("--no-compensate", action="store_true", help="keep camera motion in the map")
    ap.add_argument("--force", action="store_true", help="ignore cached result")
    args = ap.parse_args()

    res = motion_cached(args.video, force=args.force, fps=args.fps, width=args.width,
                        start_s=args.start, end_s=args.end, compensate=not args.no_compensate)
    scenes_path = cache_dir(args.video) / "scenes.json"
    scenes = load_json(scenes_path)["scenes"] if scenes_path.exists() else []
    if scenes:
        res["values"] = suppress_cuts(res["times"], res["values"], scenes, res["fps"]).tolist()
    fig = plot_motion(res, scenes, OUT_ROOT / f"{stem(args.video)}_motion_map.png", stem(args.video))
    v = np.asarray(res["values"])
    print(f"samples: {len(v)}   mean motion {v.mean():.3f}   peak {v.max():.3f} at "
          f"{res['times'][int(v.argmax())]:.1f} s")
    if scenes:
        scores = scene_scores(res["times"], res["values"], scenes)
        top = np.argsort(scores)[::-1][:5]
        print("most active scenes: " + ", ".join(f"#{i} ({scores[i]:.3f})" for i in top))
    print(f"figure: {fig}")


if __name__ == "__main__":
    main()
