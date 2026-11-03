"""Dense optical-flow motion analysis: accumulated motion heatmap and motion-over-time curve.

Usage:
    python -m video.motion data/video/input.mp4
    python -m video.motion data/video/input.mp4 --fps 5 --width 320 --no-compensate

Method:
    Farneback optical flow between consecutive sampled frames (default 5 fps, 320 px wide).
    Global motion compensation subtracts the median flow vector of each frame pair by default
    (--no-compensate to disable).
    Accumulated heatmap is normalized with the 2nd to 98th percentile and overlaid
    using the inferno colormap (alpha 0.6) onto the mean/background frame.
    Letterbox borders are automatically cropped.
Outputs:
    outputs/<video>_motion_map.png              heatmap over background and motion curve
    outputs/<video>_motion_scene_XXX.png        per-scene motion maps for top 6 scenes
    outputs/cache/<video>/motion.json           per-sample motion values
    outputs/cache/<video>/heat.npy              accumulated heatmap
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
    save_json,
    stem,
)
from watermark import save_figure

FARNEBACK = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)


def crop_letterbox(img, thresh=15):
    """Detect and return crop bounds (y1, y2, x1, x2) for black letterbox borders."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    row_means = gray.mean(axis=1)
    col_means = gray.mean(axis=0)
    rows = np.where(row_means > thresh)[0]
    cols = np.where(col_means > thresh)[0]
    if len(rows) > 0 and len(cols) > 0:
        y1, y2 = int(rows[0]), int(rows[-1] + 1)
        x1, x2 = int(cols[0]), int(cols[-1] + 1)
        if y1 > 1 or y2 < img.shape[0] - 1 or x1 > 1 or x2 < img.shape[1] - 1:
            return y1, y2, x1, x2
    return 0, img.shape[0], 0, img.shape[1]


def compute_motion(
    video,
    fps=5.0,
    width=320,
    start_s=0.0,
    end_s=None,
    compensate=True,
):
    """Compute dense optical flow, global motion compensation, and accumulated motion map."""
    info = probe(video)
    end = info["duration_s"] if end_s is None else min(end_s, info["duration_s"])
    total = int((end - start_s) * fps)

    prev_gray = None
    heat = None
    times, values = [], []
    bg_samples = []

    for k, (t, frame) in enumerate(
        tqdm(
            iter_frames(video, fps, start_s, end, width),
            total=total,
            desc="optical flow",
        )
    ):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if k % max(1, int(fps * 20)) == 0:
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
        raise SystemExit("No frames read. Check video path and OpenCV support.")

    background = (
        np.median(np.stack(bg_samples), axis=0).astype(np.uint8)
        if bg_samples
        else np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    )
    return {
        "times": times,
        "values": values,
        "heat": heat / max(len(times), 1),
        "background": background,
        "fps": fps,
        "compensate": compensate,
    }


def suppress_cuts(times, values, scenes, fps):
    """Replace optical flow samples straddling scene cuts with neighbouring minimum."""
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
    """Compute mean motion magnitude within each scene."""
    t = np.asarray(times)
    v = np.asarray(values)
    scores = []
    for sc in scenes:
        m = (t >= sc["start_s"]) & (t < sc["end_s"])
        scores.append(float(v[m].mean()) if m.any() else 0.0)
    return scores


def plot_motion(result, scenes, out_path, title=""):
    """Plot accumulated motion map (inferno, alpha 0.6) and motion-over-time with thinned ticks."""
    heat = result["heat"]
    bg = result["background"]

    # Crop letterbox borders
    y1, y2, x1, x2 = crop_letterbox(bg)
    bg = bg[y1:y2, x1:x2]
    heat = heat[y1:y2, x1:x2]

    # Normalise accumulated map with the 2nd to 98th percentile
    p2 = float(np.percentile(heat, 2.0))
    p98 = float(np.percentile(heat, 98.0))
    norm = np.clip((heat - p2) / max(p98 - p2, 1e-9), 0.0, 1.0)

    # Inferno colormap
    cmap = plt.get_cmap("inferno")
    heat_rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    heat_bgr = cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR)

    if heat_bgr.shape[:2] != bg.shape[:2]:
        heat_bgr = cv2.resize(heat_bgr, (bg.shape[1], bg.shape[0]))

    # Overlay with alpha 0.6 on mean/background frame
    overlay = cv2.addWeighted(bg, 0.4, heat_bgr, 0.6, 0)

    fig = plt.figure(figsize=(13, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.4])
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax1.set_title("Accumulated motion map (inferno, alpha 0.6)")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[1])
    t = np.asarray(result["times"])
    v = np.asarray(result["values"])
    ax2.plot(t / 60.0, v, linewidth=0.8, color="tab:blue")
    y_max = float(v.max()) * 1.08 if len(v) else 1.0
    ax2.set_ylim(bottom=0, top=y_max)

    # Thin scene-start lines: ticks at the top instead of full-height vertical lines
    if scenes:
        starts = np.array([sc["start_s"] / 60.0 for sc in scenes])
        ax2.plot(
            starts,
            np.full_like(starts, y_max * 0.98),
            "|",
            color="dimgray",
            markersize=6,
            alpha=0.75,
            label="scene starts",
        )
        ax2.legend(loc="upper right", fontsize=8)

    ax2.set_xlabel("time (min)")
    ax2.set_ylabel("mean flow magnitude (px per sample)")
    ax2.set_title("Motion over time (ticks at top mark scene starts)")
    fig.suptitle(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_scene_motion_map(video, scene, out_path, fps=5.0, width=320):
    """Compute and write motion map for a single scene."""
    s, e = scene["start_s"], scene["end_s"]
    prev_gray = None
    heat = None
    frames = []

    for _, frame in iter_frames(video, fps, start_s=s, end_s=e, width=width):
        frames.append(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None and prev_gray.shape == gray.shape:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **FARNEBACK)
            flow = flow - np.median(flow.reshape(-1, 2), axis=0)
            mag = np.linalg.norm(flow, axis=2)
            if heat is None:
                heat = np.zeros_like(mag, dtype=np.float64)
            heat += mag
        prev_gray = gray

    if heat is None or len(frames) == 0:
        return None

    bg = np.mean(np.stack(frames), axis=0).astype(np.uint8)
    y1, y2, x1, x2 = crop_letterbox(bg)
    bg = bg[y1:y2, x1:x2]
    heat = heat[y1:y2, x1:x2]

    p2 = float(np.percentile(heat, 2.0))
    p98 = float(np.percentile(heat, 98.0))
    norm = np.clip((heat - p2) / max(p98 - p2, 1e-9), 0.0, 1.0)

    cmap = plt.get_cmap("inferno")
    heat_rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    heat_bgr = cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR)
    if heat_bgr.shape[:2] != bg.shape[:2]:
        heat_bgr = cv2.resize(heat_bgr, (bg.shape[1], bg.shape[0]))

    overlay = cv2.addWeighted(bg, 0.4, heat_bgr, 0.6, 0)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Scene {scene['index']} motion map ({s:.1f}s to {e:.1f}s)")
    ax.axis("off")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, dpi=130)
    plt.close(fig)
    return out_path


def motion_cached(video, force=False, **kwargs):
    d = cache_dir(video)
    jpath, hpath = d / "motion.json", d / "heat.npy"
    if (
        jpath.exists()
        and hpath.exists()
        and (d / "motion_bg.png").exists()
        and not force
    ):
        data = load_json(jpath)
        return {
            "times": data["times"],
            "values": data["values"],
            "heat": np.load(hpath),
            "background": cv2.imread(str(d / "motion_bg.png")),
            "fps": data["fps"],
            "compensate": data["compensate"],
        }
    res = compute_motion(video, **kwargs)
    save_json(
        {
            "times": res["times"],
            "values": res["values"],
            "fps": res["fps"],
            "compensate": res["compensate"],
        },
        jpath,
    )
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
    ap.add_argument(
        "--no-compensate",
        action="store_true",
        help="disable global motion compensation",
    )
    ap.add_argument("--force", action="store_true", help="ignore cached result")
    args = ap.parse_args()

    res = motion_cached(
        args.video,
        force=args.force,
        fps=args.fps,
        width=args.width,
        start_s=args.start,
        end_s=args.end,
        compensate=not args.no_compensate,
    )
    scenes_path = cache_dir(args.video) / "scenes.json"
    scenes = load_json(scenes_path)["scenes"] if scenes_path.exists() else []

    if scenes:
        res["values"] = suppress_cuts(
            res["times"], res["values"], scenes, res["fps"]
        ).tolist()

    fig = plot_motion(
        res,
        scenes,
        OUT_ROOT / f"{stem(args.video)}_motion_map.png",
        stem(args.video),
    )
    v = np.asarray(res["values"])
    print(
        f"samples: {len(v)}   mean motion {v.mean():.3f}   peak {v.max():.3f} at "
        f"{res['times'][int(v.argmax())]:.1f} s"
    )
    print(f"figure: {fig}")

    # Write per-scene motion maps for the top 6 scenes
    if scenes:
        scores = scene_scores(res["times"], res["values"], scenes)
        top6_indices = np.argsort(scores)[::-1][:6]
        print("Writing per-scene motion maps for top 6 active scenes:")
        for rank, s_idx in enumerate(top6_indices):
            sc = scenes[s_idx]
            out_sc_path = OUT_ROOT / f"{stem(args.video)}_motion_scene_{sc['index']:03d}.png"
            p = write_scene_motion_map(args.video, sc, out_sc_path, fps=args.fps, width=args.width)
            print(
                f"  Rank {rank+1}: scene {sc['index']:3d} ({sc['start_s']:.1f}s to {sc['end_s']:.1f}s, "
                f"motion {scores[s_idx]:.3f}) -> {p}"
            )


if __name__ == "__main__":
    main()
