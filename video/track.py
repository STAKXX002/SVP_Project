"""Object tracking: YOLOv8n + ByteTrack, with background-subtraction centroid tracker as fallback.

Usage:
    python -m video.track data/video/input.mp4 --start 1405 --duration 5
    python -m video.track data/video/input.mp4 --scene 152
    python -m video.track data/video/input.mp4 --backend bgsub

Design:
    Tracks objects per scene with YOLOv8n and ByteTrack. The tracker state is reset at every
    scene boundary so track IDs do not carry across cuts. Tracks shorter than 1.0 second are dropped.
    Trajectory overlays are rendered onto a single static frame from the same scene rather than
    a cross-scene blend.
    The MOG2 background-subtraction centroid tracker is preserved as an offline fallback.

Outputs:
    outputs/<video>_tracked.mp4          annotated clip with bounding boxes, IDs, and trails
    outputs/<video>_trajectories.png     all trajectories rendered over a single scene frame
    outputs/cache/<video>/tracks_<backend>_<start>_<duration>.json
"""

import argparse
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from video.common import (
    OUT_ROOT,
    cache_dir,
    iter_frames,
    probe,
    require_ffmpeg,
    save_json,
    stem,
)
from video.scenes import detect_scenes_cached
from watermark import mark_frame, save_figure

TRAIL = 40  # Number of historical points drawn behind each bounding box


def cut_clip(video, start, duration, height):
    """Cut and downscale the analysis window with ffmpeg. Cached."""
    require_ffmpeg()
    clip = cache_dir(video) / f"clip_{int(start)}_{int(duration)}_{height}.mp4"
    if not clip.exists():
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-ss",
                str(start),
                "-t",
                str(duration),
                "-i",
                str(video),
                "-an",
                "-vf",
                f"scale=-2:{height}",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "26",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(clip),
            ],
            check=True,
        )
    return clip


def yolo_backend_per_scene(clip, fps, classes, conf, imgsz, cut_times=()):
    """YOLOv8n + ByteTrack tracker reset at every scene cut boundary."""
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    info = probe(clip)
    total_frames = int(round(info["duration_s"] * fps))
    cut_times_set = set(cut_times)

    # Process frame by frame or stream with tracker reset at scene cuts
    # When tracking across cuts, resetting tracker ensures IDs do not cross scenes
    stride = max(1, int(round(info["fps"] / fps)))
    stream = model.track(
        source=str(clip),
        stream=True,
        tracker="bytetrack.yaml",
        persist=True,
        classes=classes,
        conf=conf,
        imgsz=imgsz,
        vid_stride=stride,
        device="cpu",
        verbose=False,
    )

    prev_cut_idx = 0
    id_offset = 0
    max_id_seen = 0

    for i, r in enumerate(stream):
        t = i * stride / info["fps"]
        # If this frame crosses a scene boundary, reset tracker state
        for cut in cut_times:
            if abs(t - cut) < (stride / info["fps"]) and i > prev_cut_idx:
                # Reset tracker state at scene boundary
                if hasattr(model, "predictor") and hasattr(model.predictor, "trackers"):
                    model.predictor.trackers = None
                id_offset += max_id_seen + 1
                max_id_seen = 0
                prev_cut_idx = i

        dets = []
        b = r.boxes
        if b is not None and b.id is not None:
            raw_ids = b.id.int().cpu().tolist()
            xyxy = b.xyxy.cpu().numpy()
            cf = b.conf.cpu().numpy()
            for k, tid in enumerate(raw_ids):
                max_id_seen = max(max_id_seen, tid)
                effective_id = tid + id_offset
                dets.append((effective_id, *[float(v) for v in xyxy[k]], float(cf[k])))
        yield t, r.orig_img, dets


def yolo_available():
    """Verify Ultralytics YOLO availability and weights."""
    try:
        from ultralytics import YOLO

        YOLO("yolov8n.pt")
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


class CentroidTracker:
    """Hungarian centroid matching tracker with a missed-frame budget (fallback)."""

    def __init__(self, max_dist, max_missed=5):
        self.max_dist = max_dist
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks = {}

    def update(self, boxes):
        cents = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in boxes]
        ids = list(self.tracks)
        assigned = {}
        if ids and cents:
            cost = np.array(
                [
                    [
                        np.hypot(
                            self.tracks[i]["c"][0] - c[0],
                            self.tracks[i]["c"][1] - c[1],
                        )
                        for c in cents
                    ]
                    for i in ids
                ]
            )
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] <= self.max_dist:
                    assigned[c] = ids[r]
        used = set(assigned.values())
        for i in ids:
            if i not in used:
                self.tracks[i]["missed"] += 1
        for c, i in assigned.items():
            self.tracks[i] = {"c": cents[c], "missed": 0}
        for c in range(len(cents)):
            if c not in assigned:
                assigned[c] = self.next_id
                self.tracks[self.next_id] = {"c": cents[c], "missed": 0}
                self.next_id += 1
        for i in [i for i, t in self.tracks.items() if t["missed"] > self.max_missed]:
            del self.tracks[i]
        return [assigned[c] for c in range(len(cents))]


def _new_subtractor():
    return cv2.createBackgroundSubtractorMOG2(
        history=150, varThreshold=32, detectShadows=True
    )


def bgsub_backend(clip, fps, cuts=(), min_area_frac=0.002, warmup=10):
    """MOG2 background subtraction plus centroid tracker fallback."""
    info = probe(clip)
    sub = _new_subtractor()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tracker = CentroidTracker(max_dist=0.15 * info["width"])
    min_area = min_area_frac * info["width"] * info["height"]
    cuts = sorted(cuts)
    ci, since = 0, 0
    for t, frame in iter_frames(clip, fps, width=None):
        while ci < len(cuts) and t >= cuts[ci] - 1e-6:
            sub = _new_subtractor()
            tracker.tracks = {}
            since = 0
            ci += 1
        mask = sub.apply(frame)
        dets = []
        if since >= warmup:
            mask = (mask == 255).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.dilate(mask, kernel, iterations=2)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            boxes = []
            for c in contours:
                if cv2.contourArea(c) >= min_area:
                    x, y, w, h = cv2.boundingRect(c)
                    boxes.append((x, y, x + w, y + h))
            for tid, b in zip(tracker.update(boxes), boxes):
                dets.append((tid, *map(float, b), 1.0))
        since += 1
        yield t, frame, dets


def color_for(tid):
    rng = np.random.default_rng(tid * 7919)
    return tuple(int(v) for v in rng.integers(60, 255, 3))


def annotate(frame, dets, trails):
    out = frame.copy()
    for tid, x1, y1, x2, y2, _ in dets:
        col = color_for(tid)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        trails[tid].append((cx, cy))
        pts = np.array(trails[tid][-TRAIL:], np.int32).reshape(-1, 1, 2)
        if len(pts) > 1:
            cv2.polylines(out, [pts], False, col, 2)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        cv2.putText(
            out,
            f"ID {tid}",
            (int(x1), max(15, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            col,
            2,
        )
    return mark_frame(out)


def run_tracking(
    video,
    start,
    duration,
    fps,
    backend,
    classes,
    conf,
    imgsz,
    height,
    min_len=5,
):
    clip = cut_clip(video, start, duration, height)
    info = probe(clip)
    cuts = [
        sc["start_s"] - start
        for sc in detect_scenes_cached(video)
        if start < sc["start_s"] < start + duration
    ]
    print(f"window {start:.1f}s to {start + duration:.1f}s contains {len(cuts)} scene cuts")

    if backend == "auto":
        ok, reason = yolo_available()
        backend = "yolo" if ok else "bgsub"
        if not ok:
            print(f"YOLO backend unavailable ({reason}); falling back to bgsub.")

    source = (
        yolo_backend_per_scene(clip, fps, classes, conf, imgsz, cut_times=cuts)
        if backend == "yolo"
        else bgsub_backend(clip, fps, cuts)
    )

    raw_path = cache_dir(video) / "tracked_raw.mp4"
    writer = cv2.VideoWriter(
        str(raw_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (info["width"], info["height"]),
    )
    trails = defaultdict(list)
    tracks = defaultdict(lambda: {"t": [], "box": []})
    first_frame = None

    total = int(round(duration * fps))
    for k, (t, frame, dets) in enumerate(
        tqdm(source, total=total, desc=f"tracking ({backend})")
    ):
        if first_frame is None:
            # Single representative frame from the tracked scene, never a multi-scene blend
            first_frame = frame.copy()
        for tid, x1, y1, x2, y2, _ in dets:
            tracks[tid]["t"].append(start + t)
            tracks[tid]["box"].append([x1, y1, x2, y2])
        writer.write(annotate(frame, dets, trails))
    writer.release()

    # Drop tracks shorter than 1.0 second (min_len frames)
    min_frames = max(min_len, int(round(fps * 1.0)))
    filtered_tracks = {
        tid: v
        for tid, v in tracks.items()
        if len(v["t"]) >= min_frames and (v["t"][-1] - v["t"][0]) >= 0.8
    }

    return (
        backend,
        filtered_tracks,
        raw_path,
        first_frame,
        (info["width"], info["height"]),
    )


def track_stats(tracks):
    rows = []
    for tid, v in tracks.items():
        b = np.array(v["box"])
        c = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], axis=1)
        dist = float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum()) if len(c) > 1 else 0.0
        rows.append(
            {
                "id": int(tid),
                "first_s": v["t"][0],
                "last_s": v["t"][-1],
                "duration_s": v["t"][-1] - v["t"][0],
                "observations": len(v["t"]),
                "path_px": dist,
            }
        )
    rows.sort(key=lambda r: -(r["last_s"] - r["first_s"]))
    return rows


def plot_trajectories(tracks, background, size, out_path, title):
    fig, ax = plt.subplots(figsize=(10, 10 * size[1] / size[0]))
    if background is not None:
        ax.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    for tid, v in tracks.items():
        b = np.array(v["box"])
        cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
        col = np.array(color_for(tid)[::-1]) / 255.0
        ax.plot(cx, cy, "-", color=col, linewidth=2)
        ax.plot(cx[0], cy[0], "o", color=col, markersize=6)
        ax.text(
            cx[0],
            cy[0],
            f" ID {tid}",
            color="white",
            fontsize=9,
            fontweight="bold",
            bbox=dict(facecolor=col, alpha=0.7, pad=1, edgecolor="none"),
        )
    ax.set_xlim(0, size[0])
    ax.set_ylim(size[1], 0)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, dpi=130)
    plt.close(fig)
    return out_path


def finalize_video(raw_path, out_path):
    """Encode annotated OpenCV mp4 to H.264 yuv420p with faststart for web and browser playback."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(raw_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-crf",
            "23",
            str(out_path),
        ],
        check=True,
    )
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Track objects in a window of the video.")
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=1405.0, help="window start (s)")
    ap.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="window length (s); default 5.0 s (boy walking demo clip)",
    )
    ap.add_argument(
        "--scene",
        type=int,
        default=None,
        help="track inside this scene index",
    )
    ap.add_argument("--fps", type=float, default=5.0, help="analysis rate")
    ap.add_argument(
        "--backend",
        choices=["auto", "yolo", "bgsub"],
        default="yolo",
    )
    ap.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=[0],
        help="COCO class ids (0 = person)",
    )
    ap.add_argument("--conf", type=float, default=0.20, help="detection confidence threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="inference image size")
    ap.add_argument("--height", type=int, default=480, help="analysis resolution height")
    ap.add_argument(
        "--min-len",
        type=int,
        default=5,
        help="drop tracks seen fewer frames than this (default 5 frames / 1 s)",
    )
    args = ap.parse_args()

    info = probe(args.video)
    if args.scene is not None:
        scenes = detect_scenes_cached(args.video)
        if not 0 <= args.scene < len(scenes):
            raise SystemExit(f"--scene must be 0 to {len(scenes) - 1}")
        sc = scenes[args.scene]
        args.start = sc["start_s"]
        length = sc["end_s"] - sc["start_s"]
        args.duration = min(args.duration, length) if args.duration else length

    duration = min(args.duration or 5.0, max(1.0, info["duration_s"] - args.start))

    backend, tracks, raw, bg_frame, size = run_tracking(
        args.video,
        args.start,
        duration,
        args.fps,
        args.backend,
        args.classes,
        args.conf,
        args.imgsz,
        args.height,
        args.min_len,
    )

    name = stem(args.video)
    stats = track_stats(tracks)
    save_json(
        {
            "backend": backend,
            "start_s": args.start,
            "duration_s": duration,
            "fps": args.fps,
            "stats": stats,
            "tracks": {str(k): v for k, v in tracks.items()},
        },
        cache_dir(args.video)
        / f"tracks_{backend}_{int(args.start)}_{int(duration)}.json",
    )

    vid = finalize_video(raw, OUT_ROOT / f"{name}_tracked.mp4")
    fig = plot_trajectories(
        tracks,
        bg_frame,
        size,
        OUT_ROOT / f"{name}_trajectories.png",
        f"{len(tracks)} tracks ({backend}), {args.start:.1f} s to {args.start + duration:.1f} s",
    )

    print(f"backend: {backend}   tracks kept (>= 1.0 s): {len(tracks)}")
    for r in stats:
        print(
            f"  ID {r['id']:3d}  {r['first_s']:7.1f}s to {r['last_s']:7.1f}s  "
            f"({r['duration_s']:.1f} s, {r['observations']} obs, path {r['path_px']:.0f} px)"
        )
    print(f"annotated video: {vid}")
    print(f"trajectories:    {fig}")


if __name__ == "__main__":
    main()
