"""Object tracking: YOLOv8n + ByteTrack, with a background-subtraction centroid tracker as fallback.

Usage:
    python -m video.track data/video/input.mp4 --start 600 --duration 120
    python -m video.track data/video/input.mp4 --backend bgsub
    python -m video.track data/video/input.mp4 --classes 0 2 --fps 5

Tracking the full 30 minutes on a CPU is slow, so the default is a 120 s window starting at --start.
A clip of that window is first cut and downscaled to --height pixels with ffmpeg and cached.
Time per sampled frame on a laptop CPU is roughly 60 to 120 ms for YOLOv8n at imgsz 640, so
120 s at 5 fps (600 frames) takes about one to two minutes.

Backends:
    yolo   Ultralytics YOLOv8n detector with the built-in ByteTrack tracker. Needs the CPU build of
           torch and the yolov8n.pt weights, which are downloaded on first use (needs internet once).
           Run it once before the demo so the weights are cached.
    bgsub  OpenCV MOG2 background subtraction + Hungarian centroid matching. Needs a mostly static
           camera. Tracks any moving blob, not only people.
    auto   yolo if it loads, otherwise bgsub.

Outputs:
    outputs/<video>_tracked.mp4          annotated clip with boxes, IDs and trails
    outputs/<video>_trajectories.png     all trajectories over a background frame
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

from video.common import OUT_ROOT, cache_dir, iter_frames, probe, require_ffmpeg, save_json, stem

TRAIL = 40  # samples of trail drawn behind each box


def cut_clip(video, start, duration, height):
    """Cut and downscale the analysis window with ffmpeg. Cached."""
    require_ffmpeg()
    clip = cache_dir(video) / f"clip_{int(start)}_{int(duration)}_{height}.mp4"
    if not clip.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-t", str(duration),
             "-i", str(video), "-an", "-vf", f"scale=-2:{height}", "-c:v", "libx264",
             "-preset", "ultrafast", "-crf", "26", "-pix_fmt", "yuv420p", str(clip)], check=True)
    return clip


# ---------------------------------------------------------------- YOLO + ByteTrack backend
def yolo_backend(clip, fps, classes, conf, imgsz):
    from ultralytics import YOLO  # imported lazily so bgsub works without torch

    model = YOLO("yolov8n.pt")
    info = probe(clip)
    stride = max(1, int(round(info["fps"] / fps)))
    stream = model.track(source=str(clip), stream=True, tracker="bytetrack.yaml", persist=True,
                         classes=classes, conf=conf, imgsz=imgsz, vid_stride=stride,
                         device="cpu", verbose=False)
    for i, r in enumerate(stream):
        dets = []
        b = r.boxes
        if b is not None and b.id is not None:
            ids = b.id.int().cpu().tolist()
            xyxy = b.xyxy.cpu().numpy()
            cf = b.conf.cpu().numpy()
            for k, tid in enumerate(ids):
                dets.append((tid, *[float(v) for v in xyxy[k]], float(cf[k])))
        yield i * stride / info["fps"], r.orig_img, dets


def yolo_available():
    try:
        from ultralytics import YOLO

        YOLO("yolov8n.pt")
        return True, ""
    except Exception as exc:  # missing torch, no weights, no network, incompatible Python
        return False, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------- background subtraction backend
class CentroidTracker:
    """Greedy-free (Hungarian) nearest-centroid tracker with a missed-frame budget."""

    def __init__(self, max_dist, max_missed=5):
        self.max_dist = max_dist
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks = {}  # id -> {"c": centroid, "missed": n}

    def update(self, boxes):
        cents = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in boxes]
        ids = list(self.tracks)
        assigned = {}
        if ids and cents:
            cost = np.array([[np.hypot(self.tracks[i]["c"][0] - c[0], self.tracks[i]["c"][1] - c[1])
                              for c in cents] for i in ids])
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


def bgsub_backend(clip, fps, min_area_frac=0.002, warmup=10):
    info = probe(clip)
    sub = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=32, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tracker = CentroidTracker(max_dist=0.15 * info["width"])
    min_area = min_area_frac * info["width"] * info["height"]
    for k, (t, frame) in enumerate(iter_frames(clip, fps, width=None)):
        mask = sub.apply(frame)
        dets = []
        if k >= warmup:
            mask = (mask == 255).astype(np.uint8) * 255  # drop the grey shadow label (127)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.dilate(mask, kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = []
            for c in contours:
                if cv2.contourArea(c) >= min_area:
                    x, y, w, h = cv2.boundingRect(c)
                    boxes.append((x, y, x + w, y + h))
            for tid, b in zip(tracker.update(boxes), boxes):
                dets.append((tid, *map(float, b), 1.0))
        yield t, frame, dets


# ---------------------------------------------------------------- shared output code
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
        cv2.putText(out, f"ID {tid}", (int(x1), max(15, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, col, 2)
    return out


def run_tracking(video, start, duration, fps, backend, classes, conf, imgsz, height, min_len):
    clip = cut_clip(video, start, duration, height)
    info = probe(clip)

    if backend == "auto":
        ok, reason = yolo_available()
        backend = "yolo" if ok else "bgsub"
        if not ok:
            print(f"YOLO backend unavailable ({reason}); using background subtraction.")
    source = (yolo_backend(clip, fps, classes, conf, imgsz) if backend == "yolo"
              else bgsub_backend(clip, fps))

    raw_path = cache_dir(video) / "tracked_raw.mp4"
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (info["width"], info["height"]))
    trails = defaultdict(list)
    tracks = defaultdict(lambda: {"t": [], "box": []})
    bg_frames = []
    total = int(duration * fps)
    for k, (t, frame, dets) in enumerate(tqdm(source, total=total, desc=f"tracking ({backend})")):
        if k % max(1, int(fps * 5)) == 0:
            bg_frames.append(frame)
        for tid, x1, y1, x2, y2, _ in dets:
            tracks[tid]["t"].append(start + t)
            tracks[tid]["box"].append([x1, y1, x2, y2])
        writer.write(annotate(frame, dets, trails))
    writer.release()

    tracks = {tid: v for tid, v in tracks.items() if len(v["t"]) >= min_len}
    background = np.median(np.stack(bg_frames), axis=0).astype(np.uint8) if bg_frames else None
    return backend, tracks, raw_path, background, (info["width"], info["height"])


def track_stats(tracks):
    rows = []
    for tid, v in tracks.items():
        b = np.array(v["box"])
        c = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], axis=1)
        dist = float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum()) if len(c) > 1 else 0.0
        rows.append({"id": int(tid), "first_s": v["t"][0], "last_s": v["t"][-1],
                     "observations": len(v["t"]), "path_px": dist})
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
        ax.text(cx[0], cy[0], f" {tid}", color="white", fontsize=9,
                bbox=dict(facecolor=col, alpha=0.7, pad=1, edgecolor="none"))
    ax.set_xlim(0, size[0])
    ax.set_ylim(size[1], 0)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def finalize_video(raw_path, out_path):
    """Re-encode the OpenCV mp4v file to H.264 so it plays in browsers and slides."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_path), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "23", str(out_path)], check=True)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Track objects in a window of the video.")
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0.0, help="window start (s)")
    ap.add_argument("--duration", type=float, default=120.0, help="window length (s)")
    ap.add_argument("--fps", type=float, default=5.0, help="analysis rate; higher is slower")
    ap.add_argument("--backend", choices=["auto", "yolo", "bgsub"], default="auto")
    ap.add_argument("--classes", type=int, nargs="+", default=[0], help="COCO class ids (0 = person)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--height", type=int, default=480, help="clip height used for analysis")
    ap.add_argument("--min-len", type=int, default=3, help="drop tracks seen fewer times than this")
    args = ap.parse_args()

    info = probe(args.video)
    duration = min(args.duration, max(1.0, info["duration_s"] - args.start))
    backend, tracks, raw, bg, size = run_tracking(
        args.video, args.start, duration, args.fps, args.backend, args.classes,
        args.conf, args.imgsz, args.height, args.min_len)

    name = stem(args.video)
    stats = track_stats(tracks)
    save_json({"backend": backend, "start_s": args.start, "duration_s": duration, "fps": args.fps,
               "stats": stats, "tracks": {str(k): v for k, v in tracks.items()}},
              cache_dir(args.video) / f"tracks_{backend}_{int(args.start)}_{int(duration)}.json")
    vid = finalize_video(raw, OUT_ROOT / f"{name}_tracked.mp4")
    fig = plot_trajectories(tracks, bg, size, OUT_ROOT / f"{name}_trajectories.png",
                            f"{len(tracks)} tracks ({backend}), {args.start:.0f} s to {args.start + duration:.0f} s")
    print(f"backend: {backend}   tracks kept: {len(tracks)}")
    for r in stats[:5]:
        print(f"  ID {r['id']:3d}  {r['first_s']:7.1f}s to {r['last_s']:7.1f}s  "
              f"{r['observations']} obs, path {r['path_px']:.0f} px")
    print(f"annotated video: {vid}")
    print(f"trajectories:    {fig}")
    if backend == "bgsub":
        print("NOTE: bgsub follows any moving blob and needs a static camera. Expect merged or split blobs"
              " when people overlap.")


if __name__ == "__main__":
    main()
