"""Shared helpers for the video pipeline: paths, caching, probing and frame iteration."""

import json
import shutil
import subprocess
from pathlib import Path

import cv2

OUT_ROOT = Path("outputs")
CACHE_ROOT = OUT_ROOT / "cache"


def stem(video):
    return Path(video).stem


def cache_dir(video):
    d = CACHE_ROOT / stem(video)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))
    return path


def load_json(path):
    return json.loads(Path(path).read_text())


def probe(video):
    """Return fps, frame_count, width, height, duration_s using OpenCV."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": float(fps), "frames": n, "width": w, "height": h,
            "duration_s": n / fps if fps else 0.0}


def resize_to_width(frame, width):
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    return cv2.resize(frame, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


def iter_frames(video, target_fps=5.0, start_s=0.0, end_s=None, width=320):
    """Yield (time_s, frame) at roughly target_fps, resized to the given width.

    Uses grab() to skip frames cheaply, so only sampled frames are decoded to arrays
    (the codec still has to decode every frame, which is the real cost on long videos).
    """
    info = probe(video)
    fps = info["fps"]
    step = max(1, int(round(fps / target_fps)))
    cap = cv2.VideoCapture(str(video))
    if start_s > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
    idx0 = int(round(start_s * fps))
    i = 0
    while True:
        if not cap.grab():
            break
        t = (idx0 + i) / fps
        if end_s is not None and t > end_s:
            break
        if i % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            yield t, resize_to_width(frame, width) if width else frame
        i += 1
    cap.release()


def read_frame_at(video, t_s, width=None):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return resize_to_width(frame, width) if width else frame


def require_ffmpeg():
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} not found on PATH. Install ffmpeg first.")


def has_audio(video):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=codec_type", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True)
    return "audio" in out.stdout


def media_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def require_scenes(video, **kwargs):
    """Load cached scenes, computing them first if needed."""
    from video.scenes import detect_scenes_cached
    return detect_scenes_cached(video, **kwargs)
