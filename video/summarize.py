"""Short video summary built from the highest-motion scenes, stitched with ffmpeg.

Usage:
    python -m video.summarize data/video/input.mp4
    python -m video.summarize data/video/input.mp4 --target 75 --clip 4.5

Selection:
    1. Ranks scenes by mean optical flow motion score.
    2. Takes one 3 to 5 second excerpt from each highest-motion scene until
       the 60 to 90 second target is reached.
    3. Orders selected clips chronologically.
    4. Cuts and concatenates excerpts with ffmpeg, encoding to H.264 yuv420p faststart.

Outputs:
    outputs/<video>_summary.mp4
    outputs/cache/<video>/summary.json
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from video.common import (
    OUT_ROOT,
    cache_dir,
    has_audio,
    media_duration,
    probe,
    require_ffmpeg,
    save_json,
    stem,
)
from video.motion import motion_cached, scene_scores, suppress_cuts
from video.scenes import detect_scenes_cached
from watermark import ffmpeg_drawtext


def best_window_in_scene(times, values, scene, clip_len=4.5, min_len=3.0, step=0.25):
    """Find the window of duration clip_len within scene having highest mean motion."""
    s, e = scene["start_s"], scene["end_s"]
    length = e - s
    if length < min_len:
        return 0.0, s, e

    win = min(clip_len, length)
    t = np.asarray(times)
    v = np.asarray(values)

    starts = np.arange(s, e - win + 1e-4, step)
    if len(starts) == 0:
        starts = np.array([s])

    best_score = -1.0
    best_start = s

    for a in starts:
        m = (t >= a) & (t < a + win)
        score = float(v[m].mean()) if m.any() else 0.0
        if score > best_score:
            best_score = score
            best_start = float(a)

    return best_score, best_start, best_start + win


def plan(
    times,
    values,
    scenes,
    target=75.0,
    clip_len=4.5,
    min_clip=3.0,
    max_len=90.0,
):
    """Select one 3 to 5 s window per highest-motion scene until target duration is met."""
    scores = scene_scores(times, values, scenes)
    ranked_indices = np.argsort(scores)[::-1]

    picks = []
    total = 0.0

    for idx in ranked_indices:
        sc = scenes[idx]
        dur = sc["end_s"] - sc["start_s"]
        if dur < min_clip:
            continue

        take = min(clip_len, dur)
        if total + take > max_len:
            break

        win_score, a, b = best_window_in_scene(times, values, sc, clip_len=take, min_len=min_clip)
        picks.append(
            {
                "scene": sc["index"],
                "start_s": a,
                "end_s": b,
                "motion": float(win_score),
            }
        )
        total += b - a
        if total >= target:
            break

    # Order excerpts chronologically
    picks.sort(key=lambda p: p["start_s"])
    return picks, total


def cut_and_join(video, picks, out_path):
    """Cut planned segments and join into an H.264 yuv420p faststart mp4."""
    require_ffmpeg()
    audio = has_audio(video)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        list_file = tmp / "list.txt"
        lines = []
        for i, p in enumerate(picks):
            part = tmp / f"part_{i:03d}.mp4"
            cmd = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-ss",
                f"{p['start_s']:.3f}",
                "-i",
                str(video),
                "-t",
                f"{p['end_s'] - p['start_s']:.3f}",
                "-vf",
                f"scale=-2:'min(720,ih)',fps=30,{ffmpeg_drawtext()},format=yuv420p",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
            ]
            cmd += (
                ["-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2"]
                if audio
                else ["-an"]
            )
            cmd.append(str(part))
            subprocess.run(cmd, check=True)
            lines.append(f"file '{part}'")
        list_file.write_text("\n".join(lines) + "\n")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(out_path),
            ],
            check=True,
        )
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Build a motion-based video summary.")
    ap.add_argument("video")
    ap.add_argument("--target", type=float, default=75.0, help="target length in seconds")
    ap.add_argument("--max", type=float, default=90.0, help="hard upper limit in seconds")
    ap.add_argument(
        "--clip",
        type=float,
        default=4.5,
        help="length of each excerpt in seconds (between 3 and 5 s)",
    )
    args = ap.parse_args()

    out_file = OUT_ROOT / f"{stem(args.video)}_summary.mp4"
    if out_file.exists():
        dur = media_duration(out_file)
        if 60.0 <= dur <= 90.0:
            print(f"Summary already exists: {out_file} ({dur:.1f} s)")
            return

    scenes = detect_scenes_cached(args.video)
    res = motion_cached(args.video)
    values = suppress_cuts(res["times"], res["values"], scenes, res["fps"])
    info = probe(args.video)
    target = min(args.target, info["duration_s"])

    picks, total = plan(
        res["times"],
        values,
        scenes,
        target=target,
        clip_len=args.clip,
        max_len=args.max,
    )
    if not picks:
        raise SystemExit("No usable windows found. Check scenes.json and motion.json.")

    out = cut_and_join(args.video, picks, out_file)
    actual = media_duration(out)
    save_json(
        {
            "video": args.video,
            "target_s": target,
            "planned_s": total,
            "actual_s": actual,
            "clips": picks,
        },
        cache_dir(args.video) / "summary.json",
    )

    print(f"{len(picks)} excerpts, planned {total:.1f} s, actual {actual:.1f} s")
    for p in picks:
        print(
            f"  scene {p['scene']:3d}  {p['start_s']:8.1f}s to {p['end_s']:8.1f}s  "
            f"motion {p['motion']:.3f}"
        )
    print(f"summary video: {out}")
    print(f"duration: {actual:.1f} s")


if __name__ == "__main__":
    main()
