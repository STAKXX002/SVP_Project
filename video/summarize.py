"""Short video summary built from the most active moments, stitched with ffmpeg.

Usage:
    python -m video.summarize data/video/input.mp4
    python -m video.summarize data/video/input.mp4 --target 75 --clip 5

Needs scenes and motion; both are computed automatically if they are not cached.

Selection:
    1. In every scene find the non-overlapping --clip second windows ranked by mean motion.
    2. Take the best window of each scene, most active first, then the second-best windows, and so on,
       until --target seconds are filled. Windows below 20 percent of the top window's motion are
       skipped unless the target cannot be reached without them.
    3. A video with few cuts therefore contributes several windows per scene.
    4. Order chronologically, cut each window, re-encode to common settings and concatenate.
The result is trimmed so its length stays within --target and --max.

Outputs:
    outputs/<video>_summary.mp4
    outputs/cache/<video>/summary.json   chosen windows with scene index and motion score
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from video.common import (OUT_ROOT, cache_dir, has_audio, media_duration, probe, require_ffmpeg,
                          save_json, stem)
from video.motion import motion_cached, suppress_cuts
from video.scenes import detect_scenes_cached


def best_windows(times, values, scene, clip_len, step=0.5):
    """Return [(mean_motion, start_s, end_s)] for non-overlapping candidate windows, best first."""
    s, e = scene["start_s"], scene["end_s"]
    length = e - s
    win = min(clip_len, length)
    if win < 1.0:
        return []
    t = np.asarray(times)
    v = np.asarray(values)
    starts = np.arange(s, e - win + 1e-6, step)
    if len(starts) == 0:
        starts = np.array([s])
    scores = []
    for a in starts:
        m = (t >= a) & (t < a + win)
        scores.append(float(v[m].mean()) if m.any() else 0.0)
    order = np.argsort(scores)[::-1]
    chosen = []
    for i in order:
        a = float(starts[i])
        if all(a >= c[2] or a + win <= c[1] for c in chosen):
            chosen.append((scores[i], a, a + win))
    return chosen


def plan(times, values, scenes, target, clip_len, max_len, min_motion_frac=0.2):
    per_scene = {sc["index"]: best_windows(times, values, sc, clip_len) for sc in scenes}
    picks, total = [], 0.0

    def add(scene_idx, window):
        nonlocal total
        score, a, b = window
        take = min(b - a, max_len - total)
        if take < 1.0:
            return False
        picks.append({"scene": scene_idx, "start_s": a, "end_s": a + take, "motion": score})
        total += take
        return True

    top = max((w[0][0] for w in per_scene.values() if w), default=0.0)
    # Prefer windows with real motion; only fall back to quiet windows if the target cannot be met.
    for floor in (min_motion_frac * top, 0.0):
        used = {(p["scene"], p["start_s"]) for p in picks}
        depth = 0
        while total < target and depth < 20:
            cand = sorted(((w[depth][0], idx, w[depth]) for idx, w in per_scene.items()
                           if len(w) > depth and w[depth][0] >= floor
                           and (idx, w[depth][1]) not in used), reverse=True)
            if not cand:
                if all(len(w) <= depth for w in per_scene.values()):
                    break
                depth += 1
                continue
            for _, idx, w in cand:
                if total >= target:
                    break
                add(idx, w)
                used.add((idx, w[1]))
            depth += 1
    picks.sort(key=lambda p: p["start_s"])
    return picks, total


def cut_and_join(video, picks, out_path):
    require_ffmpeg()
    audio = has_audio(video)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        list_file = tmp / "list.txt"
        lines = []
        for i, p in enumerate(picks):
            part = tmp / f"part_{i:03d}.mp4"
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{p['start_s']:.3f}", "-i", str(video),
                   "-t", f"{p['end_s'] - p['start_s']:.3f}",
                   "-vf", "scale=-2:'min(720,ih)',fps=30,format=yuv420p",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
            cmd += ["-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2"] if audio else ["-an"]
            cmd.append(str(part))
            subprocess.run(cmd, check=True)
            lines.append(f"file '{part}'")
        list_file.write_text("\n".join(lines) + "\n")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", str(list_file), "-c", "copy", str(out_path)], check=True)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Build a motion-based video summary.")
    ap.add_argument("video")
    ap.add_argument("--target", type=float, default=75.0, help="target length in seconds")
    ap.add_argument("--max", type=float, default=90.0, help="hard upper limit in seconds")
    ap.add_argument("--clip", type=float, default=5.0, help="length of each excerpt in seconds")
    args = ap.parse_args()

    scenes = detect_scenes_cached(args.video)
    res = motion_cached(args.video)
    values = suppress_cuts(res["times"], res["values"], scenes, res["fps"])
    info = probe(args.video)
    target = min(args.target, info["duration_s"])

    picks, total = plan(res["times"], values, scenes, target, args.clip, args.max)
    if not picks:
        raise SystemExit("No usable windows found. Check scenes.json and motion.json.")
    out = cut_and_join(args.video, picks, OUT_ROOT / f"{stem(args.video)}_summary.mp4")
    actual = media_duration(out)
    save_json({"video": args.video, "target_s": target, "planned_s": total, "actual_s": actual,
               "clips": picks}, cache_dir(args.video) / "summary.json")

    print(f"{len(picks)} excerpts, planned {total:.1f} s, actual {actual:.1f} s")
    for p in picks:
        print(f"  scene {p['scene']:3d}  {p['start_s']:8.1f}s to {p['end_s']:8.1f}s  motion {p['motion']:.3f}")
    print(f"summary: {out}")
    if info["duration_s"] >= 600 and not (60 <= actual <= 90):
        print("NOTE: summary length is outside 60 to 90 s; adjust --target or --clip.")


if __name__ == "__main__":
    main()
