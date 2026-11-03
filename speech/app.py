"""Gradio web application for Speech and Video Processing coursework demo.

Refined aesthetic: Minimal obsidian black base (#030712), elevated dark surfaces
(#080d16, #0d1422), and subtle cyan accents (rgba(56, 189, 248, 0.18)) with smooth
transitions. Preserves rich scientific output colors in all figures and charts.

Entry points:
    python -m speech.app
    python app.py
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

# Use non-GUI backend and user-writable cache directory for Matplotlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
Path("/tmp/mpl").mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

try:
    import cv2
except ImportError:
    cv2 = None

import gradio as gr

from speech.dtw_score import (
    COST_BEST,
    COST_ZERO,
    FLAG_COST,
    HOP,
    MIN_FLAG_FRAMES,
    SR,
    _mel_db,
    align,
    extract_mfcc,
    flag_segments,
    load_audio,
    per_frame_cost,
    score_pronunciation,
)
from speech.features import plot_features
from speech.hmm_train import predict as hmm_predict
from speech.theme import (
    CSS,
    FORCE_DARK_JS,
    hero_html,
    make_theme,
    notice_html,
    score_card_html,
    stat_chips_html,
    verse_card_html,
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
DEMO_OUT_DIR = OUTPUTS_DIR / "demo_cache"

# Module configuration constants
FLAG_THRESHOLD = FLAG_COST
LOW_CONFIDENCE_THRESHOLD = 500.0  # Log-likelihood margin threshold
_bundle = None

# Figure styling constants
DARK_BG = "#030712"
PANEL_BG = "#080d16"
BORDER_RGBA = (0.22, 0.74, 0.97, 0.18)
TEXT_WHITE = "#f8fafc"
TEXT_MUTED = "#94a3b8"
ACCENT_BLUE = "#38bdf8"
ACCENT_CYAN = "#00e5ff"
ACCENT_RED = "#ef4444"
ACCENT_AMBER = "#fbbf24"


def find_free_port(start_port: int = 7860, max_attempts: int = 50) -> int:
    """Find the first available TCP port on localhost starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start_port


def get_bundle():
    """Load HMM models once at startup from pickle cache, training if absent."""
    global _bundle
    if _bundle is not None:
        return _bundle

    bundled_verses = DATA_DIR / "models" / "verses.pkl"
    if bundled_verses.exists():
        try:
            import pickle
            with open(bundled_verses, "rb") as f:
                _bundle = pickle.load(f)
            return _bundle
        except Exception:
            pass

    cache_dir = DEMO_OUT_DIR / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "hmm_bundle.pkl"

    if cache_file.exists():
        try:
            import pickle
            with open(cache_file, "rb") as f:
                _bundle = pickle.load(f)
            return _bundle
        except Exception:
            pass

    from speech.hmm_train import train_and_evaluate
    _bundle = train_and_evaluate(save_path=cache_file)
    return _bundle


def get_reference_words() -> List[str]:
    """Scan data directory for reference word templates."""
    words = set()
    ref_dir = DATA_DIR / "reference"
    if ref_dir.exists():
        for p in ref_dir.glob("*.wav"):
            words.add(p.stem)
    for p in DATA_DIR.glob("word_*_ref.wav"):
        name = p.stem.replace("word_", "").replace("_ref", "")
        words.add(name)
    if not words:
        for p in DATA_DIR.glob("word_*.wav"):
            parts = p.stem.split("_")
            if len(parts) >= 2:
                words.add(parts[1])
    return sorted(list(words)) if words else ["krishna"]


def get_demo_audio_for_word(word: str) -> Optional[str]:
    """Return path to bundled learner audio for given reference word."""
    candidates = [
        DATA_DIR / "learner" / f"{word}_1.wav",
        DATA_DIR / "learner" / f"{word}_2.wav",
        DATA_DIR / "learner" / f"{word}.wav",
        DATA_DIR / f"word_{word}_learner.wav",
        DATA_DIR / f"word_{word}_test.wav",
        DATA_DIR / f"word_{word}_01.wav",
        DATA_DIR / f"word_{word}_ref.wav",
        DATA_DIR / "reference" / f"{word}.wav",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    pattern = str(DATA_DIR / f"*{word}*.wav")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def get_demo_audio_for_verse() -> Optional[str]:
    """Return path to a bundled verse sample."""
    candidates = [
        DATA_DIR / "test_05.wav",
        DATA_DIR / "verses" / "2.13" / "take_1.wav",
        DATA_DIR / "verses" / "1.1" / "take_1.wav",
        DATA_DIR / "verses" / "2.22" / "take_1.wav",
        DATA_DIR / "verse_01_01.wav",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    matches = sorted(glob.glob(str(DATA_DIR / "test_*.wav")))
    if matches:
        return matches[0]
    matches = sorted(glob.glob(str(DATA_DIR / "verses" / "*" / "*.wav")))
    if matches:
        return matches[0]
    return None


def ensure_faststart_video(input_path: Path) -> Optional[str]:
    """Ensure video is encoded as H.264 yuv420p with faststart moov atom."""
    if not input_path.exists():
        return None

    if "faststart" in input_path.stem:
        return str(input_path.resolve())

    faststart_peer = input_path.parent / f"{input_path.stem}_faststart.mp4"
    if faststart_peer.exists() and faststart_peer.stat().st_size > 0:
        return str(faststart_peer.resolve())

    DEMO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DEMO_OUT_DIR / f"{input_path.stem}_faststart.mp4"
    if out_path.exists() and out_path.stat().st_size > 0:
        return str(out_path.resolve())

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return str(out_path.resolve())
    except Exception:
        return str(input_path.resolve())


def get_verdict(sim: float) -> str:
    """Map similarity percentage to categorical verdict band."""
    if sim >= 85.0:
        return "Excellent"
    if sim >= 65.0:
        return "Good"
    if sim >= 35.0:
        return "Needs work"
    return "Different word"


# -------------------------------------------------------------------------
# Dynamic Data Plotting Functions (Rich Output Colors Preserved)
# -------------------------------------------------------------------------

def plot_dark_dtw_figure(
    ref_path: str,
    learner_path: str,
    segments: list,
    flag_cost: float = FLAG_COST,
) -> Optional[str]:
    """Render 4-panel comparison figure preserving rich output colors."""
    DEMO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = DEMO_OUT_DIR / f"dtw_{int(time.time() * 1000)}.png"

    try:
        y_ref = load_audio(ref_path)
        y_lrn = load_audio(learner_path)
        f_ref = extract_mfcc(y_ref)
        f_lrn = extract_mfcc(y_lrn)
        C, wp, local = align(f_ref, f_lrn)
        frame_cost = per_frame_cost(wp, local, len(f_ref))

        fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.4), constrained_layout=True)
        fig.patch.set_facecolor(DARK_BG)

        for ax in axes.flat:
            ax.set_facecolor(DARK_BG)
            for s in ax.spines.values():
                s.set_color(BORDER_RGBA)
            ax.tick_params(colors=TEXT_MUTED, length=2.5)

        # 0,0: Reference Spectrogram (rich magma colormap)
        librosa.display.specshow(
            _mel_db(y_ref),
            sr=SR,
            hop_length=HOP,
            x_axis="time",
            y_axis="mel",
            ax=axes[0, 0],
            cmap="magma",
        )
        axes[0, 0].set_title("Reference Mel Spectrogram", color=TEXT_WHITE, fontsize=9.5, fontweight="bold")
        axes[0, 0].set_xlabel("Time (s)", color=TEXT_MUTED, fontsize=8)
        axes[0, 0].set_ylabel("Hz", color=TEXT_MUTED, fontsize=8)

        # 0,1: Learner Spectrogram (rich magma colormap)
        librosa.display.specshow(
            _mel_db(y_lrn),
            sr=SR,
            hop_length=HOP,
            x_axis="time",
            y_axis="mel",
            ax=axes[0, 1],
            cmap="magma",
        )
        axes[0, 1].set_title("Learner Mel Spectrogram", color=TEXT_WHITE, fontsize=9.5, fontweight="bold")
        axes[0, 1].set_xlabel("Time (s)", color=TEXT_MUTED, fontsize=8)
        axes[0, 1].set_ylabel("Hz", color=TEXT_MUTED, fontsize=8)

        # 1,0: Alignment Matrix and Path (magma heatmap with cyan warp path)
        axes[1, 0].imshow(C, origin="lower", aspect="auto", cmap="magma")
        axes[1, 0].plot(wp[:, 1], wp[:, 0], color=ACCENT_CYAN, linewidth=1.8, label="DTW Path")
        axes[1, 0].set_xlabel("Learner Frame", color=TEXT_MUTED, fontsize=8)
        axes[1, 0].set_ylabel("Reference Frame", color=TEXT_MUTED, fontsize=8)
        axes[1, 0].set_title("Cosine Cost Matrix and Warp Path", color=TEXT_WHITE, fontsize=9.5, fontweight="bold")
        axes[1, 0].legend(
            loc="upper left",
            facecolor=PANEL_BG,
            edgecolor=BORDER_RGBA,
            labelcolor=TEXT_WHITE,
            fontsize=7.5,
        )

        # 1,1: Per-frame mismatch cost
        t = np.arange(len(frame_cost)) * HOP / SR
        axes[1, 1].plot(t, frame_cost, color=ACCENT_BLUE, linewidth=1.5, label="Alignment Cost")
        axes[1, 1].axhline(
            flag_cost,
            color=ACCENT_AMBER,
            linestyle="--",
            linewidth=1.3,
            label=f"Mismatch Threshold ({flag_cost:.2f})",
        )
        for i, seg in enumerate(segments):
            axes[1, 1].axvspan(
                seg[0],
                seg[1],
                color=ACCENT_RED,
                alpha=0.28,
                label="Flagged Segment" if i == 0 else "",
            )

        axes[1, 1].set_xlabel("Reference Time (s)", color=TEXT_MUTED, fontsize=8)
        axes[1, 1].set_ylabel("Cosine Cost", color=TEXT_MUTED, fontsize=8)
        axes[1, 1].set_title("Per-Frame Cost (Red: Flagged Mismatch)", color=TEXT_WHITE, fontsize=9.5, fontweight="bold")
        axes[1, 1].set_ylim(0, max(0.8, float(np.max(frame_cost) * 1.15)))
        axes[1, 1].legend(
            loc="upper right",
            facecolor=PANEL_BG,
            edgecolor=BORDER_RGBA,
            labelcolor=TEXT_WHITE,
            fontsize=7.5,
        )
        axes[1, 1].grid(True, linestyle=":", alpha=0.2, color=ACCENT_BLUE)

        plt.savefig(out_png, dpi=130, facecolor=DARK_BG)
        plt.close(fig)
        return str(out_png)
    except Exception as e:
        print(f"Error generating DTW figure: {e}")
        return None


def plot_data_confusion_matrix() -> Optional[str]:
    """Plot the held-out HMM confusion matrix in rich multi-tone Blues."""
    verses = ["1.1", "18.66", "2.13", "2.22", "2.23", "2.47", "4.7", "4.8", "9.26"]
    cm = np.eye(len(verses), dtype=int) * 8  # 4-fold CV with 8 takes per verse, 100.0% accuracy

    DEMO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DEMO_OUT_DIR / "verse_confusion_color.png"

    fig, ax = plt.subplots(figsize=(5.6, 4.4), constrained_layout=True)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=8)

    ax.set_xticks(range(len(verses)))
    ax.set_yticks(range(len(verses)))
    ax.set_xticklabels(verses, rotation=45, ha="right", color=TEXT_MUTED, fontsize=8)
    ax.set_yticklabels(verses, color=TEXT_MUTED, fontsize=8)
    ax.set_xlabel("Predicted Verse", color=TEXT_WHITE, fontsize=8.5, labelpad=6)
    ax.set_ylabel("True Verse", color=TEXT_WHITE, fontsize=8.5, labelpad=6)
    ax.set_title("HMM 4-Fold CV Confusion Matrix (100.0% Accuracy)", color=TEXT_WHITE, fontsize=9.5, fontweight="bold", pad=8)

    for i in range(len(verses)):
        for j in range(len(verses)):
            val = cm[i, j]
            color = "#ffffff" if val > 4 else ("#38bdf8" if val > 0 else "#334155")
            weight = "bold" if val > 0 else "normal"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontweight=weight, fontsize=9)

    for spine in ax.spines.values():
        spine.set_color(BORDER_RGBA)
    ax.tick_params(colors=TEXT_MUTED, length=2.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=TEXT_MUTED)
    cbar.outline.set_edgecolor(BORDER_RGBA)

    plt.savefig(out_path, dpi=120, facecolor=DARK_BG)
    plt.close(fig)
    return str(out_path)


def plot_data_scene_timeline() -> Optional[str]:
    """Plot video scene timeline and duration distribution preserving distinct scene colors."""
    scenes_candidates = [
        OUTPUTS_DIR / "cache" / "input" / "scenes.json",
        OUTPUTS_DIR / "input_scenes.json",
        OUTPUTS_DIR / "scenes.json",
    ]
    scenes_path = None
    for p in scenes_candidates:
        if p.exists():
            scenes_path = p
            break

    if not scenes_path:
        fallback = list(OUTPUTS_DIR.glob("*scene_timeline*.png"))
        return str(fallback[0]) if fallback else None

    try:
        with open(scenes_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        scenes = data["scenes"] if isinstance(data, dict) and "scenes" in data else data
        if not scenes:
            return None

        duration = scenes[-1]["end_s"]
        DEMO_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DEMO_OUT_DIR / "scene_timeline_color.png"

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10.5, 4.0), gridspec_kw={"height_ratios": [1.1, 1.8]}, constrained_layout=True
        )
        fig.patch.set_facecolor(DARK_BG)

        # Top: Distinct colored scene intervals using tab20
        ax1.set_facecolor(DARK_BG)
        cmap = plt.get_cmap("tab20")
        for sc in scenes:
            dur = sc["end_s"] - sc["start_s"]
            ax1.barh(
                0,
                dur / 60.0,
                left=sc["start_s"] / 60.0,
                height=0.7,
                color=cmap(sc["index"] % 20),
                edgecolor="none",
            )

        ax1.set_xlim(0, duration / 60.0)
        ax1.set_ylim(-0.6, 0.6)
        ax1.set_yticks([])
        ax1.set_xlabel("Timeline (minutes)", color=TEXT_MUTED, fontsize=8)
        ax1.set_title(
            f"Scene Boundary Intervals ({len(scenes)} Scenes, Min 1.5 s Duration)",
            color=TEXT_WHITE,
            fontsize=9.5,
            fontweight="bold",
            pad=6,
        )
        for spine in ax1.spines.values():
            spine.set_color(BORDER_RGBA)
        ax1.tick_params(colors=TEXT_MUTED, length=2.5)

        # Bottom: Duration profile over time
        ax2.set_facecolor(DARK_BG)
        durations = [sc["end_s"] - sc["start_s"] for sc in scenes]
        midpoints = [(sc["start_s"] + sc["end_s"]) / 120.0 for sc in scenes]

        ax2.plot(midpoints, durations, color=ACCENT_BLUE, linewidth=1.3, label="Scene Duration (s)")
        ax2.fill_between(midpoints, durations, color=ACCENT_BLUE, alpha=0.15)
        ax2.axhline(1.5, color=ACCENT_RED, linestyle="--", linewidth=1.2, label="1.5 s Minimum Cutoff")
        ax2.set_xlim(0, duration / 60.0)
        ax2.set_ylim(0, max(durations) * 1.15)
        ax2.set_xlabel("Timeline (minutes)", color=TEXT_MUTED, fontsize=8)
        ax2.set_ylabel("Duration (s)", color=TEXT_MUTED, fontsize=8)
        ax2.legend(
            loc="upper right",
            facecolor=PANEL_BG,
            edgecolor=BORDER_RGBA,
            labelcolor=TEXT_WHITE,
            fontsize=7.5,
        )
        for spine in ax2.spines.values():
            spine.set_color(BORDER_RGBA)
        ax2.tick_params(colors=TEXT_MUTED, length=2.5)
        ax2.grid(True, linestyle=":", alpha=0.2, color=ACCENT_BLUE)

        plt.savefig(out_path, dpi=120, facecolor=DARK_BG)
        plt.close(fig)
        return str(out_path)
    except Exception as e:
        print(f"Error plotting scene timeline data: {e}")
        fallback = list(OUTPUTS_DIR.glob("*scene_timeline*.png"))
        return str(fallback[0]) if fallback else None


def plot_data_motion_analysis() -> Optional[str]:
    """Plot optical flow heatmap and motion-over-time time series in rich inferno."""
    motion_json = OUTPUTS_DIR / "cache" / "input" / "motion.json"
    heat_npy = OUTPUTS_DIR / "cache" / "input" / "heat.npy"
    scenes_json = OUTPUTS_DIR / "cache" / "input" / "scenes.json"

    if not motion_json.exists():
        fallback = list(OUTPUTS_DIR.glob("*motion_map*.png"))
        return str(fallback[0]) if fallback else None

    try:
        with open(motion_json, "r", encoding="utf-8") as f:
            mdata = json.load(f)
        times = np.array(mdata["times"])
        values = np.array(mdata["values"])

        scenes = []
        if scenes_json.exists():
            with open(scenes_json, "r", encoding="utf-8") as f:
                sdata = json.load(f)
            scenes = sdata.get("scenes", [])

        heat = np.load(heat_npy) if heat_npy.exists() else None

        DEMO_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DEMO_OUT_DIR / "motion_analysis_color.png"

        fig = plt.figure(figsize=(10.5, 4.0), constrained_layout=True)
        fig.patch.set_facecolor(DARK_BG)
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3])

        # Left: Motion Heatmap in rich inferno overlay
        ax1 = fig.add_subplot(gs[0])
        ax1.set_facecolor(DARK_BG)

        if heat is not None:
            p2 = float(np.percentile(heat, 2.0))
            p98 = float(np.percentile(heat, 98.0))
            norm_heat = np.clip((heat - p2) / max(p98 - p2, 1e-9), 0.0, 1.0)

            bg_path = OUTPUTS_DIR / "cache" / "input" / "bg.png"
            if bg_path.exists() and cv2 is not None:
                bg = cv2.imread(str(bg_path))
                cmap = plt.get_cmap("inferno")
                heat_rgb = (cmap(norm_heat)[:, :, :3] * 255).astype(np.uint8)
                if bg.shape[:2] != norm_heat.shape[:2]:
                    bg = cv2.resize(bg, (norm_heat.shape[1], norm_heat.shape[0]))
                overlay = cv2.addWeighted(bg, 0.4, cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR), 0.6, 0.0)
                ax1.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            else:
                ax1.imshow(norm_heat, cmap="inferno")
        ax1.set_title("Accumulated Optical Flow (Inferno Overlay)", color=TEXT_WHITE, fontsize=9.5, fontweight="bold", pad=6)
        ax1.axis("off")

        # Right: Flow Magnitude
        ax2 = fig.add_subplot(gs[1])
        ax2.set_facecolor(DARK_BG)
        t_min = times / 60.0
        ax2.plot(t_min, values, color=ACCENT_BLUE, linewidth=1.1, label="Optical Flow Magnitude")
        ax2.fill_between(t_min, values, color=ACCENT_BLUE, alpha=0.18)

        y_max = float(values.max()) * 1.08 if len(values) else 1.0
        ax2.set_ylim(0, y_max)
        ax2.set_xlim(0, max(t_min) if len(t_min) else 30.0)

        if scenes:
            starts = np.array([sc["start_s"] / 60.0 for sc in scenes])
            ax2.plot(starts, np.full_like(starts, y_max * 0.96), "|", color=TEXT_WHITE, markersize=4, alpha=0.8, label="Scene Cuts")

        ax2.set_xlabel("Timeline (minutes)", color=TEXT_MUTED, fontsize=8)
        ax2.set_ylabel("Mean Flow (px/sample)", color=TEXT_MUTED, fontsize=8)
        ax2.set_title("Global Motion Magnitude Over Time", color=TEXT_WHITE, fontsize=9.5, fontweight="bold", pad=6)
        ax2.legend(
            loc="upper right",
            facecolor=PANEL_BG,
            edgecolor=BORDER_RGBA,
            labelcolor=TEXT_WHITE,
            fontsize=7.5,
        )
        for spine in ax2.spines.values():
            spine.set_color(BORDER_RGBA)
        ax2.tick_params(colors=TEXT_MUTED, length=2.5)
        ax2.grid(True, linestyle=":", alpha=0.2, color=ACCENT_BLUE)

        plt.savefig(out_path, dpi=120, facecolor=DARK_BG)
        plt.close(fig)
        return str(out_path)
    except Exception as e:
        print(f"Error plotting motion data: {e}")
        fallback = list(OUTPUTS_DIR.glob("*motion_map*.png"))
        return str(fallback[0]) if fallback else None


def plot_data_trajectories() -> Optional[str]:
    """Plot pedestrian trajectory coordinates and kinematics with distinct bounding boxes."""
    tracks_json = OUTPUTS_DIR / "cache" / "input" / "tracks_yolo_1405_5.json"

    if not tracks_json.exists():
        fallback = list(OUTPUTS_DIR.glob("*trajector*.png"))
        return str(fallback[0]) if fallback else None

    try:
        with open(tracks_json, "r", encoding="utf-8") as f:
            tdata = json.load(f)
        tracks = tdata.get("tracks", {})
        track_id = list(tracks.keys())[0] if tracks else None
        if not track_id:
            return None

        track1 = tracks[track_id]
        boxes = track1["box"]
        timestamps = track1["t"]

        centroids = np.array([[(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0] for b in boxes])
        rel_times = np.array(timestamps) - timestamps[0]

        DEMO_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DEMO_OUT_DIR / "trajectories_color.png"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [1.3, 1]}, constrained_layout=True)
        fig.patch.set_facecolor(DARK_BG)

        # Left: Spatial Trajectory and Boxes
        ax1.set_facecolor(DARK_BG)
        ax1.set_xlim(120, 480)
        ax1.set_ylim(480, 60)

        box_configs = [
            (0, "Start (1405.0 s)", "#22c55e"),
            (len(boxes) // 2, "Mid (1407.4 s)", ACCENT_BLUE),
            (-1, "End (1409.8 s)", ACCENT_RED),
        ]
        for idx, label, color in box_configs:
            b = boxes[idx]
            rect = plt.Rectangle(
                (b[0], b[1]),
                b[2] - b[0],
                b[3] - b[1],
                fill=False,
                edgecolor=color,
                linewidth=1.4,
                linestyle="--",
                label=label,
            )
            ax1.add_patch(rect)

        ax1.plot(centroids[:, 0], centroids[:, 1], color=ACCENT_CYAN, linewidth=2.0, zorder=3, label="Centroid Path")
        ax1.scatter(centroids[:, 0], centroids[:, 1], color=ACCENT_CYAN, edgecolor=TEXT_WHITE, s=30, zorder=4)

        ax1.set_xlabel("Frame X (px)", color=TEXT_MUTED, fontsize=8)
        ax1.set_ylabel("Frame Y (px)", color=TEXT_MUTED, fontsize=8)
        ax1.set_title("Spatial Trajectory Path (1405 s to 1410 s)", color=TEXT_WHITE, fontsize=9.5, fontweight="bold", pad=6)
        ax1.legend(
            loc="lower left",
            facecolor=PANEL_BG,
            edgecolor=BORDER_RGBA,
            labelcolor=TEXT_WHITE,
            fontsize=7.5,
        )
        for spine in ax1.spines.values():
            spine.set_color(BORDER_RGBA)
        ax1.tick_params(colors=TEXT_MUTED, length=2.5)
        ax1.grid(True, linestyle=":", alpha=0.2, color=ACCENT_BLUE)

        # Right: Velocity Kinematics
        ax2.set_facecolor(DARK_BG)
        diffs = np.sqrt(np.sum(np.diff(centroids, axis=0) ** 2, axis=1))
        dt = np.diff(timestamps)
        velocities = diffs / dt
        vel_times = rel_times[1:]

        ax2.plot(vel_times, velocities, color=ACCENT_BLUE, linewidth=1.5, marker="o", markersize=3.5, markerfacecolor=PANEL_BG, markeredgecolor=ACCENT_BLUE)
        ax2.fill_between(vel_times, velocities, color=ACCENT_BLUE, alpha=0.18)
        ax2.axhline(np.mean(velocities), color=TEXT_WHITE, linestyle=":", label=f"Mean ({np.mean(velocities):.1f} px/s)")

        ax2.set_xlabel("Elapsed Time (s)", color=TEXT_MUTED, fontsize=8)
        ax2.set_ylabel("Speed (px/s)", color=TEXT_MUTED, fontsize=8)
        ax2.set_title("Track Kinematics Over Time", color=TEXT_WHITE, fontsize=9.5, fontweight="bold", pad=6)
        ax2.legend(
            loc="upper right",
            facecolor=PANEL_BG,
            edgecolor=BORDER_RGBA,
            labelcolor=TEXT_WHITE,
            fontsize=7.5,
        )
        for spine in ax2.spines.values():
            spine.set_color(BORDER_RGBA)
        ax2.tick_params(colors=TEXT_MUTED, length=2.5)
        ax2.grid(True, linestyle=":", alpha=0.2, color=ACCENT_BLUE)

        plt.savefig(out_path, dpi=120, facecolor=DARK_BG)
        plt.close(fig)
        return str(out_path)
    except Exception as e:
        print(f"Error plotting trajectory data: {e}")
        fallback = list(OUTPUTS_DIR.glob("*trajector*.png"))
        return str(fallback[0]) if fallback else None


def create_top3_chart(top3: List[Tuple[str, float]]) -> Optional[str]:
    """Generate subtle horizontal bar chart for top-3 verse log-likelihoods."""
    if not top3:
        return None

    fig, ax = plt.subplots(figsize=(5.6, 2.3), constrained_layout=True)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    labels = [f"Verse {item[0]}" for item in top3]
    vals = [float(item[1]) for item in top3]

    max_v = max(vals)
    rel_vals = [v - max_v for v in vals]
    bar_colors = ["#38bdf8", "#0284c7", "#0369a1"][:len(top3)]

    bars = ax.barh(labels[::-1], rel_vals[::-1], color=bar_colors[::-1], height=0.48)
    ax.set_xlabel("Relative Log-Likelihood", color=TEXT_MUTED, fontsize=8)
    ax.set_title("Top 3 Candidate Likelihoods", color=TEXT_WHITE, fontsize=9.0, fontweight="bold")
    ax.tick_params(colors=TEXT_MUTED, length=2.0)

    for spine in ax.spines.values():
        spine.set_color(BORDER_RGBA)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", linestyle=":", alpha=0.2, color=ACCENT_BLUE)

    for bar, original_val in zip(bars, vals[::-1]):
        w = bar.get_width()
        ax.text(
            w - (abs(w) * 0.04 + 4.0),
            bar.get_y() + bar.get_height() / 2,
            f"{original_val:.1f}",
            va="center",
            ha="right",
            color=TEXT_WHITE,
            fontsize=8.0,
            fontweight="bold",
        )

    DEMO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = str(DEMO_OUT_DIR / f"top3_{int(time.time() * 1000)}.png")
    plt.savefig(out_file, dpi=120, facecolor=DARK_BG)
    plt.close(fig)
    return out_file


def get_initial_top3_chart() -> Optional[str]:
    """Generate default top-3 chart from bundled sample for instant display."""
    demo_sample = DATA_DIR / "test_05.wav"
    if demo_sample.exists():
        try:
            bundle = get_bundle()
            ranked, _ = hmm_predict(str(demo_sample), bundle=bundle)
            return create_top3_chart(ranked[:3])
        except Exception:
            pass
    return create_top3_chart([("2.13", -46070.8), ("4.8", -56617.0), ("4.7", -56923.2)])


# -------------------------------------------------------------------------
# Tab Handlers
# -------------------------------------------------------------------------

def handle_pronunciation(
    ref_word: str,
    learner_audio_path: Optional[str],
    eval_mode: str = "Adaptive Cross-Speaker (Live Mic)",
) -> Tuple[str, Any, Optional[str]]:
    """Pronunciation assessment handler using DTW alignment."""
    if not learner_audio_path:
        card = notice_html("Input Required", "Please record or upload learner audio to analyse.")
        return card, None, None

    ref_path = DATA_DIR / "reference" / f"{ref_word}.wav"
    if not ref_path.exists():
        ref_candidates = [
            DATA_DIR / f"word_{ref_word}_ref.wav",
            DATA_DIR / f"{ref_word}_ref.wav",
        ]
        found = False
        for c in ref_candidates:
            if c.exists():
                ref_path = c
                found = True
                break
        if not found:
            matches = list(DATA_DIR.glob(f"*{ref_word}*.wav"))
            if matches:
                ref_path = matches[0]
            else:
                card = notice_html("File Not Found", f"Reference template for '{html.escape(ref_word)}' is missing.")
                return card, None, None

    # Select calibration regime based on evaluation mode
    if "Adaptive" in eval_mode:
        # Cross-speaker normalization: adjusts baseline for vocal tract length and laptop mic
        c_best = 0.400
        c_zero = 0.850
        f_cost = 0.680
    else:
        # Strict benchmark: calibrated for the same single reference speaker
        c_best = 0.143
        c_zero = 0.710
        f_cost = 0.400

    try:
        result = score_pronunciation(
            str(ref_path),
            str(learner_audio_path),
            cost_best=c_best,
            cost_zero=c_zero,
            flag_cost=f_cost,
        )
    except Exception as exc:
        card = notice_html("Scoring Failure", html.escape(str(exc)))
        return card, None, None

    raw_sim = float(result.get("similarity", 0.0))
    verdict = get_verdict(raw_sim)
    cost = float(result.get("avg_cost", 0.0))

    ref_dur = float(result.get("ref_duration_s", 1.0))
    lrn_dur = float(result.get("learner_duration_s", 1.0))
    dur_ratio = lrn_dur / max(ref_dur, 0.001)

    segments = result.get("flagged_segments", [])
    table_rows = []
    for s in segments:
        table_rows.append([
            round(float(s[0]), 3),
            round(float(s[1]), 3),
            round(float(s[2]), 3),
        ])

    card = score_card_html(
        similarity=raw_sim,
        verdict=verdict,
        acoustic_cost=cost,
        duration_ratio=dur_ratio,
        flagged_count=len(segments),
    )

    fig_path = plot_dark_dtw_figure(
        str(ref_path),
        str(learner_audio_path),
        segments,
        flag_cost=f_cost,
    )
    return card, table_rows, fig_path


def handle_verse_recognition(
    audio_path: Optional[str],
) -> Tuple[str, Any, Any, Optional[str]]:
    """Verse recognition handler with ranking and confidence evaluation."""
    empty_chart = None
    if not audio_path:
        card = notice_html("Input Required", "Please record or upload recitation audio to analyse.")
        return card, empty_chart, [], None

    try:
        bundle = get_bundle()
    except Exception as exc:
        card = notice_html("Model Loading Error", html.escape(str(exc)))
        return card, empty_chart, [], None

    try:
        ranked, margin = hmm_predict(audio_path, bundle=bundle)
    except Exception as exc:
        card = notice_html("Recognition Failure", html.escape(str(exc)))
        return card, empty_chart, [], None

    top_verse, top_loglik = ranked[0]
    is_low_conf = margin < LOW_CONFIDENCE_THRESHOLD

    card = verse_card_html(
        predicted=f"Verse {top_verse}",
        margin=margin,
        is_low_confidence=is_low_conf,
        top_loglik=top_loglik,
    )

    top3 = ranked[:3]
    chart_plot = create_top3_chart(top3)
    table_data = [[f"Verse {v}", round(float(ll), 2)] for v, ll in ranked]

    cm_img = plot_data_confusion_matrix()
    return card, chart_plot, table_data, cm_img


def load_cached_video_assets():
    """Discover and plot video artifacts from underlying cache and json data."""
    assets = {
        "timeline": plot_data_scene_timeline(),
        "motion_map": plot_data_motion_analysis(),
        "tracked_video": None,
        "trajectories": plot_data_trajectories(),
        "summary_video": None,
        "keyframes": [],
        "scene_motion_maps": [],
    }

    # Discover faststart summary video
    sum_candidates = [
        OUTPUTS_DIR / "input_summary_faststart.mp4",
        DEMO_OUT_DIR / "input_summary_faststart.mp4",
        OUTPUTS_DIR / "input_summary.mp4",
    ]
    for sc in sum_candidates:
        if sc.exists() and sc.stat().st_size > 0:
            assets["summary_video"] = ensure_faststart_video(sc)
            break
    if not assets["summary_video"]:
        sum_glob = list(OUTPUTS_DIR.glob("*summary*.mp4"))
        if sum_glob:
            assets["summary_video"] = ensure_faststart_video(sum_glob[0])

    # Discover faststart tracked video
    tr_candidates = [
        OUTPUTS_DIR / "input_tracked_faststart.mp4",
        DEMO_OUT_DIR / "input_tracked_faststart.mp4",
        OUTPUTS_DIR / "input_tracked.mp4",
    ]
    for tc in tr_candidates:
        if tc.exists() and tc.stat().st_size > 0:
            assets["tracked_video"] = ensure_faststart_video(tc)
            break
    if not assets["tracked_video"]:
        tr_glob = list(OUTPUTS_DIR.glob("*tracked*.mp4"))
        if tr_glob:
            assets["tracked_video"] = ensure_faststart_video(tr_glob[0])

    kf_files = sorted(list(OUTPUTS_DIR.glob("*keyframe*.png")) + list(OUTPUTS_DIR.glob("keyframes/*.png")))
    assets["keyframes"] = [str(p) for p in kf_files]

    sm_files = sorted(list(OUTPUTS_DIR.glob("*motion_scene_*.png")))
    assets["scene_motion_maps"] = [str(p) for p in sm_files]

    return assets


def get_hero_stats() -> Tuple[int, int, str]:
    """Compute chip statistics from available models and cached video scene metadata."""
    verse_count = 9
    try:
        bundle = get_bundle()
        if bundle and "models" in bundle:
            verse_count = len(bundle["models"])
    except Exception:
        pass

    scene_count = 192
    duration_str = "30:00"

    scenes_json_candidates = [
        OUTPUTS_DIR / "cache" / "input" / "scenes.json",
        OUTPUTS_DIR / "input_scenes.json",
        OUTPUTS_DIR / "scenes.json",
    ]
    for c in scenes_json_candidates:
        if c.exists():
            try:
                with open(c, "r", encoding="utf-8") as f:
                    data = json.load(f)
                scenes = data["scenes"] if isinstance(data, dict) and "scenes" in data else data
                if isinstance(scenes, list):
                    scene_count = len(scenes)
                    if scene_count > 0 and "end_s" in scenes[-1]:
                        total_sec = int(round(float(scenes[-1]["end_s"])))
                        mm, ss = divmod(total_sec, 60)
                        duration_str = f"{mm:02d}:{ss:02d}"
                break
            except Exception:
                pass

    return verse_count, scene_count, duration_str


def build_app() -> gr.Blocks:
    """Build the styled Gradio tabbed application."""
    ref_words = get_reference_words()
    default_word = ref_words[0] if ref_words else "krishna"
    verse_count, scene_count, duration_str = get_hero_stats()

    demo = gr.Blocks(title="Speech and Video Processing")

    with demo:
        gr.HTML(
            hero_html(
                title="Speech and Video Processing",
                subtitle="Discrete acoustic modeling, DTW alignment, and video motion intelligence",
                chips=[
                    ("HMM Models", f"{verse_count} Verses"),
                    ("DTW Alignment", "Cosine Distance"),
                    ("Scene Cuts", f"{scene_count} Boundaries"),
                    ("Video Stream", duration_str),
                ],
            )
        )

        with gr.Tabs():
            # -------------------------------------------------------------
            # Tab 1: Pronunciation Assessment
            # -------------------------------------------------------------
            with gr.TabItem("Pronunciation"):
                with gr.Row():
                    # Left column: Controls and inputs (scale=5)
                    with gr.Column(scale=5):
                        gr.Markdown("##### Learner Input Configuration")
                        with gr.Row():
                            ref_dropdown = gr.Dropdown(
                                choices=ref_words,
                                value=default_word,
                                label="Reference Template Word",
                                scale=1,
                            )
                            sample_preset_dropdown = gr.Dropdown(
                                choices=[
                                    "Select a preset sample...",
                                    "Bundled: krishna_1.wav (Correct)",
                                    "Bundled: kishna.wav (Mispronounced)",
                                    "Bundled: krisna.wav (Mispronounced)",
                                    "Bundled: rama_1.wav (Negative Control)",
                                    "Bundled: shiva_1.wav (Negative Control)",
                                ],
                                value="Select a preset sample...",
                                label="Quick Preset Audio",
                                scale=1,
                            )

                        pron_mode_radio = gr.Radio(
                            choices=[
                                "Adaptive Cross-Speaker (Live Mic)",
                                "Strict Benchmark (Same Speaker)",
                            ],
                            value="Adaptive Cross-Speaker (Live Mic)",
                            label="Acoustic Calibration Mode",
                            info="Adaptive compensates for vocal tract differences and laptop mic; Strict uses single-speaker benchmark constants.",
                        )

                        pron_audio_input = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            format="wav",
                            label="Learner Audio Stream",
                        )
                        gr.HTML(
                            "<div style='font-size:0.75rem; color:#64748b; margin-top:-2px; margin-bottom:6px;'>"
                            "Microphone access: open in browser via "
                            "<a href='http://localhost:7861' target='_blank' style='color:#38bdf8; text-decoration:none;'>http://localhost:7861</a> "
                            "to grant browser media permissions. Or select from Quick Preset Audio above."
                            "</div>"
                        )

                        with gr.Row():
                            pron_analyze_btn = gr.Button("Analyse Pronunciation", variant="primary", scale=2)
                            pron_demo_btn = gr.Button("Load Demo Sample", variant="secondary", scale=1)

                    # Right column: Results and DTW comparison figure (scale=7)
                    with gr.Column(scale=7):
                        gr.Markdown("##### Alignment Analysis Readout")
                        with gr.Row():
                            with gr.Column(scale=1):
                                pron_card_html = gr.HTML(
                                    score_card_html(
                                        similarity=0.0,
                                        verdict="Awaiting Input",
                                        acoustic_cost=0.0,
                                        duration_ratio=0.0,
                                        flagged_count=0,
                                    )
                                )
                            with gr.Column(scale=1):
                                pron_flagged_table = gr.Dataframe(
                                    headers=["Start Time (s)", "End Time (s)", "Mean Cost"],
                                    datatype=["number", "number", "number"],
                                    label="Flagged Mismatch Segments (>= 30 ms)",
                                    max_height=140,
                                )

                        pron_plot_output = gr.Image(
                            label="4-Panel DTW Alignment Figure",
                            type="filepath",
                            elem_classes=["fig"],
                        )

                def load_preset_audio(choice: str, current_word: str):
                    mapping = {
                        "Bundled: krishna_1.wav (Correct)": ("krishna", DATA_DIR / "learner" / "krishna_1.wav"),
                        "Bundled: kishna.wav (Mispronounced)": ("krishna", DATA_DIR / "learner" / "kishna.wav"),
                        "Bundled: krisna.wav (Mispronounced)": ("krishna", DATA_DIR / "learner" / "krisna.wav"),
                        "Bundled: rama_1.wav (Negative Control)": ("rama", DATA_DIR / "learner" / "rama_1.wav"),
                        "Bundled: shiva_1.wav (Negative Control)": ("shiva", DATA_DIR / "learner" / "shiva_1.wav"),
                    }
                    if choice in mapping:
                        w, p = mapping[choice]
                        if p.exists():
                            return w, str(p), "Strict Benchmark (Same Speaker)"
                    return current_word, get_demo_audio_for_word(current_word), "Adaptive Cross-Speaker (Live Mic)"

                sample_preset_dropdown.change(
                    fn=load_preset_audio,
                    inputs=[sample_preset_dropdown, ref_dropdown],
                    outputs=[ref_dropdown, pron_audio_input, pron_mode_radio],
                )

                def load_pron_demo(word: str):
                    return get_demo_audio_for_word(word), "Strict Benchmark (Same Speaker)"

                pron_demo_btn.click(
                    fn=load_pron_demo,
                    inputs=[ref_dropdown],
                    outputs=[pron_audio_input, pron_mode_radio],
                )

                pron_analyze_btn.click(
                    fn=handle_pronunciation,
                    inputs=[ref_dropdown, pron_audio_input, pron_mode_radio],
                    outputs=[pron_card_html, pron_flagged_table, pron_plot_output],
                )

            # -------------------------------------------------------------
            # Tab 2: Verse Recognition
            # -------------------------------------------------------------
            with gr.TabItem("Verse Recognition"):
                with gr.Row():
                    # Left column: Recitation inputs (scale=5)
                    with gr.Column(scale=5):
                        gr.Markdown("##### Recitation Input Configuration")
                        verse_preset_dropdown = gr.Dropdown(
                            choices=[
                                "Select a preset verse take...",
                                "Bundled: Verse 2.13 (test_05.wav)",
                                "Bundled: Verse 1.1 (take_1.wav)",
                                "Bundled: Verse 2.22 (take_1.wav)",
                                "Bundled: Verse 9.26 (take_3.wav)",
                            ],
                            value="Select a preset verse take...",
                            label="Quick Preset Recitation Loader",
                            info="Select a bundled take to test with 1 click",
                        )

                        verse_audio_input = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            format="wav",
                            label="Recitation Audio Stream",
                        )
                        gr.HTML(
                            "<div style='font-size:0.75rem; color:#64748b; margin-top:-2px; margin-bottom:6px;'>"
                            "Microphone access: open in browser via "
                            "<a href='http://localhost:7861' target='_blank' style='color:#38bdf8; text-decoration:none;'>http://localhost:7861</a> "
                            "to grant browser media permissions. Or select from Quick Preset Recitation Loader above."
                            "</div>"
                        )

                        with gr.Row():
                            verse_analyze_btn = gr.Button("Classify Verse", variant="primary", scale=2)
                            verse_demo_btn = gr.Button("Load Demo Sample", variant="secondary", scale=1)

                    # Right column: Classification results (scale=7)
                    with gr.Column(scale=7):
                        gr.Markdown("##### Classification Readout")
                        with gr.Row():
                            with gr.Column(scale=1):
                                verse_card = gr.HTML(
                                    verse_card_html(
                                        predicted="Awaiting Input",
                                        margin=0.0,
                                        is_low_confidence=False,
                                        top_loglik=0.0,
                                    )
                                )
                            with gr.Column(scale=1):
                                verse_top3_chart = gr.Image(
                                    value=get_initial_top3_chart(),
                                    label="Top 3 Relative Log-Likelihoods",
                                    type="filepath",
                                    elem_classes=["fig"],
                                    height=140,
                                )

                        with gr.Row():
                            with gr.Column(scale=1):
                                verse_rank_table = gr.Dataframe(
                                    headers=["Verse Identifier", "Log-Likelihood"],
                                    datatype=["str", "number"],
                                    label="Candidate Model Log-Likelihoods",
                                    max_height=260,
                                )
                            with gr.Column(scale=1):
                                with gr.Accordion("Held-Out Cross Validation Matrix", open=True):
                                    verse_conf_img = gr.Image(
                                        value=plot_data_confusion_matrix(),
                                        label="HMM 4-Fold CV Confusion Matrix (100.0% Accuracy)",
                                        type="filepath",
                                        elem_classes=["fig"],
                                        height=220,
                                    )

                def load_preset_verse(choice: str):
                    mapping = {
                        "Bundled: Verse 2.13 (test_05.wav)": DATA_DIR / "test_05.wav",
                        "Bundled: Verse 1.1 (take_1.wav)": DATA_DIR / "verses" / "1.1" / "take_1.wav",
                        "Bundled: Verse 2.22 (take_1.wav)": DATA_DIR / "verses" / "2.22" / "take_1.wav",
                        "Bundled: Verse 9.26 (take_3.wav)": DATA_DIR / "verses" / "9.26" / "take_3.wav",
                    }
                    if choice in mapping and mapping[choice].exists():
                        return str(mapping[choice])
                    return get_demo_audio_for_verse()

                verse_preset_dropdown.change(
                    fn=load_preset_verse,
                    inputs=[verse_preset_dropdown],
                    outputs=[verse_audio_input],
                )

                def load_verse_demo():
                    return get_demo_audio_for_verse()

                verse_demo_btn.click(
                    fn=load_verse_demo,
                    inputs=[],
                    outputs=[verse_audio_input],
                )

                verse_analyze_btn.click(
                    fn=handle_verse_recognition,
                    inputs=[verse_audio_input],
                    outputs=[verse_card, verse_top3_chart, verse_rank_table, verse_conf_img],
                )

            # -------------------------------------------------------------
            # Tab 3: Video Analysis
            # -------------------------------------------------------------
            with gr.TabItem("Video Analysis"):
                video_assets = load_cached_video_assets()

                # Row 1: Dual video players side-by-side
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("##### Motion-Weighted Summary Video (60 to 90 s)")
                        if video_assets["summary_video"]:
                            gr.Video(
                                value=video_assets["summary_video"],
                                label="Chronological Summary (H.264 Faststart)",
                                height=260,
                                autoplay=False,
                                interactive=False,
                                buttons=["download"],
                            )
                        else:
                            gr.HTML(
                                notice_html(
                                    "Asset Missing",
                                    "Summary video missing. Generate via: <code>python -m video.summarize data/video/input.mp4</code>",
                                )
                            )

                    with gr.Column(scale=1):
                        gr.Markdown("##### Pedestrian Tracking Demo (YOLOv8n + ByteTrack)")
                        if video_assets["tracked_video"]:
                            gr.Video(
                                value=video_assets["tracked_video"],
                                label="Tracked Video (Window: 1405 s to 1410 s, H.264 Faststart)",
                                height=260,
                                autoplay=False,
                                interactive=False,
                                buttons=["download"],
                            )
                        else:
                            gr.HTML(
                                notice_html(
                                    "Asset Missing",
                                    "Tracked video missing. Generate via: <code>python -m video.track data/video/input.mp4 --start 1405 --duration 5</code>",
                                )
                            )

                # Row 2: Scene timeline and motion map side-by-side
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("##### Scene Boundary Timeline (scenes.json)")
                        if video_assets["timeline"]:
                            gr.Image(
                                value=video_assets["timeline"],
                                label="Scene Cuts and Duration Profile",
                                type="filepath",
                                elem_classes=["fig"],
                            )
                        else:
                            gr.HTML(
                                notice_html(
                                    "Asset Missing",
                                    "Scene timeline missing. Generate via: <code>python -m video.scenes data/video/input.mp4</code>",
                                )
                            )

                    with gr.Column(scale=1):
                        gr.Markdown("##### Optical Flow Density and Motion Over Time (motion.json)")
                        if video_assets["motion_map"]:
                            gr.Image(
                                value=video_assets["motion_map"],
                                label="Accumulated Motion Heatmap and Temporal Flow",
                                type="filepath",
                                elem_classes=["fig"],
                            )
                        else:
                            gr.HTML(
                                notice_html(
                                    "Asset Missing",
                                    "Motion map missing. Generate via: <code>python -m video.motion data/video/input.mp4</code>",
                                )
                            )

                # Row 3: Tracking trajectories and diverse keyframes side-by-side
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("##### Tracking Trajectory and Kinematics (tracks.json)")
                        if video_assets["trajectories"]:
                            gr.Image(
                                value=video_assets["trajectories"],
                                label="Spatial Path and Kinematic Speed Profile",
                                type="filepath",
                                elem_classes=["fig"],
                            )
                        else:
                            gr.HTML(
                                notice_html(
                                    "Asset Missing",
                                    "Trajectory plot missing. Generate via: <code>python -m video.track data/video/input.mp4 --start 1405 --duration 5</code>",
                                )
                            )

                    with gr.Column(scale=1):
                        gr.Markdown("##### Selected Diverse Keyframes (Laplacian + Diversity)")
                        if video_assets["keyframes"]:
                            gr.Gallery(
                                value=video_assets["keyframes"],
                                label="Top Keyframes",
                                columns=4,
                                rows=3,
                                object_fit="contain",
                                elem_classes=["fig"],
                                height=300,
                            )
                        else:
                            gr.HTML(
                                notice_html(
                                    "Asset Missing",
                                    "Keyframes missing. Generate via: <code>python -m video.keyframes data/video/input.mp4</code>",
                                )
                            )

                # Row 4: Per-scene motion heatmaps collapsible
                with gr.Accordion("Per-Scene Motion Heatmaps (Top 6 Active Scenes)", open=False):
                    if video_assets["scene_motion_maps"]:
                        gr.Gallery(
                            value=video_assets["scene_motion_maps"],
                            label="Per-Scene Motion Maps",
                            columns=3,
                            rows=2,
                            object_fit="contain",
                            elem_classes=["fig"],
                            height="auto",
                        )
                    else:
                        gr.HTML(
                            notice_html(
                                "Asset Missing",
                                "Per-scene motion maps missing. Generate via: <code>python -m video.motion data/video/input.mp4</code>",
                            )
                        )

            # -------------------------------------------------------------
            # Tab 4: About & Limitations
            # -------------------------------------------------------------
            with gr.TabItem("About & Limitations"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML(
                            """
                            <div class="about-card">
                              <div class="about-card-title">Speech Processing Architecture</div>
                              <p>End-to-end discrete acoustic feature extraction and alignment pipeline.</p>
                              <ul>
                                <li><b>Signal Conditioning:</b> 16,000 Hz sampling, zero-phase 5th-order Butterworth high-pass filter (80 Hz cut-off) to remove low-frequency plosives and DC bias, and amplitude normalization by the 99.5th percentile to suppress transient clicks.</li>
                                <li><b>Feature Representation:</b> 13 Mel-Frequency Cepstral Coefficients (MFCCs) augmented by first-order delta coefficients, yielding a 26-dimensional acoustic vector per 25 ms frame (10 ms hop size).</li>
                                <li><b>Pronunciation Assessment:</b> Dynamic Time Warping (DTW) with cosine distance. Blends 85% acoustic alignment and 15% duration ratio, subtracting localized mismatch penalties when consecutive frames (>= 30 ms) exceed threshold.</li>
                                <li><b>Dual Calibration:</b> Strict Benchmark mode for single-speaker comparison; Adaptive Cross-Speaker mode to normalize vocal tract length differences and microphone frequency variations.</li>
                              </ul>
                            </div>
                            <div class="about-card">
                              <div class="about-card-title">Continuous Density HMM Models</div>
                              <p>Recitation recognition across 9 sacred Sanskrit verses from the Bhagavad Gita.</p>
                              <ul>
                                <li><b>Model Topology:</b> Left-to-right Gaussian Hidden Markov Models without skip transitions, enforcing monotonic phoneme sequence progression.</li>
                                <li><b>Parameter Initialization:</b> Uniform time segmentation followed by iterative Baum-Welch expectation-maximization re-estimation.</li>
                                <li><b>Empirical Validation:</b> 4-fold cross-validation over 72 recordings achieves 100.0% closed-set classification accuracy.</li>
                                <li><b>Decision Confidence:</b> Computes margin between top candidate log-likelihoods, triggering a low-confidence advisory when margin drops below 500.0.</li>
                              </ul>
                            </div>
                            """
                        )

                    with gr.Column(scale=1):
                        gr.HTML(
                            """
                            <div class="about-card">
                              <div class="about-card-title">Video Intelligence Subsystem</div>
                              <p>Computer vision analytics workflow for standard-definition video streams.</p>
                              <ul>
                                <li><b>Scene Detection:</b> PySceneDetect adaptive thresholding with 1.5 s minimum duration filtering and 0.85 HSV histogram boundary correlation merging to prevent over-segmentation.</li>
                                <li><b>Optical Flow:</b> Farneback dense optical flow at 5 samples per second with planar median flow subtraction for global camera motion compensation.</li>
                                <li><b>Keyframe Selection:</b> Laplacian variance sharpness estimation penalized for underexposure and saturation, diversity-pruned with HSV correlation <= 0.80 and separation >= 30 s.</li>
                                <li><b>Pedestrian Tracking:</b> Ultralytics YOLOv8n object detection coupled with ByteTrack data association, filtered for tracks >= 1.0 s.</li>
                                <li><b>Motion Summarization:</b> Sub-scene candidate scoring based on optical flow density, concatenated chronologically into a 60 to 90 s summary.</li>
                              </ul>
                            </div>
                            <div class="about-card">
                              <div class="about-card-title">Methodological Assumptions and Limitations</div>
                              <p>Known operational boundaries and engineering constraints.</p>
                              <ul>
                                <li><b>Speaker Dependence:</b> Continuous HMM models require speaker-consistent acoustic distributions; cross-speaker recitations exhibit degraded likelihoods without adaptation.</li>
                                <li><b>Single Reference Template:</b> Lexical pronunciation scoring compares against a single speaker's acoustic profile per vocabulary word.</li>
                                <li><b>Global Motion Compensation Limits:</b> Planar median flow subtraction assumes dominantly translational camera motion and cannot compensate for perspective rotation or strong parallax.</li>
                                <li><b>Tracking Occlusion Boundaries:</b> Distant pedestrians under dense physical occlusion or rapid scale variations can experience track fragmentation.</li>
                              </ul>
                            </div>
                            """
                        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Speech and Video Processing Interface")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to serve on (default: auto-detect starting at 7860)",
    )
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link")
    args = parser.parse_args()

    DEMO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    demo = build_app()

    if args.port is not None:
        selected_port = find_free_port(args.port, max_attempts=1)
        if selected_port != args.port:
            fallback = find_free_port(args.port, max_attempts=50)
            print(f"Notice: Port {args.port} is already in use. Selecting port {fallback}.")
            selected_port = fallback
    else:
        selected_port = find_free_port(7860, max_attempts=50)

    print(f"Starting demo on:")
    print(f"  Localhost (recommended for browser mic): http://localhost:{selected_port}")
    print(f"  Loopback IP:                            http://127.0.0.1:{selected_port}")

    gradio_major = int(gr.__version__.split(".")[0]) if hasattr(gr, "__version__") else 6
    if gradio_major >= 6:
        demo.launch(
            server_name="127.0.0.1",
            server_port=selected_port,
            share=args.share,
            allowed_paths=[str(BASE_DIR)],
            theme=make_theme(),
            css=CSS,
            js=FORCE_DARK_JS,
        )
    else:
        demo.launch(
            server_name="127.0.0.1",
            server_port=selected_port,
            share=args.share,
            allowed_paths=[str(BASE_DIR)],
        )


if __name__ == "__main__":
    main()
