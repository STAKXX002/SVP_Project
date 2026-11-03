"""Single Gradio Blocks interface for Sanskrit speech and video processing demo.

Entry point:
    python -m speech.app
Or:
    python app.py
"""

import argparse
import glob
import html
import json
import os
import subprocess
import time
from pathlib import Path

# Configure writable cache directory for Matplotlib
os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
Path("/tmp/mpl").mkdir(parents=True, exist_ok=True)

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from speech.dtw_score import format_report, score_pronunciation
from speech.hmm_train import (
    DEFAULT_DATA,
    DEFAULT_MODEL,
    features_for,
    list_dataset,
    load_bundle,
    predict,
    save_bundle,
    train_all,
)
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
from video.common import load_json, media_duration

DATA_DIR = Path("data")
REF_DIR = DATA_DIR / "reference"
LEARNER_DIR = DATA_DIR / "learner"
OUTPUTS_DIR = Path("outputs")
CACHE_DIR = OUTPUTS_DIR / "cache"
DEMO_OUT_DIR = OUTPUTS_DIR / "demo"

LOW_CONFIDENCE_THRESHOLD = 500.0  # Log-likelihood margin threshold
_bundle = None


def get_bundle():
    """Load HMM models once at startup from pickle cache, training if absent."""
    global _bundle
    if _bundle is not None:
        return _bundle

    if Path(DEFAULT_MODEL).exists():
        try:
            _bundle = load_bundle(DEFAULT_MODEL)
            return _bundle
        except Exception as exc:
            print(f"Warning: Failed to load {DEFAULT_MODEL} ({exc}), retraining cache...")

    print(f"Training HMM models from {DEFAULT_DATA} and caching to {DEFAULT_MODEL}...")
    dataset = list_dataset(DEFAULT_DATA)
    feats = {v: [features_for(p) for p in paths] for v, paths in dataset.items()}
    models = train_all(feats, n_states=None, n_mix=1)
    save_bundle(models, DEFAULT_MODEL)
    _bundle = {"models": models}
    return _bundle


def list_references():
    """Find reference wav files in data/reference and data/."""
    refs = []
    if REF_DIR.exists():
        refs.extend(sorted(p.name for p in REF_DIR.glob("*.wav")))
    if not refs and DATA_DIR.exists():
        refs.extend(sorted(p.name for p in DATA_DIR.glob("*.wav")))
    return refs


def ensure_faststart_video(mp4_path):
    """Ensure an mp4 video is H.264 yuv420p with faststart for browser playback."""
    if not mp4_path or not Path(mp4_path).exists():
        return None
    p = Path(mp4_path)
    if "faststart" in p.stem:
        return str(p)
    fast_path = p.parent / f"{p.stem}_faststart.mp4"
    if fast_path.exists() and fast_path.stat().st_mtime >= p.stat().st_mtime:
        return str(fast_path)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(p),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(fast_path),
    ]
    res = subprocess.run(cmd)
    if res.returncode == 0 and fast_path.exists():
        return str(fast_path)
    return str(p)


def get_verdict_band(similarity):
    """Determine descriptive verdict band from similarity percentage."""
    if similarity >= 80.0:
        return "Excellent"
    if similarity >= 60.0:
        return "Good"
    if similarity >= 30.0:
        return "Needs work"
    return "Different word"


def compute_hero_chips():
    """Compute summary stat chips dynamically from real data without hardcoding."""
    chips = []
    try:
        bundle = get_bundle()
        n_verses = len(bundle.get("models", {}))
        if n_verses:
            chips.append(("Verses", str(n_verses)))
    except Exception:
        pass

    scenes_path = CACHE_DIR / "input" / "scenes.json"
    if scenes_path.exists():
        try:
            sc_data = load_json(scenes_path)
            scenes = sc_data.get("scenes", [])
            if scenes:
                chips.append(("Scenes", str(len(scenes))))
                dur_s = scenes[-1].get("end_s", 0.0) - scenes[0].get("start_s", 0.0)
                dur_min = dur_s / 60.0
                chips.append(("Duration", f"{dur_min:.0f} min"))
        except Exception:
            pass

    try:
        refs = list_references()
        if refs:
            chips.append(("References", str(len(refs))))
    except Exception:
        pass

    summary_candidates = sorted(OUTPUTS_DIR.glob("*summary*.mp4"))
    if summary_candidates:
        try:
            dur = media_duration(summary_candidates[0])
            if dur > 0:
                chips.append(("Summary", f"{dur:.1f} s"))
        except Exception:
            pass

    return chips


def run_pronunciation(reference_name, learner_audio):
    """Pronunciation assessment handler returning score card HTML, segments, and figure."""
    try:
        if not reference_name:
            card = score_card_html(0.0, "Different word", ["Please select a reference word."])
            return card, [], None
        if not learner_audio:
            card = score_card_html(0.0, "Different word", ["Please record or upload audio."])
            return card, [], None

        ref_path = REF_DIR / reference_name
        if not ref_path.exists():
            ref_path = DATA_DIR / reference_name
        if not ref_path.exists():
            card = score_card_html(
                0.0, "Different word", [f"Could not find reference {reference_name}."]
            )
            return card, [], None

        t0 = time.time()
        DEMO_OUT_DIR.mkdir(parents=True, exist_ok=True)
        result = score_pronunciation(ref_path, learner_audio, DEMO_OUT_DIR)
        elapsed = time.time() - t0

        sim = result["similarity"]
        verdict = get_verdict_band(sim)

        segments = result.get("flagged_segments", [])
        if segments:
            seg_rows = [[f"{s:.2f}", f"{e:.2f}", f"{c:.3f}"] for s, e, c in segments]
        else:
            seg_rows = [["None", "None", "None"]]

        details = [
            f"Acoustic score: {result['acoustic_score']:.0f}%",
            f"Duration score: {result['duration_score']:.0f}%",
            f"Flagged segments: {len(segments)}",
            f"Computed in: {elapsed:.2f} s",
        ]
        card = score_card_html(sim, verdict, details)
        return card, seg_rows, result.get("figure")
    except Exception as exc:
        card = score_card_html(0.0, "Error", [f"Analysis failed: {type(exc).__name__}: {exc}"])
        return card, [], None


def load_pronunciation_demo():
    """Fill pronunciation tab with bundled demo files."""
    refs = list_references()
    def_ref = "krishna.wav" if "krishna.wav" in refs else (refs[0] if refs else None)
    demo_learner = LEARNER_DIR / "krishna_1.wav"
    learner_path = str(demo_learner) if demo_learner.exists() else None
    return def_ref, learner_path


def plot_top3_verses(ranked_list):
    """Render top-3 verses with log-likelihoods as a clean bar chart."""
    top3 = ranked_list[:3]
    verses = [item[0] for item in top3]
    logliks = [item[1] for item in top3]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    colors = ["#2b5c8f", "#5b8bbd", "#9cbdd9"]
    bars = ax.bar(verses, logliks, color=colors[: len(verses)], edgecolor="black", width=0.55)
    ax.set_ylabel("Total log-likelihood")
    ax.set_xlabel("Candidate verse")
    ax.set_title("Top-3 candidate verses by HMM log-likelihood")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, logliks):
        offset = abs(val) * 0.015 if val < 0 else 10
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val - offset,
            f"{val:.0f}",
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout()
    return fig


def run_verse_recognition(audio_path):
    """Verse recognition handler returning verse card HTML, bar chart, and table rows."""
    try:
        if not audio_path:
            card = verse_card_html("No audio", 0.0, True)
            return card, None, []

        bundle = get_bundle()
        ranked_list, margin = predict(audio_path, bundle=bundle)

        if not ranked_list:
            card = verse_card_html("None", 0.0, True)
            return card, None, []

        best_verse = ranked_list[0][0]
        low_conf = margin < LOW_CONFIDENCE_THRESHOLD

        card = verse_card_html(best_verse, margin, low_conf)
        fig = plot_top3_verses(ranked_list)
        table_rows = [[v, f"{ll:.1f}"] for v, ll in ranked_list]
        return card, fig, table_rows
    except Exception as exc:
        card = verse_card_html("Error", 0.0, True)
        return card, None, []


def load_verse_demo():
    """Fill verse tab with bundled demo file."""
    test_file = DATA_DIR / "test_05.wav"
    if test_file.exists():
        return str(test_file)
    fallback = list(DATA_DIR.glob("verses/*/*.wav"))
    return str(fallback[0]) if fallback else None


def discover_video_assets():
    """Discover cached video analysis outputs by globbing outputs/."""
    assets = {
        "timeline_img": None,
        "timeline_msg": "",
        "gallery_imgs": [],
        "gallery_msg": "",
        "sheet_img": None,
        "motion_img": None,
        "motion_msg": "",
        "tracked_video": None,
        "tracked_msg": "",
        "trajectories_img": None,
        "summary_video": None,
        "summary_msg": "",
        "scene_motion_maps": [],
    }

    # 1. Scene timeline image
    timelines = sorted(glob.glob("outputs/*scene_timeline*.png"))
    if timelines:
        assets["timeline_img"] = timelines[0]
    else:
        assets["timeline_msg"] = (
            "Scene timeline image missing. Generate using: "
            "`python -m video.scenes data/video/input.mp4`"
        )

    # 2. Keyframes gallery and contact sheet
    sheets = sorted(glob.glob("outputs/*keyframes*.png"))
    if sheets:
        assets["sheet_img"] = sheets[0]

    kf_files = sorted(glob.glob("outputs/keyframes/*/*.jpg"))
    if kf_files:
        assets["gallery_imgs"] = kf_files[:24]
    elif sheets:
        assets["gallery_msg"] = "Per-scene keyframe images missing; contact sheet displayed."
    else:
        assets["gallery_msg"] = (
            "Keyframe assets missing. Generate using: "
            "`python -m video.keyframes data/video/input.mp4`"
        )

    # 3. Motion map image
    motion_maps = sorted(glob.glob("outputs/*motion_map*.png"))
    if motion_maps:
        assets["motion_img"] = motion_maps[0]
    else:
        assets["motion_msg"] = (
            "Motion map missing. Generate using: "
            "`python -m video.motion data/video/input.mp4`"
        )

    # 4. Tracked video player and trajectories
    trajectories = [
        p for p in sorted(glob.glob("outputs/*trajectories*.png")) if "old_bgsub" not in p
    ]
    if trajectories:
        assets["trajectories_img"] = trajectories[0]

    tracked_videos = [
        p for p in sorted(glob.glob("outputs/*tracked*.mp4")) if "_raw" not in p
    ]
    if tracked_videos:
        fast = [p for p in tracked_videos if "faststart" in p]
        chosen = fast[0] if fast else tracked_videos[0]
        assets["tracked_video"] = ensure_faststart_video(chosen)
    else:
        assets["tracked_msg"] = (
            "Tracked video missing. Generate using: "
            "`python -m video.track data/video/input.mp4`"
        )

    # 5. Summary video player
    summary_videos = sorted(glob.glob("outputs/*summary*.mp4"))
    if summary_videos:
        fast = [p for p in summary_videos if "faststart" in p]
        chosen = fast[0] if fast else summary_videos[0]
        assets["summary_video"] = ensure_faststart_video(chosen)
    else:
        assets["summary_msg"] = (
            "Summary video missing. Generate using: "
            "`python -m video.summarize data/video/input.mp4`"
        )

    # 6. Per-scene motion maps
    assets["scene_motion_maps"] = sorted(glob.glob("outputs/*motion_scene_*.png"))

    return assets


def get_confusion_matrix_path():
    """Locate the confusion matrix image."""
    cms = sorted(glob.glob("outputs/*confusion*.png"))
    return cms[0] if cms else None


def build_app():
    """Build the multi-tab Gradio Blocks application."""
    try:
        get_bundle()
    except Exception as exc:
        print(f"Startup model warning: {exc}")

    refs = list_references()
    default_ref = refs[0] if refs else None
    cm_path = get_confusion_matrix_path()
    video_assets = discover_video_assets()
    hero_chips = compute_hero_chips()

    gradio_major = int(gr.__version__.split(".")[0]) if hasattr(gr, "__version__") else 6
    blocks_kwargs = {"title": "Speech and Video Processing"}
    if gradio_major < 6:
        blocks_kwargs.update({
            "theme": make_theme(),
            "css": CSS,
            "js": FORCE_DARK_JS,
        })

    with gr.Blocks(**blocks_kwargs) as demo:
        # Top Hero Header
        gr.HTML(
            hero_html(
                title="Speech and Video Processing",
                subtitle="Sanskrit pronunciation assessment, HMM verse recognition, and optical-flow video analysis.",
                chips=hero_chips,
            )
        )

        # ------------------------------------------------------------- Tab 1: Pronunciation
        with gr.Tab("Pronunciation"):
            with gr.Row():
                with gr.Column(scale=1):
                    ref_dropdown = gr.Dropdown(
                        choices=refs,
                        value=default_ref,
                        label="Reference Word",
                        interactive=True,
                    )
                    learner_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="Learner Audio (Record or Upload)",
                    )
                    with gr.Row():
                        pron_score_btn = gr.Button("Analyse", variant="primary")
                        pron_demo_btn = gr.Button("Load demo", variant="secondary")
                    refresh_refs_btn = gr.Button(
                        "Refresh Reference List", size="sm", variant="secondary"
                    )

                with gr.Column(scale=1):
                    initial_details = [
                        "Acoustic score: --",
                        "Duration score: --",
                        "Flagged segments: 0",
                    ]
                    score_card = gr.HTML(
                        score_card_html(0.0, "Ready", initial_details),
                        label="Score Card",
                    )
                    segments_table = gr.Dataframe(
                        headers=["Start (s)", "End (s)", "Mean Cost"],
                        label="Flagged Mismatch Segments (min 3 frames / 30 ms above cost 0.40)",
                        interactive=False,
                    )
                    dtw_figure = gr.Image(
                        label="Spectrogram Alignment, DTW Path, and Per-Frame Cost",
                        type="filepath",
                        interactive=False,
                        elem_classes=["fig"],
                    )

            # Wiring Tab 1
            refresh_refs_btn.click(
                fn=lambda: gr.Dropdown(choices=list_references()),
                outputs=ref_dropdown,
            )
            pron_demo_btn.click(
                fn=load_pronunciation_demo,
                outputs=[ref_dropdown, learner_input],
            )
            pron_score_btn.click(
                fn=run_pronunciation,
                inputs=[ref_dropdown, learner_input],
                outputs=[score_card, segments_table, dtw_figure],
            )

        # ------------------------------------------------------------- Tab 2: Verse Recognition
        with gr.Tab("Verse recognition"):
            with gr.Row():
                with gr.Column(scale=1):
                    verse_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="Verse Audio (Record or Upload)",
                    )
                    with gr.Row():
                        verse_rec_btn = gr.Button("Recognise", variant="primary")
                        verse_demo_btn = gr.Button("Load demo", variant="secondary")

                    with gr.Accordion(
                        "Held-Out Confusion Matrix (4-Fold Cross-Validation)", open=False
                    ):
                        cm_image = gr.Image(
                            value=cm_path,
                            label="Held-Out Confusion Matrix",
                            type="filepath",
                            interactive=False,
                            elem_classes=["fig"],
                        )

                with gr.Column(scale=1):
                    verse_card = gr.HTML(
                        verse_card_html("Ready", 0.0, False),
                        label="Verse Prediction",
                    )
                    top3_chart = gr.Plot(
                        label="Top-3 Verses with Log-Likelihoods",
                        elem_classes=["fig"],
                    )
                    verse_table = gr.Dataframe(
                        headers=["Verse", "Log-Likelihood"],
                        label="All Candidate Verses (Ranked)",
                        interactive=False,
                    )

            # Wiring Tab 2
            verse_demo_btn.click(
                fn=load_verse_demo,
                outputs=verse_input,
            )
            verse_rec_btn.click(
                fn=run_verse_recognition,
                inputs=verse_input,
                outputs=[verse_card, top3_chart, verse_table],
            )

        # ------------------------------------------------------------- Tab 3: Video Analysis
        with gr.Tab("Video analysis"):
            # Summary video first and large
            gr.Markdown("### Video Summary (60 to 90 s)")
            if video_assets["summary_video"]:
                summary_player = gr.Video(
                    value=video_assets["summary_video"],
                    label="Stitched Summary Video (H.264 Faststart)",
                    interactive=False,
                )
            else:
                gr.HTML(
                    notice_html(
                        "Summary video missing",
                        "Generate using: <code>python -m video.summarize data/video/input.mp4</code>",
                    )
                )

            # Two-column grid: scene timeline & motion map (left); tracked video & trajectory plot (right)
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Scene Boundary Timeline")
                    if video_assets["timeline_img"]:
                        timeline_img = gr.Image(
                            value=video_assets["timeline_img"],
                            label="Scene Timeline",
                            type="filepath",
                            interactive=False,
                            elem_classes=["fig"],
                        )
                    else:
                        gr.HTML(
                            notice_html(
                                "Scene timeline missing",
                                "Generate using: <code>python -m video.scenes data/video/input.mp4</code>",
                            )
                        )

                    gr.Markdown("### Optical Flow Motion Map")
                    if video_assets["motion_img"]:
                        motion_img = gr.Image(
                            value=video_assets["motion_img"],
                            label="Accumulated Motion Heatmap and Motion Curve",
                            type="filepath",
                            interactive=False,
                            elem_classes=["fig"],
                        )
                    else:
                        gr.HTML(
                            notice_html(
                                "Motion map missing",
                                "Generate using: <code>python -m video.motion data/video/input.mp4</code>",
                            )
                        )

                with gr.Column(scale=1):
                    gr.Markdown("### Object Tracking (YOLOv8 + ByteTrack)")
                    if video_assets["tracked_video"]:
                        tracked_player = gr.Video(
                            value=video_assets["tracked_video"],
                            label="Tracked Video Window (1405 s to 1410 s, H.264 Faststart)",
                            interactive=False,
                        )
                    else:
                        gr.HTML(
                            notice_html(
                                "Tracked video missing",
                                "Generate using: <code>python -m video.track data/video/input.mp4</code>",
                            )
                        )

                    gr.Markdown("### Trajectory Plot")
                    if video_assets["trajectories_img"]:
                        trajectories_img = gr.Image(
                            value=video_assets["trajectories_img"],
                            label="Pedestrian Trajectories on Static Scene Frame",
                            type="filepath",
                            interactive=False,
                            elem_classes=["fig"],
                        )
                    else:
                        gr.HTML(
                            notice_html(
                                "Trajectory plot missing",
                                "Generate using: <code>python -m video.track data/video/input.mp4</code>",
                            )
                        )

            # Key frame gallery below
            gr.Markdown("### Key Frames Gallery")
            if video_assets["gallery_imgs"]:
                kf_gallery = gr.Gallery(
                    value=video_assets["gallery_imgs"],
                    label="Extracted Key Frames",
                    columns=6,
                    object_fit="contain",
                    interactive=False,
                )
            else:
                gr.HTML(
                    notice_html(
                        "Keyframe assets missing",
                        "Generate using: <code>python -m video.keyframes data/video/input.mp4</code>",
                    )
                )

            if video_assets["sheet_img"]:
                sheet_img = gr.Image(
                    value=video_assets["sheet_img"],
                    label="Keyframes Contact Sheet (Top 12 Diverse Frames)",
                    type="filepath",
                    interactive=False,
                    elem_classes=["fig"],
                )

            # Per-scene motion maps in an accordion
            with gr.Accordion("Per-Scene Motion Maps (Top 6 Active Scenes)", open=False):
                if video_assets["scene_motion_maps"]:
                    gr.Gallery(
                        value=video_assets["scene_motion_maps"],
                        label="Per-Scene Motion Heatmaps",
                        columns=3,
                        object_fit="contain",
                        interactive=False,
                    )
                else:
                    gr.HTML(
                        notice_html(
                            "Per-scene motion maps missing",
                            "Generate using: <code>python -m video.motion data/video/input.mp4</code>",
                        )
                    )

        # ------------------------------------------------------------- Tab 4: About & Limitations
        with gr.Tab("About and limitations"):
            gr.Markdown(
                """
### System Architecture

This demonstration system implements speech and video processing techniques developed for
academic evaluation.

### Speech Processing Subsystem
* **Feature Representation:** Audio signals are processed at 16 kHz mono. An 80 Hz Butterworth
  high-pass filter (zero-phase) eliminates low-frequency room rumble and DC offsets. Framing uses
  a 25 ms window (400 samples) with a 10 ms hop (160 samples). 13 Mel-Frequency Cepstral Coefficients
  (MFCCs) are augmented with delta and delta-delta coefficients to yield 39-dimensional feature vectors.
  Per-utterance Cepstral Mean and Variance Normalisation (CMVN) mitigates channel and microphone variability.
* **Pronunciation Assessment:** Evaluates learner recordings against reference word templates using
  Dynamic Time Warping (DTW) with cosine distance. The similarity metric blends 85% acoustic alignment
  and 15% duration ratio, subtracting localized mismatch penalties when consecutive frames (>= 30 ms)
  exceed the mismatch threshold (FLAG_COST = 0.40).
* **Verse Recognition:** Each verse is modeled via a continuous density left-to-right Gaussian HMM.
  Model states are initialized with uniform time segmentation.

### Video Processing Subsystem
* **Input Video Source:** 30-minute standard-definition video (480x360 resolution, 29.97 frames per second,
  53,947 frames total). Features dynamic camera movements, panning, cuts, and pedestrian motion.
* **Scene Detection:** Utilizes PySceneDetect adaptive thresholding combined with minimum duration
  filtering (1.5 s minimum) and HSV color histogram boundary correlation (0.85 merge threshold) to prevent over-segmentation.
* **Keyframe Selection:** Evaluates candidate frames within scenes using Laplacian variance (sharpness)
  penalized for underexposure or saturation. Frames are greedily selected according to a blended
  motion-duration importance metric with diversity pruning (HSV correlation <= 0.80 and separation >= 30 s).
* **Motion Analysis:** Dense optical flow computed using the Farneback formulation at 5 samples per second.
  Global camera movement is compensated by subtracting the median flow vector across each frame pair.
* **Object Tracking:** Primary tracker combines Ultralytics YOLOv8n object detection with ByteTrack data
  association. Background subtraction (MOG2) with Hungarian centroid assignment serves as an offline fallback.
* **Video Summarization:** Ranks candidate sub-scene clips by motion density, selecting non-overlapping
  windows concatenated chronologically to yield a 60 to 90 second summary.

### Methodological Limitations
* **Speaker Dependence:** The continuous HMM verse models are strictly speaker-dependent. The trained
  vocabulary is restricted to 9 verses from the Bhagavad Gita. Reported cross-validation accuracy derives
  from recordings produced by the same speaker in controlled sessions. Recognition relies on maximum
  log-likelihood over the closed set without an explicit out-of-vocabulary rejection model.
* **Single Reference Speaker:** Pronunciation scoring measures DTW alignment against a single reference speaker
  template per word. Dialectal variations and differing vocal tract lengths can skew cosine distances.
* **Motion Compensation Limits:** Planar median flow subtraction cannot compensate for camera zoom, perspective
  rotation, or strong parallax.
* **Tracking Occlusions:** Small distant pedestrians and dense overlaps can cause ID switches or track fragmentation.
"""
            )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Speech and Video Processing Gradio Application")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve on")
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link")
    args = parser.parse_args()

    DEMO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    demo = build_app()

    gradio_major = int(gr.__version__.split(".")[0]) if hasattr(gr, "__version__") else 6
    if gradio_major >= 6:
        demo.launch(
            server_name="127.0.0.1",
            server_port=args.port,
            share=args.share,
            theme=make_theme(),
            css=CSS,
            js=FORCE_DARK_JS,
        )
    else:
        demo.launch(
            server_name="127.0.0.1",
            server_port=args.port,
            share=args.share,
        )


if __name__ == "__main__":
    main()
