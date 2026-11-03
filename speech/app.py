"""Single Gradio Blocks interface for Sanskrit speech and video processing demo.

Entry point:
    python -m speech.app
Or:
    python app.py
"""

import argparse
import glob
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
from watermark import TEXT

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

    # Create the cache if absent in untracked path
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


def run_pronunciation(reference_name, learner_audio):
    """Pronunciation assessment handler."""
    try:
        if not reference_name:
            return "--", "No reference selected", [], None, "Please select a reference word."
        if not learner_audio:
            return "--", "No audio provided", [], None, "Please record or upload audio."

        ref_path = REF_DIR / reference_name
        if not ref_path.exists():
            ref_path = DATA_DIR / reference_name
        if not ref_path.exists():
            return "--", "Reference file missing", [], None, f"Could not find {reference_name}."

        t0 = time.time()
        DEMO_OUT_DIR.mkdir(parents=True, exist_ok=True)
        result = score_pronunciation(ref_path, learner_audio, DEMO_OUT_DIR)
        elapsed = time.time() - t0

        sim = result["similarity"]
        verdict = get_verdict_band(sim)
        sim_text = f"{sim:.0f}%"

        segments = result.get("flagged_segments", [])
        if segments:
            seg_rows = [[f"{s:.2f}", f"{e:.2f}", f"{c:.3f}"] for s, e, c in segments]
        else:
            seg_rows = [["None", "None", "None"]]

        report_text = format_report(result) + f"\n  Analysis completed in {elapsed:.2f} s"
        return sim_text, verdict, seg_rows, result.get("figure"), report_text
    except Exception as exc:
        return "--", "Error", [], None, f"Analysis failed: {type(exc).__name__}: {exc}"


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

    # Annotate bar values
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
    """Verse recognition handler."""
    try:
        if not audio_path:
            return (
                "No audio provided",
                "--",
                "Please record or upload a verse recording.",
                None,
                [],
            )

        t0 = time.time()
        bundle = get_bundle()
        ranked_list, margin = predict(audio_path, bundle=bundle)
        elapsed = time.time() - t0

        if not ranked_list:
            return "No prediction", "--", "Model produced no likelihoods.", None, []

        best_verse = ranked_list[0][0]
        margin_text = f"{margin:.1f} log-lik"

        # Check for low confidence warning
        if margin < LOW_CONFIDENCE_THRESHOLD:
            warning_msg = (
                f"**Warning: Low confidence margin** ({margin:.1f} log-likelihood difference). "
                "The utterance may be noisy, truncated, or outside the trained vocabulary."
            )
        else:
            warning_msg = f"High confidence match (margin: {margin:.1f} log-lik, computed in {elapsed:.2f} s)."

        fig = plot_top3_verses(ranked_list)
        table_rows = [[v, f"{ll:.1f}"] for v, ll in ranked_list]
        return best_verse, margin_text, warning_msg, fig, table_rows
    except Exception as exc:
        return (
            "Error",
            "--",
            f"Recognition failed: {type(exc).__name__}: {exc}",
            None,
            [],
        )


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
    }

    # 1. Scene timeline image
    timelines = sorted(glob.glob("outputs/*scene_timeline*.png"))
    if timelines:
        assets["timeline_img"] = timelines[0]
    else:
        assets["timeline_msg"] = (
            "Scene timeline image missing. Generate it using: "
            "`python -m video.scenes data/video/input.mp4`"
        )

    # 2. Keyframes gallery and contact sheet
    sheets = sorted(glob.glob("outputs/*keyframes*.png"))
    if sheets:
        assets["sheet_img"] = sheets[0]

    kf_files = sorted(glob.glob("outputs/keyframes/*/*.jpg"))
    if kf_files:
        # Include top frames or curated selection
        assets["gallery_imgs"] = kf_files[:24]
    elif sheets:
        assets["gallery_msg"] = "Per-scene keyframe images missing; contact sheet displayed below."
    else:
        assets["gallery_msg"] = (
            "Keyframe assets missing. Generate them using: "
            "`python -m video.keyframes data/video/input.mp4`"
        )

    # 3. Motion map image
    motion_maps = sorted(glob.glob("outputs/*motion_map*.png"))
    if motion_maps:
        assets["motion_img"] = motion_maps[0]
    else:
        assets["motion_msg"] = (
            "Motion map missing. Generate it using: "
            "`python -m video.motion data/video/input.mp4`"
        )

    # 4. Tracked video player and trajectories
    trajectories = sorted(glob.glob("outputs/*trajectories*.png"))
    if trajectories:
        assets["trajectories_img"] = trajectories[0]

    tracked_videos = sorted(glob.glob("outputs/*tracked*.mp4"))
    if tracked_videos:
        assets["tracked_video"] = ensure_faststart_video(tracked_videos[0])
    else:
        assets["tracked_msg"] = (
            "Tracked video missing. Generate it using: "
            "`python -m video.track data/video/input.mp4`"
        )

    # 5. Summary video player
    summary_videos = sorted(glob.glob("outputs/*summary*.mp4"))
    if summary_videos:
        assets["summary_video"] = ensure_faststart_video(summary_videos[0])
    else:
        assets["summary_msg"] = (
            "Summary video missing. Generate it using: "
            "`python -m video.summarize data/video/input.mp4`"
        )

    return assets


def get_confusion_matrix_path():
    """Locate the confusion matrix image."""
    cms = sorted(glob.glob("outputs/*confusion*.png"))
    return cms[0] if cms else None


def build_app():
    """Build the multi-tab Gradio Blocks application."""
    # Warm up HMM bundle
    try:
        get_bundle()
    except Exception as exc:
        print(f"Startup model warning: {exc}")

    refs = list_references()
    default_ref = refs[0] if refs else None
    cm_path = get_confusion_matrix_path()
    video_assets = discover_video_assets()

    with gr.Blocks(title="Speech and Video Processing Project") as demo:
        gr.Markdown(
            "# Speech and Video Processing Demonstration System\n"
            "Computer Science Undergrad Coursework | Sanskrit Speech Assessment and Video Processing"
        )
        gr.Markdown(f"<small>{TEXT}</small>")

        # ------------------------------------------------------------- Tab 1: Pronunciation
        with gr.Tab("Pronunciation"):
            gr.Markdown(
                "### Pronunciation Assessment via Dynamic Time Warping (DTW)\n"
                "Compare a learner recording against a reference Sanskrit word using 39-dimensional MFCCs "
                "with cepstral mean and variance normalisation."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    ref_dropdown = gr.Dropdown(
                        choices=refs,
                        value=default_ref,
                        label="Reference Word",
                        interactive=True,
                    )
                    refresh_refs_btn = gr.Button("Refresh Reference List", size="sm")
                    learner_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="Learner Audio (Record or Upload)",
                    )
                    with gr.Row():
                        pron_demo_btn = gr.Button("Load demo example")
                        pron_score_btn = gr.Button("Score Pronunciation", variant="primary")

                with gr.Column(scale=1):
                    with gr.Row():
                        sim_label = gr.Textbox(
                            label="Similarity Score",
                            value="--",
                            interactive=False,
                            scale=1,
                        )
                        verdict_label = gr.Textbox(
                            label="Verdict Band",
                            value="--",
                            interactive=False,
                            scale=1,
                        )
                    segments_table = gr.Dataframe(
                        headers=["Start (s)", "End (s)", "Mean Cost"],
                        label="Flagged Mismatch Segments (>= 30 ms above cost 0.40)",
                        interactive=False,
                    )
                    pron_report = gr.Textbox(
                        label="Detailed Metrics",
                        lines=6,
                        interactive=False,
                    )

            dtw_figure = gr.Image(
                label="Spectrogram Alignment, DTW Path, and Per-Frame Cost",
                type="filepath",
                interactive=False,
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
                outputs=[sim_label, verdict_label, segments_table, dtw_figure, pron_report],
            )

        # ------------------------------------------------------------- Tab 2: Verse Recognition
        with gr.Tab("Verse recognition"):
            gr.Markdown(
                "### Bhagavad Gita Verse Recognition via Left-to-Right GMM-HMM\n"
                "Classifies an audio recording into one of 9 trained Gita verses using sequence log-likelihood."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    verse_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="Verse Audio (Record or Upload)",
                    )
                    with gr.Row():
                        verse_demo_btn = gr.Button("Load demo example")
                        verse_rec_btn = gr.Button("Recognise Verse", variant="primary")

                    with gr.Row():
                        pred_box = gr.Textbox(
                            label="Predicted Verse",
                            interactive=False,
                            scale=1,
                        )
                        margin_box = gr.Textbox(
                            label="Confidence Margin (top 1 minus top 2)",
                            interactive=False,
                            scale=1,
                        )
                    confidence_msg = gr.Markdown("Ready for evaluation.")
                    verse_table = gr.Dataframe(
                        headers=["Verse", "Log-Likelihood"],
                        label="All Candidate Verses (Ranked)",
                        interactive=False,
                    )

                with gr.Column(scale=1):
                    top3_chart = gr.Plot(label="Top-3 Verses with Log-Likelihoods")
                    cm_image = gr.Image(
                        value=cm_path,
                        label="Held-Out Confusion Matrix (4-Fold Cross-Validation)",
                        type="filepath",
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
                outputs=[pred_box, margin_box, confidence_msg, top3_chart, verse_table],
            )

        # ------------------------------------------------------------- Tab 3: Video Analysis
        with gr.Tab("Video analysis"):
            gr.Markdown(
                "### Cached Video Analysis Pipeline\n"
                "Displays precomputed scene segmentation, keyframes, dense optical flow, "
                "object tracking, and automated summaries. No live 30-minute recompute is executed."
            )
            video_refresh_btn = gr.Button("Load demo example / Refresh Assets")

            # Timeline
            gr.Markdown("#### 1. Scene Boundary Timeline")
            timeline_alert = gr.Markdown(
                video_assets["timeline_msg"] if video_assets["timeline_msg"] else ""
            )
            timeline_img = gr.Image(
                value=video_assets["timeline_img"],
                label="Scene Timeline",
                type="filepath",
                interactive=False,
            )

            # Keyframes
            gr.Markdown("#### 2. Extracted Keyframes")
            kf_alert = gr.Markdown(
                video_assets["gallery_msg"] if video_assets["gallery_msg"] else ""
            )
            with gr.Row():
                sheet_img = gr.Image(
                    value=video_assets["sheet_img"],
                    label="Keyframes Contact Sheet",
                    type="filepath",
                    interactive=False,
                    scale=1,
                )
                kf_gallery = gr.Gallery(
                    value=video_assets["gallery_imgs"],
                    label="Per-Scene Selected Keyframes",
                    interactive=False,
                    scale=1,
                )

            # Motion Map
            gr.Markdown("#### 3. Optical Flow Motion Analysis")
            motion_alert = gr.Markdown(
                video_assets["motion_msg"] if video_assets["motion_msg"] else ""
            )
            motion_img = gr.Image(
                value=video_assets["motion_img"],
                label="Accumulated Motion Heatmap and Motion Curve",
                type="filepath",
                interactive=False,
            )

            # Tracking and Summary Videos
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 4. Object Tracking (YOLOv8 + ByteTrack)")
                    tracked_alert = gr.Markdown(
                        video_assets["tracked_msg"] if video_assets["tracked_msg"] else ""
                    )
                    tracked_video = gr.Video(
                        value=video_assets["tracked_video"],
                        label="Tracked Video Window (H.264 Faststart)",
                        interactive=False,
                    )
                    trajectories_img = gr.Image(
                        value=video_assets["trajectories_img"],
                        label="Trajectory Plot",
                        type="filepath",
                        interactive=False,
                    )

                with gr.Column(scale=1):
                    gr.Markdown("#### 5. Motion-Based Video Summary (60 to 90 s)")
                    summary_alert = gr.Markdown(
                        video_assets["summary_msg"] if video_assets["summary_msg"] else ""
                    )
                    summary_video = gr.Video(
                        value=video_assets["summary_video"],
                        label="Stitched Summary Video (H.264 Faststart)",
                        interactive=False,
                    )

            def refresh_video_ui():
                assets = discover_video_assets()
                return (
                    assets["timeline_msg"],
                    assets["timeline_img"],
                    assets["gallery_msg"],
                    assets["sheet_img"],
                    assets["gallery_imgs"],
                    assets["motion_msg"],
                    assets["motion_img"],
                    assets["tracked_msg"],
                    assets["tracked_video"],
                    assets["trajectories_img"],
                    assets["summary_msg"],
                    assets["summary_video"],
                )

            video_refresh_btn.click(
                fn=refresh_video_ui,
                outputs=[
                    timeline_alert,
                    timeline_img,
                    kf_alert,
                    sheet_img,
                    kf_gallery,
                    motion_alert,
                    motion_img,
                    tracked_alert,
                    tracked_video,
                    trajectories_img,
                    summary_alert,
                    summary_video,
                ],
            )

        # ------------------------------------------------------------- Tab 4: About & Limitations
        with gr.Tab("About and limitations"):
            gr.Markdown(
                """
### System Architecture and Methodological Limitations

This demonstration system implements speech and video processing techniques developed for
academic evaluation.

#### 1. Speech Processing Subsystem
* **Feature Representation:** Audio signals are processed at 16 kHz mono. An 80 Hz Butterworth
  high-pass filter (zero-phase) eliminates low-frequency room rumble and DC offsets. Framing uses
  a 25 ms window (400 samples) with a 10 ms hop (160 samples). 13 Mel-Frequency Cepstral Coefficients
  (MFCCs) are augmented with delta and delta-delta coefficients to yield 39-dimensional feature vectors.
  Per-utterance Cepstral Mean and Variance Normalisation (CMVN) mitigates channel and microphone variability.
* **Pronunciation Assessment:** Evaluates learner recordings against reference word templates using
  Dynamic Time Warping (DTW) with cosine distance. The similarity metric blends 85% acoustic alignment
  and 15% duration ratio, subtracting localized mismatch penalties when consecutive frames (>= 30 ms)
  exceed the mismatch threshold (FLAG_COST = 0.40).
* **Limitations of Pronunciation Scoring:** Uses a single reference speaker per target word. Dialectal
  variations, accent nuances, and differing vocal tract lengths can skew cosine distances. DTW does not
  explicitly model phonemic boundaries or subtle phonetic substitutions.
* **Verse Recognition:** Each verse is modeled via a continuous density left-to-right Gaussian HMM.
  Model states are initialized with uniform time segmentation.
* **Limitations of Verse Recognition:** Models are strictly speaker-dependent. The trained vocabulary
  is restricted to 9 verses from the Bhagavad Gita. Reported cross-validation accuracy derives from
  recordings produced by the same speaker in controlled sessions. Recognition relies on maximum
  log-likelihood over the closed set without an explicit out-of-vocabulary rejection model.

#### 2. Video Processing Subsystem
* **Input Video Source:** 30-minute standard-definition video (480x360 resolution, 29.97 frames per second,
  53,947 frames total). Features dynamic camera movements, panning, cuts, and pedestrian motion.
* **Scene Detection:** Utilizes PySceneDetect adaptive thresholding combined with minimum duration
  filtering and HSV color histogram boundary correlation to prevent over-segmentation.
* **Keyframe Selection:** Evaluates candidate frames within scenes using Laplacian variance (sharpness)
  penalized for underexposure or saturation. Frames are greedily selected according to a blended
  motion-duration importance metric with diversity pruning.
* **Motion Analysis:** Dense optical flow computed using the Farneback formulation at 5 samples per second.
  Global camera movement is compensated by subtracting the median flow vector across each frame pair.
* **Object Tracking:** Primary tracker combines Ultralytics YOLOv8n object detection with ByteTrack data
  association. Background subtraction (MOG2) with Hungarian centroid assignment serves as an offline fallback.
* **Video Summarization:** Ranks candidate sub-scene clips by motion density, selecting non-overlapping
  windows concatenated chronologically to yield a 60 to 90 second summary.
* **Limitations of Video Processing:** Heavy camera panning and zooming challenge simple planar global
  motion compensation. Small distant objects and crowded scenes may cause tracking ID switches or track fragmentation.
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
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
