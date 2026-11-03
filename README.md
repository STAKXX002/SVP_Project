# Speech and Video Processing: Coursework System

Technical implementation and demonstration system for Speech and Video Processing (CS undergraduate course).

## System Architecture

The repository comprises two primary processing pipelines integrated under a unified Gradio demonstration interface:

1. **Speech Processing Subsystem:**
   - Feature extraction using 13 Mel-Frequency Cepstral Coefficients (MFCCs) augmented with delta and delta-delta features (39 dimensions total) and per-utterance Cepstral Mean and Variance Normalisation (CMVN).
   - Pronunciation error scoring using Dynamic Time Warping (DTW) with cosine distance, blended acoustic and duration penalties, and localized mismatch segment identification.
   - Continuous density left-to-right Gaussian Hidden Markov Models (HMMs) for closed-vocabulary Bhagavad Gita verse classification.

2. **Video Processing Subsystem:**
   - Scene cut detection via PySceneDetect adaptive thresholding, duration-constrained merging (minimum scene length 1.5 s), and boundary-frame HSV histogram correlation merging.
   - Keyframe selection combining Laplacian variance sharpness with greedy diversity filtering (HSV histogram correlation <= 0.80 and temporal separation >= 30 s).
   - Dense optical flow motion analysis (Farneback formulation) with planar global motion compensation (median flow vector subtraction), 2nd to 98th percentile normalization, and inferno heatmap overlays.
   - Multi-object tracking utilizing Ultralytics YOLOv8n object detection combined with ByteTrack association, evaluated per scene with boundary resets and tracks filtered to duration >= 1.0 s.
   - Automated video summarization selecting 3 to 5 second windows from highest-motion scenes concatenated chronologically to produce a 60 to 90 second summary clip.

3. **Demonstration Interface:**
   - Gradio Blocks application featuring four functional tabs: Pronunciation, Verse recognition, Video analysis, and About and limitations.

---

## Directory Structure

| Path | Description |
|---|---|
| `speech/features.py` | 16 kHz audio loading, Butterworth 80 Hz high-pass filtering, MFCC extraction, diagnostic plotting. |
| `speech/dtw_score.py` | Cosine-cost DTW pronunciation assessment, segment flagging, 4-panel comparison plotting. |
| `speech/calibrate.py` | Empirical calibration of DTW cost boundaries (`COST_BEST`, `COST_ZERO`, `FLAG_COST`). |
| `speech/hmm_train.py` | GMM-HMM training, evaluation, cross-validation, confusion matrix generation, inference. |
| `speech/app.py` | Gradio Blocks multi-tab demonstration interface. |
| `app.py` | Root entry point importing `speech.app`. |
| `video/common.py` | Shared utility routines for video probing, frame iteration, and ffmpeg execution. |
| `video/scenes.py` | Scene cut detection, 1.5 s minimum duration merging, histogram correlation boundary merging, timeline plotting. |
| `video/keyframes.py` | Sharpness evaluation, greedy diversity-constrained keyframe extraction, 12-frame contact sheet generation. |
| `video/motion.py` | Farneback optical flow computation, global motion compensation, accumulated and per-scene motion mapping. |
| `video/track.py` | YOLOv8n and ByteTrack multi-object tracking, per-scene tracker resets, trajectory mapping. |
| `video/summarize.py` | Motion-based candidate excerpt extraction and ffmpeg concatenation. |
| `watermark.py` | Standardized attribution marking applied to visual outputs. |
| `data/` | Audio recordings (references, learner takes, verse datasets) and input video source. |
| `outputs/` | Generated figures, annotated videos, and cached metrics. |
| `report/` | Coursework report outline and technical mapping. |

---

## Installation

### Prerequisites

- Linux environment (x86_64)
- Python 3.12 or 3.14 (virtual environment `.venv`)
- FFmpeg and FFprobe installed on system PATH (`/usr/bin/ffmpeg`)
- CPU-only execution (no CUDA GPU required)

### Setup Commands

```bash
# 1. Activate existing virtual environment or create a new one
source .venv/bin/activate

# 2. Install CPU-only PyTorch and Torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Install video tracking dependencies
pip install -r requirements-video.txt
```

---

## Running the Application

Launch the unified Gradio demonstration interface:

```bash
python -m speech.app
```

Alternatively:

```bash
python app.py
```

The application serves on `http://127.0.0.1:7860/` by default. Optional arguments:
- `--port PORT`: Specify custom HTTP port (e.g., `--port 7860`).
- `--share`: Generate a public Gradio share link.

---

## Figure and Pipeline Reproduction

### 1. Speech Feature Extraction and Diagnostics

```bash
python -m speech.features data/reference/krishna.wav --out outputs
```
Outputs: `outputs/krishna_features.png`.

### 2. Pronunciation Scoring and Alignment

```bash
python -m speech.dtw_score data/reference/krishna.wav data/learner/krishna_1.wav --out outputs
```
Outputs: `outputs/krishna_vs_krishna_1_dtw.png` and terminal score breakdown.

### 3. DTW Cost Calibration

```bash
python -m speech.calibrate --ref data/reference/krishna.wav \
    --same data/learner/krishna_1.wav data/learner/krishna_2.wav data/learner/krishna_3.wav \
    --bad data/learner/kishna.wav data/learner/krisna.wav \
    --neg data/learner/rama_1.wav data/learner/shiva_1.wav
```
Outputs: `outputs/calibration.png` and terminal calibration table.

### 4. HMM Verse Model Training and Cross-Validation

```bash
# Train models and save cache
python -m speech.hmm_train train --data data/verses --model data/models/verses.pkl

# Run 4-fold cross-validation and generate confusion matrix
python -m speech.hmm_train eval --data data/verses --folds 4 --out outputs

# Run single-file verse inference
python -m speech.hmm_train predict data/test_05.wav
```
Outputs: `data/models/verses.pkl`, `outputs/verse_confusion.png`.

### 5. Video Scene Boundary Detection

```bash
python -m video.scenes data/video/input.mp4
```
Outputs: `outputs/input_scene_timeline.png`, `outputs/cache/input/scenes.json`.

### 6. Keyframe Extraction

```bash
python -m video.keyframes data/video/input.mp4 --top 12
```
Outputs: `outputs/input_keyframes.png`, `outputs/keyframes/input/scene_*.jpg`.

### 7. Optical Flow Motion Analysis

```bash
python -m video.motion data/video/input.mp4
```
Outputs: `outputs/input_motion_map.png`, `outputs/input_motion_scene_*.png` (top 6 scenes), `outputs/cache/input/heat.npy`.

### 8. Object Tracking

```bash
# Track window 1405 s to 1410 s (walking pedestrian)
python -m video.track data/video/input.mp4 --start 1405 --duration 5 --backend yolo
```
Outputs: `outputs/input_tracked.mp4` (H.264 faststart), `outputs/input_trajectories.png`.

### 9. Motion-Based Video Summary

```bash
python -m video.summarize data/video/input.mp4 --target 75 --clip 4.5
```
Outputs: `outputs/input_summary.mp4` (60 to 90 s, H.264 faststart).

---

## Methodological Limitations

1. **Speaker Dependence:** The continuous HMM verse models are strictly speaker-dependent. Features capture individual vocal tract characteristics. Evaluated held-out accuracy figures reflect same-speaker recordings acquired across multiple takes.
2. **Closed Vocabulary:** Recognition is restricted to 9 pre-defined verses from the Bhagavad Gita without an out-of-vocabulary garbage state; out-of-domain recordings are assigned to the closest closed-set candidate.
3. **Single Reference Pronunciation:** Pronunciation scoring measures DTW alignment against a single reference speaker template. Phonetic variation, gender pitch offsets, and dialectal differences may induce false mismatch flags.
4. **Linear Acoustic Score Metric:** Acoustic score mapping assumes linear normalization between empirical anchor points (`COST_BEST=0.143`, `COST_ZERO=0.710`). Nonlinear acoustic differences are not explicitly modeled.
5. **Planar Motion Compensation:** Subtracting the median Farneback flow vector compensates for pure camera panning, but cannot compensate for camera zoom, perspective rotation, or depth parallax.
6. **Pedestrian Tracking Limitations:** ByteTrack with YOLOv8n operates on single-class person detections (`classes=[0]`). Severe occlusions, long-term disappearances, or camera cuts beyond the scene window break track continuity.

---

## Screenshot and Output Checklist

The table below maps each generated artifact to the corresponding coursework specification item:

| Output File | Assignment Item Satisfied | Description |
|---|---|---|
| `outputs/krishna_features.png` | 1.1 Speech Features | 3-panel display: waveform, log-mel spectrogram, and 13 MFCC coefficients. |
| `outputs/krishna_vs_krishna_2_dtw.png` | 1.2 Pronunciation Assessment | 4-panel comparison: reference/learner spectrograms, DTW path, and per-frame cost. |
| `outputs/krishna_vs_kishna_dtw.png` | 1.2 Pronunciation Error Flagging | 4-panel comparison showing localized error shading on mispronounced token. |
| `outputs/calibration.png` | 1.2 DTW Calibration | Bar chart validating same-word (100%), mispronounced (47%), and negative controls (0%). |
| `outputs/verse_confusion.png` | 1.3 Verse Recognition | 9x9 confusion matrix from 4-fold cross-validation (100% held-out accuracy). |
| `outputs/input_scene_timeline.png` | 2.1 Scene Detection | Timeline bar showing 192 post-processed scenes with forced splits hatched. |
| `outputs/input_keyframes.png` | 2.2 Keyframe Selection | 12-frame contact sheet grid enforcing diversity (corr <= 0.8, sep >= 30 s). |
| `outputs/input_motion_map.png` | 2.3 Motion Analysis | Letterbox-cropped inferno heatmap overlay on mean frame plus motion-over-time plot. |
| `outputs/input_motion_scene_*.png` | 2.3 Per-Scene Motion Maps | Individual motion maps for the top 6 highest-motion scenes. |
| `outputs/input_tracked.mp4` | 2.4 Multi-Object Tracking | Annotated H.264 video with bounding boxes, tracking IDs, and color trails (1405 s to 1410 s). |
| `outputs/input_trajectories.png` | 2.4 Trajectory Mapping | Trajectory paths drawn onto a single static frame from the tracked scene. |
| `outputs/input_summary.mp4` | 2.5 Video Summarization | 76.2-second concatenated summary compiled from highest-motion scenes. |
| Gradio App (`http://127.0.0.1:7860`) | 3.0 Integrated Demonstration | Functional UI with Pronunciation, Verse, Video, and About tabs. |

---

## License

See `LICENSE`.
