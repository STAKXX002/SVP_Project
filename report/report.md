# Speech and Video Processing: Coursework Report Outline

Author: Kamal Kumar Behera  
Institution: KIIT University  
Topic: Sanskrit Speech Assessment and Dense Video Processing System  

---

## 1. Speech Processing Subsystem

### 1.1 Acoustic Feature Representation
- **Acoustic Parameters:** 16 kHz sampling rate, single-channel (mono) pulse-code modulation. Framing uses a 25 ms analysis window (400 samples) at a 10 ms hop interval (160 samples).
- **Pre-filtering and Normalization:** Audio signals are pre-filtered using a 5th-order Butterworth high-pass filter with an 80 Hz cutoff frequency implemented via zero-phase forward-backward filtering (`sosfiltfilt`). Silence is trimmed with a 30 dB threshold. Amplitude normalization divides by the 99.5th percentile absolute amplitude to prevent click artifacts from dominating the dynamic range.
- **Spectral Descriptors:** 13 Mel-Frequency Cepstral Coefficients (MFCCs) augmented by first-order (delta) and second-order (delta-delta) temporal derivatives yield a 39-dimensional feature representation per frame. Per-utterance Cepstral Mean and Variance Normalisation (CMVN) minimizes static convolutional channel distortion.
- **Diagnostic Artifact:** `outputs/krishna_features.png` (Item 1.1: Waveform, 64-band log-mel spectrogram, and MFCC heatmap).

---

### 1.2 Pronunciation Error Scoring and DTW Alignment
- **Formulation:** Frame sequences are aligned using Dynamic Time Warping (DTW) with cosine distance over 39-dimensional CMVN feature vectors.
- **Similarity Metric:** Overall pronunciation score balances acoustic alignment and utterance duration, penalized by continuous mismatch regions:
  $$\text{Similarity} = \text{clip}\left(0.85 \times S_{\text{acoustic}} + 0.15 \times S_{\text{duration}} - P_{\text{flag}}, 0, 100\right)$$
  where $S_{\text{acoustic}}$ maps local DTW path cost linearly between `COST_BEST = 0.143` (100%) and `COST_ZERO = 0.710` (0%).
- **Mismatch Segmentation:** Consecutive frames spanning at least 30 ms (3 frames) with local cost exceeding `FLAG_COST = 0.40` are flagged as pronunciation errors. Each flagged frame percent incurs a 1.5 point penalty (capped at 40 points).
- **Calibration Empirical Data:**

| Recording Token | Category | Average DTW Cost | Acoustic Score | Overall Similarity | Verdict Band |
|---|---|---|---|---|---|
| `krishna_1` | Same word (take 1) | 0.146 | 99.4% | 98.1% | Excellent |
| `krishna_2` | Same word (take 2) | 0.172 | 96.6% | 94.1% | Excellent |
| `krishna_3` | Same word (take 3) | 0.139 | 100.0% | 98.5% | Excellent |
| `kishna` | Phonetic deletion | 0.451 | 45.7% | 12.4% | Needs work (Acoustic 46%) |
| `krisna` | Retroflex substitution | 0.446 | 46.6% | 14.6% | Needs work (Acoustic 47%) |
| `rama_1` | Negative control | 0.736 | 0.0% | 0.0% | Different word |
| `shiva_1` | Negative control | 0.684 | 4.6% | 0.0% | Different word |

- **Diagnostic Artifacts:**
  - `outputs/calibration.png` (Item 1.2: Cost separation bar plot across categories).
  - `outputs/krishna_vs_krishna_2_dtw.png` (Item 1.2: Spectrogram alignment and DTW path for correct take).
  - `outputs/krishna_vs_kishna_dtw.png` (Item 1.2: Local cost spike and highlighted mismatch interval for deletion take).

---

### 1.3 Closed-Vocabulary Verse Recognition
- **Architecture:** One continuous left-to-right Gaussian Hidden Markov Model (HMM) per verse. State topology allows self-transitions, sequential transitions to the immediate next state, and single-state skips. Model state counts are dynamically assigned based on median utterance duration (approximately 200 ms per state, bounded between 5 and 40 states).
- **Initialization and Estimation:** Models are initialized using time-aligned uniform segmentation (flat start). Diagonal covariance matrices are estimated via Baum-Welch Expectation-Maximization.
- **Decision Rule:** Maximum log-likelihood selection over 9 Bhagavad Gita verses:
  $$\hat{v} = \arg\max_{v \in V} \log P(O \mid \lambda_v)$$
- **Evaluation Performance:** 4-fold cross-validation across 72 recordings achieves 100% held-out classification accuracy with zero diagonal leakage.
- **Diagnostic Artifact:** `outputs/verse_confusion.png` (Item 1.3: 9x9 cross-validation confusion matrix).

---

## 2. Video Processing Subsystem

### 2.1 Scene Boundary Detection and Post-Processing
- **Algorithm:** Adaptive luminance and color thresholding using PySceneDetect.
- **Refinement Filters:**
  1. Short-scene elimination: Scenes shorter than 1.5 s are merged into the adjacent neighbor exhibiting the highest HSV histogram correlation.
  2. Boundary continuity merge: Adjacent scene boundary frames evaluated within 50 ms of transition are merged if HSV correlation exceeds 0.85.
  3. Forced length bounding: Scenes exceeding 60.0 s are segmented into uniform sub-scenes.
- **Dataset Execution Results:** 30.0-minute input video (53,947 frames at 29.97 fps) filtered from 285 raw cuts to 192 coherent scenes (shortest scene: 1.535 s).
- **Diagnostic Artifact:** `outputs/input_scene_timeline.png` (Item 2.1: Full-video scene timeline bar).

---

### 2.2 Keyframe Selection with Diversity Constraints
- **Objective:** Selection of informative, non-redundant visual frames summarizing scene content.
- **Method:** Frames within the central 80% of each scene are evaluated for Laplacian variance (sharpness), penalized when underexposed (mean < 25) or saturated (mean > 235). Scene importance blends normalized motion (0.6) and duration (0.4).
- **Diversity Filtering:** Candidates are selected greedily in descending importance order; candidates are rejected if temporal separation from an already selected frame is under 30.0 s or if 2D HSV histogram correlation exceeds 0.80.
- **Diagnostic Artifact:** `outputs/input_keyframes.png` (Item 2.2: 12-frame diverse contact sheet grid).

---

### 2.3 Optical Flow Motion Analysis
- **Algorithm:** Dense optical flow computed using the Farneback formulation sampled at 5 fps at 320 px analysis width.
- **Global Motion Compensation:** For each frame pair, the median horizontal and vertical displacement vector is subtracted to eliminate uniform camera panning.
- **Visualization:** Accumulated flow magnitude is cropped of letterbox borders, normalized across the 2nd to 98th percentile range, and overlaid onto the background/mean frame using the inferno colormap at 0.6 opacity.
- **Diagnostic Artifacts:**
  - `outputs/input_motion_map.png` (Item 2.3: Global accumulated motion heatmap and motion curve with thinned top tick markers).
  - `outputs/input_motion_scene_*.png` (Item 2.3: Individual motion maps for the top 6 active scenes: 075, 152, 035, 101, 077, 036).

---

### 2.4 Multi-Object Tracking
- **Framework:** Ultralytics YOLOv8n object detection combined with ByteTrack data association.
- **Tracking Constraints:** Operating confidence threshold 0.20, inference image size 640. Detection is restricted to the person class (`classes=[0]`).
- **Scene Continuity Handling:** The tracker is reset at every scene boundary to prevent spurious track persistence across camera cuts. Tracks with duration under 1.0 s (fewer than 5 observations at 5 fps) are eliminated.
- **Demo Window (1405 s to 1410 s):** Tracks a pedestrian walking across Scene 152 over 25 consecutive frames (4.8 s active duration, 375 px cumulative path). Trajectories are rendered onto a single static frame from the same scene.
- **Diagnostic Artifacts:**
  - `outputs/input_tracked.mp4` (Item 2.4: Annotated H.264 video with bounding boxes, IDs, and trails).
  - `outputs/input_trajectories.png` (Item 2.4: Color-coded trajectory vector overlay).

---

### 2.5 Automated Video Summarization
- **Selection Policy:** Identifies the highest-motion scenes from optical flow scores. Extracts a 3.0 to 5.0 s continuous window exhibiting maximum motion density per selected scene.
- **Assembly:** Selected excerpts are sequenced chronologically and concatenated using FFmpeg with video re-encoding (`scale=-2:min(ih,720)`, H.264, yuv420p, faststart) and audio resampling.
- **Summary Execution Metrics:** Compiles 18 scene excerpts into an authorized 76.2 s summary video (bounded within the 60 to 90 s specification).
- **Diagnostic Artifact:** `outputs/input_summary.mp4` (Item 2.5: Video summary player asset).

---

## 3. Methodological Limitations

1. **Speaker Dependence:** The continuous HMM verse models are strictly speaker-dependent. The models do not generalize across distinct speakers without acoustic adaptation.
2. **Small Closed Vocabulary:** The system models 9 discrete verses without an out-of-vocabulary garbage model.
3. **Template-Based Pronunciation Reference:** Scoring uses a single reference template; accent and vocal tract differences introduce variance.
4. **Planar Motion Assumption:** Median flow subtraction cannot compensate for non-planar camera motion, zoom, or perspective parallax.
5. **Detection Occlusion:** ByteTrack relies on bounding box IoU; overlapping pedestrians or static obstructions can cause track fragmentation.

---

## 4. Screenshot and Deliverable Checklist

| Deliverable Artifact | Coursework Specification Item | Validation Status |
|---|---|---|
| `outputs/krishna_features.png` | Item 1.1 Speech Features | Verified and generated. |
| `outputs/krishna_vs_krishna_2_dtw.png` | Item 1.2 Pronunciation Assessment | Verified and generated. |
| `outputs/krishna_vs_kishna_dtw.png` | Item 1.2 Error Flagging | Verified and generated. |
| `outputs/calibration.png` | 1.2 DTW Calibration | Verified and generated. |
| `outputs/verse_confusion.png` | 1.3 Verse Recognition | Verified (100% accuracy). |
| `outputs/input_scene_timeline.png` | 2.1 Scene Detection | Verified (192 scenes). |
| `outputs/input_keyframes.png` | 2.2 Keyframe Selection | Verified (12 diverse frames). |
| `outputs/input_motion_map.png` | 2.3 Motion Analysis | Verified (inferno overlay). |
| `outputs/input_motion_scene_*.png` | 2.3 Per-Scene Motion Maps | Verified (top 6 scenes). |
| `outputs/input_tracked.mp4` | 2.4 Object Tracking | Verified (H.264 faststart). |
| `outputs/input_trajectories.png` | 2.4 Trajectory Overlay | Verified (single scene frame). |
| `outputs/input_summary.mp4` | 2.5 Video Summarization | Verified (76.2 s duration). |
| Gradio Demo (`speech.app`) | 3.0 Live Demonstration | Verified (all 4 tabs functional). |
