# Speech and Video Processing: Project Report

Kamal Kumar Behera, KIIT University

## 1. Speech

### 1.1 Feature extraction
Describe: 16 kHz mono, 25 ms window, 10 ms hop, 13 MFCC + delta + delta-delta, CMVN.
Figure: `outputs/<word>_features.png` (waveform, log-mel spectrogram, MFCC).

### 1.2 Pronunciation error detection (MFCC + DTW)
Method: cosine-cost DTW, similarity formula, flagged regions.
Calibration table (fill from `python -m speech.calibrate`):

| Recording | Group | Avg DTW cost | Similarity % |
|---|---|---|---|
| krishna_2 | same word | | |
| kishna | mispronunciation | | |
| krisna | mispronunciation | | |
| rama | different word | | |

Figures: `outputs/calibration.png`, one `*_dtw.png` for a correct and one for a mispronounced take.
Discussion: which errors were flagged, which were missed, effect of the duration term.

### 1.3 Verse recognition (MFCC + HMM)
Method: one left-to-right HMM per verse, flat-start initialisation, maximum log-likelihood decision.
Setup: number of verses, takes per verse, states per model.
Result: held-out accuracy and confusion matrix (`outputs/verse_confusion.png`).
Scope statement: speaker-dependent, closed vocabulary of N verses, no reject class.

## 2. Video

For each stage: method, parameters, figure, and one paragraph of observations.

| Stage | Figure |
|---|---|
| Scene boundaries | `outputs/<video>_scene_timeline.png` |
| Key frames | `outputs/<video>_keyframes.png` |
| Motion map | `outputs/<video>_motion_map.png` |
| Tracking | `outputs/<video>_trajectories.png`, `outputs/<video>_tracked.mp4` |
| Summary | `outputs/<video>_summary.mp4`, `outputs/cache/<video>/summary.json` |

Report the video length, number of scenes (natural and forced), tracking window, backend used and
the summary length.

## 3. Limitations
See the Limitations section of the README.

## 4. Reproduction
Commands are listed in the README.
