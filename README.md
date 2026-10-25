# SVP_Project

Speech and Video Processing course project.

- **Speech:** Sanskrit pronunciation scoring (MFCC + DTW) and verse recognition (MFCC + per-verse HMM)
- **Video:** analysis of a 30-minute video: scene boundaries, key frames, motion map, object tracking, short summary

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Video tracking with YOLO needs the CPU build of torch, installed before `requirements-video.txt`:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-video.txt
```

`ffmpeg` and `ffprobe` must be on the PATH. Python 3.12 is the fallback if a dependency has no wheel for 3.14.

## Layout

```
speech/    features.py, dtw_score.py, calibrate.py, hmm_train.py, app.py
video/     common.py, scenes.py, keyframes.py, motion.py, track.py, summarize.py
data/      input audio, video and trained models (not tracked, see data/README.md)
outputs/   generated figures, videos and cached intermediate results (not tracked)
report/    write-up and screenshots
```

## Speech

### Feature extraction

```bash
python -m speech.features data/reference/krishna.wav
```

16 kHz mono, 25 ms window, 10 ms hop, 13 MFCC + delta + delta-delta (39 dims), per-utterance CMVN.
Writes a waveform, log-mel spectrogram and MFCC figure to `outputs/`.

### Pronunciation scoring

```bash
python -m speech.dtw_score data/reference/krishna.wav data/learner/kishna.wav
```

Cosine-cost DTW between the reference and learner MFCC sequences.

```
similarity = 0.85 x acoustic + 0.15 x duration - flagged-region penalty
```

- acoustic: average DTW path cost mapped linearly, `COST_BEST` to 100 and `COST_ZERO` to 0
- duration: 100 for equal length, 0 when one recording is twice as long as the other
- flagged regions: runs of at least 50 ms where the smoothed per-frame cost exceeds `FLAG_COST`;
  each percent of the reference that is flagged removes 1.5 points

Output: similarity, flagged time ranges and a four-panel figure (reference and learner spectrograms,
DTW path over the cost matrix, per-frame cost with flagged regions shaded).

Calibration on your own recordings (same word repeated, mispronunciations, a different word):

```bash
python -m speech.calibrate --ref data/reference/krishna.wav \
    --same data/learner/krishna_2.wav data/learner/krishna_3.wav \
    --bad data/learner/kishna.wav data/learner/krisna.wav \
    --neg data/learner/rama.wav data/learner/shiva.wav --write
```

`--write` stores the constants in `speech/calibration.json`, which `dtw_score` loads automatically.

### Verse recognition

Training data: `data/verses/<verse_id>/*.wav`, several takes per verse.

```bash
python -m speech.hmm_train train                 # one left-to-right HMM per verse -> data/models/verses.pkl
python -m speech.hmm_train eval --folds 4        # held-out accuracy and confusion matrix
python -m speech.hmm_train predict clip.wav      # ranked verses for one recording
```

Each state is a diagonal Gaussian (`--mix 2` enables Gaussian mixtures). The model is initialised
by uniform segmentation. The number of states follows the median recording length.
The system is speaker-dependent, uses a closed vocabulary and has no reject class.

### Demo app

```bash
python -m speech.app
```

Tab 1 scores a recording against a reference word. Tab 2 recognises the verse.

## Video

Every stage caches its result in `outputs/cache/<video>/`, so repeated runs and the live demo are fast.

```bash
python -m video.scenes     data/video/input.mp4                        # scene boundaries, timeline figure
python -m video.motion     data/video/input.mp4                        # optical-flow motion map
python -m video.keyframes  data/video/input.mp4                        # key frame per scene, contact sheet
python -m video.track      data/video/input.mp4 --start 600 --duration 120
python -m video.summarize  data/video/input.mp4                        # 60 to 90 s summary
```

| Stage | Method |
|---|---|
| Scenes | PySceneDetect `AdaptiveDetector`, or `--method hist` for HSV histogram distance; scenes over 60 s are split and marked forced |
| Key frames | sharpest well-exposed frame from 9 samples per scene; importance = 0.6 motion + 0.4 duration |
| Motion map | Farneback optical flow at 5 fps and 320 px width, median-flow camera compensation, accumulated magnitude |
| Tracking | YOLOv8n + ByteTrack on a time window; MOG2 background subtraction with Hungarian centroid matching as fallback |
| Summary | highest-motion windows per scene, cut and re-encoded with ffmpeg, concatenated |

## Results

Screenshots are stored under `report/screenshots/`.

## Limitations

- Pronunciation scoring is speaker-dependent: the reference and learner should be recorded on the same
  microphone. A word-level percentage is insensitive to a single short wrong sound, so the flagged
  regions carry most of the diagnostic value.
- Verse recognition is trained on one speaker and a handful of verses. It does not generalise to other
  speakers and always returns the closest trained verse.
- Optical flow after median compensation still contains zoom and parallax; scene-cut samples are suppressed.
- Tracking is run on a window of the video, not the full 30 minutes. The background-subtraction
  fallback requires a static camera and tracks any moving blob.

## Author

Kamal Kumar Behera, KIIT University
