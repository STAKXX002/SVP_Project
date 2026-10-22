# SVP_Project

Speech and Video Processing course project.

- **Speech:** Sanskrit pronunciation assessment (MFCC + DTW) and verse recognition (MFCC + HMM)
- **Video:** analysis of a 30-minute video: scene boundaries, key frames, motion map, object tracking, short summary

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

ffmpeg must be installed and on your PATH.

## Layout

```
speech/    features.py, dtw_score.py, hmm_train.py, app.py
video/     scenes.py, keyframes.py, motion.py, track.py, summarize.py
data/      input audio and video (not tracked, see data/README.md)
outputs/   generated figures and videos (not tracked)
report/    write-up and screenshots
```

## Speech

### Feature extraction

```bash
python -m speech.features data/reference/krishna.wav
```

Writes a waveform, spectrogram and MFCC figure to `outputs/`.

### Pronunciation scoring
_To be added._

### Verse recognition
_To be added._

## Video
_To be added._

## Results
_Screenshots will be added under `report/screenshots/`._

## Author

Kamal Kumar Behera, KIIT University
