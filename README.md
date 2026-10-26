# SVP_Project

Speech and Video Processing course project.

## Speech

- **Pronunciation assessment:** compares a learner recording of a Sanskrit word with a reference using MFCC features and dynamic time warping. Outputs a similarity percentage, flagged mismatch regions and an aligned spectrogram comparison.
- **Verse recognition:** identifies a Bhagavad Gita verse from audio using MFCC features and one HMM per verse. Speaker-dependent, small closed vocabulary.

## Video

Analysis of a 30-minute video:

- scene boundaries
- key frames
- motion map
- object tracking
- short summary video

## Structure

| Path | Contents |
|---|---|
| `speech/` | feature extraction, DTW scoring, calibration, HMM training, Gradio demo |
| `video/` | scene detection, key frames, motion map, tracking, summary |
| `report/` | project report |
| `data/`, `outputs/` | input data and generated results (not tracked) |

## Methods

| Task | Method |
|---|---|
| Speech features | 13 MFCC with delta and delta-delta, per-utterance mean and variance normalisation |
| Pronunciation | cosine-cost DTW |
| Verse recognition | left-to-right HMM per verse |
| Scenes | PySceneDetect adaptive detector |
| Motion map | Farneback optical flow |
| Tracking | YOLOv8n with ByteTrack, background subtraction as fallback |
| Summary | high-motion excerpts joined with ffmpeg |

## License

See `LICENSE`.
