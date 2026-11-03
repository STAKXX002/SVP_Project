"""MFCC feature extraction and diagnostic plots for the speech processing pipeline."""

import argparse
import subprocess
import tempfile
from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt
from watermark import save_figure

SR = 16000
N_MFCC = 13
N_FFT = 512
HOP = 160  # 10 ms at 16 kHz
WIN = 400  # 25 ms at 16 kHz


def load_audio(path, sr=SR, trim=True, top_db=30):
    """Load mono audio at a fixed sample rate.

    Applies an 80 Hz Butterworth high-pass filter (zero-phase) before trimming,
    trims trailing silence, and normalises by the 99.5th percentile amplitude
    so initial recording clicks or pops do not dominate.
    Also handles browser webm audio input via ffmpeg transcoding when needed.
    """
    path_str = str(path)
    try:
        y, _ = librosa.load(path_str, sr=sr, mono=True)
    except Exception:
        # Fallback to ffmpeg conversion for webm or unsupported formats
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav.close()
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                path_str,
                "-ar",
                str(sr),
                "-ac",
                "1",
                temp_wav.name,
            ]
            res = subprocess.run(cmd, capture_output=True)
            if res.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg failed to decode audio from {path_str}: {res.stderr.decode()}"
                )
            y, _ = librosa.load(temp_wav.name, sr=sr, mono=True)
        finally:
            Path(temp_wav.name).unlink(missing_ok=True)

    if len(y) == 0:
        return np.zeros(0, dtype=np.float32)

    # 80 Hz high-pass filter (scipy Butterworth, zero-phase) before trimming
    sos = butter(5, 80.0, btype="highpass", fs=sr, output="sos")
    y = sosfiltfilt(sos, y)

    if trim and len(y) > 0:
        y, _ = librosa.effects.trim(y, top_db=top_db)

    if len(y) > 0:
        # Normalise by the 99.5th percentile of absolute amplitude instead of peak
        p99_5 = float(np.percentile(np.abs(y), 99.5))
        if p99_5 > 1e-7:
            y = np.clip(y / p99_5, -1.0, 1.0)

    return y.astype(np.float32)


def extract_mfcc(y, sr=SR, n_mfcc=N_MFCC, with_deltas=True, cmvn=True):
    """Return features shaped (frames, dims).

    dims is n_mfcc, or 3 * n_mfcc with delta and delta-delta appended.
    cmvn applies per-utterance cepstral mean and variance normalisation,
    which attenuates channel and microphone differences.
    """
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP, win_length=WIN
    )
    feats = [mfcc]
    if with_deltas:
        feats.append(librosa.feature.delta(mfcc))
        feats.append(librosa.feature.delta(mfcc, order=2))
    feats = np.vstack(feats)
    if cmvn:
        feats = (feats - feats.mean(axis=1, keepdims=True)) / (
            feats.std(axis=1, keepdims=True) + 1e-8
        )
    return feats.T


def spectral_summary(y, sr=SR):
    """Frame-level spectral descriptors for vowel and consonant analysis."""
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=WIN, hop_length=HOP)[0]
    rms = librosa.feature.rms(y=y, frame_length=WIN, hop_length=HOP)[0]
    return {
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
        "zcr": zcr,
        "rms": rms,
        "duration_s": len(y) / sr,
    }


def plot_features(y, sr, out_path, title=""):
    """Save a three-panel figure: waveform, log-mel spectrogram, MFCCs.

    Plots the trimmed, filtered signal y.
    """
    mfcc = extract_mfcc(y, sr, with_deltas=False, cmvn=False).T
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, win_length=WIN, n_mels=64
        ),
        ref=np.max,
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    librosa.display.waveshow(y, sr=sr, ax=axes[0])
    axes[0].set_title(f"Waveform {title}".strip())

    img1 = librosa.display.specshow(
        mel, sr=sr, hop_length=HOP, x_axis="time", y_axis="mel", ax=axes[1]
    )
    axes[1].set_title("Log-mel spectrogram")
    fig.colorbar(img1, ax=axes[1], format="%+2.0f dB")

    img2 = librosa.display.specshow(mfcc, sr=sr, hop_length=HOP, x_axis="time", ax=axes[2])
    axes[2].set_title("MFCC")
    axes[2].set_ylabel("coefficient")
    fig.colorbar(img2, ax=axes[2])

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Extract and plot MFCC features.")
    parser.add_argument("audio", help="path to a wav file")
    parser.add_argument("--out", default="outputs", help="output directory")
    args = parser.parse_args()

    y = load_audio(args.audio)
    feats = extract_mfcc(y)
    stats = spectral_summary(y)
    name = Path(args.audio).stem
    fig_path = plot_features(y, SR, Path(args.out) / f"{name}_features.png", title=name)

    print(f"file:        {args.audio}")
    print(f"duration:    {stats['duration_s']:.2f} s")
    print(f"feature map: {feats.shape[0]} frames x {feats.shape[1]} dims")
    print(f"mean centroid: {stats['centroid'].mean():.0f} Hz")
    print(f"figure:      {fig_path}")


if __name__ == "__main__":
    main()
