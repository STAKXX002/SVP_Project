"""DTW-based pronunciation scoring: compare a learner recording to a reference.

Usage:
    python -m speech.dtw_score data/reference/krishna.wav data/learner/krishna_1.wav
"""

import argparse
from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import cdist

from speech.features import (
    HOP,
    N_FFT,
    SR,
    WIN,
    extract_mfcc,
    load_audio,
    spectral_summary,
)

# Calibration constants. These are heuristics: tune them on your own recordings.
# COST_ZERO: average cosine cost per path step that maps to 0% similarity.
#            Set it near the cost you see when comparing two DIFFERENT words.
# FLAG_COST: per-frame cost above which a region is flagged as a mismatch.
COST_ZERO = 0.80
FLAG_COST = 0.55
MIN_FLAG_FRAMES = 5  # ignore flagged runs shorter than 50 ms
ACOUSTIC_WEIGHT = 0.85  # remainder goes to the duration score


def align(ref_feats, learner_feats):
    """Run DTW on frame features shaped (frames, dims).

    Returns the cost matrix C (ref x learner), the warping path as an
    array of (ref_idx, learner_idx) pairs in time order, and the local
    cost of each step on the path.
    """
    C = cdist(ref_feats, learner_feats, metric="cosine")
    _, wp = librosa.sequence.dtw(C=C)
    wp = wp[::-1]  # librosa returns the path end to start
    local = C[wp[:, 0], wp[:, 1]]
    return C, wp, local


def per_frame_cost(wp, local, n_ref, smooth=5):
    """Mean path cost attributed to each reference frame, lightly smoothed."""
    sums = np.bincount(wp[:, 0], weights=local, minlength=n_ref)
    counts = np.bincount(wp[:, 0], minlength=n_ref)
    cost = sums / np.maximum(counts, 1)
    return uniform_filter1d(cost, size=smooth, mode="nearest")


def flag_segments(frame_cost, threshold=FLAG_COST, min_frames=MIN_FLAG_FRAMES):
    """Return (start_s, end_s, mean_cost) for runs of reference frames above threshold."""
    mask = frame_cost > threshold
    segments = []
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            if j - i >= min_frames:
                segments.append((i * HOP / SR, j * HOP / SR, float(frame_cost[i:j].mean())))
            i = j
        else:
            i += 1
    return segments


def duration_score(ref_len_s, learner_len_s):
    """100 when durations match, falling to 0 when one is twice the other."""
    ratio = learner_len_s / max(ref_len_s, 1e-6)
    return 100.0 * max(0.0, 1.0 - abs(np.log(ratio)) / np.log(2.0))


def score_pronunciation(ref_path, learner_path, out_dir=None):
    """Compare two recordings and return a dict of scores and diagnostics."""
    y_ref = load_audio(ref_path)
    y_lrn = load_audio(learner_path)

    f_ref = extract_mfcc(y_ref)
    f_lrn = extract_mfcc(y_lrn)

    C, wp, local = align(f_ref, f_lrn)
    avg_cost = float(local.mean())
    acoustic = 100.0 * float(np.clip(1.0 - avg_cost / COST_ZERO, 0.0, 1.0))

    ref_len = len(y_ref) / SR
    lrn_len = len(y_lrn) / SR
    dur = duration_score(ref_len, lrn_len)
    overall = ACOUSTIC_WEIGHT * acoustic + (1.0 - ACOUSTIC_WEIGHT) * dur

    frame_cost = per_frame_cost(wp, local, len(f_ref))
    segments = flag_segments(frame_cost)

    s_ref = spectral_summary(y_ref)
    s_lrn = spectral_summary(y_lrn)

    result = {
        "similarity": overall,
        "acoustic_score": acoustic,
        "duration_score": dur,
        "avg_cost": avg_cost,
        "ref_duration_s": ref_len,
        "learner_duration_s": lrn_len,
        "centroid_diff_hz": float(s_lrn["centroid"].mean() - s_ref["centroid"].mean()),
        "flagged_segments": segments,
        "figure": None,
    }

    if out_dir is not None:
        name = f"{Path(ref_path).stem}_vs_{Path(learner_path).stem}"
        result["figure"] = str(
            plot_comparison(y_ref, y_lrn, C, wp, frame_cost, segments,
                            Path(out_dir) / f"{name}_dtw.png", name)
        )
    return result


def _mel_db(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, win_length=WIN, n_mels=64
    )
    return librosa.power_to_db(mel, ref=np.max)


def plot_comparison(y_ref, y_lrn, C, wp, frame_cost, segments, out_path, title=""):
    """Four panels: both spectrograms, DTW path on the cost matrix, per-frame cost."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    librosa.display.specshow(_mel_db(y_ref), sr=SR, hop_length=HOP,
                             x_axis="time", y_axis="mel", ax=axes[0, 0])
    axes[0, 0].set_title("Reference")

    librosa.display.specshow(_mel_db(y_lrn), sr=SR, hop_length=HOP,
                             x_axis="time", y_axis="mel", ax=axes[0, 1])
    axes[0, 1].set_title("Learner")

    axes[1, 0].imshow(C, origin="lower", aspect="auto", cmap="magma")
    axes[1, 0].plot(wp[:, 1], wp[:, 0], color="cyan", linewidth=1.2)
    axes[1, 0].set_xlabel("learner frame")
    axes[1, 0].set_ylabel("reference frame")
    axes[1, 0].set_title("DTW path over cosine cost")

    t = np.arange(len(frame_cost)) * HOP / SR
    axes[1, 1].plot(t, frame_cost, color="tab:blue")
    axes[1, 1].axhline(FLAG_COST, color="tab:red", linestyle="--", label="flag threshold")
    for start, end, _ in segments:
        axes[1, 1].axvspan(start, end, color="tab:red", alpha=0.25)
    axes[1, 1].set_xlabel("reference time (s)")
    axes[1, 1].set_ylabel("local cost")
    axes[1, 1].set_title("Per-frame mismatch (shaded = flagged)")
    axes[1, 1].legend(loc="upper right")

    fig.suptitle(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def format_report(result):
    lines = [
        f"Pronunciation similarity: {result['similarity']:.0f}%",
        f"  acoustic score:  {result['acoustic_score']:.0f}%  (avg DTW cost {result['avg_cost']:.3f})",
        f"  duration score:  {result['duration_score']:.0f}%  "
        f"(ref {result['ref_duration_s']:.2f} s, learner {result['learner_duration_s']:.2f} s)",
        f"  spectral centroid difference: {result['centroid_diff_hz']:+.0f} Hz",
    ]
    if result["flagged_segments"]:
        lines.append("  flagged regions (reference time):")
        for start, end, cost in result["flagged_segments"]:
            lines.append(f"    {start:.2f}s to {end:.2f}s  (cost {cost:.2f})")
    else:
        lines.append("  no regions flagged")
    if result["figure"]:
        lines.append(f"  figure: {result['figure']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Score a learner recording against a reference.")
    parser.add_argument("reference")
    parser.add_argument("learner")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()
    print(format_report(score_pronunciation(args.reference, args.learner, args.out)))


if __name__ == "__main__":
    main()