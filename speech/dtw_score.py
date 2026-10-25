"""DTW-based pronunciation scoring: compare a learner recording to a reference.

Usage:
    python -m speech.dtw_score data/reference/krishna.wav data/learner/krishna_1.wav

Similarity = 0.85 x acoustic + 0.15 x duration - flagged-region penalty, clipped to 0..100.
Calibration constants are read from speech/calibration.json when it exists
(written by `python -m speech.calibrate --write`). Otherwise the defaults below apply.
"""

import argparse
import json
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

CALIBRATION_FILE = Path(__file__).with_name("calibration.json")

# Defaults, used only until calibration.json exists.
#   COST_BEST: average DTW cost of a good repeat of the SAME word by the same speaker.
#              Maps to 100% acoustic score. It is never 0, because two takes always differ.
#   COST_ZERO: average DTW cost of a DIFFERENT word. Maps to 0% acoustic score.
#   FLAG_COST: per-frame cost above which a reference region is flagged as a mismatch.
DEFAULTS = {"COST_BEST": 0.30, "COST_ZERO": 0.65, "FLAG_COST": 0.55}
MIN_FLAG_FRAMES = 5  # ignore flagged runs shorter than 50 ms
ACOUSTIC_WEIGHT = 0.85  # remainder goes to the duration score
FLAG_PENALTY = 1.5  # similarity points removed per percent of the reference that is flagged
MAX_FLAG_PENALTY = 40.0


def load_constants():
    consts = dict(DEFAULTS)
    if CALIBRATION_FILE.exists():
        consts.update(json.loads(CALIBRATION_FILE.read_text()))
    return consts


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


def flag_segments(frame_cost, threshold, min_frames=MIN_FLAG_FRAMES):
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


def acoustic_score(avg_cost, cost_best, cost_zero):
    """Linear map: cost_best -> 100, cost_zero -> 0, clipped."""
    span = max(cost_zero - cost_best, 1e-6)
    return 100.0 * float(np.clip((cost_zero - avg_cost) / span, 0.0, 1.0))


def pair_cost(ref_path, learner_path):
    """Average DTW cost and per-frame cost for one pair. Used by calibrate.py."""
    f_ref = extract_mfcc(load_audio(ref_path))
    f_lrn = extract_mfcc(load_audio(learner_path))
    _, wp, local = align(f_ref, f_lrn)
    return float(local.mean()), per_frame_cost(wp, local, len(f_ref))


def score_pronunciation(ref_path, learner_path, out_dir=None):
    """Compare two recordings and return a dict of scores and diagnostics."""
    consts = load_constants()
    y_ref = load_audio(ref_path)
    y_lrn = load_audio(learner_path)

    f_ref = extract_mfcc(y_ref)
    f_lrn = extract_mfcc(y_lrn)

    C, wp, local = align(f_ref, f_lrn)
    avg_cost = float(local.mean())
    acoustic = acoustic_score(avg_cost, consts["COST_BEST"], consts["COST_ZERO"])

    ref_len = len(y_ref) / SR
    lrn_len = len(y_lrn) / SR
    dur = duration_score(ref_len, lrn_len)
    frame_cost = per_frame_cost(wp, local, len(f_ref))
    segments = flag_segments(frame_cost, consts["FLAG_COST"])

    # A single wrong sound barely moves the word-level average, so also penalise flagged time.
    flagged_frac = sum(e - s for s, e, _ in segments) * SR / HOP / max(len(f_ref), 1)
    penalty = min(MAX_FLAG_PENALTY, FLAG_PENALTY * 100.0 * flagged_frac)
    overall = ACOUSTIC_WEIGHT * acoustic + (1.0 - ACOUSTIC_WEIGHT) * dur - penalty
    overall = float(np.clip(overall, 0.0, 100.0))

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
        "flagged_fraction": float(flagged_frac),
        "flag_penalty": float(penalty),
        "constants": consts,
        "figure": None,
    }

    if out_dir is not None:
        name = f"{Path(ref_path).stem}_vs_{Path(learner_path).stem}"
        result["figure"] = str(
            plot_comparison(y_ref, y_lrn, C, wp, frame_cost, segments,
                            Path(out_dir) / f"{name}_dtw.png", name, consts["FLAG_COST"])
        )
    return result


def _mel_db(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, win_length=WIN, n_mels=64
    )
    return librosa.power_to_db(mel, ref=np.max)


def plot_comparison(y_ref, y_lrn, C, wp, frame_cost, segments, out_path, title="",
                    flag_cost=DEFAULTS["FLAG_COST"]):
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
    axes[1, 1].axhline(flag_cost, color="tab:red", linestyle="--", label="flag threshold")
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
        f"  flagged: {100 * result['flagged_fraction']:.0f}% of reference "
        f"(penalty {result['flag_penalty']:.0f} points)",
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
