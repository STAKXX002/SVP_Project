"""Calibrate the DTW scoring constants from your own recordings.

Record, for one reference word (for example krishna):
    --same   one or more repeats of the correct word
    --bad    one or more mispronunciations (kishna, krisna, ...)
    --neg    one or more different words (negative control)

Example:
    python -m speech.calibrate --ref data/reference/krishna.wav \
        --same data/learner/krishna_2.wav data/learner/krishna_3.wav \
        --bad data/learner/kishna.wav data/learner/krisna.wav \
        --neg data/learner/rama.wav data/learner/shiva.wav --write

Without --write it only prints suggestions. With --write it stores them in
speech/calibration.json, which speech/dtw_score.py picks up automatically.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from speech.dtw_score import (
    ACOUSTIC_WEIGHT,
    CALIBRATION_FILE,
    acoustic_score,
    pair_cost,
)
from watermark import save_figure


def collect(ref, paths):
    rows = []
    for p in paths:
        cost, frame_cost = pair_cost(ref, p)
        rows.append((Path(p).stem, cost, frame_cost))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--same", nargs="+", required=True)
    ap.add_argument("--bad", nargs="+", default=[])
    ap.add_argument("--neg", nargs="+", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--write", action="store_true", help="store suggestions in speech/calibration.json")
    args = ap.parse_args()

    same = collect(args.ref, args.same)
    bad = collect(args.ref, args.bad)
    neg = collect(args.ref, args.neg)

    same_cost = np.mean([c for _, c, _ in same])
    neg_cost = np.mean([c for _, c, _ in neg])
    bad_cost = np.mean([c for _, c, _ in bad]) if bad else float("nan")

    if neg_cost - same_cost < 0.10:
        print("WARNING: negative control is less than 0.10 away from the same-word repeats.")
        print("         The features barely separate right from wrong. Check recording level,")
        print("         background noise and that --neg really is a different word.\n")

    cost_best = float(same_cost)
    cost_zero = float(neg_cost)

    same_frames = np.concatenate([fc for _, _, fc in same])
    # Flag a frame when it is clearly worse than the worst 5% of frames in a correct repeat.
    flag_cost = float(np.percentile(same_frames, 95) * 1.10)
    flag_cost = min(flag_cost, cost_zero)

    print(f"{'file':28s} {'group':6s} {'avg cost':>9s} {'acoustic %':>11s}")
    for group, rows in (("same", same), ("bad", bad), ("neg", neg)):
        for name, cost, _ in rows:
            print(f"{name:28s} {group:6s} {cost:9.3f} {acoustic_score(cost, cost_best, cost_zero):11.0f}")

    print()
    print(f"suggested COST_BEST = {cost_best:.3f}   (mean of same-word repeats)")
    print(f"suggested COST_ZERO = {cost_zero:.3f}   (mean of negative controls)")
    print(f"suggested FLAG_COST = {flag_cost:.3f}   (1.10 x 95th percentile of same-word frame cost)")
    if bad:
        print(f"mispronunciations average cost {bad_cost:.3f}")
        if bad_cost <= cost_best + 0.03:
            print("NOTE: mispronunciations are indistinguishable from correct repeats at word level.")
            print("      Rely on flagged regions and duration rather than the headline percentage,")
            print("      and say so in the report. Deletions of short sounds (krishna -> kishna)")
            print("      are the hardest case for MFCC + DTW.")
    print(f"overall similarity = {ACOUSTIC_WEIGHT:.2f} x acoustic + {1 - ACOUSTIC_WEIGHT:.2f} x duration")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels, vals, colors = [], [], []
    for group, rows, color in (("same", same, "tab:green"), ("bad", bad, "tab:orange"),
                               ("neg", neg, "tab:red")):
        for name, cost, _ in rows:
            labels.append(f"{name}\n({group})")
            vals.append(cost)
            colors.append(color)
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(vals)), 4.5))
    ax.bar(range(len(vals)), vals, color=colors)
    ax.axhline(cost_best, color="tab:green", linestyle="--", label="COST_BEST")
    ax.axhline(cost_zero, color="tab:red", linestyle="--", label="COST_ZERO")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("average DTW cost")
    ax.set_title("Calibration: average DTW cost against the reference")
    ax.legend()
    fig.tight_layout()
    fig_path = out / "calibration.png"
    save_figure(fig, fig_path, dpi=150)
    plt.close(fig)
    print(f"figure: {fig_path}")

    if args.write:
        CALIBRATION_FILE.write_text(json.dumps(
            {"COST_BEST": round(cost_best, 4), "COST_ZERO": round(cost_zero, 4),
             "FLAG_COST": round(flag_cost, 4)}, indent=2) + "\n")
        print(f"wrote {CALIBRATION_FILE}")


if __name__ == "__main__":
    main()
