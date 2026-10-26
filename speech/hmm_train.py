"""Verse recognition with one GMM-HMM per verse (speaker-dependent, small vocabulary).

Data layout (untracked):
    data/verses/<verse_id>/*.wav     for example data/verses/01/take_1.wav

Commands:
    python -m speech.hmm_train train                  fit one model per verse, save to data/models/verses.pkl
    python -m speech.hmm_train eval --folds 4         cross-validated accuracy and confusion matrix figure
    python -m speech.hmm_train predict some.wav       print the ranked verses for one recording

Design notes:
  * Features are the 39-dim CMVN MFCC + delta + delta-delta from speech.features.
  * Each model is left-to-right (self loop, next state, skip one) so it follows the verse in time.
  * The state count is chosen from the median recording length (about 200 ms per state).
  * Each state is one diagonal Gaussian by default (--mix 1). Mixtures (--mix 2) are supported but
    tend to degenerate with fewer than about 10 takes per verse, and then fall back automatically.
  * Models are initialised by uniform segmentation (flat start) rather than k-means.
  * Classification picks the model with the highest log-likelihood. There is no reject class,
    so an unknown verse is always assigned to the closest of the trained verses.
"""

import argparse
import logging
import pickle
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hmmlearn.hmm import GMMHMM, GaussianHMM

from speech.features import extract_mfcc, load_audio
from watermark import save_figure

logging.getLogger("hmmlearn").setLevel(logging.ERROR)  # silence per-iteration EM warnings

DEFAULT_DATA = Path("data/verses")
DEFAULT_MODEL = Path("data/models/verses.pkl")
FRAMES_PER_STATE = 20  # 200 ms at a 10 ms hop
MIN_STATES, MAX_STATES = 5, 40


def list_dataset(data_dir):
    """Return {verse_id: [wav paths]} for every subfolder that holds wav files."""
    data_dir = Path(data_dir)
    out = {}
    for sub in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        wavs = sorted(sub.glob("*.wav"))
        if wavs:
            out[sub.name] = wavs
    if not out:
        raise SystemExit(f"No verse folders with wav files found under {data_dir}")
    return out


def features_for(path):
    return extract_mfcc(load_audio(path))


def left_right_matrices(n_states):
    start = np.zeros(n_states)
    start[0] = 1.0
    trans = np.zeros((n_states, n_states))
    for i in range(n_states):
        if i == n_states - 1:
            trans[i, i] = 1.0
        elif i == n_states - 2:
            trans[i, i], trans[i, i + 1] = 0.6, 0.4
        else:
            trans[i, i], trans[i, i + 1], trans[i, i + 2] = 0.6, 0.35, 0.05
    return start, trans


def auto_states(feature_list):
    median_frames = int(np.median([len(f) for f in feature_list]))
    return int(np.clip(median_frames // FRAMES_PER_STATE, MIN_STATES, MAX_STATES))


def uniform_segment_stats(feature_list, n_states, min_var=1e-2):
    """Flat start: split every recording into n_states equal chunks and pool each chunk's frames.

    Returns per-state means and diagonal variances. This time-aligned initialisation is what makes
    a left-to-right model stable with only a handful of recordings; k-means over all frames is not.
    """
    dims = feature_list[0].shape[1]
    pools = [[] for _ in range(n_states)]
    for f in feature_list:
        edges = np.linspace(0, len(f), n_states + 1).astype(int)
        for s in range(n_states):
            seg = f[edges[s]:max(edges[s + 1], edges[s] + 1)]
            pools[s].append(seg)
    means = np.zeros((n_states, dims))
    covars = np.zeros((n_states, dims))
    for s in range(n_states):
        frames = np.vstack(pools[s])
        means[s] = frames.mean(axis=0)
        covars[s] = np.maximum(frames.var(axis=0), min_var)
    return means, covars


def _fit_gaussian(feature_list, n_states, seed):
    X = np.vstack(feature_list)
    lengths = [len(f) for f in feature_list]
    start, trans = left_right_matrices(n_states)
    means, covars = uniform_segment_stats(feature_list, n_states)
    model = GaussianHMM(
        n_components=n_states, covariance_type="diag", min_covar=1e-2,
        n_iter=20, tol=1e-2, init_params="", params="stmc", random_state=seed,
    )
    model.startprob_ = start
    model.transmat_ = trans
    model.means_ = means
    model.covars_ = covars
    model.fit(X, lengths)
    return model, X, lengths


def _fit_mixture(feature_list, n_states, n_mix, seed):
    X = np.vstack(feature_list)
    lengths = [len(f) for f in feature_list]
    start, trans = left_right_matrices(n_states)
    model = GMMHMM(
        n_components=n_states, n_mix=n_mix, covariance_type="diag", min_covar=5e-2,
        n_iter=15, tol=1e-2, init_params="mcw", params="stmcw", random_state=seed,
    )
    model.startprob_ = start
    model.transmat_ = trans
    model.fit(X, lengths)
    return model, X, lengths


def _healthy(model, X, lengths):
    try:
        return bool(np.isfinite(model.score(X, lengths)))
    except Exception:
        return False


def train_verse(feature_list, n_states=None, n_mix=1, seed=0):
    """Fit one left-to-right HMM.

    n_mix == 1 gives a single diagonal Gaussian per state (the default, stable with few takes).
    n_mix > 1 tries a Gaussian mixture per state and falls back to the single Gaussian
    model if EM goes non-finite, which happens easily with fewer than about 10 takes.
    """
    n_states = n_states or auto_states(feature_list)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if n_mix > 1:
            try:
                model, X, lengths = _fit_mixture(feature_list, n_states, n_mix, seed)
                if _healthy(model, X, lengths):
                    return model
            except Exception:
                pass
        model, X, lengths = _fit_gaussian(feature_list, n_states, seed)
        if not _healthy(model, X, lengths):
            raise RuntimeError("HMM training produced a non-finite likelihood; "
                               "try fewer states with --states or check the recordings")
        return model


def train_all(dataset_feats, n_states=None, n_mix=1):
    return {verse: train_verse(feats, n_states, n_mix) for verse, feats in dataset_feats.items()}


def rank(models, feats):
    """Return [(verse, total_loglik, per_frame_loglik)] best first."""
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for verse, m in models.items():
            try:
                ll = float(m.score(feats))
            except Exception:
                ll = float("-inf")
            rows.append((verse, ll, ll / len(feats)))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def save_bundle(models, path=DEFAULT_MODEL):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump({"models": models}, fh)
    return path


def load_bundle(path=DEFAULT_MODEL):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def classify_file(wav_path, bundle):
    """Used by the Gradio app. Returns (best_verse, ranked list, margin per frame)."""
    feats = features_for(wav_path)
    ranked = rank(bundle["models"], feats)
    margin = ranked[0][2] - ranked[1][2] if len(ranked) > 1 else float("nan")
    return ranked[0][0], ranked, margin


def cross_validate(dataset, folds, n_states=None, n_mix=1):
    """Hold out every fold-th recording of each verse, train on the rest, test on the held-out ones."""
    feats = {v: [features_for(p) for p in paths] for v, paths in dataset.items()}
    verses = list(feats)
    min_n = min(len(f) for f in feats.values())
    folds = max(2, min(folds, min_n))
    y_true, y_pred = [], []
    for k in range(folds):
        train = {v: [f for i, f in enumerate(fs) if i % folds != k] for v, fs in feats.items()}
        test = {v: [f for i, f in enumerate(fs) if i % folds == k] for v, fs in feats.items()}
        t0 = time.time()
        models = train_all(train, n_states, n_mix)
        for v, fs in test.items():
            for f in fs:
                y_true.append(v)
                y_pred.append(rank(models, f)[0][0])
        print(f"fold {k + 1}/{folds} done in {time.time() - t0:.1f} s")
    cm = np.zeros((len(verses), len(verses)), dtype=int)
    idx = {v: i for i, v in enumerate(verses)}
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    acc = float(np.mean([t == p for t, p in zip(y_true, y_pred)]))
    return verses, cm, acc


def plot_confusion(verses, cm, acc, out_path):
    fig, ax = plt.subplots(figsize=(1.0 + 0.8 * len(verses), 1.0 + 0.7 * len(verses)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(verses)))
    ax.set_yticks(range(len(verses)))
    ax.set_xticklabels(verses, rotation=45, ha="right")
    ax.set_yticklabels(verses)
    ax.set_xlabel("predicted verse")
    ax.set_ylabel("true verse")
    ax.set_title(f"Verse recognition, held-out accuracy {100 * acc:.1f}%")
    for i in range(len(verses)):
        for j in range(len(verses)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="GMM-HMM verse recognition.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("train", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--data", default=str(DEFAULT_DATA))
        p.add_argument("--states", type=int, default=None, help="fixed number of HMM states")
        p.add_argument("--mix", type=int, default=1, help="Gaussians per state (1 is the stable default)")
        if name == "train":
            p.add_argument("--model", default=str(DEFAULT_MODEL))
        else:
            p.add_argument("--folds", type=int, default=4)
            p.add_argument("--out", default="outputs")

    p = sub.add_parser("predict")
    p.add_argument("wav")
    p.add_argument("--model", default=str(DEFAULT_MODEL))

    args = ap.parse_args()

    if args.cmd == "train":
        dataset = list_dataset(args.data)
        feats = {v: [features_for(p) for p in paths] for v, paths in dataset.items()}
        for v, fs in feats.items():
            print(f"verse {v}: {len(fs)} recordings, {auto_states(fs) if args.states is None else args.states} states")
        t0 = time.time()
        models = train_all(feats, args.states, args.mix)
        path = save_bundle(models, args.model)
        print(f"trained {len(models)} models in {time.time() - t0:.1f} s, saved to {path}")

    elif args.cmd == "eval":
        dataset = list_dataset(args.data)
        verses, cm, acc = cross_validate(dataset, args.folds, args.states, args.mix)
        fig = plot_confusion(verses, cm, acc, Path(args.out) / "verse_confusion.png")
        print(f"held-out accuracy: {100 * acc:.1f}%")
        print(f"confusion matrix figure: {fig}")

    elif args.cmd == "predict":
        bundle = load_bundle(args.model)
        best, ranked, margin = classify_file(args.wav, bundle)
        print(f"predicted verse: {best}   (margin to runner-up {margin:.2f} per frame)")
        for verse, ll, per_frame in ranked:
            print(f"  {verse:10s} log-lik {ll:12.1f}   per frame {per_frame:8.3f}")


if __name__ == "__main__":
    main()
