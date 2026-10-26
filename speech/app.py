"""Gradio demo for the speech half of SVP_Project.

Run:
    python -m speech.app
Then open the printed local URL.

Tab 1 compares a learner recording to a reference word and shows the similarity score,
flagged mismatch regions and the four-panel DTW figure.
Tab 2 recognises which trained verse a recording belongs to.
Reference and verse models are read from disk at startup and cached in memory, so each
request only pays for feature extraction and DTW or HMM scoring.
"""

import argparse
import time
from pathlib import Path

import gradio as gr

from speech.dtw_score import format_report, score_pronunciation
from speech.hmm_train import DEFAULT_MODEL, classify_file, load_bundle
from watermark import TEXT

REF_DIR = Path("data/reference")
OUT_DIR = Path("outputs")

_bundle = None


def list_references():
    return sorted(p.name for p in REF_DIR.glob("*.wav")) if REF_DIR.exists() else []


def get_bundle():
    global _bundle
    if _bundle is None:
        if not Path(DEFAULT_MODEL).exists():
            raise gr.Error(f"No trained model at {DEFAULT_MODEL}. Run: python -m speech.hmm_train train")
        _bundle = load_bundle(DEFAULT_MODEL)
    return _bundle


def run_pronunciation(reference_name, learner_audio):
    if not reference_name:
        raise gr.Error(f"No reference selected. Put wav files in {REF_DIR}/ and press Refresh.")
    if learner_audio is None:
        raise gr.Error("Record or upload a learner audio clip first.")
    t0 = time.time()
    result = score_pronunciation(REF_DIR / reference_name, learner_audio, OUT_DIR / "demo")
    label = f"{result['similarity']:.0f}% similar"
    text = format_report(result) + f"\n  computed in {time.time() - t0:.1f} s"
    return label, text, result["figure"]


def run_verse(audio):
    if audio is None:
        raise gr.Error("Record or upload a clip first.")
    bundle = get_bundle()
    best, ranked, margin = classify_file(audio, bundle)
    rows = [[verse, round(ll, 1), round(pf, 3)] for verse, ll, pf in ranked]
    summary = f"Predicted verse: {best}   (margin to runner-up: {margin:.2f} log-lik per frame)"
    return summary, rows


def refresh_refs():
    refs = list_references()
    return gr.Dropdown(choices=refs, value=refs[0] if refs else None)


def build_app():
    refs = list_references()
    with gr.Blocks(title="SVP Project: Sanskrit speech") as demo:
        gr.Markdown(
            "# Sanskrit pronunciation and verse recognition\n"
            "MFCC + DTW scoring and per-verse GMM-HMM recognition. "
            "Speaker-dependent, small vocabulary."
        )
        gr.Markdown(f"<small>{TEXT}</small>")
        with gr.Tab("Pronunciation check"):
            with gr.Row():
                ref_dd = gr.Dropdown(choices=refs, value=refs[0] if refs else None,
                                     label="Reference word")
                refresh = gr.Button("Refresh references")
            learner = gr.Audio(sources=["microphone", "upload"], type="filepath",
                               label="Learner recording")
            go = gr.Button("Score", variant="primary")
            score_label = gr.Textbox(label="Similarity", interactive=False)
            report = gr.Textbox(label="Details", lines=9, interactive=False)
            fig = gr.Image(label="Alignment and mismatch regions", type="filepath")
            refresh.click(refresh_refs, outputs=ref_dd)
            go.click(run_pronunciation, inputs=[ref_dd, learner], outputs=[score_label, report, fig])

        with gr.Tab("Verse recognition"):
            verse_audio = gr.Audio(sources=["microphone", "upload"], type="filepath",
                                   label="Verse recording")
            go2 = gr.Button("Recognise", variant="primary")
            verse_summary = gr.Textbox(label="Result", interactive=False)
            table = gr.Dataframe(headers=["verse", "log-likelihood", "per frame"],
                                 label="All models, best first", interactive=False)
            go2.click(run_verse, inputs=verse_audio, outputs=[verse_summary, table])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    # Warm the model cache so the first demo click is fast.
    if Path(DEFAULT_MODEL).exists():
        get_bundle()
    build_app().launch(server_name="127.0.0.1", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
