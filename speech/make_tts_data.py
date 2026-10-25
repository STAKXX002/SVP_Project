"""Generate synthetic speech data for the pronunciation and verse experiments.

Usage:
    python -m speech.make_tts_data words                        # words and mispronunciations
    python -m speech.make_tts_data verses --file data/verses.txt # verse training takes

Engines:
    edge     Microsoft Edge neural voices via the edge-tts package. Needs internet. Better quality.
             pip install edge-tts
    espeak   espeak-ng, offline and robotic. sudo apt install espeak-ng

Both read Devanagari with a Hindi voice (hi-IN-MadhurNeural by default). Sanskrit is pronounced
approximately, which is acceptable for a pipeline demonstration and must be stated in the report.

Why takes are varied: a TTS engine returns identical audio for identical input, so repeating a word
gives a DTW cost of about zero and the calibration constants become meaningless. Each take therefore
uses a different speaking rate and pitch and gets light additive noise. This is a much smaller
variation than real repeated human speech, so accuracy on this data is optimistic.

Layout written (all 16 kHz mono wav):
    words:   data/reference/<word>.wav            neutral take of each reference word
             data/learner/<word>_<k>.wav          varied takes of each word
             data/learner/<variant>.wav           mispronunciation variants
    verses:  data/verses/<id>/take_<k>.wav

Verse file format: one verse per line, "<id><TAB><Devanagari text>".
"""

import argparse
import asyncio
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000

# name -> Devanagari text. Variants imitate the spoken mistakes in the assignment.
WORDS = {"krishna": "कृष्ण", "rama": "राम", "shiva": "शिव"}
VARIANTS = {"kishna": "किष्ण", "krisna": "कृस्ण", "krishn": "कृष्ण्"}
# (rate percent, pitch Hz offset) for take k; take 0 is the neutral reference
PROSODY = [(0, 0), (-12, -4), (10, 4), (-6, 3), (15, -3), (5, -6), (-15, 6), (8, 0)]


def convert(src, dst, noise_db, rng):
    """ffmpeg to 16 kHz mono wav, then add light noise and trim nothing (features.py trims)."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ar", str(SR), "-ac", "1",
                    str(dst)], check=True)
    if noise_db is not None:
        y, _ = sf.read(dst)
        power = np.mean(y ** 2) + 1e-12
        noise = rng.standard_normal(len(y)) * np.sqrt(power / (10 ** (noise_db / 10)))
        sf.write(dst, np.clip(y + noise, -1, 1), SR)


async def _edge(text, voice, rate, pitch, out_mp3):
    import edge_tts

    comm = edge_tts.Communicate(text, voice, rate=f"{rate:+d}%", pitch=f"{pitch:+d}Hz")
    await comm.save(str(out_mp3))


def synth(engine, text, voice, rate, pitch, dst, noise_db, rng):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / ("raw.mp3" if engine == "edge" else "raw.wav")
        if engine == "edge":
            asyncio.run(_edge(text, voice, rate, pitch, raw))
        else:
            wpm = int(165 * (1 + rate / 100))
            ppitch = int(np.clip(50 + pitch * 4, 0, 99))
            subprocess.run(["espeak-ng", "-v", voice, "-s", str(wpm), "-p", str(ppitch),
                            "-w", str(raw), text], check=True)
        convert(raw, dst, noise_db, rng)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("what", choices=["words", "verses"])
    ap.add_argument("--engine", choices=["edge", "espeak"], default="edge")
    ap.add_argument("--voice", default=None, help="edge: hi-IN-MadhurNeural (default) or hi-IN-SwaraNeural; "
                                                 "espeak: hi")
    ap.add_argument("--file", default="data/verses.txt", help="verse text file for 'verses'")
    ap.add_argument("--takes", type=int, default=8, help="takes per word or verse (max 8)")
    ap.add_argument("--noise-db", type=float, default=35.0, help="SNR of added noise, 0 disables")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data")
    args = ap.parse_args()

    voice = args.voice or ("hi-IN-MadhurNeural" if args.engine == "edge" else "hi")
    noise = args.noise_db or None
    rng = np.random.default_rng(args.seed)
    root = Path(args.data)
    takes = min(args.takes, len(PROSODY))

    if args.what == "words":
        for name, text in WORDS.items():
            synth(args.engine, text, voice, 0, 0, root / "reference" / f"{name}.wav", None, rng)
            for k in range(1, takes):
                r, p = PROSODY[k]
                synth(args.engine, text, voice, r, p, root / "learner" / f"{name}_{k}.wav", noise, rng)
        for name, text in VARIANTS.items():
            r, p = PROSODY[3]
            synth(args.engine, text, voice, r, p, root / "learner" / f"{name}.wav", noise, rng)
        print(f"wrote references to {root / 'reference'} and takes to {root / 'learner'}")
        print("calibrate with:\n  python -m speech.calibrate --ref data/reference/krishna.wav "
              "--same data/learner/krishna_1.wav data/learner/krishna_2.wav data/learner/krishna_3.wav "
              "--bad data/learner/kishna.wav data/learner/krisna.wav "
              "--neg data/learner/rama_1.wav data/learner/shiva_1.wav --write")
    else:
        lines = [ln.rstrip("\n") for ln in Path(args.file).read_text(encoding="utf-8").splitlines() if ln.strip()]
        for ln in lines:
            vid, _, text = ln.partition("\t")
            if not text:
                raise SystemExit(f"Line needs '<id><TAB><text>': {ln[:40]}")
            for k in range(takes):
                r, p = PROSODY[k]
                synth(args.engine, text.strip(), voice, r, p, root / "verses" / vid / f"take_{k}.wav",
                      noise, rng)
            print(f"verse {vid}: {takes} takes")
        print(f"wrote verse takes under {root / 'verses'}; next: python -m speech.hmm_train eval")


if __name__ == "__main__":
    main()
