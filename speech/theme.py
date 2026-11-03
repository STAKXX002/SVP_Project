"""Visual layer for the SVP demo app: Gradio theme, CSS, and HTML card helpers.

Pure presentation. Contains no data processing. Uses system fonts only so the
UI renders identically without internet access at the demo venue.
"""
from __future__ import annotations

import html

import gradio as gr

FORCE_DARK_JS = "() => { document.body.classList.add('dark'); }"


def make_theme():
    base = gr.themes.Base(
        primary_hue=gr.themes.colors.cyan,
        secondary_hue=gr.themes.colors.violet,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_lg,
        font=("Inter", "Segoe UI", "Roboto", "Arial", "sans-serif"),
        font_mono=("JetBrains Mono", "Consolas", "monospace"),
    )
    return base.set(
        body_background_fill="#0b1020",
        body_background_fill_dark="#0b1020",
        body_text_color_dark="#e6edf7",
        block_background_fill_dark="rgba(255,255,255,0.04)",
        block_border_color_dark="rgba(255,255,255,0.10)",
        block_border_width="1px",
        block_shadow="0 8px 30px rgba(0,0,0,0.35)",
        block_label_background_fill_dark="rgba(255,255,255,0.06)",
        block_title_text_color_dark="#9fb3d1",
        input_background_fill_dark="rgba(8,12,26,0.85)",
        input_border_color_dark="rgba(255,255,255,0.12)",
        button_primary_background_fill="linear-gradient(90deg,#22d3ee,#8b5cf6)",
        button_primary_background_fill_dark="linear-gradient(90deg,#22d3ee,#8b5cf6)",
        button_primary_background_fill_hover="linear-gradient(90deg,#67e8f9,#a78bfa)",
        button_primary_background_fill_hover_dark="linear-gradient(90deg,#67e8f9,#a78bfa)",
        button_primary_text_color="#06101f",
        button_primary_text_color_dark="#06101f",
        button_secondary_background_fill_dark="rgba(255,255,255,0.06)",
        button_secondary_border_color_dark="rgba(255,255,255,0.14)",
    )


CSS = """
:root{
  --bg:#0b1020; --line:rgba(255,255,255,.10); --muted:#9fb3d1;
  --cyan:#22d3ee; --violet:#8b5cf6; --green:#34d399; --amber:#fbbf24; --red:#f87171;
}
.gradio-container{
  max-width:1280px !important; margin:0 auto !important;
  background:
    radial-gradient(1100px 560px at 8% -10%, rgba(34,211,238,.16), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(139,92,246,.18), transparent 55%),
    var(--bg) !important;
}
footer{display:none !important}

/* hero */
.hero{position:relative;overflow:hidden;padding:28px 32px;border-radius:20px;
  border:1px solid var(--line);
  background:linear-gradient(135deg,rgba(34,211,238,.10),rgba(139,92,246,.14));}
.hero h1{margin:0;font-size:2.1rem;font-weight:800;letter-spacing:-.02em;
  background:linear-gradient(90deg,#e6edf7,#67e8f9 50%,#c4b5fd);
  -webkit-background-clip:text;background-clip:text;color:transparent;}
.hero p{margin:.45rem 0 0;color:var(--muted);max-width:720px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:16px}
.chip{padding:4px 12px;border-radius:999px;font-size:.78rem;color:#cfe3ff;
  border:1px solid var(--line);background:rgba(255,255,255,.05)}
.chip b{color:var(--cyan);font-weight:700}
.eq{position:absolute;right:32px;top:50%;transform:translateY(-50%);
  display:flex;align-items:flex-end;gap:5px;height:64px;opacity:.9}
.eq i{display:block;width:6px;border-radius:3px;height:100%;
  background:linear-gradient(180deg,#22d3ee,#8b5cf6);
  transform-origin:bottom;animation:eq 1.1s ease-in-out infinite}
@keyframes eq{0%,100%{transform:scaleY(.2)}50%{transform:scaleY(1)}}
@media (max-width:760px){.eq{display:none}}
@media (prefers-reduced-motion:reduce){.eq i{animation:none;transform:scaleY(.6)}}

/* tabs */
button[role="tab"]{font-weight:600;font-size:.98rem}
button[role="tab"][aria-selected="true"]{
  color:var(--cyan) !important;border-bottom:2px solid var(--cyan) !important;}

/* buttons */
button.primary{transition:box-shadow .2s, transform .15s}
button.primary:hover{box-shadow:0 0 26px rgba(34,211,238,.45);transform:translateY(-1px)}

/* score ring */
.score-wrap{display:flex;align-items:center;gap:28px;padding:18px 8px;flex-wrap:wrap}
.ring{--p:0;--c:#22d3ee;width:176px;height:176px;border-radius:50%;flex:none;
  display:grid;place-items:center;
  background:conic-gradient(var(--c) calc(var(--p) * 1%), rgba(255,255,255,.08) 0);
  box-shadow:0 0 40px -8px var(--c);}
.ring-in{width:146px;height:146px;border-radius:50%;background:#0b1020;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  border:1px solid var(--line);}
.ring-num{font-size:2.5rem;font-weight:800;line-height:1;color:#fff}
.ring-sub{margin-top:6px;font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;color:var(--c)}
.score-meta{display:flex;flex-direction:column;gap:8px;color:var(--muted);font-size:.92rem}
.score-meta b{color:#e6edf7}

/* verse card */
.verse-card{padding:20px 24px;border-radius:16px;border:1px solid var(--line);
  background:linear-gradient(135deg,rgba(139,92,246,.14),rgba(34,211,238,.08));}
.verse-card .lab{font-size:.75rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
.verse-card .val{font-size:2.6rem;font-weight:800;color:#fff;line-height:1.1}
.badge{display:inline-block;margin-top:10px;padding:4px 12px;border-radius:999px;
  font-size:.8rem;font-weight:600}
.badge.ok{background:rgba(52,211,153,.15);color:var(--green);border:1px solid rgba(52,211,153,.4)}
.badge.warn{background:rgba(251,191,36,.12);color:var(--amber);border:1px solid rgba(251,191,36,.4)}

/* notice */
.notice{padding:14px 18px;border-radius:14px;border:1px dashed rgba(251,191,36,.5);
  background:rgba(251,191,36,.06);color:#fde68a}
.notice b{color:#fff}
.notice code{background:rgba(255,255,255,.08);padding:1px 6px;border-radius:6px}

/* matplotlib figures are white; present them as light cards */
.fig img{background:#fff;border-radius:12px;padding:6px}
"""

_BAND_COLORS = (
    ("excellent", "#34d399"),
    ("good", "#22d3ee"),
    ("needs", "#fbbf24"),
    ("different", "#f87171"),
)


def _band_color(verdict: str) -> str:
    v = verdict.lower()
    for key, color in _BAND_COLORS:
        if key in v:
            return color
    return "#22d3ee"


def hero_html(title: str, subtitle: str, chips: list[tuple[str, str]]) -> str:
    """chips: list of (label, value), for example [("Verses", "9"), ("Scenes", "192")]."""
    bars = "".join(
        f'<i style="animation-delay:{i * 0.09:.2f}s"></i>' for i in range(14)
    )
    chip_html = "".join(
        f'<span class="chip">{html.escape(k)} <b>{html.escape(v)}</b></span>'
        for k, v in chips
    )
    return (
        '<div class="hero">'
        f'<div class="eq">{bars}</div>'
        f"<h1>{html.escape(title)}</h1>"
        f"<p>{html.escape(subtitle)}</p>"
        f'<div class="chips">{chip_html}</div>'
        "</div>"
    )


def score_card_html(similarity: float, verdict: str, details: list[str] | None = None) -> str:
    pct = max(0.0, min(100.0, float(similarity)))
    color = _band_color(verdict)
    meta = "".join(f"<div>{html.escape(d)}</div>" for d in (details or []))
    return (
        '<div class="score-wrap">'
        f'<div class="ring" style="--p:{pct:.1f};--c:{color}">'
        f'<div class="ring-in"><span class="ring-num">{pct:.0f}%</span>'
        f'<span class="ring-sub">{html.escape(verdict)}</span></div></div>'
        f'<div class="score-meta">{meta}</div>'
        "</div>"
    )


def verse_card_html(verse: str, margin: float, low_conf: bool) -> str:
    badge = (
        '<span class="badge warn">Low confidence</span>'
        if low_conf
        else '<span class="badge ok">Confident</span>'
    )
    return (
        '<div class="verse-card">'
        '<div class="lab">Predicted verse</div>'
        f'<div class="val">{html.escape(str(verse))}</div>'
        f'{badge} <span class="chip">margin <b>{margin:.0f}</b></span>'
        "</div>"
    )


def notice_html(title: str, body: str) -> str:
    return f'<div class="notice"><b>{html.escape(title)}</b><br>{body}</div>'


def stat_chips_html(pairs: list[tuple[str, str]]) -> str:
    inner = "".join(
        f'<span class="chip">{html.escape(k)} <b>{html.escape(v)}</b></span>'
        for k, v in pairs
    )
    return f'<div class="chips">{inner}</div>'