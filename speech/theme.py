"""Visual styling layer for the Speech and Video Processing coursework demo.

Minimal, aesthetic dark design with deep black surfaces and electric blue accents.
Uses system and common sans fonts for offline reliability.
"""

from __future__ import annotations

import html
from typing import Any, List, Optional, Tuple

import gradio as gr

FORCE_DARK_JS = "() => { document.body.classList.add('dark'); }"


def make_theme() -> gr.themes.Base:
    """Create a minimal dark theme with black base and electric blue accents."""
    base = gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_sm,
        font=("Inter", "Segoe UI", "Roboto", "system-ui", "sans-serif"),
        font_mono=("JetBrains Mono", "Consolas", "monospace"),
    )
    return base.set(
        body_background_fill="#030712",
        body_background_fill_dark="#030712",
        body_text_color="#e2e8f0",
        body_text_color_dark="#e2e8f0",
        block_background_fill="#070b14",
        block_background_fill_dark="#070b14",
        block_border_color="rgba(59, 130, 246, 0.20)",
        block_border_color_dark="rgba(59, 130, 246, 0.20)",
        block_border_width="1px",
        block_shadow="none",
        block_label_background_fill="#090e1a",
        block_label_background_fill_dark="#090e1a",
        block_label_text_color="#94a3b8",
        block_label_text_color_dark="#94a3b8",
        block_title_text_color="#f8fafc",
        block_title_text_color_dark="#f8fafc",
        input_background_fill="#050811",
        input_background_fill_dark="#050811",
        input_border_color="rgba(59, 130, 246, 0.25)",
        input_border_color_dark="rgba(59, 130, 246, 0.25)",
        button_primary_background_fill="#1d4ed8",
        button_primary_background_fill_dark="#1d4ed8",
        button_primary_background_fill_hover="#2563eb",
        button_primary_background_fill_hover_dark="#2563eb",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_primary_border_color="#3b82f6",
        button_primary_border_color_dark="#3b82f6",
        button_secondary_background_fill="#090e1a",
        button_secondary_background_fill_dark="#090e1a",
        button_secondary_background_fill_hover="#0f172a",
        button_secondary_background_fill_hover_dark="#0f172a",
        button_secondary_text_color="#93c5fd",
        button_secondary_text_color_dark="#93c5fd",
        button_secondary_border_color="rgba(59, 130, 246, 0.30)",
        button_secondary_border_color_dark="rgba(59, 130, 246, 0.30)",
    )


CSS = """
:root {
  --bg-dark: #030712;
  --panel-bg: #070b14;
  --panel-border: rgba(59, 130, 246, 0.22);
  --accent-blue: #3b82f6;
  --accent-light: #60a5fa;
  --text-primary: #f8fafc;
  --text-muted: #94a3b8;
}

body, .gradio-container {
  background-color: #030712 !important;
  color: #e2e8f0 !important;
  max-width: 1320px !important;
  margin: 0 auto !important;
}

footer { display: none !important; }

/* Tabs */
button[role="tab"] {
  font-weight: 600;
  font-size: 0.95rem;
  color: #94a3b8 !important;
  border-bottom: 2px solid transparent !important;
  transition: all 0.2s ease;
}
button[role="tab"][aria-selected="true"] {
  color: #60a5fa !important;
  border-bottom: 2px solid #3b82f6 !important;
  background: rgba(59, 130, 246, 0.08) !important;
}

/* Buttons */
button.primary {
  transition: box-shadow 0.2s, transform 0.15s;
}
button.primary:hover {
  box-shadow: 0 0 20px rgba(59, 130, 246, 0.45);
  transform: translateY(-1px);
}

/* Hero Section */
.hero {
  position: relative;
  overflow: hidden;
  padding: 24px 28px;
  border-radius: 12px;
  border: 1px solid rgba(59, 130, 246, 0.25);
  background: #070b14;
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.5);
}
.hero h1 {
  margin: 0;
  font-size: 1.85rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: #f8fafc;
}
.hero p {
  margin: 6px 0 0;
  color: #94a3b8;
  font-size: 0.92rem;
  max-width: 760px;
}
.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 14px;
}
.chip {
  padding: 4px 12px;
  border-radius: 6px;
  font-size: 0.78rem;
  color: #cbd5e1;
  border: 1px solid rgba(59, 130, 246, 0.25);
  background: #090e1a;
}
.chip b {
  color: #60a5fa;
  font-weight: 700;
}
.eq {
  position: absolute;
  right: 28px;
  top: 50%;
  transform: translateY(-50%);
  display: flex;
  align-items: flex-end;
  gap: 4px;
  height: 48px;
  opacity: 0.85;
}
.eq i {
  display: block;
  width: 4px;
  border-radius: 2px;
  height: 100%;
  background: #3b82f6;
  transform-origin: bottom;
  animation: eq 1.1s ease-in-out infinite;
}
@keyframes eq {
  0%, 100% { transform: scaleY(0.25); }
  50% { transform: scaleY(1.0); }
}
@media (max-width: 760px) { .eq { display: none; } }

/* Score Ring */
.score-wrap {
  display: flex;
  align-items: center;
  gap: 24px;
  padding: 16px 8px;
  flex-wrap: wrap;
}
.ring {
  --p: 0;
  --c: #3b82f6;
  width: 154px;
  height: 154px;
  border-radius: 50%;
  flex: none;
  display: grid;
  place-items: center;
  background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(255, 255, 255, 0.06) 0);
  box-shadow: 0 0 28px -4px rgba(59, 130, 246, 0.35);
}
.ring-in {
  width: 126px;
  height: 126px;
  border-radius: 50%;
  background: #050811;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  border: 1px solid rgba(59, 130, 246, 0.20);
}
.ring-num {
  font-size: 2.2rem;
  font-weight: 800;
  line-height: 1;
  color: #f8fafc;
}
.ring-sub {
  margin-top: 4px;
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--c);
  font-weight: 700;
}
.score-meta {
  display: flex;
  flex-direction: column;
  gap: 6px;
  color: #94a3b8;
  font-size: 0.88rem;
}
.score-meta b {
  color: #f8fafc;
}

/* Verse Card */
.verse-card {
  padding: 18px 22px;
  border-radius: 10px;
  border: 1px solid rgba(59, 130, 246, 0.22);
  border-left: 4px solid #3b82f6;
  background: #070b14;
}
.verse-card .lab {
  font-size: 0.74rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #94a3b8;
}
.verse-card .val {
  font-size: 2.3rem;
  font-weight: 800;
  color: #f8fafc;
  line-height: 1.1;
  margin: 4px 0 8px;
}
.badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 4px;
  font-size: 0.78rem;
  font-weight: 600;
}
.badge.ok {
  background: rgba(37, 99, 235, 0.15);
  color: #60a5fa;
  border: 1px solid rgba(59, 130, 246, 0.40);
}
.badge.warn {
  background: rgba(245, 158, 11, 0.12);
  color: #fbbf24;
  border: 1px solid rgba(245, 158, 11, 0.40);
}

/* Notice Cards */
.notice {
  padding: 12px 16px;
  border-radius: 8px;
  border: 1px solid rgba(59, 130, 246, 0.30);
  background: #070b14;
  color: #bfdbfe;
  font-size: 0.88rem;
}
.notice b {
  color: #f8fafc;
}
.notice code {
  background: rgba(59, 130, 246, 0.12);
  color: #93c5fd;
  padding: 2px 6px;
  border-radius: 4px;
  border: 1px solid rgba(59, 130, 246, 0.25);
}

/* Matplotlib Figures */
.fig img {
  background: #030712 !important;
  border: 1px solid rgba(59, 130, 246, 0.22) !important;
  border-radius: 8px !important;
  padding: 4px !important;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6) !important;
}
"""

_BAND_COLORS = (
    ("excellent", "#38bdf8"),
    ("good", "#60a5fa"),
    ("needs", "#fbbf24"),
    ("different", "#f87171"),
)


def _band_color(verdict: str) -> str:
    v = verdict.lower()
    for key, color in _BAND_COLORS:
        if key in v:
            return color
    return "#38bdf8"


def hero_html(title: str, subtitle: str, chips: List[Tuple[str, str]]) -> str:
    """Render minimal dark hero section with animated equalizer bars and blue chips."""
    bars = "".join(f'<i style="animation-delay:{i * 0.08:.2f}s"></i>' for i in range(14))
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


def score_card_html(
    similarity: float,
    verdict: str,
    details: Optional[List[str]] = None,
    acoustic_cost: Optional[float] = None,
    duration_ratio: Optional[float] = None,
    flagged_count: Optional[int] = None,
    **kwargs: Any,
) -> str:
    """Render minimal circular score indicator with acoustic and duration details."""
    pct = max(0.0, min(100.0, float(similarity)))
    color = _band_color(verdict)

    detail_lines = []
    if details:
        detail_lines.extend(details)
    else:
        if acoustic_cost is not None:
            detail_lines.append(f"Acoustic alignment cost: <b>{acoustic_cost:.3f}</b>")
        if duration_ratio is not None:
            detail_lines.append(f"Duration ratio (learner/ref): <b>{duration_ratio:.2f}x</b>")
        if flagged_count is not None:
            detail_lines.append(f"Flagged mismatch segments: <b>{flagged_count}</b>")

    meta_html = "".join(f"<div>{d}</div>" for d in detail_lines)
    return (
        '<div class="score-wrap">'
        f'<div class="ring" style="--p:{pct:.1f};--c:{color}">'
        f'<div class="ring-in"><span class="ring-num">{pct:.0f}%</span>'
        f'<span class="ring-sub">{html.escape(verdict)}</span></div></div>'
        f'<div class="score-meta">{meta_html}</div>'
        "</div>"
    )


def verse_card_html(
    verse: str = "",
    margin: float = 0.0,
    low_conf: bool = False,
    predicted: str = "",
    is_low_confidence: bool = False,
    top_loglik: Optional[float] = None,
    **kwargs: Any,
) -> str:
    """Render minimal verse classification card with status badge and likelihood margin."""
    val = predicted if predicted else verse
    is_warn = is_low_confidence or low_conf
    badge = (
        '<span class="badge warn">Low Confidence</span>'
        if is_warn
        else '<span class="badge ok">Confident</span>'
    )
    score_extra = (
        f' &nbsp; <span class="chip">log-lik <b>{top_loglik:.1f}</b></span>'
        if top_loglik is not None
        else ""
    )
    return (
        '<div class="verse-card">'
        '<div class="lab">Predicted Verse</div>'
        f'<div class="val">{html.escape(str(val))}</div>'
        f'{badge} <span class="chip">margin <b>{margin:.1f}</b></span>{score_extra}'
        "</div>"
    )


def notice_html(title: str, body: str = "") -> str:
    """Render informational notice card with subtle blue boundary."""
    body_part = f"<br>{body}" if body else ""
    return f'<div class="notice"><b>{html.escape(title)}</b>{body_part}</div>'


def stat_chips_html(pairs: List[Tuple[str, str]]) -> str:
    """Render a container of blue chip badges."""
    inner = "".join(
        f'<span class="chip">{html.escape(k)} <b>{html.escape(v)}</b></span>'
        for k, v in pairs
    )
    return f'<div class="chips">{inner}</div>'
