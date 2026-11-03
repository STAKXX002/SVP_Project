"""Visual styling layer for the Speech and Video Processing coursework demo.

Strict 2-color high-contrast aesthetic: Pure Black (#000000) and Pure Cyan (#00ffff).
No gradients, no pastel tones, no multi-colored badges.
"""

from __future__ import annotations

import html
from typing import Any, List, Optional, Tuple

import gradio as gr

FORCE_DARK_JS = "() => { document.body.classList.add('dark'); }"


def make_theme() -> gr.themes.Base:
    """Create a minimal monochrome theme with black base and cyan accents."""
    base = gr.themes.Base(
        primary_hue=gr.themes.colors.cyan,
        secondary_hue=gr.themes.colors.neutral,
        neutral_hue=gr.themes.colors.neutral,
        radius_size=gr.themes.sizes.radius_none,
    )
    return base.set(
        body_background_fill="#000000",
        body_background_fill_dark="#000000",
        body_text_color="#ffffff",
        body_text_color_dark="#ffffff",
        block_background_fill="#000000",
        block_background_fill_dark="#000000",
        block_border_color="#00ffff",
        block_border_color_dark="#00ffff",
        block_border_width="1px",
        block_shadow="none",
        block_label_background_fill="#000000",
        block_label_background_fill_dark="#000000",
        block_label_text_color="#00ffff",
        block_label_text_color_dark="#00ffff",
        block_title_text_color="#00ffff",
        block_title_text_color_dark="#00ffff",
        input_background_fill="#000000",
        input_background_fill_dark="#000000",
        input_border_color="#00ffff",
        input_border_color_dark="#00ffff",
        button_primary_background_fill="#000000",
        button_primary_background_fill_dark="#000000",
        button_primary_background_fill_hover="#00ffff",
        button_primary_background_fill_hover_dark="#00ffff",
        button_primary_text_color="#00ffff",
        button_primary_text_color_dark="#00ffff",
        button_primary_border_color="#00ffff",
        button_primary_border_color_dark="#00ffff",
        button_secondary_background_fill="#000000",
        button_secondary_background_fill_dark="#000000",
        button_secondary_background_fill_hover="#003333",
        button_secondary_background_fill_hover_dark="#003333",
        button_secondary_text_color="#ffffff",
        button_secondary_text_color_dark="#ffffff",
        button_secondary_border_color="#00ffff",
        button_secondary_border_color_dark="#00ffff",
    )


CSS = """
:root {
  --black: #000000;
  --cyan: #00ffff;
  --white: #ffffff;
}

* {
  border-radius: 0px !important;
  box-shadow: none !important;
  text-shadow: none !important;
}

body, .gradio-container {
  background-color: #000000 !important;
  color: #ffffff !important;
  font-family: "JetBrains Mono", Consolas, "Courier New", monospace !important;
  max-width: 1360px !important;
  margin: 0 auto !important;
}

footer { display: none !important; }

/* Technical Minimal Header */
.tech-header {
  border: 1px solid #00ffff;
  background: #000000;
  padding: 16px 20px;
  margin-bottom: 14px;
}
.tech-title {
  color: #00ffff;
  font-size: 1.35rem;
  font-weight: 800;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.tech-sub {
  color: #ffffff;
  font-size: 0.82rem;
  margin-top: 4px;
  letter-spacing: 0.02em;
}
.tech-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 12px;
  font-size: 0.76rem;
  color: #00ffff;
}
.tech-chips span {
  border: 1px solid #00ffff;
  padding: 2px 8px;
  background: #000000;
}

/* Tabs */
button[role="tab"] {
  font-family: "JetBrains Mono", Consolas, monospace !important;
  font-size: 0.86rem !important;
  font-weight: 700 !important;
  color: #00ffff !important;
  background: #000000 !important;
  border: 1px solid #00ffff !important;
  margin-right: 4px !important;
  padding: 7px 16px !important;
  text-transform: uppercase !important;
  transition: none !important;
}
button[role="tab"][aria-selected="true"] {
  color: #000000 !important;
  background: #00ffff !important;
  border: 1px solid #00ffff !important;
}
button[role="tab"]:hover {
  background: #003333 !important;
  color: #00ffff !important;
}

/* Buttons */
button.primary {
  background: #000000 !important;
  color: #00ffff !important;
  border: 1px solid #00ffff !important;
  font-family: "JetBrains Mono", Consolas, monospace !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  padding: 7px 14px !important;
  transition: none !important;
}
button.primary:hover {
  background: #00ffff !important;
  color: #000000 !important;
}
button.secondary {
  background: #000000 !important;
  color: #ffffff !important;
  border: 1px solid #00ffff !important;
  font-family: "JetBrains Mono", Consolas, monospace !important;
  padding: 7px 14px !important;
  text-transform: uppercase !important;
  transition: none !important;
}
button.secondary:hover {
  background: #002222 !important;
  color: #00ffff !important;
}

/* Technical Cards */
.tech-card {
  border: 1px solid #00ffff;
  background: #000000;
  padding: 14px 18px;
  margin-bottom: 8px;
}
.tech-label {
  font-size: 0.72rem;
  color: #00ffff;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.tech-metric-row {
  display: flex;
  align-items: baseline;
  gap: 16px;
  margin: 6px 0;
}
.tech-metric-val {
  font-size: 2.2rem;
  font-weight: 800;
  color: #00ffff;
  line-height: 1;
}
.tech-metric-verdict {
  font-size: 0.95rem;
  font-weight: 700;
  color: #ffffff;
  letter-spacing: 0.06em;
  border: 1px solid #00ffff;
  padding: 2px 8px;
  text-transform: uppercase;
}
.tech-meta-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 0.80rem;
  color: #ffffff;
  margin-top: 8px;
  border-top: 1px solid #00ffff;
  padding-top: 8px;
}
.tech-meta-list b {
  color: #00ffff;
}

/* Notice */
.notice {
  border: 1px solid #00ffff;
  background: #000000;
  color: #00ffff;
  padding: 10px 14px;
  font-size: 0.82rem;
}
.notice b { color: #ffffff; }
.notice code {
  background: #000000;
  color: #00ffff;
  border: 1px solid #00ffff;
  padding: 1px 5px;
}

/* Figures */
.fig img {
  background: #000000 !important;
  border: 1px solid #00ffff !important;
  padding: 0 !important;
}

/* Tables */
table {
  border: 1px solid #00ffff !important;
  background: #000000 !important;
  color: #ffffff !important;
  font-family: "JetBrains Mono", Consolas, monospace !important;
}
th {
  background: #000000 !important;
  color: #00ffff !important;
  border-bottom: 1px solid #00ffff !important;
  font-weight: 700 !important;
}
td {
  border-bottom: 1px solid #003333 !important;
}

/* Form Controls */
input, select, textarea {
  background: #000000 !important;
  color: #ffffff !important;
  border: 1px solid #00ffff !important;
  font-family: "JetBrains Mono", Consolas, monospace !important;
}
"""


def hero_html(title: str, subtitle: str, chips: List[Tuple[str, str]]) -> str:
    """Render technical terminal header in black and cyan with zero animation."""
    chip_spans = "".join(
        f"<span>[ {html.escape(k).upper()}: {html.escape(v).upper()} ]</span>"
        for k, v in chips
    )
    return (
        '<div class="tech-header">'
        f'<div class="tech-title">{html.escape(title).upper()} // TERMINAL</div>'
        f'<div class="tech-sub">{html.escape(subtitle)}</div>'
        f'<div class="tech-chips">{chip_spans}</div>'
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
    """Render high-contrast rectangular readout for pronunciation alignment."""
    pct = max(0.0, min(100.0, float(similarity)))

    detail_lines = []
    if details:
        detail_lines.extend(details)
    else:
        if acoustic_cost is not None:
            detail_lines.append(f"ACOUSTIC ALIGNMENT COST: <b>{acoustic_cost:.3f}</b>")
        if duration_ratio is not None:
            detail_lines.append(f"DURATION RATIO: <b>{duration_ratio:.2f}x</b>")
        if flagged_count is not None:
            detail_lines.append(f"FLAGGED MISMATCH SEGMENTS: <b>{flagged_count}</b>")

    meta_html = "".join(f"<div>{d}</div>" for d in detail_lines)
    return (
        '<div class="tech-card">'
        '<div class="tech-label">PRONUNCIATION SIMILARITY SCORE</div>'
        '<div class="tech-metric-row">'
        f'<div class="tech-metric-val">{pct:.0f}%</div>'
        f'<div class="tech-metric-verdict">{html.escape(verdict).upper()}</div>'
        "</div>"
        f'<div class="tech-meta-list">{meta_html}</div>'
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
    """Render high-contrast rectangular readout for verse classification."""
    val = predicted if predicted else verse
    is_warn = is_low_confidence or low_conf
    status_label = "LOW CONFIDENCE" if is_warn else "CONFIDENT"
    score_extra = (
        f"<div>LOG-LIKELIHOOD: <b>{top_loglik:.1f}</b></div>"
        if top_loglik is not None
        else ""
    )
    return (
        '<div class="tech-card">'
        '<div class="tech-label">VERSE RECOGNITION CLASSIFIER</div>'
        f'<div class="tech-metric-row"><div class="tech-metric-val">{html.escape(str(val)).upper()}</div></div>'
        '<div class="tech-meta-list">'
        f"<div>CLASSIFIER STATUS: <b>{status_label}</b></div>"
        f"<div>CONFIDENCE MARGIN: <b>{margin:.1f}</b></div>"
        f"{score_extra}"
        "</div>"
        "</div>"
    )


def notice_html(title: str, body: str = "") -> str:
    """Render technical notice card with cyan border."""
    body_part = f"<br>{body}" if body else ""
    return f'<div class="notice"><b>{html.escape(title).upper()}</b>{body_part}</div>'


def stat_chips_html(pairs: List[Tuple[str, str]]) -> str:
    """Render technical chips."""
    inner = "".join(
        f"<span>[ {html.escape(k).upper()}: {html.escape(v).upper()} ]</span>"
        for k, v in pairs
    )
    return f'<div class="tech-chips">{inner}</div>'
