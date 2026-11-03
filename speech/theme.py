"""Visual styling layer for the Speech and Video Processing coursework demo.

Refined dark aesthetic: Pure black (#000000) base, subtle elevated surfaces (#070b12),
and elegant cyan accents (#00ffff / #22d3ee) with smooth transitions.
"""

from __future__ import annotations

import html
from typing import Any, List, Optional, Tuple

import gradio as gr

FORCE_DARK_JS = "() => { document.body.classList.add('dark'); }"


def make_theme() -> gr.themes.Base:
    """Create a refined dark theme with black base, subtle surfaces, and cyan accents."""
    base = gr.themes.Base(
        primary_hue=gr.themes.colors.cyan,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.neutral,
        radius_size=gr.themes.sizes.radius_sm,
    )
    return base.set(
        body_background_fill="#000000",
        body_background_fill_dark="#000000",
        body_text_color="#f8fafc",
        body_text_color_dark="#f8fafc",
        block_background_fill="#070b12",
        block_background_fill_dark="#070b12",
        block_border_color="rgba(0, 255, 255, 0.22)",
        block_border_color_dark="rgba(0, 255, 255, 0.22)",
        block_border_width="1px",
        block_shadow="none",
        block_label_background_fill="#0a101a",
        block_label_background_fill_dark="#0a101a",
        block_label_text_color="#38bdf8",
        block_label_text_color_dark="#38bdf8",
        block_title_text_color="#f8fafc",
        block_title_text_color_dark="#f8fafc",
        input_background_fill="#04070e",
        input_background_fill_dark="#04070e",
        input_border_color="rgba(0, 255, 255, 0.25)",
        input_border_color_dark="rgba(0, 255, 255, 0.25)",
        button_primary_background_fill="#00ffff",
        button_primary_background_fill_dark="#00ffff",
        button_primary_background_fill_hover="#38bdf8",
        button_primary_background_fill_hover_dark="#38bdf8",
        button_primary_text_color="#000000",
        button_primary_text_color_dark="#000000",
        button_primary_border_color="#00ffff",
        button_primary_border_color_dark="#00ffff",
        button_secondary_background_fill="#0a101a",
        button_secondary_background_fill_dark="#0a101a",
        button_secondary_background_fill_hover="#111c2e",
        button_secondary_background_fill_hover_dark="#111c2e",
        button_secondary_text_color="#38bdf8",
        button_secondary_text_color_dark="#38bdf8",
        button_secondary_border_color="rgba(0, 255, 255, 0.30)",
        button_secondary_border_color_dark="rgba(0, 255, 255, 0.30)",
    )


CSS = """
:root {
  --bg-black: #000000;
  --surface: #070b12;
  --surface-elevated: #0a101a;
  --cyan-accent: #00ffff;
  --cyan-soft: #38bdf8;
  --cyan-border: rgba(0, 255, 255, 0.22);
  --cyan-glow: rgba(0, 255, 255, 0.15);
  --text-white: #f8fafc;
  --text-muted: #94a3b8;
}

body, .gradio-container {
  background-color: #000000 !important;
  color: #f8fafc !important;
  font-family: "Inter", "Segoe UI", -apple-system, system-ui, sans-serif !important;
  max-width: 1360px !important;
  margin: 0 auto !important;
}

footer { display: none !important; }

/* Refined Header */
.tech-header {
  border: 1px solid var(--cyan-border);
  background: var(--surface);
  border-radius: 8px;
  padding: 20px 24px;
  margin-bottom: 14px;
  transition: border-color 0.2s ease;
}
.tech-title {
  color: #f8fafc;
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.01em;
}
.tech-sub {
  color: var(--text-muted);
  font-size: 0.88rem;
  margin-top: 4px;
}
.tech-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
  font-size: 0.78rem;
}
.tech-chips span {
  border: 1px solid var(--cyan-border);
  padding: 3px 10px;
  background: var(--surface-elevated);
  border-radius: 4px;
  color: var(--cyan-soft);
  font-family: "JetBrains Mono", Consolas, monospace;
}
.tech-chips span b {
  color: var(--cyan-accent);
}

/* Tabs */
button[role="tab"] {
  font-family: "Inter", -apple-system, sans-serif !important;
  font-size: 0.90rem !important;
  font-weight: 600 !important;
  color: var(--text-muted) !important;
  background: transparent !important;
  border: 1px solid transparent !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 6px 6px 0 0 !important;
  margin-right: 4px !important;
  padding: 8px 18px !important;
  transition: all 0.2s ease !important;
}
button[role="tab"][aria-selected="true"] {
  color: var(--cyan-accent) !important;
  background: var(--surface-elevated) !important;
  border-color: var(--cyan-border) !important;
  border-bottom: 2px solid var(--cyan-accent) !important;
}
button[role="tab"]:hover {
  color: var(--text-white) !important;
  background: rgba(0, 255, 255, 0.05) !important;
}

/* Buttons */
button.primary {
  background: var(--cyan-accent) !important;
  color: #000000 !important;
  border: 1px solid var(--cyan-accent) !important;
  font-family: "Inter", sans-serif !important;
  font-weight: 700 !important;
  border-radius: 6px !important;
  padding: 8px 16px !important;
  transition: all 0.2s ease !important;
}
button.primary:hover {
  background: #38bdf8 !important;
  box-shadow: 0 0 14px var(--cyan-glow) !important;
  transform: translateY(-1px);
}
button.secondary {
  background: var(--surface-elevated) !important;
  color: var(--cyan-soft) !important;
  border: 1px solid var(--cyan-border) !important;
  font-family: "Inter", sans-serif !important;
  border-radius: 6px !important;
  padding: 8px 16px !important;
  transition: all 0.2s ease !important;
}
button.secondary:hover {
  background: rgba(0, 255, 255, 0.12) !important;
  border-color: var(--cyan-accent) !important;
  color: #f8fafc !important;
}

/* Technical Readout Cards */
.tech-card {
  border: 1px solid var(--cyan-border);
  background: var(--surface);
  border-radius: 8px;
  padding: 16px 20px;
  margin-bottom: 10px;
}
.tech-label {
  font-size: 0.74rem;
  color: var(--text-muted);
  letter-spacing: 0.06em;
  text-transform: uppercase;
  font-weight: 600;
}
.tech-metric-row {
  display: flex;
  align-items: baseline;
  gap: 16px;
  margin: 6px 0;
}
.tech-metric-val {
  font-size: 2.3rem;
  font-weight: 800;
  color: var(--cyan-accent);
  line-height: 1;
  font-family: "JetBrains Mono", Consolas, monospace;
}
.tech-metric-verdict {
  font-size: 0.95rem;
  font-weight: 700;
  color: #f8fafc;
  letter-spacing: 0.05em;
  border: 1px solid var(--cyan-border);
  border-radius: 4px;
  padding: 3px 10px;
  background: var(--surface-elevated);
}
.tech-meta-list {
  display: flex;
  flex-direction: column;
  gap: 5px;
  font-size: 0.84rem;
  color: var(--text-muted);
  margin-top: 10px;
  border-top: 1px solid rgba(0, 255, 255, 0.12);
  padding-top: 10px;
}
.tech-meta-list b {
  color: #f8fafc;
  font-family: "JetBrains Mono", Consolas, monospace;
}

/* Notice */
.notice {
  border: 1px solid var(--cyan-border);
  background: var(--surface);
  border-radius: 6px;
  color: var(--cyan-soft);
  padding: 12px 16px;
  font-size: 0.85rem;
}
.notice b { color: #f8fafc; }
.notice code {
  background: var(--surface-elevated);
  color: var(--cyan-accent);
  border: 1px solid var(--cyan-border);
  border-radius: 4px;
  padding: 1px 6px;
}

/* Figures */
.fig img {
  background: var(--surface) !important;
  border: 1px solid var(--cyan-border) !important;
  border-radius: 6px !important;
  padding: 2px !important;
}

/* Tables */
table {
  border: 1px solid var(--cyan-border) !important;
  background: var(--surface) !important;
  color: #f8fafc !important;
  border-radius: 6px !important;
}
th {
  background: var(--surface-elevated) !important;
  color: var(--cyan-accent) !important;
  border-bottom: 1px solid var(--cyan-border) !important;
  font-weight: 600 !important;
}
td {
  border-bottom: 1px solid rgba(0, 255, 255, 0.08) !important;
}

/* Form Controls */
input, select, textarea {
  background: #04070e !important;
  color: #f8fafc !important;
  border: 1px solid var(--cyan-border) !important;
  border-radius: 6px !important;
}
"""


def hero_html(title: str, subtitle: str, chips: List[Tuple[str, str]]) -> str:
    """Render minimal technical header with subtle cyan badges and no noisy animations."""
    chip_spans = "".join(
        f"<span>{html.escape(k)}: <b>{html.escape(v)}</b></span>"
        for k, v in chips
    )
    return (
        '<div class="tech-header">'
        f'<div class="tech-title">{html.escape(title)}</div>'
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
    """Render clean rectangular readout for pronunciation alignment."""
    pct = max(0.0, min(100.0, float(similarity)))

    detail_lines = []
    if details:
        detail_lines.extend(details)
    else:
        if acoustic_cost is not None:
            detail_lines.append(f"Acoustic Alignment Cost: <b>{acoustic_cost:.3f}</b>")
        if duration_ratio is not None:
            detail_lines.append(f"Duration Ratio: <b>{duration_ratio:.2f}x</b>")
        if flagged_count is not None:
            detail_lines.append(f"Flagged Mismatch Segments: <b>{flagged_count}</b>")

    meta_html = "".join(f"<div>{d}</div>" for d in detail_lines)
    return (
        '<div class="tech-card">'
        '<div class="tech-label">Pronunciation Similarity Score</div>'
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
    """Render clean rectangular readout for verse classification."""
    val = predicted if predicted else verse
    is_warn = is_low_confidence or low_conf
    status_label = "Low Confidence" if is_warn else "Confident"
    score_extra = (
        f"<div>Log-Likelihood: <b>{top_loglik:.1f}</b></div>"
        if top_loglik is not None
        else ""
    )
    return (
        '<div class="tech-card">'
        '<div class="tech-label">Verse Recognition Classifier</div>'
        f'<div class="tech-metric-row"><div class="tech-metric-val">{html.escape(str(val))}</div></div>'
        '<div class="tech-meta-list">'
        f"<div>Classifier Status: <b>{status_label}</b></div>"
        f"<div>Confidence Margin: <b>{margin:.1f}</b></div>"
        f"{score_extra}"
        "</div>"
        "</div>"
    )


def notice_html(title: str, body: str = "") -> str:
    """Render technical notice card with subtle cyan border."""
    body_part = f"<br>{body}" if body else ""
    return f'<div class="notice"><b>{html.escape(title)}</b>{body_part}</div>'


def stat_chips_html(pairs: List[Tuple[str, str]]) -> str:
    """Render technical chips."""
    inner = "".join(
        f"<span>{html.escape(k)}: <b>{html.escape(v)}</b></span>"
        for k, v in pairs
    )
    return f'<div class="tech-chips">{inner}</div>'
