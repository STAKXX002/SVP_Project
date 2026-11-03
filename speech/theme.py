"""Visual styling layer for the Speech and Video Processing coursework demo.

Refined aesthetic: Minimal obsidian black base (#030712), elevated dark surfaces
(#080d16, #0d1422), and subtle cyan accents (rgba(56, 189, 248, 0.18)) with smooth
cubic-bezier transitions and balanced dashboard proportions.
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
        body_background_fill="#030712",
        body_background_fill_dark="#030712",
        body_text_color="#f8fafc",
        body_text_color_dark="#f8fafc",
        block_background_fill="#080d16",
        block_background_fill_dark="#080d16",
        block_border_color="rgba(56, 189, 248, 0.16)",
        block_border_color_dark="rgba(56, 189, 248, 0.16)",
        block_border_width="1px",
        block_shadow="none",
        block_label_background_fill="#0d1422",
        block_label_background_fill_dark="#0d1422",
        block_label_text_color="#38bdf8",
        block_label_text_color_dark="#38bdf8",
        block_title_text_color="#f8fafc",
        block_title_text_color_dark="#f8fafc",
        input_background_fill="#050a12",
        input_background_fill_dark="#050a12",
        input_border_color="rgba(56, 189, 248, 0.20)",
        input_border_color_dark="rgba(56, 189, 248, 0.20)",
        button_primary_background_fill="#00e5ff",
        button_primary_background_fill_dark="#00e5ff",
        button_primary_background_fill_hover="#38bdf8",
        button_primary_background_fill_hover_dark="#38bdf8",
        button_primary_text_color="#000000",
        button_primary_text_color_dark="#000000",
        button_primary_border_color="#00e5ff",
        button_primary_border_color_dark="#00e5ff",
        button_secondary_background_fill="#0d1422",
        button_secondary_background_fill_dark="#0d1422",
        button_secondary_background_fill_hover="#131e33",
        button_secondary_background_fill_hover_dark="#131e33",
        button_secondary_text_color="#38bdf8",
        button_secondary_text_color_dark="#38bdf8",
        button_secondary_border_color="rgba(56, 189, 248, 0.25)",
        button_secondary_border_color_dark="rgba(56, 189, 248, 0.25)",
    )


CSS = """
:root {
  --bg-base: #030712;
  --surface: #080d16;
  --surface-elevated: #0d1422;
  --cyan-accent: #00e5ff;
  --cyan-soft: #38bdf8;
  --cyan-border: rgba(56, 189, 248, 0.18);
  --cyan-border-hover: rgba(56, 189, 248, 0.40);
  --cyan-glow: rgba(0, 229, 255, 0.15);
  --text-primary: #f8fafc;
  --text-secondary: #94a3b8;
  --text-muted: #64748b;
  --font-sans: "Inter", "Segoe UI", -apple-system, system-ui, sans-serif;
  --font-mono: "JetBrains Mono", Consolas, monospace;
}

body, .gradio-container {
  background-color: var(--bg-base) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-sans) !important;
  max-width: 1400px !important;
  margin: 0 auto !important;
  padding: 12px 18px !important;
}

footer { display: none !important; }

/* Subtle scrollbars */
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}
::-webkit-scrollbar-track {
  background: var(--bg-base);
}
::-webkit-scrollbar-thumb {
  background: var(--surface-elevated);
  border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
  background: var(--cyan-border-hover);
}

/* Command Header */
.tech-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 14px;
  background: var(--surface);
  border: 1px solid var(--cyan-border);
  border-radius: 8px;
  padding: 16px 22px;
  margin-bottom: 14px;
  transition: border-color 0.2s ease;
}
.tech-header:hover {
  border-color: var(--cyan-border-hover);
}
.tech-header-left {
  display: flex;
  flex-direction: column;
}
.tech-title-row {
  display: flex;
  align-items: center;
  gap: 10px;
}
.live-dot {
  width: 8px;
  height: 8px;
  background: #22c55e;
  border-radius: 50%;
  box-shadow: 0 0 8px rgba(34, 197, 94, 0.6);
  flex-shrink: 0;
}
.tech-title {
  color: var(--text-primary);
  font-size: 1.35rem;
  font-weight: 700;
  letter-spacing: -0.015em;
  margin: 0;
  line-height: 1.2;
}
.tech-version-pill {
  font-size: 0.70rem;
  color: var(--cyan-soft);
  background: rgba(56, 189, 248, 0.10);
  border: 1px solid rgba(56, 189, 248, 0.25);
  border-radius: 10px;
  padding: 1px 7px;
  font-weight: 600;
  letter-spacing: 0.02em;
}
.tech-sub {
  color: var(--text-secondary);
  font-size: 0.82rem;
  margin: 4px 0 0 0;
}
.tech-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}
.tech-chips span {
  border: 1px solid var(--cyan-border);
  padding: 4px 10px;
  background: var(--surface-elevated);
  border-radius: 5px;
  color: var(--text-secondary);
  font-size: 0.74rem;
  font-family: var(--font-sans);
  transition: all 0.2s ease;
}
.tech-chips span:hover {
  border-color: var(--cyan-border-hover);
  color: var(--text-primary);
}
.tech-chips span b {
  color: var(--cyan-soft);
  font-family: var(--font-mono);
  font-weight: 600;
}

/* Tabs */
button[role="tab"] {
  font-family: var(--font-sans) !important;
  font-size: 0.88rem !important;
  font-weight: 600 !important;
  color: var(--text-secondary) !important;
  background: transparent !important;
  border: 1px solid transparent !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 6px 6px 0 0 !important;
  margin-right: 4px !important;
  padding: 8px 16px !important;
  transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
button[role="tab"][aria-selected="true"] {
  color: var(--cyan-accent) !important;
  background: var(--surface-elevated) !important;
  border-color: var(--cyan-border) !important;
  border-bottom: 2px solid var(--cyan-accent) !important;
}
button[role="tab"]:hover {
  color: var(--text-primary) !important;
  background: rgba(56, 189, 248, 0.06) !important;
}

/* Buttons */
button.primary {
  background: var(--cyan-accent) !important;
  color: #000000 !important;
  border: 1px solid var(--cyan-accent) !important;
  font-family: var(--font-sans) !important;
  font-weight: 700 !important;
  font-size: 0.88rem !important;
  border-radius: 6px !important;
  padding: 8px 16px !important;
  transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
button.primary:hover {
  background: var(--cyan-soft) !important;
  border-color: var(--cyan-soft) !important;
  box-shadow: 0 0 14px var(--cyan-glow) !important;
  transform: translateY(-1px);
}
button.secondary {
  background: var(--surface-elevated) !important;
  color: var(--cyan-soft) !important;
  border: 1px solid var(--cyan-border) !important;
  font-family: var(--font-sans) !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  border-radius: 6px !important;
  padding: 8px 16px !important;
  transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
button.secondary:hover {
  background: rgba(56, 189, 248, 0.12) !important;
  border-color: var(--cyan-accent) !important;
  color: var(--text-primary) !important;
  transform: translateY(-1px);
}

/* Score & Metric Cards */
.score-card {
  background: var(--surface);
  border: 1px solid var(--cyan-border);
  border-radius: 8px;
  padding: 14px 18px;
  height: 100%;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  transition: border-color 0.2s ease;
}
.score-card:hover {
  border-color: var(--cyan-border-hover);
}
.score-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}
.score-card-tag {
  font-size: 0.70rem;
  font-weight: 600;
  letter-spacing: 0.05em;
  color: var(--text-secondary);
  text-transform: uppercase;
}
.score-main-row {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-top: 4px;
}
.score-big-num {
  font-size: 2.2rem;
  font-weight: 800;
  color: var(--text-primary);
  font-family: var(--font-mono);
  line-height: 1;
}
.score-big-num span {
  font-size: 1.1rem;
  color: var(--cyan-soft);
  margin-left: 2px;
}
.score-metrics-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  flex: 1;
  border-left: 1px solid rgba(56, 189, 248, 0.14);
  padding-left: 14px;
}
.score-metric-item {
  display: flex;
  flex-direction: column;
}
.metric-label {
  font-size: 0.68rem;
  color: var(--text-muted);
  font-weight: 500;
}
.metric-value {
  font-size: 0.88rem;
  font-weight: 700;
  color: var(--cyan-soft);
  font-family: var(--font-mono);
  margin-top: 2px;
}

/* Status Badges */
.badge {
  font-size: 0.72rem;
  font-weight: 700;
  border-radius: 4px;
  padding: 2px 8px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.badge-excellent {
  color: #22c55e;
  background: rgba(34, 197, 94, 0.12);
  border: 1px solid rgba(34, 197, 94, 0.35);
}
.badge-good {
  color: #38bdf8;
  background: rgba(56, 189, 248, 0.12);
  border: 1px solid rgba(56, 189, 248, 0.35);
}
.badge-needs-work {
  color: #fbbf24;
  background: rgba(251, 191, 36, 0.12);
  border: 1px solid rgba(251, 191, 36, 0.35);
}
.badge-different {
  color: #ef4444;
  background: rgba(239, 68, 68, 0.12);
  border: 1px solid rgba(239, 68, 68, 0.35);
}
.badge-awaiting {
  color: var(--text-secondary);
  background: rgba(148, 163, 184, 0.10);
  border: 1px solid rgba(148, 163, 184, 0.25);
}
.badge-confident {
  color: #38bdf8;
  background: rgba(56, 189, 248, 0.12);
  border: 1px solid rgba(56, 189, 248, 0.35);
}
.badge-warning {
  color: #fbbf24;
  background: rgba(251, 191, 36, 0.12);
  border: 1px solid rgba(251, 191, 36, 0.35);
}

/* Notice Cards */
.notice {
  border: 1px solid var(--cyan-border);
  background: var(--surface);
  border-radius: 6px;
  color: var(--cyan-soft);
  padding: 10px 14px;
  font-size: 0.82rem;
  margin-top: 6px;
}
.notice b { color: var(--text-primary); }
.notice code {
  background: var(--surface-elevated);
  color: var(--cyan-accent);
  border: 1px solid var(--cyan-border);
  border-radius: 4px;
  padding: 1px 5px;
  font-family: var(--font-mono);
  font-size: 0.78rem;
}

/* About & Limitations 2x2 Cards */
.about-card {
  background: var(--surface);
  border: 1px solid var(--cyan-border);
  border-radius: 8px;
  padding: 18px 22px;
  margin-bottom: 12px;
  transition: border-color 0.2s ease;
}
.about-card:hover {
  border-color: var(--cyan-border-hover);
}
.about-card-title {
  color: var(--cyan-soft);
  font-size: 1.05rem;
  font-weight: 700;
  margin-bottom: 10px;
  letter-spacing: -0.01em;
  border-bottom: 1px solid rgba(56, 189, 248, 0.12);
  padding-bottom: 8px;
}
.about-card p {
  font-size: 0.84rem;
  color: var(--text-secondary);
  line-height: 1.5;
  margin: 6px 0 10px 0;
}
.about-card ul {
  margin: 0;
  padding-left: 18px;
}
.about-card li {
  font-size: 0.82rem;
  color: var(--text-secondary);
  line-height: 1.6;
  margin-bottom: 6px;
}
.about-card li b {
  color: var(--text-primary);
}

/* Figures */
.fig img {
  background: var(--surface) !important;
  border: 1px solid var(--cyan-border) !important;
  border-radius: 6px !important;
  padding: 2px !important;
  transition: border-color 0.2s ease;
}
.fig img:hover {
  border-color: var(--cyan-border-hover) !important;
}

/* Tables */
table {
  border: 1px solid var(--cyan-border) !important;
  background: var(--surface) !important;
  color: var(--text-primary) !important;
  border-radius: 6px !important;
  font-size: 0.82rem !important;
}
th {
  background: var(--surface-elevated) !important;
  color: var(--cyan-soft) !important;
  border-bottom: 1px solid var(--cyan-border) !important;
  font-weight: 600 !important;
  font-size: 0.78rem !important;
  padding: 6px 10px !important;
}
td {
  border-bottom: 1px solid rgba(56, 189, 248, 0.08) !important;
  padding: 6px 10px !important;
  font-family: var(--font-mono) !important;
}

/* Form Controls */
input, select, textarea {
  background: #050a12 !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--cyan-border) !important;
  border-radius: 6px !important;
  font-size: 0.85rem !important;
}
input:focus, select:focus, textarea:focus {
  border-color: var(--cyan-soft) !important;
  box-shadow: 0 0 8px var(--cyan-glow) !important;
}
"""


def hero_html(title: str, subtitle: str, chips: List[Tuple[str, str]]) -> str:
    """Render minimal technical header with subtle cyan badges and live readiness indicator."""
    chip_spans = "".join(
        f"<span>{html.escape(k)}: <b>{html.escape(v)}</b></span>"
        for k, v in chips
    )
    return (
        '<div class="tech-header">'
        '<div class="tech-header-left">'
        '<div class="tech-title-row">'
        '<span class="live-dot" title="Pipeline Ready"></span>'
        f'<h1 class="tech-title">{html.escape(title)}</h1>'
        '<span class="tech-version-pill">Demo v2.0</span>'
        '</div>'
        f'<p class="tech-sub">{html.escape(subtitle)}</p>'
        '</div>'
        f'<div class="tech-chips">{chip_spans}</div>'
        '</div>'
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
    """Render compact, high-density rectangular readout for pronunciation alignment."""
    pct = max(0.0, min(100.0, float(similarity)))
    verdict_clean = str(verdict).strip().title()

    # Determine status badge class
    v_lower = verdict_clean.lower()
    if "excellent" in v_lower:
        badge_cls = "badge-excellent"
    elif "good" in v_lower:
        badge_cls = "badge-good"
    elif "work" in v_lower:
        badge_cls = "badge-needs-work"
    elif "different" in v_lower:
        badge_cls = "badge-different"
    else:
        badge_cls = "badge-awaiting"

    ac_val = f"{acoustic_cost:.3f}" if acoustic_cost is not None else "0.000"
    dur_val = f"{duration_ratio:.2f}x" if duration_ratio is not None else "1.00x"
    flag_val = str(flagged_count) if flagged_count is not None else "0"

    return (
        '<div class="score-card">'
        '<div class="score-card-header">'
        '<span class="score-card-tag">Pronunciation Assessment</span>'
        f'<span class="badge {badge_cls}">{html.escape(verdict_clean)}</span>'
        '</div>'
        '<div class="score-main-row">'
        f'<div class="score-big-num">{pct:.0f}<span>%</span></div>'
        '<div class="score-metrics-grid">'
        '<div class="score-metric-item">'
        '<span class="metric-label">Acoustic Cost</span>'
        f'<span class="metric-value">{ac_val}</span>'
        '</div>'
        '<div class="score-metric-item">'
        '<span class="metric-label">Duration Ratio</span>'
        f'<span class="metric-value">{dur_val}</span>'
        '</div>'
        '<div class="score-metric-item">'
        '<span class="metric-label">Flagged Spans</span>'
        f'<span class="metric-value">{flag_val}</span>'
        '</div>'
        '</div>'
        '</div>'
        '</div>'
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
    """Render compact, high-density rectangular readout for verse classification."""
    val = predicted if predicted else verse
    if not val or val.lower() == "awaiting input":
        val_display = "Awaiting Audio"
        badge_cls = "badge-awaiting"
        status_label = "Ready"
    else:
        val_display = val
        is_warn = is_low_confidence or low_conf
        badge_cls = "badge-warning" if is_warn else "badge-confident"
        status_label = "Low Confidence" if is_warn else "Confident"

    loglik_str = f"{top_loglik:.1f}" if top_loglik is not None else "0.0"

    return (
        '<div class="score-card">'
        '<div class="score-card-header">'
        '<span class="score-card-tag">Recitation Classifier</span>'
        f'<span class="badge {badge_cls}">{status_label}</span>'
        '</div>'
        '<div class="score-main-row">'
        f'<div class="score-big-num" style="font-size: 1.45rem; letter-spacing: -0.01em;">{html.escape(str(val_display))}</div>'
        '<div class="score-metrics-grid" style="grid-template-columns: repeat(2, 1fr);">'
        '<div class="score-metric-item">'
        '<span class="metric-label">Confidence Margin</span>'
        f'<span class="metric-value">{margin:.1f}</span>'
        '</div>'
        '<div class="score-metric-item">'
        '<span class="metric-label">Top Log-Likelihood</span>'
        f'<span class="metric-value">{loglik_str}</span>'
        '</div>'
        '</div>'
        '</div>'
        '</div>'
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
