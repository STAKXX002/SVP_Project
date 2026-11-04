"""Visual styling layer for the Speech and Video Processing coursework demo.

Material Design 3 (M3 / Material You) Dark Theme:
Tonal surface containers (#111318, #171a1f, #1c2026, #22262e), primary cyan accent
(#80d4f6) with on-primary (#003546) and container (#004d64), pill-shaped interactive
controls (24px radius), soft organic card radii (20px), and clean typography.
"""

from __future__ import annotations

import html
from typing import Any, List, Optional, Tuple

import gradio as gr

FORCE_DARK_JS = "() => { document.body.classList.add('dark'); }"


def make_theme() -> gr.themes.Base:
    """Create a Material Design 3 dark theme with tonal surfaces and cyan primary."""
    base = gr.themes.Base(
        primary_hue=gr.themes.colors.cyan,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.neutral,
        radius_size=gr.themes.sizes.radius_lg,
    )
    return base.set(
        body_background_fill="#111318",
        body_background_fill_dark="#111318",
        body_text_color="#e2e2e9",
        body_text_color_dark="#e2e2e9",
        block_background_fill="#1c2026",
        block_background_fill_dark="#1c2026",
        block_border_color="rgba(142, 145, 153, 0.16)",
        block_border_color_dark="rgba(142, 145, 153, 0.16)",
        block_border_width="1px",
        block_shadow="0 1px 3px 1px rgba(0, 0, 0, 0.15), 0 1px 2px 0 rgba(0, 0, 0, 0.30)",
        block_radius="*radius_lg",
        block_label_background_fill="#22262e",
        block_label_background_fill_dark="#22262e",
        block_label_text_color="#80d4f6",
        block_label_text_color_dark="#80d4f6",
        block_title_text_color="#e2e2e9",
        block_title_text_color_dark="#e2e2e9",
        input_background_fill="#171a1f",
        input_background_fill_dark="#171a1f",
        input_border_color="rgba(142, 145, 153, 0.22)",
        input_border_color_dark="rgba(142, 145, 153, 0.22)",
        input_radius="*radius_md",
        button_primary_background_fill="#80d4f6",
        button_primary_background_fill_dark="#80d4f6",
        button_primary_background_fill_hover="#67d3ff",
        button_primary_background_fill_hover_dark="#67d3ff",
        button_primary_text_color="#003546",
        button_primary_text_color_dark="#003546",
        button_primary_border_color="#80d4f6",
        button_primary_border_color_dark="#80d4f6",
        button_secondary_background_fill="#334a53",
        button_secondary_background_fill_dark="#334a53",
        button_secondary_background_fill_hover="#3e5964",
        button_secondary_background_fill_hover_dark="#3e5964",
        button_secondary_text_color="#cfe6f2",
        button_secondary_text_color_dark="#cfe6f2",
        button_secondary_border_color="rgba(142, 145, 153, 0.20)",
        button_secondary_border_color_dark="rgba(142, 145, 153, 0.20)",
    )


CSS = """
:root {
  --md-sys-color-background: #111318;
  --md-sys-color-on-background: #e2e2e9;
  --md-sys-color-surface: #111318;
  --md-sys-color-surface-dim: #0b0d11;
  --md-sys-color-surface-container-lowest: #0c0e12;
  --md-sys-color-surface-container-low: #171a1f;
  --md-sys-color-surface-container: #1c2026;
  --md-sys-color-surface-container-high: #22262e;
  --md-sys-color-surface-container-highest: #2a2f38;
  --md-sys-color-on-surface: #e2e2e9;
  --md-sys-color-on-surface-variant: #c2c7ce;
  --md-sys-color-primary: #80d4f6;
  --md-sys-color-on-primary: #003546;
  --md-sys-color-primary-container: #004d64;
  --md-sys-color-on-primary-container: #bee9ff;
  --md-sys-color-secondary: #b3cad5;
  --md-sys-color-secondary-container: #334a53;
  --md-sys-color-on-secondary-container: #cfe6f2;
  --md-sys-color-tertiary: #c3c3eb;
  --md-sys-color-outline: rgba(142, 145, 153, 0.22);
  --md-sys-color-outline-variant: rgba(142, 145, 153, 0.12);
  --md-font-sans: "Roboto", "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
  --md-font-mono: "Roboto Mono", "JetBrains Mono", Consolas, monospace;
}

body, .gradio-container {
  background-color: var(--md-sys-color-background) !important;
  color: var(--md-sys-color-on-background) !important;
  font-family: var(--md-font-sans) !important;
  max-width: 1420px !important;
  margin: 0 auto !important;
  padding: 14px 20px !important;
}

footer { display: none !important; }

/* M3 Soft Scrollbar */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}
::-webkit-scrollbar-track {
  background: var(--md-sys-color-surface-dim);
}
::-webkit-scrollbar-thumb {
  background: var(--md-sys-color-surface-container-high);
  border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
  background: var(--md-sys-color-outline);
}

/* ------------------------------------------------------------- */
/* M3 TOP APP BAR / HERO                                         */
/* ------------------------------------------------------------- */
.m3-top-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 16px;
  background: var(--md-sys-color-surface-container);
  border: 1px solid var(--md-sys-color-outline-variant);
  border-radius: 24px;
  padding: 18px 26px;
  margin-bottom: 16px;
  box-shadow: 0 1px 3px 1px rgba(0, 0, 0, 0.15);
}
.m3-title-row {
  display: flex;
  align-items: center;
  gap: 12px;
}
.m3-status-dot {
  width: 10px;
  height: 10px;
  background: var(--md-sys-color-primary);
  border-radius: 50%;
  box-shadow: 0 0 10px var(--md-sys-color-primary);
  flex-shrink: 0;
}
.m3-title {
  font-size: 1.35rem;
  font-weight: 600;
  letter-spacing: -0.01em;
  color: var(--md-sys-color-on-surface);
  margin: 0;
}
.m3-badge-pill {
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--md-sys-color-on-primary-container);
  background: var(--md-sys-color-primary-container);
  border-radius: 12px;
  padding: 3px 10px;
  letter-spacing: 0.02em;
}
.m3-sub {
  font-size: 0.84rem;
  color: var(--md-sys-color-on-surface-variant);
  margin: 4px 0 0 0;
}
.m3-chips-rack {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}
.m3-assist-chip {
  background: var(--md-sys-color-surface-container-high);
  border: 1px solid var(--md-sys-color-outline-variant);
  border-radius: 16px;
  padding: 6px 14px;
  font-size: 0.78rem;
  font-weight: 500;
  color: var(--md-sys-color-on-surface-variant);
  transition: all 0.2s ease;
}
.m3-assist-chip:hover {
  background: var(--md-sys-color-surface-container-highest);
  color: var(--md-sys-color-on-surface);
}
.m3-assist-chip b {
  color: var(--md-sys-color-primary);
  font-weight: 600;
  margin-left: 2px;
}

/* ------------------------------------------------------------- */
/* M3 PRIMARY NAVIGATION TABS (PILL INDICATOR)                   */
/* ------------------------------------------------------------- */
div[role="tablist"] {
  display: flex !important;
  background: var(--md-sys-color-surface-container-low) !important;
  border: 1px solid var(--md-sys-color-outline-variant) !important;
  border-radius: 28px !important;
  padding: 5px !important;
  gap: 6px !important;
  margin-bottom: 18px !important;
}
button[role="tab"] {
  flex: 1 !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  padding: 10px 18px !important;
  font-family: var(--md-font-sans) !important;
  font-size: 0.86rem !important;
  font-weight: 500 !important;
  color: var(--md-sys-color-on-surface-variant) !important;
  background: transparent !important;
  border: none !important;
  border-radius: 24px !important;
  transition: all 0.2s cubic-bezier(0.2, 0.0, 0, 1.0) !important;
}
button[role="tab"]:hover {
  color: var(--md-sys-color-on-surface) !important;
  background: var(--md-sys-color-surface-container-high) !important;
}
button[role="tab"][aria-selected="true"] {
  color: var(--md-sys-color-on-primary-container) !important;
  background: var(--md-sys-color-primary-container) !important;
  font-weight: 600 !important;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.20) !important;
}

/* ------------------------------------------------------------- */
/* M3 BUTTONS (FILLED & TONAL PILLS)                             */
/* ------------------------------------------------------------- */
button.primary {
  background: var(--md-sys-color-primary) !important;
  color: var(--md-sys-color-on-primary) !important;
  border: none !important;
  font-family: var(--md-font-sans) !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  letter-spacing: 0.01em !important;
  border-radius: 24px !important;
  padding: 10px 22px !important;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25) !important;
  transition: all 0.2s cubic-bezier(0.2, 0.0, 0, 1.0) !important;
}
button.primary:hover {
  background: #97e0ff !important;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.35) !important;
  transform: translateY(-1px);
}
button.secondary {
  background: var(--md-sys-color-secondary-container) !important;
  color: var(--md-sys-color-on-secondary-container) !important;
  border: 1px solid var(--md-sys-color-outline-variant) !important;
  font-family: var(--md-font-sans) !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  letter-spacing: 0.01em !important;
  border-radius: 24px !important;
  padding: 10px 22px !important;
  transition: all 0.2s cubic-bezier(0.2, 0.0, 0, 1.0) !important;
}
button.secondary:hover {
  background: #3e5964 !important;
  color: #ffffff !important;
  transform: translateY(-1px);
}

/* ------------------------------------------------------------- */
/* M3 ELEVATED CARDS & CONTAINERS                                */
/* ------------------------------------------------------------- */
.m3-card {
  background: var(--md-sys-color-surface-container);
  border: 1px solid var(--md-sys-color-outline-variant);
  border-radius: 20px;
  padding: 18px 22px;
  height: 100%;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  box-shadow: 0 1px 3px 1px rgba(0, 0, 0, 0.15);
  transition: border-color 0.2s ease;
}
.m3-card:hover {
  border-color: var(--md-sys-color-outline);
}
.m3-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}
.m3-card-label {
  font-size: 0.74rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  color: var(--md-sys-color-on-surface-variant);
  text-transform: uppercase;
}
.m3-score-main {
  display: flex;
  align-items: center;
  gap: 18px;
  margin-top: 4px;
}
.m3-big-number {
  font-size: 2.6rem;
  font-weight: 700;
  color: var(--md-sys-color-primary);
  font-family: var(--md-font-sans);
  line-height: 1;
}
.m3-big-number span.unit {
  font-size: 1.3rem;
  color: var(--md-sys-color-on-surface-variant);
  margin-left: 2px;
}
.m3-metric-chips-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  flex: 1;
}
.m3-metric-tile {
  background: var(--md-sys-color-surface-container-high);
  border: 1px solid var(--md-sys-color-outline-variant);
  border-radius: 14px;
  padding: 8px 10px;
  display: flex;
  flex-direction: column;
}
.m3-metric-label {
  font-size: 0.68rem;
  font-weight: 500;
  color: var(--md-sys-color-on-surface-variant);
}
.m3-metric-val {
  font-size: 0.90rem;
  font-weight: 600;
  font-family: var(--md-font-mono);
  color: var(--md-sys-color-on-surface);
  margin-top: 2px;
}
.m3-calib-subtext {
  font-size: 0.72rem;
  color: var(--md-sys-color-on-surface-variant);
  margin-top: 10px;
  padding-top: 8px;
  border-top: 1px solid var(--md-sys-color-outline-variant);
}

/* M3 Tonal Status Pills */
.m3-badge {
  font-size: 0.74rem;
  font-weight: 600;
  border-radius: 16px;
  padding: 3px 12px;
  letter-spacing: 0.02em;
}
.m3-badge-excellent {
  background: #0f381e;
  color: #78dc96;
  border: 1px solid rgba(120, 220, 150, 0.30);
}
.m3-badge-good {
  background: var(--md-sys-color-primary-container);
  color: var(--md-sys-color-on-primary-container);
  border: 1px solid rgba(190, 233, 255, 0.30);
}
.m3-badge-needs-work {
  background: #3c2f00;
  color: #ffe082;
  border: 1px solid rgba(255, 224, 130, 0.30);
}
.m3-badge-different {
  background: #410002;
  color: #ffb4ab;
  border: 1px solid rgba(255, 180, 171, 0.30);
}
.m3-badge-awaiting {
  background: var(--md-sys-color-surface-container-high);
  color: var(--md-sys-color-on-surface-variant);
  border: 1px solid var(--md-sys-color-outline-variant);
}
.m3-badge-confident {
  background: var(--md-sys-color-primary-container);
  color: var(--md-sys-color-on-primary-container);
  border: 1px solid rgba(190, 233, 255, 0.30);
}
.m3-badge-warning {
  background: #3c2f00;
  color: #ffe082;
  border: 1px solid rgba(255, 224, 130, 0.30);
}

/* ------------------------------------------------------------- */
/* M3 NATIVE VIDEO CONTAINER                                     */
/* ------------------------------------------------------------- */
.m3-video-card {
  background: var(--md-sys-color-surface-container);
  border: 1px solid var(--md-sys-color-outline-variant);
  border-radius: 20px;
  overflow: hidden;
  box-shadow: 0 1px 3px 1px rgba(0, 0, 0, 0.15);
  margin-bottom: 8px;
}
.m3-video-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 18px;
  background: var(--md-sys-color-surface-container-high);
  border-bottom: 1px solid var(--md-sys-color-outline-variant);
}
.m3-video-title {
  font-size: 0.84rem;
  font-weight: 600;
  color: var(--md-sys-color-on-surface);
  display: flex;
  align-items: center;
  gap: 8px;
}
.m3-video-dot {
  width: 8px;
  height: 8px;
  background: var(--md-sys-color-primary);
  border-radius: 50%;
  display: inline-block;
}
.m3-video-pill {
  font-size: 0.70rem;
  font-weight: 600;
  color: var(--md-sys-color-on-secondary-container);
  background: var(--md-sys-color-secondary-container);
  border-radius: 12px;
  padding: 2px 10px;
}
.m3-video-stage {
  background: #000000;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 4px;
}
.m3-video-stage video {
  width: 100%;
  max-height: 270px;
  background: #000000;
  border-radius: 14px;
  display: block;
}
.m3-video-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 18px;
  background: var(--md-sys-color-surface-container-low);
  font-size: 0.76rem;
  color: var(--md-sys-color-on-surface-variant);
}
.m3-dl-pill {
  color: var(--md-sys-color-on-primary-container) !important;
  background: var(--md-sys-color-primary-container) !important;
  text-decoration: none !important;
  font-weight: 600;
  border-radius: 12px;
  padding: 3px 12px;
  transition: all 0.2s ease;
}
.m3-dl-pill:hover {
  background: var(--md-sys-color-primary) !important;
  color: var(--md-sys-color-on-primary) !important;
}

/* ------------------------------------------------------------- */
/* M3 SPECIFICATION CARDS (TAB 4)                                */
/* ------------------------------------------------------------- */
.m3-spec-card {
  background: var(--md-sys-color-surface-container);
  border: 1px solid var(--md-sys-color-outline-variant);
  border-radius: 20px;
  padding: 20px 24px;
  margin-bottom: 14px;
  box-shadow: 0 1px 3px 1px rgba(0, 0, 0, 0.12);
}
.m3-spec-title {
  color: var(--md-sys-color-primary);
  font-size: 1.05rem;
  font-weight: 600;
  letter-spacing: -0.01em;
  margin-bottom: 10px;
  border-bottom: 1px solid var(--md-sys-color-outline-variant);
  padding-bottom: 8px;
}
.m3-spec-card p {
  font-size: 0.84rem;
  color: var(--md-sys-color-on-surface-variant);
  line-height: 1.5;
  margin: 6px 0 12px 0;
}
.m3-spec-card ul {
  margin: 0;
  padding-left: 18px;
}
.m3-spec-card li {
  font-size: 0.82rem;
  color: var(--md-sys-color-on-surface-variant);
  line-height: 1.6;
  margin-bottom: 8px;
}
.m3-spec-card li b {
  color: var(--md-sys-color-on-surface);
}

/* ------------------------------------------------------------- */
/* FORM CONTROLS & FIGURES IN M3                                 */
/* ------------------------------------------------------------- */
input, select, textarea {
  background: var(--md-sys-color-surface-container-low) !important;
  color: var(--md-sys-color-on-surface) !important;
  border: 1px solid var(--md-sys-color-outline) !important;
  border-radius: 12px !important;
  font-size: 0.88rem !important;
  padding: 8px 14px !important;
}
input:focus, select:focus, textarea:focus {
  border-color: var(--md-sys-color-primary) !important;
  box-shadow: 0 0 0 2px rgba(128, 212, 246, 0.25) !important;
}

/* Figures & Images */
.fig img {
  background: var(--md-sys-color-surface-container-low) !important;
  border: 1px solid var(--md-sys-color-outline-variant) !important;
  border-radius: 16px !important;
  padding: 4px !important;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15) !important;
}

/* Data Tables */
table {
  border: 1px solid var(--md-sys-color-outline-variant) !important;
  background: var(--md-sys-color-surface-container) !important;
  color: var(--md-sys-color-on-surface) !important;
  border-radius: 14px !important;
  overflow: hidden !important;
  font-size: 0.84rem !important;
}
th {
  background: var(--md-sys-color-surface-container-high) !important;
  color: var(--md-sys-color-primary) !important;
  border-bottom: 1px solid var(--md-sys-color-outline-variant) !important;
  font-weight: 600 !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.02em !important;
  padding: 8px 12px !important;
}
td {
  border-bottom: 1px solid var(--md-sys-color-outline-variant) !important;
  padding: 8px 12px !important;
  font-family: var(--md-font-mono) !important;
}

/* Notices */
.notice {
  border: 1px solid var(--md-sys-color-outline-variant);
  background: var(--md-sys-color-surface-container);
  border-radius: 14px;
  color: var(--md-sys-color-on-surface);
  padding: 12px 16px;
  font-size: 0.82rem;
  margin-top: 6px;
}
.notice b { color: var(--md-sys-color-primary); }
.notice code {
  background: var(--md-sys-color-surface-container-high);
  color: var(--md-sys-color-on-primary-container);
  border: 1px solid var(--md-sys-color-outline-variant);
  border-radius: 6px;
  padding: 2px 6px;
  font-family: var(--md-font-mono);
  font-size: 0.78rem;
}
"""


def hero_html(title: str, subtitle: str, chips: List[Tuple[str, str]]) -> str:
    """Render Material Design 3 top app bar with status indicator and assist chips."""
    chip_spans = "".join(
        f'<span class="m3-assist-chip">{html.escape(k)}: <b>{html.escape(v)}</b></span>'
        for k, v in chips
    )
    return (
        '<div class="m3-top-bar">'
        '<div class="m3-left">'
        '<div class="m3-title-row">'
        '<span class="m3-status-dot" title="Multimedia Processing Ready"></span>'
        f'<h1 class="m3-title">{html.escape(title)}</h1>'
        '<span class="m3-badge-pill">Material 3</span>'
        '</div>'
        f'<p class="m3-sub">{html.escape(subtitle)}</p>'
        '</div>'
        f'<div class="m3-chips-rack">{chip_spans}</div>'
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
    """Render Material Design 3 elevated score card."""
    pct = max(0.0, min(100.0, float(similarity)))
    verdict_clean = str(verdict).strip().title()

    v_lower = verdict_clean.lower()
    if "excellent" in v_lower:
        badge_cls = "m3-badge-excellent"
    elif "good" in v_lower:
        badge_cls = "m3-badge-good"
    elif "work" in v_lower:
        badge_cls = "m3-badge-needs-work"
    elif "different" in v_lower:
        badge_cls = "m3-badge-different"
    else:
        badge_cls = "m3-badge-awaiting"

    ac_val = f"{acoustic_cost:.3f}" if acoustic_cost is not None else "0.000"
    dur_val = f"{duration_ratio:.2f}x" if duration_ratio is not None else "1.00x"
    flag_val = str(flagged_count) if flagged_count is not None else "0"

    return (
        '<div class="m3-card">'
        '<div class="m3-card-header">'
        '<span class="m3-card-label">Pronunciation Assessment</span>'
        f'<span class="m3-badge {badge_cls}">{html.escape(verdict_clean)}</span>'
        '</div>'
        '<div class="m3-score-main">'
        f'<div class="m3-big-number">{pct:.0f}<span class="unit">%</span></div>'
        '<div class="m3-metric-chips-grid">'
        '<div class="m3-metric-tile">'
        '<span class="m3-metric-label">Acoustic Cost</span>'
        f'<span class="m3-metric-val">{ac_val}</span>'
        '</div>'
        '<div class="m3-metric-tile">'
        '<span class="m3-metric-label">Duration Ratio</span>'
        f'<span class="m3-metric-val">{dur_val}</span>'
        '</div>'
        '<div class="m3-metric-tile">'
        '<span class="m3-metric-label">Flagged Spans</span>'
        f'<span class="m3-metric-val">{flag_val}</span>'
        '</div>'
        '</div>'
        '</div>'
        '<div class="m3-calib-subtext">Acoustic Calibration: Adaptive Cross-Speaker Normalization Active</div>'
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
    """Render Material Design 3 classification card."""
    val = predicted if predicted else verse
    if not val or val.lower() == "awaiting input":
        val_display = "Awaiting Audio"
        badge_cls = "m3-badge-awaiting"
        status_label = "Ready"
    else:
        val_display = val
        is_warn = is_low_confidence or low_conf
        badge_cls = "m3-badge-warning" if is_warn else "m3-badge-confident"
        status_label = "Low Confidence" if is_warn else "Confident"

    loglik_str = f"{top_loglik:.1f}" if top_loglik is not None else "0.0"

    return (
        '<div class="m3-card">'
        '<div class="m3-card-header">'
        '<span class="m3-card-label">Recitation Classifier</span>'
        f'<span class="m3-badge {badge_cls}">{status_label}</span>'
        '</div>'
        '<div class="m3-score-main">'
        f'<div class="m3-big-number" style="font-size: 1.6rem;">{html.escape(str(val_display))}</div>'
        '<div class="m3-metric-chips-grid" style="grid-template-columns: repeat(2, 1fr);">'
        '<div class="m3-metric-tile">'
        '<span class="m3-metric-label">Confidence Margin</span>'
        f'<span class="m3-metric-val">{margin:.1f}</span>'
        '</div>'
        '<div class="m3-metric-tile">'
        '<span class="m3-metric-label">Top Log-Likelihood</span>'
        f'<span class="m3-metric-val">{loglik_str}</span>'
        '</div>'
        '</div>'
        '</div>'
        '<div class="m3-calib-subtext">HMM Topology: Left-to-Right 5-State Continuous Density Model</div>'
        '</div>'
    )


def video_player_html(
    video_url: str,
    poster_url: str,
    title: str,
    tag: str,
    footer_info: str,
) -> str:
    """Render Material Design 3 framed native video player."""
    return (
        '<div class="m3-video-card">'
        '<div class="m3-video-header">'
        f'<div class="m3-video-title"><span class="m3-video-dot"></span> {html.escape(title)}</div>'
        f'<span class="m3-video-pill">{html.escape(tag)}</span>'
        '</div>'
        '<div class="m3-video-stage">'
        f'<video controls preload="metadata" poster="{html.escape(poster_url)}" playsinline>'
        f'<source src="{html.escape(video_url)}" type="video/mp4">'
        'Your browser does not support HTML5 video.'
        '</video>'
        '</div>'
        '<div class="m3-video-footer">'
        f'<span>{html.escape(footer_info)}</span>'
        f'<a href="{html.escape(video_url)}" download class="m3-dl-pill">Download MP4</a>'
        '</div>'
        '</div>'
    )


def notice_html(title: str, body: str = "") -> str:
    """Render Material Design 3 notification card."""
    body_part = f"<br>{body}" if body else ""
    return f'<div class="notice"><b>{html.escape(title)}</b>{body_part}</div>'


def stat_chips_html(pairs: List[Tuple[str, str]]) -> str:
    """Render Material Design 3 assist chips."""
    inner = "".join(
        f'<span class="m3-assist-chip">{html.escape(k)}: <b>{html.escape(v)}</b></span>'
        for k, v in pairs
    )
    return f'<div class="m3-chips-rack">{inner}</div>'
