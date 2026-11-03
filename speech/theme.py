"""Visual styling layer for the Speech and Video Processing coursework demo.

Precision Studio Theme: Pure pitch black (#000000) base, razor-sharp surfaces (#050505),
and electric cyan (#00ffff) accents with high-contrast typography and hardware-inspired
segmented tab navigation.
"""

from __future__ import annotations

import html
from typing import Any, List, Optional, Tuple

import gradio as gr

FORCE_DARK_JS = "() => { document.body.classList.add('dark'); }"


def make_theme() -> gr.themes.Base:
    """Create a precision dark theme with pure black base and electric cyan accents."""
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
        block_background_fill="#050505",
        block_background_fill_dark="#050505",
        block_border_color="rgba(0, 255, 255, 0.20)",
        block_border_color_dark="rgba(0, 255, 255, 0.20)",
        block_border_width="1px",
        block_shadow="none",
        block_label_background_fill="#0a0a0a",
        block_label_background_fill_dark="#0a0a0a",
        block_label_text_color="#00ffff",
        block_label_text_color_dark="#00ffff",
        block_title_text_color="#ffffff",
        block_title_text_color_dark="#ffffff",
        input_background_fill="#000000",
        input_background_fill_dark="#000000",
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
        button_secondary_background_fill="#0a0a0a",
        button_secondary_background_fill_dark="#0a0a0a",
        button_secondary_background_fill_hover="#141414",
        button_secondary_background_fill_hover_dark="#141414",
        button_secondary_text_color="#00ffff",
        button_secondary_text_color_dark="#00ffff",
        button_secondary_border_color="rgba(0, 255, 255, 0.35)",
        button_secondary_border_color_dark="rgba(0, 255, 255, 0.35)",
    )


CSS = """
:root {
  --bg-pitch: #000000;
  --surface-panel: #050505;
  --surface-sub: #0a0a0a;
  --cyan-primary: #00ffff;
  --cyan-glow: rgba(0, 255, 255, 0.25);
  --cyan-border: rgba(0, 255, 255, 0.20);
  --cyan-border-bright: rgba(0, 255, 255, 0.50);
  --text-pure: #ffffff;
  --text-dim: #94a3b8;
  --text-muted: #64748b;
  --font-sans: "Inter", "Segoe UI", -apple-system, sans-serif;
  --font-mono: "JetBrains Mono", "SF Mono", Consolas, monospace;
}

body, .gradio-container {
  background-color: var(--bg-pitch) !important;
  color: var(--text-pure) !important;
  font-family: var(--font-sans) !important;
  max-width: 1420px !important;
  margin: 0 auto !important;
  padding: 10px 16px !important;
}

footer { display: none !important; }

/* Custom Sleek Scrollbar */
::-webkit-scrollbar {
  width: 5px;
  height: 5px;
}
::-webkit-scrollbar-track {
  background: #000000;
}
::-webkit-scrollbar-thumb {
  background: #18181b;
  border-radius: 2px;
}
::-webkit-scrollbar-thumb:hover {
  background: var(--cyan-primary);
}

/* ------------------------------------------------------------- */
/* TOP HUD MASTHEAD                                              */
/* ------------------------------------------------------------- */
.hud-masthead {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
  background: #050505;
  border: 1px solid var(--cyan-border);
  border-top: 2px solid var(--cyan-primary);
  border-radius: 4px;
  padding: 14px 20px;
  margin-bottom: 14px;
}
.hud-brand-group {
  display: flex;
  align-items: center;
  gap: 10px;
}
.hud-pulse-dot {
  width: 8px;
  height: 8px;
  background: var(--cyan-primary);
  border-radius: 50%;
  box-shadow: 0 0 10px var(--cyan-primary);
  flex-shrink: 0;
}
.hud-title {
  font-size: 1.25rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: #ffffff;
  margin: 0;
  text-transform: uppercase;
}
.hud-ver-tag {
  font-size: 0.68rem;
  font-family: var(--font-mono);
  color: #000000;
  background: var(--cyan-primary);
  border-radius: 2px;
  padding: 2px 6px;
  font-weight: 800;
  letter-spacing: 0.04em;
}
.hud-subtitle {
  font-size: 0.80rem;
  color: var(--text-dim);
  margin: 4px 0 0 0;
}
.hud-telemetry {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
}
.hud-telemetry-chip {
  background: #0a0a0a;
  border: 1px solid var(--cyan-border);
  border-radius: 3px;
  padding: 4px 9px;
  font-size: 0.72rem;
  font-family: var(--font-mono);
  color: var(--text-dim);
}
.hud-telemetry-chip b {
  color: var(--cyan-primary);
  font-weight: 700;
}

/* ------------------------------------------------------------- */
/* HARDWARE SEGMENTED TABS                                       */
/* ------------------------------------------------------------- */
div[role="tablist"] {
  display: flex !important;
  background: #050505 !important;
  border: 1px solid var(--cyan-border) !important;
  border-radius: 4px !important;
  padding: 4px !important;
  gap: 4px !important;
  margin-bottom: 16px !important;
}
button[role="tab"] {
  flex: 1 !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  padding: 10px 14px !important;
  font-family: var(--font-sans) !important;
  font-size: 0.82rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.05em !important;
  text-transform: uppercase !important;
  color: #64748b !important;
  background: transparent !important;
  border: 1px solid transparent !important;
  border-radius: 3px !important;
  transition: all 0.15s ease !important;
}
button[role="tab"]:hover {
  color: #ffffff !important;
  background: #0f0f0f !important;
  border-color: rgba(255, 255, 255, 0.15) !important;
}
button[role="tab"][aria-selected="true"] {
  color: #000000 !important;
  background: var(--cyan-primary) !important;
  border-color: var(--cyan-primary) !important;
  font-weight: 800 !important;
  box-shadow: 0 0 14px var(--cyan-glow) !important;
}

/* ------------------------------------------------------------- */
/* BUTTONS & CONTROLS                                            */
/* ------------------------------------------------------------- */
button.primary {
  background: var(--cyan-primary) !important;
  color: #000000 !important;
  border: 1px solid var(--cyan-primary) !important;
  font-family: var(--font-sans) !important;
  font-weight: 800 !important;
  font-size: 0.84rem !important;
  letter-spacing: 0.04em !important;
  text-transform: uppercase !important;
  border-radius: 4px !important;
  padding: 9px 16px !important;
  transition: all 0.15s ease !important;
}
button.primary:hover {
  background: #38bdf8 !important;
  border-color: #38bdf8 !important;
  box-shadow: 0 0 12px var(--cyan-glow) !important;
}
button.secondary {
  background: #0a0a0a !important;
  color: var(--cyan-primary) !important;
  border: 1px solid var(--cyan-border) !important;
  font-family: var(--font-sans) !important;
  font-weight: 700 !important;
  font-size: 0.84rem !important;
  letter-spacing: 0.04em !important;
  text-transform: uppercase !important;
  border-radius: 4px !important;
  padding: 9px 16px !important;
  transition: all 0.15s ease !important;
}
button.secondary:hover {
  background: #141414 !important;
  border-color: var(--cyan-primary) !important;
  color: #ffffff !important;
}

/* ------------------------------------------------------------- */
/* PRECISION INSTRUMENT SCORE DISPLAY                            */
/* ------------------------------------------------------------- */
.hud-score-display {
  background: #050505;
  border: 1px solid var(--cyan-border);
  border-top: 2px solid var(--cyan-primary);
  border-radius: 4px;
  padding: 14px 18px;
  height: 100%;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}
.hud-score-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}
.hud-label {
  font-size: 0.70rem;
  font-family: var(--font-mono);
  font-weight: 700;
  letter-spacing: 0.06em;
  color: var(--text-dim);
  text-transform: uppercase;
}
.hud-score-body {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-top: 4px;
}
.hud-big-metric {
  font-size: 2.3rem;
  font-weight: 900;
  font-family: var(--font-mono);
  color: #ffffff;
  line-height: 1;
}
.hud-big-metric span.hud-unit {
  font-size: 1.1rem;
  color: var(--cyan-primary);
  margin-left: 2px;
}
.hud-submetrics {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  flex: 1;
  border-left: 1px solid rgba(0, 255, 255, 0.18);
  padding-left: 14px;
}
.hud-subcell {
  display: flex;
  flex-direction: column;
}
.hud-sublabel {
  font-size: 0.66rem;
  color: var(--text-muted);
  font-weight: 600;
  text-transform: uppercase;
}
.hud-subvalue {
  font-size: 0.88rem;
  font-weight: 700;
  font-family: var(--font-mono);
  color: var(--cyan-primary);
  margin-top: 2px;
}
.hud-calib-tag {
  font-size: 0.68rem;
  font-family: var(--font-mono);
  color: var(--text-muted);
  margin-top: 8px;
  border-top: 1px dashed rgba(255, 255, 255, 0.08);
  padding-top: 6px;
}

/* Status Badges */
.badge-excellent {
  color: #000000;
  background: #22c55e;
  border: 1px solid #22c55e;
  font-weight: 800;
  font-size: 0.70rem;
  padding: 2px 7px;
  border-radius: 2px;
  letter-spacing: 0.04em;
}
.badge-good {
  color: #000000;
  background: var(--cyan-primary);
  border: 1px solid var(--cyan-primary);
  font-weight: 800;
  font-size: 0.70rem;
  padding: 2px 7px;
  border-radius: 2px;
  letter-spacing: 0.04em;
}
.badge-needs-work {
  color: #000000;
  background: #fbbf24;
  border: 1px solid #fbbf24;
  font-weight: 800;
  font-size: 0.70rem;
  padding: 2px 7px;
  border-radius: 2px;
  letter-spacing: 0.04em;
}
.badge-different {
  color: #ffffff;
  background: #dc2626;
  border: 1px solid #dc2626;
  font-weight: 800;
  font-size: 0.70rem;
  padding: 2px 7px;
  border-radius: 2px;
  letter-spacing: 0.04em;
}
.badge-awaiting {
  color: var(--text-dim);
  background: #141414;
  border: 1px solid #262626;
  font-weight: 700;
  font-size: 0.70rem;
  padding: 2px 7px;
  border-radius: 2px;
  letter-spacing: 0.04em;
}
.badge-confident {
  color: #000000;
  background: var(--cyan-primary);
  border: 1px solid var(--cyan-primary);
  font-weight: 800;
  font-size: 0.70rem;
  padding: 2px 7px;
  border-radius: 2px;
  letter-spacing: 0.04em;
}
.badge-warning {
  color: #000000;
  background: #fbbf24;
  border: 1px solid #fbbf24;
  font-weight: 800;
  font-size: 0.70rem;
  padding: 2px 7px;
  border-radius: 2px;
  letter-spacing: 0.04em;
}

/* ------------------------------------------------------------- */
/* NATIVE HTML5 STUDIO VIDEO PLAYER                              */
/* ------------------------------------------------------------- */
.studio-video-card {
  background: #050505;
  border: 1px solid var(--cyan-border);
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 6px;
}
.studio-video-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  background: #0a0a0a;
  border-bottom: 1px solid var(--cyan-border);
}
.studio-video-title {
  font-size: 0.76rem;
  font-weight: 700;
  font-family: var(--font-mono);
  color: #ffffff;
  display: flex;
  align-items: center;
  gap: 8px;
  text-transform: uppercase;
}
.studio-dot {
  width: 6px;
  height: 6px;
  background: var(--cyan-primary);
  border-radius: 50%;
  display: inline-block;
}
.studio-tag {
  font-size: 0.68rem;
  font-family: var(--font-mono);
  color: var(--cyan-primary);
  background: rgba(0, 255, 255, 0.10);
  border: 1px solid rgba(0, 255, 255, 0.25);
  border-radius: 2px;
  padding: 1px 6px;
}
.studio-video-frame {
  background: #000000;
  display: flex;
  justify-content: center;
  align-items: center;
}
.studio-video-frame video {
  width: 100%;
  max-height: 270px;
  background: #000000;
  display: block;
}
.studio-video-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 12px;
  background: #080808;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
  font-size: 0.72rem;
  font-family: var(--font-mono);
  color: var(--text-muted);
}
.studio-dl-link {
  color: var(--cyan-primary) !important;
  text-decoration: none !important;
  font-weight: 700;
  transition: all 0.15s ease;
}
.studio-dl-link:hover {
  text-decoration: underline !important;
  color: #ffffff !important;
}

/* ------------------------------------------------------------- */
/* SPECIFICATION CARDS (TAB 4)                                   */
/* ------------------------------------------------------------- */
.spec-card {
  background: #050505;
  border: 1px solid var(--cyan-border);
  border-radius: 4px;
  padding: 16px 20px;
  margin-bottom: 12px;
}
.spec-card-title {
  color: var(--cyan-primary);
  font-size: 0.95rem;
  font-weight: 800;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  margin-bottom: 8px;
  border-bottom: 1px solid rgba(0, 255, 255, 0.15);
  padding-bottom: 6px;
}
.spec-card p {
  font-size: 0.82rem;
  color: var(--text-dim);
  line-height: 1.5;
  margin: 6px 0 10px 0;
}
.spec-card ul {
  margin: 0;
  padding-left: 18px;
}
.spec-card li {
  font-size: 0.80rem;
  color: var(--text-dim);
  line-height: 1.6;
  margin-bottom: 6px;
}
.spec-card li b {
  color: #ffffff;
}

/* Figures & Images */
.fig img {
  background: #050505 !important;
  border: 1px solid var(--cyan-border) !important;
  border-radius: 4px !important;
  padding: 2px !important;
}

/* Tables */
table {
  border: 1px solid var(--cyan-border) !important;
  background: #050505 !important;
  color: #ffffff !important;
  border-radius: 4px !important;
  font-size: 0.82rem !important;
}
th {
  background: #0a0a0a !important;
  color: var(--cyan-primary) !important;
  border-bottom: 1px solid var(--cyan-border) !important;
  font-weight: 700 !important;
  font-size: 0.74rem !important;
  letter-spacing: 0.04em !important;
  text-transform: uppercase !important;
  padding: 6px 10px !important;
}
td {
  border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important;
  padding: 6px 10px !important;
  font-family: var(--font-mono) !important;
}

/* Notices */
.notice {
  border: 1px solid var(--cyan-border);
  background: #050505;
  border-radius: 4px;
  color: var(--cyan-primary);
  padding: 10px 14px;
  font-size: 0.80rem;
  font-family: var(--font-mono);
  margin-top: 6px;
}
.notice b { color: #ffffff; }
.notice code {
  background: #0a0a0a;
  color: var(--cyan-primary);
  border: 1px solid var(--cyan-border);
  border-radius: 2px;
  padding: 1px 5px;
}
"""


def hero_html(title: str, subtitle: str, chips: List[Tuple[str, str]]) -> str:
    """Render minimal technical masthead with cyan pulse indicator and system telemetry."""
    chip_spans = "".join(
        f'<span class="hud-telemetry-chip">{html.escape(k)}: <b>{html.escape(v)}</b></span>'
        for k, v in chips
    )
    return (
        '<div class="hud-masthead">'
        '<div class="hud-left">'
        '<div class="hud-brand-group">'
        '<span class="hud-pulse-dot" title="Acoustic & Vision Pipelines Active"></span>'
        f'<span class="hud-title">{html.escape(title)}</span>'
        '<span class="hud-ver-tag">STUDIO PRO</span>'
        '</div>'
        f'<p class="hud-subtitle">{html.escape(subtitle)}</p>'
        '</div>'
        f'<div class="hud-telemetry">{chip_spans}</div>'
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
    """Render precision instrument readout for pronunciation scoring."""
    pct = max(0.0, min(100.0, float(similarity)))
    verdict_clean = str(verdict).strip().title()

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
        '<div class="hud-score-display">'
        '<div class="hud-score-header">'
        '<span class="hud-label">PRONUNCIATION ASSESSMENT</span>'
        f'<span class="badge {badge_cls}">{html.escape(verdict_clean.upper())}</span>'
        '</div>'
        '<div class="hud-score-body">'
        f'<div class="hud-big-metric">{pct:.0f}<span class="hud-unit">%</span></div>'
        '<div class="hud-submetrics">'
        '<div class="hud-subcell">'
        '<span class="hud-sublabel">Acoustic Cost</span>'
        f'<span class="hud-subvalue">{ac_val}</span>'
        '</div>'
        '<div class="hud-subcell">'
        '<span class="hud-sublabel">Duration Ratio</span>'
        f'<span class="hud-subvalue">{dur_val}</span>'
        '</div>'
        '<div class="hud-subcell">'
        '<span class="hud-sublabel">Flagged Spans</span>'
        f'<span class="hud-subvalue">{flag_val}</span>'
        '</div>'
        '</div>'
        '</div>'
        '<div class="hud-calib-tag">Calibration: Adaptive Cross-Speaker Normalization Active</div>'
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
    """Render precision instrument readout for verse classification."""
    val = predicted if predicted else verse
    if not val or val.lower() == "awaiting input":
        val_display = "Awaiting Audio"
        badge_cls = "badge-awaiting"
        status_label = "READY"
    else:
        val_display = val
        is_warn = is_low_confidence or low_conf
        badge_cls = "badge-warning" if is_warn else "badge-confident"
        status_label = "LOW CONFIDENCE" if is_warn else "CONFIDENT"

    loglik_str = f"{top_loglik:.1f}" if top_loglik is not None else "0.0"

    return (
        '<div class="hud-score-display">'
        '<div class="hud-score-header">'
        '<span class="hud-label">RECITATION CLASSIFIER</span>'
        f'<span class="badge {badge_cls}">{status_label}</span>'
        '</div>'
        '<div class="hud-score-body">'
        f'<div class="hud-big-metric" style="font-size: 1.45rem; letter-spacing: -0.01em;">{html.escape(str(val_display).upper())}</div>'
        '<div class="hud-submetrics" style="grid-template-columns: repeat(2, 1fr);">'
        '<div class="hud-subcell">'
        '<span class="hud-sublabel">Confidence Margin</span>'
        f'<span class="hud-subvalue">{margin:.1f}</span>'
        '</div>'
        '<div class="hud-subcell">'
        '<span class="hud-sublabel">Top Log-Likelihood</span>'
        f'<span class="hud-subvalue">{loglik_str}</span>'
        '</div>'
        '</div>'
        '</div>'
        '<div class="hud-calib-tag">Topology: Left-to-Right 5-State Continuous HMM</div>'
        '</div>'
    )


def video_player_html(
    video_url: str,
    poster_url: str,
    title: str,
    tag: str,
    footer_info: str,
) -> str:
    """Render framed native HTML5 video player with baked poster and controls."""
    return (
        '<div class="studio-video-card">'
        '<div class="studio-video-header">'
        f'<div class="studio-video-title"><span class="studio-dot"></span> {html.escape(title)}</div>'
        f'<span class="studio-tag">{html.escape(tag)}</span>'
        '</div>'
        '<div class="studio-video-frame">'
        f'<video controls preload="metadata" poster="{html.escape(poster_url)}" playsinline>'
        f'<source src="{html.escape(video_url)}" type="video/mp4">'
        'Your browser does not support HTML5 video.'
        '</video>'
        '</div>'
        '<div class="studio-video-footer">'
        f'<span>{html.escape(footer_info)}</span>'
        f'<a href="{html.escape(video_url)}" download class="studio-dl-link">Direct MP4 Link</a>'
        '</div>'
        '</div>'
    )


def notice_html(title: str, body: str = "") -> str:
    """Render technical notice card with cyan border."""
    body_part = f"<br>{body}" if body else ""
    return f'<div class="notice"><b>{html.escape(title)}</b>{body_part}</div>'


def stat_chips_html(pairs: List[Tuple[str, str]]) -> str:
    """Render technical chips."""
    inner = "".join(
        f'<span class="hud-telemetry-chip">{html.escape(k)}: <b>{html.escape(v)}</b></span>'
        for k, v in pairs
    )
    return f'<div class="hud-telemetry">{inner}</div>'
