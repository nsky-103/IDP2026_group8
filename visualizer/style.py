"""Page styling and small display helpers shared by the panels."""

from __future__ import annotations

import html

import streamlit as st

CSS = """
<style>
.block-container { padding-top: 3.2rem; padding-bottom: 0; max-width: 100%; }
.sv-title { font-size: 1.25rem; font-weight: 700; }
.sv-chip {
    float: right; font-family: monospace; font-size: 0.8rem;
    padding: 0.3rem 0.7rem; border: 1px solid rgba(128,128,128,0.35); border-radius: 6px;
}
.sv-section {
    display: flex; justify-content: space-between; margin: 0.9rem 0 0.3rem;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; opacity: 0.65;
}
.sv-card { background: #18222d; color: #eef2f6; border-radius: 10px; padding: 1rem; margin-bottom: 0.8rem; }
.sv-card-label { font-size: 0.75rem; opacity: 0.75; text-transform: uppercase; letter-spacing: 0.06em; }
.sv-card-value { font-size: 1.8rem; font-weight: 700; }
.sv-card-note { font-size: 0.75rem; opacity: 0.65; }
/* Left / centre / right panels: each fits the window and scrolls on its own */
[data-testid="stColumn"]:has(.st-key-left_panel),
[data-testid="stColumn"]:has(.st-key-center_panel),
[data-testid="stColumn"]:has(.st-key-right_panel) {
    max-height: calc(100vh - 4.5rem); overflow-y: auto; overscroll-behavior: contain;
}
/* Heatmap box: tighter than the default bordered container */
.st-key-heatmap_box { padding: 0.5rem 0.6rem; }
.st-key-heatmap_box .sv-section { margin-top: 0; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def section(title: str, note: str = "") -> None:
    """Small uppercase section heading with an optional note on the right."""
    st.markdown(
        f'<div class="sv-section"><span>{html.escape(title)}</span>'
        f"<span>{html.escape(note)}</span></div>",
        unsafe_allow_html=True,
    )


def card(label: str, value: str = "", notes: list[str] = ()) -> None:
    """Dark summary card (text is escaped)."""
    parts = [f'<div class="sv-card-label">{html.escape(label)}</div>']
    if value:
        parts.append(f'<div class="sv-card-value">{html.escape(value)}</div>')
    parts += [f'<div class="sv-card-note">{html.escape(n)}</div>' for n in notes]
    st.markdown(f'<div class="sv-card">{"".join(parts)}</div>', unsafe_allow_html=True)


def fmt_time(seconds: float) -> str:
    """65.2 -> '01:05.20'"""
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds - 60 * minutes:05.2f}"
