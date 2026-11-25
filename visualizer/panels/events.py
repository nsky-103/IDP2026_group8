"""Right column: detection summary and event list (largest first).

Both are expandable so the Detection block can be collapsed when you only need
totals / the table.
"""

from __future__ import annotations

import streamlit as st

import state
from style import fmt_time, section

MAX_EVENTS_LISTED = 25
MAX_TABLE_ROWS = 500

# Not shown in the UI: size/amount estimates are unreliable from these signals alone.
_HIDDEN_EVENT_COLS = ("size_mm", "area")


def render(run: str, total_s: float) -> None:
    render_detection_totals(run)
    render_list(run, total_s)


def render_detection_totals(run: str) -> None:
    """Total detection metrics + full table — open/close as one block."""
    detection = state.detection(run)
    section("Detection")
    if detection is None:
        st.caption("Run **Detect peaks** to see totals and events.")
        return

    n_valid = int(detection.events["valid"].sum()) if "valid" in detection.events else 0
    n_all = len(detection.events)
    with st.expander(f"Detection summary · {n_valid} valid / {n_all} total", expanded=False):
        for name, value in detection.summary.items():
            if str(name).lower().startswith("amount"):
                continue  # hide size/amount estimates in the UI
            st.metric(name, value)
        table = detection.events.drop(
            columns=[c for c in _HIDDEN_EVENT_COLS if c in detection.events.columns],
            errors="ignore",
        )
        if len(table) > MAX_TABLE_ROWS:  # a huge table freezes the page
            st.caption(f"Showing {MAX_TABLE_ROWS} / {len(table)} rows (largest amplitudes).")
            table = table.nlargest(MAX_TABLE_ROWS, "amplitude")
        st.dataframe(table, width="stretch")


def render_list(run: str, total_s: float) -> None:
    events = state.accepted_events(run)
    detection = state.detection(run)
    n = len(events)
    with st.expander(f"Events · {n}", expanded=False):
        st.caption("largest first")
        if detection is None:
            st.caption("Run **Detect peaks** to list events (after the 1 kHz summary is ready).")
            return

        selected = st.session_state.get("selected_event")
        largest = events.sort_values("amplitude", ascending=False)
        for event_id, e in largest.head(MAX_EVENTS_LISTED).iterrows():
            label = f"{event_id} · {fmt_time(e['t_peak'])} · {e['label']} · amp {e['amplitude']:.1f}"
            st.button(
                label,
                key=f"show_{event_id}",
                width="stretch",
                type="primary" if event_id == selected else "secondary",
                on_click=state.select_event,
                args=(event_id, float(e["t_peak"]), total_s),
            )
        if len(largest) > MAX_EVENTS_LISTED:
            st.caption(f"+ {len(largest) - MAX_EVENTS_LISTED} more events")
