"""Everything the app remembers between clicks (``st.session_state``), in one place.

Streamlit reruns the whole script on every click, so state lives here:
    runs            {run name: {"csv": csv path, "phase": "preview" | "ready"}}
    active_run      name of the selected run
    detections      {run name: detectors.Detection}
    window          (start_s, end_s) window range
    selected_event  event ID, e.g. "E3"

The viewing window is set by two widgets (overview frame and a window-size slider), and
Streamlit forbids writing a widget's key after the widget was drawn in the same run.
``set_window`` and ``sync_slider`` handle that; panels only call ``set_window``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import viewport
from detectors import Detection

DEFAULT_WINDOW_S = 5.0
SLIDER_KEY = "overview_size_slider"  # window width (s); the frame sets the position
SLIDER_STEP_S = 0.1  # fixed step, so round widths such as the 5 s default land exactly
_VIEW_KEYS = ["window", "selected_event", SLIDER_KEY,
              "_pending_slider_from_window", "_overview_slider_live"]

ss = st.session_state


# ---- runs and detections ----------------------------------------------------

def runs() -> dict:
    return ss.setdefault("runs", {})


def detections() -> dict[str, Detection]:
    return ss.setdefault("detections", {})


def active_run() -> str | None:
    run = ss.get("active_run")
    return run if run in runs() else None


def register_run(csv_path: Path, phase: str) -> None:
    """Add a run once its preview is available, or update it when its summary is ready."""
    name = csv_path.stem
    prev = runs().get(name)
    runs()[name] = {"csv": str(csv_path), "phase": phase}
    if prev is None or (phase == "preview" and prev["phase"] != phase):
        detections().pop(name, None)  # new or reloaded run: select it, start fresh
        ss["active_run"] = name
        reset_view()


def is_ready(run: str | None) -> bool:
    """True when the run's 1 kHz summary is loaded (needed for detection and the heatmap)."""
    return bool(run) and runs().get(run, {}).get("phase") == "ready"


def detection(run: str) -> Detection | None:
    return detections().get(run)


def accepted_events(run: str) -> pd.DataFrame:
    """Valid events in time order, indexed by ID: E1, E2, ..."""
    det = detection(run)
    if det is None:
        return pd.DataFrame(columns=["t_peak", "amplitude", "valid", "label", "polarity"])
    events = det.events[det.events["valid"]].reset_index(drop=True)
    events.index = [f"E{i + 1}" for i in range(len(events))]
    return events


def reset_view() -> None:
    """Forget the time range and selected event (e.g. when switching runs)."""
    for key in _VIEW_KEYS:
        ss.pop(key, None)


def keep_or_reset_view(total_s: float) -> None:
    """At index milestone (#30): keep the preview window if it still fits the exact length.

    Only reset to the default 0–5 s range when there is no window or it no longer fits.
    """
    total_s = float(total_s)
    w = ss.get("window")
    if w is None:
        reset_view()
        return
    start_s, end_s = float(w[0]), float(w[1])
    if start_s < 0 or end_s > total_s or end_s <= start_s:
        reset_view()
        return
    # Keep selection; clamp via set_window so slider stays consistent.
    set_window(start_s, end_s, total_s)


# ---- viewing window ---------------------------------------------------------

def window(total_s: float) -> viewport.Window:
    w = ss.get("window")
    if w is None or w[1] > total_s or w[0] < 0:
        ss["window"] = (0.0, min(float(total_s), DEFAULT_WINDOW_S))
    return ss["window"]


def set_window(start_s: float, end_s: float, total_s: float) -> None:
    """Store the viewing window and set the size slider to its width."""
    ss["window"] = viewport.clamp(start_s, end_s, total_s)
    if ss.get("_overview_slider_live"):
        ss["_pending_slider_from_window"] = True  # slider already drawn: update next run
    else:
        ss[SLIDER_KEY] = _width(ss["window"])
        ss.pop("_pending_slider_from_window", None)


def sync_slider(total_s: float) -> viewport.Window:
    """Call before the size slider is drawn: reconcile slider and window, return the window."""
    ss["_overview_slider_live"] = False
    w = window(total_s)
    if ss.pop("_pending_slider_from_window", False) or SLIDER_KEY not in ss:
        ss[SLIDER_KEY] = _width(w)
        return w
    # The slider snaps a width set by dragging the frame to its step; only a bigger change
    # means the user moved the slider. Then keep the centre.
    if abs(float(ss[SLIDER_KEY]) - _width(w)) > 0.5 * SLIDER_STEP_S:
        set_window(*viewport.centred(0.5 * (w[0] + w[1]), float(ss[SLIDER_KEY]), total_s), total_s)
    return ss["window"]


def _width(w: viewport.Window) -> float:
    return round(w[1] - w[0], 4)


def slider_drawn() -> None:
    """Call right after the range slider is drawn."""
    ss["_overview_slider_live"] = True


def select_event(event_id: str, t_peak: float, total_s: float, half_width_s: float) -> None:
    """Select an event and zoom the window to it."""
    ss["selected_event"] = event_id
    set_window(*viewport.around(t_peak, half_width_s, total_s), total_s)


# ---- chart keys -------------------------------------------------------------
# A new widget key makes Streamlit draw a chart from scratch (and forget its selection).

def bump(counter: str) -> None:
    ss[counter] = ss.get(counter, 0) + 1


def chart_key(prefix: str, counter: str) -> str:
    return f"{prefix}_{ss.get(counter, 0)}"
