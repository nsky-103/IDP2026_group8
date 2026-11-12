"""Viewing-window math: (start_s, end_s) pairs in, (start_s, end_s) pairs out.

No Streamlit here; ``state.py`` stores the result.
Width is capped only by the recording length (and MIN_WIDTH_S).
"""

from __future__ import annotations

MIN_WIDTH_S = 0.5
Window = tuple[float, float]


def clamp(start_s: float, end_s: float, total_s: float) -> Window:
    """Order the ends, keep them inside the recording and at least MIN_WIDTH_S apart."""
    total = float(total_s)
    a = max(0.0, min(float(start_s), total))
    b = max(0.0, min(float(end_s), total))
    if b < a:
        a, b = b, a
    if b - a < MIN_WIDTH_S:
        b = min(total, a + MIN_WIDTH_S)
        if b - a < MIN_WIDTH_S:
            a = max(0.0, b - MIN_WIDTH_S)
    return a, b


def _shift_inside(start: float, end: float, total_s: float) -> Window:
    """Move a window that sticks out of the recording back inside, keeping its width."""
    if start < 0.0:
        end, start = end - start, 0.0
    if end > total_s:
        start, end = start - (end - total_s), float(total_s)
    return max(0.0, start), min(float(total_s), end)


def zoom(window: Window, total_s: float, factor: float) -> Window:
    """Zoom around the centre (factor < 1 zooms in). At an edge the free side grows instead."""
    start, end = window
    centre = 0.5 * (start + end)
    width = max(MIN_WIDTH_S, min((end - start) * factor, float(total_s)))
    return _shift_inside(centre - 0.5 * width, centre + 0.5 * width, total_s)


def pan(window: Window, total_s: float, direction: float) -> Window:
    """Move left (-1) or right (+1) by half the window width."""
    start, end = window
    shift = direction * 0.5 * (end - start)
    return _shift_inside(start + shift, end + shift, total_s)


def centred(centre_s: float, width_s: float, total_s: float) -> Window:
    """Window of ``width_s`` around ``centre_s``, shifted inside the recording."""
    width = min(max(width_s, MIN_WIDTH_S), float(total_s))
    start = min(max(0.0, centre_s - 0.5 * width), total_s - width)
    return start, start + width


def around(t_s: float, half_width_s: float, total_s: float) -> Window:
    """``t_s`` ± ``half_width_s``, cut at the ends of the recording."""
    return max(0.0, t_s - half_width_s), min(float(total_s), t_s + half_width_s)
