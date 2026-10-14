"""
Plotly figure builders for the visualizer.

No Streamlit here: each function takes data and returns a ``go.Figure``;
the app decides where to show it. Add new charts here as new functions.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

SENSOR1 = "Sensor 1 (adc1)"
SENSOR2 = "Sensor 2 (adc2)"
SENSOR_COLORS = {SENSOR1: "#1f4fa3", SENSOR2: "#e07020"}

PEAK_CLASS_COLORS = {
    "high": "#b00020",
    "medium": "#e65100",
    # Method B (core) piece labels
    "dyed": "#b00020",
    "solid_no_dye": "#e65100",
    "water": "#888",
}
REJECTED_COLOR = "#888"
OTHER_CLASS_COLOR = "#333"

# Shared Plotly config: trackpad / mouse-wheel zoom, drag to box-zoom
PLOTLY_ZOOM_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "doubleClick": "reset",
    "modeBarButtonsToAdd": ["resetScale2d"],
}


def _y_on_trace(t: np.ndarray, y: np.ndarray, t_peak: float) -> float:
    """Y value of the plotted curve at ``t_peak`` (nearest sample).

    Detection amplitudes are on the *baseline-corrected* signal, while the
    detail window shows raw ADC — so markers must sit on the drawn trace,
    not at mean(raw) ± corrected_amplitude (that looked vertically shifted).
    """
    if len(t) == 0 or len(y) == 0:
        return 0.0
    i = int(np.argmin(np.abs(t - float(t_peak))))
    return float(y[i])


def peak_marker(event: dict, y_on_signal: float) -> go.Scatter:
    """One triangle marker for a detected peak (gray when rejected)."""
    positive = event["polarity"] == "positive"

    if event["valid"]:
        color = PEAK_CLASS_COLORS.get(event["class"], OTHER_CLASS_COLOR)
    else:
        color = REJECTED_COLOR

    return go.Scatter(
        x=[event["t_peak"]],
        y=[y_on_signal],
        mode="markers",
        marker=dict(size=11, color=color, symbol="triangle-up" if positive else "triangle-down"),
        name=f"{event['class']} @ {event['t_peak']:.1f}s",
        showlegend=False,
        hovertext=(
            f"t={event['t_peak']:.2f}s  class={event['class']}  "
            f"amp={event['amplitude']:.1f} (corrected)  valid={event['valid']}"
        ),
        hoverinfo="text",
    )


def signal_figure(
    t, s1, s2,
    channels: list[str],
    start_s: float,
    end_s: float,
    events: list[dict] | None = None,
    show_rejected: bool = True,
    height: int = 420,
) -> go.Figure:
    """Sensor traces for the display window, with optional peak markers."""
    fig = go.Figure()

    for name, y in [(SENSOR1, s1), (SENSOR2, s2)]:
        if name in channels:
            fig.add_trace(go.Scattergl(
                x=t, y=y, mode="lines", name=name,
                line=dict(width=1, color=SENSOR_COLORS[name]),
            ))

    if events and len(t) and len(s2):
        # Prefer Sensor 2 for marker height (detection target); fall back to s1.
        guide = s2 if SENSOR2 in channels and len(s2) else s1
        y_lo = float(np.min(guide)) if len(guide) else 0.0
        y_hi = float(np.max(guide)) if len(guide) else 1.0
        for event in events:
            in_window = start_s <= event["t_peak"] <= end_s
            hidden = not event["valid"] and not show_rejected
            if not (in_window and not hidden):
                continue
            t_peak = float(event["t_peak"])
            y_mark = _y_on_trace(t, guide, t_peak)
            if event["valid"]:
                color = PEAK_CLASS_COLORS.get(event["class"], OTHER_CLASS_COLOR)
            else:
                color = REJECTED_COLOR
            # Thin stem at the true detection time (helps when LTTB flattens tips).
            fig.add_trace(go.Scatter(
                x=[t_peak, t_peak],
                y=[y_lo, y_hi],
                mode="lines",
                line=dict(width=1, color=color, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            ))
            fig.add_trace(peak_marker(event, y_mark))

    fig.update_layout(
        height=int(height),
        margin=dict(l=48, r=20, t=30, b=40),
        xaxis_title="Time (s) — scroll / pinch to zoom, drag to box-zoom, double-click to reset",
        yaxis_title="ADC amplitude",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        dragmode="zoom",
    )
    return fig


def overview_figure(
    t, s1, s2,
    channels: list[str],
    window: tuple[float, float],
    events: list[dict] | None = None,
    height: int = 140,
) -> go.Figure:
    """Compact full-recording strip with the current zoom window highlighted."""
    fig = go.Figure()
    for name, y in [(SENSOR1, s1), (SENSOR2, s2)]:
        if name in channels:
            fig.add_trace(go.Scattergl(
                x=t, y=y, mode="lines", name=name,
                line=dict(width=1, color=SENSOR_COLORS[name]),
                showlegend=False,
            ))

    if events:
        ys = []
        for ch, y in [(SENSOR1, s1), (SENSOR2, s2)]:
            if ch in channels and len(y):
                ys.append(np.max(y))
        y_top = float(np.max(ys)) if ys else 1.0
        for event in events:
            if not event.get("valid"):
                continue
            fig.add_trace(go.Scatter(
                x=[event["t_peak"], event["t_peak"]],
                y=[y_top * 0.92, y_top],
                mode="lines",
                line=dict(width=2, color="#e07020"),
                showlegend=False,
                hovertext=f"event @ {event['t_peak']:.1f}s",
                hoverinfo="text",
            ))

    w0, w1 = window
    # Template-style viewing frame: dim outside, bright fill + thick black box
    t_min = float(np.min(t)) if len(t) else 0.0
    t_max = float(np.max(t)) if len(t) else 1.0
    if w0 > t_min:
        fig.add_vrect(
            x0=t_min, x1=w0,
            fillcolor="rgba(0, 0, 0, 0.18)",
            line=dict(width=0),
            layer="above",
        )
    if w1 < t_max:
        fig.add_vrect(
            x0=w1, x1=t_max,
            fillcolor="rgba(0, 0, 0, 0.18)",
            line=dict(width=0),
            layer="above",
        )
    fig.add_vrect(
        x0=w0, x1=w1,
        fillcolor="rgba(255, 255, 255, 0.08)",
        line=dict(width=0),
        layer="below",
    )
    fig.add_shape(
        type="rect",
        x0=w0, x1=w1, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="#000000", width=4),
        fillcolor="rgba(0,0,0,0)",
        layer="above",
    )

    # Edge handles (visual cues for the selectable window ends)
    for x, label in ((w0, "start"), (w1, "end")):
        fig.add_vline(
            x=x,
            line=dict(color="#000000", width=3),
            layer="above",
        )
        fig.add_annotation(
            x=x, y=1.08, yref="paper",
            text=label,
            showarrow=False,
            font=dict(size=11, color="#111"),
            xanchor="center",
        )

    fig.update_layout(
        height=int(height),
        margin=dict(l=48, r=16, t=28, b=30),
        xaxis_title="",
        yaxis_title="",
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        # Horizontal box-drag sets the viewing window (Streamlit range bar is separate)
        dragmode="select",
        selectdirection="h",
        newselection=dict(line=dict(color="#000000", width=3, dash="solid")),
        activeselection=dict(fillcolor="rgba(0,0,0,0.12)", opacity=1.0),
    )
    fig.update_xaxes(showgrid=False, fixedrange=True)
    fig.update_yaxes(showticklabels=False, showgrid=False, fixedrange=True)
    return fig


def peaks_overview_figure(
    events: list[dict],
    corrected_t: np.ndarray | None = None,
    corrected: np.ndarray | None = None,
    threshold: float | None = None,
    show_rejected: bool = True,
) -> go.Figure:
    """One chart with every detected peak (full recording, not just the window)."""
    fig = go.Figure()

    if corrected_t is not None and corrected is not None and len(corrected_t) > 0:
        n = len(corrected_t)
        stride = max(1, n // 20_000)
        fig.add_trace(go.Scattergl(
            x=corrected_t[::stride],
            y=corrected[::stride],
            mode="lines",
            name="corrected Sensor 2",
            line=dict(width=1, color="#4a90c8"),
        ))

    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line_dash="dot",
            line_color="green",
            annotation_text="+threshold",
            annotation_position="bottom right",
        )

    shown = 0
    for event in events:
        if not event["valid"] and not show_rejected:
            continue
        positive = event["polarity"] == "positive"
        y = event["amplitude"] if positive else -event["amplitude"]
        if event["valid"]:
            color = PEAK_CLASS_COLORS.get(event["class"], OTHER_CLASS_COLOR)
        else:
            color = REJECTED_COLOR
        label = event["class"] if event["valid"] else "rejected"
        fig.add_trace(go.Scatter(
            x=[event["t_peak"], event["t_peak"]],
            y=[0.0, y],
            mode="lines",
            line=dict(width=1.5, color=color),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[event["t_peak"]],
            y=[y],
            mode="markers+text",
            marker=dict(
                size=11,
                color=color,
                symbol="triangle-up" if positive else "triangle-down",
            ),
            text=[f"P{shown + 1}"],
            textposition="top center" if positive else "bottom center",
            textfont=dict(size=10),
            name=f"P{shown + 1} · {label}",
            showlegend=True,
            hovertext=(
                f"P{shown + 1}  t={event['t_peak']:.2f}s  "
                f"amp={event['amplitude']:.1f}  class={label}"
            ),
            hoverinfo="text",
        ))
        shown += 1

    fig.update_layout(
        height=360,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title="Time (s) — full recording",
        yaxis_title="Peak amplitude (corrected ADC)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
        title=dict(text=f"All peaks ({shown})", font=dict(size=14)),
        hovermode="closest",
        dragmode="zoom",
    )
    return fig