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

PEAK_CLASS_COLORS = {"high": "#b00020", "medium": "#e65100"}
REJECTED_COLOR = "#888"
OTHER_CLASS_COLOR = "#333"


def peak_marker(event: dict, s2_mean: float) -> go.Scatter:
    """One triangle marker for a detected peak (gray when rejected)."""
    positive = event["polarity"] == "positive"
    y = s2_mean + (event["amplitude"] if positive else -event["amplitude"])

    if event["valid"]:
        color = PEAK_CLASS_COLORS.get(event["class"], OTHER_CLASS_COLOR)
    else:
        color = REJECTED_COLOR

    return go.Scatter(
        x=[event["t_peak"]],
        y=[y],
        mode="markers",
        marker=dict(size=10, color=color, symbol="triangle-up" if positive else "triangle-down"),
        name=f"{event['class']} @ {event['t_peak']:.1f}s",
        showlegend=False,
        hovertext=f"class={event['class']} amp={event['amplitude']:.1f} valid={event['valid']}",
    )


def signal_figure(
    t, s1, s2,
    channels: list[str],
    start_s: float,
    end_s: float,
    events: list[dict] | None = None,
    show_rejected: bool = True,
) -> go.Figure:
    """Sensor traces for the display window, with optional peak markers."""
    fig = go.Figure()

    for name, y in [(SENSOR1, s1), (SENSOR2, s2)]:
        if name in channels:
            fig.add_trace(go.Scattergl(
                x=t, y=y, mode="lines", name=name,
                line=dict(width=1, color=SENSOR_COLORS[name]),
            ))

    if events:
        s2_mean = float(np.mean(s2)) if len(s2) else 0.0
        for event in events:
            in_window = start_s <= event["t_peak"] <= end_s
            hidden = not event["valid"] and not show_rejected
            if in_window and not hidden:
                fig.add_trace(peak_marker(event, s2_mean))

    fig.update_layout(
        height=420,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title="Time (s)",
        yaxis_title="ADC amplitude",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    return fig
