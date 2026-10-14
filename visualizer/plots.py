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
        # Batch markers into few traces (per class) so long windows stay responsive.
        by_class: dict[str, list[tuple[float, float]]] = {}
        for event in events:
            in_window = start_s <= event["t_peak"] <= end_s
            hidden = not event["valid"] and not show_rejected
            if not (in_window and not hidden):
                continue
            t_peak = float(event["t_peak"])
            y_mark = _y_on_trace(t, guide, t_peak)
            key = event["class"] if event["valid"] else "rejected"
            by_class.setdefault(key, []).append((t_peak, y_mark))

        for key, pts in by_class.items():
            color = PEAK_CLASS_COLORS.get(key, REJECTED_COLOR if key == "rejected" else OTHER_CLASS_COLOR)
            fig.add_trace(go.Scattergl(
                x=[p[0] for p in pts],
                y=[p[1] for p in pts],
                mode="markers",
                marker=dict(size=11, color=color, symbol="triangle-up"),
                name=key,
                showlegend=False,
                hovertemplate=f"{key} @ %{{x:.2f}}s<br>y=%{{y:.1f}}<extra></extra>",
            ))

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
    s1_min=None,
    s1_max=None,
    s2_min=None,
    s2_max=None,
    approximate: bool = False,
) -> go.Figure:
    """Compact full-recording strip with the current zoom window highlighted.

    Optional min/max envelopes (1 kHz summary or rough preview). When
    ``approximate`` is True the title cue is left to the caller via caption.
    """
    fig = go.Figure()
    envelopes = {
        SENSOR1: (s1_min, s1_max),
        SENSOR2: (s2_min, s2_max),
    }
    means = {SENSOR1: s1, SENSOR2: s2}
    for name in (SENSOR1, SENSOR2):
        if name not in channels:
            continue
        lo, hi = envelopes[name]
        color = SENSOR_COLORS[name]
        if lo is not None and hi is not None and len(lo) and len(hi):
            # Band: max then min (reversed) so Plotly fills the envelope.
            alpha = 0.35 if approximate else 0.20
            # Convert #rrggbb → rgba for fill
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            fig.add_trace(go.Scatter(
                x=list(t) + list(t[::-1]),
                y=list(hi) + list(lo[::-1]),
                fill="toself",
                fillcolor=f"rgba({r},{g},{b},{alpha})",
                line=dict(width=0),
                name=f"{name} range",
                showlegend=False,
                hoverinfo="skip",
            ))
        y = means[name]
        if y is not None and len(y):
            fig.add_trace(go.Scattergl(
                x=t, y=y, mode="lines", name=name,
                line=dict(width=1, color=color),
                showlegend=False,
            ))

    if events:
        ys = []
        for ch, y in [(SENSOR1, s1), (SENSOR2, s2)]:
            if ch in channels and len(y):
                ys.append(np.max(y))
        y_top = float(np.max(ys)) if ys else 1.0
        # One scatter for all valid pieces (not one trace per event — that
        # freezes the browser on long recordings).
        xs = [float(e["t_peak"]) for e in events if e.get("valid")]
        if len(xs) > 2_000:
            step = max(1, len(xs) // 2_000)
            xs = xs[::step]
        if xs:
            fig.add_trace(go.Scattergl(
                x=xs,
                y=[y_top] * len(xs),
                mode="markers",
                marker=dict(size=6, color="#e07020", symbol="triangle-down"),
                name="pieces",
                showlegend=False,
                hovertemplate="piece @ %{x:.1f}s<extra></extra>",
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


def _fmt_mmss(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def time_heatmap_figure(
    values: np.ndarray,
    bin_s: float,
    row_s: float,
    total_s: float,
    window: tuple[float, float] | None = None,
    colorscale: str = "Cividis",
    colorbar_title: str = "",
) -> go.Figure:
    """
    Recording folded into rows of ``row_s`` seconds (like lines of text):
    row r, column c is the bin starting at ``r * row_s + c * bin_s``.

    ``values`` holds one number per bin in time order (e.g. from
    ``signal_io.binned_peak_to_peak``). The current viewing window is outlined
    in every row it touches. A transparent marker sits on each cell so the app
    can pick up clicks (Plotly heatmaps do not report selections themselves).
    """
    cols = max(1, int(round(row_s / bin_s)))
    n_rows = max(1, -(-len(values) // cols))
    grid = np.full(n_rows * cols, np.nan)
    grid[: len(values)] = values
    grid = grid.reshape(n_rows, cols)

    x = (np.arange(cols) + 0.5) * bin_s  # cell centers, seconds into the row
    y = np.arange(n_rows)
    row_labels = [_fmt_mmss(r * row_s) for r in range(n_rows)]
    cell_start = y[:, None] * row_s + (x[None, :] - 0.5 * bin_s)

    hover = np.vectorize(lambda s, v: (
        f"{_fmt_mmss(s)} – {_fmt_mmss(s + bin_s)}<br>{colorbar_title}: {v:.0f}"
        if np.isfinite(v) else ""
    ), otypes=[object])(cell_start, grid)

    colorbar = dict(title=dict(text=colorbar_title, side="right"), thickness=10, xpad=4)

    fig = go.Figure(go.Heatmap(
        z=grid, x=x, y=y,
        colorscale=colorscale,
        xgap=0.2, ygap=5,
        colorbar=colorbar,
        hoverinfo="skip",
    ))

    # Invisible click targets, one per (non-empty) cell. They also carry the
    # tooltip: Plotly only sends click events for traces that receive hover.
    yy, xx = np.meshgrid(y, x, indexing="ij")
    valid = np.isfinite(grid)
    fig.add_trace(go.Scatter(
        x=xx[valid], y=yy[valid],
        mode="markers",
        marker=dict(size=14, opacity=0, symbol="square"),
        text=hover[valid],
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))

    if window is not None:
        w0, w1 = window
        for r in range(n_rows):
            r0, r1 = r * row_s, (r + 1) * row_s
            a, b = max(w0, r0), min(w1, r1, total_s)
            if b <= a:
                continue
            fig.add_shape(
                type="rect",
                x0=a - r0, x1=b - r0, y0=r - 0.5, y1=r + 0.5,
                line=dict(color="#ffffff", width=2, dash="dot"),
                fillcolor="rgba(0,0,0,0)",
                layer="above",
            )

    tick_step = row_s / 6 if row_s >= 30 else row_s / 5
    fig.update_layout(
        height=int(50 + 20 * n_rows),  # fixed row height; the app scrolls tall maps
        margin=dict(l=44, r=0, t=20, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        clickmode="event+select",
        # Not False: Plotly only handles click-to-select on its drag layer.
        # Axes are fixedrange, so "zoom" cannot actually zoom.
        dragmode="zoom",
        hovermode="closest",
    )
    fig.update_xaxes(
        range=[0, cols * bin_s],
        showticklabels=False,  # drawn as annotations below, flush with the cells
        showgrid=False, zeroline=False, fixedrange=True,
    )
    # Time labels sit directly on the top edge (stay in view when a tall
    # heatmap scrolls). Annotations instead of tick labels so the gap can be
    # set exactly (yshift); ticklabelstandoff needs a newer Plotly.
    for v in np.arange(0, row_s + 1e-9, tick_step):
        fig.add_annotation(
            x=v, y=1, xref="x", yref="paper",
            text=_fmt_mmss(v) if row_s >= 60 else f"{v:g}s",
            showarrow=False, yanchor="bottom", yshift=1,
            font=dict(size=11),
        )
    fig.update_yaxes(
        tickvals=y, ticktext=row_labels,
        # Exact range, reversed (row 0 on top). Autorange would add padding
        # for the click-target markers and leave a gap under the time labels.
        range=[n_rows - 0.5, -0.5],
        showgrid=False, zeroline=False, fixedrange=True,
        automargin=True,
        tickfont=dict(family="monospace", size=11),
    )
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

    # Batch by class — one stem polyline + one marker scatter per class.
    by_class: dict[str, list[tuple[float, float]]] = {}
    for event in events:
        if not event["valid"] and not show_rejected:
            continue
        positive = event["polarity"] == "positive"
        y = event["amplitude"] if positive else -event["amplitude"]
        key = event["class"] if event["valid"] else "rejected"
        by_class.setdefault(key, []).append((float(event["t_peak"]), float(y)))

    shown = 0
    for key, pts in by_class.items():
        if len(pts) > 2_000:
            step = max(1, len(pts) // 2_000)
            pts = pts[::step]
        color = PEAK_CLASS_COLORS.get(key, REJECTED_COLOR if key == "rejected" else OTHER_CLASS_COLOR)
        # Stem segments as NaN-separated polyline (single trace).
        xs: list[float | None] = []
        ys: list[float | None] = []
        for tx, ty in pts:
            xs.extend([tx, tx, None])
            ys.extend([0.0, ty, None])
        fig.add_trace(go.Scattergl(
            x=xs, y=ys, mode="lines",
            line=dict(width=1.5, color=color),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scattergl(
            x=[p[0] for p in pts],
            y=[p[1] for p in pts],
            mode="markers",
            marker=dict(size=9, color=color, symbol="triangle-up"),
            name=f"{key} ({len(pts)})",
            showlegend=True,
            hovertemplate=f"{key} @ %{{x:.2f}}s<br>amp=%{{y:.1f}}<extra></extra>",
        ))
        shown += len(pts)

    fig.update_layout(
        height=360,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title="Time (s) — full recording",
        yaxis_title="Peak amplitude (corrected ADC)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
        title=dict(text=f"Pieces ({shown})", font=dict(size=14)),
        hovermode="closest",
        dragmode="zoom",
    )
    return fig