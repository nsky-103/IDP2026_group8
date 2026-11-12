"""
Plotly figure builders for the visualizer.

No Streamlit here: each function takes data and returns a ``go.Figure``;
the app decides where to show it. Add new charts here as new functions.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go

SENSOR1 = "Sensor 1 (adc1)"
SENSOR2 = "Sensor 2 (adc2)"
SENSOR_COLORS = {SENSOR1: "#1f4fa3", SENSOR2: "#e07020"}

# Event colours by label; a label not listed gets a fixed colour from LABEL_PALETTE.
LABEL_COLORS = {"dyed": "#b00020", "solid_no_dye": "#e65100", "water": "#888", "event": "#b00020"}
LABEL_PALETTE = ["#6a1b9a", "#2e7d32", "#00838f", "#ad1457", "#1565c0", "#4e342e"]
REJECTED_COLOR = "#888"
EVENT_COLUMNS = ["t_peak", "amplitude", "valid", "label", "polarity"]
MAX_MARKERS = 2_000  # per trace; more markers make long recordings sluggish


def event_color(label: str, valid: bool) -> str:
    if not valid:
        return REJECTED_COLOR
    return LABEL_COLORS.get(label) or LABEL_PALETTE[zlib.crc32(str(label).encode()) % len(LABEL_PALETTE)]


def _shown(events: pd.DataFrame | None, show_rejected: bool = True) -> pd.DataFrame:
    """Events to draw. ``events`` is ``detectors.Detection.events`` (one row per event)."""
    if events is None or events.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return events if show_rejected else events[events["valid"]]


def _by_label(events: pd.DataFrame):
    """(name, colour, rows) per label; rejected events of any label are grouped as "rejected".

    Drawing one trace per group instead of one per event keeps long recordings responsive.
    """
    names = events["label"].astype(str).where(events["valid"], "rejected")
    for name, rows in events.groupby(names, sort=False):
        yield name, event_color(name, name != "rejected"), rows


def _thin(x: np.ndarray, limit: int = MAX_MARKERS) -> np.ndarray:
    return x[:: max(1, len(x) // limit)] if len(x) > limit else x

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


def signal_figure(
    t, s1, s2,
    channels: list[str],
    start_s: float,
    end_s: float,
    events: pd.DataFrame | None = None,
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

    shown = _shown(events, show_rejected)
    shown = shown[(shown["t_peak"] >= start_s) & (shown["t_peak"] <= end_s)]
    if len(shown) and len(t) and len(s2):
        # Prefer Sensor 2 for marker height (detection target); fall back to s1.
        guide = s2 if SENSOR2 in channels and len(s2) else s1
        for name, color, rows in _by_label(shown):
            x = rows["t_peak"].to_numpy(dtype=float)
            fig.add_trace(go.Scattergl(
                x=x,
                y=[_y_on_trace(t, guide, tp) for tp in x],
                mode="markers",
                marker=dict(size=11, color=color, symbol="triangle-up"),
                name=name,
                showlegend=False,
                hovertemplate=f"{name} @ %{{x:.2f}}s<br>y=%{{y:.1f}}<extra></extra>",
            ))

    fig.update_layout(
        height=int(height),
        margin=dict(l=48, r=20, t=30, b=40),
        xaxis_title="Time (s)",
        yaxis_title="ADC amplitude",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        dragmode="zoom",
        xaxis=dict(range=[float(start_s), float(end_s)]),
    )
    return fig


def overview_event_times(events: pd.DataFrame | None) -> np.ndarray:
    """Times of valid events for the overview strip, thinned like the chart markers."""
    return _thin(_shown(events, show_rejected=False)["t_peak"].to_numpy(dtype=float))


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
    events: pd.DataFrame,
    trace: tuple[np.ndarray, np.ndarray] | None = None,
    threshold: float | None = None,
    show_rejected: bool = True,
) -> go.Figure:
    """One chart with every detected peak (full recording, not just the window).

    ``trace`` is the (t, y) signal the detector worked on, drawn behind the stems.
    """
    fig = go.Figure()

    if trace is not None and len(trace[0]) > 0:
        t, y = trace
        stride = max(1, len(t) // 20_000)
        fig.add_trace(go.Scattergl(
            x=t[::stride],
            y=y[::stride],
            mode="lines",
            name="detector signal",
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
    for name, color, rows in _by_label(_shown(events, show_rejected)):
        rows = rows.iloc[_thin(np.arange(len(rows)))]
        x = rows["t_peak"].to_numpy(dtype=float)
        y = np.where(rows["polarity"] == "positive", 1.0, -1.0) * rows["amplitude"].to_numpy(dtype=float)
        fig.add_trace(go.Scattergl(  # all stems of this label: (x, 0), (x, y), gap, ...
            x=np.repeat(x, 3),
            y=np.column_stack([np.zeros_like(y), y, np.full_like(y, np.nan)]).ravel(),
            mode="lines",
            line=dict(width=1.5, color=color),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scattergl(
            x=x, y=y,
            mode="markers",
            marker=dict(size=9, color=color, symbol="triangle-up"),
            name=f"{name} ({len(x)})",
            showlegend=True,
            hovertemplate=f"{name} @ %{{x:.2f}}s<br>amp=%{{y:.1f}}<extra></extra>",
        ))
        shown += len(x)

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