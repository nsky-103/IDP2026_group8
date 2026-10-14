"""Left panel, below the runs: channels, detection, heatmap and display settings."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

import detectors
import plots
import signal_io
import state
from detectors import Detector, Param
from panels import heatmap
from style import section


@dataclass
class Settings:
    """Everything the user chose in the left panel."""

    channels: list[str]
    fs_hz: float
    max_points: int
    downsample_method: str
    show_rejected: bool
    detector: Detector
    params: dict
    detect_clicked: bool
    heatmap_row_s: float
    heatmap_cells: int
    heatmap_metric: str


def render() -> Settings:
    section("Channels")
    channels = [name for i, name in enumerate((plots.SENSOR1, plots.SENSOR2), start=1)
                if st.checkbox(f"Ch{i} · {name}", value=True)]

    section("Detection")
    detector = detectors.DETECTORS[st.selectbox("Algorithm", list(detectors.DETECTORS))]
    with st.expander("Detection parameters"):
        if detector.description:
            st.caption(detector.description)
        params = {p.key: _param_widget(p, key=f"param_{detector.name}_{p.key}") for p in detector.params}
    show_rejected = st.checkbox("Show rejected / water (gray)", value=True)
    run = state.active_run()
    detect_clicked = st.button("Detect peaks", type="primary", disabled=not state.is_ready(run),
                               width="stretch",
                               help="Enabled after the 1 kHz summary is ready.")
    if run and not state.is_ready(run):
        st.caption("Waiting for 1 kHz summary…")

    section("Heatmap")
    row_label = st.selectbox("Row length", list(heatmap.ROW_CHOICES), index=2, key="hm_row")
    cells = st.selectbox("Cells / row", [10, 20, 30, 60, 120], index=3, key="hm_cells")
    metrics = list(heatmap.METRICS)
    if not state.detection(state.active_run() or ""):
        metrics.remove(heatmap.EVENTS_METRIC)
    metric = st.selectbox("Colour by", metrics, key="hm_metric")

    with st.expander("Display settings"):
        fs_hz = st.number_input("Sample rate (Hz)", value=float(signal_io.DEFAULT_FS_HZ),
                                min_value=1.0, step=1000.0)
        max_points = st.slider("Max plot points (detail)", 2_000, 100_000, 20_000, 1_000)
        downsample_method = st.selectbox(
            "Detail downsample", ["lttb", "stride"],
            format_func=lambda m: "LTTB (keeps peaks)" if m == "lttb" else "Every-nth sample",
        )

    return Settings(
        channels=channels,
        fs_hz=fs_hz,
        max_points=max_points,
        downsample_method=downsample_method,
        show_rejected=show_rejected,
        detector=detector,
        params=params,
        detect_clicked=detect_clicked,
        heatmap_row_s=heatmap.ROW_CHOICES[row_label],
        heatmap_cells=cells,
        heatmap_metric=metric,
    )


def _param_widget(p: Param, key: str):
    """Widget for one detector parameter, chosen by the type of its default."""
    if isinstance(p.default, bool):
        return st.checkbox(p.label, value=p.default, help=p.help or None, key=key)
    if isinstance(p.default, str):
        return st.text_input(p.label, value=p.default, help=p.help or None, key=key)
    kind = type(p.default)
    if p.min is not None and p.max is not None:
        return st.slider(p.label, kind(p.min), kind(p.max), p.default,
                         kind(p.step) if p.step else None, help=p.help or None, key=key)
    return st.number_input(p.label, value=p.default, step=p.step, help=p.help or None, key=key)


def run_detection(run: str, recording: signal_io.Recording, settings: Settings) -> None:
    """Run the chosen detector on the loaded recording and store the result."""
    with st.spinner(f"Running {settings.detector.name} on the 1 kHz summary…"):
        try:
            state.detections()[run] = detectors.run(settings.detector, recording, settings.params)
        except Exception as exc:
            st.error(f"Detection failed: {exc}")
            return
    st.session_state.pop("selected_event", None)
    st.rerun()  # redraw so the runs list shows the new event count
