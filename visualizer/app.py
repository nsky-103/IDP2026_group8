"""
Marginum signal visualizer (Objective 1).

Shows arbitrarily large recordings without loading the full CSV into RAM:
CSV → int16 .npy (once), then memory-mapped windowed + downsampled plots.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from signal_io import (
    DEFAULT_FS_HZ,
    duration_s,
    ensure_npy,
    load_window,
    open_signal,
)

DEFAULT_CSV = (
    Path(__file__).resolve().parent.parent
    / "idp2023"
    / "Test signals from previous years"
    / "Suction data"
    / "Original"
    / "5. suction_HIGH_CONC_1ml_and_no_resuction.csv"
)

st.set_page_config(page_title="Marginum Signal Visualizer", layout="wide")
st.title("Marginum Signal Visualizer")
st.caption("Objective 1 — large-file overview without loading the whole CSV into memory")


with st.sidebar:
    st.header("Data")
    path_str = st.text_input(
        "Path to CSV or NPY",
        value=str(DEFAULT_CSV) if DEFAULT_CSV.exists() else "",
        help="Use a local path (do not upload multi-GB files through the browser).",
    )
    fs_hz = st.number_input("Sample rate (Hz)", value=float(DEFAULT_FS_HZ), min_value=1.0, step=1000.0)
    max_points = st.slider("Max plot points", min_value=2_000, max_value=100_000, value=20_000, step=1_000)
    channels = st.multiselect(
        "Channels",
        options=["Sensor 1 (adc1)", "Sensor 2 (adc2)"],
        default=["Sensor 1 (adc1)", "Sensor 2 (adc2)"],
    )
    convert_clicked = st.button("Load / convert", type="primary")


if convert_clicked or "signal_npy" in st.session_state:
    path = Path(path_str.strip()) if path_str.strip() else None
    if path is None or not path.exists():
        st.error("File not found. Check the path.")
        st.stop()

    if convert_clicked or st.session_state.get("source_path") != str(path):
        progress = st.progress(0.0, text="Preparing signal…")

        def on_progress(p: float) -> None:
            progress.progress(min(max(p, 0.0), 1.0), text=f"Converting CSV → NPY… {p:.0%}")

        try:
            npy = ensure_npy(path, progress_callback=on_progress if path.suffix.lower() == ".csv" else None)
        except Exception as exc:
            progress.empty()
            st.error(f"Failed to prepare signal: {exc}")
            st.stop()

        progress.empty()
        st.session_state["signal_npy"] = str(npy)
        st.session_state["source_path"] = str(path)

    npy_path = Path(st.session_state["signal_npy"])
    signal = open_signal(npy_path)
    n_samples = signal.shape[0]
    total_s = duration_s(n_samples, fs_hz)

    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", f"{n_samples:,}")
    c2.metric("Duration", f"{total_s:.2f} s")
    c3.metric("Binary file", npy_path.name)

    st.subheader("Time window")
    default_end = min(total_s, 30.0)
    start_s, end_s = st.slider(
        "Display range (seconds)",
        min_value=0.0,
        max_value=float(total_s),
        value=(0.0, float(default_end)),
        step=max(total_s / 1000.0, 0.01),
    )

    if not channels:
        st.warning("Select at least one channel.")
        st.stop()

    t, s1, s2 = load_window(signal, start_s, end_s, fs_hz=fs_hz, max_points=max_points)

    fig = go.Figure()
    if "Sensor 1 (adc1)" in channels:
        fig.add_trace(go.Scattergl(x=t, y=s1, mode="lines", name="Sensor 1 (adc1)", line=dict(width=1)))
    if "Sensor 2 (adc2)" in channels:
        fig.add_trace(go.Scattergl(x=t, y=s2, mode="lines", name="Sensor 2 (adc2)", line=dict(width=1)))

    fig.update_layout(
        height=520,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title="Time (s)",
        yaxis_title="ADC amplitude",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Window statistics"):
        cols = st.columns(2)
        if "Sensor 1 (adc1)" in channels:
            cols[0].write(
                {
                    "sensor1_min": float(s1.min()),
                    "sensor1_max": float(s1.max()),
                    "sensor1_mean": float(s1.mean()),
                }
            )
        if "Sensor 2 (adc2)" in channels:
            cols[1].write(
                {
                    "sensor2_min": float(s2.min()),
                    "sensor2_max": float(s2.max()),
                    "sensor2_mean": float(s2.mean()),
                }
            )

    st.info(
        "Tip: first load converts CSV → `.npy` next to the file. Later loads reuse the binary "
        "and only map the selected time window into memory."
    )
else:
    st.write(
        "Enter a path to a Marginum CSV/NPY in the sidebar, then click **Load / convert**. "
        "A medium suction file is pre-filled if available."
    )
