"""
Marginum signal visualizer — noise reduction & graph visualization.

Shows arbitrarily large recordings without loading the full CSV into RAM:
CSV → int16 .npy (once), then memory-mapped windowed + downsampled plots.

Plot downsampling defaults to Largest-Triangle-Three-Buckets (LTTB)
(Steinarsson 2013, University of Iceland):
https://skemman.is/bitstream/1946/15343/3/SS_MSthesis.pdf

Primary workflow: Load → Denoise / baseline → inspect cleaned plots.
Peak detection is optional (teammate algorithm / later integration).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import lttb as _lttb
import signal_io as _signal_io

importlib.reload(_lttb)
importlib.reload(_signal_io)

DEFAULT_FS_HZ = _signal_io.DEFAULT_FS_HZ
duration_s = _signal_io.duration_s
ensure_npy = _signal_io.ensure_npy
load_window = _signal_io.load_window
open_signal = _signal_io.open_signal

ROOT = Path(__file__).resolve().parent.parent
FLUOSENSE_SRC = ROOT / "fluosense_code_only"
if str(FLUOSENSE_SRC) not in sys.path:
    sys.path.insert(0, str(FLUOSENSE_SRC))

DEFAULT_CSV = (
    ROOT
    / "idp2023"
    / "Test signals from previous years"
    / "Suction data"
    / "Original"
    / "5. suction_HIGH_CONC_1ml_and_no_resuction.csv"
)
DEFAULT_BLANK = (
    ROOT / "idp2023" / "Test signals from previous years" / "just water.csv"
)

st.set_page_config(page_title="Marginum Signal Visualizer", layout="wide")
st.title("Marginum Signal Visualizer")
st.caption(
    "Noise reduction + graph visualization · "
    "plot downsampling: LTTB (Steinarsson 2013) · "
    "peak detection optional (other team member)"
)


with st.sidebar:
    st.header("Data")
    path_str = st.text_input(
        "Path to CSV or NPY",
        value=str(DEFAULT_CSV) if DEFAULT_CSV.exists() else "",
        help="Use a local path (do not upload multi-GB files through the browser).",
    )
    fs_hz = st.number_input("Sample rate (Hz)", value=float(DEFAULT_FS_HZ), min_value=1.0, step=1000.0)
    max_points = st.slider("Max plot points", min_value=2_000, max_value=100_000, value=20_000, step=1_000)
    downsample_method = st.selectbox(
        "Plot downsample",
        options=["lttb", "stride"],
        index=0,
        format_func=lambda m: (
            "LTTB (Steinarsson 2013)" if m == "lttb" else "Every-nth stride"
        ),
        help=(
            "LTTB keeps sharp peaks when Max plot points is small. "
            "With 20k points the two methods often look almost the same."
        ),
    )
    compare_methods = st.checkbox(
        "Compare LTTB vs stride (Sensor 2)",
        value=False,
        help="Overlay both methods at a low point budget so the difference is obvious.",
    )
    compare_points = st.slider(
        "Compare point budget",
        min_value=50,
        max_value=2_000,
        value=300,
        step=50,
        disabled=not compare_methods,
        help="Fewer points → larger visual gap (LTTB keeps spikes; stride often skips them).",
    )
    channels = st.multiselect(
        "Channels",
        options=["Sensor 1 (adc1)", "Sensor 2 (adc2)"],
        default=["Sensor 1 (adc1)", "Sensor 2 (adc2)"],
    )
    convert_clicked = st.button("Load / convert", type="primary")

    st.divider()
    st.header("Noise reduction")
    st.caption("Runs downsample → denoise → baseline. No peak / feature detection.")
    blank_str = st.text_input(
        "Blank (just water) CSV — optional noise floor",
        value=str(DEFAULT_BLANK) if DEFAULT_BLANK.exists() else "",
    )
    baseline_method = st.selectbox("Baseline", ["rolling_median", "asls"], index=0)
    show_stages = st.multiselect(
        "Show preprocess stages (Sensor 2)",
        options=["raw (downsampled)", "denoised", "baseline", "corrected"],
        default=["raw (downsampled)", "denoised", "corrected"],
    )
    preprocess_clicked = st.button("Run preprocessing", type="primary")

    st.divider()
    with st.expander("Peak detection (optional / teammate)", expanded=False):
        st.caption(
            "Not required for the noise-reduction / visualization workstream. "
            "Kept so a teammate algorithm can be plugged into the same UI later."
        )
        sigma_mult = st.slider("Noise σ multiplier", 3.0, 20.0, 5.0, 0.5)
        channel_ratio = st.slider("Min adc2/adc1 ratio", 0.5, 3.0, 1.3, 0.1)
        show_rejected = st.checkbox("Show rejected candidates (gray)", value=True)
        detect_clicked = st.button("Detect peaks")


def _reload_fluosense():
    from src import config as cfg
    from src import detect as detect_mod
    from src import pipeline as pipeline_mod

    importlib.reload(cfg)
    importlib.reload(detect_mod)
    importlib.reload(pipeline_mod)
    return cfg, detect_mod, pipeline_mod


def _run_preprocess(fs: float, baseline: str, blank_path: str):
    cfg, _, pipeline_mod = _reload_fluosense()
    cfg.FS_HZ = float(fs)
    cfg.BASELINE_METHOD = baseline
    signal_path = Path(st.session_state["source_path"])
    blank = blank_path.strip() or None
    if blank and not Path(blank).exists():
        raise FileNotFoundError(f"Blank file not found: {blank}")
    return pipeline_mod.run_preprocess_only(
        str(signal_path),
        fs_hz=fs,
        baseline_method=baseline,
        blank_path=blank,
    )


def _run_detection(blank_path: str, fs: float, baseline: str,
                   sigma: float, min_ratio: float):
    cfg, _, pipeline_mod = _reload_fluosense()
    cfg.FS_HZ = float(fs)
    cfg.NOISE_SIGMA_MULT = float(sigma)
    cfg.BASELINE_METHOD = baseline
    cfg.MIN_CHANNEL_RATIO = float(min_ratio)
    if not hasattr(cfg, "ACCEPT_NEGATIVE_PEAKS"):
        cfg.ACCEPT_NEGATIVE_PEAKS = False

    blank_noise = None
    if blank_path.strip():
        bp = Path(blank_path.strip())
        if bp.exists():
            from src.detect import noise_stats as _noise_stats
            blank_res = pipeline_mod.run_preprocess_only(
                str(bp), fs_hz=fs, baseline_method=baseline
            )
            blank_noise = _noise_stats(blank_res["proc2"]["corrected"])

    signal_path = Path(st.session_state["source_path"])
    return pipeline_mod.run_pipeline(
        str(signal_path),
        fs_hz=fs,
        blank_noise=blank_noise,
        baseline_method=baseline,
    )


def _maybe_downsample_series(t, y, max_pts: int, method: str):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) <= max_pts:
        return t, y
    if method == "lttb":
        from lttb import downsample_lttb
        return downsample_lttb(t, y, max_points=max_pts)
    stride = max(1, len(t) // max_pts)
    return t[::stride], y[::stride]


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
        st.session_state.pop("preprocess_result", None)
        st.session_state.pop("detect_result", None)

    npy_path = Path(st.session_state["signal_npy"])
    signal = open_signal(npy_path)
    n_samples = signal.shape[0]
    total_s = duration_s(n_samples, fs_hz)

    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", f"{n_samples:,}")
    c2.metric("Duration", f"{total_s:.2f} s")
    c3.metric("Binary file", npy_path.name)

    if preprocess_clicked:
        with st.spinner("Running noise reduction / baseline…"):
            try:
                st.session_state["preprocess_result"] = _run_preprocess(
                    fs_hz, baseline_method, blank_str
                )
            except Exception as exc:
                st.error(f"Preprocessing failed: {exc}")
                st.stop()

    if detect_clicked:
        with st.spinner("Running peak detection…"):
            try:
                st.session_state["detect_result"] = _run_detection(
                    blank_str, fs_hz, baseline_method, sigma_mult, channel_ratio
                )
                # Keep preprocess stages available if detection also produced them
                det = st.session_state["detect_result"]
                st.session_state["preprocess_result"] = {
                    "path": det["path"],
                    "proc1": det["proc1"],
                    "proc2": det["proc2"],
                    "noise": det["noise"],
                    "baseline_method": det["baseline_method"],
                    "fs_hz": float(det["proc2"].get("fs_hz") or fs_hz),
                }
            except Exception as exc:
                st.error(f"Detection failed: {exc}")
                st.stop()

    st.subheader("Raw browse (time window)")
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

    t, s1, s2 = load_window(
        signal, start_s, end_s, fs_hz=fs_hz, max_points=max_points, method=downsample_method
    )

    fig = go.Figure()
    if "Sensor 1 (adc1)" in channels:
        fig.add_trace(go.Scattergl(x=t, y=s1, mode="lines", name="Sensor 1 (adc1)", line=dict(width=1)))
    if "Sensor 2 (adc2)" in channels:
        fig.add_trace(go.Scattergl(x=t, y=s2, mode="lines", name="Sensor 2 (adc2)", line=dict(width=1)))

    result = st.session_state.get("detect_result")
    if result is not None:
        s2_mean = float(np.mean(s2)) if len(s2) else 0.0
        for e in result["events"]:
            if not (start_s <= e["t_peak"] <= end_s):
                continue
            if not e["valid"] and not show_rejected:
                continue
            y = s2_mean + (e["amplitude"] if e["polarity"] == "positive" else -e["amplitude"])
            color = "#888" if not e["valid"] else (
                "#b00020" if e["class"] == "high" else (
                    "#e65100" if e["class"] == "medium" else "#333"
                )
            )
            fig.add_trace(go.Scatter(
                x=[e["t_peak"]],
                y=[y],
                mode="markers",
                marker=dict(
                    size=10,
                    color=color,
                    symbol="triangle-up" if e["polarity"] == "positive" else "triangle-down",
                ),
                name=f"{e['class']} @ {e['t_peak']:.1f}s",
                showlegend=False,
                hovertext=f"class={e['class']} amp={e['amplitude']:.1f} valid={e['valid']}",
            ))

    fig.update_layout(
        height=420,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title="Time (s)",
        yaxis_title="ADC amplitude",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    if compare_methods:
        st.subheader(f"LTTB vs stride @ {compare_points} points (Sensor 2)")
        st.caption(
            "Same time window, same point budget. LTTB (blue) should keep suction spikes; "
            "stride (orange, dashed) often flattens or misses them when the budget is tight."
        )
        t_lttb, _, s2_lttb = load_window(
            signal, start_s, end_s, fs_hz=fs_hz, max_points=compare_points, method="lttb"
        )
        t_stride, _, s2_stride = load_window(
            signal, start_s, end_s, fs_hz=fs_hz, max_points=compare_points, method="stride"
        )
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scattergl(
            x=t_lttb, y=s2_lttb, mode="lines", name="LTTB",
            line=dict(width=1.5, color="#1565c0"),
        ))
        fig_cmp.add_trace(go.Scattergl(
            x=t_stride, y=s2_stride, mode="lines", name="stride",
            line=dict(width=1.5, color="#ef6c00", dash="dash"),
        ))
        fig_cmp.update_layout(
            height=320,
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis_title="Time (s)",
            yaxis_title="ADC (Sensor 2)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            hovermode="x unified",
        )
        st.plotly_chart(fig_cmp, use_container_width=True)
        c_a, c_b = st.columns(2)
        c_a.metric("LTTB points", f"{len(t_lttb):,}")
        c_b.metric("stride points", f"{len(t_stride):,}")
        if len(s2_lttb) and len(s2_stride):
            c_a.metric("LTTB max", f"{float(np.max(s2_lttb)):.1f}")
            c_b.metric("stride max", f"{float(np.max(s2_stride)):.1f}")

    prep = st.session_state.get("preprocess_result")
    if prep is not None:
        st.subheader("Preprocessed Sensor 2")
        st.caption(
            f"baseline={prep['baseline_method']} · "
            f"effective rate ≈ {prep.get('fs_hz', float('nan')):.1f} Hz · "
            "stages: downsample → median+Savitzky–Golay → baseline correction"
        )
        proc2 = prep["proc2"]
        pt_full = np.asarray(proc2["t"], dtype=float)
        # Restrict preprocess plots to the same display window when possible
        win = (pt_full >= start_s) & (pt_full <= end_s)
        if win.sum() < 2:
            win = slice(None)
        else:
            win = win

        stage_map = {
            "raw (downsampled)": ("raw_ds", "#90a4ae"),
            "denoised": ("denoised", "#1565c0"),
            "baseline": ("baseline", "#7b1fa2"),
            "corrected": ("corrected", "#2e7d32"),
        }
        fig_p = go.Figure()
        for label in show_stages:
            key, color = stage_map[label]
            y_full = np.asarray(proc2[key], dtype=float)
            if isinstance(win, np.ndarray):
                tt, yy = pt_full[win], y_full[win]
            else:
                tt, yy = pt_full, y_full
            tt, yy = _maybe_downsample_series(tt, yy, max_points, downsample_method)
            fig_p.add_trace(go.Scattergl(
                x=tt, y=yy, mode="lines", name=label, line=dict(width=1.2, color=color),
            ))

        noise = prep.get("noise")
        if noise is not None and "corrected" in show_stages:
            fig_p.add_hline(
                y=noise["threshold"],
                line_dash="dot",
                line_color="green",
                annotation_text=f"+threshold ({noise['threshold']:.1f})",
                annotation_position="bottom right",
            )
            st.caption(
                f"Blank noise floor: σ={noise['sigma']:.2f}, "
                f"threshold={noise['threshold']:.2f} (for reference only — not detection)."
            )

        fig_p.update_layout(
            height=380,
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis_title="Time (s)",
            yaxis_title="ADC",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            hovermode="x unified",
        )
        st.plotly_chart(fig_p, use_container_width=True)

    if result is not None:
        st.subheader("Detection summary (optional)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Detected peaks", result["n_valid"])
        m2.metric("high", result["class_counts"].get("high", 0))
        m3.metric("medium", result["class_counts"].get("medium", 0))
        m4.metric("clusters", len(result["clusters"]))
        st.dataframe(pd.DataFrame(result["events_table"]), use_container_width=True)

    with st.expander("Window statistics (raw browse)"):
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
        "Workflow: **Load / convert** → **Run preprocessing** to inspect denoised / "
        "baseline-corrected Sensor 2. Plot points use **LTTB** "
        "([Steinarsson 2013](https://skemman.is/bitstream/1946/15343/3/SS_MSthesis.pdf)). "
        "Peak detection is optional and lives under the teammate expander."
    )
else:
    st.write(
        "Enter a path to a Marginum CSV/NPY in the sidebar, then click **Load / convert**. "
        "Next click **Run preprocessing** for noise reduction + cleaned plots."
    )
