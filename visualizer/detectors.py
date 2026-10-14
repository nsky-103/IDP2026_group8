"""Detection algorithms that the visualizer (or any script) can choose from.

Every detector is a plain function

    def my_detector(csv_path, fs_hz, **params) -> Detection

registered in ``DETECTORS`` together with its parameters. The visualizer builds its
controls from the ``Param`` list, so adding an algorithm needs no UI code.

``Detection.events`` is one row per detected event:

    required  t_peak (s), amplitude, valid (bool), label (str)
    optional  start_s, end_s, polarity ("positive" / "negative"), size_mm
    anything else is shown as-is in the detection table

No Streamlit here.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_COLUMNS = ["t_peak", "amplitude", "valid", "label"]


@dataclass
class Detection:
    events: pd.DataFrame
    trace: tuple[np.ndarray, np.ndarray] | None = None  # (t, y) the detector worked on
    threshold: float | None = None                     # on ``trace``
    summary: dict[str, float] = field(default_factory=dict)  # numbers for the summary box


@dataclass
class Param:
    """One tunable parameter. Type of ``default`` decides the widget: float/int, bool or str."""

    key: str
    label: str
    default: float | int | bool | str
    min: float | None = None
    max: float | None = None
    step: float | None = None
    help: str = ""


@dataclass
class Detector:
    name: str
    run: Callable[..., Detection]
    params: list[Param] = field(default_factory=list)
    description: str = ""


def run(detector: Detector, csv_path: str | Path, fs_hz: float, params: dict) -> Detection:
    """Run a detector and check that its result follows the format above."""
    if Path(csv_path).suffix.lower() != ".csv":
        raise ValueError("Detection needs the original CSV file, not the .npy copy.")
    result = detector.run(Path(csv_path), float(fs_hz), **params)
    missing = [c for c in REQUIRED_COLUMNS if c not in result.events.columns]
    if missing:
        raise ValueError(f"{detector.name} returned events without columns {missing}")
    events = result.events[REQUIRED_COLUMNS + [c for c in result.events if c not in REQUIRED_COLUMNS]]
    if "polarity" not in events:
        events = events.assign(polarity="positive")
    result.events = events.astype({"valid": bool}).sort_values("t_peak", ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# Pieces: this repository's pipeline (core/pipeline.py)
# ---------------------------------------------------------------------------

def core_pieces(csv_path: Path, fs_hz: float, k_sigma: float, gap_s: float,
                no_dye_level: float) -> Detection:
    """CSV -> 1 kHz -> Part A (clean) -> Part B (pieces). One event per aspirated piece."""
    from core import detection, pipeline, preprocessing

    _, summary, _ = pipeline.open_recording(csv_path)
    clean = preprocessing.preprocess(summary["mean"], fs=summary["fs_hz"])
    pieces, _ = detection.find_pieces(clean, k=k_sigma, gap_s=gap_s, no_dye_level=no_dye_level)

    fs, a2 = clean["fs"], clean["adc2"]
    t_peak = [
        (int(s * fs) + int(np.argmax(a2[int(s * fs):max(int(e * fs), int(s * fs) + 1)]))) / fs
        for s, e in zip(pieces["start_s"], pieces["end_s"])
    ]
    events = pieces.assign(t_peak=t_peak, amplitude=pieces["max_height"], valid=True)
    events = events.drop(columns="max_height")
    return Detection(
        events=events,
        trace=(np.arange(len(a2)) / fs, a2),
        threshold=k_sigma * clean["sigma_adc2"],
        summary={"Pieces": len(events), "Noise sigma (adc2)": round(clean["sigma_adc2"], 2)},
    )


# ---------------------------------------------------------------------------
# Threshold: the fluosense pipeline (lives outside this repository)
# ---------------------------------------------------------------------------

FLUOSENSE_CANDIDATES = [
    REPO_ROOT / "fluosense_code_only",
    REPO_ROOT.parent / "fluosense_code_only",
]
_BLANK = REPO_ROOT / "idp2023" / "Test signals from previous years" / "just water.csv"


def _load_fluosense():
    """Import the fluosense modules (reloaded so edits to them show up immediately)."""
    folder = next((f for f in FLUOSENSE_CANDIDATES if (f / "src" / "pipeline.py").exists()), None)
    if folder is None:
        searched = "\n".join(f"  {p}" for p in FLUOSENSE_CANDIDATES)
        raise FileNotFoundError(f"fluosense_code_only not found. Looked in:\n{searched}")
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

    from src import config, detect, io_utils, pipeline, preprocess

    for module in (config, detect, preprocess, io_utils, pipeline):
        importlib.reload(module)
    return config, pipeline


def fluosense_threshold(csv_path: Path, fs_hz: float, sigma_mult: float, channel_ratio: float,
                        blank_path: str) -> Detection:
    """Peaks above a noise threshold on adc2, checked against adc1 (fluosense pipeline).

    ``blank_path`` (optional) is a just-water recording; its noise sets the threshold.
    Without it, the recording's own noise is used.
    """
    config, pipeline = _load_fluosense()
    config.FS_HZ = fs_hz
    config.NOISE_SIGMA_MULT = sigma_mult
    config.MIN_CHANNEL_RATIO = channel_ratio

    blank_noise = None
    if blank_path.strip():
        if not Path(blank_path.strip()).exists():
            raise FileNotFoundError(f"Blank file not found: {blank_path}")
        blank_noise = pipeline.run_pipeline(blank_path.strip(), fs_hz=fs_hz)["noise"]

    result = pipeline.run_pipeline(str(csv_path), fs_hz=fs_hz, blank_noise=blank_noise)
    events = pd.DataFrame([
        {
            "t_peak": e["t_peak"],
            "start_s": e["t_left"],
            "end_s": e["t_right"],
            "amplitude": e["amplitude"],
            "polarity": e["polarity"],
            "valid": e["valid"],
            "label": "event" if e["valid"] else "rejected",
            "width_s": e["width_sec"],
            "channel_ratio": e["channel_ratio"],
            "reject_reasons": "; ".join(e["reject_reasons"]),
        }
        for e in result["events"]
    ], columns=REQUIRED_COLUMNS + ["start_s", "end_s", "polarity", "width_s", "channel_ratio",
                                   "reject_reasons"])
    proc2 = result["proc2"]
    return Detection(
        events=events,
        trace=(proc2["t"], proc2["corrected"]),
        threshold=result["noise"]["threshold"],
        summary={"Detected peaks": int(events["valid"].sum()), "Clusters": len(result["clusters"])},
    )


# ---------------------------------------------------------------------------
# Registry: the order here is the order in the app's algorithm menu
# ---------------------------------------------------------------------------

DETECTORS: dict[str, Detector] = {d.name: d for d in [
    Detector(
        name="Pieces · core pipeline",
        run=core_pieces,
        description="1 kHz -> median + Savitzky-Golay + ALS -> peaks grouped into pieces and labelled.",
        params=[
            Param("k_sigma", "Peak prominence (x sigma)", 10.0, 3.0, 30.0, 0.5),
            Param("gap_s", "Max gap inside one piece (s)", 1.0, 0.1, 5.0, 0.1),
            Param("no_dye_level", "No-dye level (counts)", 68.9, 0.0, 300.0, 0.1),
        ],
    ),
    Detector(
        name="Threshold · fluosense",
        run=fluosense_threshold,
        description="External fluosense_code_only pipeline (not in this repository).",
        params=[
            Param("sigma_mult", "Noise sigma multiplier", 5.0, 3.0, 20.0, 0.5),
            Param("channel_ratio", "Min adc2/adc1 ratio", 1.3, 0.5, 3.0, 0.1),
            Param("blank_path", "Blank (just water) CSV, optional", str(_BLANK) if _BLANK.exists() else ""),
        ],
    ),
]}
