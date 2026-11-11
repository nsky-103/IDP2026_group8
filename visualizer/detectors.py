"""Detection algorithms that the visualizer (or any script) can choose from.

Every detector is a plain function

    def my_detector(recording, **params) -> Detection

registered in ``DETECTORS`` together with its parameters. ``recording`` is the loaded
``signal_io.Recording`` (path, row index and 1 kHz summary), so detection does not re-read
the CSV. The visualizer builds its controls from the ``Param`` list, so adding an
algorithm needs no UI code.

``Detection.events`` is one row per detected event:

    required  t_peak (s), amplitude, valid (bool), label (str)
    optional  start_s, end_s, polarity ("positive" / "negative"), size_mm
    anything else is shown as-is in the detection table

No Streamlit here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

import signal_io

REQUIRED_COLUMNS = ["t_peak", "amplitude", "valid", "label"]


@dataclass
class Detection:
    events: pd.DataFrame
    trace: tuple[np.ndarray, np.ndarray] | None = None  # (t, y) the detector worked on
    threshold: float | None = None                     # on ``trace``
    summary: dict[str, float] = field(default_factory=dict)  # numbers for the summary box
    timings: dict[str, float] = field(default_factory=dict)  # seconds per step


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


def run(detector: Detector, recording: signal_io.Recording, params: dict) -> Detection:
    """Run a detector and check that its result follows the format above."""
    if recording.summary is None:
        raise ValueError("Wait until the 1 kHz summary is ready before detecting.")
    result = detector.run(recording, **params)
    missing = [c for c in REQUIRED_COLUMNS if c not in result.events.columns]
    if missing:
        raise ValueError(f"{detector.name} returned events without columns {missing}")
    events = result.events[REQUIRED_COLUMNS + [c for c in result.events if c not in REQUIRED_COLUMNS]]
    if "polarity" not in events:
        events = events.assign(polarity="positive")
    result.events = events.astype({"valid": bool}).sort_values("t_peak", ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# Method B: this repository's pipeline (core Part A -> Part B)
# ---------------------------------------------------------------------------

VALID_LABELS = {"dyed", "solid_no_dye"}  # water pieces are listed as rejected


def core_pieces(recording: signal_io.Recording, k_sigma: float, gap_s: float,
                no_dye_level: float) -> Detection:
    """1 kHz summary -> Part A (clean) -> Part B (pieces). One event per aspirated piece.

    One event per piece, not per peak: a multi-hour file can have tens of thousands of
    adc2 peaks, which freezes the browser.
    """
    signal_io._ensure_core_on_path()
    from core import detection, preprocessing

    t0 = time.perf_counter()
    # Prefer Part A precomputed during load (#17); otherwise run it now.
    if recording.clean is not None:
        clean = recording.clean
        t1 = time.perf_counter()
        part_a_tag = "Part A: clean (memory)"
    else:
        clean = preprocessing.preprocess(recording.summary["mean"], fs=recording.summary_fs_hz)
        recording.clean = clean
        t1 = time.perf_counter()
        part_a_tag = "Part A: clean"
    pieces, _ = detection.find_pieces(clean, k=k_sigma, gap_s=gap_s, no_dye_level=no_dye_level)
    t2 = time.perf_counter()
    amt = detection.amount(pieces)

    fs, a2 = clean["fs"], clean["adc2"]
    t_peak = [
        (int(s * fs) + int(np.argmax(a2[int(s * fs):max(int(e * fs), int(s * fs) + 1)]))) / fs
        for s, e in zip(pieces["start_s"], pieces["end_s"])
    ]
    events = pieces.assign(t_peak=t_peak, amplitude=pieces["max_height"],
                           valid=pieces["label"].isin(VALID_LABELS))
    events = events.drop(columns="max_height")
    return Detection(
        events=events,
        trace=(np.arange(len(a2)) / fs, a2),
        threshold=k_sigma * clean["sigma_adc2"],
        summary={
            "Detected pieces (valid)": int(events["valid"].sum()),
            "All pieces": len(events),
            "Amount: total area (counts·ms)": round(amt["area"]),
            "Amount: dyed pieces": int(amt["dyed"]),
            "Noise sigma (adc2)": round(clean["sigma_adc2"], 2),
        },
        timings={part_a_tag: round(t1 - t0, 3), "Part B: pieces": round(t2 - t1, 3)},
    )


# ---------------------------------------------------------------------------
# Registry: the order here is the order in the app's algorithm menu
# ---------------------------------------------------------------------------

DETECTORS: dict[str, Detector] = {d.name: d for d in [
    Detector(
        name="Method B · core Part A/B",
        run=core_pieces,
        description="1 kHz summary -> median + Savitzky-Golay + ALS -> peaks grouped into pieces "
                    "and labelled (dyed / solid_no_dye / water).",
        params=[
            Param("k_sigma", "Peak prominence (x sigma)", 10.0, 3.0, 30.0, 0.5),
            Param("gap_s", "Max gap inside one piece (s)", 1.0, 0.1, 5.0, 0.1),
            Param("no_dye_level", "No-dye level (counts)", 68.9, 0.0, 300.0, 0.1),
        ],
    ),
]}
