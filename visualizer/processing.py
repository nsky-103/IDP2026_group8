"""
Wrappers around Method B detection (core: Part A → Part B).

Prefer the already-loaded Recording summary in memory so Detect does not
re-scan a multi-GB CSV. Falls back to ``pipeline.analyse`` (disk cache / CSV).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

CORE_ROOT_CANDIDATES = [
    REPO_ROOT,
    REPO_ROOT / "IDP2026_group8",
    REPO_ROOT.parent / "IDP2026_group8",
]


def _find_core_root() -> Path:
    for folder in CORE_ROOT_CANDIDATES:
        if (folder / "core" / "pipeline.py").exists():
            return folder
    searched = "\n".join(f"  {p}" for p in CORE_ROOT_CANDIDATES)
    raise FileNotFoundError(f"core package not found. Looked in:\n{searched}")


def _load_core():
    root = str(_find_core_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from core import detection, pipeline, preprocessing

    return pipeline, detection, preprocessing


def _csv_for_core(source: Path) -> Path:
    if source.suffix.lower() == ".csv":
        return source
    sibling = source.with_suffix(".csv")
    if sibling.exists():
        return sibling
    raise ValueError(
        "Method B needs the original CSV path "
        "(or a sibling .csv next to the loaded file)."
    )


def _events_from_core_pieces(pieces, peaks, clean) -> list[dict]:
    """Map core pieces to the visualizer event schema (one marker per piece).

    Per-peak markers were dropped: a multi-hour file can yield tens of thousands
    of adc2 peaks, which freezes Plotly/Streamlit after Detect.
    """
    events: list[dict] = []
    if pieces is None or len(pieces) == 0:
        return events

    fs = float(clean["fs"])
    adc2 = clean["adc2"]

    for _, row in pieces.iterrows():
        start_s = float(row["start_s"])
        end_s = float(row["end_s"])
        i0 = max(0, int(start_s * fs))
        i1 = min(len(adc2), max(i0 + 1, int(end_s * fs)))
        if i1 > i0 and len(adc2):
            local = int(np.argmax(adc2[i0:i1])) + i0
            t_peak = local / fs
        else:
            t_peak = 0.5 * (start_s + end_s)

        label = str(row["label"])
        valid = label in {"dyed", "solid_no_dye"}
        events.append({
            "polarity": "positive",
            "t_peak": float(t_peak),
            "amplitude": float(row["max_height"]),
            "width_sec": float(row["duration_s"]),
            "channel_ratio": 1.0 if bool(row["confirmed"]) else 0.0,
            "class": label,
            "valid": valid,
            "reject_reasons": [] if valid else [f"label={label}"],
            "n_peaks": int(row["n_peaks"]),
        })

    events.sort(key=lambda e: e["t_peak"])
    return events


def _analyse_from_summary(
    detection,
    preprocessing,
    summary: dict,
    *,
    index: Any = None,
    source_label: str = "memory",
) -> dict:
    """Part A + Part B only — no CSV re-read."""
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    timings[f"read ({source_label})"] = 0.0

    t0 = time.perf_counter()
    clean = preprocessing.preprocess(summary["mean"], fs=float(summary["fs_hz"]))
    timings["Part A: clean"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    pieces, peaks = detection.find_pieces(clean)
    timings["Part B: pieces"] = time.perf_counter() - t0

    return {
        "index": index,
        "summary": summary,
        "clean": clean,
        "pieces": pieces,
        "peaks": peaks,
        "from_cache": True,
        "timings": timings,
    }


def run_detection(source_path: str | Path, recording: Any | None = None) -> dict:
    """Run Method B Part A/B.

    If ``recording`` (or the load-job registry) already has a 1 kHz summary,
    Detect skips the multi-GB CSV scan. Otherwise falls back to
    ``pipeline.analyse`` (disk cache, then CSV).
    """
    pipeline, detection, preprocessing = _load_core()
    csv_path = _csv_for_core(Path(source_path)).resolve()

    # 1) Explicit Recording from the app
    rec = recording
    # 2) Same process registry filled by load_jobs
    if rec is None or getattr(rec, "summary", None) is None:
        try:
            import load_jobs

            rec = load_jobs.get_recording(csv_path)
        except Exception:
            rec = None

    if rec is not None and getattr(rec, "summary", None) is not None:
        raw = _analyse_from_summary(
            detection,
            preprocessing,
            rec.summary,
            index=getattr(rec, "index", None),
            source_label="memory",
        )
    else:
        raw = pipeline.analyse(csv_path)

    events = _events_from_core_pieces(raw["pieces"], raw["peaks"], raw["clean"])
    n = len(raw["clean"]["adc2"])
    sigma = float(raw["clean"]["sigma_adc2"])
    result = {
        "path": str(csv_path),
        "events": events,
        "pieces": raw["pieces"],
        "peaks": raw["peaks"],
        "clean": raw["clean"],
        "timings": raw["timings"],
        "from_cache": raw["from_cache"],
        "clusters": [],
        "noise": {
            "sigma": sigma,
            "threshold": sigma * float(detection.K_SIGMA),
        },
        "proc2": {
            "t": np.arange(n, dtype=np.float64) / float(raw["clean"]["fs"]),
            "corrected": raw["clean"]["adc2"],
        },
        "backend": "core",
    }
    return _add_summary(result)


def _add_summary(result: dict) -> dict:
    for event in result["events"]:
        event.setdefault("class", "event" if event["valid"] else "rejected")

    result["n_valid"] = sum(1 for e in result["events"] if e["valid"])
    result["events_table"] = [
        {
            "t_peak (s)": round(e["t_peak"], 2),
            "polarity": e["polarity"],
            "amplitude": round(e["amplitude"], 2),
            "width (s)": round(e["width_sec"], 2),
            "adc2/adc1 ratio": round(e["channel_ratio"], 2),
            "valid": e["valid"],
            "class": e.get("class", ""),
            "reject reasons": "; ".join(e["reject_reasons"]),
        }
        for e in result["events"]
    ]
    pieces = result.get("pieces")
    if pieces is not None and hasattr(pieces, "to_dict"):
        result["pieces_table"] = pieces.to_dict(orient="records")
    return result
