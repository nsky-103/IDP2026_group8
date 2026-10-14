"""
Wrappers around detection backends used by the Streamlit app.

Backends
--------
- Method A: ``fluosense_code_only`` (legacy / Task-2 threshold pipeline)
- Method B: in-repo ``core`` package (fast_io → Part A preprocess → Part B pieces)

No Streamlit here: plain arguments in, plain dicts out.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

FLUOSENSE_CANDIDATES = [
    REPO_ROOT / "fluosense_code_only",
    REPO_ROOT.parent / "fluosense_code_only",
    REPO_ROOT / "IDP2026_group8" / "fluosense_code_only",
]

# Directory that contains the ``core`` package (pipeline-a-to-b).
CORE_ROOT_CANDIDATES = [
    REPO_ROOT,  # when this file lives in IDP2026_group8/visualizer/
    REPO_ROOT / "IDP2026_group8",
    REPO_ROOT.parent / "IDP2026_group8",
]

METHOD_A = "Method A · Threshold"
METHOD_B = "Method B · Part A/B (core)"


def _find_fluosense() -> Path:
    for folder in FLUOSENSE_CANDIDATES:
        if (folder / "src" / "pipeline.py").exists():
            return folder
    searched = "\n".join(f"  {p}" for p in FLUOSENSE_CANDIDATES)
    raise FileNotFoundError(f"fluosense_code_only not found. Looked in:\n{searched}")


def _find_core_root() -> Path:
    for folder in CORE_ROOT_CANDIDATES:
        if (folder / "core" / "pipeline.py").exists():
            return folder
    searched = "\n".join(f"  {p}" for p in CORE_ROOT_CANDIDATES)
    raise FileNotFoundError(f"core package not found. Looked in:\n{searched}")


def _load_fluosense():
    folder = str(_find_fluosense())
    if folder not in sys.path:
        sys.path.insert(0, folder)

    from src import config, detect, io_utils, pipeline, preprocess

    for module in (config, detect, preprocess, io_utils, pipeline):
        importlib.reload(module)
    return config, io_utils, pipeline, preprocess, detect


def _load_core():
    root = str(_find_core_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from core import detection, pipeline, preprocessing

    for module in (preprocessing, detection, pipeline):
        importlib.reload(module)
    return pipeline, detection, preprocessing


def run_preprocess(source_path: str | Path, fs_hz: float) -> dict:
    """Downsample → denoise → baseline-correct both channels (fluosense path)."""
    config, io_utils, _, preprocess, _detect = _load_fluosense()
    config.FS_HZ = float(fs_hz)

    path = Path(source_path)
    if path.suffix.lower() == ".npy":
        t, adc1, adc2, fs_eff = io_utils.load_downsampled(
            str(path), fs_hz, config.TARGET_RATE_HZ
        )
        return {
            "proc1": preprocess.preprocess_channel(
                t, adc1, fs_eff, already_downsampled=True
            ),
            "proc2": preprocess.preprocess_channel(
                t, adc2, fs_eff, already_downsampled=True
            ),
        }

    df = io_utils.load_raw(str(path), fs_hz)
    t = df["t"].values
    return {
        "proc1": preprocess.preprocess_channel(t, df["adc1"].values, fs_hz),
        "proc2": preprocess.preprocess_channel(t, df["adc2"].values, fs_hz),
    }


def _blank_noise(blank_path: Path, fs_hz: float, config, io_utils, pipeline, preprocess, detect):
    suffix = blank_path.suffix.lower()
    if suffix == ".csv":
        return pipeline.run_pipeline(str(blank_path), fs_hz=fs_hz)["noise"]
    if suffix == ".npy":
        t, _adc1, adc2, fs_eff = io_utils.load_downsampled(
            str(blank_path), fs_hz, config.TARGET_RATE_HZ
        )
        blank_proc2 = preprocess.preprocess_channel(
            t, adc2, fs_eff, already_downsampled=True
        )
        return detect.noise_stats(blank_proc2["corrected"])
    raise ValueError(f"Blank must be .csv or .npy, got {blank_path.suffix}")


def _detect_from_npy(npy_path: Path, fs_hz: float, blank_noise, config, io_utils, pipeline):
    _t, adc1, adc2, fs_eff = io_utils.load_downsampled(
        str(npy_path), fs_hz, config.TARGET_RATE_HZ
    )
    return pipeline.run_from_arrays(
        adc1,
        adc2,
        fs_hz=fs_eff,
        blank_noise=blank_noise,
        target_hz=config.TARGET_RATE_HZ,
        already_downsampled=True,
    )


def _csv_for_core(source: Path) -> Path:
    """Method B reads CSV via fast_io; accept sibling .csv next to a .npy."""
    if source.suffix.lower() == ".csv":
        return source
    sibling = source.with_suffix(".csv")
    if sibling.exists():
        return sibling
    raise ValueError(
        "Method B (core) needs the original CSV path "
        "(or a sibling .csv next to the loaded .npy)."
    )


def _events_from_core_pieces(pieces, peaks, clean) -> list[dict]:
    """Map core piece / peak tables to the visualizer event schema."""
    events: list[dict] = []
    fs = float(clean["fs"])
    adc2 = clean["adc2"]

    # Prefer one marker per adc2 peak, labelled by the enclosing piece.
    peak_times = list(peaks.get("adc2", []))
    if len(peak_times) == 0 and pieces is not None and len(pieces):
        for _, row in pieces.iterrows():
            peak_times.append(0.5 * (float(row["start_s"]) + float(row["end_s"])))

    for t_peak in peak_times:
        label = "water"
        amplitude = 0.0
        width_sec = 0.0
        confirmed = False
        if pieces is not None and len(pieces):
            hit = pieces[(pieces["start_s"] <= t_peak) & (t_peak <= pieces["end_s"])]
            if len(hit):
                row = hit.iloc[0]
                label = str(row["label"])
                amplitude = float(row["max_height"])
                width_sec = float(row["duration_s"])
                confirmed = bool(row["confirmed"])
        if amplitude == 0.0 and len(adc2):
            i = int(round(t_peak * fs))
            i = max(0, min(len(adc2) - 1, i))
            amplitude = float(adc2[i])

        valid = label in {"dyed", "solid_no_dye"}
        events.append({
            "polarity": "positive",
            "t_peak": float(t_peak),
            "amplitude": float(amplitude),
            "width_sec": float(width_sec),
            "channel_ratio": 1.0 if confirmed else 0.0,
            "class": label,
            "valid": valid,
            "reject_reasons": [] if valid else [f"label={label}"],
        })

    events.sort(key=lambda e: e["t_peak"])
    return events


def run_detection_core(source_path: str | Path) -> dict:
    """Run Method B: core.fast_io → Part A → Part B."""
    pipeline, detection, _preprocessing = _load_core()
    csv_path = _csv_for_core(Path(source_path))
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


def run_detection_fluosense(
    source_path: str | Path,
    fs_hz: float,
    blank_path: str,
    sigma_mult: float,
    channel_ratio: float,
) -> dict:
    """Run Method A (fluosense)."""
    source = Path(source_path)
    suffix = source.suffix.lower()
    if suffix not in {".csv", ".npy"}:
        raise ValueError(f"Detection needs a .csv or .npy file, got {source.suffix}")

    config, io_utils, pipeline, preprocess, detect = _load_fluosense()
    config.FS_HZ = float(fs_hz)
    config.NOISE_SIGMA_MULT = float(sigma_mult)
    config.MIN_CHANNEL_RATIO = float(channel_ratio)

    blank_noise = None
    blank_path = blank_path.strip()
    if blank_path:
        blank = Path(blank_path)
        if not blank.exists():
            raise FileNotFoundError(f"Blank file not found: {blank_path}")
        blank_noise = _blank_noise(
            blank, fs_hz, config, io_utils, pipeline, preprocess, detect
        )

    if suffix == ".npy":
        result = _detect_from_npy(
            source, fs_hz, blank_noise, config, io_utils, pipeline
        )
    else:
        result = pipeline.run_pipeline(
            str(source), fs_hz=fs_hz, blank_noise=blank_noise
        )
    result["backend"] = "fluosense"
    return _add_summary(result)


def run_detection(
    source_path: str | Path,
    fs_hz: float,
    blank_path: str,
    sigma_mult: float,
    channel_ratio: float,
    algorithm: str = METHOD_A,
) -> dict:
    """Dispatch to Method A (fluosense) or Method B (core Part A/B)."""
    if algorithm == METHOD_B or "Method B" in str(algorithm) or "Part A/B" in str(algorithm):
        return run_detection_core(source_path)
    return run_detection_fluosense(
        source_path, fs_hz, blank_path, sigma_mult, channel_ratio
    )


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
    # Optional pieces table for Method B
    pieces = result.get("pieces")
    if pieces is not None and hasattr(pieces, "to_dict"):
        result["pieces_table"] = pieces.to_dict(orient="records")
    return result
