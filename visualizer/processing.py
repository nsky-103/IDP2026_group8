"""
Wrappers around the fluosense pipeline (noise reduction + peak detection).

The pipeline code lives outside this repo in a folder called
``fluosense_code_only``; its Python package is named ``src``, which is why
we import ``from src import ...`` below.

No Streamlit here: everything takes plain arguments and returns plain data, so it
can be used from notebooks / scripts / tests as well as from the visualizer app.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Places we look for the fluosense_code_only folder (first one that exists wins).
FLUOSENSE_CANDIDATES = [
    REPO_ROOT / "fluosense_code_only",          # Industrial Project/fluosense_code_only
    REPO_ROOT.parent / "fluosense_code_only",   # next to IDP2026_group8 clone
    REPO_ROOT / "IDP2026_group8" / "fluosense_code_only",
]


def _find_fluosense() -> Path:
    for folder in FLUOSENSE_CANDIDATES:
        if (folder / "src" / "pipeline.py").exists():
            return folder
    searched = "\n".join(f"  {p}" for p in FLUOSENSE_CANDIDATES)
    raise FileNotFoundError(f"fluosense_code_only not found. Looked in:\n{searched}")


def _load_fluosense():
    """Import the fluosense modules (reloaded so edits to them show up immediately)."""
    folder = str(_find_fluosense())
    if folder not in sys.path:
        sys.path.insert(0, folder)

    from src import config, detect, io_utils, pipeline, preprocess

    for module in (config, detect, preprocess, io_utils, pipeline):
        importlib.reload(module)
    return config, io_utils, pipeline, preprocess, detect


def run_preprocess(source_path: str | Path, fs_hz: float) -> dict:
    """Downsample → denoise → baseline-correct both channels (no peak detection)."""
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
    """Noise floor from a blank recording (CSV or NPY)."""
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
    """Peak detection on a memmap .npy without re-reading a multi-GB CSV."""
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


def run_detection(
    source_path: str | Path,
    fs_hz: float,
    blank_path: str,
    sigma_mult: float,
    channel_ratio: float,
) -> dict:
    """Run the full pipeline (preprocess → detect → cluster) on one recording.

    Accepts ``.csv`` or ``.npy``. NPY uses strided memmap load so ~GB files
    do not need a full float copy of the raw recording.

    ``blank_path`` (optional) is a just-water CSV/NPY; its noise sets the
    detection threshold. Without it, the recording's own noise is used.
    """
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
    return _add_summary(result)


def _add_summary(result: dict) -> dict:
    """Add the extra fields the app shows (count, table, class) to a pipeline result."""
    for event in result["events"]:
        # The pipeline does not classify events yet; the app shows this label.
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
            "reject reasons": "; ".join(e["reject_reasons"]),
        }
        for e in result["events"]
    ]
    return result
