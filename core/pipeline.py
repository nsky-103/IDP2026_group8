"""Whole analysis of one recording: CSV -> 1 kHz -> Part A -> Part B.

    result = analyse("recording.csv")
    result["pieces"]     # one row per aspirated piece (see detection.py)
    result["timings"]    # seconds per step

The row index and the 1 kHz summary are saved in a cache folder the first time, so opening
the same file again skips reading the CSV. Cleaning and detection run again every time
(about 2 s for 4.7 h), so changed parameters always take effect.

Author: Sora, except the cache marked below (Unmona's idea).
"""

from __future__ import annotations

import hashlib
import os
import time
import unicodedata
from pathlib import Path

import numpy as np

from core import detection, fast_io, preprocessing

DEFAULT_CACHE = Path(os.environ.get("IDP_CACHE_DIR", Path.home() / ".cache" / "idp2026"))
_FINGERPRINT_BYTES = 1 << 16


# ---- Idea: Unmona (udgithubit/IDP, idp/fast.py: convert writes a small binary once, detect reuses it)
# Save the reduced data once and read it back on the next open instead of the multi-GB CSV.
# Stored outside the repository (measurement-derived data must not be committed).
def _cache_dir(csv_path: Path, cache_root: Path) -> Path:
    """Cache folder of a recording, found from its content, not its path (GitHub #18).

    The key is the file size plus 64 KB read at five places (start, quarter, half, three
    quarters, end). The same recording therefore hits the cache however its path is written
    (macOS paths pasted from Finder use a different Unicode form) and when it was copied
    (e.g. uploaded to a temp folder, which also changes the modification time).
    """
    size = csv_path.stat().st_size
    h = hashlib.sha1(str(size).encode())
    with open(csv_path, "rb") as f:
        for pos in (0, size // 4, size // 2, 3 * size // 4, max(0, size - _FINGERPRINT_BYTES)):
            f.seek(pos)
            h.update(f.read(_FINGERPRINT_BYTES))
    key = h.hexdigest()[:12]
    for existing in cache_root.glob(f"*_{key}") if cache_root.is_dir() else []:
        return existing  # same content saved under another file name
    name = unicodedata.normalize("NFC", csv_path.stem)[:40]  # only to make the folder readable
    return cache_root / f"{name}_{key}"


def _save(folder: Path, index: fast_io.CsvIndex, summary: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / "index.npz", byte_starts=index.byte_starts, byte_ends=index.byte_ends,
             row_starts=index.row_starts, n_rows=index.n_rows)
    np.savez(folder / "summary.npz", **summary)


def _load(folder: Path, csv_path: Path) -> tuple[fast_io.CsvIndex, dict] | None:
    try:
        i, s = np.load(folder / "index.npz"), np.load(folder / "summary.npz")
    except (OSError, ValueError):
        return None
    index = fast_io.CsvIndex(csv_path, i["byte_starts"], i["byte_ends"], i["row_starts"], int(i["n_rows"]))
    return index, {k: (float(s[k]) if k == "fs_hz" else s[k]) for k in s.files}


def _part_a_key(params: dict | None = None) -> str:
    """Filename suffix from Part A parameters so a param change misses the cache."""
    p = params or preprocessing.PARAMS
    raw = "|".join(f"{k}={p[k]}" for k in sorted(p))
    return hashlib.sha1(raw.encode()).hexdigest()[:10]


def save_part_a(folder: Path, clean: dict, params: dict | None = None) -> None:
    """Persist Part A output next to the summary (GitHub #17)."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"part_a_{_part_a_key(params)}.npz"
    np.savez(
        path,
        adc1=clean["adc1"], adc2=clean["adc2"],
        sigma_adc1=clean["sigma_adc1"], sigma_adc2=clean["sigma_adc2"],
        fs=clean["fs"],
        adc1_response=clean.get("adc1_response", np.nan),
        adc1_usable=bool(clean.get("adc1_usable", False)),
    )


def load_part_a(folder: Path, params: dict | None = None) -> dict | None:
    path = folder / f"part_a_{_part_a_key(params)}.npz"
    try:
        z = np.load(path)
    except (OSError, ValueError):
        return None
    return {
        "fs": float(z["fs"]),
        "adc1": z["adc1"],
        "adc2": z["adc2"],
        "sigma_adc1": float(z["sigma_adc1"]),
        "sigma_adc2": float(z["sigma_adc2"]),
        "adc1_response": float(z["adc1_response"]) if "adc1_response" in z.files else float("nan"),
        "adc1_usable": bool(z["adc1_usable"]) if "adc1_usable" in z.files else False,
    }
# ---- end of Unmona's idea ---------------------------------------------------------------------


def open_recording(csv_path: str | Path, cache_root: Path | None = DEFAULT_CACHE,
                   workers: int | None = None, progress_callback=None) -> tuple[fast_io.CsvIndex, dict, bool]:
    """Row index and 1 kHz summary, from the cache when possible. Returns (index, summary, from_cache)."""
    csv_path = Path(csv_path)
    folder = _cache_dir(csv_path, cache_root) if cache_root else None
    if folder and (hit := _load(folder, csv_path)):
        return hit[0], hit[1], True
    index = fast_io.build_index(csv_path, workers=workers)
    summary = fast_io.summarize(index, workers=workers, progress_callback=progress_callback)
    if folder:
        _save(folder, index, summary)
    return index, summary, False


def analyse(csv_path: str | Path, cache_root: Path | None = DEFAULT_CACHE, workers: int | None = None,
            progress_callback=None) -> dict:
    timings = {}
    t = time.perf_counter()
    index, summary, from_cache = open_recording(csv_path, cache_root, workers, progress_callback)
    timings["read (cache)" if from_cache else "read CSV -> 1 kHz"] = time.perf_counter() - t

    t = time.perf_counter()
    folder = _cache_dir(Path(csv_path), cache_root) if cache_root else None
    clean = load_part_a(folder) if folder else None
    if clean is None:
        clean = preprocessing.preprocess(summary["mean"], fs=summary["fs_hz"])
        if folder:
            save_part_a(folder, clean)
        timings["Part A: clean"] = time.perf_counter() - t
    else:
        timings["Part A: clean (cache)"] = time.perf_counter() - t

    t = time.perf_counter()
    pieces, peaks = detection.find_pieces(clean)
    timings["Part B: pieces"] = time.perf_counter() - t

    return {"index": index, "summary": summary, "clean": clean, "pieces": pieces, "peaks": peaks,
            "from_cache": from_cache, "timings": timings}
