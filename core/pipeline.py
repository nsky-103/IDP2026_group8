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
from pathlib import Path

import numpy as np

from core import detection, fast_io, preprocessing

DEFAULT_CACHE = Path(os.environ.get("IDP_CACHE_DIR", Path.home() / ".cache" / "idp2026"))


# ---- Idea: Unmona (udgithubit/IDP, idp/fast.py: convert writes a small binary once, detect reuses it)
# Save the reduced data once and read it back on the next open instead of the multi-GB CSV.
# Stored outside the repository (measurement-derived data must not be committed).
def _cache_dir(csv_path: Path, cache_root: Path) -> Path:
    st = csv_path.stat()
    key = f"{csv_path.resolve()}|{st.st_size}|{st.st_mtime_ns}"
    return cache_root / f"{csv_path.stem[:40]}_{hashlib.sha1(key.encode()).hexdigest()[:12]}"


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
    clean = preprocessing.preprocess(summary["mean"], fs=summary["fs_hz"])
    timings["Part A: clean"] = time.perf_counter() - t

    t = time.perf_counter()
    pieces, peaks = detection.find_pieces(clean)
    timings["Part B: pieces"] = time.perf_counter() - t

    return {"index": index, "summary": summary, "clean": clean, "pieces": pieces, "peaks": peaks,
            "from_cache": from_cache, "timings": timings}
