"""Chunked CSV → binary conversion and memory-mapped signal access."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

DEFAULT_FS_HZ = 50_000.0
DTYPE = np.int16


def npy_path_for(csv_path: Path) -> Path:
    return csv_path.with_suffix(".npy")


def count_csv_rows(csv_path: Path) -> int:
    """Count data rows (excluding header) without loading values."""
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for _ in reader)


def convert_csv_to_npy(
    csv_path: Path,
    npy_path: Path | None = None,
    chunk_rows: int = 500_000,
    progress_callback=None,
) -> Path:
    """
    Convert a two-column ADC CSV into a memmap-friendly (N, 2) int16 .npy file.

    Reads the CSV in chunks so multi-GB files do not need to fit in RAM.
    """
    csv_path = Path(csv_path)
    npy_path = Path(npy_path) if npy_path else npy_path_for(csv_path)

    n_rows = count_csv_rows(csv_path)
    if n_rows == 0:
        raise ValueError(f"No data rows in {csv_path}")

    out = np.lib.format.open_memmap(
        npy_path, mode="w+", dtype=DTYPE, shape=(n_rows, 2)
    )

    written = 0
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        buf: list[list[int]] = []
        for row in reader:
            buf.append([int(row[0]), int(row[1])])
            if len(buf) >= chunk_rows:
                chunk = np.asarray(buf, dtype=DTYPE)
                out[written : written + len(chunk)] = chunk
                written += len(chunk)
                buf.clear()
                if progress_callback:
                    progress_callback(written / n_rows)

        if buf:
            chunk = np.asarray(buf, dtype=DTYPE)
            out[written : written + len(chunk)] = chunk
            written += len(chunk)
            if progress_callback:
                progress_callback(1.0)

    out.flush()
    del out
    return npy_path


def open_signal(npy_path: Path) -> np.memmap:
    """Open a (N, 2) signal as a read-only memory map."""
    arr = np.load(npy_path, mmap_mode="r")
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected shape (N, 2), got {arr.shape} in {npy_path}")
    return arr


def ensure_npy(path: Path, progress_callback=None) -> Path:
    """Return .npy path, converting from CSV if needed."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return path
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Unsupported file type: {path.suffix}")
    out = npy_path_for(path)
    if out.exists() and out.stat().st_mtime >= path.stat().st_mtime:
        return out
    return convert_csv_to_npy(path, out, progress_callback=progress_callback)


def duration_s(n_samples: int, fs_hz: float = DEFAULT_FS_HZ) -> float:
    return n_samples / fs_hz


def sample_range(start_s: float, end_s: float, n_samples: int, fs_hz: float = DEFAULT_FS_HZ) -> tuple[int, int]:
    start_i = max(0, int(start_s * fs_hz))
    end_i = min(n_samples, int(end_s * fs_hz))
    if end_i <= start_i:
        end_i = min(n_samples, start_i + 1)
    return start_i, end_i


def load_window(
    signal: np.ndarray,
    start_s: float,
    end_s: float,
    fs_hz: float = DEFAULT_FS_HZ,
    max_points: int = 20_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a time window and downsample for plotting.

    Returns (t_seconds, sensor1, sensor2), each of length ≤ max_points.
    """
    n = signal.shape[0]
    start_i, end_i = sample_range(start_s, end_s, n, fs_hz)
    window = signal[start_i:end_i]

    n_win = window.shape[0]
    stride = max(1, n_win // max_points)
    idx = np.arange(0, n_win, stride)
    samples = start_i + idx
    t = samples / fs_hz
    s1 = window[idx, 0].astype(np.float64)
    s2 = window[idx, 1].astype(np.float64)
    return t, s1, s2
