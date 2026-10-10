"""Chunked CSV → binary conversion and memory-mapped signal access."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

DEFAULT_FS_HZ = 50_000.0
DTYPE = np.int16


def npy_path_for(csv_path: Path) -> Path:
    return csv_path.with_suffix(".npy")


def count_csv_rows(
    csv_path: Path,
    progress_callback=None,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> int:
    """Count data rows (excluding header) by scanning raw bytes for ``\\n``.

    Faster than text ``readline()``: reads multi‑MB binary chunks and counts
    LF bytes. Assumes one physical line per CSV record (true for Marginum
    ``adc1,adc2`` exports). Progress is based on file byte offset (0–35%).
    """
    csv_path = Path(csv_path)
    file_size = max(csv_path.stat().st_size, 1)
    n_newlines = 0
    # If the file does not end with a newline, the last line still counts.
    ends_with_newline = True

    with csv_path.open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            n_newlines += chunk.count(b"\n")
            ends_with_newline = chunk.endswith(b"\n")
            if progress_callback:
                progress_callback(
                    0.35 * (f.tell() / file_size),
                    "Counting rows (binary)…",
                )

    # newline count == number of lines when file ends with \\n; otherwise +1
    n_lines = n_newlines if ends_with_newline else n_newlines + 1
    # subtract header
    n_rows = max(n_lines - 1, 0)
    if progress_callback:
        progress_callback(0.35, "Counting done")
    return n_rows


def convert_csv_to_npy(
    csv_path: Path,
    npy_path: Path | None = None,
    chunk_rows: int = 500_000,
    progress_callback=None,
) -> Path:
    """
    Convert a two-column ADC CSV into a memmap-friendly (N, 2) int16 .npy file.

    Reads the CSV in chunks so multi-GB files do not need to fit in RAM.
    ``progress_callback(fraction, message)`` is optional; ``message`` may be omitted
    by callers that only pass one argument (we accept both).
    """
    csv_path = Path(csv_path)
    npy_path = Path(npy_path) if npy_path else npy_path_for(csv_path)

    def report(p: float, msg: str) -> None:
        if not progress_callback:
            return
        try:
            progress_callback(p, msg)
        except TypeError:
            progress_callback(p)

    n_rows = count_csv_rows(csv_path, progress_callback=report if progress_callback else None)
    if n_rows == 0:
        raise ValueError(f"No data rows in {csv_path}")

    # Write to a temp file first so a crashed convert does not leave a bad .npy
    tmp_path = npy_path.with_suffix(".npy.partial")
    out = np.lib.format.open_memmap(
        tmp_path, mode="w+", dtype=DTYPE, shape=(n_rows, 2)
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
                report(0.35 + 0.65 * (written / n_rows), f"Converting… {written:,}/{n_rows:,}")

        if buf:
            chunk = np.asarray(buf, dtype=DTYPE)
            out[written : written + len(chunk)] = chunk
            written += len(chunk)

    out.flush()
    del out
    report(1.0, "Finishing…")

    if npy_path.exists():
        npy_path.unlink()
    tmp_path.replace(npy_path)
    return npy_path


def open_signal(npy_path: Path) -> np.memmap:
    """Open a (N, 2) signal as a read-only memory map."""
    arr = np.load(npy_path, mmap_mode="r")
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected shape (N, 2), got {arr.shape} in {npy_path}")
    return arr


def ensure_npy(path: Path, progress_callback=None) -> Path:
    """Return .npy path, converting from CSV if needed (blocking)."""
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
    method: str = "lttb",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a time window and downsample for plotting.

    ``method``:
      - ``"lttb"`` — Largest-Triangle-Three-Buckets (Steinarsson 2013), guided
        by Sensor 2 so fluorescence peaks/troughs survive aggressive downsampling.
      - ``"stride"`` — every-nth sample (fast, may miss narrow spikes).

    Returns (t_seconds, sensor1, sensor2), each of length ≤ max_points.
    """
    from lttb import downsample_lttb

    n = signal.shape[0]
    start_i, end_i = sample_range(start_s, end_s, n, fs_hz)
    window = signal[start_i:end_i]

    n_win = window.shape[0]
    s1_full = window[:, 0].astype(np.float64)
    s2_full = window[:, 1].astype(np.float64)
    t_full = (start_i + np.arange(n_win, dtype=np.float64)) / fs_hz

    if n_win <= max_points:
        return t_full, s1_full, s2_full

    if method == "stride":
        stride = max(1, n_win // max_points)
        idx = np.arange(0, n_win, stride)
        return t_full[idx], s1_full[idx], s2_full[idx]

    if method == "lttb":
        return downsample_lttb(t_full, s1_full, s2_full, max_points=max_points, guide=s2_full)

    raise ValueError(f"Unknown downsample method: {method!r}")
