"""Fast access to large Marginum CSV recordings (two integer columns: adc1, adc2).

Four building blocks, meant to be used in this order so the user never waits:

    idx = build_index(csv)                      # ~1-2 s: byte offset <-> row number
    pv  = preview(csv, idx)                     # < 1 s: rough min/max over the whole file
    t, a1, a2 = read_window(csv, idx, 600, 610) # tens of ms: exact samples of one window
    summ = summarize(idx)                       # ~7 s for 7.6 GB with 4 workers, run in the background

``summarize`` gives the 1 kHz mean/min/max for the exact overview and for analysis.
Zoomed windows come straight from the CSV via ``read_window``, so a full-rate copy is not needed.

CSV blocks are parsed with pyarrow when it is installed (about 1.6x faster than pandas,
identical result), otherwise with pandas. With numba installed, ``summarize`` parses and
reduces each block in one compiled pass instead (about 2.8x faster per block than pyarrow,
identical result) for files over NUMBA_MIN_BYTES; the first run compiles it once and caches
it next to this file.

Worker processes need about 0.4 GB of RAM each. Callers that start the pool must be
importable modules (Windows uses "spawn"), and scripts must use ``if __name__ == "__main__":``.
"""

from __future__ import annotations

import importlib.util
import io
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DTYPE = np.int16
DEFAULT_FS_HZ = 50_000.0
WORKER_RAM_GB = 0.4


def default_workers(max_workers: int = 8) -> int:
    """One worker per CPU core, at most max_workers (4 on the 4-core target laptop)."""
    return max(1, min(max_workers, os.cpu_count() or 1))


@dataclass(frozen=True)
class CsvIndex:
    """Row number at the start of each block of the file."""

    path: Path
    byte_starts: np.ndarray  # first byte of each block (always the start of a line)
    byte_ends: np.ndarray
    row_starts: np.ndarray   # data row number of the first line of each block
    n_rows: int

    def block_of_row(self, row: int) -> int:
        return int(np.searchsorted(self.row_starts, row, side="right") - 1)


# ---------------------------------------------------------------- index

def _header_bytes(path: Path) -> int:
    with open(path, "rb") as f:
        first = f.readline()
    if not first[:1].isalpha():
        raise ValueError(f"Expected a header line like 'adc1,adc2' in {path}, got {first[:40]!r}")
    return len(first)


def _block_bounds(path: Path, n_blocks: int) -> list[tuple[int, int]]:
    size = path.stat().st_size
    start = _header_bytes(path)
    edges = [start]
    with open(path, "rb") as f:
        for k in range(1, n_blocks):
            pos = start + (size - start) * k // n_blocks
            if pos <= edges[-1]:
                continue
            f.seek(pos)
            f.readline()
            if f.tell() < size and f.tell() > edges[-1]:
                edges.append(f.tell())
    edges.append(size)
    return list(zip(edges[:-1], edges[1:]))


def _count_lines(args: tuple[str, int, int]) -> int:
    path, a, b = args
    n = 0
    with open(path, "rb") as f:
        f.seek(a)
        left = b - a
        while left > 0:
            buf = f.read(min(left, 16 << 20))
            n += buf.count(b"\n")
            left -= len(buf)
    return n


def build_index(csv_path: str | Path, n_blocks: int = 1024, workers: int | None = None) -> CsvIndex:
    """Count lines per block in parallel. Reads the file once, parses nothing."""
    path = Path(csv_path)
    bounds = _block_bounds(path, n_blocks)
    with ProcessPoolExecutor(workers or default_workers()) as ex:
        counts = list(ex.map(_count_lines, [(str(path), a, b) for a, b in bounds], chunksize=16))
    with open(path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) != b"\n":
            counts[-1] += 1  # last line without a trailing newline
    counts = np.asarray(counts, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return CsvIndex(path, np.array([a for a, _ in bounds], np.int64),
                    np.array([b for _, b in bounds], np.int64), starts, int(counts.sum()))


# ---------------------------------------------------------------- reading

try:
    import pyarrow as _pa
    import pyarrow.csv as _pacsv
except ImportError:  # pandas fallback, about 1.6x slower
    _pa = None


def _parse(buf: bytes) -> np.ndarray:
    if not buf.strip():
        return np.empty((0, 2), DTYPE)
    if _pa is None:
        return pd.read_csv(io.BytesIO(buf), header=None, dtype=DTYPE, engine="c").to_numpy()
    table = _pacsv.read_csv(
        io.BytesIO(buf),
        read_options=_pacsv.ReadOptions(autogenerate_column_names=True, use_threads=False),
        convert_options=_pacsv.ConvertOptions(column_types={"f0": _pa.int16(), "f1": _pa.int16()}),
    )
    return np.column_stack([table.column(0).to_numpy(), table.column(1).to_numpy()])


def _read_bytes(path: Path, a: int, b: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(a)
        return f.read(b - a)


def read_rows(index: CsvIndex, row_start: int, row_end: int) -> np.ndarray:
    """Exact rows [row_start, row_end) as an (n, 2) int16 array, without converting the file."""
    row_start = max(0, row_start)
    row_end = min(index.n_rows, row_end)
    if row_end <= row_start:
        return np.empty((0, 2), DTYPE)
    b0, b1 = index.block_of_row(row_start), index.block_of_row(row_end - 1)
    data = _parse(_read_bytes(index.path, int(index.byte_starts[b0]), int(index.byte_ends[b1])))
    off = row_start - int(index.row_starts[b0])
    return data[off: off + (row_end - row_start)]


def read_window(index: CsvIndex, start_s: float, end_s: float,
                fs_hz: float = DEFAULT_FS_HZ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact samples between start_s and end_s. Returns (t_seconds, adc1, adc2)."""
    r0, r1 = int(start_s * fs_hz), int(end_s * fs_hz)
    data = read_rows(index, r0, r1)
    t = (max(0, r0) + np.arange(len(data))) / fs_hz
    return t, data[:, 0], data[:, 1]


def _row_at_byte(pos: int, index: CsvIndex | None, header: int, bytes_per_row: float) -> float:
    """Approximate row number at a byte position (exact at block starts when an index exists)."""
    if index is None:
        return (pos - header) / bytes_per_row
    b = int(np.searchsorted(index.byte_starts, pos, side="right") - 1)
    a, e = int(index.byte_starts[b]), int(index.byte_ends[b])
    rows_in_block = (index.row_starts[b + 1] if b + 1 < len(index.row_starts) else index.n_rows) - index.row_starts[b]
    return float(index.row_starts[b]) + (pos - a) / max(1, e - a) * rows_in_block


def preview(csv_path: str | Path, index: CsvIndex | None = None, n_points: int = 2000,
            block_bytes: int = 8192, fs_hz: float = DEFAULT_FS_HZ) -> dict[str, np.ndarray]:
    """Rough overview from n_points short reads spread over the file.

    Works before the index exists (times then come from the average line length).
    Only about n_points * block_bytes bytes are read, so short peaks between the
    read positions are missed. Show it as an approximation until the full
    conversion is done.
    """
    path = Path(csv_path)
    header = _header_bytes(path)
    end = path.stat().st_size
    with open(path, "rb") as f:
        f.seek(header)
        sample = f.read(1 << 16)
    bytes_per_row = len(sample) / max(1, sample.count(b"\n"))

    positions = np.linspace(header, max(header, end - block_bytes), n_points).astype(np.int64)
    t, mins, maxs = [], [], []
    with open(path, "rb") as f:
        for pos in positions:
            f.seek(pos)
            buf = f.read(block_bytes)
            first = 0 if pos == header else buf.find(b"\n") + 1
            last = buf.rfind(b"\n") + 1
            chunk = _parse(buf[first:last])
            if not len(chunk):
                continue
            t.append(_row_at_byte(int(pos + first), index, header, bytes_per_row) / fs_hz)
            mins.append(chunk.min(axis=0))
            maxs.append(chunk.max(axis=0))
    mins, maxs = np.array(mins), np.array(maxs)
    return {"t": np.array(t), "adc1_min": mins[:, 0], "adc1_max": maxs[:, 0],
            "adc2_min": mins[:, 1], "adc2_max": maxs[:, 1]}


# ---------------------------------------------------------------- 1 kHz summary

# numba is optional and imported only by the worker processes that use it: importing it costs
# 0.3-0.5 s per process, which only pays off for large files (NUMBA_MIN_BYTES).
HAVE_NUMBA = importlib.util.find_spec("numba") is not None
NUMBA_MIN_BYTES = 500_000_000
_numba_kernel = None


def _get_numba_kernel():
    global _numba_kernel
    if _numba_kernel is None:
        import numba

        _numba_kernel = numba.njit(cache=True)(_parse_summarize_py)
    return _numba_kernel


def _parse_summarize_py(buf, row_start, factor, g0, sums, mins, maxs, counts):
    """One pass over 'int,int' lines: parse and add each row to its group of `factor` rows.

    Same result as parsing first and reducing (checked on the whole 7.6 GB recording);
    '\r' is skipped, a minus sign is honoured. Returns the number of rows parsed.
    """
    row = row_start
    v = 0
    a = 0
    neg = False
    col = 0
    have = False
    for i in range(buf.shape[0] + 1):
        c = buf[i] if i < buf.shape[0] else 10  # a missing final newline ends the last row
        if 48 <= c <= 57:
            v = v * 10 + (c - 48)
            have = True
        elif c == 45:  # '-'
            neg = True
        elif c == 44:  # ','
            a = -v if neg else v
            v = 0
            neg = False
            have = False
            col = 1
        elif c == 10:  # '\n'
            if col == 1 and have:
                b = -v if neg else v
                g = row // factor - g0
                sums[g, 0] += a
                sums[g, 1] += b
                if a < mins[g, 0]:
                    mins[g, 0] = a
                if b < mins[g, 1]:
                    mins[g, 1] = b
                if a > maxs[g, 0]:
                    maxs[g, 0] = a
                if b > maxs[g, 1]:
                    maxs[g, 1] = b
                counts[g] += 1
                row += 1
            v = 0
            neg = False
            have = False
            col = 0
    return row - row_start


def _summarize_block_nb(buf: np.ndarray, row_start: int, factor: int) -> tuple:
    g0 = row_start // factor
    n_groups = len(buf) // (4 * factor) + 2   # a row is at least 4 bytes ("0,0\n")
    sums = np.zeros((n_groups, 2), np.int32)
    mins = np.full((n_groups, 2), np.iinfo(DTYPE).max, np.int32)
    maxs = np.full((n_groups, 2), np.iinfo(DTYPE).min, np.int32)
    counts = np.zeros(n_groups, np.int64)
    n_rows = _get_numba_kernel()(buf, row_start, factor, g0, sums, mins, maxs, counts)
    used = (row_start + n_rows - 1) // factor - g0 + 1
    return g0, sums[:used], counts[:used], mins[:used], maxs[:used]


def _summarize_block(args: tuple) -> tuple:
    csv_path, a, b, row_start, factor, use_numba = args
    if use_numba:
        return _summarize_block_nb(np.frombuffer(_read_bytes(Path(csv_path), a, b), np.uint8), row_start, factor)
    data = _parse(_read_bytes(Path(csv_path), a, b)).astype(np.int32)
    groups = (row_start + np.arange(len(data))) // factor
    cuts = np.flatnonzero(np.diff(groups)) + 1
    starts = np.concatenate([[0], cuts])
    return (int(groups[0]), np.add.reduceat(data, starts, axis=0), np.diff(np.append(starts, len(data))),
            np.minimum.reduceat(data, starts, axis=0), np.maximum.reduceat(data, starts, axis=0))


def summarize(index: CsvIndex, factor: int = 50, workers: int | None = None,
              progress_callback=None, partial_callback=None,
              partial_interval_s: float = 0.5) -> dict[str, np.ndarray]:
    """Mean, min and max of every `factor` samples (50 -> 1 ms at 50 kHz), both channels.

    Enough for the exact whole-recording overview and for analysis at 1 kHz;
    a 4.7 h recording gives about 270 MB instead of a 3.4 GB full-rate copy.
    Full-rate samples for a zoomed window come from read_window().

    partial_callback(summary) is called about every partial_interval_s while the blocks are
    processed (in file order), so a viewer can draw the finished part early. In that partial
    summary, "mean" is NaN where no data has arrived yet; use it to mask "min" / "max" too.
    The dict is the one being filled, so copy it if it must not change.
    """
    n_groups = -(-index.n_rows // factor)
    total = np.zeros((n_groups, 2), np.int32)   # 50 samples x 4095 fits easily
    count = np.zeros(n_groups, np.int16)
    lo = np.full((n_groups, 2), np.iinfo(DTYPE).max, DTYPE)
    hi = np.full((n_groups, 2), np.iinfo(DTYPE).min, DTYPE)
    mean = np.full((n_groups, 2), np.nan, np.float32)
    result = {"mean": mean, "min": lo, "max": hi, "fs_hz": DEFAULT_FS_HZ / factor}
    last = time.perf_counter()
    use_numba = HAVE_NUMBA and index.path.stat().st_size >= NUMBA_MIN_BYTES
    jobs = [(str(index.path), int(a), int(b), int(r), factor, use_numba)
            for a, b, r in zip(index.byte_starts, index.byte_ends, index.row_starts)]
    with ProcessPoolExecutor(workers or default_workers()) as ex:
        for i, (g0, s, c, mn, mx) in enumerate(ex.map(_summarize_block, jobs, chunksize=4)):
            sl = slice(g0, g0 + len(c))
            total[sl] += s
            count[sl] += c
            lo[sl] = np.minimum(lo[sl], mn)
            hi[sl] = np.maximum(hi[sl], mx)
            mean[sl] = total[sl] / count[sl, None].astype(np.float32)
            if progress_callback:
                progress_callback((i + 1) / len(jobs))
            if partial_callback and time.perf_counter() - last >= partial_interval_s:
                partial_callback(result)
                last = time.perf_counter()
    return result
