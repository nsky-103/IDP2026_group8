"""Measure how fast core.fast_io handles a recording on this computer.

    python tools/benchmark_io.py PATH_TO_CSV [--workers 4] [--out DIR]

Prints the time of each step and the peak memory, then checks that the converted
.npy matches the CSV at a few random places. The .npy goes to --out (default: a
temporary folder), never next to the measurement data.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import fast_io  # noqa: E402


def peak_memory_mb() -> tuple[float, float]:
    """(this process, largest child process) in MB. Not available on Windows."""
    try:
        import resource
    except ImportError:
        return float("nan"), float("nan")
    scale = 1 if sys.platform == "darwin" else 1024  # macOS reports bytes, Linux kilobytes
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 1e6
    child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale / 1e6
    return own, child


def step(name: str, fn):
    t0 = time.perf_counter()
    result = fn()
    print(f"  {name:<36}{time.perf_counter() - t0:8.2f} s")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--workers", type=int, default=fast_io.default_workers())
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--skip-npy", action="store_true", help="only build the 1 kHz summary")
    args = ap.parse_args()

    out_dir = args.out or Path(tempfile.mkdtemp(prefix="fast_io_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    npy = out_dir / (args.csv.stem + ".npy")
    print(f"{args.csv.name}: {args.csv.stat().st_size / 1e9:.2f} GB, {args.workers} workers\n")

    pv = step("preview without index", lambda: fast_io.preview(args.csv))
    idx = step("build index", lambda: fast_io.build_index(args.csv, workers=args.workers))
    hours = idx.n_rows / fast_io.DEFAULT_FS_HZ / 3600
    print(f"    {idx.n_rows:,} rows = {hours:.2f} h at 50 kHz, {len(pv['t'])} preview points")
    step("preview with index", lambda: fast_io.preview(args.csv, idx))
    mid = idx.n_rows / fast_io.DEFAULT_FS_HZ / 2
    for seconds in (10, 60):
        step(f"read {seconds} s window", lambda s=seconds: fast_io.read_window(idx, mid, mid + s))
    summ = step("1 kHz summary (mean/min/max)", lambda: fast_io.summarize(idx, workers=args.workers))
    if not args.skip_npy:
        step("full-rate .npy (optional)", lambda: fast_io.convert_to_npy(args.csv, idx, npy, workers=args.workers))

    own, child = peak_memory_mb()
    print(f"\n  peak memory: main {own:.0f} MB, largest worker {child:.0f} MB "
          f"(~{own + args.workers * child:.0f} MB in total)")

    rng = np.random.default_rng(0)
    ok = True
    for g in rng.integers(0, len(summ["mean"]) - 100, 5):
        raw = fast_io.read_rows(idx, int(g) * 50, int(g) * 50 + 5000).reshape(-1, 50, 2)
        ok &= np.allclose(summ["mean"][g: g + 100], raw.mean(1), atol=1e-3)
        ok &= np.array_equal(summ["min"][g: g + 100], raw.min(1)) and np.array_equal(summ["max"][g: g + 100], raw.max(1))
    print(f"  summary vs the CSV at 5 random places: {'OK' if ok else 'MISMATCH'}")
    if not args.skip_npy:
        arr = np.load(npy, mmap_mode="r")
        ok = arr.shape == (idx.n_rows, 2)
        for r in rng.integers(0, idx.n_rows - 1000, 5):
            ok &= np.array_equal(arr[r: r + 1000], fast_io.read_rows(idx, int(r), int(r) + 1000))
        print(f"  .npy vs the CSV at 5 random places: {'OK' if ok else 'MISMATCH'}  ({npy})")


if __name__ == "__main__":
    main()
