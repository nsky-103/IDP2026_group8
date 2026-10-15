"""Benchmark: how fast does this computer open and analyse a large recording?

    python tools/benchmark_io.py PATH_TO_CSV [--workers 4] [--all DATA_FOLDER] [--json out.json]

Run it from the repository root, with the viewer's environment (visualizer/.venv). It prints:

  1. This computer      OS, CPU cores, RAM, which speed-ups (numba, pyarrow) are installed
  2. First open         the viewer's steps, measured FIRST so the file is read from disk
                        (run `sudo purge` on macOS just before, or restart, for a true first open)
  3. File in memory     the same steps again (the OS now keeps the file in memory) + zoomed windows
  4. Second open        from the app's cache (~/.cache/idp2026 is not touched: a temp folder is used)
  5. Old methods        first viewer (csv module, one process) and a pandas chunked reader
                        (as in udgithubit/IDP scripts/demo_preprocessing_standalone.py);
                        timed on the first SAMPLE_MB of the file, scaled to the whole file
  6. Summary            pandas (until last week) vs now; peak memory; summary vs the CSV
  7. All recordings     (with --all) pieces and labels for every recording under DATA_FOLDER

Layout after udgithubit/IDP scripts/demo_preprocessing_standalone.py: one file, settings at
the top, one line per step.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import detection, fast_io, pipeline, preprocessing  # noqa: E402

# ---- settings (edit here) -----------------------------------------------------
SAMPLE_MB = 200           # old methods are timed on this much of the file, then scaled
WINDOWS_S = (10, 60)      # zoomed windows to read
RESULT = {}               # everything printed, for --json
# -------------------------------------------------------------------------------


def line(name: str, seconds: float, note: str = "") -> float:
    print(f"  {name:<44}{seconds:8.2f} s  {note}", flush=True)
    RESULT[name] = round(seconds, 3)
    return seconds


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


def ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, AttributeError, OSError):  # Windows
        return float("nan")


def peak_memory_mb() -> tuple[float, float]:
    """(this process, largest child process) in MB. Not available on Windows."""
    try:
        import resource
    except ImportError:
        return float("nan"), float("nan")
    scale = 1 if sys.platform == "darwin" else 1024  # macOS reports bytes, Linux kilobytes
    return (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 1e6,
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale / 1e6)


def installed(module: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(module) is not None


# ---- 3. old methods, on the first SAMPLE_MB --------------------------------------

def sample_bytes(path: Path, mb: int) -> bytes:
    with open(path, "rb") as f:
        buf = f.read(mb << 20)
    return buf[: buf.rfind(b"\n") + 1]


def old_viewer(buf: bytes, chunk_rows: int = 500_000) -> int:
    """First viewer (visualizer/signal_io.convert_csv_to_npy before 10-14): csv module, row by row,
    500 000 rows at a time into an int16 array (writing the .npy itself is left out)."""
    reader = csv.reader(io.StringIO(buf.decode()))
    next(reader, None)
    rows, n = [], 0
    for r in reader:
        rows.append([int(r[0]), int(r[1])])
        if len(rows) >= chunk_rows:
            n += len(np.asarray(rows, dtype=np.int16))
            rows.clear()
    return n + len(np.asarray(rows, dtype=np.int16))


def pandas_chunks(buf: bytes, block: int = 50) -> np.ndarray:
    """pandas read_csv in chunks + mean of 50, one process (udgithubit/IDP standalone demo)."""
    out, carry = [], None
    for ch in pd.read_csv(io.BytesIO(buf), usecols=["adc1", "adc2"], dtype=np.float32, chunksize=5_000_000):
        if carry is not None:
            ch = pd.concat([carry, ch], ignore_index=True)
        n = len(ch) // block * block
        carry = ch.iloc[n:]
        out.append(ch.iloc[:n].to_numpy().reshape(-1, block, 2).mean(1))
    return np.concatenate(out)


# ---- 7. all recordings --------------------------------------------------------------

def find_recordings(folder: Path) -> list[Path]:
    """Every file under folder that starts with an 'adc1,adc2' header (with or without .csv),
    each recording once: copies are recognised by content (the same key as the cache)."""
    found, seen = [], set()
    for f in sorted(p for p in folder.rglob("*") if p.is_file()):
        try:
            with open(f, "rb") as h:
                if not h.read(16).lower().startswith(b"adc1"):
                    continue
        except OSError:
            continue
        key = pipeline._cache_dir(f, Path(tempfile.gettempdir()) / "no_such_dir").name.rsplit("_", 1)[-1]
        if key not in seen:
            seen.add(key)
            found.append(f)
    return found


def all_recordings(folder: Path, workers: int) -> pd.DataFrame:
    rows = []
    for f in find_recordings(folder):
        t, r = timed(lambda f=f: pipeline.analyse(f, workers=workers))
        pc, c = r["pieces"], r["clean"]
        lab = pc["label"].value_counts() if len(pc) else pd.Series(dtype=int)
        row = {"file": f.name[:48], "minutes": round(len(c["adc2"]) / c["fs"] / 60, 1), "pieces": len(pc),
               **{k: int(lab.get(k, 0)) for k in ("dyed", "solid_no_dye", "water", "undecided")},
               "seconds": round(t, 1)}
        rows.append(row)
        print(f"  {row['file']:<50}{row['minutes']:7.1f} min  pieces {row['pieces']:4d}  dyed {row['dyed']:4d}"
              f"  no dye {row['solid_no_dye']:4d}  water {row['water']:4d}  undecided {row['undecided']:4d}"
              f"  ({t:.1f} s)", flush=True)
    df = pd.DataFrame(rows)
    if len(df):
        tot = df[["pieces", "dyed", "solid_no_dye", "water", "undecided"]].sum()
        print(f"  {'TOTAL (' + str(len(df)) + ' recordings)':<57}  pieces {tot.pieces:4d}  dyed {tot.dyed:4d}"
              f"  no dye {tot.solid_no_dye:4d}  water {tot.water:4d}  undecided {tot.undecided:4d}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", type=Path)
    ap.add_argument("--workers", type=int, default=fast_io.default_workers())
    ap.add_argument("--all", type=Path, default=None, help="also judge every recording under this folder")
    ap.add_argument("--skip-old", action="store_true", help="do not time the old methods (section 5)")
    ap.add_argument("--json", type=Path, default=None, help="write all numbers to this file")
    args = ap.parse_args()
    warnings.simplefilter("ignore", FutureWarning)  # a scipy dtype notice from Part A, once per file
    path, w = args.csv, args.workers
    size = path.stat().st_size
    method = ("numba" if installed("numba") and size >= fast_io.NUMBA_MIN_BYTES
              else "pyarrow" if installed("pyarrow") else "pandas")

    print("1. This computer")
    info = {"os": f"{platform.system()} {platform.release()}", "python": platform.python_version(),
            "cpu_cores": os.cpu_count(), "ram_gb": round(ram_gb(), 1), "workers": w,
            "numba": installed("numba"), "pyarrow": installed("pyarrow"), "summary_method": method,
            "file": path.name, "file_gb": round(size / 1e9, 2)}
    for k, v in info.items():
        print(f"  {k:<15}{v}")
    RESULT["computer"] = info

    def open_steps(label: str):
        """The viewer's first-open steps, one line each. Returns (times, index, summary)."""
        t = {}
        t["preview"] = line(f"{label}: rough preview (no index)", timed(lambda: fast_io.preview(path))[0])
        t["index"], idx = timed(lambda: fast_io.build_index(path, workers=w))
        line(f"{label}: row index (reads the whole file)", t["index"], f"({size / 1e6 / t['index']:,.0f} MB/s)")
        t["summary"], summ = timed(lambda: fast_io.summarize(idx, workers=w))
        line(f"{label}: 1 kHz summary ({method})", t["summary"])
        t["part_a"], clean = timed(lambda: preprocessing.preprocess(summ["mean"], fs=summ["fs_hz"]))
        line(f"{label}: Part A (median + SG + ALS, sigma)", t["part_a"])
        t["part_b"], (pieces, _) = timed(lambda: detection.find_pieces(clean))
        line(f"{label}: Part B (pieces and labels)", t["part_b"], f"({len(pieces)} pieces)")
        t["total"] = t["index"] + t["summary"] + t["part_a"] + t["part_b"]
        line(f"{label}: total (index + summary + A + B)", t["total"])
        return t, idx, summ

    print("\n2. First open (measured first, so the file is read from disk if it is not in memory)")
    cold, _, _ = open_steps("first")

    print("\n3. Same again, file now in memory")
    warm, idx, summ = open_steps("in memory")
    if cold["index"] > 1.5 * warm["index"]:
        print(f"  -> section 2 read from disk ({size / 1e6 / cold['index']:,.0f} MB/s): it is a real first open")
    else:
        print("  -> section 2 was as fast as this: the file was already in memory. For a true first open,"
              " run `sudo purge` (macOS) or restart, then this script.")
    RESULT["first_open_from_disk"] = bool(cold["index"] > 1.5 * warm["index"])
    mid = idx.n_rows / fast_io.DEFAULT_FS_HZ / 2
    for s in WINDOWS_S:
        line(f"zoomed {s} s window (exact samples)", timed(lambda s=s: fast_io.read_window(idx, mid, mid + s))[0])

    print("\n4. Second open, from the app's cache (a temp folder; ~/.cache/idp2026 is not touched)")
    with tempfile.TemporaryDirectory() as d:
        pipeline.analyse(path, cache_root=Path(d), workers=w)          # fills the temp cache
        t2, r2 = timed(lambda: pipeline.analyse(path, cache_root=Path(d), workers=w))
        line("second open", t2, "  ".join(f"{k} {v:.2f}" for k, v in r2["timings"].items()))

    if not args.skip_old:
        print(f"\n5. Old methods (timed on the first {SAMPLE_MB} MB, file in memory, scaled to {size / 1e9:.1f} GB)")
        buf = sample_bytes(path, SAMPLE_MB)
        for name, fn in (("first viewer (csv module, 1 process)", old_viewer),
                         ("pandas chunks (1 process, IDP standalone)", pandas_chunks)):
            dt, _ = timed(lambda fn=fn: fn(buf))
            line(name, dt * size / len(buf), f"(~{len(buf) / 1e6 / dt:,.0f} MB/s, estimate)")

    print("\n6. Summary")
    os.environ["IDP_NO_PYARROW"] = "1"          # workers inherit it: parse with pandas
    keep = fast_io.NUMBA_MIN_BYTES
    fast_io.NUMBA_MIN_BYTES = 1 << 62           # and no numba
    t_pandas = timed(lambda: fast_io.summarize(idx, workers=w))[0]
    fast_io.NUMBA_MIN_BYTES = keep
    del os.environ["IDP_NO_PYARROW"]
    RESULT["1 kHz summary (pandas, until last week), in memory"] = round(t_pandas, 3)
    print(f"  1 kHz summary, file in memory: pandas (until last week) {t_pandas:.2f} s -> {method} {warm['summary']:.2f} s"
          f"  ({t_pandas / warm['summary']:.1f}x faster)")
    print(f"  first open: {cold['total']:.1f} s{' (from disk)' if RESULT['first_open_from_disk'] else ' (file was in memory)'}"
          f";  file in memory: {warm['total']:.1f} s;  second open (cache): {t2:.1f} s")
    own, child = peak_memory_mb()
    print(f"  peak memory: main {own:.0f} MB, largest worker {child:.0f} MB (~{own + w * child:.0f} MB in total;"
          " includes this benchmark's own copies)")
    RESULT["peak_memory_mb"] = {"main": own, "worker": child}
    rng = np.random.default_rng(0)
    ok = True
    for g in rng.integers(0, len(summ["mean"]) - 100, 5):
        raw = fast_io.read_rows(idx, int(g) * 50, int(g) * 50 + 5000).reshape(-1, 50, 2)
        ok &= np.allclose(summ["mean"][g: g + 100], raw.mean(1), atol=1e-3)
        ok &= np.array_equal(summ["min"][g: g + 100], raw.min(1)) and np.array_equal(summ["max"][g: g + 100], raw.max(1))
    print(f"  1 kHz summary vs the CSV at 5 random places: {'OK' if ok else 'MISMATCH'}")
    RESULT["summary_matches_csv"] = bool(ok)

    if args.all:
        print(f"\n7. All recordings under {args.all} (cached in {pipeline.DEFAULT_CACHE})")
        RESULT["recordings"] = all_recordings(args.all, w).to_dict("records")

    if args.json:
        args.json.write_text(json.dumps(RESULT, indent=1, default=float))
        print(f"\nsaved {args.json}")


if __name__ == "__main__":
    main()
