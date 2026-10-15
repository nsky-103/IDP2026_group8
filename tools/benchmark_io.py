"""Benchmark: how fast does this computer open and analyse a large recording?

    python tools/benchmark_io.py PATH_TO_CSV [--workers 4] [--all DATA_FOLDER] [--json out.json]

Run it from the repository root, with the viewer's environment (visualizer/.venv). It prints:

  1. This computer      OS, CPU cores, RAM, which speed-ups (numba, pyarrow) are installed
  2. Disk               time to read the whole file once (slow disks / cloud folders show here)
  3. Old methods        first viewer (csv module, one process) and a pandas chunked reader
                        (as in udgithubit/IDP scripts/demo_preprocessing_standalone.py);
                        timed on the first SAMPLE_MB of the file, scaled to the whole file
  4. Now                preview, row index, 1 kHz summary (numba > pyarrow > pandas, whichever is
                        installed), zoomed windows, Part A (clean), Part B (pieces)
  5. Opening            first open and second open (from the cache), as the viewer does it
  6. Memory and check   peak memory; the 1 kHz summary vs the CSV at random places
  7. Summary            pandas (until last week) vs the fastest method now
  8. All recordings     (with --all) pieces and labels for every CSV under DATA_FOLDER

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
READ_CHUNK = 64 << 20     # disk read test
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


# ---- 8. all recordings --------------------------------------------------------------

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
    ap.add_argument("--all", type=Path, default=None, help="also judge every CSV under this folder")
    ap.add_argument("--skip-old", action="store_true", help="do not time the old methods (section 3)")
    ap.add_argument("--json", type=Path, default=None, help="write all numbers to this file")
    args = ap.parse_args()
    warnings.simplefilter("ignore", FutureWarning)  # a scipy dtype notice from Part A, once per file
    path, w = args.csv, args.workers
    size = path.stat().st_size

    print("1. This computer")
    info = {"os": f"{platform.system()} {platform.release()}", "python": platform.python_version(),
            "cpu_cores": os.cpu_count(), "ram_gb": round(ram_gb(), 1), "workers": w,
            "numba": installed("numba"), "pyarrow": installed("pyarrow"),
            "file": path.name, "file_gb": round(size / 1e9, 2)}
    for k, v in info.items():
        print(f"  {k:<12}{v}")
    RESULT["computer"] = info

    print("\n2. Disk (whole file once; slow here = slow disk, cloud folder or antivirus)")

    def read_all():
        with open(path, "rb") as f:
            while f.read(READ_CHUNK):
                pass
    dt, _ = timed(read_all)
    line("read the whole file", dt, f"({size / 1e6 / dt:,.0f} MB/s)")

    if not args.skip_old:
        print(f"\n3. Old methods (timed on the first {SAMPLE_MB} MB, scaled to {size / 1e9:.1f} GB)")
        buf = sample_bytes(path, SAMPLE_MB)
        for name, fn in (("first viewer (csv module, 1 process)", old_viewer),
                         ("pandas chunks (1 process, IDP standalone)", pandas_chunks)):
            dt, _ = timed(lambda fn=fn: fn(buf))
            line(name, dt * size / len(buf), f"(~{len(buf) / 1e6 / dt:,.0f} MB/s, estimate)")

    print(f"\n4. Now ({w} workers)")
    line("rough preview (no index)", timed(lambda: fast_io.preview(path))[0])
    t_ix, idx = timed(lambda: fast_io.build_index(path, workers=w))
    line("row index", t_ix)
    # What summarize uses here: numba for files over NUMBA_MIN_BYTES when installed, else pyarrow, else pandas
    method = ("numba" if installed("numba") and size >= fast_io.NUMBA_MIN_BYTES
              else "pyarrow" if installed("pyarrow") else "pandas")
    fast_io.summarize(idx, workers=w)  # first run on a new computer compiles numba once
    t_now, summ = timed(lambda: fast_io.summarize(idx, workers=w))
    line(f"1 kHz summary ({method})", t_now)
    mid = idx.n_rows / fast_io.DEFAULT_FS_HZ / 2
    for s in WINDOWS_S:
        line(f"zoomed {s} s window (exact samples)", timed(lambda s=s: fast_io.read_window(idx, mid, mid + s))[0])
    t_a, clean = timed(lambda: preprocessing.preprocess(summ["mean"], fs=summ["fs_hz"]))
    line("Part A: clean (median + SG + ALS, sigma)", t_a)
    t_b, (pieces, _) = timed(lambda: detection.find_pieces(clean))
    line("Part B: pieces and labels", t_b, f"({len(pieces)} pieces)")

    print("\n5. Opening, as the viewer does it (empty cache, then again)")
    with tempfile.TemporaryDirectory() as d:
        t1, r1 = timed(lambda: pipeline.analyse(path, cache_root=Path(d), workers=w))
        line("first open", t1, "  ".join(f"{k} {v:.1f}" for k, v in r1["timings"].items()))
        t2, r2 = timed(lambda: pipeline.analyse(path, cache_root=Path(d), workers=w))
        line("second open (cache)", t2, "  ".join(f"{k} {v:.1f}" for k, v in r2["timings"].items()))

    print("\n6. Memory and check")
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

    print("\n7. Summary: pandas (until last week) vs now")
    os.environ["IDP_NO_PYARROW"] = "1"          # workers inherit it: parse with pandas
    keep = fast_io.NUMBA_MIN_BYTES
    fast_io.NUMBA_MIN_BYTES = 1 << 62           # and no numba
    t_pandas = timed(lambda: fast_io.summarize(idx, workers=w))[0]
    fast_io.NUMBA_MIN_BYTES = keep
    del os.environ["IDP_NO_PYARROW"]
    RESULT["1 kHz summary (pandas, until last week)"] = round(t_pandas, 3)
    print(f"  1 kHz summary: pandas {t_pandas:.2f} s -> {method} {t_now:.2f} s  ({t_pandas / t_now:.1f}x faster)")
    print(f"  first open:    with pandas ~{t1 - t_now + t_pandas:.1f} s -> now {t1:.1f} s;  second open {t2:.1f} s")
    RESULT["speedup_summary"] = t_pandas / t_now

    if args.all:
        print(f"\n8. All recordings under {args.all} (cached in {pipeline.DEFAULT_CACHE})")
        RESULT["recordings"] = all_recordings(args.all, w).to_dict("records")

    if args.json:
        args.json.write_text(json.dumps(RESULT, indent=1, default=float))
        print(f"\nsaved {args.json}")


if __name__ == "__main__":
    main()
