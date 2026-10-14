"""Run the whole analysis on recordings and print one line per file.

    python tools/run_pipeline.py FILE_OR_FOLDER [...] [--out DIR] [--no-cache] [--workers 4]

With --out, writes one CSV of pieces per recording (never put it in the repository).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import pipeline  # noqa: E402


def csv_files(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        out += sorted(p.rglob("*.csv")) if p.is_dir() else [p]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    for f in csv_files(args.paths):
        r = pipeline.analyse(f, cache_root=None if args.no_cache else pipeline.DEFAULT_CACHE,
                             workers=args.workers)
        pc, c = r["pieces"], r["clean"]
        counts = pc["label"].value_counts().to_dict() if len(pc) else {}
        minutes = len(c["adc2"]) / c["fs"] / 60
        steps = "  ".join(f"{k} {v:.1f} s" for k, v in r["timings"].items())
        print(f"{f.name[:50]:<52}{minutes:7.1f} min  pieces {len(pc):4d}  {counts}  "
              f"sigma {c['sigma_adc2']:.2f}  |  {steps}",
              flush=True)
        if args.out:
            pc.to_csv(args.out / f"{f.stem}.pieces.csv", index=False)


if __name__ == "__main__":
    main()
