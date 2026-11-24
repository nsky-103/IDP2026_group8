"""Benchmark: the current Part B methods against simpler ones, on the same recordings.

    python tools/benchmark_methods.py DATA_FOLDER [--workers 4] [--json out.json]

Run it from the repository root, with the viewer's environment (visualizer/.venv). Every file
under DATA_FOLDER that starts with an 'adc1,adc2' header is used. It prints:

  1. Noise level sigma  now: idle blocks (no flow) are dropped before taking the quietest 10%
                        simple: all blocks are used
                        -> sigma of adc2 and the number of events per recording
  2. Two-channel check  now: the adc1 window is centred on the recording's channel lag, and the
                        adc1 peak must be at least AMP_RATIO x the adc2 peak
                        simple: window centred on 0, any adc1 peak
                        -> chance rate, confirmation rate (dyed events also confirmed), and
                           whether the check is used (chance rate <= MAX_CHANCE)
  3. Big pieces         labels per section for both two-channel checks (if the Big pieces
                        recording is in DATA_FOLDER)

Only the compared step is changed; everything else is the current pipeline. core/ is not
modified: the simple sigma uses quiet_sigma(idle_cut=0), the simple check temporarily sets
detection.AMP_RATIO = 0 and the channel lag to 0 inside this script.
Recordings are opened through the app's cache (~/.cache/idp2026), so a second run is faster.
For the speed of reading files, see tools/benchmark_io.py.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_io import find_recordings  # noqa: E402
from core import detection, pipeline, preprocessing  # noqa: E402

# ---- settings (edit here) -----------------------------------------------------
# Big pieces reference: section times in s (15 pieces each, then water), read from the recording
BIG_PIECES_NAME = "Big pieces"
BIG_SECTIONS = {"high": (98, 139), "middle": (27, 61), "low": (161, 203), "zero": (223, 259),
                "water": (281, 328)}
RESULT = {}               # everything printed, for --json
# -------------------------------------------------------------------------------


@contextlib.contextmanager
def simple_two_channel():
    """Two-channel check without the lag correction and without the height condition."""
    amp, lag = detection.AMP_RATIO, detection.channel_lag
    detection.AMP_RATIO = 0.0
    detection.channel_lag = lambda t_top, t1: 0.0
    try:
        yield
    finally:
        detection.AMP_RATIO, detection.channel_lag = amp, lag


def simple_sigma(clean: dict) -> dict:
    """The same cleaned signal, sigma from all 200 ms blocks (idle blocks kept)."""
    fs = clean["fs"]
    return dict(clean,
                sigma_adc1=preprocessing.quiet_sigma(clean["adc1"].astype(float), fs, idle_cut=0),
                sigma_adc2=preprocessing.quiet_sigma(clean["adc2"].astype(float), fs, idle_cut=0))


def check_numbers(pieces: pd.DataFrame, peaks: dict) -> dict:
    tc = peaks["two_channel"]
    dyed = pieces[pieces["label"] == "dyed"]
    return {"used": bool(tc["used"]), "chance": tc["chance"],
            "confirmed_dyed": int(dyed["confirmed"].sum()), "dyed": len(dyed)}


def big_pieces_table(pieces: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for name, (t0, t1) in BIG_SECTIONS.items():
        p = pieces[(pieces["start_s"] >= t0 - 1) & (pieces["start_s"] < t1 + 1)]
        rows[name] = {"events": len(p), **p["label"].value_counts().to_dict()}
    cols = ["events", "dyed", "solid_no_dye", "water", "undecided"]
    return pd.DataFrame(rows).T.reindex(columns=cols).fillna(0).astype(int)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    rows, big = [], None
    print("Each column: current method / simpler method. events: sigma changed only; "
          "chance and confirmed: two-channel check changed only.")
    print("recording                                   sigma now/simple   events now/simple   "
          "chance now/simple   confirmed now/simple")
    for f in find_recordings(args.folder):
        r = pipeline.analyse(f, workers=args.workers)
        clean, now = r["clean"], r["pieces"]
        simple_s, _ = detection.find_pieces(simple_sigma(clean))
        row = {"file": f.name, "sigma_now": clean["sigma_adc2"],
               "sigma_simple": preprocessing.quiet_sigma(clean["adc2"].astype(float), clean["fs"], idle_cut=0),
               "events_now": len(now), "events_simple": len(simple_s)}
        if clean["adc1_usable"]:
            _, peaks_now = detection.find_pieces(clean)
            with simple_two_channel():
                simple_c, peaks_simple = detection.find_pieces(clean)
            for tag, p, pk in (("now", now, peaks_now), ("simple", simple_c, peaks_simple)):
                for k, v in check_numbers(p, pk).items():
                    row[f"{k}_{tag}"] = v
            if BIG_PIECES_NAME in f.name:
                big = {"now": big_pieces_table(now), "simple": big_pieces_table(simple_c)}
        rows.append(row)
        chance = (f"{row['chance_now']:6.2f} / {row['chance_simple']:4.2f}" if "chance_now" in row
                  else "   adc1 not used ")
        conf = (f"{row['confirmed_dyed_now'] / max(1, row['dyed_now']):6.2f} / "
                f"{row['confirmed_dyed_simple'] / max(1, row['dyed_simple']):4.2f}" if "chance_now" in row else "")
        print(f"{f.name[:42]:<44}{row['sigma_now']:6.2f} / {row['sigma_simple']:5.2f}   "
              f"{row['events_now']:6d} / {row['events_simple']:5d}     {chance}     {conf}", flush=True)

    d = pd.DataFrame(rows)
    print("\n1. Noise level sigma (adc2)\n"
          "   sigma with idle blocks dropped (now) or kept (simple), and how the number of events changes")
    changed = d[d["events_now"] != d["events_simple"]]
    print(f"  recordings: {len(d)}, events now {d['events_now'].sum()}, simple {d['events_simple'].sum()}; "
          f"different in {len(changed)} recordings")
    biggest = (changed["events_now"] - changed["events_simple"]).abs().sort_values(ascending=False).index[:5]
    for _, r in changed.loc[biggest].iterrows():
        print(f"    {r['file'][:50]:<52} sigma {r['sigma_now']:.2f} -> {r['sigma_simple']:.2f}   "
              f"events {r['events_now']} (now) vs {r['events_simple']} (simple)")
    RESULT["sigma"] = d[["file", "sigma_now", "sigma_simple", "events_now", "events_simple"]].to_dict("records")

    c = d.dropna(subset=["chance_now"]) if "chance_now" in d else d.iloc[0:0]
    if len(c):
        print("\n2. Two-channel check (recordings where adc1 responds)\n"
              "   how often the check passes by chance, and how many dyed events it confirms")
        for tag in ("now", "simple"):
            conf = c[f"confirmed_dyed_{tag}"].sum() / max(1, c[f"dyed_{tag}"].sum())
            print(f"  {tag:<7} chance rate median {c[f'chance_{tag}'].median():.2f}, max {c[f'chance_{tag}'].max():.2f}; "
                  f"check used in {int(c[f'used_{tag}'].sum())} of {len(c)}; "
                  f"dyed events also confirmed {conf:.0%}")
        RESULT["two_channel"] = c.filter(regex="file|chance|confirmed|dyed|used").to_dict("records")

    if big is not None:
        print("\n3. Big pieces, labels per section\n"
              "   labels with the current (now) and the simple two-channel check, against the known pieces")
        for tag, t in big.items():
            print(f"  {tag}:\n" + t.to_string().replace("\n", "\n    ").join(["    ", ""]))
            RESULT[f"big_pieces_{tag}"] = t.to_dict()

    if args.json:
        args.json.write_text(json.dumps(RESULT, indent=1, default=float))
        print(f"\nwritten: {args.json}")


if __name__ == "__main__":
    main()
