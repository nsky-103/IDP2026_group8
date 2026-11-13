# core

Code shared by the visualizer and scripts. No UI code here.

## fast_io.py — open large recordings without making the user wait

| Step | Function | 7.6 GB / 4.7 h, 4 workers |
|---|---|---|
| Rough overview (min/max at 2000 places) | `preview(csv)` | 0.2 s |
| Byte offset ↔ row number index | `build_index(csv)` | 3.7 s |
| Exact samples of a zoomed window | `read_window(idx, start_s, end_s)` | 10 s: 0.02 s, 60 s: 0.09 s |
| 1 kHz mean / min / max of the whole file | `summarize(idx)` | 4.6 s with numba (7.9 s pyarrow, 11 s pandas); 3.6 s with 8 workers |

Measured on a MacBook (8 cores) with the file already read once. Peak memory about 2.2 GB in total
(0.85 GB main process + about 0.35 GB per worker), well within a 16 GB laptop.
Run `python tools/benchmark_io.py PATH_TO_CSV` to measure on another computer.

With numba installed (`pip install numba`), `summarize` parses and reduces each block of files over 500 MB in one
compiled pass (about 2.8x faster per block than pyarrow, identical result; compiled once on the first run and
cached in `core/__pycache__`). Smaller files skip it: importing numba costs 0.3–0.5 s per worker.

CSV blocks are parsed with pyarrow if it is installed (`pip install pyarrow`, about 1.6x faster, identical
result), otherwise with pandas. `default_workers()` uses one worker per CPU core, at most 8.
`summarize(..., partial_callback=f)` calls `f(summary)` about every 0.5 s with the part finished so far
(in file order; `mean` is NaN where nothing has arrived yet), so a viewer can draw it early.

Suggested order in the app:

1. `preview` → draw the rough overview at once (mark it as approximate: short peaks between the read places are missed)
2. `build_index` → from now on, zooming reads exact samples with `read_window`
3. `summarize` in the background → replace the overview with the exact one; run analysis on the 1 kHz data

Checked against `pandas.read_csv` of the whole file: row count, parsed values, random windows,
first/last rows and times all match, for both `\r\n` and `\n` line endings.

Notes:
- Worker processes are started with `ProcessPoolExecutor`. On Windows, code that calls these functions
  must be importable and scripts need `if __name__ == "__main__":`.
- The sampling rate is assumed to be 50 kHz (`DEFAULT_FS_HZ`).
- Output files must not go into the repository (see the data policy in `visualizer/README.md`).

## pipeline.py — whole analysis of one recording

```python
from core import pipeline
r = pipeline.analyse("recording.csv")
r["pieces"]    # one row per piece: start_s, end_s, duration_s, n_peaks, max_height, area, confirmed, label
r["peaks"]     # peak times (s) of adc2 and adc1, for plotting
r["clean"]     # cleaned 1 kHz adc1 / adc2 and sigma per channel
```

| Step | Module | Origin |
|---|---|---|
| CSV -> row index + 1 kHz summary | `fast_io` | Sora |
| Part A: median + Savitzky-Golay + ALS, sigma (quietest 10% of 200 ms blocks, idle blocks dropped) | `preprocessing` | Unmona's `idp/method.py`; idle-block drop by Sora (#3) |
| Part B: peaks, pieces, labels, features | `detection` | Sora's rules, unchanged |

The cache (row index + 1 kHz summary + Part A output, about 270 MB for 4.7 h) goes to `~/.cache/idp2026`
(change with the environment variable `IDP_CACHE_DIR`). The Part A file name includes a hash of
`preprocessing.PARAMS`, so changed Part A parameters run Part A again. Detection (Part B) always runs again.

### Part B rules (`detection.py`)

- Peaks: prominence > 10 x sigma on the cleaned signal; peaks in the first and last 0.5 s are dropped.
- Pieces: peaks less than 1 s apart are one piece (one aspirated tissue piece).
- Label from the largest peak of the piece: higher than the no-dye level (68.9 counts, top 0.1% of
  water-only peaks) -> `dyed`; otherwise confirmed by adc1 -> `solid_no_dye`; otherwise `water`.
- Two-channel check: an adc1 peak within ±20 ms of the top adc2 peak shifted by the recording's channel
  lag (median over the dyed pieces), at least 5% as high. `peaks["two_channel"]` gives the lag and the
  chance rate (adc1 shifted by 2–20 s). Chance rate above 20%: the check is not used, dim pieces `undecided`.
- If adc1 hardly responds (adc1 / adc2 at the largest peaks below 0.1, as in the 2023 test signals),
  adc1 is not used and dim pieces are `undecided`.
- The channel lag differs per recording: adc1 is 18–33 ms earlier in the 2025 known-volume recordings and
  9–14 ms later in the tumour ones, so it is measured per recording.
- Features: a piece spans from 30 ms before its first peak to 30 ms after its last peak. Start, end,
  duration, number of peaks, max height and area (sum of the positive signal, counts x ms).
- Amount (`amount()`): total area of the `dyed` and `solid_no_dye` pieces, over the whole recording or the
  pieces that start in a time range. The number of pieces is for reference only: it stops growing when
  pieces merge in dense suctions.

### Part A rules (`preprocessing.py`)

- 1 kHz means keep peak height (99.3–99.9%) and area, because the signal has no content above about 500 Hz.
- `clean_channel()`: median filter (3 samples) -> Savitzky-Golay (7 samples, order 3) -> minus `baseline()`
  (ALS fitted on 10 Hz block means, interpolated back to 1 kHz).
- `quiet_sigma()`: RMS of the std of the quietest 10% of 200 ms blocks. Idle blocks (std below half the
  median block std, i.e. no flow) are dropped first; otherwise recordings with long idle periods get a sigma
  below the water wiggles and pieces merge (#3).
- `adc1_response()`: median adc1 / adc2 at the 50 largest adc2 peaks (#4).
