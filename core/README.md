# core

Code shared by the visualizer and scripts. No UI code here.

## fast_io.py — open large recordings without making the user wait

| Step | Function | 7.6 GB / 4.7 h, 4 workers |
|---|---|---|
| Rough overview (min/max at 2000 places) | `preview(csv)` | 0.7 s |
| Byte offset ↔ row number index | `build_index(csv)` | 3 s |
| Exact samples of a zoomed window | `read_window(idx, start_s, end_s)` | 10 s: 0.04 s, 60 s: 0.13 s |
| 1 kHz mean / min / max of the whole file | `summarize(idx)` | 12 s |
| Full-rate (N, 2) int16 `.npy` (optional) | `convert_to_npy(csv, idx, out)` | 12 s |

Measured on a MacBook (8 cores) with the file already read once. Peak memory about 2.2 GB in total
(0.85 GB main process + about 0.35 GB per worker), well within a 16 GB laptop.
Run `python tools/benchmark_io.py PATH_TO_CSV` to measure on another computer.

Suggested order in the app:

1. `preview` → draw the rough overview at once (mark it as approximate: short peaks between the read places are missed)
2. `build_index` → from now on, zooming reads exact samples with `read_window`
3. `summarize` in the background → replace the overview with the exact one; run analysis on the 1 kHz data

`convert_to_npy` writes the same format as `visualizer/signal_io.convert_csv_to_npy`, so the
existing `open_signal` can read it. It is only needed if a full-rate copy is wanted; `read_window`
already gives full-rate samples for any window.

Checked against `pandas.read_csv` of the whole file: row count, converted values, random windows,
first/last rows and times all match, for both `\r\n` and `\n` line endings.

Notes:
- Worker processes are started with `ProcessPoolExecutor`. On Windows, code that calls these functions
  must be importable and scripts need `if __name__ == "__main__":`.
- The sampling rate is assumed to be 50 kHz (`DEFAULT_FS_HZ`).
- Output files must not go into the repository (see the data policy in `visualizer/README.md`).
