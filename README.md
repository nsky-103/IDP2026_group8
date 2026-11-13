# IDP2026 group 8 — Marginum fluorescence signal

Detect and judge aspirated tissue pieces in Marginum recordings.

**Input:** CSV with a header line `adc1,adc2` and two integer columns, sampled at 50 kHz.
**Output:** one row per tissue piece (start, end, height, area, label) and the total amount of tissue.

| Folder | What |
|---|---|
| `core/` | Data access and algorithms shared by the app and the scripts, no UI (see [`core/README.md`](core/README.md)) |
| `tools/` | Command-line scripts: `run_pipeline.py` (whole analysis), `benchmark_io.py` (speed test) |
| `visualizer/` | Streamlit app to browse a recording and its detected pieces (see [`visualizer/README.md`](visualizer/README.md)) |
| `report/` | Course report (LaTeX, `report.tex` + `sections/`) |
| `result.json` | Last benchmark: timings on the 7.6 GB recording and piece counts per test recording |

> **Data policy:** measurement data and anything derived from it are never committed. Keep recordings on
> UEF OneDrive and delete local copies when the course ends (details in [`visualizer/README.md`](visualizer/README.md#data-policy)).

## Setup

Python 3.11 or newer. All dependencies are listed in `visualizer/requirements.txt` (also used by `core/` and `tools/`).

```bash
python3.11 -m venv .venv            # Windows: py -3.11 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r visualizer/requirements.txt
```

`numba` and `pyarrow` are optional but make the first open of a large file 2–3x faster.

## Usage

Analyse one recording, or every `*.csv` under a folder:

```bash
python tools/run_pipeline.py PATH_TO_CSV_OR_FOLDER
```

| Option | Effect |
|---|---|
| `--out DIR` | Also write `DIR/<name>.pieces.csv` (one row per piece) for each recording. Keep `DIR` outside the repository. |
| `--no-cache` | Do not read or write the cache (always reads the CSV again) |
| `--workers N` | Number of worker processes (default: one per CPU core, at most 8) |

It prints one line per recording: length, number of pieces per label, noise level sigma, whether adc1 is
used, and the time of each step.

Open the interactive viewer:

```bash
streamlit run visualizer/app.py
```

## Processing flow

```
tools/run_pipeline.py  main()
  └─ core/pipeline.py  analyse()
       ├─ 1. open_recording()            CSV -> row index + 1 kHz summary (or read from the cache)
       ├─ 2. preprocessing.preprocess()  Part A: median -> Savitzky-Golay -> ALS baseline -> sigma (or read from the cache)
       └─ 3. detection.find_pieces()     Part B: peaks -> pieces -> labels -> features
```

Time per step for the 7.6 GB / 4.7 h recording on a MacBook (4 workers, numba and pyarrow installed;
from `result.json`):

| Step | First open | Next open |
|---|---|---|
| 1. CSV -> 1 kHz | 8.4 s (row index 3.7 s + summary 4.6 s) | about 0.1 s (cache) |
| 2. Part A | 1.6 s | about 0 s (cache) |
| 3. Part B | 1.9 s | 1.9 s |
| **Total** | **12.5 s** | **2.0 s** |

### Cache

The row index, the 1 kHz summary and the Part A output are saved in `~/.cache/idp2026` (change it with
the environment variable `IDP_CACHE_DIR`), about 270 MB for the 4.7 h recording. The Part A file name
includes a hash of the Part A parameters, so changing a parameter runs Part A again. Part B always runs
again. Use `--no-cache` to skip the cache.

### 1. Read the CSV and reduce it to 1 kHz — `core/pipeline.py`, `core/fast_io.py`

If the file is not in the cache:

1. `fast_io.build_index()` splits the file into 1024 blocks and counts the lines of each block in
   parallel, so any row can be found by its byte position.
2. `fast_io.summarize()` reads the blocks in parallel and takes the mean, min and max of every 50 samples
   (50 kHz -> 1 kHz; parsed with numba, else pyarrow, else pandas).
   - The **mean** is the input for the analysis. The signal has no content above about 500 Hz, so 1 ms
     means keep peak height (99.3–99.9%) and area.
   - **Min and max** are used to draw the whole recording without hiding narrow peaks.
3. The index and the summary are saved to the cache.

### 2. Part A: clean the signal — `core/preprocessing.py`

`preprocess()` does the following for adc1 and adc2:

1. `clean_channel()`
   - median filter (3 samples): removes single-sample spikes
   - Savitzky-Golay filter (7 samples, order 3): smooths small wiggles but keeps the peak shape
   - minus `baseline()`: the signal is averaged to 10 Hz, `als()` (asymmetric least squares) fits a
     slow line under the peaks, and the line is interpolated back to 1 kHz
2. `quiet_sigma()`: noise level sigma = RMS of the std of the quietest 10% of 200 ms blocks.
   Idle blocks (std below half the median block std, i.e. no flow) are dropped first. Otherwise
   recordings with long idle periods get a sigma below the water wiggles, and pieces merge
   ([#3](https://github.com/nsky-103/IDP2026_group8/issues/3)).
3. `adc1_response()`: median adc1 / adc2 at the 50 largest adc2 peaks. If it is below 0.1 (as in the
   2023 test signals, where adc1 hardly responds), adc1 is not used for the two-channel check
   ([#4](https://github.com/nsky-103/IDP2026_group8/issues/4)).

Output: cleaned adc1 / adc2 (baseline = 0, in counts), sigma per channel, and whether adc1 is used.

### 3. Part B: find and judge the pieces — `core/detection.py`

`find_pieces()`:

1. **Peaks**: `detect_peaks()` finds peaks with a prominence above 10 x sigma, on adc2 and on adc1.
   Peaks in the first and last 0.5 s of the file are dropped.
2. **Pieces**: peaks less than 1 s apart belong to the same piece (one aspirated tissue piece).
3. **Label**, from the largest peak of the piece:

   | Condition | Label |
   |---|---|
   | higher than the no-dye level (68.9 counts) | `dyed` |
   | lower, but confirmed by adc1 (two-channel check below) | `solid_no_dye` (tissue without dye) |
   | lower and not confirmed | `water` |
   | lower, and the check is not used in this recording | `undecided` |

   **Two-channel check** ([#11](https://github.com/nsky-103/IDP2026_group8/issues/11)): the piece is
   confirmed if there is an adc1 peak
   - within ±20 ms of the top adc2 peak, after shifting by the recording's channel lag, and
   - at least 5% as high as that adc2 peak (so small adc1 noise peaks do not pass by chance).

   The lag differs per recording (adc1 is 18–33 ms earlier in the 2025 known-volume recordings and
   9–14 ms later in the tumour ones), so it is measured on the dyed pieces of each recording.
   The chance rate is measured per recording by shifting adc1 by 2–20 s. The check is not used if the
   chance rate is above 20%, or if adc1 hardly responds (Part A step 3).

4. **Features**: a piece spans from 30 ms before its first peak to 30 ms after its last peak.
   For each piece: start, end, duration, number of peaks, max height, and area (sum of the positive
   signal, counts x ms).

5. **Amount**: `amount()` adds up the area of the `dyed` and `solid_no_dye` pieces, over the whole
   recording or over a time range (pieces that start in it). The area is the main measure of the amount
   of tissue. The number of pieces is given for reference only, because it stops growing when pieces
   merge in dense suctions. There is no ml value yet: the area per ml depends on the dye concentration
   and on re-suction.

Output: a table with one row per piece, the peak times for plotting, and the amount measures.

## Contributions

Who did what, per file. When code is based on another member's idea, the exact lines are marked in the
file with `# ---- Idea: <name> ...` and `# ---- end of <name>'s idea`.

| Part | Files | Who |
|---|---|---|
| Visualizer app | `visualizer/` | tut-morimoto, KeitoTsubaki (see git history) |
| Fast CSV access: row index, zoomed windows, 1 kHz summary | `core/fast_io.py`, `tools/benchmark_io.py` | Sora |
| Preprocessing: median + Savitzky-Golay, ALS baseline fitted on 10 Hz block means, sigma (quietest 10% of 200 ms blocks) | `core/preprocessing.py` (marked block) | **Unmona** (idea and original code in [udgithubit/IDP](https://github.com/udgithubit/IDP), `idp/method.py`); moved into this repository by Sora |
| Peak detection, grouping into pieces (1 s), judgement (no-dye level, two-channel check), features, amount (total area) | `core/detection.py` | Sora |
| Cache of the reduced data for the next open | `core/pipeline.py` (marked block) | **Unmona** (idea, `idp/fast.py`); written by Sora |
| Cache of the Part A output ([#17](https://github.com/nsky-103/IDP2026_group8/issues/17)) | `core/pipeline.py` (`save_part_a`, `load_part_a`) | tut-morimoto |
| Dropping idle blocks before sigma (#3) | `core/preprocessing.py` (marked inside `quiet_sigma`) | Sora |
| adc1 response check, `undecided` label (#4) | `core/preprocessing.py`, `core/detection.py` | Sora |
| Pipeline and command-line runner | `core/pipeline.py`, `tools/run_pipeline.py` | Sora |

The analysis behind the values (no-dye level 68.9, 1 s gap, ±20 ms) was done by Sora outside this
repository; the results are summarised in [`core/README.md`](core/README.md).
