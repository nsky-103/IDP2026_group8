# IDP2026 group 8 — Marginum fluorescence signal

Detect and judge aspirated tissue pieces in Marginum recordings (CSV, two integer columns `adc1,adc2`, 50 kHz).

| Folder | What |
|---|---|
| `visualizer/` | Streamlit app (see `visualizer/README.md`) |
| `core/` | Data access and algorithms shared by the app and scripts, no UI (see `core/README.md`) |
| `tools/` | Command-line scripts: benchmark, run the whole analysis |

Quick run on one recording or a folder:

```bash
python tools/run_pipeline.py PATH_TO_CSV_OR_FOLDER
```

Measurement data and anything derived from it are never committed (data policy in `visualizer/README.md`).

## Processing flow

```
tools/run_pipeline.py  main()
  └─ core/pipeline.py  analyse()
       ├─ 1. open_recording()          CSV -> row index + 1 kHz summary (or read from the cache)
       ├─ 2. preprocessing.preprocess()  Part A: median -> Savitzky-Golay -> ALS baseline -> sigma
       └─ 3. detection.find_pieces()     Part B: peaks -> pieces -> labels -> features
```

Times are for the 7.6 GB / 4.7 h recording on a MacBook (4 workers).

### 1. Read the CSV and reduce it to 1 kHz — `core/pipeline.py`, `core/fast_io.py`

`open_recording()` first looks in the cache (`~/.cache/idp2026`, or `IDP_CACHE_DIR`). If the file was
opened before, the row index and the 1 kHz data are loaded from there (0.1 s). Otherwise:

1. `fast_io.build_index()` splits the file into 1024 blocks and counts the lines of each block in
   parallel, so any row can be found by its byte position (3.6 s).
2. `fast_io.summarize()` reads the blocks in parallel and takes the mean, min and max of every 50 samples
   (50 kHz -> 1 kHz, 4.2 s with numba, 7.9 s with pyarrow, 11 s with pandas). The mean is the input for the analysis: the signal has no content above
   about 500 Hz, so 1 ms means keep peak height (99.3–99.9%) and area. Min and max are for drawing the
   whole recording without hiding narrow peaks.
3. The index and the summary are saved to the cache for the next open.

### 2. Part A: clean the signal — `core/preprocessing.py`

`preprocess()` does the following for adc1 and adc2 (1.8 s for both):

1. `clean_channel()`
   - median filter (3 samples): removes single-sample spikes
   - Savitzky-Golay filter (7 samples, order 3): smooths small wiggles but keeps the peak shape
   - minus `baseline()`: the signal is averaged to 10 Hz, `als()` (asymmetric least squares) fits a
     slow line under the peaks, and the line is interpolated back to 1 kHz
2. `quiet_sigma()`: noise level sigma = RMS of the std of the quietest 10% of 200 ms blocks.
   Idle blocks (std below half the median block std, i.e. no flow) are dropped first; otherwise
   recordings with long idle periods get a sigma below the water wiggles and pieces merge (#3).

3. `adc1_response()`: median adc1 / adc2 at the 50 largest adc2 peaks. Below 0.1 (the 2023 test
   signals, where adc1 hardly responds) adc1 is not used for the two-channel check (#4).

Output: cleaned adc1 / adc2 (baseline = 0, in counts), sigma per channel and whether adc1 is used.

### 3. Part B: find and judge the pieces — `core/detection.py`

`find_pieces()` (1.8 s):

1. **Peaks**: `detect_peaks()` finds peaks whose prominence is above 10 x sigma, on adc2 and on adc1.
   Peaks in the first and last 0.5 s of the file are dropped.
2. **Pieces**: peaks less than 1 s apart belong to the same piece (one aspirated tissue piece).
3. **Label**, from the largest peak of the piece:

   | Condition | Label |
   |---|---|
   | higher than the no-dye level (68.9 counts) | `dyed` |
   | lower, but confirmed by adc1 (two-channel check below) | `solid_no_dye` (tissue without dye) |
   | lower and not confirmed | `water` |
   | lower, and the check is not used in this recording | `undecided` |

   Two-channel check (#11): an adc1 peak within ±20 ms of the top adc2 peak shifted by the recording's channel lag, at least 5% as high as that adc2 peak. The lag differs per recording (adc1 18–33 ms earlier in the 2025 known-volume recordings, 9–14 ms later in the tumour ones) and is measured on the dyed pieces. The height condition keeps small adc1 noise peaks from passing by chance. The chance rate is measured per recording by shifting adc1 by 2–20 s; above 20%, or when adc1 hardly responds (Part A step 3), the check is not used.

4. **Features**: the piece spans from 30 ms before its first peak to 30 ms after its last peak.
   Start, end, duration, number of peaks, max height and area (sum of the positive signal, counts x ms).

Output: a table with one row per piece, and the peak times for plotting.

| Step | First open | Next open |
|---|---|---|
| 1. CSV -> 1 kHz | about 7.5 s (numba; index 3.3 s + summary 4.2 s) | 0.1 s (cache) |
| 2. Part A | 1.8 s | 1.8 s |
| 3. Part B | 1.8 s | 1.8 s |
| Total | about 11 s | about 3.7 s |

## Contributions

Who did what, per file. When code is based on another member's idea, the exact lines are marked in the
file with `# ---- Idea: <name> ...` and `# ---- end of <name>'s idea`.

| Part | Files | Who |
|---|---|---|
| Visualizer app | `visualizer/` | tut-morimoto, KeitoTsubaki (see git history) |
| Fast CSV access: row index, zoomed windows, 1 kHz summary, parallel `.npy` conversion | `core/fast_io.py`, `tools/benchmark_io.py` | Sora |
| Preprocessing: median + Savitzky-Golay, ALS baseline fitted on 10 Hz block means, sigma (quietest 10% of 200 ms blocks) | `core/preprocessing.py` (marked block) | **Unmona** (idea and original code in [udgithubit/IDP](https://github.com/udgithubit/IDP), `idp/method.py`); moved into this repository by Sora |
| Peak detection, grouping into pieces (1 s), judgement (no-dye level, two-channel check), features | `core/detection.py` | Sora |
| Cache of the reduced data for the next open | `core/pipeline.py` (marked block) | **Unmona** (idea, `idp/fast.py`); written by Sora |
| Dropping idle blocks before sigma (#3) | `core/preprocessing.py` (marked inside `quiet_sigma`) | Sora |
| adc1 response check, `undecided` label (#4) | `core/preprocessing.py`, `core/detection.py` | Sora |
| Pipeline and command-line runner | `core/pipeline.py`, `tools/run_pipeline.py` | Sora |

Analysis behind the values (no-dye level 68.9, 1 s gap, ±20 ms) was done by Sora outside this
repository; the results are summarised in `core/README.md`.
