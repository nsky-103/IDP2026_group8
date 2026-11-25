# IDP2026 group 8 — Marginum fluorescence signal

Marginum measures the fluorescence of tissue pieces while they are aspirated during surgery. This project
reads the raw two-channel recordings, finds each aspirated tissue piece, judges whether it is dyed tissue,
undyed tissue or only water, and sums up the amount of tissue. It has a command-line analysis and an
interactive viewer.

![Interactive viewer: settings on the left, full-recording overview and detail window in the center, activity heatmap and detection results on the right](images/viewer.png)

Contents: [Requirements](#requirements) · [Setup](#setup) · [Data](#data-information) ·
[How to run](#how-to-run) · [Benchmarks](#benchmarks) · [Code structure](#code-structure) ·
[Method summary](#method-summary) · [Contributions](#contributions)

## Requirements

| | |
|---|---|
| Python | 3.11 (other 3.10+ versions are likely to work but are not tested) |
| OS | Tested on Windows 11 and macOS |
| Memory | 8 GB is enough; the 7.6 GB recording peaks at about 4.6 GB (3.6 GB main + 0.26 GB per worker) |
| Disk | Cache of about 270 MB per 4.7 h recording, in `~/.cache/idp2026` (outside the repository; safe to delete) |
| Packages | `visualizer/requirements.txt`: numpy, pandas, scipy, pybaselines, streamlit, plotly, matplotlib, pyarrow, numba |

## Setup

Run from the repository root. The one `requirements.txt` (in `visualizer/`) covers `core/`, `tools/` and
the viewer.

macOS / Linux:

```bash
git clone https://github.com/nsky-103/IDP2026_group8.git
cd IDP2026_group8
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r visualizer/requirements.txt
```

Windows (PowerShell):

```powershell
git clone https://github.com/nsky-103/IDP2026_group8.git
cd IDP2026_group8
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r visualizer/requirements.txt
```

Measurement data and anything derived from it are never committed (data policy in `visualizer/README.md`).

## Data information

The recordings are **not in this repository** (data policy below). They are provided by Marginum
through the course, as the archive `2025-10-29 Suction data for tumor size estimation.7z`
(about 12 GB unpacked). Extract it anywhere outside the repository.

**Format:** CSV, a header line `adc1,adc2`, then two integer columns sampled at 50 kHz.
`adc2` is the main fluorescence channel; `adc1` is the second channel used to confirm undyed tissue.

```
adc1,adc2
510,604
499,605
```

**Known-volume recordings** (`Suction data/Original/`, from `Info on measurements.odt`):

| # | Dye concentration | Volume (ml) | Re-suctions | Size |
|---|---|---|---|---|
| 1 | HIGH | 4 | 3 | 7.6 GB, 4.7 h |
| 2 | HIGH | 4 | 2 | |
| 3 | HIGH | 2 | 2 | |
| 4 | HIGH | 1 | 2 | |
| 5 | HIGH | 1 | 0 | 30 MB, 1.2 min — **use this one for a first try** |
| 6 | MIDDLE | 3 | 0 | |
| 7 | MIDDLE | 2 | 1 | |
| 8 | MIDDLE | 1 | 0 | |
| 9 | MIDDLE | 1 | 0 | |

**Data policy:** signal files belong to Marginum / the course. Do not commit CSV, ZIP, NPY or any file
derived from them (`.gitignore` blocks the common ones). Keep data on UEF OneDrive and delete local copies
when the course ends.

## How to run

With the virtual environment activated, from the repository root:

```bash
streamlit run visualizer/app.py
```

Open http://localhost:8501 in the browser. Quick start:

1. **Browse…** (top left) → pick `5. suction_HIGH_CONC_1ml_and_no_resuction.csv`.
2. Wait until the run shows *ready* → **Detect peaks**.
3. Open **Events** (right) and click a piece to move the detail window to it.

The same analysis can also be run without the viewer: `python tools/run_pipeline.py PATH_TO_CSV_OR_FOLDER`
(options: `python tools/run_pipeline.py --help`).

### Using the viewer

The page has three columns: settings on the left, plots in the center, heatmap and results on the right.

**Left — load and settings**

| Control | What it does |
|---|---|
| **Browse…** | Opens the file dialog. Only the path is passed, so multi-GB files are not copied. |
| *…or paste a local path* → **Load** | Same, by typing or pasting the path (quotes are removed). |
| **Runs** | Every loaded recording, with its state (*preview* → *ready*) and number of events. Click one to switch to it. |
| **Channels** | Show or hide Sensor 1 (adc1) and Sensor 2 (adc2) in the plots. |
| **Heatmap settings** | Row length (10 s – 5 min), cells per row, and what the colour shows (signal activity of Sensor 2 or 1, or number of detected pieces). After **Detect peaks** it switches to the number of pieces. |
| **Detect peaks** | Runs detection with the method described below. Enabled once the 1 kHz summary is ready (a few seconds; the first load of a large file takes longer). |
| **Detection parameters** | Peak prominence in x sigma, max gap inside one piece and no-dye level; the defaults are the tuned values. |
| **Show rejected / water (gray)** | Also show `water` and `undecided` pieces, in gray. |
| **Display settings** | Sample rate (50 kHz), max plot points, and how the detail plot is thinned (LTTB keeps the peaks). |

**Center — plots**

| Area | How to use it |
|---|---|
| **Full recording** (top) | The whole recording, with a mark for each detected piece. The black frame is the part shown below: drag it to move it, drag its sides to change its width, or click outside it to centre it there. **Window size** sets the width. |
| **Detail window** (below) | Exact samples of both channels in the frame, with the pieces marked by triangles (rejected ones in gray). Buttons: **Zoom in / out**, **Pan left / right**, **Default** (first 5 s), **Fit all** (whole recording). Drag the plot to pan. |

**Right — heatmap and results**

| Area | What it shows |
|---|---|
| **Activity heatmap** (top) | The recording folded into rows (e.g. one row per minute); bright cells = strong signal, or pieces after detection. The dotted outline is the current frame. Click a cell to jump there. |
| **Detection summary** | Counts per label and the full table of pieces. |
| **Events** | The accepted pieces, largest first, each with its time, label and amplitude. Click one to centre the frame on it (the width stays the same). |

## Benchmarks

Two scripts measure the speed and the comparison with simpler methods on your own computer. Run them from the repository
root with the virtual environment activated. They only read the data; `--json out.json` also saves
all numbers.

```bash
python tools/benchmark_io.py PATH_TO_CSV
python tools/benchmark_methods.py DATA_FOLDER
```

**`benchmark_io.py`**: how fast one recording opens (use the largest one, e.g. the 4.7 h recording).

| Section | How to read it |
|---|---|
| 1. This computer | CPU, RAM and the optional speed-ups (numba, pyarrow); results depend on these |
| 2. First open | Time for each step of the viewer when the file is opened for the first time. For a true first open, run `sudo purge` (macOS) or restart just before |
| 3. File in memory | The same steps again. If section 2 was about as fast, the file was already in memory |
| 4. Second open | Opening again from the app's cache; this is what the viewer does after the first time |
| 5. Old methods | Earlier readers (csv module, pandas chunks), timed on part of the file and scaled up |
| 6. Summary | Main numbers, peak memory, and `OK` if the 1 kHz data match the CSV |

**`benchmark_methods.py`**: the current Part A/B methods against simpler ones, on every recording in
`DATA_FOLDER` (the first run reads all files, later runs use the cache).

| Section | How to read it |
|---|---|
| 1. Noise level sigma | Events per recording with the current sigma (idle parts dropped) and with all parts. With all parts, idle periods make sigma too small and separate pieces join into one event, so there are fewer events, mainly in recordings with long idle parts |
| 2. Two-channel check | Chance rate: how often adc1 shows a peak by chance (lower is better; above 0.20 the check is not used). Confirmed: share of dyed events also seen on adc1 (higher is better) |
| 3. Big pieces | Labels per section with both checks, if the Big pieces recording is in the folder |

## Code structure

```
core/                    data access and algorithms, no UI (details: core/README.md)
  fast_io.py             parallel CSV reading: row index, exact windows, 1 kHz summary
  preprocessing.py       Part A: filters, ALS baseline, noise level sigma
  detection.py           Part B: peaks, pieces, labels, features, amount
  pipeline.py            analyse(): runs 1 -> Part A -> Part B, with the cache
tools/
  run_pipeline.py        command-line analysis of files or folders
  benchmark_io.py        speed test of each step (with --json, saves all numbers)
  benchmark_methods.py   current Part A/B methods against simpler ones
visualizer/              Streamlit viewer (details: visualizer/README.md)
  app.py                 page layout
  panels/                one area of the page each (import, controls, overview, detail, heatmap, events)
  detectors.py           algorithms offered in the viewer; the default calls core Part A/B
  signal_io.py, load_jobs.py   reading a recording and loading it in the background
  plots.py, lttb.py      figures and LTTB downsampling for display
  requirements.txt       dependencies of the whole project
report/                  course report (LaTeX: report.tex + sections/)
result.json              last benchmark: timings and piece counts per test recording
images/                  screenshots used in this README
```

The viewer and the command line use the same `core` code, so they give the same pieces.

## Method summary

```
CSV (50 kHz)  ->  1 kHz summary  ->  Part A: clean  ->  Part B: find and judge pieces  ->  amount
               core/fast_io.py    core/preprocessing.py  core/detection.py
```

1. **Reduce to 1 kHz**: the CSV is read in parallel blocks and averaged over every 50 samples. The
   signal has no content above about 500 Hz, so peak height and area are kept.
2. **Part A, clean**: median + Savitzky-Golay filters remove spikes and small wiggles, an ALS baseline
   removes the slow drift, and the noise level sigma is taken from the quietest parts of the recording.
3. **Part B, find and judge pieces**: peaks higher than 10 x sigma, closer than 1 s, are grouped into one
   piece (one aspirated tissue piece). Each piece is labelled:
   - `dyed`: higher than the no-dye level (68.9 counts)
   - `solid_no_dye`: lower, but adc1 also shows a peak at the same time (tissue without dye)
   - `water`: lower and not confirmed by adc1
   - `undecided`: lower, and adc1 cannot be used in this recording
4. **Amount**: the area of the `dyed` and `solid_no_dye` pieces is summed up as a measure of the amount of
   tissue. There is no ml value yet, because the area per ml depends on the dye concentration.

Details and the reasons for each value: [`core/README.md`](core/README.md) and the report (`report/`).

## Contributions

One row per component. When code is based on another member's idea, the exact lines are marked in the
file with `# ---- Idea: <name> ...` and `# ---- end of <name>'s idea`. Details are in the git history.

| Component | Files | Who | What |
|---|---|---|---|
| Visualizer | `visualizer/` | tut-morimoto, KeitoTsubaki | The Streamlit app: layout, panels, background loading, plots, detector interface |
| Data access | `core/fast_io.py`, `tools/benchmark_io.py` | Sora | Parallel CSV reading: row index, exact zoomed windows, 1 kHz summary; speed benchmark |
| Part A: preprocessing | `core/preprocessing.py` | Unmona, Sora | Unmona: filters, ALS baseline and sigma (original code in [udgithubit/IDP](https://github.com/udgithubit/IDP), `idp/method.py`) and their speed-up. Sora: moved it into this repository, idle-block handling (#3), adc1 response check (#4) |
| Part B: detection | `core/detection.py`, `tools/benchmark_methods.py` | Sora | Peaks, grouping into pieces, labels (no-dye level, two-channel check), features, amount; comparison with simpler methods. The analysis behind the values (68.9 counts, 1 s, ±20 ms) was done by Sora outside this repository (summary in [`core/README.md`](core/README.md)) |
| Pipeline and cache | `core/pipeline.py`, `tools/run_pipeline.py` | Sora, Unmona, tut-morimoto | Sora: pipeline and command-line runner. Unmona: idea of caching the reduced data (`idp/fast.py`). tut-morimoto: cache of the Part A output (#17) |
| Documentation | `README.md`, `core/README.md`, `visualizer/README.md` | Sora, KeitoTsubaki, tut-morimoto | Sora: `README.md` and `core/README.md`. KeitoTsubaki: setup, usage and viewer guide. tut-morimoto, KeitoTsubaki: `visualizer/README.md` |
