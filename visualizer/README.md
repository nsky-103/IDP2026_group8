# Marginum Signal Visualizer (IDP2026)

Noise reduction + interactive visualization for large Marginum CSV recordings.

## Scope (this workstream)

- Load CSV / NPY and browse raw windows
- **Run preprocessing**: downsample → denoise (median + Savitzky–Golay) → baseline
- Plot cleaned stages (raw / denoised / baseline / corrected)
- Plot downsampling with **LTTB** (Steinarsson 2013)

Peak / feature detection is **optional** (expander in the sidebar) for teammate algorithms.

## Plot downsampling (LTTB)

- Sveinn Steinarsson (2013). *Downsampling Time Series for Visual Representation*. M.Sc. thesis, University of Iceland.  
  https://skemman.is/bitstream/1946/15343/3/SS_MSthesis.pdf  
  Implementation: `lttb.py` (Algorithm 4.2).

## Setup

```bash
cd visualizer
python3.11 -m venv .venv  # Windows: py 3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

1. Enter a CSV/NPY path → **Load**
2. Pick an **Algorithm**, adjust its parameters → **Detect peaks**
3. Click an event (or a heatmap cell) to zoom to it

## Code layout

Drawing and algorithms are kept apart:

| Part | Files | Streamlit? |
|---|---|---|
| Page layout | `app.py` | yes |
| One area of the page each | `panels/` (`data_import`, `controls`, `overview`, `detail`, `heatmap`, `events`) | yes |
| State kept between clicks | `state.py` | yes |
| Viewing-window math | `viewport.py` | no |
| Figures | `plots.py`, `lttb.py` | no |
| Reading the signal | `signal_io.py`, `convert_jobs.py` | no |
| Detection algorithms | `detectors.py` | no |

Every algorithm returns the same `Detection` (an `events` table with at least `t_peak`, `amplitude`,
`valid`, `label`), so all panels work with any algorithm.

**Add an algorithm:** write `def my_detector(csv_path, fs_hz, **params) -> Detection` in
`detectors.py` and add a `Detector(...)` entry with its `Param`s to `DETECTORS`. It appears in
the *Algorithm* menu with its own controls; no UI code needed. Add a `size_mm` column to fill the
"estimated size" on the event card.

The *Threshold · fluosense* algorithm needs `fluosense_code_only` next to this repository (it is not
part of it). *Pieces · core pipeline* uses only this repository.

## Data policy

Signal files belong to Marginum / the course. **Do not commit CSV, ZIP, or `.npy` measurement data**. Keep data on UEF OneDrive (or Google Drive only if using Colab), and delete local copies when the course ends.
