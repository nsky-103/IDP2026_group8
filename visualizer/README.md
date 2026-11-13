# Marginum Signal Visualizer (IDP2026)

Interactive viewer for large Marginum CSV recordings and the pieces detected in them.

## What it does

- Opens multi-GB CSV files quickly: a rough overview first, then the exact 1 kHz summary in the
  background (`core.fast_io`, cached in `~/.cache/idp2026`)
- Overview of the whole recording, a detail zoom with exact samples, and an activity heatmap
- Runs a detection algorithm from the *Algorithm* menu. The default, `Method B · core Part A/B`, is the
  same analysis as `tools/run_pipeline.py` (see the [root README](../README.md#processing-flow))
- Shows the detected pieces as an events list and on the plots; click one to zoom to it
- The detail plot is downsampled for display with **LTTB** (Steinarsson 2013)

## Plot downsampling (LTTB)

- Sveinn Steinarsson (2013). *Downsampling Time Series for Visual Representation*. M.Sc. thesis, University of Iceland.  
  https://skemman.is/bitstream/1946/15343/3/SS_MSthesis.pdf  
  Implementation: `lttb.py` (Algorithm 4.2).

## Setup

```bash
cd visualizer
python3.11 -m venv .venv  # Windows: py -3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

1. Enter a CSV path → **Load** (rough overview first, then the 1 kHz summary in the background)
2. When the run shows *ready*: pick an **Algorithm**, adjust its parameters → **Detect peaks**
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
| Reading the signal | `signal_io.py` (`Recording`), `load_jobs.py` (background load) | no |
| Detection algorithms | `detectors.py` | no |

Every algorithm returns the same `Detection` (an `events` table with at least `t_peak`, `amplitude`,
`valid`, `label`), so all panels work with any algorithm.

**Add an algorithm:** write `def my_detector(recording, **params) -> Detection` in `detectors.py`
(`recording.summary` holds the 1 kHz mean/min/max, so the CSV is not read again) and add a
`Detector(...)` entry with its `Param`s to `DETECTORS`. It appears in the *Algorithm* menu with its
own controls; no UI code needed. Add a `size_mm` column to fill the "estimated size" on the event
card.

## Data policy

Signal files belong to Marginum / the course. **Do not commit CSV, ZIP, or `.npy` measurement data**. Keep data on UEF OneDrive (or Google Drive only if using Colab), and delete local copies when the course ends.
