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
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

1. Enter a CSV/NPY path → **Load / convert**
2. **Run preprocessing** → inspect Sensor 2 stages
3. (Optional) open *Peak detection (optional / teammate)* if needed

Preprocessing code is shared from `../fluosense_code_only` (`run_preprocess_only`).

## Data policy

Signal files belong to Marginum / the course. **Do not commit CSV, ZIP, or `.npy` measurement data**. Keep data on UEF OneDrive (or Google Drive only if using Colab), and delete local copies when the course ends.
