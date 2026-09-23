# Marginum Signal Visualizer (IDP2026)

Objective 1 prototype: browse large Marginum CSV recordings in the browser without loading the whole file into RAM.

## Stack

- **Streamlit** — UI
- **Plotly** — plots
- **NumPy** — memory-mapped windowed I/O

## Setup

```bash
cd visualizer
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL (usually http://localhost:8501), enter a path to a CSV or `.npy` file, and click **Load / convert**.

## Data policy

Signal files belong to Marginum / the course. **Do not commit CSV, ZIP, or `.npy` measurement data** to this repository. Keep data on UEF OneDrive (or Google Drive only if using Colab), and delete local copies when the course ends.
