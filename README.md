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
| Pipeline and command-line runner | `core/pipeline.py`, `tools/run_pipeline.py` | Sora |

Analysis behind the values (no-dye level 68.9, 1 s gap, ±20 ms) was done by Sora outside this
repository; the results are summarised in `core/README.md`.
