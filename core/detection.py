"""Part B: find the aspirated pieces in the cleaned signal and judge each one.

    pieces, peaks = find_pieces(clean)      # clean = preprocessing.preprocess(...)

1. Detect peaks on adc2 (and adc1): prominence > K_SIGMA x sigma.
2. Group: peaks less than GAP_S apart belong to one piece (one aspirated tissue piece).
3. Judge each piece by its largest peak:
   - higher than NO_DYE_LEVEL               -> "dyed"
   - otherwise an adc1 peak within +-PAIR_S -> "solid_no_dye" (tissue without dye)
   - otherwise                              -> "water"
   - if adc1 does not respond (clean["adc1_usable"] False, as in the 2023 test signals),
     dim pieces are "undecided" instead of the last two
4. Features per piece: start, end, duration, number of peaks, max height, area.

NO_DYE_LEVEL is the top 0.1% of sub-peak heights in two water-only recordings
(rule chosen on the Big pieces reference file).

Author: Sora.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

K_SIGMA = 10.0
GAP_S = 1.0
PAIR_S = 0.020
NO_DYE_LEVEL = 68.9
EDGE_S = 0.5
MIN_DIST_S = 0.004

COLUMNS = ["start_s", "end_s", "duration_s", "n_peaks", "max_height", "area", "confirmed", "label"]


def detect_peaks(x: np.ndarray, sigma: float, fs: float, k: float = K_SIGMA) -> np.ndarray:
    """Indices of peaks with prominence > k * sigma."""
    return find_peaks(x, prominence=k * sigma, distance=max(1, int(MIN_DIST_S * fs)))[0]


def find_pieces(clean: dict, k: float = K_SIGMA, gap_s: float = GAP_S,
                no_dye_level: float = NO_DYE_LEVEL) -> tuple[pd.DataFrame, dict]:
    """Pieces as a table (one row per piece) and the peak times for plotting."""
    fs = clean["fs"]
    a1, a2 = clean["adc1"], clean["adc2"]
    n = len(a2)
    edge = int(EDGE_S * fs)

    p2 = detect_peaks(a2, clean["sigma_adc2"], fs, k)
    p2 = p2[(p2 > edge) & (p2 < n - edge)]
    use_adc1 = clean.get("adc1_usable", True)
    p1 = detect_peaks(a1, clean["sigma_adc1"], fs, k) if use_adc1 else np.empty(0, np.int64)
    peaks = {"adc2": p2 / fs, "adc1": p1 / fs}
    if len(p2) == 0:
        return pd.DataFrame(columns=COLUMNS), peaks

    t1 = p1 / fs
    pad = int(0.030 * fs)
    rows = []
    for g in np.split(p2, np.flatnonzero(np.diff(p2) > gap_s * fs) + 1):
        top = g[np.argmax(a2[g])]
        h = float(a2[top])
        j = np.searchsorted(t1, top / fs)
        confirmed = any(abs(t1[q] - top / fs) <= PAIR_S for q in (j - 1, j) if 0 <= q < len(t1))
        s, e = max(0, g[0] - pad), min(n, g[-1] + pad)
        rows.append({
            "start_s": s / fs, "end_s": e / fs, "duration_s": (e - s) / fs,
            "n_peaks": len(g), "max_height": round(h, 1),
            "area": round(float(np.clip(a2[s:e], 0, None).sum() * 1000 / fs), 0),  # counts x ms
            "confirmed": bool(confirmed),
            "label": ("dyed" if h > no_dye_level else "undecided" if not use_adc1
                      else "solid_no_dye" if confirmed else "water"),
        })
    return pd.DataFrame(rows, columns=COLUMNS), peaks
