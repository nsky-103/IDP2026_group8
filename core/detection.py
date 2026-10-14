"""Part B: find the aspirated pieces in the cleaned signal and judge each one.

    pieces, peaks = find_pieces(clean)      # clean = preprocessing.preprocess(...)

1. Detect peaks on adc2 (and adc1): prominence > K_SIGMA x sigma.
2. Group: peaks less than GAP_S apart belong to one piece (one aspirated tissue piece).
3. Judge each piece by its largest peak:
   - higher than NO_DYE_LEVEL                     -> "dyed"
   - otherwise confirmed by adc1 (two-channel check) -> "solid_no_dye" (tissue without dye)
   - otherwise                                    -> "water"
   - if adc1 does not respond (clean["adc1_usable"] False, as in the 2023 test signals) or the
     check passes too often by chance in this recording, dim pieces are "undecided" instead
4. Features per piece: start, end, duration, number of peaks, max height, area.

Two-channel check (GitHub #11): an adc1 peak within +-PAIR_S of the top adc2 peak shifted by the
recording's channel lag, at least AMP_RATIO x as high as that adc2 peak.
- The lag between the sensors differs per recording (adc1 18-33 ms earlier in the 2025
  known-volume recordings, 9-14 ms later in the tumour ones), so it is measured on the dyed
  pieces, which are judged by brightness alone.
- A real piece shows on both sensors (adc1/adc2 at large peaks 0.12-0.64); the height condition
  keeps small adc1 noise peaks from passing by chance (4.7 h recording: 42% -> 15%).
- The chance rate is measured per recording by shifting the adc1 peaks by 2-20 s.

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

AMP_RATIO = 0.05        # adc1 peak at least this x the top adc2 peak
LAG_SEARCH_S = 0.150    # nearest adc1 peak within this of a dyed piece's top -> one lag sample
MIN_LAG_PIECES = 5      # fewer lag samples: assume no lag
MAX_CHANCE = 0.20       # chance rate above this: the check is not used in this recording
CHANCE_SHIFTS = 50

COLUMNS = ["start_s", "end_s", "duration_s", "n_peaks", "max_height", "area", "confirmed", "label"]


def detect_peaks(x: np.ndarray, sigma: float, fs: float, k: float = K_SIGMA) -> np.ndarray:
    """Indices of peaks with prominence > k * sigma."""
    return find_peaks(x, prominence=k * sigma, distance=max(1, int(MIN_DIST_S * fs)))[0]


def channel_lag(t_top: np.ndarray, t1: np.ndarray) -> float:
    """Median (adc1 peak time - top adc2 peak time) over the given pieces, in s (0 if too few)."""
    if len(t1) == 0:
        return 0.0
    j = np.clip(np.searchsorted(t1, t_top), 1, len(t1) - 1)
    d = np.where(np.abs(t1[j] - t_top) < np.abs(t1[j - 1] - t_top), t1[j] - t_top, t1[j - 1] - t_top)
    d = d[np.abs(d) <= LAG_SEARCH_S]
    return float(np.median(d)) if len(d) >= MIN_LAG_PIECES else 0.0


def two_channel_confirmed(t_top: np.ndarray, h_top: np.ndarray, t1: np.ndarray, h1: np.ndarray,
                          lag: float) -> np.ndarray:
    """Per piece: is there an adc1 peak within +-PAIR_S of (top + lag), at least AMP_RATIO x as high?"""
    out = np.zeros(len(t_top), bool)
    if len(t1) == 0:
        return out
    lo = np.searchsorted(t1, t_top + lag - PAIR_S)
    hi = np.searchsorted(t1, t_top + lag + PAIR_S, side="right")
    for i in np.flatnonzero(hi > lo):
        out[i] = h1[lo[i]:hi[i]].max() >= AMP_RATIO * h_top[i]
    return out


def chance_rate(t_top, h_top, t1, h1, lag, total_s, seed: int = 0) -> float:
    """Share of pieces confirmed when adc1 is shifted by 2-20 s (no real pairs left), mean of shifts."""
    if len(t1) == 0 or len(t_top) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(CHANCE_SHIFTS):
        shifted = (t1 + rng.uniform(2, 20) * rng.choice([-1, 1])) % total_s
        order = np.argsort(shifted)
        rates.append(two_channel_confirmed(t_top, h_top, shifted[order], h1[order], lag).mean())
    return float(np.mean(rates))


def find_pieces(clean: dict, k: float = K_SIGMA, gap_s: float = GAP_S,
                no_dye_level: float = NO_DYE_LEVEL) -> tuple[pd.DataFrame, dict]:
    """Pieces as a table (one row per piece) and the peak times for plotting.

    peaks["two_channel"] reports the check for this recording: used, lag_s, chance.
    """
    fs = clean["fs"]
    a1, a2 = clean["adc1"], clean["adc2"]
    n = len(a2)
    edge = int(EDGE_S * fs)

    p2 = detect_peaks(a2, clean["sigma_adc2"], fs, k)
    p2 = p2[(p2 > edge) & (p2 < n - edge)]
    use_adc1 = clean.get("adc1_usable", True)
    p1 = detect_peaks(a1, clean["sigma_adc1"], fs, k) if use_adc1 else np.empty(0, np.int64)
    peaks = {"adc2": p2 / fs, "adc1": p1 / fs,
             "two_channel": {"used": False, "lag_s": 0.0, "chance": float("nan")}}
    if len(p2) == 0:
        return pd.DataFrame(columns=COLUMNS), peaks

    groups = np.split(p2, np.flatnonzero(np.diff(p2) > gap_s * fs) + 1)
    tops = np.array([g[np.argmax(a2[g])] for g in groups])
    h_top = a2[tops].astype(float)
    t_top = tops / fs
    dyed = h_top > no_dye_level

    confirmed = np.zeros(len(groups), bool)
    if use_adc1:
        t1, h1 = p1 / fs, a1[p1].astype(float)
        lag = channel_lag(t_top[dyed], t1)
        confirmed = two_channel_confirmed(t_top, h_top, t1, h1, lag)
        chance = chance_rate(t_top, h_top, t1, h1, lag, n / fs)
        use_adc1 = chance <= MAX_CHANCE
        peaks["two_channel"] = {"used": bool(use_adc1), "lag_s": lag, "chance": chance}

    pad = int(0.030 * fs)
    rows = []
    for i, g in enumerate(groups):
        s, e = max(0, g[0] - pad), min(n, g[-1] + pad)
        rows.append({
            "start_s": s / fs, "end_s": e / fs, "duration_s": (e - s) / fs,
            "n_peaks": len(g), "max_height": round(float(h_top[i]), 1),
            "area": round(float(np.clip(a2[s:e], 0, None).sum() * 1000 / fs), 0),  # counts x ms
            "confirmed": bool(confirmed[i]),
            "label": ("dyed" if dyed[i] else "undecided" if not use_adc1
                      else "solid_no_dye" if confirmed[i] else "water"),
        })
    return pd.DataFrame(rows, columns=COLUMNS), peaks
