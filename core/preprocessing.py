"""Part A: clean the 1 kHz signal from ``fast_io.summarize`` before peak detection.

    summ  = fast_io.summarize(idx)          # 1 kHz block means, (n, 2) adc1 / adc2
    clean = preprocess(summ["mean"])        # denoised, baseline = 0, sigma per channel

Steps per channel: median (3) -> Savitzky-Golay (7, order 3) -> ALS baseline subtracted,
then sigma = noise level of the quietest 10% of 200 ms blocks.

Contributions (see the README at the repository root):
- The processing itself is Unmona's preprocessing (udgithubit/IDP, idp/method.py), marked below.
- Input from fast_io and the output dict for detection.py: Sora.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.signal import medfilt, savgol_filter
from scipy.sparse.linalg import spsolve

FS_HZ = 1000.0

# ---- Idea: Unmona (udgithubit/IDP, idp/method.py: P, als, baseline, clean, quiet_sigma) ---------
# Median + Savitzky-Golay denoising, then an ALS baseline fitted on 10 Hz block means and
# interpolated back to 1 kHz (fitting ALS at 10 Hz is what makes it take about 1 s on 4.7 h).
# sigma: RMS of the std of the quietest 10% of 200 ms blocks.
# The values are Unmona's first guesses; the slide values may replace them.
PARAMS = {"median": 3, "sg_window": 7, "sg_order": 3,
          "als_lam": 1e5, "als_p": 0.01, "als_iter": 10, "als_block": 100}


def als(y: np.ndarray, lam: float, p: float, n_iter: int) -> np.ndarray:
    """Asymmetric least squares baseline (Eilers & Boelens 2005)."""
    n = len(y)
    d = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
    h = lam * (d.T @ d)
    w = np.ones(n)
    z = y
    for _ in range(n_iter):
        z = spsolve((sparse.diags(w) + h).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def baseline(y: np.ndarray, params: dict = PARAMS) -> np.ndarray:
    """ALS on block means (10 Hz for 1 kHz input), interpolated back to every sample."""
    b = params["als_block"]
    n = len(y) // b * b
    c = y[:n].reshape(-1, b).mean(1)
    z = als(c, params["als_lam"], params["als_p"], params["als_iter"])
    return np.interp(np.arange(len(y)), (np.arange(len(c)) + 0.5) * b, z)


def clean_channel(x: np.ndarray, params: dict = PARAMS) -> np.ndarray:
    x = medfilt(np.asarray(x, np.float64), params["median"])
    x = savgol_filter(x, params["sg_window"], params["sg_order"])
    return x - baseline(x, params)


def quiet_sigma(x: np.ndarray, fs: float = FS_HZ, block_s: float = 0.2, frac: float = 0.10) -> float:
    b = int(block_s * fs)
    n = len(x) // b
    s = np.sort(x[: n * b].reshape(n, b).std(1))
    return float(np.sqrt(np.mean(s[: max(1, int(frac * n))] ** 2)))
# ---- end of Unmona's idea ---------------------------------------------------------------------


def preprocess(mean_1khz: np.ndarray, fs: float = FS_HZ, params: dict = PARAMS) -> dict:
    """(n, 2) 1 kHz block means -> cleaned adc1 / adc2 (float32) and sigma per channel."""
    a1 = clean_channel(mean_1khz[:, 0], params)
    a2 = clean_channel(mean_1khz[:, 1], params)
    return {
        "fs": fs,
        "adc1": a1.astype(np.float32),
        "adc2": a2.astype(np.float32),
        "sigma_adc1": quiet_sigma(a1, fs),
        "sigma_adc2": quiet_sigma(a2, fs),
    }
