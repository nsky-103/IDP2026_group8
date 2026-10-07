"""
Largest-Triangle-Three-Buckets (LTTB) downsampling for line-chart display.

Implements Algorithm 4.2 from:

  Sveinn Steinarsson, 2013.
  "Downsampling Time Series for Visual Representation."
  M.Sc. thesis, University of Iceland.
  https://skemman.is/bitstream/1946/15343/3/SS_MSthesis.pdf
  http://hdl.handle.net/1946/15343

LTTB keeps perceptually important peaks/troughs better than every-nth
striding by selecting, in each bucket, the point that forms the largest
triangle with the previously chosen point and the average of the next bucket.
"""

from __future__ import annotations

import numpy as np


def lttb_indices(x: np.ndarray, y: np.ndarray, threshold: int) -> np.ndarray:
    """
    Return indices of points selected by LTTB.

    Parameters
    ----------
    x, y : 1-D arrays of equal length (independent / dependent variables).
    threshold : desired number of output points (must be >= 2).

    Returns
    -------
    idx : int ndarray of length <= threshold, strictly increasing.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("x and y must be 1-D arrays of equal length")

    n = x.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)
    if threshold < 2:
        raise ValueError("threshold must be >= 2")
    if threshold >= n:
        return np.arange(n, dtype=np.int64)

    # Bucket size for the interior points (first/last buckets are single points).
    # See Steinarsson §4.2 / Algorithm 4.2.
    bucket_size = (n - 2) / (threshold - 2)

    sampled = np.empty(threshold, dtype=np.int64)
    sampled[0] = 0
    sampled[-1] = n - 1

    a = 0  # index of previously selected point
    for i in range(threshold - 2):
        # Average point of the *next* bucket (temporary fixed point C).
        avg_start = int(np.floor((i + 1) * bucket_size) + 1)
        avg_end = int(np.floor((i + 2) * bucket_size) + 1)
        avg_end = min(avg_end, n)
        if avg_start >= avg_end:
            avg_start = max(0, n - 1)
            avg_end = n
        avg_x = float(np.mean(x[avg_start:avg_end]))
        avg_y = float(np.mean(y[avg_start:avg_end]))

        # Current bucket range (points to rank).
        range_offs = int(np.floor(i * bucket_size) + 1)
        range_to = int(np.floor((i + 1) * bucket_size) + 1)
        range_to = min(range_to, n - 1)
        if range_offs >= range_to:
            range_offs = min(range_offs, n - 2)
            range_to = range_offs + 1

        ax = x[a]
        ay = y[a]
        # Area of triangle (A, candidate, C) — factor 0.5 omitted (ranking only).
        xs = x[range_offs:range_to]
        ys = y[range_offs:range_to]
        areas = np.abs((ax - avg_x) * (ys - ay) - (ax - xs) * (avg_y - ay))
        next_a = range_offs + int(np.argmax(areas))

        sampled[i + 1] = next_a
        a = next_a

    return sampled


def downsample_lttb(
    t: np.ndarray,
    *series: np.ndarray,
    max_points: int,
    guide: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """
    Downsample one or more y-series sharing time axis ``t`` with LTTB.

    Indices are chosen from ``guide`` (default: first series) so all channels
    stay time-aligned — important for dual-ADC overlays.
    """
    if not series:
        raise ValueError("at least one y-series is required")
    t = np.asarray(t, dtype=np.float64)
    ys = [np.asarray(s, dtype=np.float64) for s in series]
    for s in ys:
        if s.shape[0] != t.shape[0]:
            raise ValueError("all series must match t length")

    guide_y = np.asarray(guide if guide is not None else ys[0], dtype=np.float64)
    idx = lttb_indices(t, guide_y, max_points)
    return (t[idx],) + tuple(s[idx] for s in ys)
