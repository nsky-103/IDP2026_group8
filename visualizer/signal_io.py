"""Recording access via core.fast_io (index + 1 kHz summary + on-demand windows)."""

from __future__ import annotations

import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_FS_HZ = 50_000.0
DTYPE = np.int16
LONG_WINDOW_S = 60.0  # above this, detail uses 1 kHz envelope (issue #23)
_WINDOW_CACHE_MAX = 8

_VISUALIZER_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _VISUALIZER_DIR.parent
_CORE_ROOT_CANDIDATES = [
    _REPO_ROOT,
    _REPO_ROOT / "IDP2026_group8",
    _REPO_ROOT.parent / "IDP2026_group8",
]


def _ensure_core_on_path() -> Path:
    for folder in _CORE_ROOT_CANDIDATES:
        if (folder / "core" / "fast_io.py").exists():
            root = str(folder)
            if root not in sys.path:
                sys.path.insert(0, root)
            return folder
    searched = "\n".join(f"  {p}" for p in _CORE_ROOT_CANDIDATES)
    raise FileNotFoundError(f"core package not found. Looked in:\n{searched}")


def _bucket_reduce(arr: np.ndarray, n_out: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downsample ``arr`` (n, c) to ``n_out`` bins via reduceat min/max/mean.

    Issue #24: every-n-th sample misses narrow peaks; reduceat keeps true extrema.
    """
    n = len(arr)
    if n_out >= n:
        mean = arr.astype(np.float64)
        return mean, arr.min(axis=1).astype(np.float64) if arr.ndim == 1 else arr.astype(np.float64), arr.astype(np.float64)

    edges = np.linspace(0, n, n_out + 1, dtype=np.int64)
    starts = edges[:-1]
    # drop empty buckets (can happen at the end with linspace)
    starts = starts[starts < n]
    if len(starts) == 0:
        starts = np.array([0], dtype=np.int64)

    if arr.ndim == 1:
        lo = np.minimum.reduceat(arr, starts)
        hi = np.maximum.reduceat(arr, starts)
        # mean via sum / count
        totals = np.add.reduceat(arr.astype(np.float64), starts)
        counts = np.diff(np.append(starts, n))
        mean = totals / np.maximum(counts, 1)
        return mean, lo.astype(np.float64), hi.astype(np.float64)

    lo = np.minimum.reduceat(arr, starts, axis=0)
    hi = np.maximum.reduceat(arr, starts, axis=0)
    totals = np.add.reduceat(arr.astype(np.float64), starts, axis=0)
    counts = np.diff(np.append(starts, n)).astype(np.float64)[:, None]
    mean = totals / np.maximum(counts, 1.0)
    return mean, lo.astype(np.float64), hi.astype(np.float64)


@dataclass
class Recording:
    """One Marginum CSV opened the Method B way.

    ``phase``:
      - ``preview`` — rough overview only (summary still building; index may be None)
      - ``ready`` — 1 kHz summary available; Detect and exact overview OK
    """

    path: Path
    index: Any = None  # core.fast_io.CsvIndex | None until index is ready
    summary: dict[str, np.ndarray] | None = None
    preview: dict[str, np.ndarray] | None = None
    phase: str = "preview"
    from_cache: bool = False
    estimated_n_rows: int | None = None
    clean: dict | None = None  # Part A output when precomputed (#17)
    _window_cache: OrderedDict = field(default_factory=OrderedDict, repr=False, compare=False)

    @property
    def n_rows(self) -> int:
        if self.index is not None:
            return int(self.index.n_rows)
        if self.estimated_n_rows is not None:
            return int(self.estimated_n_rows)
        if self.preview is not None and len(self.preview.get("t", [])):
            return int(float(self.preview["t"][-1]) * self.fs_hz)
        return 0

    @property
    def fs_hz(self) -> float:
        return float(DEFAULT_FS_HZ)

    @property
    def summary_fs_hz(self) -> float:
        if self.summary is not None and "fs_hz" in self.summary:
            return float(self.summary["fs_hz"])
        return 1000.0

    def duration_s(self, fs_hz: float | None = None) -> float:
        return self.n_rows / float(fs_hz or self.fs_hz)

    def overview_series(
        self, max_points: int = 4_000
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, bool]:
        """Return overview traces for plotting (reduceat bands when summary is present)."""
        if self.summary is not None:
            mean = np.asarray(self.summary["mean"])
            lo = np.asarray(self.summary["min"])
            hi = np.asarray(self.summary["max"])
            fs = self.summary_fs_hz
            n = len(mean)
            valid = np.isfinite(mean[:, 0])
            if np.any(valid):
                last = int(np.where(valid)[0][-1]) + 1
                mean_f, lo_f, hi_f = mean[:last], lo[:last], hi[:last]
                n_out = max(1, min(max_points, last))
                starts = np.linspace(0, last, n_out + 1, dtype=np.int64)[:-1]
                starts = starts[starts < last]
                edges = np.append(starts, last)
                t = 0.5 * (edges[:-1] + edges[1:]) / fs
                counts = np.diff(edges).astype(np.float64)[:, None]
                m_mean = (
                    np.add.reduceat(mean_f.astype(np.float64), starts, axis=0)
                    / np.maximum(counts, 1.0)
                )
                s1_lo = np.minimum.reduceat(lo_f[:, 0], starts)
                s1_hi = np.maximum.reduceat(hi_f[:, 0], starts)
                s2_lo = np.minimum.reduceat(lo_f[:, 1], starts)
                s2_hi = np.maximum.reduceat(hi_f[:, 1], starts)
                return (
                    t.astype(np.float64),
                    m_mean[:, 0],
                    m_mean[:, 1],
                    s1_lo.astype(np.float64),
                    s1_hi.astype(np.float64),
                    s2_lo.astype(np.float64),
                    s2_hi.astype(np.float64),
                    bool(last < n),
                )

        if self.preview is None:
            raise RuntimeError("Recording has neither summary nor preview")

        pv = self.preview
        t = np.asarray(pv["t"], dtype=np.float64)
        s1_min = np.asarray(pv["adc1_min"], dtype=np.float64)
        s1_max = np.asarray(pv["adc1_max"], dtype=np.float64)
        s2_min = np.asarray(pv["adc2_min"], dtype=np.float64)
        s2_max = np.asarray(pv["adc2_max"], dtype=np.float64)
        s1 = 0.5 * (s1_min + s1_max)
        s2 = 0.5 * (s2_min + s2_max)
        return t, s1, s2, s1_min, s1_max, s2_min, s2_max, True

    def _envelope_window(
        self, start_s: float, end_s: float, max_points: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Min/max envelope from 1 kHz summary for long detail windows (#23)."""
        if self.summary is None:
            raise RuntimeError("1 kHz summary required for long-window envelope")
        fs = self.summary_fs_hz
        lo = np.asarray(self.summary["min"])
        hi = np.asarray(self.summary["max"])
        i0 = max(0, int(start_s * fs))
        i1 = min(len(lo), max(i0 + 1, int(end_s * fs)))
        seg_lo, seg_hi = lo[i0:i1], hi[i0:i1]
        n = len(seg_lo)
        n_bins = min(max_points // 2, n)  # each bin -> two points (hi then lo zig-zag) conceptually
        n_bins = max(1, n_bins)
        starts = np.linspace(0, n, n_bins + 1, dtype=np.int64)[:-1]
        starts = starts[starts < n]
        s1_lo = np.minimum.reduceat(seg_lo[:, 0], starts)
        s1_hi = np.maximum.reduceat(seg_hi[:, 0], starts)
        s2_lo = np.minimum.reduceat(seg_lo[:, 1], starts)
        s2_hi = np.maximum.reduceat(seg_hi[:, 1], starts)
        # Interleave hi/lo so Plotly line shows the envelope thickness roughly
        edges = np.append(starts, n)
        t_centers = (i0 + 0.5 * (edges[:-1] + edges[1:])) / fs
        # Return mid of envelope as the line (detail also gets band via caller if needed);
        # for a single line that preserves peak height, plot max for positive-going fluorescence.
        s1 = s1_hi.astype(np.float64)
        s2 = s2_hi.astype(np.float64)
        # Zigzag: for each bin emit hi then lo so LTTB-less plot keeps peaks
        t = np.empty(2 * len(t_centers))
        out1 = np.empty(2 * len(t_centers))
        out2 = np.empty(2 * len(t_centers))
        t[0::2] = t_centers
        t[1::2] = t_centers
        out1[0::2] = s1_hi
        out1[1::2] = s1_lo
        out2[0::2] = s2_hi
        out2[1::2] = s2_lo
        return t, out1, out2

    def load_window(
        self,
        start_s: float,
        end_s: float,
        fs_hz: float | None = None,
        max_points: int = 20_000,
        method: str = "lttb",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Window for detail: short = CSV read_window; long = 1 kHz envelope (#23)."""
        key = (round(start_s, 4), round(end_s, 4), method, int(max_points), self.phase)
        if key in self._window_cache:
            self._window_cache.move_to_end(key)
            return self._window_cache[key]

        width = float(end_s) - float(start_s)
        if width > LONG_WINDOW_S and self.summary is not None:
            result = self._envelope_window(start_s, end_s, max_points)
        else:
            if self.index is None:
                raise RuntimeError("Row index not ready yet for a short detail window")
            _ensure_core_on_path()
            from core import fast_io
            from lttb import downsample_lttb

            fs = float(fs_hz or self.fs_hz)
            t, adc1, adc2 = fast_io.read_window(self.index, start_s, end_s, fs_hz=fs)
            s1 = adc1.astype(np.float64)
            s2 = adc2.astype(np.float64)
            n = len(t)
            if n <= max_points:
                result = (t, s1, s2)
            elif method == "stride":
                stride = max(1, n // max_points)
                idx = np.arange(0, n, stride)
                result = (t[idx], s1[idx], s2[idx])
            elif method == "lttb":
                result = downsample_lttb(t, s1, s2, max_points=max_points, guide=s2)
            else:
                raise ValueError(f"Unknown downsample method: {method!r}")

        self._window_cache[key] = result
        while len(self._window_cache) > _WINDOW_CACHE_MAX:
            self._window_cache.popitem(last=False)
        return result

    def binned_peak_to_peak(self, bin_s: float, channel: int = 1) -> np.ndarray:
        """Peak-to-peak per ``bin_s`` from the 1 kHz summary (max−min envelope)."""
        if self.summary is None:
            raise RuntimeError("1 kHz summary required for heatmap (wait until ready)")
        fs = self.summary_fs_hz
        lo = np.asarray(self.summary["min"][:, channel], dtype=np.float64)
        hi = np.asarray(self.summary["max"][:, channel], dtype=np.float64)
        bin_n = max(1, int(round(bin_s * fs)))
        n = len(lo)
        n_bins = -(-n // bin_n)
        out = np.empty(n_bins, dtype=np.float64)
        for i in range(n_bins):
            a = i * bin_n
            b = min(n, a + bin_n)
            out[i] = float(hi[a:b].max()) - float(lo[a:b].min())
        return out


def duration_s(n_samples: int, fs_hz: float = DEFAULT_FS_HZ) -> float:
    return n_samples / fs_hz
