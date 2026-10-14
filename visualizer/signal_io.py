"""Recording access via core.fast_io (index + 1 kHz summary + on-demand windows)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_FS_HZ = 50_000.0
DTYPE = np.int16

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


@dataclass
class Recording:
    """One Marginum CSV opened the Method B way.

    ``phase``:
      - ``preview`` — rough overview only (summary still building)
      - ``ready`` — 1 kHz summary available; Detect and exact overview OK
    """

    path: Path
    index: Any  # core.fast_io.CsvIndex
    summary: dict[str, np.ndarray] | None = None
    preview: dict[str, np.ndarray] | None = None
    phase: str = "preview"
    from_cache: bool = False

    @property
    def n_rows(self) -> int:
        return int(self.index.n_rows)

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
        """Return overview traces for plotting.

        Returns
        -------
        t, s1, s2, s1_min, s1_max, s2_min, s2_max, approximate
            Mean (or mid) lines plus optional envelopes. ``approximate`` is True
            when only the rough preview is available.
        """
        if self.summary is not None:
            mean = np.asarray(self.summary["mean"])
            lo = np.asarray(self.summary["min"])
            hi = np.asarray(self.summary["max"])
            fs = self.summary_fs_hz
            n = len(mean)
            stride = max(1, n // max_points)
            t = (np.arange(0, n, stride, dtype=np.float64)) / fs
            s1 = mean[::stride, 0].astype(np.float64)
            s2 = mean[::stride, 1].astype(np.float64)
            return (
                t,
                s1,
                s2,
                lo[::stride, 0].astype(np.float64),
                hi[::stride, 0].astype(np.float64),
                lo[::stride, 1].astype(np.float64),
                hi[::stride, 1].astype(np.float64),
                False,
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

    def load_window(
        self,
        start_s: float,
        end_s: float,
        fs_hz: float | None = None,
        max_points: int = 20_000,
        method: str = "lttb",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Exact samples for [start_s, end_s) via ``read_window``, then downsample."""
        _ensure_core_on_path()
        from core import fast_io
        from lttb import downsample_lttb

        fs = float(fs_hz or self.fs_hz)
        t, adc1, adc2 = fast_io.read_window(self.index, start_s, end_s, fs_hz=fs)
        s1 = adc1.astype(np.float64)
        s2 = adc2.astype(np.float64)
        n = len(t)
        if n <= max_points:
            return t, s1, s2

        if method == "stride":
            stride = max(1, n // max_points)
            idx = np.arange(0, n, stride)
            return t[idx], s1[idx], s2[idx]

        if method == "lttb":
            return downsample_lttb(t, s1, s2, max_points=max_points, guide=s2)

        raise ValueError(f"Unknown downsample method: {method!r}")

    def binned_peak_to_peak(self, bin_s: float, channel: int = 1) -> np.ndarray:
        """Peak-to-peak per ``bin_s`` from the 1 kHz summary (max−min envelope).

        Approximate vs full-rate NPY bins, but cheap and matches Method B load.
        """
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
