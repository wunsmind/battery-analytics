"""Joint price scenarios via bootstrapped forecast-error day-shapes.

Fixes the marginal-quantile flaw: instead of gluing independent per-hour quantiles
(which ignores how hours move together), we take the central forecast and add
*whole-day forecast-error shapes* observed historically. Each sampled error-day
carries the real temporal structure (a missed evening peak, a shifted cheap
window), so the scenarios differ in SHAPE — the risk a battery actually faces —
not just in level.

Errors should come from an out-of-sample (validation) period so their spread
reflects true forecast uncertainty, not in-sample optimism.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_residual_blocks(actual: pd.Series, pred: pd.Series, steps: int = 24) -> np.ndarray:
    """Reshape residuals (actual − pred) into non-overlapping day blocks.

    Returns shape (n_days, steps); each row is one day's error shape.
    """
    a, p = actual.align(pred, join="inner")
    res = (a - p).to_numpy(dtype=float)
    n = (len(res) // steps) * steps
    if n == 0:
        return np.empty((0, steps), dtype=float)
    return res[:n].reshape(-1, steps)


def sample_scenarios(
    point_window: np.ndarray, blocks: np.ndarray, n: int, rng: np.random.Generator
) -> list[list[float]]:
    """n price paths = point forecast for the window + sampled residual day-shapes."""
    pw = np.asarray(point_window, dtype=float)
    L = len(pw)
    if blocks.shape[0] == 0:
        return [pw.tolist()]
    idx = rng.integers(0, blocks.shape[0], size=n)
    out = []
    for j in idx:
        b = blocks[j]
        b = b[:L] if len(b) >= L else np.pad(b, (0, L - len(b)))
        out.append((pw + b).tolist())
    return out
