"""Rainflow cycle counting (ASTM E1049-85).

Pure, dependency-free counting of a discrete signal into closed cycles. The
battery use is: feed a realized SoC trajectory (MWh, or any consistent unit) and
get back the (range, count) pairs — each a charge/discharge swing of some depth,
weighted 1.0 for a full closed cycle or 0.5 for a residual half cycle. A
degradation model then maps each swing's depth-of-discharge to wear (see
`degradation.RainflowDegradationModel`).

This is the standard four-point algorithm; verified against the canonical ASTM
example [-2, 1, -3, 5, -1, 3, -4, 4, -2] in tests.
"""

from __future__ import annotations

from collections import Counter
from collections import deque
from collections.abc import Sequence


def _reversals(series: Sequence[float]) -> list[float]:
    """Turning points of the series, including the first and last point.

    Consecutive duplicates are collapsed first (a flat SoC hold is not a turn),
    then interior points are kept only where the slope changes sign.
    """
    pts: list[float] = []
    for x in series:
        if not pts or x != pts[-1]:
            pts.append(x)
    if len(pts) < 2:
        return []
    rev = [pts[0]]
    for i in range(1, len(pts) - 1):
        if (pts[i] - pts[i - 1]) * (pts[i + 1] - pts[i]) < 0:
            rev.append(pts[i])
    rev.append(pts[-1])
    return rev


def extract_cycles(series: Sequence[float]) -> list[tuple[float, float]]:
    """Rainflow-count `series` into (range, count) pairs.

    `range` is the magnitude of a closed swing; `count` is 1.0 for a full cycle
    or 0.5 for a half cycle. Order is not significant — aggregate by range if a
    histogram is wanted (see `count_ranges`).
    """
    cycles: list[tuple[float, float]] = []
    points: deque[float] = deque()
    for x in _reversals(series):
        points.append(x)
        while len(points) >= 3:
            x1, x2, x3 = points[-3], points[-2], points[-1]
            y = abs(x2 - x1)  # range of the prior segment (candidate cycle)
            x = abs(x3 - x2)  # range of the newest segment
            if x < y:
                break  # cycle not yet closed; wait for more points
            if len(points) == 3:
                # Y reaches back to the start of the record → half cycle.
                cycles.append((y, 0.5))
                points.popleft()
            else:
                # Interior full cycle: discard its two inner points, keep x3.
                # Pop by position (not value — SoC values repeat).
                cycles.append((y, 1.0))
                last = points.pop()  # x3
                points.pop()         # x2
                points.pop()         # x1
                points.append(last)
    # Whatever is left forms the residual half cycles of the record.
    while len(points) > 1:
        cycles.append((abs(points[1] - points[0]), 0.5))
        points.popleft()
    return cycles


def count_ranges(series: Sequence[float]) -> dict[float, float]:
    """Aggregate `extract_cycles` into {range: total_count}."""
    hist: Counter[float] = Counter()
    for rng, count in extract_cycles(series):
        hist[rng] += count
    return dict(hist)
