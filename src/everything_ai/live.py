# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""A run you can watch (decision 1.11): while a demo trains, its chart grows step by step in the cell, in place of a
line of text. A lesson makes one with `L.live(...)` before the run, and its progress callback gives it every step;
the chart is drawn again at most every half second, so a fast run does not spend its time drawing. The line of the
run is in the colour of its role: loss for a loss, ink for a reward or an accuracy (style guide, chapter 1)."""
from __future__ import annotations

import math
import time

import marimo as mo

from everything_ai import chart


class Live:
    """The chart of a run while it trains: the value of every step so far, and its mean over the last `window` steps
    as the thicker line, against an x-axis of all `steps` the run will take, so that you see how far it is. `labels`
    has title, x, y and alt; `low` and `high` fix the y-axis, else it follows the values. `line(text)` shows a line
    of progress under the chart, or alone before the first step; `reference(points, label)` adds another run to hold
    this one against."""

    def __init__(self, labels: dict, steps: int, role: str = "loss", lang: str = "en", low: float | None = None,
                 high: float | None = None, window: int | None = None, every: float = 0.5):
        self.labels, self.steps, self.role, self.lang = labels, max(int(steps), 1), role, lang
        self.low, self.high, self.every = low, high, every
        self.window = window or max(1, self.steps // 50)
        self.xs: list = []
        self.ys: list = []
        self.others: list = []
        self._sides: dict = {}
        self.text = ""
        self._last = 0.0

    def add(self, x: float, y: float, text: str = "") -> None:
        """One step of the run: its x and its value; `text` replaces the line under the chart."""
        self.extend([(x, y)], text)

    def extend(self, points: list, text: str = "") -> None:
        """Several steps at once, as (x, value), such as all the steps since the run last reported; drawn once."""
        for x, y in points:
            if math.isfinite(y):
                self.xs.append(x)
                self.ys.append(y)
        if text:
            self.text = text
        now = time.perf_counter()
        if now - self._last >= self.every or (self.xs and self.xs[-1] >= self.steps):
            self._last = now
            self.show()

    def reference(self, points: list, label: str) -> None:
        """Another run to hold this one against, such as the course's or an earlier run of yours, as (x, value): drawn
        thin and dashed in ink-2, with its name at its end, on the side away from this run; its values count for the
        y-axis too."""
        points = [(x, y) for x, y in points if math.isfinite(y)]
        if points:
            self.others.append((points, label))

    def line(self, text: str) -> None:
        """Only the line of progress changes, such as while the run measures before or after it trains."""
        self.text = text
        self.show()

    def show(self) -> None:
        parts = [mo.Html(self.picture())] if self.xs or self.others else []
        if self.text:
            parts.append(mo.md(self.text))
        mo.output.replace(mo.vstack(parts) if parts else mo.md(""))

    def means(self) -> list:
        k, out = self.window, []
        for i in range(len(self.ys)):
            part = self.ys[max(0, i - k + 1): i + 1]
            out.append(sum(part) / len(part))
        return out

    def picture(self) -> str:
        """The run so far as SVG; also for a test, or for the chart that stays after the run."""
        mean = self.means()
        every = self.ys + [v for points, _ in self.others for _, v in points]
        low = self.low if self.low is not None else min(every)
        high = self.high if self.high is not None else max(every)
        if high - low < 1e-9:
            low, high = low - 0.5, high + 0.5
        pad = 0.06 * (high - low)
        y = (low - pad if self.low is None else low, high + pad if self.high is None else high)
        p = chart.Panel((0, self.steps), y, title=self.labels["title"], xlabel=self.labels["x"], ylabel=self.labels["y"])
        for points, label in self.others:                               # the runs to hold this one against
            p.line([x for x, _ in points], [v for _, v in points], "ink-2", 1.4, dashed=True)
            above = self._sides.get(label, bool(self.xs) and mean[-1] < points[-1][1])   # the side away from this run,
            if len(self.xs) >= max(1, self.steps // 10):                 # fixed after a tenth of the run, so that the
                self._sides.setdefault(label, above)                        # name does not jump while the lines are close
            p.text(*name_at(points, label, above, self.steps, y), label, "ink-2", "end")
        if len(self.xs) > 1:
            p.line(self.xs, self.ys, "axis", 0.8)                      # every step, thin and light grey
            p.line(self.xs, mean, self.role, 2.4)                       # its mean over the last steps, in its role
        if self.xs:
            p.points([self.xs[-1]], [mean[-1]], self.role, r=3.5)       # where the run is now
        svg = chart.figure([p], alt=self.labels["alt"], width=360, height=240)
        return chart.with_decimal_comma(svg) if self.lang == "nl" else svg


def name_at(points: list, label: str, above: bool, steps: float, y: tuple) -> tuple:
    """Where the name of another run goes, as (x, y) for words that end at its last point: above or below every point
    of it under the name (about 0.03 of the x-axis a letter), so that its own line does not cross it, and inside `y`."""
    h = y[1] - y[0]
    under = [v for x, v in points if x >= points[-1][0] - 0.03 * len(label) * steps]
    at = max(under) + 0.07 * h if above else min(under) - 0.1 * h
    return points[-1][0], min(max(at, y[0] + 0.05 * h), y[1] - 0.05 * h)
