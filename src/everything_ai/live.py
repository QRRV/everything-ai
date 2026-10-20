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
    of progress under the chart, or alone before the first step."""

    def __init__(self, labels: dict, steps: int, role: str = "loss", lang: str = "en", low: float | None = None,
                 high: float | None = None, window: int | None = None, every: float = 0.5):
        self.labels, self.steps, self.role, self.lang = labels, max(int(steps), 1), role, lang
        self.low, self.high, self.every = low, high, every
        self.window = window or max(1, self.steps // 50)
        self.xs: list = []
        self.ys: list = []
        self.text = ""
        self._last = 0.0

    def add(self, x: float, y: float, text: str = "") -> None:
        """One step of the run: its x and its value; `text` replaces the line under the chart."""
        if not math.isfinite(y):
            return
        self.xs.append(x)
        self.ys.append(y)
        if text:
            self.text = text
        now = time.perf_counter()
        if now - self._last >= self.every or x >= self.steps:
            self._last = now
            self.show()

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

    def line(self, text: str) -> None:
        """Only the line of progress changes, such as while the run measures before or after it trains."""
        self.text = text
        self.show()

    def show(self) -> None:
        parts = [mo.Html(self.picture())] if self.xs else []
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
        low = self.low if self.low is not None else min(self.ys)
        high = self.high if self.high is not None else max(self.ys)
        if high - low < 1e-9:
            low, high = low - 0.5, high + 0.5
        pad = 0.06 * (high - low)
        y = (low - pad if self.low is None else low, high + pad if self.high is None else high)
        p = chart.Panel((0, self.steps), y, title=self.labels["title"], xlabel=self.labels["x"], ylabel=self.labels["y"])
        if len(self.xs) > 1:
            p.line(self.xs, self.ys, "axis", 0.8)                      # every step, thin and light grey
            p.line(self.xs, mean, self.role, 2.4)                       # its mean over the last steps, in its role
        p.points([self.xs[-1]], [mean[-1]], self.role, r=3.5)           # where the run is now
        svg = chart.figure([p], alt=self.labels["alt"], width=360, height=240)
        return chart.with_decimal_comma(svg) if self.lang == "nl" else svg
