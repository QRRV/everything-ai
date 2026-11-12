# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""A run you can watch (decision 1.11): while a demo trains, its chart grows step by step in the cell, in place of a
line of text. A lesson makes one with `L.live(...)` before the run, and its progress callback gives it every step;
the chart is drawn again at most every half second, so a fast run does not spend its time drawing. The line of the
run is in the colour of its role: loss for a loss, ink for a reward or an accuracy (style guide, chapter 1).

Next to the chart you see what the model makes at that moment (decision 1.17): `watch(now)` gives the chart a
function that draws it, such as the class maps of a few test scenes or a sentence the model writes, and the feature
maps of one layer of the learner's network. It is called only when the chart is drawn, so a model is looked at at
most every half second, or less often with `look_every`; a lesson looks at its model in eval mode and without a graph, and puts it back in train mode,
so that looking never changes the run."""
from __future__ import annotations

import html
import math
import re
import time

import marimo as mo

from everything_ai import chart
from everything_ai.check import Timeout


CHART_WIDTH = 360   # px of the chart of a run


class Live:
    """The chart of a run while it trains: the value of every step so far, and its mean over the last `window` steps
    as the thicker line, against an x-axis of all `steps` the run will take, so that you see how far it is. `labels`
    has title, x, y and alt; `low` and `high` fix the y-axis, else it follows the values. `line(text)` shows a line
    of progress under the chart, or alone before the first step; `reference(points, label)` adds another run to hold
    this one against. `key` says under the chart what each line is, as (role, dashed, words), for lines whose names
    would sit in each other's way, such as a run and its test loss that end close together. `every` is how often the
    chart is drawn at most, in seconds; `look_every` how often `watch` looks at the model, by default as often: a
    look that takes long, such as a model that writes a sentence, is spread out more than the chart."""

    def __init__(self, labels: dict, steps: int, role: str = "loss", lang: str = "en", low: float | None = None,
                 high: float | None = None, window: int | None = None, every: float = 0.5, xlog: bool = False,
                 key: list | None = None, look_every: float | None = None):
        self.labels, self.steps, self.role, self.lang = labels, max(int(steps), 1), role, lang
        self.key = key or []   # under the chart, what each line is: (role, dashed, words)
        self.low, self.high, self.every, self.xlog = low, high, every, xlog
        self.look_every = every if look_every is None else look_every
        self.window = window or max(1, self.steps // 50)
        self.xs: list = []
        self.ys: list = []
        self.others: list = []
        self.above: list = []   # what shows above the chart, such as the finished chart of an earlier part of the demo
        self._sides: dict = {}
        self.text = ""
        self._last = 0.0
        self._now = None
        self._seen = None   # (when, parts) of the last look

    def watch(self, now) -> None:
        """What the model makes at this moment, drawn with the chart each time the chart is drawn: `now()` gives HTML,
        or a list of HTML parts, the first beside the chart (under it on a narrow screen) and the others under them.
        Set it before the step it belongs to; a later call replaces it. A picture that fails is left out: the chart is
        the course's own, and never stops the learner's run."""
        self._now, self._seen = now, None

    def looked(self) -> list:
        """What `watch` draws now, as a list of HTML parts; empty without a watch, or when drawing it fails. The model
        is looked at again only `look_every` seconds after the last look ended, also when the line of progress changes
        faster: a slow look, such as a model writing a story, then takes a small part of the run however slow it is."""
        if self._now is None:
            return []
        at = time.perf_counter()
        if self._seen is not None and at - self._seen[0] < self.look_every:
            return self._seen[1]
        try:
            out = self._now()
        except Timeout:   # the time limit of the run is not the picture's error
            raise
        except Exception:
            out = []
        parts = [p for p in ([out] if isinstance(out, str) else list(out or [])) if p]
        self._seen = (time.perf_counter(), parts)
        return parts

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

    def reference(self, points: list, label: str, role: str = "ink-2", replace: bool = False) -> None:
        """Another run to hold this one against, such as the course's or an earlier run of yours, as (x, value): drawn
        thin and dashed, with its name at its end, on the side away from this run; its values count for the y-axis
        too. Its `role` is ink-2 for a line the run is measured against, such as ln 2; for another run of the same
        thing, the role of this run (style guide 1.3: the same role keeps its colour, the line style tells them apart).
        An empty `label` draws no name, for several runs that the text under the chart names together; a run with the
        name of one already there takes its place, such as one that grows along with this run; with `replace`, also
        a run without a name takes the place of the one in its role, such as the test loss of this run after every
        epoch."""
        points = [(x, y) for x, y in points if math.isfinite(y)]
        if label or replace:
            self.others = [o for o in self.others if o[1] != label or (replace and o[2] != role)]
        if points:
            self.others.append((points, label, role))

    def line(self, text: str) -> None:
        """Only the line of progress changes, such as while the run measures before or after it trains."""
        self.text = text
        self.show()

    def stopped(self, note: str):
        """What a cell shows when its run stops early and the cell gives its output itself: the chart so far, when the
        run took a step, with what the model made then beside it (watch), and `note` under it."""
        if not self.xs:
            return mo.vstack([*self.above, mo.md(note)])
        now = self.looked()   # what the model made when it stopped, beside the chart as during the run
        return mo.vstack([*self.above, mo.Html(beside((self.drawn(), CHART_WIDTH), now[0]) if now else self.drawn()),
                          *[mo.Html(p) for p in now[1:]], mo.md(note)])

    def drawn(self) -> str:
        """The chart with its key under it."""
        return self.picture() + (key_html(self.key) if self.key else "")

    def parts(self) -> list:
        """What the cell shows now: what stays above, the chart with what the model makes now beside it and under
        it, and the line of progress."""
        parts = [*self.above]
        now = self.looked()
        if self.xs or self.others:
            parts.append(mo.Html(beside((self.drawn(), CHART_WIDTH), now[0]) if now else self.drawn()))
            now = now[1:]
        parts += [mo.Html(p) for p in now]
        if self.text:
            parts.append(mo.md(self.text))
        return parts

    def show(self) -> None:
        parts = self.parts()
        mo.output.replace(mo.vstack(parts) if parts else mo.md(""))

    def means(self) -> list:
        k, out = self.window, []
        for i in range(len(self.ys)):
            part = self.ys[max(0, i - k + 1): i + 1]
            out.append(sum(part) / len(part))
        return out

    def picture(self, at: float | None = None) -> str:
        """The run so far as SVG; also for a test, or for the chart that stays after the run. With `at`, a line at that
        step, for a rewind through the run."""
        mean = self.means()
        every = self.ys + [v for points, _, _ in self.others for _, v in points]
        low = self.low if self.low is not None else min(every)
        high = self.high if self.high is not None else max(every)
        if high - low < 1e-9:
            low, high = low - 0.5, high + 0.5
        pad = 0.06 * (high - low)
        y = (low - pad if self.low is None else low, high + pad if self.high is None else high)
        p = chart.Panel((1 if self.xlog else 0, self.steps), y, title=self.labels["title"], xlabel=self.labels["x"],
                        ylabel=self.labels["y"], xlog=self.xlog)
        for points, label, role in self.others:                         # the runs to hold this one against
            p.line([x for x, _ in points], [v for _, v in points], role, 1.4, dashed=True)
            if not label:
                continue
            above = self._sides.get(label, bool(self.xs) and mean[-1] < value_at(points, self.xs[-1]))   # the side away
            if self.xs and self.xs[-1] >= max(1, self.steps // 10):     # from this run, where it is now; fixed after a
                self._sides.setdefault(label, above)                        # tenth of it, so the name does not jump about
            p.text(*name_at(points, label, above, self.steps, y, self.xlog, list(zip(self.xs, mean))), label, role, "end")
        if len(self.xs) > 1:
            p.line(self.xs, self.ys, "axis", 0.8)                      # every step, thin and light grey
            p.line(self.xs, mean, self.role, 2.4)                       # its mean over the last steps, in its role
        if self.xs and self.steps <= 50 and all(float(x).is_integer() for x in self.xs):
            p.points(self.xs, mean, self.role, r=2.5)                   # a run of few steps: a dot at every step
        if self.xs:
            p.points([self.xs[-1]], [mean[-1]], self.role, r=3.5)       # where the run is now
        if at is not None:   # where a rewind is: up to the highest line, under the names above it
            p.line([at, at], [y[0], min(y[1], max(every)) if every else y[1]], "ink-2", 1.2)
        svg = chart.figure([p], alt=self.labels["alt"], width=CHART_WIDTH, height=240)
        return chart.with_decimal_comma(svg) if self.lang == "nl" else svg


def beside(*parts) -> str:
    """Pictures side by side that wrap one under the other where the screen is too narrow. A part is HTML at its own
    width, or (HTML, width in px) for a chart, which fills the width it gets: it gets that width, or less on a phone."""
    def cell(part) -> str:
        if isinstance(part, tuple):
            html, width = part
            return f'<div style="flex:0 1 {width}px;width:{width}px;min-width:0;max-width:100%">{html}</div>'
        return f'<div style="flex:0 1 auto;min-width:0;max-width:100%">{part}</div>'
    cells = "".join(cell(p) for p in parts)
    return f'<div class="ea-beside" style="display:flex;flex-wrap:wrap;gap:0.75rem 1.5rem;align-items:flex-start">{cells}</div>'


def rewound(curves: str, parts: list, width: int | None = None) -> str:
    """A rewind through a run at one epoch (decision 1.17): `curves`, a chart of the run with a line at that epoch,
    beside the first of `parts`, what the model made then, and the others under them, such as the layer the demo
    watched. A chart in a row needs its width: it gets `width`, by default the width chart.figure drew it at, so its
    words keep their size, or less on a phone."""
    if not parts:
        return curves
    if width is None:
        drawn = re.search(r"max-width:min\(100%,(\d+)px\)", curves)
        width = int(drawn.group(1)) if drawn else CHART_WIDTH
    return beside((curves, width), parts[0]) + "".join(parts[1:])


def key_html(entries: list) -> str:
    """Under a chart, what each line is: a short piece of the line in its role, solid or dashed, and its words. As the
    chart draws them: a run solid and 2.4 px wide, another run dashed and 1.4 px wide."""
    items = []
    for role, dashed, words in entries:
        dash = ";stroke-dasharray:5 5" if dashed else ""
        width = 1.4 if dashed else 2.4
        sample = (f'<svg width="26" height="10" viewBox="0 0 26 10" aria-hidden="true" '   # its size in its style: a rewind
                  f'style="flex:none;display:inline-block;width:26px;height:10px">'   # sizes every svg in it to its column
                  f'<line x1="2" y1="5" x2="24" y2="5" style="stroke:{chart._colour(role)};stroke-width:{width};'
                  f'stroke-linecap:round{dash}"/></svg>')
        items.append(f'<span style="display:inline-flex;align-items:center;gap:0.35rem">{sample}{html.escape(words)}</span>')
    return (f'<div class="ea-linekey" style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2);'
            f'display:flex;flex-wrap:wrap;gap:0.2rem 1rem;margin:0.2rem 0 0">{"".join(items)}</div>')


def value_at(points: list, x: float) -> float:
    """The value of a line through these points at x, straight between two of them; before the first point or after
    the last, the value there. So that a run is held against another where it is now, not where the other ends."""
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[0][1] if x < points[0][0] else points[-1][1]


def name_at(points: list, label: str, above: bool, steps: float, y: tuple, xlog: bool = False, avoid=()) -> tuple:
    """Where the name of another run goes, as (x, y) for words that end at its last point: above or below every point
    of it under the name (about 0.03 of the x-axis a letter), so that its own line does not cross it, and inside `y`.
    The points of `avoid`, such as those of the run itself, count too where they are under the name.
    Below a line that ends at the bottom of the chart there is no room, so its name then goes above it; above a line
    that ends at the top, below it. A line that ends near the start of the x-axis has its name just after it, so that
    the name does not begin left of the axis. When neither side has room beyond both lines, such as for a line near the
    top over a run near the bottom, the name goes just beside its own line, on a side clear of the run's points."""
    h = y[1] - y[0]
    width = 0.03 * len(label)   # of the x-axis; on a log scale a share of its decades
    end = max(points[-1][0], steps ** width if xlog else width * steps)   # the x-axis begins at 1 on a log scale, else at 0
    left = end / steps ** width if xlog else end - width * steps
    under = [v for x, v in points if x >= left] + [v for x, v in avoid if left <= x <= end]
    up, down = max(under) + 0.07 * h, min(under) - 0.1 * h
    if not above and down < y[0] + 0.05 * h and up <= y[1] - 0.05 * h:
        above = True
    elif above and up > y[1] - 0.05 * h and down >= y[0] + 0.05 * h:
        above = False
    at = up if above else down
    if not y[0] + 0.05 * h <= at <= y[1] - 0.05 * h:   # no room beyond both: beside its own line, clear of the run
        own = [v for x, v in points if x >= left]
        sides = (max(own) + 0.07 * h, min(own) - 0.1 * h)
        for spot in (sides if above else sides[::-1]):
            if y[0] + 0.05 * h <= spot <= y[1] - 0.05 * h and all(abs(v - spot) > 0.1 * h for x, v in avoid if left <= x <= end):
                at = spot
                break
    return end, min(max(at, y[0] + 0.05 * h), y[1] - 0.05 * h)


def _measuring(model):
    """The training loss of every batch that `model` trains on, as follow and walked hear it: the cross-entropy of
    what the model gives on a batch in train mode against the labels of that batch, put in labels[0] by the loader
    before the batch goes out. A forward hook hears it, without a graph and without drawing a random number, so the
    run stays the same; a batch it cannot measure, such as one whose labels do not fit the output, adds nothing.
    Gives (labels, losses, unhook): losses gets one number per batch, and unhook() takes the hook off. Without a
    model (no nn.Module), no hook and no losses."""
    import torch   # here, not at the top: lessons without torch load this module too
    import torch.nn.functional as F

    labels, losses, hooked = [None], [], []

    def measured(module, args, out):
        if not module.training:
            return
        yb, labels[0] = labels[0], None   # one loss per batch, also when the model is called again on it
        if yb is None or not torch.is_tensor(out) or out.dim() < 2 or out.shape[:1] != yb.shape[:1]:
            return
        try:
            with torch.no_grad():
                loss = F.cross_entropy(out.detach().reshape(-1, out.shape[-1]).float(), yb.to(out.device).reshape(-1))
            if torch.isfinite(loss):
                losses.append(float(loss))
        except Timeout:   # the time limit of the run, also when it falls while the hook measures
            raise
        except Exception:   # labels that do not fit the model's output: the chart then only shows how far it is
            pass

    if isinstance(model, torch.nn.Module):
        hooked.append(model.register_forward_hook(measured))

    def unhook():
        for h in hooked:
            h.remove()
        hooked.clear()
    return labels, losses, unhook


def follow(trainer, show, name: str, every: int = 50) -> None:
    """Let `show` hear a run of the learner's Trainer of nn-trainer while it fits, without changing their code: after
    every epoch its validation loss ({"run": name, "epoch", "epochs", "val_loss"}), and every `every` batches how far
    the epoch is, with the mean training loss of those batches ({"run", "epoch", "epochs", "batch", "batches",
    "loss"}). This wraps self.train_loader, which their fit walks once an epoch, and self.evaluate, on this one
    trainer. An epoch counts from its first batch, and its validation loss is the first evaluate after it: one before
    the first epoch, or a second in the same epoch, is not heard, and a fit that measures without self.evaluate only
    shows how far each epoch is.

    The training loss is the cross-entropy of what self.model gives on a batch in train mode, as train_epoch of
    nn-trainer measures it, against the labels of that batch: a hook on the model hears it, without a graph and
    without drawing a random number, so the run stays the same. A report leaves "loss" out when the model gave nothing
    it can measure, such as a trainer without self.model; the hook goes once the last epoch is measured, or when
    self.fit stops sooner."""
    import torch   # here, not at the top: lessons without torch load this module too

    evaluate, loader, epoch, heard = trainer.evaluate, trainer.train_loader, [0], [0]
    labels, losses, unhooked = _measuring(getattr(trainer, "model", None))

    def evaluated(*args, **kwargs):
        loss = evaluate(*args, **kwargs)
        if heard[0] < epoch[0]:
            heard[0] = epoch[0]
            if epoch[0] >= trainer.epochs:   # the last epoch: the model trains no more under this trainer
                unhooked()
            show({"run": name, "epoch": epoch[0], "epochs": trainer.epochs, "val_loss": float(loss)})
        return loss

    class Counted:   # the loader as it is, also for len() and its other attributes, telling how far an epoch is
        def __len__(self):
            return len(loader)

        def __getattr__(self, key):
            return getattr(loader, key)

        def __iter__(self):
            epoch[0] += 1
            losses.clear()
            for i, batch in enumerate(loader, 1):
                labels[0] = batch[1] if isinstance(batch, (tuple, list)) and len(batch) == 2 and torch.is_tensor(batch[1]) else None
                yield batch   # the fit trains on it before it asks for the next one
                labels[0] = None
                if i % every == 0:
                    report = {"run": name, "epoch": epoch[0], "epochs": trainer.epochs, "batch": i, "batches": len(loader)}
                    if losses:
                        report["loss"] = sum(losses) / len(losses)
                        losses.clear()
                    show(report)

    trainer.evaluate, trainer.train_loader = evaluated, Counted()
    fit = getattr(trainer, "fit", None)
    if callable(fit):
        def fitted(*args, **kwargs):   # also when the fit stops sooner, at the time limit or on an error of theirs
            try:
                return fit(*args, **kwargs)
            finally:
                unhooked()
        trainer.fit = fitted


def walked(loader, model, show, name: str, epochs: int, every: int = 10):
    """The loader as it is, for a fit of the learner's own that walks it once an epoch, such as fit of
    torch-mlp-digits (decision 1.17): every `every` batches `show` hears how far the epoch is, with the mean training
    loss of those batches ({"run": name, "epoch", "epochs", "batch", "batches", "loss"}), and after the last batch of
    every epoch, before the fit goes on, {"run", "epoch", "epochs", "done": True}, so that a demo can look at the model
    as that epoch left it. An epoch counts from its first batch.

    The training loss is measured as follow measures it (_measuring). The hook goes after the last epoch; `.unhook()`
    takes it off sooner, such as when the fit stops with an error."""
    import torch   # here, not at the top: lessons without torch load this module too

    epoch = [0]
    labels, losses, unhooked = _measuring(model)

    class Walked:   # the loader as it is, also for len() and its other attributes, telling how far an epoch is
        def __len__(self):
            return len(loader)

        def __getattr__(self, key):
            return getattr(loader, key)

        def unhook(self):
            unhooked()

        def __iter__(self):
            epoch[0] += 1
            losses.clear()
            for i, batch in enumerate(loader, 1):
                labels[0] = batch[1] if isinstance(batch, (tuple, list)) and len(batch) == 2 and torch.is_tensor(batch[1]) else None
                yield batch   # the fit trains on it before it asks for the next one
                labels[0] = None
                if i % every == 0:
                    report = {"run": name, "epoch": epoch[0], "epochs": epochs, "batch": i, "batches": len(loader)}
                    if losses:
                        report["loss"] = sum(losses) / len(losses)
                        losses.clear()
                    show(report)
            if epoch[0] >= epochs:   # the last epoch: the model trains no more on this loader
                self.unhook()
            show({"run": name, "epoch": epoch[0], "epochs": epochs, "done": True})

    return Walked()
