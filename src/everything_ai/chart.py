# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Fast charts as SVG text, for pictures that change while you move a slider. Matplotlib takes over 0.1 second per
picture, and what sits behind a slider must be ready within 0.1 second (lesson template); this takes about a
millisecond. The look follows plots.style(): a calm horizontal grid, no box, the title on the left, tick labels in
ink-3. Every colour is a token (var(--ea-*)), so a chart follows the light and the dark theme, and a role gets its
colour from the style guide: "param" is orange, "loss" purple, and so on."""
from __future__ import annotations

import html
import itertools
import math
import re
from dataclasses import dataclass, field

_ROLES = {"data", "param", "pred", "residual", "loss", "grad", "hyper", "hidden"}
_CLIPS = itertools.count()   # every clip path on a page needs its own id


def _colour(role: str) -> str:
    if role in _ROLES:
        return f"var(--ea-{role})"
    return {"ink": "var(--ea-ink)", "ink-2": "var(--ea-ink-2)", "ink-3": "var(--ea-ink-3)", "axis": "var(--ea-axis)"}[role]


def ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """Round numbers between lo and hi, about n of them, at a step of 1, 2, 2.5 or 5 times a power of ten."""
    raw = (hi - lo) / max(n, 1)
    power = 10 ** math.floor(math.log10(raw))
    step = next(m * power for m in (1, 2, 2.5, 5, 10) if m * power >= raw)
    first = math.ceil(lo / step - 1e-9) * step
    out, k = [], 0
    while first + k * step <= hi + 1e-9 * step:
        out.append(first + k * step)
        k += 1
    return out


def _label(v: float, values: list[float]) -> str:
    """A tick label with as many decimals as the step between ticks needs, and a real minus sign."""
    step = abs(values[1] - values[0]) if len(values) > 1 else 1
    decimals = f"{step:.10f}".rstrip("0").rstrip(".").partition(".")[2]
    text = f"{v:.{len(decimals)}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    return text.replace("-", "\u2212")


def log_ticks(lo: float, hi: float) -> list[float]:
    """The powers of 10 between lo and hi, for an axis with a log scale; over a very wide range only every second,
    fourth or eighth power, so that the labels do not run into each other."""
    first, last = math.ceil(math.log10(lo) - 1e-9), math.floor(math.log10(hi) + 1e-9)
    step = next(s for s in (1, 2, 4, 8, 16, 32, 64) if (last - first) // s < 8)
    return [10.0 ** k for k in range(first, last + 1) if k % step == 0]


_SUPERSCRIPT = str.maketrans("-0123456789", "\u207b\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079")


def _log_label(v: float, powers: bool = False) -> str:
    """A tick label on a log scale: 0.01, 0.1, 1, 10, 100 and so on, written out, or as powers of ten, 10⁻⁸ and 10¹⁶."""
    k = round(math.log10(v))
    if powers:
        return "10" + str(k).translate(_SUPERSCRIPT)
    return str(10 ** k) if k >= 0 else f"{v:.{-k}f}"


def _log_labels(values: list[float]):
    """How the ticks of one log axis are written: all as powers of ten as soon as one would have more than four zeros,
    because more zeros than that nobody counts, and one axis in one way."""
    powers = any(abs(round(math.log10(v))) > 4 for v in values)
    return lambda v: _log_label(v, powers)


@dataclass
class Panel:
    """One chart: its ranges, its words, and what is drawn on it, in data units."""
    x: tuple[float, float]
    y: tuple[float, float]
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    marks: list[tuple] = field(default_factory=list)
    xticks: list[float] | None = None   # the ticks on the x-axis, when round numbers do not fit, such as the faces of a die
    yticks: list[float] | None = None   # the ticks on the y-axis, the same; [] for none, when the rows of a chart are groups
    xlog: bool = False   # a log scale on the x-axis: equal steps are equal factors, with a tick at each power of 10
    ylog: bool = False   # the same on the y-axis

    def line(self, xs, ys, role: str = "data", width: float = 2.2, dashed: bool = False):
        self.marks.append(("line", list(xs), list(ys), role, width, dashed))
        return self

    def arrow(self, x0, y0, x1, y1, role: str = "data", width: float = 2.4):
        """A line from (x0, y0) to (x1, y1) with a head at the end, such as a vector from the origin."""
        self.marks.append(("arrow", x0, y0, x1, y1, role, width))
        return self

    def text(self, x, y, words: str, role: str = "ink", anchor: str = "middle"):
        """Words centred on (x, y), such as the name of a vector next to its head; with anchor "start", they begin
        at x, as in a key."""
        self.marks.append(("text", x, y, words, role, anchor))
        return self

    def points(self, xs, ys, role: str = "data", r: float = 4, hollow: bool = False):
        """Dots; `hollow` draws them open, the second of two groups (style guide 1.7: filled and open, no colour)."""
        self.marks.append(("points", list(xs), list(ys), role, r, hollow))
        return self

    def group(self, xs, ys, k: int, r: float = 4.5):
        """Dots of group k (0 to 3) in its group colour and its own shape: circle, triangle, square, diamond. For three
        or four classes or clusters (style guide 1.7); the caption says that colour means a group here."""
        self.marks.append(("group", list(xs), list(ys), k, r))
        return self

    def rect(self, x0, y0, x1, y1, role: str = "loss"):
        """A rectangle between two corners, as a light wash with an outline in the role's colour."""
        self.marks.append(("rect", x0, y0, x1, y1, role))
        return self


def figure(panels: list[Panel], *, alt: str, width: int = 330, height: int = 250, drag: int | None = None) -> str:
    """Panels side by side, each `width` wide and an SVG of its own, in a figure that is never wider than the page. In
    a narrow window, such as beside an editor, the panels go one below the other before their text gets too small to
    read. With `drag`, the index of a panel, that panel says where its x-axis lies, so that play.Explore can turn
    dragging along it into a value of its slider."""
    svgs = []
    narrowest = math.ceil(width * 11 / 12)   # tick labels are 12 px: a panel never shrinks them below 11 px (lessjabloon)
    for i, p in enumerate(panels):
        where = ""
        if i == drag:
            L, R = _sides(p, 0, width)
            where = f' data-drag="{L:.1f} {R:.1f} {p.x[0]} {p.x[1]} {int(p.xlog)}"'
        svgs.append(f'<svg viewBox="0 0 {width} {height}" aria-hidden="true"{where} style="flex:1 1 0;min-width:min(100%,{narrowest}px);'
                    f'max-width:{width}px;height:auto;font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums">'
                    f'{_panel(p, i, 0, width, height)}</svg>')
    return (f'<figure class="ea-figure ea-plot" role="img" aria-label="{html.escape(alt)}" style="margin:0.25rem 0 0.75rem;'
            f'max-width:min(100%,{width * len(panels)}px);display:flex;flex-wrap:wrap;align-items:flex-start">{"".join(svgs)}</figure>')


def with_decimal_comma(svg: str) -> str:
    """The same chart with a decimal comma in the numbers along its axes, as Dutch writes them: 0,5 and not 0.5."""
    return re.sub(r'(<text class="ea-tick"[^>]*>)([^<]*)(</text>)', lambda m: m[1] + m[2].replace(".", ",") + m[3], svg)


def _yticks(p: Panel) -> tuple[list[float], object]:
    """The ticks of the y-axis and how each is written."""
    y0, y1 = p.y
    yt = p.yticks if p.yticks is not None else (log_ticks(y0, y1) if p.ylog else ticks(y0, y1))
    return yt, (_log_labels(yt) if p.ylog and p.yticks is None else (lambda v: _label(v, yt)))


def _sides(p: Panel, left: float, width: float) -> tuple[float, float]:
    """Where the x-axis of a panel begins and ends: room on the left for the widest tick label and the axis label."""
    yt, ylabel_of = _yticks(p)
    widest = max((len(ylabel_of(v)) for v in yt), default=1)
    return left + max(58, 32 + 7 * widest), left + width - 14


def _panel(p: Panel, index: int, left: float, width: float, height: float) -> str:
    (x0, x1), (y0, y1) = p.x, p.y
    yt, ylabel_of = _yticks(p)
    xt = p.xticks if p.xticks is not None else (log_ticks(x0, x1) if p.xlog else ticks(x0, x1))
    xlabel_of = _log_labels(xt) if p.xlog and p.xticks is None else (lambda v: _label(v, xt))
    fx = math.log10 if p.xlog else float
    fy = math.log10 if p.ylog else float
    L, R = _sides(p, left, width)
    T, B = 34, height - 46

    def X(v):
        return L + (fx(v) - fx(x0)) / (fx(x1) - fx(x0)) * (R - L)

    def Y(v):
        return B - (fy(v) - fy(y0)) / (fy(y1) - fy(y0)) * (B - T)

    def drawn(a, b):
        """Whether a point can be drawn: on a log scale, only above 0."""
        return (a > 0 or not p.xlog) and (b > 0 or not p.ylog)

    clip = f"ea-clip-{next(_CLIPS)}"
    out = [f'<clipPath id="{clip}"><rect x="{L - 6}" y="{T - 6}" width="{R - L + 12}" height="{B - T + 12}"/></clipPath>']
    for v in yt:
        out.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{R}" y2="{Y(v):.1f}" style="stroke:var(--ea-line);stroke-width:0.8"/>'
                   f'<text class="ea-tick" x="{L - 7}" y="{Y(v) + 4:.1f}" style="font-size:12px;fill:var(--ea-ink-3);text-anchor:end">{ylabel_of(v)}</text>')
    for v in xt:
        out.append(f'<line x1="{X(v):.1f}" y1="{B}" x2="{X(v):.1f}" y2="{B + 4}" style="stroke:var(--ea-axis)"/>'
                   f'<text class="ea-tick" x="{X(v):.1f}" y="{B + 18}" style="font-size:12px;fill:var(--ea-ink-3);text-anchor:middle">{xlabel_of(v)}</text>')
    out.append(f'<line x1="{L}" y1="{T - 4}" x2="{L}" y2="{B}" style="stroke:var(--ea-axis)"/>'
               f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" style="stroke:var(--ea-axis)"/>')
    if p.title:
        out.append(f'<text x="{L}" y="18" style="font-size:14px;font-weight:600;fill:var(--ea-ink)">{html.escape(p.title)}</text>')
    if p.xlabel:
        out.append(f'<text x="{(L + R) / 2:.1f}" y="{height - 8}" style="font-size:13px;fill:var(--ea-ink-2);text-anchor:middle">{html.escape(p.xlabel)}</text>')
    if p.ylabel:
        cy = (T + B) / 2
        out.append(f'<text transform="translate({left + 14},{cy:.1f}) rotate(-90)" style="font-size:13px;fill:var(--ea-ink-2);text-anchor:middle">{html.escape(p.ylabel)}</text>')
    body = []
    for mark in p.marks:
        kind = mark[0]
        if kind == "line":
            _, xs, ys, role, w, dashed = mark
            pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(xs, ys) if drawn(a, b))
            dash = ";stroke-dasharray:5 5" if dashed else ""
            body.append(f'<polyline points="{pts}" style="fill:none;stroke:{_colour(role)};stroke-width:{w};stroke-linecap:round;stroke-linejoin:round{dash}"/>')
        elif kind == "arrow":
            _, a0, b0, a1, b1, role, w = mark
            px0, py0, px1, py1 = X(a0), Y(b0), X(a1), Y(b1)
            size = math.hypot(px1 - px0, py1 - py0)
            if size > 0:
                ux, uy = (px1 - px0) / size, (py1 - py0) / size
                head = min(11.0, size * 0.6)
                bx, by = px1 - ux * head, py1 - uy * head   # the base of the head, where the line stops
                corners = f"{px1:.1f},{py1:.1f} {bx - uy * head * 0.45:.1f},{by + ux * head * 0.45:.1f} {bx + uy * head * 0.45:.1f},{by - ux * head * 0.45:.1f}"
                body.append(f'<line x1="{px0:.1f}" y1="{py0:.1f}" x2="{bx:.1f}" y2="{by:.1f}" style="stroke:{_colour(role)};stroke-width:{w};stroke-linecap:round"/>'
                            f'<polygon points="{corners}" style="fill:{_colour(role)}"/>')
        elif kind == "text":
            _, a, b, words, role, anchor = mark
            body.append(f'<text x="{X(a):.1f}" y="{Y(b) + 5:.1f}" style="font-size:14px;font-weight:600;fill:{_colour(role)};text-anchor:{anchor}">{html.escape(words)}</text>')
        elif kind == "points":
            _, xs, ys, role, r, hollow = mark
            look = (f"fill:var(--ea-page);stroke:{_colour(role)};stroke-width:1.8" if hollow
                    else f"fill:{_colour(role)};stroke:var(--ea-page);stroke-width:1.2")
            for a, b in zip(xs, ys):
                if not drawn(a, b):
                    continue
                body.append(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="{r}" style="{look}"/>')
        elif kind == "group":
            _, xs, ys, k, r = mark
            look = f'style="fill:var(--ea-group-{k + 1});stroke:var(--ea-page);stroke-width:1"'
            for a, b in zip(xs, ys):
                if not drawn(a, b):
                    continue
                cx, cy = X(a), Y(b)
                if k == 0:
                    body.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" {look}/>')
                elif k == 2:
                    s = 1.7 * r
                    body.append(f'<rect x="{cx - s / 2:.1f}" y="{cy - s / 2:.1f}" width="{s:.1f}" height="{s:.1f}" {look}/>')
                else:   # a triangle with its point up, or a diamond
                    pts = ([(0, -1.35), (-1.2, 0.8), (1.2, 0.8)] if k == 1 else [(0, -1.35), (1.35, 0), (0, 1.35), (-1.35, 0)])
                    corners = " ".join(f"{cx + u * r:.1f},{cy + v * r:.1f}" for u, v in pts)
                    body.append(f'<polygon points="{corners}" {look}/>')
        elif kind == "rect":
            _, a0, b0, a1, b1, role = mark
            xa, xb, ya, yb = sorted((X(a0), X(a1))) + sorted((Y(b0), Y(b1)))
            body.append(f'<rect x="{xa:.1f}" y="{ya:.1f}" width="{xb - xa:.1f}" height="{yb - ya:.1f}" '
                        f'style="fill:{_colour(role)};fill-opacity:0.18;stroke:{_colour(role)};stroke-width:1.2"/>')
    out.append(f'<g clip-path="url(#{clip})">{"".join(body)}</g>')
    return "".join(out)
