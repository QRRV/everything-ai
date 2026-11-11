# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Pictures of what the learner made, drawn from their own code each time the check runs, so they change while the
learner types: everything a learner makes is shown, live (besluiten, 2026-11-11).

An exercise names one as its `picture`, such as `picture=yours.steps()`. Each function here gives back such a
picture: a function of a check.Seen that gives HTML text, or "" when there is nothing to show yet. The picture
stands under the check message. Every colour is a token (var(--ea-*)), so it follows the light and dark theme, and
every word comes from course/phrases.yaml or from the lesson, in its language.

    steps()     what the code does line by line: the names after each line and what it shows (py-for, py-if)
    names()     the names the code made, each value in boxes: a text letter by letter, a list item by item
    curve()     a function of one number next to the right one, as two lines on one chart
    objects()   the objects a class makes, each a card with its attributes
    bars()      a dictionary of counts as bars, with the right count over each bar
    arrays()    arrays as grids of numbers: what the code starts from, what it makes, and the right one if it differs
    tangent()   a slope the code measured, as a line through a point of the curve
    vectors()   pairs of numbers as arrows in the plane, before and after the function
    descent()   the steps of the learner's gradient descent, on the loss curve or as the loss after each step
    series()    lists of numbers the code makes, such as a loss for each degree, as lines
    table()     a pandas table the code makes, and the right one beside it when it differs
"""
from __future__ import annotations

import copy
import html
import inspect
import io
import contextlib
import math
import reprlib
import sys
import types
import warnings
from dataclasses import dataclass, field

from everything_ai import chart, check as checking, course

Text = dict   # {"en": ..., "nl": ...}

_MONO = "font-family:var(--ea-font-code);font-variant-ligatures:none;font-feature-settings:&quot;calt&quot; 0,&quot;liga&quot; 0"
_MAX_RECORDED = 3000   # steps kept with their names; a longer run is only counted


def _word(key: str, lang: str, **values) -> str:
    return course.phrase(key, lang, **values)


def _learners(value) -> bool:
    """An object of a class the learner wrote, in the box or in a cell of the lesson."""
    return type(value).__module__ in ("__learner__", "__main__") and hasattr(value, "__dict__") and not isinstance(value, type)


class _Short(reprlib.Repr):
    def repr1(self, value, level):
        if value is Ellipsis:   # `...`, the gap the learner still fills in, is written the way they typed it
            return "..."
        kind = type(value).__name__
        if kind == "ndarray" and value.size > 8:   # a big array by its shape: its numbers do not fit on a line anyway
            return f"array{tuple(int(n) for n in value.shape)}".replace(",)", ")")
        if kind == "DataFrame" and type(value).__module__.startswith("pandas"):
            return f"DataFrame({value.shape[0]}×{value.shape[1]})"
        if kind == "Series" and type(value).__module__.startswith("pandas"):
            return f"Series({len(value)})"
        if kind == "Tensor" and type(value).__module__.startswith("torch") and value.numel() > 8:
            return f"tensor{tuple(int(n) for n in value.shape)}".replace(",)", ")")
        if _learners(value) and type(value).__repr__ is object.__repr__:   # Counter(count=2), not <... object at 0x7f...>
            if level <= 0:
                return f"{type(value).__name__}(…)"
            inner = ", ".join(f"{k}={self.repr1(v, level - 1)}" for k, v in list(vars(value).items())[:4])
            return f"{type(value).__name__}({inner})"
        return super().repr1(value, level)


_SHORT = _Short(maxstring=34, maxother=34, maxlist=8, maxtuple=8, maxset=8, maxdict=6, maxlevel=2)
_SHORT.fillvalue = "…"


def _short(value) -> str:
    """A value the way Python writes it, cut short with … when it is long."""
    try:
        text = _SHORT.repr(value)
    except Exception:   # a learner's object whose __repr__ fails
        text = f"<{type(value).__name__}>"
    return text if len(text) <= 40 else text[:39] + "…"


def _shown_name(name: str, value) -> bool:
    """A name worth showing: one the code made for a value, not a module, function, class or Python's own."""
    return (not name.startswith("_") and not isinstance(value, (types.ModuleType, types.FunctionType, type,
                                                                 types.BuiltinFunctionType, types.MethodType)))


def _figure(title: str, body: str, *, alt: str = "") -> str:
    """A picture of the learner's work: its title, then the body, in the column of the lesson."""
    return (f'<figure class="ea-figure ea-yours" role="group" aria-label="{html.escape(alt or title)}" '
            f'style="margin:0.6rem 0 0.75rem;max-width:min(100%,44rem)">'
            f'<figcaption style="font-family:var(--ea-font-ui);font-size:0.92rem;font-weight:600;color:var(--ea-ink);'
            f'margin:0 0 0.35rem">{html.escape(title)}</figcaption>{body}</figure>')


def digit_batch(flat: bool = False):
    """The first test digit of the course's QMNIST sample (data/qmnist-small.npz) as a batch of one, from 0 to 1:
    (1, 1, 28, 28), or (1, 784) with `flat`. What netview.network sends through a learner's digit network, so its
    feature maps show a digit rather than noise."""
    import numpy as np
    import torch
    from everything_ai import paths
    with np.load(paths.ROOT / "data" / "qmnist-small.npz") as data:
        x = torch.tensor(data["test_images"][:1], dtype=torch.float32) / 255
    return x if flat else x.reshape(1, 1, 28, 28)


def scene_batch(size: int = 32, seed: int = 18):
    """A scene of the course (scenes.py) as a colour batch of one, (1, 3, size, size) from 0 to 1, made smaller by
    averaging: what netview.network sends through a learner's colour network."""
    import torch
    import torch.nn.functional as F
    from everything_ai import scenes
    x = torch.tensor(scenes.scene(seed).image).permute(2, 0, 1).unsqueeze(0).float() / 255
    return x if x.shape[-1] == size else F.interpolate(x, size=(size, size), mode="area")


def figure(title: str, body: str, *, alt: str = "") -> str:
    """A picture of the learner's work that a lesson draws itself, such as a grid world with the values of their
    value iteration, framed as the pictures here are: its title, then the body."""
    return _figure(title, body, alt=alt)


# --- steps: what the code does, line by line ---------------------------------------------------------------------

@dataclass
class _Step:
    kind: str                  # "line", "call", "return", "yield" or "error"
    depth: int                 # 0 for the code in the box, 1 inside a function it calls, and so on
    number: int = 0            # the line of the code, counted from 1
    line: str = ""             # what that line says
    names: dict = field(default_factory=dict)   # name -> value as Python writes it, after the line
    out: str = ""              # what the line printed
    text: str = ""             # for a call, a return or a yield: what to say
    frame: int = 0             # which call of a function the line belongs to
    start: int = 0             # how much was printed before the line


class _Recorder:
    """A trace function (sys.settrace) that writes down each line of the learner's code as it runs: the names after
    it and what it printed. `mine(code)` says whether a code object is the learner's."""

    def __init__(self, mine, source):
        self.mine, self.source = mine, source
        self.steps: list[_Step] = []
        self.count = 0          # every line that ran, also the ones not kept
        self.depth = 0
        self.open: dict[int, _Step] = {}   # the line each frame is on, until its names are known
        self.raising: set[int] = set()     # frames an error is passing through, unless a line of theirs runs again
        self.stopped = False               # the code stopped with an error: its last line is where

    def _printed(self) -> str:
        out = sys.stdout
        return out.getvalue() if isinstance(out, io.StringIO) else ""

    def _names(self, frame) -> dict:
        ns = frame.f_globals if frame.f_code.co_name == "<module>" else frame.f_locals
        return {k: _short(v) for k, v in list(ns.items()) if _shown_name(k, v)}

    def _finish(self, frame) -> None:
        step = self.open.pop(id(frame), None)
        if step is not None:
            step.names = self._names(frame)
            step.out = self._printed()[step.start:]

    def _keep(self, step: _Step) -> None:
        if len(self.steps) < _MAX_RECORDED:
            self.steps.append(step)

    def __call__(self, frame, event, arg):
        code = frame.f_code
        # the code in the box and the functions it defines; not the body of a class, nor a generator expression
        if event != "call" or not self.mine(code) or not (code.co_name == "<module>" or code.co_flags & inspect.CO_OPTIMIZED) \
                or code.co_name.startswith("<") and code.co_name != "<module>":
            return None
        if code.co_name != "<module>":
            self.depth += 1
            args = inspect.getargvalues(frame)
            shown = ", ".join(_short(args.locals[a]) for a in args.args if a != "self")
            if not (code.co_flags & inspect.CO_GENERATOR and frame.f_lasti > 0):   # a generator that goes on is no new call
                self._keep(_Step("call", self.depth, text=f"{code.co_name}({shown})", frame=id(frame)))
        return self._local

    def _local(self, frame, event, arg):
        if event == "line":
            self._finish(frame)
            self.raising.discard(id(frame))   # the error was caught: the frame goes on
            self.count += 1
            if len(self.steps) < _MAX_RECORDED:
                step = _Step("line", self.depth, number=frame.f_lineno, line=self.source(frame), frame=id(frame),
                             start=len(self._printed()))
                self.steps.append(step)
                self.open[id(frame)] = step
        elif event == "exception":
            self.raising.add(id(frame))
        elif event == "return":
            self._finish(frame)
            if frame.f_code.co_name != "<module>":
                generator = frame.f_code.co_flags & inspect.CO_GENERATOR
                # a frame that ends with an error gives nothing back; a generator that ends hands out nothing
                if id(frame) not in self.raising and not (generator and arg is None) and frame.f_code.co_name != "__init__":
                    self._keep(_Step("yield" if generator else "return", self.depth, text=_short(arg), frame=id(frame)))
                self.depth -= 1
            self.raising.discard(id(frame))
        return self._local


def _line_of(lines: list[str], first: int = 1):
    def source(frame) -> str:
        i = frame.f_lineno - first
        return lines[i].strip() if 0 <= i < len(lines) else ""
    return source


def _codes(fn) -> set:
    """The code objects of a function and of everything defined inside it."""
    out, todo = set(), [getattr(fn, "__code__", None)]
    while todo:
        code = todo.pop()
        if isinstance(code, types.CodeType) and code not in out:
            out.add(code)
            todo.extend(c for c in code.co_consts if isinstance(c, types.CodeType))
    return out


def _record_code(text: str, names: dict) -> _Recorder:
    """Run the code in the box with a recorder."""
    rec = _Recorder(lambda code: code.co_filename == "<your code>", _line_of(text.splitlines()))
    run = checking.run_code(text, names, trace=rec)
    rec.stopped = run.error is not None
    return rec


def _record_call(fn, case: tuple, text: str | None) -> _Recorder:
    """Call the learner's function on one case with a recorder. Its lines come from the box, or for a function in a
    cell of the lesson from that cell."""
    if text is not None:
        rec = _Recorder(lambda code: code.co_filename == "<your code>", _line_of(text.splitlines()))
    else:
        codes = _codes(fn)
        try:
            lines, first = inspect.getsourcelines(fn)
        except (OSError, TypeError):
            lines, first = [], 1
        rec = _Recorder(lambda code: code in codes, _line_of(lines, first))
    before = sys.gettrace()
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            def go():
                sys.settrace(rec)
                try:
                    value = fn(*(checking._copy(a) for a in case))
                    if inspect.isgenerator(value):   # a generator does its work while its values are taken
                        value = list(value)
                    return value
                finally:
                    sys.settrace(before)
            checking.with_time_limit(go)
    except checking.Timeout:   # the steps so far are what it did
        pass
    except Exception:
        rec.stopped = True
    return rec


def steps(call: int | None = None, *, scenario: int | None = None, limit: int = 22, title: Text | None = None):
    """What the learner's code does, line by line, as a table: each line that runs, the names after it (the ones it
    changed in ink, the rest grey) and what it shows. An `if` shows by the lines that ran which way it went; a loop
    shows a row for each line of each round; a function it calls is indented, with what it gets and gives back. Where
    the code stops with an error, the last row says so. With more than `limit` lines, the middle is left out.

    For a function exercise (a FunctionBox, or a BugFix whose function is in a cell), `call` is the case to follow:
    an index into the exercise's visible cases, then its hidden ones. For a class or a decorator (a ClassBox),
    `scenario` is the index of the scenario whose lines run after the learner's code."""
    def picture(seen: checking.Seen) -> str:
        ex = seen.exercise
        lang = seen.lang
        if scenario is not None:
            rec = _record_code(seen.text + "\n\n" + ex.scenarios[scenario], seen.names)
            heading = _word("yours-steps", lang)
        elif call is None:
            rec = _record_code(seen.text, seen.names)
            heading = _word("yours-steps", lang)
        else:
            case = (list(getattr(ex, "visible", [])) + list(getattr(ex, "hidden", [])))[call]
            if isinstance(seen.text, str):
                if seen.mine is None:
                    return ""
                fn = seen.mine.names.get(ex.function)
                text = seen.text
            else:
                fn, text = seen.text, None
            if not callable(fn):
                return ""
            rec = _record_call(fn, case, text)
            heading = _word("yours-steps-call", lang, call=ex._call_text(case))
        if title is not None:
            heading = seen.word(title)
        return _steps_html(rec, heading, lang, limit)
    return picture


def _cut(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n - 1] + "…"


def _steps_html(rec: _Recorder, heading: str, lang: str, limit: int) -> str:
    rows = rec.steps
    lines = [i for i, s in enumerate(rows) if s.kind == "line"]
    if not lines:
        return ""
    cell = "padding:0.22rem 0.6rem;vertical-align:top;border-top:1px solid var(--ea-line);white-space:pre-wrap;overflow-wrap:anywhere"
    head = "padding:0.25rem 0.6rem;text-align:left;font-family:var(--ea-font-ui);font-size:0.78rem;font-weight:600;color:var(--ea-ink-2)"
    # which rows to show: the first ones and the last few, with a row that counts the ones in between
    keep = list(range(len(rows)))
    hidden = 0
    if len(lines) > limit:
        first_end = lines[limit - 4]          # the first limit - 3 lines, with the calls and returns between them
        last_start = lines[-3]
        keep = list(range(first_end + 1)) + [None] + list(range(last_start, len(rows)))
        hidden = sum(1 for i in lines if first_end < i < last_start)
    hidden += max(0, rec.count - len(lines))   # lines that ran after the record was full
    before: dict[int, dict] = {}
    body = []
    for i in keep:
        if i is None:
            body.append(f'<tr><td colspan="3" style="{cell};font-family:var(--ea-font-ui);font-style:italic;color:var(--ea-ink-3)">'
                        f'{html.escape(_word("yours-more-steps", lang, n=hidden))}</td></tr>')
            continue
        s = rows[i]
        pad = f"padding-left:{0.6 + 1.2 * s.depth:.1f}rem"
        if s.kind in ("call", "return", "yield"):
            words = {"call": "→ " + s.text,
                     "return": "← " + _word("yours-gives-back", lang, value=s.text),
                     "yield": "← " + _word("yours-hands-out", lang, value=s.text)}[s.kind]
            body.append(f'<tr><td colspan="3" style="{cell};{pad};color:var(--ea-ink-2)">{html.escape(words)}</td></tr>')
            continue
        old = before.get(s.frame, {})
        shown = []
        for name, value in s.names.items():
            new = old.get(name) != value
            if not new and len(value) > 14:   # a long value that stays the same was shown in an earlier row
                continue
            shown.append(f'<span style="color:var(--ea-ink{"" if new else "-3"});font-weight:{600 if new else 400}">'
                         f'{html.escape(f"{name} = {value}")}</span>')
        before[s.frame] = s.names
        out = "<br>".join(html.escape(x) for x in s.out.rstrip("\n").split("\n")) if s.out else ""
        if rec.stopped and i == lines[-1]:
            out = (out + "<br>" if out else "") + (f'<span style="font-family:var(--ea-font-ui);font-weight:600;color:var(--ea-ink)">'
                                                   f'{html.escape(_word("yours-stops", lang))}</span>')
        body.append(f'<tr><td style="{cell};{pad}"><span style="color:var(--ea-ink-3)">{s.number}</span>  '
                    f'<span style="color:var(--ea-ink)">{html.escape(_cut(s.line, 36))}</span></td>'
                    f'<td style="{cell}">{"  ".join(shown)}</td><td style="{cell};color:var(--ea-ink)">{out}</td></tr>')
    table = (f'<div style="overflow-x:auto;border:1px solid var(--ea-line);border-radius:var(--ea-radius-small,6px)">'
             f'<table style="border-collapse:collapse;width:100%;{_MONO};font-size:0.8rem;line-height:1.45">'
             f'<thead><tr><th style="{head}">{html.escape(_word("yours-line", lang))}</th>'
             f'<th style="{head}">{html.escape(_word("yours-names-after", lang))}</th>'
             f'<th style="{head}">{html.escape(_word("yours-shows", lang))}</th></tr></thead>'
             f'<tbody>{"".join(body)}</tbody></table></div>')
    return _figure(heading, table)


# --- names: the values the code made, in boxes --------------------------------------------------------------------

_BH = 36        # the height of a box
_CH = 28        # the width of the box of one letter


def _box(x: float, y: float, w: float, text: str, *, size: int = 15, dashed: bool = False) -> str:
    dash = ";stroke-dasharray:4 3" if dashed else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{_BH}" rx="5" style="fill:var(--ea-inset);'
            f'stroke:var(--ea-line){dash}"/>'
            f'<text x="{x + w / 2:.1f}" y="{y + _BH / 2 + size * 0.36:.1f}" style="{_MONO};font-size:{size}px;'
            f'fill:var(--ea-ink);text-anchor:middle;white-space:pre">{html.escape(text)}</text>')


def _index(x: float, y: float, i: int) -> str:
    return (f'<text x="{x:.1f}" y="{y - 5:.1f}" style="{_MONO};font-size:11.5px;fill:var(--ea-ink-3);'
            f'text-anchor:middle">{i}</text>')


def _width(text: str, size: int = 15) -> float:
    return max(_CH, 16 + 0.62 * size * len(text))


def _indexed(value) -> bool:
    """Whether _value draws places above the boxes of this value."""
    return (isinstance(value, str) and 0 < len(value) <= 16
            or isinstance(value, (list, tuple)) and 0 < len(value) <= 40)


def _boxes_svg(width: float, height: float, alt: str, parts: list[str]) -> str:
    """Boxes drawn at the size of the text in the lesson; in a narrow window they shrink to three quarters of that,
    so their text stays readable (at least 11 px), and the rest scrolls sideways."""
    return (f'<div style="overflow-x:auto"><svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" style="width:100%;'
            f'max-width:{width:.0f}px;min-width:{0.75 * width:.0f}px;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(alt)}</title>{"".join(parts)}</svg></div>')


def _value(value, x: float, y: float, more: str) -> tuple[str, float, bool]:
    """A value drawn from (x, y): (its SVG, how wide it is, whether it has indices above). A text letter by letter,
    a list, tuple or set item by item, a dictionary key to value; anything else in one box."""
    parts, cx, indexed = [], x, False
    if isinstance(value, str) and 0 < len(value) <= 16:
        indexed = True
        for i, ch in enumerate(value):
            parts.append(_index(cx + _CH / 2, y, i) + _box(cx, y, _CH, "␣" if ch == " " else ch, size=16))
            cx += _CH + 3
        return "".join(parts), cx - x, indexed
    if isinstance(value, (list, tuple, set, frozenset)) and value and len(value) <= 40:
        items = sorted(value, key=repr) if isinstance(value, (set, frozenset)) else list(value)
        ordered = not isinstance(value, (set, frozenset))
        shown = items[:10]
        start = cx
        for i, item in enumerate(shown):
            text = _short(item)
            w = _width(text)
            if ordered:
                parts.append(_index(cx + w / 2, y, i))
            parts.append(_box(cx, y, w, text))
            cx += w + 4
        if len(items) > len(shown):
            text = more.format(n=len(items) - len(shown))
            parts.append(f'<text x="{cx + 4:.1f}" y="{y + _BH / 2 + 5:.1f}" style="font-family:var(--ea-font-ui);'
                         f'font-size:13px;fill:var(--ea-ink-2)">{html.escape(text)}</text>')
            cx += 12 + 6.5 * len(text)
        if not ordered:   # a set has no order: its boxes lie in a dashed bag, without places
            parts.insert(0, f'<rect x="{start - 5:.1f}" y="{y - 5:.1f}" width="{cx - start + 6:.1f}" height="{_BH + 10}" '
                            f'rx="12" style="fill:none;stroke:var(--ea-ink-3);stroke-dasharray:4 4"/>')
        return "".join(parts), cx - x, ordered
    if isinstance(value, dict) and value and len(value) <= 40:
        pairs = list(value.items())
        shown = pairs[:6]
        for k, v in shown:
            kt, vt = _short(k), _short(v)
            kw, vw = _width(kt, 14), _width(vt, 14)
            parts.append(_box(cx, y, kw, kt, size=14))
            ax = cx + kw + 4
            parts.append(f'<path d="M{ax:.1f} {y + _BH / 2} h12 m-5 -4 l5 4 l-5 4" style="fill:none;stroke:var(--ea-ink-2);'
                         f'stroke-width:1.4;stroke-linecap:round;stroke-linejoin:round"/>')
            parts.append(_box(ax + 18, y, vw, vt, size=14))
            cx = ax + 18 + vw + 14
        if len(pairs) > len(shown):
            text = more.format(n=len(pairs) - len(shown))
            parts.append(f'<text x="{cx:.1f}" y="{y + _BH / 2 + 5:.1f}" style="font-family:var(--ea-font-ui);'
                         f'font-size:13px;fill:var(--ea-ink-2)">{html.escape(text)}</text>')
            cx += 8 + 6.5 * len(text)
        return "".join(parts), cx - x, False
    text = _short(value)
    w = _width(text)
    return _box(cx, y, w, text), w, False


def names(*, only: list[str] | None = None, hide: tuple[str, ...] = (), title: Text | None = None):
    """The names the learner's code made, one row each: the name, then its value in boxes. A text gets a box per
    letter with its place above, a list a box per item with its index, a set its items in a dashed bag (no order),
    a dictionary each key with an arrow to its value. `only` picks names and their order; `hide` leaves names out.
    Names the lesson gives the code (`names` of the exercise) are left out, unless the code changed them."""
    def picture(seen: checking.Seen) -> str:
        mine = seen.mine
        if mine is None:
            return ""
        ns = mine.names
        given = seen.names
        chosen = [k for k in (only if only is not None else ns)
                  if k in ns and k not in hide and k != "__name__" and _shown_name(k, ns[k])
                  and not (k in given and given[k] is ns[k] and only is None)]
        if not chosen:
            return ""
        lang = seen.lang
        more = _word("yours-more", lang)
        label_w = max(60, 16 + 8.6 * max(len(k) for k in chosen))
        parts, y, width = [], 4, 300
        for name in chosen:
            top = y + (16 if _indexed(ns[name]) else 2)   # room for the places above the boxes
            drawn, w, _ = _value(ns[name], label_w, top, more)
            parts.append(f'<text x="{label_w - 12:.1f}" y="{top + _BH / 2 + 5:.1f}" style="{_MONO};font-size:14px;'
                         f'font-weight:600;fill:var(--ea-ink);text-anchor:end">{html.escape(name)}</text>{drawn}')
            width = max(width, label_w + w + 8)
            y = top + _BH + 12
        heading = seen.word(title) if title is not None else _word("yours-names", lang)
        alt = "; ".join(f"{k} = {_short(ns[k])}" for k in chosen)
        return _figure(heading, _boxes_svg(width, y, alt, parts), alt=alt)
    return picture


# --- curve: a function of one number, next to the right one -------------------------------------------------------

def _at(fn, x):
    """fn(x) as a float, or None where it stops with an error or gives no number."""
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")   # NumPy's "divide by zero" and the like: the gap in the line says it
            y = float(fn(x))
    except Exception:
        return None
    return y if math.isfinite(y) else None


def _segments(xs, ys, lo: float, hi: float) -> list[list[tuple[float, float]]]:
    """The stretches of a line where it has a number between lo and hi: a function that gives no number, or one far
    off the chart, leaves a gap."""
    parts: list[list[tuple[float, float]]] = [[]]
    for x, v in zip(xs, ys):
        if v is None or not lo <= v <= hi:
            parts.append([])
        else:
            parts[-1].append((x, v))
    return [p for p in parts if p]


def curve(function: str, xs, *, at=None, role: str = "pred", title: Text | None = None, xlabel: Text | str = "x",
          ylabel: Text | str = "", y: tuple[float, float] | None = None, xlog: bool = False):
    """The learner's function of one number as a line in the colour of its role, with the right one dashed and grey
    under it, over the values `xs`; where the two lie on top of each other, the learner sees one line. Where the
    learner's function stops with an error or gives no number, its line has a gap. `y` fixes the range of the
    y-axis; without it, the range fits both lines. For a function of more numbers, `at(x)` gives its arguments for
    each x, such as `lambda w: (w, 4, 7)` for loss_slope(w, x, y). With `xlog`, the x-axis has a log scale, for xs
    that run over powers of ten, such as model sizes."""
    xs = list(xs)
    args = at or (lambda x: (x,))

    def picture(seen: checking.Seen) -> str:
        mine = seen.mine
        if mine is None:
            return ""
        fn, right = mine.names.get(function), seen.want.names.get(function)
        if not callable(fn) or not callable(right):
            return ""
        lang = seen.lang
        # each call gets its own copy of the arguments, as in the check, so a function that changes a list it gets
        # does not change the next call's
        ys_mine = checking.with_time_limit(lambda: [_at(lambda v: fn(*copy.deepcopy(args(v))), x) for x in xs])
        ys_right = [_at(lambda v: right(*copy.deepcopy(args(v))), x) for x in xs]
        known = [v for v in ys_mine + ys_right if v is not None]
        if not known:
            return ""
        lo, hi = y if y is not None else (min(known), max(known))
        if hi - lo < 1e-9:
            lo, hi = lo - 1, hi + 1
        pad = 0 if y is not None else 0.06 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        heading = seen.word(title) if title is not None else _word("yours-curve", lang, function=function)
        panel = chart.Panel(x=(xs[0], xs[-1]), y=(lo, hi), xlabel=seen.word(xlabel), ylabel=seen.word(ylabel), xlog=xlog)
        for part in _segments(xs, ys_mine, lo - (hi - lo), hi + (hi - lo)):   # a gap where the learner's gives no number
            panel.line([a for a, _ in part], [b for _, b in part], role=role)
        # the right one on top, dashed: where the learner's line is right, it shows through between the dashes; where
        # the right function gives no number either, such as a split with an empty part, both have a gap
        for part in _segments(xs, ys_right, lo - (hi - lo), hi + (hi - lo)):
            panel.line([a for a, _ in part], [b for _, b in part], role="ink-3", width=2, dashed=True)
        key = (f'<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);display:flex;gap:1.2rem;'
               f'flex-wrap:wrap;margin-top:0.1rem">'
               f'<span><svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" style="stroke:var(--ea-{role});'
               f'stroke-width:2.4;stroke-linecap:round"/></svg> {html.escape(_word("yours-code", lang))}</span>'
               f'<span><svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" style="stroke:var(--ea-ink-3);'
               f'stroke-width:2;stroke-dasharray:5 5"/></svg> {html.escape(_word("yours-right", lang))}</span></div>')
        figure = chart.figure([panel], alt=heading, width=420, height=250)
        return _figure(heading, figure + key)
    return picture


# --- objects: what the objects of the learner's classes hold --------------------------------------------------------

def objects(scenario: int = 0, *, title: Text | None = None):
    """The objects the learner's class makes in one of the exercise's scenarios (a ClassBox), each as a card: its
    name and class on top, the class it inherits from after it, and every attribute with its value in boxes, as in
    names(). The scenario runs up to its last line, the value the check compares, which only reads."""
    def picture(seen: checking.Seen) -> str:
        if seen.mine is None:
            return ""
        lines = seen.exercise.scenarios[scenario].rstrip().split("\n")
        run = checking.run_code(seen.text + "\n\n" + "\n".join(lines[:-1]), seen.names)
        if run.error is not None or run.timeout:
            return ""
        made = [(k, v) for k, v in run.names.items() if not k.startswith("_") and _learners(v)]
        if not made:
            return ""
        lang = seen.lang
        more = _word("yours-more", lang)
        parts, y, width = [], 2, 320
        for name, obj in made:
            cls = type(obj)
            parents = [b.__name__ for b in cls.__bases__ if b is not object]
            head = f"{name}: {cls.__name__}" + (f"({', '.join(parents)})" if parents else "")
            attrs = list(vars(obj).items())
            label_w = max(70, 30 + 8.6 * max([len(k) for k, _ in attrs] or [4]))
            rows, inner_w, ry = [], 200, y + 34
            for k, v in attrs:
                ry += 16 if _indexed(v) else 2
                drawn, w, _ = _value(v, 12 + label_w, ry, more)
                rows.append(f'<text x="{label_w:.1f}" y="{ry + _BH / 2 + 5:.1f}" style="{_MONO};font-size:14px;'
                            f'fill:var(--ea-ink-2);text-anchor:end">{html.escape(k)}</text>{drawn}')
                inner_w = max(inner_w, 12 + label_w + w + 14)
                ry += _BH + 6
            h = max(ry - y + 4, 40)
            inner_w = max(inner_w, 24 + 8.4 * len(head))
            parts.append(f'<rect x="1" y="{y:.1f}" width="{inner_w:.1f}" height="{h:.1f}" rx="8" style="fill:none;'
                         f'stroke:var(--ea-ink-3);stroke-width:1.2"/>'
                         f'<text x="12" y="{y + 21:.1f}" style="{_MONO};font-size:14px;font-weight:600;fill:var(--ea-ink)">'
                         f'{html.escape(head)}</text>{"".join(rows)}')
            width = max(width, inner_w + 4)
            y += h + 12
        heading = seen.word(title) if title is not None else _word("yours-objects", lang)
        alt = "; ".join(f"{k} = {_short(v)}" for k, v in made)
        return _figure(heading, _boxes_svg(width, y, alt, parts), alt=alt)
    return picture


# --- bars: a count per key, next to the right one ------------------------------------------------------------------

def bars(call: int = 0, *, title: Text | None = None, ylabel: Text | str = "", top: int = 26):
    """What the learner's function gives back for one case, a dictionary of counts such as {"l": 3, "o": 2}, as a bar
    per key, with the right count as a dashed line over each bar: where a bar reaches its line, the count is right.
    Keys in order; with more than `top` keys, the ones with the highest counts."""
    def picture(seen: checking.Seen) -> str:
        ex = seen.exercise
        mine = seen.mine
        if mine is None:
            return ""
        case = (list(ex.visible) + list(ex.hidden))[call]
        fn, right = mine.names.get(ex.function), seen.want.names.get(ex.function)
        if not callable(fn) or not callable(right):
            return ""
        try:
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                got = checking.with_time_limit(lambda: fn(*(checking._copy(a) for a in case)))
        except Exception:
            return ""
        want = right(*(checking._copy(a) for a in case))
        if not isinstance(got, dict):
            return ""
        def number(v):
            return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0
        keys = sorted(set(got) | set(want), key=lambda k: (str(type(k)), k))
        if len(keys) > top:
            keys = sorted(sorted(keys, key=lambda k: -max(number(got.get(k)), number(want.get(k))))[:top],
                          key=lambda k: (str(type(k)), k))
        if not keys:
            return ""
        lang = seen.lang
        what = ex._call_text(case)
        heading = (seen.word(title) if title is not None
                   else _word("yours-counts", lang, call=what if len(what) <= 34 else f"{ex.function}(…)"))
        hi = max([number(got.get(k)) for k in keys] + [number(want.get(k)) for k in keys] + [1])
        key = (f'<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);display:flex;gap:1.2rem;'
               f'flex-wrap:wrap;margin-top:0.3rem">'
               f'<span><svg width="14" height="12" aria-hidden="true"><rect x="1" y="1" width="12" height="10" style="fill:var(--ea-data);'
               f'fill-opacity:0.18;stroke:var(--ea-data);stroke-width:1.2"/></svg> {html.escape(_word("yours-code", lang))}</span>'
               f'<span><svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" style="stroke:var(--ea-ink);'
               f'stroke-width:2;stroke-dasharray:5 5"/></svg> {html.escape(_word("yours-right", lang))}</span></div>')
        if max(len(str(k)) for k in keys) > 3:   # names that do not fit under a bar: a bar per row, its name before it
            rows = []
            for k in keys:
                v, w = number(got.get(k)), number(want.get(k))
                rows.append(
                    f'<span style="{_MONO};font-size:0.8rem;color:var(--ea-ink-2);text-align:right;white-space:nowrap;overflow:hidden;'
                    f'text-overflow:ellipsis">{html.escape(_cut(str(k), 24))}</span>'
                    f'<span style="position:relative;height:1.15rem;display:block">'
                    f'<span style="position:absolute;left:0;top:0.1rem;bottom:0.1rem;width:{100 * v / hi:.1f}%;'
                    f'background:color-mix(in srgb,var(--ea-data) 18%,transparent);border:1.2px solid var(--ea-data)"></span>'
                    f'<span style="position:absolute;left:{100 * w / hi:.1f}%;top:0;bottom:0;border-left:2px dashed var(--ea-ink)"></span>'
                    f'</span><span style="{_MONO};font-size:0.8rem;color:var(--ea-ink)">{html.escape(_cell(got.get(k, 0)))}</span>')
            body = (f'<div role="img" aria-label="{html.escape(heading)}" style="display:grid;grid-template-columns:minmax(3rem,'
                    f'max-content) minmax(6rem,1fr) max-content;gap:0.25rem 0.6rem;align-items:center;max-width:36rem">'
                    f'{"".join(rows)}</div>')
            return _figure(heading, body + key)
        whole = all(float(v).is_integer() for v in [number(got.get(k)) for k in keys] + [number(want.get(k)) for k in keys])
        yticks = [float(t) for t in chart.ticks(0, hi) if not whole or float(t).is_integer()]
        panel = chart.Panel(x=(-0.6, len(keys) - 0.4), y=(0, hi * 1.08), ylabel=seen.word(ylabel), yticks=yticks,
                            xticks=list(range(len(keys))), xtick_labels=[str(k) for k in keys])
        for i, k in enumerate(keys):
            v = number(got.get(k))
            if v > 0:
                panel.rect(i - 0.34, 0, i + 0.34, v, role="data")
            panel.line([i - 0.42, i + 0.42], [number(want.get(k))] * 2, role="ink", width=2, dashed=True)
        width = max(330, min(640, 70 + 34 * len(keys)))
        return _figure(heading, chart.figure([panel], alt=heading, width=width, height=230) + key)
    return picture


# --- arrays: what the code makes of arrays, as grids of numbers ------------------------------------------------------

_CW, _CHT = 46, 30   # the width and height of one cell of a grid


def _cell(v) -> str:
    """One number of an array, short: 0.333 and 12, not 0.3333333 and 12.0."""
    if isinstance(v, bool) or type(v).__name__ in ("bool_", "bool"):   # NumPy 2 calls its own "bool"
        return "True" if v else "False"
    if isinstance(v, str) or type(v).__name__ == "str_":
        return repr(str(v))
    try:
        f = float(v)
    except (TypeError, ValueError):
        return _short(v)
    if not math.isfinite(f):
        return str(f)
    if f.is_integer() and abs(f) < 1e7:
        return str(int(f))
    if abs(f) >= 1000:
        return f"{f:.0f}"
    text = f"{f:.3f}".rstrip("0").rstrip(".")
    return text if text not in ("0", "-0") else f"{f:.1e}"


def _as_grid(value):
    """A value as rows of cells: a number is one cell, a vector one row; None for what is not an array of numbers."""
    try:
        import numpy as np
    except ImportError:   # pragma: no cover - NumPy comes with the course
        return None
    if type(value).__module__.startswith("torch"):
        try:
            value = value.detach().cpu().numpy()
        except Exception:
            return None
    if isinstance(value, (str, dict)) or callable(value):
        return None
    try:
        a = np.asarray(value)
    except Exception:
        return None
    if a.dtype == object or a.ndim > 2:
        return None
    return a.reshape(1, -1) if a.ndim <= 1 else a


def _grid_svg(label: str, a, other=None, *, shade: bool = False) -> str:
    """One array as a grid of boxes, with its name and shape above it. Cells that differ from `other` (an array of the
    same shape) get a thick border."""
    rows, cols = a.shape
    shown_r, shown_c = min(rows, 10), min(cols, 20 if rows == 1 else 10)
    texts = [[_cell(a[i, j]) for j in range(shown_c)] for i in range(shown_r)]
    cw = max(_CW, max((7.8 * len(t) + 14 for row in texts for t in row), default=_CW))
    top = 26
    width = max(cw * shown_c + (22 if cols > shown_c else 0) + 2, 8 * len(label) + 4)
    height = top + _CHT * shown_r + (18 if rows > shown_r else 0) + 2
    peak = 0.0
    if shade:
        try:
            peak = float(abs(a.astype(float)).max()) or 1.0
        except (TypeError, ValueError):
            shade = False
    parts = [f'<text x="1" y="15" style="{_MONO};font-size:13px;fill:var(--ea-ink-2)">{html.escape(label)}</text>']
    for i in range(shown_r):
        for j in range(shown_c):
            x, y = 1 + j * cw, top + i * _CHT
            differs = other is not None and other.shape == a.shape and not _same_cell(a[i, j], other[i, j])
            fill = "var(--ea-page)"
            wash = ""
            if shade:
                wash = (f'<rect x="{x:.1f}" y="{y}" width="{cw:.1f}" height="{_CHT}" style="fill:var(--ea-data);'
                        f'fill-opacity:{0.04 + 0.3 * abs(float(a[i, j])) / peak:.2f}"/>')
            stroke = "stroke:var(--ea-ink);stroke-width:2.2" if differs else "stroke:var(--ea-line);stroke-width:1"
            parts.append(f'<rect x="{x:.1f}" y="{y}" width="{cw:.1f}" height="{_CHT}" style="fill:{fill};{stroke}"/>{wash}'
                         f'<text x="{x + cw / 2:.1f}" y="{y + _CHT / 2 + 4.5:.1f}" style="{_MONO};font-size:13px;'
                         f'font-weight:{600 if differs else 400};fill:var(--ea-ink);text-anchor:middle">{html.escape(texts[i][j])}</text>')
    if cols > shown_c:
        parts.append(f'<text x="{1 + shown_c * cw + 6:.1f}" y="{top + _CHT / 2 + 4:.1f}" style="font-size:13px;fill:var(--ea-ink-3)">…</text>')
    if rows > shown_r:
        parts.append(f'<text x="{1 + cw / 2:.1f}" y="{top + shown_r * _CHT + 13:.1f}" style="font-size:13px;fill:var(--ea-ink-3);text-anchor:middle">⋮</text>')
    return (f'<svg viewBox="0 0 {width:.0f} {height:.0f}" aria-hidden="true" style="width:{width:.0f}px;max-width:100%;height:auto;'
            f'font-family:var(--ea-font-ui);flex:0 0 auto">{"".join(parts)}</svg>')


def _same_cell(x, y) -> bool:
    try:
        fx, fy = float(x), float(y)
    except (TypeError, ValueError):
        return x == y
    return math.isclose(fx, fy, rel_tol=1e-6, abs_tol=1e-9) or (math.isnan(fx) and math.isnan(fy))


def _shape(value) -> str:
    shape = getattr(value, "shape", None)
    if shape is None:
        try:
            import numpy as np
            shape = np.asarray(value).shape
        except Exception:
            return ""
    if len(shape) == 0:   # one number has no shape worth writing
        return ""
    return "(" + ", ".join(str(int(s)) for s in shape) + ("," if len(shape) == 1 else "") + ")"


def _several(value) -> bool:
    """A list or tuple of arrays of different shapes, such as [dW, dX], or an array and a number, such as (grad_w,
    grad_b)."""
    if not isinstance(value, (list, tuple)) or not 1 < len(value) <= 4:
        return False
    grids = [_as_grid(v) for v in value]
    if not all(g is not None for g in grids) or not any(getattr(v, "ndim", 0) >= 1 or isinstance(v, (list, tuple)) for v in value):
        return False
    # what a function gives back as a tuple, such as (X, Y), is several things even when they have one shape; a list
    # of rows of one length is one table
    return isinstance(value, tuple) or len({g.shape for g in grids}) > 1


def arrays(call: int | None = None, *, show: tuple[str, ...] = (), result: str | None = None, shade: bool = False,
           column: bool = False, title: Text | None = None):
    """What the learner's code makes of arrays, as grids of numbers, each with its name and shape: first the arrays it
    starts from, then the result. Where the result is not right, the right one stands next to it, and the cells that
    differ have a thick border in both. A code box shows the names in `show` from its code, then `result` (a name)
    or the value of its last line. A function exercise shows the arrays of case `call` it is given, then what it
    gives back. With `shade`, each cell gets a grey wash as strong as its number is large. With `column`, a result
    with one number per row of what it starts from (a mean along axis 1) stands as a column, beside those rows."""
    def picture(seen: checking.Seen) -> str:
        mine = seen.mine
        if mine is None:
            return ""
        lang = seen.lang
        out = _outcome(seen, call, show, result)
        if out is None:
            return ""
        got, want, start, what = out
        if _several(got):   # a list of arrays, such as [dW, dX]: a grid for each
            pieces = [_grid_svg(f"{name}  {_shape(value)}", g, shade=shade) for name, value in start
                      if (g := _as_grid(value)) is not None and g.size > 1]
            results = []
            for i, value in enumerate(got):
                g = _as_grid(value)
                other = _as_grid(want[i]) if _several(want) and i < len(want) else None
                if g is not None:
                    results.append(_grid_svg(f"{_word('yours-code', lang)}[{i}]  {_shape(value)}", g, other, shade=shade))
                if other is not None and (g is None or g.shape != other.shape or not all(
                        _same_cell(x, y) for x, y in zip(g.ravel(), other.ravel()))):
                    results.append(_grid_svg(f"{_word('yours-right', lang)}[{i}]  {_shape(want[i])}", other, g, shade=shade))
            if not results:
                return ""
            arrow = '<span aria-hidden="true" style="align-self:center;color:var(--ea-ink-3);font-size:1.2rem">→</span>'
            body = (f'<div style="display:flex;flex-wrap:wrap;gap:0.5rem 1.1rem;align-items:flex-start">'
                    f'{"".join(pieces)}{arrow if pieces else ""}{"".join(results)}</div>')
            heading = seen.word(title) if title is not None else _word("yours-arrays", lang)
            return _figure(heading, body, alt=f"{what} = {_short(got)}")
        mine_grid, want_grid = _as_grid(got), _as_grid(want)
        if mine_grid is None:
            return ""
        heading = seen.word(title) if title is not None else _word("yours-arrays", lang)
        if not column and mine_grid.shape[0] == 1 and mine_grid.shape[1] > _LONG and mine_grid.dtype.kind in "iuf":
            # a long row of numbers, such as a prediction for each of 206 penguins, does not fit as cells: as dots
            dots = _dot_rows(start, mine_grid[0], want_grid, f"{_word('yours-code', lang)}: {what}", lang)
            if dots:
                return _figure(heading, dots, alt=f"{what} = {_short(got)}")
        if column:
            mine_grid = mine_grid.T if mine_grid.shape[0] == 1 else mine_grid
            want_grid = want_grid.T if want_grid is not None and want_grid.shape[0] == 1 else want_grid
        same = want_grid is not None and mine_grid.shape == want_grid.shape and all(
            _same_cell(x, y) for x, y in zip(mine_grid.ravel(), want_grid.ravel()))
        pieces = []
        for name, value in start:
            g = _as_grid(value)
            if g is not None and g.size > 1:
                pieces.append(_grid_svg(f"{name}  {_shape(value)}", g, shade=shade))
        arrow = '<span aria-hidden="true" style="align-self:center;color:var(--ea-ink-3);font-size:1.2rem">→</span>'
        mine_label = f"{_word('yours-code', lang)}: {what}  {_shape(got)}"
        result_pieces = [_grid_svg(mine_label, mine_grid, None if same else want_grid, shade=shade)]
        if not same and want_grid is not None:
            result_pieces.append(_grid_svg(f"{_word('yours-right', lang)}  {_shape(want)}", want_grid, mine_grid, shade=shade))
        body = (f'<div style="display:flex;flex-wrap:wrap;gap:0.5rem 1.1rem;align-items:flex-start">'
                f'{"".join(pieces)}{arrow if pieces else ""}{"".join(result_pieces)}</div>')
        return _figure(heading, body, alt=f"{what} = {_short(got)}")
    return picture


_LONG = 24   # a row longer than this is drawn as dots, not as cells


def _dot_rows(start, mine, want_grid, label: str, lang: str) -> str:
    """A long row of numbers as dots, one per place along the row: the rows the code starts from with the same
    length in the colour of data, the learner's in the colour of a prediction, and the right one, where it differs,
    as open grey dots."""
    import numpy as np
    n = len(mine)
    given = [(name, g[0].astype(float)) for name, value in start
             if (g := _as_grid(value)) is not None and g.shape == (1, n) and g.dtype.kind in "iuf"]
    want = want_grid[0].astype(float) if want_grid is not None and want_grid.shape == (1, n) and want_grid.dtype.kind in "iuf" else None
    differs = want is not None and not all(_same_cell(x, y) for x, y in zip(mine, want))
    rows = [v for _, v in given] + [mine.astype(float)] + ([want] if differs else [])
    every = np.concatenate(rows)
    every = every[np.isfinite(every)]
    if not every.size:
        return ""
    lo, hi = float(every.min()), float(every.max())
    pad = 0.06 * (hi - lo or 1)
    panel = chart.Panel(x=(0, max(n - 1, 1)), y=(lo - pad, hi + pad), xlabel=_word("yours-place", lang))
    at = list(range(n))
    r = 2.2 if n > 80 else 3
    for _, v in given:
        panel.points(at, v, role="data", r=r)
    if differs:
        panel.points(at, want, role="ink-3", r=r + 0.6, hollow=True)
    panel.points(at, mine.astype(float), role="pred", r=r)

    def entry(role: str, words: str, hollow: bool = False) -> str:
        dot = (f'<circle cx="6" cy="6" r="3.6" style="fill:var(--ea-page);stroke:var(--ea-{role});stroke-width:1.5"/>' if hollow
               else f'<circle cx="6" cy="6" r="3.6" style="fill:var(--ea-{role})"/>')
        return f'<span><svg width="12" height="12" aria-hidden="true">{dot}</svg> {html.escape(words)}</span>'
    key = "".join([entry("data", f"{name}  {_shape(value)}") for name, value in start
                   if any(name == g for g, _ in given)] + [entry("pred", f"{label}  ({n},)")]
                  + ([entry("ink-3", _word("yours-right", lang), hollow=True)] if differs else []))
    return (chart.figure([panel], alt=label, width=460, height=230)
            + f'<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);display:flex;gap:1.2rem;'
              f'flex-wrap:wrap;margin-top:0.1rem">{key}</div>')


# --- tangent: a slope the code measured, as a line along the curve ----------------------------------------------

def tangent(x: str, f, *, slope: str | None = None, span: float = 1.0, role: str = "loss", title: Text | None = None,
            xlabel: Text | str = "", ylabel: Text | str = ""):
    """A slope the learner's code works out at one point, drawn as a straight line through that point of the curve,
    in the colour of a gradient, with the right slope dashed: a slope that is right lies along the curve. `x` is the
    name of the point in the code, `f(names, v)` the curve at v with the names the code made, such as
    `lambda ns, b: ns["loss"](ns["w"], b)`; the slope is the value of the last line, or the name `slope`. The curve
    is drawn from x - span to x + span, in the colour of `role`."""
    def picture(seen: checking.Seen) -> str:
        mine = seen.mine
        if mine is None:
            return ""
        ns = mine.names
        try:
            x0 = float(ns[x])
            k = float(ns[slope] if slope else mine.value)
            right = float(seen.want.names[slope] if slope else seen.want.value)
            xs = [x0 - span + 2 * span * i / 60 for i in range(61)]
            ys = checking.with_time_limit(lambda: [float(f(ns, v)) for v in xs])
            y0 = float(f(ns, x0))
        except Exception:
            return ""
        if not all(math.isfinite(v) for v in ys + [k, right, y0]):
            return ""
        lo, hi = min(ys), max(ys)
        pad = 0.15 * (hi - lo or 1)
        lo, hi = lo - pad, hi + pad
        lang = seen.lang
        panel = chart.Panel(x=(xs[0], xs[-1]), y=(lo, hi), xlabel=seen.word(xlabel) or x, ylabel=seen.word(ylabel))
        panel.line(xs, ys, role=role)
        a, b = xs[0], xs[-1]
        panel.line([a, b], [y0 + k * (a - x0), y0 + k * (b - x0)], role="grad", width=2.4)
        panel.line([a, b], [y0 + right * (a - x0), y0 + right * (b - x0)], role="ink-3", width=2, dashed=True)
        panel.points([x0], [y0], role=role, r=5)
        heading = seen.word(title) if title is not None else _word("yours-tangent", lang)
        key = (f'<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);display:flex;gap:1.2rem;'
               f'flex-wrap:wrap;margin-top:0.1rem">'
               f'<span><svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" style="stroke:var(--ea-grad);'
               f'stroke-width:2.4;stroke-linecap:round"/></svg> {html.escape(_word("yours-code", lang))}: {html.escape(_cell(k))}</span>'
               f'<span><svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" style="stroke:var(--ea-ink-3);'
               f'stroke-width:2;stroke-dasharray:5 5"/></svg> {html.escape(_word("yours-right", lang))}</span></div>')
        return _figure(heading, chart.figure([panel], alt=heading, width=420, height=250) + key)
    return picture


# --- vectors: pairs of numbers as arrows in the plane -------------------------------------------------------------

def vectors(call: int = 0, *, role: str = "hidden", title: Text | None = None):
    """What a function does to vectors of two numbers, as arrows from the origin: the vectors it is given in grey,
    what it gives back in the colour of `role`, and the right ones dashed. A flat array [x0, y0, x1, y1, ...] counts
    as pairs, as does an array of shape (n, 2)."""
    def pairs(value):
        g = _as_grid(value)
        if g is None or g.dtype.kind not in "iuf":
            return None
        flat = [float(v) for v in g.ravel()]
        if len(flat) % 2 or not flat:
            return None
        return list(zip(flat[0::2], flat[1::2]))

    def picture(seen: checking.Seen) -> str:
        ex = seen.exercise
        mine = seen.mine
        if mine is None:
            return ""
        case = (list(ex.visible) + list(ex.hidden))[call]
        fn, right = mine.names.get(ex.function), seen.want.names.get(ex.function)
        if not callable(fn) or not callable(right):
            return ""
        try:
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                got = checking.with_time_limit(lambda: fn(*(checking._copy(a) for a in case)))
        except Exception:
            return ""
        want = right(*(checking._copy(a) for a in case))
        given, mine_p, want_p = pairs(case[0]), pairs(got), pairs(want)
        if mine_p is None or want_p is None:
            return ""
        every = (given or []) + mine_p + want_p
        r = max([abs(v) for p in every for v in p] + [1]) * 1.15
        panel = chart.Panel(x=(-r, r), y=(-r, r))
        for p in given or []:
            panel.arrow(0, 0, *p, role="data", width=1.6)
        for p in mine_p:
            panel.arrow(0, 0, *p, role=role)
        for p in want_p:
            panel.line([0, p[0]], [0, p[1]], role="ink-3", width=2, dashed=True)
        lang = seen.lang
        call_text = ex._call_text(case)
        heading = seen.word(title) if title is not None else _word(
            "yours-vectors", lang, call=call_text if len(call_text) <= 34 else f"{ex.function}(…)")
        key = (f'<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);display:flex;gap:1.2rem;'
               f'flex-wrap:wrap;margin-top:0.1rem">'
               f'<span><svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" style="stroke:var(--ea-data);'
               f'stroke-width:1.6"/></svg> {html.escape(_word("yours-given", lang))}</span>'
               f'<span><svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" style="stroke:var(--ea-{role});'
               f'stroke-width:2.4"/></svg> {html.escape(_word("yours-code", lang))}</span>'
               f'<span><svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" style="stroke:var(--ea-ink-3);'
               f'stroke-width:2;stroke-dasharray:5 5"/></svg> {html.escape(_word("yours-right", lang))}</span></div>')
        return _figure(heading, chart.figure([panel], alt=heading, width=300, height=300) + key)
    return picture


# --- descent: the steps of the learner's training, on the loss ----------------------------------------------------

def _copied(v):
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return v.copy()
    except ImportError:   # pragma: no cover
        pass
    return copy.deepcopy(v) if isinstance(v, (list, dict)) else v


def _path(run, param: str, loss, keep):
    """Run the learner's code (`run(trace)`) and write down, each time `param` changes, (its value, loss(names))."""
    path: list[tuple] = []
    seen: list = [None, None]   # the object `param` named at the last line, and how it looked

    def look(v) -> str:
        text = _short(v)
        return repr(v) if len(text) >= 38 else text   # a value cut short could hide a change

    def local(frame, event, arg):
        if event == "line" and len(path) < 5000:
            ns = frame.f_globals if frame.f_code.co_name == "<module>" else frame.f_locals
            if param in ns:
                v = ns[param]
                # the same number or tensor as at the last line has not changed; an array can change in place
                if path and v is seen[0] and not hasattr(v, "shape"):
                    return local
                now = look(v)
                if not path or now != seen[1]:
                    try:
                        value = float(loss(ns))
                    except Exception:
                        value = None
                    path.append((_copied(v), value))
                seen[0], seen[1] = v, now
        return local

    def tracer(frame, event, arg):
        return local if event == "call" and keep(frame.f_code) else None
    run(tracer)
    return path


def descent(param: str, loss, *, over=None, call: int | None = None, span: float | None = None,
            title: Text | None = None, xlabel: Text | str = ""):
    """The learner's gradient descent, step by step: each time `param` changes in their code, the loss at that moment
    (`loss(names)`, with the names of the code or of the function as they are then). With `over(names, v)`, the loss
    for every value v of one parameter, the steps stand as orange dots on that curve; otherwise the loss after each
    step is a line, with the right run dashed. For a function exercise, `call` is the case it follows."""
    def record(seen, text):
        ex = seen.exercise
        if call is None:
            runs = []
            path = _path(lambda tr: runs.append(checking.run_code(text, seen.names, trace=tr)), param, loss,
                         lambda code: code.co_filename == "<your code>")
            return None if not runs or runs[0].error is not None or runs[0].timeout else path
        case = (list(ex.visible) + list(ex.hidden))[call]
        made = checking.run_code(text, seen.names)
        fn = made.names.get(ex.function)
        if made.error is not None or not callable(fn):
            return None

        def go(tr):
            before = sys.gettrace()
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sys.settrace(tr)
                try:
                    fn(*(checking._copy(a) for a in case))
                except Exception:
                    pass
                finally:
                    sys.settrace(before)
        return _path(lambda tr: checking.with_time_limit(lambda: go(tr)), param, loss,
                     lambda code: code.co_filename == "<your code>")

    def picture(seen: checking.Seen) -> str:
        if not isinstance(seen.text, str):
            return ""
        lang = seen.lang
        # the learner's code runs once, traced; a run that stops with an error has no last step to show
        try:
            mine = checking.with_time_limit(lambda: record(seen, seen.text))
        except Exception:
            return ""
        if mine is None:
            return ""
        right = seen.remember(("descent", param, call), lambda: record(seen, seen.exercise.solution)) or []
        mine = [(v, l) for v, l in mine if l is not None and math.isfinite(l)]
        right = [(v, l) for v, l in right if l is not None and math.isfinite(l)]
        if len(mine) < 2:
            return ""
        heading = seen.word(title) if title is not None else _word("yours-descent", lang)
        if over is not None:
            vs = [float(v) for v, _ in mine]
            centre = [float(v) for v, _ in right] or vs
            lo = min(vs + centre)
            hi = max(vs + centre)
            if span is not None:
                lo, hi = min(lo, centre[-1] - span), max(hi, centre[-1] + span)
            pad = 0.08 * (hi - lo or 1)
            lo, hi = lo - pad, hi + pad
            xs = [lo + (hi - lo) * i / 80 for i in range(81)]
            ns = seen.mine.names
            try:
                ys = [float(over(ns, v)) for v in xs]
            except Exception:
                return ""
            ls = [l for _, l in mine]
            end = _cell(vs[-1]).replace(".", ",") if lang == "nl" else _cell(vs[-1])   # in a sentence: Dutch writes 57,7
            top = max(ys + ls)
            bottom = min(ys + ls)
            panel = chart.Panel(x=(lo, hi), y=(bottom - 0.05 * (top - bottom), top + 0.05 * (top - bottom)),
                                xlabel=seen.word(xlabel) or param, ylabel="loss")
            panel.line(xs, ys, role="loss")
            panel.line(vs, ls, role="param", width=1.2)
            panel.points(vs, ls, role="param", r=3.6)
            panel.points(vs[-1:], ls[-1:], role="param", r=6)
            key = (f'<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.1rem">'
                   f'{html.escape(_word("yours-descent-steps", lang, n=len(vs) - 1, param=param, value=end))}</div>')
            return _figure(heading, chart.figure([panel], alt=heading, width=460, height=260) + key)
        ls = [l for _, l in mine]
        rs = [l for _, l in right]
        top, bottom = max(ls + rs), min(ls + rs)
        # a loss that drops by a factor of a hundred or more in the first steps leaves a flat line on a plain axis;
        # on a log scale each step down by the same factor is the same distance, so the whole run stays readable
        log = bottom > 0 and top / bottom > 30
        panel = chart.Panel(x=(0, max(len(ls), len(rs), 2) - 1), y=(bottom / 1.3, top * 1.3) if log else (0, top * 1.05 or 1),
                            xlabel=_word("yours-step", lang), ylabel="loss", ylog=log)
        if rs:
            panel.line(range(len(rs)), rs, role="ink-3", width=2, dashed=True)
        panel.line(range(len(ls)), ls, role="loss")
        key = (f'<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);display:flex;gap:1.2rem;'
               f'flex-wrap:wrap;margin-top:0.1rem">'
               f'<span><svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" style="stroke:var(--ea-loss);'
               f'stroke-width:2.4;stroke-linecap:round"/></svg> {html.escape(_word("yours-code", lang))}</span>'
               f'<span><svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" style="stroke:var(--ea-ink-3);'
               f'stroke-width:2;stroke-dasharray:5 5"/></svg> {html.escape(_word("yours-right", lang))}</span></div>')
        return _figure(heading, chart.figure([panel], alt=heading, width=460, height=250) + key)
    return picture


# --- series: lists of numbers the code made, as lines ---------------------------------------------------------------

def series(call: int | None = None, *, of: tuple[str, ...] = (), xs: list | None = None, names: list[Text | str] | None = None,
           role: str = "loss", xlabel: Text | str = "", ylabel: Text | str = "", title: Text | None = None):
    """A list of numbers the learner's code makes, such as a validation loss for each degree, as a line with a dot at
    each value, with the right one dashed and grey; a pair of lists, such as (train_mses, val_mses), as two lines, the
    second with open dots, each named in the key by `names`. `xs` are the places along the x-axis (default 0, 1, 2...).
    A code box shows the value of its last line, or the lists it made under the names in `of`, such as
    ("train_losses", "val_losses")."""
    def lines(value):
        g = _as_grid(value)
        if g is None or g.dtype.kind not in "iuf" or g.size < 2:
            return None
        return [[float(v) for v in row] for row in g]

    def picture(seen: checking.Seen) -> str:
        mine = seen.mine
        if mine is None:
            return ""
        if call is None and of:
            if not all(k in mine.names for k in of):
                return ""
            got = [mine.names[k] for k in of] if len(of) > 1 else mine.names[of[0]]
            want = [seen.want.names.get(k) for k in of] if len(of) > 1 else seen.want.names.get(of[0])
        else:
            out = _outcome(seen, call)
            if out is None:
                return ""
            got, want = out[0], out[1]
        mine_l, want_l = lines(got), lines(want)
        if mine_l is None:
            return ""
        every = [v for row in mine_l + (want_l or []) for v in row if math.isfinite(v)]
        if not every:
            return ""
        n = max(len(r) for r in mine_l + (want_l or []))
        at = list(xs) if xs is not None and len(xs) >= n else list(range(n))
        lo, hi = min(every + [0]), max(every)
        panel = chart.Panel(x=(at[0], at[n - 1] if n > 1 else at[0] + 1), y=(lo, hi + 0.08 * (hi - lo or 1)),
                            xlabel=seen.word(xlabel), ylabel=seen.word(ylabel),
                            xticks=at[:n] if n <= 12 else None)
        for row in want_l or []:
            panel.line(at[:len(row)], row, role="ink-3", width=2, dashed=True)
        lang = seen.lang
        for i, row in enumerate(mine_l):
            ok = [(a, v) for a, v in zip(at, row) if math.isfinite(v)]
            panel.line([a for a, _ in ok], [v for _, v in ok], role=role)
            panel.points([a for a, _ in ok], [v for _, v in ok], role=role, r=4, hollow=i == 1)
        heading = seen.word(title) if title is not None else _word("yours-series", lang)
        # each line of the learner's by its name in the key, with its dot: lines that end close together would hide
        # names written at their ends
        named = [seen.word(names[i]) if names and i < len(names) else _word("yours-code", lang) for i in range(len(mine_l))]
        dot = ['<circle cx="13" cy="5" r="3.4" style="fill:var(--ea-{0})"/>',
               '<circle cx="13" cy="5" r="3.4" style="fill:var(--ea-page);stroke:var(--ea-{0});stroke-width:1.6"/>']
        key = (f'<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);display:flex;gap:1.2rem;'
               f'flex-wrap:wrap;margin-top:0.1rem">'
               + "".join(f'<span><svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" '
                         f'style="stroke:var(--ea-{role});stroke-width:2.4;stroke-linecap:round"/>{dot[min(i, 1)].format(role)}'
                         f'</svg> {html.escape(word)}</span>' for i, word in enumerate(named[:2])) +
               f'<span><svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" style="stroke:var(--ea-ink-3);'
               f'stroke-width:2;stroke-dasharray:5 5"/></svg> {html.escape(_word("yours-right", lang))}</span></div>')
        return _figure(heading, chart.figure([panel], alt=heading, width=460, height=250) + key)
    return picture


# --- table: a table of data the code made ---------------------------------------------------------------------------

def _html_table(df, label: str, lang: str, other=None) -> str:
    """The first rows of a pandas table or series as an HTML table, with the name and shape above it. Cells that
    differ from `other` (a table of the same shape) are bold with a border."""
    import pandas as pd
    if isinstance(df, pd.Series):
        df = df.to_frame(name=df.name if df.name is not None else "")
    rows = df.head(8)
    cell = "padding:0.2rem 0.55rem;border-top:1px solid var(--ea-line);text-align:right;white-space:nowrap"
    head = "padding:0.25rem 0.55rem;font-weight:600;color:var(--ea-ink-2);text-align:right;white-space:nowrap"
    out = ["<tr>", f'<th style="{head}"></th>']
    out += [f'<th style="{head}">{html.escape(str(c))}</th>' for c in rows.columns]
    out.append("</tr>")
    same_shape = other is not None and getattr(other, "shape", None) == df.shape and list(other.columns) == list(df.columns)
    for i, (index, row) in enumerate(rows.iterrows()):
        out.append(f'<tr><td style="{cell};color:var(--ea-ink-3)">{html.escape(str(index))}</td>')
        for j, v in enumerate(row):
            differs = same_shape and not _same_cell(v, other.iloc[i, j])
            look = "font-weight:600;outline:2px solid var(--ea-ink);outline-offset:-2px" if differs else ""
            text = _cell(v) if not isinstance(v, str) else v
            out.append(f'<td style="{cell};color:var(--ea-ink);{look}">{html.escape(_cut(str(text), 24))}</td>')
        out.append("</tr>")
    more = ""
    if len(df) > len(rows):
        more = (f'<div style="font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-3);padding:0.2rem 0.55rem">'
                f'{html.escape(_word("yours-more-rows", lang, n=len(df) - len(rows)))}</div>')
    return (f'<div style="flex:0 1 auto;min-width:0"><div style="{_MONO};font-size:13px;color:var(--ea-ink-2);margin-bottom:0.2rem">'
            f'{html.escape(label)}</div><div style="overflow-x:auto;border:1px solid var(--ea-line);border-radius:6px">'
            f'<table style="border-collapse:collapse;{_MONO};font-size:0.8rem">{"".join(out)}</table>{more}</div></div>')


def table(call: int | None = None, *, title: Text | None = None):
    """A pandas table or series the learner's code makes (the value of its last line, or what a function gives back
    for case `call`), its first rows, with its shape; beside it the right one when it differs, with the cells that
    differ marked."""
    def picture(seen: checking.Seen) -> str:
        import pandas as pd
        ex = seen.exercise
        mine = seen.mine
        if mine is None:
            return ""
        if call is None:
            got, want = mine.value, seen.want.value
        else:
            case = (list(ex.visible) + list(ex.hidden))[call]
            fn, right = mine.names.get(ex.function), seen.want.names.get(ex.function)
            if not callable(fn) or not callable(right):
                return ""
            try:
                with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    got = checking.with_time_limit(lambda: fn(*(checking._copy(a) for a in case)))
            except Exception:
                return ""
            want = right(*(checking._copy(a) for a in case))
        if not isinstance(got, (pd.DataFrame, pd.Series)):
            return ""
        frame = lambda v: v.to_frame(name=v.name if v.name is not None else "") if isinstance(v, pd.Series) else v   # noqa: E731
        g, w = frame(got), frame(want) if isinstance(want, (pd.DataFrame, pd.Series)) else None
        same = w is not None and g.shape == w.shape and list(g.columns) == list(w.columns) and all(
            _same_cell(a, b) for a, b in zip(g.to_numpy().ravel(), w.to_numpy().ravel()))
        lang = seen.lang
        parts = [_html_table(g, f"{_word('yours-code', lang)}  {_shape(g)}", lang, None if same else w)]
        if not same and w is not None:
            parts.append(_html_table(w, f"{_word('yours-right', lang)}  {_shape(w)}", lang, g))
        heading = seen.word(title) if title is not None else _word("yours-table", lang)
        body = f'<div style="display:flex;flex-wrap:wrap;gap:0.6rem 1.1rem;align-items:flex-start">{"".join(parts)}</div>'
        return _figure(heading, body)
    return picture


def _outcome(seen: checking.Seen, call: int | None, show: tuple[str, ...] = (), result: str | None = None):
    """What the learner's code makes and what it should make, with what it starts from: (got, want, start, what).
    A code box gives the value of its last line, or of the name `result`, and starts from the names in `show`; a
    function exercise gives what it gives back for case `call`, and starts from that case's arguments. `what` says
    in a few signs where the value comes from. None when the learner's code does not run or stops with an error.
    A class exercise (ClassBox) runs scenario `call` after the learner's code, with `result` as its last line if given
    (an expression, such as `class_map(scene(18))`), and starts from the names in `show` that the scenario makes."""
    ex = seen.exercise
    mine = seen.mine
    if mine is None:
        return None
    if isinstance(ex, checking.ClassBox) and call is not None:
        lines = ex.scenarios[call].rstrip().split("\n")
        last = result or lines[-1].strip()
        setup = "\n".join(lines[:-1])

        def made(code: str):
            return checking.run_code(code + "\n\n" + setup + "\n" + last, seen.names)
        run = made(seen.text)
        if run.error is not None or run.timeout or not run.has_value:
            return None
        right = seen.remember(("scenario", call, last), lambda: made(ex.solution))
        start = [(k, run.names[k]) for k in show if k in run.names]
        return run.value, right.value, start, _cut(last, 34)
    if call is None:
        got = mine.names.get(result) if result else mine.value
        want = seen.want.names.get(result) if result else seen.want.value
        start = [(k, mine.names[k]) for k in show if k in mine.names]
        what = result or (seen.text.strip().splitlines() or [""])[-1].strip()
        return got, want, start, _cut(what, 34)
    case = (list(ex.visible) + list(ex.hidden))[call]
    fn, right = mine.names.get(ex.function), seen.want.names.get(ex.function)
    if not callable(fn) or not callable(right):
        return None
    try:
        params = list(inspect.signature(right).parameters)
    except (TypeError, ValueError):
        params = []
    start = [(params[i] if i < len(params) else f"#{i}", a) for i, a in enumerate(case)]
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = checking.with_time_limit(lambda: fn(*(checking._copy(a) for a in case)))
    except Exception:
        return None
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = right(*(checking._copy(a) for a in case))
    what = ex._call_text(case)
    return got, want, start, (what if len(what) <= 34 else f"{ex.function}(…)")


# --- histogram: many numbers, by how often they fall in each stretch ----------------------------------------------

def histogram(call: int | None = None, *, bins: int = 24, role: str = "data", xlabel: Text | str = "",
              title: Text | None = None):
    """Many numbers the learner's code makes, such as 1000 bootstrap means, as a histogram: how many fall in each
    stretch, as bars in the colour of `role`, with the right ones as a dashed outline over them. A code box shows the
    value of its last line, a function exercise what it gives back for case `call`."""
    import numpy as np

    def values(v):
        a = _as_array(v)
        if a is None or a.size < 2:
            return None
        a = a.astype(float).ravel()
        return a[np.isfinite(a)]

    def picture(seen: checking.Seen) -> str:
        out = _outcome(seen, call)
        if out is None:
            return ""
        got, want, _, what = out
        mine, right = values(got), values(want)
        if mine is None or not mine.size:
            return ""
        lang = seen.lang
        both = np.concatenate([mine] + ([right] if right is not None and right.size else []))
        lo, hi = float(both.min()), float(both.max())
        if hi - lo < 1e-12:
            lo, hi = lo - 0.5, hi + 0.5
        edges = np.linspace(lo, hi, bins + 1)
        counts = np.histogram(mine, edges)[0]
        rights = np.histogram(right, edges)[0] if right is not None and right.size else None
        top = max(int(counts.max()), int(rights.max()) if rights is not None else 0, 1)
        panel = chart.Panel(x=(lo, hi), y=(0, top * 1.08), xlabel=seen.word(xlabel), ylabel=_word("yours-how-many", lang))
        for i, c in enumerate(counts):
            if c:
                panel.rect(edges[i], 0, edges[i + 1], int(c), role=role)
        same = rights is not None and bool((counts == rights).all())
        if rights is not None and not same:
            xs, ys = [], []
            for i, c in enumerate(rights):   # the right counts as steps along the tops of their bars
                xs += [edges[i], edges[i + 1]]
                ys += [int(c), int(c)]
            panel.line(xs, ys, role="ink-3", width=2, dashed=True)
        heading = seen.word(title) if title is not None else _word("yours-histogram", lang)
        key = (f'<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);display:flex;gap:1.2rem;'
               f'flex-wrap:wrap;margin-top:0.1rem"><span><svg width="12" height="12" aria-hidden="true"><rect x="1" y="1" '
               f'width="10" height="10" style="fill:var(--ea-{role})"/></svg> {html.escape(_word("yours-code", lang))}: '
               f'{html.escape(what)}  ({mine.size})</span>'
               + ("" if same or rights is None else
                  f'<span><svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" style="stroke:var(--ea-ink-3);'
                  f'stroke-width:2;stroke-dasharray:5 5"/></svg> {html.escape(_word("yours-right", lang))}</span>') + '</div>')
        return _figure(heading, chart.figure([panel], alt=heading, width=460, height=240) + key)
    return picture


# --- image: an array that is a picture, as that picture ----------------------------------------------------------

def _png(value, size: int) -> str | None:
    """An array as an <img> of itself: 2-D in grey, 3-D with 3 or 4 channels in colour. Numbers from 0 to 1, colours
    and whole numbers from 0 to 255 stand as they are, except labels (whole numbers up to 15, such as classes 0 to 3),
    which spread from black to white, as other numbers do from the smallest to the largest.
    A small image is drawn larger, with sharp pixels, so that each pixel can be seen."""
    import base64
    import numpy as np
    try:
        from PIL import Image
    except ImportError:   # pragma: no cover - Pillow comes with the course
        return None
    if type(value).__module__.startswith("torch"):
        try:
            value = value.detach().cpu().numpy()
        except Exception:
            return None
    try:
        a = np.asarray(value)
    except Exception:
        return None
    if a.dtype.kind == "b":
        a = a.astype(float)
    if a.dtype.kind not in "iuf" or a.size == 0 or not (a.ndim == 2 or a.ndim == 3 and a.shape[2] in (1, 3, 4)):
        return None
    colour = a.ndim == 3 and a.shape[2] in (3, 4)
    whole = a.dtype.kind in "iu"
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    a = np.nan_to_num(a.astype(float))
    lo, hi = float(a.min()), float(a.max())
    if lo >= 0 and hi <= 1:
        a = a * 255
    elif not (lo >= 0 and hi <= 255 and (colour or whole and hi > 15)):   # colours from 0 to 255 stay the colours they are
        a = (a - lo) / ((hi - lo) or 1) * 255
    img = Image.fromarray(np.clip(a, 0, 255).round().astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    h, w = a.shape[:2]
    scale = size / max(h, w)
    sharp = "image-rendering:pixelated;" if scale >= 2 else ""
    return (f'<img src="data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}" alt="" '
            f'width="{round(w * scale)}" height="{round(h * scale)}" style="{sharp}display:block;max-width:100%;height:auto;'
            f'border:1px solid var(--ea-line);background:var(--ea-inset)">')


def image(call: int | None = None, *, show: tuple[str, ...] = (), result: str | None = None, size: int = 160,
          title: Text | None = None):
    """What the learner's code makes of an image, as images: first the ones it starts from (`show`, or the arguments
    of case `call` of a function exercise that are images), then the result (`result`, a name, or the value of the
    last line), and the right one beside it where the learner's differs. Each with its name and shape above it."""
    def picture(seen: checking.Seen) -> str:
        out = _outcome(seen, call, show, result)
        if out is None:
            return ""
        got, want, start, what = out
        lang = seen.lang
        if _images(got):
            return _image_list(seen, got, want, what, size, title)
        mine_img = _png(got, size)
        if mine_img is None:
            return ""

        def piece(label: str, img: str) -> str:
            return (f'<div style="display:flex;flex-direction:column;gap:0.2rem">'
                    f'<span style="{_MONO};font-size:0.8rem;color:var(--ea-ink-2)">{html.escape(label)}</span>{img}</div>')
        pieces = [piece(f"{name}  {_shape(value)}", img) for name, value in start if (img := _png(value, size)) is not None]
        results = [piece(f"{_word('yours-code', lang)}: {what}  {_shape(got)}", mine_img)]
        g, w = _as_array(got), _as_array(want)
        same = g is not None and w is not None and g.shape == w.shape and bool(
            (abs(g.astype(float) - w.astype(float)) <= 1e-6 + 1e-4 * abs(w.astype(float))).all())
        if not same and (right_img := _png(want, size)) is not None:
            results.append(piece(f"{_word('yours-right', lang)}  {_shape(want)}", right_img))
        arrow = '<span aria-hidden="true" style="align-self:center;color:var(--ea-ink-3);font-size:1.2rem">→</span>'
        body = (f'<div style="display:flex;flex-wrap:wrap;gap:0.5rem 1.1rem;align-items:flex-start">'
                f'{"".join(pieces)}{arrow if pieces else ""}{"".join(results)}</div>')
        heading = seen.word(title) if title is not None else _word("yours-image", lang)
        return _figure(heading, body, alt=f"{what}  {_shape(got)}")
    return picture


def _images(value) -> bool:
    """A list or tuple of 2 to 4 images, such as the levels of a feature pyramid."""
    return (isinstance(value, (list, tuple)) and 1 < len(value) <= 4
            and all((a := _as_array(v)) is not None and (a.ndim == 2 or a.ndim == 3 and a.shape[2] in (1, 3, 4))
                    for v in value))


def _image_list(seen: checking.Seen, got, want, what: str, size: int, title: Text | None) -> str:
    """The images of image(), for a list: each as large as it is next to the largest, the right one under it where
    the learner's differs."""
    lang = seen.lang
    biggest = max(max(_as_array(v).shape[:2]) for v in got)
    columns = []
    for i, value in enumerate(got):
        a = _as_array(value)
        px = max(40, round(size * max(a.shape[:2]) / biggest))
        pieces = [f'<span style="{_MONO};font-size:0.8rem;color:var(--ea-ink-2)">[{i}]  {_shape(value)}</span>{_png(value, px)}']
        other = _as_array(want[i]) if _images(want) and i < len(want) else None
        if other is not None and (other.shape != a.shape or not bool(
                (abs(a.astype(float) - other.astype(float)) <= 1e-6 + 1e-4 * abs(other.astype(float))).all())):
            pieces.append(f'<span style="{_MONO};font-size:0.8rem;color:var(--ea-ink-2)">{_word("yours-right", lang)}</span>'
                          f'{_png(want[i], px)}')
        columns.append(f'<div style="display:flex;flex-direction:column;gap:0.2rem">{"".join(pieces)}</div>')
    body = (f'<div style="{_MONO};font-size:0.8rem;color:var(--ea-ink-2);margin-bottom:0.3rem">{_word("yours-code", lang)}: '
            f'{html.escape(what)}</div><div style="display:flex;flex-wrap:wrap;gap:0.5rem 1.1rem;align-items:flex-start">'
            f'{"".join(columns)}</div>')
    heading = seen.word(title) if title is not None else _word("yours-image", lang)
    return _figure(heading, body, alt=what)


def _as_array(value):
    import numpy as np
    if type(value).__module__.startswith("torch"):
        try:
            value = value.detach().cpu().numpy()
        except Exception:
            return None
    try:
        a = np.asarray(value)
    except Exception:
        return None
    return a if a.dtype.kind in "biuf" else None


# --- search: a search the learner wrote, in its maze ---------------------------------------------------------------

def search(call: int = 0, *, title: Text | None = None):
    """A search the learner wrote, on the maze of case `call` (its first argument, a list of strings): the states it
    expanded in grey and the path it found as a line, as draw.maze_grids draws them, beside what the course's search
    expands and finds when that differs. The function gives back (path, done), as in the search lessons."""
    def cells(maze, value) -> list[tuple[int, int]]:
        out = []
        for s in value or ():
            if (isinstance(s, tuple) and len(s) == 2 and all(isinstance(v, int) for v in s)
                    and 0 <= s[0] < len(maze) and 0 <= s[1] < len(maze[0])):
                out.append(s)
        return out

    def found(fn, case):
        out = fn(*(checking._copy(a) for a in case))
        if not isinstance(out, tuple) or len(out) != 2:
            return None
        path, done = out
        return (cells(case[0], path) if isinstance(path, list) else None), set(cells(case[0], done))

    def picture(seen: checking.Seen) -> str:
        from everything_ai import draw
        ex = seen.exercise
        mine = seen.mine
        if mine is None:
            return ""
        case = (list(ex.visible) + list(ex.hidden))[call]
        fn, right = mine.names.get(ex.function), seen.want.names.get(ex.function)
        if not callable(fn) or not callable(right):
            return ""
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                got = checking.with_time_limit(lambda: found(fn, case))
        except Exception:
            return ""
        want = found(right, case)
        if got is None:
            return ""
        lang = seen.lang
        maze = [str(r) for r in case[0]]

        def count(n: int, what: str) -> str:
            return _word(f"yours-one-{what}", lang) if n == 1 else _word(f"yours-n-{what}s", lang, n=n)

        def words(who: str, out) -> str:
            path, done = out
            if path:
                return _word("yours-search-path", lang, who=who, steps=count(len(path) - 1, "step"),
                             done=count(len(done), "state"))
            return _word("yours-search-none", lang, who=who, done=count(len(done), "state"))
        # every shortest path is right: the course's stands beside the learner's only when the length of the path or
        # the number of states it expanded differs
        same = want is not None and len(got[1]) == len(want[1]) and len(got[0] or ()) == len(want[0] or ())
        shown = [(_word("yours-code", lang), got)] + ([] if same or want is None else [(_word("yours-right", lang), want)])
        panels = [{"rows": maze, "title": who, "done": sorted(out[1]), "path": out[0] or None, "index": len(shown) == 1}
                  for who, out in shown]
        heading = seen.word(title) if title is not None else _word("yours-search", lang)
        alt = "; ".join(words(who, out) for who, out in shown)
        grid = draw.maze_grids(panels, alt=alt, cell=24 if len(shown) > 1 else 26).text
        key = "".join(f'<div>{html.escape(words(who, out))}</div>' for who, out in shown)
        return _figure(heading, grid + f'<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);'
                                      f'margin-top:0.2rem">{key}</div>', alt=alt)
    return picture



# --- grid: a function of two whole numbers, as a table of what it gives -------------------------------------------

def grid(function: str, rows: int, cols: int, *, at=None, shade: bool = True, title: Text | None = None):
    """What the learner's function gives for each row i and column j, such as can_see(kind, i, j) for a mask, as a
    grid of cells, with the right grid beside it where a cell differs; with `shade`, each cell gets a grey wash as
    strong as its number. `at(i, j)` gives the arguments, such as `lambda i, j: ("decoder", i, j)`."""
    import numpy as np
    args = at or (lambda i, j: (i, j))

    def table(fn):
        out = []
        for i in range(rows):
            row = []
            for j in range(cols):
                try:
                    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        v = fn(*copy.deepcopy(args(i, j)))
                except Exception:
                    v = "!"
                row.append(v if isinstance(v, (bool, int, float, str)) or type(v).__module__ == "numpy" else _short(v))
            out.append(row)
        return np.array(out, dtype=object)

    def picture(seen: checking.Seen) -> str:
        mine = seen.mine
        if mine is None:
            return ""
        fn, right = mine.names.get(function), seen.want.names.get(function)
        if not callable(fn) or not callable(right):
            return ""
        got = checking.with_time_limit(lambda: table(fn))
        want = table(right)
        numeric = all(isinstance(v, (bool, int, float)) or type(v).__module__ == "numpy" for v in got.ravel())
        same = all(_same_cell(x, y) for x, y in zip(got.ravel(), want.ravel()))
        lang = seen.lang
        pieces = [_grid_svg(f"{_word('yours-code', lang)}  ({rows}, {cols})", got, None if same else want, shade=shade and numeric)]
        if not same:
            pieces.append(_grid_svg(f"{_word('yours-right', lang)}  ({rows}, {cols})", want, got, shade=shade))
        heading = seen.word(title) if title is not None else _word("yours-grid", lang, function=function)
        body = f'<div style="display:flex;flex-wrap:wrap;gap:0.5rem 1.1rem;align-items:flex-start">{"".join(pieces)}</div>'
        return _figure(heading, body)
    return picture


# --- tokens: a text cut into tokens, each piece in its own box ----------------------------------------------------

def _chips(label: str, pieces: list, other: list | None = None) -> str:
    """Pieces of text (or token ids) side by side as boxes, by turns a little darker and lighter, so that the learner
    sees where one ends and the next begins; a space shows as ·, a new line as ↵. Pieces that differ from `other` at
    the same place get a thick border."""
    out = []
    for i, p in enumerate(pieces[:120]):
        text = str(p) if not isinstance(p, str) else (p.replace(" ", "·").replace("\n", "↵") or "∅")
        differs = other is not None and (i >= len(other) or other[i] != p)
        border = "2px solid var(--ea-ink)" if differs else "1px solid var(--ea-line)"
        out.append(f'<span style="display:inline-block;padding:0.12rem 0.32rem;margin:0 0.12rem 0.25rem 0;border:{border};'
                   f'border-radius:3px;background:color-mix(in srgb,var(--ea-data) {18 if i % 2 else 8}%,var(--ea-page));'
                   f'{_MONO};font-size:0.85rem;color:var(--ea-ink);white-space:pre">{html.escape(text)}</span>')
    more = f'<span style="color:var(--ea-ink-3);font-size:0.85rem"> … {len(pieces) - 120}</span>' if len(pieces) > 120 else ""
    return (f'<div style="margin-bottom:0.35rem"><div style="{_MONO};font-size:0.8rem;color:var(--ea-ink-2);margin-bottom:0.15rem">'
            f'{html.escape(label)}</div><div style="line-height:1.9">{"".join(out)}{more}</div></div>')


def tokens(call: int | None = 0, *, title: Text | None = None):
    """A text cut into tokens: the list of pieces (or ids) the learner's code gives back for case `call`, or the
    value of the last line of a code box, each piece in its own box, with the right pieces under it where they
    differ."""
    def picture(seen: checking.Seen) -> str:
        out = _outcome(seen, call)
        if out is None:
            return ""
        got, want, _, what = out
        if isinstance(got, tuple):
            got = list(got)
        if isinstance(want, tuple):
            want = list(want)
        if not isinstance(got, list) or not got or not all(isinstance(p, (str, int, bytes)) for p in got):
            return ""
        lang = seen.lang
        same = got == want
        body = _chips(f"{_word('yours-code', lang)}: {what}  ({len(got)})", got, None if same or not isinstance(want, list) else want)
        if not same and isinstance(want, list):
            body += _chips(f"{_word('yours-right', lang)}  ({len(want)})", want, got)
        heading = seen.word(title) if title is not None else _word("yours-tokens", lang)
        return _figure(heading, body)
    return picture
