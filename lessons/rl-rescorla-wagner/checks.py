# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rl-rescorla-wagner: the Rescorla-Wagner model gives every signal its own V, and every signal
present in a trial takes the same step, η times the surprise: the food minus what all signals present predict together.
That explains blocking: a light added to a sound that already predicts the food learns almost nothing. The course's dog
with a sound and a light, rw_step and run, the functions the learner writes, the pictures, and the demo.

The rule is that of Rescorla and Wagner (1972), with their α and β taken together as one learning rate η; blocking is
Kamin's (1969), who found it with rats, a noise, a light and a mild shock. The numbers and the pictures are the course's
own."""
import copy
import html
import math

import numpy as np

from everything_ai import live, yours
from everything_ai.check import FunctionBox, Mistake, Timeout

LESSON = "rl-rescorla-wagner"
LANGS = ("en", "nl")
ETA = 0.2        # the dog's learning rate
PHASE = 20       # trials of each part of an experiment
SOUND, LIGHT = 0, 1   # the places of the two signals in an array of V's
FIRST_OPTIONS = (0, 5, 10, 20)   # the demo: trials with the sound alone before the light joins
ETA_OPTIONS = (0.1, 0.2, 0.5)


# --- the functions of the exercises, as the course writes them ----------------------------------------------------------

def rw_step(v, present, food, eta):
    surprise = food - present @ v
    return v + eta * surprise * present


def run(trials, eta):
    v = np.zeros(2)
    seen = [v]
    for present, food in trials:
        v = rw_step(v, present, food, eta)
        seen.append(v)
    return np.array(seen)


# --- the course's experiments -------------------------------------------------------------------------------------------

class Trials(list):
    """The trials of an experiment, each a pair (present, food), that shows its name in a check message, such as
    `blocking`, instead of all its arrays."""

    def __init__(self, values, name: str):
        super().__init__(values)
        self.name = name

    def __deepcopy__(self, memo):
        return Trials(copy.deepcopy(list(self), memo), self.name)

    def __repr__(self) -> str:
        return self.name


def _trial(sound: int, light: int, food: int) -> tuple:
    return (np.array([sound, light]), food)


def blocking_trials(first: int = PHASE, second: int = PHASE) -> list:
    """`first` trials with the sound alone and food, then `second` with the sound and the light together and food."""
    return [_trial(1, 0, 1) for _ in range(first)] + [_trial(1, 1, 1) for _ in range(second)]


BLOCKING = Trials(blocking_trials(), "blocking")
CONTROL = Trials([_trial(0, 0, 0) for _ in range(PHASE)] + [_trial(1, 1, 1) for _ in range(PHASE)], "control")
INHIBITION = Trials([t for _ in range(PHASE) for t in (_trial(1, 0, 1), _trial(1, 1, 0))], "inhibition")
SHORT = Trials([_trial(1, 1, 1), _trial(0, 1, 1), _trial(1, 0, 0)], "short")


def surprises(trials: list, values) -> list:
    """The surprise of every trial: its food minus what the signals present predicted before it."""
    return [float(food - np.asarray(present) @ values[t]) for t, (present, food) in enumerate(trials)]


# --- words and numbers ---------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem;max-width:var(--ea-measure)"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem;max-width:var(--ea-measure)"


def _marked(text_: str) -> str:
    """Escaped text in which `name` is set as code and *word* in italics."""
    out = []
    for k, part in enumerate(html.escape(text_).split("`")):
        if k % 2:
            out.append(f"<code>{part}</code>")
        else:
            out.append("".join(f"<i>{p}</i>" if j % 2 else p for j, p in enumerate(part.split("*"))))
    return "".join(out)


def _note(text_: str, style: str = _UI) -> str:
    return f'<div style="{style}">{_marked(text_)}</div>'


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma and a thousands dot in Dutch, and a real minus sign."""
    out = f"{x:,.{digits}f}"
    if out.startswith("-") and float(out.replace(",", "")) == 0:
        out = out[1:]
    out = out.replace("-", "−")
    return out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out


def short_number(x: float, lang: str, digits: int = 3) -> str:
    """A number rounded to `digits` decimals, without the zeros at its end: 0.6, 0.002, 1."""
    out = number(x, digits, lang)
    point = "," if lang == "nl" else "."
    if point in out:
        out = out.rstrip("0").rstrip(point)
    return "0" if out in ("−0", "") else out


# --- drawing ------------------------------------------------------------------------------------------------------------

_HALO = "paint-order:stroke;stroke:var(--ea-page);stroke-width:3px;stroke-linejoin:round"
TOP = 38   # the room above a picture for its title: a name, and a line under it


def _title(x: float, y: float, words: str) -> str:
    """A title of one or two lines: the part before ": " in bold, the rest in gray under it."""
    name, _, rest = words.partition(": ")
    out = f'<text x="{x}" y="{y}" style="font-size:13.5px;font-weight:600;fill:var(--ea-ink)">{html.escape(name)}</text>'
    if rest:
        out += f'<text x="{x}" y="{y + 16}" style="font-size:12px;fill:var(--ea-ink-2)">{html.escape(rest)}</text>'
    return out


def _words(x: float, y: float, text_: str, anchor: str = "middle", size: float = 11.5, fill: str = "var(--ea-ink-2)",
           weight: int = 400) -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" style="font-size:{size}px;font-weight:{weight};fill:{fill};'
            f'text-anchor:{anchor};{_HALO}">{html.escape(text_)}</text>')


def _svg(width: float, height: float, parts: list) -> str:
    return (f'<svg viewBox="0 0 {width:.0f} {height:.0f}" aria-hidden="true" style="flex:0 1 auto;width:100%;'
            f'max-width:{width:.0f}px;height:auto;font-family:var(--ea-font-ui)">{"".join(parts)}</svg>')


def _figure(svgs: list, alt: str, width: int) -> str:
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(alt)}" style="margin:0.25rem 0 0.5rem;'
            f'max-width:min(100%,{width}px);display:flex;flex-wrap:wrap;gap:0.6rem 1.4rem;align-items:flex-start">'
            f'{"".join(svgs)}</figure>')


def _arrow(x0: float, y0: float, x1: float, y1: float, color: str = "var(--ea-ink-2)", width: float = 1.6) -> str:
    """A straight arrow from (x0, y0) to (x1, y1)."""
    d = math.hypot(x1 - x0, y1 - y0) or 1
    ux, uy = (x1 - x0) / d, (y1 - y0) / d
    bx, by = x1 - 7 * ux, y1 - 7 * uy
    return (f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{bx:.1f}" y2="{by:.1f}" style="stroke:{color};stroke-width:{width}"/>'
            f'<polygon points="{x1:.1f},{y1:.1f} {bx - 4 * uy:.1f},{by + 4 * ux:.1f} {bx + 4 * uy:.1f},{by - 4 * ux:.1f}" '
            f'style="fill:{color}"/>')


def _path_arrow(points: list, color: str, width: float = 1.4) -> str:
    """Straight pieces through `points`, with an arrowhead at the last one."""
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in points[:-1])
    (x0, y0), (x1, y1) = points[-2], points[-1]
    d = math.hypot(x1 - x0, y1 - y0) or 1
    bx, by = x1 - 7 * (x1 - x0) / d, y1 - 7 * (y1 - y0) / d
    return (f'<polyline points="{line} {bx:.1f},{by:.1f}" style="fill:none;stroke:{color};stroke-width:{width}"/>'
            + _arrow(bx, by, x1, y1, color, width))


def _box(x: float, y: float, w: float, h: float, words: str, stroke: str = "var(--ea-line)", size: float = 11,
         width: float = 1.1) -> str:
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="5" '
            f'style="fill:var(--ea-page);stroke:{stroke};stroke-width:{width}"/>'
            + _words(x + w / 2, y + h / 2 + 4, words, size=size, fill="var(--ea-ink)"))


def _plot(values: list, x0: float, y0: float, w: float, h: float, low: float, high: float, color: str, width: float = 2.2,
          dashed: bool = False) -> str:
    """A line through values, one per moment, from the first at x0 to the last at x0 + w, between low and high."""
    n = max(len(values) - 1, 1)
    pts = []
    for i, v in enumerate(values):
        shown = min(max(float(v), low), high)
        pts.append(f"{x0 + i / n * w:.1f},{y0 + h - (shown - low) / (high - low) * h:.1f}")
    dash = ";stroke-dasharray:5 4" if dashed else ""
    return (f'<polyline points="{" ".join(pts)}" style="fill:none;stroke:{color};stroke-width:{width};'
            f'stroke-linejoin:round{dash}"/>')


def _frame(x0: float, y0: float, w: float, h: float, low: float, high: float, yticks: list, last: int, xticks: list,
           lang: str, xlabel: str) -> list:
    """The axes of a chart of trials: a y-axis with its ticks, and the trials along the bottom, 0 at the start."""
    parts = [f'<line x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}" style="stroke:var(--ea-line);stroke-width:1"/>',
             f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + h}" style="stroke:var(--ea-line);stroke-width:1"/>']
    for t in yticks:
        y = y0 + h - (t - low) / (high - low) * h
        parts.append(f'<line x1="{x0 - 3}" y1="{y:.1f}" x2="{x0 + w}" y2="{y:.1f}" style="stroke:var(--ea-line);stroke-width:0.6;opacity:0.6"/>')
        digits = 0 if float(t).is_integer() else 1
        parts.append(_words(x0 - 6, y + 4, number(t, digits, lang), anchor="end", size=10.5))
    for t in xticks:
        x = x0 + t / max(last, 1) * w
        parts.append(_words(x, y0 + h + 14, number(t, 0, lang), size=10.5))
    parts.append(_words(x0 + w / 2, y0 + h + 30, xlabel, size=11))
    return parts


def _xticks(last: int) -> list:
    step = 5 if last <= 25 else 10
    return list(range(0, last + 1, step))


# --- the pictures of the idea --------------------------------------------------------------------------------------------

_IDEA_TEXT = {
    "en": {"plan": "The experiment of the guess",
           "first": "trials 1 to 20", "second": "trials 21 to 40", "test": "at the end",
           "sound": "sound", "light": "light", "food": "food", "what": "?",
           "alt_plan": "The experiment in three rows. Trials 1 to 20: the sound, then food. Trials 21 to 40: the sound and the light together, then food. At the end: what does the light alone predict?",
           "shared": "One surprise for all signals: every signal present takes the same step",
           "v_sound": "V sound", "v_light": "V light", "together": "together: ΣV", "food_f": "food f",
           "surprise": "surprise: f − ΣV", "back": "+ η × surprise",
           "alt_shared": "A diagram. The V of the sound and the V of the light are added up into ΣV, what the two predict together. The food f minus ΣV is the surprise. The surprise goes back to both V's: each signal present gets η times the surprise added.",
           "blocking": "Blocking: the sound first, then the sound and the light",
           "control": "Without the first part: the sound and the light at once",
           "alone": "sound alone", "nothing": "nothing", "both": "sound and light", "trial": "trial",
           "alt_runs": "Two charts of V per trial, η 0.2. Left, blocking: in trials 1 to 20 only the sound comes before the food, and its V rises to 0.99; from trial 21 the light comes with it, and the V of the light stays near 0, at 0.006 after trial 40. Right, without the first part: nothing happens in trials 1 to 20, then the sound and the light come together before the food, and both V's rise to 0.50."},
    "nl": {"plan": "Het experiment van de gok",
           "first": "trials 1 tot en met 20", "second": "trials 21 tot en met 40", "test": "aan het eind",
           "sound": "geluid", "light": "lamp", "food": "eten", "what": "?",
           "alt_plan": "Het experiment in drie rijen. Trials 1 tot en met 20: het geluid, dan eten. Trials 21 tot en met 40: het geluid en de lamp samen, dan eten. Aan het eind: wat voorspelt de lamp alleen?",
           "shared": "Eén verrassing voor alle signalen: elk aanwezig signaal zet dezelfde stap",
           "v_sound": "V geluid", "v_light": "V lamp", "together": "samen: ΣV", "food_f": "eten f",
           "surprise": "verrassing: f − ΣV", "back": "+ η × verrassing",
           "alt_shared": "Een schema. De V van het geluid en de V van de lamp worden opgeteld tot ΣV, wat de twee samen voorspellen. Het eten f min ΣV is de verrassing. De verrassing gaat terug naar beide V's: elk aanwezig signaal krijgt η keer de verrassing erbij.",
           "blocking": "Blocking: eerst het geluid, dan geluid en lamp",
           "control": "Zonder het eerste deel: geluid en lamp meteen",
           "alone": "alleen geluid", "nothing": "niets", "both": "geluid en lamp", "trial": "trial",
           "alt_runs": "Twee grafieken van V per trial, η 0,2. Links, blocking: in trials 1 tot en met 20 komt alleen het geluid voor het eten, en de V van het geluid stijgt naar 0,99; vanaf trial 21 komt de lamp erbij, en de V van de lamp blijft bij 0, op 0,006 na trial 40. Rechts, zonder het eerste deel: in trials 1 tot en met 20 gebeurt niets, dan komen geluid en lamp samen voor het eten, en beide V's stijgen naar 0,50."},
}


def _plan_svg(lang: str, W: int = 330) -> str:
    """The blocking experiment in three rows: the sound before food, then the sound and the light, then the light alone."""
    w = _IDEA_TEXT[lang]
    parts = [_title(2, 14, w["plan"])]
    bw, bh, gap = 58, 26, 22
    xs = [150, 150 + bw + gap]
    y = TOP - 6
    rows = [(w["first"], [[(w["sound"], False)], [(w["food"], False)]]),
            (w["second"], [[(w["sound"], False), (w["light"], True)], [(w["food"], False)]]),
            (w["test"], [[(w["light"], True)], [(w["what"], None)]])]
    for label, (signals, after) in rows:
        height = bh * len(signals) + 6 * (len(signals) - 1)
        parts.append(_words(2, y + height / 2 + 4, label, anchor="start", size=11, fill="var(--ea-ink)", weight=600))
        for k, (words, new) in enumerate(signals):
            stroke = "var(--ea-ink)" if new else "var(--ea-line)"
            parts.append(_box(xs[0], y + k * (bh + 6), bw, bh, words, stroke=stroke, width=1.6 if new else 1.1))
        words, kind = after[0]
        mid = y + height / 2
        if kind is None:
            parts.append(_words(xs[1] + bw / 2, mid + 6, words, size=17, fill="var(--ea-ink)", weight=600))
        else:
            parts.append(_box(xs[1], mid - bh / 2, bw, bh, words))
        parts.append(_arrow(xs[0] + bw + 3, mid, xs[1] - 3, mid))
        y += height + 16
    return _svg(W, y, parts)


def plan_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    return _figure([_plan_svg(lang)], _IDEA_TEXT[lang]["alt_plan"], 330)


def _shared_svg(lang: str, W: int = 330) -> str:
    """The rule as a diagram: both V's added up, the food minus that is the surprise, and the surprise back to both."""
    w = _IDEA_TEXT[lang]
    parts = [_title(2, 14, w["shared"])]
    loss = "var(--ea-loss-text)"
    bx, bw, bh = 24, 78, 26
    ys, yl = TOP + 22, TOP + 78          # the boxes of the sound and the light
    mid = (ys + yl + bh) / 2
    cx, r = 132, 11
    sx, sw = 198, 128                    # the box of the surprise
    parts.append(_box(bx, ys, bw, bh, w["v_sound"], stroke="var(--ea-param)", width=1.6))
    parts.append(_box(bx, yl, bw, bh, w["v_light"], stroke="var(--ea-param)", width=1.6))
    parts.append(_arrow(bx + bw + 2, ys + bh / 2, cx - r * 0.75, mid - r * 0.65))
    parts.append(_arrow(bx + bw + 2, yl + bh / 2, cx - r * 0.75, mid + r * 0.65))
    parts.append(f'<circle cx="{cx}" cy="{mid:.1f}" r="{r}" style="fill:var(--ea-page);stroke:var(--ea-ink-2);stroke-width:1.3"/>')
    parts.append(_words(cx, mid + 5, "+", size=15, fill="var(--ea-ink)", weight=600))
    parts.append(_arrow(cx + r + 2, mid, sx - 2, mid))
    parts.append(_words((cx + r + sx) / 2, mid - 7, w["together"].split(": ")[1], size=11))
    parts.append(_words((cx + r + sx) / 2, mid + 17, w["together"].split(": ")[0], size=11))
    parts.append(_box(sx + 22, ys - 4, sw - 44, bh - 4, w["food_f"], size=11))
    parts.append(_arrow(sx + sw / 2, ys + bh - 8, sx + sw / 2, mid - bh / 2 - 2))
    parts.append(_box(sx, mid - bh / 2, sw, bh, w["surprise"], stroke=loss, width=1.6))
    bottom = yl + bh + 26
    parts.append(f'<polyline points="{sx + sw / 2:.1f},{mid + bh / 2:.1f} {sx + sw / 2:.1f},{bottom:.1f} {10},{bottom:.1f} {10},{ys + bh / 2 + 4:.1f}" '
                 f'style="fill:none;stroke:{loss};stroke-width:1.4"/>')
    parts.append(_arrow(10, ys + bh / 2 + 4, bx - 2, ys + bh / 2 + 4, loss, 1.4))
    parts.append(_arrow(bx + bw / 2, bottom, bx + bw / 2, yl + bh + 2, loss, 1.4))
    parts.append(_words((sx + sw / 2 + bx + bw / 2) / 2 + 20, bottom - 6, w["back"], size=11, fill=loss))
    return _svg(W, bottom + 8, parts)


def _runs_svg(values, title: str, first_label: str, lang: str, course=None, W: int = 330) -> str:
    """V per trial of a run of 40 trials: the sound dashed and the light solid, both in the color of what a model
    learns, with the first 20 trials shaded. With `course`, the values are the learner's and wrong: drawn in the color
    of mistakes, over the course's in gray."""
    w = _IDEA_TEXT[lang]
    values = np.asarray(values, dtype=float)
    last = len(values) - 1
    low, high = (0.0, 1.0) if course is None else (-0.2, 1.2)
    x0, y0, cw, ch = 34, TOP + 24, W - 70, 140
    parts = [_title(2, 14, title)]
    mid = x0 + PHASE / last * cw
    parts.append(f'<rect x="{x0}" y="{y0}" width="{mid - x0:.1f}" height="{ch}" style="fill:var(--ea-inset)"/>')
    parts.append(_words((x0 + mid) / 2, y0 - 6, first_label, size=10.5))
    parts.append(_words((mid + x0 + cw) / 2, y0 - 6, w["both"], size=10.5))
    parts += _frame(x0, y0, cw, ch, low, high, [0, 0.5, 1], last, _xticks(last), lang, w["trial"])
    color = "var(--ea-param)"
    if course is not None:
        course = np.asarray(course, dtype=float)
        parts.append(_plot(list(course[:, SOUND]), x0, y0, cw, ch, low, high, "var(--ea-ink-3)", 1.3, dashed=True))
        parts.append(_plot(list(course[:, LIGHT]), x0, y0, cw, ch, low, high, "var(--ea-ink-3)", 1.3))
        color = "var(--ea-loss-text)"
    parts.append(_plot(list(values[:, SOUND]), x0, y0, cw, ch, low, high, color, 2.0, dashed=True))
    parts.append(_plot(list(values[:, LIGHT]), x0, y0, cw, ch, low, high, color, 2.4))
    ends = [(float(values[-1, SOUND]), w["sound"]), (float(values[-1, LIGHT]), w["light"])]
    together = abs(ends[0][0] - ends[1][0]) < 0.12
    if together:   # the two lines end in the same place: one name for both, above their end
        ends = [((ends[0][0] + ends[1][0]) / 2, w["both"])]
    for value, words in ends:
        shown = min(max(value, low), high)
        y = y0 + ch - (shown - low) / (high - low) * ch
        if together:
            parts.append(_words(x0 + cw, y - 8, words, anchor="end", size=10.5, fill="var(--ea-ink)"))
        else:
            parts.append(_words(x0 + cw + 4, y + 4, words, anchor="start", size=10.5, fill="var(--ea-ink)"))
    return _svg(W, y0 + ch + 40, parts)


def shared_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    return _figure([_shared_svg(lang)], _IDEA_TEXT[lang]["alt_shared"], 330)


def runs_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang]
    return _figure([_runs_svg(run(BLOCKING, ETA), w["blocking"], w["alone"], lang),
                    _runs_svg(run(CONTROL, ETA), w["control"], w["nothing"], lang)], w["alt_runs"], 690)


def bars_svg(v, surprise: float, title: str, lang: str, W: int = 330) -> str:
    """The V of the sound and of the light, in the color of what a model learns, and the surprise, in the color of
    mistakes, as bars from 0 to 1."""
    w = _IDEA_TEXT[lang]
    left, row, span = 74, 24, W - 74 - 46
    parts = [_title(2, 14, title)]
    rows = [(w["v_sound"], float(v[SOUND]), "var(--ea-param)"), (w["v_light"], float(v[LIGHT]), "var(--ea-param)"),
            (w["surprise"].split(":")[0], float(surprise), "var(--ea-loss)")]
    top = TOP + 4
    for i, (label, value, color) in enumerate(rows):
        y = top + i * row
        parts.append(_words(left - 6, y + 15, label, anchor="end", size=11, fill="var(--ea-ink)"))
        length = min(max(value, 0.0), 1.0) * span
        parts.append(f'<rect x="{left}" y="{y + 4}" width="{max(length, 1):.1f}" height="{row - 8}" style="fill:{color};fill-opacity:0.85"/>')
        parts.append(_words(left + length + 5, y + 15, short_number(value, lang), anchor="start", size=10.5))
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + len(rows) * row}" style="stroke:var(--ea-line);stroke-width:1"/>')
    return _svg(W, top + len(rows) * row + 8, parts)


# --- comparing and showing ---------------------------------------------------------------------------------------------

def _fine(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError, OverflowError):
        return False


def _array(a):
    """`a` as an array of floats, or None when it is not numbers."""
    try:
        out = np.asarray(a, dtype=float)
    except (TypeError, ValueError):
        return None
    return out if out.size and np.isfinite(out).all() else None


def same_array(a, b) -> bool:
    """rw_step and run: an array of the same shape as the course's, every number the same."""
    got, want = _array(a), np.asarray(b, dtype=float)
    return got is not None and got.shape == want.shape and bool(np.allclose(got, want, rtol=0, atol=1e-9))


def _short(v) -> str:
    got = _array(v)
    if got is not None and got.ndim == 1:
        rounded = [round(float(x), 3) for x in got]
        shown = [str(int(x)) if x.is_integer() else repr(x) for x in rounded]
        return "[" + ", ".join(shown) + "]"
    text_ = repr(v)
    return text_ if len(text_) <= 80 else text_[:77] + "..."


def show_step(v) -> dict:
    shown = f"`{_short(v)}`"
    return {"en": shown, "nl": shown}


def show_rows(v) -> dict:
    """What run gives, in a message: how many rows, and the last row, the V's after the last trial."""
    got = _array(v)
    if got is None or got.ndim != 2:
        shown = f"`{_short(v)}`"
        return {"en": shown, "nl": shown}
    last = f"`{_short(got[-1])}`"
    return {"en": f"an array of {len(got)} rows, the last {last}", "nl": f"een array van {len(got)} rijen, de laatste {last}"}


def _arr_code(a) -> str:
    a = np.asarray(a)
    if a.dtype.kind in "iu":
        return "np.array([" + ", ".join(str(int(x)) for x in a) + "])"
    return "np.array([" + ", ".join(repr(round(float(x), 6)) for x in a) + "])"


class StepBox(FunctionBox):
    """rw_step in a code box: a call shows its arrays as `np.array([...])`, as the learner would write them."""

    def _call_text(self, case) -> str:
        v, present, food, eta = case
        return f"rw_step({_arr_code(v)}, {_arr_code(present)}, {food}, {eta})"


class RunBox(FunctionBox):
    """run in a code box: a call shows the name of its trials, such as `run(blocking, 0.2)`."""

    def _call_text(self, case) -> str:
        return f"run({case[0].name}, {case[1]})"


# --- the pictures of the learner's work -----------------------------------------------------------------------------------

_STOPPED = object()   # what _call gives for a function that stopped with an error
_SLOW = object()      # what _call gives for a function that ran out of time
PICTURE_SECONDS = 4   # what the learner's function gets for all the calls of one picture together


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _call(make_, seconds: float = PICTURE_SECONDS):
    """make_() within a time limit, with what it prints thrown away; _SLOW when the time runs out, _STOPPED when it
    stops with an error."""
    import contextlib
    import io
    from everything_ai.check import with_time_limit
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(make_, max(seconds, 0.05))
    except Timeout:
        return _SLOW
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


STEP_CASES = [(np.array([0.5, 0.0]), np.array([1, 1]), 1, ETA), (np.array([0.99, 0.0]), np.array([1, 1]), 1, ETA)]

_YOURS_TEXT = {
    "en": {"rw_step": {"title": "Your rw_step",
                       "intro": "One trial with the sound and the light together and food, η 0.2, from two starting points: the bars of V before the trial pale, after it as your `rw_step` gives it, and under them the surprise of the trial; outlined, the course's V after the trial where yours differs, with yours in the color of mistakes.",
                       "first": "V 0.5 and 0: the sound half learned",
                       "second": "V 0.99 and 0: the sound already learned",
                       "say": "With V 0.5 and 0, your `rw_step` gives {a} and {b}; with 0.99 and 0, it gives {c} and {d}.",
                       "none": "Your `rw_step` stops with an error, takes too long, or gives something other than an array of two numbers, so nothing can be drawn yet."},
           "run": {"title": "Your run",
                   "intro": "V per trial as your `run` gives it, with η 0.2, for `blocking` and for `control`: the sound dashed, the light solid; in gray, the course's where yours differs, with yours in the color of mistakes.",
                   "say": "After trial 40, the light has V {b} in `blocking` and {c} in `control`.",
                   "out": "Some of your values lie outside the frame, from {low} to {high}.",
                   "none": "Your `run` stops with an error, takes too long, or gives something other than an array with 41 rows of two numbers, so nothing can be drawn yet."}},
    "nl": {"rw_step": {"title": "Je rw_step",
                       "intro": "Eén trial met geluid en lamp samen en eten, η 0,2, vanaf twee beginpunten: de balken van V voor de trial licht, na de trial zoals je `rw_step` V geeft, en daaronder de verrassing van de trial; omlijnd de V van de course na de trial waar je eigen V anders is, met je eigen V in de kleur van fouten.",
                       "first": "V 0,5 en 0: het geluid half geleerd",
                       "second": "V 0,99 en 0: het geluid al geleerd",
                       "say": "Met V 0,5 en 0 geeft je `rw_step` {a} en {b}; met 0,99 en 0 geeft je `rw_step` {c} en {d}.",
                       "none": "Je `rw_step` stopt met een fout, duurt te lang, of geeft iets anders dan een array van twee getallen, dus is er nog geen afbeelding."},
           "run": {"title": "Je run",
                   "intro": "V per trial zoals je `run` V geeft, met η 0,2, voor `blocking` en voor `control`: het geluid gestreept, de lamp doorgetrokken; in grijs de V van de course waar je eigen V anders is, met je eigen V in de kleur van fouten.",
                   "say": "Na trial 40 heeft de lamp V {b} in `blocking` en {c} in `control`.",
                   "out": "Een paar van je waarden liggen buiten het kader, van {low} tot {high}.",
                   "none": "Je `run` stopt met een fout, duurt te lang, of geeft iets anders dan een array met 41 rijen van twee getallen, dus is er nog geen afbeelding."}},
}


def _step_svg(case, mine, title: str, lang: str, W: int = 330) -> str:
    """One trial of rw_step as bars: for each signal its V before (pale) and after (as the learner's rw_step gives
    it), then the surprise. Where the learner's V after differs from the course's, the course's is outlined."""
    w = _IDEA_TEXT[lang]
    v, present, food, eta = case
    want = rw_step(v, present, food, eta)
    surprise = float(food - present @ v)
    right = same_array(mine, want)
    shown = [float(x) for x in v] + [float(x) for x in mine] + [float(x) for x in want] + [surprise]
    low, high = min(0.0, min(shown)), max(1.0, max(shown))
    low, high = max(low, -1.5), min(high, 2.5)
    left, span = 74, W - 74 - 52
    zero = left + (0 - low) / (high - low) * span

    def x_of(value: float) -> float:
        return left + (min(max(value, low), high) - low) / (high - low) * span

    parts = [_title(2, 14, title)]
    y = TOP + 4
    for i, label in enumerate((w["v_sound"], w["v_light"])):
        parts.append(_words(left - 6, y + 22, label, anchor="end", size=11, fill="var(--ea-ink)"))
        for k, (value, style) in enumerate(((float(v[i]), "fill:var(--ea-param);fill-opacity:0.35"),
                                            (float(mine[i]), "fill:var(--ea-param);fill-opacity:0.95" if right
                                             else "fill:var(--ea-loss-text);fill-opacity:0.85"))):
            top = y + 2 + k * 15
            a, b = sorted((zero, x_of(value)))
            parts.append(f'<rect x="{a:.1f}" y="{top:.1f}" width="{max(b - a, 1):.1f}" height="12" style="{style}"/>')
            if k == 1 and not right:
                a2, b2 = sorted((zero, x_of(float(want[i]))))
                parts.append(f'<rect x="{a2:.1f}" y="{top:.1f}" width="{max(b2 - a2, 1):.1f}" height="12" '
                             f'style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.2;stroke-dasharray:3 2"/>')
            parts.append(_words(max(b, zero) + 5, top + 10, short_number(value, lang), anchor="start", size=10))
        y += 38
    parts.append(_words(left - 6, y + 13, w["surprise"].split(":")[0], anchor="end", size=11, fill="var(--ea-ink)"))
    a, b = sorted((zero, x_of(surprise)))
    parts.append(f'<rect x="{a:.1f}" y="{y + 3:.1f}" width="{max(b - a, 1):.1f}" height="13" style="fill:var(--ea-loss);fill-opacity:0.8"/>')
    parts.append(_words(max(b, zero) + 5, y + 13, short_number(surprise, lang), anchor="start", size=10))
    parts.append(f'<line x1="{zero:.1f}" y1="{TOP + 2}" x2="{zero:.1f}" y2="{y + 20:.1f}" style="stroke:var(--ea-line);stroke-width:1"/>')
    return _svg(W, y + 28, parts)


def step_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["rw_step"]
    fn = _fn(seen)
    got = _call(lambda: [fn(*copy.deepcopy(case)) for case in STEP_CASES]) if fn is not None else None
    mine = [_array(g) for g in got] if isinstance(got, list) else None
    if mine is None or any(m is None or m.shape != (2,) for m in mine):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    svgs = [_step_svg(case, m, w[key], lang) for case, m, key in zip(STEP_CASES, mine, ("first", "second"))]
    say = w["say"].format(a=short_number(mine[0][0], lang), b=short_number(mine[0][1], lang),
                          c=short_number(mine[1][0], lang), d=short_number(mine[1][1], lang))
    return yours.figure(w["title"], _note(w["intro"]) + _figure(svgs, w["title"], 690) + _note(say), alt=w["title"])


def run_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["run"]
    idea = _IDEA_TEXT[lang]
    fn = _fn(seen)
    got = _call(lambda: [fn(copy.deepcopy(t), ETA) for t in (BLOCKING, CONTROL)]) if fn is not None else None
    mine = [_array(g) for g in got] if isinstance(got, list) else None
    shape = (len(BLOCKING) + 1, 2)
    if mine is None or any(m is None or m.shape != shape for m in mine):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    want = [run(BLOCKING, ETA), run(CONTROL, ETA)]
    svgs = []
    for m, c, key, first in zip(mine, want, ("blocking", "control"), ("alone", "nothing")):
        svgs.append(_runs_svg(m, idea[key], idea[first], lang, course=None if same_array(m, c) else c))
    say = w["say"].format(b=short_number(mine[0][-1, LIGHT], lang), c=short_number(mine[1][-1, LIGHT], lang))
    out = ""
    low, high = min(float(m.min()) for m in mine), max(float(m.max()) for m in mine)
    if low < -0.2 or high > 1.2:
        out = _note(w["out"].format(low=number(low, 2, lang), high=number(high, 2, lang)), _LOSS_UI)
    return yours.figure(w["title"], _note(w["intro"]) + _figure(svgs, w["title"], 690) + out + _note(say), alt=w["title"])


# --- the demo ----------------------------------------------------------------------------------------------------------------

DEMO_TEXT = {
    "en": {"title": "V of the light, and dashed of the sound", "x": "trial", "y": "V",
           "alt": "V of the light in every trial so far, and dashed the V of the sound.",
           "first": "Trials with the sound alone", "eta": "Learning rate η",
           "light": "light", "sound": "sound",
           "now": "After trial {t}", "start": "Before the first trial",
           "say": "In trial {t}, {signals} came, and food; together they predicted {together}, so the surprise was {surprise}. After it, V of the sound is {s} and V of the light {l}.",
           "alone": "the sound alone", "both": "the sound and the light",
           "who_mine": "your `run`", "who_course": "the course's `run`, since yours does not pass its check yet",
           "learning": "Trial {t} of {n}",
           "verdict": "With {who}: after {k} trials with the sound alone, the light ends at **{light}** and the sound at **{sound}**. With the sound and the light together from the start, the light would end at {control}.",
           "verdict_none": "With {who}: without trials with the sound alone, the sound and the light share what there is to learn, and both end at **{light}**."},
    "nl": {"title": "V van de lamp, en gestreept van het geluid", "x": "trial", "y": "V",
           "alt": "V van de lamp in elke trial tot nu toe, en gestreept de V van het geluid.",
           "first": "Trials met alleen het geluid", "eta": "Learning rate η",
           "light": "lamp", "sound": "geluid",
           "now": "Na trial {t}", "start": "Voor de eerste trial",
           "say": "In trial {t} kwam {signals}, en eten; samen voorspelden de signalen {together}, dus was de verrassing {surprise}. Daarna is V van het geluid {s} en V van de lamp {l}.",
           "alone": "alleen het geluid", "both": "het geluid met de lamp",
           "who_mine": "je `run`", "who_course": "de `run` van de course, omdat je eigen `run` de check nog niet haalt",
           "learning": "Trial {t} van {n}",
           "verdict": "Met {who}: na {k} trials met alleen het geluid eindigt de lamp op **{light}** en het geluid op **{sound}**. Met geluid en lamp samen vanaf het begin zou de lamp eindigen op {control}.",
           "verdict_none": "Met {who}: zonder trials met alleen het geluid delen geluid en lamp wat er te leren valt, en eindigen beide op **{light}**."},
}


def first_options(lang: str) -> dict:
    return {number(n, 0, lang): n for n in FIRST_OPTIONS}


def eta_options(lang: str) -> dict:
    return {number(e, 1, lang): e for e in ETA_OPTIONS}


def chart_labels(lang: str) -> dict:
    w = DEMO_TEXT[lang]
    return {"title": w["title"], "x": w["x"], "y": w["y"], "alt": w["alt"]}


def chart_key(lang: str) -> list:
    w = DEMO_TEXT[lang]
    return [("param", False, w["light"]), ("param", True, w["sound"])]


def demo_xticks(first: int) -> list:
    last = first + PHASE
    return [1] + [t for t in _xticks(last) if t > 1]


class Demo:
    """The demo's experiment: `first` trials with the sound alone, then PHASE with the sound and the light, and the
    V's of a run over them, shown one trial at a time."""

    def __init__(self, first: int, eta: float, values):
        self.first, self.eta = first, eta
        self.trials = blocking_trials(first)
        self.values = np.asarray(values, dtype=float)
        self.surprises = surprises(self.trials, self.values)
        self.t = 0   # the trials shown so far

    def points(self, signal: int, t: int | None = None) -> list:
        t = self.t if t is None else t
        return [(k, float(self.values[k, signal])) for k in range(1, t + 1)]


def demo_run(run_, first: int, eta: float, show=None, pause: float = 0.0, seconds: float = 4.0) -> Demo | str:
    """The demo: the V's of `run_` over the experiment, then `show(t, demo)` after every trial; "error" when the
    learner's run stops with an error, takes too long, or gives something other than a row of two V's per moment.
    `pause` waits that long after each trial, so that you can follow the trials."""
    import time
    trials = blocking_trials(first)
    got = _call(lambda: run_(copy.deepcopy(trials), eta), seconds)
    values = _array(got) if got is not _STOPPED and got is not _SLOW else None
    if values is None or values.shape != (len(trials) + 1, 2):
        return "error"
    demo = Demo(first, eta, values)
    for t in range(1, len(trials) + 1):
        demo.t = t
        if show is not None:
            show(t, demo)
        if pause:
            time.sleep(pause)
    return demo


def _bars_at(demo: Demo, t: int, lang: str) -> str:
    w = DEMO_TEXT[lang]
    title = w["now"].format(t=t) if t else w["start"]
    surprise = demo.surprises[t - 1] if t else 0.0
    return _figure([bars_svg(demo.values[t], surprise, title, lang)], title, 330)


def now_picture(demo: Demo, lang: str) -> str:
    """Beside the demo's chart: the V's after the last trial shown, and the surprise of that trial."""
    lang = lang if lang in LANGS else "en"
    return _bars_at(demo, demo.t, lang)


def _demo_panel(demo: Demo, t: int, lang: str, W: int = 330) -> str:
    """The demo's chart as a small picture, with a line at trial t."""
    w = DEMO_TEXT[lang]
    last = len(demo.trials)
    x0, y0, cw, ch = 34, TOP + 24, W - 76, 140
    parts = [_title(2, 14, w["title"])]
    if demo.first:
        mid = x0 + demo.first / last * cw
        parts.append(f'<rect x="{x0}" y="{y0}" width="{mid - x0:.1f}" height="{ch}" style="fill:var(--ea-inset)"/>')
    parts += _frame(x0, y0, cw, ch, 0, 1, [0, 0.5, 1], last, _xticks(last), lang, w["x"])
    parts.append(_plot(list(demo.values[:, SOUND]), x0, y0, cw, ch, 0, 1, "var(--ea-param)", 2.0, dashed=True))
    parts.append(_plot(list(demo.values[:, LIGHT]), x0, y0, cw, ch, 0, 1, "var(--ea-param)", 2.4))
    for value, words in ((float(demo.values[-1, SOUND]), w["sound"]), (float(demo.values[-1, LIGHT]), w["light"])):
        y = y0 + ch - min(max(value, 0), 1) * ch
        if abs(float(demo.values[-1, SOUND]) - float(demo.values[-1, LIGHT])) < 0.12:
            y += -6 if words == w["sound"] else 8
        parts.append(_words(x0 + cw + 4, y + 4, words, anchor="start", size=10.5, fill="var(--ea-ink)"))
    if t:
        x = x0 + t / last * cw
        parts.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0 + ch}" style="stroke:var(--ea-ink-2);stroke-width:1.2"/>')
    return _svg(W, y0 + ch + 40, parts)


def rewind_frame(demo: Demo, t: int, lang: str) -> tuple:
    """A stop of the rewind after the demo: the chart with a line at trial t, the V's and the surprise then, and a
    sentence."""
    lang = lang if lang in LANGS else "en"
    w = DEMO_TEXT[lang]
    t = int(t)
    picture = _bars_at(demo, t, lang)
    svg = _figure([_demo_panel(demo, t, lang)], w["alt"], 330)
    say = ""
    if t:
        present, _ = demo.trials[t - 1]
        before = demo.values[t - 1]
        say = w["say"].format(t=t, signals=w["both"] if present[LIGHT] else w["alone"],
                              together=short_number(float(present @ before), lang),
                              surprise=short_number(demo.surprises[t - 1], lang),
                              s=short_number(demo.values[t, SOUND], lang), l=short_number(demo.values[t, LIGHT], lang))
    return live.rewound(svg, [picture], width=330), say


def verdict(demo: Demo, mine: bool, lang: str) -> str:
    w = DEMO_TEXT[lang]
    who = w["who_mine"] if mine else w["who_course"]
    light, sound = float(demo.values[-1, LIGHT]), float(demo.values[-1, SOUND])
    if not demo.first:
        return w["verdict_none"].format(who=who, light=number(light, 2, lang))
    control = float(run(blocking_trials(0), demo.eta)[-1, LIGHT])
    return w["verdict"].format(who=who, k=demo.first, light=short_number(light, lang), sound=number(sound, 2, lang),
                               control=number(control, 2, lang))


# --- the exercises -----------------------------------------------------------------------------------------------------

REQUIRED = ["rw_step", "run", "latent", "td"]


def _dots_left(got, text_) -> bool:
    """A `...` of the start code left in the box: an Ellipsis in the code, not three dots inside a string."""
    import ast
    try:
        tree = ast.parse(text_)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Constant) and node.value is Ellipsis for node in ast.walk(tree))


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. Can you replace it with your own code?",
                nl="Er staat nog een `...` in je code. Kun je de `...` vervangen door je eigen code?")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


# exercise 1: rw_step
def _s(line: str = "food - present @ v") -> str:
    return ("def rw_step(v, present, food, eta):\n"
            f"    surprise = {line}\n"
            "    return v + eta * surprise * present\n")


_S_START = _s("...")
_S = _s()
_S_OWN = _s("food - v")
_S_SIGN = _s("present @ v - food")
_S_ALL = _s("food - v.sum()")
_S_FIXED = _s("1 - present @ v")
S_NAMES = {"np": np}


def _scalar_matmul(e: BaseException) -> bool:
    return isinstance(e, ValueError) and "matmul" in str(e)


# exercise 2: run
def _r(line: str = "rw_step(v, present, food, eta)") -> str:
    return ("def run(trials, eta):\n"
            "    v = np.zeros(2)\n"
            "    seen = [v]\n"
            "    for present, food in trials:\n"
            f"        v = {line}\n"
            "        seen.append(v)\n"
            "    return np.array(seen)\n")


_R_START = _r("...")
_R = _r()
_R_SWAP = _r("rw_step(v, present, eta, food)")
_R_RESTART = _r("rw_step(np.zeros(2), present, food, eta)")
_R_DOUBLE = _r("v + rw_step(v, present, food, eta)")
_R_FIXED = _r("rw_step(v, present, food, 0.2)")
R_NAMES = {"np": np, "rw_step": rw_step}


def run_names(step=None) -> dict:
    """The names a run in a code box may use: NumPy, and an rw_step, the course's unless another is given."""
    return {"np": np, "rw_step": step if step is not None else rw_step}


EXERCISES = {
    "rw_step": StepBox(
        name="rw_step",
        function="rw_step",
        start=_S_START,
        solution=_S,
        names=S_NAMES,
        compare=same_array,
        show=show_step,
        visible=[(np.array([0.5, 0.0]), np.array([1, 1]), 1, ETA)],
        hidden=[(np.array([0.99, 0.0]), np.array([1, 1]), 1, ETA),
                (np.array([0.6, -0.3]), np.array([1, 0]), 1, ETA),
                (np.array([0.5, 0.5]), np.array([1, 1]), 0, 0.5),
                (np.array([0.0, 0.0, 0.0]), np.array([1, 0, 1]), 1, 0.1)],
        first=[_DOTS],
        picture=step_picture,
        mistakes=[
            Mistake("scalar-matmul", error=_scalar_matmul,
                    en="With `{call}`, your function stops: `@` needs an array on both sides, and one side of yours is a single number. Which two arrays give, with `@`, what the signals present predict together?",
                    nl="Met `{call}` stopt je functie: `@` heeft aan beide kanten een array nodig, en aan een kant van je `@` staat één getal. Welke twee arrays geven met `@` wat de aanwezige signalen samen voorspellen?"),
            Mistake("own", wrong=_S_OWN,
                    en="`{call}` gives {got}, and it should give {want}: every signal gets its own surprise, the food minus its own V, as if it were alone. In the Rescorla-Wagner model, the signals present share one surprise: the food minus what they predict together. How do you add up the V's of the signals present?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elk signaal krijgt een eigen verrassing, het eten min de eigen V, alsof het signaal alleen was. In het Rescorla-Wagner model delen de aanwezige signalen één verrassing: het eten min wat de signalen samen voorspellen. Hoe tel je de V's van de aanwezige signalen op?"),
            Mistake("sign", wrong=_S_SIGN,
                    en="`{call}` gives {got}, and it should give {want}: with the prediction minus the food, the V's move away from the food. Which of the two comes first in the surprise?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: met de voorspelling min het eten bewegen de V's van het eten af. Welke van de twee komt eerst in de verrassing?"),
            Mistake("all", wrong=_S_ALL,
                    en="`{call}` gives {got}, and it should give {want}: you add up the V's of all signals, also of a signal that is absent in the trial. Which array says which signals are present, so that only their V's count?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je telt de V's van alle signalen op, ook van een signaal dat in de trial ontbreekt. Welke array zegt welke signalen aanwezig zijn, zodat alleen de V's van de aanwezige signalen meetellen?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the surprise of the trial: the food minus what the signals present predict together.",
                      "nl": "Vervang de `...` door de verrassing van de trial: het eten min wat de aanwezige signalen samen voorspellen."},
            "arguments": {"en": "The check calls `{call}`, with the V's, the signals present, the food and a learning rate, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(v, present, food, eta)`?",
                          "nl": "De check roept `{call}` aan, met de V's, de aanwezige signalen, het eten en een learning rate, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(v, present, food, eta)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is `surprise` the food minus what the signals present predict together, one number for the whole trial?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Is `surprise` het eten min wat de aanwezige signalen samen voorspellen, één getal voor de hele trial?"},
            "almost": {"en": "Right for the trials with food. But `{call}` gives {got}, and it should give {want}. Does your function use the `food` it is given?",
                       "nl": "Goed voor de trials met eten. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je functie het `food` dat de functie krijgt?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with the line that starts with `return`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met de regel die met `return` begint?"},
            "correct": {"en": "Right in all {n} cases: with V 0.5 and 0, the surprise is 0.5 and the light gets 0.1; with V 0.99 and 0, the surprise is 0.01 and the light gets 0.002. In the picture, how big is the surprise bar of the second trial?",
                        "nl": "Goed in alle {n} gevallen: met V 0,5 en 0 is de verrassing 0,5 en krijgt de lamp 0,1; met V 0,99 en 0 is de verrassing 0,01 en krijgt de lamp 0,002. Hoe groot is de balk van de verrassing bij de tweede trial in de afbeelding?"},
        },
    ),
    "run": RunBox(
        name="run",
        function="run",
        start=_R_START,
        solution=_R,
        names=R_NAMES,
        compare=same_array,
        show=show_rows,
        visible=[(BLOCKING, ETA)],
        hidden=[(CONTROL, ETA), (INHIBITION, ETA), (BLOCKING, 0.5), (SHORT, 0.3)],
        first=[_DOTS],
        picture=run_picture,
        mistakes=[
            Mistake("swap", wrong=_R_SWAP,
                    en="`{call}` gives {got}, and it should give {want}: `rw_step` gets the learning rate where it expects the food, and the food where it expects the learning rate. In which order does `rw_step` take its values?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `rw_step` krijgt de learning rate waar `rw_step` het eten verwacht, en het eten waar de learning rate hoort. In welke volgorde neemt `rw_step` de waarden?"),
            Mistake("restart", wrong=_R_RESTART,
                    en="`{call}` gives {got}, and it should give {want}: every trial starts again from V's of 0, so the dog forgets what it learned in the trials before. Which V's should a trial start from?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke trial begint weer bij V's van 0, dus vergeet de hond wat de hond in de eerdere trials leerde. Bij welke V's moet een trial beginnen?"),
            Mistake("double", wrong=_R_DOUBLE,
                    en="`{call}` gives {got}, and it should give {want}: `rw_step` already gives the new V's, the old ones plus the step, and you add the old ones once more. What does `rw_step` give back?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `rw_step` geeft al de nieuwe V's, de oude plus de stap, en je telt de oude er nog een keer bij op. Wat geeft `rw_step` terug?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the new `v`: one trial of the Rescorla-Wagner model, with `rw_step`.",
                      "nl": "Vervang de `...` door de nieuwe `v`: één trial van het Rescorla-Wagner model, met `rw_step`."},
            "arguments": {"en": "The check calls `{call}`, with the trials and a learning rate, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(trials, eta)`?",
                          "nl": "De check roept `{call}` aan, met de trials en een learning rate, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(trials, eta)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does every trial give `rw_step` the V's so far, the signals present, the food and the learning rate, in that order?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Geeft elke trial `rw_step` de V's tot nu toe, de aanwezige signalen, het eten en de learning rate, in deze volgorde?"},
            "almost": {"en": "Right for `blocking` with η 0.2. But `{call}` gives {got}, and it should give {want}. Does your function use the `eta` it is given?",
                       "nl": "Goed voor `blocking` met η 0,2. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je functie de `eta` die de functie krijgt?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return np.array(seen)`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return np.array(seen)`?"},
            "correct": {"en": "Right in all {n} cases: in `blocking` the light ends at 0.006, in `control` at 0.50, and in `inhibition` it goes below 0, to −0.81. In the picture, where does the line of the light part from the line of the sound?",
                        "nl": "Goed in alle {n} gevallen: in `blocking` eindigt de lamp op 0,006, in `control` op 0,50, en in `inhibition` gaat de lamp onder 0, naar −0,81. Waar gaat in de afbeelding de lijn van de lamp uit elkaar met de lijn van het geluid?"},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("rw_step", _S_START, "empty", None),
    ("rw_step", _S, "correct", None),
    ("rw_step", _s("food - ..."), "wrong", "dots-left"),
    ("rw_step", _S_OWN, "wrong", "own"),
    ("rw_step", _s("food - present * v"), "wrong", "own"),
    ("rw_step", _S_SIGN, "wrong", "sign"),
    ("rw_step", _S_ALL, "wrong", "all"),
    ("rw_step", _s("food - present @ v.sum()"), "error", "scalar-matmul"),
    ("rw_step", _S.replace("    return v + eta * surprise * present\n", "    v + eta * surprise * present\n"), "wrong", "no-return"),
    # other right answers: the same number written another way
    ("rw_step", _s("food - np.dot(present, v)"), "correct", None),
    ("rw_step", _s("food - (present * v).sum()"), "correct", None),
    # the food written in: right for the trials with food only
    ("rw_step", _S_FIXED, "almost", None),
    ("run", _R_START, "empty", None),
    ("run", _R, "correct", None),
    ("run", _r("rw_step(v, present, ..., eta)"), "wrong", "dots-left"),
    ("run", _R_SWAP, "wrong", "swap"),
    ("run", _R_RESTART, "wrong", "restart"),
    ("run", _R_DOUBLE, "wrong", "double"),
    ("run", _R.replace("    return np.array(seen)\n", ""), "wrong", "no-return"),
    # another right answer: the keyword names of rw_step
    ("run", _r("rw_step(v, present, food=food, eta=eta)"), "correct", None),
    # η written in: right for the example only
    ("run", _R_FIXED, "almost", None),
]
