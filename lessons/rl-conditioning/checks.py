# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rl-conditioning: psychology studied learning from reward long before computers. In classical
conditioning a signal comes to predict food (Pavlov); in instrumental conditioning an action followed by a reward
becomes more likely (Thorndike's law of effect). The course's dog (a prediction of food, moved with the update of
Q-learning), the course's cat in a puzzle box (strengths per action, the action that opens the door made stronger),
predictions and choose, the functions the learner writes, the pictures, and the demo.

The experiments are those of Pavlov (Conditioned Reflexes, 1927) and Thorndike (1898; the law of effect in Animal
Intelligence, 1911, p. 244). The models, the numbers and the pictures are the course's own."""
import html
import math
from dataclasses import dataclass

import numpy as np

from everything_ai import live, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, kept

LESSON = "rl-conditioning"
LANGS = ("en", "nl")
ETA = 0.2        # the dog's learning rate
PAIRED = 20      # trials with food after the sound, then as many without
ACTIONS = 6      # what the cat can do in the box; the last one, pulling the cord, opens the door
CORD = ACTIONS - 1   # the place of the cord among the strengths
GAIN = 1.0       # how much stronger the law of effect makes the action that opened the door
TRIALS = 20      # times a cat is put in the box
CATS = 100       # cats of a picture and of the demo, seeds 0 to 99
MANY = 1000      # cats of the numbers in the text, seeds 0 to 999
MAX_TRIES = 1000   # a cat that has not pulled the cord after this many tries is taken out
ACTION_OPTIONS = (4, 6, 10)
GAIN_OPTIONS = (0.5, 1.0, 2.0)

ACTION_NAMES = {
    "en": ["scratch the bars", "bite the bars", "push the door", "reach through", "meow", "claw the floor",
           "climb the side", "sniff a corner", "push the roof"],
    "nl": ["aan de tralies krabben", "in de tralies bijten", "tegen de deur duwen", "door de tralies reiken", "miauwen",
           "aan de vloer krabben", "tegen de wand klimmen", "aan een hoek snuffelen", "tegen het dak duwen"],
}
CORD_NAME = {"en": "pull the cord", "nl": "aan het touwtje trekken"}


def names(actions: int, lang: str) -> list:
    """The names of the actions of a cat with `actions` actions: the first others, and pulling the cord last."""
    return ACTION_NAMES[lang][:actions - 1] + [CORD_NAME[lang]]


# --- the functions of the exercises, as the course writes them ----------------------------------------------------------

def predictions(food, eta):
    v = 0.0
    seen = []
    for f in food:
        seen.append(v)
        v = v + eta * (f - v)
    return seen


def choose(strength, rng):
    chances = strength / strength.sum()
    return rng.choice(len(strength), p=chances)


# --- the course's dog and cats -------------------------------------------------------------------------------------------

class NamedList(list):
    """The food of a run of trials, 1 after the sound and 0 without, that shows its name in a check message, such as
    `pavlov`, instead of all its numbers."""

    def __init__(self, values, name: str):
        super().__init__(values)
        self.name = name

    def __deepcopy__(self, memo):
        return NamedList(list(self), self.name)

    def __repr__(self) -> str:
        return self.name


PAVLOV = NamedList([1] * PAIRED + [0] * PAIRED, "pavlov")        # 20 trials with food, then 20 without
ALTERNATE = NamedList([1, 0] * (PAIRED // 2), "alternate")      # food after every other sound


def expected(actions: int, gain: float, k: int) -> float:
    """The tries a cat needs on average after k escapes: the cord has strength 1 + k * gain against actions - 1 for
    the others, so the cat pulls it with that chance, and an action with chance p takes 1 / p tries on average."""
    return (actions + k * gain) / (1 + k * gain)


def trials(choose_, seed: int, actions: int = ACTIONS, gain: float = GAIN, count: int = TRIALS, history: bool = False):
    """A cat in the puzzle box `count` times, its choices made by `choose_(strength, rng)` with a generator seeded
    `seed`: the tries of every trial until it pulls the cord, the law of effect making the cord `gain` stronger
    after each escape. With `history`, also the strengths at the start and after every trial."""
    rng = np.random.default_rng(seed)
    strength = np.ones(actions)
    cord = actions - 1
    out, seen = [], [strength.copy()]
    for _ in range(count):
        n = 0
        while n < MAX_TRIES:
            n += 1
            action = choose_(strength.copy(), rng)
            if action is None:   # a choose without return: the check says so
                return None
            if int(action) == cord:
                strength[cord] = strength[cord] + gain
                break
        out.append(n)
        seen.append(strength.copy())
    return (out, seen) if history else out


def many(choose_=choose, cats: int = MANY, actions: int = ACTIONS, gain: float = GAIN) -> np.ndarray:
    """The tries of `cats` cats, seeds 0 to cats - 1: a row per cat, a column per trial."""
    return np.array([trials(choose_, s, actions, gain) for s in range(cats)])


def course_many(cats: int = MANY, actions: int = ACTIONS, gain: float = GAIN) -> np.ndarray:
    return kept((LESSON, "many", cats, actions, gain), lambda: many(choose, cats, actions, gain))


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


def _box(x: float, y: float, w: float, h: float, words: str, learned: bool = False, size: float = 11) -> str:
    """A box with a word or two in it; one the dog has learned is drawn in the color of what a model learns."""
    stroke = "var(--ea-param)" if learned else "var(--ea-line)"
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="5" '
            f'style="fill:var(--ea-page);stroke:{stroke};stroke-width:{1.6 if learned else 1.1}"/>'
            + _words(x + w / 2, y + h / 2 + 4, words, size=size, fill="var(--ea-ink)"))


def _plot(values: list, x0: float, y0: float, w: float, h: float, low: float, high: float, color: str, width: float = 2.2,
          dashed: bool = False) -> str:
    """A line through values, one per trial, from trial 1 at x0 to the last trial at x0 + w, between low and high."""
    n = max(len(values) - 1, 1)
    pts = []
    for i, v in enumerate(values):
        shown = min(max(float(v), low), high)
        pts.append(f"{x0 + i / n * w:.1f},{y0 + h - (shown - low) / (high - low) * h:.1f}")
    dash = ";stroke-dasharray:5 4" if dashed else ""
    return (f'<polyline points="{" ".join(pts)}" style="fill:none;stroke:{color};stroke-width:{width};'
            f'stroke-linejoin:round{dash}"/>')


def _frame(x0: float, y0: float, w: float, h: float, low: float, high: float, yticks: list, first: int, last: int,
           xticks: list, lang: str, xlabel: str) -> list:
    """The axes of a chart of trials: a y-axis with its ticks, and the trials along the bottom."""
    parts = [f'<line x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}" style="stroke:var(--ea-line);stroke-width:1"/>',
             f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + h}" style="stroke:var(--ea-line);stroke-width:1"/>']
    for t in yticks:
        y = y0 + h - (t - low) / (high - low) * h
        parts.append(f'<line x1="{x0 - 3}" y1="{y:.1f}" x2="{x0 + w}" y2="{y:.1f}" style="stroke:var(--ea-line);stroke-width:0.6;opacity:0.6"/>')
        digits = 0 if float(t).is_integer() else 1
        parts.append(_words(x0 - 6, y + 4, number(t, digits, lang), anchor="end", size=10.5))
    for t in xticks:
        x = x0 + (t - first) / max(last - first, 1) * w
        parts.append(_words(x, y0 + h + 14, number(t, 0, lang), size=10.5))
    parts.append(_words(x0 + w / 2, y0 + h + 30, xlabel, size=11))
    return parts


# --- the pictures of the idea --------------------------------------------------------------------------------------------

_IDEA_TEXT = {
    "en": {"steps": "Pavlov's dog: before, during and after",
           "before": "before", "during": "20 times", "after": "after",
           "sound": "sound", "food": "food", "saliva": "saliva", "nothing": "nothing",
           "curve": "What the sound predicts: V per trial, η 0.2",
           "with": "food after the sound", "without": "no food", "trial": "trial",
           "alt": "Two pictures. Left: before conditioning, a sound gives nothing and food gives saliva; for 20 trials the sound comes just before the food; after, the sound alone gives saliva. Right: V, what the sound predicts, per trial with a learning rate of 0.2: it rises from 0 toward 1 over 20 trials with food, quickly at first and then more slowly, and falls back toward 0 over 20 trials without food.",
           "box": "The puzzle box: a hungry cat, food outside",
           "cage": "cage", "door": "door", "cord": "cord", "bowl": "food",
           "bars": "The cat's strengths: after 5 escapes",
           "chance": "chance of the cord: 6 of 11, so 0.55",
           "alt3": "A cage with bars, a door in one side, a cord hanging from the roof that opens the door, and food outside.",
           "alt2": "Two pictures. Left: a cage with bars, a door in one side, a cord hanging from the roof that opens the door, and food outside. Right: the strengths of the six actions of the course's cat after 5 escapes: every action has strength 1, except pulling the cord, which has 6, so the cat pulls the cord with a chance of 6 out of 11."},
    "nl": {"steps": "De hond van Pavlov: voor, tijdens en na",
           "before": "voor", "during": "20 keer", "after": "na",
           "sound": "geluid", "food": "eten", "saliva": "speeksel", "nothing": "niets",
           "curve": "Wat het geluid voorspelt: V per trial, η 0,2",
           "with": "eten na het geluid", "without": "geen eten", "trial": "trial",
           "alt": "Twee afbeeldingen. Links: voor de conditionering geeft een geluid niets en eten speeksel; 20 trials lang komt het geluid vlak voor het eten; daarna geeft het geluid alleen al speeksel. Rechts: V, wat het geluid voorspelt, per trial met een learning rate van 0,2: V stijgt van 0 richting 1 over 20 trials met eten, eerst snel en dan langzamer, en zakt terug richting 0 over 20 trials zonder eten.",
           "box": "De puzzelkooi: een hongerige kat, eten buiten",
           "cage": "kooi", "door": "deur", "cord": "touwtje", "bowl": "eten",
           "bars": "De sterktes van de kat: na 5 ontsnappingen",
           "chance": "kans op het touwtje: 6 van 11, dus 0,55",
           "alt3": "Een kooi met tralies, een deur in een zijkant, een touwtje dat van het dak hangt en de deur opent, en eten buiten.",
           "alt2": "Twee afbeeldingen. Links: een kooi met tralies, een deur in een zijkant, een touwtje dat van het dak hangt en de deur opent, en eten buiten. Rechts: de sterktes van de zes actions van de kat van de course na 5 ontsnappingen: elke action heeft sterkte 1, behalve aan het touwtje trekken, met 6, dus trekt de kat met een kans van 6 op 11 aan het touwtje."},
}


def _steps_svg(lang: str, W: int = 330) -> str:
    """Classical conditioning in three rows: before, during and after."""
    w = _IDEA_TEXT[lang]
    parts = [_title(2, 14, w["steps"])]
    bw, bh = 62, 26
    rows = [(w["before"], [(w["sound"], False), None, (w["nothing"], None)], [(w["food"], False), None, (w["saliva"], False)]),
            (w["during"], [(w["sound"], False), (w["food"], False), (w["saliva"], False)], None),
            (w["after"], [(w["sound"], False), None, (w["saliva"], True)], None)]
    y = TOP + 10
    xs = [78, 78 + bw + 26, 78 + 2 * (bw + 26)]
    for label, first, second in rows:
        parts.append(_words(2, y + bh / 2 + 4, label, anchor="start", size=11.5, fill="var(--ea-ink)", weight=600))
        for line in [r for r in (first, second) if r]:
            for i, item in enumerate(line):
                if item is None:
                    continue
                words, learned = item
                if learned is None:
                    parts.append(_words(xs[i] + bw / 2, y + bh / 2 + 4, words, size=11, fill="var(--ea-ink-3)"))
                else:
                    parts.append(_box(xs[i], y, bw, bh, words, learned=learned))
            filled = [i for i, item in enumerate(line) if item is not None]
            for a, b in zip(filled, filled[1:]):
                parts.append(_arrow(xs[a] + bw + 2, y + bh / 2, xs[b] - 2, y + bh / 2))
            y += bh + 8
        y += 12
    return _svg(W, y, parts)


def _curve_svg(lang: str, W: int = 330) -> str:
    w = _IDEA_TEXT[lang]
    values = predictions(PAVLOV, ETA) + [predictions(PAVLOV + [0], ETA)[-1]]
    x0, y0, cw, ch = 40, TOP + 26, W - 52, 150
    parts = [_title(2, 14, w["curve"])]
    mid = x0 + PAIRED / (len(values) - 1) * cw
    parts.append(f'<rect x="{x0}" y="{y0}" width="{mid - x0:.1f}" height="{ch}" style="fill:var(--ea-inset)"/>')
    parts.append(_words((x0 + mid) / 2, y0 - 6, w["with"], size=10.5))
    parts.append(_words((mid + x0 + cw) / 2, y0 - 6, w["without"], size=10.5))
    parts += _frame(x0, y0, cw, ch, 0, 1, [0, 0.5, 1], 1, len(values), [1, 11, 21, 31, 41], lang, w["trial"])
    parts.append(_plot(values, x0, y0, cw, ch, 0, 1, "var(--ea-param)", 2.4))
    return _svg(W, y0 + ch + 40, parts)


def pavlov_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    return _figure([_steps_svg(lang), _curve_svg(lang)], _IDEA_TEXT[lang]["alt"], 690)


def _cage_svg(lang: str, W: int = 330) -> str:
    """The puzzle box, as plain lines: a cage with bars, a door, a cord from the roof, and food outside."""
    w = _IDEA_TEXT[lang]
    parts = [_title(2, 14, w["box"])]
    x0, y0, cw, ch = 30, TOP + 32, 180, 130
    parts.append(f'<rect x="{x0}" y="{y0}" width="{cw}" height="{ch}" style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.6"/>')
    for k in range(1, 9):
        x = x0 + k * cw / 9
        if k < 8:
            parts.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0 + ch}" style="stroke:var(--ea-line);stroke-width:1.2"/>')
    door_x = x0 + cw
    cord_x = x0 + cw * 0.62
    # the door, shut, and the string from the cord over the roof to its latch
    parts.append(f'<line x1="{door_x}" y1="{y0 + 40}" x2="{door_x}" y2="{y0 + ch}" style="stroke:var(--ea-ink);stroke-width:4.5"/>')
    parts.append(f'<circle cx="{door_x}" cy="{y0 + 40}" r="3" style="fill:var(--ea-ink)"/>')
    parts.append(f'<polyline points="{cord_x:.1f},{y0} {cord_x:.1f},{y0 - 10} {door_x + 8},{y0 - 10} {door_x + 8},{y0 + 40} {door_x + 2},{y0 + 40}" '
                 f'style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.1;stroke-dasharray:3 3"/>')
    parts.append(_words(door_x + 14, y0 + 62, w["door"], anchor="start", size=11))
    parts.append(f'<line x1="{cord_x:.1f}" y1="{y0}" x2="{cord_x:.1f}" y2="{y0 + 62}" style="stroke:var(--ea-param);stroke-width:2"/>')
    parts.append(f'<circle cx="{cord_x:.1f}" cy="{y0 + 70}" r="8" style="fill:none;stroke:var(--ea-param);stroke-width:2"/>')
    parts.append(_words(cord_x - 12, y0 + 74, w["cord"], anchor="end", size=11, fill="var(--ea-ink)"))
    bowl_x = door_x + 70
    parts.append(f'<path d="M{bowl_x - 22},{y0 + ch - 18} Q{bowl_x},{y0 + ch + 4} {bowl_x + 22},{y0 + ch - 18} Z" '
                 f'style="fill:var(--ea-inset);stroke:var(--ea-ink-2);stroke-width:1.4"/>')
    parts.append(_words(bowl_x, y0 + ch - 26, w["bowl"], size=11))
    parts.append(_words(x0 + cw / 2, y0 + ch + 18, w["cage"], size=11))
    return _svg(W, y0 + ch + 28, parts)


def bars_svg(strength, title: str, lang: str, note: str = "", W: int = 330) -> str:
    """The strengths of a cat, a bar per action with its name, pulling the cord last, in the color of what a model
    learns; `note` goes under the bars."""
    strength = [float(s) for s in strength]
    labels = names(len(strength), lang)
    left, row = 150, 19
    top = TOP + 6
    biggest = max(max(strength), 1.0)
    span = W - left - 40
    parts = [_title(2, 14, title)]
    for i, (s, label) in enumerate(zip(strength, labels)):
        y = top + i * row
        cord = i == len(strength) - 1
        parts.append(_words(left - 6, y + 13, label, anchor="end", size=11, fill="var(--ea-ink)" if cord else "var(--ea-ink-2)",
                            weight=600 if cord else 400))
        length = s / biggest * span
        parts.append(f'<rect x="{left}" y="{y + 3}" width="{max(length, 1):.1f}" height="{row - 6}" '
                     f'style="fill:var(--ea-param);fill-opacity:{0.95 if cord else 0.55}"/>')
        digits = 0 if s.is_integer() else 1
        parts.append(_words(left + length + 5, y + 13, number(s, digits, lang), anchor="start", size=10.5))
    y = top + len(strength) * row + 16
    if note:
        parts.append(_words(2, y, note, anchor="start", size=11))
        y += 10
    return _svg(W, y + 4, parts)


def cage_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang]
    return _figure([_cage_svg(lang)], w["alt3"], 330)


def cat_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang]
    strength = [1.0] * (ACTIONS - 1) + [1.0 + 5 * GAIN]
    return _figure([_cage_svg(lang), bars_svg(strength, w["bars"], lang, w["chance"])], w["alt2"], 690)


# --- comparing and showing ---------------------------------------------------------------------------------------------

def _fine(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError, OverflowError):
        return False


def same_values(a, b) -> bool:
    """predictions: a list of numbers, each as the course's."""
    try:
        return len(a) == len(b) and all(_fine(x) and abs(float(x) - float(y)) < 1e-9 for x, y in zip(a, b))
    except TypeError:
        return False


def same_tries(a, b) -> bool:
    return isinstance(a, list) and a == b


def _short(v) -> str:
    if isinstance(v, (list, tuple, np.ndarray)) and all(_fine(x) for x in v):
        shown = [str(int(x)) if float(x).is_integer() else f"{round(float(x), 3)!r}" for x in v]
        if len(shown) > 8:
            shown = shown[:6] + ["..."] + shown[-1:]
        return "[" + ", ".join(shown) + "]"
    text_ = repr(v)
    return text_ if len(text_) <= 80 else text_[:77] + "..."


def show_values(v) -> dict:
    shown = f"`{_short(v)}`"
    return {"en": shown, "nl": shown}


@dataclass
class TrialsBox(FunctionBox):
    """choose in a code box: the check puts a cat in the puzzle box 20 times with it, seeded, and compares the tries
    of every trial."""

    def _run(self, fn, case):
        return trials(fn, *case)

    def _call_text(self, case) -> str:
        return f"trials(choose, {case[0]})" if len(case) == 1 else f"trials(choose, {case[0]}, actions={case[1]})"


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


_YOURS_TEXT = {
    "en": {"predictions": {"title": "Your dog",
                           "intro": "V before every trial of `pavlov`, as your `predictions` gives it with η 0.2: 20 trials with food after the sound, then 20 without; dashed, the course's where yours differs, and yours in the color of mistakes.",
                           "panel": "V per trial", "trial": "trial", "with": "food", "without": "no food",
                           "say": "V before trial 21 is {top}, after 20 trials with food; before trial 40 it is {end}, after 19 trials without.",
                           "out": "Some of your values lie outside the frame, from {low} to {high}.",
                           "none": "Your `predictions` stops with an error, takes too long, or gives something other than a list of 40 numbers, so nothing can be drawn yet."},
           "choose": {"title": "Your cats",
                      "intro": "The tries in every trial, the mean of 100 cats that choose with your `choose`; dashed, the tries on average by the chances alone, and the course's cats where yours differ, with yours in the color of mistakes.",
                      "panel": "Tries per trial, the mean of 100 cats", "trial": "trial", "chance": "by the chances",
                      "say": "Your cats need {first} tries on average in the first trial and {last} in the twentieth; by the chances alone, 6 and 1.25.",
                      "none": "Your `choose` stops with an error, takes too long, or gives something other than an action, so the cats cannot be put in the box yet."}},
    "nl": {"predictions": {"title": "Je hond",
                           "intro": "V voor elke trial van `pavlov`, zoals je `predictions` V geeft met η 0,2: 20 trials met eten na het geluid, dan 20 zonder; gestreept de V van de course waar je eigen V anders is, en je eigen V in de kleur van fouten.",
                           "panel": "V per trial", "trial": "trial", "with": "eten", "without": "geen eten",
                           "say": "V voor trial 21 is {top}, na 20 trials met eten; voor trial 40 is V {end}, na 19 trials zonder.",
                           "out": "Een paar van je waarden liggen buiten het kader, van {low} tot {high}.",
                           "none": "Je `predictions` stopt met een fout, duurt te lang, of geeft iets anders dan een lijst van 40 getallen, dus is er nog geen afbeelding."},
           "choose": {"title": "Je katten",
                      "intro": "De pogingen in elke trial, het gemiddelde van 100 katten die kiezen met je `choose`; gestreept de pogingen gemiddeld volgens alleen de kansen, en de katten van de course waar de jouwe anders zijn, met de jouwe in de kleur van fouten.",
                      "panel": "Pogingen per trial, het gemiddelde van 100 katten", "trial": "trial", "chance": "volgens de kansen",
                      "say": "Je katten hebben gemiddeld {first} pogingen nodig in de eerste trial en {last} in de twintigste; volgens alleen de kansen 6 en 1,25.",
                      "none": "Je `choose` stopt met een fout, duurt te lang, of geeft iets anders dan een action, dus kunnen de katten nog niet in de kooi."}},
}


def predictions_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["predictions"]
    fn = _fn(seen)
    got = _call(lambda: list(fn(list(PAVLOV), ETA))) if fn is not None else None
    if not (isinstance(got, list) and len(got) == len(PAVLOV) and all(_fine(v) for v in got)):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    want = predictions(list(PAVLOV), ETA)
    mine = [float(v) for v in got]
    right = same_values(mine, want)
    low, high = -0.2, 1.2
    W, x0, y0, cw, ch = 520, 44, TOP + 26, 452, 170
    parts = [_title(2, 14, w["panel"])]
    mid = x0 + PAIRED / (len(want) - 1) * cw
    parts.append(f'<rect x="{x0}" y="{y0}" width="{mid - x0:.1f}" height="{ch}" style="fill:var(--ea-inset)"/>')
    parts.append(_words((x0 + mid) / 2, y0 - 6, w["with"], size=10.5))
    parts.append(_words((mid + x0 + cw) / 2, y0 - 6, w["without"], size=10.5))
    parts += _frame(x0, y0, cw, ch, low, high, [0, 0.5, 1], 1, len(want), [1, 10, 20, 30, 40], lang, w["trial"])
    if not right:
        parts.append(_plot(want, x0, y0, cw, ch, low, high, "var(--ea-ink-2)", 1.6, dashed=True))
    parts.append(_plot(mine, x0, y0, cw, ch, low, high, "var(--ea-param)" if right else "var(--ea-loss-text)", 2.4))
    svg = _figure([_svg(W, y0 + ch + 40, parts)], w["panel"], W)
    say = w["say"].format(top=number(mine[PAIRED], 2, lang), end=number(mine[-1], 2, lang))
    out = ""
    if min(mine) < low or max(mine) > high:
        out = _note(w["out"].format(low=number(min(mine), 2, lang), high=number(max(mine), 2, lang)), _LOSS_UI)
    return yours.figure(w["title"], _note(w["intro"]) + svg + out + _note(say), alt=w["title"])


def choose_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["choose"]
    fn = _fn(seen)
    got = _call(lambda: many(fn, CATS)) if fn is not None else None
    if not isinstance(got, np.ndarray) or got.shape != (CATS, TRIALS):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    mine = got.mean(axis=0)
    want = course_many(CATS).mean(axis=0)
    right = bool(np.array_equal(got, course_many(CATS)))
    chances = [expected(ACTIONS, GAIN, k) for k in range(TRIALS)]
    high = max(8.0, math.ceil(float(mine.max())) + 1)
    W, x0, y0, cw, ch = 520, 44, TOP + 18, 452, 170
    parts = [_title(2, 14, w["panel"])]
    ticks = list(range(0, int(high) + 1, 2 if high <= 10 else 5))
    parts += _frame(x0, y0, cw, ch, 0, high, ticks, 1, TRIALS, [1, 5, 10, 15, 20], lang, w["trial"])
    parts.append(_plot(chances, x0, y0, cw, ch, 0, high, "var(--ea-ink-2)", 1.4, dashed=True))
    parts.append(_words(x0 + cw, y0 + ch - (chances[-1] / high) * ch - 8, w["chance"], anchor="end", size=10.5))
    if not right:
        parts.append(_plot(list(want), x0, y0, cw, ch, 0, high, "var(--ea-ink-3)", 1.4, dashed=True))
    parts.append(_plot(list(mine), x0, y0, cw, ch, 0, high, "var(--ea-ink)" if right else "var(--ea-loss-text)", 2.4))
    svg = _figure([_svg(W, y0 + ch + 40, parts)], w["panel"], W)
    say = w["say"].format(first=number(float(mine[0]), 2, lang), last=number(float(mine[-1]), 2, lang))
    return yours.figure(w["title"], _note(w["intro"]) + svg + _note(say), alt=w["title"])


# --- the demo ----------------------------------------------------------------------------------------------------------------

DEMO_TEXT = {
    "en": {"title": "Tries per trial: the mean of 100 cats", "x": "trial", "y": "tries",
           "alt": "The mean number of tries of 100 cats in every trial so far, and dashed the tries on average by the chances alone.",
           "actions": "Actions", "gain": "Gain", "chance": "by the chances",
           "cat": "Cat 1: strengths after trial {t}", "start": "Cat 1: strengths before the first trial",
           "say": "After trial {t}, the cord of cat 1 has strength {s} out of {total}, a chance of {p}; in trial {t}, the 100 cats needed {m} tries on average.",
           "who_mine": "your `choose`", "who_course": "the course's `choose`, since yours does not pass its check yet",
           "learning": "Trial {t} of {n}",
           "verdict": "With {who}, the 100 cats needed **{first}** tries on average in the first trial and **{last}** in the last; by the chances alone, {e_first} and {e_last}."},
    "nl": {"title": "Pogingen per trial: het gemiddelde van 100 katten", "x": "trial", "y": "pogingen",
           "alt": "Het gemiddelde aantal pogingen van 100 katten in elke trial tot nu toe, en gestreept de pogingen gemiddeld volgens alleen de kansen.",
           "actions": "Actions", "gain": "Toename", "chance": "volgens de kansen",
           "cat": "Kat 1: sterktes na trial {t}", "start": "Kat 1: sterktes voor de eerste trial",
           "say": "Na trial {t} heeft het touwtje van kat 1 sterkte {s} van {total}, een kans van {p}; in trial {t} hadden de 100 katten gemiddeld {m} pogingen nodig.",
           "who_mine": "je `choose`", "who_course": "de `choose` van de course, omdat je eigen `choose` de check nog niet haalt",
           "learning": "Trial {t} van {n}",
           "verdict": "Met {who} hadden de 100 katten gemiddeld **{first}** pogingen nodig in de eerste trial en **{last}** in de laatste; volgens alleen de kansen {e_first} en {e_last}."},
}


def action_options(lang: str) -> dict:
    return {number(n, 0, lang): n for n in ACTION_OPTIONS}


def gain_options(lang: str) -> dict:
    return {number(g, 1, lang): g for g in GAIN_OPTIONS}


def chart_labels(lang: str) -> dict:
    w = DEMO_TEXT[lang]
    return {"title": w["title"], "x": w["x"], "y": w["y"], "alt": w["alt"]}


def chance_line(actions: int, gain: float) -> list:
    return [(k + 1, expected(actions, gain, k)) for k in range(TRIALS)]


def demo_high(actions: int) -> float:
    """The top of the demo's chart: room above the first trial, which takes `actions` tries on average."""
    return float(actions + 4)


def verdict(run: "Run", mine: bool, lang: str) -> str:
    w = DEMO_TEXT[lang]
    means = run.means()
    who = w["who_mine"] if mine else w["who_course"]
    return w["verdict"].format(who=who, first=number(means[0], 2, lang), last=number(means[-1], 2, lang),
                               e_first=number(expected(run.actions, run.gain, 0), 2, lang),
                               e_last=number(expected(run.actions, run.gain, TRIALS - 1), 2, lang))


class Run:
    """The demo's cats, trial by trial: each with its own generator and strengths, its choices made by `choose_`."""

    def __init__(self, choose_, actions: int, gain: float, cats: int = CATS):
        self.choose, self.actions, self.gain = choose_, actions, gain
        self.rngs = [np.random.default_rng(s) for s in range(cats)]
        self.strengths = [np.ones(actions) for _ in range(cats)]
        self.tries: list = []       # a list per trial: the tries of every cat
        self.first: list = [self.strengths[0].copy()]   # the strengths of cat 1 at the start and after every trial

    def trial(self) -> float:
        """Every cat in the box once; the mean of their tries."""
        cord = self.actions - 1
        out = []
        for rng, strength in zip(self.rngs, self.strengths):
            n = 0
            while n < MAX_TRIES:
                n += 1
                if int(self.choose(strength.copy(), rng)) == cord:
                    strength[cord] = strength[cord] + self.gain
                    break
            out.append(n)
        self.tries.append(out)
        self.first.append(self.strengths[0].copy())
        return float(np.mean(out))

    def means(self) -> list:
        return [float(np.mean(t)) for t in self.tries]


def demo_run(choose_, actions: int, gain: float, show=None, pause: float = 0.0) -> Run | str:
    """The demo: 100 cats, TRIALS trials, `show(t, run)` after every trial; "error" when the learner's choose stops
    with an error or gives something that is not an action. `pause` waits that long after each trial, so that you
    can follow the trials: all of them together take a fraction of a second."""
    import time
    run = Run(choose_, actions, gain)
    try:
        for t in range(1, TRIALS + 1):
            run.trial()
            if show is not None:
                show(t, run)
            if pause:
                time.sleep(pause)
    except Exception:   # noqa: BLE001   the demo says what went wrong
        return "error"
    return run


def now_picture(run: Run, lang: str) -> str:
    """Beside the demo's chart: the strengths of cat 1 now."""
    lang = lang if lang in LANGS else "en"
    t = len(run.tries)
    w = DEMO_TEXT[lang]
    title = w["cat"].format(t=t) if t else w["start"]
    return _figure([bars_svg(run.first[-1], title, lang)], title, 330)


def _trial_panel(run: Run, t: int, lang: str) -> str:
    """The demo's chart as a small picture, with a line at trial t."""
    w = DEMO_TEXT[lang]
    means = run.means()
    chances = [e for _, e in chance_line(run.actions, run.gain)]
    high = max(8.0, math.ceil(max(means + chances)) + 1)
    W, x0, y0, cw, ch = 330, 40, TOP + 18, 276, 150
    parts = [_title(2, 14, w["title"])]
    ticks = list(range(0, int(high) + 1, 2 if high <= 10 else 5))
    parts += _frame(x0, y0, cw, ch, 0, high, ticks, 1, TRIALS, [1, 5, 10, 15, 20], lang, w["x"])
    parts.append(_plot(chances, x0, y0, cw, ch, 0, high, "var(--ea-ink-2)", 1.4, dashed=True))
    parts.append(_plot(means, x0, y0, cw, ch, 0, high, "var(--ea-ink)", 2.2))
    if t:
        x = x0 + (t - 1) / (TRIALS - 1) * cw
        parts.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0 + ch}" style="stroke:var(--ea-ink-2);stroke-width:1.2"/>')
    return _svg(W, y0 + ch + 40, parts)


def rewind_frame(run: Run, t: int, lang: str) -> tuple:
    """A stop of the rewind after the demo: the chart with a line at trial t, the strengths of cat 1 then, and a
    sentence."""
    lang = lang if lang in LANGS else "en"
    w = DEMO_TEXT[lang]
    t = int(t)
    strength = run.first[t]
    title = w["cat"].format(t=t) if t else w["start"]
    picture = _figure([bars_svg(strength, title, lang)], title, 330)
    svg = _figure([_trial_panel(run, t, lang)], w["alt"], 330)
    say = ""
    if t:
        cord, total = float(strength[-1]), float(strength.sum())
        say = w["say"].format(t=t, s=number(cord, 0 if cord.is_integer() else 1, lang),
                              total=number(total, 0 if total.is_integer() else 1, lang), p=number(cord / total, 2, lang),
                              m=number(run.means()[t - 1], 2, lang))
    return live.rewound(svg, [picture], width=330), say


# --- the exercises -----------------------------------------------------------------------------------------------------

REQUIRED = ["predictions", "choose", "kind", "link"]


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


# exercise 1: predictions
def _v(line: str = "v + eta * (f - v)", after: bool = False) -> str:
    body = ([f"        v = {line}\n", "        seen.append(v)\n"] if after
            else ["        seen.append(v)\n", f"        v = {line}\n"])
    return ("def predictions(food, eta):\n"
            "    v = 0.0\n"
            "    seen = []\n"
            "    for f in food:\n"
            + "".join(body)
            + "    return seen\n")


_V_START = _v("...")
_V = _v()
_V_PLUS = _v("v + eta * f")
_V_JUMP = _v("f")
_V_SIGN = _v("v + eta * (v - f)")
_V_STEP = _v("eta * (f - v)")
_V_AFTER = _v(after=True)
_V_FIXED = _v("v + 0.2 * (f - v)")


# exercise 2: choose
def _c(line: str = "strength / strength.sum()") -> str:
    return ("def choose(strength, rng):\n"
            f"    chances = {line}\n"
            "    return rng.choice(len(strength), p=chances)\n")


_C_START = _c("...")
_C = _c()
_C_EQUAL = _c("np.ones(len(strength)) / len(strength)")
_C_RAW = _c("strength")
_C_COUNT = _c("strength / len(strength)")
_C_CORD = _c("strength[-1] / strength.sum()")
C_NAMES = {"np": np}


def _no_sum(e: BaseException) -> bool:
    return isinstance(e, ValueError) and "sum to 1" in str(e)


def _one_number(e: BaseException) -> bool:
    return isinstance(e, TypeError) and "has no len()" in str(e)


EXERCISES = {
    "predictions": FunctionBox(
        name="predictions",
        function="predictions",
        start=_V_START,
        solution=_V,
        compare=same_values,
        show=show_values,
        visible=[(PAVLOV, ETA)],
        hidden=[(PAVLOV, 0.5), (ALTERNATE, ETA), ([1, 0, 1], 0.5)],
        first=[_DOTS],
        picture=predictions_picture,
        mistakes=[
            Mistake("plus-food", wrong=_V_PLUS,
                    en="`{call}` gives {got}, and it should give {want}: you add η times the food to V, so V grows by 0.2 after every trial with food, past 1 and on. How do you move V *toward* the food, by a part of the difference between the food and V?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je telt η keer het eten bij V op, dus groeit V na elke trial met eten met 0,2, voorbij 1 en verder. Hoe beweeg je V *naar* het eten toe, met een deel van het verschil tussen het eten en V?"),
            Mistake("jump", wrong=_V_JUMP,
                    en="`{call}` gives {got}, and it should give {want}: V becomes the food of the last trial at once, so it jumps between 0 and 1. How do you let V take only a step of size η toward the food?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: V wordt meteen het eten van de laatste trial, dus springt V tussen 0 en 1. Hoe laat je V alleen een stap van grootte η naar het eten zetten?"),
            Mistake("sign", wrong=_V_SIGN,
                    en="`{call}` gives {got}, and it should give {want}: with V minus the food, V moves away from the food, below 0. Which of the two comes first in the difference?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: met V min het eten beweegt V van het eten af, onder 0. Welke van de twee komt eerst in het verschil?"),
            Mistake("no-old", wrong=_V_STEP,
                    en="`{call}` gives {got}, and it should give {want}: the new V is only the step, without the old V, so the dog forgets what it learned before. What should the step be added to?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de nieuwe V is alleen de stap, zonder de oude V, dus vergeet de hond wat de hond eerder leerde. Waar moet de stap bij opgeteld worden?"),
            Mistake("after", wrong=_V_AFTER,
                    en="`{call}` gives {got}, and it should give {want}: your list holds V after each trial, so it starts at 0.2 instead of 0. Should `seen` get V before or after the update of a trial?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je lijst bevat V na elke trial, dus begint de lijst bij 0,2 in plaats van 0. Moet `seen` V krijgen voor of na de update van een trial?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the new V: the old V moved a step of size η toward the food `f` of the trial.",
                      "nl": "Vervang de `...` door de nieuwe V: de oude V, een stap van grootte η naar het eten `f` van de trial toe."},
            "arguments": {"en": "The check calls `{call}`, with the food of the trials and a learning rate, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(food, eta)`?",
                          "nl": "De check roept `{call}` aan, met het eten van de trials en een learning rate, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(food, eta)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does V move a step of size η toward the food after every trial, starting from 0?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Zet V na elke trial een stap van grootte η naar het eten toe, vanaf 0?"},
            "almost": {"en": "Right for `pavlov` with η 0.2. But `{call}` gives {got}, and it should give {want}. Does your function use the `eta` it is given?",
                       "nl": "Goed voor `pavlov` met η 0,2. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je functie de `eta` die de functie krijgt?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return seen`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return seen`?"},
            "correct": {"en": "Right in all {n} cases: V is 0.99 before trial 21, after the 20 trials with food, and 0.01 before trial 40. In the picture, is the first step of the rise its biggest?",
                        "nl": "Goed in alle {n} gevallen: V is 0,99 voor trial 21, na de 20 trials met eten, en 0,01 voor trial 40. Is de eerste stap van de stijging in de afbeelding de grootste?"},
        },
    ),
    "choose": TrialsBox(
        name="choose",
        function="choose",
        start=_C_START,
        solution=_C,
        names=C_NAMES,
        compare=same_tries,
        show=show_values,
        visible=[(0,)],
        hidden=[(1,), (2,), (3, 4)],
        first=[_DOTS],
        picture=choose_picture,
        mistakes=[
            Mistake("no-sum", error=_no_sum,
                    en="With `{call}`, `rng.choice` stops: the chances in `chances` must add up to 1, and yours do not, at least once the cord has grown stronger. What can you divide every strength by, so that they add up to 1 whatever the strengths are?",
                    nl="Met `{call}` stopt `rng.choice`: de kansen in `chances` moeten samen 1 zijn, en je kansen zijn dat niet, in elk geval niet meer zodra het touwtje sterker is geworden. Waardoor kun je elke sterkte delen, zodat de kansen samen 1 zijn, wat de sterktes ook zijn?"),
            Mistake("one-number", error=_one_number,
                    en="With `{call}`, `rng.choice` stops: `chances` must hold a chance for every action, an array as long as `strength`, and yours is one number. How do you divide the whole array `strength` at once?",
                    nl="Met `{call}` stopt `rng.choice`: `chances` moet een kans voor elke action bevatten, een array zo lang als `strength`, en je `chances` is één getal. Hoe deel je de hele array `strength` in één keer?"),
            Mistake("equal", wrong=_C_EQUAL,
                    en="`{call}` gives {got}, and it should give {want}: every action gets the same chance, so the strengths do nothing and the cat never gets faster. How do you give a stronger action a bigger chance?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke action krijgt dezelfde kans, dus doen de sterktes niets en wordt de kat nooit sneller. Hoe geef je een sterkere action een grotere kans?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the chance of every action: its strength, divided so that the chances of all actions add up to 1.",
                      "nl": "Vervang de `...` door de kans van elke action: de sterkte van de action, zo gedeeld dat de kansen van alle actions samen 1 zijn."},
            "arguments": {"en": "The check calls your function with the strengths and a random number generator, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(strength, rng)`?",
                          "nl": "De check roept je functie aan met de sterktes en een random number generator, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(strength, rng)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does `chances` hold every strength divided by the same number, so that a twice as strong action gets twice the chance?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Bevat `chances` elke sterkte gedeeld door hetzelfde getal, zodat een twee keer zo sterke action twee keer de kans krijgt?"},
            "almost": {"en": "Right with seed 0. But `{call}` gives {got}, and it should give {want}. Does your function work for any strengths, and any number of actions?",
                       "nl": "Goed met seed 0. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie voor alle sterktes, en elk aantal actions?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with the line that starts with `return`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met de regel die met `return` begint?"},
            "correct": {"en": "Right in all {n} cases: with seed 0, the cat needs 6 tries in the first trial and 1 or 2 in each of the last ten. In the picture, how close do your 100 cats stay to the dashed line?",
                        "nl": "Goed in alle {n} gevallen: met seed 0 heeft de kat 6 pogingen nodig in de eerste trial en 1 of 2 in elk van de laatste tien. Hoe dicht blijven je 100 katten in de afbeelding bij de gestreepte lijn?"},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("predictions", _V_START, "empty", None),
    ("predictions", _V, "correct", None),
    ("predictions", _v("v + eta * ..."), "wrong", "dots-left"),
    ("predictions", _V_PLUS, "wrong", "plus-food"),
    ("predictions", _V_JUMP, "wrong", "jump"),
    ("predictions", _V_SIGN, "wrong", "sign"),
    ("predictions", _V_STEP, "wrong", "no-old"),
    ("predictions", _V_AFTER, "wrong", "after"),
    ("predictions", _V.replace("    return seen\n", ""), "wrong", "no-return"),
    # other right answers: the same step written another way
    ("predictions", _v("(1 - eta) * v + eta * f"), "correct", None),
    ("predictions", _V.replace("        v = v + eta * (f - v)\n", "        v += eta * (f - v)\n"), "correct", None),
    # η written in: right for the example only
    ("predictions", _V_FIXED, "almost", None),
    ("choose", _C_START, "empty", None),
    ("choose", _C, "correct", None),
    ("choose", _c("strength / ..."), "wrong", "dots-left"),
    ("choose", _C_RAW, "error", "no-sum"),
    ("choose", _C_COUNT, "error", "no-sum"),
    ("choose", _C_CORD, "error", "one-number"),
    ("choose", _C_EQUAL, "wrong", "equal"),
    ("choose", _C.replace("    return rng.choice(len(strength), p=chances)\n", "    rng.choice(len(strength), p=chances)\n"), "wrong", "no-return"),
    # other right answers: NumPy's function, and the sum of a Python list
    ("choose", _c("strength / np.sum(strength)"), "correct", None),
    ("choose", _c("strength / sum(strength)"), "correct", None),
]
