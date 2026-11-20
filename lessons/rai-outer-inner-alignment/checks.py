# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-outer-inner-alignment: the course's levels, a grid of 5 by 5 with a start, an exit and
three flowers, and five candidates that each walk their own way through a level, four of them a breadth-first search to
their own target. The base optimizer of this small version keeps the candidates with the highest average reward on the
training levels, in which the exit is always in the top-right corner. The learner writes flower_reward, a reward that
also counts the flowers (outer alignment), and best_objectives, the candidates that tie for the best (inner alignment)."""
import contextlib
import copy
import html
import io
import re
from collections import deque

import numpy as np

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, Timeout, kept, with_time_limit

# --- the levels and the candidates -----------------------------------------------------------------------------------

SIZE = 5                 # a level is a grid of 5 by 5 cells; a cell is (row, column), and row 0 is the top
CORNER = (0, 4)          # the top-right corner: the exit of every training level
FLOWERS = 3              # flowers in a level
EXIT_REWARD = 10         # the course's reward: +10 when the path ends on the exit, and -1 for every step
FLOWER_COST = 5          # flower_reward also takes 5 off for every flower the path steps on
SEED_TRAIN, SEED_TEST, SEED_ELSE = 0, 1, 2
N_TRAIN, N_TEST = 100, 500
ELSEWHERE = [0, 1, 2, 5, 10, 20, 50]   # the demo: training levels (of 100) with the exit somewhere else
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]   # up, right, down, left: the order in which the search looks
CELLS = [(r, c) for r in range(SIZE) for c in range(SIZE)]


def search(start, target, avoid=()):
    """Breadth-first search on the grid from start to target, never through a cell in `avoid`: the cells of a shortest
    way, start and target included, or None when there is no way."""
    before = {start: None}
    frontier = deque([start])
    while frontier:
        cell = frontier.popleft()
        if cell == target:
            way = []
            while cell is not None:
                way.append(cell)
                cell = before[cell]
            return way[::-1]
        for dr, dc in MOVES:
            nxt = (cell[0] + dr, cell[1] + dc)
            if 0 <= nxt[0] < SIZE and 0 <= nxt[1] < SIZE and nxt not in before and nxt not in avoid:
                before[nxt] = cell
                frontier.append(nxt)
    return None


def right_then_up(start):
    """A habit, not a search: to the right until the wall, then up until the wall."""
    way, (r, c) = [start], start
    while c < SIZE - 1:
        c += 1
        way.append((r, c))
    while r > 0:
        r -= 1
        way.append((r, c))
    return way


PLANNERS = {
    "exit, shortest": lambda level: search(level["start"], level["exit"]),
    "exit, around flowers": lambda level: search(level["start"], level["exit"], level["flowers"]),
    "corner, shortest": lambda level: search(level["start"], CORNER),
    "corner, around flowers": lambda level: search(level["start"], CORNER, level["flowers"]),
    "right, then up": lambda level: right_then_up(level["start"]),
}


def walk(name, level):
    """The cells candidate `name` walks through in a level, from the start: its own way, cut off at the exit, since
    the level ends when the exit is reached."""
    way = PLANNERS[name](level)
    return way[:way.index(level["exit"]) + 1] if level["exit"] in way else way


def reward(path, level):
    """The course's reward: +10 when the path ends on the exit, and -1 for every step."""
    return (EXIT_REWARD if path[-1] == level["exit"] else 0) - (len(path) - 1)


def flower_reward(path, level):
    """The course's reward, with 5 off for every flower the path steps on: what the learner writes in Exercise 1."""
    return reward(path, level) - FLOWER_COST * sum(cell in level["flowers"] for cell in path)


class Level(dict):
    """A level, which a message calls by its place, such as TRAIN[7]. The learner's code gets a plain dict."""

    def __init__(self, values: dict, name: str):
        super().__init__(values)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(dict(self), memo)


class Named:
    """A value the check gives the learner's function, shown in a message by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


def make_levels(count: int, seed: int, exit_at=CORNER, name: str = "TRAIN", away_from_corner: bool = False) -> list:
    """`count` levels: a start, an exit (at `exit_at`, or anywhere when it is None) and three flowers, never on the
    start, the exit or the corner, and never so that they close off the way around them to the exit or the corner."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < count:
        start = CELLS[rng.integers(len(CELLS))]
        exit_ = exit_at if exit_at is not None else CELLS[rng.integers(len(CELLS))]
        if exit_ == start or (away_from_corner and exit_ == CORNER):
            continue
        free = [c for c in CELLS if c not in (start, exit_, CORNER)]
        flowers = [free[i] for i in sorted(rng.choice(len(free), FLOWERS, replace=False))]
        if search(start, exit_, flowers) is None or search(start, CORNER, flowers) is None:
            continue
        out.append(Level({"start": start, "exit": exit_, "flowers": flowers}, f"{name}[{len(out)}]"))
    return out


TRAIN = make_levels(N_TRAIN, SEED_TRAIN)                                         # the exit always in the corner
TEST = make_levels(N_TEST, SEED_TEST, exit_at=None, name="TEST")                 # the exit anywhere
ELSE = make_levels(N_TRAIN, SEED_ELSE, exit_at=None, name="ELSE", away_from_corner=True)   # never in the corner
EXAMPLE_TRAIN, EXAMPLE_TEST = 7, 1   # the levels of the worked example


def mixed(k: int) -> list:
    """100 training levels of which the first k have the exit somewhere other than the corner."""
    return ELSE[:k] + TRAIN[k:]


def returns(levels, rewards=reward) -> dict:
    """Per candidate, the reward of its path in each of the levels."""
    return {name: [rewards(walk(name, level), level) for level in levels] for name in PLANNERS}


def averages(levels, rewards=reward) -> dict:
    return {name: float(np.mean(v)) for name, v in returns(levels, rewards).items()}


def best_objectives(returns_):
    """The names with the highest average reward, in the order of the dict: what the learner writes in Exercise 2."""
    means = {name: sum(v) / len(v) for name, v in returns_.items()}
    best = max(means.values())
    return [name for name in means if means[name] == best]


def test_results() -> dict:
    """Per candidate, on the 500 test levels: the share in which it reaches the exit, and the flowers it steps on per
    level. Kept."""
    def make():
        out = {}
        for name in PLANNERS:
            paths_ = [walk(name, level) for level in TEST]
            out[name] = {"reached": float(np.mean([p[-1] == level["exit"] for p, level in zip(paths_, TEST)])),
                         "flowers": float(np.mean([sum(c in level["flowers"] for c in p) for p, level in zip(paths_, TEST)]))}
        return out
    return kept(("rai-outer-inner-alignment", "test"), make)


def train_flowers(name: str) -> float:
    """The flowers candidate `name` steps on per training level."""
    return float(np.mean([sum(c in level["flowers"] for c in walk(name, level)) for level in TRAIN]))


def number(x: float, lang: str, places: int | None = None) -> str:
    """A number for running text: whole numbers without a decimal point (or `places` decimals), a decimal comma in
    Dutch, a real minus sign."""
    x = float(x)
    if places is not None:
        text = f"{x:.{places}f}"
        if float(text) == 0:
            text = text.lstrip("-")
    else:
        text = f"{int(x):d}" if x.is_integer() else f"{x:g}"
    return (text.replace(".", ",") if lang == "nl" else text).replace("-", "−")


def percent(x: float, lang: str) -> str:
    text = f"{100 * x:.1f}".rstrip("0").rstrip(".")
    return (text.replace(".", ",") if lang == "nl" else text) + "%"


# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.15rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;color:var(--ea-ink);vertical-align:top")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"
_CODE = "font-family:var(--ea-font-code);font-size:0.78rem;white-space:nowrap"


def _note(text: str) -> str:
    return f'<div style="{_UI}">{html.escape(text)}</div>'


def _table(rows: list, head: list) -> str:
    """A table with a head row; each row is (its cells, a style of the row), and each cell is (html, style)."""
    top = "".join(f'<th style="{_HEAD}">{html.escape(h)}</th>' for h in head)
    body = "".join(f'<tr style="{style}">' + "".join(f'<td style="{_CELL}{";" + s if s else ""}">{cell}</td>'
                                                    for cell, s in cells_) + "</tr>" for cells_, style in rows)
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.3rem 0">'
            f'<tr>{top}</tr>{body}</table></div>')


def _side(blocks: list) -> str:
    return '<div style="display:flex;flex-wrap:wrap;gap:0.5rem 1.6rem;align-items:flex-start">' + "".join(
        f'<div>{b}</div>' for b in blocks) + '</div>'


def _code(text: str) -> str:
    return f'<span style="{_CODE}">{html.escape(text)}</span>'


# the ways a grid shows: the candidate, its line, and how far it is moved sideways so that ways on the same cells
# stay apart
_WAYS = [("exit, shortest", "stroke:var(--ea-loss);stroke-width:2", -4),
         ("exit, around flowers", "stroke:var(--ea-ink);stroke-width:2.5", 0),
         ("corner, around flowers", "stroke:var(--ea-data);stroke-width:2.5;stroke-dasharray:5 3", 4)]


def _flower(x: float, y: float) -> str:
    petals = "".join(f'<circle cx="{x + dx:.1f}" cy="{y + dy:.1f}" r="3.2" style="fill:var(--ea-ink-3)"/>'
                     for dx, dy in ((0, -4), (4, 0), (0, 4), (-4, 0)))
    return petals + f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.4" style="fill:var(--ea-page)"/>'


def _grid(level, title: str, cell: int = 34) -> str:
    """One level as SVG: the exit washed with E, the start S, the flowers, the corner outlined, and the ways of
    _WAYS."""
    top = 20
    W, H = SIZE * cell + 2, top + SIZE * cell + 2
    parts = [f'<text x="1" y="14" style="font-size:12px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>']
    for r, c in CELLS:
        x, y = 1 + c * cell, top + r * cell
        fill = "var(--ea-data-wash)" if (r, c) == level["exit"] else "var(--ea-page)"
        parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" style="fill:{fill};stroke:var(--ea-line)"/>')
    x, y = 1 + CORNER[1] * cell, top + CORNER[0] * cell
    parts.append(f'<rect x="{x + 2}" y="{y + 2}" width="{cell - 4}" height="{cell - 4}" '
                 f'style="fill:none;stroke:var(--ea-ink-3);stroke-width:1;stroke-dasharray:2 2"/>')
    for r, c in level["flowers"]:
        parts.append(_flower(1 + c * cell + cell / 2, top + r * cell + cell / 2))
    for name, style, shift in _WAYS:
        path = walk(name, level)
        if len(path) > 1:
            pts = " ".join(f"{1 + c * cell + cell / 2 + shift:.1f},{top + r * cell + cell / 2 + shift:.1f}" for r, c in path)
            parts.append(f'<polyline points="{pts}" style="fill:none;{style};stroke-linecap:round;stroke-linejoin:round"/>')
    for (r, c), sign in ((level["start"], "S"), (level["exit"], "E")):
        parts.append(f'<text x="{1 + c * cell + cell / 2:.1f}" y="{top + r * cell + cell / 2 + 4:.1f}" '
                     f'style="font-size:12px;font-weight:700;fill:var(--ea-ink);text-anchor:middle;paint-order:stroke;'
                     f'stroke:var(--ea-page);stroke-width:3px">{sign}</text>')
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}" style="width:{W}px;'
            f'max-width:100%;height:auto;display:block;font-family:var(--ea-font-ui)">{"".join(parts)}</svg>')


def _legend() -> str:
    parts, x = [], 1
    for name, style, _ in _WAYS:
        parts.append(f'<line x1="{x}" y1="9" x2="{x + 22}" y2="9" style="{style}"/>')
        parts.append(f'<text x="{x + 28}" y="13" style="font-size:11px;fill:var(--ea-ink-2);'
                     f'font-family:var(--ea-font-code)">{html.escape(name)}</text>')
        x += 40 + 7.2 * len(name)
    return (f'<svg viewBox="0 0 {x:.0f} 18" aria-hidden="true" style="width:{x:.0f}px;max-width:100%;height:auto;'
            f'display:block;margin-top:0.3rem">{"".join(parts)}</svg>')


_LEVEL_WORDS = {
    "en": {"title": "Three candidates in a training level and in a test level",
           "train": "Training level {i}", "test": "Test level {i}",
           "note": "S: the start. E: the exit; in every training level it is in the top-right corner, the cell with "
                   "the dotted outline. The other marks are flowers. In the training level, \"exit, around flowers\" and "
                   "\"corner, around flowers\" walk the same cells; in the test level they part ways."},
    "nl": {"title": "Drie kandidaten in een trainingslevel en in een testlevel",
           "train": "Trainingslevel {i}", "test": "Testlevel {i}",
           "note": "S: de start. E: de uitgang; in elk trainingslevel ligt die in de hoek rechtsboven, het vakje met de "
                   "gestippelde rand. De andere tekens zijn bloemen. In het trainingslevel lopen \"exit, around flowers\" "
                   "en \"corner, around flowers\" over dezelfde vakjes; in het testlevel gaan ze uit elkaar."},
}


def level_picture(lang: str) -> str:
    w = _LEVEL_WORDS[lang]
    body = _side([_grid(TRAIN[EXAMPLE_TRAIN], w["train"].format(i=EXAMPLE_TRAIN)),
                  _grid(TEST[EXAMPLE_TEST], w["test"].format(i=EXAMPLE_TEST))]) + _legend() + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"] + ". " + w["note"])


_DEMO_WORDS = {
    "en": {"title": "The exit elsewhere in {k} of the 100 training levels",
           "title0": "The exit in the corner in all 100 training levels",
           "head": ["candidate", "average flower_reward in training", "best", "reaches the exit in the test",
                    "flowers per test level"],
           "best": "best",
           "tie": "With the exit in the top-right corner in all 100 training levels, \"{a}\" and \"{b}\" tie for the "
                  "best average flower_reward, both {v}. In the test, \"{a}\" reaches the exit in {ra} of the levels, "
                  "\"{b}\" in {rb}.",
           "one": "With the exit elsewhere in {k} of the 100 training levels, only \"{a}\" is best: {v}, against {v2} "
                  "for \"{b}\". In the test, \"{a}\" reaches the exit in {ra} of the levels, \"{b}\" in {rb}.",
           "note": "The test: 500 levels with the exit in any cell. The candidates do not learn; the base optimizer only "
                   "keeps the best of them."},
    "nl": {"title": "De uitgang ergens anders in {k} van de 100 trainingslevels",
           "title0": "De uitgang in de hoek in alle 100 trainingslevels",
           "head": ["kandidaat", "gemiddelde flower_reward in de training", "beste", "bereikt de uitgang in de test",
                    "bloemen per testlevel"],
           "best": "beste",
           "tie": "Met de uitgang in de hoek rechtsboven in alle 100 trainingslevels delen \"{a}\" en \"{b}\" de beste "
                  "gemiddelde flower_reward, allebei {v}. In de test bereikt \"{a}\" de uitgang in {ra} van de levels, "
                  "\"{b}\" in {rb}.",
           "one": "Met de uitgang ergens anders in {k} van de 100 trainingslevels is alleen \"{a}\" de beste: {v}, tegen "
                  "{v2} voor \"{b}\". In de test bereikt \"{a}\" de uitgang in {ra} van de levels, \"{b}\" in {rb}.",
           "note": "De test: 500 levels met de uitgang in een willekeurig vakje. De kandidaten leren niet; de base "
                   "optimizer houdt alleen de beste over."},
}

_RIGHT, _CORNER_WAY = "exit, around flowers", "corner, around flowers"   # the two that tie when every exit is in the corner


def demo_frame(k, lang: str) -> tuple[str, str]:
    """The demo: with the exit elsewhere in k of the 100 training levels, each candidate's average flower_reward in
    training, which ones the base optimizer keeps, and how each does in the test."""
    k = min(ELSEWHERE, key=lambda x: abs(x - float(k)))
    w = _DEMO_WORDS[lang]
    rets = returns(mixed(k), flower_reward)
    means = {name: float(np.mean(v)) for name, v in rets.items()}
    kept_ = best_objectives(rets)
    tests = test_results()
    rows = []
    for name in PLANNERS:
        on = name in kept_
        rows.append(([(_code(name), ""), (number(means[name], lang, 2), "font-weight:600" if on else ""),
                      (w["best"] if on else "", "font-weight:600"), (percent(tests[name]["reached"], lang), ""),
                      (number(tests[name]["flowers"], lang, 2), "")], ""))
    title = w["title0"] if k == 0 else w["title"].format(k=k)
    words = dict(a=_RIGHT, b=_CORNER_WAY, ra=percent(tests[_RIGHT]["reached"], lang),
                 rb=percent(tests[_CORNER_WAY]["reached"], lang), v=number(means[_RIGHT], lang, 2),
                 v2=number(means[_CORNER_WAY], lang, 2), k=k)
    assert kept_ == ([_RIGHT, _CORNER_WAY] if k == 0 else [_RIGHT]), kept_
    said = (w["tie"] if k == 0 else w["one"]).format(**words)
    picture = yours.figure(title, _table(rows, w["head"]) + _note(w["note"]), alt=title + ". " + said)
    return picture, said


# --- the pictures under the checks -----------------------------------------------------------------------------------

def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _all gives for a function that stopped with an error


def _all(fn, calls: list):
    """The learner's function on copies of each list of arguments in `calls`, all within one time limit of a check;
    _STOPPED when it stops with an error, and the Timeout passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(lambda: [fn(*(copy.deepcopy(a) for a in args)) for args in calls])
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _number_like(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


_FLOWER_WORDS = {
    "en": {"title": "The 100 training levels: the average reward of each candidate",
           "head": ["candidate", "the course's reward", "the course's flower_reward", "yours"],
           "best": "Best with the course's reward: {a}. Best with flower_reward: {b}. Best with yours: {c}.",
           "note": "A dashed row is one where your flower_reward gives something other than the course's.",
           "stopped": "error", "none": "nothing"},
    "nl": {"title": "De 100 trainingslevels: de gemiddelde reward van elke kandidaat",
           "head": ["kandidaat", "reward van de course", "flower_reward van de course", "de jouwe"],
           "best": "Beste met de reward van de course: {a}. Beste met flower_reward: {b}. Beste met de jouwe: {c}.",
           "note": "Een gestippelde rij is er een waar jouw flower_reward iets anders geeft dan die van de course.",
           "stopped": "fout", "none": "niets"},
}


def flower_reward_picture(seen) -> str:
    """Per candidate, its average reward over the training levels with the course's reward, with flower_reward and with
    the learner's, and which candidates are best with each."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w, lang = _FLOWER_WORDS[seen.lang], seen.lang
    plain, flowers = averages(TRAIN, reward), averages(TRAIN, flower_reward)
    calls = [(walk(name, level), level) for name in PLANNERS for level in TRAIN]
    got = _all(fn, calls)
    mine = {}
    if got is not _STOPPED and all(_number_like(v) for v in got):
        for i, name in enumerate(PLANNERS):
            mine[name] = float(np.mean(got[i * len(TRAIN):(i + 1) * len(TRAIN)]))
    rows = []
    for name in PLANNERS:
        shown = number(mine[name], lang, 2) if name in mine else w["stopped"]
        off = name not in mine or abs(mine[name] - flowers[name]) > 1e-9
        rows.append(([(_code(name), ""), (number(plain[name], lang, 2), ""), (number(flowers[name], lang, 2), ""),
                      (html.escape(shown), "font-weight:600")], _OFF if off else ""))

    def tops(values: dict) -> str:
        if not values:
            return w["none"]
        top = max(values.values())
        return "; ".join(n for n in values if abs(values[n] - top) < 1e-9)
    said = w["best"].format(a=tops(plain), b=tops(flowers), c=tops(mine))
    return yours.figure(w["title"], _table(rows, w["head"]) + _note(said) + _note(w["note"]),
                        alt=w["title"] + ". " + said)


_BEST_WORDS = {
    "en": {"title": "Which candidates the base optimizer keeps",
           "settings": ["100 training levels, the course's reward", "100 training levels, flower_reward",
                        "10 of them with the exit elsewhere, flower_reward"],
           "head": ["training", "the course's best_objectives", "yours"],
           "note": "A dashed row is one where your best_objectives gives something other than the course's.",
           "stopped": "error"},
    "nl": {"title": "Welke kandidaten de base optimizer overhoudt",
           "settings": ["100 trainingslevels, reward van de course", "100 trainingslevels, flower_reward",
                        "10 ervan met de uitgang ergens anders, flower_reward"],
           "head": ["training", "best_objectives van de course", "de jouwe"],
           "note": "Een gestippelde rij is er een waar jouw best_objectives iets anders geeft dan die van de course.",
           "stopped": "fout"},
}


def _settings() -> list:
    return [returns(TRAIN, reward), returns(TRAIN, flower_reward), returns(mixed(10), flower_reward)]


def best_objectives_picture(seen) -> str:
    """Three kinds of training, with the candidates the course's best_objectives keeps and those the learner's keeps."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _BEST_WORDS[seen.lang]
    sets = _settings()
    got = _all(fn, [(s,) for s in sets])
    rows, said = [], []
    for i, (label, s) in enumerate(zip(w["settings"], sets)):
        want = best_objectives(s)
        if got is _STOPPED:
            shown, off = html.escape(w["stopped"]), True
        else:
            mine = got[i]
            off = not (isinstance(mine, list) and mine == want)
            if isinstance(mine, list) and all(isinstance(n, str) for n in mine):
                shown = "<br>".join(_code(n) for n in mine) or "[]"
            else:
                text = repr(mine)
                shown = html.escape(text if len(text) <= 40 else text[:37] + "...")
        rows.append(([(html.escape(label), ""), ("<br>".join(_code(n) for n in want), ""), (shown, "font-weight:600")],
                     _OFF if off else ""))
        said.append(f"{label}: {'; '.join(want)}.")
    return yours.figure(w["title"], _table(rows, w["head"]) + _note(w["note"]), alt=w["title"] + ". " + " ".join(said))


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["flower_reward", "best_objectives", "vacuum", "claim"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _names_error(e) -> bool:
    return isinstance(e, NameError) and re.search(r"'(TRAIN|TEST|PLANNERS|C|walk|reward)'", str(e)) is not None


_NAMES = Mistake("lesson-names", error=_names_error,
                 en="Your function uses a name of the lesson, such as `TRAIN`, `reward` or `C`, and the check does not give it that name: use the names between the parentheses of the `def` line.",
                 nl="Je functie gebruikt een naam van de les, zoals `TRAIN`, `reward` of `C`, en de check geeft die naam niet mee: gebruik de namen tussen de haakjes van de `def`-regel.")


def _level(start, exit_, flowers) -> dict:
    return {"start": start, "exit": exit_, "flowers": flowers}


# exercise 1: flower_reward; the first case is the shortest way to the exit in training level 7, through two flowers
_EX = TRAIN[EXAMPLE_TRAIN]
FLOWER_CASES = [(Named(f"walk(\"exit, shortest\", TRAIN[{EXAMPLE_TRAIN}])", walk("exit, shortest", _EX)), _EX),
                ([(2, 2), (1, 2), (0, 2)], _level((2, 2), (0, 2), [(1, 1), (3, 3), (4, 0)])),
                ([(4, 0), (3, 0)], _level((4, 0), (0, 4), [(3, 0), (2, 0), (1, 1)])),
                ([(1, 1)], _level((1, 1), (0, 0), [(2, 2), (3, 3), (4, 4)])),
                ([(3, 3), (3, 4), (2, 4), (1, 4)], _level((3, 3), (1, 4), [(3, 4), (2, 4), (0, 0)]))]
_F_START = "def flower_reward(path, level):\n    ...\n"
_F = ("def flower_reward(path, level):\n"
      "    total = -(len(path) - 1)\n"
      "    if path[-1] == level[\"exit\"]:\n"
      "        total = total + 10\n"
      "    for cell in path:\n"
      "        if cell in level[\"flowers\"]:\n"
      "            total = total - 5\n"
      "    return total\n")
_F_NO_STEPS = _F.replace("    total = -(len(path) - 1)\n", "    total = 0\n")
_F_LEN = _F.replace("    total = -(len(path) - 1)\n", "    total = -len(path)\n")
_F_ONCE = _F.replace("            total = total - 5\n", "            total = total - 5\n            break\n")
_F_NO_EXIT = _F.replace("    if path[-1] == level[\"exit\"]:\n        total = total + 10\n", "")

# exercise 2: best_objectives; the first three are the course's training, the others small dicts of its own
BEST_CASES = [(Named("returns(TRAIN, reward)", returns(TRAIN, reward)),),
              (Named("returns(TRAIN, flower_reward)", returns(TRAIN, flower_reward)),),
              (Named("returns(mixed(10), flower_reward)", returns(mixed(10), flower_reward)),),
              ({"a": [3], "b": [2, 2]},),
              ({"a": [1, 3], "b": [2, 2], "c": [0, 1]},),
              ({"x": [-1, -3], "y": [-4, -2], "z": [-5, -5]},)]
_B_START = "def best_objectives(returns):\n    ...\n"
_B = ("def best_objectives(returns):\n"
      "    means = {}\n"
      "    for name in returns:\n"
      "        means[name] = sum(returns[name]) / len(returns[name])\n"
      "    best = max(means.values())\n"
      "    tied = []\n"
      "    for name in means:\n"
      "        if means[name] == best:\n"
      "            tied.append(name)\n"
      "    return tied\n")
_B_FIRST = _B.replace("            tied.append(name)\n", "            tied.append(name)\n            break\n")
_B_SUM = _B.replace("sum(returns[name]) / len(returns[name])", "sum(returns[name])")
_B_MIN = _B.replace("best = max(means.values())", "best = min(means.values())")
_B_ALL = _B.replace("        if means[name] == best:\n            tied.append(name)\n", "        tied.append(name)\n")


EXERCISES = {
    "flower_reward": FunctionBox(
        name="flower_reward",
        picture=flower_reward_picture,
        function="flower_reward",
        start=_F_START,
        solution=_F,
        visible=[FLOWER_CASES[0]],
        hidden=FLOWER_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("no-steps", wrong=_F_NO_STEPS,
                    en="`{call}` gives {got}, and it should give {want}: every step costs 1, and a path of n cells has n − 1 steps. Start the total with minus the number of steps.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke stap kost 1, en een pad van n vakjes heeft n − 1 stappen. Begin het totaal met min het aantal stappen."),
            Mistake("len", wrong=_F_LEN,
                    en="`{call}` gives {got}, and it should give {want}: the path also holds the start cell, so a path of n cells has n − 1 steps, not n.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het pad bevat ook het startvakje, dus een pad van n vakjes heeft n − 1 stappen, niet n."),
            Mistake("once", wrong=_F_ONCE,
                    en="`{call}` gives {got}, and it should give {want}: this path steps on two flowers, and each costs 5. Go on through the whole path, also after the first flower.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dit pad stapt op twee bloemen, en elke bloem kost 5. Loop het hele pad door, ook na de eerste bloem."),
            Mistake("no-exit", wrong=_F_NO_EXIT,
                    en="`{call}` gives {got}, and it should give {want}: a path that ends on the exit gets 10 extra, as in the course's reward. Compare the last cell of the path with the exit of the level.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een pad dat op de uitgang eindigt, krijgt 10 extra, zoals in de reward van de course. Vergelijk het laatste vakje van het pad met de uitgang van het level."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: take 1 off for every step, add 10 when the path ends on the exit, and take 5 off for every flower on the path.",
                      "nl": "Vervang de `...`: trek 1 af voor elke stap, tel er 10 bij op als het pad op de uitgang eindigt, en trek 5 af voor elke bloem op het pad."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The reward is minus the number of steps (the cells of the path minus 1), plus 10 when the last cell is the exit, minus 5 for every cell of the path that is a flower.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De reward is min het aantal stappen (de vakjes van het pad min 1), plus 10 als het laatste vakje de uitgang is, min 5 voor elk vakje van het pad dat een bloem is."},
            "almost": {"en": f"Right for the shortest way in training level {EXAMPLE_TRAIN}. But `{{call}}` gives {{got}}, and it should give {{want}}. Does your function use the `path` and `level` it gets?",
                       "nl": f"Goed voor de kortste weg in trainingslevel {EXAMPLE_TRAIN}. Maar `{{call}}` geeft {{got}}, en dat moet {{want}} zijn. Gebruikt je functie het `path` en `level` die ze krijgt?"},
            "correct": {"en": "Your `flower_reward` is right in all {n} cases. Under the check, the candidates that go around the flowers now come out best in training.",
                        "nl": "Je `flower_reward` klopt in alle {n} gevallen. Onder de check komen de kandidaten die om de bloemen heen gaan nu als beste uit de training."},
        },
    ),
    "best_objectives": FunctionBox(
        name="best_objectives",
        picture=best_objectives_picture,
        function="best_objectives",
        start=_B_START,
        solution=_B,
        visible=[BEST_CASES[0]],
        hidden=BEST_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("first", wrong=_B_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: more than one candidate has the highest average here, and they all belong in the list. Go on through all names.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: hier heeft meer dan één kandidaat het hoogste gemiddelde, en die horen allemaal in de lijst. Loop alle namen door."),
            Mistake("sum", wrong=_B_SUM,
                    en="`{call}` gives {got}, and it should give {want}: \"b\" has the larger sum, but over more levels. Compare averages: the sum divided by the number of rewards.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: \"b\" heeft de grootste som, maar over meer levels. Vergelijk gemiddelden: de som gedeeld door het aantal rewards."),
            Mistake("min", wrong=_B_MIN,
                    en="`{call}` gives {got}, and it should give {want}: those are the candidates with the lowest average. The base optimizer keeps the highest: `max`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat zijn de kandidaten met het laagste gemiddelde. De base optimizer houdt de hoogste over: `max`."),
            Mistake("all", wrong=_B_ALL,
                    en="`{call}` gives {got}, and it should give {want}: that is every name. Keep only the names whose average equals the highest.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat zijn alle namen. Houd alleen de namen over waarvan het gemiddelde gelijk is aan het hoogste."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: work out the average reward of every name, find the highest, and return a list of the names that have it.",
                      "nl": "Vervang de `...`: reken het gemiddelde van elke naam uit, zoek het hoogste, en geef een lijst terug met de namen die dat hebben."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The answer is a list of every name whose average reward equals the highest average, in the order of the dict.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Het antwoord is een lijst met elke naam waarvan het gemiddelde gelijk is aan het hoogste gemiddelde, in de volgorde van de dict."},
            "almost": {"en": "Right for the course's training levels with its reward. But `{call}` gives {got}, and it should give {want}. Does your function use the `returns` it gets?",
                       "nl": "Goed voor de trainingslevels van de course met zijn reward. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `returns` die ze krijgt?"},
            "correct": {"en": "Your `best_objectives` is right in all {n} cases. With the course's reward, three candidates tie; with flower_reward, two.",
                        "nl": "Je `best_objectives` klopt in alle {n} gevallen. Met de reward van de course delen drie kandidaten de beste plaats; met flower_reward twee."},
        },
    ),
}

CASES = [
    ("flower_reward", _F_START, "empty", None),
    ("flower_reward", _F, "correct", None),
    ("flower_reward", "def flower_reward(path, level):\n    steps = len(path) - 1\n    bonus = 10 if path[-1] == level[\"exit\"] else 0\n    trampled = len([c for c in path if c in level[\"flowers\"]])\n    return bonus - steps - 5 * trampled\n", "correct", None),
    ("flower_reward", _F_NO_STEPS, "wrong", "no-steps"),
    ("flower_reward", _F_LEN, "wrong", "len"),
    ("flower_reward", _F_ONCE, "wrong", "once"),
    ("flower_reward", _F_NO_EXIT, "wrong", "no-exit"),
    ("flower_reward", f"def flower_reward(path, level):\n    return {flower_reward(FLOWER_CASES[0][0].value, _EX)}\n", "almost", None),
    ("flower_reward", "def flower_reward(path, level):\n    return reward(path, level)\n", "error", "lesson-names"),
    ("best_objectives", _B_START, "empty", None),
    ("best_objectives", _B, "correct", None),
    ("best_objectives", "import numpy as np\n\n\ndef best_objectives(returns):\n    means = {name: np.mean(returns[name]) for name in returns}\n    return [name for name in means if means[name] == max(means.values())]\n", "correct", None),
    ("best_objectives", _B_FIRST, "wrong", "first"),
    ("best_objectives", _B_SUM, "wrong", "sum"),
    ("best_objectives", _B_MIN, "wrong", "min"),
    ("best_objectives", _B_ALL, "wrong", "all"),
    ("best_objectives", f"def best_objectives(returns):\n    return {best_objectives(returns(TRAIN, reward))!r}\n", "almost", None),
    ("best_objectives", "def best_objectives(returns):\n    return list(PLANNERS)\n", "error", "lesson-names"),
]
