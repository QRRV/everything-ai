# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rl-milestones: the milestones of RL in games, and self-play in miniature. The game: 21 sticks,
two players take 1, 2 or 3 in turn, and whoever takes the last stick wins. One table V holds, for every number of
sticks, how good the position is for the player who must move; the same table plays both sides. Choosing a move
looks for the position that is worst for the opponent, and learning moves V toward minus the opponent's value.
After a couple of hundred games against itself, the table finds the rule nobody told it: a multiple of 4 is lost.

The game, its numbers and the pictures are the course's own; the milestones in the text are from the papers named
in lesson.yaml."""
import html
import math

import numpy as np

from everything_ai import live, yours
from everything_ai.check import FunctionBox, Mistake, Timeout

LESSON = "rl-milestones"
LANGS = ("en", "nl")
STICKS = 21                    # the sticks at the start of a game
TAKES = (1, 2, 3)              # how many sticks a move may take
ETA = 0.1
EPSILON = 0.1
GAMES = 200                    # the games of self-play in the course's run
SEED = 0
WINNABLE = [s for s in range(1, STICKS + 1) if s % 4]   # the 16 positions where the player to move can win


# --- the functions of the exercises, as the course writes them ----------------------------------------------------------

def choose(V, sticks):
    best = 1
    for take in (1, 2, 3):
        if take <= sticks and V[sticks - take] < V[sticks - best]:
            best = take
    return best


def learn(V, sticks, left, eta):
    target = -V[left]
    return V[sticks] + eta * (target - V[sticks])


# --- the game and the course's self-play -------------------------------------------------------------------------------

def start_table() -> np.ndarray:
    """A table of zeros, with V[0] = -1: the player who must move when no sticks are left has lost."""
    V = np.zeros(STICKS + 1)
    V[0] = -1.0
    return V


def perfect(sticks: int) -> int:
    """The perfect player: leave the opponent a multiple of 4 when it can, else take 1."""
    return sticks % 4 or 1


def wrong_positions(V, choose_=choose) -> list:
    """The winnable positions where `choose_` with the table V does not leave the opponent a multiple of 4."""
    return [s for s in WINNABLE if (s - int(choose_(np.array(V, dtype=float), s))) % 4]


def _legal(sticks: int, take) -> bool:
    return not isinstance(take, bool) and isinstance(take, (int, np.integer)) and 1 <= int(take) <= min(3, sticks)


def _checked_choose(choose_):
    """The learner's choose, made safe for play: a move that is not 1, 2 or 3 sticks, or more than there are, stops
    the run with an error."""
    def checked(V, sticks):
        take = choose_(np.array(V, dtype=float), sticks)
        if not _legal(sticks, take):
            raise ValueError("not a move")
        return int(take)
    return checked


def _checked_learn(learn_):
    def checked(V, sticks, left, eta):
        new = float(learn_(np.array(V, dtype=float), sticks, left, eta))
        if not math.isfinite(new):
            raise ValueError("not a number")
        return new
    return checked


class Run:
    """Self-play: the table after every game, and how many winnable positions it plays wrong after every game."""

    def __init__(self, choose_=choose):
        self.V = start_table()
        self.tables = [self.V.copy()]
        self.wrong = [len(wrong_positions(self.V, choose_))]

    @property
    def games(self) -> int:
        return len(self.tables) - 1


def train(learn_=learn, choose_=choose, games: int = GAMES, eps: float = EPSILON, eta: float = ETA, seed: int = SEED,
          show=None) -> Run:
    """Self-play: one table plays both sides. At every move the player to move picks with ε-greedy, and V of the
    position learns from minus V of the position the opponent gets. `show(run)` after every game."""
    learn_, choose_ = _checked_learn(learn_), _checked_choose(choose_)
    rng = np.random.default_rng(seed)
    run_ = Run(choose_)
    V = run_.V
    for _ in range(games):
        sticks = STICKS
        while sticks > 0:
            if rng.random() < eps:
                take = int(rng.integers(1, min(3, sticks) + 1))
            else:
                take = choose_(V.copy(), sticks)
            V[sticks] = learn_(V.copy(), sticks, sticks - take, eta)
            sticks -= take
        run_.tables.append(V.copy())
        run_.wrong.append(len(wrong_positions(V, choose_)))
        if show is not None:
            show(run_)
    return run_


_RUNS: dict = {}


def course_run(eps: float = EPSILON, games: int = GAMES) -> Run:
    """The course's run: self-play with seed 0."""
    key = (eps, games)
    if key not in _RUNS:
        _RUNS[key] = train(eps=eps, games=games)
    return _RUNS[key]


def first_perfect(run_: Run) -> int | None:
    """The first game after which the table plays every winnable position right, and keeps doing so."""
    for g in range(len(run_.wrong)):
        if not any(run_.wrong[g:]):
            return g
    return None


def play(V, choose_=choose) -> list:
    """A game from 21 sticks: the table moves first, the perfect player second. The moves as (who, sticks, take)."""
    moves, sticks, who = [], STICKS, "table"
    while sticks > 0:
        take = int(choose_(np.array(V, dtype=float), sticks)) if who == "table" else perfect(sticks)
        moves.append((who, sticks, take))
        sticks -= take
        who = "perfect" if who == "table" else "table"
    return moves


# --- numbers and words -------------------------------------------------------------------------------------------------

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


def _and(items: list, lang: str) -> str:
    items = [str(i) for i in items]
    if len(items) < 2:
        return "".join(items)
    return ", ".join(items[:-1]) + (" and " if lang == "en" else " en ") + items[-1]


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


def _figure(svgs: list, alt: str, width: int, gap: str = "0.6rem 1rem") -> str:
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(alt)}" style="margin:0.25rem 0 0.5rem;'
            f'max-width:min(100%,{width}px);display:flex;flex-wrap:wrap;gap:{gap};align-items:flex-start">'
            f'{"".join(svgs)}</figure>')


_BOARD_TEXT = {
    "en": {"value": "V", "take": "takes", "sticks": "sticks left", "zero": "Game 0: a table of zeros",
           "after": "After 200 games: against itself",
           "alt_tables": "Two pictures of the table V for 1 to 21 sticks: a bar up for a position that is good for the player who must move, down for a bad one, and under each position the number of sticks the table takes there. Before self-play, every value is 0, and the table takes 1 stick everywhere except at 2 and 3 sticks, where it takes them all. After 200 games against itself, the values are below 0 at exactly 4, 8, 12, 16 and 20 sticks, and everywhere else the table takes what leaves the opponent a multiple of 4."},
    "nl": {"value": "V", "take": "pakt", "sticks": "stokjes over", "zero": "Partij 0: een tabel met nullen",
           "after": "Na 200 partijen: tegen zichzelf",
           "alt_tables": "Twee afbeeldingen van de tabel V voor 1 tot en met 21 stokjes: een staaf omhoog voor een positie die goed is voor de speler aan zet, omlaag voor een slechte, en onder elke positie het aantal stokjes dat de tabel daar pakt. Voor de self-play is elke waarde 0 en pakt de tabel overal 1 stokje, behalve bij 2 en 3 stokjes: daar pakt de tabel alle stokjes. Na 200 partijen tegen zichzelf zijn de waarden onder 0 bij precies 4, 8, 12, 16 en 20 stokjes, en pakt de tabel overal anders wat de tegenstander een veelvoud van 4 laat."},
}


def _board_svg(V, title: str, lang: str, choose_=choose, course=None, W: int = 330) -> str:
    """The table V for 1 to 21 sticks as bars, in the color of what a model learns, from -1 to 1; the lost positions
    (multiples of 4) shaded. Under it what the table takes at every position: in ink where that wins, in the color of
    mistakes where a winnable position is played wrong, gray at a lost position. With `course` (V and choose), the
    course's values are outlined and its moves marked where they differ."""
    w = _BOARD_TEXT[lang]
    V = np.array(V, dtype=float)
    x0, x1 = 36, W - 6
    cw = (x1 - x0) / STICKS
    y_top = TOP + 6
    half = 46
    zero = y_top + half

    def x_of(s: int) -> float:
        return x0 + (s - 1) * cw

    def y_of(v: float) -> float:
        return zero - max(-1.0, min(1.0, float(v))) * half

    parts = [_title(2, 14, title)]
    for s in range(4, STICKS + 1, 4):
        parts.append(f'<rect x="{x_of(s):.1f}" y="{y_top}" width="{cw:.1f}" height="{2 * half + 40}" '
                     f'style="fill:var(--ea-line);fill-opacity:0.35"/>')
    for v in (1, 0, -1):
        parts.append(_words(x0 - 4, y_of(v) + 3.5, number(v, 0, lang), anchor="end", size=9))
    parts.append(f'<line x1="{x0}" y1="{zero}" x2="{x1}" y2="{zero}" style="stroke:var(--ea-line);stroke-width:1"/>')
    for s in range(1, STICKS + 1):
        v = V[s]
        y = y_of(v)
        top, height = min(y, zero), max(abs(zero - y), 0.6)
        parts.append(f'<rect x="{x_of(s) + cw * 0.18:.1f}" y="{top:.1f}" width="{cw * 0.64:.1f}" height="{height:.1f}" '
                     f'style="fill:var(--ea-param);fill-opacity:0.85"/>')
        if course is not None and abs(float(course[0][s]) - float(v)) > 0.005:
            cy = y_of(course[0][s])
            ctop, ch = min(cy, zero), max(abs(zero - cy), 0.6)
            parts.append(f'<rect x="{x_of(s) + cw * 0.1:.1f}" y="{ctop:.1f}" width="{cw * 0.8:.1f}" height="{ch:.1f}" '
                         f'style="fill:none;stroke:var(--ea-ink-2);stroke-width:1;stroke-dasharray:3 2"/>')
    # what the table takes at every position
    ty = zero + half + 16
    parts.append(_words(x0 - 4, ty, w["take"], anchor="end", size=8.5))
    for s in range(1, STICKS + 1):
        take = int(choose_(V.copy(), s))
        if s % 4 == 0:
            fill = "var(--ea-ink-2)"
        elif (s - take) % 4:
            fill = "var(--ea-loss)"
        else:
            fill = "var(--ea-ink)"
        parts.append(_words(x_of(s) + cw / 2, ty, str(take), size=10.5, fill=fill, weight=600))
        if course is not None and int(course[1](np.array(course[0], dtype=float), s)) != take:
            parts.append(f'<rect x="{x_of(s) + 0.5:.1f}" y="{ty - 11:.1f}" width="{cw - 1:.1f}" height="15" rx="2" '
                         f'style="fill:none;stroke:var(--ea-ink-2);stroke-width:1;stroke-dasharray:3 2"/>')
    by = ty + 16
    for s in range(1, STICKS + 1):
        weight = 600 if s % 4 == 0 else 400
        parts.append(_words(x_of(s) + cw / 2, by, str(s), size=8.5, weight=weight))
    parts.append(_words((x0 + x1) / 2, by + 14, w["sticks"], size=10))
    return _svg(W, by + 20, parts)


def tables_picture(lang: str) -> str:
    """The idea: the table before self-play, and after the course's 200 games."""
    lang = lang if lang in LANGS else "en"
    w = _BOARD_TEXT[lang]
    return _figure([_board_svg(start_table(), w["zero"], lang), _board_svg(course_run().V, w["after"], lang)],
                   w["alt_tables"], 690)


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
    "en": {"choose": {"title": "Your choose",
                      "intro": "The course's table after 200 games of self-play, with the move your `choose` makes at every position. Dashed, the course's move where yours differs.",
                      "chart": "Your choose: with the course's table",
                      "say": "Your `choose` plays {right} of the 16 winnable positions right.",
                      "none": "Your `choose` stops with an error, takes too long, or gives something other than a move of 1, 2 or 3 sticks, at most the sticks there are, so nothing can be drawn yet."},
           "learn": {"title": "Your learn",
                     "intro": "200 games of self-play with your `learn`, and the course's `choose`: the table after them. Dashed, the course's table where yours differs.",
                     "chart": "Your learn: after 200 games",
                     "say": "With your `learn`, the table is below 0 at {lost}, and it plays {right} of the 16 winnable positions right.",
                     "nowhere": "no position",
                     "none": "Your `learn` stops with an error, takes too long, or gives something other than a number, so nothing can be drawn yet."}},
    "nl": {"choose": {"title": "Je choose",
                      "intro": "De tabel van de course na 200 partijen self-play, met de zet die je `choose` bij elke positie kiest. Gestreept de zet van de course waar je eigen zet anders is.",
                      "chart": "Je choose: met de tabel van de course",
                      "say": "Je `choose` speelt {right} van de 16 winbare posities goed.",
                      "none": "Je `choose` stopt met een fout, duurt te lang, of geeft iets anders dan een zet van 1, 2 of 3 stokjes, hooguit zoveel als er liggen, dus is er nog geen afbeelding."},
           "learn": {"title": "Je learn",
                     "intro": "200 partijen self-play met je `learn` en de `choose` van de course: de tabel daarna. Gestreept de tabel van de course waar je eigen tabel anders is.",
                     "chart": "Je learn: na 200 partijen",
                     "say": "Met je `learn` staat de tabel onder 0 bij {lost}, en speelt de tabel {right} van de 16 winbare posities goed.",
                     "nowhere": "geen enkele positie",
                     "none": "Je `learn` stopt met een fout, duurt te lang, of geeft iets anders dan een getal, dus is er nog geen afbeelding."}},
}


def choose_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["choose"]
    fn = _fn(seen)
    V = course_run().V
    got = _call(lambda: wrong_positions(V, _checked_choose(fn))) if fn is not None else None
    if not isinstance(got, list):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    svg = _board_svg(V, w["chart"], lang, choose_=_checked_choose(fn), course=(V, choose))
    say = w["say"].format(right=len(WINNABLE) - len(got))
    return yours.figure(w["title"], _note(w["intro"]) + _figure([svg], w["title"], 330) + _note(say), alt=w["title"])


def learn_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["learn"]
    fn = _fn(seen)
    course_run()   # trained once, outside the time the learner's learn gets
    got = _call(lambda: train(learn_=fn)) if fn is not None else None
    if not isinstance(got, Run):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    svg = _board_svg(got.V, w["chart"], lang, course=(course_run().V, choose))
    lost = [s for s in range(1, STICKS + 1) if got.V[s] < 0]
    say = w["say"].format(lost=_and(lost, lang) if lost else w["nowhere"], right=len(WINNABLE) - got.wrong[-1])
    return yours.figure(w["title"], _note(w["intro"]) + _figure([svg], w["title"], 330) + _note(say), alt=w["title"])


# --- the demo: self-play, then a game against a perfect player --------------------------------------------------------------

GAME_OPTIONS = (25, 50, 100, 200)
EPS_OPTIONS = (0.0, 0.1, 0.3)

DEMO_TEXT = {
    "en": {"title": "Positions played wrong", "x": "game", "y": "wrong",
           "alt": "After every game of self-play, how many of the 16 winnable positions the table still plays wrong.",
           "games": "Games against itself", "eps": "ε, random moves",
           "wrong": "winnable positions played wrong",
           "now": "Game {g}: the table now", "start": "Game 0: a table of zeros",
           "playing": "Game {g} of {n}: {wrong} of 16 winnable positions played wrong",
           "say": "After {g} games, the table plays {wrong} of the 16 winnable positions wrong; it is below 0 at {lost}.",
           "say_start": "Before the first game, every value is 0 except V[0]: the table takes all the sticks at 2 and 3, and 1 stick everywhere else, which is right at only 8 of the 16 winnable positions.",
           "nowhere": "no position",
           "game": "A game against a perfect player", "table": "the table", "perfect": "the perfect player",
           "move": "{who} takes {take} of {sticks}",
           "won_table": "The table wins: it took the last stick.", "won_perfect": "The perfect player wins: the table left a position it could not win.",
           "who_mine": "your `learn` and `choose`", "who_course": "the course's `learn` and `choose` (yours do not pass their checks yet)",
           "who_learn": "your `learn` and the course's `choose` (yours does not pass its check yet)",
           "who_choose": "your `choose` and the course's `learn` (yours does not pass its check yet)",
           "verdict": "Self-play with {who}, ε {eps}, {n} games: the table plays **{wrong}** of the 16 winnable positions wrong at the end, {first}. {won}",
           "first": "and every one right from game **{g}** on", "never": "and not yet every one right"},
    "nl": {"title": "Posities fout gespeeld", "x": "partij", "y": "fout",
           "alt": "Na elke partij self-play: hoeveel van de 16 winbare posities de tabel nog fout speelt.",
           "games": "Partijen tegen zichzelf", "eps": "ε, willekeurige zetten",
           "wrong": "winbare posities fout gespeeld",
           "now": "Partij {g}: de tabel nu", "start": "Partij 0: een tabel met nullen",
           "playing": "Partij {g} van {n}: {wrong} van 16 winbare posities fout gespeeld",
           "say": "Na {g} partijen speelt de tabel {wrong} van de 16 winbare posities fout; de tabel staat onder 0 bij {lost}.",
           "say_start": "Voor de eerste partij is elke waarde 0 behalve V[0]: de tabel pakt bij 2 en 3 stokjes alles, en overal anders 1 stokje, wat bij slechts 8 van de 16 winbare posities goed is.",
           "nowhere": "geen enkele positie",
           "game": "Een partij tegen een perfecte speler", "table": "de tabel", "perfect": "de perfecte speler",
           "move": "{who} pakt {take} van {sticks}",
           "won_table": "De tabel wint: de tabel pakte het laatste stokje.", "won_perfect": "De perfecte speler wint: de tabel liet een positie over waarin de tabel niet kon winnen.",
           "who_mine": "je `learn` en `choose`", "who_course": "de `learn` en `choose` van de course (je eigen functies halen de check nog niet)",
           "who_learn": "je `learn` en de `choose` van de course (je eigen `choose` haalt de check nog niet)",
           "who_choose": "je `choose` en de `learn` van de course (je eigen `learn` haalt de check nog niet)",
           "verdict": "Self-play met {who}, ε {eps}, {n} partijen: aan het eind speelt de tabel **{wrong}** van de 16 winbare posities fout, {first}. {won}",
           "first": "en vanaf partij **{g}** alle posities goed", "never": "en nog niet alle posities goed"},
}


def games_options(lang: str) -> dict:
    return {number(g, 0, lang): g for g in GAME_OPTIONS}


def eps_options(lang: str) -> dict:
    return {number(e, 1, lang) if e else "0": e for e in EPS_OPTIONS}


def chart_labels(lang: str) -> dict:
    w = DEMO_TEXT[lang]
    return {"title": w["title"], "x": w["x"], "y": w["y"], "alt": w["alt"]}


def chart_key(lang: str) -> list:
    return [("loss", False, DEMO_TEXT[lang]["wrong"])]


def demo_xticks(games: int) -> list:
    return [0] + [games * k // 4 for k in range(1, 5)]


def demo_run(learn_, choose_, games: int, eps: float, show=None, pause: float = 0.0, seconds: float = 30.0) -> Run | str:
    """The demo: self-play with the learner's learn and choose; "error" when one of them stops with an error, takes
    too long, or gives something other than a number or a move. `pause` waits that long after each game."""
    import time

    def shown(run_):
        if show is not None:
            show(run_)
        if pause:
            time.sleep(pause)
    got = _call(lambda: train(learn_, choose_, games=games, eps=eps, show=shown), seconds + pause * games)
    return got if isinstance(got, Run) else "error"


def _lost(V, lang: str) -> str:
    lost = [s for s in range(1, STICKS + 1) if V[s] < 0]
    return _and(lost, lang) if lost else DEMO_TEXT[lang]["nowhere"]


def now_picture(run_: Run, lang: str, choose_=choose) -> str:
    w = DEMO_TEXT[lang]
    g = run_.games
    title = w["now"].format(g=number(g, 0, lang)) if g else w["start"]
    return _figure([_board_svg(run_.V, title, lang, choose_=choose_)], title, 330)


def _chart_svg(run_: Run, g: int, lang: str, W: int = 330) -> str:
    """The rewind's chart: the positions played wrong after every game, with a line at game g."""
    w = DEMO_TEXT[lang]
    n = run_.games
    x0, x1, y0, h = 34, W - 18, 28, 140

    def x_of(x: float) -> float:
        return x0 + x / max(n, 1) * (x1 - x0)

    def y_of(v: float) -> float:
        return y0 + h - min(max(v, 0.0), 16.0) / 16.0 * h

    parts = [_words(2, 11, w["title"], anchor="start", size=12, fill="var(--ea-ink)", weight=600)]
    for v in (0, 4, 8, 12, 16):
        parts.append(f'<line x1="{x0}" y1="{y_of(v):.1f}" x2="{x1}" y2="{y_of(v):.1f}" style="stroke:var(--ea-line);stroke-width:0.8"/>')
        parts.append(_words(x0 - 5, y_of(v) + 3.5, str(v), anchor="end", size=9.5))
    shown = run_.wrong[:g + 1]
    line = " ".join(f"{x_of(k):.1f},{y_of(v):.1f}" for k, v in enumerate(shown))
    parts.append(f'<polyline points="{line}" style="fill:none;stroke:var(--ea-loss);stroke-width:2;stroke-linejoin:round"/>')
    parts.append(f'<line x1="{x_of(g):.1f}" y1="{y0}" x2="{x_of(g):.1f}" y2="{y0 + h}" style="stroke:var(--ea-ink-2);stroke-width:1"/>')
    for x in demo_xticks(n):
        parts.append(_words(x_of(x), y0 + h + 14, number(x, 0, lang), size=9.5))
    parts.append(_words((x0 + x1) / 2, y0 + h + 28, w["x"], size=10.5))
    return _svg(W, y0 + h + 34, parts)


def rewind_frame(run_: Run, g: int, lang: str, choose_=choose) -> tuple:
    """A stop of the rewind after the demo: the chart with a line at game g, the table then, and a sentence."""
    lang = lang if lang in LANGS else "en"
    w = DEMO_TEXT[lang]
    g = int(g)
    V = run_.tables[g]
    title = w["now"].format(g=number(g, 0, lang)) if g else w["start"]
    picture = _figure([_board_svg(V, title, lang, choose_=choose_)], title, 330)
    svg = _figure([_chart_svg(run_, g, lang)], w["alt"], 330)
    if g:
        say = w["say"].format(g=number(g, 0, lang), wrong=run_.wrong[g], lost=_lost(V, lang))
    else:
        say = w["say_start"]
    return live.rewound(svg, [picture], width=330), say


def game_picture(run_: Run, lang: str, choose_=choose) -> str:
    """The game of the trained table against the perfect player, move by move."""
    w = DEMO_TEXT[lang]
    moves = play(run_.V, choose_)
    rows = []
    for who, sticks, take in moves:
        name = w["table"] if who == "table" else w["perfect"]
        text_ = w["move"].format(who=name[0].upper() + name[1:], take=take, sticks=sticks)
        color = "var(--ea-ink)" if who == "table" else "var(--ea-ink-2)"
        if who == "table" and sticks % 4 and (sticks - take) % 4:
            color = "var(--ea-loss-text)"
        rows.append(f'<li style="color:{color}">{html.escape(text_)}</li>')
    won = w["won_table"] if moves[-1][0] == "table" else w["won_perfect"]
    return (f'<div style="font-family:var(--ea-font-ui);font-size:0.85rem;max-width:var(--ea-measure)">'
            f'<div style="font-weight:600;color:var(--ea-ink);margin-bottom:0.2rem">{html.escape(w["game"])}</div>'
            f'<ol style="margin:0;padding-left:1.4rem;columns:2;column-gap:1.5rem">{"".join(rows)}</ol>'
            f'<div style="margin-top:0.3rem;color:var(--ea-ink-2)">{html.escape(won)}</div></div>')


def verdict(run_: Run, mine_learn: bool, mine_choose: bool, eps: float, lang: str, choose_=choose) -> str:
    w = DEMO_TEXT[lang]
    if mine_learn and mine_choose:
        who = w["who_mine"]
    elif mine_learn:
        who = w["who_learn"]
    elif mine_choose:
        who = w["who_choose"]
    else:
        who = w["who_course"]
    g = first_perfect(run_)
    first = w["first"].format(g=number(g, 0, lang)) if g is not None else w["never"]
    moves = play(run_.V, choose_)
    won = w["won_table"] if moves[-1][0] == "table" else w["won_perfect"]
    return w["verdict"].format(who=who, eps=number(eps, 1, lang) if eps else "0", n=number(run_.games, 0, lang),
                               wrong=run_.wrong[-1], first=first, won=won)


# --- the exercises -----------------------------------------------------------------------------------------------------

REQUIRED = ["choose", "learn", "from_rules", "league"]


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


def same_take(a, b) -> bool:
    return not isinstance(a, bool) and isinstance(a, (int, np.integer)) and int(a) == int(b)


def same_value(a, b) -> bool:
    try:
        return not isinstance(a, bool) and abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return False


def _num(x) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return repr(x)
    return str(int(v)) if v.is_integer() else repr(round(v, 4))


def show_value(v) -> dict:
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        shown = f"`{_num(v)}`"
    else:
        text_ = repr(v)
        shown = f"`{text_ if len(text_) <= 60 else text_[:57] + '...'}`"
    return {"en": shown, "nl": shown}


TABLE = [-1.0, 0.9, 0.7, 0.5, -0.4, 0.3, 0.1, 0.2]   # a table for 0 to 7 sticks, partly learned: the check's example
ZEROS = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


# exercise 1: choose
def _c(test: str = "V[sticks - take] < V[sticks - best]") -> str:
    return ("def choose(V, sticks):\n"
            "    best = 1\n"
            "    for take in (1, 2, 3):\n"
            f"        if take <= sticks and {test}:\n"
            "            best = take\n"
            "    return best\n")


_C_START = _c("...")
_C = _c()
_C_GREATER = _c("V[sticks - take] > V[sticks - best]")
_C_TAKEN = _c("V[take] < V[best]")
_C_TIES = _c("V[sticks - take] <= V[sticks - best]")


# exercise 2: learn
def _l(line: str = "-V[left]") -> str:
    return ("def learn(V, sticks, left, eta):\n"
            f"    target = {line}\n"
            "    return V[sticks] + eta * (target - V[sticks])\n")


_L_START = _l("...")
_L = _l()
_L_NO_MINUS = _l("V[left]")
_L_SAME = _l("-V[sticks]")
_L_ONE = _l("-V[sticks - 1]")

EXERCISES = {
    "choose": FunctionBox(
        name="choose",
        function="choose",
        start=_C_START,
        solution=_C,
        compare=same_take,
        visible=[(TABLE, 6)],
        hidden=[(TABLE, 1), (TABLE, 2), (TABLE, 7), (ZEROS, 7), (TABLE, 3)],
        first=[_DOTS],
        picture=choose_picture,
        mistakes=[
            Mistake("greater", wrong=_C_GREATER,
                    en="`{call}` gives {got}, and it should give {want}: your `choose` keeps the move that leaves the opponent the highest value, the position that is best for the opponent. Which value should be as low as possible?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je `choose` houdt de zet die de tegenstander de hoogste waarde laat, de positie die het best is voor de tegenstander. Welke waarde moet zo laag mogelijk zijn?"),
            Mistake("taken", wrong=_C_TAKEN,
                    en="`{call}` gives {got}, and it should give {want}: your test looks up the number of sticks a move takes, not the position the opponent gets. How many sticks are left after taking `take`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je test zoekt het aantal stokjes op dat een zet pakt, niet de positie die de tegenstander krijgt. Hoeveel stokjes liggen er nog na `take` pakken?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the test for a better move: the position this move leaves the opponent has a lower value than the position the best move so far leaves.",
                      "nl": "Vervang de `...` door de test voor een betere zet: de positie die deze zet de tegenstander laat, heeft een lagere waarde dan de positie die de beste zet tot nu toe laat."},
            "arguments": {"en": "The check calls `{call}`, with a table and a number of sticks, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(V, sticks)`?",
                          "nl": "De check roept `{call}` aan, met een tabel en een aantal stokjes, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(V, sticks)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does `best` become `take` exactly when `V[sticks - take]` is lower than the value of the position `best` leaves?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Wordt `best` precies `take` als `V[sticks - take]` lager is dan de waarde van de positie die `best` laat?"},
            "almost": {"en": "Right when one move is clearly best. But `{call}` gives {got}, and it should give {want}: when moves have the same value, your `choose` keeps the last of them, and the course's the first. Does `best` change only when a move is strictly better?",
                       "nl": "Goed als één zet duidelijk het best is. Maar `{call}` geeft {got}, en het moet {want} zijn: bij zetten met dezelfde waarde houdt je `choose` de laatste, en de course de eerste. Verandert `best` alleen bij een echt betere zet?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return best`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return best`?"},
            "correct": {"en": "Right in all {n} cases: with 6 sticks, taking 2 leaves the opponent 4, the only one of the three positions below 0. In the picture, what does the table take with 21 sticks, and how many are left for the opponent?",
                        "nl": "Goed in alle {n} gevallen: bij 6 stokjes laat 2 pakken de tegenstander 4 over, als enige van de drie posities onder 0. Wat pakt de tabel in de afbeelding bij 21 stokjes, en hoeveel blijven er over voor de tegenstander?"},
        },
    ),
    "learn": FunctionBox(
        name="learn",
        function="learn",
        start=_L_START,
        solution=_L,
        compare=same_value,
        show=show_value,
        visible=[(TABLE, 5, 4, 0.5)],
        hidden=[(TABLE, 3, 0, 0.5), (TABLE, 7, 4, 0.1), (TABLE, 6, 4, 0.5), (TABLE, 1, 0, 1.0), (ZEROS, 4, 1, 0.1)],
        first=[_DOTS],
        picture=learn_picture,
        mistakes=[
            Mistake("no-minus", wrong=_L_NO_MINUS,
                    en="`{call}` gives {got}, and it should give {want}: `V[left]` says how good the position is for the opponent, who moves next, and your target counts that as good for you. What sign does the opponent's value get from your side?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `V[left]` zegt hoe goed de positie is voor de tegenstander, die daarna aan zet is, en je target telt dat als goed voor jou. Welk teken krijgt de waarde van de tegenstander vanaf jouw kant?"),
            Mistake("same", wrong=_L_SAME,
                    en="`{call}` gives {got}, and it should give {want}: your target looks at the position you were in, not at the position the opponent gets after your move. Which argument holds that position?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je target kijkt naar de positie waar je zelf was, niet naar de positie die de tegenstander na je zet krijgt. Welk argument bevat deze positie?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the target: the value of the position the opponent gets, `left`, seen from your side.",
                      "nl": "Vervang de `...` door het target: de waarde van de positie die de tegenstander krijgt, `left`, gezien vanaf jouw kant."},
            "arguments": {"en": "The check calls `{call}`, with a table, the sticks before and after the move, and η, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(V, sticks, left, eta)`?",
                          "nl": "De check roept `{call}` aan, met een tabel, de stokjes voor en na de zet, en η, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(V, sticks, left, eta)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is `target` the value of the opponent's position `left`, with its sign turned around?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Is `target` de waarde van de positie `left` van de tegenstander, met het teken omgedraaid?"},
            "almost": {"en": "Right when the move took one stick. But `{call}` gives {got}, and it should give {want}: this move took more than one. Which argument holds the position the opponent gets?",
                       "nl": "Goed als de zet één stokje pakte. Maar `{call}` geeft {got}, en het moet {want} zijn: deze zet pakte er meer. Welk argument bevat de positie die de tegenstander krijgt?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with the `return` line?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met de `return`-regel?"},
            "correct": {"en": "Right in all {n} cases: 4 sticks are bad for the opponent, so 5 learns toward 0.4, and taking the last stick learns toward 1. In the picture, at which positions is your table below 0 after 200 games?",
                        "nl": "Goed in alle {n} gevallen: 4 stokjes zijn slecht voor de tegenstander, dus leert 5 richting 0,4, en het laatste stokje pakken leert richting 1. Bij welke posities staat je tabel in de afbeelding na 200 partijen onder 0?"},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("choose", _C_START, "empty", None),
    ("choose", _C, "correct", None),
    ("choose", _c("V[sticks - take] < ..."), "wrong", "dots-left"),
    ("choose", _C_GREATER, "wrong", "greater"),
    ("choose", _C_TAKEN, "wrong", "taken"),
    ("choose", _C.replace("    return best\n", ""), "wrong", "no-return"),
    # other right answers: the same test written another way
    ("choose", _c("V[sticks - best] > V[sticks - take]"), "correct", None),
    ("choose", _c("not V[sticks - take] >= V[sticks - best]"), "correct", None),
    # a tie goes to the last move: right when one move is best, wrong for a table of zeros
    ("choose", _C_TIES, "almost", None),
    ("learn", _L_START, "empty", None),
    ("learn", _L, "correct", None),
    ("learn", _l("-V[...]"), "wrong", "dots-left"),
    ("learn", _L_NO_MINUS, "wrong", "no-minus"),
    ("learn", _L_SAME, "wrong", "same"),
    ("learn", _L.replace("    return ", "    new = "), "wrong", "no-return"),
    # other right answers: the same target written another way
    ("learn", _l("0 - V[left]"), "correct", None),
    ("learn", _l("-1 * V[left]"), "correct", None),
    # the position after taking one stick: right only when the move took one
    ("learn", _L_ONE, "almost", None),
]
