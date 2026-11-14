# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson build-automate-work: the pairs of steps that directly follow each other in the made-up event
log of a webshop's returns, and the days each return took. And the pictures: the directly-follows graph of the
learner's pairs, and the days of the returns with and without photos asked. The two logs are made by `make_log`."""
import html
import math
import random
import re

import pandas as pd   # here, so the first import of pandas never counts against the time limit of a check

from everything_ai.check import Code, CodeBox, Mistake
from everything_ai import yours

LOG = "lessons/build-automate-work/returns.csv"
LOG_2 = "lessons/build-automate-work/returns-2.csv"
REQUESTED, LABEL, RECEIVED, CHECKED, PHOTOS, REFUND, REFUSED = (
    "Return requested", "Label sent", "Package received", "Package checked", "Photos asked", "Refund paid", "Return refused")
STEPS = [REQUESTED, LABEL, RECEIVED, CHECKED, PHOTOS, REFUND, REFUSED]
# The process as the webshop wrote it down: every return goes this way.
DRAWN = [REQUESTED, LABEL, RECEIVED, CHECKED, REFUND]


def make_log(seed: int, n: int) -> pd.DataFrame:
    """A made-up event log of n returns, in the order of the returns and, within one, of the steps. Most go as drawn;
    some customers send the package without the label; a check can ask for photos and check again, up to three
    times; a few are refused, and a few refunds are paid twice. Only random.random() is used, which gives the same
    numbers for a seed in every version of Python."""
    rng = random.Random(seed)
    rows = []
    for i in range(1, n + 1):
        case, day = f"R{i:03d}", 0
        rows.append((case, REQUESTED, day))
        if rng.random() < 0.85:
            day += 1 + int(rng.random() * 2)
            rows.append((case, LABEL, day))
        day += 2 + int(rng.random() * 4)
        rows.append((case, RECEIVED, day))
        day += 1 + int(rng.random() * 3)
        rows.append((case, CHECKED, day))
        loops = 0
        while loops < 3 and rng.random() < (0.25 if loops == 0 else 0.3):
            day += 3 + int(rng.random() * 5)
            rows.append((case, PHOTOS, day))
            day += 2 + int(rng.random() * 3)
            rows.append((case, CHECKED, day))
            loops += 1
        if rng.random() < 0.9:
            day += 1 + int(rng.random() * 2)
            rows.append((case, REFUND, day))
            if rng.random() < 0.03:
                day += 1 + int(rng.random() * 6)
                rows.append((case, REFUND, day))
        else:
            day += 1 + int(rng.random() * 2)
            rows.append((case, REFUSED, day))
    return pd.DataFrame(rows, columns=["case", "activity", "day"])


MADE = {LOG: (2026, 300), LOG_2: (7, 40)}   # the seed and the number of returns of each file


def log(path: str = LOG) -> pd.DataFrame:
    return pd.read_csv(path)


def pairs_of(df: pd.DataFrame) -> list:
    """Every two steps of one return that directly follow each other, in the order of the log."""
    cases, steps = list(df["case"]), list(df["activity"])
    return [(steps[i], steps[i + 1]) for i in range(len(steps) - 1) if cases[i] == cases[i + 1]]


def counted(pairs: list) -> dict:
    """How often each pair occurs, most often first; pairs as often as each other in the order they first occur."""
    out = {}
    for p in pairs:
        out[p] = out.get(p, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def took(df: pd.DataFrame) -> pd.Series:
    """The days each return took: the day of its last step, since its first step is on day 0."""
    return df.groupby("case")["day"].max()


def with_photos(df: pd.DataFrame) -> set:
    return set(df[df["activity"] == PHOTOS]["case"])


def facts(path: str = LOG) -> dict:
    """The numbers the texts say about a log."""
    df = log(path)
    days = took(df)
    photo = with_photos(df)
    c = counted(pairs_of(df))
    twice = sorted(set(df[df["activity"] == REFUND].groupby("case").size().loc[lambda s: s > 1].index))
    return {"returns": df["case"].nunique(), "rows": len(df), "pairs": sum(c.values()), "kinds": len(c),
            "no_label": c.get((REQUESTED, RECEIVED), 0), "photos": c.get((CHECKED, PHOTOS), 0), "photo_cases": len(photo),
            "twice": len(twice), "refused": c.get((CHECKED, REFUSED), 0),
            "days_photos": days[days.index.isin(photo)].mean(), "days_other": days[~days.index.isin(photo)].mean(),
            "as_drawn": sum(1 for _, g in df.groupby("case") if list(g["activity"]) == DRAWN)}


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


# --- the directly-follows graph --------------------------------------------------------------------------------------

_AT = {REQUESTED: (64, 150), LABEL: (210, 58), RECEIVED: (356, 150), CHECKED: (524, 150), PHOTOS: (524, 270),
       REFUND: (692, 66), REFUSED: (692, 234)}
_HW, _HH = 58, 19   # half the width and height of a box


def _edge_point(a: tuple, b: tuple, pad: float = 4) -> tuple:
    """Where the line from the middle of box a towards b leaves box a."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = min(_HW / abs(dx) if dx else math.inf, _HH / abs(dy) if dy else math.inf)
    n = math.hypot(dx, dy) or 1
    return a[0] + dx * t + dx / n * pad, a[1] + dy * t + dy / n * pad


def dfg_svg(pairs: list, lang: str, title: str) -> str:
    """The directly-follows graph of `pairs`: a box per step, an arrow from each step to every step that directly
    follows it in a return, as thick as how often, with that count beside it. A step that follows itself gets a loop
    above its box. Steps the log does not have are left out. The boxes have the names of the log, in both languages."""
    c = {p: n for p, n in counted(pairs).items() if p[0] in _AT and p[1] in _AT}
    top = max(c.values(), default=1)
    parts = ['<defs><marker id="ea-dfg-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="10" markerHeight="10" '
             'markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--ea-ink-2)"/></marker></defs>']
    for (a, b), n in c.items():
        w = 1 + 4 * n / top
        if a == b:
            x, y = _AT[a]
            parts.append(f'<path d="M{x + 20} {y - _HH} C{x + 30} {y - _HH - 34}, {x + 58} {y - _HH - 20}, {x + 46} {y - _HH - 2}" '
                         f'style="fill:none;stroke:var(--ea-ink-2);stroke-width:{w:.1f}" marker-end="url(#ea-dfg-arrow)"/>'
                         f'<text x="{x + 52}" y="{y - _HH - 24}" style="font-size:12px;fill:var(--ea-ink);font-variant-numeric:tabular-nums">{n}</text>')
            continue
        p, q = _AT[a], _AT[b]
        dx, dy = q[0] - p[0], q[1] - p[1]
        L = math.hypot(dx, dy)
        ox, oy = -dy / L * 16, dx / L * 16   # bend to the left of the direction, so a pair and its reverse stay apart
        mid = ((p[0] + q[0]) / 2 + ox, (p[1] + q[1]) / 2 + oy)
        s, e = _edge_point(p, mid), _edge_point(q, mid, pad=6)
        on = ((s[0] + 2 * mid[0] + e[0]) / 4, (s[1] + 2 * mid[1] + e[1]) / 4)   # the middle of the curve itself
        parts.append(f'<path d="M{s[0]:.1f} {s[1]:.1f} Q{mid[0]:.1f} {mid[1]:.1f} {e[0]:.1f} {e[1]:.1f}" '
                     f'style="fill:none;stroke:var(--ea-ink-2);stroke-width:{w:.1f}" marker-end="url(#ea-dfg-arrow)"/>'
                     f'<text x="{on[0] + ox * 0.9:.1f}" y="{on[1] + oy * 0.9 + 4:.1f}" text-anchor="middle" '
                     f'style="font-size:12px;fill:var(--ea-ink);font-variant-numeric:tabular-nums;paint-order:stroke;'
                     f'stroke:var(--ea-page);stroke-width:3px">{n}</text>')
    for step, (x, y) in _AT.items():
        parts.append(f'<rect x="{x - _HW}" y="{y - _HH}" width="{2 * _HW}" height="{2 * _HH}" rx="6" '
                     f'style="fill:var(--ea-page);stroke:var(--ea-ink);stroke-width:1.2"/>'
                     f'<text x="{x}" y="{y + 4}" text-anchor="middle" style="font-size:12px;fill:var(--ea-ink)">{html.escape(step)}</text>')
    return (f'<svg viewBox="0 -6 760 306" role="img" style="width:100%;max-width:46rem;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(title)}</title>{"".join(parts)}</svg>')


def course_graph(lang: str) -> str:
    title = {"en": "The directly-follows graph of the 300 returns", "nl": "De directly-follows graph van de 300 retouren"}[lang]
    return f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem">{dfg_svg(pairs_of(log()), lang, title)}</figure>'


def _pairs_picture(seen) -> str:
    """The graph of the learner's pairs, drawn from the value of their last line, while it is a list of pairs."""
    mine = seen.mine
    got = getattr(mine, "value", None) if mine is not None else None
    if not isinstance(got, list) or not got or not all(isinstance(p, tuple) and len(p) == 2 for p in got):
        return ""
    title = seen.word({"en": "Your directly-follows graph", "nl": "Jouw directly-follows graph"})
    return yours.figure(title, dfg_svg(got, seen.lang, title), alt=title)


def _days_picture(seen) -> str:
    """The learner's days per return as dots on a line of days, one row for the returns with photos asked and one for
    the rest, each with its mean as a short line."""
    mine = seen.mine
    got = getattr(mine, "value", None) if mine is not None else None
    if type(got).__name__ != "Series" or got.empty or got.dtype.kind not in "iuf":
        return ""
    photo = with_photos(seen.names.get("log") if isinstance(seen.names.get("log"), pd.DataFrame) else log())
    rows = [(seen.word({"en": "photos asked", "nl": "foto's gevraagd"}), [v for k, v in got.items() if k in photo]),
            (seen.word({"en": "no photos", "nl": "geen foto's"}), [v for k, v in got.items() if k not in photo])]
    hi = max(40, int(got.max()) + 2)
    X0, X1 = 130, 660
    sx = lambda d: X0 + (X1 - X0) * d / hi
    parts = []
    for r, (label, vals) in enumerate(rows):
        y = 34 + r * 46
        parts.append(f'<text x="0" y="{y + 4}" style="font-size:12px;fill:var(--ea-ink)">{html.escape(label)} ({len(vals)})</text>')
        seen_at = {}
        for v in vals:
            k = seen_at[v] = seen_at.get(v, 0) + 1
            parts.append(f'<circle cx="{sx(v):.1f}" cy="{y - 8 + (k % 5) * 4}" r="2.4" style="fill:var(--ea-ink-2);fill-opacity:0.7"/>')
        if vals:
            m = sum(vals) / len(vals)
            parts.append(f'<line x1="{sx(m):.1f}" x2="{sx(m):.1f}" y1="{y - 16}" y2="{y + 16}" style="stroke:var(--ea-ink);stroke-width:2.5"/>'
                         f'<text x="{sx(m) + 5:.1f}" y="{y - 12}" style="font-size:12px;fill:var(--ea-ink)">'
                         f'{number(m, 1, seen.lang)}</text>')
    axis_y = 34 + 2 * 46 - 10
    parts.append(f'<line x1="{X0}" x2="{X1}" y1="{axis_y}" y2="{axis_y}" style="stroke:var(--ea-line)"/>')
    for d in range(0, hi + 1, 10):
        parts.append(f'<text x="{sx(d):.1f}" y="{axis_y + 15}" text-anchor="middle" style="font-size:12px;fill:var(--ea-ink-2)">{d}</text>')
    days = seen.word({"en": "days", "nl": "dagen"})
    parts.append(f'<text x="{X1}" y="{axis_y + 30}" text-anchor="end" style="font-size:12px;fill:var(--ea-ink-2)">{days}</text>')
    title = seen.word({"en": "How long each return took", "nl": "Hoe lang elke retour duurde"})
    svg = (f'<svg viewBox="0 0 680 {axis_y + 36}" role="img" style="width:100%;max-width:42rem;height:auto;font-family:var(--ea-font-ui)">'
           f'<title>{title}</title>{"".join(parts)}</svg>')
    return yours.figure(title, svg, alt=title)


# The lesson is done when these are correct. "which-automation" and "never-twice" are choice questions in the lesson.
REQUIRED = ["which-automation", "pairs", "took", "never-twice"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_READ = f'log = pd.read_csv("{LOG}")'
_LISTS = 'cases = list(log["case"])\nsteps = list(log["activity"])'
_PAIRS_START = "import pandas as pd\n\n" + _READ + "\n" + _LISTS + "\npairs = []\n...\npairs\n"
_LOOP = ("for i in range(len(steps) - 1):\n    if cases[i] == cases[i + 1]:\n        pairs.append((steps[i], steps[i + 1]))")
_PAIRS = "import pandas as pd\n\n" + _READ + "\n" + _LISTS + "\npairs = []\n" + _LOOP + "\npairs"


def _pairs(loop: str) -> str:
    return _PAIRS.replace(_LOOP, loop)


def _is_pairs(got) -> bool:
    return isinstance(got, list) and all(isinstance(p, tuple) and len(p) == 2 for p in got)


def _across(got) -> bool:
    """Pairs that also join the last step of one return to the first step of the next."""
    return _is_pairs(got) and (REFUND, REQUESTED) in got


EXERCISES = {
    "pairs": CodeBox(
        name="pairs",
        picture=_pairs_picture,
        start=_PAIRS_START,
        solution=_PAIRS,
        keep=["import pandas as pd", _READ, 'cases = list(log["case"])', 'steps = list(log["activity"])'],
        # The check also runs the box with another log, of 40 returns.
        others=[{"log": Code(f'pd.read_csv("{LOG_2}")')}],
        first=[_DOTS],
        mistakes=[
            Mistake("index", error=lambda e: isinstance(e, IndexError),
                    en="Your code asks for a step after the last one: with `i + 1`, `i` may go up to the last but one row. Let the loop stop one sooner: `range(len(steps) - 1)`.",
                    nl="Je code vraagt om een stap na de laatste: met `i + 1` mag `i` tot de op één na laatste rij gaan. Laat de loop één eerder stoppen: `range(len(steps) - 1)`."),
            Mistake("across", matches=lambda got, text: _across(got),
                    en="Your pairs also join two returns: the last step of one, such as `Refund paid`, to `Return requested` of the next. Only count two steps when they belong to the same return: `if cases[i] == cases[i + 1]:`.",
                    nl="Je paren verbinden ook twee retouren: de laatste stap van de ene, zoals `Refund paid`, met `Return requested` van de volgende. Tel twee stappen alleen als ze bij dezelfde retour horen: `if cases[i] == cases[i + 1]:`."),
            Mistake("steps-only", matches=lambda got, text: isinstance(got, list) and bool(got) and set(map(str, got)) <= set(STEPS) and not _is_pairs(got),
                    en="Your list holds single steps. Each item should be a pair of two steps that follow each other: `(steps[i], steps[i + 1])`.",
                    nl="Je lijst bevat losse stappen. Elk item moet een paar zijn van twee stappen die op elkaar volgen: `(steps[i], steps[i + 1])`."),
            Mistake("text-pairs", matches=lambda got, text: isinstance(got, list) and bool(got) and all(isinstance(p, str) for p in got),
                    en="Your list holds texts, one for each pair. The graph needs each pair as two steps between parentheses, a tuple: `(steps[i], steps[i + 1])`.",
                    nl="Je lijst bevat teksten, één voor elk paar. De directly-follows graph heeft elk paar nodig als twee stappen tussen haakjes, een tuple: `(steps[i], steps[i + 1])`."),
            Mistake("reversed", wrong=_pairs("for i in range(len(steps) - 1):\n    if cases[i] == cases[i + 1]:\n        pairs.append((steps[i + 1], steps[i]))"),
                    en="Your pairs are the wrong way round: each has the later step first. Put the step that comes first, `steps[i]`, first.",
                    nl="Je paren staan andersom: in elk staat de latere stap eerst. Zet de stap die eerst komt, `steps[i]`, vooraan."),
            Mistake("short", wrong=_pairs("for i in range(len(steps) - 2):\n    if cases[i] == cases[i + 1]:\n        pairs.append((steps[i], steps[i + 1]))"),
                    en="Your code misses the last pair of the log: the loop stops one too soon. With `range(len(steps) - 1)`, `i + 1` reaches the last row.",
                    nl="Je code mist het laatste paar van het log: de loop stopt één te vroeg. Met `range(len(steps) - 1)` komt `i + 1` bij de laatste rij."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a loop that adds every two steps of one return that follow each other to `pairs`.",
                      "nl": "Vervang de `...` door een loop die elke twee stappen van één retour die op elkaar volgen aan `pairs` toevoegt."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries another log by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf een ander log."},
            "no_value": {"en": "The last line of the box shows nothing. Put `pairs` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `pairs` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives something else than the 1,402 pairs of steps that directly follow each other within one return, in the order of the log. Do you go through the rows with `i`, and add `(steps[i], steps[i + 1])` when both belong to the same return?",
                      "nl": "Je laatste regel geeft iets anders dan de 1.402 paren van stappen die binnen één retour direct op elkaar volgen, in de volgorde van het log. Ga je met `i` langs de rijen, en voeg je `(steps[i], steps[i + 1])` toe als ze allebei bij dezelfde retour horen?"},
            "other": {"en": "Right for this log. But with {where}, your last line gives other pairs. Do you work them out from `cases` and `steps` themselves?",
                      "nl": "Goed voor dit log. Maar met {where} geeft je laatste regel andere paren. Reken je ze uit met `cases` en `steps` zelf?"},
            "other_error": {"en": "Right for this log. But with {where}, your code stops with an error. Do you work it out from `cases` and `steps` themselves?",
                            "nl": "Goed voor dit log. Maar met {where} stopt je code met een fout. Reken je het uit met `cases` en `steps` zelf?"},
            "correct": {"en": "All 1,402 pairs, in the order of the log; the graph under the box counts them. The check also tried a log of 40 returns, and your code found its pairs too.",
                        "nl": "Alle 1.402 paren, in de volgorde van het log; de directly-follows graph onder het vak telt ze. De check probeerde ook een log van 40 retouren, en je code vond ook daar de paren."},
        },
    ),
}


def _kind(got) -> str:
    return type(got).__name__


def _same_days(a, b) -> bool:
    """The same days for the same returns: a Series with the returns as its row labels."""
    from everything_ai.check import same_value
    return _kind(a) == "Series" and _kind(b) == "Series" and [str(k) for k in a.index] == [str(k) for k in b.index] and same_value(a, b)


_TOOK_START = "import pandas as pd\n\n" + _READ + "\ntook = ...\ntook\n"
_TOOK = "import pandas as pd\n\n" + _READ + '\ntook = log.groupby("case")["day"].max()\ntook'


def _took(new: str) -> str:
    return _TOOK.replace('took = log.groupby("case")["day"].max()', new)


EXERCISES["took"] = CodeBox(
    name="took",
    picture=_days_picture,
    start=_TOOK_START,
    solution=_TOOK,
    keep=["import pandas as pd", _READ],
    others=[{"log": Code(f'pd.read_csv("{LOG_2}")')}],
    first=[_DOTS],
    compare=_same_days,
    digits=1,
    mistakes=[
        Mistake("no-column", error=lambda e: isinstance(e, KeyError),
                en="The log has no column with that name. Its columns are `case`, `activity` and `day`, in small letters, each between quotes.",
                nl="Het log heeft geen kolom met die naam. De kolommen zijn `case`, `activity` en `day`, in kleine letters, elk tussen aanhalingstekens."),
        Mistake("one-number", matches=lambda got, text: isinstance(got, (int, float)) or _kind(got) in ("int64", "float64"),
                en="Your last line gives one number, `{got}`, and the task asks for the days of each return. Split the rows into returns first: `log.groupby(\"case\")`.",
                nl="Je laatste regel geeft één getal, `{got}`, en de opdracht vraagt om de dagen van elke retour. Splits de rijen eerst in retouren: `log.groupby(\"case\")`."),
        Mistake("mean", wrong=_took('took = log.groupby("case")["day"].mean()'),
                en="Your last line gives the mean day of the steps of each return. A return took as long as its last step: the largest day, `.max()`.",
                nl="Je laatste regel geeft de gemiddelde dag van de stappen van elke retour. Een retour duurde zo lang als zijn laatste stap: de grootste dag, `.max()`."),
        Mistake("count", wrong=_took('took = log.groupby("case")["day"].count()'),
                en="Your last line gives how many steps each return had, not how many days it took. Take the largest day: `.max()`.",
                nl="Je laatste regel geeft hoeveel stappen elke retour had, niet hoeveel dagen hij duurde. Neem de grootste dag: `.max()`."),
        Mistake("sum", wrong=_took('took = log.groupby("case")["day"].sum()'),
                en="Your last line adds up the days of all steps of a return. Each day is counted from the start of the return, so the last step's day is how long it took: `.max()`.",
                nl="Je laatste regel telt de dagen van alle stappen van een retour op. Elke dag telt vanaf het begin van de retour, dus de dag van de laatste stap is hoe lang hij duurde: `.max()`."),
        Mistake("by-activity", matches=lambda got, text: _kind(got) == "Series" and set(map(str, got.index)) <= set(STEPS),
                en="Your last line gives a number for each step, not for each return. Group by the return: `log.groupby(\"case\")`.",
                nl="Je laatste regel geeft een getal voor elke stap, niet voor elke retour. Groepeer op de retour: `log.groupby(\"case\")`."),
        Mistake("groups-only", matches=lambda got, text: _kind(got) in ("SeriesGroupBy", "DataFrameGroupBy"),
                en="That is the groups themselves, not yet a number per return. Ask for the largest day: `.max()` at the end.",
                nl="Dat zijn de groepen zelf, nog geen getal per retour. Vraag om de grootste dag: `.max()` aan het eind."),
        Mistake("table", matches=lambda got, text: _kind(got) == "DataFrame",
                en="Your last line gives a table with several columns. Take only the column `day`, with `[\"day\"]`, before you ask for the largest.",
                nl="Je laatste regel geeft een tabel met meerdere kolommen. Neem alleen de kolom `day`, met `[\"day\"]`, voordat je om de grootste vraagt."),
    ],
    texts={
        "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                    "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
        "empty": {"en": "Replace the `...` with the days of each return: the day of its last step.",
                  "nl": "Vervang de `...` door de dagen van elke retour: de dag van zijn laatste stap."},
        "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                    "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
        "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries another log by itself.",
                 "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf een ander log."},
        "no_value": {"en": "The last line of the box shows nothing. Put `took` on the last line, so the check can see it.",
                     "nl": "De laatste regel van het vak laat niets zien. Zet `took` op de laatste regel, zodat de check het kan zien."},
        "wrong": {"en": "Your last line gives:\n\n```\n{got}\n```\n\nand it should be the days each of the 300 returns took: for each return, the day of its last step. Do you group by `case`, and take the largest `day`?",
                  "nl": "Je laatste regel geeft:\n\n```\n{got}\n```\n\nen dat moeten de dagen zijn die elk van de 300 retouren duurde: voor elke retour de dag van zijn laatste stap. Groepeer je op `case`, en neem je de grootste `day`?"},
        "other": {"en": "Right for this log. But with {where}, your last line gives:\n\n```\n{got}\n```\n\nand it should give:\n\n```\n{want}\n```\n\nDo you work it out from `log` itself?",
                  "nl": "Goed voor dit log. Maar met {where} geeft je laatste regel:\n\n```\n{got}\n```\n\nen dat moet dit zijn:\n\n```\n{want}\n```\n\nReken je het uit met `log` zelf?"},
        "other_error": {"en": "Right for this log. But with {where}, your code stops with an error. Do you work it out from `log` itself?",
                        "nl": "Goed voor dit log. Maar met {where} stopt je code met een fout. Reken je het uit met `log` zelf?"},
        "correct": {"en": "The days of all 300 returns, each the day of its last step. The picture under the box splits them: the returns where photos were asked took far longer. The check also tried the log of 40 returns.",
                    "nl": "De dagen van alle 300 retouren, elk de dag van zijn laatste stap. Het plaatje onder het vak splitst ze: de retouren waarbij foto's werden gevraagd, duurden veel langer. De check probeerde ook het log van 40 retouren."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_P = "import pandas as pd\n\n" + _READ + "\n" + _LISTS + "\npairs = []\n"
_T = "import pandas as pd\n\n" + _READ + "\n"
CASES = [
    ("pairs", _PAIRS_START, "empty", None),
    ("pairs", _PAIRS, "correct", None),
    ("pairs", _P + "for i in range(1, len(steps)):\n    if cases[i - 1] == cases[i]:\n        pairs.append((steps[i - 1], steps[i]))\npairs", "correct", None),
    ("pairs", _P + "for i in range(len(steps) - 1):\n    if cases[i] != cases[i + 1]:\n        continue\n    pairs.append((steps[i], steps[i + 1]))\npairs", "correct", None),
    ("pairs", _P + "for i in range(len(steps) - 1):\n    pairs.append((steps[i], steps[i + 1]))\npairs", "wrong", "across"),
    ("pairs", _P + "for i in range(len(steps)):\n    if cases[i] == cases[i + 1]:\n        pairs.append((steps[i], steps[i + 1]))\npairs", "error", "index"),
    ("pairs", _P + "for i in range(len(steps) - 1):\n    if cases[i] == cases[i + 1]:\n        pairs.append(steps[i] + ' -> ' + steps[i + 1])\npairs", "wrong", "text-pairs"),
    ("pairs", _P + "for i in range(len(steps) - 1):\n    if cases[i] == cases[i + 1]:\n        pairs.append((steps[i + 1], steps[i]))\npairs", "wrong", "reversed"),
    ("pairs", _P + "for i in range(len(steps) - 2):\n    if cases[i] == cases[i + 1]:\n        pairs.append((steps[i], steps[i + 1]))\npairs", "wrong", "short"),
    ("pairs", _P + "for i in range(len(steps) - 1):\n    if cases[i] == cases[i + 1]:\n        pairs.append(steps[i])\npairs", "wrong", "steps-only"),
    ("pairs", _P + "for i in range(len(steps) - 1):\n    if cases[i] == cases[i + 1]:\n        pairs.append((steps[i], steps[i + 1]))", "wrong", "no-value"),
    ("pairs", _P + "for i in range(len(steps) - 1):\n    if cases[i] == cases[i + 1]:\n        pairs.append([steps[i], steps[i + 1]])\npairs", "wrong", None),
    ("pairs", _P + "for i in range(len(steps) - 1):\n    ...\npairs", "wrong", "dots-left"),
    ("pairs", _PAIRS.replace(_READ, f'log = pd.read_csv("{LOG_2}")'), "wrong", "changed-line"),
    ("took", _TOOK_START, "empty", None),
    ("took", _TOOK, "correct", None),
    ("took", _T + 'took = log.groupby("case")["day"].agg("max")\ntook', "correct", None),
    ("took", _T + 'took = log.groupby("case").max()["day"]\ntook', "correct", None),
    ("took", _T + 'took = log.groupby("case")["day"].mean()\ntook', "wrong", "mean"),
    ("took", _T + 'took = log.groupby("case")["day"].count()\ntook', "wrong", "count"),
    ("took", _T + 'took = log.groupby("case")["day"].sum()\ntook', "wrong", "sum"),
    ("took", _T + 'took = log["day"].max()\ntook', "wrong", "one-number"),
    ("took", _T + 'took = log.groupby("activity")["day"].max()\ntook', "wrong", "by-activity"),
    ("took", _T + 'took = log.groupby("case")["day"]\ntook', "wrong", "groups-only"),
    ("took", _T + 'took = log.groupby("case").max()\ntook', "wrong", "table"),
    ("took", _T + 'took = log.groupby("Case")["day"].max()\ntook', "error", "no-column"),
    ("took", _T + 'took = log.groupby("case")["day"].min()\ntook', "wrong", None),
    ("took", _T + 'took = log.groupby("case")["day"].max()\ntook.mean()', "wrong", "one-number"),
]
