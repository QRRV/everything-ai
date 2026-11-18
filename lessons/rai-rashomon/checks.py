# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-rashomon: fifty decision trees on three columns of UCI Adult, five of each depth from 1 to
10, each grown on its own bootstrap sample of one half of the table and tested on the other half. Most of them are
about equally accurate, yet they give many people different answers. The learner measures that with ambiguity (Marx
and colleagues), picks out the trees no more than a point below the best (the Rashomon set), says which tree Rudin would
choose from it, and what a number found by retraining does and does not prove; with the trees and their set for each
size of the margin, what the learner's ambiguity() finds on the trees of the worked example, and the learner's set."""
import functools
import html
import math
import re

from everything_ai import chart, yours
from everything_ai.check import CodeBox, FunctionBox, Mistake

COLUMNS = ["age", "education_num", "hours_per_week"]
DEPTHS = list(range(1, 11))     # the depths of the trees
EACH = 5                        # trees of each depth, each on its own bootstrap sample
EPS = 0.01                      # one percentage point: the margin of the worked example
STOPS = [0, 0.25, 0.5, 1, 2, 5]   # the stops of the slider: the margin in percentage points
PERSON = (36, 13, 40)           # aged 36, a bachelor's degree, 40 hours a week: the person of *Why this matters*

# How often each of the 50 trees is right on the test half, rounded to four decimals, in the order they are grown
# (tests/test_rai_rashomon.py grows them again). Exercise 2 starts from these.
RIGHT = [0.7585, 0.7585, 0.7585, 0.7585, 0.7585, 0.7889, 0.7887, 0.7889, 0.7844, 0.7844,
         0.791, 0.7921, 0.7911, 0.7913, 0.7911, 0.7949, 0.7905, 0.7913, 0.7951, 0.7901,
         0.7949, 0.7982, 0.7961, 0.7955, 0.7985, 0.7955, 0.7912, 0.7934, 0.7937, 0.7931,
         0.7957, 0.7946, 0.7924, 0.7949, 0.7939, 0.7901, 0.7949, 0.7941, 0.7933, 0.7949,
         0.7847, 0.7899, 0.793, 0.7878, 0.7864, 0.786, 0.7817, 0.7845, 0.7876, 0.7881]


@functools.lru_cache(maxsize=1)
def table():
    """The three columns of UCI Adult as an array, and who earns more than 50,000 dollars. Read-only."""
    import pandas as pd
    from everything_ai import paths
    people = pd.read_csv(paths.ROOT / "data" / "adult.csv.gz")
    X = people[COLUMNS].to_numpy()
    y = (people["income"] == ">50K").to_numpy()
    for a in (X, y):
        a.flags.writeable = False
    return X, y


@functools.lru_cache(maxsize=1)
def pool() -> dict:
    """The trees of the worked example: the table shuffled with seed 0 and cut in two halves; then, with seed 1, for
    each depth from 1 to 10 five trees, each on a bootstrap sample of the training half. What each says for every
    person of the test half, how often it is right there, its depth, the trees themselves, the positions of the test
    half, and the best tree. Under a second."""
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    X, y = table()
    order = np.random.default_rng(0).permutation(len(y))
    train, test = order[: len(y) // 2], order[len(y) // 2:]
    rng = np.random.default_rng(1)
    answers, right, depths, trees = [], [], [], []
    for depth in DEPTHS:
        for k in range(EACH):
            rows = train[rng.integers(0, len(train), size=len(train))]
            tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X[rows], y[rows])
            answers.append(tree.predict(X[test]))
            right.append(float(np.mean(answers[-1] == y[test])))
            depths.append(depth)
            trees.append(tree)
    answers = np.array(answers)
    answers.flags.writeable = False
    return {"answers": answers, "right": right, "depths": depths, "trees": trees, "test": test,
            "best": int(np.argmax(right))}


def level_set(eps: float) -> list:
    """The positions of the trees no more than eps below the best, on the test half: the eps-level set."""
    p = pool()
    best = p["right"][p["best"]]
    return [k for k, r in enumerate(p["right"]) if r >= best - eps]


@functools.lru_cache(maxsize=None)
def measures(eps: float) -> tuple:
    """Over the trees no more than eps below the best, against the best tree, on the test half: the share of people
    at least one of them gives another answer (ambiguity), and the largest share one of them changes (discrepancy)."""
    p = pool()
    differ = p["answers"][level_set(eps)] != p["answers"][p["best"]]
    return float(differ.any(axis=0).mean()), float(differ.mean(axis=1).max())


def _show(x: float, lang: str, digits: int = 3) -> str:
    text = f"{x:.{digits}f}".rstrip("0").rstrip(".")
    text = "0" if text in ("-0", "") else text
    return text.replace(".", ",") if lang == "nl" else text


def _pct(share: float, lang: str) -> str:
    """A share as a percentage with one decimal, also when that decimal is 0: 80.0."""
    text = f"{share * 100:.1f}"
    return text.replace(".", ",") if lang == "nl" else text


def _count(n: int, lang: str) -> str:
    return f"{n:,}".replace(",", "." if lang == "nl" else ",")


def _points(e: float, lang: str) -> str:
    """A margin in percentage points, in words: 1 point, 0.25 points."""
    shown = _show(e, lang, 2)
    if lang == "nl":
        return f"{shown} procentpunt"
    return f"{shown} point" if e == 1 else f"{shown} points"


_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.2rem"


def _trees_panel(right, depths, chosen, lang: str, cut: float | None = None) -> chart.Panel:
    """Each tree as a dot over its depth, at how often it is right on the test half: filled when it is in `chosen`,
    open when not; with `cut`, a dashed line at that share."""
    words = _WORDS[lang]
    p = chart.Panel(x=(0.4, 10.6), y=(75.4, 80.4), title=words["trees"], xlabel=words["depth"], ylabel=words["right"],
                    xticks=DEPTHS, yticks=[76, 77, 78, 79, 80])
    if cut is not None:
        p.line([0.4, 10.6], [cut * 100, cut * 100], role="hyper", width=1.4, dashed=True)
    seen = {}
    ins, outs = ([], []), ([], [])
    for k, (r, d) in enumerate(zip(right, depths)):
        place = seen.get(d, 0)
        seen[d] = place + 1
        x = d - 0.28 + 0.14 * min(place, 4)
        target = ins if k in chosen else outs
        target[0].append(x)
        target[1].append(r * 100)
    if outs[0]:
        p.points(outs[0], outs[1], role="ink-3", r=3.6, hollow=True)
    if ins[0]:
        p.points(ins[0], ins[1], role="pred", r=3.8)
    return p


_WORDS = {
    "en": {"trees": "50 trees", "depth": "depth", "right": "% right on the test half",
           "measures": "against the best tree", "ambiguity": "ambiguity", "discrepancy": "discrepancy",
           "share": "% of the test people",
           "alt": "Left, fifty dots, five for each depth from 1 to 10: how often each tree is right on the test half, "
                  "from {lo}% to {hi}%. A dashed line lies {eps} below the best tree, {best}%; the {n} trees on or above "
                  "it are filled, the others open. Right, two bars: ambiguity {amb}% and discrepancy {disc}%.",
           "said_one": "Within {eps} of the best tree ({best}%) lies only the best tree itself: no one gets another "
                       "answer.",
           "span": "depth {a} to {b}", "span_one": "all of depth {a}",
           "said": "Within {eps} of the best tree ({best}%): {n} of the 50 trees, {span}. For "
                   "<span class=\"c-pred ea-num\">{amb}%</span> of the 24,421 test people, at least one of them gives "
                   "another answer than the best tree; switching to one of them changes the answer for at most "
                   "{disc}%."},
    "nl": {"trees": "50 bomen", "depth": "diepte", "right": "% goed op de testhelft",
           "measures": "tegen de beste boom", "ambiguity": "ambiguity", "discrepancy": "discrepancy",
           "share": "% van de testpersonen",
           "alt": "Links vijftig stippen, vijf voor elke diepte van 1 tot 10: hoe vaak elke boom goed zit op de "
                  "testhelft, van {lo}% tot {hi}%. Een gestippelde lijn ligt {eps} onder de beste boom, {best}%; de {n} "
                  "bomen op of boven die lijn zijn gevuld, de andere open. Rechts twee staven: ambiguity {amb}% en "
                  "discrepancy {disc}%.",
           "said_one": "Binnen {eps} van de beste boom ({best}%) ligt alleen de beste boom zelf: niemand krijgt een "
                       "ander antwoord.",
           "span": "diepte {a} tot {b}", "span_one": "allemaal met diepte {a}",
           "said": "Binnen {eps} van de beste boom ({best}%): {n} van de 50 bomen, {span}. Bij "
                   "<span class=\"c-pred ea-num\">{amb}%</span> van de 24.421 testpersonen geeft minstens één ervan een "
                   "ander antwoord dan de beste boom; overstappen op één ervan verandert het antwoord voor hoogstens "
                   "{disc}%."},
}


def eps_picture(e: float, lang: str) -> tuple[str, str]:
    """For the slider of *The idea*: the trees, with the ones no more than e points below the best filled, and the
    ambiguity and discrepancy of that set; and the sentence under them."""
    w = _WORDS[lang]
    p = pool()
    eps = float(e) / 100
    chosen = level_set(eps)
    best = p["right"][p["best"]]
    amb, disc = measures(eps)
    left = _trees_panel(p["right"], p["depths"], set(chosen), lang, cut=best - eps)
    right = chart.Panel(x=(0, 2), y=(0, 36), title=w["measures"], xlabel="", ylabel=w["share"],
                        xticks=[0.5, 1.5], xtick_labels=[w["ambiguity"], w["discrepancy"]], yticks=[0, 10, 20, 30])
    right.rect(0.2, 0, 0.8, amb * 100, role="pred", solid=True)
    right.rect(1.2, 0, 1.8, disc * 100, role="pred")
    depths = [p["depths"][k] for k in chosen]
    fill = {"eps": _points(e, lang), "best": _pct(best, lang), "n": len(chosen), "a": min(depths), "b": max(depths),
            "amb": _pct(amb, lang), "disc": _pct(disc, lang), "lo": _pct(min(p["right"]), lang),
            "hi": _pct(max(p["right"]), lang)}
    fill["span"] = w["span_one" if fill["a"] == fill["b"] else "span"].format(**fill)
    svg = chart.figure([left, right], alt=w["alt"].format(**fill), width=330, height=250)
    svg = chart.with_decimal_comma(svg) if lang == "nl" else svg
    said = (w["said_one"] if len(chosen) == 1 else w["said"]).format(**fill)
    return svg, said


def says(person) -> tuple[bool, int, int]:
    """For everyone in the test half with these three values: what the best tree says, how many of the trees no more
    than a point below the best say yes, and how many such people the test half holds."""
    import numpy as np
    X, y = table()
    p = pool()
    where = np.flatnonzero((X[p["test"]] == np.asarray(person)).all(axis=1))
    good = level_set(EPS)
    return bool(p["answers"][p["best"], where[0]]), int(p["answers"][good, where[0]].sum()), len(where)


# --- exercise 1: ambiguity -------------------------------------------------------------------------------------

_LISTS = {}


def _lists() -> dict:
    """What the best tree says for each test person, and what the trees of three sets say, as Python lists."""
    if not _LISTS:
        p = pool()
        _LISTS["base"] = p["answers"][p["best"]].tolist()
        for name, eps in (("half", 0.005), ("one", EPS), ("all", 1.0)):
            _LISTS[name] = [p["answers"][k].tolist() for k in level_set(eps)]
    return _LISTS


def _mine(seen):
    """The learner's function, or None when the code stopped or the name is not a function."""
    mine = seen.mine
    if mine is None:
        return None
    fn = mine.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _number(v):
    """A float from what a function gave back, or None when it is not one number."""
    if isinstance(v, bool):
        return None
    if hasattr(v, "item") and getattr(v, "size", 1) == 1:
        v = v.item()
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    return None


_AMB_WORDS = {
    "en": {"title": "Your ambiguity(), on the trees of the worked example",
           "labels": ["16 trees", "35 trees", "all 50"],
           "alt": "Three bars: what your ambiguity() gives against the best tree, on the 24,421 people of the test half. "
                  "With the 16 trees within half a point of the best, {a}%; with the 35 within one point, {b}%; with "
                  "all 50 trees, {c}%.",
           "key": "Against the best tree, on the 24,421 people of the test half. Within half a point of the best "
                  "(16 trees): {a}%. Within one point (35 trees): {b}%. All 50 trees, the five of depth 1 too: {c}%."},
    "nl": {"title": "Jouw ambiguity(), op de bomen van het uitgewerkte voorbeeld",
           "labels": ["16 bomen", "35 bomen", "alle 50"],
           "alt": "Drie staven: wat je ambiguity() geeft tegen de beste boom, bij de 24.421 mensen van de testhelft. "
                  "Met de 16 bomen binnen een half procentpunt van de beste {a}%; met de 35 binnen één procentpunt "
                  "{b}%; met alle 50 bomen {c}%.",
           "key": "Tegen de beste boom, bij de 24.421 mensen van de testhelft. Binnen een half procentpunt van de beste "
                  "(16 bomen): {a}%. Binnen één procentpunt (35 bomen): {b}%. Alle 50 bomen, ook de vijf met diepte 1: "
                  "{c}%."},
}


def _amb_picture(seen) -> str:
    fn = _mine(seen)
    if fn is None:
        return ""
    lists = _lists()

    def run():
        out = []
        for name in ("half", "one", "all"):
            got = _number(fn(lists["base"], lists[name]))
            if got is None or not 0 <= got <= 1:
                return None
            out.append(got)
        return out

    from everything_ai import check as checking
    try:
        got = checking.with_time_limit(run)
    except Exception:
        return ""
    if got is None:
        return ""
    w = _AMB_WORDS[seen.lang]
    p = chart.Panel(x=(0, 3), y=(0, max(36, max(got) * 105)), title="", xlabel="", ylabel=_WORDS[seen.lang]["share"],
                    xticks=[0.5, 1.5, 2.5], xtick_labels=w["labels"])
    for i, share in enumerate(got):
        p.rect(i + 0.2, 0, i + 0.8, share * 100, role="pred", solid=True)
        p.text(i + 0.5, share * 100 + 1.6, _pct(share, seen.lang), "ink")
    fill = {"a": _pct(got[0], seen.lang), "b": _pct(got[1], seen.lang), "c": _pct(got[2], seen.lang)}
    svg = chart.figure([p], alt=w["alt"].format(**fill), width=440, height=240)
    svg = chart.with_decimal_comma(svg) if seen.lang == "nl" else svg
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(w["key"].format(**fill))}</div>')


# --- exercise 2: the trees no more than a point below the best ---------------------------------------------------

_GOOD_WORDS = {
    "en": {"title": "Your good: the trees no more than eps below the best",
           "alt": "Fifty dots, five for each depth from 1 to 10: how often each tree is right on the test half. The {n} "
                  "trees in your good are filled, the others open.",
           "key": "Your good holds {n} of the 50 trees, of depth {a} to {b}. The best, of depth {bd}, is right for "
                  "{best}% of the test half. The smallest tree in your set has depth {a}: at most {leaves} leaves."},
    "nl": {"title": "Jouw good: de bomen die hoogstens eps onder de beste liggen",
           "alt": "Vijftig stippen, vijf voor elke diepte van 1 tot 10: hoe vaak elke boom goed zit op de testhelft. De "
                  "{n} bomen in je good zijn gevuld, de andere open.",
           "key": "Je good bevat {n} van de 50 bomen, met diepte {a} tot {b}. De beste, met diepte {bd}, zit bij "
                  "{best}% van de testhelft goed. De kleinste boom in je set heeft diepte {a}: hoogstens {leaves} "
                  "bladeren."},
}


def _good_picture(seen) -> str:
    mine = seen.mine
    if mine is None:
        return ""
    right, depths, good = mine.names.get("right"), mine.names.get("depths"), mine.value
    if not (isinstance(right, list) and isinstance(depths, list) and len(right) == len(depths) == len(RIGHT)):
        return ""
    if not isinstance(good, list) or not good:
        return ""
    chosen = set()
    for k in good:
        if isinstance(k, bool) or not hasattr(k, "__index__") or not 0 <= int(k) < len(right):
            return ""
        chosen.add(int(k))
    if any(_number(r) is None or not 0 <= float(r) <= 1 for r in right):
        return ""
    if any(isinstance(d, bool) or not isinstance(d, int) for d in depths):
        return ""
    w = _GOOD_WORDS[seen.lang]
    top = max(range(len(right)), key=lambda k: float(right[k]))
    ds = [depths[k] for k in chosen]
    fill = {"n": len(chosen), "a": min(ds), "b": max(ds), "bd": depths[top], "best": _pct(float(right[top]), seen.lang),
            "leaves": 2 ** min(ds)}
    panel = _trees_panel([float(r) for r in right], depths, chosen, seen.lang)
    svg = chart.figure([panel], alt=w["alt"].format(**fill), width=440, height=250)
    svg = chart.with_decimal_comma(svg) if seen.lang == "nl" else svg
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(w["key"].format(**fill))}</div>')


# The lesson is done when these are correct. "pick" and "sure" are choice questions.
REQUIRED = ["ambiguity", "good", "pick", "sure"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
            "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."}

# --- exercise 1: ambiguity
_A_HEAD = ("def ambiguity(base, others):\n"
           "    # base: what the chosen model says for each person, such as [1, 0, 0, 1, 0]\n"
           "    # others: what each of the other good models says, one list per model\n")
_A_START = _A_HEAD + "    ...\n"


def _a(test: str = "            if answers[i] != base[i]:\n                differs = True\n",
       reset: str = "        differs = False\n", outside: str = "", count: str = "        if differs:\n"
       "            conflicts = conflicts + 1\n", ret: str = "    return conflicts / len(base)\n") -> str:
    """The body of ambiguity with one part changed: the test per model, where `differs` starts (`outside` puts it
    before the loop over the people), how a person is counted, and what is returned."""
    return (_A_HEAD + "    conflicts = 0\n" + outside + "    for i in range(len(base)):\n" + reset
            + "        for answers in others:\n" + test + count + ret)


_A_SOLUTION = _a()

# Small made-up answers: the first is the example in the lesson's text.
_A_VISIBLE = [([1, 0, 0, 1, 0], [[1, 1, 0, 1, 0], [0, 1, 0, 1, 0]])]
_A_HIDDEN = [([1, 1, 0], [[1, 1, 0]]),
             ([0, 0, 0, 0, 0], [[1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [1, 0, 0, 0, 1]]),
             ([True, False], [[False, True], [False, False]]),
             ([1, 0, 1, 0], [[1, 0, 1, 0], [0, 1, 0, 1]]),
             ([0, 1, 1, 0, 1, 0], [[0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 1], [1, 1, 1, 0, 1, 0]]),
             ([1, 0, 1, 1], [[1, 0, 1, 1], [1, 0, 1, 1]]),
             ([0, 1, 0, 1, 1, 0, 0, 1], [[0, 1, 0, 1, 1, 0, 0, 0]]),
             ([1, 0], [[0, 0], [1, 1], [1, 0]])]

_A_PAIRS = (_A_HEAD + "    conflicts = 0\n    for i in range(len(base)):\n        for answers in others:\n"
            "            if answers[i] != base[i]:\n                conflicts = conflicts + 1\n"
            "    return conflicts / len(base)\n")
_A_ALL = _a(test="            if answers[i] == base[i]:\n                differs = False\n",
            reset="        differs = True\n")
_A_LAST = _a(test="            differs = answers[i] != base[i]\n")
_A_NO_RESET = _a(reset="", outside="    differs = False\n")
_A_FIRST = _a(ret="        return conflicts / len(base)\n")

# --- exercise 2: the trees no more than eps below the best


def _literal(name: str, values: list, width: int = 110) -> str:
    """`name = [v, v, ...]` over as many lines as it needs, none longer than `width`."""
    lines, line = [], f"{name} = ["
    indent = " " * len(line)
    for i, v in enumerate(values):
        piece = f"{v}" + ("]" if i == len(values) - 1 else ",")
        if len(line) + 1 + len(piece) > width and line.strip() not in (f"{name} = [", ""):
            lines.append(line.rstrip())
            line = indent + piece
        else:
            line = line + piece if line.endswith("[") else line + " " + piece
    lines.append(line)
    return "\n".join(lines) + "\n"


_EPS_LINE = "eps = 0.01"
_G_HEAD = ("# How often each of the 50 trees of the worked example is right on the test half, from 0 to 1, and its depth\n"
           + _literal("right", RIGHT)
           + _literal("depths", [d for d in DEPTHS for k in range(EACH)])
           + f"{_EPS_LINE}\n")
_G_START = _G_HEAD + "best = ...\ngood = []\nfor k in range(len(right)):\n    ...\ngood\n"


def _g(best: str = "max(right)", test: str = "right[k] >= best - eps", add: str = "k") -> str:
    return (_G_HEAD + f"best = {best}\ngood = []\nfor k in range(len(right)):\n    if {test}:\n"
            f"        good.append({add})\ngood\n")


_G_SOLUTION = _g()


def _every(got, text) -> bool:
    return isinstance(got, list) and len(got) == len(RIGHT) and sorted(got) == list(range(len(RIGHT)))


EXERCISES = {
    "ambiguity": FunctionBox(
        name="ambiguity",
        picture=_amb_picture,
        function="ambiguity",
        start=_A_START,
        solution=_A_SOLUTION,
        visible=_A_VISIBLE,
        hidden=_A_HIDDEN,
        first=[_DOTS],
        mistakes=[
            Mistake("count", wrong=_a(ret="    return conflicts\n"),
                    en="`{call}` gives {got}, and it should give {want}. Your function gives how many people get another answer. Ambiguity is a share: divide by the number of people, `len(base)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie geeft hoeveel mensen een ander antwoord krijgen. Ambiguity is een deel: deel door het aantal mensen, `len(base)`."),
            Mistake("pairs", wrong=_A_PAIRS,
                    en="`{call}` gives {got}, and it should give {want}. Your function counts every model that gives another answer, so a person with two such models counts twice, and the share can pass 1. Count each person once: keep in a variable whether some model differs for this person, and count the person after the loop over the models.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie telt elk model dat een ander antwoord geeft, dus een persoon met twee van zulke modellen telt twee keer, en het deel kan boven 1 komen. Tel elke persoon één keer: houd in een variabele bij of een model voor deze persoon verschilt, en tel de persoon na de loop over de modellen."),
            Mistake("all", wrong=_A_ALL,
                    en="`{call}` gives {got}, and it should give {want}. Your function counts a person only when every other model gives another answer. One is enough: a person counts as soon as at least one model differs.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie telt een persoon alleen als elk ander model een ander antwoord geeft. Eén is genoeg: een persoon telt zodra minstens één model verschilt."),
            Mistake("agree", wrong=_a(ret="    return 1 - conflicts / len(base)\n"),
                    en="`{call}` gives {got}, and it should give {want}: the share of people for whom all models agree. Ambiguity is the other part: the share for whom at least one model gives another answer.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het deel van de mensen bij wie alle modellen het eens zijn. Ambiguity is het andere deel: het deel bij wie minstens één model een ander antwoord geeft."),
            Mistake("models", wrong=_a(ret="    return conflicts / len(others)\n"),
                    en="`{call}` gives {got}, and it should give {want}. You divide by the number of models, `len(others)`. Ambiguity is a share of the people: divide by `len(base)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je deelt door het aantal modellen, `len(others)`. Ambiguity is een deel van de mensen: deel door `len(base)`."),
            Mistake("last-only", wrong=_A_LAST,
                    en="`{call}` gives {got}, and it should give {want}: only the last model counts. Each model sets your variable again, also when it agrees, so a model that differed is forgotten. Set it to `True` only when a model differs, and to `False` once per person, before the loop over the models.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: alleen het laatste model telt. Elk model zet je variabele opnieuw, ook als het het eens is, dus een model dat verschilde wordt vergeten. Zet hem alleen op `True` als een model verschilt, en één keer per persoon op `False`, vóór de loop over de modellen."),
            Mistake("no-reset", wrong=_A_NO_RESET,
                    en="`{call}` gives {got}, and it should give {want}. Your variable is set to `False` only once, before the loop over the people, so after the first person who gets another answer, everyone counts. Set it to `False` again for each person, at the start of the loop over the people.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je variabele wordt maar één keer op `False` gezet, vóór de loop over de mensen, dus na de eerste persoon met een ander antwoord telt iedereen. Zet hem voor elke persoon opnieuw op `False`, aan het begin van de loop over de mensen."),
            Mistake("first-person", wrong=_A_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: what you get from the first person alone. If your `return` stands inside the `for` loop over the people, Python leaves the function after the first person.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: wat je krijgt van alleen de eerste persoon. Als je `return` binnen de `for`-loop over de mensen staat, verlaat Python de functie na de eerste persoon."),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...` with the body of `ambiguity`: go over the people, and count each person for whom at least one list in `others` says something other than `base`. Return the share.",
                      "nl": "Vervang de `...` door de inhoud van `ambiguity`: ga de mensen langs, en tel elke persoon bij wie minstens één lijst in `others` iets anders zegt dan `base`. Geef het deel terug."},
            "cleared": _CLEARED,
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Work it out by hand: go over the people one by one; a person counts once if at least one model in `others` says something other than `base`; then divide by the number of people. Which step is different in your function?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Reken het met de hand uit: ga de mensen een voor een langs; een persoon telt één keer als minstens één model in `others` iets anders zegt dan `base`; deel dan door het aantal mensen. Welke stap is anders in je functie?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for any number of people and of models, and does it use `base` and `others` as they are given?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elk aantal mensen en modellen, en gebruikt hij `base` en `others` zoals ze worden meegegeven?"},
            "correct": {"en": "Your `ambiguity` is right in all {n} cases. The picture runs it on the trees of the worked example: the more trees you compare with the best one, the more people turn out to get another answer from at least one of them. A number found this way is a lower bound, not the whole story.",
                        "nl": "Je `ambiguity` klopt in alle {n} gevallen. De tekening voert hem uit op de bomen van het uitgewerkte voorbeeld: hoe meer bomen je met de beste vergelijkt, hoe meer mensen een ander antwoord blijken te krijgen van minstens één ervan. Een getal dat je zo vindt, is een ondergrens, niet het hele verhaal."},
        },
    ),
    "good": CodeBox(
        name="good",
        picture=_good_picture,
        start=_G_START,
        solution=_G_SOLUTION,
        keep=[_EPS_LINE],
        # The check also runs the box with another margin: a 0.01 typed in by hand does not change with it.
        others=[{"eps": 0.005}],
        first=[_DOTS],
        mistakes=[
            Mistake("every", matches=_every,
                    en="Your `good` holds all 50 trees, also the five of depth 1 that say no to everyone and are right for 75.9 percent. Keep only the trees no more than `eps` below the best: is `best` the highest share in `right`, and is your test `right[k] >= best - eps`? `right` holds shares from 0 to 1, so one point is 0.01.",
                    nl="Je `good` bevat alle 50 bomen, ook de vijf met diepte 1 die tegen iedereen nee zeggen en bij 75,9 procent goed zitten. Houd alleen de bomen die hoogstens `eps` onder de beste liggen: is `best` het hoogste deel in `right`, en is je test `right[k] >= best - eps`? `right` bevat delen van 0 tot 1, dus één procentpunt is 0,01."),
            Mistake("outside", wrong=_g(test="right[k] < best - eps"),
                    en="Your `good` holds the trees that lie more than `eps` below the best: the other side. Keep the trees whose share is at least `best - eps`.",
                    nl="Je `good` bevat de bomen die meer dan `eps` onder de beste liggen: de andere kant. Houd de bomen waarvan het deel minstens `best - eps` is."),
            Mistake("depths", wrong=_g(add="depths[k]"),
                    en="Your `good` holds the depths of the trees, not their positions. Add the position: `good.append(k)`. Their depths stay in `depths`.",
                    nl="Je `good` bevat de diepte van de bomen, niet hun plaats. Voeg de plaats toe: `good.append(k)`. Hun diepte blijft in `depths`."),
            Mistake("empty", matches=lambda got, text: isinstance(got, list) and got == [],
                    en="Your `good` is empty: no tree passes your test, not even the best one. Is `best` the highest share itself, `max(right)`, and not its position? And do you keep the trees whose share is at least `best - eps`?",
                    nl="Je `good` is leeg: geen boom haalt je test, zelfs de beste niet. Is `best` het hoogste deel zelf, `max(right)`, en niet de plaats ervan? En houd je de bomen waarvan het deel minstens `best - eps` is?"),
            Mistake("one-tree", matches=lambda got, text: isinstance(got, list) and len(got) == 1,
                    en="Your `good` holds only one tree, while 35 trees lie no more than a point below the best. Does your test use `best - eps`, and does `good.append(k)` add every tree that passes, inside the loop?",
                    nl="Je `good` bevat maar één boom, terwijl 35 bomen hoogstens een procentpunt onder de beste liggen. Gebruikt je test `best - eps`, en voegt `good.append(k)` elke boom toe die slaagt, binnen de loop?"),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the two `...`: the highest share in `right` as `best`, and inside the loop, `k` added to `good` when tree `k` lies no more than `eps` below the best.",
                      "nl": "Vervang de twee `...`: het hoogste deel in `right` als `best`, en binnen de loop `k` toegevoegd aan `good` als boom `k` hoogstens `eps` onder de beste ligt."},
            "cleared": _CLEARED,
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries another margin by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf een andere marge."},
            "no_value": {"en": "The last line of the box shows nothing. Put `good` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `good` op de laatste regel, zodat de check hem kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. `good` should hold the positions `k` of the trees whose `right[k]` is at least `best - eps`, with `best` the highest share in `right`. Which step is different in your code?",
                      "nl": "Je laatste regel geeft `{got}`. `good` moet de plaatsen `k` bevatten van de bomen waarvan `right[k]` minstens `best - eps` is, met `best` het hoogste deel in `right`. Welke stap is anders in je code?"},
            "other": {"en": "Right here. But with {where}, your `good` is different from what it should be. Does your test use `eps`, instead of 0.01?",
                      "nl": "Hier goed. Maar met {where} is je `good` anders dan hij moet zijn. Gebruikt je test `eps`, in plaats van 0,01?"},
            "other_error": {"en": "Right for this margin. But with {where}, your code stops with an error. Do you use `eps` in your test, instead of 0.01?",
                            "nl": "Goed voor deze marge. Maar met {where} stopt je code met een fout. Gebruik je `eps` in je test, in plaats van 0,01?"},
            "correct": {"en": "These 35 trees are the Rashomon set of the worked example: no more than a point below the best, from depth 2 to depth 9. The picture shows them filled. The check also tried half a point, which keeps 16. Look at the smallest depth in your set: the next question is about it.",
                        "nl": "Deze 35 bomen zijn de Rashomon set van het uitgewerkte voorbeeld: hoogstens een procentpunt onder de beste, van diepte 2 tot diepte 9. De tekening laat ze gevuld zien. De check probeerde ook een half procentpunt, wat er 16 overlaat. Kijk naar de kleinste diepte in je set: daar gaat de volgende vraag over."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("ambiguity", _A_START, "empty", None),
    ("ambiguity", _A_SOLUTION, "correct", None),
    ("ambiguity", _A_HEAD + "    conflicts = 0\n    for i in range(len(base)):\n        votes = 0\n"
                  "        for answers in others:\n            votes += answers[i] != base[i]\n"
                  "        if votes > 0:\n            conflicts += 1\n    return conflicts / len(base)\n", "correct", None),
    ("ambiguity", _a(ret="    return conflicts\n"), "wrong", "count"),
    ("ambiguity", _A_PAIRS, "wrong", "pairs"),
    ("ambiguity", _A_ALL, "wrong", "all"),
    ("ambiguity", _a(ret="    return 1 - conflicts / len(base)\n"), "wrong", "agree"),
    ("ambiguity", _a(ret="    return conflicts / len(others)\n"), "wrong", "models"),
    ("ambiguity", _A_LAST, "wrong", "last-only"),
    ("ambiguity", _A_NO_RESET, "wrong", "no-reset"),
    ("ambiguity", _A_FIRST, "wrong", "first-person"),
    ("ambiguity", _A_HEAD + "    return 0.4\n", "almost", None),
    ("ambiguity", _a(test="            if answers[i] == base[i]:\n                differs = True\n"), "wrong", None),
    ("ambiguity", _a(ret=""), "wrong", "no-return"),
    ("ambiguity", _A_HEAD + "    ...\n    return 0.4\n", "wrong", "dots-left"),
    ("good", _G_START, "empty", None),
    ("good", _G_SOLUTION, "correct", None),
    ("good", _G_HEAD + "best = 0\nfor r in right:\n    if r > best:\n        best = r\ngood = []\n"
             "for k in range(len(right)):\n    if best - right[k] <= eps:\n        good.append(k)\ngood\n", "correct", None),
    ("good", _g(test="right[k] >= best - 1"), "wrong", "every"),
    ("good", _g(best="right[0]"), "wrong", "every"),
    ("good", _g(test="right[k] < best - eps"), "wrong", "outside"),
    ("good", _g(add="depths[k]"), "wrong", "depths"),
    ("good", _g(test="right[k] >= best + eps"), "wrong", "empty"),
    ("good", _g(best="right.index(max(right))"), "wrong", "empty"),
    ("good", _g(test="right[k] >= best"), "wrong", "one-tree"),
    ("good", _g(test="right[k] >= best - 0.01"), "almost", "other-value"),
    ("good", _g(test="right[k] >= best - eps / 2"), "wrong", None),
    ("good", _g().replace(_EPS_LINE, "eps = 0.02"), "wrong", "changed-line"),
    ("good", _G_SOLUTION.replace("        good.append(k)\n", "        ...\n"), "wrong", "dots-left"),
]
