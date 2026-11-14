# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-automl: 27 candidates drawn at random, half k-NN with scaled numbers (k and p) and half a
random forest of 50 trees (its depth and the number of features per split), each scored by the mean ROC AUC of
cross-validation with KFold(5) on the first 778, 2,333 or all 7,000 training runs of AI4I 2020 (split as in
ml-imbalanced). Successive halving gives all 27 candidates 778 runs, the best 9 of them 2,333 and the best 3 all 7,000.
The scores in capitals were measured once; tests/test_ml_automl.py measures them again."""
import contextlib
import io
import warnings

from everything_ai import chart, check as checking, yours
from everything_ai.check import FunctionBox, Mistake

DATA = "data/ai4i-2020.csv.gz"
NUMBERS = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
ROUNDS = [778, 2333, 7000]     # the training runs every candidate gets in round 1, 2 and 3: 7,000 / 9, 7,000 / 3, 7,000
FULL = 27 * 7000               # the budget of the full search: all 27 candidates on all 7,000 runs

# The 27 candidates of draw(0, 27), and the mean AUC of the 5 folds of each, rounded to 4 decimals, on the first 778,
# 2,333 and 7,000 training runs; TEST_AUC: on the 3,000 test runs, fitted on all 7,000 training runs.
CANDIDATES = [
    ('forest', 4, 2), ('neighbors', 103, 3.74), ('forest', 14, 1), ('forest', 25, 2), ('neighbors', 133, 1.1),
    ('forest', 3, 5), ('forest', 8, 1), ('neighbors', 1, 1.37), ('forest', 11, 2), ('forest', 5, 3),
    ('forest', 12, 5), ('forest', 12, 4), ('neighbors', 61, 2.58), ('neighbors', 15, 3.67), ('forest', 5, 2),
    ('forest', 4, 4), ('neighbors', 9, 3.67), ('neighbors', 34, 1.25), ('forest', 16, 3), ('neighbors', 148, 1.18),
    ('neighbors', 2, 2.35), ('forest', 3, 4), ('neighbors', 3, 1.27), ('forest', 4, 1), ('forest', 3, 4),
    ('neighbors', 1, 2.89), ('forest', 6, 5),
]
SCORES = {
    778: [
        0.9643, 0.9242, 0.9573, 0.9652, 0.9137, 0.9181, 0.9576, 0.6597, 0.9651, 0.9483, 0.8994, 0.8977, 0.9317,
        0.8949, 0.965, 0.9419, 0.8536, 0.9311, 0.9316, 0.9089, 0.775, 0.9401, 0.7895, 0.9492, 0.9401, 0.6911, 0.9171,
    ],
    2333: [
        0.9519, 0.8986, 0.9432, 0.9382, 0.9051, 0.9287, 0.9486, 0.6784, 0.9507, 0.9528, 0.9413, 0.9375, 0.9094,
        0.8958, 0.9549, 0.9422, 0.8653, 0.9117, 0.9246, 0.909, 0.7396, 0.9392, 0.7694, 0.9369, 0.9392, 0.6684, 0.96,
    ],
    7000: [
        0.9531, 0.9452, 0.954, 0.9479, 0.9452, 0.9389, 0.9572, 0.7117, 0.9643, 0.9611, 0.9607, 0.9687, 0.9495,
        0.9101, 0.9576, 0.9519, 0.8821, 0.9307, 0.9578, 0.9448, 0.7906, 0.9446, 0.8297, 0.9472, 0.9446, 0.7179,
        0.9639,
    ],
}
TEST_AUC = [
    0.9683, 0.9575, 0.9681, 0.9691, 0.9604, 0.9448, 0.9743, 0.7672, 0.9758, 0.9731, 0.9731, 0.9744, 0.9629, 0.9245,
    0.9715, 0.9636, 0.9074, 0.9586, 0.974, 0.959, 0.8198, 0.9627, 0.8615, 0.9652, 0.9627, 0.7805, 0.9757,
]
# Successive halving took about 14 s and the full search about 27 s, measured once on one thread of the course's
# computer: the lesson says "about", because a learner's computer differs.


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def draw(seed: int, n: int) -> list:
    """The candidates as the worked example draws them: a coin flip between k-NN and a forest, then its settings."""
    import numpy as np
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        if rng.random() < 0.5:
            out.append(("neighbors", int(10 ** rng.uniform(0, np.log10(300))), round(float(rng.uniform(1, 4)), 2)))
        else:
            out.append(("forest", int(10 ** rng.uniform(np.log10(2), np.log10(30))), int(rng.integers(1, 6))))
    return out


def score(c, runs: int) -> float:
    """The measured score of candidate c on the first `runs` training runs."""
    return SCORES[runs][CANDIDATES.index(tuple(c))]


def halving() -> list:
    """The positions of the candidates in every round of the course's successive halving, and the winner's position."""
    alive = list(range(len(CANDIDATES)))
    rounds = []
    for runs in ROUNDS:
        rounds.append(alive)
        ranked = sorted(alive, key=lambda i: -SCORES[runs][i])
        alive = ranked[:len(alive) // 3]
    return rounds, alive[0]


def full_winner() -> int:
    return max(range(len(CANDIDATES)), key=lambda i: SCORES[7000][i])


def describe(c, lang: str) -> str:
    if c[0] == "forest":
        return (f"een random forest met diepte {c[1]} en {c[2]} features per splitsing" if lang == "nl" else
                f"a random forest with depth {c[1]} and {c[2]} features per split")
    p = f"{c[2]:g}".replace(".", ",") if lang == "nl" else f"{c[2]:g}"
    return f"k-NN met k = {c[1]} en p = {p}" if lang == "nl" else f"k-NN with k = {c[1]} and p = {p}"


def spent(rounds: int) -> int:
    """The budget of the first `rounds` rounds: the training runs handed out, 778 to each of 27 candidates and so on."""
    sizes = halving()[0]
    return sum(len(sizes[i]) * ROUNDS[i] for i in range(rounds))


# --- the slider -----------------------------------------------------------------------------------------------------

def round_said(r: int, lang: str) -> str:
    sizes, winner = halving()
    n, runs = len(sizes[r - 1]), ROUNDS[r - 1]
    best = max(sizes[r - 1], key=lambda i: SCORES[runs][i])
    budget = number(spent(r), lang)
    if lang == "nl":
        head = {1: f"Ronde 1: alle 27 kandidaten op {number(runs, lang)} runs. De beste 9 gaan door, allemaal random forests; de beste scoort {number(SCORES[runs][best], lang, 3)}.",
                2: f"Ronde 2: die 9 op {number(runs, lang)} runs. De beste 3 gaan door; de beste scoort {number(SCORES[runs][best], lang, 3)}.",
                3: f"Ronde 3: die 3 op alle {number(runs, lang)} runs. De winnaar is {describe(CANDIDATES[winner], lang)}, met AUC {number(SCORES[runs][winner], lang, 3)}."}[r]
        tail = f" Budget tot nu toe: {budget} runs, tegen {number(FULL, lang)} voor alle 27 op alle 7.000."
        if r == 3:
            f = full_winner()
            place = sorted(sizes[0], key=lambda i: -SCORES[778][i]).index(f) + 1
            tail += (f" De volledige zoektocht vindt {describe(CANDIDATES[f], lang)}, met AUC {number(SCORES[7000][f], lang, 3)}; "
                     f"op 778 runs scoorde dat {number(SCORES[778][f], lang, 3)}, plek {place} van 27, en viel af in ronde 1.")
        return head + tail
    head = {1: f"Round 1: all 27 candidates on {number(runs, lang)} runs. The best 9 go on, all random forests; the best scores {number(SCORES[runs][best], lang, 3)}.",
            2: f"Round 2: those 9 on {number(runs, lang)} runs. The best 3 go on; the best scores {number(SCORES[runs][best], lang, 3)}.",
            3: f"Round 3: those 3 on all {number(runs, lang)} runs. The winner is {describe(CANDIDATES[winner], lang)}, with AUC {number(SCORES[runs][winner], lang, 3)}."}[r]
    tail = f" Budget so far: {budget} runs, against {number(FULL, lang)} for all 27 on all 7,000."
    if r == 3:
        f = full_winner()
        place = sorted(sizes[0], key=lambda i: -SCORES[778][i]).index(f) + 1
        tail += (f" The full search finds {describe(CANDIDATES[f], lang)}, with AUC {number(SCORES[7000][f], lang, 3)}; "
                 f"on 778 runs it scored {number(SCORES[778][f], lang, 3)}, place {place} of 27, and was dropped in round 1.")
    return head + tail


def _x(runs: int, c: int) -> float:
    """Where candidate c sits across in the column of `runs`: a little to the side, so the dots do not cover each other."""
    return runs * 10 ** (((c * 7) % 9 - 4) * 0.034)


def _rounds_panel(asked: list, kept: list, show_full: bool, labels: dict, lang: str, runs_of: list | None = None):
    """asked[i]: the positions scored in round i; kept[i]: the positions that go on from round i; runs_of[i]: the runs
    of round i, the course's rounds unless given. The tick labels stay plain numbers: Dutch charts swap every point in
    them for a comma."""
    runs_of = runs_of or ROUNDS[:len(asked)]
    p = chart.Panel((520, 10500), (0.62, 1.0), title=labels["rounds"], xlabel=labels["runs"], ylabel="AUC", xlog=True,
                    xticks=ROUNDS, xtick_labels=[str(r) for r in ROUNDS], yticks=[0.7, 0.8, 0.9, 1.0])
    for i in range(1, len(asked)):
        a, b = runs_of[i - 1], runs_of[i]
        for c in asked[i]:
            p.line([_x(a, c), _x(b, c)], [SCORES[a][c], SCORES[b][c]], role="ink-3", width=1)
    if show_full:
        everyone = range(len(CANDIDATES))
        p.points([_x(7000, c) for c in everyone], SCORES[7000], role="ink-3", r=2.2)
        f = full_winner()
        p.points([_x(7000, f)], [SCORES[7000][f]], role="ink", r=7, hollow=True)
    for i, scored in enumerate(asked):
        runs = runs_of[i]
        on = [c for c in scored if c in kept[i]]
        off = [c for c in scored if c not in kept[i]]
        p.points([_x(runs, c) for c in off], [SCORES[runs][c] for c in off], role="hyper", r=3.2, hollow=True)
        p.points([_x(runs, c) for c in on], [SCORES[runs][c] for c in on], role="hyper", r=3.6)
    return p


def _budget_panel(used: int, labels: dict, lang: str):
    top = max(210, used / 1000 * 1.15)   # a search that spends more than the full search stays in the picture
    step = 50 if top <= 320 else 100
    p = chart.Panel((-0.6, 1.6), (0, top), title=labels["budget"], ylabel=labels["thousands"], xticks=[0, 1],
                    xtick_labels=[labels["halving"], labels["full"]], yticks=list(range(0, int(top) + 1, step)))
    p.rect(-0.3, 0, 0.3, used / 1000, role="hyper", solid=True)
    p.rect(0.7, 0, 1.3, FULL / 1000, role="ink-3")
    p.text(0, used / 1000 + top * 0.048, number(used / 1000, lang), role="ink", anchor="middle")
    p.text(1, FULL / 1000 + top * 0.048, number(FULL / 1000, lang), role="ink", anchor="middle")
    return p


def round_picture(r: int, labels: dict, lang: str) -> str:
    """Left: the score of every candidate in the rounds so far, filled when it goes on, and at round 3 the scores of
    all 27 on all 7,000 runs in gray, the best ringed. Right: the budget spent so far against the full search."""
    sizes, winner = halving()
    asked = sizes[:r]
    kept = [sizes[i + 1] if i + 1 < len(sizes) else [winner] for i in range(r)]
    left = _rounds_panel(asked, kept, r == 3, labels, lang)
    right = _budget_panel(spent(r), labels, lang)
    return chart.figure([left, right], alt=labels["alt"], width=330, height=270)


# The lesson is done when these are correct. "small-first" is the choice question in the lesson.
REQUIRED = ["plan", "halving", "small-first"]


class _Val:
    """A test value that the check's messages write by its name, such as `score`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value)


def _function(seen):
    fn = seen.mine.names.get(seen.exercise.function) if seen.mine is not None else None
    return fn if callable(fn) else None


def _call(seen, case: tuple):
    fn = _function(seen)
    if fn is None:
        return None
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return checking.with_time_limit(lambda: fn(*(checking._copy(a) for a in case)))
    except Exception:
        return None


def _key(text: str) -> str:
    return ('<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.15rem">'
            + text + "</div>")


_SAME = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: any("..." in line.split("#")[0] for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


# --- exercise 1: the plan of the rounds --------------------------------------------------------------------------------

def _steps(v):
    """v as a list of (candidates, runs) pairs of numbers, or None."""
    try:
        steps = [tuple(t) for t in v]
        if not all(len(t) == 2 for t in steps):
            return None
        return [(float(a), float(b)) for a, b in steps]
    except Exception:
        return None


def _same_steps(got, want) -> bool:
    mine = _steps(got)
    return mine is not None and mine == _steps(want)


def _shown_steps(v):
    steps = _steps(v)
    if steps is None:
        return repr(v)[:80]
    text = "[" + ", ".join(f"({a:.6g}, {b:.6g})" for a, b in steps) + "]"
    return {"en": text, "nl": text.replace(".", ",")}   # only decimals have a point here: no thousands separators


_P_VISIBLE = (27, 7000, 3, 3)
_P_HIDDEN = [(16, 1000, 2, 4), (81, 8100, 3, 4), (10, 500, 10, 1)]

_P_HEAD = "def plan(n, total, factor, rounds):\n    steps = []\n    for r in range(rounds):\n"


def _p(candidates: str = "round(n / factor ** r)", runs: str = "round(total / factor ** (rounds - 1 - r))") -> str:
    return _P_HEAD + f"        candidates = {candidates}\n        runs = {runs}\n        steps.append((candidates, runs))\n    return steps\n"


_P_START = _p("...", "...")
_P = _p()

_PL_WORDS = {
    "title": {"en": "Your plan: the budget of every round", "nl": "Jouw plan: het budget van elke ronde"},
    "y": {"en": "runs handed out (thousands)", "nl": "uitgedeelde runs (duizenden)"},
    "round": {"en": "round {r}", "nl": "ronde {r}"},
    "full": {"en": "all 27", "nl": "alle 27"},
    "right": {"en": "Each round hands out about 21,000 runs: 27 · 778, 9 · 2,333 and 3 · 7,000. Together that is 63,003, a third of the 189,000 runs of all 27 candidates on all 7,000.",
              "nl": "Elke ronde deelt ongeveer 21.000 runs uit: 27 · 778, 9 · 2.333 en 3 · 7.000. Samen is dat 63.003, een derde van de 189.000 runs van alle 27 kandidaten op alle 7.000."},
    "said": {"en": "Your plan hands out {each} runs in its rounds, {total} in all, against 189,000 for all 27 candidates on all 7,000. The course's plan hands out about 21,000 in every round.",
             "nl": "Jouw plan deelt in zijn rondes {each} runs uit, samen {total}, tegen 189.000 voor alle 27 kandidaten op alle 7.000. Het plan van de course deelt in elke ronde ongeveer 21.000 uit."},
}


def _pl_picture(seen) -> str:
    lang = seen.lang
    steps = _steps(_call(seen, _P_VISIBLE))
    if not steps or len(steps) > 6 or any(a < 0 or b < 0 for a, b in steps):
        return ""
    words = {k: v[lang] for k, v in _PL_WORDS.items()}
    costs = [a * b for a, b in steps]
    if _same_steps(steps, [(27, 778), (9, 2333), (3, 7000)]):
        text = words["right"]
    else:
        joined = " + ".join(number(c, lang) for c in costs)
        text = words["said"].format(each=joined, total=number(sum(costs), lang))
    top = max(FULL, max(costs)) / 1000 * 1.15
    n = len(steps)
    p = chart.Panel((-0.6, n + 0.6), (0, top), ylabel=words["y"], xticks=list(range(n + 1)),
                    xtick_labels=[words["round"].format(r=i + 1) for i in range(n)] + [words["full"]],
                    yticks=[t for t in (0, 50, 100, 150, 200) if t <= top])
    for i, c in enumerate(costs):
        p.rect(i - 0.3, 0, i + 0.3, c / 1000, role="hyper", solid=True)
        p.text(i, c / 1000 + top * 0.05, number(c / 1000, lang), role="ink", anchor="middle")
    p.rect(n - 0.3, 0, n + 0.3, FULL / 1000, role="ink-3")
    p.text(n, FULL / 1000 + top * 0.05, number(FULL / 1000, lang), role="ink", anchor="middle")
    return yours.figure(words["title"], chart.figure([p], alt=text, width=460, height=260) + _key(text), alt=text)


EXERCISES = {
    "plan": FunctionBox(
        name="plan",
        picture=_pl_picture,
        function="plan",
        start=_P_START,
        solution=_P,
        show=_shown_steps,
        compare=_same_steps,
        visible=[_P_VISIBLE],
        hidden=_P_HIDDEN,
        first=[_DOTS],
        mistakes=[
            Mistake("runs-first", wrong=_p(runs="round(total / factor ** r)"),
                    en="`{call}` gives {got}: the most candidates get the most runs. Successive halving starts small: round 0 gets the fewest runs, and the last round gets all `total`. Count the exponent of the runs down from `rounds - 1` to 0.",
                    nl="`{call}` geeft {got}: de meeste kandidaten krijgen de meeste runs. Successive halving begint klein: ronde 0 krijgt de minste runs, en de laatste ronde krijgt alle `total`. Tel de exponent van de runs af van `rounds - 1` naar 0."),
            Mistake("all-candidates", wrong=_p("n"),
                    en="`{call}` gives {got}: every round keeps all `n` candidates. After every round only the best part goes on: divide `n` by `factor` once for every round so far.",
                    nl="`{call}` geeft {got}: elke ronde houdt alle `n` kandidaten. Na elke ronde gaat alleen het beste deel door: deel `n` door `factor`, één keer voor elke ronde tot nu toe."),
            Mistake("cut-off", wrong=_p(runs="int(total / factor ** (rounds - 1 - r))"),
                    en="`{call}` gives {got}: your runs are cut off, not rounded, so 7,000 / 9 = 777.8 becomes 777. `round` goes to the nearest whole number: 778.",
                    nl="`{call}` geeft {got}: je runs worden afgekapt, niet afgerond, dus 7.000 / 9 = 777,8 wordt 777. `round` rondt af op het dichtstbijzijnde gehele getal: 778."),
            Mistake("not-rounded", wrong=_p(runs="total / factor ** (rounds - 1 - r)"),
                    en="`{call}` gives {got}: 7,000 / 9 is 777.8, and a number of runs is a whole number. `round` goes to the nearest one: 778.",
                    nl="`{call}` geeft {got}: 7.000 / 9 is 777,8, en een aantal runs is een geheel getal. `round` rondt af op het dichtstbijzijnde: 778."),
            Mistake("one-too-many", wrong=_p(runs="round(total / factor ** (rounds - r))"),
                    en="`{call}` gives {got}: even the last round gets only a part of `total`. The last round, `r = rounds - 1`, should get all of it, with exponent 0.",
                    nl="`{call}` geeft {got}: zelfs de laatste ronde krijgt maar een deel van `total`. De laatste ronde, `r = rounds - 1`, moet alles krijgen, met exponent 0."),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the two `...`: how many candidates are left in round `r`, and how many runs each of them gets.",
                      "nl": "Vervang de twee `...`: hoeveel kandidaten er in ronde `r` over zijn, en hoeveel runs elk ervan krijgt."},
            "arguments": {"en": "The check calls `{call}`, with four values, and the `def` line of your function has a different number of names between its parentheses. Keep `(n, total, factor, rounds)`.",
                          "nl": "De check roept `{call}` aan, met vier waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(n, total, factor, rounds)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}: in every round a `factor` times fewer candidates, each with a `factor` times more runs, and all `total` runs in the last round.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn: in elke ronde `factor` keer minder kandidaten, elk met `factor` keer meer runs, en alle `total` runs in de laatste ronde."},
            "almost": {"en": "Right for the plan of the lesson. But `{call}` gives {got}, and it should give {want}. Does your function use its four values, and no fixed number such as 27 or 3?",
                       "nl": "Goed voor het plan van de les. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de vier waarden die ze krijgt, en geen vast getal zoals 27 of 3?"},
            "correct": {"en": "Your function is right in all {n} cases. Every round of the lesson's plan hands out about 21,000 runs, so three rounds cost a third of the full search.",
                        "nl": "Je functie klopt in alle {n} gevallen. Elke ronde van het plan van de les deelt ongeveer 21.000 runs uit, dus drie rondes kosten een derde van de volledige zoektocht."},
        },
    ),
}


# --- exercise 2: successive halving -------------------------------------------------------------------------------------

def _letters_score(c, runs):
    table = {10: dict(zip("abcdefghi", [0.61, 0.72, 0.55, 0.70, 0.64, 0.58, 0.69, 0.50, 0.66])),
             100: dict(zip("abcdefghi", [0.80, 0.71, 0.60, 0.83, 0.77, 0.62, 0.74, 0.57, 0.79]))}
    return table[runs][c]


def _numbers_score(c, runs):
    return -abs(c - runs / 7) + c / 1000


_H_VISIBLE = (_Val("candidates", CANDIDATES), _Val("score", score), _Val("rounds", ROUNDS))
_H_HIDDEN = [
    (list("abcdefghi"), _Val("score_letters", _letters_score), [10, 100]),
    (list(range(81)), _Val("score_numbers", _numbers_score), [10, 30, 90, 270]),
    (["x", "y", "z"], _Val("score_letters", lambda c, runs: {"x": 0.3, "y": 0.9, "z": 0.6}[c]), [5]),
]

_H_HEAD = "import numpy as np\n\n\ndef successive_halving(candidates, score, rounds):\n    for runs in rounds:\n        scores = []\n        for c in candidates:\n"


def _h(append: str = "score(c, runs)", best: str = "np.argsort(scores)[-keep:]") -> str:
    return (_H_HEAD + f"            scores.append({append})\n        keep = round(len(candidates) / 3)\n        best = {best}\n"
            "        kept = []\n        for i in best:\n            kept.append(candidates[i])\n        candidates = kept\n"
            "    return candidates\n")


_H_START = _h("...", "...")
_H = _h()


def _same_list(got, want) -> bool:
    try:
        return sorted(map(repr, [tuple(x) if isinstance(x, list) else x for x in got])) == sorted(map(repr, want))
    except Exception:
        return False


def _shown_list(v):
    try:
        items = list(v)
    except Exception:
        return repr(v)[:80]
    if len(items) > 4:
        return {"en": f"a list of {len(items)} candidates", "nl": f"een lijst van {len(items)} kandidaten"}
    return "[" + ", ".join(repr(tuple(x) if isinstance(x, list) else x) for x in items) + "]"


_HV_WORDS = {
    "title": {"en": "Your successive halving on the 27 candidates", "nl": "Jouw successive halving op de 27 kandidaten"},
    "rounds": {"en": "Scores your search asked for", "nl": "Scores die je zoektocht vroeg"},
    "runs": {"en": "training runs", "nl": "runs om op te trainen"},
    "budget": {"en": "Budget", "nl": "Budget"},
    "thousands": {"en": "runs (thousands)", "nl": "runs (duizenden)"},
    "halving": {"en": "yours", "nl": "jouw"},
    "full": {"en": "all 27", "nl": "alle 27"},
    "right": {"en": "Your search asked for 27 scores on 778 runs, 9 on 2,333 and 3 on 7,000, and kept {winner}, AUC {auc}. The open dots dropped out, the filled ones went on; the gray dots are all 27 on 7,000 runs, the best of them ringed.",
              "nl": "Je zoektocht vroeg 27 scores op 778 runs, 9 op 2.333 en 3 op 7.000, en hield {winner} over, AUC {auc}. De open stippen vielen af, de dichte gingen door; de grijze stippen zijn alle 27 op 7.000 runs, de beste ervan omcirkeld."},
    "said": {"en": "Your search asked for {asked} scores on 778, 2,333 and 7,000 runs, and gave back {back}. The course's search asks for 27, 9 and 3, and keeps one candidate.",
             "nl": "Je zoektocht vroeg {asked} scores op 778, 2.333 en 7.000 runs, en gaf {back} terug. De zoektocht van de course vraagt er 27, 9 en 3, en houdt één kandidaat over."},
    "one": {"en": "1 candidate", "nl": "1 kandidaat"},
    "many": {"en": "{n} candidates", "nl": "{n} kandidaten"},
}


def _hv_picture(seen) -> str:
    lang = seen.lang
    fn = _function(seen)
    if fn is None:
        return ""
    asked = {runs: [] for runs in ROUNDS}

    def logged(c, runs):
        i = CANDIDATES.index(tuple(c))
        asked[runs].append(i)
        return SCORES[runs][i]

    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = checking.with_time_limit(lambda: fn(list(CANDIDATES), logged, list(ROUNDS)))
        back = [tuple(x) for x in back]
        positions = [CANDIDATES.index(x) for x in back]
    except Exception:
        return ""
    words = {k: v[lang] for k, v in _HV_WORDS.items()}
    sizes, winner = halving()
    runs_of = [r for r in ROUNDS if asked[r]]   # the rounds that asked for scores, each drawn at its own runs
    rounds = [sorted(set(asked[r])) for r in runs_of]
    kept = [rounds[i + 1] if i + 1 < len(rounds) else positions for i in range(len(rounds))]
    if [len(asked[r]) for r in ROUNDS] == [len(s) for s in sizes] and positions == [winner]:   # two candidates are the same
        text = words["right"].format(winner=describe(CANDIDATES[winner], lang), auc=number(SCORES[7000][winner], lang, 3))
    else:
        n = len(back)
        text = words["said"].format(asked=", ".join(str(len(asked[r])) for r in ROUNDS),
                                    back=words["one"] if n == 1 else words["many"].format(n=n))
    used = sum(len(asked[r]) * r for r in ROUNDS)
    labels = {k: words[k] for k in ("rounds", "runs", "budget", "thousands", "halving", "full")}
    left = _rounds_panel(rounds, kept, True, labels, lang, runs_of)
    right = _budget_panel(used, labels, lang)
    return yours.figure(words["title"], chart.figure([left, right], alt=text, width=330, height=270) + _key(text), alt=text)


EXERCISES["halving"] = FunctionBox(
    name="halving",
    picture=_hv_picture,
    function="successive_halving",
    start=_H_START,
    solution=_H,
    show=_shown_list,
    compare=_same_list,
    visible=[_H_VISIBLE],
    hidden=_H_HIDDEN,
    first=[_DOTS],
    mistakes=[
        Mistake("lowest", wrong=_h(best="np.argsort(scores)[:keep]"),
                en="`{call}` gives {got}: `np.argsort` puts the lowest score first, so your search keeps the worst third of every round. Take the positions from the other end.",
                nl="`{call}` geeft {got}: `np.argsort` zet de laagste score vooraan, dus je zoektocht houdt in elke ronde het slechtste derde deel. Neem de posities van de andere kant."),
        Mistake("drop-worst", wrong=_h(best="np.argsort(scores)[keep:]"),
                en="`{call}` gives {got}: your search drops only the worst `keep` of every round and keeps all the others, so more than one candidate is left at the end. Keep only the best `keep`.",
                nl="`{call}` geeft {got}: je zoektocht laat in elke ronde alleen de slechtste `keep` vallen en houdt alle andere, dus aan het eind is er meer dan één kandidaat over. Houd alleen de beste `keep`."),
        Mistake("argmax", error=lambda e: isinstance(e, TypeError) and "not iterable" in str(e),
                en="`best` is one number, and the loop `for i in best` needs several: the positions of the best `keep`. `np.argmax`, or an index without `:` such as `[-keep]`, gives one position; a slice gives several.",
                nl="`best` is één getal, en de loop `for i in best` heeft er meer nodig: de posities van de beste `keep`. `np.argmax`, of een index zonder `:` zoals `[-keep]`, geeft één positie; een slice geeft er meer."),
        Mistake("score-values", error=lambda e: isinstance(e, KeyError) or isinstance(e, TypeError) and (
                    "positional argument" in str(e) or "unhashable" in str(e)),
                en="`score` needs two values, in this order: the candidate `c` and the runs of this round, `runs` (one number, not the list `rounds`).",
                nl="`score` heeft twee waarden nodig, in deze volgorde: de kandidaat `c` en de runs van deze ronde, `runs` (één getal, niet de lijst `rounds`)."),
        Mistake("all-runs", wrong=_h("score(c, rounds[-1])"),
                en="`{call}` gives {got}: every round scores the candidates on the runs of the last round, so the search costs more than trying all of them on everything once. Score each candidate on the `runs` of this round.",
                nl="`{call}` geeft {got}: elke ronde geeft de kandidaten een score op de runs van de laatste ronde, dus de zoektocht kost meer dan ze allemaal één keer op alles proberen. Geef elke kandidaat een score op de `runs` van deze ronde."),
        Mistake("values", error=lambda e: isinstance(e, (TypeError, IndexError)) and "integer" in str(e),
                en="`best` holds scores, not positions: `candidates[i]` needs a whole number. `np.sort(scores)` or a slice of `scores` gives the scores themselves; `np.argsort(scores)` gives their positions, from the lowest score to the highest.",
                nl="`best` bevat scores, geen posities: `candidates[i]` heeft een geheel getal nodig. `np.sort(scores)` of een slice van `scores` geeft de scores zelf; `np.argsort(scores)` geeft hun posities, van de laagste score naar de hoogste."),
    ],
    texts=_SAME | {
        "empty": {"en": "Replace the two `...`: the score of candidate `c` on the runs of this round, and the positions of the best `keep` scores.",
                  "nl": "Vervang de twee `...`: de score van kandidaat `c` op de runs van deze ronde, en de posities van de beste `keep` scores."},
        "arguments": {"en": "The check calls `{call}`, with three values, and the `def` line of your function has a different number of names between its parentheses. Keep `(candidates, score, rounds)`.",
                      "nl": "De check roept `{call}` aan, met drie waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(candidates, score, rounds)`."},
        "wrong": {"en": "`{call}` gives {got}, and it should give {want}: in every round the scores of the candidates on that round's runs, and the best `keep` of them go on.",
                  "nl": "`{call}` geeft {got}, en dat moet {want} zijn: in elke ronde de scores van de kandidaten op de runs van die ronde, en de beste `keep` ervan gaan door."},
        "almost": {"en": "Right for the 27 candidates. But `{call}` gives {got}, and it should give {want}. Does your function use its `score` and `rounds`, and no fixed number such as 27 or 9?",
                   "nl": "Goed voor de 27 kandidaten. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `score` en `rounds` die ze krijgt, en geen vast getal zoals 27 of 9?"},
        "correct": {"en": "Your function is right in all {n} cases. On the 27 candidates it keeps the forest with depth 5 and 3 features per split, for a third of the budget of the full search.",
                    "nl": "Je functie klopt in alle {n} gevallen. Op de 27 kandidaten houdt ze het random forest met diepte 5 en 3 features per splitsing over, voor een derde van het budget van de volledige zoektocht."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("plan", _P_START, "empty", None),
    ("plan", _p(runs="..."), "wrong", "dots-left"),
    ("plan", _P, "correct", None),
    ("plan", _p("int(n / factor ** r)", "round(total * factor ** r / factor ** (rounds - 1))"), "correct", None),
    ("plan", _p("n // factor ** r"), "correct", None),
    ("plan", _p(runs="round(total / factor ** r)"), "wrong", "runs-first"),
    ("plan", _p("n"), "wrong", "all-candidates"),
    ("plan", _p(runs="round(total / factor ** (rounds - r))"), "wrong", "one-too-many"),
    ("plan", _p(runs="int(total / factor ** (rounds - 1 - r))"), "wrong", "cut-off"),
    ("plan", _p(runs="total // factor ** (rounds - 1 - r)"), "wrong", "cut-off"),
    ("plan", _p(runs="total / factor ** (rounds - 1 - r)"), "wrong", "not-rounded"),
    ("halving", _H_START, "empty", None),
    ("halving", _h(best="..."), "wrong", "dots-left"),
    ("halving", _H, "correct", None),
    ("halving", _h(best="np.argsort(scores)[::-1][:keep]"), "correct", None),
    ("halving", _h(best="np.argsort(-np.array(scores))[:keep]"), "correct", None),
    ("halving", _h(best="np.argsort(scores)[:keep]"), "wrong", "lowest"),
    ("halving", _h(best="np.argsort(scores)[keep:]"), "wrong", "drop-worst"),
    ("halving", _h("score(c, rounds[-1])"), "wrong", "all-runs"),
    ("halving", _h(best="np.sort(scores)[-keep:]"), "error", "values"),
    ("halving", _h(best="np.argmax(scores)"), "error", "argmax"),
    ("halving", _h(best="np.argsort(scores)[-keep]"), "error", "argmax"),
    ("halving", _h(best="scores[-keep:]"), "error", "values"),
    ("halving", _h("score(runs, c)"), "error", "score-values"),
    ("halving", _h("score(c)"), "error", "score-values"),
    ("halving", _h("score(c, rounds)"), "error", "score-values"),
]
