# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-grid-random-search: k-NN with scaled numbers on the 7,000 training runs of AI4I 2020 (split
as in ml-imbalanced), scored by the mean ROC AUC of cross-validation with KFold(5), for two settings: k from 1 to 300
and the power p of the distance from 1 to 4. A grid spreads m values of each evenly over its range, away from the
ends (k on a log scale); a random search draws k on a log scale and p evenly, with NumPy's generator. The tries and
their scores in capitals were measured once; tests/test_ml_grid_random_search.py measures them again."""
import contextlib
import io
import warnings

from everything_ai import chart, check as checking, yours
from everything_ai.check import FunctionBox, Mistake

DATA = "data/ai4i-2020.csv.gz"
NUMBERS = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
K_RANGE, P_RANGE = (1, 300), (1, 4)
BUDGETS = [4, 9, 16, 25]

# Measured once, as (k, p, mean AUC of the 5 folds), rounded to 4 decimals; tests/test_ml_grid_random_search.py
# measures them again. GRID: the course's grid for 4, 9, 16 and 25 tries. RANDOM: draws(0, 25). SEEDS: the best
# AUC of draws(seed, 9) for seeds 0 to 19. CURVE: (k, AUC) with p = 2. TEST_AUC: the AUC on the 3,000 test runs
# of k-NN fitted on all 7,000 training runs with the winner of the random search of 9 tries and of the grid of 9.
GRID = {
    4: [
        (4, 1.75, 0.8461), (4, 3.25, 0.8449), (72, 1.75, 0.951), (72, 3.25, 0.9496),
    ],
    9: [
        (3, 1.5, 0.8344), (3, 2.5, 0.8279), (3, 3.5, 0.8318), (17, 1.5, 0.9097), (17, 2.5, 0.9134),
        (17, 3.5, 0.9147), (116, 1.5, 0.9467), (116, 2.5, 0.9456), (116, 3.5, 0.9459),
    ],
    16: [
        (2, 1.375, 0.799), (2, 2.125, 0.7923), (2, 2.875, 0.7914), (2, 3.625, 0.7926), (8, 1.375, 0.876),
        (8, 2.125, 0.8767), (8, 2.875, 0.8767), (8, 3.625, 0.8753), (35, 1.375, 0.9304), (35, 2.125, 0.9356),
        (35, 2.875, 0.9341), (35, 3.625, 0.9338), (147, 1.375, 0.9445), (147, 2.125, 0.9442), (147, 2.875, 0.9433),
        (147, 3.625, 0.943),
    ],
    25: [
        (2, 1.3, 0.7937), (2, 1.9, 0.7963), (2, 2.5, 0.7905), (2, 3.1, 0.7886), (2, 3.7, 0.7904), (6, 1.3, 0.8612),
        (6, 1.9, 0.8636), (6, 2.5, 0.8666), (6, 3.1, 0.8668), (6, 3.7, 0.8625), (17, 1.3, 0.9074), (17, 1.9, 0.9169),
        (17, 2.5, 0.9134), (17, 3.1, 0.9169), (17, 3.7, 0.9128), (54, 1.3, 0.9446), (54, 1.9, 0.9439),
        (54, 2.5, 0.9402), (54, 3.1, 0.9406), (54, 3.7, 0.9375), (170, 1.3, 0.9425), (170, 1.9, 0.9427),
        (170, 2.5, 0.9414), (170, 3.1, 0.9405), (170, 3.7, 0.94),
    ],
}
RANDOM = [
    (37, 1.81, 0.9332), (1, 1.05, 0.6997), (103, 3.74, 0.9452), (31, 3.19, 0.9278), (22, 3.81, 0.9247),
    (104, 1.01, 0.9479), (133, 1.1, 0.9452), (64, 1.53, 0.9524), (137, 2.62, 0.944), (5, 2.27, 0.8569),
    (1, 1.37, 0.7117), (45, 2.94, 0.9396), (33, 2.15, 0.9332), (295, 3.94, 0.9323), (49, 2.95, 0.9394),
    (50, 2.17, 0.9432), (2, 3.16, 0.7908), (20, 1.93, 0.9182), (15, 3.67, 0.9101), (205, 2.07, 0.9384),
    (26, 1.97, 0.9252), (29, 2.01, 0.9263), (9, 3.67, 0.8821), (3, 2.87, 0.8322), (1, 3.5, 0.7133),
]
SEEDS = [
    0.9524, 0.9496, 0.9479, 0.9499, 0.946, 0.9458, 0.9453, 0.948, 0.9462, 0.9449, 0.9472, 0.9392, 0.9404, 0.947,
    0.9502, 0.9519, 0.9431, 0.9473, 0.9504, 0.9492,
]
CURVE = [
    (1, 0.7162), (2, 0.7919), (3, 0.8342), (4, 0.8479), (5, 0.8584), (6, 0.864), (7, 0.8717), (9, 0.8783),
    (11, 0.8842), (13, 0.9005), (16, 0.9124), (19, 0.9161), (23, 0.9189), (28, 0.9265), (34, 0.9332), (42, 0.9378),
    (51, 0.9437), (62, 0.9516), (76, 0.9505), (92, 0.9485), (112, 0.9459), (137, 0.9448), (166, 0.9429),
    (202, 0.9387), (246, 0.9373), (300, 0.9332),
]
TEST_AUC = {"random": 0.960, "grid": 0.962}


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def grid_values(m: int) -> tuple:
    """The m values of k and of p of the course's grid: spread evenly over each range, away from the ends, k on a log
    scale."""
    import numpy as np
    ks = (10 ** ((np.arange(m) + 0.5) / m * np.log10(K_RANGE[1]))).round().astype(int).tolist()
    ps = (1 + 3 * (np.arange(m) + 0.5) / m).round(3).tolist()
    return ks, ps


def draws(seed: int, n: int) -> list:
    """The tries of a random search, as the worked example draws them: k on a log scale, then p evenly."""
    import numpy as np
    rng = np.random.default_rng(seed)
    tries = []
    for _ in range(n):
        k = int(10 ** rng.uniform(0, np.log10(K_RANGE[1])))
        p = round(float(rng.uniform(*P_RANGE)), 2)
        tries.append((k, p))
    return tries


def best(tries: list) -> tuple:
    """The try with the highest score: (k, p, AUC)."""
    return max(tries, key=lambda t: t[2])


# --- the slider -----------------------------------------------------------------------------------------------------

def _p(x: float, lang: str) -> str:
    text = f"{x:g}"
    return text.replace(".", ",") if lang == "nl" else text


def budget_said(n: int, lang: str) -> str:
    """The sentence under the picture of the slider."""
    g, r = GRID[n], RANDOM[:n]
    gk, gp, ga = best(g)
    rk, rp, ra = best(r)
    m, distinct = int(round(n ** 0.5)), len({t[0] for t in r})
    if lang == "nl":
        return (f"{n} pogingen. Het grid probeert {m} waarden van k en vindt AUC {number(ga, lang, 3)} bij k = {gk}, "
                f"p = {_p(gp, lang)}. De random search probeert {distinct} waarden van k en vindt {number(ra, lang, 3)} "
                f"bij k = {rk}, p = {_p(rp, lang)}.")
    return (f"{n} tries. The grid tries {m} values of k and finds AUC {number(ga, lang, 3)} at k = {gk}, p = {_p(gp, lang)}. "
            f"The random search tries {distinct} values of k and finds {number(ra, lang, 3)} at k = {rk}, p = {_p(rp, lang)}.")


_KTICKS = [1, 3, 10, 30, 100, 300]


def budget_picture(n: int, labels: dict) -> str:
    """Left: the tries of the grid (open) and of the random search (filled) on the plane of k and p. Right: the
    score of every try against its k, beside the line of p = 2."""
    g, r = GRID[n], RANDOM[:n]
    left = chart.Panel((0.8, 380), (0.8, 4.2), title=labels["plane"], xlabel="k", ylabel=labels["p"], xlog=True,
                       xticks=_KTICKS, xtick_labels=[str(t) for t in _KTICKS], yticks=[1, 2, 3, 4])
    right = chart.Panel((0.8, 380), (0.68, 0.97), title=labels["score"], xlabel="k", ylabel="AUC", xlog=True,
                        xticks=_KTICKS, xtick_labels=[str(t) for t in _KTICKS], yticks=[0.7, 0.8, 0.9])
    right.line([c[0] for c in CURVE], [c[1] for c in CURVE], role="ink-3", width=1.4)
    for tries in (g, r):                                   # the rings first, so that every dot shows inside its ring
        k, p, auc = best(tries)
        left.points([k], [p], role="ink", r=7, hollow=True)
        right.points([k], [auc], role="ink", r=7, hollow=True)
    left.points([t[0] for t in g], [t[1] for t in g], role="hyper", r=4, hollow=True)
    left.points([t[0] for t in r], [t[1] for t in r], role="hyper", r=3.6)
    right.points([t[0] for t in g], [t[2] for t in g], role="hyper", r=4, hollow=True)
    right.points([t[0] for t in r], [t[2] for t in r], role="hyper", r=3.6)
    return chart.figure([left, right], alt=labels["alt"], width=330, height=260)


# The lesson is done when these are correct. "few-matter" is the choice question in the lesson.
REQUIRED = ["grid", "log-uniform", "few-matter"]


class _Val:
    """A test value that the check's messages write by its name, such as `rng`; the function gets a copy."""

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


# --- exercise 1: every combination ------------------------------------------------------------------------------------

def _pairs(v):
    """v as a sorted list of (k, p) pairs, or None."""
    try:
        pairs = [tuple(t) for t in v]
    except Exception:
        return None
    if not all(len(t) == 2 for t in pairs):
        return None
    try:
        return sorted((float(a), float(b)) for a, b in pairs)
    except (TypeError, ValueError):
        return None


def _flat(v) -> bool:
    """True when v is a list of single numbers instead of pairs."""
    return isinstance(v, list) and len(v) > 0 and all(isinstance(t, (int, float)) for t in v)


def _same_pairs(got, want) -> bool:
    mine = _pairs(got)
    return mine is not None and mine == _pairs(want)


def _shown_pairs(v):
    pairs = _pairs(v)
    if pairs is None:
        return repr(v)[:60]
    if len(pairs) > 9:
        return {"en": f"a list of {len(pairs)} pairs", "nl": f"een lijst van {len(pairs)} paren"}
    return "[" + ", ".join(f"({a:g}, {b:g})" for a, b in pairs) + "]"


_KS, _PS = grid_values(3)
_G_VISIBLE = (_Val("ks", _KS), _Val("ps", _PS))
_G_HIDDEN = [([1, 2], [10, 20, 30]), ([5, 6, 7, 8], [2.0]), ([40], [1.0, 4.0])]

_G_HEAD = "def grid(ks, ps):\n    tries = []\n"


def _g(body: str = "    for k in ks:\n        for p in ps:\n            tries.append((k, p))\n") -> str:
    return _G_HEAD + body + "    return tries\n"


_G_START = _g("    ...\n")
_G = _g()

_GR_WORDS = {
    "title": {"en": "Your grid on the plane of k and p", "nl": "Jouw grid op het vlak van k en p"},
    "right": {"en": "Your grid has 9 tries: every one of the 3 values of k with every one of the 3 values of p. Each value of k is tried 3 times, once with every p.",
              "nl": "Jouw grid heeft 9 pogingen: elk van de 3 waarden van k met elk van de 3 waarden van p. Elke waarde van k wordt 3 keer geprobeerd, één keer met elke p."},
    "said": {"en": "Your function gives {n} tries, with {kn} values of k and {pn} of p; the grid of 3 values of k and 3 of p has 9 tries.",
             "nl": "Je functie geeft {n} pogingen, met {kn} waarden van k en {pn} van p; het grid van 3 waarden van k en 3 van p heeft 9 pogingen."},
}


def _gr_picture(seen) -> str:
    lang = seen.lang
    pairs = _pairs(_call(seen, _G_VISIBLE))
    if pairs is None:
        return ""
    words = {k: v[lang] for k, v in _GR_WORDS.items()}
    if _same_pairs(pairs, [(k, p) for k in _KS for p in _PS]):
        text = words["right"]
    else:
        text = words["said"].format(n=len(pairs), kn=len({a for a, _ in pairs}), pn=len({b for _, b in pairs}))
    p = chart.Panel((0.8, 380), (0.8, 4.2), xlabel="k", ylabel="p", xlog=True, xticks=_KTICKS,
                    xtick_labels=[str(t) for t in _KTICKS], yticks=[1, 2, 3, 4])
    shown = [(a, b) for a, b in pairs if 0.8 <= a <= 380 and 0.8 <= b <= 4.2]
    p.points([a for a, _ in shown], [b for _, b in shown], role="hyper", r=4.5, hollow=True)
    for k in sorted({a for a, _ in shown}):
        p.line([k, k], [0.8, 4.2], role="ink-3", width=0.8, dashed=True)
    return yours.figure(words["title"], chart.figure([p], alt=text, width=460, height=240) + _key(text), alt=text)


EXERCISES = {
    "grid": FunctionBox(
        name="grid",
        picture=_gr_picture,
        function="grid",
        start=_G_START,
        solution=_G,
        show=_shown_pairs,
        compare=_same_pairs,
        visible=[_G_VISIBLE],
        hidden=_G_HIDDEN,
        first=[_DOTS],
        mistakes=[
            Mistake("zip", wrong=_g("    for k, p in zip(ks, ps):\n        tries.append((k, p))\n"),
                    en="`{call}` gives {got}: `zip` pairs the first k with the first p, the second with the second, and so on, so every k meets only one p. A grid tries every k with every p: a loop over `ps` inside a loop over `ks`.",
                    nl="`{call}` geeft {got}: `zip` koppelt de eerste k aan de eerste p, de tweede aan de tweede, enzovoort, dus elke k ontmoet maar één p. Een grid probeert elke k met elke p: een loop over `ps` binnen een loop over `ks`."),
            Mistake("outer-only", wrong=_g("    for k in ks:\n        for p in ps:\n            pass\n        tries.append((k, p))\n"),
                    en="`{call}` gives {got}: the `append` is in the outer loop, so it runs once per k, with the last p. Indent it one step further, into the loop over `ps`.",
                    nl="`{call}` geeft {got}: de `append` staat in de buitenste loop, dus die regel draait één keer per k, met de laatste p. Zet die een stap verder in, in de loop over `ps`."),
            Mistake("flat", matches=lambda got, text: _flat(got),
                    en="`{call}` gives {got}: the values one after the other, not as pairs. Every try is a pair `(k, p)`: append `(k, p)` inside a loop over `ps` inside a loop over `ks`.",
                    nl="`{call}` geeft {got}: de waarden achter elkaar, niet als paren. Elke poging is een paar `(k, p)`: voeg `(k, p)` toe binnen een loop over `ps` binnen een loop over `ks`."),
            Mistake("swapped", wrong=_g("    for k in ks:\n        for p in ps:\n            tries.append((p, k))\n"),
                    en="`{call}` gives {got}: your pairs are `(p, k)`. Put k first: `(k, p)`.",
                    nl="`{call}` geeft {got}: je paren zijn `(p, k)`. Zet k voorop: `(k, p)`."),
            Mistake("append-two", error=lambda e: isinstance(e, TypeError) and "append" in str(e),
                    en="`append` takes one value: put the pair in its own parentheses, `(k, p)`.",
                    nl="`append` neemt één waarde: zet het paar tussen eigen haakjes, `(k, p)`."),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the `...` with code that adds every pair `(k, p)` to `tries`.",
                      "nl": "Vervang de `...` door code die elk paar `(k, p)` aan `tries` toevoegt."},
            "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has a different number of names between its parentheses. Keep `(ks, ps)`.",
                          "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(ks, ps)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}: every value of `ks` with every value of `ps`, as pairs `(k, p)`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn: elke waarde van `ks` met elke waarde van `ps`, als paren `(k, p)`."},
            "almost": {"en": "Right for the grid of the lesson. But `{call}` gives {got}, and it should give {want}. Does your function use the `ks` and `ps` it gets?",
                       "nl": "Goed voor het grid van de les. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `ks` en `ps` die ze krijgt?"},
            "correct": {"en": "Your function is right in all {n} cases. With 3 values of each setting the grid has 9 tries; with 4 settings it would have 81.",
                        "nl": "Je functie klopt in alle {n} gevallen. Met 3 waarden per instelling heeft het grid 9 pogingen; met 4 instellingen zouden het er 81 zijn."},
        },
    ),
}


# --- exercise 2: evenly on a log scale --------------------------------------------------------------------------------

def _rng(seed: int):
    import numpy as np
    return np.random.default_rng(seed)


_L_VISIBLE = (_Val("np.random.default_rng(0)", _rng(0)), 1, 300, 9)
_L_HIDDEN = [(_Val(f"np.random.default_rng({seed})", _rng(seed)), low, high, n)
             for seed, low, high, n in ((1, 0.0001, 0.1, 5), (2, 10, 1000, 4), (3, 2, 8, 6))]

_L_HEAD = "import numpy as np\n\n\ndef log_uniform(rng, low, high, n):\n"


def _l(logs: str = "rng.uniform(np.log10(low), np.log10(high), n)", back: str = "10 ** logs") -> str:
    return _L_HEAD + f"    logs = {logs}\n    return {back}\n"


_L_START = _l("rng.uniform(..., ..., n)", "...")
_L = _l()


def _floats(v):
    import numpy as np
    try:
        a = np.asarray(v, dtype=float)
    except Exception:
        return None
    return a if a.ndim == 1 else None


def _same_floats(got, want) -> bool:
    import numpy as np
    a, b = _floats(got), _floats(want)
    return a is not None and b is not None and a.shape == b.shape and bool(np.allclose(a, b, rtol=1e-9, atol=0))


def _shown_floats(v):
    a = _floats(v)
    if a is None:
        return repr(v)[:60]
    return "[" + ", ".join(f"{x:.4g}" for x in a[:6]) + (", ..." if len(a) > 6 else "") + "]"


def _bands(values) -> tuple:
    """How many of the values lie below 10, from 10 to 100, and above 100."""
    import numpy as np
    v = np.asarray(values, dtype=float)
    return int((v < 10).sum()), int(((v >= 10) & (v <= 100)).sum()), int((v > 100).sum())


_LU_WORDS = {
    "title": {"en": "200 draws of your function from 1 to 300", "nl": "200 trekkingen van jouw functie van 1 tot 300"},
    "x": {"en": "k (log scale)", "nl": "k (logaritmische schaal)"},
    "right": {"en": "Every dot is one draw. On the log scale they spread evenly: {low} below 10 and {mid} from 10 to 100, about as many per factor 10, and {high} from 100 to 300, which is less than half a factor 10, so fewer than half as many. Drawn evenly between 1 and 300, only about 6 of 200 would be below 10.",
              "nl": "Elke stip is één trekking. Op de logaritmische schaal liggen ze gelijk verdeeld: {low} onder 10 en {mid} van 10 tot 100, ongeveer evenveel per factor 10, en {high} van 100 tot 300, minder dan een halve factor 10, dus minder dan half zoveel. Gelijk verdeeld getrokken tussen 1 en 300 zouden er maar ongeveer 6 van de 200 onder 10 liggen."},
    "said": {"en": "Every dot is one draw of your function: {low} below 10, {mid} from 10 to 100 and {high} above 100. Evenly on the log scale gives about {want_low}, {want_mid} and {want_high}.",
             "nl": "Elke stip is één trekking van jouw functie: {low} onder 10, {mid} van 10 tot 100 en {high} boven 100. Gelijk verdeeld op de logaritmische schaal geeft ongeveer {want_low}, {want_mid} en {want_high}."},
}


def _lu_picture(seen) -> str:
    import numpy as np
    lang = seen.lang
    got = _floats(_call(seen, (_Val("np.random.default_rng(1)", _rng(1)), 1, 300, 200)))
    if got is None or len(got) != 200 or not np.all(np.isfinite(got)):
        return ""
    words = {k: v[lang] for k, v in _LU_WORDS.items()}
    want = 10 ** _rng(1).uniform(0, np.log10(300), 200)
    low, mid, high = _bands(got)
    if np.allclose(got, want):
        text = words["right"].format(low=low, mid=mid, high=high)
    else:
        wl, wm, wh = _bands(want)
        text = words["said"].format(low=low, mid=mid, high=high, want_low=wl, want_mid=wm, want_high=wh)
    p = chart.Panel((0.8, 380), (-0.6, 0.6), xlabel=words["x"], xlog=True, xticks=_KTICKS,
                    xtick_labels=[str(t) for t in _KTICKS], yticks=[])
    for edge in (10, 100):
        p.line([edge, edge], [-0.6, 0.6], role="ink-3", width=0.8, dashed=True)
    shown = [(x, ((i * 7919) % 101) / 101 - 0.5) for i, x in enumerate(got) if 0.8 <= x <= 380]
    p.points([x for x, _ in shown], [0.9 * j for _, j in shown], role="hyper", r=2.6)
    return yours.figure(words["title"], chart.figure([p], alt=text, width=460, height=150) + _key(text), alt=text)


EXERCISES["log-uniform"] = FunctionBox(
    name="log-uniform",
    picture=_lu_picture,
    function="log_uniform",
    start=_L_START,
    solution=_L,
    show=_shown_floats,
    compare=_same_floats,
    visible=[_L_VISIBLE],
    hidden=_L_HIDDEN,
    first=[_DOTS],
    mistakes=[
        Mistake("linear", wrong=_l("rng.uniform(low, high, n)", "logs"),
                en="`{call}` gives {got}: drawn evenly between `low` and `high` themselves. Then most draws of k land above 30, and the small values, where a factor 2 matters, hardly get a try. Draw the logs evenly, between `np.log10(low)` and `np.log10(high)`, and turn them back with `10 **`.",
                nl="`{call}` geeft {got}: gelijk verdeeld getrokken tussen `low` en `high` zelf. Dan komen de meeste trekkingen van k boven 30 uit, en krijgen de kleine waarden, waar een factor 2 ertoe doet, nauwelijks een poging. Trek de logs gelijk verdeeld, tussen `np.log10(low)` en `np.log10(high)`, en maak er met `10 **` weer getallen van."),
        Mistake("logs-back", wrong=_l(back="logs"),
                en="`{call}` gives {got}: the logs themselves, such as 1.9 for 79. Turn every log back into a number: `10 ** logs`.",
                nl="`{call}` geeft {got}: de logs zelf, zoals 1,9 voor 79. Maak van elke log weer een getal: `10 ** logs`."),
        Mistake("bases", wrong=_l("rng.uniform(np.log(low), np.log(high), n)"),
                en="`{call}` gives {got}: `np.log` has base e, and `10 **` turns only a log with base 10 back into its number. Use the same base on both sides: `np.log10` with `10 **`, or `np.log` with `np.exp`.",
                nl="`{call}` geeft {got}: `np.log` heeft grondtal e, en `10 **` maakt alleen van een log met grondtal 10 weer het getal. Gebruik aan beide kanten hetzelfde grondtal: `np.log10` met `10 **`, of `np.log` met `np.exp`."),
        Mistake("bases-exp", wrong=_l(back="np.exp(logs)"),
                en="`{call}` gives {got}: `np.exp` turns only a log with base e back into its number, and `np.log10` has base 10. Use the same base on both sides: `np.log10` with `10 **`, or `np.log` with `np.exp`.",
                nl="`{call}` geeft {got}: `np.exp` maakt alleen van een log met grondtal e weer het getal, en `np.log10` heeft grondtal 10. Gebruik aan beide kanten hetzelfde grondtal: `np.log10` met `10 **`, of `np.log` met `np.exp`."),
    ],
    texts=_SAME | {
        "empty": {"en": "Replace the three `...`: the log of `low` and of `high`, and the numbers back from the logs.",
                  "nl": "Vervang de drie `...`: de log van `low` en van `high`, en de getallen terug uit de logs."},
        "arguments": {"en": "The check calls `{call}`, with four values, and the `def` line of your function has a different number of names between its parentheses. Keep `(rng, low, high, n)`.",
                      "nl": "De check roept `{call}` aan, met vier waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(rng, low, high, n)`."},
        "wrong": {"en": "`{call}` gives {got}, and it should give {want}: n numbers between `low` and `high`, drawn evenly on a log scale, with the `rng` it gets.",
                  "nl": "`{call}` geeft {got}, en dat moet {want} zijn: n getallen tussen `low` en `high`, gelijk verdeeld getrokken op een logaritmische schaal, met de `rng` die ze krijgt."},
        "almost": {"en": "Right for k from 1 to 300. But `{call}` gives {got}, and it should give {want}. Does your function use its `low`, `high` and `n`?",
                   "nl": "Goed voor k van 1 tot 300. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `low`, `high` en `n` die ze krijgt?"},
        "correct": {"en": "Your function is right in all {n} cases. With it, a random search gives every factor 10 of k about as many tries.",
                    "nl": "Je functie klopt in alle {n} gevallen. Daarmee geeft een random search elke factor 10 van k ongeveer evenveel pogingen."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("grid", _G_START, "empty", None),
    ("grid", _g("    for k in ks:\n        ...\n"), "wrong", "dots-left"),
    ("grid", _G, "correct", None),
    ("grid", _g("    for p in ps:\n        for k in ks:\n            tries.append((k, p))\n"), "correct", None),
    ("grid", _g("    for k in ks:\n        for p in ps:\n            tries.append([k, p])\n"), "correct", None),
    ("grid", _g("    for k, p in zip(ks, ps):\n        tries.append((k, p))\n"), "wrong", "zip"),
    ("grid", _g("    for k in ks:\n        for p in ps:\n            pass\n        tries.append((k, p))\n"), "wrong", "outer-only"),
    ("grid", _g("    for k in ks:\n        tries.append(k)\n    for p in ps:\n        tries.append(p)\n"), "wrong", "flat"),
    ("grid", _g("    for k in ks:\n        for p in ps:\n            tries.append((p, k))\n"), "wrong", "swapped"),
    ("grid", _g("    for k in ks:\n        for p in ps:\n            tries.append(k, p)\n"), "error", "append-two"),
    ("log-uniform", _L_START, "empty", None),
    ("log-uniform", _l(back="..."), "wrong", "dots-left"),
    ("log-uniform", _L, "correct", None),
    ("log-uniform", _l("rng.uniform(np.log(low), np.log(high), n)", "np.exp(logs)"), "correct", None),
    ("log-uniform", _l("[rng.uniform(np.log10(low), np.log10(high)) for _ in range(n)]", "10 ** np.array(logs)"), "correct", None),
    ("log-uniform", _l("rng.uniform(low, high, n)", "logs"), "wrong", "linear"),
    ("log-uniform", _l(back="logs"), "wrong", "logs-back"),
    ("log-uniform", _l("rng.uniform(np.log(low), np.log(high), n)"), "wrong", "bases"),
    ("log-uniform", _l(back="np.exp(logs)"), "wrong", "bases-exp"),
]
