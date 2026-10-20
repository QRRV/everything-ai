# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson repro-own-compare: vs_paper, the comparison of the kit, which gives the mean gain of the seeds
with a bootstrap interval and says whether the paper's gain lies outside it, saved as own/compare.py; and the paper's
own numbers, with where in the paper they stand, saved as own/paper_numbers.json. The play block is the course's
main result for the dropout paper against the paper's own numbers, in points of test error and as a share of the
error without dropout, with one to three seeds."""
import importlib.util
import json
import math
import re
from dataclasses import dataclass, field

import numpy as np

from everything_ai import paths, projects
from everything_ai.check import Exercise, FunctionBox, Mistake, Result, _fill, _show_rounded


def _checks(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = _checks("repro-own-ablation", "everything_ai_checks_repro_own_ablation_for_compare")
F, B, P = A.F, A.B, A.P
SEEDS, number, short, whole, median, signed = B.SEEDS, B.number, B.short, B.whole, F.median, A.signed
BUDGET = F.COURSE_STEPS
N_BOOT = 10_000

# The dropout paper, its results on MNIST (permutation invariant, in the text and in Table 2): the best networks without
# dropout or unsupervised pretraining reach about 1.60% (Simard and others, 2003); "with dropout the error reduces to
# 1.35%", with another network.
PAPER = {"with": 1.35, "without": 1.60,
         "where": "The results on MNIST, in the text and in Table 2: the best networks without dropout, of Simard and others (2003), reach about 1.60% test error; with dropout, another network reaches 1.35%."}
PAPER_WHERE = {"en": PAPER["where"],
               "nl": "De resultaten op MNIST, in de tekst en in Table 2: de beste netwerken zonder dropout, van Simard en anderen (2003), halen ongeveer 1,60% testfout; met dropout haalt een ander netwerk 1,35%."}


def course_results() -> tuple[list[float], list[float]]:
    """The test error of seeds 0, 1 and 2 at BUDGET steps, with dropout and without: the course's main result."""
    return list(B.RUNS[BUDGET]["claim"]), list(B.RUNS[BUDGET]["other"])


def gains(claim: list[float], other: list[float], direction: str = "lower") -> list[float]:
    """Seed by seed, how far the claim's setting is ahead of the other, in points of the metric."""
    return [round((o - c) if direction == "lower" else (c - o), 4) for c, o in zip(claim, other)]


def shares(claim: list[float], other: list[float], direction: str = "lower") -> list[float]:
    """Seed by seed, the gain as a share of the other setting's metric, in %."""
    return [round(100 * g / o, 4) for g, o in zip(gains(claim, other, direction), other)]


def paper_gain(paper: dict, direction: str = "lower") -> float:
    return round((paper["without"] - paper["with"]) if direction == "lower" else (paper["with"] - paper["without"]), 4)


def paper_share(paper: dict, direction: str = "lower") -> float:
    return round(100 * paper_gain(paper, direction) / paper["without"], 2)


def interval(xs: list[float], n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float, float]:
    """The course's bootstrap: the mean of xs, and the 2.5th and 97.5th percentile of the means of n_boot resamples."""
    rng = np.random.default_rng(seed)
    xs = np.array(xs, dtype=float)
    means = [rng.choice(xs, size=len(xs)).mean() for _ in range(n_boot)]
    return float(xs.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare_picture(k: int, labels: dict) -> str:
    """Two panels side by side, for the first k seeds of the course's main result: the gain in points of test error,
    and as a share of the error without dropout. Each seed a dot, its interval a grey band with the mean as a ring,
    and the paper's own number a dashed line. Ink: a difference between runs, nothing a model learns."""
    from everything_ai import chart
    claim, other = course_results()
    panels = []
    for values, paper, (x0, x1), ticks, title in ((gains(claim, other)[:k], paper_gain(PAPER), (0, 1.5), [0, 0.5, 1, 1.5], labels["points"]),
                                                  (shares(claim, other)[:k], paper_share(PAPER), (0, 18), [0, 5, 10, 15], labels["share"])):
        mean, low, high = interval(values)
        p = chart.Panel((x0, x1), (-1, 1), title=title, xlabel=labels["x_points"] if x1 < 5 else labels["x_share"], ylabel="",
                        yticks=[], xticks=ticks)
        p.rect(low, -0.35, max(high, low + (x1 - x0) * 0.004), 0.35, "ink-3")
        p.line([paper, paper], [-0.9, 0.9], "ink", 1.6, dashed=True)
        right = paper < (x0 + x1) / 2   # the name beside the line, on the side with room
        p.text(paper + (x1 - x0) * (0.015 if right else -0.015), 0.62, labels["paper"], "ink", "start" if right else "end")
        p.points(values, [0] * len(values), "ink", r=3.5)
        p.points([mean], [0], "ink", r=8, hollow=True)
        panels.append(p)
    return chart.figure(panels, alt=labels["alt"], width=300, height=170)


# --- exercise 1: the comparison ------------------------------------------------------------------------------------------

_V_HEAD = ("import numpy as np\n\n\n"
           "def vs_paper(gains, paper_gain, n_boot, seed):\n"
           "    # your gains of every seed against the paper's one gain: the mean of your gains, the\n"
           "    # 0.025 and the 0.975 quantile of the means of n_boot resamples of them (drawn with\n"
           "    # replacement), and real: whether the paper's gain lies outside that interval\n")
_V_BODY = ("    rng = np.random.default_rng(seed)\n"
           "    means = []\n"
           "    for _ in range(n_boot):\n"
           "        means.append(np.mean(rng.choice(gains, size=len(gains))))\n"
           "    low, high = np.quantile(means, [0.025, 0.975])\n")
_V_TAIL = ("    real = paper_gain < low or paper_gain > high\n"
           "    return {\"mean\": float(np.mean(gains)), \"low\": float(low), \"high\": float(high), \"real\": bool(real)}\n")
_V_START = _V_HEAD + "    # 1. the interval: n_boot means of resamples of your gains, and two quantiles\n    ...\n" \
           "    # 2. real, and the result\n    ...\n"
V_CODE = _V_HEAD + _V_BODY + _V_TAIL
_V_WRONG = {
    "no-replacement": V_CODE.replace("np.mean(rng.choice(gains, size=len(gains)))", "np.mean(rng.permutation(gains))"),
    "ninety": V_CODE.replace("np.quantile(means, [0.025, 0.975])", "np.quantile(means, [0.05, 0.95])"),
    "inside": V_CODE.replace("real = paper_gain < low or paper_gain > high", "real = low <= paper_gain <= high"),
    "one-side": V_CODE.replace("real = paper_gain < low or paper_gain > high", "real = paper_gain < low"),
}
_COURSE_CASE = ([0.4, 0.7, 1.2], 0.25, N_BOOT, 0)
_VISIBLE = ([0.6, 1.1, 0.8, 0.5], 0.3, N_BOOT, 0)   # the lesson's example; not the worked example's own gains


def _show(v):
    """A value as a message shows it: floats rounded to 4 decimals, also those in the dictionary vs_paper gives."""
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k!r}: {_show_rounded(x, 4)}" for k, x in v.items()) + "}"
    return _show_rounded(v, 4)


def _close(got, want) -> bool:
    """The same mean and verdict, and ends within 0.02 of the course's: another way of drawing gives other means."""
    if not (isinstance(got, dict) and isinstance(want, dict)) or set(got) != set(want):
        return got == want
    try:
        return (abs(got["mean"] - want["mean"]) < 1e-6 and got["real"] == want["real"]
                and abs(got["low"] - want["low"]) <= 0.02 + 0.02 * abs(want["low"]) and abs(got["high"] - want["high"]) <= 0.02 + 0.02 * abs(want["high"]))
    except (TypeError, KeyError):
        return False


EXERCISES = {
    "vs-paper": FunctionBox(
        name="vs-paper",
        function="vs_paper",
        start=_V_START,
        solution=V_CODE,
        visible=[_VISIBLE],
        hidden=[_COURSE_CASE,
                ([3.1, 5.15, 8.82], 15.62, N_BOOT, 0),
                ([0.4, 0.7, 1.2], 0.9, N_BOOT, 0),
                ([0.2, -0.1, 0.4, 0.1, 0.3], 0.0, N_BOOT, 1),
                ([1.5], 1.0, 1000, 0)],
        compare=_close,
        show=_show,
        save_as=("own", "compare"),
        first=[B._DOTS],
        mistakes=[
            Mistake("no-replacement", wrong=_V_WRONG["no-replacement"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: your interval is a single point, the mean of your gains. A resample that takes every gain once, as `rng.permutation` or `replace=False` do, only puts them in another order, and the mean of the same numbers never changes. A resample draws with replacement, so that a seed can come twice and another not at all: `rng.choice(gains, size=len(gains))`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je interval is één punt, het gemiddelde van je winsten. Een resample die elke winst één keer neemt, zoals `rng.permutation` of `replace=False`, zet ze alleen in een andere volgorde, en het gemiddelde van dezelfde getallen verandert nooit. Een resample trekt met teruglegging, zodat een seed twee keer kan komen en een andere helemaal niet: `rng.choice(gains, size=len(gains))`."),
            Mistake("ninety", wrong=_V_WRONG["ninety"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: the quantiles 0.05 and 0.95 hold 90% of the means, not 95%. Take 0.025 and 0.975: 2.5% below, 2.5% above.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de kwantielen 0,05 en 0,95 omsluiten 90% van de gemiddelden, niet 95%. Neem 0,025 en 0,975: 2,5% eronder, 2,5% erboven."),
            Mistake("inside", wrong=_V_WRONG["inside"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: `real` is turned around. A gain of the paper inside your interval is one your seeds could have given: that difference is not shown to be real. Real is outside: `paper_gain < low or paper_gain > high`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: `real` is omgedraaid. Een winst van de paper binnen je interval is er een die je seeds hadden kunnen geven: dat verschil is niet aangetoond. Echt is erbuiten: `paper_gain < low or paper_gain > high`."),
            Mistake("one-side", wrong=_V_WRONG["one-side"], state="almost",
                    en="`{call}` gives `{got}`, and it should give `{want}`: your `real` looks only below the interval. A paper whose gain is larger than all of yours differs just as much: look at both ends.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je `real` kijkt alleen onder het interval. Een paper waarvan de winst groter is dan al de jouwe, verschilt net zo goed: kijk naar beide kanten."),
        ],
        texts={
            **F._KEEP,
            "empty": {"en": "Replace both `...`: first the interval, `n_boot` means of resamples of your gains and their 0.025 and 0.975 quantile; then `real`, and the dictionary.",
                      "nl": "Vervang beide `...`: eerst het interval, `n_boot` gemiddelden van resamples van je winsten en hun 0,025- en 0,975-kwantiel; dan `real`, en de dictionary."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does it draw `n_boot` resamples of the gains with replacement, take the 0.025 and the 0.975 quantile of their means, and call the difference real when the paper's gain lies outside?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Trekt hij `n_boot` resamples van de winsten met teruglegging, neemt hij het 0,025- en het 0,975-kwantiel van hun gemiddelden, en noemt hij het verschil echt als de winst van de paper erbuiten ligt?"},
            "almost": {"en": "Your `vs_paper` is right for the course's gains. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for a paper whose gain lies inside your interval, for one whose gain is larger than yours, for five seeds, and for one?",
                       "nl": "Je `vs_paper` klopt voor de winsten van de course. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor een paper waarvan de winst binnen je interval ligt, voor een waarvan de winst groter is dan de jouwe, voor vijf seeds, en voor één?"},
            "kept": {"en": "Your `vs_paper` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `vs_paper` klopt. Je hebt `{saved}` zelf veranderd, dus laat de check dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
            "correct": {"en": "Your `vs_paper` gives the mean gain of your seeds with its interval, and says whether the paper's gain lies outside it. It is in your kit now, as `{saved}`.",
                        "nl": "Je `vs_paper` geeft de gemiddelde winst van je seeds met zijn interval, en zegt of de winst van de paper erbuiten ligt. Hij zit nu in je kit, als `{saved}`."},
        },
    ),
}


# --- exercise 2: the paper's numbers -----------------------------------------------------------------------------------

def numbers_path():
    return projects.folder() / "own" / "paper_numbers.json"


def saved_numbers() -> dict | None:
    try:
        n = json.loads(numbers_path().read_text(encoding="utf-8"))
        return n if isinstance(n, dict) and all(_finite(n.get(k)) for k in ("with", "without")) else None
    except (OSError, ValueError):
        return None


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


@dataclass
class NumbersReview(Exercise):
    """The paper's own numbers, from the lesson's form: {"with": the metric with the method, "without": without it,
    "where": where in the paper}. Once both are numbers and the place is named, they are saved in the learner's kit
    as paper_numbers.json."""
    start: dict = field(default_factory=dict)
    solution: dict = field(default_factory=dict)

    def check(self, value) -> Result:
        if not isinstance(value, dict):
            return Result("error", self.message("missing"))
        a, b, where = value.get("with"), value.get("without"), (value.get("where") or "").strip()
        if not a and not b and not where:
            return Result("empty", self.message("empty"))
        if not _finite(a) or not _finite(b) or a == b == 0:   # both still 0, as the form starts
            return self._mistake("numbers")
        if a == b:
            return self._mistake("same")
        if len(re.findall(r"\w+", where)) < 4:
            return self._mistake("where")
        path = numbers_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"with": a, "without": b, "where": where}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        d = {lang: signed(round(a - b, 4), lang) for lang in ("en", "nl")}
        return Result("correct", _fill(self.texts["correct"], saved=projects.shown(path), d=d))

    def _mistake(self, key: str, **values) -> Result:
        m = next(m for m in self.mistakes if m.key == key)
        return Result(m.state, _fill(m.text, **values), mistake=key)


EXERCISES["paper-numbers"] = NumbersReview(
    name="paper-numbers",
    start={"with": 0, "without": 0, "where": ""},
    solution={"with": PAPER["with"], "without": PAPER["without"], "where": PAPER["where"]},
    mistakes=[
        Mistake("numbers",
                en="Fill in both numbers of the paper, in the unit of your metric: the one with the method of the claim, and the one without.",
                nl="Vul beide getallen van de paper in, in de eenheid van je metric: dat met de methode van de claim, en dat zonder."),
        Mistake("same",
                en="Both numbers are the same, so the paper would show no gain at all. Look again: is one of them from another row of the table?",
                nl="Beide getallen zijn gelijk, dus zou de paper helemaal geen winst laten zien. Kijk nog eens: komt een ervan uit een andere rij van de tabel?"),
        Mistake("where",
                en="Say where in the paper the numbers stand, in a sentence: the table, figure or section, and which networks they belong to. A reader of your report checks them there, and so do you, if they surprise you.",
                nl="Zeg in een zin waar in de paper de getallen staan: de tabel, figuur of sectie, en bij welke netwerken ze horen. Een lezer van je verslag controleert ze daar, en jij ook, als ze je verrassen."),
    ],
    texts={
        "missing": {"en": "I cannot see the form. Run the cell with the form again.",
                    "nl": "Ik zie het formulier niet. Voer de cel met het formulier opnieuw uit."},
        "empty": {"en": "Fill in the paper's number with the method and without it, and in a sentence where in the paper they stand.",
                  "nl": "Vul het getal van de paper met de methode en zonder in, en in een zin waar in de paper ze staan."},
        "correct": {"en": "The paper's numbers are in your kit now, `{saved}`: with the method minus without it is {d}.",
                    "nl": "De getallen van de paper zitten nu in je kit, `{saved}`: met de methode min zonder is {d}."},
    },
)

REQUIRED = ["vs-paper", "paper-numbers", "why-differs"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("vs-paper", _V_START, "empty", None),
    ("vs-paper", V_CODE, "correct", None),
    ("vs-paper", V_CODE.replace("np.mean(rng.choice(gains, size=len(gains)))", "rng.choice(gains, size=len(gains)).mean()"), "correct", None),
] + [("vs-paper", code, "almost" if key == "one-side" else "wrong", key) for key, code in _V_WRONG.items()] + [
    ("paper-numbers", None, "error", None),
    ("paper-numbers", {"with": 0, "without": 0, "where": ""}, "empty", None),
    ("paper-numbers", {"with": 1.35, "without": None, "where": PAPER["where"]}, "wrong", "numbers"),
    ("paper-numbers", {"with": 0, "without": 0, "where": PAPER["where"]}, "wrong", "numbers"),
    ("paper-numbers", {"with": 1.35, "without": 1.35, "where": PAPER["where"]}, "wrong", "same"),
    ("paper-numbers", {"with": 1.35, "without": 1.60, "where": "table 2"}, "wrong", "where"),
]


# --- the demo -------------------------------------------------------------------------------------------------------------

def my_comparison() -> dict | str | None:
    """The learner's main result at their budget against their paper's numbers, read with their own/compare.py, in
    points and as a share of the other setting: {"gains", "shares", "paper_gain", "paper_share", "points", "share"};
    the shares None when the metric without the method is not above 0, in a run or in the paper, as a log-likelihood:
    then a share means nothing. None when a part is missing, "error" when their vs_paper stops."""
    import importlib
    plan, budget, numbers, lines = P.saved_plan(), F.saved_budget(), saved_numbers(), F.my_log()
    if not plan or not budget or not numbers or not lines:
        return None
    a, b = P._settings(plan)
    res = F.course_results_at(lines, budget["steps"], plan)
    n = int(plan["seeds"])
    if len(res[a]) < n or len(res[b]) < n:
        return None
    if not (projects.folder() / "own" / "compare.py").is_file():
        return None
    direction = plan.get("direction", "lower")
    claim, other = res[a][:n], res[b][:n]
    share = all(o > 0 for o in other) and numbers["without"] > 0
    out = {"gains": gains(claim, other, direction), "shares": shares(claim, other, direction) if share else None,
           "paper_gain": paper_gain(numbers, direction), "paper_share": paper_share(numbers, direction) if share else None}
    projects.on_path()
    projects.forget()
    try:
        fn = importlib.import_module("own.compare").vs_paper
        out["points"] = fn(out["gains"], out["paper_gain"], N_BOOT, 0)
        out["share"] = fn(out["shares"], out["paper_share"], N_BOOT, 0) if share else None
        if not all(isinstance(out[k], dict) and {"mean", "low", "high", "real"} <= set(out[k]) for k in (("points", "share") if share else ("points",))):
            return "error"
    except Exception:
        return "error"
    finally:
        projects.forget()
    return out
