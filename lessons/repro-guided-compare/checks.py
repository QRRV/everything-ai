# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson repro-guided-compare: compare, which pairs the runs of two settings by seed and gives how many
factors of 10 later validation reaches 99% in one than in the other, with a paired bootstrap interval, saved as
repro/compare.py; and the sort of the details of the setup into what the paper says, what it does not say, and what
the reproduction does differently on purpose. The course's runs over five seeds are in data/course-seeds.jsonl; the
demo compares the learner's own runs with the course's, seed by seed."""
import functools
import importlib.util
import json
import math

from everything_ai import paths, projects
from everything_ai.check import ClassBox, Mistake, Sort, _show_rounded, run_code


def _checks(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = _checks("repro-guided-ablation", "everything_ai_checks_repro_guided_ablation_for_compare")
G = A.G
PAPER, FRACTION, STEPS, THRESHOLD, DELAY, number, read = A.PAPER, A.FRACTION, A.STEPS, A.THRESHOLD, A.DELAY, A.number, A.read
# The kit so far: where a module is missing or broken, the course's version stands in.
BUILDS_ON = A.BUILDS_ON + (("repro", "ablation", "repro-guided-ablation", "run-grid"),)
projects.builds_on(*BUILDS_ON)

SEEDS = [0, 1, 2, 3, 4]
WIDE = 0.9                    # the fraction where validation reaches 99% with and without weight decay
RUNS_FILE = paths.ROOT / "data" / "course-seeds.jsonl"
N_BOOT = 10_000


@functools.cache
def course_runs() -> list[dict]:
    """The course's runs over SEEDS: weight decay 1 and 0 at WIDE, and weight decay 1 at FRACTION; run_seed of
    repro-guided-figure, STEPS steps each, with the weight decay and the fraction they use."""
    return [json.loads(line) for line in RUNS_FILE.read_text(encoding="utf-8").splitlines()]


def runs_of(weight_decay: float, fraction: float) -> list[dict]:
    return sorted((r for r in course_runs() if r["weight_decay"] == weight_decay and r["fraction"] == fraction), key=lambda r: r["seed"])


@functools.cache
def _compare():
    return run_code(_C).names["compare"]


def compare(runs_a: list, runs_b: list, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """The course's compare: the solution of the exercise."""
    return _compare()(runs_a, runs_b, n_boot, seed)


def factors(n: int) -> list[float | None]:
    """For the first n seeds: how many times as late validation reaches 99% at WIDE without weight decay as with it;
    None for a seed where it does not reach 99% within STEPS in one of the two."""
    off, on = runs_of(0.0, WIDE)[:n], runs_of(1.0, WIDE)[:n]
    v = [(read(a)["v"], read(b)["v"]) for a, b in zip(off, on)]
    return [a / b if a is not None and b is not None else None for a, b in v]


def at_least(seed: int) -> float:
    """For a seed where validation does not reach 99% within STEPS without weight decay: the least its factor can be,
    STEPS over the step at which it does with weight decay."""
    return STEPS / read(runs_of(1.0, WIDE)[seed])["v"]


def factors_picture(n: int, labels: dict) -> str:
    """For each of the first n seeds, a dot: how many times as late validation reaches 99% without weight decay as with
    it, on a log axis, or for a seed where it does not get there within STEPS an arrow up from the least it can be, with
    labels["none"]; a thin line at 1, where both are as fast; and the mean of the pairs with its bootstrap interval as
    a grey wash, from compare. Ink: a difference between two runs, not something a model learns."""
    from everything_ai import chart
    r = compare(runs_of(0.0, WIDE)[:n], runs_of(1.0, WIDE)[:n])
    low, high, mean = 10 ** r["low"], 10 ** r["high"], 10 ** r["mean"]
    p = chart.Panel((-0.6, len(SEEDS) - 0.4), (0.5, 40), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"], ylog=True,
                    xticks=list(range(len(SEEDS))), yticks=[1, 2, 5, 10, 20])
    p.rect(-0.6, low, len(SEEDS) - 0.4, high, "data")   # the interval: a grey wash, as data
    p.line([-0.6, len(SEEDS) - 0.4], [1, 1], "ink-3", 1.0)
    p.line([-0.6, len(SEEDS) - 0.4], [mean, mean], "ink", 1.6, dashed=True)
    f = factors(n)
    p.points([s for s in range(n) if f[s] is not None], [x for x in f if x is not None], "ink", r=5)
    for s in (s for s in range(n) if f[s] is None):
        p.arrow(s, at_least(s), s, 33, "ink", 1.6)
        p.points([s], [at_least(s)], "ink", r=5, hollow=True)
        p.text(s - 0.14, 22, labels["none"], "ink-2", "end")
    p.text(len(SEEDS) - 0.5, low / 1.22, labels["band"], "ink", "end")   # under the band at the right, where no dot comes
    return chart.figure([p], alt=labels["alt"], width=460, height=300)


def _close(got, want) -> bool:
    """Numbers, also inside lists and dictionaries, equal to 4 decimals; anything else equal."""
    if isinstance(got, dict) and isinstance(want, dict):
        return got.keys() == want.keys() and all(_close(got[k], want[k]) for k in want)
    if isinstance(got, (list, tuple)) and isinstance(want, (list, tuple)):
        return len(got) == len(want) and all(_close(g, w) for g, w in zip(got, want))
    if isinstance(got, bool) or isinstance(want, bool):
        return got == want
    if isinstance(got, (int, float)) and isinstance(want, (int, float)):
        return abs(got - want) <= 1e-4 + 1e-4 * abs(want)
    return got == want


_DOTS = G._DOTS
_KEEP = G._KEEP

# --- exercise 1: compare -----------------------------------------------------------------------------------------------

_C_HEAD = ("import math\n\nimport numpy as np\n\nfrom repro.plan import THRESHOLD, first_step\n\n\n"
           "def paired_bootstrap(a, b, n_boot, seed):\n"
           "    # as in Is the difference real?: the mean difference of the pairs, and a 95% bootstrap interval around it\n"
           "    diffs = np.array(a) - np.array(b)\n    rng = np.random.default_rng(seed)\n"
           "    means = [np.mean(rng.choice(diffs, size=len(diffs))) for _ in range(n_boot)]\n"
           "    low, high = np.quantile(means, [0.025, 0.975])\n    return [float(np.mean(diffs)), float(low), float(high)]\n\n\n"
           "def compare(runs_a, runs_b, n_boot, seed):\n"
           "    # pairs each run of runs_a with the run of runs_b with the same seed; for each pair, log10 of the step at which\n"
           "    # validation reaches THRESHOLD in a, and in b; a pair where either does not reach it is left out, and counted\n"
           "    by_seed = {log[\"seed\"]: log for log in runs_b}\n    a, b, left_out = [], [], 0\n    for log in runs_a:\n")
_C_BODY = ("        if log[\"seed\"] in by_seed:\n            other = by_seed[log[\"seed\"]]\n"
           "            va = first_step(log[\"step\"], log[\"val\"], THRESHOLD)\n"
           "            vb = first_step(other[\"step\"], other[\"val\"], THRESHOLD)\n"
           "            if va is None or vb is None:\n                left_out = left_out + 1\n"
           "            else:\n                a.append(math.log10(va))\n                b.append(math.log10(vb))\n")
_C_TAIL = ("    mean, low, high = paired_bootstrap(a, b, n_boot, seed)\n"
           "    return {\"pairs\": len(a), \"left_out\": left_out, \"mean\": mean, \"low\": low, \"high\": high}\n")
_C_START = (_C_HEAD + "        # the run of runs_b with the same seed, if there is one; the steps at which validation reaches\n"
            "        # THRESHOLD in both; left out when either is None, else their log10 added to a and b\n        ...\n" + _C_TAIL)
_C = _C_HEAD + _C_BODY + _C_TAIL
_C_NO_LOG = _C.replace("a.append(math.log10(va))", "a.append(va)").replace("b.append(math.log10(vb))", "b.append(vb)")
_C_TRAIN = _C.replace("first_step(log[\"step\"], log[\"val\"], THRESHOLD)", "first_step(log[\"step\"], log[\"train\"], THRESHOLD)") \
             .replace("first_step(other[\"step\"], other[\"val\"], THRESHOLD)", "first_step(other[\"step\"], other[\"train\"], THRESHOLD)")
_C_SWAP = _C.replace("a.append(math.log10(va))", "a.append(math.log10(vb))").replace("b.append(math.log10(vb))", "b.append(math.log10(va))")
_C_NO_COUNT = _C.replace("                left_out = left_out + 1\n", "                pass\n")

_C_RUN = ("def run(seed, v, t=100):   # a run whose training reaches 99% at step t, and validation at step v, or never when v is None\n"
          "    steps = [10, 100, 1000, 10000]\n"
          "    return {\"seed\": seed, \"step\": steps, \"train\": [1.0 if s >= t else 0.5 for s in steps],\n"
          "            \"val\": [1.0 if v is not None and s >= v else 0.3 for s in steps]}\n\n\n")
_C_SHOW = "\nr"   # {got} and {want} show its numbers to 4 decimals (digits=4)
_C_SCENARIOS = [
    # the lesson's example: three seeds, a always 10 times as late as b but once; training is not always as fast
    _C_RUN + "a = [run(0, 1000), run(1, 10000), run(2, 1000)]\nb = [run(0, 100, 10), run(1, 1000), run(2, 1000)]\n"
             "r = compare(a, b, 1000, 0)" + _C_SHOW,
    # seeds in another order in b, and a seed that only a has
    _C_RUN + "a = [run(0, 1000), run(1, 100), run(5, 100)]\nb = [run(1, 10), run(0, 1000)]\n"
             "r = compare(a, b, 1000, 0)" + _C_SHOW,
    # a pair where b never reaches 99%: left out, and counted
    _C_RUN + "a = [run(0, 1000), run(1, 1000), run(2, 10000)]\nb = [run(0, 100), run(1, None), run(2, 1000)]\n"
             "r = compare(a, b, 500, 3)" + _C_SHOW,
]
_C_EN, _C_NL = "Your `compare` gives another result than the course's", "Je `compare` geeft een ander resultaat dan die van de course"


def _show(v) -> str:
    if isinstance(v, dict):
        return "{" + ", ".join(f"{json.dumps(k)}: {_show_rounded(x, 4)}" for k, x in v.items()) + "}"
    return _show_rounded(v, 4)


# --- exercise 2: the details of the setup ----------------------------------------------------------------------------------

_DETAILS = {   # mixed, so that the order of the rows does not give the groups away
    "budget": {"en": "12,000 steps per run instead of 100,000", "nl": "12.000 stappen per run in plaats van 100.000"},
    "tokens": {"en": "Each equation is five tokens: a, the operation, b, = and the answer", "nl": "Elke vergelijking is vijf tokens: a, de bewerking, b, = en het antwoord"},
    "decay-on": {"en": "Which parameters weight decay pulls toward 0: also the embeddings?", "nl": "Welke parameters weight decay naar 0 trekt: ook de embeddings?"},
    "median": {"en": "Figure 1 (center) gives the median over seeds", "nl": "Figuur 1 (midden) geeft de mediaan over seeds"},
    "task": {"en": "Weight decay tested on addition modulo 97, not on the product in S₅", "nl": "Weight decay getest op optellen modulo 97, niet op het product in S₅"},
    "answer": {"en": "The loss and the accuracy count only the answer", "nl": "De loss en de accuracy tellen alleen het antwoord"},
    "how-often": {"en": "How often validation accuracy was measured during a run", "nl": "Hoe vaak de validation accuracy tijdens een run gemeten werd"},
    "modern": {"en": "RMSNorm, SwiGLU and RoPE in the model", "nl": "RMSNorm, SwiGLU en RoPE in het model"},
    "split": {"en": "The training data is a random part of all equations", "nl": "De training data is een willekeurig deel van alle vergelijkingen"},
}
_GROUPS = {
    "paper": {"en": "The paper says it", "nl": "De paper zegt het"},
    "unsaid": {"en": "The paper does not say it: look in its code, or ask the authors", "nl": "De paper zegt het niet: kijk in zijn code, of vraag het de auteurs"},
    "ours": {"en": "Your reproduction does it differently, on purpose", "nl": "Je reproductie doet het bewust anders"},
}
_SOLUTION = {"budget": "ours", "tokens": "paper", "decay-on": "unsaid", "median": "paper", "task": "ours", "answer": "paper",
             "how-often": "unsaid", "modern": "ours", "split": "paper"}

EXERCISES = {
    "compare": ClassBox(
        name="compare",
        start=_C_START,
        solution=_C,
        scenarios=_C_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("repro", "compare"),
        first=[_DOTS],
        mistakes=[
            Mistake("no-log", wrong=_C_NO_LOG,
                    en=_C_EN + ": you compare the steps themselves, so a pair at 10,000 and 1,000 weighs a thousand times as much as one at 10 and 1. Compare how many times as late: the difference of `math.log10` of each.",
                    nl=_C_NL + ": je vergelijkt de stappen zelf, dus weegt een paar bij 10.000 en 1.000 duizend keer zo zwaar als een paar bij 10 en 1. Vergelijk hoeveel keer zo laat: het verschil van `math.log10` van elk."),
            Mistake("train", wrong=_C_TRAIN,
                    en=_C_EN + ": you read the moment at which training reaches 99%. The claim is about validation: `first_step(log[\"step\"], log[\"val\"], THRESHOLD)`.",
                    nl=_C_NL + ": je leest het moment waarop training 99% haalt. De claim gaat over validatie: `first_step(log[\"step\"], log[\"val\"], THRESHOLD)`."),
            Mistake("swap", wrong=_C_SWAP,
                    en=_C_EN + ": the sign is turned around. `a` holds the runs of `runs_a`, `b` those of `runs_b`, so that a positive mean means: later in a.",
                    nl=_C_NL + ": het teken is omgedraaid. `a` bevat de runs van `runs_a`, `b` die van `runs_b`, zodat een positief gemiddelde betekent: later in a."),
            Mistake("no-count", wrong=_C_NO_COUNT, state="almost",
                    en="Your `compare` gives the course's result for the first example, where every run reaches 99%. But a pair that is left out is not counted, so the result does not say that a seed is missing from it. Add 1 to `left_out` for it.",
                    nl="Je `compare` geeft het resultaat van de course voor het eerste voorbeeld, waarin elke run 99% haalt. Maar een paar dat wegvalt, wordt niet geteld, dus zegt het resultaat niet dat er een seed in ontbreekt. Tel er 1 bij `left_out` voor op."),
        ],
        texts={
            **_KEEP,
            "empty": {"en": "Replace the `...`: for each run of `runs_a` whose seed is in `by_seed`, the steps at which validation reaches `THRESHOLD` in both runs; a pair with a `None` is counted in `left_out`, the others go into `a` and `b` as `math.log10` of their step.",
                      "nl": "Vervang de `...`: voor elke run van `runs_a` waarvan de seed in `by_seed` staat, de stappen waarop validatie in beide runs `THRESHOLD` haalt; een paar met een `None` tel je in `left_out`, de andere gaan als `math.log10` van hun stap in `a` en `b`."},
            "wrong": {"en": _C_EN + ". Does each run of `runs_a` find the run with its seed in `by_seed`, read the moment of validation in both with `first_step`, and add `math.log10` of each to `a` and `b`, or count the pair in `left_out`?",
                      "nl": _C_NL + ". Vindt elke run van `runs_a` de run met zijn seed in `by_seed`, leest hij in beide het moment van validatie af met `first_step`, en voegt hij `math.log10` van elk toe aan `a` en `b`, of telt hij het paar in `left_out`?"},
            "almost": {"en": "Your `compare` gives the course's result for the first example. But with these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`. Does it pair by seed, not by place in the list, skip a seed that only `runs_a` has, and count a pair where either run does not reach 99%?",
                       "nl": "Je `compare` geeft het resultaat van de course voor het eerste voorbeeld. Maar met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn. Koppelt hij op seed, niet op plek in de lijst, slaat hij een seed over die alleen `runs_a` heeft, en telt hij een paar waarin een van beide runs geen 99% haalt?"},
            "correct": {"en": "Your `compare` pairs the runs by seed and says how many factors of 10 later validation reaches 99% in one setting than in the other, with an interval. It is now in your project, as `work/projects/repro/compare.py`.",
                        "nl": "Je `compare` koppelt de runs op seed en zegt hoeveel factoren 10 later validatie in de ene instelling 99% haalt dan in de andere, met een interval. Hij staat nu in je project, als `work/projects/repro/compare.py`."},
            "kept": {"en": "Your `compare` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `compare` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "details": Sort(
        name="details",
        items=_DETAILS,
        groups=_GROUPS,
        solution=_SOLUTION,
        nudge={
            "tokens": {"en": "Look at Appendix A.1.1: how does it write an equation?", "nl": "Kijk in bijlage A.1.1: hoe schrijft die een vergelijking?"},
            "answer": {"en": "Look at the first sentence of Appendix A.1.2.", "nl": "Kijk naar de eerste zin van bijlage A.1.2."},
            "split": {"en": "Look at the end of Appendix A.1.1.", "nl": "Kijk naar het eind van bijlage A.1.1."},
            "median": {"en": "Look at section 3.1.1, after its first paragraph.", "nl": "Kijk in paragraaf 3.1.1, na de eerste alinea."},
            "decay-on": {"en": "A.1.2 gives the weight decay, 1. Does it say which parameters it applies to?", "nl": "A.1.2 geeft de weight decay, 1. Zegt hij ook op welke parameters die werkt?"},
            "how-often": {"en": "The paper says \"first reaches 99%\". Does it say how often it looked?", "nl": "De paper zegt \"voor het eerst 99%\". Zegt hij ook hoe vaak hij keek?"},
            "modern": {"en": "The paper trains \"a standard decoder-only transformer\". Which model did your kit build?", "nl": "De paper traint \"een standaard decoder-only transformer\". Welk model bouwde je kit?"},
            "budget": {"en": "What is the paper's budget, and what is yours?", "nl": "Wat is het budget van de paper, en wat is het jouwe?"},
            "task": {"en": "Which task does the caption of Figure 2 (left) name, and which does your kit train?", "nl": "Welke taak noemt het bijschrift van Figuur 2 (links), en welke traint je kit?"},
        },
        texts={
            "correct": {"en": "Each detail is where it belongs. What the paper does not say is where an undocumented difference can hide; what your reproduction does differently are differences you know, and your report names them.",
                        "nl": "Elk detail staat waar het hoort. Wat de paper niet zegt, zijn de plekken waar een ongedocumenteerd verschil kan zitten; wat je reproductie bewust anders doet, zijn verschillen die je kent, en je verslag noemt ze."},
            "missing": {"en": "I cannot see your choices. Run the cell with the details again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de details opnieuw uit."},
            "empty": {"en": "Put each detail in a group: what the paper says, what it does not say, and what your reproduction does differently on purpose.",
                      "nl": "Zet elk detail in een groep: wat de paper zegt, wat hij niet zegt, en wat je reproductie bewust anders doet."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "\"{item}\" is not in its group.", "nl": "\"{item}\" staat niet in zijn groep."},
        },
    ),
}

# The lesson is done when these are correct. "agrees" is the choice question in the lesson.
REQUIRED = ["compare", "details", "agrees"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("compare", _C_START, "empty", None),
    ("compare", _C, "correct", None),
    ("compare", _C_NO_LOG, "wrong", "no-log"),
    ("compare", _C_TRAIN, "wrong", "train"),
    ("compare", _C_SWAP, "wrong", "swap"),
    ("compare", _C_NO_COUNT, "almost", "no-count"),
]


def demo() -> dict | str | None:
    """The learner's compare on their own runs of repro-guided-figure against the course's at FRACTION over SEEDS,
    seed by seed: does their reproduction match the course's reference? None when repro/compare.py is not in the project yet; "compare" when it
    stops with an error; "runs" when there are no runs of the learner to compare yet; otherwise compare's result, the
    seeds of the pairs, and "stand_ins"."""
    folder = projects.folder() / "repro"
    if not (folder / "compare.py").is_file():
        return None
    mine = G.my_runs()
    if not mine:
        return "runs"
    projects.on_path()
    projects.forget()
    stand = projects.use_course_versions()
    try:
        import importlib
        try:
            compare_ = importlib.import_module("repro.compare").compare
            r = compare_(mine, runs_of(1.0, FRACTION), N_BOOT, 0)
        except Exception:
            return "compare"
        return {**r, "seeds": sorted({x["seed"] for x in mine} & set(SEEDS)),
                "stand_ins": [f"{package}/{module}.py" for package, module, _, _ in stand if (package, module) == ("repro", "plan")]}
    finally:
        projects.forget()
