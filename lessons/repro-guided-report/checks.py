# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson repro-guided-report: result_rows, one row of a results table for every run of a log, also for
a run that does not reach 99% within its budget, saved as repro/results.py; and report, the template of the
reproduction report: its five sections in a fixed order, with the ones not written yet marked, saved as
repro/report.py. The demo writes the learner's report from their kit and their runs, or the course's where theirs are
not there yet, and saves it as work/projects/repro/report.md."""
import importlib
import importlib.util
import math

from everything_ai import paths, projects
from everything_ai.check import ClassBox, FunctionBox, Mistake, _show_rounded, run_code


def _checks(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _checks("repro-guided-compare", "everything_ai_checks_repro_guided_compare_for_report")
A, G = C.A, C.G
PAPER, FRACTION, STEPS, THRESHOLD, DELAY, SEEDS, WIDE = C.PAPER, C.FRACTION, C.STEPS, C.THRESHOLD, C.DELAY, C.SEEDS, C.WIDE
number, read, factors, runs_of, compare = C.number, C.read, C.factors, C.runs_of, C.compare
# The kit so far: where a module is missing or broken, the course's version stands in.
BUILDS_ON = C.BUILDS_ON + (("repro", "compare", "repro-guided-compare", "compare"),)
projects.builds_on(*BUILDS_ON)

REPORT = "report.md"   # the learner's report, in work/projects/repro/
SECTIONS = ["Claim", "Setup", "Results", "Differences", "Not checked"]

# The claim of the course's report, as the box in the demo starts: the claim of repro-guided-claim, and the one about
# weight decay that the ablation and the comparison test. In English, as the report is.
CLAIM = ("Power and others (2022), Figure 1 (left): \"a dramatic example of generalization far after overfitting on an "
         "algorithmic dataset\", on division modulo 97 with 50% of the data. And section 3.3 with Figure 2 (left): weight "
         "decay \"improves generalization the most\" of the interventions tried, measured on the product in S5 as how much "
         "data it takes. Tested here on addition modulo 97. "
         f"Rule of the plan, written before any run: training and validation count as reached at {round(100 * THRESHOLD)}% "
         f"accuracy, and a run is grokked when validation gets there at least {DELAY} times as late as training.")
# The course's own text for "Not checked": the model answer of exercise 3, as a report would put it.
NOT_CHECKED = ("The paper's own setup: the product in S5 (its weight decay claim) and division modulo 97 at 50% (its Figure 1, left), "
               "with its budgets: 100,000 steps for most runs, 500,000 for the median over seven seeds of Figure 1 (center) "
               "and 1,000,000 for Figure 1 (left). The paper's measure for "
               "weight decay, how much data it takes, and its other interventions. Whether the choices the paper does not "
               "give are the ones its authors made: that needs its code, or a question to them. And where seed 4 at 90% "
               "without weight decay gets to 99%, beyond 12,000 steps.\n\n"
               "Verdict: partly reproduced. On addition modulo 97, all three seeds grokked at 30% of the table, and at 90% "
               "validation reached 99% later without weight decay than with it in every pair; the paper's own tasks, measure "
               "and budgets were not checked.")


def every() -> float:
    """How many times as late validation reaches 99% at WIDE without weight decay, over all pairs of the comparison."""
    return 10 ** compare(runs_of(0.0, WIDE), runs_of(1.0, WIDE))["mean"]


def best_first(k: int) -> list[int]:
    """The k seeds of the comparison at WIDE whose factor suits the claim best, the largest first; seeds without a
    factor have none to choose."""
    f = factors(len(SEEDS))
    return sorted((s for s in range(len(SEEDS)) if f[s] is not None), key=lambda s: -f[s])[:k]


def reported(k: int) -> float:
    """How many times as late validation reaches 99% without weight decay, as a report of only those k seeds would
    give it: 10 to the mean of their log factors."""
    f = factors(len(SEEDS))
    return 10 ** (sum(math.log10(f[s]) for s in best_first(k)) / k)


def chosen_picture(k: int, labels: dict) -> str:
    """The comparison at WIDE, seed by seed, as in repro-guided-compare: a filled dot for each seed a report of the k
    best-suited seeds gives, an open one for each it leaves out, and seed 4's arrow up from the least it can be; the
    dashed line is the number that report gives, the thin dashed one the mean of all pairs. Ink: differences between
    runs."""
    from everything_ai import chart
    f, chosen, n = factors(len(SEEDS)), best_first(k), len(SEEDS)
    every_ = every()
    p = chart.Panel((-0.6, n - 0.4), (0.5, 40), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"], ylog=True,
                    xticks=list(range(n)), yticks=[1, 2, 5, 10, 20])
    p.line([-0.6, n - 0.4], [1, 1], "ink-3", 1.0)
    p.line([-0.6, n - 0.4], [every_, every_], "ink-3", 1.2, dashed=True)
    p.line([-0.6, n - 0.4], [reported(k), reported(k)], "ink", 1.8, dashed=True)
    p.points([s for s in chosen], [f[s] for s in chosen], "ink", r=5)
    rest = [s for s in range(n) if f[s] is not None and s not in chosen]
    p.points(rest, [f[s] for s in rest], "ink", r=5, hollow=True)
    for s in (s for s in range(n) if f[s] is None):
        p.arrow(s, C.at_least(s), s, 33, "ink", 1.6)
        p.points([s], [C.at_least(s)], "ink", r=5, hollow=True)
        p.text(s - 0.14, 22, labels["none"], "ink-2", "end")
    p.text(-0.5, reported(k) * 1.2, labels["reported"], "ink", "start")
    p.text(n - 0.5, every_ / 1.25, labels["every"], "ink-2", "end")   # at the right: seed 0's dot lies under the line at the left
    return chart.figure([p], alt=labels["alt"], width=460, height=300)


_DOTS = G._DOTS
_KEEP = G._KEEP

# --- exercise 1: result_rows -------------------------------------------------------------------------------------------

_R_HEAD = ("from repro.plan import THRESHOLD, first_step, verdict\n\n\n"
           "def result_rows(runs, budget):\n"
           "    # one row of the results table for each run, in the order of the log: its seed, t, v and verdict; where\n"
           "    # training or validation does not reach THRESHOLD, the text \"not within \" and the budget in place of the step\n"
           "    rows = []\n    for log in runs:\n")
_R_T = "        t = first_step(log[\"step\"], log[\"train\"], THRESHOLD)\n"
_R_V = "        v = first_step(log[\"step\"], log[\"val\"], THRESHOLD)\n"
_R_TEXT = ("        if t is None:\n            t = f\"not within {budget}\"\n"
           "        if v is None:\n            v = f\"not within {budget}\"\n")
_R_ROW = "        rows.append([log[\"seed\"], t, v, verdict(log[\"step\"], log[\"train\"], log[\"val\"])])\n"
_R_TAIL = "    return rows\n"
_R_START = (_R_HEAD + "        # t and v of the run, the text for a step that is not reached, and the row of the run\n        ...\n" + _R_TAIL)
_R = _R_HEAD + _R_T + _R_V + _R_TEXT + _R_ROW + _R_TAIL
_R_SKIP = _R_HEAD + _R_T + _R_V + "        if v is not None:\n    " + _R_ROW.replace("\n", "\n    ").rstrip(" ") + _R_TAIL
_R_NONE = _R_HEAD + _R_T + _R_V + _R_ROW + _R_TAIL
_R_BARE = _R.replace("f\"not within {budget}\"", "\"not reached\"")
_R_TRAIN = _R.replace(_R_V, "        v = first_step(log[\"step\"], log[\"train\"], THRESHOLD)\n")

_R_RUN = ("def run(seed, t, v):   # a run whose training reaches 99% at step t and validation at step v; None: never\n"
          "    steps = [10, 100, 1000, 10000]\n"
          "    return {\"seed\": seed, \"step\": steps, \"train\": [1.0 if t is not None and s >= t else 0.5 for s in steps],\n"
          "            \"val\": [1.0 if v is not None and s >= v else 0.3 for s in steps]}\n\n\n")
_R_SCENARIOS = [
    # the lesson's example: one grokked, one without delay, one whose validation does not get there
    _R_RUN + "result_rows([run(0, 10, 1000), run(1, 100, 100), run(2, 10, None)], 10000)",
    # training that does not get there either, and seeds out of order
    _R_RUN + "result_rows([run(5, 10, 100), run(3, None, None), run(4, 100, None)], 5000)",
    # an empty log
    _R_RUN + "result_rows([], 10000)",
]
_R_EN, _R_NL = "Your `result_rows` gives other rows than the course's", "Je `result_rows` geeft andere rijen dan die van de course"


def _same_rows(got, want) -> bool:
    """Rows equal cell by cell as text: a step as 1000 or as "1000" is the same cell; tuples count as lists."""
    if not isinstance(got, (list, tuple)) or len(got) != len(want):
        return False
    return all(isinstance(g, (list, tuple)) and [str(x) for x in g] == [str(x) for x in w] for g, w in zip(got, want))


# --- exercise 2: report ------------------------------------------------------------------------------------------------

_P_HEAD = ("SECTIONS = [\"Claim\", \"Setup\", \"Results\", \"Differences\", \"Not checked\"]\n\n\n"
           "def report(title, parts):\n"
           "    # the report in Markdown: \"# \" and the title; then for each name in SECTIONS, in that order, \"## \" and the\n"
           "    # name, and its text from parts, or \"*Not written yet.*\" when parts has no text for it; an empty line between\n"
           "    # each of them\n"
           "    lines = [\"# \" + title]\n")
_P_LOOP = ("    for name in SECTIONS:\n        lines.append(\"## \" + name)\n"
           "        if name in parts and parts[name] != \"\":\n            lines.append(parts[name])\n"
           "        else:\n            lines.append(\"*Not written yet.*\")\n")
_P_TAIL = "    return \"\\n\\n\".join(lines)\n"
_P_START = _P_HEAD + "    # every section, in the order of SECTIONS\n    ...\n" + _P_TAIL
_P = _P_HEAD + _P_LOOP + _P_TAIL
_P_ORDER = _P.replace("    for name in SECTIONS:\n", "    for name in parts:\n")
_P_SKIP = _P_HEAD + ("    for name in SECTIONS:\n        if name in parts:\n            lines.append(\"## \" + name)\n"
                     "            lines.append(parts[name])\n") + _P_TAIL
_P_EMPTY = _P.replace("        if name in parts and parts[name] != \"\":\n", "        if name in parts:\n")

_P_VISIBLE = ("Weight decay on addition", {"Results": "Later without weight decay in every seed.", "Claim": "Weight decay helps.",
                                           "Setup": "Five seeds.", "Differences": "Addition, not S5."})
_P_EN, _P_NL = "Your `report` gives another text than the course's", "Je `report` geeft een andere tekst dan die van de course"


def _show_text(v) -> str:
    """A text as the check shows it: in quotes, with its line breaks as \\n, as Python writes it."""
    return repr(v) if isinstance(v, str) else _show_rounded(v, 4)


EXERCISES = {
    "result-rows": ClassBox(
        name="result-rows",
        start=_R_START,
        solution=_R,
        scenarios=_R_SCENARIOS,
        compare=_same_rows,
        save_as=("repro", "results"),
        first=[_DOTS],
        mistakes=[
            Mistake("skip", wrong=_R_SKIP,
                    en=_R_EN + ": a run whose validation does not reach 99% gets no row. That is a negative result, and it belongs in the table: give it a row, with `\"not within \"` and the budget for its step.",
                    nl=_R_NL + ": een run waarvan de validatie geen 99% haalt, krijgt geen rij. Dat is een negatief resultaat, en het hoort in de tabel: geef het een rij, met `\"not within \"` en het budget voor zijn stap."),
            Mistake("none", wrong=_R_NONE,
                    en=_R_EN + ": where a step is not reached, the row says `None`. A reader of the table does not know what `None` means: write `\"not within \"` followed by the budget in its place.",
                    nl=_R_NL + ": waar een stap niet gehaald wordt, staat in de rij `None`. Een lezer van de tabel weet niet wat `None` betekent: zet `\"not within \"` gevolgd door het budget op die plek."),
            Mistake("bare", wrong=_R_BARE,
                    en=_R_EN + ": a step that is not reached says \"not reached\", without the budget. Then it reads as never, while the run only stopped: write `\"not within \"` followed by the budget.",
                    nl=_R_NL + ": een stap die niet gehaald wordt, zegt \"not reached\", zonder het budget. Dan leest het als nooit, terwijl de run alleen stopte: schrijf `\"not within \"` gevolgd door het budget."),
            Mistake("train", wrong=_R_TRAIN,
                    en=_R_EN + ": `v` comes from the training accuracy. `v` is the step at which validation reaches 99%: `first_step(log[\"step\"], log[\"val\"], THRESHOLD)`.",
                    nl=_R_NL + ": `v` komt uit de training-accuracy. `v` is de stap waarop validatie 99% haalt: `first_step(log[\"step\"], log[\"val\"], THRESHOLD)`."),
        ],
        texts={
            **_KEEP,
            "empty": {"en": "Replace the `...`: for each run, `t` and `v` with `first_step`, the text `\"not within \"` followed by the budget for one that is `None`, and a row with the seed, `t`, `v` and the verdict of your plan.",
                      "nl": "Vervang de `...`: voor elke run `t` en `v` met `first_step`, de tekst `\"not within \"` gevolgd door het budget voor een die `None` is, en een rij met de seed, `t`, `v` en het oordeel van je plan."},
            "wrong": {"en": _R_EN + ": for the first example, the last line gives `{got}`, and it should give `{want}`. Does every run get a row, with its seed, `t` and `v` from `first_step`, the text `\"not within \"` followed by the budget for a step that is `None`, and `verdict` of the run?",
                      "nl": _R_NL + ": voor het eerste voorbeeld geeft de laatste regel `{got}`, en dat moet `{want}` zijn. Krijgt elke run een rij, met zijn seed, `t` en `v` uit `first_step`, de tekst `\"not within \"` gevolgd door het budget voor een stap die `None` is, en `verdict` van de run?"},
            "almost": {"en": "Your `result_rows` gives the course's rows for the first example. But with these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`. Does it work when training does not reach 99% either, keep the order of the log, and give an empty table for an empty log?",
                       "nl": "Je `result_rows` geeft de rijen van de course voor het eerste voorbeeld. Maar met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn. Werkt hij ook als training geen 99% haalt, houdt hij de volgorde van het log aan, en geeft hij een lege tabel voor een leeg log?"},
            "correct": {"en": "Your `result_rows` gives every run a row, also the ones that did not get there within the budget. It is now in your project, as `work/projects/repro/results.py`.",
                        "nl": "Je `result_rows` geeft elke run een rij, ook de runs die er binnen het budget niet kwamen. Hij staat nu in je project, als `work/projects/repro/results.py`."},
            "kept": {"en": "Your `result_rows` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `result_rows` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "report": FunctionBox(
        name="report",
        function="report",
        start=_P_START,
        solution=_P,
        visible=[_P_VISIBLE],
        hidden=[("All five", {name: f"Text of {name.lower()}." for name in SECTIONS}),
                ("Nothing yet", {}),
                ("An empty part", {"Claim": "C.", "Setup": "S.", "Results": "R.", "Differences": "D.", "Not checked": ""})],
        show=_show_text,
        save_as=("repro", "report"),
        first=[_DOTS],
        mistakes=[
            Mistake("order", wrong=_P_ORDER,
                    en="`{call}` gives `{got}`, and it should give `{want}`: your sections come in the order of `parts`, and one that `parts` does not have is missing. Loop over `SECTIONS`: then every report has the same five sections in the same order.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je secties komen in de volgorde van `parts`, en een die `parts` niet heeft, ontbreekt. Loop over `SECTIONS`: dan heeft elk verslag dezelfde vijf secties in dezelfde volgorde."),
            Mistake("skip", wrong=_P_SKIP,
                    en="`{call}` gives `{got}`, and it should give `{want}`: a section that `parts` does not have is left out, so a reader cannot see that it is missing. Give it its heading too, with `\"*Not written yet.*\"`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: een sectie die `parts` niet heeft, valt weg, dus ziet een lezer niet dat hij ontbreekt. Geef hem ook zijn kop, met `\"*Not written yet.*\"`."),
            Mistake("empty", wrong=_P_EMPTY, state="almost",
                    en="Your `report` gives the course's text for the first example. But `{call}` gives `{got}`, and it should give `{want}`: a part with an empty text counts as written. Give it `\"*Not written yet.*\"` too, when its text is `\"\"`.",
                    nl="Je `report` geeft de tekst van de course voor het eerste voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn: een deel met een lege tekst telt als geschreven. Geef ook dat `\"*Not written yet.*\"`, als zijn tekst `\"\"` is."),
        ],
        texts={
            **_KEEP,
            "empty": {"en": "Replace the `...`: for each name in `SECTIONS`, `\"## \"` and the name, then its text from `parts`, or `\"*Not written yet.*\"` when there is none.",
                      "nl": "Vervang de `...`: voor elke naam in `SECTIONS` `\"## \"` en de naam, dan zijn tekst uit `parts`, of `\"*Not written yet.*\"` als die er niet is."},
            "arguments": {"en": "The check calls `{call}`, with a title and the parts, and the `def` line of your `report` has another number of names between its parentheses. Keep `(title, parts)`.",
                          "nl": "De check roept `{call}` aan, met een titel en de delen, en de `def`-regel van je `report` heeft een ander aantal namen tussen de haakjes. Houd `(title, parts)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does every name of `SECTIONS`, in that order, get `\"## \"` and the name, and then its text, or `\"*Not written yet.*\"`?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Krijgt elke naam van `SECTIONS`, in die volgorde, `\"## \"` en de naam, en dan zijn tekst, of `\"*Not written yet.*\"`?"},
            "almost": {"en": "Your `report` gives the course's text for the first example, but `{call}` gives `{got}`, and it should give `{want}`. Does it work when every part is there, when none is, and when one is empty?",
                       "nl": "Je `report` geeft de tekst van de course voor het eerste voorbeeld, maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij als elk deel er is, als er geen is, en als er een leeg is?"},
            "correct": {"en": "Your `report` gives every report the same five sections in the same order, and shows the ones not written yet. It is now in your project, as `work/projects/repro/report.py`.",
                        "nl": "Je `report` geeft elk verslag dezelfde vijf secties in dezelfde volgorde, en laat zien welke nog niet geschreven zijn. Hij staat nu in je project, als `work/projects/repro/report.py`."},
            "kept": {"en": "Your `report` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `report` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# The lesson is done when these are correct. "not-checked" is the open question, "verdict" the choice question.
REQUIRED = ["result-rows", "report", "not-checked", "verdict"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("result-rows", _R_START, "empty", None),
    ("result-rows", _R, "correct", None),
    ("result-rows", _R_SKIP, "wrong", "skip"),
    ("result-rows", _R_NONE, "wrong", "none"),
    ("result-rows", _R_BARE, "wrong", "bare"),
    ("result-rows", _R_TRAIN, "wrong", "train"),
    ("report", _P_START, "empty", None),
    ("report", _P, "correct", None),
    ("report", _P_ORDER, "wrong", "order"),
    ("report", _P_SKIP, "wrong", "skip"),
    ("report", _P_EMPTY, "almost", "empty"),
]


# --- the demo: the learner's report ------------------------------------------------------------------------------------

def _course_fn(name: str, key: str):
    projects.builds_on(*BUILDS_ON)   # the kit this lesson's own code imports, also when another lesson set it since
    return run_code(EXERCISES[key].solution).names[name]


def _table(head: list[str], rows: list[list]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


def setup_text(settings: dict) -> str:
    """The Setup section: the settings the runs used, the budget and how often they logged, the seeds, and what the
    reproduction does differently on purpose, the "ours" group of the sort of repro-guided-compare."""
    s = {**PAPER, **settings}
    return (f"The ModernGPT of the kit, {s['layers']} layers of width {s['width']} with {s['heads']} heads, on addition modulo {s['p']}. "
            f"AdamW with learning rate {s['lr']}, weight decay {s['weight_decay']}, betas {tuple(s['betas'])}, {s['warmup']} warmup steps "
            f"and batches of {s['batch']}, as in Appendix A.1.2. Training data: {round(100 * FRACTION)}% of the table for the figure, "
            f"30% to 90% for the ablation, {round(100 * WIDE)}% for the comparison over seeds. Each run {STEPS:,} steps, "
            f"validation logged at 20 steps per factor of 10. Seeds {', '.join(str(x) for x in G.SEEDS)} for the figure, seed 0 for "
            f"every combination of the ablation, seeds {SEEDS[0]} to {SEEDS[-1]} for the comparison.\n\n"
            f"Different on purpose: {STEPS:,} steps per run instead of the 100,000 of most of the paper's runs; weight decay tested on addition modulo "
            f"{s['p']}, not on the product in S5; and RMSNorm, SwiGLU and RoPE in the model, where the paper has a standard "
            "decoder-only transformer.")


def differences_text() -> str:
    """The Differences section: the task and the measure that differ from the paper's claim, and what the paper does
    not say, the "unsaid" group of the sort of repro-guided-compare, with what the reproduction chose there."""
    return (f"Figure 1 (left) of the paper is division modulo 97 at 50% of the table, trained with Adam without weight decay for "
            f"1,000,000 steps (Appendix A.1.2); this reproduction's figure is addition at {round(100 * FRACTION)}%, with the AdamW "
            f"settings above, for {STEPS:,} steps. "
            "The paper's claim about weight decay is on the product in S5, measured as how much data it takes; this "
            f"reproduction measures steps, on addition at {round(100 * WIDE)}% of the table. The paper does not say which "
            "parameters weight decay applies to: here it applies to all of them, the embeddings too. Nor does it say how "
            "often validation was measured: here at 20 steps per factor of 10, so a moment is known to within a logged step.")


def results_text(rows_of, compare_of, figure_runs: list, grid_runs: list) -> str:
    """The Results section: the figure's runs and the ablation's grid as tables of result_rows, and the comparison over
    seeds at WIDE as a sentence, with the pairs left out."""
    budget = f"{STEPS:,}"
    figure = _table(["seed", "t", "v", "verdict"], rows_of(figure_runs, budget))
    grid = _table(["weight decay", "fraction", "seed", "t", "v", "verdict"],
                  [[r["weight_decay"], f"{round(100 * r['fraction'])}%", *row] for r, row in zip(grid_runs, rows_of(grid_runs, budget))])
    r = compare_of(runs_of(0.0, WIDE), runs_of(1.0, WIDE), C.N_BOOT, 0)
    lost = [s for s, x in enumerate(factors(len(SEEDS))) if x is None]
    gone = (f" Left out: seed {', '.join(str(s) for s in lost)}, where validation without weight decay did not reach 99% within "
            f"{budget} steps; its factor is more than {C.at_least(lost[0]):.1f}, so the mean is, if anything, too low." if lost else "")
    return (f"The key figure, at {round(100 * FRACTION)}% of the table:\n\n{figure}\n\nThe ablation:\n\n{grid}\n\n"
            f"Over seeds {SEEDS[0]} to {SEEDS[-1]} at {round(100 * WIDE)}% of the table: without weight decay, validation reaches 99% "
            f"{10 ** r['mean']:.1f} times as late as with it (paired bootstrap, 95% interval {10 ** r['low']:.1f} to "
            f"{10 ** r['high']:.1f}), over {r['pairs']} pairs.{gone}")


def course_report(claim: str = CLAIM, not_checked: str = NOT_CHECKED) -> str:
    """The course's report: its own functions, its own runs."""
    parts = {"Claim": claim, "Setup": setup_text({}),
             "Results": results_text(_course_fn("result_rows", "result-rows"), C.compare, G.course_runs(), A.course_runs()),
             "Differences": differences_text(), "Not checked": not_checked}
    return _course_fn("report", "report")(TITLE, parts)


TITLE = "Grokking (Power and others, 2022), reproduced on addition modulo 97"


def demo(claim: str, not_checked: str) -> dict:
    """The learner's report: their result_rows, report and compare from their project, where they pass their checks
    the course's otherwise; their settings, their runs of repro-guided-figure and their grid of repro-guided-ablation,
    the course's where theirs are not there yet. Gives the text, and which parts are the course's."""
    folder = projects.folder() / "repro"
    projects.builds_on(*BUILDS_ON)
    projects.on_path()
    projects.forget()
    stand = projects.use_course_versions()
    course = {"result_rows": _course_fn("result_rows", "result-rows"), "report": _course_fn("report", "report"), "compare": C.compare}
    fns, theirs = {}, []
    try:
        for name, module in (("result_rows", "results"), ("report", "report"), ("compare", "compare")):
            fns[name] = course[name]
            if (folder / f"{module}.py").is_file():
                try:
                    fns[name] = getattr(importlib.import_module(f"repro.{module}"), name)
                    theirs.append(name)
                except Exception:
                    pass
        try:
            settings = importlib.import_module("repro.settings").SETTINGS
        except Exception:
            settings = {}
        figure, grid = G.my_runs() or G.course_runs(), A.my_runs() or A.course_runs()
        try:
            setup = setup_text(settings if isinstance(settings, dict) else {})
        except Exception:   # settings changed by hand into something the text cannot read: the paper's
            setup = setup_text({})
        parts = {"Claim": claim.strip(), "Setup": setup,
                 "Differences": differences_text(), "Not checked": not_checked.strip()}
        try:
            parts["Results"] = results_text(fns["result_rows"], fns["compare"], figure, grid)
        except Exception:
            parts["Results"] = results_text(course["result_rows"], course["compare"], figure, grid)
            theirs = [x for x in theirs if x == "report"]
        try:
            text = fns["report"](TITLE, parts)
            assert isinstance(text, str)
        except Exception:
            text = course["report"](TITLE, parts)
            theirs = [x for x in theirs if x != "report"]
        return {"text": text, "theirs": theirs, "my_figure": G.my_runs() is not None, "my_grid": A.my_runs() is not None,
                "stand_ins": [f"{package}/{module}.py" for package, module, _, _ in stand]}
    finally:
        projects.forget()


def save(text: str) -> str:
    """Saves the report as work/projects/repro/report.md; gives back where."""
    path = projects.folder() / "repro" / REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return f"work/projects/repro/{REPORT}"
