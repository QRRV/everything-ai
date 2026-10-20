# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson repro-own-figure: run_plan, the runner of the kit, which runs every seed, budget and setting
of the plan into one experiment log, one line of JSON per run, and goes on where it stopped, saved as own/runner.py;
results_at, which reads the results at one budget from that log the way the plan's verdict takes them, saved as
own/results.py; and the budget of the main result, saved as own/budget.json before the first run. The play block is
the course's main result for the dropout paper, as its experiment log fills."""
import functools
import importlib.util
import json
import math
import re
from dataclasses import dataclass, field

import torch  # noqa: F401  loaded here: a first import inside the time limit of a check would not finish in it

from everything_ai import paths, projects
from everything_ai.check import ClassBox, Exercise, Mistake, Result, _fill, run_code


def _checks(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _checks("repro-own-build", "everything_ai_checks_repro_own_build_for_figure")
G = _checks("repro-guided-figure", "everything_ai_checks_repro_guided_figure_for_own")
P = B.P
E, SETTINGS, SEEDS, BUDGETS, RUNS, number, short, whole = B.E, B.SETTINGS, B.SEEDS, B.BUDGETS, B.RUNS, B.number, B.short, B.whole
LADDER = 8          # the budgets of the main result: the budget of the plan and seven halvings of it
LEAST_STEPS = 2 ** (LADDER - 1)   # the smallest budget whose ladder has eight budgets of at least one step
COURSE_STEPS = BUDGETS[-1]        # the course's budget for the dropout paper
COURSE_MINUTES = 17               # all 48 runs of the course's main result on one thread: 76,500 steps each of 9.0 and 4.4 ms
_DOTS, _KEEP = B._DOTS, G._KEEP


def ladder(steps: int) -> list[int]:
    """The budgets of the main result, smallest first: the budget and its halvings, as whole numbers of steps."""
    return sorted({steps // 2 ** k for k in range(LADDER)} - {0})


def course_log() -> list[dict]:
    """The course's main result for the dropout paper, as the runner writes it: for each seed and budget, "claim" and
    then "other", one line each."""
    return [{"setting": s, "seed": seed, "steps": steps, "result": RUNS[steps][s][i]}
            for i, seed in enumerate(SEEDS) for steps in BUDGETS for s in SETTINGS]


@functools.cache
def _verdict():
    return run_code(P._V).names["verdict"]


def course_results_at(lines: list[dict], steps: int, plan: dict) -> dict:
    """The course's results_at: for each setting of the plan, by its name, the result of every seed at `steps`."""
    a, b = P._settings(plan)
    names = {"claim": a, "other": b}
    results = {a: [], b: []}
    for line in sorted(lines, key=lambda line: line["seed"]):
        if line["steps"] == steps:
            results[names[line["setting"]]].append(line["result"])
    return results


def course_verdict(lines: list[dict], plan: dict, steps: int = COURSE_STEPS) -> str:
    """The course's rule on the runs of the log at a budget."""
    return _verdict()(plan, course_results_at(lines, steps, plan))


def median(xs: list[float]) -> float:
    s = sorted(xs)
    return s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2


def figure_picture(lines: list[dict], names: tuple[str, str], budget: int, labels: dict, last: dict | None = None,
                   ylim: tuple | None = None, role: str = "loss") -> str:
    """The main figure from the lines of an experiment log: the result of each seed at each budget, a dot, filled for
    "claim" and hollow for "other", on a log axis of steps; a line through the medians of the budgets where every
    seed of a setting is in, solid for "claim" and dashed for "other"; a thin line up at the budget of the plan; the
    newest line, `last`, ringed; and under it the names of the settings; `ylim` fixes the y-axis, as the play block does
    while its log fills. `role` is the colour: "loss" for the course's test error, which counts mistakes; "ink" for the
    metric of a learner's plan, which can be an accuracy. A result that is not a finite number is left out."""
    from everything_ai import chart
    lines = [line for line in lines if math.isfinite(line["result"])]
    budgets = sorted({line["steps"] for line in lines} | {budget})
    seeds = sorted({line["seed"] for line in lines})
    values = [line["result"] for line in lines] or [0.0, 1.0]
    lo, hi = ylim or (math.floor(min(values)) - 1, math.ceil(max(values)) + 1)
    x0, x1 = (BUDGETS[0] / 1.5, BUDGETS[-1] * 1.5) if ylim else (budgets[0] / 1.5, budgets[-1] * 1.5)
    p = chart.Panel((x0, x1), (lo, hi), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"], xlog=True)
    p.line([budget, budget], [lo, hi], "ink-3", 1.0)
    p.text(budget, hi - (hi - lo) * 0.04, labels["budget"], "ink-2", "end")
    for s, hollow, dashed in ((SETTINGS[0], False, False), (SETTINGS[1], True, True)):
        mine = [line for line in lines if line["setting"] == s]
        whole_budgets = [b for b in budgets if seeds and len([m for m in mine if m["steps"] == b]) == len(seeds)]
        p.line(whole_budgets, [median([m["result"] for m in mine if m["steps"] == b]) for b in whole_budgets], role, 1.6, dashed=dashed)
        if mine:
            p.points([m["steps"] for m in mine], [m["result"] for m in mine], role, r=3.5, hollow=hollow)
    if last is not None and math.isfinite(last["result"]):
        p.points([last["steps"]], [last["result"]], role, r=8, hollow=True)
    return chart.figure([p], alt=labels["alt"], width=360, height=260) + _legend(names, role)


def _legend(names: tuple[str, str], role: str = "loss") -> str:
    """The two settings under the figure: a filled dot on a solid line, and a hollow one on a dashed line."""
    import html

    def item(name: str, hollow: bool) -> str:
        dash = ' stroke-dasharray="5 5"' if hollow else ""
        fill = "var(--ea-page)" if hollow else f"var(--ea-{role})"
        return (f'<span style="display:inline-flex;align-items:center;gap:6px;white-space:nowrap"><svg width="30" height="12" aria-hidden="true">'
                f'<line x1="0" y1="6" x2="30" y2="6" style="stroke:var(--ea-{role});stroke-width:1.6"{dash}/>'
                f'<circle cx="15" cy="6" r="3.5" style="fill:{fill};stroke:var(--ea-{role});stroke-width:1.8"/></svg>{html.escape(name)}</span>')
    return (f'<div style="display:flex;flex-wrap:wrap;gap:4px 18px;margin:2px 0 0 44px;font-size:13px;color:var(--ea-ink-2);'
            f'font-family:var(--ea-font-ui)">{item(names[0], False)}{item(names[1], True)}</div>')


def stop_at(n: int) -> dict | None:
    """The run that is going after `n` lines of the course's log, which a stop would cost; None when all are done."""
    log = course_log()
    return log[n] if n < len(log) else None


# --- exercise 1: the runner ---------------------------------------------------------------------------------------------

_R_HEAD = ("import json\n\n\n"
           "def run_plan(run, seeds, budgets, path, progress=None, settings=(\"claim\", \"other\")):\n"
           "    # every run of the plan: for each seed and each budget, each of the settings in turn.\n"
           "    # Each run adds one line of JSON to the experiment log at path, right after it ran:\n"
           "    # {\"setting\": ..., \"seed\": ..., \"steps\": ..., \"result\": ...}. A run whose line is in\n"
           "    # the log already is not run again, so a stopped plan goes on where it stopped.\n"
           "    # Each new line also goes to progress, when there is one.\n"
           "    with open(path, \"a\") as f:   # \"a\" makes the log when there is none yet, and keeps it\n"
           "        f.write(\"\")\n")
_R_DONE = ("    done = []   # the setting, the seed and the steps of every line in the log\n"
           "    for text in open(path):\n"
           "        old = json.loads(text)\n"
           "        done.append((old[\"setting\"], old[\"seed\"], old[\"steps\"]))\n")
_R_LOOPS = ("    for seed in seeds:\n"
            "        for steps in budgets:\n"
            "            for setting in settings:\n")
_R_RUN = ("                if (setting, seed, steps) not in done:\n"
          "                    line = {\"setting\": setting, \"seed\": seed, \"steps\": steps}\n"
          "                    line[\"result\"] = run(setting, seed, steps)\n"
          "                    with open(path, \"a\") as f:\n"
          "                        f.write(json.dumps(line) + \"\\n\")\n"
          "                    if progress is not None:\n"
          "                        progress(line)\n")
_R_START = (_R_HEAD + "    # 1. done: the setting, the seed and the steps of every line in the log, a tuple each\n    ...\n" + _R_LOOPS
            + "                # 2. a run that is not done: run it, add its line to the log, give it to progress\n"
              "                ...\n")


def _r(done: str = _R_DONE, run: str = _R_RUN) -> str:
    return _R_HEAD + done + _R_LOOPS + run


_R = _r()
_R_WRONG = {
    "seed-only": _r(done=_R_DONE.replace("done.append((old[\"setting\"], old[\"seed\"], old[\"steps\"]))", "done.append(old[\"seed\"])"),
                    run=_R_RUN.replace("if (setting, seed, steps) not in done:", "if seed not in done:")),
    "no-setting": _r(done=_R_DONE.replace("(old[\"setting\"], old[\"seed\"], old[\"steps\"])", "(old[\"seed\"], old[\"steps\"])"),
                     run=_R_RUN.replace("if (setting, seed, steps) not in done:", "if (seed, steps) not in done:")),
    "no-steps": _r(done=_R_DONE.replace("(old[\"setting\"], old[\"seed\"], old[\"steps\"])", "(old[\"setting\"], old[\"seed\"])"),
                   run=_R_RUN.replace("if (setting, seed, steps) not in done:", "if (setting, seed) not in done:")),
    "at-the-end": _r(run=("                if (setting, seed, steps) not in done:\n"
                          "                    line = {\"setting\": setting, \"seed\": seed, \"steps\": steps}\n"
                          "                    line[\"result\"] = run(setting, seed, steps)\n"
                          "                    done.append(line)\n"
                          "    with open(path, \"a\") as f:   # all new lines at once, at the end\n"
                          "        for line in done:\n"
                          "            if isinstance(line, dict):\n"
                          "                f.write(json.dumps(line) + \"\\n\")\n"
                          "                if progress is not None:\n"
                          "                    progress(line)\n")),
    "again": _r(run=_R_RUN.replace("if (setting, seed, steps) not in done:", "if True:")),
    "overwrite": _r(run=_R_RUN.replace("                    with open(path, \"a\") as f:\n", "                    with open(path, \"w\") as f:\n")),
    "no-progress": _r(run=_R_RUN.replace("                    if progress is not None:\n                        progress(line)\n", "")),
    "fixed-settings": _r().replace("            for setting in settings:\n", "            for setting in [\"claim\", \"other\"]:\n"),
}

_R_FAKE = ("import json\nimport tempfile\nfrom pathlib import Path\n\n"
           "path = str(Path(tempfile.mkdtemp()) / \"runs.jsonl\")\n"
           "calls = []   # each run: its setting, seed and steps, and how many lines the log had then\n\n\n"
           "def run(setting, seed, steps):   # a quick stand-in for your experiment: no training\n"
           "    lines = len(Path(path).read_text().splitlines()) if Path(path).exists() else 0\n"
           "    calls.append([setting, seed, steps, lines])\n"
           "    return 10 + seed + steps / 100 + (1 if setting == \"other\" else 0)\n\n\n")
_R_READ = "\n[calls, [json.loads(text) for text in open(path)]]"
_R_SCENARIOS = [
    # the lesson's example: two seeds and two budgets into a new log
    _R_FAKE + "run_plan(run, [0, 1], [100, 200], path)" + _R_READ,
    # a log that has three runs of seed 0 already: only the fourth is run, and the three lines stay
    _R_FAKE + "with open(path, \"w\") as f:\n"
              "    for setting, steps in [(\"claim\", 100), (\"other\", 100), (\"claim\", 200)]:\n"
              "        f.write(json.dumps({\"setting\": setting, \"seed\": 0, \"steps\": steps, \"result\": 9}) + \"\\n\")\n"
              "run_plan(run, [0], [100, 200], path)" + _R_READ,
    # a plan that stops in its second run, between the two settings, as when you close the lesson, and is run again
    _R_FAKE + "def stops(setting, seed, steps):\n    if len(calls) == 1:\n        raise RuntimeError(\"stopped\")\n"
              "    return run(setting, seed, steps)\n\n\n"
              "try:\n    run_plan(stops, [0], [100, 200], path)\nexcept RuntimeError:\n    pass\n"
              "run_plan(run, [0], [100, 200], path)" + _R_READ,
    # progress gets every new line, as it is written
    _R_FAKE + "seen = []\nrun_plan(run, [3], [50], path, seen.append)\nseen",
    # other settings, such as the ablation of the next lesson: only those run
    _R_FAKE + "run_plan(run, [0, 1], [100], path, None, [\"ablation\"])" + _R_READ,
]
_R_EN, _R_NL = "Your `run_plan` gives another experiment log than the course's", "Je `run_plan` geeft een ander logboek dan dat van de course"

# --- exercise 2: the results at the budget --------------------------------------------------------------------------------

_A_HEAD = ("def results_at(lines, steps, plan):\n"
           "    # the results of the runs of `steps` steps, as the plan's verdict takes them: for each of\n"
           "    # the plan's two settings, by its name in the plan, the result of every seed, the smallest\n"
           "    # seed first. lines are the lines of the experiment log, each a dictionary\n"
           "    names = {\"claim\": plan[\"settings\"][0], \"other\": plan[\"settings\"][1]}\n"
           "    results = {plan[\"settings\"][0]: [], plan[\"settings\"][1]: []}\n")
_A_LOOP = ("    for line in sorted(lines, key=lambda line: line[\"seed\"]):\n"
           "        if line[\"steps\"] == steps:\n"
           "            results[names[line[\"setting\"]]].append(line[\"result\"])\n")
_A_TAIL = "    return results\n"
_A_START = _A_HEAD + "    ...\n" + _A_TAIL
_A = _A_HEAD + _A_LOOP + _A_TAIL
_A_WRONG = {
    "log-order": _A_HEAD + _A_LOOP.replace("sorted(lines, key=lambda line: line[\"seed\"])", "lines") + _A_TAIL,
    "all-steps": _A_HEAD + _A_LOOP.replace("        if line[\"steps\"] == steps:\n            results", "        results") + _A_TAIL,
    "swapped": _A_HEAD.replace("names = {\"claim\": plan[\"settings\"][0], \"other\": plan[\"settings\"][1]}",
                               "names = {\"claim\": plan[\"settings\"][1], \"other\": plan[\"settings\"][0]}") + _A_LOOP + _A_TAIL,
    "kit-names": _A_HEAD + "    results = {\"claim\": [], \"other\": []}\n"
                 + _A_LOOP.replace("results[names[line[\"setting\"]]]", "results[line[\"setting\"]]") + _A_TAIL,
    "by-size": _A_HEAD + _A_LOOP.replace("key=lambda line: line[\"seed\"]", "key=lambda line: line[\"result\"]") + _A_TAIL,
}
_A_LOG = ("plan = {\"settings\": [\"with dropout\", \"without dropout\"]}\n"
          "log = []   # a made-up experiment log: two seeds, two budgets\n"
          "for seed in [0, 1]:\n"
          "    for steps in [100, 200]:\n"
          "        for setting in [\"claim\", \"other\"]:\n"
          "            result = 20 - steps / 100 + (seed if setting == \"claim\" else 3 - seed)\n"
          "            log.append({\"setting\": setting, \"seed\": seed, \"steps\": steps, \"result\": result})\n")
_A_SCENARIOS = [
    _A_LOG + "results_at(log, 200, plan)",
    # the "claim" run of seed 1 at 200 steps first, as when it ran apart to try it out: in the order of the log,
    # place 0 of "claim" would be seed 1, and place 0 of "other" seed 0
    _A_LOG + "results_at(log[6:7] + log[:6] + log[7:], 200, plan)",
    _A_LOG + "results_at(log, 100, {\"settings\": [\"rank 8\", \"full\"]})",
]
_A_EN, _A_NL = "Your `results_at` gives other results than the course's", "Je `results_at` geeft andere resultaten dan die van de course"

EXERCISES = {
    "run-plan": ClassBox(
        name="run-plan",
        start=_R_START,
        solution=_R,
        scenarios=_R_SCENARIOS,
        save_as=("own", "runner"),
        first=[_DOTS],
        mistakes=[
            Mistake("no-line", error=G.EXERCISES["run-seeds"].mistakes[0].error,
                    en="Reading your experiment log back stops: two runs stand on one line, because each line needs a newline at its end. Write `json.dumps(line) + \"\\n\"`.",
                    nl="Je logboek teruglezen stopt: twee runs staan op één regel, omdat elke regel aan het eind een regeleinde nodig heeft. Schrijf `json.dumps(line) + \"\\n\"`."),
            Mistake("str", error=G.EXERCISES["run-seeds"].mistakes[1].error,
                    en="Reading your experiment log back stops: a line of it is not JSON. It has single quotes, the way Python shows a dictionary, and JSON needs double quotes. Write each line with `json.dumps(line)`.",
                    nl="Je logboek teruglezen stopt: een regel ervan is geen JSON. Hij heeft enkele aanhalingstekens, zoals Python een dictionary laat zien, en JSON heeft dubbele nodig. Schrijf elke regel met `json.dumps(line)`."),
            Mistake("seed-only", wrong=_R_WRONG["seed-only"],
                    en=_R_EN + ": once a seed has one line in the log, none of its other runs is run any more. A seed alone does not name a run here: with two settings and a ladder of budgets, a seed has many. Keep the setting, the seed and the steps of each line in `done`, as a tuple, and look for all three.",
                    nl=_R_NL + ": zodra een seed één regel in het logboek heeft, wordt geen van zijn andere runs meer gedraaid. Een seed alleen wijst hier geen run aan: met twee settings en een ladder van budgetten heeft een seed er veel. Houd de setting, de seed en de stappen van elke regel bij in `done`, als tuple, en zoek naar alle drie."),
            Mistake("no-setting", wrong=_R_WRONG["no-setting"],
                    en=_R_EN + ": when the log has the `\"claim\"` run of a seed and a budget, its `\"other\"` run is never run, as after a stop between the two. The setting is part of what names a run: keep it in the tuple too.",
                    nl=_R_NL + ": als het logboek de run `\"claim\"` van een seed en een budget heeft, wordt zijn run `\"other\"` nooit gedraaid, zoals na een stop tussen die twee. De setting hoort bij wat een run aanwijst: houd hem ook in de tuple bij."),
            Mistake("no-steps", wrong=_R_WRONG["no-steps"],
                    en=_R_EN + ": once a setting of a seed has a line in the log, none of its other budgets is run any more, as after a stop in the second run. A seed and a setting do not name a run here: with a ladder, each has a run for every budget. Keep the steps in the tuple too.",
                    nl=_R_NL + ": zodra een setting van een seed een regel in het logboek heeft, wordt geen van zijn andere budgetten meer gedraaid, zoals na een stop in de tweede run. Een seed en een setting wijzen hier geen run aan: met een ladder heeft elk een run voor elk budget. Houd ook de stappen in de tuple bij."),
            Mistake("at-the-end", wrong=_R_WRONG["at-the-end"],
                    en=_R_EN + ": it writes its lines only at the end, when every run is done. A plan that stops in its last run then has none of the hours before it in its log. Write each line right after its run, with `open(path, \"a\")` inside the loop.",
                    nl=_R_NL + ": hij schrijft zijn regels pas aan het eind, als elke run klaar is. Een plan dat in zijn laatste run stopt, heeft dan niets van de uren ervoor in zijn logboek. Schrijf elke regel meteen na zijn run, met `open(path, \"a\")` in de loop."),
            Mistake("again", wrong=_R_WRONG["again"],
                    en=_R_EN + ": a run whose line is in the log already is run again, and then stands in it twice. Run it only when its tuple is not in `done`, `if (setting, seed, steps) not in done:`, with `done` a list of those tuples, one for each line of the log.",
                    nl=_R_NL + ": een run waarvan de regel al in het logboek staat, wordt opnieuw gedraaid, en staat er dan twee keer in. Draai hem alleen als zijn tuple niet in `done` staat, `if (setting, seed, steps) not in done:`, met `done` een lijst van die tuples, één voor elke regel van het logboek."),
            Mistake("overwrite", wrong=_R_WRONG["overwrite"],
                    en=_R_EN + ": only the last run is in it. `open(path, \"w\")` empties the file first, so each run wipes out the ones before. Add to it with `\"a\"`.",
                    nl=_R_NL + ": alleen de laatste run staat erin. `open(path, \"w\")` maakt het bestand eerst leeg, dus wist elke run de vorige uit. Voeg eraan toe met `\"a\"`."),
            Mistake("fixed-settings", wrong=_R_WRONG["fixed-settings"], state="almost",
                    en="Your `run_plan` writes the course's experiment log, but it runs `\"claim\"` and `\"other\"` whatever `settings` says. The next lesson gives it other settings, such as `[\"ablation\"]`: loop over `settings`.",
                    nl="Je `run_plan` schrijft het logboek van de course, maar draait `\"claim\"` en `\"other\"`, wat `settings` ook zegt. De volgende les geeft hem andere settings, zoals `[\"ablation\"]`: loop over `settings`."),
            Mistake("no-progress", wrong=_R_WRONG["no-progress"], state="almost",
                    en="Your `run_plan` writes the course's experiment log, but it does not give the new lines to `progress`, so the lesson cannot show your plan while it runs. After writing a line: `if progress is not None: progress(line)`.",
                    nl="Je `run_plan` schrijft het logboek van de course, maar geeft de nieuwe regels niet aan `progress`, dus kan de les je plan niet laten zien terwijl het draait. Na het schrijven van een regel: `if progress is not None: progress(line)`."),
        ],
        texts={
            **_KEEP,
            "empty": {"en": "Replace both `...`: first `done`, a tuple of the setting, the seed and the steps for every line of the log; then, inside the loops, a run that is not in `done`, its line added to the log and given to `progress`.",
                      "nl": "Vervang beide `...`: eerst `done`, een tuple van de setting, de seed en de stappen voor elke regel van het logboek; dan, in de loops, een run die niet in `done` staat, zijn regel toegevoegd aan het logboek en gegeven aan `progress`."},
            "wrong": {"en": _R_EN + " for the lesson's example. With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`: each run with how many lines the log had then, and the lines of the log. Does it run each run that is not in `done`, and add its line to the log right after it, with `\"a\"`, as `json.dumps(line) + \"\\n\"`?",
                      "nl": _R_NL + " voor het voorbeeld van de les. Met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn: elke run met hoeveel regels het logboek toen had, en de regels van het logboek. Draait hij elke run die niet in `done` staat, en voegt hij zijn regel er meteen na toe aan het logboek, met `\"a\"`, als `json.dumps(line) + \"\\n\"`?"},
            "almost": {"en": "Your `run_plan` writes the course's experiment log for the lesson's example. But with these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`. Does it skip a run only when its setting, seed and steps all have a line already, write each line right after its run, and give it to `progress`?",
                       "nl": "Je `run_plan` schrijft het logboek van de course voor het voorbeeld van de les. Maar met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn. Slaat hij een run alleen over als zijn setting, seed en stappen alle drie al een regel hebben, schrijft hij elke regel meteen na zijn run, en geeft hij hem aan `progress`?"},
            "correct": {"en": "Your `run_plan` runs every seed, budget and setting into one experiment log, a line right after each run, and goes on where it stopped. It is in your kit now, as `{saved}`.",
                        "nl": "Je `run_plan` draait elke seed, elk budget en elke setting in één logboek, een regel meteen na elke run, en gaat verder waar hij stopte. Hij zit nu in je kit, als `{saved}`."},
        },
    ),
    "results-at": ClassBox(
        name="results-at",
        start=_A_START,
        solution=_A,
        scenarios=_A_SCENARIOS,
        save_as=("own", "results"),
        first=[_DOTS],
        mistakes=[
            Mistake("kit-names", wrong=_A_WRONG["kit-names"],
                    en=_A_EN + ": its keys are `\"claim\"` and `\"other\"`, the names of the kit, and the plan's verdict looks for the names in the plan, such as `\"with dropout\"`. Use `names` to put each result under the plan's name of its setting.",
                    nl=_A_NL + ": de sleutels zijn `\"claim\"` en `\"other\"`, de namen van de kit, en het oordeel van het plan zoekt naar de namen in het plan, zoals `\"with dropout\"`. Gebruik `names` om elk resultaat onder de naam in het plan van zijn setting te zetten."),
            Mistake("all-steps", wrong=_A_WRONG["all-steps"],
                    en=_A_EN + ": it gives the results of every budget, not only of `steps`. The verdict takes the first results of each setting as seed 0, 1, 2, and those would be the shortest runs. Keep only the lines whose `\"steps\"` is `steps`.",
                    nl=_A_NL + ": hij geeft de resultaten van elk budget, niet alleen van `steps`. Het oordeel neemt de eerste resultaten van elke setting als seed 0, 1, 2, en dat zouden de kortste runs zijn. Houd alleen de regels waarvan `\"steps\"` gelijk is aan `steps`."),
            Mistake("swapped", wrong=_A_WRONG["swapped"],
                    en=_A_EN + ": the results of `\"claim\"` stand under the name of the other setting, and the other way round, so the verdict would turn every gain around. `\"claim\"` is the first setting of the plan, `plan[\"settings\"][0]`.",
                    nl=_A_NL + ": de resultaten van `\"claim\"` staan onder de naam van de andere setting, en andersom, dus zou het oordeel elke winst omdraaien. `\"claim\"` is de eerste setting van het plan, `plan[\"settings\"][0]`."),
            Mistake("by-size", wrong=_A_WRONG["by-size"],
                    en=_A_EN + ": the results of each setting stand in the order of their size, not of their seeds. The verdict compares the two settings place by place, so place 0 of both lists must be seed 0, place 1 seed 1, and so on. Sort the lines by their seed: `sorted(lines, key=lambda line: line[\"seed\"])`.",
                    nl=_A_NL + ": de resultaten van elke setting staan op volgorde van hun grootte, niet van hun seeds. Het oordeel vergelijkt de twee settings plek voor plek, dus moet plek 0 van beide lijsten seed 0 zijn, plek 1 seed 1, enzovoort. Sorteer de regels op hun seed: `sorted(lines, key=lambda line: line[\"seed\"])`."),
            Mistake("log-order", wrong=_A_WRONG["log-order"],
                    en=_A_EN + " when the lines of seed 1 come before those of seed 0 in the log. The verdict compares the results of the two settings place by place, as seed 0, seed 1, and so on: in the order of the log, it could compare runs of two seeds that saw other starting weights and other batches. Go through the lines sorted by their seed: `sorted(lines, key=lambda line: line[\"seed\"])`.",
                    nl=_A_NL + " als de regels van seed 1 vóór die van seed 0 in het logboek staan. Het oordeel vergelijkt de resultaten van de twee settings plek voor plek, als seed 0, seed 1, enzovoort: in de volgorde van het logboek kan het runs van twee seeds vergelijken die andere startwaarden en andere batches zagen. Loop door de regels gesorteerd op hun seed: `sorted(lines, key=lambda line: line[\"seed\"])`."),
        ],
        texts={
            **_KEEP,
            "empty": {"en": "Replace the `...` with a loop over the lines of the log: each line of `steps` steps adds its result to the list of the plan's name for its setting, `names[line[\"setting\"]]`, the smallest seed first.",
                      "nl": "Vervang de `...` door een loop over de regels van het logboek: elke regel van `steps` stappen voegt zijn resultaat toe aan de lijst van de naam in het plan van zijn setting, `names[line[\"setting\"]]`, de kleinste seed eerst."},
            "wrong": {"en": _A_EN + " for the lesson's example. With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`. Does it keep only the lines of `steps` steps, and put each result in the list of the plan's name of its setting?",
                      "nl": _A_NL + " voor het voorbeeld van de les. Met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn. Houdt hij alleen de regels van `steps` stappen, en zet hij elk resultaat in de lijst van de naam in het plan van zijn setting?"},
            "almost": {"en": "Your `results_at` gives the course's results for the lesson's example. But with these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`. Does it go through the lines in the order of their seeds, and take the names from `plan`?",
                       "nl": "Je `results_at` geeft de resultaten van de course voor het voorbeeld van de les. Maar met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn. Loopt hij door de regels in de volgorde van hun seeds, en haalt hij de namen uit `plan`?"},
            "correct": {"en": "Your `results_at` reads the results at one budget from your experiment log, the way your plan's verdict takes them: by the names of the plan, the smallest seed first. It is in your kit now, as `{saved}`.",
                        "nl": "Je `results_at` leest de resultaten bij één budget uit je logboek, zoals het oordeel van je plan ze neemt: met de namen van het plan, de kleinste seed eerst. Hij zit nu in je kit, als `{saved}`."},
        },
    ),
}


# --- exercise 3: the budget -------------------------------------------------------------------------------------------------

def log_path():
    return projects.folder() / "own" / "runs.jsonl"


def my_log() -> list[dict] | None:
    """The lines of the learner's experiment log, work/projects/own/runs.jsonl; None when there is none, or a line of
    it is not a run as run_plan writes it."""
    path = log_path()
    if not path.is_file():
        return None
    try:
        lines = [json.loads(t) for t in path.read_text(encoding="utf-8").splitlines() if t.strip()]
        ok = all(isinstance(r, dict) and r.get("setting") in SETTINGS and isinstance(r.get("seed"), int)
                 and isinstance(r.get("steps"), int) and isinstance(r.get("result"), (int, float)) for r in lines)
    except ValueError:
        return None
    return lines if lines and ok else None


def budget_path():
    return projects.folder() / "own" / "budget.json"


def saved_budget() -> dict | None:
    """The budget of the main result in the learner's kit, work/projects/own/budget.json: {"steps", "why", "written"}."""
    try:
        b = json.loads(budget_path().read_text(encoding="utf-8"))
        return b if isinstance(b, dict) and isinstance(b.get("steps"), int) and b["steps"] >= LEAST_STEPS else None
    except (OSError, ValueError):
        return None


def save_budget(steps: int, why: str) -> str:
    """Saves the budget as work/projects/own/budget.json, with the date it was written: the old date while the steps
    stay the same, also when only the reason changes. Gives back where, as the lesson shows it."""
    import datetime
    old = saved_budget() or {}
    same = old.get("steps") == steps
    keep = {"steps": steps, "why": why, "written": old.get("written") if same and old.get("written") else datetime.date.today().isoformat()}
    path = budget_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(keep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return projects.shown(path)


def _words(text) -> int:
    return len(re.findall(r"\w+", text or ""))


@dataclass
class BudgetReview(Exercise):
    """The budget of the main result, from the lesson's form: {"steps": a number, "why": a text}. Once it is a whole
    number of at least LEAST_STEPS, with a reason, it is saved in the learner's kit as budget.json. `start` is what
    the form holds before anything is filled in."""
    start: dict = field(default_factory=dict)
    solution: dict = field(default_factory=dict)

    def check(self, value) -> Result:
        if not isinstance(value, dict):
            return Result("error", self.message("missing"))
        steps, why = value.get("steps"), (value.get("why") or "").strip()
        if not steps and not why:
            return Result("empty", self.message("empty"))
        if not isinstance(steps, (int, float)) or not math.isfinite(steps) or steps != int(steps) or steps < LEAST_STEPS:
            return self._mistake("steps", least=str(LEAST_STEPS))
        steps = int(steps)
        if _words(why) < 6:
            return self._mistake("why")
        ran = {line["steps"] for line in my_log() or []}
        old, off = saved_budget(), ran - set(ladder(steps))   # off: runs that are not on the ladder of this budget
        if ran and ((old and old["steps"] != steps) or off):
            before = old["steps"] if old and old["steps"] != steps else max(off)   # the budget the runs were made with
            return self._mistake("changed", ran={"en": whole(before, "en"), "nl": whole(before, "nl")}, steps={"en": whole(steps, "en"), "nl": whole(steps, "nl")})
        saved = save_budget(steps, why)
        budgets = ladder(steps)
        return Result("correct", _fill(self.texts["correct"], saved=saved, steps={lang: whole(steps, lang) for lang in ("en", "nl")},
                                       ladder={lang: ", ".join(whole(b, lang) for b in budgets) for lang in ("en", "nl")},
                                       total={lang: whole(sum(budgets), lang) for lang in ("en", "nl")}))

    def _mistake(self, key: str, **values) -> Result:
        m = next(m for m in self.mistakes if m.key == key)
        return Result(m.state, _fill(m.text, **values), mistake=key)


_B_WHY = {"en": "The paper trains its networks long. 12,800 steps of 100 pictures are about 213 passes over the 6,000 training pictures, and the whole ladder fits well within the plan's 0.3 hours a seed.",
          "nl": "De paper traint zijn netwerken lang. 12.800 stappen van 100 afbeeldingen zijn ongeveer 213 keer door de 6000 trainingsafbeeldingen, en de hele ladder past ruim in de 0,3 uur per seed van het plan."}

EXERCISES["budget"] = BudgetReview(
    name="budget",
    start={"steps": 0, "why": ""},
    solution={"steps": COURSE_STEPS, "why": _B_WHY["en"]},
    mistakes=[
        Mistake("steps",
                en="The budget is a whole number of steps, at least {least}: the ladder halves it seven times, and its smallest run needs one step at least.",
                nl="Het budget is een heel aantal stappen, minstens {least}: de ladder halveert het zeven keer, en zijn kleinste run heeft minstens één stap nodig."),
        Mistake("why",
                en="Say in a sentence where your budget comes from: how long the paper trains, and how many hours your plan has for a seed. A reviewer of your report asks for it, and so do you, if the result surprises you.",
                nl="Zeg in een zin waar je budget vandaan komt: hoe lang de paper traint, en hoeveel uur je plan voor een seed heeft. Een reviewer van je verslag vraagt erom, en jij ook, als het resultaat je verrast."),
        Mistake("changed", state="almost",
                en="Your experiment log has runs of a budget of {ran} steps already, and your budget is now {steps}. Changing the budget after runs is choosing it after a look at results. If those runs were a test, delete `work/projects/own/runs.jsonl`, and say in your report why you changed it.",
                nl="Je logboek heeft al runs van een budget van {ran} stappen, en je budget is nu {steps}. Wie het budget verandert na runs, kiest het na een blik op resultaten. Waren die runs een test, verwijder dan `work/projects/own/runs.jsonl`, en zeg in je verslag waarom je het veranderde."),
    ],
    texts={
        "missing": {"en": "I cannot see the form. Run the cell with the form again.",
                    "nl": "Ik zie het formulier niet. Voer de cel met het formulier opnieuw uit."},
        "empty": {"en": "Fill in the budget of your main run, in steps, and in a sentence where it comes from.",
                  "nl": "Vul het budget van je hoofdrun in, in stappen, en in een zin waar het vandaan komt."},
        "correct": {"en": "Your budget is in your kit now, `{saved}`: {steps} steps. The ladder of your main result: {ladder} steps, {total} steps for each seed and setting, less than twice the budget.",
                    "nl": "Je budget zit nu in je kit, `{saved}`: {steps} stappen. De ladder van je hoofdresultaat: {ladder} stappen, {total} stappen voor elke seed en setting, minder dan twee keer het budget."},
    },
)

REQUIRED = ["run-plan", "results-at", "budget", "why-main"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("run-plan", _R_START, "empty", None),
    ("run-plan", _R, "correct", None),
    ("run-plan", _R.replace("json.dumps(line) + \"\\n\"", "json.dumps(line)"), "error", "no-line"),
    ("run-plan", _R.replace("json.dumps(line)", "str(line)"), "error", "str"),
    ("run-plan", _R_START.replace("    # 1. done", "    done = []\n    # 1. done"), "wrong", "dots-left"),
] + [("run-plan", code, "almost" if key in ("no-progress", "fixed-settings") else "wrong", key) for key, code in _R_WRONG.items()] + [
    ("results-at", _A_START, "empty", None),
    ("results-at", _A, "correct", None),
] + [("results-at", code, "wrong", key) for key, code in _A_WRONG.items()] + [
    ("budget", None, "error", None),
    ("budget", {"steps": None, "why": ""}, "empty", None),
    ("budget", {"steps": 12800.5, "why": _B_WHY["en"]}, "wrong", "steps"),
    ("budget", {"steps": 100, "why": _B_WHY["en"]}, "wrong", "steps"),
    ("budget", {"steps": 12800, "why": "the paper"}, "wrong", "why"),
    ("budget", {"steps": float("nan"), "why": _B_WHY["en"]}, "wrong", "steps"),
]


def demo(progress=None) -> dict | str | None:
    """Runs the learner's main result with their kit: their run_plan with their run, over the seeds of their plan and
    the ladder of its budget, into their experiment log work/projects/own/runs.jsonl. None when a part of the kit is
    not there yet; "budget" when the kit has no budget yet; "log" when a line of the log is not a run as run_plan writes
    it; "error" when the runner or the experiment stops with an error; otherwise the lines of the log."""
    folder = projects.folder() / "own"
    plan, budget = P.saved_plan(), saved_budget()
    if not all((folder / f"{m}.py").is_file() for m in ("runner", "experiment")) or plan is None:
        return None
    if budget is None:
        return "budget"
    if log_path().is_file() and log_path().read_text(encoding="utf-8").strip() and my_log() is None:
        return "log"
    projects.on_path()
    projects.forget()
    try:
        import importlib
        run_plan = importlib.import_module("own.runner").run_plan
        run = importlib.import_module("own.experiment").run
        run_plan(run, list(range(int(plan["seeds"]))), ladder(budget["steps"]), str(log_path()), progress)
    except Exception:
        return "error"
    finally:
        projects.forget()
    lines = my_log()
    return "log" if lines is None else lines


def my_verdict(lines: list[dict], plan: dict, steps: int) -> str | None:
    """What the learner's own rule, own/plan.py, says about their runs at the budget of their plan, read with their
    own/results.py; None when one of them is missing or stops with an error."""
    import importlib
    projects.on_path()
    projects.forget()
    try:
        results = importlib.import_module("own.results").results_at(lines, steps, plan)
        return importlib.import_module("own.plan").verdict(plan, results)
    except Exception:
        return None
    finally:
        projects.forget()


def main_table(lines: list[dict], plan: dict, steps: int, labels: dict, lang: str) -> str:
    """The runs at the budget of the plan as a table: a row per seed, the result of both settings and the gain the
    claim's way."""
    import html
    a, b = P._settings(plan)
    res = course_results_at(lines, steps, plan)
    pad = "padding:3px 14px 3px 0"
    head = "".join(f"<th style='{pad};text-align:right'>{html.escape(c)}</th>" for c in (labels["seed"], a, b, labels["gain"]))
    rows = ""
    for i, (x, y) in enumerate(zip(res[a], res[b])):
        gain = (x - y) if plan.get("direction") == "higher" else (y - x)
        cells = [str(i), short(x, lang), short(y, lang), short(round(gain, 4), lang)]
        rows += "<tr>" + "".join(f"<td style='{pad};text-align:right'>{html.escape(c.replace('-', '−'))}</td>" for c in cells) + "</tr>"
    return f"<table style='border-collapse:collapse;font-variant-numeric:tabular-nums'><tr>{head}</tr>{rows}</table>"
