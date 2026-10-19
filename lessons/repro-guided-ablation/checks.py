# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson repro-guided-ablation: grid, every combination of a few values of some settings, saved as
repro/grid.py; run_grid, the ablation runner that trains each combination with run_seed into an experiment log,
with the values it changes, and runs no combination twice, saved as repro/ablation.py; and the figure of the
moments t and v against the fraction of the table. The course's own ablation, 7 fractions with and without weight
decay, is in data/course-ablation.jsonl, made with run_seed of repro-guided-figure; the demo trains the learner's
grid of six with their kit."""
import functools
import importlib.util
import json
import re

from everything_ai import paths, projects
from everything_ai.check import ClassBox, FunctionBox, Mistake, PlotBox


def _checks(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _checks("repro-guided-figure", "everything_ai_checks_repro_guided_figure_for_ablation")
PAPER, FRACTION, STEPS, THRESHOLD, DELAY, number, read = G.PAPER, G.FRACTION, G.STEPS, G.THRESHOLD, G.DELAY, G.number, G.read
# The kit so far: where a module is missing or broken, the course's version stands in.
BUILDS_ON = G.BUILDS_ON + (("repro", "seeds", "repro-guided-figure", "run-seeds"), ("repro", "plan", "repro-guided-claim", "verdict"),
                           ("repro", "grid", "repro-guided-ablation", "grid"))
projects.builds_on(*BUILDS_ON)

CHANGES = {"weight_decay": [1.0, 0.0], "fraction": [0.3, 0.5, 0.9]}   # the learner's grid: six runs
FRACTIONS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]                        # the course's ablation, with and without weight decay
RUNS_FILE = paths.ROOT / "data" / "course-ablation.jsonl"
SECONDS_PER_STEP = G.SECONDS_PER_STEP


@functools.cache
def course_runs() -> list[dict]:
    """The course's ablation: run_seed of repro-guided-figure, seed 0, STEPS steps, the paper's settings with each
    weight decay of CHANGES and each of FRACTIONS; with weight decay first, each by fraction."""
    return [json.loads(line) for line in RUNS_FILE.read_text(encoding="utf-8").splitlines()]


def run(weight_decay: float, fraction: float) -> dict:
    return next(r for r in course_runs() if r["weight_decay"] == weight_decay and r["fraction"] == fraction)


def first_reached(weight_decay: float) -> float | None:
    """The smallest fraction at which validation reaches 99% within the budget, with this weight decay."""
    return next((f for f in FRACTIONS if read(run(weight_decay, f))["v"] is not None), None)


def val_picture(fraction: float, labels: dict) -> str:
    """Validation accuracy against the steps on a log axis, at one fraction of the table: with weight decay solid,
    without it dashed, both ink (accuracy is not what the model optimises); a thin line at 99%, and a dot at v."""
    from everything_ai import chart
    p = chart.Panel((1, STEPS), (0, 1.08), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"], xlog=True,
                    yticks=[0, 0.5, 1.0])
    p.line([1, STEPS], [THRESHOLD, THRESHOLD], "ink-3", 1.0)
    for wd, dashed in ((1.0, False), (0.0, True)):
        log = run(wd, fraction)
        p.line(log["step"], log["val"], "ink", 2.0, dashed=dashed)
        r = read(log)
        if r["v"] is not None:
            p.points([r["v"]], [log["val"][r["v_index"]]], "ink", r=4)
    for y, dashed, name in ((0.9, False, "with"), (0.78, True, "without")):   # a small legend at the top left, where
        p.line([1.5, 4], [y, y], "ink", 2.0, dashed=dashed)                    # validation is still near 1%
        p.text(5, y - 0.025, labels[name], "ink", "start")
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

# --- exercise 1: grid --------------------------------------------------------------------------------------------------

_G_HEAD = ("def grid(settings, changes):\n"
           "    # every combination of the values in changes, each a copy of settings with those values; the first key\n"
           "    # of changes changes slowest, as in a table read row by row\n"
           "    runs = [dict(settings)]\n    for key in changes:\n")
_G_BODY = ("        new = []\n        for run in runs:\n            for value in changes[key]:\n"
           "                one = dict(run)\n                one[key] = value\n                new.append(one)\n        runs = new\n")
_G_TAIL = "    return runs\n"
_G_START = _G_HEAD + "        # each run so far, once for every value of key\n        ...\n" + _G_TAIL
_G = _G_HEAD + _G_BODY + _G_TAIL
_G_SHARED = _G.replace("                one = dict(run)\n", "                one = run\n")
_G_ORDER = _G.replace("        for run in runs:\n            for value in changes[key]:\n", "        for value in changes[key]:\n            for run in runs:\n")
_G_NO_RESET = _G.replace("        runs = new\n", "")
_G_ONE_AT_A_TIME = ("def grid(settings, changes):\n"
                    "    # every combination of the values in changes, each a copy of settings with those values; the first key\n"
                    "    # of changes changes slowest, as in a table read row by row\n"
                    "    runs = [dict(settings)]\n    for key in changes:\n        for value in changes[key]:\n"
                    "            one = dict(settings)\n            one[key] = value\n            runs.append(one)\n" + _G_TAIL)

_G_SETTINGS = '{"lr": 0.001, "weight_decay": 1.0, "batch": 512}'

# --- exercise 2: run_grid ----------------------------------------------------------------------------------------------

_A_HEAD = ("import json\n\nfrom repro.grid import grid\nfrom repro.run import run_seed\n\n\n"
           "def run_grid(settings, changes, steps, path, device=\"cpu\", progress=None):\n"
           "    # one run of run_seed with seed 0 for each combination of grid(settings, changes), added to the experiment\n"
           "    # log at path with the values it changes; a combination that is in the log already is not run again\n"
           "    keys = list(changes)\n"
           "    with open(path, \"a\") as f:   # \"a\" makes the log when there is none yet, and keeps what is in it\n"
           "        f.write(\"\")\n"
           "    done = [[json.loads(line)[key] for key in keys] for line in open(path)]   # the values of each run in the log\n"
           "    for one in grid(settings, changes):\n")
_A_VALUES = "        values = [one[key] for key in keys]\n"
_A_IF = "        if values not in done:\n"
_A_RUN = "            log = run_seed(one, 0, steps, device, progress)\n"
_A_KEYS = "            for key in keys:\n                log[key] = one[key]\n"
_A_WRITE = "            with open(path, \"a\") as f:\n                f.write(json.dumps(log) + \"\\n\")\n"
_A_START = (_A_HEAD + "        # its values of the keys; when they are not done: its run, with those values added to the log,\n"
            "        # and the log as one line of JSON at the end of the file\n        ...\n")


def _a(values: str = _A_VALUES, test: str = _A_IF, run_: str = _A_RUN, keys: str = _A_KEYS, write: str = _A_WRITE) -> str:
    return _A_HEAD + values + test + run_ + keys + write


_A = _a()
_A_SAME = _a(run_=_A_RUN.replace("run_seed(one,", "run_seed(settings,"))
_A_NO_KEYS = _a(keys="")
_A_AGAIN = _a(test="        if True:\n")
_A_OVERWRITE = _a(write=_A_WRITE.replace("open(path, \"a\")", "open(path, \"w\")"))
_A_NO_DEVICE = _a(run_="            log = run_seed(one, 0, steps, progress=progress)\n")
_A_NO_LINE = _a(write=_A_WRITE.replace(" + \"\\n\"", ""))
_A_STR = _a(write=_A_WRITE.replace("json.dumps(log)", "str(log)"))

_A_FAKE = ("import tempfile\nfrom pathlib import Path\n\ncalls = []\n\n\n"
           "def run_seed(settings, seed, steps, device=\"cpu\", progress=None):   # a quick stand-in: no training\n"
           "    calls.append((settings[\"weight_decay\"], settings[\"fraction\"], seed, device))\n"
           "    log = {\"seed\": seed, \"step\": [1, steps], \"train\": [0.5, 1.0], \"val\": [0.0, settings[\"fraction\"]]}\n"
           "    if progress is not None:\n        progress(log)\n    return log\n\n\n"
           "path = str(Path(tempfile.mkdtemp()) / \"ablation.jsonl\")\n"
           "S = {\"p\": 97, \"weight_decay\": 1.0, \"fraction\": 0.3}\n")
def _read(*keys: str) -> str:
    """The calls and the lines of the log, each with its run and the values the grid changes: a log may keep more
    settings than those, such as with log.update(one)."""
    kept = repr(("seed", "step", "train", "val") + keys).replace("'", '"')
    return f"\n[calls, [{{k: v for k, v in json.loads(line).items() if k in {kept}}} for line in open(path)]]"

_A_SCENARIOS = [
    # the lesson's example: weight decay on and off at two fractions, into a new log
    _A_FAKE + "run_grid(S, {\"weight_decay\": [1.0, 0.0], \"fraction\": [0.3, 0.9]}, 10, path)" + _read("weight_decay", "fraction"),
    # a log that has one combination already: only the others are run, and its line stays
    _A_FAKE + "with open(path, \"w\") as f:\n    f.write(json.dumps({\"seed\": 0, \"step\": [1], \"train\": [1.0], \"val\": [1.0], \"fraction\": 0.5}) + \"\\n\")\n"
              "run_grid(S, {\"fraction\": [0.3, 0.5]}, 10, path)" + _read("fraction"),
    # the same grid twice: the second time every combination is done
    _A_FAKE + "run_grid(S, {\"fraction\": [0.5]}, 10, path)\nrun_grid(S, {\"fraction\": [0.5]}, 10, path)" + _read("fraction"),
    # device and progress go on to run_seed
    _A_FAKE + "seen = []\nrun_grid(S, {\"weight_decay\": [0.0]}, 7, path, \"cuda\", lambda log: seen.append(log[\"seed\"]))\n[seen, calls]",
]
_A_EN, _A_NL = "Your `run_grid` gives another experiment log than the course's", "Je `run_grid` geeft een ander logboek dan die van de course"

# --- exercise 3: the figure of t and v against the fraction --------------------------------------------------------------

_F_HEAD = ("import json\n\nimport matplotlib.pyplot as plt\n\nfrom repro.plan import THRESHOLD, first_step\n\n"
           "runs = [json.loads(line) for line in open(\"data/course-ablation.jsonl\")]   # the course's ablation\n"
           "fractions, t, v = [], [], []\n"
           "for log in runs:\n    if log[\"weight_decay\"] == 1.0:\n")
_F_BODY = ("        fractions.append(log[\"fraction\"])\n"
           "        t.append(first_step(log[\"step\"], log[\"train\"], THRESHOLD))\n"
           "        v.append(first_step(log[\"step\"], log[\"val\"], THRESHOLD))\n")
_F_LINES = "plt.plot(fractions, t, \"o-\")\nplt.plot(fractions, v, \"o--\")\n"
_F_SCALE = "plt.yscale(\"log\")\n"
_F_LEGEND = "plt.legend([\"training reaches 99%\", \"validation reaches 99%\"])   # names the first two lines\n"
_F_TITLE = "\"The smaller the fraction, the longer validation lags behind training\""
_F_X, _F_Y = "\"fraction of the table used for training\"", "\"step\""


def _f(body: str = _F_BODY, lines: str = _F_LINES, scale: str = _F_SCALE, title: str = _F_TITLE, x: str = _F_X, y: str = _F_Y) -> str:
    return _F_HEAD + body + lines + scale + _F_LEGEND + f"plt.title({title})\nplt.xlabel({x})\nplt.ylabel({y})\nplt.show()\n"


_F = _f()
_F_START = (_F_HEAD + "        # its fraction, and the first logged steps at which training and validation reach THRESHOLD\n        ...\n"
            "# t and v against the fraction, each a line with dots, on a log y-axis\n...\n"
            + _F_LEGEND + "plt.title(\"\")\nplt.xlabel(\"\")\nplt.ylabel(\"\")\nplt.show()\n")
_F_LINEAR = _f(scale="")
_F_ORDER = _f(lines="plt.plot(fractions, v, \"o--\")\nplt.plot(fractions, t, \"o-\")\n")
_F_ONLY_V = _f(lines="plt.plot(fractions, v, \"o--\")\n")
_F_PERCENT = _f(body=_F_BODY.replace("fractions.append(log[\"fraction\"])", "fractions.append(100 * log[\"fraction\"])"))
_F_INDEX = _f(body=_F_BODY.replace("first_step(log[\"step\"], log[\"train\"], THRESHOLD)", "log[\"train\"].index(1.0)")
                          .replace("first_step(log[\"step\"], log[\"val\"], THRESHOLD)", "log[\"val\"].index(1.0)"))


def _lines_of(got) -> list:
    return [c for c in (got or {}).get("charts", []) if c["kind"] in ("line", "scatter")]


@functools.cache
def _want_tv() -> tuple:
    rows = [read(r) | {"fraction": r["fraction"]} for r in course_runs() if r["weight_decay"] == 1.0]
    return [x["fraction"] for x in rows], [x["t"] for x in rows], [x["v"] for x in rows]


def _both_drawn(got) -> bool:
    fr, t, v = _want_tv()
    have = [(c["x"], c["y"]) for c in _lines_of(got)]
    return all(any(_close(x, fr) and _close(y, want) for x, y in have) for want in (t, v))


def _percent(got, text) -> bool:
    fr, t, v = _want_tv()
    return any(_close(c["x"], [100 * f for f in fr]) and (_close(c["y"], t) or _close(c["y"], v)) for c in _lines_of(got))


def _linear(got, text) -> bool:
    return _both_drawn(got) and got["yscale"] != "log"


def _first_two_wrong(got, text) -> bool:
    fr, t, v = _want_tv()
    lines = _lines_of(got)
    return (_both_drawn(got) and "plt.legend([" in text and len(lines) >= 2
            and not (_close(lines[0]["y"], t) and _close(lines[1]["y"], v)))


# --- exercise 4: what the ablation shows, a choice question in the lesson -------------------------------------------------

EXERCISES = {
    "grid": FunctionBox(
        name="grid",
        function="grid",
        start=_G_START,
        solution=_G,
        visible=[(eval(_G_SETTINGS), {"lr": [0.001, 0.003], "batch": [256, 512]})],
        hidden=[(eval(_G_SETTINGS), {}), (eval(_G_SETTINGS), {"lr": [0.003]}),
                (eval(_G_SETTINGS), {"weight_decay": [1.0, 0.0], "lr": [0.001, 0.003, 0.01]}),
                ({"a": 1, "b": 2, "c": 3}, {"c": [0, 1], "a": [5, 6], "b": [7, 8]})],
        save_as=("repro", "grid"),
        first=[_DOTS],
        mistakes=[
            Mistake("shared", wrong=_G_SHARED,
                    en="`{call}` gives `{got}`, and it should give `{want}`: some of your runs are one and the same dictionary, so setting the next value changes the runs before it too. Make a new copy for every value, inside the loop over the values: `one = dict(run)`, then `one[key] = value`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: een paar van je runs zijn een en dezelfde dictionary, dus verandert de volgende waarde ook de runs ervoor. Maak voor elke waarde een nieuwe kopie, binnen de loop over de waarden: `one = dict(run)`, dan `one[key] = value`."),
            Mistake("order", wrong=_G_ORDER,
                    en="`{call}` gives `{got}`, and it should give `{want}`: the same runs, in another order. Your loop over the values is the outer one, so the last key changes slowest. Loop over `runs` outside and over the values inside: then the first key changes slowest.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dezelfde runs, in een andere volgorde. Je loop over de waarden is de buitenste, dus verandert de laatste sleutel het langzaamst. Loop buiten over `runs` en binnen over de waarden: dan verandert de eerste sleutel het langzaamst."),
            Mistake("one-at-a-time", wrong=_G_ONE_AT_A_TIME,
                    en="`{call}` gives `{got}`, and it should give `{want}`: your runs change one setting at a time, from the start. A grid takes every combination: each run so far, once with every value of the next key.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je runs veranderen één instelling tegelijk, vanaf het begin. Een grid neemt elke combinatie: elke run tot nu toe, één keer met elke waarde van de volgende sleutel."),
            Mistake("no-reset", wrong=_G_NO_RESET,
                    en="`{call}` gives `{got}`, and it should give `{want}`: `runs` never becomes `new`, so each key starts again from the settings alone. After the loop over `runs`, put `runs = new`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: `runs` wordt nooit `new`, dus begint elke sleutel weer bij alleen de instellingen. Zet na de loop over `runs` de regel `runs = new`."),
        ],
        texts={
            **_KEEP,
            "empty": {"en": "Replace the `...`: a new list, and for each run so far a copy for every value of `key`, with that value; then `runs` becomes that new list.",
                      "nl": "Vervang de `...`: een nieuwe lijst, en voor elke run tot nu toe een kopie voor elke waarde van `key`, met die waarde; daarna wordt `runs` die nieuwe lijst."},
            "arguments": {"en": "The check calls `{call}`, with the settings and the changes, and the `def` line of your `grid` has another number of names between its parentheses. Keep `(settings, changes)`.",
                          "nl": "De check roept `{call}` aan, met de instellingen en de veranderingen, en de `def`-regel van je `grid` heeft een ander aantal namen tussen de haakjes. Houd `(settings, changes)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does each key make a new list from every run so far, once for every value, each a copy with that value, and does `runs` become that list?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Maakt elke sleutel een nieuwe lijst uit elke run tot nu toe, één keer voor elke waarde, elk een kopie met die waarde, en wordt `runs` die lijst?"},
            "almost": {"en": "Your `grid` gives the course's runs for the first example, but `{call}` gives `{got}`, and it should give `{want}`. Does it work for no changes, for one key, and for three?",
                       "nl": "Je `grid` geeft de runs van de course voor het eerste voorbeeld, maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij zonder veranderingen, met één sleutel, en met drie?"},
            "correct": {"en": "Your `grid` gives every combination of the values, in the order of a table read row by row. It is now in your project, as `work/projects/repro/grid.py`.",
                        "nl": "Je `grid` geeft elke combinatie van de waarden, in de volgorde van een tabel die je rij voor rij leest. Hij staat nu in je project, als `work/projects/repro/grid.py`."},
            "kept": {"en": "Your `grid` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `grid` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "run-grid": ClassBox(
        name="run-grid",
        start=_A_START,
        solution=_A,
        scenarios=_A_SCENARIOS,
        compare=_close,
        save_as=("repro", "ablation"),
        first=[_DOTS],
        mistakes=[
            Mistake("no-line", error=lambda e: type(e).__name__ == "JSONDecodeError" and "Extra data" in str(e),
                    en="Reading your experiment log back stops: two runs stand on one line, because each line needs a newline at its end. Write `json.dumps(log) + \"\\n\"`.",
                    nl="Je logboek teruglezen stopt: twee runs staan op één regel, omdat elke regel aan het eind een regeleinde nodig heeft. Schrijf `json.dumps(log) + \"\\n\"`."),
            Mistake("str", error=lambda e: type(e).__name__ == "JSONDecodeError" and "double quotes" in str(e),
                    en="Reading your experiment log back stops: a line of it is not JSON. It has single quotes, the way Python shows a dictionary, and JSON needs double quotes. Turn the log into a line with `json.dumps(log)`.",
                    nl="Je logboek teruglezen stopt: een regel ervan is geen JSON. Hij heeft enkele aanhalingstekens, zoals Python een dictionary laat zien, en JSON heeft dubbele nodig. Maak van het logboek een regel met `json.dumps(log)`."),
            Mistake("same-settings", wrong=_A_SAME,
                    en=_A_EN + ": every run trains with `settings`, so all of them are the same run. Train each combination: `run_seed(one, 0, steps, device, progress)`.",
                    nl=_A_NL + ": elke run traint met `settings`, dus zijn het allemaal dezelfde run. Train elke combinatie: `run_seed(one, 0, steps, device, progress)`."),
            Mistake("no-keys", wrong=_A_NO_KEYS,
                    en=_A_EN + ": its lines do not say which combination they are, so nothing can read the log back, and `done` stops at the next run. Add the values of the keys to the log before you write it: `log[key] = one[key]` for every key.",
                    nl=_A_NL + ": zijn regels zeggen niet welke combinatie ze zijn, dus kan niets het logboek teruglezen, en stopt `done` bij de volgende run. Voeg de waarden van de sleutels toe aan het logboek voordat je het schrijft: `log[key] = one[key]` voor elke sleutel."),
            Mistake("again", wrong=_A_AGAIN,
                    en=_A_EN + ": a combination that is in the log already is run again, and then stands in it twice. Run it only when its values are not done yet: `values = [one[key] for key in keys]`, a list in the order of `keys` as each item of `done` is, then `if values not in done:`.",
                    nl=_A_NL + ": een combinatie die al in het logboek staat, wordt opnieuw gedraaid, en staat er dan twee keer in. Draai hem alleen als zijn waarden nog niet klaar zijn: `values = [one[key] for key in keys]`, een lijst in de volgorde van `keys`, zoals elk item van `done`, en dan `if values not in done:`."),
            Mistake("overwrite", wrong=_A_OVERWRITE,
                    en=_A_EN + ": only the last run is in it. `open(path, \"w\")` empties the file first, so each run wipes out the one before. Add to it with `\"a\"`.",
                    nl=_A_NL + ": alleen de laatste run staat erin. `open(path, \"w\")` maakt het bestand eerst leeg, dus wist elke run de vorige uit. Voeg eraan toe met `\"a\"`."),
            Mistake("no-device", wrong=_A_NO_DEVICE, state="almost",
                    en="Your `run_grid` does not pass `device` on to `run_seed`, so it would train on the CPU even when the lesson gives it a GPU. Give it along: `run_seed(one, 0, steps, device, progress)`.",
                    nl="Je `run_grid` geeft `device` niet door aan `run_seed`, dus zou hij op de CPU trainen, ook als de les hem een GPU geeft. Geef hem mee: `run_seed(one, 0, steps, device, progress)`."),
        ],
        texts={
            **_KEEP,
            "empty": {"en": "Replace the `...`: the values of `one` for the keys; when they are not in `done`, its run with seed 0, those values added to the log, and the log as one line of JSON at the end of the file.",
                      "nl": "Vervang de `...`: de waarden van `one` voor de sleutels; als die niet in `done` staan, zijn run met seed 0, die waarden toegevoegd aan het logboek, en het logboek als één regel JSON aan het eind van het bestand."},
            "wrong": {"en": _A_EN + ". Does each combination run only when its values are not in `done`, with `one` as its settings, and is each log written with its values, as one line, with `\"a\"`?",
                      "nl": _A_NL + ". Draait elke combinatie alleen als zijn waarden niet in `done` staan, met `one` als instellingen, en wordt elk logboek met zijn waarden geschreven, als één regel, met `\"a\"`?"},
            "almost": {"en": "Your `run_grid` writes the course's experiment log for the first example. But with these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`. Does it skip a combination that was in the log before, and pass `device` and `progress` on to `run_seed`?",
                       "nl": "Je `run_grid` schrijft het logboek van de course voor het eerste voorbeeld. Maar met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn. Slaat hij een combinatie over die al in het logboek stond, en geeft hij `device` en `progress` door aan `run_seed`?"},
            "correct": {"en": "Your `run_grid` trains every combination into one experiment log, each line with the values it changes, and goes on where it stopped. It is now in your project, as `work/projects/repro/ablation.py`.",
                        "nl": "Je `run_grid` traint elke combinatie in één logboek, elke regel met de waarden die hij verandert, en gaat verder waar hij stopte. Hij staat nu in je project, als `work/projects/repro/ablation.py`."},
            "kept": {"en": "Your `run_grid` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `run_grid` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "figure": PlotBox(
        name="figure",
        start=_F_START,
        solution=_F,
        merge=True,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("percent", matches=_percent, state="almost",
                    en="Your fractions are percentages, 30 to 90, and the logs hold shares, 0.3 to 0.9. Both are fine to read, but this exercise draws the fractions as the logs give them: append `log[\"fraction\"]` as it is, and say in the label what it is.",
                    nl="Je delen van de tabel zijn percentages, 30 tot 90, en de logboeken geven ze als 0,3 tot 0,9. Allebei goed te lezen, maar deze oefening tekent de delen zoals de logboeken ze geven: voeg `log[\"fraction\"]` toe zoals hij is, en zeg in het label wat het is."),
            Mistake("linear", matches=_linear, state="almost",
                    en="Both lines are there, but on a normal y-axis the steps of training, a few hundred, lie flat along the bottom next to the thousands of validation. On a log y-axis both show, and how many times as late one is as the other: `plt.yscale(\"log\")`.",
                    nl="Beide lijnen staan erin, maar op een gewone y-as liggen de stappen van training, een paar honderd, plat langs de onderkant naast de duizenden van validatie. Op een logaritmische y-as zijn ze allebei te zien, en hoeveel keer zo laat de ene is als de andere: `plt.yscale(\"log\")`."),
            Mistake("order", matches=_first_two_wrong, state="almost",
                    en="Both lines are there, but the given legend names the first two things you draw: training, then validation. In your chart, those two are not `t` and then `v`. Draw `t` first and `v` second, each as one line with dots: `plt.plot(fractions, t, \"o-\")`.",
                    nl="Beide lijnen staan erin, maar de gegeven legenda noemt de eerste twee dingen die je tekent: training, dan validatie. In jouw grafiek zijn die twee niet `t` en dan `v`. Teken eerst `t` en dan `v`, elk als één lijn met stippen: `plt.plot(fractions, t, \"o-\")`."),
        ],
        texts={
            **_KEEP,
            "empty": {"en": "Replace the first `...` with three lines that append the fraction, `t` and `v` of each run with weight decay, with `first_step`. Replace the second with the two lines and the scale of the y-axis, then fill in the title and the labels.",
                      "nl": "Vervang de eerste `...` door drie regels die het deel van de tabel, `t` en `v` van elke run met weight decay toevoegen, met `first_step`. Vervang de tweede door de twee lijnen en de schaal van de y-as, en vul daarna de titel en de labels in."},
            "nothing": {"en": "I see no lines in your chart. Draw `t` and `v` against `fractions`, each with `plt.plot`.",
                        "nl": "Ik zie geen lijnen in je grafiek. Teken `t` en `v` tegen `fractions`, elk met `plt.plot`."},
            "kind": {"en": "Your chart is {kind}, and `t` and `v` are lines through the fractions: draw them with `{draw}`.",
                     "nl": "Je grafiek is {kind}, en `t` en `v` zijn lijnen door de delen van de tabel: teken ze met `{draw}`."},
            "values": {"en": "Not both lines of the course's figure are in your chart. Does the loop take, for each run with weight decay, its fraction and the first steps at which training and validation reach `THRESHOLD`, with `first_step(log[\"step\"], ..., THRESHOLD)`? And does it draw both `t` and `v` against `fractions`?",
                       "nl": "Niet beide lijnen van de figuur van de course staan in je grafiek. Neemt de loop voor elke run met weight decay zijn deel van de tabel en de eerste stappen waarop training en validatie `THRESHOLD` halen, met `first_step(log[\"step\"], ..., THRESHOLD)`? En tekent hij zowel `t` als `v` tegen `fractions`?"},
            "title": {"en": "The lines are right. Now the title: the one message of the figure, as a sentence. What happens to the gap between the lines as the fraction grows?",
                      "nl": "De lijnen kloppen. Nu de titel: de ene boodschap van de figuur, als zin. Wat gebeurt er met de afstand tussen de lijnen als het deel van de tabel groeit?"},
            "xlabel": {"en": "The lines are right. Now say along the x-axis what it shows: the fraction of the table used for training.",
                       "nl": "De lijnen kloppen. Zeg nu langs de x-as wat hij laat zien: het deel van de tabel dat voor training gebruikt wordt."},
            "ylabel": {"en": "The lines are right. Now say along the y-axis what it shows: the step at which each reaches 99%.",
                       "nl": "De lijnen kloppen. Zeg nu langs de y-as wat hij laat zien: de stap waarop elk 99% haalt."},
            "correct": {"en": "Your figure shows the moments of your plan against the fraction: training reaches 99% after a few hundred steps everywhere, and validation lags further behind the smaller the fraction.",
                        "nl": "Je figuur laat de momenten van je plan zien tegen het deel van de tabel: training haalt overal na een paar honderd stappen 99%, en validatie blijft verder achter naarmate dat deel kleiner is."},
        },
    ),
}

# The lesson is done when these are correct. "what-shows" is the choice question in the lesson.
REQUIRED = ["grid", "run-grid", "figure", "what-shows"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("grid", _G_START, "empty", None),
    ("grid", _G, "correct", None),
    ("grid", _G_SHARED, "wrong", "shared"),
    ("grid", _G_ORDER, "wrong", "order"),
    ("grid", _G_ONE_AT_A_TIME, "wrong", "one-at-a-time"),
    ("grid", _G_NO_RESET, "wrong", "no-reset"),
    ("run-grid", _A_START, "empty", None),
    ("run-grid", _A, "correct", None),
    ("run-grid", _a(keys="            log.update(one)\n"), "correct", None),   # all settings in the log: more than needed, and fine
    ("run-grid", _A_SAME, "wrong", "same-settings"),
    ("run-grid", _A_NO_KEYS, "wrong", "no-keys"),
    ("run-grid", _A_AGAIN, "wrong", "again"),
    ("run-grid", _A_OVERWRITE, "wrong", "overwrite"),
    ("run-grid", _A_NO_DEVICE, "almost", "no-device"),
    ("run-grid", _A_NO_LINE, "error", "no-line"),
    ("run-grid", _A_STR, "error", "str"),
    ("figure", _f(lines="plt.plot(fractions, t, \"-\")\nplt.scatter(fractions, t)\nplt.plot(fractions, v, \"--\")\nplt.scatter(fractions, v)\n"), "almost", "order"),
    ("grid", _G.replace("            for value in changes[key]:\n                one = dict(run)\n", "            one = dict(run)\n            for value in changes[key]:\n"), "wrong", "shared"),
    ("figure", _F_START, "empty", None),
    ("figure", _F, "correct", None),
    ("figure", _f(lines="plt.semilogy(fractions, t, \"o-\")\nplt.semilogy(fractions, v, \"o--\")\n", scale=""), "correct", None),
    ("figure", _F_LINEAR, "almost", "linear"),
    ("figure", _F_ORDER, "almost", "order"),
    ("figure", _F_PERCENT, "almost", "percent"),
    ("figure", _F_ONLY_V, "wrong", "values"),
    ("figure", _F_INDEX, "wrong", "values"),
    ("figure", _f(title="\"\""), "almost", "no-title"),
]


def demo(progress=None, steps: int = STEPS, changes: dict | None = None) -> dict | str | None:
    """Trains the learner's grid with their kit: their run_grid with their grid and run_seed, on their ModernGPT with
    their data and settings, into work/projects/repro/ablation.jsonl, on a GPU when there is one. None when
    repro/grid.py or repro/ablation.py is not in the project yet; "grid" when run_grid stops with an error, "model" when
    the model, the data or the settings do; otherwise the runs of the log, as from my_runs. Where a module it uses is
    missing or broken, the course's version stands in, and "stand_ins" names it."""
    folder = projects.folder() / "repro"
    if not (folder / "grid.py").is_file() or not (folder / "ablation.py").is_file():
        return None
    projects.on_path()
    projects.forget()
    stand = projects.use_course_versions()
    try:
        import importlib
        import torch
        try:
            run_grid = importlib.import_module("repro.ablation").run_grid
        except Exception:
            return "grid"
        try:
            settings = importlib.import_module("repro.settings").SETTINGS
            importlib.import_module("llm.modern_gpt")
            importlib.import_module("repro.data").dataset(settings["p"], settings["fraction"], 0)
        except Exception:
            return "model"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            run_grid(settings, changes or CHANGES, steps, str(folder / "ablation.jsonl"), device, progress)
        except Exception:
            return "grid"
        used = {"llm", "repro/data", "repro/settings", "repro/log", "repro/run", "repro/grid"}   # what training the grid uses
        return {"runs": my_runs(), "stand_ins": [f"{package}/{module}.py" for package, module, _, _ in stand
                                                 if package == "llm" or f"{package}/{module}" in used]}
    finally:
        projects.forget()


def my_runs() -> list[dict] | None:
    """The runs in the learner's log work/projects/repro/ablation.jsonl that have a weight decay and a fraction, with
    weight decay first, each by fraction; None when there is no log, or a line of it is not such a log."""
    path = projects.folder() / "repro" / "ablation.jsonl"
    if not path.is_file():
        return None
    try:
        runs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        ok = all(isinstance(r, dict) and len(r["step"]) == len(r["train"]) == len(r["val"]) >= 2
                 and isinstance(r["weight_decay"], (int, float)) and isinstance(r["fraction"], (int, float)) for r in runs)
    except (ValueError, KeyError, TypeError):
        return None
    return sorted(runs, key=lambda r: (-r["weight_decay"], r["fraction"])) if runs and ok else None
