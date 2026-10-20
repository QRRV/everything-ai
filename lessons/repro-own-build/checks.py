# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson repro-own-build: contract, which says in what way a run of an experiment breaks the contract of
the kit, saved as own/contract.py; and the learner's own experiment, run(setting, seed, steps), which the course's
contract holds to the contract and which is saved as own/experiment.py. The course's experiment for the dropout paper
is the solution, and its runs over three seeds at eight budgets are the play block."""
import importlib.util
import math
import re
import time
from dataclasses import dataclass

import torch  # noqa: F401  loaded here: a first import inside the time limit of a check would not finish in it

from everything_ai import paths
from everything_ai.check import (MISSING, PLAIN_ERRORS, TIME_LIMIT, CodeBox, FunctionBox, Mistake, Result, Run, Timeout, _fill,
                                 _plain_kind, _tb_frames, is_empty_text, run_code, with_time_limit)


def _checks(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _checks("repro-own-claim", "everything_ai_checks_repro_own_claim_for_build")
O = P.O
papers, saved_choice, saved_plan, plain, short, number = O.papers, O.saved_choice, P.saved_plan, O.plain, P.short, O.number
SETTINGS = ("claim", "other")   # the two settings of every experiment of the kit: the claim's, and the one it is compared with
STEPS = 20                      # the steps of a run the contract makes: enough to see that a run works, not the claim
RUN_LIMIT = 60.0                # seconds one run of the learner's experiment gets in the check
CHECK_LIMIT = 180.0             # seconds all the runs of one check get together
DEMO_LIMIT = 120.0              # seconds one run of the demo gets


# --- the course's experiment for the dropout paper ---------------------------------------------------------------------

E = ("import math\n\nimport numpy as np\nimport torch\nfrom torch import nn\n\n"
     "SETTINGS = {\"claim\": (0.2, 0.5), \"other\": (0.0, 0.0)}   # the dropout of the input and of the hidden layers\n"
     "K = 10               # the logits: a row of K numbers for each picture, one for each kind of clothing\n"
     "FROM_SCRATCH = True  # the model starts from random weights, not from a trained model\n\n\n"
     "def load():\n"
     "    # the training and the test pictures, as numbers from 0 to 1, and their labels\n"
     "    data = np.load(\"data/fashion-mnist-small.npz\")\n"
     "    x = torch.tensor(data[\"train_images\"], dtype=torch.float32) / 255\n"
     "    y = torch.tensor(data[\"train_labels\"], dtype=torch.long)\n"
     "    x_test = torch.tensor(data[\"test_images\"], dtype=torch.float32) / 255\n"
     "    y_test = torch.tensor(data[\"test_labels\"], dtype=torch.long)\n"
     "    return x, y, x_test, y_test\n\n\n"
     "def make_model(setting, seed):\n"
     "    # the model of this setting before any training; the seed sets its starting weights and, later, its dropout masks\n"
     "    p_in, p = SETTINGS[setting]   # an unknown setting stops here, with a KeyError\n"
     "    torch.manual_seed(seed)\n"
     "    return nn.Sequential(nn.Dropout(p_in), nn.Linear(784, 512), nn.ReLU(), nn.Dropout(p),\n"
     "                         nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p), nn.Linear(512, K))\n\n\n"
     "def one_batch():\n"
     "    # the first 32 training pictures and their labels: the batch the sanity checks learn by heart\n"
     "    x, y, _, _ = load()\n"
     "    return x[:32], y[:32]\n\n\n"
     "def run(setting, seed, steps):\n"
     "    # the test error in %, after `steps` steps of one run with this seed: \"claim\" with dropout, \"other\" without\n"
     "    x, y, x_test, y_test = load()\n"
     "    model = make_model(setting, seed)\n"
     "    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)\n"
     "    decay = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 0.5 * (1 + math.cos(math.pi * k / steps)))\n"
     "    batches = torch.Generator().manual_seed(seed)   # the order of the batches\n"
     "    model.train()\n"
     "    for step in range(steps):\n"
     "        rows = torch.randint(0, len(x), (100,), generator=batches)\n"
     "        loss = nn.functional.cross_entropy(model(x[rows]), y[rows])\n"
     "        opt.zero_grad()\n"
     "        loss.backward()\n"
     "        opt.step()\n"
     "        decay.step()\n"
     "    model.eval()   # no dropout when it is tested\n"
     "    with torch.no_grad():\n"
     "        wrong = (model(x_test).argmax(1) != y_test).float().mean().item()\n"
     "    return round(100 * wrong, 1)\n")

SEEDS = (0, 1, 2)                                        # the seeds of the course's plan for the dropout paper
BUDGETS = (100, 200, 400, 800, 1600, 3200, 6400, 12800)  # the steps of the course's runs: each a run of its own, with its own decay
AT_LEAST = 0.5   # the course's plan for the dropout paper: lower by at least 0.5 points of test error, in the median

# The course's runs of E, with data/tools/measure_dropout.py on one thread: the test error in % with dropout ("claim")
# and without ("other"), seeds 0, 1 and 2, after each budget of steps. And seed 0 after STEPS steps, for the guess.
RUNS = {
    100: {"claim": [28.0, 26.6, 27.6], "other": [24.6, 23.6, 23.8]},
    200: {"claim": [21.4, 21.2, 20.8], "other": [17.7, 17.4, 17.7]},
    400: {"claim": [16.4, 17.5, 16.6], "other": [15.5, 15.3, 15.5]},
    800: {"claim": [14.5, 15.1, 15.5], "other": [14.9, 14.1, 14.2]},
    1600: {"claim": [13.7, 13.6, 14.5], "other": [14.1, 13.9, 14.1]},
    3200: {"claim": [13.2, 13.7, 13.9], "other": [15.0, 14.1, 14.4]},
    6400: {"claim": [13.5, 13.1, 13.5], "other": [13.9, 13.6, 13.5]},
    12800: {"claim": [12.5, 12.9, 12.4], "other": [12.9, 13.6, 13.6]},
}
SHORT_RUN = {"claim": 66.9, "other": 53.4}


def whole(n: int, lang: str) -> str:
    """A whole number as the text writes it: four digits without a separator, five and more with one, as the slider."""
    return number(n, 0, lang) if abs(n) >= 10000 else str(n)


def gains(steps: int) -> list[float]:
    """How many points lower the test error is with dropout than without, for each seed, after `steps` steps."""
    r = RUNS[steps]
    return [round(b - a, 1) for a, b in zip(r["claim"], r["other"])]


def median_gain(steps: int) -> float:
    return sorted(gains(steps))[len(SEEDS) // 2]


def supported(steps: int) -> bool:
    """What the rule of the course's plan says about the runs of `steps` steps."""
    return median_gain(steps) >= AT_LEAST


def gains_picture(k: int, labels: dict) -> str:
    """The gain of each seed, a dot, for every budget, on a log axis of steps, with the median of each budget as a ring
    and a line through the medians; BUDGETS[k], the slider's, gets a thin line up from the axis and a larger ring. A dashed line
    at AT_LEAST, and a thin one at 0. Ink: differences between runs, nothing a model learns. Every budget shows from the
    start, so the picture says from which budget on dropout leads before anything is touched."""
    from everything_ai import chart
    shown = BUDGETS
    p = chart.Panel((70, 18000), (-4.5, 2.5), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"], xlog=True,
                    yticks=[-4, -2, 0, 2])
    p.line([70, 18000], [0, 0], "ink-3", 1.0)
    p.line([70, 18000], [AT_LEAST, AT_LEAST], "ink", 1.6, dashed=True)
    p.text(80, AT_LEAST + 0.4, labels["line"], "ink", "start")   # top left, where the gains are all below 0
    p.line([BUDGETS[k]] * 2, [-4.5, median_gain(BUDGETS[k]) - 0.3], "ink-3", 1.0)   # from the axis up to its ring
    p.points([BUDGETS[k]], [median_gain(BUDGETS[k])], "ink", r=10, hollow=True)
    p.line(list(shown), [median_gain(s) for s in shown], "ink", 1.6)
    for s in shown:
        p.points([s], [median_gain(s)], "ink", r=7, hollow=True)
        p.points([s] * len(SEEDS), gains(s), "ink", r=3.5)
    return chart.figure([p], alt=labels["alt"], width=360, height=260, drag=0)


# --- exercise 1: contract ---------------------------------------------------------------------------------------------

class FakeRun:
    """A made-up run for the cases of the contract: quick, wrong in one way at most, which its name says. A case shows
    it by its name, and gets the run itself, not a copy: a run with its own luck keeps counting."""

    def __init__(self, name: str, fn):
        self.name, self.fn = name, fn

    def __call__(self, setting, seed, steps):
        return self.fn(setting, seed, steps)

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return self


def _value(setting, seed, steps):
    """A made-up test error that every argument changes, for the settings of the kit only."""
    if setting not in SETTINGS:
        raise ValueError(f"unknown setting {setting!r}")
    return round(10 + 0.3 * seed + (1.0 if setting == "claim" else 0.0) + 50 / steps, 4)


def _own_luck():
    calls = [0]

    def fn(setting, seed, steps):
        calls[0] += 1
        return _value(setting, seed, steps) + 0.1 * calls[0]   # something random that the seed does not set
    return fn


def _from_a_dict(setting, seed, steps):
    p = {"claim": 0.5, "other": 0.0}[setting]   # a KeyError for any other setting
    return round(10 + 0.3 * seed - 2 * p + 50 / steps, 4)


FAKES = {
    "good_run": _value,
    "run_without_seed": lambda setting, seed, steps: _value(setting, 0, steps),
    "run_without_steps": lambda setting, seed, steps: _value(setting, seed, 100),
    "run_without_setting": lambda setting, seed, steps: _value("claim" if setting in SETTINGS else setting, seed, steps),
    "run_with_nan": lambda setting, seed, steps: float("nan"),
    "run_with_its_own_luck": _own_luck(),
    "run_from_a_dict": _from_a_dict,
    "run_counting_mistakes": lambda setting, seed, steps: int(10 * _value(setting, seed, steps)),
    "run_with_nothing": lambda setting, seed, steps: round(10 + 50 / steps, 4),   # neither its seed nor its setting
}


def fake(name: str) -> FakeRun:
    return FakeRun(name, FAKES[name])


_C_HEAD = ("import math\n\n\n"
           "def contract(run, steps):\n"
           "    # the first way in which `run` breaks the contract of the kit, or None when it keeps it:\n"
           "    # \"not a number\"          run(\"claim\", 0, steps) gives back NaN\n"
           "    # \"not the same twice\"    the same call again gives another number\n"
           "    # \"steps do nothing\"      twice the steps gives the same number\n"
           "    # \"seed does nothing\"     seed 1 gives the same number as seed 0\n"
           "    # \"setting does nothing\"  \"other\" gives the same number as \"claim\"\n")
_C_NUMBER = ("    a = run(\"claim\", 0, steps)\n"
             "    if math.isnan(a):\n"
             "        return \"not a number\"\n")
_C_REST = ("    if run(\"claim\", 0, steps) != a:\n"
           "        return \"not the same twice\"\n"
           "    if run(\"claim\", 0, 2 * steps) == a:\n"
           "        return \"steps do nothing\"\n"
           "    if run(\"claim\", 1, steps) == a:\n"
           "        return \"seed does nothing\"\n"
           "    if run(\"other\", 0, steps) == a:\n"
           "        return \"setting does nothing\"\n"
           "    return None\n")
C_CODE = _C_HEAD + _C_NUMBER + _C_REST
_C_START = _C_HEAD + "    ...\n"

_C_NAN = C_CODE.replace("    if math.isnan(a):\n        return \"not a number\"\n", "")
_C_SAME_SEED = C_CODE.replace("run(\"claim\", 1, steps) == a", "run(\"claim\", 0, steps) == a")
_C_SAME_STEPS = C_CODE.replace("run(\"claim\", 0, 2 * steps) == a", "run(\"claim\", 0, steps) == a")
_C_ORDER = _C_HEAD + _C_NUMBER + _C_REST.replace(
    "    if run(\"claim\", 1, steps) == a:\n        return \"seed does nothing\"\n    if run(\"other\", 0, steps) == a:\n        return \"setting does nothing\"\n",
    "    if run(\"other\", 0, steps) == a:\n        return \"setting does nothing\"\n    if run(\"claim\", 1, steps) == a:\n        return \"seed does nothing\"\n")

_VISIBLE = (fake("run_without_seed"), STEPS)
_HIDDEN = [(fake(n), STEPS) for n in ("good_run", "run_without_steps", "run_without_setting", "run_with_nan",
                                      "run_with_its_own_luck", "run_from_a_dict", "run_counting_mistakes", "run_with_nothing")]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

EXERCISES = {
    "contract": FunctionBox(
        name="contract",
        function="contract",
        start=_C_START,
        solution=C_CODE,
        visible=[_VISIBLE],
        hidden=_HIDDEN,
        python_values=True,
        save_as=("own", "contract"),
        first=[_DOTS],
        mistakes=[
            Mistake("nan", wrong=_C_NAN,
                    en="`{call}` gives `{got}`, and it should give `{want}`: that run gives back NaN, which is a float, but not a number you can compare: NaN is not even equal to itself. Check it with `math.isnan(a)` before anything else.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: die run geeft NaN terug, wat wel een float is, maar geen getal dat je kunt vergelijken: NaN is niet eens gelijk aan zichzelf. Controleer het met `math.isnan(a)`, vóór al het andere."),
            Mistake("order", wrong=_C_ORDER,
                    en="`{call}` gives `{got}`, and it should give `{want}`: that run uses neither its seed nor its setting, and the contract looks at the seed first. Keep the order of the comments, so that the first answer is the first thing to fix.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: die run gebruikt zijn seed noch zijn setting, en het contract kijkt eerst naar de seed. Houd de volgorde van het commentaar aan, zodat het eerste antwoord het eerste is om te repareren."),
            Mistake("same-steps", wrong=_C_SAME_STEPS,
                    en="`{call}` gives `{got}`, and it should give `{want}`: to see whether the steps do something, call the run with twice the steps, `2 * steps`. With the same steps, it gives the same number when it is the same twice.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: om te zien of de stappen iets doen, roep je de run aan met twee keer zoveel stappen, `2 * steps`. Met dezelfde stappen geeft hij hetzelfde getal als hij twee keer hetzelfde is."),
            Mistake("same-seed", wrong=_C_SAME_SEED,
                    en="`{call}` gives `{got}`, and it should give `{want}`: to see whether the seed does something, compare seed 0 with seed 1. You call seed 0 again, and a run that keeps its contract gives the same number for it.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: om te zien of de seed iets doet, vergelijk je seed 0 met seed 1. Jij roept seed 0 nog eens aan, en een run die zijn contract houdt, geeft daarvoor hetzelfde getal."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with your code: call the run, and give back the first of the five answers that holds, or `None`.",
                      "nl": "Vervang de `...` door je eigen code: roep de run aan, en geef het eerste van de vijf antwoorden terug dat klopt, of `None`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with a run and a number of steps, and the `def` line of your `contract` has another number of names between its parentheses. Keep `(run, steps)`.",
                          "nl": "De check roept `{call}` aan, met een run en een aantal stappen, en de `def`-regel van je `contract` heeft een ander aantal namen tussen de haakjes. Houd `(run, steps)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you try the five ways in the order of the comments, each with the calls it names, and give back the first that holds?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Probeer je de vijf manieren in de volgorde van het commentaar, elk met de aanroepen die het noemt, en geef je de eerste terug die klopt?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your contract hold for every run: one that keeps it, and one that breaks it in each of the five ways?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Klopt je contract voor elke run: een die het houdt, en een die het op elk van de vijf manieren breekt?"},
            "correct": {"en": "Your contract is right for all {n} runs: three that keep it, one of them counting whole mistakes, and six that break it, one with NaN and one in two ways at once. It is in your kit now, as `{saved}`: the next lessons hold every run to it.",
                        "nl": "Je contract klopt voor alle {n} runs: drie die het houden, waarvan één hele fouten telt, en zes die het breken, een met NaN en een op twee manieren tegelijk. Het zit nu in je kit, als `{saved}`: de volgende lessen houden elke run eraan."},
            "kept": {"en": "Your contract is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je contract is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}


# --- exercise 2: your experiment --------------------------------------------------------------------------------------

def _number(x) -> bool:
    """One Python number, as the kit takes it: an int or a float, not a bool, a string, a tensor or a NumPy number."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _shown(x) -> str:
    """What a run gave back, the way a message shows it in code: numbers without float noise, also inside a tuple or a
    list, and anything else as Python shows it."""
    if _number(x):
        return "nan" if math.isnan(x) else f"{x:.4f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x)
    if isinstance(x, (tuple, list)) and len(x) <= 4:
        inner = ", ".join(_shown(v) for v in x) + ("," if isinstance(x, tuple) and len(x) == 1 else "")
        return f"({inner})" if isinstance(x, tuple) else f"[{inner}]"
    return " ".join(repr(x).split())[:80]


def _said(x: float) -> dict:
    """A number in a sentence: with a decimal comma in Dutch."""
    return {"en": _shown(x), "nl": _shown(x).replace(".", ",")}


class _NoNumber(Exception):
    """A run of the contract gave back something other than one Python number."""

    def __init__(self, call: str, value):
        super().__init__(call)
        self.call, self.value = call, value


@dataclass
class ExperimentBox(CodeBox):
    """The learner's experiment in a code box: the check runs the box and holds its `run` to the contract of the kit,
    in the order of contract, with STEPS steps, each run within RUN_LIMIT seconds. Once it keeps the contract, it is
    saved as own/experiment.py, and the message gives the numbers of seed 0 in both settings."""

    def check(self, text, inputs=None) -> Result:
        if text is MISSING or text is None:
            return Result("error", self.message("missing"))
        if is_empty_text(text, self.start):
            cleared = text.strip() != self.start.strip() and "cleared" in self.texts
            return Result("empty", self.message("cleared" if cleared else "empty"))
        for m in self.first:
            if m.matches(None, text):
                return Result(m.state, m.text, mistake=m.key)
        ran = run_code(text)
        if ran.timeout:   # the lines outside `run`: run_code gives them TIME_LIMIT
            return Result("slow", self.message("slow-top", s=f"{TIME_LIMIT:g}"))
        if ran.error is not None:
            return Result("error", self._plain_error(ran, text))
        fn = ran.names.get("run")
        if not callable(fn):
            return self._mistake("no-run")
        missing = missing_parts(ran.names)
        if missing:
            return self._mistake("no-parts", missing=", ".join(f"`{name}`" for name in missing))

        now = [""]   # the call of the contract that runs
        began = time.monotonic()

        def call(setting, seed, steps):
            now[0] = f'run("{setting}", {seed}, {steps})'
            left = CHECK_LIMIT - (time.monotonic() - began)   # every run gets a minute, and all of them three
            if left <= 0:
                raise Timeout()
            x = with_time_limit(lambda: fn(setting, seed, steps), min(RUN_LIMIT, left))
            if not _number(x) or math.isnan(x):
                raise _NoNumber(now[0], x)
            return x

        try:
            a = call("claim", 0, STEPS)
            again = call("claim", 0, STEPS)
            if again != a:
                return self._mistake("not-the-same-twice", a=_said(a), b=_said(again))
            if call("claim", 0, 2 * STEPS) == a:
                return self._mistake("steps-do-nothing", a=_said(a))
            if call("claim", 1, STEPS) == a:
                return self._mistake("seed-does-nothing", a=_said(a))
            other = call("other", 0, STEPS)
            if other == a:
                return self._mistake("setting-does-nothing", a=_said(a))
        except _NoNumber as no:
            item = hasattr(no.value, "item") and not isinstance(no.value, (str, bytes))   # a tensor or a NumPy number
            return self._mistake("item" if item else "not-a-number", call=no.call, got=_shown(no.value))
        except Timeout:
            return Result("slow", self.message("slow", call=now[0], steps=STEPS))
        except Exception as e:
            if isinstance(e, KeyError) and e.args and e.args[0] in SETTINGS:   # the dict has other keys, such as the plan's names
                return self._mistake("setting-keys", call=now[0], name=e.args[0])
            frames = [fr for fr in _tb_frames(e) if fr[0] == "<your code>"]
            plain = self._plain_error(Run(error=e, line=frames[-1][1] if frames else None), text)
            if _plain_kind(e) == "other":   # nothing to explain but the line
                plain = {lang: t.replace(PLAIN_ERRORS["other"][lang], "", 1).strip() for lang, t in plain.items()}
            return Result("error", {lang: (self.texts["stops"][lang].format(call=now[0]) + " " + plain[lang]).strip() for lang in plain})
        try:
            got = with_time_limit(lambda: fn("both", 0, STEPS), RUN_LIMIT)
        except Timeout:   # it does not stop with an error, it runs: the time limit cannot tell what
            return Result("slow", self.message("slow", call='run("both", 0, ' + str(STEPS) + ")", steps=STEPS))
        except Exception:   # it stops with an error, so a typo cannot run another experiment
            return self._correct(text, a=short(a, "en"), b=short(other, "en"), a_nl=short(a, "nl"), b_nl=short(other, "nl"))
        return self._mistake("unknown-setting-runs", got=_shown(got))

    def _mistake(self, key: str, **values) -> Result:
        m = next(m for m in self.mistakes if m.key == key)
        return Result(m.state, _fill(m.text, steps=STEPS, **values), mistake=key)

    def _correct(self, text: str, **values) -> Result:
        from everything_ai import projects
        path, kept = projects.save(*self.save_as, text)
        key = "kept" if kept else "correct"
        t = self.texts[key]
        shown = projects.shown(path)
        return Result("correct", {"en": t["en"].format(saved=shown, steps=STEPS, a=values["a"], b=values["b"]),
                                  "nl": t["nl"].format(saved=shown, steps=STEPS, a=values["a_nl"], b=values["b_nl"])})


_E_START = ("SETTINGS = {\"claim\": ..., \"other\": ...}   # what differs between the two settings of your plan\n"
            "K = ...              # the logits: a row of K numbers for each example, or for each place of a text\n"
            "FROM_SCRATCH = ...   # True when the model starts from random weights, False when from a trained one\n\n\n"
            "def make_model(setting, seed):\n"
            "    # the model of this setting before any training, its starting weights set by the seed\n"
            "    what = SETTINGS[setting]   # an unknown setting stops here, with a KeyError\n"
            "    ...\n\n\n"
            "def one_batch():\n"
            "    # one small batch of the training data, x and y: the sanity checks learn it by heart\n"
            "    ...\n\n\n"
            "def run(setting, seed, steps):\n"
            "    # one run of the experiment of your plan: `steps` steps with this seed, in setting \"claim\" or \"other\",\n"
            "    # with the model of make_model; it gives back the metric of your plan, one number\n"
            "    ...\n")
_E_WRONG = {
    "not-a-number": E.replace("    return round(100 * wrong, 1)\n", "    return round(100 * wrong, 1), loss.item()\n"),
    "not-the-same-twice": E.replace("    torch.manual_seed(seed)\n", ""),
    "steps-do-nothing": E.replace("    model = make_model(setting, seed)\n", "    model = make_model(setting, seed)\n    steps = 30\n"),
    "seed-does-nothing": E.replace("torch.manual_seed(seed)", "torch.manual_seed(0)").replace("Generator().manual_seed(seed)", "Generator().manual_seed(0)"),
    "setting-does-nothing": E.replace("SETTINGS[setting]", "SETTINGS[\"claim\"]"),
    "unknown-setting-runs": E.replace("SETTINGS[setting]   # an unknown setting stops here, with a KeyError", "SETTINGS.get(setting, (0.0, 0.0))"),
    "item": E.replace("    return round(100 * wrong, 1)\n", "    return 100 * (model(x_test).argmax(1) != y_test).float().mean()\n"),
}
assert all(code != E for code in _E_WRONG.values())   # every wrong version is another code
_E_WRONG["setting-keys"] = E.replace('SETTINGS = {"claim": (0.2, 0.5), "other": (0.0, 0.0)}', 'SETTINGS = {"with dropout": (0.2, 0.5), "without dropout": (0.0, 0.0)}')
_E_WRONG["no-run"] = E.replace("def run(setting, seed, steps):", "def experiment(setting, seed, steps):")
_E_WRONG["no-parts"] = E.replace("def one_batch():", "def first_batch():")
PARTS = ("make_model", "one_batch", "K", "FROM_SCRATCH")   # what the sanity checks of the next lesson take from the experiment


def missing_parts(names: dict) -> list[str]:
    """The parts of PARTS an experiment does not have, or not of their kind: two functions, a whole number from 2 on,
    and True or False."""
    k, scratch = names.get("K"), names.get("FROM_SCRATCH")
    kinds = {"make_model": callable(names.get("make_model")), "one_batch": callable(names.get("one_batch")),
             "K": isinstance(k, int) and not isinstance(k, bool) and k >= 2, "FROM_SCRATCH": isinstance(scratch, bool)}
    return [name for name in PARTS if not kinds[name]]

EXERCISES["experiment"] = ExperimentBox(
    name="experiment",
    start=_E_START,
    solution=E,
    save_as=("own", "experiment"),
    first=[_DOTS],
    mistakes=[
        Mistake("no-run", wrong=_E_WRONG["no-run"],
                en="Your box makes no function `run`. Keep the line `def run(setting, seed, steps):` of the start, with that name: the next lessons call it.",
                nl="Je vak maakt geen functie `run`. Houd de regel `def run(setting, seed, steps):` van het begin, met die naam: de volgende lessen roepen hem aan."),
        Mistake("no-parts", wrong=_E_WRONG["no-parts"],
                en="Your box has a `run`, but not yet everything the sanity checks of the next lesson take from it: {missing}. Keep them as the start has them: `make_model(setting, seed)` and `one_batch()` as functions, `K` as a whole number from 2 on, and `FROM_SCRATCH` as `True` or `False`.",
                nl="Je vak heeft een `run`, maar nog niet alles wat de sanity checks van de volgende les eruit halen: {missing}. Houd ze zoals het begin ze heeft: `make_model(setting, seed)` en `one_batch()` als functies, `K` als een heel getal vanaf 2, en `FROM_SCRATCH` als `True` of `False`."),
        Mistake("not-a-number", wrong=_E_WRONG["not-a-number"],
                en="`{call}` gives back `{got}`, and the contract wants one number: the metric of your plan, such as the test error in %. Give back that number alone; a loss or a second number goes in the log of a later lesson. If it is `nan`, your training breaks down within {steps} steps, for instance with a learning rate that is too high.",
                nl="`{call}` geeft `{got}` terug, en het contract wil één getal: de metric van je plan, zoals de testfout in %. Geef alleen dat getal terug; een loss of een tweede getal gaat in de log van een latere les. Is het `nan`, dan loopt je training binnen {steps} stappen vast, bijvoorbeeld door een te hoge learning rate."),
        Mistake("not-the-same-twice", wrong=_E_WRONG["not-the-same-twice"],
                en="`run(\"claim\", 0, {steps})` gives {a}, and called again {b}. Does something random in your run not follow the seed? Seed it at the start of the run with `torch.manual_seed(seed)`, and give the order of the batches a generator of its own, `torch.Generator().manual_seed(seed)`. Or does a run change something the next one starts from, such as a model or data made outside `run`? Make those inside `run`. On an NVIDIA card, some fast operations also add numbers up in a changing order: `torch.use_deterministic_algorithms(True)` stops that, and the PyTorch documentation *Reproducibility* says what else it needs.",
                nl="`run(\"claim\", 0, {steps})` geeft {a}, en nog eens aangeroepen {b}. Volgt iets willekeurigs in je run de seed niet? Leg het vast aan het begin van de run met `torch.manual_seed(seed)`, en geef de volgorde van de batches een eigen generator, `torch.Generator().manual_seed(seed)`. Of verandert een run iets waar de volgende mee begint, zoals een model of data die je buiten `run` maakt? Maak die binnen `run`. Op een NVIDIA-kaart tellen sommige snelle operaties bovendien getallen op in een wisselende volgorde: `torch.use_deterministic_algorithms(True)` stopt dat, en de PyTorch-documentatie *Reproducibility* zegt wat het verder nodig heeft."),
        Mistake("steps-do-nothing", wrong=_E_WRONG["steps-do-nothing"],
                en="With {steps} steps and with twice as many, your run gives the same number, {a}. Does it train for `steps` steps, and not for a number of its own?",
                nl="Met {steps} stappen en met twee keer zoveel geeft je run hetzelfde getal, {a}. Traint hij `steps` stappen, en niet een eigen aantal?"),
        Mistake("seed-does-nothing", wrong=_E_WRONG["seed-does-nothing"],
                en="Seeds 0 and 1 give the same number, {a}. Does your run take its randomness from `seed`, and not from a number of its own, such as `torch.manual_seed(0)`?",
                nl="Seed 0 en 1 geven hetzelfde getal, {a}. Haalt je run zijn willekeur uit `seed`, en niet uit een eigen getal, zoals `torch.manual_seed(0)`?"),
        Mistake("setting-does-nothing", wrong=_E_WRONG["setting-does-nothing"],
                en="The settings `\"claim\"` and `\"other\"` give the same number, {a}. Does the one thing your plan changes between them depend on `setting`?",
                nl="De settings `\"claim\"` en `\"other\"` geven hetzelfde getal, {a}. Hangt het ene ding dat je plan tussen de twee verandert af van `setting`?"),
        Mistake("setting-keys", wrong=_E_WRONG["setting-keys"],
                en="`{call}` stops with a KeyError: your dict has no key `\"{name}\"`. The kit calls your run with the settings `\"claim\"` and `\"other\"`, whatever your plan calls them: make those two the keys of your dict.",
                nl="`{call}` stopt met een KeyError: je dict heeft geen key `\"{name}\"`. De kit roept je run aan met de settings `\"claim\"` en `\"other\"`, hoe je plan ze ook noemt: maak die twee de keys van je dict."),
        Mistake("item", wrong=_E_WRONG["item"],
                en="`{call}` gives back `{got}`: a number inside a tensor or a NumPy array, not a Python number. The lessons after this one take one Python number from your run: give back its `.item()`, as `accuracy` did in Guided reproduction 4.",
                nl="`{call}` geeft `{got}` terug: een getal in een tensor of een NumPy-array, geen Python-getal. De lessen hierna nemen één Python-getal uit je run: geef de `.item()` ervan terug, zoals `accuracy` deed in Begeleide reproductie 4."),
        Mistake("unknown-setting-runs", wrong=_E_WRONG["unknown-setting-runs"],
                en="`run(\"both\", 0, {steps})` runs, and gives back `{got}`: a setting with a typo would then run another experiment without a warning. Look the setting up in a dict, as the start of the box does: `SETTINGS[setting]` stops on any other setting with a KeyError.",
                nl="`run(\"both\", 0, {steps})` draait, en geeft `{got}` terug: een setting met een typefout zou dan zonder waarschuwing een ander experiment draaien. Zoek de setting op in een dict, zoals het begin van het vak doet: `SETTINGS[setting]` stopt bij elke andere setting met een KeyError."),
    ],
    texts={
        "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                    "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
        "empty": {"en": "Replace the `...` with your experiment: the part both settings share, the one thing that differs, and the metric of your plan as one number.",
                  "nl": "Vervang de `...` door je experiment: het deel dat beide settings delen, het ene ding dat verschilt, en de metric van je plan als één getal."},
        "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                    "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
        "slow": {"en": "`{call}` is not done after a minute. Does your run train for `steps` steps, and not for a number of its own? A big model or dataset takes time to load too: start with a small one.",
                 "nl": "`{call}` is na een minuut nog niet klaar. Traint je run `steps` stappen, en niet een eigen aantal? Ook een groot model of een grote dataset laden kost tijd: begin met een kleine."},
        "slow-top": {"en": "The lines of your box outside `run` are not done after {s} seconds, all the time the check gives them. Load data and models inside `run`: there each run gets a minute.",
                     "nl": "De regels van je vak buiten `run` zijn na {s} seconden nog niet klaar, en meer tijd geeft de check ze niet. Laad data en modellen binnen `run`: daar krijgt elke run een minuut."},
        "stops": {"en": "`{call}`, one of the runs of the contract, stops with an error.",
                  "nl": "`{call}`, een van de runs van het contract, stopt met een fout."},
        "correct": {"en": "Your experiment keeps the contract of the kit: one number, the same twice, the steps, the seed and the setting each change it, and an unknown setting stops it. With {steps} steps, seed 0 gives {a} for \"claim\" and {b} for \"other\": your code runs, and that is all a run this short says. It is in your kit now, as `{saved}`.",
                    "nl": "Je experiment houdt het contract van de kit: één getal, twee keer hetzelfde, de stappen, de seed en de setting veranderen het elk, en een onbekende setting stopt het. Met {steps} stappen geeft seed 0 {a} voor \"claim\" en {b} voor \"other\": je code draait, en meer zegt een run die zo kort is niet. Het staat nu in je kit, als `{saved}`."},
        "kept": {"en": "Your experiment keeps the contract. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                 "nl": "Je experiment houdt het contract. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
    },
)

# The lesson is done when these are correct. "why-short" is the open question.
REQUIRED = ["contract", "experiment", "why-short"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("contract", _C_START, "empty", None),
    ("contract", C_CODE, "correct", None),
    ("contract", _C_HEAD + _C_NUMBER.replace("    if math.isnan(a):\n", "    if a != a:   # NaN is the only number not equal to itself\n") + _C_REST, "correct", None),
    ("contract", _C_HEAD + _C_NUMBER + _C_REST.replace("    return None\n", ""), "correct", None),   # no return at the end is None too
    ("experiment", _E_START, "empty", None),
    ("experiment", E, "correct", None),
    ("experiment", E.replace("    p_in, p = SETTINGS[setting]   # an unknown setting stops here, with a KeyError\n",
                             "    if setting == \"claim\":\n        p_in, p = 0.2, 0.5\n    elif setting == \"other\":\n        p_in, p = 0.0, 0.0\n"),
     "correct", None),   # no dict: an unknown setting stops at the first use of p_in
    ("experiment", E.replace("    return round(100 * wrong, 1)\n", "    return int(1000 * wrong)   # the number of mistakes\n"), "correct", None),
    ("experiment", E.replace("data/fashion-mnist-small.npz", "data/fashion.npz"), "error", None),
    ("experiment", E.replace("    return round(100 * wrong, 1)\n", "    return f\"{100 * wrong:.1f}\"\n"), "wrong", "not-a-number"),   # text is no number
    ("experiment", E.replace("    return round(100 * wrong, 1)\n", "    return np.float32(round(100 * wrong, 1))\n"), "wrong", "item"),
    ("experiment", E.replace("def run(", "import time\ntime.sleep(3)\n\n\ndef run("), "slow", None),   # the lines outside run
] + [("contract", code, "wrong", key) for key, code in (("nan", _C_NAN), ("order", _C_ORDER), ("same-steps", _C_SAME_STEPS),
                                                          ("same-seed", _C_SAME_SEED))] \
  + [("experiment", code, "wrong", key) for key, code in _E_WRONG.items()]


# --- help for the learner's paper --------------------------------------------------------------------------------------

def outline_text(choice: dict | None, plan: dict | None, lang: str) -> str:
    """What the learner's run is, in Markdown: what "claim" and "other" are, what stays the same and what it gives back,
    from their own plan when it is for the paper they chose, else from the course's plan for it; what they can reuse;
    and how far the course has gone with this experiment itself."""
    nl = lang == "nl"
    if not choice:
        return ("Kies eerst je paper in [Je reproductie 1](/course/open/repro-own-choose?lang=nl), en schrijf je plan in [Je reproductie 2](/course/open/repro-own-claim?lang=nl)."
                if nl else "Choose your paper in [Your reproduction 1](/course/open/repro-own-choose?lang=en) first, and write your plan in [Your reproduction 2](/course/open/repro-own-claim?lang=en).")
    pid = choice.get("id")
    p = plan if plan and plan.get("paper") == pid else (P.course_plan(pid, lang) if pid in P.PLANS else None)
    out = []
    if p:
        a, b = (plain(s) for s in P._settings(p))
        out.append(f"**In je run** is `\"claim\"` *{a}* en `\"other\"` *{b}*. Hetzelfde in beide: {plain(P._text(p.get('setup')))}. Hij geeft {plain(P._text(p.get('metric')))} terug."
                   if nl else f"**In your run**, `\"claim\"` is *{a}* and `\"other\"` is *{b}*. The same in both: {plain(P._text(p.get('setup')))}. It gives back {plain(P._text(p.get('metric')))}.")
    paper = next((x for x in papers() if x["id"] == pid), None)
    if paper:
        out.append(f"**Wat je kunt hergebruiken:** {O.builds_on_text(paper, lang)}." if nl else f"**What you can reuse:** {O.builds_on_text(paper, lang)}.")
    if pid == "dropout":
        out.append("**De versie van de course** van dit experiment is de oplossing, in de laatste hint." if nl
                   else "**The course's version** of this experiment is the solution, in the last hint.")
    elif paper:
        out.append("De course heeft dit experiment zelf nog niet gebouwd en gemeten: jij bent de eerste. Begin klein, houd het aan het contract, en laat het dan groeien."
                   if nl else "The course has not built and measured this experiment itself yet: you are the first. Start small, hold it to the contract, and let it grow from there.")
    else:
        out.append("Je paper is een eigen paper: de lessen en de code van je kit mag je hergebruiken, zover ze passen."
                   if nl else "Your paper is one of your own: the lessons and the code of your kit are yours to reuse, as far as they fit.")
    return "\n\n".join(out)


# --- the demo: your experiment from your kit ---------------------------------------------------------------------------

DEMO_STEPS = 200


def demo_runs(steps: int = DEMO_STEPS) -> dict | str | None:
    """The learner's own/experiment.py for SEEDS in both settings: {"claim": [...], "other": [...]}; None when the
    file is not there yet, "error" when it stops with an error, "slow" when one run takes more than DEMO_LIMIT."""
    import importlib
    from everything_ai import projects
    projects.on_path()
    if not (projects.folder() / "own" / "experiment.py").exists():
        return None
    projects.forget()
    try:
        run = importlib.import_module("own.experiment").run
        return {s: [float(with_time_limit(lambda: run(s, seed, steps), DEMO_LIMIT)) for seed in SEEDS] for s in SETTINGS}
    except Timeout:
        return "slow"
    except Exception:
        return "error"


def demo_verdict(runs: dict, plan: dict | None) -> str | None:
    """What the learner's own rule, own/plan.py, says about the demo's runs with their plan: None without a plan for
    their paper or a rule that runs."""
    import importlib
    if not plan or P.problem(plan) is not None:
        return None
    a, b = P._settings(plan)
    try:
        mine = importlib.import_module("own.plan")
        return mine.verdict(plan, {a: runs["claim"], b: runs["other"]})
    except Exception:
        return None


def demo_table(runs: dict, labels: dict, lang: str) -> str:
    """The demo's runs as a table: a row per setting, a column per seed."""
    import html
    pad = "padding:3px 14px 3px 0"
    head = "".join(f"<th style='{pad};text-align:right'>{labels['seed']} {s}</th>" for s in SEEDS)
    rows = "".join(f"<tr><td style='{pad}'><code>\"{s}\"</code></td>" + "".join(f"<td style='{pad};text-align:right'>{html.escape(short(x, lang).replace('-', '−'))}</td>" for x in runs[s]) + "</tr>"
                   for s in SETTINGS)
    return f"<table class='ea-cards' style='border-collapse:collapse'><thead><tr><th style='{pad}'></th>{head}</tr></thead><tbody>{rows}</tbody></table>"
