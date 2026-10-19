# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson llm-eval-multiple-choice: option_logprob, the log-probability a model gives an answer option
after a context; best_option, the option whose log-probability rises most through the context; and a choice. The demo
scores the 500 story-cloze questions of data/story-cloze.json with the learner's model of llm-pretrain-capstone (or the
course's) under four rules: the plain sum, per token, per byte, and against the option on its own."""
import importlib.util
import json
import operator
import time

from everything_ai import paths, projects
from everything_ai.check import FunctionBox, Mistake, ScenarioBox, run_code


def _checks(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E = _checks("llm-eval-base", "everything_ai_checks_llm_eval_base_for_choice")
P, S, B = E.P, E.S, E.B
BUILDS_ON = E.BUILDS_ON
projects.builds_on(*BUILDS_ON)

RULES = ("sum", "token", "byte", "alone")   # the four ways the demo picks an option
CHANCE = 0.25                               # four options


def number(x: float, digits: int, lang: str) -> str:
    return S.number(x, digits, lang)


def questions() -> list:
    """The 500 questions of data/story-cloze.json whose stories no model of the course trained on: a context of three
    sentences, four options and the index of the right one."""
    return json.loads((paths.ROOT / "data" / "story-cloze.json").read_text(encoding="utf-8"))["items"]


def rules_picture(labels: dict):
    """The right answers of the course's run of the capstone under the four rules, as bars; ink only, as for a
    measurement."""
    from everything_ai import draw
    m = MEASURED["yours"]
    return draw.hbars([(labels[rule], m["right"][rule]) for rule in RULES], top=250, title=labels["title"],
                      note=labels["note"], alt=labels["alt"])


def score_all(option_logprob, best_option, show) -> dict | str:
    """The learner's option_logprob on their model (or the course's) for every option of every question, after its
    context and after an end-of-text alone, the neutral start of a story; and how many questions each rule gets right,
    the last with the learner's best_option. "run" when a module of theirs stops with an error, "learner" when their
    option_logprob or best_option does."""
    try:
        model, info, where = P.load()
        encode = run_code("from llm.encode import encode").names["encode"]
        merges = {tuple(pair): 256 + i for i, pair in enumerate(info["merges"])}
    except Exception:
        return "run"
    end = 256 + len(merges)
    right = {rule: 0 for rule in RULES}
    picks = []
    qs = questions()
    start = time.perf_counter()
    for n, q in enumerate(qs):
        try:
            context = encode(q["context"], merges)
            options = [encode(" " + option, merges) for option in q["options"]]
        except Exception:
            return "run"
        scores = {rule: [] for rule in RULES}
        with_context, alone = [], []
        try:
            for option, ids in zip(q["options"], options):
                before = context[max(0, len(context) - (P.BLOCK + 1 - len(ids))):]   # what fits in the window with the option
                a, b = float(option_logprob(model, before, ids)), float(option_logprob(model, [end], ids))
                with_context.append(a)
                alone.append(b)
                scores["sum"].append(a)
                scores["token"].append(a / len(ids))
                scores["byte"].append(a / len((" " + option).encode("utf-8")))
            chosen = {rule: max(range(len(q["options"])), key=lambda i: scores[rule][i]) for rule in RULES[:3]}
            chosen["alone"] = int(best_option(with_context, alone))
        except Exception:
            return "learner"
        for rule in RULES:
            right[rule] += chosen[rule] == q["answer"]
        picks.append(chosen)
        if n % 50 == 49:
            show({"done": n + 1, "total": len(qs), "seconds": time.perf_counter() - start})
    return {"right": right, "total": len(picks), "picks": picks, "source": where, "params": sum(p.numel() for p in model.parameters()),
            "seconds": time.perf_counter() - start}


# --- measured: the course's runs of the demo with its own solutions, on the CPU, one thread ---
# "yours": the course's model of the demo of llm-pretrain-capstone (TinyStories: only its numbers are shared);
# "course": the course's own model, data/course-capstone.pt.
MEASURED = {
    "yours": {"right": {"sum": 138, "token": 151, "byte": 156, "alone": 220}, "total": 500, "params": 575337, "seconds": 51},
}
# --- end of measured ---

REQUIRED = ["option-logprob", "best-option", "why-short"]

_DOTS = S._DOTS
_TEXTS = S._TEXTS
_CALL, _CALL_NL = S._CALL, S._CALL_NL

# --- exercise 1: option_logprob --------------------------------------------------------------------------------------

_O_HEAD = "import torch\n\n\ndef option_logprob(model, context, option):\n"
_O = (_O_HEAD + "    ids = context + option\n    with torch.no_grad():\n        logits = model(torch.tensor([ids[:-1]]))[0]\n"
      "    logprobs = torch.log_softmax(logits, dim=-1)\n    total = 0.0\n    for i in range(len(option)):\n"
      "        total += logprobs[len(context) - 1 + i, option[i]].item()\n    return total\n")
_O_START = (_O_HEAD + "    ids = context + option\n    with torch.no_grad():\n        logits = model(torch.tensor([ids[:-1]]))[0]\n"
            "    logprobs = torch.log_softmax(logits, dim=-1)\n    total = 0.0\n    for i in range(len(option)):\n"
            "        # the log-probability of option[i], read in the row of the token before it\n        ...\n    return total\n")
_O_SETUP = ("import torch\ntorch.manual_seed(0)\n"
            "class Toy(torch.nn.Module):  # logits that depend on the token before, and on its place\n"
            "    def __init__(self):\n        super().__init__()\n        self.table = torch.nn.Embedding(7, 7)\n"
            "        self.places = torch.nn.Embedding(16, 7)\n        self.calls = 0\n"
            "    def forward(self, ids):\n"
            "        assert ids.dim() == 2, \"the model takes a batch of windows\"\n"
            "        self.calls += 1\n"
            "        return self.table(ids) + self.places(torch.arange(ids.shape[1]))\n"
            "model = Toy()\n")
_O_SCENARIOS = [
    # the lesson's example; float(): a tensor of one number counts too, as in the demo
    _O_SETUP + "round(float(option_logprob(model, [3, 1, 4], [1, 5])), 4)",
    # a longer option, and a context of one token
    _O_SETUP + "round(float(option_logprob(model, [6], [2, 6, 5, 3])), 4)",
    # a log-probability is never above 0, and two options can be compared
    _O_SETUP + "[round(float(option_logprob(model, [3, 1], [o])), 4) for o in range(7)]",
    # one run of the model scores every token of the option: a run per token makes the demo ten times as slow
    _O_SETUP + "option_logprob(model, [3, 1, 4], [1, 5, 2])\nmodel.calls",
]
_O_SHIFT = _O.replace("logprobs[len(context) - 1 + i, option[i]]", "logprobs[len(context) + i - 2, option[i]]")
_O_PROBS = _O.replace("torch.log_softmax(logits, dim=-1)", "torch.softmax(logits, dim=-1)")
_O_RAW = _O.replace("torch.log_softmax(logits, dim=-1)", "logits")
_O_LOOP = (_O_HEAD + "    total = 0.0\n    with torch.no_grad():\n        for i in range(len(option)):\n"
           "            logits = model(torch.tensor([context + option[:i]]))[0]\n"
           "            total += torch.log_softmax(logits[-1], dim=-1)[option[i]].item()\n    return total\n")
_O_ALL = _O.replace("    for i in range(len(option)):\n        total += logprobs[len(context) - 1 + i, option[i]].item()\n",
                    "    for i in range(len(ids) - 1):\n        total += logprobs[i, ids[i + 1]].item()\n")

# --- exercise 2: best_option -----------------------------------------------------------------------------------------

_P = ("def best_option(with_context, alone):\n    best = 0\n    for i in range(len(with_context)):\n"
      "        if with_context[i] - alone[i] > with_context[best] - alone[best]:\n            best = i\n    return best\n")
_P_START = ("def best_option(with_context, alone):\n    # the index of the option whose log-probability rises most through the context\n    ...\n")


def _same(got, want) -> bool:
    """A whole number equal to the index: an int, or NumPy's or PyTorch's, as np.argmax and torch.argmax give it."""
    if isinstance(got, bool):
        return False
    try:
        return operator.index(got) == want
    except TypeError:
        return False


EXERCISES = {
    "option-logprob": ScenarioBox(
        name="option-logprob",
        start=_O_START,
        solution=_O,
        scenarios=_O_SCENARIOS,
        first=[_DOTS],
        mistakes=[
            Mistake("shifted", wrong=_O_SHIFT,
                    en="With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`: you read each token in another row than the one that predicts it. Row `j` of the logits holds the prediction after token `j` of `ids`, so `option[i]`, at place `len(context) + i`, is in row `len(context) - 1 + i`.",
                    nl="Met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn: je leest elk token in een andere rij dan die het voorspelt. Rij `j` van de logits bevat de voorspelling na token `j` van `ids`, dus `option[i]`, op plek `len(context) + i`, staat in rij `len(context) - 1 + i`."),
            Mistake("probs", wrong=_O_PROBS,
                    en="With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`: you add probabilities. The probability of the whole option is their product, and its log is the sum of the logs: use `torch.log_softmax`.",
                    nl="Met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn: je telt kansen op. De kans op de hele optie is hun product, en de log daarvan is de som van de logs: gebruik `torch.log_softmax`."),
            Mistake("raw", wrong=_O_RAW,
                    en="With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`: you add logits, which are not log-probabilities yet. `torch.log_softmax(logits, dim=-1)` turns them into log-probabilities.",
                    nl="Met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn: je telt logits op, en dat zijn nog geen log-kansen. `torch.log_softmax(logits, dim=-1)` maakt er log-kansen van."),
            Mistake("all", wrong=_O_ALL,
                    en="With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`: you add the log-probabilities of the context too. They are the same for every option; add only those of the tokens of `option`.",
                    nl="Met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn: je telt ook de log-kansen van de context op. Die zijn voor elke optie gelijk; tel alleen die van de tokens van `option` op."),
            Mistake("every-token", matches=lambda got, text: isinstance(got, int) and not isinstance(got, bool) and got > 1,
                    en="With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`: your function runs the model more than once for one option. One run on `ids` without its last id gives the log-probabilities of all tokens of the option at once, each in its own row; a run per token makes the demo about ten times as slow.",
                    nl="Met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn: je functie laat het model meer dan één keer lopen voor één optie. Eén keer lopen op `ids` zonder de laatste id geeft de log-kansen van alle tokens van de optie tegelijk, elk in een eigen rij; één keer per token maakt de demo ongeveer tien keer zo traag."),
            Mistake("no-batch", error=lambda e: "takes a batch of windows" in str(e),
                    en="A model takes a batch of windows: a tensor with two dimensions, one row per window. Give it one window in a list of one, `torch.tensor([ids[:-1]])`, and take its logits with `model(...)[0]`.",
                    nl="Een model neemt een batch vensters: een tensor met twee dimensies, één rij per venster. Geef het één venster in een lijst van één, `torch.tensor([ids[:-1]])`, en neem de logits ervan met `model(...)[0]`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: add the log-probability of `option[i]` to `total`.",
                      "nl": "Vervang de `...`: tel de log-kans van `option[i]` op bij `total`."},
            "wrong": {"en": "With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`. Does your function add, for every token of `option`, the log-probability in the row of the token before it?",
                      "nl": "Met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn. Telt je functie voor elk token van `option` de log-kans op in de rij van het token ervoor?"},
            "almost": {"en": "Right for the first example. But with these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`. Does it work for every length of context and option?",
                       "nl": "Goed voor het eerste voorbeeld. Maar met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn. Werkt hij voor elke lengte van context en optie?"},
            "correct": {"en": "Your function gives the log-probability of a whole answer after its context: how a base model scores an option without writing a word.",
                        "nl": "Je functie geeft de log-kans op een heel antwoord na zijn context: zo scoort een base model een optie zonder een woord te schrijven."},
        },
    ),
    "best-option": FunctionBox(
        name="best-option",
        function="best_option",
        start=_P_START,
        solution=_P,
        visible=[([-12.5, -9.0, -20.1, -15.0], [-14.0, -8.5, -26.0, -16.2])],
        # the most likely option after the context is not the most likely one on its own; a long option rises most by 10,
        # not by a ratio of 2 to 3; the tie comes last, so that it is what "almost" points at
        hidden=[([-3.0, -4.0], [-3.5, -6.0]), ([-30.0, -10.0, -40.0], [-31.0, -12.0, -35.0]), ([-5.0, -6.0], [-7.0, -4.0]),
                ([-2.0, -20.0], [-4.0, -30.0]), ([-5.0], [-9.0]), ([-1.0, -2.0, -3.0], [-1.0, -2.0, -3.0])],
        compare=_same,
        show=S._rounded,
        first=[_DOTS],
        mistakes=[
            Mistake("context-only", wrong="def best_option(with_context, alone):\n    best = 0\n    for i in range(len(with_context)):\n"
                                          "        if with_context[i] > with_context[best]:\n            best = i\n    return best\n",
                    en=_CALL + ": your function picks the option that is most likely after the context, and a short or common option wins that way. Pick the one whose log-probability rises most: the largest `with_context[i] - alone[i]`.",
                    nl=_CALL_NL + ": je functie kiest de optie die na de context het waarschijnlijkst is, en zo wint een korte of gewone optie. Kies de optie waarvan de log-kans het meest stijgt: de grootste `with_context[i] - alone[i]`."),
            Mistake("reversed", wrong=_P.replace("with_context[i] - alone[i] > with_context[best] - alone[best]", "alone[i] - with_context[i] > alone[best] - with_context[best]"),
                    en=_CALL + ": your function picks the option whose log-probability rises least through the context. Pick the one whose rise `with_context[i] - alone[i]` is largest.",
                    nl=_CALL_NL + ": je functie kiest de optie waarvan de log-kans door de context het minst stijgt. Kies de optie waarvan de stijging `with_context[i] - alone[i]` het grootst is."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: walk through the options, and give back the index of the one with the largest `with_context[i] - alone[i]`.",
                      "nl": "Vervang de `...`: loop door de opties, en geef de index terug van de optie met de grootste `with_context[i] - alone[i]`."},
            "wrong": {"en": _CALL + ". The index of the option with the largest `with_context[i] - alone[i]`, counted from 0?",
                      "nl": _CALL_NL + ". De index van de optie met de grootste `with_context[i] - alone[i]`, geteld vanaf 0?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function work for any number of options, and give the first index when two are equal?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Werkt je functie voor elk aantal opties, en geeft hij de eerste index als twee gelijk zijn?"},
            "correct": {"en": "Your function picks the option that the context makes most likely, compared with how likely it is on its own.",
                        "nl": "Je functie kiest de optie die de context het waarschijnlijkst maakt, vergeleken met hoe waarschijnlijk ze op zich is."},
        },
    ),
}

CASES = [
    ("option-logprob", _O_START, "empty", None),
    ("option-logprob", _O, "correct", None),
    ("option-logprob", _O.replace("        total += logprobs[len(context) - 1 + i, option[i]].item()\n",
                                  "        total = total + float(logprobs[len(ids) - len(option) - 1 + i][option[i]])\n"), "correct", None),
    ("option-logprob", _O_SHIFT, "wrong", "shifted"),
    ("option-logprob", _O_PROBS, "wrong", "probs"),
    ("option-logprob", _O_RAW, "wrong", "raw"),
    ("option-logprob", _O_ALL, "wrong", "all"),
    ("option-logprob", _O_LOOP, "wrong", "every-token"),
    ("option-logprob", _O.replace("model(torch.tensor([ids[:-1]]))[0]", "model(torch.tensor(ids[:-1]))"), "error", "no-batch"),
    ("option-logprob", _O.replace("logprobs[len(context) - 1 + i, option[i]].item()", "logprobs[len(context) - 1 + i, option[i]]"), "correct", None),
    ("option-logprob", _O.replace("    return total\n", "    return total / len(option)\n"), "wrong", None),
    ("best-option", _P_START, "empty", None),
    ("best-option", _P, "correct", None),
    ("best-option", "def best_option(with_context, alone):\n    rises = [w - a for w, a in zip(with_context, alone)]\n    return rises.index(max(rises))\n", "correct", None),
    ("best-option", "def best_option(with_context, alone):\n    best = 0\n    for i in range(len(with_context)):\n"
                    "        if with_context[i] > with_context[best]:\n            best = i\n    return best\n", "wrong", "context-only"),
    ("best-option", _P.replace("with_context[i] - alone[i] > with_context[best] - alone[best]", "alone[i] - with_context[i] > alone[best] - with_context[best]"), "wrong", "reversed"),
    ("best-option", _P.replace(" > with_context[best] - alone[best]", " < with_context[best] - alone[best]"), "wrong", "reversed"),
    ("best-option", "import numpy as np\n\n\ndef best_option(with_context, alone):\n    return np.argmax(np.array(with_context) - np.array(alone))\n", "correct", None),
    ("best-option", "def best_option(with_context, alone):\n    return max(range(len(with_context)), key=lambda i: with_context[i] - alone[i])\n", "correct", None),
    ("best-option", _P.replace(" > with_context[best] - alone[best]", " >= with_context[best] - alone[best]"), "almost", None),
    ("best-option", "def best_option(with_context, alone):\n    ratios = [w / a for w, a in zip(with_context, alone)]\n    return ratios.index(min(ratios))\n", "almost", None),
    ("best-option", "def best_option(with_context, alone):\n    rises = [w - a for w, a in zip(with_context, alone)]\n    return max(rises)\n", "wrong", None),
    ("best-option", "def best_option(with_context, alone):\n    return alone.index(max(alone))\n", "wrong", None),
]
