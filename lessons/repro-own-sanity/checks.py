# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson repro-own-sanity: sanity, the sanity checks of the kit for any model whose logits have the
shape of its labels and then k numbers, saved as own/sanity.py; and the learner's experiment of Your reproduction 3,
own/experiment.py, put through the course's version of those checks in both settings. The play block is the first
loss of the course's experiment for the dropout paper, with starting weights from 1 to 3 times as large."""
import functools
import importlib.util
import math
import re
from dataclasses import dataclass

import torch  # noqa: F401  loaded here: a first import inside the time limit of a check would not finish in it

from everything_ai import paths, projects
from everything_ai.check import (MISSING, PLAIN_ERRORS, TIME_LIMIT, ClassBox, CodeBox, Mistake, Result, Run, Timeout, _fill, _plain_kind,
                                 _tb_frames, run_code, with_time_limit)


def _checks(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _checks("repro-own-build", "everything_ai_checks_repro_own_build_for_sanity")
G = _checks("repro-guided-sanity", "everything_ai_checks_repro_guided_sanity_for_own")
E, SETTINGS, PARTS, missing_parts, number, short = B.E, B.SETTINGS, B.PARTS, B.missing_parts, B.number, B.short
SANITY_STEPS = 200   # the steps the check gives a model to learn its batch by heart
RUN_LIMIT = 60.0     # seconds the sanity checks of one setting get
NO_FILE = "# There is no work/projects/own/experiment.py yet.\n"


def course(name: str):
    """make_model, one_batch, K or FROM_SCRATCH of the course's experiment for the dropout paper."""
    return _course_names()[name]


@functools.cache
def _course_names() -> dict:
    return run_code(E).names


# --- the play block: starting weights that are too large ---------------------------------------------------------------

SCALES = [1 + 0.25 * i for i in range(9)]   # 1 to 3


@functools.cache
def first_loss(scale: float, setting: str = "claim") -> float:
    """The first loss of the course's model for the dropout paper, seed 0, on its one batch, with every starting weight
    `scale` times as large, as the sanity check measures it: in training mode, so with the dropout of the setting, before
    any step."""
    import torch.nn.functional as F
    model = course("make_model")(setting, 0)
    x, y = course("one_batch")()
    with torch.no_grad():
        for w in model.parameters():
            w.mul_(scale)
        return round(F.cross_entropy(model(x), y).item(), 4)


def loss_number(v: float, lang: str) -> str:
    """A loss with 3 decimals, or with as many as it takes to show its first digit below 0.001: 0.000 would read as
    exactly 0."""
    places = 3
    while places < 8 and 0 < abs(v) < 10 ** -places:
        places += 1
    return number(v, places, lang)


def first_ok(scale: float) -> bool:
    return abs(first_loss(scale) - math.log(course("K"))) < 0.5


def first_picture(scale: float, labels: dict) -> str:
    """The first loss at every scale of SCALES, a dot each, with a line through them; the band within 0.5 of ln K, where
    the sanity check passes, as a wash; and the slider's scale marked with a ring. Purple: a loss."""
    from everything_ai import chart
    ln = math.log(course("K"))
    p = chart.Panel((0.9, 3.1), (0, 6.5), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"], xticks=[1, 1.5, 2, 2.5, 3],
                    yticks=[0, 2, 4, 6])
    p.rect(0.9, ln - 0.5, 3.1, ln + 0.5, "loss")
    p.text(3.05, ln - 0.85, labels["band"], "loss", "end")   # under the band: the line runs through it and above it
    shown = [s for s in SCALES if first_loss(s) <= 6.5]
    p.line(shown, [first_loss(s) for s in shown], "loss", 1.6)
    if first_loss(scale) <= 6.5:   # the ring first, so that the dot of its scale stays on top
        p.points([scale], [first_loss(scale)], "loss", r=8, hollow=True)
    p.points(shown, [first_loss(s) for s in shown], "loss", r=3.5)
    return chart.figure([p], alt=labels["alt"], width=360, height=260, drag=0)


# --- exercise 1: sanity -------------------------------------------------------------------------------------------------

_S_HEAD = ("import math\n\nimport torch\nimport torch.nn.functional as F\n\n\n"
           "def sanity(model, x, y, k, steps, from_scratch):\n"
           "    # the sanity checks of the kit, for a model whose logits for x have the shape\n"
           "    # of y and then k: a row of k numbers for each example, or for each place of a text\n"
           "    with torch.no_grad():\n"
           "        logits = model(x)\n")
_S_SHAPE = ("    if tuple(logits.shape) != tuple(y.shape) + (k,):   # the other checks make no sense then\n"
            "        return {\"shape\": False, \"first\": None, \"learns\": None, \"first_loss\": None, \"last_loss\": None}\n")
_S_FIRST = "    first = F.cross_entropy(logits.reshape(-1, k), y.reshape(-1)).item()\n"
_S_LEARN = ("    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)\n"
            "    for step in range(steps):\n"
            "        loss = F.cross_entropy(model(x).reshape(-1, k), y.reshape(-1))\n"
            "        opt.zero_grad()\n"
            "        loss.backward()\n"
            "        opt.step()\n")
_S_TAIL = ("    return {\"shape\": True, \"first\": abs(first - math.log(k)) < 0.5 if from_scratch else None,\n"
           "            \"learns\": loss.item() < 0.05, \"first_loss\": first, \"last_loss\": loss.item()}\n")
_S_START = (_S_HEAD + "    # 1. shape: when the logits do not have the shape of y and then k,\n"
            "    #    a report of False and four times None\n"
            "    # 2. first: the loss on the batch before any training, over every row of k logits\n"
            "    # 3. learn the batch by heart: steps steps of AdamW at a learning rate of 1e-3,\n"
            "    #    without weight decay\n    ...\n" + _S_TAIL)


def _s(shape: str = _S_SHAPE, first: str = _S_FIRST, learn: str = _S_LEARN, tail: str = _S_TAIL) -> str:
    return _S_HEAD + shape + first + learn + tail


S_CODE = _s()
_DATA = ("import torch\nfrom torch import nn\n\ntorch.manual_seed(0)\nx = torch.randn(12, 6)\ny = torch.randint(0, 5, (12,))\n"
         "model = nn.Sequential(nn.Linear(6, 128), nn.ReLU(), nn.Linear(128, 5))\n")
_ROUND = "\n{key: (round(v, 4) if isinstance(v, float) else v) for key, v in report.items()}"
_SCENARIOS = [
    # a model from scratch that works: every check passes
    _DATA + "report = sanity(model, x, y, 5, 300, True)" + _ROUND,
    # the k the check is told is not the model's
    _DATA + "report = sanity(model, x, y, 7, 5, True)" + _ROUND,
    # starting weights 5 times too large: a first loss far above ln 5
    _DATA + "with torch.no_grad():\n    for w in model.parameters():\n        w.mul_(5)\nreport = sanity(model, x, y, 5, 5, True)" + _ROUND,
    # a model that cannot learn: all but one bias frozen
    _DATA + "for w in model.parameters():\n    w.requires_grad_(False)\nmodel[2].bias.requires_grad_(True)\n"
            "report = sanity(model, x, y, 5, 50, True)" + _ROUND,
    # a trained model: its first loss is not compared with ln k
    _DATA + "report = sanity(model, x, y, 5, 300, False)" + _ROUND,
    # a model of text: a row of k logits for each place, y a label for each place
    "import torch\nfrom torch import nn\n\ntorch.manual_seed(0)\nx = torch.randint(0, 5, (4, 6))\ny = (x + 1) % 5\n"
    "model = nn.Sequential(nn.Embedding(5, 32), nn.Linear(32, 5))\nreport = sanity(model, x, y, 5, 500, True)" + _ROUND,
    # a model with dropout, which is on while it learns
    _DATA + "model = nn.Sequential(nn.Linear(6, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 5))\n"
            "report = sanity(model, x, y, 5, 500, True)" + _ROUND,
]
_S_EN, _S_NL = "Your `sanity` gives another report than the course's", "Je `sanity` geeft een ander rapport dan die van de course"
_S_WRONG = {
    "first-after": _s(first="", tail="    first = F.cross_entropy(model(x).reshape(-1, k), y.reshape(-1)).item()\n" + _S_TAIL),
    "no-reshape": _s(first="    first = F.cross_entropy(logits, y).item()\n",
                     learn=_S_LEARN.replace("model(x).reshape(-1, k), y.reshape(-1)", "model(x), y")),
    "scratch-ignored": _s(tail=_S_TAIL.replace(" if from_scratch else None", "")),
    "no-shape": _s(shape=""),
}

_DOTS = B._DOTS

EXERCISES = {
    "sanity": ClassBox(
        name="sanity",
        start=_S_START,
        solution=S_CODE,
        scenarios=_SCENARIOS,
        compare=G._close,
        digits=4,
        save_as=("own", "sanity"),
        first=[_DOTS],
        mistakes=[
            Mistake("no-shape", wrong=_S_WRONG["no-shape"], error=lambda e: isinstance(e, RuntimeError) and "shape" in str(e),
                    en="Your `sanity` stops with an error when the model's logits do not have the shape of `y` and then `k`: it computes a loss on logits it cannot use. Check the shape first, and give back the report of `False` and four times `None` when it is wrong.",
                    nl="Je `sanity` stopt met een fout als de logits van het model niet de vorm van `y` en dan `k` hebben: hij berekent een loss op logits die hij niet kan gebruiken. Kijk eerst naar de vorm, en geef het rapport van `False` en vier keer `None` terug als die niet klopt."),
            Mistake("no-reshape", wrong=_S_WRONG["no-reshape"], error=lambda e: isinstance(e, (RuntimeError, ValueError)) and ("target" in str(e).lower() or "size" in str(e).lower()),
                    en="Your `sanity` stops with an error for a model of text, whose logits have a row of `k` for each place: `F.cross_entropy` wants one row of `k` for each label. Make one long list of rows first, `logits.reshape(-1, k)`, and of labels, `y.reshape(-1)`.",
                    nl="Je `sanity` stopt met een fout bij een model van tekst, met een rij van `k` logits voor elke plek: `F.cross_entropy` wil één rij van `k` voor elk label. Maak er eerst één lange lijst rijen van, `logits.reshape(-1, k)`, en van de labels, `y.reshape(-1)`."),
            Mistake("first-after", wrong=_S_WRONG["first-after"],
                    en=_S_EN + ": your first loss is measured after learning the batch. The first loss is the one before any training: compute it from `logits`, before the loop.",
                    nl=_S_NL + ": je eerste loss is gemeten na het leren van de batch. De eerste loss is die vóór elke training: bereken hem uit `logits`, vóór de loop."),
            Mistake("scratch-ignored", wrong=_S_WRONG["scratch-ignored"],
                    en=_S_EN + ": for a trained model, `from_scratch` is `False`, and its first loss is not compared with ln k: a trained model knows more than chance. Give back `None` for `\"first\"` then.",
                    nl=_S_NL + ": voor een getraind model is `from_scratch` `False`, en dan vergelijk je zijn eerste loss niet met ln k: een getraind model weet meer dan toeval. Geef dan `None` terug voor `\"first\"`."),
            Mistake("other-lr", matches=lambda got, text: re.search(r"AdamW\([^\n]*?\blr\s*=\s*(?!1e-3\b|0\.001\b)[\d.e-]+", text) is not None,
                    en=_S_EN + ": your batch is learned at another learning rate. The kit's check uses `1e-3`, the one Guided reproduction 3 used as well: with another one, a model that cannot learn may still pass, or one that can may fail.",
                    nl=_S_NL + ": je batch wordt met een andere learning rate geleerd. De check van de kit gebruikt `1e-3`, net als Begeleide reproductie 3: met een andere kan een model dat niet kan leren toch slagen, of een model dat het wel kan zakken."),
            Mistake("betas", matches=lambda got, text: re.search(r"AdamW\([^\n]*\bbetas\s*=", text) is not None,
                    en=_S_EN + ": your AdamW has betas of its own, such as the paper's `(0.9, 0.98)` in Guided reproduction 3. The kit's check is for any model, so it uses AdamW as PyTorch makes it, with only `lr=1e-3` and `weight_decay=0.0`.",
                    nl=_S_NL + ": je AdamW heeft eigen betas, zoals die van de paper, `(0.9, 0.98)`, in Begeleide reproductie 3. De check van de kit is voor elk model, dus gebruikt hij AdamW zoals PyTorch hem maakt, met alleen `lr=1e-3` en `weight_decay=0.0`."),
            Mistake("weight-decay", matches=lambda got, text: re.search(r"AdamW\((?![^\n]*\bweight_decay\s*=\s*0(\.0*)?\s*[,)])[^\n]*\)", text) is not None,
                    en=_S_EN + ": your AdamW keeps a weight decay, its own 0.01 when you do not give one. The check learns the batch without it, `weight_decay=0.0`, as Guided reproduction 3 did: weight decay pulls the other way, and the test asks only whether the model can learn.",
                    nl=_S_NL + ": je AdamW houdt een weight decay, zijn eigen 0,01 als je er geen geeft. De check leert de batch zonder, `weight_decay=0.0`, net als Begeleide reproductie 3: weight decay trekt de andere kant op, en de test vraagt alleen of het model kan leren."),
        ],
        texts={
            **G._KEEP,
            "empty": {"en": "Replace the `...` with the three parts the comments describe: the shape, the first loss, and learning the batch by heart.",
                      "nl": "Vervang de `...` door de drie delen die het commentaar beschrijft: de vorm, de eerste loss, en de batch uit het hoofd leren."},
            "wrong": {"en": _S_EN + ". Does it check the shape first, measure the first loss over every row of `k` before any training, and learn the batch with AdamW at `1e-3`?",
                      "nl": _S_NL + ". Kijkt hij eerst naar de vorm, meet hij de eerste loss over elke rij van `k` vóór elke training, en leert hij de batch met AdamW bij `1e-3`?"},
            "almost": {"en": "Your `sanity` passes a model that works. But with these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`. Does it work for a k that is not the model's, starting weights that are too large, a frozen model, a trained one, a model of text and one with dropout?",
                       "nl": "Je `sanity` laat een model dat werkt slagen. Maar met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn. Werkt hij voor een k die niet die van het model is, te grote startwaarden, een bevroren model, een getraind model, een model van tekst en een met dropout?"},
            "correct": {"en": "Your sanity check tests the shape, the first loss of a model from scratch, and whether a model can learn one batch, also of text and with dropout. It is in your kit now, as `{saved}`.",
                        "nl": "Je sanity check test de vorm, de eerste loss van een model vanaf nul, en of een model één batch kan leren, ook van tekst en met dropout. Hij staat nu in je kit, als `{saved}`."},
            "kept": {"en": "Your sanity check is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je sanity check is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}


# --- exercise 2: your experiment through the sanity checks -------------------------------------------------------------

def experiment_text() -> str:
    """The learner's own/experiment.py without the course's header, or NO_FILE when there is none yet."""
    path = projects.folder() / "own" / "experiment.py"
    if not path.is_file():
        return NO_FILE
    text = path.read_text(encoding="utf-8")
    if projects._MARK in text:
        text = text.split(projects._MARK, 1)[1].split("\n", 1)[1]
    return text


@functools.cache
def course_sanity():
    names = {}
    exec(compile(S_CODE, "<the course's sanity check>", "exec"), names)   # not "<your code>": an error names the learner's line
    return names["sanity"]


def _device() -> str:
    """Where the checks run the parts: on an NVIDIA card when there is one, as `run` does (Your reproduction 3)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class SanityRun(CodeBox):
    """The learner's experiment, the text of own/experiment.py, put through the course's sanity check in both
    settings, seed 0, with SANITY_STEPS steps on its one batch: the first thing that fails is the message. It saves
    nothing: the experiment is the file of Your reproduction 3, which the learner fixes there."""

    def check(self, text, inputs=None) -> Result:
        if text is MISSING or text is None:
            return Result("error", self.message("missing"))
        if not text.strip():
            return Result("empty", self.message("empty"))
        if text == NO_FILE:
            return self._mistake("no-file")
        ran = run_code(text)
        if ran.timeout:
            return Result("slow", self.message("slow-top", s=f"{TIME_LIMIT:g}"))
        if ran.error is not None:
            return Result("error", self.message("stops"))
        missing = missing_parts(ran.names)
        if missing:
            return self._mistake("no-parts", missing=", ".join(f"`{name}`" for name in missing))
        names, sanity = ran.names, course_sanity()
        k, scratch = names["K"], names["FROM_SCRATCH"]
        ln = {lang: number(math.log(k), 2, lang) for lang in ("en", "nl")}
        reports, dev = {}, _device()
        for setting in B.SETTINGS:
            try:
                model = names["make_model"](setting, 0).to(dev)
                x, y = names["one_batch"]()
                x, y = x.to(dev), y.to(dev)
                r = with_time_limit(lambda: sanity(model, x, y, k, SANITY_STEPS, scratch), RUN_LIMIT)
            except Timeout:
                return self._mistake("slow", setting=setting)
            except Exception as e:
                return self._stopped(e, text, setting)
            reports[setting] = r
            got = {lang: number(r["first_loss"], 2, lang) if r["first_loss"] is not None else "" for lang in ("en", "nl")}
            if not r["shape"]:
                with torch.no_grad():
                    shape = tuple(names["make_model"](setting, 0).to(dev)(x).shape)
                return self._mistake("shape", setting=setting, got=str(shape), want=str(tuple(y.shape) + (k,)), k=str(k))
            if r["first"] is False:
                return self._mistake("first" if r["first_loss"] > math.log(k) else "first-low", setting=setting, first=got, ln=ln)
            if not r["learns"]:
                return self._mistake("learns", setting=setting, steps=str(SANITY_STEPS),
                                     last={lang: number(r["last_loss"], 3, lang) for lang in ("en", "nl")})
            if not scratch and r["first_loss"] > math.log(k) - 0.1:   # a trained model knows more than chance
                return self._mistake("scratch", setting=setting, first=got, ln=ln)
        first = {lang: " / ".join(number(reports[s]["first_loss"], 2, lang) for s in B.SETTINGS) for lang in ("en", "nl")}
        last = {lang: " / ".join(loss_number(reports[s]["last_loss"], lang) for s in B.SETTINGS) for lang in ("en", "nl")}
        return Result("correct", _fill(self.texts["correct" if scratch else "correct-trained"], first=first, last=last, ln=ln,
                                       steps=str(SANITY_STEPS)))

    def _mistake(self, key: str, **values) -> Result:
        m = next(m for m in self.mistakes if m.key == key)
        return Result(m.state, _fill(m.text, **values), mistake=key)

    def _stopped(self, e: BaseException, text: str, setting: str) -> Result:
        """The parts stop with an error: the message, and the line of the experiment where it happens, when it is there."""
        r = self._mistake("parts-stop", setting=setting)
        frames = [fr for fr in _tb_frames(e) if fr[0] == "<your code>"]
        if not frames:
            return r
        plain = self._plain_error(Run(error=e, line=frames[-1][1]), text)
        if _plain_kind(e) == "other":   # nothing to explain but the line
            plain = {lang: t.replace(PLAIN_ERRORS["other"][lang], "", 1).strip() for lang, t in plain.items()}
        return Result(r.state, {lang: (r.text[lang] + " " + plain[lang]).strip() for lang in r.text}, mistake=r.mistake)


_R3 = {"en": "[Your reproduction 3](/course/open/repro-own-build?lang=en)", "nl": "[Je reproductie 3](/course/open/repro-own-build?lang=nl)"}
_E_WRONG = {
    "shape": E.replace("K = 10 ", "K = 12 ").replace("nn.Linear(512, K))", "nn.Linear(512, 10))"),
    "first": E.replace("    return nn.Sequential(", "    model = nn.Sequential(").replace(
        "nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p), nn.Linear(512, K))\n",
        "nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p), nn.Linear(512, K))\n"
        "    with torch.no_grad():\n        for w in model.parameters():\n            w.mul_(3)   # starting weights 3 times too large\n"
        "    return model\n"),
    "learns": E.replace("    return nn.Sequential(", "    model = nn.Sequential(").replace(
        "nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p), nn.Linear(512, K))\n",
        "nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p), nn.Linear(512, K))\n"
        "    for w in list(model.parameters())[:-1]:\n        w.requires_grad_(False)   # all frozen but the last bias\n"
        "    return model\n"),
    "first-low": E.replace("    return nn.Sequential(", "    model = nn.Sequential(").replace(
        "nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p), nn.Linear(512, K))\n",
        "nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p), nn.Linear(512, K))\n"
        "    x, y, _, _ = load()   # trained on the first 1000 pictures, as if loaded from a checkpoint\n"
        "    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)\n"
        "    for _ in range(60):\n        loss = nn.functional.cross_entropy(model(x[:1000]), y[:1000])\n"
        "        opt.zero_grad()\n        loss.backward()\n        opt.step()\n    return model\n"),
    "scratch": E.replace("FROM_SCRATCH = True ", "FROM_SCRATCH = False"),
    "no-parts": E.replace("def one_batch():", "def first_batch():"),
    "parts-stop": E.replace("    return x[:32], y[:32]\n", "    return x[:32].numpy(), y[:32]\n"),
}
assert all(code != E for code in _E_WRONG.values())

EXERCISES["sanity-run"] = SanityRun(
    name="sanity-run",
    start="",
    solution=E,
    mistakes=[
        Mistake("no-file", en="Your experiment is not in your kit yet: build it in exercise 2 of " + _R3["en"] + " first. This check then reads it from `work/projects/own/experiment.py`.",
                nl="Je experiment zit nog niet in je kit: bouw het eerst in oefening 2 van " + _R3["nl"] + ". Deze check leest het dan uit `work/projects/own/experiment.py`.", state="error"),
        Mistake("no-parts", wrong=_E_WRONG["no-parts"],
                en="Your experiment does not have everything the sanity checks take from it: {missing}. Add what is missing in exercise 2 of " + _R3["en"] + ", as its start has it: `make_model(setting, seed)` and `one_batch()` as functions, `K` as a whole number from 2 on, and `FROM_SCRATCH` as `True` or `False`.",
                nl="Je experiment heeft niet alles wat de sanity checks eruit halen: {missing}. Voeg toe wat ontbreekt in oefening 2 van " + _R3["nl"] + ", zoals het in het begin daar staat: `make_model(setting, seed)` en `one_batch()` als functies, `K` als een heel getal vanaf 2, en `FROM_SCRATCH` als `True` of `False`."),
        Mistake("parts-stop", wrong=_E_WRONG["parts-stop"], state="error",
                en="With setting `\"{setting}\"`, the sanity check stops with an error on your parts. Does `one_batch()` give back two tensors, `x` and `y`, with whole numbers from 0 to `K` - 1 in `y`, and is `model(x)` of the model of `make_model` a tensor of logits, not a tuple or an object that holds them? Fix it in " + _R3["en"] + ", and try your parts with a few lines of your own first.",
                nl="Met setting `\"{setting}\"` stopt de sanity check met een fout op je delen. Geeft `one_batch()` twee tensors terug, `x` en `y`, met hele getallen van 0 tot `K` - 1 in `y`, en is `model(x)` van het model van `make_model` een tensor met logits, geen tuple of een object waar ze in zitten? Repareer het in " + _R3["nl"] + ", en probeer je delen eerst met een paar eigen regels."),
        Mistake("slow", state="slow",
                en="With setting `\"{setting}\"`, the sanity checks are not done after a minute. One batch should be small, a few dozen examples: does `one_batch()` give back more? Or is your model so large that 200 steps take more than a minute on this computer? Then make the batch smaller still: fewer examples, and for text shorter ones.",
                nl="Met setting `\"{setting}\"` zijn de sanity checks na een minuut nog niet klaar. Eén batch hoort klein te zijn, een paar tientallen voorbeelden: geeft `one_batch()` meer terug? Of is je model zo groot dat 200 stappen op deze computer langer dan een minuut duren? Maak de batch dan nog kleiner: minder voorbeelden, en bij tekst kortere."),
        Mistake("shape", wrong=_E_WRONG["shape"],
                en="With setting `\"{setting}\"`, the logits of your model for `one_batch()` have the shape {got}, and the check wants {want}: the shape of `y` and then `K`, which is {k} in your experiment. Is `K` the number of logits your model gives for each example, and does `y` hold one label for each example, or for each place of a text?",
                nl="Met setting `\"{setting}\"` hebben de logits van je model voor `one_batch()` de vorm {got}, en de check wil {want}: de vorm van `y` en dan `K`, dat in je experiment {k} is. Is `K` het aantal logits dat je model voor elk voorbeeld geeft, en heeft `y` één label voor elk voorbeeld, of voor elke plek van een tekst?"),
        Mistake("first", wrong=_E_WRONG["first"],
                en="With setting `\"{setting}\"`, the first loss of your model is {first}, more than half a point above ln K = {ln}, the loss of a model from scratch that knows nothing yet: your model starts out sure, and wrong. Its starting weights may be too large, as in the picture above: an initialization of your own, or a scale copied wrong from the paper. Fix it in " + _R3["en"] + ".",
                nl="Met setting `\"{setting}\"` is de eerste loss van je model {first}, meer dan een half punt boven ln K = {ln}, de loss van een model vanaf nul dat nog niets weet: je model begint zeker, en fout. Misschien zijn de startwaarden te groot, zoals in het plaatje hierboven: een eigen initialisatie, of een schaal die je verkeerd overnam uit de paper. Repareer het in " + _R3["nl"] + "."),
        Mistake("first-low", wrong=_E_WRONG["first-low"],
                en="With setting `\"{setting}\"`, the first loss of your model is {first}, more than half a point below ln K = {ln}, the loss of a model from scratch that knows nothing yet. Does `make_model` load a trained model? Then set `FROM_SCRATCH = False` in " + _R3["en"] + ". If it really starts from random weights, it sees the answer somewhere: look for the label in what `one_batch()` gives as `x`.",
                nl="Met setting `\"{setting}\"` is de eerste loss van je model {first}, meer dan een half punt onder ln K = {ln}, de loss van een model vanaf nul dat nog niets weet. Laadt `make_model` een getraind model? Zet dan `FROM_SCRATCH = False` in " + _R3["nl"] + ". Begint het echt met willekeurige weights, dan ziet het het antwoord ergens: zoek het label in wat `one_batch()` als `x` geeft."),
        Mistake("learns", wrong=_E_WRONG["learns"],
                en="With setting `\"{setting}\"`, {steps} steps of AdamW bring the loss on one batch only to {last}, not below 0.05: your model cannot even learn a few examples by heart. Is a part of it frozen, or cut off from the loss, or does its output not depend on its weights? Or does the same input come with different labels in your batch, as for a model of text that sees only the token before? Fix it in " + _R3["en"] + ".",
                nl="Met setting `\"{setting}\"` brengen {steps} stappen AdamW de loss op één batch maar tot {last}, niet onder 0,05: je model kan zelfs een paar voorbeelden niet uit het hoofd leren. Is een deel bevroren, of losgekoppeld van de loss, of hangt zijn output niet van zijn weights af? Of zit dezelfde input met verschillende labels in je batch, zoals bij een model van tekst dat alleen het token ervoor ziet? Repareer het in " + _R3["nl"] + "."),
        Mistake("scratch", wrong=_E_WRONG["scratch"], state="almost",
                en="Your experiment says `FROM_SCRATCH = False`, a trained model, but with setting `\"{setting}\"` its first loss is {first}, not below ln K = {ln}, the loss of a model that knows nothing yet. A trained model starts lower. Does `make_model` really load the trained weights? If your model starts from random weights, or its last layer does, set `FROM_SCRATCH = True` in " + _R3["en"] + ".",
                nl="Je experiment zegt `FROM_SCRATCH = False`, een getraind model, maar met setting `\"{setting}\"` is zijn eerste loss {first}, niet onder ln K = {ln}, de loss van een model dat nog niets weet. Een getraind model begint lager. Laadt `make_model` de getrainde weights echt? Begint je model met willekeurige weights, of zijn laatste layer, zet dan `FROM_SCRATCH = True` in " + _R3["nl"] + "."),
    ],
    texts={
        "missing": {"en": "I cannot see your experiment. Run the cell with the button again.",
                    "nl": "Ik zie je experiment niet. Voer de cel met de knop opnieuw uit."},
        "empty": {"en": "Press the button: the check reads your experiment from your kit, and puts it through the sanity checks in both settings.",
                  "nl": "Druk op de knop: de check leest je experiment uit je kit, en haalt het door de sanity checks in beide settings."},
        "slow-top": {"en": "The lines of your `work/projects/own/experiment.py` outside its functions are not done after {s} seconds, all the time the check gives them. Load data, models and large libraries inside `make_model` and `one_batch`, in " + _R3["en"] + ".",
                     "nl": "De regels van je `work/projects/own/experiment.py` buiten zijn functies zijn na {s} seconden nog niet klaar, en meer tijd geeft de check ze niet. Laad data, modellen en grote libraries binnen `make_model` en `one_batch`, in " + _R3["nl"] + "."},
        "stops": {"en": "Your `work/projects/own/experiment.py` stops with an error before any check runs. Check it again in " + _R3["en"] + ".",
                  "nl": "Je `work/projects/own/experiment.py` stopt met een fout voordat er een check draait. Check het opnieuw in " + _R3["nl"] + "."},
        "correct": {"en": "Your experiment passes the sanity checks of your kit in both settings: the shape of the logits is right, the first loss is {first} for \"claim\" / \"other\", near ln K = {ln}, and {steps} steps learn one batch by heart, to a loss of {last}. Now a long run is worth its hours.",
                    "nl": "Je experiment haalt de sanity checks van je kit in beide settings: de vorm van de logits klopt, de eerste loss is {first} voor \"claim\" / \"other\", dicht bij ln K = {ln}, en {steps} stappen leren één batch uit het hoofd, tot een loss van {last}. Nu is een lange run zijn uren waard."},
        "correct-trained": {"en": "Your experiment passes the sanity checks of your kit in both settings: the shape of the logits is right, the first loss of your trained model is {first} for \"claim\" / \"other\", below the ln K = {ln} of a model that knows nothing, and {steps} steps learn one batch by heart, to a loss of {last}. Now a long run is worth its hours.",
                            "nl": "Je experiment haalt de sanity checks van je kit in beide settings: de vorm van de logits klopt, de eerste loss van je getrainde model is {first} voor \"claim\" / \"other\", onder de ln K = {ln} van een model dat niets weet, en {steps} stappen leren één batch uit het hoofd, tot een loss van {last}. Nu is een lange run zijn uren waard."},
    },
)

# The lesson is done when these are correct. "why-sanity" is the open question.
REQUIRED = ["sanity", "sanity-run", "why-sanity"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("sanity", _S_START, "empty", None),
    ("sanity", S_CODE, "correct", None),
    ("sanity", _s(first="    first = F.cross_entropy(logits.view(-1, k), y.view(-1)).item()\n"), "correct", None),
    ("sanity-run", "", "empty", None),
    ("sanity-run", E, "correct", None),
    ("sanity-run", NO_FILE, "error", "no-file"),
    ("sanity-run", "x = 1 / 0\n", "error", None),
] + [("sanity", code, "error" if key in ("no-shape", "no-reshape") else "wrong", key) for key, code in _S_WRONG.items()] \
  + [("sanity", S_CODE.replace("lr=1e-3", "lr=1e-2"), "wrong", "other-lr"),
     ("sanity", S_CODE.replace(", weight_decay=0.0", ""), "wrong", "weight-decay"),
     ("sanity", S_CODE.replace("lr=1e-3, ", "lr=1e-3, betas=(0.9, 0.98), "), "wrong", "betas"),
     ("sanity-run", "import time\n\ntime.sleep(3)\n" + E, "slow", None)] \
  + [("sanity-run", code, {"parts-stop": "error", "scratch": "almost"}.get(key, "wrong"), key) for key, code in _E_WRONG.items()]


# --- the demo: your own sanity check on your experiment ------------------------------------------------------------------

def demo() -> dict | str | None:
    """The learner's own/sanity.py on their own/experiment.py, both settings: a report per setting; None when one of
    the two files is not there yet; "error" when it stops with an error; "slow" when one setting takes over a minute."""
    import importlib
    projects.on_path()
    folder = projects.folder() / "own"
    if not (folder / "sanity.py").is_file() or not (folder / "experiment.py").is_file():
        return None
    projects.forget()
    try:
        mine = importlib.import_module("own.sanity").sanity
        ex = importlib.import_module("own.experiment")
        dev = _device()
        x, y = ex.one_batch()
        x, y = x.to(dev), y.to(dev)
        reports = {s: with_time_limit(lambda: mine(ex.make_model(s, 0).to(dev), x, y, ex.K, SANITY_STEPS, ex.FROM_SCRATCH), RUN_LIMIT)
                   for s in B.SETTINGS}
    except Timeout:
        return "slow"
    except Exception:
        return "error"
    keys = {"shape", "first", "learns", "first_loss", "last_loss"}
    if not all(isinstance(r, dict) and keys <= set(r) for r in reports.values()):   # a sanity.py changed after its check
        return "error"
    return reports


def demo_table(reports: dict, labels: dict, lang: str) -> str:
    """The two reports as a table: a row per setting, a column per check."""
    import html
    pad = "padding:3px 14px 3px 0"
    cols = ("shape", "first", "learns", "first_loss", "last_loss")

    def cell(v):
        if isinstance(v, bool) or v is None:
            return labels[str(v)]
        try:
            return loss_number(float(v), lang)
        except (TypeError, ValueError):   # anything else your sanity.py gives back, as it is
            return str(v)
    head = "".join(f"<th style='{pad};text-align:left'><code>{c}</code></th>" for c in cols)
    rows = "".join(f"<tr><td style='{pad}'><code>\"{s}\"</code></td>" + "".join(f"<td style='{pad}'>{html.escape(cell(r[c]))}</td>" for c in cols) + "</tr>"
                   for s, r in reports.items())
    return f"<table class='ea-cards' style='border-collapse:collapse'><thead><tr><th style='{pad}'></th>{head}</tr></thead><tbody>{rows}</tbody></table>"
