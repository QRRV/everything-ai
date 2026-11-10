# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-pretrained-features: train_probe, which trains a linear probe, one linear layer, on the
features of photos. It is checked on made-up features, so it works without DINOv2. DINOv2-small (88 MB, from
Hugging Face) and Caltech-101 (137 MB, from CaltechDATA) the lesson downloads once, when the learner asks for them
(everything_ai.hub, everything_ai.caltech): the worked example and the demo need them, the exercises do not. The
slider uses what the course measured with them (REFERENCE)."""
import errno
import functools
import html
import re
import urllib.error

import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch import nn

from everything_ai import caltech, chart, dinov2, hub
from everything_ai.lesson import download_card
from everything_ai.check import FunctionBox, Mistake, run_code, with_time_limit

DOWNLOAD_LINE = {
    "en": "Works on every computer. DINOv2-small, 88 MB, is downloaded once from Hugging Face, and Caltech-101, 137 MB of photos, from Caltech, when you click their button in the lesson; after that, the lesson works without a connection. The exercises work without them too.",
    "nl": "Werkt op elke computer. DINOv2-small, 88 MB, wordt één keer gedownload van Hugging Face, en Caltech-101, 137 MB aan foto's, van Caltech, als je in de les op de knop ervoor klikt; daarna werkt de les zonder verbinding. De oefeningen werken ook zonder.",
}
CLASSES = list(caltech.CHECKED)   # the 10 classes of Caltech-101 the lesson uses, in alphabetical order: label 0 to 9
NAMES = {
    "en": {"chandelier": "chandelier", "dragonfly": "dragonfly", "ewer": "ewer", "lotus": "lotus", "nautilus": "nautilus",
           "rooster": "rooster", "starfish": "starfish", "sunflower": "sunflower", "water_lilly": "water lily",
           "windsor_chair": "Windsor chair"},
    "nl": {"chandelier": "kroonluchter", "dragonfly": "libel", "ewer": "schenkkan", "lotus": "lotus", "nautilus": "nautilus",
           "rooster": "haan", "starfish": "zeester", "sunflower": "zonnebloem", "water_lilly": "waterlelie",
           "windsor_chair": "windsorstoel"},
}
PER_CLASS = [1, 2, 3, 5, 10, 20]   # the stops of the slider: photos per class to learn from
TRAIN, TEST = 20, (20, 30)         # of each class's 30 photos: the first 20 to learn from, photos 20 to 29 to test on
LR, EPOCHS = 0.1, 100              # the probe's training: SGD on all training photos at once

# What the course measured: everything_ai.dinov2 with its pinned file, the 30 photos of each class
# (everything_ai.caltech.checked), the probe of the worked example (torch.manual_seed(0), SGD, LR, EPOCHS) on the
# first n photos of each class, tested on the 100 test photos: how many it gets right. "pixels" is the same probe on
# the prepared photos themselves, 3 x 224 x 224 = 150,528 numbers each. "mixed" counts the mistakes on DINOv2's
# features that take a lotus for a water lily or the other way around; "seeds" is the lowest and highest count with
# 1 photo per class over the seeds 0 to 4. tests/test_cv_pretrained_features.py measures them again when
# EAI_DINOV2_FILE names the downloaded file and Caltech-101 is unpacked.
REFERENCE = {
    "features": {1: 95, 2: 95, 3: 94, 5: 97, 10: 96, 20: 98},
    "pixels": {1: 20, 2: 26, 3: 25, 5: 28, 10: 34, 20: 33},
    "mixed": {1: 4, 2: 5, 3: 6, 5: 3, 10: 4, 20: 2},
    "seeds": (93, 98),
}
PIXELS = 3 * 224 * 224


def params(features: int, classes: int) -> int:
    """The numbers a linear probe learns: a weight per feature and class, and a bias per class."""
    return features * classes + classes


def number(x: int, lang: str) -> str:
    """A whole number with the thousands mark of the language: 150,528, or 150.528 in Dutch."""
    text = f"{x:,}"
    return text.replace(",", ".") if lang == "nl" else text


# --- the picture: a photo through frozen DINOv2 and a linear probe ---------------------------------------------------

def probe_picture(labels: dict) -> str:
    """From top to bottom: a photo, DINOv2-small (frozen, grey), its 384 features (an embedding: hidden), the linear
    probe (param: the only part that learns), and 10 logits. `labels` has "title", "photo", "dino", "dino_note",
    "features", "probe", "probe_note", "logits", "frozen", "learns" and "alt"."""
    W = 330
    frozen = "fill:var(--ea-ink);fill-opacity:0.06;stroke:var(--ea-ink-3);stroke-width:1.2"
    learns = "fill:var(--ea-param-wash);stroke:var(--ea-param);stroke-width:1.5"
    plain = "fill:none;stroke:var(--ea-ink-3);stroke-width:1;stroke-dasharray:3 3"
    hidden = "fill:var(--ea-hidden-wash);stroke:var(--ea-hidden);stroke-width:1.2"
    arrow = 'style="fill:none;stroke:var(--ea-ink-3);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round"'
    rows = [(labels["photo"], "", plain, "", 32), (labels["dino"], labels["dino_note"], frozen, labels["frozen"], 44),
            (labels["features"], "", hidden, "", 32), (labels["probe"], labels["probe_note"], learns, labels["learns"], 44),
            (labels["logits"], "", plain, "", 32)]
    parts = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    y = 28
    for k, (name, note, style, state, h) in enumerate(rows):
        parts.append(f'<rect x="1" y="{y}" width="{W - 2}" height="{h}" rx="6" style="{style}"/>')
        if note:
            parts.append(f'<text x="12" y="{y + 18}" style="font-size:13px;font-weight:600;fill:var(--ea-ink)">{html.escape(name)}</text>'
                         f'<text x="12" y="{y + 35}" style="font-size:12px;fill:var(--ea-ink-2)">{html.escape(note)}</text>'
                         f'<text x="{W - 10}" y="{y + 18}" style="font-size:12px;font-style:italic;fill:var(--ea-ink-2);text-anchor:end">{html.escape(state)}</text>')
        else:
            parts.append(f'<text x="12" y="{y + h / 2 + 4.5}" style="font-size:12.5px;fill:var(--ea-ink)">{html.escape(name)}</text>')
        y += h
        if k < len(rows) - 1:
            parts.append(f'<path d="M{W / 2} {y + 3} V{y + 15} M{W / 2 - 5} {y + 10} L{W / 2} {y + 15} L{W / 2 + 5} {y + 10}" {arrow}/>')
            y += 18
    y += 4
    alt = labels["title"] + ". " + labels["alt"]
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:min(100%,{W}px)">'
            f'<svg viewBox="0 0 {W} {y}" role="img" aria-label="{html.escape(alt)}" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(alt)}</title>' + "".join(parts) + "</svg></figure>")


# --- the slider: photos per class against test photos right ----------------------------------------------------------

def curve(n: int, labels: dict, lang: str) -> str:
    """The test photos the probe gets right, of 100, against the photos per class it learned from: on DINOv2's features
    (a prediction: solid) and on the pixels (dashed), with guessing at 10. The stop n is marked on both lines.
    `labels` has "title", "x", "y", "features", "pixels", "guess" and "alt"."""
    xs = PER_CLASS
    panel = chart.Panel((0.8, 25), (0, 100), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                        xticks=PER_CLASS, yticks=[0, 25, 50, 75, 100], xlog=True)
    panel.line([0.8, 25], [10, 10], role="ink-3", width=1.2, dashed=True)
    panel.line(xs, [REFERENCE["features"][x] for x in xs], role="pred")
    panel.line(xs, [REFERENCE["pixels"][x] for x in xs], role="pred", dashed=True)
    panel.points([n], [REFERENCE["features"][n]], role="pred", r=5)
    panel.points([n], [REFERENCE["pixels"][n]], role="pred", r=5, hollow=True)
    panel.text(1.6, 85, labels["features"], role="ink", anchor="start")
    panel.text(1.6, 44, labels["pixels"], role="ink", anchor="start")
    panel.text(7, 15, labels["guess"], role="ink-3", anchor="start")
    svg = chart.figure([panel], alt=labels["alt"], width=330, height=250)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


# --- DINOv2 and Caltech-101 in the lesson -----------------------------------------------------------------------------

def photos(name: str) -> list:
    """The 30 photos of a class the lesson uses, without people: everything_ai.caltech.checked."""
    return caltech.checked(name, 30)


def ready() -> bool:
    return dinov2.downloaded() and caltech.unpacked()


@functools.lru_cache(maxsize=1)
def _dino():
    return dinov2.load()


def dino_or_stop(button, texts: dict):
    """DINOv2-small for the lesson's cells, with Caltech-101 unpacked. When either is not there yet, the cell stops
    with `texts["ask"]` and the button; a click downloads what is missing and unpacks the photos, with the progress in
    the cell. While it downloads, the lesson's other cells wait. `texts` has "ask", "busy" ({name}, {done} and {total}
    in MB), "unpacking" ({done} photos), "no_net", "not_pinned", "no_room" and "failed" ({why}): why it did not work;
    and "here"."""
    import marimo as mo
    failed = None
    if not ready() and button.value:
        try:
            for name, pin in (("DINOv2-small", dinov2.FILE), ("Caltech-101", caltech.ZIP)):
                if hub.pinned(*pin) is not None:
                    continue
                total = round(hub.PINNED[pin]["size"] / 1e6)
                last = [-1]

                def progress(done, size, name=name, total=total, last=last):
                    mb = int(done / 1e6)
                    if mb // 5 != last[0] // 5:   # every 5 MB
                        last[0] = mb
                        mo.output.replace(mo.md(texts["busy"].format(name=name, done=mb, total=total)))
                hub.download(*pin, progress=progress)
            caltech.unpack(progress=lambda done: mo.output.replace(mo.md(texts["unpacking"].format(done=done))))
        except hub.NotPinned:
            failed = texts["not_pinned"]
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            failed = texts["no_net"]
        except OSError as e:
            failed = texts["no_room"] if e.errno == errno.ENOSPC else texts["failed"].format(why=html.escape(str(e)))
    if not ready():
        mo.stop(True, download_card(texts["ask"], button, failed))
    model = _dino()
    mo.output.replace(mo.md(texts["here"]))
    return model


@functools.lru_cache(maxsize=1)
def features(model) -> dict:
    """The features `model` (DINOv2) gives the 30 photos of every class, computed once: a tensor of shape (30, 384)
    per class. The 30 photos of a class go through it together, which is faster than one by one and gives the same
    features."""
    with torch.no_grad():
        return {name: model(torch.cat([dinov2.prepare(p) for p in photos(name)])) for name in CLASSES}


def split(feats: dict, n: int):
    """The features and labels of the first n photos of each class, to learn from, and of the 100 test photos."""
    X = torch.cat([feats[c][:n] for c in CLASSES])
    X_test = torch.cat([feats[c][TEST[0]:TEST[1]] for c in CLASSES])
    y = torch.arange(len(CLASSES)).repeat_interleave(n)
    y_test = torch.arange(len(CLASSES)).repeat_interleave(TEST[1] - TEST[0])
    return X, y, X_test, y_test


def right(probe, X, y) -> int:
    with torch.no_grad():
        return (probe(X).argmax(dim=1) == y).sum().item()


# The lesson is done when these are correct. "params" and "pixels" are the choice questions, "no-labels" the open one.
REQUIRED = ["params", "probe", "pixels", "no-labels"]

# --- Exercise 2: train_probe ------------------------------------------------------------------------------------------

_HEAD = "import torch\nfrom torch import nn\n\n\ndef train_probe(X, y, k):\n    torch.manual_seed(0)\n"
_TAIL = ("    loss_fn = nn.CrossEntropyLoss()\n    opt = torch.optim.SGD(probe.parameters(), lr=0.1)\n"
         "    for epoch in range(100):\n{comment}        loss = {loss}\n{zero}        loss.backward()\n        opt.step()\n"
         "    return probe\n")


def _probe(layer: str = "nn.Linear(X.shape[1], k)", loss: str = "loss_fn(probe(X), y)", zero: bool = True,
           comments: bool = False) -> str:
    first = "    # a linear layer from the features of a photo to k classes\n" if comments else ""
    second = "        # the loss of the probe's logits for all photos, against their labels\n" if comments else ""
    return (_HEAD + first + f"    probe = {layer}\n"
            + _TAIL.format(comment=second, loss=loss, zero="        opt.zero_grad()\n" if zero else ""))


_START = _probe(layer="...", loss="...", comments=True)
_SOLUTION = _probe()


class _Data:
    """Made-up features or labels as a test case: written as their name in the check's messages; the function gets a
    copy of the tensor itself."""

    def __init__(self, shown: str, value: torch.Tensor):
        self.shown, self.value = shown, value

    def __repr__(self):
        return self.shown

    def __deepcopy__(self, memo):
        return self.value.clone()


def _made_up(k: int, d: int, per: int, seed: int, shown: str = "X"):
    """k classes of `per` photos each, with d features: a centre per class, and noise around it."""
    g = torch.Generator().manual_seed(seed)
    centres = 3 * torch.randn(k, d, generator=g)
    X = centres.repeat_interleave(per, dim=0) + torch.randn(k * per, d, generator=g)
    y = torch.arange(k).repeat_interleave(per)
    return (_Data(shown, X), _Data("y" if shown == "X" else "y" + shown[1:], y), k)


def _summary(probe):
    """What the check compares: how many features and classes the probe's linear layer has, and the class it gives 8
    fixed points; anything that is not a model, as it is."""
    if not isinstance(probe, nn.Module):
        return probe
    layer = probe if isinstance(probe, nn.Linear) else next((m for m in probe.modules() if isinstance(m, nn.Linear)), None)
    if layer is None:
        return "no-layer"
    points = 3 * torch.randn(8, layer.in_features, generator=torch.Generator().manual_seed(1))
    try:
        with torch.no_grad():
            classes = probe(points).argmax(dim=1).tolist()
    except Exception:
        classes = "error"
    return [[layer.in_features, layer.out_features], classes]


def _shown(v):
    if isinstance(v, list) and len(v) == 2 and isinstance(v[0], list):
        (d, k), classes = v
        if classes == "error":
            return {"en": f"a probe from {d} features to {k} classes, which stops with an error",
                    "nl": f"een probe van {d} features naar {k} klassen, die stopt met een fout"}
        return {"en": f"a probe from {d} features to {k} classes, which gives 8 test points the classes {classes}",
                "nl": f"een probe van {d} features naar {k} klassen, die 8 testpunten de klassen {classes} geeft"}
    if isinstance(v, torch.Tensor):
        return {"en": f"a tensor of shape {tuple(v.shape)}", "nl": f"een tensor met shape {tuple(v.shape)}"}
    return repr(v)


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_CALL = "`{call}` gives {got}, and it should give {want}"
_CALL_NL = "`{call}` geeft {got}, en dat moet {want} zijn"
_CASES = [_made_up(3, 4, 2, seed=0), _made_up(5, 6, 3, seed=1), _made_up(2, 384, 1, seed=2), _made_up(4, 8, 5, seed=3)]

EXERCISES = {
    "probe": FunctionBox(
        name="probe",
        function="train_probe",
        start=_START,
        solution=_SOLUTION,
        visible=_CASES[:1],
        hidden=_CASES[1:],
        collect=_summary,
        show=_shown,
        first=[_DOTS],
        mistakes=[
            Mistake("fixed-384", syntax=r"nn\.Linear\(\s*384\b",
                    en="Your code stops with an error at `{line}`: the check's made-up photos have 4 features each, not 384. Take the number of features from `X`, whatever it is: `X.shape[1]`, the length of a row.",
                    nl="Je code stopt met een fout bij `{line}`: de verzonnen foto's van de check hebben elk 4 features, geen 384. Haal het aantal features uit `X`, wat het ook is: `X.shape[1]`, de lengte van een rij."),
            Mistake("rows", syntax=r"nn\.Linear\(\s*(X\.shape\[0\]|len\(\s*X\s*\))",
                    en="Your code stops with an error at `{line}`: `X.shape[0]` is the number of photos, the rows of `X`. The layer gets the features of one photo: `X.shape[1]`, the length of a row.",
                    nl="Je code stopt met een fout bij `{line}`: `X.shape[0]` is het aantal foto's, de rijen van `X`. De layer krijgt de features van één foto: `X.shape[1]`, de lengte van een rij."),
            Mistake("swapped", syntax=r"nn\.Linear\(\s*k\s*,",
                    en="Your code stops with an error at `{line}`: `nn.Linear(k, ...)` takes `k` numbers in. The probe takes the features of a photo in and gives `k` logits out: `nn.Linear(X.shape[1], k)`.",
                    nl="Je code stopt met een fout bij `{line}`: `nn.Linear(k, ...)` neemt `k` getallen aan. De probe neemt de features van een foto aan en geeft `k` logits terug: `nn.Linear(X.shape[1], k)`."),
            Mistake("loss-order", syntax=r"loss_fn\(\s*y\s*,",
                    en="Your code stops with an error at `{line}`: `nn.CrossEntropyLoss` gets the logits first and the labels second: `loss_fn(probe(X), y)`.",
                    nl="Je code stopt met een fout bij `{line}`: `nn.CrossEntropyLoss` krijgt eerst de logits en dan de labels: `loss_fn(probe(X), y)`."),
            Mistake("loss-features", syntax=r"loss_fn\(\s*X\s*,",
                    en="Your code stops with an error at `{line}`: the loss gets the features themselves, so the probe plays no part and there is nothing to learn. Give it the probe's logits: `loss_fn(probe(X), y)`.",
                    nl="Je code stopt met een fout bij `{line}`: de loss krijgt de features zelf, dus de probe doet niet mee en er valt niets te leren. Geef hem de logits van de probe: `loss_fn(probe(X), y)`."),
            Mistake("argmax", syntax=r"loss_fn\(.*argmax",
                    en="Your code stops with an error at `{line}`: `argmax` gives the classes as whole numbers, and the loss needs the logits themselves, which it can follow back to the weights. Give it `probe(X)`.",
                    nl="Je code stopt met een fout bij `{line}`: `argmax` geeft de klassen als hele getallen, en de loss heeft de logits zelf nodig, die hij terug kan volgen naar de weights. Geef hem `probe(X)`."),
            Mistake("ten", wrong=_probe(layer="nn.Linear(X.shape[1], 10)"),
                    en=_CALL + ". Your probe always has 10 classes, as in the worked example; the check also tries other numbers of classes. Use `k`: `nn.Linear(X.shape[1], k)`.",
                    nl=_CALL_NL + ". Je probe heeft altijd 10 klassen, zoals in het uitgewerkte voorbeeld; de check probeert ook andere aantallen klassen. Gebruik `k`: `nn.Linear(X.shape[1], k)`."),
            Mistake("no-zero-grad", wrong=_probe(zero=False),
                    en=_CALL + ". Without `opt.zero_grad()`, every step adds its gradient to all the earlier ones, and the probe takes steps that are far too large. Keep `opt.zero_grad()` before `loss.backward()`.",
                    nl=_CALL_NL + ". Zonder `opt.zero_grad()` telt elke stap zijn gradient op bij alle eerdere, en zet de probe veel te grote stappen. Houd `opt.zero_grad()` voor `loss.backward()`."),
            Mistake("not-a-layer", matches=lambda got, text: not isinstance(got, list),
                    en="`{call}` gives {got}, and it should give the trained probe itself, the linear layer: keep `return probe` as the last line.",
                    nl="`{call}` geeft {got}, en dat moet de getrainde probe zelf zijn, de lineaire layer: houd `return probe` als laatste regel."),
            Mistake("other-random", matches=lambda got, text: isinstance(got, list) and len(got) == 2 and isinstance(got[1], list),
                    en=_CALL + ". The shape is right, but the probe ends up elsewhere. Keep the course's lines as they are, make only the one `nn.Linear`, right after `torch.manual_seed(0)`, and give the loss the probe's logits for all photos, `probe(X)`, and their labels, `y`.",
                    nl=_CALL_NL + ". De vorm klopt, maar de probe komt ergens anders uit. Laat de regels van de course zoals ze zijn, maak alleen die ene `nn.Linear`, direct na `torch.manual_seed(0)`, en geef de loss de logits van de probe voor alle foto's, `probe(X)`, en hun labels, `y`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "empty": {"en": "Replace the two `...`: `probe` is a linear layer from the features of a photo to `k` classes, and `loss` is the loss of its logits for all photos, `probe(X)`, against their labels, `y`.",
                      "nl": "Vervang de twee `...`: `probe` is een lineaire layer van de features van een foto naar `k` klassen, en `loss` is de loss van zijn logits voor alle foto's, `probe(X)`, tegen hun labels, `y`."},
            "arguments": {"en": "The check calls `{call}`, with the features, the labels and the number of classes, and the `def` line of your function has another number of names between its parentheses. Keep `(X, y, k)`.",
                          "nl": "De check roept `{call}` aan, met de features, de labels en het aantal klassen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(X, y, k)`."},
            "wrong": {"en": _CALL + ": a linear layer from the 4 features of a photo to the 3 classes, trained for 100 epochs. Step 4 of the recipe?",
                      "nl": _CALL_NL + ": een lineaire layer van de 4 features van een foto naar de 3 klassen, 100 epochs getraind. Stap 4 van het recept?"},
            "almost": {"en": "Right for the example. But " + _CALL + ". Does your probe take its numbers from `X` and `k`, for any number of features and classes?",
                       "nl": "Goed voor het voorbeeld. Maar " + _CALL_NL + ". Haalt je probe zijn getallen uit `X` en `k`, voor elk aantal features en klassen?"},
            "correct": {"en": "Your function is right in all {n} cases: a linear layer from the features to `k` classes, trained with cross-entropy. The demo below gives it DINOv2's features of the photos.",
                        "nl": "Je functie klopt in alle {n} gevallen: een lineaire layer van de features naar `k` klassen, getraind met cross-entropy. De demo hieronder geeft hem de features van DINOv2 voor de foto's."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("probe", _START, "empty", None),
    ("probe", _SOLUTION, "correct", None),
    ("probe", _probe(comments=True), "correct", None),
    ("probe", _probe(layer="nn.Linear(len(X[0]), k)", loss="nn.functional.cross_entropy(probe(X), y)"), "correct", None),
    ("probe", _probe(layer="nn.Linear(384, k)"), "error", "fixed-384"),
    ("probe", _probe(layer="nn.Linear(X.shape[0], k)"), "error", "rows"),
    ("probe", _probe(layer="nn.Linear(k, X.shape[1])"), "error", "swapped"),
    ("probe", _probe(loss="loss_fn(y, probe(X))"), "error", "loss-order"),
    ("probe", _probe(loss="loss_fn(X, y)"), "error", "loss-features"),
    ("probe", _probe(loss="loss_fn(probe(X).argmax(dim=1), y)"), "error", "argmax"),
    ("probe", _probe(layer="nn.Linear(X.shape[1], 10)"), "wrong", "ten"),
    ("probe", _probe(zero=False), "wrong", "no-zero-grad"),
    ("probe", _SOLUTION.replace("return probe", "return probe(X)"), "wrong", "not-a-layer"),
    ("probe", _SOLUTION.replace("    probe = nn.Linear(X.shape[1], k)\n", "    nn.Linear(4, 4)\n    probe = nn.Linear(X.shape[1], k)\n"),
     "wrong", "other-random"),
    ("probe", _probe(loss="..."), "wrong", "dots-left"),
]


# --- the demo: the learner's train_probe on DINOv2's features of the photos -------------------------------------------

def course_train_probe():
    return run_code(_SOLUTION).names["train_probe"]


def demo(model, text) -> dict:
    """The learner's train_probe once it is right, else the course's, on the features `model` (DINOv2) gives the
    first n photos of each class, for every n of the slider: how many of the 100 test photos the probe gets right,
    and whose function it was. If the learner's function fails here after all, the course's runs instead. The cell
    passes the model it got from dino_or_stop, so that it runs only once the downloads are there."""
    ran = run_code(text) if text else None
    fn = ran.names.get("train_probe") if ran is not None and ran.error is None else None
    yours = callable(fn)
    fn = fn if yours else course_train_probe()
    feats = features(model)

    def runs(f):
        out = {}
        for n in PER_CLASS:
            X, y, X_test, y_test = split(feats, n)
            out[n] = right(f(X, y, len(CLASSES)), X_test, y_test)
        return out
    try:
        return {"right": with_time_limit(lambda: runs(fn), 60), "yours": yours}
    except Exception:
        return {"right": with_time_limit(lambda: runs(course_train_probe()), 60), "yours": False}
