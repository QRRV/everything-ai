# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-try-classifier: top_k, the k most likely classes with their probabilities, and
top_k_accuracy, the share of photos whose label is among their k best guesses. Both are checked on small lists and
arrays, so they work without ResNet-18. ResNet-18 itself, with torchvision's weights trained on ImageNet, 47 MB, the
lesson downloads once from Hugging Face, when the learner asks for it (everything_ai.hub): the worked example needs
it, the exercises do not. The pictures and the slider use what the course measured with it (REFERENCE)."""
import errno
import functools
import html
import urllib.error

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai import hub, models
from everything_ai.lesson import download_card
from everything_ai.check import FunctionBox, Mistake
from everything_ai import yours

DOWNLOADS = {"ResNet-18": models.RESNET18}   # what the lesson downloads, for the line under its header
CLASSES = "data/imagenet-classes.txt"        # the names of ImageNet's 1,000 classes, in the order of the logits
PHOTOS = {"coffee": "data/photos/coffee.png", "cat": "data/photos/chelsea.png", "fruit": "data/photos/fruit.png"}

# What torchvision measured for these weights (ResNet18_Weights.IMAGENET1K_V1) on ImageNet's 50,000 validation
# photos, in percent: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
TOP1, TOP5 = 69.758, 89.078

# What ResNet-18 gives, measured with the course's code (everything_ai.models.resnet18), its pinned file and the
# preparation of cv-prepare-photo: the 10 most likely classes of each photo, with their probabilities, rounded.
# "upright" is the upright piece of the cat of cv-prepare-photo. tests/test_cv_try_classifier.py checks them again
# when EAI_RESNET18_FILE names the downloaded file.
REFERENCE = {
    "coffee": [("espresso", 0.8517), ("chocolate sauce", 0.0403), ("consomme", 0.0391), ("soup bowl", 0.0255), ("cup", 0.0202),
               ("coffee mug", 0.0118), ("eggnog", 0.0014), ("mixing bowl", 0.001), ("teapot", 0.0008), ("tray", 0.0008)],
    "cat": [("Egyptian cat", 0.3914), ("tiger cat", 0.3861), ("tabby", 0.2176), ("lynx", 0.0045), ("Persian cat", 0.0002),
            ("lens cap", 0.0), ("mouse", 0.0), ("remote control", 0.0), ("tiger", 0.0), ("computer keyboard", 0.0)],
    "upright": [("tiger cat", 0.5668), ("tabby", 0.2821), ("Egyptian cat", 0.1445), ("lynx", 0.0055), ("tiger", 0.0005),
                ("cheetah", 0.0003), ("leopard", 0.0002), ("Persian cat", 0.0001), ("snow leopard", 0.0), ("cougar", 0.0)],
    "fruit": [("balloon", 0.7349), ("ping-pong ball", 0.1971), ("Granny Smith", 0.0277), ("punching bag", 0.0254), ("croquet ball", 0.0068),
              ("planetarium", 0.0019), ("pool table", 0.0016), ("orange", 0.0012), ("maraca", 0.0006), ("basketball", 0.0004)],
}
# The label the lesson gives each photo: what a person who labels photos for ImageNet would choose.
LABELS = {"coffee": "espresso", "cat": "tabby", "upright": "tabby", "fruit": "orange"}
KS = list(range(1, 11))   # the stops of the slider


def share(p: float, sep: str) -> str:
    """A probability with 2 decimals: 0.39, or 0,39 in Dutch."""
    return f"{p:.2f}".replace(".", sep)


def percent(x: float, sep: str) -> str:
    """A percentage with at most one decimal: 99.5, or 99,5 in Dutch."""
    return f"{round(x, 1):g}".replace(".", sep)


def rank(photo: str, label: str | None = None) -> int:
    """Where the label of a photo stands among its guesses: 1 for the best guess; 0 when it is not in the 10."""
    names = [n for n, _ in REFERENCE[photo]]
    label = LABELS[photo] if label is None else label
    return names.index(label) + 1 if label in names else 0


def shown(p: float, sep: str) -> str:
    """A probability as a bar shows it: with 2 decimals, and "< 0.01" for what would be 0.00."""
    return share(p, sep) if p >= 0.005 else "< " + share(0.01, sep)


def held(photo: str, k: int) -> float:
    """The probability the k best guesses hold together."""
    return sum(p for _, p in REFERENCE[photo][:k])


def guesses(photo: str, k: int = 5) -> list[str]:
    return [n for n, _ in REFERENCE[photo][:k]]


def _short(label: str, n: int = 15) -> str:
    return label if len(label) <= n else label[:n - 1] + "…"


def bars(photo: str, k: int, labels: dict, sep: str, mark: str | None = None) -> str:
    """The k most likely classes of a photo: a bar per class, as long as its probability, with the probability at its
    end. The bars are what the model puts out, so they have the colour of a prediction; the best guess is full.
    Below 0.005 the number says "< 0.01".
    `mark` is the photo's label: its name is drawn bold, with a tick, where it is among the k. Drawn 330 wide, for a
    phone. `labels` has "title" and "alt"."""
    W, LW, BW, ROW, TOP = 330, 126, 150, 24, 30
    pairs = REFERENCE[photo][:k]
    top = max(p for _, p in pairs) or 1
    parts = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    for j, (name, p) in enumerate(pairs):
        y = TOP + j * ROW
        w = max(BW * p / top, 2)
        hit = name == mark
        text = ("✓ " if hit else "") + _short(name)
        parts.append(f'<text x="{LW - 8}" y="{y + 14}" style="font-size:13px;fill:var(--ea-ink);text-anchor:end;'
                     f'font-weight:{700 if hit else 400}">{html.escape(text)}</text>'
                     f'<rect x="{LW}" y="{y + 3}" width="{w:.1f}" height="15" rx="3" style="fill:var(--ea-pred);opacity:{1 if j == 0 else 0.45}"/>'
                     f'<text x="{LW + w + 6:.1f}" y="{y + 15}" style="font-size:12.5px;fill:var(--ea-pred-text)">{html.escape(shown(p, sep))}</text>')
    H = TOP + len(pairs) * ROW + 2
    alt = labels["title"] + ". " + labels["alt"]
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:min(100%,{W}px)">'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(alt)}" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(alt)}</title>' + "".join(parts) + "</svg></figure>")


# --- ResNet-18 in the lesson -----------------------------------------------------------------------------------------

def downloaded() -> bool:
    return hub.pinned(*models.RESNET18) is not None


@functools.lru_cache(maxsize=1)
def _resnet18():
    return models.resnet18()


def resnet_or_stop(button, texts: dict):
    """ResNet-18 for the lesson's cells. When it is not downloaded yet, the cell stops with `texts["ask"]` and the
    button; a click downloads it, with the progress in the cell. While it downloads, the lesson's other cells wait.
    `texts` has "ask", "busy" ({done} and {total} in MB), "no_net", "not_pinned", "no_room" and "failed" ({why}): why
    a download did not work; and "here"."""
    import marimo as mo
    failed = None
    if not downloaded() and button.value:
        total = round(hub.PINNED[models.RESNET18]["size"] / 1e6)
        last = [-1]

        def progress(done, size):
            mb = int(done / 1e6)
            if mb // 5 != last[0] // 5:   # every 5 MB
                last[0] = mb
                mo.output.replace(mo.md(texts["busy"].format(done=mb, total=total)))
        try:
            hub.download(*models.RESNET18, progress=progress)
        except hub.NotPinned:
            failed = texts["not_pinned"]
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            failed = texts["no_net"]
        except OSError as e:
            failed = texts["no_room"] if e.errno == errno.ENOSPC else texts["failed"].format(why=html.escape(str(e)))
    if not downloaded():
        mo.stop(True, download_card(texts["ask"], button, failed))
    model = _resnet18()
    mo.output.replace(mo.md(texts["here"]))
    return model


# The lesson is done when these are correct. "rank" is the choice question and "balloon" the open question.
REQUIRED = ["rank", "topk", "accuracy", "balloon"]


class _Arr(np.ndarray):
    """The probabilities of a case, shown in the check's messages as the code that makes them."""

    def __repr__(self):
        return "np.array([" + ", ".join(f"{float(v):g}" for v in self) + "])"


def _arr(*values) -> np.ndarray:
    return np.array(values, dtype=float).view(_Arr)


def _num(v):
    """A number as a message shows it: 0.39, not np.float64(0.39)."""
    if isinstance(v, (np.generic, float)):
        return f"{float(v):g}"
    return repr(v)


def _shown(v):
    """What the learner's function gave, without NumPy's np.float64(...) around the numbers."""
    if isinstance(v, np.ndarray):
        return "[" + ", ".join(_num(x) for x in v.tolist()) + "]"
    if isinstance(v, (list, tuple)):
        inner = ", ".join(_shown(x) for x in v)
        if isinstance(v, tuple):
            return "(" + inner + ("," if len(v) == 1 else "") + ")"
        return "[" + inner + "]"
    return _num(v)


_SAME = {"en": "`{call}` gives `{got}`, and it should give `{want}`.", "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn."}


def _m(key_, en, nl, **kw):
    """A mistake whose message starts with what the call gives and what it should give."""
    return Mistake(key_, en=_SAME["en"] + " " + en, nl=_SAME["nl"] + " " + nl, **kw)


# Exercise 2: the k most likely classes
_TOPK_HEAD = "def top_k(p, names, k):\n"
_TOPK_START = _TOPK_HEAD + "    pairs = ...\n    return ...\n"


def _topk(pairs: str = "[(p[i], names[i]) for i in range(len(names))]", back: str = "sorted(pairs, reverse=True)[:k]") -> str:
    return _TOPK_HEAD + f"    pairs = {pairs}\n    return {back}\n"


_TOPK = _topk()
_CATS = ["tabby", "tiger cat", "Egyptian cat", "lynx"]

# Exercise 3: top-k accuracy
_ACC_HEAD = "def top_k_accuracy(guesses, labels, k):\n"
_ACC_START = _ACC_HEAD + "    right = 0\n    for i in range(len(labels)):\n        ...\n    return ...\n"


def _acc(body: str = "if labels[i] in guesses[i][:k]:\n            right = right + 1", back: str = "right / len(labels)") -> str:
    return _ACC_HEAD + f"    right = 0\n    for i in range(len(labels)):\n        {body}\n    return {back}\n"


_ACC = _acc()


class _Named(list):
    """The guesses or labels of the four photos, shown in the check's messages by the name the lesson gives them."""

    def __init__(self, values, name: str):
        super().__init__(values)
        self.name = name

    def __repr__(self):
        return self.name


FOUR = ("coffee", "cat", "upright", "fruit")
_FOUR = _Named([guesses(p) for p in FOUR], "guesses")
_FOUR_LABELS = _Named([LABELS[p] for p in FOUR], "labels")
_OTHER = [["a", "b", "c"], ["b", "c", "a"], ["c", "a", "b"], ["a", "c", "b"], ["b", "a", "c"]]
_OTHER_LABELS = ["a", "a", "a", "b", "b"]


EXERCISES = {
    "topk": FunctionBox(
        name="topk",
        picture=yours.steps(call=0),
        function="top_k",
        start=_TOPK_START,
        solution=_TOPK,
        visible=[(_arr(0.22, 0.39, 0.38, 0.01), _CATS, 2)],
        hidden=[(_arr(0.1, 0.6, 0.3), ["fox", "dog", "cat"], 1), (_arr(0.05, 0.15, 0.5, 0.2, 0.1), ["a", "b", "c", "d", "e"], 3),
                (_arr(0.3, 0.7), ["yes", "no"], 2), (_arr(0.05, 0.3, 0.1, 0.2, 0.15, 0.12, 0.08), ["a", "b", "c", "d", "e", "f", "g"], 2)],
        show=_shown,
        mistakes=[
            _m("small", "`sorted` puts the pairs from small to large, so the first `k` are the least likely. `reverse=True` turns it around: from large to small.",
               "`sorted` zet de paren van klein naar groot, dus de eerste `k` zijn de minst waarschijnlijke. `reverse=True` draait het om: van groot naar klein.",
               wrong=_topk(back="sorted(pairs)[:k]")),
            _m("ascending", "These are the right `k` classes, but from small to large: the best guess should come first. Sort with `reverse=True` and take the first `k`.",
               "Dit zijn de goede `k` klassen, maar van klein naar groot: de beste gok hoort vooraan. Sorteer met `reverse=True` en neem de eerste `k`.",
               wrong=_topk(back="sorted(pairs)[-k:]")),
            _m("by-name", "Your pairs start with the name, so `sorted` orders them by name, not by probability. Put the probability first, `(p[i], names[i])`: tuples are sorted by their first value.",
               "Je paren beginnen met de naam, dus zet `sorted` ze op naam, niet op kans. Zet de kans vooraan, `(p[i], names[i])`: tuples worden gesorteerd op hun eerste waarde.",
               wrong=_topk(pairs="[(names[i], p[i]) for i in range(len(names))]")),
            _m("name-first", "These are the right classes in the right order, but each pair has the name first. Give back `(probability, name)`, as the pairs are made: `(p[i], names[i])`.",
               "Dit zijn de goede klassen in de goede volgorde, maar elk paar heeft de naam vooraan. Geef `(kans, naam)` terug, zoals de paren gemaakt worden: `(p[i], names[i])`.",
               wrong=_topk(back="[(n, q) for q, n in sorted(pairs, reverse=True)[:k]]")),
            _m("name-first-any", "Each of your pairs has the name first. `sorted` compares pairs by their first value, so put the probability first: `(p[i], names[i])`.",
               "Elk van je paren heeft de naam vooraan. `sorted` vergelijkt paren op hun eerste waarde, dus zet de kans vooraan: `(p[i], names[i])`.",
               matches=lambda got, text: isinstance(got, list) and len(got) > 0 and all(isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str) for x in got)),
            _m("one-pair", "That is one pair, the one at place `k`. `[:k]` gives the first `k`, as a list.",
               "Dat is één paar, dat op plek `k`. `[:k]` geeft de eerste `k`, als lijst.",
               matches=lambda got, text: isinstance(got, tuple)),
            _m("names-only", "Your function gives only the names. Give back the pairs, so that each guess has its probability with it.",
               "Je functie geeft alleen de namen. Geef de paren terug, zodat elke gok zijn kans bij zich heeft.",
               matches=lambda got, text: isinstance(got, list) and len(got) > 0 and all(isinstance(x, str) for x in got)),
            _m("probs-only", "Your function gives only the probabilities. Sort the pairs `(p[i], names[i])` instead of `p`, so that each probability keeps its name.",
               "Je functie geeft alleen de kansen. Sorteer de paren `(p[i], names[i])` in plaats van `p`, zodat elke kans zijn naam houdt.",
               matches=lambda got, text: isinstance(got, (list, np.ndarray)) and len(got) > 0 and all(isinstance(x, (float, np.floating)) for x in got)),
            _m("one-more", "That is one guess too many: `[:k]` takes the first `k`, from place 0 to place `k - 1`.",
               "Dat is één gok te veel: `[:k]` neemt de eerste `k`, van plek 0 tot en met plek `k - 1`.",
               wrong=_topk(back="sorted(pairs, reverse=True)[:k + 1]")),
            _m("one-less", "That is one guess too few: `[:k]` takes the first `k`, from place 0 to place `k - 1`.",
               "Dat is één gok te weinig: `[:k]` neemt de eerste `k`, van plek 0 tot en met plek `k - 1`.",
               wrong=_topk(back="sorted(pairs, reverse=True)[:k - 1]")),
            _m("all", "Your function gives all the classes. Take only the first `k` of the sorted pairs: `[:k]`.",
               "Je functie geeft alle klassen. Neem alleen de eerste `k` van de gesorteerde paren: `[:k]`.",
               wrong=_topk(back="sorted(pairs, reverse=True)")),
            _m("five", "Your function always gives 5 guesses, whatever `k` is. Take the first `k`: `[:k]`.",
               "Je functie geeft altijd 5 gokken, wat `k` ook is. Neem de eerste `k`: `[:k]`.",
               wrong=_topk(back="sorted(pairs, reverse=True)[:5]")),
            Mistake("over-names", syntax=r"for \w+ in names\b",
                    en="Your code stops with an error at `{line}`: `for ... in names` gives the names themselves, such as `'tabby'`, not their places, so `p[...]` cannot look them up. Loop over the places: `for i in range(len(names))`.",
                    nl="Je code stopt met een fout bij `{line}`: `for ... in names` geeft de namen zelf, zoals `'tabby'`, niet hun plekken, dus kan `p[...]` ze niet opzoeken. Loop over de plekken: `for i in range(len(names))`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace each `...`: `pairs` is a list with a pair `(p[i], names[i])` per class, and the function gives back the `k` pairs with the highest probability, largest first.",
                      "nl": "Vervang elke `...`: `pairs` is een lijst met een paar `(p[i], names[i])` per klasse, en de functie geeft de `k` paren met de hoogste kans terug, de grootste eerst."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the probabilities, the names and `k`, and the `def` line of your function has a different number of names between its parentheses. Keep `(p, names, k)`.",
                          "nl": "De check roept `{call}` aan, met de kansen, de namen en `k`, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(p, names, k)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: the 2 classes with the highest probability, as pairs of probability and name, the best guess first.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn: de 2 klassen met de hoogste kans, als paren van kans en naam, de beste gok eerst."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for every `k`, and for any number of classes?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elke `k`, en voor elk aantal klassen?"},
            "correct": {"en": "Your function is right in all {n} cases. On ResNet-18's 1,000 probabilities for the coffee photo, `top_k(p, names, 5)` gives espresso first, at 0.85.",
                        "nl": "Je functie klopt in alle {n} gevallen. Op de 1000 kansen van ResNet-18 voor de koffiefoto geeft `top_k(p, names, 5)` espresso eerst, met 0,85."},
        },
    ),
    "accuracy": FunctionBox(
        name="accuracy",
        picture=yours.steps(call=0),
        function="top_k_accuracy",
        start=_ACC_START,
        solution=_ACC,
        visible=[(_FOUR, _FOUR_LABELS, 1)],
        hidden=[(_FOUR, _FOUR_LABELS, 5), (_FOUR, _FOUR_LABELS, 2), (_OTHER, _OTHER_LABELS, 2), (_OTHER, _OTHER_LABELS, 3)],
        mistakes=[
            Mistake("index", error=lambda e: isinstance(e, IndexError),
                    en="With `{call}`, your function stops with an error: a list of 5 guesses has places 0 to 4, so place 5 does not exist. And `guesses[i][k]` is only one guess, the one at place `k`. The first `k` guesses are a slice: `guesses[i][:k]`.",
                    nl="Met `{call}` stopt je functie met een fout: een lijst van 5 gokken heeft de plekken 0 tot en met 4, dus plek 5 bestaat niet. En `guesses[i][k]` is maar één gok, die op plek `k`. De eerste `k` gokken zijn een slice: `guesses[i][:k]`."),
            _m("count", "That is how many photos are right, not which share of them. Divide by the number of photos: `right / len(labels)`.",
               "Dat is hoeveel foto's goed zijn, niet welk deel ervan. Deel door het aantal foto's: `right / len(labels)`.",
               wrong=_acc(back="right")),
            _m("first-only", "Your function looks only at the best guess, whatever `k` is. Look at the first `k` guesses: `labels[i] in guesses[i][:k]`.",
               "Je functie kijkt alleen naar de beste gok, wat `k` ook is. Kijk naar de eerste `k` gokken: `labels[i] in guesses[i][:k]`.",
               wrong=_acc(body="if labels[i] == guesses[i][0]:\n            right = right + 1")),
            _m("all-guesses", "Your function looks at all the guesses of a photo, whatever `k` is. Only the first `k` count: `guesses[i][:k]`.",
               "Je functie kijkt naar alle gokken van een foto, wat `k` ook is. Alleen de eerste `k` tellen: `guesses[i][:k]`.",
               wrong=_acc(body="if labels[i] in guesses[i]:\n            right = right + 1")),
            _m("one-less", "`guesses[i][:k - 1]` holds only `k - 1` guesses. The first `k` are `guesses[i][:k]`.",
               "`guesses[i][:k - 1]` bevat maar `k - 1` gokken. De eerste `k` zijn `guesses[i][:k]`.",
               wrong=_acc(body="if labels[i] in guesses[i][:k - 1]:\n            right = right + 1")),
            _m("one-more", "`guesses[i][:k + 1]` holds `k + 1` guesses. The first `k` are `guesses[i][:k]`.",
               "`guesses[i][:k + 1]` bevat `k + 1` gokken. De eerste `k` zijn `guesses[i][:k]`.",
               wrong=_acc(body="if labels[i] in guesses[i][:k + 1]:\n            right = right + 1")),
            _m("per-k", "You divide by `k`. Top-k accuracy is a share of the photos: divide by the number of photos, `len(labels)`.",
               "Je deelt door `k`. Top-k accuracy is een deel van de foto's: deel door het aantal foto's, `len(labels)`.",
               wrong=_acc(back="right / k")),
            _m("four", "You divide by 4, the number of photos in the example; the check also tries other photos. Divide by `len(labels)`.",
               "Je deelt door 4, het aantal foto's in het voorbeeld; de check probeert ook andere foto's. Deel door `len(labels)`.",
               wrong=_acc(back="right / 4")),
            _m("percent", "That is a percentage. A share lies between 0 and 1: leave out the `* 100`.",
               "Dat is een percentage. Een deel ligt tussen 0 en 1: laat de `* 100` weg.",
               wrong=_acc(back="right / len(labels) * 100")),
            _m("floor", "`//` rounds down to a whole number, so a share such as 0.25 becomes 0. Divide with `/`.",
               "`//` rondt af naar beneden op een heel getal, dus een deel zoals 0,25 wordt 0. Deel met `/`.",
               wrong=_acc(back="right // len(labels)")),
            _m("wrong-share", "That is the share of photos that are wrong. Count the photos whose label is among the first `k` guesses.",
               "Dat is het deel van de foto's dat fout is. Tel de foto's waarvan het label bij de eerste `k` gokken staat.",
               wrong=_acc(body="if labels[i] not in guesses[i][:k]:\n            right = right + 1")),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace each `...`: in the loop, add 1 to `right` when the label of photo `i` is among its first `k` guesses, and give back the share of the photos that is right.",
                      "nl": "Vervang elke `...`: tel in de loop 1 op bij `right` als het label van foto `i` bij zijn eerste `k` gokken staat, en geef het deel van de foto's terug dat goed is."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the guesses, the labels and `k`, and the `def` line of your function has a different number of names between its parentheses. Keep `(guesses, labels, k)`.",
                          "nl": "De check roept `{call}` aan, met de gokken, de labels en `k`, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(guesses, labels, k)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. With `k` 1 only the best guess counts: the coffee photo is right, the two cats and the fruit are not, so 1 of the 4.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Met `k` 1 telt alleen de beste gok: de koffiefoto is goed, de twee katten en het fruit niet, dus 1 van de 4."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function look at exactly the first `k` guesses, for every `k`, and divide by the number of photos, `len(labels)`?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Kijkt je functie naar precies de eerste `k` gokken, voor elke `k`, en deelt hij door het aantal foto's, `len(labels)`?"},
            "correct": {"en": "Your function is right in all {n} cases. On the four photos, top-1 accuracy is 0.25 and top-5 accuracy 0.75: the upright piece of the cat counts from `k` 2, the whole cat from `k` 3, and the fruit not even in the top 5.",
                        "nl": "Je functie klopt in alle {n} gevallen. Op de vier foto's is de top-1 accuracy 0,25 en de top-5 accuracy 0,75: het staande stuk van de kat telt mee vanaf `k` 2, de hele kat vanaf `k` 3, en het fruit niet eens in de top 5."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("topk", _TOPK_START, "empty", None),
    ("topk", _TOPK, "correct", None),
    ("topk", _TOPK_HEAD + "    return sorted([(p[i], names[i]) for i in range(len(p))])[::-1][:k]\n", "correct", None),
    ("topk", _TOPK_HEAD + "    pairs = []\n    for i in range(len(names)):\n        pairs.append((p[i], names[i]))\n"
             "    return sorted(pairs, reverse=True)[:k]\n", "correct", None),
    ("topk", _topk(back="sorted(pairs)[:k]"), "wrong", "small"),
    ("topk", _topk(back="sorted(pairs)[-k:]"), "wrong", "ascending"),
    ("topk", _topk(pairs="[(names[i], p[i]) for i in range(len(names))]"), "wrong", "by-name"),
    ("topk", _topk(back="[(n, q) for q, n in sorted(pairs, reverse=True)[:k]]"), "wrong", "name-first"),
    ("topk", _topk(back="[n for q, n in sorted(pairs, reverse=True)[:k]]"), "wrong", "names-only"),
    ("topk", _topk(back="sorted(p, reverse=True)[:k]"), "wrong", "probs-only"),
    ("topk", _topk(back="sorted(pairs, reverse=True)[:k + 1]"), "wrong", "one-more"),
    ("topk", _topk(back="sorted(pairs, reverse=True)[:k - 1]"), "wrong", "one-less"),
    ("topk", _topk(back="sorted(pairs, reverse=True)"), "wrong", "all"),
    ("topk", _topk(back="sorted(pairs, reverse=True)[:5]"), "wrong", "five"),
    ("topk", _topk(back="sorted(pairs, reverse=True)[k]"), "wrong", "one-pair"),
    ("topk", _topk(pairs="[(names[i], p[i]) for i in range(len(names))]", back="sorted(pairs)[:k]"), "wrong", "name-first-any"),
    ("topk", _topk(pairs="[(p[i], names[i]) for i in names]"), "error", "over-names"),
    ("accuracy", _ACC_START, "empty", None),
    ("accuracy", _ACC, "correct", None),
    ("accuracy", _ACC_HEAD + "    right = 0\n    for i in range(len(labels)):\n        right = right + (labels[i] in guesses[i][:k])\n"
                 "    return right / len(labels)\n", "correct", None),
    ("accuracy", _acc(body="if labels[i] == guesses[i][k]:\n            right = right + 1"), "error", "index"),
    ("accuracy", _ACC_HEAD + "    right = 0\n    for i in range(len(labels)):\n        if labels[i] in guesses[i][:k]:\n            right = right + 1\n"
                 "    return right\n", "wrong", "count"),
    ("accuracy", _acc(body="if labels[i] == guesses[i][0]:\n            right = right + 1"), "wrong", "first-only"),
    ("accuracy", _acc(body="if labels[i] in guesses[i]:\n            right = right + 1"), "wrong", "all-guesses"),
    ("accuracy", _acc(body="if labels[i] in guesses[i][:k - 1]:\n            right = right + 1"), "wrong", "one-less"),
    ("accuracy", _acc(body="if labels[i] in guesses[i][:k + 1]:\n            right = right + 1"), "wrong", "one-more"),
    ("accuracy", _acc(back="right / k"), "wrong", "per-k"),
    ("accuracy", _acc(back="right // len(labels)"), "wrong", "floor"),
    ("accuracy", _acc(back="right / 4"), "wrong", "four"),
    ("accuracy", _acc(back="right / len(labels) * 100"), "wrong", "percent"),
    ("accuracy", _acc(back="100 * right / len(labels)"), "wrong", "percent"),
    ("accuracy", _acc(body="if labels[i] not in guesses[i][:k]:\n            right = right + 1"), "wrong", "wrong-share"),
]
