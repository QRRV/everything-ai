# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson res-formula-to-code: smoothed_loss, the cross-entropy with label smoothing of Szegedy and
others (2016), written from the formula."""
import math
import re

from everything_ai.check import FunctionBox, Mistake


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def targets_picture(labels: dict) -> str:
    """The target of one example with K = 5 classes and the right class 2: the one-hot target, a faint bar, and the
    smoothed target with ε = 0.2, a strong bar (four washes) in the data colour, side by side for every class.
    Targets are data."""
    from everything_ai import chart
    K, Y, EPS = 5, 2, 0.2
    p = chart.Panel((-0.6, K - 0.4), (0, 1.12), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                    xticks=list(range(K)), yticks=[0, 0.5, 1])

    def bar(x0, x1, h, strong):
        for _ in range(4 if strong else 1):
            p.rect(x0, 0, x1, h, "data" if strong else "ink-3")

    for k in range(K):
        hard = 1.0 if k == Y else 0.0
        if hard:
            bar(k - 0.36, k - 0.02, hard, False)
        bar(k + 0.02, k + 0.36, (1 - EPS) * hard + EPS / K, True)
    p.text(Y - 0.19, 1.07, "1", "ink-2", "middle")
    p.text(Y + 0.41, 0.79, labels["right"], "data", "start")
    p.text(0.19, 0.12, labels["rest"], "data", "middle")
    p.rect(-0.5, 0.9, -0.32, 0.98, "ink-3")
    p.text(-0.22, 0.91, labels["hard"], "ink-2", "start")
    for _ in range(4):
        p.rect(-0.5, 0.76, -0.32, 0.84, "data")
    p.text(-0.22, 0.77, labels["soft"], "ink-2", "start")
    svg = chart.figure([p], alt=labels["alt"], width=330, height=260)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


# --- exercise 1: smoothed_loss --------------------------------------------------------------------------------------------------

_HEAD = ("import torch\n\n\ndef smoothed_loss(z, y, eps):\n"
         "    # z: the logits, (B, K); y: the right classes, (B,); eps: the epsilon of the formula\n")
_SETUP = "    K = z.shape[1]\n    p = torch.softmax(z, dim=1)\n"
_Q = "    q = (1 - eps) * torch.eye(K)[y] + eps / K\n"
_LOSS = "    return -(q * torch.log(p)).sum(dim=1).mean()\n"
_START = _HEAD + "    # the smoothed target q' and the loss H(q', p), averaged over the batch\n    ...\n"


def _s(setup: str = _SETUP, q: str = _Q, loss: str = _LOSS) -> str:
    return _HEAD + setup + q + loss


class _Named:
    """A tensor of a test case, written in the check's messages the way the lesson writes it; made with seed `seed`."""

    def __init__(self, shown: str, seed: int = 0):
        self.shown, self.seed = shown, seed

    def __repr__(self):
        return self.shown

    def __deepcopy__(self, memo):
        import torch
        torch.manual_seed(self.seed)
        return eval(self.shown, {"torch": torch, "math": math})


def _value(v):
    """The number in what the function gives back: a float, or a tensor with one number; None for anything else."""
    try:
        import torch
        if isinstance(v, torch.Tensor):
            return float(v.item()) if v.numel() == 1 and v.dim() == 0 else None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    except Exception:
        return None
    return None


def _close(got, want) -> bool:
    g, w = _value(got), _value(want)
    # abs_tol 1e-4: two numbers that differ by more always look different with 4 decimals, as {got} and {want} show them
    return g is not None and w is not None and math.isclose(g, w, rel_tol=1e-5, abs_tol=1e-4)


def _show(v):
    """What the function gave back, in a message: a number with 4 decimals, the shape of a tensor or an array, or the
    kind of a list or tuple; never float noise."""
    x = _value(v)
    if x is not None:
        text = f"{round(x, 4) + 0.0:.4f}"   # + 0.0: a tiny negative number shows as 0.0000, not -0.0000
        return {"en": text, "nl": text.replace(".", ",")}
    shape = getattr(v, "shape", None)
    if shape is not None:
        kind = "tensor" if type(v).__name__ == "Tensor" else "array"
        return {"en": f"{'a' if kind == 'tensor' else 'an'} {kind} of shape {tuple(shape)}", "nl": f"een {kind} met shape {tuple(shape)}"}
    if isinstance(v, (list, tuple)):
        kind, n = type(v).__name__, len(v)
        return {"en": f"a {kind} of {n} value{'' if n == 1 else 's'}", "nl": f"een {kind} met {n} waarde{'' if n == 1 else 'n'}"}
    return repr(v)


def _per_example(got, text) -> bool:
    """One loss per example, on the visible case, where B = 1: a tensor or array of shape (1,). A shape (K,) is not."""
    shape = getattr(got, "shape", None)
    return shape is not None and tuple(shape) == (1,)


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^\s*\.\.\.\s*(#.*)?$", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_WRONG = {
    "others": _s(q="    hot = torch.eye(K)[y]\n    q = (1 - eps) * hot + eps / (K - 1) * (1 - hot)\n"),
    "no-uniform": _s(q="    q = (1 - eps) * torch.eye(K)[y]\n"),
    "no-shrink": _s(q="    q = torch.eye(K)[y] + eps / K\n"),
    "sum-batch": _s(loss="    return -(q * torch.log(p)).sum()\n"),
    "sign": _s(loss="    return (q * torch.log(p)).sum(dim=1).mean()\n"),
}

EXERCISES = {
    "smoothed-loss": FunctionBox(
        name="smoothed-loss",
        function="smoothed_loss",
        start=_START,
        solution=_s(),
        visible=[(_Named("torch.tensor([[math.log(3), 0.0]])"), _Named("torch.tensor([0])"), 0.2)],
        hidden=[(_Named("torch.randn(4, 5)", 1), _Named("torch.tensor([2, 0, 4, 1])"), 0.1),
                (_Named("torch.randn(3, 10)", 2), _Named("torch.tensor([7, 7, 0])"), 0.0),
                (_Named("torch.randn(2, 3)", 3), _Named("torch.tensor([1, 2])"), 0.3)],
        show=_show,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("per-example", matches=_per_example,
                    en="`{call}` gives {got}: one loss per example. The exercise asks for one number for the batch, as PyTorch gives: take their mean with `.mean()`.",
                    nl="`{call}` geeft {got}: één loss per voorbeeld. De oefening vraagt één getal voor de batch, zoals PyTorch geeft: neem er het gemiddelde van met `.mean()`."),
            Mistake("others", wrong=_WRONG["others"],
                    en="`{call}` gives {got}, and it should give {want}: you spread ε over the other K − 1 classes only. That is a variant, the one in the transformer's own code, but in this formula ε u(k), with u(k) = 1/K, goes to every class, the right one too: `eps / K` added to all.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je verdeelt ε alleen over de andere K − 1 klassen. Dat is een variant, die in de eigen code van de transformer, maar in deze formule gaat ε u(k), met u(k) = 1/K, naar elke klasse, ook de juiste: `eps / K` bij allemaal opgeteld."),
            Mistake("no-uniform", wrong=_WRONG["no-uniform"],
                    en="`{call}` gives {got}, and it should give {want}: your target is only (1 − ε) δ. The second term, ε u(k) with u(k) = 1/K, is missing: add `eps / K` to every class.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je target is alleen (1 − ε) δ. De tweede term, ε u(k) met u(k) = 1/K, ontbreekt: tel `eps / K` bij elke klasse op."),
            Mistake("no-shrink", wrong=_WRONG["no-shrink"],
                    en="`{call}` gives {got}, and it should give {want}: your target adds up to 1 + ε. The one-hot part is multiplied by (1 − ε) in the formula, so that the target stays a distribution.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je target telt op tot 1 + ε. Het one-hot-deel wordt in de formule met (1 − ε) vermenigvuldigd, zodat het target een verdeling blijft."),
            Mistake("sum-batch", wrong=_WRONG["sum-batch"],
                    en="`{call}` gives {got}, and it should give {want}: you add up the losses of all examples. Sum over the classes, `.sum(dim=1)`, and take the mean over the batch, as PyTorch does.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt de losses van alle voorbeelden op. Tel op over de klassen, `.sum(dim=1)`, en neem het gemiddelde over de batch, zoals PyTorch doet."),
            Mistake("sign", wrong=_WRONG["sign"],
                    en="`{call}` gives {got}, and it should give {want}: the minus sign is missing. The log of a probability is negative or 0, and H(q', p) starts with −.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het minteken ontbreekt. De log van een kans is negatief of 0, en H(q', p) begint met −."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with your code: the smoothed target, and the loss averaged over the batch.",
                      "nl": "Vervang de `...` door je code: het verzachte target, en de loss gemiddeld over de batch."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the logits, the right classes and ε, and the `def` line of your function has another number of names between its parentheses. Keep `(z, y, eps)`.",
                          "nl": "De check roept `{call}` aan, met de logits, de juiste klassen en ε, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(z, y, eps)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Go through the symbols of the formula one by one: which tensor is each, with which shape?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Loop de tekens van de formule één voor één langs: welke tensor is elk, met welke shape?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for more examples at once, more classes, and ε = 0?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie ook voor meer voorbeelden tegelijk, meer klassen en ε = 0?"},
            "correct": {"en": "Your function gives the paper's loss in all {n} cases, also with ε = 0, where it is the plain cross-entropy.",
                        "nl": "Je functie geeft de loss van de paper in alle {n} gevallen, ook met ε = 0, waar het de gewone cross-entropy is."},
        },
    ),
}

# The lesson is done when these are correct. "factor-four" is the choice question, "why-toy" the open question.
REQUIRED = ["smoothed-loss", "factor-four", "why-toy"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("smoothed-loss", _START, "empty", None),
    ("smoothed-loss", _s(), "correct", None),
    ("smoothed-loss", _HEAD + "    import torch.nn as nn\n    return nn.CrossEntropyLoss(label_smoothing=eps)(z, y)\n", "correct", None),
    ("smoothed-loss", _HEAD + "    K = z.shape[1]\n    logp = torch.log(torch.softmax(z, dim=1))\n    hot = torch.eye(K)[y]\n"
     "    loss = -((1 - eps) * (hot * logp).sum(dim=1) + eps / K * logp.sum(dim=1))\n    return loss.mean()\n", "correct", None),
    ("smoothed-loss", _s(loss="    return -(q * torch.log(p)).sum(dim=1)\n"), "wrong", "per-example"),
    ("smoothed-loss", _s(loss="    return round(-(q * torch.log(p)).sum(dim=1).mean().item(), 4)\n"), "correct", None),
    ("smoothed-loss", _s(loss="    return -(q * torch.log(p)).sum(dim=0)\n"), "wrong", None),
    ("smoothed-loss", _s(loss="    return (-(q * torch.log(p)).sum(dim=1)).tolist()\n"), "wrong", None),
    ("smoothed-loss", _s(setup="    K = z.shape[1]\n    p = torch.softmax(z, dim=0)\n"), "wrong", None),
] + [("smoothed-loss", code, "wrong", key) for key, code in _WRONG.items()]
