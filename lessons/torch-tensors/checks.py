# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson torch-tensors: the predictions and the mean squared error of a linear model with tensors,
and the loss as a Python number with `.item()`; and every row of a tensor of whole numbers minus its own mean, with
float32, `dim` and `keepdim`. And the picture: three tensors of the worked example, with 0, 1 and 2 axes."""
import re

import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai.check import CodeBox, Code, Mistake, same_value

# --- the picture: the tensors of the worked example ---------------------------------------------------------------

def picture(notes: list[str]) -> list[tuple]:
    """The grids for draw.array_grids: the loss (0 axes), y_pred (1 axis) and X (2 axes), each number as PyTorch
    prints it, with its shape. `notes` are the second caption lines, in the lesson's language."""
    X = [["1.", "2."], ["3.", "1."], ["0.", "2."], ["2.", "2."]]
    return [("loss", [["0.4375"]], ["torch.Size([])", notes[0]]),
            ("y_pred", [["4.5000", "5.0000", "3.5000", "5.5000"]], ["torch.Size([4])", notes[1]]),
            ("X", X, ["torch.Size([4, 2])", notes[2]])]


# The lesson is done when these are correct. "same-device" is the choice question in the lesson.
REQUIRED = ["same-device", "loss", "centered"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_NUMPY = Mistake("numpy", syntax=r"\bnp\.",
                 en="Use the tensor's own methods, such as `t.mean()` and `t.sum()`. NumPy's `np.mean` and `np.sum` do not work on a tensor, and this box needs no NumPy.",
                 nl="Gebruik de eigen methods van de tensor, zoals `t.mean()` en `t.sum()`. `np.mean` en `np.sum` van NumPy werken niet op een tensor, en dit vak heeft NumPy niet nodig.")

_KEEP_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
    "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back.",
             "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug."},
}

# --- exercise 2: the loss as a number --------------------------------------------------------------------------------

# y_pred is (3, 5, 2), so the residuals y - y_pred are 1, -2 and 1: they add up to 0, so a mean without squares gives 0
# in either order, and every known mistake gives an outcome of its own.
_L_HEAD = ("import torch\n\nX = torch.tensor([[2, 1], [0, 3], [1, 1]], dtype=torch.float32)\n"
           "y = torch.tensor([4, 3, 3], dtype=torch.float32)\nw = torch.tensor([1.0, 2.0])\nb = -1.0\n\n")
_L_START = _L_HEAD + "y_pred = ...\nloss = ...\nloss\n"


def _l(pred: str = "X @ w + b", loss: str = "((y - y_pred) ** 2).mean().item()") -> str:
    return _L_HEAD + f"y_pred = {pred}\nloss = {loss}\nloss"


_L = _l()


def _number(got, want) -> bool:
    """The loss as a Python number: a tensor, even with the right value in it, is not one yet. That is what
    `.item()` is for, so this exercise does not count a tensor with no axes as its number."""
    return not isinstance(got, torch.Tensor) and same_value(got, want)


# --- exercise 3: every row minus its own mean ------------------------------------------------------------------------

_C_HEAD = "import torch\n\nscores = torch.tensor([[6, 8, 7], [4, 9, 5]])\n"
_C_START = _C_HEAD + "centered = ...\ncentered\n"


def _c(line: str) -> str:
    return _C_HEAD + line + "\ncentered"


_C = _c("centered = scores - scores.float().mean(dim=1, keepdim=True)")

EXERCISES = {
    "loss": CodeBox(
        name="loss",
        start=_L_START,
        solution=_L,
        keep=["X = torch.tensor([[2, 1], [0, 3], [1, 1]], dtype=torch.float32)", "y = torch.tensor([4, 3, 3], dtype=torch.float32)"],
        compare=_number,
        # other weights and another bias term, so a typed number shows
        others=[{"w": Code("torch.tensor([0.5, -1.0])"), "b": Code("0.5")}, {"b": Code("2.0")}],
        digits=4,
        first=[_DOTS],
        mistakes=[
            _NUMPY,
            Mistake("order", error=lambda e: isinstance(e, RuntimeError) and "cannot be multiplied" in str(e),
                    en="PyTorch stops: `w @ X` puts a vector of 2 numbers in front of a matrix with 3 rows, and that does not fit. Write the shapes next to it: `X @ w` is (3, 2) @ (2,), and gives one prediction per example, shape (3,).",
                    nl="PyTorch stopt: `w @ X` zet een vector van 2 getallen voor een matrix met 3 rijen, en dat past niet. Schrijf de shapes erbij: `X @ w` is (3, 2) @ (2,), en geeft één voorspelling per voorbeeld, shape (3,)."),
            Mistake("tensor", matches=lambda got, text: isinstance(got, torch.Tensor) and got.dim() == 0,
                    en="Your last line gives `{got}`: a tensor with no axes, not a Python number. `.item()` takes the number out: `loss = (...).item()`.",
                    nl="Je laatste regel geeft `{got}`: een tensor zonder axes, geen Python-getal. `.item()` haalt het getal eruit: `loss = (...).item()`."),
            Mistake("vector", matches=lambda got, text: isinstance(got, torch.Tensor) and got.dim() > 0,
                    en="Your last line gives `{got}`: a number per example, not one loss. The mean squared error is the mean of the squares: `.mean()`, and then `.item()`.",
                    nl="Je laatste regel geeft `{got}`: een getal per voorbeeld, niet één loss. De mean squared error is het gemiddelde van de kwadraten: `.mean()`, en dan `.item()`."),
            Mistake("sum", wrong=_l(loss="((y - y_pred) ** 2).sum().item()"),
                    en="Your last line gives `{got}`: the sum of the squared residuals. The mean squared error divides by the number of examples: `.mean()` instead of `.sum()`.",
                    nl="Je laatste regel geeft `{got}`: de som van de gekwadrateerde residuen. De mean squared error deelt door het aantal voorbeelden: `.mean()` in plaats van `.sum()`."),
            Mistake("no-square", wrong=_l(loss="(y - y_pred).mean().item()"),
                    en="Your last line gives `{got}`: the mean of the residuals themselves. Here they add up to 0, because the positive and the negative ones cancel out. Square them first: `((y - y_pred) ** 2).mean()`.",
                    nl="Je laatste regel geeft `{got}`: het gemiddelde van de residuen zelf. Hier tellen ze op tot 0, omdat de positieve en de negatieve elkaar opheffen. Kwadrateer ze eerst: `((y - y_pred) ** 2).mean()`."),
        ],
        texts={
            **_KEEP_TEXTS,
            "empty": {"en": "Replace the two `...`: the predictions `X @ w + b`, and the mean squared error as a Python number.",
                      "nl": "Vervang de twee `...`: de voorspellingen `X @ w + b`, en de mean squared error als Python-getal."},
            "no_value": {"en": "The last line of the box shows nothing. Put `loss` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `loss` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`, and it should be 2. Do you work out `y_pred` as `X @ w + b`, then the mean of the squared residuals, and then `.item()`?",
                      "nl": "Je laatste regel geeft `{got}`, en dat moet 2 zijn. Reken je `y_pred` uit als `X @ w + b`, dan het gemiddelde van de gekwadrateerde residuen, en dan `.item()`?"},
            "other_error": {"en": "Right for these numbers. But with {where}, Python stops with an error. Does your code work for other weights and bias terms?",
                            "nl": "Goed voor deze getallen. Maar met {where} stopt Python met een fout. Werkt je code ook voor andere weights en bias terms?"},
            "other": {"en": "Right for these numbers. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you use `w` and `b` themselves, rather than numbers you typed?",
                      "nl": "Goed voor deze getallen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruik je `w` en `b` zelf, en geen getallen die je hebt ingetypt?"},
            "correct": {"en": "The loss is 2, now as a Python number. The check also tried other weights and bias terms.",
                        "nl": "De loss is 2, nu als Python-getal. De check probeerde ook andere weights en bias terms."},
        },
    ),
    "centered": CodeBox(
        name="centered",
        start=_C_START,
        solution=_C,
        keep=[],
        # three rows of two: a mean along the wrong axis, or a typed number, shows
        # floats: the lesson's own scores already try whole numbers, and code that makes `scores` floats on its own
        # line, as step 1 of the recipe does, stays right when the check puts other scores in that line
        others=[{"scores": Code("torch.tensor([[1.0, 2.0], [3.0, 7.0], [0.0, 0.0]])")}],
        first=[_DOTS],
        mistakes=[
            _NUMPY,
            Mistake("whole-numbers", error=lambda e: isinstance(e, RuntimeError) and "could not infer output dtype" in str(e),
                    en="PyTorch stops: `scores` holds whole numbers, dtype int64, and PyTorch takes a mean of floats only. Make them floats first: `scores.float()`.",
                    nl="PyTorch stopt: `scores` bevat gehele getallen, dtype int64, en PyTorch neemt alleen een gemiddelde van floats. Maak er eerst floats van: `scores.float()`."),
            Mistake("no-keepdim", error=lambda e: isinstance(e, RuntimeError) and "must match the size" in str(e),
                    en="PyTorch stops: `mean(dim=1)` gives one number per row, shape (2,), and (2, 3) minus (2,) does not fit: lined up on the right, 3 and 2 differ. Keep the means as a column with `keepdim=True`: shape (2, 1).",
                    nl="PyTorch stopt: `mean(dim=1)` geeft één getal per rij, shape (2,), en (2, 3) min (2,) past niet: rechts uitgelijnd verschillen 3 en 2. Houd de gemiddelden als kolom met `keepdim=True`: shape (2, 1)."),
            Mistake("columns", wrong=_c("centered = scores - scores.float().mean(dim=0, keepdim=True)"),
                    en="Your last line gives `{got}`: each column minus its own mean. `dim=0` runs down the rows, so it gives one mean per column. One mean per row runs along each row: `dim=1`.",
                    nl="Je laatste regel geeft `{got}`: elke kolom min zijn eigen gemiddelde. `dim=0` loopt langs de rijen omlaag, dus het geeft één gemiddelde per kolom. Eén gemiddelde per rij loopt langs elke rij: `dim=1`."),
            Mistake("all", wrong=_c("centered = scores - scores.float().mean()"),
                    en="Your last line gives `{got}`: every number minus the mean of all six. Each row needs its own mean: `mean(dim=1, keepdim=True)`.",
                    nl="Je laatste regel geeft `{got}`: elk getal min het gemiddelde van alle zes. Elke rij heeft zijn eigen gemiddelde nodig: `mean(dim=1, keepdim=True)`."),
        ],
        texts={
            **_KEEP_TEXTS,
            "empty": {"en": "Replace the `...` with `scores` minus the mean of each row, kept as a column.",
                      "nl": "Vervang de `...` door `scores` min het gemiddelde van elke rij, als kolom gehouden."},
            "no_value": {"en": "The last line of the box shows nothing. Put `centered` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `centered` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`, and it should be `tensor([[-1., 1., 0.], [-2., 3., -1.]])`. Do you take the mean of each row with `dim=1`, keep it as a column with `keepdim=True`, and take it away from `scores`?",
                      "nl": "Je laatste regel geeft `{got}`, en dat moet `tensor([[-1., 1., 0.], [-2., 3., -1.]])` zijn. Neem je het gemiddelde van elke rij met `dim=1`, houd je het als kolom met `keepdim=True`, en trek je het van `scores` af?"},
            "other_error": {"en": "Right for these scores. But with {where}, PyTorch stops with an error. Does your code work for any number of rows and columns?",
                            "nl": "Goed voor deze scores. Maar met {where} stopt PyTorch met een fout. Werkt je code voor elk aantal rijen en kolommen?"},
            "other": {"en": "Right for these scores. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you use `scores` itself, with `dim=1`, rather than numbers you typed?",
                      "nl": "Goed voor deze scores. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruik je `scores` zelf, met `dim=1`, en geen getallen die je hebt ingetypt?"},
            "correct": {"en": "Each row now has mean 0: you took 7 from the first row and 6 from the second. The check also tried three rows of two.",
                        "nl": "Elke rij heeft nu gemiddelde 0: van de eerste rij ging 7 af, van de tweede 6. De check probeerde ook drie rijen van twee."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("loss", _L_START, "empty", None),
    ("loss", _l(loss="torch.mean((y_pred - y) ** 2).item()"), "correct", None),
    ("loss", _l(pred="torch.matmul(X, w) + b", loss="((y - y_pred) * (y - y_pred)).mean().item()"), "correct", None),
    ("loss", _l(loss="((y_pred - y) ** 2).sum().item() / 3"), "correct", None),
    ("loss", _l(loss="((y_pred - y) ** 2).mean().item()"), "correct", None),
    ("loss", _l(loss="float(((y - y_pred) ** 2).mean())"), "correct", None),
    ("loss", _l(loss="((y - y_pred) ** 2).mean()"), "wrong", "tensor"),
    ("loss", _l(loss="(y - y_pred) ** 2"), "wrong", "vector"),
    ("loss", _l(loss="((y - y_pred) ** 2).sum().item()"), "wrong", "sum"),
    ("loss", _l(loss="(y - y_pred).mean().item()"), "wrong", "no-square"),
    ("loss", _l(loss="(y_pred - y).mean().item()"), "wrong", "no-square"),
    ("loss", _l(pred="w @ X + b"), "error", "order"),
    ("loss", _l(loss="np.mean((y - y_pred) ** 2)"), "error", "numpy"),
    ("loss", _l(pred="X @ w - 1.0"), "almost", "other-value"),
    ("centered", _C_START, "empty", None),
    ("centered", _c("centered = scores - scores.mean(dim=1, keepdim=True)"), "error", "whole-numbers"),
    ("centered", _c("centered = scores - scores.float().mean(dim=1)"), "error", "no-keepdim"),
    ("centered", _c("scores = scores.float()\ncentered = scores - scores.mean(axis=1, keepdims=True)"), "correct", None),
    ("centered", _c("scores = scores.to(torch.float32)\ncentered = scores - scores.sum(dim=1, keepdim=True) / scores.shape[1]"), "correct", None),
    ("centered", _c("centered = scores - scores.float().mean(dim=0, keepdim=True)"), "wrong", "columns"),
    ("centered", _c("centered = scores - scores.float().mean()"), "wrong", "all"),
    ("centered", _c("centered = scores - torch.tensor([[7.0], [6.0]])"), "almost", "other-value"),
    ("centered", _c("centered = scores - scores.sum(dim=1, keepdim=True) / 3"), "almost", "other-value"),
    ("centered", _C_HEAD.replace("]])", "]], dtype=torch.float32)") + "centered = scores - scores.mean(dim=1, keepdim=True)\ncentered", "correct", None),
]
