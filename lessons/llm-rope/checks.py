# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson llm-rope: rope, rotary position embeddings for queries and keys. And the demo: with the
learner's rope, the score of one query and one key at different places, the same distance apart."""
import re

import numpy as np

from everything_ai.check import FunctionBox, Mistake, run_code


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def _rope(x, base: float = 10000.0):
    """The course's rope, for the picture and the demo."""
    import torch
    n, d = x.shape[-2], x.shape[-1]
    theta = base ** (-torch.arange(0, d, 2, dtype=torch.float32) / d)
    angles = torch.arange(n, dtype=torch.float32)[:, None] * theta[None, :]
    cos, sin = torch.cos(angles), torch.sin(angles)
    out = torch.zeros_like(x)
    out[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
    out[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos
    return out


def distance_picture(labels: dict) -> str:
    """The score of one query and one key, both 64 ones, with RoPE, against the distance between their places, 0 to 60:
    it is highest at distance 0 and swings lower as the distance grows. A score, no learned value: ink."""
    import torch
    from everything_ai import chart
    d, far = 64, 61
    x = torch.ones(far, d)
    r = _rope(x)
    scores = (r[0:1] @ r.T)[0] / d ** 0.5   # the query at place 0 against the key at place 0 to 60, divided by √64 as in attention
    p = chart.Panel((0, 60), (-3.2, 8.8), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                    xticks=[0, 20, 40, 60], yticks=[0, 4, 8])
    p.line([0, 60], [0, 0], "ink-3", 0.8)
    p.line(list(range(far)), scores.tolist(), "ink", 2.2)
    svg = chart.figure([p], alt=labels["alt"], width=330, height=250)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


# --- the demo: the same distance, other places ---------------------------------------------------------------------------------

PAIRS = [(3, 0), (10, 7), (503, 500), (10, 5), (10, 0)]


def demo(code: str) -> list | str:
    """The score of one random query and one random key of 64 numbers, at the places of PAIRS, with the learner's rope:
    (query place, key place, score). "error" when their rope stops with an error."""
    import torch
    yours = run_code(code)
    if yours.error is not None or "rope" not in yours.names:
        return "error"
    rope = yours.names["rope"]
    torch.manual_seed(0)
    q, k = torch.randn(64), torch.randn(64)
    n = 1 + max(max(pair) for pair in PAIRS)
    try:
        qs = rope(q.expand(n, 64).clone())   # the same query at every place, turned by its place
        ks = rope(k.expand(n, 64).clone())
        return [(m, j, float(qs[m] @ ks[j]) / 8) for m, j in PAIRS]
    except Exception:
        return "error"


# --- exercise 1: rope ---------------------------------------------------------------------------------------------------------

_HEAD = "import torch\n\n\ndef rope(x, base=10000.0):\n"
_BODY = ("    n, d = x.shape[-2], x.shape[-1]\n"
         "    theta = base ** (-torch.arange(0, d, 2) / d)\n"
         "    angles = torch.arange(n)[:, None] * theta[None, :]\n"
         "    cos, sin = torch.cos(angles), torch.sin(angles)\n"
         "    out = torch.zeros_like(x)\n"
         "    out[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin\n"
         "    out[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos\n"
         "    return out\n")
_START = (_HEAD + "    # the places 0 to n - 1 along the second-to-last axis, and d numbers per vector\n    n, d = x.shape[-2], x.shape[-1]\n"
          "    # the angle of every place and every pair, and each pair turned by its angle\n    ...\n")


def _r(body: str = _BODY) -> str:
    return _HEAD + body


class _Named:
    """A tensor of a test case, written in the check's messages the way the lesson writes it."""

    def __init__(self, shown: str):
        self.shown = shown

    def __repr__(self):
        return self.shown

    def __deepcopy__(self, memo):
        import torch
        torch.manual_seed(0)
        return eval(self.shown, {"torch": torch})


def _close(got, want) -> bool:
    try:
        import torch
        return (isinstance(got, torch.Tensor) and tuple(got.shape) == tuple(want.shape)
                and bool(torch.allclose(got.float(), want.float(), atol=1e-4, rtol=1e-4)))
    except Exception:
        return False


def _show(v) -> str:
    try:
        import torch
    except ImportError:
        return repr(v)
    if isinstance(v, torch.Tensor):
        fmt = lambda a: f"{a:.4f}".rstrip("0").rstrip(".") if a == a else "nan"
        if v.dim() == 2 and v.numel() <= 12:
            return "[" + ", ".join("[" + ", ".join(fmt(a) for a in row) + "]" for row in v.tolist()) + "]"
        flat = v.reshape(-1).tolist()
        return f"a tensor of shape {tuple(v.shape)} that starts [{', '.join(fmt(a) for a in flat[:4])}, …]"
    return repr(v)


# a line with only `...` on it: `x[..., 0::2]` uses `...` too, as all the axes in front
_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^\s*\.\.\.\s*(#.*)?$", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

EXERCISES = {
    "rope": FunctionBox(
        name="rope",
        function="rope",
        start=_START,
        solution=_r(),
        visible=[(_Named("torch.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]])"),)],
        hidden=[(_Named("torch.randn(5, 6)"),), (_Named("torch.randn(2, 4, 8)"),), (_Named("torch.randn(2, 3, 5, 4)"),),
                (_Named("torch.randn(3, 2)"),), (_Named("torch.randn(3, 4)"), 100.0)],
        show=_show,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("from-one", wrong=_r(_BODY.replace("torch.arange(n)[:, None]", "torch.arange(1, n + 1)[:, None]")),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your places start at 1. The first token is at place 0, and is not turned at all.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je plekken beginnen bij 1. Het eerste token staat op plek 0, en wordt helemaal niet gedraaid."),
            Mistake("other-way", wrong=_r(_BODY.replace("* cos - x[..., 1::2] * sin", "* cos + x[..., 1::2] * sin").replace("x[..., 0::2] * sin + x[..., 1::2] * cos", "-x[..., 0::2] * sin + x[..., 1::2] * cos")),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you turn every pair the other way, clockwise. Turn it as in [Rotating vectors and complex numbers](/course/open/math-complex-rotation?lang=en): (a cos − b sin, a sin + b cos).",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je draait elk paar de andere kant op, met de klok mee. Draai het zoals in [Vectoren draaien en complexe getallen](/course/open/math-complex-rotation?lang=nl): (a cos − b sin, a sin + b cos)."),
            Mistake("halves", wrong=_r(_BODY.replace("    out = torch.zeros_like(x)\n    out[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin\n    out[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos\n",
                                                    "    a, b = x[..., : d // 2], x[..., d // 2:]\n    out = torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)\n")),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you pair the first half of a vector with the second half. That is another layout that some code uses; this lesson pairs neighbors, entries 0 and 1, 2 and 3, as in [Rotating vectors and complex numbers](/course/open/math-complex-rotation?lang=en).",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je koppelt de eerste helft van een vector aan de tweede helft. Dat is een andere indeling die sommige code gebruikt; deze les koppelt buren, getal 0 en 1, 2 en 3, zoals in [Vectoren draaien en complexe getallen](/course/open/math-complex-rotation?lang=nl)."),
            Mistake("no-two", wrong=_r(_BODY.replace("theta = base ** (-torch.arange(0, d, 2) / d)", "theta = base ** (-torch.arange(d // 2) / d)")),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your pairs turn too fast. Pair j turns at `base ** (-2 * j / d)`: the exponent counts the entries, 0, 2, 4, …, not the pairs.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je paren draaien te snel. Paar j draait met `base ** (-2 * j / d)`: de exponent telt de getallen, 0, 2, 4, …, niet de paren."),
            Mistake("same-angle", wrong=_r(_BODY.replace("angles = torch.arange(n)[:, None] * theta[None, :]", "angles = torch.arange(n)[:, None] * torch.ones(d // 2)[None, :]")),
                    en="`{call}` gives `{got}`, and it should give `{want}`: every pair turns by the same angle, the place itself. Each pair has its own speed, `theta`: the angle of pair j at place t is t times theta of j.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elk paar draait over dezelfde hoek, de plek zelf. Elk paar heeft een eigen snelheid, `theta`: de hoek van paar j op plek t is t maal theta van j."),
            Mistake("back-halves", wrong=_r(_BODY.replace("    out = torch.zeros_like(x)\n    out[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin\n    out[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos\n",
                                                         "    a, b = x[..., 0::2], x[..., 1::2]\n    out = torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)\n")),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you turn the right pairs, but put the results back as two halves, first all the first entries, then all the second. Put each back where it came from: the first entry of each pair at 0, 2, 4, …, the second at 1, 3, 5, ….",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je draait de goede paren, maar zet de uitkomsten terug als twee helften, eerst alle eerste getallen, dan alle tweede. Zet elk terug waar het vandaan kwam: het eerste getal van elk paar op 0, 2, 4, …, het tweede op 1, 3, 5, …."),
            Mistake("overwrite", wrong=_r(_BODY.replace("out[..., 1::2] = x[..., 0::2] * sin", "out[..., 1::2] = out[..., 0::2] * sin")),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the second entries of your pairs are worked out from first entries that are already turned. Take both entries of a pair from `x` before you write anything back: `a * sin + b * cos` needs the `a` from before the turn.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de tweede getallen van je paren reken je uit met eerste getallen die al gedraaid zijn. Neem beide getallen van een paar uit `x` voordat je iets terugschrijft: `a * sin + b * cos` heeft de `a` van vóór het draaien nodig."),
            Mistake("fixed-base", wrong=_r(_BODY.replace("theta = base **", "theta = 10000 **")),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your speeds always use 10000, whatever `base` the call gives. Use `base` in the speeds.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je snelheden gebruiken altijd 10000, welke `base` de aanroep ook meegeeft. Gebruik `base` in de snelheden."),
            Mistake("one-number", error=lambda e: "only one element tensors" in str(e),
                    en="With `{call}`, a function that takes one number, such as `math.cos`, gets a whole tensor. `torch.cos` and `torch.sin` work on every number of a tensor at once, as `np.cos` and `np.sin` do on an array.",
                    nl="Met `{call}` krijgt een functie die één getal neemt, zoals `math.cos`, een hele tensor. `torch.cos` en `torch.sin` werken op elk getal van een tensor tegelijk, zoals `np.cos` en `np.sin` op een array."),
            Mistake("two-axes", syntax=r"\[\s*(?::\s*,\s*)+[01]?\s*::\s*2\s*\]",
                    en="Your code takes the pairs with a `:` for each axis in front, as on the line `{line}`. That only fits a tensor with exactly that many axes, and the check calls `rope` with two, three and four axes. Write `x[..., 0::2]` and `x[..., 1::2]`: `...` stands for all the axes in front, however many there are.",
                    nl="Je code pakt de paren met een `:` voor elke axis ervoor, zoals op de regel `{line}`. Dat past alleen bij een tensor met precies zoveel axes, en de check roept `rope` aan met twee, drie en vier axes. Schrijf `x[..., 0::2]` en `x[..., 1::2]`: `...` staat voor alle axes ervoor, hoeveel het er ook zijn."),
            Mistake("sizes", error=lambda e: isinstance(e, RuntimeError) and "must match the size" in str(e),
                    en="With `{call}`, the sizes of your angles and of `x` do not fit together. Do your angles have one row per place, with the places along the second-to-last axis, `x.shape[-2]`, and one column per pair? In a mini-batch of shape (B, n, d), the first axis counts the texts, not the places.",
                    nl="Met `{call}` passen de maten van je hoeken en van `x` niet bij elkaar. Hebben je hoeken één rij per plek, met de plekken langs de op een na laatste axis, `x.shape[-2]`, en één kolom per paar? In een mini-batch met shape (B, n, d) telt de eerste axis de teksten, niet de plekken."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with your code: the angles of all places and pairs, and a tensor with every pair turned.",
                      "nl": "Vervang de `...` door je code: de hoeken van alle plekken en paren, en een tensor met elk paar gedraaid."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, and the `def` line of your function does not fit that call. Keep `(x, base=10000.0)`: `x`, and `base` with 10000.0 when the call gives no other.",
                          "nl": "De check roept `{call}` aan, en de `def`-regel van je functie past niet bij die aanroep. Houd `(x, base=10000.0)`: `x`, en `base` met 10000.0 als de aanroep geen andere meegeeft."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does pair j of the vector at place t turn by t times `base ** (-2 * j / d)`, counterclockwise?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Draait paar j van de vector op plek t over t maal `base ** (-2 * j / d)`, tegen de klok in?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for any number of places and pairs, and for a mini-batch with more axes in front?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elk aantal plekken en paren, en voor een mini-batch met meer axes ervoor?"},
            "correct": {"en": "Your rope turns every pair of every vector by its place times its own speed, for a single text and for a mini-batch of heads alike.",
                        "nl": "Je rope draait elk paar van elke vector over zijn plek maal zijn eigen snelheid, voor één tekst en voor een mini-batch met heads op dezelfde manier."},
        },
    ),
}

# The lesson is done when these are correct. "where" is the choice question, "not-the-values" the open question.
REQUIRED = ["rope", "where", "not-the-values"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("rope", _START, "empty", None),
    ("rope", _r(), "correct", None),
    ("rope", _r("    n, d = x.shape[-2], x.shape[-1]\n    theta = 1.0 / base ** (torch.arange(0, d, 2) / d)\n"
                "    t = torch.arange(n).unsqueeze(1)\n    a = t * theta\n"
                "    x1, x2 = x[..., 0::2], x[..., 1::2]\n"
                "    y = torch.stack([x1 * torch.cos(a) - x2 * torch.sin(a), x1 * torch.sin(a) + x2 * torch.cos(a)], dim=-1)\n"
                "    return y.flatten(-2)\n"), "correct", None),
    ("rope", _r(_BODY.replace("torch.arange(n)[:, None]", "torch.arange(1, n + 1)[:, None]")), "wrong", "from-one"),
    ("rope", _r(_BODY.replace("* cos - x[..., 1::2] * sin", "* cos + x[..., 1::2] * sin").replace("x[..., 0::2] * sin + x[..., 1::2] * cos", "-x[..., 0::2] * sin + x[..., 1::2] * cos")), "wrong", "other-way"),
    ("rope", _r(_BODY.replace("    out = torch.zeros_like(x)\n    out[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin\n    out[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos\n",
                              "    a, b = x[..., : d // 2], x[..., d // 2:]\n    out = torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)\n")), "wrong", "halves"),
    ("rope", _r(_BODY.replace("theta = base ** (-torch.arange(0, d, 2) / d)", "theta = base ** (-torch.arange(d // 2) / d)")), "wrong", "no-two"),
    ("rope", _r(_BODY.replace("angles = torch.arange(n)[:, None] * theta[None, :]", "angles = torch.arange(n)[:, None] * torch.ones(d // 2)[None, :]")), "wrong", "same-angle"),
    ("rope", _r(_BODY.replace("n, d = x.shape[-2], x.shape[-1]", "n, d = x.shape[0], x.shape[-1]")), "error", "sizes"),
    ("rope", _r(_BODY.replace("    out = torch.zeros_like(x)\n    out[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin\n    out[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos\n",
                              "    a, b = x[..., 0::2], x[..., 1::2]\n    out = torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-2).flatten(-2)\n")), "wrong", "back-halves"),
    ("rope", _r(_BODY.replace("    out = torch.zeros_like(x)\n", "    out = x.clone()\n").replace("out[..., 1::2] = x[..., 0::2] * sin", "out[..., 1::2] = out[..., 0::2] * sin")), "wrong", "overwrite"),
    ("rope", _r(_BODY.replace("theta = base **", "theta = 10000.0 **")), "wrong", "fixed-base"),
    ("rope", "import math\n" + _r(_BODY.replace("torch.cos(angles), torch.sin(angles)", "math.cos(angles), math.sin(angles)")), "error", "one-number"),
    ("rope", _r(_BODY.replace("[..., ", "[:, ")), "error", "two-axes"),
    ("rope", _HEAD + "    return x\n", "wrong", None),
]
