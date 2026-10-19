# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson llm-gqa: gqa, causal attention in which groups of query heads share one key and value head."""
import re

from everything_ai.check import FunctionBox, Mistake


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def heads_picture(labels: dict) -> str:
    """Eight query heads, and below them, with a stronger outline, the key and value heads they read: 8 with multi-head attention,
    2 with grouped-query attention, 1 with multi-query attention, with a line from every query head to the head it
    uses. Heads hold no learned values themselves: ink."""
    from everything_ai import chart
    p = chart.Panel((0, 8), (0.4, 10.4), title=labels["title"], xlabel="", ylabel="", xticks=[], yticks=[])
    qx = [0.5 + i for i in range(8)]
    for name, g, top in ((labels["mha"], 8, 9.0), (labels["gqa"], 2, 5.7), (labels["mqa"], 1, 2.4)):
        p.text(0.2, top + 0.95, name, "ink", "start")
        y = top - 1.4
        group = 8 // g
        kx = [sum(qx[j * group:(j + 1) * group]) / group for j in range(g)]
        for i, x in enumerate(qx):
            p.line([x, kx[i // group]], [top - 0.3, y + 0.3], "ink-3", 1.0)
            p.rect(x - 0.3, top - 0.3, x + 0.3, top + 0.3, "ink-3")
        for x in kx:
            p.rect(x - 0.3, y - 0.3, x + 0.3, y + 0.3, "ink")
    svg = chart.figure([p], alt=labels["alt"], width=330, height=300)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


# --- exercise 1: gqa ----------------------------------------------------------------------------------------------------------

_HEAD = ("import math\nimport torch\n\n\ndef gqa(q, k, v, n_heads, n_kv_heads):\n"
         "    # q: (B, n, n_heads * d_head); k and v: (B, n, n_kv_heads * d_head)\n"
         "    n, d_head = q.shape[1], int(q.shape[2] / n_heads)\n    future = torch.tril(torch.ones(n, n)) == 0\n"
         "    outputs = []\n    for i in range(n_heads):\n        a, b = i * d_head, (i + 1) * d_head\n")
_KV = "        j = int(i / (n_heads / n_kv_heads))\n        c, e = j * d_head, (j + 1) * d_head\n"
_SCORES = "        S = q[:, :, a:b] @ k[:, :, c:e].transpose(1, 2) / math.sqrt(d_head)\n"
_MASK = "        S = S.masked_fill(future, float(\"-inf\"))\n"
_WEIGHTS = "        outputs.append(torch.softmax(S, dim=2) @ v[:, :, c:e])\n"
_TAIL = "    return torch.cat(outputs, dim=2)\n"
_START = (_HEAD + "        # the key and value head of the group of query head i, and its columns\n        ...\n"
          "        # head i, as in causal attention: the scaled scores, the mask, the softmax, times the values\n        ...\n" + _TAIL)


def _g(kv: str = _KV, scores: str = _SCORES, mask: str = _MASK, weights: str = _WEIGHTS) -> str:
    return _HEAD + kv + scores + mask + weights + _TAIL


class _Named:
    """A tensor of a test case, written in the check's messages the way the lesson writes it; made with seed `seed`."""

    def __init__(self, shown: str, seed: int):
        self.shown, self.seed = shown, seed

    def __repr__(self):
        return self.shown

    def __deepcopy__(self, memo):
        import torch
        torch.manual_seed(self.seed)
        return eval(self.shown, {"torch": torch})


def _case(q: str, kv: str, h: int, g: int, seed: int) -> tuple:
    return (_Named(q, seed), _Named(kv, seed + 1), _Named(kv, seed + 2), h, g)


def _close(got, want) -> bool:
    try:
        import torch
        return (isinstance(got, torch.Tensor) and tuple(got.shape) == tuple(want.shape)
                and bool(torch.allclose(got.float(), want.float(), atol=1e-4, rtol=1e-4)))
    except Exception:
        return False


def _row(values: list) -> str:
    """The numbers of one token: all of them up to 8, else the first 3 and the last 3, so that the first and the last
    group of heads both show."""
    fmt = lambda a: f"{round(a, 4) + 0.0:.4f}".rstrip("0").rstrip(".") if a == a else "nan"   # + 0.0: never -0
    shown = [fmt(a) for a in values] if len(values) <= 8 else [fmt(a) for a in values[:3]] + ["…"] + [fmt(a) for a in values[-3:]]
    return "`[" + ", ".join(shown) + "]`"


def _show(v):
    try:
        import torch
    except ImportError:
        return f"`{v!r}`"
    if isinstance(v, list) and v and all(isinstance(t, torch.Tensor) for t in v):
        return {"en": f"a list of {len(v)} tensors", "nl": f"een lijst van {len(v)} tensors"}
    if isinstance(v, torch.Tensor):
        shape = tuple(v.shape)
        if v.dim() < 2 or v.numel() == 0:
            return {"en": f"a tensor of shape {shape}", "nl": f"een tensor met shape {shape}"}
        rows = v.reshape(-1, v.shape[-1])
        # the first token sees only itself, the last one every token: a mistake shows in one of them
        first, last = _row(rows[0].tolist()), _row(rows[-1].tolist())
        return {"en": f"a tensor of shape {shape}, first token {first}, last token {last}",
                "nl": f"een tensor met shape {shape}, eerste token {first}, laatste token {last}"}
    return f"`{v!r}`"


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^\s*\.\.\.\s*(#.*)?$", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

EXERCISES = {
    "gqa": FunctionBox(
        name="gqa",
        function="gqa",
        start=_START,
        solution=_g(),
        visible=[_case("torch.randn(1, 3, 8)", "torch.randn(1, 3, 4)", 4, 2, 0)],
        hidden=[_case("torch.randn(2, 5, 12)", "torch.randn(2, 5, 3)", 4, 1, 3), _case("torch.randn(2, 4, 12)", "torch.randn(2, 4, 12)", 3, 3, 6),
                _case("torch.randn(1, 6, 16)", "torch.randn(1, 6, 8)", 8, 4, 9)],
        show=_show,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("empty-keys", error=lambda e: isinstance(e, RuntimeError) and re.search(r"but got: \[\d+, 0\]", str(e)) is not None,
                    en="With `{call}`, your function stops with an error: the keys of a head have no columns. `k` is narrower than `q`: it only has the columns of `n_kv_heads` heads, so the columns of query head i, or of a head number that is too large, lie past its end. Do the columns of `k` and `v` come from the key and value head of the group of head i?",
                    nl="Met `{call}` stopt je functie met een fout: de keys van een head hebben geen kolommen. `k` is smaller dan `q`: hij heeft alleen de kolommen van `n_kv_heads` heads, dus de kolommen van query head i, of van een te groot headnummer, liggen voorbij zijn eind. Komen de kolommen van `k` en `v` van de key- en value-head van de groep van head i?"),
            Mistake("float-columns", error=lambda e: isinstance(e, TypeError) and "slice indices must be integers" in str(e),
                    en="With `{call}`, your function stops with an error: the edges of a slice must be whole numbers, and `/` always gives a number with a point, such as `1.0`, even when it divides exactly. Make them whole with `int(...)`, as in the worked example.",
                    nl="Met `{call}` stopt je functie met een fout: de grenzen van een slice moeten gehele getallen zijn, en `/` geeft altijd een getal met een komma, zoals `1.0`, ook als de deling precies uitkomt. Maak ze geheel met `int(...)`, zoals in het uitgewerkte voorbeeld."),
            Mistake("modulo", wrong=_g(kv=_KV.replace("j = int(i / (n_heads / n_kv_heads))", "j = i % n_kv_heads")),
                    en="`{call}` gives {got}, and it should give {want}: your query heads take turns, 0, 1, 0, 1, …. A group is query heads next to each other: with 4 query heads and 2 key heads, heads 0 and 1 share key head 0, heads 2 and 3 key head 1. That is `int(i / (n_heads / n_kv_heads))`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je query heads wisselen elkaar af, 0, 1, 0, 1, …. Een groep is query heads naast elkaar: met 4 query heads en 2 key heads delen heads 0 en 1 key head 0, en heads 2 en 3 key head 1. Dat is `int(i / (n_heads / n_kv_heads))`."),
            Mistake("no-mask", wrong=_g(mask=""),
                    en="`{call}` gives {got}, and it should give {want}: your heads look at later tokens too. Mask them with `future`, as in causal attention.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je heads kijken ook naar latere tokens. Maskeer die met `future`, zoals in causale attention."),
            Mistake("no-scale", wrong=_g(scores=_SCORES.replace(" / math.sqrt(d_head)", "")),
                    en="`{call}` gives {got}, and it should give {want}: your scores are not scaled. Divide them by the square root of the width of a head: `/ math.sqrt(d_head)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je scores zijn niet geschaald. Deel ze door de wortel uit de breedte van een head: `/ math.sqrt(d_head)`."),
            Mistake("own-values", wrong=_g(weights="        outputs.append(torch.softmax(S, dim=2) @ v[:, :, a:b])\n"),
                    en="`{call}` gives {got}, and it should give {want}: your weights take the values of the columns of the query head. The values come from the shared head of the group, like the keys: `v[:, :, c:e]`, or whatever you call its columns.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je weights nemen de values van de kolommen van de query head. De values komen van de gedeelde head van de groep, net als de keys: `v[:, :, c:e]`, of hoe je zijn kolommen ook noemt."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: the key and value head of query head i, and the attention of head i with it.",
                      "nl": "Vervang de twee `...`: de key- en value-head van query head i, en de attention van head i ermee."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the queries, keys and values, the number of query heads and the number of key and value heads, and the `def` line of your function has another number of names between its parentheses. Keep `(q, k, v, n_heads, n_kv_heads)`.",
                          "nl": "De check roept `{call}` aan, met de queries, keys en values, het aantal query heads en het aantal key- en value-heads, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(q, k, v, n_heads, n_kv_heads)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does query head i use the key and value head of its group, the columns of `k` and `v` of that head, and causal attention as before?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Gebruikt query head i de key- en value-head van zijn groep, de kolommen van `k` en `v` van die head, en causale attention zoals eerder?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for any number of heads, from one shared key head to one per query head?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elk aantal heads, van één gedeelde key head tot één per query head?"},
            "correct": {"en": "Your gqa works from multi-query attention, one key and value head for all, to multi-head attention, one for every query head, and every group of heads in between.",
                        "nl": "Je gqa werkt van multi-query attention, één key- en value-head voor allemaal, tot multi-head attention, één per query head, en elke groepering daartussen."},
        },
    ),
}

# The lesson is done when these are correct. "share" is the choice question, "why-keys" the open question.
REQUIRED = ["gqa", "share", "why-keys"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("gqa", _START, "empty", None),
    ("gqa", _g(), "correct", None),
    ("gqa", _g(kv="        group = int(n_heads / n_kv_heads)\n        c, e = int(i / group) * d_head, (int(i / group) + 1) * d_head\n"), "correct", None),
    ("gqa", _g(kv=_KV.replace("j = int(i / (n_heads / n_kv_heads))", "j = i % n_kv_heads")), "wrong", "modulo"),
    ("gqa", _g(mask=""), "wrong", "no-mask"),
    ("gqa", _g(scores=_SCORES.replace(" / math.sqrt(d_head)", "")), "wrong", "no-scale"),
    ("gqa", _g(weights="        outputs.append(torch.softmax(S, dim=2) @ v[:, :, a:b])\n"), "wrong", "own-values"),
    ("gqa", _g(scores=_SCORES.replace("k[:, :, c:e]", "k[:, :, a:b]"), weights=_WEIGHTS.replace("c:e", "a:b")), "error", "empty-keys"),
    ("gqa", _g(kv=_KV.replace("int(i / (n_heads / n_kv_heads))", "int(i / n_kv_heads)")), "error", "empty-keys"),
    ("gqa", _g(kv=_KV.replace("int(i / (n_heads / n_kv_heads))", "i / (n_heads / n_kv_heads)")), "error", "float-columns"),
    ("gqa", _g(kv="        c, e = 0, d_head\n"), "wrong", None),
]
