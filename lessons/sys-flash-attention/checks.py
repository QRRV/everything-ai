# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-flash-attention: online_attention, attention for one query with a softmax worked out
block by block, with a running maximum, sum and output; and merge, two such running states put together. And the
picture: the score matrix in blocks, the ones causal FlashAttention computes and the ones it skips. The demo times
attention that stores the whole score matrix against PyTorch's fused attention."""
import html
import math
import re
import time

import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai.check import FunctionBox, Mistake


def grouped(n: float, lang: str) -> str:
    return f"{n:,.0f}".replace(",", "." if lang == "nl" else ",")


def size(nbytes: float, lang: str) -> str:
    """A number of bytes in KB, MB or GB (powers of 1,024), without decimals below 10."""
    for unit, k in (("GB", 3), ("MB", 2), ("KB", 1)):
        v = nbytes / 1024 ** k
        if v >= 1:
            text = f"{v:.0f}" if v >= 10 else f"{v:.1f}"
            return f"{text.replace('.', ',') if lang == 'nl' else text} {unit}"
    return f"{nbytes:.0f} B"


def blocks_picture(labels: dict) -> str:
    """The score matrix of 8 by 8 blocks with the causal mask: the blocks on and below the diagonal are computed
    (hidden: worked out, not learned), the ones above skipped; one row of blocks marked, as one query block walks it."""
    nb, cell, left, top = 8, 30, 34, 52
    W, H = left + nb * cell + 10, top + nb * cell + 34
    p = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>',
         f'<text x="{left}" y="38" style="font-size:12px;fill:var(--ea-ink-2)">{html.escape(labels["keys"])}</text>',
         f'<text x="12" y="{top + nb * cell / 2:.0f}" transform="rotate(-90 12 {top + nb * cell / 2:.0f})" '
         f'style="font-size:12px;fill:var(--ea-ink-2);text-anchor:middle">{html.escape(labels["queries"])}</text>']
    row = 4
    for i in range(nb):
        for j in range(nb):
            x, y = left + j * cell, top + i * cell
            if j <= i:
                op = 0.75 if i == row else 0.3
                p.append(f'<rect x="{x + 1}" y="{y + 1}" width="{cell - 2}" height="{cell - 2}" rx="3" '
                         f'style="fill:var(--ea-hidden);fill-opacity:{op}"/>')
            else:
                p.append(f'<rect x="{x + 1.5}" y="{y + 1.5}" width="{cell - 3}" height="{cell - 3}" rx="3" '
                         f'style="fill:none;stroke:var(--ea-line);stroke-dasharray:3 3"/>')
    y = top + row * cell + cell / 2
    x1 = left + (row + 1) * cell + 4
    p.append(f'<path d="M{left + 6} {y} H{x1} M{x1 - 5} {y - 4} L{x1} {y} L{x1 - 5} {y + 4}" '
             f'style="fill:none;stroke:var(--ea-ink);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round"/>')
    p.append(f'<text x="{x1 + 6}" y="{y + 4}" style="font-size:12px;fill:var(--ea-ink)">{html.escape(labels["walk"])}</text>')
    p.append(f'<text x="{left}" y="{H - 8}" style="font-size:12px;fill:var(--ea-ink-2)">{html.escape(labels["note"])}</text>')
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:{W}px">'
            f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["alt"])}</title>{"".join(p)}</svg></figure>')


# --- exercise 1: online_attention -----------------------------------------------------------------------------------------------

_A_HEAD = ("import math\nimport torch\n\n\ndef online_attention(q, K, V, block):\n"
           "    q, K, V = torch.tensor(q), torch.tensor(K), torch.tensor(V)\n"
           "    m = float(\"-inf\")              # the largest score so far\n"
           "    l = 0.0                        # the sum of e^(score - m) so far\n"
           "    o = torch.zeros(V.shape[1])    # the sum of e^(score - m) times the values so far\n"
           "    for start in range(0, K.shape[0], block):\n"
           "        s = K[start:start + block] @ q / math.sqrt(q.shape[0])   # the scores of this block\n"
           "        m_new = max(m, s.max().item())\n")
_A_BODY = ("        scale = math.exp(m - m_new)\n        p = torch.exp(s - m_new)\n"
           "        l = l * scale + p.sum().item()\n        o = o * scale + p @ V[start:start + block]\n")
_A_TAIL = "        m = m_new\n    return o / l\n"
_A_START = _A_HEAD + "        # rescale what you have to the new maximum, and add this block\n        ...\n" + _A_TAIL


def _a(body: str = _A_BODY) -> str:
    return _A_HEAD + body + _A_TAIL


_A_WRONG = {
    "no-rescale": _a("        p = torch.exp(s - m_new)\n        l = l + p.sum().item()\n        o = o + p @ V[start:start + block]\n"),
    "sum-only": _a("        scale = math.exp(m - m_new)\n        p = torch.exp(s - m_new)\n"
                   "        l = l * scale + p.sum().item()\n        o = o + p @ V[start:start + block]\n"),
    "output-only": _a("        scale = math.exp(m - m_new)\n        p = torch.exp(s - m_new)\n"
                      "        l = l + p.sum().item()\n        o = o * scale + p @ V[start:start + block]\n"),
}

# --- exercise 2: merge ---------------------------------------------------------------------------------------------------------

_M_HEAD = "import math\n\n\ndef merge(m1, l1, o1, m2, l2, o2):\n"
_M_BODY = ("    m = max(m1, m2)\n    a, b = math.exp(m1 - m), math.exp(m2 - m)\n"
           "    l = l1 * a + l2 * b\n    o = [o1[i] * a + o2[i] * b for i in range(len(o1))]\n")
_M_TAIL = "    return m, l, o\n"
_M_START = (_M_HEAD + "    # the new maximum; then both sums and both outputs, each rescaled from its own maximum to the new one\n"
            "    ...\n" + _M_TAIL)


def _mg(body: str = _M_BODY) -> str:
    return _M_HEAD + body + _M_TAIL


_M_WRONG = {
    "no-rescale": _mg("    m = max(m1, m2)\n    l = l1 + l2\n    o = [x + y for x, y in zip(o1, o2)]\n"),
    "swapped": _mg("    m = max(m1, m2)\n    a, b = math.exp(m2 - m), math.exp(m1 - m)\n"
                   "    l = l1 * a + l2 * b\n    o = [x * a + y * b for x, y in zip(o1, o2)]\n"),
    "min": _mg("    m = min(m1, m2)\n    a, b = math.exp(m1 - m), math.exp(m2 - m)\n"
               "    l = l1 * a + l2 * b\n    o = [x * a + y * b for x, y in zip(o1, o2)]\n"),
}


def _flat(v):
    """An outcome as a flat list of floats: a tensor, a list, or (m, l, o) or [m, l, o] with o a list; None when it is not numbers."""
    try:
        if isinstance(v, (str, bytes)):
            return None
        if isinstance(v, tuple) or isinstance(v, list) and any(not isinstance(x, (int, float)) for x in v):
            out = []
            for part in v:
                f = _flat(part)
                if f is None:
                    return None
                out += f
            return out
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return [float(v)]
        return [float(x) for x in torch.as_tensor(v, dtype=torch.float64).reshape(-1).tolist()]
    except (TypeError, ValueError, RuntimeError):
        return None


def _close(got, want) -> bool:
    a, b = _flat(got), _flat(want)
    return (a is not None and b is not None and len(a) == len(b)
            and all(math.isclose(x, y, rel_tol=1e-5, abs_tol=1e-6) for x, y in zip(a, b)))


def _num(x: float) -> str:
    return f"{round(x, 4) + 0.0:.4f}".rstrip("0").rstrip(".")


def _show(v):
    """Numbers rounded to 4 decimals: a list for a vector, (m, l, [o]) for a state."""
    def one(x, lang):
        f = _flat(x)
        if f is None:
            return repr(x)
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            t = _num(float(x))
            return t.replace(".", ",") if lang == "nl" else t
        parts = [_num(y) for y in f]
        sep = "; " if lang == "nl" else ", "
        return "[" + sep.join(t.replace(".", ",") if lang == "nl" else t for t in parts) + "]"

    if isinstance(v, (tuple, list)) and len(v) == 3 and _flat(v) is not None and not all(isinstance(x, (int, float)) for x in v):
        a, b = ("(", ")") if isinstance(v, tuple) else ("[", "]")
        return {lang: a + ("; " if lang == "nl" else ", ").join(one(x, lang) for x in v) + b for lang in ("en", "nl")}
    return {lang: one(v, lang) for lang in ("en", "nl")}


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_Q = [1.0, 0.0]
_K = [[2.0, 0.0], [0.0, 1.0], [4.0, 0.0], [1.0, 1.0], [-1.0, 0.0]]
_V = [[1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [0.0, -1.0], [3.0, 1.0]]

EXERCISES = {
    "online-attention": FunctionBox(
        name="online-attention",
        function="online_attention",
        start=_A_START,
        solution=_a(),
        visible=[(_Q, _K, _V, 2)],
        hidden=[(_Q, _K, _V, 1), (_Q, _K, _V, 5), ([0.5, -1.0, 2.0], [[1.0, 0.0, 1.0], [0.0, 2.0, -1.0], [3.0, 1.0, 0.0], [-2.0, 0.0, 2.0]],
                                                    [[1.0], [2.0], [3.0], [4.0]], 3)],
        show=_show,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("no-rescale", wrong=_A_WRONG["no-rescale"],
                    en="`{call}` gives {got}, and it should give {want}: when a block brings a larger maximum, what you added before still counts e to the power of its score minus the old maximum. Multiply `l` and `o` by `math.exp(m - m_new)` before you add the new block.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: als een block een groter maximum meebrengt, telt wat je eerder optelde nog e tot de macht zijn score min het oude maximum. Vermenigvuldig `l` en `o` met `math.exp(m - m_new)` voordat je het nieuwe block optelt."),
            Mistake("sum-only", wrong=_A_WRONG["sum-only"],
                    en="`{call}` gives {got}, and it should give {want}: you rescale the sum `l` to the new maximum, but not the output `o`. Both hold terms e to the power of a score minus the old maximum: multiply `o` by the same factor.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je schaalt de som `l` naar het nieuwe maximum, maar de output `o` niet. Allebei bevatten ze termen e tot de macht een score min het oude maximum: vermenigvuldig `o` met dezelfde factor."),
            Mistake("output-only", wrong=_A_WRONG["output-only"],
                    en="`{call}` gives {got}, and it should give {want}: you rescale the output `o` to the new maximum, but not the sum `l`. Both hold terms e to the power of a score minus the old maximum: multiply `l` by the same factor.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je schaalt de output `o` naar het nieuwe maximum, maar de som `l` niet. Allebei bevatten ze termen e tot de macht een score min het oude maximum: vermenigvuldig `l` met dezelfde factor."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: rescale `l` and `o` from the old maximum to the new one, and add the terms of this block.",
                      "nl": "Vervang de `...`: schaal `l` en `o` van het oude maximum naar het nieuwe, en tel de termen van dit block erbij op."},
            "arguments": {"en": "The check calls `{call}`, with the query, the keys, the values and the block size, and the `def` line of your function has another number of names between its parentheses. Keep `(q, K, V, block)`.",
                          "nl": "De check roept `{call}` aan, met de query, de keys, de values en de grootte van een block, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(q, K, V, block)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. For every block: the factor from the old maximum to the new one for `l` and `o`, then the terms e to the power of score minus the new maximum, for `l` and, times the values, for `o`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Voor elk block: de factor van het oude maximum naar het nieuwe voor `l` en `o`, dan de termen e tot de macht score min het nieuwe maximum, voor `l` en, maal de values, voor `o`."},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does it work for blocks of 1, for one block of everything, and for other sizes?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt het ook voor blocks van 1, voor één block met alles, en voor andere maten?"},
            "correct": {"en": "Your online attention gives the output of softmax attention in all {n} cases, whatever the size of the blocks. Between two blocks, the recipe keeps only m, l and o, never all scores at once.",
                        "nl": "Je online attention geeft de output van softmax-attention in alle {n} gevallen, wat de grootte van de blocks ook is. Tussen twee blocks houdt het recept alleen m, l en o bij, nooit alle scores tegelijk."},
        },
    ),
    "merge": FunctionBox(
        name="merge",
        function="merge",
        start=_M_START,
        solution=_mg(),
        visible=[(2.0, 1.5, [3.0, 1.0], 3.0, 1.0, [2.0, 0.0])],
        hidden=[(5.0, 2.0, [1.0, 1.0, 1.0], 1.0, 4.0, [2.0, -2.0, 0.0]), (0.0, 1.0, [1.0], 0.0, 3.0, [2.0]),
                (-1.0, 2.5, [0.5, 4.0], 2.0, 0.5, [1.0, 1.0])],
        show=_show,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("no-rescale", wrong=_M_WRONG["no-rescale"],
                    en="`{call}` gives {got}, and it should give {want}: you add the sums and the outputs as they are, but each was taken relative to its own maximum. Rescale each to the new one first: times `math.exp(m1 - m)` and `math.exp(m2 - m)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt de sommen en de outputs op zoals ze zijn, maar elk werd genomen ten opzichte van zijn eigen maximum. Schaal ze eerst naar het nieuwe: maal `math.exp(m1 - m)` en `math.exp(m2 - m)`."),
            Mistake("swapped", wrong=_M_WRONG["swapped"],
                    en="`{call}` gives {got}, and it should give {want}: the factors are swapped. The state of part 1 was taken relative to `m1`, so it gets `math.exp(m1 - m)`; part 2 gets `math.exp(m2 - m)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de factoren zijn verwisseld. De state van deel 1 werd genomen ten opzichte van `m1`, dus die krijgt `math.exp(m1 - m)`; deel 2 krijgt `math.exp(m2 - m)`."),
            Mistake("min", wrong=_M_WRONG["min"],
                    en="`{call}` gives {got}, and it should give {want}: the new maximum is the larger of the two, `max(m1, m2)`. With the smaller one, a factor becomes larger than 1, and with large scores e to that power overflows.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het nieuwe maximum is de grootste van de twee, `max(m1, m2)`. Met de kleinste wordt een factor groter dan 1, en bij grote scores loopt e tot die macht over."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: the new maximum, then the two sums and the two outputs, each rescaled to it, added up.",
                      "nl": "Vervang de `...`: het nieuwe maximum, dan de twee sommen en de twee outputs, elk ernaartoe geschaald, opgeteld."},
            "arguments": {"en": "The check calls `{call}`, with the two states, and the `def` line of your function has another number of names between its parentheses. Keep `(m1, l1, o1, m2, l2, o2)`.",
                          "nl": "De check roept `{call}` aan, met de twee states, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(m1, l1, o1, m2, l2, o2)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The new maximum is the larger one; each state is rescaled from its own maximum to it before the two are added.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Het nieuwe maximum is het grootste; elke state wordt van zijn eigen maximum ernaartoe geschaald voordat de twee worden opgeteld."},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does it work when the first maximum is the larger one, when both are equal, and for longer outputs?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt het ook als het eerste maximum het grootste is, als ze gelijk zijn, en voor langere outputs?"},
            "correct": {"en": "Your merge gives the state of all keys together in all {n} cases: two programs can each take half the keys, and their results are put together at the end.",
                        "nl": "Je merge geeft de state van alle keys samen in alle {n} gevallen: twee programma's kunnen elk de helft van de keys nemen, en hun uitkomsten worden aan het eind samengevoegd."},
        },
    ),
}

# The lesson is done when these are correct. "why-faster" is the choice question.
REQUIRED = ["online-attention", "merge", "why-faster"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("online-attention", _A_START, "empty", None),
    ("online-attention", _a(), "correct", None),
    ("online-attention", _a("        p = torch.exp(s - m_new)\n        c = math.exp(m - m_new)\n"
                            "        o = c * o + (p[:, None] * V[start:start + block]).sum(dim=0)\n        l = c * l + float(p.sum())\n"), "correct", None),
] + [("online-attention", code, "wrong", key) for key, code in _A_WRONG.items()] + [
    ("merge", _M_START, "empty", None),
    ("merge", _mg(), "correct", None),
    ("merge", _mg("    m = max(m1, m2)\n    l = l1 * math.exp(m1 - m) + l2 * math.exp(m2 - m)\n"
                  "    o = []\n    for i in range(len(o1)):\n        o.append(o1[i] * math.exp(m1 - m) + o2[i] * math.exp(m2 - m))\n"), "correct", None),
] + [("merge", code, "wrong", key) for key, code in _M_WRONG.items()]


# --- the demo: attention that stores the score matrix, against PyTorch's fused attention -------------------------------------

def demo(n: int = 2048, heads: int = 8, d: int = 64, runs: int = 3) -> dict:
    """Causal attention on one text of n tokens, timed twice on the computer's best device: with the whole score matrix
    worked out, masked and stored, and with torch.nn.functional.scaled_dot_product_attention, which works in blocks."""
    F = torch.nn.functional
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device="cpu").manual_seed(0)
    q, k, v = (torch.randn(1, heads, n, d, generator=g).to(device) for _ in range(3))
    future = torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), 1)

    def stored():
        s = (q @ k.transpose(-1, -2) / math.sqrt(d)).masked_fill(future, float("-inf"))
        return torch.softmax(s, dim=-1) @ v

    def fused():
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    def timed(fn):
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - start) / runs

    with torch.no_grad():
        t_stored, t_fused = timed(stored), timed(fused)
        diff = float((stored() - fused()).abs().max())
    return {"device": device, "n": n, "heads": heads, "stored": t_stored, "fused": t_fused, "diff": diff,
            "matrix": heads * n * n * 4}
