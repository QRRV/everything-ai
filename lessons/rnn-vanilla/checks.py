# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rnn-vanilla: bptt, backpropagation through time by hand for a recurrent network with one
number of memory, h_t = tanh(w * h_{t-1} + x_t), checked against PyTorch's autograd."""
import functools
import html
import math
import re

from everything_ai.check import FunctionBox, Mistake

STEPS, NOISE = 30, 0.5


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


@functools.lru_cache(maxsize=4)
def reach(w: float, seed: int = 0) -> tuple:
    """How much input x_t still counts for the last hidden state, |dh_T/dx_t|, relative to the last input, for every
    step back from 0 to STEPS - 1, with PyTorch's autograd. Inputs drawn with seed `seed`, std 0.5."""
    import torch
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(seed)
    xs = (torch.randn(STEPS, generator=g) * NOISE).requires_grad_(True)
    h = torch.tensor(0.0)
    for t in range(STEPS):
        h = torch.tanh(w * h + xs[t])
    h.backward()
    grads = xs.grad.abs()
    return tuple(float(grads[STEPS - 1 - k] / grads[-1]) for k in range(STEPS))


def unroll_picture(labels: dict) -> str:
    """The network unrolled over three steps: the hidden states (hidden) in a row, the inputs (data) below them, the same
    w (param) on every arrow forward, and above, the gradient (grad) going back, times one factor per step."""
    W, H, R = 340, 184, 17
    xs = [34, 124, 214, 304]
    yh, yx, yb = 92, 142, 46
    sub = "₀₁₂₃"
    font = "font-feature-settings:'calt' 0,'liga' 0"
    p = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    for k, cx in enumerate(xs):
        p.append(f'<circle cx="{cx}" cy="{yh}" r="{R}" style="fill:var(--ea-hidden-wash);stroke:var(--ea-hidden);stroke-width:1.5"/>'
                 f'<text x="{cx}" y="{yh + 5}" style="font-size:14px;fill:var(--ea-ink);text-anchor:middle;{font}">h{sub[k]}</text>')
        if k == 0:
            continue
        a, b = xs[k - 1] + R + 3, cx - R - 3
        p.append(f'<path d="M{a} {yh} H{b} M{b - 5} {yh - 4} L{b} {yh} L{b - 5} {yh + 4}" '
                 f'style="fill:none;stroke:var(--ea-ink-3);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round"/>'
                 f'<text x="{(a + b) / 2}" y="{yh - 6}" style="font-size:14px;font-style:italic;fill:var(--ea-param-text);text-anchor:middle">w</text>'
                 f'<rect x="{cx - 16}" y="{yx - 11}" width="32" height="22" rx="5" style="fill:var(--ea-data-wash);stroke:var(--ea-data)"/>'
                 f'<text x="{cx}" y="{yx + 5}" style="font-size:14px;fill:var(--ea-ink);text-anchor:middle;{font}">x{sub[k]}</text>'
                 f'<path d="M{cx} {yx - 13} V{yh + R + 3} M{cx - 4} {yh + R + 8} L{cx} {yh + R + 3} L{cx + 4} {yh + R + 8}" '
                 f'style="fill:none;stroke:var(--ea-ink-3);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round"/>')
        a, b = cx - 6, xs[k - 1] + 6
        p.append(f'<path d="M{a} {yb} H{b} M{b + 5} {yb - 4} L{b} {yb} L{b + 5} {yb + 4}" '
                 f'style="fill:none;stroke:var(--ea-grad);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round"/>'
                 f'<text x="{(a + b) / 2}" y="{yb - 7}" style="font-size:12px;fill:var(--ea-grad-text);text-anchor:middle;{font}">× w(1 − h{sub[k]}²)</text>')
    p.append(f'<text x="{W / 2}" y="{H - 2}" style="font-size:12px;fill:var(--ea-ink-2);text-anchor:middle">{html.escape(labels["note"])}</text>')
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:{W}px">'
            f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["alt"])}</title>{"".join(p)}</svg></figure>')


def reach_picture(labels: dict) -> str:
    """How much an input counts for the last hidden state, against how many steps back it was read, on a log scale,
    for w = 0.5 and w = 0.9. Gradients: the gradient colour, w = 0.5 dashed."""
    from everything_ai import chart
    p = chart.Panel((0, STEPS - 1), (1e-12, 1), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                    ylog=True, xticks=[0, 10, 20, 29])
    ks = list(range(STEPS))
    for w, dashed, (tx, ty) in ((0.9, False, (12, 0.15)), (0.5, True, (16, 1e-9))):
        p.line(ks, [max(v, 1e-12) for v in reach(w)], "grad", 2.2, dashed=dashed)
        p.text(tx, ty, labels["w"].format(w=number(w, 1, labels.get("lang", "en"))), "grad", "start")
    svg = chart.figure([p], alt=labels["alt"], width=330, height=260)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


# --- exercise 1: bptt ----------------------------------------------------------------------------------------------------------

_HEAD = ("import math\n\n\ndef bptt(xs, w):\n"
         "    # forward: h_t = tanh(w * h_{t-1} + x_t), from h_0 = 0; hs keeps every h\n"
         "    hs = [0.0]\n    for x in xs:\n        hs.append(math.tanh(w * hs[-1] + x))\n"
         "    # backward through time: the gradient of the last h for w\n"
         "    grad, dw = 1.0, 0.0\n    for t in range(len(xs), 0, -1):\n")
_LOCAL = "        local = grad * (1 - hs[t] ** 2)\n"
_DW = "        dw = dw + local * hs[t - 1]\n"
_ON = "        grad = local * w\n"
_TAIL = "    return dw\n"
_START = (_HEAD + "        # through tanh at step t; step t's part of the gradient for w; on to h_{t-1}\n        ...\n" + _TAIL)


def _b(local: str = _LOCAL, dw: str = _DW, on: str = _ON) -> str:
    return _HEAD + local + dw + on + _TAIL


def _close(got, want) -> bool:
    try:
        return not isinstance(got, (str, bytes, bool)) and math.isclose(float(got), float(want), rel_tol=1e-5, abs_tol=1e-9)
    except (TypeError, ValueError):
        return False


def _show(v):
    try:
        if isinstance(v, (str, bytes, bool)) or v is None:
            return repr(v)
        x = float(v)
    except (TypeError, ValueError):
        return repr(v)
    if x != x or math.isinf(x):
        return str(x)
    return {"en": f"{x:.4f}", "nl": f"{x:.4f}".replace(".", ",")}


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^\s*\.\.\.\s*(#.*)?$", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_WRONG = {
    "no-carry": _b(on=""),
    "no-tanh": _b(local="        local = grad\n"),
    "h-t": _b(dw="        dw = dw + local * hs[t]\n"),
    "no-w": _b(on="        grad = local\n"),
    "no-sum": _b(dw="        dw = local * hs[t - 1]\n"),
}

EXERCISES = {
    "bptt": FunctionBox(
        name="bptt",
        function="bptt",
        start=_START,
        solution=_b(),
        visible=[([0.6, -0.2, 0.4], 0.8)],
        hidden=[([1.0], 0.7), ([0.2, 0.4, -0.6, 0.1, 0.9, -0.2], -0.8), ([0.3, -0.1, 0.5, 0.2], 1.5), ([0.5, 0.5], 0.0)],
        show=_show,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("no-carry", wrong=_WRONG["no-carry"],
                    en="`{call}` gives {got}, and it should give {want}: every step adds its part as if it were the last. The gradient that reaches step t has gone back through every later step: carry it on, `grad = local * w`, before the next step back.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke stap telt zijn deel op alsof hij de laatste was. De gradient die bij stap t aankomt, is door elke latere stap teruggegaan: geef hem door, `grad = local * w`, voor de volgende stap terug."),
            Mistake("no-tanh", wrong=_WRONG["no-tanh"],
                    en="`{call}` gives {got}, and it should give {want}: the gradient does not go through tanh. Its slope at step t is `1 - hs[t] ** 2`: multiply by it, as `local = grad * (1 - hs[t] ** 2)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de gradient gaat niet door tanh. De helling ervan bij stap t is `1 - hs[t] ** 2`: vermenigvuldig ermee, als `local = grad * (1 - hs[t] ** 2)`."),
            Mistake("h-t", wrong=_WRONG["h-t"],
                    en="`{call}` gives {got}, and it should give {want}: at step t, w multiplies the hidden state that came in, `hs[t - 1]`, not the one that comes out, `hs[t]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: bij stap t vermenigvuldigt w de hidden state die binnenkwam, `hs[t - 1]`, niet die eruit komt, `hs[t]`."),
            Mistake("no-w", wrong=_WRONG["no-w"],
                    en="`{call}` gives {got}, and it should give {want}: on its way back to `hs[t - 1]`, the gradient passes w, which multiplies `hs[t - 1]` in the forward pass. Multiply by it: `grad = local * w`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: op weg terug naar `hs[t - 1]` komt de gradient langs w, die `hs[t - 1]` vermenigvuldigt in de forward pass. Vermenigvuldig ermee: `grad = local * w`."),
            Mistake("no-sum", wrong=_WRONG["no-sum"],
                    en="`{call}` gives {got}, and it should give {want}: what comes back is 0 in every case. The part of step 1 is always 0, because `hs[0]` is 0: does every step add its part to what `dw` already holds, with `dw = dw + ...`?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: er komt in alle gevallen 0 uit. Het deel van stap 1 is altijd 0, omdat `hs[0]` 0 is: telt elke stap zijn deel op bij wat `dw` al heeft, met `dw = dw + ...`?"),
            Mistake("local-first", error=lambda e: isinstance(e, UnboundLocalError) and "'local'" in str(e),
                    en="With `{call}`, your function uses `local` before the line that gives it its value. In each step back, first the line `local = ...`, then the lines that use it.",
                    nl="Met `{call}` gebruikt je functie `local` vóór de regel die hem zijn waarde geeft. Bij elke stap terug eerst de regel `local = ...`, daarna de regels die hem gebruiken."),
            Mistake("rounded", matches=lambda got, text: re.search(r"\bround\s*\(", text) is not None,
                    en="`{call}` gives {got}, and it should give {want}: you round with `round`, and the check compares more digits than it shows. Give back the number as it comes out, without rounding.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je rondt af met `round`, en de check vergelijkt meer cijfers dan hij laat zien. Geef het getal terug zoals het uitkomt, zonder af te ronden."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the three lines of a step back: through tanh, step t's part for w, and on to the step before.",
                      "nl": "Vervang de `...` door de drie regels van een stap terug: door tanh, het deel van stap t voor w, en door naar de stap ervoor."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the inputs and the weight w, and the `def` line of your function has another number of names between its parentheses. Keep `(xs, w)`.",
                          "nl": "De check roept `{call}` aan, met de inputs en het gewicht w, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(xs, w)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. At each step back: does the gradient go through tanh, does step t add its part with the hidden state that came in, and does the gradient go on through w?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Bij elke stap terug: gaat de gradient door tanh, telt stap t zijn deel op met de hidden state die binnenkwam, en gaat de gradient door w verder?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does it work for one step, for a negative w, for w larger than 1, and for w = 0?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt het ook voor één stap, voor een negatieve w, voor w groter dan 1, en voor w = 0?"},
            "correct": {"en": "Your backpropagation through time gives PyTorch's gradient in all {n} cases: one shared w, and a part of its gradient from every step.",
                        "nl": "Je backpropagation through time geeft de gradient van PyTorch in alle {n} gevallen: één gedeelde w, en een deel van zijn gradient uit elke stap."},
        },
    ),
}

# The lesson is done when these are correct. "shared-grad" and "long-text" are the choice questions.
REQUIRED = ["bptt", "shared-grad", "long-text"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("bptt", _START, "empty", None),
    ("bptt", _b(), "correct", None),
    ("bptt", _b(local="        d = 1 - hs[t] ** 2\n        local = grad * d\n", dw="        dw += local * hs[t - 1]\n", on="        grad = w * local\n"), "correct", None),
    ("bptt", _b(dw=_ON, on=_DW), "correct", None),
    ("bptt", _b(local=_DW, dw=_LOCAL), "error", "local-first"),
    ("bptt", _b().replace("    return dw\n", "    return round(dw, 4)\n"), "wrong", "rounded"),
] + [("bptt", code, "wrong", key) for key, code in _WRONG.items()]


def autograd_check(text) -> tuple | None:
    """The learner's bptt against PyTorch's autograd on the worked example's inputs: (theirs, PyTorch's)."""
    import torch
    from everything_ai.check import run_code
    ran = run_code(text) if text else None
    fn = ran.names.get("bptt") if ran is not None and ran.error is None else None
    if not callable(fn):
        return None
    xs, w = [0.5, -0.3, 0.8, 0.1, -0.7], 0.9
    wt = torch.tensor(w, requires_grad=True)
    h = torch.tensor(0.0)
    for x in xs:
        h = torch.tanh(wt * h + x)
    h.backward()
    return float(fn(xs, w)), float(wt.grad)
