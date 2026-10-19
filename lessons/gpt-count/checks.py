# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson gpt-count: gpt_params, the number of parameters of the GPT of gpt-assemble from its sizes, part
by part; and train_flops, the FLOPs of training on one token: about 2 per parameter and a part for attention that grows
with the length of the text, forward, and three times that with the backward pass. The demo compares the learner's
count with the parameters PyTorch counts in their own GPT, and shows how the share of attention grows with the text."""
import math
import re
from decimal import Decimal

from everything_ai import projects
from everything_ai.check import FunctionBox, Mistake, Timeout, run_code, with_time_limit

# The demo makes the learner's GPT of gpt-assemble, which imports their Block and CausalSelfAttention; where one is
# missing or broken, the course's version stands in (template 1.3).
projects.builds_on(("llm", "attention", "attn-transformer-block", "causal-self-attention"),
                   ("llm", "block", "attn-transformer-block", "block"), ("llm", "gpt", "gpt-assemble", "gpt"))

SIZES = [(65, 64, 4, 4, 64), (65, 384, 6, 6, 256), (50257, 768, 12, 12, 1024)]   # vocab, d, h, n_layers, block


def parts(vocab: int, d: int, n_layers: int, block: int) -> dict:
    """The parameters of the GPT of gpt-assemble, part by part."""
    one_block = (4 * d * d + d) + 2 * (2 * d) + (8 * d * d + 5 * d)
    return {"tok": vocab * d, "pos": block * d, "blocks": n_layers * one_block, "norm": 2 * d, "head": d * vocab + vocab}


def grouped(n: int, lang: str) -> str:
    return f"{n:,}".replace(",", "." if lang == "nl" else ",")


def number(x: float, digits: int, lang: str) -> str:
    text_ = f"{x:,.{digits}f}"
    if lang == "nl":
        text_ = text_.replace(",", "_").replace(".", ",").replace("_", ".")
    return text_


def parts_picture(labels: dict) -> str:
    """Where the parameters of two GPTs sit, the GPT of gpt-assemble and the sizes of the smallest GPT-2: per part one
    bar of its share. Every part is parameters, so every bar is in the param role; the parts differ by label and place,
    not by color. The last layer normalization, under 0.1%, is left out."""
    import html
    rows = [(labels["small"], parts(65, 64, 4, 64)), (labels["gpt2"], parts(50257, 768, 12, 1024))]
    names = [("tok", labels["tok"]), ("pos", labels["pos"]), ("blocks", labels["blocks"]), ("head", labels["head"])]
    W, BX, BW, BH = 360, 116, 200, 14
    p = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    y = 34
    for label, part in rows:
        total = sum(part.values())
        p.append(f'<text x="0" y="{y + 12}" style="font-size:13px;fill:var(--ea-ink-2)">{html.escape(label.format(n=grouped(total, labels["lang"])))}</text>')
        y += 22
        for key, name in names:
            share = part[key] / total
            w = max(BW * share, 1.5)
            pct = "&lt; 1%" if share < 0.01 else f"{round(100 * share)}%"
            p.append(f'<text x="0" y="{y + 11}" style="font-size:12px;fill:var(--ea-ink-2)">{html.escape(name)}</text>'
                     f'<rect x="{BX}" y="{y}" width="{w:.1f}" height="{BH}" style="fill:var(--ea-param)"/>'
                     f'<text x="{BX + w + 6:.1f}" y="{y + 11}" style="font-size:12px;fill:var(--ea-ink-2)">{pct}</text>')
            y += BH + 6
        y += 14
    H = y - 10
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:{W}px">'
            f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["alt"])}</title>{"".join(p)}</svg></figure>')


# The lesson is done when these are correct. "forward-flops" is the choice question in the lesson.
REQUIRED = ["forward-flops", "gpt-params", "train-flops"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_CALL = "`{call}` gives `{got}`, and it should give `{want}`"
_CALL_NL = "`{call}` geeft `{got}`, en dat moet `{want}` zijn"


def _close(got, want) -> bool:
    if isinstance(got, str):   # a number written as text is not a number: float("6393216") would pass
        return False
    try:
        return math.isclose(float(got), float(want), rel_tol=1e-9, abs_tol=1e-9)
    except (TypeError, ValueError):
        return False


def _shown(v) -> str:
    """A number for the messages: whole above 1,000, else rounded to 4 decimals, or 4 digits after the zeros of a small
    number; never in e-notation."""
    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
        if abs(v) >= 1000:
            if isinstance(v, float) and abs(v) >= 1e15:   # past the digits a float holds: 6 digits, then zeros
                return format(Decimal(f"{v:.6g}"), "f")
            return str(round(v))
        digits = 4 if abs(v) >= 0.001 or v == 0 else 3 - math.floor(math.log10(abs(v)))
        return f"{v:.{digits}f}".rstrip("0").rstrip(".")
    return repr(v)


# --- exercise 2: gpt_params ---------------------------------------------------------------------------------------------------

_P_HEAD = "def gpt_params(vocab, d, n_layers, block):\n"
_P_EMB = "    embeddings = vocab * d + block * d\n"
_P_BLOCK = "    one_block = (4 * d * d + d) + 2 * (2 * d) + (d * 4 * d + 4 * d + 4 * d * d + d)\n"
_P_END = "    end = 2 * d + d * vocab + vocab\n"
_P_RETURN = "    return embeddings + n_layers * one_block + end\n"
_P_START = (_P_HEAD + "    # the table of tokens and the table of places\n    embeddings = ...\n"
            "    # one block: attention (query, key, value and out, with the bias terms of out),\n"
            "    # two layer normalizations, and the MLP from d to 4 * d and back, with its bias terms\n    one_block = ...\n"
            "    # the last layer normalization, and the output head with its bias terms\n    end = ...\n" + _P_RETURN)


def _p(emb: str = _P_EMB, block: str = _P_BLOCK, end: str = _P_END) -> str:
    return _P_HEAD + emb + block + end + _P_RETURN


# --- exercise 3: train_flops --------------------------------------------------------------------------------------------------

_F_HEAD = "def train_flops(n_params, n_layers, n, d):\n"
_F_FORWARD = "    forward = 2 * n_params + 2 * n_layers * n * d\n"
_F_RETURN = "    return 3 * forward\n"
_F_START = (_F_HEAD + "    # one token forward: 2 per parameter, and attention in every block, for a text of n tokens\n    forward = ...\n"
            "    # training on it: the forward pass, and a backward pass that costs about twice as much\n    return ...\n")


def _f(forward: str = _F_FORWARD, ret: str = _F_RETURN) -> str:
    return _F_HEAD + forward + ret


EXERCISES = {
    "gpt-params": FunctionBox(
        name="gpt-params",
        function="gpt_params",
        start=_P_START,
        solution=_p(),
        visible=[(65, 64, 4, 64)],
        hidden=[(10, 8, 2, 6), (5, 4, 1, 8), (100, 16, 3, 32), (50257, 768, 12, 1024)],
        python_values=True,
        first=[_DOTS],
        mistakes=[
            Mistake("no-pos", wrong=_p(emb="    embeddings = vocab * d\n"),
                    en=_CALL + ": the table of places is missing: `block` rows of `d` numbers, `block * d`.",
                    nl=_CALL_NL + ": de tabel van plekken ontbreekt: `block` rijen van `d` getallen, `block * d`."),
            Mistake("tied", wrong=_p(end="    end = 2 * d\n"),
                    en=_CALL + ": the output head is missing. GPT-2 shares it with the table of tokens, but your GPT has an output head of its own: `d * vocab` weights and `vocab` bias terms.",
                    nl=_CALL_NL + ": de output head ontbreekt. GPT-2 deelt hem met de tabel van tokens, maar je GPT heeft een eigen output head: `d * vocab` weights en `vocab` bias terms."),
            Mistake("three", wrong=_p(block="    one_block = (3 * d * d) + 2 * (2 * d) + (d * 4 * d + 4 * d + 4 * d * d + d)\n"),
                    en=_CALL + ": your attention has three matrices of d by d, for the queries, keys and values. It has a fourth, the output projection, with `d` bias terms: `4 * d * d + d`.",
                    nl=_CALL_NL + ": je attention heeft drie matrices van d bij d, voor de queries, keys en values. Hij heeft een vierde, de output-projectie, met `d` bias terms: `4 * d * d + d`."),
            Mistake("narrow", wrong=_p(block="    one_block = (4 * d * d + d) + 2 * (2 * d) + (d * d + d + d * d + d)\n"),
                    en=_CALL + ": your MLP goes from d to d. It widens to `4 * d` and back: `d * 4 * d + 4 * d` for the first layer, `4 * d * d + d` for the second.",
                    nl=_CALL_NL + ": je MLP gaat van d naar d. Het wordt `4 * d` breed en weer smal: `d * 4 * d + 4 * d` voor de eerste layer, `4 * d * d + d` voor de tweede."),
            Mistake("no-norms", wrong=_p(block="    one_block = (4 * d * d + d) + (d * 4 * d + 4 * d + 4 * d * d + d)\n", end="    end = d * vocab + vocab\n"),
                    en=_CALL + ": the layer normalizations are missing. Each has a scale and a bias term for every one of the d numbers: `2 * d`, two in every block and one at the end.",
                    nl=_CALL_NL + ": de layer normalizations ontbreken. Elk heeft een schaal en een bias term voor elk van de d getallen: `2 * d`, twee in elk block en één aan het eind."),
            Mistake("qkv-bias", wrong=_p(block=_P_BLOCK.replace("(4 * d * d + d)", "(4 * d * d + 4 * d)")),
                    en=_CALL + ": every block has `3 * d` too many. GPT-2 has bias terms for query, key and value; in your GPT of the last lesson they have none. In attention only the output projection has `d`, `4 * d * d + d`, and the MLP has `4 * d` and `d`.",
                    nl=_CALL_NL + ": elk block heeft er `3 * d` te veel. GPT-2 heeft bias terms voor query, key en value; in je GPT van de vorige les hebben die er geen. In attention heeft alleen de output-projectie er `d`, `4 * d * d + d`, en het MLP heeft er `4 * d` en `d`."),
            Mistake("no-bias", wrong=_p(block="    one_block = (4 * d * d) + 2 * (2 * d) + (d * 4 * d + 4 * d * d)\n", end="    end = 2 * d + d * vocab\n"),
                    en=_CALL + ": the bias terms are missing: `d` for the output projection, `4 * d` and `d` for the two layers of the MLP, and `vocab` for the output head. Only query, key and value have none.",
                    nl=_CALL_NL + ": de bias terms ontbreken: `d` voor de output-projectie, `4 * d` en `d` voor de twee layers van het MLP, en `vocab` voor de output head. Alleen query, key en value hebben er geen."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the three `...`: the two tables; the parameters of one block, part by part; and the last layer normalization with the output head.",
                      "nl": "Vervang de drie `...`: de twee tabellen; de parameters van één block, deel voor deel; en de laatste layer normalization met de output head."},
            "wrong": {"en": _CALL + ". The tables, `vocab * d` and `block * d`; per block, attention, two layer normalizations and the MLP, each with their bias terms; and at the end a layer normalization and the head: steps 1 to 3 of the recipe?",
                      "nl": _CALL_NL + ". De tabellen, `vocab * d` en `block * d`; per block attention, twee layer normalizations en het MLP, elk met hun bias terms; en aan het eind een layer normalization en de head: stap 1 tot en met 3 van het recept?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function use `vocab`, `d`, `n_layers` and `block`, and no fixed numbers?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Gebruikt je functie `vocab`, `d`, `n_layers` en `block`, en geen vaste getallen?"},
            "correct": {"en": "Your count is exact: 211,777 for the GPT of the last lesson, as PyTorch counts it, and every block has 12d² + 10d parameters.",
                        "nl": "Je telling klopt precies: 211.777 voor de GPT van de vorige les, zoals PyTorch hem telt, en elk block heeft 12d² + 10d parameters."},
        },
    ),
    "train-flops": FunctionBox(
        name="train-flops",
        function="train_flops",
        start=_F_START,
        solution=_f(),
        visible=[(1_000_000, 4, 128, 128)],
        hidden=[(10_788_929, 6, 256, 384), (124_412_160, 12, 1024, 768), (163_059_793, 12, 32768, 768), (5, 2, 3, 4), (0, 1, 1, 1)],
        compare=_close,
        show=_shown,
        first=[_DOTS],
        mistakes=[
            Mistake("forward-only", wrong=_f(ret="    return forward\n"),
                    en=_CALL + ": you count the forward pass only. Training on a token also goes backward, which costs about twice as much: three forward passes in all.",
                    nl=_CALL_NL + ": je telt alleen de forward pass. Trainen op een token gaat ook terug, en dat kost ongeveer twee keer zoveel: drie forward passes in totaal."),
            Mistake("backward-only", wrong=_f(ret="    return 2 * forward\n"),
                    en=_CALL + ": you count the backward pass, about twice the forward pass. Training does the forward pass as well: three forward passes in all.",
                    nl=_CALL_NL + ": je telt de backward pass, ongeveer twee keer de forward pass. Trainen doet de forward pass ook: drie forward passes in totaal."),
            Mistake("no-attention", wrong=_f(forward="    forward = 2 * n_params\n"),
                    en=_CALL + ": you leave out attention. In every block, a token compares its query with the keys up to it and mixes their values: about `2 * n_layers * n * d` FLOPs for a text of `n` tokens, on top of `2 * n_params`.",
                    nl=_CALL_NL + ": je laat attention weg. In elk block vergelijkt een token zijn query met de keys tot en met zichzelf en mengt het hun values: ongeveer `2 * n_layers * n * d` FLOPs voor een tekst van `n` tokens, bovenop `2 * n_params`."),
            Mistake("one-block", wrong=_f(forward="    forward = 2 * n_params + 2 * n * d\n"),
                    en=_CALL + ": your attention is in one block only. Every block has attention of its own: times `n_layers`.",
                    nl=_CALL_NL + ": je attention zit maar in één block. Elk block heeft zijn eigen attention: keer `n_layers`."),
            Mistake("all-tokens", wrong=_f(forward="    forward = 2 * n_params + 4 * n_layers * n * d\n"),
                    en=_CALL + ": you count attention as if every token looked at all `n` tokens of the text, as your attention computes them before the mask. The formula counts the work a token needs: through the causal mask it looks only at the tokens up to it, on average about half of them: `2 * n_layers * n * d`.",
                    nl=_CALL_NL + ": je telt attention alsof elk token naar alle `n` tokens van de tekst kijkt, zoals je attention ze vóór de mask berekent. De formule telt het werk dat een token nodig heeft: door de causal mask kijkt het alleen naar de tokens tot en met zichzelf, gemiddeld ongeveer de helft: `2 * n_layers * n * d`."),
            Mistake("attention-once", wrong=_f(forward="    forward = 2 * n_params\n", ret="    return 3 * forward + 2 * n_layers * n * d\n"),
                    en=_CALL + ": you count attention once, as if it only went forward. The multiplications of attention go backward too: training is three times the whole forward pass, attention included.",
                    nl=_CALL_NL + ": je telt attention één keer, alsof die alleen vooruit gaat. Ook de vermenigvuldigingen van attention gaan terug: trainen is drie keer de hele forward pass, attention erbij."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the FLOPs of one token forward, the parameters and attention, and the FLOPs of training on it.",
                      "nl": "Vervang de twee `...`: de FLOPs van één token vooruit, de parameters en attention, en de FLOPs van het trainen erop."},
            "wrong": {"en": _CALL + ". Forward 2 FLOPs per parameter, and attention in every block; training about three times that: steps 4 and 5 of the recipe?",
                      "nl": _CALL_NL + ". Vooruit 2 FLOPs per parameter, en attention in elk block; trainen ongeveer drie keer zoveel: stap 4 en 5 van het recept?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function use `n_params`, `n_layers`, `n` and `d`, and no fixed numbers?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Gebruikt je functie `n_params`, `n_layers`, `n` en `d`, en geen vaste getallen?"},
            "correct": {"en": "Your count works for any GPT and any length of text: 6,393,216 FLOPs to train a million parameters in 4 blocks of d = 128 on one token of a text of 128 tokens.",
                        "nl": "Je telling werkt voor elke GPT en elke lengte van tekst: 6.393.216 FLOPs om een miljoen parameters in 4 blocks met d = 128 te trainen op één token van een tekst van 128 tokens."},
        },
    ),
}


# --- the demo: the learner's count against their own GPT, and two estimates -------------------------------------------------------

LENGTHS = [1024, 8192, 32768]   # texts of these lengths, for a GPT with GPT-2's sizes and a table of places as long as the text


def demo(params_code: str, flops_code: str):
    """For the demo, with the learner's code of exercises 2 and 3: for every size in SIZES, their count and the count
    of PyTorch in their own GPT of gpt-assemble (None for GPT-2's size, which takes some 650 MB of memory); and their
    FLOPs of training on one token, for the GPT of the book at its windows of 256 and for GPT-2's other sizes at windows
    of LENGTHS, its table of places as long as the window, each with the share of attention. "gpt" when their GPT stops with an error, "slow" when it takes too long."""
    yours = run_code(params_code + "\n" + flops_code)
    if yours.error is not None:
        return "gpt"
    gpt_params, train_flops = yours.names["gpt_params"], yours.names["train_flops"]
    run = run_code("from llm.gpt import GPT")
    if run.error is not None:
        return "gpt"
    GPT = run.names["GPT"]
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]

    def work():
        rows = []
        for vocab, d, h, n_layers, block in SIZES:
            counted = None
            if d < 768:
                model = GPT(vocab, d, h, n_layers, block)
                counted = int(sum(p.numel() for p in model.parameters()))
            rows.append({"sizes": (vocab, d, n_layers, block), "yours": int(gpt_params(vocab, d, n_layers, block)), "pytorch": counted})
        return rows

    def flops(n_params, n_layers, n, d) -> dict:
        total = int(train_flops(n_params, n_layers, n, d))
        return {"n": n, "flops": total, "attention": 1 - int(train_flops(n_params, n_layers, 0, d)) / total}

    try:
        rows = with_time_limit(work, 60)
        _, d, _, n_layers, block = SIZES[1]
        book = flops(rows[1]["yours"], n_layers, block, d)
        vocab, d, _, n_layers, _ = SIZES[2]
        gpt2 = [flops(int(gpt_params(vocab, d, n_layers, n)), n_layers, n, d) for n in LENGTHS]
    except Timeout:
        return "slow"
    except Exception:
        return "gpt"
    return {"rows": rows, "book": book, "gpt2": gpt2, "stand_ins": stand_ins}


# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("gpt-params", _P_START, "empty", None),
    ("gpt-params", _p(), "correct", None),
    ("gpt-params", _p(block="    one_block = 12 * d * d + 10 * d\n"), "correct", None),
    ("gpt-params", _p(emb="    embeddings = vocab * d\n"), "wrong", "no-pos"),
    ("gpt-params", _p(block=_P_BLOCK.replace("(4 * d * d + d)", "(4 * d * d + 4 * d)")), "wrong", "qkv-bias"),
    ("gpt-params", _p(block=_P_BLOCK.replace("4 * d * d + d)\n", "4 * d * d + 4 * d)\n")), "wrong", "qkv-bias"),
    ("gpt-params", _p(end="    end = 2 * d\n"), "wrong", "tied"),
    ("gpt-params", _p(block="    one_block = (3 * d * d) + 2 * (2 * d) + (d * 4 * d + 4 * d + 4 * d * d + d)\n"), "wrong", "three"),
    ("gpt-params", _p(block="    one_block = (4 * d * d + d) + 2 * (2 * d) + (d * d + d + d * d + d)\n"), "wrong", "narrow"),
    ("gpt-params", _p(block="    one_block = (4 * d * d + d) + (d * 4 * d + 4 * d + 4 * d * d + d)\n", end="    end = d * vocab + vocab\n"), "wrong", "no-norms"),
    ("gpt-params", _p(block="    one_block = (4 * d * d) + 2 * (2 * d) + (d * 4 * d + 4 * d * d)\n", end="    end = 2 * d + d * vocab\n"), "wrong", "no-bias"),
    ("gpt-params", _p(end="    end = 2 * d + d * vocab + 65\n"), "almost", None),
    ("train-flops", _F_START, "empty", None),
    ("train-flops", _f(), "correct", None),
    ("train-flops", _f(forward="    forward = 2 * (n_params + n_layers * n * d)\n", ret="    return forward + 2 * forward\n"), "correct", None),
    ("train-flops", _f(ret="    return forward\n"), "wrong", "forward-only"),
    ("train-flops", _f(ret="    return 2 * forward\n"), "wrong", "backward-only"),
    ("train-flops", _f(forward="    forward = 2 * n_params\n"), "wrong", "no-attention"),
    ("train-flops", _f(forward="    forward = 2 * n_params + 2 * n * d\n"), "wrong", "one-block"),
    ("train-flops", _f(forward="    forward = 2 * n_params + 4 * n_layers * n * d\n"), "wrong", "all-tokens"),
    ("train-flops", _f(forward="    forward = 2 * n_params\n", ret="    return 3 * forward + 2 * n_layers * n * d\n"), "wrong", "attention-once"),
    ("train-flops", _f(forward="    forward = 6 * n_params + 2 * n_layers * n * d\n", ret="    return forward\n"), "wrong", "attention-once"),
    ("train-flops", _f(ret="    return str(3 * forward)\n"), "wrong", None),
    ("train-flops", _f(forward="    forward = 2 * n_params + 2 * 4 * n * d\n"), "almost", None),
]
