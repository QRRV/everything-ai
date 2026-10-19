# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson lm-peek-real-llm: next_probs, the next-token distribution from a model's logits, and
top_tokens, its most likely tokens as pieces of text. Both are checked on small made-up tensors, so they work
without GPT-2. GPT-2 small itself, 548 MB, the lesson downloads once from OpenAI's repository on Hugging Face, when
the learner asks for it (everything_ai.hub): the worked example and the demo need it, the exercises do not."""
import errno
import functools
import html
import re
import urllib.error

import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai import gpt2_tokenizer, hub, models
from everything_ai.check import FunctionBox, Mistake, run_code

gpt2_tokenizer.offline()   # before any tiktoken.get_encoding("gpt2"), also in the lesson's own cells
import tiktoken  # noqa: E402  (after offline(): tiktoken reads its cache folder when it loads a file)

ENC = tiktoken.get_encoding("gpt2")
DOWNLOADS = {"GPT-2": models.GPT2}   # what the lesson downloads, for the line under its header
VOCAB = 50257

# What GPT-2 small gives, measured with the course's code (everything_ai.models.gpt2) and pinned file: the 10 most
# likely next tokens after each text, with their probabilities, rounded. tests/test_gpt2.py checks them again
# whenever GPT-2 is downloaded.
REFERENCE = {
    "The cat sat on the": [(" floor", 0.0764), (" bed", 0.0653), (" couch", 0.0541), (" ground", 0.0521), (" edge", 0.0478),
                           (" bench", 0.0316), (" table", 0.0315), (" sofa", 0.0288), (" other", 0.0199), (" back", 0.019)],
    "I sat on the": [(" couch", 0.088), (" floor", 0.0874), (" bench", 0.0817), (" edge", 0.0692), (" sofa", 0.0365),
                     (" porch", 0.0253), (" ground", 0.0245), (" back", 0.015), (" other", 0.0149), (" bed", 0.0146)],
    "The capital of France is": [(" the", 0.0846), (" now", 0.0479), (" a", 0.0462), (" France", 0.0324), (" Paris", 0.0322),
                                 (" in", 0.0266), (" also", 0.0264), (" not", 0.0238), (" home", 0.0233), (" still", 0.0155)],
    "Once upon a time": [(",", 0.4269), (" the", 0.0646), (" I", 0.0406), (" he", 0.0369), (" there", 0.0294),
                         (" in", 0.0276), (" it", 0.0234), (" when", 0.0229), (" of", 0.0227), (" they", 0.0175)],
}


def piece(text: str) -> str:
    """A token's piece of text as the course shows it: a space as ␣ and a new line as ↵, so that you see them."""
    return text.replace(" ", "␣").replace("\n", "↵")


def share(p: float, lang: str) -> str:
    """A probability with 3 decimals, as in the lesson's prose: 0.085, or 0,085 in Dutch."""
    text = f"{p:.3f}"
    return text.replace(".", ",") if lang == "nl" else text


def count(n: int, lang: str) -> str:
    """A whole number for the lesson's prose: 1,024, or 1.024 in Dutch."""
    text = f"{n:,}"
    return text.replace(",", ".") if lang == "nl" else text


def _tail(text: str, n: int = 36) -> str:
    """The end of a long text: what the model's next token depends on most."""
    return text if len(text) <= n else "…" + text[-(n - 1):]


def _short(label: str, n: int = 9) -> str:
    return label if len(label) <= n else label[:n - 1] + "…"


def token_bars(text: str, pairs: list, labels: dict, lang: str) -> str:
    """The most likely next tokens after `text`: a bar per token, as long as its probability, with the probability at
    its end. The bars are what the model puts out, so they have the colour of a prediction; the most likely one is
    full. Drawn 330 wide, for a phone. `labels` has "title" and "alt"."""
    W, LW, BW, ROW, TOP = 330, 86, 176, 24, 50
    top = max(p for _, p in pairs) or 1
    parts = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>',
             f'<text x="0" y="38" style="font-size:13.5px;font-style:italic;fill:var(--ea-ink-2)">{html.escape(_tail(text))} …</text>']
    for k, (token, p) in enumerate(pairs):
        y = TOP + k * ROW
        w = max(BW * p / top, 2)
        parts.append(f'<text x="{LW - 8}" y="{y + 14}" style="font-family:var(--ea-font-code);font-size:13px;fill:var(--ea-ink);'
                     f'text-anchor:end">{html.escape(_short(piece(token)))}</text>'
                     f'<rect x="{LW}" y="{y + 3}" width="{w:.1f}" height="15" rx="3" style="fill:var(--ea-pred);opacity:{1 if k == 0 else 0.45}"/>'
                     f'<text x="{LW + w + 6:.1f}" y="{y + 15}" style="font-size:12.5px;fill:var(--ea-pred-text)">{share(p, lang)}</text>')
    H = TOP + len(pairs) * ROW + 2
    alt = labels["title"] + ". " + labels["alt"]
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:min(100%,{W}px)">'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(alt)}" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(alt)}</title>' + "".join(parts) + "</svg></figure>")


# --- GPT-2 in the lesson ---------------------------------------------------------------------------------------------

def downloaded() -> bool:
    return hub.pinned(*models.GPT2) is not None


@functools.lru_cache(maxsize=1)
def _gpt2():
    return models.gpt2()


def gpt2_or_stop(button, texts: dict):
    """GPT-2 small for the lesson's cells. When it is not downloaded yet, the cell stops with `texts["ask"]` and the
    button; a click downloads it, with the progress in the cell. While it downloads, the lesson's other cells wait.
    `texts` has "ask", "busy" ({done} and {total} in MB), "no_net", "not_pinned", "no_room" and "failed" ({why}): why
    a download did not work; and "here"."""
    import marimo as mo
    failed = None
    if not downloaded() and button.value:
        total = round(hub.PINNED[models.GPT2]["size"] / 1e6)
        shown = [-1]

        def progress(done, size):
            mb = int(done / 1e6)
            if mb // 10 != shown[0] // 10:   # every 10 MB
                shown[0] = mb
                mo.output.replace(mo.md(texts["busy"].format(done=mb, total=total)))
        try:
            hub.download(*models.GPT2, progress=progress)
        except hub.NotPinned:
            failed = texts["not_pinned"]
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            failed = texts["no_net"]
        except OSError as e:
            failed = texts["no_room"] if e.errno == errno.ENOSPC else texts["failed"].format(why=html.escape(str(e)))
    if not downloaded():
        parts = [mo.md(failed)] if failed else []
        mo.stop(True, mo.vstack(parts + [mo.md(texts["ask"]), button]))
    model = _gpt2()
    mo.output.replace(mo.md(texts["here"]))
    return model


def course_next(model, text: str):
    """The course's next-token distribution of `text`, from `model`."""
    with torch.no_grad():
        logits = model(torch.tensor(ENC.encode(text)))
    return torch.softmax(logits[-1], dim=0)


# The lesson is done when these are correct. "context" is the choice question in the lesson.
REQUIRED = ["next-probs", "top-tokens", "context"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code left in the box. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts again with the course's lines.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_CALL = "`{call}` gives `{got}`, and it should give `{want}`"
_CALL_NL = "`{call}` geeft `{got}`, en dat moet `{want}` zijn"


class _Tensor:
    """A tensor as a test case: written with its name in the check's messages; every call gets its own copy."""

    def __init__(self, name: str, values):
        self.name, self.values = name, values

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return torch.tensor(self.values, dtype=torch.float32)


def _probs(at: dict) -> list:
    p = [0.0] * VOCAB
    for i, v in at.items():
        p[i] = v
    return p


def _rounded(value):
    """What the check compares for next_probs: the numbers, rounded to 3 decimals, as a list; anything else as it is."""
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    elif hasattr(value, "tolist"):   # a NumPy array
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return type(value)(_rounded(v) for v in value)
    if isinstance(value, float):
        return round(value, 3)
    return value


def _num(x):
    """A tensor or NumPy value of one number as a plain number, and a float rounded to 3 decimals."""
    if isinstance(x, torch.Tensor) and x.numel() == 1 or getattr(x, "ndim", None) == 0 and hasattr(x, "item"):
        x = x.item()
    return round(x, 3) if isinstance(x, float) else x


def _pairs(value):
    """What the check compares for top_tokens: each pair as (piece of text, probability rounded to 3 decimals); any
    other number rounded too, so that a message never shows float noise."""
    if isinstance(value, dict):
        return {k: _num(v) for k, v in value.items()}
    if not isinstance(value, (list, tuple)):
        return _num(value)
    out = []
    for pair in value:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            text, p = (_num(x) for x in pair)
            out.append((text, round(float(p), 3) if isinstance(p, (int, float)) else p))
        else:
            out.append(_num(pair))
    return out


# Two positions, a vocabulary of 3 tokens; and 4 positions, a vocabulary of 5.
_LOGITS = [[2.0, 0.0, 1.0], [0.0, 3.0, 1.0]]
_LOGITS_2 = [[0.5, 1.0, -1.0, 2.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0], [3.0, -2.0, 0.0, 1.0, 2.0], [-1.0, 0.0, 2.5, 1.0, 0.5]]
_P_HEAD = "import torch\n\n\ndef next_probs(logits):\n"
_P_RIGHT = "    return torch.softmax(logits[-1], dim=0)\n"
_P_START = _P_HEAD + "    # the softmax of the last row: the token after the whole text\n    return ...\n"
_P_WRONG = {
    "first-row": _P_HEAD + "    return torch.softmax(logits[0], dim=0)\n",
    "columns": _P_HEAD + "    return torch.softmax(logits, dim=0)[-1]\n",
    "no-softmax": _P_HEAD + "    return logits[-1]\n",
    "every-row": _P_HEAD + "    return torch.softmax(logits, dim=1)\n",
}

# Probabilities over GPT-2's 50,257 tokens: 262 is " the", 257 " a", 1169 "the", 13 ".", 11 ",", 290 " and".
_PROBS = _probs({262: 0.5, 257: 0.3, 1169: 0.15, 13: 0.05})
_PROBS_2 = _probs({11: 0.25, 290: 0.35, 13: 0.1, 262: 0.2, 257: 0.1})
_T_HEAD = "import torch\nimport tiktoken\n\nenc = tiktoken.get_encoding(\"gpt2\")\n\n\ndef top_tokens(probs, k):\n"
_T_TOP = "    top = torch.topk(probs, k)\n"
_T_LOOP = ("    pairs = []\n    for j in range(k):\n        i = top.indices[j].item()\n"
           "        pairs.append((enc.decode([i]), top.values[j].item()))\n    return pairs\n")
_T_START = (_T_HEAD + "    # the k largest probabilities, and their token IDs\n    top = ...\n    pairs = []\n    for j in range(k):\n"
            "        i = top.indices[j].item()\n        # the piece of text of token ID i, with its probability\n        pairs.append(...)\n"
            "    return pairs\n")
_T_WRONG = {
    "ids": _T_HEAD + _T_TOP + _T_LOOP.replace("(enc.decode([i]), ", "(i, "),
    "whole-ids": _T_HEAD + _T_TOP + _T_LOOP.replace("(enc.decode([i]), ", "(enc.decode(top.indices.tolist()), "),
}

EXERCISES = {
    "next-probs": FunctionBox(
        name="next-probs",
        function="next_probs",
        start=_P_START,
        solution=_P_HEAD + _P_RIGHT,
        visible=[(_Tensor("logits", _LOGITS),)],
        hidden=[(_Tensor("logits", _LOGITS_2),)],
        collect=_rounded,
        first=[_DOTS],
        mistakes=[
            Mistake("first-row", wrong=_P_WRONG["first-row"],
                    en=_CALL + ": that is the distribution after the first token only. The token after the whole text comes from the last row: `logits[-1]`.",
                    nl=_CALL_NL + ": dat is de verdeling na alleen het eerste token. Het token na de hele tekst komt uit de laatste rij: `logits[-1]`."),
            Mistake("columns", wrong=_P_WRONG["columns"],
                    en=_CALL + ": your softmax runs down each column, over the positions, so your numbers do not add up to 1. Take the last row first, then its softmax: `torch.softmax(logits[-1], dim=0)`.",
                    nl=_CALL_NL + ": je softmax loopt langs elke kolom, over de posities, dus samen zijn je getallen geen 1. Neem eerst de laatste rij, dan de softmax ervan: `torch.softmax(logits[-1], dim=0)`."),
            Mistake("no-softmax", wrong=_P_WRONG["no-softmax"],
                    en=_CALL + ": those are the logits of the last row, not yet probabilities. Softmax turns them into numbers from 0 to 1 that add up to 1.",
                    nl=_CALL_NL + ": dat zijn de logits van de laatste rij, nog geen kansen. Softmax maakt er getallen van 0 tot 1 van, die samen 1 zijn."),
            Mistake("every-row", wrong=_P_WRONG["every-row"],
                    en=_CALL + ": a distribution for every position. Only the last row is about the token after the whole text: `logits[-1]`.",
                    nl=_CALL_NL + ": een verdeling voor elke positie. Alleen de laatste rij gaat over het token na de hele tekst: `logits[-1]`."),
            Mistake("no-dim", error=lambda e: isinstance(e, TypeError) and "softmax" in str(e) and "dim" in str(e),
                    en="`torch.softmax` needs to know along which dimension it works: `dim=0` for a row on its own, as in `torch.softmax(logits[-1], dim=0)`.",
                    nl="`torch.softmax` moet weten langs welke dimensie het werkt: `dim=0` voor een rij op zichzelf, zoals in `torch.softmax(logits[-1], dim=0)`."),
            Mistake("dim-range", error=lambda e: isinstance(e, IndexError) and "Dimension out of range" in str(e),
                    en="You name a dimension the tensor does not have. `logits` has two, 0 for the positions and 1 for the tokens; one row on its own, such as `logits[-1]`, has only dimension 0: `dim=0`.",
                    nl="Je noemt een dimensie die de tensor niet heeft. `logits` heeft er twee, 0 voor de posities en 1 voor de tokens; één rij op zichzelf, zoals `logits[-1]`, heeft alleen dimensie 0: `dim=0`."),
            Mistake("too-many-indices", error=lambda e: isinstance(e, IndexError) and "too many indices" in str(e),
                    en="You index with more places than the tensor has dimensions. `logits` has two: the positions and the tokens. Its last row is `logits[-1]`.",
                    nl="Je indexeert met meer plaatsen dan de tensor dimensies heeft. `logits` heeft er twee: de posities en de tokens. De laatste rij is `logits[-1]`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: the softmax of the last row of `logits`.",
                      "nl": "Vervang de `...`: de softmax van de laatste rij van `logits`."},
            "arguments": {"en": "The check calls `{call}`, and the `def` line of your function has another number of names between its parentheses. Keep `(logits)`.",
                          "nl": "De check roept `{call}` aan, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(logits)`."},
            "wrong": {"en": _CALL + ". The last row of `logits` holds the logits of the token after the whole text; softmax turns them into probabilities.",
                      "nl": _CALL_NL + ". De laatste rij van `logits` heeft de logits van het token na de hele tekst; softmax maakt er kansen van."},
            "almost": {"en": "Right for these logits. But with logits of 4 positions and 5 tokens, " + _CALL[0].lower() + _CALL[1:] + ". Does your function use the `logits` it gets, whatever their shape?",
                       "nl": "Goed voor deze logits. Maar met logits van 4 posities en 5 tokens: " + _CALL_NL + ". Gebruikt je functie de `logits` die ze krijgt, welke shape die ook hebben?"},
            "correct": {"en": "The softmax of the last row: a probability for every token of the vocabulary, for the token after the whole text.",
                        "nl": "De softmax van de laatste rij: een kans voor elk token van de vocabulary, voor het token na de hele tekst."},
        },
    ),
    "top-tokens": FunctionBox(
        name="top-tokens",
        function="top_tokens",
        start=_T_START,
        solution=_T_HEAD + _T_TOP + _T_LOOP,
        visible=[(_Tensor("probs", _PROBS), 2)],
        hidden=[(_Tensor("probs", _PROBS_2), 3)],
        collect=_pairs,
        first=[_DOTS],
        mistakes=[
            Mistake("ids", wrong=_T_WRONG["ids"],
                    en=_CALL + ": token IDs, not pieces of text. `enc.decode([i])` gives the piece of text of token ID `i`.",
                    nl=_CALL_NL + ": token IDs, geen stukken tekst. `enc.decode([i])` geeft het stuk tekst van token ID `i`."),
            Mistake("whole-ids", wrong=_T_WRONG["whole-ids"],
                    en=_CALL + ": every pair gets the text of all k tokens together. Decode one token ID at a time: `enc.decode([i])`.",
                    nl=_CALL_NL + ": elk paar krijgt de tekst van alle k tokens samen. Decodeer één token ID tegelijk: `enc.decode([i])`."),
            Mistake("decode-int", error=lambda e: "not an instance of 'Sequence'" in str(e),
                    en="`enc.decode` takes a list of token IDs, also for one ID: `enc.decode([i])`, with square brackets.",
                    nl="`enc.decode` krijgt een lijst van token IDs, ook voor één ID: `enc.decode([i])`, met vierkante haken."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: `torch.topk(probs, k)`, and the pair of the piece of text of `i` and its probability.",
                      "nl": "Vervang de twee `...`: `torch.topk(probs, k)`, en het paar van het stuk tekst van `i` en zijn kans."},
            "arguments": {"en": "The check calls `{call}`, and the `def` line of your function has another number of names between its parentheses. Keep `(probs, k)`.",
                          "nl": "De check roept `{call}` aan, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(probs, k)`."},
            "wrong": {"en": _CALL + ". `probs` has a probability for every token ID: " + "the largest at 262, the piece \" the\", then 257, \" a\". Each pair is the piece of text and its probability.",
                      "nl": _CALL_NL + ". `probs` heeft een kans voor elk token ID: de grootste bij 262, het stuk \" the\", dan 257, \" a\". Elk paar is het stuk tekst en zijn kans."},
            "almost": {"en": "Right for these probabilities. But with other probabilities and `k` = 3, " + _CALL[0].lower() + _CALL[1:] + ". Does your function use the `probs` and the `k` it gets?",
                       "nl": "Goed voor deze kansen. Maar met andere kansen en `k` = 3: " + _CALL_NL + ". Gebruikt je functie de `probs` en de `k` die ze krijgt?"},
            "correct": {"en": "The k most likely tokens, as pieces of text with their probabilities: what the picture of the idea shows for GPT-2.",
                        "nl": "De k waarschijnlijkste tokens, als stukken tekst met hun kansen: wat de figuur van het idee laat zien voor GPT-2."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("next-probs", _P_START, "empty", None),
    ("next-probs", _P_HEAD + _P_RIGHT, "correct", None),
    ("next-probs", _P_HEAD + "    return torch.softmax(logits, dim=1)[-1]\n", "correct", None),
    ("next-probs", _P_HEAD + "    z = logits[-1]\n    e = torch.exp(z - z.max())\n    return e / e.sum()\n", "correct", None),
    ("next-probs", _P_HEAD + "    return torch.softmax(logits[-1])\n", "error", "no-dim"),
    ("next-probs", _P_HEAD + "    return torch.softmax(logits[-1], dim=1)\n", "error", "dim-range"),
    ("next-probs", _P_HEAD + "    return torch.softmax(logits[:, -1, :], dim=-1)\n", "error", "too-many-indices"),
    ("next-probs", _P_HEAD + "    return torch.softmax(logits[1], dim=0)\n", "almost", None),
] + [("next-probs", code, "wrong", key) for key, code in _P_WRONG.items()] + [
    ("top-tokens", _T_START, "empty", None),
    ("top-tokens", _T_HEAD + _T_TOP + _T_LOOP, "correct", None),
    ("top-tokens", _T_HEAD + _T_TOP + "    pairs = []\n    for j in range(k):\n        pairs.append((enc.decode([int(top.indices[j])]), float(top.values[j])))\n    return pairs\n", "correct", None),
    ("top-tokens", _T_HEAD + _T_TOP + _T_LOOP.replace("enc.decode([i])", "enc.decode(i)"), "error", "decode-int"),
    ("top-tokens", _T_HEAD + "    top = torch.topk(probs, 2)\n" + _T_LOOP.replace("range(k)", "range(2)"), "almost", None),
] + [("top-tokens", code, "wrong", key) for key, code in _T_WRONG.items()]


# --- the demo: your own text through GPT-2 ---------------------------------------------------------------------------

def _function(text, name):
    ran = run_code(text) if text else None
    fn = ran.names.get(name) if ran is not None and ran.error is None else None
    return fn if callable(fn) else None


CONTEXT = 1024   # the most tokens GPT-2 reads: of a longer text, the demo gives it the last ones


def demo(model, text: str, probs_code, top_code, k: int = 10) -> dict:
    """The k most likely tokens after `text`, with the learner's next_probs and top_tokens once they are right, else
    the course's; whose they were; and how many tokens at the start GPT-2 did not read ("cut")."""
    mine_p, mine_t = _function(probs_code, "next_probs"), _function(top_code, "top_tokens")
    ids = ENC.encode(text, disallowed_special=())   # "<|endoftext|>" in a text is text here, not GPT-2's special token
    with torch.no_grad():
        logits = model(torch.tensor(ids[-CONTEXT:]))
    try:
        probs = torch.as_tensor(mine_p(logits)) if mine_p else torch.softmax(logits[-1], dim=0)
        top = _pairs(mine_t(probs, k)) if mine_t else None
    except Exception:
        mine_p = mine_t = None
        probs, top = torch.softmax(logits[-1], dim=0), None
    if top is None:
        values = torch.topk(probs, k)
        top = [(ENC.decode([i]), round(p, 3)) for p, i in zip(values.values.tolist(), values.indices.tolist())]
    return {"pieces": [ENC.decode([i]) for i in ids], "top": top, "yours": bool(mine_p and mine_t),
            "cut": max(len(ids) - CONTEXT, 0)}
