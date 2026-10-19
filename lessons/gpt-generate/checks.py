# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson gpt-generate: gpt_probs, the probabilities of the next character from the learner's GPT, with
a temperature; and gpt_generate, the sampling loop of lm-sampling with those probabilities in place of a row of the
bigram table. Both are saved in the learner's project. The demo lets their trained GPT of gpt-train-chars write, or the
course's GPT when they have not trained one, next to their bigram model of lm-sampling."""
import html
import json
import re
from functools import cache

import numpy as np
import torch

from everything_ai import paths, projects
from everything_ai.check import Mistake, ScenarioBox, Timeout, run_code, with_time_limit

# The demo uses the learner's GPT of gpt-assemble, with their Block and attention, and their generate of lm-sampling
# for the bigram model; gpt_generate imports this lesson's own gpt_probs. Where one is missing or broken, the course's
# version stands in (template 1.3).
projects.builds_on(("llm", "attention", "attn-transformer-block", "causal-self-attention"),
                   ("llm", "block", "attn-transformer-block", "block"), ("llm", "gpt", "gpt-assemble", "gpt"),
                   ("llm", "sampling", "lm-sampling", "generate"),
                   ("llm", "gpt_probs", "gpt-generate", "gpt-probs"))

DATA = "data/tiny-shakespeare.txt"   # about 1.1 MB of plays by Shakespeare: see data/README.md
COURSE_GPT = "data/course-gpt-chars"   # .pt and .json: the course's GPT of gpt-train-chars, see data/README.md
START = "ROMEO:\n"
LENGTH = 300
TEMPERATURES = [0.5, 0.8, 1.0, 1.5]


@cache
def text() -> str:
    return (paths.ROOT / DATA).read_text(encoding="utf-8")


@cache
def chars() -> tuple[str, ...]:
    return tuple(sorted(set(text())))


@cache
def bigram_probs() -> np.ndarray:
    """The table of lm-sampling: row i is the distribution of the next character after chars()[i]."""
    t, cs = text(), chars()
    counts = {}
    for i in range(len(t) - 1):
        counts[t[i:i + 2]] = counts.get(t[i:i + 2], 0) + 1
    n = np.array([[counts.get(a + b, 0) for b in cs] for a in cs], dtype=float)
    return n / n.sum(axis=1, keepdims=True)


def number(x: float, digits: int, lang: str) -> str:
    text_ = f"{x:.{digits}f}"
    return text_.replace(".", ",") if lang == "nl" else text_


def loop_picture(labels: dict) -> str:
    """One step of writing: the text so far (data), its last `block` characters as the window, the GPT, its logits at
    every place of the window (only those of the last place in the prediction role), the probabilities they give, and
    the drawn character added at the end of the text (prediction, dashed), as the start of the next step."""
    W, BW, BH, STEP = 340, 24, 26, 27
    text_ = "ROMEO:\nNo"
    x0 = 35
    xs = [x0 + i * STEP for i in range(len(text_) + 1)]
    first = len(text_) - 6                      # a window of 6 characters, for the picture
    left, right = xs[first], xs[len(text_) - 1] + BW
    mono = "font-family:var(--ea-font-code)"
    p = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    for i, ch in enumerate(text_):
        shown = "↵" if ch == "\n" else ch
        p.append(f'<rect x="{xs[i]}" y="28" width="{BW}" height="{BH}" rx="4" style="fill:var(--ea-data-wash);stroke:var(--ea-data)"/>'
                 f'<text x="{xs[i] + BW / 2}" y="46" style="font-size:13px;fill:var(--ea-ink);text-anchor:middle;{mono}">{html.escape(shown)}</text>')
    new = xs[len(text_)]
    p.append(f'<rect x="{new}" y="28" width="{BW}" height="{BH}" rx="4" style="fill:var(--ea-pred-wash);stroke:var(--ea-pred);stroke-dasharray:3 2"/>'
             f'<text x="{new + BW / 2}" y="46" style="font-size:13px;fill:var(--ea-pred-text);text-anchor:middle;{mono}">,</text>')
    line = "fill:none;stroke:var(--ea-ink-3);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round"
    # the window: a bracket under its characters, and its name on the left
    p.append(f'<path d="M{left} 58 V62 H{right} V58" style="{line}"/>'
             f'<text x="{left - 6}" y="66" style="font-size:12px;fill:var(--ea-ink-2);text-anchor:end">{html.escape(labels["window"])}</text>')
    mid = (left + right) / 2
    p.append(f'<path d="M{mid} 64 V80 M{mid - 3.5} 76 L{mid} 80 L{mid + 3.5} 76" style="{line}"/>')
    p.append(f'<rect x="{left}" y="82" width="{right - left}" height="26" rx="6" style="fill:var(--ea-page);stroke:var(--ea-ink-3)"/>'
             f'<text x="{mid}" y="99" style="font-size:13px;fill:var(--ea-ink);text-anchor:middle">{html.escape(labels["gpt"])}</text>')
    # the logits: one row of scores at every place; only the last place's are about the next character
    for i in range(first, len(text_)):
        last = i == len(text_) - 1
        style = "fill:var(--ea-pred-wash);stroke:var(--ea-pred)" if last else "fill:var(--ea-page);stroke:var(--ea-ink-3)"
        p.append(f'<rect x="{xs[i]}" y="118" width="{BW}" height="14" rx="3" style="{style}"/>')
    p.append(f'<text x="{left - 6}" y="129" style="font-size:12px;fill:var(--ea-ink-2);text-anchor:end">{html.escape(labels["logits"])}</text>')
    lx = xs[len(text_) - 1] + BW / 2
    p.append(f'<path d="M{lx} 134 V146 M{lx - 3.5} 142 L{lx} 146 L{lx + 3.5} 142" style="{line}"/>')
    # the probabilities of the next character, after the temperature and softmax
    # the course's GPT after "ROMEO:\nNo" at a temperature of 0.8: its three likeliest next characters
    for k, (ch, pr) in enumerate([(",", 0.39), ("t", 0.23), ("␣", 0.16)]):
        y = 152 + k * 18
        p.append(f'<text x="{left + 6}" y="{y + 11}" style="font-size:12px;fill:var(--ea-ink);text-anchor:middle;{mono}">{ch}</text>'
                 f'<rect x="{left + 16}" y="{y + 1}" width="{pr * 200:.0f}" height="12" rx="2" style="fill:var(--ea-pred)"/>')
    p.append(f'<text x="{left - 6}" y="165" style="font-size:12px;fill:var(--ea-ink-2);text-anchor:end">{html.escape(labels["probs"])}</text>')
    # draw, add the character to the text, and go again
    rx = 326
    p.append(f'<path d="M{left + 16 + 0.39 * 200 + 6} 159 H{rx} V41 H{new + BW + 3} M{new + BW + 7} 37.5 L{new + BW + 3} 41 L{new + BW + 7} 44.5" style="{line}"/>'
             f'<text x="{rx - 6}" y="214" style="font-size:12px;fill:var(--ea-ink-2);text-anchor:end">{html.escape(labels["draw"])}</text>')
    H = 222
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:{W}px">'
            f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["alt"])}</title>{"".join(p)}</svg></figure>')


# The lesson is done when these are correct. "passes" is the choice question in the lesson.
REQUIRED = ["passes", "gpt-probs", "gpt-generate"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_SCENARIO = "With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`"
_SCENARIO_NL = "Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn"


def _close(got, want) -> bool:
    if isinstance(want, (list, tuple)):
        return isinstance(got, (list, tuple)) and len(got) == len(want) and all(_close(g, w) for g, w in zip(got, want))
    if isinstance(want, float):
        return isinstance(got, (int, float)) and not isinstance(got, bool) and abs(got - want) <= 1e-4
    return type(got) is type(want) and got == want


# A small GPT of the learner's, untrained, for 5 characters, texts of at most 4: what the checks give it.
_MODEL = ("import torch\nfrom llm.gpt import GPT\n\ntorch.manual_seed(0)\nmodel = GPT(5, 8, 2, 1, 4)\nchars = \"abcde\"\n"
          "row_of = {}\nfor k in range(len(chars)):\n    row_of[chars[k]] = k\n")
# The same, with a GPT that reads 6 characters: a function that always reads 4 is not right.
_MODEL_6 = _MODEL.replace("GPT(5, 8, 2, 1, 4)", "GPT(5, 8, 2, 1, 6)")

# --- exercise 2: gpt_probs ----------------------------------------------------------------------------------------------

_P_HEAD = ("import torch\n\n\ndef gpt_probs(model, row_of, text, tau):\n"
           "    # the longest text your GPT reads: the rows of its table of places\n    block = model.pos.num_embeddings\n")
_P_IDS = "    ids = []\n    for ch in text[-block:]:\n        ids.append(row_of[ch])\n"
_P_LOGITS = "    with torch.no_grad():\n        logits = model(torch.tensor([ids]))\n"
_P_RETURN = "    return torch.softmax(logits[0, -1] / tau, dim=0).numpy()\n"
_P_START = (_P_HEAD + "    # the ids of the last block characters of the text\n    ...\n"
            "    # the logits of your GPT, without a graph\n    ...\n"
            "    # the logits of the last place, divided by tau: the probabilities of the next character, as NumPy\n"
            "    return ...\n")


def _p(ids: str = _P_IDS, logits: str = _P_LOGITS, ret: str = _P_RETURN) -> str:
    return _P_HEAD + ids + logits + ret


_P_SCENARIOS = [
    # the lesson's example: a text shorter than block
    _MODEL + "p = gpt_probs(model, row_of, \"abc\", 1.0)\np.round(4).tolist()",
    # a text longer than block: the GPT reads its last 4 characters
    _MODEL + "p = gpt_probs(model, row_of, \"eeeeabcd\", 1.0)\np.round(4).tolist()",
    # two temperatures
    _MODEL + "[gpt_probs(model, row_of, \"abc\", 0.5).round(4).tolist(), gpt_probs(model, row_of, \"abc\", 2.0).round(4).tolist()]",
    # a GPT that reads 6 characters
    _MODEL_6 + "p = gpt_probs(model, row_of, \"eeeeeeabcd\", 1.0)\np.round(4).tolist()",
]

# --- exercise 3: gpt_generate ---------------------------------------------------------------------------------------------

_G_HEAD = ("import numpy as np\nfrom llm.gpt_probs import gpt_probs\n\n\ndef gpt_generate(model, chars, start, n, tau, seed):\n"
           "    rng = np.random.default_rng(seed)\n    row_of = {}\n    for k in range(len(chars)):\n        row_of[chars[k]] = k\n"
           "    text = start\n")
_G_LOOP = "    while len(text) < n:\n"
_G_PROBS = "        p = gpt_probs(model, row_of, text, tau)\n"
_G_DRAW = "        j = rng.choice(len(chars), p=p)\n        text = text + chars[j]\n"
_G_END = "    return text\n"
_G_START = _G_HEAD + "    # draw characters with the probabilities of your GPT until the text has n of them\n    ...\n" + _G_END


def _g(loop: str = _G_LOOP, probs: str = _G_PROBS, draw: str = _G_DRAW) -> str:
    return _G_HEAD + loop + probs + draw + _G_END


_G_SCENARIOS = [
    # the lesson's example
    _MODEL + "gpt_generate(model, chars, \"ab\", 9, 1.0, 1)",
    # other starts, seeds and temperatures, and texts longer than the GPT reads
    _MODEL + "[gpt_generate(model, chars, \"e\", 7, 0.5, 2), gpt_generate(model, chars, \"abcde\", 14, 1.5, 3)]",
    # a GPT that reads 6 characters
    _MODEL_6 + "gpt_generate(model, chars, \"abcdeab\", 14, 1.0, 4)",
]

EXERCISES = {
    "gpt-probs": ScenarioBox(
        name="gpt-probs",
        start=_P_START,
        solution=_p(),
        scenarios=_P_SCENARIOS,
        compare=_close,
        save_as=("llm", "gpt_probs"),
        digits=4,
        first=[_DOTS],
        mistakes=[
            Mistake("too-long", error=lambda e: isinstance(e, IndexError) and "index out of range in self" in str(e),
                    en="Your GPT stops with an error: it gets a text longer than its table of places, and has no row for those places. Give it at most `block` characters: the last ones, `text[-block:]`.",
                    nl="Je GPT stopt met een fout: hij krijgt een tekst die langer is dan zijn tabel van plekken, en heeft geen rij voor die plekken. Geef hem hooguit `block` tekens: de laatste, `text[-block:]`."),
            Mistake("no-batch", error=lambda e: isinstance(e, IndexError) and "tuple index out of range" in str(e),
                    en="Your GPT stops with an error: it reads a mini-batch of texts, shape (1, n) for one text, and gets the ids as one row, shape (n,). Put your one text in a list: `torch.tensor([ids])`.",
                    nl="Je GPT stopt met een fout: hij leest een mini-batch van teksten, shape (1, n) voor één tekst, en krijgt de ids als één rij, shape (n,). Zet je ene tekst in een lijst: `torch.tensor([ids])`."),
            Mistake("not-array", error=lambda e: isinstance(e, TypeError) and "round()" in str(e)
                    or isinstance(e, AttributeError) and "has no attribute 'round'" in str(e),
                    en="The lines after your function call `.round(4)` on what `gpt_probs` gives back, as on a NumPy array, and yours gives back something else, such as a tensor or a list. Give back a NumPy array, as a row of your bigram table was: end with `.numpy()`.",
                    nl="De regels na je functie roepen `.round(4)` aan op wat `gpt_probs` teruggeeft, zoals op een NumPy-array, en de jouwe geeft iets anders terug, zoals een tensor of een lijst. Geef een NumPy-array terug, zoals een rij van je bigram-tabel was: eindig met `.numpy()`."),
            Mistake("no-grad", error=lambda e: isinstance(e, RuntimeError) and "requires grad" in str(e),
                    en="PyTorch stops at `.numpy()`: your logits still hang in a graph, as if you were going to train. Writing needs no gradients: compute them inside `with torch.no_grad():`.",
                    nl="PyTorch stopt bij `.numpy()`: je logits hangen nog in een graaf, alsof je gaat trainen. Schrijven heeft geen gradients nodig: bereken ze binnen `with torch.no_grad():`."),
            Mistake("first-chars", wrong=_p(ids=_P_IDS.replace("text[-block:]", "text[:block]")),
                    en=_SCENARIO + ": for a long text, your GPT reads its first `block` characters. The next character follows the end of the text: give it the last ones, `text[-block:]`.",
                    nl=_SCENARIO_NL + ": bij een lange tekst leest je GPT de eerste `block` tekens. Het volgende teken volgt op het eind van de tekst: geef hem de laatste, `text[-block:]`."),
            Mistake("first-place", wrong=_p(ret=_P_RETURN.replace("logits[0, -1]", "logits[0, 0]")),
                    en=_SCENARIO + ": you take the logits of the first place, about the character after the first one. The next character of the text comes from the last place: `logits[0, -1]`.",
                    nl=_SCENARIO_NL + ": je neemt de logits van de eerste plek, over het teken na het eerste. Het volgende teken van de tekst komt van de laatste plek: `logits[0, -1]`."),
            Mistake("no-tau", wrong=_p(ret=_P_RETURN.replace(" / tau", "")),
                    en=_SCENARIO + ": your probabilities are the same for every temperature. Divide the logits by `tau` before softmax.",
                    nl=_SCENARIO_NL + ": je kansen zijn bij elke temperature hetzelfde. Deel de logits door `tau` vóór softmax."),
            Mistake("times-tau", wrong=_p(ret=_P_RETURN.replace(" / tau", " * tau")),
                    en=_SCENARIO + ": you multiply the logits by `tau`, so a low temperature makes the probabilities flatter. Divide by it: a low temperature makes them sharper.",
                    nl=_SCENARIO_NL + ": je vermenigvuldigt de logits met `tau`, dus een lage temperature maakt de kansen vlakker. Deel erdoor: een lage temperature maakt ze scherper."),
            Mistake("after-softmax", wrong=_p(ret="    return (torch.softmax(logits[0, -1], dim=0) / tau).numpy()\n"),
                    en=_SCENARIO + ": you divide the probabilities by `tau`, after softmax, so they no longer add up to 1. The temperature divides the logits, before softmax.",
                    nl=_SCENARIO_NL + ": je deelt de kansen door `tau`, na softmax, dus ze tellen niet meer op tot 1. De temperature deelt de logits, vóór softmax."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the three `...`: the ids of the last `block` characters, the logits of your GPT without a graph, and the probabilities of the next character.",
                      "nl": "Vervang de drie `...`: de ids van de laatste `block` tekens, de logits van je GPT zonder graaf, en de kansen op het volgende teken."},
            "wrong": {"en": _SCENARIO + ". The last `block` characters as ids, the logits of the last place, divided by `tau`, and softmax: steps 1 to 3 of the recipe?",
                      "nl": _SCENARIO_NL + ". De laatste `block` tekens als ids, de logits van de laatste plek, gedeeld door `tau`, en softmax: stap 1 tot en met 3 van het recept?"},
            "almost": {"en": "Right for the first example. But " + _SCENARIO[0].lower() + _SCENARIO[1:] + ". Does your function read at most `block` characters, the last ones, and use `tau`?",
                       "nl": "Goed voor het eerste voorbeeld. Maar " + _SCENARIO_NL[0].lower() + _SCENARIO_NL[1:] + ". Leest je functie hooguit `block` tekens, de laatste, en gebruikt hij `tau`?"},
            "correct": {"en": "Your GPT gives the probabilities of the next character after any text, at any temperature: a row of numbers, one for every character, that add up to 1, as one row of your bigram table did. It is in your project now, as `{saved}`.",
                        "nl": "Je GPT geeft de kansen op het volgende teken na elke tekst, bij elke temperature: een rij getallen, één voor elk teken, die samen 1 zijn, zoals één rij van je bigram-tabel. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `gpt_probs` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `gpt_probs` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "gpt-generate": ScenarioBox(
        name="gpt-generate",
        start=_G_START,
        solution=_g(),
        scenarios=_G_SCENARIOS,
        compare=_close,
        save_as=("llm", "gpt_generate"),
        first=[_DOTS],
        mistakes=[
            Mistake("last-char", wrong=_g(probs=_G_PROBS.replace("row_of, text, tau", "row_of, text[-1], tau")),
                    en=_SCENARIO + ": you give your GPT only the last character, as the bigram model had. Give it the whole text: your `gpt_probs` takes the last `block` characters itself.",
                    nl=_SCENARIO_NL + ": je geeft je GPT alleen het laatste teken, zoals het bigram-model had. Geef hem de hele tekst: je `gpt_probs` neemt zelf de laatste `block` tekens."),
            Mistake("greedy", wrong=_g(draw=_G_DRAW.replace("rng.choice(len(chars), p=p)", "np.argmax(p)")),
                    en=_SCENARIO + ": your function always takes the most likely character, as greedy decoding does. Draw it with the probabilities: `rng.choice(len(chars), p=p)`.",
                    nl=_SCENARIO_NL + ": je functie neemt altijd het waarschijnlijkste teken, zoals greedy decoding. Trek het met de kansen: `rng.choice(len(chars), p=p)`."),
            Mistake("n-more", wrong=_g(loop="    while len(text) < len(start) + n:\n"),
                    en=_SCENARIO + ": your text gets `n` new characters, after the start. As in lm-sampling, `n` is the length of the whole text: `while len(text) < n:`.",
                    nl=_SCENARIO_NL + ": je tekst krijgt `n` nieuwe tekens, na het begin. Net als in lm-sampling is `n` de lengte van de hele tekst: `while len(text) < n:`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the loop of your `generate` of lm-sampling, with the probabilities of your GPT in place of a row of the table.",
                      "nl": "Vervang de `...` door de loop van je `generate` uit lm-sampling, met de kansen van je GPT in plaats van een rij van de tabel."},
            "wrong": {"en": _SCENARIO + ". As long as the text is shorter than `n`: the probabilities of your GPT for the whole text so far, a drawn character, added to the text: steps 4 and 5 of the recipe?",
                      "nl": _SCENARIO_NL + ". Zolang de tekst korter is dan `n`: de kansen van je GPT voor de hele tekst tot nu toe, een getrokken teken, achter de tekst gezet: stap 4 en 5 van het recept?"},
            "almost": {"en": "Right for the first example. But " + _SCENARIO[0].lower() + _SCENARIO[1:] + ". Does your function use `start`, `n`, `tau` and `seed`, and no fixed values?",
                       "nl": "Goed voor het eerste voorbeeld. Maar " + _SCENARIO_NL[0].lower() + _SCENARIO_NL[1:] + ". Gebruikt je functie `start`, `n`, `tau` en `seed`, en geen vaste waarden?"},
            "correct": {"en": "Your sampling loop writes with your GPT: the same loop as for the bigram model, with one line changed. It is in your project now, as `{saved}`.",
                        "nl": "Je sampling-loop schrijft met je GPT: dezelfde loop als voor het bigram-model, met één regel anders. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `gpt_generate` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `gpt_generate` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# --- the demo: the learner's GPT writes, next to their bigram model -----------------------------------------------------------


def _saved(module: str) -> bool:
    return (projects.folder() / "llm" / f"{module}.py").exists()


def trained() -> dict:
    """Which trained GPT the demo writes with: the learner's of gpt-train-chars ("yours"), else the course's
    ("course"); with its sizes, its characters, its best validation loss and its checkpoint."""
    folder = projects.folder() / "llm"
    for source, base in (("yours", folder / "gpt_chars"), ("course", paths.ROOT / COURSE_GPT)):
        pt, info = base.with_suffix(".pt"), base.with_suffix(".json")
        if pt.exists() and info.exists():
            try:
                meta = json.loads(info.read_text(encoding="utf-8"))
            except ValueError:
                continue
            return {"source": source, "sizes": meta["sizes"], "chars": meta["chars"], "best": meta["best"], "path": pt}
    return {"source": None}


def demo(tau: float, seed: int = 1):
    """For the demo: None when gpt_probs.py or gpt_generate.py is not in the project yet; "gpt" when the trained GPT
    does not load into the learner's GPT or their code stops with an error; "slow" when writing takes more than a minute;
    else a dict with the GPT's text, the bigram model's text, which GPT wrote, and the modules the course stood in for."""
    if not (_saved("gpt_probs") and _saved("gpt_generate")):
        return None
    which = trained()
    if which["source"] is None:
        return "gpt"
    run = run_code("from llm.gpt import GPT\nfrom llm.gpt_generate import gpt_generate\nfrom llm.sampling import generate")
    if run.error is not None:
        return "gpt"
    GPT, gpt_generate, generate = run.names["GPT"], run.names["gpt_generate"], run.names["generate"]
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]
    d, h, n_layers, block = which["sizes"]

    def work():
        model = GPT(len(which["chars"]), d, h, n_layers, block)
        model.load_state_dict(torch.load(which["path"], map_location="cpu", weights_only=True))
        model.eval()
        return {"gpt": str(gpt_generate(model, list(which["chars"]), START, LENGTH, tau, seed)),
                "bigram": str(generate(bigram_probs(), list(chars()), START, LENGTH, seed))}

    try:
        out = with_time_limit(work, 60)
    except Timeout:
        return "slow"
    except Exception:
        return "gpt"
    return {**out, "source": which["source"], "best": which["best"], "stand_ins": stand_ins}


def shown_text(text_: str) -> str:
    """A written text in a block of its own: as it is, in a fenced block, so that its new lines show."""
    return "```text\n" + text_.replace("```", "'''") + "\n```"


# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("gpt-probs", _P_START, "empty", None),
    ("gpt-probs", _p(), "correct", None),
    ("gpt-probs", _p(ids="    ids = []\n    for ch in text:\n        ids.append(row_of[ch])\n    ids = ids[-block:]\n"), "correct", None),
    ("gpt-probs", _p(ret="    p = torch.softmax(logits[0, -1] / tau, dim=0)\n    return p.numpy()\n"), "correct", None),
    ("gpt-probs", _p(ids="    ids = []\n    for ch in text:\n        ids.append(row_of[ch])\n"), "error", "too-long"),
    ("gpt-probs", _p(logits="    logits = model(torch.tensor([ids]))\n"), "error", "no-grad"),
    ("gpt-probs", _p(logits="    with torch.no_grad():\n        logits = model(torch.tensor(ids))\n"), "error", "no-batch"),
    ("gpt-probs", _p(ret="    return torch.softmax(logits[0, -1] / tau, dim=0)\n"), "error", "not-array"),
    ("gpt-probs", _p(ret="    return torch.softmax(logits[0, -1] / tau, dim=0).tolist()\n"), "error", "not-array"),
    ("gpt-probs", _p(ids="    ids = []\n    for i in range(block):\n        ids.append(row_of[text[i]])\n"), "error", None),
    ("gpt-probs", _p(ids=_P_IDS.replace("text[-block:]", "text[-4:]")), "almost", None),
    ("gpt-probs", _p(ids=_P_IDS.replace("text[-block:]", "text[:block]")), "wrong", "first-chars"),
    ("gpt-probs", _p(ret=_P_RETURN.replace("logits[0, -1]", "logits[0, 0]")), "wrong", "first-place"),
    ("gpt-probs", _p(ret=_P_RETURN.replace(" / tau", "")), "wrong", "no-tau"),
    ("gpt-probs", _p(ret=_P_RETURN.replace(" / tau", " * tau")), "wrong", "times-tau"),
    ("gpt-probs", _p(ret="    return (torch.softmax(logits[0, -1], dim=0) / tau).numpy()\n"), "wrong", "after-softmax"),
    ("gpt-generate", _G_START, "empty", None),
    ("gpt-generate", _g(), "correct", None),
    ("gpt-generate", _g(draw="        j = rng.choice(len(chars), p=p)\n        text += chars[j]\n"), "correct", None),
    ("gpt-generate", _g(probs=_G_PROBS.replace("row_of, text, tau", "row_of, text[-1], tau")), "wrong", "last-char"),
    ("gpt-generate", _g(draw=_G_DRAW.replace("rng.choice(len(chars), p=p)", "np.argmax(p)")), "wrong", "greedy"),
    ("gpt-generate", _g(loop="    while len(text) < len(start) + n:\n"), "wrong", "n-more"),
    ("gpt-generate", _g(probs=_G_PROBS.replace("row_of, text, tau", "row_of, text, 1.0")), "almost", None),
    ("gpt-generate", _g(probs=_G_PROBS.replace("row_of, text, tau", "row_of, text[-4:], tau")), "almost", None),
]
