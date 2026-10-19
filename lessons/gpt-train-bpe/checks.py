# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the project lesson gpt-train-bpe: train_bpe_fast, byte pair encoding that counts every different piece
of a text once, with how often it occurs; and encode_fast, which encodes every different piece once. Both are saved in
the learner's project (work/projects/llm/fast_bpe.py and fast_encode.py) when right. The demo trains the learner's
tokenizer on short stories of TinyStories, then their GPT on its tokens, behind a button, on a GPU when there is one,
and lets it write a story; the course's run of a GPT on the characters of the same stories stands next to it."""
import importlib.util
import json
import math
import re
import time
from functools import cache

import torch

from everything_ai import paths, projects
from everything_ai.check import FunctionBox, Mistake, Result, run_code


def _train_chars():
    """The checks of gpt-train-chars, whose run this lesson repeats on tokens: their device, plans and helpers.
    Loading them registers that lesson's project modules; this lesson's own list follows below."""
    spec = importlib.util.spec_from_file_location("everything_ai_checks_gpt_train_chars_for_bpe",
                                                  paths.LESSONS / "gpt-train-chars" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _train_chars()

# The run uses the learner's tokenizer of the tok- lessons and this lesson, and their GPT, gpt_loaders and Trainer;
# where one is missing or broken, the course's version stands in (template 1.3).
BUILDS_ON = (("llm", "merge", "tok-bpe", "merge"), ("llm", "bpe", "tok-bpe", "train-bpe"),
             ("llm", "byte_bpe", "tok-byte-level", "train-byte-bpe"), ("llm", "byte_decode", "tok-byte-level", "decode"),
             ("llm", "chunk_pairs", "tok-pre-tokenization", "chunk-pair-counts"),
             ("llm", "pre_bpe", "tok-pre-tokenization", "train-pre-bpe"),
             ("llm", "encode", "tok-special-tokens", "encode"), ("llm", "decode", "tok-special-tokens", "decode"),
             ("llm", "attention", "attn-transformer-block", "causal-self-attention"),
             ("llm", "block", "attn-transformer-block", "block"), ("llm", "gpt", "gpt-assemble", "gpt"),
             ("llm", "gpt_data", "gpt-assemble", "gpt-examples"),
             ("nn", "train_epoch", "nn-trainer", "train-epoch"), ("nn", "trainer", "nn-trainer", "trainer"),
             ("llm", "gpt_loaders", "gpt-train-chars", "gpt-loaders"),
             ("llm", "fast_bpe", "gpt-train-bpe", "train-bpe-fast"), ("llm", "fast_encode", "gpt-train-bpe", "encode-fast"))
projects.builds_on(*BUILDS_ON)

DATA = "data/tinystories-small.txt"   # 6,189 short stories of TinyStories, joined by <|endoftext|>: see data/README.md
END = "<|endoftext|>"
PATTERN = r" ?\w+| ?[^\w\s]+|\s+"   # the pieces of tok-pre-tokenization
MERGES = 1000                      # merge rules of the tokenizer: ids 256 to 1255, and 1256 for end-of-text
VOCAB = 256 + MERGES + 1
# What the demo trains: the GPT of gpt-train-chars on each plan, with the tokens as its vocabulary; 20 epochs on a GPU,
# 10 on a CPU, so that every run does the same work and can be held against the course's runs.
PLANS = {"gpu": {"sizes": (384, 6, 6, 256), "batch": 64, "epochs": 20},
         "cpu": {"sizes": (128, 4, 5, 128), "batch": 32, "epochs": 10}}
START = "Once upon a time"
LENGTH = 150   # tokens of the story the trained GPT writes
TAU = 0.8      # its temperature, as in gpt-generate

# The pieces of the first 90% of the stories, the text the tokenizer trains on (tests/test_lessons.py checks them):
# all of them, the different ones, and the five most common, with their counts. FIRST_PIECES: the same for the first 20,000
# characters, the worked example (FIRST_PIECES).
PIECES, DISTINCT = 1_082_711, 7_598
TOP_PIECES = [(".", 85043), (",", 47688), (" the", 42223), (" and", 39445), (" a", 30678)]
FIRST_PIECES = {"pieces": 4857, "different": 845}
SPEEDUP_PIECES = round(PIECES / DISTINCT)
# On the course's computer, 4 cores, one thread: the course's train_pre_bpe of tok-pre-tokenization with 1,000 rules on
# those pieces, and its train_bpe_fast, in seconds; and the characters per token of the course's tokenizer on all stories.
# --- measured: data/tools and the course's runs (made by the course's scripts; see the notes) ---
NAIVE_SECONDS, FAST_SECONDS = 1263, 14
NAIVE_MINUTES = round(NAIVE_SECONDS / 60)
SPEEDUP = round(NAIVE_SECONDS / FAST_SECONDS / 10) * 10
CHARS_PER_TOKEN = 3.382   # all stories, with the course's tokenizer
# The course's runs of the demo's train with its own solutions, one thread, PyTorch 2.14.1: per run its seed, minutes,
# best validation loss per token and per character, and the loss after every epoch (epoch, training, validation).
# The GPU plan ran on the course's CPU, with 4 threads: nearly the numbers of a GPU, only slower.
REFERENCE = {
    "cpu": [
        {"seed": 0, "minutes": 34.9, "best": 2.4808, "best_epoch": 10, "best_per_char": 0.7371, "val_tokens": 147860, "val_chars": 497664,
         "history": [(1, 4.7366, 3.6671), (2, 3.3982, 3.1972), (3, 3.0491, 2.9572), (4, 2.8306, 2.7916), (5, 2.6778, 2.6853), (6, 2.5548, 2.5963), (7, 2.4704, 2.534), (8, 2.4048, 2.4999), (9, 2.3663, 2.483), (10, 2.3465, 2.4808)]},
        {"seed": 1, "minutes": 34.3, "best": 2.4858, "best_epoch": 10, "best_per_char": 0.7386, "val_tokens": 147860, "val_chars": 497664,
         "history": [(1, 4.7322, 3.6772), (2, 3.4084, 3.2128), (3, 3.055, 2.9588), (4, 2.837, 2.7973), (5, 2.6754, 2.6796), (6, 2.5613, 2.6004), (7, 2.4704, 2.5397), (8, 2.4076, 2.5045), (9, 2.3656, 2.4887), (10, 2.3473, 2.4858)]},
        {"seed": 2, "minutes": 33.9, "best": 2.4652, "best_epoch": 10, "best_per_char": 0.7324, "val_tokens": 147860, "val_chars": 497664,
         "history": [(1, 4.7604, 3.6713), (2, 3.3969, 3.1956), (3, 3.0442, 2.9539), (4, 2.8209, 2.7841), (5, 2.6596, 2.6615), (6, 2.5376, 2.574), (7, 2.4196, 2.5216), (8, 2.3839, 2.4839), (9, 2.3458, 2.4678), (10, 2.3251, 2.4652)]},
    ],
    "gpu": [
        {"seed": 0, "minutes": 143.6, "best": 1.9717, "best_epoch": 17, "best_per_char": 0.5858, "val_tokens": 147860, "val_chars": 497664,
         "history": [(1, 5.3716, 4.0407), (2, 3.6818, 3.4595), (3, 3.2538, 3.099), (4, 2.8825, 2.7566), (5, 2.5826, 2.5065), (6, 2.3498, 2.3738), (7, 2.1793, 2.2223), (8, 2.0606, 2.1494), (9, 1.9438, 2.0951), (10, 1.8436, 2.0499), (11, 1.7705, 2.0193), (12, 1.6912, 2.0014), (13, 1.6032, 1.9902), (14, 1.5494, 1.9821), (15, 1.5002, 1.9783), (16, 1.4363, 1.9747), (17, 1.4064, 1.9717), (18, 1.3561, 1.974), (19, 1.3439, 1.9743), (20, 1.347, 1.9749)]},
    ],
}
# The course's GPT on characters of the same stories: the CPU plan of gpt-train-chars, 3 epochs, as many steps as the
# CPU plan's 10 epochs on tokens; seed 0, one thread. Its best validation loss per character, and its story from
# START at temperature TAU, 500 characters.
CHARS = {"cpu": {"epochs": 3, "best": 0.9791, "story": "Once upon a time, in a small and said, \"Of, little catch is not anymore. We can help it.\" They made a nodder truck, and they to play are green back to play with the same. Tim and the sun warm surprised and had to surprise. He had a smile and said, \"Thank you, look to will be the door, much!\" Sue smiled and said, \"You loud!\"\nHer mom smiled. \"Yes: lank the other metone? I want to be want to be fall on the strong beaute you loud under the new clean. Please was not looking at a room dog named Lily. "}}
# A run that ends at or below this validation loss per token has learned as the course's runs did: the highest end
# of those runs, plus a margin for another computer, another PyTorch, and float rounding on a GPU.
THRESHOLD = {"cpu": 2.55, "gpu": 2.1}
# --- end of measured ---


@cache
def text() -> str:
    return (paths.ROOT / DATA).read_text(encoding="utf-8")


def split_text() -> tuple[str, str]:
    """The stories to train the tokenizer on, the first 90% of the text up to the end-of-text after it, and the rest."""
    t = text()
    cut = t.index(END, int(0.9 * len(t)))
    return t[:cut], t[cut + len(END):]


def n_params(vocab: int, d: int, n_layers: int, block: int) -> int:
    return T.n_params(vocab, d, n_layers, block)


number, grouped = T.number, T.grouped


def per_char(loss_per_token: float, tokens: int, chars: int) -> float:
    """A loss per token as a loss per character: the same total over the characters the tokens hold."""
    return loss_per_token * tokens / chars


def pieces_picture(labels: dict) -> str:
    """The pieces of "the the the cat cow cup": left every piece, right every different piece once with its count;
    under them the count of the pair t h, from both sides. Ink only: these are counts of the text."""
    import html
    pieces = ["the", " the", " the", " cat", " cow", " cup"]
    once = [("the", 1), (" the", 2), (" cat", 1), (" cow", 1), (" cup", 1)]
    W, ROW, TOP = 330, 26, 52
    mono = "font-family:var(--ea-font-code)"
    box = "fill:var(--ea-inset);stroke:var(--ea-ink-3);stroke-width:1"
    parts = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>',
             f'<text x="0" y="40" style="font-size:12.5px;fill:var(--ea-ink-2)">{html.escape(labels["all"])}</text>',
             f'<text x="150" y="40" style="font-size:12.5px;fill:var(--ea-ink-2)">{html.escape(labels["once"])}</text>']
    for k, piece in enumerate(pieces):
        y = TOP + k * ROW
        parts.append(f'<rect x="0" y="{y}" width="92" height="20" rx="4" style="{box}"/>'
                     f'<text x="8" y="{y + 14.5}" style="{mono};font-size:13px;fill:var(--ea-ink)">{html.escape(piece.replace(" ", "␣"))}</text>')
    for k, (piece, n) in enumerate(once):
        y = TOP + k * ROW
        parts.append(f'<rect x="150" y="{y}" width="92" height="20" rx="4" style="{box}"/>'
                     f'<text x="158" y="{y + 14.5}" style="{mono};font-size:13px;fill:var(--ea-ink)">{html.escape(piece.replace(" ", "␣"))}</text>'
                     f'<text x="252" y="{y + 14.5}" style="font-size:13px;font-weight:600;fill:var(--ea-ink)">× {n}</text>')
    y = TOP + len(pieces) * ROW + 16
    parts.append(f'<text x="0" y="{y}" style="font-size:13px;fill:var(--ea-ink)">{html.escape(labels["pair"])}</text>')
    H = y + 8
    text_ = labels["title"] + ". " + labels["alt"]
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:min(100%,{W}px)">'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(text_)}" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(text_)}</title>' + "".join(parts) + "</svg></figure>")


def history_picture(history: list, labels: dict, epochs: int | None = None) -> str:
    """The loss per token of the run after every epoch: training solid, validation dashed and open, as in
    gpt-train-chars; a grey line at the loss of guessing evenly among the 1,257 ids."""
    from everything_ai import chart
    n = max(epochs or 0, history[-1]["epoch"], 2)
    xs = [hist["epoch"] for hist in history]
    low = min([hist["train_loss"] for hist in history] + [2.0])
    guess = math.log(VOCAB)
    panel = chart.Panel((0.5, n + 0.5), (math.floor(low * 2 - 0.4) / 2, 8.0), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    panel.line([0.5, n + 0.5], [guess] * 2, "ink-3", 1, dashed=True)
    panel.text(n + 0.4, guess + 0.3, labels["guess"], "ink-3", "end")
    panel.line(xs, [hist["train_loss"] for hist in history], "loss", 2.4)
    panel.points(xs, [hist["train_loss"] for hist in history], "loss")
    panel.line(xs, [hist["val_loss"] for hist in history], "loss", 1.6, dashed=True)
    panel.points(xs, [hist["val_loss"] for hist in history], "loss", hollow=True)
    panel.text(0.6, 6.3, f"● {labels['train']}   ○ {labels['val']}", "loss", "start")
    svg = chart.figure([panel], alt=labels["alt"], width=330, height=260)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


# The lesson is done when these are correct. "per-char" is the choice question in the lesson.
REQUIRED = ["train-bpe-fast", "encode-fast", "per-char"]

_DOTS = Mistake("dots-left", matches=lambda got, text_: re.search(r"^[^#\n]*\.\.\.", text_, re.MULTILINE) is not None,
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


# --- exercise 1: train_bpe_fast --------------------------------------------------------------------------------------------

_F_HEAD = ("import re\nfrom llm.byte_bpe import merge_ids\nfrom llm.pre_bpe import PATTERN\n\nEND = \"<|endoftext|>\"\n\n\n"
           "def train_bpe_fast(text, n):\n")
_F_COUNT = ("    counts_of = {}\n    for part in text.split(END):\n        for piece in re.findall(PATTERN, part):\n"
            "            if piece in counts_of:\n                counts_of[piece] = counts_of[piece] + 1\n"
            "            else:\n                counts_of[piece] = 1\n")
_F_WORDS = ("    words = []\n    weights = []\n    for piece in counts_of:\n        words.append(list(piece.encode(\"utf-8\")))\n"
            "        weights.append(counts_of[piece])\n")
_F_LOOP = ("    merges = {}\n    for step in range(n):\n        counts = {}\n        for k in range(len(words)):\n"
           "            ids = words[k]\n            for i in range(len(ids) - 1):\n                pair = (ids[i], ids[i + 1])\n")
_F_ADD = ("                if pair in counts:\n                    counts[pair] = counts[pair] + weights[k]\n"
          "                else:\n                    counts[pair] = weights[k]\n")
_F_BEST = ("        best = None\n        most = 0\n        for pair in counts:\n            if counts[pair] > most:\n"
           "                best = pair\n                most = counts[pair]\n        new_id = 256 + step\n        merges[best] = new_id\n"
           "        words = [merge_ids(ids, best, new_id) for ids in words]\n    return merges\n")
_F_START = (_F_HEAD + "    # every different piece of every part between two end-of-texts, with how often it occurs\n"
            "    counts_of = {}\n    ...\n" + _F_WORDS + _F_LOOP
            + "                # the pair counts as often as its piece occurs\n                ...\n" + _F_BEST)


def _f(count: str = _F_COUNT, add: str = _F_ADD, words: str = _F_WORDS, best: str = _F_BEST) -> str:
    return _F_HEAD + count + words + _F_LOOP + add + best


_F_WRONG = {
    "unweighted": _f(add=_F_ADD.replace("+ weights[k]", "+ 1").replace("= weights[k]", "= 1")),
    "end-as-text": _f(count=_F_COUNT.replace("    for part in text.split(END):\n        for piece in re.findall(PATTERN, part):\n",
                                             "    for part in [text]:\n        for piece in re.findall(PATTERN, part):\n")),
    "no-pieces": _f(count=_F_COUNT.replace("        for piece in re.findall(PATTERN, part):\n", "        for piece in [part]:\n")),
}


def _sentence_text(n: int) -> str:
    """A long text for a hidden case: the first stories of the course's data, again and again, to n characters."""
    stories = text()[:4000]
    return (stories * (n // len(stories) + 1))[:n]


# --- exercise 2: encode_fast -----------------------------------------------------------------------------------------------

_E_HEAD = ("import re\nfrom llm.encode import END, encode_part\nfrom llm.pre_bpe import PATTERN\n\n\n"
           "def encode_fast(text, merges):\n    end_id = 256 + len(merges)\n    known = {}\n    ids = []\n"
           "    parts = text.split(END)\n    for k in range(len(parts)):\n        if k > 0:\n            ids.append(end_id)\n"
           "        for piece in re.findall(PATTERN, parts[k]):\n")
_E_BODY = ("            if piece not in known:\n                known[piece] = encode_part(piece, merges)\n"
           "            for i in known[piece]:\n                ids.append(i)\n")
_E_START = (_E_HEAD + "            # a new piece: encode it once and keep its ids in known; then its ids at the end of ids\n"
            "            ...\n    return ids\n")


def _e(body: str = _E_BODY) -> str:
    return _E_HEAD + body + "    return ids\n"


_E_WRONG = {
    "only-new": _e(body=("            if piece not in known:\n                known[piece] = encode_part(piece, merges)\n"
                         "                for i in known[piece]:\n                    ids.append(i)\n")),
    "nested": _e(body=("            if piece not in known:\n                known[piece] = encode_part(piece, merges)\n"
                       "            ids.append(known[piece])\n")),
}


def _merge_ids(ids: list, pair: tuple, new_id: int) -> list:
    """The course's merge_ids of tok-byte-level: every place where `pair` stands becomes `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i = i + 2
        else:
            out.append(ids[i])
            i = i + 1
    return out


@cache
def _merges_for_hidden() -> dict:
    """300 merge rules of the course's train_bpe_fast on the first stories: for a hidden case with many rules. Made
    here with the course's merge_ids, not with run_code: run_code first tests the learner's fast_encode.py on the cases
    of encode-fast, this one among them, and would ask for these rules again before they are known."""
    code = "\n".join(line for line in _f().splitlines() if not line.startswith("from llm."))
    names = {"__name__": "__course__", "merge_ids": _merge_ids, "PATTERN": PATTERN}
    exec(compile(code, "<course train_bpe_fast>", "exec"), names)
    return names["train_bpe_fast"](text()[:20000], 300)


class _Merges:
    """The 300 rules of _merges_for_hidden as a test case: written `merges` in the check's messages."""

    def __repr__(self):
        return "merges"

    def __deepcopy__(self, memo):
        return dict(_merges_for_hidden())


class _Text:
    """A long text as a test case: written `stories` in the check's messages, n characters of the course's stories."""

    def __init__(self, n: int):
        self.n = n

    def __repr__(self):
        return "stories"

    def __deepcopy__(self, memo):
        return _sentence_text(self.n)


class _Box(FunctionBox):
    """A FunctionBox with hidden cases of long stories: when the outcome of one of those differs from the course's,
    the message names the first rule or id that differs, instead of showing all 100 rules or 70,000 ids."""

    def check(self, text, inputs=None) -> Result:
        if any(isinstance(a, _Merges) for case in self.hidden for a in case):
            _merges_for_hidden()   # made once, before the time limit of the learner's function starts
        result = super().check(text, inputs)
        if result.state != "almost" or len(result.text["en"]) < 600:
            return result
        for case in self.hidden:
            call = self._call_text(case)
            if not isinstance(case[0], _Text) or f"`{call}`" not in result.text["en"]:
                continue
            try:   # it gave its outcome within the time limit a moment ago
                got, want = self._run(self._fn(text), case), self._run(self._fn(self.solution), case)
                g, w = ([(list(v.items()) if isinstance(v, dict) else list(v)) for v in (got, want)])
            except Exception:
                return result
            k = next((i for i in range(min(len(g), len(w))) if g[i] != w[i]), min(len(g), len(w)))

            def item(v, lang):
                if k >= len(v):
                    return {"en": "nothing", "nl": "niets"}[lang]
                return f"`{v[k][0]}: {v[k][1]}`" if isinstance(v[k], tuple) and isinstance(v[k][0], tuple) else f"`{v[k]}`"
            size = {lang: "" if len(g) == len(w) else self.texts["size"][lang].format(n_got=grouped(len(g), lang), n_want=grouped(len(w), lang))
                    for lang in ("en", "nl")}
            return Result("almost", {lang: self.texts["long"][lang].format(
                call=call, size=size[lang], k=grouped(k + 1, lang), got=item(g, lang), want=item(w, lang)) for lang in ("en", "nl")})
        return result


EXERCISES = {
    "train-bpe-fast": _Box(
        name="train-bpe-fast",
        function="train_bpe_fast",
        start=_F_START,
        solution=_f(),
        visible=[("the the the cat cow cup", 2)],
        hidden=[("", 0), ("low low low lower", 4), ("hi<|endoftext|>hi<|endoftext|>ho", 2), ("aaaa aaaa", 3),
                ("café café", 2), ("the cat<|endoftext|>the hat<|endoftext|>a cat", 3), (_Text(300_000), 100)],
        python_values=True,
        save_as=("llm", "fast_bpe"),
        first=[_DOTS],
        mistakes=[
            Mistake("unweighted", wrong=_F_WRONG["unweighted"],
                    en=_CALL + ": every different piece counts once, however often it occurs. A pair in \" the\", which occurs twice, should count twice: `counts_of` holds how often every piece occurs, and a pair adds `weights[k]` to its count, not 1.",
                    nl=_CALL_NL + ": elk verschillend stuk telt één keer, hoe vaak het ook voorkomt. Een paar in \" the\", dat twee keer voorkomt, hoort twee keer te tellen: `counts_of` houdt bij hoe vaak elk stuk voorkomt, en een paar telt `weights[k]` op bij zijn telling, niet 1."),
            Mistake("end-as-text", wrong=_F_WRONG["end-as-text"],
                    en=_CALL + ": the pieces of `<|endoftext|>` itself are counted, as if it were text. Cut the text at every end-of-text first, `text.split(END)`, and count the pieces of every part.",
                    nl=_CALL_NL + ": de stukken van `<|endoftext|>` zelf worden geteld, alsof het tekst is. Knip de tekst eerst bij elke end-of-text, `text.split(END)`, en tel de stukken van elk deel."),
            Mistake("no-pieces", wrong=_F_WRONG["no-pieces"],
                    en=_CALL + ": every part is one piece, so pairs cross from one word into the next. Cut every part into pieces, `re.findall(PATTERN, part)`, as in your train_pre_bpe.",
                    nl=_CALL_NL + ": elk deel is één stuk, dus paren gaan van het ene woord naar het volgende. Knip elk deel in stukken, `re.findall(PATTERN, part)`, zoals in je train_pre_bpe."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: count every piece of every part in `counts_of`; then, in the loop, add `weights[k]` to the count of the pair.",
                      "nl": "Vervang de twee `...`: tel elk stuk van elk deel in `counts_of`; tel dan in de loop `weights[k]` op bij de telling van het paar."},
            "arguments": {"en": "The check calls `{call}`, and the `def` line of your function has another number of names between its parentheses. Keep `(text, n)`.",
                          "nl": "De check roept `{call}` aan, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(text, n)`."},
            "wrong": {"en": _CALL + ". The same merge rules as your train_pre_bpe: every piece of every part counted once, with how often it occurs, and every pair as often as its piece.",
                      "nl": _CALL_NL + ". Dezelfde merge rules als je train_pre_bpe: elk stuk van elk deel één keer geteld, met hoe vaak het voorkomt, en elk paar zo vaak als zijn stuk."},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function count every piece, also a piece that occurs many times, and use the `n` it gets?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Telt je functie elk stuk, ook een stuk dat vaak voorkomt, en gebruikt hij de `n` die hij krijgt?"},
            "slow": {"en": "Your function is still not done after 2 seconds. The check gives it the short texts first, then 300,000 characters of stories with 100 rules: does it go through every piece of the text in every round, instead of every different piece once? Or does a loop keep going forever?",
                     "nl": "Je functie is na 2 seconden nog niet klaar. De check geeft hem eerst de korte teksten, dan 300.000 tekens aan verhalen met 100 regels: gaat hij in elke ronde langs elk stuk van de tekst, in plaats van één keer langs elk verschillend stuk? Of gaat er een loop eindeloos door?"},
            "long": {"en": "Right for the short texts, but with `{call}`, rule {k} is {got}, and it should be {want}{size}. Does your function count every piece, also a piece that occurs many times, and every pair as often as its piece?",
                     "nl": "Goed voor de korte teksten, maar met `{call}` is regel {k} {got}, en dat moet {want} zijn{size}. Telt je functie elk stuk, ook een stuk dat vaak voorkomt, en elk paar zo vaak als zijn stuk?"},
            "size": {"en": "; it gives {n_got} rules, and it should give {n_want}",
                     "nl": "; hij geeft {n_got} regels, en dat moeten er {n_want} zijn"},
            "correct": {"en": "The same merge rules as your train_pre_bpe, from every different piece once. It is in your project now, as `{saved}`.",
                        "nl": "Dezelfde merge rules als je train_pre_bpe, uit elk verschillend stuk één keer. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `train_bpe_fast` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `train_bpe_fast` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "encode-fast": _Box(
        name="encode-fast",
        function="encode_fast",
        start=_E_START,
        solution=_e(),
        visible=[("hi hi<|endoftext|>hi", {(104, 105): 256})],
        hidden=[("", {}), ("<|endoftext|>", {}), ("hi hi hi", {(104, 105): 256, (32, 256): 257}),
                ("é é<|endoftext|>é", {(195, 169): 256}), (_Text(300_000), _Merges()), (_Text(400_000), {})],
        python_values=True,
        save_as=("llm", "fast_encode"),
        first=[_DOTS],
        mistakes=[
            Mistake("only-new", wrong=_E_WRONG["only-new"],
                    en=_CALL + ": the ids of a piece are only added the first time you meet it. Add them every time, also when `known` has them already: the loop over `known[piece]` belongs outside the `if`.",
                    nl=_CALL_NL + ": de ids van een stuk worden alleen toegevoegd als je het voor het eerst tegenkomt. Voeg ze elke keer toe, ook als `known` ze al heeft: de loop over `known[piece]` hoort buiten de `if`."),
            Mistake("nested", wrong=_E_WRONG["nested"],
                    en=_CALL + ": every piece is a list inside `ids`. Add its ids one by one: `for i in known[piece]: ids.append(i)`.",
                    nl=_CALL_NL + ": elk stuk is een lijst binnen `ids`. Voeg zijn ids één voor één toe: `for i in known[piece]: ids.append(i)`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: a piece that is not in `known` yet gets `encode_part(piece, merges)` there; then its ids from `known`, one by one, at the end of `ids`.",
                      "nl": "Vervang de `...`: een stuk dat nog niet in `known` staat, krijgt daar `encode_part(piece, merges)`; dan zijn ids uit `known`, één voor één, achteraan in `ids`."},
            "arguments": {"en": "The check calls `{call}`, and the `def` line of your function has another number of names between its parentheses. Keep `(text, merges)`.",
                          "nl": "De check roept `{call}` aan, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(text, merges)`."},
            "wrong": {"en": _CALL + ". The same ids as your encode: every piece through `encode_part` once, its ids from `known` every time it occurs.",
                      "nl": _CALL_NL + ". Dezelfde ids als je encode: elk stuk één keer door `encode_part`, zijn ids uit `known` elke keer dat het voorkomt."},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function use the `merges` it gets, for every piece?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Gebruikt je functie de `merges` die hij krijgt, voor elk stuk?"},
            "slow": {"en": "Your function is still not done after 2 seconds. The check gives it the short texts first, then long texts of stories: does it encode a piece again every time it occurs, or build a new list for every piece, as `ids = ids + ...` does? Or does a loop keep going forever?",
                     "nl": "Je functie is na 2 seconden nog niet klaar. De check geeft hem eerst de korte teksten, dan lange teksten aan verhalen: encodeert hij een stuk elke keer opnieuw, of maakt hij voor elk stuk een nieuwe lijst, zoals `ids = ids + ...` doet? Of gaat er een loop eindeloos door?"},
            "long": {"en": "Right for the short texts, but with `{call}`, id number {k} is {got}, and it should be {want}{size}. Does a piece that occurs again get its ids from `known`, every time, and end-of-text its own id?",
                     "nl": "Goed voor de korte teksten, maar met `{call}` is id nummer {k} {got}, en dat moet {want} zijn{size}. Krijgt een stuk dat opnieuw voorkomt zijn ids uit `known`, elke keer, en end-of-text zijn eigen id?"},
            "size": {"en": "; it gives {n_got} ids, and it should give {n_want}",
                     "nl": "; hij geeft {n_got} ids, en dat moeten er {n_want} zijn"},
            "correct": {"en": "The same ids as your encode, with every different piece encoded once. It is in your project now, as `{saved}`.",
                        "nl": "Dezelfde ids als je encode, met elk verschillend stuk één keer geëncodeerd. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `encode_fast` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `encode_fast` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("train-bpe-fast", _F_START, "empty", None),
    ("train-bpe-fast", _f(), "correct", None),
    ("train-bpe-fast", _f(add="                counts[pair] = counts.get(pair, 0) + weights[k]\n"), "correct", None),
    ("train-bpe-fast", _f(count=_F_COUNT.replace("            if piece in counts_of:\n                counts_of[piece] = counts_of[piece] + 1\n"
                                                 "            else:\n                counts_of[piece] = 1\n",
                                                 "            counts_of[piece] = counts_of.get(piece, 0) + 1\n")), "correct", None),
    ("train-bpe-fast", ("import re\nfrom llm.byte_bpe import merge_ids\nfrom llm.pre_bpe import PATTERN\n\nEND = \"<|endoftext|>\"\n\n\n"
                        "def train_bpe_fast(text, n):\n    chunks = []\n    for part in text.split(END):\n"
                        "        for piece in re.findall(PATTERN, part):\n            chunks.append(list(piece.encode(\"utf-8\")))\n"
                        "    merges = {}\n    for step in range(n):\n        counts = {}\n        for ids in chunks:\n"
                        "            for i in range(len(ids) - 1):\n                pair = (ids[i], ids[i + 1])\n"
                        "                counts[pair] = counts.get(pair, 0) + 1\n        best = None\n        most = 0\n"
                        "        for pair in counts:\n            if counts[pair] > most:\n                best = pair\n                most = counts[pair]\n"
                        "        merges[best] = 256 + step\n        chunks = [merge_ids(ids, best, 256 + step) for ids in chunks]\n    return merges\n"),
     "slow", None),
    ("train-bpe-fast", _f(add=_F_ADD.replace("+ weights[k]", "+ weights[k] * weights[k]").replace("= weights[k]\n", "= weights[k] * weights[k]\n")),
     "almost", None),   # right on the short texts, whose pieces occur at most twice: told by the first rule that differs
] + [("train-bpe-fast", code, "wrong", key) for key, code in _F_WRONG.items()] + [
    ("encode-fast", _E_START, "empty", None),
    ("encode-fast", _e(), "correct", None),
    ("encode-fast", _e(body="            if piece not in known:\n                known[piece] = encode_part(piece, merges)\n            ids.extend(known[piece])\n"), "correct", None),
    ("encode-fast", _e(body="            ids.extend(encode_part(piece, merges))\n"), "slow", None),
    ("encode-fast", _e(body="            if parts[k] not in known:\n                known[parts[k]] = encode_part(parts[k], merges)\n"
                            "                ids.extend(known[parts[k]])\n"), "almost", None),   # a cache keyed by part: the ids of a part only once
    ("encode-fast", _e(body="            if piece not in known:\n                known[piece] = encode_part(piece, merges)\n            ids = ids + known[piece]\n"), "slow", None),
] + [("encode-fast", code, "wrong", key) for key, code in _E_WRONG.items()]


# --- the demo: your tokenizer and your GPT on the stories ------------------------------------------------------------------

def _saved(module: str) -> bool:
    return (projects.folder() / "llm" / f"{module}.py").exists()


def plan() -> dict:
    """What the demo trains on this computer: the device, the sizes, the mini-batch, the epochs and the parameters."""
    where = T.device()
    p = PLANS["cpu" if where["device"] == "cpu" else "gpu"]
    d, h, n_layers, block = p["sizes"]
    return {**where, **p, "params": n_params(VOCAB, d, n_layers, block)}


def _paths(name: str = "gpt_bpe") -> tuple:
    folder = projects.folder() / "llm"
    return folder / f"{name}.pt", folder / f"{name}.json"


def last_run(name: str = "gpt_bpe") -> dict | None:
    """The last run the demo saved, or None."""
    _, info = _paths(name)
    try:
        return json.loads(info.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


COURSE_GPT = "data/course-gpt-bpe"   # .pt and .json: the course's GPT on tokens, trained on Cosmopedia, see data/README.md


def trained() -> dict:
    """Which trained GPT on tokens a lesson uses: the learner's of this lesson ("yours"), else the course's ("course"),
    with what its JSON holds (sizes, merge rules, losses, story) and its checkpoint."""
    folder = projects.folder() / "llm"
    for source, base in (("yours", folder / "gpt_bpe"), ("course", paths.ROOT / COURSE_GPT)):
        pt, info = base.with_suffix(".pt"), base.with_suffix(".json")
        if pt.exists() and info.exists():
            try:
                meta = json.loads(info.read_text(encoding="utf-8"))
            except ValueError:
                continue
            return {**meta, "source": source, "path": pt}
    return {"source": None}


def _shifted(gpt_loaders, ids: list, block: int, batch: int, where: str, epochs: int, seed: int):
    """As in gpt-train-chars: for every epoch after the first, the training windows start a random number of tokens
    under `block` later, so that the GPT does not see the same windows every epoch; the part it trains on still ends
    where the run's own split does."""
    cut = int(0.9 * len(ids))
    g = torch.Generator().manual_seed(seed)
    offsets = [0] + torch.randint(1, block, (max(epochs - 1, 0),), generator=g).tolist()

    def fresh(epoch: int):
        off = offsets[epoch % len(offsets)]
        n = math.ceil((cut - off) / 0.9)
        while int(0.9 * n) > cut - off:
            n -= 1
        while int(0.9 * n) < cut - off:
            n += 1
        train_loader, _ = gpt_loaders(ids[off:off + n], block, batch, where)
        return train_loader

    return fresh


def _tools(own: bool) -> dict | str:
    """The learner's modules the run needs, by name; "run" when one of them cannot be imported."""
    run = run_code("from llm.fast_bpe import train_bpe_fast\nfrom llm.fast_encode import encode_fast\nfrom llm.decode import decode\n"
                   "from llm.encode import encode\nfrom llm.gpt import GPT\nfrom llm.gpt_loaders import gpt_loaders\nfrom nn.trainer import Trainer")
    if run.error is not None:
        return "run"
    return run.names


def write(model, merges: dict, block: int, seed: int = 0, decode=None, encode=None) -> str:
    """A story of LENGTH tokens from START, drawn with the probabilities of `model` at temperature TAU, as in
    gpt-generate; it ends early at an end-of-text. With the learner's encode and decode when given."""
    import numpy as np
    tools = None if decode and encode else _tools(False)
    decode = decode or tools["decode"]
    encode = encode or tools["encode"]
    end_id = 256 + len(merges)
    rng = np.random.default_rng(seed)
    ids = encode(START, merges)
    model.eval()
    with torch.no_grad():
        for _ in range(LENGTH):
            logits = model(torch.tensor([ids[-block:]]))[0, -1].double()
            p = torch.softmax(logits / TAU, dim=0).numpy()
            j = int(rng.choice(len(p), p=p / p.sum()))
            if j == end_id:
                break
            ids.append(j)
    return decode(ids, merges)


def train(show, seed: int = 0, epochs: int | None = None) -> dict | str | None:
    """Train the learner's tokenizer on the first 90% of the stories, then the learner's GPT on the tokens of all of
    them, as gpt-train-chars does on characters. `show` gets a dict with how far the run is. None when fast_bpe.py or
    fast_encode.py is not in the project yet; "run" when the run stops with an error; "memory" when the GPU runs out
    of memory; "stopped" when the stop button ended the run before its first epoch did; else the run, saved as
    work/projects/llm/gpt_bpe.pt and gpt_bpe.json, unless an earlier run had a lower validation loss."""
    if not (_saved("fast_bpe") and _saved("fast_encode")):
        return None
    tools = _tools(True)
    if isinstance(tools, str):
        return tools
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]
    p = plan()
    where = p["device"]
    epochs = p["epochs"] if epochs is None else epochs
    d, h, n_layers, block = p["sizes"]
    if where == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    checkpoint, info = _paths()
    fresh = checkpoint.with_name("gpt_bpe_new.pt")
    fresh.unlink(missing_ok=True)
    start = time.time()
    try:
        show({"phase": "tokenizer"})
        merges = tools["train_bpe_fast"](split_text()[0], MERGES)
        show({"phase": "encode", "seconds_bpe": time.time() - start})
        ids = tools["encode_fast"](text(), merges)
        if len(merges) != MERGES or max(ids) >= VOCAB:
            return "run"
        torch.manual_seed(seed)
        train_loader, val_loader = tools["gpt_loaders"](ids, block, p["batch"], where)
        model = tools["GPT"](VOCAB, d, h, n_layers, block).to(where)
        trainer = tools["Trainer"](model, train_loader, val_loader, epochs, lr=0.001, warmup=100, max_norm=1.0, path=str(fresh))
        trainer.model = T._OnDevice(trainer.model, where)
    except Exception:
        return "run"
    steps = len(trainer.train_loader)
    windows = len(getattr(trainer.train_loader, "dataset", [])) or steps * p["batch"]
    state = {"step": 0, "shown": 0.0, "start": time.time()}

    def step():
        state["step"] += 1
        now = time.time()
        if now - state["shown"] >= 2 or state["step"] == 1:
            state["shown"] = now
            done = state["step"] - 1
            seconds = now - state["start"]
            left = seconds / done * (epochs * steps - done) if done >= 10 else None
            show({"phase": "train", "epoch": done // steps + 1, "epochs": epochs, "step": done, "steps": epochs * steps,
                  "seconds": seconds, "left": left, "history": list(trainer.history)})

    trainer.train_loader = T._Counted(trainer.train_loader, step, _shifted(tools["gpt_loaders"], ids, block, p["batch"], where, epochs, seed))
    stopped = False
    try:
        trainer.fit()
        T._sync(where)
    except KeyboardInterrupt:   # the stop button of marimo
        stopped = True
    except RuntimeError as e:
        return "memory" if "out of memory" in str(e).lower() else "run"
    except Exception:
        return "run"
    history = [{k: int(v) if k == "epoch" else float(v) for k, v in hist.items()} for hist in trainer.history]
    if not history or not fresh.exists():
        fresh.unlink(missing_ok=True)
        return "stopped"
    cut = int(0.9 * len(ids))
    val_tokens, val_chars = len(ids) - cut, len(tools["decode"](ids[cut:], merges))
    best = min(history, key=lambda hist: hist["val_loss"])
    result = {"device": where, "name": p["name"], "sizes": list(p["sizes"]), "batch": p["batch"], "epochs": epochs,
              "params": p["params"], "seed": seed, "stopped": stopped, "minutes": (time.time() - start) / 60,
              "history": history, "merges": [[a, b] for (a, b) in merges], "stand_ins": stand_ins,
              "tokens": len(ids), "chars": len(text()), "val_tokens": val_tokens, "val_chars": val_chars,
              "best": best["val_loss"], "best_epoch": best["epoch"],
              "best_per_char": per_char(best["val_loss"], val_tokens, val_chars)}
    try:
        model = tools["GPT"](VOCAB, d, h, n_layers, block)
        model.load_state_dict(torch.load(fresh, map_location="cpu", weights_only=True))
        result["story"] = write(model, merges, block, seed, tools["decode"], tools["encode"])
    except Exception:
        result["story"] = None
    old = last_run()
    if old is None or not checkpoint.exists() or result["best"] <= old["best"]:
        fresh.replace(checkpoint)
        info.write_text(json.dumps(result, indent=1), encoding="utf-8")
        result["kept"] = None
    else:
        fresh.unlink()
        result["kept"] = old
    return result
