# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-dutch-models: models made for Dutch, and what their tokenizers make of it. The learner cuts
a word into the pieces of a tokenizer (pieces), works out how many words of a text fit in the context of a model
(words_in_context), and the share of a language in the training text of a model (share). The sort asks which jobs an
encoder is enough for, and which need a decoder.

The tokenizers of the exercises are GPT-2's (tiktoken, from the course's own copy, everything_ai.gpt2_tokenizer), which
is also Fietje's, and RobBERT v2's, made for Dutch (data/robbert-v2-tokenizer.json.gz, read by everything_ai.bpe). The
demo adds those of BERTje, Mistral 7B (under GEITje), EuroLLM and Qwen3, downloaded in the lesson
(everything_ai.dutch_models). The picture of the idea draws on data/dutch-tokenizer-counts.tsv, what the seven
tokenizers make of all 1,997 sentences of NTREX-128 in Dutch and in English, so that it works before the download.
Every number in the text and the pictures is computed by the course (tests/test_nlp_dutch_models.py computes them
again)."""
import ast
import csv
import html
import math

from everything_ai import dutch_models, lesson as lesson_helpers, paths, yours
from everything_ai.check import FunctionBox, Mistake, Sort, kept
from everything_ai.lesson import download_size

LANGS = ("en", "nl")
ENC = dutch_models.gpt2().enc      # GPT-2's tokenizer, a tiktoken Encoding: also Fietje's
ROBBERT = dutch_models.robbert().tok   # RobBERT v2's tokenizer, made for Dutch (everything_ai.bpe.Tokenizer)
COUNTS_FILE = paths.ROOT / "data" / "dutch-tokenizer-counts.tsv"
FILES = dutch_models.FILES   # what the demo downloads
KEYS = dutch_models.KEYS
CONTEXT = 2048   # Fietje's context, Phi-2's: the tokens it reads at once

NAMES = {"gpt2": "GPT-2", "fietje": "Fietje", "mistral": "Mistral 7B", "eurollm": "EuroLLM", "qwen3": "Qwen3",
         "bertje": "BERTje", "robbert": "RobBERT v2"}
USED_BY = {   # the models of the lesson that use each tokenizer, in the picture of the idea
    "en": {"gpt2": "GPT-2", "fietje": "Fietje, Phi-2", "mistral": "GEITje, Mistral 7B", "eurollm": "EuroLLM",
           "qwen3": "Qwen3", "bertje": "BERTje", "robbert": "RobBERT v2"},
    "nl": {"gpt2": "GPT-2", "fietje": "Fietje, Phi-2", "mistral": "GEITje, Mistral 7B", "eurollm": "EuroLLM",
           "qwen3": "Qwen3", "bertje": "BERTje", "robbert": "RobBERT v2"},
}

DOWNLOAD_LINE = {   # what the lesson downloads, for the line under the header
    "en": f"Works on every computer. The tokenizers of BERTje, Mistral 7B, Fietje, EuroLLM and Qwen3 are downloaded once, {download_size(FILES, 'en')}, when you click their button in the lesson; the demo needs them, the rest of the lesson works without them.",
    "nl": f"Werkt op elke computer. De tokenizers van BERTje, Mistral 7B, Fietje, EuroLLM en Qwen3 worden één keer gedownload, {download_size(FILES, 'nl')}, als je in de les op de knop van de download klikt; de demo heeft deze bestanden nodig, de rest van de les werkt zonder.",
}

# The course's own sentences: three sentences in Dutch, and the same in English.
OWN = {
    "nl": ["De gemeente legt volgend jaar een nieuw fietspad aan langs het kanaal.",
           "Bij de bakker op de hoek haal je vanaf zeven uur verse broodjes.",
           "Door de harde wind vertrekken de veerboten naar Texel vandaag later."],
    "en": ["Next year the town will build a new cycle path along the canal.",
           "At the bakery on the corner you can get fresh rolls from seven in the morning.",
           "Because of the strong wind, the ferries to Texel leave later today."],
}
WORDS = ["fietsenstalling", "gemeenteraad", "ziekenhuis", "broodjes", "uitzendbureau", "library", "Saturday"]
TRY_START = "De gemeenteraad keurt vanavond de nieuwe verkeersregels voor fietsers in de binnenstad goed."
TRY_CAP = 400   # characters of a typed text that the demo cuts

# The training text of four models in billions of tokens, as their makers give it.
MIXES = {
    "bertje": {"nl": 2.4},                                # de Vries and others 2019: about 2.4 billion tokens, all Dutch
    "gpt-nl": {"nl": 29, "en": 208, "other": 48},         # gpt-nl.nl, the start of the training: the text, without code
    "fietje": {"phi-2": 1400, "nl": 28},                  # Phi-2's 1.4 trillion, then 28 billion of Dutch (Vanroy 2024)
    "prelude": {"nl": 47.5, "rest": 7767.8},              # OpenEuroLLM's Prelude, its first stage of 7,815 billion
}
MIX_NAMES = {"bertje": "BERTje", "gpt-nl": "GPT-NL", "fietje": "Fietje", "prelude": "OpenEuroLLM Prelude"}


# --- the solutions of the exercises ------------------------------------------------------------------------------

def pieces(word, enc):
    out = []
    for token in enc.encode(" " + word):
        piece = enc.decode([token])
        out.append(piece)
    return out


def words_in_context(context, texts, enc):
    tokens = 0
    words = 0
    for text in texts:
        tokens = tokens + len(enc.encode(text))
        words = words + len(text.split())
    return context * words / tokens


def share(mix, language):
    total = 0
    for name, tokens in mix.items():
        total = total + tokens
    return mix[language] / total


# --- the counts and the tokenizers -------------------------------------------------------------------------------

def counts() -> dict:
    """data/dutch-tokenizer-counts.tsv: per tokenizer key a dictionary of whole numbers, with "size", "nld_words",
    "nld_tokens", "eng_words" and "eng_tokens" (data/tools/make_dutch_tokenizer_counts.py)."""
    def read():
        with open(COUNTS_FILE, encoding="utf-8", newline="") as f:
            return {row["tokenizer"]: {k: int(v) for k, v in row.items() if k != "tokenizer"}
                    for row in csv.DictReader(f, delimiter="\t")}
    return kept("nlp-dutch-models:counts", read)


def per_word(key: str, code: str) -> float:
    """Tokens per word of a tokenizer on all 1,997 sentences of NTREX-128, in Dutch ("nld") or English ("eng")."""
    c = counts()[key]
    return c[f"{code}_tokens"] / c[f"{code}_words"]


def tokenizer(key: str):
    """One of the seven tokenizers, by its key, with `encode` and `shown` (everything_ai.dutch_models), once downloaded."""
    return kept(f"nlp-dutch-models:tokenizer:{key}", lambda: dutch_models.tokenizer(key))


def ready() -> bool:
    """Whether the tokenizers of the demo are downloaded."""
    return dutch_models.ready()


def files_or_stop(button, texts: dict) -> None:
    """Stops the cell with `texts["ask"]` and the button until the tokenizers of the demo are downloaded; a click
    downloads what is missing, with the progress in the cell."""
    lesson_helpers.pins_or_stop(button, texts, FILES)


# --- the parts of the pictures -----------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);margin-top:0.45rem"
_CELL = ("padding:0.1rem 0.4rem;text-align:left;vertical-align:middle;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.82rem")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap"
_DATA = "color:var(--ea-data-text)"
_PRED = "color:var(--ea-pred-text);font-weight:600"
_LOSS = "color:var(--ea-loss-text);font-weight:600"
_BAR = "display:inline-block;height:0.62rem;border-radius:2px;vertical-align:middle"
_CHIP_BASE = ("display:inline-block;margin:0.08rem 0.12rem 0.08rem 0;padding:0 0.25rem;border-radius:3px;"
              "font-family:var(--ea-font-code);font-size:0.78rem;line-height:1.5;white-space:pre")
_CHIP = _CHIP_BASE + ";background:var(--ea-data-wash);color:var(--ea-data-text)"
_CHIP_BYTES = _CHIP_BASE + ";font-style:italic;color:var(--ea-ink-2);background:transparent;border:1px dashed var(--ea-line)"
_CHIP_PRED = _CHIP_BASE + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
_CHIP_LOSS = _CHIP_BASE + ";background:var(--ea-loss-wash);color:var(--ea-loss-text)"
_TEXT_CAP = 120   # characters of a learner's value that a picture shows


def _marked(text_: str) -> str:
    """Escaped text in which `name` is set as code and *word* in italics."""
    out = []
    for k, part in enumerate(html.escape(text_).split("`")):
        if k % 2:
            out.append(f"<code>{part}</code>")
        else:
            out.append("".join(f"<i>{p}</i>" if j % 2 else p for j, p in enumerate(part.split("*"))))
    return "".join(out)


def _note(text_: str) -> str:
    return f'<div style="{_UI}">{_marked(text_)}</div>'


def _label(text_: str) -> str:
    return f'<div style="{_LABEL}">{_marked(text_)}</div>'


def _colors(w: dict, body: str) -> str:
    """The note on the colors under a picture: gray always, blue and purple only when the picture draws them."""
    parts = [w["gray"]]
    if "var(--ea-pred-text)" in body:
        parts.append(w["blue"])
    if "var(--ea-loss-text)" in body:
        parts.append(w["purple"])
    return _note(" ".join(parts))


def _num(n, lang: str) -> str:
    """A whole number with a separator for thousands: 1,234 in English, 1.234 in Dutch."""
    out = f"{round(n):,}"
    return out.replace(",", ".") if lang == "nl" else out


def _dec(x: float, lang: str, digits: int = 2) -> str:
    out = f"{x:,.{digits}f}"
    return out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out


def _pct(x: float, lang: str, digits: int = 1) -> str:
    out = f"{100 * x:.{digits}f}%"
    return out.replace(".", ",") if lang == "nl" else out


def _short(v) -> str:
    """A learner's value as text, cut to _TEXT_CAP characters."""
    try:
        out = v if isinstance(v, str) else repr(v)
    except Exception:   # noqa: BLE001   a value that cannot even be written
        out = "?"
    return out if len(out) <= _TEXT_CAP else out[:_TEXT_CAP] + "…"


def _cell(text_: str, style: str = "", extra: str = "") -> str:
    return f'<td style="{_CELL}{extra};{style}">{text_}</td>'


def _table(rows: list, head: list, numbers=()) -> str:
    cells = "".join(f'<th style="{_HEAD}{_NUMBER if i in numbers else ""}">{html.escape(h)}</th>' for i, h in enumerate(head))
    return (f'<table style="border-collapse:collapse;margin:0.3rem 0 0.2rem"><thead><tr>{cells}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def _bar(value: float, top: float, style: str = "background:var(--ea-data)", width: int = 110) -> str:
    size = 0 if not math.isfinite(value) or top <= 0 else max(0.0, min(value, top)) / top * width
    return f'<span style="{_BAR};width:{size:.0f}px;{style}"></span>'


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error
_SLOW = object()      # what _call gives for a function that ran out of time
PICTURE_SECONDS = 2   # what the learner's function gets for all the calls of one picture together


def _call(make, seconds: float = PICTURE_SECONDS):
    """make() within a time limit, with what it prints thrown away; _SLOW when the time runs out, _STOPPED when it
    stops with an error."""
    import contextlib
    import io
    from everything_ai.check import Timeout, with_time_limit
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(make, max(seconds, 0.05))
    except Timeout:
        return _SLOW
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _empty(w: dict, key: str) -> str:
    """The picture that stays empty, with why: the learner's function stopped, was too slow, or gave something odd."""
    text_ = w[key].format(s=PICTURE_SECONDS, kind=w.get("kind", ""))
    return yours.figure(w["title"], _note(text_), alt=text_)


_WHY_EMPTY = {
    "en": {"stopped": "Your function stops with an error, so there is nothing to draw yet. The check above says why.",
           "slow": "Your function took longer than {s} seconds, so there is nothing to draw yet.",
           "odd": "Your function gives something other than {kind}, so there is nothing to draw yet. The check above says what."},
    "nl": {"stopped": "Je functie stopt met een fout, dus er valt nog niets te tekenen. De check hierboven zegt waarom.",
           "slow": "Je functie deed er langer over dan {s} seconden, dus er valt nog niets te tekenen.",
           "odd": "Je functie geeft iets anders dan {kind}, dus er valt nog niets te tekenen. De check hierboven zegt wat."},
}


class _Odd(Exception):
    """A learner's function gave something the picture cannot draw."""


def _guarded(make):
    try:
        return make()
    except _Odd:
        return None


def _number(x) -> float:
    """A learner's value as a number for a picture; _Odd for anything else, such as a text or a list."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise _Odd
    return float(x)


def _texts(x) -> list:
    """A learner's value as a list of texts for a picture, of at most 60; _Odd for anything else."""
    if not isinstance(x, list) or len(x) > 60 or not all(isinstance(p, str) for p in x):
        raise _Odd
    return x


def _chip(token, style: str = _CHIP) -> str:
    """One token as a chip: a text with a space as ␣, bytes that are whole characters the same way, and bytes that
    hold part of a character as two hex digits each, dashed."""
    if isinstance(token, bytes):
        try:
            token = token.decode("utf-8")
        except UnicodeDecodeError:
            return f'<span style="{_CHIP_BYTES}">{" ".join(f"{x:02x}" for x in token)}</span>'
    return f'<span style="{style}">{html.escape(token[:40].replace(" ", "␣"))}</span>'


def _chips(tokens, style: str = _CHIP) -> str:
    return f'<span style="line-height:1.7">{"".join(_chip(t, style) for t in tokens)}</span>'


# --- the picture of the idea -------------------------------------------------------------------------------------

_IDEA_TEXT = {
    "en": {"title": "What seven tokenizers make of the same 1,997 sentences",
           "head": ["Tokenizer of", "Tokens", "Dutch", "", "English", ""],
           "intro": "All 1,997 sentences of NTREX-128, the same news in Dutch ({nw} words) and in English ({ew} words). Each number is tokens per word: the tokens of all the sentences divided by their words. *Tokens* is the size of the vocabulary.",
           "gray": "Gray: tokens per word of each tokenizer, the darker bar for Dutch and the lighter one for English."},
    "nl": {"title": "Wat zeven tokenizers maken van dezelfde 1.997 zinnen",
           "head": ["Tokenizer van", "Tokens", "Nederlands", "", "Engels", ""],
           "intro": "Alle 1.997 zinnen van NTREX-128, hetzelfde nieuws in het Nederlands ({nw} woorden) en in het Engels ({ew} woorden). Elk getal is tokens per woord: de tokens van alle zinnen gedeeld door de woorden. *Tokens* is de grootte van de vocabulary.",
           "gray": "Grijs: tokens per woord van elke tokenizer, de donkere balk voor het Nederlands en de lichte voor het Engels."},
}


def idea_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang]
    c = counts()
    top = max(max(per_word(k, "nld"), per_word(k, "eng")) for k in KEYS)
    rows = []
    for key in sorted(KEYS, key=lambda k: (per_word(k, "nld"), KEYS.index(k))):
        nl, en = per_word(key, "nld"), per_word(key, "eng")
        rows.append("<tr>" + _cell(html.escape(USED_BY[lang][key])) + _cell(_num(c[key]["size"], lang), _DATA, _NUMBER)
                    + _cell(_dec(nl, lang), _DATA, _NUMBER) + _cell(_bar(nl, top))
                    + _cell(_dec(en, lang), _DATA, _NUMBER) + _cell(_bar(en, top, "background:var(--ea-data);opacity:0.55"))
                    + "</tr>")
    body = _label(w["intro"].format(nw=_num(c["gpt2"]["nld_words"], lang), ew=_num(c["gpt2"]["eng_words"], lang)))
    body += _table(rows, w["head"], (1, 2, 4))
    return yours.figure(w["title"], body + _note(w["gray"]), alt=w["title"])


# --- the pictures of the exercises -------------------------------------------------------------------------------

_PIECES_TEXT = {
    "en": {"title": "Your pieces: five Dutch words and two English ones", "kind": "a list of texts",
           "intro": "Each word with a space before it, as it stands in a sentence, cut by your function with two tokenizers. ␣ is the space.",
           "head": ["Word", "GPT-2, also Fietje", "RobBERT v2"],
           "course": "the course:",
           "gray": "Gray: the words, and the course's pieces where yours differ.",
           "blue": "Blue: your pieces where they are the course's.",
           "purple": "Purple: your pieces where they differ from the course's."},
    "nl": {"title": "Je stukjes: vijf Nederlandse woorden en twee Engelse", "kind": "een lijst van teksten",
           "intro": "Elk woord met een spatie vooraan, zoals het in een zin staat, geknipt door je functie met twee tokenizers. ␣ is de spatie.",
           "head": ["Woord", "GPT-2, ook Fietje", "RobBERT v2"],
           "course": "de course:",
           "gray": "Grijs: de woorden, en de stukjes van de course waar jouw stukjes verschillen.",
           "blue": "Blauw: je stukjes waar deze gelijk zijn aan de stukjes van de course.",
           "purple": "Paars: je stukjes waar deze verschillen van de stukjes van de course."},
}

_CONTEXT_TEXT = {
    "en": {"title": "Your words in a context of 2,048 tokens", "kind": "a number",
           "intro": "Fietje reads 2,048 tokens at once, like Phi-2. How many words fit, by your function on the course's three sentences:",
           "head": ["Tokenizer", "Dutch: yours", "The course", "", "English: yours", "The course"],
           "ntrex": "And by the course's counts of all 1,997 sentences of NTREX-128, for all seven tokenizers:",
           "head2": ["Tokenizer of", "Dutch words", "", "English words"],
           "gray": "Gray: the course's words in 2,048 tokens, and the bars.",
           "blue": "Blue: your words where they are the course's.",
           "purple": "Purple: your words where they differ from the course's."},
    "nl": {"title": "Je woorden in een context van 2.048 tokens", "kind": "een getal",
           "intro": "Fietje leest 2.048 tokens tegelijk, net als Phi-2. Hoeveel woorden passen erin, volgens je functie op de drie zinnen van de course:",
           "head": ["Tokenizer", "Nederlands: van jou", "De course", "", "Engels: van jou", "De course"],
           "ntrex": "En volgens de tellingen van de course van alle 1.997 zinnen van NTREX-128, voor alle zeven tokenizers:",
           "head2": ["Tokenizer van", "Nederlandse woorden", "", "Engelse woorden"],
           "gray": "Grijs: de woorden in 2.048 tokens volgens de course, en de balken.",
           "blue": "Blauw: je woorden waar deze gelijk zijn aan de woorden van de course.",
           "purple": "Paars: je woorden waar deze verschillen van de woorden van de course."},
}

_SHARE_TEXT = {
    "en": {"title": "Your share of Dutch in the training text of four models", "kind": "a number",
           "intro": "The training text of each model in billions of tokens, as its makers give it.",
           "head": ["Model", "Dutch", "All", "Your share", "The course", ""],
           "gray": "Gray: the tokens, the course's shares and the bars.",
           "blue": "Blue: your share where it is the course's.",
           "purple": "Purple: your share where it differs from the course's."},
    "nl": {"title": "Je aandeel Nederlands in de trainingstekst van vier modellen", "kind": "een getal",
           "intro": "De trainingstekst van elk model in miljarden tokens, zoals de makers deze opgeven.",
           "head": ["Model", "Nederlands", "Alles", "Jouw aandeel", "De course", ""],
           "gray": "Grijs: de tokens, de aandelen van de course en de balken.",
           "blue": "Blauw: je aandeel waar dit gelijk is aan het aandeel van de course.",
           "purple": "Paars: je aandeel waar dit verschilt van het aandeel van de course."},
}


def _same(a: float, b: float) -> bool:
    return abs(a - b) < 1e-9


def _yours_cell(mine: float, want: float, text_: str) -> str:
    return _cell(html.escape(text_), _PRED if _same(mine, want) else _LOSS, _NUMBER)


def pieces_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _PIECES_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    encs = (ENC, ROBBERT)
    got = _call(lambda: _guarded(lambda: [[_texts(fn(word, enc)) for enc in encs] for word in WORDS]))
    if got is _SLOW or got is _STOPPED or got is None:
        return _empty(w, "slow" if got is _SLOW else "stopped" if got is _STOPPED else "odd")
    rows = []
    for word, mine in zip(WORDS, got):
        cells = _cell(html.escape(word))
        for enc, m in zip(encs, mine):
            want = pieces(word, enc)
            if m == want:
                cells += _cell(_chips(m, _CHIP_PRED))
            else:
                cells += _cell(_chips(m, _CHIP_LOSS) + f'<div style="{_UI}">{html.escape(w["course"])} {_chips(want)}</div>')
        rows.append("<tr>" + cells + "</tr>")
    body = _label(w["intro"]) + _table(rows, w["head"])
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


def context_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _CONTEXT_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    encs = {"GPT-2": ENC, "RobBERT v2": ROBBERT}
    got = _call(lambda: _guarded(lambda: {(name, k): _number(fn(CONTEXT, list(OWN[k]), enc))
                                         for name, enc in encs.items() for k in OWN}))
    if got is _SLOW or got is _STOPPED or got is None:
        return _empty(w, "slow" if got is _SLOW else "stopped" if got is _STOPPED else "odd")
    c = counts()
    top = max(CONTEXT * c[k][f"{code}_words"] / c[k][f"{code}_tokens"] for k in KEYS for code in ("nld", "eng"))
    rows = []
    for name, enc in encs.items():
        cells = _cell(html.escape(name))
        for k in ("nl", "en"):
            want = words_in_context(CONTEXT, OWN[k], enc)
            cells += _yours_cell(got[(name, k)], want, _dec(got[(name, k)], lang, 1)) + _cell(_dec(want, lang, 1), _DATA, _NUMBER)
            if k == "nl":
                cells += _cell(_bar(want, top))
        rows.append("<tr>" + cells + "</tr>")
    body = _label(w["intro"]) + _table(rows, w["head"], (1, 2, 4, 5))
    rows2 = []
    for key in sorted(KEYS, key=lambda k: (-c[k]["nld_words"] / c[k]["nld_tokens"], KEYS.index(k))):
        nl = CONTEXT * c[key]["nld_words"] / c[key]["nld_tokens"]
        en = CONTEXT * c[key]["eng_words"] / c[key]["eng_tokens"]
        rows2.append("<tr>" + _cell(html.escape(USED_BY[lang][key])) + _cell(_num(nl, lang), _DATA, _NUMBER)
                     + _cell(_bar(nl, top)) + _cell(_num(en, lang), _DATA, _NUMBER) + "</tr>")
    body += _label(w["ntrex"]) + _table(rows2, w["head2"], (1, 3))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


def share_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _SHARE_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    got = _call(lambda: _guarded(lambda: {k: _number(fn(dict(mix), "nl")) for k, mix in MIXES.items()}))
    if got is _SLOW or got is _STOPPED or got is None:
        return _empty(w, "slow" if got is _SLOW else "stopped" if got is _STOPPED else "odd")
    rows = []
    for k, mix in MIXES.items():
        want = share(mix, "nl")
        all_ = sum(mix.values())
        rows.append("<tr>" + _cell(html.escape(MIX_NAMES[k])) + _cell(_dec(mix["nl"], lang, 1), _DATA, _NUMBER)
                    + _cell(_dec(all_, lang, 1), _DATA, _NUMBER) + _yours_cell(got[k], want, _pct(got[k], lang))
                    + _cell(_pct(want, lang), _DATA, _NUMBER) + _cell(_bar(want, 1.0)) + "</tr>")
    body = _label(w["intro"]) + _table(rows, w["head"], (1, 2, 3, 4))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


# --- the demo ----------------------------------------------------------------------------------------------------

_TRY_TEXT = {
    "en": {"title": "Your text, seven tokenizers",
           "line": "{name}: {t} tokens, {pw} per word",
           "none": "Type a text in the box, and the picture cuts it with every tokenizer.",
           "cut": "The box holds more than {n} characters: the picture cuts the first {n}.",
           "gray": "Gray: the tokens of each tokenizer, fewest first; ␣ is a space, and a dashed token holds part of a character, as its bytes in hex. BERTje marks a piece inside a word with ##."},
    "nl": {"title": "Je tekst, zeven tokenizers",
           "line": "{name}: {t} tokens, {pw} per woord",
           "none": "Typ een tekst in het vak, en het plaatje knipt deze tekst met elke tokenizer.",
           "cut": "Het vak bevat meer dan {n} tekens: het plaatje knipt de eerste {n}.",
           "gray": "Grijs: de tokens van elke tokenizer, de minste eerst; ␣ is een spatie, en een gestippeld token bevat een deel van een teken, als de bytes ervan in hex. BERTje zet ## voor een stukje binnen een woord."},
}


def try_picture(lang: str, text_: str = TRY_START) -> str:
    lang = lang if lang in LANGS else "en"
    w = _TRY_TEXT[lang]
    text_ = text_ if isinstance(text_, str) else ""
    body = ""
    if len(text_) > TRY_CAP:
        body += _note(w["cut"].format(n=TRY_CAP))
        text_ = text_[:TRY_CAP]
    words = len(text_.split())
    if not words:
        return yours.figure(w["title"], _note(w["none"]), alt=w["title"])
    shown = {key: tokenizer(key).shown(text_) for key in KEYS}
    for key in sorted(KEYS, key=lambda k: (len(shown[k]), KEYS.index(k))):
        t = len(shown[key])
        body += _label(w["line"].format(name=USED_BY[lang][key], t=_num(t, lang), pw=_dec(t / words, lang)))
        body += f'<div style="margin:0.1rem 0 0.2rem">{_chips(shown[key])}</div>'
    return yours.figure(w["title"], body + _note(w["gray"]), alt=w["title"])


# --- the exercises -----------------------------------------------------------------------------------------------

class _Val:
    """A test value that the check's messages write by its name, such as `C.ENC`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value)


class _Shared(_Val):
    """A test value the function gets itself, not a copy: a tokenizer."""

    def __deepcopy__(self, memo):
        return self.value


def _value(v) -> str:
    """A value the way code writes it, with double quotes."""
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, list):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_value(k)}: {_value(x)}" for k, x in v.items()) + "}"
    return _short(v)


def _written(v) -> _Val:
    return _Val(_value(v), v)


def _show(v):
    if isinstance(v, float):
        return repr(round(v, 4))
    out = _short(v)
    return out if len(out) <= 300 else out[:300] + "…"


def _close(a, b) -> bool:
    if isinstance(a, bool) or not isinstance(a, (int, float)):
        return False
    return abs(a - b) < 1e-9


def _dots_left(got, text_) -> bool:
    """A `...` of the start code left in the box: an Ellipsis in the code, not three dots inside a string."""
    try:
        tree = ast.parse(text_)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Constant) and node.value is Ellipsis for node in ast.walk(tree))


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang de `...` door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

GPT2_VAL = _Shared("C.ENC", ENC)
ROBBERT_VAL = _Shared("C.ROBBERT", ROBBERT)


def _own(k: str, i: int | None = None) -> _Val:
    if i is None:
        return _Val(f'C.OWN["{k}"]', OWN[k])
    return _written([OWN[k][i]])


_PI_HEAD = 'def pieces(word, enc):\n    out = []\n    for token in enc.encode(" " + word):\n'
_PI_END = "        out.append(piece)\n    return out\n"


def _pi(line: str) -> str:
    return _PI_HEAD + line + _PI_END


_PI = _pi("        piece = enc.decode([token])\n")
_PI_START = _pi("        piece = ...\n")
_PI_IDS = _pi("        piece = token\n")
_PI_DIGITS = _pi("        piece = str(token)\n")
_PI_WHOLE = _pi('        piece = enc.decode(enc.encode(" " + word))\n')
_PI_BARE = _pi("        piece = enc.decode(token)\n")

_WC_HEAD = ("def words_in_context(context, texts, enc):\n    tokens = 0\n    words = 0\n    for text in texts:\n"
            "        tokens = tokens + len(enc.encode(text))\n        words = words + len(text.split())\n")


def _wc(line: str) -> str:
    return _WC_HEAD + line


_WC = _wc("    return context * words / tokens\n")
_WC_START = _wc("    return ...\n")
_WC_PER_WORD = _wc("    return context * tokens / words\n")
_WC_NO_CONTEXT = _wc("    return words / tokens\n")
_WC_FLOOR = _wc("    return context * words // tokens\n")
_WC_TOKENS = _wc("    return context / tokens\n")

_SH_HEAD = "def share(mix, language):\n    total = 0\n    for name, tokens in mix.items():\n        total = total + tokens\n"


def _sh(line: str) -> str:
    return _SH_HEAD + line


_SH = _sh("    return mix[language] / total\n")
_SH_START = _sh("    return ...\n")
_SH_LAST = _sh("    return tokens / total\n")
_SH_UPSIDE = _sh("    return total / mix[language]\n")
_SH_PERCENT = _sh("    return mix[language] / total * 100\n")

GPT_NL = MIXES["gpt-nl"]
EUROLLM = {"en": 50, "code": 5, "other": 45}   # EuroLLM's main phase, in percent of its tokens (Martins and others 2024)

REQUIRED = ["pieces", "words_in_context", "share", "fits"]

EXERCISES = {
    "pieces": FunctionBox(
        name="pieces",
        picture=pieces_picture,
        function="pieces",
        start=_PI_START,
        solution=_PI,
        show=_show,
        visible=[(_written("fietsenstalling"), GPT2_VAL)],
        hidden=[(_written("fietsenstalling"), ROBBERT_VAL), (_written("ziekenhuis"), GPT2_VAL),
                (_written("ziekenhuis"), ROBBERT_VAL), (_written("Saturday"), GPT2_VAL), (_written("Saturday"), ROBBERT_VAL),
                (_written("uitzendbureau"), GPT2_VAL), (_written("the"), ROBBERT_VAL)],
        first=[_DOTS],
        mistakes=[
            Mistake("ids", wrong=_PI_IDS,
                    en="`{call}` gives {got}, and it should give {want}: those are the ids of the tokens, numbers, and not their text. Which method of `enc` turns ids back into text?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: dat zijn de ids van de tokens, getallen, en niet de tekst ervan. Welke methode van `enc` maakt van ids weer tekst?"),
            Mistake("digits", wrong=_PI_DIGITS,
                    en="`{call}` gives {got}, and it should give {want}: `str` writes the id of a token in digits, and not the text it stands for. Which method of `enc` turns ids back into text?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `str` schrijft het id van een token in cijfers, en niet de tekst waar het token voor staat. Welke methode van `enc` maakt van ids weer tekst?"),
            Mistake("whole", wrong=_PI_WHOLE,
                    en="`{call}` gives {got}, and it should give {want}: every piece is the whole word, because you decode all its tokens at once. How do you decode one token, `token`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elk stukje is het hele woord, omdat je alle tokens tegelijk decodeert. Hoe decodeer je één token, `token`?"),
            Mistake("bare", error=lambda e: isinstance(e, TypeError) and ("not iterable" in str(e) or "Sequence" in str(e)),
                    en="With `{call}`, your function stops with a `TypeError`: `enc.decode` takes a list of ids, and gets one id. How do you make a list with only `token` in it?",
                    nl="Met `{call}` stopt je functie met een `TypeError`: `enc.decode` verwacht een lijst met ids, en krijgt één id. Hoe maak je een lijst met alleen `token` erin?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: the text of `token`, with `enc.decode`.",
                      "nl": "Vervang de `...`: de tekst van `token`, met `enc.decode`."},
            "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has another number of names between its parentheses. Keep `(word, enc)`.",
                          "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(word, enc)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does `piece` get the text of one token, `token`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Krijgt `piece` de tekst van één token, `token`?"},
            "almost": {"en": "Right for *fietsenstalling* with GPT-2's tokenizer. But `{call}` gives {got}, and it should give {want}. Does your function work for every word and both tokenizers?",
                       "nl": "Goed voor *fietsenstalling* met de tokenizer van GPT-2. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie voor elk woord en beide tokenizers?"},
            "correct": {"en": "Right: GPT-2 cuts *fietsenstalling* into five pieces, right through *fietsen* and *stalling*, and RobBERT into exactly those two words. What does RobBERT make of *Saturday*?",
                        "nl": "Goed: GPT-2 knipt *fietsenstalling* in vijf stukjes, dwars door *fietsen* en *stalling* heen, en RobBERT in precies deze twee woorden. Wat maakt RobBERT van *Saturday*?"},
        },
    ),
    "words_in_context": FunctionBox(
        name="words_in_context",
        picture=context_picture,
        function="words_in_context",
        start=_WC_START,
        solution=_WC,
        show=_show,
        compare=_close,
        visible=[(_Val("2048", 2048), _own("nl", 0), GPT2_VAL)],
        hidden=[(_Val("2048", 2048), _own("nl"), GPT2_VAL), (_Val("2048", 2048), _own("nl"), ROBBERT_VAL),
                (_Val("2048", 2048), _own("en"), GPT2_VAL), (_Val("512", 512), _own("en"), ROBBERT_VAL),
                (_Val("8192", 8192), _own("nl", 1), ROBBERT_VAL)],
        first=[_DOTS],
        mistakes=[
            Mistake("per-word", wrong=_WC_PER_WORD,
                    en="`{call}` gives {got}, and it should give {want}: you multiply `context` by the tokens per word, and that gives more words for a tokenizer that cuts more. How much of a word does one token hold?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je vermenigvuldigt `context` met de tokens per woord, en zo geeft een tokenizer die meer knipt meer woorden. Hoeveel van een woord bevat één token?"),
            Mistake("no-context", wrong=_WC_NO_CONTEXT,
                    en="`{call}` gives {got}, and it should give {want}: that is how much of a word one token holds. How many tokens does the context hold?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: dat is hoeveel van een woord één token bevat. Hoeveel tokens passen in de context?"),
            Mistake("floor", wrong=_WC_FLOOR,
                    en="`{call}` gives {got}, and it should give {want}: `//` rounds down to a whole number. Can you divide with `/`, so the check sees the exact number?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `//` rondt af naar beneden, op een heel getal. Kun je delen met `/`, zodat de check het precieze getal ziet?"),
            Mistake("tokens-only", wrong=_WC_TOKENS,
                    en="`{call}` gives {got}, and it should give {want}: you divide `context` by the tokens of the texts, and the words play no part. How many words does one token hold?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je deelt `context` door de tokens van de teksten, en de woorden doen niet mee. Hoeveel woorden bevat één token?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: the words that fit in `context` tokens, worked out from `words` and `tokens`.",
                      "nl": "Vervang de `...`: de woorden die in `context` tokens passen, uitgerekend met `words` en `tokens`."},
            "arguments": {"en": "The check calls `{call}`, with three values, and the `def` line of your function has another number of names between its parentheses. Keep `(context, texts, enc)`.",
                          "nl": "De check roept `{call}` aan, met drie waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(context, texts, enc)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your last line multiply `context` by the words per token?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Vermenigvuldigt je laatste regel `context` met de woorden per token?"},
            "almost": {"en": "Right for the Dutch sentence about the cycle path. But `{call}` gives {got}, and it should give {want}. Does your function work for every list of texts and every context?",
                       "nl": "Goed voor de Nederlandse zin over het fietspad. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie voor elke lijst van teksten en elke context?"},
            "correct": {"en": "Right: GPT-2 makes 29 tokens of the 12 Dutch words, so 2,048 tokens hold about 847 words of such Dutch. How many words of the same Dutch fit with RobBERT's tokenizer?",
                        "nl": "Goed: GPT-2 maakt 29 tokens van de 12 Nederlandse woorden, dus in 2.048 tokens passen ongeveer 847 woorden van zulk Nederlands. Hoeveel woorden van hetzelfde Nederlands passen erin met de tokenizer van RobBERT?"},
        },
    ),
    "share": FunctionBox(
        name="share",
        picture=share_picture,
        function="share",
        start=_SH_START,
        solution=_SH,
        show=_show,
        compare=_close,
        visible=[(_written(GPT_NL), _written("nl"))],
        hidden=[(_written(GPT_NL), _written("en")), (_written(GPT_NL), _written("other")),
                (_written(MIXES["fietje"]), _written("nl")), (_written(MIXES["bertje"]), _written("nl")),
                (_written(MIXES["prelude"]), _written("nl")), (_written(EUROLLM), _written("en"))],
        first=[_DOTS],
        mistakes=[
            Mistake("last", wrong=_SH_LAST,
                    en="`{call}` gives {got}, and it should give {want}: after the loop, `tokens` holds the tokens of the last language in `mix`. Where in `mix` are the tokens of `language`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: na de loop bevat `tokens` de tokens van de laatste taal in `mix`. Waar in `mix` staan de tokens van `language`?"),
            Mistake("upside-down", wrong=_SH_UPSIDE,
                    en="`{call}` gives {got}, and it should give {want}: you divide all the tokens by the tokens of the language, and a share is never more than 1. Which number goes on top?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je deelt alle tokens door de tokens van de taal, en een aandeel is nooit meer dan 1. Welk getal hoort boven de streep?"),
            Mistake("percent", wrong=_SH_PERCENT,
                    en="`{call}` gives {got}, and it should give {want}: that is a percentage. Can you give the share as a number from 0 to 1, without the `* 100`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: dat is een percentage. Kun je het aandeel geven als getal van 0 tot 1, zonder de `* 100`?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: the tokens of `language` in `mix`, divided by `total`.",
                      "nl": "Vervang de `...`: de tokens van `language` in `mix`, gedeeld door `total`."},
            "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has another number of names between its parentheses. Keep `(mix, language)`.",
                          "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(mix, language)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your last line divide the tokens of `language` by `total`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Deelt je laatste regel de tokens van `language` door `total`?"},
            "almost": {"en": "Right for Dutch in GPT-NL. But `{call}` gives {got}, and it should give {want}. Does your function work for every language and every mix?",
                       "nl": "Goed voor het Nederlands in GPT-NL. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie voor elke taal en elke mix?"},
            "correct": {"en": "Right: Dutch is 29 of the 285 billion tokens of text of GPT-NL, 10.2%, and English 73.0%. What share does Dutch have in Fietje, with the 1.4 trillion tokens of Phi-2 before it?",
                        "nl": "Goed: het Nederlands is 29 van de 285 miljard tokens tekst van GPT-NL, 10,2%, en het Engels 73,0%. Welk aandeel heeft het Nederlands in Fietje, als je de 1,4 biljoen tokens van Phi-2 meetelt?"},
        },
    ),
    "fits": Sort(
        name="fits",
        groups={
            "encoder": {"en": "An encoder is enough", "nl": "Een encoder volstaat"},
            "decoder": {"en": "Needs a decoder", "nl": "Vraagt om een decoder"},
        },
        items={
            "reviews": {"en": "**Reviews.** Sort 50,000 Dutch reviews of books into positive and negative, quickly and cheaply.",
                        "nl": "**Recensies.** Deel 50.000 Nederlandse recensies van boeken snel en goedkoop in als positief of negatief."},
            "reply": {"en": "**A reply.** Write an answer in Dutch to an email from a customer.",
                      "nl": "**Een antwoord.** Schrijf in het Nederlands een antwoord op een e-mail van een klant."},
            "names": {"en": "**Names.** Mark the names of places and organizations in Dutch news.",
                      "nl": "**Namen.** Markeer de namen van plaatsen en organisaties in Nederlands nieuws."},
            "simple": {"en": "**Simple Dutch.** Rewrite a letter from the tax office in plain, simple Dutch.",
                       "nl": "**Eenvoudig Nederlands.** Herschrijf een brief van de Belastingdienst in gewoon, eenvoudig Nederlands."},
            "diedat": {"en": "**die or dat.** Check whether a Dutch text puts *die* and *dat* in the right places.",
                       "nl": "**Die of dat.** Controleer of een Nederlandse tekst *die* en *dat* op de goede plekken zet."},
            "summary": {"en": "**A short version.** Summarize a twenty-page report in five new sentences.",
                        "nl": "**Een korte versie.** Vat een rapport van twintig pagina's samen in vijf nieuwe zinnen."},
        },
        solution={"reviews": "encoder", "reply": "decoder", "names": "encoder", "simple": "decoder", "diedat": "encoder",
                  "summary": "decoder"},
        nudge={
            "reviews": {"en": "What does the job give back for each review: new text, or one of two labels?",
                        "nl": "Wat geeft de taak terug voor elke recensie: nieuwe tekst, of een van twee labels?"},
            "reply": {"en": "An encoder reads a whole text and gives labels. Can it write an email?",
                      "nl": "Een encoder leest een hele tekst en geeft labels. Kan een encoder een e-mail schrijven?"},
            "names": {"en": "What did BERTje and RobBERT give for each word of the news of CoNLL-2002?",
                      "nl": "Wat gaven BERTje en RobBERT voor elk woord van het nieuws van CoNLL-2002?"},
            "simple": {"en": "Does the job give back labels, or new sentences?",
                       "nl": "Geeft de taak labels terug, of nieuwe zinnen?"},
            "diedat": {"en": "What did RobBERT do with a [MASK] at the place of *die* or *dat*, without any fine-tuning?",
                       "nl": "Wat deed RobBERT met een [MASK] op de plek van *die* of *dat*, zonder fine-tuning?"},
            "summary": {"en": "Do the five sentences already stand in the report, or does a model have to write them?",
                        "nl": "Staan de vijf zinnen al in het rapport, of moet een model de zinnen schrijven?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the groups again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de groepen opnieuw uit."},
            "empty": {"en": "Sort each job: is an encoder enough, or does it need a decoder?",
                      "nl": "Deel elke taak in: volstaat een encoder, of vraagt de taak om een decoder?"},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Je indeling klopt tot nu toe. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} belongs in the other group.",
                      "nl": "{item} hoort bij de andere groep."},
            "correct": {"en": "All {n} are right. A job that reads a text and gives labels, a label per text or per word, or a word at a [MASK], is enough for an encoder such as RobBERT: about 117 million parameters, fast and cheap, after fine-tuning on labeled examples. A job that writes new text needs a decoder, such as Fietje, EuroLLM or GPT-NL, with billions of parameters. A decoder can label too, if you ask it, but at a much higher price per text.",
                        "nl": "Alle {n} kloppen. Leest een taak een tekst en geeft de taak labels, een label per tekst of per woord, of een woord op een [MASK], dan volstaat een encoder zoals RobBERT: ongeveer 117 miljoen parameters, snel en goedkoop, na fine-tuning op gelabelde voorbeelden. Schrijft een taak nieuwe tekst, dan vraagt de taak om een decoder, zoals Fietje, EuroLLM of GPT-NL, met miljarden parameters. Een decoder kan ook labels geven als je erom vraagt, maar tegen een veel hogere prijs per tekst."},
        },
    ),
}

CASES = [
    ("pieces", _PI_START, "empty", None),
    ("pieces", _PI, "correct", None),
    ("pieces", _pi("        piece = enc.decode([...])\n"), "wrong", "dots-left"),
    ("pieces", _PI_IDS, "wrong", "ids"),
    ("pieces", _PI_DIGITS, "wrong", "digits"),
    ("pieces", _PI_WHOLE, "wrong", "whole"),
    ("pieces", _PI_BARE, "error", "bare"),
    ("pieces", 'def pieces(word, enc):\n    return [enc.decode([t]) for t in enc.encode(" " + word)]\n', "correct", None),
    ("pieces", _pi("        piece = enc.decode([token]).strip()\n"), "wrong", None),
    ("pieces", 'def pieces(word, enc):\n    out = []\n    for token in enc.encode(" fietsenstalling"):\n        out.append(enc.decode([token]))\n    return out\n',
     "almost", None),
    ("words_in_context", _WC_START, "empty", None),
    ("words_in_context", _WC, "correct", None),
    ("words_in_context", _wc("    return context * ...\n"), "wrong", "dots-left"),
    ("words_in_context", _WC_PER_WORD, "wrong", "per-word"),
    ("words_in_context", _WC_NO_CONTEXT, "wrong", "no-context"),
    ("words_in_context", _WC_FLOOR, "wrong", "floor"),
    ("words_in_context", _WC_TOKENS, "wrong", "tokens-only"),
    ("words_in_context", _wc("    return context / tokens * words\n"), "correct", None),
    ("words_in_context", _wc("    return words * (context / tokens)\n"), "correct", None),
    ("words_in_context", _wc("    return context * words / (tokens + 1)\n"), "wrong", None),
    ("words_in_context", _wc("    return 2048 * words / tokens\n"), "almost", None),
    ("share", _SH_START, "empty", None),
    ("share", _SH, "correct", None),
    ("share", _sh("    return mix[...] / total\n"), "wrong", "dots-left"),
    ("share", _SH_LAST, "wrong", "last"),
    ("share", _SH_UPSIDE, "wrong", "upside-down"),
    ("share", _SH_PERCENT, "wrong", "percent"),
    ("share", "def share(mix, language):\n    return mix[language] / sum(mix.values())\n", "correct", None),
    ("share", _sh("    return mix[language] / (total + 1)\n"), "wrong", None),
    ("share", _sh('    return mix["nl"] / total\n'), "almost", None),
    ("fits", {"reviews": "encoder", "reply": "decoder", "names": "encoder", "simple": "decoder", "diedat": "encoder",
              "summary": "decoder"}, "correct", None),
    ("fits", {}, "empty", None),
    ("fits", {"reviews": "encoder", "reply": "decoder"}, "wrong", "unfinished"),
    ("fits", {"reviews": "decoder"}, "wrong", "wrong:reviews"),
]
