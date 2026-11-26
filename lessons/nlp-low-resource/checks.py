# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-low-resource: what the same text costs in tokens in different languages. The lesson
measures it in three ways, each over sentences that say the same in every language: the premium of Petrov and others
(NeurIPS 2023), the tokens of a language divided by the tokens of the same sentences in English (premium); the tokens
per byte of UTF-8, which is 1 when a byte-level tokenizer learned no piece of a language at all (per_byte); and the
proportion of continued words of Rust and others (ACL 2021), the share of the words that the tokenizer cuts into more
than one token (continued). The sort asks what helps a language with little data and what falls short.

The tokenizers are GPT-2's (tiktoken, from the course's own copy, everything_ai.gpt2_tokenizer) and Qwen3's
(everything_ai.qwen3, downloaded in the lesson). The sentences of the exercises are the course's own, in seven
languages. The sentences of the demo are the first 100 of NTREX-128 (Federmann, Kocmi and Xin 2022) in 48 languages,
downloaded in the lesson (everything_ai.ntrex); the picture of the idea draws on data/ntrex-token-counts.tsv, the
counts of those sentences, so that it works before the download. Every number in the text and the pictures is
computed by the course (tests/test_nlp_low_resource.py computes them again)."""
import ast
import csv
import html
import math

from everything_ai import gpt2_tokenizer, hub, lesson as lesson_helpers, ntrex, paths, yours
from everything_ai.check import FunctionBox, Mistake, Sort, kept
from everything_ai.lesson import download_size

gpt2_tokenizer.offline()   # before any tiktoken.get_encoding("gpt2"), also in the lesson's own cells

import tiktoken  # noqa: E402  (after offline(): tiktoken reads its cache folder when it loads a file)

LANGS = ("en", "nl")
ENC = tiktoken.get_encoding("gpt2")
COUNTS_FILE = paths.ROOT / "data" / "ntrex-token-counts.tsv"
QWEN_REPO = "Qwen/Qwen3-0.6B-Base"   # everything_ai.qwen3.REPO, without loading PyTorch, which qwen3 imports
QWEN_FILES = [(QWEN_REPO, "vocab.json"), (QWEN_REPO, "tokenizer_config.json")]
FILES = ntrex.FILES + QWEN_FILES   # what the demo downloads
TOKENIZERS = ("gpt2", "qwen3")
TRY_START = "tam"   # the language the demo starts with: the one of the guess

DOWNLOAD_LINE = {   # what the lesson downloads, for the line under the header
    "en": f"Works on every computer. The first 100 sentences of NTREX-128 in 48 languages and the tokenizer of Qwen3 are downloaded once, {download_size(FILES, 'en')}, when you click their button in the lesson; the demo needs them, the rest of the lesson works without them.",
    "nl": f"Werkt op elke computer. De eerste 100 zinnen van NTREX-128 in 48 talen en de tokenizer van Qwen3 worden één keer gedownload, {download_size(FILES, 'nl')}, als je in de les op de knop van de download klikt; de demo heeft deze bestanden nodig, de rest van de les werkt zonder.",
}

# The course's own sentences: three sentences that say the same in seven languages.
OWN = {
    "en": ["The library opens at nine on Saturday.", "The train to Utrecht leaves in ten minutes.",
           "Most people in the world speak more than one language."],
    "nl": ["De bibliotheek gaat zaterdag om negen uur open.", "De trein naar Utrecht vertrekt over tien minuten.",
           "De meeste mensen op de wereld spreken meer dan één taal."],
    "de": ["Die Bibliothek öffnet am Samstag um neun Uhr.", "Der Zug nach Utrecht fährt in zehn Minuten ab.",
           "Die meisten Menschen auf der Welt sprechen mehr als eine Sprache."],
    "fr": ["La bibliothèque ouvre samedi à neuf heures.", "Le train pour Utrecht part dans dix minutes.",
           "La plupart des gens dans le monde parlent plus d'une langue."],
    "ru": ["Библиотека открывается в субботу в девять часов.", "Поезд в Утрехт отправляется через десять минут.",
           "Большинство людей в мире говорят более чем на одном языке."],
    "el": ["Η βιβλιοθήκη ανοίγει το Σάββατο στις εννέα.", "Το τρένο για την Ουτρέχτη φεύγει σε δέκα λεπτά.",
           "Οι περισσότεροι άνθρωποι στον κόσμο μιλούν περισσότερες από μία γλώσσες."],
    "ja": ["図書館は土曜日の九時に開きます。", "ユトレヒト行きの電車は十分後に出発します。", "世界のほとんどの人は二つ以上の言語を話します。"],
}
OWN_NAMES = {
    "en": {"en": "English", "nl": "Dutch", "de": "German", "fr": "French", "ru": "Russian", "el": "Greek", "ja": "Japanese"},
    "nl": {"en": "Engels", "nl": "Nederlands", "de": "Duits", "fr": "Frans", "ru": "Russisch", "el": "Grieks", "ja": "Japans"},
}


# --- the solutions of the exercises ------------------------------------------------------------------------------

def premium(texts, english, enc):
    tokens = 0
    for text in texts:
        tokens = tokens + len(enc.encode(text))
    english_tokens = 0
    for text in english:
        english_tokens = english_tokens + len(enc.encode(text))
    return tokens / english_tokens


def per_byte(texts, enc):
    tokens = 0
    n_bytes = 0
    for text in texts:
        tokens = tokens + len(enc.encode(text))
        n_bytes = n_bytes + len(text.encode("utf-8"))
    return tokens / n_bytes


def continued(texts, enc):
    words = 0
    count = 0
    for text in texts:
        for word in text.split():
            words = words + 1
            if len(enc.encode(" " + word)) > 1:
                count = count + 1
    return count / words


# --- the tokenizers and the counts of NTREX ----------------------------------------------------------------------

class QwenTokenizer:
    """Qwen3's tokenizer with the encode of tiktoken, as a lesson hands it to a learner's function as `enc`: it puts
    the text in Unicode's form NFC first, as Qwen's own code does (everything_ai.qwen3.encode)."""

    def __init__(self):
        from everything_ai import qwen3
        self.enc, self._encode = qwen3.tokenizer(), qwen3.encode

    def encode(self, text: str) -> list:
        return self._encode(self.enc, text)

    def decode_single_token_bytes(self, token: int) -> bytes:
        return self.enc.decode_single_token_bytes(token)


def tokenizer(name: str):
    """GPT-2's tokenizer ("gpt2") or Qwen3's ("qwen3"), once downloaded."""
    if name == "gpt2":
        return ENC
    return kept("nlp-low-resource:qwen3", QwenTokenizer)


def counts() -> dict:
    """data/ntrex-token-counts.tsv: per language code a dictionary of whole numbers, with "joshi", "words",
    "characters", "bytes", "gpt2", "qwen3", "gpt2_continued" and "qwen3_continued" (data/tools/make_ntrex_counts.py)."""
    def read():
        with open(COUNTS_FILE, encoding="utf-8", newline="") as f:
            return {row["code"]: {k: int(v) for k, v in row.items() if k != "code"} for row in csv.DictReader(f, delimiter="\t")}
    return kept("nlp-low-resource:counts", read)


def course_premium(code: str, tok: str) -> float:
    c = counts()
    return c[code][tok] / c["eng"][tok]


def course_per_byte(code: str, tok: str) -> float:
    c = counts()[code]
    return c[tok] / c["bytes"]


def course_continued(code: str, tok: str) -> float:
    c = counts()[code]
    return c[f"{tok}_continued"] / c["words"]


def course_fertility(code: str, tok: str) -> float:
    c = counts()[code]
    return c[tok] / c["words"]


def ready() -> bool:
    """Whether the demo's files are downloaded: the sentences of NTREX and Qwen3's tokenizer."""
    return all(hub.pinned(*pin) is not None for pin in FILES)


def files_or_stop(button, texts: dict, lang: str) -> dict:
    """The languages of the demo (try_options), once the sentences of NTREX and Qwen3's tokenizer are downloaded. Until
    then the cell stops with `texts["ask"]` and the button; a click downloads what is missing, with the progress in the
    cell."""
    lesson_helpers.pins_or_stop(button, texts, FILES)
    return try_options(lang)


# --- the parts of the pictures -----------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);margin-top:0.45rem"
_CELL = ("padding:0.1rem 0.4rem;text-align:left;vertical-align:middle;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.82rem")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap"
_DATA = "color:var(--ea-data-text)"
_FAINT = "color:var(--ea-ink-2)"
_PRED = "color:var(--ea-pred-text);font-weight:600"
_LOSS = "color:var(--ea-loss-text);font-weight:600"
_BOLD = "font-weight:600"
_BAR = "display:inline-block;height:0.62rem;border-radius:2px;vertical-align:middle"
_CHIP = ("display:inline-block;margin:0.08rem 0.12rem 0.08rem 0;padding:0 0.25rem;border-radius:3px;"
         "font-family:var(--ea-font-code);font-size:0.78rem;line-height:1.5;white-space:pre;"
         "background:var(--ea-data-wash);color:var(--ea-data-text)")
_CHIP_BYTES = _CHIP + ";font-style:italic;color:var(--ea-ink-2);background:transparent;border:1px dashed var(--ea-line)"
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
    out = f"{x:.{digits}f}"
    return out.replace(".", ",") if lang == "nl" else out


def _pct(x: float, lang: str) -> str:
    out = f"{100 * x:.1f}%"
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


def _bar(value: float, top: float, style: str = "background:var(--ea-data)", width: int = 120) -> str:
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


def _number(x) -> float:
    """A learner's value as a number for a picture; _Odd for anything else, such as a text or a list."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise _Odd
    return float(x)


def _mine(fn, *args) -> float:
    return _number(fn(*args))


def _pieces(enc, text_: str) -> str:
    """The tokens of a text as chips: a token that is whole characters shows them, with a space as ␣; a token that
    holds part of a character shows its bytes, as two hex digits each."""
    out = []
    for t in enc.encode(text_):
        b = enc.decode_single_token_bytes(t)
        try:
            out.append(f'<span style="{_CHIP}">{html.escape(b.decode("utf-8").replace(" ", "␣"))}</span>')
        except UnicodeDecodeError:
            out.append(f'<span style="{_CHIP_BYTES}">{" ".join(f"{x:02x}" for x in b)}</span>')
    return f'<div style="margin:0.15rem 0 0.25rem;line-height:1.7">{"".join(out)}</div>'


def _sentence(text_: str) -> str:
    return f'<div style="font-family:var(--ea-font-prose);font-size:0.9rem;margin:0.2rem 0 0">{html.escape(text_)}</div>'


# --- the picture of the idea -------------------------------------------------------------------------------------

_IDEA_TEXT = {
    "en": {"title": "What the same 100 sentences cost, against English",
           "head": ["Language", "Class", "GPT-2", "", "Qwen3", ""],
           "intro": "The first 100 sentences of NTREX-128, the same news in every language. English costs GPT-2 {g} tokens and Qwen3 {q}. Each bar is the premium: the tokens of a language divided by the tokens of English. *Class* is the class of Joshi and others, from 1, little data, to 5, the most.",
           "gray": "Gray: the premium of each language for each tokenizer; English and Dutch in bold."},
    "nl": {"title": "Wat dezelfde 100 zinnen kosten, tegenover het Engels",
           "head": ["Taal", "Groep", "GPT-2", "", "Qwen3", ""],
           "intro": "De eerste 100 zinnen van NTREX-128, hetzelfde nieuws in elke taal. Het Engels kost GPT-2 {g} tokens en Qwen3 {q}. Elke balk is de premium: de tokens van een taal gedeeld door de tokens van het Engels. *Groep* is de groep van Joshi en anderen, van 1, weinig data, tot 5, de meeste.",
           "gray": "Grijs: de premium van elke taal voor elke tokenizer; het Engels en het Nederlands vet."},
}


def idea_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang]
    c = counts()
    top = max(course_premium(code, "gpt2") for code in c)
    order = sorted(c, key=lambda code: (code != "eng", course_premium(code, "gpt2")))
    rows = []
    for code in order:
        style = _BOLD if code in ("eng", "nld") else ""
        g, q = course_premium(code, "gpt2"), course_premium(code, "qwen3")
        rows.append("<tr>" + _cell(html.escape(ntrex.name(code, lang)), style) + _cell(str(c[code]["joshi"]), _FAINT, _NUMBER)
                    + _cell(_dec(g, lang), _DATA, _NUMBER) + _cell(_bar(g, top))
                    + _cell(_dec(q, lang), _DATA, _NUMBER) + _cell(_bar(q, top, "background:var(--ea-data);opacity:0.55"))
                    + "</tr>")
    body = _label(w["intro"].format(g=_num(c["eng"]["gpt2"], lang), q=_num(c["eng"]["qwen3"], lang)))
    body += _table(rows, w["head"], (1, 2, 4))
    return yours.figure(w["title"], body + _note(w["gray"]), alt=w["title"])


# --- the pictures of the exercises -------------------------------------------------------------------------------

_PREMIUM_TEXT = {
    "en": {"title": "Your premium: the course's sentences in seven languages", "kind": "a number",
           "intro": "The course's three sentences, with GPT-2's tokenizer. English costs {n} tokens.",
           "head": ["Language", "Tokens", "Your premium", "The course", ""],
           "gray": "Gray: the tokens, the course's premium and the bars.",
           "blue": "Blue: your premium where it is the course's.",
           "purple": "Purple: your premium where it differs from the course's."},
    "nl": {"title": "Je premium: de zinnen van de course in zeven talen", "kind": "een getal",
           "intro": "De drie zinnen van de course, met de tokenizer van GPT-2. Het Engels kost {n} tokens.",
           "head": ["Taal", "Tokens", "Jouw premium", "De course", ""],
           "gray": "Grijs: de tokens, de premium van de course en de balken.",
           "blue": "Blauw: je premium waar deze gelijk is aan de premium van de course.",
           "purple": "Paars: je premium waar deze verschilt van de premium van de course."},
}

_PER_BYTE_TEXT = {
    "en": {"title": "Your tokens per byte: the course's sentences in seven languages", "kind": "a number",
           "intro": "The course's three sentences, with GPT-2's tokenizer.",
           "head": ["Language", "Bytes", "Tokens", "Your tokens per byte", "The course"],
           "pieces": "The tokens of the Greek sentence about the library. A dashed token holds part of a character: its bytes, in hex.",
           "gray": "Gray: the bytes, the tokens, the course's values and the tokens of the sentence.",
           "blue": "Blue: your values where they are the course's.",
           "purple": "Purple: your values where they differ from the course's."},
    "nl": {"title": "Je tokens per byte: de zinnen van de course in zeven talen", "kind": "een getal",
           "intro": "De drie zinnen van de course, met de tokenizer van GPT-2.",
           "head": ["Taal", "Bytes", "Tokens", "Jouw tokens per byte", "De course"],
           "pieces": "De tokens van de Griekse zin over de bibliotheek. Een gestippeld token bevat een deel van een teken: de bytes ervan, in hex.",
           "gray": "Grijs: de bytes, de tokens, de waarden van de course en de tokens van de zin.",
           "blue": "Blauw: je waarden waar deze gelijk zijn aan de waarden van de course.",
           "purple": "Paars: je waarden waar deze verschillen van de waarden van de course."},
}

_CONTINUED_TEXT = {
    "en": {"title": "Your continued words: the course's sentences in seven languages", "kind": "a number",
           "intro": "The course's three sentences, with GPT-2's tokenizer.",
           "head": ["Language", "Words", "Your continued words", "The course"],
           "pieces": "The words of the Dutch sentence about the library, each with a space before it, cut into their tokens:",
           "gray": "Gray: the words, the course's values and the words of the sentence.",
           "blue": "Blue: your values where they are the course's.",
           "purple": "Purple: your values where they differ from the course's."},
    "nl": {"title": "Je continued words: de zinnen van de course in zeven talen", "kind": "een getal",
           "intro": "De drie zinnen van de course, met de tokenizer van GPT-2.",
           "head": ["Taal", "Woorden", "Jouw continued words", "De course"],
           "pieces": "De woorden van de Nederlandse zin over de bibliotheek, elk met een spatie vooraan, geknipt in tokens:",
           "gray": "Grijs: de woorden, de waarden van de course en de woorden van de zin.",
           "blue": "Blauw: je waarden waar deze gelijk zijn aan de waarden van de course.",
           "purple": "Paars: je waarden waar deze verschillen van de waarden van de course."},
}


def _same(a: float, b: float) -> bool:
    return abs(a - b) < 1e-9


def _yours_cell(mine: float, want: float, lang: str, digits: int, pct: bool = False) -> str:
    text_ = _pct(mine, lang) if pct else _dec(mine, lang, digits)
    return _cell(html.escape(text_), _PRED if _same(mine, want) else _LOSS, _NUMBER)


def premium_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _PREMIUM_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    got = _call(lambda: _guarded(lambda: {k: _mine(fn, list(OWN[k]), list(OWN["en"]), ENC) for k in OWN}))
    if got is _SLOW or got is _STOPPED or got is None:
        return _empty(w, "slow" if got is _SLOW else "stopped" if got is _STOPPED else "odd")
    top = max(premium(OWN[k], OWN["en"], ENC) for k in OWN)
    rows = []
    for k in OWN:
        want = premium(OWN[k], OWN["en"], ENC)
        tokens = sum(len(ENC.encode(s)) for s in OWN[k])
        rows.append("<tr>" + _cell(html.escape(OWN_NAMES[lang][k])) + _cell(_num(tokens, lang), _DATA, _NUMBER)
                    + _yours_cell(got[k], want, lang, 2) + _cell(_dec(want, lang), _DATA, _NUMBER)
                    + _cell(_bar(want, top)) + "</tr>")
    english = sum(len(ENC.encode(s)) for s in OWN["en"])
    body = _label(w["intro"].format(n=_num(english, lang))) + _table(rows, w["head"], (1, 2, 3))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


def _guarded(make):
    try:
        return make()
    except _Odd:
        return None


def per_byte_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _PER_BYTE_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    got = _call(lambda: _guarded(lambda: {k: _mine(fn, list(OWN[k]), ENC) for k in OWN}))
    if got is _SLOW or got is _STOPPED or got is None:
        return _empty(w, "slow" if got is _SLOW else "stopped" if got is _STOPPED else "odd")
    rows = []
    for k in OWN:
        want = per_byte(OWN[k], ENC)
        n_bytes = sum(len(s.encode("utf-8")) for s in OWN[k])
        tokens = sum(len(ENC.encode(s)) for s in OWN[k])
        rows.append("<tr>" + _cell(html.escape(OWN_NAMES[lang][k])) + _cell(_num(n_bytes, lang), _DATA, _NUMBER)
                    + _cell(_num(tokens, lang), _DATA, _NUMBER) + _yours_cell(got[k], want, lang, 3)
                    + _cell(_dec(want, lang, 3), _DATA, _NUMBER) + "</tr>")
    body = _label(w["intro"]) + _table(rows, w["head"], (1, 2, 3, 4))
    body += _label(w["pieces"]) + _sentence(OWN["el"][0]) + _pieces(ENC, OWN["el"][0])
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


def _word_pieces(enc, word: str) -> str:
    parts = [enc.decode_single_token_bytes(t) for t in enc.encode(" " + word)]
    try:
        shown = "·".join(p.decode("utf-8") for p in parts)
    except UnicodeDecodeError:
        shown = word
    return f'<span style="{_CHIP}">{html.escape(shown.replace(" ", "␣"))}</span>'


def continued_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _CONTINUED_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    got = _call(lambda: _guarded(lambda: {k: _mine(fn, list(OWN[k]), ENC) for k in OWN}))
    if got is _SLOW or got is _STOPPED or got is None:
        return _empty(w, "slow" if got is _SLOW else "stopped" if got is _STOPPED else "odd")
    rows = []
    for k in OWN:
        want = continued(OWN[k], ENC)
        words = sum(len(s.split()) for s in OWN[k])
        rows.append("<tr>" + _cell(html.escape(OWN_NAMES[lang][k])) + _cell(_num(words, lang), _DATA, _NUMBER)
                    + _yours_cell(got[k], want, lang, 3, pct=True) + _cell(_pct(want, lang), _DATA, _NUMBER) + "</tr>")
    body = _label(w["intro"]) + _table(rows, w["head"], (1, 2, 3))
    chips = "".join(_word_pieces(ENC, word) for word in OWN["nl"][0].split())
    body += _label(w["pieces"]) + f'<div style="margin:0.15rem 0 0.25rem;line-height:1.7">{chips}</div>'
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


# --- the demo ----------------------------------------------------------------------------------------------------

_TRY_TEXT = {
    "en": {"title": "One sentence, two tokenizers",
           "english": "The first sentence in English, with {t}:",
           "other": "The same sentence in {name}, with {t}:",
           "head": ["", "GPT-2", "Qwen3"],
           "rows": {"words": "Words", "tokens": "Tokens", "premium": "Premium", "per_byte": "Tokens per byte",
                    "continued": "Continued words", "fertility": "Tokens per word"},
           "all": "Over the first 100 sentences of {name}:",
           "no_spaces": "{name} writes no space between its words, so the words of `split` are not words, and the last two rows say little.",
           "gray": "Gray: the sentences, their tokens and the measures over the 100 sentences; a dashed token holds part of a character, as its bytes in hex."},
    "nl": {"title": "Eén zin, twee tokenizers",
           "english": "De eerste zin in het Engels, met {t}:",
           "other": "Dezelfde zin in het {name}, met {t}:",
           "head": ["", "GPT-2", "Qwen3"],
           "rows": {"words": "Woorden", "tokens": "Tokens", "premium": "Premium", "per_byte": "Tokens per byte",
                    "continued": "Continued words", "fertility": "Tokens per woord"},
           "all": "Over de eerste 100 zinnen van het {name}:",
           "no_spaces": "Het {name} zet geen spatie tussen de woorden, dus de woorden van `split` zijn geen woorden, en de laatste twee rijen zeggen weinig.",
           "gray": "Grijs: de zinnen, de tokens en de maten over de 100 zinnen; een gestippeld token bevat een deel van een teken, als de bytes ervan in hex."},
}
TOKENIZER_NAMES = {"gpt2": "GPT-2", "qwen3": "Qwen3"}


def try_options(lang: str) -> dict:
    """The languages of the demo, by their name in the lesson's language, in alphabetical order, English left out."""
    lang = lang if lang in LANGS else "en"
    names = {ntrex.name(code, lang): code for code in ntrex.LANGUAGES if code != "eng"}
    return {name: names[name] for name in sorted(names)}


def measures(code: str, tok: str) -> dict:
    """The measures of the demo for the first 100 sentences of a language, with the course's functions."""
    def make():
        enc = tokenizer(tok)
        texts, english = ntrex.sentences(code), ntrex.sentences("eng")
        words = sum(len(s.split()) for s in texts)
        tokens = sum(len(enc.encode(s)) for s in texts)
        return {"words": words, "tokens": tokens, "premium": premium(texts, english, enc),
                "per_byte": per_byte(texts, enc), "continued": continued(texts, enc), "fertility": tokens / words}
    return kept(f"nlp-low-resource:measures:{code}:{tok}", make)


def try_picture(lang: str, code: str = TRY_START, tok: str = "gpt2") -> str:
    lang = lang if lang in LANGS else "en"
    code = code if code in ntrex.LANGUAGES else TRY_START
    tok = tok if tok in TOKENIZERS else "gpt2"
    w = _TRY_TEXT[lang]
    enc = tokenizer(tok)
    name = ntrex.name(code, lang)
    shown = name if lang == "en" else name
    first_en, first = ntrex.sentences("eng", 1)[0], ntrex.sentences(code, 1)[0]
    body = _label(w["english"].format(t=TOKENIZER_NAMES[tok])) + _sentence(first_en) + _pieces(enc, first_en)
    body += _label(w["other"].format(name=shown, t=TOKENIZER_NAMES[tok])) + _sentence(first) + _pieces(enc, first)
    m = {t: measures(code, t) for t in TOKENIZERS}
    rows = []
    for key, label_ in w["rows"].items():
        cells = ""
        for t in TOKENIZERS:
            v = m[t][key]
            text_ = (_num(v, lang) if key in ("words", "tokens") else _pct(v, lang) if key == "continued"
                     else _dec(v, lang, 3 if key == "per_byte" else 2))
            cells += _cell(html.escape(text_), _DATA + (";" + _BOLD if t == tok else ""), _NUMBER)
        rows.append("<tr>" + _cell(html.escape(label_)) + cells + "</tr>")
    body += _label(w["all"].format(name=shown)) + _table(rows, w["head"], (1, 2))
    if not ntrex.LANGUAGES[code][3]:
        body += _note(w["no_spaces"].format(name=shown))
    return yours.figure(w["title"], body + _note(w["gray"]), alt=w["title"])


# --- the exercises -----------------------------------------------------------------------------------------------

class _Val:
    """A test value that the check's messages write by its name, such as `enc`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value)


class _Shared(_Val):
    """A test value the function gets itself, not a copy: the tokenizer."""

    def __deepcopy__(self, memo):
        return self.value


def _value(v) -> str:
    """A value the way code writes it, with double quotes."""
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, list):
        return "[" + ", ".join(_value(x) for x in v) + "]"
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

ENC_VAL = _Shared("enc", ENC)
TRAIN = 1   # the sentence about the train, the visible case of the exercises


def _own(k: str, i: int | None = None) -> _Val:
    if i is None:
        return _Val(f'C.OWN["{k}"]', OWN[k])
    return _written([OWN[k][i]])


_PR_HEAD = ("def premium(texts, english, enc):\n    tokens = 0\n    for text in texts:\n")
_PR_MID = "    english_tokens = 0\n    for text in english:\n"
_PR_END = "    return tokens / english_tokens\n"


def _pr(first: str = "        tokens = tokens + len(enc.encode(text))\n",
        second: str = "        english_tokens = english_tokens + len(enc.encode(text))\n", end: str = _PR_END) -> str:
    return _PR_HEAD + first + _PR_MID + second + end


_PR = _pr()
_PR_START = _pr("        tokens = tokens + ...\n", "        english_tokens = english_tokens + ...\n")
_PR_CHARS = _pr("        tokens = tokens + len(text)\n", "        english_tokens = english_tokens + len(text)\n")
_PR_WORDS = _pr("        tokens = tokens + len(text.split())\n", "        english_tokens = english_tokens + len(text.split())\n")
_PR_UPSIDE = _pr(end="    return english_tokens / tokens\n")
_PR_LAST = _pr("        tokens = len(enc.encode(text))\n", "        english_tokens = len(enc.encode(text))\n")

_PB_HEAD = "def per_byte(texts, enc):\n    tokens = 0\n    n_bytes = 0\n    for text in texts:\n        tokens = tokens + len(enc.encode(text))\n"
_PB_END = "    return tokens / n_bytes\n"


def _pb(line: str) -> str:
    return _PB_HEAD + line + _PB_END


_PB = _pb("        n_bytes = n_bytes + len(text.encode(\"utf-8\"))\n")
_PB_START = _pb("        n_bytes = n_bytes + ...\n")
_PB_CHARS = _pb("        n_bytes = n_bytes + len(text)\n")
_PB_TOKENS = _pb("        n_bytes = n_bytes + len(enc.encode(text))\n")
_PB_WORDS = _pb("        n_bytes = n_bytes + len(text.split())\n")

_CO_HEAD = ("def continued(texts, enc):\n    words = 0\n    count = 0\n    for text in texts:\n"
            "        for word in text.split():\n            words = words + 1\n")
_CO_END = "                count = count + 1\n    return count / words\n"


def _co(line: str) -> str:
    return _CO_HEAD + line + _CO_END


_CO = _co("            if len(enc.encode(\" \" + word)) > 1:\n")
_CO_START = _co("            if ...:\n")
_CO_NO_SPACE = _co("            if len(enc.encode(word)) > 1:\n")
_CO_ALL = _co("            if len(enc.encode(\" \" + word)) >= 1:\n")
_CO_TWO = _co("            if len(enc.encode(\" \" + word)) > 2:\n")
_CO_LETTERS = _co("            if len(word) > 1:\n")

REQUIRED = ["premium", "per_byte", "continued", "helps"]

EXERCISES = {
    "premium": FunctionBox(
        name="premium",
        picture=premium_picture,
        function="premium",
        start=_PR_START,
        solution=_PR,
        show=_show,
        compare=_close,
        visible=[(_own("nl", TRAIN), _own("en", TRAIN), ENC_VAL)],
        hidden=[(_own("nl"), _own("en"), ENC_VAL), (_own("ja"), _own("en"), ENC_VAL), (_own("el"), _own("en"), ENC_VAL),
                (_own("en"), _own("en"), ENC_VAL), (_own("en"), _own("ru"), ENC_VAL),
                (_written(["Hallo!", "Dank je wel."]), _written(["Hello!", "Thank you."]), ENC_VAL)],
        first=[_DOTS],
        mistakes=[
            Mistake("characters", wrong=_PR_CHARS,
                    en="`{call}` gives {got}, and it should give {want}: you count characters, not tokens. Which list holds the tokens of `text`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je telt tekens, geen tokens. Welke lijst bevat de tokens van `text`?"),
            Mistake("words", wrong=_PR_WORDS,
                    en="`{call}` gives {got}, and it should give {want}: you count words, not tokens, and a word can be several tokens. Which list holds the tokens of `text`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je telt woorden, geen tokens, en een woord kan uit meer tokens bestaan. Welke lijst bevat de tokens van `text`?"),
            Mistake("last-only", wrong=_PR_LAST,
                    en="`{call}` gives {got}, and it should give {want}: each count holds only the tokens of the last text. What should a count add its new tokens to?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke telling bevat alleen de tokens van de laatste tekst. Waarbij moet een telling de nieuwe tokens optellen?"),
            Mistake("upside-down", wrong=_PR_UPSIDE,
                    en="`{call}` gives {got}, and it should give {want}: the last line divides English by the other language. Did you change that line?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de laatste regel deelt het Engels door de andere taal. Heb je deze regel veranderd?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: the number of tokens of `text`, once in each loop.",
                      "nl": "Vervang de twee `...`: het aantal tokens van `text`, één keer in elke loop."},
            "arguments": {"en": "The check calls `{call}`, with three values, and the `def` line of your function has another number of names between its parentheses. Keep `(texts, english, enc)`.",
                          "nl": "De check roept `{call}` aan, met drie waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(texts, english, enc)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does each loop add the number of tokens of every `text` to its count?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Telt elke loop het aantal tokens van elke `text` op bij de eigen telling?"},
            "almost": {"en": "Right for the sentence about the train. But `{call}` gives {got}, and it should give {want}. Does your function count every sentence of both lists?",
                       "nl": "Goed voor de zin over de trein. Maar `{call}` geeft {got}, en het moet {want} zijn. Telt je functie elke zin van beide lijsten?"},
            "correct": {"en": "Right: the sentence about the train costs GPT-2 17 tokens in Dutch and 11 in English, a premium of 1.55, and all three sentences 1.83. Which of the seven languages in the picture costs the most?",
                        "nl": "Goed: de zin over de trein kost GPT-2 in het Nederlands 17 tokens en in het Engels 11, een premium van 1,55, en alle drie de zinnen 1,83. Welke van de zeven talen in het plaatje kost het meest?"},
        },
    ),
    "per_byte": FunctionBox(
        name="per_byte",
        picture=per_byte_picture,
        function="per_byte",
        start=_PB_START,
        solution=_PB,
        show=_show,
        compare=_close,
        visible=[(_own("el", TRAIN), ENC_VAL)],
        hidden=[(_own(k), ENC_VAL) for k in OWN] + [(_written(["ünïcödé", "a"]), ENC_VAL)],
        first=[_DOTS],
        mistakes=[
            Mistake("characters", wrong=_PB_CHARS,
                    en="`{call}` gives {got}, and it should give {want}: you count characters, not bytes, and a Greek letter is two bytes in UTF-8. What turns a text into its bytes?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je telt tekens, geen bytes, en een Griekse letter is in UTF-8 twee bytes. Wat maakt van een tekst de bytes?"),
            Mistake("tokens", wrong=_PB_TOKENS,
                    en="`{call}` gives {got}, and it should give {want}: `n_bytes` counts tokens too, so the function always gives 1. What turns a text into its bytes?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `n_bytes` telt ook tokens, dus de functie geeft altijd 1. Wat maakt van een tekst de bytes?"),
            Mistake("words", wrong=_PB_WORDS,
                    en="`{call}` gives {got}, and it should give {want}: you count words, not bytes. What turns a text into its bytes?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je telt woorden, geen bytes. Wat maakt van een tekst de bytes?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: the number of bytes of `text` in UTF-8.",
                      "nl": "Vervang de `...`: het aantal bytes van `text` in UTF-8."},
            "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has another number of names between its parentheses. Keep `(texts, enc)`.",
                          "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(texts, enc)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does the loop add the number of bytes of every `text` in UTF-8?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Telt de loop het aantal bytes van elke `text` in UTF-8 op?"},
            "almost": {"en": "Right for the Greek sentence about the train. But `{call}` gives {got}, and it should give {want}. Does your function count the bytes of every sentence?",
                       "nl": "Goed voor de Griekse zin over de trein. Maar `{call}` geeft {got}, en het moet {want} zijn. Telt je functie de bytes van elke zin?"},
            "correct": {"en": "Right: the Greek sentence about the train has 85 bytes, and GPT-2 makes 55 tokens of them, 0.647 per byte; of the English one, 0.256 per byte. Which language in the picture comes closest to one token per byte?",
                        "nl": "Goed: de Griekse zin over de trein heeft 85 bytes, en GPT-2 maakt er 55 tokens van, 0,647 per byte; van de Engelse zin 0,256 per byte. Welke taal in het plaatje komt het dichtst bij één token per byte?"},
        },
    ),
    "continued": FunctionBox(
        name="continued",
        picture=continued_picture,
        function="continued",
        start=_CO_START,
        solution=_CO,
        show=_show,
        compare=_close,
        visible=[(_own("nl", TRAIN), ENC_VAL)],
        hidden=[(_own(k), ENC_VAL) for k in OWN] + [(_written(["a a a", "unbelievable"]), ENC_VAL)],
        first=[_DOTS],
        mistakes=[
            Mistake("no-space", wrong=_CO_NO_SPACE,
                    en="`{call}` gives {got}, and it should give {want}: you encode each word without the space before it, and GPT-2 has other tokens for a word at the start of a text. How does the word stand in the sentence?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je codeert elk woord zonder de spatie vooraan, en GPT-2 heeft andere tokens voor een woord aan het begin van een tekst. Hoe staat het woord in de zin?"),
            Mistake("every-word", wrong=_CO_ALL,
                    en="`{call}` gives {got}, and it should give {want}: every word is at least one token, so your test counts every word. When is a word cut?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elk woord is minstens één token, dus je test telt elk woord. Wanneer is een woord geknipt?"),
            Mistake("three", wrong=_CO_TWO,
                    en="`{call}` gives {got}, and it should give {want}: you count only words of three tokens or more. Is a word of two tokens cut?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je telt alleen woorden van drie of meer tokens. Is een woord van twee tokens geknipt?"),
            Mistake("letters", wrong=_CO_LETTERS,
                    en="`{call}` gives {got}, and it should give {want}: you count the words of more than one character, not of more than one token. What gives the tokens of a word?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je telt de woorden van meer dan één teken, niet van meer dan één token. Wat geeft de tokens van een woord?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: the test whether the tokenizer cuts `word`, with a space before it, into more than one token.",
                      "nl": "Vervang de `...`: de test of de tokenizer `word`, met een spatie vooraan, in meer dan één token knipt."},
            "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has another number of names between its parentheses. Keep `(texts, enc)`.",
                          "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(texts, enc)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your test count a word when the tokenizer cuts it, with a space before it, into two tokens or more?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Telt je test een woord als de tokenizer het woord, met een spatie vooraan, in twee of meer tokens knipt?"},
            "almost": {"en": "Right for the Dutch sentence about the train. But `{call}` gives {got}, and it should give {want}. Does your test work for every word?",
                       "nl": "Goed voor de Nederlandse zin over de trein. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je test voor elk woord?"},
            "correct": {"en": "Right: GPT-2 cuts six of the eight Dutch words about the train, 75%, and two of the eight English words: `Utrecht`, and `minutes.` with its period. What happens to the Japanese sentences, written without spaces?",
                        "nl": "Goed: GPT-2 knipt zes van de acht Nederlandse woorden over de trein, 75%, en twee van de acht Engelse woorden: `Utrecht`, en `minutes.` met de punt. Wat gebeurt er met de Japanse zinnen, zonder spaties geschreven?"},
        },
    ),
    "helps": Sort(
        name="helps",
        groups={
            "helps": {"en": "Helps", "nl": "Helpt"},
            "short": {"en": "Falls short", "nl": "Schiet tekort"},
        },
        items={
            "related": {"en": "**A related language.** Fine-tune a multilingual model on labeled data of a related language with a lot of it.",
                        "nl": "**Een verwante taal.** Finetune een meertalig model op gelabelde data van een verwante taal, een taal met veel gelabelde data."},
            "few": {"en": "**A few hundred examples.** After fine-tuning on another language, add a few hundred labeled sentences in the language itself.",
                    "nl": "**Een paar honderd voorbeelden.** Voeg na de fine-tuning in een andere taal een paar honderd gelabelde zinnen toe in de taal zelf."},
            "pieces": {"en": "**Its own pieces.** Give a multilingual model a tokenizer made for the language.",
                       "nl": "**De eigen stukjes.** Geef een meertalig model een tokenizer voor de taal zelf."},
            "scratch": {"en": "**A BERT of its own.** Train a BERT from scratch on the little text the language has.",
                        "nl": "**Een eigen BERT.** Train een BERT vanaf nul op de weinige tekst van de taal."},
            "order": {"en": "**Another word order.** Fine-tune only on a language with another order of subject, verb and object.",
                      "nl": "**Een andere woordvolgorde.** Finetune alleen op een taal met een andere volgorde van onderwerp, werkwoord en lijdend voorwerp."},
            "english": {"en": "**English pieces.** Use the tokenizer of an English model, such as GPT-2's.",
                        "nl": "**Engelse stukjes.** Gebruik de tokenizer van een Engels model, zoals de tokenizer van GPT-2."},
        },
        solution={"related": "helps", "few": "helps", "pieces": "helps", "scratch": "short", "order": "short",
                  "english": "short"},
        nudge={
            "related": {"en": "Pires and colleagues fine-tuned on one language and measured on another: when did that work best?",
                        "nl": "Pires en collega's deden fine-tuning in één taal en maten in een andere: wanneer werkte dat het best?"},
            "few": {"en": "What did Lauscher and colleagues find for parsing with as few as 10 sentences in the target language?",
                    "nl": "Wat vonden Lauscher en collega's voor parsing met niet meer dan tien zinnen in de doeltaal?"},
            "pieces": {"en": "What did Rust and colleagues find when they swapped a multilingual tokenizer for one made for the language?",
                       "nl": "Wat vonden Rust en collega's toen de onderzoekers een meertalige tokenizer vervingen door een tokenizer voor de taal zelf?"},
            "scratch": {"en": "What did Wu and Dredze find for monolingual BERT models of low-resource languages?",
                        "nl": "Wat vonden Wu en Dredze voor eentalige BERT-modellen van talen met weinig data?"},
            "order": {"en": "In the study of Pires and colleagues, how did transfer from SVO to SOV compare with transfer from SVO to SVO?",
                      "nl": "Hoe verhield de transfer van SVO naar SOV zich tot de transfer van SVO naar SVO, in het onderzoek van Pires en collega's?"},
            "english": {"en": "What premium did GPT-2's tokenizer give Tamil, and what did Rust and colleagues find a tokenizer does to a model?",
                        "nl": "Welke premium gaf de tokenizer van GPT-2 het Tamil, en wat doet een tokenizer met een model, volgens Rust en collega's?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the groups again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de groepen opnieuw uit."},
            "empty": {"en": "Sort each way into whether it helps a language with little data or falls short.",
                      "nl": "Deel elke aanpak in: helpt de aanpak een taal met weinig data, of schiet de aanpak tekort?"},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Je indeling klopt tot nu toe. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} belongs in the other group.",
                      "nl": "{item} hoort bij de andere groep."},
            "correct": {"en": "All {n} are right. A language with little data borrows: from a related language with labeled data, through a multilingual model, and from a tokenizer made for it, which helped mBERT in almost every task and language that Rust and colleagues tried; a few hundred examples of its own then go a long way. A model of its own has too little text to learn from, transfer from another word order works less well, and a tokenizer that learned its pieces from English cuts the language into many more tokens.",
                        "nl": "Alle {n} kloppen. Een taal met weinig data leent: van een verwante taal met gelabelde data, via een meertalig model, en van een tokenizer voor de taal zelf, wat mBERT hielp bij bijna elke taak en taal van Rust en collega's; een paar honderd eigen voorbeelden helpen dan veel. Een eigen model heeft te weinig tekst om van te leren, transfer vanuit een andere woordvolgorde werkt minder goed, en een tokenizer met stukjes uit het Engels knipt de taal in veel meer tokens."},
        },
    ),
}

CASES = [
    ("premium", _PR_START, "empty", None),
    ("premium", _PR, "correct", None),
    ("premium", _pr(second="        english_tokens = english_tokens + ...\n"), "wrong", "dots-left"),
    ("premium", _PR_CHARS, "wrong", "characters"),
    ("premium", _PR_WORDS, "wrong", "words"),
    ("premium", _PR_LAST, "wrong", "last-only"),
    ("premium", _PR_UPSIDE, "wrong", "upside-down"),
    ("premium", _pr("        tokens += len(enc.encode(text))\n", "        english_tokens += len(enc.encode(text))\n"), "correct", None),
    ("premium", "def premium(texts, english, enc):\n    return sum(len(enc.encode(t)) for t in texts) / sum(len(enc.encode(t)) for t in english)\n",
     "correct", None),
    ("premium", _pr(second="        english_tokens = len(enc.encode(text))\n"), "almost", None),
    ("premium", _pr("        tokens = tokens + len(enc.encode(text)) - 1\n", "        english_tokens = english_tokens + len(enc.encode(text)) - 1\n"),
     "wrong", None),
    ("per_byte", _PB_START, "empty", None),
    ("per_byte", _PB, "correct", None),
    ("per_byte", _pb("        n_bytes = n_bytes + len(...)\n"), "wrong", "dots-left"),
    ("per_byte", _PB_CHARS, "wrong", "characters"),
    ("per_byte", _PB_TOKENS, "wrong", "tokens"),
    ("per_byte", _PB_WORDS, "wrong", "words"),
    ("per_byte", _pb("        n_bytes = n_bytes + len(text.encode())\n"), "correct", None),
    ("per_byte", _pb("        n_bytes += len(bytes(text, \"utf-8\"))\n"), "correct", None),
    ("per_byte", _pb("        n_bytes = n_bytes + len(text.encode(\"utf-16\"))\n"), "wrong", None),
    ("per_byte", _pb("        n_bytes = n_bytes + len(texts[0].encode(\"utf-8\"))\n"), "almost", None),
    ("continued", _CO_START, "empty", None),
    ("continued", _CO, "correct", None),
    ("continued", _co("            if len(enc.encode(...)) > 1:\n"), "wrong", "dots-left"),
    ("continued", _CO_NO_SPACE, "wrong", "no-space"),
    ("continued", _CO_ALL, "wrong", "every-word"),
    ("continued", _CO_TWO, "wrong", "three"),
    ("continued", _CO_LETTERS, "wrong", "letters"),
    ("continued", _co("            if len(enc.encode(\" \" + word)) >= 2:\n"), "correct", None),
    ("continued", _co("            if not len(enc.encode(\" \" + word)) == 1:\n"), "correct", None),
    ("continued", _co("            if len(enc.encode(\" \" + word)) > 1 and len(word) > 2:\n"), "almost", None),
    ("helps", {}, "empty", None),
    ("helps", dict(EXERCISES["helps"].solution), "correct", None),
    ("helps", {"related": "helps", "few": "helps"}, "wrong", "unfinished"),
] + [("helps", {item: ("short" if group == "helps" else "helps")}, "wrong", f"wrong:{item}")
     for item, group in EXERCISES["helps"].solution.items()]
