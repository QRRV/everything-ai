# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-tokenize: cut a text into tokens with one pattern (tokens), say whether a word cut at
spaces ends a sentence (ends_sentence), and give the longest start of a text without spaces that is a word
(longest_start), and sort cases by the fix they need: a list of exceptions, a pattern for the form, or a dictionary and
the longest match.

The sentences with their words come from the treebanks UD_Dutch-Alpino and UD_English-EWT of Universal Dependencies
2.19 (CC BY-SA 4.0) in data/ud-nl-alpino.conllu.gz and data/ud-en-ewt.conllu.gz (made by data/tools/make_ud.py), each
sentence taken by its sent_id. The dictionary of MaxMatch is every word of the Dutch sentences of Tatoeba (CC BY 2.0 FR)
in data/tatoeba-nl.tsv.gz, in small letters. Every number in the pictures is counted by the course; the list of
abbreviations, the strings in the cases of the checks and the items of the sort are the course's own."""
import contextlib
import copy
import gzip
import html
import io
import re

from everything_ai import books, paths, yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, kept, with_time_limit

# --- the course's solutions ------------------------------------------------------------------------------------------


def tokens(text):
    """The course's solution of Exercise 1: the text cut into tokens, a run of letters, digits and _ or one character
    that is neither of these nor white space, in order."""
    return re.findall(r"\w+|[^\w\s]", text)


def ends_sentence(word, abbreviations):
    """The course's solution of Exercise 2: True when a word cut at spaces ends a sentence: it ends with . ? or !, and
    it is neither an abbreviation (in small letters in `abbreviations`) nor an initial such as J."""
    if word.lower() in abbreviations or re.search(r"^[A-Z]\.$", word):
        return False
    return re.search(r"[.?!]$", word) is not None


def longest_start(text, words):
    """The course's solution of Exercise 3: the longest start of `text` that is in `words`, and the first character
    when no start is."""
    for end in range(len(text), 0, -1):
        if text[:end] in words:
            return text[:end]
    return text[:1]


def longest_end(text, words):
    """The longest end of `text` that is in `words`, and the last character when no end is: the step of backward
    MaxMatch."""
    for start in range(len(text)):
        if text[start:] in words:
            return text[start:]
    return text[-1:]


def max_match(text, words, step=longest_start) -> list:
    """MaxMatch: take the longest start that is a word, then go on after it, until the text is used up. `step` is a
    function such as longest_start; a step that gives back something other than a start of the text that is left
    raises ValueError, so that a learner's function cannot keep it going forever."""
    out = []
    while text:
        piece = step(text, words)
        if not isinstance(piece, str) or not piece or text[:len(piece)] != piece:
            raise ValueError(piece)
        out.append(piece)
        text = text[len(piece):]
    return out


def backward_max_match(text, words) -> list:
    """Backward MaxMatch: the same from the end of the text."""
    out = []
    while text:
        piece = longest_end(text, words)
        out.insert(0, piece)
        text = text[:len(text) - len(piece)]
    return out


# --- the treebanks ---------------------------------------------------------------------------------------------------

UD = {"nl": paths.ROOT / "data" / "ud-nl-alpino.conllu.gz", "en": paths.ROOT / "data" / "ud-en-ewt.conllu.gz"}


def _read(path) -> tuple:
    """The sentences of a CoNLL-U file of the course: (sent_id, text, words). The course's files hold syntactic words
    only, without the lines of multiword tokens (make_ud.py)."""
    out, words, sent_id, text = [], [], None, None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# sent_id = "):
                sent_id = line[len("# sent_id = "):]
            elif line.startswith("# text = "):
                text = line[len("# text = "):]
            elif not line:
                if words:
                    out.append((sent_id, text, tuple(words)))
                words = []
            elif not line.startswith("#"):
                words.append(line.split("\t", 2)[1])
    if words:
        out.append((sent_id, text, tuple(words)))
    return tuple(out)


def treebank(lang: str = "nl") -> tuple:
    """Every sentence of UD_Dutch-Alpino (nl) or UD_English-EWT (en) as (sent_id, text, words)."""
    return kept(f"nlp-tokenize:{lang}", lambda: _read(UD[lang]))


def _by_id() -> dict:
    return kept("nlp-tokenize:by-id", lambda: {sid: (text, words) for sid, text, words in treebank("nl")})


def sentence(sent_id: str) -> str:
    """The text of the sentence of Alpino with this sent_id."""
    return _by_id()[sent_id][0]


def treebank_words(sent_id: str) -> list:
    """The words of the sentence of Alpino with this sent_id, as the treebank cuts them."""
    return list(_by_id()[sent_id][1])


def n_sentences(lang: str = "nl") -> int:
    return len(treebank(lang))


# the sentences the lesson shows, by their sent_id in Alpino
CITIES_ID = "cdb\\1528"
EXAMPLE_ID = "cdb\\4577"
IDEA_ID = "WR-P-P-H-0000000017\\WR-P-P-H-0000000017.p.1.s.6"


def cities() -> str:
    """The sentence of the visible case of Exercise 1: three cities, a comma and a question mark."""
    return sentence(CITIES_ID)


def example() -> str:
    """The sentence of the worked example: an abbreviation and two initials before its end."""
    return sentence(EXAMPLE_ID)


def spans(text: str, words) -> list:
    """Where each word stands in the text, as (start, end), looking for each word after the one before it; None for a
    word that is not found."""
    out, at = [], 0
    for w in words:
        i = text.find(w, at)
        if i < 0:
            out.append(None)
            continue
        out.append((i, i + len(w)))
        at = i + len(w)
    return out


def exact(fn, lang: str = "nl") -> int:
    """In how many sentences of the treebank `fn` gives exactly the words of the treebank."""
    return sum(1 for _, text, words in treebank(lang) if list(fn(text)) == list(words))


# --- what a tokenizer cuts differently: the kinds of words -------------------------------------------------------------

KINDS = ["hyphen", "initial", "abbreviation", "apostrophe", "number", "punctuation", "other"]


def kind(word: str) -> str:
    """The kind of a word of the treebank, for the picture under Exercise 1."""
    if re.fullmatch(r"[A-Z]\.", word):
        return "initial"
    if re.search(r"\w", word):
        if "'" in word or "’" in word:
            return "apostrophe"
        if re.fullmatch(r"\d+(?:[.,]\d+)+", word):
            return "number"
        if "-" in word:
            return "hyphen"
        if word.endswith("."):
            return "abbreviation"
        return "other"
    return "punctuation"


def differences(results: list) -> tuple:
    """For a tokenizer's results on all sentences of Alpino, in their order: (number cut exactly, for each kind the
    number of sentences in which the treebank has a word of that kind that the tokenizer does not give, for each kind
    the words most often missed, as (word, count))."""
    per = {k: 0 for k in KINDS}
    words_of = {k: {} for k in KINDS}
    right = 0
    for (_, _, words), mine in zip(treebank("nl"), results):
        if list(mine) == list(words):
            right += 1
            continue
        left = {}
        for w in mine:
            left[w] = left.get(w, 0) + 1
        seen = set()
        for w in words:
            if left.get(w, 0):
                left[w] -= 1
                continue
            k = kind(w)
            seen.add(k)
            words_of[k][w] = words_of[k].get(w, 0) + 1
        for k in seen:
            per[k] += 1
    common = {k: sorted(v.items(), key=lambda kv: (-kv[1], kv[0])) for k, v in words_of.items()}
    return right, per, common


def rule_differences() -> tuple:
    return kept("nlp-tokenize:rule-diff", lambda: differences([tokens(t) for _, t, _ in treebank("nl")]))


# --- small helpers for the pictures ----------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums"
_MISS = "color:var(--ea-loss-text);font-weight:600"
_LINE = "font-family:var(--ea-font-ui);font-size:0.9rem;color:var(--ea-ink);line-height:1.9"
_CODE = "font-family:var(--ea-font-code);font-size:0.82rem;color:var(--ea-ink)"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);margin-top:0.35rem"
_CHIP = ("display:inline-block;margin:0.12rem 0.2rem 0.12rem 0;padding:0 0.3rem;border-radius:3px;"
         "font-family:var(--ea-font-code);font-size:0.82rem;line-height:1.5;white-space:pre")
_CHIP_RIGHT = _CHIP + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
_CHIP_WRONG = _CHIP + ";background:var(--ea-loss-wash);color:var(--ea-loss-text)"
_CHIP_DATA = _CHIP + ";border:1px solid var(--ea-line);color:var(--ea-ink)"


def _note(text: str) -> str:
    """A line under a picture; a word between asterisks is set in italics, as in the lesson's text."""
    parts = html.escape(text).split("*")
    marked = "".join(f"<i>{p}</i>" if k % 2 else p for k, p in enumerate(parts))
    return f'<div style="{_UI}">{marked}</div>'


def _num(n, lang: str) -> str:
    """A whole number with a separator for thousands: 2,319 in English, 2.319 in Dutch."""
    text = f"{round(n):,}"
    return text.replace(",", ".") if lang == "nl" else text


def _pct(part: int, whole: int, lang: str) -> str:
    """A share as a percentage with one decimal: 89.2% in English, 89,2% in Dutch."""
    text = f"{100 * part / whole:.1f}%" if whole else "–"
    return text.replace(".", ",") if lang == "nl" else text


def _table(rows_: list, head: list, numbers: tuple = (), styles: dict | None = None) -> str:
    """A small table; rows hold HTML that is already escaped; numbers are the columns set to the right; styles maps
    (row, column) to extra style."""
    styles = styles or {}
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{_NUMBER if j in numbers else ""}{styles.get((i, j), "")}">'
                                    f'{c}</td>' for j, c in enumerate(row)) + "</tr>" for i, row in enumerate(rows_))
    return (f'<table style="border-collapse:collapse;margin:0.3rem 0 0.2rem"><thead><tr>{top}</tr></thead>'
            f"<tbody>{body}</tbody></table>")


def _chips(pieces: list, styles: list) -> str:
    return "".join(f'<span style="{s}">{html.escape(p)}</span>' for p, s in zip(pieces, styles))


# --- the idea's picture: one sentence cut three ways ------------------------------------------------------------------

_IDEA_WORDS = {
    "en": {"title": "One sentence, cut three ways",
           "spaces": "At the spaces: {k} of {n} pieces as in the treebank",
           "rule": "At the spaces, and every punctuation mark on its own: {k} of {n}",
           "bank": "The treebank: {n} words",
           "note": "Blue: a piece that is also a word of the treebank. Purple: a piece the treebank cuts differently.",
           "alt": "The sentence {s}, cut three ways. {rows}. {note}"},
    "nl": {"title": "Eén zin, op drie manieren geknipt",
           "spaces": "Bij de spaties: {k} van {n} stukken als in de treebank",
           "rule": "Bij de spaties, en elk leesteken apart: {k} van {n}",
           "bank": "De treebank: {n} woorden",
           "note": "Blauw: een stuk dat ook een woord van de treebank is. Paars: een stuk dat de treebank anders knipt.",
           "alt": "De zin {s}, op drie manieren geknipt. {rows}. {note}"},
}


def cut_rows(text: str, words) -> list:
    """For the picture of the idea: (pieces, number of pieces that are words of the treebank) for the cut at spaces
    and for the rule of Exercise 1."""
    gold = set(spans(text, words))
    out = []
    for pattern in (r"\S+", r"\w+|[^\w\s]"):
        found = list(re.finditer(pattern, text))
        out.append(([(m.group(), (m.start(), m.end()) in gold) for m in found],
                    sum(1 for m in found if (m.start(), m.end()) in gold)))
    return out


def idea_picture(lang: str) -> str:
    w = _IDEA_WORDS[lang]
    text = sentence(IDEA_ID)
    words = treebank_words(IDEA_ID)
    rows_, alts = [], []
    for key, (pieces, k) in zip(("spaces", "rule"), cut_rows(text, words)):
        label = w[key].format(k=k, n=len(pieces))
        rows_.append(f'<div style="{_LABEL}">{html.escape(label)}</div>'
                     + _chips([p for p, _ in pieces], [_CHIP_RIGHT if ok else _CHIP_WRONG for _, ok in pieces]))
        alts.append(label + ": " + " | ".join(p for p, _ in pieces))
    label = w["bank"].format(n=len(words))
    rows_.append(f'<div style="{_LABEL}">{html.escape(label)}</div>' + _chips(words, [_CHIP_DATA] * len(words)))
    alts.append(label + ": " + " | ".join(words))
    body = f'<div style="{_LINE}">{html.escape(text)}</div>' + "".join(rows_) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["alt"].format(s=text, rows="; ".join(alts), note=w["note"]))


# --- running the learner's function ----------------------------------------------------------------------------------


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error
PICTURE_LIMIT = 10.0   # seconds the learner's function gets for all sentences of a picture


def _call(make, seconds: float = PICTURE_LIMIT):
    """make() within a time limit, with what it prints thrown away; _STOPPED when it stops with an error, and the
    Timeout passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(make, seconds)
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _strings(v) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


# --- the picture under exercise 1: your tokens against the treebank ----------------------------------------------------

_KIND_WORDS = {
    "en": {"hyphen": "with a hyphen", "initial": "an initial", "abbreviation": "an abbreviation",
           "apostrophe": "with an apostrophe", "number": "a number with a period or comma",
           "punctuation": "punctuation", "other": "something else"},
    "nl": {"hyphen": "met een koppelteken", "initial": "een initiaal", "abbreviation": "een afkorting",
           "apostrophe": "met een apostrof", "number": "een getal met een punt of komma",
           "punctuation": "leestekens", "other": "iets anders"},
}
_TOKEN_WORDS = {
    "en": {"title": "Your tokens and the treebank",
           "line": "Your function cuts {k} of the {n} sentences of Alpino exactly as the treebank does ({p}).",
           "head": ["the treebank has a word", "sentences", "most often"],
           "note": "In each other sentence, the treebank has at least one word that your function does not give; a "
                   "sentence can have words of more than one kind.",
           "all": "Every sentence is cut exactly as in the treebank.",
           "stopped": "Your function stopped with an error on one of the sentences, so the picture stays empty.",
           "slow": "Your function took more than {s:g} seconds for all sentences, so the picture stays empty.",
           "odd": "Your function does not give back a list of strings, so the picture stays empty."},
    "nl": {"title": "Jouw tokens en de treebank",
           "line": "Je functie knipt {k} van de {n} zinnen van Alpino precies zoals de treebank ({p}).",
           "head": ["de treebank heeft een woord", "zinnen", "het vaakst"],
           "note": "In elke andere zin heeft de treebank minstens één woord dat je functie niet geeft; een zin kan "
                   "woorden van meer dan één soort hebben.",
           "all": "Elke zin is precies zo geknipt als in de treebank.",
           "stopped": "Je functie stopte met een fout op een van de zinnen, dus de afbeelding blijft leeg.",
           "slow": "Je functie deed er langer dan {s:g} seconden over voor alle zinnen, dus de afbeelding blijft leeg.",
           "odd": "Je functie geeft geen lijst met strings terug, dus de afbeelding blijft leeg."},
}


def tokens_picture(seen) -> str:
    lang = seen.lang
    w = _TOKEN_WORDS[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    texts = [t for _, t, _ in treebank("nl")]
    try:
        got = _call(lambda: [fn(t) for t in texts])
    except Timeout:
        text = w["slow"].format(s=PICTURE_LIMIT)
        return yours.figure(w["title"], _note(text), alt=text)
    if got is _STOPPED:
        return yours.figure(w["title"], _note(w["stopped"]), alt=w["stopped"])
    if not all(_strings(g) for g in got):
        return yours.figure(w["title"], _note(w["odd"]), alt=w["odd"])
    right, per, common = differences(got)
    n = len(texts)
    line = w["line"].format(k=_num(right, lang), n=_num(n, lang), p=_pct(right, n, lang))
    rows_ = [[html.escape(_KIND_WORDS[lang][k]), _num(per[k], lang),
              _chips([x for x, _ in common[k][:4]], [_CHIP_DATA] * 4)] for k in KINDS if per[k]]
    alt = (w["title"] + ". " + line + " " + "; ".join(f"{_KIND_WORDS[lang][k]}: {per[k]} "
                                                      f"({', '.join(x for x, _ in common[k][:4])})"
                                                      for k in KINDS if per[k]))
    if not rows_:
        return yours.figure(w["title"], f'<div style="{_LINE}">{html.escape(line)}</div>' + _note(w["all"]),
                            alt=alt + " " + w["all"])
    body = (f'<div style="{_LINE}">{html.escape(line)}</div>' + _table(rows_, w["head"], numbers=(1,))
            + _note(w["note"]))
    return yours.figure(w["title"], body, alt=alt + " " + w["note"])


# --- the picture under exercise 2: where the sentences of Alpino end ---------------------------------------------------

# the course's own short list of Dutch abbreviations, in small letters
ABBREVIATIONS = frozenset({
    "a.s.", "bijv.", "blz.", "bv.", "ca.", "d.w.z.", "dhr.", "dr.", "drs.", "e.d.", "enz.", "etc.", "i.p.v.", "incl.",
    "ir.", "jl.", "m.b.t.", "mevr.", "mld.", "mln.", "mr.", "n.v.", "nr.", "o.a.", "o.m.", "pct.", "prof.", "resp.",
    "st.", "t.o.v.",
})


def abbreviations() -> set:
    """The course's list of abbreviations, as a new set each time."""
    return set(ABBREVIATIONS)


def every_mark(word, abbreviations):
    """The simplest rule: every word that ends with . ? or ! ends a sentence."""
    return re.search(r"[.?!]$", word) is not None


def _words_and_ends() -> tuple:
    """Every word of Alpino cut at spaces, the sentences one after another, as (word, True for the last word of a
    sentence)."""
    out = []
    for _, text, _ in treebank("nl"):
        ws = re.findall(r"\S+", text)
        out += [(x, i == len(ws) - 1) for i, x in enumerate(ws)]
    return tuple(out)


def words_and_ends() -> tuple:
    return kept("nlp-tokenize:words-ends", _words_and_ends)


def score(says: list) -> dict:
    """How a rule's answers, one per word of words_and_ends(), compare with the real ends: ends found, false ends,
    missed ends, and the words of the false and of the missed ends."""
    found = false = missed = 0
    false_words, missed_words = {}, {}
    for (word, end), said in zip(words_and_ends(), says):
        if said and end:
            found += 1
        elif said:
            false += 1
            false_words[word] = false_words.get(word, 0) + 1
        elif end:
            missed += 1
            missed_words[word] = missed_words.get(word, 0) + 1
    return {"found": found, "false": false, "missed": missed, "false_words": false_words, "missed_words": missed_words}


def rule_score(rule) -> dict:
    return score([rule(x, ABBREVIATIONS) for x, _ in words_and_ends()])


def every_mark_score() -> dict:
    return kept("nlp-tokenize:every-mark", lambda: rule_score(every_mark))


def missed_kinds(missed_words: dict, abbr=ABBREVIATIONS) -> dict:
    """The missed ends by how the last word ends: with a letter or digit (no mark), a colon, a quotation mark, a
    period of an abbreviation or an initial, or something else."""
    out = {"none": 0, "colon": 0, "quote": 0, "abbreviation": 0, "other": 0}
    for word, n in missed_words.items():
        if re.search(r"\w$", word):
            out["none"] += n
        elif word.endswith(":"):
            out["colon"] += n
        elif re.search(r"[\"'”’]$", word):
            out["quote"] += n
        elif word.lower() in abbr or re.fullmatch(r"[A-Z]\.", word):
            out["abbreviation"] += n
        else:
            out["other"] += n
    return out


_END_WORDS = {
    "en": {"title": "Sentence ends in Alpino",
           "head": ["", "ends found", "false ends", "missed", "precision", "recall"],
           "every": "every . ? ! at the end", "yours": "your ends_sentence",
           "line": "The {n} sentences of Alpino, one after another, cut at spaces: {w} words, and {n} real ends.",
           "false": "False ends that are left, most often first: {words}.",
           "missed": "Missed ends: {none} sentences end without a mark, such as a headline, {colon} with a colon, "
                     "{quote} with a quotation mark, and {abbreviation} with an abbreviation or an initial.",
           "stopped": "Your function stopped with an error on one of the words, so the picture stays empty.",
           "slow": "Your function took more than {s:g} seconds for all words, so the picture stays empty.",
           "odd": "Your function does not give back True or False for every word, so the picture stays empty."},
    "nl": {"title": "Zinseinden in Alpino",
           "head": ["", "einden gevonden", "onterechte einden", "gemist", "precision", "recall"],
           "every": "elke . ? ! aan het eind", "yours": "jouw ends_sentence",
           "line": "De {n} zinnen van Alpino achter elkaar, geknipt bij de spaties: {w} woorden, en {n} echte einden.",
           "false": "Onterechte einden die overblijven, de meest voorkomende eerst: {words}.",
           "missed": "Gemiste einden: {none} zinnen eindigen zonder teken, zoals een kop, {colon} met een dubbele punt, "
                     "{quote} met een aanhalingsteken, en {abbreviation} met een afkorting of een initiaal.",
           "stopped": "Je functie stopte met een fout op een van de woorden, dus de afbeelding blijft leeg.",
           "slow": "Je functie deed er langer dan {s:g} seconden over voor alle woorden, dus de afbeelding blijft leeg.",
           "odd": "Je functie geeft niet voor elk woord True of False terug, dus de afbeelding blijft leeg."},
}


def _score_row(name: str, s: dict, lang: str) -> list:
    return [html.escape(name), _num(s["found"], lang), _num(s["false"], lang), _num(s["missed"], lang),
            _pct(s["found"], s["found"] + s["false"], lang), _pct(s["found"], s["found"] + s["missed"], lang)]


def ends_picture(seen) -> str:
    lang = seen.lang
    w = _END_WORDS[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    words = [x for x, _ in words_and_ends()]
    try:
        got = _call(lambda: [fn(x, ABBREVIATIONS) for x in words])
    except Timeout:
        text = w["slow"].format(s=PICTURE_LIMIT)
        return yours.figure(w["title"], _note(text), alt=text)
    if got is _STOPPED:
        return yours.figure(w["title"], _note(w["stopped"]), alt=w["stopped"])
    if not all(isinstance(g, bool) for g in got):
        return yours.figure(w["title"], _note(w["odd"]), alt=w["odd"])
    mine = score(got)
    rows_ = [_score_row(w["every"], every_mark_score(), lang), _score_row(w["yours"], mine, lang)]
    n = n_sentences()
    line = w["line"].format(n=_num(n, lang), w=_num(len(words), lang))
    false = sorted(mine["false_words"].items(), key=lambda kv: (-kv[1], kv[0]))[:6]
    kinds = {k: _num(v, lang) for k, v in missed_kinds(mine["missed_words"]).items()}
    notes = []
    if false:
        notes.append(w["false"].format(words=", ".join(f"{x} ({_num(k, lang)}×)" for x, k in false)))
    notes.append(w["missed"].format(**kinds))
    alt = (w["title"] + ". " + line + " " + "; ".join(f"{r[0]}: {', '.join(r[1:])}" for r in rows_) + ". "
           + " ".join(notes))
    body = (f'<div style="{_LINE}">{html.escape(line)}</div>' + _table(rows_, w["head"], numbers=(1, 2, 3, 4, 5))
            + "".join(_note(t) for t in notes))
    return yours.figure(w["title"], body, alt=alt)


# --- the picture under exercise 3: MaxMatch on sentences of Tatoeba without spaces ------------------------------------


def _dictionary() -> frozenset:
    return frozenset(x.lower() for _, s in books.dutch_sentences() for x in books.words(s))


def dictionary_kept() -> frozenset:
    """Every word of the Dutch sentences of Tatoeba, in small letters: the dictionary of MaxMatch."""
    return kept("nlp-tokenize:dictionary", _dictionary)


def dictionary() -> set:
    """The dictionary as a new set, for the lesson's WORDS."""
    return set(dictionary_kept())


SAMPLE_STEP = 100   # every hundredth sentence of Tatoeba
_PRONOUNS = {"hij", "zij", "haar", "hem", "zijn", "ze"}   # the course names nobody by a gendered pronoun


def _sample() -> tuple:
    out = []
    for _, s in books.dutch_sentences()[::SAMPLE_STEP]:
        words = tuple(x.lower() for x in books.words(s))
        if words:
            out.append(words)
    return tuple(out)


def sample() -> tuple:
    """Every hundredth Dutch sentence of Tatoeba as its words in small letters, without the sentences without words."""
    return kept("nlp-tokenize:sample", _sample)


def match_results(step=longest_start, backward: bool = False) -> list:
    """MaxMatch with `step` (or backward MaxMatch) on every sentence of the sample, run together."""
    words = dictionary_kept()
    if backward:
        return [backward_max_match("".join(ws), words) for ws in sample()]
    return [max_match("".join(ws), words, step) for ws in sample()]


def match_counts() -> tuple:
    """(sentences of the sample cut right forward, cut right backward)."""
    def count():
        fw = sum(1 for got, ws in zip(match_results(), sample()) if got == list(ws))
        bw = sum(1 for got, ws in zip(match_results(backward=True), sample()) if got == list(ws))
        return fw, bw
    return kept("nlp-tokenize:match-counts", count)


def piece_styles(pieces: list, words) -> list:
    """Blue for a piece that is one of the words, at its place; purple for one that is not."""
    starts, at = set(), 0
    for x in words:
        starts.add((at, at + len(x)))
        at += len(x)
    out, at = [], 0
    for p in pieces:
        out.append(_CHIP_RIGHT if (at, at + len(p)) in starts else _CHIP_WRONG)
        at += len(p)
    return out


_MATCH_WORDS = {
    "en": {"title": "MaxMatch with your longest_start",
           "line": "With your longest_start, MaxMatch cuts {k} of {n} sentences of Tatoeba exactly into their words "
                   "({p}).",
           "wrong": "A few of the others, each with its run-together text, MaxMatch's cut, and the words:",
           "note": "The sentences are every hundredth Dutch sentence of Tatoeba, in small letters, without spaces and "
                   "punctuation. The dictionary holds every word of the Dutch sentences of Tatoeba, so no word here is "
                   "missing from it: every purple piece comes from taking the longest word.",
           "stopped": "Your function stopped with an error on one of the sentences, so the picture stays empty.",
           "slow": "MaxMatch with your function took more than {s:g} seconds, so the picture stays empty.",
           "odd": "On one of the sentences, your function gives back an empty string, or something that is not a start "
                  "of the text; MaxMatch can only go on after a start of one character or more. So the picture stays "
                  "empty."},
    "nl": {"title": "MaxMatch met jouw longest_start",
           "line": "Met jouw longest_start knipt MaxMatch {k} van {n} zinnen van Tatoeba precies in hun woorden ({p}).",
           "wrong": "Een paar van de andere, elk met de aan elkaar geschreven tekst, wat MaxMatch knipt, en de woorden:",
           "note": "De zinnen zijn elke honderdste Nederlandse zin van Tatoeba, in kleine letters, zonder spaties en "
                   "leestekens. De dictionary heeft elk woord van de Nederlandse zinnen van Tatoeba, dus er ontbreekt hier "
                   "geen enkel woord in: elk paars stuk komt doordat MaxMatch steeds het langste woord neemt.",
           "stopped": "Je functie stopte met een fout op een van de zinnen, dus de afbeelding blijft leeg.",
           "slow": "MaxMatch met je functie deed er langer dan {s:g} seconden over, dus de afbeelding blijft leeg.",
           "odd": "Op een van de zinnen geeft je functie een lege string terug, of iets dat geen begin van de tekst is; "
                  "MaxMatch kan alleen verder na een begin van een of meer tekens. Dus de afbeelding blijft leeg."},
}
MATCH_SHOWN = 4      # wrong cuts the picture shows
MATCH_LONGEST = 28   # the longest run-together text the picture shows


def match_picture(seen) -> str:
    lang = seen.lang
    w = _MATCH_WORDS[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    texts = ["".join(ws) for ws in sample()]
    words = dictionary_kept()

    def run():
        out = []
        for t in texts:
            try:
                out.append(max_match(t, words, fn))
            except ValueError:
                return None
        return out

    try:
        got = _call(run)
    except Timeout:
        text = w["slow"].format(s=PICTURE_LIMIT)
        return yours.figure(w["title"], _note(text), alt=text)
    if got is _STOPPED:
        return yours.figure(w["title"], _note(w["stopped"]), alt=w["stopped"])
    if got is None:
        return yours.figure(w["title"], _note(w["odd"]), alt=w["odd"])
    right = sum(1 for g, ws in zip(got, sample()) if g == list(ws))
    n = len(texts)
    line = w["line"].format(k=_num(right, lang), n=_num(n, lang), p=_pct(right, n, lang))
    shown = [(t, g, ws) for t, g, ws in zip(texts, got, sample())
             if g != list(ws) and len(t) <= MATCH_LONGEST and not (set(ws) | set(g)) & _PRONOUNS][:MATCH_SHOWN]
    rows_ = "".join(f'<div style="border-bottom:1px solid var(--ea-line);padding:0.2rem 0">'
                    f'<div style="{_CODE};color:var(--ea-ink-2)">{html.escape(t)}</div>'
                    f'<div>{_chips(g, piece_styles(g, ws))}</div>'
                    f'<div>{_chips(list(ws), [_CHIP_DATA] * len(ws))}</div></div>' for t, g, ws in shown)
    alt = (w["title"] + ". " + line + " " + "; ".join(f"{t}: {' '.join(g)} ({' '.join(ws)})" for t, g, ws in shown)
           + ". " + w["note"])
    body = (f'<div style="{_LINE}">{html.escape(line)}</div>'
            + (f'<div style="{_LABEL}">{html.escape(w["wrong"])}</div>{rows_}' if shown else "") + _note(w["note"]))
    return yours.figure(w["title"], body, alt=alt)


# --- the demo: MaxMatch forward and backward on a text of your own ----------------------------------------------------

DEMO_LONGEST = 60
_DEMO_WORDS = {
    "en": {"title": "MaxMatch on your text",
           "forward": "Forward, the longest word at the start: {n} pieces",
           "backward": "Backward, the longest word at the end: {n} pieces",
           "same": "Both directions give the same cut.",
           "empty": "Type a text in the box; spaces are taken out.",
           "long": "This text is longer than {k} characters without its spaces; try a shorter one.",
           "note": "The dictionary: the {n} words of the Dutch sentences of Tatoeba, in small letters. Purple: a piece "
                   "that is not in it, one character that MaxMatch took because no word fit.",
           "alt": "Your text cut by MaxMatch forward and backward."},
    "nl": {"title": "MaxMatch op jouw tekst",
           "forward": "Vooruit, het langste woord aan het begin: {n} stukken",
           "backward": "Achteruit, het langste woord aan het eind: {n} stukken",
           "same": "Beide richtingen knippen hetzelfde.",
           "empty": "Typ een tekst in het vak; spaties gaan eruit.",
           "long": "Deze tekst is zonder spaties langer dan {k} tekens; probeer een kortere.",
           "note": "De dictionary: de {n} woorden van de Nederlandse zinnen van Tatoeba, in kleine letters. Paars: een "
                   "stuk dat er niet in staat, één teken dat MaxMatch nam omdat er geen woord paste.",
           "alt": "Je tekst geknipt door MaxMatch, vooruit en achteruit."},
}


def demo_text(text) -> str:
    """What the demo cuts: the text in small letters, without white space."""
    return re.sub(r"\s+", "", str(text or "")).lower()


def demo_picture(text, lang: str) -> str:
    w = _DEMO_WORDS[lang]
    t = demo_text(text)
    title = w["title"]
    if not t:
        return yours.figure(title, _note(w["empty"]), alt=w["empty"])
    if len(t) > DEMO_LONGEST:
        msg = w["long"].format(k=DEMO_LONGEST)
        return yours.figure(title, _note(msg), alt=msg)
    words = dictionary_kept()
    rows_, alts = [], []
    cuts = (max_match(t, words), backward_max_match(t, words))
    for key, pieces in zip(("forward", "backward"), cuts):
        label = w[key].format(n=len(pieces))
        styles = [_CHIP_RIGHT if p in words else _CHIP_WRONG for p in pieces]
        rows_.append(f'<div style="{_LABEL}">{html.escape(label)}</div>{_chips(pieces, styles)}')
        alts.append(label + ": " + " | ".join(pieces))
    notes = ([w["same"]] if cuts[0] == cuts[1] else []) + [w["note"].format(n=_num(len(words), lang))]
    body = f'<div style="{_CODE};color:var(--ea-ink-2)">{html.escape(t)}</div>' + "".join(rows_) + "".join(
        _note(x) for x in notes)
    return yours.figure(title, body, alt=w["alt"] + " " + "; ".join(alts) + ". " + " ".join(notes))


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["tokens", "ends-sentence", "longest-start", "which-fix"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: "..." in "\n".join(
                    line.split("#")[0] for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang het door je eigen code.")


def _value(v) -> str:
    """A short value the way the lessons write it: strings in double quotes, also inside a list or a set."""
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    if isinstance(v, list):
        return "[" + ", ".join(_value(p) for p in v) + "]"
    if isinstance(v, (set, frozenset)):
        return "{" + ", ".join(_value(p) for p in sorted(v)) + "}" if v else "set()"
    if isinstance(v, re.Match):
        return f"a match of {_value(v.group())}"
    return repr(v)


_MATCH = {"en": "a match of {g}", "nl": "een match van {g}"}


def _show(v):
    """{got} and {want}: a match of re.search by what it matched, in the language of the lesson."""
    if isinstance(v, re.Match):
        return {lang: _MATCH[lang].format(g=_value(v.group())) for lang in _MATCH}
    return _value(v)


class Named:
    """A value of the lesson that a message shows by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


class _Lazy:
    """A value of the lesson made when a check first needs it, and shown in a message by its name. `make` gives a new
    copy, or a value that cannot be changed, such as a frozenset."""

    def __init__(self, name: str, make):
        self.name, self.make = name, make

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return self.make()


def _lit(v) -> Named:
    """A value of the check's own, shown in a message as Python, with strings in double quotes."""
    return Named(_value(v), v)


SENTENCE_N = _Lazy("SENTENCE", cities)
ABBREVIATIONS_N = Named("ABBREVIATIONS", ABBREVIATIONS)
WORDS_N = _Lazy("WORDS", dictionary_kept)   # a frozenset: the learner's code gets it as it is, without a copy of 38,682 words


def _a_match(got, text) -> bool:
    return isinstance(got, re.Match)


def _listed(got, text) -> bool:
    """The second choice is a set of a few marks, such as [.,?!], not a set with ^."""
    return re.search(r"\|\s*\[(?!\^)", text) is not None


def _empty_string(got, text) -> bool:
    return got == ""


# exercise 1: tokens. Besides the sentence of the three cities: a clock time, an apostrophe, two marks in a row, an
# ellipsis, letters with accents, brackets, extra spaces, a newline, and an empty text.
TOKEN_CASES = [
    (SENTENCE_N,),
    (_lit("Ik kom om 8.30 uur."),),
    (_lit("zo'n auto"),),
    (_lit("Wat?!"),),
    (_lit("Nou..."),),
    (_lit("naïef café"),),
    (_lit("(30 gulden)"),),
    (_lit("  twee   spaties  "),),
    (_lit("Eén.\nTwee."),),
    (_lit(""),),
]
_HEAD_RE = "import re\n\n"
_T_START = _HEAD_RE + "def tokens(text):\n    ...\n"
_T = _HEAD_RE + "def tokens(text):\n    return re.findall(r\"\\w+|[^\\w\\s]\", text)\n"
_T_SPACES = _T.replace("\\w+|[^\\w\\s]", "\\S+")
_T_WORDS = _T.replace("\\w+|[^\\w\\s]", "\\w+")
_T_RUNS = _T.replace("[^\\w\\s]", "[^\\w\\s]+")
_T_LETTERS = _T.replace("\\w+|", "\\w|")
_T_NON_WORD = _T.replace("[^\\w\\s]", "\\W")
_T_ASCII = _T.replace("\\w+|", "[A-Za-z]+|")
_T_SPACE_ONLY = _T.replace("[^\\w\\s]", "[^\\w ]")
_T_APART = (_HEAD_RE + "def tokens(text):\n"
            "    return re.findall(r\"\\w+\", text) + re.findall(r\"[^\\w\\s]\", text)\n")

# exercise 2: ends_sentence. Besides "drs.": a word at the end of a sentence, an initial, an abbreviation with a
# capital, a word in capitals at the end, a question, an exclamation, a number with a period inside, a word without a
# mark, and a word with an empty list of abbreviations.
END_CASES = [
    (_lit("drs."), ABBREVIATIONS_N),
    (_lit("geworden."), ABBREVIATIONS_N),
    (_lit("J."), ABBREVIATIONS_N),
    (_lit("Dr."), ABBREVIATIONS_N),
    (_lit("NAVO."), ABBREVIATIONS_N),
    (_lit("kwam?"), ABBREVIATIONS_N),
    (_lit("Ja!"), ABBREVIATIONS_N),
    (_lit("10.000"), ABBREVIATIONS_N),
    (_lit("benoeming"), ABBREVIATIONS_N),
    (_lit("O.a."), ABBREVIATIONS_N),
    (_lit("mr."), _lit(set())),
]
_E_START = _HEAD_RE + "def ends_sentence(word, abbreviations):\n    ...\n"
_E = (_HEAD_RE + "def ends_sentence(word, abbreviations):\n"
      "    if word.lower() in abbreviations or re.search(r\"^[A-Z]\\.$\", word):\n"
      "        return False\n"
      "    return re.search(r\"[.?!]$\", word) is not None\n")
_E_MATCH = _E.replace(" is not None\n", "\n")
_E_NO_LOWER = _E.replace("word.lower() in", "word in")
_E_NO_INITIAL = _E.replace(" or re.search(r\"^[A-Z]\\.$\", word)", "")
_E_NO_LIST = _E.replace("word.lower() in abbreviations or ", "")
_E_ONLY_DOT = _E.replace("r\"[.?!]$\"", "r\"\\.$\"")
_E_NO_END = _E.replace("r\"[.?!]$\"", "r\"[.?!]\"")
_E_INITIAL_ANYWHERE = _E.replace("r\"^[A-Z]\\.$\"", "r\"[A-Z]\\.\"")
_E_ORDER = (_HEAD_RE + "def ends_sentence(word, abbreviations):\n"
            "    if re.search(r\"[.?!]$\", word):\n"
            "        return True\n"
            "    if word.lower() in abbreviations or re.search(r\"^[A-Z]\\.$\", word):\n"
            "        return False\n"
            "    return False\n")

# exercise 3: longest_start. Besides "zoterug": a whole sentence, a text that is a word itself, a word with a shorter
# word inside, a text that starts with no word, a text of three letters, a long sentence, a euro sign, and one letter.
START_CASES = [
    (_lit("zoterug"), WORDS_N),
    (_lit("ikbenzoterug"), WORDS_N),
    (_lit("theta"), _lit({"t", "the", "theta"})),
    (_lit("thetabled"), _lit({"t", "the", "theta", "bled"})),
    (_lit("xyz"), _lit({"ab", "xz"})),
    (_lit("kat"), _lit({"k", "ka", "kat"})),
    (_lit("schrijfalstublieftmeteenpen"), WORDS_N),
    (_lit("€5"), WORDS_N),
    (_lit("a"), _lit({"a"})),
]
_L_START = "def longest_start(text, words):\n    ...\n"
_L = ("def longest_start(text, words):\n"
      "    for end in range(len(text), 0, -1):\n"
      "        if text[:end] in words:\n"
      "            return text[:end]\n"
      "    return text[:1]\n")
_L_SHORTEST = _L.replace("range(len(text), 0, -1)", "range(1, len(text) + 1)")
_L_ONE_SHORT = _L.replace("range(len(text), 0, -1)", "range(len(text) - 1, 0, -1)")
_L_WHOLE = _L.replace("    return text[:1]\n", "    return text\n")
_L_ENDS = _L.replace("text[:end]", "text[end:]")

EXERCISES = {
    "tokens": FunctionBox(
        name="tokens",
        picture=tokens_picture,
        function="tokens",
        start=_T_START,
        solution=_T,
        visible=TOKEN_CASES[:1],
        hidden=TOKEN_CASES[1:],
        first=[_DOTS],
        show=_show,
        mistakes=[
            Mistake("spaces", wrong=_T_SPACES,
                    en="`{call}` gives {got}, and it should give {want}: `\\S+` takes everything between two spaces, so a comma or a question mark stays stuck to the word before it. Give the pattern a second choice with `|`: one punctuation mark on its own.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `\\S+` neemt alles tussen twee spaties, dus een komma of een vraagteken blijft aan het woord ervoor vastzitten. Geef het patroon een tweede keuze met `|`: één leesteken los."),
            Mistake("words-only", wrong=_T_WORDS,
                    en="`{call}` gives {got}, and it should give {want}: your pattern finds the words, and the punctuation marks fall out. Each punctuation mark is a token too: add a second choice with `|`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je patroon vindt de woorden, en de leestekens vallen weg. Elk leesteken is ook een token: voeg een tweede keuze toe met `|`."),
            Mistake("runs", wrong=_T_RUNS,
                    en="`{call}` gives {got}, and it should give {want}: the `+` after your second choice takes a run of punctuation marks as one token, such as `?!`. Each punctuation mark is a token of its own.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de `+` na je tweede keuze neemt een reeks leestekens als één token, zoals `?!`. Elk leesteken is een eigen token."),
            Mistake("letters", wrong=_T_LETTERS,
                    en="`{call}` gives {got}, and it should give {want}: without a `+`, `\\w` takes one letter at a time, so every letter becomes a token. Let the first choice take a whole run of letters and digits.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: zonder `+` neemt `\\w` één letter tegelijk, dus elke letter wordt een token. Laat de eerste keuze een hele reeks letters en cijfers nemen."),
            Mistake("non-word", wrong=_T_NON_WORD,
                    en="`{call}` gives {got}, and it should give {want}: `\\W` is every character that `\\w` does not take, and a space is one of them, so your tokens hold spaces. Leave white space out of the second choice.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `\\W` is elk teken dat `\\w` niet neemt, en een spatie is er een van, dus je tokens bevatten spaties. Laat witruimte buiten de tweede keuze."),
            Mistake("space-only", wrong=_T_SPACE_ONLY,
                    en="`{call}` gives {got}, and it should give {want}: your second choice leaves out a space, but not a newline or a tab, so a newline becomes a token. `\\s` stands for every kind of white space.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je tweede keuze laat een spatie weg, maar geen regeleinde of tab, dus een regeleinde wordt een token. `\\s` staat voor elke soort witruimte."),
            Mistake("listed", matches=_listed,
                    en="`{call}` gives {got}, and it should give {want}: your second choice lists a few punctuation marks, so a mark that is not on your list falls out, such as an apostrophe or a bracket. Describe that character by what it is not: no letter or digit, and no white space.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je tweede keuze somt een paar leestekens op, dus een leesteken dat niet in je rijtje staat, valt weg, zoals een apostrof of een haakje. Beschrijf dat teken door wat het niet is: geen letter of cijfer, en geen witruimte."),
            Mistake("ascii", wrong=_T_ASCII,
                    en="`{call}` gives {got}, and it should give {want}: `[A-Za-z]` takes only the letters from a to z, so digits fall out, and letters with an accent too: *naïef* comes apart, and *café* becomes *caf*. `\\w` also takes digits and letters with an accent.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `[A-Za-z]` neemt alleen de letters van a tot z, dus cijfers vallen weg, en letters met een accent ook: *naïef* valt uit elkaar, en *café* wordt *caf*. `\\w` neemt ook cijfers en letters met een accent."),
            Mistake("apart", wrong=_T_APART,
                    en="`{call}` gives {got}, and it should give {want}: the right tokens, but first all words and then all punctuation marks. One `re.findall` with one pattern with two choices keeps them in the order of the text.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de goede tokens, maar eerst alle woorden en dan alle leestekens. Eén `re.findall` met één patroon met twee keuzes houdt ze in de volgorde van de tekst."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back the tokens of `text` with `re.findall` and one pattern with two choices: a run of letters and digits, or one punctuation mark.",
                      "nl": "Vervang de `...`: geef de tokens van `text` terug met `re.findall` en één patroon met twee keuzes: een reeks letters en cijfers, of één leesteken."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your pattern take a whole run of letters and digits, or else one character that is not a letter, a digit or white space?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Neemt je patroon een hele reeks letters en cijfers, of anders één teken dat geen letter, geen cijfer en geen witruimte is?"},
            "almost": {"en": "Right for SENTENCE. But `{call}` gives {got}, and it should give {want}. Does your function work for any text, also two marks in a row, letters with accents, or a newline?",
                       "nl": "Goed voor SENTENCE. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke tekst, ook met twee tekens achter elkaar, letters met accenten, of een regeleinde?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back. Does every way through it end with `return`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt elke weg erdoorheen met `return`?"},
            "correct": {"en": "Your `tokens` is right in all {n} cases. Look at the picture: your function on every sentence of the Dutch treebank, and what the treebank cuts differently.",
                        "nl": "Je `tokens` klopt in alle {n} gevallen. Kijk naar de afbeelding: je functie op elke zin van de Nederlandse treebank, en wat de treebank anders knipt."},
        },
    ),
    "ends-sentence": FunctionBox(
        name="ends-sentence",
        picture=ends_picture,
        function="ends_sentence",
        start=_E_START,
        solution=_E,
        visible=END_CASES[:1],
        hidden=END_CASES[1:],
        first=[_DOTS],
        show=_show,
        mistakes=[
            Mistake("match", matches=_a_match,
                    en="`{call}` gives {got}, and it should give {want}: `re.search` gives a match or `None`, not `True` or `False`. Compare what it gives with `None`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `re.search` geeft een match of `None`, geen `True` of `False`. Vergelijk wat het geeft met `None`."),
            Mistake("no-list", wrong=_E_NO_LIST,
                    en="`{call}` gives {got}, and it should give {want}: your function does not look in `abbreviations`, so the period of an abbreviation ends a sentence. Give back `False` first when the word is in that set.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie kijkt niet in `abbreviations`, dus de punt van een afkorting beëindigt een zin. Geef eerst `False` terug als het woord in die set staat."),
            Mistake("no-initial", wrong=_E_NO_INITIAL,
                    en="`{call}` gives {got}, and it should give {want}: an initial such as *J.* does not end a sentence, and your function treats it as an end. Give back `False` for a word that is one capital letter with a period, too.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een initiaal zoals *J.* is geen einde van een zin, en je functie ziet het als een einde. Geef ook `False` terug voor een woord dat één hoofdletter met een punt is."),
            Mistake("no-lower", wrong=_E_NO_LOWER,
                    en="`{call}` gives {got}, and it should give {want}: `abbreviations` holds small letters, and at the start of a sentence an abbreviation such as *Dr.* has a capital. Look up the word in small letters.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `abbreviations` bevat kleine letters, en aan het begin van een zin heeft een afkorting zoals *Dr.* een hoofdletter. Zoek het woord op in kleine letters."),
            Mistake("initial-anywhere", wrong=_E_INITIAL_ANYWHERE,
                    en="`{call}` gives {got}, and it should give {want}: your pattern for an initial also matches the end of a longer word, such as the *O.* of *NAVO.*, so that real end is lost. An initial is the whole word.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je patroon voor een initiaal matcht ook het eind van een langer woord, zoals de *O.* van *NAVO.*, dus dat echte einde gaat verloren. Een initiaal is het hele woord."),
            Mistake("only-dot", wrong=_E_ONLY_DOT,
                    en="`{call}` gives {got}, and it should give {want}: your pattern takes only a period, and a question mark or an exclamation mark ends a sentence too. Let the last character be one of the three.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je patroon neemt alleen een punt, en een vraagteken of een uitroepteken beëindigt ook een zin. Laat het laatste teken een van de drie zijn."),
            Mistake("no-end", wrong=_E_NO_END,
                    en="`{call}` gives {got}, and it should give {want}: your pattern finds a period anywhere in the word, also inside a number such as *10.000*. Only the last character counts: tie the pattern to the end.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je patroon vindt een punt overal in het woord, ook in een getal zoals *10.000*. Alleen het laatste teken telt: maak het patroon vast aan het eind."),
            Mistake("order", wrong=_E_ORDER,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back `True` for every word with a mark at the end before it looks at the abbreviations and the initials. Look at those first.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft `True` terug voor elk woord met een teken aan het eind, nog voor de afkortingen en de initialen aan de beurt zijn. Kijk eerst naar die twee."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back `False` for an abbreviation or an initial, and otherwise whether the word ends with `.`, `?` or `!`.",
                      "nl": "Vervang de `...`: geef `False` terug voor een afkorting of een initiaal, en anders of het woord eindigt op `.`, `?` of `!`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function give back `False` for an abbreviation in `abbreviations`, also with a capital, and for an initial, and otherwise `True` only when the last character is `.`, `?` or `!`?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Geeft je functie `False` terug voor een afkorting in `abbreviations`, ook met een hoofdletter, en voor een initiaal, en anders alleen `True` als het laatste teken `.`, `?` of `!` is?"},
            "almost": {"en": "Right for \"drs.\". But `{call}` gives {got}, and it should give {want}. Does your function work for every word: an initial, an abbreviation with a capital, a question, a number?",
                       "nl": "Goed voor \"drs.\". Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elk woord: een initiaal, een afkorting met een hoofdletter, een vraag, een getal?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back. Does every way through it end with `return`, also for a word without a mark at the end?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt elke weg erdoorheen met `return`, ook voor een woord zonder teken aan het eind?"},
            "correct": {"en": "Your `ends_sentence` is right in all {n} cases. Look at the picture: your function on every word of the Dutch treebank, against the rule that every mark at the end ends a sentence.",
                        "nl": "Je `ends_sentence` klopt in alle {n} gevallen. Kijk naar de afbeelding: je functie op elk woord van de Nederlandse treebank, naast de regel dat elk teken aan het eind een zin beëindigt."},
        },
    ),
    "longest-start": FunctionBox(
        name="longest-start",
        picture=match_picture,
        function="longest_start",
        start=_L_START,
        solution=_L,
        visible=START_CASES[:1],
        hidden=START_CASES[1:],
        first=[_DOTS],
        show=_show,
        mistakes=[
            Mistake("shortest", wrong=_L_SHORTEST,
                    en="`{call}` gives {got}, and it should give {want}: your loop tries the shortest start first, so it stops at the first word it meets. MaxMatch wants the longest: let the loop count down.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je loop probeert eerst het kortste begin, en stopt dus bij het eerste woord. MaxMatch wil het langste: laat de loop aftellen."),
            Mistake("one-short", wrong=_L_ONE_SHORT,
                    en="`{call}` gives {got}, and it should give {want}: your loop never tries the whole text itself, so a text that is one word gets cut short. Where does the count of `range` start?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je loop probeert nooit de hele tekst zelf, dus een tekst die één woord is, wordt te kort geknipt. Waar begint de telling van `range`?"),
            Mistake("empty", matches=_empty_string,
                    en="`{call}` gives {got}, and it should give {want}: when no start is a word, MaxMatch takes one character, so that the text gets shorter. An empty string would leave it as it is. Give back the first character.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: als geen begin een woord is, neemt MaxMatch één teken, zodat de tekst korter wordt. Met een lege string blijft de tekst even lang. Geef het eerste teken terug."),
            Mistake("whole", wrong=_L_WHOLE,
                    en="`{call}` gives {got}, and it should give {want}: when no start is a word, MaxMatch takes one character, not the whole text. Give back the first character.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: als geen begin een woord is, neemt MaxMatch één teken, niet de hele tekst. Geef het eerste teken terug."),
            Mistake("ends", wrong=_L_ENDS,
                    en="`{call}` gives {got}, and it should give {want}: `text[end:]` is the end of the text from place `end` on. The start up to `end` is `text[:end]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `text[end:]` is het eind van de tekst vanaf plaats `end`. Het begin tot `end` is `text[:end]`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: try the starts of `text` from the longest to the shortest, give back the first one that is in `words`, and after the loop the first character.",
                      "nl": "Vervang de `...`: probeer elk begin van `text`, van het langste tot het kortste, geef het eerste begin terug dat in `words` staat, en na de loop het eerste teken."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function try the longest start first, give back the first start that is in `words`, and else the first character?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Probeert je functie het langste begin eerst, en geeft je functie het eerste begin terug dat in `words` staat, en anders het eerste teken?"},
            "almost": {"en": "Right for \"zoterug\". But `{call}` gives {got}, and it should give {want}. Does your function work for every text, also one that is a word itself, or one that starts with no word?",
                       "nl": "Goed voor \"zoterug\". Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke tekst, ook een die zelf een woord is, of een die met geen enkel woord begint?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back: no start of that text is in `words`, and after the loop your function ends without `return`. Give back the first character then.",
                          "nl": "Met `{call}` geeft je functie niets terug: geen begin van die tekst staat in `words`, en na de loop eindigt je functie zonder `return`. Geef dan het eerste teken terug."},
            "correct": {"en": "Your `longest_start` is right in all {n} cases. Look at the picture: the course's MaxMatch calls your function again and again, on sentences of Tatoeba without spaces.",
                        "nl": "Je `longest_start` klopt in alle {n} gevallen. Kijk naar de afbeelding: de MaxMatch van de course roept je functie steeds opnieuw aan, op zinnen van Tatoeba zonder spaties."},
        },
    ),
    "which-fix": Sort(
        name="which-fix",
        groups={
            "list": {"en": "A list of exceptions: words that look like something else",
                     "nl": "Een lijst met uitzonderingen: woorden die op iets anders lijken"},
            "pattern": {"en": "A pattern for the form: characters in a fixed order",
                        "nl": "Een patroon voor de vorm: tekens in een vaste volgorde"},
            "dictionary": {"en": "A dictionary and the longest match: words without spaces between them",
                           "nl": "Een dictionary en de langste match: woorden zonder spaties ertussen"},
        },
        items={
            "bijv": {"en": "**Keep *bijv.* together, and end no sentence after it.**",
                     "nl": "**Houd *bijv.* bij elkaar, en laat er geen zin na eindigen.**"},
            "t": {"en": "**Keep *'t* in *'t Is laat.* together as one token.**",
                  "nl": "**Houd *'t* in *'t Is laat.* bij elkaar als één token.**"},
            "pct": {"en": "**End no sentence after *pct.* in *3 pct. meer*.**",
                    "nl": "**Laat geen zin eindigen na *pct.* in *3 pct. meer*.**"},
            "number": {"en": "**Keep *10.000* together as one number.**",
                       "nl": "**Houd *10.000* bij elkaar als één getal.**"},
            "email": {"en": "**Keep an e-mail address such as *info@voorbeeld.nl* together.**",
                      "nl": "**Houd een e-mailadres zoals *info@voorbeeld.nl* bij elkaar.**"},
            "time": {"en": "**Keep the clock time *8.30* together.**",
                     "nl": "**Houd de kloktijd *8.30* bij elkaar.**"},
            "hashtag": {"en": "**Cut the hashtag *#ikbenzoterug* into words.**",
                        "nl": "**Knip de hashtag *#ikbenzoterug* in woorden.**"},
            "chinese": {"en": "**Cut the Chinese sentence 姚明进入总决赛 into words.**",
                        "nl": "**Knip de Chinese zin 姚明进入总决赛 in woorden.**"},
        },
        solution={"bijv": "list", "t": "list", "pct": "list", "number": "pattern", "email": "pattern",
                  "time": "pattern", "hashtag": "dictionary", "chinese": "dictionary"},
        nudge={
            "bijv": {"en": "Could a rule tell *bijv.* from a word at the end of a sentence, without knowing this word?",
                     "nl": "Kan een regel *bijv.* onderscheiden van een woord aan het eind van een zin, zonder dit woord te kennen?"},
            "t": {"en": "An apostrophe also opens a quotation, as in *'hallo'*: could a rule tell the two apart without knowing these few short words?",
                  "nl": "Een apostrof opent ook een citaat, zoals in *'hallo'*: kan een regel die twee uit elkaar houden zonder deze paar korte woorden te kennen?"},
            "pct": {"en": "Does *pct.* look different from *dag.* at the end of a sentence?",
                    "nl": "Ziet *pct.* er anders uit dan *dag.* aan het eind van een zin?"},
            "number": {"en": "Do all numbers written like this look alike: digits, a period, three digits?",
                       "nl": "Zien alle getallen die zo geschreven zijn er hetzelfde uit: cijfers, een punt, drie cijfers?"},
            "email": {"en": "What does every e-mail address have, in the same place?",
                      "nl": "Wat heeft elk e-mailadres, op dezelfde plaats?"},
            "time": {"en": "Which characters make up a clock time, and in what order?",
                     "nl": "Uit welke tekens bestaat een kloktijd, en in welke volgorde?"},
            "hashtag": {"en": "What shows where one word stops and the next one starts?",
                        "nl": "Wat laat zien waar het ene woord stopt en het volgende begint?"},
            "chinese": {"en": "Where are the spaces in this sentence?",
                        "nl": "Waar staan de spaties in deze zin?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the cases again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de gevallen opnieuw uit."},
            "empty": {"en": "Put each bold case under the fix that it needs.",
                      "nl": "Zet elk vetgedrukt geval bij de oplossing die het nodig heeft."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item}: this case goes somewhere else.",
                      "nl": "{item}: dit geval hoort ergens anders."},
            "correct": {"en": "All {n} are right. A list holds words that look like something else, a pattern catches a form that comes back in many strings, and where there are no spaces, a dictionary says which words there are.",
                        "nl": "Alle {n} kloppen. Een lijst bevat woorden die op iets anders lijken, een patroon vangt een vorm die in veel strings terugkomt, en waar geen spaties staan, zegt een dictionary welke woorden er zijn."},
        },
    ),
}

CASES = [
    ("tokens", _T_START, "empty", None),
    ("tokens", _T, "correct", None),
    ("tokens", _HEAD_RE + "def tokens(text):\n    found = re.findall(r\"[\\w]+|[^\\s\\w]\", text)\n    return found\n", "correct", None),
    ("tokens", _HEAD_RE + "def tokens(text):\n    return re.findall(r\"\\w+|[^\\w\\s]{1}\", text)\n", "correct", None),
    ("tokens", _T_SPACES, "wrong", "spaces"),
    ("tokens", _T_WORDS, "wrong", "words-only"),
    ("tokens", _T_RUNS, "wrong", "runs"),
    ("tokens", _T_LETTERS, "wrong", "letters"),
    ("tokens", _T_NON_WORD, "wrong", "non-word"),
    ("tokens", _T_SPACE_ONLY, "wrong", "space-only"),
    ("tokens", _T_ASCII, "wrong", "ascii"),
    ("tokens", _T_APART, "wrong", "apart"),
    ("tokens", _T.replace("[^\\w\\s]", "[.,?!]"), "wrong", "listed"),
    ("tokens", _T.replace("[^\\w\\s]", "\\S"), "correct", None),
    ("tokens", _T.replace("    return re.findall", "    re.findall"), "wrong", "no-return"),
    ("tokens", _HEAD_RE + "def tokens(text):\n    return [\"Maar\", \"waar\", \"blijven\", \"Amsterdam\", \",\", \"Rotterdam\", \"en\", \"Den\", \"Haag\", \"?\"]\n", "almost", None),
    ("ends-sentence", _E_START, "empty", None),
    ("ends-sentence", _E, "correct", None),
    ("ends-sentence", _HEAD_RE + "def ends_sentence(word, abbreviations):\n    if word.lower() in abbreviations:\n        return False\n    if re.fullmatch(r\"[A-Z]\\.\", word):\n        return False\n    return re.search(r\"(\\.|\\?|!)$\", word) is not None\n", "correct", None),
    ("ends-sentence", _HEAD_RE + "def ends_sentence(word, abbreviations):\n    if word.lower() not in abbreviations and not re.search(r\"^[A-Z]\\.$\", word):\n        if re.search(r\"[.!?]$\", word):\n            return True\n    return False\n", "correct", None),
    ("ends-sentence", _E_MATCH, "wrong", "match"),
    ("ends-sentence", _E_NO_LIST, "wrong", "no-list"),
    ("ends-sentence", _E_NO_INITIAL, "wrong", "no-initial"),
    ("ends-sentence", _E_NO_LOWER, "wrong", "no-lower"),
    ("ends-sentence", _E_INITIAL_ANYWHERE, "wrong", "initial-anywhere"),
    ("ends-sentence", _E_ONLY_DOT, "wrong", "only-dot"),
    ("ends-sentence", _E_NO_END, "wrong", "no-end"),
    ("ends-sentence", _E_ORDER, "wrong", "order"),
    ("ends-sentence", _HEAD_RE + "def ends_sentence(word, abbreviations):\n    return False\n", "almost", None),
    ("longest-start", _L_START, "empty", None),
    ("longest-start", _L, "correct", None),
    ("longest-start", "def longest_start(text, words):\n    for end in range(len(text), 1, -1):\n        if text[:end] in words:\n            return text[:end]\n    return text[0]\n", "correct", None),
    ("longest-start", "def longest_start(text, words):\n    best = text[:1]\n    for end in range(1, len(text) + 1):\n        if text[:end] in words:\n            best = text[:end]\n    return best\n", "correct", None),
    ("longest-start", _L_SHORTEST, "wrong", "shortest"),
    ("longest-start", _L_ONE_SHORT, "wrong", "one-short"),
    ("longest-start", _L.replace("    return text[:1]\n", "    return \"\"\n"), "wrong", "empty"),
    ("longest-start", _L_WHOLE, "wrong", "whole"),
    ("longest-start", _L_ENDS, "wrong", "ends"),
    ("longest-start", _L.replace("    return text[:1]\n", ""), "wrong", "unreached"),
    ("longest-start", "def longest_start(text, words):\n    return \"zot\"\n", "almost", None),
    ("which-fix", {}, "empty", None),
    ("which-fix", {"bijv": "list", "t": "list", "pct": "list", "number": "pattern", "email": "pattern",
                   "time": "pattern", "hashtag": "dictionary", "chinese": "dictionary"}, "correct", None),
    ("which-fix", {"bijv": "list", "number": "pattern"}, "wrong", "unfinished"),
    ("which-fix", {"chinese": "pattern"}, "wrong", "wrong:chinese"),
    ("which-fix", {"bijv": "pattern", "t": "list", "pct": "list", "number": "pattern", "email": "pattern",
                   "time": "pattern", "hashtag": "dictionary", "chinese": "dictionary"}, "wrong", "wrong:bijv"),
    ("which-fix", {"bijv": "list", "t": "list", "pct": "list", "number": "pattern", "email": "pattern",
                   "time": "pattern", "hashtag": "list", "chinese": "dictionary"}, "wrong", "wrong:hashtag"),
]
