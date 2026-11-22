# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-language-and-society: count the words of a treebank in standard Dutch (word_counts), work
out which share of the words of a list of sentences a word list does not know (share_unknown), count those unknown
words (unknown_counts), and sort statements about variation in language under region, group or situation.

The word list comes from UD_Dutch-Alpino of Universal Dependencies 2.19 (data/ud-nl-alpino.conllu.gz, made by
data/tools/make_ud.py). The sentences in Frisian, Gronings and Limburgish, each with a Dutch translation, are those of
Tatoeba (CC BY 2.0 FR) in data/tatoeba-nl-pairs.tsv.gz (made by data/tools/make_tatoeba_pairs.py). Every number in the
pictures is counted by the course; the cases of the checks and the statements of the sort are the course's own."""
import contextlib
import copy
import gzip
import html
import io

from everything_ai import books, chart, paths, yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, kept, with_time_limit

# --- the course's solutions ------------------------------------------------------------------------------------------


def word_counts(sentences):
    """The course's solution of Exercise 1: how often each word occurs, in small letters, over every sentence."""
    counts = {}
    for sentence in sentences:
        for word in sentence:
            w = word.lower()
            if w in counts:
                counts[w] = counts[w] + 1
            else:
                counts[w] = 1
    return counts


def share_unknown(sentences, known):
    """The course's solution of Exercise 2: the share of all words of the sentences that `known` does not hold in small
    letters; 0 when there are no words."""
    total = 0
    unknown = 0
    for sentence in sentences:
        total = total + len(sentence)
        for word in sentence:
            if word.lower() not in known:
                unknown = unknown + 1
    if total == 0:
        return 0
    return unknown / total


def unknown_counts(sentences, known):
    """The course's solution of Exercise 3: the counts of the words, in small letters, that `known` does not hold."""
    counts = word_counts(sentences)
    unknown = {}
    for w in counts:
        if w not in known:
            unknown[w] = counts[w]
    return unknown


# --- the data --------------------------------------------------------------------------------------------------------

UD_NL = paths.ROOT / "data" / "ud-nl-alpino.conllu.gz"
VARIETIES = ("fry", "gos", "lim")
VARIETY_NAMES = {"en": {"fry": "Frisian", "gos": "Gronings", "lim": "Limburgish"},
                 "nl": {"fry": "Fries", "gos": "Gronings", "lim": "Limburgs"}}
NAMES = {"fry": ("FRISIAN", "FRISIAN_NL"), "gos": ("GRONINGS", "GRONINGS_NL"), "lim": ("LIMBURGISH", "LIMBURGISH_NL")}


def _read_forms() -> tuple:
    """The sentences of UD_Dutch-Alpino, each a tuple of its word forms, as the treebank writes them."""
    out, words = [], []
    with gzip.open(UD_NL, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if words:
                    out.append(tuple(words))
                words = []
            elif not line.startswith("#"):
                words.append(line.split("\t")[1])
    if words:
        out.append(tuple(words))
    return tuple(out)


def _forms() -> tuple:
    return kept("nlp-language-and-society:alpino", _read_forms)


def treebank() -> list:
    """The Dutch treebank as the exercises take it: a new list of sentences, each a list of words."""
    return [list(s) for s in _forms()]


def _counts() -> dict:
    return kept("nlp-language-and-society:counts", lambda: word_counts(_forms()))


def counts() -> dict:
    """The course's word counts of the Dutch treebank, as a new dict: what word_counts(ALPINO) gives."""
    return dict(_counts())


def _split(lang: str) -> tuple:
    def make():
        rows = [r for r in books.dutch_pairs() if r[0] == lang]
        return (tuple(tuple(books.words(r[2])) for r in rows), tuple(tuple(books.words(r[4])) for r in rows),
                tuple(r[1:] for r in rows))
    return kept(f"nlp-language-and-society:pairs:{lang}", make)


def sentences(lang: str) -> tuple:
    """The sentences of a variety of Tatoeba and their Dutch translations, as two new lists of lists of words, in the
    same order: sentence k of the first list says what sentence k of the second says."""
    variety, dutch, _ = _split(lang)
    return [list(s) for s in variety], [list(s) for s in dutch]


def rows(lang: str) -> tuple:
    """(id, text, nl_id, nl_text) of every pair of a variety, in the order of sentences()."""
    return _split(lang)[2]


# --- small helpers for the pictures ----------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums"
_MISS = ("color:var(--ea-loss-text);font-weight:600;text-decoration:underline wavy var(--ea-loss);"
         "text-underline-offset:0.18em")
_LINE = "font-family:var(--ea-font-ui);font-size:0.9rem;color:var(--ea-ink);line-height:1.6"
_TAG = "font-family:var(--ea-font-ui);font-size:0.75rem;color:var(--ea-ink-3);margin-right:0.4rem"


def _note(text: str) -> str:
    """A line under a picture; a word between asterisks is set in italics, as in the lesson's text."""
    parts = html.escape(text).split("*")
    marked = "".join(f"<i>{p}</i>" if k % 2 else p for k, p in enumerate(parts))
    return f'<div style="{_UI}">{marked}</div>'


def _num(n, lang: str) -> str:
    """A whole number with a separator for thousands: 2,319 in English, 2.319 in Dutch."""
    text = f"{round(n):,}"
    return text.replace(",", ".") if lang == "nl" else text


def _pct(x: float, lang: str) -> str:
    text = f"{100 * x:.1f}"
    return (text.replace(".", ",") if lang == "nl" else text) + "%"


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


def _svg(panels: list, alt: str, lang: str, width: int = 330, height: int = 250) -> str:
    svg = chart.figure(panels, alt=alt, width=width, height=height)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


def marked(text: str, known) -> str:
    """A sentence as HTML, with the words that `known` does not hold in small letters underlined in purple, the way a
    spelling checker marks them."""
    out, at = [], 0
    for m in books._WORD.finditer(text):
        out.append(html.escape(text[at:m.start()]))
        word = html.escape(m.group())
        out.append(word if m.group().lower() in known else f'<span style="{_MISS}">{word}</span>')
        at = m.end()
    out.append(html.escape(text[at:]))
    return "".join(out)


def _pair_html(lang_code: str, text: str, nl_text: str, known, lang: str) -> str:
    name = VARIETY_NAMES[lang][lang_code]
    dutch = {"en": "Dutch", "nl": "Nederlands"}[lang]
    return (f'<div style="{_LINE};margin:0.35rem 0"><div><span style="{_TAG}">{html.escape(name)}</span>'
            f'{marked(text, known)}</div><div><span style="{_TAG}">{dutch}</span>{marked(nl_text, known)}</div></div>')


def _unknown_in(text: str, known) -> tuple:
    words = books.words(text)
    return sum(1 for w in words if w.lower() not in known), len(words)


# --- the idea's picture: two sentences and their translations --------------------------------------------------------

# (variety, Tatoeba id of the sentence): a proverb in Frisian, a question in Gronings
IDEA_PAIRS = [("fry", 382301), ("gos", 8031978)]
_IDEA_WORDS = {
    "en": {"title": "What a word list of standard Dutch knows",
           "note": "Underlined: not in the word list of the Dutch treebank of the course. Sentences of Tatoeba: "
                   "{fry} of {fry_n} words unknown in Frisian, {fry_nl} in its Dutch translation; {gos} of {gos_n} "
                   "in Gronings, {gos_nl} in Dutch.",
           "alt": "Two sentences with their Dutch translations, the words the word list does not know underlined."},
    "nl": {"title": "Wat een woordenlijst van standaardnederlands kent",
           "note": "Onderstreept: niet in de woordenlijst van de Nederlandse treebank van de course. Zinnen van "
                   "Tatoeba: {fry} van {fry_n} woorden onbekend in het Fries, {fry_nl} in de Nederlandse vertaling; "
                   "{gos} van {gos_n} in het Gronings, {gos_nl} in het Nederlands.",
           "alt": "Twee zinnen met hun Nederlandse vertaling, de woorden die de woordenlijst niet kent onderstreept."},
}


def _row(lang_code: str, sid: int) -> tuple:
    for r in rows(lang_code):
        if r[0] == sid:
            return r
    raise KeyError(sid)


def idea_picture(lang: str) -> str:
    w = _IDEA_WORDS[lang]
    known = _counts()
    body, numbers = [], {}
    for code, sid in IDEA_PAIRS:
        _, text, _, nl_text = _row(code, sid)
        body.append(_pair_html(code, text, nl_text, known, lang))
        u, n = _unknown_in(text, known)
        numbers[code], numbers[code + "_n"], numbers[code + "_nl"] = u, n, _unknown_in(nl_text, known)[0]
    note = w["note"].format(**numbers)
    return yours.figure(w["title"], "".join(body) + _note(note), alt=w["alt"] + " " + note)


# --- running the learner's function ----------------------------------------------------------------------------------


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error


def _call(fn, *args):
    """The learner's function within the time limit of a check; _STOPPED when it stops with an error, and the Timeout
    passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(lambda: fn(*(copy.deepcopy(a) for a in args)))
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


# --- the picture under exercise 1: words that are standard Dutch in Belgium, beside the usual words ---------------

# (a word of standard Dutch, a word that is standard Dutch in Belgium only, for the same thing), by Taaladvies of the
# Taalunie: ziekenhuis, leuk and schoon are standard in the whole language area, and hospitaal, plezant and proper (in
# the sense of clean) also in Belgium; bankkaart beside pinpas (standard in the Netherlands), and burgemeester en
# schepenen (in Belgium) beside burgemeester en wethouders (in the Netherlands)
WORD_PAIRS = [("ziekenhuis", "hospitaal"), ("leuk", "plezant"), ("schoon", "proper"), ("pinpas", "bankkaart"),
              ("wethouder", "schepen")]
_COUNT_WORDS = {
    "en": {"title": "Belgian standard words in your counts",
           "head": ["word", "times", "in Belgium", "times"],
           "size": "Your counts hold {n} different words, from {t} words of the treebank. Each word in the third "
                   "column is standard language in Belgium, by Taaladvies.net; of the first column, ziekenhuis, leuk and "
                   "schoon are standard in the whole language area, pinpas and wethouder in the Netherlands.",
           "stopped": "Your function stopped with an error on the treebank, so the picture stays empty.",
           "odd": "Your function does not give back a dict of counts, so the picture stays empty."},
    "nl": {"title": "Belgische standaardwoorden in je telling",
           "head": ["woord", "keer", "in België", "keer"],
           "size": "Je telling heeft {n} verschillende woorden, uit {t} woorden van de treebank. Elk woord in de "
                   "derde kolom is standaardtaal in België, volgens Taaladvies.net; uit de eerste kolom zijn ziekenhuis, "
                   "leuk en schoon standaardtaal in het hele taalgebied, pinpas en wethouder in Nederland.",
           "stopped": "Je functie stopte met een fout op de treebank, dus de afbeelding blijft leeg.",
           "odd": "Je functie geeft geen dictionary met aantallen terug, dus de afbeelding blijft leeg."},
}


def _count_of(got: dict, word: str) -> int:
    v = got.get(word, 0)
    return v if isinstance(v, int) and not isinstance(v, bool) else 0


def counts_picture(seen) -> str:
    lang = seen.lang
    w = _COUNT_WORDS[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    got = _call(fn, _forms())
    if got is _STOPPED:
        return yours.figure(w["title"], _note(w["stopped"]), alt=w["stopped"])
    if not isinstance(got, dict):
        return yours.figure(w["title"], _note(w["odd"]), alt=w["odd"])
    rows_, styles = [], {}
    for i, (nl_word, be_word) in enumerate(WORD_PAIRS):
        a, b = _count_of(got, nl_word), _count_of(got, be_word)
        rows_.append([html.escape(nl_word), _num(a, lang), html.escape(be_word), _num(b, lang)])
        if b == 0:
            styles[(i, 3)] = ";" + _MISS
    size = w["size"].format(n=_num(len(got), lang), t=_num(sum(len(s) for s in _forms()), lang))
    alt = w["title"] + ". " + "; ".join(f"{r[0]} {r[1]}, {r[2]} {r[3]}" for r in rows_) + ". " + size
    return yours.figure(w["title"], _table(rows_, w["head"], numbers=(1, 3), styles=styles) + _note(size), alt=alt)


# --- the picture under exercise 2: the shares of unknown words -------------------------------------------------------

_SHARE_WORDS = {
    "en": {"title": "Unknown words: a variety and its translation",
           "panel": "words not in the list", "y": "% unknown",
           "key": "filled: the sentences in the variety; open: their Dutch translations",
           "note": "With your share_unknown and the course's counts of the treebank. Sentence pairs of Tatoeba: "
                   "{fry} in Frisian, {gos} in Gronings, {lim} in Limburgish.",
           "stopped": "Your function stopped with an error on the sentences of Tatoeba, so the picture stays empty.",
           "odd": "Your function does not give back a number, so the picture stays empty."},
    "nl": {"title": "Onbekende woorden: een variëteit en de vertaling",
           "panel": "woorden niet in de lijst", "y": "% onbekend",
           "key": "gevuld: de zinnen in de variëteit; open: de Nederlandse vertalingen",
           "note": "Met jouw share_unknown en de telling van de treebank van de course. Zinsparen van Tatoeba: "
                   "{fry} in het Fries, {gos} in het Gronings, {lim} in het Limburgs.",
           "stopped": "Je functie stopte met een fout op de zinnen van Tatoeba, dus de afbeelding blijft leeg.",
           "odd": "Je functie geeft geen getal terug, dus de afbeelding blijft leeg."},
}


def _a_share(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def share_picture(seen) -> str:
    lang = seen.lang
    w = _SHARE_WORDS[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    known = _counts()
    values = []
    for code in VARIETIES:
        variety, dutch, _ = _split(code)
        a, b = _call(fn, variety, known), _call(fn, dutch, known)
        if a is _STOPPED or b is _STOPPED:
            return yours.figure(w["title"], _note(w["stopped"]), alt=w["stopped"])
        if not (_a_share(a) and _a_share(b)):
            return yours.figure(w["title"], _note(w["odd"]), alt=w["odd"])
        values.append((code, a, b))
    top = max(max(a, b) for _, a, b in values)
    top = max(top * 100, 1) * 1.2
    names = [VARIETY_NAMES[lang][c] for c, _, _ in values]
    p = chart.Panel((0.4, len(values) + 0.6), (0, top), title=w["panel"], ylabel=w["y"],
                    xticks=list(range(1, len(values) + 1)), xtick_labels=names)
    for i, (code, a, b) in enumerate(values, start=1):
        p.rect(i - 0.34, 0, i - 0.02, 100 * a, role="loss", solid=True)
        p.rect(i + 0.02, 0, i + 0.34, 100 * b, role="loss")
        p.text(i - 0.18, 100 * a + top * 0.04, _pct(a, lang).rstrip("%"), role="ink-2")
        p.text(i + 0.18, 100 * b + top * 0.04, _pct(b, lang).rstrip("%"), role="ink-2")
    n = {c: _num(len(_split(c)[0]), lang) for c in VARIETIES}
    note = w["note"].format(**n)
    pairs = "; ".join(f"{name} {_pct(a, lang)}, Dutch {_pct(b, lang)}" if lang == "en" else
                      f"{name} {_pct(a, lang)}, Nederlands {_pct(b, lang)}" for name, (_, a, b) in zip(names, values))
    alt = f'{w["title"]}. {pairs}. {w["key"]}. {note}'
    return yours.figure(w["title"], _svg([p], alt, lang, width=460, height=250) + _note(w["key"]) + _note(note),
                        alt=alt)


# --- the picture under exercise 3: the most common unknown words -----------------------------------------------------

_TOP_WORDS = {
    "en": {"title": "The words the list misses most often",
           "head": ["word", "times"], "dutch": "Dutch, beside Gronings",
           "note": "With your unknown_counts and the course's counts of the treebank, on all sentences of Tatoeba in "
                   "Frisian and Gronings and on the Dutch translations of the Gronings ones.",
           "stopped": "Your function stopped with an error on the sentences of Tatoeba, so the picture stays empty.",
           "odd": "Your function does not give back a dict of counts, so the picture stays empty."},
    "nl": {"title": "De woorden die de lijst het vaakst mist",
           "head": ["woord", "keer"], "dutch": "Nederlands, naast Gronings",
           "note": "Met jouw unknown_counts en de telling van de treebank van de course, op alle zinnen van Tatoeba in "
                   "het Fries en het Gronings en op de Nederlandse vertalingen van de Groningse.",
           "stopped": "Je functie stopte met een fout op de zinnen van Tatoeba, dus de afbeelding blijft leeg.",
           "odd": "Je functie geeft geen dictionary met aantallen terug, dus de afbeelding blijft leeg."},
}


def top(counted: dict, k: int = 8) -> list:
    """The k words with the highest counts, the highest first, and words with the same count in alphabetical order."""
    good = [(word, n) for word, n in counted.items()
            if isinstance(word, str) and isinstance(n, int) and not isinstance(n, bool)]
    return sorted(good, key=lambda wn: (-wn[1], wn[0]))[:k]


def unknown_picture(seen) -> str:
    lang = seen.lang
    w = _TOP_WORDS[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    known = _counts()
    columns = [(VARIETY_NAMES[lang]["fry"], _split("fry")[0]), (VARIETY_NAMES[lang]["gos"], _split("gos")[0]),
               (w["dutch"], _split("gos")[1])]
    blocks = []
    for name, sents in columns:
        got = _call(fn, sents, known)
        if got is _STOPPED:
            return yours.figure(w["title"], _note(w["stopped"]), alt=w["stopped"])
        if not isinstance(got, dict):
            return yours.figure(w["title"], _note(w["odd"]), alt=w["odd"])
        best = top(got)
        table = _table([[html.escape(word), _num(n, lang)] for word, n in best], w["head"], numbers=(1,))
        blocks.append((name, best, f'<div><div style="{_HEAD};border:none;padding-left:0.45rem">{html.escape(name)}'
                                   f'</div>{table}</div>'))
    body = ('<div style="display:flex;flex-wrap:wrap;gap:0.4rem 1.6rem">' + "".join(b for _, _, b in blocks) + "</div>")
    alt = w["title"] + ". " + "; ".join(f"{name}: " + ", ".join(f"{word} {_num(n, lang)}" for word, n in best)
                                        for name, best, _ in blocks) + ". " + w["note"]
    return yours.figure(w["title"], body + _note(w["note"]), alt=alt)


# --- the demo: sentence pairs with the unknown words marked ----------------------------------------------------------

DEMO_SIZE = 6
_DEMO_WORDS = {
    "en": {"title": "{name} and Dutch, pairs {a} to {b} of {n}",
           "note": "In these {k} pairs, the list misses {u} of {t} words in {name} and {u_nl} of {t_nl} in Dutch. In "
                   "all {n} pairs: {share} against {share_nl}.",
           "alt": "Sentence pairs in {name} and Dutch, with the words the word list does not know underlined."},
    "nl": {"title": "{name} en Nederlands, paren {a} tot en met {b} van {n}",
           "note": "In deze {k} paren mist de lijst {u} van {t} woorden in het {name} en {u_nl} van {t_nl} in het "
                   "Nederlands. In alle {n} paren: {share} tegen {share_nl}.",
           "alt": "Zinsparen in het {name} en het Nederlands, met de woorden die de woordenlijst niet kent "
                  "onderstreept."},
}


def variety_options(lang: str) -> dict:
    """The varieties for the demo's dropdown: name -> code, in the order of VARIETIES."""
    return {VARIETY_NAMES[lang][c]: c for c in VARIETIES}


def pair_count(code: str) -> int:
    return len(rows(code))


def last_start(code: str) -> int:
    """The highest start of the demo's slider for a variety: the first of its last six pairs."""
    return max(1, pair_count(code) - DEMO_SIZE + 1)


def demo_picture(code: str, start: int, lang: str) -> str:
    w = _DEMO_WORDS[lang]
    known = _counts()
    all_rows = rows(code)
    n = len(all_rows)
    a = max(1, min(int(start), n - DEMO_SIZE + 1))   # a start near the end still shows the last six pairs
    chosen = all_rows[a - 1:a - 1 + DEMO_SIZE]
    name = VARIETY_NAMES[lang][code]
    body = "".join(_pair_html(code, text, nl_text, known, lang) for _, text, _, nl_text in chosen)
    u = t = u_nl = t_nl = 0
    for _, text, _, nl_text in chosen:
        x, y = _unknown_in(text, known)
        x_nl, y_nl = _unknown_in(nl_text, known)
        u, t, u_nl, t_nl = u + x, t + y, u_nl + x_nl, t_nl + y_nl
    variety, dutch, _ = _split(code)
    note = w["note"].format(k=len(chosen), u=u, t=t, u_nl=u_nl, t_nl=t_nl, name=name if lang == "en" else name,
                            n=_num(n, lang), share=_pct(share_unknown(variety, known), lang),
                            share_nl=_pct(share_unknown(dutch, known), lang))
    title = w["title"].format(name=name, a=_num(a, lang), b=_num(a - 1 + len(chosen), lang), n=_num(n, lang))
    return yours.figure(title, body + _note(note), alt=w["alt"].format(name=name) + " " + note)


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["word-counts", "share-unknown", "unknown-counts", "region-group-situation"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: "..." in "\n".join(
                    line.split("#")[0] for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _value(v) -> str:
    """A short value the way the lessons write it: strings in double quotes, also inside a tuple, a list or a dict."""
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, tuple):
        return "(" + ", ".join(_value(p) for p in v) + ("," if len(v) == 1 else "") + ")"
    if isinstance(v, list):
        return "[" + ", ".join(_value(p) for p in v) + "]"
    if isinstance(v, (set, frozenset)):
        return "{" + ", ".join(_value(p) for p in sorted(v, key=str)) + "}" if v else "set()"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_value(k)}: {_value(x)}" for k, x in v.items()) + "}"
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".") if v != int(v) else f"{v:.1f}"
    return repr(v)


_MORE = {"en": "{shown}, and {k} more", "nl": "{shown}, en nog {k}"}


def _show(v):
    """{got} and {want}: a dict of counts from the highest count down, and a long dict, set or list cut short, with how
    many more it holds, in the language of the lesson."""
    if isinstance(v, dict) and len(v) > 6:
        best = top(v, 6)
        shown = "{" + ", ".join(f"{_value(k)}: {_value(n)}" for k, n in best) + ", ...}"
        return {lang: _MORE[lang].format(shown=shown, k=_num(len(v) - len(best), lang)) for lang in _MORE}
    if isinstance(v, dict) and all(isinstance(n, int) for n in v.values()):
        return _value(dict(top(v, len(v)))) if len(top(v, len(v))) == len(v) else _value(v)
    if isinstance(v, (set, frozenset, list)) and len(v) > 6:
        items = sorted(v, key=str)[:6] if not isinstance(v, list) else list(v)[:6]
        open_, close = ("[", "]") if isinstance(v, list) else ("{", "}")
        shown = open_ + ", ".join(_value(x) for x in items) + ", ..." + close
        return {lang: _MORE[lang].format(shown=shown, k=_num(len(v) - 6, lang)) for lang in _MORE}
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
    """A value of the lesson made when a check first needs it, given as a new copy each time, and shown in a message
    by its name."""

    def __init__(self, name: str, make):
        self.name, self.make = name, make

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return self.make()


def _lit(v) -> Named:
    """A value of the check's own, shown in a message as Python, with strings in double quotes."""
    return Named(_value(v), v)


ALPINO_N = _Lazy("ALPINO", treebank)
COUNTS_N = _Lazy("COUNTS", counts)
FRISIAN_N = _Lazy("FRISIAN", lambda: sentences("fry")[0])
FRISIAN_NL_N = _Lazy("FRISIAN_NL", lambda: sentences("fry")[1])
GRONINGS_N = _Lazy("GRONINGS", lambda: sentences("gos")[0])

# exercise 1: word_counts. Besides the treebank: two short sentences with a word in capitals, one word three times in
# three ways of writing it, a sentence without words, and no sentences.
WORD_CASES = [
    (ALPINO_N,),
    (_lit([["De", "kat", "slaapt"], ["de", "hond", "slaapt", "niet"]]),),
    (_lit([["Kat", "kat", "KAT"]]),),
    (_lit([["fan", "de", "beam"], [], ["Fan", "it", "hûs"]]),),
    (_lit([[]]),),
    ([],),
]
_W_START = "def word_counts(sentences):\n    ...\n"
_W = ("def word_counts(sentences):\n"
      "    counts = {}\n"
      "    for sentence in sentences:\n"
      "        for word in sentence:\n"
      "            w = word.lower()\n"
      "            if w in counts:\n"
      "                counts[w] = counts[w] + 1\n"
      "            else:\n"
      "                counts[w] = 1\n"
      "    return counts\n")
_W_ONE = _W.replace("                counts[w] = counts[w] + 1\n", "                counts[w] = 1\n")
_W_INSIDE = _W.replace("    counts = {}\n    for sentence in sentences:\n",
                       "    for sentence in sentences:\n        counts = {}\n")
_W_CAPITALS = _W.replace("            w = word.lower()\n", "            w = word\n")


def _a_list(got, text) -> bool:
    return isinstance(got, list)


def _a_set(got, text) -> bool:
    return isinstance(got, (set, frozenset))


# exercise 2: share_unknown. Besides the Frisian sentences and their translations: words in capitals, a word that is
# known, a sentence without words, no sentences, and a long sentence with a short one, where the share of all words
# and the mean of the shares per sentence differ.
SMALL = {"de", "kat", "slaapt", "appel", "valt", "niet", "ver", "van", "boom"}
SHARE_CASES = [
    (FRISIAN_N, COUNTS_N),
    (FRISIAN_NL_N, COUNTS_N),
    (_lit([["De", "kat", "slaapt"], ["De", "poes", "sliep"]]), _lit(SMALL)),
    (_lit([["De", "appel", "falt", "net", "fier", "fan", "de", "beam"], ["Wolkom"]]), _lit(SMALL)),
    (_lit([["Kat"]]), _lit({"kat"})),
    (_lit([[], ["hûs"]]), _lit({"huis"})),
    ([], _lit(SMALL)),
]
_S_START = "def share_unknown(sentences, known):\n    ...\n"
_S = ("def share_unknown(sentences, known):\n"
      "    total = 0\n"
      "    unknown = 0\n"
      "    for sentence in sentences:\n"
      "        total = total + len(sentence)\n"
      "        for word in sentence:\n"
      "            if word.lower() not in known:\n"
      "                unknown = unknown + 1\n"
      "    if total == 0:\n"
      "        return 0\n"
      "    return unknown / total\n")
_S_CAPITALS = _S.replace("if word.lower() not in known:", "if word not in known:")
_S_MEAN = ("def share_unknown(sentences, known):\n"
           "    shares = []\n"
           "    for sentence in sentences:\n"
           "        if len(sentence) > 0:\n"
           "            unknown = 0\n"
           "            for word in sentence:\n"
           "                if word.lower() not in known:\n"
           "                    unknown = unknown + 1\n"
           "            shares.append(unknown / len(sentence))\n"
           "    if len(shares) == 0:\n"
           "        return 0\n"
           "    return sum(shares) / len(shares)\n")
_S_KNOWN = _S.replace("if word.lower() not in known:", "if word.lower() in known:")


def _zero_division(e) -> bool:
    return isinstance(e, ZeroDivisionError)


def _percent(got, text) -> bool:
    return _a_share(got) and got > 1.5


def _whole(got, text) -> bool:
    return isinstance(got, int) and not isinstance(got, bool) and got > 1


# exercise 3: unknown_counts. Besides the Frisian sentences: words in capitals, a word that comes back, known words
# only, a sentence without words, and no sentences.
UNKNOWN_CASES = [
    (FRISIAN_N, COUNTS_N),
    (GRONINGS_N, COUNTS_N),
    (_lit([["De", "appel", "falt", "net", "fier", "fan", "de", "beam"], ["Fan", "hûs", "net"]]), _lit(SMALL)),
    (_lit([["De", "kat", "slaapt"]]), _lit(SMALL)),
    (_lit([["Wolkom"], [], ["wolkom", "WOLKOM"]]), _lit(SMALL)),
    ([], _lit(SMALL)),
]
_U_START = "def unknown_counts(sentences, known):\n    ...\n"
_U = ("def unknown_counts(sentences, known):\n"
      "    counts = word_counts(sentences)\n"
      "    unknown = {}\n"
      "    for w in counts:\n"
      "        if w not in known:\n"
      "            unknown[w] = counts[w]\n"
      "    return unknown\n")
_U_ALL = _U.replace("        if w not in known:\n            unknown[w] = counts[w]\n",
                    "        unknown[w] = counts[w]\n")
_U_KNOWN = _U.replace("if w not in known:", "if w in known:")
_U_ONES = _U.replace("unknown[w] = counts[w]", "unknown[w] = 1")


class CountsArguments(TypeError):
    """The course's word_counts, as the check gives it to unknown_counts, got something other than a list of
    sentences."""


def _given_word_counts(sentences):
    if not isinstance(sentences, (list, tuple)) or any(not isinstance(s, (list, tuple)) for s in sentences):
        raise CountsArguments("word_counts takes a list of sentences, each a list of words")
    return word_counts(sentences)


def _counts_arguments(e) -> bool:
    return isinstance(e, CountsArguments)


def _same_counts(a, b) -> bool:
    """Two dicts of counts are the same when they hold the same words with the same counts, in any order."""
    return isinstance(a, dict) and isinstance(b, dict) and dict(a) == dict(b)


EXERCISES = {
    "word-counts": FunctionBox(
        name="word-counts",
        picture=counts_picture,
        function="word_counts",
        start=_W_START,
        solution=_W,
        visible=WORD_CASES[:1],
        hidden=WORD_CASES[1:],
        first=[_DOTS],
        show=_show,
        compare=_same_counts,
        mistakes=[
            Mistake("a-set", matches=_a_set,
                    en="`{call}` gives a set: a word list, but without the counts. Give back a dict from each word to how often it occurs: {want}.",
                    nl="`{call}` geeft een set: een woordenlijst, maar zonder de aantallen. Geef een dictionary terug van elk woord naar hoe vaak het voorkomt: {want}."),
            Mistake("a-list", matches=_a_list,
                    en="`{call}` gives a list, and it should give a dict from each word to how often it occurs: {want}.",
                    nl="`{call}` geeft een lijst, en dat moet een dictionary zijn van elk woord naar hoe vaak het voorkomt: {want}."),
            Mistake("capitals", wrong=_W_CAPITALS,
                    en="`{call}` gives {got}, and it should give {want}: your counts keep *De* and *de* apart. Count every word in small letters.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telling houdt *De* en *de* uit elkaar. Tel elk woord in kleine letters."),
            Mistake("one", wrong=_W_ONE,
                    en="`{call}` gives {got}, and it should give {want}: every count stays at 1. What has to happen to the count of a word you meet again?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elk aantal blijft 1. Wat moet er gebeuren met het aantal van een woord dat je opnieuw tegenkomt?"),
            Mistake("inside", wrong=_W_INSIDE,
                    en="`{call}` gives {got}, and it should give {want}: your counts start again at every sentence, so only the last sentence is left. Make the dict once, before the loops.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telling begint bij elke zin opnieuw, dus alleen de laatste zin blijft over. Maak de dictionary één keer, vóór de loops."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: go through every word of every sentence, and count in a dict how often each word occurs, in small letters.",
                      "nl": "Vervang de `...`: ga elk woord van elke zin langs, en tel in een dictionary hoe vaak elk woord voorkomt, in kleine letters."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function count every word of every sentence, in small letters?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Telt je functie elk woord van elke zin, in kleine letters?"},
            "almost": {"en": "Right for ALPINO. But `{call}` gives {got}, and it should give {want}. Does your function work for any list of sentences, also a short one, one with an empty sentence, or an empty list?",
                       "nl": "Goed voor ALPINO. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke lijst zinnen, ook een korte, een met een lege zin, of een lege lijst?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back: it has a `return`, but Python does not get to it. Is the `return` the last line, after both loops?",
                          "nl": "Met `{call}` geeft je functie niets terug: er staat een `return` in, maar Python komt er niet aan toe. Is de `return` de laatste regel, na beide loops?"},
            "correct": {"en": "Your `word_counts` is right in all {n} cases. Look at the picture: words that are standard Dutch in Belgium, beside the words for the same thing, in your counts of the treebank.",
                        "nl": "Je `word_counts` klopt in alle {n} gevallen. Kijk naar de afbeelding: woorden die in België standaardtaal zijn, naast de woorden voor hetzelfde, in je telling van de treebank."},
        },
    ),
    "share-unknown": FunctionBox(
        name="share-unknown",
        picture=share_picture,
        function="share_unknown",
        start=_S_START,
        solution=_S,
        visible=SHARE_CASES[:1],
        hidden=SHARE_CASES[1:],
        first=[_DOTS],
        show=_show,
        mistakes=[
            Mistake("zero", error=_zero_division,
                    en="With `{call}`, your function divides by 0: there are no words at all. Then give back 0, before you divide.",
                    nl="Met `{call}` deelt je functie door 0: er zijn helemaal geen woorden. Geef dan 0 terug, voordat je deelt."),
            Mistake("whole", matches=_whole,
                    en="`{call}` gives {got}, and it should give {want}: the number of unknown words, and not yet their share. Divide by the number of all words.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het aantal onbekende woorden, en nog niet het deel. Deel door het aantal van alle woorden."),
            Mistake("percent", matches=_percent,
                    en="`{call}` gives {got}, and it should give {want}: a share between 0 and 1, not a percentage. Leave out the 100.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een deel tussen 0 en 1, geen percentage. Laat de 100 weg."),
            Mistake("capitals", wrong=_S_CAPITALS,
                    en="`{call}` gives {got}, and it should give {want}: a word with a capital, such as *De* at the start of a sentence, counts as unknown. The list holds its words in small letters, so look each word up in small letters.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een woord met een hoofdletter, zoals *De* aan het begin van een zin, telt als onbekend. De lijst heeft de woorden in kleine letters, dus zoek elk woord op in kleine letters."),
            Mistake("known", wrong=_S_KNOWN,
                    en="`{call}` gives {got}, and it should give {want}: the share of the words the list knows. Count the words that are not in `known`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het deel van de woorden die de lijst wel kent. Tel de woorden die niet in `known` staan."),
            Mistake("mean", wrong=_S_MEAN,
                    en="`{call}` gives {got}, and it should give {want}: your function takes the mean of the shares per sentence, so a sentence of one word weighs as much as a long one. Count over all words together, and divide once, at the end.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie neemt het gemiddelde van de delen per zin, dus een zin van één woord weegt even zwaar als een lange. Tel over alle woorden samen, en deel één keer, aan het eind."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: count all words of the sentences, and the words that are not in `known` in small letters, and give back the share of the second count in the first.",
                      "nl": "Vervang de `...`: tel alle woorden van de zinnen, en de woorden die in kleine letters niet in `known` staan, en geef het deel van het tweede aantal in het eerste terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function count all words of all sentences, look each one up in small letters, and divide the unknown ones by all?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Telt je functie alle woorden van alle zinnen, zoekt de functie elk woord op in kleine letters, en deelt de functie de onbekende door alle?"},
            "almost": {"en": "Right for FRISIAN. But `{call}` gives {got}, and it should give {want}. Does your function work for any sentences and any word list, also with words in capitals, an empty sentence, or no sentences at all?",
                       "nl": "Goed voor FRISIAN. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor alle zinnen en elke woordenlijst, ook met woorden in hoofdletters, een lege zin, of helemaal geen zinnen?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back. With no words at all, give back 0; and does every way through the function end with a `return`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Zonder woorden geef je 0 terug; en eindigt elke weg door de functie met een `return`?"},
            "correct": {"en": "Your `share_unknown` is right in all {n} cases. Look at the picture: three varieties of Tatoeba against their Dutch translations.",
                        "nl": "Je `share_unknown` klopt in alle {n} gevallen. Kijk naar de afbeelding: drie variëteiten van Tatoeba tegenover hun Nederlandse vertalingen."},
        },
    ),
    "unknown-counts": FunctionBox(
        name="unknown-counts",
        picture=unknown_picture,
        function="unknown_counts",
        start=_U_START,
        solution=_U,
        visible=UNKNOWN_CASES[:1],
        hidden=UNKNOWN_CASES[1:],
        first=[_DOTS],
        names={"word_counts": _given_word_counts},
        show=_show,
        compare=_same_counts,
        mistakes=[
            Mistake("counts-arguments", error=_counts_arguments,
                    en="With `{call}`, your function calls `word_counts` with something other than a list of sentences. Call it as `word_counts(sentences)`.",
                    nl="Met `{call}` roept je functie `word_counts` aan met iets anders dan een lijst zinnen. Roep de functie aan als `word_counts(sentences)`."),
            Mistake("a-list", matches=_a_list,
                    en="`{call}` gives a list, and it should give a dict from each unknown word to how often it occurs: {want}.",
                    nl="`{call}` geeft een lijst, en dat moet een dictionary zijn van elk onbekend woord naar hoe vaak het voorkomt: {want}."),
            Mistake("all", wrong=_U_ALL,
                    en="`{call}` gives {got}, and it should give {want}: your dict holds every word, also the words the list knows. Keep only the words that are not in `known`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je dictionary heeft elk woord, ook de woorden die de lijst kent. Houd alleen de woorden die niet in `known` staan."),
            Mistake("known", wrong=_U_KNOWN,
                    en="`{call}` gives {got}, and it should give {want}: your dict holds the words the list knows. Keep the words that are not in `known`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je dictionary heeft de woorden die de lijst kent. Houd de woorden die niet in `known` staan."),
            Mistake("ones", wrong=_U_ONES,
                    en="`{call}` gives {got}, and it should give {want}: every word gets the count 1. How often does the word occur in the sentences?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elk woord krijgt het aantal 1. Hoe vaak komt het woord voor in de zinnen?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: count the words of the sentences with `word_counts`, and keep the words that are not in `known`, with their counts, in a new dict.",
                      "nl": "Vervang de `...`: tel de woorden van de zinnen met `word_counts`, en houd de woorden die niet in `known` staan, met hun aantallen, in een nieuwe dictionary."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function keep exactly the words that are not in `known`, each with how often it occurs, in small letters?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Houdt je functie precies de woorden die niet in `known` staan, elk met hoe vaak het voorkomt, in kleine letters?"},
            "almost": {"en": "Right for FRISIAN. But `{call}` gives {got}, and it should give {want}. Does your function work for any sentences, also with words in capitals, only known words, or no sentences at all?",
                       "nl": "Goed voor FRISIAN. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor alle zinnen, ook met woorden in hoofdletters, alleen bekende woorden, of helemaal geen zinnen?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back: it has a `return`, but Python does not get to it. Is the `return` the last line, after the loop?",
                          "nl": "Met `{call}` geeft je functie niets terug: er staat een `return` in, maar Python komt er niet aan toe. Is de `return` de laatste regel, na de loop?"},
            "correct": {"en": "Your `unknown_counts` is right in all {n} cases. Look at the picture: the words the list misses most often in Frisian, in Gronings and in Dutch.",
                        "nl": "Je `unknown_counts` klopt in alle {n} gevallen. Kijk naar de afbeelding: de woorden die de lijst het vaakst mist in het Fries, in het Gronings en in het Nederlands."},
        },
    ),
    "region-group-situation": Sort(
        name="region-group-situation",
        groups={
            "region": {"en": "Region: where the speakers live or grew up",
                       "nl": "Regio: waar de sprekers wonen of opgroeiden"},
            "group": {"en": "Group: the social group the speakers belong to",
                      "nl": "Groep: de sociale groep waar de sprekers bij horen"},
            "situation": {"en": "Situation: the moment, and how formal or careful the talk is",
                          "nl": "Situatie: het moment, en hoe formeel of zorgvuldig er gesproken wordt"},
        },
        items={
            "stores": {"en": "**In 1962, salespeople at Saks, the highest-ranked of three department stores in New York, pronounced the *r* in *fourth floor* more often than salespeople at S. Klein, the lowest-ranked.**",
                       "nl": "**In 1962 spraken verkopers bij Saks, het hoogst aangeschreven van drie warenhuizen in New York, de *r* in *fourth floor* vaker uit dan verkopers bij S. Klein, het laagst aangeschreven.**"},
            "nait": {"en": "**Gronings has *nait* where standard Dutch has *niet*.**",
                     "nl": "**Het Gronings heeft *nait* waar het standaardnederlands *niet* heeft.**"},
            "repeat": {"en": "**Asked to say *fourth floor* once more, the salespeople at S. Klein pronounced the *r* more often than the first time.**",
                       "nl": "**Gevraagd om *fourth floor* nog eens te zeggen, spraken de verkopers bij S. Klein de *r* vaker uit dan de eerste keer.**"},
            "aae": {"en": "**African-American English is spoken across the United States, and its speakers have close ties with communities of African Americans.**",
                    "nl": "**Afro-Amerikaans Engels wordt in de hele Verenigde Staten gesproken, en de sprekers hebben nauwe banden met gemeenschappen van Afro-Amerikanen.**"},
            "bankkaart": {"en": "**A Belgian speaks of a *bankkaart* and a Dutch person of a *pinpas*, and both words are standard language.**",
                          "nl": "**Een Belg heeft het over een *bankkaart*, een Nederlander over een *pinpas*, en beide woorden zijn standaardtaal.**"},
            "tussentaal": {"en": "**Flemish speakers of standard Dutch use tussentaal in more personal or informal moments.**",
                           "nl": "**Vlaamse sprekers van standaardtaal gebruiken tussentaal op persoonlijker of informelere momenten.**"},
        },
        solution={"stores": "group", "nait": "region", "repeat": "situation", "aae": "group", "bankkaart": "region",
                  "tussentaal": "situation"},
        nudge={
            "stores": {"en": "All three stores are in one city, and every answer came in the same way. What differed, then?",
                       "nl": "Alle drie de warenhuizen staan in één stad, en elk antwoord kwam op dezelfde manier. Wat verschilde er dan?"},
            "nait": {"en": "Where do people speak Gronings?",
                     "nl": "Waar spreken mensen Gronings?"},
            "repeat": {"en": "Were these other speakers, or the same ones a few seconds later?",
                       "nl": "Waren dit andere sprekers, of dezelfde, een paar seconden later?"},
            "aae": {"en": "Is this variety tied to one part of the country?",
                    "nl": "Hoort deze variëteit bij één deel van het land?"},
            "bankkaart": {"en": "Both words are standard language. What decides which of the two someone uses?",
                          "nl": "Beide woorden zijn standaardtaal. Wat bepaalt welke van de twee iemand gebruikt?"},
            "tussentaal": {"en": "Do these speakers use tussentaal everywhere and with everyone?",
                           "nl": "Gebruiken deze sprekers tussentaal overal en met iedereen?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the statements again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de uitspraken opnieuw uit."},
            "empty": {"en": "Put each bold statement under what the difference in it goes with.",
                      "nl": "Zet elke vetgedrukte uitspraak bij waar het verschil erin mee samengaat."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} goes somewhere else.",
                      "nl": "{item} hoort ergens anders."},
            "correct": {"en": "All {n} are right. One study can show more than one kind of difference: in Labov's stores, the *r* went with the rank of the store, and among the same salespeople also with how carefully they spoke.",
                        "nl": "Alle {n} kloppen. Eén onderzoek kan meer dan één soort verschil laten zien: in de warenhuizen van Labov hing de *r* samen met de rang van het warenhuis, en bij dezelfde verkopers ook met hoe zorgvuldig ze spraken."},
        },
    ),
}

CASES = [
    ("word-counts", _W_START, "empty", None),
    ("word-counts", _W, "correct", None),
    ("word-counts", "def word_counts(sentences):\n    counts = {}\n    for sentence in sentences:\n        for word in sentence:\n            counts[word.lower()] = counts.get(word.lower(), 0) + 1\n    return counts\n", "correct", None),
    ("word-counts", _W_ONE, "wrong", "one"),
    ("word-counts", _W_INSIDE, "wrong", "inside"),
    ("word-counts", _W_CAPITALS, "wrong", "capitals"),
    ("word-counts", "def word_counts(sentences):\n    known = set()\n    for sentence in sentences:\n        for word in sentence:\n            known.add(word.lower())\n    return known\n", "wrong", "a-set"),
    ("word-counts", "def word_counts(sentences):\n    words = []\n    for sentence in sentences:\n        for word in sentence:\n            words.append(word.lower())\n    return words\n", "wrong", "a-list"),
    ("word-counts", _W.replace("    return counts\n", ""), "wrong", "no-return"),
    ("share-unknown", _S_START, "empty", None),
    ("share-unknown", _S, "correct", None),
    ("share-unknown", "def share_unknown(sentences, known):\n    words = [w.lower() for s in sentences for w in s]\n    if not words:\n        return 0.0\n    return sum(1 for w in words if w not in known) / len(words)\n", "correct", None),
    ("share-unknown", _S_CAPITALS, "wrong", "capitals"),
    ("share-unknown", _S_MEAN, "wrong", "mean"),
    ("share-unknown", _S_KNOWN, "wrong", "known"),
    ("share-unknown", _S.replace("    if total == 0:\n        return 0\n", ""), "error", "zero"),
    ("share-unknown", _S.replace("    return unknown / total\n", "    return 100 * unknown / total\n"), "wrong", "percent"),
    ("share-unknown", _S.replace("    return unknown / total\n", "    return unknown\n"), "wrong", "whole"),
    ("share-unknown", "def share_unknown(sentences, known):\n    return 1568 / 2668\n", "almost", None),
    ("unknown-counts", _U_START, "empty", None),
    ("unknown-counts", _U, "correct", None),
    ("unknown-counts", "def unknown_counts(sentences, known):\n    unknown = {}\n    for sentence in sentences:\n        for word in sentence:\n            w = word.lower()\n            if w not in known:\n                unknown[w] = unknown.get(w, 0) + 1\n    return unknown\n", "correct", None),
    ("unknown-counts", _U_ALL, "wrong", "all"),
    ("unknown-counts", _U_KNOWN, "wrong", "known"),
    ("unknown-counts", _U_ONES, "wrong", "ones"),
    ("unknown-counts", _U.replace("word_counts(sentences)", "word_counts(sentences[0])"), "error", "counts-arguments"),
    ("unknown-counts", "def unknown_counts(sentences, known):\n    found = []\n    for sentence in sentences:\n        for word in sentence:\n            if word.lower() not in known:\n                found.append(word.lower())\n    return found\n", "wrong", "a-list"),
    ("unknown-counts", _U.replace("    return unknown\n", ""), "wrong", "no-return"),
    ("region-group-situation", {}, "empty", None),
    ("region-group-situation", {"stores": "group", "nait": "region", "repeat": "situation", "aae": "group",
                                "bankkaart": "region", "tussentaal": "situation"}, "correct", None),
    ("region-group-situation", {"stores": "group", "nait": "region"}, "wrong", "unfinished"),
    ("region-group-situation", {"tussentaal": "region"}, "wrong", "wrong:tussentaal"),
    ("region-group-situation", {"stores": "group", "nait": "region", "repeat": "group", "aae": "group",
                                "bankkaart": "region", "tussentaal": "situation"}, "wrong", "wrong:repeat"),
    ("region-group-situation", {"stores": "group", "nait": "region", "repeat": "situation", "aae": "region",
                                "bankkaart": "region", "tussentaal": "situation"}, "wrong", "wrong:aae"),
]
