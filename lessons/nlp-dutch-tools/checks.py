# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-dutch-tools: find the separable verbs of a sentence by the lemmas that Alpino gives them
(separable), count how far the particles of a treebank stand from their verbs (gap_counts), and choose a corpus for a
task.

The course's sentences are its own, with lemmas, tags and relations the course gave them as UD_Dutch-Alpino and
UD_English-EWT give them. The treebanks are UD_English-EWT (without the emails of Enron) and UD_Dutch-Alpino of
Universal Dependencies 2.19 (data/ud-en-ewt.conllu.gz and data/ud-nl-alpino.conllu.gz, made by data/tools/make_ud.py).
Every number in the pictures is counted by the course; the cases of the checks are the course's own."""
import collections
import contextlib
import copy
import gzip
import html
import io

from everything_ai import chart, paths, yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, kept, with_time_limit

# --- the course's sentences ------------------------------------------------------------------------------------------

# word i of a sentence is at place i - 1: (form, lemma, tag, head, relation); head 0 is the root
BEL = [("Ik", "ik", "PRON", 2, "nsubj"), ("bel", "op_bellen", "VERB", 0, "root"), ("je", "je", "PRON", 2, "obj"),
       ("morgen", "morgen", "ADV", 2, "advmod"), ("op", "op", "ADP", 2, "compound:prt"),
       (".", ".", "PUNCT", 2, "punct")]                                             # I'll call you tomorrow
OPGELOST = [("We", "we", "PRON", 6, "nsubj"), ("hebben", "hebben", "AUX", 6, "aux"), ("het", "het", "DET", 4, "det"),
            ("probleem", "probleem", "NOUN", 6, "obj"), ("samen", "samen", "ADV", 6, "advmod"),
            ("opgelost", "op_lossen", "VERB", 0, "root"), (".", ".", "PUNCT", 6, "punct")]   # we solved the problem together
PICK = [("Pick", "pick", "VERB", 0, "root"), ("it", "it", "PRON", 1, "obj"), ("up", "up", "ADP", 1, "compound:prt"),
        (".", ".", "PUNCT", 1, "punct")]
# sentences of the checks' own
TERUG = [("Ik", "ik", "PRON", 2, "nsubj"), ("bel", "op_bellen", "VERB", 0, "root"), ("je", "je", "PRON", 2, "obj"),
         ("op", "op", "ADP", 2, "compound:prt"), ("en", "en", "CCONJ", 6, "cc"),
         ("kom", "terug_komen", "VERB", 2, "conj"), ("daarna", "daarna", "ADV", 6, "advmod"),
         ("snel", "snel", "ADJ", 6, "advmod"), ("terug", "terug", "ADV", 6, "compound:prt"),
         (".", ".", "PUNCT", 2, "punct")]                                           # I'll call you and come back soon after
GRONDWET = [("De", "de", "DET", 2, "det"), ("grondwet", "grond_wet", "NOUN", 4, "nsubj:pass"),
            ("is", "zijn", "AUX", 4, "aux:pass"), ("aangepast", "aan_passen", "VERB", 0, "root"),
            (".", ".", "PUNCT", 4, "punct")]                                        # the constitution has been changed
VAAK = [("Dat", "dat", "PRON", 2, "nsubj"), ("komt", "voor_komen", "VERB", 0, "root"),
        ("vaak", "vaak", "ADV", 2, "advmod"), ("voor", "voor", "ADP", 2, "compound:prt"),
        (".", ".", "PUNCT", 2, "punct")]                                            # that happens often
VOORKOMEN = [("Dat", "dat", "PRON", 4, "obj"), ("kun", "kunnen", "AUX", 4, "aux"), ("je", "je", "PRON", 4, "nsubj"),
             ("voorkomen", "voorkomen", "VERB", 0, "root"), (".", ".", "PUNCT", 4, "punct")]   # you can prevent that
DICHT = [("We", "we", "PRON", 2, "nsubj"), ("proberen", "proberen", "VERB", 0, "root"), ("de", "de", "DET", 4, "det"),
         ("deur", "deur", "NOUN", 7, "obj"), ("dicht", "dicht", "ADJ", 7, "compound:prt"),
         ("te", "te", "ADP", 7, "mark"), ("doen", "dicht_doen", "VERB", 2, "xcomp"),
         (".", ".", "PUNCT", 2, "punct")]                                           # we try to close the door
PLAATS = [("De", "de", "DET", 2, "det"), ("les", "les", "NOUN", 3, "nsubj"),
          ("vindt", "plaats_vinden", "VERB", 0, "root"), ("morgen", "morgen", "ADV", 3, "advmod"),
          ("plaats", "plaats", "NOUN", 3, "compound:prt"), (".", ".", "PUNCT", 3, "punct")]   # the lesson takes place tomorrow
BOEK = [("We", "we", "PRON", 2, "nsubj"), ("lezen", "lezen", "VERB", 0, "root"), ("een", "een", "DET", 4, "det"),
        ("boek", "boek", "NOUN", 2, "obj"), (".", ".", "PUNCT", 2, "punct")]       # we read a book
LIGHT = [("Turn", "turn", "VERB", 0, "root"), ("the", "the", "DET", 4, "det"),
         ("kitchen", "kitchen", "NOUN", 4, "compound"), ("light", "light", "NOUN", 1, "obj"),
         ("off", "off", "ADP", 1, "compound:prt"), (".", ".", "PUNCT", 1, "punct")]
DUITS = [("We", "we", "PRON", 2, "nsubj"), ("lezen", "lezen", "VERB", 0, "root"), ("een", "een", "DET", 5, "det"),
         ("Duitstalige", "Duits_talig", "ADJ", 5, "amod"), ("krant", "krant", "NOUN", 2, "obj"),
         (".", ".", "PUNCT", 2, "punct")]                                          # we read a German-language paper
BELME = [("Bel", "op_bellen", "VERB", 0, "root"), ("me", "me", "PRON", 1, "obj"), ("morgen", "morgen", "ADV", 1, "advmod"),
         ("op", "op", "ADP", 1, "compound:prt"), (".", ".", "PUNCT", 1, "punct")]   # call me tomorrow
# a sentence of the treebank (cgn_exs\53) without a full stop: its last word is the particle
TERUGBEL = [("ik", "ik", "PRON", 2, "nsubj"), ("bel", "terug_bellen", "VERB", 0, "root"), ("je", "je", "PRON", 2, "obj"),
            ("morgen", "morgen", "ADV", 2, "advmod"), ("terug", "terug", "ADV", 2, "compound:prt")]   # I'll call you back tomorrow

GLOSS = {"BEL": "I'll call you tomorrow", "OPGELOST": "we solved the problem together",
         "TERUG": "I'll call you and come back soon after", "GRONDWET": "the constitution has been changed",
         "VAAK": "that happens often", "VOORKOMEN": "you can prevent that", "DICHT": "we try to close the door",
         "PLAATS": "the lesson takes place tomorrow", "BOEK": "we read a book",
         "DUITS": "we read a German-language paper", "BELME": "call me tomorrow",
         "TERUGBEL": "I'll call you back tomorrow"}


def separable(words):
    """The course's solution of Exercise 1: the verbs whose lemma joins a particle to them, written as one word."""
    found = []
    for form, lemma, tag, head, relation in words:
        if tag == "VERB" and "_" in lemma:
            found.append(lemma.replace("_", ""))
    return found


def gap_counts(sentences):
    """The course's solution of Exercise 2: for every gap from a verb to its particle, how many particles stand there."""
    counts = {}
    for words in sentences:
        for i in range(1, len(words) + 1):
            form, lemma, tag, head, relation = words[i - 1]
            if relation == "compound:prt":
                gap = i - head
                if gap in counts:
                    counts[gap] = counts[gap] + 1
                else:
                    counts[gap] = 1
    return counts


def particles(words):
    """The worked example: (particle, verb) for every word with the relation compound:prt."""
    found = []
    for form, lemma, tag, head, relation in words:
        if relation == "compound:prt":
            found.append((form, words[head - 1][0]))
    return found


# --- the treebanks ---------------------------------------------------------------------------------------------------

UD = {"en": paths.ROOT / "data" / "ud-en-ewt.conllu.gz", "nl": paths.ROOT / "data" / "ud-nl-alpino.conllu.gz"}
TREEBANK_NAMES = {"en": "UD_English-EWT", "nl": "UD_Dutch-Alpino"}


def _read(path) -> tuple:
    """The sentences of a CoNLL-U file of the course: (sent_id, text, words), each word a tuple (form, lemma, tag,
    head, relation)."""
    out, words, sent_id, text = [], [], None, ""
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
                words, text = [], ""
            elif not line.startswith("#"):
                cols = line.split("\t")
                words.append((cols[1], cols[2], cols[3], int(cols[6]), cols[7]))
    if words:
        out.append((sent_id, text, tuple(words)))
    return tuple(out)


def _full(lang: str) -> tuple:
    return kept(f"nlp-dutch-tools:{lang}", lambda: _read(UD[lang]))


def treebank(lang: str) -> list:
    """The sentences of a treebank as the exercises take them: a new list of sentences, each a list of words."""
    return [list(words) for _, _, words in _full(lang)]


def _course_gaps(lang: str) -> dict:
    return kept(f"nlp-dutch-tools:gaps:{lang}", lambda: gap_counts(words for _, _, words in _full(lang)))


def course_gaps(lang: str) -> dict:
    """{gap: count} of the particles of a treebank, from the course's gap_counts."""
    return dict(_course_gaps(lang))


def _separable_facts() -> dict:
    """What the picture under Exercise 1 says about the Dutch treebank: the course's separable for each sentence, how
    many verbs it finds, how many of them have a particle of their own in the sentence, and the most common."""
    per_sentence, split, total = [], 0, 0
    counter = collections.Counter()
    for _, _, words in _full("nl"):
        found = separable(words)
        per_sentence.append(tuple(found))
        counter.update(found)
        total += len(found)
        for i, (form, lemma, tag, head, relation) in enumerate(words, start=1):
            if tag == "VERB" and "_" in lemma and any(w[3] == i and w[4] == "compound:prt" for w in words):
                split += 1
    return {"per_sentence": tuple(per_sentence), "total": total, "split": split,
            "top": tuple(counter.most_common(8))}


def separable_facts() -> dict:
    return kept("nlp-dutch-tools:separable", _separable_facts)


# --- numbers -----------------------------------------------------------------------------------------------------------

def _num(n, lang: str) -> str:
    """A whole number with a separator for thousands: 2,319 in English, 2.319 in Dutch."""
    text = f"{round(n):,}"
    return text.replace(",", ".") if lang == "nl" else text


def _dec(x: float, lang: str, digits: int = 1) -> str:
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def _signed(g: int) -> str:
    return f"−{-g}" if g < 0 else str(g)


# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums"
_CODE = ";font-family:var(--ea-font-code);font-size:0.78rem"
_OFF = ";color:var(--ea-loss-text);font-weight:600"
# how each kind of arc is drawn: stroke, width, dashes, label color
_ARC = {
    "ink": ("var(--ea-ink-2)", 1.3, "", "var(--ea-ink-2)"),
    "bold": ("var(--ea-ink)", 2.6, "", "var(--ea-ink)"),
    "dim": ("var(--ea-ink-3)", 1.0, "", "var(--ea-ink-3)"),
}


def _note(text: str) -> str:
    """A line under a picture; a word between asterisks is set in italics, as in the lesson's text."""
    parts = html.escape(text).split("*")
    marked = "".join(f"<i>{p}</i>" if k % 2 else p for k, p in enumerate(parts))
    return f'<div style="{_UI}">{marked}</div>'


def _bare(text: str) -> str:
    return text.replace("*", "")


def _table(rows: list, head: list, numbers: tuple = (), styles: dict | None = None) -> str:
    """A small table; rows hold HTML that is already escaped; numbers are the columns set to the right; styles maps
    (row, column) to extra style."""
    styles = styles or {}
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{_NUMBER if j in numbers else ""}{styles.get((i, j), "")}">'
                                    f'{c}</td>' for j, c in enumerate(row)) + "</tr>" for i, row in enumerate(rows))
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.3rem 0 0.2rem"><thead><tr>'
            f'{top}</tr></thead><tbody>{body}</tbody></table></div>')


def _svg(panels: list, alt: str, lang: str, width: int = 330, height: int = 250) -> str:
    svg = chart.figure(panels, alt=alt, width=width, height=height)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


def arcs_svg(words: list, arcs: list, alt: str, tags=None) -> str:
    """A dependency tree as arcs above the words, as in Who depends on whom: each arc (d, h, label, kind) goes from
    head h to word d, with the arrow at d; head 0 is the root, drawn as an arrow from above. Words count from 1. Under
    each word, its entry of `tags`, here its lemma or nothing. Drawn at its own size, also in the explore, whose
    pictures otherwise fill the width; a long sentence scrolls."""
    n = len(words)
    tags = tags if tags is not None and len(tags) == n else None
    widths = [max(30.0, 8.2 * len(w) + 14, (7.4 * len(tags[k]) + 10) if tags else 0.0) for k, w in enumerate(words)]
    xs, x = [], 10.0
    for wd in widths:
        xs.append(x + wd / 2)
        x += wd + 10
    width = x + 6
    real = [(d, h, label, kind) for d, h, label, kind in arcs if 1 <= d <= n and 0 <= h <= n and h != d]
    spans = {}
    for d, h, _, _ in real:
        if h:
            spans[(d, h)] = (min(d, h), max(d, h))
    level = {}
    for key in sorted(spans, key=lambda k: spans[k][1] - spans[k][0]):
        lo, hi = spans[key]
        inside = [level[o] for o in level if spans[o] != (lo, hi) and lo <= spans[o][0] and spans[o][1] <= hi]
        level[key] = 1 + max(inside, default=0)
    top_level = max(level.values(), default=0)
    y0 = 40 + 22 * top_level
    height = y0 + (44 if tags else 28)
    parts = []
    for d, h, label, kind in real:
        stroke, sw, dash, ink = _ARC.get(kind, _ARC["ink"])
        style = f"fill:none;stroke:{stroke};stroke-width:{sw}" + (f";stroke-dasharray:{dash}" if dash else "")
        dx = xs[d - 1]
        if not h:
            parts.append(f'<path d="M{dx:.1f},8 L{dx:.1f},{y0 - 7:.1f}" style="{style}"/>')
            top, mid, anchor = 16.0, dx + 5, "start"
        else:
            hx = xs[h - 1]
            side = 1 if d > h else -1
            hx, dx = hx + 4 * side, dx - 4 * side
            top = y0 - 12 - 22 * level[(d, h)]
            r = 6
            parts.append(f'<path d="M{hx:.1f},{y0:.1f} L{hx:.1f},{top + r:.1f} Q{hx:.1f},{top:.1f} {hx + r * side:.1f},'
                         f'{top:.1f} L{dx - r * side:.1f},{top:.1f} Q{dx:.1f},{top:.1f} {dx:.1f},{top + r:.1f} '
                         f'L{dx:.1f},{y0 - 7:.1f}" style="{style}"/>')
            mid, anchor = (hx + dx) / 2, "middle"
        parts.append(f'<polygon points="{dx - 3.6:.1f},{y0 - 8:.1f} {dx + 3.6:.1f},{y0 - 8:.1f} {dx:.1f},{y0 - 1:.1f}" '
                     f'style="fill:{stroke}"/>')
        if label:
            weight = ";font-weight:600" if kind == "bold" else ""
            parts.append(f'<text x="{mid:.1f}" y="{top - 3:.1f}" text-anchor="{anchor}" style="font-family:'
                         f'var(--ea-font-code);font-size:11px;fill:{ink};stroke:var(--ea-page);stroke-width:4;'
                         f'paint-order:stroke{weight}">{html.escape(label)}</text>')
    for k, w in enumerate(words):
        parts.append(f'<text x="{xs[k]:.1f}" y="{y0 + 17:.1f}" text-anchor="middle" style="font-family:var(--ea-font-ui);'
                     f'font-size:14px;fill:var(--ea-ink);font-weight:500">{html.escape(w)}</text>')
        if tags and tags[k]:
            parts.append(f'<text x="{xs[k]:.1f}" y="{y0 + 34:.1f}" text-anchor="middle" style="font-family:'
                         f'var(--ea-font-code);font-size:10.5px;fill:var(--ea-ink-3)">{html.escape(tags[k])}</text>')
    return (f'<div style="overflow-x:auto"><svg role="img" aria-label="{html.escape(alt)}" viewBox="0 0 {width:.0f} '
            f'{height:.0f}" width="{width:.0f}" height="{height:.0f}" style="width:{width:.0f}px;height:{height:.0f}px;'
            f'max-width:none;display:block">{"".join(parts)}</svg></div>')


def _tree_arcs(words, bold: str = "compound:prt", rest: str = "ink") -> list:
    """The arcs of a sentence: the arcs with the relation `bold` in bold, with their label; the others in `rest`, with
    their labels too when `rest` is ink."""
    return [(i, w[3], w[4] if w[4] == bold or rest == "ink" else "", "bold" if w[4] == bold else rest)
            for i, w in enumerate(words, start=1)]


_IDEA_WORDS = {
    "en": {"title": "Dependency tree of Ik bel je morgen op (I'll call you tomorrow), with the lemma under each word",
           "legend": "Each arc goes from a head to a word that depends on it. In bold: the particle op, which hangs "
                     "from bel with the relation compound:prt, three words further on. Under bel, the lemma Alpino "
                     "gives it: op_bellen, the particle joined to the verb with an underscore."},
    "nl": {"title": "Dependency tree van Ik bel je morgen op, met het lemma onder elk woord",
           "legend": "Elke boog gaat van een hoofdwoord naar een woord dat ervan afhangt. Vetgedrukt: het partikel "
                     "op, dat aan bel hangt met de relatie compound:prt, drie woorden verderop. Onder bel het lemma "
                     "dat Alpino het geeft: op_bellen, het partikel met een underscore aan het werkwoord vast."},
}


def idea_picture(lang: str) -> str:
    """The tree of BEL, with the lemmas under the words and the arc of the particle in bold."""
    w = _IDEA_WORDS[lang]
    forms = [x[0] for x in BEL]
    alt = f'{w["title"]}. {w["legend"]}'
    return yours.figure(w["title"], arcs_svg(forms, _tree_arcs(BEL), alt, [x[1] for x in BEL]) + _note(w["legend"]),
                        alt=alt)


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error


def _call(fn, *args, seconds: float | None = None):
    """The learner's function within the time limit of a check; _STOPPED when it stops with an error, and the Timeout
    passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            run = lambda: fn(*(copy.deepcopy(a) for a in args))   # noqa: E731
            return with_time_limit(run) if seconds is None else with_time_limit(run, seconds)
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _short(value, limit: int = 60) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit - 1] + "…"


# the course's sentences in the picture under Exercise 1, in their order
_SEPARABLE_ROWS = [("BEL", BEL), ("OPGELOST", OPGELOST), ("GRONDWET", GRONDWET), ("TERUG", TERUG), ("VAAK", VAAK),
                   ("VOORKOMEN", VOORKOMEN), ("DICHT", DICHT), ("PLAATS", PLAATS), ("PICK", PICK), ("BOEK", BOEK),
                   ("DUITS", DUITS), ("BELME", BELME)]


def _sentence_text(words) -> str:
    return " ".join(w[0] for w in words if w[2] != "PUNCT")


_SEP_WORDS = {
    "en": {"title": "Your separable, on the sentences of the checks",
           "head": ["Sentence", "Your function gives", "Separable verbs"],
           "differs": "Your function gives something else for {n} of the {m} sentences; those are in purple.",
           "error": "Your function stops with an error on {name}.",
           "slow": "Your function takes too long on {name}.",
           "fine": "Right for all {m} sentences, also for opgelost and aangepast, which stand in the sentence as one "
                   "word. In all {s} sentences of UD_Dutch-Alpino, your function finds {total} verbs, nearly all of them "
                   "separable. Only {split} of them have their particle apart, as op with bel in Ik bel je morgen op; "
                   "nearly all of the other {joined} are written as one word, and a search for particles does not find them. The most "
                   "common:",
           "top": ["Verb", "Times"],
           "bank_off": "On the {s} sentences of UD_Dutch-Alpino, your function gives something else than the course's "
                       "for {n} sentences, the first in {sid}.",
           "bank_error": "On the sentences of UD_Dutch-Alpino, your function stops with an error, the first time in {sid}.",
           "bank_slow": "Your function takes too long on the {s} sentences of UD_Dutch-Alpino."},
    "nl": {"title": "Jouw separable op de zinnen van de checks",
           "head": ["Zin", "Je functie geeft", "Scheidbare werkwoorden"],
           "differs": "Je functie geeft voor {n} van de {m} zinnen iets anders; die zinnen staan in paars.",
           "error": "Je functie stopt met een fout bij {name}.",
           "slow": "Je functie doet er te lang over bij {name}.",
           "fine": "Goed voor alle {m} zinnen, ook voor opgelost en aangepast, die als één woord in de zin staan. In "
                   "alle {s} zinnen van UD_Dutch-Alpino vindt je functie {total} werkwoorden, bijna allemaal scheidbaar. "
                   "Maar {split} daarvan hebben het partikel los, zoals op bij bel in Ik bel je morgen op; bijna "
                   "alle andere {joined} staan als één woord in de zin, en wie naar partikels zoekt, vindt deze "
                   "werkwoorden niet. De meest voorkomende:",
           "top": ["Werkwoord", "Keer"],
           "bank_off": "Op de {s} zinnen van UD_Dutch-Alpino geeft je functie voor {n} zinnen iets anders dan de "
                       "functie van de course, de eerste in {sid}.",
           "bank_error": "Op de zinnen van UD_Dutch-Alpino stopt je functie met een fout, de eerste keer in {sid}.",
           "bank_slow": "Je functie doet er te lang over op de {s} zinnen van UD_Dutch-Alpino."},
}


def _bank_run(fn) -> tuple:
    """The learner's separable on every Dutch sentence: ("ok", lists), ("error", sent_id) or ("slow", None)."""
    sentences = _full("nl")

    def run():
        out = []
        for sid, _, words in sentences:
            try:
                out.append(fn(list(words)))
            except Timeout:
                raise
            except Exception:   # noqa: BLE001   the picture says where
                return ("error", sid)
        return ("ok", out)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(run, 20)
    except Timeout:
        return ("slow", None)


def separable_picture(seen) -> str:
    """The course's sentences, with what the learner's separable gives for each; when all are right, the counts of the
    learner's function over the Dutch treebank."""
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _SEP_WORDS[lang]
    rows, styles, off, end = [], {}, 0, ""
    for k, (name, words) in enumerate(_SEPARABLE_ROWS):
        try:
            got = _call(fn, words)
        except Timeout:
            end = w["slow"].format(name=name)
            break
        if got is _STOPPED:
            end = w["error"].format(name=name)
            break
        want = separable(words)
        text = f'<i>{html.escape(_sentence_text(words))}</i>'
        rows.append((text, html.escape(_short(got)), html.escape(repr(want))))
        styles[(k, 1)] = _CODE + ("" if got == want else _OFF)
        styles[(k, 2)] = _CODE
        off += got != want
    body = _table(rows, w["head"], styles=styles) if rows else ""
    m = len(_SEPARABLE_ROWS)
    if end:
        text = end
    elif off:
        text = w["differs"].format(n=off, m=m)
    else:
        facts = separable_facts()
        s = len(facts["per_sentence"])
        kind, got = _bank_run(fn)
        if kind == "slow":
            text = w["bank_slow"].format(s=_num(s, lang))
        elif kind == "error":
            text = w["bank_error"].format(sid=got)
        else:
            wrong = [i for i, (a, b) in enumerate(zip(got, facts["per_sentence"])) if not isinstance(a, list) or tuple(a) != b]
            if wrong:
                text = w["bank_off"].format(s=_num(s, lang), n=_num(len(wrong), lang), sid=_full("nl")[wrong[0]][0])
            else:
                text = w["fine"].format(m=m, s=_num(s, lang), total=_num(facts["total"], lang),
                                        split=_num(facts["split"], lang),
                                        joined=_num(facts["total"] - facts["split"], lang))
                top = [(html.escape(v), _num(c, lang)) for v, c in facts["top"]]
                alt = f'{w["title"]}. ' + _bare(text) + " " + ", ".join(f"{v} {c}" for v, c in facts["top"])
                return yours.figure(w["title"], body + _note(text) + _table(top, w["top"], numbers=(1,)), alt=alt)
    alt = f'{w["title"]}. ' + _bare(text)
    return yours.figure(w["title"], body + _note(text), alt=alt)


LOW, HIGH = -6, 12   # the bars of the picture under Exercise 2: a gap below LOW counts at LOW, above HIGH at HIGH

_GAP_WORDS = {
    "en": {"title": "Your gap_counts on two treebanks",
           "x": "gap: places from the verb to its particle", "y": "share of the particles (%)",
           "ends": "The bar at −6 holds every gap of −6 or less, the one at 12 every gap of 12 or more.",
           "fine": "Gray: the share of the particles at each gap, counted by your function. In English, {en1} of the "
                   "{en_n} particles, {en_pct}%, stand right after their verb, and none before it. In Dutch, {nl1} of "
                   "{nl_n}, {nl_pct}%: {nl_before} stand before their verb, and the farthest stands {far} places after it.",
           "differs": "Purple: a bar where your counts differ from the course's, in {name}; gray: a bar where they match.",
           "error": "Your function stops with an error on the sentences of {name}.",
           "slow": "Your function takes too long on the {n} sentences of {name}.",
           "not_dict": "Your function does not give back a dictionary for {name}, so there is nothing to draw yet.",
           "not_whole": "The keys of your dictionary for {name} are not all whole numbers, such as {example}: the "
                        "bars show only the whole numbers."},
    "nl": {"title": "Jouw gap_counts op twee treebanks",
           "x": "afstand: plaatsen van het werkwoord tot het partikel", "y": "deel van de partikels (%)",
           "ends": "De staaf bij −6 bevat elke afstand van −6 of minder, de staaf bij 12 elke afstand van 12 of meer.",
           "fine": "Grijs: het deel van de partikels bij elke afstand, geteld met je functie. In het Engels staan {en1} "
                   "van de {en_n} partikels, {en_pct}%, direct na het werkwoord, en geen partikel ervoor. In het Nederlands staan er "
                   "{nl1} van de {nl_n} direct na het werkwoord, {nl_pct}%; {nl_before} staan ervoor, en het verste "
                   "partikel staat {far} plaatsen erna.",
           "differs": "Paars: een staaf waar je aantallen verschillen van de aantallen van de course, in {name}; grijs: "
                      "een staaf waar de aantallen gelijk zijn.",
           "error": "Je functie stopt met een fout op de zinnen van {name}.",
           "slow": "Je functie doet er te lang over op de {n} zinnen van {name}.",
           "not_dict": "Je functie geeft voor {name} geen dictionary terug, dus er valt nog niets te tekenen.",
           "not_whole": "De sleutels van je dictionary voor {name} zijn niet allemaal hele getallen, zoals {example}: "
                        "de staven tonen alleen de hele getallen."},
}


def _bins(counts: dict) -> list:
    """The counts of the gaps LOW to HIGH, with the gaps beyond them added to the bars at the ends."""
    out = [0] * (HIGH - LOW + 1)
    for g, c in counts.items():
        if isinstance(g, int) and not isinstance(g, bool) and isinstance(c, (int, float)) and not isinstance(c, bool):
            out[min(max(g, LOW), HIGH) - LOW] += c
    return out


def gaps_picture(seen) -> str:
    """The learner's gap_counts on both treebanks, as the share of the particles at each gap."""
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _GAP_WORDS[lang]
    panels, notes, fine = [], [], 0
    for tb in ("en", "nl"):
        name = TREEBANK_NAMES[tb]
        sentences = treebank(tb)
        try:
            got = _call(fn, sentences, seconds=20)
        except Timeout:
            notes.append(w["slow"].format(n=_num(len(sentences), lang), name=name))
            continue
        if got is _STOPPED:
            notes.append(w["error"].format(name=name))
            continue
        if not isinstance(got, dict):
            notes.append(w["not_dict"].format(name=name))
            continue
        want = course_gaps(tb)
        mine, right = _bins(got), _bins(want)
        total = sum(mine) or 1
        shares = [100 * c / total for c in mine]
        top = max(max(shares), 1)
        p = chart.Panel((LOW - 0.6, HIGH + 0.6), (0, top * 1.12), title=name, xlabel=w["x"], ylabel=w["y"],
                        xticks=[-6, -3, 0, 3, 6, 9, 12])
        for k, (m, r) in enumerate(zip(mine, right)):
            g = LOW + k
            if m:
                p.rect(g - 0.4, 0, g + 0.4, 100 * m / total, role="data" if m == r else "loss", solid=m == r)
        panels.append(p)
        odd = [k for k in got if not (isinstance(k, int) and not isinstance(k, bool))]
        if mine != right or odd or got != want:
            notes.append(w["differs"].format(name=name))
        if odd:
            notes.append(w["not_whole"].format(name=name, example=_short(odd[0], 20)))
        if got == want:
            fine += 1
    if fine == 2:
        en, nl = course_gaps("en"), course_gaps("nl")
        en_n, nl_n = sum(en.values()), sum(nl.values())
        notes.append(w["fine"].format(en1=_num(en.get(1, 0), lang), en_n=_num(en_n, lang),
                                      en_pct=_dec(100 * en.get(1, 0) / en_n, lang), nl1=_num(nl.get(1, 0), lang),
                                      nl_n=_num(nl_n, lang), nl_pct=_dec(100 * nl.get(1, 0) / nl_n, lang),
                                      nl_before=_num(sum(c for g, c in nl.items() if g < 0), lang), far=max(nl)))
    notes.append(w["ends"])
    text = " ".join(notes)
    alt = f'{w["title"]}. ' + _bare(text)
    if not panels:
        return yours.figure(w["title"], _note(text), alt=alt)
    return yours.figure(w["title"], _svg(panels, alt, lang) + _note(text), alt=alt)


# --- the explore: a sentence of the Dutch treebank for every gap ------------------------------------------------------

# a sentence of UD_Dutch-Alpino for each gap, with one particle, whose verb has a lemma with an underscore; chosen by
# the course among the short sentences without names
EXPLORE = {
    -3: "cdb\\4846",
    -2: "cgn_exs\\56",
    -1: "cdb\\3360",
    1: "WR-P-P-H-0000000101\\WR-P-P-H-0000000101.p.1.s.5",
    2: "eans\\02_03_02_05_2",
    3: "cgn_exs\\53",
    4: "WR-P-P-H-0000000047\\WR-P-P-H-0000000047.p.3.s.4",
    5: "eans\\02_04_08_03_03_3",
    6: "eans\\02_04_08_03_01_3",
    7: "cdb\\3701",
    8: "cdb\\2483",
    9: "cdb\\3719",
    10: "cdb\\3573",
    11: "cdb\\6385",
    12: "wpspel\\974",
}
EXPLORE_GAPS = sorted(EXPLORE)
EXPLORE_GLOSS = {
    -3: "that does not at all mean that the Conservatives would already give up the fight",
    -2: "I'll try to call you back tomorrow",
    -1: "I am going to make an old dream come true",
    1: "the lightning struck the middle of the field",
    2: "do come in",
    3: "I'll call you back tomorrow",
    4: "the doors closed just in time",
    5: "I get off at the next stop",
    6: "I get up at seven in the morning",
    7: "to that static art object you now add light and movement",
    8: "according to the doctors, the boys are left with no harm from it",
    9: "so always leave dry clothes behind on the shore",
    10: "in the middle of the Rijnmond debate, the uproar broke out not in front of the town hall but inside it",
    11: "the oxygen level varied widely in the summers of 1966 to 1969",
    12: "which index shows how the 20 largest shares on the Brussels stock exchange move",
}


def explore_sentence(g: int) -> tuple:
    """(text, words) of the sentence of the explore at gap g."""
    sid = EXPLORE[g]
    for s, text, words in _full("nl"):
        if s == sid:
            return text, words
    raise KeyError(sid)


_EXPLORE_WORDS = {
    "en": {"title": "Gap {g}",
           "said": "**{gs}**: {n} of the {total} particles in UD_Dutch-Alpino stand there. *{text}* ({gloss}): the "
                   "particle *{prt}* hangs from *{verb}*, with the lemma `{lemma}`.",
           "alt": "Dependency tree of {text}, with the arc from {verb} to its particle {prt} in bold."},
    "nl": {"title": "Afstand {g}",
           "said": "**{gs}**: {n} van de {total} partikels in UD_Dutch-Alpino staan daar. *{text}*: het partikel "
                   "*{prt}* hangt aan *{verb}*, met het lemma `{lemma}`.",
           "alt": "Dependency tree van {text}, met de boog van {verb} naar het partikel {prt} vetgedrukt."},
}


def explore_frame(g, lang: str) -> tuple:
    """The picture and the sentence of the explore at gap g: the tree, with the particle's arc in bold and the
    lemma under the verb."""
    g = int(g)
    w = _EXPLORE_WORDS[lang]
    text, words = explore_sentence(g)
    i = next(k for k, x in enumerate(words, start=1) if x[4] == "compound:prt")
    h = words[i - 1][3]
    forms = [x[0] for x in words]
    tags = [words[h - 1][1] if k == h else "" for k in range(1, len(words) + 1)]
    counts = course_gaps("nl")
    title = w["title"].format(g=_signed(g))
    alt = w["alt"].format(text=text, verb=forms[h - 1], prt=forms[i - 1])
    svg = yours.figure(title, arcs_svg(forms, _tree_arcs(words, rest="dim"), alt, tags), alt=alt)
    said = w["said"].format(gs=_signed(g), n=_num(counts.get(g, 0), lang), total=_num(sum(counts.values()), lang),
                            text=text, gloss=EXPLORE_GLOSS[g], prt=forms[i - 1], verb=forms[h - 1],
                            lemma=words[h - 1][1])
    return svg, said


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["separable", "gap_counts", "corpus", "rule"]


class Named:
    """A value the check gives the learner's function, shown in a message by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


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


def _index_error(e) -> bool:
    return isinstance(e, IndexError)


def _unpack_error(e) -> bool:
    return isinstance(e, ValueError) and "unpack" in str(e)


def _key_error(e) -> bool:
    return isinstance(e, KeyError)


def _relation_used(got, text) -> bool:
    return "compound:prt" in text and got == []


def _nothing(got, text) -> bool:
    return got == []


def _a_list(got, text) -> bool:
    return isinstance(got, list)


# exercise 1: separable. BEL first; then a verb written as one word, a compound noun, two verbs, the two verbs
# voorkomen (one separable, one not: the form alone cannot tell), a particle before its verb, a noun as the particle,
# an English sentence, whose lemmas join nothing, a sentence without a separable verb, a compound adjective, and a
# sentence that starts with its separable verb.
SEPARABLE_CASES = [
    (Named("BEL", BEL),),
    (Named("OPGELOST", OPGELOST),),
    (Named("GRONDWET", GRONDWET),),
    (Named("TERUG", TERUG),),
    (Named("VAAK", VAAK),),
    (Named("VOORKOMEN", VOORKOMEN),),
    (Named("DICHT", DICHT),),
    (Named("PLAATS", PLAATS),),
    (Named("PICK", PICK),),
    (Named("BOEK", BOEK),),
    (Named("DUITS", DUITS),),
    (Named("BELME", BELME),),
]
_SEP_START = ("def separable(words):\n"
              "    # the separable verbs of the sentence, each written as one word, from left to right\n"
              "    ...\n")
_SEP = ("def separable(words):\n"
        "    found = []\n"
        "    for form, lemma, tag, head, relation in words:\n"
        "        if tag == \"VERB\" and \"_\" in lemma:\n"
        "            found.append(lemma.replace(\"_\", \"\"))\n"
        "    return found\n")
_SEP_ANY = _SEP.replace("tag == \"VERB\" and ", "")
_SEP_NOUN = _SEP.replace("tag == \"VERB\"", "tag != \"NOUN\"")
_SEP_KEEP = _SEP.replace("lemma.replace(\"_\", \"\")", "lemma")
_SEP_SPACE = _SEP.replace("lemma.replace(\"_\", \"\")", "lemma.replace(\"_\", \" \")")
_SEP_FORM = _SEP.replace("lemma.replace(\"_\", \"\")", "form")
_SEP_FIRST = _SEP.replace("            found.append(lemma.replace(\"_\", \"\"))\n",
                          "            return [lemma.replace(\"_\", \"\")]\n")
_SEP_EARLY = _SEP.replace("    return found\n", "        return found\n")
_SEP_RELATION = ("def separable(words):\n"
                 "    found = []\n"
                 "    for form, lemma, tag, head, relation in words:\n"
                 "        if relation == \"compound:prt\":\n"
                 "            found.append(words[head - 1][1].replace(\"_\", \"\"))\n"
                 "    return found\n")

# exercise 2: gap_counts, on lists of the course's sentences: the example, a gap that comes twice, a particle before
# its verb, an English noun compound with the relation compound, no particle, no sentence, and two gaps of 2 in
# different sentences, and a sentence without a full stop, whose last word is the particle.
GAP_CASES = [
    (Named("[BEL, PICK, OPGELOST]", [BEL, PICK, OPGELOST]),),
    (Named("[BEL, TERUG]", [BEL, TERUG]),),
    (Named("[DICHT]", [DICHT]),),
    (Named("[LIGHT]", [LIGHT]),),
    (Named("[BOEK]", [BOEK]),),
    (Named("[]", []),),
    (Named("[VAAK, PLAATS, BEL]", [VAAK, PLAATS, BEL]),),
    (Named("[TERUG, BOEK, DICHT, VAAK]", [TERUG, BOEK, DICHT, VAAK]),),
    (Named("[TERUGBEL]", [TERUGBEL]),),
]
_G_START = ("def gap_counts(sentences):\n"
            "    # for every gap from a verb to its particle: how many particles of the sentences stand there\n"
            "    ...\n")
_G = ("def gap_counts(sentences):\n"
      "    counts = {}\n"
      "    for words in sentences:\n"
      "        for i in range(1, len(words) + 1):\n"
      "            form, lemma, tag, head, relation = words[i - 1]\n"
      "            if relation == \"compound:prt\":\n"
      "                gap = i - head\n"
      "                if gap in counts:\n"
      "                    counts[gap] = counts[gap] + 1\n"
      "                else:\n"
      "                    counts[gap] = 1\n"
      "    return counts\n")
_G_UP = _G.replace("gap = i - head", "gap = head - i")
_G_ABS = _G.replace("gap = i - head", "gap = abs(i - head)")
_G_ZERO = _G.replace("range(1, len(words) + 1)", "range(len(words))").replace("words[i - 1]", "words[i]")
_G_HALF = _G.replace("range(1, len(words) + 1)", "range(len(words))")
_G_ALL = _G.replace("            if relation == \"compound:prt\":\n                gap = i - head\n",
                    "            if True:\n                gap = i - head\n")
_G_COMPOUND = _G.replace("if relation == \"compound:prt\":", "if \"compound\" in relation:")
_G_ONE = _G.replace("                    counts[gap] = counts[gap] + 1\n", "                    counts[gap] = 1\n")
_G_RESET = _G.replace("    counts = {}\n    for words in sentences:\n", "    for words in sentences:\n        counts = {}\n")
_G_EARLY = _G.replace("    return counts\n", "        return counts\n")
_G_FORMS = _G.replace("                gap = i - head\n", "                gap = form\n")
_G_HEAD = _G.replace("                gap = i - head\n", "                gap = head\n")


EXERCISES = {
    "separable": FunctionBox(
        name="separable",
        picture=separable_picture,
        function="separable",
        start=_SEP_START,
        solution=_SEP,
        visible=SEPARABLE_CASES[:1],
        hidden=SEPARABLE_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("relation", matches=_relation_used,
                    en="`{call}` gives {got}, and it should give {want}. Your function goes by the relation `compound:prt`, so it finds a separable verb only when its particle stands apart. In *We hebben het probleem samen opgelost* (we solved the problem together), *opgelost* is a separable verb written as one word, without a particle of its own. Look at the lemmas of the verbs instead.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie kijkt naar de relatie `compound:prt`, dus vindt je functie een scheidbaar werkwoord alleen als het partikel los staat. In *We hebben het probleem samen opgelost* is *opgelost* een scheidbaar werkwoord dat als één woord in de zin staat, zonder een eigen partikel. Kijk in plaats daarvan naar de lemma's van de werkwoorden."),
            Mistake("any-tag", wrong=_SEP_ANY,
                    en="`{call}` gives {got}, and it should give {want}. *grondwet* (constitution) is a noun: Alpino marks the parts of a compound in its lemma too, `grond_wet`. Keep only the words with the tag `VERB`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. *grondwet* is een zelfstandig naamwoord: Alpino geeft ook de delen van een samenstelling aan in het lemma, `grond_wet`. Houd alleen de woorden met de tag `VERB`."),
            Mistake("noun", wrong=_SEP_NOUN,
                    en="`{call}` gives {got}, and it should give {want}. *Duitstalige* (German-language) is an adjective, with the tag `ADJ`: Alpino marks the parts of a compound adjective in its lemma too, `Duits_talig`. Keep only the words with the tag `VERB`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. *Duitstalige* is een bijvoeglijk naamwoord, met de tag `ADJ`: Alpino geeft ook de delen van een samengesteld bijvoeglijk naamwoord aan in het lemma, `Duits_talig`. Houd alleen de woorden met de tag `VERB`."),
            Mistake("keep", wrong=_SEP_KEEP,
                    en="`{call}` gives {got}, and it should give {want}: your verbs still have Alpino's underscore. Write each verb as one word, as a dictionary has it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je werkwoorden hebben de underscore van Alpino nog. Schrijf elk werkwoord als één woord, zoals het in een woordenboek staat."),
            Mistake("space", wrong=_SEP_SPACE,
                    en="`{call}` gives {got}, and it should give {want}: your verbs have a space in the middle. Write each verb as one word, as a dictionary has it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je werkwoorden hebben een spatie in het midden. Schrijf elk werkwoord als één woord, zoals het in een woordenboek staat."),
            Mistake("form", wrong=_SEP_FORM,
                    en="`{call}` gives {got}, and it should give {want}: your function gives the verbs as they stand in the sentence. Give the lemma, the verb as a dictionary has it, written as one word.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft de vormen van de werkwoorden in de zin. Geef het lemma, het werkwoord zoals het in een woordenboek staat, als één woord geschreven."),
            Mistake("first", wrong=_SEP_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: your function stops at the first separable verb. Collect all of them, and give back the list after the loop.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie stopt bij het eerste scheidbare werkwoord. Verzamel alle scheidbare werkwoorden, en geef de lijst na de loop terug."),
            Mistake("early", wrong=_SEP_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back the list after the first word. Move `return found` to the left, under `for`, so that it comes after the loop.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft de lijst al na het eerste woord terug. Zet `return found` meer naar links, onder `for`, zodat de `return` na de loop komt."),
            Mistake("nothing", matches=_nothing,
                    en="`{call}` gives {got}, and it should give {want}: your function finds no separable verb in this sentence. Which of the five parts of a word shows that a verb is separable, as `op_bellen` does for *bel*? Look at that part as the treebank wrote it, before you change it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie vindt geen scheidbaar werkwoord in deze zin. Welk van de vijf delen van een woord laat zien dat een werkwoord scheidbaar is, zoals `op_bellen` bij *bel*? Kijk naar dat deel zoals de treebank het schreef, voordat je het verandert."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: go over the words of the sentence, and collect the separable verbs, each as one word.",
                      "nl": "Vervang de `...`: loop langs de woorden van de zin, en verzamel de scheidbare werkwoorden, elk als één woord."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The function gives back the separable verbs of the sentence, each written as one word, from left to right.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De functie geeft de scheidbare werkwoorden van de zin terug, elk als één woord, van links naar rechts."},
            "almost": {"en": "Right for `BEL`. But `{call}` gives {got}, and it should give {want}. Does your function use the lemma and the tag of every word?",
                       "nl": "Goed voor `BEL`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie het lemma en de tag van elk woord?"},
            "correct": {"en": "Your `separable` is right in all {n} cases: also for *opgelost*, written as one word, and for the two verbs *voorkomen*, of which only *voor_komen* is separable. Look at the picture: your function on the whole Dutch treebank.",
                        "nl": "Je `separable` klopt in alle {n} gevallen: ook voor *opgelost*, als één woord geschreven, en voor de twee werkwoorden *voorkomen*, waarvan alleen *voor_komen* scheidbaar is. Kijk naar de afbeelding: je functie op de hele Nederlandse treebank."},
        },
    ),
    "gap_counts": FunctionBox(
        name="gap_counts",
        picture=gaps_picture,
        function="gap_counts",
        start=_G_START,
        solution=_G,
        visible=GAP_CASES[:1],
        hidden=GAP_CASES[1:],
        collect=lambda value: dict(value) if isinstance(value, dict) else value,   # a Counter is a dictionary too
        first=[_DOTS],
        mistakes=[
            Mistake("index", error=_index_error,
                    en="With `{call}`, your function looks past the end of a list. Word number `i` is at place `i - 1`: the words count from 1, the places in a list from 0.",
                    nl="Met `{call}` kijkt je functie voorbij het eind van een lijst. Woord nummer `i` staat op plaats `i - 1`: de woorden tellen vanaf 1, de plaatsen in een lijst vanaf 0."),
            Mistake("unpack", error=_unpack_error,
                    en="With `{call}`, your function takes something apart into five parts that is not a word. `sentences` is a list of sentences, and each sentence a list of words: go over the sentences, and inside that loop over the words of each sentence.",
                    nl="Met `{call}` haalt je functie iets in vijf delen uit elkaar dat geen woord is. `sentences` is een lijst met zinnen, en elke zin een lijst met woorden: loop langs de zinnen, en binnen die loop langs de woorden van elke zin."),
            Mistake("key", error=_key_error,
                    en="With `{call}`, your function stops with a `KeyError`: the first time a gap comes up, it is not in `counts` yet, so `counts[gap] + 1` cannot work. Look first with `if gap in counts:`, and start a new gap at 1.",
                    nl="Met `{call}` stopt je functie met een `KeyError`: de eerste keer dat een afstand voorkomt, staat de afstand nog niet in `counts`, dus `counts[gap] + 1` kan niet. Kijk eerst met `if gap in counts:`, en begin een nieuwe afstand bij 1."),
            Mistake("list", matches=_a_list,
                    en="`{call}` gives back a list, {got}, and it should give a dictionary: {want}. Count each gap: the gap is the key, and how often it comes up the value.",
                    nl="`{call}` geeft een lijst terug, {got}, en dat moet een dictionary zijn: {want}. Tel elke afstand: de afstand is de sleutel, en hoe vaak de afstand voorkomt de waarde."),
            Mistake("upside", wrong=_G_UP,
                    en="`{call}` gives {got}, and it should give {want}: each gap has the wrong sign. A particle after its verb, as *op* in *Ik bel je morgen op*, has a positive gap; one before it a negative gap.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke afstand heeft het verkeerde teken. Een partikel na het werkwoord, zoals *op* in *Ik bel je morgen op*, heeft een positieve afstand; een partikel ervoor een negatieve."),
            Mistake("abs", wrong=_G_ABS,
                    en="`{call}` gives {got}, and it should give {want}: your gaps are never negative. In *We proberen de deur dicht te doen* (we try to close the door), *dicht* stands two places before *doen*: a gap of −2.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je afstanden zijn nooit negatief. In *We proberen de deur dicht te doen* staat *dicht* twee plaatsen vóór *doen*: een afstand van −2."),
            Mistake("zero", wrong=_G_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: each gap is one too low. Your function counts the places in the list, which start at 0; `head` counts the words from 1.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke afstand is één te laag. Je functie telt de plaatsen in de lijst, en de plaatsen beginnen bij 0; `head` telt de woorden vanaf 1."),
            Mistake("half", wrong=_G_HALF,
                    en="`{call}` gives {got}, and it should give {want}: your loop goes over the places 0 to `len(words) - 1`, but `words[i - 1]` takes word numbers. With `i` as 0, `words[-1]` is the last word of the sentence, and the last word is never word number `i`. Let `i` run over the word numbers, from 1 to `len(words)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je loop gaat langs de plaatsen 0 tot `len(words) - 1`, maar `words[i - 1]` neemt woordnummers. Met `i` gelijk aan 0 is `words[-1]` het laatste woord van de zin, en het laatste woord is nooit woord nummer `i`. Laat `i` langs de woordnummers lopen, van 1 tot en met `len(words)`."),
            Mistake("all", wrong=_G_ALL,
                    en="`{call}` gives {got}, and it should give {want}: your function counts a gap for every word. Count only the particles: the words with the relation `compound:prt`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt een afstand voor elk woord. Tel alleen de partikels: de woorden met de relatie `compound:prt`."),
            Mistake("compound", wrong=_G_COMPOUND,
                    en="`{call}` gives {got}, and it should give {want}. In *Turn the kitchen light off*, *kitchen* has the relation `compound`, of a noun compound written as two words, and it is not a particle. Compare the whole relation with `compound:prt`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. In *Turn the kitchen light off* heeft *kitchen* de relatie `compound`, van een samenstelling die het Engels als twee woorden schrijft, en dat is geen partikel. Vergelijk de hele relatie met `compound:prt`."),
            Mistake("one", wrong=_G_ONE,
                    en="`{call}` gives {got}, and it should give {want}: every gap gets the count 1, also a gap that comes up twice. When the gap is in `counts` already, add 1 to its count.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke afstand krijgt het aantal 1, ook een afstand die twee keer voorkomt. Staat de afstand al in `counts`, tel dan 1 op bij het aantal."),
            Mistake("reset", wrong=_G_RESET,
                    en="`{call}` gives {got}, and it should give {want}: your function counts only the last sentence. It makes `counts` again for each sentence; make it once, before the loop over the sentences.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt alleen de laatste zin. Je functie maakt `counts` opnieuw voor elke zin; maak `counts` één keer, vóór de loop langs de zinnen."),
            Mistake("early", wrong=_G_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back the counts after the first sentence. Move `return counts` to the left, under the first `for`, so that it comes after both loops.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft de aantallen al na de eerste zin terug. Zet `return counts` meer naar links, onder de eerste `for`, zodat de `return` na beide loops komt."),
            Mistake("forms", wrong=_G_FORMS,
                    en="`{call}` gives {got}, and it should give {want}: your keys are the particles themselves. The key is the gap: how many places the particle stands from its verb.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je sleutels zijn de partikels zelf. De sleutel is de afstand: hoeveel plaatsen het partikel van het werkwoord staat."),
            Mistake("head", wrong=_G_HEAD,
                    en="`{call}` gives {got}, and it should give {want}: your keys are the numbers of the verbs. The key is the gap: how many places the particle stands from its verb.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je sleutels zijn de nummers van de werkwoorden. De sleutel is de afstand: hoeveel plaatsen het partikel van het werkwoord staat."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: go over the sentences, and in each over the word numbers; for every particle, add 1 to the count of its gap.",
                      "nl": "Vervang de `...`: loop langs de zinnen, en in elke zin langs de woordnummers; tel voor elk partikel 1 op bij het aantal van de afstand."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The function gives back a dictionary: for every gap, how many particles of the sentences stand at that gap.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De functie geeft een dictionary terug: voor elke afstand hoeveel partikels van de zinnen op die afstand staan."},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function go over all the sentences it gets, and count every particle?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Loopt je functie langs alle zinnen in de lijst, en telt je functie elk partikel?"},
            "correct": {"en": "Your `gap_counts` is right in all {n} cases, also for a particle before its verb and for no sentences at all. Look at the picture: your function on both treebanks.",
                        "nl": "Je `gap_counts` klopt in alle {n} gevallen, ook voor een partikel vóór het werkwoord en voor een lege lijst met zinnen. Kijk naar de afbeelding: je functie op beide treebanks."},
        },
    ),
    "corpus": Sort(
        name="corpus",
        groups={
            "sonar": {"en": "SoNaR", "nl": "SoNaR"},
            "cgn": {"en": "Corpus Gesproken Nederlands", "nl": "Corpus Gesproken Nederlands"},
            "small": {"en": "Lassy Small", "nl": "Lassy Klein"},
            "large": {"en": "Lassy Large", "nl": "Lassy Groot"},
        },
        items={
            "speech": {"en": "Train a program that writes out what people say, in Dutch from Flanders and the Netherlands",
                       "nl": "Een programma trainen dat uitschrijft wat mensen zeggen, in het Nederlands uit Vlaanderen en Nederland"},
            "test": {"en": "Measure how often a parser for Dutch gets the tree right, against trees that people checked",
                     "nl": "Meten hoe vaak een parser voor het Nederlands de boom goed heeft, tegenover bomen die mensen hebben gecontroleerd"},
            "names": {"en": "Test a program that finds the names of people, places and organizations in a text, against labels that people checked",
                      "nl": "Een programma testen dat de namen van personen, plaatsen en organisaties in een tekst vindt, tegenover labels die mensen hebben gecontroleerd"},
            "rare": {"en": "Find thousands of sentences with a rare construction, each with its tree, even if about one tree in five is wrong",
                     "nl": "Duizenden zinnen vinden met een zeldzame constructie, elk met de boom erbij, ook al is ongeveer één boom op de vijf fout"},
            "phone": {"en": "Study how people talk to each other on the phone",
                      "nl": "Onderzoeken hoe mensen met elkaar praten aan de telefoon"},
        },
        solution={"speech": "cgn", "test": "small", "names": "sonar", "rare": "large", "phone": "cgn"},
        nudge={
            "speech": {"en": "Which corpus holds recordings of speech, with every word written out?",
                       "nl": "Welk corpus bevat opnames van spraak, met elk woord uitgeschreven?"},
            "test": {"en": "Which treebank did people check sentence by sentence?",
                     "nl": "Welke treebank hebben mensen zin voor zin gecontroleerd?"},
            "names": {"en": "Which corpus has a part of one million words with labels for names that people checked?",
                      "nl": "Welk corpus heeft een deel van een miljoen woorden met labels voor namen die mensen hebben gecontroleerd?"},
            "rare": {"en": "A rare construction needs many sentences. Which treebank is the larger, and what does it give up for that?",
                     "nl": "Een zeldzame constructie vraagt veel zinnen. Welke treebank is de grootste, en wat lever je daarvoor in?"},
            "phone": {"en": "Which corpus has telephone calls among its recordings?",
                      "nl": "Welk corpus heeft telefoongesprekken bij de opnames?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the tasks again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de taken opnieuw uit."},
            "empty": {"en": "Put each task under the corpus that fits it best.",
                      "nl": "Zet elke taak bij het corpus dat er het best bij past."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} belongs under another corpus.",
                      "nl": "{item} hoort bij een ander corpus."},
            "correct": {"en": "All {n} are right. Speech is in the CGN, trees that people checked in Lassy Small, many trees that no one checked in Lassy Large, and names that people checked in SoNaR-1.",
                        "nl": "Alle {n} kloppen. Spraak staat in het CGN, bomen die mensen controleerden in Lassy Klein, veel bomen die niemand controleerde in Lassy Groot, en namen die mensen controleerden in SoNaR-1."},
        },
    ),
}

CASES = [
    ("separable", _SEP_START, "empty", None),
    ("separable", _SEP, "correct", None),
    ("separable", "def separable(words):\n    return [w[1].replace('_', '') for w in words if w[2] == 'VERB' and '_' in w[1]]\n", "correct", None),
    ("separable", "def separable(words):\n    found = []\n    for form, lemma, tag, head, relation in words:\n        if '_' in lemma and tag in ('VERB', 'AUX'):\n            found.append(''.join(lemma.split('_')))\n    return found\n", "correct", None),
    ("separable", _SEP_RELATION, "wrong", "relation"),
    ("separable", _SEP_ANY, "wrong", "any-tag"),
    ("separable", _SEP_KEEP, "wrong", "keep"),
    ("separable", _SEP_SPACE, "wrong", "space"),
    ("separable", _SEP_FORM, "wrong", "form"),
    ("separable", _SEP_FIRST, "wrong", "first"),
    ("separable", _SEP_EARLY, "wrong", "early"),
    ("separable", _SEP_NOUN, "wrong", "noun"),
    ("separable", _SEP.replace("\"_\" in lemma", "\"_\" in form"), "wrong", "nothing"),
    ("separable", _SEP.replace("        if tag", "        lemma = lemma.replace(\"_\", \"\")\n        if tag"), "wrong", "nothing"),
    # the particles' verbs and the verbs with a joined lemma, both: more verbs than there are, not the relation's message
    ("separable", _SEP.replace("            found.append(lemma.replace(\"_\", \"\"))\n",
                               "            found.append(lemma.replace(\"_\", \"\"))\n        if relation == \"compound:prt\":\n"
                               "            found.append(words[head - 1][1].replace(\"_\", \"\"))\n"), "wrong", None),
    ("separable", "def separable(words):\n    return ['opbellen']\n", "almost", None),
    ("gap_counts", _G_START, "empty", None),
    ("gap_counts", _G, "correct", None),
    ("gap_counts", "def gap_counts(sentences):\n    counts = {}\n    for words in sentences:\n        i = 0\n        for form, lemma, tag, head, relation in words:\n            i = i + 1\n            if relation == 'compound:prt':\n                counts[i - head] = counts.get(i - head, 0) + 1\n    return counts\n", "correct", None),
    ("gap_counts", _G_UP, "wrong", "upside"),
    ("gap_counts", _G_ABS, "wrong", "abs"),
    ("gap_counts", _G_ZERO, "wrong", "zero"),
    ("gap_counts", _G_ALL, "wrong", "all"),
    ("gap_counts", _G_COMPOUND, "wrong", "compound"),
    ("gap_counts", _G_ONE, "wrong", "one"),
    ("gap_counts", _G_RESET, "wrong", "reset"),
    ("gap_counts", _G_EARLY, "wrong", "early"),
    ("gap_counts", _G_FORMS, "wrong", "forms"),
    ("gap_counts", _G_HEAD, "wrong", "head"),
    ("gap_counts", _G_HALF, "wrong", "half"),
    ("gap_counts", "from collections import Counter\n\ndef gap_counts(sentences):\n    counts = Counter()\n    for words in sentences:\n        for i, (form, lemma, tag, head, relation) in enumerate(words, start=1):\n            if relation == 'compound:prt':\n                counts[i - head] += 1\n    return counts\n", "correct", None),
    ("gap_counts", _G.replace("                if gap in counts:\n                    counts[gap] = counts[gap] + 1\n                else:\n                    counts[gap] = 1\n",
                              "                counts[gap] = counts[gap] + 1\n"), "error", "key"),
    ("gap_counts", _G.replace("words[i - 1]", "words[i]"), "error", "index"),
    ("gap_counts", "def gap_counts(sentences):\n    counts = {}\n    for i in range(1, len(sentences) + 1):\n        form, lemma, tag, head, relation = sentences[i - 1]\n        if relation == 'compound:prt':\n            counts[i - head] = 1\n    return counts\n", "error", "unpack"),
    ("gap_counts", "def gap_counts(sentences):\n    found = []\n    for words in sentences:\n        for i in range(1, len(words) + 1):\n            if words[i - 1][4] == 'compound:prt':\n                found.append(i - words[i - 1][3])\n    return found\n", "wrong", "list"),
    ("gap_counts", "def gap_counts(sentences):\n    return {3: 1, 2: 1}\n", "almost", None),
    ("corpus", {}, "empty", None),
    ("corpus", {"speech": "cgn", "test": "small", "names": "sonar", "rare": "large", "phone": "cgn"}, "correct", None),
    ("corpus", {"speech": "cgn", "test": "small"}, "wrong", "unfinished"),
    ("corpus", {"test": "large"}, "wrong", "wrong:test"),
    ("corpus", {"speech": "cgn", "test": "small", "names": "sonar", "rare": "small", "phone": "cgn"}, "wrong", "wrong:rare"),
    ("corpus", {"names": "small"}, "wrong", "wrong:names"),
]
