# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-dependencies: find the dependents of a word, follow the heads from a word to the root,
name the relations of the words of a sentence, and say where Universal Dependencies hangs a preposition. The two
short sentences of the course are the course's own, with relations the course gave them by the guidelines of UD; the
demo shows sentences of two treebanks of Universal Dependencies 2.19, UD_English-EWT (without the emails of Enron)
and UD_Dutch-Alpino (data/ud-en-ewt.conllu.gz and data/ud-nl-alpino.conllu.gz, made by data/tools/make_ud.py)."""
import contextlib
import copy
import gzip
import html
import io
import random

from everything_ai import paths, yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, kept, with_time_limit

# --- the course's sentences ------------------------------------------------------------------------------------------

# word i of a sentence is at place i - 1: (form, head, relation); head 0 is the root. The heads are the same in
# English and in Dutch, so a case on numbers gives the same answer in both languages.
SENTENCES = {
    "en": [("the", 3, "det"), ("late", 3, "amod"), ("train", 4, "nsubj"), ("left", 0, "root"), ("the", 6, "det"),
           ("city", 4, "obj"), ("after", 8, "case"), ("midnight", 4, "obl")],
    "nl": [("de", 3, "det"), ("late", 3, "amod"), ("trein", 4, "nsubj"), ("verliet", 0, "root"), ("de", 6, "det"),
           ("stad", 4, "obj"), ("na", 8, "case"), ("middernacht", 4, "obl")],
}
SENTENCE_TAGS = ["DET", "ADJ", "NOUN", "VERB", "DET", "NOUN", "ADP", "NOUN"]
ROOFS = {
    "en": [("the", 2, "det"), ("roof", 9, "nsubj"), ("of", 5, "case"), ("the", 5, "det"), ("house", 2, "nmod"),
           ("of", 8, "case"), ("the", 8, "det"), ("baker", 5, "nmod"), ("leaked", 0, "root")],
    "nl": [("het", 2, "det"), ("dak", 9, "nsubj"), ("van", 5, "case"), ("het", 5, "det"), ("huis", 2, "nmod"),
           ("van", 8, "case"), ("de", 8, "det"), ("bakker", 5, "nmod"), ("lekte", 0, "root")],
}
ROOF_TAGS = ["DET", "NOUN", "ADP", "DET", "NOUN", "ADP", "DET", "NOUN", "VERB"]
# sentences of the check's own: the root first, one word alone
DOOR = [("Close", 0, "root"), ("the", 3, "det"), ("door", 1, "obj")]
HELLO = [("Hello", 0, "root")]


def dependents(sentence, h):
    """The course's solution of Exercise 1: the numbers of the words whose head is h, from left to right."""
    found = []
    for i in range(1, len(sentence) + 1):
        form, head, relation = sentence[i - 1]
        if head == h:
            found.append(i)
    return found


def path_to_root(sentence, i):
    """The course's solution of Exercise 2: the numbers of the words from word i up to the root."""
    path = [i]
    for step in range(len(sentence)):
        form, head, relation = sentence[i - 1]
        if head == 0:
            return path
        path.append(head)
        i = head
    return path


def head_of(sentence, i):
    """The worked example: the form of the head of word i, or ROOT."""
    form, head, relation = sentence[i - 1]
    if head == 0:
        return "ROOT"
    return sentence[head - 1][0]


# --- the treebanks of the demo ---------------------------------------------------------------------------------------

UD_EN = paths.ROOT / "data" / "ud-en-ewt.conllu.gz"
UD_NL = paths.ROOT / "data" / "ud-nl-alpino.conllu.gz"


def _read(path) -> tuple:
    """The sentences of a CoNLL-U file of the course: (sent_id, words), each word a tuple (form, tag, head, relation)."""
    out, words, sent_id = [], [], None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# sent_id = "):
                sent_id = line[len("# sent_id = "):]
            elif not line:
                if words:
                    out.append((sent_id, tuple(words)))
                words = []
            elif not line.startswith("#"):
                cols = line.split("\t")
                words.append((cols[1], cols[3], int(cols[6]), cols[7]))
    if words:
        out.append((sent_id, tuple(words)))
    return tuple(out)


TREEBANKS = {"en": kept("nlp-dependencies:ewt", lambda: _read(UD_EN)),
             "nl": kept("nlp-dependencies:alpino", lambda: _read(UD_NL))}
TREEBANK_NAMES = {"en": "UD_English-EWT", "nl": "UD_Dutch-Alpino"}
# what keeps a sentence out of the demo: the posts of EWT are written by anyone, so the demo leaves out its weblogs and
# newsgroups and keeps its reviews and answers, and in both treebanks no word may be one of _SKIP or hold one of _STEMS
_SKIP = {"ass", "god", "sex", "war", "dead", "die", "died", "damn", "crap", "jezus", "kut", "dood", "doden", "oorlog",
         "bom", "lijk"}
_STEMS = ("fuck", "shit", "bitch", "kill", "murder", "suicid", "terror", "bomb", "retard", "piss", "rape", "nazi",
          "porn", "drug", "shoot", "weapon", "death", "blood", "moord", "gedood", "verkracht", "wapen", "neuk", "hoer")
_SKIP_SOURCES = ("weblog-", "newsgroup-")


_ODD = ("@", "http", "www", ".com", "..", "[", "]")   # addresses, runs of dots, and the brackets of web pages


def demo_pool(which: str, max_words: int) -> list:
    """The sentences the demo picks from: 5 to max_words words (the punctuation included), a verb as the root, a
    subject, and nothing that looks like an address, a web page's message or a long run of signs."""
    out = []
    for sent_id, words in TREEBANKS[which]:
        if not 5 <= len(words) <= max_words or sent_id.startswith(_SKIP_SOURCES):
            continue
        if not any(r == "root" and t == "VERB" for _, t, _, r in words):
            continue
        if not any(r.startswith("nsubj") for _, _, _, r in words):
            continue
        if any(len(f) > 18 or f.lower() in _SKIP or f.startswith("!!") or any(x in f for x in _ODD)
               or any(x in f.lower() for x in _STEMS) for f, _, _, _ in words):
            continue
        out.append((sent_id, words))
    return out


def objects_before_verb(which: str) -> tuple:
    """The numbers of the guess, the course's own count: how many objects (obj) of a verb come before that verb in a
    treebank, and how many objects of a verb there are."""
    before = total = 0
    for _, words in TREEBANKS[which]:
        for i, (form, tag, head, rel) in enumerate(words, start=1):
            if rel == "obj" and head and words[head - 1][1] == "VERB":
                total += 1
                before += i < head
    return before, total


def who_did_what(words) -> list:
    """For each subject (nsubj, so no subject of a passive) whose head is a verb: (subject, verb, object or None),
    the forms, from the columns of UD."""
    rows = []
    for i, (form, tag, head, rel) in enumerate(words, start=1):
        if rel == "nsubj" and head and words[head - 1][1] == "VERB":
            obj = next((w[0] for w in words if w[2] == head and w[3].split(":")[0] == "obj"), None)
            rows.append((form, words[head - 1][0], obj))
    return rows


# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_CODE = ";font-family:var(--ea-font-code);font-size:0.78rem"
# how each kind of arc is drawn: stroke, width, dashes, label colour
_ARC = {
    "ink": ("var(--ea-ink-2)", 1.3, "", "var(--ea-ink-2)"),
    "bold": ("var(--ea-ink)", 2.6, "", "var(--ea-ink)"),
    "dim": ("var(--ea-ink-3)", 1.0, "", "var(--ea-ink-3)"),
    "extra": ("var(--ea-loss)", 1.6, "5 3", "var(--ea-loss-text)"),
    "missing": ("var(--ea-ink-3)", 1.2, "3 3", "var(--ea-ink-3)"),
}


def _note(text: str) -> str:
    return f'<div style="{_UI}">{html.escape(text)}</div>'


def _table(rows: list, head: list, code: tuple = ()) -> str:
    """A table with a head row; `code` holds the columns in code letters."""
    top = "".join(f'<th style="{_HEAD}">{html.escape(h)}</th>' for h in head)
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{_CODE if j in code else ""}">{html.escape(str(c))}</td>'
                                    for j, c in enumerate(row)) + "</tr>" for row in rows)
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.3rem 0">'
            f'<tr>{top}</tr>{body}</table></div>')


def arcs_svg(words: list, arcs: list, alt: str, tags=None) -> str:
    """A dependency tree as arcs above the words: each arc (d, h, label, kind) goes from head h to word d, with the
    arrow at d; head 0 is the root, drawn as an arrow from above. Words count from 1."""
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
        if tags:
            parts.append(f'<text x="{xs[k]:.1f}" y="{y0 + 34:.1f}" text-anchor="middle" style="font-family:'
                         f'var(--ea-font-code);font-size:10.5px;fill:var(--ea-ink-3)">{html.escape(tags[k])}</text>')
    return (f'<div style="overflow-x:auto"><svg role="img" aria-label="{html.escape(alt)}" viewBox="0 0 {width:.0f} '
            f'{height:.0f}" width="{width:.0f}" height="{height:.0f}" style="max-width:none;display:block">{"".join(parts)}</svg></div>')


def _words(sentence) -> list:
    return [w[0] for w in sentence]


def _course_arcs(sentence, kind: str = "ink") -> list:
    return [(i, w[1], w[2], kind) for i, w in enumerate(sentence, start=1)]


_IDEA_WORDS = {
    "en": {"title": "Dependency tree of the sentence: the late train left the city after midnight",
           "legend": "Each arc goes from a head to a word that depends on it, with the relation as its label; the "
                     "arrow points at the dependent. The arrow from above makes left the root, and after hangs from "
                     "midnight. Under each word: its part of speech."},
    "nl": {"title": "Dependency tree van de zin: de late trein verliet de stad na middernacht",
           "legend": "Elke boog gaat van een hoofdwoord naar een woord dat ervan afhangt, met de relatie als label; "
                     "de pijl wijst naar het afhankelijke woord. De pijl van boven maakt verliet de root, en na hangt "
                     "aan middernacht. Onder elk woord: de woordsoort."},
}


def idea_picture(lang: str) -> str:
    """The dependency tree of the sentence of The idea."""
    w = _IDEA_WORDS[lang]
    sentence = SENTENCES[lang]
    said = "; ".join(f"{rel}({sentence[h - 1][0] if h else 'ROOT'}, {form})" for form, h, rel in sentence)
    alt = f'{w["title"]}: {said}. {w["legend"]}'
    return yours.figure(w["title"], arcs_svg(_words(sentence), _course_arcs(sentence), alt, SENTENCE_TAGS)
                        + _note(w["legend"]), alt=alt)


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


def _numbers(got, n: int) -> bool:
    """Whether got is a list of word numbers, each from 1 to n."""
    return isinstance(got, list) and all(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= n for v in got)


def _short(value, limit: int = 60) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit - 1] + "…"


_DEPS_WORDS = {
    "en": {"title": "Your dependents, drawn as arcs: the late train left the city after midnight",
           "fine": "Your dependents found all 8 arcs, one into each word: each word has one head, and the arrow from "
                   "above makes left the root.",
           "differs": "Your arcs differ from the course's. Purple dashed: an arc your function gives, and the tree does "
                      "not have. Grey dashed: an arc of the tree that your function misses.",
           "not_ids": "With h = {h}, your function gives back {got}: that is not a list of word numbers from 1 to {n}.",
           "error": "Your function stops with an error at h = {h}.",
           "slow": "Your function takes too long."},
    "nl": {"title": "Jouw dependents, getekend als bogen: de late trein verliet de stad na middernacht",
           "fine": "Je dependents vond alle 8 bogen, één naar elk woord: elk woord heeft één hoofdwoord, en de pijl "
                   "van boven maakt verliet de root.",
           "differs": "Je bogen verschillen van die van de course. Paars gestreept: een boog die je functie geeft en "
                      "die niet in de boom zit. Grijs gestreept: een boog van de boom die je functie mist.",
           "not_ids": "Bij h = {h} geeft je functie {got} terug: dat is geen lijst met woordnummers van 1 tot en met {n}.",
           "error": "Je functie stopt met een fout bij h = {h}.",
           "slow": "Je functie doet er te lang over."},
}


def deps_picture(seen) -> str:
    """The tree of the sentence of The idea, drawn from what the learner's dependents gives for every head 0 to 8."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _DEPS_WORDS[seen.lang]
    sentence = SENTENCES[seen.lang]
    n = len(sentence)
    mine, end = set(), ""
    try:
        for h in range(n + 1):
            got = _call(fn, sentence, h)
            if got is _STOPPED:
                end = w["error"].format(h=h)
                break
            if not _numbers(got, n):
                end = w["not_ids"].format(h=h, got=_short(got), n=n)
                break
            mine |= {(d, h) for d in got}
    except Timeout:
        end = w["slow"]
    course = {(i, word[1]) for i, word in enumerate(sentence, start=1)}
    labels = {(i, word[1]): word[2] for i, word in enumerate(sentence, start=1)}
    if end:
        arcs = _course_arcs(sentence, "dim")
        after = end
    else:
        arcs = [(d, h, labels[(d, h)], "ink" if (d, h) in mine else "missing") for d, h in sorted(course)]
        arcs += [(d, h, "", "extra") for d, h in sorted(mine - course)]
        after = w["fine"] if mine == course else w["differs"]
    said = ", ".join(f"{d}←{h}" for d, h in sorted(mine))
    alt = f'{w["title"]}: {said}. {after}'
    return yours.figure(w["title"], arcs_svg(_words(sentence), arcs, alt, SENTENCE_TAGS) + _note(after), alt=alt)


_PATH_WORDS = {
    "en": {"title": "Your path_to_root on: the roof of the house of the baker leaked",
           "fine": "From each of the 9 words, your path_to_root reaches the root. In bold: the path from word 7, "
                   "one of the two longest, with 4 steps: {path}.",
           "differs": "From word {i}, your function gives {got}, and the path is {want}.",
           "not_path": "From word {i}, your function gives back {got}: that is not a list of word numbers from 1 to {n}.",
           "error": "Your function stops with an error from word {i}.",
           "slow": "Your function takes too long from word {i}. Does each step go on from the head?"},
    "nl": {"title": "Jouw path_to_root op: het dak van het huis van de bakker lekte",
           "fine": "Vanaf elk van de 9 woorden komt je path_to_root bij de root. Vetgedrukt: het pad vanaf woord 7, "
                   "een van de twee langste, met 4 stappen: {path}.",
           "differs": "Vanaf woord {i} geeft je functie {got}, en het pad is {want}.",
           "not_path": "Vanaf woord {i} geeft je functie {got} terug: dat is geen lijst met woordnummers van 1 tot en met {n}.",
           "error": "Je functie stopt met een fout vanaf woord {i}.",
           "slow": "Je functie doet te lang over het pad vanaf woord {i}. Gaat elke stap verder bij het hoofdwoord?"},
}


def path_picture(seen) -> str:
    """The tree of the roof sentence, with the learner's path from word 7, the deepest, in bold."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _PATH_WORDS[seen.lang]
    sentence = ROOFS[seen.lang]
    n = len(sentence)
    after, deepest = "", None
    for i in list(range(1, n + 1)):
        try:
            got = _call(fn, sentence, i)
        except Timeout:
            after = w["slow"].format(i=i)
            break
        if got is _STOPPED:
            after = w["error"].format(i=i)
            break
        if not _numbers(got, n):
            after = w["not_path"].format(i=i, got=_short(got), n=n)
            break
        want = path_to_root(sentence, i)
        if got != want:
            after = w["differs"].format(i=i, got=_short(got), want=want)
            break
        if i == 7:
            deepest = got
    bold = set()
    if not after and deepest:
        bold = {(a, b) for a, b in zip(deepest, deepest[1:])}
        words = " → ".join(sentence[k - 1][0] for k in deepest) + {"en": ", the root", "nl": ", de root"}[seen.lang]
        after = w["fine"].format(path=words)
    arcs = [(i, word[1], word[2], "bold" if (i, word[1]) in bold else ("ink" if not bold else "dim"))
            for i, word in enumerate(sentence, start=1)]
    alt = f'{w["title"]}. {after}'
    return yours.figure(w["title"], arcs_svg(_words(sentence), arcs, alt, ROOF_TAGS) + _note(after), alt=alt)


_DEMO_WORDS = {
    "en": {"title": "A sentence of {name}", "none": "No sentence of {name} has between 5 and {n} words, a verb as "
                                                    "its root and a subject.",
           "who": ["subject", "verb", "object"], "cols": ["ID", "FORM", "UPOS", "HEAD", "DEPREL"],
           "who_note": "Who did what, read off the relations: each nsubj of a verb, with an obj of that verb if "
                       "there is one.",
           "cols_note": "The same sentence in the columns of CoNLL-U that the arcs come from (sent_id {sid}).",
           "names": ["label", "name in UD"],
           "names_note": "The labels of this sentence, with the names the UD page of relations gives them.",
           "sub": "A label with a colon, such as {label} here, is a subtype: a universal relation, made more "
                  "precise."},
    "nl": {"title": "Een zin uit {name}", "none": "Geen zin uit {name} heeft tussen 5 en {n} woorden, een werkwoord "
                                                  "als root en een onderwerp.",
           "who": ["onderwerp", "werkwoord", "lijdend voorwerp"], "cols": ["ID", "FORM", "UPOS", "HEAD", "DEPREL"],
           "who_note": "Wie deed wat, afgelezen aan de relaties: elke nsubj van een werkwoord, met een obj van dat "
                       "werkwoord als dat er is.",
           "cols_note": "Dezelfde zin in de kolommen van CoNLL-U waar de bogen uit komen (sent_id {sid}).",
           "names": ["label", "naam in UD"],
           "names_note": "De labels van deze zin, met de namen die de pagina met relaties van UD ze geeft, in het "
                         "Engels.",
           "sub": "Een label met een dubbele punt, zoals hier {label}, is een subtype: een universele relatie, "
                  "preciezer gemaakt."},
}


# the names that the UD page of relations gives the 37 universal relations and the subtypes of the two treebanks it lists
UD_NAMES = {
    "acl": "clausal modifier of noun (adnominal clause)", "acl:relcl": "relative clause modifier",
    "advcl": "adverbial clause modifier", "advcl:relcl": "adverbial relative clause modifier",
    "advmod": "adverbial modifier", "amod": "adjectival modifier", "appos": "appositional modifier",
    "aux": "auxiliary", "aux:pass": "passive auxiliary", "case": "case marking", "cc": "coordinating conjunction",
    "cc:preconj": "preconjunct", "ccomp": "clausal complement", "clf": "classifier", "compound": "compound",
    "compound:prt": "phrasal verb particle", "conj": "conjunct", "cop": "copula", "csubj": "clausal subject",
    "csubj:outer": "outer clause clausal subject", "dep": "unspecified dependency", "det": "determiner",
    "discourse": "discourse element", "dislocated": "dislocated elements", "expl": "expletive",
    "expl:pv": "reflexive clitic with an inherently reflexive verb", "fixed": "fixed multiword expression",
    "flat": "flat expression", "goeswith": "goes with", "iobj": "indirect object", "list": "list", "mark": "marker",
    "nmod": "nominal modifier", "nmod:poss": "possessive nominal modifier", "nsubj": "nominal subject",
    "nsubj:outer": "outer clause nominal subject", "nsubj:pass": "passive nominal subject",
    "nummod": "numeric modifier", "obj": "object", "obl": "oblique nominal",
    "obl:agent": "oblique agent in passive construction", "obl:arg": "oblique argument", "orphan": "orphan",
    "parataxis": "parataxis", "punct": "punctuation", "reparandum": "overridden disfluency", "root": "root",
    "vocative": "vocative", "xcomp": "open clausal complement",
}


def ud_name(label: str, lang: str) -> str:
    """The name of a relation as the UD page of relations gives it; for a subtype the page does not list, the name
    of its universal relation."""
    if label in UD_NAMES:
        return UD_NAMES[label]
    base = label.split(":")[0]
    if base in UD_NAMES:
        return {"en": "a subtype of {b}, {n}", "nl": "een subtype van {b}, {n}"}[lang].format(b=base, n=UD_NAMES[base])
    return "–"


def demo_picture(which: str, max_words: int, seed: int, lang: str) -> str:
    """A random sentence of a treebank, as arcs, with who did what and its columns of CoNLL-U."""
    w = _DEMO_WORDS[lang]
    which = which if which in TREEBANKS else lang
    name = TREEBANK_NAMES[which]
    pool = demo_pool(which, max(5, min(int(max_words), 20)))
    if not pool:
        text = w["none"].format(name=name, n=max_words)
        return yours.figure(w["title"].format(name=name), _note(text), alt=text)
    sent_id, words = random.Random(31 * int(seed) + int(max_words) + (0 if which == "en" else 7)).choice(pool)
    forms = [f for f, _, _, _ in words]
    arcs = [(i, h, rel, "ink") for i, (_, _, h, rel) in enumerate(words, start=1)]
    said = " ".join(forms)
    title = w["title"].format(name=name)
    who = who_did_what(words)
    rows = [(s, v, o if o else "–") for s, v, o in who]
    cols = [(i, f, t, h, r) for i, (f, t, h, r) in enumerate(words, start=1)]
    labels = sorted({r for _, _, _, r in words})
    names = [(r, ud_name(r, lang)) for r in labels]
    subs = [r for r in labels if ":" in r]
    body = (arcs_svg(forms, arcs, f"{title}: {said}", [t for _, t, _, _ in words])
            + (_note(w["who_note"]) + _table(rows, w["who"]) if rows else "")
            + _note(w["cols_note"].format(sid=sent_id)) + _table(cols, w["cols"], code=(2, 3, 4))
            + _note(w["names_note"]) + _table(names, w["names"], code=(0,))
            + (_note(w["sub"].format(label=subs[0])) if subs else ""))
    return yours.figure(title, body, alt=f"{title}: {said}")


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["dependents", "path_to_root", "relations", "case"]


class Named:
    """A value the check gives the learner's function, shown in a message by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


SENTENCE_N = Named("SENTENCE", SENTENCES["en"])   # the answers are word numbers: the same in both languages
ROOF_N = Named("ROOF", ROOFS["en"])

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


# exercise 1: dependents
DEPENDENTS_CASES = [
    (SENTENCE_N, 4),
    (SENTENCE_N, 3),
    (SENTENCE_N, 0),
    (SENTENCE_N, 1),
    (ROOF_N, 5),
    (ROOF_N, 9),
    (DOOR, 1),
    (HELLO, 0),
]
_DEP_START = "def dependents(sentence, h):\n    ...\n"
_DEP = ("def dependents(sentence, h):\n"
        "    found = []\n"
        "    for i in range(1, len(sentence) + 1):\n"
        "        form, head, relation = sentence[i - 1]\n"
        "        if head == h:\n"
        "            found.append(i)\n"
        "    return found\n")
_DEP_FORMS = _DEP.replace("found.append(i)", "found.append(form)")
_DEP_ZERO = _DEP.replace("range(1, len(sentence) + 1)", "range(len(sentence))").replace("sentence[i - 1]", "sentence[i]")
_DEP_SHORT = _DEP.replace("range(1, len(sentence) + 1)", "range(1, len(sentence))")
_DEP_UP = _DEP.replace("        if head == h:\n            found.append(i)\n",
                       "        if i == h:\n            found.append(head)\n")
_DEP_FIRST = _DEP.replace("            found.append(i)\n", "            return [i]\n")
_DEP_EARLY = _DEP.replace("    return found\n", "        return found\n")

# exercise 2: path_to_root
PATH_CASES = [
    (SENTENCE_N, 1),
    (SENTENCE_N, 4),
    (SENTENCE_N, 7),
    (ROOF_N, 7),
    (ROOF_N, 3),
    (DOOR, 2),
    (HELLO, 1),
]
_P_START = "def path_to_root(sentence, i):\n    ...\n"
_P = ("def path_to_root(sentence, i):\n"
      "    path = [i]\n"
      "    for step in range(len(sentence)):\n"
      "        form, head, relation = sentence[i - 1]\n"
      "        if head == 0:\n"
      "            return path\n"
      "        path.append(head)\n"
      "        i = head\n"
      "    return path\n")
_P_NOSTART = _P.replace("path = [i]", "path = []")
_P_ZERO = _P.replace("        if head == 0:\n            return path\n        path.append(head)\n",
                     "        path.append(head)\n        if head == 0:\n            return path\n")
_P_STAY = _P.replace("        i = head\n", "")
_P_FORMS = _P.replace("path = [i]", "path = [sentence[i - 1][0]]").replace("path.append(head)",
                                                                            "path.append(sentence[head - 1][0])")
_P_ONE = _P.replace("        i = head\n", "        return path\n")
_P_INDEX = _P.replace("sentence[i - 1]", "sentence[i]")


EXERCISES = {
    "dependents": FunctionBox(
        name="dependents",
        picture=deps_picture,
        function="dependents",
        start=_DEP_START,
        solution=_DEP,
        visible=DEPENDENTS_CASES[:1],
        hidden=DEPENDENTS_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("index", error=_index_error,
                    en="With `{call}`, your function looks past the end of the list. Word number `i` is at place `i - 1`: the words count from 1, the places in a list from 0.",
                    nl="Met `{call}` kijkt je functie voorbij het eind van de lijst. Woord nummer `i` staat op plaats `i - 1`: de woorden tellen vanaf 1, de plaatsen in een lijst vanaf 0."),
            Mistake("forms", wrong=_DEP_FORMS,
                    en="`{call}` gives back the words themselves, and it should give their numbers: {want}. The form of word number `i` is at place `i - 1`; the number is `i` itself.",
                    nl="`{call}` geeft de woorden zelf terug, en het moeten hun nummers zijn: {want}. Het woord met nummer `i` staat op plaats `i - 1`; het nummer is `i` zelf."),
            Mistake("zero", wrong=_DEP_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: each number is one too low. Your function gives places in the list, which count from 0; word numbers count from 1.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elk nummer is één te laag. Je functie geeft plaatsen in de lijst, en die tellen vanaf 0; woordnummers tellen vanaf 1."),
            Mistake("short", wrong=_DEP_SHORT,
                    en="`{call}` gives {got}, and it should give {want}: your function never looks at the last word. Let the numbers go up to and including the length of the sentence.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie kijkt nooit naar het laatste woord. Laat de nummers lopen tot en met de lengte van de zin."),
            Mistake("up", wrong=_DEP_UP,
                    en="`{call}` gives {got}, and it should give {want}: your function gives the head of word `h`. Give the words that have `h` as their head.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft het hoofdwoord van woord `h`. Geef de woorden die `h` als hoofdwoord hebben."),
            Mistake("first", wrong=_DEP_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: your function stops at the first dependent. Collect all of them, and give back the list after the loop.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie stopt bij het eerste afhankelijke woord. Verzamel ze allemaal, en geef de lijst na de loop terug."),
            Mistake("early", wrong=_DEP_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back the list after the first word. Move `return found` to the left, under `for`, so that it comes after the loop.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft de lijst al na het eerste woord terug. Zet `return found` meer naar links, onder `for`, zodat die na de loop komt."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: go over the word numbers, and collect the numbers of the words whose head is `h`.",
                      "nl": "Vervang de `...`: loop langs de woordnummers, en verzamel de nummers van de woorden met `h` als hoofdwoord."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The function gives back the numbers of the words whose head is `h`, from left to right.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De functie geeft de nummers terug van de woorden met `h` als hoofdwoord, van links naar rechts."},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function use the `sentence` and `h` it gets?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `sentence` en `h` die ze krijgt?"},
            "correct": {"en": "Your `dependents` is right in all {n} cases, also for the root and for a word with no dependents. Look at the picture: the tree, drawn from your function.",
                        "nl": "Je `dependents` klopt in alle {n} gevallen, ook voor de root en voor een woord zonder afhankelijke woorden. Kijk naar de afbeelding: de boom, getekend met je functie."},
        },
    ),
    "path_to_root": FunctionBox(
        name="path_to_root",
        picture=path_picture,
        function="path_to_root",
        start=_P_START,
        solution=_P,
        visible=PATH_CASES[:1],
        hidden=PATH_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("index", error=_index_error,
                    en="With `{call}`, your function looks past the end of the list. Word number `i` is at place `i - 1`: the words count from 1, the places in a list from 0.",
                    nl="Met `{call}` kijkt je functie voorbij het eind van de lijst. Woord nummer `i` staat op plaats `i - 1`: de woorden tellen vanaf 1, de plaatsen in een lijst vanaf 0."),
            Mistake("no-start", wrong=_P_NOSTART,
                    en="`{call}` gives {got}, and it should give {want}: the word you start from is missing. The path begins with word `i` itself.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het woord waar je begint, ontbreekt. Het pad begint bij woord `i` zelf."),
            Mistake("zero", wrong=_P_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: your path ends with 0. Head 0 means the word is the root; 0 is not a word. Stop before you add it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je pad eindigt met 0. Hoofdwoord 0 betekent dat het woord de root is; 0 is geen woord. Stop voordat je het toevoegt."),
            Mistake("stay", wrong=_P_STAY,
                    en="`{call}` gives {got}, and it should give {want}: your path adds the same head again and again. After each step, go on from the head: make it the new `i`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je pad voegt steeds hetzelfde hoofdwoord toe. Ga na elke stap verder bij het hoofdwoord: maak dat de nieuwe `i`."),
            Mistake("forms", wrong=_P_FORMS,
                    en="`{call}` gives back the words themselves, and it should give their numbers: {want}. Put `i` and the numbers of the heads in the path.",
                    nl="`{call}` geeft de woorden zelf terug, en het moeten hun nummers zijn: {want}. Zet `i` en de nummers van de hoofdwoorden in het pad."),
            Mistake("one", wrong=_P_ONE,
                    en="`{call}` gives {got}, and it should give {want}: your function stops after one step. Go on from the head until you reach the root.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie stopt na één stap. Ga verder bij het hoofdwoord tot je bij de root bent."),
            Mistake("index-wrong", wrong=_P_INDEX,
                    en="`{call}` gives {got}, and it should give {want}: for word number `i`, your function reads the word after it. Word number `i` is at place `i - 1`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: voor woord nummer `i` leest je functie het woord erna. Woord nummer `i` staat op plaats `i - 1`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: follow the heads from word `i` until a word has head 0, and give back the numbers on the way.",
                      "nl": "Vervang de `...`: volg de hoofdwoorden vanaf woord `i` tot een woord hoofdwoord 0 heeft, en geef de nummers onderweg terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The function gives back the numbers of the words from word `i` up to the root, the root included.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De functie geeft de nummers terug van de woorden vanaf woord `i` tot en met de root."},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function use the `sentence` and `i` it gets, also when `i` is the root?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `sentence` en `i` die ze krijgt, ook als `i` de root is?"},
            "correct": {"en": "Your `path_to_root` is right in all {n} cases, also from the root itself. Look at the picture: one of the two longest paths of a longer sentence.",
                        "nl": "Je `path_to_root` klopt in alle {n} gevallen, ook vanaf de root zelf. Kijk naar de afbeelding: een van de twee langste paden van een langere zin."},
        },
    ),
    "relations": Sort(
        name="relations",
        groups={
            "nsubj": {"en": "nsubj", "nl": "nsubj"},
            "obj": {"en": "obj", "nl": "obj"},
            "det": {"en": "det", "nl": "det"},
            "amod": {"en": "amod", "nl": "amod"},
        },
        items={
            "a": {"en": "**A**", "nl": "**Een**"},
            "tired": {"en": "**tired**", "nl": "**vermoeide**"},
            "guard": {"en": "**guard**", "nl": "**bewaker**"},
            "the": {"en": "**the**", "nl": "**de**"},
            "heavy": {"en": "**heavy**", "nl": "**zware**"},
            "gate": {"en": "**gate**", "nl": "**poort**"},
        },
        solution={"a": "det", "tired": "amod", "guard": "nsubj", "the": "det", "heavy": "amod", "gate": "obj"},
        nudge={
            "a": {"en": "Does *A* describe the guard, or is it a small word before the noun, as *the* is in *the late train*?",
                  "nl": "Beschrijft *Een* de bewaker, of is het een klein woord vóór het zelfstandig naamwoord, zoals *de* in *de late trein*?"},
            "tired": {"en": "Does *tired* describe the guard, as *late* describes the train in *the late train*?",
                      "nl": "Beschrijft *vermoeide* de bewaker, zoals *late* de trein beschrijft in *de late trein*?"},
            "guard": {"en": "Who opened something?",
                      "nl": "Wie opende iets?"},
            "the": {"en": "Does *the* describe the gate, or is it a small word before the noun, as *the* is in *the late train*?",
                    "nl": "Beschrijft *de* de poort, of is het een klein woord vóór het zelfstandig naamwoord, zoals *de* in *de late trein*?"},
            "heavy": {"en": "Does *heavy* describe the gate, as *late* describes the train in *the late train*?",
                      "nl": "Beschrijft *zware* de poort, zoals *late* de trein beschrijft in *de late trein*?"},
            "gate": {"en": "What was opened?",
                     "nl": "Wat werd er geopend?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the words again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de woorden opnieuw uit."},
            "empty": {"en": "Give each bold word the relation to its head.",
                      "nl": "Geef elk vetgedrukt woord de relatie met het hoofdwoord ervan."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} belongs under another relation.",
                      "nl": "{item} hoort bij een andere relatie."},
            "correct": {"en": "All {n} are right. The subject and the object hang from the verb, and the determiners and adjectives from their nouns.",
                        "nl": "Alle {n} kloppen. Het onderwerp en het lijdend voorwerp hangen aan het werkwoord, en de lidwoorden en bijvoeglijke naamwoorden aan hun zelfstandig naamwoord."},
        },
    ),
}

CASES = [
    ("dependents", _DEP_START, "empty", None),
    ("dependents", _DEP, "correct", None),
    ("dependents", "def dependents(sentence, h):\n    return [i + 1 for i in range(len(sentence)) if sentence[i][1] == h]\n", "correct", None),
    ("dependents", "def dependents(sentence, h):\n    found = []\n    i = 0\n    for form, head, relation in sentence:\n        i = i + 1\n        if head == h:\n            found.append(i)\n    return found\n", "correct", None),
    ("dependents", _DEP_FORMS, "wrong", "forms"),
    ("dependents", _DEP_ZERO, "wrong", "zero"),
    ("dependents", _DEP_SHORT, "wrong", "short"),
    ("dependents", "def dependents(sentence, h):\n    found = []\n    for i in range(1, len(sentence) + 2):\n        form, head, relation = sentence[i - 1]\n        if head == h:\n            found.append(i)\n    return found\n", "error", "index"),
    ("dependents", _DEP_UP, "wrong", "up"),
    ("dependents", _DEP_FIRST, "wrong", "first"),
    ("dependents", _DEP_EARLY, "wrong", "early"),
    ("dependents", "def dependents(sentence, h):\n    return [3, 6, 8]\n", "almost", None),
    ("path_to_root", _P_START, "empty", None),
    ("path_to_root", _P, "correct", None),
    ("path_to_root", "def path_to_root(sentence, i):\n    path = [i]\n    for step in sentence:\n        head = sentence[path[-1] - 1][1]\n        if head == 0:\n            break\n        path.append(head)\n    return path\n", "correct", None),
    ("path_to_root", _P_NOSTART, "wrong", "no-start"),
    ("path_to_root", _P_ZERO, "wrong", "zero"),
    ("path_to_root", _P_STAY, "wrong", "stay"),
    ("path_to_root", _P_FORMS, "wrong", "forms"),
    ("path_to_root", _P_ONE, "wrong", "one"),
    ("path_to_root", _P_INDEX, "wrong", "index-wrong"),
    ("path_to_root", _P.replace("    path = [i]\n", "    path = [i, sentence[len(sentence)][1]]\n"), "error", "index"),
    ("path_to_root", "def path_to_root(sentence, i):\n    return [1, 3, 4]\n", "almost", None),
    ("relations", {}, "empty", None),
    ("relations", {"a": "det", "tired": "amod", "guard": "nsubj", "the": "det", "heavy": "amod", "gate": "obj"},
     "correct", None),
    ("relations", {"a": "det", "tired": "amod"}, "wrong", "unfinished"),
    ("relations", {"guard": "obj"}, "wrong", "wrong:guard"),
    ("relations", {"a": "det", "tired": "det", "guard": "nsubj", "the": "det", "heavy": "amod", "gate": "obj"},
     "wrong", "wrong:tired"),
]
