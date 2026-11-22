# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-stemming: apply one group of suffix rules to a word, the rule of the longest suffix first
(apply_rules), count Porter's measure m of a word (measure), and look a word up in a table of lemmas, with the word
itself when the table does not have it (lemmatize); and sort tasks by what they need: a stemmer, a lemmatizer, or
the forms as they are.

The words and their lemmas come from the treebanks UD_English-EWT and UD_Dutch-Alpino of Universal Dependencies,
version 2.19 (CC BY-SA 4.0), in data/ud-en-ewt.conllu.gz and data/ud-nl-alpino.conllu.gz. The lesson counts every word
that the treebank does not mark as a name, a number, punctuation, a symbol or other (the tags PROPN, NUM, PUNCT, SYM
and X), in small letters. The stems come from the course's own stemmers in everything_ai.stemmers: Porter's stemmer
for English and Porter's stemmer for Dutch. Every number in the pictures is counted by the course; the words of the
cases and the items of the sort are the course's own."""
import collections
import contextlib
import copy
import gzip
import html
import io
import re
import types

from everything_ai import paths, stemmers, yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, kept, with_time_limit

# --- the course's solutions ------------------------------------------------------------------------------------------

STEP_1A = {"sses": "ss", "ies": "i", "ss": "ss", "s": ""}   # step 1a of Porter's stemmer, the longest suffix first


def apply_rules(word, rules):
    """The course's solution of Exercise 1: the first rule of `rules` whose suffix ends the word decides: the word
    without that suffix, with the rule's new ending. When no rule fits, the word as it is."""
    for suffix in rules:
        if word.endswith(suffix):
            return word[:len(word) - len(suffix)] + rules[suffix]
    return word


vowel = stemmers.vowel   # Porter's vowel: a, e, i, o, u, or a y after a consonant; the lesson shows its code


def measure(word):
    """The course's solution of Exercise 2: Porter's m, how often a vowel is followed by a consonant."""
    m = 0
    for i in range(1, len(word)):
        if vowel(word, i - 1) and not vowel(word, i):
            m = m + 1
    return m


def lemmatize(word, table):
    """The course's solution of Exercise 3: the lemma that `table` gives the word, or the word itself."""
    if word in table:
        return table[word]
    return word


# --- the data --------------------------------------------------------------------------------------------------------

UD = {"en": paths.ROOT / "data" / "ud-en-ewt.conllu.gz", "nl": paths.ROOT / "data" / "ud-nl-alpino.conllu.gz"}
LEFT_OUT = {"PROPN", "NUM", "PUNCT", "SYM", "X"}   # names, numbers, punctuation, symbols and other
STEMMER = {"en": stemmers.porter, "nl": stemmers.dutch}
TREEBANK = {"en": "EWT", "nl": "Alpino"}   # the treebank of each language, as the pictures name it


def _read(lang: str) -> tuple:
    """Every word of the treebank that is not left out, as (form, lemma, split), both in small letters; split is
    train, dev or test."""
    out, split = [], None
    with gzip.open(UD[lang], "rt", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# split = "):
                split = line[len("# split = "):]
            elif line and not line.startswith("#"):
                cols = line.split("\t")
                if "-" in cols[0] or "." in cols[0] or cols[3] in LEFT_OUT:
                    continue
                out.append((cols[1].lower(), cols[2].lower(), split))
    return tuple(out)


def words(lang: str = "en") -> tuple:
    """The words of UD_English-EWT (en) or UD_Dutch-Alpino (nl) as (form, lemma, split)."""
    return kept(f"nlp-stemming:words-{lang}", lambda: _read(lang))


def form_counts(lang: str = "en") -> collections.Counter:
    """How often each form occurs, over the whole treebank."""
    return kept(f"nlp-stemming:forms-{lang}", lambda: collections.Counter(w for w, _, _ in words(lang)))


def _lemma_counts(lang: str, split=None) -> dict:
    out = collections.defaultdict(collections.Counter)
    for w, lemma, s in words(lang):
        if split is None or s == split:
            out[w][lemma] += 1
    return dict(out)


def lemma_counts(lang: str = "en") -> dict:
    """For each form, how often the whole treebank gives it each lemma: liep gets lopen 17 times and op_lopen 7."""
    return kept(f"nlp-stemming:lemma-counts-{lang}", lambda: _lemma_counts(lang))


def _best(counts: collections.Counter) -> str:
    """The most common lemma; on a tie, the first in alphabetical order."""
    return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def lemma_of(lang: str = "en") -> dict:
    """For each form of the whole treebank, its most common lemma."""
    return kept(f"nlp-stemming:lemma-of-{lang}",
                lambda: types.MappingProxyType({w: _best(c) for w, c in lemma_counts(lang).items()}))


def table_kept(lang: str = "en"):
    """The course's table of lemmas: for each form of the training part of the treebank, its most common lemma
    there. It cannot be changed."""
    return kept(f"nlp-stemming:table-{lang}",
                lambda: types.MappingProxyType({w: _best(c) for w, c in _lemma_counts(lang, "train").items()}))


def table(lang: str = "en") -> dict:
    """The table of lemmas as a new dictionary, for the lesson's TABLE and TABLE_NL."""
    return dict(table_kept(lang))


def test_words(lang: str = "en") -> list:
    """The words of the test part of the treebank, as (form, lemma): a lemmatizer is measured on words that its
    table was not made from."""
    return [(w, lemma) for w, lemma, s in words(lang) if s == "test"]


def stems(lang: str = "en") -> dict:
    """For each form, its stem by the course's stemmer for the language."""
    return kept(f"nlp-stemming:stems-{lang}",
                lambda: types.MappingProxyType({w: STEMMER[lang](w) for w in form_counts(lang)}))


# the course names nobody by a gendered pronoun: pictures leave out a group that holds one
_PRONOUNS = {"hij", "zij", "haar", "hem", "zijn", "ze", "z'n", "he", "she", "him", "her", "his", "hers", "himself",
             "herself"}


def _clean(forms_) -> bool:
    return not set(forms_) & _PRONOUNS


def groups(key: dict) -> dict:
    """The forms by what `key` gives them, such as their stem or their lemma: {stem: [form, ...]}."""
    out = collections.defaultdict(list)
    for w, k in key.items():
        out[k].append(w)
    return dict(out)


def compare(stem_of: dict, lemma: dict) -> dict:
    """What a stemmer does against the lemmas of the treebank: how many groups each makes, how many stems bring forms
    of more than one lemma together (overstemming), and how many lemmas have their forms under more than one stem
    (understemming)."""
    by_stem, by_lemma = groups(stem_of), groups({w: lemma[w] for w in stem_of})
    over = [s for s, fs in by_stem.items() if len({lemma[f] for f in fs}) > 1]
    under = [lm for lm, fs in by_lemma.items() if len({stem_of[f] for f in fs}) > 1]
    return {"forms": len(stem_of), "stems": len(by_stem), "lemmas": len(by_lemma), "over": over, "under": under,
            "by_stem": by_stem, "by_lemma": by_lemma}


def compare_kept(lang: str = "en") -> dict:
    return kept(f"nlp-stemming:compare-{lang}", lambda: compare(dict(stems(lang)), lemma_of(lang)))


def frequent(fs, lang: str, least: int = 3) -> list:
    """The forms of letters only that occur at least `least` times, the most common first: a typo occurs once or
    twice, and a piece such as 's or n't is not a word a reader would look up."""
    count = form_counts(lang)
    return sorted([f for f in fs if count[f] >= least and f.isalpha()], key=lambda f: (-count[f], f))


def examples_over(lang: str, stem_of: dict, k: int = 4, least: int = 3) -> list:
    """The stems that bring forms of different lemmas together most often, as (stem, [form, ...]), with only forms
    that occur at least `least` times."""
    lemma, count = lemma_of(lang), form_counts(lang)
    out = []
    for s, fs in groups(stem_of).items():
        shown = frequent(fs, lang, least)
        if len({lemma[f] for f in shown}) > 1 and _clean(fs) and _clean([lemma[f] for f in fs]):
            out.append((sum(count[f] for f in shown), s, shown))
    out.sort(key=lambda t: (-t[0], t[1]))
    return [(s, fs) for _, s, fs in out[:k]]


def examples_under(lang: str, stem_of: dict, k: int = 4, least: int = 3) -> list:
    """The lemmas whose forms are spread over more than one stem, the most common first, as (lemma, [form, ...])."""
    lemma, count = lemma_of(lang), form_counts(lang)
    out = []
    for lm, fs in groups({w: lemma[w] for w in stem_of}).items():
        shown = frequent(fs, lang, least)
        if len({stem_of[f] for f in shown}) > 1 and _clean(fs) and lm not in _PRONOUNS:
            out.append((sum(count[f] for f in shown), lm, shown))
    out.sort(key=lambda t: (-t[0], t[1]))
    return [(lm, fs) for _, lm, fs in out[:k]]


# --- the look of the pictures ------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums"
_LINE = "font-family:var(--ea-font-ui);font-size:0.9rem;color:var(--ea-ink);line-height:1.9"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);margin-top:0.35rem"
_CHIP = ("display:inline-block;margin:0.12rem 0.2rem 0.12rem 0;padding:0 0.3rem;border-radius:3px;"
         "font-family:var(--ea-font-code);font-size:0.82rem;line-height:1.5;white-space:pre")
_CHIP_BLUE = _CHIP + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
_CHIP_PURPLE = _CHIP + ";background:var(--ea-loss-wash);color:var(--ea-loss-text)"
_CHIP_DATA = _CHIP + ";border:1px solid var(--ea-line);color:var(--ea-ink)"


def _note(text: str) -> str:
    """A line under a picture; a word between asterisks is set in italics, as in the lesson's text."""
    parts = html.escape(text).split("*")
    marked = "".join(f"<i>{p}</i>" if k % 2 else p for k, p in enumerate(parts))
    return f'<div style="{_UI}">{marked}</div>'


def _label(text: str) -> str:
    return f'<div style="{_LABEL}">{html.escape(text)}</div>'


def _num(n, lang: str) -> str:
    """A whole number with a separator for thousands: 2,319 in English, 2.319 in Dutch."""
    text = f"{round(n):,}"
    return text.replace(",", ".") if lang == "nl" else text


def _pct(part: int, whole: int, lang: str) -> str:
    """A share as a percentage with one decimal: 96.6% in English, 96,6% in Dutch."""
    text = f"{100 * part / whole:.1f}%" if whole else "–"
    return text.replace(".", ",") if lang == "nl" else text


def _table(rows_: list, head: list, numbers: tuple = ()) -> str:
    """A small table; rows hold HTML that is already escaped; numbers are the columns set to the right."""
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{_NUMBER if j in numbers else ""}">{c}</td>'
                                    for j, c in enumerate(row)) + "</tr>" for row in rows_)
    return (f'<table style="border-collapse:collapse;margin:0.3rem 0 0.2rem"><thead><tr>{top}</tr></thead>'
            f"<tbody>{body}</tbody></table>")


def _chip(text: str, style: str = _CHIP_DATA) -> str:
    return f'<span style="{style}">{html.escape(text)}</span>'


def _chips(pieces, style: str = _CHIP_DATA) -> str:
    return "".join(_chip(p, style) for p in pieces)


# --- the idea's picture: forms, their stems and their lemmas ----------------------------------------------------------

IDEA = {"en": [["running", "runs", "ran"], ["university", "universe"], ["use", "us"]],
        "nl": [["lopen", "loopt", "liep", "gelopen"], ["gaan", "gaat", "gat"]]}
_IDEA_WORDS = {
    "en": {"title": "Forms, stems and lemmas",
           "head": ["form", "stem", "lemma"],
           "lang": {"EWT": "English: Porter's stemmer, and the lemmas of UD_English-EWT",
                    "Alpino": "Dutch: Porter's stemmer for Dutch, and the lemmas of UD_Dutch-Alpino"},
           "note": "Blue: the stem the stemmer gives. Purple: a stem that does not match the lemmas: it brings forms "
                   "of two lemmas together, or it stands apart from the most common stem of its lemma. The lemmas "
                   "come from the treebank.",
           "alt": "{title}. {rows}. {note}"},
    "nl": {"title": "Vormen, stammen en lemma's",
           "head": ["vorm", "stam", "lemma"],
           "lang": {"EWT": "Engels: de stemmer van Porter, en de lemma's van UD_English-EWT",
                    "Alpino": "Nederlands: de stemmer van Porter voor het Nederlands, en de lemma's van UD_Dutch-Alpino"},
           "note": "Blauw: de stam die de stemmer geeft. Paars: een stam die niet bij de lemma's past: de stam brengt "
                   "vormen van twee lemma's samen, of staat los van de meest voorkomende stam van het lemma. De "
                   "lemma's komen uit de treebank.",
           "alt": "{title}. {rows}. {note}"},
}


def odd_stems(forms_: list, lang: str) -> set:
    """The forms of a shown group whose stem does not match the lemmas: the stem is shared with a form of another
    lemma, or it is not the most common stem of the forms of its lemma (counted by how often each form occurs)."""
    lemma, count, stem_of = lemma_of(lang), form_counts(lang), stems(lang)
    odd = set()
    for f in forms_:
        if any(stem_of[g] == stem_of[f] and lemma[g] != lemma[f] for g in forms_):
            odd.add(f)
        votes = collections.Counter()
        for g in forms_:
            if lemma[g] == lemma[f]:
                votes[stem_of[g]] += count[g]
        if stem_of[f] != _best(votes):
            odd.add(f)
    return odd


def idea_picture(lang: str) -> str:
    w = _IDEA_WORDS[lang]
    body, alts = "", []
    for data_lang in ("en", "nl"):
        lemma, stem_of = lemma_of(data_lang), stems(data_lang)
        rows_ = []
        for forms_ in IDEA[data_lang]:
            odd = odd_stems(forms_, data_lang)
            for f in forms_:
                rows_.append([_chip(f), _chip(stem_of[f], _CHIP_PURPLE if f in odd else _CHIP_BLUE),
                              _chip(lemma[f])])
                alts.append(f"{f}: {stem_of[f]}, {lemma[f]}" + (" (purple)" if f in odd and lang == "en" else
                                                                 " (paars)" if f in odd else ""))
        body += _label(w["lang"][TREEBANK[data_lang]]) + _table(rows_, w["head"])
    body += _note(w["note"])
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows="; ".join(alts), note=w["note"]))


# --- the comparison: Porter's stems against the lemmas -------------------------------------------------------------

_COMPARE_WORDS = {
    "en": {"title": "Stems against lemmas",
           "head": ["", "English, EWT", "Dutch, Alpino"],
           "forms": "different forms", "lemmas": "groups by lemma", "stems": "groups by stem",
           "over": "stems that bring forms of more than one lemma together",
           "under": "lemmas whose forms get more than one stem",
           "over_label": "Together under one stem, with different lemmas:",
           "under_label": "One lemma, more than one stem:",
           "note": "Each form counts with its most common lemma. The counts hold every form, also typos: EWT gives "
                   "a typo the lemma of the word meant, such as *teh* the lemma *the*, which no stemmer can see.",
           "alt": "{title}. {rows}. {over} {under} {note}"},
    "nl": {"title": "Stam tegen lemma",
           "head": ["", "Engels, EWT", "Nederlands, Alpino"],
           "forms": "verschillende vormen", "lemmas": "groepen op lemma", "stems": "groepen op stam",
           "over": "stammen die vormen van meer dan één lemma samenbrengen",
           "under": "lemma's waarvan de vormen meer dan één stam krijgen",
           "over_label": "Samen onder één stam, met verschillende lemma's:",
           "under_label": "Eén lemma, meer dan één stam:",
           "note": "Elke vorm telt met het meest voorkomende lemma. De aantallen tellen elke vorm, ook tikfouten: EWT "
                   "geeft een tikfout het lemma van het bedoelde woord, zoals *teh* het lemma *the*, en dat ziet geen "
                   "stemmer.",
           "alt": "{title}. {rows}. {over} {under} {note}"},
}


# the examples of the comparison: forms that one stem brings together while the treebank gives them different lemmas,
# and forms of one lemma that get different stems; tests/test_nlp_stemming.py checks that each still is one
OVER_SHOWN = {"en": [["use", "us"], ["business", "busy"]], "nl": [["het", "heet"], ["zal", "zaal"]]}
UNDER_SHOWN = {"en": [["go", "going", "went", "gone"], ["see", "saw", "seen"]],
               "nl": [["gaan", "gaat", "ging", "gegaan"], ["werken", "werkt", "werkte", "gewerkt"]]}


def _group_line(name: str, forms_: list, stem_of: dict, lemma: dict, by: str) -> str:
    """One example: the stem in blue or the treebank's lemma as data, then each form with what the other side gives it."""
    if by == "stem":
        return "<div>" + _chip(name, _CHIP_BLUE) + " " + _chips([f"{f} → {lemma[f]}" for f in forms_]) + "</div>"
    return "<div>" + _chip(name) + " " + _chips([f"{f} → {stem_of[f]}" for f in forms_]) + "</div>"


def compare_table(results: dict, lang: str) -> tuple:
    """The rows of the comparison table and their text for the alt, for {"en": compare(...), "nl": compare(...)}."""
    w = _COMPARE_WORDS[lang]
    rows_, alts = [], []
    for key in ("forms", "lemmas", "stems", "over", "under"):
        vals = [len(results[d][key]) if isinstance(results[d][key], list) else results[d][key] for d in ("en", "nl")]
        rows_.append([html.escape(w[key])] + [_num(v, lang) for v in vals])
        alts.append(f"{w[key]}: {_num(vals[0], lang)} / {_num(vals[1], lang)}")
    return rows_, alts


def compare_picture(lang: str) -> str:
    w = _COMPARE_WORDS[lang]
    results = {d: compare_kept(d) for d in ("en", "nl")}
    rows_, alts = compare_table(results, lang)
    over_lines, under_lines, over_alt, under_alt = "", "", [], []
    for d in ("en", "nl"):
        stem_of, lemma = stems(d), lemma_of(d)
        for fs in OVER_SHOWN[d]:
            s = stem_of[fs[0]]
            over_lines += _group_line(s, fs, stem_of, lemma, "stem")
            over_alt.append(f"{s}: " + ", ".join(f"{f} ({lemma[f]})" for f in fs))
        for fs in UNDER_SHOWN[d]:
            lm = lemma[fs[0]]
            under_lines += _group_line(lm, fs, stem_of, lemma, "lemma")
            under_alt.append(f"{lm}: " + ", ".join(f"{f} ({stem_of[f]})" for f in fs))
    body = (_table(rows_, w["head"], numbers=(1, 2)) + _label(w["over_label"]) + over_lines
            + _label(w["under_label"]) + under_lines + _note(w["note"]))
    alt = w["alt"].format(title=w["title"], rows="; ".join(alts), over=w["over_label"] + " " + "; ".join(over_alt) + ".",
                          under=w["under_label"] + " " + "; ".join(under_alt) + ".", note=w["note"].replace("*", ""))
    return yours.figure(w["title"], body, alt=alt)


# --- the demo: your own words -----------------------------------------------------------------------------------------

_DEMO_WORDS = {
    "en": {"title": "Your words: stem and lemma",
           "head": ["form", "stem", "lemma in the treebank"],
           "unknown": "not in the treebank",
           "stemmer": {"EWT": "Porter's stemmer, and the lemmas of UD_English-EWT",
                       "Alpino": "Porter's stemmer for Dutch, and the lemmas of UD_Dutch-Alpino"},
           "note": "Blue: the stem. Next to it, the treebank's most common lemma; when the treebank gives a form more "
                   "than one lemma, the others follow with how often. Your words are cut into runs of letters and made small first.",
           "empty": "Type some words in the box.",
           "long": "This text is longer than {k} characters; try a shorter one.",
           "alt": "{title}, with {stemmer}. {rows}. {note}"},
    "nl": {"title": "Jouw woorden: stam en lemma",
           "head": ["vorm", "stam", "lemma in de treebank"],
           "unknown": "niet in de treebank",
           "stemmer": {"EWT": "de stemmer van Porter, en de lemma's van UD_English-EWT",
                       "Alpino": "de stemmer van Porter voor het Nederlands, en de lemma's van UD_Dutch-Alpino"},
           "note": "Blauw: de stam. Daarnaast het meest voorkomende lemma in de treebank; als de treebank een vorm meer "
                   "dan één lemma geeft, volgen de andere met hoe vaak. Je woorden worden eerst in reeksen letters geknipt en klein gemaakt.",
           "empty": "Typ een paar woorden in het vak.",
           "long": "Deze tekst is langer dan {k} tekens; probeer een kortere.",
           "alt": "{title}, met {stemmer}. {rows}. {note}"},
}
DEMO_LONGEST = 200


def demo_words(text: str) -> list:
    """The runs of letters of a text, in small letters, each once, in order."""
    out = []
    for word in re.findall(r"[^\W\d_]+", text.lower()):
        if word not in out:
            out.append(word)
    return out


def demo_picture(text, data_lang: str, lang: str) -> str:
    w = _DEMO_WORDS[lang]
    text = str(text or "")
    if not demo_words(text):
        return yours.figure(w["title"], _note(w["empty"]), alt=w["empty"])
    if len(text) > DEMO_LONGEST:
        msg = w["long"].format(k=DEMO_LONGEST)
        return yours.figure(w["title"], _note(msg), alt=msg)
    data_lang = data_lang if data_lang in ("en", "nl") else "nl"
    counts = lemma_counts(data_lang)
    rows_, alts = [], []
    for f in demo_words(text):
        s = STEMMER[data_lang](f)
        if f in counts:
            ranked = sorted(counts[f].items(), key=lambda kv: (-kv[1], kv[0]))
            lemma_html = _chip(ranked[0][0]) + _chips([f"{lm} {_num(n, lang)}" for lm, n in ranked[1:4]])
            lemma_alt = ", ".join(f"{lm} {_num(n, lang)}" for lm, n in ranked[:4])
        else:
            lemma_html = f'<span style="{_UI}">{html.escape(w["unknown"])}</span>'
            lemma_alt = w["unknown"]
        rows_.append([_chip(f), _chip(s, _CHIP_BLUE), lemma_html])
        alts.append(f"{f}: {s}; {lemma_alt}")
    stemmer = w["stemmer"][TREEBANK[data_lang]]
    body = _label(stemmer) + _table(rows_, w["head"]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], stemmer=stemmer, rows="; ".join(alts),
                                                             note=w["note"]))


# --- running the learner's function ----------------------------------------------------------------------------------


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error
PICTURE_LIMIT = 10.0   # seconds the learner's function gets for all words of a picture


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


def _run_or_say(make, w: dict, ok) -> tuple:
    """(what make() gives, None), or (None, a figure that says why there is no picture): an error, too slow, or a
    result that `ok` does not accept."""
    try:
        got = _call(make)
    except Timeout:
        text = w["slow"].format(s=PICTURE_LIMIT)
        return None, yours.figure(w["title"], _note(text), alt=text)
    if got is _STOPPED:
        return None, yours.figure(w["title"], _note(w["stopped"]), alt=w["stopped"])
    if not ok(got):
        return None, yours.figure(w["title"], _note(w["odd"]), alt=w["odd"])
    return got, None


def _strings(v) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


# --- the picture under exercise 1: step 1a on the words of EWT ---------------------------------------------------------

_RULES_WORDS = {
    "en": {"title": "Your apply_rules with STEP_1A on the words of EWT",
           "head": ["the forms of EWT", "number"],
           "forms": "different forms", "changed": "forms your rules change", "groups": "groups after your rules",
           "right": "Together, and the treebank agrees: one lemma",
           "wrong": "Together, but the treebank gives them different lemmas",
           "note": "Blue: what your function gives; after each form, its lemma in the treebank. The examples show "
                   "forms that occur at least three times. Step 1a is the first of Porter's steps: *running* still "
                   "stays apart from *run*.",
           "stopped": "Your function stopped with an error on one of the words, so the picture stays empty.",
           "slow": "Your function took more than {s:g} seconds for all words, so the picture stays empty.",
           "odd": "Your function does not give back a string for every word, so the picture stays empty."},
    "nl": {"title": "Jouw apply_rules met STEP_1A op de woorden van EWT",
           "head": ["de vormen van EWT", "aantal"],
           "forms": "verschillende vormen", "changed": "vormen die door je regels veranderen",
           "groups": "groepen na je regels",
           "right": "Samen, en de treebank is het ermee eens: één lemma",
           "wrong": "Samen, maar de treebank geeft deze vormen verschillende lemma's",
           "note": "Blauw: wat je functie geeft; na elke vorm het lemma in de treebank. De voorbeelden tonen vormen "
                   "die minstens drie keer voorkomen. Stap 1a is de eerste stap van Porter: *running* staat nog los "
                   "van *run*.",
           "stopped": "Je functie stopte met een fout op een van de woorden, dus de afbeelding blijft leeg.",
           "slow": "Je functie deed er langer dan {s:g} seconden over voor alle woorden, dus de afbeelding blijft leeg.",
           "odd": "Je functie geeft niet voor elk woord een string terug, dus de afbeelding blijft leeg."},
}
RULES_SHOWN = 4


def examples_right(lang: str, stem_of: dict, k: int = 4, least: int = 3) -> list:
    """The groups of more than one form with one lemma, the most common first, as (stem, [form, ...])."""
    lemma, count = lemma_of(lang), form_counts(lang)
    out = []
    for s, fs in groups(stem_of).items():
        shown = frequent(fs, lang, least)
        if len(shown) > 1 and len({lemma[f] for f in fs}) == 1 and _clean(fs) and _clean([lemma[f] for f in fs]):
            out.append((sum(count[f] for f in shown), s, shown))
    out.sort(key=lambda t: (-t[0], t[1]))
    return [(s, fs) for _, s, fs in out[:k]]


def rules_picture(seen) -> str:
    lang = seen.lang
    w = _RULES_WORDS[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    forms_ = sorted(form_counts("en"))
    got, said = _run_or_say(lambda: [fn(f, dict(STEP_1A)) for f in forms_], w, _strings)
    if said:
        return said
    mine = dict(zip(forms_, got))
    lemma = lemma_of("en")
    rows_ = [[html.escape(w["forms"]), _num(len(forms_), lang)],
             [html.escape(w["changed"]), _num(sum(1 for f in forms_ if mine[f] != f), lang)],
             [html.escape(w["groups"]), _num(len(set(got)), lang)]]
    right = examples_right("en", mine, RULES_SHOWN)
    wrong = examples_over("en", mine, RULES_SHOWN)
    lines = {key: "".join(_group_line(s, fs, mine, lemma, "stem") for s, fs in found)
             for key, found in (("right", right), ("wrong", wrong))}
    alt = (w["title"] + ". " + "; ".join(f"{html.unescape(r[0])}: {r[1]}" for r in rows_) + ". "
           + w["right"] + ": " + "; ".join(f"{s}: " + ", ".join(f"{f} ({lemma[f]})" for f in fs) for s, fs in right)
           + ". " + w["wrong"] + ": " + "; ".join(f"{s}: " + ", ".join(f"{f} ({lemma[f]})" for f in fs) for s, fs in wrong)
           + ". " + w["note"].replace("*", ""))
    body = (_table(rows_, w["head"], numbers=(1,)) + (_label(w["right"]) + lines["right"] if right else "")
            + (_label(w["wrong"]) + lines["wrong"] if wrong else "") + _note(w["note"]))
    return yours.figure(w["title"], body, alt=alt)


# --- the picture under exercise 2: your measure in Porter's rules --------------------------------------------------------

EMENT_SHOWN = ["replacement", "management", "statement", "agreement", "settlement", "cement"]
_MEASURE_WORDS = {
    "en": {"title": "Your measure in Porter's stemmer",
           "rule": "The rule (m > 1) EMENT → on its own, with your measure:",
           "head": ["word", "what is left", "your m", "after the rule"],
           "porter": "Porter's whole stemmer with your measure, on the {n} different forms of EWT: {k} get the same "
                     "stem as with the course's measure.",
           "porter_all": "Porter's whole stemmer with your measure, on the {n} different forms of EWT: every form "
                         "gets the same stem as with the course's measure, and they make {g} groups.",
           "differ": "Forms that get another stem with your measure:",
           "note": "Blue: what is left after the rule; purple: a word the rule leaves as it is, because m of what "
                   "would be left is not above 1.",
           "stopped": "Your function stopped with an error on one of the words, so the picture stays empty.",
           "slow": "Your function took more than {s:g} seconds for all words, so the picture stays empty.",
           "odd": "Your function does not give back a whole number for every word, so the picture stays empty."},
    "nl": {"title": "Jouw measure in de stemmer van Porter",
           "rule": "De regel (m > 1) EMENT → los, met jouw measure:",
           "head": ["woord", "wat overblijft", "jouw m", "na de regel"],
           "porter": "De hele stemmer van Porter met jouw measure, op de {n} verschillende vormen van EWT: {k} krijgen "
                     "dezelfde stam als met de measure van de course.",
           "porter_all": "De hele stemmer van Porter met jouw measure, op de {n} verschillende vormen van EWT: elke "
                         "vorm krijgt dezelfde stam als met de measure van de course, en de vormen maken samen {g} groepen.",
           "differ": "Vormen die met jouw measure een andere stam krijgen:",
           "note": "Blauw: wat na de regel overblijft; paars: een woord dat de regel laat staan, omdat m van wat zou "
                   "overblijven niet boven 1 komt.",
           "stopped": "Je functie stopte met een fout op een van de woorden, dus de afbeelding blijft leeg.",
           "slow": "Je functie deed er langer dan {s:g} seconden over voor alle woorden, dus de afbeelding blijft leeg.",
           "odd": "Je functie geeft niet voor elk woord een geheel getal terug, dus de afbeelding blijft leeg."},
}


def _whole(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def measure_picture(seen) -> str:
    lang = seen.lang
    w = _MEASURE_WORDS[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    forms_ = sorted(form_counts("en"))

    def run():
        ms = [fn(word[:-len("ement")]) for word in EMENT_SHOWN]
        if not all(_whole(m) for m in ms):
            return ms, None
        return ms, [stemmers.porter(f, measure=fn) for f in forms_]

    got, said = _run_or_say(run, w, lambda g: all(_whole(m) for m in g[0]) and _strings(g[1]))
    if said:
        return said
    ms, mine = got
    rows_, alts = [], []
    for word, m in zip(EMENT_SHOWN, ms):
        rest = word[:-len("ement")]
        after = _chip(rest, _CHIP_BLUE) if m > 1 else _chip(word, _CHIP_PURPLE)
        rows_.append([_chip(word), _chip(rest), _num(m, lang), after])
        alts.append(f"{word}: {rest}, m = {m}, {rest if m > 1 else word}")
    course = stems("en")
    differ = [f for f, s in zip(forms_, mine) if s != course[f]]
    count = form_counts("en")
    if differ:
        line = w["porter"].format(n=_num(len(forms_), lang), k=_num(len(forms_) - len(differ), lang))
        shown = sorted(differ, key=lambda f: (-count[f], f))[:6]
        mine_of = dict(zip(forms_, mine))
        extra = _label(w["differ"]) + "<div>" + _chips([f"{f} → {mine_of[f]} ({course[f]})" for f in shown]) + "</div>"
        extra_alt = w["differ"] + " " + ", ".join(f"{f}: {mine_of[f]} ({course[f]})" for f in shown)
    else:
        line = w["porter_all"].format(n=_num(len(forms_), lang), g=_num(len(set(mine)), lang))
        extra, extra_alt = "", ""
    body = (_label(w["rule"]) + _table(rows_, w["head"], numbers=(2,)) + _note(w["note"])
            + f'<div style="{_LINE}">{html.escape(line)}</div>' + extra)
    alt = (w["title"] + ". " + w["rule"] + " " + "; ".join(alts) + ". " + w["note"] + " " + line + " " + extra_alt)
    return yours.figure(w["title"], body, alt=alt)


# --- the picture under exercise 3: your lemmatizer on the test part ---------------------------------------------------

_LEMMA_WORDS = {
    "en": {"title": "Your lemmatize on the test part",
           "head": ["treebank", "words of the test part", "in the table", "lemma right", "right, not in the table"],
           "names": {"EWT": "EWT, with TABLE", "Alpino": "Alpino, with TABLE_NL"},
           "known": "In the table, but the treebank gives another lemma here:",
           "unknown": "Not in the table, and not its own lemma:",
           "seen": "{f} → {got}, and the treebank says {want} ({n}×)",
           "note": "The tables come from the training part of each treebank; the test part holds {en} English and "
                   "{nl} Dutch words. A word that is not in the table counts as right when your function gives the "
                   "lemma the treebank gives it.",
           "stopped": "Your function stopped with an error on one of the words, so the picture stays empty.",
           "slow": "Your function took more than {s:g} seconds for all words, so the picture stays empty.",
           "odd": "Your function does not give back a string for every word, so the picture stays empty."},
    "nl": {"title": "Jouw lemmatize op het testdeel",
           "head": ["treebank", "woorden van het testdeel", "in de tabel", "lemma goed", "goed, niet in de tabel"],
           "names": {"EWT": "EWT, met TABLE", "Alpino": "Alpino, met TABLE_NL"},
           "known": "In de tabel, maar de treebank geeft hier een ander lemma:",
           "unknown": "Niet in de tabel, en niet het eigen lemma:",
           "seen": "{f} → {got}, en de treebank zegt {want} ({n}×)",
           "note": "De tabellen komen uit het trainingsdeel van elke treebank; het testdeel heeft {en} Engelse en {nl} "
                   "Nederlandse woorden. Een woord dat niet in de tabel staat, telt als goed wanneer je functie het "
                   "lemma geeft dat de treebank het woord geeft.",
           "stopped": "Je functie stopte met een fout op een van de woorden, dus de afbeelding blijft leeg.",
           "slow": "Je functie deed er langer dan {s:g} seconden over voor alle woorden, dus de afbeelding blijft leeg.",
           "odd": "Je functie geeft niet voor elk woord een string terug, dus de afbeelding blijft leeg."},
}
LEMMA_SHOWN = 2   # misses per treebank and per kind


def lemma_picture(seen) -> str:
    lang = seen.lang
    w = _LEMMA_WORDS[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    tests = {d: test_words(d) for d in ("en", "nl")}

    def run():
        out = {}
        for d in ("en", "nl"):
            t = table_kept(d)
            out[d] = [fn(f, t) for f, _ in tests[d]]
        return out

    got, said = _run_or_say(run, w, lambda g: all(_strings(g[d]) for d in ("en", "nl")))
    if said:
        return said
    rows_, alts, lines, wrong_alt = [], [], {"known": "", "unknown": ""}, {"known": [], "unknown": []}
    for d in ("en", "nl"):
        t = table_kept(d)
        pairs = tests[d]
        known = sum(1 for f, _ in pairs if f in t)
        right = sum(1 for (f, lm), g in zip(pairs, got[d]) if g == lm)
        unknown_right = sum(1 for (f, lm), g in zip(pairs, got[d]) if f not in t and g == lm)
        rows_.append([html.escape(w["names"][TREEBANK[d]]), _num(len(pairs), lang), _pct(known, len(pairs), lang),
                      _pct(right, len(pairs), lang), _num(unknown_right, lang) + " / " + _num(len(pairs) - known, lang)])
        alts.append(f"{w['names'][TREEBANK[d]]}: {len(pairs)}, {rows_[-1][2]}, {rows_[-1][3]}, {rows_[-1][4]}")
        misses = collections.Counter((f, g, lm) for (f, lm), g in zip(pairs, got[d]) if g != lm and f.isalpha() and len(f) > 2)
        ranked = sorted(misses.items(), key=lambda kv: (-kv[1], kv[0]))
        for kind in ("known", "unknown"):
            shown = [(k, n) for k, n in ranked if (k[0] in t) == (kind == "known") and _clean(k)][:LEMMA_SHOWN]
            for (f, g, lm), n in shown:
                text = w["seen"].format(f=f, got=g, want=lm, n=_num(n, lang))
                lines[kind] += _chip(text)
                wrong_alt[kind].append(text)
    note = w["note"].format(en=_num(len(tests["en"]), lang), nl=_num(len(tests["nl"]), lang))
    body = _table(rows_, w["head"], numbers=(1, 2, 3, 4))
    for kind in ("known", "unknown"):
        if lines[kind]:
            body += _label(w[kind]) + "<div>" + lines[kind] + "</div>"
    body += _note(note)
    alt = (w["title"] + ". " + "; ".join(alts) + ". " + " ".join(w[k] + " " + "; ".join(wrong_alt[k]) + "."
                                                              for k in ("known", "unknown")) + " " + note)
    return yours.figure(w["title"], body, alt=alt)


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["apply-rules", "measure", "lemmatize", "stem-or-lemma"]

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
    """A short value the way the lessons write it: strings in double quotes, also inside a list or a dictionary."""
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_value(p) for p in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_value(k)}: {_value(x)}" for k, x in v.items()) + "}"
    return repr(v)


class Named:
    """A value of the lesson that a message shows by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


class _Lazy:
    """A value of the lesson made when a check first needs it, and shown in a message by its name. `make` gives a
    value that cannot be changed, such as a table of lemmas."""

    def __init__(self, name: str, make):
        self.name, self.make = name, make

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return self.make()


def _lit(v) -> Named:
    """A value of the check's own, shown in a message as Python, with strings in double quotes."""
    return Named(_value(v), v)


STEP_1A_N = Named("STEP_1A", STEP_1A)
TABLE_N = _Lazy("TABLE", lambda: table_kept("en"))
TABLE_NL_N = _Lazy("TABLE_NL", lambda: table_kept("nl"))


def _a_list(got, text) -> bool:
    return isinstance(got, list)


def _key_error(e) -> bool:
    return isinstance(e, KeyError)


# exercise 1: apply_rules. Besides "caresses": each other rule of step 1a, a word that no rule fits with an s inside
# it, a word with an s at the start, rules of the check's own where the longer suffix must come first, Dutch rules,
# no rules at all, and an empty word.
RULES_CASES = [
    (_lit("caresses"), STEP_1A_N),
    (_lit("ponies"), STEP_1A_N),
    (_lit("caress"), STEP_1A_N),
    (_lit("cats"), STEP_1A_N),
    (_lit("dust"), STEP_1A_N),
    (_lit("sits"), STEP_1A_N),
    (_lit("relational"), _lit({"ational": "ate", "tional": "tion"})),
    (_lit("conditional"), _lit({"ational": "ate", "tional": "tion"})),
    (_lit("mogelijkheden"), _lit({"heden": "heid", "en": ""})),
    (_lit("cats"), _lit({})),
    (_lit(""), STEP_1A_N),
]
_A_START = "def apply_rules(word, rules):\n    ...\n"
_A = ("def apply_rules(word, rules):\n"
      "    for suffix in rules:\n"
      "        if word.endswith(suffix):\n"
      "            return word[:len(word) - len(suffix)] + rules[suffix]\n"
      "    return word\n")
_A_NO_NEW = _A.replace(" + rules[suffix]", "")
_A_FIRST = _A.replace("    return word\n", "        return word\n")
_A_REPLACE = _A.replace("word[:len(word) - len(suffix)] + rules[suffix]", "word.replace(suffix, rules[suffix])")
_A_IN = _A.replace("if word.endswith(suffix):", "if suffix in word:")
_A_LAST = ("def apply_rules(word, rules):\n"
           "    new = word\n"
           "    for suffix in rules:\n"
           "        if word.endswith(suffix):\n"
           "            new = word[:len(word) - len(suffix)] + rules[suffix]\n"
           "    return new\n")
_A_PAIR = _A.replace("return word[:len(word) - len(suffix)] + rules[suffix]",
                     "return [word[:len(word) - len(suffix)], rules[suffix]]")

# exercise 2: measure. Besides "trouble": the other examples of Porter's paper, a y after a consonant and after a
# vowel, the longest example of Snowball, and words without a vowel or without letters.
MEASURE_CASES = [
    (_lit("trouble"),),
    (_lit("tree"),),
    (_lit("troubles"),),
    (_lit("private"),),
    (_lit("oaten"),),
    (_lit("ivy"),),
    (_lit("toy"),),
    (_lit("syzygy"),),
    (_lit("crepuscular"),),
    (_lit("by"),),
    (_lit(""),),
]
_M_START = "def measure(word):\n    ...\n"
_M = ("def measure(word):\n"
      "    m = 0\n"
      "    for i in range(1, len(word)):\n"
      "        if vowel(word, i - 1) and not vowel(word, i):\n"
      "            m = m + 1\n"
      "    return m\n")
_M_VOWELS = ("def measure(word):\n"
             "    m = 0\n"
             "    for i in range(len(word)):\n"
             "        if vowel(word, i):\n"
             "            m = m + 1\n"
             "    return m\n")
_M_CV = _M.replace("if vowel(word, i - 1) and not vowel(word, i):", "if not vowel(word, i - 1) and vowel(word, i):")
_M_VV = _M.replace("and not vowel(word, i)", "and vowel(word, i)")
_M_AEIOU = _M.replace("if vowel(word, i - 1) and not vowel(word, i):",
                      "if word[i - 1] in \"aeiou\" and word[i] not in \"aeiou\":")
_M_RETURN_IN = _M.replace("    return m\n", "        return m\n")

# exercise 3: lemmatize. Besides "liep": English and Dutch words of the tables, a word the table does not have, a
# word that is its own lemma, a table of the check's own with a word in it and one not in it, and an empty table.
LEMMA_CASES = [
    (_lit("liep"), TABLE_NL_N),
    (_lit("ran"), TABLE_N),
    (_lit("mice"), TABLE_N),
    (_lit("kinderen"), TABLE_NL_N),
    (_lit("blorp"), TABLE_N),
    (_lit("run"), TABLE_N),
    (_lit("went"), _lit({"went": "go"})),
    (_lit("gone"), _lit({"went": "go"})),
    (_lit("fiets"), _lit({})),
]
_L_START = "def lemmatize(word, table):\n    ...\n"
_L = ("def lemmatize(word, table):\n"
      "    if word in table:\n"
      "        return table[word]\n"
      "    return word\n")
_L_NO_CHECK = "def lemmatize(word, table):\n    return table[word]\n"
_L_EMPTY = _L.replace("    return word\n", "    return \"\"\n")
_L_WORD = "def lemmatize(word, table):\n    return word\n"

EXERCISES = {
    "apply-rules": FunctionBox(
        name="apply-rules",
        picture=rules_picture,
        function="apply_rules",
        start=_A_START,
        solution=_A,
        visible=RULES_CASES[:1],
        hidden=RULES_CASES[1:],
        first=[_DOTS],
        show=_value,
        mistakes=[
            Mistake("a-list", matches=_a_list,
                    en="`{call}` gives {got}: your function gives back a list of two parts, as `cut_suffix` did. Here it should give back one string, {want}: what is left of the word, with the rule's new ending glued to it.",
                    nl="`{call}` geeft {got}: je functie geeft een lijst van twee delen terug, zoals `cut_suffix` deed. Hier moet de functie één string teruggeven, {want}: wat er van het woord overblijft, met de nieuwe uitgang van de regel eraan vast."),
            Mistake("no-new-ending", wrong=_A_NO_NEW,
                    en="`{call}` gives {got}, and it should give {want}: your function cuts the suffix off, but leaves out the rule's new ending. Glue `rules[suffix]` to what is left.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie knipt de uitgang eraf, maar laat de nieuwe uitgang van de regel weg. Plak `rules[suffix]` aan wat overblijft."),
            Mistake("last-fit", wrong=_A_LAST,
                    en="`{call}` gives {got}, and it should give {want}: your function goes on after the first rule that fits, so the last rule that fits wins, with a shorter suffix. Give back the new word as soon as a rule fits.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie gaat door na de eerste regel die past, dus wint de laatste regel die past, met een kortere uitgang. Geef het nieuwe woord terug zodra een regel past."),
            Mistake("first-rule-only", wrong=_A_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: your function stops after the first rule, also when that rule does not fit. Give back the word as it is only after the loop, when no rule fitted.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie stopt na de eerste regel, ook als die regel niet past. Geef het woord pas na de loop onveranderd terug, als geen enkele regel paste."),
            Mistake("replace", wrong=_A_REPLACE,
                    en="`{call}` gives {got}, and it should give {want}: `replace` changes every place where the suffix occurs, also the *s* at the start of *sits*. Cut off only the end, with slicing, as in the worked example.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `replace` verandert elke plek waar de uitgang staat, ook de *s* aan het begin van *sits*. Knip alleen het eind eraf, met slicing, zoals in het uitgewerkte voorbeeld."),
            Mistake("in-not-endswith", wrong=_A_IN,
                    en="`{call}` gives {got}, and it should give {want}: `suffix in word` is also `True` for an *s* in the middle of *dust*. Ask whether the word ends with the suffix, with `word.endswith(suffix)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `suffix in word` is ook `True` voor een *s* midden in *dust*. Vraag of het woord op de uitgang eindigt, met `word.endswith(suffix)`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: go over the suffixes of `rules`, and for the first one the word ends with, give back the word without it, with the rule's new ending. When none fits, give back the word.",
                      "nl": "Vervang de `...`: ga de uitgangen van `rules` langs, en geef voor de eerste uitgang waarop het woord eindigt het woord zonder die uitgang terug, met de nieuwe uitgang van de regel. Als er geen past, geef dan het woord terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function take the first suffix of `rules` that the word ends with, cut it off, and glue the rule's new ending on?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Neemt je functie de eerste uitgang van `rules` waarop het woord eindigt, knipt je functie die eraf, en plakt je functie de nieuwe uitgang van de regel eraan?"},
            "almost": {"en": "Right for \"caresses\". But `{call}` gives {got}, and it should give {want}. Does your function work for every word and every dictionary of rules: a word no rule fits, other rules, no rules at all?",
                       "nl": "Goed voor \"caresses\". Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elk woord en elke dictionary met regels: een woord waar geen regel bij past, andere regels, helemaal geen regels?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back. Does every way through it end with `return`: also when no rule fits?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt elke weg erdoorheen met `return`: ook als er geen regel past?"},
            "correct": {"en": "Your `apply_rules` is right in all {n} cases. Look at the picture: your function with `STEP_1A` on every form of the English treebank, and which forms it brings together.",
                        "nl": "Je `apply_rules` klopt in alle {n} gevallen. Kijk naar de afbeelding: je functie met `STEP_1A` op elke vorm van de Engelse treebank, en welke vormen je functie samenbrengt."},
        },
    ),
    "measure": FunctionBox(
        name="measure",
        picture=measure_picture,
        function="measure",
        start=_M_START,
        solution=_M,
        names={"vowel": vowel},
        visible=MEASURE_CASES[:1],
        hidden=MEASURE_CASES[1:],
        first=[_DOTS],
        show=_value,
        mistakes=[
            Mistake("vowels", wrong=_M_VOWELS,
                    en="`{call}` gives {got}, and it should give {want}: your function counts the vowels. m counts the places where a vowel is followed by a consonant: in *trouble*, only *ou* followed by *b*.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt de klinkers. m telt de plekken waar een medeklinker op een klinker volgt: in *trouble* alleen de *b* na *ou*."),
            Mistake("cv", wrong=_M_CV,
                    en="`{call}` gives {got}, and it should give {want}: your function counts a consonant followed by a vowel. m counts it the other way round: a vowel, then a consonant.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt een medeklinker gevolgd door een klinker. m telt het andersom: een klinker, en dan een medeklinker."),
            Mistake("vv", wrong=_M_VV,
                    en="`{call}` gives {got}, and it should give {want}: your function counts two vowels in a row, such as the *ee* of *tree*. The second letter should be a consonant: `not vowel(word, i)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt twee klinkers na elkaar, zoals de *ee* van *tree*. De tweede letter moet een medeklinker zijn: `not vowel(word, i)`."),
            Mistake("aeiou", wrong=_M_AEIOU,
                    en="`{call}` gives {got}, and it should give {want}: for Porter, a *y* after a consonant is a vowel, and *syzygy* has three of them. Ask `vowel(word, i)`, which knows that.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: voor Porter is een *y* na een medeklinker een klinker, en *syzygy* heeft er drie. Vraag het aan `vowel(word, i)`: die functie weet dat."),
            Mistake("return-in-loop", wrong=_M_RETURN_IN,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back m after the first letter it looks at. Put `return m` after the loop, at the same depth as `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft m al terug na de eerste letter die je functie bekijkt. Zet `return m` na de loop, even ver ingesprongen als `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: count the places where the letter before is a vowel and the letter itself is not, with `vowel(word, i)`, and give back that number.",
                      "nl": "Vervang de `...`: tel de plekken waar de letter ervoor een klinker is en de letter zelf niet, met `vowel(word, i)`, en geef dat aantal terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function count each place where a vowel is followed by a consonant, as `vowel` says?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Telt je functie elke plek waar een medeklinker op een klinker volgt, zoals `vowel` het zegt?"},
            "almost": {"en": "Right for \"trouble\". But `{call}` gives {got}, and it should give {want}. Does your function work for every word: a y, a short word, an empty one?",
                       "nl": "Goed voor \"trouble\". Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elk woord: een y, een kort woord, een leeg woord?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back. Does it end with `return m`, after the loop?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie met `return m`, na de loop?"},
            "correct": {"en": "Your `measure` is right in all {n} cases. Look at the picture: your function decides which words lose *-ement*, and it runs inside the course's Porter stemmer for every form of the English treebank.",
                        "nl": "Je `measure` klopt in alle {n} gevallen. Kijk naar de afbeelding: je functie beslist welke woorden *-ement* kwijtraken, en draait in de stemmer van Porter van de course voor elke vorm van de Engelse treebank."},
        },
    ),
    "lemmatize": FunctionBox(
        name="lemmatize",
        picture=lemma_picture,
        function="lemmatize",
        start=_L_START,
        solution=_L,
        visible=LEMMA_CASES[:1],
        hidden=LEMMA_CASES[1:],
        first=[_DOTS],
        show=_value,
        mistakes=[
            Mistake("no-check", error=_key_error,
                    en="With `{call}`, your function stops with an error: the word is not a key of the table, and `table[word]` only works for a key. Ask first whether the table has the word, and give back the word itself when it does not.",
                    nl="Met `{call}` stopt je functie met een fout: het woord is geen key van de tabel, en `table[word]` werkt alleen voor een key. Vraag eerst of de tabel het woord heeft, en geef anders het woord zelf terug."),
            Mistake("empty-string", wrong=_L_EMPTY,
                    en="`{call}` gives {got}, and it should give {want}: for a word that is not in the table, the word itself is the best guess: many words are their own lemma. Give back `word`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: voor een woord dat niet in de tabel staat, is het woord zelf de beste gok: van veel woorden is het woord zelf het lemma. Geef `word` terug."),
            Mistake("word-only", wrong=_L_WORD,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back the word itself, also when the table has it. Look the word up first: `table[word]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft het woord zelf terug, ook als de tabel het heeft. Zoek het woord eerst op: `table[word]`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: when the word is a key of `table`, give back its value; otherwise give back the word itself.",
                      "nl": "Vervang de `...`: als het woord een key van `table` is, geef dan de bijbehorende waarde terug; geef anders het woord zelf terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function give the table's lemma for a word in the table, and the word itself for any other word?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Geeft je functie het lemma uit de tabel voor een woord in de tabel, en het woord zelf voor elk ander woord?"},
            "almost": {"en": "Right for \"liep\". But `{call}` gives {got}, and it should give {want}. Does your function work for every word and every table: a word the table does not have, an empty table?",
                       "nl": "Goed voor \"liep\". Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elk woord en elke tabel: een woord dat de tabel niet heeft, een lege tabel?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back. Does every way through it end with `return` and a value: also for a word that is not in the table?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt elke weg erdoorheen met `return` en een waarde: ook voor een woord dat niet in de tabel staat?"},
            "correct": {"en": "Your `lemmatize` is right in all {n} cases. Look at the picture: your function on the words of the test part of both treebanks, which its tables were not made from.",
                        "nl": "Je `lemmatize` klopt in alle {n} gevallen. Kijk naar de afbeelding: je functie op de woorden van het testdeel van beide treebanks, waar de tabellen niet van gemaakt zijn."},
        },
    ),
    "stem-or-lemma": Sort(
        name="stem-or-lemma",
        groups={
            "stem": {"en": "A stemmer is enough: only the groups matter",
                     "nl": "Een stemmer is genoeg: alleen de groepen tellen"},
            "lemma": {"en": "A lemmatizer: the dictionary form is needed",
                      "nl": "Een lemmatizer: de vorm uit het woordenboek is nodig"},
            "keep": {"en": "Keep the forms: the ending carries what is needed",
                     "nl": "Houd de vormen: de uitgang bevat wat nodig is"},
        },
        items={
            "connect": {"en": "**A search box that finds pages with *connected* and *connections* when someone types *connection*.**",
                        "nl": "**Een zoekvak dat pagina's met *connected* en *connections* vindt als iemand *connection* typt.**"},
            "gemeente": {"en": "**A search in Dutch news where *gemeenten* (municipalities) also finds *gemeente*.**",
                         "nl": "**Zoeken in Nederlands nieuws, waarbij *gemeenten* ook *gemeente* vindt.**"},
            "invoice": {"en": "**A search in old emails for *invoice* that also finds *invoices* and *invoiced*.**",
                        "nl": "**Een zoekopdracht in oude e-mails naar *factuur* die ook *facturen* vindt.**"},
            "went": {"en": "**Count how often a text uses the verb *go*, with *went* and *gone* counted too.**",
                     "nl": "**Tel hoe vaak een tekst het werkwoord *gaan* gebruikt, met *ging* en *gegaan* erbij.**"},
            "dictionary": {"en": "**A dictionary app: someone taps *mice* in a text and should see the entry *mouse*.**",
                           "nl": "**Een woordenboek-app: iemand tikt in een tekst op *liep* en hoort dan *lopen* te zien.**"},
            "cloud": {"en": "**A word cloud of a book that shows the most common words as a reader knows them: *pony*, not *poni*.**",
                      "nl": "**Een woordwolk van een boek die de meest voorkomende woorden toont in de vorm die een lezer kent: *lopen*, niet *lop*.**"},
            "grammar": {"en": "**A grammar checker that marks *Tom walk home* and wants *Tom walks home*.**",
                        "nl": "**Een grammaticacontrole die *een grote huis* aanstreept en *een groot huis* wil.**"},
            "translate": {"en": "**Translate *books* into Dutch: *boeken*, not *boek*.**",
                          "nl": "**Vertaal *boeken* naar het Engels: *books*, niet *book*.**"},
            "past": {"en": "**Tell whether a review is about the past: *worked* or *works*.**",
                     "nl": "**Zeg of een recensie over het verleden gaat: *werkte* of *werkt*.**"},
        },
        solution={"connect": "stem", "gemeente": "stem", "invoice": "stem", "went": "lemma", "dictionary": "lemma",
                  "cloud": "lemma", "grammar": "keep", "translate": "keep", "past": "keep"},
        nudge={
            "connect": {"en": "Does the person searching care what the shared form looks like, as long as these words come out together?",
                        "nl": "Kan het degene die zoekt schelen hoe de gedeelde vorm eruitziet, zolang deze woorden samen gevonden worden?"},
            "gemeente": {"en": "Do *gemeenten* and *gemeente* differ only at the end?",
                         "nl": "Verschillen *gemeenten* en *gemeente* alleen aan het eind?"},
            "invoice": {"en": "Do the forms share their first letters, so that chopping the ending brings them together?",
                        "nl": "Beginnen de vormen met dezelfde letters, zodat de uitgang eraf hakken de vormen samenbrengt?"},
            "went": {"en": "Does *went* share any ending with *go* that a rule could chop off?",
                     "nl": "Heeft *ging* een uitgang met *gaan* gemeen die een regel eraf kan hakken?"},
            "dictionary": {"en": "What does the person who taps the word expect to see on the screen?",
                           "nl": "Wat verwacht degene die op het woord tikt op het scherm te zien?"},
            "cloud": {"en": "Who reads the result here, and would that reader know *poni*?",
                      "nl": "Wie leest de uitkomst hier, en kent die lezer *lop*?"},
            "grammar": {"en": "What tells the right sentence from the wrong one, and what would be left of it after stemming?",
                        "nl": "Wat onderscheidt de goede zin van de foute, en wat blijft daarvan over na stemming?"},
            "translate": {"en": "Where in *books* is the plural, and does that survive a trip to *book*?",
                          "nl": "Waar in *boeken* zit het meervoud, en blijft dat bewaard als je naar *boek* gaat?"},
            "past": {"en": "Which letters show the past, and what happens to them in a stemmer?",
                     "nl": "Welke letters laten het verleden zien, en wat gebeurt ermee in een stemmer?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the cases again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de gevallen opnieuw uit."},
            "empty": {"en": "Put each bold task under what it needs.",
                      "nl": "Zet elke vetgedrukte taak bij wat de taak nodig heeft."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item}: this task goes somewhere else.",
                      "nl": "{item}: deze taak hoort ergens anders."},
            "correct": {"en": "All {n} are right. A stemmer is enough where only the groups count and the forms share their start; a lemmatizer is needed where a person reads the result, or where the forms look nothing alike; and where the ending tells the past, the plural or who does it, both throw away what the task needs.",
                        "nl": "Alle {n} kloppen. Een stemmer is genoeg waar alleen de groepen tellen en de vormen hetzelfde begin hebben; een lemmatizer is nodig waar een mens de uitkomst leest, of waar de vormen helemaal niet op elkaar lijken; en waar de uitgang de tijd, het meervoud of de persoon aangeeft, gooien allebei weg wat de taak nodig heeft."},
        },
    ),
}

CASES = [
    ("apply-rules", _A_START, "empty", None),
    ("apply-rules", _A, "correct", None),
    ("apply-rules", "def apply_rules(word, rules):\n    for suffix, new in rules.items():\n        if word[len(word) - len(suffix):] == suffix:\n            return word[:len(word) - len(suffix)] + new\n    return word\n", "correct", None),
    ("apply-rules", "def apply_rules(word, rules):\n    for suffix in rules:\n        if word.endswith(suffix):\n            return word.removesuffix(suffix) + rules[suffix]\n    return word\n", "correct", None),
    ("apply-rules", _A_PAIR, "wrong", "a-list"),
    ("apply-rules", _A_NO_NEW, "wrong", "no-new-ending"),
    ("apply-rules", _A_LAST, "wrong", "last-fit"),
    ("apply-rules", _A_FIRST, "wrong", "first-rule-only"),
    ("apply-rules", _A_REPLACE, "wrong", "replace"),
    ("apply-rules", _A_IN, "wrong", "in-not-endswith"),
    ("apply-rules", _A.replace("    return word\n", ""), "wrong", "unreached"),
    ("apply-rules", "def apply_rules(word, rules):\n    return \"caress\"\n", "almost", None),
    ("measure", _M_START, "empty", None),
    ("measure", _M, "correct", None),
    ("measure", "def measure(word):\n    m = 0\n    for i in range(len(word) - 1):\n        if vowel(word, i) and not vowel(word, i + 1):\n            m += 1\n    return m\n", "correct", None),
    ("measure", "def measure(word):\n    pairs = [i for i in range(1, len(word)) if vowel(word, i - 1) and not vowel(word, i)]\n    return len(pairs)\n", "correct", None),
    ("measure", _M_VOWELS, "wrong", "vowels"),
    ("measure", _M_CV, "wrong", "cv"),
    ("measure", _M_VV, "wrong", "vv"),
    ("measure", _M_AEIOU, "wrong", "aeiou"),
    ("measure", _M_RETURN_IN, "wrong", "return-in-loop"),
    ("measure", "def measure(word):\n    return 1\n", "almost", None),
    ("lemmatize", _L_START, "empty", None),
    ("lemmatize", _L, "correct", None),
    ("lemmatize", "def lemmatize(word, table):\n    if word not in table:\n        return word\n    return table[word]\n", "correct", None),
    ("lemmatize", "def lemmatize(word, table):\n    return table.get(word, word)\n", "correct", None),
    ("lemmatize", _L_NO_CHECK, "error", "no-check"),
    ("lemmatize", _L_EMPTY, "wrong", "empty-string"),
    ("lemmatize", _L_WORD, "wrong", "word-only"),
    ("lemmatize", "def lemmatize(word, table):\n    return table.get(word)\n", "wrong", "unreached"),
    ("lemmatize", "def lemmatize(word, table):\n    return \"lopen\"\n", "almost", None),
    ("stem-or-lemma", {}, "empty", None),
    ("stem-or-lemma", {"connect": "stem", "gemeente": "stem", "invoice": "stem", "went": "lemma", "dictionary": "lemma",
                       "cloud": "lemma", "grammar": "keep", "translate": "keep", "past": "keep"}, "correct", None),
    ("stem-or-lemma", {"connect": "stem", "went": "lemma"}, "wrong", "unfinished"),
    ("stem-or-lemma", {"went": "stem"}, "wrong", "wrong:went"),
    ("stem-or-lemma", {"connect": "stem", "gemeente": "stem", "invoice": "stem", "went": "lemma", "dictionary": "lemma",
                       "cloud": "stem", "grammar": "keep", "translate": "keep", "past": "keep"}, "wrong", "wrong:cloud"),
    ("stem-or-lemma", {"connect": "stem", "gemeente": "stem", "invoice": "stem", "went": "lemma", "dictionary": "lemma",
                       "cloud": "lemma", "grammar": "keep", "translate": "lemma", "past": "keep"}, "wrong", "wrong:translate"),
]
