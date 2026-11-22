# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-languages-of-the-world: read the order of subject, verb and object of a clause from its
dependency tree (order), count those orders over a treebank (count_orders), decide with the rule of the World Atlas of
Language Structures whether one order is dominant (dominant), and sort statements under language families or types.

The course's sentences are its own, in Dutch in both languages of the lesson, with relations the course gave them by
the guidelines of Universal Dependencies. The treebanks are UD_English-EWT (without the emails of Enron) and
UD_Dutch-Alpino of Universal Dependencies 2.19 (data/ud-en-ewt.conllu.gz and data/ud-nl-alpino.conllu.gz, made by
data/tools/make_ud.py). The counts of languages and the map come from WALS Online v2020.6 (Dryer and Haspelmath, eds.,
2013, CC BY 4.0), in data/wals-word-order.tsv.gz (made by data/tools/make_wals_word_order.py). Every number in the
pictures is counted by the course; the cases of the checks and the statements of the sort are the course's own."""
import contextlib
import copy
import csv
import gzip
import html
import io
import math

from everything_ai import chart, paths, yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, kept, with_time_limit

# --- the course's sentences ------------------------------------------------------------------------------------------

# word i of a sentence is at place i - 1: (form, head, relation); head 0 is the root
SENTENCE = [("gisteren", 2, "advmod"), ("zag", 0, "root"), ("de", 4, "det"), ("kat", 2, "nsubj"), ("de", 6, "det"),
            ("vogel", 2, "obj")]                                                     # yesterday saw the cat the bird
CLAUSE = [("omdat", 6, "mark"), ("de", 3, "det"), ("kat", 6, "nsubj"), ("de", 5, "det"), ("vogel", 6, "obj"),
          ("ziet", 0, "root")]                                                       # because the cat the bird sees
# sentences of the check's own
MAIN = [("de", 2, "det"), ("kat", 3, "nsubj"), ("zag", 0, "root"), ("de", 5, "det"), ("vogel", 3, "obj")]
SLEEPS = [("de", 2, "det"), ("kat", 3, "nsubj"), ("slaapt", 0, "root")]
RELATIVE = [("de", 2, "det"), ("vogel", 0, "root"), ("die", 6, "obj"), ("de", 5, "det"), ("kat", 6, "nsubj"),
            ("ziet", 2, "acl:relcl")]                                                # the bird that the cat sees
SAW_THAT = [("de", 2, "det"), ("kat", 3, "nsubj"), ("zag", 0, "root"), ("dat", 9, "mark"), ("de", 6, "det"),
            ("hond", 9, "nsubj"), ("de", 8, "det"), ("vogel", 9, "obj"), ("ving", 3, "ccomp")]
DOOR = [("Close", 0, "root"), ("the", 3, "det"), ("door", 1, "obj")]
HELLO = [("hallo", 0, "root")]

ORDERS = ["SOV", "SVO", "VSO", "VOS", "OVS", "OSV"]   # in the order WALS lists them


def order(sentence, h):
    """The course's solution of Exercise 1: the letters S, V and O of verb h, its subject and its object, in the order
    of the words."""
    letters = ""
    for i in range(1, len(sentence) + 1):
        form, head, relation = sentence[i - 1]
        if i == h:
            letters = letters + "V"
        if head == h and relation == "nsubj":
            letters = letters + "S"
        if head == h and relation == "obj":
            letters = letters + "O"
    return letters


def count_orders(sentences):
    """The course's solution of Exercise 2: how often each order of three letters occurs, over every word of every
    sentence."""
    counts = {}
    for sentence in sentences:
        for h in range(1, len(sentence) + 1):
            letters = order(sentence, h)
            if len(letters) == 3:
                if letters in counts:
                    counts[letters] = counts[letters] + 1
                else:
                    counts[letters] = 1
    return counts


def dominant(counts):
    """The course's solution of Exercise 3: the most common order when it is more than twice as common as the next,
    and else the empty string, by the rule of thumb of WALS."""
    ranked = sorted(counts.values(), reverse=True)
    if len(ranked) > 1 and ranked[0] <= 2 * ranked[1]:
        return ""
    for letters in counts:
        if counts[letters] == ranked[0]:
            return letters


# --- the treebanks ---------------------------------------------------------------------------------------------------

UD = {"en": paths.ROOT / "data" / "ud-en-ewt.conllu.gz", "nl": paths.ROOT / "data" / "ud-nl-alpino.conllu.gz"}
TREEBANK_NAMES = {"en": "UD_English-EWT", "nl": "UD_Dutch-Alpino"}


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


def _full(lang: str) -> tuple:
    """The sentences of a treebank with their tags, read once per process."""
    return kept(f"nlp-languages-of-the-world:{lang}", lambda: _read(UD[lang]))


def _triples(lang: str) -> tuple:
    return tuple(tuple((f, h, r) for f, _, h, r in words) for _, words in _full(lang))


def treebank(lang: str) -> list:
    """The sentences of a treebank as the exercises take them: a new list of sentences, each a list of words (form,
    head, relation)."""
    return [list(s) for s in kept(f"nlp-languages-of-the-world:triples:{lang}", lambda: _triples(lang))]


def clause_kind(words, h) -> str:
    """What kind of clause verb h heads, from the relations of UD: a relative clause (acl:relcl), a clause after a
    conjunction such as omdat or because (mark), a clause with an auxiliary (aux), or none of those."""
    relations = [w[3] for w in words if w[2] == h]
    if words[h - 1][3].startswith("acl:relcl"):
        return "relative"
    if "mark" in relations:
        return "conjunction"
    if any(r.startswith("aux") for r in relations):
        return "auxiliary"
    return "plain"


def _kinds(lang: str) -> dict:
    """{kind: {order: count}} for the clauses of a treebank with a subject, a verb and an object; "all" holds them all."""
    out = {k: {} for k in ("all", "plain", "auxiliary", "conjunction", "relative")}
    for _, words in _full(lang):
        triples = [(f, h, r) for f, _, h, r in words]
        for h in range(1, len(words) + 1):
            letters = order(triples, h)
            if len(letters) == 3:
                for k in ("all", clause_kind(words, h)):
                    out[k][letters] = out[k].get(letters, 0) + 1
    return out


def kinds(lang: str) -> dict:
    """The counts of _kinds, made once per process."""
    return kept(f"nlp-languages-of-the-world:kinds:{lang}", lambda: _kinds(lang))


# --- WALS ------------------------------------------------------------------------------------------------------------

WALS_FILE = paths.ROOT / "data" / "wals-word-order.tsv.gz"


def _read_wals() -> tuple:
    with gzip.open(WALS_FILE, "rt", encoding="utf-8", newline="") as f:
        return tuple(csv.DictReader(f, delimiter="\t"))


def wals() -> tuple:
    """The languages of data/wals-word-order.tsv.gz, each a dict with the columns of the file, read once per process."""
    return kept("nlp-languages-of-the-world:wals", _read_wals)


# for each feature: its column, its values in the order of WALS with a short label for a bar and the group of their
# dots on the map (None: open dots, for no dominant order)
FEATURES = {
    "81A": {"column": "order",
            "values": [("SOV", "SOV", "SOV", 0), ("SVO", "SVO", "SVO", 1), ("VSO", "VSO", "VSO", 2),
                       ("VOS", "VOS", "VOS", 2), ("OVS", "OVS", "OVS", 3), ("OSV", "OSV", "OSV", 3),
                       ("No dominant order", "none", "geen", None)]},
    "83A": {"column": "object_verb",
            "values": [("OV", "OV", "OV", 0), ("VO", "VO", "VO", 1), ("No dominant order", "none", "geen", None)]},
    "85A": {"column": "adposition",
            "values": [("Postpositions", "post", "achter", 0), ("Prepositions", "pre", "voor", 1),
                       ("Inpositions", "other", "anders", 2), ("No adpositions", "other", "anders", 2),
                       ("No dominant order", "none", "geen", None)]},
}
FEATURE_NAMES = {
    "en": {"81A": "Subject, object and verb (81A)", "83A": "Object and verb (83A)",
           "85A": "Adposition and noun phrase (85A)"},
    "nl": {"81A": "Onderwerp, lijdend voorwerp en werkwoord (81A)", "83A": "Lijdend voorwerp en werkwoord (83A)",
           "85A": "Voorzetsel of achterzetsel (85A)"},
}
FAMILIES = ["", "Niger-Congo", "Austronesian", "Sino-Tibetan", "Afro-Asiatic", "Indo-European", "Trans-New Guinea",
            "Pama-Nyungan"]
FAMILY_NAMES = {
    "en": {"": "All languages", "Niger-Congo": "Niger-Congo", "Austronesian": "Austronesian",
           "Sino-Tibetan": "Sino-Tibetan", "Afro-Asiatic": "Afro-Asiatic", "Indo-European": "Indo-European",
           "Trans-New Guinea": "Trans-New Guinea", "Pama-Nyungan": "Pama-Nyungan"},
    "nl": {"": "Alle talen", "Niger-Congo": "Niger-Congo", "Austronesian": "Austronesisch",
           "Sino-Tibetan": "Sino-Tibetaans", "Afro-Asiatic": "Afro-Aziatisch", "Indo-European": "Indo-Europees",
           "Trans-New Guinea": "Trans-Nieuw-Guinea", "Pama-Nyungan": "Pama-Nyunga"},
}
# languages a learner may know, named first among the examples, with their Dutch names
KNOWN = {
    "English": "Engels", "Mandarin": "Mandarijn", "Japanese": "Japans", "Turkish": "Turks", "Hindi": "Hindi",
    "Indonesian": "Indonesisch", "Swahili": "Swahili", "Welsh": "Welsh", "Irish": "Iers",
    "Arabic (Modern Standard)": "Modern Standaardarabisch", "Malagasy": "Malagassisch", "Hixkaryana": "Hixkaryana",
    "Nadëb": "Nadëb", "Basque": "Baskisch", "Dutch": "Nederlands", "German": "Duits", "Frisian": "Fries",
    "Korean": "Koreaans", "Persian": "Perzisch", "Russian": "Russisch", "French": "Frans", "Spanish": "Spaans",
    "Hungarian": "Hongaars", "Tagalog": "Tagalog", "Vietnamese": "Vietnamees", "Thai": "Thai", "Hawaiian": "Hawaïaans",
    "Maori": "Maori", "Yoruba": "Yoruba", "Zulu": "Zoeloe", "Hausa": "Hausa", "Amharic": "Amhaars",
    "Hebrew (Modern)": "Hebreeuws (modern)", "Finnish": "Fins", "Greek (Modern)": "Grieks (modern)",
    "Italian": "Italiaans", "Portuguese": "Portugees", "Polish": "Pools", "Tamil": "Tamil", "Burmese": "Birmaans",
    "Cantonese": "Kantonees", "Somali": "Somalisch", "Fijian": "Fijisch", "Samoan": "Samoaans", "Tongan": "Tongaans",
    "Nias": "Nias", "Tuvaluan": "Tuvaluaans", "Danish": "Deens", "Swedish": "Zweeds", "Norwegian": "Noors",
    "Icelandic": "IJslands", "Urdu": "Urdu", "Georgian": "Georgisch", "Navajo": "Navajo", "Warlpiri": "Warlpiri",
    "Igbo": "Igbo", "Wolof": "Wolof", "Ewe": "Ewe", "Maasai": "Maasai", "Chamorro": "Chamorro", "Kiribati": "Kiribatisch",
}
_KNOWN_ORDER = list(KNOWN)


def language_name(name: str, lang: str) -> str:
    return KNOWN.get(name, name) if lang == "nl" else name


def feature_names(lang: str) -> dict:
    """The options of the demo's first dropdown: {label: feature}."""
    return {FEATURE_NAMES[lang][k]: k for k in FEATURES}


def family_names(lang: str) -> dict:
    """The options of the demo's second dropdown: {label: family}, "" for all languages."""
    return {FAMILY_NAMES[lang][k]: k for k in FAMILIES}


def counts_of(feature: str, family: str = "") -> dict:
    """{value: number of languages} for a feature of WALS, over all languages or one family."""
    column = FEATURES[feature]["column"]
    out = {}
    for row in wals():
        if row[column] and (not family or row["family"] == family):
            out[row[column]] = out.get(row[column], 0) + 1
    return out


# --- small helpers for the pictures ----------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums"
_CODE = ";font-family:var(--ea-font-code);font-size:0.78rem"
_OFF = ";color:var(--ea-loss-text);font-weight:600"


def _note(text: str) -> str:
    """A line under a picture; a word between asterisks is set in italics, as in the lesson's text."""
    parts = html.escape(text).split("*")
    marked = "".join(f"<i>{p}</i>" if k % 2 else p for k, p in enumerate(parts))
    return f'<div style="{_UI}">{marked}</div>'


def _bare(text: str) -> str:
    return text.replace("*", "")


def _num(n, lang: str) -> str:
    """A whole number with a separator for thousands: 2,319 in English, 2.319 in Dutch."""
    text = f"{round(n):,}"
    return text.replace(",", ".") if lang == "nl" else text


def _dec(x: float, lang: str, digits: int = 1) -> str:
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def _pct(x: float, lang: str) -> str:
    return _dec(100 * x, lang, 1) + "%"


def _table(rows: list, head: list, numbers: tuple = (), styles: dict | None = None) -> str:
    """A small table; rows hold HTML that is already escaped; numbers are the columns set to the right; styles maps
    (row, column) to extra style."""
    styles = styles or {}
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{_NUMBER if j in numbers else ""}{styles.get((i, j), "")}">'
                                    f'{c}</td>' for j, c in enumerate(row)) + "</tr>" for i, row in enumerate(rows))
    return (f'<table style="border-collapse:collapse;margin:0.3rem 0 0.2rem"><thead><tr>{top}</tr></thead>'
            f"<tbody>{body}</tbody></table>")


def _svg(panels: list, alt: str, lang: str, width: int = 330, height: int = 250) -> str:
    svg = chart.figure(panels, alt=alt, width=width, height=height)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


def _verdict(v, lang: str) -> str:
    if v == "":
        return {"en": "no dominant order", "nl": "geen dominante volgorde"}[lang]
    return str(v)


# --- the idea's picture: the orders of WALS --------------------------------------------------------------------------

_IDEA_WORDS = {
    "en": {"title": "The dominant order in 1,376 languages",
           "panel": "WALS, feature 81A", "y": "languages",
           "alt": "Bars for the number of languages in WALS with each dominant order: {pairs}.",
           "legend": "Examples: SOV {SOV}; SVO {SVO}; VSO {VSO}; VOS {VOS}; OVS {OVS}; OSV {OSV}; none: no dominant "
                     "order, {none}. Counted by the course in WALS Online v2020.6."},
    "nl": {"title": "De dominante volgorde in 1.376 talen",
           "panel": "WALS, eigenschap 81A", "y": "talen",
           "alt": "Staven voor het aantal talen in WALS met elke dominante volgorde: {pairs}.",
           "legend": "Voorbeelden: SOV {SOV}; SVO {SVO}; VSO {VSO}; VOS {VOS}; OVS {OVS}; OSV {OSV}; geen: geen "
                     "dominante volgorde, {none}. Geteld door de course in WALS Online v2020.6."},
}
_IDEA_EXAMPLES = {"SOV": ["Japanese", "Turkish", "Hindi", "Basque"], "SVO": ["English", "Mandarin", "Swahili"],
                  "VSO": ["Welsh", "Irish", "Arabic (Modern Standard)"], "VOS": ["Malagasy"], "OVS": ["Hixkaryana"],
                  "OSV": ["Nadëb"], "none": ["Dutch", "German", "Frisian"]}


def idea_picture(lang: str) -> str:
    w = _IDEA_WORDS[lang]
    counts = counts_of("81A")
    values = FEATURES["81A"]["values"]
    labels = [v[1] if lang == "en" else v[2] for v in values]
    ns = [counts.get(v[0], 0) for v in values]
    p = chart.Panel((0.4, len(values) + 0.6), (0, max(ns) * 1.15), title=w["panel"], ylabel=w["y"],
                    xticks=list(range(1, len(values) + 1)), xtick_labels=labels)
    for i, (v, n) in enumerate(zip(values, ns), start=1):
        role = "ink-3" if v[3] is None else f"group-{v[3] + 1}"
        p.rect(i - 0.32, 0, i + 0.32, n, role=role, solid=v[3] is not None)
        p.text(i, n + max(ns) * 0.05, _num(n, lang), role="ink-2")
    names = {k: ", ".join(language_name(x, lang) for x in xs) for k, xs in _IDEA_EXAMPLES.items()}
    legend = w["legend"].format(**names)
    pairs = "; ".join(f"{label} {_num(n, lang)}" for label, n in zip(labels, ns))
    alt = f'{w["title"]}. ' + w["alt"].format(pairs=pairs) + " " + legend
    return yours.figure(w["title"], _svg([p], alt, lang, width=520, height=260) + _note(legend), alt=alt)


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


# --- the picture under exercise 1: seven clauses of the treebanks ----------------------------------------------------

# (treebank, sent_id, the verb): five orders of one Dutch sentence, from Alpino, and two English
# clauses from reviews and answers in EWT
EXAMPLES = [("nl", "g_suite\\54", 2), ("nl", "g_suite\\64", 2), ("nl", "g_suite\\55", 5), ("nl", "g_suite\\60", 5),
            ("nl", "g_suite\\59", 3), ("en", "reviews-043020-0007", 2), ("en", "answers-20111108071852AAxbh5F_ans-0003", 4)]
_GLOSSES = {"g_suite\\54": "I miss the train", "g_suite\\64": "today I miss the train",
            "g_suite\\55": "I have missed the train", "g_suite\\60": "the train I have missed",
            "g_suite\\59": "the train I miss", "reviews-043020-0007": "we hoorden niets",
            "answers-20111108071852AAxbh5F_ans-0003": "wat doe ik?"}


def _sentence(lang: str, sent_id: str) -> tuple:
    for sid, words in _full(lang):
        if sid == sent_id:
            return words
    raise KeyError(sent_id)


def _marked(words, h) -> str:
    """The words of a sentence, with the verb h, its subject and its object in bold and their letter after them."""
    out = []
    for i, (form, _, head, rel) in enumerate(words, start=1):
        letter = "V" if i == h else ("S" if head == h and rel == "nsubj" else ("O" if head == h and rel == "obj" else ""))
        if letter:
            out.append(f'<b>{html.escape(form)}</b><sub style="color:var(--ea-ink-3);font-size:0.68rem">{letter}</sub>')
        else:
            out.append(html.escape(form))
    return " ".join(out)


_ORDER_WORDS = {
    "en": {"title": "Your order, at work: seven clauses of the treebanks",
           "head": ["Clause", "Treebank", "Your order"],
           "fine": "Your function reads all seven orders right. The five Dutch clauses are one sentence in five orders, "
                   "from Alpino: Dutch can put the subject, the object or another word in front, and puts the verb "
                   "second; with an auxiliary such as *heb* (have), the main verb goes to the end.",
           "differs": "In {n} of the seven clauses, your order differs from the marks under the words; those are "
                      "purple.",
           "error": "Your function stops with an error on {n} of the seven clauses.",
           "slow": "Your function takes too long on a clause of a few words."},
    "nl": {"title": "Jouw order aan het werk: zeven zinnen uit de treebanks",
           "head": ["Zin", "Treebank", "Jouw volgorde"],
           "fine": "Je functie leest alle zeven volgordes goed. De vijf Nederlandse zinnen zijn één zin in vijf "
                   "volgordes, uit Alpino: het Nederlands kan het onderwerp, het lijdend voorwerp of een ander woord "
                   "vooraan zetten, en zet het werkwoord op de tweede plaats; met een hulpwerkwoord zoals *heb* gaat "
                   "het hoofdwerkwoord naar het eind.",
           "differs": "In {n} van de zeven zinnen wijkt je volgorde af van de letters bij de woorden; die zijn paars.",
           "error": "Je functie stopt met een fout op {n} van de zeven zinnen.",
           "slow": "Je functie doet er te lang over op een zin van een paar woorden."},
}


def order_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _ORDER_WORDS[lang]
    rows, styles, off, errors = [], {}, 0, 0
    for i, (tb, sent_id, h) in enumerate(EXAMPLES):
        words = _sentence(tb, sent_id)
        triples = [list((f, hh, r)) for f, _, hh, r in words]
        triples = [tuple(t) for t in triples]
        try:
            got = _call(fn, triples, h)
        except Timeout:
            return yours.figure(w["title"], _note(w["slow"]), alt=f'{w["title"]}. {w["slow"]}')
        want = order(triples, h)
        gloss = "" if tb == lang else f' <span style="color:var(--ea-ink-3)">({html.escape(_GLOSSES[sent_id])})</span>'
        if got is _STOPPED:
            errors += 1
            shown = "⚠"
        else:
            shown = html.escape('"' + got + '"' if isinstance(got, str) else repr(got))
        if got is _STOPPED or got != want:
            off += 1
            styles[(i, 2)] = _CODE + _OFF
        else:
            styles[(i, 2)] = _CODE
        rows.append([_marked(words, h) + gloss, html.escape(TREEBANK_NAMES[tb]), shown])
    if errors:
        text = w["error"].format(n=errors)
    elif off:
        text = w["differs"].format(n=off)
    else:
        text = w["fine"]
    alt = f'{w["title"]}. ' + " ".join(f"{_bare(r[0])}" for r in rows[:0]) + _bare(text)
    return yours.figure(w["title"], _table(rows, w["head"], styles=styles) + _note(text), alt=alt)


# --- the picture under exercise 2: the orders of two treebanks -------------------------------------------------------

_COUNT_WORDS = {
    "en": {"title": "Your count_orders, at work: every clause of two treebanks",
           "y": "clauses",
           "fine": "Your function finds {n_en} clauses with a subject, a verb and an object in the English treebank, "
                   "and {n_nl} in the Dutch one. In English, {top_en} is {r_en} times as common as the next order; in "
                   "Dutch, {top_nl} is only {r_nl} times as common as {next_nl}.",
           "differs": "Your counts for {name} differ from the right ones in {n} of the six orders; those bars are "
                      "purple.",
           "short": "Your counts for {name} also hold orders with fewer than three letters, such as {example}: the "
                    "bars show only the six orders of three letters.",
           "error": "Your function stops with an error on the sentences of {name}.",
           "slow": "Your function takes too long on the {n} sentences of {name}.",
           "not_dict": "Your function does not give back a dict for {name}, so there is nothing to draw yet."},
    "nl": {"title": "Jouw count_orders aan het werk: elke zin van twee treebanks",
           "y": "zinnen",
           "fine": "Je functie vindt {n_en} zinnen met een onderwerp, een werkwoord en een lijdend voorwerp in de "
                   "Engelse treebank, en {n_nl} in de Nederlandse. In het Engels komt {top_en} {r_en} keer zo vaak "
                   "voor als de volgorde daarna; in het Nederlands komt {top_nl} maar {r_nl} keer zo vaak voor als "
                   "{next_nl}.",
           "differs": "Je aantallen voor {name} wijken in {n} van de zes volgordes af van de goede; die staven zijn "
                      "paars.",
           "short": "Je aantallen voor {name} bevatten ook volgordes met minder dan drie letters, zoals {example}: de "
                    "staven tonen alleen de zes volgordes van drie letters.",
           "error": "Je functie stopt met een fout op de zinnen van {name}.",
           "slow": "Je functie doet er te lang over op de {n} zinnen van {name}.",
           "not_dict": "Je functie geeft voor {name} geen dictionary terug, dus er valt nog niets te tekenen."},
}
_SENTENCE_NAMES = {"en": {"en": "the English treebank", "nl": "the Dutch treebank"},
                   "nl": {"en": "de Engelse treebank", "nl": "de Nederlandse treebank"}}


def _ranked(counts: dict) -> list:
    """(order, count) from the most to the least common; equal counts in the order of ORDERS."""
    return sorted(counts.items(), key=lambda kv: (-kv[1], ORDERS.index(kv[0]) if kv[0] in ORDERS else 99))


def counts_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _COUNT_WORDS[lang]
    panels, notes, fine = [], [], {}
    for tb in ("en", "nl"):
        name = _SENTENCE_NAMES[lang][tb]
        sentences = treebank(tb)
        try:
            got = _call(fn, sentences)
        except Timeout:
            notes.append(w["slow"].format(n=_num(len(sentences), lang), name=name))
            continue
        if got is _STOPPED:
            notes.append(w["error"].format(name=name))
            continue
        if not isinstance(got, dict):
            notes.append(w["not_dict"].format(name=name))
            continue
        want = kinds(tb)["all"]
        mine = [got.get(o, 0) if isinstance(got.get(o, 0), (int, float)) else 0 for o in ORDERS]
        right = [want.get(o, 0) for o in ORDERS]
        top = max(max(mine), max(right), 1)
        p = chart.Panel((0.4, 6.6), (0, top * 1.18), title=TREEBANK_NAMES[tb], ylabel=w["y"],
                        xticks=list(range(1, 7)), xtick_labels=ORDERS)
        for i, (m, r) in enumerate(zip(mine, right), start=1):
            p.rect(i - 0.32, 0, i + 0.32, m, role="data" if m == r else "loss", solid=m == r)
            p.text(i, m + top * 0.06, _num(m, lang), role="ink-2")
        panels.append(p)
        off = sum(m != r for m, r in zip(mine, right))
        shorter = [k for k in got if isinstance(k, str) and len(k) < 3]
        if off:
            notes.append(w["differs"].format(name=name, n=off))
        if shorter:
            notes.append(w["short"].format(name=name, example='"' + shorter[0] + '"'))
        if not off and not shorter:
            fine[tb] = _ranked(want)
    if len(fine) == 2:
        (top_en, a_en), (_, b_en) = fine["en"][:2]
        (top_nl, a_nl), (next_nl, b_nl) = fine["nl"][:2]
        notes.append(w["fine"].format(n_en=_num(sum(kinds("en")["all"].values()), lang),
                                      n_nl=_num(sum(kinds("nl")["all"].values()), lang), top_en=top_en,
                                      r_en=_dec(a_en / b_en, lang), top_nl=top_nl, r_nl=_dec(a_nl / b_nl, lang),
                                      next_nl=next_nl))
    text = " ".join(notes)
    alt = f'{w["title"]}. ' + _bare(text)
    if not panels:
        return yours.figure(w["title"], _note(text), alt=alt)
    return yours.figure(w["title"], _svg(panels, alt, lang) + _note(text), alt=alt)


# --- the picture under exercise 3: the rule of WALS, clause by clause ------------------------------------------------

KIND_ORDER = ("all", "auxiliary", "conjunction", "relative", "plain")
_DOMINANT_WORDS = {
    "en": {"title": "Your dominant, at work: kinds of clauses in two treebanks",
           "head": ["Clauses", "Number", "Most common", "Next", "Your verdict"],
           "kinds": {"all": "every clause", "plain": "the other clauses", "auxiliary": "with an auxiliary",
                     "conjunction": "after a subordinating conjunction", "relative": "relative clause"},
           "fine": "Your function follows the rule in all ten rows. WALS gives English the order SVO and Dutch no "
                   "dominant order, with two: SOV or SVO. The rows show where each comes from: SVO in the other Dutch "
                   "clauses, SOV after a subordinating conjunction and with an auxiliary. Below *every clause*, each "
                   "clause counts once: as a relative clause if it is one, else after a subordinating conjunction, "
                   "else with an auxiliary, else under the other clauses.",
           "differs": "In {n} of the ten rows, your verdict differs from the rule; those are purple.",
           "error": "Your function stops with an error on {n} of the ten rows.",
           "slow": "Your function takes too long on a dict of a few counts."},
    "nl": {"title": "Jouw dominant aan het werk: soorten zinnen in twee treebanks",
           "head": ["Zinnen", "Aantal", "Vaakst", "Daarna", "Jouw oordeel"],
           "kinds": {"all": "elke zin", "plain": "de overige zinnen",
                     "auxiliary": "met een hulpwerkwoord", "conjunction": "na een onderschikkend voegwoord",
                     "relative": "betrekkelijke bijzin"},
           "fine": "Je functie volgt de regel in alle tien rijen. WALS geeft het Engels de volgorde SVO en het "
                   "Nederlands geen dominante volgorde, met twee: SOV of SVO. De rijen laten zien waar elk vandaan "
                   "komt: SVO in de overige Nederlandse zinnen, SOV na een onderschikkend voegwoord en met een "
                   "hulpwerkwoord. Onder *elke zin* telt elke zin "
                   "één keer: als betrekkelijke bijzin als het er een is, anders na een onderschikkend voegwoord, anders "
                   "met een hulpwerkwoord, anders bij de overige zinnen.",
           "differs": "In {n} van de tien rijen wijkt je oordeel af van de regel; die zijn paars.",
           "error": "Je functie stopt met een fout op {n} van de tien rijen.",
           "slow": "Je functie doet er te lang over op een dictionary met een paar aantallen."},
}


def dominant_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _DOMINANT_WORDS[lang]
    tables, off, errors = [], 0, 0
    for tb in ("en", "nl"):
        rows, styles = [], {}
        for kind in KIND_ORDER:
            counts = kinds(tb)[kind]
            ranked = _ranked(counts)
            try:
                got = _call(fn, dict(counts))
            except Timeout:
                return yours.figure(w["title"], _note(w["slow"]), alt=f'{w["title"]}. {w["slow"]}')
            want = dominant(counts)
            i = len(rows)
            if got is _STOPPED:
                errors += 1
                shown = "⚠"
            else:
                shown = html.escape(_verdict(got, lang) if isinstance(got, str) else repr(got))
            if got is _STOPPED or got != want:
                off += 1
                styles[(i, 4)] = _OFF
            first = f"{ranked[0][0]} {_num(ranked[0][1], lang)}" if ranked else ""
            second = f"{ranked[1][0]} {_num(ranked[1][1], lang)}" if len(ranked) > 1 else ""
            rows.append([html.escape(w["kinds"][kind]), _num(sum(counts.values()), lang), first, second, shown])
        tables.append(f'<div style="{_HEAD};border:none;padding:0.4rem 0 0">{html.escape(TREEBANK_NAMES[tb])}</div>'
                      + _table(rows, w["head"], numbers=(1,), styles=styles))
    text = (w["error"].format(n=errors) if errors else w["differs"].format(n=off) if off else w["fine"])
    alt = f'{w["title"]}. ' + _bare(text)
    return yours.figure(w["title"], "".join(tables) + _note(text), alt=alt)


# --- the demo: a map of WALS -----------------------------------------------------------------------------------------

_DEMO_WORDS = {
    "en": {"title": "{feature}: {family}",
           "map": "Where the languages are spoken", "bars": "Languages per value", "y": "languages",
           "sizes": "{n} languages{of} have a value for this feature in WALS. The most common value is {top}: {k} of "
                    "them, {share}.",
           "of": " of the family {family}",
           "legend": {"81A": "Filled shapes: SOV circles, SVO triangles, verb first (VSO, VOS) squares, object first "
                             "(OVS, OSV) diamonds; open circles: no dominant order.",
                      "83A": "Filled shapes: OV circles, VO triangles; open circles: no dominant order.",
                      "85A": "Filled shapes: postpositions (post) circles, prepositions (pre) triangles, other "
                             "(inpositions, inside the noun phrase, or no adpositions) squares; open circles: no "
                             "dominant order."},
           "rest": "Small grey dots: the languages of other families.",
           "head": ["Value", "Languages", "Examples"],
           "place": "Each language is drawn at the place WALS gives it, by longitude and latitude: a point, also for "
                    "a language spoken over a wide area."},
    "nl": {"title": "{feature}: {family}",
           "map": "Waar de talen gesproken worden", "bars": "Talen per waarde", "y": "talen",
           "sizes": "{n} talen{of} hebben een waarde voor deze eigenschap in WALS. De waarde die het vaakst voorkomt, "
                    "is {top}: {k} ervan, {share}.",
           "of": " van de familie {family}",
           "legend": {"81A": "Dichte vormen: SOV rondjes, SVO driehoeken, werkwoord vooraan (VSO, VOS) vierkantjes, "
                             "lijdend voorwerp vooraan (OVS, OSV) ruitjes; open rondjes: geen dominante volgorde.",
                      "83A": "Dichte vormen: OV rondjes, VO driehoeken; open rondjes: geen dominante volgorde.",
                      "85A": "Dichte vormen: achterzetsels (achter) rondjes, voorzetsels (voor) driehoeken, anders "
                             "(binnen de woordgroep, of geen) vierkantjes; open rondjes: geen dominante volgorde."},
           "rest": "Kleine grijze stippen: de talen van andere families.",
           "head": ["Waarde", "Talen", "Voorbeelden"],
           "place": "Elke taal staat op de plaats die WALS ervoor geeft, naar lengte- en breedtegraad: één punt, ook "
                    "voor een taal die in een groot gebied gesproken wordt."},
}
_VALUE_NAMES = {"en": {"No dominant order": "no dominant order", "Postpositions": "postpositions",
                       "Prepositions": "prepositions", "Inpositions": "inpositions", "No adpositions": "no adpositions"},
                "nl": {"No dominant order": "geen dominante volgorde", "Postpositions": "achterzetsels",
                       "Prepositions": "voorzetsels", "Inpositions": "binnen de woordgroep",
                       "No adpositions": "geen adposities"}}

_W, _H = 660, 300                    # the map's size in pixels
_LON, _LAT = (-180.0, 180.0), (-58.0, 78.0)


def _xy(lon: float, lat: float) -> tuple:
    x = 8 + (lon - _LON[0]) / (_LON[1] - _LON[0]) * (_W - 16)
    y = 8 + (_LAT[1] - lat) / (_LAT[1] - _LAT[0]) * (_H - 16)
    return x, y


def _shape(x: float, y: float, k, r: float = 3.2) -> str:
    """A dot of group k on the map, in the shapes of chart.Panel.group; None draws an open circle."""
    if k is None:
        return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r - 0.6:.1f}" style="fill:var(--ea-page);'
                f'stroke:var(--ea-ink-3);stroke-width:1.1"/>')
    look = f'style="fill:var(--ea-group-{k + 1});stroke:var(--ea-page);stroke-width:0.7"'
    if k == 0:
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" {look}/>'
    if k == 2:
        s = 1.7 * r
        return f'<rect x="{x - s / 2:.1f}" y="{y - s / 2:.1f}" width="{s:.1f}" height="{s:.1f}" {look}/>'
    pts = [(0, -1.35), (-1.2, 0.8), (1.2, 0.8)] if k == 1 else [(0, -1.35), (1.35, 0), (0, 1.35), (-1.35, 0)]
    return f'<polygon points="{" ".join(f"{x + u * r:.1f},{y + v * r:.1f}" for u, v in pts)}" {look}/>'


def map_svg(feature: str, family: str, alt: str) -> str:
    """Every language with a value for the feature at its place; the family's languages in the shape and color of
    their value, the others as small grey dots."""
    column = FEATURES[feature]["column"]
    group = {v[0]: v[3] for v in FEATURES[feature]["values"]}
    lines = []
    for lat in (-30, 0, 30, 60):
        _, y = _xy(0, lat)
        dash = "" if lat == 0 else ";stroke-dasharray:3 4"
        lines.append(f'<line x1="8" y1="{y:.1f}" x2="{_W - 8}" y2="{y:.1f}" style="stroke:var(--ea-line);'
                     f'stroke-width:0.8{dash}"/>')
    rest, mine = [], []
    for row in wals():
        if not row[column]:
            continue
        x, y = _xy(float(row["longitude"]), float(row["latitude"]))
        if family and row["family"] != family:
            rest.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.3" style="fill:var(--ea-ink-3);opacity:0.55"/>')
        else:
            mine.append((group[row[column]] is None, _shape(x, y, group[row[column]])))
    dots = "".join(rest) + "".join(s for _, s in sorted(mine, key=lambda t: not t[0]))   # open circles first, below
    return (f'<figure class="ea-figure ea-plot" role="img" aria-label="{html.escape(alt)}" '
            f'style="margin:0.25rem 0 0.5rem;max-width:min(100%,{_W}px)">'
            f'<svg viewBox="0 0 {_W} {_H}" aria-hidden="true" style="width:100%;height:auto">'
            f'<rect x="0.5" y="0.5" width="{_W - 1}" height="{_H - 1}" rx="6" style="fill:none;stroke:var(--ea-line)"/>'
            f'{"".join(lines)}{dots}</svg></figure>')


def _examples(feature: str, family: str, value: str, lang: str, k: int = 4) -> str:
    column = FEATURES[feature]["column"]
    names = [row["name"] for row in wals() if row[column] == value and (not family or row["family"] == family)]
    known = sorted((n for n in names if n in KNOWN), key=_KNOWN_ORDER.index)
    others = sorted(n for n in names if n not in KNOWN)
    picked = (known + others)[:k]
    return ", ".join(language_name(n, lang) for n in picked)


def demo_picture(feature: str, family: str, lang: str) -> str:
    w = _DEMO_WORDS[lang]
    spec = FEATURES[feature]
    counts = counts_of(feature, family)
    total = sum(counts.values())
    family_name = FAMILY_NAMES[lang][family]
    title = w["title"].format(feature=FEATURE_NAMES[lang][feature], family=family_name)
    # the bars: one per label, values with the same label (other) together
    labels, groups, ns = [], [], []
    for value, en, nl, k in spec["values"]:
        label = en if lang == "en" else nl
        if label in labels:
            ns[labels.index(label)] += counts.get(value, 0)
            continue
        labels.append(label)
        groups.append(k)
        ns.append(counts.get(value, 0))
    top = max(max(ns), 1)
    p = chart.Panel((0.4, len(labels) + 0.6), (0, top * 1.18), title=w["bars"], ylabel=w["y"],
                    xticks=list(range(1, len(labels) + 1)), xtick_labels=labels)
    for i, (n, k) in enumerate(zip(ns, groups), start=1):
        p.rect(i - 0.32, 0, i + 0.32, n, role="ink-3" if k is None else f"group-{k + 1}", solid=k is not None)
        p.text(i, n + top * 0.06, _num(n, lang), role="ink-2")
    best = max(counts, key=lambda v: counts[v]) if counts else ""
    best_name = _VALUE_NAMES[lang].get(best, best)
    sizes = w["sizes"].format(n=_num(total, lang), of=w["of"].format(family=family_name) if family else "",
                              top=best_name, k=_num(counts.get(best, 0), lang),
                              share=_pct(counts.get(best, 0) / total, lang) if total else "")
    legend = w["legend"][feature] + (" " + w["rest"] if family else "")
    rows = []
    for value, en, nl, k in spec["values"]:
        if counts.get(value):
            rows.append([html.escape(_VALUE_NAMES[lang].get(value, value)), _num(counts[value], lang),
                         html.escape(_examples(feature, family, value, lang))])
    alt = f"{title}. {sizes} " + "; ".join(f"{_bare(r[0])} {r[1]}: {_bare(r[2])}" for r in rows) + f". {legend}"
    body = (map_svg(feature, family, alt) + _note(legend + " " + w["place"]) +
            _svg([p], alt, lang, width=380, height=240) + _note(sizes) +
            _table(rows, w["head"], numbers=(1,)))
    return yours.figure(title, body, alt=alt)


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["order", "count-orders", "dominant", "family-or-type"]

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
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_value(k)}: {_value(x)}" for k, x in v.items()) + "}"
    return repr(v)


def _show_counts(v):
    """{got} and {want} of count_orders: a dict from the most to the least common order."""
    if isinstance(v, dict) and all(isinstance(x, int) for x in v.values()):
        return _value(dict(sorted(v.items(), key=lambda kv: -kv[1])))
    return _value(v)


class Named:
    """A value of the lesson that a message shows by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


class _Treebank:
    """The sentences of a treebank, made when a check first needs them, and shown in a message by a name."""

    def __init__(self, name: str, lang: str):
        self.name, self.lang = name, lang

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return treebank(self.lang)


class _Counts:
    """The counts of the orders of a treebank, made when a check first needs them, and shown in a message by a name."""

    def __init__(self, name: str, lang: str):
        self.name, self.lang = name, lang

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return dict(kinds(self.lang)["all"])


SENTENCE_N = Named("SENTENCE", SENTENCE)
CLAUSE_N = Named("CLAUSE", CLAUSE)


def _lit(v) -> Named:
    """A value of the check's own, shown in a message as Python, with strings in double quotes."""
    return Named(_value(v), v)


DUTCH_N = _Treebank("DUTCH", "nl")
COUNTS_N = _Counts("COUNTS", "nl")

# exercise 1: order. The cases of the check's own hold the three most common orders, a verb with only a subject, a
# relative clause, a verb as the first word, and a sentence with two verbs, each with a subject of its own.
ORDER_CASES = [
    (SENTENCE_N, 2),
    (CLAUSE_N, 6),
    (_lit(MAIN), 3),
    (_lit(SLEEPS), 3),
    (_lit(RELATIVE), 6),
    (_lit(DOOR), 1),
    (_lit(SAW_THAT), 3),
    (_lit(SAW_THAT), 9),
    (_lit(HELLO), 1),
]
_O_START = "def order(sentence, h):\n    ...\n"
_O = ("def order(sentence, h):\n"
      "    letters = \"\"\n"
      "    for i in range(1, len(sentence) + 1):\n"
      "        form, head, relation = sentence[i - 1]\n"
      "        if i == h:\n"
      "            letters = letters + \"V\"\n"
      "        if head == h and relation == \"nsubj\":\n"
      "            letters = letters + \"S\"\n"
      "        if head == h and relation == \"obj\":\n"
      "            letters = letters + \"O\"\n"
      "    return letters\n")
_O_ANY = (_O.replace('if head == h and relation == "nsubj":', 'if relation == "nsubj":')
          .replace('if head == h and relation == "obj":', 'if relation == "obj":'))
_O_NO_V = _O.replace("        if i == h:\n            letters = letters + \"V\"\n", "")
_O_ZERO = _O.replace("range(1, len(sentence) + 1)", "range(len(sentence))").replace("sentence[i - 1]", "sentence[i]")
_O_HEAD = _O.replace("        if i == h:\n", "        if head == 0:\n")


def _index_error(e) -> bool:
    return isinstance(e, IndexError)


def _letters_list(got, text) -> bool:
    return isinstance(got, list) and all(isinstance(x, str) for x in got)


def _numbers_not_letters(got, text) -> bool:
    return isinstance(got, (list, tuple)) and bool(got) and all(isinstance(x, int) for x in got)


# exercise 2: count_orders. Besides the Dutch treebank, short lists of the course's sentences: two orders once, an
# order twice, a sentence without object, two clauses of other orders, sentences without any clause of three letters,
# and an empty list.
COUNT_CASES = [
    (DUTCH_N,),
    (Named("[SENTENCE, CLAUSE]", [SENTENCE, CLAUSE]),),
    (Named("[CLAUSE, CLAUSE, SENTENCE]", [CLAUSE, CLAUSE, SENTENCE]),),
    (_lit([MAIN, SLEEPS]),),
    (_lit([SAW_THAT, RELATIVE]),),
    (_lit([DOOR, HELLO]),),
    ([],),
]
_C_START = "def count_orders(sentences):\n    ...\n"
_C = ("def count_orders(sentences):\n"
      "    counts = {}\n"
      "    for sentence in sentences:\n"
      "        for h in range(1, len(sentence) + 1):\n"
      "            letters = order(sentence, h)\n"
      "            if len(letters) == 3:\n"
      "                if letters in counts:\n"
      "                    counts[letters] = counts[letters] + 1\n"
      "                else:\n"
      "                    counts[letters] = 1\n"
      "    return counts\n")
_C_LAST = _C.replace("range(1, len(sentence) + 1)", "range(1, len(sentence))")
_C_ONE = _C.replace("                    counts[letters] = counts[letters] + 1\n", "                    counts[letters] = 1\n")
_C_INSIDE = _C.replace("    counts = {}\n    for sentence in sentences:\n",
                       "    for sentence in sentences:\n        counts = {}\n")


class OrderArguments(TypeError):
    """The course's order, as the check gives it to count_orders, got something other than a sentence and a number."""


def _given_order(sentence, h):
    if not isinstance(sentence, (list, tuple)) or not isinstance(h, int) or isinstance(h, bool):
        raise OrderArguments("order takes a sentence first and then the number of a word")
    return order(sentence, h)


def _order_arguments(e) -> bool:
    return isinstance(e, OrderArguments)


def _short_orders(got, text) -> bool:
    return isinstance(got, dict) and any(isinstance(k, str) and len(k) < 3 for k in got)


def _a_list(got, text) -> bool:
    return isinstance(got, list)


# exercise 3: dominant. Besides the Dutch counts: the English ones, one order alone, an order exactly twice as common
# as the next and one just more, the most common order not first in the dict, two orders equally common, and an order
# far ahead of two others.
DOMINANT_CASES = [
    (COUNTS_N,),
    ({"SVO": 5001, "OSV": 271, "VOS": 1},),
    ({"SOV": 5},),
    ({"SVO": 10, "SOV": 5},),
    ({"SVO": 11, "SOV": 5},),
    ({"VSO": 3, "SOV": 20, "SVO": 2},),
    ({"SOV": 4, "SVO": 4},),
    ({"OVS": 7, "VSO": 2, "SVO": 2},),
]
_D_START = "def dominant(counts):\n    ...\n"
_D = ("def dominant(counts):\n"
      "    ranked = sorted(counts.values(), reverse=True)\n"
      "    if len(ranked) > 1 and ranked[0] <= 2 * ranked[1]:\n"
      "        return \"\"\n"
      "    for letters in counts:\n"
      "        if counts[letters] == ranked[0]:\n"
      "            return letters\n")
_D_TWICE = _D.replace("ranked[0] <= 2 * ranked[1]", "ranked[0] < 2 * ranked[1]")
_D_NO_RULE = _D.replace("    if len(ranked) > 1 and ranked[0] <= 2 * ranked[1]:\n        return \"\"\n", "")
_D_LAST = _D.replace("ranked[0] <= 2 * ranked[1]", "ranked[0] <= 2 * ranked[-1]")


def _a_number(got, text) -> bool:
    return isinstance(got, (int, float)) and not isinstance(got, bool)


def _words_not_empty(got, text) -> bool:
    return isinstance(got, str) and len(got) > 3


EXERCISES = {
    "order": FunctionBox(
        name="order",
        picture=order_picture,
        function="order",
        start=_O_START,
        solution=_O,
        visible=ORDER_CASES[:1],
        hidden=ORDER_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("index", error=_index_error,
                    en="With `{call}`, your function looks for a word that the sentence does not have. Word `i` is at place `i - 1` of the list, since the words are numbered from 1.",
                    nl="Met `{call}` zoekt je functie een woord dat de zin niet heeft. Woord `i` staat op plaats `i - 1` van de lijst, want de woorden zijn genummerd vanaf 1."),
            Mistake("list", matches=_letters_list,
                    en="`{call}` gives a list of letters, and it should give one string: {want}. Start with the empty string `\"\"` and add each letter with `+`.",
                    nl="`{call}` geeft een lijst letters, en dat moet één string zijn: {want}. Begin met de lege string `\"\"` en voeg elke letter toe met `+`."),
            Mistake("numbers", matches=_numbers_not_letters,
                    en="`{call}` gives word numbers, and it should give letters: {want}. Add `\"V\"`, `\"S\"` or `\"O\"` for each of the three words, in the order of the words.",
                    nl="`{call}` geeft nummers van woorden, en dat moeten letters zijn: {want}. Voeg `\"V\"`, `\"S\"` of `\"O\"` toe voor elk van de drie woorden, in de volgorde van de woorden."),
            Mistake("any-verb", wrong=_O_ANY,
                    en="`{call}` gives {got}, and it should give {want}: your function also takes the subject and the object of another verb in the sentence, and only those of verb `h` count.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie neemt ook het onderwerp en het lijdend voorwerp van een ander werkwoord in de zin mee, en alleen die van werkwoord `h` tellen mee."),
            Mistake("no-v", wrong=_O_NO_V,
                    en="`{call}` gives {got}, and it should give {want}: the verb itself is missing. Add `\"V\"` when you reach word `h`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het werkwoord zelf ontbreekt. Voeg `\"V\"` toe als je bij woord `h` bent."),
            Mistake("zero", wrong=_O_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: your numbers start at 0, and the words of a sentence are numbered from 1, as the heads are. Go through `range(1, len(sentence) + 1)`, and take word `i` from place `i - 1`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je nummers beginnen bij 0, en de woorden van een zin zijn genummerd vanaf 1, net als de hoofdwoorden. Ga `range(1, len(sentence) + 1)` langs, en neem woord `i` van plaats `i - 1`."),
            Mistake("root", wrong=_O_HEAD,
                    en="`{call}` gives {got}, and it should give {want}: your function puts the V at the root of the sentence, and the verb to look at is word `h`, which need not be the root.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie zet de V bij de root van de zin, en het werkwoord waar het om gaat, is woord `h`, dat niet de root hoeft te zijn."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: go through the words of `sentence` in order, and add `\"V\"` for word `h`, `\"S\"` for its subject and `\"O\"` for its object.",
                      "nl": "Vervang de `...`: ga de woorden van `sentence` op volgorde langs, en voeg `\"V\"` toe voor woord `h`, `\"S\"` voor het onderwerp en `\"O\"` voor het lijdend voorwerp."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function add one letter for word `h`, one for its subject and one for its object, in the order of the words?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Voegt je functie één letter toe voor woord `h`, één voor het onderwerp en één voor het lijdend voorwerp, in de volgorde van de woorden?"},
            "almost": {"en": "Right for SENTENCE. But `{call}` gives {got}, and it should give {want}. Does your function work for any sentence and any verb, also one without an object, or one in a sentence with two verbs?",
                       "nl": "Goed voor SENTENCE. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke zin en elk werkwoord, ook een werkwoord zonder lijdend voorwerp, of een werkwoord in een zin met twee werkwoorden?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back: it has a `return`, but Python does not get to it. Is the `return` the last line, after the loop?",
                          "nl": "Met `{call}` geeft je functie niets terug: er staat een `return` in, maar Python komt er niet aan toe. Is de `return` de laatste regel, na de loop?"},
            "correct": {"en": "Your `order` is right in all {n} cases. Look at the picture: the orders of seven clauses from the treebanks.",
                        "nl": "Je `order` klopt in alle {n} gevallen. Kijk naar de afbeelding: de volgordes van zeven zinnen uit de treebanks."},
        },
    ),
    "count-orders": FunctionBox(
        name="count-orders",
        picture=counts_picture,
        function="count_orders",
        start=_C_START,
        solution=_C,
        visible=COUNT_CASES[:1],
        hidden=COUNT_CASES[1:],
        first=[_DOTS],
        names={"order": _given_order},
        show=_show_counts,
        mistakes=[
            Mistake("order-arguments", error=_order_arguments,
                    en="With `{call}`, your function calls `order` with something other than a sentence and the number of a word. Call it as `order(sentence, h)`, with one sentence of `sentences` and a number from 1 on.",
                    nl="Met `{call}` roept je functie `order` aan met iets anders dan een zin en het nummer van een woord. Roep de functie aan als `order(sentence, h)`, met één zin uit `sentences` en een nummer vanaf 1."),
            Mistake("short", matches=_short_orders,
                    en="`{call}` gives {got}, and it should give {want}: your counts also hold orders with fewer than three letters, such as the letters of a word without subject or object. Count only the orders with all three letters.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je aantallen bevatten ook volgordes met minder dan drie letters, zoals de letters van een woord zonder onderwerp of lijdend voorwerp. Tel alleen de volgordes met alle drie de letters."),
            Mistake("a-list", matches=_a_list,
                    en="`{call}` gives a list, and it should give a dict from each order to how often it occurs: {want}.",
                    nl="`{call}` geeft een lijst, en dat moet een dictionary zijn van elke volgorde naar hoe vaak die voorkomt: {want}."),
            Mistake("last-word", wrong=_C_LAST,
                    en="`{call}` gives {got}, and it should give {want}: your function skips the last word of every sentence, and in Dutch the verb is often there.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie slaat het laatste woord van elke zin over, en in het Nederlands staat het werkwoord daar vaak."),
            Mistake("one", wrong=_C_ONE,
                    en="`{call}` gives {got}, and it should give {want}: every count stays at 1. An order you meet again gets its count plus 1.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elk aantal blijft 1. Een volgorde die je opnieuw tegenkomt, krijgt het aantal plus 1."),
            Mistake("inside", wrong=_C_INSIDE,
                    en="`{call}` gives {got}, and it should give {want}: your counts start again at every sentence, so only the last sentence is left. Make the dict once, before the loops.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je aantallen beginnen bij elke zin opnieuw, dus alleen de laatste zin blijft over. Maak de dictionary één keer, vóór de loops."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: call `order(sentence, h)` for every word `h` of every sentence, and count how often each order of three letters comes back, in a dict.",
                      "nl": "Vervang de `...`: roep `order(sentence, h)` aan voor elk woord `h` van elke zin, en tel in een dictionary hoe vaak elke volgorde van drie letters terugkomt."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function ask `order` about every word of every sentence, and count only the orders of three letters?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Vraagt je functie `order` naar elk woord van elke zin, en telt de functie alleen de volgordes van drie letters?"},
            "almost": {"en": "Right for DUTCH. But `{call}` gives {got}, and it should give {want}. Does your function work for any list of sentences, also a short one, or an empty one?",
                       "nl": "Goed voor DUTCH. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke lijst zinnen, ook een korte, of een lege?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back: it has a `return`, but Python does not get to it. Is the `return` the last line, after both loops?",
                          "nl": "Met `{call}` geeft je functie niets terug: er staat een `return` in, maar Python komt er niet aan toe. Is de `return` de laatste regel, na beide loops?"},
            "correct": {"en": "Your `count_orders` is right in all {n} cases. Look at the picture: the orders of every clause of the two treebanks.",
                        "nl": "Je `count_orders` klopt in alle {n} gevallen. Kijk naar de afbeelding: de volgordes van elke zin van de twee treebanks."},
        },
    ),
    "dominant": FunctionBox(
        name="dominant",
        picture=dominant_picture,
        function="dominant",
        start=_D_START,
        solution=_D,
        visible=DOMINANT_CASES[:1],
        hidden=DOMINANT_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("a-number", matches=_a_number,
                    en="`{call}` gives a number, and it should give the letters of an order, or the empty string: {want}.",
                    nl="`{call}` geeft een getal, en dat moeten de letters van een volgorde zijn, of de lege string: {want}."),
            Mistake("words", matches=_words_not_empty,
                    en="`{call}` gives {got}, and it should give {want}: when no order is dominant, give back the empty string `\"\"`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: als geen volgorde dominant is, geef dan de lege string `\"\"` terug."),
            Mistake("twice", wrong=_D_TWICE,
                    en="`{call}` gives {got}, and it should give {want}: an order exactly twice as common as the next is not dominant yet. It has to be more than twice as common.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een volgorde die precies twee keer zo vaak voorkomt als de volgende, is nog niet dominant. Het moet meer dan twee keer zo vaak zijn."),
            Mistake("no-rule", wrong=_D_NO_RULE,
                    en="`{call}` gives {got}, and it should give {want}: your function gives the most common order every time. That order is only dominant when it is more than twice as common as the next one.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft elke keer de volgorde die het vaakst voorkomt. Die is pas dominant als die meer dan twee keer zo vaak voorkomt als de volgende."),
            Mistake("last", wrong=_D_LAST,
                    en="`{call}` gives {got}, and it should give {want}: your function compares with the least common order. The rule compares with the next one. Where in `ranked` is its count?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie vergelijkt met de volgorde die het minst vaak voorkomt. De regel vergelijkt met de volgende. Waar in `ranked` staat het aantal daarvan?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back the most common order of `counts` when it is more than twice as common as the next, and else the empty string.",
                      "nl": "Vervang de `...`: geef de volgorde van `counts` die het vaakst voorkomt terug als die meer dan twee keer zo vaak voorkomt als de volgende, en anders de lege string."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function compare the two highest counts, and give back the letters of the most common order only when it is more than twice as common as the next?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Vergelijkt je functie de twee hoogste aantallen, en geeft de functie de letters van de volgorde die het vaakst voorkomt alleen terug als die meer dan twee keer zo vaak voorkomt als de volgende?"},
            "almost": {"en": "Right for COUNTS. But `{call}` gives {got}, and it should give {want}. Does your function work for any dict of counts, also one with a single order, or with the most common order not first?",
                       "nl": "Goed voor COUNTS. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke dictionary met aantallen, ook een met één volgorde, of met de volgorde die het vaakst voorkomt niet vooraan?"},
            "no_return": {"en": "Your function gives nothing back. When no order is dominant, give back the empty string `\"\"`, and else the letters of the order, both with `return`.",
                          "nl": "Je functie geeft niets terug. Als geen volgorde dominant is, geef dan de lege string `\"\"` terug, en anders de letters van de volgorde, allebei met `return`."},
            "unreached": {"en": "With `{call}`, your function gives nothing back. When no order is dominant, give back the empty string `\"\"`; and does every way through the function end with a `return`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Als geen volgorde dominant is, geef dan de lege string `\"\"` terug; en eindigt elke weg door de functie met een `return`?"},
            "correct": {"en": "Your `dominant` is right in all {n} cases. Look at the picture: the rule of WALS on kinds of clauses in the two treebanks.",
                        "nl": "Je `dominant` klopt in alle {n} gevallen. Kijk naar de afbeelding: de regel van WALS op soorten zinnen in de twee treebanks."},
        },
    ),
    "family-or-type": Sort(
        name="family-or-type",
        groups={
            "family": {"en": "Family: where the languages come from, and their relatives",
                       "nl": "Familie: waar de talen vandaan komen, en hun verwanten"},
            "type": {"en": "Type: how the languages are built, whatever their family",
                     "nl": "Type: de talen zijn op dezelfde manier gebouwd, verwant of niet"},
        },
        items={
            "west": {"en": "**Dutch, English and Frisian all go back to West Germanic.**",
                     "nl": "**Het Nederlands, het Engels en het Fries gaan alle drie terug op het West-Germaans.**"},
            "verb-last": {"en": "**Japanese, Turkish and Basque all put the verb after the object.**",
                          "nl": "**Het Japans, het Turks en het Baskisch zetten alle drie het werkwoord na het lijdend voorwerp.**"},
            "basque": {"en": "**Basque has no known relatives among the languages of the world.**",
                       "nl": "**Het Baskisch heeft geen bekende verwanten onder de talen van de wereld.**"},
            "postpositions": {"en": "**Most languages with the object before the verb have postpositions.**",
                              "nl": "**De meeste talen met het lijdend voorwerp vóór het werkwoord hebben achterzetsels.**"},
            "afrikaans": {"en": "**In Glottolog, Afrikaans sits in the same branch as Dutch.**",
                          "nl": "**In Glottolog staat het Afrikaans in dezelfde tak als het Nederlands.**"},
            "verb-first": {"en": "**Welsh and Modern Standard Arabic both put the verb first.**",
                           "nl": "**Het Welsh en het Modern Standaardarabisch zetten allebei het werkwoord vooraan.**"},
        },
        solution={"west": "family", "verb-last": "type", "basque": "family", "postpositions": "type",
                  "afrikaans": "family", "verb-first": "type"},
        nudge={
            "west": {"en": "Does this say how the three languages build a sentence, or what they come from?",
                     "nl": "Zegt dit hoe de drie talen een zin bouwen, of waar de talen uit voortkomen?"},
            "verb-last": {"en": "In WALS, no two of these three languages are related. What do they share, then?",
                          "nl": "In WALS zijn geen twee van deze drie talen verwant. Wat hebben de drie dan gemeen?"},
            "basque": {"en": "Does this say anything about how Basque orders its words?",
                       "nl": "Zegt dit iets over de volgorde van de woorden in het Baskisch?"},
            "postpositions": {"en": "Do the languages in this statement have to be related?",
                              "nl": "Moeten de talen in deze uitspraak verwant zijn?"},
            "afrikaans": {"en": "What does a branch in the tree of Glottolog stand for?",
                          "nl": "Waar staat een tak in de boom van Glottolog voor?"},
            "verb-first": {"en": "Welsh is Indo-European, and Arabic Afro-Asiatic. What do the two share, then?",
                           "nl": "Het Welsh is Indo-Europees, en het Arabisch Afro-Aziatisch. Wat hebben de twee dan gemeen?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the statements again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de uitspraken opnieuw uit."},
            "empty": {"en": "Put each bold statement under what it is about.",
                      "nl": "Zet elke vetgedrukte uitspraak bij waar die over gaat."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} belongs under another group.",
                      "nl": "{item} hoort bij een andere groep."},
            "correct": {"en": "All {n} are right. A family is about where languages come from, a type about how they are built: Dutch and English are related and still differ in word order, while Welsh and Modern Standard Arabic are not related and share theirs.",
                        "nl": "Alle {n} kloppen. Een familie gaat over waar talen vandaan komen, een type over hoe talen gebouwd zijn: het Nederlands en het Engels zijn verwant en verschillen toch in woordvolgorde, terwijl het Welsh en het Modern Standaardarabisch niet verwant zijn en dezelfde volgorde hebben."},
        },
    ),
}

CASES = [
    ("order", _O_START, "empty", None),
    ("order", _O, "correct", None),
    ("order", "def order(sentence, h):\n    letters = \"\"\n    for i in range(len(sentence)):\n        form, head, relation = sentence[i]\n        if i + 1 == h:\n            letters += \"V\"\n        elif head == h and relation == \"nsubj\":\n            letters += \"S\"\n        elif head == h and relation == \"obj\":\n            letters += \"O\"\n    return letters\n", "correct", None),
    ("order", "def order(sentence, h):\n    marks = {\"nsubj\": \"S\", \"obj\": \"O\"}\n    letters = []\n    for i, (form, head, relation) in enumerate(sentence, start=1):\n        if i == h:\n            letters.append(\"V\")\n        elif head == h and relation in marks:\n            letters.append(marks[relation])\n    return \"\".join(letters)\n", "correct", None),
    ("order", _O_ANY, "wrong", "any-verb"),
    ("order", _O_NO_V, "wrong", "no-v"),
    ("order", _O_ZERO, "wrong", "zero"),
    ("order", _O_HEAD, "wrong", "root"),
    ("order", _O.replace("sentence[i - 1]", "sentence[i]"), "error", "index"),
    ("order", _O.replace("letters = \"\"", "letters = []").replace("letters = letters + \"V\"", "letters.append(\"V\")").replace("letters = letters + \"S\"", "letters.append(\"S\")").replace("letters = letters + \"O\"", "letters.append(\"O\")"), "wrong", "list"),
    ("order", "def order(sentence, h):\n    found = []\n    for i in range(1, len(sentence) + 1):\n        form, head, relation = sentence[i - 1]\n        if i == h or head == h and relation in (\"nsubj\", \"obj\"):\n            found.append(i)\n    return found\n", "wrong", "numbers"),
    ("order", _O.replace("    return letters\n", ""), "wrong", "no-return"),
    ("order", "def order(sentence, h):\n    return \"VSO\"\n", "almost", None),
    ("count-orders", _C_START, "empty", None),
    ("count-orders", _C, "correct", None),
    ("count-orders", "def count_orders(sentences):\n    counts = {}\n    for sentence in sentences:\n        for h in range(len(sentence)):\n            letters = order(sentence, h + 1)\n            if len(letters) == 3:\n                counts[letters] = counts.get(letters, 0) + 1\n    return counts\n", "correct", None),
    ("count-orders", _C.replace("if len(letters) == 3:", "if len(letters) > 0:"), "wrong", "short"),
    ("count-orders", _C_LAST, "wrong", "last-word"),
    ("count-orders", _C.replace("range(1, len(sentence) + 1)", "range(len(sentence))"), "wrong", "last-word"),
    ("count-orders", _C_ONE, "wrong", "one"),
    ("count-orders", _C_INSIDE, "wrong", "inside"),
    ("count-orders", _C.replace("order(sentence, h)", "order(h, sentence)"), "error", "order-arguments"),
    ("count-orders", "def count_orders(sentences):\n    found = []\n    for sentence in sentences:\n        for h in range(1, len(sentence) + 1):\n            letters = order(sentence, h)\n            if len(letters) == 3:\n                found.append(letters)\n    return found\n", "wrong", "a-list"),
    ("count-orders", _C.replace("    return counts\n", ""), "wrong", "no-return"),
    ("count-orders", "def count_orders(sentences):\n    return {\"SOV\": 2319, \"SVO\": 1564, \"VSO\": 716, \"OSV\": 359, \"OVS\": 250, \"VOS\": 3}\n", "almost", None),
    ("dominant", _D_START, "empty", None),
    ("dominant", _D, "correct", None),
    ("dominant", "def dominant(counts):\n    best = \"\"\n    for letters in counts:\n        if best == \"\" or counts[letters] > counts[best]:\n            best = letters\n    for letters in counts:\n        if letters != best and 2 * counts[letters] >= counts[best]:\n            return \"\"\n    return best\n", "correct", None),
    ("dominant", "def dominant(counts):\n    pairs = sorted([(n, letters) for letters, n in counts.items()], reverse=True)\n    if len(pairs) > 1 and pairs[0][0] <= 2 * pairs[1][0]:\n        return \"\"\n    return pairs[0][1]\n", "correct", None),
    ("dominant", _D_TWICE, "wrong", "twice"),
    ("dominant", _D_NO_RULE, "wrong", "no-rule"),
    ("dominant", _D_LAST, "wrong", "last"),
    ("dominant", _D.replace("            return letters\n", "            return counts[letters]\n").replace("        return \"\"\n", "        return 0\n"), "wrong", "a-number"),
    ("dominant", _D.replace("        return \"\"\n", "        return \"no dominant order\"\n"), "wrong", "words"),
    ("dominant", _D.replace("        return \"\"\n", "        return None\n"), "wrong", "unreached"),
    ("dominant", "def dominant(counts):\n    return \"\"\n", "almost", None),
    ("family-or-type", {}, "empty", None),
    ("family-or-type", {"west": "family", "verb-last": "type", "basque": "family", "postpositions": "type",
                        "afrikaans": "family", "verb-first": "type"}, "correct", None),
    ("family-or-type", {"west": "family", "verb-last": "type"}, "wrong", "unfinished"),
    ("family-or-type", {"verb-first": "family"}, "wrong", "wrong:verb-first"),
    ("family-or-type", {"west": "type", "verb-last": "type", "basque": "family", "postpositions": "type",
                        "afrikaans": "family", "verb-first": "type"}, "wrong", "wrong:west"),
]
