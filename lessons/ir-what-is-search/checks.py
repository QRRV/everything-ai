# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ir-what-is-search: the simplest search engine, one that gives every document that contains all
the words of a query (matches), judged against the information needs behind a query (split: the results that help, the
results that do not, and the relevant documents the search missed), and the three kinds of need of Broder (2002),
guessed from the words of a query with clue words as Jansen, Booth and Spink (2008) did (kind_of).

The documents, the needs, the clue words and the queries are the course's own. Two needs come from Manning, Raghavan and
Schütze, Introduction to Information Retrieval (2008), section 8.1: buying a pet python, and whether red wine protects
the heart better than white wine. The topic of the sort is TREC-8 ad hoc topic 427, *UV damage, eyes* (NIST, 1999); its
documents are the course's. Every number in the text and the pictures is computed by the course
(tests/test_ir_what_is_search.py computes them again)."""
import html
import importlib.util
import re

from everything_ai import paths, yours
from everything_ai.check import FunctionBox, Mistake, Sort

LANGS = ("en", "nl")


def _summarize():
    spec = importlib.util.spec_from_file_location("everything_ai_checks_nlp_summarize_for_what_is_search",
                                                  paths.LESSONS / "nlp-summarize" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _summarize()   # the parts of the pictures and the exercises

# --- the collection --------------------------------------------------------------------------------------------------------

# The course's small collection: the same thirteen documents in both languages, in the same order.
TEXTS = {
    "en": [
        "Python for beginners: install it, write your first program, and learn lists and loops.",
        "Ball pythons for sale: young snakes from a registered breeder, with a care sheet.",
        "The python is a large snake that kills its prey by squeezing it, in Africa, Asia and Australia.",
        "A pet python needs a warm tank: where to buy one, and what it costs in a year.",
        "A new version of Python is out: what changed in the language, and what still works as before.",
        "How to care for a python at home: the tank, the heat lamp and the food.",
        "Red wine, white wine and the heart: does the color of the wine change the risk of a heart attack? What studies found.",
        "Heart-healthy food: fruit, vegetables, nuts, and now and then a glass of red wine.",
        "The heart of the old town: a walk past the market, the church and a bar with good red wine.",
        "Alcohol and heart attacks: do beer, white and red differ? An overview of the studies.",
        "Red wine with dinner: which red goes with fish, and which with cheese?",
        "Train times from Boston to New York, with the price of a ticket.",
        "How to boil an egg so that the yolk stays soft.",
    ],
    "nl": [
        "Python voor beginners: installeer het, schrijf je eerste programma, en leer lijsten en loops.",
        "Koningspythons te koop: jonge slangen van een erkende fokker, met een verzorgingsblad.",
        "De python is een grote slang die een prooi doodt door het dier te wurgen, in Afrika, Azië en Australië.",
        "Een python als huisdier heeft een warm terrarium nodig: waar je er een koopt, en wat het per jaar kost.",
        "Een nieuwe versie van Python is uit: wat er veranderd is in de taal, en wat nog werkt zoals altijd.",
        "Een python thuis verzorgen: het terrarium, de warmtelamp en het eten.",
        "Rode wijn, witte wijn en het hart: maakt de kleur van de wijn verschil voor de kans op een hartaanval? Wat onderzoek vond.",
        "Eten voor een gezond hart: fruit, groente, noten, en af en toe een glas rode wijn.",
        "Het hart van de oude stad: een wandeling langs de markt, de kerk en een café met goede rode wijn.",
        "Alcohol en hartaanvallen: verschillen bier, wit en rood? Een overzicht van het onderzoek.",
        "Rode wijn bij het eten: welke rode past bij vis, en welke bij kaas?",
        "Treintijden van Utrecht naar Amsterdam, met de prijs van een kaartje.",
        "Een ei koken zodat de dooier zacht blijft.",
    ],
}


def words(text_: str) -> list:
    """The words of a text in small letters, as the lessons on tokenizing cut them; a dot inside a word stays, as in
    bol.com."""
    return re.findall(r"\w+(?:\.\w+)*", str(text_).lower())


DOCS = {lang: [words(t) for t in TEXTS[lang]] for lang in LANGS}   # each document as a list of words

# The two queries of the lesson, as lists of words, and the needs behind them with the documents that help with each:
# the course's judgments. The first two needs share the query python (Manning and others, section 8.1).
QUERIES = {"en": {"python": ["python"], "wine": ["red", "wine", "heart"]},
           "nl": {"python": ["python"], "wine": ["rode", "wijn", "hart"]}}
NEEDS = {
    "buy": {"query": "python", "relevant": [1, 3]},
    "learn": {"query": "python", "relevant": [0]},
    "wine": {"query": "wine", "relevant": [6, 9]},
}
NEED_TEXT = {
    "en": {"buy": "Where can I buy a pet python?",
           "learn": "How do I start learning the programming language Python?",
           "wine": "Is drinking red wine more effective than white wine at lowering the risk of a heart attack?"},
    "nl": {"buy": "Waar kan ik een python als huisdier kopen?",
           "learn": "Hoe begin ik met het leren van de programmeertaal Python?",
           "wine": "Verlaagt rode wijn de kans op een hartaanval meer dan witte wijn?"},
}


# The course's solutions, the same code as the exercises.
def matches(query, document):
    """True when every word of the query is in the document."""
    for word in query:
        if word not in document:
            return False
    return True


def search(query, docs, match=matches) -> list:
    """The numbers of the documents that match the query, in the order of the collection: the results."""
    return [i for i in range(len(docs)) if match(query, docs[i])]


def split(results, relevant):
    """The results that are relevant, the results that are not, and the relevant documents that are not results."""
    found = []
    extra = []
    for doc in results:
        if doc in relevant:
            found.append(doc)
        else:
            extra.append(doc)
    missed = [doc for doc in relevant if doc not in results]
    return found, extra, missed


def results_for(need: str, lang: str, match=matches) -> list:
    return search(QUERIES[lang][NEEDS[need]["query"]], DOCS[lang], match)


# --- the kinds of need -------------------------------------------------------------------------------------------------------

KINDS = ("navigational", "informational", "transactional")
KIND_NAMES = {"en": {"navigational": "navigational", "informational": "informational", "transactional": "transactional"},
              "nl": {"navigational": "navigationeel", "informational": "informationeel", "transactional": "transactioneel"}}
# Clue words, after the characteristics of Jansen, Booth and Spink (2008), section 5.2: names of sites and of the
# organizations behind them, and words such as login, point to a navigational need; words for buying, booking and
# downloading to a transactional one. A query without a clue is informational, as in their rules. The kinds are tested
# in this order.
CLUES = {
    "en": {"navigational": ["www", "login", "homepage", "website", "amtrak", "wikipedia", "youtube", "ebay", "facebook"],
           "transactional": ["buy", "download", "order", "book", "tickets", "cheap", "sale", "price"]},
    "nl": {"navigational": ["www", "inloggen", "homepage", "website", "ns.nl", "wikipedia", "youtube", "bol.com",
                            "facebook"],
           "transactional": ["kopen", "downloaden", "bestellen", "boeken", "kaartjes", "goedkoop", "goedkope",
                             "aanbieding", "prijs"]},
}
# Queries with the kind of need the course reads in them; only the person who typed a query knows its need for sure.
KIND_QUERIES = {
    "en": [("amtrak", "navigational"), ("youtube", "navigational"), ("facebook login", "navigational"),
           ("wikipedia python", "navigational"), ("lufthansa airlines", "navigational"),
           ("buy a ball python", "transactional"), ("download python", "transactional"),
           ("cheap flights to miami", "transactional"), ("book hotel amsterdam", "transactional"),
           ("leukemia", "informational"), ("provence", "informational"), ("how to learn python", "informational"),
           ("red wine heart attack", "informational"), ("ebay history", "informational"),
           ("book about snakes", "informational")],
    "nl": [("ns.nl", "navigational"), ("youtube", "navigational"), ("facebook inloggen", "navigational"),
           ("wikipedia python", "navigational"), ("lufthansa", "navigational"),
           ("koningspython kopen", "transactional"), ("python downloaden", "transactional"),
           ("goedkope vluchten naar barcelona", "transactional"), ("hotel boeken amsterdam", "transactional"),
           ("leukemie", "informational"), ("provence", "informational"), ("hoe leer ik python", "informational"),
           ("rode wijn hart", "informational"), ("geschiedenis van bol.com", "informational"),
           ("boeken over slangen", "informational")],
}


def kind_of(query, clues):
    """The first kind, in the order of clues, with a clue word in the query; informational when there is none."""
    for kind in clues:
        for word in query:
            if word in clues[kind]:
                return kind
    return "informational"


def kinds(lang: str, guess=kind_of) -> list:
    """For each query of the lesson: (query, the course's reading, what guess gives)."""
    return [(q, reading, guess(words(q), CLUES[lang])) for q, reading in KIND_QUERIES[lang]]


# --- the pictures ----------------------------------------------------------------------------------------------------------

_UI, _LABEL, _CELL, _HEAD, _NUMBER = S._UI, S._LABEL, S._CELL, S._HEAD, S._NUMBER
_DATA, _FAINT, _OUT, _PRED, _LOSS = S._DATA, S._FAINT, S._OUT, S._PRED, S._LOSS
_CHIP, _CHIP_PRED, _CHIP_DATA, _CHIP_LOSS = S._CHIP, S._CHIP_PRED, S._CHIP_DATA, S._CHIP_LOSS
_note, _colors, _label, _num = S._note, S._colors, S._label, S._num
_cell, _table = S._cell, S._table
_fn, _call, _empty, _SLOW, _STOPPED = S._fn, S._call, S._empty, S._SLOW, S._STOPPED
_WHY_EMPTY = S._WHY_EMPTY
_MONO = "font-family:var(--ea-font-code);font-size:0.8rem"
_DASH = f'<span style="{_UI}">–</span>'
_DOC_CAP = 60   # characters of a document that a chip shows


def _chip(text_: str, style: str = _CHIP_PRED) -> str:
    return f'<span style="{style}">{html.escape(text_)}</span>'


def _code(text_: str) -> str:
    return f'<code style="{_MONO}">{html.escape(text_)}</code>'


def _doc_text(i: int, lang: str, cap: int = 0) -> str:
    t = TEXTS[lang][i]
    return t if not cap or len(t) <= cap else t[:cap].rstrip() + "…"


def _doc_chip(i: int, lang: str, style: str) -> str:
    return _chip(f"{i} {_doc_text(i, lang, _DOC_CAP)}", style)


def _query_text(query) -> str:
    return " ".join(query)


class _Odd(Exception):
    """A learner's function gave something a picture cannot use."""


def _guarded(name: str, fn, lang: str = "en"):
    """The learner's function, stopping with _Odd when it gives what the picture cannot use."""
    if name == "matches":
        def step(query, document):
            out = fn(list(query), list(document))
            if not isinstance(out, bool):
                raise _Odd
            return out
    elif name == "split":
        def step(results, relevant):
            out = fn(list(results), list(relevant))
            if not (isinstance(out, (tuple, list)) and len(out) == 3
                    and all(isinstance(part, (list, tuple, set)) for part in out)
                    and all(isinstance(d, int) and not isinstance(d, bool) and 0 <= d < len(TEXTS[lang])
                            for part in out for d in part)):
                raise _Odd
            return tuple(list(part) for part in out)
    else:
        def step(query, clues):
            out = fn(list(query), {k: list(v) for k, v in clues.items()})
            if out not in KINDS:
                raise _Odd
            return out
    return step


_IDEA_TEXT = {
    "en": {"title": "One query, two needs",
           "query": "The query, as a list of words:",
           "needs": "Two information needs behind it:",
           "results": "The results: the documents that contain every word of the query, judged against each need:",
           "head": ["", "Document", "Need 1", "Need 2"],
           "helps": "helps", "not": "does not help",
           "missed": "Relevant, but not a result: it does not contain the word *python*:",
           "missed_for": "missed for need 1",
           "gray": "Gray: the query, the needs and the documents.",
           "blue": "Blue: a result that helps with the need.",
           "purple": "Purple: a result that does not help with the need."},
    "nl": {"title": "Eén zoekvraag, twee informatiebehoeften",
           "query": "De zoekvraag, als lijst woorden:",
           "needs": "Twee informatiebehoeften erachter:",
           "results": "De resultaten: de documenten met elk woord van de zoekvraag, beoordeeld op elke behoefte:",
           "head": ["", "Document", "Behoefte 1", "Behoefte 2"],
           "helps": "helpt", "not": "helpt niet",
           "missed": "Relevant, maar geen resultaat: het woord *python* staat er niet in:",
           "missed_for": "gemist bij behoefte 1",
           "gray": "Grijs: de zoekvraag, de behoeften en de documenten.",
           "blue": "Blauw: een resultaat dat helpt bij de behoefte.",
           "purple": "Paars: een resultaat dat niet helpt bij de behoefte."},
}


def idea_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang]
    query = QUERIES[lang]["python"]
    results = search(query, DOCS[lang])
    body = _label(w["query"]) + f'<div style="{_DATA}">{_code(repr(query))}</div>'
    body += _label(w["needs"]) + "".join(f'<div style="{_DATA}">{k}. {html.escape(NEED_TEXT[lang][n])}</div>'
                                         for k, n in ((1, "buy"), (2, "learn")))
    rows = []
    for i in results:
        cells = "".join(_cell(_chip(w["helps"]) if i in NEEDS[n]["relevant"] else _chip(w["not"], _CHIP_LOSS))
                        for n in ("buy", "learn"))
        rows.append("<tr>" + _cell(str(i), _FAINT, _NUMBER) + _cell(html.escape(TEXTS[lang][i]), _DATA) + cells + "</tr>")
    body += _label(w["results"]) + _table(rows, w["head"], (0,))
    missed = split(results, NEEDS["buy"]["relevant"])[2]
    rows = ["<tr>" + _cell(str(i), _FAINT, _NUMBER) + _cell(html.escape(TEXTS[lang][i]), _DATA)
            + _cell(_chip(w["missed_for"], _CHIP_DATA)) + "</tr>" for i in missed]
    body += _label(w["missed"]) + _table(rows, w["head"][:3], (0,))
    body += _colors(w, body)
    return yours.figure(w["title"], body, alt=w["title"])


_MATCH_TEXT = {
    "en": {"title": "Your matches on the course's collection",
           "kind": "True or False",
           "head": "Document",
           "result": "result", "not": "not a result",
           "summary": "With your matches, the query *{q1}* gives {k1} {r1} and *{q2}* gives {k2} {r2}; with the course's matches, they give {c1} and {c2}.",
           "one": "result", "many": "results",
           "gray": "Gray: the documents.",
           "blue": "Blue: a result, as with the course's matches.",
           "purple": "Purple: where your matches and the course's differ."},
    "nl": {"title": "Je matches op de verzameling van de course",
           "kind": "True of False",
           "head": "Document",
           "result": "resultaat", "not": "geen resultaat",
           "summary": "Met je matches geeft de zoekvraag *{q1}* {k1} {r1} en *{q2}* {k2} {r2}; met de matches van de course zijn het er {c1} en {c2}.",
           "one": "resultaat", "many": "resultaten",
           "gray": "Grijs: de documenten.",
           "blue": "Blauw: een resultaat, net als bij de matches van de course.",
           "purple": "Paars: waar je matches en de matches van de course verschillen."},
}


def match_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _MATCH_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    step = _guarded("matches", fn, lang)
    names = ("python", "wine")

    def make():
        try:
            return {q: search(QUERIES[lang][q], DOCS[lang], step) for q in names}
        except _Odd:
            return None

    got = _call(make)
    if got is _SLOW:
        return _empty(w, "slow")
    if got is _STOPPED:
        return _empty(w, "stopped")
    if got is None:
        return _empty(w, "odd")
    want = {q: search(QUERIES[lang][q], DOCS[lang]) for q in names}
    rows = []
    for i in range(len(TEXTS[lang])):
        cells = ""
        for q in names:
            mine, course = i in got[q], i in want[q]
            if mine:
                cells += _cell(_chip(w["result"], _CHIP_PRED if course else _CHIP_LOSS))
            elif course:
                cells += _cell(_chip(w["not"], _CHIP_LOSS))
            else:
                cells += _cell(_DASH)
        rows.append("<tr>" + _cell(str(i), _FAINT, _NUMBER) + _cell(html.escape(TEXTS[lang][i]), _DATA) + cells + "</tr>")
    q1, q2 = (_query_text(QUERIES[lang][q]) for q in names)
    head = ["", w["head"], q1, q2]
    k1, k2 = len(got["python"]), len(got["wine"])
    body = _note(w["summary"].format(q1=q1, q2=q2, k1=_num(k1, lang), k2=_num(k2, lang),
                                     r1=w["one"] if k1 == 1 else w["many"], r2=w["one"] if k2 == 1 else w["many"], c1=_num(len(want["python"]), lang),
                                     c2=_num(len(want["wine"]), lang)))
    body += _table(rows, head, (0,))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


_SPLIT_TEXT = {
    "en": {"title": "Your split on the three needs of the lesson",
           "kind": "three lists of document numbers",
           "need": "Need {k}: {need} The query: *{q}*.",
           "parts": ["Found", "Extra", "Missed"],
           "course": "The course's split gives: found {f}, extra {e}, missed {m}.",
           "same": "Your split gives what the course's split gives for all three needs.",
           "differs": "Your split differs from the course's for {k} of the three needs.",
           "gray": "Gray: a relevant document that the search did not return.",
           "blue": "Blue: a result that helps with the need.",
           "purple": "Purple: a result that does not help with the need, and what the course's split gives where yours differs."},
    "nl": {"title": "Je split op de drie behoeften van de les",
           "kind": "drie lijsten met nummers van documenten",
           "need": "Behoefte {k}: {need} De zoekvraag: *{q}*.",
           "parts": ["Gevonden", "Te veel", "Gemist"],
           "course": "De split van de course geeft: gevonden {f}, te veel {e}, gemist {m}.",
           "same": "Je split geeft hetzelfde als de split van de course, bij alle drie de behoeften.",
           "differs": "Je split verschilt van de split van de course bij {k} van de drie behoeften.",
           "gray": "Grijs: een relevant document dat de zoekmachine niet teruggaf.",
           "blue": "Blauw: een resultaat dat helpt bij de behoefte.",
           "purple": "Paars: een resultaat dat niet helpt bij de behoefte, en wat de split van de course geeft waar jouw split verschilt."},
}


def split_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _SPLIT_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    step = _guarded("split", fn, lang)

    def make():
        try:
            return {n: step(results_for(n, lang), NEEDS[n]["relevant"]) for n in NEEDS}
        except _Odd:
            return None

    got = _call(make)
    if got is _SLOW:
        return _empty(w, "slow")
    if got is _STOPPED:
        return _empty(w, "stopped")
    if got is None:
        return _empty(w, "odd")
    body, differs = "", 0
    for k, n in enumerate(NEEDS, 1):
        want = split(results_for(n, lang), NEEDS[n]["relevant"])
        q = _query_text(QUERIES[lang][NEEDS[n]["query"]])
        body += _label(w["need"].format(k=k, need=NEED_TEXT[lang][n], q=q))
        rows = []
        for part, style in zip(range(3), (_CHIP_PRED, _CHIP_LOSS, _CHIP_DATA)):
            chips = "".join(_doc_chip(d, lang, style) for d in got[n][part]) or _DASH
            rows.append("<tr>" + _cell(html.escape(w["parts"][part]), _FAINT) + _cell(chips) + "</tr>")
        body += _table(rows, ["", ""])
        if tuple(got[n]) != tuple(want):
            differs += 1
            body += f'<div style="{_LOSS}">' + html.escape(w["course"].format(f=list(want[0]), e=list(want[1]),
                                                                               m=list(want[2]))) + "</div>"
    body += _note(w["same"] if not differs else w["differs"].format(k=_num(differs, lang)))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


_KIND_TEXT = {
    "en": {"title": "Your kind_of on queries of the lesson",
           "kind": "one of the three kinds",
           "head": ["Query", "The course's reading", "Your kind_of"],
           "summary": "Your kind_of agrees with the course's reading on {k} of the {n} queries.",
           "gray": "Gray: the queries, and the kind of need the course reads in each.",
           "blue": "Blue: what your kind_of gives, where it agrees with the course's reading.",
           "purple": "Purple: where it does not."},
    "nl": {"title": "Je kind_of op zoekvragen van de les",
           "kind": "een van de drie soorten",
           "head": ["Zoekvraag", "Wat de course erin leest", "Je kind_of"],
           "summary": "Je kind_of komt bij {k} van de {n} zoekvragen overeen met wat de course erin leest.",
           "gray": "Grijs: de zoekvragen, en de soort informatiebehoefte die de course in elke zoekvraag leest.",
           "blue": "Blauw: wat je kind_of geeft, waar het overeenkomt met wat de course leest.",
           "purple": "Paars: waar het niet overeenkomt."},
}


def kind_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _KIND_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    step = _guarded("kind_of", fn, lang)

    def make():
        try:
            return kinds(lang, step)
        except _Odd:
            return None

    got = _call(make)
    if got is _SLOW:
        return _empty(w, "slow")
    if got is _STOPPED:
        return _empty(w, "stopped")
    if got is None:
        return _empty(w, "odd")
    names = KIND_NAMES[lang]
    rows = ["<tr>" + _cell(_code(q), _DATA) + _cell(html.escape(names[reading]), _DATA)
            + _cell(_chip(names[mine], _CHIP_PRED if mine == reading else _CHIP_LOSS)) + "</tr>"
            for q, reading, mine in got]
    same = sum(1 for _, reading, mine in got if mine == reading)
    body = _note(w["summary"].format(k=_num(same, lang), n=_num(len(got), lang))) + _table(rows, w["head"])
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


_TRY_TEXT = {
    "en": {"title": "Your query on the course's collection",
           "head": ["", "Document", "Result?"],
           "query": "Your query, as a list of words: {q}",
           "empty": "Type one or more words in the box, and the picture shows which documents contain all of them.",
           "result": "result",
           "lacks": "lacks",
           "summary": "{k} of the {n} documents {verb} every word of your query.",
           "one": "contains", "many": "contain",
           "more": "The search reads the first {n} words.",
           "gray": "Gray: the documents, and the words of your query that a document lacks.",
           "blue": "Blue: a result: a document with every word of your query."},
    "nl": {"title": "Je zoekvraag op de verzameling van de course",
           "head": ["", "Document", "Resultaat?"],
           "query": "Je zoekvraag, als lijst woorden: {q}",
           "empty": "Typ een of meer woorden in het vak, dan laat het plaatje zien welke documenten elk van deze woorden bevatten.",
           "result": "resultaat",
           "lacks": "mist",
           "summary": "{k} van de {n} documenten {verb} elk woord van je zoekvraag.",
           "one": "bevat", "many": "bevatten",
           "more": "De zoekmachine leest de eerste {n} woorden.",
           "gray": "Grijs: de documenten, en de woorden van je zoekvraag die een document mist.",
           "blue": "Blauw: een resultaat: een document met elk woord van je zoekvraag."},
}
TRY_WORDS = 8       # words of the try-it box that the search reads
TRY_CHARS = 300     # characters of the box that the search reads
TRY_START = {"en": "snake", "nl": "slang"}


def try_picture(lang: str, text_: str = "") -> str:
    lang = lang if lang in LANGS else "en"
    w = _TRY_TEXT[lang]
    all_words = words((text_ or "")[:TRY_CHARS])
    query = all_words[:TRY_WORDS]
    if not query:
        return yours.figure(w["title"], _note(w["empty"]), alt=w["empty"])
    rows = []
    for i, doc in enumerate(DOCS[lang]):
        if matches(query, doc):
            mark = _chip(w["result"])
        else:
            lacking = []
            for word in query:
                if word not in doc and word not in lacking:
                    lacking.append(word)
            mark = f'<span style="{_UI}">{html.escape(w["lacks"])}</span> ' + "".join(_chip(x, _CHIP_DATA) for x in lacking)
        rows.append("<tr>" + _cell(str(i), _FAINT, _NUMBER) + _cell(html.escape(TEXTS[lang][i]), _DATA) + _cell(mark)
                    + "</tr>")
    k = len(search(query, DOCS[lang]))
    body = f'<div style="{_DATA}">{html.escape(w["query"].format(q=repr(query)))}</div>'
    body += _note(w["summary"].format(k=_num(k, lang), n=_num(len(DOCS[lang]), lang), verb=w["one"] if k == 1 else w["many"]))
    body += _table(rows, w["head"], (0,))
    if len(all_words) > TRY_WORDS:
        body += _note(w["more"].format(n=_num(TRY_WORDS, lang)))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------------

_Val = S._Val
_DOTS = S._DOTS
_SAME_TEXTS = S._SAME_TEXTS


def _show(v):
    return S._show(v)


def _written(v):
    return S._written(v)


def _doc(lang: str, i: int) -> _Val:
    return _Val(f'C.DOCS["{lang}"][{i}]', DOCS[lang][i])


def _clues(lang: str) -> _Val:
    return _Val(f'C.CLUES["{lang}"]', CLUES[lang])


_MA_HEAD = "def matches(query, document):\n    for word in query:\n"
_MA_END = "            return False\n    return True\n"
_MA = _MA_HEAD + "        if word not in document:\n" + _MA_END
_MA_START = _MA_HEAD + "        if ...:\n" + _MA_END
_MA_ANY = _MA_HEAD + "        if word in document:\n            return True\n    return False\n"
_MA_FIRST = _MA_HEAD + "        if word not in document:\n            return False\n        return True\n"
_MA_INVERTED = _MA_HEAD + "        if word in document:\n" + _MA_END
_MA_WHOLE = _MA_HEAD + "        if query not in document:\n" + _MA_END
_MA_NO_TRUE = _MA_HEAD + "        if word not in document:\n            return False\n"

_SP_HEAD = ("def split(results, relevant):\n    found = []\n    extra = []\n    for doc in results:\n"
            "        if doc in relevant:\n            found.append(doc)\n        else:\n            extra.append(doc)\n")
_SP_RET = "    return found, extra, missed\n"
_SP = _SP_HEAD + "    missed = [doc for doc in relevant if doc not in results]\n" + _SP_RET
_SP_START = _SP_HEAD + "    missed = ...\n" + _SP_RET
_SP_ALL = _SP_HEAD + "    missed = relevant\n" + _SP_RET
_SP_EXTRA = _SP_HEAD + "    missed = [doc for doc in results if doc not in relevant]\n" + _SP_RET
_SP_FOUND = _SP_HEAD + "    missed = [doc for doc in relevant if doc in results]\n" + _SP_RET
_SP_SET = _SP_HEAD + "    missed = set(relevant) - set(results)\n" + _SP_RET

_KI_HEAD = "def kind_of(query, clues):\n    for kind in clues:\n        for word in query:\n"
_KI_END = "                return kind\n    return \"informational\"\n"
_KI = _KI_HEAD + "            if word in clues[kind]:\n" + _KI_END
_KI_START = _KI_HEAD + "            if ...:\n" + _KI_END
_KI_WORDS_FIRST = ("def kind_of(query, clues):\n    for word in query:\n        for kind in clues:\n"
                   "            if word in clues[kind]:\n" + _KI_END)
_KI_NAME = _KI_HEAD + "            if word in kind:\n" + _KI_END
_KI_EQUAL = _KI_HEAD + "            if word == clues[kind]:\n" + _KI_END
_KI_NO_DEFAULT = _KI_HEAD + "            if word in clues[kind]:\n                return kind\n"
_KI_INSIDE = _KI_HEAD + "            if word in clues[kind]:\n                return kind\n        return \"informational\"\n"
_KI_ELSE = _KI_HEAD + "            if word in clues[kind]:\n                return kind\n            else:\n" \
    "                return \"informational\"\n"

# The small cases the exercise texts show; the same in both lessons.
TOY_MATCH = ((["python"], ("en", 1)), (["rode", "wijn", "hart"], ("nl", 7)))
TOY_SPLIT = (([0, 2, 3, 4, 5], [1, 3]), ([6, 7, 8], [6, 9]))
TOY_KIND = ((["facebook", "login"], "en"), (["hoe", "leer", "ik", "python"], "nl"))
# Two kinds with one clue each, in both orders, to show that the order of clues decides.
TOY_CLUES = {"navigational": ["home"], "transactional": ["buy"]}
TOY_CLUES_BACK = {"transactional": ["buy"], "navigational": ["home"]}


def _is_set_split(got, text_) -> bool:
    return isinstance(got, tuple) and len(got) == 3 and isinstance(got[2], set)


def _never(got, text_) -> bool:
    return got == "informational"


REQUIRED = ["matches", "split", "kind_of", "judge"]

EXERCISES = {
    "matches": FunctionBox(
        name="matches",
        picture=match_picture,
        function="matches",
        start=_MA_START,
        solution=_MA,
        show=_show,
        visible=[(_written(q), _doc(*where)) for q, where in TOY_MATCH],
        hidden=[(_written(["red", "wine", "heart"]), _doc("en", 10)),
                (_written(["heart", "wine"]), _doc("en", 10)),
                (_written(["python"]), _doc("en", 0)),
                (_written(["heart", "wine", "red"]), _doc("en", 8)),
                (_written(["rode", "wijn", "hart"]), _doc("nl", 9)),
                (_written(["rode", "wijn", "hart"]), _doc("nl", 6)),
                (_written(["red", "red"]), _written(["red", "wine"])),
                (_written(["python"]), _written([]))],
        first=[_DOTS],
        mistakes=[
            Mistake("any", wrong=_MA_ANY,
                    en="`{call}` gives {got}, and it should give {want}: your function finds a result as soon as one word of the query is in the document. Should a result have one word of the query, or every word?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie vindt een resultaat zodra één woord van de zoekvraag in het document staat. Moet een resultaat één woord van de zoekvraag hebben, of elk woord?"),
            Mistake("first", wrong=_MA_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: your function decides after the first word of the query, because `return True` stands inside the loop. Where does `return True` belong, so that the function looks at every word first?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie beslist na het eerste woord van de zoekvraag, want `return True` staat in de loop. Waar hoort `return True`, zodat de functie eerst naar elk woord kijkt?"),
            Mistake("inverted", wrong=_MA_INVERTED,
                    en="`{call}` gives {got}, and it should give {want}: your function stops with `False` when a word is in the document. When can the function know for sure that a document is not a result?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie stopt met `False` als een woord in het document staat. Wanneer weet de functie zeker dat een document geen resultaat is?"),
            Mistake("whole", wrong=_MA_WHOLE,
                    en="`{call}` gives {got}, and it should give {want}: your test looks for the whole list `query` among the words of the document, and a list is never one of its words. Which name holds one word of the query?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je test zoekt de hele lijst `query` tussen de woorden van het document, en een lijst is nooit een van de woorden. Welke naam bevat één woord van de zoekvraag?"),
            Mistake("no-true", wrong=_MA_NO_TRUE,
                    en="`{call}` gives `None`, and it should give {want}: when every word is in the document, your function gets to its end without a `return`. What should it give back there?",
                    nl="`{call}` geeft `None`, en het moet {want} zijn: als elk woord in het document staat, komt je functie aan het eind zonder `return`. Wat moet de functie daar teruggeven?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the test that makes the document not a result: a word of the query that the document lacks.",
                      "nl": "Vervang de `...` door de test die van het document geen resultaat maakt: een woord van de zoekvraag dat het document mist."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function give `False` as soon as a word of the query is missing from the document, and `True` when none is?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Geeft je functie `False` zodra een woord van de zoekvraag in het document ontbreekt, en `True` als er geen woord ontbreekt?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does your function give `False` as soon as one word is missing, also for a query with a word twice or an empty document?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Geeft je functie `False` zodra er één woord ontbreekt, ook bij een zoekvraag met twee keer hetzelfde woord of een leeg document?"},
            "correct": {"en": "Right: your function decides what the simplest search engine returns. In the picture below, it runs both queries of the lesson on the course's collection: which documents does *python* not find, although they are about pythons?",
                        "nl": "Goed: je functie bepaalt wat de eenvoudigste zoekmachine teruggeeft. In het plaatje hieronder draaien beide zoekvragen van de les ermee op de verzameling van de course: welke documenten over pythons vindt *python* niet?"},
        },
    ),
    "split": FunctionBox(
        name="split",
        picture=split_picture,
        function="split",
        start=_SP_START,
        solution=_SP,
        show=_show,
        visible=[tuple(_written(x) for x in case) for case in TOY_SPLIT],
        hidden=[(_written([0, 2, 3, 4, 5]), _written([0])),
                (_written([5, 3, 9]), _written([9, 4, 3, 1])),
                (_written([]), _written([1, 2])),
                (_written([1, 2]), _written([])),
                (_written([2, 1]), _written([1, 2]))],
        first=[_DOTS],
        mistakes=[
            Mistake("all-relevant", wrong=_SP_ALL,
                    en="`{call}` gives {got}, and it should give {want}: your `missed` holds every relevant document, also the ones the search did return. Which relevant documents did the search miss?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je `missed` bevat elk relevant document, ook de documenten die de zoekmachine wel teruggaf. Welke relevante documenten miste de zoekmachine?"),
            Mistake("extra", wrong=_SP_EXTRA,
                    en="`{call}` gives {got}, and it should give {want}: your `missed` is the same as `extra`, the results that do not help. Are the missed documents in `results` or in `relevant`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je `missed` is hetzelfde als `extra`, de resultaten die niet helpen. Staan de gemiste documenten in `results` of in `relevant`?"),
            Mistake("found", wrong=_SP_FOUND,
                    en="`{call}` gives {got}, and it should give {want}: your `missed` holds the relevant documents that are results, the same as `found`. Should a missed document be among the results?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je `missed` bevat de relevante documenten die ook een resultaat zijn, net als `found`. Hoort een gemist document bij de resultaten?"),
            Mistake("set", matches=_is_set_split,
                    en="`{call}` gives {got}, and it should give {want}: your `missed` is a set, and `found` and `extra` are lists. Can you make `missed` a list too, in the order of `relevant`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je `missed` is een set, en `found` en `extra` zijn lijsten. Kun je van `missed` ook een lijst maken, in de volgorde van `relevant`?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with a list of the relevant documents that the search did not return.",
                      "nl": "Vervang de `...` door een lijst met de relevante documenten die de zoekmachine niet teruggaf."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does `missed` hold each document of `relevant` that is not in `results`, in the order of `relevant`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Bevat `missed` elk document van `relevant` dat niet in `results` staat, in de volgorde van `relevant`?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does `missed` keep the order of `relevant`, and does your function also work without results, or without relevant documents?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Houdt `missed` de volgorde van `relevant` aan, en werkt je functie ook zonder resultaten, of zonder relevante documenten?"},
            "correct": {"en": "Right: your function judges results the way people judge a search engine. In the picture below, it splits the results of the three needs of the lesson: which need has the same results as another, but other documents that help?",
                        "nl": "Goed: je functie beoordeelt resultaten zoals mensen een zoekmachine beoordelen. In het plaatje hieronder verdeelt de functie de resultaten bij de drie behoeften van de les: welke behoefte heeft dezelfde resultaten als een andere, maar andere documenten die helpen?"},
        },
    ),
    "kind_of": FunctionBox(
        name="kind_of",
        picture=kind_picture,
        function="kind_of",
        start=_KI_START,
        solution=_KI,
        show=_show,
        visible=[(_written(q), _clues(lang)) for q, lang in TOY_KIND],
        hidden=[(_written(["buy", "a", "ball", "python"]), _clues("en")),
                (_written(["python", "downloaden"]), _clues("nl")),
                (_written(["buy", "home"]), _written(TOY_CLUES)),
                (_written(["buy", "home"]), _written(TOY_CLUES_BACK)),
                (_written(["a", "home"]), _written(TOY_CLUES)),
                (_written([]), _clues("en")),
                (_written(["python"]), _written({}))],
        first=[_DOTS],
        mistakes=[
            Mistake("words-first", wrong=_KI_WORDS_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: your function takes the clue of the first word that has one, and the kinds should be tested in the order of `clues`. Which loop should be the outer one?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie neemt de aanwijzing van het eerste woord met een aanwijzing, en de soorten moeten getest worden in de volgorde van `clues`. Welke loop hoort buiten te staan?"),
            Mistake("name", wrong=_KI_NAME,
                    en="`{call}` gives {got}, and it should give {want}: your test looks for the word in `kind`, the name of the kind as a string, not in its list of clue words. Where is that list?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je test zoekt het woord in `kind`, de naam van de soort als string, niet in de lijst met aanwijzingen. Waar staat de lijst met aanwijzingen van de soort?"),
            Mistake("no-default", wrong=_KI_NO_DEFAULT,
                    en="`{call}` gives `None`, and it should give {want}: a query without a clue word gets to the end of your function without a `return`. What kind is a query without a clue?",
                    nl="`{call}` geeft `None`, en het moet {want} zijn: een zoekvraag zonder aanwijzing komt aan het eind van je functie zonder `return`. Welke soort is een zoekvraag zonder aanwijzing?"),
            Mistake("inside", wrong=_KI_INSIDE,
                    en="`{call}` gives {got}, and it should give {want}: your function gives `\"informational\"` after the first kind, because `return \"informational\"` stands inside the outer loop. Where does it belong, so that the function tests every kind first?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie geeft `\"informational\"` na de eerste soort, want `return \"informational\"` staat in de buitenste `for`. Waar hoort deze regel, zodat de functie eerst elke soort test?"),
            Mistake("never", matches=_never,
                    en="`{call}` gives {got}, and it should give {want}: the query holds a clue word, and your function does not find it. Does your test look up the word in the list of clue words of the kind, and does the function test every word and every kind before it gives `\"informational\"`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de zoekvraag bevat een aanwijzing, en je functie vindt deze niet. Zoekt je test het woord op in de lijst met aanwijzingen van de soort, en geeft de functie pas `\"informational\"` als elk woord en elke soort getest is?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the test for a clue: the word is one of the clue words of the kind.",
                      "nl": "Vervang de `...` door de test voor een aanwijzing: het woord is een van de aanwijzingen van de soort."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function give the first kind, in the order of `clues`, with a clue word in the query, and `\"informational\"` when there is none?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Geeft je functie de eerste soort, in de volgorde van `clues`, met een aanwijzing in de zoekvraag, en `\"informational\"` als er geen is?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does the order of `clues` decide when a query has clues of two kinds?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Beslist de volgorde van `clues` als een zoekvraag aanwijzingen van twee soorten heeft?"},
            "correct": {"en": "Right: your function guesses the kind of need from the words, as rules can. In the picture below, it runs on queries of the lesson: where do the words point one way and the need another?",
                        "nl": "Goed: je functie raadt de soort informatiebehoefte uit de woorden, zoals regels dat kunnen. In het plaatje hieronder draait de functie op zoekvragen van de les: waar wijzen de woorden de ene kant op, en de behoefte de andere?"},
        },
    ),
}

# The sort: TREC-8 ad hoc topic 427, UV damage, eyes, with the course's own documents.
SORT_QUERY = {"en": ["uv", "damage", "eyes"], "nl": ["uv", "schade", "ogen"]}
EXERCISES["judge"] = Sort(
    name="judge",
    groups={
        "found": {"en": "Found: relevant, with every word of the query", "nl": "Gevonden: relevant, met elk woord van de zoekvraag"},
        "missed": {"en": "Missed: relevant, but a word of the query is not in it", "nl": "Gemist: relevant, maar een woord van de zoekvraag ontbreekt"},
        "extra": {"en": "Extra: every word of the query, but not relevant", "nl": "Te veel: elk woord van de zoekvraag, maar niet relevant"},
    },
    items={
        "years": {"en": "*Years of strong sun can damage the eyes: UV light is one cause of cataracts.*",
                  "nl": "*Jaren felle zon doen schade aan de ogen: uv is een van de oorzaken van staar.*"},
        "school": {"en": "*A school program against UV damage to the eyes: hats and sunglasses on the playground.*",
                   "nl": "*Een schoolprogramma tegen schade door uv aan de ogen: petten en zonnebrillen op het schoolplein.*"},
        "treat": {"en": "*UV damage to the eyes, such as cataracts, can be treated: an overview of the options.*",
                  "nl": "*Schade aan de ogen door uv, zoals staar, is te behandelen: een overzicht van de mogelijkheden.*"},
        "lens": {"en": "*A new treatment for cataracts: the cloudy lens is replaced in twenty minutes.*",
                 "nl": "*Een nieuwe behandeling van staar: de troebele lens wordt in twintig minuten vervangen.*"},
        "melanoma": {"en": "*Ocular melanoma: how doctors treat this rare cancer of the eye.*",
                     "nl": "*Oogmelanoom: hoe artsen deze zeldzame kanker van het oog behandelen.*"},
        "snow": {"en": "*Snow blindness: after a day on a glacier without sunglasses, the eyes burn from ultraviolet light.*",
                 "nl": "*Sneeuwblindheid: na een dag op een gletsjer zonder zonnebril branden de ogen door ultraviolet licht.*"},
        "laser": {"en": "*Laser pointers can damage the eyes in seconds, much faster than UV light from the sun.*",
                  "nl": "*Een laserpen kan in enkele seconden schade toebrengen aan de ogen, veel sneller dan uv uit het zonlicht.*"},
        "skin": {"en": "*Sunscreen protects the skin from UV damage; keep it away from the eyes.*",
                 "nl": "*Zonnebrandcrème beschermt de huid tegen schade door uv; houd de crème weg van de ogen.*"},
    },
    solution={"years": "found", "school": "found", "treat": "found", "lens": "missed", "melanoma": "missed",
              "snow": "missed", "laser": "extra", "skin": "extra"},
    nudge={
        "years": {"en": "Which words of the query are in it, and is the damage it describes damage from UV light of the sun?",
                  "nl": "Welke woorden van de zoekvraag staan erin, en komt de schade die het beschrijft door uv-licht van de zon?"},
        "school": {"en": "The narrative counts programs that help prevent damage. Which words of the query are in it?",
                   "nl": "De toelichting telt voorlichting om schade te voorkomen mee. Welke woorden van de zoekvraag staan erin?"},
        "treat": {"en": "The narrative counts treatments for the damage. Are *uv*, *damage* and *eyes* all in it?",
                  "nl": "De toelichting telt behandelingen van de schade mee. Staan *uv*, *schade* en *ogen* er alle drie in?"},
        "lens": {"en": "The narrative counts treatment of cataracts as relevant even without a cause. Is any word of the query in it?",
                 "nl": "De toelichting telt een behandeling van staar als relevant, ook zonder oorzaak. Staat er een woord van de zoekvraag in?"},
        "melanoma": {"en": "The narrative counts the treatment of ocular melanoma, even without a cause. Is *eye* the same word as *eyes*?",
                     "nl": "De toelichting telt de behandeling van oogmelanoom mee, ook zonder oorzaak. Is *oog* hetzelfde woord als *ogen*?"},
        "snow": {"en": "Ultraviolet light is UV light. Is the word *uv* in it, and the word *damage*?",
                 "nl": "Ultraviolet licht is uv-licht. Staat het woord *uv* erin, en het woord *schade*?"},
        "laser": {"en": "All three words are in it. What does the narrative say about damage from lasers?",
                  "nl": "Alle drie de woorden staan erin. Wat zegt de toelichting over schade door lasers?"},
        "skin": {"en": "All three words are in it. Is it about damage that UV light does to the eyes?",
                 "nl": "Alle drie de woorden staan erin. Gaat het over schade die uv-licht aan de ogen doet?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the groups again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de groepen opnieuw uit."},
        "empty": {"en": "Sort each document into the group that fits.",
                  "nl": "Deel elk document in bij de groep die klopt."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "{item} belongs with another group.",
                  "nl": "{item} hoort bij een andere groep."},
        "correct": {"en": "All {n} are right. A search for *uv damage eyes* returns five of these eight documents, and only three of them help; three relevant documents use other words, such as *ultraviolet* and *eye*. Judging takes the need, here written out in the narrative: the words of the query alone do not tell.",
                    "nl": "Alle {n} kloppen. Een zoekvraag *uv schade ogen* geeft vijf van deze acht documenten terug, en maar drie daarvan helpen; drie relevante documenten gebruiken andere woorden, zoals *ultraviolet* en *oog*. Voor het oordeel is de behoefte nodig, hier uitgeschreven in de toelichting: de woorden van de zoekvraag zeggen het niet."},
    },
)

# What the tests feed each check, and what it must say: (exercise, code or choices, state, known mistake).
CASES = [
    ("matches", _MA_START, "empty", None),
    ("matches", _MA, "correct", None),
    ("matches", _MA_HEAD + "        if word not in ...:\n" + _MA_END, "wrong", "dots-left"),
    ("matches", _MA_ANY, "wrong", "any"),
    ("matches", _MA_FIRST, "wrong", "first"),
    ("matches", _MA_INVERTED, "wrong", "inverted"),
    ("matches", _MA_WHOLE, "wrong", "whole"),
    ("matches", _MA_NO_TRUE, "wrong", "no-true"),
    ("matches", "def matches(query, document):\n    return all(word in document for word in query)\n", "correct", None),
    ("matches", "def matches(query, document):\n    return set(query) <= set(document)\n", "correct", None),
    ("matches", "def matches(query, document):\n    found = 0\n    for word in query:\n        if word in document:\n"
     "            found = found + 1\n    return found == len(query)\n", "correct", None),
    ("split", _SP_START, "empty", None),
    ("split", _SP, "correct", None),
    ("split", _SP_HEAD + "    missed = [doc for doc in ... if doc not in results]\n" + _SP_RET, "wrong", "dots-left"),
    ("split", _SP_ALL, "wrong", "all-relevant"),
    ("split", _SP_EXTRA, "wrong", "extra"),
    ("split", _SP_FOUND, "wrong", "found"),
    ("split", _SP_SET, "wrong", "set"),
    ("split", _SP_HEAD + "    missed = [doc for doc in relevant if doc not in found]\n" + _SP_RET, "correct", None),
    ("split", _SP_HEAD + "    missed = []\n    for doc in relevant:\n        if doc not in results:\n"
     "            missed.append(doc)\n" + _SP_RET, "correct", None),
    ("kind_of", _KI_START, "empty", None),
    ("kind_of", _KI, "correct", None),
    ("kind_of", _KI_HEAD + "            if word in ...:\n" + _KI_END, "wrong", "dots-left"),
    ("kind_of", _KI_WORDS_FIRST, "wrong", "words-first"),
    ("kind_of", _KI_NAME, "wrong", "name"),
    ("kind_of", _KI_NO_DEFAULT, "wrong", "no-default"),
    ("kind_of", _KI_EQUAL, "wrong", "never"),
    ("kind_of", _KI_INSIDE, "wrong", "inside"),
    ("kind_of", _KI_ELSE, "wrong", "never"),
    ("kind_of", "def kind_of(query, clues):\n    for kind, clue_words in clues.items():\n        for word in query:\n"
     "            if word in clue_words:\n                return kind\n    return \"informational\"\n", "correct", None),
    ("judge", {}, "empty", None),
    ("judge", dict(EXERCISES["judge"].solution), "correct", None),
    ("judge", {"years": "found", "laser": "extra"}, "wrong", "unfinished"),
] + [("judge", {item: ("extra" if group != "extra" else "found")}, "wrong", f"wrong:{item}")
     for item, group in EXERCISES["judge"].solution.items()]
