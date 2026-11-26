# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ir-vertical-search: a search engine for one domain, here the course's own site for second-hand
things. Each listing has fields: a title and a text, which BM25F weighs before it saturates the count of a word
(combined_count), as in Robertson and Zaragoza, The Probabilistic Relevance Framework: BM25 and Beyond (2009), section
3.6; and a price, the days since it was placed and the distance to the searcher, which a filter uses to keep or drop a
listing (keep), and a decay uses to lower its score smoothly (decay), as the exponential decay of Elasticsearch's
function_score query does with decay 0.5. The sort asks which domain needs what: news, products, places nearby, the
documents of a company, or law (Manning, Raghavan and Schütze, Introduction to Information Retrieval, 2008, sections
1.4, 4.6 and 6.1).

The 24 listings are the course's, the same in both languages. The guess uses the 2,067 paragraphs and the questions of
XQuAD (everything_ai.xquad), with the title of the Wikipedia article of each paragraph as a field. Words are cut as the
lesson ir-what-is-search cuts them, the IDF is log10(N / df) over whole listings, and BM25 has k1 = 1.2 and b = 0.75, as
in the lesson ir-bm25. Every number in the text and the pictures is computed by the course
(tests/test_ir_vertical_search.py computes them again)."""
import html
import importlib.util
import math

import numpy as np

from everything_ai import paths, xquad, yours
from everything_ai.check import FunctionBox, Mistake, Sort, kept

LANGS = ("en", "nl")


def _bm25():
    spec = importlib.util.spec_from_file_location("everything_ai_checks_ir_bm25_for_vertical_search",
                                                  paths.LESSONS / "ir-bm25" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _bm25()             # BM25: the term weight, the settings, and the paragraphs and questions of XQuAD
V = M.V                 # the vector space model, for the parts of its pictures
S = M.S                 # the parts of the pictures and the exercises
words = M.words         # a text as a list of words in small letters
term_weight = M.term_weight
K1, B = M.K1, M.B       # k1 = 1.2 and b = 0.75, the usual settings of BM25


# --- the course's solutions, the same code as the exercises -------------------------------------------------------------------

def combined_count(word, fields, weights):
    """The count of BM25F: for each field of a document, the count of the word there times the weight of the field,
    added up. `fields` holds the words of each field, `weights` the weight of each field."""
    count = 0
    for name in fields:
        count = count + weights[name] * fields[name].count(word)
    return count


def keep(listing, max_price, max_km):
    """Whether a listing passes the two filters: a price of at most max_price and a distance of at most max_km."""
    return listing["price"] <= max_price and listing["km"] <= max_km


def decay(distance, half):
    """The factor a score is multiplied by at a distance from the best value, such as an age in days or a distance in
    km: 1 at distance 0, a half at distance `half`, a quarter at twice `half`. Works on numbers and on NumPy arrays."""
    return 0.5 ** (distance / half)


def combined_length(fields, weights):
    """The length of BM25F: for each field, its number of words times the weight of the field, added up."""
    total = 0
    for name in fields:
        total = total + weights[name] * len(fields[name])
    return total


# --- the listings ---------------------------------------------------------------------------------------------------------------

# The course's site for second-hand things: the same 24 listings in both languages, in the same order. Each has a title
# and a text in both languages, a price in euros, the days since it was placed, and its distance to the searcher in km.
_LISTINGS = (
    ("Tent for four people", "Tent voor vier personen",
     "Used for two summers. All poles and pegs are there, and everything packs into a small bag.",
     "Twee zomers gebruikt. Alle stokken en haringen zijn er, en alles past in een kleine tas.", 85, 3, 12),
    ("Sleeping bag", "Slaapzak",
     "Warm sleeping bag for cold nights in the tent. Rolls up small, so there is room left in the tent.",
     "Warme slaapzak voor koude nachten in de tent. Rolt klein op, zodat er ruimte overblijft in de tent.", 25, 8, 20),
    ("New tent for two people", "Nieuwe tent voor twee personen",
     "Never used, still in the box.", "Nooit gebruikt, zit nog in de doos.", 140, 2, 30),
    ("Camping stove", "Kampeerbrander",
     "Small gas stove with two burners. Never cook inside the tent.",
     "Kleine gasbrander met twee pitten. Kook nooit binnen in de tent.", 15, 6, 9),
    ("Large tent for six people", "Grote tent voor zes personen",
     "Room to stand up, two sleeping cabins and a porch. Weighs 18 kg.",
     "Je kunt erin staan, met twee slaapcabines en een voortent. Weegt 18 kg.", 220, 20, 3),
    ("Tent for three people", "Tent voor drie personen",
     "Light and quick to put up. A small tear in the door, easy to mend.",
     "Licht en snel opgezet. Een klein scheurtje in de deur, makkelijk te maken.", 60, 90, 8),
    ("Tent for two people", "Tent voor twee personen",
     "As good as new, used for one weekend.", "Zo goed als nieuw, één weekend gebruikt.", 45, 5, 140),
    ("Air mattress for two people", "Luchtbed voor twee personen",
     "Fits in a tent for two people. Comes with a pump.", "Past in een tent voor twee personen. Met pomp.", 20, 10, 18),
    ("Camping lamp", "Kampeerlamp",
     "Bright lamp on batteries, with a hook to hang it in the tent.",
     "Felle lamp op batterijen, met een haakje om op te hangen in de tent.", 8, 4, 22),
    ("Folding chair", "Klapstoel",
     "Light chair for the beach or the campsite.", "Lichte stoel voor het strand of de camping.", 12, 7, 6),
    ("Children's bike", "Kinderfiets",
     "Bike for ages four to six, with training wheels.", "Fiets voor vier tot zes jaar, met zijwieltjes.", 50, 12, 4),
    ("Desk lamp", "Bureaulamp",
     "Desk lamp with an arm that bends.", "Bureaulamp met een arm om te buigen.", 15, 8, 15),
    ("Dining table with four chairs", "Eettafel met vier stoelen",
     "Oak table, 180 by 90 cm. The chairs have new seats.",
     "Eikenhouten tafel, 180 bij 90 cm. De stoelen hebben nieuwe zittingen.", 150, 15, 25),
    ("Road bike", "Racefiets",
     "Aluminum frame, 56 cm, new tires.", "Aluminium frame, 56 cm, nieuwe banden.", 300, 2, 35),
    ("Book about camping in France", "Boek over kamperen in Frankrijk",
     "Campsites, walks and recipes to cook at the tent.", "Campings, wandelingen en recepten om bij de tent te koken.", 5, 30, 10),
    ("Backpack, 60 liters", "Rugzak van 60 liter",
     "Holds a small tent, a sleeping bag and food for a week.",
     "Er passen een kleine tent, een slaapzak en eten voor een week in.", 40, 9, 20),
    ("Steel pegs, 20 pieces", "Stalen haringen, 20 stuks",
     "For a tent on hard ground.", "Voor een tent op harde grond.", 6, 3, 11),
    ("Rain jacket", "Regenjas",
     "Waterproof jacket, size M, with a hood.", "Waterdichte jas, maat M, met capuchon.", 30, 6, 7),
    ("Lawn mower", "Grasmaaier",
     "Electric mower with a cable of 20 meters. Cuts well.", "Elektrische maaier met een kabel van 20 meter. Maait goed.", 70, 14, 16),
    ("Guitar for beginners", "Gitaar voor beginners",
     "Acoustic guitar with a bag and new strings.", "Akoestische gitaar met een tas en nieuwe snaren.", 55, 4, 28),
    ("Puzzle of 1,000 pieces", "Puzzel van 1000 stukjes",
     "Complete, only done once.", "Compleet, één keer gelegd.", 4, 21, 9),
    ("Coffee maker", "Koffiezetapparaat",
     "Makes ten cups at once. Works well.", "Zet tien kopjes tegelijk. Werkt goed.", 18, 11, 13),
    ("Bookcase", "Boekenkast",
     "White bookcase with five shelves, 200 cm high.", "Witte boekenkast met vijf planken, 200 cm hoog.", 35, 25, 19),
    ("Winter coat, size L", "Winterjas, maat L",
     "Warm coat with a hood, worn for one winter.", "Warme jas met capuchon, één winter gedragen.", 45, 40, 24),
)
N = len(_LISTINGS)
QUERY = "tent"                          # the query of the lesson, the same word in both languages
TENTS = (0, 2, 4, 5, 6)                 # the listings that are tents
WANTED = 0                              # the tent that fits the search: Tent for four people
WEIGHTS = {"title": 3, "text": 1}       # the weights of the lesson: a word in the title counts three times
PLAIN = {"title": 1, "text": 1}         # title and text as one list of words: BM25 itself
MAX_PRICE, MAX_KM = 100, 50             # the filters of the lesson
HALF_DAYS, HALF_KM = 14, 25             # the decay of the lesson: half the score at 14 days, and at 25 km


def listing(lang: str, i: int) -> dict:
    """Listing i as a dictionary: its title and text in the language, its price in euros, the days since it was placed,
    and its distance in km."""
    title_en, title_nl, text_en, text_nl, price, days, km = _LISTINGS[i]
    return {"title": title_en if lang == "en" else title_nl, "text": text_en if lang == "en" else text_nl,
            "price": price, "days": days, "km": km}


def listings(lang: str) -> list:
    return [listing(lang, i) for i in range(N)]


def fields(lang: str, i: int) -> dict:
    """The text fields of listing i as lists of words: {"title": [...], "text": [...]}."""
    item = listing(lang, i)
    return {"title": words(item["title"]), "text": words(item["text"])}


def _idf(lang: str) -> dict:
    """The IDF of each word, log10(N / df), over whole listings: a word counts once for a listing, whatever its field."""
    df = {}
    for i in range(N):
        for w in set(words(listing(lang, i)["title"]) + words(listing(lang, i)["text"])):
            df[w] = df.get(w, 0) + 1
    return {w: math.log10(N / c) for w, c in df.items()}


IDF = {lang: _idf(lang) for lang in LANGS}


def average_length(lang: str, weights: dict) -> float:
    """The average length of the listings with these weights, as combined_length gives it."""
    return sum(combined_length(fields(lang, i), weights) for i in range(N)) / N


def scores(lang: str, weights: dict, query=(QUERY,), counts=None) -> list:
    """The BM25F score of each listing for the query (a list of words, each once): for each word of the query, its IDF
    times the term weight of its combined count, with the combined length against the average. `counts`, for a query
    of one word, gives the combined count of each listing instead of the course's combined_count: the learner's."""
    average = average_length(lang, weights)
    out = []
    for i in range(N):
        f = fields(lang, i)
        length = combined_length(f, weights)
        score = 0.0
        for word in dict.fromkeys(query):
            c = counts[i] if counts is not None else combined_count(word, f, weights)
            if c:
                score += IDF[lang].get(word, 0.0) * term_weight(c, length, average, K1, B)
        out.append(float(score))
    return out


def final_scores(lang: str, weights: dict = WEIGHTS, max_price=MAX_PRICE, max_km=MAX_KM, half_days=HALF_DAYS,
                 half_km=HALF_KM, kept_fn=keep, decay_fn=decay) -> dict:
    """The score of each listing that passes the filters: its BM25F score times the decay of its days and of its km.
    max_price None leaves the filters out, and half_days None leaves the decay out."""
    base = scores(lang, weights)
    out = {}
    for i, s in enumerate(base):
        item = listing(lang, i)
        if max_price is not None and not kept_fn(item, max_price, max_km):
            continue
        if half_days is not None:
            s = s * decay_fn(item["days"], half_days) * decay_fn(item["km"], half_km)
        out[i] = float(s)
    return out


def ranking(lang: str, step: str) -> list:
    """The listings for the query tent, as (place in _LISTINGS, score), best first; ties keep the order of the listings.
    step is "plain" (BM25 on title and text as one), "fields" (BM25F with the title three times), "filters" (and the
    two filters) or "decay" (and the decay of the days and the km)."""
    if step == "plain":
        got = dict(enumerate(scores(lang, PLAIN)))
    elif step == "fields":
        got = dict(enumerate(scores(lang, WEIGHTS)))
    elif step == "filters":
        got = final_scores(lang, half_days=None)
    else:
        got = final_scores(lang)
    return sorted(((i, s) for i, s in got.items() if s > 0), key=lambda p: (-p[1], p[0]))


def place_of(lang: str, step: str, i: int) -> int:
    """The place of listing i in the ranking of a step, counted from 1."""
    return [j for j, _ in ranking(lang, step)].index(i) + 1


STEPS = ("plain", "fields", "filters", "decay")


# --- the guess: the title of the article as a field of each paragraph -------------------------------------------------------------

TITLE_WEIGHT = 2   # each word of the title counts twice


def title_field(lang: str) -> dict:
    """BM25 on the 2,067 paragraphs of XQuAD, with and without the title of the article of each paragraph as a field
    with weight 2 (the title written into the paragraph twice, as Robertson and Zaragoza describe BM25F with whole
    weights): the share of the questions whose own paragraph comes first, the share whose first paragraph is in the
    article of the own paragraph, the share of the questions that hold a word of the title of their article, and the
    average number of paragraphs of an article."""
    def make():
        body = M.I._paragraph_words(lang)
        paras = xquad.paragraphs(lang)
        qs = xquad.questions(lang)
        asked = [words(q.question) for q in qs]
        own = np.array([q.paragraph for q in qs])
        article = np.array([p.article for p in paras])
        titles = [words(p.title) for p in paras]
        wanted = np.zeros((len(qs), len(paras)), dtype=bool)
        wanted[np.arange(len(qs)), own] = True
        out = {}
        for weight in (0, TITLE_WEIGHT):
            docs = [list(b) + t * weight for b, t in zip(body, titles)]
            coll = M._collection(docs)
            sc = M.bm25_scores(coll, M._queries(coll, asked), K1, B)
            out[weight] = {"hit": float(M._shared_first(sc, wanted).mean()),
                           "article": float((article[sc.argmax(axis=1)] == article[own]).mean())}
        named = [bool(set(a) & set(titles[o])) for a, o in zip(asked, own)]
        out["named"] = float(np.mean(named))
        out["per_article"] = len(paras) / (int(article.max()) + 1)
        return out
    return kept(f"ir-vertical-search:title-field:{lang}", make)


# --- the pictures ----------------------------------------------------------------------------------------------------------

_UI, _LABEL, _CELL, _HEAD, _NUMBER = S._UI, S._LABEL, S._CELL, S._HEAD, S._NUMBER
_DATA, _FAINT, _PRED, _LOSS = S._DATA, S._FAINT, S._PRED, S._LOSS
_CHIP, _CHIP_PRED, _CHIP_DATA, _CHIP_LOSS = S._CHIP, S._CHIP_PRED, S._CHIP_DATA, S._CHIP_LOSS
_note, _colors, _label, _num, _pct, _dec = S._note, S._colors, S._label, S._num, S._pct, S._dec
_cell, _table = S._cell, S._table
_fn, _call, _empty, _SLOW, _STOPPED = S._fn, S._call, S._empty, S._SLOW, S._STOPPED
_WHY_EMPTY = S._WHY_EMPTY
_chip, _is_number, _same, _Odd = V._chip, V._is_number, V._same, V._Odd
_DASH = V._DASH


def _euro(x, lang: str) -> str:
    return f"€ {_num(x, lang)}" if lang == "nl" else f"€{_num(x, lang)}"


def _listing_chip(i: int, lang: str) -> str:
    """The title of listing i on a background: blue for a tent, gray for the rest."""
    return _chip(listing(lang, i)["title"], _CHIP_PRED if i in TENTS else _CHIP_DATA)


def _facts(i: int, lang: str) -> str:
    """The price, the days and the km of listing i, small, after its title."""
    item = listing(lang, i)
    said = f'{_euro(item["price"], lang)}, {_num(item["days"], lang)} {_UNITS[lang]["days"]}, {_num(item["km"], lang)} km'
    return f'<span style="{_UI};margin-left:0.3rem">{html.escape(said)}</span>'


_UNITS = {"en": {"days": "days"}, "nl": {"days": "dagen"}}

_IDEA_TEXT = {
    "en": {"title": "One query, four steps",
           "label": "The listings for the query *tent*, in four steps. Each step keeps what the step before it does, and adds one thing:",
           "head": ["Place", "BM25 on title and text", "Title counts 3 times", "And the filters", "And the decay"],
           "tents": "The five tents, what the filters do with them, and what the decay multiplies their score by:",
           "head_tents": ["Tent", "Price", "Days", "km", "Filters", "Decay of the days", "Decay of the km"],
           "kept": "kept", "dropped": "dropped",
           "wanted": "*{t}* in the four steps: place {a}, {b}, {c} and {d}.",
           "gray": "Gray: the listings that are not tents, the scores, the prices, days and distances, and what the filters drop.",
           "blue": "Blue: the tents, what the filters keep, and what the decay multiplies a score by."},
    "nl": {"title": "Eén zoekvraag, vier stappen",
           "label": "De advertenties voor de zoekvraag *tent*, in vier stappen. Elke stap doet hetzelfde als de stap links ervan, en voegt één ding toe:",
           "head": ["Plaats", "BM25 op titel en tekst", "Titel telt 3 keer", "En de filters", "En de decay"],
           "tents": "De vijf tenten, wat de filters ermee doen, en waarmee de decay de score vermenigvuldigt:",
           "head_tents": ["Tent", "Prijs", "Dagen", "km", "Filters", "Decay van de dagen", "Decay van de km"],
           "kept": "blijft", "dropped": "valt af",
           "wanted": "*{t}* in de vier stappen: plaats {a}, {b}, {c} en {d}.",
           "gray": "Grijs: de advertenties die geen tent zijn, de scores, de prijzen, dagen en afstanden, en wat de filters laten afvallen.",
           "blue": "Blauw: de tenten, wat de filters houden, en waarmee de decay een score vermenigvuldigt."},
}
IDEA_PLACES = 5   # places that the picture of the idea shows for each step


def idea_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang]
    tops = {step: ranking(lang, step) for step in STEPS}
    rows = []
    for place in range(IDEA_PLACES):
        cells = ""
        for step in STEPS:
            if place < len(tops[step]):
                i, s = tops[step][place]
                cells += _cell(_listing_chip(i, lang) + f' <span style="{_DATA}">{html.escape(_dec(s, lang, 3))}</span>')
            else:
                cells += _cell(_DASH)
        rows.append("<tr>" + _cell(str(place + 1), _FAINT, _NUMBER) + cells + "</tr>")
    places = [place_of(lang, step, WANTED) for step in STEPS]
    body = _label(w["label"]) + _table(rows, w["head"], (0,))
    body += f'<div style="{_LABEL};{_PRED}">' + S._marked(w["wanted"].format(t=listing(lang, WANTED)["title"], a=places[0], b=places[1],
                                                                     c=places[2], d=places[3])) + "</div>"
    rows = []
    for i in TENTS:
        item = listing(lang, i)
        passes = keep(item, MAX_PRICE, MAX_KM)
        rows.append("<tr>" + _cell(_listing_chip(i, lang)) + _cell(html.escape(_euro(item["price"], lang)), _DATA, _NUMBER)
                    + _cell(_num(item["days"], lang), _DATA, _NUMBER) + _cell(_num(item["km"], lang), _DATA, _NUMBER)
                    + _cell(html.escape(w["kept" if passes else "dropped"]), _PRED if passes else _DATA)
                    + _cell(html.escape(_dec(decay(item["days"], HALF_DAYS), lang, 3)), _PRED, _NUMBER)
                    + _cell(html.escape(_dec(decay(item["km"], HALF_KM), lang, 3)), _PRED, _NUMBER) + "</tr>")
    body += _label(w["tents"]) + _table(rows, w["head_tents"], (1, 2, 3, 5, 6))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


_CC_TEXT = {
    "en": {"title": "Your combined_count on the listings for the query tent",
           "kind": "a number",
           "label": "Each listing that holds *tent*, with its count in the title and in the text, and your combined count with the title weight 3 and the text weight 1:",
           "head": ["Listing", "Title", "Text", "Your combined count"],
           "ranks": "BM25F with your combined counts, and with the combined lengths of the course. The first five listings:",
           "head_ranks": ["Place", "Listing", "Score"],
           "course": "With the course's combined counts, the first listing is *{t}*.",
           "gray": "Gray: the listings that are not tents, the counts in each field, and the course's combined count where yours differs.",
           "blue": "Blue: the tents, and your combined counts and scores where they are the course's.",
           "purple": "Purple: your combined counts and scores where they differ from the course's."},
    "nl": {"title": "Je combined_count op de advertenties voor de zoekvraag tent",
           "kind": "een getal",
           "label": "Elke advertentie met *tent* erin, met hoe vaak het woord in de titel en in de tekst staat, en je combined count met weging 3 voor de titel en 1 voor de tekst:",
           "head": ["Advertentie", "Titel", "Tekst", "Jouw combined count"],
           "ranks": "BM25F met je combined counts, en met de gecombineerde lengtes van de course. De eerste vijf advertenties:",
           "head_ranks": ["Plaats", "Advertentie", "Score"],
           "course": "Met de combined counts van de course staat *{t}* bovenaan.",
           "gray": "Grijs: de advertenties die geen tent zijn, hoe vaak het woord in elk veld staat, en de combined count van de course waar de jouwe verschilt.",
           "blue": "Blauw: de tenten, en je combined counts en scores waar deze gelijk zijn aan wat de course geeft.",
           "purple": "Paars: je combined counts en scores waar deze verschillen van wat de course geeft."},
}
_SHOWN = 5   # listings that the pictures under the checks show in a ranking


def _with_query() -> list:
    """The listings that hold the query in a field, in their order."""
    return [i for i in range(N) if combined_count(QUERY, fields("en", i), PLAIN)]


def _ranked(got: dict) -> list:
    return sorted(((i, s) for i, s in got.items() if s > 0), key=lambda p: (-p[1], p[0]))


def combined_count_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _CC_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")

    def make():
        out = {}
        for i in range(N):
            x = fn(QUERY, fields(lang, i), dict(WEIGHTS))
            if not _is_number(x) or x < 0:
                raise _Odd
            out[i] = float(x)
        return out

    def guarded():
        try:
            return make()
        except _Odd:
            return None

    got = _call(guarded)
    if got is _SLOW:
        return _empty(w, "slow")
    if got is _STOPPED:
        return _empty(w, "stopped")
    if got is None:
        return _empty(w, "odd")
    rows, all_same = [], True
    for i in _with_query():
        f = fields(lang, i)
        want = combined_count(QUERY, f, WEIGHTS)
        same = _same(got[i], want)
        all_same = all_same and same
        mine = f'<span style="{_PRED if same else _LOSS}">{html.escape(_dec(got[i], lang, 1) if got[i] != int(got[i]) else _num(got[i], lang))}</span>'
        if not same:
            mine += f' <span style="{_DATA}">({_num(want, lang)})</span>'
        rows.append("<tr>" + _cell(_listing_chip(i, lang)) + _cell(_num(f["title"].count(QUERY), lang), _DATA, _NUMBER)
                    + _cell(_num(f["text"].count(QUERY), lang), _DATA, _NUMBER) + _cell(mine, "", _NUMBER) + "</tr>")
    body = _label(w["label"]) + _table(rows, w["head"], (1, 2, 3))
    mine = _ranked(dict(enumerate(scores(lang, WEIGHTS, counts=[got[i] for i in range(N)]))))
    want = ranking(lang, "fields")
    same_rank = all_same and [i for i, _ in mine[:_SHOWN]] == [i for i, _ in want[:_SHOWN]]
    rows = []
    for place, (i, s) in enumerate(mine[:_SHOWN], 1):
        rows.append("<tr>" + _cell(str(place), _FAINT, _NUMBER) + _cell(_listing_chip(i, lang))
                    + _cell(html.escape(_dec(s, lang, 3)), _PRED if same_rank else _LOSS, _NUMBER) + "</tr>")
    body += _label(w["ranks"]) + (_table(rows, w["head_ranks"], (0, 2)) if rows else _note(_DASH))
    if not same_rank:
        body += f'<div style="{_LOSS}">' + S._marked(w["course"].format(t=listing(lang, want[0][0])["title"])) + "</div>"
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


_KEEP_TEXT = {
    "en": {"title": "Your keep on the listings for the query tent, with at most €100 and at most 50 km",
           "kind": "True or False",
           "label": "Each listing that holds *tent*, with its price and distance, and what your keep does with it:",
           "head": ["Listing", "Price", "km", "Your keep"],
           "kept": "kept", "dropped": "dropped",
           "ranks": "BM25F with the title weight 3, on the listings that your keep keeps. The first five listings:",
           "head_ranks": ["Place", "Listing", "Score"],
           "course": "The course's keep keeps {k} of these {n} listings.",
           "gray": "Gray: the listings that are not tents, the prices and distances, what your keep drops where the course's drops it too, and the scores.",
           "blue": "Blue: the tents, and what your keep keeps where the course's keeps it too.",
           "purple": "Purple: what your keep does where it differs from the course's."},
    "nl": {"title": "Je keep op de advertenties voor de zoekvraag tent, met hoogstens € 100 en hoogstens 50 km",
           "kind": "True of False",
           "label": "Elke advertentie met *tent* erin, met de prijs en de afstand, en wat je keep ermee doet:",
           "head": ["Advertentie", "Prijs", "km", "Jouw keep"],
           "kept": "blijft", "dropped": "valt af",
           "ranks": "BM25F met weging 3 voor de titel, op de advertenties die je keep houdt. De eerste vijf advertenties:",
           "head_ranks": ["Plaats", "Advertentie", "Score"],
           "course": "De keep van de course laat {k} van deze {n} advertenties staan.",
           "gray": "Grijs: de advertenties die geen tent zijn, de prijzen en afstanden, wat je keep laat afvallen waar de keep van de course dat ook doet, en de scores.",
           "blue": "Blauw: de tenten, en wat je keep houdt waar de keep van de course dat ook doet.",
           "purple": "Paars: wat je keep doet waar het verschilt van de keep van de course."},
}


def keep_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _KEEP_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")

    def make():
        out = {}
        for i in range(N):
            x = fn(listing(lang, i), MAX_PRICE, MAX_KM)
            if not isinstance(x, (bool, np.bool_)):
                raise _Odd
            out[i] = bool(x)
        return out

    def guarded():
        try:
            return make()
        except _Odd:
            return None

    got = _call(guarded)
    if got is _SLOW:
        return _empty(w, "slow")
    if got is _STOPPED:
        return _empty(w, "stopped")
    if got is None:
        return _empty(w, "odd")
    shown = _with_query()
    rows, all_same = [], True
    for i in shown:
        item = listing(lang, i)
        want = keep(item, MAX_PRICE, MAX_KM)
        same = got[i] == want
        all_same = all_same and same
        style = (_PRED if got[i] else _DATA) if same else _LOSS
        rows.append("<tr>" + _cell(_listing_chip(i, lang)) + _cell(html.escape(_euro(item["price"], lang)), _DATA, _NUMBER)
                    + _cell(_num(item["km"], lang), _DATA, _NUMBER)
                    + _cell(html.escape(w["kept" if got[i] else "dropped"]), style) + "</tr>")
    body = _label(w["label"]) + _table(rows, w["head"], (1, 2))
    base = scores(lang, WEIGHTS)
    mine = _ranked({i: base[i] for i in range(N) if got[i]})
    rows = []
    for place, (i, s) in enumerate(mine[:_SHOWN], 1):
        rows.append("<tr>" + _cell(str(place), _FAINT, _NUMBER) + _cell(_listing_chip(i, lang))
                    + _cell(html.escape(_dec(s, lang, 3)), _DATA, _NUMBER) + "</tr>")
    body += _label(w["ranks"]) + (_table(rows, w["head_ranks"], (0, 2)) if rows else _note(_DASH))
    if not all_same:
        k = sum(keep(listing(lang, i), MAX_PRICE, MAX_KM) for i in shown)
        body += f'<div style="{_LOSS}">' + html.escape(w["course"].format(k=_num(k, lang), n=_num(len(shown), lang))) + "</div>"
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


_DECAY_TEXT = {
    "en": {"title": "Your decay on the ages and distances of the listings",
           "kind": "a number",
           "days": "Your decay for ages in days, with half at {h} days:",
           "km": "Your decay for distances in km, with half at {h} km:",
           "head_days": ["Days", "Your decay"], "head_km": ["km", "Your decay"],
           "ranks": "BM25F with the title weight 3, the two filters, and the score times your decay of the days and of the km. The first five listings:",
           "head_ranks": ["Place", "Listing", "Score"],
           "course": "With the course's decay, the first listing is *{t}*.",
           "gray": "Gray: the days and the distances, and the course's decay where yours differs.",
           "blue": "Blue: the tents, and your decay and scores where they are the course's.",
           "purple": "Purple: your decay and scores where they differ from the course's."},
    "nl": {"title": "Je decay op de leeftijden en afstanden van de advertenties",
           "kind": "een getal",
           "days": "Je decay voor leeftijden in dagen, met de helft bij {h} dagen:",
           "km": "Je decay voor afstanden in km, met de helft bij {h} km:",
           "head_days": ["Dagen", "Jouw decay"], "head_km": ["km", "Jouw decay"],
           "ranks": "BM25F met weging 3 voor de titel, de twee filters, en de score maal je decay van de dagen en van de km. De eerste vijf advertenties:",
           "head_ranks": ["Plaats", "Advertentie", "Score"],
           "course": "Met de decay van de course staat *{t}* bovenaan.",
           "gray": "Grijs: de dagen en de afstanden, en de decay van de course waar de jouwe verschilt.",
           "blue": "Blauw: de tenten, en je decay en scores waar deze gelijk zijn aan wat de course geeft.",
           "purple": "Paars: je decay en scores waar deze verschillen van wat de course geeft."},
}
DECAY_DAYS = (0, 3, 7, 14, 28, 90)
DECAY_KM = (0, 12, 25, 50, 100)


def decay_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _DECAY_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    need = sorted({(d, HALF_DAYS) for d in DECAY_DAYS} | {(k, HALF_KM) for k in DECAY_KM}
                  | {(item["days"], HALF_DAYS) for item in listings(lang)} | {(item["km"], HALF_KM) for item in listings(lang)})

    def make():
        out = {}
        for d, h in need:
            x = fn(d, h)
            if not _is_number(x):
                raise _Odd
            out[d, h] = float(x)
        return out

    def guarded():
        try:
            return make()
        except _Odd:
            return None

    got = _call(guarded)
    if got is _SLOW:
        return _empty(w, "slow")
    if got is _STOPPED:
        return _empty(w, "stopped")
    if got is None:
        return _empty(w, "odd")
    all_same = all(_same(got[d, h], decay(d, h)) for d, h in need)

    def table(values, half, head):
        rows = []
        for d in values:
            mine, want = got[d, half], decay(d, half)
            same = _same(mine, want)
            text = f'<span style="{_PRED if same else _LOSS}">{html.escape(_dec(mine, lang, 3))}</span>'
            if not same:
                text += f' <span style="{_DATA}">({html.escape(_dec(want, lang, 3))})</span>'
            rows.append("<tr>" + _cell(_num(d, lang), _DATA, _NUMBER) + _cell(text, "", _NUMBER) + "</tr>")
        return _table(rows, head, (0, 1))

    body = (_label(w["days"].format(h=_num(HALF_DAYS, lang))) + table(DECAY_DAYS, HALF_DAYS, w["head_days"])
            + _label(w["km"].format(h=_num(HALF_KM, lang))) + table(DECAY_KM, HALF_KM, w["head_km"]))
    mine = _ranked(final_scores(lang, decay_fn=lambda d, h: got[d, h]))
    want = ranking(lang, "decay")
    same_rank = all_same and [i for i, _ in mine[:_SHOWN]] == [i for i, _ in want[:_SHOWN]]
    rows = []
    for place, (i, s) in enumerate(mine[:_SHOWN], 1):
        rows.append("<tr>" + _cell(str(place), _FAINT, _NUMBER) + _cell(_listing_chip(i, lang) + _facts(i, lang))
                    + _cell(html.escape(_dec(s, lang, 3)), _PRED if same_rank else _LOSS, _NUMBER) + "</tr>")
    body += _label(w["ranks"]) + (_table(rows, w["head_ranks"], (0, 2)) if rows else _note(_DASH))
    if not same_rank:
        body += f'<div style="{_LOSS}">' + S._marked(w["course"].format(t=listing(lang, want[0][0])["title"])) + "</div>"
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


_TRY_TEXT = {
    "en": {"title": "Your search for tent",
           "label": "The first six listings with your settings: the BM25F score, what the decay of the days and of the km multiplies it by, and the final score:",
           "head": ["Place", "Listing", "BM25F", "Decay of the days", "Decay of the km", "Final"],
           "dropped": "The filters drop {n} of the 24 listings: {t}.",
           "none": "The filters drop none of the 24 listings.",
           "nothing": "None of the listings that hold *tent* passes your filters.",
           "gray": "Gray: the listings that are not tents, the BM25F scores, the decays, and what the filters drop.",
           "blue": "Blue: the tents, and the final scores."},
    "nl": {"title": "De zoekvraag tent met jouw instellingen",
           "label": "De eerste zes advertenties met jouw instellingen: de BM25F-score, waarmee de decay van de dagen en van de km de score vermenigvuldigt, en de eindscore:",
           "head": ["Plaats", "Advertentie", "BM25F", "Decay van de dagen", "Decay van de km", "Eind"],
           "dropped": "De filters laten {n} van de 24 advertenties afvallen: {t}.",
           "none": "De filters laten geen van de 24 advertenties afvallen.",
           "nothing": "Geen van de advertenties met *tent* erin komt door je filters.",
           "gray": "Grijs: de advertenties die geen tent zijn, de BM25F-scores, de decays, en wat de filters laten afvallen.",
           "blue": "Blauw: de tenten, en de eindscores."},
}
TRY_START = {"weight": 3, "price": MAX_PRICE, "km": MAX_KM, "half_days": HALF_DAYS, "half_km": HALF_KM}
TRY_RANGES = {"weight": (1, 10, 1), "price": (10, 300, 10), "km": (5, 200, 5), "half_days": (1, 60, 1), "half_km": (5, 100, 5)}
TRY_PLACES = 6


def _setting(value, key: str) -> float:
    """A slider's value, inside its range and on its steps; the start when it is not a number."""
    low, high, step = TRY_RANGES[key]
    try:
        x = float(value)
    except (TypeError, ValueError):
        return TRY_START[key]
    if not math.isfinite(x):
        return TRY_START[key]
    return int(round(min(high, max(low, x)) / step) * step)


def try_picture(lang: str, weight=3, price=MAX_PRICE, km=MAX_KM, half_days=HALF_DAYS, half_km=HALF_KM) -> str:
    lang = lang if lang in LANGS else "en"
    w = _TRY_TEXT[lang]
    s = {key: _setting(v, key) for key, v in (("weight", weight), ("price", price), ("km", km),
                                              ("half_days", half_days), ("half_km", half_km))}
    weights = {"title": s["weight"], "text": 1}
    base = scores(lang, weights)
    final = final_scores(lang, weights, s["price"], s["km"], s["half_days"], s["half_km"])
    top = _ranked(final)[:TRY_PLACES]
    rows = []
    for place, (i, f) in enumerate(top, 1):
        item = listing(lang, i)
        rows.append("<tr>" + _cell(str(place), _FAINT, _NUMBER) + _cell(_listing_chip(i, lang) + _facts(i, lang))
                    + _cell(html.escape(_dec(base[i], lang, 3)), _DATA, _NUMBER)
                    + _cell(html.escape(_dec(decay(item["days"], s["half_days"]), lang, 3)), _DATA, _NUMBER)
                    + _cell(html.escape(_dec(decay(item["km"], s["half_km"]), lang, 3)), _DATA, _NUMBER)
                    + _cell(html.escape(_dec(f, lang, 3)), _PRED, _NUMBER) + "</tr>")
    body = _label(w["label"]) + (_table(rows, w["head"], (0, 2, 3, 4, 5)) if rows else _note(w["nothing"]))
    dropped = [i for i in range(N) if i not in final]
    if dropped:
        body += _note(w["dropped"].format(n=_num(len(dropped), lang), t=", ".join(listing(lang, i)["title"] for i in dropped)))
    else:
        body += _note(w["none"])
    title = w["title"]
    return yours.figure(title, body + _colors(w, body), alt=title)


# --- the exercises ---------------------------------------------------------------------------------------------------------

_Val = S._Val
_DOTS = S._DOTS
_SAME_TEXTS = S._SAME_TEXTS
_value, _show, _written = V._value, V._show, V._written

_CC_HEAD = "def combined_count(word, fields, weights):\n    count = 0\n    for name in fields:\n"
_CC_END = "    return count\n"
_CC = _CC_HEAD + "        count = count + weights[name] * fields[name].count(word)\n" + _CC_END
_CC_START = _CC_HEAD + "        count = ...\n" + _CC_END
_CC_NO_WEIGHT = _CC_HEAD + "        count = count + fields[name].count(word)\n" + _CC_END
_CC_LAST = _CC_HEAD + "        count = weights[name] * fields[name].count(word)\n" + _CC_END
_CC_WEIGHT_ONLY = _CC_HEAD + "        count = count + weights[name]\n" + _CC_END
_CC_PLUS = _CC_HEAD + "        count = count + weights[name] + fields[name].count(word)\n" + _CC_END

_KP_HEAD = "def keep(listing, max_price, max_km):\n"
_KP = _KP_HEAD + '    return listing["price"] <= max_price and listing["km"] <= max_km\n'
_KP_START = _KP_HEAD + "    return ...\n"
_KP_STRICT = _KP_HEAD + '    return listing["price"] < max_price and listing["km"] < max_km\n'
_KP_OR = _KP_HEAD + '    return listing["price"] <= max_price or listing["km"] <= max_km\n'
_KP_PRICE = _KP_HEAD + '    return listing["price"] <= max_price\n'
_KP_TURNED = _KP_HEAD + '    return listing["price"] >= max_price and listing["km"] >= max_km\n'

_DE_HEAD = "def decay(distance, half):\n"
_DE = _DE_HEAD + "    return 0.5 ** (distance / half)\n"
_DE_START = _DE_HEAD + "    return ...\n"
_DE_NO_PARENS = _DE_HEAD + "    return 0.5 ** distance / half\n"
_DE_LINEAR = _DE_HEAD + "    return 1 - distance / half\n"
_DE_NO_HALF = _DE_HEAD + "    return 0.5 ** distance\n"
_DE_TIMES = _DE_HEAD + "    return 0.5 * distance / half\n"

# The small cases the exercise texts show; the same in both lessons.
TOY_COMBINED = (("tent", {"title": ["tent", "for", "two"], "text": ["a", "small", "tent", "in", "a", "tent", "bag"]}, {"title": 3, "text": 1}),
                ("pomp", {"title": ["luchtbed", "voor", "twee"], "text": ["met", "een", "pomp"]}, {"title": 3, "text": 1}))
TOY_KEEP = (({"title": "Tent for two people", "price": 45, "days": 5, "km": 140}, 100, 50),
            ({"title": "Tent voor vier personen", "price": 85, "days": 3, "km": 12}, 85, 50))
TOY_DECAY = ((14, 14), (50, 25))


def _course_fields(lang: str, i: int) -> tuple:
    """combined_count for the word tent and listing i of the course, with the weights of the lesson."""
    return (_written(QUERY), _Val(f'C.fields("{lang}", {i})', fields(lang, i)), _Val("C.WEIGHTS", dict(WEIGHTS)))


def _course_listing(lang: str, i: int) -> tuple:
    """keep for listing i of the course, with the filters of the lesson."""
    return (_Val(f'C.listing("{lang}", {i})', listing(lang, i)), _written(MAX_PRICE), _written(MAX_KM))


REQUIRED = ["combined_count", "keep", "decay", "domains"]

EXERCISES = {
    "combined_count": FunctionBox(
        name="combined_count",
        picture=combined_count_picture,
        function="combined_count",
        start=_CC_START,
        solution=_CC,
        show=_show,
        visible=[tuple(_written(v) for v in case) for case in TOY_COMBINED],
        hidden=[tuple(_written(v) for v in case) for case in (
            ("tent", {"title": ["sleeping", "bag"], "text": ["warm", "and", "light"]}, {"title": 3, "text": 1}),
            ("tent", {"title": ["tent", "for", "two"], "text": ["a", "tent", "and", "a", "tent", "bag"]}, {"title": 0, "text": 1}),
            ("lamp", {"title": ["camping", "lamp"], "text": ["a", "lamp"], "brand": ["lamp", "co"]}, {"title": 3, "text": 1, "brand": 2}),
            ("bike", {"title": ["road", "bike"], "text": ["bike", "with", "new", "tires"]}, {"title": 2.5, "text": 1}),
            ("tent", {}, {}))]
        + [_course_fields("en", 0), _course_fields("en", 1), _course_fields("nl", 5)],
        first=[_DOTS],
        mistakes=[
            Mistake("no-weight", wrong=_CC_NO_WEIGHT,
                    en="`{call}` gives {got}, and it should give {want}: your function adds the counts without the weights, so a word in the title counts as much as a word in the text. What should each count be multiplied by?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie telt de aantallen op zonder weging, dus een woord in de titel telt even zwaar als een woord in de tekst. Waarmee moet elk aantal vermenigvuldigd worden?"),
            Mistake("last", wrong=_CC_LAST,
                    en="`{call}` gives {got}, and it should give {want}: each round throws away `count` and keeps only the part of the last field. What should the new `count` be made of?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke ronde gooit `count` weg en bewaart alleen het deel van het laatste veld. Waaruit bestaat de nieuwe `count`?"),
            Mistake("weight-only", wrong=_CC_WEIGHT_ONLY,
                    en="`{call}` gives {got}, and it should give {want}: your function adds the weight of every field, whether the word is in the field or not. Where does the count of `word` come in?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie telt de weging van elk veld op, of het woord in het veld staat of niet. Waar komt het aantal keer dat `word` erin staat erbij?"),
            Mistake("plus", wrong=_CC_PLUS,
                    en="`{call}` gives {got}, and it should give {want}: your function adds the weight to the count, where it should multiply. What should a weight of 3 do to a word that occurs once in the title?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie telt de weging bij het aantal op, waar je functie moet vermenigvuldigen. Wat moet een weging van 3 doen met een woord dat één keer in de titel staat?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the new `count`: the count so far, plus the weight of the field times the count of the word in the words of the field.",
                      "nl": "Vervang de `...` door de nieuwe `count`: de telling tot nu toe, plus de weging van het veld maal hoe vaak het woord in de woorden van het veld staat."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does each field add its weight times the count of the word in its words?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Telt je functie voor elk veld de weging maal hoe vaak het woord in de woorden van het veld staat bij `count` op?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does your function also work for a word in no field, a weight of 0, three fields, a weight that is not a whole number, and no fields at all?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie ook voor een woord in geen enkel veld, een weging van 0, drie velden, een weging die geen heel getal is, en helemaal geen velden?"},
            "correct": {"en": "Right: your function weighs each field before the counts are added up. In the picture below, it gives the combined count of every listing that holds *tent*: which listings come first now?",
                        "nl": "Goed: je functie geeft elk veld de weging voordat de aantallen worden opgeteld. In het plaatje hieronder geeft je functie de combined count van elke advertentie met *tent* erin: welke advertenties staan nu bovenaan?"},
        },
    ),
    "keep": FunctionBox(
        name="keep",
        picture=keep_picture,
        function="keep",
        start=_KP_START,
        solution=_KP,
        show=_show,
        visible=[tuple(_written(v) for v in case) for case in TOY_KEEP],
        hidden=[tuple(_written(v) for v in case) for case in (
            ({"title": "Camping lamp", "price": 100, "days": 4, "km": 22}, 100, 50),
            ({"title": "Bookcase", "price": 35, "days": 25, "km": 50}, 100, 50),
            ({"title": "New tent for two people", "price": 140, "days": 2, "km": 30}, 100, 50),
            ({"title": "Road bike", "price": 300, "days": 2, "km": 35}, 400, 30),
            ({"title": "Desk lamp", "price": 15, "days": 8, "km": 15}, 10, 10))]
        + [_course_listing("en", 4), _course_listing("nl", 0), _course_listing("nl", 6)],
        first=[_DOTS],
        mistakes=[
            Mistake("strict", wrong=_KP_STRICT,
                    en="`{call}` gives {got}, and it should give {want}: your function drops a listing whose price or distance is exactly the limit. *At most* includes the limit itself: which comparison says that?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie laat een advertentie afvallen waarvan de prijs of de afstand precies op de grens ligt. *Hoogstens* telt de grens zelf mee: welke vergelijking zegt dat?"),
            Mistake("or", wrong=_KP_OR,
                    en="`{call}` gives {got}, and it should give {want}: your function keeps a listing that passes one of the two filters. Should a tent 140 km away stay because its price is fine?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie laat een advertentie die door één van de twee filters komt, staan. Moet een tent op 140 km blijven staan omdat de prijs goed is?"),
            Mistake("price", wrong=_KP_PRICE,
                    en="`{call}` gives {got}, and it should give {want}: your function only looks at the price. Where does the distance come in?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie kijkt alleen naar de prijs. Waar komt de afstand erbij?"),
            Mistake("turned", wrong=_KP_TURNED,
                    en="`{call}` gives {got}, and it should give {want}: your function keeps the listings above the limits, the other way around. Which way should the two comparisons point?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie laat juist de advertenties boven de grenzen staan. Welke kant moeten de twee vergelijkingen op wijzen?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the two filters: the price at most `max_price`, and the distance at most `max_km`.",
                      "nl": "Vervang de `...` door de twee filters: de prijs hoogstens `max_price`, en de afstand hoogstens `max_km`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function keep a listing when its price is at most `max_price` and its distance at most `max_km`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Laat je functie een advertentie staan als de prijs hoogstens `max_price` is en de afstand hoogstens `max_km`?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does your function also keep a listing whose price is exactly `max_price`, or whose distance is exactly `max_km`, and drop one that is over either limit?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Laat je functie ook een advertentie staan waarvan de prijs precies `max_price` is, of de afstand precies `max_km`, en laat de functie een advertentie die over een van de twee grenzen gaat, afvallen?"},
            "correct": {"en": "Right: your function keeps a listing only when it passes both filters. In the picture below, it filters the listings for the query *tent*: which tents are left?",
                        "nl": "Goed: je functie laat een advertentie alleen staan als deze door beide filters komt. In het plaatje hieronder filtert je functie de advertenties voor de zoekvraag *tent*: welke tenten blijven over?"},
        },
    ),
    "decay": FunctionBox(
        name="decay",
        picture=decay_picture,
        function="decay",
        start=_DE_START,
        solution=_DE,
        show=_show,
        visible=[tuple(_written(v) for v in case) for case in TOY_DECAY],
        hidden=[tuple(_written(v) for v in case) for case in ((0, 25), (7, 14), (90, 14), (12, 25), (3, 2.5))]
        + [(_written(np.array([0.0, 14.0, 28.0])), _written(14))],
        first=[_DOTS],
        mistakes=[
            Mistake("no-parens", wrong=_DE_NO_PARENS,
                    en="`{call}` gives {got}, and it should give {want}: Python raises to a power before it divides, so your function divides 0.5 to the power of the distance by `half`. How do parentheses make the power the distance divided by `half`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: Python rekent een macht uit voordat het deelt, dus je functie deelt 0,5 tot de macht van de afstand door `half`. Hoe maak je met haakjes de afstand gedeeld door `half` tot de macht?"),
            Mistake("linear", wrong=_DE_LINEAR,
                    en="`{call}` gives {got}, and it should give {want}: your function goes down by the same amount with every step, and reaches 0 at the distance `half`. What should it give at `half`, and at twice `half`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie gaat bij elke stap evenveel omlaag, en komt bij de afstand `half` op 0. Wat moet je functie geven bij `half`, en bij twee keer `half`?"),
            Mistake("no-half", wrong=_DE_NO_HALF,
                    en="`{call}` gives {got}, and it should give {want}: your function raises 0.5 to the power of the distance itself, so the score is almost gone after a few days. Where does `half` come in?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie neemt 0,5 tot de macht van de afstand zelf, dus na een paar dagen is de score bijna weg. Waar komt `half` erbij?"),
            Mistake("times", wrong=_DE_TIMES,
                    en="`{call}` gives {got}, and it should give {want}: your function multiplies 0.5 by the distance against `half`, where it should raise 0.5 to that power. Which operator raises a number to a power in Python?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie vermenigvuldigt 0,5 met de afstand ten opzichte van `half`, waar je functie 0,5 tot deze macht moet nemen. Met welke operator neem je in Python een macht?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the decay: 0.5 to the power of the distance divided by `half`.",
                      "nl": "Vervang de `...` door de decay: 0,5 tot de macht van de afstand gedeeld door `half`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function raise 0.5 to the power of the distance divided by `half`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Neemt je functie 0,5 tot de macht van de afstand gedeeld door `half`?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does your function also work for a distance of 0, a distance that is not a multiple of `half`, a `half` that is not a whole number, and a NumPy array of distances?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie ook voor een afstand van 0, een afstand die geen veelvoud van `half` is, een `half` die geen heel getal is, en een NumPy-array met afstanden?"},
            "correct": {"en": "Right: your function halves a score with every `half` of distance. In the picture below, it lowers the scores of the listings by their age and their distance: which listing comes first now?",
                        "nl": "Goed: je functie halveert een score bij elke `half` aan afstand. In het plaatje hieronder verlaagt je functie de score van elke advertentie met de leeftijd en de afstand: welke advertentie staat nu bovenaan?"},
        },
    ),
}

# The sort: what a search engine for each domain needs.
EXERCISES["domains"] = Sort(
    name="domains",
    groups={
        "news": {"en": "News", "nl": "Nieuws"},
        "products": {"en": "Products", "nl": "Producten"},
        "local": {"en": "Places nearby", "nl": "Plekken in de buurt"},
        "enterprise": {"en": "The documents of a company", "nl": "De documenten van een bedrijf"},
        "law": {"en": "Law", "nl": "Recht"},
    },
    items={
        "fresh": {"en": "a news report of this morning beats a better match from last year",
                  "nl": "een nieuwsbericht van vanochtend wint van een betere treffer van vorig jaar"},
        "storm": {"en": "the query *storm* is about the storm of this week, not about one of ten years ago",
                  "nl": "de zoekvraag *storm* gaat over de storm van deze week, niet over een storm van tien jaar geleden"},
        "size": {"en": "the searcher sets a top price and a size, and only what fits is left",
                 "nl": "je stelt een hoogste prijs en een maat in, en alleen wat past blijft over"},
        "reviews": {"en": "a match in the name of the item counts for more than a match in its reviews",
                    "nl": "een treffer in de naam van het product telt zwaarder dan een treffer in de reviews"},
        "bakery": {"en": "a bakery two streets away beats a better-matching one across the country",
                   "nl": "een bakker twee straten verderop wint van een bakker aan de andere kant van het land met een betere treffer"},
        "town": {"en": "the same query gives other results in another town",
                 "nl": "dezelfde zoekvraag geeft in een andere stad andere resultaten"},
        "access": {"en": "the results may only hold documents that the searcher is allowed to open",
                   "nl": "alleen documenten die je mag openen, mogen in de resultaten staan"},
        "secret": {"en": "even the title of a document in the results can give away a secret",
                   "nl": "zelfs de titel van een document in de resultaten kan een geheim verraden"},
        "precise": {"en": "long, precise queries with words that must appear in the same sentence",
                    "nl": "lange, precieze zoekvragen met woorden die in dezelfde zin moeten staan"},
        "recall": {"en": "a librarian wants every document that matches, not only the best ten",
                   "nl": "een bibliothecaris wil elk document dat past, niet alleen de beste tien"},
    },
    solution={"fresh": "news", "storm": "news", "size": "products", "reviews": "products", "bakery": "local",
              "town": "local", "access": "enterprise", "secret": "enterprise", "precise": "law", "recall": "law"},
    nudge={
        "fresh": {"en": "Which domain is about what happened today?",
                  "nl": "Welk domein gaat over wat er vandaag gebeurde?"},
        "storm": {"en": "What decides here what *storm* is about: the place, or the time?",
                  "nl": "Wat beslist hier waar *storm* over gaat: de plek, of de tijd?"},
        "size": {"en": "In which domain does the thing you search for have a price and a size?",
                 "nl": "In welk domein heeft wat je zoekt een prijs en een maat?"},
        "reviews": {"en": "In which domain do the things have a name and reviews?",
                    "nl": "In welk domein hebben de dingen een naam en reviews?"},
        "bakery": {"en": "What decides here: the words, or the distance?",
                   "nl": "Wat beslist hier: de woorden, of de afstand?"},
        "town": {"en": "What changes when the searcher is in another town: the words, or the distance?",
                 "nl": "Wat verandert er als de zoeker in een andere stad is: de woorden, of de afstand?"},
        "access": {"en": "In which domain is not every document meant for everyone?",
                   "nl": "In welk domein is niet elk document voor iedereen bedoeld?"},
        "secret": {"en": "*The idea* quotes the book on documents that a searcher is barred from opening. Where was that?",
                   "nl": "*Het idee* noemt wat het boek zegt over documenten die je niet mag openen. Waar ging dat over?"},
        "precise": {"en": "*The idea* quotes the book on long, precise queries. Which search service was that about?",
                    "nl": "*Het idee* noemt wat het boek zegt over lange, precieze zoekvragen. Over welke zoekdienst ging dat?"},
        "recall": {"en": "*The idea* says which librarians recommend Boolean queries to miss nothing. Which librarians were they?",
                   "nl": "*Het idee* zegt welke bibliothecarissen Booleaanse zoekvragen aanraden om niets te missen. Welke bibliothecarissen waren dat?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the groups again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de groepen opnieuw uit."},
        "empty": {"en": "Sort each statement into the group that fits.",
                  "nl": "Deel elke uitspraak in bij de groep die klopt."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "{item} belongs with another group.",
                  "nl": "{item} hoort bij een andere groep."},
        "correct": {"en": "All {n} are right. A search engine for news ranks by freshness, one for products filters and weighs the fields of its items, one for places ranks by distance, one for a company shows only what the searcher may open, and one for law takes precise queries that should miss nothing.",
                    "nl": "Alle {n} kloppen. Een zoekmachine voor nieuws rangschikt op versheid, een voor producten filtert en weegt de velden van de producten, een voor plekken rangschikt op afstand, een voor een bedrijf laat alleen zien wat je mag openen, en een voor recht krijgt precieze zoekvragen die niets mogen missen."},
    },
)

# What the tests feed each check, and what it must say: (exercise, code or choices, state, known mistake).
CASES = [
    ("combined_count", _CC_START, "empty", None),
    ("combined_count", _CC, "correct", None),
    ("combined_count", _CC_HEAD + "        count = count + weights[name] * ...\n" + _CC_END, "wrong", "dots-left"),
    ("combined_count", _CC_NO_WEIGHT, "wrong", "no-weight"),
    ("combined_count", _CC_LAST, "wrong", "last"),
    ("combined_count", _CC_WEIGHT_ONLY, "wrong", "weight-only"),
    ("combined_count", _CC_PLUS, "wrong", "plus"),
    ("combined_count", _CC_HEAD + "        count += fields[name].count(word) * weights[name]\n" + _CC_END, "correct", None),
    ("combined_count", _CC_HEAD + "        count = count + weights[name] * fields[name].count(word)\n        return count\n",
     "wrong", None),
    ("keep", _KP_START, "empty", None),
    ("keep", _KP, "correct", None),
    ("keep", _KP_HEAD + '    return listing["price"] <= max_price and ...\n', "wrong", "dots-left"),
    ("keep", _KP_STRICT, "wrong", "strict"),
    ("keep", _KP_OR, "wrong", "or"),
    ("keep", _KP_PRICE, "wrong", "price"),
    ("keep", _KP_TURNED, "wrong", "turned"),
    ("keep", _KP_HEAD + '    return not (listing["price"] > max_price or listing["km"] > max_km)\n', "correct", None),
    ("keep", _KP_HEAD + '    if listing["price"] > max_price:\n        return False\n    return listing["km"] <= max_km\n',
     "correct", None),
    ("decay", _DE_START, "empty", None),
    ("decay", _DE, "correct", None),
    ("decay", _DE_HEAD + "    return 0.5 ** (...)\n", "wrong", "dots-left"),
    ("decay", _DE_NO_PARENS, "wrong", "no-parens"),
    ("decay", _DE_LINEAR, "wrong", "linear"),
    ("decay", _DE_NO_HALF, "wrong", "no-half"),
    ("decay", _DE_TIMES, "wrong", "times"),
    ("decay", _DE_HEAD + "    return 2 ** (-distance / half)\n", "correct", None),
    ("decay", _DE_HEAD + "    return (1 / 2) ** (distance / half)\n", "correct", None),
    ("domains", {}, "empty", None),
    ("domains", dict(EXERCISES["domains"].solution), "correct", None),
    ("domains", {"fresh": "news", "access": "enterprise"}, "wrong", "unfinished"),
] + [("domains", {item: ("law" if group == "news" else "news")}, "wrong", f"wrong:{item}")
     for item, group in EXERCISES["domains"].solution.items()]
