# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-bm25: BM25's inverse document frequency of a word, and the BM25 score of a text for a question;
what the course measured on the points of the course's summaries: how often the word search of build-rag-first and
BM25 put the right point or lesson first, for the questions and for every lesson's title as a question; the
pictures, the worked example, the scores of all points at once with NumPy, and the demo with the learner's BM25."""
import functools
import html

import numpy as np

from everything_ai import summaries, yours
from everything_ai.check import FunctionBox, Mistake, run_code, with_time_limit

POINTS = summaries.points("en")
ABOUT = summaries.about()
K1, B = summaries.K1, summaries.B


@functools.cache
def index() -> dict:
    """The words of every point as a list, the idf of every word, and the average number of words of a point."""
    return summaries.bm25_index(POINTS)


def having(word: str) -> int:
    """How many points have the word."""
    return sum(1 for ws in index()["words"] if word in ws)


def titles() -> list:
    """Every lesson of the points once, with its English title, in the order of the points."""
    out = {}
    for about in ABOUT:
        out.setdefault(about["lesson"], about["title"]["en"])
    return list(out.items())


def lesson_of(text: str) -> dict:
    return ABOUT[POINTS.index(text)]


# --- the scores of all points at once, with NumPy: the same order as summaries.search and summaries.bm25_search ------

@functools.cache
def _postings() -> dict:
    out: dict = {}
    for i, ws in enumerate(index()["words"]):
        for w in set(ws):
            out.setdefault(w, []).append((i, ws.count(w)))
    return out


@functools.cache
def _column(word: str) -> np.ndarray:
    """How often each point has the word."""
    f = np.zeros(len(POINTS))
    for i, c in _postings().get(word, []):
        f[i] = c
    return f


@functools.cache
def _rank() -> np.ndarray:
    """The place of each point among all points in alphabetical order: sorting pairs puts the later text first."""
    order = sorted(range(len(POINTS)), key=lambda i: POINTS[i])
    rank = np.zeros(len(POINTS), dtype=int)
    rank[order] = np.arange(len(POINTS))
    return rank


def scores(question: str, way: str) -> np.ndarray:
    """The score of every point for the question: the shared words ("overlap") or BM25 ("bm25"), the words of the
    question in alphabetical order, as summaries.bm25_search adds them."""
    lengths = np.array([len(ws) for ws in index()["words"]])
    out = np.zeros(len(POINTS))
    for word in sorted(summaries.words(question)):
        f = _column(word)
        if way == "overlap":
            out = out + (f > 0)
        else:
            out = out + index()["idf"].get(word, 0.0) * f * (K1 + 1) / (f + K1 * (1 - B + B * lengths / index()["average"]))
    return out


def first(question: str, way: str, k: int) -> list:
    """The k points with the highest score, highest first, as sorting (score, text) pairs from high to low gives them."""
    order = np.lexsort((_rank(), scores(question, way)))[::-1]
    return [POINTS[i] for i in order[:k]]


@functools.cache
def findings() -> dict:
    """How often each way puts the right lesson first or in its first three, for every lesson's title as a question;
    and the right point, for the questions of each kind."""
    out = {"titles": len(titles())}
    for way in ("overlap", "bm25"):
        top = [[lesson_of(t)["lesson"] for t in first(title, way, 3)] for _, title in titles()]
        out[way] = {"title_1": sum(t[0] == lesson for t, (lesson, _) in zip(top, titles())),
                    "title_3": sum(lesson in t for t, (lesson, _) in zip(top, titles()))}
        for kind in ("words", "meaning"):
            qs = summaries.questions(kind)
            top = [[lesson_of(t)["id"] for t in first(q["text"]["en"], way, 3)] for q in qs]
            out[way][kind + "_1"] = sum(t[0] == q["gold"] for t, q in zip(top, qs))
            out[way][kind + "_3"] = sum(q["gold"] in t for t, q in zip(top, qs))
            out[kind] = len(qs)
    return out


# --- the worked example: a title where the two ways differ ---------------------------------------------------------------

EXAMPLE = "Clusters of any shape: DBSCAN"   # the title of ml-dbscan


def parts(question: str, text: str) -> list:
    """Per word of the question, in alphabetical order: how many points have it, its idf, how often the text has it,
    and what it adds to the text's BM25 score."""
    ws = summaries.word_list(text)
    out = []
    for word in sorted(summaries.words(question)):
        n, f = having(word), ws.count(word)
        i = index()["idf"].get(word, 0.0)
        out.append({"word": word, "n": n, "idf": i, "f": f,
                    "part": i * f * (K1 + 1) / (f + K1 * (1 - B + B * len(ws) / index()["average"]))})
    return out


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


_LABEL = "font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)"

_IDF_WORDS = {
    "en": {"title": "What each word of the question counts: its idf over the 1,266 points", "points": "in {n} points"},
    "nl": {"title": "Wat elk woord van de vraag telt: de idf over de 1.266 punten", "points": "in {n} punten"},
}


def idf_picture(lang: str) -> str:
    """Per word of the example question a bar of its idf, as input data, with how many points have the word."""
    w = _IDF_WORDS[lang]
    rows = []
    most = max(p["idf"] for p in parts(EXAMPLE, POINTS[0]))
    for p in parts(EXAMPLE, POINTS[0]):
        bar = (f'<span style="display:inline-block;height:12px;width:{max(2, round(220 * p["idf"] / most))}px;background:var(--ea-data-wash);'
               f'border:1px solid var(--ea-data);border-radius:2px;vertical-align:middle"></span>')
        rows.append(f'<div style="display:grid;grid-template-columns:6rem minmax(0,15rem) auto;gap:0.5rem;align-items:center;margin:0.2rem 0">'
                    f'<span style="font-family:var(--ea-font-code);font-size:13px;color:var(--ea-ink)">{html.escape(p["word"])}</span>'
                    f'<span>{bar} <span style="{_LABEL}">{number(p["idf"], lang, 2)}</span></span>'
                    f'<span style="{_LABEL}">{html.escape(w["points"].format(n=number(p["n"], lang)))}</span></div>')
    return yours.figure(w["title"], "".join(rows), alt=w["title"])


_RESULT_WORDS = {
    "en": {"title": "How often the right lesson comes first, with each lesson's title as the question",
           "overlap": "shared words", "bm25": "BM25", "first": "first", "three": "in the first three", "of": "{k} of {n}"},
    "nl": {"title": "Hoe vaak de goede les bovenaan komt, met de titel van elke les als vraag",
           "overlap": "gedeelde woorden", "bm25": "BM25", "first": "bovenaan", "three": "bij de eerste drie", "of": "{k} van de {n}"},
}


def titles_picture(lang: str) -> str:
    """Per way and per measure a bar of how many of the titles found their lesson: first, and in the first three, as
    input data."""
    w = _RESULT_WORDS[lang]
    f = findings()
    rows = []
    for way in ("overlap", "bm25"):
        for key, label in (("title_1", "first"), ("title_3", "three")):
            k = f[way][key]
            bar = (f'<span style="display:inline-block;height:12px;width:{round(260 * k / f["titles"])}px;background:var(--ea-data-wash);'
                   f'border:1px solid var(--ea-data);border-radius:2px;vertical-align:middle"></span>')
            rows.append(f'<div style="display:grid;grid-template-columns:12rem minmax(0,1fr);gap:0.5rem;align-items:center;margin:0.2rem 0">'
                        f'<span style="{_LABEL}">{html.escape(w[way])}, {html.escape(w[label])}</span>'
                        f'<span>{bar} <span style="{_LABEL}">{html.escape(w["of"].format(k=k, n=f["titles"]))}</span></span></div>')
    return yours.figure(w["title"], "".join(rows), alt=w["title"])


# --- exercise 1: idf ----------------------------------------------------------------------------------------------------------

START_IDF = '''import math


def idf(n, total):
    return math.log(total / n)
'''
SOLUTION_IDF = '''import math


def idf(n, total):
    return math.log((total - n + 0.5) / (n + 0.5) + 1)
'''

_IDF_PICTURE = {
    "en": {"title": "What your idf gives words of the points, from rare to common", "points": "in {n} of 1,266 points",
           "error": "your idf stops here"},
    "nl": {"title": "Wat je idf geeft aan woorden van de punten, van zeldzaam tot gewoon", "points": "in {n} van de 1.266 punten",
           "error": "je idf stopt hier"},
}
_IDF_WORDS_SHOWN = ["dbscan", "perplexity", "parameters", "loss", "of", "the"]


def idf_exercise_picture(seen) -> str:
    """The learner's idf of six words of the points, from rare to common, as bars of input data; a value below 0 or an
    error as a mistake."""
    w = _IDF_PICTURE[seen.lang]
    mine = seen.mine
    fn = mine.names.get("idf") if mine is not None else None
    if not callable(fn):
        return ""
    rows = []
    for word in _IDF_WORDS_SHOWN:
        n = having(word)
        try:
            value = with_time_limit(lambda: fn(n, len(POINTS)))
            value = float(value)
        except Exception:   # the learner's code; the check says what went wrong
            value = None
        if value is None:
            bar, label = "", f'<span style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-loss-text)">{html.escape(w["error"])}</span>'
        else:
            bad = value < 0
            bar = (f'<span style="display:inline-block;height:12px;width:{max(2, min(220, round(40 * abs(value))))}px;'
                   f'background:var(--ea-{"loss" if bad else "data"}-wash);border:1px solid var(--ea-{"loss" if bad else "data"});border-radius:2px;vertical-align:middle"></span>')
            label = f'<span style="{_LABEL}">{number(value, seen.lang, 2)}</span>'
        rows.append(f'<div style="display:grid;grid-template-columns:6rem minmax(0,14rem) auto;gap:0.5rem;align-items:center;margin:0.2rem 0">'
                    f'<span style="font-family:var(--ea-font-code);font-size:13px;color:var(--ea-ink)">{html.escape(word)}</span>'
                    f'<span>{bar} {label}</span><span style="{_LABEL}">{html.escape(w["points"].format(n=number(n, seen.lang)))}</span></div>')
    return yours.figure(w["title"], "".join(rows), alt=w["title"])


# --- exercise 2: bm25 -----------------------------------------------------------------------------------------------------

START_BM25 = '''def bm25(asked, words, idfs, average):
    score = 0
    for word in asked:
        f = words.count(word)
        score = score + idfs[word] * f
    return score
'''
SOLUTION_BM25 = '''def bm25(asked, words, idfs, average):
    score = 0
    for word in asked:
        f = words.count(word)
        score = score + idfs[word] * f * (K1 + 1) / (f + K1 * (1 - B + B * len(words) / average))
    return score
'''
NO_LENGTH = SOLUTION_BM25.replace(" * (1 - B + B * len(words) / average)", "")
TEXT = ["the", "loss", "falls", "the", "loss", "loss"]
IDFS = {"loss": 2.0, "the": 0.1, "dbscan": 6.0}

_BM25_PICTURE = {
    "en": {"title": "What a word adds to the score in your bm25, by how often the text has it", "times": "{f} times",
           "once": "once", "plain": "every time the same"},
    "nl": {"title": "Wat een woord bijdraagt aan de score in je bm25, naar hoe vaak de tekst het heeft", "times": "{f} keer",
           "once": "één keer", "plain": "elke keer even veel"},
}


def bm25_picture(seen) -> str:
    """What one word with idf 1 adds in the learner's bm25 to a text of average length that has it 1 to 8 times: bars
    as input data, next to a dashed line where each time would add the same."""
    w = _BM25_PICTURE[seen.lang]
    mine = seen.mine
    fn = mine.names.get("bm25") if mine is not None else None
    if not callable(fn):
        return ""
    rows = []
    for f in range(1, 9):
        words = ["word"] * f + ["other"] * (10 - f)
        try:
            value = float(with_time_limit(lambda: fn(["word"], words, {"word": 1.0}, 10.0)))
        except Exception:   # the learner's code; the check says what went wrong
            return ""
        bar = (f'<span style="display:inline-block;height:12px;width:{max(2, min(260, round(30 * value)))}px;background:var(--ea-data-wash);'
               f'border:1px solid var(--ea-data);border-radius:2px;vertical-align:middle"></span>')
        plain = f'<span style="display:inline-block;width:{30 * f}px;border-top:1px dashed var(--ea-ink-3);vertical-align:middle"></span>'
        rows.append(f'<div style="display:grid;grid-template-columns:4.5rem minmax(0,1fr);gap:0.5rem;align-items:center;margin:0.15rem 0">'
                    f'<span style="{_LABEL}">{html.escape(w["once"] if f == 1 else w["times"].format(f=f))}</span>'
                    f'<span style="position:relative">{bar} <span style="{_LABEL}">{number(value, seen.lang, 2)}</span><br>{plain}</span></div>')
    key = (f'<div style="margin-top:0.3rem;{_LABEL}"><span style="display:inline-block;width:24px;border-top:1px dashed var(--ea-ink-3);'
           f'vertical-align:middle;margin-right:0.3rem"></span>{html.escape(w["plain"])}</div>')
    return yours.figure(w["title"], "".join(rows) + key, alt=w["title"])


_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}


EXERCISES = {
    "idf": FunctionBox(
        name="idf",
        picture=idf_exercise_picture,
        function="idf",
        start=START_IDF,
        solution=SOLUTION_IDF,
        visible=[(8, 1266)],
        hidden=[(1092, 1266), (633, 1266), (1266, 1266), (0, 10)],
        mistakes=[
            Mistake("no-one", wrong=SOLUTION_IDF.replace(" + 1)", ")"),
                    en="`{call}` gives `{got}`. Without the `+ 1` inside the log, a word that more than half of the texts have gets an idf below 0, and would lower the score of every text that has it. Add 1 to the fraction, inside `math.log`.",
                    nl="`{call}` geeft `{got}`. Zonder de `+ 1` binnen de log krijgt een woord dat meer dan de helft van de teksten heeft een idf onder 0, en zou het de score verlagen van elke tekst die het heeft. Tel 1 op bij de breuk, binnen `math.log`."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`idf` is still the classic idf, `log(total / n)`: it divides by zero for a word that no text has. Write BM25's idf from the idea instead.",
                      "nl": "`idf` is nog de klassieke idf, `log(total / n)`: die deelt door nul voor een woord dat geen enkele tekst heeft. Schrijf in plaats daarvan de idf van BM25 uit het idee."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: the log of (total − n + 0.5) divided by (n + 0.5), plus 1.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: de log van (total − n + 0,5) gedeeld door (n + 0,5), plus 1."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for a word that every text has, and for one that none has?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Werkt je functie voor een woord dat elke tekst heeft, en voor een dat geen enkele heeft?"},
            "no_function": {"en": "Your box does not make a function `idf`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `idf`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `idf` is right in all {n} cases: high for a rare word, near 0 for a word almost every text has, and never below 0.",
                        "nl": "Je `idf` klopt in alle {n} gevallen: hoog voor een zeldzaam woord, bijna 0 voor een woord dat bijna elke tekst heeft, en nooit onder 0."},
        },
    ),
    "bm25": FunctionBox(
        name="bm25",
        picture=bm25_picture,
        function="bm25",
        start=START_BM25,
        solution=SOLUTION_BM25,
        names={"K1": K1, "B": B},
        visible=[(["loss", "the"], TEXT, IDFS, 4.0)],
        hidden=[(["loss", "the"], TEXT, IDFS, 6.0), (["dbscan"], TEXT, IDFS, 4.0), (["loss"], ["loss"], IDFS, 30.0), ([], TEXT, IDFS, 4.0)],
        mistakes=[
            Mistake("no-length", wrong=NO_LENGTH,
                    en="`{call}` gives `{got}`: each extra time adds less, but a long text still gets a head start. Multiply `K1` below the line by `(1 - B + B * len(words) / average)`.",
                    nl="`{call}` geeft `{got}`: elke extra keer telt minder, maar een lange tekst krijgt nog voorsprong. Vermenigvuldig `K1` onder de streep met `(1 - B + B * len(words) / average)`."),
            Mistake("no-plus-one", wrong=SOLUTION_BM25.replace("f * (K1 + 1) /", "f * K1 /"),
                    en="`{call}` gives `{got}`: above the line BM25 has `K1 + 1`, so that a word that appears once in a text of average length counts its idf once.",
                    nl="`{call}` geeft `{got}`: boven de streep heeft BM25 `K1 + 1`, zodat een woord dat één keer voorkomt in een tekst van gemiddelde lengte zijn idf één keer telt."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`bm25` still adds `idf * f`: every time a word appears counts the same, and a long text gets a head start. Replace `f` by BM25's form from the idea.",
                      "nl": "`bm25` telt nog `idf * f` op: elke keer dat een woord voorkomt telt even zwaar, en een lange tekst krijgt voorsprong. Vervang `f` door de vorm van BM25 uit het idee."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: per asked word its idf times `f * (K1 + 1)`, divided by `f + K1 * (1 - B + B * len(words) / average)`.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: per gevraagd woord de idf maal `f * (K1 + 1)`, gedeeld door `f + K1 * (1 - B + B * len(words) / average)`."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for a text of another length, and for a word the text does not have?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Werkt je functie voor een tekst van een andere lengte, en voor een woord dat de tekst niet heeft?"},
            "no_function": {"en": "Your box does not make a function `bm25`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `bm25`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `bm25` is right in all {n} cases: rare words count more, each extra time less, and a long text gets no head start.",
                        "nl": "Je `bm25` klopt in alle {n} gevallen: zeldzame woorden tellen zwaarder, elke extra keer minder, en een lange tekst krijgt geen voorsprong."},
        },
    ),
}

REQUIRED = ["idf", "bm25", "twice"]

CASES = [
    ("idf", START_IDF, "empty", None),
    ("idf", "", "empty", None),
    ("idf", SOLUTION_IDF, "correct", None),
    ("idf", "import math\n\n\ndef idf(n, total):\n    rare = (total - n + 0.5) / (n + 0.5)\n    return math.log(1 + rare)\n", "correct", None),
    ("idf", SOLUTION_IDF.replace(" + 1)", ")"), "wrong", "no-one"),
    ("idf", SOLUTION_IDF.replace("(n + 0.5)", "(n + 1)"), "wrong", None),
    ("bm25", START_BM25, "empty", None),
    ("bm25", SOLUTION_BM25, "correct", None),
    ("bm25", SOLUTION_BM25.replace("        score = score + idfs[word] * f * (K1 + 1) / (f + K1 * (1 - B + B * len(words) / average))\n",
                                   "        length = 1 - B + B * len(words) / average\n"
                                   "        score = score + idfs[word] * f * (K1 + 1) / (f + K1 * length)\n"), "correct", None),
    ("bm25", NO_LENGTH, "wrong", "no-length"),
    ("bm25", SOLUTION_BM25.replace("f * (K1 + 1) /", "f * K1 /"), "wrong", "no-plus-one"),
    ("bm25", SOLUTION_BM25.replace("len(words) / average", "average / len(words)"), "wrong", None),
]


# --- one rung up: the scores of all points at once with NumPy -----------------------------------------------------------

NUMPY_CODE = '''import numpy as np

asked = sorted(words(question))
# 1. a row per point, a column per word of the question: how often the point has it
f = np.array([[point.count(word) for word in asked] for point in point_words])
# 2. the idf of each word, and the length of each point as a column
idf = np.array([idfs[word] for word in asked])
lengths = np.array([[len(point)] for point in point_words])
# 3. the formula on the whole table at once, then a sum along each row
part = f * (K1 + 1) / (f + K1 * (1 - B + B * lengths / average))
scores = (part * idf).sum(axis=1)
'''


def numpy_scores(question: str) -> np.ndarray:
    """What NUMPY_CODE gives for a question, run as it stands."""
    point_words = index()["words"]
    idfs = {w: index()["idf"].get(w, 0.0) for w in summaries.words(question)}
    run = run_code(NUMPY_CODE, {"words": summaries.words, "question": question, "point_words": point_words, "idfs": idfs,
                                "K1": K1, "B": B, "average": index()["average"]})
    return run.names["scores"]


# --- the demo: your BM25 on the points ------------------------------------------------------------------------------------

def learner_bm25(idf_text, bm25_text):
    """The learner's idf and bm25 from exercises 1 and 2 when both are right; else None."""
    from everything_ai.check import MISSING
    for name, text in (("idf", idf_text), ("bm25", bm25_text)):
        if text is MISSING or text is None or EXERCISES[name].check(text).state != "correct":
            return None
    return run_code(idf_text).names["idf"], run_code(bm25_text, {"K1": K1, "B": B}).names["bm25"]


def demo(fns, question: str, words: dict, lang: str) -> str:
    """The first three points for the question with the shared words of build-rag-first, and with the learner's idf
    and bm25 over all points, side by side, each with its lesson."""
    idf_fn, bm25_fn = fns
    asked = sorted(w for w in summaries.words(question) if w)   # a question of only punctuation has no words
    if not asked:
        return f"<p><em>{html.escape(words['empty'])}</em></p>"
    try:
        def run():
            idfs = {w: idf_fn(having(w), len(POINTS)) for w in asked}
            found = [(bm25_fn(asked, ws, idfs, index()["average"]), t) for ws, t in zip(index()["words"], POINTS)]
            return [t for s, t in sorted(found, reverse=True)[:3]]
        mine = with_time_limit(run, 20)
    except Exception as e:   # the learner's code
        return f"<p><em>{html.escape(words['failed'].format(why=f'{type(e).__name__}: {e}'))}</em></p>"
    cols = []
    for label, found in ((words["overlap"], first(question, "overlap", 3)), (words["bm25"], mine)):
        items = "".join(f'<div style="margin:0.25rem 0;padding:0.25rem 0.4rem;border-radius:3px;background:var(--ea-data-wash)">'
                        f'<div style="{_LABEL}">{i + 1}. {html.escape(lesson_of(t)["title"][lang])}</div>'
                        f'<div style="font-family:var(--ea-font-ui);font-size:13px;color:var(--ea-ink)">{html.escape(t)}</div></div>'
                        for i, t in enumerate(found))
        cols.append(f'<div><div style="font-family:var(--ea-font-ui);font-size:13px;font-weight:600;color:var(--ea-ink)">{html.escape(label)}</div>{items}</div>')
    body = f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(16rem,1fr));gap:0.8rem">{"".join(cols)}</div>'
    return yours.figure(words["title"], body, alt=words["title"])
