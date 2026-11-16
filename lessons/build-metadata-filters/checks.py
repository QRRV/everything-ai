# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-metadata-filters: a store of the course's points in both languages, each with Granite Embedding's
embedding (data/course-summaries-granite.npz) and its metadata: language, lesson, domain and level. matches says
whether an item fits a filter, and filtered_search keeps only the items that fit before it takes the k best. For the
28 questions of the course: how often a store with both languages gives the same point twice, how many results a
search that filters afterwards keeps, and what a filter does to the IVF index of build-vector-index; the pictures, and
the search to try with the learner's own functions."""
import copy
import functools
import html
import json

import numpy as np

from everything_ai import summaries, yours
from everything_ai.check import MISSING, FunctionBox, Mistake, run_code, with_time_limit

LANGS = ("en", "nl")
QUESTIONS = summaries.questions()
ANSWERABLE = [q for q in QUESTIONS if q["kind"] != "none"]
ABOUT = summaries.about()
K = 3
WORKED = "q23"   # the worked example: a Dutch question whose point is in domain bouwen, but not among the three best of all
WORKED_WHERE = {"lang": "nl", "domain": "bouwen"}
FILTERS = [{"lang": "nl"}, {"lang": "nl", "level": "basis"}, {"lang": "nl", "domain": "bouwen"}]   # what the course measured
SHOWN = [{}, {"lang": "nl"}, {"lang": "nl", "level": "basis"}, {"lang": "nl", "domain": "bouwen"}, {"domain": "bouwen", "level": "expert"}]
DOMAIN = "bouwen"   # the filter the course tried on the IVF index


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def question(qid: str) -> dict:
    return next(q for q in QUESTIONS if q["id"] == qid)


def question_vector(q: dict, lang: str) -> np.ndarray:
    return summaries.vectors("questions", lang)[QUESTIONS.index(q)]


@functools.cache
def store() -> list:
    """The course's points in both languages: an item per point and language, with its text, its embedding and its
    metadata. The English points come first, then the Dutch ones, in the same order."""
    items = []
    for lang in LANGS:
        for i, (text, vector) in enumerate(zip(summaries.points(lang), summaries.vectors("points", lang))):
            items.append({"text": text, "vector": vector, "lang": lang, "lesson": ABOUT[i]["lesson"],
                          "domain": ABOUT[i]["domain"], "level": ABOUT[i]["level"]})
    return items


def point_of(item_number: int) -> int:
    """The number of the point an item of the store is about, the same in both languages."""
    return item_number % len(ABOUT)


def gold(q: dict) -> int | None:
    """The number of the point that answers a question; None when no point does."""
    return summaries.points("en").index(summaries.text_of(q["gold"])) if q["gold"] else None


@functools.cache
def _matrix() -> np.ndarray:
    return np.array([item["vector"] for item in store()])


def order(q: dict, lang: str) -> list:
    """The numbers of all items of the store, the best fit for a question first."""
    return [int(j) for j in np.argsort(-(_matrix() @ question_vector(q, lang)), kind="stable")]


# --- exercise 1: matches -------------------------------------------------------------------------------------------------

START_MATCHES = '''def matches(item, where):
    return True
'''
SOLUTION_MATCHES = '''def matches(item, where):
    for key in where:
        if item[key] != where[key]:
            return False
    return True
'''
ANY = '''def matches(item, where):
    for key in where:
        if item[key] == where[key]:
            return True
    return False
'''
FIRST_KEY = '''def matches(item, where):
    for key in where:
        return item[key] == where[key]
    return True
'''
matches = run_code(SOLUTION_MATCHES).names["matches"]

NOTE = {"lang": "nl", "level": "basis", "lesson": "ml-kmeans"}   # the item of exercise 1: its metadata

_LABEL = "font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)"
_CODE = "font-family:var(--ea-font-code);font-size:12px;color:var(--ea-ink)"


def where_text(where: dict) -> str:
    return json.dumps(where)


_MATCHES_WORDS = {
    "en": {"title": "How many of the course's {n} items fit, by your matches", "items": "{n} items", "one": "1 item",
           "alt": "A bar per filter, as long as the number of items of the course's store that fit it by the learner's matches."},
    "nl": {"title": "Hoeveel van de {n} items van de course passen, volgens je matches", "items": "{n} items", "one": "1 item",
           "alt": "Een balk per filter, zo lang als het aantal items van de store van de course dat er volgens matches van de leerling bij past."},
}


def matches_picture(seen) -> str:
    """The learner's matches on every item of the course's store, for a few filters: a bar per filter, as long as the
    number of items that fit."""
    w = _MATCHES_WORDS[seen.lang]
    mine = seen.mine
    fn = mine.names.get("matches") if mine is not None else None
    if not callable(fn):
        return ""
    items = [{k: v for k, v in item.items() if k != "vector"} for item in store()]
    counts = []
    for where in SHOWN:
        try:
            counts.append(int(with_time_limit(lambda: sum(bool(fn(dict(item), dict(where))) for item in items), 10)))
        except Exception:   # the learner's code; the check says what went wrong
            return ""
    rows = []
    for where, n in zip(SHOWN, counts):
        size = w["one"] if n == 1 else w["items"].format(n=number(n, seen.lang))
        rows.append(f'<span style="{_CODE}">{html.escape(where_text(where))}</span>'
                    f'<span style="display:block;height:12px;width:{100 * n / len(items):.1f}%;background:var(--ea-data);border-radius:2px"></span>'
                    f'<span style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink);white-space:nowrap">{html.escape(size)}</span>')
    body = (f'<div style="display:grid;grid-template-columns:minmax(0,auto) minmax(0,8rem) auto;gap:0.25rem 0.6rem;align-items:center">'
            f'{"".join(rows)}</div>')
    return yours.figure(w["title"].format(n=number(len(items), seen.lang)), body, alt=w["alt"])


# --- exercise 2: filtered_search -----------------------------------------------------------------------------------------

START_SEARCH = '''def filtered_search(q, store, where, k):
    found = []
    for i in range(len(store)):
        found.append((store[i]["vector"] @ q, i))
    best = sorted(found, reverse=True)[:k]
    return [store[i]["text"] for score, i in best if matches(store[i], where)]
'''
SOLUTION_SEARCH = '''def filtered_search(q, store, where, k):
    found = []
    for i in range(len(store)):
        if matches(store[i], where):
            found.append((store[i]["vector"] @ q, i))
    best = sorted(found, reverse=True)[:k]
    return [store[i]["text"] for score, i in best]
'''
IGNORED = START_SEARCH.replace(' if matches(store[i], where)]', ']')
SEARCH_NAMES = {"matches": matches}
filtered_search = run_code(SOLUTION_SEARCH, dict(SEARCH_NAMES)).names["filtered_search"]


class _Array(np.ndarray):
    """An array that a check message writes the way the lesson's code does: np.array([...]). The learner's code gets
    a plain copy."""

    def __repr__(self):
        return f"np.array({json.dumps(np.asarray(self).tolist())})"

    def __deepcopy__(self, memo):
        return np.array(self)


def _a(values) -> np.ndarray:
    return np.array(values, dtype=float).view(_Array)


class _Named(list):
    """A store that a check message calls by its name, as the lesson shows it. The learner's code gets a plain copy."""

    def __init__(self, name: str, items: list):
        super().__init__(items)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return [copy.deepcopy(item, memo) for item in self]


# six made-up notes with embeddings of two numbers, in English and in Dutch, as the lesson shows them
_NOTES = [("k-means in brief", [1.0, 0.0], "en", "basis"), ("k-means in het kort", [0.98, 0.2], "nl", "basis"),
          ("HNSW in depth", [0.6, 0.8], "en", "expert"), ("HNSW uitgediept", [0.62, 0.78], "nl", "expert"),
          ("Faiss in brief", [0.0, 1.0], "en", "basis"), ("Faiss in het kort", [0.1, 0.99], "nl", "basis")]
NOTES = _Named("NOTES", [{"text": t, "vector": _a(v), "lang": lang, "level": level} for t, v, lang, level in _NOTES])
NOTES_CODE = "NOTES = [\n" + "".join(
    '    {{"text": "{t}", "vector": np.array([{x}, {y}]), "lang": "{lang}", "level": "{level}"}},\n'.format(t=t, x=v[0], y=v[1], lang=lang, level=level)
    for t, v, lang, level in _NOTES) + "]"


_SEARCH_WORDS = {
    "en": {"title": "What your filtered_search finds for: {question}", "where": "Your filtered_search with where = {where}: {n} of {k}",
           "all": "The three best without a filter", "none": "nothing", "right": "the point that answers the question",
           "error": "Your filtered_search stops with an error on the course's store."},
    "nl": {"title": "Wat je filtered_search vindt voor: {question}", "where": "Je filtered_search met where = {where}: {n} van de {k}",
           "all": "De drie beste zonder filter", "none": "niets", "right": "het punt dat de vraag beantwoordt",
           "error": "Je filtered_search stopt met een fout op de store van de course."},
}


def _tag(item: dict) -> str:
    return (f'<span style="font-family:var(--ea-font-code);font-size:11px;color:var(--ea-ink-2);margin-right:0.35rem">'
            f'{html.escape(item["lang"])} · {html.escape(item["domain"])} · {html.escape(item["level"])}</span>')


def _rows(numbers: list, q: dict) -> str:
    rows = []
    for j in numbers:
        item = store()[j]
        right = point_of(j) == gold(q)
        style = "background:var(--ea-data-wash);border-left:3px solid var(--ea-data)" if right else "border-left:3px solid var(--ea-line)"
        rows.append(f'<div style="margin:2px 0;padding:0.1rem 0.4rem;{style};font-family:var(--ea-font-ui);font-size:12px;'
                    f'color:var(--ea-ink);white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{_tag(item)}{html.escape(item["text"])}</div>')
    return "".join(rows)


def _swatch(label: str) -> str:
    return (f'<span style="display:inline-flex;align-items:center;gap:0.25rem;{_LABEL}">'
            f'<span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:var(--ea-data)"></span>{html.escape(label)}</span>')


def searched(fn, qid: str, where: dict, lang: str, asked: str) -> str:
    """The learner's filtered_search on the course's store for one question, asked in language `asked`, with one
    filter: the items it finds, with their metadata, and the three best without a filter."""
    w = _SEARCH_WORDS[lang]
    q = question(qid)
    qv = question_vector(q, asked)
    items = store()
    try:
        got = with_time_limit(lambda: fn(qv.copy(), [dict(item) for item in items], dict(where), K), 10)
    except Exception:   # the learner's code; the check says what went wrong
        return f'<p style="{_LABEL}">{html.escape(w["error"])}</p>'
    if not isinstance(got, list):
        return ""
    texts = [item["text"] for item in items]
    found = [texts.index(t) for t in got[:K] if isinstance(t, str) and t in texts]
    head = w["where"].format(where=where_text(where), n=len(got[:K]), k=K)
    body = (f'<div style="margin:0.3rem 0 0.5rem"><div style="{_LABEL}">{html.escape(head)}</div>'
            f'{_rows(found, q) or html.escape(w["none"])}</div>'
            f'<div style="margin:0.3rem 0 0.5rem"><div style="{_LABEL}">{html.escape(w["all"])}</div>{_rows(order(q, asked)[:K], q)}</div>')
    title = w["title"].format(question=q["text"][asked])
    legend = _swatch(w["right"]) if gold(q) is not None else ""
    return yours.figure(title, body + f"<div>{legend}</div>", alt=title)


def search_picture(seen) -> str:
    """The learner's filtered_search on the course's store, for the question of the worked example, in Dutch, with the
    filter of the worked example."""
    mine = seen.mine
    fn = mine.names.get("filtered_search") if mine is not None else None
    if not callable(fn):
        return ""
    return searched(fn, WORKED, WORKED_WHERE, seen.lang, "nl")


_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}

EXERCISES = {
    "matches": FunctionBox(
        name="matches",
        picture=matches_picture,
        function="matches",
        start=START_MATCHES,
        solution=SOLUTION_MATCHES,
        visible=[(NOTE, {"lang": "en", "level": "basis"})],
        hidden=[(NOTE, {"lang": "nl"}), (NOTE, {}), (NOTE, {"lang": "nl", "level": "expert"}),
                (NOTE, {"lesson": "ml-kmeans", "lang": "nl", "level": "basis"}), (NOTE, {"level": "expert", "lang": "en"})],
        mistakes=[
            Mistake("any", wrong=ANY,
                    en="`{call}` gives `{got}`: your function says yes when one key fits. A filter needs every key of `where` to fit: give `False` as soon as one does not.",
                    nl="`{call}` geeft `{got}`: je functie zegt ja als één key past. Een filter heeft nodig dat elke key van `where` past: geef `False` zodra er één niet past."),
            Mistake("first-key", wrong=FIRST_KEY,
                    en="`{call}` gives `{got}`: your function decides after the first key of `where`, because `return` ends the function in the first round of the loop. Give `False` in the loop only when a key does not fit, and `True` after the loop.",
                    nl="`{call}` geeft `{got}`: je functie beslist na de eerste key van `where`, omdat `return` de functie in de eerste ronde van de loop beëindigt. Geef in de loop alleen `False` als een key niet past, en `True` na de loop."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`matches` says `True` for every item, so nothing is filtered. Go through the keys of `where`, and give `False` as soon as the item has another value for one of them.",
                      "nl": "`matches` zegt `True` voor elk item, dus er wordt niets gefilterd. Ga de keys van `where` langs, en geef `False` zodra het item voor een ervan een andere waarde heeft."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: `True` when the item has the value of `where` for every key of `where`, else `False`.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: `True` als het item voor elke key van `where` de waarde van `where` heeft, anders `False`."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function check every key of `where`, and say `True` when `where` is empty?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Controleert je functie elke key van `where`, en zegt hij `True` als `where` leeg is?"},
            "no_function": {"en": "Your box does not make a function `matches`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `matches`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `matches` is right in all {n} cases: an item fits when every key of `where` fits.",
                        "nl": "Je `matches` klopt in alle {n} gevallen: een item past als elke key van `where` past."},
        },
    ),
    "filtered_search": FunctionBox(
        name="filtered_search",
        picture=search_picture,
        function="filtered_search",
        names=SEARCH_NAMES,
        start=START_SEARCH,
        solution=SOLUTION_SEARCH,
        visible=[(_a([1.0, 0.2]), NOTES, {"lang": "nl"}, 2)],
        hidden=[(_a([1.0, 0.2]), NOTES, {}, 2), (_a([1.0, 0.2]), NOTES, {"lang": "en", "level": "expert"}, 2),
                (_a([1.0, 0.2]), NOTES, {"lang": "de"}, 2), (_a([0.0, 1.0]), NOTES, {"lang": "en", "level": "basis"}, 1),
                (_a([0.0, 1.0]), NOTES, {"lang": "nl"}, 3), (_a([0.0, 1.0]), NOTES, {"lang": "nl"}, 0)],
        mistakes=[
            Mistake("after", wrong=START_SEARCH,
                    en="`{call}` gives `{got}`: your search takes the `k` best of all items first, and only then drops those that do not fit, so fewer than `k` are left. Filter first: give only the items that fit a score.",
                    nl="`{call}` geeft `{got}`: je zoekfunctie neemt eerst de `k` beste van alle items, en laat pas daarna weg wat niet past, dus blijven er minder dan `k` over. Filter eerst: geef alleen de items die passen een score."),
            Mistake("ignored", wrong=IGNORED,
                    en="`{call}` gives `{got}`: the `k` best of all items, also those that do not fit `where`. Give only the items for which `matches(store[i], where)` is `True` a score.",
                    nl="`{call}` geeft `{got}`: de `k` beste van alle items, ook die niet bij `where` passen. Geef alleen de items waarvoor `matches(store[i], where)` `True` is een score."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "The box still takes the `k` best of all items first, and filters afterwards. Move the filter into the loop, so that only the items that fit get a score.",
                      "nl": "Het vak neemt nog eerst de `k` beste van alle items, en filtert daarna. Zet het filter in de loop, zodat alleen de items die passen een score krijgen."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: the texts of the `k` items with the highest dot product with `q`, among the items that fit `where`.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: de teksten van de `k` items met het hoogste inproduct met `q`, tussen de items die bij `where` passen."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work when no item fits, when fewer than `k` fit, and with `k` 0?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Werkt je functie als geen item past, als er minder dan `k` passen, en met `k` 0?"},
            "no_function": {"en": "Your box does not make a function `filtered_search`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `filtered_search`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `filtered_search` is right in all {n} cases: it filters first, so it gives `k` items whenever `k` fit.",
                        "nl": "Je `filtered_search` klopt in alle {n} gevallen: hij filtert eerst, dus hij geeft `k` items zodra er `k` passen."},
        },
    ),
}

# The lesson is done when these are right; "twice" and "access" are choice questions in the lesson.
REQUIRED = ["matches", "filtered_search", "twice", "access"]

CASES = [
    ("matches", START_MATCHES, "empty", None),
    ("matches", "", "empty", None),
    ("matches", SOLUTION_MATCHES, "correct", None),
    ("matches", SOLUTION_MATCHES.replace("def matches(", "def fits("), "wrong", "no-function"),
    ("matches", "def matches(item, where):\n    return all(item[key] == where[key] for key in where)\n", "correct", None),
    ("matches", "def matches(item, where):\n    for key, value in where.items():\n        if item.get(key) != value:\n"
                "            return False\n    return True\n", "correct", None),
    ("matches", ANY, "wrong", "any"),
    ("matches", FIRST_KEY, "wrong", "first-key"),
    ("matches", "def matches(item, where):\n    return any(item[key] == where[key] for key in where)\n", "wrong", "any"),
    ("matches", SOLUTION_MATCHES.replace("    return True\n", "    return len(where) > 0\n"), "almost", None),
    ("matches", "def matches(item, where):\n    return False\n", "almost", None),
    ("filtered_search", START_SEARCH, "empty", None),
    ("filtered_search", "", "empty", None),
    ("filtered_search", SOLUTION_SEARCH, "correct", None),
    ("filtered_search", SOLUTION_SEARCH.replace("def filtered_search(", "def search("), "wrong", "no-function"),
    ("filtered_search", START_SEARCH.replace("    found = []\n", "    store = [item for item in store if matches(item, where)]\n    found = []\n")
                                    .replace(" if matches(store[i], where)]", "]"), "correct", None),
    ("filtered_search", START_SEARCH.replace("[:k]", "[:k + 0]"), "wrong", "after"),
    ("filtered_search", IGNORED, "wrong", "ignored"),
    ("filtered_search", SOLUTION_SEARCH.replace("sorted(found, reverse=True)", "sorted(found)"), "wrong", None),
    ("filtered_search", SOLUTION_SEARCH.replace("[:k]", "[:2]"), "almost", None),
]


# --- what the course measured --------------------------------------------------------------------------------------------

@functools.cache
def languages() -> dict:
    """Per language of the questions, searching the store of both languages without a filter: how many of the three
    best are in the question's language, for how many questions the three hold the same point twice, and for how many
    of the questions a point answers its point is among the three, without and with a filter on the language."""
    out = {}
    for lang in LANGS:
        same = twice = right = right_filtered = 0
        for q in QUESTIONS:
            ranked = order(q, lang)
            top = ranked[:K]
            same += sum(store()[j]["lang"] == lang for j in top)
            twice += len({point_of(j) for j in top}) < K
            if gold(q) is not None:
                right += gold(q) in [point_of(j) for j in top]
                right_filtered += gold(q) in [point_of(j) for j in [j for j in ranked if store()[j]["lang"] == lang][:K]]
        out[lang] = {"same": same, "twice": twice, "right": right, "right_filtered": right_filtered}
    return out


def _fits(where: dict) -> list:
    return [j for j, item in enumerate(store()) if matches(item, where)]


@functools.cache
def afterwards() -> list:
    """For each filter of FILTERS, with the Dutch questions: how many items fit, how many of the three best of all
    items a search that filters afterwards keeps on average, for how many questions it keeps nothing, and for how many
    questions filtering first gives three."""
    out = []
    for where in FILTERS:
        fits = set(_fits(where))
        kept = [sum(j in fits for j in order(q, "nl")[:K]) for q in QUESTIONS]
        first = [len([j for j in order(q, "nl") if j in fits][:K]) for q in QUESTIONS]
        out.append({"where": where, "fits": len(fits), "kept": sum(kept) / len(kept), "nothing": kept.count(0),
                    "three": first.count(K)})
    return out


@functools.cache
def with_index() -> dict:
    """The IVF index of build-vector-index, of the English points, with a filter on domain DOMAIN among the points of
    the clusters it looks in: per number of clusters, 1 and 2, for how many of the 28 English questions it finds nothing,
    and for how many fewer than three."""
    from everything_ai.lesson import lesson
    vi = lesson("build-vector-index", "en").checks
    index = vi.index()
    fits = {i for i in range(len(ABOUT)) if ABOUT[i]["domain"] == DOMAIN}
    out = {"fits": len(fits), "empty_clusters": sum(not fits & set(members) for members in index["lists"])}
    for probes in (1, 2):
        found = []
        for q in QUESTIONS:
            v = question_vector(q, "en")
            near = np.argsort(np.linalg.norm(index["centroids"] - v, axis=1))[:probes]
            found.append(min(K, len([i for c in near for i in index["lists"][c] if i in fits])))
        out[probes] = {"nothing": found.count(0), "fewer": sum(0 < n < K for n in found)}
    return out


def worked() -> dict:
    """The worked example: the three best of all items for the Dutch question, what filtering afterwards keeps of them,
    and what filtering first finds, with the place of each among all items."""
    q = question(WORKED)
    ranked = order(q, "nl")
    fits = set(_fits(WORKED_WHERE))
    first = [j for j in ranked if j in fits][:K]
    return {"question": q, "top": ranked[:K], "kept": [j for j in ranked[:K] if j in fits], "first": first,
            "places": [ranked.index(j) + 1 for j in first], "fits": len(fits)}


TWICE = "q11"   # the question of exercise 3: in English, its three best hold one point in both languages


def twice_scores() -> dict:
    """For the question of exercise 3, asked in English: the score of the point that comes twice, per language."""
    q = question(TWICE)
    top = order(q, "en")[:K]
    point = next(point_of(j) for j in top if [point_of(i) for i in top].count(point_of(j)) == 2)
    v = question_vector(q, "en")
    return {item["lang"]: float(item["vector"] @ v) for j, item in enumerate(store()) if j in top and point_of(j) == point}


def item_title(j: int, lang: str) -> str:
    """The title of the lesson an item of the store comes from."""
    return ABOUT[point_of(j)]["title"][lang]


_TABLE_WORDS = {
    "en": {"title": "Filtering afterwards or first, for the {q} questions in Dutch",
           "where": "where", "fits": "items that fit", "kept": "filter afterwards: kept of {k}, on average",
           "nothing": "filter afterwards: questions with nothing", "first": "filter first: questions with {k}"},
    "nl": {"title": "Achteraf of eerst filteren, voor de {q} vragen in het Nederlands",
           "where": "where", "fits": "items die passen", "kept": "achteraf filteren: gehouden van de {k}, gemiddeld",
           "nothing": "achteraf filteren: vragen met niets", "first": "eerst filteren: vragen met {k}"},
}


def table_picture(lang: str) -> str:
    """A row per filter of FILTERS: how many items fit, what a search that filters afterwards keeps, and what
    filtering first gives."""
    w = _TABLE_WORDS[lang]
    cell = "padding:0.25rem 0.5rem;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink)"
    head = "".join(f'<th style="{cell};color:var(--ea-ink-2);font-weight:600;text-align:left">{html.escape(t)}</th>'
                   for t in (w["where"], w["fits"], w["kept"].format(k=K), w["nothing"], w["first"].format(k=K)))
    rows = []
    for r in afterwards():
        loss = "color:var(--ea-loss-text);font-weight:600" if r["nothing"] else ""
        rows.append(f'<tr><td style="{cell};font-family:var(--ea-font-code)">{html.escape(where_text(r["where"]))}</td>'
                    f'<td style="{cell}">{number(r["fits"], lang)}</td><td style="{cell}">{number(r["kept"], lang, 1)}</td>'
                    f'<td style="{cell};{loss}">{r["nothing"]}</td><td style="{cell}">{r["three"]}</td></tr>')
    body = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;min-width:100%">'
            f'<thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    title = w["title"].format(q=len(QUESTIONS))
    return yours.figure(title, body, alt=title)


# --- try it: your filtered_search on a question of the course ------------------------------------------------------------

def learner_search(matches_text, search_text):
    """The learner's filtered_search, with the learner's matches, when both exercises are right; else None."""
    for name, text in (("matches", matches_text), ("filtered_search", search_text)):
        if text is MISSING or text is None or EXERCISES[name].check(text).state != "correct":
            return None
    mine = run_code(matches_text).names["matches"]
    return run_code(search_text, {"matches": mine}).names["filtered_search"]


def question_options(lang: str) -> dict:
    return {q["text"][lang]: q["id"] for q in QUESTIONS}


def domains() -> list:
    """The domains of the points, the one with the most points first."""
    counts: dict = {}
    for a in ABOUT:
        counts[a["domain"]] = counts.get(a["domain"], 0) + 1
    return sorted(counts, key=lambda d: (-counts[d], d))


LEVELS = ("basis", "gevorderd", "expert")


def where_of(lang: str, domain: str, level: str) -> dict:
    """The filter the dropdowns of the search to try make: a key for each that is not "any" ("" here)."""
    return {key: value for key, value in (("lang", lang), ("domain", domain), ("level", level)) if value}
