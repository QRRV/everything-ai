# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-zipf-heaps: count the words of a text by rank (rank_counts), multiply each count by its
rank with NumPy (rank_times_count), follow how many different words a text holds as it gets longer (growth), and sort
statements under Zipf's law or Heaps' law.

The corpora are the course's own: the eight books of Project Gutenberg in data/gutenberg-books.json.gz (five English,
three Dutch with their old spelling; made by data/tools/make_gutenberg.py) and the Dutch sentences of Tatoeba in
data/tatoeba-nl.tsv.gz (CC BY 2.0 FR, Tatoeba and its contributors; made by data/tools/make_tatoeba_nl.py).
everything_ai.books cuts both into words. Every number in the pictures is counted by the course; the cases of the
checks and the statements of the sort are the course's own."""
import contextlib
import copy
import html
import io
import math

import numpy as np

from everything_ai import books, chart, yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, with_time_limit

# --- the corpora -----------------------------------------------------------------------------------------------------

ENGLISH = [k for k in books.keys() if books.book(k)["lang"] == "en"]
DUTCH = [k for k in books.keys() if books.book(k)["lang"] == "nl"]


def corpus_words(key: str) -> list:
    """The words of a corpus of the course, as a new list: "english" is the five English books one after another,
    "dutch-books" the three Dutch books, "tatoeba" the Dutch sentences of Tatoeba, and a book's key that book."""
    if key == "english":
        return [w for k in ENGLISH for w in books.book_words(k)]
    if key == "dutch-books":
        return [w for k in DUTCH for w in books.book_words(k)]
    if key == "tatoeba":
        return books.dutch_sentence_words()
    return books.book_words(key)


def book_words(key: str) -> list:
    """The words of one of the course's books, as a new list (as in nlp-corpora)."""
    return books.book_words(key)


# --- the course's own answers ----------------------------------------------------------------------------------------


def rank_counts(words):
    counts = {}
    for w in words:
        w = w.lower()
        if w in counts:
            counts[w] = counts[w] + 1
        else:
            counts[w] = 1
    return sorted(counts.values(), reverse=True)


def rank_times_count(counts):
    ranks = np.arange(1, len(counts) + 1)
    return ranks * counts


def growth(words, step):
    seen = set()
    points = []
    for i in range(len(words)):
        seen.add(words[i].lower())
        if (i + 1) % step == 0:
            points.append((i + 1, len(seen)))
    return points


def counts_by_rank(words) -> list:
    """The counts of the different words of a list, from the most to the least common, with the words in small
    letters: what rank_counts gives, for the cells of the lesson that need it before the learner's own works."""
    return rank_counts(words)


def ranked_words(words) -> list:
    """(word, count) of every different word in small letters, from the most to the least common; words with the same
    count in the order in which the text first uses them."""
    counts = {}
    for w in words:
        w = w.lower()
        counts[w] = counts.get(w, 0) + 1
    return sorted(counts.items(), key=lambda p: -p[1])   # sorted keeps the order of the dict among equal counts


def corpus_names(lang: str) -> dict:
    """{name: key} of the corpora of the demo, in the language of the lesson."""
    w = _CORPUS_NAMES[lang]
    out = {w["english"]: "english", w["dutch-books"]: "dutch-books", w["tatoeba"]: "tatoeba"}
    for b in books.books():
        out[f'{b["title"]} ({b["year"]})'] = b["key"]
    return out


_TITLE_NAMES = {
    "en": {"english": "the five English books", "dutch-books": "the three Dutch books", "tatoeba": "Tatoeba's Dutch sentences"},
    "nl": {"english": "de vijf Engelse boeken", "dutch-books": "de drie Nederlandse boeken",
           "tatoeba": "de Nederlandse zinnen van Tatoeba"},
}
_CORPUS_NAMES = {
    "en": {"english": "The five English books", "dutch-books": "The three Dutch books (old spelling)",
           "tatoeba": "Tatoeba: Dutch of today"},
    "nl": {"english": "De vijf Engelse boeken", "dutch-books": "De drie Nederlandse boeken (oude spelling)",
           "tatoeba": "Tatoeba: Nederlands van nu"},
}


# --- numbers and words in the pictures -------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums"
_OFF = ";color:var(--ea-loss-text);font-weight:600"
# words that the tables of the course leave out: the course names nobody by a gendered pronoun, so a table moves on to
# the next rank (as nlp-corpora keeps them out of its example sentences)
_PRONOUNS = {"he", "she", "him", "her", "his", "hers", "himself", "herself", "hij", "zij", "haar", "hem", "zijn", "ze",
             "zich"}


def _note(text: str) -> str:
    """A line under a picture; a word between asterisks is set in italics, as in the lesson's text."""
    parts = html.escape(text).split("*")
    marked = "".join(f"<i>{p}</i>" if k % 2 else p for k, p in enumerate(parts))
    return f'<div style="{_UI}">{marked}</div>'


def _bare(text: str) -> str:
    return text.replace("*", "")


def _num(n, lang: str) -> str:
    """A whole number with a separator for thousands: 26,776 in English, 26.776 in Dutch."""
    text = f"{round(n):,}"
    return text.replace(",", ".") if lang == "nl" else text


def _dec(x: float, lang: str, digits: int = 2) -> str:
    """A number with a fixed number of decimals: 0.64 in English, 0,64 in Dutch."""
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def _pct(x: float, lang: str) -> str:
    return _dec(100 * x, lang, 1) + "%"


def _svg(panels: list, alt: str, lang: str, width: int = 330, height: int = 250) -> str:
    svg = chart.figure(panels, alt=alt, width=width, height=height)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


def _log_sample(n: int, k: int = 240) -> list:
    """About k whole numbers from 1 to n at equal factors apart, for a line on a log scale: the first ranks one by one,
    later ones far apart."""
    if n <= 0:
        return []
    return sorted({max(1, min(n, round(10 ** (j * math.log10(n) / (k - 1))))) for j in range(k)})


def _beta(points: dict, a: int, b: int):
    """The exponent of Heaps' law from two points (length, different words): the slope on a log-log plot."""
    if a in points and b in points and points[a] > 0 and points[b] > 0:
        return (math.log10(points[b]) - math.log10(points[a])) / (math.log10(b) - math.log10(a))
    return None


# --- the idea picture ------------------------------------------------------------------------------------------------

_TWO = ("english", "tatoeba")

_IDEA_WORDS = {
    "en": {"title": "The ten most common words, and what Zipf's law predicts",
           "names": {"english": "Five English books", "tatoeba": "Tatoeba: Dutch of today"},
           "y": "times in the text",
           "legend": "The bars are how often each word occurs, counted by the course in small letters. The open dots "
                     "are what Zipf's law predicts from the first word: half as often for the second, a third as often "
                     "for the third, and so on.",
           "alt": "{name}: {pairs}."},
    "nl": {"title": "De tien woorden die het vaakst voorkomen, en wat de wet van Zipf voorspelt",
           "names": {"english": "Vijf Engelse boeken", "tatoeba": "Tatoeba: Nederlands van nu"},
           "y": "keer in de tekst",
           "legend": "De staven zijn hoe vaak elk woord voorkomt, geteld door de course in kleine letters. De open "
                     "punten zijn wat de wet van Zipf voorspelt vanuit het eerste woord: half zo vaak voor het tweede, "
                     "een derde zo vaak voor het derde, enzovoort.",
           "alt": "{name}: {pairs}."},
}


def _top(key: str) -> list:
    return ranked_words(corpus_words(key))


def idea_picture(lang: str) -> str:
    w = _IDEA_WORDS[lang]
    panels, alts = [], []
    for key in _TWO:
        top = _top(key)[:10]
        first = top[0][1]
        high = max(first, max(c for _, c in top)) * 1.08
        p = chart.Panel((0.4, 10.6), (0, high), title=w["names"][key], ylabel=w["y"],
                        xticks=list(range(1, 11)), xtick_labels=[t for t, _ in top])
        for r, (_, c) in enumerate(top, start=1):
            p.rect(r - 0.32, 0, r + 0.32, c, role="data", solid=True)
        p.points(list(range(1, 11)), [first / r for r in range(1, 11)], role="ink", r=4.2, hollow=True)
        panels.append(p)
        pairs = "; ".join(f"{t} {_num(c, lang)} ({_num(first / r, lang)})" for r, (t, c) in enumerate(top, start=1))
        alts.append(w["alt"].format(name=w["names"][key], pairs=pairs))
    alt = f'{w["title"]}. ' + " ".join(alts) + " " + w["legend"]
    return yours.figure(w["title"], _svg(panels, alt, lang, height=260) + _note(w["legend"]), alt=alt)


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


def _is_counts(v) -> bool:
    return isinstance(v, list) and all(type(c) is int and c > 0 for c in v)


# --- the picture under exercise 1 ------------------------------------------------------------------------------------

_RANK_WORDS = {
    "en": {"title": "Your rank_counts, at work: every word of two corpora",
           "names": {"english": "Five English books", "tatoeba": "Tatoeba: Dutch of today"},
           "x": "rank", "y": "count",
           "fine": "Your function counts {v_en} different words in the five English books and {v_nl} in the Dutch "
                   "sentences of Tatoeba. The dashed line falls as 1 over the rank, through the word at rank 10: "
                   "Zipf's law. On these log scales your counts follow it roughly, and fall faster near the end.",
           "same": "Your counts for {name} differ from the right ones, which start with {start}. Where your line differs, "
                   "it is purple.",
           "differs": "Your function gives {got} counts for {name}, and the right list has {n}. Where your line "
                      "differs, it is purple.",
           "error": "Your function stops with an error on the words of {name}.",
           "slow": "Your function takes too long on the {n} words of {name}.",
           "not_counts": "Your function does not give back a list of whole numbers above 0 for {name}, so there is "
                         "nothing to draw yet."},
    "nl": {"title": "Jouw rank_counts aan het werk: elk woord van twee corpora",
           "names": {"english": "Vijf Engelse boeken", "tatoeba": "Tatoeba: Nederlands van nu"},
           "x": "rang", "y": "aantal",
           "fine": "Je functie telt {v_en} verschillende woorden in de vijf Engelse boeken en {v_nl} in de "
                   "Nederlandse zinnen van Tatoeba. De gestippelde lijn daalt als 1 gedeeld door de rang, door het "
                   "woord op rang 10: de wet van Zipf. Op deze logaritmische schalen volgen je aantallen die lijn "
                   "ongeveer, en tegen het eind dalen de aantallen sneller.",
           "same": "Je aantallen voor {name} wijken af van de goede, die beginnen met {start}. Waar je lijn afwijkt, is "
                   "die paars.",
           "differs": "Je functie geeft {got} aantallen voor {name}, en de goede lijst heeft er {n}. Waar je lijn "
                      "afwijkt, is die paars.",
           "error": "Je functie stopt met een fout op de woorden van {name}.",
           "slow": "Je functie doet er te lang over op de {n} woorden van {name}.",
           "not_counts": "Je functie geeft voor {name} geen lijst gehele getallen boven 0 terug, dus er valt nog niets "
                         "te tekenen."},
}


def rank_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _RANK_WORDS[lang]
    panels, sizes, notes = [], {}, []
    for key in _TWO:
        name, title = _TITLE_NAMES[lang][key], w["names"][key]
        words = corpus_words(key)
        try:
            got = _call(fn, words)
        except Timeout:
            notes.append(w["slow"].format(n=_num(len(words), lang), name=name))
            continue
        if got is _STOPPED:
            notes.append(w["error"].format(name=name))
            continue
        if not _is_counts(got) or not got:
            notes.append(w["not_counts"].format(name=name))
            continue
        want = rank_counts(words)
        sizes[key] = len(got)
        n = max(len(got), len(want))
        ranks = _log_sample(len(got))
        anchor = want[9] * 10 if len(want) >= 10 else want[0]
        p = chart.Panel((1, max(n, 10) * 1.3), (0.7, max(got[0], want[0], anchor) * 1.6), title=title, xlabel=w["x"],
                        ylabel=w["y"], xlog=True, ylog=True)
        p.line([1, len(want)], [anchor, anchor / len(want)], role="ink-3", width=1.6, dashed=True)
        role = "data" if got == want else "loss"
        p.line(ranks, [got[r - 1] for r in ranks], role=role, width=2.2)
        panels.append(p)
        if got != want:
            if len(got) == len(want):
                notes.append(w["same"].format(name=name, start=", ".join(_num(c, lang) for c in want[:3])))
            else:
                notes.append(w["differs"].format(got=_num(len(got), lang), n=_num(len(want), lang), name=name))
    if len(sizes) == 2 and not notes:
        notes.append(w["fine"].format(v_en=_num(sizes["english"], lang), v_nl=_num(sizes["tatoeba"], lang)))
    text = " ".join(notes)
    if not panels:
        return yours.figure(w["title"], _note(text), alt=f'{w["title"]}. {_bare(text)}')
    alt = f'{w["title"]}. ' + _bare(text)
    return yours.figure(w["title"], _svg(panels, alt, lang) + _note(text), alt=alt)


# --- the picture under exercise 2 ------------------------------------------------------------------------------------

_PRODUCT_WORDS = {
    "en": {"title": "Your rank_times_count, at work: rank × count down the list",
           "names": {"english": "Five English books", "tatoeba": "Tatoeba: Dutch of today"},
           "x": "rank", "y": "number", "count": "count", "product": "rank × count",
           "fine": "{name}: from rank 1 to rank {last}, the count falls from {top} to 1, a factor of {top}. Rank × count "
                   "stays between {low} and {high}: a factor of {spread}.",
           "differs": "For {name}, your function gives something other than rank × count at {n} of the ranks; those "
                      "points are purple.",
           "error": "Your function stops with an error on the counts of {name}.",
           "slow": "Your function takes too long on the counts of {name}.",
           "not_numbers": "Your function does not give back one number for every rank of {name}, so there is nothing "
                          "to draw yet."},
    "nl": {"title": "Jouw rank_times_count aan het werk: rang × aantal, de lijst af",
           "names": {"english": "Vijf Engelse boeken", "tatoeba": "Tatoeba: Nederlands van nu"},
           "x": "rang", "y": "getal", "count": "aantal", "product": "rang × aantal",
           "fine": "{name}: van rang 1 tot rang {last} daalt het aantal van {top} naar 1, een factor {top}. Rang × "
                   "aantal blijft tussen {low} en {high}: een factor {spread}.",
           "differs": "Voor {name} geeft je functie bij {n} van de rangen iets anders dan rang × aantal; die punten "
                      "zijn paars.",
           "error": "Je functie stopt met een fout op de aantallen van {name}.",
           "slow": "Je functie doet er te lang over op de aantallen van {name}.",
           "not_numbers": "Je functie geeft niet voor elke rang van {name} één getal terug, dus er valt nog niets te "
                          "tekenen."},
}


def _numbers(v, n: int):
    """The learner's outcome as a list of n floats, or None."""
    try:
        values = [float(x) for x in np.asarray(v, dtype=float).reshape(-1)]
    except (TypeError, ValueError):
        return None
    return values if len(values) == n and np.asarray(v).ndim == 1 else None


def product_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _PRODUCT_WORDS[lang]
    panels, notes = [], []
    for key in _TWO:
        name, title = _TITLE_NAMES[lang][key], w["names"][key]
        counts = np.array(rank_counts(corpus_words(key)))
        try:
            got = _call(fn, counts)
        except Timeout:
            notes.append(w["slow"].format(name=name))
            continue
        if got is _STOPPED:
            notes.append(w["error"].format(name=name))
            continue
        values = _numbers(got, len(counts))
        if values is None:
            notes.append(w["not_numbers"].format(name=name))
            continue
        want = rank_times_count(counts)
        ranks = _log_sample(len(counts))
        p = chart.Panel((1, len(counts) * 1.3), (0.7, max(max(want), counts[0]) * 3), title=title, xlabel=w["x"],
                        ylabel=w["y"], xlog=True, ylog=True)
        p.line(ranks, [counts[r - 1] for r in ranks], role="ink-3", width=1.6)
        p.text(2.2, counts[1] * 0.42, w["count"], role="ink-3", anchor="start", halo=True)
        off = [r for r in ranks if not math.isclose(values[r - 1], want[r - 1], rel_tol=1e-6)]
        good = [r for r in ranks if r not in off and values[r - 1] > 0]
        p.line(good, [values[r - 1] for r in good], role="data", width=2.2)
        p.text(2.2, max(want) * 1.9, w["product"], role="data", anchor="start", halo=True)
        if off:
            shown = [r for r in off if values[r - 1] > 0]
            p.points(shown, [values[r - 1] for r in shown], role="loss", r=2.6)
            wrong = sum(1 for got_v, want_v in zip(values, want) if not math.isclose(got_v, want_v, rel_tol=1e-6))
            notes.append(w["differs"].format(name=name, n=_num(wrong, lang)))
        else:
            low, high = min(want), max(want)
            notes.append(w["fine"].format(name=title, last=_num(len(counts), lang), top=_num(counts[0], lang),
                                          low=_num(low, lang), high=_num(high, lang),
                                          spread=_dec(high / low, lang, 1)))
        panels.append(p)
    text = " ".join(notes)
    if not panels:
        return yours.figure(w["title"], _note(text), alt=f'{w["title"]}. {_bare(text)}')
    alt = f'{w["title"]}. ' + _bare(text)
    return yours.figure(w["title"], _svg(panels, alt, lang) + _note(text), alt=alt)


# --- the picture under exercise 3 ------------------------------------------------------------------------------------

STEP = 1000   # words between two points of the picture under exercise 3

_GROWTH_WORDS = {
    "en": {"title": "Your growth, at work: new words keep coming",
           "names": {"english": "five English books", "tatoeba": "Tatoeba"},
           "normal": "Normal axes", "loglog": "Log-log axes", "x": "words read", "y": "different words",
           "fine": "After 10,000 words, your function counts {a_en} different words in the English books and {a_nl} in "
                   "the Dutch sentences; after 100,000 words, {b_en} and {b_nl}. The curves bend on normal axes and "
                   "run almost straight on log-log axes: a power law. Its exponent β, the slope between 10,000 and "
                   "100,000 words, is {beta_en} for English and {beta_nl} for Dutch.",
           "differs": "For {name}, your points differ from the right ones: the right list has {n} points, the "
                      "first {first}.",
           "error": "Your function stops with an error on the words of {name}.",
           "slow": "Your function takes too long on the {n} words of {name}. Does it go through the words once, "
                   "adding each to one set?",
           "not_points": "Your function does not give back a list of pairs (words read, different words) for "
                         "{name}, so there is nothing to draw yet."},
    "nl": {"title": "Jouw growth aan het werk: er komen steeds nieuwe woorden bij",
           "names": {"english": "vijf Engelse boeken", "tatoeba": "Tatoeba"},
           "normal": "Gewone assen", "loglog": "Log-log-assen", "x": "gelezen woorden", "y": "verschillende woorden",
           "fine": "Na 10.000 woorden telt je functie {a_en} verschillende woorden in de Engelse boeken en {a_nl} in "
                   "de Nederlandse zinnen; na 100.000 woorden {b_en} en {b_nl}. Op gewone assen buigen de lijnen af, "
                   "op log-log-assen lopen de lijnen bijna recht: een machtswet. De exponent β, de helling tussen 10.000 en "
                   "100.000 woorden, is {beta_en} voor het Engels en {beta_nl} voor het Nederlands.",
           "differs": "Voor {name} wijken je punten af van de goede: de goede lijst heeft {n} punten, het eerste "
                      "{first}.",
           "error": "Je functie stopt met een fout op de woorden van {name}.",
           "slow": "Je functie doet er te lang over op de {n} woorden van {name}. Gaat de functie de woorden één keer "
                   "langs, en stopt de functie elk woord in één set?",
           "not_points": "Je functie geeft voor {name} geen lijst paren (gelezen woorden, verschillende woorden) "
                         "terug, dus er valt nog niets te tekenen."},
}


def _is_points(v) -> bool:
    return isinstance(v, list) and all(isinstance(p, tuple) and len(p) == 2 and all(type(x) is int for x in p)
                                       for p in v)


def growth_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _GROWTH_WORDS[lang]
    got_points, notes = {}, []
    for key in _TWO:
        name = _TITLE_NAMES[lang][key]
        words = corpus_words(key)
        try:
            got = _call(fn, words, STEP)
        except Timeout:
            notes.append(w["slow"].format(n=_num(len(words), lang), name=name))
            continue
        if got is _STOPPED:
            notes.append(w["error"].format(name=name))
            continue
        if not _is_points(got) or not got:
            notes.append(w["not_points"].format(name=name))
            continue
        want = growth(words, STEP)
        if got != want:
            first = f"({_num(want[0][0], lang)}, {_num(want[0][1], lang)})"
            notes.append(w["differs"].format(name=name, n=_num(len(want), lang), first=first))
        got_points[key] = (got, got == want)
    if not got_points:
        text = " ".join(notes)
        return yours.figure(w["title"], _note(text), alt=f'{w["title"]}. {_bare(text)}')
    longest = max(p[-1][0] for p, _ in got_points.values())
    tallest = max(max(v for _, v in p) for p, _ in got_points.values())
    normal = chart.Panel((0, longest), (0, tallest * 1.12), title=w["normal"], xlabel=w["x"], ylabel=w["y"])
    loglog = chart.Panel((STEP * 0.8, longest * 2.2), (100, tallest * 3), title=w["loglog"], xlabel=w["x"],
                         ylabel=w["y"], xlog=True, ylog=True)
    for key, (pts, right) in got_points.items():
        role = "data" if right else "loss"
        thin = pts[::max(1, len(pts) // 200)] + [pts[-1]]
        normal.line([n for n, _ in thin], [v for _, v in thin], role=role, width=2.2 if key == "english" else 1.8,
                    dashed=key == "tatoeba")
        idx = sorted({i - 1 for i in _log_sample(len(pts), 120)})
        loglog.line([pts[i][0] for i in idx], [pts[i][1] for i in idx], role=role,
                    width=2.2 if key == "english" else 1.8, dashed=key == "tatoeba")
        n_end, v_end = pts[-1]
        if key == "english":   # under the end of its line, where the dashed line of Tatoeba runs above it
            normal.text(n_end + longest * 0.03, v_end - tallest * 0.1, w["names"][key], role="ink-2", anchor="start",
                        halo=True)
        else:
            normal.text(n_end, v_end + tallest * 0.06, w["names"][key], role="ink-2", anchor="end", halo=True)
    if len(got_points) == 2 and all(right for _, right in got_points.values()):
        at = {key: dict(pts) for key, (pts, _) in got_points.items()}
        notes.append(w["fine"].format(
            a_en=_num(at["english"][10_000], lang), a_nl=_num(at["tatoeba"][10_000], lang),
            b_en=_num(at["english"][100_000], lang), b_nl=_num(at["tatoeba"][100_000], lang),
            beta_en=_dec(_beta(at["english"], 10_000, 100_000), lang),
            beta_nl=_dec(_beta(at["tatoeba"], 10_000, 100_000), lang)))
    text = " ".join(notes)
    alt = f'{w["title"]}. ' + _bare(text)
    return yours.figure(w["title"], _svg([normal, loglog], alt, lang) + _note(text), alt=alt)


# --- the demo --------------------------------------------------------------------------------------------------------

_DEMO_WORDS = {
    "en": {"title": "The words of {name}",
           "sizes": "{words} words, {types} of them different. {hapax} different words occur only once: {hapax_share} "
                    "of them. The {half} most common words make up half of the text.",
           "beta": "From 10,000 to 100,000 words, the number of different words grows with the exponent β = {beta}.",
           "short": "The text is shorter than 100,000 words, so the exponent β is worked out from 1,000 to 10,000 "
                    "words: {beta}.",
           "zipf": "Rank and count", "heaps": "New words keep coming",
           "rank": "rank", "count": "count", "read": "words read", "different": "different words",
           "head": ["Rank", "Word", "Count", "Rank × count"],
           "legend": "The dashed line falls as 1 over the rank, through the word at rank 10. The table gives the words "
                     "at ranks 1, 10, 100, 1,000 and 10,000, in small letters."},
    "nl": {"title": "De woorden van {name}",
           "sizes": "{words} woorden, waarvan {types} verschillende. {hapax} verschillende woorden komen maar één keer "
                    "voor: {hapax_share} ervan. De {half} woorden die het vaakst voorkomen, maken samen de helft van de "
                    "tekst uit.",
           "beta": "Van 10.000 tot 100.000 woorden groeit het aantal verschillende woorden met de exponent β = {beta}.",
           "short": "De tekst is korter dan 100.000 woorden, dus de exponent β is berekend van 1.000 tot 10.000 "
                    "woorden: {beta}.",
           "zipf": "Rang en aantal", "heaps": "Steeds nieuwe woorden",
           "rank": "rang", "count": "aantal", "read": "gelezen woorden", "different": "verschillende woorden",
           "head": ["Rang", "Woord", "Aantal", "Rang × aantal"],
           "legend": "De gestippelde lijn daalt als 1 gedeeld door de rang, door het woord op rang 10. De tabel geeft "
                     "de woorden op rang 1, 10, 100, 1.000 en 10.000, in kleine letters."},
}
_PRONOUN_NOTE = {
    "en": "Where a pronoun that names a person by gender stands at such a rank, the table gives the next rank.",
    "nl": "Waar een voornaamwoord met een geslacht op zo'n rang staat, geeft de tabel de rang erna.",
}
TABLE_RANKS = (1, 10, 100, 1000, 10_000)


def _demo_numbers(key: str) -> dict:
    words = corpus_words(key)
    ranked = ranked_words(words)
    counts = [c for _, c in ranked]
    total, half, cum = len(words), 0, 0
    for i, c in enumerate(counts):
        cum += c
        if 2 * cum >= total:
            half = i + 1
            break
    return {"words": total, "ranked": ranked, "counts": counts, "half": half,
            "hapax": sum(1 for c in counts if c == 1), "growth": growth(words, 1000)}


_DEMO_CACHE: dict = {}


def demo_picture(key: str, lang: str) -> str:
    names = {v: k for k, v in corpus_names(lang).items()}
    key = key if key in names else "english"
    if key not in _DEMO_CACHE:
        _DEMO_CACHE[key] = _demo_numbers(key)
    d = _DEMO_CACHE[key]
    w = _DEMO_WORDS[lang]
    name = _TITLE_NAMES[lang].get(key) or names[key]
    title = w["title"].format(name=name)
    counts, ranked = d["counts"], d["ranked"]
    sizes = w["sizes"].format(words=_num(d["words"], lang), types=_num(len(counts), lang), hapax=_num(d["hapax"], lang),
                              hapax_share=_pct(d["hapax"] / len(counts), lang), half=_num(d["half"], lang))
    at = dict(d["growth"])
    if 100_000 in at:
        beta = w["beta"].format(beta=_dec(_beta(at, 10_000, 100_000), lang))
    else:
        beta = w["short"].format(beta=_dec(_beta(at, 1000, 10_000), lang))
    ranks = _log_sample(len(counts))
    zipf = chart.Panel((1, len(counts) * 1.3), (0.7, counts[0] * 1.6), title=w["zipf"], xlabel=w["rank"],
                       ylabel=w["count"], xlog=True, ylog=True)
    anchor = counts[9] * 10 if len(counts) >= 10 else counts[0]
    zipf.line([1, len(counts)], [anchor, anchor / len(counts)], role="ink-3", width=1.6, dashed=True)
    zipf.line(ranks, [counts[r - 1] for r in ranks], role="data", width=2.2)
    pts = d["growth"]
    idx = sorted({i - 1 for i in _log_sample(len(pts), 120)})
    heaps = chart.Panel((800, pts[-1][0] * 2.2), (100, pts[-1][1] * 3), title=w["heaps"], xlabel=w["read"],
                        ylabel=w["different"], xlog=True, ylog=True)
    heaps.line([pts[i][0] for i in idx], [pts[i][1] for i in idx], role="data", width=2.2)
    rows, skipped = "", False
    for r in TABLE_RANKS:
        if r > len(ranked):
            break
        k = r
        while k <= len(ranked) and ranked[k - 1][0] in _PRONOUNS:
            k += 1
        skipped = skipped or k != r
        if k > len(ranked):
            break
        word, c = ranked[k - 1]
        rows += (f'<tr><td style="{_CELL}{_NUMBER}">{_num(k, lang)}</td><td style="{_CELL}">{html.escape(word)}</td>'
                 f'<td style="{_CELL}{_NUMBER}">{_num(c, lang)}</td><td style="{_CELL}{_NUMBER}">{_num(k * c, lang)}</td>'
                 f'</tr>')
    head = "".join(f'<th style="{_HEAD}">{h}</th>' for h in w["head"])
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.4rem 0">'
             f'<tr>{head}</tr>{rows}</table></div>')
    legend = w["legend"] + (" " + _PRONOUN_NOTE[lang] if skipped else "")
    alt = f"{title}. {sizes} {beta} {legend}"
    body = _note(sizes) + _note(beta) + _svg([zipf, heaps], alt, lang) + table + _note(legend)
    return yours.figure(title, body, alt=alt)


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["rank-counts", "rank-times-count", "growth", "laws"]

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
    """A short value the way the lessons write it: strings in double quotes, also inside a tuple or a list."""
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, tuple):
        return "(" + ", ".join(_value(p) for p in v) + ("," if len(v) == 1 else "") + ")"
    if isinstance(v, list):
        return "[" + ", ".join(_value(p) for p in v) + "]"
    if isinstance(v, np.ndarray):
        return "array([" + ", ".join(_value(p.item()) for p in v.reshape(-1)) + "])"
    if isinstance(v, (np.integer, np.floating)):
        return _value(v.item())
    return repr(v)


def _show_list(v, what_en: str, what_nl: str, first: int = 3):
    """{got} and {want}: a long list as its length and its first values."""
    if isinstance(v, list) and len(v) > 6:
        start = ", ".join(_value(p) for p in v[:first])
        return {"en": f"a list of {_num(len(v), 'en')} {what_en}, starting with {start}",
                "nl": f"een lijst van {_num(len(v), 'nl')} {what_nl}, die begint met {start}"}
    return _value(v)


def _show_counts(v):
    return _show_list(v, "counts", "aantallen")


def _show_products(v):
    if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) > 6:
        start = ", ".join(_value(p.item()) for p in v[:3])
        return {"en": f"an array of {_num(len(v), 'en')} numbers, starting with {start}",
                "nl": f"een array van {_num(len(v), 'nl')} getallen, die begint met {start}"}
    return _show_list(v, "numbers", "getallen")


def _show_points(v):
    return _show_list(v, "points", "punten", first=2)


class _Words:
    """The words of a corpus of the course, made when a check first needs them, and shown in a message by a name."""

    def __init__(self, name: str, key: str):
        self.name, self.key = name, key

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return corpus_words(self.key)


class _Counts:
    """The counts by rank of a book as a NumPy array, made when a check first needs them, shown by a name."""

    def __init__(self, name: str, key: str):
        self.name, self.key = name, key

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return np.array(rank_counts(corpus_words(self.key)))


ALICE_N = _Words("ALICE", "alice")
COUNTS_N = _Counts("COUNTS", "alice")

# exercise 1: rank_counts. The cases of the check's own hold a word with capitals, words that occur equally often, a
# list of one word and an empty list.
RANK_CASES = [
    (ALICE_N,),
    (["the", "cat", "saw", "the", "dog"],),
    (["The", "the", "THE", "end"],),
    (["b", "a", "b", "c", "c", "c"],),
    (["one", "two", "three"],),
    (["word"],),
    ([],),
]
_R_START = "def rank_counts(words):\n    ...\n"
_R = ("def rank_counts(words):\n"
      "    counts = {}\n"
      "    for w in words:\n"
      "        w = w.lower()\n"
      "        if w in counts:\n"
      "            counts[w] = counts[w] + 1\n"
      "        else:\n"
      "            counts[w] = 1\n"
      "    return sorted(counts.values(), reverse=True)\n")
_R_CASE = _R.replace("        w = w.lower()\n", "")
_R_UP = _R.replace("sorted(counts.values(), reverse=True)", "sorted(counts.values())")
_R_ZERO = _R.replace("            counts[w] = 1\n", "            counts[w] = 0\n")


def _a_dict(got, text) -> bool:
    return isinstance(got, dict)


def _words_not_counts(got, text) -> bool:
    return isinstance(got, list) and bool(got) and all(isinstance(v, str) for v in got)


def _pairs_not_counts(got, text) -> bool:
    return isinstance(got, list) and bool(got) and all(isinstance(v, tuple) for v in got)


def _not_list(got, text) -> bool:
    return not isinstance(got, (list, dict))


# exercise 2: rank_times_count. The cases of the check's own are short arrays, with whole numbers as rank_counts
# gives them; one follows Zipf's law exactly, and one is empty.
PRODUCT_CASES = [
    (COUNTS_N,),
    (np.array([6, 3, 2]),),
    (np.array([12, 6, 4, 3]),),
    (np.array([5, 5, 1, 1]),),
    (np.array([7]),),
    (np.array([], dtype=int),),
]
_T_START = "import numpy as np\n\n\ndef rank_times_count(counts):\n    ...\n"
_T = ("import numpy as np\n\n\n"
      "def rank_times_count(counts):\n"
      "    ranks = np.arange(1, len(counts) + 1)\n"
      "    return ranks * counts\n")
_T_ZERO = _T.replace("np.arange(1, len(counts) + 1)", "np.arange(len(counts))")
_T_DIVIDE = _T.replace("return ranks * counts", "return counts / ranks")


def _broadcast(e) -> bool:
    return isinstance(e, ValueError) and "broadcast" in str(e)


def _one_number(got, text) -> bool:
    return isinstance(got, (int, float, np.integer, np.floating)) or (isinstance(got, np.ndarray) and got.ndim == 0)


# exercise 3: growth. The cases of the check's own hold words with capitals, a step that does not fit the length, a
# step longer than the list, and an empty list.
GROWTH_CASES = [
    (ALICE_N, 5000),
    (["the", "cat", "saw", "the", "dog"], 2),
    (["A", "a", "b"], 1),
    (["to", "be", "or", "not", "to", "be"], 3),
    (["x", "y", "z"], 5),
    ([], 3),
]
_G_START = "def growth(words, step):\n    ...\n"
_G = ("def growth(words, step):\n"
      "    seen = set()\n"
      "    points = []\n"
      "    for i in range(len(words)):\n"
      "        seen.add(words[i].lower())\n"
      "        if (i + 1) % step == 0:\n"
      "            points.append((i + 1, len(seen)))\n"
      "    return points\n")
_G_CASE = _G.replace("seen.add(words[i].lower())", "seen.add(words[i])")
_G_I = _G.replace("if (i + 1) % step == 0:", "if i % step == 0:")
_G_N = _G.replace("points.append((i + 1, len(seen)))", "points.append((i, len(seen)))")
_G_INSIDE = _G.replace("    seen = set()\n    points = []\n    for i in range(len(words)):\n",
                       "    points = []\n    for i in range(len(words)):\n        seen = set()\n")
_G_LIST = _G.replace("seen = set()", "seen = []").replace("seen.add(", "seen.append(")


def _sets_in_points(got, text) -> bool:
    return isinstance(got, list) and any(isinstance(p, tuple) and any(isinstance(x, (set, list)) for x in p)
                                         for p in got)


def _no_pairs(got, text) -> bool:
    return isinstance(got, list) and any(not isinstance(p, tuple) for p in got)


def _no_step(got, text) -> bool:
    """A point after every word: the step is not used."""
    return (isinstance(got, list) and len(got) >= 2 and all(isinstance(p, tuple) and len(p) == 2 for p in got)
            and [p[0] for p in got] == list(range(1, len(got) + 1)))


EXERCISES = {
    "rank-counts": FunctionBox(
        name="rank-counts",
        picture=rank_picture,
        function="rank_counts",
        start=_R_START,
        solution=_R,
        visible=RANK_CASES[:1],
        hidden=RANK_CASES[1:],
        first=[_DOTS],
        show=_show_counts,
        mistakes=[
            Mistake("a-dict", matches=_a_dict,
                    en="`{call}` gives a dict. The check wants only the counts, in a list from the highest to the lowest: the values of your dict, sorted.",
                    nl="`{call}` geeft een dictionary. De check wil alleen de aantallen, in een lijst van hoog naar laag: de waarden van je dictionary, gesorteerd."),
            Mistake("words", matches=_words_not_counts,
                    en="`{call}` gives words, and it should give how often each occurs: {want}. Which part of your dict holds the counts?",
                    nl="`{call}` geeft woorden, en dat moet hun aantal zijn: {want}. Welk deel van je dictionary bevat de aantallen?"),
            Mistake("pairs", matches=_pairs_not_counts,
                    en="`{call}` gives pairs, and it should give only the counts: {want}. The rank of a word is its place in the list, so the word itself can stay out.",
                    nl="`{call}` geeft paren, en dat moeten alleen de aantallen zijn: {want}. De rang van een woord is de plaats in de lijst, dus het woord zelf kan eruit."),
            Mistake("not-a-list", matches=_not_list,
                    en="`{call}` gives {got}, and it should give a list of counts: {want}. Does your function give back the sorted counts after the loop?",
                    nl="`{call}` geeft {got}, en dat moet een lijst aantallen zijn: {want}. Geeft je functie na de loop de gesorteerde aantallen terug?"),
            Mistake("case", wrong=_R_CASE,
                    en="`{call}` gives {got}, and it should give {want}: *The* and *the* count as two words in your function. Count each word in small letters.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: *The* en *the* tellen in je functie als twee woorden. Tel elk woord in kleine letters."),
            Mistake("upward", wrong=_R_UP,
                    en="`{call}` gives {got}, and it should give {want}: your counts go from low to high. Rank 1 is the most common word, so the highest count comes first.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je aantallen lopen van laag naar hoog. Rang 1 is het woord dat het vaakst voorkomt, dus het hoogste aantal komt eerst."),
            Mistake("zero", wrong=_R_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: every count is one too low. A word you meet for the first time has occurred once.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elk aantal is één te laag. Een woord dat je voor het eerst tegenkomt, is één keer voorgekomen."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: count how often each word of `words` occurs, in small letters, and give back the counts in a list, from the highest to the lowest.",
                      "nl": "Vervang de `...`: tel hoe vaak elk woord van `words` voorkomt, in kleine letters, en geef de aantallen terug in een lijst, van hoog naar laag."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function count every word in small letters, and sort the counts from the highest to the lowest?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Telt je functie elk woord in kleine letters, en sorteert de functie de aantallen van hoog naar laag?"},
            "almost": {"en": "Right for ALICE. But `{call}` gives {got}, and it should give {want}. Does your function work for any list of words, also an empty one?",
                       "nl": "Goed voor ALICE. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke lijst woorden, ook een lege?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back: it has a `return`, but Python does not get to it. Is the `return` the last line, after the loop?",
                          "nl": "Met `{call}` geeft je functie niets terug: er staat een `return` in, maar Python komt er niet aan toe. Is de `return` de laatste regel, na de loop?"},
            "correct": {"en": "Your `rank_counts` is right in all {n} cases. Look at the picture: your counts of two whole corpora, on log scales.",
                        "nl": "Je `rank_counts` klopt in alle {n} gevallen. Kijk naar de afbeelding: je aantallen van twee hele corpora, op logaritmische schalen."},
        },
    ),
    "rank-times-count": FunctionBox(
        name="rank-times-count",
        picture=product_picture,
        function="rank_times_count",
        start=_T_START,
        solution=_T,
        visible=PRODUCT_CASES[:1],
        hidden=PRODUCT_CASES[1:],
        first=[_DOTS],
        show=_show_products,
        mistakes=[
            Mistake("shape", error=_broadcast,
                    en="With `{call}`, NumPy cannot multiply your ranks by `counts`: the two do not have the same length. There is one rank for every count, from 1 up to and including `len(counts)`.",
                    nl="Met `{call}` kan NumPy je rangen niet met `counts` vermenigvuldigen: de twee zijn niet even lang. Er is één rang voor elk aantal, van 1 tot en met `len(counts)`."),
            Mistake("one-number", matches=_one_number,
                    en="`{call}` gives one number, and it should give an array with one number for every rank: {want}.",
                    nl="`{call}` geeft één getal, en dat moet een array zijn met één getal voor elke rang: {want}."),
            Mistake("from-zero", wrong=_T_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: your ranks start at 0. The most common word has rank 1, so the ranks go from 1 up to and including `len(counts)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je rangen beginnen bij 0. Het woord dat het vaakst voorkomt, heeft rang 1, dus de rangen lopen van 1 tot en met `len(counts)`."),
            Mistake("divide", wrong=_T_DIVIDE,
                    en="`{call}` gives {got}, and it should give {want}: you divide the counts by the ranks. Zipf's law says that rank times count stays about the same, so multiply.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je deelt de aantallen door de rangen. De wet van Zipf zegt dat rang keer aantal ongeveer gelijk blijft, dus vermenigvuldig."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back an array with the rank of every count times that count, for the ranks 1, 2, 3 and so on.",
                      "nl": "Vervang de `...`: geef een array terug met de rang van elk aantal keer dat aantal, voor de rangen 1, 2, 3 enzovoort."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does each count get multiplied by its own rank, starting at 1?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Wordt elk aantal vermenigvuldigd met de eigen rang, vanaf 1?"},
            "almost": {"en": "Right for COUNTS. But `{call}` gives {got}, and it should give {want}. Does your function work for any array of counts, also a short one?",
                       "nl": "Goed voor COUNTS. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke array aantallen, ook een korte?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back: it has a `return`, but Python does not get to it. Is the `return` the last line?",
                          "nl": "Met `{call}` geeft je functie niets terug: er staat een `return` in, maar Python komt er niet aan toe. Is de `return` de laatste regel?"},
            "correct": {"en": "Your `rank_times_count` is right in all {n} cases. Look at the picture: the count falls steeply down the list, and rank × count much less.",
                        "nl": "Je `rank_times_count` klopt in alle {n} gevallen. Kijk naar de afbeelding: het aantal daalt steil de lijst af, rang × aantal veel minder."},
        },
    ),
    "growth": FunctionBox(
        name="growth",
        picture=growth_picture,
        function="growth",
        start=_G_START,
        solution=_G,
        visible=GROWTH_CASES[:1],
        hidden=GROWTH_CASES[1:],
        first=[_DOTS],
        show=_show_points,
        mistakes=[
            Mistake("sets", matches=_sets_in_points,
                    en="`{call}` gives points with the words themselves in them. The second number of a point is how many different words you have seen: the length of your set.",
                    nl="`{call}` geeft punten met de woorden zelf erin. Het tweede getal van een punt is hoeveel verschillende woorden je hebt gezien: de lengte van je set."),
            Mistake("no-pairs", matches=_no_pairs,
                    en="`{call}` gives {got}, and it should give {want}: a pair in round brackets for each point, a tuple of two numbers, as in `(5000, 1066)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een paar tussen ronde haakjes voor elk punt, een tuple van twee getallen, zoals `(5000, 1066)`."),
            Mistake("every-word", matches=_no_step,
                    en="`{call}` gives a point after every word, and it should give one after every `step` words: {want}. When is the number of words read so far a whole number of steps?",
                    nl="`{call}` geeft een punt na elk woord, en dat moet er een zijn na elke `step` woorden: {want}. Wanneer is het aantal woorden dat je tot nu toe gelezen hebt een heel aantal stappen?"),
            Mistake("case", wrong=_G_CASE,
                    en="`{call}` gives {got}, and it should give {want}: *A* and *a* count as two different words in your function. Put each word in the set in small letters.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: *A* en *a* tellen in je functie als twee verschillende woorden. Stop elk woord in kleine letters in de set."),
            Mistake("from-zero", wrong=_G_I,
                    en="`{call}` gives {got}, and it should give {want}: your first point comes after one word, and the others one word late. `i` counts from 0, so `i % step` is already 0 at the very first word. How many words have you read after the word at place `i`?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je eerste punt komt na één woord, en de andere één woord te laat. `i` telt vanaf 0, dus `i % step` is al 0 bij het allereerste woord. Hoeveel woorden heb je gelezen na het woord op plaats `i`?"),
            Mistake("length", wrong=_G_N,
                    en="`{call}` gives {got}, and it should give {want}: the first number of each point is one too low. After word `i`, you have read `i + 1` words, since `i` counts from 0.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het eerste getal van elk punt is één te laag. Na woord `i` heb je `i + 1` woorden gelezen, want `i` telt vanaf 0."),
            Mistake("inside", wrong=_G_INSIDE,
                    en="`{call}` gives {got}, and it should give {want}: your set holds one word at every point. Make the set once, before the loop, so that it keeps every word you have seen.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je set heeft bij elk punt één woord. Maak de set één keer, vóór de loop, zodat de set elk woord bewaart dat je hebt gezien."),
            Mistake("list", wrong=_G_LIST,
                    en="`{call}` gives {got}, and it should give {want}: you count every word, also one you have seen before. A list keeps each word as often as you add it; a set keeps it once.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt elk woord, ook een woord dat je al hebt gezien. Een lijst bewaart elk woord zo vaak als je het toevoegt; een set bewaart het één keer."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: go through `words`, and after every `step` words, add a pair (words read so far, different words so far) to a list.",
                      "nl": "Vervang de `...`: ga `words` langs, en voeg na elke `step` woorden een paar (tot nu toe gelezen woorden, tot nu toe verschillende woorden) toe aan een lijst."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does each point hold how many words you have read and how many different words, in small letters, you have seen so far?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Bevat elk punt hoeveel woorden je hebt gelezen en hoeveel verschillende woorden, in kleine letters, je tot nu toe hebt gezien?"},
            "almost": {"en": "Right for ALICE. But `{call}` gives {got}, and it should give {want}. Does your function work for any list of words and any step?",
                       "nl": "Goed voor ALICE. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke lijst woorden en elke stap?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back: it has a `return`, but Python does not get to it. Is the `return` the last line, after the loop?",
                          "nl": "Met `{call}` geeft je functie niets terug: er staat een `return` in, maar Python komt er niet aan toe. Is de `return` de laatste regel, na de loop?"},
            "correct": {"en": "Your `growth` is right in all {n} cases. Look at the picture: how many different words two corpora hold as you read on.",
                        "nl": "Je `growth` klopt in alle {n} gevallen. Kijk naar de afbeelding: hoeveel verschillende woorden twee corpora hebben terwijl je verder leest."},
        },
    ),
    "laws": Sort(
        name="laws",
        groups={
            "zipf": {"en": "Zipf's law: how often each word occurs, by its rank",
                     "nl": "De wet van Zipf: hoe vaak elk woord voorkomt, naar de rang"},
            "heaps": {"en": "Heaps' law: how many different words a text holds, by its length",
                      "nl": "De wet van Heaps: hoeveel verschillende woorden een tekst heeft, naar de lengte"},
        },
        items={
            "second": {"en": "**The second most common word occurs about half as often as the first.**",
                       "nl": "**Het op één na meest voorkomende woord komt ongeveer half zo vaak voor als het eerste.**"},
            "longer": {"en": "**A text ten times as long holds about four times as many different words.**",
                       "nl": "**Een tekst die tien keer zo lang is, heeft ongeveer vier keer zoveel verschillende woorden.**"},
            "product": {"en": "**Rank times count stays roughly the same down the list.**",
                        "nl": "**Rang keer aantal blijft de lijst af ongeveer gelijk.**"},
            "index": {"en": "**The index of a search engine needs room for new words with every batch of pages it adds.**",
                      "nl": "**De index van een zoekmachine heeft bij elke nieuwe stapel pagina's ruimte nodig voor nieuwe woorden.**"},
            "half": {"en": "**Fewer than a hundred different words make up half of a novel.**",
                     "nl": "**Minder dan honderd verschillende woorden maken samen de helft van een roman uit.**"},
            "names": {"en": "**Every new book brings names that no earlier book had.**",
                      "nl": "**Elk nieuw boek brengt namen mee die nog in geen enkel eerder boek stonden.**"},
        },
        solution={"second": "zipf", "longer": "heaps", "product": "zipf", "index": "heaps", "half": "zipf",
                  "names": "heaps"},
        nudge={
            "second": {"en": "Are two words of one list compared here, or two texts?",
                       "nl": "Worden hier twee woorden van één lijst vergeleken, of twee teksten?"},
            "longer": {"en": "What changes here: the place of a word in a list, or the length of the text?",
                       "nl": "Wat verandert hier: de plaats van een woord in een lijst, of de lengte van de tekst?"},
            "product": {"en": "Which two numbers are multiplied here, and where do they come from?",
                        "nl": "Welke twee getallen worden hier vermenigvuldigd, en waar komen die getallen vandaan?"},
            "index": {"en": "What grows here as more pages come in?",
                      "nl": "Wat groeit hier als er meer pagina's bijkomen?"},
            "half": {"en": "Is this about the most common words of one text, or about texts of different lengths?",
                     "nl": "Gaat dit over de woorden die het vaakst voorkomen in één tekst, of over teksten van verschillende lengte?"},
            "names": {"en": "What happens here to the number of different words as you read more?",
                      "nl": "Wat gebeurt hier met het aantal verschillende woorden als je meer leest?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the statements again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de uitspraken opnieuw uit."},
            "empty": {"en": "Put each bold statement under the law it is about.",
                      "nl": "Zet elke vetgedrukte uitspraak bij de wet waar die over gaat."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} belongs under another group.",
                      "nl": "{item} hoort bij een andere groep."},
            "correct": {"en": "All {n} are right. The two laws describe the same texts: because most words are rare, a longer text keeps meeting words it has not seen before.",
                        "nl": "Alle {n} kloppen. De twee wetten beschrijven dezelfde teksten: omdat de meeste woorden zeldzaam zijn, komt een langere tekst steeds woorden tegen die nog niet eerder voorkwamen."},
        },
    ),
}

CASES = [
    ("rank-counts", _R_START, "empty", None),
    ("rank-counts", _R, "correct", None),
    ("rank-counts", "def rank_counts(words):\n    counts = {}\n    for w in words:\n        counts[w.lower()] = counts.get(w.lower(), 0) + 1\n    return sorted(counts.values(), reverse=True)\n", "correct", None),
    ("rank-counts", "def rank_counts(words):\n    small = [w.lower() for w in words]\n    return sorted([small.count(w) for w in set(small)], reverse=True)\n", "correct", None),
    ("rank-counts", "def rank_counts(words):\n    counts = {}\n    for w in words:\n        w = w.lower()\n        if w not in counts:\n            counts[w] = 0\n        counts[w] = counts[w] + 1\n    return sorted(counts.values())[::-1]\n", "correct", None),
    ("rank-counts", _R_CASE, "wrong", "case"),
    ("rank-counts", _R_UP, "wrong", "upward"),
    ("rank-counts", _R_ZERO, "wrong", "zero"),
    ("rank-counts", _R.replace("    return sorted(counts.values(), reverse=True)\n", "    return counts\n"), "wrong", "a-dict"),
    ("rank-counts", _R.replace("sorted(counts.values(), reverse=True)", "sorted(counts, reverse=True)"), "wrong", "words"),
    ("rank-counts", _R.replace("sorted(counts.values(), reverse=True)", "sorted(counts.items(), reverse=True)"), "wrong", "pairs"),
    ("rank-counts", _R.replace("sorted(counts.values(), reverse=True)", "len(counts)"), "wrong", "not-a-list"),
    ("rank-counts", _R.replace("    return sorted(counts.values(), reverse=True)\n", "    sorted(counts.values(), reverse=True)\n"), "wrong", "no-return"),
    ("rank-counts", "def rank_counts(words):\n    return " + repr(rank_counts(book_words("alice"))) + "\n", "almost", None),
    ("rank-times-count", _T_START, "empty", None),
    ("rank-times-count", _T, "correct", None),
    ("rank-times-count", "import numpy as np\n\n\ndef rank_times_count(counts):\n    return counts * np.arange(1, len(counts) + 1)\n", "correct", None),
    ("rank-times-count", "def rank_times_count(counts):\n    return [(i + 1) * counts[i] for i in range(len(counts))]\n", "correct", None),
    ("rank-times-count", _T_ZERO, "wrong", "from-zero"),
    ("rank-times-count", _T_DIVIDE, "wrong", "divide"),
    ("rank-times-count", _T.replace("np.arange(1, len(counts) + 1)", "np.arange(1, len(counts))"), "error", "shape"),
    ("rank-times-count", _T.replace("return ranks * counts", "return np.sum(ranks * counts)"), "wrong", "one-number"),
    ("rank-times-count", _T.replace("    return ranks * counts\n", "    ranks * counts\n"), "wrong", "no-return"),
    ("rank-times-count", "import numpy as np\n\n\ndef rank_times_count(counts):\n    return np.array(" + repr(rank_times_count(np.array(rank_counts(book_words("alice")))).tolist()) + ")\n", "almost", None),
    ("growth", _G_START, "empty", None),
    ("growth", _G, "correct", None),
    ("growth", "def growth(words, step):\n    seen = set()\n    points = []\n    n = 0\n    for w in words:\n        seen.add(w.lower())\n        n = n + 1\n        if n % step == 0:\n            points.append((n, len(seen)))\n    return points\n", "correct", None),
    ("growth", "def growth(words, step):\n    seen = {}\n    points = []\n    for i in range(len(words)):\n        seen[words[i].lower()] = True\n        if (i + 1) % step == 0:\n            points.append((i + 1, len(seen)))\n    return points\n", "correct", None),
    ("growth", _G_CASE, "wrong", "case"),
    ("growth", _G_I, "wrong", "from-zero"),
    ("growth", _G_N, "wrong", "length"),
    ("growth", _G_INSIDE, "wrong", "inside"),
    ("growth", _G_LIST, "wrong", "list"),
    ("growth", _G.replace("points.append((i + 1, len(seen)))", "points.append((i + 1, seen))"), "wrong", "sets"),
    ("growth", _G.replace("points.append((i + 1, len(seen)))", "points.append(len(seen))"), "wrong", "no-pairs"),
    ("growth", _G.replace("        if (i + 1) % step == 0:\n            points.append((i + 1, len(seen)))\n", "        points.append((i + 1, len(seen)))\n"), "wrong", "every-word"),
    ("growth", _G.replace("    return points\n", ""), "wrong", "no-return"),
    ("growth", "def growth(words, step):\n    return " + repr(growth(book_words("alice"), 5000)) + "\n", "almost", None),
    ("laws", {}, "empty", None),
    ("laws", {"second": "zipf", "longer": "heaps", "product": "zipf", "index": "heaps", "half": "zipf",
              "names": "heaps"}, "correct", None),
    ("laws", {"second": "zipf", "longer": "heaps"}, "wrong", "unfinished"),
    ("laws", {"names": "zipf"}, "wrong", "wrong:names"),
    ("laws", {"second": "zipf", "longer": "zipf", "product": "zipf", "index": "heaps", "half": "zipf",
              "names": "heaps"}, "wrong", "wrong:longer"),
]
