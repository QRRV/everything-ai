# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-language-id: cut a text into character n-grams (ngrams), measure how far the profile of a
text lies from the profile of a language (out_of_place), and give the nearest language, or "?" when no language
comes near enough (identify); and sort texts by what that program gives for them.

The method follows William Cavnar and John Trenkle, *N-Gram-Based Text Categorization* (1994), written again in the
course's own code and words, with trigrams only and with the course's own choices: a missing trigram costs the size
of the language's profile, and the sum is divided by the most it can be, so that a limit can say "no language". The
sentences are the course's own selection from Tatoeba in data/tatoeba-langs.tsv.gz (CC BY 2.0 FR, Tatoeba and its
contributors; made by data/tools/make_tatoeba_langs.py). The other texts, the cases of the checks and the pictures are
the course's own."""
import contextlib
import copy
import html
import io

from everything_ai import books, yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, kept, with_time_limit

# --- the sentences -----------------------------------------------------------------------------------------------------

LANGUAGES = ("nl", "af", "fy", "de", "en", "da", "nb", "sv", "fr", "es", "pt", "it")
LANG_NAMES = {
    "en": {"nl": "Dutch", "af": "Afrikaans", "fy": "Frisian", "de": "German", "en": "English", "da": "Danish",
           "nb": "Bokmål", "sv": "Swedish", "fr": "French", "es": "Spanish", "pt": "Portuguese", "it": "Italian"},
    "nl": {"nl": "Nederlands", "af": "Afrikaans", "fy": "Fries", "de": "Duits", "en": "Engels", "da": "Deens",
           "nb": "Bokmål", "sv": "Zweeds", "fr": "Frans", "es": "Spaans", "pt": "Portugees", "it": "Italiaans"},
}
N = 3          # the n of the n-grams of a profile
SIZE = 300     # trigrams in a profile, at most
LIMIT = 0.9    # the distance at which identify gives "?" in the lesson
HELD_OUT = 5   # every fifth sentence of a language stays out of its profile, to try the program on


def sentences(lang: str) -> list:
    """The Tatoeba sentences of one language, by id, as a new list of strings."""
    return [text for code, _, text in books.language_sentences() if code == lang]


def learned(lang: str) -> list:
    """The sentences of a language that its profile is made from: all but every fifth."""
    return [t for k, t in enumerate(sentences(lang)) if k % HELD_OUT != HELD_OUT - 1]


def held_out(lang: str) -> list:
    """Every fifth sentence of a language, which its profile has not seen."""
    return [t for k, t in enumerate(sentences(lang)) if k % HELD_OUT == HELD_OUT - 1]


def broken(text: str) -> str:
    """The text with its UTF-8 bytes read as Windows-1252, the most common way text breaks: *café* becomes *cafÃ©*.
    A byte that Windows-1252 lacks becomes a question mark in a box."""
    return text.encode("utf-8").decode("cp1252", errors="replace")


# --- the course's solutions --------------------------------------------------------------------------------------------

def ngrams(text, n):
    """The course's solution of Exercise 1: every run of n characters of `text` with a space before and after it, in
    order, repeats included."""
    padded = " " + text + " "
    grams = []
    for i in range(len(padded) - n + 1):
        grams.append(padded[i:i + n])
    return grams


def _profile(text: str, size: int, n: int) -> dict:
    counts = {}
    for gram in ngrams(" ".join(text.lower().split()), n):
        counts[gram] = counts.get(gram, 0) + 1
    best = sorted(counts, key=lambda g: (-counts[g], g))[:size]
    return {g: rank for rank, g in enumerate(best)}


def profile(text, size=SIZE):
    """The profile of a text: its trigrams, in small letters, with one space between words and one at each end, the
    most common first, at most `size` of them; each with its rank, 0 for the most common. Trigrams that occur equally
    often go in alphabetical order."""
    return _profile(text, size, N)


def out_of_place(doc, lang):
    """The course's solution of Exercise 2: for every trigram of the profile `doc`, how far its rank lies from its
    rank in the profile `lang`, or the size of `lang` when `lang` lacks it; all added up."""
    total = 0
    for gram in doc:
        if gram in lang:
            total = total + max(doc[gram], lang[gram]) - min(doc[gram], lang[gram])
        else:
            total = total + len(lang)
    return total


def distance(doc, lang):
    """The out-of-place sum divided by the most it can be, len(doc) times len(lang): 0 when every trigram of `doc` has
    the same rank in `lang`, 1 when `lang` has none of them; 1 for an empty `doc` too."""
    if len(doc) == 0 or len(lang) == 0:
        return 1.0
    return out_of_place(doc, lang) / (len(doc) * len(lang))


def identify(text, profiles, limit):
    """The course's solution of Exercise 3: the language of `profiles` whose profile lies nearest to the profile of
    `text`, or "?" when even the nearest lies at `limit` or farther."""
    doc = profile(text)
    best = "?"
    best_distance = limit
    for lang in profiles:
        d = distance(doc, profiles[lang])
        if d < best_distance:
            best = lang
            best_distance = d
    return best


def _profiles(size: int = SIZE, n: int = N) -> dict:
    def make():
        return {lang: _profile(" ".join(learned(lang)), size, n) for lang in LANGUAGES}
    return kept(f"nlp-language-id:profiles:{size}:{n}", make)


def language_profiles() -> dict:
    """The profile of each of the twelve languages, made from its sentences except every fifth, as a new dict:
    {"nl": {"en ": 0, "et ": 1, ...}, ...}."""
    return copy.deepcopy(_profiles())


def counts_of(lang: str) -> dict:
    """How often each trigram occurs in the sentences that the profile of `lang` is made from."""
    def make():
        counts = {}
        for gram in ngrams(" ".join(" ".join(learned(lang)).lower().split()), N):
            counts[gram] = counts.get(gram, 0) + 1
        return counts
    return kept(f"nlp-language-id:counts:{lang}", make)


# --- how well the program does -----------------------------------------------------------------------------------------

def _nearest(doc: dict, profiles: dict, limit: float) -> str:
    best, best_distance = "?", limit
    for lang in profiles:
        d = distance(doc, profiles[lang])
        if d < best_distance:
            best, best_distance = lang, d
    return best


def score(size: int = SIZE, n: int = N, join: int = 1, limit: float = LIMIT) -> dict:
    """What the course's program gives for the held-out sentences, `join` of them together as one text, with profiles
    of `size` n-grams: {"answers": [(lang, text, answer), ...], "right": k, "unknown": k, "total": k}."""
    def make():
        profiles = _profiles(size, n)
        answers = []
        for lang in LANGUAGES:
            out = held_out(lang)
            for j in range(0, len(out) - join + 1, join):
                text = " ".join(out[j:j + join])
                answers.append((lang, text, _nearest(_profile(text, size, n), profiles, limit)))
        return {"answers": answers, "right": sum(a == lang for lang, _, a in answers),
                "unknown": sum(a == "?" for _, _, a in answers), "total": len(answers)}
    return kept(f"nlp-language-id:score:{size}:{n}:{join}:{limit}", make)


def mixed_up(answers: list) -> dict:
    """How often each pair (language, answer) occurs among wrong answers that are a language."""
    pairs = {}
    for lang, _, a in answers:
        if a != lang and a != "?":
            pairs[(lang, a)] = pairs.get((lang, a), 0) + 1
    return pairs


# --- the look of the pictures ------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);margin-top:0.45rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;vertical-align:baseline;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums"
_CHIP = ("display:inline-block;margin:0.12rem 0.2rem 0.12rem 0;padding:0 0.3rem;border-radius:3px;"
         "font-family:var(--ea-font-code);font-size:0.82rem;line-height:1.5;white-space:pre")
_CHIP_DATA = _CHIP + ";border:1px solid var(--ea-line);color:var(--ea-ink)"
_CHIP_PARAM = _CHIP + ";background:var(--ea-param-wash);color:var(--ea-param-text)"
_CHIP_PRED = _CHIP + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
_CHIP_PURPLE = _CHIP + ";background:var(--ea-loss-wash);color:var(--ea-loss-text)"
_CHIP_MISSING = _CHIP + ";border:1px dashed var(--ea-loss);color:var(--ea-loss-text)"
_CHIP_FAINT = _CHIP + ";color:var(--ea-ink-3)"
_BAR = "display:inline-block;height:0.6rem;border-radius:2px;vertical-align:middle"
_TEXT = "font-family:var(--ea-font-code);font-size:0.8rem;color:var(--ea-ink);white-space:pre-wrap;word-break:break-all"
SPACE = "␣"   # how a picture shows a space inside an n-gram


def _note(text: str) -> str:
    """A line under a picture; a word between asterisks is set in italics, as in the lesson's text."""
    parts = html.escape(text).split("*")
    marked = "".join(f"<i>{p}</i>" if k % 2 else p for k, p in enumerate(parts))
    return f'<div style="{_UI}">{marked}</div>'


def _label(text: str) -> str:
    parts = html.escape(text).split("*")
    marked = "".join(f"<i>{p}</i>" if k % 2 else p for k, p in enumerate(parts))
    return f'<div style="{_LABEL}">{marked}</div>'


def _num(n, lang: str) -> str:
    """A whole number with a separator for thousands: 1,190 in English, 1.190 in Dutch."""
    text = f"{round(n):,}"
    return text.replace(",", ".") if lang == "nl" else text


def _pct(x: float, lang: str) -> str:
    """A percentage with one decimal: 83.4% in English, 83,4% in Dutch."""
    text = f"{x:.1f}%"
    return text.replace(".", ",") if lang == "nl" else text


def _dec(x: float, lang: str, digits: int = 2) -> str:
    """A number with `digits` decimals: 0.67 in English, 0,67 in Dutch."""
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def shown(gram: str) -> str:
    """An n-gram as a picture shows it, with ␣ for each space, so that a space at either end can be seen."""
    return gram.replace(" ", SPACE)


def _chip(text: str, style: str = _CHIP_DATA) -> str:
    return f'<span style="{style}">{html.escape(text)}</span>'


def _gram(gram: str, style: str = _CHIP_DATA) -> str:
    return _chip(shown(gram), style)


def _bar(x: float, most: float, color: str, width: float = 8.0) -> str:
    """A bar for a number: as long as `width` for `most`, never shorter than a sliver."""
    w = 0 if not most or not x else max(0.15, width * x / most)
    return f'<span style="{_BAR};width:{w:.2f}rem;background:{color}"></span>' if w else ""


def _limit_bar(d: float, limit: float, color: str, width: float = 9.0) -> str:
    """A bar for a distance from 0 to 1, with a thin mark at the limit, in the color of a setting you choose."""
    w = max(0.15, width * min(d, 1.0))
    mark = width * limit
    return (f'<span style="display:inline-block;position:relative;width:{width:.2f}rem;height:0.8rem;'
            f'vertical-align:middle"><span style="position:absolute;left:0;top:0.1rem;{_BAR};width:{w:.2f}rem;'
            f'background:{color}"></span><span style="position:absolute;left:{mark:.2f}rem;top:0;width:2px;'
            f'height:0.8rem;background:var(--ea-hyper)"></span></span>')


def _table(rows_: list, head: list, numbers: tuple = ()) -> str:
    """A small table; rows hold HTML that is already escaped; numbers are the columns set to the right; no head when
    `head` is empty."""
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{_NUMBER if j in numbers else ""}">{c}</td>'
                                    for j, c in enumerate(row)) + "</tr>" for row in rows_)
    head_html = f"<thead><tr>{top}</tr></thead>" if head else ""
    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;margin:0.3rem 0 0.2rem">'
            f"{head_html}<tbody>{body}</tbody></table></div>")


def _text(text: str, longest: int = 60) -> str:
    """A text of the learner's or the course's, as code, cut off after `longest` characters."""
    cut = text if len(text) <= longest else text[:longest - 1] + "…"
    return f'<span style="{_TEXT}">{html.escape(cut)}</span>'


# --- the idea's picture: the start of four profiles --------------------------------------------------------------------

IDEA_LANGUAGES = {"en": ("en", "nl", "de", "af"), "nl": ("nl", "af", "de", "en")}
IDEA_ROWS = 10

_IDEA_WORDS = {
    "en": {"title": "The most common trigrams of four languages",
           "rank": "Rank",
           "note": "Orange: the start of each language's profile, counted in 400 of its sentences from Tatoeba. ␣ is a "
                   "space: *␣th* starts a word, *he␣* ends one.",
           "alt": "{title}. {columns}. {note}"},
    "nl": {"title": "De vaakste trigrams van vier talen",
           "rank": "Rang",
           "note": "Oranje: het begin van het profiel van elke taal, geteld in 400 van de zinnen van die taal uit "
                   "Tatoeba. ␣ is een spatie: *␣de* begint een woord, *en␣* sluit een woord af.",
           "alt": "{title}. {columns}. {note}"},
}


def idea_picture(lang: str) -> str:
    w = _IDEA_WORDS[lang]
    profiles = _profiles()
    langs = IDEA_LANGUAGES[lang]
    tops = {code: list(profiles[code])[:IDEA_ROWS] for code in langs}
    rows_ = [[str(r)] + [_gram(tops[code][r], _CHIP_PARAM) for code in langs] for r in range(IDEA_ROWS)]
    body = _table(rows_, [w["rank"]] + [LANG_NAMES[lang][code] for code in langs], numbers=(0,)) + _note(w["note"])
    columns = "; ".join(f"{LANG_NAMES[lang][code]}: " + ", ".join(shown(g) for g in tops[code]) for code in langs)
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], columns=columns, note=w["note"]))


# --- running the learner's function ----------------------------------------------------------------------------------

def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error
PICTURE_LIMIT = 10.0   # seconds the learner's function gets for everything a picture asks of it


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


_SAY = {
    "en": {"stopped": "Your function stopped with an error on one of the texts, so the picture stays empty.",
           "slow": "Your function took more than {s:g} seconds for all texts, so the picture stays empty."},
    "nl": {"stopped": "Je functie stopte met een fout op een van de teksten, dus de afbeelding blijft leeg.",
           "slow": "Je functie deed er langer dan {s:g} seconden over voor alle teksten, dus de afbeelding blijft leeg."},
}


# --- the picture under exercise 1: your n-grams ------------------------------------------------------------------------

NGRAM_CALLS = {"en": [("the cat", 3), ("it", 3), ("home", 2), ("no", 1)],
               "nl": [("de kat", 3), ("ik", 3), ("huis", 2), ("ja", 1)]}
NGRAM_LANGUAGE = {"en": "en", "nl": "nl"}   # the language whose sentences the picture counts with the learner's function
NGRAM_TOP = 10

_NGRAM_WORDS = {
    "en": {"title": "Your n-grams",
           "head": ["Call", "Your n-grams"],
           "counted": "Your trigrams of the 400 English sentences, counted: the {k} most common",
           "none": "(none)",
           "note": "Gray: your n-grams, with ␣ for a space. Purple: an n-gram that should not be there or not in that "
                   "place; dashed: one that is missing. Orange: the most common of your trigrams, the start of a "
                   "profile.",
           "odd": "Your function gave something other than a list of strings, so the picture stays empty.",
           "alt": "{title}. {rows}. {counted}: {top}. {note}"},
    "nl": {"title": "Je n-grams",
           "head": ["Aanroep", "Je n-grams"],
           "counted": "Je trigrams van de 400 Nederlandse zinnen, geteld: de {k} vaakste",
           "none": "(geen)",
           "note": "Grijs: je n-grams, met ␣ voor een spatie. Paars: een n-gram dat er niet hoort of niet op die plek; "
                   "met een gestreepte rand: een n-gram dat ontbreekt. Oranje: de vaakste van je trigrams, het begin van een "
                   "profiel.",
           "odd": "Je functie gaf iets anders dan een lijst strings, dus de afbeelding blijft leeg.",
           "alt": "{title}. {rows}. {counted}: {top}. {note}"},
}


def _strings(v) -> bool:
    return isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v)


def _compare_chips(mine: list, want: list) -> tuple:
    """Chips for the learner's list against the right one, place by place, and the missing ones dashed."""
    chips = []
    for k, g in enumerate(mine):
        chips.append(_gram(g, _CHIP_DATA if k < len(want) and want[k] == g else _CHIP_PURPLE))
    chips += [_gram(g, _CHIP_MISSING) for g in want[len(mine):]]
    return chips


def ngrams_picture(seen) -> str:
    lang = seen.lang
    w = _NGRAM_WORDS[lang] | _SAY[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    calls = NGRAM_CALLS[lang]
    text = " ".join(" ".join(learned(NGRAM_LANGUAGE[lang])).lower().split())
    got, said = _run_or_say(lambda: ([fn(t, n) for t, n in calls], fn(text, N)), w,
                            lambda g: all(_strings(v) for v in g[0]) and _strings(g[1]))
    if said:
        return said
    mine_calls, mine_all = got
    rows_, alts = [], []
    for (t, n), mine in zip(calls, mine_calls):
        mine = list(mine)
        chips = _compare_chips(mine, ngrams(t, n)) or [_chip(w["none"], _CHIP_FAINT)]
        rows_.append([f'<code>ngrams("{html.escape(t)}", {n})</code>', "".join(chips)])
        alts.append(f'ngrams("{t}", {n}): ' + (", ".join(shown(g) for g in mine) or w["none"]))
    counts = {}
    for g in mine_all:
        counts[g] = counts.get(g, 0) + 1
    top = sorted(counts, key=lambda g: (-counts[g], g))[:NGRAM_TOP]
    most = counts[top[0]] if top else 0
    count_rows = [[_gram(g, _CHIP_PARAM), _num(counts[g], lang), _bar(counts[g], most, "var(--ea-param)")] for g in top]
    counted = w["counted"].format(k=len(top))
    body = _table(rows_, w["head"]) + _label(counted) + _table(count_rows, [], numbers=(1,)) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["alt"].format(
        title=w["title"], rows="; ".join(alts), counted=counted,
        top=", ".join(f"{shown(g)} {_num(counts[g], lang)}" for g in top), note=w["note"]))


# --- the picture under exercise 2: how far out of place ----------------------------------------------------------------

OOP_TEXT = {"en": ("I am going home now.", "en"), "nl": ("Ik fiets naar huis.", "nl")}

_OOP_WORDS = {
    "en": {"title": "Your out-of-place measure",
           "head": ["Trigram", "Rank in the text", "Rank in {name}", "Out of place"],
           "missing": "missing",
           "sum": "Your sum for *{text}* against {name}: {total}",
           "distances": "Your sum divided by the most it can be, for each language: the distance",
           "note": "Gray: the trigrams of the text and their ranks, all of them once in the text, so in alphabetical "
                   "order. Orange: the ranks in the profile of {name}. Purple: what each trigram adds with your "
                   "function, and the distances; a trigram that {name} lacks adds {size}. Blue: the nearest language.",
           "odd": "Your function gave something other than a number, so the picture stays empty.",
           "alt": "{title}. {rows}. {sum}. {distances}: {list}. {note}"},
    "nl": {"title": "Je out-of-place-maat",
           "head": ["Trigram", "Rang in de tekst", "Rang in het {name}", "Uit de plaats"],
           "missing": "ontbreekt",
           "sum": "Je som voor *{text}* tegen het {name}: {total}",
           "distances": "Je som gedeeld door het hoogste dat de som kan zijn, voor elke taal: de afstand",
           "note": "Grijs: elk trigram van de tekst met de rang; elk trigram staat één keer in de tekst, dus de rang volgt het alfabet. "
                   "Oranje: de rang in het profiel van het {name}. Paars: wat elk trigram met je functie optelt, en "
                   "de afstanden; een trigram dat het {name} mist, telt {size} op. Blauw: de taal die het dichtst bij "
                   "ligt.",
           "odd": "Je functie gaf iets anders dan een getal, dus de afbeelding blijft leeg.",
           "alt": "{title}. {rows}. {sum}. {distances}: {list}. {note}"},
}


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def oop_picture(seen) -> str:
    lang = seen.lang
    w = _OOP_WORDS[lang] | _SAY[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    text, code = OOP_TEXT[lang]
    profiles = _profiles()
    doc = profile(text)
    target = profiles[code]

    def make():
        each = [fn({g: r}, dict(target)) for g, r in doc.items()]
        sums = {c: fn(dict(doc), dict(profiles[c])) for c in LANGUAGES}
        return each, sums
    got, said = _run_or_say(make, w, lambda g: all(_number(v) for v in g[0]) and all(_number(v) for v in g[1].values()))
    if said:
        return said
    each, sums = got
    name = LANG_NAMES[lang][code]
    most = max([abs(v) for v in each] + [1])
    rows_ = []
    for (g, r), cost in zip(doc.items(), each):
        there = (f'<span style="color:var(--ea-param-text)">{target[g]}</span>' if g in target
                 else f'<span style="color:var(--ea-ink-3)">{w["missing"]}</span>')
        rows_.append([_gram(g), str(r), there, f"{_num(cost, lang)} {_bar(abs(cost), most, 'var(--ea-loss)', 6.0)}"])
    sum_line = w["sum"].format(text=text, name=name, total=_num(sums[code], lang))
    dists = {c: sums[c] / (len(doc) * len(profiles[c])) for c in LANGUAGES}
    order = sorted(LANGUAGES, key=lambda c: dists[c])
    nearest = order[0]
    dist_rows = [[_chip(LANG_NAMES[lang][c], _CHIP_PRED if c == nearest else _CHIP_DATA), _dec(dists[c], lang),
                  _bar(max(dists[c], 0), 1.0, "var(--ea-loss)", 9.0)] for c in order]
    head = [h.format(name=name) for h in w["head"]]
    note = w["note"].format(name=name, size=len(target))
    body = (_table(rows_, head, numbers=(1, 2, 3)) + _label(sum_line) + _label(w["distances"])
            + _table(dist_rows, [], numbers=(1,)) + _note(note))
    alt_rows = "; ".join(f"{shown(g)} {r} {target.get(g, w['missing'])} {cost}" for (g, r), cost in zip(doc.items(), each))
    return yours.figure(w["title"], body, alt=w["alt"].format(
        title=w["title"], rows=alt_rows, sum=sum_line.replace("*", ""), distances=w["distances"],
        list=", ".join(f"{LANG_NAMES[lang][c]} {_dec(dists[c], lang)}" for c in order), note=note))


# --- the picture under exercise 3: your program on 1,190 sentences -----------------------------------------------------

JUNK = [
    ("html", '<div class="nav"><a href="/home">Home</a></div>'),
    ("json", '{"id": 17, "name": "x", "tags": ["a", "b"]}'),
    ("date", "2026-11-22 14:05 +0100"),
    ("code", "for (i = 0; i < n; i++) { x[i] = 0; }"),
    ("russian", "Я люблю читать книги."),
    ("russian-broken", broken("Я люблю читать книги.")),
    ("turkish", "Bugün hava çok güzel."),
]
JUNK_NAMES = {
    "en": {"html": "A line of a web page (HTML)", "json": "Data (JSON)", "date": "A date and a time",
           "code": "A line of code", "russian": "Russian", "russian-broken": "The same Russian, read with the wrong "
           "encoding", "turkish": "Turkish"},
    "nl": {"html": "Een regel van een webpagina (HTML)", "json": "Data (JSON)", "date": "Een datum en een tijd",
           "code": "Een regel code", "russian": "Russisch", "russian-broken": "Hetzelfde Russisch, gelezen met de "
           "verkeerde encoding", "turkish": "Turks"},
}

_IDENTIFY_WORDS = {
    "en": {"title": "Your program on sentences it has not seen",
           "head": ["Language", "Right", "", "Most often taken for", '"?"'],
           "total": "Right: {right} of {total} ({pct}); \"?\": {unknown}",
           "junk": "Your program on texts outside the twelve languages, with a limit of {limit}",
           "junk_head": ["Text", "", "Your answer"],
           "none": "–",
           "note": "Blue: the sentences for which your program gives the right language, and its answers. Every fifth sentence "
                   "of each language stays out of the profiles, so the program meets it here for the first time.",
           "odd": "Your function gave something other than a string, so the picture stays empty.",
           "alt": "{title}. {total}. {rows}. {junk}: {junk_rows}. {note}"},
    "nl": {"title": "Je programma op zinnen die het niet eerder zag",
           "head": ["Taal", "Goed", "", "Het vaakst aangezien voor", '"?"'],
           "total": "Goed: {right} van {total} ({pct}); \"?\": {unknown}",
           "junk": "Je programma op teksten buiten de twaalf talen, met een grens van {limit}",
           "junk_head": ["Tekst", "", "Je antwoord"],
           "none": "–",
           "note": "Blauw: de zinnen waarvan je programma de goede taal geeft, en de antwoorden. Elke vijfde zin van "
                   "elke taal blijft buiten de profielen, dus het programma ziet die zin hier voor het eerst.",
           "odd": "Je functie gaf iets anders dan een string, dus de afbeelding blijft leeg.",
           "alt": "{title}. {total}. {rows}. {junk}: {junk_rows}. {note}"},
}


def identify_picture(seen) -> str:
    lang = seen.lang
    w = _IDENTIFY_WORDS[lang] | _SAY[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    profiles = _profiles()
    tests = [(code, t) for code in LANGUAGES for t in held_out(code)]

    def make():
        given = copy.deepcopy(profiles)
        return [fn(t, given, LIMIT) for _, t in tests], [fn(t, given, LIMIT) for _, t in JUNK]
    got, said = _run_or_say(make, w, lambda g: all(isinstance(a, str) for a in g[0] + g[1]))
    if said:
        return said
    answers, junk = got
    triples = [(code, t, a) for (code, t), a in zip(tests, answers)]
    pairs = mixed_up(triples)
    rows_, alts = [], []
    for code in LANGUAGES:
        mine = [a for c, _, a in triples if c == code]
        right = sum(a == code for a in mine)
        unknown = sum(a == "?" for a in mine)
        other = sorted(((k, b) for (a, b), k in pairs.items() if a == code), key=lambda p: (-p[0], p[1]))
        taken = f"{LANG_NAMES[lang].get(other[0][1], other[0][1])} ({other[0][0]})" if other else w["none"]
        rows_.append([LANG_NAMES[lang][code], f"{right}/{len(mine)}", _bar(right, len(mine), "var(--ea-pred)", 6.0),
                      html.escape(taken), str(unknown)])
        alts.append(f"{LANG_NAMES[lang][code]} {right}/{len(mine)}, {taken}, ? {unknown}")
    right = sum(a == c for c, _, a in triples)
    unknown = sum(a == "?" for _, _, a in triples)
    total = w["total"].format(right=_num(right, lang), total=_num(len(triples), lang),
                              pct=_pct(100 * right / len(triples), lang), unknown=unknown)
    junk_rows = [[html.escape(JUNK_NAMES[lang][key]), _text(t, 44), _chip(a, _CHIP_PRED)] for (key, t), a in zip(JUNK, junk)]
    junk_label = w["junk"].format(limit=_dec(LIMIT, lang, 1))
    body = (_label(total) + _table(rows_, w["head"], numbers=(1, 4)) + _label(junk_label)
            + _table(junk_rows, w["junk_head"]) + _note(w["note"]))
    return yours.figure(w["title"], body, alt=w["alt"].format(
        title=w["title"], total=total, rows="; ".join(alts), junk=junk_label,
        junk_rows="; ".join(f"{JUNK_NAMES[lang][key]}: {a}" for (key, _), a in zip(JUNK, junk)), note=w["note"]))


# --- the pictures of the text after the exercises ----------------------------------------------------------------------

JOINS = (1, 2, 3, 5)

_LENGTH_WORDS = {
    "en": {"title": "Longer texts, fewer mistakes",
           "head": ["Sentences in one text", "Texts", "Right", ""],
           "scand": "Danish, Bokmål and Swedish, one sentence at a time: what the program answers",
           "scand_head": ["Language", "Danish", "Bokmål", "Swedish", "Other", '"?"'],
           "of": "of",
           "note": "Blue: the share of texts for which the course's program gives the right language, with profiles of 300 "
                   "trigrams and a limit of 0.9. The held-out sentences of each language are put together two, three "
                   "or five at a time.",
           "alt": "{title}. {rows}. {scand}: {scand_rows}. {note}"},
    "nl": {"title": "Langere teksten, minder fouten",
           "head": ["Zinnen in één tekst", "Teksten", "Goed", ""],
           "scand": "Deens, Bokmål en Zweeds, één zin tegelijk: wat het programma antwoordt",
           "scand_head": ["Taal", "Deens", "Bokmål", "Zweeds", "Andere", '"?"'],
           "of": "van",
           "note": "Blauw: het deel van de teksten waarvan het programma van de course de goede taal geeft, met "
                   "profielen van 300 trigrams en een grens van 0,9. De achtergehouden zinnen van elke taal zijn per "
                   "twee, drie of vijf samengevoegd.",
           "alt": "{title}. {rows}. {scand}: {scand_rows}. {note}"},
}
SCANDINAVIAN = ("da", "nb", "sv")


def length_picture(lang: str) -> str:
    w = _LENGTH_WORDS[lang]
    rows_, alts = [], []
    for j in JOINS:
        s = score(join=j)
        share = 100 * s["right"] / s["total"]
        rows_.append([str(j), _num(s["total"], lang), _pct(share, lang), _bar(share, 100, "var(--ea-pred)")])
        alts.append(f"{j}: {_pct(share, lang)} {w['of']} {_num(s['total'], lang)}")
    answers = score()["answers"]
    scand_rows, scand_alts = [], []
    for code in SCANDINAVIAN:
        mine = [a for c, _, a in answers if c == code]
        cells = [str(sum(a == other for a in mine)) for other in SCANDINAVIAN]
        rest = sum(a not in SCANDINAVIAN and a != "?" for a in mine)
        unknown = sum(a == "?" for a in mine)
        scand_rows.append([LANG_NAMES[lang][code]] + cells + [str(rest), str(unknown)])
        scand_alts.append(f"{LANG_NAMES[lang][code]}: " + ", ".join(cells) + f", {rest}, {unknown}")
    body = (_table(rows_, w["head"], numbers=(0, 1, 2)) + _label(w["scand"])
            + _table(scand_rows, w["scand_head"], numbers=(1, 2, 3, 4, 5)) + _note(w["note"]))
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows="; ".join(alts), scand=w["scand"],
                                                              scand_rows="; ".join(scand_alts), note=w["note"]))


_JUNK_WORDS = {
    "en": {"title": "Text outside the twelve languages",
           "head": ["Text", "", "Nearest language", "Distance", "Answer"],
           "none": "none",
           "note": "Purple: the distance to the nearest of the twelve profiles; *none* where no profile holds a single "
                   "trigram of the text. Cyan mark: the limit of 0.9, a setting chosen by hand, not learned. Blue: what the program gives "
                   "back; at the limit or beyond, \"?\".",
           "alt": "{title}. {rows}. {note}"},
    "nl": {"title": "Tekst buiten de twaalf talen",
           "head": ["Tekst", "", "Dichtstbijzijnde taal", "Afstand", "Antwoord"],
           "none": "geen",
           "note": "Paars: de afstand tot het dichtstbijzijnde van de twaalf profielen; *geen* waar geen profiel ook "
                   "maar één trigram van de tekst bevat. Cyaan streepje: de grens van 0,9, met de hand gekozen en niet geleerd. Blauw: "
                   "wat het programma teruggeeft; op de grens of erboven \"?\".",
           "alt": "{title}. {rows}. {note}"},
}


def junk_picture(lang: str) -> str:
    w = _JUNK_WORDS[lang]
    profiles = _profiles()
    rows_, alts = [], []
    for key, t in JUNK:
        doc = profile(t)
        dists = {c: distance(doc, profiles[c]) for c in LANGUAGES}
        near = min(LANGUAGES, key=lambda c: dists[c])
        answer = identify(t, profiles, LIMIT)
        name = LANG_NAMES[lang][near] if dists[near] < 1 else w["none"]   # at 1, no language shares a trigram
        rows_.append([html.escape(JUNK_NAMES[lang][key]), _text(t, 44), html.escape(name),
                      f'<span style="white-space:nowrap">{_dec(dists[near], lang)} '
                      f"{_limit_bar(dists[near], LIMIT, 'var(--ea-loss)', 6.0)}</span>", _chip(answer, _CHIP_PRED)])
        alts.append(f"{JUNK_NAMES[lang][key]}: {name} {_dec(dists[near], lang)}, {answer}")
    body = _table(rows_, w["head"], numbers=(3,)) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows="; ".join(alts), note=w["note"]))


# --- the demo: your own text -------------------------------------------------------------------------------------------

DEMO_LONGEST = 2000   # characters of the demo's text that the program looks at
DEMO_TEXT = "Ik ga morgen naar the meeting, see you there."   # the demo starts with Dutch and English mixed
DEMO_GRAMS = 40       # trigrams of the text that the demo shows

_DEMO_WORDS = {
    "en": {"title": "Which language is your text?",
           "answer": "Answer with a limit of {limit}:",
           "grams": "The trigrams of your text, the most common first: in the profile of {name}, or not",
           "more": "… and {k} more",
           "empty": "Type a text first.",
           "note": "Purple: the distance to each language, from 0 (every trigram in the same place) to 1 (none of "
                   "them in the profile). Cyan mark: the limit. Blue: the answer. Gray trigrams are in the profile "
                   "of the nearest language; dashed purple ones are not.",
           "alt": "{title}. {answer} {got}. {list}. {note}"},
    "nl": {"title": "Welke taal is je tekst?",
           "answer": "Antwoord met een grens van {limit}:",
           "grams": "De trigrams van je tekst, de vaakste eerst: in het profiel van het {name}, of niet",
           "more": "… en nog {k}",
           "empty": "Typ eerst een tekst.",
           "note": "Paars: de afstand tot elke taal, van 0 (elk trigram op dezelfde plaats) tot 1 (geen van de "
                   "trigrams in het profiel). Cyaan streepje: de grens. Blauw: het antwoord. Grijze trigrams staan in "
                   "het profiel van de dichtstbijzijnde taal, trigrams met een gestreepte paarse rand niet.",
           "alt": "{title}. {answer} {got}. {list}. {note}"},
}


def demo_picture(text, limit, lang: str) -> str:
    w = _DEMO_WORDS[lang]
    text = (text or "")[:DEMO_LONGEST]
    if not text.strip():
        return yours.figure(w["title"], _note(w["empty"]), alt=w["empty"])
    profiles = _profiles()
    doc = profile(text)
    dists = {c: distance(doc, profiles[c]) for c in LANGUAGES}
    order = sorted(LANGUAGES, key=lambda c: dists[c])
    got = identify(text, profiles, limit)
    near = order[0]
    rows_ = [[_chip(LANG_NAMES[lang][c], _CHIP_PRED if c == got else _CHIP_DATA), _dec(dists[c], lang),
              _limit_bar(dists[c], limit, "var(--ea-loss)")] for c in order]
    grams = list(doc)
    chips = [_gram(g, _CHIP_DATA if g in profiles[near] else _CHIP_MISSING) for g in grams[:DEMO_GRAMS]]
    if len(grams) > DEMO_GRAMS:
        chips.append(_chip(w["more"].format(k=_num(len(grams) - DEMO_GRAMS, lang)), _CHIP_FAINT))
    answer = w["answer"].format(limit=_dec(limit, lang))
    shown_answer = LANG_NAMES[lang].get(got, got)
    body = (f'<div style="{_LABEL}">{html.escape(answer)} {_chip(shown_answer, _CHIP_PRED)}</div>'
            + _table(rows_, [], numbers=(1,)) + _label(w["grams"].format(name=LANG_NAMES[lang][near]))
            + "<div>" + "".join(chips) + "</div>" + _note(w["note"]))
    return yours.figure(w["title"], body, alt=w["alt"].format(
        title=w["title"], answer=answer, got=shown_answer,
        list=", ".join(f"{LANG_NAMES[lang][c]} {_dec(dists[c], lang)}" for c in order), note=w["note"]))


# --- explore: the size of a profile and the n of the n-grams -----------------------------------------------------------

SIZES = (30, 100, 300, 1000, 3000)
NS = (1, 2, 3, 4, 5)

_EXPLORE_WORDS = {
    "en": {"title": "Profiles of at most {size} {grams}",
           "grams": {1: "single characters", 2: "bigrams", 3: "trigrams", 4: "4-grams", 5: "5-grams"},
           "head": ["n", "Right", '"?"', "Wrong language", ""],
           "sizes": "Trigrams, with profiles of each size",
           "size_head": ["Size", "Right", '"?"', "Wrong language", ""],
           "note": "Blue: right; gray: \"?\"; purple: a wrong language; on the {total} held-out sentences, with a "
                   "limit of 0.9. Framed: your choice.",
           "alt": "{title}. {rows}. {sizes}: {size_rows}. {note}"},
    "nl": {"title": "Profielen van hoogstens {size} {grams}",
           "grams": {1: "losse tekens", 2: "bigrams", 3: "trigrams", 4: "4-grams", 5: "5-grams"},
           "head": ["n", "Goed", '"?"', "Verkeerde taal", ""],
           "sizes": "Trigrams, met profielen van elke grootte",
           "size_head": ["Grootte", "Goed", '"?"', "Verkeerde taal", ""],
           "note": "Blauw: goed; grijs: \"?\"; paars: een verkeerde taal; op de {total} achtergehouden zinnen, met een "
                   "grens van 0,9. Omlijnd: je keuze.",
           "alt": "{title}. {rows}. {sizes}: {size_rows}. {note}"},
}


def _stack(s: dict, width: float = 10.0) -> str:
    """One bar in three parts: right (blue), "?" (gray), wrong (purple)."""
    total = s["total"]
    wrong = total - s["right"] - s["unknown"]
    parts = ""
    for k, color in ((s["right"], "var(--ea-pred)"), (s["unknown"], "var(--ea-data)"), (wrong, "var(--ea-loss)")):
        if k:
            parts += f'<span style="{_BAR};border-radius:0;width:{width * k / total:.2f}rem;background:{color}"></span>'
    return f'<span style="display:inline-block;white-space:nowrap">{parts}</span>'


def explore_picture(size, n, lang: str) -> str:
    w = _EXPLORE_WORDS[lang]
    size, n = int(size), int(n)
    title = w["title"].format(size=_num(size, lang), grams=w["grams"][n])

    def row(first: str, s: dict, mine: bool) -> list:
        wrong = s["total"] - s["right"] - s["unknown"]
        cell = _chip(first, _CHIP_DATA) if mine else html.escape(first)
        return [cell, _pct(100 * s["right"] / s["total"], lang), str(s["unknown"]), str(wrong), _stack(s)]
    rows_ = [row(str(k), score(size, k), k == n) for k in NS]
    size_rows = [row(_num(z, lang), score(z, N), z == size and n == N) for z in SIZES]
    total = score(size, n)["total"]
    note = w["note"].format(total=_num(total, lang))
    body = (_table(rows_, w["head"], numbers=(1, 2, 3)) + _label(w["sizes"])
            + _table(size_rows, w["size_head"], numbers=(1, 2, 3)) + _note(note))
    alt_rows = "; ".join(f"n {k}: {_pct(100 * score(size, k)['right'] / total, lang)}" for k in NS)
    alt_sizes = "; ".join(f"{_num(z, lang)}: {_pct(100 * score(z, N)['right'] / total, lang)}" for z in SIZES)
    return yours.figure(title, body, alt=w["alt"].format(title=title, rows=alt_rows, sizes=w["sizes"],
                                                         size_rows=alt_sizes, note=note))


# --- the exercises -----------------------------------------------------------------------------------------------------

REQUIRED = ["ngrams", "out-of-place", "identify", "what-comes-out"]

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
    if isinstance(v, (set, frozenset)):
        return "{" + ", ".join(_value(p) for p in sorted(v, key=str)) + "}" if v else "set()"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_value(k)}: {_value(x)}" for k, x in v.items()) + "}"
    return repr(v)


def _shown_value(v):
    """A value for a message: a long list or dictionary cut after 8 items, with how many more, in each language."""
    if isinstance(v, (list, dict)) and len(v) > 8:
        items = list(v)[:8] if isinstance(v, list) else [f"{_value(k)}: {_value(x)}" for k, x in list(v.items())[:8]]
        start = ("[" if isinstance(v, list) else "{") + ", ".join(_value(p) if isinstance(v, list) else p for p in items)
        end = "]" if isinstance(v, list) else "}"
        return {"en": start + f", ... and {len(v) - 8} more{end}", "nl": start + f", ... en nog {len(v) - 8}{end}"}
    return _value(v)


class Named:
    """A value of the check's own that a message shows under a name, such as the profiles of the lesson."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


def _lit(v) -> Named:
    return Named(_value(v), v)


def _profiles_value(name: str = "profiles"):
    return Named(name, _profiles())


def _profile_of(text: str) -> Named:
    return Named(f"profile({_value(text)})", profile(text))


def _language(code: str) -> Named:
    return Named(f"profiles[{_value(code)}]", _profiles()[code])


def _key_error(e) -> bool:
    return isinstance(e, KeyError)


def _a_dict(got, text) -> bool:
    return isinstance(got, dict)


def _a_float(got, text) -> bool:
    return isinstance(got, float)


def _a_number(got, text) -> bool:
    return isinstance(got, (int, float)) and not isinstance(got, bool)


# exercise 1: ngrams. The examples of the lesson, then short texts: one letter, none at all, n of 1 and 2, repeats,
# capitals, a longer n, and an n longer than the padded text
NGRAM_CASES = [
    (_lit("de kat"), _lit(3)),
    (_lit("the cat"), _lit(3)),
    (_lit("ik"), _lit(3)),
    (_lit("a"), _lit(3)),
    (_lit(""), _lit(3)),
    (_lit("ja"), _lit(1)),
    (_lit("huis"), _lit(2)),
    (_lit("aaa"), _lit(2)),
    (_lit("De Kat"), _lit(3)),
    (_lit("een fiets"), _lit(4)),
    (_lit("x"), _lit(5)),
]

_G_START = "def ngrams(text, n):\n    ...\n"
_G = ("def ngrams(text, n):\n"
      "    padded = \" \" + text + \" \"\n"
      "    grams = []\n"
      "    for i in range(len(padded) - n + 1):\n"
      "        grams.append(padded[i:i + n])\n"
      "    return grams\n")
_G_NO_PAD = _G.replace("padded = \" \" + text + \" \"", "padded = text")
_G_FRONT = _G.replace("padded = \" \" + text + \" \"", "padded = \" \" + text")
_G_BACK = _G.replace("padded = \" \" + text + \" \"", "padded = text + \" \"")
_G_LOWER = _G.replace("padded = \" \" + text + \" \"", "padded = \" \" + text.lower() + \" \"")
_G_NO_LAST = _G.replace("range(len(padded) - n + 1)", "range(len(padded) - n)")
_G_TOO_FAR = _G.replace("range(len(padded) - n + 1)", "range(len(padded))")
_G_TEXT_COUNT = _G.replace("range(len(padded) - n + 1)", "range(len(text) - n + 1)")
_G_TEXT_SLICE = _G.replace("padded[i:i + n]", "text[i:i + n]")
_G_SLICE_END = _G.replace("padded[i:i + n]", "padded[i:n]")
_G_RETURN_IN = _G.replace("    return grams\n", "        return grams\n")


# exercise 2: out_of_place. The small example of the lesson, a sentence against its own language, then: no trigrams,
# the same profile, nothing in common, profiles of different sizes, ranks in both directions, and a language whose
# profile holds trigrams that the text lacks
SMALL_DOC = {" de": 0, "de ": 1, "en ": 2, " ka": 3}
SMALL_LANG = {"en ": 0, " de": 1, "de ": 2, "et ": 3, " he": 4}
OOP_CASES = [
    (_lit(SMALL_DOC), _lit(SMALL_LANG)),
    (_profile_of("Ik fiets naar huis."), _language("nl")),
    (_profile_of("I am going home now."), _language("en")),
    (_lit({}), _lit(SMALL_LANG)),
    (_lit(SMALL_LANG), _lit(SMALL_LANG)),
    (_lit({"xyz": 0, "zyx": 1}), _lit(SMALL_LANG)),
    (_lit({"ab": 0, "bc": 1, "cd": 2}), _lit({"cd": 0, "ab": 1, "bc": 2})),
    (_lit({"ab": 3, "bc": 0}), _lit({"ab": 0, "bc": 5, "cd": 1, "de": 2, "ef": 3, "fg": 4, "gh": 6})),
    (_lit({"ab": 0}), _lit({"zz": 0, "yy": 1, "ab": 2, "xx": 3})),
    (_profile_of("Ik fiets naar huis."), _language("it")),
]

_O_START = "def out_of_place(doc, lang):\n    ...\n"
_O = ("def out_of_place(doc, lang):\n"
      "    total = 0\n"
      "    for gram in doc:\n"
      "        if gram in lang:\n"
      "            total = total + max(doc[gram], lang[gram]) - min(doc[gram], lang[gram])\n"
      "        else:\n"
      "            total = total + len(lang)\n"
      "    return total\n")
_O_SIGNED = _O.replace("max(doc[gram], lang[gram]) - min(doc[gram], lang[gram])", "doc[gram] - lang[gram]")
_O_SIGNED_BACK = _O.replace("max(doc[gram], lang[gram]) - min(doc[gram], lang[gram])", "lang[gram] - doc[gram]")
_O_FREE = _O.replace("        else:\n            total = total + len(lang)\n", "")
_O_DOC = _O.replace("total + len(lang)", "total + len(doc)")
_O_ONE = _O.replace("total + len(lang)", "total + 1")
_O_LANG_SIDE = _O.replace("for gram in doc:", "for gram in lang:").replace("if gram in lang:", "if gram in doc:")
_O_NO_CHECK = ("def out_of_place(doc, lang):\n"
               "    total = 0\n"
               "    for gram in doc:\n"
               "        total = total + max(doc[gram], lang[gram]) - min(doc[gram], lang[gram])\n"
               "    return total\n")
_O_RETURN_IN = _O.replace("    return total\n", "        return total\n")
_O_DIVIDED = _O.replace("    return total\n", "    return total / (len(doc) * len(lang))\n")

# exercise 3: identify. The examples of the lesson, then: an English sentence, a lower limit, no text at all, another
# alphabet, broken encoding of a Dutch sentence, small profiles of the check's own where the nearest language comes
# second or where every language lies beyond the limit, and a limit of 1
TINY = {"aa": {" a ": 0, " aa": 1, "aa ": 2}, "bb": {" b ": 0, " bb": 1, "bb ": 2, " bc": 3}}
IDENTIFY_CASES = [
    (_lit("Ik fiets naar huis."), _profiles_value(), _lit(0.9)),
    (_lit('<div class="nav"><a href="/home">Home</a></div>'), _profiles_value(), _lit(0.9)),
    (_lit("I am going home now."), _profiles_value(), _lit(0.9)),
    (_lit("Ik fiets naar huis."), _profiles_value(), _lit(0.6)),
    (_lit(""), _profiles_value(), _lit(0.9)),
    (_lit("Я люблю читать книги."), _profiles_value(), _lit(0.9)),
    (_lit(broken("Wij drinken koffie in het café.")), _profiles_value(), _lit(0.9)),
    (_lit("bb"), _lit(TINY), _lit(0.9)),
    (_lit("aa"), _lit(TINY), _lit(0.9)),
    (_lit("zz"), _lit(TINY), _lit(0.9)),
    (_lit("Ik fiets naar huis."), _profiles_value(), _lit(1.0)),
]

_I_START = "def identify(text, profiles, limit):\n    ...\n"
_I = ("def identify(text, profiles, limit):\n"
      "    doc = profile(text)\n"
      "    best = \"?\"\n"
      "    best_distance = limit\n"
      "    for lang in profiles:\n"
      "        d = distance(doc, profiles[lang])\n"
      "        if d < best_distance:\n"
      "            best = lang\n"
      "            best_distance = d\n"
      "    return best\n")
_I_NO_LIMIT = _I.replace("    best_distance = limit\n", "    best_distance = 2\n")
_I_FARTHEST = _I.replace("    best_distance = limit\n", "    best_distance = 0\n").replace("if d < best_distance:",
                                                                                         "if d > best_distance:")
_I_NOT_UPDATED = _I.replace("            best_distance = d\n", "")
_I_A_NUMBER = _I.replace("    return best\n", "    return best_distance\n")
_I_RAW = _I.replace("d = distance(doc, profiles[lang])", "d = out_of_place(doc, profiles[lang])")
_I_CODE = _I.replace("d = distance(doc, profiles[lang])", "d = distance(doc, lang)")
_I_TEXT = _I.replace("d = distance(doc, profiles[lang])", "d = distance(text, profiles[lang])")
_I_NONE = _I.replace("    best = \"?\"\n", "    best = None\n")
_I_RETURN_IN = _I.replace("    return best\n", "        return best\n")

_NAMES_I = {"profile": profile, "distance": distance, "out_of_place": out_of_place}

EXERCISES = {
    "ngrams": FunctionBox(
        name="ngrams",
        picture=ngrams_picture,
        function="ngrams",
        start=_G_START,
        solution=_G,
        visible=NGRAM_CASES[:2],
        hidden=NGRAM_CASES[2:],
        first=[_DOTS],
        show=_shown_value,
        mistakes=[
            Mistake("a-dict", matches=_a_dict,
                    en="`{call}` gives {got}: a dictionary of counts. Here your function gives the n-grams themselves, as a list in order, repeats included: {want}. The course counts them when it makes a profile.",
                    nl="`{call}` geeft {got}: een dictionary met aantallen. Hier geeft je functie de n-grams zelf, als een lijst op volgorde, met herhalingen: {want}. De course telt de n-grams pas bij het maken van een profiel."),
            Mistake("order-or-repeats", matches=lambda got, text: isinstance(got, (set, frozenset)),
                    en="`{call}` gives {got}: a set, without order and without repeats. Give back a list, with the n-grams in the order of the text and every repeat, since the counts of a profile need them: {want}.",
                    nl="`{call}` geeft {got}: een set, zonder volgorde en zonder herhalingen. Geef een lijst terug, met de n-grams in de volgorde van de tekst en elke herhaling, want de aantallen van een profiel hebben de herhalingen nodig: {want}."),
            Mistake("no-pad", wrong=_G_NO_PAD,
                    en="`{call}` gives {got}, and it should give {want}: your n-grams lack the spaces at the start and the end. Put a space before `text` and one after it, so that the n-grams show where the text starts and ends.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je n-grams missen de spaties aan het begin en het eind. Zet een spatie voor `text` en een erachter, zodat de n-grams laten zien waar de tekst begint en eindigt."),
            Mistake("no-last", wrong=_G_NO_LAST,
                    en="`{call}` gives {got}, and it should give {want}: the last n-gram is missing, the one with the space at the end. Is there a space after `text` too, and does your loop get to the last place where a run of `n` characters still fits?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het laatste n-gram ontbreekt, dat met de spatie aan het eind. Staat er ook een spatie achter `text`, en komt je loop bij de laatste plaats waar nog een rijtje van `n` tekens past?"),
            Mistake("space-back", wrong=_G_BACK,
                    en="`{call}` gives {got}, and it should give {want}: there is a space after `text`, but none before it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: er staat een spatie achter `text`, maar niet ervoor."),
            Mistake("lower", wrong=_G_LOWER,
                    en="`{call}` gives {got}, and it should give {want}: your function lowercases the text. `ngrams` gives the n-grams with the letters of the text itself; the course's `profile` lowercases the text before it cuts it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie zet de tekst in kleine letters. `ngrams` geeft de n-grams met de letters van de tekst zelf; `profile` van de course zet de tekst eerst in kleine letters en knipt de tekst daarna."),
            Mistake("too-far", wrong=_G_TOO_FAR,
                    en="`{call}` gives {got}, and it should give {want}: the last strings are shorter than `n` characters, because your loop goes on to the very end. Stop at the last place where a run of `n` characters still fits.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de laatste strings zijn korter dan `n` tekens, want je loop gaat door tot het eind. Stop bij de laatste plaats waar nog een rijtje van `n` tekens past."),
            Mistake("text-count", wrong=_G_TEXT_COUNT,
                    en="`{call}` gives {got}, and it should give {want}: the last n-grams are missing. Your loop counts the places of `text`, without the two spaces; count those of `padded`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de laatste n-grams ontbreken. Je loop telt de plaatsen van `text`, zonder de twee spaties; tel de plaatsen van `padded`."),
            Mistake("text-slice", wrong=_G_TEXT_SLICE,
                    en="`{call}` gives {got}, and it should give {want}: you take the n-grams from `text`, without the spaces. Take them from `padded`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je neemt de n-grams uit `text`, zonder de spaties. Neem de n-grams uit `padded`."),
            Mistake("slice-end", wrong=_G_SLICE_END,
                    en="`{call}` gives {got}, and it should give {want}: your n-grams do not all have `n` characters. A slice that starts at `i` and holds `n` characters ends at `i + n`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je n-grams hebben niet allemaal `n` tekens. Een slice die bij `i` begint en `n` tekens bevat, eindigt bij `i + n`."),
            Mistake("return-in-loop", wrong=_G_RETURN_IN,
                    en="`{call}` gives {got}, and it should give {want}: only the first n-gram. The `return` stands inside the loop, so the function stops after one round. Move it back, under `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: alleen het eerste n-gram. De `return` staat in de loop, dus de functie stopt na één ronde. Zet de `return` terug, onder `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: put a space before and after `text`, and give back every run of `n` characters of that, in order, as a list.",
                      "nl": "Vervang de `...`: zet een spatie voor en achter `text`, en geef elk rijtje van `n` tekens daarvan terug, op volgorde, als een lijst."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function put one space before `text` and one after, and take a run of `n` characters at every place where one still fits?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Zet je functie één spatie voor `text` en één erachter, en neemt je functie een rijtje van `n` tekens op elke plaats waar dat nog past?"},
            "almost": {"en": "Right for the examples of the lesson. But `{call}` gives {got}, and it should give {want}. Does your function work for every text and every `n`: one letter, no letters at all, `n` of 1, an `n` longer than the padded text?",
                       "nl": "Goed voor de voorbeelden van de les. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke tekst en elke `n`: één letter, helemaal geen letters, `n` gelijk aan 1, een `n` die langer is dan de tekst met de spaties?"},
            "correct": {"en": "Your `ngrams` is right in all {n} cases. Look at the picture: your n-grams of four short texts, and your trigrams of 400 English sentences, counted.",
                        "nl": "Je `ngrams` klopt in alle {n} gevallen. Kijk naar de afbeelding: je n-grams van vier korte teksten, en je trigrams van 400 Nederlandse zinnen, geteld."},
        },
    ),
    "out-of-place": FunctionBox(
        name="out-of-place",
        picture=oop_picture,
        function="out_of_place",
        start=_O_START,
        solution=_O,
        visible=OOP_CASES[:2],
        hidden=OOP_CASES[2:],
        first=[_DOTS],
        show=_shown_value,
        mistakes=[
            Mistake("lookup", error=_key_error,
                    en="With `{call}`, your function stops with a `KeyError`: it asks `lang` for the rank of a trigram that `lang` lacks. Ask first whether the trigram is a key of `lang`.",
                    nl="Met `{call}` stopt je functie met een `KeyError`: je functie vraagt `lang` om de rang van een trigram dat `lang` mist. Vraag eerst of het trigram een key van `lang` is."),
            Mistake("a-fraction", matches=_a_float,
                    en="`{call}` gives {got}: a fraction. Give back the sum itself, a whole number: {want}. The course's `distance` divides it.",
                    nl="`{call}` geeft {got}: een breuk. Geef de som zelf terug, een geheel getal: {want}. `distance` van de course deelt de som."),
            Mistake("signed", wrong=_O_SIGNED,
                    en="`{call}` gives {got}, and it should give {want}: a trigram whose rank in `doc` is lower than in `lang` takes something off the sum. How far out of place it is, is the larger of the two ranks minus the smaller.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een trigram met een lagere rang in `doc` dan in `lang` haalt iets van de som af. Hoe ver een trigram uit de plaats staat, is de grootste van de twee rangen min de kleinste."),
            Mistake("signed-back", wrong=_O_SIGNED_BACK,
                    en="`{call}` gives {got}, and it should give {want}: a trigram whose rank in `lang` is lower than in `doc` takes something off the sum. How far out of place it is, is the larger of the two ranks minus the smaller.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een trigram met een lagere rang in `lang` dan in `doc` haalt iets van de som af. Hoe ver een trigram uit de plaats staat, is de grootste van de twee rangen min de kleinste."),
            Mistake("missing-free", wrong=_O_FREE,
                    en="`{call}` gives {got}, and it should give {want}: in your function, a trigram that `lang` lacks adds nothing. It should add `len(lang)`, more than any trigram that `lang` has can be out of place.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: in je functie telt een trigram dat `lang` mist niets op. Dat trigram moet `len(lang)` optellen, meer dan een trigram uit `lang` ooit uit de plaats kan staan."),
            Mistake("missing-doc", wrong=_O_DOC,
                    en="`{call}` gives {got}, and it should give {want}: a trigram that `lang` lacks adds `len(doc)`. It should add `len(lang)`, the size of the language's profile.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een trigram dat `lang` mist, telt `len(doc)` op. Dat moet `len(lang)` zijn, de grootte van het profiel van de taal."),
            Mistake("missing-one", wrong=_O_ONE,
                    en="`{call}` gives {got}, and it should give {want}: a trigram that `lang` lacks adds 1, less than many trigrams that `lang` has. It should add `len(lang)`, the size of the language's profile.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een trigram dat `lang` mist, telt 1 op, minder dan veel trigrams die `lang` wel heeft. Dat moet `len(lang)` zijn, de grootte van het profiel van de taal."),
            Mistake("lang-side", wrong=_O_LANG_SIDE,
                    en="`{call}` gives {got}, and it should give {want}: your function goes over the trigrams of `lang`. Go over those of `doc`: the measure asks how far each trigram of the text lies from its place in the language.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie gaat langs de trigrams van `lang`. Ga langs de trigrams van `doc`: de maat vraagt hoe ver elk trigram van de tekst van de plaats in de taal ligt."),
            Mistake("return-in-loop", wrong=_O_RETURN_IN,
                    en="`{call}` gives {got}, and it should give {want}: the sum after the first trigram only. The `return` stands inside the loop, so the function stops after one round. Move it back, under `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de som na alleen het eerste trigram. De `return` staat in de loop, dus de functie stopt na één ronde. Zet de `return` terug, onder `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: for every trigram of `doc`, add how far its rank lies from its rank in `lang`, or `len(lang)` when `lang` lacks it, and give back the sum.",
                      "nl": "Vervang de `...`: tel voor elk trigram van `doc` op hoe ver de rang ligt van de rang in `lang`, of `len(lang)` als `lang` het trigram mist, en geef de som terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function go over the trigrams of `doc`, add the larger rank minus the smaller for a trigram that `lang` has, and `len(lang)` for one it lacks?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Gaat je functie langs de trigrams van `doc`, telt je functie de grootste rang min de kleinste op voor een trigram dat `lang` heeft, en `len(lang)` voor een trigram dat ontbreekt?"},
            "almost": {"en": "Right for the examples of the lesson. But `{call}` gives {got}, and it should give {want}. Does your function work for every pair of profiles: an empty `doc`, the same profile twice, nothing in common, profiles of different sizes?",
                       "nl": "Goed voor de voorbeelden van de les. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elk paar profielen: een lege `doc`, twee keer hetzelfde profiel, niets gemeen, profielen van verschillende grootte?"},
            "correct": {"en": "Your `out_of_place` is right in all {n} cases. Look at the picture: what each trigram of a sentence adds, and the distance to each of the twelve languages, with your function.",
                        "nl": "Je `out_of_place` klopt in alle {n} gevallen. Kijk naar de afbeelding: wat elk trigram van een zin optelt, en de afstand tot elk van de twaalf talen, met je functie."},
        },
    ),
    "identify": FunctionBox(
        name="identify",
        picture=identify_picture,
        function="identify",
        start=_I_START,
        solution=_I,
        names=_NAMES_I,
        visible=IDENTIFY_CASES[:2],
        hidden=IDENTIFY_CASES[2:],
        first=[_DOTS],
        show=_value,
        mistakes=[
            Mistake("a-number", matches=_a_number,
                    en="`{call}` gives {got}: a number. Give back the language itself, its key in `profiles`, such as `\"nl\"`, or `\"?\"`: {want}.",
                    nl="`{call}` geeft {got}: een getal. Geef de taal zelf terug, de key in `profiles`, zoals `\"nl\"`, of `\"?\"`: {want}."),
            Mistake("no-limit", wrong=_I_NO_LIMIT,
                    en="`{call}` gives {got}, and it should give {want}: your function picks the nearest language even when it lies far away. A language counts only when its distance is below `limit`; when none is, give back `\"?\"`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie kiest de dichtstbijzijnde taal, ook als de afstand groot is. Een taal telt alleen als de afstand onder `limit` ligt; als geen taal dat doet, geef dan `\"?\"` terug."),
            Mistake("farthest", wrong=_I_FARTHEST,
                    en="`{call}` gives {got}, and it should give {want}: your function keeps the language that lies farthest away. Keep a language when its distance is smaller than the best so far.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie houdt de taal die het verst weg ligt. Houd een taal als de afstand kleiner is dan de beste tot nu toe."),
            Mistake("not-updated", wrong=_I_NOT_UPDATED,
                    en="`{call}` gives {got}, and it should give {want}: your function keeps the last language below the limit, not the nearest. When you keep a language, does the distance to beat change too?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie houdt de laatste taal onder de grens, niet de dichtstbijzijnde. Als je een taal houdt, verandert dan ook de afstand die de volgende taal moet verslaan?"),
            Mistake("raw-sum", wrong=_I_RAW,
                    en="`{call}` gives {got}, and it should give {want}: your function compares the sum of `out_of_place` with `limit`. That sum runs into the thousands; `distance` divides it by the most it can be, so that it lies between 0 and 1, as `limit` does.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie vergelijkt de som van `out_of_place` met `limit`. Die som loopt in de duizenden; `distance` deelt de som door het hoogste dat de som kan zijn, zodat de afstand tussen 0 en 1 ligt, net als `limit`."),
            Mistake("not-profiles", wrong=_I_CODE,
                    en="`{call}` gives {got}, and it should give {want}: every distance comes out as 1, since your function gives `distance` something that is not a profile. `distance` takes two profiles: the profile of `text`, made with `profile`, and the profile of a language, which `profiles` holds under its code, such as `\"nl\"`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke afstand wordt 1, want je functie geeft `distance` iets wat geen profiel is. `distance` krijgt twee profielen: het profiel van `text`, gemaakt met `profile`, en het profiel van een taal, dat `profiles` bewaart onder de code, zoals `\"nl\"`."),
            Mistake("return-in-loop", wrong=_I_RETURN_IN,
                    en="`{call}` gives {got}, and it should give {want}: your function looks at the first language only. The `return` stands inside the loop, so the function stops after one round. Move it back, under `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie kijkt alleen naar de eerste taal. De `return` staat in de loop, dus de functie stopt na één ronde. Zet de `return` terug, onder `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: make the profile of `text`, measure its `distance` to the profile of each language in `profiles`, and give back the nearest language; `\"?\"` when no language lies below `limit`.",
                      "nl": "Vervang de `...`: maak het profiel van `text`, meet de `distance` tot het profiel van elke taal in `profiles`, en geef de dichtstbijzijnde taal terug; `\"?\"` als geen taal onder `limit` ligt."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function keep the language with the smallest distance, and only a language whose distance is below `limit`?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Houdt je functie de taal met de kleinste afstand, en alleen een taal met een afstand onder `limit`?"},
            "almost": {"en": "Right for the examples of the lesson. But `{call}` gives {got}, and it should give {want}. Does your function work for every text and limit: a lower limit, no text at all, other profiles, where the nearest language is not the first?",
                       "nl": "Goed voor de voorbeelden van de les. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke tekst en elke grens: een lagere grens, helemaal geen tekst, andere profielen, waarin de dichtstbijzijnde taal niet de eerste is?"},
            "unreached": {"en": "With `{call}`, your function gives back `None`, and it should give a language or `\"?\"`. What does `best` hold when no language lies below the limit? Let it start as `\"?\"`, before the loop. And does the function end with `return best`, after the loop?",
                          "nl": "Met `{call}` geeft je functie `None` terug, en dat moet een taal of `\"?\"` zijn. Wat staat er in `best` als geen taal onder de grens ligt? Laat `best` beginnen als `\"?\"`, voor de loop. En eindigt je functie met `return best`, na de loop?"},
            "correct": {"en": "Your `identify` is right in all {n} cases. Look at the picture: your program on the 1,190 sentences it has not seen, and on seven texts outside the twelve languages.",
                        "nl": "Je `identify` klopt in alle {n} gevallen. Kijk naar de afbeelding: je programma op de 1.190 zinnen die het niet eerder zag, en op zeven teksten buiten de twaalf talen."},
        },
    ),
    "what-comes-out": Sort(
        name="what-comes-out",
        groups={
            "right": {"en": "The language it is in", "nl": "De taal waarin het staat"},
            "other": {"en": "One of the twelve, but the wrong one", "nl": "Een van de twaalf, maar de verkeerde"},
            "unknown": {"en": "\"?\"", "nl": "\"?\""},
        },
        items={
            "greek": {"en": "**The Greek sentence *Μου αρέσει να διαβάζω βιβλία.*, *I like to read books.***",
                      "nl": "**De Griekse zin *Μου αρέσει να διαβάζω βιβλία.*, *Ik lees graag boeken.***"},
            "broken": {"en": "**The Dutch sentence *Ik heb Ã©Ã©n cafÃ© in BelgiÃ« gezien.*, whose *é* and *ë* came out broken**",
                       "nl": "**De Nederlandse zin *Ik heb Ã©Ã©n cafÃ© in BelgiÃ« gezien.*, waarvan de *é* en de *ë* kapot zijn**"},
            "finnish": {"en": "**The Finnish sentence *Tänään on todella kaunis ilma.*, *The weather is really nice today.***",
                        "nl": "**De Finse zin *Tänään on todella kaunis ilma.*, *Het is vandaag echt mooi weer.***"},
            "url": {"en": "**The web address *https://www.example.com/index.php?id=4821&lang=nl***",
                    "nl": "**Het webadres *https://www.example.com/index.php?id=4821&lang=nl***"},
            "market": {"en": "**Three Dutch sentences: *Gisteren ben ik met de fiets naar de markt gegaan. Daar heb ik appels, kaas en brood gekocht. Het was druk, maar het weer was mooi.***",
                       "nl": "**Drie Nederlandse zinnen: *Gisteren ben ik met de fiets naar de markt gegaan. Daar heb ik appels, kaas en brood gekocht. Het was druk, maar het weer was mooi.***"},
            "polish": {"en": "**The Polish sentence *Dzisiaj jest bardzo ładna pogoda.*, *The weather is very nice today.***",
                       "nl": "**De Poolse zin *Dzisiaj jest bardzo ładna pogoda.*, *Het is vandaag heel mooi weer.***"},
        },
        solution={"greek": "unknown", "broken": "right", "finnish": "other", "url": "unknown", "market": "right",
                  "polish": "other"},
        nudge={
            "greek": {"en": "Which of the twelve profiles holds a trigram in Greek letters? What distance does that give?",
                      "nl": "Welk van de twaalf profielen bevat een trigram in Griekse letters? Welke afstand geeft dat?"},
            "broken": {"en": "Only *é* and *ë* came out broken. What about the trigrams of *Ik heb*, *in* and *gezien*?",
                       "nl": "Alleen de *é* en de *ë* zijn kapot. Hoe zit het met de trigrams van *Ik heb*, *in* en *gezien*?"},
            "finnish": {"en": "Finnish is written with the same letters as the twelve. Can the program answer *Finnish*?",
                        "nl": "Het Fins wordt met dezelfde letters geschreven als de twaalf. Kan het programma *Fins* antwoorden?"},
            "url": {"en": "How many trigrams of the address, with its slashes, dots and question mark, does any language's profile hold?",
                    "nl": "Hoeveel trigrams van het adres, met de schuine strepen, punten en het vraagteken, staan in het profiel van een taal?"},
            "market": {"en": "Three sentences give many trigrams of Dutch. Is that a text where the program goes wrong?",
                       "nl": "Drie zinnen geven veel trigrams van het Nederlands. Is dat een tekst waar het programma fout gaat?"},
            "polish": {"en": "Polish is not one of the twelve, and is written mostly with the same letters. Can the program answer *Polish*?",
                       "nl": "Het Pools is niet een van de twaalf, en wordt vooral met dezelfde letters geschreven. Kan het programma *Pools* antwoorden?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the texts again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de teksten opnieuw uit."},
            "empty": {"en": "Put each bold text under what `identify` gives for it, with the twelve profiles and a limit of 0.9.",
                      "nl": "Zet elke vetgedrukte tekst bij wat `identify` ervoor geeft, met de twaalf profielen en een grens van 0,9."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item}: this text goes somewhere else.",
                      "nl": "{item}: deze tekst hoort ergens anders."},
            "correct": {"en": "All {n} are right. The limit catches text that shares almost no trigram with any of the twelve: other letters, such as Greek, and markup full of slashes and dots. A few broken letters do not hide a language, and a longer text gives more trigrams to go on. But a language that is written with the same letters gets the nearest of the twelve, and *?* only when its distance stays at or above the limit: the program can only say what it knows.",
                        "nl": "Alle {n} kloppen. De grens vangt tekst die bijna geen trigram met een van de twaalf gemeen heeft: andere letters, zoals Grieks, en markup vol schuine strepen en punten. Een paar kapotte letters verbergen een taal niet, en een langere tekst geeft meer trigrams om op af te gaan. Maar een taal die met dezelfde letters geschreven wordt, krijgt de dichtstbijzijnde van de twaalf, en alleen \"?\" als de afstand op of boven de grens blijft: het programma kan alleen zeggen wat het kent."},
        },
    ),
}

CASES = [
    ("ngrams", _G_START, "empty", None),
    ("ngrams", _G, "correct", None),
    ("ngrams", "def ngrams(text, n):\n    padded = \" \" + text + \" \"\n    return [padded[i:i + n] for i in range(len(padded) - n + 1)]\n", "correct", None),
    ("ngrams", "def ngrams(text, n):\n    padded = \" \" + text + \" \"\n    grams = []\n    i = 0\n    while i + n <= len(padded):\n        grams.append(padded[i:i + n])\n        i = i + 1\n    return grams\n", "correct", None),
    ("ngrams", _G.replace("    return grams\n", "    counts = {}\n    for g in grams:\n        counts[g] = 1\n    return counts\n"), "wrong", "a-dict"),
    ("ngrams", _G.replace("    return grams\n", "    return set(grams)\n"), "wrong", "order-or-repeats"),
    ("ngrams", _G_NO_PAD, "wrong", "no-pad"),
    ("ngrams", _G_FRONT, "wrong", "no-last"),
    ("ngrams", _G_BACK, "wrong", "space-back"),
    ("ngrams", _G_LOWER, "wrong", "lower"),
    ("ngrams", _G_NO_LAST, "wrong", "no-last"),
    ("ngrams", _G_TOO_FAR, "wrong", "too-far"),
    ("ngrams", _G_TEXT_COUNT, "wrong", "text-count"),
    ("ngrams", _G_TEXT_SLICE, "wrong", "text-slice"),
    ("ngrams", _G_SLICE_END, "wrong", "slice-end"),
    ("ngrams", _G_RETURN_IN, "wrong", "return-in-loop"),
    ("ngrams", _G.replace("    return grams\n", ""), "wrong", "no-return"),
    ("ngrams", "def ngrams(text, n):\n    if text == \"de kat\":\n        return [\" de\", \"de \", \"e k\", \" ka\", \"kat\", \"at \"]\n    return [\" th\", \"the\", \"he \", \"e c\", \" ca\", \"cat\", \"at \"]\n", "almost", None),
    ("out-of-place", _O_START, "empty", None),
    ("out-of-place", _O, "correct", None),
    ("out-of-place", "def out_of_place(doc, lang):\n    total = 0\n    for gram in doc:\n        if gram in lang:\n            total += abs(doc[gram] - lang[gram])\n        else:\n            total += len(lang)\n    return total\n", "correct", None),
    ("out-of-place", "def out_of_place(doc, lang):\n    return sum(abs(r - lang[g]) if g in lang else len(lang) for g, r in doc.items())\n", "correct", None),
    ("out-of-place", _O_NO_CHECK, "error", "lookup"),
    ("out-of-place", _O_DIVIDED, "wrong", "a-fraction"),
    ("out-of-place", _O_SIGNED, "wrong", "signed"),
    ("out-of-place", _O_SIGNED_BACK, "wrong", "signed-back"),
    ("out-of-place", _O_FREE, "wrong", "missing-free"),
    ("out-of-place", _O_DOC, "wrong", "missing-doc"),
    ("out-of-place", _O_ONE, "wrong", "missing-one"),
    ("out-of-place", _O_LANG_SIDE, "wrong", "lang-side"),
    ("out-of-place", _O_RETURN_IN, "wrong", "return-in-loop"),
    ("out-of-place", _O.replace("    return total\n", ""), "wrong", "no-return"),
    ("out-of-place", "def out_of_place(doc, lang):\n    if len(doc) == 4:\n        return 9\n    return 3841\n", "almost", None),
    ("identify", _I_START, "empty", None),
    ("identify", _I, "correct", None),
    ("identify", "def identify(text, profiles, limit):\n    doc = profile(text)\n    best = None\n    for lang in profiles:\n        if best is None or distance(doc, profiles[lang]) < distance(doc, profiles[best]):\n            best = lang\n    if best is None or distance(doc, profiles[best]) >= limit:\n        return \"?\"\n    return best\n", "correct", None),
    ("identify", "def identify(text, profiles, limit):\n    doc = profile(text)\n    best = min(profiles, key=lambda lang: distance(doc, profiles[lang]))\n    if distance(doc, profiles[best]) < limit:\n        return best\n    return \"?\"\n", "correct", None),
    ("identify", _I_TEXT, "wrong", "not-profiles"),
    ("identify", _I_A_NUMBER, "wrong", "a-number"),
    ("identify", _I_NO_LIMIT, "wrong", "no-limit"),
    ("identify", _I_FARTHEST, "wrong", "farthest"),
    ("identify", _I_NOT_UPDATED, "wrong", "not-updated"),
    ("identify", _I_RAW, "wrong", "raw-sum"),
    ("identify", _I_CODE, "wrong", "not-profiles"),
    ("identify", _I_RETURN_IN, "wrong", "return-in-loop"),
    ("identify", _I_NONE, "wrong", "unreached"),
    ("identify", "def identify(text, profiles, limit):\n    if text == \"Ik fiets naar huis.\":\n        return \"nl\"\n    return \"?\"\n", "almost", None),
    ("what-comes-out", {}, "empty", None),
    ("what-comes-out", {"greek": "unknown", "broken": "right", "finnish": "other", "url": "unknown", "market": "right",
                        "polish": "other"}, "correct", None),
    ("what-comes-out", {"greek": "unknown", "market": "right"}, "wrong", "unfinished"),
    ("what-comes-out", {"broken": "unknown"}, "wrong", "wrong:broken"),
    ("what-comes-out", {"greek": "unknown", "broken": "right", "finnish": "unknown", "url": "unknown", "market": "right",
                        "polish": "other"}, "wrong", "wrong:finnish"),
    ("what-comes-out", {"greek": "unknown", "broken": "right", "finnish": "other", "url": "right", "market": "right",
                        "polish": "other"}, "wrong", "wrong:url"),
]
