# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-information-extraction: the dates with a day, a month and a year that a pattern with three
groups finds in a text, as rows (year, month, day) (date_rows); the right, wrong and missed rows of a rule against the
course's list of every date (score); the rows written as ISO 8601 dates and sorted (iso_dates); a box for a pattern of
the learner's own on all 2,067 paragraphs; and a choice about why a pattern finds dates well and names of people badly.

The paragraphs are the development set of SQuAD 1.1 (Rajpurkar, Zhang, Lopyrev and Liang, EMNLP 2016; CC-BY-SA-4.0) in
English and in the Dutch of SQuAD-NL v1.1 (de Vries, Wieling and Nissim, EMNLP 2023; CC-BY-SA-4.0), in
data/xquad-squad-nl.json.gz, read with everything_ai.xquad. xquad.dates(lang) is the course's list of the 245 dates
in each language, the same dates in the same paragraphs. The names of people are those of Universal NER v1's English
Web Treebank (Mayhew and others, NAACL 2024; CC-BY-SA-4.0), in data/uner-en-ewt.tsv.gz, read with everything_ai.uner.
Paragraphs and sentences marked as not shown are counted but never shown. Every number in the text and the pictures is
computed by the course (tests/test_nlp_information_extraction.py computes them again)."""
import ast
import collections
import contextlib
import functools
import html
import io
import math
import re
import time
import tokenize
import warnings

from everything_ai import uner, xquad, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

LANGS = ("en", "nl")
PAGE_FIRST = 0      # the page Super Bowl 50: every date month first in English, day first in Dutch
PAGE_TIMELINE = 14  # the page Apollo program, Apolloprogramma: the timeline of exercise 3


def months(lang: str) -> dict:
    """The month names of `lang` with their numbers: {"January": 1, ...} and {"januari": 1, ...}."""
    return {name: k + 1 for k, name in enumerate(xquad.MONTHS[lang])}


MONTHS_EN = months("en")
MONTHS_NL = months("nl")

_HIDDEN = {"en": "(a paragraph the course does not show)", "nl": "(een alinea die de course niet laat zien)"}
_PRONOUN = {"en": re.compile(r"\b(he|she|him|her|his|hers|himself|herself)\b", flags=re.I),
            "nl": re.compile(r"\b(hij|zij|ze|haar|hem|hun|hen|zijn|z'n|diens|wiens)\b", flags=re.I)}


@functools.cache
def showable(i: int, lang: str) -> bool:
    """A paragraph the course shows: marked as shown, and without he or she (nor their Dutch) in it."""
    p = xquad.paragraphs(lang)[i]
    return p.shown and _PRONOUN[lang].search(p.text) is None


def text(lang: str, i: int) -> str:
    """Paragraph i of the 2,067 in `lang`, counting from 0; a paragraph marked as not shown gives a note instead."""
    p = xquad.paragraphs(lang)[i]
    return p.text if p.shown else _HIDDEN[lang]


def page(lang: str, i: int) -> str:
    return xquad.paragraphs(lang)[i].title


# --- the course's list of dates, and the rules the pictures compare with it -------------------------------------------

def gold_rows(lang: str) -> list:
    """Every date of the paragraphs as a row (paragraph, year, month, day)."""
    return [(d.paragraph, d.year, d.month, d.day) for d in xquad.dates(lang)]


def page_dates(lang: str, article: int) -> list:
    """The dates of one page, in the order of its text: (text as written, (year, month, day))."""
    ps = xquad.paragraphs(lang)
    return [(ps[d.paragraph].text[d.start:d.end], (d.year, d.month, d.day))
            for d in xquad.dates(lang) if ps[d.paragraph].article == article]


def _names(lang: str) -> str:
    return "|".join(xquad.MONTHS[lang])


def day_first(lang: str) -> str:
    """The pattern of exercise 1 for `lang`: a day, a month name and a year, each in a group."""
    return r"\b(\d{1,2}) (" + _names(lang) + r") (\d{4})\b"


def month_first(lang: str) -> str:
    return r"\b(" + _names(lang) + r") (\d{1,2}), (\d{4})\b"


ANY_WORD = r"\b(\d{1,2}) ([A-Za-z]+) (\d{4})\b"   # a day, any word, a year: the loose rule of the pictures


@functools.cache
def _rule_rows(lang: str, rule: str) -> tuple:
    """The rows (paragraph, year, month, day) of one of the course's rules on all paragraphs: day first, any word (a
    word that is not a month stays a word, so its row is wrong), or day first and month first."""
    number = months(lang)
    out = []
    for i, p in enumerate(xquad.paragraphs(lang)):
        if rule in ("day-first", "both"):
            out += [(i, int(y), number[m], int(d)) for d, m, y in re.findall(day_first(lang), p.text)]
        if rule == "both":
            out += [(i, int(y), number[m], int(d)) for m, d, y in re.findall(month_first(lang), p.text)]
        if rule == "any-word":
            out += [(i, int(y), number.get(m, m), int(d)) for d, m, y in re.findall(ANY_WORD, p.text)]
    return tuple(out)


def rule_rows(lang: str, rule: str) -> list:
    return list(_rule_rows(lang, rule))


RULES = ("day-first", "any-word", "both")


def counts(found: list, gold: list) -> tuple:
    """(right, wrong, missed) of a list of rows against the right ones, as exercise 2 counts them."""
    f, g = set(found), set(gold)
    return len(f & g), len(f - g), len(g - f)


# --- the course's solutions --------------------------------------------------------------------------------------------

_DR_HEAD = 'import re\n\n\ndef date_rows(text, months):\n    names = "|".join(months)\n'
_DR_PATTERN = r'r"\b(\d{1,2}) (" + names + r") (\d{4})\b"'


def _dr(pattern: str = _DR_PATTERN, add: str = "(int(year), months[month], int(day))",
        end: str = "    return rows\n") -> str:
    return (_DR_HEAD + f"    pattern = {pattern}\n    rows = []\n    for day, month, year in re.findall(pattern, text):\n"
            f"        rows.append({add})\n" + end)


_DR = _dr()
_DR_START = (_DR_HEAD + "    pattern = ...\n    rows = []\n    for day, month, year in re.findall(pattern, text):\n"
             "        ...\n    return rows\n")

_SC_HEAD = "def score(found, gold):\n    found = set(found)\n    gold = set(gold)\n    right = 0\n    for row in found:\n"


def _sc(test: str = "row in gold", end: str = "    return right, len(found) - right, len(gold) - right\n") -> str:
    return _SC_HEAD + f"        if {test}:\n            right = right + 1\n" + end


_SC = _sc()
_SC_START = _SC_HEAD + "        ...\n    return ...\n"

_ISO_HEAD = "def iso_dates(rows):\n    dates = []\n    for year, month, day in rows:\n"


def _iso(add: str = 'f"{year}-{month:02d}-{day:02d}"', end: str = "    return sorted(dates)\n") -> str:
    return _ISO_HEAD + f"        dates.append({add})\n" + end


_ISO = _iso()
_ISO_START = _ISO_HEAD + "        ...\n    return ...\n"


def _course(code: str, name: str):
    space: dict = {}
    exec(compile(code, f"<course {name}>", "exec"), space)
    return space[name]


date_rows = _course(_DR, "date_rows")
score = _course(_SC, "score")
iso_dates = _course(_ISO, "iso_dates")


# --- the examples of the exercises ---------------------------------------------------------------------------------------

class _Val:
    """A test value that the check's messages write by its name, such as `sentence_en`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value, memo)


class _Lazy(_Val):
    """A test value made only when a check needs it, such as the rows of a rule on all paragraphs; written by its name."""

    def __init__(self, name: str, make):
        super().__init__(name, None)
        self.make = make

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.make(), memo)


SENTENCE_EN = "Doctor Who originally ran for 26 seasons on BBC One, from 23 November 1963 until 6 December 1989."
SENTENCE_NL = "Doctor Who liep oorspronkelijk 26 seizoenen op BBC One, van 23 november 1963 tot 6 december 1989."
SENTENCE_PARAGRAPH = 1458   # where both sentences come from
FOUND_ROWS = [(1963, 11, 23), (1989, 12, 7), (2005, 3, 26)]
GOLD_ROWS = [(1963, 11, 23), (1989, 12, 6), (2005, 3, 26), (2008, 7, 27)]
APOLLO_ROWS = [(1969, 7, 20), (1961, 5, 25), (1967, 1, 27)]
CASE_PARAGRAPHS = (371, 6)   # Martin Luther: two dates day first and one written another way; Super Bowl 50: two dates

_SE = _Val("sentence_en", SENTENCE_EN)
_SN = _Val("sentence_nl", SENTENCE_NL)
_ME = _Val("months_en", MONTHS_EN)
_MN = _Val("months_nl", MONTHS_NL)
_FR = _Val("found_rows", FOUND_ROWS)
_GR = _Val("gold_rows", GOLD_ROWS)
_AR = _Val("apollo_rows", APOLLO_ROWS)


def _paragraph_value(lang: str, i: int) -> _Lazy:
    return _Lazy(f'C.text("{lang}", {i})', lambda: text(lang, i))


_ROWS = {(lang, rule): _Lazy(f"{rule.replace('-', '_')}_{lang}", lambda lang=lang, rule=rule: rule_rows(lang, rule))
         for lang in LANGS for rule in RULES}
_GOLD = {lang: _Lazy(f"gold_{lang}", lambda lang=lang: gold_rows(lang)) for lang in LANGS}
_TIMELINE = _Lazy("apollo_program", lambda: [r for _, r in page_dates("en", PAGE_TIMELINE)])
_ALL = _Lazy("all_dates_nl", lambda: [r[1:] for r in gold_rows("nl")])


# --- the pictures: a few helpers ------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);margin-top:0.45rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;vertical-align:middle;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap"
_CHIP = ("display:inline-block;margin:0.12rem 0.2rem 0.12rem 0;padding:0 0.3rem;border-radius:3px;"
         "font-family:var(--ea-font-code);font-size:0.8rem;line-height:1.5;white-space:pre")
_CHIP_DATA = _CHIP + ";background:var(--ea-data-wash);color:var(--ea-data-text)"
_CHIP_INK = _CHIP + ";border:1px solid var(--ea-line);color:var(--ea-ink)"
_CHIP_PRED = _CHIP + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
_CHIP_PURPLE = _CHIP + ";background:var(--ea-loss-wash);color:var(--ea-loss-text)"
_CHIP_MISSING = _CHIP + ";border:1px dashed var(--ea-loss);color:var(--ea-loss-text)"
_BAR = "display:inline-block;height:0.6rem;border-radius:2px;vertical-align:middle"
_DASHED_BAR = "display:inline-block;height:0.6rem;border-radius:2px;vertical-align:middle;border:1px dashed var(--ea-loss)"
_FAINT = "color:var(--ea-ink-3)"
_QUOTE = ("font-family:var(--ea-font-ui);font-size:0.84rem;line-height:1.9;color:var(--ea-ink-2);max-width:44rem;"
          "margin:0.2rem 0")
_MARK = "outline:1px solid var(--ea-ink-3);outline-offset:1px;border-radius:3px;color:var(--ea-ink)"
_HIT = "background:var(--ea-pred-wash);color:var(--ea-pred-text);border-radius:3px;padding:0 0.1rem"
_WRONG_HIT = "background:var(--ea-loss-wash);color:var(--ea-loss-text);border-radius:3px;padding:0 0.1rem"
_MISSED = "outline:1px dashed var(--ea-loss);outline-offset:1px;border-radius:3px;color:var(--ea-loss-text)"


def _marked(text_: str) -> str:
    """Escaped text in which `name` is set as code and *word* in italics."""
    out = []
    for k, part in enumerate(html.escape(text_).split("`")):
        if k % 2:
            out.append(f"<code>{part}</code>")
        else:
            out.append("".join(f"<i>{p}</i>" if j % 2 else p for j, p in enumerate(part.split("*"))))
    return "".join(out)


def _note(text_: str) -> str:
    """A line under a picture; a word between asterisks is set in italics and a name between backticks as code."""
    return f'<div style="{_UI}">{_marked(text_)}</div>'


def _label(text_: str) -> str:
    return f'<div style="{_LABEL}">{_marked(text_)}</div>'


def _num(n, lang: str) -> str:
    """A whole number with a separator for thousands: 2,067 in English, 2.067 in Dutch."""
    out = f"{round(n):,}"
    return out.replace(",", ".") if lang == "nl" else out


def _dec(x: float, lang: str, digits: int = 2) -> str:
    """A number with `digits` decimals: 0.54 in English, 0,54 in Dutch."""
    out = f"{round(float(x), digits) + 0.0:.{digits}f}"
    return out.replace(".", ",") if lang == "nl" else out


def _chip(text_: str, style: str = _CHIP_INK) -> str:
    return f'<span style="{style}">{html.escape(text_)}</span>'


def _bar(x: float, most: float, color: str, width: float = 6.0, dashed: bool = False) -> str:
    """A bar for a number of 0 or more: as long as `width` for `most`, never shorter than a sliver."""
    w = 0 if not most or not x or x <= 0 else max(0.15, width * min(x, most) / most)
    if not w:
        return ""
    if dashed:
        return f'<span style="{_DASHED_BAR};width:{w:.2f}rem"></span>'
    return f'<span style="{_BAR};width:{w:.2f}rem;background:{color}"></span>'


def _table(rows_: list, head: list, numbers_: tuple = ()) -> str:
    """A small table; rows hold HTML that is already escaped; numbers_ are the columns set to the right."""
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers_ else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{_NUMBER if j in numbers_ else ""}">{c}</td>'
                                    for j, c in enumerate(r)) + "</tr>" for r in rows_)
    head_html = f"<thead><tr>{top}</tr></thead>" if head else ""
    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;margin:0.3rem 0 0.2rem">'
            f"{head_html}<tbody>{body}</tbody></table></div>")


def _whole(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _row_ok(r, n: int = 3) -> bool:
    return isinstance(r, (tuple, list)) and len(r) == n and all(_whole(x) for x in r)


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error
_SLOW = object()      # what _call gives for a function that ran out of time
PICTURE_SECONDS = 5   # what the learner's function gets for all the calls of one picture together


def _call(make, seconds: float = PICTURE_SECONDS):
    """make() within a time limit, with what it prints thrown away; _SLOW when the time runs out, _STOPPED when it
    stops with an error."""
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return with_time_limit(make, max(seconds, 0.05))
    except Timeout:
        return _SLOW
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _left(deadline: float) -> float:
    """The seconds a picture has left until `deadline`, a time.monotonic()."""
    return deadline - time.monotonic()


def _empty(w: dict, key: str) -> str:
    """The picture that stays empty, with why: the learner's function stopped, was too slow, or gave something odd."""
    text_ = w[key].format(s=PICTURE_SECONDS)
    return yours.figure(w["title"], _note(text_), alt=text_)


# --- the idea: from four sentences to a table ------------------------------------------------------------------------------

IDEA_SENTENCES = {
    "en": ((133, "On 1 May 1191, Richard's fleet arrived in the port of Limassol on Cyprus."),
           (151, "Tesla died on 7 January 1943."),
           (730, "Armstrong and Aldrin performed the first landing at the Sea of Tranquility at 20:17:40 UTC on "
                 "July 20, 1969."),
           (1813, "Queen Elizabeth II opened the new building on 9 October 2004.")),
    "nl": ((133, "Op 1 mei 1191 arriveerde de vloot van Richard in de haven van Limassol op Cyprus."),
           (151, "Tesla stierf op 7 januari 1943."),
           (730, "Armstrong en Aldrin voerden de eerste landing uit op de Sea of Tranquility om 20:17:40 UTC op "
                 "20 juli 1969."),
           (1813, "Koningin Elizabeth II opende het nieuwe gebouw op 9 oktober 2004.")),
}

_IDEA_TEXT = {
    "en": {"title": "From text to a table",
           "text": "The text: four sentences from four pages",
           "table": "The table: a row for each date, and the date in one fixed form",
           "head": ["Page", "As written", "Year", "Month", "Day", "ISO 8601"],
           "note": "Gray: the sentences and the dates as written, data, with each date framed in its sentence. In the "
                   "color of the text: the rest of the table the course made from them, with a column for each field.",
           "alt": "{title}. {rows}"},
    "nl": {"title": "Van tekst naar een tabel",
           "text": "De tekst: vier zinnen van vier pagina's",
           "table": "De tabel: een rij voor elke datum, en de datum in één vaste vorm",
           "head": ["Pagina", "Zoals geschreven", "Jaar", "Maand", "Dag", "ISO 8601"],
           "note": "Grijs: de zinnen en de datums zoals geschreven, data, met elke datum omlijnd in de zin. In de kleur "
                   "van de tekst: de rest van de tabel die de course ervan maakte, met een kolom voor elk veld.",
           "alt": "{title}. {rows}"},
}


def idea_dates(lang: str) -> list:
    """(page, sentence, start and end of the date in the sentence, date text, (year, month, day)) for each date of
    the idea's sentences, from the course's list of dates."""
    ps = xquad.paragraphs(lang)
    out = []
    for i, sentence in IDEA_SENTENCES[lang]:
        where = ps[i].text.index(sentence)
        for d in xquad.dates(lang):
            if d.paragraph == i and where <= d.start and d.end <= where + len(sentence):
                out.append((ps[i].title, sentence, d.start - where, d.end - where, ps[i].text[d.start:d.end],
                            (d.year, d.month, d.day)))
    return out


def idea_picture(lang: str) -> str:
    w = _IDEA_TEXT[lang]
    found = idea_dates(lang)
    lines, rows_, alts = [], [], []
    for title, sentence, a, b, written, (y, m, d) in found:
        lines.append(f'<div style="{_QUOTE};color:var(--ea-data-text)">{html.escape(sentence[:a])}'
                     f'<span style="{_MARK};color:var(--ea-data-text)">'
                     f'{html.escape(sentence[a:b])}</span>{html.escape(sentence[b:])}</div>')
        iso = f"{y}-{m:02d}-{d:02d}"
        rows_.append([html.escape(title), _chip(written, _CHIP_DATA), str(y), str(m), str(d), _chip(iso)])
        alts.append(f"{title}: {written}, {iso}")
    body = (_label(w["text"]) + "".join(lines) + _label(w["table"]) + _table(rows_, w["head"], (2, 3, 4))
            + _note(w["note"]))
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows="; ".join(alts)))


# --- exercise 1: the dates of all paragraphs, from the learner's date_rows ---------------------------------------------

SHOWN_PAGES = 10   # the pages with the most dates in the picture; the others are added up in one row

_ROWS_TEXT = {
    "en": {"title": "The dates of all 2,067 paragraphs, from your date_rows",
           "sum": "Your `date_rows` finds {right} of the {n} dates of the English paragraphs. It misses {missed}, and "
                  "gives {wrong} rows that are not dates in the text.",
           "head": ["Page", "Dates", "Found and missed", "Wrong"],
           "others": "The other {k} pages",
           "first": "The dates of the page *{page}*",
           "note": "Your `date_rows(text, months_en)` on every English paragraph. Blue: the dates it finds. Purple, "
                   "dashed: the dates it misses. Purple: its rows that are not dates in the text.",
           "stopped": "Your function stops with an error on one of the paragraphs, so the picture stays empty.",
           "slow": "Your function is still not done after {s} seconds on all the paragraphs, so the picture stays "
                   "empty.",
           "odd": "Your function does not give a list of rows of three whole numbers for every paragraph, so the "
                  "picture stays empty.",
           "alt": "{sum} {pages}"},
    "nl": {"title": "De datums van alle 2.067 alinea's, uit je date_rows",
           "sum": "Je `date_rows` vindt {right} van de {n} datums van de Nederlandse alinea's. De functie mist er "
                  "{missed}, en geeft {wrong} rijen die geen datum uit de tekst zijn.",
           "head": ["Pagina", "Datums", "Gevonden en gemist", "Fout"],
           "others": "De andere {k} pagina's",
           "first": "De datums van de pagina *{page}*",
           "note": "Je `date_rows(text, months_nl)` op elke Nederlandse alinea. Blauw: de datums die de functie "
                   "vindt. Paars, gestippeld: de datums die de functie mist. Paars: rijen van de functie die geen "
                   "datum uit de tekst zijn.",
           "stopped": "Je functie stopt met een fout op een van de alinea's, dus de afbeelding blijft leeg.",
           "slow": "Je functie is na {s} seconden nog niet klaar met alle alinea's, dus de afbeelding blijft leeg.",
           "odd": "Je functie geeft niet voor elke alinea een lijst van rijen van drie gehele getallen, dus de "
                  "afbeelding blijft leeg.",
           "alt": "{sum} {pages}"},
}


def compare_rows(found_per: list, lang: str) -> tuple:
    """The rows of a date_rows for each paragraph against the course's dates: ({article: [right, missed, wrong]},
    the set of (paragraph, start) of the dates found, the total right, missed and wrong)."""
    ps = xquad.paragraphs(lang)
    gold = collections.defaultdict(list)
    for d in xquad.dates(lang):
        gold[d.paragraph].append(d)
    per = collections.defaultdict(lambda: [0, 0, 0])
    hit = set()
    for i, rows_ in enumerate(found_per):
        mine = collections.Counter(tuple(r) for r in rows_)
        for d in gold[i]:
            key = (d.year, d.month, d.day)
            if mine[key] > 0:
                mine[key] -= 1
                hit.add((i, d.start))
                per[ps[i].article][0] += 1
            else:
                per[ps[i].article][1] += 1
        per[ps[i].article][2] += sum(v for v in mine.values() if v > 0)
    totals = [sum(v[k] for v in per.values()) for k in range(3)]
    return dict(per), hit, totals


def _found(r: int) -> str:
    return f'<span style="color:var(--ea-pred-text);display:inline-block;min-width:1.6rem">{r}</span>'


def rows_picture(seen) -> str:
    """The learner's date_rows on every paragraph in the lesson's language: how many dates it finds, misses and gets
    wrong, per page, and the dates of the page Super Bowl 50."""
    lang = seen.lang
    w = _ROWS_TEXT[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    ps = xquad.paragraphs(lang)
    number = months(lang)
    out = _call(lambda: [fn(p.text, dict(number)) for p in ps])
    if out is _STOPPED or out is _SLOW:
        return _empty(w, "stopped" if out is _STOPPED else "slow")
    if not all(isinstance(r, list) and all(_row_ok(x) for x in r) for r in out):
        return _empty(w, "odd")
    per, hit, (right, missed, wrong) = compare_rows(out, lang)
    n = len(xquad.dates(lang))
    gold_per = collections.Counter(ps[d.paragraph].article for d in xquad.dates(lang))
    order = sorted(per, key=lambda a: (-(gold_per[a] + per[a][2]), a))
    most = max(max(gold_per.values()), sum(gold_per[a] for a in order[SHOWN_PAGES:]))
    rows_, alts = [], []
    for a in order[:SHOWN_PAGES]:
        r, m, wr = per[a]
        rows_.append([html.escape(xquad.titles(lang)[a]), str(gold_per[a]),
                      _found(r) + _bar(r, most, "var(--ea-pred)") + _bar(m, most, "", dashed=True),
                      f'<span style="color:var(--ea-loss-text)">{wr}</span>' if wr else ""])
        alts.append(f"{xquad.titles(lang)[a]} {r}/{gold_per[a]}")
    rest = order[SHOWN_PAGES:]
    if rest:
        r, m, wr = (sum(per[a][k] for a in rest) for k in range(3))
        rows_.append([f'<span style="{_FAINT}">{html.escape(w["others"].format(k=len(rest)))}</span>',
                      str(sum(gold_per[a] for a in rest)),
                      _found(r) + _bar(r, most, "var(--ea-pred)") + _bar(m, most, "", dashed=True),
                      f'<span style="color:var(--ea-loss-text)">{wr}</span>' if wr else ""])
    first = []
    for d in xquad.dates(lang):
        if ps[d.paragraph].article == PAGE_FIRST:
            written = ps[d.paragraph].text[d.start:d.end]
            first.append(_chip(written, _CHIP_PRED if (d.paragraph, d.start) in hit else _CHIP_MISSING))
    say = w["sum"].format(right=_num(right, lang), n=_num(n, lang), missed=_num(missed, lang), wrong=_num(wrong, lang))
    body = (f'<div style="{_QUOTE}">{_marked(say)}</div>'
            + _table(rows_, w["head"], (1, 3))
            + _label(w["first"].format(page=xquad.titles(lang)[PAGE_FIRST])) + " ".join(first) + _note(w["note"]))
    return yours.figure(w["title"], body, alt=w["alt"].format(sum=say.replace("`", ""), pages="; ".join(alts)))


# --- exercise 2: three rules against the course's dates, from the learner's score --------------------------------------

_SCORE_TEXT = {
    "en": {"title": "Three rules on all 2,067 paragraphs, counted by your score",
           "head": ["Rule", "Pattern", "Right", "Wrong", "Missed", "Precision", "Recall"],
           "rules": {"day-first": "Day first", "any-word": "Day first, any word", "both": "Day or month first"},
           "missed": "Of the {k} dates that *Day first* misses, {mdy} are written month first, such as {e1}, and {other} "
                     "in other ways, such as {e2}.",
           "note": "The course's rules on every English paragraph, counted against the course's {n} dates by your "
                   "`score`. Blue: your counts, with precision and recall worked out from them. Purple: where a "
                   "count of yours differs, with the right one dashed beside it.",
           "stopped": "Your function stops with an error on the rows of a rule, so the picture stays empty.",
           "slow": "Your function is still not done after {s} seconds on the rows of the rules, so the picture stays "
                   "empty.",
           "odd": "Your function does not give three whole numbers (right, wrong, missed), so the picture stays empty.",
           "alt": "{title}: {rows}"},
    "nl": {"title": "Drie regels op alle 2.067 alinea's, geteld door je score",
           "head": ["Regel", "Patroon", "Goed", "Fout", "Gemist", "Precision", "Recall"],
           "rules": {"day-first": "Dag eerst", "any-word": "Dag eerst, elk woord", "both": "Dag of maand eerst"},
           "missed": "De {k} datums die *Dag eerst* mist, zijn op een andere manier geschreven: {list}.",
           "note": "De regels van de course op elke Nederlandse alinea, geteld tegen de {n} datums van de course door "
                   "je `score`. Blauw: jouw aantallen, met de precision en de recall die eruit volgen. Paars: waar een "
                   "aantal van jou afwijkt, met het goede gestippeld ernaast.",
           "stopped": "Je functie stopt met een fout op de rijen van een regel, dus de afbeelding blijft leeg.",
           "slow": "Je functie is na {s} seconden nog niet klaar met de rijen van de regels, dus de afbeelding blijft "
                   "leeg.",
           "odd": "Je functie geeft geen drie gehele getallen (goed, fout, gemist), dus de afbeelding blijft leeg.",
           "alt": "{title}: {rows}"},
}

_SHORT_PATTERN = {"day-first": r"\b(\d{1,2}) (month) (\d{4})\b", "any-word": ANY_WORD,
                  "both": r"day first | \b(month) (\d{1,2}), (\d{4})\b"}
_SHORT_MONTH = {"en": "month", "nl": "maand"}
_SHORT_DAY_FIRST = {"en": "day first", "nl": "dag eerst"}


def short_pattern(rule: str, lang: str) -> str:
    """The pattern of a rule with its month names written as one word, so that it fits in the table."""
    return (_SHORT_PATTERN[rule].replace("(month)", f"({_SHORT_MONTH[lang]})")
            .replace("day first", _SHORT_DAY_FIRST[lang]))


def missed_by_form(lang: str) -> dict:
    """The dates that the rule Day first misses, grouped by their form: {"mdy": [...], "other": [...]}, each a list of
    the texts as written."""
    ps = xquad.paragraphs(lang)
    found = set(rule_rows(lang, "day-first"))
    out = {"mdy": [], "other": []}
    for d in xquad.dates(lang):
        if (d.paragraph, d.year, d.month, d.day) not in found:
            out[d.form].append(ps[d.paragraph].text[d.start:d.end])
    return out


def _ratio(a, b, lang: str) -> str:
    return _dec(a / b, lang) if b else "–"


def score_picture(seen) -> str:
    """The learner's score for three of the course's rules on every paragraph in the lesson's language, with the
    precision and the recall that follow from it, and where the misses of the first rule come from."""
    lang = seen.lang
    w = _SCORE_TEXT[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    gold = gold_rows(lang)
    deadline = time.monotonic() + PICTURE_SECONDS
    rows_, alts = [], []
    for rule in RULES:
        found = rule_rows(lang, rule)
        out = _call(lambda: fn(list(found), list(gold)), _left(deadline))
        if out is _STOPPED or out is _SLOW:
            return _empty(w, "stopped" if out is _STOPPED else "slow")
        if not (isinstance(out, (tuple, list)) and len(out) == 3 and all(_whole(x) and x >= 0 for x in out)):
            return _empty(w, "odd")
        want = counts(found, gold)
        cells = []
        for k, (mine, right) in enumerate(zip(out, want)):
            if mine == right:
                style = "color:var(--ea-pred-text)"
                cells.append(f'<span style="{style}">{_num(mine, lang)}</span>')
            else:
                cells.append(_chip(_num(mine, lang), _CHIP_PURPLE) + _chip(_num(right, lang), _CHIP_MISSING))
        r, wr, m = out
        rows_.append([html.escape(w["rules"][rule]), f"<code>{html.escape(short_pattern(rule, lang))}</code>", *cells,
                      f'<span style="color:var(--ea-pred-text)">{_ratio(r, r + wr, lang)}</span>',
                      f'<span style="color:var(--ea-pred-text)">{_ratio(r, r + m, lang)}</span>'])
        alts.append(f"{w['rules'][rule]}: {r}, {wr}, {m}")
    miss = missed_by_form(lang)
    k = len(miss["mdy"]) + len(miss["other"])
    if lang == "en":
        said = w["missed"].format(k=_num(k, lang), mdy=_num(len(miss["mdy"]), lang), e1=miss["mdy"][0],
                                  other=_num(len(miss["other"]), lang), e2=miss["other"][2])
    else:
        said = w["missed"].format(k=_num(k, lang), list=", ".join(miss["other"]))
    body = (_table(rows_, w["head"], (2, 3, 4, 5, 6)) + f'<div style="{_QUOTE}">{_marked(said)}</div>'
            + _note(w["note"].format(n=_num(len(gold), lang))))
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows="; ".join(alts)))


# --- exercise 3: a timeline, from the learner's iso_dates --------------------------------------------------------------

_ISO_TEXT = {
    "en": {"title": "The dates of Apollo program, sorted as text",
           "head": ["As written", "Your ISO dates"],
           "note": "The {n} dates of the page *Apollo program*, from the course's list. Gray: the dates as the page "
                   "writes them, sorted as text: by their first letter. Blue: your `iso_dates` of the same dates, "
                   "sorted as text, which is the order of time. Purple: where yours differs, with the right one dashed "
                   "beside it. Each column is a list of its own, so a row does not pair a date with its ISO form.",
           "stopped": "Your function stops with an error on the dates of the page, so the picture stays empty.",
           "slow": "Your function is still not done after {s} seconds, so the picture stays empty.",
           "odd": "Your function does not give a list of texts, one for each date, so the picture stays empty.",
           "alt": "{title}: {rows}"},
    "nl": {"title": "De datums van Apolloprogramma, gesorteerd als tekst",
           "head": ["Zoals geschreven", "Jouw ISO-datums"],
           "note": "De {n} datums van de pagina *Apolloprogramma*, uit de lijst van de course. Grijs: de datums zoals "
                   "op de pagina, gesorteerd als tekst: op het eerste cijfer. Blauw: je `iso_dates` van "
                   "dezelfde datums, gesorteerd als tekst, en dat is de volgorde in de tijd. Paars: waar jouw datum "
                   "afwijkt, met de goede gestippeld ernaast. Elke kolom is een eigen lijst: een rij zet een datum dus "
                   "niet naast dezelfde datum in ISO-vorm.",
           "stopped": "Je functie stopt met een fout op de datums van de pagina, dus de afbeelding blijft leeg.",
           "slow": "Je functie is na {s} seconden nog niet klaar, dus de afbeelding blijft leeg.",
           "odd": "Je functie geeft geen lijst van teksten, één voor elke datum, dus de afbeelding blijft leeg.",
           "alt": "{title}: {rows}"},
}


def iso_picture(seen) -> str:
    """The dates of the page Apollo program as written, sorted as text, beside the learner's ISO dates of them."""
    lang = seen.lang
    w = _ISO_TEXT[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    dates = page_dates(lang, PAGE_TIMELINE)
    rows_in = [r for _, r in dates]
    out = _call(lambda: fn(list(rows_in)))
    if out is _STOPPED or out is _SLOW:
        return _empty(w, "stopped" if out is _STOPPED else "slow")
    if not (isinstance(out, list) and len(out) == len(dates) and all(isinstance(x, str) for x in out)):
        return _empty(w, "odd")
    want = iso_dates(rows_in)
    written = sorted(t for t, _ in dates)
    rows_ = []
    for k, (a, mine, right) in enumerate(zip(written, out, want)):
        cell = _chip(mine, _CHIP_PRED) if mine == right else _chip(mine, _CHIP_PURPLE) + _chip(right, _CHIP_MISSING)
        rows_.append([_chip(a, _CHIP_DATA), cell])
    body = _table(rows_, w["head"]) + _note(w["note"].format(n=len(dates)))
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows=", ".join(out)))


# --- try it: a pattern of the learner's own on all paragraphs -----------------------------------------------------------

PATTERN_START = r"\b\d{1,2} [A-Za-z]+ \d{4}\b"   # day first with any word between: the box starts with it
PATTERN_LONGEST = 300
PATTERN_SECONDS = 2.0   # what a pattern gets for all the paragraphs
MOST_MATCHES = 100_000   # a pattern that matches more, such as . or \w+, gets a sentence instead of a picture
SHOWN_EXAMPLES = 6
_ESCAPED = re.compile(r"\\.|\[(?:\\.|[^\]\\])*\]")   # an escaped character, or a character class
_GROUP_START = re.compile(r"^\?(?::|=|!|<=|<!|P<\w+>|<\w+>)")


def repeats_a_repeat(pattern: str) -> bool:
    """True when the pattern repeats a group (with *, + or {...}) that itself holds a repetition or a choice with |,
    such as (a+)+ or (a|aa)*: the kind of pattern that can backtrack for a very long time (as in nlp-regex)."""
    plain = _ESCAPED.sub("x", pattern)
    starts = []
    for i, ch in enumerate(plain):
        if ch == "(":
            starts.append(i)
        elif ch == ")" and starts:
            start = starts.pop()
            after = plain[i + 1:i + 2]
            body = _GROUP_START.sub("", plain[start + 1:i])
            if after in ("*", "+", "{") and re.search(r"[*+?{|]", body):
                return True
    return False


def coverage(pattern: str, lang: str, most: int | None = None) -> dict:
    """What a pattern finds in every paragraph in `lang`: the matches (paragraph, start, end) of at least one
    character, the dates (paragraph, start) that one match covers whole, and the matches that cover no date. With
    `most`, it stops after that many matches, and "more" is True."""
    rx = re.compile(pattern)
    gold = collections.defaultdict(list)
    for d in xquad.dates(lang):
        gold[d.paragraph].append(d)
    matches, found, wrong = [], set(), []
    for i, p in enumerate(xquad.paragraphs(lang)):
        for m in rx.finditer(p.text):
            if m.end() == m.start():
                continue
            matches.append((i, m.start(), m.end()))
            if most is not None and len(matches) > most:
                return {"matches": matches, "found": found, "wrong": wrong, "more": True}
            covered = [d for d in gold[i] if m.start() <= d.start and d.end <= m.end()]
            found.update((i, d.start) for d in covered)
            if not covered:
                wrong.append((i, m.start(), m.end()))
    return {"matches": matches, "found": found, "wrong": wrong, "more": False}


_PATTERN_TEXT = {
    "en": {"title": "Your pattern on all 2,067 English paragraphs",
           "sum": "{m} matches. {f} of the {n} dates lie whole within a match; {w} matches hold no date.",
           "hits": "Two of the matches that hold a date",
           "wrong": "Matches that hold no date",
           "missed": "Dates that no match holds",
           "none": "none",
           "hidden": "{k} of them in paragraphs the course does not show",
           "empty": "Type a pattern in the box.",
           "long": "This pattern is longer than {k} characters; try a shorter one.",
           "error": "Python cannot read this pattern; it gets stuck at character {pos}. Is a parenthesis missing, or "
                    "is there a backslash in front of a letter that means nothing after it?",
           "nested": "This pattern repeats a group that repeats something itself, such as (a+)+. That can take a "
                     "very long time, so the box does not run it.",
           "slow": "This pattern took more than {k:g} seconds on all the paragraphs, so the box stopped it.",
           "many": "This pattern matches more than {k} pieces of the paragraphs, so the picture stays empty: a date is "
                   "only a few words. Can you make it match less?",
           "note": "Blue: a match that holds a date. Purple: a match that holds none. Purple, dashed: a date that no "
                   "match holds. A few of each, spread over the paragraphs, with the text around them.",
           "alt": "{sum}"},
    "nl": {"title": "Jouw patroon op alle 2.067 Nederlandse alinea's",
           "sum": "{m} matches. {f} van de {n} datums vallen helemaal binnen een match; {w} matches bevatten geen "
                  "datum.",
           "hits": "Twee van de matches met een datum",
           "wrong": "Matches zonder datum",
           "missed": "Datums die in geen enkele match vallen",
           "none": "geen",
           "hidden": "{k} ervan in alinea's die de course niet laat zien",
           "empty": "Typ een patroon in het vak.",
           "long": "Dit patroon is langer dan {k} tekens; probeer een korter patroon.",
           "error": "Python kan dit patroon niet lezen; het loopt vast bij teken {pos}. Mist er een haakje, of staat er "
                    "een backslash voor een letter die daarna geen betekenis heeft?",
           "nested": "Dit patroon herhaalt een groep die zelf iets herhaalt, zoals (a+)+. Dat kan heel lang duren, dus "
                     "het vak voert het niet uit.",
           "slow": "Dit patroon duurde langer dan {k:g} seconden voor alle alinea's, dus het vak stopte het.",
           "many": "Dit patroon matcht meer dan {k} stukken van de alinea's, dus de afbeelding blijft leeg: een datum "
                   "is maar een paar woorden. Kun je het patroon minder laten matchen?",
           "note": "Blauw: een match met een datum. Paars: een match zonder datum. Paars, gestippeld: een datum die in "
                   "geen enkele match valt. Van elk een paar, verspreid over de alinea's, met de tekst eromheen.",
           "alt": "{sum}"},
}

_AROUND = 45   # characters of text on each side of an example


def _spread(items: list, k: int) -> list:
    n = len(items)
    if n <= k:
        return items
    return [items[round(j * (n - 1) / (k - 1))] for j in range(k)]


def _context(lang: str, i: int, a: int, b: int, style: str) -> str:
    t = xquad.paragraphs(lang)[i].text
    left, right = max(0, a - _AROUND), min(len(t), b + _AROUND)
    return (f'<div style="{_QUOTE}"><span style="{_FAINT}">{html.escape(page(lang, i))}:</span> '
            f'{"…" if left else ""}{html.escape(t[left:a])}<span style="{style}">{html.escape(t[a:b])}</span>'
            f'{html.escape(t[b:right])}{"…" if right < len(t) else ""}</div>')


def pattern_picture(pattern, lang: str) -> str:
    w = _PATTERN_TEXT[lang]
    pattern = str(pattern or "")
    title = w["title"]

    def say(text_):
        return yours.figure(title, _note(text_), alt=text_)

    if not pattern:
        return say(w["empty"])
    if len(pattern) > PATTERN_LONGEST:
        return say(w["long"].format(k=PATTERN_LONGEST))
    try:
        re.compile(pattern)
    except re.error as e:
        return say(w["error"].format(pos=(e.pos or 0) + 1))
    if repeats_a_repeat(pattern):
        return say(w["nested"])
    try:
        got = with_time_limit(lambda: coverage(pattern, lang, MOST_MATCHES), PATTERN_SECONDS)
    except Timeout:
        return say(w["slow"].format(k=PATTERN_SECONDS))
    if got["more"]:
        return say(w["many"].format(k=_num(MOST_MATCHES, lang)))
    dates = xquad.dates(lang)
    summary = w["sum"].format(m=_num(len(got["matches"]), lang), f=_num(len(got["found"]), lang),
                              n=_num(len(dates), lang), w=_num(len(got["wrong"]), lang))

    def block(label, items, style):
        shown = [x for x in items if showable(x[0], lang)]
        hidden = len(items) - len(shown)
        lines = [_context(lang, i, a, b, style) for i, a, b in _spread(shown, SHOWN_EXAMPLES)]
        count = f"{_num(len(items), lang)}" + (f"; {w['hidden'].format(k=_num(hidden, lang))}" if hidden else "")
        inner = "".join(lines) if items else f'<div style="{_QUOTE}">{html.escape(w["none"])}</div>'
        return _label(f"{label} ({count})") + inner

    missed = [(d.paragraph, d.start, d.end) for d in dates if (d.paragraph, d.start) not in got["found"]]
    wrong = set(got["wrong"])
    hits = [x for x in got["matches"] if x not in wrong]
    body = (f'<div style="{_QUOTE}">{html.escape(summary)}</div>'
            + block(w["wrong"], got["wrong"], _WRONG_HIT) + block(w["missed"], missed, _MISSED)
            + (_label(w["hits"]) + "".join(_context(lang, i, a, b, _HIT) for i, a, b in
                                         _spread([x for x in hits if showable(x[0], lang)], 2)) if hits else "")
            + _note(w["note"]))
    return yours.figure(title, body, alt=w["alt"].format(sum=summary))


# --- names of people: a pattern of capitals against Universal NER ------------------------------------------------------

_CAPITAL = re.compile(r"^[A-Z][a-z]+$")


@functools.cache
def name_runs() -> dict:
    """Runs of two or more words that each start with a capital and go on in small letters, against the names of
    people in the 11,722 English sentences of Universal NER: how many runs, how many of them are a name of a person,
    and how many names there are; with a few runs that are not names, and names of one word, from sentences that can be
    shown."""
    runs = right = names_ = 0
    not_names, single = [], []
    for s in uner.sentences():
        words = [w for w, _ in s.pairs]
        gold = {(a, b) for a, b, k in uner.spans([t for _, t in s.pairs]) if k == "PER"}
        found = set()
        i = 0
        while i < len(words):
            j = i
            while j < len(words) and _CAPITAL.match(words[j]):
                j += 1
            if j - i >= 2:
                found.add((i, j))
            i = max(j, i + 1)
        runs += len(found)
        right += len(found & gold)
        names_ += len(gold)
        if s.shown:
            not_names += [" ".join(words[a:b]) for a, b in sorted(found - gold)]
            single += [words[a] for a, b in sorted(gold - found) if b - a == 1 and _CAPITAL.match(words[a])]
    return {"sentences": len(uner.sentences()), "runs": runs, "right": right, "names": names_,
            "not_names": not_names, "single": single}


# --- the numbers of the text ----------------------------------------------------------------------------------------------

def numbers(lang: str) -> dict:
    """The numbers the lesson's text names for the paragraphs in `lang`."""
    dates = xquad.dates(lang)
    forms = collections.Counter(d.form for d in dates)
    gold = gold_rows(lang)
    pages_ = collections.defaultdict(set)
    for d in dates:
        pages_[xquad.paragraphs(lang)[d.paragraph].article].add("mdy" if d.form == "mdy" else "dmy")
    return {"paragraphs": len(xquad.paragraphs(lang)), "pages": len(xquad.titles(lang)), "dates": len(dates),
            "dmy": forms["dmy"], "mdy": forms["mdy"], "other": forms["other"],
            "pages_with_dates": len(pages_), "one_form": sum(1 for v in pages_.values() if len(v) == 1),
            "rules": {rule: counts(rule_rows(lang, rule), gold) for rule in RULES}}


# --- the exercises ---------------------------------------------------------------------------------------------------------

_DOTS = Mistake("dots-left", matches=lambda got, text_: re.search(r"^[^#\n]*\.\.\.", text_, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang de `...` door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _show(v):
    """A value for a message, the way code writes it: a long list cut short after three items."""
    if isinstance(v, bool) or v is None:
        return repr(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple)):
        items = [_show(x) for x in v[:3]] + (["…"] if len(v) > 3 else [])
        inner = ", ".join(items)
        return f"[{inner}]" if isinstance(v, list) else f"({inner}{',' if len(v) == 1 else ''})"
    return repr(v)


def _not_raw(got, text_) -> bool:
    """No rows at all, and the code has a string without r in front in which Python turned \\b into a backspace."""
    if got != [] or not isinstance(text_, str):
        return False
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text_).readline))
    except (tokenize.TokenError, SyntaxError):
        return False
    for t in tokens:
        if t.type == tokenize.STRING and "r" not in t.string.split(t.string[-1], 1)[0].lower():
            with contextlib.suppress(ValueError, SyntaxError), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if "\b" in ast.literal_eval(t.string):
                    return True
    return False


def _not_list(got, text_) -> bool:
    return not isinstance(got, list)


def _list_rows(got, text_) -> bool:
    """Rows as lists, not tuples."""
    return isinstance(got, list) and bool(got) and all(isinstance(r, list) for r in got)


def _not_three(got, text_) -> bool:
    return not (isinstance(got, (tuple, list)) and len(got) == 3)


def _not_texts(got, text_) -> bool:
    return isinstance(got, list) and bool(got) and not all(isinstance(x, str) for x in got)


REQUIRED = ["date_rows", "score", "iso_dates", "names"]

EXERCISES = {
    "date_rows": FunctionBox(
        name="date_rows",
        picture=rows_picture,
        function="date_rows",
        start=_DR_START,
        solution=_DR,
        show=_show,
        visible=[(_SE, _ME), (_SN, _MN)],
        hidden=[("On 1 May 1191, Richard's fleet arrived in the port of Limassol on Cyprus.", _ME),
                ("Warsaw was occupied by Germany from 4 August 1915 until November 1918.", _ME),
                ("The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa "
                 "Clara, California.", _ME),
                ("TVOntario picked up the show in 1976 beginning with The Three Doctors and aired each series "
                 "(several years late) through to series 24 in 1991.", _ME),
                ("Koningin Elizabeth II opende het nieuwe gebouw op 9 oktober 2004.", _MN),
                ("", _ME),
                ("On 3 Feb 2020 and on 4 March 2020.", {"Jan": 1, "Feb": 2, "Mar": 3}),
                (_paragraph_value("en", 371), _ME), (_paragraph_value("nl", 371), _MN),
                (_paragraph_value("en", 6), _ME), (_paragraph_value("nl", 6), _MN)],
        first=[_DOTS],
        mistakes=[
            Mistake("not-list", matches=_not_list,
                    en="`{call}` gives {got}, and it should give a list with a row for each date. Where does your `return` stand, and what does it give back?",
                    nl="`{call}` geeft {got}, en het moet een lijst zijn met een rij voor elke datum. Waar staat je `return`, en wat geeft je functie terug?"),
            Mistake("not-raw", matches=_not_raw,
                    en="`{call}` gives `[]`: your pattern finds no date at all. What does Python make of `\\b` in a string without `r` in front?",
                    nl="`{call}` geeft `[]`: je patroon vindt helemaal geen datum. Wat maakt Python van `\\b` in een string zonder `r` ervoor?"),
            Mistake("groups", error=lambda e: isinstance(e, ValueError) and "unpack" in str(e),
                    en="`{call}` stops with a ValueError, *values to unpack*: the loop takes three values, `day`, `month` and `year`, from each match that `re.findall` gives, and the matches have another number of values. Each group of the pattern gives one value. How many groups does your pattern have, and does the choice between the month names stay inside one of them?",
                    nl="`{call}` stopt met een ValueError, *values to unpack*: de loop neemt drie waarden, `day`, `month` en `year`, uit elke match die `re.findall` geeft, en de matches hebben een ander aantal waarden. Elke groep van het patroon geeft één waarde. Hoeveel groepen heeft je patroon, en blijft de keuze tussen de namen van de maanden binnen één groep?"),
            Mistake("not-a-month", error=lambda e: isinstance(e, KeyError),
                    en="`{call}` stops with a KeyError: a word that your pattern takes as the month is not a key of `months`. Which words can the group of the month match in your pattern?",
                    nl="`{call}` stopt met een KeyError: een woord dat je patroon als de maand neemt, is geen key van `months`. Welke woorden kan de groep van de maand matchen in je patroon?"),
            Mistake("strings", wrong=_dr(add="(year, months[month], day)"),
                    en="`{call}` gives {got}, and it should give {want}: the year and the day are still the strings that `re.findall` gave. What turns a string of digits into a whole number?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het jaar en de dag zijn nog de strings die `re.findall` gaf. Wat maakt van een string met cijfers een geheel getal?"),
            Mistake("day-first", wrong=_dr(add="(int(day), months[month], int(year))"),
                    en="`{call}` gives {got}, and it should give {want}: the right numbers, with the day first, as the text writes it. In which order do the fields of a row stand?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de goede getallen, met de dag eerst, zoals de tekst het schrijft. In welke volgorde staan de velden van een rij?"),
            Mistake("month-name", wrong=_dr(add="(int(year), month, int(day))"),
                    en="`{call}` gives {got}, and it should give {want}: the month is still its name. Where do you find the number of a month?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de maand is nog de naam. Waar vind je het nummer van een maand?"),
            Mistake("two-digits", wrong=_dr(pattern=r'r"\b(\d{2}) (" + names + r") (\d{4})\b"'),
                    en="`{call}` gives {got}, and it should give {want}: a day of one digit, such as the 6 of *6 December 1989*, is missing. How many digits can a day have?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een dag van één cijfer, zoals de 6 van *6 december 1989*, ontbreekt. Hoeveel cijfers kan een dag hebben?"),
            Mistake("lists", matches=_list_rows,
                    en="`{call}` gives {got}, and it should give {want}: each row is a list, and it should be a tuple. How do you write a tuple of three values?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke rij is een lijst, en het moet een tuple zijn. Hoe schrijf je een tuple van drie waarden?"),
            Mistake("return-in-loop", wrong=_dr(end="        return rows\n    return rows\n"),
                    en="`{call}` gives {got}, and it should give {want}: only the first date. Your `return` stands inside the loop, so the function stops after one date. Where does `return rows` belong?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: alleen de eerste datum. Je `return` staat binnen de loop, dus de functie stopt na één datum. Waar hoort `return rows`?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: `pattern`, with a group for the day, one for the month and one for the year, and the line in the loop that adds the row `(year, month, day)` to `rows`, each as a whole number.",
                      "nl": "Vervang de twee keer `...`: `pattern`, met een groep voor de dag, een voor de maand en een voor het jaar, en de regel in de loop die de rij `(year, month, day)` toevoegt aan `rows`, elk als geheel getal."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which dates of the text are written as a day, a month name and a year, in that order?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Welke datums in de tekst zijn geschreven als eerst een dag, dan de naam van een maand en dan een jaar?"},
            "almost": {"en": "Right for `date_rows(sentence_en, months_en)` and `date_rows(sentence_nl, months_nl)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `months` it gets, and give the dates in the order in which the text names them?",
                       "nl": "Goed voor `date_rows(sentence_en, months_en)` en `date_rows(sentence_nl, months_nl)`. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je functie de `months` uit de aanroep, en geeft de functie de datums in de volgorde van de tekst?"},
            "correct": {"en": "Your `date_rows` is right in all {n} cases.",
                        "nl": "Je `date_rows` klopt in alle {n} gevallen."},
        },
    ),
    "score": FunctionBox(
        name="score",
        picture=score_picture,
        function="score",
        start=_SC_START,
        solution=_SC,
        show=_show,
        visible=[(_FR, _GR)],
        hidden=[([], _GR), (_FR, []), ([], []), (_GR, _GR),
                ([(1, 2, 3), (4, 5, 6)], [(4, 5, 6), (7, 8, 9), (1, 2, 4)]),
                (_ROWS["en", "day-first"], _GOLD["en"]), (_ROWS["nl", "day-first"], _GOLD["nl"]),
                (_ROWS["en", "any-word"], _GOLD["en"])],
        first=[_DOTS],
        mistakes=[
            Mistake("not-three", matches=_not_three,
                    en="`{call}` gives {got}, and it should give three numbers: right, wrong and missed. Where does your `return` stand, and what does it give back?",
                    nl="`{call}` geeft {got}, en het moet drie getallen zijn: goed, fout en gemist. Waar staat je `return`, en wat geeft je functie terug?"),
            Mistake("swapped", wrong=_sc(end="    return right, len(gold) - right, len(found) - right\n"),
                    en="`{call}` gives {got}, and it should give {want}: the wrong and the missed rows have changed places. A wrong row is one your rule found. Out of which list does it come?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de foute en de gemiste rijen staan op elkaars plaats. Een foute rij is een rij die je regel vond. Uit welke lijst komt zo'n rij?"),
            Mistake("totals", wrong=_sc(end="    return right, len(found), len(gold)\n"),
                    en="`{call}` gives {got}, and it should give {want}: every row that was found counts as wrong, also a right one, and every right row as missed, also one that was found. Which rows of `found` are wrong?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke gevonden rij telt als fout, ook een goede, en elke goede rij als gemist, ook een gevonden rij. Welke rijen uit `found` zijn fout?"),
            Mistake("inverted", wrong=_sc(test="row not in gold"),
                    en="`{call}` gives {got}, and it should give {want}: `right` counts the rows that are not in `gold`. When is a row that was found right?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `right` telt de rijen die niet in `gold` staan. Wanneer is een gevonden rij goed?"),
            Mistake("list", wrong=_sc(end="    return [right, len(found) - right, len(gold) - right]\n"),
                    en="`{call}` gives {got}, and it should give {want}: the right numbers in a list, and it should be a tuple. How do you give back three values as a tuple?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de goede getallen in een lijst, en het moet een tuple zijn. Hoe geef je drie waarden terug als tuple?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: in the loop, the lines that add 1 to `right` for a row of `found` that is also in `gold`, and after it, the three numbers right, wrong and missed.",
                      "nl": "Vervang de twee keer `...`: in de loop de regels die 1 optellen bij `right` voor een rij uit `found` die ook in `gold` staat, en daarna de drie getallen goed, fout en gemist."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. How many rows of `found` are in `gold`, how many are not, and how many rows of `gold` are not in `found`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Hoeveel rijen uit `found` staan in `gold`, hoeveel niet, en hoeveel rijen uit `gold` staan niet in `found`?"},
            "almost": {"en": "Right for `score(found_rows, gold_rows)`. But `{call}` gives {got}, and it should give {want}. Does your function count with the lists it gets, also an empty one?",
                       "nl": "Goed voor `score(found_rows, gold_rows)`. Maar `{call}` geeft {got}, en het moet {want} zijn. Telt je functie met de lijsten uit de aanroep, ook met een lege lijst?"},
            "correct": {"en": "Your `score` is right in all {n} cases.",
                        "nl": "Je `score` klopt in alle {n} gevallen."},
        },
    ),
    "iso_dates": FunctionBox(
        name="iso_dates",
        picture=iso_picture,
        function="iso_dates",
        start=_ISO_START,
        solution=_ISO,
        show=_show,
        visible=[(_AR,)],
        hidden=[([],), ([(2016, 2, 7)],), ([(2013, 5, 21), (2013, 5, 3), (2012, 10, 16)],), (_TIMELINE,), (_ALL,)],
        first=[_DOTS],
        mistakes=[
            Mistake("not-list", matches=_not_list,
                    en="`{call}` gives {got}, and it should give a list with a text for each row. Where does your `return` stand, and what does it give back?",
                    nl="`{call}` geeft {got}, en het moet een lijst zijn met een tekst voor elke rij. Waar staat je `return`, en wat geeft je functie terug?"),
            Mistake("not-texts", matches=_not_texts,
                    en="`{call}` gives {got}, and it should give {want}: each date should be a text, such as `\"1969-07-20\"`. Which kind of string puts the values of `year`, `month` and `day` in a text?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke datum hoort een tekst te zijn, zoals `\"1969-07-20\"`. Welk soort string zet de waarden van `year`, `month` en `day` in een tekst?"),
            Mistake("no-zeros", wrong=_iso(add='f"{year}-{month}-{day}"'),
                    en="`{call}` gives {got}, and it should give {want}: a month or a day of one digit has no 0 in front, so the texts do not sort in order of time. How many digits does ISO 8601 give a month and a day?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een maand of dag van één cijfer heeft geen 0 ervoor, dus de teksten staan na het sorteren niet op volgorde van de tijd. Hoeveel cijfers geeft ISO 8601 een maand en een dag?"),
            Mistake("day-first", wrong=_iso(add='f"{day:02d}-{month:02d}-{year}"'),
                    en="`{call}` gives {got}, and it should give {want}: the day comes first, as many countries write it. In which order does ISO 8601 write the year, the month and the day?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de dag staat vooraan, zoals in een Nederlandse brief. In welke volgorde schrijft ISO 8601 het jaar, de maand en de dag?"),
            Mistake("month-day", wrong=_iso(add='f"{year}-{day:02d}-{month:02d}"'),
                    en="`{call}` gives {got}, and it should give {want}: the day and the month have changed places. Which of the two comes right after the year?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de dag en de maand staan op elkaars plaats. Welke van de twee komt direct na het jaar?"),
            Mistake("unsorted", wrong=_iso(end="    return dates\n"),
                    en="`{call}` gives {got}, and it should give {want}: the right texts, in the order of the rows. In which order should they stand?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de goede teksten, in de volgorde van de rijen. In welke volgorde horen de teksten te staan?"),
            Mistake("return-in-loop", wrong=_iso(end="        return sorted(dates)\n    return sorted(dates)\n"),
                    en="`{call}` gives {got}, and it should give {want}: only the first date. Your `return` stands inside the loop, so the function stops after one row. Where does the `return` belong?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: alleen de eerste datum. Je `return` staat binnen de loop, dus de functie stopt na één rij. Waar hoort de `return`?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: in the loop, the line that adds the date of the row to `dates` as a text in the form YYYY-MM-DD, and after it, the texts in order.",
                      "nl": "Vervang de twee keer `...`: in de loop de regel die de datum van de rij als tekst in de vorm YYYY-MM-DD toevoegt aan `dates`, en daarna de teksten op volgorde."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. How does ISO 8601 write a date, and in which order do the texts stand?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Hoe schrijft ISO 8601 een datum, en in welke volgorde staan de teksten?"},
            "almost": {"en": "Right for `iso_dates(apollo_rows)`. But `{call}` gives {got}, and it should give {want}. Does your function work for any list of rows, also an empty one?",
                       "nl": "Goed voor `iso_dates(apollo_rows)`. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie voor elke lijst rijen, ook een lege?"},
            "correct": {"en": "Your `iso_dates` is right in all {n} cases.",
                        "nl": "Je `iso_dates` klopt in alle {n} gevallen."},
        },
    ),
}

CASES = [
    ("date_rows", _DR_START, "empty", None),
    ("date_rows", _DR, "correct", None),
    ("date_rows", _dr(pattern="..."), "wrong", "dots-left"),
    ("date_rows", _DR_HEAD + "    return 3\n", "wrong", "not-list"),
    ("date_rows", _dr(pattern=r'r"\b(\d{1,2}) " + names + r" (\d{4})\b"'), "error", "groups"),
    ("date_rows", _dr(pattern=r'r"\b(\d{1,2}) ([A-Za-z]+) (\d{4})\b"'), "error", "not-a-month"),
    ("date_rows", _dr(add="(year, months[month], day)"), "wrong", "strings"),
    ("date_rows", _dr(pattern='"\\b(\\\\d{1,2}) (" + names + ") (\\\\d{4})\\b"'), "wrong", "not-raw"),
    ("date_rows", _dr(add="(int(day), months[month], int(year))"), "wrong", "day-first"),
    ("date_rows", _dr(add="(int(year), month, int(day))"), "wrong", "month-name"),
    ("date_rows", _dr(pattern=r'r"\b(\d{2}) (" + names + r") (\d{4})\b"'), "wrong", "two-digits"),
    ("date_rows", _dr(add="[int(year), months[month], int(day)]"), "wrong", "lists"),
    ("date_rows", _dr(end="        return rows\n    return rows\n"), "wrong", "return-in-loop"),
    ("date_rows", _dr(add="(int(year), months[month], 1)"), "wrong", None),
    ("date_rows", _dr(pattern=r'r"(\d{1,2}) (" + names + r") (\d{4})"'), "correct", None),
    ("date_rows", _dr(pattern=r'r"\b(\d{1,2}) (January|February|March|April|May|June|July|August|September|October|November|December) (\d{4})\b"'),
     "wrong", None),
    ("date_rows", 'import re\n\n\ndef date_rows(text, months):\n    rows = []\n    for name in months:\n'
     '        for day, year in re.findall(r"\\b(\\d{1,2}) " + name + r" (\\d{4})\\b", text):\n'
     '            rows.append((int(year), months[name], int(day)))\n    return rows\n', "almost", None),
    ("score", _SC_START, "empty", None),
    ("score", _SC, "correct", None),
    ("score", _sc(test="..."), "wrong", "dots-left"),
    ("score", _SC_HEAD + "        right = right + 1\n    return right\n", "wrong", "not-three"),
    ("score", _sc(end="    return right, len(gold) - right, len(found) - right\n"), "wrong", "swapped"),
    ("score", _sc(end="    return right, len(found), len(gold)\n"), "wrong", "totals"),
    ("score", _sc(test="row not in gold"), "wrong", "inverted"),
    ("score", _sc(end="    return [right, len(found) - right, len(gold) - right]\n"), "wrong", "list"),
    ("score", _sc(end="    return right, len(found) - right, 2\n"), "almost", None),
    ("score", _sc(end="    return right, 0, len(gold) - right\n"), "wrong", None),
    ("score", "def score(found, gold):\n    right = len(set(found) & set(gold))\n"
     "    return right, len(set(found)) - right, len(set(gold)) - right\n", "correct", None),
    ("iso_dates", _ISO_START, "empty", None),
    ("iso_dates", _ISO, "correct", None),
    ("iso_dates", _iso(add="..."), "wrong", "dots-left"),
    ("iso_dates", _ISO_HEAD + "        dates.append(year)\n    return len(dates)\n", "wrong", "not-list"),
    ("iso_dates", _iso(add="(year, month, day)"), "wrong", "not-texts"),
    ("iso_dates", _iso(add='f"{year}-{month}-{day}"'), "wrong", "no-zeros"),
    ("iso_dates", _iso(add='f"{day:02d}-{month:02d}-{year}"'), "wrong", "day-first"),
    ("iso_dates", _iso(add='f"{year}-{day:02d}-{month:02d}"'), "wrong", "month-day"),
    ("iso_dates", _iso(end="    return dates\n"), "wrong", "unsorted"),
    ("iso_dates", _iso(end="        return sorted(dates)\n    return sorted(dates)\n"), "wrong", "return-in-loop"),
    ("iso_dates", _iso(add='f"{year}/{month:02d}/{day:02d}"'), "wrong", None),
    ("iso_dates", _iso(add='f"{year}-{month:02d}-{day:02d}" if year > 1900 else "1900-01-01"'), "almost", None),
    ("iso_dates", "def iso_dates(rows):\n    return [f\"{y:04d}-{m:02d}-{d:02d}\" for y, m, d in sorted(rows)]\n",
     "correct", None),
]
