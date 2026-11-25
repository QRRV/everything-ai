# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-what-is-nlg: a weather report written from a week of numbers in the three steps of classic
NLG. Content selection keeps the measurements of a day that reach their limit, as messages (select); microplanning
turns the messages of a day into words and gives days in a row with the same words one sentence (aggregate); surface
realization fills the template of the language, with commas between the days and *and* before the last (join_names).

The three weeks are the course's own, made up in the range of Dutch weather in April, July and October. The words, the
templates and the limits are the course's too. Every number in the text and the pictures is computed by the course
(tests/test_nlp_what_is_nlg.py computes them again)."""
import ast
import html
import re

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, Sort

LANGS = ("en", "nl")
MEASURES = ("high", "low", "rain", "wind", "sun")   # the columns of a day: °C, °C, mm, Beaufort, hours
DAYS = {"en": ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"),
        "nl": ("maandag", "dinsdag", "woensdag", "donderdag", "vrijdag", "zaterdag", "zondag")}

# What the report says, in this order, and from which value on.
LIMITS = {"high": 25, "sun": 10, "rain": 1, "wind": 6}


def _day(high, low, rain, wind, sun) -> dict:
    return {"high": high, "low": low, "rain": rain, "wind": wind, "sun": sun}


WEEKS = {
    "april": (_day(12, 4, 0.0, 3, 8), _day(14, 5, 0.0, 2, 11), _day(11, 7, 6.5, 6, 1), _day(10, 6, 3.2, 7, 2),
              _day(13, 3, 0.4, 4, 6), _day(15, 4, 0.0, 3, 10), _day(16, 6, 0.0, 3, 11)),
    "july": (_day(24, 14, 0.0, 3, 9), _day(27, 16, 0.0, 2, 13), _day(29, 18, 0.0, 3, 12), _day(31, 19, 12.0, 5, 7),
             _day(22, 15, 2.5, 6, 4), _day(21, 13, 0.6, 4, 8), _day(23, 12, 0.0, 3, 10)),
    "october": (_day(14, 9, 4.0, 5, 2), _day(13, 10, 9.5, 7, 0), _day(12, 8, 15.0, 8, 1), _day(11, 6, 1.0, 6, 3),
                _day(12, 5, 0.0, 4, 5), _day(13, 7, 0.3, 3, 4), _day(12, 8, 2.2, 5, 1)),
}
WEEK_NAMES = {"en": {"april": "A week in April", "july": "A week in July", "october": "A week in October"},
              "nl": {"april": "Een week in april", "july": "Een week in juli", "october": "Een week in oktober"}}
WORKED = ("april", 2)    # the worked example: Wednesday of the week in April

# The words of each kind, the word for a day without messages, the word for *and*, and the template of a sentence.
WORDS = {"en": {"high": "warm", "sun": "sunny", "rain": "rainy", "wind": "windy"},
         "nl": {"high": "warm", "sun": "zonnig", "rain": "regenachtig", "wind": "winderig"}}
DRY = {"en": "dry", "nl": "droog"}
AND = {"en": "and", "nl": "en"}
TEMPLATE = {"en": "On {days}, it will be {words}.", "nl": "Op {days} wordt het {words}."}


# --- the three steps, as the course writes them ----------------------------------------------------------------------------

def select(day, limits):
    """Content selection: a message (kind, value) for each kind of `limits`, in its order, whose value reaches the limit."""
    messages = []
    for kind in limits:
        if day[kind] >= limits[kind]:
            messages.append((kind, day[kind]))
    return messages


def aggregate(phrases):
    """Aggregation: days in a row with the same words share a group (days, words)."""
    groups = []
    for i in range(len(phrases)):
        if len(groups) > 0 and groups[-1][1] == phrases[i]:
            groups[-1][0].append(i)
        else:
            groups.append(([i], phrases[i]))
    return groups


def join_names(names, and_word):
    """Names in a row: a comma between them, and `and_word` between the last two."""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " " + and_word + " " + names[-1]


def phrase(messages, lang: str, join=join_names) -> str:
    """Lexical choice: the words of a day's messages, joined; a day without messages is dry."""
    lang = lang if lang in LANGS else "en"
    words_ = [WORDS[lang][kind] for kind, _ in messages]
    return join(words_, AND[lang]) if words_ else DRY[lang]


def sentence(days, words_: str, lang: str, join=join_names) -> str:
    """Surface realization: the template of the language with the names of the days and the words."""
    lang = lang if lang in LANGS else "en"
    return TEMPLATE[lang].format(days=join([DAYS[lang][d] for d in days], AND[lang]), words=words_)


def steps(week: str, lang: str, limits=None, select_=select, aggregate_=aggregate, join=join_names) -> tuple:
    """Every step of the report of a week: the messages and the words of each day, the groups, and the sentences."""
    limits = dict(LIMITS if limits is None else limits)
    messages = [select_(dict(d), dict(limits)) for d in WEEKS[week]]
    phrases = [phrase(m, lang, join) for m in messages]
    groups = aggregate_(list(phrases))
    sentences = [sentence(days, p, lang, join) for days, p in groups]
    return messages, phrases, groups, sentences


def report(week: str, lang: str, limits=None) -> list:
    """The sentences of the report of a week, with the course's three steps."""
    return steps(week, lang, limits)[3]


def numbers() -> dict:
    """The numbers the text states, worked out from the weeks."""
    n = {"weeks": len(WEEKS), "days": 7 * len(WEEKS), "measures": len(MEASURES), "week_numbers": 7 * len(MEASURES)}
    n["numbers"] = n["days"] * len(MEASURES)
    messages = {w: steps(w, "en")[0] for w in WEEKS}
    n["messages"] = sum(len(m) for ms in messages.values() for m in ms)
    for w in WEEKS:
        n["messages_" + w] = sum(len(m) for m in messages[w])
        n["sentences_" + w] = len(report(w, "en"))
    for kind in LIMITS:
        n["kept_" + kind] = sum(1 for ms in messages.values() for m in ms for k, _ in m if k == kind)
    n["sentences"] = sum(n["sentences_" + w] for w in WEEKS)
    n["dry_days"] = sum(1 for ms in messages.values() for m in ms if not m)
    n["rain_left"] = sum(1 for d in _all_days() if 0 < d["rain"] < LIMITS["rain"])
    return n


def _all_days() -> list:
    return [d for w in WEEKS for d in WEEKS[w]]


def false_dry(week: str, phrases: list, lang: str) -> list:
    """The days that the words call dry while 1 mm of rain or more falls: what the course's limit counts as rain."""
    lang = lang if lang in LANGS else "en"
    return [i for i, (d, p) in enumerate(zip(WEEKS[week], phrases)) if p == DRY[lang] and d["rain"] >= LIMITS["rain"]]


# --- styles and small parts of the pictures ------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);margin-top:0.45rem"
_CELL = ("padding:0.12rem 0.4rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.84rem")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap"
_DATA = "color:var(--ea-data-text)"
_FAINT = "color:var(--ea-ink-2)"
_PRED = "color:var(--ea-pred-text);font-weight:600"
_LOSS = "color:var(--ea-loss-text);font-weight:600"
_CHIP = ("display:inline-block;margin:0.08rem 0.2rem 0.08rem 0;padding:0 0.3rem;border-radius:3px;"
         "font-family:var(--ea-font-code);font-size:0.78rem;line-height:1.5;white-space:nowrap")
_CHIP_PRED = _CHIP + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
_CHIP_DATA = _CHIP + ";background:var(--ea-data-wash);color:var(--ea-data-text);font-weight:600"
_CHIP_LOSS = _CHIP + ";background:var(--ea-loss-wash);color:var(--ea-loss-text);font-weight:600"
_TEXT_CAP = 120   # characters of a learner's value that a picture shows


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


def _colors(w: dict, body: str) -> str:
    """The note on the colors under a picture: gray always, blue and purple only when the picture draws them."""
    parts = [w["gray"]]
    if "var(--ea-pred-text)" in body:
        parts.append(w["blue"])
    if "var(--ea-loss-text)" in body:
        parts.append(w["purple"])
    return _note(" ".join(parts))


def _label(text_: str) -> str:
    return f'<div style="{_LABEL}">{_marked(text_)}</div>'


def _num(n, lang: str) -> str:
    """A whole number with a separator for thousands: 1,234 in English, 1.234 in Dutch."""
    out = f"{round(n):,}"
    return out.replace(",", ".") if lang == "nl" else out


def _value_text(v, lang: str) -> str:
    """A measured value as the table writes it: a whole number as it is, and rain with one decimal (a comma in Dutch).
    Anything else a learner's function gives, such as a text, is written as it is, cut to _TEXT_CAP characters."""
    if isinstance(v, float):
        out = f"{v:.1f}"
    elif isinstance(v, int) and not isinstance(v, bool):
        out = _short(v)
    else:
        return _short(v)
    out = out if len(out) <= _TEXT_CAP else out[:_TEXT_CAP] + "…"
    return out.replace(".", ",") if lang == "nl" else out


def _chip(text_: str, style: str) -> str:
    return f'<span style="{style}">{html.escape(text_)}</span>'


def _short(v) -> str:
    """A learner's value as text, cut to _TEXT_CAP characters."""
    try:
        out = v if isinstance(v, str) else repr(v)
    except Exception:   # noqa: BLE001   a value that cannot even be written
        out = "?"
    return out if len(out) <= _TEXT_CAP else out[:_TEXT_CAP] + "…"


def _message_chips(messages, style: str, lang: str) -> str:
    if not messages:
        return f'<span style="{_FAINT}">–</span>'
    return "".join(_chip(f"{kind} {_value_text(value, lang)}", style) for kind, value in messages)


def _cell(text_: str, style: str = "", extra: str = "", rows: int = 1) -> str:
    span = f' rowspan="{rows}"' if rows > 1 else ""
    return f'<td style="{_CELL}{extra};{style}"{span}>{text_}</td>'


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error
_SLOW = object()      # what _call gives for a function that ran out of time
PICTURE_SECONDS = 2   # what the learner's function gets for all the calls of one picture together


def _call(make, seconds: float = PICTURE_SECONDS):
    """make() within a time limit, with what it prints thrown away; _SLOW when the time runs out, _STOPPED when it
    stops with an error."""
    import contextlib
    import io
    from everything_ai.check import Timeout, with_time_limit
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(make, max(seconds, 0.05))
    except Timeout:
        return _SLOW
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _empty(w: dict, key: str) -> str:
    """The picture that stays empty, with why: the learner's function stopped, was too slow, or gave something odd."""
    text_ = w[key].format(s=PICTURE_SECONDS, kind=w.get("kind", ""))
    return yours.figure(w["title"], _note(text_), alt=text_)


_WHY_EMPTY = {
    "en": {"stopped": "Your function stops with an error, so there is nothing to draw yet. The check above says why.",
           "slow": "Your function took longer than {s} seconds, so there is nothing to draw yet.",
           "odd": "Your function gives something other than {kind}, so there is nothing to draw yet. The check above says what."},
    "nl": {"stopped": "Je functie stopt met een fout, dus er valt nog niets te tekenen. De check hierboven zegt waarom.",
           "slow": "Je functie deed er langer over dan {s} seconden, dus er valt nog niets te tekenen.",
           "odd": "Je functie geeft iets anders dan {kind}, dus er valt nog niets te tekenen. De check hierboven zegt wat."},
}

_TABLE_TEXT = {
    "en": {"head": ["Day", "High", "Low", "Rain", "Wind", "Sun", "Content selection", "Microplanning", "Surface realization"],
           "units": "High and low in °C, rain in mm, wind in Beaufort, sun in hours.",
           "report": "The report:"},
    "nl": {"head": ["Dag", "Max", "Min", "Regen", "Wind", "Zon", "Content selection", "Microplanning", "Surface realization"],
           "units": "Max en min in °C, regen in mm, wind in Beaufort, zon in uren.",
           "report": "Het weerbericht:"},
}


def _is_messages(v) -> bool:
    return isinstance(v, list) and all(isinstance(m, tuple) and len(m) == 2 and m[0] in LIMITS for m in v)


def _is_groups(v) -> bool:
    return (isinstance(v, list) and all(isinstance(g, tuple) and len(g) == 2 and isinstance(g[0], list)
                                        and all(isinstance(d, int) and not isinstance(d, bool) and 0 <= d < 7 for d in g[0])
                                        and g[0] and isinstance(g[1], str) for g in v))


def _in_order(groups) -> bool:
    """The groups take the seven days once each, in order: the sentences can stand beside the days they are about."""
    return [d for days, _ in groups for d in days] == list(range(7))


def _week_table(week: str, lang: str, got: tuple, want: tuple | None = None, wrong_days=()) -> str:
    """The table of a week: the numbers, the messages and the words of each day, and each sentence beside its days.
    `got` are the steps to draw; where they differ from `want`, they are purple. Days in `wrong_days` are purple too."""
    w = _TABLE_TEXT[lang]
    messages, phrases, groups, sentences = got
    want = want or got
    selected = [{k for k, _ in m} for m in messages]
    top = "".join(f'<th style="{_HEAD}{_NUMBER if 1 <= j <= 5 else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(w["head"]))
    beside = _in_order(groups)
    starts = {days[0]: (k, len(days)) for k, (days, _) in enumerate(groups)} if beside else {}
    rows = []
    for i, day in enumerate(WEEKS[week]):
        cells = [_cell(html.escape(DAYS[lang][i]), _FAINT)]
        for kind in MEASURES:
            text_ = html.escape(_value_text(day[kind], lang))
            cells.append(_cell(_chip(_value_text(day[kind], lang), _CHIP_DATA) if kind in selected[i] else text_,
                               _DATA, _NUMBER))
        mine_m = messages[i] == want[0][i]
        cells.append(_cell(_message_chips(messages[i], _CHIP_PRED if mine_m else _CHIP_LOSS, lang)))
        good = phrases[i] == want[1][i] and i not in wrong_days
        cells.append(_cell(_chip(_short(phrases[i]), _CHIP_PRED if good else _CHIP_LOSS)))
        if i in starts:
            k, n = starts[i]
            same = (k < len(want[3]) and sentences[k] == want[3][k] and groups[k] == want[2][k]
                    and not set(groups[k][0]) & set(wrong_days))
            cells.append(_cell(html.escape(_short(sentences[k])), _PRED if same else _LOSS, rows=n))
        rows.append("<tr>" + "".join(cells) + "</tr>")
    table = (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;margin:0.3rem 0 0.2rem">'
             f"<thead><tr>{top}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")
    out = table + _note(w["units"])
    if not beside:   # the groups skip or repeat days: the sentences come under the table
        lines = []
        for k, s in enumerate(sentences):
            same = k < len(want[3]) and s == want[3][k] and groups[k] == want[2][k]
            lines.append(f'<div style="{_PRED if same else _LOSS}">{html.escape(_short(s))}</div>')
        out += _label(w["report"]) + "".join(lines)
    return out


# --- the idea: the three steps on the week in April ------------------------------------------------------------------------

_IDEA_TEXT = {
    "en": {"title": "From numbers to a report, in three steps",
           "gray": "Gray: the numbers of the table, what goes in; a number with a background reaches its limit and is kept.",
           "blue": "Blue: what each step makes: the messages, the words, and the sentences, each sentence beside the days it is about."},
    "nl": {"title": "Van getallen naar een weerbericht, in drie stappen",
           "gray": "Grijs: de getallen van de tabel, wat erin gaat; een getal met een achtergrond haalt de grens en blijft.",
           "blue": "Blauw: wat elke stap maakt: de boodschappen, de woorden, en de zinnen, elke zin naast de dagen waarover de zin gaat."},
}


def idea_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang]
    body = _week_table(WORKED[0], lang, steps(WORKED[0], lang))
    body += _colors(w, body)
    return yours.figure(w["title"], body, alt=w["title"])


# --- try it: a week and the limits of the learner's choice -----------------------------------------------------------------

_TRY_TEXT = {
    "en": {"title": "The report of {week}",
           "kept": "Kept: {k} of the {n} numbers of the week, in {s} sentences.",
           "false": "{days}: called *dry*, while 1 mm of rain or more falls. Leaving rain out made the text say something that is not true.",
           "gray": "Gray: the numbers of the week; a number with a background reaches its limit.",
           "blue": "Blue: what the steps make.",
           "purple": "Purple: a day called *dry* while 1 mm of rain or more falls, and the sentence that says it."},
    "nl": {"title": "Het weerbericht van {week}",
           "kept": "Gekozen: {k} van de {n} getallen van de week, in {s} zinnen.",
           "false": "{days}: *droog* genoemd, terwijl er 1 mm regen of meer valt. Door de regen weg te laten zegt de tekst iets wat niet waar is.",
           "gray": "Grijs: de getallen van de week; een getal met een achtergrond haalt de grens.",
           "blue": "Blauw: wat de stappen maken.",
           "purple": "Paars: een dag die *droog* heet terwijl er 1 mm regen of meer valt, en de zin die dat zegt."},
}


def _limit(v, default: float, low: float, high: float) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return default if x != x else min(max(x, low), high)


def try_limits(rain=None, wind=None, high=None, sun=None) -> dict:
    """The limits the sliders set, in the order of LIMITS; a value that is not a number gives the course's limit."""
    return {"high": _limit(high, LIMITS["high"], 15, 35), "sun": _limit(sun, LIMITS["sun"], 1, 15),
            "rain": _limit(rain, LIMITS["rain"], 0.5, 20), "wind": _limit(wind, LIMITS["wind"], 1, 12)}


def try_picture(lang: str, week: str, rain=None, wind=None, high=None, sun=None) -> str:
    lang = lang if lang in LANGS else "en"
    week = week if week in WEEKS else "april"
    w = _TRY_TEXT[lang]
    limits = try_limits(rain, wind, high, sun)
    got = steps(week, lang, limits)
    wrong = false_dry(week, got[1], lang)
    body = _week_table(week, lang, got, wrong_days=wrong)
    kept = sum(len(m) for m in got[0])
    body += _note(w["kept"].format(k=_num(kept, lang), n=_num(7 * len(MEASURES), lang), s=_num(len(got[3]), lang)))
    if wrong:
        body += _note(w["false"].format(days=join_names([DAYS[lang][i] for i in wrong], AND[lang])))
    title = w["title"].format(week=WEEK_NAMES[lang][week][0].lower() + WEEK_NAMES[lang][week][1:])
    return yours.figure(title, body + _colors(w, body), alt=title)


# --- the pictures under the exercises ----------------------------------------------------------------------------------------

_SELECT_TEXT = {
    "en": {"title": "Your select on the week in April",
           "summary": "Over the {d} days of the three weeks, your select keeps {k} of the {n} numbers.",
           "differs": "The course's select keeps {want}.",
           "gray": "Gray: the numbers of the week; a number with a background is one your select keeps.",
           "blue": "Blue: what your select gives, and the words and sentences that follow from it, when they are what the course's select gives.",
           "purple": "Purple: what differs from the course's select.",
           "kind": "a list of tuples, each with a kind and a value"},
    "nl": {"title": "Je select op de week in april",
           "summary": "Over de {d} dagen van de drie weken houdt je select {k} van de {n} getallen.",
           "differs": "De select van de course houdt er {want}.",
           "gray": "Grijs: de getallen van de week; een getal met een achtergrond houdt je select.",
           "blue": "Blauw: wat je select geeft, en de woorden en zinnen die eruit volgen, als dat is wat de select van de course geeft.",
           "purple": "Paars: wat anders is dan bij de select van de course.",
           "kind": "een lijst met tuples, elk met een soort en een waarde"},
}


def select_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _SELECT_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")

    def make():
        return [[_as_messages(fn(dict(d), dict(LIMITS))) for d in WEEKS[week]] for week in WEEKS]

    mine = _call(make)
    if mine is _SLOW:
        return _empty(w, "slow")
    if mine is _STOPPED:
        return _empty(w, "stopped")
    if not all(_is_messages(m) for week in mine for m in week):
        return _empty(w, "odd")
    first = list(WEEKS).index(WORKED[0])
    by_day = iter(mine[first])
    got = _call(lambda: steps(WORKED[0], lang, select_=lambda d, lim: next(by_day)))
    if got in (_SLOW, _STOPPED):
        return _empty(w, "stopped")
    body = _week_table(WORKED[0], lang, got, steps(WORKED[0], lang))
    n = numbers()
    kept = sum(len(m) for week in mine for m in week)
    summary = w["summary"].format(d=_num(n["days"], lang), k=_num(kept, lang), n=_num(n["numbers"], lang))
    if mine != [steps(week, lang)[0] for week in WEEKS]:
        summary += " " + w["differs"].format(want=_num(n["messages"], lang))
    return yours.figure(w["title"], body + _note(summary) + _colors(w, body), alt=w["title"])


_AGGREGATE_TEXT = {
    "en": {"title": "Your aggregate on the week in October",
           "summary": "Your aggregate writes {k} sentences for the {d} days of the three weeks; without aggregation, that would be {d}.",
           "differs": "The course's aggregate writes {want}.",
           "gray": "Gray: the numbers of the week.",
           "blue": "Blue: the messages and the words, and each sentence beside the days of its group, when your groups are those of the course's aggregate.",
           "purple": "Purple: a group that differs from the course's aggregate.",
           "kind": "a list of groups, each a tuple of a list of days and the words"},
    "nl": {"title": "Je aggregate op de week in oktober",
           "summary": "Je aggregate schrijft {k} zinnen voor de {d} dagen van de drie weken; zonder aggregatie zouden het er {d} zijn.",
           "differs": "De aggregate van de course schrijft er {want}.",
           "gray": "Grijs: de getallen van de week.",
           "blue": "Blauw: de boodschappen en de woorden, en elke zin naast de dagen van de groep, als je groepen gelijk zijn aan de groepen van de aggregate van de course.",
           "purple": "Paars: een groep die anders is dan bij de aggregate van de course.",
           "kind": "een lijst met groepen, elk een tuple van een lijst met dagen en de woorden"},
}

AGGREGATE_WEEK = "october"


def aggregate_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _AGGREGATE_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")

    def make():
        return [_as_groups(fn(list(steps(week, lang)[1]))) for week in WEEKS]

    mine = _call(make)
    if mine is _SLOW:
        return _empty(w, "slow")
    if mine is _STOPPED:
        return _empty(w, "stopped")
    if not all(_is_groups(g) for g in mine):
        return _empty(w, "odd")
    groups = mine[list(WEEKS).index(AGGREGATE_WEEK)]
    got = steps(AGGREGATE_WEEK, lang, aggregate_=lambda phrases: groups)
    body = _week_table(AGGREGATE_WEEK, lang, got, steps(AGGREGATE_WEEK, lang))
    n = numbers()
    k = sum(len(g) for g in mine)
    summary = w["summary"].format(k=_num(k, lang), d=_num(n["days"], lang))
    if mine != [steps(week, lang)[2] for week in WEEKS]:
        summary += " " + w["differs"].format(want=_num(n["sentences"], lang))
    return yours.figure(w["title"], body + _note(summary) + _colors(w, body), alt=w["title"])


_JOIN_TEXT = {
    "en": {"title": "Your join_names in the report of the week in October",
           "summary": "In the {s} sentences of the three weeks, your join_names writes {k} as the course's join_names does.",
           "gray": "Gray: the numbers of the week.",
           "blue": "Blue: the messages, and the words and the sentences with your join_names for the days and for the words, when they are what the course's join_names gives.",
           "purple": "Purple: what differs from the course's join_names.",
           "kind": "a text"},
    "nl": {"title": "Je join_names in het weerbericht van de week in oktober",
           "summary": "Van de {s} zinnen van de drie weken schrijft je join_names er {k} zoals de join_names van de course.",
           "gray": "Grijs: de getallen van de week.",
           "blue": "Blauw: de boodschappen, en de woorden en de zinnen met je join_names voor de dagen en voor de woorden, als dat is wat de join_names van de course geeft.",
           "purple": "Paars: wat anders is dan bij de join_names van de course.",
           "kind": "een tekst"},
}


def join_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _JOIN_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    odd = []

    def join(names, and_word):
        out = fn(list(names), and_word)
        if not isinstance(out, str):
            odd.append(out)
            return ""
        return out

    mine = _call(lambda: {week: steps(week, lang, join=join) for week in WEEKS})
    if mine is _SLOW:
        return _empty(w, "slow")
    if mine is _STOPPED:
        return _empty(w, "stopped")
    if odd:
        return _empty(w, "odd")
    want = {week: steps(week, lang) for week in WEEKS}
    body = _week_table(AGGREGATE_WEEK, lang, mine[AGGREGATE_WEEK], want[AGGREGATE_WEEK])
    s = sum(len(want[week][3]) for week in WEEKS)
    k = sum(1 for week in WEEKS for a, b in zip(mine[week][3], want[week][3]) if a == b)
    summary = w["summary"].format(s=_num(s, lang), k=_num(k, lang))
    return yours.figure(w["title"], body + _note(summary) + _colors(w, body), alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------------

class _Val:
    """A test value that the check's messages write by its name, such as `C.LIMITS`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value, memo)


def _value(v) -> str:
    """A value the way code writes it, with double quotes."""
    if isinstance(v, bool) or v is None:
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    if isinstance(v, tuple):
        return "(" + ", ".join(_value(x) for x in v) + ("," if len(v) == 1 else "") + ")"
    if isinstance(v, list):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_value(k)}: {_value(x)}" for k, x in v.items()) + "}"
    try:
        return _short(v)
    except Exception:   # noqa: BLE001
        return "?"


def _show(v):
    """A value for a message, the way the lesson writes it, cut when it is very long."""
    out = _value(v)
    return out if len(out) <= 300 else out[:300] + "…"


def _written(v) -> _Val:
    """A small test value that the messages write the way the lesson writes it, with double quotes."""
    return _Val(_value(v), v)


def _as_messages(v):
    """What select gives, with a message that is a list made a tuple: ["rain", 6.5] counts as ("rain", 6.5)."""
    if isinstance(v, (list, tuple)) and all(isinstance(m, (list, tuple)) and len(m) == 2 for m in v):
        return [tuple(m) for m in v]
    return v


def _as_groups(v):
    """What aggregate gives, with a group that is a list made a tuple, and its days a list: [[0, 1], "dry"] counts as
    ([0, 1], "dry")."""
    if isinstance(v, (list, tuple)) and all(isinstance(g, (list, tuple)) and len(g) == 2 for g in v):
        return [(list(g[0]) if isinstance(g[0], (list, tuple)) else g[0], g[1]) for g in v]
    return v


def _dots_left(got, text_) -> bool:
    """A `...` of the start code left in the box: an Ellipsis in the code, not three dots inside a string."""
    try:
        tree = ast.parse(text_)
    except SyntaxError:
        return re.search(r"^[^#\n]*\.\.\.", text_, re.MULTILINE) is not None
    return any(isinstance(n, ast.Constant) and n.value is Ellipsis for n in ast.walk(tree))


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang de `...` door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# The small cases the exercise texts show; the same in both lessons.
TOY_SELECT = ((_day(11, 7, 6.5, 6, 1), {"rain": 1, "wind": 6}),
              (_day(27, 16, 0.0, 2, 13), {"high": 25, "sun": 10, "rain": 1, "wind": 6}))
TOY_AGGREGATE = (["dry", "dry", "rainy", "rainy", "dry"], ["droog", "zonnig", "zonnig"])
TOY_JOIN = ((["Monday", "Wednesday", "Thursday"], "and"), (["zaterdag", "zondag"], "en"))

_SE_HEAD = "def select(day, limits):\n    messages = []\n    for kind in limits:\n"
_SE_IF = "        if day[kind] >= limits[kind]:\n"
_SE_ADD = "            messages.append((kind, day[kind]))\n"


def _se(check: str = _SE_IF, add: str = _SE_ADD) -> str:
    return _SE_HEAD + check + add + "    return messages\n"


_SE = _se()
_SE_START = _se("        if ...:\n", "            messages.append(...)\n")
_SE_KIND = _se(add="            messages.append(kind)\n")
_SE_VALUE = _se(add="            messages.append(day[kind])\n")
_SE_LIMIT = _se(add="            messages.append((kind, limits[kind]))\n")
_SE_SWAPPED = _se(add="            messages.append((day[kind], kind))\n")
_SE_STRICT = _se("        if day[kind] > limits[kind]:\n")
_SE_INVERTED = _se("        if day[kind] <= limits[kind]:\n")
_SE_INSIDE = _SE_HEAD + _SE_IF + _SE_ADD + "        return messages\n"

_AG_HEAD = "def aggregate(phrases):\n    groups = []\n    for i in range(len(phrases)):\n"
_AG_IF = "        if len(groups) > 0 and groups[-1][1] == phrases[i]:\n"
_AG_NEW = "            groups.append(([i], phrases[i]))\n"


def _ag(check: str = _AG_IF, new: str = _AG_NEW) -> str:
    return _AG_HEAD + check + "            groups[-1][0].append(i)\n        else:\n" + new + "    return groups\n"


_AG = _ag()
_AG_START = _ag("        if len(groups) > 0 and ...:\n", "            groups.append(...)\n")
_AG_NO_LEN = _ag("        if groups[-1][1] == phrases[i]:\n")
_AG_OWN = _ag("        if len(groups) > 0 and groups[-1][0] == phrases[i]:\n")
_AG_NUMBER = _ag(new="            groups.append((i, phrases[i]))\n")
_AG_ALL = _ag(new="            groups.append(([i], phrases))\n")
_AG_ANY = (_AG_HEAD + "        found = False\n        for g in groups:\n            if g[1] == phrases[i]:\n"
           "                g[0].append(i)\n                found = True\n        if not found:\n" + _AG_NEW
           + "    return groups\n")
_AG_INSIDE = _ag().replace("    return groups\n", "        return groups\n")

_JN_HEAD = "def join_names(names, and_word):\n    if len(names) == 1:\n"
_JN_ONE = "        return names[0]\n"
_JN_MORE = '    return ", ".join(names[:-1]) + " " + and_word + " " + names[-1]\n'


def _jn(one: str = _JN_ONE, more: str = _JN_MORE) -> str:
    return _JN_HEAD + one + more


_JN = _jn()
_JN_START = _jn("        return ...\n", "    return ...\n")
_JN_TWICE = _jn(more='    return ", ".join(names) + " " + and_word + " " + names[-1]\n')
_JN_NO_SPACE = _jn(more='    return ",".join(names[:-1]) + " " + and_word + " " + names[-1]\n')
_JN_LIST = _jn(one="        return names\n")
_JN_LETTERS = _jn(more='    return ", ".join(names[-1]) + " " + and_word + " " + names[-1]\n')
_JN_FIRST = _jn(more='    return ", ".join(names[1:]) + " " + and_word + " " + names[-1]\n')


def _not_list(got, text_) -> bool:
    return not isinstance(got, list)


def _not_text(got, text_) -> bool:
    return not isinstance(got, str)


def _type_error(e) -> bool:
    return isinstance(e, TypeError) and "not supported between" in str(e)


def _append_two(e) -> bool:
    return isinstance(e, TypeError) and "append" in str(e)


REQUIRED = ["select", "aggregate", "join_names", "steps"]

_DAY_VALS = [_Val(f'C.WEEKS["{week}"][{i}]', WEEKS[week][i]) for week, i in (("april", 4), ("july", 3), ("october", 3))]
_LIMITS_VAL = _Val("C.LIMITS", LIMITS)

EXERCISES = {
    "select": FunctionBox(
        name="select",
        picture=select_picture,
        function="select",
        start=_SE_START,
        solution=_SE,
        show=_show,
        collect=_as_messages,
        visible=[(_written(a), _written(b)) for a, b in TOY_SELECT],
        hidden=[(_written(_day(13, 3, 0.4, 4, 6)), _written({"rain": 1, "wind": 6})),
                (_written(_day(11, 6, 1.0, 6, 3)), _written({"rain": 1, "wind": 6})),
                (_written(_day(11, 7, 6.5, 6, 1)), _written({})),
                (_written(_day(31, 19, 12.0, 5, 7)), _written({"rain": 1, "high": 25})),
                (_written(_day(5, -2, 0.0, 9, 3)), _written({"wind": 6, "low": -10}))]
               + [(d, _LIMITS_VAL) for d in _DAY_VALS],
        first=[_DOTS],
        mistakes=[
            Mistake("append-two", error=_append_two,
                    en="`{call}` stops with a `TypeError`: `append` takes one item, and your code gives it two. How do the kind and the value become one tuple?",
                    nl="`{call}` stopt met een `TypeError`: `append` neemt één item, en je code geeft er twee. Hoe worden de soort en de waarde samen één tuple?"),
            Mistake("type-error", error=_type_error,
                    en="`{call}` stops with a `TypeError`: Python cannot compare what your code puts on the two sides of the comparison, such as a number and a whole dictionary. Does your `if` compare the day's value with the limit of that one kind?",
                    nl="`{call}` stopt met een `TypeError`: Python kan niet vergelijken wat je code aan de twee kanten van de vergelijking zet, zoals een getal en een hele dictionary. Vergelijkt je `if` de waarde van de dag met de grens van deze ene soort?"),
            Mistake("not-list", matches=_not_list,
                    en="`{call}` gives {got}, and it should give {want}: a list of messages. Does your function give back the list `messages`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een lijst met boodschappen. Geeft je functie de lijst `messages` terug?"),
            Mistake("kind-only", wrong=_SE_KIND,
                    en="`{call}` gives {got}, and it should give {want}: the kinds alone. The next step needs the value too, such as the 6.5 mm of rain. What goes in a tuple with the kind?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: alleen de soorten. De volgende stap heeft ook de waarde nodig, zoals de 6,5 mm regen. Wat komt er samen met de soort in een tuple?"),
            Mistake("value-only", wrong=_SE_VALUE,
                    en="`{call}` gives {got}, and it should give {want}: the values alone, so the next step cannot tell rain from wind. What goes in a tuple with the value?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: alleen de waarden, dus de volgende stap ziet niet of het om regen of wind gaat. Wat komt er samen met de waarde in een tuple?"),
            Mistake("limit-value", wrong=_SE_LIMIT,
                    en="`{call}` gives {got}, and it should give {want}: each message holds the limit, 1, and not what was measured, 6.5. Which dictionary holds the measurements of the day?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke boodschap bevat de grens, 1, en niet wat er gemeten is, 6,5. Welke dictionary bevat de metingen van de dag?"),
            Mistake("swapped", wrong=_SE_SWAPPED,
                    en="`{call}` gives {got}, and it should give {want}: each tuple has the value first and the kind second. Which comes first in a message?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke tuple heeft eerst de waarde en dan de soort. Wat komt er eerst in een boodschap?"),
            Mistake("strict", wrong=_SE_STRICT,
                    en="`{call}` gives {got}, and it should give {want}: wind force 6 is not kept, and the limit for wind is 6. A value equal to the limit counts: which comparison says *at least*?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: windkracht 6 blijft niet, en de grens voor wind is 6. Een waarde gelijk aan de grens telt mee: welke vergelijking zegt *minstens*?"),
            Mistake("inverted", wrong=_SE_INVERTED,
                    en="`{call}` gives {got}, and it should give {want}: your function keeps what stays at or below the limit, and leaves out the 6.5 mm of rain. Which side of the limit should a kept value be on?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie houdt wat op of onder de grens blijft, en laat de 6,5 mm regen weg. Aan welke kant van de grens moet een waarde zitten om te blijven?"),
            Mistake("return-inside", wrong=_SE_INSIDE,
                    en="`{call}` gives {got}, and it should give {want}: your function stops after the first kind of `limits`. Where does your `return` stand: inside the `for` loop, or after it?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie stopt na de eerste soort van `limits`. Waar staat je `return`: in de `for`-loop, of erna?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: the `if` asks whether the day's value of `kind` reaches its limit, and `messages.append` adds a tuple of the kind and that value.",
                      "nl": "Vervang de twee `...`: de `if` vraagt of de waarde van de dag voor `kind` de grens haalt, en `messages.append` voegt een tuple toe van de soort en de waarde."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. For each key of `limits`: is the day's value at least the limit? Then a tuple of the kind and the value goes in the list. Are the messages in the order of `limits`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Is voor elke key van `limits` de waarde van de dag minstens de grens? Dan komt er een tuple van de soort en de waarde in de lijst. Staan de boodschappen in de volgorde van `limits`?"},
            "almost": {"en": "Right for the two small cases. But `{call}` gives {got}, and it should give {want}. Does your function keep a value that equals its limit, look only at the kinds of `limits` and in their order, and give an empty list when nothing reaches its limit?",
                       "nl": "Goed voor de twee kleine gevallen. Maar `{call}` geeft {got}, en het moet {want} zijn. Houdt je functie een waarde die gelijk is aan de grens, kijkt de functie alleen naar de soorten van `limits` en in de volgorde van `limits`, en geeft de functie een lege lijst als niets de grens haalt?"},
            "correct": {"en": "Right: your function chooses what the report says. In the picture below, which numbers of the week does it leave out?",
                        "nl": "Goed: je functie kiest wat het weerbericht zegt. Welke getallen van de week laat de functie weg, in het plaatje hieronder?"},
        },
    ),
    "aggregate": FunctionBox(
        name="aggregate",
        picture=aggregate_picture,
        function="aggregate",
        start=_AG_START,
        solution=_AG,
        show=_show,
        collect=_as_groups,
        visible=[(_written(a),) for a in TOY_AGGREGATE],
        hidden=[(_written(["sunny"]),),
                (_written(["dry"] * 7),),
                (_written(["dry", "sunny", "rainy", "windy", "warm", "rainy and windy", "warm and sunny"]),),
                (_written(["rainy", "dry", "rainy", "dry"]),)]
               + [(_Val(f'C.PHRASES["{week}"]', list(steps(week, "en")[1])),) for week in WEEKS],
        first=[_DOTS],
        mistakes=[
            Mistake("append-two", error=_append_two,
                    en="`{call}` stops with a `TypeError`: `append` takes one item, and your code gives it two. How do the list of days and the words become one tuple?",
                    nl="`{call}` stopt met een `TypeError`: `append` neemt één item, en je code geeft er twee. Hoe worden de lijst met dagen en de woorden samen één tuple?"),
            Mistake("first-day", error=lambda e: isinstance(e, IndexError),
                    en="`{call}` stops with an `IndexError`: at the first day, `groups` is still empty, so it has no last group `groups[-1]`. What does the `if` ask first, so that the comparison only runs when there is a group?",
                    nl="`{call}` stopt met een `IndexError`: bij de eerste dag is `groups` nog leeg, dus er is geen laatste groep `groups[-1]`. Wat vraagt de `if` eerst, zodat de vergelijking alleen loopt als er een groep is?"),
            Mistake("number-days", error=lambda e: isinstance(e, AttributeError) and "append" in str(e),
                    en="`{call}` stops with an `AttributeError`: `groups[-1][0].append(i)` adds a day to the days of the last group, and those are a number, not a list. What should a new group start with, so that later days can join it?",
                    nl="`{call}` stopt met een `AttributeError`: `groups[-1][0].append(i)` voegt een dag toe aan de dagen van de laatste groep, en dat is een getal, geen lijst. Waarmee moet een nieuwe groep beginnen, zodat latere dagen erbij kunnen?"),
            Mistake("not-list", matches=_not_list,
                    en="`{call}` gives {got}, and it should give {want}: a list of groups. Does your function give back the list `groups`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een lijst met groepen. Geeft je functie de lijst `groups` terug?"),
            Mistake("own-group", wrong=_AG_OWN,
                    en="`{call}` gives {got}, and it should give {want}: every day has a group of its own, so the comparison is never true. `groups[-1]` is a tuple: what is at place 0, and what at place 1?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke dag heeft een eigen groep, dus de vergelijking is nooit waar. `groups[-1]` is een tuple: wat staat er op plaats 0, en wat op plaats 1?"),
            Mistake("all-phrases", wrong=_AG_ALL,
                    en="`{call}` gives {got}, and it should give {want}: each new group holds the whole list `phrases`, and not the words of its day. Which item of `phrases` belongs to day `i`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke nieuwe groep bevat de hele lijst `phrases`, en niet de woorden van de eigen dag. Welk item van `phrases` hoort bij dag `i`?"),
            Mistake("not-in-a-row", wrong=_AG_ANY,
                    en="`{call}` gives {got}, and it should give {want}: Friday joins Monday and Tuesday, though Wednesday and Thursday lie between them. A report goes through the days in order: which group may day `i` join?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: vrijdag komt bij maandag en dinsdag, terwijl woensdag en donderdag ertussen liggen. Een weerbericht loopt de dagen op volgorde af: bij welke groep mag dag `i` komen?"),
            Mistake("return-inside", wrong=_AG_INSIDE,
                    en="`{call}` gives {got}, and it should give {want}: only the first day. Where does your `return` stand: inside the `for` loop, or after it?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: alleen de eerste dag. Waar staat je `return`: in de `for`-loop, of erna?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: the `if` asks whether the last group has the same words as day `i`, and `groups.append` starts a new group: a tuple of a list with day `i` and its words.",
                      "nl": "Vervang de twee `...`: de `if` vraagt of de laatste groep dezelfde woorden heeft als dag `i`, en `groups.append` begint een nieuwe groep: een tuple van een lijst met dag `i` en de woorden van dag `i`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does each day join the group before it when the words are the same, and start a new group when they differ?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Komt elke dag bij de laatste groep als de woorden hetzelfde zijn, en begint de dag een nieuwe groep als de woorden verschillen?"},
            "almost": {"en": "Right for the two small cases. But `{call}` gives {got}, and it should give {want}. Does your function work for one day, for seven days with the same words, and for words that come back later in the week?",
                       "nl": "Goed voor de twee kleine gevallen. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie voor één dag, voor zeven dagen met dezelfde woorden, en voor woorden die later in de week terugkomen?"},
            "correct": {"en": "Right: your function gives days in a row with the same weather one sentence. How many sentences does that save in the picture below?",
                        "nl": "Goed: je functie geeft dagen achter elkaar met hetzelfde weer één zin. Hoeveel zinnen scheelt dat in het plaatje hieronder?"},
        },
    ),
    "join_names": FunctionBox(
        name="join_names",
        picture=join_picture,
        function="join_names",
        start=_JN_START,
        solution=_JN,
        show=_show,
        visible=[(_written(a), _written(b)) for a, b in TOY_JOIN],
        hidden=[(_written(["Friday"]), _written("and")),
                (_written(["warm", "sunny", "windy"]), _written("and")),
                (_written(list(DAYS["nl"])), _written("en")),
                (_written(["regenachtig", "winderig"]), _written("en")),
                (_written(["Sunday"]), _written("en"))],
        first=[_DOTS],
        mistakes=[
            Mistake("not-text", matches=_not_text,
                    en="`{call}` gives {got}, and it should give {want}: one text. What does your function give back for one name, and what for more?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: één tekst. Wat geeft je functie terug bij één naam, en wat bij meer?"),
            Mistake("last-twice", wrong=_JN_TWICE,
                    en="`{call}` gives {got}, and it should give {want}: the last name comes twice, once in the commas and once after `and_word`. Which slice gives every name but the last?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de laatste naam staat er twee keer, één keer tussen de komma's en één keer na `and_word`. Welke slice geeft elke naam behalve de laatste?"),
            Mistake("first-lost", wrong=_JN_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: the first name is missing and the last comes twice. Where does a slice start when you leave out the first number?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de eerste naam ontbreekt en de laatste staat er twee keer. Waar begint een slice als je het eerste getal weglaat?"),
            Mistake("letters", wrong=_JN_LETTERS,
                    en="`{call}` gives {got}, and it should give {want}: `join` on one text puts a comma between its letters. Which list of names goes before `and_word`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `join` op één tekst zet een komma tussen de letters. Welke lijst met namen hoort in de `join`?"),
            Mistake("no-space", wrong=_JN_NO_SPACE,
                    en="`{call}` gives {got}, and it should give {want}: no space after each comma. What text goes before `.join`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: na elke komma ontbreekt een spatie. Welke tekst staat er direct links van `.join`?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: for one name, give back that name; for more, `\", \".join` gets every name but the last, and `and_word` and the last name come after it.",
                      "nl": "Vervang de twee `...`: geef bij één naam precies deze naam terug; bij meer krijgt `\", \".join` elke naam behalve de laatste, met daarna `and_word` en de laatste naam."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is every name there once, with a comma and a space between them, and `and_word` between the last two?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Staat elke naam er één keer, met een komma en een spatie ertussen, en `and_word` tussen de laatste twee?"},
            "almost": {"en": "Right for the two small cases. But `{call}` gives {got}, and it should give {want}. Does your function give one name as it is, and work for seven names and for words that are not days?",
                       "nl": "Goed voor de twee kleine gevallen. Maar `{call}` geeft {got}, en het moet {want} zijn. Geeft je functie één naam onveranderd terug, en werkt de functie voor zeven namen en voor woorden die geen dagen zijn?"},
            "correct": {"en": "Right: your function writes a list the way the language does. In the picture below it joins the days and the words: where does it do both in one sentence?",
                        "nl": "Goed: je functie schrijft een opsomming zoals de taal dat doet. In het plaatje hieronder voegt de functie de dagen en de woorden samen: waar doet de functie allebei in één zin?"},
        },
    ),
    "steps": Sort(
        name="steps",
        groups={
            "selection": {"en": "Content selection", "nl": "Content selection"},
            "micro": {"en": "Microplanning", "nl": "Microplanning"},
            "surface": {"en": "Surface realization", "nl": "Surface realization"},
        },
        items={
            "night": {"en": "**The night.** The table holds the lowest temperature of each night, and the report never says it.",
                      "nl": "**De nacht.** De tabel bevat de laagste temperatuur van elke nacht, en het weerbericht noemt de nachttemperatuur nooit."},
            "windy": {"en": "**Windy.** Wind force 6 becomes the word *windy*, not a number.",
                      "nl": "**Winderig.** Windkracht 6 wordt het woord *winderig*, geen getal."},
            "comma": {"en": "**Commas.** A comma goes between the days, and *and* before the last one.",
                      "nl": "**Komma's.** Tussen de dagen staat een komma, en tussen de laatste twee *en*."},
            "millimeter": {"en": "**One millimeter.** Rain counts from 1 mm on.",
                           "nl": "**Eén millimeter.** Regen telt pas vanaf 1 mm."},
            "share": {"en": "**One sentence.** Wednesday and Thursday, with the same weather, share a sentence.",
                      "nl": "**Eén zin.** Woensdag en donderdag, met hetzelfde weer, delen een zin."},
            "verb": {"en": "**Op maandag wordt het.** Dutch puts *het* (it) after the verb, where English puts *it* first.",
                     "nl": "**Op maandag wordt het.** Het Nederlands zet *het* na het werkwoord, waar het Engels *it* eerst zet."},
            "order": {"en": "**Monday first.** The report goes through the days in order.",
                      "nl": "**Maandag eerst.** Het weerbericht loopt de dagen op volgorde af."},
            "again": {"en": "**Again.** *On Friday, it will be dry again*: a word that ties a sentence to the earlier ones.",
                      "nl": "**Opnieuw.** *Op vrijdag wordt het opnieuw droog*: een woord dat een zin aan de eerdere zinnen vastknoopt."},
        },
        solution={"night": "selection", "windy": "micro", "comma": "surface", "millimeter": "selection", "share": "micro",
                  "verb": "surface", "order": "selection", "again": "micro"},
        nudge={
            "night": {"en": "Does this decide what the report says, which words it uses, or how a sentence is written out?",
                      "nl": "Bepaalt dit wat het weerbericht zegt, welke woorden het gebruikt, of hoe een zin wordt uitgeschreven?"},
            "windy": {"en": "The fact is chosen already: does this decide the fact, the word for it, or the commas and the order of the words?",
                      "nl": "Het feit is al gekozen: bepaalt dit het feit, het woord dat het feit uitdrukt, of de komma's en de volgorde van de woorden?"},
            "comma": {"en": "The days and the words are chosen already: what is left to do?",
                      "nl": "De dagen en de woorden zijn al gekozen: wat moet er nog gebeuren?"},
            "millimeter": {"en": "Does a day with 0.4 mm get a message at all?",
                           "nl": "Krijgt een dag met 0,4 mm wel een boodschap?"},
            "share": {"en": "The facts stay the same: does this decide what is said, how it is planned in sentences, or how a sentence is written out?",
                      "nl": "De feiten blijven hetzelfde: bepaalt dit wat er gezegd wordt, hoe het over zinnen wordt verdeeld, of hoe een zin wordt uitgeschreven?"},
            "verb": {"en": "The sentence says the same in both languages: what differs is the order of its words. Which step writes that out?",
                     "nl": "De zin zegt in beide talen hetzelfde: wat verschilt, is de volgorde van de woorden. Welke stap schrijft dat uit?"},
            "order": {"en": "Reiter names deciding what to say and how to order it together: the two are usually done at the same time. Which step is that?",
                      "nl": "Reiter noemt kiezen wat je zegt en de volgorde ervan samen: meestal gebeuren de twee tegelijk. Welke stap is dat?"},
            "again": {"en": "The word adds no fact and needs no grammar rule: it plans how the sentences hang together. Which step plans sentences?",
                      "nl": "Het woord voegt geen feit toe en is geen grammaticaregel: het plant hoe de zinnen samenhangen. Welke stap plant zinnen?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the groups again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de groepen opnieuw uit."},
            "empty": {"en": "Put each decision in the step that makes it.",
                      "nl": "Zet elke beslissing bij de stap die de beslissing neemt."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} belongs in another step.",
                      "nl": "{item} hoort bij een andere stap."},
            "correct": {"en": "All {n} are right. Content selection decides what is said and in which order; microplanning decides the words and how the facts fill sentences; surface realization writes each sentence out by the rules of its language.",
                        "nl": "Alle {n} kloppen. Content selection bepaalt wat er gezegd wordt en in welke volgorde; microplanning bepaalt de woorden en hoe de feiten over de zinnen worden verdeeld; surface realization schrijft elke zin uit volgens de regels van de taal."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, code or choices, state, known mistake).
CASES = [
    ("select", _SE_START, "empty", None),
    ("select", _SE, "correct", None),
    ("select", _se(add="            messages.append(...)\n"), "wrong", "dots-left"),
    ("select", _se("        if day[kind] >= limits:\n"), "error", "type-error"),
    ("select", _se(add="            messages.append(kind, day[kind])\n"), "error", "append-two"),
    ("select", _se().replace("    return messages\n", "    return len(messages)\n"), "wrong", "not-list"),
    ("select", _SE_KIND, "wrong", "kind-only"),
    ("select", _SE_VALUE, "wrong", "value-only"),
    ("select", _SE_LIMIT, "wrong", "limit-value"),
    ("select", _SE_SWAPPED, "wrong", "swapped"),
    ("select", _SE_STRICT, "wrong", "strict"),
    ("select", _SE_INVERTED, "wrong", "inverted"),
    ("select", _SE_INSIDE, "wrong", "return-inside"),
    ("select", _se("        if limits[kind] <= day[kind]:\n"), "correct", None),
    ("select", _se(add="            messages.append([kind, day[kind]])\n"), "correct", None),
    ("select", "def select(day, limits):\n    return [(k, day[k]) for k in limits if day[k] >= limits[k]]\n", "correct", None),
    ("select", "def select(day, limits):\n    messages = []\n    for kind, limit in limits.items():\n"
     "        if day[kind] >= limit:\n            messages = messages + [(kind, day[kind])]\n    return messages\n",
     "correct", None),
    ("aggregate", _AG_START, "empty", None),
    ("aggregate", _AG, "correct", None),
    ("aggregate", _ag(new="            groups.append(...)\n"), "wrong", "dots-left"),
    ("aggregate", _AG_NO_LEN, "error", "first-day"),
    ("aggregate", _ag(new="            groups.append([i], phrases[i])\n"), "error", "append-two"),
    ("aggregate", _AG_NUMBER, "error", "number-days"),
    ("aggregate", _ag().replace("    return groups\n", "    return len(groups)\n"), "wrong", "not-list"),
    ("aggregate", _AG_OWN, "wrong", "own-group"),
    ("aggregate", _AG_ALL, "wrong", "all-phrases"),
    ("aggregate", _AG_ANY, "wrong", "not-in-a-row"),
    ("aggregate", _AG_INSIDE, "wrong", "return-inside"),
    ("aggregate", _ag(new="            groups.append([[i], phrases[i]])\n"), "correct", None),
    ("aggregate", _ag("        if groups != [] and phrases[i] == groups[-1][1]:\n"), "correct", None),
    ("aggregate", "def aggregate(phrases):\n    groups = []\n    for i, p in enumerate(phrases):\n"
     "        if i > 0 and phrases[i - 1] == p:\n            groups[-1][0].append(i)\n        else:\n"
     "            groups.append(([i], p))\n    return groups\n", "correct", None),
    ("join_names", _JN_START, "empty", None),
    ("join_names", _JN, "correct", None),
    ("join_names", _jn("        return ...\n"), "wrong", "dots-left"),
    ("join_names", _JN_LIST, "wrong", "not-text"),
    ("join_names", _JN_TWICE, "wrong", "last-twice"),
    ("join_names", _JN_FIRST, "wrong", "first-lost"),
    ("join_names", _JN_LETTERS, "wrong", "letters"),
    ("join_names", _JN_NO_SPACE, "wrong", "no-space"),
    ("join_names", _jn(more='    return ", ".join(names[:len(names) - 1]) + " " + and_word + " " + names[-1]\n'),
     "correct", None),
    ("join_names", 'def join_names(names, and_word):\n    if len(names) > 1:\n'
     '        return ", ".join(names[:-1]) + f" {and_word} " + names[-1]\n    return names[0]\n', "correct", None),
    ("join_names", _jn("        return ''.join(names)\n"), "correct", None),
    ("steps", {}, "empty", None),
    ("steps", {"night": "selection", "windy": "micro", "comma": "surface", "millimeter": "selection", "share": "micro",
               "verb": "surface", "order": "selection", "again": "micro"}, "correct", None),
    ("steps", {"night": "selection", "windy": "micro"}, "wrong", "unfinished"),
    ("steps", {"night": "micro"}, "wrong", "wrong:night"),
    ("steps", {"night": "selection", "windy": "surface"}, "wrong", "wrong:windy"),
    ("steps", {"night": "selection", "windy": "micro", "comma": "micro"}, "wrong", "wrong:comma"),
    ("steps", {"millimeter": "micro"}, "wrong", "wrong:millimeter"),
    ("steps", {"share": "surface"}, "wrong", "wrong:share"),
    ("steps", {"verb": "micro"}, "wrong", "wrong:verb"),
    ("steps", {"order": "surface"}, "wrong", "wrong:order"),
    ("steps", {"again": "surface"}, "wrong", "wrong:again"),
]

PHRASES = {week: list(steps(week, "en")[1]) for week in WEEKS}   # the words of each day, in English, as aggregate gets them
