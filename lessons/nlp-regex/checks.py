# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-regex: find every number of four digits in a text (four_digits), say whether a string is a
Dutch postcode (is_postcode), and find every clock time in a text as (hours, minutes) (clock_times), and sort tasks by
the tool that fits them: a plain `in`, re.search, re.findall, or more than a pattern.

The sentences are Dutch sentences of Tatoeba (CC BY 2.0 FR) in data/tatoeba-nl.tsv.gz (made by
data/tools/make_tatoeba_nl.py), each taken by its id. Every number in the pictures is counted by the course; the other
strings in the cases of the checks and the tasks of the sort are the course's own."""
import contextlib
import copy
import html
import io
import re

from everything_ai import books, yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, kept, with_time_limit

# --- the course's solutions ------------------------------------------------------------------------------------------


def four_digits(text):
    """The course's solution of Exercise 1: every number of exactly four digits in the text, as strings, in order; a
    number that is part of a longer one does not count."""
    return re.findall(r"\b\d{4}\b", text)


def is_postcode(s):
    """The course's solution of Exercise 2: True when the whole string is a Dutch postcode: four digits, the first not
    0, at most one space, and two capital letters."""
    return re.search(r"^[1-9]\d{3} ?[A-Z]{2}$", s) is not None


def clock_times(text):
    """The course's solution of Exercise 3: every clock time in the text, written with a dot or a colon, as a pair of
    whole numbers (hours, minutes)."""
    found = re.findall(r"\b(\d{1,2})[.:](\d\d)\b", text)
    return [(int(h), int(m)) for h, m in found]


# --- the sentences of Tatoeba ----------------------------------------------------------------------------------------


def _by_id() -> dict:
    return kept("nlp-regex:by-id", lambda: dict(books.dutch_sentences()))


def sentence(i: int) -> str:
    """The Dutch sentence of Tatoeba with id i."""
    return _by_id()[i]


def _text() -> str:
    return kept("nlp-regex:text", lambda: "\n".join(t for _, t in books.dutch_sentences()))


def tatoeba_text() -> str:
    """All Dutch sentences of Tatoeba as one string, a sentence on each line."""
    return _text()


def n_sentences() -> int:
    return len(_by_id())


# the sentences the lesson shows, by their ids at Tatoeba
STAR_ID, ECLIPSE_ID, PAINTER_ID = 11358565, 7022790, 403752
PHONE_ID, YEAR_2345_ID, BROCHURE_ID, WAR_ID, WINDOWS_ID = 4543728, 13839744, 4552646, 3600696, 690765
TEACHERS_ID, TITANIC_ID, SUNSET_ID, BREAK_ID, TOM_ID = 3591387, 4558680, 3037204, 8229022, 5459019


def star() -> str:
    """The sentence of the visible case of Exercise 1: a year and a longer number."""
    return sentence(STAR_ID)


def eclipse() -> str:
    """The sentence of the worked example and of the visible case of Exercise 3: two clock times."""
    return sentence(ECLIPSE_ID)


# --- small helpers for the pictures ----------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums"
_MISS = "color:var(--ea-loss-text);font-weight:600"
_LINE = "font-family:var(--ea-font-ui);font-size:0.9rem;color:var(--ea-ink);line-height:1.9"
_CODE = "font-family:var(--ea-font-code);font-size:0.82rem;color:var(--ea-ink)"
_HIT = "background:var(--ea-pred-wash);color:var(--ea-pred-text);border-radius:3px;padding:0.05rem 0.1rem"
_WRONG_HIT = "background:var(--ea-loss-wash);color:var(--ea-loss-text);border-radius:3px;padding:0.05rem 0.1rem"
_MISSED = "outline:1px dashed var(--ea-ink-3);outline-offset:1px;border-radius:3px"
_BAR = "display:inline-block;height:0.7rem;background:var(--ea-data);border-radius:2px;vertical-align:middle"


def _note(text: str) -> str:
    """A line under a picture; a word between asterisks is set in italics, as in the lesson's text."""
    parts = html.escape(text).split("*")
    marked = "".join(f"<i>{p}</i>" if k % 2 else p for k, p in enumerate(parts))
    return f'<div style="{_UI}">{marked}</div>'


def _num(n, lang: str) -> str:
    """A whole number with a separator for thousands: 2,319 in English, 2.319 in Dutch."""
    text = f"{round(n):,}"
    return text.replace(",", ".") if lang == "nl" else text


def _table(rows_: list, head: list, numbers: tuple = (), styles: dict | None = None) -> str:
    """A small table; rows hold HTML that is already escaped; numbers are the columns set to the right; styles maps
    (row, column) to extra style."""
    styles = styles or {}
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{_NUMBER if j in numbers else ""}{styles.get((i, j), "")}">'
                                    f'{c}</td>' for j, c in enumerate(row)) + "</tr>" for i, row in enumerate(rows_))
    return (f'<table style="border-collapse:collapse;margin:0.3rem 0 0.2rem"><thead><tr>{top}</tr></thead>'
            f"<tbody>{body}</tbody></table>")


def _block(title: str, inner: str) -> str:
    return f'<div><div style="{_HEAD};border:none;padding-left:0.45rem">{html.escape(title)}</div>{inner}</div>'


def marked(text: str, spans: list) -> str:
    """The text as HTML, with each (start, end, style) of `spans` in a span with that style; spans do not overlap."""
    out, at = [], 0
    for start, end, style in sorted(spans):
        if start < at:
            continue
        out.append(html.escape(text[at:start]))
        out.append(f'<span style="{style}">{html.escape(text[start:end])}</span>')
        at = end
    out.append(html.escape(text[at:]))
    return "".join(out)


def _bar(n: int, most: int, width: float = 9.0) -> str:
    w = 0 if not most else max(0.15, width * n / most) if n else 0
    return f'<span style="{_BAR};width:{w:.2f}rem"></span>' if w else ""


# --- the idea's picture: three patterns for the word de ---------------------------------------------------------------

IDEA_PATTERNS = [r"de", r"[dD]e", r"\b[dD]e\b"]
_WORD_DE = re.compile(r"\b[dD]e\b")


def pattern_counts() -> dict:
    """How many matches each pattern of the idea's picture finds in all Dutch sentences of Tatoeba."""
    return kept("nlp-regex:idea-counts", lambda: {p: len(re.findall(p, _text())) for p in IDEA_PATTERNS})


_IDEA_WORDS = {
    "en": {"title": "Three patterns for the word de",
           "row": "{p}: {k} matches here, {n} in Tatoeba",
           "note": "Blue: a match that is the word de or De. Purple: a match inside another word, a false positive. "
                   "Dashed: the word De, which the pattern misses, a false negative. The word de or De occurs {n} "
                   "times in the Dutch sentences of Tatoeba.",
           "alt": "The sentence {s} with the matches of three patterns. {rows}. {note}"},
    "nl": {"title": "Drie patronen voor het woord de",
           "row": "{p}: hier {k} matches, in Tatoeba {n}",
           "note": "Blauw: een match die het woord de of De is. Paars: een match binnen een ander woord, een false "
                   "positive. Gestippeld: het woord De, dat het patroon mist, een false negative. Het woord de of De "
                   "komt {n} keer voor in de Nederlandse zinnen van Tatoeba.",
           "alt": "De zin {s} met de matches van drie patronen. {rows}. {note}"},
}


def idea_rows(text: str) -> list:
    """For each pattern of the picture: (pattern, spans with their styles, number of matches) on the text."""
    words = [(m.start(), m.end()) for m in _WORD_DE.finditer(text)]
    rows_ = []
    for p in IDEA_PATTERNS:
        spans, found = [], set()
        for m in re.finditer(p, text):
            hit = (m.start(), m.end())
            found.add(hit)
            spans.append((m.start(), m.end(), _HIT if hit in words else _WRONG_HIT))
        spans += [(a, b, _MISSED) for a, b in words if (a, b) not in found]
        rows_.append((p, spans, len(found)))
    return rows_


def idea_picture(lang: str) -> str:
    w = _IDEA_WORDS[lang]
    text = sentence(PAINTER_ID)
    counts = pattern_counts()
    body, alts = [], []
    for p, spans, k in idea_rows(text):
        line = w["row"].format(p=p, k=k, n=_num(counts[p], lang))
        body.append(f'<div style="margin:0.35rem 0"><div style="{_CODE};color:var(--ea-ink-2)">{html.escape(line)}</div>'
                    f'<div style="{_LINE}">{marked(text, spans)}</div></div>')
        alts.append(line)
    words = len(_WORD_DE.findall(_text()))
    note = w["note"].format(n=_num(words, lang))
    return yours.figure(w["title"], "".join(body) + _note(note),
                        alt=w["alt"].format(s=text, rows="; ".join(alts), note=note))


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


def _strings(v) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


def _pairs(v) -> bool:
    return isinstance(v, list) and all(isinstance(x, tuple) and len(x) == 2 and all(
        isinstance(p, int) and not isinstance(p, bool) for p in x) for x in v)


# --- the picture under exercise 1: the numbers of four digits in Tatoeba ----------------------------------------------

CENTURIES = list(range(10, 21))   # 1000-1099 up to 2000-2099
_FOUR_WORDS = {
    "en": {"title": "Numbers of four digits in Tatoeba",
           "head": ["numbers", "times", ""],
           "century": "{a}–{b}", "other": "other numbers",
           "line": "Your four_digits finds {n} numbers in the {s} Dutch sentences of Tatoeba.",
           "note": "The most common other numbers: {others}.",
           "stopped": "Your function stopped with an error on the sentences, so the picture stays empty.",
           "odd": "Your function does not give back a list of strings, so the picture stays empty."},
    "nl": {"title": "Getallen van vier cijfers in Tatoeba",
           "head": ["getallen", "keer", ""],
           "century": "{a}–{b}", "other": "andere getallen",
           "line": "Jouw four_digits vindt {n} getallen in de {s} Nederlandse zinnen van Tatoeba.",
           "note": "De meest voorkomende andere getallen: {others}.",
           "stopped": "Je functie stopte met een fout op de zinnen, dus de afbeelding blijft leeg.",
           "odd": "Je functie geeft geen lijst met strings terug, dus de afbeelding blijft leeg."},
}


def four_rows(found: list) -> tuple:
    """How many of the found numbers fall in each century from 1000 to 2099, how many fall outside, and the most common
    numbers outside, as (number, count)."""
    per = {c: 0 for c in CENTURIES}
    others = {}
    for s in found:
        n = int(s) if s.isdigit() else None
        if n is not None and 1000 <= n <= 2099:
            per[n // 100] += 1
        else:
            others[s] = others.get(s, 0) + 1
    common = sorted(others.items(), key=lambda kv: (-kv[1], kv[0]))
    return per, sum(others.values()), common


def four_picture(seen) -> str:
    lang = seen.lang
    w = _FOUR_WORDS[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    got = _call(fn, _text())
    if got is _STOPPED:
        return yours.figure(w["title"], _note(w["stopped"]), alt=w["stopped"])
    if not _strings(got):
        return yours.figure(w["title"], _note(w["odd"]), alt=w["odd"])
    per, other, common = four_rows(got)
    most = max(list(per.values()) + [other])
    rows_ = [[w["century"].format(a=c * 100, b=c * 100 + 99), _num(per[c], lang), _bar(per[c], most)] for c in CENTURIES]
    rows_.append([html.escape(w["other"]), _num(other, lang), _bar(other, most)])
    line = w["line"].format(n=_num(len(got), lang), s=_num(n_sentences(), lang))
    others = ", ".join(f"{s} ({_num(k, lang)}×)" for s, k in common[:5])
    note = w["note"].format(others=others) if common else ""
    alt = (w["title"] + ". " + line + " " + "; ".join(f"{r[0]}: {r[1]}" for r in rows_) + ". " + note)
    body = (f'<div style="{_LINE}">{html.escape(line)}</div>' + _table(rows_, w["head"], numbers=(1,))
            + (_note(note) if note else ""))
    return yours.figure(w["title"], body, alt=alt)


# --- the picture under exercise 2: what your function says of what people type in a form ------------------------------

FORM_INPUTS = ["1234 AB", "1234AB", "2514 AT", "1000 AA", "1234 ab", "0123 AB", "123 AB", "12345 AB", "1234 ABC",
               "1234  AB", "1234-AB", "Postcode: 1234 AB", ""]
_POST_WORDS = {
    "en": {"title": "What your form lets through",
           "head": ["typed", "yours", "the course"], "yes": "postcode", "no": "not a postcode", "empty": "(nothing)",
           "note": "Each line is a text someone could type in the postcode field of a form. Purple: your function "
                   "says something different from the course's pattern.",
           "stopped": "Your function stopped with an error on one of these texts, so the picture stays empty."},
    "nl": {"title": "Wat je formulier doorlaat",
           "head": ["getypt", "jouw functie", "de course"], "yes": "postcode", "no": "geen postcode",
           "empty": "(niets)",
           "note": "Elke regel is een tekst die iemand in het postcodeveld van een formulier kan typen. Paars: je "
                   "functie zegt iets anders dan het patroon van de course.",
           "stopped": "Je functie stopte met een fout op een van deze teksten, dus de afbeelding blijft leeg."},
}


def postcode_picture(seen) -> str:
    lang = seen.lang
    w = _POST_WORDS[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    rows_, styles, alts = [], {}, []
    for i, s in enumerate(FORM_INPUTS):
        got = _call(fn, s)
        if got is _STOPPED:
            return yours.figure(w["title"], _note(w["stopped"]), alt=w["stopped"])
        want = is_postcode(s)
        mine = w["yes"] if got is True else w["no"] if got is False else repr(got)[:40]
        shown = f'<span style="{_CODE};white-space:pre">{html.escape(repr(s)[1:-1] if s else w["empty"])}</span>'
        rows_.append([shown, html.escape(mine), html.escape(w["yes"] if want else w["no"])])
        if got is not want:
            styles[(i, 1)] = ";" + _MISS
        alts.append(f"{s or w['empty']}: {mine}, {w['yes'] if want else w['no']}")
    return yours.figure(w["title"], _table(rows_, w["head"], styles=styles) + _note(w["note"]),
                        alt=w["title"] + ". " + "; ".join(alts) + ". " + w["note"])


# --- the picture under exercise 3: the clock times in Tatoeba, per hour ------------------------------------------------

WHOLE_HOURS = re.compile(r"\bom \d{1,2} uur\b")
_TIME_WORDS = {
    "en": {"title": "Clock times in Tatoeba, per hour",
           "line": "Your clock_times finds {n} times in the Dutch sentences of Tatoeba; {k} of them are 2:30.",
           "odd_times": "Not a time of day: {odd}.",
           "windows": "The {v} of *Windows voor werkgroepen {v}* is the number of a version, not a time.",
           "missed": "Missed: a time without a dot or a colon, such as *om 8 uur*: *om*, a number and *uur* occur together "
                     "{m} times in Tatoeba.",
           "stopped": "Your function stopped with an error on the sentences, so the picture stays empty.",
           "odd": "Your function does not give back a list of pairs of whole numbers, so the picture stays empty."},
    "nl": {"title": "Kloktijden in Tatoeba, per uur",
           "line": "Jouw clock_times vindt {n} tijden in de Nederlandse zinnen van Tatoeba; {k} daarvan zijn 2:30.",
           "odd_times": "Geen tijd van de dag: {odd}.",
           "windows": "De {v} van *Windows voor werkgroepen {v}* is het nummer van een versie, geen tijd.",
           "missed": "Gemist: een tijd zonder punt of dubbele punt, zoals *om 8 uur*: *om*, een getal en *uur* staan {m} "
                     "keer achter elkaar in Tatoeba.",
           "stopped": "Je functie stopte met een fout op de zinnen, dus de afbeelding blijft leeg.",
           "odd": "Je functie geeft geen lijst met paren hele getallen terug, dus de afbeelding blijft leeg."},
}


def whole_hours() -> int:
    """How often *om*, a number and *uur* occur together, such as "om 8 uur", in the Dutch sentences of Tatoeba: times
    without a dot or a colon, which the pattern of exercise 3 misses."""
    return kept("nlp-regex:om-hours", lambda: len(WHOLE_HOURS.findall(_text())))


def _hours_chart(per: dict, lang: str) -> str:
    most = max(per.values()) or 1
    cols = []
    for h in range(24):
        n = per[h]
        height = 4.0 * n / most
        bar = (f'<div style="width:0.9rem;height:{height:.2f}rem;background:var(--ea-data);border-radius:2px 2px 0 0">'
               f'</div>') if n else '<div style="height:0"></div>'
        cols.append(f'<div style="display:flex;flex-direction:column;align-items:center;justify-content:flex-end;'
                    f'width:1.25rem"><div style="{_UI};margin:0;font-size:0.7rem">{n or ""}</div>{bar}'
                    f'<div style="{_UI};margin:0;font-size:0.7rem;border-top:1px solid var(--ea-line);width:100%;'
                    f'text-align:center">{h}</div></div>')
    return (f'<div style="display:flex;align-items:flex-end;gap:0.05rem;height:6.2rem;margin:0.4rem 0;'
            f'overflow-x:auto">{"".join(cols)}</div>')


def times_picture(seen) -> str:
    lang = seen.lang
    w = _TIME_WORDS[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    got = _call(fn, _text())
    if got is _STOPPED:
        return yours.figure(w["title"], _note(w["stopped"]), alt=w["stopped"])
    if not _pairs(got):
        return yours.figure(w["title"], _note(w["odd"]), alt=w["odd"])
    per = {h: 0 for h in range(24)}
    odd = []
    for h, m in got:
        if 0 <= h <= 23 and 0 <= m <= 59:
            per[h] += 1
        else:
            odd.append(f"{h}:{m:02d}")
    line = w["line"].format(n=_num(len(got), lang), k=_num(got.count((2, 30)), lang))
    notes = []
    if odd:
        notes.append(w["odd_times"].format(odd=", ".join(sorted(set(odd))[:6])))
    if (3, 11) in got:
        notes.append(w["windows"].format(v="3.11"))
    notes.append(w["missed"].format(m=_num(whole_hours(), lang)))
    alt = (w["title"] + ". " + line + " " + ", ".join(f"{h}: {per[h]}" for h in range(24) if per[h]) + ". "
           + " ".join(notes).replace("*", ""))
    body = f'<div style="{_LINE}">{html.escape(line)}</div>' + _hours_chart(per, lang) + "".join(_note(n) for n in notes)
    return yours.figure(w["title"], body, alt=alt)


# --- the demo: a pattern of your own on all sentences -----------------------------------------------------------------

DEMO_LINES = 8
DEMO_LIMIT = 2.0   # seconds a pattern gets for all sentences
DEMO_LONGEST = 200
_DEMO_WORDS = {
    "en": {"title": "Your pattern in Tatoeba",
           "count": "{k} matches in {n} of the {s} sentences.",
           "none": "No match in the {s} sentences.",
           "empty": "Type a pattern in the box.",
           "long": "This pattern is longer than {k} characters; try a shorter one.",
           "error": "Python cannot read this pattern; it gets stuck at character {pos}. Is a bracket missing, or does "
                    "a backslash come before a letter that has no meaning after it?",
           "nested": "This pattern repeats a group that repeats something itself, such as (a+)+. That can take very "
                     "long, as the watch-out says, so the demo does not run it.",
           "slow": "This pattern took more than {k:g} seconds on all the sentences, so the demo stopped it. That "
                   "happens with a pattern whose repeating parts can match the same characters, and sometimes with one "
                   "that matches almost everywhere.",
           "zero": "The pattern also matches empty text; only matches of at least one character are counted.",
           "note": "A few of the sentences with a match, spread over all of them; each match is blue.",
           "alt": "How many sentences of Tatoeba the pattern matches, and a few of them with the matches marked."},
    "nl": {"title": "Jouw patroon in Tatoeba",
           "count": "{k} matches in {n} van de {s} zinnen.",
           "none": "Geen match in de {s} zinnen.",
           "empty": "Typ een patroon in het vak.",
           "long": "Dit patroon is langer dan {k} tekens; probeer een korter patroon.",
           "error": "Python kan dit patroon niet lezen; het loopt vast bij teken {pos}. Mist er een haakje, of staat er "
                    "een backslash voor een letter die daarna geen betekenis heeft?",
           "nested": "Dit patroon herhaalt een groep die zelf iets herhaalt, zoals (a+)+. Dat kan heel lang duren, zoals "
                     "de waarschuwing zegt, dus de demo voert het niet uit.",
           "slow": "Dit patroon duurde langer dan {k:g} seconden voor alle zinnen, dus de demo stopte het. Dat "
                   "gebeurt bij een patroon waarvan de herhalende delen dezelfde tekens kunnen matchen, en soms bij een "
                   "patroon dat bijna overal matcht.",
           "zero": "Het patroon matcht ook lege tekst; alleen matches van minstens één teken tellen mee.",
           "note": "Een paar zinnen met een match, verspreid over alle zinnen; elke match is blauw.",
           "alt": "In hoeveel zinnen van Tatoeba het patroon matcht, en een paar ervan met de matches gemarkeerd."},
}
_ESCAPED = re.compile(r"\\.|\[(?:\\.|[^\]\\])*\]")   # an escaped character, or a character class
_GROUP_START = re.compile(r"^\?(?::|=|!|<=|<!|P<\w+>|<\w+>)")


def repeats_a_repeat(pattern: str) -> bool:
    """True when the pattern repeats a group (with *, + or {...}) that itself holds a repetition or a choice with |,
    such as (a+)+ or (a|aa)*: the kind of pattern that can backtrack for a very long time."""
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


def demo_lines(pattern, k: int = DEMO_LINES) -> tuple:
    """(number of matches, number of sentences with a match, k of those sentences spread over all, with the spans of
    their matches); matches of no characters are left out."""
    rx = re.compile(pattern)
    sentences = [s for _, s in books.dutch_sentences()]
    hits = []    # where the sentences with a match are; only the chosen ones get their spans, which keeps it quick
    total = 0
    for i, s in enumerate(sentences):
        if rx.groups == 0:
            found = rx.findall(s)
            count = len(found) - found.count("")
        else:
            count = sum(1 for m in rx.finditer(s) if m.end() > m.start())
        if count:
            total += count
            hits.append(i)
    n = len(hits)
    if n > k:
        hits = [hits[round(j * (n - 1) / (k - 1))] for j in range(k)]
    chosen = [(sentences[i], [(m.start(), m.end()) for m in rx.finditer(sentences[i]) if m.end() > m.start()])
              for i in hits]
    return total, n, chosen


def demo_picture(pattern, lang: str) -> str:
    w = _DEMO_WORDS[lang]
    pattern = str(pattern or "")
    title = w["title"]

    def say(text):
        return yours.figure(title, _note(text), alt=text)

    if not pattern:
        return say(w["empty"])
    if len(pattern) > DEMO_LONGEST:
        return say(w["long"].format(k=DEMO_LONGEST))
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return say(w["error"].format(pos=(e.pos or 0) + 1))
    if repeats_a_repeat(pattern):
        return say(w["nested"])
    try:
        total, n, chosen = with_time_limit(lambda: demo_lines(pattern), DEMO_LIMIT)
    except Timeout:
        return say(w["slow"].format(k=DEMO_LIMIT))
    s = _num(n_sentences(), lang)
    if not n:
        return say(w["none"].format(s=s))
    line = w["count"].format(k=_num(total, lang), n=_num(n, lang), s=s)
    rows_ = "".join(f'<div style="{_LINE};border-bottom:1px solid var(--ea-line);padding:0.1rem 0.45rem">'
                    f'{marked(text, [(a, b, _HIT) for a, b in spans])}</div>' for text, spans in chosen)
    notes = [w["note"]] + ([w["zero"]] if rx.fullmatch("") is not None else [])
    return yours.figure(title, f'<div style="{_LINE}">{html.escape(line)}</div>{rows_}' + "".join(_note(t) for t in notes),
                        alt=w["alt"] + " " + line)


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["four-digits", "is-postcode", "clock-times", "which-tool"]

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
    """A short value the way the lessons write it: strings in double quotes, also inside a tuple or a list."""
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, tuple):
        return "(" + ", ".join(_value(p) for p in v) + ("," if len(v) == 1 else "") + ")"
    if isinstance(v, list):
        return "[" + ", ".join(_value(p) for p in v) + "]"
    if isinstance(v, re.Match):
        return f"a match of {_value(v.group())}"
    return repr(v)


_MATCH = {"en": "a match of {g}", "nl": "een match van {g}"}


def _show(v):
    """{got} and {want}: a match of re.search by what it matched, in the language of the lesson."""
    if isinstance(v, re.Match):
        return {lang: _MATCH[lang].format(g=_value(v.group())) for lang in _MATCH}
    return _value(v)


class Named:
    """A value of the lesson that a message shows by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


class _Lazy:
    """A value of the lesson made when a check first needs it, given as a new copy each time, and shown in a message
    by its name."""

    def __init__(self, name: str, make):
        self.name, self.make = name, make

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return self.make()


def _lit(v) -> Named:
    """A value of the check's own, shown in a message as Python, with strings in double quotes."""
    return Named(_value(v), v)


def _sentence(i: int) -> _Lazy:
    """A sentence of Tatoeba as a case: read when a check first needs it, and shown in a message as itself."""
    return _Lazy(_value(sentence(i)), lambda: sentence(i))


SENTENCE_N = _Lazy("SENTENCE", star)
ECLIPSE_N = _Lazy("ECLIPSE", eclipse)


def _value_error(e) -> bool:
    return isinstance(e, ValueError)


def _a_match(got, text) -> bool:
    return isinstance(got, re.Match)


def _maybe_match(got, text) -> bool:
    return got is None or isinstance(got, re.Match)


def _numbers(got, text) -> bool:
    return isinstance(got, list) and bool(got) and all(isinstance(x, int) and not isinstance(x, bool) for x in got)


def _text_pairs(got, text) -> bool:
    return isinstance(got, list) and bool(got) and all(
        isinstance(x, tuple) and len(x) == 2 and all(isinstance(p, str) for p in x) for x in got)


def _whole_matches(got, text) -> bool:
    return isinstance(got, list) and bool(got) and all(isinstance(x, str) for x in got)


# exercise 1: four_digits. Besides the sentence of the star: a phone number, a year outside 1000-2099, a toll-free
# number with hyphens, two years, a share with a decimal comma, a text without numbers, and an empty text.
FOUR_CASES = [
    (SENTENCE_N,),
    (_sentence(PHONE_ID),),
    (_sentence(YEAR_2345_ID),),
    (_sentence(BROCHURE_ID),),
    (_sentence(WAR_ID),),
    (_lit("0,0033%"),),
    (_lit("Geen getal hier."),),
    (_lit(""),),
]
_HEAD_RE = "import re\n\n"
_F_START = _HEAD_RE + "def four_digits(text):\n    ...\n"
_F = _HEAD_RE + "def four_digits(text):\n    return re.findall(r\"\\b\\d{4}\\b\", text)\n"
_F_NO_BOUNDARY = _F.replace("\\b\\d{4}\\b", "\\d{4}")
_F_PLUS = _F.replace("\\b\\d{4}\\b", "\\d+")
_F_NOT_RAW = _F.replace("r\"\\b", "\"\\b")
_F_SEARCH = _F.replace("re.findall", "re.search")
_F_INTS = _HEAD_RE + "def four_digits(text):\n    return [int(x) for x in re.findall(r\"\\b\\d{4}\\b\", text)]\n"

# exercise 2: is_postcode. Besides "1234 AB": the same without a space, small letters, a first digit 0, too few and too
# many digits, too many letters, two spaces, a hyphen, a word before it, and nothing.
POST_CASES = [
    (_lit("1234 AB"),),
    (_lit("1234AB"),),
    (_lit("2514 AT"),),
    (_lit("1234 ab"),),
    (_lit("0123 AB"),),
    (_lit("123 AB"),),
    (_lit("12345 AB"),),
    (_lit("1234 ABC"),),
    (_lit("1234  AB"),),
    (_lit("1234-AB"),),
    (_lit("Postcode: 1234 AB"),),
    (_lit(""),),
]
_P_START = _HEAD_RE + "def is_postcode(s):\n    ...\n"
_P = _HEAD_RE + "def is_postcode(s):\n    return re.search(r\"^[1-9]\\d{3} ?[A-Z]{2}$\", s) is not None\n"
_P_NO_ANCHORS = _P.replace("r\"^[1-9]\\d{3} ?[A-Z]{2}$\"", "r\"[1-9]\\d{3} ?[A-Z]{2}\"")
_P_ONLY_START = _P.replace("[A-Z]{2}$\"", "[A-Z]{2}\"")
_P_ONLY_END = _P.replace("r\"^[1-9]", "r\"[1-9]")
_P_MATCH = _P.replace(" is not None\n", "\n")
_P_ZERO = _P.replace("^[1-9]\\d{3}", "^\\d{4}")
_P_SPACE_NEEDED = _P.replace(" ?[A-Z]", " [A-Z]")
_P_NO_SPACE = _P.replace(" ?[A-Z]", "[A-Z]")
_P_SMALL = _P.replace("[A-Z]{2}", "[A-Za-z]{2}")
_P_FIVE = _P.replace("^[1-9]\\d{3}", "^[1-9]\\d{4}")

# exercise 3: clock_times. Besides the eclipse: a time with a dot, two times in one sentence, a sunset with a zero in
# the minutes, a time after a date with a year, a version number, a Dutch number with a dot for the thousands, an amount
# in dollars with a decimal point, and an empty text.
TIME_CASES = [
    (ECLIPSE_N,),
    (_sentence(TOM_ID),),
    (_sentence(BREAK_ID),),
    (_sentence(SUNSET_ID),),
    (_lit("Op 1 maart 2027 om 15.00 uur."),),
    (_sentence(WINDOWS_ID),),
    (_lit("Een auto van 12.500 euro."),),
    (_lit("Een kaartje kost $ 140.00."),),
    (_lit(""),),
]
_T_START = _HEAD_RE + "def clock_times(text):\n    ...\n"
_T = (_HEAD_RE + "def clock_times(text):\n"
      "    found = re.findall(r\"\\b(\\d{1,2})[.:](\\d\\d)\\b\", text)\n"
      "    return [(int(h), int(m)) for h, m in found]\n")
_T_STRINGS = _HEAD_RE + "def clock_times(text):\n    return re.findall(r\"\\b(\\d{1,2})[.:](\\d\\d)\\b\", text)\n"
_T_WHOLE = _HEAD_RE + "def clock_times(text):\n    return re.findall(r\"\\b\\d{1,2}[.:]\\d\\d\\b\", text)\n"
_T_ONE_GROUP = (_HEAD_RE + "def clock_times(text):\n"
                "    found = re.findall(r\"\\b\\d{1,2}[.:]\\d\\d\\b\", text)\n"
                "    return [int(t) for t in found]\n")
_T_COLON = _T.replace("[.:]", ":")
_T_DOT = _T.replace("[.:]", "\\.")
_T_ANY = _T.replace("[.:]", ".")
_T_NO_END = _T.replace("(\\d\\d)\\b", "(\\d\\d)")
_T_NO_START = _T.replace("r\"\\b(\\d{1,2})", "r\"(\\d{1,2})")
_T_TWO = _T.replace("(\\d{1,2})", "(\\d\\d)")

_RIGHT_FOR = {"en": "Right for {name}.", "nl": "Goed voor {name}."}

EXERCISES = {
    "four-digits": FunctionBox(
        name="four-digits",
        picture=four_picture,
        function="four_digits",
        start=_F_START,
        solution=_F,
        visible=FOUR_CASES[:1],
        hidden=FOUR_CASES[1:],
        first=[_DOTS],
        show=_show,
        mistakes=[
            Mistake("search", matches=_maybe_match,
                    en="`{call}` gives {got}, and it should give {want}: `re.search` gives only the first match, or `None`. Which function of `re` gives every match, as a list?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `re.search` geeft alleen de eerste match, of `None`. Welke functie van `re` geeft elke match, als lijst?"),
            Mistake("ints", matches=_numbers,
                    en="`{call}` gives {got}, and it should give {want}: whole numbers, and the exercise asks for the numbers as `re.findall` finds them, as strings. Give back the list of strings itself.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: hele getallen, en de oefening vraagt om de getallen zoals `re.findall` die vindt, als strings. Geef de lijst met strings zelf terug."),
            Mistake("not-raw", wrong=_F_NOT_RAW,
                    en="`{call}` gives {got}, and it should give {want}: your pattern finds nothing. Without an `r` in front of the quotes, Python reads `\\b` in a string as one character, a backspace, before `re` ever sees the pattern. Write the pattern as a raw string, `r\"...\"`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je patroon vindt niets. Zonder een `r` voor de aanhalingstekens leest Python `\\b` in een string als één teken, een backspace, nog voordat `re` het patroon ziet. Schrijf het patroon als raw string, `r\"...\"`."),
            Mistake("no-boundary", wrong=_F_NO_BOUNDARY,
                    en="`{call}` gives {got}, and it should give {want}: your pattern also takes four digits out of a longer number, such as 2867 out of 28678. What makes sure that no digit comes right before or after the four?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je patroon haalt ook vier cijfers uit een langer getal, zoals 2867 uit 28678. Wat zorgt ervoor dat er geen cijfer direct voor of na de vier staat?"),
            Mistake("plus", wrong=_F_PLUS,
                    en="`{call}` gives {got}, and it should give {want}: `+` takes one digit or more, so your pattern finds numbers of any length. Say exactly how many digits, with a number between curly brackets.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `+` neemt één cijfer of meer, dus je patroon vindt getallen van elke lengte. Zeg precies hoeveel cijfers, met een getal tussen accolades."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back every number of exactly four digits in `text`, with `re.findall` and a pattern written as a raw string.",
                      "nl": "Vervang de `...`: geef elk getal van precies vier cijfers in `text` terug, met `re.findall` en een patroon dat als raw string geschreven is."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your pattern take exactly four digits, with no digit or letter right before or after them?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Neemt je patroon precies vier cijfers, zonder cijfer of letter er direct voor of na?"},
            "almost": {"en": "Right for SENTENCE. But `{call}` gives {got}, and it should give {want}. Does your function work for any text, also a phone number, two years, or no numbers at all?",
                       "nl": "Goed voor SENTENCE. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke tekst, ook een telefoonnummer, twee jaartallen, of helemaal geen getallen?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back. Does every way through it end with `return`, also when there is no match?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt elke weg erdoorheen met `return`, ook als er geen match is?"},
            "correct": {"en": "Your `four_digits` is right in all {n} cases. Look at the picture: every number of four digits in the Dutch sentences of Tatoeba, per hundred years.",
                        "nl": "Je `four_digits` klopt in alle {n} gevallen. Kijk naar de afbeelding: elk getal van vier cijfers in de Nederlandse zinnen van Tatoeba, per honderd jaar."},
        },
    ),
    "is-postcode": FunctionBox(
        name="is-postcode",
        picture=postcode_picture,
        function="is_postcode",
        start=_P_START,
        solution=_P,
        visible=POST_CASES[:1],
        hidden=POST_CASES[1:],
        first=[_DOTS],
        show=_show,
        mistakes=[
            Mistake("match", matches=_a_match,
                    en="`{call}` gives {got}, and it should give {want}: `re.search` gives a match or `None`, not `True` or `False`. Compare what it gives with `None`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `re.search` geeft een match of `None`, geen `True` of `False`. Vergelijk wat het geeft met `None`."),
            Mistake("no-anchors", wrong=_P_NO_ANCHORS,
                    en="`{call}` gives {got}, and it should give {want}: `re.search` finds the pattern anywhere in the text, so a postcode inside a longer text counts too. The whole text has to be the postcode: tie the pattern to the start and to the end.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `re.search` vindt het patroon overal in de tekst, dus een postcode in een langere tekst telt ook. De hele tekst moet de postcode zijn: maak het patroon vast aan het begin en aan het eind."),
            Mistake("only-start", wrong=_P_ONLY_START,
                    en="`{call}` gives {got}, and it should give {want}: your pattern is tied to the start of the text, but not to the end, so letters after the postcode slip through. Tie it to the end too.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je patroon zit vast aan het begin van de tekst, maar niet aan het eind, dus letters na de postcode glippen erdoor. Maak het ook vast aan het eind."),
            Mistake("only-end", wrong=_P_ONLY_END,
                    en="`{call}` gives {got}, and it should give {want}: your pattern is tied to the end of the text, but not to the start, so anything before the postcode slips through. Tie it to the start too.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je patroon zit vast aan het eind van de tekst, maar niet aan het begin, dus alles voor de postcode glipt erdoor. Maak het ook vast aan het begin."),
            Mistake("zero", wrong=_P_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: the first digit of a postcode is not 0. Let the first digit come from 1 to 9, and then three digits more.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het eerste cijfer van een postcode is geen 0. Laat het eerste cijfer van 1 tot en met 9 lopen, en dan nog drie cijfers."),
            Mistake("five", wrong=_P_FIVE,
                    en="`{call}` gives {got}, and it should give {want}: your pattern asks for five digits. The `[1-9]` is already the first digit; how many come after it?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je patroon vraagt om vijf cijfers. De `[1-9]` is al het eerste cijfer; hoeveel komen er nog na?"),
            Mistake("space-needed", wrong=_P_SPACE_NEEDED,
                    en="`{call}` gives {got}, and it should give {want}: your pattern needs the space, and a postcode may also be written without it. Which sign in the table lets the piece before it come zero times or once?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je patroon heeft de spatie nodig, en een postcode mag ook zonder spatie geschreven worden. Welk teken uit de tabel laat het stuk ervoor nul keer of één keer komen?"),
            Mistake("no-space", wrong=_P_NO_SPACE,
                    en="`{call}` gives {got}, and it should give {want}: your pattern has no place for a space between the digits and the letters. A space is a character too, and here it may be there or not.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je patroon heeft geen plaats voor een spatie tussen de cijfers en de letters. Een spatie is ook een teken, en hier mag die er wel of niet staan."),
            Mistake("small", wrong=_P_SMALL,
                    en="`{call}` gives {got}, and it should give {want}: the exercise asks for two capital letters, and your pattern also takes small ones.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de oefening vraagt om twee hoofdletters, en je patroon neemt ook kleine letters."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back `True` when the whole of `s` is a postcode, and `False` otherwise, with `re.search` and a pattern tied to the start and the end.",
                      "nl": "Vervang de `...`: geef `True` terug als heel `s` een postcode is, en anders `False`, met `re.search` en een patroon dat vastzit aan het begin en het eind."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your pattern take four digits, the first not 0, then a space or none, then two capital letters, and nothing before or after?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Neemt je patroon vier cijfers, het eerste geen 0, dan een spatie of geen, dan twee hoofdletters, en niets ervoor of erna?"},
            "almost": {"en": "Right for \"1234 AB\". But `{call}` gives {got}, and it should give {want}. Does your function work for every text someone could type, also one that is almost a postcode?",
                       "nl": "Goed voor \"1234 AB\". Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke tekst die iemand kan typen, ook een die bijna een postcode is?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back. Does every way through it end with `return`, also when there is no match?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt elke weg erdoorheen met `return`, ook als er geen match is?"},
            "correct": {"en": "Your `is_postcode` is right in all {n} cases. Look at the picture: what your function says of texts someone could type in a form.",
                        "nl": "Je `is_postcode` klopt in alle {n} gevallen. Kijk naar de afbeelding: wat je functie zegt van teksten die iemand in een formulier kan typen."},
        },
    ),
    "clock-times": FunctionBox(
        name="clock-times",
        picture=times_picture,
        function="clock_times",
        start=_T_START,
        solution=_T,
        visible=TIME_CASES[:1],
        hidden=TIME_CASES[1:],
        first=[_DOTS],
        show=_show,
        mistakes=[
            Mistake("one-group", error=_value_error,
                    en="With `{call}`, `int` gets a whole time such as \"21:30\", and it can only read digits. Put the hours and the minutes each in a group of their own, so that `re.findall` gives them apart.",
                    nl="Met `{call}` krijgt `int` een hele tijd zoals \"21:30\", en het kan alleen cijfers lezen. Zet de uren en de minuten elk in een eigen groep, zodat `re.findall` de uren en de minuten los geeft."),
            Mistake("strings", matches=_text_pairs,
                    en="`{call}` gives {got}, and it should give {want}: pairs of strings. Turn the hours and the minutes of each pair into whole numbers with `int`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: paren strings. Maak van de uren en de minuten van elk paar hele getallen met `int`."),
            Mistake("whole", matches=_whole_matches,
                    en="`{call}` gives {got}, and it should give {want}: each time as one string. Put the hours and the minutes each in a group of their own, and turn both into whole numbers.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke tijd als één string. Zet de uren en de minuten elk in een eigen groep, en maak van allebei hele getallen."),
            Mistake("colon", wrong=_T_COLON,
                    en="`{call}` gives {got}, and it should give {want}: your pattern takes only a colon between the hours and the minutes, and Dutch text often writes a dot, as in 6.30 uur. Let that place take either one.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je patroon neemt alleen een dubbele punt tussen de uren en de minuten, en Nederlandse tekst schrijft vaak een punt, zoals in 6.30 uur. Laat op die plaats allebei toe."),
            Mistake("dot", wrong=_T_DOT,
                    en="`{call}` gives {got}, and it should give {want}: your pattern takes only a dot between the hours and the minutes, and many sentences write a colon, as in 21:30. Let that place take either one.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je patroon neemt alleen een punt tussen de uren en de minuten, en veel zinnen schrijven een dubbele punt, zoals in 21:30. Laat op die plaats allebei toe."),
            Mistake("any", wrong=_T_ANY,
                    en="`{call}` gives {got}, and it should give {want}: a bare `.` in a pattern stands for any character, so your pattern also cuts the year 2027 into 2 and 27. Which characters may come between the hours and the minutes?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een losse `.` in een patroon staat voor elk teken, dus je patroon knipt ook het jaar 2027 in 2 en 27. Welke tekens mogen er tussen de uren en de minuten staan?"),
            Mistake("two", wrong=_T_TWO,
                    en="`{call}` gives {got}, and it should give {want}: your pattern needs two digits for the hours, so it misses a time such as 6.30. Let the hours have one digit or two.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je patroon heeft twee cijfers nodig voor de uren, dus het mist een tijd zoals 6.30. Laat de uren één of twee cijfers hebben."),
            Mistake("no-end", wrong=_T_NO_END,
                    en="`{call}` gives {got}, and it should give {want}: your pattern also takes a time out of a number such as 12.500, where the dot separates the thousands. What makes sure that no digit follows the minutes?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je patroon haalt ook een tijd uit een getal zoals 12.500, waar de punt de duizendtallen scheidt. Wat zorgt ervoor dat er geen cijfer na de minuten komt?"),
            Mistake("no-start", wrong=_T_NO_START,
                    en="`{call}` gives {got}, and it should give {want}: your pattern also takes a time out of the end of a longer number, such as 40.00 out of 140.00. What makes sure that no digit comes before the hours?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je patroon haalt ook een tijd uit het eind van een langer getal, zoals 40.00 uit 140.00. Wat zorgt ervoor dat er geen cijfer voor de uren staat?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: find the times with `re.findall` and a pattern with a group for the hours and a group for the minutes, and give back each pair as two whole numbers.",
                      "nl": "Vervang de `...`: vind de tijden met `re.findall` en een patroon met een groep voor de uren en een groep voor de minuten, en geef elk paar terug als twee hele getallen."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your pattern take one or two digits, a dot or a colon, and two digits, with no digit right before or after, and does your function turn each pair into whole numbers?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Neemt je patroon één of twee cijfers, een punt of een dubbele punt, en twee cijfers, zonder cijfer er direct voor of na, en maakt je functie van elk paar hele getallen?"},
            "almost": {"en": "Right for ECLIPSE. But `{call}` gives {got}, and it should give {want}. Does your function work for any text, also a time with a dot, a year, or a number with a dot in it?",
                       "nl": "Goed voor ECLIPSE. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke tekst, ook een tijd met een punt, een jaartal, of een getal met een punt erin?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back. Does every way through it end with `return`, also when there is no match?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt elke weg erdoorheen met `return`, ook als er geen match is?"},
            "correct": {"en": "Your `clock_times` is right in all {n} cases. Look at the picture: every clock time in the Dutch sentences of Tatoeba, per hour.",
                        "nl": "Je `clock_times` klopt in alle {n} gevallen. Kijk naar de afbeelding: elke kloktijd in de Nederlandse zinnen van Tatoeba, per uur."},
        },
    ),
    "which-tool": Sort(
        name="which-tool",
        groups={
            "in": {"en": "A plain `in`: one fixed piece of text, the same every time",
                   "nl": "Een gewone `in`: één vast stuk tekst, elke keer hetzelfde"},
            "search": {"en": "`re.search`: is there a match of a pattern anywhere in the text?",
                       "nl": "`re.search`: staat er ergens in de tekst een match van een patroon?"},
            "findall": {"en": "`re.findall`: every match of a pattern, as a list",
                        "nl": "`re.findall`: elke match van een patroon, als lijst"},
            "meaning": {"en": "Beyond a pattern: the answer depends on what the words mean",
                        "nl": "Meer dan een patroon: het antwoord hangt af van wat de woorden betekenen"},
        },
        items={
            "order": {"en": "**Check whether an e-mail contains the order number ORD-77812, exactly as written.**",
                      "nl": "**Kijken of een e-mail het ordernummer ORD-77812 bevat, precies zo geschreven.**"},
            "unsubscribe": {"en": "**Check whether a web page contains the word *Afmelden*, with that capital.**",
                            "nl": "**Kijken of een webpagina het woord *Afmelden* bevat, met die hoofdletter.**"},
            "year": {"en": "**Check whether a sentence holds a number of four digits.**",
                     "nl": "**Kijken of een zin een getal van vier cijfers bevat.**"},
            "address": {"en": "**Check whether a line of an address has a postcode somewhere in it.**",
                        "nl": "**Kijken of er ergens een postcode in een regel van een adres staat.**"},
            "amounts": {"en": "**List the amounts in euros that an invoice names.**",
                        "nl": "**De bedragen in euro's opsommen die een factuur noemt.**"},
            "hashtags": {"en": "**List the hashtags that a post uses.**",
                         "nl": "**De hashtags opsommen die een bericht gebruikt.**"},
            "review": {"en": "**Tell whether a review that says *great, broken again after one day* is positive.**",
                       "nl": "**Zeggen of een review die *top, na één dag alweer kapot* zegt, positief is.**"},
            "hours": {"en": "**Tell whether a customer who writes *Can I come by on Sunday?* asks about opening hours.**",
                      "nl": "**Zeggen of een klant die *Kan ik zondag langskomen?* schrijft, naar openingstijden vraagt.**"},
        },
        solution={"order": "in", "unsubscribe": "in", "year": "search", "address": "search", "amounts": "findall",
                  "hashtags": "findall", "review": "meaning", "hours": "meaning"},
        nudge={
            "order": {"en": "Can the text you look for differ from one e-mail to the next?",
                      "nl": "Kan de tekst die je zoekt per e-mail verschillen?"},
            "unsubscribe": {"en": "Is it a kind of text you look for, or these exact letters?",
                            "nl": "Zoek je een soort tekst, of precies deze letters?"},
            "year": {"en": "Do you want every number back, or only a yes or a no?",
                     "nl": "Wil je elk getal terug, of alleen een ja of een nee?"},
            "address": {"en": "Is the answer a list, or a yes or a no?",
                        "nl": "Is het antwoord een lijst, of een ja of een nee?"},
            "amounts": {"en": "Is one answer of yes or no enough here?",
                        "nl": "Is één antwoord ja of nee hier genoeg?"},
            "hashtags": {"en": "Is every hashtag the same text, and do you want one or all?",
                         "nl": "Is elke hashtag dezelfde tekst, en wil je er één of allemaal?"},
            "review": {"en": "Which word in it would a pattern find, and does that word say what the writer thinks?",
                       "nl": "Welk woord erin zou een patroon vinden, en zegt dat woord wat de schrijver vindt?"},
            "hours": {"en": "Which letters in that question show that it is about the time the shop is open?",
                      "nl": "Welke letters in die vraag laten zien dat het gaat over de tijd dat de winkel open is?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the tasks again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de taken opnieuw uit."},
            "empty": {"en": "Put each bold task under the tool that fits it.",
                      "nl": "Zet elke vetgedrukte taak bij het gereedschap dat erbij past."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item}: this task goes somewhere else.",
                      "nl": "{item}: deze taak hoort ergens anders."},
            "correct": {"en": "All {n} are right. A pattern finds letters, as the keywords of the bike shop did: when the answer depends on what the words mean, a pattern alone is not enough.",
                        "nl": "Alle {n} kloppen. Een patroon vindt letters, net als de trefwoorden van de fietsenwinkel: als het antwoord afhangt van wat de woorden betekenen, is een patroon alleen niet genoeg."},
        },
    ),
}

CASES = [
    ("four-digits", _F_START, "empty", None),
    ("four-digits", _F, "correct", None),
    ("four-digits", _HEAD_RE + "def four_digits(text):\n    return re.findall(r\"\\b[0-9][0-9][0-9][0-9]\\b\", text)\n", "correct", None),
    ("four-digits", _HEAD_RE + "def four_digits(text):\n    found = re.findall(r\"\\b\\d\\d\\d\\d\\b\", text)\n    return found\n", "correct", None),
    ("four-digits", _F_NO_BOUNDARY, "wrong", "no-boundary"),
    ("four-digits", _F_PLUS, "wrong", "plus"),
    ("four-digits", _F_NOT_RAW, "wrong", "not-raw"),
    ("four-digits", _F_SEARCH, "wrong", "search"),
    ("four-digits", _F_INTS, "wrong", "ints"),
    ("four-digits", _F.replace("    return re.findall", "    re.findall"), "wrong", "no-return"),
    ("four-digits", _HEAD_RE + "def four_digits(text):\n    return [\"2019\"]\n", "almost", None),
    ("is-postcode", _P_START, "empty", None),
    ("is-postcode", _P, "correct", None),
    ("is-postcode", _HEAD_RE + "def is_postcode(s):\n    return re.fullmatch(r\"[1-9][0-9]{3} ?[A-Z]{2}\", s) is not None\n", "correct", None),
    ("is-postcode", _HEAD_RE + "def is_postcode(s):\n    if re.search(r\"^[1-9]\\d\\d\\d ?[A-Z][A-Z]$\", s):\n        return True\n    return False\n", "correct", None),
    ("is-postcode", _P_MATCH, "wrong", "match"),
    ("is-postcode", _P_NO_ANCHORS, "wrong", "no-anchors"),
    ("is-postcode", _P_ONLY_START, "wrong", "only-start"),
    ("is-postcode", _P_ONLY_END, "wrong", "only-end"),
    ("is-postcode", _P_ZERO, "wrong", "zero"),
    ("is-postcode", _P_FIVE, "wrong", "five"),
    ("is-postcode", _P_SPACE_NEEDED, "wrong", "space-needed"),
    ("is-postcode", _P_NO_SPACE, "wrong", "no-space"),
    ("is-postcode", _P_SMALL, "wrong", "small"),
    ("is-postcode", _HEAD_RE + "def is_postcode(s):\n    return True\n", "almost", None),
    ("clock-times", _T_START, "empty", None),
    ("clock-times", _T, "correct", None),
    ("clock-times", _HEAD_RE + "def clock_times(text):\n    times = []\n    for h, m in re.findall(r\"\\b(\\d?\\d)[:.](\\d{2})\\b\", text):\n        times.append((int(h), int(m)))\n    return times\n", "correct", None),
    ("clock-times", _T_STRINGS, "wrong", "strings"),
    ("clock-times", _T_WHOLE, "wrong", "whole"),
    ("clock-times", _T_ONE_GROUP, "error", "one-group"),
    ("clock-times", _T_COLON, "wrong", "colon"),
    ("clock-times", _T_DOT, "wrong", "dot"),
    ("clock-times", _T_ANY, "wrong", "any"),
    ("clock-times", _T_TWO, "wrong", "two"),
    ("clock-times", _T_NO_END, "wrong", "no-end"),
    ("clock-times", _T_NO_START, "wrong", "no-start"),
    ("clock-times", _T.replace("    return [", "    [", 1), "wrong", "no-return"),
    ("which-tool", {}, "empty", None),
    ("which-tool", {"order": "in", "unsubscribe": "in", "year": "search", "address": "search", "amounts": "findall",
                    "hashtags": "findall", "review": "meaning", "hours": "meaning"}, "correct", None),
    ("which-tool", {"order": "in", "year": "search"}, "wrong", "unfinished"),
    ("which-tool", {"hours": "in"}, "wrong", "wrong:hours"),
    ("which-tool", {"order": "in", "unsubscribe": "in", "year": "findall", "address": "search", "amounts": "findall",
                    "hashtags": "findall", "review": "meaning", "hours": "meaning"}, "wrong", "wrong:year"),
    ("which-tool", {"order": "search", "unsubscribe": "in", "year": "search", "address": "search", "amounts": "findall",
                    "hashtags": "findall", "review": "meaning", "hours": "meaning"}, "wrong", "wrong:order"),
]
