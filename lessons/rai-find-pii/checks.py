# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-find-pii: finding names, Dutch mobile numbers and citizen service numbers (BSN) in ten
made-up messages to a made-up municipality, and hiding them before the messages go into a training set. The three
functions the learner writes: eleven_test (the check of a BSN: the digits times 9, 8, 7, 6, 5, 4, 3, 2 and -1, added
up, divisible by 11), hide (one word: "[NAME]", "[PHONE]", "[BSN]" or the word itself) and share_found (the share of
a list of words that hide hides); with the pictures of what they give. The course marked by hand which words of the
messages are personal data (TRUTH, and all of them in a row as PII) and which other numbers are in them (OTHER). All
names and numbers are made up."""
import copy
import html
import math
import re

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, run_code, with_time_limit


class Named(list):
    """A list the check calls by the name the lesson gives it, such as PII, so that a message stays short. The
    learner's code gets its own copy, a plain list."""

    def __init__(self, values, name: str):
        super().__init__(values)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(list(self), memo)


WEIGHTS = Named([9, 8, 7, 6, 5, 4, 3, 2, -1], "WEIGHTS")   # the weights of the eleven test, digit by digit

# the course's two helpers, as the lesson shows them
IS_NUMBER = '''def is_number(word):
    # True when word is not empty and every character of it is a digit
    if word == "":
        return False
    for ch in word:
        if ch not in "0123456789":
            return False
    return True
'''
IS_PHONE = '''def is_phone(word):
    # a Dutch mobile number: ten digits, starting with 06, but not with 066 or 067
    if len(word) != 10:
        return False
    if word[:2] != "06":
        return False
    if word[2] == "6":
        return False
    if word[2] == "7":
        return False
    return is_number(word)
'''
is_number = run_code(IS_NUMBER).names["is_number"]
is_phone = run_code(IS_PHONE, {"is_number": is_number}).names["is_phone"]

# the names of the people who wrote to the municipality, as its own records have them
NAMES = Named(["Sanne", "Bakker", "Daan", "Visser", "Noor", "Jansen", "Lotte", "Smit", "Fatima", "Yilmaz", "Milan", "Bos",
               "Thijs", "Meijer", "Eva", "Mulder", "Ruben", "Kok", "Sara", "Peters"], "NAMES")

# ten made-up messages to a made-up municipality; the words that are personal data are the same in both languages
MESSAGES = {
    "en": [
        "Hello, this is Sanne Bakker. I moved last month and my new address is not in your system yet. My citizen "
        "service number is 111222333 and you can reach me on 0612345678.",
        "Daan Visser here. Could someone call me back on 06-23456789 about my parking permit? The permit costs 45 "
        "euros, right?",
        "My BSN is 123456782 and my case number is 202600041. When will I hear something? Regards, Noor Jansen",
        "Good morning, I am Lotte Smit. The waste container has not been emptied for 3 weeks. You can call me on "
        "+31687654321.",
        "For the form: my BSN is 222.333.777. Kind regards, Fatima Yilmaz",
        "My BSN is 111223333, I hope I typed it right. Milan Bos",
        "The resident next door, Kees Dekker, has been parking a van on the sidewalk for weeks. Thijs Meijer, "
        "0698765432",
        "hi this is eva mulder, is my passport ready? please call me on 0611223344",
        "Please send the decision to ruben.kok@example.org and not by post. Thanks, Ruben Kok",
        "Case 202600038 has been open since March 2026. Could you look at it? Regards, Sara Peters",
    ],
    "nl": [
        "Hallo, met Sanne Bakker. Ik ben vorige maand verhuisd en mijn nieuwe adres staat nog niet in uw systeem. Mijn "
        "burgerservicenummer is 111222333 en u kunt mij bereiken op 0612345678.",
        "Met Daan Visser. Kan iemand mij terugbellen op 06-23456789 over mijn parkeervergunning? De vergunning kost "
        "toch 45 euro?",
        "Mijn BSN is 123456782 en mijn zaaknummer is 202600041. Wanneer hoor ik iets? Groet, Noor Jansen",
        "Goedemorgen, ik ben Lotte Smit. De container is al 3 weken niet geleegd. U kunt mij bellen op +31687654321.",
        "Voor het formulier: mijn BSN is 222.333.777. Met vriendelijke groet, Fatima Yilmaz",
        "Mijn BSN is 111223333, ik hoop dat ik het goed heb getypt. Milan Bos",
        "De bewoner naast mij, Kees Dekker, zet al weken een busje op de stoep. Thijs Meijer, 0698765432",
        "hoi met eva mulder, is mijn paspoort klaar? bel me graag op 0611223344",
        "Stuur het besluit graag naar ruben.kok@example.org en niet per post. Bedankt, Ruben Kok",
        "Zaak 202600038 staat sinds maart 2026 open. Kunt u ernaar kijken? Groet, Sara Peters",
    ],
}

# what the course marked by hand: the words of each message that are personal data
TRUTH = [
    ["Sanne", "Bakker", "111222333", "0612345678"],
    ["Daan", "Visser", "06-23456789"],
    ["123456782", "Noor", "Jansen"],
    ["Lotte", "Smit", "+31687654321"],
    ["222.333.777", "Fatima", "Yilmaz"],
    ["111223333", "Milan", "Bos"],
    ["Kees", "Dekker", "Thijs", "Meijer", "0698765432"],
    ["eva", "mulder", "0611223344"],
    ["ruben.kok@example.org", "Ruben", "Kok"],
    ["Sara", "Peters"],
]
PII = Named([w for words in TRUTH for w in words], "PII")
# the other numbers in the messages: no personal data
OTHER = Named(["45", "202600041", "3", "202600038", "2026"], "OTHER")

_EDGE = ".,:;!?()\"'"   # what the course takes off the front and the end of a word before hide sees it


def words_of(message: str) -> list:
    """The words of a message, each as (what comes before, the word itself, what comes after): the punctuation at the
    front and the end comes off, so "Bakker." gives ("", "Bakker", ".")."""
    out = []
    for token in message.split():
        core = token.strip(_EDGE)
        if not core:
            out.append((token, "", ""))
            continue
        start = token.index(core)
        out.append((token[:start], core, token[start + len(core):]))
    return out


def redact(message: str, fn) -> str:
    """The message with every word replaced by what `fn` gives for it, the punctuation put back."""
    return " ".join(before + (fn(core) if core else "") + after for before, core, after in words_of(message))


# --- the course's own functions -----------------------------------------------------------------------------------------

_E_HEAD = ("def eleven_test(number):\n"
           "    # number is a string of nine digits; WEIGHTS holds 9, 8, 7, 6, 5, 4, 3, 2, -1\n"
           "    total = 0\n"
           "    for i in range(9):\n")
_E_START = _E_HEAD + "        total = ...\n    return ...\n"
_E_LOOP = "        total = total + WEIGHTS[i] * int(number[i])\n"
_E = _E_HEAD + _E_LOOP + "    return total % 11 == 0\n"
eleven_test = run_code(_E, {"WEIGHTS": WEIGHTS}).names["eleven_test"]

_H_HEAD = ("def hide(word):\n"
           "    # NAMES, is_phone, is_number and eleven_test are ready to use\n"
           "    if word in NAMES:\n"
           "        return \"[NAME]\"\n")
_H_START = _H_HEAD + "    return word\n"
_H_PHONE = ("    if is_phone(word):\n"
            "        return \"[PHONE]\"\n")
_H_BSN = ("    if len(word) == 9:\n"
          "        if is_number(word):\n"
          "            if eleven_test(word):\n"
          "                return \"[BSN]\"\n")
_H = _H_HEAD + _H_PHONE + _H_BSN + "    return word\n"
_HIDE_NAMES = {"NAMES": NAMES, "is_phone": is_phone, "is_number": is_number, "eleven_test": eleven_test}
hide = run_code(_H, _HIDE_NAMES).names["hide"]

_F_HEAD = ("def share_found(words):\n"
           "    # hide is ready to use: it gives back the word itself when it hides nothing\n"
           "    found = 0\n"
           "    for word in words:\n")
_F_START = _F_HEAD + "        if ...:\n            found = found + 1\n    return ...\n"
_F = _F_HEAD + "        if hide(word) != word:\n            found = found + 1\n    return found / len(words)\n"
_FOUND_NAMES = {"hide": hide}
share_found = run_code(_F, _FOUND_NAMES).names["share_found"]


# --- formatting -------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_BOX = "padding:0.25rem 0.5rem;margin:0.2rem 0;border-radius:4px;"
_PLAIN = "border:1px solid var(--ea-line);color:var(--ea-ink)"
_WRONG = "border:1.5px dashed var(--ea-loss);color:var(--ea-loss-text)"
_CODE = "font-family:var(--ea-font-code);font-size:0.8rem"
_PILL = ("font-family:var(--ea-font-code);font-size:0.78rem;background:var(--ea-inset);border-radius:4px;"
         "padding:0 0.25rem;white-space:nowrap;")
_HIDDEN = _PILL + "border:1px solid var(--ea-line);color:var(--ea-ink-2)"
_WRONGLY = _PILL + "border:1.5px dashed var(--ea-loss);color:var(--ea-loss-text)"
_MISSED = ("color:var(--ea-loss-text);text-decoration:underline;text-decoration-color:var(--ea-loss);"
           "text-decoration-thickness:2px;text-underline-offset:3px")


def number(v: float, lang: str) -> str:
    """A share as the lesson writes it: 0.71875 in English, 0,71875 in Dutch."""
    text = f"{v:.5g}"
    return text.replace(".", ",") if lang == "nl" else text


def _row(cells: list, style: str, widths: list) -> str:
    parts = "".join(f'<div style="flex:{w};min-width:0;overflow-wrap:anywhere">{c}</div>' for c, w in zip(cells, widths))
    return f'<div style="display:flex;gap:0.6rem;align-items:baseline;{_BOX}{style}">{parts}</div>'


def _head(cells: list, widths: list) -> str:
    parts = "".join(f'<div style="flex:{w};min-width:0;overflow-wrap:anywhere">{html.escape(c)}</div>'
                    for c, w in zip(cells, widths))
    return (f'<div style="display:flex;gap:0.6rem;font-weight:600;color:var(--ea-ink);padding:0 0.5rem;'
            f'border-bottom:2px solid var(--ea-ink-3);margin-bottom:0.2rem">{parts}</div>')


def _table(head: list, rows: list, widths: list, width: str = "30rem") -> str:
    return (f'<div style="font-family:var(--ea-font-ui);font-size:0.85rem;max-width:{width};flex:1 1 18rem">'
            f'{_head(head, widths)}{"".join(rows)}</div>')


def _code(text: str) -> str:
    return f'<span style="{_CODE}">{html.escape(text)}</span>'


def _note(parts: list) -> str:
    return f'<div style="{_UI}">{html.escape(" ".join(p for p in parts if p))}</div>'


def _fn(seen):
    mine = seen.mine
    if mine is None:
        return None
    fn = mine.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _shown(v) -> str:
    """A value a function gave back, the way Python shows it, kept short."""
    try:
        text = repr(v)
    except Exception:
        text = "?"
    return text if len(text) <= 60 else text[:57] + "..."


_STOPPED = object()   # what a function gave back when it stopped with an error


def _call(fn, *args):
    try:
        return with_time_limit(lambda: fn(*args))
    except Exception:
        return _STOPPED


# --- the picture under exercise 1: eleven_test ----------------------------------------------------------------------

SHOWN = ["111222333", "111223333", "123456782", "123456789", "202600038"]
GRID_FIRST = 111222000   # the grid draws the 1000 numbers from here on
GRID_COLUMNS = 50

_ELEVEN_WORDS = {
    "en": {"title": "Your eleven_test on five numbers, and on 1,000 in a row",
           "head": ["number", "what it is", "your eleven_test"],
           "what": ["a made-up BSN", "the same, one digit wrong", "another made-up BSN", "the digits 1 to 9",
                    "a case number"],
           "grid": "Each square is one of the 1,000 numbers from 111222000 to 111222999, in order, 50 to a row; a "
                   "filled square is one that your eleven_test lets pass: {k} of the 1,000.",
           "off": "Your eleven_test gives {got} for {n}; the grid draws True and False.",
           "stops": "Your eleven_test stops with an error on {n}: the check above says where.",
           "fine": "In the table, your eleven_test gives the same as the course's.",
           "wrong": "A dashed frame marks a result that differs from the course's."},
    "nl": {"title": "Jouw eleven_test op vijf nummers, en op 1000 achter elkaar",
           "head": ["nummer", "wat het is", "jouw eleven_test"],
           "what": ["een verzonnen BSN", "hetzelfde, één cijfer fout", "nog een verzonnen BSN", "de cijfers 1 tot en met 9",
                    "een zaaknummer"],
           "grid": "Elk vakje is een van de 1000 nummers van 111222000 tot en met 111222999, op volgorde, 50 per rij; "
                   "een gevuld vakje is een nummer dat jouw eleven_test laat slagen: {k} van de 1000.",
           "off": "Jouw eleven_test geeft {got} voor {n}; het rooster tekent True en False.",
           "stops": "Jouw eleven_test stopt met een fout bij {n}: de check hierboven zegt waar.",
           "fine": "In de tabel geeft jouw eleven_test hetzelfde als die van de course.",
           "wrong": "Een gestreept kader wijst een uitkomst aan die verschilt van die van de course."},
}


def _grid(values: list) -> str:
    cell, gap = 8, 1
    rows = math.ceil(len(values) / GRID_COLUMNS)
    width, height = GRID_COLUMNS * (cell + gap), rows * (cell + gap)
    parts = []
    for i, v in enumerate(values):
        row, column = divmod(i, GRID_COLUMNS)
        x, y = column * (cell + gap), row * (cell + gap)
        style = "fill:var(--ea-ink)" if v else "fill:none;stroke:var(--ea-line)"
        parts.append(f'<rect x="{x + 0.5}" y="{y + 0.5}" width="{cell - 1}" height="{cell - 1}" style="{style}"/>')
    return (f'<svg viewBox="0 0 {width} {height}" role="img" aria-hidden="true" '
            f'style="width:100%;max-width:{width}px;height:auto;display:block;margin-top:0.5rem">{"".join(parts)}</svg>')


def eleven_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _ELEVEN_WORDS[seen.lang]
    rows, wrong_any, widths = [], False, [3, 4, 3]
    for n, what in zip(SHOWN, w["what"]):
        got = _call(fn, n)
        wrong = got is _STOPPED or got is not eleven_test(n)
        wrong_any |= wrong
        rows.append(_row([_code(n), html.escape(what), _code("–" if got is _STOPPED else _shown(got))],
                         _WRONG if wrong else _PLAIN, widths))
    notes = [w["wrong"] if wrong_any else w["fine"]]
    body = _table(w["head"], rows, widths)
    values = []
    for i in range(1000):
        n = str(GRID_FIRST + i)
        got = _call(fn, n)
        if got is _STOPPED:
            notes.append(w["stops"].format(n=n))
            break
        if not isinstance(got, bool):
            notes.append(w["off"].format(got=_shown(got), n=n))
            break
        values.append(got)
    if len(values) == 1000:
        body += _grid(values)
        notes.append(w["grid"].format(k=sum(values)))
    return yours.figure(w["title"], body + _note(notes))


# --- the picture under exercise 2: hide, on the ten messages --------------------------------------------------------

_HIDE_WORDS = {
    "en": {"title": "Your hide on the ten messages to the municipality",
           "found": "In gray, what your hide put in place of a word: it hid {f} of the {n} pieces of personal data that "
                    "the course marked by hand.",
           "missed": "Underlined in purple: the {m} it left in the text.",
           "wrong_one": "In a dashed purple frame: 1 word it hid that the course did not mark as personal data.",
           "wrong_many": "In a dashed purple frame: {k} words it hid that the course did not mark as personal data.",
           "off": "Your hide gives {got} for “{word}”; the picture draws text, a string.",
           "stops": "Your hide stops with an error on “{word}”: the check above says where."},
    "nl": {"title": "Jouw hide op de tien berichten aan de gemeente",
           "found": "In grijs wat jouw hide op de plaats van een woord zette: het verborg {f} van de {n} persoonsgegevens "
                    "die de course met de hand aanwees.",
           "missed": "Paars onderstreept: de {m} die het in de tekst liet staan.",
           "wrong_one": "In een gestreept paars kader: 1 woord dat het verborg en dat de course niet als persoonsgegeven "
                        "aanwees.",
           "wrong_many": "In een gestreept paars kader: {k} woorden die het verborg en die de course niet als "
                         "persoonsgegeven aanwees.",
           "off": "Jouw hide geeft {got} voor “{word}”; het plaatje tekent tekst, een string.",
           "stops": "Jouw hide stopt met een fout bij “{word}”: de check hierboven zegt waar."},
}


def marked(fn, lang: str):
    """The ten messages with what `fn` (a hide) does to each word: a list of messages, each a list of
    (before, word, after, what fn gives, kind), kind being "hidden", "missed", "wrongly" or "plain". On a word where
    `fn` stops with an error or gives no string: ("stops" or "off", that word, what it gave) instead."""
    out = []
    for message, truth in zip(MESSAGES[lang], TRUTH):
        line = []
        for before, core, after in words_of(message):
            if not core:
                line.append((before, "", after, "", "plain"))
                continue
            got = _call(fn, core)
            if got is _STOPPED:
                return ("stops", core, None)
            if not isinstance(got, str):
                return ("off", core, got)
            personal = core in truth
            kind = ("hidden" if personal else "wrongly") if got != core else ("missed" if personal else "plain")
            line.append((before, core, after, got, kind))
        out.append(line)
    return out


def _word_html(before: str, core: str, after: str, got: str, kind: str) -> str:
    if kind == "hidden":
        middle = f'<span style="{_HIDDEN}">{html.escape(got)}</span>'
    elif kind == "wrongly":
        middle = f'<span style="{_WRONGLY}">{html.escape(got)}</span>'
    elif kind == "missed":
        middle = f'<span style="{_MISSED}">{html.escape(core)}</span>'
    else:
        middle = html.escape(core)
    return html.escape(before) + middle + html.escape(after)


def _messages_html(lines: list) -> str:
    out = []
    for line in lines:
        text = " ".join(_word_html(*word) for word in line)
        out.append(f'<div style="border-left:2px solid var(--ea-line);padding:0.15rem 0 0.15rem 0.6rem;margin:0.35rem 0;'
                   f'line-height:1.7">{text}</div>')
    return f'<div style="font-family:var(--ea-font-ui);font-size:0.86rem;color:var(--ea-ink)">{"".join(out)}</div>'


def counts(lines: list) -> dict:
    kinds = [word[4] for line in lines for word in line]
    return {k: kinds.count(k) for k in ("hidden", "missed", "wrongly")}


def hide_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _HIDE_WORDS[seen.lang]
    lines = marked(fn, seen.lang)
    if isinstance(lines, tuple):
        what, word, got = lines
        note = w["stops"].format(word=word) if what == "stops" else w["off"].format(got=_shown(got), word=word)
        return yours.figure(w["title"], _note([note]))
    c = counts(lines)
    notes = [w["found"].format(f=c["hidden"], n=len(PII))]
    if c["missed"]:
        notes.append(w["missed"].format(m=c["missed"]))
    if c["wrongly"] == 1:
        notes.append(w["wrong_one"])
    elif c["wrongly"]:
        notes.append(w["wrong_many"].format(k=c["wrongly"]))
    return yours.figure(w["title"], _messages_html(lines) + _note(notes))


# --- the demo: hide on a message of the learner's own ----------------------------------------------------------------

EXAMPLE_MESSAGE = {
    "en": "Hello, this is Eva Mulder. My BSN is 111222333 and my number is 06-12345678.",
    "nl": "Hallo, met Eva Mulder. Mijn BSN is 111222333 en mijn nummer is 06-12345678.",
}

_TRY_WORDS = {
    "en": {"title": "{who} on your message",
           "yours": "Your hide", "course": "The course's hide",
           "said": "In gray, what it put in place of a word: {k} of the {n} words.",
           "off": "{who} gives {got} for “{word}”; the picture draws text, a string.",
           "stops": "{who} stops with an error on “{word}”."},
    "nl": {"title": "{who} op jouw bericht",
           "yours": "Jouw hide", "course": "De hide van de course",
           "said": "In grijs wat het op de plaats van een woord zette: {k} van de {n} woorden.",
           "off": "{who} geeft {got} voor “{word}”; het plaatje tekent tekst, een string.",
           "stops": "{who} stopt met een fout bij “{word}”."},
}


def mine(box_text, correct: bool):
    """The learner's own hide when exercise 2 is right, else None: the demo then uses the course's."""
    if not correct or not isinstance(box_text, str):
        return None
    run = run_code(box_text, dict(_HIDE_NAMES))
    fn = run.names.get("hide") if run.error is None else None
    return fn if callable(fn) else None


def try_picture(message: str, fn, lang: str) -> str:
    """What `fn` (the learner's hide, or None for the course's) does to a message of the learner's own."""
    w = _TRY_WORDS[lang]
    who = w["yours"] if fn is not None else w["course"]
    fn = fn or hide
    words, n, k = [], 0, 0
    for before, core, after in words_of(message):
        if not core:
            words.append(html.escape(before))
            continue
        got = _call(fn, core)
        if got is _STOPPED:
            return yours.figure(w["title"].format(who=who), _note([w["stops"].format(who=who, word=core)]))
        if not isinstance(got, str):
            return yours.figure(w["title"].format(who=who), _note([w["off"].format(who=who, got=_shown(got), word=core)]))
        n += 1
        k += got != core
        words.append(_word_html(before, core, after, got, "hidden" if got != core else "plain"))
    body = (f'<div style="font-family:var(--ea-font-ui);font-size:0.9rem;color:var(--ea-ink);line-height:1.8">'
            f'{" ".join(words)}</div>')
    return yours.figure(w["title"].format(who=who), body + _note([w["said"].format(k=k, n=n)]))


# --- the picture under exercise 3: share_found ----------------------------------------------------------------------

_FOUND_WORDS = {
    "en": {"title": "Your share_found on the personal data and on the other numbers",
           "pii": "share_found(PII)", "other": "share_found(OTHER)",
           "said": "The first bar is what your share_found gives for the {n} pieces of personal data in PII, the second "
                   "what it gives for the {m} other numbers in OTHER.",
           "course": "The course's hide finds {f} of the {n} pieces of personal data, and hides {o} of the {m} other "
                     "numbers by mistake.",
           "off": "Your share_found gives {got} for {which}; the bars draw a number from 0 to 1.",
           "stops": "Your share_found stops with an error on {which}: the check above says where."},
    "nl": {"title": "Jouw share_found op de persoonsgegevens en op de andere getallen",
           "pii": "share_found(PII)", "other": "share_found(OTHER)",
           "said": "De eerste balk is wat jouw share_found geeft voor de {n} persoonsgegevens in PII, de tweede wat het "
                   "geeft voor de {m} andere getallen in OTHER.",
           "course": "De hide van de course vindt {f} van de {n} persoonsgegevens, en verbergt {o} van de {m} andere "
                     "getallen per vergissing.",
           "off": "Jouw share_found geeft {got} voor {which}; de balken tekenen een getal van 0 tot 1.",
           "stops": "Jouw share_found stopt met een fout bij {which}: de check hierboven zegt waar."},
}


def _bar(label: str, value: float, lang: str, style: str) -> str:
    return (f'<div style="display:flex;align-items:center;gap:0.6rem;margin:0.35rem 0;font-family:var(--ea-font-ui);'
            f'font-size:0.85rem;color:var(--ea-ink)"><div style="flex:0 0 9.5rem">{_code(label)}</div>'
            f'<div style="flex:1 1 auto;max-width:18rem;height:0.9rem;border:1px solid var(--ea-line);border-radius:3px">'
            f'<div style="width:{100 * value:.1f}%;height:100%;{style}"></div></div>'
            f'<div style="flex:0 0 4.5rem">{_code(number(value, lang))}</div></div>')


def found_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w, lang = _FOUND_WORDS[seen.lang], seen.lang
    bars = []
    for words, which, style in ((PII, "PII", "background:var(--ea-ink-2)"), (OTHER, "OTHER", "background:var(--ea-loss)")):
        got = _call(fn, list(words))
        if got is _STOPPED:
            return yours.figure(w["title"], _note([w["stops"].format(which=which)]))
        if isinstance(got, bool) or not isinstance(got, (int, float)) or not 0 <= got <= 1:
            return yours.figure(w["title"], _note([w["off"].format(got=_shown(got), which=which)]))
        bars.append(_bar(w["pii"] if which == "PII" else w["other"], got, lang, style))
    found = sum(hide(word) != word for word in PII)
    hidden = sum(hide(word) != word for word in OTHER)
    notes = [w["said"].format(n=len(PII), m=len(OTHER)),
             w["course"].format(f=found, n=len(PII), o=hidden, m=len(OTHER))]
    return yours.figure(w["title"], "".join(bars) + _note(notes))


# --- the exercises ----------------------------------------------------------------------------------------------------

REQUIRED = ["eleven-test", "hide", "share-found", "pass", "anonymous"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# exercise 1: eleven_test
_E_REMAINDER = _E_HEAD + _E_LOOP + "    return total % 11\n"
_E_TOTAL = _E_HEAD + _E_LOOP + "    return total\n"
_E_LAST = _E_HEAD + "        total = WEIGHTS[i] * int(number[i])\n    return total % 11 == 0\n"
_E_INSIDE = _E_HEAD + _E_LOOP + "        return total % 11 == 0\n"
_E_NOT = _E_HEAD + _E_LOOP + "    return total % 11 != 0\n"
_E_STR = _E_HEAD + "        total = total + WEIGHTS[i] * number[i]\n    return total % 11 == 0\n"

# exercise 2: hide
_H_NO_BSN = _H_HEAD + _H_PHONE + "    return word\n"
_H_NO_TEST = _H_HEAD + _H_PHONE + ("    if len(word) == 9:\n"
                                   "        if is_number(word):\n"
                                   "            return \"[BSN]\"\n") + "    return word\n"
_H_NO_NUMBER = _H_HEAD + _H_PHONE + ("    if len(word) == 9:\n"
                                     "        if eleven_test(word):\n"
                                     "            return \"[BSN]\"\n") + "    return word\n"
_H_NO_LENGTH = _H_HEAD + _H_PHONE + ("    if is_number(word):\n"
                                     "        if eleven_test(word):\n"
                                     "            return \"[BSN]\"\n") + "    return word\n"
_H_NO_PHONE = _H_HEAD + _H_BSN + "    return word\n"

# exercise 3: share_found
_F_COUNT = _F_HEAD + "        if hide(word) != word:\n            found = found + 1\n    return found\n"
_F_MISSED = _F_HEAD + "        if hide(word) == word:\n            found = found + 1\n    return found / len(words)\n"
_F_INSIDE = _F_HEAD + "        if hide(word) != word:\n            found = found + 1\n        return found / len(words)\n"


def _int_error(e: BaseException) -> bool:
    return isinstance(e, ValueError) and "invalid literal for int()" in str(e)


def _index_error(e: BaseException) -> bool:
    return isinstance(e, IndexError)


def _str_plus(e: BaseException) -> bool:
    return isinstance(e, TypeError) and "'int' and 'str'" in str(e)


EXERCISES = {
    "eleven-test": FunctionBox(
        name="eleven-test",
        picture=eleven_picture,
        function="eleven_test",
        start=_E_START,
        solution=_E,
        names={"WEIGHTS": WEIGHTS},
        visible=[("111222333",)],
        hidden=[("111223333",), ("123456782",), ("123456789",), ("202600038",), ("202600041",), ("000000000",),
                ("999999999",), ("222333777",), ("112122333",), ("111111110",)],
        first=[_DOTS],
        mistakes=[
            Mistake("str", error=_str_plus,
                    en="With `{call}`, your function stops with an error: Python cannot add a number and a string. `number[i]` is a character, such as `\"3\"`; `int` makes a number of it, which you can multiply by its weight.",
                    nl="Met `{call}` stopt je functie met een fout: Python kan een getal en een string niet optellen. `number[i]` is een teken, zoals `\"3\"`; `int` maakt er een getal van, dat je met zijn factor kunt vermenigvuldigen."),
            Mistake("remainder", wrong=_E_REMAINDER,
                    en="`{call}` gives {got}: the remainder after dividing by 11. The test asks whether that remainder is 0, so give back `True` or `False`: compare it with `0` using `==`.",
                    nl="`{call}` geeft {got}: de rest bij deling door 11. De test vraagt of die rest 0 is, dus geef `True` of `False` terug: vergelijk hem met `0` met `==`."),
            Mistake("total", wrong=_E_TOTAL,
                    en="`{call}` gives {got}: the weighted sum itself. The test asks whether it is divisible by 11: whether `total % 11` is 0. Give back `True` or `False`.",
                    nl="`{call}` geeft {got}: de gewogen som zelf. De test vraagt of die deelbaar is door 11: of `total % 11` gelijk is aan 0. Geef `True` of `False` terug."),
            Mistake("last", wrong=_E_LAST,
                    en="`{call}` gives {got}, and it should give {want}. Each time round the loop, your `total` gets only the last digit times its weight. Add it to the total so far instead, so that after the loop `total` holds all nine.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Elke keer in de loop krijgt je `total` alleen het laatste cijfer keer zijn factor. Tel het juist op bij het totaal tot nu toe, zodat `total` na de loop alle negen bevat."),
            Mistake("inside", wrong=_E_INSIDE,
                    en="`{call}` gives {got}, and it should give {want}. Is your `return` inside the loop? Then the function stops after the first digit. Put it after the loop, at the level of `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Staat je `return` in de loop? Dan stopt de functie na het eerste cijfer. Zet hem na de loop, op de hoogte van `for`."),
            Mistake("not", wrong=_E_NOT,
                    en="`{call}` gives {got}, and it should give {want}: the other way round. A number passes when the remainder after dividing by 11 is 0, so `==`, not `!=`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: precies andersom. Een nummer slaagt als de rest bij deling door 11 gelijk is aan 0, dus `==`, niet `!=`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: in the loop, add each digit times its weight to `total`; after the loop, give back whether `total` is divisible by 11.",
                      "nl": "Vervang de twee `...`: tel in de loop elk cijfer keer zijn factor op bij `total`; geef na de loop terug of `total` deelbaar is door 11."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Multiply digit i by `WEIGHTS[i]`, add all nine up, and give back whether the remainder after dividing by 11 is 0.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Vermenigvuldig cijfer i met `WEIGHTS[i]`, tel alle negen op, en geef terug of de rest bij deling door 11 gelijk is aan 0."},
            "almost": {"en": "Right for `eleven_test(\"111222333\")`. But `{call}` gives {got}, and it should give {want}. Does your function use the `number` it gets, digit by digit?",
                       "nl": "Goed voor `eleven_test(\"111222333\")`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie het `number` dat hij krijgt, cijfer voor cijfer?"},
            "correct": {"en": "Your `eleven_test` is right in all {n} cases. The picture runs it on five numbers, and on 1,000 in a row.",
                        "nl": "Je `eleven_test` klopt in alle {n} gevallen. Het plaatje laat hem lopen op vijf nummers, en op 1000 achter elkaar."},
        },
    ),
    "hide": FunctionBox(
        name="hide",
        picture=hide_picture,
        function="hide",
        start=_H_START,
        solution=_H,
        names=_HIDE_NAMES,
        visible=[("111222333",)],
        hidden=[("0612345678",), ("Sanne",), ("202600041",), ("4250",), ("afternoon",), ("12-34-567",), ("0661234567",),
                ("06-23456789",), ("111223333",), ("hello",), ("",), ("1234567890",)],
        first=[_DOTS],
        mistakes=[
            Mistake("no-number", error=_int_error,
                    en="With `{call}`, your function stops with an error: `eleven_test` got a word of nine characters that are not all digits, and `int` cannot make a number of a letter or a dash. Ask `is_number(word)` first, and call `eleven_test` only when it says `True`.",
                    nl="Met `{call}` stopt je functie met een fout: `eleven_test` kreeg een woord van negen tekens die niet allemaal cijfers zijn, en `int` kan van een letter of een streepje geen getal maken. Vraag eerst `is_number(word)`, en roep `eleven_test` alleen aan als dat `True` zegt."),
            Mistake("no-length", error=_index_error,
                    en="With `{call}`, your function stops with an error: `eleven_test` got a number of fewer than nine digits, and looks for a digit that is not there. Ask `len(word) == 9` first.",
                    nl="Met `{call}` stopt je functie met een fout: `eleven_test` kreeg een getal van minder dan negen cijfers, en zoekt een cijfer dat er niet is. Vraag eerst of `len(word) == 9`."),
            Mistake("no-bsn", wrong=_H_NO_BSN,
                    en="`{call}` gives {got}, and it should give {want}. Your `hide` does not hide citizen service numbers yet: nine digits that pass the eleven test.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je `hide` verbergt nog geen burgerservicenummers: negen cijfers die door de elfproef komen."),
            Mistake("no-test", wrong=_H_NO_TEST,
                    en="`{call}` gives {got}, and it should give {want}. Your `hide` hides every number of nine digits, also one that fails the eleven test and so cannot be a BSN. Ask the eleven test as well.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je `hide` verbergt elk getal van negen cijfers, ook een dat niet door de elfproef komt en dus geen BSN kan zijn. Vraag ook de elfproef."),
            Mistake("no-phone", wrong=_H_NO_PHONE,
                    en="`{call}` gives {got}, and it should give {want}. Your `hide` does not hide mobile numbers yet: ask `is_phone(word)`, and give back `\"[PHONE]\"`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je `hide` verbergt nog geen mobiele nummers: vraag `is_phone(word)`, en geef `\"[PHONE]\"` terug."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Add two rules before `return word`: one for a mobile number, which gives back `\"[PHONE]\"`, and one for nine digits that pass the eleven test, which gives back `\"[BSN]\"`.",
                      "nl": "Voeg vóór `return word` twee regels toe: een voor een mobiel nummer, die `\"[PHONE]\"` teruggeeft, en een voor negen cijfers die door de elfproef komen, die `\"[BSN]\"` teruggeeft."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. A name from `NAMES` gives `\"[NAME]\"`, a mobile number `\"[PHONE]\"`, nine digits that pass the eleven test `\"[BSN]\"`, and every other word the word itself.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Een naam uit `NAMES` geeft `\"[NAME]\"`, een mobiel nummer `\"[PHONE]\"`, negen cijfers die door de elfproef komen `\"[BSN]\"`, en elk ander woord het woord zelf."},
            "almost": {"en": "Right for `hide(\"111222333\")`. But `{call}` gives {got}, and it should give {want}. Which of the rules does this word meet?",
                       "nl": "Goed voor `hide(\"111222333\")`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Aan welke regel voldoet dit woord?"},
            "correct": {"en": "Your `hide` is right in all {n} cases. The picture runs it on every word of the ten messages.",
                        "nl": "Je `hide` klopt in alle {n} gevallen. Het plaatje laat hem lopen op elk woord van de tien berichten."},
        },
    ),
    "share-found": FunctionBox(
        name="share-found",
        picture=found_picture,
        function="share_found",
        start=_F_START,
        solution=_F,
        names=_FOUND_NAMES,
        visible=[(PII,)],
        hidden=[(OTHER,), (["Sanne"],), (["hello", "world"],), (["0612345678", "hello"],), (["111222333", "123456782", "4"],)],
        first=[_DOTS],
        mistakes=[
            Mistake("count", wrong=_F_COUNT,
                    en="`{call}` gives {got}: how many words your filter hides. The share is that count divided by the number of words, `len(words)`.",
                    nl="`{call}` geeft {got}: hoeveel woorden je filter verbergt. Het aandeel is dat aantal gedeeld door het aantal woorden, `len(words)`."),
            Mistake("missed", wrong=_F_MISSED,
                    en="`{call}` gives {got}: the share of words that `hide` leaves as they are. A word is hidden when `hide` gives back something else than the word, so `!=`.",
                    nl="`{call}` geeft {got}: het aandeel woorden dat `hide` laat zoals het is. Een woord is verborgen als `hide` iets anders teruggeeft dan het woord, dus `!=`."),
            Mistake("inside", wrong=_F_INSIDE,
                    en="`{call}` gives {got}, and it should give {want}. Is your `return` inside the loop? Then the function stops after the first word. Put it after the loop, at the level of `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Staat je `return` in de loop? Dan stopt de functie na het eerste woord. Zet hem na de loop, op de hoogte van `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: count a word when `hide` gives back something else than the word; after the loop, give back that count divided by the number of words.",
                      "nl": "Vervang de twee `...`: tel een woord als `hide` iets anders teruggeeft dan het woord; geef na de loop dat aantal gedeeld door het aantal woorden terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Count the words for which `hide` gives back something else than the word, and divide by the number of words.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Tel de woorden waarvoor `hide` iets anders teruggeeft dan het woord, en deel door het aantal woorden."},
            "almost": {"en": "Right for `share_found(PII)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `words` it gets?",
                       "nl": "Goed voor `share_found(PII)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `words` die hij krijgt?"},
            "correct": {"en": "Your `share_found` is right in all {n} cases. The picture shows it on the personal data and on the other numbers.",
                        "nl": "Je `share_found` klopt in alle {n} gevallen. Het plaatje laat hem zien op de persoonsgegevens en op de andere getallen."},
        },
    ),
}

CASES = [
    ("eleven-test", _E_START, "empty", None),
    ("eleven-test", _E, "correct", None),
    ("eleven-test", _E_HEAD + "        total += WEIGHTS[i] * int(number[i])\n    return total % 11 == 0\n", "correct", None),
    ("eleven-test", "def eleven_test(number):\n    total = 0\n    for w, d in zip(WEIGHTS, number):\n        total += w * int(d)\n    return total % 11 == 0\n", "correct", None),
    ("eleven-test", _E_HEAD + _E_LOOP + "    if total % 11 == 0:\n        return True\n    return False\n", "correct", None),
    ("eleven-test", _E_REMAINDER, "wrong", "remainder"),
    ("eleven-test", _E_TOTAL, "wrong", "total"),
    ("eleven-test", _E_LAST, "wrong", "last"),
    ("eleven-test", _E_INSIDE, "wrong", "inside"),
    ("eleven-test", _E_NOT, "wrong", "not"),
    ("eleven-test", _E_STR, "error", "str"),
    ("eleven-test", _E_HEAD + "        total = total + WEIGHTS[i] * int(number[i])\n    return total % 10 == 0\n", "wrong", None),
    ("hide", _H_START, "empty", None),
    ("hide", _H, "correct", None),
    ("hide", _H_HEAD + _H_PHONE + "    if len(word) == 9 and is_number(word) and eleven_test(word):\n        return \"[BSN]\"\n    return word\n", "correct", None),
    ("hide", _H_HEAD + "    if is_number(word):\n        if len(word) == 9:\n            if eleven_test(word):\n                return \"[BSN]\"\n" + _H_PHONE + "    return word\n", "correct", None),
    ("hide", _H_NO_BSN, "wrong", "no-bsn"),
    ("hide", _H_NO_TEST, "wrong", "no-test"),
    ("hide", _H_NO_NUMBER, "error", "no-number"),
    ("hide", _H_NO_LENGTH, "error", "no-length"),
    ("hide", _H_NO_PHONE, "wrong", "no-phone"),
    ("hide", _H.replace("[BSN]", "[bsn]"), "wrong", None),
    ("share-found", _F_START, "empty", None),
    ("share-found", _F, "correct", None),
    ("share-found", "def share_found(words):\n    found = [w for w in words if hide(w) != w]\n    return len(found) / len(words)\n", "correct", None),
    ("share-found", _F_COUNT, "wrong", "count"),
    ("share-found", _F_MISSED, "wrong", "missed"),
    ("share-found", _F_INSIDE, "wrong", "inside"),
    ("share-found", _F_HEAD + "        if hide(word) != word:\n            found = found + 1\n    return found / 32\n", "almost", None),
]
