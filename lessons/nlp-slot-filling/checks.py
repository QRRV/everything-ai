# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-slot-filling: a task-oriented dialogue system that books a train, built the way GUS (Bobrow
and others, 1977) planned a trip by air. A frame has three slots, to, from and date; find_slots fills slots from what the
user says with a regular expression for each slot; fill keeps the frame up to date after each turn; next_slot picks the
first slot that is still empty, so that the system can ask for it.

The commands of the lesson come from MASSIVE 1.1 (data/massive-en-nl.tsv.gz, read with everything_ai.books): the 187
commands of the intent transport_ticket, in English and in Dutch, with their slots marked. The stations and the
conversations are the course's own."""
import html
import importlib.util
import re

from everything_ai import books, paths, yours
from everything_ai.check import FunctionBox, Mistake, Sort

LANGS = ("en", "nl")
_KEY = "nlp-slot-filling"


def _summarize():
    spec = importlib.util.spec_from_file_location("everything_ai_checks_nlp_summarize_for_slot_filling",
                                                  paths.LESSONS / "nlp-summarize" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _summarize()   # the parts of the pictures and the exercises

# --- the frame -----------------------------------------------------------------------------------------------------------

SLOTS = ("to", "from", "date")   # in the order in which the system asks for them, as GUS asked where to first
# The stations the system knows: cities of the English commands of MASSIVE (en-US), and Dutch stations for the Dutch ones.
STATIONS = {
    "en": ["atlanta", "boston", "chicago", "cleveland", "dallas", "denver", "las vegas", "los angeles", "miami",
           "new york", "orlando", "philadelphia", "phoenix", "san diego", "san francisco", "seattle", "washington"],
    "nl": ["alkmaar", "amsterdam", "arnhem", "breda", "delft", "den haag", "eindhoven", "enschede", "groningen",
           "haarlem", "leeuwarden", "leiden", "maastricht", "nijmegen", "rotterdam", "utrecht", "zwolle"],
}
DAYS = {
    "en": ["today", "tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
    "nl": ["vandaag", "morgen", "maandag", "dinsdag", "woensdag", "donderdag", "vrijdag", "zaterdag", "zondag"],
}
# The word in front of a station for each slot; the system also puts it in front of a bare answer to its question.
BEFORE = {"en": {"to": "to", "from": "from", "date": "on"}, "nl": {"to": "naar", "from": "van", "date": "op"}}


def _patterns(lang: str) -> dict:
    names = "|".join(STATIONS[lang])
    days = "|".join(DAYS[lang])
    return {"to": r"\b" + BEFORE[lang]["to"] + r" (" + names + r")\b",
            "from": r"\b" + BEFORE[lang]["from"] + r" (" + names + r")\b",
            "date": r"\b(" + days + r")\b"}


PATTERNS = {lang: _patterns(lang) for lang in LANGS}   # one regular expression for each slot, with one group

QUESTIONS = {   # the questions of the frame for air travel in Jurafsky and Martin's appendix K, Figure K.2
    "en": {"to": "Where are you going?", "from": "From what city are you leaving?",
           "date": "What day would you like to leave?"},
    "nl": {"to": "Waar ga je naartoe?", "from": "Uit welke stad vertrek je?", "date": "Op welke dag wil je vertrekken?"},
}
GREETING = {"en": "Hello. I can book a train ticket for you.", "nl": "Hallo. Ik kan een treinkaartje voor je boeken."}
CONFIRM = {"en": "Shall I book it?", "nl": "Zal ik het kaartje boeken?"}
BOOKED = {"en": "Booked: {trip}. Have a good trip!", "nl": "Geboekt: {trip}. Goede reis!"}
NOTHING = {"en": "I found no station and no day in that.", "nl": "Ik vond daarin geen station en geen dag."}
CHANGE = {"en": "What should I change?", "nl": "Wat moet ik veranderen?"}
YES = {"en": {"yes", "yeah", "yep", "sure", "ok", "okay"}, "nl": {"ja", "jazeker", "prima", "goed", "ok", "oké"}}
NO = {"en": {"no", "nope"}, "nl": {"nee", "neen"}}


def new_frame() -> dict:
    return {slot: None for slot in SLOTS}


# The course's solutions, the same code as the exercises.
def find_slots(text, patterns):
    """For each slot whose regular expression matches in text: what its group matched, the first time."""
    found = {}
    for slot in patterns:
        m = re.search(patterns[slot], text)
        if m is not None:
            found[slot] = m.group(1)
    return found


def fill(frame, found):
    """A new frame: the slots that this turn filled get the new value, the others keep theirs."""
    new = {}
    for slot in frame:
        if slot in found:
            new[slot] = found[slot]
        else:
            new[slot] = frame[slot]
    return new


def next_slot(frame):
    """The first slot that is still empty, or None when every slot is filled."""
    for slot in frame:
        if frame[slot] is None:
            return slot
    return None


def shown(slot: str, value, lang: str) -> str:
    """A value the way the system says it: a station with capitals (Den Haag), an English day with a capital."""
    if not isinstance(value, str):
        return S._short(value)
    if slot == "date":
        return value.capitalize() if lang == "en" and value not in ("today", "tomorrow") else value
    return " ".join(w.capitalize() for w in value.split())


def trip(frame: dict, lang: str) -> str:
    """The frame as a phrase: a train from New York to Boston on Friday; op vrijdag een trein van Groningen naar Utrecht."""
    lang = lang if lang in LANGS else "en"
    get = frame.get if isinstance(frame, dict) else (lambda k: None)
    date = get("date")
    places = "".join(f" {BEFORE[lang][slot]} {shown(slot, get(slot), lang)}" for slot in ("from", "to") if get(slot))
    if lang == "en":
        when = "" if not date else (" " + shown("date", date, lang) if date in ("today", "tomorrow")
                                    else " on " + shown("date", date, lang))
        return "a train" + places + when
    when = "" if not date else (shown("date", date, lang) + " " if date in ("vandaag", "morgen")
                                else "op " + shown("date", date, lang) + " ")
    return when + "een trein" + places


def _sentence(phrase: str) -> str:
    return phrase[:1].upper() + phrase[1:] + "."


def _first_word(text_: str) -> str:
    ws = re.findall(r"[^\W\d_]+", text_.lower())
    return ws[0] if ws else ""


class Turn:
    """One turn of the user and the system's reply: what was said, what was found (and whether it was found by
    reading the line as an answer to the question asked), the frame after it, and the reply with how it came about:
    question, confirm, booked, change or nothing."""

    def __init__(self, said, found, answered, frame, asked, reply, how):
        self.said, self.found, self.answered, self.frame = said, found, answered, frame
        self.asked, self.reply, self.how = asked, reply, how


def talk(lines, lang: str, find_slots=find_slots, fill=fill, next_slot=next_slot):
    """The system: its greeting, and a Turn for each line the user says. After a booking, the next line starts a new
    frame."""
    lang = lang if lang in LANGS else "en"
    patterns = PATTERNS[lang]
    frame = new_frame()
    asked = next_slot(frame)
    if asked in SLOTS:
        greeting = GREETING[lang] + " " + QUESTIONS[lang][asked]
    else:   # a next_slot that finds no empty slot in a new frame: the system thinks it is done
        asked = "confirm"
        greeting = GREETING[lang] + " " + _sentence(trip(frame, lang)) + " " + CONFIRM[lang]
    turns = []
    for said in lines:
        text_ = said.lower()
        found, answered = find_slots(text_, dict(patterns)), False
        if not found and asked in SLOTS:
            found = find_slots(BEFORE[lang][asked] + " " + text_, dict(patterns))
            answered = bool(found)
        word = _first_word(text_)
        if asked == "confirm" and not found and word in YES[lang]:
            turns.append(Turn(said, found, answered, frame, asked, BOOKED[lang].format(trip=trip(frame, lang)), "booked"))
            frame = new_frame()
            asked = next_slot(frame)
            continue
        if asked == "confirm" and not found and word in NO[lang]:
            turns.append(Turn(said, found, answered, frame, asked, CHANGE[lang], "change"))
            continue
        if not found:
            again = QUESTIONS[lang][asked] if asked in SLOTS else _sentence(trip(frame, lang)) + " " + CONFIRM[lang]
            turns.append(Turn(said, found, answered, frame, asked, NOTHING[lang] + " " + again, "nothing"))
            continue
        frame = fill(frame, found)
        slot = next_slot(frame)
        if slot is None:
            asked = "confirm"
            turns.append(Turn(said, found, answered, frame, asked, _sentence(trip(frame, lang)) + " " + CONFIRM[lang],
                              "confirm"))
        else:
            asked = slot
            turns.append(Turn(said, found, answered, frame, asked,
                              _sentence(trip(frame, lang)) + " " + QUESTIONS[lang].get(slot, str(slot)), "question"))
    return greeting, turns


# The course's conversation: two slots at once, a bare answer, a correction and a yes.
CONVERSATION = {
    "en": ["I want to go to Boston on Friday.", "New York.", "No, on Saturday.", "Yes, please."],
    "nl": ["Ik wil vrijdag naar Utrecht.", "Groningen.", "Nee, op zaterdag.", "Ja, graag."],
}


def run(lang: str, **steps):
    """The course's conversation with the system, with any of its steps replaced."""
    lang = lang if lang in LANGS else "en"
    return talk(CONVERSATION[lang], lang, **steps)


# --- MASSIVE -------------------------------------------------------------------------------------------------------------

_MARK = re.compile(r"\[(\w+) : ([^\]]+)\]")


def ticket_commands() -> list:
    """The 187 commands of MASSIVE with the intent transport_ticket, as (id, en, nl, en_slots, nl_slots)."""
    return [(r[0], r[5], r[6], r[7], r[8]) for r in books.assistant_commands() if r[3] == "transport_ticket"]


def marks(marked: str) -> list:
    """The slots MASSIVE marks in a command, as (type, words): [("place_name", "boston"), ("date", "tomorrow")]."""
    return [(m.group(1), m.group(2)) for m in _MARK.finditer(marked)]


def frame_counts(lang: str = "en") -> dict:
    """How many ticket commands mark how many places (0, 1, or 2 and more) with or without a date: {(places, date): n}."""
    col = 3 if lang == "en" else 4
    out = {}
    for c in ticket_commands():
        types = [t for t, _ in marks(c[col])]
        key = (min(types.count("place_name"), 2), "date" in types)
        out[key] = out.get(key, 0) + 1
    return out


def station_commands(lang: str) -> list:
    """The ticket commands whose marked places are all stations of the lesson, as (id, text, marked, places, dates),
    in the order of the file; only those a lesson shows as an example."""
    lang = lang if lang in LANGS else "en"
    shown_ids = {r[0] for r in books.assistant_commands() if r[4] == 1}
    out = []
    for c in ticket_commands():
        text_, marked = (c[1], c[3]) if lang == "en" else (c[2], c[4])
        ms = marks(marked)
        places = [v for t, v in ms if t == "place_name"]
        if c[0] in shown_ids and places and all(p in STATIONS[lang] for p in places):
            out.append((c[0], text_, marked, places, [v for t, v in ms if t == "date"]))
    return out


def score(found: dict, places: list, dates: list) -> tuple:
    """What a found frame gets right against MASSIVE's marks: (right slots, wrong slots, missed marks). A station in to or
    from is right when MASSIVE marks it as a place, a day in date when MASSIVE marks it as a date."""
    places, dates = list(places), list(dates)
    right, wrong = [], []
    for slot in SLOTS:
        if slot not in found:
            continue
        pool = dates if slot == "date" else places
        if found[slot] in pool:
            pool.remove(found[slot])
            right.append(slot)
        else:
            wrong.append(slot)
    return right, wrong, places + dates


# --- the pictures --------------------------------------------------------------------------------------------------------

_UI, _LABEL, _CELL, _HEAD, _NUMBER = S._UI, S._LABEL, S._CELL, S._HEAD, S._NUMBER
_DATA, _FAINT, _OUT, _PRED, _LOSS = S._DATA, S._FAINT, S._OUT, S._PRED, S._LOSS
_CHIP, _CHIP_PRED, _CHIP_DATA, _CHIP_LOSS = S._CHIP, S._CHIP_PRED, S._CHIP_DATA, S._CHIP_LOSS
_note, _colors, _label, _num = S._note, S._colors, S._label, S._num
_cell, _table = S._cell, S._table
_fn, _call, _empty, _SLOW, _STOPPED = S._fn, S._call, S._empty, S._SLOW, S._STOPPED
_WHY_EMPTY = S._WHY_EMPTY
_MONO = "font-family:var(--ea-font-code);font-size:0.8rem"
_DASH = f'<span style="{_UI}">–</span>'


def _chip(text_: str, style: str = _CHIP_PRED) -> str:
    return f'<span style="{style}">{html.escape(text_)}</span>'


def _code(text_: str) -> str:
    return f'<code style="{_MONO}">{html.escape(text_)}</code>'


def _short_pattern(pattern: str) -> str:
    """A pattern with its long list of stations cut short: \\bto (atlanta|boston|…|washington)\\b."""
    parts = pattern.split("|")
    if len(parts) <= 4:
        return pattern
    return "|".join(parts[:2]) + "|…|" + parts[-1]


SLOT_NAMES = {"en": {"to": "to", "from": "from", "date": "date"}, "nl": {"to": "naar", "from": "van", "date": "datum"}}


def _slot_chips(found, lang: str, want=None) -> str:
    """The slots of a frame or of what a turn found as chips, `naar: utrecht`, with the values as the frame holds them;
    blue, or purple where it differs from `want`; an empty slot as a dash."""
    if not isinstance(found, dict):
        return _chip(S._short(found), _CHIP_LOSS)
    out = []
    for slot in SLOTS:
        if slot in found and found[slot] is not None:
            bad = want is not None and (not isinstance(want, dict) or want.get(slot) != found[slot])
            out.append(_chip(f"{SLOT_NAMES[lang][slot]}: {found[slot]}", _CHIP_LOSS if bad else _CHIP_PRED))
        elif want is not None and isinstance(want, dict) and want.get(slot) is not None:
            out.append(_chip(f"{SLOT_NAMES[lang][slot]}: –", _CHIP_LOSS))
    return "".join(out) or _DASH


_IDEA_TEXT = {
    "en": {"title": "One turn of the system, step by step",
           "frame": "The frame of this lesson, with the question of each slot. Before turn 1, every slot is empty:",
           "head": ["Slot", "Question", "Value"],
           "asked": "The system starts with its greeting and the question of the first empty slot:",
           "said": "Turn 1 of the user:",
           "find": "1. Slot filling: each slot has a regular expression with one group. What its first match puts in the group is the value:",
           "find_head": ["Slot", "Regular expression", "Value"],
           "frame_after": "2. The frame after the turn: the values found go into their slots, and the other slots keep what they had:",
           "next": "3. The first empty slot is {slot}. The system says what it understood, and asks the question of that slot:",
           "gray": "Gray: what the user said, and the questions and regular expressions of the frame.",
           "blue": "Blue: what the system makes of it: the values, the frame and the reply."},
    "nl": {"title": "Eén beurt van het systeem, stap voor stap",
           "frame": "Het frame van deze les, met de vraag van elk slot. Bij het begin is elk slot leeg:",
           "head": ["Slot", "Vraag", "Waarde"],
           "asked": "Het systeem begint met een groet en de vraag van het eerste lege slot:",
           "said": "Beurt 1 van de gebruiker:",
           "find": "1. Slot filling: elk slot heeft een reguliere expressie met één groep. Wat de eerste match in de groep zet, is de waarde:",
           "find_head": ["Slot", "Reguliere expressie", "Waarde"],
           "frame_after": "2. Het frame na de beurt: elke gevonden waarde gaat in het eigen slot, en de andere slots houden wat erin stond:",
           "next": "3. Het eerste lege slot is {slot}. Het systeem zegt wat het begreep, en stelt de vraag van dat slot:",
           "gray": "Grijs: wat de gebruiker zei, en de vragen en reguliere expressies van het frame.",
           "blue": "Blauw: wat het systeem ervan maakt: de waarden, het frame en het antwoord."},
}


def _frame_table(frame: dict, lang: str, w: dict, questions: bool = True) -> str:
    rows = []
    for slot in SLOTS:
        value = frame.get(slot)
        cell = _chip(value) if value is not None else _DASH
        rows.append("<tr>" + _cell(html.escape(SLOT_NAMES[lang][slot]), _FAINT)
                    + (_cell(html.escape(QUESTIONS[lang][slot]), _DATA) if questions else "") + _cell(cell) + "</tr>")
    head = w["head"] if questions else [w["head"][0], w["head"][2]]
    return _table(rows, head)


def idea_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang]
    greeting, turns = run(lang)
    first = turns[0]
    body = _label(w["frame"]) + _frame_table(new_frame(), lang, w)
    body += _label(w["asked"]) + f'<div style="{_PRED}">{html.escape(greeting)}</div>'
    body += _label(w["said"]) + f'<div style="{_DATA}">{html.escape(first.said)}</div>'
    rows = []
    for slot in SLOTS:
        value = first.found.get(slot)
        rows.append("<tr>" + _cell(html.escape(SLOT_NAMES[lang][slot]), _FAINT)
                    + _cell(_code(_short_pattern(PATTERNS[lang][slot])), _DATA)
                    + _cell(_chip(value) if value is not None else _DASH) + "</tr>")
    body += _label(w["find"]) + _table(rows, w["find_head"])
    body += _label(w["frame_after"]) + _frame_table(first.frame, lang, w, questions=False)
    body += _label(w["next"].format(slot=SLOT_NAMES[lang][first.asked]))
    body += f'<div style="{_PRED}">{html.escape(first.reply)}</div>'
    body += _colors(w, body)
    return yours.figure(w["title"], body, alt=w["title"])


class _Odd(Exception):
    """A learner's function gave something a step of the system cannot use."""


def _guarded(name: str, fn):
    """The learner's function as a step of the system, stopping with _Odd when it gives what the step cannot use."""
    if name == "find_slots":
        def step(text_, patterns):
            out = fn(text_, dict(patterns))
            if not isinstance(out, dict) or not all(k in SLOTS and isinstance(v, str) for k, v in out.items()):
                raise _Odd
            return out
    elif name == "fill":
        def step(frame, found):
            out = fn(dict(frame), dict(found))
            if not isinstance(out, dict) or not all(k in SLOTS and (v is None or isinstance(v, str)) for k, v in out.items()):
                raise _Odd
            return out
    else:
        def step(frame):
            out = fn(dict(frame))
            if out is not None and out not in SLOTS:
                raise _Odd
            return out
    return {name: step}


_FIND_TEXT = {
    "en": {"title": "Your find_slots on ticket commands of MASSIVE",
           "kind": "a dictionary from slots to strings",
           "head": ["Command", "Your slots", "Marked, not found"],
           "summary": "Your find_slots ran on the {n} ticket commands of MASSIVE whose places are all stations of the lesson. Of the {marks} places and dates that MASSIVE marks in them, it finds {right}, and it fills {wrong} slot{s} with a value that MASSIVE does not mark there.",
           "table": "The table shows {shown} of the commands: first a few where everything was found, then every command with a value missed or wrong.",
           "table_none": "The table shows every command with a value missed or wrong: {shown} commands.",
           "table_all": "The table shows {shown} of the commands, in which everything was found.",
           "gray": "Gray: the commands, and what MASSIVE marks in them that your find_slots does not find.",
           "blue": "Blue: a value your find_slots fills that MASSIVE marks too.",
           "purple": "Purple: a value MASSIVE does not mark there."},
    "nl": {"title": "Je find_slots op kaartjesopdrachten van MASSIVE",
           "kind": "een dictionary van slots naar strings",
           "head": ["Opdracht", "Je slots", "Gemarkeerd, niet gevonden"],
           "summary": "Je find_slots draaide op de {n} kaartjesopdrachten van MASSIVE met alleen stations van de les als plaats. Van de {marks} plaatsen en datums die MASSIVE erin markeert, vindt de functie er {right}, en vult de functie {wrong} slot{s} met een waarde die MASSIVE daar niet markeert.",
           "table": "De tabel toont {shown} van de opdrachten: eerst een paar waarin alles gevonden is, daarna elke opdracht met een gemiste of verkeerde waarde.",
           "table_none": "De tabel toont elke opdracht met een gemiste of verkeerde waarde: {shown} opdrachten.",
           "table_all": "De tabel toont {shown} van de opdrachten, waarin alles gevonden is.",
           "gray": "Grijs: de opdrachten, en wat MASSIVE erin markeert maar je find_slots niet vindt.",
           "blue": "Blauw: een waarde van je find_slots die MASSIVE ook markeert.",
           "purple": "Paars: een waarde die MASSIVE daar niet markeert."},
}
FIND_ALL_RIGHT = 3   # commands without a miss that the picture shows first


def find_measure(lang: str, find=find_slots) -> list:
    """For each station command: (text, found, right, wrong, missed)."""
    out = []
    for _, text_, _, places, dates in station_commands(lang):
        found = find(text_, dict(PATTERNS[lang]))
        right, wrong, missed = score(found, places, dates)
        out.append((text_, found, right, wrong, missed))
    return out


def find_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _FIND_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")

    def make():
        try:
            return find_measure(lang, _guarded("find_slots", fn)["find_slots"])
        except _Odd:
            return None

    got = _call(make)
    if got is _SLOW:
        return _empty(w, "slow")
    if got is _STOPPED:
        return _empty(w, "stopped")
    if got is None:
        return _empty(w, "odd")
    clean = [g for g in got if not g[3] and not g[4]][:FIND_ALL_RIGHT]
    rest = [g for g in got if g[3] or g[4]]
    rows = []
    for text_, found, right, wrong, missed in clean + rest:
        chips = "".join(_chip(f"{SLOT_NAMES[lang][s]}: {found[s]}", _CHIP_LOSS if s in wrong else _CHIP_PRED)
                        for s in SLOTS if s in found) or _DASH
        lost = "".join(_chip(m, _CHIP_DATA) for m in missed) or _DASH
        rows.append("<tr>" + _cell(html.escape(text_), _DATA) + _cell(chips) + _cell(lost) + "</tr>")
    right = sum(len(g[2]) for g in got)
    wrong = sum(len(g[3]) for g in got)
    marks_ = sum(len(g[2]) + len(g[4]) for g in got)
    table = w["table"] if clean and rest else w["table_all"] if clean else w["table_none"]
    body = _note(w["summary"].format(n=_num(len(got), lang), marks=_num(marks_, lang), right=_num(right, lang),
                                     wrong=_num(wrong, lang), s="" if wrong == 1 else "s")
                 + " " + table.format(shown=_num(len(rows), lang)))
    body += _table(rows, w["head"])
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


_RUN_TEXT = {
    "en": {"head": ["", "User", "Frame after the turn", "System"],
           "same": "With your {fn}, the system gives the same reply as the course's system in {k} of its {n} replies, the greeting included.",
           "was": "Course: {r}",
           "start": "start",
           "gray": "Gray: what the user says, and what the course's system says where yours says something else.",
           "blue": "Blue: a slot or a reply that is the same as in the course's system.",
           "purple": "Purple: a slot or a reply that differs from the course's system."},
    "nl": {"head": ["", "Gebruiker", "Frame na de beurt", "Systeem"],
           "same": "Met je {fn} geeft het systeem hetzelfde antwoord als het systeem van de course bij {k} van de {n} antwoorden, de groet meegerekend.",
           "was": "De course: {r}",
           "start": "begin",
           "gray": "Grijs: wat de gebruiker zegt, en wat het systeem van de course zegt waar jouw systeem iets anders zegt.",
           "blue": "Blauw: een slot of een antwoord dat hetzelfde is als bij het systeem van de course.",
           "purple": "Paars: een slot of een antwoord dat anders is dan bij het systeem van de course."},
}
_FILL_TEXT = {
    "en": {"title": "The course's conversation, with your fill", "kind": "a frame: a dictionary from slots to values"},
    "nl": {"title": "Het gesprek van de course, met je fill", "kind": "een frame: een dictionary van slots naar waarden"},
}
_NEXT_TEXT = {
    "en": {"title": "The course's conversation, with your next_slot", "kind": "a slot of the frame, or None"},
    "nl": {"title": "Het gesprek van de course, met je next_slot", "kind": "een slot van het frame, of None"},
}


def _conversation_picture(seen, name: str, w: dict) -> str:
    """The course's conversation with the learner's function as one step of the system, against the course's system."""
    lang = seen.lang if seen.lang in LANGS else "en"
    w = w[lang] | _RUN_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")

    def make():
        try:
            return run(lang, **_guarded(name, fn))
        except _Odd:
            return None

    mine = _call(make)
    if mine is _SLOW:
        return _empty(w, "slow")
    if mine is _STOPPED:
        return _empty(w, "stopped")
    if mine is None:
        return _empty(w, "odd")
    (greeting, got), (want_greeting, want) = mine, run(lang)

    def said(text_, ref):
        good = text_ == ref
        out = f'<div style="{_PRED if good else _LOSS}">{html.escape(text_)}</div>'
        return out + ("" if good else f'<div style="{_UI}">{html.escape(w["was"].format(r=ref))}</div>'), good

    first, good_first = said(greeting, want_greeting)
    rows = ["<tr>" + _cell(html.escape(w["start"]), _FAINT) + _cell(_DASH) + _cell(_DASH) + _cell(first) + "</tr>"]
    same = good_first
    for i, (g, ref) in enumerate(zip(got, want), 1):
        reply, good = said(g.reply, ref.reply)
        same += good
        rows.append("<tr>" + _cell(_num(i, lang), _FAINT, _NUMBER) + _cell(html.escape(g.said), _DATA)
                    + _cell(_slot_chips(g.frame, lang, want=ref.frame)) + _cell(reply) + "</tr>")
    body = _table(rows, w["head"], (0,)) + _note(w["same"].format(fn=name, k=_num(same, lang), n=_num(len(want) + 1, lang)))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


def fill_picture(seen) -> str:
    return _conversation_picture(seen, "fill", _FILL_TEXT)


def next_picture(seen) -> str:
    return _conversation_picture(seen, "next_slot", _NEXT_TEXT)


_TRY_TEXT = {
    "en": {"title": "Your conversation with the system",
           "head": ["", "You", "Frame after the turn", "System"],
           "answer": "read as an answer to: {q}",
           "empty": "Type a line in the box, and the system answers it.",
           "more": "The system answers the first {n} lines.",
           "start": "start",
           "gray": "Gray: what you typed, and how the system read a line that only made sense as an answer.",
           "blue": "Blue: what the system makes of it: the slots of the frame and its replies."},
    "nl": {"title": "Je gesprek met het systeem",
           "head": ["", "Jij", "Frame na de beurt", "Systeem"],
           "answer": "gelezen als antwoord op: {q}",
           "empty": "Typ een zin in het vak, dan antwoordt het systeem.",
           "more": "Het systeem antwoordt op de eerste {n} zinnen.",
           "start": "begin",
           "gray": "Grijs: wat je typte, en hoe het systeem een zin las die alleen als antwoord iets betekende.",
           "blue": "Blauw: wat het systeem ervan maakt: de slots van het frame en de antwoorden."},
}
TRY_LINES = 12      # lines of the try-it box that the system answers
TRY_CHARS = 300     # characters of a line that the system reads
TRY_START = {
    "en": "I need a train ticket to Chicago.\nDenver.\nTomorrow.\nNo.\nTo Seattle.\nYes.",
    "nl": "Ik wil een kaartje naar Amsterdam.\nZwolle.\nMorgen.\nNee.\nNaar Leiden.\nJa.",
}


def try_picture(lang: str, text_: str = "") -> str:
    lang = lang if lang in LANGS else "en"
    w = _TRY_TEXT[lang]
    lines = [line.strip()[:TRY_CHARS] for line in (text_ or "").splitlines() if line.strip()]
    if not lines:
        return yours.figure(w["title"], _note(w["empty"]), alt=w["empty"])
    greeting, turns = talk(lines[:TRY_LINES], lang)
    rows = ["<tr>" + _cell(html.escape(w["start"]), _FAINT) + _cell(_DASH) + _cell(_DASH)
            + _cell(f'<div style="{_OUT}">{html.escape(greeting)}</div>') + "</tr>"]
    asked = SLOTS[0]
    for i, t in enumerate(turns, 1):
        you = f'<div style="{_DATA}">{html.escape(t.said)}</div>'
        if t.answered:
            you += f'<div style="{_UI}">{html.escape(w["answer"].format(q=QUESTIONS[lang][asked]))}</div>'
        rows.append("<tr>" + _cell(_num(i, lang), _FAINT, _NUMBER) + _cell(you) + _cell(_slot_chips(t.frame, lang))
                    + _cell(f'<div style="{_OUT}">{html.escape(t.reply)}</div>') + "</tr>")
        asked = SLOTS[0] if t.how == "booked" else t.asked
    body = _table(rows, w["head"], (0,))
    if len(lines) > TRY_LINES:
        body += _note(w["more"].format(n=_num(TRY_LINES, lang)))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------------

_Val = S._Val
_DOTS = S._DOTS
_SAME_TEXTS = S._SAME_TEXTS


def _show(v):
    return S._show(v)


def _written(v):
    return S._written(v)


def _has_match(got, text_) -> bool:
    return isinstance(got, dict) and any(isinstance(v, re.Match) for v in got.values())


def _has_list(got, text_) -> bool:
    return isinstance(got, dict) and any(isinstance(v, list) for v in got.values())


def _has_none(got, text_) -> bool:
    return isinstance(got, dict) and any(v is None for v in got.values())


_FS_TOP = "import re\n\n\ndef find_slots(text, patterns):\n    found = {}\n    for slot in patterns:\n" \
          "        m = re.search(patterns[slot], text)\n"
_FS_HEAD = _FS_TOP + "        if m is not None:\n"
_FS_RET = "    return found\n"
_FS = _FS_HEAD + "            found[slot] = m.group(1)\n" + _FS_RET
_FS_START = _FS_TOP + "        ...\n" + _FS_RET
_FS_NO_IF = _FS_TOP + "        found[slot] = m.group(1)\n" + _FS_RET
_FS_WHOLE = _FS_HEAD + "            found[slot] = m.group()\n" + _FS_RET
_FS_MATCH = _FS_HEAD + "            found[slot] = m\n" + _FS_RET
_FS_NONE = _FS_HEAD + "            found[slot] = m.group(1)\n        else:\n            found[slot] = None\n" + _FS_RET
_FS_ALWAYS = ("import re\n\n\ndef find_slots(text, patterns):\n    found = {}\n    for slot in patterns:\n"
              "        found[slot] = re.search(patterns[slot], text).group(1)\n" + _FS_RET)
_FS_ALL = ("import re\n\n\ndef find_slots(text, patterns):\n    found = {}\n    for slot in patterns:\n"
           "        m = re.findall(patterns[slot], text)\n        if len(m) > 0:\n            found[slot] = m\n" + _FS_RET)

_FI_HEAD = "def fill(frame, found):\n    new = {}\n    for slot in frame:\n"
_FI_IF = "        if slot in found:\n            new[slot] = found[slot]\n        else:\n"
_FI_RET = "    return new\n"
_FI = _FI_HEAD + _FI_IF + "            new[slot] = frame[slot]\n" + _FI_RET
_FI_START = _FI_HEAD + _FI_IF + "            ...\n" + _FI_RET
_FI_FORGET = _FI_HEAD + _FI_IF + "            new[slot] = None\n" + _FI_RET
_FI_MISSING = _FI_HEAD + _FI_IF + "            pass\n" + _FI_RET
_FI_KEYS = _FI_HEAD + "        new[slot] = found[slot]\n" + _FI_RET
_FI_OLD = (_FI_HEAD + "        if slot in found and frame[slot] is None:\n            new[slot] = found[slot]\n        else:\n"
           "            new[slot] = frame[slot]\n" + _FI_RET)

_NS_HEAD = "def next_slot(frame):\n    for slot in frame:\n"
_NS_BODY = "            return slot\n    return None\n"
_NS = _NS_HEAD + "        if frame[slot] is None:\n" + _NS_BODY
_NS_START = _NS_HEAD + "        if ...:\n" + _NS_BODY
_NS_FILLED = _NS_HEAD + "        if frame[slot] is not None:\n" + _NS_BODY
_NS_NEVER = _NS_HEAD + "        if slot is None:\n" + _NS_BODY
_NS_LAST = ("def next_slot(frame):\n    last = None\n    for slot in frame:\n        if frame[slot] is None:\n"
            "            last = slot\n    return last\n")

# The small cases the exercise texts show; the same in both lessons.
TOY_FIND = (("i want to go to boston on friday", "en"), ("ik wil vrijdag naar utrecht", "nl"))
TOY_FILL = (({"to": None, "from": None, "date": None}, {"to": "boston", "date": "friday"}),
            ({"to": "utrecht", "from": None, "date": "vrijdag"}, {"from": "groningen"}))
TOY_NEXT = (({"to": "boston", "from": None, "date": "friday"},),
            ({"to": "utrecht", "from": "groningen", "date": "vrijdag"},))
# Two slots of another frame, to show that the functions work for any frame.
TOY_PATTERNS = {"color": r"\b(red|blue|green)\b", "thing": r"\ba (car|bike|boat)\b"}


def _p(lang: str) -> _Val:
    return _Val(f'C.PATTERNS["{lang}"]', PATTERNS[lang])


REQUIRED = ["find_slots", "fill", "next_slot", "what"]

EXERCISES = {
    "find_slots": FunctionBox(
        name="find_slots",
        picture=find_picture,
        function="find_slots",
        start=_FS_START,
        solution=_FS,
        show=_show,
        visible=[(_written(t), _p(lang)) for t, lang in TOY_FIND],
        hidden=[(_written(""), _p("en")),
                (_written("from boston to chicago on monday"), _p("en")),
                (_written("i want to go to new york"), _p("en")),
                (_written("not to boston, to chicago"), _p("en")),
                (_written("van den haag naar zwolle op zondag"), _p("nl")),
                (_written("a red car and a blue bike"), _written(TOY_PATTERNS)),
                (_written("a boat"), _written(TOY_PATTERNS)),
                (_written("to boston"), _written({}))],
        first=[_DOTS],
        mistakes=[
            Mistake("whole", wrong=_FS_WHOLE,
                    en="`{call}` gives {got}, and it should give {want}: the stations still start with the word in front of them, such as *to* or *from*. That is what the whole match covers: which part of the match holds the station alone?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de stations beginnen nog met het woord van het patroon, zoals *to* of *naar*. Dat is wat de hele match omvat: welk deel van de match bevat alleen het station?"),
            Mistake("match", matches=_has_match,
                    en="`{call}` gives {got}, and it should give {want}: the values are matches, not strings. What gives the text that the group of a match covers?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de waarden zijn matches, geen strings. Wat geeft de tekst die de groep van een match omvat?"),
            Mistake("list", matches=_has_list,
                    en="`{call}` gives {got}, and it should give {want}: the values are lists. A slot holds one value: what does `re.search` give, and how do you get one string from it?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de waarden zijn lijsten. Een slot bevat één waarde: wat geeft `re.search`, en hoe haal je daar één string uit?"),
            Mistake("none", matches=_has_none,
                    en="`{call}` gives {got}, and it should give {want}: a slot that the text says nothing about is in it with `None`. Later, every key of `found` becomes a new value in the frame: what would happen to a slot that was filled in an earlier turn?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een slot waar de tekst niets over zegt, staat erin met `None`. Later wordt elke sleutel van `found` een nieuwe waarde in het frame: wat gebeurt er dan met een slot dat in een eerdere beurt gevuld was?"),
            Mistake("no-match", error=lambda e: isinstance(e, AttributeError),
                    en="`{call}` stops with an `AttributeError`: your function asks for a group where `re.search` found nothing and gave `None`. Does your function read a match only when there is one?",
                    nl="`{call}` stopt met een `AttributeError`: je functie vraagt om een groep waar `re.search` niets vond en `None` gaf. Leest je functie een match alleen als er een is?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with code that puts the value of the slot into `found` when `m` is a match.",
                      "nl": "Vervang de `...` door code die de waarde van het slot in `found` zet als `m` een match is."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does each slot whose regular expression matches get the text of its group, and does a slot without a match stay out of `found`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Krijgt elk slot met een passende reguliere expressie de tekst van de groep, en blijft een slot zonder match uit `found`?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does your function give an empty dictionary when nothing matches, and take the first match when there are two?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Geeft je functie een lege dictionary als niets past, en neemt de functie de eerste match als er twee zijn?"},
            "correct": {"en": "Right: your function fills slots the way the system does. In the picture below, it runs on ticket commands of MASSIVE: which words with a station, and which ways of saying a date, do the regular expressions not know?",
                        "nl": "Goed: je functie vult slots zoals het systeem het doet. In het plaatje hieronder draait de functie op kaartjesopdrachten van MASSIVE: welke woorden bij een station, en welke manieren om een datum te zeggen, kennen de reguliere expressies niet?"},
        },
    ),
    "fill": FunctionBox(
        name="fill",
        picture=fill_picture,
        function="fill",
        start=_FI_START,
        solution=_FI,
        show=_show,
        visible=[tuple(_written(x) for x in case) for case in TOY_FILL],
        hidden=[(_written(new_frame()), _written({})),
                (_written({"to": "boston", "from": "new york", "date": "friday"}), _written({"date": "saturday"})),
                (_written({"to": "boston", "from": "new york", "date": "friday"}),
                 _written({"to": "chicago", "from": "denver", "date": "monday"})),
                (_written({"to": None, "from": "zwolle", "date": None}), _written({"to": "leiden", "date": "morgen"})),
                (_written({"color": None, "thing": "car"}), _written({"color": "red"})),
                (_written({}), _written({}))],
        first=[_DOTS],
        mistakes=[
            Mistake("forget", wrong=_FI_FORGET,
                    en="`{call}` gives {got}, and it should give {want}: a slot that this turn says nothing about loses its value. What did that slot hold in the frame before the turn?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een slot waar deze beurt niets over zegt, raakt de waarde kwijt. Wat stond er in dat slot van het frame bij het begin van de beurt?"),
            Mistake("missing", wrong=_FI_MISSING,
                    en="`{call}` gives {got}, and it should give {want}: a slot that this turn says nothing about is gone from the frame. Should every slot of `frame` be in the new frame?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een slot waar deze beurt niets over zegt, is weg uit het frame. Hoort elk slot van `frame` in het nieuwe frame?"),
            Mistake("keep-old", wrong=_FI_OLD,
                    en="`{call}` gives {got}, and it should give {want}: a slot that was filled already keeps its old value, so the user cannot change it, as in *No, on Saturday.* What should a new value do to an old one?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een slot dat al gevuld was, houdt de oude waarde, dus de gebruiker kan het niet veranderen, zoals met *Nee, op zaterdag.* Wat moet een nieuwe waarde met een oude doen?"),
            Mistake("keys", error=lambda e: isinstance(e, KeyError),
                    en="`{call}` stops with a `KeyError`: your function looks up a slot in `found` that this turn did not fill. Where is the value of such a slot?",
                    nl="`{call}` stopt met een `KeyError`: je functie zoekt in `found` een slot op dat deze beurt niet vulde. Waar staat de waarde van zo'n slot?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: a slot that this turn did not fill keeps the value it has in `frame`.",
                      "nl": "Vervang de `...`: een slot dat deze beurt niet vulde, houdt de waarde die het in `frame` heeft."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does each slot of `frame` get the value in `found` when this turn filled it, and keep its own value when not?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Krijgt elk slot van `frame` de waarde uit `found` als deze beurt het vulde, en houdt het slot anders de eigen waarde?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does a new value take the place of an old one, and does each slot that this turn did not fill keep its value?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Neemt een nieuwe waarde de plaats in van een oude, en houdt elk slot dat deze beurt niet vulde de waarde?"},
            "correct": {"en": "Right: your function keeps the frame up to date the way the system does. In the picture below, the system uses your fill on the course's conversation: in which turn does a slot get a new value in place of an old one?",
                        "nl": "Goed: je functie houdt het frame bij zoals het systeem het doet. In het plaatje hieronder gebruikt het systeem je fill in het gesprek van de course: in welke beurt krijgt een slot een nieuwe waarde in plaats van een oude?"},
        },
    ),
    "next_slot": FunctionBox(
        name="next_slot",
        picture=next_picture,
        function="next_slot",
        start=_NS_START,
        solution=_NS,
        show=_show,
        visible=[tuple(_written(x) for x in case) for case in TOY_NEXT],
        hidden=[(_written(new_frame()),),
                (_written({"to": None, "from": "boston", "date": None}),),
                (_written({"to": "boston", "from": "denver", "date": None}),),
                (_written({"to": "leiden", "from": None, "date": None}),),
                (_written({"color": None, "thing": None}),),
                (_written({}),)],
        first=[_DOTS],
        mistakes=[
            Mistake("filled", wrong=_NS_FILLED,
                    en="`{call}` gives {got}, and it should give {want}: the first slot that is filled already. Which slot should the system ask about?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het eerste slot dat al gevuld is. Naar welk slot moet het systeem vragen?"),
            Mistake("never", wrong=_NS_NEVER,
                    en="`{call}` gives {got}, and it should give {want}: your function finds no empty slot, also when the frame has one. Does your test look at the name of the slot or at its value in the frame?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie vindt geen leeg slot, ook als het frame er een heeft. Kijkt je test naar de naam van het slot of naar de waarde in het frame?"),
            Mistake("last", wrong=_NS_LAST,
                    en="`{call}` gives {got}, and it should give {want}: the last empty slot. The system asks in the order of the frame: which empty slot comes first?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het laatste lege slot. Het systeem vraagt in de volgorde van het frame: welk leeg slot komt eerst?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the test for a slot that is still empty: its value in `frame` is still the `None` of a new frame.",
                      "nl": "Vervang de `...` door de test voor een slot dat nog leeg is: de waarde in `frame` is nog de `None` van een nieuw frame."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function give the first slot whose value is still `None`, and `None` when every slot is filled?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Geeft je functie het eerste slot met nog `None` als waarde, en `None` als elk slot gevuld is?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does your function give the first empty slot of the frame, also when there are more?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Geeft je functie het eerste lege slot van het frame, ook als er meer zijn?"},
            "unreached": {"en": "`{call}` gives `None`, but the frame has an empty slot. Does your function get to `return slot` with the first empty slot?",
                          "nl": "`{call}` geeft `None`, maar het frame heeft een leeg slot. Komt je functie met het eerste lege slot bij `return slot`?"},
            "no_return": {"en": "`{call}` gives `None`, but the frame has an empty slot. Does your function give back the slot it found?",
                          "nl": "`{call}` geeft `None`, maar het frame heeft een leeg slot. Geeft je functie het gevonden slot terug?"},
            "correct": {"en": "Right: your function picks the next question the way the system does. In the picture below, the system uses your next_slot on the course's conversation: after which turn are all slots filled, and what does the system ask then?",
                        "nl": "Goed: je functie kiest de volgende vraag zoals het systeem het doet. In het plaatje hieronder gebruikt het systeem je next_slot in het gesprek van de course: na welke beurt is geen slot meer leeg, en wat vraagt het systeem dan?"},
        },
    ),
}

EXERCISES["what"] = Sort(
    name="what",
    groups={
        "right": {"en": "It fills only right values", "nl": "Alles wat het invult, klopt"},
        "wrong": {"en": "It fills a slot with a wrong value", "nl": "Het vult een slot met een verkeerde waarde"},
        "nothing": {"en": "It fills no slot", "nl": "Het vult geen enkel slot"},
    },
    items={
        "all-three": {"en": "*From Boston to Chicago, tomorrow.*", "nl": "*Van Groningen naar Utrecht, morgen.*"},
        "bare": {"en": "*Chicago.*", "nl": "*Zwolle.*"},
        "leaving": {"en": "*I'm leaving Denver on Monday.*", "nl": "*Ik vertrek maandag uit Arnhem.*"},
        "not-to": {"en": "*Not to Boston, to Chicago.*", "nl": "*Niet naar Breda, naar Zwolle.*"},
        "week-after": {"en": "*To New York, a week from Monday.*", "nl": "*Naar Leiden, maandag over een week.*"},
        "day-word": {"en": "*Two tickets to Seattle the day after tomorrow.*", "nl": "*Naar Haarlem, om negen uur in de morgen.*"},
        "aunt": {"en": "*I'd like to visit my aunt in Miami.*", "nl": "*Ik wil bij mijn tante in Delft langs.*"},
        "home": {"en": "*I want to go home.*", "nl": "*Ik wil naar huis.*"},
    },
    solution={"all-three": "right", "bare": "right", "leaving": "right", "not-to": "wrong",
              "week-after": "wrong", "day-word": "wrong", "aunt": "nothing", "home": "nothing"},
    nudge={
        "all-three": {"en": "Which regular expression matches in this answer, and what does each group take?",
                      "nl": "Welke reguliere expressie past in dit antwoord, en wat neemt elke groep?"},
        "bare": {"en": "No regular expression matches the answer as it is. What does the system put in front of an answer to *Where are you going?*, and what does it read then?",
                 "nl": "Geen reguliere expressie past op het antwoord zoals het er staat. Wat zet het systeem bij een antwoord op *Waar ga je naartoe?* aan het begin, en wat leest het dan?"},
        "leaving": {"en": "The regular expression for the slot *from* wants the word *from* in front of a station. What does the system find, and is that value wrong or only incomplete?",
                    "nl": "De reguliere expressie voor het slot *van* wil het woord *van* bij een station. Wat vindt het systeem wel, en is deze waarde verkeerd of alleen onvolledig?"},
        "not-to": {"en": "`re.search` takes the first match. Which station comes first after the word *to*?",
                   "nl": "`re.search` neemt de eerste match. Welk station komt als eerste na het woord *naar*?"},
        "week-after": {"en": "The regular expression for the slot *date* finds the word *monday*. Is that the day the user wants to leave?",
                       "nl": "De reguliere expressie voor het slot *datum* vindt het woord *maandag*. Is dat de dag waarop de gebruiker wil vertrekken?"},
        "day-word": {"en": "Which day word does the regular expression for the slot *date* find, and what does it mean in this sentence?",
                     "nl": "Welk woord voor een dag vindt de reguliere expressie voor het slot *datum*, en wat betekent het in deze zin?"},
        "aunt": {"en": "Is *Miami* after *to* or *from*? What does the system read when it puts *to* in front of the answer?",
                 "nl": "Staat *Delft* na *naar* of *van*? Wat leest het systeem als het *naar* aan het begin van het antwoord zet?"},
        "home": {"en": "Is *home* one of the stations of the system?",
                 "nl": "Is *huis* een van de stations van het systeem?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the groups again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de groepen opnieuw uit."},
        "empty": {"en": "Sort each answer into the group that fits.",
                  "nl": "Deel elk antwoord in bij de groep die klopt."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "{item} belongs with another group.",
                  "nl": "{item} hoort bij een andere groep."},
        "correct": {"en": "All {n} are right. The system fills a slot only where a station or a day stands in a place it expects. It misses what it has no pattern for, such as the aunt in Miami, and it fills a slot wrongly when words fit a pattern but mean something else, as in *a week from Monday*. That is where the question *Shall I book it?* earns its place: a wrong value can still be corrected before anything is booked.",
                    "nl": "Alle {n} kloppen. Het systeem vult een slot alleen waar een station of een dag op een verwachte plaats staat. Het mist alles zonder patroon, zoals de tante in Delft, en het vult een slot verkeerd als woorden op een patroon passen maar iets anders betekenen, zoals in *maandag over een week*. Daar is de vraag *Zal ik het kaartje boeken?* nuttig: een verkeerde waarde kan nog verbeterd worden, voordat er iets geboekt is."},
    },
)

# What the tests feed each check, and what it must say: (exercise, code or choices, state, known mistake).
CASES = [
    ("find_slots", _FS_START, "empty", None),
    ("find_slots", _FS, "correct", None),
    ("find_slots", _FS_HEAD + "            found[...] = m.group(1)\n" + _FS_RET, "wrong", "dots-left"),
    ("find_slots", _FS_WHOLE, "wrong", "whole"),
    ("find_slots", _FS_MATCH, "wrong", "match"),
    ("find_slots", _FS_ALL, "wrong", "list"),
    ("find_slots", _FS_NONE, "wrong", "none"),
    ("find_slots", _FS_ALWAYS, "error", "no-match"),
    ("find_slots", _FS_NO_IF, "error", "no-match"),
    ("find_slots", "import re\n\n\ndef find_slots(text, patterns):\n    found = {}\n    for slot, pattern in patterns.items():\n"
     "        for m in re.finditer(pattern, text):\n            found[slot] = m.group(1)\n            break\n    return found\n",
     "correct", None),
    ("find_slots", _FS_HEAD + "            found[slot] = m[1]\n" + _FS_RET, "correct", None),
    ("fill", _FI_START, "empty", None),
    ("fill", _FI, "correct", None),
    ("fill", _FI_HEAD + _FI_IF + "            new[slot] = ...[slot]\n" + _FI_RET, "wrong", "dots-left"),
    ("fill", _FI_FORGET, "wrong", "forget"),
    ("fill", _FI_MISSING, "wrong", "missing"),
    ("fill", _FI_OLD, "wrong", "keep-old"),
    ("fill", _FI_KEYS, "error", "keys"),
    ("fill", "def fill(frame, found):\n    new = dict(frame)\n    for slot in found:\n        new[slot] = found[slot]\n"
     "    return new\n", "correct", None),
    ("fill", "def fill(frame, found):\n    return {slot: found.get(slot, frame[slot]) for slot in frame}\n", "correct", None),
    ("next_slot", _NS_START, "empty", None),
    ("next_slot", _NS, "correct", None),
    ("next_slot", _NS_HEAD + "        if ... is None:\n" + _NS_BODY, "wrong", "dots-left"),
    ("next_slot", _NS_FILLED, "wrong", "filled"),
    ("next_slot", _NS_NEVER, "wrong", "never"),
    ("next_slot", _NS_LAST, "wrong", "last"),
    ("next_slot", "def next_slot(frame):\n    empty = [s for s in frame if frame[s] is None]\n"
     "    return empty[0] if empty else None\n", "correct", None),
    ("next_slot", "def next_slot(frame):\n    for slot, value in frame.items():\n        if value == None:\n"
     "            return slot\n", "correct", None),
    ("what", {}, "empty", None),
    ("what", dict(EXERCISES["what"].solution), "correct", None),
    ("what", {"all-three": "right", "aunt": "nothing"}, "wrong", "unfinished"),
] + [("what", {item: ("nothing" if group != "nothing" else "right")}, "wrong", f"wrong:{item}")
     for item, group in EXERCISES["what"].solution.items()]
