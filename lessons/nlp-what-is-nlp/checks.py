# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-what-is-nlp: sorting systems into the two sides of NLP, understanding (NLU) and making
(NLG), and a small assistant for a bike shop that finds what a request is about by its keywords. The systems, the
requests, the keywords and the replies are made up by the course."""
import contextlib
import copy
import html
import io
import re

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

# --- the course's systems and requests -------------------------------------------------------------------------------

# Made-up systems, one line each, with what goes in and what comes out: "text" (human language) or "data" (numbers,
# labels, a list).
SYSTEMS = [
    {"text": "A mail program moves the messages it takes for spam to a separate folder.",
     "input": "text", "output": "data"},
    {"text": "A weather site writes a short forecast from the measured temperature, wind and rain.",
     "input": "data", "output": "text"},
    {"text": "An app translates a Spanish menu into Dutch.", "input": "text", "output": "text"},
    {"text": "A shop sorts its incoming emails into complaints, questions and orders.",
     "input": "text", "output": "data"},
    {"text": "A news site writes a match report from the score and the minutes of the goals.",
     "input": "data", "output": "text"},
    {"text": "A tool sums up a long report in five sentences.", "input": "text", "output": "text"},
    {"text": "A spreadsheet adds up the sales of each month.", "input": "data", "output": "data"},
    {"text": "A train app writes a delay notice from the timetable and where the train is now.",
     "input": "data", "output": "text"},
    {"text": "A chat assistant answers questions about a bike shop's opening hours.",
     "input": "text", "output": "text"},
]
SYSTEMS_NL = [
    "Een mailprogramma zet de berichten die het voor spam houdt in een aparte map.",
    "Een weersite schrijft een korte verwachting uit de gemeten temperatuur, wind en regen.",
    "Een app vertaalt een Spaanse menukaart in het Nederlands.",
    "Een winkel deelt de binnenkomende mails in: klachten, vragen en bestellingen.",
    "Een nieuwssite schrijft een wedstrijdverslag uit de uitslag en de minuten van de doelpunten.",
    "Een tool vat een lang rapport samen in vijf zinnen.",
    "Een spreadsheet telt de verkoop van elke maand op.",
    "Een treinapp schrijft een melding over vertraging uit de dienstregeling en de plek van de trein.",
    "Een chatassistent beantwoordt vragen over de openingstijden van een fietsenwinkel.",
]
SIDES = ["NLU", "NLG", "both", "none"]

# A made-up assistant of a bike shop: what a request can be about, and the words that point to it. The order matters:
# the first kind with a matching word wins.
KEYWORDS = {
    "hours": ["open", "close", "hours"],
    "repair": ["repair", "flat", "broken"],
    "price": ["cost", "price", "how much"],
}
REQUESTS = [
    "When do you open on Saturday?",
    "My back tire is flat.",
    "How much does a new chain cost?",
    "Can you repair my brakes today?",
    "Open on Sunday?",
    "The lock on my bike won't open.",
    "Do you sell helmets?",
    "What is the price of a basket?",
    "My bike makes a strange noise when I brake.",
]
REQUESTS_NL = [
    "Hoe laat gaan jullie zaterdag open?",
    "Mijn achterband is lek.",
    "Wat kost een nieuwe ketting?",
    "Kunnen jullie vandaag mijn remmen repareren?",
    "Zondag open?",
    "Het slot van mijn fiets gaat niet open.",
    "Verkopen jullie helmen?",
    "Wat is de prijs van een mandje?",
    "Mijn fiets maakt een raar geluid als ik rem.",
]
# what each request is really about, as a person reads it
MEANT = ["hours", "repair", "price", "repair", "hours", "repair", "unknown", "price", "repair"]

# the course's replies: making text from what the assistant understood
REPLIES = {
    "en": {"hours": "We are open Monday to Saturday, from 9 a.m. to 6 p.m.",
           "repair": "Bring your bike in: most repairs are done the same day.",
           "price": "Every item has its price on a tag, and we are happy to give a quote.",
           "unknown": "Sorry, I did not understand that. Could you say it another way?"},
    "nl": {"hours": "We zijn open van maandag tot en met zaterdag, van 9.00 tot 18.00 uur.",
           "repair": "Breng je fiets langs: de meeste reparaties zijn dezelfde dag klaar.",
           "price": "Alles in de winkel heeft een prijskaartje, en we maken graag een offerte.",
           "unknown": "Sorry, dat begreep ik niet. Kun je het anders zeggen?"},
}
INTENT_NL = {"hours": "openingstijden", "repair": "reparatie", "price": "prijs", "unknown": "onbekend"}


def side(system: dict) -> str:
    """The course's own Exercise 1: text in and text out is both sides, text in is understanding (NLU), text out is
    making (NLG), and with text on neither side it is not NLP."""
    if system["input"] == "text" and system["output"] == "text":
        return "both"
    elif system["input"] == "text":
        return "NLU"
    elif system["output"] == "text":
        return "NLG"
    return "none"


def intent(request: str, keywords: dict) -> str:
    """The course's own Exercise 2: the first kind of request with a keyword in the request, else "unknown"."""
    text = request.lower()
    for name, words in keywords.items():
        for word in words:
            if word in text:
                return name
    return "unknown"


def reply(kind: str, lang: str) -> str:
    """The assistant's answer to a kind of request, in the lesson's language."""
    return REPLIES[lang].get(kind, REPLIES[lang]["unknown"])


class Named:
    """A value the check gives the learner's function, shown in a message by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


SYSTEMS_N = [Named(f"SYSTEMS[{i}]", s) for i, s in enumerate(SYSTEMS)]
REQUESTS_N = [Named(f"REQUESTS[{i}]", r) for i, r in enumerate(REQUESTS)]
KEYWORDS_N = Named("KEYWORDS", KEYWORDS)


def system_text(i: int, lang: str) -> str:
    return SYSTEMS_NL[i] if lang == "nl" else SYSTEMS[i]["text"]


def request_text(i: int, lang: str) -> str:
    return REQUESTS_NL[i] if lang == "nl" else REQUESTS[i]


def request_options(lang: str) -> dict:
    """The requests for the demo's dropdown: the English request, with its Dutch meaning in a Dutch lesson."""
    if lang == "nl":
        return {f"{i + 1}. {r} ({REQUESTS_NL[i]})": i for i, r in enumerate(REQUESTS)}
    return {f"{i + 1}. {r}": i for i, r in enumerate(REQUESTS)}


# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.15rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;color:var(--ea-ink);vertical-align:top")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"
_BOX = ("flex:1 1 8rem;min-width:7.5rem;border:1px solid var(--ea-line);border-radius:6px;padding:0.4rem 0.5rem;"
        "font-family:var(--ea-font-ui);font-size:0.85rem;color:var(--ea-ink)")
_STEP = ("flex:0 1 9rem;min-width:7rem;padding:0.3rem 0.4rem;font-family:var(--ea-font-ui);font-size:0.8rem;"
         "color:var(--ea-ink-2);text-align:center;align-self:center")


def _note(text: str) -> str:
    return f'<div style="{_UI}">{html.escape(text)}</div>'


def _word(text: str) -> str:
    """A short value as Python writes it, kept on one line in a table cell."""
    return f'<span style="font-family:var(--ea-font-code);font-size:12px;white-space:nowrap">{html.escape(text)}</span>'


def _table(rows: list, head: list) -> str:
    """A table with a head row; each row is a list of (html, style)."""
    top = "".join(f'<th style="{_HEAD}">{html.escape(h)}</th>' for h in head)
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{";" + style if style else ""}">{cell}</td>'
                                    for cell, style in row) + "</tr>" for row in rows)
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.3rem 0">'
            f'<tr>{top}</tr>{body}</table></div>')


def _chain(parts: list) -> str:
    """Boxes and the steps between them in a row: parts alternate (title, text) of a box and the name of a step."""
    out = []
    for k, part in enumerate(parts):
        if k % 2 == 0:
            title, text = part
            out.append(f'<div style="{_BOX}"><div style="font-weight:600">{html.escape(title)}</div>'
                       f'<div style="color:var(--ea-ink-2);margin-top:0.2rem">{html.escape(text)}</div></div>')
        else:
            out.append(f'<div style="{_STEP}">{html.escape(part)} &rarr;</div>')
    return f'<div style="display:flex;flex-wrap:wrap;gap:0.3rem;align-items:stretch">{"".join(out)}</div>'


_SIDES_WORDS = {
    "en": {"title": "The two sides of NLP",
           "text": "text", "meaning": "meaning", "nlu": "understanding (NLU)", "nlg": "making (NLG)",
           "said": "When do you open on Saturday?", "meant": "kind: hours; day: Saturday",
           "out": "On Saturday we are open from 9 a.m. to 6 p.m.",
           "note": "A spam filter only understands: text in, a label out. A weather report written from numbers only "
                   "makes text. Translating and summing up do both: text in, text out."},
    "nl": {"title": "De twee kanten van NLP",
           "text": "tekst", "meaning": "betekenis", "nlu": "begrijpen (NLU)", "nlg": "maken (NLG)",
           "said": "Hoe laat gaan jullie zaterdag open?", "meant": "soort: openingstijden; dag: zaterdag",
           "out": "Zaterdag zijn we open van 9.00 tot 18.00 uur.",
           "note": "Een spamfilter begrijpt alleen: tekst erin, een label eruit. Een weerbericht uit getallen maakt "
                   "alleen tekst. Vertalen en samenvatten doen allebei: tekst erin, tekst eruit."},
}


def sides_picture(lang: str) -> str:
    """Text, its meaning and text again, with understanding and making between them."""
    w = _SIDES_WORDS[lang]
    body = _chain([(w["text"], w["said"]), w["nlu"], (w["meaning"], w["meant"]), w["nlg"], (w["text"], w["out"])])
    return yours.figure(w["title"], body + _note(w["note"]),
                        alt=f'{w["title"]}: {w["text"]} ({w["said"]}), {w["nlu"]}, {w["meaning"]} ({w["meant"]}), '
                            f'{w["nlg"]}, {w["text"]} ({w["out"]}). {w["note"]}')


_GRID_WORDS = {
    "en": {"title": "What goes in, what comes out", "in": "in", "out": "out", "text": "text", "data": "data",
           "cells": {"both": "both sides, such as translating", "NLU": "understanding (NLU), such as a spam filter",
                     "NLG": "making (NLG), such as a weather report from numbers", "none": "not NLP, such as a spreadsheet"}},
    "nl": {"title": "Wat erin gaat, wat eruit komt", "in": "erin", "out": "eruit", "text": "tekst", "data": "data",
           "cells": {"both": "beide kanten, zoals vertalen", "NLU": "begrijpen (NLU), zoals een spamfilter",
                     "NLG": "maken (NLG), zoals een weerbericht uit getallen", "none": "geen NLP, zoals een spreadsheet"}},
}


def grid_picture(lang: str) -> str:
    """The four answers of the recipe: text or data in, against text or data out."""
    w = _GRID_WORDS[lang]
    kinds = {("text", "text"): "both", ("text", "data"): "NLU", ("data", "text"): "NLG", ("data", "data"): "none"}
    head = [f'{w["in"]} ↓, {w["out"]} →', w["text"], w["data"]]
    rows = [[(html.escape(w[i]), "font-weight:600")] + [(html.escape(w["cells"][kinds[(i, o)]]), "")
                                                        for o in ("text", "data")] for i in ("text", "data")]
    said = "; ".join(f'{w["in"]} {w[i]}, {w["out"]} {w[o]}: {w["cells"][kinds[(i, o)]]}'
                     for i in ("text", "data") for o in ("text", "data"))
    return yours.figure(w["title"], _table(rows, head), alt=f'{w["title"]}. {said}.')


_DEMO_WORDS = {
    "en": {"title": "Request {i} to the bike shop's assistant", "request": "the request", "kind": "the kind it finds",
           "reply": "the reply it makes", "nlu": "understanding", "nlg": "making",
           "meant": "A person reads it as: {meant}.", "same": "The assistant got it right.",
           "differs": "The assistant got it wrong: its keywords do not see what the request means."},
    "nl": {"title": "Verzoek {i} aan het programma van de fietsenwinkel", "request": "het verzoek",
           "kind": "de soort die het vindt", "reply": "het antwoord dat het maakt", "nlu": "begrijpen", "nlg": "maken",
           "meant": "Een mens leest het als: {meant}.", "same": "Het programma had het goed.",
           "differs": "Het programma had het fout: de trefwoorden zien niet wat het verzoek betekent."},
}


def _kind_name(kind: str, lang: str) -> str:
    return INTENT_NL.get(kind, kind) if lang == "nl" else kind


def demo_picture(i: int, lang: str) -> str:
    """One request: what the course's intent finds in it, and the reply the assistant makes from that."""
    i = max(0, min(int(i), len(REQUESTS) - 1))
    w = _DEMO_WORDS[lang]
    found = intent(REQUESTS[i], KEYWORDS)
    said = REQUESTS[i] if lang == "en" else f"{REQUESTS[i]} ({REQUESTS_NL[i]})"
    body = _chain([(w["request"], said), w["nlu"], (w["kind"], _kind_name(found, lang)), w["nlg"],
                   (w["reply"], reply(found, lang))])
    lines = [w["meant"].format(meant=_kind_name(MEANT[i], lang)), w["same"] if found == MEANT[i] else w["differs"]]
    title = w["title"].format(i=i + 1)
    return yours.figure(title, body + "".join(_note(x) for x in lines),
                        alt=f'{title}. {said} {w["kind"]}: {_kind_name(found, lang)}. {w["reply"]}: '
                            f'{reply(found, lang)} ' + " ".join(lines))


# --- the pictures under the checks -----------------------------------------------------------------------------------

def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error


def _call(fn, *args):
    """The learner's function on copies of args within the time limit of a check; _STOPPED when it stops with an
    error, and the Timeout passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(lambda: fn(*(copy.deepcopy(a) for a in args)))
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _shown(v, lang: str) -> str:
    """What the learner's function gave, for a cell of a table, as Python writes it."""
    if v is _STOPPED:
        return {"en": "error", "nl": "fout"}[lang]
    text = repr(v)
    return text if len(text) <= 40 else text[:37] + "..."


_CHECK_WORDS = {
    "en": {"side": {"title": "Your side on the nine systems",
                    "head": ["system", "in", "out", "yours", "the course's"],
                    "fine": "The same as the course's for all nine systems: {counts}.",
                    "differs": "It differs from the course's for {k} of the nine systems: they have a dashed frame."},
           "intent": {"title": "Your intent(request, KEYWORDS) on the nine requests",
                      "head": ["request", "yours", "the course's", "what it means"],
                      "fine": "The same as the course's for all nine requests. It gets {wrong} of them wrong, as "
                              "the course's does: compare the last two columns.",
                      "differs": "It differs from the course's for {k} of the nine requests: they have a dashed frame."},
           "alt": "{title}: {rows}. {after}"},
    "nl": {"side": {"title": "Jouw side op de negen systemen",
                    "head": ["systeem", "erin", "eruit", "de jouwe", "die van de course"],
                    "fine": "Bij alle negen systemen hetzelfde als die van de course: {counts}.",
                    "differs": "Bij {k} van de negen systemen anders dan die van de course: die hebben een gestreept kader."},
           "intent": {"title": "Jouw intent(request, KEYWORDS) op de negen verzoeken",
                      "head": ["verzoek", "de jouwe", "die van de course", "wat het betekent"],
                      "fine": "Bij alle negen verzoeken hetzelfde als die van de course. Net als bij die van de course "
                              "zijn er {wrong} fout: vergelijk de laatste twee kolommen.",
                      "differs": "Bij {k} van de negen verzoeken anders dan die van de course: die hebben een gestreept kader."},
           "alt": "{title}: {rows}. {after}"},
}


def side_picture(seen) -> str:
    """The nine systems, with the learner's side beside the course's."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _CHECK_WORDS[seen.lang]
    words = w["side"]
    rows, said, k = [], [], 0
    for i, system in enumerate(SYSTEMS):
        got, want = _call(fn, system), side(system)
        differs = got != want
        k += differs
        rows.append([(html.escape(system_text(i, seen.lang)), ""), (_word(system["input"]), ""),
                     (_word(system["output"]), ""), (_word(_shown(got, seen.lang)), _OFF if differs else ""),
                     (_word(repr(want)), "")])
        said.append(f"{i + 1}: {_shown(got, seen.lang)}, {want!r}")
    counts = ", ".join(f"{sum(side(s) == kind for s in SYSTEMS)} {kind!r}" for kind in SIDES)
    after = words["differs"].format(k=k) if k else words["fine"].format(counts=counts)
    return yours.figure(words["title"], _table(rows, words["head"]) + _note(after),
                        alt=w["alt"].format(title=words["title"], rows="; ".join(said), after=after))


def intent_picture(seen) -> str:
    """The nine requests, with the learner's kind beside the course's and what each request means."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _CHECK_WORDS[seen.lang]
    words = w["intent"]
    rows, said, k = [], [], 0
    for i, request in enumerate(REQUESTS):
        got, want = _call(fn, request, KEYWORDS), intent(request, KEYWORDS)
        differs = got != want
        k += differs
        shown = html.escape(request) + (f'<div style="color:var(--ea-ink-3)">{html.escape(REQUESTS_NL[i])}</div>'
                                        if seen.lang == "nl" else "")
        rows.append([(shown, ""), (_word(_shown(got, seen.lang)), _OFF if differs else ""), (_word(repr(want)), ""),
                     (_word(repr(MEANT[i])), "")])
        said.append(f"{request} {_shown(got, seen.lang)}, {want!r}, {MEANT[i]!r}")
    wrong = sum(intent(r, KEYWORDS) != m for r, m in zip(REQUESTS, MEANT))
    after = words["differs"].format(k=k) if k else words["fine"].format(wrong=wrong)
    return yours.figure(words["title"], _table(rows, words["head"]) + _note(after),
                        alt=w["alt"].format(title=words["title"], rows="; ".join(said), after=after))


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["side", "intent", "names", "open"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _names_error(e) -> bool:
    return isinstance(e, NameError) and re.search(r"'(SYSTEMS|REQUESTS|KEYWORDS|C)'", str(e)) is not None


_NAMES = Mistake("lesson-names", error=_names_error,
                 en="Your function uses a name of the lesson, such as `SYSTEMS`, `KEYWORDS` or `C`, and the check does not give it that name: use the names between the parentheses of the `def` line.",
                 nl="Je functie gebruikt een naam van de les, zoals `SYSTEMS`, `KEYWORDS` of `C`, en de check geeft die naam niet mee: gebruik de namen tussen de haakjes van de `def`-regel.")

# exercise 1: side; the last case is data in and text out once more, written by the check itself
SIDE_CASES = [(s,) for s in SYSTEMS_N] + [({"text": "", "input": "data", "output": "text"},)]
_S_START = "def side(system):\n    ...\n"
_S = ("def side(system):\n"
      "    if system[\"input\"] == \"text\" and system[\"output\"] == \"text\":\n"
      "        return \"both\"\n"
      "    elif system[\"input\"] == \"text\":\n"
      "        return \"NLU\"\n"
      "    elif system[\"output\"] == \"text\":\n"
      "        return \"NLG\"\n"
      "    return \"none\"\n")
_S_ORDER = ("def side(system):\n"
            "    if system[\"input\"] == \"text\":\n"
            "        return \"NLU\"\n"
            "    elif system[\"output\"] == \"text\":\n"
            "        return \"NLG\"\n"
            "    return \"none\"\n")
_S_SWAP = _S.replace("\"NLU\"", "\"X\"").replace("\"NLG\"", "\"NLU\"").replace("\"X\"", "\"NLG\"")
_S_NONE = ("def side(system):\n"
           "    if system[\"input\"] == \"text\" and system[\"output\"] == \"text\":\n"
           "        return \"both\"\n"
           "    elif system[\"input\"] == \"text\":\n"
           "        return \"NLU\"\n"
           "    return \"NLG\"\n")

# exercise 2: intent
INTENT_CASES = [(r, KEYWORDS_N) for r in REQUESTS_N] + [("CLOSED ON MONDAY?", {"days": ["monday"], "x": ["closed"]}),
                                                        ("A flat tire", {"a": [], "b": ["tire", "flat"]})]
_I_START = "def intent(request, keywords):\n    ...\n"
_I = ("def intent(request, keywords):\n"
      "    text = request.lower()\n"
      "    for name, words in keywords.items():\n"
      "        for word in words:\n"
      "            if word in text:\n"
      "                return name\n"
      "    return \"unknown\"\n")
_I_WORD = _I.replace("                return name\n", "                return word\n")
_I_EARLY = _I.replace("                return name\n", "                return name\n            return \"unknown\"\n")
_I_LOWER = _I.replace("    text = request.lower()\n", "    text = request\n")


EXERCISES = {
    "side": FunctionBox(
        name="side",
        picture=side_picture,
        function="side",
        start=_S_START,
        solution=_S,
        visible=[SIDE_CASES[0]],
        hidden=SIDE_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("swap", wrong=_S_SWAP,
                    en="`{call}` gives {got}, and it should give {want}: text goes in and a label comes out, so the system gets something out of text. That is understanding; making is the side where text comes out.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: er gaat tekst in en er komt een label uit, dus het systeem haalt iets uit tekst. Dat is begrijpen; maken is de kant waar tekst uitkomt."),
            Mistake("order", wrong=_S_ORDER,
                    en="`{call}` gives {got}, and it should give {want}: text goes in and text comes out. Your function stops at the first question, about what goes in; ask first whether both are text.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: er gaat tekst in en er komt tekst uit. Jouw functie stopt bij de eerste vraag, over wat erin gaat; vraag eerst of allebei tekst zijn."),
            Mistake("none", wrong=_S_NONE,
                    en="`{call}` gives {got}, and it should give {want}: numbers go in and numbers come out, so no human language is involved at all. Your function only knows three answers; give `\"none\"` when neither is text.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: er gaan getallen in en er komen getallen uit, dus er komt geen menselijke taal aan te pas. Jouw functie kent maar drie antwoorden; geef `\"none\"` als geen van beide tekst is."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: compare what goes in and what comes out with `\"text\"`, and give back one of `\"both\"`, `\"NLU\"`, `\"NLG\"` or `\"none\"`.",
                      "nl": "Vervang de `...`: vergelijk wat erin gaat en wat eruit komt met `\"text\"`, en geef een van `\"both\"`, `\"NLU\"`, `\"NLG\"` of `\"none\"` terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The function looks at the values under the keys `\"input\"` and `\"output\"`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De functie kijkt naar de waarden onder de keys `\"input\"` en `\"output\"`."},
            "almost": {"en": "Right for `SYSTEMS[0]`. But `{call}` gives {got}, and it should give {want}. Does your function look at the `system` it gets?",
                       "nl": "Goed voor `SYSTEMS[0]`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Kijkt je functie naar het `system` dat ze krijgt?"},
            "correct": {"en": "Your `side` is right in all {n} cases. Of the course's nine systems, two only understand, three only make text, three do both, and the spreadsheet is not NLP at all. A chat assistant does both: first get what you mean, then write an answer.",
                        "nl": "Je `side` klopt in alle {n} gevallen. Van de negen systemen van de course begrijpen er twee alleen, maken er drie alleen tekst, doen er drie allebei, en is de spreadsheet helemaal geen NLP. Een chatassistent doet allebei: eerst snappen wat je bedoelt, dan een antwoord schrijven."},
        },
    ),
    "intent": FunctionBox(
        name="intent",
        picture=intent_picture,
        function="intent",
        start=_I_START,
        solution=_I,
        visible=[INTENT_CASES[0]],
        hidden=INTENT_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("word", wrong=_I_WORD,
                    en="`{call}` gives {got}, and it should give {want}: that is the keyword it found. The function gives the kind of request the word belongs to, the key of the dict.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is het trefwoord dat ze vond. De functie geeft de soort verzoek waar het woord bij hoort, de key van de dict."),
            Mistake("early", wrong=_I_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your function gives up after the first word that is not in the request. Say `\"unknown\"` only after both loops have tried every word.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw functie geeft het op na het eerste woord dat niet in het verzoek staat. Zeg pas `\"unknown\"` als beide loops elk woord geprobeerd hebben."),
            Mistake("lower", wrong=_I_LOWER,
                    en="`{call}` gives {got}, and it should give {want}: the request starts with a capital letter, and the keywords are written in lowercase. Make the request lowercase first, with `.lower()`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het verzoek begint met een hoofdletter, en de trefwoorden zijn met kleine letters geschreven. Zet het verzoek eerst in kleine letters, met `.lower()`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: go over the kinds and their words, and give back the first kind with a word in the request.",
                      "nl": "Vervang de `...`: ga de soorten en hun woorden langs, en geef de eerste soort terug met een woord in het verzoek."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The function gives the first key of `keywords` with a word that occurs in the request, in lowercase, and `\"unknown\"` when no word occurs in it.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De functie geeft de eerste key van `keywords` met een woord dat in het verzoek voorkomt, in kleine letters, en `\"unknown\"` als er geen woord in voorkomt."},
            "almost": {"en": "Right for the first request. But `{call}` gives {got}, and it should give {want}. Does your function use the `request` and `keywords` it gets?",
                       "nl": "Goed voor het eerste verzoek. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie het `request` en de `keywords` die ze krijgt?"},
            "correct": {"en": "Your `intent` is right in all {n} cases. Look at the picture: on the course's nine requests it gets seven right and two wrong, just like the course's own version.",
                        "nl": "Je `intent` klopt in alle {n} gevallen. Kijk naar de afbeelding: van de negen verzoeken van de course zijn er zeven goed en twee fout, net als bij de versie van de course."},
        },
    ),
}

CASES = [
    ("side", _S_START, "empty", None),
    ("side", _S, "correct", None),
    ("side", "def side(system):\n    a = system['input'] == 'text'\n    b = system['output'] == 'text'\n    if a and b:\n        return 'both'\n    if a:\n        return 'NLU'\n    if b:\n        return 'NLG'\n    return 'none'\n", "correct", None),
    ("side", _S_SWAP, "wrong", "swap"),
    ("side", _S_ORDER, "wrong", "order"),
    ("side", _S_NONE, "wrong", "none"),
    ("side", "def side(system):\n    return 'NLU'\n", "almost", None),
    ("side", "def side(system):\n    return side(SYSTEMS[0])\n", "error", "lesson-names"),
    ("intent", _I_START, "empty", None),
    ("intent", _I, "correct", None),
    ("intent", "def intent(request, keywords):\n    for name in keywords:\n        for word in keywords[name]:\n            if word in request.lower():\n                return name\n    return 'unknown'\n", "correct", None),
    ("intent", _I_WORD, "wrong", "word"),
    ("intent", _I_EARLY, "wrong", "early"),
    ("intent", _I_LOWER, "wrong", "lower"),
    ("intent", "def intent(request, keywords):\n    return 'hours'\n", "almost", None),
    ("intent", "def intent(request, keywords):\n    return KEYWORDS\n", "error", "lesson-names"),
]
