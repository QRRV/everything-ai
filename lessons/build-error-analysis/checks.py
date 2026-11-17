# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-error-analysis: read the answers of an app before you measure it. The app answers customers of an
online bank: Qwen3-0.6B after a system prompt with six facts, one per topic, answering in JSON with a topic and a
reply. The course's run on the 120 test questions of data/banking77-small.json (data/course-production-qwen.json), the
course's notes on each reply (the kinds of mistake it found, a failure taxonomy), two functions on those notes (count
the kinds, find the biggest), the pictures and the answers to read."""
import functools
import html
import json
import re

from everything_ai import paths, yours
from everything_ai.check import FunctionBox, Mistake


FACTS = ("- A new card arrives within 5 working days. Its status is in the app under Cards. There is no tracking number.\n"
         "- A lost or stolen card: freeze it at once in the app under Cards > Freeze, then order a new card there. "
         "A new card is free.\n"
         "- A payment in another currency gets the exchange rate at the moment of payment; on weekends 0.5% is added. "
         "The app shows the rate before you pay.\n"
         "- When a card payment got the wrong exchange rate: send the date and the amount of the payment in this chat, "
         "and the team checks it within 2 working days.\n"
         "- The PIN is blocked after 3 wrong tries. Unblock it in the app under Cards > PIN, 24 hours after the last try.\n"
         "- Refunds: ask the shop first. If the shop has not helped within 14 days, the team can open a dispute.")
TOPICS = ("card_arrival", "lost_or_stolen_card", "exchange_rate", "card_payment_wrong_exchange_rate", "pin_blocked",
          "request_refund")
SYSTEM = ("You answer customers of an online bank in the chat of its app. Use only these facts:\n" + FACTS + "\n"
          "If the facts do not answer the question, say that a person from the team will reply.\n"
          'Answer with JSON only: {"topic": "...", "reply": "..."}. topic is one of ' + ", ".join(TOPICS) + ". "
          "reply is your answer to the customer, in at most three sentences.")
RECORD = {"v1": {"labels": "test", "system": SYSTEM, "max_tokens": 200}}
DATA = paths.ROOT / "data" / "course-production-qwen.json"
BANKING = paths.ROOT / "data" / "banking77-small.json"


class Named(list):
    """A list of the lesson's, written in the check's messages by its name, such as NOTES."""

    def __init__(self, items, name: str):
        super().__init__(items)
        self.name = name

    def __repr__(self):
        return self.name


class NamedDict(dict):
    """A dict of the lesson's, written in the check's messages by its name, such as COUNTS."""

    def __init__(self, items, name: str):
        super().__init__(items)
        self.name = name

    def __repr__(self):
        return self.name


@functools.cache
def _banking() -> dict:
    return json.loads(BANKING.read_text(encoding="utf-8"))


QUESTIONS = tuple(text for _, text in _banking()["test"])
LABELS = tuple(label for label, _ in _banking()["test"])


@functools.cache
def answers(lesson: str = "build-error-analysis", key: str = "v1") -> tuple:
    """The course's run: what Qwen3 answered to each test question, in the order of the file."""
    return tuple(json.loads(DATA.read_text(encoding="utf-8"))["parts"][lesson][key]["answers"])


def parsed(i: int) -> dict:
    """Answer i as the JSON it is: all 120 answers of the first version are."""
    return json.loads(answers()[i]["text"])


def sentences(text: str) -> int:
    """The number of sentences in a text: pieces that end with . ! or ?, and a last piece without one."""
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])


# --- the course's notes: what it found wrong in each reply ----------------------------------------------------------------

KINDS = ("wrong-topic", "other-problem", "admits", "made-up", "elsewhere", "too-long")   # in the order the notes first have them
KIND = {
    "wrong-topic": {"en": "wrong topic", "nl": "verkeerd onderwerp"},
    "other-problem": {"en": "another problem", "nl": "ander probleem"},
    "admits": {"en": "admits a mistake", "nl": "geeft een fout toe"},
    "made-up": {"en": "made up", "nl": "verzonnen"},
    "elsewhere": {"en": "sends elsewhere", "nl": "stuurt door"},
    "too-long": {"en": "too long", "nl": "te lang"},
}
RULE = {   # what each kind means: the rule the course read every reply with
    "wrong-topic": {"en": "The topic is not the one the question is about.",
                    "nl": "Het onderwerp is niet het onderwerp waar de vraag over gaat."},
    "other-problem": {"en": "The reply asks the customer to do what belongs to another problem, such as sending the date and amount of a payment for a refund.",
                      "nl": "Het antwoord vraagt de klant te doen wat bij een ander probleem hoort, zoals de datum en het bedrag van een betaling sturen voor een terugbetaling."},
    "admits": {"en": "The reply says the bank's exchange rate was wrong, before anyone checked.",
               "nl": "Het antwoord zegt dat de wisselkoers van de bank fout was, voordat iemand het controleerde."},
    "made-up": {"en": "The reply says something about the bank, its cards or its app that the facts do not say, or the opposite.",
                "nl": "Het antwoord zegt iets over de bank, haar kaarten of haar app dat niet in de feiten staat, of het tegendeel."},
    "elsewhere": {"en": "The reply sends the customer to the bank or its customer service, while this chat is the bank's own.",
                  "nl": "Het antwoord stuurt de klant naar de bank of haar klantenservice, terwijl deze chat van de bank zelf is."},
    "too-long": {"en": "The reply has more than three sentences.",
                 "nl": "Het antwoord heeft meer dan drie zinnen."},
}
_LETTER = {"T": "wrong-topic", "O": "other-problem", "A": "admits", "M": "made-up", "E": "elsewhere", "L": "too-long"}
# The course read every reply against the facts and wrote down the kinds that apply; a reply without an entry is right.
_NOTED = {0: "TO", 3: "TOA", 7: "M", 9: "TM", 10: "A", 11: "A", 12: "E", 13: "TO", 14: "A", 15: "A", 17: "TA",
          22: "TO", 23: "A", 24: "TO", 25: "TOM", 26: "TO", 27: "M", 29: "T", 30: "A", 31: "TE", 32: "TO", 34: "TO",
          37: "OM", 39: "M", 42: "A", 44: "M", 45: "L", 47: "M", 48: "M", 50: "TOAE", 51: "TO", 54: "TO", 58: "TO",
          59: "M", 60: "T", 63: "M", 64: "A", 68: "A", 69: "M", 71: "A", 73: "A", 76: "OM", 78: "T", 79: "M",
          80: "TO", 82: "T", 84: "TO", 85: "TM", 86: "TO", 88: "TO", 89: "TO", 90: "A", 91: "A", 92: "M", 95: "TM",
          96: "M", 98: "TO", 99: "TOM", 100: "M", 101: "TO", 102: "T", 104: "A", 105: "A", 106: "T", 107: "TO",
          111: "M", 113: "TO", 114: "TO", 115: "O", 116: "M", 117: "TOE", 118: "TOM", 119: "TO"}
NOTES = Named([[_LETTER[c] for c in _NOTED.get(i, "")] for i in range(len(QUESTIONS))], "NOTES")
# what the course wrote in its own words, before it sorted the notes into kinds, for the replies the lesson shows
NOTE_TEXT = {
    1: {"en": "Right: the facts about a new card, in two sentences.",
        "nl": "Goed: de feiten over een nieuwe kaart, in twee zinnen."},
    3: {"en": "The customer's PIN is blocked. The reply says the exchange rate of a payment was wrong, and asks for its date and amount. Topic: a wrong exchange rate.",
        "nl": "De pincode van de klant is geblokkeerd. Het antwoord zegt dat de wisselkoers van een betaling fout was, en vraagt om de datum en het bedrag. Onderwerp: een foute wisselkoers."},
    79: {"en": "Says the app gives a 0.5% discount on weekends; the facts say 0.5% is added.",
         "nl": "Zegt dat de app in het weekend 0,5% korting geeft; volgens de feiten komt er 0,5% bij."},
    12: {"en": "Tells the customer to contact the bank, in the bank's own chat, and does not say how to unblock the PIN.",
         "nl": "Zegt de klant contact op te nemen met de bank, in de chat van de bank zelf, en zegt niet hoe je de pincode deblokkeert."},
}
WORKED = (1, 3, 79)   # the replies of the worked example
WHICH = 48            # the reply of exercise 3


def counts(notes=NOTES) -> dict:
    """How many replies have each kind, in the order the notes first have them."""
    out = {}
    for kinds in notes:
        for kind in kinds:
            out[kind] = out.get(kind, 0) + 1
    return out


COUNTS = NamedDict(counts(), "COUNTS")


def _with(*kinds) -> list:
    return [i for i, k in enumerate(NOTES) if all(x in k for x in kinds)]


def findings() -> dict:
    """The numbers the lesson gives about the course's run and its notes."""
    a = answers()
    replies = [parsed(i) for i in range(len(a))]
    refunds = [i for i, label in enumerate(LABELS) if label == "request_refund"]
    pins = [i for i, label in enumerate(LABELS) if label == "pin_blocked"]
    return {
        "n": len(a),
        "json": sum(isinstance(r, dict) and set(r) == {"topic", "reply"} for r in replies),
        "stopped": sum(x["finish"] == "stop" for x in a),
        "person": sum("person" in r["reply"] for r in replies),
        "noted": sum(bool(k) for k in NOTES),
        "fine": sum(not k for k in NOTES),
        "counts": counts(),
        "one_kind": sum(len(k) == 1 for k in NOTES),
        "refunds": len(refunds),
        "refunds_rate": sum(replies[i]["topic"] == "card_payment_wrong_exchange_rate" for i in refunds),
        "refunds_right": sum(replies[i]["topic"] == "request_refund" for i in refunds),
        "pins": len(pins),
        "pins_rate": sum(replies[i]["topic"] == "card_payment_wrong_exchange_rate" for i in pins),
        "topic_and_other": len(_with("wrong-topic", "other-problem")),
        "rate_topic": sum(r["topic"] == "card_payment_wrong_exchange_rate" for r in replies),
        "rate_asks": sum("date and amount" in replies[i]["reply"] for i in refunds + pins
                         if replies[i]["topic"] == "card_payment_wrong_exchange_rate"),
        "seconds": sum(x["seconds"] for x in a) / len(a),
    }


# --- pictures ---------------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui)"
_MONO = "font-family:var(--ea-font-code)"
_WORDS = {
    "en": {"customer": "Customer", "about": "about", "reply": "Qwen3's answer", "note": "The course's note", "kinds": "Kinds",
           "fine": "no mistake found", "answer": "Answer {i} of {n}",
           "loop": ["read", "note", "sort", "count", "fix the biggest"], "again": "after the fix, read again",
           "loop_title": "Error analysis, then evaluation-driven development",
           "loop_alt": "Five steps in a row, joined by arrows: read the answers, note what goes wrong in each, sort the notes into kinds, count each kind, fix the biggest kind. An arrow from the last step goes back to the first: after the fix, you read again.",
           "table_title": "The course's failure taxonomy of the 120 replies", "kind": "kind", "rule": "what it means",
           "replies": "replies", "none_row": "no mistake found",
           "table_alt": "A table with a row for each kind of mistake the course found in the 120 replies: its name, the rule the course read with, and the number of replies that have it. A reply can have more than one kind.",
           "bars": "Your count_kinds(NOTES)"},
    "nl": {"customer": "Klant", "about": "over", "reply": "Het antwoord van Qwen3", "note": "De notitie van de course", "kinds": "Soorten",
           "fine": "geen fout gevonden", "answer": "Antwoord {i} van {n}",
           "loop": ["lezen", "noteren", "indelen", "tellen", "grootste aanpakken"], "again": "na de verbetering opnieuw lezen",
           "loop_title": "Error analysis, en daarna evaluation-driven development",
           "loop_alt": "Vijf stappen op een rij, verbonden door pijlen: de antwoorden lezen, bij elk noteren wat er misgaat, de notities in soorten indelen, elke soort tellen, de grootste soort aanpakken. Vanaf de laatste stap gaat een pijl terug naar de eerste: na de verbetering lees je opnieuw.",
           "table_title": "De failure taxonomy van de course voor de 120 antwoorden", "kind": "soort", "rule": "wat het betekent",
           "replies": "antwoorden", "none_row": "geen fout gevonden",
           "table_alt": "Een tabel met een rij voor elke soort fout die de course vond in de 120 antwoorden: de naam, de regel waarmee de course las, en het aantal antwoorden met die soort. Een antwoord kan meer dan één soort hebben.",
           "bars": "Jouw count_kinds(NOTES)"},
}


def kind_name(kind: str, lang: str) -> str:
    return KIND[kind][lang] if kind in KIND else kind


def _chip(kind: str, lang: str) -> str:
    return (f'<span style="{_UI};font-size:12px;color:var(--ea-loss-text);border:1px solid var(--ea-loss);border-radius:999px;'
            f'padding:0 0.5rem;margin-right:0.3rem;white-space:nowrap">{html.escape(kind_name(kind, lang))}</span>')


def answer_card(i: int, lang: str, note: bool = True, title: str | None = None) -> str:
    """Reply i: the customer's question with what it is about, Qwen3's answer, and the course's note."""
    w = _WORDS[lang]
    e = html.escape
    rows = (f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2)">{e(w["customer"])} · {e(w["about"])} '
            f'<span style="{_MONO}">{e(LABELS[i])}</span></div>'
            f'<div style="font-size:14px;color:var(--ea-ink);margin:0.15rem 0 0.5rem">{e(QUESTIONS[i])}</div>'
            f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2)">{e(w["reply"])}</div>'
            f'<pre style="{_MONO};font-size:12px;color:var(--ea-ink);background:var(--ea-inset);padding:0.4rem 0.6rem;'
            f'margin:0.15rem 0 0.5rem;white-space:pre-wrap;word-break:break-word">{e(answers()[i]["text"])}</pre>')
    if note:
        kinds = NOTES[i]
        chips = "".join(_chip(k, lang) for k in kinds) if kinds else (
            f'<span style="{_UI};font-size:12px;color:var(--ea-ink-2)">{e(w["fine"])}</span>')
        text = NOTE_TEXT.get(i, {}).get(lang)
        rows += (f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2)">{e(w["note"])}</div>'
                 + (f'<div style="font-size:14px;color:var(--ea-ink);margin:0.15rem 0 0.3rem">{e(text)}</div>' if text else "")
                 + f'<div style="margin-top:0.2rem">{chips}</div>')
    return yours.figure(title or w["answer"].format(i=i + 1, n=len(QUESTIONS)), rows)


def prompt_html(text: str) -> str:
    """A prompt as the model gets it, line for line and wrapped to the page. A ```text block in mo.md would put a blank
    line before every line that starts with "- "."""
    return (f'<pre style="{_MONO};font-size:12px;line-height:1.5;color:var(--ea-ink);background:var(--ea-inset);'
            f'padding:0.6rem 0.8rem;border-radius:var(--ea-radius-small);white-space:pre-wrap;word-break:break-word;'
            f'margin:0.2rem 0 0.8rem">{html.escape(text)}</pre>')


def idea_picture(lang: str) -> str:
    """The loop: read, note, sort, count, fix the biggest, and read again; boxes that wrap on a narrow page."""
    w = _WORDS[lang]
    e = html.escape
    boxes = []
    for k, step in enumerate(w["loop"]):
        last = k == len(w["loop"]) - 1
        border = "var(--ea-loss)" if last else "var(--ea-ink-2)"
        boxes.append(f'<span style="{_UI};font-size:14px;color:var(--ea-ink);border:1.4px solid {border};border-radius:6px;'
                     f'padding:0.35rem 0.8rem;white-space:nowrap">{e(step)}</span>')
    arrow = f'<span aria-hidden="true" style="{_UI};font-size:16px;color:var(--ea-ink-2)">→</span>'
    row = (f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:0.45rem 0.5rem">{arrow.join(boxes)}</div>'
           f'<div style="{_UI};font-size:13px;color:var(--ea-ink-2);margin-top:0.5rem">↻ {e(w["again"])}</div>')
    return yours.figure(w["loop_title"], f'<div role="img" aria-label="{e(w["loop_alt"])}">{row}</div>', alt=w["loop_alt"])


def kinds_picture(lang: str) -> str:
    """The course's failure taxonomy: each kind with its rule and the number of replies that have it, biggest first."""
    w = _WORDS[lang]
    e = html.escape
    c = counts()
    order = sorted(c, key=lambda k: -c[k])
    cell = f'style="{_UI};font-size:13px;color:var(--ea-ink);padding:0.25rem 0.6rem;vertical-align:top;border-top:1px solid var(--ea-line)"'
    head = f'style="{_UI};font-size:12px;font-weight:400;color:var(--ea-ink-2);padding:0.2rem 0.6rem;text-align:left"'
    rows = "".join(f'<tr><td {cell}><span style="white-space:nowrap">{_chip(k, lang)}</span></td><td {cell}>{e(RULE[k][lang])}</td>'
                   f'<td {cell.replace("vertical-align:top", "vertical-align:top;text-align:right;font-weight:600")}>{c[k]}</td></tr>'
                   for k in order)
    rows += (f'<tr><td {cell}><span style="color:var(--ea-ink-2)">{e(w["none_row"])}</span></td><td {cell}></td>'
             f'<td {cell.replace("vertical-align:top", "vertical-align:top;text-align:right")}>{sum(not k for k in NOTES)}</td></tr>')
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%;max-width:44rem">'
             f'<tr><th {head}>{e(w["kind"])}</th><th {head}>{e(w["rule"])}</th><th {head.replace("text-align:left", "text-align:right")}>'
             f'{e(w["replies"])}</th></tr>{rows}</table></div>')
    return yours.figure(w["table_title"], table, alt=w["table_alt"])


def group_options(lang: str) -> dict:
    """The groups of replies to read in the demo: all, each kind biggest first, and those without a mistake."""
    c = counts()
    words = {"en": ("all {n} replies", "{k}: {n}", "no mistake found: {n}", ("reply", "replies")),
             "nl": ("alle {n} antwoorden", "{k}: {n}", "geen fout gevonden: {n}", ("antwoord", "antwoorden"))}[lang]

    def replies(n: int) -> str:
        return f"{n} {words[3][n != 1]}"
    out = {words[0].format(n=len(NOTES)): "all"}
    for k in sorted(c, key=lambda k: -c[k]):
        out[words[1].format(k=kind_name(k, lang), n=replies(c[k]))] = k
    out[words[2].format(n=replies(sum(not k for k in NOTES)))] = "none"
    return out


def group(which: str) -> list:
    """The numbers of the replies in a group of group_options."""
    if which == "all":
        return list(range(len(NOTES)))
    if which == "none":
        return [i for i, k in enumerate(NOTES) if not k]
    return [i for i, k in enumerate(NOTES) if which in k]


# --- exercise 1: count the kinds --------------------------------------------------------------------------------------------

START_COUNT = '''def count_kinds(notes):
    counts = {}
    for kinds in notes:
        for kind in kinds:
            counts[kind] = 1
    return counts
'''
SOLUTION_COUNT = START_COUNT.replace('''            counts[kind] = 1
''', '''            if kind in counts:
                counts[kind] = counts[kind] + 1
            else:
                counts[kind] = 1
''')
_COUNT_NO_START = START_COUNT.replace("counts[kind] = 1", "counts[kind] = counts[kind] + 1")
_COUNT_PER_ANSWER = '''def count_kinds(notes):
    counts = {}
    for kinds in notes:
        for kind in kinds:
            counts[kind] = len(kinds)
    return counts
'''
_COUNT_NOTES = '''def count_kinds(notes):
    counts = {}
    for kinds in notes:
        for kind in kinds:
            if kind in counts:
                counts[kind] = counts[kind] + len(kinds)
            else:
                counts[kind] = len(kinds)
    return counts
'''


def _key_error(e) -> bool:
    return isinstance(e, KeyError)


_MISSING = {"en": "Run the cell with `count_kinds` first.", "nl": "Voer eerst de cel met `count_kinds` uit."}
_CLEARED = {"en": "Your box is empty. Click *Start over* under the box to get the start code back.",
            "nl": "Je vak is leeg. Klik op *Opnieuw beginnen* onder het vak om de begincode terug te krijgen."}

# --- exercise 2: the biggest kind -------------------------------------------------------------------------------------------

START_BIGGEST = '''def biggest(counts):
    best = None
    most = 0
    for kind in counts:
        best = kind
    return best
'''
SOLUTION_BIGGEST = START_BIGGEST.replace('''        best = kind
''', '''        if counts[kind] > most:
            best = kind
            most = counts[kind]
''')
_BIGGEST_GE = SOLUTION_BIGGEST.replace("counts[kind] > most", "counts[kind] >= most")
_BIGGEST_NUMBER = SOLUTION_BIGGEST.replace("return best", "return most")
_BIGGEST_NO_MOST = SOLUTION_BIGGEST.replace('''            best = kind
            most = counts[kind]
''', '''            best = kind
''')
_BIGGEST_MAX = '''def biggest(counts):
    return max(counts)
'''


EXERCISES = {
    "count_kinds": FunctionBox(
        name="count_kinds",
        picture=yours.bars(0, title={"en": _WORDS["en"]["bars"], "nl": _WORDS["nl"]["bars"]}),
        function="count_kinds",
        start=START_COUNT,
        solution=SOLUTION_COUNT,
        visible=[(NOTES,)],
        hidden=[([["made-up"], [], ["made-up", "admits"], ["admits"], ["made-up"]],), ([[], []],), ([["too-long"]],),
                ([["wrong-topic", "other-problem"], ["wrong-topic"], ["other-problem"], []],)],
        mistakes=[
            Mistake("no-start", error=_key_error,
                    en="With `{call}`, your function stops with a `KeyError`: the first time a kind comes along, it is not in `counts` yet, so `counts[kind]` has no value to add 1 to. Start a new kind at 1, and add 1 to a kind that is already there.",
                    nl="Met `{call}` stopt je functie met een `KeyError`: de eerste keer dat een soort langskomt, staat die nog niet in `counts`, dus `counts[kind]` heeft geen waarde om 1 bij op te tellen. Begin een nieuwe soort bij 1, en tel 1 op bij een soort die er al staat."),
            Mistake("per-answer", wrong=_COUNT_PER_ANSWER,
                    en="`{call}` gives `{got}`: you put in the number of kinds of one reply. Count replies: each time a kind comes along, its count goes up by 1.",
                    nl="`{call}` geeft `{got}`: je zet er het aantal soorten van één antwoord in. Tel antwoorden: elke keer dat een soort langskomt, gaat zijn aantal 1 omhoog."),
            Mistake("notes", wrong=_COUNT_NOTES,
                    en="`{call}` gives `{got}`: you add the number of kinds of the reply, `len(kinds)`. Each reply with a kind counts once for that kind: add 1.",
                    nl="`{call}` geeft `{got}`: je telt het aantal soorten van het antwoord op, `len(kinds)`. Elk antwoord met een soort telt één keer voor die soort: tel 1 op."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "Your `count_kinds` still puts 1 in for every kind, however often it comes along. When a kind is already in `counts`, add 1 to it instead.",
                      "nl": "Je `count_kinds` zet nog voor elke soort 1 neer, hoe vaak die ook langskomt. Staat een soort al in `counts`, tel er dan 1 bij op."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: for each kind, the number of replies that have it.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: voor elke soort het aantal antwoorden dat die heeft."},
            "almost": {"en": "Right for the course's notes. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work when a reply has two kinds, or when no reply has a kind?",
                       "nl": "Goed voor de notities van de course. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Werkt je functie als een antwoord twee soorten heeft, of als geen enkel antwoord een soort heeft?"},
            "no_function": {"en": "Your box does not make a function `count_kinds`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `count_kinds`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `count_kinds` is right in all {n} cases: the failure taxonomy of the course's notes, with the number of replies of each kind.",
                        "nl": "Je `count_kinds` klopt in alle {n} gevallen: de failure taxonomy van de notities van de course, met het aantal antwoorden van elke soort."},
        },
    ),
    "biggest": FunctionBox(
        name="biggest",
        function="biggest",
        start=START_BIGGEST,
        solution=SOLUTION_BIGGEST,
        visible=[(COUNTS,)],
        hidden=[({"made-up": 3, "admits": 5, "elsewhere": 1},), ({"admits": 4, "made-up": 4},), ({"too-long": 1},),
                ({"elsewhere": 2, "admits": 7, "made-up": 7, "too-long": 6},)],
        mistakes=[
            Mistake("ge", wrong=_BIGGEST_GE, state="almost",
                    en="Almost. `{call}` gives `{got}`, and it should give `{want}`: with `>=`, a later kind with the same count takes the place of the first. Use `>`, so that the first one stays.",
                    nl="Bijna. `{call}` geeft `{got}`, en het zou `{want}` moeten zijn: met `>=` neemt een latere soort met hetzelfde aantal de plaats van de eerste in. Gebruik `>`, dan blijft de eerste staan."),
            Mistake("number", wrong=_BIGGEST_NUMBER,
                    en="`{call}` gives `{got}`: that is the count of the biggest kind. Give back the kind itself, `best`.",
                    nl="`{call}` geeft `{got}`: dat is het aantal van de grootste soort. Geef de soort zelf terug, `best`."),
            Mistake("no-most", wrong=_BIGGEST_NO_MOST,
                    en="`{call}` gives `{got}`: `most` stays 0, so every kind with a count above 0 takes the place of the one before. When a kind becomes `best`, make `most` its count too.",
                    nl="`{call}` geeft `{got}`: `most` blijft 0, dus elke soort met een aantal boven 0 neemt de plaats van de vorige in. Maak `most` ook het aantal van een soort als die `best` wordt."),
            Mistake("max", wrong=_BIGGEST_MAX,
                    en="`{call}` gives `{got}`: `max(counts)` compares the names of the kinds as words, and gives the one that comes last in the alphabet. Compare their counts.",
                    nl="`{call}` geeft `{got}`: `max(counts)` vergelijkt de namen van de soorten als woorden, en geeft de naam die het laatst in het alfabet komt. Vergelijk hun aantallen."),
        ],
        texts={
            "missing": {"en": "Run the cell with `biggest` first.", "nl": "Voer eerst de cel met `biggest` uit."},
            "cleared": _CLEARED,
            "empty": {"en": "Your `biggest` gives the last kind in `counts`, whatever its count. Let a kind become `best` only when its count is higher than `most`.",
                      "nl": "Je `biggest` geeft de laatste soort in `counts`, wat zijn aantal ook is. Laat een soort alleen `best` worden als zijn aantal hoger is dan `most`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: the kind with the highest count.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: de soort met het hoogste aantal."},
            "almost": {"en": "Right for `COUNTS`. But `{call}` gives `{got}`, and it should give `{want}`. When two kinds have the same count, the first one counts.",
                       "nl": "Goed voor `COUNTS`. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Als twee soorten hetzelfde aantal hebben, telt de eerste."},
            "no_function": {"en": "Your box does not make a function `biggest`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `biggest`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `biggest` is right in all {n} cases: the kind to fix first.",
                        "nl": "Je `biggest` klopt in alle {n} gevallen: de soort die je als eerste aanpakt."},
        },
    ),
}

REQUIRED = ["count_kinds", "biggest", "which-kind", "fix-first"]

CASES = [
    ("count_kinds", START_COUNT, "empty", None),
    ("count_kinds", "", "empty", None),
    ("count_kinds", SOLUTION_COUNT, "correct", None),
    ("count_kinds", '''def count_kinds(notes):
    counts = {}
    for kinds in notes:
        for kind in kinds:
            counts[kind] = counts.get(kind, 0) + 1
    return counts
''', "correct", None),
    ("count_kinds", _COUNT_NO_START, "error", "no-start"),
    ("count_kinds", _COUNT_PER_ANSWER, "wrong", "per-answer"),
    ("count_kinds", _COUNT_NOTES, "wrong", "notes"),
    ("count_kinds", SOLUTION_COUNT.replace("counts[kind] + 1", "counts[kind] + 2"), "wrong", None),
    ("count_kinds", SOLUTION_COUNT.replace("    return counts\n", "    counts.pop(\"too-long\", None)\n    return counts\n"), "wrong", None),
    ("biggest", START_BIGGEST, "empty", None),
    ("biggest", "", "empty", None),
    ("biggest", SOLUTION_BIGGEST, "correct", None),
    ("biggest", '''def biggest(counts):
    return max(counts, key=counts.get)
''', "correct", None),
    ("biggest", _BIGGEST_GE, "almost", "ge"),
    ("biggest", _BIGGEST_NUMBER, "wrong", "number"),
    ("biggest", _BIGGEST_NO_MOST, "wrong", "no-most"),
    ("biggest", _BIGGEST_MAX, "wrong", "max"),
    ("biggest", SOLUTION_BIGGEST.replace("if counts[kind] > most:", "if most == 0:"), "almost", None),
]
