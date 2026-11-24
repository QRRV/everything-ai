# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-extraction-with-llms: the prompt that asks a chat model to fill five fields for an event,
with an instruction, examples and the text between <text> tags (fields_prompt); the answer read back as a row of the
fields the model filled in (read_fields); and the fields whose value is nowhere in the text (unsupported).

The texts are twelve short notices of events in a town that the course wrote for this lesson, in English and in Dutch,
with the same facts, and two more as the examples of the prompt. The course knows each field of each notice (GOLD):
eight of the 60 fields are left out of the notices on purpose. The answers are the course's run of Qwen3-0.6B, the
chat model of the course, in data/course-extraction-qwen.json (data/tools/record_course_extraction_qwen.py): every
notice in three ways (without examples, with the two examples, and with the two examples and all twelve notices in
one newsletter), with the seeds 0 to 3. Every number in the text and the pictures is computed from that file by the
course (tests/test_nlp_extraction_with_llms.py computes them again)."""
import functools
import html
import json
import re

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake

LANGS = ("en", "nl")
COURSE = "data/course-extraction-qwen.json"
DOWNLOADS = {"Qwen3-0.6B": [("Qwen/Qwen3-0.6B", "model.safetensors"), ("Qwen/Qwen3-0.6B-Base", "vocab.json"),
                            ("Qwen/Qwen3-0.6B-Base", "tokenizer_config.json")]}
WAYS = ("zero", "examples", "long")
SEEDS = 4
MAX_TOKENS = 120
FIELDS = {"en": ["what", "date", "time", "place", "price"], "nl": ["wat", "datum", "tijd", "plaats", "prijs"]}
INSTRUCTION = {
    "en": "Fill in five fields for the event in the text: what (the event), date, time (when it starts), place and price "
          "(of a ticket or entry). Copy each value exactly as it is written in the text. Write - for a field the text "
          "does not give. Answer with one line per field, as field: value.",
    "nl": "Vul vijf velden in voor de activiteit in de tekst: wat (de activiteit), datum, tijd (wanneer het begint), "
          "plaats en prijs (van een kaartje of de toegang). Neem elke waarde letterlijk over uit de tekst. Schrijf - bij "
          "een veld dat niet in de tekst staat. Antwoord met één regel per veld, als veld: waarde.",
}
ONLY = {"en": "Fill in the fields for this event only: ", "nl": "Vul de velden alleen in voor deze activiteit: "}
HEAD = {"en": "What's on in town this spring", "nl": "Uit in de stad dit voorjaar"}

# (text, gold) with gold the five values in the order of FIELDS, None where the text does not give the field.
EXAMPLES = {
    "en": [("The Repair Café is back! Bring a broken lamp, toaster or bike to the community center on Mill Street on "
            "Saturday 7 March, between 10:00 and 13:00. Entry is free.",
            ["Repair Café", "Saturday 7 March", "10:00", "the community center on Mill Street", "free"]),
           ("Open evening at the music school on Thursday 12 March from 19:00: try an instrument and meet the teachers.",
            ["Open evening", "Thursday 12 March", "19:00", "the music school", None])],
    "nl": [("Het Repair Café is terug! Breng op zaterdag 7 maart tussen 10.00 en 13.00 uur een kapotte lamp, "
            "broodrooster of fiets naar het buurthuis aan de Molenstraat. De toegang is gratis.",
            ["Repair Café", "zaterdag 7 maart", "10.00", "het buurthuis aan de Molenstraat", "gratis"]),
           ("Open avond op de muziekschool op donderdag 12 maart vanaf 19.00 uur: probeer een instrument en maak "
            "kennis met de docenten.",
            ["Open avond", "donderdag 12 maart", "19.00", "de muziekschool", None])],
}
NOTICES = {
    "en": [
        ("Book market in the town library on Saturday 14 March, from 10:00 to 16:00. Entry is free.",
         ["Book market", "Saturday 14 March", "10:00", "the town library", "free"]),
        ("Swimming lessons for adults start on Monday 16 March at 19:30. Ten lessons cost €60.",
         ["Swimming lessons for adults", "Monday 16 March", "19:30", None, "€60"]),
        ("Night of the Stars: look through a big telescope at the observatory on the hill. Friday 20 March, from 21:00. "
         "Bring warm clothes.",
         ["Night of the Stars", "Friday 20 March", "21:00", "the observatory on the hill", None]),
        ("Flea market on Market Square, Sunday 22 March. The stalls open at 9:00. Entry €2; parking in the garage costs €4.",
         ["Flea market", "Sunday 22 March", "9:00", "Market Square", "€2"]),
        ("Film evening on Wednesday 25 March: a silent film with live piano music in the old cinema on Bridge Street. "
         "Tickets €7.50.",
         ["Film evening", "Wednesday 25 March", None, "the old cinema on Bridge Street", "€7.50"]),
        ("Spring cleaning on Saturday 28 March at 10:00: we pick up litter together in Willow Park. Gloves and bags are there.",
         ["Spring cleaning", "Saturday 28 March", "10:00", "Willow Park", None]),
        ("Chess tournament for all ages in the sports hall on School Lane. Saturday 4 April, the first round starts at "
         "9:30. Entry €5, players under 12 free.",
         ["Chess tournament", "Saturday 4 April", "9:30", "the sports hall on School Lane", "€5"]),
        ("Cooking workshop: make your own pasta. Tuesday 7 April, 18:00, in the kitchen of the community center. Places "
         "are limited, so sign up soon.",
         ["Cooking workshop", "Tuesday 7 April", "18:00", "the kitchen of the community center", None]),
        ("Garden tour on Sunday 12 April: twelve gardens in the old town open their gates. Tickets cost €10 and come "
         "with a map.",
         ["Garden tour", "Sunday 12 April", None, "the old town", "€10"]),
        ("Lecture on the history of the canal in the town hall, Thursday 16 April at 20:00. Entry €3, coffee included.",
         ["Lecture on the history of the canal", "Thursday 16 April", "20:00", "the town hall", "€3"]),
        ("Open day at the fire station on Station Road: see the trucks and try the hose. Saturday 18 April, 11:00 to 15:00.",
         ["Open day", "Saturday 18 April", "11:00", "the fire station on Station Road", None]),
        ("Jazz in the park: three bands play on the open-air stage in Willow Park, from 14:00. Tickets €12 at the gate.",
         ["Jazz in the park", None, "14:00", "the open-air stage in Willow Park", "€12"]),
    ],
    "nl": [
        ("Boekenmarkt in de bibliotheek op zaterdag 14 maart, van 10.00 tot 16.00 uur. De toegang is gratis.",
         ["Boekenmarkt", "zaterdag 14 maart", "10.00", "de bibliotheek", "gratis"]),
        ("Zwemles voor volwassenen begint op maandag 16 maart om 19.30 uur. Tien lessen kosten € 60.",
         ["Zwemles voor volwassenen", "maandag 16 maart", "19.30", None, "€ 60"]),
        ("Sterrennacht: kijk door een grote telescoop in de sterrenwacht op de heuvel. Vrijdag 20 maart, vanaf 21.00 uur. "
         "Trek warme kleren aan.",
         ["Sterrennacht", "Vrijdag 20 maart", "21.00", "de sterrenwacht op de heuvel", None]),
        ("Rommelmarkt op de Markt, zondag 22 maart. De kramen gaan om 9.00 uur open. Toegang € 2; parkeren in de garage "
         "kost € 4.",
         ["Rommelmarkt", "zondag 22 maart", "9.00", "de Markt", "€ 2"]),
        ("Filmavond op woensdag 25 maart: een stomme film met live pianomuziek in de oude bioscoop aan de Brugstraat. "
         "Kaartjes € 7,50.",
         ["Filmavond", "woensdag 25 maart", None, "de oude bioscoop aan de Brugstraat", "€ 7,50"]),
        ("Voorjaarsschoonmaak op zaterdag 28 maart om 10.00 uur: samen ruimen we zwerfafval op in het Wilgenpark. "
         "Handschoenen en zakken liggen klaar.",
         ["Voorjaarsschoonmaak", "zaterdag 28 maart", "10.00", "het Wilgenpark", None]),
        ("Schaaktoernooi voor alle leeftijden in de sporthal aan de Schoollaan. Zaterdag 4 april, de eerste ronde begint "
         "om 9.30 uur. Inschrijven € 5, spelers onder de 12 gratis.",
         ["Schaaktoernooi", "Zaterdag 4 april", "9.30", "de sporthal aan de Schoollaan", "€ 5"]),
        ("Kookworkshop: maak je eigen pasta. Dinsdag 7 april, 18.00 uur, in de keuken van het buurthuis. Er zijn weinig "
         "plaatsen, dus schrijf je snel in.",
         ["Kookworkshop", "Dinsdag 7 april", "18.00", "de keuken van het buurthuis", None]),
        ("Tuinroute op zondag 12 april: twaalf tuinen in de oude binnenstad gaan open. Kaartjes kosten € 10, met een "
         "plattegrond erbij.",
         ["Tuinroute", "zondag 12 april", None, "de oude binnenstad", "€ 10"]),
        ("Lezing over de geschiedenis van de gracht in het stadhuis, donderdag 16 april om 20.00 uur. Toegang € 3, "
         "koffie inbegrepen.",
         ["Lezing over de geschiedenis van de gracht", "donderdag 16 april", "20.00", "het stadhuis", "€ 3"]),
        ("Open dag bij de brandweerkazerne aan de Stationsweg: bekijk de wagens en probeer de brandslang. Zaterdag "
         "18 april, van 11.00 tot 15.00 uur.",
         ["Open dag", "Zaterdag 18 april", "11.00", "de brandweerkazerne aan de Stationsweg", None]),
        ("Jazz in het park: drie bands spelen op het openluchtpodium in het Wilgenpark, vanaf 14.00 uur. Kaartjes € 12 "
         "bij de ingang.",
         ["Jazz in het park", None, "14.00", "het openluchtpodium in het Wilgenpark", "€ 12"]),
    ],
}


def answer_text(gold: list, lang: str) -> str:
    """The answer the course wants for a notice: one line per field, as field: value, with - for a field left out."""
    return "\n".join(f"{f}: {'-' if v is None else v}" for f, v in zip(FIELDS[lang], gold))


SHOTS = {lang: [(t, answer_text(g, lang)) for t, g in EXAMPLES[lang]] for lang in LANGS}


def fields_prompt(instruction: str, examples: list, text: str) -> str:
    """The course's prompt: the instruction, then every example as its text between <text> tags with its answer
    under it, then the text, each part after a blank line."""
    prompt = instruction + "\n\n"
    for example, answer in examples:
        prompt = prompt + "<text>\n" + example + "\n</text>\n" + answer + "\n\n"
    return prompt + "<text>\n" + text + "\n</text>"


def newsletter(lang: str) -> str:
    """All twelve notices in one text, under a heading, each after a blank line."""
    return HEAD[lang] + "\n\n" + "\n\n".join(t for t, _ in NOTICES[lang])


def prompt(way: str, lang: str, i: int) -> str:
    """The prompt of notice i (counting from 0) in one of the three ways: zero (no examples), examples (the two
    examples) or long (the two examples, the whole newsletter as the text, and a line that names the event)."""
    text, gold = NOTICES[lang][i]
    if way == "zero":
        return fields_prompt(INSTRUCTION[lang], [], text)
    if way == "examples":
        return fields_prompt(INSTRUCTION[lang], SHOTS[lang], text)
    if way == "long":
        return fields_prompt(INSTRUCTION[lang], SHOTS[lang], newsletter(lang)) + "\n\n" + ONLY[lang] + gold[0]
    raise ValueError(way)


def read_fields(answer: str) -> dict:
    """The fields the model filled in: from each line field: value of its answer, the value without spaces around it,
    except where it is -."""
    row = {}
    for field, value in re.findall(r"(\w+): (.+)", answer):
        value = value.strip()
        if value != "-":
            row[field] = value
    return row


def unsupported(row: dict, text: str) -> list:
    """The fields of a row whose value is not in the text, in lower case, in the order of the row."""
    fields = []
    for field, value in row.items():
        if value.lower() not in text.lower():
            fields.append(field)
    return fields


# --- the course's run ----------------------------------------------------------------------------------------------------

@functools.cache
def course() -> dict:
    """The course's run of Qwen3-0.6B, data/course-extraction-qwen.json."""
    from everything_ai import paths
    return json.loads((paths.ROOT / COURSE).read_text(encoding="utf-8"))


def answer(lang: str, way: str, i: int, seed: int = 0) -> dict:
    """Qwen3's answer to notice i in one way with one seed: its text, tokens, prompt tokens, seconds and finish."""
    return course()["answers"][lang][way][i]["seeds"][seed]


def answer_lines(text: str) -> dict:
    """Every line field: value of an answer, the course's way of reading it, with a - kept as None."""
    out = {}
    for field, value in re.findall(r"(\w+): (.+)", text):
        value = value.strip()
        out[field] = None if value == "-" else value
    return out


def gold(lang: str, i: int) -> dict:
    """What notice i really gives: each field with its value, None for a field the notice leaves out."""
    return dict(zip(FIELDS[lang], NOTICES[lang][i][1]))


def _plain(v: str) -> str:
    return v.lower().strip().rstrip(".").strip()


def verdict(lang: str, field: str, got, want) -> str:
    """How the course sees one field of an answer: right (the value holds what the notice gives), part (it is part of
    it, such as 14 March for Saturday 14 March), wrong (another value), made-up (a value for a field the notice leaves
    out), missed (- for a field the notice gives), left-out (- for a field the notice leaves out, as it should) or
    unread (no line field: value for it)."""
    if got is _UNREAD:
        return "unread"
    if want is None:
        return "left-out" if got is None else "made-up"
    if got is None:
        return "missed"
    g, w = _plain(got), _plain(want)
    if w in g:
        return "right"
    if g and g in w:
        return "part"
    return "wrong"


_UNREAD = object()


def verdicts(lang: str, way: str, i: int, seed: int = 0) -> dict:
    """The verdict of each of the five fields of one answer."""
    lines = answer_lines(answer(lang, way, i, seed)["text"])
    return {f: verdict(lang, f, lines.get(f, _UNREAD), w) for f, w in gold(lang, i).items()}


def flagged(lang: str, way: str, i: int, seed: int = 0, against: str = "notice") -> list:
    """The fields of one answer that the course's check flags: their value is not in the notice (or the newsletter)."""
    row = read_fields(answer(lang, way, i, seed)["text"])
    return unsupported(row, NOTICES[lang][i][0] if against == "notice" else newsletter(lang))


def copied_example(lang: str, way: str, i: int, seed: int = 0) -> bool:
    """Whether the answer fills in an example of the prompt instead of the notice: its what is the what of an example
    and not of the notice."""
    what = answer_lines(answer(lang, way, i, seed)["text"]).get(FIELDS[lang][0])
    if not what:
        return False
    shots = [_plain(g[0]) for _, g in EXAMPLES[lang]]
    return _plain(what) in shots and _plain(what) != _plain(NOTICES[lang][i][1][0])


def tally(lang: str, way: str, seeds=None) -> dict:
    """What the course counts over the twelve notices and the seeds of one way: the answers, the answers with all five
    lines, every verdict, the made-up and wrong values the check flags, the right ones it flags, and the answers that
    fill in an example."""
    seeds = range(SEEDS) if seeds is None else seeds
    out = {"answers": 0, "all_five": 0, "copied": 0, "flagged": 0, "made_up_flagged": 0, "wrong_flagged": 0,
           "right_flagged": 0, "values": 0, "length": 0}
    for v in ("right", "part", "wrong", "made-up", "missed", "left-out", "unread"):
        out[v] = 0
    for seed in seeds:
        for i in range(len(NOTICES[lang])):
            out["answers"] += 1
            out["length"] += answer(lang, way, i, seed)["finish"] == "length"
            vs = verdicts(lang, way, i, seed)
            out["all_five"] += all(v != "unread" for v in vs.values())
            out["copied"] += copied_example(lang, way, i, seed)
            flags = set(flagged(lang, way, i, seed))
            row = read_fields(answer(lang, way, i, seed)["text"])
            for f, v in vs.items():
                out[v] += 1
                if f in row:
                    out["values"] += 1
                    out["flagged"] += f in flags
                    out["made_up_flagged"] += v == "made-up" and f in flags
                    out["wrong_flagged"] += v == "wrong" and f in flags
                    out["right_flagged"] += v in ("right", "part") and f in flags
    return out


# --- how the pictures write things ---------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);margin-top:0.45rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.84rem")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap"
_CODE = ("font-family:var(--ea-font-code);font-size:0.8rem;line-height:1.45;white-space:pre-wrap;overflow-wrap:anywhere;"
         "margin:0;padding:0.35rem 0.5rem;border-radius:var(--ea-radius-small);border:1px solid var(--ea-line)")
_PART = "font-family:var(--ea-font-ui);font-size:0.72rem;color:var(--ea-ink-2);margin:0.35rem 0 0.1rem"
_DATA = "color:var(--ea-data-text)"
_PRED = "color:var(--ea-pred-text)"
_FAINT = "color:var(--ea-ink-2)"
_CHIP = ("display:inline-block;margin:0.08rem 0.2rem 0.08rem 0;padding:0 0.3rem;border-radius:3px;"
         "font-family:var(--ea-font-code);font-size:0.78rem;line-height:1.5")
_CHIP_PRED = _CHIP + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
_CHIP_DATA = _CHIP + ";background:var(--ea-data-wash);color:var(--ea-data-text)"
_CHIP_LOSS = _CHIP + ";background:var(--ea-loss-wash);color:var(--ea-loss-text);font-weight:600"
_CHIP_NONE = _CHIP + ";border:1px dashed var(--ea-line);color:var(--ea-ink-3)"
_WRONG = "color:var(--ea-loss-text);font-weight:600"


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
    """A whole number with a separator for thousands: 1,234 in English, 1.234 in Dutch."""
    out = f"{round(n):,}"
    return out.replace(",", ".") if lang == "nl" else out


def _dec(x: float, lang: str, digits: int = 1) -> str:
    out = f"{round(float(x), digits) + 0.0:.{digits}f}"
    return out.replace(".", ",") if lang == "nl" else out


def _chip(text_: str, style: str) -> str:
    return f'<span style="{style}">{html.escape(text_)}</span>'


def _code(text_: str, style: str = _DATA) -> str:
    return f'<pre style="{_CODE};{style}">{html.escape(text_)}</pre>'


def _table(rows_: list, head: list, numbers_: tuple = ()) -> str:
    """A small table; rows hold HTML that is already escaped; numbers_ are the columns set to the right."""
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers_ else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{_NUMBER if j in numbers_ else ""}">{c}</td>'
                                    for j, c in enumerate(r)) + "</tr>" for r in rows_)
    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;margin:0.3rem 0 0.2rem">'
            f"<thead><tr>{top}</tr></thead><tbody>{body}</tbody></table></div>")


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

# The notice the pictures under exercises 1 and 2 and exercise 4 look at: Night of the Stars, with no price.
STARS = 2
SWIM = 1   # the notice of the idea and the worked example: Swimming lessons for adults, with no place


# --- the solutions, and wrong versions of them -------------------------------------------------------------------------

_FP_HEAD = "def fields_prompt(instruction, examples, text):\n    prompt = instruction + \"\\n\\n\"\n    for example, answer in examples:\n"
_FP_SHOT = '        prompt = prompt + "<text>\\n" + example + "\\n</text>\\n" + answer + "\\n\\n"\n'
_FP_END = '    return prompt + "<text>\\n" + text + "\\n</text>"\n'


def _fp(shot: str = _FP_SHOT, end: str = _FP_END, head: str = _FP_HEAD) -> str:
    return head + shot + end


_FP = _fp()
_FP_START = _FP_HEAD + "        ...\n    return ...\n"

_RF_HEAD = "import re\n\n\ndef read_fields(answer):\n    row = {}\n"
_RF_LOOP = '    for field, value in re.findall(r"(\\w+): (.+)", answer):\n'
_RF_BODY = '        value = value.strip()\n        if value != "-":\n            row[field] = value\n'


def _rf(loop: str = _RF_LOOP, body: str = _RF_BODY, end: str = "    return row\n") -> str:
    return _RF_HEAD + loop + body + end


_RF = _rf()
_RF_START = _RF_HEAD + "    for field, value in re.findall(..., answer):\n        ...\n    return row\n"

_US_HEAD = "def unsupported(row, text):\n    fields = []\n    for field, value in row.items():\n"
_US_TEST = "        if value.lower() not in text.lower():\n"


def _us(test: str = _US_TEST, add: str = "            fields.append(field)\n", end: str = "    return fields\n") -> str:
    return _US_HEAD + test + add + end


_US = _us()
_US_START = _US_HEAD + "        ...\n    return fields\n"


def _course(code: str, name: str):
    space: dict = {}
    exec(compile(code, f"<course {name}>", "exec"), space)
    return space[name]



# --- the picture under exercise 1: your prompt, and what Qwen3 answered to it ---------------------------------------------

_PROMPT_TEXT = {
    "en": {"title": "Your fields_prompt for Night of the Stars",
           "zero": "fields_prompt(instruction, [], text): without examples, {n} characters",
           "examples": "fields_prompt(instruction, examples, text): with the two examples, {n} characters",
           "answer": "What Qwen3 wrote after this prompt, seed 0, from the course's run:",
           "differs": "Line {k} of your prompt differs from the course's. The course has:",
           "longer": "Your prompt goes on after the end of the course's prompt.",
           "shorter": "The course's prompt goes on after the end of yours, with:",
           "note_right": "Gray: your prompt, what Qwen3 reads, with the notice the course wrote. Blue: what Qwen3 wrote after it, from the course's run.",
           "note_wrong": "Gray: your prompt. Purple: the first line that differs from the course's prompt; under it, in gray, the course's line.",
           "note_mixed": "Gray: your prompt, and under a line that differs, the course's line. Blue: what Qwen3 wrote after the prompt that is right. Purple: the first line that differs from the course's prompt.",
           "kind": "a text"},
    "nl": {"title": "Je fields_prompt voor Sterrennacht",
           "zero": "fields_prompt(instruction, [], text): zonder voorbeelden, {n} tekens",
           "examples": "fields_prompt(instruction, examples, text): met de twee voorbeelden, {n} tekens",
           "answer": "Wat Qwen3 na deze prompt schreef, seed 0, uit de run van de course:",
           "differs": "Regel {k} van je prompt is anders dan bij de course. De course heeft:",
           "longer": "Je prompt gaat nog door na het eind van de prompt van de course.",
           "shorter": "De prompt van de course gaat nog door na het eind van de jouwe, met:",
           "note_right": "Grijs: je prompt, wat Qwen3 leest, met de aankondiging die de course schreef. Blauw: wat Qwen3 erna schreef, uit de run van de course.",
           "note_wrong": "Grijs: je prompt. Paars: de eerste regel die anders is dan in de prompt van de course; eronder, in grijs, de regel van de course.",
           "note_mixed": "Grijs: je prompt, en onder een regel die anders is de regel van de course. Blauw: wat Qwen3 schreef na de prompt die klopt. Paars: de eerste regel die anders is dan in de prompt van de course.",
           "kind": "een tekst"},
}


def _first_difference(mine: str, want: str):
    """The first line (counting from 1) where two texts differ, or None when they are the same."""
    a, b = mine.split("\n"), want.split("\n")
    for k in range(max(len(a), len(b))):
        if k >= len(a) or k >= len(b) or a[k] != b[k]:
            return k + 1
    return None


def _diff_block(mine: str, want: str, w: dict) -> str:
    """Your prompt up to and with its first line that differs from the course's, that line in purple, then the course's."""
    k = _first_difference(mine, want)
    a, b = mine.split("\n"), want.split("\n")
    before = "\n".join(a[:k - 1])
    out = f'<pre style="{_CODE};{_DATA}">{html.escape(before)}'
    if k <= len(a):
        out += ("\n" if before else "") + f'<span style="{_WRONG}">{html.escape(a[k - 1]) or "⏎"}</span>'
    out += "</pre>"
    if k <= len(b) and k <= len(a):
        out += _label(w["differs"].format(k=k)) + _code(b[k - 1] or "⏎")
    elif k > len(b):
        out += _label(w["longer"])
    else:
        out += _label(w["shorter"]) + _code(b[k - 1] or "⏎")
    return out


def prompt_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _PROMPT_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    text_ = NOTICES[lang][STARS][0]
    mine = _call(lambda: {way: fn(INSTRUCTION[lang], list(shots), text_) for way, shots in (("zero", []), ("examples", SHOTS[lang]))})
    if mine is _SLOW:
        return _empty(w, "slow")
    if mine is _STOPPED:
        return _empty(w, "stopped")
    if not all(isinstance(v, str) for v in mine.values()):
        return _empty(w, "odd")
    parts, rights = [], []
    for way in ("zero", "examples"):
        want = prompt(way, lang, STARS)
        parts.append(_label(w[way].format(n=_num(len(mine[way]), lang))))
        rights.append(mine[way] == want)
        if mine[way] == want:
            parts.append(_code(mine[way]) + _label(w["answer"]) + _code(answer(lang, way, STARS)["text"], _PRED))
        else:
            parts.append(_diff_block(mine[way], want, w))
    parts.append(_note(w["note_right" if all(rights) else "note_mixed" if any(rights) else "note_wrong"]))
    return yours.figure(w["title"], "".join(parts), alt=w["title"])


# --- the picture under exercise 2: Qwen3's answers read by your read_fields -----------------------------------------------

# The answers the picture shows, seed 0: one without examples (in English its form went wrong, in Dutch it repeats
# the instruction for the place), and one with examples.
READ_PICKS = {"en": [("zero", 6), ("examples", STARS)], "nl": [("zero", SWIM), ("examples", STARS)]}

_READ_TEXT = {
    "en": {"title": "Qwen3's answers, read by your read_fields",
           "way": {"zero": "without examples", "examples": "with the two examples", "long": "with the newsletter"},
           "pick": "Notice {i}, {way}, seed 0: the answer, then your row",
           "head": ["", "Rows with all five values", "Values read", "Fields with -"],
           "course": "the course's read_fields gives {want}",
           "empty_row": "an empty row",
           "note": "Blue: what Qwen3 wrote, from the course's run, and the values your function reads from it. Purple: a value, or a count in the table, that differs from what the course's read_fields gives. The table counts the twelve answers of seed 0 in each way.",
           "kind": "a dictionary"},
    "nl": {"title": "De antwoorden van Qwen3, gelezen door je read_fields",
           "way": {"zero": "zonder voorbeelden", "examples": "met de twee voorbeelden", "long": "met de nieuwsbrief"},
           "pick": "Aankondiging {i}, {way}, seed 0: het antwoord, dan je rij",
           "head": ["", "Rijen met alle vijf waarden", "Gelezen waarden", "Velden met -"],
           "course": "de read_fields van de course geeft {want}",
           "empty_row": "een lege rij",
           "note": "Blauw: wat Qwen3 schreef, uit de run van de course, en de waarden die je functie eruit leest. Paars: een waarde, of een aantal in de tabel, die anders is dan wat de read_fields van de course geeft. De tabel telt de twaalf antwoorden van seed 0 in elke manier.",
           "kind": "een dictionary"},
}


def _row_chips(row: dict, want: dict) -> str:
    """The fields of a row as chips field: value, purple where the row differs from the course's."""
    chips = [_chip(f"{f}: {v}", _CHIP_PRED if want.get(f) == v else _CHIP_LOSS) for f, v in row.items()]
    return "".join(chips)


def read_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _READ_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    texts = {way: [answer(lang, way, i)["text"] for i in range(len(NOTICES[lang]))] for way in ("zero", "examples")}
    mine = _call(lambda: {way: [fn(t) for t in ts] for way, ts in texts.items()})
    if mine is _SLOW:
        return _empty(w, "slow")
    if mine is _STOPPED:
        return _empty(w, "stopped")
    if not all(isinstance(r, dict) for rows in mine.values() for r in rows):
        return _empty(w, "odd")
    parts = []
    for way, i in READ_PICKS[lang]:
        row, want = mine[way][i], read_fields(texts[way][i])
        parts.append(_label(w["pick"].format(i=i + 1, way=w["way"][way])))
        parts.append(_code(texts[way][i], _PRED))
        shown = _row_chips(row, want) if row else _chip(w["empty_row"], _CHIP_LOSS if want else _CHIP_NONE)
        if row != want:
            shown += _note(w["course"].format(want=_value(want)))
        parts.append(f'<div style="margin:0.2rem 0 0.4rem">{shown}</div>')
    rows_ = []
    for way in ("zero", "examples"):
        wants = [read_fields(t) for t in texts[way]]
        cells = [html.escape(w["way"][way])]
        for count in (lambda rs: sum(all(f in r for f in FIELDS[lang]) for r in rs),
                      lambda rs: sum(len(r) for r in rs),
                      lambda rs: sum(sum(1 for f in FIELDS[lang] if answer_lines(t).get(f, "") is None and f not in r)
                                     for r, t in zip(rs, texts[way]))):
            got, ok = _safe_count(count, mine[way]), _safe_count(count, wants)
            cells.append(f'<span style="{"" if got == ok else _WRONG}">{got if got is not None else "?"}</span>')
        rows_.append(cells)
    parts.append(_table(rows_, w["head"], numbers_=(1, 2, 3)))
    parts.append(_note(w["note"]))
    return yours.figure(w["title"], "".join(parts), alt=w["title"])


def _safe_count(count, rows):
    try:
        return int(count(rows))
    except Exception:   # noqa: BLE001   a row of the learner's that cannot be counted
        return None


# --- the picture under exercise 3: the values your unsupported flags ------------------------------------------------------

_FLAG_TEXT = {
    "en": {"title": "Your unsupported on Qwen3's answers with the two examples",
           "head": ["Notice"],
           "summary": "Your function flags {flags} of the {values} values. The course knows what each notice gives: of the {missing} fields that the notices leave out, Qwen3 filled in {made_up}, and your function flags {made_up_flagged} of those. Values that hold the right one but are written differently: {right_flagged} flagged. Wrong values that are somewhere in the notice: {wrong_passed} let through.",
           "differs": "The course's unsupported flags {want}.",
           "note": "Blue: a value Qwen3 wrote, from the course's run, seed 0, that your function does not flag. Purple: a value your function flags. Dashed: a field Qwen3 filled in with -. Gray: the name of the event, as the notice gives it.",
           "kind": "a list"},
    "nl": {"title": "Je unsupported op de antwoorden van Qwen3 met de twee voorbeelden",
           "head": ["Aankondiging"],
           "summary": "Je functie markeert {flags} van de {values} waarden. De course weet wat elke aankondiging geeft: van de {missing} velden die niet in de aankondigingen staan, vulde Qwen3 er {made_up} in, en je functie markeert daarvan {made_up_flagged}. Waarden met de goede waarde erin, maar anders geschreven: {right_flagged} gemarkeerd. Foute waarden die ergens in de aankondiging staan: {wrong_passed} doorgelaten.",
           "differs": "De unsupported van de course markeert {want}.",
           "note": "Blauw: een waarde die Qwen3 schreef, uit de run van de course, seed 0, en die je functie niet markeert. Paars: een waarde die je functie markeert. Gestippeld: een veld dat Qwen3 met - invulde. Grijs: de naam van de activiteit, zoals in de aankondiging.",
           "kind": "een lijst"},
}


def flag_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _FLAG_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    n = len(NOTICES[lang])
    rows = [read_fields(answer(lang, "examples", i)["text"]) for i in range(n)]
    mine = _call(lambda: [fn(dict(r), NOTICES[lang][i][0]) for i, r in enumerate(rows)])
    if mine is _SLOW:
        return _empty(w, "slow")
    if mine is _STOPPED:
        return _empty(w, "stopped")
    if not all(isinstance(m, list) for m in mine):
        return _empty(w, "odd")
    body, count = [], {"flags": 0, "values": 0, "missing": 0, "made_up": 0, "made_up_flagged": 0, "right_flagged": 0,
                       "wrong_passed": 0}
    differs = []
    for i, (row, flags) in enumerate(zip(rows, mine)):
        vs = verdicts(lang, "examples", i)
        lines = answer_lines(answer(lang, "examples", i)["text"])
        cells = [f'<span style="{_DATA}">{html.escape(NOTICES[lang][i][1][0])}</span>']
        for f in FIELDS[lang]:
            if f in row:
                cells.append(_chip(row[f], _CHIP_LOSS if f in flags else _CHIP_PRED))
                count["values"] += 1
                count["flags"] += f in flags
                count["made_up_flagged"] += vs[f] == "made-up" and f in flags
                count["right_flagged"] += vs[f] in ("right", "part") and f in flags
                count["wrong_passed"] += vs[f] in ("wrong", "made-up") and f not in flags
            else:
                cells.append(_chip("-", _CHIP_NONE) if lines.get(f, "") is None else "")
            count["missing"] += vs[f] in ("made-up", "left-out", "unread") and gold(lang, i)[f] is None
            count["made_up"] += vs[f] == "made-up"
        body.append(cells)
        want = unsupported(row, NOTICES[lang][i][0])
        if sorted(map(str, flags)) != sorted(want):
            differs.append((i, want))
    parts = [_table(body, w["head"] + FIELDS[lang])]
    parts.append(f'<div style="{_UI}">{html.escape(w["summary"].format(**count))}</div>')
    for i, want in differs[:2]:
        parts.append(_note(f'{NOTICES[lang][i][1][0]}: ' + w["differs"].format(want=_value(want))))
    parts.append(_note(w["note"]))
    return yours.figure(w["title"], "".join(parts), alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------------

class _Val:
    """A test value that the check's messages write by its name, such as `shots_en`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value, memo)


def _value(v) -> str:
    """A value the way code writes it, with double quotes, and a long text as its start and its length."""
    if isinstance(v, bool) or v is None:
        return repr(v)
    if isinstance(v, str):
        if len(v) > 60:
            return _value(v[:40])[:-1] + f'…" ({len(v)} characters)'
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    if isinstance(v, tuple):
        return "(" + ", ".join(_value(x) for x in v) + ("," if len(v) == 1 else "") + ")"
    if isinstance(v, list):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_value(k)}: {_value(x)}" for k, x in v.items()) + "}"
    return repr(v)


def _show(v):
    """A value for a message; a long text as its start and how many characters it has, in each language."""
    if isinstance(v, str) and len(v) > 60:
        start = _value(v[:40])[:-1]
        return {"en": f'{start}…", a text of {len(v):,} characters', "nl": f'{start}…", een tekst van {_num(len(v), "nl")} tekens'}
    return _value(v)


def _written(v) -> _Val:
    """A small test value that the messages write the way the lesson writes it, with double quotes."""
    return _Val(_value(v), v)


_DOTS = Mistake("dots-left", matches=lambda got, text_: any(line.split("#", 1)[0].count("...")
                                                            for line in text_.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang de `...` door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# The small cases the exercise texts show, in English and in Dutch.
TOY_INSTRUCTION = {"en": "Fill in two fields: what and date.", "nl": "Vul twee velden in: wat en datum."}
TOY_SHOTS = {"en": [("Quiz in the café on Friday 6 March.", "what: Quiz\ndate: Friday 6 March")],
             "nl": [("Quiz in het café op vrijdag 6 maart.", "wat: Quiz\ndatum: vrijdag 6 maart")]}
TOY_TEXT = {"en": "Bingo on Monday 9 March.", "nl": "Bingo op maandag 9 maart."}
TOY_ANSWER = {"en": "what: Bingo  \ndate: Monday 9 March  \nprice: -", "nl": "wat: Bingo  \ndatum: maandag 9 maart  \nprijs: -"}
TOY_ROW = {"en": {"what": "Bingo", "date": "Monday 9 March", "price": "free"},
           "nl": {"wat": "Bingo", "datum": "maandag 9 maart", "prijs": "gratis"}}


def _toy_prompt(lang):
    return (_written(TOY_INSTRUCTION[lang]), _written(TOY_SHOTS[lang]), _written(TOY_TEXT[lang]))


def _real(name: str, value) -> _Val:
    return _Val(name, value)


_INSTR = {lang: _real(f"instruction_{lang}", INSTRUCTION[lang]) for lang in LANGS}
_SHOTS = {lang: _real(f"examples_{lang}", SHOTS[lang]) for lang in LANGS}
_STARS_TEXT = {lang: _real(f"stars_{lang}", NOTICES[lang][STARS][0]) for lang in LANGS}


def _answer_val(lang: str, way: str, i: int, seed: int = 0) -> _Val:
    return _Val(f"answers_{lang}_{way}[{i}]", answer(lang, way, i, seed)["text"])


class _Lazy(_Val):
    """A test value made only when a check needs it, such as an answer from the course's run; written by its name."""

    def __init__(self, name: str, make):
        super().__init__(name, None)
        self.make = make

    def __deepcopy__(self, memo):
        return self.make()


def _answer_lazy(lang: str, way: str, i: int, seed: int = 0) -> _Lazy:
    return _Lazy(f"answers_{lang}_{way}[{i}]", lambda: answer(lang, way, i, seed)["text"])


def _row_lazy(lang: str, way: str, i: int, seed: int = 0) -> _Lazy:
    return _Lazy(f"rows_{lang}_{way}[{i}]", lambda: read_fields(answer(lang, way, i, seed)["text"]))


def _notice_val(lang: str, i: int) -> _Val:
    return _Val(f"notices_{lang}[{i}]", NOTICES[lang][i][0])


def _not_str(got, text_) -> bool:
    return not isinstance(got, str)


def _not_dict(got, text_) -> bool:
    return not isinstance(got, dict)


def _not_list(got, text_) -> bool:
    return not isinstance(got, list)


def _unpack(e) -> bool:
    return isinstance(e, ValueError) and "unpack" in str(e)


REQUIRED = ["fields_prompt", "read_fields", "unsupported", "why_passes"]

EXERCISES = {
    "fields_prompt": FunctionBox(
        name="fields_prompt",
        picture=prompt_picture,
        function="fields_prompt",
        start=_FP_START,
        solution=_FP,
        show=_show,
        visible=[_toy_prompt("en"), _toy_prompt("nl")],
        hidden=[(_written(TOY_INSTRUCTION["en"]), _written([]), _written(TOY_TEXT["en"])),
                (_INSTR["en"], _SHOTS["en"], _STARS_TEXT["en"]), (_INSTR["en"], _written([]), _STARS_TEXT["en"]),
                (_INSTR["nl"], _SHOTS["nl"], _STARS_TEXT["nl"])],
        first=[_DOTS],
        mistakes=[
            Mistake("not-str", matches=_not_str,
                    en="`{call}` gives {got}, and it should give one text: the whole prompt. What does your `return` give back?",
                    nl="`{call}` geeft {got}, en het moet één tekst zijn: de hele prompt. Wat geeft je `return` terug?"),
            Mistake("no-examples", wrong=_fp(shot="        pass\n"),
                    en="`{call}` gives {got}, and it should give {want}: your prompt leaves out the examples. What should the loop add to `prompt` for each example?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je prompt laat de voorbeelden weg. Wat moet de loop voor elk voorbeeld aan `prompt` toevoegen?"),
            Mistake("no-answers", wrong=_fp(shot='        prompt = prompt + "<text>\\n" + example + "\\n</text>\\n\\n"\n'),
                    en="`{call}` gives {got}, and it should give {want}: each example has its text, but not its answer. How does the model see what a finished answer looks like?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elk voorbeeld heeft de tekst, maar niet het antwoord. Hoe ziet het model hoe een ingevuld antwoord eruitziet?"),
            Mistake("answer-in-tags", wrong=_fp(shot='        prompt = prompt + "<text>\\n" + example + "\\n" + answer + "\\n</text>\\n\\n"\n'),
                    en="`{call}` gives {got}, and it should give {want}: the answer of each example stands inside the tags, as if it were part of the text. Where does the text end and the answer start?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het antwoord van elk voorbeeld staat binnen de tags, alsof het bij de tekst hoort. Waar houdt de tekst op en begint het antwoord?"),
            Mistake("no-tags", wrong=_fp(end="    return prompt + text\n"),
                    en="`{call}` gives {got}, and it should give {want}: the text at the end has no tags around it. How do the examples mark where their text starts and ends?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de tekst aan het eind heeft geen tags eromheen. Hoe geven de voorbeelden aan waar de tekst van een voorbeeld begint en eindigt?"),
            Mistake("no-close", wrong=_fp(end='    return prompt + "<text>\\n" + text\n'),
                    en="`{call}` gives {got}, and it should give {want}: the text at the end has an opening tag, but no `</text>` after it. How does each example close its text?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de tekst aan het eind heeft een openingstag, maar geen `</text>` erna. Hoe sluit elk voorbeeld de tekst af?"),
            Mistake("one-newline", wrong=_fp(shot='        prompt = prompt + "<text>\\n" + example + "\\n</text>\\n" + answer + "\\n"\n'),
                    en="`{call}` gives {got}, and it should give {want}: after the answer of an example comes one line break, not a blank line. What stands between the parts of the prompt?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: na het antwoord van een voorbeeld komt één regeleinde, geen lege regel. Wat staat er tussen de delen van de prompt?"),
            Mistake("return-in-loop", wrong=_FP_HEAD + _FP_SHOT + '        return prompt + "<text>\\n" + text + "\\n</text>"\n' + _FP_END,
                    en="`{call}` gives {got}, and it should give {want}: only the first example. Where should the `return` stand?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: alleen het eerste voorbeeld. Waar hoort de `return` te staan?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: in the loop, the line that adds an example to `prompt`, its text between the tags with its answer under it and a blank line; and after the loop, the prompt with the text between the tags at the end.",
                      "nl": "Vervang de twee keer `...`: in de loop de regel die een voorbeeld aan `prompt` toevoegt, de tekst tussen de tags met het antwoord eronder en een lege regel; en na de loop de prompt met de tekst tussen de tags aan het eind."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The picture under the check shows the first line where your prompt differs from the course's: what is missing there, or extra?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Het plaatje onder de check laat de eerste regel zien waar je prompt anders is dan de prompt van de course: wat ontbreekt daar, of wat staat er te veel?"},
            "almost": {"en": "Right for the two small cases. But `{call}` gives {got}, and it should give {want}. Does your function work for no examples, and for more than one?",
                       "nl": "Goed voor de twee kleine gevallen. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie ook zonder voorbeelden, en met meer dan één?"},
            "correct": {"en": "Your `fields_prompt` is right in all {n} cases.",
                        "nl": "Je `fields_prompt` klopt in alle {n} gevallen."},
        },
    ),
    "read_fields": FunctionBox(
        name="read_fields",
        picture=read_picture,
        function="read_fields",
        start=_RF_START,
        solution=_RF,
        show=_show,
        visible=[(_written(TOY_ANSWER["en"]),), (_written(TOY_ANSWER["nl"]),)],
        hidden=[(_written(""),), (_answer_lazy("en", "zero", 0),), (_answer_lazy("en", "zero", 3),),
                (_answer_lazy("en", "examples", 8),), (_answer_lazy("en", "examples", 11),),
                (_answer_lazy("nl", "zero", 0),), (_answer_lazy("nl", "examples", STARS),)],
        first=[_DOTS],
        mistakes=[
            Mistake("not-dict", matches=_not_dict,
                    en="`{call}` gives {got}, and it should give a dictionary from each field to its value. Where does your `return` stand, and what does it give back?",
                    nl="`{call}` geeft {got}, en het moet een dictionary zijn van elk veld naar de waarde. Waar staat je `return`, en wat geeft je functie terug?"),
            Mistake("no-groups", error=_unpack,
                    en="`{call}` stops with a ValueError about unpacking: `re.findall` gives one string for each match, not a field and a value. What does a pattern need, so that `re.findall` gives a tuple of two for each match?",
                    nl="`{call}` stopt met een ValueError over uitpakken: `re.findall` geeft één string voor elke match, geen veld en waarde. Wat heeft een patroon nodig, zodat `re.findall` voor elke match een tuple van twee geeft?"),
            Mistake("no-strip", wrong=_rf(body='        if value.strip() != "-":\n            row[field] = value\n'),
                    en="`{call}` gives {got}, and it should give {want}: some values end in spaces, as Qwen3 wrote them. How do you take the spaces off both ends of a string?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: sommige waarden eindigen op spaties, zoals Qwen3 de waarden schreef. Hoe haal je de spaties aan beide kanten van een string weg?"),
            Mistake("keeps-dash", wrong=_rf(body="        row[field] = value.strip()\n"),
                    en="`{call}` gives {got}, and it should give {want}: a field with the value `-` is in your row. The model writes `-` for a field the text does not give: should such a field be in the row?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een veld met de waarde `-` staat in je rij. Het model schrijft `-` bij een veld dat niet in de tekst staat: hoort zo'n veld in de rij?"),
            Mistake("dash-only", wrong=_rf(body='        value = value.strip()\n        if value == "-":\n            row[field] = value\n'),
                    en="`{call}` gives {got}, and it should give {want}: your row holds only the fields with `-`. Which fields should it hold?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je rij bevat alleen de velden met `-`. Welke velden horen erin?"),
            Mistake("no-space", wrong=_rf(loop='    for field, value in re.findall(r"(\\w+):\\s*(.+)", answer):\n'),
                    en="`{call}` gives {got}, and it should give {want}: your pattern also matches inside a time such as 10:00, with 10 as a field. What stands right after the colon of a field, and not after the colon of a time?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je patroon matcht ook binnen een tijd zoals 10:00, met 10 als veld. Wat staat er direct na de dubbele punt van een veld, en niet na de dubbele punt van een tijd?"),
            Mistake("greedy", wrong=_rf(loop='    for field, value in re.findall(r"(.+): (.+)", answer):\n'),
                    en="`{call}` gives {got}, and it should give {want}: a field in your row can hold spaces and brackets, so it is more than one word. Which characters can a field name hold, so that the field is the one word right against the colon?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een veld in je rij kan spaties en haakjes bevatten, dus meer dan één woord. Welke tekens kan de naam van een veld bevatten, zodat het veld het ene woord is dat tegen de dubbele punt aan staat?"),
            Mistake("return-in-loop", wrong=_rf(end="        return row\n    return row\n"),
                    en="`{call}` gives {got}, and it should give {want}: only the first field. Where should the `return` stand?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: alleen het eerste veld. Waar hoort de `return` te staan?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: the pattern with a group for the field and a group for the value, and the lines that take the spaces off the value and put it in `row`, unless it is `-`.",
                      "nl": "Vervang de twee keer `...`: het patroon met een groep voor het veld en een groep voor de waarde, en de regels die de spaties van de waarde halen en de waarde in `row` zetten, behalve als het `-` is."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which lines of the answer have the form field: value, and which of those values should be in the row?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Welke regels van het antwoord hebben de vorm veld: waarde, en welke van de waarden horen in de rij?"},
            "almost": {"en": "Right for the two small answers. But `{call}` gives {got}, and it should give {want}. Does your function read every answer of Qwen3, also one with lines in another form?",
                       "nl": "Goed voor de twee kleine antwoorden. Maar `{call}` geeft {got}, en het moet {want} zijn. Leest je functie elk antwoord van Qwen3, ook een met regels in een andere vorm?"},
            "correct": {"en": "Your `read_fields` is right in all {n} cases.",
                        "nl": "Je `read_fields` klopt in alle {n} gevallen."},
        },
    ),
    "unsupported": FunctionBox(
        name="unsupported",
        picture=flag_picture,
        function="unsupported",
        start=_US_START,
        solution=_US,
        show=_show,
        visible=[(_written(TOY_ROW["en"]), _written(TOY_TEXT["en"])), (_written(TOY_ROW["nl"]), _written(TOY_TEXT["nl"]))],
        hidden=[(_written({}), _written(TOY_TEXT["en"])),
                (_written({"what": "bingo", "date": "9 March"}), _written(TOY_TEXT["en"])),
                (_row_lazy("en", "zero", 0), _notice_val("en", 0)), (_row_lazy("en", "examples", 1), _notice_val("en", 1)),
                (_row_lazy("en", "examples", STARS), _notice_val("en", STARS)),
                (_row_lazy("nl", "examples", 1), _notice_val("nl", 1))],
        first=[_DOTS],
        mistakes=[
            Mistake("not-list", matches=_not_list,
                    en="`{call}` gives {got}, and it should give a list of the fields whose value is not in the text. Where does your `return` stand, and what does it give back?",
                    nl="`{call}` geeft {got}, en het moet een lijst zijn van de velden waarvan de waarde niet in de tekst staat. Waar staat je `return`, en wat geeft je functie terug?"),
            Mistake("case", wrong=_us(test="        if value not in text:\n"),
                    en="`{call}` gives {got}, and it should give {want}: your function flags a value that is in the text with other capitals. How do you compare two texts without looking at capitals?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie markeert een waarde die met andere hoofdletters in de tekst staat. Hoe vergelijk je twee teksten zonder op hoofdletters te letten?"),
            Mistake("found", wrong=_us(test="        if value.lower() in text.lower():\n"),
                    en="`{call}` gives {got}, and it should give {want}: the fields whose value is in the text. Which fields should the list hold?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de velden waarvan de waarde in de tekst staat. Welke velden horen in de lijst?"),
            Mistake("text-in-value", wrong=_us(test="        if text.lower() not in value.lower():\n"),
                    en="`{call}` gives {got}, and it should give {want}: your function asks whether the whole text is in the value. Which of the two should be found in the other?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie kijkt of de hele tekst in de waarde staat. Welke van de twee moet in de andere staan?"),
            Mistake("values", wrong=_us(add="            fields.append(value)\n"),
                    en="`{call}` gives {got}, and it should give {want}: the values, not the names of their fields. What should the list hold?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de waarden, niet de namen van de velden. Wat hoort er in de lijst?"),
            Mistake("return-in-loop", wrong=_us(end="        return fields\n    return fields\n"),
                    en="`{call}` gives {got}, and it should give {want}: your function looks at the first field only. Where should the `return` stand?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie kijkt alleen naar het eerste veld. Waar hoort de `return` te staan?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: the lines that add a field to `fields` when its value, in lower case, is not in the text, in lower case.",
                      "nl": "Vervang de `...`: de regels die een veld aan `fields` toevoegen als de waarde, in kleine letters, niet in de tekst staat, in kleine letters."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which values of the row are not in the text, when you leave capitals out of it?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Welke waarden van de rij staan niet in de tekst, als je hoofdletters buiten beschouwing laat?"},
            "almost": {"en": "Right for the two small rows. But `{call}` gives {got}, and it should give {want}. Does your function look at every field of the row, without looking at capitals?",
                       "nl": "Goed voor de twee kleine rijen. Maar `{call}` geeft {got}, en het moet {want} zijn. Kijkt je functie naar elk veld van de rij, zonder op hoofdletters te letten?"},
            "correct": {"en": "Your `unsupported` is right in all {n} cases.",
                        "nl": "Je `unsupported` klopt in alle {n} gevallen."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, code, state, known mistake).
CASES = [
    ("fields_prompt", _FP_START, "empty", None),
    ("fields_prompt", _FP, "correct", None),
    ("fields_prompt", _fp(shot="        ...\n"), "wrong", "dots-left"),
    ("fields_prompt", "def fields_prompt(instruction, examples, text):\n    return len(text)\n", "wrong", "not-str"),
    ("fields_prompt", _fp(shot="        pass\n"), "wrong", "no-examples"),
    ("fields_prompt", _fp(shot='        prompt = prompt + "<text>\\n" + example + "\\n</text>\\n\\n"\n'), "wrong", "no-answers"),
    ("fields_prompt", _fp(shot='        prompt = prompt + "<text>\\n" + example + "\\n" + answer + "\\n</text>\\n\\n"\n'), "wrong", "answer-in-tags"),
    ("fields_prompt", _fp(end="    return prompt + text\n"), "wrong", "no-tags"),
    ("fields_prompt", _fp(end='    return prompt + "<text>\\n" + text\n'), "wrong", "no-close"),
    ("fields_prompt", _fp(shot='        prompt = prompt + "<text>\\n" + example + "\\n</text>\\n" + answer + "\\n"\n'), "wrong", "one-newline"),
    ("fields_prompt", _FP_HEAD + _FP_SHOT + '        return prompt + "<text>\\n" + text + "\\n</text>"\n' + _FP_END, "wrong", "return-in-loop"),
    ("fields_prompt", "def fields_prompt(instruction, examples, text):\n    prompt = instruction + \"\\n\\n\"\n"
     "    for example, answer in examples:\n        prompt += f\"<text>\\n{example}\\n</text>\\n{answer}\\n\\n\"\n"
     "    return prompt + f\"<text>\\n{text}\\n</text>\"\n", "correct", None),
    ("fields_prompt", "def fields_prompt(instruction, examples, text):\n    parts = [instruction]\n"
     "    for example, answer in examples:\n        parts.append(\"<text>\\n\" + example + \"\\n</text>\\n\" + answer)\n"
     "    parts.append(\"<text>\\n\" + text + \"\\n</text>\")\n    return \"\\n\\n\".join(parts)\n", "correct", None),
    ("read_fields", _RF_START, "empty", None),
    ("read_fields", _RF, "correct", None),
    ("read_fields", _rf(body="        ...\n"), "wrong", "dots-left"),
    ("read_fields", "import re\n\n\ndef read_fields(answer):\n    return re.findall(r\"(\\w+): (.+)\", answer)\n", "wrong", "not-dict"),
    ("read_fields", _rf(loop='    for field, value in re.findall(r"\\w+: .+", answer):\n'), "error", "no-groups"),
    ("read_fields", _rf(body='        if value.strip() != "-":\n            row[field] = value\n'), "wrong", "no-strip"),
    ("read_fields", _rf(body="        row[field] = value.strip()\n"), "wrong", "keeps-dash"),
    ("read_fields", _rf(body='        value = value.strip()\n        if value == "-":\n            row[field] = value\n'), "wrong", "dash-only"),
    ("read_fields", _rf(loop='    for field, value in re.findall(r"(\\w+):\\s*(.+)", answer):\n'), "wrong", "no-space"),
    ("read_fields", _rf(loop='    for field, value in re.findall(r"(\\w+):(.+)", answer):\n'), "wrong", "no-space"),
    ("read_fields", _rf(loop='    for field, value in re.findall(r"(.+): (.+)", answer):\n'), "wrong", "greedy"),
    ("read_fields", _rf(end="        return row\n    return row\n"), "wrong", "return-in-loop"),
    ("read_fields", "import re\n\n\ndef read_fields(answer):\n    row = {}\n"
     "    for field, value in re.findall(r\"(\\w+): (.+)\", answer):\n        if value.strip() != \"-\":\n"
     "            row[field] = value.strip()\n    return row\n", "correct", None),
    ("unsupported", _US_START, "empty", None),
    ("unsupported", _US, "correct", None),
    ("unsupported", _us(test="        if value.lower() not in ...:\n"), "wrong", "dots-left"),
    ("unsupported", "def unsupported(row, text):\n    return len(row)\n", "wrong", "not-list"),
    ("unsupported", _us(test="        if value not in text:\n"), "wrong", "case"),
    ("unsupported", _us(test="        if value.lower() in text.lower():\n"), "wrong", "found"),
    ("unsupported", _us(test="        if text.lower() not in value.lower():\n"), "wrong", "text-in-value"),
    ("unsupported", _us(add="            fields.append(value)\n"), "wrong", "values"),
    ("unsupported", _us(end="        return fields\n    return fields\n"), "wrong", "return-in-loop"),
    ("unsupported", "def unsupported(row, text):\n    return [f for f, v in row.items() if v.lower() not in text.lower()]\n", "correct", None),
    ("unsupported", "def unsupported(row, text):\n    fields = []\n    text = text.lower()\n    for field in row:\n"
     "        if not row[field].lower() in text:\n            fields.append(field)\n    return fields\n", "correct", None),
]


# --- the box: one answer of the course's run, read and checked --------------------------------------------------------

_TRY_TEXT = {
    "en": {"title": "Qwen3 on notice {i}: {what}",
           "way": {"zero": "without examples", "examples": "with the two examples", "long": "with the newsletter"},
           "prompt": "The prompt, {way}: {chars} characters, {tokens} tokens for Qwen3",
           "answer": "What Qwen3 wrote, seed {seed}:",
           "head": ["Field", "Qwen3's value", "The notice gives", "In the notice?"],
           "head_long": ["Field", "Qwen3's value", "The notice gives", "In the notice?", "In the newsletter?"],
           "yes": "yes", "no": "no", "nothing": "-", "unread": "(no line)",
           "copied": "Qwen3 filled in an example of the prompt, not this notice.",
           "note": "Blue: what Qwen3 wrote, from the course's run. Gray: what the notice gives, as the course wrote it, and the prompt. Purple: a value that is not in the notice, or not in the newsletter. A field that Qwen3 filled in with - is left out of the row, so the check does not look at it."},
    "nl": {"title": "Qwen3 op aankondiging {i}: {what}",
           "way": {"zero": "zonder voorbeelden", "examples": "met de twee voorbeelden", "long": "met de nieuwsbrief"},
           "prompt": "De prompt, {way}: {chars} tekens, {tokens} tokens voor Qwen3",
           "answer": "Wat Qwen3 schreef, seed {seed}:",
           "head": ["Veld", "Waarde van Qwen3", "De aankondiging geeft", "In de aankondiging?"],
           "head_long": ["Veld", "Waarde van Qwen3", "De aankondiging geeft", "In de aankondiging?", "In de nieuwsbrief?"],
           "yes": "ja", "no": "nee", "nothing": "-", "unread": "(geen regel)",
           "copied": "Qwen3 vulde een voorbeeld uit de prompt in, niet deze aankondiging.",
           "note": "Blauw: wat Qwen3 schreef, uit de run van de course. Grijs: wat de aankondiging geeft, zoals de course de aankondiging schreef, en de prompt. Paars: een waarde die niet in de aankondiging staat, of niet in de nieuwsbrief. Een veld dat Qwen3 met - invulde, staat niet in de rij, dus de check kijkt er niet naar."},
}


def try_picture(lang: str, way: str, seed, i) -> str:
    """The box: one answer of the course's run, with its prompt, the row the course reads from it, what the notice
    really gives, and whether each value is in the notice (and, for the way long, in the newsletter)."""
    lang = lang if lang in LANGS else "en"
    w = _TRY_TEXT[lang]
    seed, i = int(seed), int(i)
    a = answer(lang, way, i, seed)
    p = prompt(way, lang, i)
    lines = answer_lines(a["text"])
    row = read_fields(a["text"])
    text_ = NOTICES[lang][i][0]
    long_ = way == "long"
    rows_ = []
    for f, want in gold(lang, i).items():
        got = lines.get(f, _UNREAD)
        if got is _UNREAD:
            shown = f'<span style="{_FAINT}">{html.escape(w["unread"])}</span>'
        elif got is None:
            shown = _chip("-", _CHIP_NONE)
        else:
            shown = _chip(got, _CHIP_PRED if f in row and row[f].lower() in text_.lower() else _CHIP_LOSS)
        cells = [html.escape(f), shown, f'<span style="{_DATA}">{html.escape(w["nothing"] if want is None else want)}</span>']
        for where in ([text_, newsletter(lang)] if long_ else [text_]):
            if f in row:
                ok = row[f].lower() in where.lower()
                cells.append(f'<span style="{"" if ok else _WRONG}">{w["yes" if ok else "no"]}</span>')
            else:
                cells.append("")
        rows_.append(cells)
    parts = [_label(w["prompt"].format(way=w["way"][way], chars=_num(len(p), lang), tokens=_num(a["prompt_tokens"], lang))),
             f'<details><summary style="{_UI}">{html.escape(p[:80])}…</summary>{_code(p)}</details>',
             _label(w["answer"].format(seed=seed)), _code(a["text"], _PRED),
             _table(rows_, w["head_long" if long_ else "head"])]
    if copied_example(lang, way, i, seed):
        parts.append(_note(w["copied"]))
    parts.append(_note(w["note"]))
    title = w["title"].format(i=i + 1, what=NOTICES[lang][i][1][0])
    return yours.figure(title, "".join(parts), alt=title)


def notice_options(lang: str) -> dict:
    """The twelve notices for the dropdown of the box: their number and name, to their place counting from 0."""
    return {f"{i + 1}. {g[0]}": i for i, (_, g) in enumerate(NOTICES[lang])}


# --- the table: what the course counts in each way -----------------------------------------------------------------------

_TABLE_TEXT = {
    "en": {"title": "Qwen3 in three ways: the twelve notices, four seeds",
           "head": ["", "without examples", "with the two examples", "with the newsletter"],
           "rows": {"all_five": "Answers with all five fields, of {answers}",
                    "found": "Values right or part of the right one, of {given}",
                    "made_up": "Values for a field the notice leaves out, of {missing}",
                    "made_up_flagged": "Of those, flagged by the check",
                    "copied": "Answers that fill in an example instead",
                    "right_flagged": "Right values flagged, written differently"},
           "note": "Counted by the course over the twelve notices and the seeds 0 to 3 of the course's run: {answers} answers in each way. Purple: values that need a person to look at them, filled in for a field that the notice leaves out. The check looks up each value in the notice of its event, also in the way with the newsletter."},
    "nl": {"title": "Qwen3 op drie manieren: de twaalf aankondigingen, vier seeds",
           "head": ["", "zonder voorbeelden", "met de twee voorbeelden", "met de nieuwsbrief"],
           "rows": {"all_five": "Antwoorden met alle vijf velden, van {answers}",
                    "found": "Waarden goed of een deel van de goede, van {given}",
                    "made_up": "Waarden bij een veld dat niet in de aankondiging staat, van {missing}",
                    "made_up_flagged": "Daarvan gemarkeerd door de check",
                    "copied": "Antwoorden die in plaats daarvan een voorbeeld invullen",
                    "right_flagged": "Goede waarden gemarkeerd, anders geschreven"},
           "note": "Geteld door de course over de twaalf aankondigingen en de seeds 0 tot en met 3 van de run van de course: {answers} antwoorden per manier. Paars: waarden waar een mens naar moet kijken, ingevuld bij een veld dat niet in de aankondiging staat. De check zoekt elke waarde op in de aankondiging van de activiteit, ook bij de manier met de nieuwsbrief."},
}


def table_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = _TABLE_TEXT[lang]
    t = {way: tally(lang, way) for way in WAYS}
    answers = t["zero"]["answers"]
    missing = sum(g is None for _, gs in NOTICES[lang] for g in gs) * SEEDS
    given = len(NOTICES[lang]) * 5 * SEEDS - missing
    rows_ = []
    for key, label in w["rows"].items():
        cells = [html.escape(label.format(answers=answers, missing=missing, given=given))]
        for way in WAYS:
            v = t[way]["right"] + t[way]["part"] if key == "found" else t[way]["made-up" if key == "made_up" else key]
            style = _WRONG if key == "made_up" else ""
            cells.append(f'<span style="{style}">{v}</span>')
        rows_.append(cells)
    body = _table(rows_, w["head"], numbers_=(1, 2, 3)) + _note(w["note"].format(answers=answers))
    return yours.figure(w["title"], body, alt=w["title"])


# --- your own notice, with Qwen3 on your computer ------------------------------------------------------------------------

YOUR_NOTICE = {"en": "Pancake breakfast in the scout hall on Sunday 19 April, from 8:30. Bring your own plate.",
               "nl": "Pannenkoekenontbijt in het clubhuis van de scouting op zondag 19 april, vanaf 8.30 uur. Neem je eigen bord mee."}


def downloaded() -> bool:
    from everything_ai import chat
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    from everything_ai import chat
    chat.download_or_stop(button, texts)


_LIVE_TEXT = {
    "en": {"title": "Qwen3 on your notice", "loading": "Loading Qwen3 …", "writing": "Qwen3 is writing …",
           "done": "{tokens} tokens in {seconds} seconds, with the two examples, seed 0.",
           "head": ["Field", "Qwen3's value", "In your notice?"], "yes": "yes", "no": "no",
           "note": "Blue: what Qwen3 wrote on your computer, and a value that is in your notice. Purple: a value that is not in your notice.",
           "load": "Loading Qwen3 stops with an error. Close other programs and press the button again; if it happens again, report it on GitHub.",
           "memory": "The memory of your device is full. Close other programs, and press the button again."},
    "nl": {"title": "Qwen3 op je aankondiging", "loading": "Qwen3 laden …", "writing": "Qwen3 schrijft …",
           "done": "{tokens} tokens in {seconds} seconden, met de twee voorbeelden, seed 0.",
           "head": ["Veld", "Waarde van Qwen3", "In je aankondiging?"], "yes": "ja", "no": "nee",
           "note": "Blauw: wat Qwen3 op je computer schreef, en een waarde die in je aankondiging staat. Paars: een waarde die niet in je aankondiging staat.",
           "load": "Qwen3 laden stopt met een fout. Sluit andere programma's en druk opnieuw op de knop; gebeurt het weer, meld het dan op GitHub.",
           "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."},
}


def _live_picture(text_: str, notice: str, note: str, w: dict, done: bool) -> str:
    body = _code(text_, _PRED) + _note(note)
    if done:
        row = read_fields(text_)
        flags = unsupported(row, notice)
        rows_ = [[html.escape(f), _chip(v, _CHIP_LOSS if f in flags else _CHIP_PRED),
                  f'<span style="{_WRONG if f in flags else ""}">{w["no" if f in flags else "yes"]}</span>'] for f, v in row.items()]
        body += _table(rows_, w["head"]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


def ask_live(notice: str, show, lang: str) -> str:
    """Qwen3 on the learner's own notice, with the two examples and seed 0, what it writes so far shown by `show(html)`
    at most five times a second; then the row and whether each value is in the notice."""
    from everything_ai import chat
    lang = lang if lang in LANGS else "en"
    w = _LIVE_TEXT[lang]
    messages = [{"role": "user", "content": fields_prompt(INSTRUCTION[lang], SHOTS[lang], notice)}]
    a = chat.answer_live(messages, lambda t: show(_live_picture(t or "", notice, w["loading"] if t is None else w["writing"], w, False)),
                         max_tokens=MAX_TOKENS, seed=0)
    if isinstance(a, str):
        return f'<p><em>{html.escape(w[a])}</em></p>'
    note = w["done"].format(tokens=a["tokens"], seconds=_dec(a["seconds"], lang))
    return _live_picture(a["message"]["content"], notice, note, w, True)


# --- the numbers the text names ------------------------------------------------------------------------------------------



def numbers() -> dict:
    """The numbers the lesson's text names, per language: what the course counts in each way, and the seed-0 answers
    the text quotes."""
    out = {}
    for lang in LANGS:
        n = {"missing": sum(g is None for _, gs in NOTICES[lang] for g in gs), "notices": len(NOTICES[lang])}
        for way in WAYS:
            n[way] = tally(lang, way)
            n[way + "0"] = tally(lang, way, [0])
            n[way + "_tokens"] = [answer(lang, way, i)["prompt_tokens"] for i in range(n["notices"])]
            n[way + "_seconds"] = sum(answer(lang, way, i, s)["seconds"] for i in range(n["notices"]) for s in range(SEEDS))
        n["long_doc_flagged"] = sum(len(flagged(lang, "long", i, s, "newsletter")) for i in range(n["notices"]) for s in range(SEEDS))
        n["long_notice_flagged"] = sum(len(flagged(lang, "long", i, s)) for i in range(n["notices"]) for s in range(SEEDS))
        n["newsletter_chars"] = len(newsletter(lang))
        out[lang] = n
    return out


# --- the idea: one notice, its prompt, and Qwen3's answer -------------------------------------------------------------------

_IDEA_TEXT = {
    "en": {"title": "Qwen3 fills in the fields of a notice",
           "instruction": "The instruction: the five fields, the form of the answer, and - for a field the text does not give",
           "example": "Example {k}, between the tags, with its finished answer",
           "notice": "The notice, between the tags: the text to fill in the fields for",
           "answer": "What Qwen3 wrote after the prompt, seed 0, from the course's run",
           "head": ["Field", "Value", "In the notice?"], "yes": "yes", "no": "no",
           "note": "Gray: the prompt, as the course wrote it. Blue: what Qwen3 wrote, and a value that is in the notice. Purple: a value that is not in the notice."},
    "nl": {"title": "Qwen3 vult de velden van een aankondiging in",
           "instruction": "De instructie: de vijf velden, de vorm van het antwoord, en - bij een veld dat niet in de tekst staat",
           "example": "Voorbeeld {k}, tussen de tags, met het afgemaakte antwoord",
           "notice": "De aankondiging, tussen de tags: de tekst waarvoor de velden ingevuld moeten worden",
           "answer": "Wat Qwen3 na de prompt schreef, seed 0, uit de run van de course",
           "head": ["Veld", "Waarde", "In de aankondiging?"], "yes": "ja", "no": "nee",
           "note": "Grijs: de prompt, zoals de course de prompt schreef. Blauw: wat Qwen3 schreef, en een waarde die in de aankondiging staat. Paars: een waarde die niet in de aankondiging staat."},
}


def idea_picture(lang: str, i: int = SWIM) -> str:
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang]
    text_ = NOTICES[lang][i][0]
    parts = [_label(w["instruction"]), _code(INSTRUCTION[lang])]
    for k, (example, ans) in enumerate(SHOTS[lang]):
        parts += [_label(w["example"].format(k=k + 1)), _code(f"<text>\n{example}\n</text>\n{ans}")]
    parts += [_label(w["notice"]), _code(f"<text>\n{text_}\n</text>")]
    a = answer(lang, "examples", i)["text"]
    parts += [_label(w["answer"]), _code(a, _PRED)]
    row = read_fields(a)
    flags = unsupported(row, text_)
    rows_ = [[html.escape(f), _chip(v, _CHIP_LOSS if f in flags else _CHIP_PRED),
              f'<span style="{_WRONG if f in flags else ""}">{w["no" if f in flags else "yes"]}</span>'] for f, v in row.items()]
    parts += [_table(rows_, w["head"]), _note(w["note"])]
    return yours.figure(w["title"], "".join(parts), alt=w["title"])
