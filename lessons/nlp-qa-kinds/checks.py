# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-qa-kinds: where an answer stands in a text, as a span (find_span); which text of a small
collection shares the most words with a question (best_text); and open-domain question answering, which first picks
a text and then lets a chat model read it (open_domain).

The collection is six short texts that the course wrote for this lesson, in English and in Dutch, with the same facts:
three about the world (spiders, Mercury, the Eiffel Tower) and three about a town that no model can know (its library,
repair café and outdoor pool). Each text has two questions with the answer as people write it (QUESTIONS): seven
answers stand word for word in the text, five need a small calculation. The answers of the model are the course's run of
Qwen3-0.6B, the chat model of the course, in data/course-qa-kinds-qwen.json (data/tools/record_course_qa_kinds_qwen.py):
every question without a text (closed-book) and with each of the six texts, with the seeds 0 to 3. Every number in the
text and the pictures is computed from that file by the course (tests/test_nlp_qa_kinds.py computes them again)."""
import functools
import html
import json
import re

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake

LANGS = ("en", "nl")
COURSE = "data/course-qa-kinds-qwen.json"
SEEDS = 4
MAX_TOKENS = 60
INSTRUCTION = {
    "en": {"closed": "Answer the question in one short sentence.",
           "text": "Answer the question in one short sentence, using the text."},
    "nl": {"closed": "Beantwoord de vraag in één korte zin.",
           "text": "Beantwoord de vraag in één korte zin, met behulp van de tekst."},
}
ASKED = {"en": "Question: ", "nl": "Vraag: "}

# The six texts of the collection, in the same order in both languages.
TEXTS = {
    "en": [
        "A spider has eight legs and is not an insect: insects have six. All spiders make silk, but not all of them "
        "spin webs.",
        "Mercury is the planet closest to the Sun. A year on Mercury lasts 88 days, but a single day there, from one "
        "sunrise to the next, lasts 176 Earth days.",
        "The Eiffel Tower in Paris was finished in 1889 for a world's fair. With its antennas it is about 330 meters "
        "tall. It was meant to stand for only twenty years.",
        "The town library is open from 9:00 to 18:00 on weekdays. On Saturday it opens at 10:00 and closes at 14:30, "
        "and on Sunday it stays closed. Borrowing a book is free; a new card costs €3.",
        "The repair café meets on the first Saturday of every month in the community center. Volunteers fix broken "
        "lamps, toasters and bikes for free; a cup of coffee costs €0.80.",
        "The outdoor pool of the town opens on 15 May and closes on 15 September. A day ticket costs €4.50, and "
        "children under six swim for free. On hot days the pool stays open until 21:00.",
    ],
    "nl": [
        "Een spin heeft acht poten en is geen insect: insecten hebben er zes. Alle spinnen maken zijde, maar niet "
        "alle spinnen weven een web.",
        "Mercurius is de planeet die het dichtst bij de zon staat. Een jaar op Mercurius duurt 88 dagen, maar één "
        "dag daar, van zonsopkomst tot zonsopkomst, duurt 176 aardse dagen.",
        "De Eiffeltoren in Parijs was in 1889 klaar, voor een wereldtentoonstelling. Met de antennes erbij is de "
        "toren ongeveer 330 meter hoog. Eigenlijk zou de toren slechts twintig jaar blijven staan.",
        "De bibliotheek van het dorp is op werkdagen open van 9.00 tot 18.00 uur. Op zaterdag gaat de bibliotheek om "
        "10.00 uur open en om 14.30 uur dicht, en op zondag is de bibliotheek gesloten. Een boek lenen is gratis; "
        "een nieuwe pas kost € 3.",
        "Het repaircafé komt elke eerste zaterdag van de maand samen in het buurthuis. Vrijwilligers repareren "
        "kapotte lampen, broodroosters en fietsen gratis; een kop koffie kost € 0,80.",
        "Het openluchtzwembad van het dorp gaat op 15 mei open en op 15 september dicht. Een dagkaartje kost € 4,50, "
        "en kinderen onder de zes zwemmen gratis. Op warme dagen blijft het zwembad open tot 21.00 uur.",
    ],
}
# For each text two questions: (the text that answers it, the question, the answer as people write it, the forms
# of the answer that count as right when they stand in what the model wrote, as whole words, ignoring capitals).
QUESTIONS = {
    "en": [
        (0, "How many legs does a spider have?", "eight", ["eight", "8"]),
        (0, "How many more legs does a spider have than an insect?", "two",
         ["two more", "2 more", "difference is two", "difference is 2", "= 2"]),
        (1, "How long is a year on Mercury?", "88 days", ["88"]),
        (1, "How many years on Mercury fit in one day there?", "two", ["two", "2"]),
        (2, "In which year was the Eiffel Tower finished?", "1889", ["1889"]),
        (2, "How old was the Eiffel Tower in 1989?", "100 years", ["100", "hundred"]),
        (3, "When does the town library close on Saturday?", "14:30", ["14:30", "2:30"]),
        (3, "On which day is the town library closed?", "Sunday", ["sunday"]),
        (4, "How much does a cup of coffee cost at the repair café?", "€0.80", ["0.80", "80 cents"]),
        (4, "How many times a year does the repair café meet?", "twelve", ["twelve", "12"]),
        (5, "How much is a day ticket for the outdoor pool?", "€4.50", ["4.50"]),
        (5, "How many months a year is the pool open?", "four", ["four", "4"]),
    ],
    "nl": [
        (0, "Hoeveel poten heeft een spin?", "acht", ["acht", "8"]),
        (0, "Hoeveel poten heeft een spin meer dan een insect?", "twee",
         ["twee meer", "2 meer", "twee poten meer", "2 poten meer", "verschil is twee", "verschil is 2", "= 2"]),
        (1, "Hoe lang duurt een jaar op Mercurius?", "88 dagen", ["88"]),
        (1, "Hoeveel jaren van Mercurius passen in één dag daar?", "twee", ["twee", "2"]),
        (2, "In welk jaar was de Eiffeltoren klaar?", "1889", ["1889"]),
        (2, "Hoe oud was de Eiffeltoren in 1989?", "100 jaar", ["100", "honderd"]),
        (3, "Hoe laat gaat de bibliotheek van het dorp op zaterdag dicht?", "14.30 uur",
         ["14.30", "14:30", "half drie"]),
        (3, "Op welke dag is de bibliotheek van het dorp gesloten?", "zondag", ["zondag"]),
        (4, "Hoeveel kost een kop koffie in het repaircafé?", "€ 0,80", ["0,80", "0.80", "80 cent"]),
        (4, "Hoe vaak per jaar komt het repaircafé samen?", "twaalf keer", ["twaalf", "12"]),
        (5, "Hoeveel kost een dagkaartje voor het openluchtzwembad?", "€ 4,50", ["4,50", "4.50"]),
        (5, "Hoeveel maanden per jaar is het zwembad open?", "vier maanden", ["vier", "4"]),
    ],
}

_WORD = re.compile(r"[^\W_]+")


def words(text: str) -> list:
    """The words of a text in lower case, without punctuation: "On Sunday, it stays closed." gives
    ["on", "sunday", "it", "stays", "closed"]. A number is a word too: "14:30" gives ["14", "30"]."""
    return _WORD.findall(text.lower())


def prompt(lang: str, i: int, k=None) -> str:
    """What Qwen3 got for question i: the instruction and the question (k None, closed-book), or the instruction,
    text k of the collection between <text> tags, and the question."""
    q = QUESTIONS[lang][i][1]
    if k is None:
        return INSTRUCTION[lang]["closed"] + "\n\n" + ASKED[lang] + q
    return INSTRUCTION[lang]["text"] + "\n\n<text>\n" + TEXTS[lang][k] + "\n</text>\n\n" + ASKED[lang] + q


# The name of each text, for the pictures.
TITLES = {"en": ["Spiders", "Mercury", "The Eiffel Tower", "The town library", "The repair café", "The outdoor pool"],
          "nl": ["Spinnen", "Mercurius", "De Eiffeltoren", "De bibliotheek van het dorp", "Het repaircafé",
                 "Het openluchtzwembad"]}
WORLD = (0, 1, 2)    # the texts about the world; 3, 4 and 5 are about the town
LIBRARY = 6          # the question of the worked example: when the town library closes on Saturday
POOL = 11            # the question that best_text sends to the wrong text: the months the pool is open


# --- the course's own answers to the exercises ---------------------------------------------------------------------------

def find_span(text: str, answer: str):
    """Where answer stands in text, the first time: (start, end) with text[start:end] == answer; None when it is not
    there."""
    for start in range(len(text) - len(answer) + 1):
        end = start + len(answer)
        if text[start:end] == answer:
            return (start, end)
    return None


def shared_words(question: str, text: str) -> list:
    """The different words of the question that are also in the text, in the order of the question."""
    in_text = set(words(text))
    out = []
    for w in words(question):
        if w in in_text and w not in out:
            out.append(w)
    return out


def best_text(question: str, texts: list) -> int:
    """The place of the text that shares the most different words with the question; the first of them on a tie."""
    asked = set(words(question))
    best, most = 0, -1
    for k in range(len(texts)):
        shared = len([w for w in asked if w in words(texts[k])])
        if shared > most:
            best = k
            most = shared
    return best


def open_domain(question: str, texts: list) -> str:
    """Open-domain question answering in two steps: pick the text that shares the most words with the question, then
    let Qwen3 answer with that text (seed 0, from the course's run)."""
    return ask(question, texts[best_text(question, texts)])


# --- the course's run ----------------------------------------------------------------------------------------------------

@functools.cache
def course() -> dict:
    """The course's run of Qwen3-0.6B, data/course-qa-kinds-qwen.json."""
    from everything_ai import paths
    return json.loads((paths.ROOT / COURSE).read_text(encoding="utf-8"))


def answer(lang: str, i: int, k=None, seed: int = 0) -> dict:
    """Qwen3's answer to question i, without a text (k None) or with text k, with one seed: its text, tokens, prompt
    tokens, seconds and finish."""
    q = course()["answers"][lang][i]
    return (q["closed"] if k is None else q["texts"][k])["seeds"][seed]


class NotRecorded(ValueError):
    """What ask() raises for a question or a text that is not in the course's run; `kind` says what it got."""

    def __init__(self, kind: str):
        super().__init__({"question": "ask() knows only the twelve questions of the lesson",
                          "index": "ask() wants the text itself, not its place in the list",
                          "list": "ask() wants one text, not the whole list",
                          "text": "ask() knows only the six texts of the lesson"}[kind])
        self.kind = kind


def _where(question) -> tuple:
    for lang in LANGS:
        for i, q in enumerate(QUESTIONS[lang]):
            if q[1] == question:
                return lang, i
    raise NotRecorded("question")


def ask(question: str, text=None, seed: int = 0) -> str:
    """What Qwen3 answered in the course's run to one of the lesson's questions: without a text (closed-book) when
    text is None, else with that text of the collection; with seed 0 unless another seed is given."""
    lang, i = _where(question)
    if text is None:
        return answer(lang, i, None, seed)["text"]
    if isinstance(text, bool) or isinstance(text, int):
        raise NotRecorded("index")
    if isinstance(text, (list, tuple)):
        raise NotRecorded("list")
    if text not in TEXTS[lang]:
        raise NotRecorded("text")
    return answer(lang, i, TEXTS[lang].index(text), seed)["text"]


def _form(form: str) -> re.Pattern:
    """A form of an answer as a whole word: not inside a longer word or a longer number, such as 4 in 4.50 or 14:30."""
    return re.compile(r"(?<![\w])(?<!\d[.,:])" + re.escape(form) + r"(?![\w])(?![.,:]\d)", re.IGNORECASE)


def right(lang: str, i: int, said: str) -> bool:
    """Whether an answer to question i holds the right answer: one of its forms, as a whole word."""
    return any(_form(f).search(said) for f in QUESTIONS[lang][i][3])


def is_right(lang: str, i: int, k=None, seed: int = 0) -> bool:
    return right(lang, i, answer(lang, i, k, seed)["text"])


def tally(lang: str, questions, way: str, seeds=None) -> int:
    """How many answers hold the right answer, over some questions and the seeds: way "closed" without a text, "right"
    with the text that answers the question, "found" with the text best_text picks from all six."""
    seeds = range(SEEDS) if seeds is None else seeds
    n = 0
    for i in questions:
        k_right, q = QUESTIONS[lang][i][0], QUESTIONS[lang][i][1]
        k = {"closed": None, "right": k_right, "found": best_text(q, TEXTS[lang])}[way]
        n += sum(is_right(lang, i, k, s) for s in seeds)
    return n


def town(lang: str) -> list:
    """The questions about the town, answered by texts 3, 4 and 5."""
    return [i for i, q in enumerate(QUESTIONS[lang]) if q[0] not in WORLD]


def world(lang: str) -> list:
    return [i for i, q in enumerate(QUESTIONS[lang]) if q[0] in WORLD]


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


_ANSWER = ("padding:0.02rem 0.25rem;border-radius:3px;font-family:var(--ea-font-ui);font-size:0.84rem;line-height:1.55;"
           "-webkit-box-decoration-break:clone;box-decoration-break:clone")
_ANSWER_RIGHT = _ANSWER + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
_ANSWER_WRONG = _ANSWER + ";background:var(--ea-loss-wash);color:var(--ea-loss-text)"
_MARK = "padding:0 0.1rem;border-radius:2px;font-weight:600"
_MARK_PRED = _MARK + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
_MARK_LOSS = _MARK + ";background:var(--ea-loss-wash);color:var(--ea-loss-text)"


def _said(text_: str, ok: bool) -> str:
    """An answer of Qwen3, blue when it holds the right answer and purple when it does not."""
    return f'<span style="{_ANSWER_RIGHT if ok else _ANSWER_WRONG}">{html.escape(text_)}</span>'


def _gray(text_: str) -> str:
    return f'<span style="{_DATA}">{html.escape(text_)}</span>'


def _snippet(text_: str, start: int, end: int, style: str, room: int = 28) -> str:
    """The part of a text around text[start:end], with that span marked; … where the text goes on."""
    a, b = max(0, start - room), min(len(text_), end + room)
    head = ("…" if a > 0 else "") + text_[a:start]
    tail = text_[end:b] + ("…" if b < len(text_) else "")
    return (f'<span style="{_DATA}">{html.escape(head)}</span><span style="{style}">{html.escape(text_[start:end])}</span>'
            f'<span style="{_DATA}">{html.escape(tail)}</span>')


def _is_span(v, text_: str) -> bool:
    """Whether v is a tuple (start, end) of two whole numbers that mark a part of the text."""
    return (isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in v)
            and 0 <= v[0] <= v[1] <= len(text_))


# --- the picture under exercise 1: where each answer stands in its text -------------------------------------------------

_SPAN_TEXT = {
    "en": {"title": "Your find_span on the twelve answers",
           "head": ["Question", "Answer", "Where your find_span finds it"],
           "none": "not in the text",
           "course": "the course's find_span gives {want}",
           "summary": "Your function finds {found} of the {n} answers in the text that answers the question: extractive question answering can point to those. The other {other} are written in other words, or need a small calculation: only abstractive question answering can give them.",
           "differs": "Your function finds the right span for {found} of the {total} answers that are in the text.",
           "note": "Gray: the question, the answer as people write it, and the text around the span. Blue: the span your function finds. Dashed: an answer your function does not find in the text. Purple: a result that differs from the course's find_span.",
           "kind": "a tuple (start, end) or None"},
    "nl": {"title": "Je find_span op de twaalf antwoorden",
           "head": ["Vraag", "Antwoord", "Waar je find_span het vindt"],
           "none": "niet in de tekst",
           "course": "de find_span van de course geeft {want}",
           "summary": "Je functie vindt {found} van de {n} antwoorden in de tekst die de vraag beantwoordt: extractive question answering kan naar deze antwoorden wijzen. De andere {other} zijn in andere woorden geschreven, of vragen een kleine som: alleen abstractive question answering kan deze antwoorden geven.",
           "differs": "Je functie vindt de goede span bij {found} van de {total} antwoorden die in de tekst staan.",
           "note": "Grijs: de vraag, het antwoord zoals mensen het schrijven, en de tekst rond de span. Blauw: de span die je functie vindt. Gestippeld: een antwoord dat je functie niet in de tekst vindt. Paars: een uitkomst die anders is dan wat de find_span van de course geeft.",
           "kind": "een tuple (start, end) of None"},
}


def span_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _SPAN_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    pairs = [(TEXTS[lang][k], a) for k, _, a, _ in QUESTIONS[lang]]
    mine = _call(lambda: [fn(t, a) for t, a in pairs])
    if mine is _SLOW:
        return _empty(w, "slow")
    if mine is _STOPPED:
        return _empty(w, "stopped")
    if not all(m is None or isinstance(m, (tuple, list)) for m in mine):
        return _empty(w, "odd")
    rows_, found = [], 0
    for (t, a), m, (_, q, _, _) in zip(pairs, mine, QUESTIONS[lang]):
        want = find_span(t, a)
        found += m == want and m is not None
        if m == want and m is None:
            where = _chip(w["none"], _CHIP_NONE)
        elif m == want:
            where = _snippet(t, m[0], m[1], _MARK_PRED)
        elif _is_span(m, t):
            where = _snippet(t, m[0], m[1], _MARK_LOSS) + _note(w["course"].format(want=_value(want)))
        else:
            where = _chip(_value(m), _CHIP_LOSS) + _note(w["course"].format(want=_value(want)))
        rows_.append([_gray(q), _chip(a, _CHIP_DATA), where])
    wants = [find_span(t, a) for t, a in pairs]
    total = sum(want is not None for want in wants)
    summary = (w["summary"].format(found=found, n=len(pairs), other=len(pairs) - found) if list(mine) == wants
               else w["differs"].format(found=found, total=total))
    body = (_table(rows_, w["head"])
            + f'<div style="{_UI}">{html.escape(summary)}</div>'
            + _note(w["note"]))
    return yours.figure(w["title"], body, alt=w["title"])


# --- the picture under exercise 2: the text your best_text picks for each question ---------------------------------------

_PICK_TEXT = {
    "en": {"title": "Your best_text on the twelve questions, with all six texts",
           "head": ["Question", "Text your best_text picks", "Words it shares with the question"],
           "course": "the course's best_text picks {want}",
           "summary": "Your function picks the text that answers the question for {right} of the {n} questions.",
           "scores": "Different words that “{q}” shares with each text:",
           "note": "Gray: the question, and the words it shares with the text your function picks. Blue: the text your function picks, when it answers the question. Purple: a text that does not answer the question, or a pick that differs from the course's best_text.",
           "kind": "the place of a text in the list"},
    "nl": {"title": "Je best_text op de twaalf vragen, met alle zes teksten",
           "head": ["Vraag", "Tekst die je best_text kiest", "Woorden die de tekst met de vraag deelt"],
           "course": "de best_text van de course kiest {want}",
           "summary": "Je functie kiest bij {right} van de {n} vragen de tekst die de vraag beantwoordt.",
           "scores": "Verschillende woorden die „{q}” met elke tekst deelt:",
           "note": "Grijs: de vraag, en de woorden die de vraag deelt met de tekst die je functie kiest. Blauw: de tekst die je functie kiest, als de tekst de vraag beantwoordt. Paars: een tekst die de vraag niet beantwoordt, of een keuze die anders is dan bij de best_text van de course.",
           "kind": "de plaats van een tekst in de lijst"},
}


def _scores_line(lang: str, i: int) -> str:
    q = QUESTIONS[lang][i][1]
    return ", ".join(f"{TITLES[lang][k]} {len(shared_words(q, t))}" for k, t in enumerate(TEXTS[lang]))


def pick_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _PICK_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    texts = TEXTS[lang]
    mine = _call(lambda: [fn(q, list(texts)) for _, q, _, _ in QUESTIONS[lang]])
    if mine is _SLOW:
        return _empty(w, "slow")
    if mine is _STOPPED:
        return _empty(w, "stopped")
    if not all(isinstance(m, int) and not isinstance(m, bool) and 0 <= m < len(texts) for m in mine):
        return _empty(w, "odd")
    rows_, right_ = [], 0
    for m, (k, q, _, _) in zip(mine, QUESTIONS[lang]):
        want = best_text(q, texts)
        right_ += m == k
        picked = _chip(TITLES[lang][m], _CHIP_PRED if m == k else _CHIP_LOSS)
        if m != want:
            picked += _note(w["course"].format(want=TITLES[lang][want]))
        shared = "".join(_chip(x, _CHIP_DATA) for x in shared_words(q, texts[m]))
        rows_.append([_gray(q), picked, shared])
    body = (_table(rows_, w["head"])
            + f'<div style="{_UI}">{html.escape(w["summary"].format(right=right_, n=len(mine)))}</div>'
            + _note(w["scores"].format(q=QUESTIONS[lang][POOL][1]) + " " + _scores_line(lang, POOL) + ".")
            + _note(w["note"]))
    return yours.figure(w["title"], body, alt=w["title"])


# --- the picture under exercise 3: without a text, and with the text your open_domain picks --------------------------------

_OPEN_TEXT = {
    "en": {"title": "Qwen3 without a text, and with the text your open_domain picks",
           "head": ["Question", "Without a text (closed-book)", "Your open_domain"],
           "course": "the course's open_domain gives {want}",
           "summary": "Answers that hold the right answer, of the {n} with seed 0: without a text {closed}, with the text your open_domain picks {mine}.",
           "note": "Gray: the question. Blue: an answer of Qwen3, from the course's run, seed 0, that holds the right answer. Purple: an answer that does not.",
           "kind": "a text: the answer of Qwen3"},
    "nl": {"title": "Qwen3 zonder tekst, en met de tekst die je open_domain kiest",
           "head": ["Vraag", "Zonder tekst (closed-book)", "Je open_domain"],
           "course": "de open_domain van de course geeft {want}",
           "summary": "Antwoorden met het goede antwoord erin, van de {n} met seed 0: zonder tekst {closed}, met de tekst die je open_domain kiest {mine}.",
           "note": "Grijs: de vraag. Blauw: een antwoord van Qwen3, uit de run van de course, seed 0, met het goede antwoord erin. Paars: een antwoord zonder het goede antwoord.",
           "kind": "een tekst: het antwoord van Qwen3"},
}


def open_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _OPEN_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    texts = TEXTS[lang]
    mine = _call(lambda: [fn(q, list(texts)) for _, q, _, _ in QUESTIONS[lang]])
    if mine is _SLOW:
        return _empty(w, "slow")
    if mine is _STOPPED:
        return _empty(w, "stopped")
    if not all(isinstance(m, str) for m in mine):
        return _empty(w, "odd")
    rows_, closed, got = [], 0, 0
    for i, (m, (_, q, _, _)) in enumerate(zip(mine, QUESTIONS[lang])):
        c = answer(lang, i)["text"]
        closed += right(lang, i, c)
        got += right(lang, i, m)
        cell = _said(m, right(lang, i, m))
        want = open_domain(q, texts)
        if m != want:
            cell += _note(w["course"].format(want=_value(want)))
        rows_.append([_gray(q), _said(c, right(lang, i, c)), cell])
    body = (_table(rows_, w["head"])
            + f'<div style="{_UI}">{html.escape(w["summary"].format(n=len(mine), closed=closed, mine=got))}</div>'
            + _note(w["note"]))
    return yours.figure(w["title"], body, alt=w["title"])


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
TOY_SPAN = {"en": ("The library opens at 10:00.", "10:00"), "nl": ("De bibliotheek gaat om 10.00 uur open.", "10.00 uur")}
TOY_PICK = {"en": ("When does the pool open?", ["The library opens at 10:00.", "The pool opens at 9:00."]),
            "nl": ("Wanneer gaat het zwembad open?", ["De bibliotheek gaat om 10.00 uur open.", "Het zwembad gaat om 9.00 uur open."])}

_FS_HEAD = "def find_span(text, answer):\n    for start in range(len(text) - len(answer) + 1):\n"
_FS_TEST = "        if text[start:start + len(answer)] == answer:\n"
_FS_RET = "            return (start, start + len(answer))\n"


def _fs(head: str = _FS_HEAD, test: str = _FS_TEST, ret: str = _FS_RET, end: str = "    return None\n") -> str:
    return head + test + ret + end


_FS = _fs()
_FS_START = _FS_HEAD + "        ...\n    return None\n"

_BT_HEAD = "def best_text(question, texts):\n    asked = set(words(question))\n    best = 0\n    most = -1\n    for k in range(len(texts)):\n"
_BT_COUNT = "        shared = len([w for w in asked if w in words(texts[k])])\n"
_BT_TEST = "        if shared > most:\n"
_BT_KEEP = "            best = k\n            most = shared\n"


def _bt(count: str = _BT_COUNT, test: str = _BT_TEST, keep: str = _BT_KEEP, end: str = "    return best\n") -> str:
    return _BT_HEAD + count + test + keep + end


_BT = _bt()
_BT_START = _BT_HEAD + "        ...\n    return best\n"

_OD_HEAD = "def open_domain(question, texts):\n"
_OD = _OD_HEAD + "    return ask(question, texts[best_text(question, texts)])\n"
_OD_START = _OD_HEAD + "    ...\n"


def _course_val(lang: str, k: int) -> _Val:
    return _Val(f'C.TEXTS["{lang}"][{k}]', TEXTS[lang][k])


def _texts_val(lang: str, start: int = 0) -> _Val:
    return _Val(f'C.TEXTS["{lang}"]' + (f"[{start}:]" if start else ""), list(TEXTS[lang][start:]))


def _question_val(lang: str, i: int) -> _Val:
    return _written(QUESTIONS[lang][i][1])


def _not_span(got, text_) -> bool:
    return got is not None and not (isinstance(got, tuple) and len(got) == 2)


def _not_int(got, text_) -> bool:
    return not isinstance(got, int) or isinstance(got, bool)


def _not_str(got, text_) -> bool:
    return not isinstance(got, str)


def _not_recorded(kind: str):
    return lambda e: isinstance(e, NotRecorded) and e.kind == kind


REQUIRED = ["find_span", "best_text", "open_domain", "which_kind"]

EXERCISES = {
    "find_span": FunctionBox(
        name="find_span",
        picture=span_picture,
        function="find_span",
        start=_FS_START,
        solution=_FS,
        show=_show,
        visible=[(_written(TOY_SPAN["en"][0]), _written(TOY_SPAN["en"][1])),
                 (_written(TOY_SPAN["nl"][0]), _written(TOY_SPAN["nl"][1]))],
        hidden=[(_written("Open until 21:00"), _written("21:00")), (_written("The pool opens at 9:00."), _written("10:00")),
                (_written("Mercury is the planet closest to the Sun."), _written("Mercury")),
                (_written("Open at 9:00, and on Sunday at 9:00 too."), _written("9:00")),
                (_course_val("en", 3), _written("14:30")), (_course_val("nl", 5), _written("€ 4,50")),
                (_course_val("en", 0), _written("two"))],
        first=[_DOTS],
        mistakes=[
            Mistake("the-text", wrong=_fs(ret="            return text[start:start + len(answer)]\n"),
                    en="`{call}` gives {got}, and it should give {want}: the answer itself, not where it stands. Which two numbers say where a span starts and ends?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het antwoord zelf, niet waar het staat. Welke twee getallen zeggen waar een span begint en eindigt?"),
            Mistake("not-span", matches=_not_span,
                    en="`{call}` gives {got}, and it should give {want}: a tuple of two numbers, where the answer starts and where it ends, or `None`. What does your `return` give back?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een tuple van twee getallen, waar het antwoord begint en waar het eindigt, of `None`. Wat geeft je `return` terug?"),
            Mistake("last-char", wrong=_fs(ret="            return (start, start + len(answer) - 1)\n"),
                    en="`{call}` gives {got}, and it should give {want}: your end is the place of the last character of the answer. In a slice, the end is the place just after it: what does `text[start:end]` give with your two numbers?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je end is de plaats van het laatste teken van het antwoord. Bij een slice is end de plaats net erna: wat geeft `text[start:end]` met je twee getallen?"),
            Mistake("length", wrong=_fs(ret="            return (start, len(answer))\n"),
                    en="`{call}` gives {got}, and it should give {want}: the second number is the length of the answer, not the place where it ends. Where does the answer end, counted from the start of the text?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het tweede getal is de lengte van het antwoord, niet de plaats waar het eindigt. Waar eindigt het antwoord, geteld vanaf het begin van de tekst?"),
            Mistake("gives-up", wrong=_fs(ret=_FS_RET + "        else:\n            return None\n"),
                    en="`{call}` gives {got}, and it should give {want}: your function stops after the first place in the text. When may it give `None`: after one place, or after all of them?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie stopt na de eerste plaats in de tekst. Wanneer mag de functie `None` geven: na één plaats, of na alle plaatsen?"),
            Mistake("short-range", wrong=_fs(head="def find_span(text, answer):\n    for start in range(len(text) - len(answer)):\n"),
                    en="`{call}` gives {got}, and it should give {want}: your function misses an answer at the very end of the text. Which is the last place where the answer can start, and does your `range` reach it?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie mist een antwoord helemaal aan het eind van de tekst. Wat is de laatste plaats waar het antwoord kan beginnen, en komt je `range` daar?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: the lines that compare the part of `text` from `start` on, as long as the answer, with `answer`, and give back `(start, end)` when they are the same.",
                      "nl": "Vervang de `...`: de regels die het stuk van `text` vanaf `start`, zo lang als het antwoord, vergelijken met `answer`, en `(start, end)` teruggeven als de twee gelijk zijn."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Where does the answer start in the text, and where is the place just after its last character?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Waar begint het antwoord in de tekst, en waar is de plaats net na het laatste teken?"},
            "almost": {"en": "Right for the two small cases. But `{call}` gives {got}, and it should give {want}. Does your function work for an answer at the start or at the end of the text, and for one that is not in it?",
                       "nl": "Goed voor de twee kleine gevallen. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie ook voor een antwoord aan het begin of het eind van de tekst, en voor een antwoord dat er niet in staat?"},
            "correct": {"en": "Your `find_span` is right in all {n} cases.",
                        "nl": "Je `find_span` klopt in alle {n} gevallen."},
        },
    ),
    "best_text": FunctionBox(
        name="best_text",
        picture=pick_picture,
        function="best_text",
        start=_BT_START,
        solution=_BT,
        show=_show,
        names={"words": words},
        visible=[(_written(TOY_PICK["en"][0]), _written(TOY_PICK["en"][1])),
                 (_written(TOY_PICK["nl"][0]), _written(TOY_PICK["nl"][1]))],
        hidden=[(_written("When does the pool open?"), _written(["The pool opens at 9:00.", "The library opens at 10:00."])),
                (_written("Is it open?"), _written(["It is open.", "Open: it is."])),
                (_written("Is it open?"), _written(["Wit: this island reopens.", "It is open."])),
                (_written("When does the pool open?"),
                 _written(["The pool is big, the pool is open, the pool is blue.", "When does the pool open? At 9:00."])),
                (_question_val("en", LIBRARY), _texts_val("en")), (_question_val("en", POOL), _texts_val("en")),
                (_question_val("nl", LIBRARY), _texts_val("nl"))],
        first=[_DOTS],
        mistakes=[
            Mistake("not-int", matches=_not_int,
                    en="`{call}` gives {got}, and it should give {want}: the place of the text in the list, a whole number. What does your `return` give back?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de plaats van de tekst in de lijst, een geheel getal. Wat geeft je `return` terug?"),
            Mistake("count", wrong=_bt(end="    return most\n"),
                    en="`{call}` gives {got}, and it should give {want}: the number of shared words of the best text, not its place in the list. Which of the two names holds the place?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het aantal gedeelde woorden van de beste tekst, niet de plaats in de lijst. Welke van de twee namen bewaart de plaats?"),
            Mistake("no-most", wrong=_bt(keep="            best = k\n"),
                    en="`{call}` gives {got}, and it should give {want}: your function always picks the last text. It changes `best`, but does it remember the highest count so far?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie kiest altijd de laatste tekst. De functie verandert `best`, maar onthoudt de functie ook het hoogste aantal tot nu toe?"),
            Mistake("last-on-tie", wrong=_bt(test="        if shared >= most:\n"),
                    en="`{call}` gives {got}, and it should give {want}: when two texts share as many words with the question, your function takes the last of them. The exercise asks for the first: which comparison keeps the first?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: als twee teksten evenveel woorden met de vraag delen, neemt je functie de laatste. De oefening vraagt om de eerste: welke vergelijking houdt de eerste vast?"),
            Mistake("letters", wrong=_bt(count="        shared = len([w for w in asked if w in texts[k]])\n"),
                    en="`{call}` gives {got}, and it should give {want}: `in` on a text looks for letters, so *it* is also found inside *Wit*. How do you look among the words of the text instead?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `in` op een tekst zoekt naar letters, dus *it* staat ook in *Wit*. Hoe zoek je in plaats daarvan tussen de woorden van de tekst?"),
            Mistake("repeats", wrong=_bt(count="        shared = len([w for w in words(texts[k]) if w in asked])\n"),
                    en="`{call}` gives {got}, and it should give {want}: your function counts a word each time it occurs in the text, so a text that repeats *the* wins. How do you count each different word of the question once?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie telt een woord elke keer dat het in de tekst staat, dus een tekst die *the* herhaalt, wint. Hoe tel je elk verschillend woord van de vraag één keer?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: count the different words of the question that are also in text `k`, and when that count is higher than `most`, remember `k` and the count.",
                      "nl": "Vervang de `...`: tel de verschillende woorden van de vraag die ook in tekst `k` staan, en als dat aantal hoger is dan `most`, onthoud dan `k` en het aantal."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. How many different words does the question share with each text, and which text comes first with the most?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Hoeveel verschillende woorden deelt de vraag met elke tekst, en welke tekst komt als eerste met de meeste?"},
            "almost": {"en": "Right for the two small cases. But `{call}` gives {got}, and it should give {want}. Does your function take the first text on a tie, and count each word of the question once?",
                       "nl": "Goed voor de twee kleine gevallen. Maar `{call}` geeft {got}, en het moet {want} zijn. Neemt je functie bij gelijkspel de eerste tekst, en telt de functie elk woord van de vraag één keer?"},
            "correct": {"en": "Your `best_text` is right in all {n} cases.",
                        "nl": "Je `best_text` klopt in alle {n} gevallen."},
        },
    ),
    "open_domain": FunctionBox(
        name="open_domain",
        picture=open_picture,
        function="open_domain",
        start=_OD_START,
        solution=_OD,
        show=_show,
        names={"best_text": best_text, "ask": ask},
        visible=[(_question_val("en", LIBRARY), _texts_val("en")), (_question_val("nl", LIBRARY), _texts_val("nl"))],
        hidden=[(_question_val("en", 4), _texts_val("en")), (_question_val("en", POOL), _texts_val("en")),
                (_question_val("en", 0), _texts_val("en", 3)), (_question_val("nl", 9), _texts_val("nl"))],
        first=[_DOTS],
        mistakes=[
            Mistake("index", error=_not_recorded("index"),
                    en="`{call}` stops: `ask` got a number, the place of a text in the list. `ask` wants the text itself: what is at that place in `texts`?",
                    nl="`{call}` stopt: `ask` kreeg een getal, de plaats van een tekst in de lijst. `ask` wil de tekst zelf: wat staat er op deze plaats in `texts`?"),
            Mistake("list", error=_not_recorded("list"),
                    en="`{call}` stops: `ask` got the whole list of texts. Open-domain question answering first picks one text: which one, and how do you take it out of `texts`?",
                    nl="`{call}` stopt: `ask` kreeg de hele lijst met teksten. Open-domain question answering kiest eerst één tekst: welke, en hoe haal je de tekst uit `texts`?"),
            Mistake("question", error=_not_recorded("question"),
                    en="`{call}` stops: the first thing `ask` got is not one of the lesson's questions. In what order does `ask` want the question and the text?",
                    nl="`{call}` stopt: het eerste wat `ask` kreeg, is geen vraag van de les. In welke volgorde wil `ask` de vraag en de tekst?"),
            Mistake("text", error=_not_recorded("text"),
                    en="`{call}` stops: `ask` got a text that is not one of the six texts. Which text of `texts` should Qwen3 read?",
                    nl="`{call}` stopt: `ask` kreeg een tekst die niet bij de zes teksten hoort. Welke tekst uit `texts` moet Qwen3 lezen?"),
            Mistake("not-str", matches=_not_str,
                    en="`{call}` gives {got}, and it should give {want}: the answer of Qwen3, a text. What does your `return` give back?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het antwoord van Qwen3, een tekst. Wat geeft je `return` terug?"),
            Mistake("closed", wrong=_OD_HEAD + "    return ask(question)\n",
                    en="`{call}` gives {got}, and it should give {want}: Qwen3's answer without a text, closed-book. Which text should `ask` get as well?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het antwoord van Qwen3 zonder tekst, closed-book. Welke tekst moet `ask` er ook bij krijgen?"),
            Mistake("first", wrong=_OD_HEAD + "    return ask(question, texts[0])\n",
                    en="`{call}` gives {got}, and it should give {want}: the answer with the first text, whatever the question. Which text shares the most words with the question?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het antwoord met de eerste tekst, wat de vraag ook is. Welke tekst deelt de meeste woorden met de vraag?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back what `ask` answers to the question, with the text of `texts` that `best_text` picks.",
                      "nl": "Vervang de `...`: geef terug wat `ask` antwoordt op de vraag, met de tekst uit `texts` die `best_text` kiest."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which text does `best_text` pick for this question, and what does `ask` answer with it?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Welke tekst kiest `best_text` bij deze vraag, en wat antwoordt `ask` met deze tekst?"},
            "almost": {"en": "Right for the two questions about the library. But `{call}` gives {got}, and it should give {want}. Which text does `best_text` pick from the `texts` your function gets, and what does `ask` answer with it?",
                       "nl": "Goed voor de twee vragen over de bibliotheek. Maar `{call}` geeft {got}, en het moet {want} zijn. Welke tekst kiest `best_text` uit de `texts` die je functie krijgt, en wat antwoordt `ask` met deze tekst?"},
            "correct": {"en": "Your `open_domain` is right in all {n} cases.",
                        "nl": "Je `open_domain` klopt in alle {n} gevallen."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, code, state, known mistake).
CASES = [
    ("find_span", _FS_START, "empty", None),
    ("find_span", _FS, "correct", None),
    ("find_span", _FS_HEAD + "        ...\n" + _FS_RET + "    return None\n", "wrong", "dots-left"),
    ("find_span", _fs(ret="            return text[start:start + len(answer)]\n"), "wrong", "the-text"),
    ("find_span", _fs(ret="            return [start, start + len(answer)]\n"), "wrong", "not-span"),
    ("find_span", _fs(ret="            return (start, start + len(answer) - 1)\n"), "wrong", "last-char"),
    ("find_span", _fs(ret="            return (start, len(answer))\n"), "wrong", "length"),
    ("find_span", _fs(ret=_FS_RET + "        else:\n            return None\n"), "wrong", "gives-up"),
    ("find_span", _fs(head="def find_span(text, answer):\n    for start in range(len(text) - len(answer)):\n"), "wrong", "short-range"),
    ("find_span", "def find_span(text, answer):\n    for start in range(len(text) - len(answer) + 1):\n"
     "        end = start + len(answer)\n        if text[start:end] == answer:\n            return (start, end)\n", "correct", None),
    ("find_span", "def find_span(text, answer):\n    start = text.find(answer)\n    if start == -1:\n        return None\n"
     "    return (start, start + len(answer))\n", "correct", None),
    ("best_text", _BT_START, "empty", None),
    ("best_text", _BT, "correct", None),
    ("best_text", _bt(keep="            ...\n"), "wrong", "dots-left"),
    ("best_text", _bt(end="    return texts[best]\n"), "wrong", "not-int"),
    ("best_text", _bt(end="    return most\n"), "wrong", "count"),
    ("best_text", _bt(keep="            best = k\n"), "wrong", "no-most"),
    ("best_text", _bt(test="        if shared >= most:\n"), "wrong", "last-on-tie"),
    ("best_text", _bt(count="        shared = len([w for w in words(texts[k]) if w in asked])\n"), "wrong", "repeats"),
    ("best_text", _bt(count="        shared = len([w for w in asked if w in texts[k]])\n"), "wrong", "letters"),
    ("best_text", _bt(count="        shared = len([w for w in asked if w in texts[k].lower()])\n"), "wrong", "letters"),
    ("best_text", _bt(count="        shared = len(asked & set(words(texts[k])))\n"), "correct", None),
    ("best_text", "def best_text(question, texts):\n    asked = set(words(question))\n"
     "    counts = [len([w for w in asked if w in words(t)]) for t in texts]\n    return counts.index(max(counts))\n", "correct", None),
    ("open_domain", _OD_START, "empty", None),
    ("open_domain", _OD, "correct", None),
    ("open_domain", _OD_HEAD + "    k = ...\n    return ask(question, texts[k])\n", "wrong", "dots-left"),
    ("open_domain", _OD_HEAD + "    return ask(question, best_text(question, texts))\n", "error", "index"),
    ("open_domain", _OD_HEAD + "    return ask(question, texts)\n", "error", "list"),
    ("open_domain", _OD_HEAD + "    return ask(texts[best_text(question, texts)], question)\n", "error", "question"),
    ("open_domain", _OD_HEAD + "    return ask(question, question)\n", "error", "text"),
    ("open_domain", _OD_HEAD + "    return best_text(question, texts)\n", "wrong", "not-str"),
    ("open_domain", _OD_HEAD + "    return ask(question)\n", "wrong", "closed"),
    ("open_domain", _OD_HEAD + "    return ask(question, texts[0])\n", "wrong", "first"),
    ("open_domain", _OD_HEAD + "    k = best_text(question, texts)\n    text = texts[k]\n    return ask(question, text)\n", "correct", None),
]


# --- the idea: the kinds of question answering ------------------------------------------------------------------------------

_KINDS_TEXT = {
    "en": {"title": "Kinds of question answering",
           "head": ["Where the answer comes from", "Points to a span (extractive)", "Writes the answer (abstractive)"],
           "rows": [("A text that comes with the question: reading comprehension",
                     "SQuAD (2016): each answer is a piece of a paragraph of Wikipedia",
                     "NarrativeQA (2018): people wrote the answers in their own words"),
                    ("Texts the system finds itself: open-domain",
                     "DrQA (2017): searches all of Wikipedia, then marks a span in what it found",
                     "A chat model that first searches and then writes its answer"),
                    ("Only what the model learned: closed-book",
                     None,
                     "T5 (2020) without a text, or a chat model without search")],
           "none": "not possible: without a text, nothing to point into",
           "note": "Each row is a place the answer comes from, each column a form of the answer. Dashed: a kind that cannot exist."},
    "nl": {"title": "Soorten question answering",
           "head": ["Waar het antwoord vandaan komt", "Wijst een span aan (extractive)", "Schrijft het antwoord (abstractive)"],
           "rows": [("Een tekst die met de vraag meekomt: reading comprehension",
                     "SQuAD (2016): elk antwoord is een stuk van een alinea van Wikipedia",
                     "NarrativeQA (2018): mensen schreven de antwoorden in eigen woorden"),
                    ("Teksten die het systeem zelf vindt: open-domain",
                     "DrQA (2017): zoekt in heel Wikipedia, en wijst dan een span aan in wat het vond",
                     "Een chat model dat eerst zoekt en dan het antwoord schrijft"),
                    ("Alleen wat het model leerde: closed-book",
                     None,
                     "T5 (2020) zonder tekst, of een chat model zonder zoeken")],
           "none": "kan niet: zonder tekst valt er niets aan te wijzen",
           "note": "Elke rij is een plek waar het antwoord vandaan komt, elke kolom een vorm van het antwoord. Gestippeld: een soort die niet kan bestaan."},
}


def kinds_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = _KINDS_TEXT[lang]
    rows_ = [[f"<b>{html.escape(a)}</b>", html.escape(b) if b else _chip(w["none"], _CHIP_NONE), html.escape(c)]
             for a, b, c in w["rows"]]
    return yours.figure(w["title"], _table(rows_, w["head"]) + _note(w["note"]), alt=w["title"])


# --- the worked example: one question in the four kinds ---------------------------------------------------------------------

_FOUR_TEXT = {
    "en": {"title": "One question, four kinds of question answering",
           "question": "The question, about the town library",
           "extractive": "Extractive, with the text: the span ({start}, {end}) in the text",
           "abstractive": "Abstractive, with the text: what Qwen3 wrote",
           "open": "Open-domain: the different words each text shares with the question, then the same reading with the text that shares the most",
           "closed": "Closed-book, without a text: what Qwen3 wrote",
           "note": "Gray: the question and the texts, as the course wrote them. Blue: the span, the text that shares the most words with the question, and an answer of Qwen3 that holds the right answer, from the course's run, seed 0. Purple: an answer that does not."},
    "nl": {"title": "Eén vraag, vier soorten question answering",
           "question": "De vraag, over de bibliotheek van het dorp",
           "extractive": "Extractive, met de tekst: de span ({start}, {end}) in de tekst",
           "abstractive": "Abstractive, met de tekst: wat Qwen3 schreef",
           "open": "Open-domain: de verschillende woorden die elke tekst met de vraag deelt, en dan hetzelfde lezen met de tekst die er de meeste deelt",
           "closed": "Closed-book, zonder tekst: wat Qwen3 schreef",
           "note": "Grijs: de vraag en de teksten, zoals de course de teksten schreef. Blauw: de span, de tekst die de meeste woorden met de vraag deelt, en een antwoord van Qwen3 met het goede antwoord erin, uit de run van de course, seed 0. Paars: een antwoord zonder het goede antwoord."},
}


def four_picture(lang: str, i: int = LIBRARY) -> str:
    lang = lang if lang in LANGS else "en"
    w = _FOUR_TEXT[lang]
    k, q, a, _ = QUESTIONS[lang][i]
    text_ = TEXTS[lang][k]
    start, end = find_span(text_, a)
    read = answer(lang, i, k)["text"]
    closed = answer(lang, i)["text"]
    picked = best_text(q, TEXTS[lang])
    counts = "".join(_chip(f"{TITLES[lang][j]}: {len(shared_words(q, t))}", _CHIP_PRED if j == picked else _CHIP_DATA)
                     for j, t in enumerate(TEXTS[lang]))
    parts = [_label(w["question"]), f'<div>{_gray(q)}</div>',
             _label(w["extractive"].format(start=start, end=end)),
             f'<div style="font-family:var(--ea-font-ui);font-size:0.84rem;line-height:1.55">{_snippet(text_, start, end, _MARK_PRED, room=len(text_))}</div>',
             _label(w["abstractive"]), f"<div>{_said(read, right(lang, i, read))}</div>",
             _label(w["open"]), f"<div>{counts}</div>",
             _label(w["closed"]), f"<div>{_said(closed, right(lang, i, closed))}</div>",
             _note(w["note"])]
    return yours.figure(w["title"], "".join(parts), alt=w["title"])


# --- the table: how often Qwen3 is right in each way -------------------------------------------------------------------------

_TABLE_TEXT = {
    "en": {"title": "Qwen3 on the twelve questions, four seeds",
           "head": ["", "About the world", "About the town", "All twelve"],
           "rows": {"closed": "Without a text (closed-book)",
                    "found": "With the text best_text picks (open-domain)",
                    "right": "With the text that answers it (reading comprehension)"},
           "of": "{n} of {total}",
           "note": "Counted by the course over the seeds 0 to 3 of its run. An answer counts as right when it holds the answer in one of the forms the course lists, such as *eight* or *8*, as a whole word; for the legs of a spider, only a difference such as *two more* or *= 2*, since an answer such as *a spider has two legs* holds *two* but is wrong."},
    "nl": {"title": "Qwen3 op de twaalf vragen, vier seeds",
           "head": ["", "Over de wereld", "Over het dorp", "Alle twaalf"],
           "rows": {"closed": "Zonder tekst (closed-book)",
                    "found": "Met de tekst die best_text kiest (open-domain)",
                    "right": "Met de tekst die de vraag beantwoordt (reading comprehension)"},
           "of": "{n} van {total}",
           "note": "Geteld door de course over de seeds 0 tot en met 3 van de run. Een antwoord telt als goed als het goede antwoord erin staat in een van de vormen die de course noemt, zoals *acht* of *8*, als heel woord; bij de poten van een spin alleen een verschil, zoals *twee meer* of *= 2*, want een antwoord als *een spin heeft twee poten* bevat *twee*, maar is fout. Een antwoord dat meer zegt, telt ook: *gesloten op zondag en dinsdag* telt als goed."},
}


def table_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = _TABLE_TEXT[lang]
    groups = [world(lang), town(lang), list(range(len(QUESTIONS[lang])))]
    rows_ = []
    for way, label in w["rows"].items():
        rows_.append([html.escape(label)] + [html.escape(w["of"].format(n=tally(lang, g, way), total=len(g) * SEEDS))
                                             for g in groups])
    body = _table(rows_, w["head"], numbers_=(1, 2, 3)) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


# --- the box: every answer of the course's run to one question ---------------------------------------------------------------

_TRY_TEXT = {
    "en": {"title": "Qwen3 on question {i}: {q}",
           "head": ["Text Qwen3 read", "Shared words", "What Qwen3 wrote, seed {seed}"],
           "none": "no text (closed-book)",
           "picked": "best_text picks this one",
           "answers": "answers the question",
           "right": "The answer as people write it: {a}",
           "note": "Gray: the number of different words each text shares with the question. Blue: an answer of Qwen3, from the course's run, that holds the right answer. Purple: an answer that does not."},
    "nl": {"title": "Qwen3 op vraag {i}: {q}",
           "head": ["Tekst die Qwen3 las", "Gedeelde woorden", "Wat Qwen3 schreef, seed {seed}"],
           "none": "geen tekst (closed-book)",
           "picked": "best_text kiest deze",
           "answers": "beantwoordt de vraag",
           "right": "Het antwoord zoals mensen het schrijven: {a}",
           "note": "Grijs: het aantal verschillende woorden dat elke tekst met de vraag deelt. Blauw: een antwoord van Qwen3, uit de run van de course, met het goede antwoord erin. Paars: een antwoord zonder het goede antwoord."},
}


def try_picture(lang: str, i, seed) -> str:
    """The box: what Qwen3 wrote to one question without a text and with each of the six texts, with one seed."""
    lang = lang if lang in LANGS else "en"
    w = _TRY_TEXT[lang]
    i, seed = int(i), int(seed)
    k_right, q, a, _ = QUESTIONS[lang][i]
    picked = best_text(q, TEXTS[lang])
    closed = answer(lang, i, None, seed)["text"]
    rows_ = [[f'<span style="{_FAINT}">{html.escape(w["none"])}</span>', "", _said(closed, right(lang, i, closed))]]
    for k, t in enumerate(TEXTS[lang]):
        said = answer(lang, i, k, seed)["text"]
        tags = [w["picked"]] if k == picked else []
        tags += [w["answers"]] if k == k_right else []
        name = f"<b>{html.escape(TITLES[lang][k])}</b>" + (_note("; ".join(tags)) if tags else "")
        rows_.append([name, f'<span style="{_DATA}">{len(shared_words(q, t))}</span>', _said(said, right(lang, i, said))])
    head = w["head"][:2] + [w["head"][2].format(seed=seed)]
    body = _label(w["right"].format(a=a)) + _table(rows_, head) + _note(w["note"])
    title = w["title"].format(i=i + 1, q=q)
    return yours.figure(title, body, alt=title)


def question_options(lang: str) -> dict:
    """The twelve questions for the dropdown of the box: their number and the question, to their place from 0."""
    return {f"{i + 1}. {q[1]}": i for i, q in enumerate(QUESTIONS[lang])}


# --- the numbers the text names ----------------------------------------------------------------------------------------------

def numbers() -> dict:
    """The numbers the lesson's text names, per language: how often Qwen3 is right in each way, over the four seeds
    and with seed 0, per group of questions."""
    out = {}
    for lang in LANGS:
        n = {"spans": sum(find_span(TEXTS[lang][k], a) is not None for k, _, a, _ in QUESTIONS[lang]),
             "picked": [best_text(q, TEXTS[lang]) for _, q, _, _ in QUESTIONS[lang]]}
        for way in ("closed", "found", "right"):
            for name, group in (("world", world(lang)), ("town", town(lang))):
                n[f"{way}_{name}"] = tally(lang, group, way)
                n[f"{way}_{name}0"] = tally(lang, group, way, [0])
        n["right0"] = [is_right(lang, i, QUESTIONS[lang][i][0]) for i in range(len(QUESTIONS[lang]))]
        n["closed0"] = [is_right(lang, i) for i in range(len(QUESTIONS[lang]))]
        n["seconds"] = sum(answer(lang, i, k, s)["seconds"] for i in range(len(QUESTIONS[lang]))
                           for k in [None, *range(len(TEXTS[lang]))] for s in range(SEEDS))
        out[lang] = n
    return out
