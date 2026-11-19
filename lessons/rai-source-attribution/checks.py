# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-source-attribution: the course's own sources about a made-up island and questions about
them (CONTEXTS, QUESTIONS, ASKS), Qwen3-0.6B's answers with all five sources and with each one left out
(data/course-explain-qwen.json, written by data/tools/record_course_explain_qwen.py), and the three functions the
learner writes: cited (which sources an answer cites), needed (the sources without which the answer changes) and
confirmed_share (the share of citations that leaving out confirms); with the pictures of what the learner's
functions give."""
import html
import json
import re

from everything_ai import chat, paths, yours
from everything_ai.check import FunctionBox, Mistake

# --- the course's sources and questions -----------------------------------------------------------------------------
# Four short texts about a made-up island, each in five numbered sources. The library's text says one fact twice, in
# other words ([2] and [4]); the quiz night's holds two facts that every model knows.
CONTEXTS = {
    "ferry": ["The ferry to Kestrel Island leaves at 7:40 every morning.",
              "The ferry back from the island leaves at 18:15.",
              "A ticket for the ferry costs 6 euros.",
              "The trip to the island takes 35 minutes.",
              "Bikes may go on the ferry for free."],
    "library": ["The library of Kestrel Island has 4,200 books.",
                "The library opens at 10:00 on Saturdays.",
                "The library is in the old school on Harbor Street.",
                "On Saturdays, the doors of the library open at 10:00.",
                "Members can borrow up to 8 books at a time."],
    "quiz": ["The quiz night of Kestrel Island is on the first Friday of the month.",
             "Paris is the capital of France.",
             "The winner of the last quiz night got 48 points.",
             "Water boils at 100 degrees Celsius at sea level.",
             "Each team at the quiz night has at most 5 players."],
    "bakery": ["The bakery on Kestrel Island sells about 300 loaves a day.",
               "The bakery closes at 14:00.",
               "The oven of the bakery is 60 years old.",
               "A loaf of rye bread costs 3 euros.",
               "The bakery is closed on Mondays."],
}
# (text, question, the words of the right answer, the sources that say it)
QUESTIONS = [
    ("ferry", "When does the ferry to Kestrel Island leave in the morning?", "7:40", [1]),
    ("ferry", "How much does a ticket for the ferry cost?", "6 euros", [3]),
    ("ferry", "How long does the trip to the island take?", "35 minutes", [4]),
    ("library", "When does the library open on Saturdays?", "10:00", [2, 4]),
    ("library", "How many books can a member borrow at a time?", "8 books", [5]),
    ("library", "In which building is the library?", "old school", [3]),
    ("quiz", "What is the capital of France?", "Paris", [2]),
    ("quiz", "At what temperature does water boil at sea level?", "100 degrees", [4]),
    ("quiz", "How many points did the winner of the last quiz night get?", "48 points", [3]),
    ("bakery", "On which day is the bakery closed?", "Monday", [5]),
    ("bakery", "How many loaves does the bakery sell a day?", "300 loaves", [1]),
    ("bakery", "When does the bakery close?", "14:00", [2]),
]
PROMPT = ("Answer the question using only the sources below. Answer in one short sentence, and put the number of the "
          "source you used in square brackets, like [2]. If the sources do not answer the question, say so."
          "\n\nSources:\n{sources}\n\nQuestion: {question}")
SOURCES = 5
MAX_TOKENS = 80
SEEDS = 1
COURSE = "data/course-explain-qwen.json"
DOWNLOADS = {"Qwen3-0.6B": chat.FILES}   # what the demo downloads, for the line under the header


def question_text(q: int, left_out: int | None = None) -> str:
    """The prompt of question q with its five sources, or without source `left_out`; the others keep their
    numbers."""
    context, question, _, _ = QUESTIONS[q]
    lines = [f"[{i}] {s}" for i, s in enumerate(CONTEXTS[context], 1) if i != left_out]
    return PROMPT.format(sources="\n".join(lines), question=question)


def _asks() -> dict:
    """Every prompt the course asked, by key "q|all" or "q|without-i": greedy, without thinking, once."""
    out = {}
    for q in range(len(QUESTIONS)):
        for left in (None, *range(1, SOURCES + 1)):
            key = f"{q}|all" if left is None else f"{q}|without-{left}"
            out[key] = {"messages": [{"role": "user", "content": question_text(q, left)}], "max_tokens": MAX_TOKENS,
                        "temperature": 0}
    return out


ASKS = _asks()
ASK_EXAMPLE = question_text(0)   # the box of the demo starts with it


def ask_messages(text: str) -> list:
    """The learner's sources and question as the course asked its own: one message from the user."""
    return [{"role": "user", "content": text}]


def course() -> dict:
    """The course's run: every ask with its messages and its one answer."""
    return json.loads((paths.ROOT / COURSE).read_text(encoding="utf-8"))["parts"]["rai-source-attribution"]


def full(q: int) -> str:
    """Qwen3's answer to question q with all five sources."""
    return course()[f"{q}|all"]["answers"][0]["text"]


def without(q: int) -> list:
    """Qwen3's answers to question q with one source left out: a list of (that source's number, the answer)."""
    run = course()
    return [(i, run[f"{q}|without-{i}"]["answers"][0]["text"]) for i in range(1, SOURCES + 1)]


# --- the course's own versions of the three functions ---------------------------------------------------------------

def cited(text, n):
    """The numbers from 1 to n that the text cites in square brackets, smallest first."""
    found = []
    for i in range(1, n + 1):
        if f"[{i}]" in text:
            found.append(i)
    return found


def needed(without, value):
    """The numbers of the sources without which the answer no longer holds `value`."""
    found = []
    for number, text in without:
        if value not in text:
            found.append(number)
    return found


def confirmed_share(rows):
    """Of all citations, the share whose source the answer needed."""
    citations = 0
    confirmed = 0
    for row in rows:
        for number in row["cited"]:
            citations = citations + 1
            if number in row["needed"]:
                confirmed = confirmed + 1
    return confirmed / citations


def says(q: int) -> list:
    """The numbers of the sources of question q that hold the words of its right answer."""
    context, _, value, _ = QUESTIONS[q]
    return [i for i, s in enumerate(CONTEXTS[context], 1) if value in s]


def rows() -> list:
    """Per question: its number, the sources Qwen3 cited with all five, the sources the answer needed, and the
    sources that say the right answer."""
    return [{"question": q, "cited": cited(full(q), SOURCES), "needed": needed(without(q), QUESTIONS[q][2]),
             "says": says(q)} for q in range(len(QUESTIONS))]


# What Qwen3 wrote without the source it cited: still right, saying that the sources do not answer, or another answer
SAYS_NO = ["not specified", "not provided", "not given", "do not answer", "does not say", "no information"]


def kind(text: str, value: str) -> str:
    """"right" when the answer holds the words of the right answer, "says-no" when it says the sources do not answer,
    else "other"."""
    if value in text:
        return "right"
    if any(p in text.lower() for p in SAYS_NO):
        return "says-no"
    return "other"


def without_cited(q: int) -> str:
    """Qwen3's answer to question q without the one source it cited with all five."""
    (number,) = cited(full(q), SOURCES)
    return course()[f"{q}|without-{number}"]["answers"][0]["text"]


def cited_note(prompt: str, answer: str, lang: str) -> str:
    """A sentence under the demo's answer: which sources it cites, with their text from the learner's box."""
    numbers = []
    for x in re.findall(r"\[(\d+)\]", answer):
        if int(x) not in numbers:
            numbers.append(int(x))
    if not numbers:
        return {"en": "The answer cites no source in square brackets.",
                "nl": "Het antwoord citeert geen bron tussen vierkante haken."}[lang]
    parts = []
    for n in numbers:
        line = re.search(rf"^\[{n}\][ \t]*(.+)$", prompt, re.MULTILINE)
        parts.append(f"[{n}] {line.group(1).strip()}" if line else
                     {"en": f"[{n}], which is not in your box", "nl": f"[{n}], die niet in je vak staat"}[lang])
    return {"en": "It cites ", "nl": "Het citeert "}[lang] + "; ".join(parts)


class Named(list):
    """A list the check calls by a name, such as WITHOUT, so that a message stays short; made the first time it is
    used, since Qwen3's answers are read from the course's run only then. The learner's code gets a copy."""

    def __init__(self, make, name: str):
        super().__init__()
        self.make, self.made, self.name = make, False, name

    def _fill(self):
        if not self.made:
            self.made = True
            self.extend(self.make())

    def __iter__(self):
        self._fill()
        return super().__iter__()

    def __len__(self):
        self._fill()
        return super().__len__()

    def __getitem__(self, i):
        self._fill()
        return super().__getitem__(i)

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        self._fill()
        return copy.deepcopy(list(super().__iter__()), memo)


ANSWER = "The ferry to Kestrel Island leaves at 7:40 every morning [1]."   # Qwen3's answer to question 1: a test checks
WITHOUT = Named(lambda: without(0), "WITHOUT")
OPENING = Named(lambda: without(3), "OPENING")      # the library's opening time, which two sources say
ROWS = Named(rows, "ROWS")


def _row(cited_, needed_, says_):
    return {"question": 0, "cited": cited_, "needed": needed_, "says": says_}


# short lists of the check's own: an answer with two citations, one that needed two sources, one that cites none
FEW = Named(lambda: [_row([1], [1], [1]), _row([2, 4], [], [2, 4]), _row([3], [3, 5], [3])], "FEW")
SOME = Named(lambda: [_row([1, 2], [1], [1, 2]), _row([], [3], [3])], "SOME")
MIXED = Named(lambda: [_row([2], [2], [2, 5]), _row([4], [1], [1]), _row([5], [], [1])], "MIXED")


# --- the pictures ---------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.2rem"


def number(x: float, lang: str, digits: int = 3) -> str:
    text = f"{x:.{digits}f}".replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _fn(seen):
    mine = seen.mine
    if mine is None:
        return None
    fn = mine.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _call(fn, *args):
    import copy
    from everything_ai import check as checking
    return checking.with_time_limit(lambda: fn(*(copy.deepcopy(a) for a in args)))


def _numbers(v) -> list | None:
    """A list of source numbers the learner's function gave, or None when it is not a list of whole numbers."""
    if not isinstance(v, list) or any(isinstance(x, bool) or not isinstance(x, int) for x in v):
        return None
    return v


def _grid(filled: list, framed: list, words: dict, lang: str) -> str:
    """A row per question and a square per source: solid in the colour of what a model makes where `filled[q]` holds
    the source's number, with a frame in ink around the sources in `framed[q]`."""
    from everything_ai import chart
    n = len(QUESTIONS)
    panel = chart.Panel(x=(0, SOURCES + 1.6), y=(0, n + 1.4), yticks=[], xticks=[])
    for i in range(1, SOURCES + 1):
        panel.text(i + 0.95, n + 0.75, f"[{i}]", "ink-3")
    for q in range(n):
        y = n - 1 - q + 0.2
        panel.text(0.05, y + 0.32, str(q + 1), "ink-3", anchor="start")
        for i in range(1, SOURCES + 1):
            x = i + 0.6
            if i in framed[q]:
                panel.rect(x - 0.12, y - 0.12, x + 0.82, y + 0.74, role="ink")
            panel.rect(x, y, x + 0.7, y + 0.62, role="pred" if i in filled[q] else "data", solid=i in filled[q])
    svg = chart.figure([panel], alt=words["alt"], width=360, height=34 + 24 * (n + 1))
    return svg


_CITED_WORDS = {
    "en": {"title": "Your cited on Qwen3's answers",
           "key": "A row per question, a square per source: filled where your cited finds the source's number in "
                  "Qwen3's answer with all five sources, open where it does not. A frame marks the sources that say "
                  "the right answer.",
           "said": "Your cited finds {n} citations, {k} of them of a source that says the right answer.",
           "right": "The course's cited finds {n}, {k} of them of a source that says it.",
           "alt": "Your cited finds {n} citations in Qwen3's 12 answers."},
    "nl": {"title": "Jouw cited op de antwoorden van Qwen3",
           "key": "Een rij per vraag, een vierkantje per bron: gevuld waar jouw cited het nummer van de bron vindt in "
                  "het antwoord van Qwen3 met alle vijf de bronnen, open waar hij dat niet doet. Een kader wijst de "
                  "bronnen aan die het goede antwoord noemen.",
           "said": "Jouw cited vindt {n} citaties, waarvan {k} van een bron die het goede antwoord noemt.",
           "right": "De cited van de course vindt er {n}, waarvan {k} van een bron die het noemt.",
           "alt": "Jouw cited vindt {n} citaties in de 12 antwoorden van Qwen3."},
}


def cited_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _CITED_WORDS[seen.lang]
    got = []
    for q in range(len(QUESTIONS)):
        try:
            v = _numbers(_call(fn, full(q), SOURCES))
        except Exception:
            return ""
        if v is None:
            return ""
        got.append(v)
    framed = [says(q) for q in range(len(QUESTIONS))]
    want = [cited(full(q), SOURCES) for q in range(len(QUESTIONS))]

    def count(lists):
        n = sum(len(set(x for x in a if 1 <= x <= SOURCES)) for a in lists)
        k = sum(len(set(x for x in a if x in f)) for a, f in zip(lists, framed))
        return n, k

    n, k = count(got)
    svg = _grid(got, framed, {"alt": w["alt"].format(n=n)}, seen.lang)
    note = w["key"] + " " + w["said"].format(n=n, k=k)
    if [sorted(set(a)) for a in got] != want:
        n0, k0 = count(want)
        note += " " + w["right"].format(n=n0, k=k0)
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(note)}</div>')


_NEEDED_WORDS = {
    "en": {"title": "Your needed on Qwen3's answers",
           "key": "A row per question, a square per source: filled where your needed says that, without the source, "
                  "Qwen3's answer no longer holds the words of the right answer. A frame marks the source Qwen3 cited "
                  "with all five.",
           "said": "Your needed fills {n} squares; {k} of the {c} framed ones are filled.",
           "right": "The course's needed fills {n}, {k} of them framed.",
           "alt": "Your needed fills {n} squares, {k} of the {c} cited sources."},
    "nl": {"title": "Jouw needed op de antwoorden van Qwen3",
           "key": "Een rij per vraag, een vierkantje per bron: gevuld waar jouw needed zegt dat het antwoord van Qwen3 "
                  "zonder die bron de woorden van het goede antwoord niet meer bevat. Een kader wijst de bron aan die "
                  "Qwen3 citeerde met alle vijf de bronnen.",
           "said": "Jouw needed vult {n} vierkantjes; {k} van de {c} met een kader zijn gevuld.",
           "right": "De needed van de course vult er {n}, waarvan {k} met een kader.",
           "alt": "Jouw needed vult {n} vierkantjes, {k} van de {c} geciteerde bronnen."},
}


def needed_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _NEEDED_WORDS[seen.lang]
    got = []
    for q in range(len(QUESTIONS)):
        try:
            v = _numbers(_call(fn, without(q), QUESTIONS[q][2]))
        except Exception:
            return ""
        if v is None:
            return ""
        got.append(v)
    framed = [cited(full(q), SOURCES) for q in range(len(QUESTIONS))]
    want = [needed(without(q), QUESTIONS[q][2]) for q in range(len(QUESTIONS))]
    c = sum(len(f) for f in framed)

    def count(lists):
        n = sum(len(set(x for x in a if 1 <= x <= SOURCES)) for a in lists)
        k = sum(len(set(x for x in a if x in f)) for a, f in zip(lists, framed))
        return n, k

    n, k = count(got)
    svg = _grid(got, framed, {"alt": w["alt"].format(n=n, k=k, c=c)}, seen.lang)
    note = w["key"] + " " + w["said"].format(n=n, k=k, c=c)
    if [sorted(set(a)) for a in got] != want:
        n0, k0 = count(want)
        note += " " + w["right"].format(n=n0, k=k0)
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(note)}</div>')


def _bars(rows_: list, xlabel: str, alt: str, lang: str, x_max: float = 1.0, ticks: list | None = None,
          digits: int = 3) -> str:
    """Bars from 0, one per row (label, value, role), each with its value, or None without a bar."""
    from everything_ai import chart
    n = len(rows_)
    ticks = ticks or [0, 0.25, 0.5, 0.75, 1]
    panel = chart.Panel(x=(0, x_max), y=(0, 1.05 * n), xlabel=xlabel, yticks=[], xticks=ticks,
                        xtick_labels=[f"{t:g}" for t in ticks])
    for i, (label, value, role) in enumerate(rows_):
        y = 1.05 * (n - 1 - i) + 0.1
        shown = "None" if value is None else (number(value, lang, digits) if digits else str(value))
        panel.text(0.002 * x_max, y + 0.62, f"{label}: {shown}", "ink", anchor="start")
        if value is not None and value > 0:
            panel.rect(0, y, min(value, x_max), y + 0.36, role=role, solid=True)
    svg = chart.figure([panel], alt=alt, width=520, height=70 + 62 * n)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


_CONFIRMED_WORDS = {
    "en": {"title": "Your confirmed_share of Qwen3's citations",
           "xlabel": "share of the citations",
           "rows": ["confirmed by leaving the source out (your confirmed_share)",
                    "the source says the right answer (citation precision)"],
           "key": "Your confirmed_share on ROWS, the 12 questions of the course's run. Below it, the course's "
                  "citation precision of the same citations: the share whose source holds the words of the right "
                  "answer.",
           "right": "The right confirmed_share: {want}.",
           "alt": "Your confirmed_share: {got}; citation precision: {precision}."},
    "nl": {"title": "Jouw confirmed_share van de citaties van Qwen3",
           "xlabel": "deel van de citaties",
           "rows": ["bevestigd door de bron weg te laten (jouw confirmed_share)",
                    "de bron noemt het goede antwoord (citation precision)"],
           "key": "Jouw confirmed_share op ROWS, de 12 vragen uit de run van de course. Daaronder de "
                  "citation precision van de course voor dezelfde citaties: het deel waarvan de bron de woorden van "
                  "het goede antwoord bevat.",
           "right": "De goede confirmed_share: {want}.",
           "alt": "Jouw confirmed_share: {got}; citation precision: {precision}."},
}


def precision(rows_) -> float:
    """Of all citations, the share whose source says the right answer: the citation precision, by the course's
    reading."""
    citations = sum(len(r["cited"]) for r in rows_)
    return sum(1 for r in rows_ for n in r["cited"] if n in r["says"]) / citations


def confirmed_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _CONFIRMED_WORDS[seen.lang]
    try:
        v = _call(fn, rows())
    except Exception:
        return ""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1:
        return ""
    v = float(v)
    want, prec = confirmed_share(rows()), precision(rows())
    svg = _bars([(w["rows"][0], v, "pred"), (w["rows"][1], prec, "data")], w["xlabel"],
                w["alt"].format(got=number(v, seen.lang), precision=number(prec, seen.lang)), seen.lang)
    note = w["key"]
    if abs(v - want) > 1e-9:
        note += " " + w["right"].format(want=number(want, seen.lang))
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(note)}</div>')


def measured_picture(lang: str) -> str:
    """The course's figure of What Qwen3's answers show: what Qwen3 wrote without the source it cited."""
    kinds = [kind(without_cited(q), QUESTIONS[q][2]) for q in range(len(QUESTIONS))]
    words = {"en": {"title": "What Qwen3 wrote without the source it cited",
                    "rows": {"right": "still right, citing another source", "says-no": "says the sources do not answer",
                             "other": "another answer, with a citation"},
                    "xlabel": "number of the 12 questions",
                    "key": "Greedy, without thinking. Right means that the answer holds the words of the right answer, "
                           "such as 7:40 or 6 euros.",
                    "alt": "Without the source it cited: {said}."},
             "nl": {"title": "Wat Qwen3 schreef zonder de bron die hij citeerde",
                    "rows": {"right": "nog steeds goed, met een andere bron", "says-no": "zegt dat de bronnen geen "
                             "antwoord geven", "other": "een ander antwoord, met een citatie"},
                    "xlabel": "aantal van de 12 vragen",
                    "key": "Greedy, zonder denken. Goed betekent dat het antwoord de woorden van het goede antwoord "
                           "bevat, zoals 7:40 of 6 euros.",
                    "alt": "Zonder de bron die hij citeerde: {said}."}}[lang]
    order = [("right", "data"), ("says-no", "data"), ("other", "loss")]
    rows_ = [(words["rows"][k], kinds.count(k), role) for k, role in order]
    said = "; ".join(f"{label} {value}" for label, value, _ in rows_)
    svg = _bars(rows_, words["xlabel"], words["alt"].format(said=said), lang, x_max=len(QUESTIONS),
                ticks=[0, 3, 6, 9, 12], digits=0)
    return yours.figure(words["title"], svg + f'<div style="{_UI}">{html.escape(words["key"])}</div>')


# --- the exercises ----------------------------------------------------------------------------------------------------

REQUIRED = ["cited", "needed", "confirmed", "duplicate", "headline"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# exercise 1: cited
_C_HEAD = ("def cited(text, n):\n"
           "    # the numbers from 1 to n that the text cites in square brackets, smallest first\n"
           "    found = []\n"
           "    for i in range(1, n + 1):\n")
_C_TAIL = "    return found\n"
_C_START = _C_HEAD + "        ...\n" + _C_TAIL
_C = _C_HEAD + '        if f"[{i}]" in text:\n            found.append(i)\n' + _C_TAIL
_C_BARE = _C_HEAD + "        if str(i) in text:\n            found.append(i)\n" + _C_TAIL
_C_FIRST = _C_HEAD + '        if f"[{i}]" in text:\n            found.append(i)\n            return found\n' + _C_TAIL
_C_TEXT = _C_HEAD + '        if f"[{i}]" in text:\n            found.append(f"[{i}]")\n' + _C_TAIL

# exercise 2: needed
_N_HEAD = ("def needed(without, value):\n"
           "    # the numbers of the sources without which the answer no longer holds value\n"
           "    found = []\n"
           "    for number, text in without:\n")
_N_TAIL = "    return found\n"
_N_START = _N_HEAD + "        ...\n" + _N_TAIL
_N = _N_HEAD + "        if value not in text:\n            found.append(number)\n" + _N_TAIL
_N_KEPT = _N_HEAD + "        if value in text:\n            found.append(number)\n" + _N_TAIL
_N_TEXT = _N_HEAD + "        if value not in text:\n            found.append(text)\n" + _N_TAIL
_N_ALL = _N_HEAD + "        found.append(number)\n" + _N_TAIL

# exercise 3: confirmed_share
_S_HEAD = ("def confirmed_share(rows):\n"
           "    # of all citations, the share whose source the answer needed\n"
           "    citations = 0\n"
           "    confirmed = 0\n"
           "    for row in rows:\n")
_S_TAIL = "    return confirmed / citations\n"
_S_START = _S_HEAD + "        ...\n" + _S_TAIL
_S_BODY = ('        for number in row["cited"]:\n'
           '            citations = citations + 1\n'
           '            if number in row["needed"]:\n'
           '                confirmed = confirmed + 1\n')
_S = _S_HEAD + _S_BODY + _S_TAIL
_S_ROWS = _S_HEAD + ('        citations = citations + 1\n'
                     '        for number in row["cited"]:\n'
                     '            if number in row["needed"]:\n'
                     '                confirmed = confirmed + 1\n') + _S_TAIL
_S_SWAPPED = _S_HEAD + ('        for number in row["needed"]:\n'
                        '            citations = citations + 1\n'
                        '            if number in row["cited"]:\n'
                        '                confirmed = confirmed + 1\n') + _S_TAIL
_S_SAYS = _S.replace('row["needed"]', 'row["says"]')
_S_COUNT = _S.replace("    return confirmed / citations\n", "    return confirmed\n")


def show_share(v):
    """A share as the check shows it: three decimals, so that 0.917 is not 0.9166666666666666."""
    if isinstance(v, float):
        return {"en": f"{v:.3f}", "nl": f"{v:.3f}".replace(".", ",")}
    return {"en": repr(v), "nl": repr(v)}


EXERCISES = {
    "cited": FunctionBox(
        name="cited",
        picture=cited_picture,
        function="cited",
        start=_C_START,
        solution=_C,
        visible=[(ANSWER, SOURCES)],
        hidden=[("The number of books is not specified. [2] and [1] mention the library's size and opening hours.", 5),
                ("Open from 10:00 to 14:00 [3].", 5), ("It is in source [6].", 5), ("It costs [5] dollars.", 4),
                ("[5] Members can borrow up to 8 books at a time.", 5), ("The sources do not say.", 5)],
        first=[_DOTS],
        mistakes=[
            Mistake("bare", wrong=_C_BARE,
                    en="`{call}` gives {got}, and it should give {want}. Your function also counts a number that stands without square brackets, such as the 4 in 7:40. Look for each number with its brackets around it, such as `[2]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie telt ook een getal mee dat zonder vierkante haken staat, zoals de 4 in 7:40. Zoek elk getal met de haken eromheen, zoals `[2]`."),
            Mistake("first", wrong=_C_FIRST,
                    en="`{call}` gives {got}, and it should give {want}. Your function gives only the first number it finds, but an answer can cite more than one source. Let the loop go through every number from 1 to `n`, and give back `found` after the loop.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie geeft alleen het eerste getal dat hij vindt, maar een antwoord kan meer dan één bron citeren. Laat de loop elk getal van 1 tot en met `n` langsgaan, en geef `found` na de loop terug."),
            Mistake("text", wrong=_C_TEXT,
                    en="`{call}` gives {got}, and it should give {want}. Your list holds the texts with their brackets; give the numbers themselves: `found.append(i)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je lijst bevat de teksten met hun haken; geef de getallen zelf: `found.append(i)`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with two lines: when the text holds `i` in square brackets, add `i` to `found`.",
                      "nl": "Vervang de `...` door twee regels: als de tekst `i` tussen vierkante haken bevat, voeg `i` dan toe aan `found`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Look for every number from 1 to `n` with its square brackets, such as `[2]`, and add the numbers you find to `found`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Zoek elk getal van 1 tot en met `n` met zijn vierkante haken, zoals `[2]`, en voeg de getallen die je vindt toe aan `found`."},
            "almost": {"en": "Right for `cited(ANSWER, 5)`. But `{call}` gives {got}, and it should give {want}. Does your function look in the text it gets, for the numbers from 1 to `n`?",
                       "nl": "Goed voor `cited(ANSWER, 5)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Zoekt je functie in de tekst die hij krijgt, naar de getallen van 1 tot en met `n`?"},
            "correct": {"en": "Your `cited` is right in all {n} cases. The picture uses it on Qwen3's 12 answers with all five sources.",
                        "nl": "Je `cited` klopt in alle {n} gevallen. Het plaatje gebruikt hem op de 12 antwoorden van Qwen3 met alle vijf de bronnen."},
        },
    ),
    "needed": FunctionBox(
        name="needed",
        picture=needed_picture,
        function="needed",
        start=_N_START,
        solution=_N,
        visible=[(WITHOUT, "7:40")],
        hidden=[(OPENING, "10:00"), ([(1, "It opens at 9."), (2, "It opens at 10."), (3, "It opens at 10.")], "10"),
                ([(1, "Yes."), (2, "No.")], "Maybe"), ([(2, "It costs 6 euros."), (4, "It costs 5 euros.")], "6 euros")],
        first=[_DOTS],
        mistakes=[
            Mistake("kept", wrong=_N_KEPT,
                    en="`{call}` gives {got}, and it should give {want}. Your function gives the sources without which the answer still holds `value`: the ones it could do without. The question is the other way round: `if value not in text:`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie geeft de bronnen zonder welke het antwoord `value` nog steeds bevat: die het kon missen. De vraag is andersom: `if value not in text:`."),
            Mistake("text", wrong=_N_TEXT,
                    en="`{call}` gives {got}, and it should give {want}. Your list holds the answers; give the numbers of the sources: `found.append(number)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je lijst bevat de antwoorden; geef de nummers van de bronnen: `found.append(number)`."),
            Mistake("all", wrong=_N_ALL,
                    en="`{call}` gives {got}, and it should give {want}. Your function gives every source. Keep only those without which the answer no longer holds `value`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie geeft elke bron. Houd alleen die over zonder welke het antwoord `value` niet meer bevat."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with two lines: when `value` does not occur in `text`, add `number` to `found`.",
                      "nl": "Vervang de `...` door twee regels: als `value` niet in `text` voorkomt, voeg `number` dan toe aan `found`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Give the numbers of the sources whose answer, with that source left out, no longer holds `value`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Geef de nummers van de bronnen waarvan het antwoord, met die bron weggelaten, `value` niet meer bevat."},
            "almost": {"en": "Right for `needed(WITHOUT, \"7:40\")`. But `{call}` gives {got}, and it should give {want}. Does your function use the list and the `value` it gets?",
                       "nl": "Goed voor `needed(WITHOUT, \"7:40\")`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de lijst en de `value` die hij krijgt?"},
            "correct": {"en": "Your `needed` is right in all {n} cases. The picture uses it on all 12 questions.",
                        "nl": "Je `needed` klopt in alle {n} gevallen. Het plaatje gebruikt hem op alle 12 vragen."},
        },
    ),
    "confirmed": FunctionBox(
        name="confirmed",
        picture=confirmed_picture,
        function="confirmed_share",
        start=_S_START,
        solution=_S,
        show=show_share,
        visible=[(ROWS,)],
        hidden=[(FEW,), (SOME,), (MIXED,)],
        first=[_DOTS],
        mistakes=[
            Mistake("rows", wrong=_S_ROWS,
                    en="`{call}` gives {got}, and it should give {want}: your function divides as if every answer had one citation. An answer can cite more than one source, or none: add 1 to `citations` for every number in `row[\"cited\"]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie deelt alsof elk antwoord één citatie had. Een antwoord kan meer dan één bron citeren, of geen: tel 1 op bij `citations` voor elk nummer in `row[\"cited\"]`."),
            Mistake("swapped", wrong=_S_SWAPPED,
                    en="`{call}` gives {got}, and it should give {want}: the share of the needed sources that are cited. The question is the other way round: of the citations, the share whose source was needed. Loop over `row[\"cited\"]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het deel van de bronnen die nodig waren, dat geciteerd is. De vraag is andersom: van de citaties, het deel waarvan de bron nodig was. Loop over `row[\"cited\"]`."),
            Mistake("says", wrong=_S_SAYS,
                    en="`{call}` gives {got}, and it should give {want}: the share of the citations whose source says the right answer, the citation precision. The question is whether leaving the source out changed the answer: `row[\"needed\"]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het deel van de citaties waarvan de bron het goede antwoord zegt, de citation precision. De vraag is of het weglaten van de bron het antwoord veranderde: `row[\"needed\"]`."),
            Mistake("count", wrong=_S_COUNT,
                    en="`{call}` gives {got}, and it should give {want}: you give back how many citations were confirmed. Give back the share: `confirmed / citations`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je geeft terug hoeveel citaties bevestigd werden. Geef het deel terug: `confirmed / citations`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with four lines: for every number in `row[\"cited\"]`, add 1 to `citations`, and when that number is also in `row[\"needed\"]`, add 1 to `confirmed`.",
                      "nl": "Vervang de `...` door vier regels: tel voor elk nummer in `row[\"cited\"]` 1 op bij `citations`, en als dat nummer ook in `row[\"needed\"]` staat, 1 bij `confirmed`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Count every citation of every row in `citations`, and in `confirmed` those whose number is in that row's `\"needed\"`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Tel elke citatie van elke rij in `citations`, en in `confirmed` die waarvan het nummer in de `\"needed\"` van die rij staat."},
            "almost": {"en": "Right for `confirmed_share(ROWS)`. But `{call}` gives {got}, and it should give {want}. Does your function count the citations, and not the rows?",
                       "nl": "Goed voor `confirmed_share(ROWS)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Telt je functie de citaties, en niet de rijen?"},
            "correct": {"en": "Your `confirmed_share` is right in all {n} cases. The picture puts it beside the citation precision of the same citations.",
                        "nl": "Je `confirmed_share` klopt in alle {n} gevallen. Het plaatje zet hem naast de citation precision van dezelfde citaties."},
        },
    ),
}

CASES = [
    ("cited", _C_START, "empty", None),
    ("cited", _C_HEAD + '        if f"[{i}]" in text:\n            ...\n' + _C_TAIL, "wrong", "dots-left"),
    ("cited", _C, "correct", None),
    ("cited", _C_HEAD + '        if "[" + str(i) + "]" in text:\n            found = found + [i]\n' + _C_TAIL,
     "correct", None),
    ("cited", _C_HEAD + '        if text.find(f"[{i}]") >= 0:\n            found.append(i)\n' + _C_TAIL, "correct", None),
    ("cited", _C_BARE, "wrong", "bare"),
    ("cited", _C_FIRST, "wrong", "first"),
    ("cited", _C_TEXT, "wrong", "text"),
    ("cited", _C_HEAD + '        if f"[{i}]" not in text:\n            found.append(i)\n' + _C_TAIL, "wrong", None),
    ("needed", _N_START, "empty", None),
    ("needed", _N_HEAD + "        if value not in text:\n            ...\n" + _N_TAIL, "wrong", "dots-left"),
    ("needed", _N, "correct", None),
    ("needed", _N_HEAD + "        if value in text:\n            continue\n        found.append(number)\n" + _N_TAIL,
     "correct", None),
    ("needed", _N_HEAD + "        if not value in text:\n            found = found + [number]\n" + _N_TAIL,
     "correct", None),
    ("needed", _N_KEPT, "wrong", "kept"),
    ("needed", _N_TEXT, "wrong", "text"),
    ("needed", _N_ALL, "wrong", "all"),
    ("needed", _N_HEAD + "        if value not in text:\n            found.append(number + 1)\n" + _N_TAIL,
     "wrong", None),
    ("confirmed", _S_START, "empty", None),
    ("confirmed", _S_HEAD + '        for number in row["cited"]:\n            ...\n' + _S_TAIL, "wrong", "dots-left"),
    ("confirmed", _S, "correct", None),
    ("confirmed", _S_HEAD + ('        citations += len(row["cited"])\n'
                             '        for number in row["cited"]:\n'
                             '            if number in row["needed"]:\n'
                             '                confirmed += 1\n') + _S_TAIL, "correct", None),
    ("confirmed", _S_ROWS, "wrong", "rows"),
    ("confirmed", _S_SWAPPED, "wrong", "swapped"),
    ("confirmed", _S_SAYS, "wrong", "says"),
    ("confirmed", _S_COUNT, "wrong", "count"),
    ("confirmed", _S_HEAD + ('        for number in row["cited"]:\n'
                             '            citations = citations + 1\n'
                             '            if number not in row["needed"]:\n'
                             '                confirmed = confirmed + 1\n') + _S_TAIL, "wrong", None),
]
