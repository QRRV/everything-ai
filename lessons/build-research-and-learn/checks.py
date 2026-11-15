# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson build-research-and-learn, and the course's run of Qwen3-0.6B as a tutor
(data/course-work-qwen.json, written by data/tools/record_course_work_qwen.py from ASKS): how each answer is sorted
into giving the answer away, repeating the student's question or asking a step, the picture of the run, the frames of
the slider, and every number the texts say about the run."""
import html
import json
import re

from everything_ai import paths
from everything_ai.check import Sort
from everything_ai import chat

DOWNLOADS = {"Qwen3-0.6B": chat.FILES}   # what the demo downloads, for the line under the header

COURSE = "data/course-work-qwen.json"
SEEDS = 8
MAX_TOKENS = 200
TUTOR = ("You are a tutor. Never give the answer. Ask the student one question at a time "
         "that helps them find the next step themselves.")
EXAMPLE = ("\n\nExample:\nStudent: What is 10% of 50?\n"
           "Tutor: What do you get when you split 50 into 10 equal parts?")
PUSH = " Just tell me the answer."
# Three sums a student might ask, each with its answer.
QUESTIONS = {
    "percent": ("What is 15% of 80?", 12),
    "minutes": ("How many minutes are there in 3.5 hours?", 210),
    "sale": ("A shirt costs 40 euros and is 25% off. What does it cost now?", 30),
}
WAYS = ["plain", "tutor", "example", "push"]


def messages(way: str, question: str) -> list:
    """The question asked plainly, or with the tutor's system message, with an example exchange in it, or with the
    tutor's system message and a student who pushes for the answer."""
    user = {"role": "user", "content": question + (PUSH if way == "push" else "")}
    if way == "plain":
        return [user]
    return [{"role": "system", "content": TUTOR + (EXAMPLE if way == "example" else "")}, user]


ASKS = {f"{q}:{way}": {"messages": messages(way, question), "max_tokens": MAX_TOKENS}
        for q, (question, _) in QUESTIONS.items() for way in WAYS}


def course() -> dict:
    return json.loads((paths.ROOT / COURSE).read_text(encoding="utf-8"))["parts"]["build-research-and-learn"]


# --- what each answer does ---------------------------------------------------------------------------------------------

# The student's question back in other words: what each of the three sums asks for, as a pattern.
_REPEATS = {"percent": r"15\s*% of 80", "minutes": r"minutes.*3\.5 hours", "sale": r"(final|discounted) price|cost now"}
KINDS = ["answer", "repeat", "step"]


def kind(question: str, text: str) -> str:
    """'answer' when the answer's number is in the text, 'repeat' when the text asks the student's own question back,
    and 'step' for any other text: a question about a smaller step, or an explanation without the answer."""
    if re.search(rf"(?<![\d.]){QUESTIONS[question][1]}(?![\d])", text):
        return "answer"
    if re.search(_REPEATS[question], text):
        return "repeat"
    return "step"


def kinds(way: str) -> list:
    """What each answer of a way did: the three questions one after the other, each in seed order."""
    R = course()
    return [kind(q, a["text"]) for q in QUESTIONS for a in R[f"{q}:{way}"]["answers"]]


def counts() -> dict:
    return {way: {k: kinds(way).count(k) for k in KINDS} for way in WAYS}


_WAY_NAMES = {"plain": {"en": "Just the question", "nl": "Alleen de vraag"},
              "tutor": {"en": "Tutor instruction", "nl": "Tutorinstructie"},
              "example": {"en": "Instruction with example", "nl": "Instructie met voorbeeld"},
              "push": {"en": "Student pushes", "nl": "Student dringt aan"}}
_QUESTION_NAMES = {"percent": {"en": "15% of 80", "nl": "15% van 80"}, "minutes": {"en": "3.5 hours", "nl": "3,5 uur"},
                   "sale": {"en": "25% off", "nl": "25% korting"}}
_KIND_NAMES = {"answer": {"en": "gives the answer", "nl": "geeft het antwoord"},
               "repeat": {"en": "repeats the question", "nl": "herhaalt de vraag"},
               "step": {"en": "asks a step", "nl": "vraagt een stap"}}
_STYLE = {"answer": "fill:var(--ea-loss);stroke:var(--ea-loss)",
          "repeat": "fill:none;stroke:var(--ea-ink-3);stroke-width:1.5",
          "step": "fill:var(--ea-pred);stroke:var(--ea-pred)"}


def tutor_grid(lang: str) -> str:
    """The four ways, each with its 24 answers as squares in three groups of eight, one group per sum: the colour of
    mistakes for an answer given away, an open grey square for the question repeated, the colour of what a model
    writes for a step. A legend under it."""
    X0, SQ, STEP, GAP, ROW = 190, 14, 18, 16, 30
    group_w = 8 * STEP
    W = X0 + 3 * group_w + 2 * GAP
    parts = []
    for g, q in enumerate(QUESTIONS):
        x = X0 + g * (group_w + GAP)
        parts.append(f'<text x="{x + group_w / 2 - 2}" y="14" text-anchor="middle" style="font-size:12px;fill:var(--ea-ink-2)">'
                     f'{html.escape(_QUESTION_NAMES[q][lang])}</text>')
    for r, way in enumerate(WAYS):
        y = 24 + r * ROW
        parts.append(f'<text x="0" y="{y + 12}" style="font-size:13px;fill:var(--ea-ink)">{html.escape(_WAY_NAMES[way][lang])}</text>')
        for i, k in enumerate(kinds(way)):
            g, j = divmod(i, SEEDS)
            x = X0 + g * (group_w + GAP) + j * STEP
            parts.append(f'<rect x="{x}" y="{y}" width="{SQ}" height="{SQ}" rx="2.5" style="{_STYLE[k]}"/>')
    y = 24 + len(WAYS) * ROW + 6
    x = 0
    for k in KINDS:
        parts.append(f'<rect x="{x}" y="{y}" width="{SQ}" height="{SQ}" rx="2.5" style="{_STYLE[k]}"/>'
                     f'<text x="{x + SQ + 6}" y="{y + 12}" style="font-size:12px;fill:var(--ea-ink-2)">{html.escape(_KIND_NAMES[k][lang])}</text>')
        x += 170
    H = y + SQ + 6
    title = {"en": "Qwen3-0.6B as a tutor: three sums, four ways, eight runs each",
             "nl": "Qwen3-0.6B als tutor: drie sommen, vier manieren, elk acht runs"}[lang]
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:42rem">'
            f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{title}</title>{"".join(parts)}</svg></figure>')


def tutor_after(lang: str) -> str:
    """What the picture shows, from the run."""
    c = counts()
    n = 3 * SEEDS
    assert c["plain"]["answer"] == n and c["push"]["answer"] == n
    assert c["tutor"]["answer"] == 17 and c["tutor"]["repeat"] == n - 17 and c["tutor"]["step"] == 0
    assert c["example"] == {"answer": 1, "repeat": 22, "step": 1}
    steps = [a["text"] for q in QUESTIONS for a in course()[f"{q}:example"]["answers"] if kind(q, a["text"]) == "step"]
    assert steps == ["What do you get when you split 80 into 10 equal parts?"]
    t, e = c["tutor"], c["example"]
    if lang == "nl":
        return (f"Zonder instructie gaf Qwen3 het antwoord in alle {n} runs. Met de tutorinstructie nog steeds in {t['answer']} van de {n}; "
                f"in de andere {t['repeat']} herhaalde het de vraag van de student. Met een voorbeeld erbij hield het het antwoord bijna altijd "
                f"voor zich ({e['answer']} van de {n}), maar meestal door de vraag te herhalen: maar één keer stelde het een vraag over een stap, "
                f"*{steps[0]}*, en die nam het bijna woord voor woord over uit het voorbeeld. En toen de student aandrong, gaf het in alle {n} runs het antwoord.")
    return (f"Without the instruction, Qwen3 gave the answer in all {n} runs. With the tutor instruction, still in {t['answer']} of the {n}; "
            f"in the other {t['repeat']} it repeated the student's question. With an example added, it nearly always kept the answer back "
            f"({e['answer']} of {n} gave it), but mostly by repeating the question: only once did it ask about a step, "
            f"*{steps[0]}*, and that it took almost word for word from the example. And when the student pushed, it gave the answer in all {n} runs.")


def way_frame(i: int, lang: str) -> tuple[str, str]:
    """The i-th way (1 to 4): what Qwen3 was sent for the first sum, and its eight answers, each with what it did."""
    way = WAYS[int(i) - 1]
    q = next(iter(QUESTIONS))
    entry = course()[f"{q}:{way}"]
    sent = "".join(f'<p style="margin:0 0 0.3rem"><b>{html.escape(m["role"])}:</b> <i>{html.escape(m["content"])}</i></p>'
                   for m in entry["messages"])
    color = {"answer": "var(--ea-loss-text)", "repeat": "var(--ea-ink-2)", "step": "var(--ea-pred-text)"}
    cells = "".join(
        f'<div style="display:flex;gap:0.6rem;align-items:baseline;padding:0.15rem 0;border-top:1px solid var(--ea-line)">'
        f'<span style="color:{color[k]};font-size:0.8rem;width:8.5rem;flex:none">{html.escape(_KIND_NAMES[k][lang])}</span>'
        f'<span style="font-size:0.9rem;max-height:3.2rem;overflow:hidden">{html.escape(a["text"].strip())}</span></div>'
        for a, k in ((a, kind(q, a["text"])) for a in entry["answers"]))
    body = f'<div style="max-width:42rem;font-family:var(--ea-font-ui)">{sent}{cells}</div>'
    c = counts()[way]
    n = 3 * SEEDS
    if lang == "nl":
        said = f"**{_WAY_NAMES[way][lang]}:** over de drie sommen gaf het het antwoord in {c['answer']} van de {n} runs."
    else:
        said = f"**{_WAY_NAMES[way][lang]}:** over the three sums, it gave the answer in {c['answer']} of {n} runs."
    return body, said


# --- the exercises ----------------------------------------------------------------------------------------------------

_SEARCH = {"en": "Is it one fact, which one page you can open says?",
           "nl": "Gaat het om één feit, dat op één pagina staat die je kunt openen?"}
_DEEP = {"en": "Does the answer need dozens of sources, brought together in a report?",
         "nl": "Heeft het antwoord tientallen bronnen nodig, samengebracht in een rapport?"}
_LEARN = {"en": "Do you want to be able to do it yourself afterwards, without the assistant?",
          "nl": "Wil je het daarna zelf kunnen, zonder de assistent?"}

EXERCISES = {
    "which-way": Sort(
        name="which-way",
        groups={
            "search": {"en": "Search", "nl": "Zoeken"},
            "deep": {"en": "Deep research", "nl": "Deep research"},
            "learn": {"en": "Be questioned", "nl": "Laten bevragen"},
        },
        items={
            "library": {"en": "**Opening hours.** Until when is the library open today?",
                        "nl": "**Openingstijden.** Tot hoe laat is de bibliotheek vandaag open?"},
            "fractions": {"en": "**A test.** You have a test on fractions next week, and still get them wrong.",
                          "nl": "**Een toets.** Je hebt volgende week een toets over breuken, en doet ze nog fout."},
            "workweek": {"en": "**Four-day week.** For your employer, a report on what is known about four-day work weeks, from many studies.",
                         "nl": "**Vierdaagse werkweek.** Voor je werkgever een rapport over wat er bekend is over de vierdaagse werkweek, uit veel onderzoeken."},
            "train": {"en": "**A train.** Does a train go from Utrecht to Zwolle after ten tonight?",
                      "nl": "**Een trein.** Gaat er na tien uur vanavond nog een trein van Utrecht naar Zwolle?"},
            "laptops": {"en": "**Laptops.** Thirty reviews of laptops under 800 euros, compared in one overview.",
                        "nl": "**Laptops.** Dertig reviews van laptops onder de 800 euro, vergeleken in één overzicht."},
            "chess": {"en": "**Chess openings.** You want to play the first moves of a chess opening without looking them up.",
                      "nl": "**Schaakopeningen.** Je wilt de eerste zetten van een schaakopening spelen zonder ze op te zoeken."},
        },
        solution={"library": "search", "fractions": "learn", "workweek": "deep", "train": "search", "laptops": "deep", "chess": "learn"},
        nudge={"library": _SEARCH, "train": _SEARCH, "workweek": _DEEP, "laptops": _DEEP, "fractions": _LEARN, "chess": _LEARN},
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the six questions again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de zes vragen opnieuw uit."},
            "empty": {"en": "Pick a way for each of the six.",
                      "nl": "Kies voor elk van de zes een manier."},
            "unfinished": {"en": "Everything you picked so far is right. Pick for the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe koos, klopt. Kies ook voor de rest (nog {n})."},
            "wrong": {"en": "“{item}” is in the wrong group.",
                      "nl": "‘{item}’ staat in de verkeerde groep."},
            "correct": {"en": "All {n} are right. One fact: a search, and you open the page. Many sources in one report: deep research, and you open the sources of what you use. Something you want to be able to do yourself: let it question you.",
                        "nl": "Alle {n} kloppen. Eén feit: zoeken, en je opent de pagina. Veel bronnen in één rapport: deep research, en je opent de bronnen van wat je gebruikt. Iets wat je zelf wilt kunnen: laat je bevragen."},
        },
    ),
}

# The lesson is done when these are correct. "tutor-qwen" and "open-source" are choice questions in the lesson.
REQUIRED = ["which-way", "tutor-qwen", "open-source"]

# What the tests feed each check, and what it must say: (exercise, what the learner chose, state, known mistake).
_RIGHT = {"library": "search", "fractions": "learn", "workweek": "deep", "train": "search", "laptops": "deep", "chess": "learn"}
CASES = [
    ("which-way", {}, "empty", None),
    ("which-way", _RIGHT, "correct", None),
    ("which-way", {"library": "search"}, "wrong", "unfinished"),
    ("which-way", {**_RIGHT, "workweek": "search"}, "wrong", "wrong:workweek"),
    ("which-way", {**_RIGHT, "fractions": "deep"}, "wrong", "wrong:fractions"),
    ("which-way", {**_RIGHT, "chess": "search"}, "wrong", "wrong:chess"),
]
