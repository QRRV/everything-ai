# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson build-make-without-code, and the course's run of Qwen3-0.6B making a small program from a
description (data/course-work-qwen.json, written by data/tools/record_course_work_qwen.py from ASKS): what each of the
16 programs gives on eight test months, the picture of that, the frames of the slider, and every number the texts say.

The lesson never runs the programs Qwen3 wrote. What they give is in OUTPUTS and WRONG below; the lesson's test runs
each program, in a separate Python with a time limit, and checks that these tables are what the programs really give."""
import calendar
import html
import json
import re

from everything_ai import paths
from everything_ai.check import Sort

COURSE = "data/course-work-qwen.json"
SEEDS = 8
MAX_TOKENS = 300
DESCRIPTIONS = {
    "short": "Write a Python function days_in_month(month, year) that returns how many days a month has. Reply with the code only.",
    "leap": ("Write a Python function days_in_month(month, year) that returns how many days a month has, "
             "also for February in leap years. Reply with the code only."),
}
ASKS = {k: {"messages": [{"role": "user", "content": d}], "max_tokens": MAX_TOKENS} for k, d in DESCRIPTIONS.items()}


def course() -> dict:
    return json.loads((paths.ROOT / COURSE).read_text(encoding="utf-8"))["parts"]["build-make-without-code"]


# --- what the 16 programs give -----------------------------------------------------------------------------------------

TESTS = [(1, 2025), (4, 2025), (8, 2025), (12, 2025), (2, 2025), (2, 2024), (2, 2000), (2, 1900)]   # (month, year)
YEARS = range(1890, 2111)   # WRONG counts the months of these years a program gets wrong: 2,652 months in all
# What each program gives for the months of TESTS, in that order, and for how many months of YEARS it is wrong.
OUTPUTS = {
    "short:0": [31, 30, 31, 31, 28, 28, 28, 28], "short:1": [28, 30, 31, 30, 29, 29, 29, 29],
    "short:2": [31, 31, 31, 31, 31, 31, 31, 31], "short:3": [31, 30, 31, 31, 28, 29, 29, 29],
    "short:4": [31, 31, 31, 31, 31, 31, 31, 31], "short:5": [31, 31, 31, 31, 31, 31, 31, 31],
    "short:6": [31, 31, 31, 31, 31, 31, 31, 31], "short:7": [31, 30, 31, 31, 28, 28, 28, 28],
    "leap:0": [31, 30, 31, 31, 29, 28, 29, 28], "leap:1": [30, 30, 30, 30, 28, 29, 29, 28],
    "leap:2": [31, 31, 31, 31, 28, 29, 29, 28], "leap:3": [31, 31, 31, 31, 28, 29, 29, 28],
    "leap:4": [31, 31, 31, 31, 28, 29, 29, 28], "leap:5": [31, 30, 30, 30, 28, 29, 29, 28],
    "leap:6": [31, 31, 31, 31, 28, 29, 29, 28], "leap:7": [30, 30, 30, 30, 28, 29, 29, 30],
}
WRONG = {"short:0": 53, "short:1": 831, "short:2": 1105, "short:3": 2, "short:4": 1105, "short:5": 1105, "short:6": 1105,
         "short:7": 53, "leap:0": 218, "leap:1": 1547, "leap:2": 884, "leap:3": 884, "leap:4": 884, "leap:5": 663,
         "leap:6": 884, "leap:7": 1549}


def program(key: str) -> str:
    """The code of one program: the first Python block of its answer."""
    d, seed = key.split(":")
    text = course()[d]["answers"][int(seed)]["text"]
    return re.search(r"```python\n(.*?)```", text, re.S).group(1)


def right(month: int, year: int) -> int:
    return calendar.monthrange(year, month)[1]


def name(key: str) -> str:
    """A1 to A8 for the programs from the short description, B1 to B8 for those from the one with leap years."""
    d, seed = key.split(":")
    return f"{'A' if d == 'short' else 'B'}{int(seed) + 1}"


def oks(key: str) -> list:
    return [got == right(m, y) for got, (m, y) in zip(OUTPUTS[key], TESTS)]


_MONTHS = {"en": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
           "nl": ["jan", "feb", "mrt", "apr", "mei", "jun", "jul", "aug", "sep", "okt", "nov", "dec"]}


def month_name(month: int, year: int, lang: str) -> str:
    return f"{_MONTHS[lang][month - 1]} {year}"


def programs_grid(lang: str) -> str:
    """The 16 programs as rows and the eight test months as columns, each cell what the program gives; a wrong number
    in the colour of mistakes, on its wash. The right number heads each column."""
    X0, CW, RH, Y0 = 150, 64, 22, 54
    parts = []
    for j, (m, y) in enumerate(TESTS):
        x = X0 + j * CW + CW / 2
        parts.append(f'<text x="{x}" y="16" text-anchor="middle" style="font-size:12px;fill:var(--ea-ink-2)">{month_name(m, y, lang)}</text>'
                     f'<text x="{x}" y="34" text-anchor="middle" style="font-size:12px;fill:var(--ea-ink);font-weight:600">{right(m, y)}</text>')
    parts.append(f'<text x="0" y="34" style="font-size:12px;fill:var(--ea-ink-2)">{ {"en": "right number", "nl": "goede getal"}[lang] }</text>')
    heads = {"short": {"en": "Description A", "nl": "Beschrijving A"}, "leap": {"en": "Description B, with leap years", "nl": "Beschrijving B, met schrikkeljaren"}}
    y = Y0
    for d in ("short", "leap"):
        parts.append(f'<text x="0" y="{y + 10}" style="font-size:12px;fill:var(--ea-ink-2);font-style:italic">{heads[d][lang]}</text>')
        y += 18
        for seed in range(SEEDS):
            key = f"{d}:{seed}"
            parts.append(f'<text x="0" y="{y + 15}" style="font-size:13px;fill:var(--ea-ink)">{ {"en": "Program", "nl": "Programma"}[lang] } {name(key)}</text>')
            for j, (got, ok) in enumerate(zip(OUTPUTS[key], oks(key))):
                x = X0 + j * CW
                if not ok:
                    parts.append(f'<rect x="{x + 14}" y="{y + 2}" width="{CW - 28}" height="{RH - 4}" rx="3" style="fill:var(--ea-loss-wash)"/>')
                style = "fill:var(--ea-loss-text);font-weight:600" if not ok else "fill:var(--ea-ink-2)"
                parts.append(f'<text x="{x + CW / 2}" y="{y + 15}" text-anchor="middle" style="font-size:13px;{style};'
                             f'font-variant-numeric:tabular-nums">{got}</text>')
            y += RH
        y += 8
    W = X0 + len(TESTS) * CW
    title = {"en": "What the 16 programs Qwen3 wrote give for eight months", "nl": "Wat de 16 programma's die Qwen3 schreef geven voor acht maanden"}[lang]
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:46rem">'
            f'<svg viewBox="0 0 {W} {y}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{title}</title>{"".join(parts)}</svg></figure>')


def findings() -> dict:
    """The numbers the text under the picture gives, from the tables."""
    keys = {d: [f"{d}:{s}" for s in range(SEEDS)] for d in ("short", "leap")}
    feb = [j for j, (m, _) in enumerate(TESTS) if m == 2]
    other = [j for j, (m, _) in enumerate(TESTS) if m != 2]
    return {
        "months": len(YEARS) * 12,
        "all_31": sum(1 for k in keys["short"] if set(OUTPUTS[k]) == {31}),
        "feb_right": {d: sum(1 for k in keys[d] if all(oks(k)[j] for j in feb)) for d in keys},
        "one_number": sum(1 for k in keys["leap"] if len({OUTPUTS[k][j] for j in other}) == 1),
        "other_wrong": {d: sum(1 for k in keys[d] if not all(oks(k)[j] for j in other)) for d in keys},
        "all_right": sum(1 for k in OUTPUTS if WRONG[k] == 0),
        "best": min(WRONG, key=WRONG.get),
    }


def grid_after(lang: str) -> str:
    F = findings()
    assert F["months"] == 2652 and F["all_31"] == 4 and F["feb_right"] == {"short": 0, "leap": 6}
    assert F["one_number"] == 6 and F["other_wrong"] == {"short": 5, "leap": 7} and F["all_right"] == 0 and F["best"] == "short:3" and WRONG["short:3"] == 2
    assert "year % 4 == 0" in program("short:3") and "100" not in program("short:3")   # leap years without the centuries
    assert sum("This is a placeholder" in program(f"short:{s}") for s in range(SEEDS) if set(OUTPUTS[f"short:{s}"]) == {31}) == 2
    months = f"{F['months']:,}".replace(",", "." if lang == "nl" else ",")
    best = name(F["best"])
    if lang == "nl":
        return (f"Bij beschrijving A gaven {F['all_31']} van de 8 programma's 31 voor elke maand, februari ook; twee ervan zeggen "
                f"het zelfs in commentaar: *This is a placeholder* (dit is een opvulling). Geen ervan had februari in alle vier de jaren goed. Bij beschrijving B, "
                f"met schrikkeljaren erin, had {F['feb_right']['leap']} van de 8 februari goed in alle vier, maar zaten er "
                f"{F['other_wrong']['leap']} naast bij een andere maand, tegen {F['other_wrong']['short']} bij A: {F['one_number']} gaven alle "
                f"andere maanden hetzelfde getal, 30 of 31. De course liet alle 16 ook de {months} maanden van "
                f"1890 tot en met 2110 uitrekenen: geen enkel programma had ze allemaal goed. Het beste, {best}, zat er alleen naast "
                f"bij februari 1900 en februari 2100.")
    return (f"For description A, {F['all_31']} of the 8 programs gave 31 for every month, February too; two of them even say "
            f"so in a comment: *This is a placeholder*. None of them got February right in all four years. For description B, with leap "
            f"years in it, {F['feb_right']['leap']} of the 8 got February right in all four, but {F['other_wrong']['leap']} got "
            f"another month wrong, against {F['other_wrong']['short']} for A: {F['one_number']} gave all the other months the same "
            f"number, 30 or 31. The course also had all 16 work out the {months} months from 1890 to 2110: "
            f"not one program got them all right. The best, {best}, was wrong only for February 1900 and February 2100.")


def program_frame(i: int, lang: str) -> tuple[str, str]:
    """The i-th program (1 to 16): its code, and what it gives for the test months, each marked right or wrong."""
    key = list(OUTPUTS)[int(i) - 1]
    cells = "".join(
        f'<span style="display:inline-block;margin:0 0.6rem 0.2rem 0;font-size:0.85rem">{month_name(m, y, lang)}: '
        f'<b style="color:{"var(--ea-ink)" if ok else "var(--ea-loss-text)"}">{got}</b>{"" if ok else " ✗"}</span>'
        for got, ok, (m, y) in zip(OUTPUTS[key], oks(key), TESTS))
    body = (f'<div style="max-width:42rem;font-family:var(--ea-font-ui)">'
            f'<pre style="font-family:var(--ea-font-code);font-size:0.8rem;background:var(--ea-inset);padding:0.5rem 0.7rem;'
            f'border-radius:6px;overflow-x:auto;margin:0 0 0.4rem">{html.escape(program(key).rstrip())}</pre>'
            f'<div>{cells}</div></div>')
    n = f"{len(YEARS) * 12:,}".replace(",", "." if lang == "nl" else ",")
    w = f"{WRONG[key]:,}".replace(",", "." if lang == "nl" else ",")
    if lang == "nl":
        said = f"**Programma {name(key)}:** fout bij {w} van de {n} maanden van 1890 tot en met 2110."
    else:
        said = f"**Program {name(key)}:** wrong for {w} of the {n} months from 1890 to 2110."
    return body, said


# --- the exercise -------------------------------------------------------------------------------------------------------

_VIBE = {"en": "If it goes wrong, who is hurt, and can you simply try again?",
         "nl": "Als het misgaat, wie heeft er dan last van, en kun je het gewoon opnieuw proberen?"}
_READ = {"en": "Does money, personal data or someone's health depend on it?",
         "nl": "Hangt er geld, persoonsgegevens of iemands gezondheid van af?"}

EXERCISES = {
    "vibe-or-read": Sort(
        name="vibe-or-read",
        groups={
            "vibe": {"en": "Fine to vibe code", "nl": "Vibe coding is prima"},
            "read": {"en": "Someone reads the code", "nl": "Iemand leest de code"},
        },
        items={
            "salaries": {"en": "**Salaries.** A program that works out the pay of the 20 people at a company.",
                         "nl": "**Salarissen.** Een programma dat het loon van de 20 mensen in een bedrijf uitrekent."},
            "photos": {"en": "**Holiday photos.** A one-off program that renames your own holiday photos by date; you have a copy.",
                       "nl": "**Vakantiefoto's.** Een programma dat je eigen vakantiefoto's één keer op datum hernoemt; je hebt een kopie."},
            "members": {"en": "**Members.** A website where the members of a club fill in their address and bank account.",
                        "nl": "**Leden.** Een website waar de leden van een club hun adres en rekeningnummer invullen."},
            "game": {"en": "**A game.** A small game for yourself, to play in your browser.",
                     "nl": "**Een spelletje.** Een klein spelletje voor jezelf, om in je browser te spelen."},
            "medicine": {"en": "**Medicine.** An app that tells patients how much of their medicine to take.",
                         "nl": "**Medicijnen.** Een app die patiënten vertelt hoeveel ze van hun medicijn moeten nemen."},
            "quiz": {"en": "**A quiz.** A quiz for a party, with questions you wrote yourself.",
                     "nl": "**Een quiz.** Een quiz voor een feestje, met vragen die je zelf schreef."},
        },
        solution={"salaries": "read", "photos": "vibe", "members": "read", "game": "vibe", "medicine": "read", "quiz": "vibe"},
        nudge={"photos": _VIBE, "game": _VIBE, "quiz": _VIBE, "salaries": _READ, "members": _READ, "medicine": _READ},
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the six programs again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de zes programma's opnieuw uit."},
            "empty": {"en": "Pick for each of the six whether vibe coding is fine.",
                      "nl": "Kies voor elk van de zes of vibe coding prima is."},
            "unfinished": {"en": "Everything you picked so far is right. Pick for the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe koos, klopt. Kies ook voor de rest (nog {n})."},
            "wrong": {"en": "“{item}” is in the wrong group.",
                      "nl": "‘{item}’ staat in de verkeerde groep."},
            "correct": {"en": "All {n} are right. When a mistake only costs you a try, vibe coding is fine: you test it by using it. When money, personal data or health depend on it, someone has to read and test the code, because the mistakes you do not see are the ones that hurt.",
                        "nl": "Alle {n} kloppen. Als een fout je alleen een poging kost, is vibe coding prima: je test het door het te gebruiken. Als er geld, persoonsgegevens of gezondheid van afhangen, moet iemand de code lezen en testen, want de fouten die je niet ziet, zijn de fouten die pijn doen."},
        },
    ),
}

# The lesson is done when these are correct. "only-recent" and "longer-description" are choice questions in the lesson.
REQUIRED = ["vibe-or-read", "only-recent", "longer-description"]

# What the tests feed each check, and what it must say: (exercise, what the learner chose, state, known mistake).
_RIGHT = {"salaries": "read", "photos": "vibe", "members": "read", "game": "vibe", "medicine": "read", "quiz": "vibe"}
CASES = [
    ("vibe-or-read", {}, "empty", None),
    ("vibe-or-read", _RIGHT, "correct", None),
    ("vibe-or-read", {"salaries": "read"}, "wrong", "unfinished"),
    ("vibe-or-read", {**_RIGHT, "salaries": "vibe"}, "wrong", "wrong:salaries"),
    ("vibe-or-read", {**_RIGHT, "quiz": "read"}, "wrong", "wrong:quiz"),
    ("vibe-or-read", {**_RIGHT, "members": "vibe"}, "wrong", "wrong:members"),
]
