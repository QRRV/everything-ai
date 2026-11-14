# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson build-coding-workflow, and the course's run of Qwen3-0.6B writing one function from a vague
request, from a spec, from a spec with a plan first, and from the vague request with project notes before it
(data/course-work-qwen.json, written by data/tools/record_course_work_qwen.py from ASKS); what the 32 programs give,
and their picture."""
import html
import json
import re

from everything_ai import paths
from everything_ai.check import Sort

COURSE = "data/course-work-qwen.json"
SEEDS = 8
MAX_TOKENS = 500
VAGUE = "Write a Python function clean_phone(text) that cleans up a phone number."
SPEC = ("Write a Python function clean_phone(text) for Dutch phone numbers. Keep only the digits; a number that starts "
        "with +31 or 0031 starts with 0 instead; return the 10 digits as a string, or an empty string when there are not "
        "10 digits left. Examples: '06-12345678' gives '0612345678', '+31 6 1234 5678' gives '0612345678', '12345' gives ''.")
CONTEXT = ("# Project notes\n\nThis project handles Dutch phone numbers. Every phone number is stored as 10 digits, as a "
           "string, starting with 0: +31 or 0031 at the start becomes 0, and everything that is not a digit goes. "
           "A number that does not come to 10 digits is stored as an empty string.\n\n"
           "Examples: '06-12345678' becomes '0612345678', '+31 6 1234 5678' becomes '0612345678', '12345' becomes ''.")
CODE_ONLY = " Reply with the code only."
WAYS = {
    "vague": [{"role": "user", "content": VAGUE + CODE_ONLY}],
    "spec": [{"role": "user", "content": SPEC + CODE_ONLY}],
    "plan": [{"role": "user", "content": SPEC + " First write a short plan in numbered steps, then the code."}],
    "context": [{"role": "system", "content": CONTEXT}, {"role": "user", "content": VAGUE + CODE_ONLY}],
}
ASKS = {k: {"messages": m, "max_tokens": MAX_TOKENS} for k, m in WAYS.items()}


def course() -> dict:
    return json.loads((paths.ROOT / COURSE).read_text(encoding="utf-8"))["parts"]["build-coding-workflow"]


# --- what the 32 programs give ------------------------------------------------------------------------------------------

# The nine numbers the course gave every program, and what a Dutch address book needs from each. The spec's own three
# examples are among them (EXAMPLES).
INPUTS = ["06-12345678", "+31 6 1234 5678", "0031612345678", "12345", "(020) 123 4567", "+31 20 123 4567",
          "06 1234 567", "0612345678", "06-1234-56789"]
WANT = ["0612345678", "0612345678", "0612345678", "", "0201234567", "0201234567", "", "0612345678", ""]
EXAMPLES = [0, 1, 3]
PLUS_31 = [1, 2, 5]   # the numbers that start with +31 or 0031
# What each program gives for INPUTS, in that order. tests/test_build_coding_workflow.py runs every program again.
OUTPUTS = {
    "vague:0": ["0612345678", "31612345678", "0031612345678", "12345", "0201234567", "31201234567", "061234567", "0612345678", "06123456789"],
    "vague:1": ["612345678", "31612345678", "031612345678", "12345", "201234567", "31201234567", "61234567", "612345678", "6123456789"],
    "vague:2": ["06-12345678", "+31 6 1234 5678", "0031612345678", "12345", "(020) 123 4567", "+31 20 123 4567", "06 1234 567", "0612345678", "06-1234-56789"],
    "vague:3": ["06-12345678", "+31 6 1234 5678", "0031612345678", "12345", "(020) 123 4567", "+31 20 123 4567", "06 1234 567", "0612345678", "06-1234-56789"],
    "vague:4": ["612345678", "31612345678", "031612345678", "12345", "201234567", "31201234567", "61234567", "612345678", "6123456789"],
    "vague:5": ["0612345678", "31612345678", "0031612345678", "12345", "0201234567", "31201234567", "061234567", "0612345678", "06123456789"],
    "vague:6": ["612345678", "31612345678", "031612345678", "12345", "201234567", "31201234567", "61234567", "612345678", "6123456789"],
    "vague:7": ["0612345678", "31612345678", "0031612345678", "0000012345", "0201234567", "31201234567", "0061234567", "0612345678", "06123456789"],
    "spec:0": ["0612345678", "", "", "", "0201234567", "", "", "0612345678", ""],
    "spec:1": ["", "", "", "", "", "", "", "", "6123456789"],
    "spec:2": ["0612345678", "31612345678", "0031612345678", "", "0201234567", "31201234567", "", "0612345678", "06123456789"],
    "spec:3": ["0612345678", "31612345678", "0031612345678", "", "0201234567", "31201234567", "", "0612345678", "06123456789"],
    "spec:4": ["0612345678", "", "", "", "0201234567", "", "", "0612345678", ""],
    "spec:5": ["0612345678", "31612345678", "0031612345678", "", "0201234567", "31201234567", "", "0612345678", "06123456789"],
    "spec:6": ["0612345678", "31612345678", "31612345678", "", "0201234567", "31201234567", "", "0612345678", "06123456789"],
    "spec:7": ["0612345678", "31612345678", "0031612345678", "", "0201234567", "31201234567", "", "0612345678", "06123456789"],
    "plan:0": ["", "", "0031612345", "", "", "", "", "0612345678", ""],
    "plan:1": ["0612345678", "", "", "", "0201234567", "", "", "0612345678", ""],
    "plan:2": ["0612345678", "", "", "", "0201234567", "", "", "0612345678", ""],
    "plan:3": ["0612345678", "31612345678", "1612345678", "", "0201234567", "31201234567", "061234567", "0612345678", "06123456789"],
    "plan:4": ["", "31612345678", "031612345678", "", "", "31201234567", "", "", "6123456789"],
    "plan:5": ["0612345678", "", "", "", "0201234567", "", "", "0612345678", ""],
    "plan:6": ["0612345678", "31612345678", "0031612345678", "", "0201234567", "31201234567", "061234567", "0612345678", "06123456789"],
    "plan:7": ["0612345678", "", "", "", "0201234567", "", "", "0612345678", ""],
    "context:0": ["0612345678", "31612345678", "0031612345678", "012345", "0201234567", "31201234567", "0061234567", "0612345678", "06123456789"],
    "context:1": ["0612345678", "31612345678", "1612345678", "12345", "0201234567", "31201234567", "061234567", "0612345678", "06123456789"],
    "context:2": ["0612345678", "031612345678", "00031612345678", "012345", "0201234567", "031201234567", "0061234567", "0612345678", "006123456789"],
    "context:3": ["6-12345678", "+31 6 1234 5678", "31612345678", "00000", "(020) 123 4567", "+31 20 123 4567", "6 1234 567", "0", "6-1234-56789"],
    "context:4": ["0612345678", "31612345678", "0031612345678", "12345", "0201234567", "31201234567", "061234567", "0612345678", "06123456789"],
    "context:5": ["0612345678", "31612345678", "0031612345678", "12345", "0201234567", "31201234567", "061234567", "0612345678", "06123456789"],
    "context:6": ["0612345678", "31612345678", "0031612345678", "12345", "0201234567", "31201234567", "061234567", "0612345678", "06123456789"],
    "context:7": ["0612345678", "31612345678", "0031612345678", "0000012345", "0201234567", "31201234567", "0061234567", "0612345678", "06123456789"],
}
_CLAIM = re.compile(r"\+31 6 1234 5678'?\)?[^\n]{0,40}0612345678")   # the spec's second example, written out in an answer


def answer(key: str) -> str:
    way, seed = key.split(":")
    return course()[way]["answers"][int(seed)]["text"]


def program(key: str) -> str:
    """The code of one program: the first Python block of its answer."""
    return re.search(r"```python\n(.*?)```", answer(key), re.S).group(1)


def plan(key: str) -> str:
    """What a "plan first" answer wrote before its code."""
    return answer(key).split("```")[0].strip()


def claims(key: str) -> bool:
    """Whether the answer writes, outside its code, that '+31 6 1234 5678' gives '0612345678'."""
    return bool(_CLAIM.search(answer(key).replace(program(key), "")))


def oks(key: str) -> list:
    return [got == want for got, want in zip(OUTPUTS[key], WANT)]


def keys(way: str) -> list:
    return [f"{way}:{s}" for s in range(SEEDS)]


def right(way: str, j: int) -> int:
    """How many of the eight programs of one way give the right answer for input j."""
    return sum(oks(k)[j] for k in keys(way))


def total(way: str) -> int:
    return sum(sum(oks(k)) for k in keys(way))


def _n(x: int, lang: str) -> str:
    return f"{x:,}".replace(",", "." if lang == "nl" else ",")


_WAY_HEADS = {
    "vague": {"en": ["Vague", "request"], "nl": ["Vaag", "verzoek"]},
    "spec": {"en": ["Spec", "with examples"], "nl": ["Spec met", "voorbeelden"]},
    "plan": {"en": ["Spec,", "plan first"], "nl": ["Spec, eerst", "een plan"]},
    "context": {"en": ["Project notes,", "vague request"], "nl": ["Projectnotities,", "vaag verzoek"]},
}
_WAY_NAMES = {"vague": {"en": "vague request", "nl": "vaag verzoek"},
              "spec": {"en": "spec with examples", "nl": "spec met voorbeelden"},
              "plan": {"en": "spec, plan first", "nl": "spec, eerst een plan"},
              "context": {"en": "project notes, vague request", "nl": "projectnotities, vaag verzoek"}}
_EMPTY = {"en": "(empty)", "nl": "(leeg)"}


def ways_grid(lang: str) -> str:
    """The nine numbers as rows and the four ways as columns; each cell says how many of the eight programs gave the right
    answer, a cell where none did in the colour of mistakes, on its wash. The spec's three examples are in bold."""
    X1, X0, CW, RH, Y0 = 136, 250, 104, 24, 58
    of = {"en": "of", "nl": "van"}[lang]
    parts = [f'<text x="0" y="{Y0 - 10}" style="font-size:12px;fill:var(--ea-ink-2)">{ {"en": "number", "nl": "nummer"}[lang] }</text>',
             f'<text x="{X1}" y="{Y0 - 10}" style="font-size:12px;fill:var(--ea-ink-2)">{ {"en": "right answer", "nl": "goede antwoord"}[lang] }</text>']
    for c, way in enumerate(_WAY_HEADS):
        x = X0 + c * CW + CW / 2
        top, bottom = _WAY_HEADS[way][lang]
        parts.append(f'<text x="{x}" y="{Y0 - 26}" text-anchor="middle" style="font-size:12px;fill:var(--ea-ink);font-weight:600">{top}</text>'
                     f'<text x="{x}" y="{Y0 - 10}" text-anchor="middle" style="font-size:12px;fill:var(--ea-ink);font-weight:600">{bottom}</text>')
    parts.append(f'<line x1="0" x2="{X0 + 4 * CW}" y1="{Y0 - 2}" y2="{Y0 - 2}" style="stroke:var(--ea-line)"/>')
    y = Y0
    for j, (given, want) in enumerate(zip(INPUTS, WANT)):
        weight = 700 if j in EXAMPLES else 400
        parts.append(f'<text x="0" y="{y + 16}" style="font-size:12px;fill:var(--ea-ink);font-family:var(--ea-font-code);'
                     f'white-space:pre;font-weight:{weight}">{html.escape(given)}</text>')
        shown = (f'<tspan style="font-family:var(--ea-font-ui);font-style:italic;fill:var(--ea-ink-2)">{_EMPTY[lang]}</tspan>'
                 if want == "" else html.escape(want))
        parts.append(f'<text x="{X1}" y="{y + 16}" style="font-size:12px;fill:var(--ea-ink);font-family:var(--ea-font-code);'
                     f'font-weight:{weight}">{shown}</text>')
        for c, way in enumerate(_WAY_HEADS):
            k = right(way, j)
            x = X0 + c * CW
            if k == 0:
                parts.append(f'<rect x="{x + 22}" y="{y + 3}" width="{CW - 44}" height="{RH - 6}" rx="3" style="fill:var(--ea-loss-wash)"/>')
            style = "fill:var(--ea-loss-text);font-weight:600" if k == 0 else "fill:var(--ea-ink)"
            parts.append(f'<text x="{x + CW / 2}" y="{y + 16}" text-anchor="middle" style="font-size:12px;{style};'
                         f'font-variant-numeric:tabular-nums">{k} {of} 8</text>')
        y += RH
    parts.append(f'<line x1="0" x2="{X0 + 4 * CW}" y1="{y + 2}" y2="{y + 2}" style="stroke:var(--ea-line)"/>')
    parts.append(f'<text x="0" y="{y + 22}" style="font-size:12px;fill:var(--ea-ink-2)">{ {"en": "right in all", "nl": "goed in totaal"}[lang] }</text>')
    for c, way in enumerate(_WAY_HEADS):
        parts.append(f'<text x="{X0 + c * CW + CW / 2}" y="{y + 22}" text-anchor="middle" style="font-size:13px;fill:var(--ea-ink);'
                     f'font-weight:700;font-variant-numeric:tabular-nums">{total(way)} {of} 72</text>')
    y += 32
    note = {"en": "In bold: the three examples that were in the spec.", "nl": "Vet: de drie voorbeelden die in de spec stonden."}[lang]
    parts.append(f'<text x="0" y="{y + 8}" style="font-size:11px;fill:var(--ea-ink-2)">{note}</text>')
    y += 16
    W = X0 + 4 * CW
    title = {"en": "How many of the eight programs gave the right answer, for each number and each way of asking",
             "nl": "Hoeveel van de acht programma's het goede antwoord gaven, per nummer en per manier van vragen"}[lang]
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:44rem">'
            f'<svg viewBox="0 0 {W} {y}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{title}</title>{"".join(parts)}</svg></figure>')


def findings() -> dict:
    """The numbers the text under the picture gives, from the tables and the answers."""
    every = list(OUTPUTS)
    return {
        "totals": {w: total(w) for w in _WAY_HEADS},
        "plus_31": sum(oks(k)[j] for k in every for j in PLUS_31),
        "no_zero": sum(1 for k in keys("vague") if OUTPUTS[k][7] == "612345678"),
        "unchanged": sum(1 for k in keys("vague") if OUTPUTS[k] == INPUTS),
        "short": {w: right(w, 3) for w in _WAY_HEADS},
        "kept_zero": right("context", 0),
        "all_examples": sum(1 for k in every if all(oks(k)[j] for j in EXAMPLES)),
        "all_right": sum(1 for k in every if all(oks(k))),
        "plans_without": sum(1 for k in keys("plan") if "+31" not in plan(k) and "0031" not in plan(k)),
        "claims": sum(claims(k) for k in every),
    }


def grid_after(lang: str) -> str:
    F = findings()
    assert F["totals"] == {"vague": 11, "spec": 39, "plan": 38, "context": 21} and F["plus_31"] == 0
    assert F["no_zero"] == 3 and F["unchanged"] == 2 and F["short"] == {"vague": 0, "spec": 8, "plan": 8, "context": 0}
    assert F["kept_zero"] == 7 and F["all_examples"] == 0 and F["all_right"] == 0 and F["claims"] == 10
    t = F["totals"]
    if lang == "nl":
        return (f"Bij het vage verzoek kwamen {t['vague']} van de 72 nummers goed uit: het model raadde wat *opschonen* betekende. "
                f"Van de 8 programma's haalden er {F['no_zero']} de 0 vooraan weg, en {F['unchanged']} gaven de tekst onveranderd terug. "
                f"Met de spec werden het er {t['spec']}: nu gaf elk programma een lege string voor '12345'. Eerst om een plan vragen "
                f"gaf er {t['plan']}, niet beter. De projectnotities, met een verzoek dat net zo vaag was als het eerste, gaven er "
                f"{t['context']}: {F['kept_zero']} van de 8 hielden de 0 en haalden de streepjes en spaties weg, maar geen enkel programma "
                f"gaf een lege string voor '12345'. En de drie rijen met +31 en 0031: van de 32 programma's maakte er niet één een 0 "
                f"van +31, terwijl de spec het zegt, met een voorbeeld. In {F['claims']} antwoorden staat dat voorbeeld zelfs onder de "
                f"code, '+31 6 1234 5678' geeft '0612345678', alsof de code dat deed.")
    return (f"With the vague request, {t['vague']} of the 72 numbers came out right: the model guessed what *clean up* meant. "
            f"Of the 8 programs, {F['no_zero']} dropped the 0 at the start, and {F['unchanged']} gave the text back unchanged. With the "
            f"spec, {t['spec']}: now every program gave an empty string for '12345'. Asking for a plan first gave {t['plan']}, no "
            f"better. The project notes, with a request just as vague as the first, gave {t['context']}: {F['kept_zero']} of the 8 "
            f"kept the 0 and took out the dashes and spaces, but not one gave an empty string for '12345'. And the three rows with +31 "
            f"and 0031: of the 32 programs, not one turned +31 into 0, although the spec says so, with an example. In {F['claims']} "
            f"answers, that example even stands under the code, '+31 6 1234 5678' gives '0612345678', as if the code did that.")


def plans_after(lang: str) -> str:
    F = findings()
    assert F["plans_without"] == 4
    if lang == "nl":
        return (f"Kijk ook naar de plannen: {F['plans_without']} van de 8 noemen +31 helemaal niet. Wie het plan las, had dat "
                f"gezien voordat er één regel code was. Een plan helpt vooral jou: je ziet wat er gaat gebeuren, en wat er "
                f"ontbreekt, terwijl verbeteren nog niets kost.")
    return (f"Look at the plans too: {F['plans_without']} of the 8 do not mention +31 at all. Anyone who read the plan would have "
            f"seen that before a single line of code existed. A plan helps you most of all: you see what is going to happen, and "
            f"what is missing, while fixing it still costs nothing.")


def program_frame(i: int, lang: str) -> tuple[str, str]:
    """The i-th program (1 to 32): for a "plan first" answer its plan, then its code, and what it gives for the nine numbers,
    each marked right or wrong."""
    key = list(OUTPUTS)[int(i) - 1]
    way, seed = key.split(":")
    rows = "".join(
        f'<tr><td style="font-family:var(--ea-font-code);padding:0 0.8rem 0 0;white-space:pre">{html.escape(given)}</td>'
        f'<td style="font-family:var(--ea-font-code);padding:0 0.6rem 0 0;white-space:pre;color:{"var(--ea-ink)" if ok else "var(--ea-loss-text)"};'
        f'font-weight:{400 if ok else 600}">{html.escape(got) if got else f"<i style=\"font-family:var(--ea-font-ui)\">{_EMPTY[lang]}</i>"}</td>'
        f'<td style="color:var(--ea-loss-text)">{"" if ok else "✗"}</td></tr>'
        for given, got, ok in zip(INPUTS, OUTPUTS[key], oks(key)))
    box = ("font-family:var(--ea-font-code);font-size:0.8rem;background:var(--ea-inset);padding:0.5rem 0.7rem;"
           "border-radius:6px;overflow-x:auto;margin:0 0 0.4rem;white-space:pre-wrap")
    planned = (f'<div style="font-size:0.8rem;color:var(--ea-ink-2);margin-bottom:0.15rem">{ {"en": "The plan it wrote first:", "nl": "Het plan dat het eerst schreef:"}[lang] }</div>'
               f'<pre style="{box};max-height:13rem;overflow-y:auto">{html.escape(plan(key))}</pre>') if way == "plan" else ""
    body = (f'<div style="max-width:42rem;font-family:var(--ea-font-ui)">{planned}'
            f'<pre style="{box}">{html.escape(program(key).rstrip())}</pre>'
            f'<table style="font-size:0.85rem;border-collapse:collapse">{rows}</table></div>')
    n = sum(oks(key))
    claim = claims(key)
    if lang == "nl":
        said = f"**{_WAY_NAMES[way][lang].capitalize()}, programma {int(seed) + 1} van 8:** {n} van de 9 goed."
        if claim:
            said += " Onder de code schreef het dat '+31 6 1234 5678' '0612345678' geeft; de code geeft dat niet."
    else:
        said = f"**{_WAY_NAMES[way][lang].capitalize()}, program {int(seed) + 1} of 8:** {n} of the 9 right."
        if claim:
            said += " Under the code, it wrote that '+31 6 1234 5678' gives '0612345678'; the code does not give that."
    return body, said


# --- the exercise -------------------------------------------------------------------------------------------------------

_SPEC = {"en": "Does it say what must come out, whatever way the code gets there?",
         "nl": "Zegt het wat eruit moet komen, hoe de code er ook komt?"}
_PLAN = {"en": "Is it a step on the way, in an order?",
         "nl": "Is het een stap onderweg, in een volgorde?"}
_CONTEXT = {"en": "Does it hold for every task in this project, not just this one?",
            "nl": "Geldt het voor elke taak in dit project, niet alleen voor deze?"}

EXERCISES = {
    "spec-plan-context": Sort(
        name="spec-plan-context",
        groups={
            "spec": {"en": "Spec", "nl": "Spec"},
            "plan": {"en": "Plan", "nl": "Plan"},
            "context": {"en": "AGENTS.md", "nl": "AGENTS.md"},
        },
        items={
            "first": {"en": "**First.** Turn +31 or 0031 at the start into 0, before anything else.",
                      "nl": "**Eerst.** Maak van +31 of 0031 vooraan een 0, vóór al het andere."},
            "example": {"en": "**An example.** '+31 6 1234 5678' gives '0612345678'.",
                        "nl": "**Een voorbeeld.** '+31 6 1234 5678' geeft '0612345678'."},
            "tests": {"en": "**Tests.** Run all the tests before you call any task done.",
                      "nl": "**Tests.** Draai alle tests voordat je een taak af noemt."},
            "then": {"en": "**Then.** Remove everything that is not a digit, and count what is left.",
                     "nl": "**Daarna.** Haal alles weg wat geen cijfer is, en tel wat er over is."},
            "data": {"en": "**The data folder.** Never change the files in the folder data/; they come from the club's records.",
                     "nl": "**De map data.** Verander nooit de bestanden in de map data/; ze komen uit de administratie van de club."},
            "short": {"en": "**Too short.** A number that does not come to 10 digits gives an empty string.",
                      "nl": "**Te kort.** Een nummer dat niet op 10 cijfers uitkomt, geeft een lege string."},
        },
        solution={"first": "plan", "example": "spec", "tests": "context", "then": "plan", "data": "context", "short": "spec"},
        nudge={"example": _SPEC, "short": _SPEC, "first": _PLAN, "then": _PLAN, "tests": _CONTEXT, "data": _CONTEXT},
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the six sentences again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de zes zinnen opnieuw uit."},
            "empty": {"en": "Pick a place for each of the six sentences.",
                      "nl": "Kies voor elk van de zes zinnen een plek."},
            "unfinished": {"en": "Everything you picked so far is right. Pick for the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe koos, klopt. Kies ook voor de rest (nog {n})."},
            "wrong": {"en": "“{item}” is in the wrong group.",
                      "nl": "‘{item}’ staat in de verkeerde groep."},
            "correct": {"en": "All {n} are right. The spec says what comes out, so you can check it; the plan says how, so you can correct it before there is code; AGENTS.md says what holds for every task, so you do not have to say it again each time.",
                        "nl": "Alle {n} kloppen. De spec zegt wat eruit komt, zodat je het kunt nakijken; het plan zegt hoe, zodat je het kunt verbeteren voordat er code is; AGENTS.md zegt wat voor elke taak geldt, zodat je het niet elke keer opnieuw hoeft te zeggen."},
        },
    ),
}

# The lesson is done when these are correct. "which-helped" and "plus-31" are choice questions in the lesson.
REQUIRED = ["spec-plan-context", "which-helped", "plus-31"]

# What the tests feed each check, and what it must say: (exercise, what the learner chose, state, known mistake).
_RIGHT = {"first": "plan", "example": "spec", "tests": "context", "then": "plan", "data": "context", "short": "spec"}
CASES = [
    ("spec-plan-context", {}, "empty", None),
    ("spec-plan-context", _RIGHT, "correct", None),
    ("spec-plan-context", {"example": "spec"}, "wrong", "unfinished"),
    ("spec-plan-context", {**_RIGHT, "first": "spec"}, "wrong", "wrong:first"),
    ("spec-plan-context", {**_RIGHT, "tests": "plan"}, "wrong", "wrong:tests"),
    ("spec-plan-context", {**_RIGHT, "short": "context"}, "wrong", "wrong:short"),
]
