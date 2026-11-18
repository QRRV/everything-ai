# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-design-with-values: whether a hiring tool meets the requirements that come from three
values, sorting a value, a norm and a requirement, and the picture of a values hierarchy."""
import html
import re

from everything_ai.check import Code, CodeBox, Mistake, Sort
from everything_ai import yours

# A hiring tool, made up by the course. Each requirement comes from a value through a norm; the limit is the most the
# requirement allows, and measured is what the team found when it tested the tool.
LIMIT = {"days kept": 30, "error gap": 0.05, "no reason": 0.05}
MEASURED = {"days kept": 45, "error gap": 0.03, "no reason": 0.12}


def met(measured: dict, limit: dict) -> dict:
    """For each requirement, whether the measured value is at most its limit."""
    return {r: measured[r] <= limit[r] for r in limit}


_UI = "font-family:var(--ea-font-ui)"


def picture(labels: dict) -> str:
    """Three columns, one per value: the value, the norm under it, and the requirement under that. `labels` gives
    "title", "levels" (three words: value, norm, requirement), "columns" (three lists of three texts) and "alt"."""
    def box(text: str, level: int) -> str:
        weight = "600" if level == 0 else "400"
        colour = "var(--ea-ink)" if level != 1 else "var(--ea-ink-2)"
        return (f'<div style="border:1px solid var(--ea-line);border-radius:var(--ea-radius-small);padding:0.35rem 0.5rem;'
                f'{_UI};font-size:0.85rem;font-weight:{weight};color:{colour}">{html.escape(text)}</div>')

    arrow = f'<div style="{_UI};text-align:center;color:var(--ea-ink-3);font-size:0.85rem;line-height:1.1">↓</div>'
    columns = []
    for column in labels["columns"]:
        parts = []
        for level, text in enumerate(column):
            parts.append(f'<div style="{_UI};font-size:0.72rem;color:var(--ea-ink-3);margin:0.1rem 0">{html.escape(labels["levels"][level])}</div>')
            parts.append(box(text, level))
            if level < 2:
                parts.append(arrow)
        columns.append(f'<div style="min-width:0">{"".join(parts)}</div>')
    body = (f'<div role="img" aria-label="{html.escape(labels["alt"])}" style="display:grid;'
            f'grid-template-columns:repeat(auto-fit,minmax(min(100%,170px),1fr));gap:0.8rem">{"".join(columns)}</div>')
    return yours.figure(labels["title"], body, alt=labels["alt"])


def _met_picture(seen) -> str:
    """Each requirement with its limit and the measured value, and what the learner's code says about it."""
    mine = seen.mine
    if mine is None or not mine.has_value or not isinstance(mine.value, dict):
        return ""
    rows = []
    for requirement, limit in LIMIT.items():
        value = mine.value.get(requirement)
        if value is True:
            word = seen.word({"en": "met", "nl": "gehaald"})
        elif value is False:
            word = seen.word({"en": "not met", "nl": "niet gehaald"})
        else:
            word = "?"
        at_most = seen.word({"en": f"at most {limit:g}", "nl": f"hoogstens {limit:g}".replace(".", ",")})
        found = seen.word({"en": f"measured {MEASURED[requirement]:g}", "nl": f"gemeten {MEASURED[requirement]:g}".replace(".", ",")})
        rows.append(f'<tr><td style="padding:0.15rem 0.8rem 0.15rem 0"><code>{requirement}</code></td>'
                    f'<td style="padding:0.15rem 0.8rem;color:var(--ea-ink-2)">{at_most}</td>'
                    f'<td style="padding:0.15rem 0.8rem;color:var(--ea-ink-2)">{found}</td>'
                    f'<td style="padding:0.15rem 0;font-weight:600;color:var(--ea-ink)">{word}</td></tr>')
    title = seen.word({"en": "What your code says about each requirement", "nl": "Wat jouw code over elke eis zegt"})
    table = f'<table style="{_UI};font-size:0.9rem;border-collapse:collapse">{"".join(rows)}</table>'
    return yours.figure(title, table)


# The lesson is done when these are correct. "participation" is the choice question in the lesson.
REQUIRED = ["met", "levels", "participation"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _line(name: str, values: dict) -> str:
    return f"{name} = {values!r}".replace("'", '"')


_HEAD = (_line("limit", LIMIT) + "\n" + _line("measured", MEASURED) + "\n\nmet = {}\nfor requirement in limit:\n")
_START = _HEAD + "    ...\nmet\n"
_LINE = "    met[requirement] = measured[requirement] <= limit[requirement]\n"
_SOLUTION = _HEAD + _LINE + "met"


def _m(body: str) -> str:
    return _HEAD + body + "met"


EXERCISES = {
    "met": CodeBox(
        name="met",
        picture=_met_picture,
        start=_START,
        solution=_SOLUTION,
        # The check also runs the box with other measurements, two of them exactly at their limit.
        keep=[_line("limit", LIMIT)],
        others=[{"measured": Code('{"days kept": 30, "error gap": 0.06, "no reason": 0.05}')}],
        first=[_DOTS],
        mistakes=[
            Mistake("dicts-compared", error=lambda e: isinstance(e, TypeError) and "dict" in str(e),
                    en="Your code compares whole dictionaries. Compare the values for this one requirement: `measured[requirement]` and `limit[requirement]`.",
                    nl="Je code vergelijkt hele dictionaries. Vergelijk de waarden voor deze ene eis: `measured[requirement]` en `limit[requirement]`."),
            Mistake("appended", error=lambda e: isinstance(e, AttributeError) and "append" in str(e),
                    en="`met` is a dictionary, and a dictionary has no `append`. Put a value under a key with `met[requirement] = ...`.",
                    nl="`met` is een dictionary, en een dictionary heeft geen `append`. Zet een waarde onder een key met `met[requirement] = ...`."),
            Mistake("turned", wrong=_m("    met[requirement] = measured[requirement] > limit[requirement]\n"),
                    en="Your last line gives `{got}`: True where the tool goes over the limit. A requirement is met when the measured value is at most the limit: turn the comparison around.",
                    nl="Je laatste regel geeft `{got}`: True waar de tool over de grens gaat. Een eis is gehaald als de gemeten waarde hoogstens de grens is: draai de vergelijking om."),
            Mistake("values", wrong=_m("    met[requirement] = measured[requirement]\n"),
                    en="Your last line gives the measured values themselves. Put in `met` whether each one is at most its limit: a comparison such as `<=` gives True or False.",
                    nl="Je laatste regel geeft de gemeten waarden zelf. Zet in `met` of elke waarde hoogstens de grens is: een vergelijking zoals `<=` geeft True of False."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with one line that puts under `met[requirement]` whether the measured value is at most the limit.",
                      "nl": "Vervang de `...` door één regel die onder `met[requirement]` zet of de gemeten waarde hoogstens de grens is."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other measurements by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere metingen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `met` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `met` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does your loop put under each requirement whether `measured[requirement]` is at most `limit[requirement]`?",
                      "nl": "Je laatste regel geeft `{got}`. Zet je loop onder elke eis of `measured[requirement]` hoogstens `limit[requirement]` is?"},
            "other": {"en": "Right for this tool. But with {where}, your last line gives `{got}`, and it should give `{want}`. Does your loop compare `measured[requirement]` with `limit[requirement]`? And *at most* allows a value exactly at the limit: does your comparison allow it too?",
                      "nl": "Goed voor deze tool. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Vergelijkt je loop `measured[requirement]` met `limit[requirement]`? En *hoogstens* staat een waarde toe die precies op de grens ligt: staat jouw vergelijking dat ook toe?"},
            "other_error": {"en": "Right for this tool. But with {where}, your code stops with an error. Do you compare `measured` with `limit` for each requirement?",
                            "nl": "Goed voor deze tool. Maar met {where} stopt je code met een fout. Vergelijk je `measured` met `limit` voor elke eis?"},
            "correct": {"en": "`{got}`: the tool meets the requirement that comes from fairness, and fails the two from privacy and transparency. Each failure now points to a value, so the team knows what is at stake and what to fix. The check also tried values exactly at the limit.",
                        "nl": "`{got}`: de tool haalt de eis die uit fairness komt, en zakt voor de twee uit privacy en transparantie. Elke mislukte eis wijst nu naar een waarde, dus het team weet wat er op het spel staat en wat het moet repareren. De check probeerde ook waarden precies op de grens."},
        },
    ),
    "levels": Sort(
        name="levels",
        groups={
            "value": {"en": "Value", "nl": "Waarde"},
            "norm": {"en": "Norm", "nl": "Norm"},
            "requirement": {"en": "Requirement", "nl": "Eis"},
        },
        items={
            "autonomy": {"en": "**Autonomy.**", "nl": "**Autonomie.**"},
            "choose": {"en": "**Choose.** Users choose what the app does with their data.",
                       "nl": "**Kiezen.** Gebruikers kiezen wat de app met hun data doet."},
            "off": {"en": "**Off.** Every data setting is off until the user turns it on.",
                    "nl": "**Uit.** Elke instelling voor data staat uit tot de gebruiker hem aanzet."},
            "safety": {"en": "**Safety.**", "nl": "**Veiligheid.**"},
            "hurt": {"en": "**No hurt.** The app never gives advice that can hurt someone.",
                     "nl": "**Geen schade.** De app geeft nooit advies dat iemand schade kan doen."},
            "list": {"en": "**A list.** Every answer about medicines is tested against a list of 200 dangerous combinations.",
                     "nl": "**Een lijst.** Elk antwoord over medicijnen wordt getoetst aan een lijst van 200 gevaarlijke combinaties."},
        },
        solution={"autonomy": "value", "safety": "value", "choose": "norm", "hurt": "norm", "off": "requirement", "list": "requirement"},
        nudge={
            "autonomy": {"en": "One word, broad enough to mean many things. Which level is that?", "nl": "Eén woord, breed genoeg om veel te betekenen. Welk niveau is dat?"},
            "safety": {"en": "One word, broad enough to mean many things. Which level is that?", "nl": "Eén woord, breed genoeg om veel te betekenen. Welk niveau is dat?"},
            "choose": {"en": "It says what the value asks of this app, but could an engineer test it as it stands?", "nl": "Het zegt wat de waarde van deze app vraagt, maar kan een engineer het zo testen?"},
            "hurt": {"en": "Never giving harmful advice is a rule, but how would you test it?", "nl": "Nooit schadelijk advies geven is een regel, maar hoe zou je dat testen?"},
            "off": {"en": "You can look at every setting and see whether it is off. Which level is that?", "nl": "Je kunt bij elke instelling kijken of hij uit staat. Welk niveau is dat?"},
            "list": {"en": "A test against a fixed list can pass or fail. Which level is that?", "nl": "Een toets aan een vaste lijst kan slagen of zakken. Welk niveau is dat?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the statements again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de uitspraken opnieuw uit."},
            "empty": {"en": "Put each statement at its level of the values hierarchy.",
                      "nl": "Zet elke uitspraak op haar niveau van de waardenhiërarchie."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The statement *{item}* belongs at another level.",
                      "nl": "De uitspraak *{item}* hoort op een ander niveau."},
            "correct": {"en": "All {n} are right. Autonomy leads to letting users choose, and that to settings that are off until turned on; safety leads to no harmful advice, and that to a test against a list. Each step down is a choice the team can explain.",
                        "nl": "Alle {n} kloppen. Autonomie leidt tot gebruikers laten kiezen, en dat tot instellingen die uit staan tot ze aangezet worden; veiligheid leidt tot geen schadelijk advies, en dat tot een toets aan een lijst. Elke stap omlaag is een keuze die het team kan uitleggen."},
        },
    ),
}

_SORTED = {"autonomy": "value", "safety": "value", "choose": "norm", "hurt": "norm", "off": "requirement", "list": "requirement"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("met", _START, "empty", None),
    ("met", _SOLUTION, "correct", None),
    ("met", _m("    met[requirement] = limit[requirement] >= measured[requirement]\n"), "correct", None),
    ("met", _m("    met[requirement] = measured[requirement] < limit[requirement]\n"), "almost", "other-value"),
    ("met", _m("    met[requirement] = measured[requirement] > limit[requirement]\n"), "wrong", "turned"),
    ("met", _m("    met[requirement] = measured[requirement]\n"), "wrong", "values"),
    ("met", _m("    met[requirement] = measured <= limit\n"), "error", "dicts-compared"),
    ("met", _m("    met.append(measured[requirement] <= limit[requirement])\n"), "error", "appended"),
    ("met", _HEAD + "    ...\nmet = {}\nmet", "wrong", "dots-left"),
    ("met", _m("    met[requirement] = limit[requirement]\n"), "wrong", None),
    ("met", _HEAD + "    pass\nmet = {'days kept': False, 'error gap': True, 'no reason': False}\nmet", "almost", "other-value"),
    ("met", _SOLUTION.replace(_line("limit", LIMIT), _line("limit", {**LIMIT, "days kept": 60})), "wrong", "changed-line"),
    ("levels", {}, "empty", None),
    ("levels", _SORTED, "correct", None),
    ("levels", {"autonomy": "value", "off": "requirement"}, "wrong", "unfinished"),
    ("levels", {**_SORTED, "hurt": "requirement"}, "wrong", "wrong:hurt"),
    ("levels", {**_SORTED, "off": "norm"}, "wrong", "wrong:off"),
    ("levels", {**_SORTED, "safety": "norm"}, "wrong", "wrong:safety"),
]
