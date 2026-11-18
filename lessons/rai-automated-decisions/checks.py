# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-automated-decisions: a function that says whether a decision letter lets someone object
(a reason, a place to object and six weeks), sorting decisions that check a fact from those that weigh circumstances,
and the picture of the letters against the three tests."""
import html
import re

from everything_ai.check import Code, CodeBox, Mistake, Sort
from everything_ai import yours

# Five decision letters of a benefits office, made up by the course: the reason the letter gives, where to object
# ("" when it says nothing), and how many days it gives for an objection. Dutch law gives six weeks: 42 days.
LETTERS = [
    {"reason": "income above the limit", "contact": "objections desk", "days": 42},
    {"reason": "", "contact": "objections desk", "days": 42},
    {"reason": "missing documents", "contact": "", "days": 42},
    {"reason": "address not found", "contact": "phone line", "days": 14},
    {"reason": "score above threshold", "contact": "objections desk", "days": 56},
]
WEEKS_6 = 42


def contestable(letter: dict) -> bool:
    """A reason, a place to object, and at least six weeks."""
    return letter["reason"] != "" and letter["contact"] != "" and letter["days"] >= WEEKS_6


def picture(labels: dict) -> str:
    """The letters against the three tests: a filled dot where a letter passes one, and the verdict beside it.
    `labels` gives "title", "letter" (the word before each number), "tests" (three short names), "yes", "no" and
    "alt"."""
    from everything_ai import chart
    rows = len(LETTERS)
    panel = chart.Panel(x=(-1.6, 4.4), y=(-0.6, rows - 0.4), title=labels["title"], yticks=[],
                        xticks=[0, 1, 2], xtick_labels=labels["tests"])
    for k, letter in enumerate(LETTERS):
        y = rows - 1 - k
        panel.text(-1.55, y - 0.12, f'{labels["letter"]} {k + 1}', "ink-2", "start")
        passed = [letter["reason"] != "", letter["contact"] != "", letter["days"] >= WEEKS_6]
        for i, ok in enumerate(passed):
            panel.points([i], [y], "data", r=6, hollow=not ok)
        panel.text(2.6, y - 0.12, labels["yes"] if contestable(letter) else labels["no"], "ink", "start")
    return chart.figure([panel], alt=labels["alt"], width=470, height=32 * rows + 70)


def _results_picture(seen) -> str:
    """Each letter with what the learner's function said, and the course's answer where it differs."""
    mine = seen.mine
    if mine is None or not mine.has_value or not isinstance(mine.value, list):
        return ""
    rows = []
    for k, letter in enumerate(LETTERS):
        got = mine.value[k] if k < len(mine.value) else "?"
        right = contestable(letter)
        mark = "" if got is right else seen.word({"en": f"(the course: {right})", "nl": f"(de course: {right})"})
        rows.append(f'<tr><td style="padding:0.15rem 0.8rem 0.15rem 0">'
                    f'{seen.word({"en": "letter", "nl": "brief"})} {k + 1}</td>'
                    f'<td style="padding:0.15rem 0.8rem;color:var(--ea-ink-2)"><code>{html.escape(letter["reason"] or chr(34) * 2)}</code></td>'
                    f'<td style="padding:0.15rem 0.8rem;font-weight:600;color:var(--ea-ink)">{html.escape(str(got))}</td>'
                    f'<td style="padding:0.15rem 0;color:var(--ea-ink-2)">{mark}</td></tr>')
    title = seen.word({"en": "What your function says about each letter", "nl": "Wat jouw functie over elke brief zegt"})
    table = f'<table style="font-family:var(--ea-font-ui);font-size:0.9rem;border-collapse:collapse">{"".join(rows)}</table>'
    return yours.figure(title, table)


# The lesson is done when these are correct. "score" is the choice question in the lesson.
REQUIRED = ["contestable", "fact", "score"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _letter(d: dict) -> str:
    return "{" + ", ".join(f'"{k}": {v!r}'.replace("'", '"') for k, v in d.items()) + "}"


_LETTERS = "letters = [\n" + "".join(f"    {_letter(d)},\n" for d in LETTERS) + "]"
_DEF = "def contestable(letter):"
_LOOP = "results = []\nfor letter in letters:\n    results.append(contestable(letter))\n"
_START = _LETTERS + "\n\n" + _DEF + "\n    ...\n\n" + _LOOP + "results\n"
_BODY = ('    if letter["reason"] == "":\n        return False\n'
         '    if letter["contact"] == "":\n        return False\n'
         '    return letter["days"] >= 42\n')
_SOLUTION = _LETTERS + "\n\n" + _DEF + "\n" + _BODY + "\n" + _LOOP + "results"


def _f(body: str) -> str:
    return _LETTERS + "\n\n" + _DEF + "\n" + body + "\n" + _LOOP + "results"


_RIGHT = [contestable(d) for d in LETTERS]

EXERCISES = {
    "contestable": CodeBox(
        name="contestable",
        picture=_results_picture,
        start=_START,
        solution=_SOLUTION,
        keep=[_LETTERS, _DEF, "results = []", "for letter in letters:", "    results.append(contestable(letter))"],
        # The check also runs the box with other letters.
        others=[{"letters": Code('[{"reason": "", "contact": "", "days": 60}, '
                                 '{"reason": "debt not paid", "contact": "objections desk", "days": 42}, '
                                 '{"reason": "income above the limit", "contact": "phone line", "days": 41}]')}],
        first=[_DOTS],
        mistakes=[
            Mistake("no-end", matches=lambda got, text: isinstance(got, list) and None in got,
                    en="Your last line gives `{got}`: for some letters your function reaches its end without a `return`, and then gives `None`. What should it give back for a letter that passes every test?",
                    nl="Je laatste regel geeft `{got}`: bij sommige brieven komt je functie aan het eind zonder `return`, en geeft dan `None`. Wat moet ze teruggeven bij een brief die elke toets haalt?"),
            Mistake("strict", wrong=_f(_BODY.replace(">= 42", "> 42")),
                    en="Your last line gives `{got}`: letter 1 gives exactly 42 days, the six weeks the law asks for, and your function turns it down. Is 42 enough?",
                    nl="Je laatste regel geeft `{got}`: brief 1 geeft precies 42 dagen, de zes weken die de wet vraagt, en je functie keurt hem af. Is 42 genoeg?"),
            Mistake("no-days", wrong=_f('    if letter["reason"] == "":\n        return False\n'
                                        '    if letter["contact"] == "":\n        return False\n    return True\n'),
                    en="Your last line gives `{got}`: letter 4 gives only 14 days, and your function lets it pass. The last test is the time: six weeks, 42 days.",
                    nl="Je laatste regel geeft `{got}`: brief 4 geeft maar 14 dagen, en je functie laat hem door. De laatste toets is de tijd: zes weken, 42 dagen."),
            Mistake("only-reason", wrong=_f('    if letter["reason"] == "":\n        return False\n    return True\n'),
                    en="Your last line gives `{got}`: your function only looks at the reason. A letter also needs a place to object, and enough time.",
                    nl="Je laatste regel geeft `{got}`: je functie kijkt alleen naar de reden. Een brief heeft ook een plek nodig om bezwaar te maken, en genoeg tijd."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the body of the function: an `if` with a `return` for each empty field, and a last `return` for the days.",
                      "nl": "Vervang de `...` door de body van de functie: een `if` met een `return` voor elk leeg veld, en een laatste `return` voor de dagen."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other letters by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere brieven."},
            "no_value": {"en": "The last line of the box shows nothing. Put `results` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `results` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does your function give back `False` for a letter without a reason or without a place to object, and otherwise whether it gives 42 days or more?",
                      "nl": "Je laatste regel geeft `{got}`. Geeft je functie `False` terug bij een brief zonder reden of zonder plek voor bezwaar, en anders of hij 42 dagen of meer geeft?"},
            "other": {"en": "Right for these letters. But with {where}, your last line gives `{got}`, and it should give `{want}`. Does your function look at the letter it gets?",
                      "nl": "Goed voor deze brieven. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Kijkt je functie naar de brief die ze krijgt?"},
            "other_error": {"en": "Right for these letters. But with {where}, your code stops with an error. Does your function look at the letter it gets?",
                            "nl": "Goed voor deze brieven. Maar met {where} stopt je code met een fout. Kijkt je functie naar de brief die ze krijgt?"},
            "correct": {"en": "`{got}`: letters 2, 3 and 4 each miss one thing, a reason, a place to object or enough time, and each of those alone is enough to make objecting hard. Letter 5 passes, but look at its reason. The check also tried other letters.",
                        "nl": "`{got}`: brieven 2, 3 en 4 missen elk één ding, een reden, een plek voor bezwaar of genoeg tijd, en elk daarvan maakt bezwaar maken al moeilijk. Brief 5 haalt het, maar kijk naar de reden. De check probeerde ook andere brieven."},
        },
    ),
    "fact": Sort(
        name="fact",
        groups={
            "fact": {"en": "Checks a fact", "nl": "Controleert een feit"},
            "judgment": {"en": "Weighs circumstances", "nl": "Weegt omstandigheden"},
        },
        items={
            "age": {"en": "**Old enough?** Whether someone is 18 or older on the first of the month.",
                    "nl": "**Oud genoeg?** Of iemand op de eerste van de maand 18 jaar of ouder is."},
            "paid": {"en": "**Paid?** Whether a parking fee was paid for a car a camera saw.",
                     "nl": "**Betaald?** Of er parkeergeld betaald is voor een auto die een camera zag."},
            "income": {"en": "**Above the limit?** Whether last year's income, as the tax office has it, is above a fixed limit.",
                       "nl": "**Boven de grens?** Of het inkomen van vorig jaar, zoals de Belastingdienst het heeft, boven een vaste grens ligt."},
            "fraud": {"en": "**Fraud?** Whether someone gave wrong information on purpose.",
                      "nl": "**Fraude?** Of iemand met opzet verkeerde informatie gaf."},
            "hardship": {"en": "**Hardship?** Whether a debt may be paid back more slowly because of illness.",
                         "nl": "**Schrijnend?** Of een schuld langzamer terugbetaald mag worden vanwege ziekte."},
            "child": {"en": "**Best for the child?** Which parent a child should live with after a divorce.",
                      "nl": "**Het beste voor het kind?** Bij welke ouder een kind na een scheiding moet wonen."},
        },
        solution={"age": "fact", "paid": "fact", "income": "fact", "fraud": "judgment", "hardship": "judgment", "child": "judgment"},
        nudge={
            "age": {"en": "Could two people who look it up come to a different answer?", "nl": "Kunnen twee mensen die het opzoeken op een ander antwoord uitkomen?"},
            "paid": {"en": "Is there anything to weigh, or only something to look up?", "nl": "Valt er iets te wegen, of alleen iets op te zoeken?"},
            "income": {"en": "The limit is fixed and the number is known. What is left to judge?", "nl": "De grens staat vast en het getal is bekend. Wat valt er nog te beoordelen?"},
            "fraud": {"en": "On purpose or by mistake: can a record tell you that?", "nl": "Met opzet of per ongeluk: kan een bestand je dat vertellen?"},
            "hardship": {"en": "Does the answer depend on this person's situation?", "nl": "Hangt het antwoord af van de situatie van deze persoon?"},
            "child": {"en": "Is there one number that settles it?", "nl": "Is er één getal dat het beslist?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the decisions again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de besluiten opnieuw uit."},
            "empty": {"en": "Put each decision with what it does.",
                      "nl": "Zet elk besluit bij wat het doet."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The decision *{item}* does something else.",
                      "nl": "Het besluit *{item}* doet iets anders."},
            "correct": {"en": "All {n} are right. A system can check a fact the same way every time, as long as people can still correct a wrong record. Weighing circumstances needs someone who can see that this case is different.",
                        "nl": "Alle {n} kloppen. Een systeem kan een feit elke keer op dezelfde manier controleren, zolang mensen een fout in het bestand nog kunnen laten verbeteren. Omstandigheden wegen vraagt iemand die kan zien dat dit geval anders is."},
        },
    ),
}

_SORTED = {"age": "fact", "paid": "fact", "income": "fact", "fraud": "judgment", "hardship": "judgment", "child": "judgment"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("contestable", _START, "empty", None),
    ("contestable", _SOLUTION, "correct", None),
    ("contestable", _f('    if letter["reason"] == "":\n        return False\n    if letter["contact"] == "":\n        return False\n'
                       '    if letter["days"] < 42:\n        return False\n    return True\n'), "correct", None),
    ("contestable", _f('    if letter["reason"] == "":\n        return False\n    if letter["contact"] == "":\n        return False\n'
                       '    if letter["days"] >= 42:\n        return True\n'), "wrong", "no-end"),
    ("contestable", _f(_BODY.replace(">= 42", "> 42")), "wrong", "strict"),
    ("contestable", _f('    if letter["reason"] == "":\n        return False\n    if letter["contact"] == "":\n        return False\n    return True\n'), "wrong", "no-days"),
    ("contestable", _f('    if letter["reason"] == "":\n        return False\n    return True\n'), "wrong", "only-reason"),
    ("contestable", _f('    return letter["days"] >= 42\n'), "wrong", None),
    ("contestable", _LETTERS + "\n\n" + _DEF + "\n" + _BODY + "\n" + _LOOP + "results = [True, False, False, False, True]\nresults",
     "almost", "other-value"),
    ("contestable", _SOLUTION.replace('"days": 14', '"days": 42'), "wrong", "changed-line"),
    ("contestable", _f("    ...\n    return True\n"), "wrong", "dots-left"),
    ("fact", {}, "empty", None),
    ("fact", _SORTED, "correct", None),
    ("fact", {"age": "fact", "fraud": "judgment"}, "wrong", "unfinished"),
    ("fact", {**_SORTED, "income": "judgment"}, "wrong", "wrong:income"),
    ("fact", {**_SORTED, "fraud": "fact"}, "wrong", "wrong:fraud"),
]
