# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-principles: which principles more than half of 84 sets of AI principles name, sorting what
different guidelines mean by transparency, and the picture of how often each principle is named."""
import re

from everything_ai.check import Code, CodeBox, Mistake, Sort
from everything_ai import yours

# Jobin, Ienca and Vayena (2019), "The global landscape of AI ethics guidelines", Table 2: in how many of the 84
# documents they found each principle.
NAMED = {"transparency": 73, "justice and fairness": 68, "non-maleficence": 60, "responsibility": 60, "privacy": 47,
         "beneficence": 41, "freedom and autonomy": 34, "trust": 28, "sustainability": 14, "dignity": 13, "solidarity": 6}
DOCUMENTS = 84


def common(named: dict, documents: int) -> list[str]:
    """The principles named in more than half of the documents."""
    return [p for p in named if named[p] / documents > 0.5]


def picture(labels: dict) -> str:
    """For each principle, in how many of the 84 documents it is named, with a line at half. `labels` gives "title",
    "names" (the principles in the order of NAMED), "half", "x" and "alt"."""
    from everything_ai import chart
    rows = len(NAMED)
    panel = chart.Panel(x=(-75, 92), y=(-0.8, rows + 0.6), title=labels["title"], xlabel=labels["x"], yticks=[],
                        xticks=[0, 21, 42, 63, 84])
    for k, (principle, count) in enumerate(NAMED.items()):
        y = rows - 1 - k
        panel.rect(0, y - 0.32, count, y + 0.32, "data")
        panel.text(-74, y - 0.18, labels["names"][k], "ink-2", "start")
        panel.text(count + 1.2, y - 0.18, str(count), "ink-2", "start", halo=True)
    panel.line([DOCUMENTS / 2] * 2, [-0.6, rows - 0.4], "ink", 1.4, dashed=True)
    panel.text(DOCUMENTS / 2, rows - 0.1, labels["half"], "ink", "middle")
    return chart.figure([panel], alt=labels["alt"], width=560, height=22 * rows + 100)


def _common_picture(seen) -> str:
    """The principles of the box with their counts, and whether the learner's code keeps them."""
    mine = seen.mine
    if mine is None or not mine.has_value or not isinstance(mine.value, list):
        return ""
    kept = set(str(p) for p in mine.value)
    rows = []
    for principle, count in NAMED.items():
        word = seen.word({"en": "kept", "nl": "over"}) if principle in kept else ""
        rows.append(f'<tr><td style="padding:0.15rem 0.8rem 0.15rem 0"><code>{principle}</code></td>'
                    f'<td style="padding:0.15rem 0.8rem;color:var(--ea-ink-2)">{count}/{DOCUMENTS}</td>'
                    f'<td style="padding:0.15rem 0;font-weight:600;color:var(--ea-ink)">{word}</td></tr>')
    title = seen.word({"en": "What your code keeps of the 11 principles", "nl": "Wat jouw code overhoudt van de 11 principes"})
    table = f'<table style="font-family:var(--ea-font-ui);font-size:0.9rem;border-collapse:collapse">{"".join(rows)}</table>'
    return yours.figure(title, table)


# The lesson is done when these are correct. "fairness" is the choice question in the lesson.
REQUIRED = ["common", "meanings", "fairness"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_NAMED_LINES = ('named = {"transparency": 73, "justice and fairness": 68, "non-maleficence": 60, "responsibility": 60,\n'
                '         "privacy": 47, "beneficence": 41, "freedom and autonomy": 34, "trust": 28,\n'
                '         "sustainability": 14, "dignity": 13, "solidarity": 6}\n')
_HEAD = _NAMED_LINES + "documents = 84\n\ncommon = []\nfor principle in named:\n"
_START = _HEAD + "    ...\ncommon\n"
_LINE = "    if named[principle] / documents > 0.5:\n        common.append(principle)\n"
_SOLUTION = _HEAD + _LINE + "common"


def _c(body: str) -> str:
    return _HEAD + body + "common"


EXERCISES = {
    "common": CodeBox(
        name="common",
        picture=_common_picture,
        start=_START,
        solution=_SOLUTION,
        keep=[_NAMED_LINES.rstrip("\n")],
        # The check also runs the box as if more documents had been found.
        others=[{"documents": Code("100")}],
        first=[_DOTS],
        mistakes=[
            Mistake("name-divided", error=lambda e: isinstance(e, TypeError) and "str" in str(e),
                    en="Your code divides `principle`, which is a name, by a number. The count of a principle is `named[principle]`.",
                    nl="Je code deelt `principle`, en dat is een naam, door een getal. Het aantal van een principe is `named[principle]`."),
            Mistake("count", wrong=_c("    if named[principle] > 0.5:\n        common.append(principle)\n"),
                    en="Your last line gives all 11 principles. Every count, even 6, is more than 0.5 on its own: keep a principle only when its share, the count divided by `documents`, is more than 0.5.",
                    nl="Je laatste regel geeft alle 11 principes. Elk aantal, zelfs 6, is op zich al meer dan 0,5: houd een principe alleen over als het aandeel, het aantal gedeeld door `documents`, meer dan 0,5 is."),
            Mistake("fewer", wrong=_c("    if named[principle] / documents < 0.5:\n        common.append(principle)\n"),
                    en="Your last line gives the principles named in fewer than half of the documents. Turn the comparison around.",
                    nl="Je laatste regel geeft de principes die in minder dan de helft van de documenten staan. Draai de vergelijking om."),
            Mistake("shares", matches=lambda got, text: isinstance(got, list) and len(got) > 0
                    and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in got),
                    en="Your last line gives numbers. Append the name of the principle, `principle`, not its count or its share.",
                    nl="Je laatste regel geeft getallen. Voeg de naam van het principe toe, `principle`, niet het aantal of het aandeel."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with two lines: an `if` that asks whether the share of this principle is more than half, and under it, append the principle to `common`.",
                      "nl": "Vervang de `...` door twee regels: een `if` die vraagt of het aandeel van dit principe meer dan de helft is, en daaronder het principe toevoegen aan `common`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries another number of documents by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf een ander aantal documenten."},
            "no_value": {"en": "The last line of the box shows nothing. Put `common` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `common` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does your loop append a principle only when its count divided by `documents` is more than 0.5?",
                      "nl": "Je laatste regel geeft `{got}`. Voegt je loop een principe alleen toe als het aantal gedeeld door `documents` meer dan 0,5 is?"},
            "other": {"en": "Right for 84 documents. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you divide by `documents` itself?",
                      "nl": "Goed voor 84 documenten. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Deel je door `documents` zelf?"},
            "other_error": {"en": "Right for 84 documents. But with {where}, your code stops with an error. Do you divide by `documents` itself?",
                            "nl": "Goed voor 84 documenten. Maar met {where} stopt je code met een fout. Deel je door `documents` zelf?"},
            "correct": {"en": "`{got}`: the five principles Jobin, Ienca and Vayena found in more than half of the documents. Beneficence, doing good, just misses: 41 of 84. The check also tried another number of documents.",
                        "nl": "`{got}`: de vijf principes die Jobin, Ienca en Vayena in meer dan de helft van de documenten vonden. Weldoen haalt het net niet: 41 van de 84. De check probeerde ook een ander aantal documenten."},
        },
    ),
    "meanings": Sort(
        name="meanings",
        groups={
            "explain": {"en": "Explain one decision", "nl": "Eén beslissing uitleggen"},
            "disclose": {"en": "Say that AI is used", "nl": "Zeggen dat er AI wordt gebruikt"},
            "open": {"en": "Open up the system itself", "nl": "Het systeem zelf openleggen"},
        },
        items={
            "loan": {"en": "**A refused loan.** Someone who is refused a loan can ask which facts weighed most.",
                     "nl": "**Een geweigerde lening.** Wie een lening geweigerd krijgt, kan vragen welke feiten het zwaarst wogen."},
            "data": {"en": "**The data.** The maker publishes which data the model learned from.",
                     "nl": "**De data.** De maker publiceert van welke data het model leerde."},
            "bot": {"en": "**A chat window.** It says at the start that you are talking to a bot, not a person.",
                    "nl": "**Een chatvenster.** Het zegt aan het begin dat je met een bot praat, niet met een mens."},
            "flag": {"en": "**A flagged message.** The system shows which words made it flag this message.",
                     "nl": "**Een gemarkeerd bericht.** Het systeem laat zien door welke woorden het dit bericht markeerde."},
            "limits": {"en": "**The limits.** The maker publishes where the model fails, and for whom it works less well.",
                       "nl": "**De grenzen.** De maker publiceert waar het model faalt, en voor wie het minder goed werkt."},
            "label": {"en": "**A label.** A picture made with AI carries a mark that says so.",
                      "nl": "**Een label.** Een plaatje dat met AI gemaakt is, draagt een merkteken dat dat zegt."},
        },
        solution={"loan": "explain", "flag": "explain", "bot": "disclose", "label": "disclose", "data": "open", "limits": "open"},
        nudge={
            "loan": {"en": "Is this about one person's case, or about the system as a whole?", "nl": "Gaat dit over het geval van één persoon, of over het systeem als geheel?"},
            "flag": {"en": "It is about this one message. What does the system tell?", "nl": "Het gaat over dit ene bericht. Wat vertelt het systeem?"},
            "bot": {"en": "Does it explain anything, or only say what you are dealing with?", "nl": "Legt het iets uit, of zegt het alleen waar je mee te maken hebt?"},
            "label": {"en": "The mark explains nothing about how the picture was made. What does it say?", "nl": "Het merkteken legt niets uit over hoe het plaatje gemaakt is. Wat zegt het wel?"},
            "data": {"en": "Is this about one decision, or about how the whole system was made?", "nl": "Gaat dit over één beslissing, of over hoe het hele systeem gemaakt is?"},
            "limits": {"en": "Where the model fails holds for all its decisions, not for one.", "nl": "Waar het model faalt, geldt voor al zijn beslissingen, niet voor één."},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the cases again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de gevallen opnieuw uit."},
            "empty": {"en": "Put each case with what it makes transparent.",
                      "nl": "Zet elk geval bij wat het transparant maakt."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The case *{item}* belongs with another meaning.",
                      "nl": "Het geval *{item}* hoort bij een andere betekenis."},
            "correct": {"en": "All {n} are right. Each case is called transparency in some guideline, and each asks for something else: an explanation for one person, a notice that AI is at work, or a look inside the system. A guideline that only says transparency leaves open which.",
                        "nl": "Alle {n} kloppen. Elk geval heet in een of andere richtlijn transparantie, en elk vraagt iets anders: uitleg voor één persoon, een melding dat er AI aan het werk is, of een blik in het systeem. Een richtlijn die alleen transparantie zegt, laat open welke."},
        },
    ),
}

_SORTED = {"loan": "explain", "flag": "explain", "bot": "disclose", "label": "disclose", "data": "open", "limits": "open"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("common", _START, "empty", None),
    ("common", _SOLUTION, "correct", None),
    ("common", _c("    if named[principle] > documents / 2:\n        common.append(principle)\n"), "correct", None),
    ("common", _c("    if named[principle] / documents >= 0.5:\n        common.append(principle)\n"), "correct", None),
    ("common", _c("    if 2 * named[principle] > documents:\n        common.append(principle)\n"), "correct", None),
    ("common", _c("    if named[principle] > 42:\n        common.append(principle)\n"), "almost", "other-value"),
    ("common", _c("    if named[principle] / 84 > 0.5:\n        common.append(principle)\n"), "almost", "other-value"),
    ("common", _c("    if named[principle] > 0.5:\n        common.append(principle)\n"), "wrong", "count"),
    ("common", _c("    common.append(principle)\n"), "wrong", "count"),
    ("common", _c("    if named[principle] / documents < 0.5:\n        common.append(principle)\n"), "wrong", "fewer"),
    ("common", _c("    if named[principle] / documents > 0.5:\n        common.append(named[principle] / documents)\n"), "wrong", "shares"),
    ("common", _c("    if named[principle] / documents > 0.5:\n        common.append(named[principle])\n"), "wrong", "shares"),
    ("common", _c("    if principle / documents > 0.5:\n        common.append(principle)\n"), "error", "name-divided"),
    ("common", _HEAD + "    ...\ncommon = []\ncommon", "wrong", "dots-left"),
    ("common", _c("    if named[principle] / documents > 0.7:\n        common.append(principle)\n"), "wrong", None),
    ("meanings", {}, "empty", None),
    ("meanings", _SORTED, "correct", None),
    ("meanings", {"loan": "explain", "bot": "disclose"}, "wrong", "unfinished"),
    ("meanings", {**_SORTED, "limits": "explain"}, "wrong", "wrong:limits"),
    ("meanings", {**_SORTED, "label": "open"}, "wrong", "wrong:label"),
    ("meanings", {**_SORTED, "flag": "disclose"}, "wrong", "wrong:flag"),
]
