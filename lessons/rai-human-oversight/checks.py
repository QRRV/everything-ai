# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-human-oversight: how many of a model's mistakes a reviewer put right, sorting what a
reviewer is missing (time, information or the power to say no), and the picture of the model's mistakes against the
reviewer's changes."""
import html
import re

from everything_ai.check import Code, CodeBox, Mistake, Sort
from everything_ai import yours

# Twenty loan applications, made up by the course: 1 is yes, 0 is no. `model` is what the model suggested, `person`
# what the reviewer decided after looking at the suggestion, `right` what an audit later found to be right.
MODEL = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0]
RIGHT = [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0]
PERSON = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0]


def counted(model: list, person: list, right: list) -> dict:
    """The model's mistakes, how many of them the reviewer put right, how often the reviewer changed the model's
    decision, and how often the reviewer agreed."""
    out = {"mistakes": 0, "caught": 0, "changed": 0, "agreed": 0}
    for m, p, r in zip(model, person, right):
        out["mistakes"] += m != r
        out["caught"] += m != r and p == r
        out["changed"] += p != m
        out["agreed"] += p == m
    return out


def picture(labels: dict) -> str:
    """The twenty cases in a row: on top a filled dot where the model was wrong, below a filled dot where the reviewer
    changed the model's decision. `labels` gives "title", "wrong", "changed", "xlabel" and "alt"."""
    from everything_ai import chart
    n = len(MODEL)
    panel = chart.Panel(x=(-10.5, n + 0.6), y=(-0.6, 1.6), title=labels["title"], xlabel=labels["xlabel"], yticks=[],
                        xticks=[1, 5, 10, 15, 20])
    panel.text(-10.4, 0.88, labels["wrong"], "ink-2", "start")
    panel.text(-10.4, -0.12, labels["changed"], "ink-2", "start")
    for i in range(n):
        wrong = MODEL[i] != RIGHT[i]
        panel.points([i + 1], [1], "loss" if wrong else "ink-2", r=6, hollow=not wrong)
        changed = PERSON[i] != MODEL[i]
        panel.points([i + 1], [0], "data" if changed else "ink-2", r=6, hollow=not changed)
    return chart.figure([panel], alt=labels["alt"], width=560, height=170)


def _caught_picture(seen) -> str:
    """The model's mistakes one by one: what the reviewer decided, and the learner's count beside the course's."""
    mine = seen.mine
    if mine is None or not mine.has_value:
        return ""
    right = counted(MODEL, PERSON, RIGHT)["caught"]
    rows = []
    for i in range(len(MODEL)):
        if MODEL[i] != RIGHT[i]:
            fixed = PERSON[i] == RIGHT[i]
            word = seen.word({"en": "put right", "nl": "rechtgezet"} if fixed else {"en": "let through", "nl": "doorgelaten"})
            rows.append(f'<tr><td style="padding:0.15rem 0.8rem 0.15rem 0">'
                        f'{seen.word({"en": "application", "nl": "aanvraag"})} {i + 1}</td>'
                        f'<td style="padding:0.15rem 0;font-weight:{600 if fixed else 400};color:var(--ea-ink)">{word}</td></tr>')
    got = html.escape(str(mine.value))
    count = seen.word({"en": f"Your <code>caught</code>: {got}", "nl": f"Jouw <code>caught</code>: {got}"})
    if mine.value != right:
        count += seen.word({"en": f" (the course: {right})", "nl": f" (de course: {right})"})
    title = seen.word({"en": "What the reviewer did with each mistake of the model", "nl": "Wat de beoordelaar met elke fout van het model deed"})
    table = f'<table style="font-family:var(--ea-font-ui);font-size:0.9rem;border-collapse:collapse">{"".join(rows)}</table>'
    return yours.figure(title, table + f'<p style="font-family:var(--ea-font-ui);font-size:0.9rem;margin:0.4rem 0 0">{count}</p>')


# The lesson is done when these are correct. "audit" is the choice question in the lesson.
REQUIRED = ["caught", "missing", "audit"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_LISTS = ("model = " + str(MODEL) + "\nperson = " + str(PERSON) + "\nright = " + str(RIGHT))
_HEAD = ("# 1 is yes, 0 is no\n" + _LISTS + "\n\nmistakes = 0\ncaught = 0\nfor i in range(len(model)):\n"
         "    if model[i] != right[i]:\n        mistakes = mistakes + 1\n")
_START = _HEAD + "        ...\ncaught\n"
_LINES = "        if person[i] == right[i]:\n            caught = caught + 1\n"
_SOLUTION = _HEAD + _LINES + "caught"


def _c(body: str) -> str:
    return _HEAD + body + "caught"


EXERCISES = {
    "caught": CodeBox(
        name="caught",
        picture=_caught_picture,
        start=_START,
        solution=_SOLUTION,
        keep=["model = " + str(MODEL), "right = " + str(RIGHT)],
        # The check also runs the box with another reviewer.
        others=[{"person": Code("[1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0]")}],
        first=[_DOTS],
        mistakes=[
            Mistake("missed", wrong=_c("        if person[i] == model[i]:\n            caught = caught + 1\n"),
                    en="Your last line gives `{got}`: those are the mistakes the reviewer let through, by agreeing with the model. Count the ones where the reviewer's decision is the right one.",
                    nl="Je laatste regel geeft `{got}`: dat zijn de fouten die de beoordelaar doorliet, door het met het model eens te zijn. Tel de fouten waarbij de beslissing van de beoordelaar de juiste is."),
            Mistake("changed", wrong=_HEAD + "    if person[i] != model[i]:\n        caught = caught + 1\ncaught",
                    en="Your last line gives `{got}`: how often the reviewer changed the model's decision. One of those changes made a right decision wrong. Count only inside the `if` for the model's mistakes.",
                    nl="Je laatste regel geeft `{got}`: hoe vaak de beoordelaar de beslissing van het model veranderde. Een van die veranderingen maakte een goede beslissing fout. Tel alleen binnen de `if` voor de fouten van het model."),
            Mistake("all-right", wrong=_HEAD + "    if person[i] == right[i]:\n        caught = caught + 1\ncaught",
                    en="Your last line gives `{got}`: every application where the reviewer's decision is right, also where the model was already right. Put your `if` inside the one for the model's mistakes.",
                    nl="Je laatste regel geeft `{got}`: elke aanvraag waarbij de beslissing van de beoordelaar goed is, ook waar het model al goed zat. Zet je `if` binnen die voor de fouten van het model."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with two lines: an `if` about the reviewer's decision, and under it a line that adds 1 to `caught`.",
                      "nl": "Vervang de `...` door twee regels: een `if` over de beslissing van de beoordelaar, en daaronder een regel die 1 optelt bij `caught`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries another reviewer by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf een andere beoordelaar."},
            "no_value": {"en": "The last line of the box shows nothing. Put `caught` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `caught` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Among the model's mistakes, does your code count the ones where the reviewer's decision is the right one?",
                      "nl": "Je laatste regel geeft `{got}`. Telt je code, onder de fouten van het model, de fouten waarbij de beslissing van de beoordelaar de juiste is?"},
            "other": {"en": "Right for this reviewer. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you count from the lists themselves?",
                      "nl": "Goed voor deze beoordelaar. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Tel je vanuit de lijsten zelf?"},
            "other_error": {"en": "Right for this reviewer. But with {where}, your code stops with an error. Do you count from the lists themselves?",
                            "nl": "Goed voor deze beoordelaar. Maar met {where} stopt je code met een fout. Tel je vanuit de lijsten zelf?"},
            "correct": {"en": "`{got}`: the model made 4 mistakes, and the reviewer put right only 1 of them, while agreeing with the model 18 times out of 20. The one other change made a right decision wrong. The check also tried another reviewer.",
                        "nl": "`{got}`: het model maakte 4 fouten, en de beoordelaar zette er maar 1 recht, terwijl die het 18 van de 20 keer met het model eens was. De enige andere verandering maakte een goede beslissing fout. De check probeerde ook een andere beoordelaar."},
        },
    ),
    "missing": Sort(
        name="missing",
        groups={
            "time": {"en": "Not enough time", "nl": "Te weinig tijd"},
            "information": {"en": "Not enough information", "nl": "Te weinig informatie"},
            "power": {"en": "No power to say no", "nl": "Geen macht om nee te zeggen"},
        },
        items={
            "minute": {"en": "**A minute each.** A reviewer must get through 400 cases a day.",
                       "nl": "**Een minuut per stuk.** Een beoordelaar moet 400 gevallen per dag afhandelen."},
            "alarms": {"en": "**Every few seconds.** On the night shift, a new alarm comes in before the last one is read.",
                       "nl": "**Om de paar seconden.** In de nachtdienst komt er een nieuw alarm binnen voordat het vorige gelezen is."},
            "score": {"en": "**Only a score.** The reviewer sees the model's score, but not the application itself.",
                      "nl": "**Alleen een score.** De beoordelaar ziet de score van het model, maar niet de aanvraag zelf."},
            "reason": {"en": "**No reason.** The system gives no reason, and the data it used is hidden from the reviewer.",
                       "nl": "**Geen reden.** Het systeem geeft geen reden, en de data die het gebruikte, is voor de beoordelaar verborgen."},
            "bonus": {"en": "**A bonus for agreeing.** Reviewers are rated on how often they match the model.",
                      "nl": "**Een bonus voor instemmen.** Beoordelaars worden beoordeeld op hoe vaak ze met het model overeenkomen."},
            "sent": {"en": "**Already sent.** The letter with the model's decision goes out before the review.",
                     "nl": "**Al verstuurd.** De brief met de beslissing van het model gaat de deur uit vóór de controle."},
        },
        solution={"minute": "time", "alarms": "time", "score": "information", "reason": "information", "bonus": "power", "sent": "power"},
        nudge={
            "minute": {"en": "How long can a reviewer look at one case here?", "nl": "Hoe lang kan een beoordelaar hier naar één geval kijken?"},
            "alarms": {"en": "Is there a moment to think before the next one comes?", "nl": "Is er een moment om na te denken voordat het volgende komt?"},
            "score": {"en": "Could the reviewer find a mistake without seeing the case?", "nl": "Kan de beoordelaar een fout vinden zonder het geval te zien?"},
            "reason": {"en": "What does the reviewer know about why the system decided this?", "nl": "Wat weet de beoordelaar over waarom het systeem dit besliste?"},
            "bonus": {"en": "What happens to a reviewer who disagrees?", "nl": "Wat gebeurt er met een beoordelaar die het oneens is?"},
            "sent": {"en": "Can the reviewer still change anything?", "nl": "Kan de beoordelaar nog iets veranderen?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the cases again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de gevallen opnieuw uit."},
            "empty": {"en": "Put each case with what the reviewer is missing.",
                      "nl": "Zet elk geval bij wat de beoordelaar mist."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "In the case *{item}*, the reviewer is missing something else.",
                      "nl": "In het geval *{item}* mist de beoordelaar iets anders."},
            "correct": {"en": "All {n} are right. A reviewer without time agrees to keep up, one without information cannot see a mistake, and one without power gains nothing by finding it. Oversight needs all three.",
                        "nl": "Alle {n} kloppen. Een beoordelaar zonder tijd stemt in om bij te blijven, een beoordelaar zonder informatie kan een fout niet zien, en een beoordelaar zonder macht heeft er niets aan om hem te vinden. Toezicht heeft alle drie nodig."},
        },
    ),
}

_SORTED = {"minute": "time", "alarms": "time", "score": "information", "reason": "information", "bonus": "power", "sent": "power"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("caught", _START, "empty", None),
    ("caught", _SOLUTION, "correct", None),
    ("caught", _c("        if right[i] == person[i]:\n            caught = caught + 1\n"), "correct", None),
    ("caught", _c("        if person[i] == model[i]:\n            caught = caught + 1\n"), "wrong", "missed"),
    ("caught", _HEAD + "    if person[i] != model[i]:\n        caught = caught + 1\ncaught", "wrong", "changed"),
    ("caught", _HEAD + "    if person[i] == right[i]:\n        caught = caught + 1\ncaught", "wrong", "all-right"),
    ("caught", _c("        caught = caught + 1\n"), "wrong", None),
    ("caught", _c("        pass\ncaught = 1\n"), "almost", "other-value"),
    ("caught", _SOLUTION.replace("right = " + str(RIGHT), "right = " + str(MODEL)), "wrong", "changed-line"),
    ("caught", _HEAD + "        ...\ncaught = 1\ncaught", "wrong", "dots-left"),
    ("missing", {}, "empty", None),
    ("missing", _SORTED, "correct", None),
    ("missing", {"minute": "time", "score": "information"}, "wrong", "unfinished"),
    ("missing", {**_SORTED, "bonus": "time"}, "wrong", "wrong:bonus"),
    ("missing", {**_SORTED, "reason": "power"}, "wrong", "wrong:reason"),
    ("missing", {**_SORTED, "sent": "information"}, "wrong", "wrong:sent"),
]
