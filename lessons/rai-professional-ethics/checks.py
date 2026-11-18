# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-professional-ethics: the cost of each setting of a check for hateful messages when a
false alarm counts heavier than a miss, sorting situations under principles of the ACM code, and the pictures of the
mistakes and the costs."""
import re

from everything_ai.check import CodeBox, Mistake, Sort
from everything_ai import yours

# A check that flags hateful messages on a forum, tested at three settings on 1,000 messages that people checked by
# hand, made up by the course: how many hateful messages each setting lets through, and how many harmless ones it flags.
MISSES = {"strict": 4, "middle": 10, "loose": 25}
ALARMS = {"strict": 60, "middle": 25, "loose": 8}
# The worked example's costs (a miss counts 10 times as heavy) and the exercise's (a false alarm counts 3 times).
EXAMPLE_COSTS = (10, 1)
MISS_COST = 1
ALARM_COST = 3


def cost(misses: dict, alarms: dict, miss_cost: float, alarm_cost: float) -> dict:
    """For each setting, the cost of its misses plus the cost of its false alarms."""
    return {s: miss_cost * misses[s] + alarm_cost * alarms[s] for s in misses}


_UI = "font-family:var(--ea-font-ui)"


def _key(items: list[tuple[str, str]]) -> str:
    """A line under a chart that says what each kind of mark means: (svg, words) pairs."""
    spans = "".join(f"<span>{svg} {words}</span>" for svg, words in items)
    return (f'<div style="{_UI};font-size:0.82rem;color:var(--ea-ink-2);display:flex;gap:1.2rem;flex-wrap:wrap;'
            f'margin-top:0.3rem">{spans}</div>')


def _square(solid: bool) -> str:
    fill = "fill:var(--ea-loss)" if solid else "fill:var(--ea-loss);fill-opacity:0.18"
    return (f'<svg width="14" height="12" aria-hidden="true"><rect x="1" y="1" width="12" height="10" style="{fill};'
            f'stroke:var(--ea-loss);stroke-width:1.2"/></svg>')


def picture(labels: dict) -> str:
    """Per setting, the hateful messages it lets through (a filled bar) and the harmless messages it flags (an open bar),
    each with its number. `labels` gives "title", "y", "misses", "alarms" and "alt"."""
    from everything_ai import chart
    settings = list(MISSES)
    panel = chart.Panel(x=(-0.6, len(settings) - 0.4), y=(0, 70), title=labels["title"], ylabel=labels["y"],
                        yticks=[0, 20, 40, 60], xticks=list(range(len(settings))), xtick_labels=settings)
    for i, s in enumerate(settings):
        panel.rect(i - 0.34, 0, i - 0.03, MISSES[s], role="loss", solid=True)
        panel.rect(i + 0.03, 0, i + 0.34, ALARMS[s], role="loss")
        panel.text(i - 0.185, MISSES[s] + 2, str(MISSES[s]), "ink")
        panel.text(i + 0.185, ALARMS[s] + 2, str(ALARMS[s]), "ink")
    key = _key([(_square(True), labels["misses"]), (_square(False), labels["alarms"])])
    return chart.figure([panel], alt=labels["alt"], width=480, height=260) + key


def _number(v) -> float | None:
    """A number the picture can draw, or None."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _cost_picture(seen) -> str:
    """The learner's cost per setting as bars, a dashed line at the right cost, and the lowest of the learner's bars
    marked."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value or not isinstance(mine.value, dict):
        return ""
    got, want = mine.value, cost(MISSES, ALARMS, MISS_COST, ALARM_COST)
    settings = list(MISSES)
    top = 240
    panel = chart.Panel(x=(-0.6, len(settings) - 0.4), y=(0, top), ylabel=seen.word({"en": "cost", "nl": "kosten"}),
                        yticks=[0, 50, 100, 150, 200], xticks=list(range(len(settings))), xtick_labels=settings)
    numbers = {s: _number(got.get(s)) for s in settings}
    for i, s in enumerate(settings):
        v = numbers[s]
        if v is not None and v > 0:
            panel.rect(i - 0.3, 0, i + 0.3, min(v, top), role="loss")
        panel.line([i - 0.38, i + 0.38], [want[s]] * 2, role="ink", width=2, dashed=True)
    drawn = [s for s in settings if numbers[s] is not None]
    if len(drawn) == len(settings):
        low = min(drawn, key=lambda s: numbers[s])
        y = min(max(numbers[low], want[low]), top - 30) + 12
        panel.text(settings.index(low), y, seen.word({"en": "lowest", "nl": "laagste"}), "ink", halo=True)
    title = seen.word({"en": "Your cost per setting", "nl": "Jouw kosten per instelling"})
    words = {"en": ("your code", "right"), "nl": ("jouw code", "goed")}[seen.lang]
    dash = ('<svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" style="stroke:var(--ea-ink);'
            'stroke-width:2;stroke-dasharray:5 5"/></svg>')
    return yours.figure(title, chart.figure([panel], alt=title, width=480, height=250)
                        + _key([(_square(False), words[0]), (dash, words[1])]))


# The lesson is done when these are correct. "values" is the choice question in the lesson.
REQUIRED = ["cost", "principles", "values"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _line(name: str, values) -> str:
    return f"{name} = {values!r}".replace("'", '"')


_DATA = [_line("misses", MISSES), _line("alarms", ALARMS), _line("miss_cost", MISS_COST), _line("alarm_cost", ALARM_COST)]
_HEAD = ("# per 1,000 messages: hateful ones let through, and harmless ones flagged\n" + "\n".join(_DATA[:2])
         + "\n# what one mistake of each kind costs\n" + "\n".join(_DATA[2:]) + "\n\ncost = {}\nfor setting in misses:\n")
_START = _HEAD + "    ...\ncost\n"
_LINE = "    cost[setting] = miss_cost * misses[setting] + alarm_cost * alarms[setting]\n"
_SOLUTION = _HEAD + _LINE + "cost"


def _c(body: str) -> str:
    return _HEAD + body + "cost"


EXERCISES = {
    "cost": CodeBox(
        name="cost",
        picture=_cost_picture,
        start=_START,
        solution=_SOLUTION,
        keep=_DATA[:2],
        # The check also runs the box with other costs.
        others=[{"miss_cost": 2, "alarm_cost": 1}],
        first=[_DOTS],
        mistakes=[
            Mistake("dicts-multiplied", error=lambda e: isinstance(e, TypeError) and "dict" in str(e),
                    en="Your code multiplies a whole dictionary. Multiply the count for this one setting: `misses[setting]` and `alarms[setting]`.",
                    nl="Je code vermenigvuldigt een hele dictionary. Vermenigvuldig het aantal voor deze ene instelling: `misses[setting]` en `alarms[setting]`."),
            Mistake("swapped", wrong=_c("    cost[setting] = miss_cost * alarms[setting] + alarm_cost * misses[setting]\n"),
                    en="Your last line gives `{got}`: the costs are swapped. A miss costs `miss_cost`, a false alarm `alarm_cost`.",
                    nl="Je laatste regel geeft `{got}`: de kosten zijn verwisseld. Een gemist bericht kost `miss_cost`, een vals alarm `alarm_cost`."),
            Mistake("no-costs", wrong=_c("    cost[setting] = misses[setting] + alarms[setting]\n"),
                    en="Your last line gives `{got}`: all mistakes, each counted once. Multiply each count by what one such mistake costs.",
                    nl="Je laatste regel geeft `{got}`: alle fouten, elk één keer geteld. Vermenigvuldig elk aantal met wat één zo'n fout kost."),
            Mistake("misses-only", wrong=_c("    cost[setting] = miss_cost * misses[setting]\n"),
                    en="Your last line gives `{got}`: only the cost of the misses. Add the cost of the false alarms.",
                    nl="Je laatste regel geeft `{got}`: alleen de kosten van de gemiste berichten. Tel de kosten van de valse alarmen erbij op."),
            Mistake("alarms-only", wrong=_c("    cost[setting] = alarm_cost * alarms[setting]\n"),
                    en="Your last line gives `{got}`: only the cost of the false alarms. Add the cost of the misses.",
                    nl="Je laatste regel geeft `{got}`: alleen de kosten van de valse alarmen. Tel de kosten van de gemiste berichten erbij op."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with one line that puts under `cost[setting]` the cost of this setting's misses plus the cost of its false alarms.",
                      "nl": "Vervang de `...` door één regel die onder `cost[setting]` de kosten van de gemiste berichten van deze instelling zet, plus de kosten van zijn valse alarmen."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other costs by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere kosten."},
            "no_value": {"en": "The last line of the box shows nothing. Put `cost` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `cost` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does your loop put under each setting its misses times `miss_cost`, plus its false alarms times `alarm_cost`?",
                      "nl": "Je laatste regel geeft `{got}`. Zet je loop onder elke instelling de gemiste berichten keer `miss_cost`, plus de valse alarmen keer `alarm_cost`?"},
            "other": {"en": "Right for these costs. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you use `miss_cost` and `alarm_cost` themselves?",
                      "nl": "Goed voor deze kosten. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruik je `miss_cost` en `alarm_cost` zelf?"},
            "other_error": {"en": "Right for these costs. But with {where}, your code stops with an error. Do you use `miss_cost` and `alarm_cost` themselves?",
                            "nl": "Goed voor deze kosten. Maar met {where} stopt je code met een fout. Gebruik je `miss_cost` en `alarm_cost` zelf?"},
            "correct": {"en": "`{got}`: with these costs, `loose` is cheapest, while with the worked example's costs `strict` was. The counts stayed the same; only the values changed, so which setting is best is a choice about whose harm counts more. The check also tried other costs.",
                        "nl": "`{got}`: met deze kosten is `loose` het goedkoopst, terwijl met de kosten van het uitgewerkte voorbeeld `strict` dat was. De aantallen bleven gelijk; alleen de waarden veranderden, dus welke instelling de beste is, is een keuze over wiens schade zwaarder telt. De check probeerde ook andere kosten."},
        },
    ),
    "principles": Sort(
        name="principles",
        groups={
            "harm": {"en": "1.2 Avoid harm", "nl": "1.2 Vermijd schade"},
            "fair": {"en": "1.4 Be fair and take action not to discriminate", "nl": "1.4 Wees rechtvaardig en ga discriminatie actief tegen"},
            "privacy": {"en": "1.6 Respect privacy", "nl": "1.6 Respecteer privacy"},
        },
        items={
            "bridge": {"en": "**The bridge.** A model that looks for cracks in bridges misses some in tests, and the team is asked to ship it anyway.",
                       "nl": "**De brug.** Een model dat scheuren in bruggen zoekt, mist er in tests een paar, en het team moet het toch opleveren."},
            "diet": {"en": "**The diet.** A chatbot for teenagers sometimes gives dangerous advice about losing weight.",
                     "nl": "**Het dieet.** Een chatbot voor tieners geeft soms gevaarlijk advies over afvallen."},
            "accent": {"en": "**The accent.** A speech system understands some accents much worse than others, and nobody has measured by how much.",
                       "nl": "**Het accent.** Een spraaksysteem verstaat sommige accenten veel slechter dan andere, en niemand heeft gemeten hoeveel."},
            "loan": {"en": "**The loan.** A loan model gives people from some neighborhoods lower scores at the same income.",
                     "nl": "**De lening.** Een model voor leningen geeft mensen uit sommige wijken lagere scores bij hetzelfde inkomen."},
            "location": {"en": "**The location.** An app keeps every user's full location history, though it needs only the city.",
                         "nl": "**De locatie.** Een app bewaart van elke gebruiker alle plekken waar die was, terwijl hij alleen de stad nodig heeft."},
            "messages": {"en": "**The messages.** A team wants to read users' private messages to find examples for its model.",
                         "nl": "**De berichten.** Een team wil privéberichten van gebruikers lezen om voorbeelden voor zijn model te vinden."},
        },
        solution={"bridge": "harm", "diet": "harm", "accent": "fair", "loan": "fair", "location": "privacy", "messages": "privacy"},
        nudge={
            "bridge": {"en": "What can happen to people when a crack is missed?", "nl": "Wat kan er met mensen gebeuren als een scheur gemist wordt?"},
            "diet": {"en": "Is anyone treated unequally here, or is someone put at risk?", "nl": "Wordt hier iemand ongelijk behandeld, of loopt iemand gevaar?"},
            "accent": {"en": "Does the system work equally well for everyone?", "nl": "Werkt het systeem voor iedereen even goed?"},
            "loan": {"en": "Two people with the same income get different scores. Why?", "nl": "Twee mensen met hetzelfde inkomen krijgen verschillende scores. Waarom?"},
            "location": {"en": "Is the problem what the app does with the data, or how much it keeps?", "nl": "Is het probleem wat de app met de data doet, of hoeveel hij bewaart?"},
            "messages": {"en": "Whose messages are they, and did they expect anyone to read them?", "nl": "Van wie zijn de berichten, en verwachtten ze dat iemand ze zou lezen?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the situations again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de situaties opnieuw uit."},
            "empty": {"en": "Put each situation under the principle it is mainly about.",
                      "nl": "Zet elke situatie bij het principe waar die vooral over gaat."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The situation *{item}* is mainly about another principle.",
                      "nl": "De situatie *{item}* gaat vooral over een ander principe."},
            "correct": {"en": "All {n} are right. Real cases often touch more than one principle at once, but each of these has one that stands out, and the code gives an engineer words to name it and to say no.",
                        "nl": "Alle {n} kloppen. Echte gevallen raken vaak meer dan één principe tegelijk, maar bij elk van deze springt er één uit, en de code geeft een engineer houvast om het te benoemen en nee te zeggen."},
        },
    ),
}

_SORTED = {"bridge": "harm", "diet": "harm", "accent": "fair", "loan": "fair", "location": "privacy", "messages": "privacy"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("cost", _START, "empty", None),
    ("cost", _SOLUTION, "correct", None),
    ("cost", _c("    cost[setting] = alarm_cost * alarms[setting] + miss_cost * misses[setting]\n"), "correct", None),
    ("cost", _c("    cost[setting] = miss_cost * alarms[setting] + alarm_cost * misses[setting]\n"), "wrong", "swapped"),
    ("cost", _c("    cost[setting] = misses[setting] + alarms[setting]\n"), "wrong", "no-costs"),
    ("cost", _c("    cost[setting] = miss_cost * misses[setting]\n"), "wrong", "misses-only"),
    ("cost", _c("    cost[setting] = alarm_cost * alarms[setting]\n"), "wrong", "alarms-only"),
    ("cost", _c("    cost[setting] = miss_cost * misses + alarm_cost * alarms\n"), "error", "dicts-multiplied"),
    ("cost", _c("    cost[setting] = 2 * misses[setting] + alarms[setting]\n"), "wrong", None),
    ("cost", _SOLUTION.replace(_DATA[0], _DATA[0].replace('"loose": 25', '"loose": 5')), "wrong", "changed-line"),
    ("cost", _HEAD + "    pass\ncost = {'strict': 184, 'middle': 85, 'loose': 49}\ncost", "almost", "other-value"),
    ("cost", _HEAD + "    ...\ncost = {}\ncost", "wrong", "dots-left"),
    ("principles", {}, "empty", None),
    ("principles", _SORTED, "correct", None),
    ("principles", {"bridge": "harm", "loan": "fair"}, "wrong", "unfinished"),
    ("principles", {**_SORTED, "diet": "fair"}, "wrong", "wrong:diet"),
    ("principles", {**_SORTED, "accent": "harm"}, "wrong", "wrong:accent"),
    ("principles", {**_SORTED, "messages": "harm"}, "wrong", "wrong:messages"),
]
