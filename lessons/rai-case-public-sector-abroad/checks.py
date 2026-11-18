# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-case-public-sector-abroad: the income that averaging over fortnights makes up for a person
made up by the course, sorting four cases from abroad by the way they went wrong, and the pictures of what averaging
sees in each fortnight."""
import html
import re

from everything_ai.check import Code, CodeBox, Mistake, Sort
from everything_ai import yours

# Made up by the course: what one person earned in each of the 26 fortnights of a year. In the exercise, 1,000 dollars
# in each of the first 13 fortnights and nothing after; in the worked example, a casual worker who earned 400 and 600
# dollars in turns; in the picture of the idea, 1,300 dollars in each of the first 10 fortnights and nothing after.
EARNED = [1000] * 13 + [0] * 13
EXAMPLE = [400, 600] * 13
IDEA = [1300] * 10 + [0] * 16
# The other person the check tries: 300 and 700 dollars in turns.
OTHER = [300, 700] * 13


def averaged(earned: list) -> float:
    """The income of the year spread evenly over its fortnights."""
    return sum(earned) / len(earned)


def invented(earned: list) -> float:
    """The income that averaging puts in fortnights where it was not earned: over the fortnights in which the average is
    higher than what was earned, by how much."""
    a = averaged(earned)
    return sum(a - x for x in earned if a > x)


_UI = "font-family:var(--ea-font-ui)"


def _number(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _fortnights(earned: list, average, words: dict, want=None) -> str:
    """Bars of what was earned in each fortnight, a line at `average`, the space between a bar and the line shaded where
    the line is higher, and a dashed line at `want` when it is given. `words` gives "axis", "fortnight" and "alt"."""
    from everything_ai import chart
    top = 1400
    n = len(earned)
    panel = chart.Panel(x=(-0.7, n - 0.3), y=(0, top), xlabel=words["fortnight"], ylabel=words["axis"],
                        yticks=[0, 500, 1000], xticks=[0, 5, 10, 15, 20, 25], xtick_labels=["1", "6", "11", "16", "21", "26"])
    for i, x in enumerate(earned):
        if x > 0:
            panel.rect(i - 0.38, 0, i + 0.38, min(x, top), role="data", solid=True)
        if average is not None and 0 < average and average > x:
            panel.rect(i - 0.38, x, i + 0.38, min(average, top), role="loss", solid=False)
    if want is not None:
        panel.line([-0.7, n - 0.3], [want] * 2, role="ink", width=2, dashed=True)
    if average is not None and 0 < average < top:
        panel.line([-0.7, n - 0.3], [average] * 2, role="pred", width=2.2)
    return chart.figure([panel], alt=words["alt"], width=460, height=250)


def _key(words: tuple, dashed: bool) -> str:
    bar = ('<svg width="14" height="12" aria-hidden="true"><rect x="1" y="1" width="12" height="10" style="fill:var(--ea-data);'
           'stroke:var(--ea-data);stroke-width:1.2"/></svg>')
    line = ('<svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" style="stroke:var(--ea-pred);'
            'stroke-width:2.2"/></svg>')
    shade = ('<svg width="14" height="12" aria-hidden="true"><rect x="1" y="1" width="12" height="10" style="fill:var(--ea-loss);'
             'fill-opacity:0.18;stroke:var(--ea-loss);stroke-width:1.2"/></svg>')
    dash = ('<svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" style="stroke:var(--ea-ink);'
            'stroke-width:2;stroke-dasharray:5 5"/></svg>')
    spans = [f"<span>{bar} {html.escape(words[0])}</span>", f"<span>{line} {html.escape(words[1])}</span>",
             f"<span>{shade} {html.escape(words[2])}</span>"]
    if dashed:
        spans.append(f"<span>{dash} {html.escape(words[3])}</span>")
    return (f'<div style="{_UI};font-size:0.82rem;color:var(--ea-ink-2);display:flex;gap:1.2rem;flex-wrap:wrap;margin-top:0.3rem">'
            + "".join(spans) + "</div>")


def picture(labels: dict) -> str:
    """What averaging sees for the person of the idea: what was earned in each fortnight, the averaged income as a
    line, and the income averaging made up shaded. `labels` gives "title", "axis", "fortnight", "earned", "averaged",
    "invented" and "alt"."""
    body = _fortnights(IDEA, averaged(IDEA), labels)
    return yours.figure(labels["title"], body + _key((labels["earned"], labels["averaged"], labels["invented"]), False))


def _invented_picture(seen) -> str:
    """The learner's `average` as a line over what the person of the exercise earned, the income it makes up shaded,
    a dashed line at the right average, and above it their `invented`."""
    mine = seen.mine
    if mine is None:
        return ""
    average = _number(mine.names.get("average"))
    words = {"en": {"axis": "dollars", "fortnight": "fortnight", "alt": "Your averaged income per fortnight"},
             "nl": {"axis": "dollar", "fortnight": "periode van twee weken", "alt": "Jouw gemiddelde inkomen per periode"}}[seen.lang]
    key = {"en": ("earned", "your average", "made up by your average", "right average"),
           "nl": ("verdiend", "jouw gemiddelde", "verzonnen door jouw gemiddelde", "goed gemiddelde")}[seen.lang]
    got = _number(mine.value) if mine.has_value else None
    said = ""
    if got is not None:
        said = (f'<div style="{_UI};font-size:0.9rem;color:var(--ea-ink);margin-bottom:0.2rem">'
                f'<code>invented = {html.escape(f"{got:.1f}")}</code></div>')
    title = seen.word({"en": "What your average sees in each fortnight", "nl": "Wat jouw gemiddelde ziet in elke periode"})
    return yours.figure(title, said + _fortnights(EARNED, average, words, want=averaged(EARNED)) + _key(key, True))


# The lesson is done when these are correct. "small" is the choice question in the lesson.
REQUIRED = ["averaging", "failures", "small"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _list(name: str, values: list) -> str:
    """A list of 26 numbers over two lines, 13 on each."""
    pad = " " * (len(name) + 4)
    return f"{name} = [" + ", ".join(map(str, values[:13])) + ",\n" + pad + ", ".join(map(str, values[13:])) + "]"


_EARNED_LINES = _list("earned", EARNED)
_HEAD = "# what one person earned in each fortnight of a year, made up by the course\n" + _EARNED_LINES + "\n\n"
_START = _HEAD + "...\ninvented\n"
_BODY = ("total = 0\nfor amount in earned:\n    total = total + amount\naverage = total / len(earned)\n\n"
         "invented = 0\nfor amount in earned:\n    if average > amount:\n        invented = invented + (average - amount)\n")
_SOLUTION = _HEAD + _BODY + "invented"


def _s(body: str) -> str:
    return _HEAD + body + "invented"


def _swap(old: str, new: str) -> str:
    assert _BODY.count(old) == 1
    return _s(_BODY.replace(old, new))


EXERCISES = {
    "averaging": CodeBox(
        name="averaging",
        picture=_invented_picture,
        start=_START,
        solution=_SOLUTION,
        keep=[_EARNED_LINES],
        # The check also runs the box for someone who earned 300 and 700 dollars in turns.
        others=[{"earned": Code("[300, 700] * 13")}],
        first=[_DOTS],
        digits=2,
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_earned`. In the box the names have none: `earned`, `total`, `amount`, `average`, `invented`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_earned`. In het vak hebben de namen dat niet: `earned`, `total`, `amount`, `average`, `invented`."),
            Mistake("total", matches=lambda got, text: _number(got) is not None and _number(got) > sum(EARNED) * 2 + 0.5,
                    en="Your last line gives `{got}`: more than twice what this person earned in the whole year. In the second loop, compare each `amount` with `average`, not with `total`.",
                    nl="Je laatste regel geeft `{got}`: meer dan twee keer wat deze persoon in het hele jaar verdiende. Vergelijk in de tweede loop elke `amount` met `average`, niet met `total`."),
            Mistake("no-if", matches=lambda got, text: _number(got) == 0,
                    en="Your last line gives `{got}`. Does your second loop add only in the fortnights in which `average > amount`? Without the `if`, the fortnights in which averaging puts in too little cancel out the ones in which it puts in too much.",
                    nl="Je laatste regel geeft `{got}`. Telt je tweede loop alleen op in de periodes waarin `average > amount`? Zonder de `if` heffen de periodes waarin het middelen te weinig ziet de periodes op waarin het te veel ziet."),
            Mistake("counted", wrong=_swap("invented = invented + (average - amount)", "invented = invented + 1"),
                    en="Your last line gives `{got}`: the number of fortnights in which the average is too high. Add by how much it is too high in each of them.",
                    nl="Je laatste regel geeft `{got}`: het aantal periodes waarin het gemiddelde te hoog is. Tel op hoeveel het in elk daarvan te hoog is."),
            Mistake("both-ways", matches=lambda got, text: _number(got) is not None and abs(_number(got) - 2 * invented(EARNED)) < 0.5
                    and re.search(r"amount\s*-\s*average|abs\(", text) is not None,
                    en="Your last line gives `{got}`: twice too much. You also count the fortnights in which `average` is lower than `amount`; count only those in which it is higher.",
                    nl="Je laatste regel geeft `{got}`: twee keer te veel. Je telt ook de periodes waarin `average` lager is dan `amount`; tel alleen die waarin het hoger is."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the lines of the worked example: add up `earned` in `total`, divide it by the number of fortnights for `average`, and then add up in `invented` what averaging puts in the fortnights where it sees more than was earned.",
                      "nl": "Vervang de `...` door de regels van het uitgewerkte voorbeeld: tel `earned` op in `total`, deel dat door het aantal periodes voor `average`, en tel daarna in `invented` op wat het middelen erbij zet in de periodes waarin het meer ziet dan er verdiend is."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the lines of `earned` exactly as they were: put them back. The check tries another person by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `earned` nodig zoals ze waren: zet ze terug. De check probeert zelf een andere persoon."},
            "no_value": {"en": "The last line of the box shows nothing. Put `invented` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `invented` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Is `average` the total of `earned` divided by the number of fortnights, and does `invented` add, in each fortnight in which `average` is higher than `amount`, the difference?",
                      "nl": "Je laatste regel geeft `{got}`. Is `average` het totaal van `earned` gedeeld door het aantal periodes, en telt `invented` in elke periode waarin `average` hoger is dan `amount` het verschil op?"},
            "other": {"en": "Right for this person. But for someone who earned 300 and 700 dollars in turns, your last line gives `{got}`, and it should give `{want}`. Do you add the difference between `average` and `amount`, and only where `average` is higher?",
                      "nl": "Goed voor deze persoon. Maar voor iemand die om en om 300 en 700 dollar verdiende, geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Tel je het verschil tussen `average` en `amount` op, en alleen waar `average` hoger is?"},
            "other_error": {"en": "Right for this person. But for someone who earned 300 and 700 dollars in turns, your code stops with an error. Do you use `earned` itself?",
                            "nl": "Goed voor deze persoon. Maar voor iemand die om en om 300 en 700 dollar verdiende, stopt je code met een fout. Gebruik je `earned` zelf?"},
            "correct": {"en": "`{got}`: averaging made up 6,500 dollars of income, all of it in the 13 fortnights in which this person earned nothing. Where someone also got a benefit in such a fortnight, Robodebt counted that income against the benefit, and so raised a debt. The check also tried someone who earned 300 and 700 dollars in turns: then averaging makes up 2,600 dollars. Only with the same income in every fortnight does it make up nothing.",
                        "nl": "`{got}`: het middelen verzon 6.500 dollar aan inkomen, allemaal in de 13 periodes waarin deze persoon niets verdiende. Kreeg iemand in zo'n periode ook een uitkering, dan telde Robodebt dat inkomen mee tegen de uitkering, en ontstond er dus een schuld. De check probeerde ook iemand die om en om 300 en 700 dollar verdiende: dan verzint het middelen 2.600 dollar. Alleen bij hetzelfde inkomen in elke periode verzint het niets."},
        },
    ),
    "failures": Sort(
        name="failures",
        groups={
            "calc": {"en": "A calculation that was wrong for many people", "nl": "Een berekening die voor veel mensen niet klopte"},
            "average": {"en": "A rule that was right on average, not for many people", "nl": "Een regel die gemiddeld klopte, voor veel mensen niet"},
            "judgment": {"en": "A number or a rule in place of judgment", "nl": "Een getal of een regel in plaats van een oordeel"},
        },
        items={
            "fortnights": {"en": "**The fortnights.** Robodebt spread the income of a year evenly over fortnights, as if everyone earned the same in each one.",
                           "nl": "**De periodes.** Robodebt verdeelde het inkomen van een jaar gelijk over periodes van twee weken, alsof iedereen in elke periode evenveel verdiende."},
            "amato": {"en": "**3,215.38 dollars.** Deanna Amato was told of a debt of 3,215.38 dollars; when Centrelink looked again, it was 1.48 dollars.",
                      "nl": "**3.215,38 dollar.** Deanna Amato kreeg een schuld van 3.215,38 dollar; toen Centrelink opnieuw keek, was het 1,48 dollar."},
            "past": {"en": "**The school's past.** The model predicted each school's spread of grades from its results in the three years before, and shared those grades out over its students.",
                     "nl": "**Het verleden van de school.** Het model voorspelde de verdeling van de cijfers op elke school uit haar resultaten in de drie jaar ervoor, en verdeelde die cijfers over haar leerlingen."},
            "totals": {"en": "**The totals.** Across England, the share of A and A* grades rose by only 2.4 points; yet 39.1 of every 100 grades came out lower than the teacher's.",
                       "nl": "**De totalen.** In heel Engeland steeg het aandeel A- en A*-cijfers maar met 2,4 punten; toch viel van elke 100 cijfers 39,1 lager uit dan dat van de docent."},
            "questionnaire": {"en": "**The questionnaire.** In Michigan, not returning a questionnaire could lead to a fraud finding, with a penalty of four times the benefits.",
                              "nl": "**De vragenlijst.** In Michigan kon het niet terugsturen van een vragenlijst leiden tot een vaststelling van fraude, met een boete van vier keer de uitkering."},
            "screen-in": {"en": "**The required investigation.** Above some scores, the Allegheny tool says a call must be investigated, unless a supervisor decides otherwise.",
                          "nl": "**Het verplichte onderzoek.** Boven sommige scores zegt het hulpmiddel van Allegheny dat een melding onderzocht moet worden, tenzij een leidinggevende anders beslist."},
        },
        solution={"fortnights": "calc", "amato": "calc", "past": "average", "totals": "average",
                  "questionnaire": "judgment", "screen-in": "judgment"},
        nudge={
            "fortnights": {"en": "Was the arithmetic itself right for someone whose income goes up and down?", "nl": "Klopte de rekensom zelf voor iemand van wie het inkomen op en neer gaat?"},
            "amato": {"en": "Where did the difference between the two amounts come from?", "nl": "Waar kwam het verschil tussen de twee bedragen vandaan?"},
            "past": {"en": "Does a school's past say how the school does, or how one student does?", "nl": "Zegt het verleden van een school hoe de school het doet, of hoe één leerling het doet?"},
            "totals": {"en": "Was it the total that was wrong, or the grades of students?", "nl": "Klopte het totaal niet, of de cijfers van leerlingen?"},
            "questionnaire": {"en": "Did anyone weigh why the questionnaire did not come back?", "nl": "Woog iemand af waarom de vragenlijst niet terugkwam?"},
            "screen-in": {"en": "Who decides when the score says *must*?", "nl": "Wie beslist er als de score *moet* zegt?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the facts again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de feiten opnieuw uit."},
            "empty": {"en": "Put each fact under the way it went wrong.",
                      "nl": "Zet elk feit bij de manier waarop het misging."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The fact *{item}* belongs with another way of going wrong.",
                      "nl": "Het feit *{item}* hoort bij een andere manier van misgaan."},
            "correct": {"en": "All {n} are right. A calculation can be tested against what really happened; a rule that is right on average needs a look at the people it gets wrong; and a number in place of judgment needs a person who can see the case and overrule it.",
                        "nl": "Alle {n} kloppen. Een berekening kun je testen tegen wat er echt gebeurde; bij een regel die gemiddeld klopt moet je kijken naar de mensen voor wie hij niet klopt; en een getal in plaats van een oordeel vraagt om iemand die de zaak kan zien en het getal kan overrulen."},
        },
    ),
}

_SORTED = {"fortnights": "calc", "amato": "calc", "past": "average", "totals": "average",
           "questionnaire": "judgment", "screen-in": "judgment"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("averaging", _START, "empty", None),
    ("averaging", _SOLUTION, "correct", None),
    ("averaging", _swap("    if average > amount:\n        invented = invented + (average - amount)\n",
                        "    if amount > average:\n        invented = invented + (amount - average)\n"), "correct", None),
    ("averaging", _swap("total = 0\nfor amount in earned:\n    total = total + amount\naverage = total / len(earned)\n",
                        "average = sum(earned) / 26\n"), "correct", None),
    ("averaging", _s(_BODY.replace("in earned", "in _earned")), "error", "underscore"),
    ("averaging", _swap("    if average > amount:\n        invented = invented + (average - amount)\n",
                        "    if total > amount:\n        invented = invented + (total - amount)\n"), "wrong", "total"),
    ("averaging", _swap("    if average > amount:\n        invented = invented + (average - amount)\n",
                        "    invented = invented + (average - amount)\n"), "wrong", "no-if"),
    ("averaging", _swap("invented = invented + (average - amount)", "invented = invented + 1"), "wrong", "counted"),
    ("averaging", _swap("    if average > amount:\n        invented = invented + (average - amount)\n",
                        "    if average > amount:\n        invented = invented + (average - amount)\n"
                        "    else:\n        invented = invented + (amount - average)\n"), "wrong", "both-ways"),
    ("averaging", _swap("    if average > amount:\n        invented = invented + (average - amount)\n",
                        "    if amount > average:\n        invented = invented + amount\n"), "wrong", None),
    ("averaging", _s("invented = 5\n"), "wrong", None),
    ("averaging", _SOLUTION.replace("[1000, 1000,", "[1000, 900,"), "wrong", "changed-line"),
    ("averaging", _swap("invented = invented + (average - amount)", "invented = invented + average"), "almost", "other-value"),
    ("averaging", _s("invented = 6500.0\n"), "almost", "other-value"),
    ("averaging", _s("...\ninvented = 0\n"), "wrong", "dots-left"),
    ("failures", {}, "empty", None),
    ("failures", _SORTED, "correct", None),
    ("failures", {"fortnights": "calc", "screen-in": "judgment"}, "wrong", "unfinished"),
    ("failures", {**_SORTED, "amato": "judgment"}, "wrong", "wrong:amato"),
    ("failures", {**_SORTED, "past": "calc"}, "wrong", "wrong:past"),
    ("failures", {**_SORTED, "totals": "calc"}, "wrong", "wrong:totals"),
    ("failures", {**_SORTED, "screen-in": "average"}, "wrong", "wrong:screen-in"),
]
