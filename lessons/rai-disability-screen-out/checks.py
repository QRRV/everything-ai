# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-disability-screen-out: counting how many people a filter on gaps in a CV removes while
they can do the job, sorting cases by what the EEOC warned about, and what Dutch law asks of an employer; with the
pictures of who the filter removes."""
import html
import re

from everything_ai.check import CodeBox, Mistake, Sort
from everything_ai import yours

# Made-up applicants: months without work in their CV, and whether they can do the job, perhaps with an adjustment.
APPLICANTS = [
    {"name": "Ada", "gap": 0, "can_do_job": "yes"},
    {"name": "Bram", "gap": 14, "can_do_job": "yes"},
    {"name": "Chen", "gap": 3, "can_do_job": "no"},
    {"name": "Dewi", "gap": 9, "can_do_job": "yes"},
    {"name": "Emre", "gap": 1, "can_do_job": "no"},
    {"name": "Fleur", "gap": 24, "can_do_job": "no"},
    {"name": "Gijs", "gap": 7, "can_do_job": "yes"},
    {"name": "Jesse", "gap": 6, "can_do_job": "yes"},
]
LIMIT = 6


def removed(applicants: list, limit: int) -> list:
    """The names of whoever has a gap of more than `limit` months."""
    return [a["name"] for a in applicants if a["gap"] > limit]


def wrongly(applicants: list, limit: int) -> int:
    """How many of the removed can do the job."""
    return len([a for a in applicants if a["gap"] > limit and a["can_do_job"] == "yes"])


def _ok(applicants) -> bool:
    if not isinstance(applicants, list) or not 1 <= len(applicants) <= 12:
        return False
    for a in applicants:
        if type(a) is not dict or not isinstance(a.get("gap"), int) or isinstance(a.get("gap"), bool):
            return False
        if not isinstance(a.get("name"), str) or a.get("can_do_job") not in ("yes", "no"):
            return False
    return True


def _gap_picture(applicants: list, limit, words: dict, title: str) -> str:
    """Each applicant as a dot on a scale of months without work, in one row for whoever can do the job and one for
    whoever cannot; a dashed line at the limit, and the dots to its right removed by the filter."""
    from everything_ai import chart
    top = max(26, max(a["gap"] for a in applicants) + 2)
    p = chart.Panel(x=(-2, top), y=(0.2, 4.2), xlabel=words["xlabel"], yticks=[],
                    xticks=[0, 6, 12, 18, 24], xtick_labels=["0", "6", "12", "18", "24"])
    rows = {"yes": 2.45, "no": 1.05}
    for key, y in rows.items():
        p.line([0, top - 1], [y, y], role="ink-3", width=1)
        p.text(-1.6, y - 0.42, words[key], "ink-2", anchor="start")
    if isinstance(limit, (int, float)) and not isinstance(limit, bool) and -2 <= limit <= top:
        p.line([limit + 0.5, limit + 0.5], [0.8, 3.65], role="ink-3", width=1, dashed=True)
        p.text(limit + 1.0, 3.55, words["limit"], "ink-3", anchor="start", halo=True)
    levels = {"yes": [], "no": []}   # per row, the x of the names already placed at each height
    for a in sorted(applicants, key=lambda a: a["gap"]):
        y = rows[a["can_do_job"]]
        out = isinstance(limit, (int, float)) and a["gap"] > limit
        p.points([a["gap"]], [y], role="ink", r=5, hollow=not out)
        placed = levels[a["can_do_job"]]
        lift = 0
        while lift < len(placed) and placed[lift] > a["gap"] - 2.6:
            lift += 1
        if lift == len(placed):
            placed.append(a["gap"])
        else:
            placed[lift] = a["gap"]
        p.text(a["gap"], y + 0.3 + 0.32 * lift, html.escape(a["name"]), "ink-2", halo=True)
    p.text(top - 1, 4.0, words["key"], "ink-3", anchor="end")
    alt = (f'{title}. ' + "; ".join(f'{a["name"]}: {a["gap"]} {words["months"]}, {words[a["can_do_job"]]}'
                                    for a in applicants) + ".")
    return yours.figure(title, chart.figure([p], alt=alt, width=540, height=280))


def picture(labels: dict) -> str:
    """The filter of the worked example on the made-up applicants. `labels` gives "title", "xlabel", "yes", "no",
    "limit", "months" and "key"."""
    return _gap_picture(APPLICANTS, LIMIT, labels, labels["title"])


def _number(v) -> int | None:
    if isinstance(v, bool) or not isinstance(v, int):
        return None
    return v


def _count_picture(seen) -> str:
    """The learner's applicants and limit, with the count they work out in the title."""
    mine = seen.mine
    if mine is None or not mine.has_value or _number(mine.value) is None:
        return ""
    names = mine.names if isinstance(mine.names, dict) else {}
    applicants = names.get("applicants", APPLICANTS)
    if not _ok(applicants):
        return ""
    words = {"en": {"xlabel": "months without work in the CV", "yes": "can do the job", "no": "cannot do the job",
                    "limit": "removed: more than {limit} months", "months": "months",
                    "key": "filled: removed by the filter, open: kept",
                    "title": "Removed, but able to do the job: {count}"},
             "nl": {"xlabel": "maanden zonder werk in het cv", "yes": "kan het werk doen", "no": "kan het werk niet doen",
                    "limit": "eruit: meer dan {limit} maanden", "months": "maanden",
                    "key": "gevuld: eruit gefilterd, open: blijft",
                    "title": "Eruit, maar kan het werk doen: {count}"}}[seen.lang]
    limit = names.get("limit", LIMIT)
    if isinstance(limit, bool) or not isinstance(limit, (int, float)):
        return ""
    words = {**words, "limit": words["limit"].format(limit=limit)}
    return _gap_picture(applicants, limit, words, words["title"].format(count=mine.value))


# The lesson is done when these are correct. "adjust" is the choice question in the lesson.
REQUIRED = ["wrongly", "concerns", "adjust"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _line(a: dict) -> str:
    return f"    {a!r},".replace("'", '"')


_APPLICANT_LINES = [_line(a) for a in APPLICANTS]
_LIMIT_LINE = f"limit = {LIMIT}"
_HEAD = ("# made-up applicants: months without work in their CV, and whether they can do the job\n"
         "applicants = [\n" + "\n".join(_APPLICANT_LINES) + "\n]\n"
         "# the filter: nobody with a gap of more than this many months\n" + _LIMIT_LINE + "\n\n")
_START = _HEAD + "...\nwrongly\n"
_BODY = ("wrongly = 0\nfor applicant in applicants:\n    if applicant[\"gap\"] > limit:\n"
         "        if applicant[\"can_do_job\"] == \"yes\":\n            wrongly = wrongly + 1\n")
_SOLUTION = _HEAD + _BODY + "wrongly"


def _s(body: str) -> str:
    return _HEAD + body + "wrongly"


def _swap(old: str, new: str) -> str:
    assert _BODY.count(old) == 1
    return _s(_BODY.replace(old, new))


def _is(value: int):
    return lambda got, text: _number(got) == value


EXERCISES = {
    "wrongly": CodeBox(
        name="wrongly",
        picture=_count_picture,
        start=_START,
        solution=_SOLUTION,
        keep=["applicants = ["] + _APPLICANT_LINES + [_LIMIT_LINE],
        # The check also runs the box with a filter of 12 months and one of 3 months.
        others=[{"limit": 12}, {"limit": 3}],
        first=[_DOTS],
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_applicants`. In the box the names have none: `applicants`, `limit`, `applicant`, `wrongly`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_applicants`. In het vak hebben de namen dat niet: `applicants`, `limit`, `applicant`, `wrongly`."),
            Mistake("key", error=lambda e: isinstance(e, KeyError),
                    en="Your code asks an applicant for a key it does not have. Each applicant has `\"name\"`, `\"gap\"` and `\"can_do_job\"`.",
                    nl="Je code vraagt een sollicitant om een sleutel die er niet is. Elke sollicitant heeft `\"name\"`, `\"gap\"` en `\"can_do_job\"`."),
            Mistake("strict", matches=lambda got, text: _number(got) == 4 and ">=" in text,
                    en="Your last line gives `4`. The filter removes a gap of *more than* `limit` months: Jesse, with exactly 6, stays. Use `>`, not `>=`.",
                    nl="Je laatste regel geeft `4`. Het filter haalt een gat van *meer dan* `limit` maanden eruit: Jesse, met precies 6, blijft. Gebruik `>`, niet `>=`."),
            Mistake("everyone", matches=_is(4),
                    en="Your last line gives `4`: everyone the filter removes. Count only those of them who can do the job: `applicant[\"can_do_job\"]` is `\"yes\"`.",
                    nl="Je laatste regel geeft `4`: iedereen die het filter eruit haalt. Tel alleen wie van hen het werk kan doen: `applicant[\"can_do_job\"]` is `\"yes\"`."),
            Mistake("cannot", matches=_is(1),
                    en="Your last line gives `1`: the removed applicants who *cannot* do the job. Count those who can: `\"yes\"`.",
                    nl="Je laatste regel geeft `1`: de sollicitanten die eruit gaan en het werk *niet* kunnen doen. Tel wie het wel kan: `\"yes\"`."),
            Mistake("kept", matches=_is(2),
                    en="Your last line gives `2`: the applicants who can do the job and whom the filter *keeps*. Count the ones it removes: a gap of more than `limit`.",
                    nl="Je laatste regel geeft `2`: de sollicitanten die het werk kunnen doen en die het filter *houdt*. Tel wie het eruit haalt: een gat van meer dan `limit`."),
            Mistake("all", matches=_is(5),
                    en="Your last line gives `5`: everyone who can do the job, also those the filter keeps. Count only the ones with a gap of more than `limit`.",
                    nl="Je laatste regel geeft `5`: iedereen die het werk kan doen, ook wie het filter houdt. Tel alleen wie een gat van meer dan `limit` heeft."),
            Mistake("list", matches=lambda got, text: isinstance(got, list),
                    en="Your last line gives a list, `{got}`. `wrongly` is a number: start at `0` and add 1 for each applicant you count.",
                    nl="Je laatste regel geeft een lijst, `{got}`. `wrongly` is een getal: begin bij `0` en tel er 1 bij op voor elke sollicitant die je telt."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a loop over the applicants, as in the worked example, that counts in `wrongly` the removed applicants who can do the job.",
                      "nl": "Vervang de `...` door een loop over de sollicitanten, zoals in het uitgewerkte voorbeeld, die in `wrongly` telt wie eruit gaat en het werk kan doen."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the lines of `applicants` and `limit` exactly as they were: put them back. The check tries another limit by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `applicants` en `limit` nodig zoals ze waren: zet ze terug. De check probeert zelf een andere grens."},
            "no_value": {"en": "The last line of the box shows nothing. Put `wrongly` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `wrongly` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Do you count an applicant only when the gap is more than `limit` and `can_do_job` is `\"yes\"`?",
                      "nl": "Je laatste regel geeft `{got}`. Tel je een sollicitant alleen als het gat groter is dan `limit` en `can_do_job` `\"yes\"` is?"},
            "other": {"en": "Right for these numbers. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you use `applicants` and `limit` themselves, and not their numbers?",
                      "nl": "Goed voor deze getallen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruik je `applicants` en `limit` zelf, en niet hun getallen?"},
            "other_error": {"en": "Right for these numbers. But with {where}, your code stops with an error. Do you use `applicants` and `limit` themselves?",
                            "nl": "Goed voor deze getallen. Maar met {where} stopt je code met een fout. Gebruik je `applicants` en `limit` zelf?"},
            "correct": {"en": "`3`: Bram, Dewi and Gijs can do the job, but the filter removes them. Of the four it removes, only Fleur cannot. A filter can look neutral and still keep out people whose gap came from an illness.",
                        "nl": "`3`: Bram, Dewi en Gijs kunnen het werk doen, maar het filter haalt ze eruit. Van de vier die eruit gaan, kan alleen Fleur het niet. Een filter kan neutraal lijken en toch mensen buiten houden van wie het gat door een ziekte kwam."},
        },
    ),
    "concerns": Sort(
        name="concerns",
        groups={
            "screen": {"en": "Screen out", "nl": "Uitfilteren"},
            "adjust": {"en": "An adjustment", "nl": "Een aanpassing"},
            "ask": {"en": "A question about health", "nl": "Een vraag naar gezondheid"},
        },
        items={
            "gap": {"en": "**The gap.** A chatbot rejects everyone with more than a year without work; one applicant spent that year in treatment.",
                    "nl": "**Het gat.** Een chatbot wijst iedereen af die meer dan een jaar zonder werk was; een sollicitant was dat jaar in behandeling."},
            "game": {"en": "**The memory game.** A job requires 90 percent on a memory game, and a blind applicant with a good memory cannot play it.",
                     "nl": "**Het geheugenspel.** Een baan vraagt 90 procent op een geheugenspel, en een blinde sollicitant met een goed geheugen kan het niet spelen."},
            "oral": {"en": "**Out loud.** An applicant who cannot type fast may give the answers to a timed keyboard test out loud.",
                     "nl": "**Hardop.** Een sollicitant die niet snel kan typen, mag de antwoorden op een getimede typtoets hardop geven."},
            "register": {"en": "**The lower register.** A cashier who uses a wheelchair gets a cash register that is lower.",
                         "nl": "**De lagere kassa.** Een kassamedewerker in een rolstoel krijgt een kassa die lager staat."},
            "illness": {"en": "**The form.** An online form asks every applicant whether they were ever treated for a mental illness.",
                        "nl": "**Het formulier.** Een online formulier vraagt elke sollicitant of die ooit is behandeld voor een psychische ziekte."},
            "medicines": {"en": "**Medicines.** A chatbot asks each applicant which medicines they take.",
                          "nl": "**Medicijnen.** Een chatbot vraagt elke sollicitant welke medicijnen die gebruikt."},
        },
        solution={"gap": "screen", "game": "screen", "oral": "adjust", "register": "adjust", "illness": "ask",
                  "medicines": "ask"},
        nudge={
            "gap": {"en": "Does the rule remove someone who could do the job?",
                    "nl": "Haalt de regel iemand eruit die het werk zou kunnen doen?"},
            "game": {"en": "Does the game measure memory for this applicant, or something else?",
                     "nl": "Meet het spel bij deze sollicitant het geheugen, of iets anders?"},
            "oral": {"en": "Is this a change that lets someone take part?",
                     "nl": "Is dit een verandering die iemand laat meedoen?"},
            "register": {"en": "Does this remove someone, or change the work so that they can do it?",
                         "nl": "Haalt dit iemand eruit, of verandert het het werk zodat die het kan doen?"},
            "illness": {"en": "What does the tool make the applicant tell?",
                        "nl": "Wat laat het programma de sollicitant vertellen?"},
            "medicines": {"en": "What can an employer learn from a list of medicines?",
                          "nl": "Wat kan een werkgever leren uit een lijst medicijnen?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the cases again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de gevallen opnieuw uit."},
            "empty": {"en": "Sort each case by what the EEOC warned about.",
                      "nl": "Deel elk geval in bij waar de EEOC voor waarschuwde."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The case *{item}* belongs somewhere else.",
                      "nl": "Het geval *{item}* hoort ergens anders."},
            "correct": {"en": "All {n} are right. These are the three warnings of the EEOC in 2022: screen out, adjustments, and questions about health that a tool makes applicants answer.",
                        "nl": "Alle {n} kloppen. Dit zijn de drie waarschuwingen van de EEOC uit 2022: uitfilteren, aanpassingen, en vragen naar gezondheid die een programma sollicitanten laat beantwoorden."},
        },
    ),
}

_SORTED = {"gap": "screen", "game": "screen", "oral": "adjust", "register": "adjust", "illness": "ask",
           "medicines": "ask"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("wrongly", _START, "empty", None),
    ("wrongly", _SOLUTION, "correct", None),
    ("wrongly", _s("wrongly = 0\nfor applicant in applicants:\n    if applicant[\"can_do_job\"] == \"yes\":\n"
                   "        if applicant[\"gap\"] > limit:\n            wrongly = wrongly + 1\n"), "correct", None),
    ("wrongly", _s("found = []\nfor applicant in applicants:\n    if applicant[\"gap\"] > limit:\n"
                   "        if applicant[\"can_do_job\"] == \"yes\":\n            found.append(applicant[\"name\"])\n"
                   "wrongly = len(found)\n"), "correct", None),
    ("wrongly", _s(_BODY.replace("in applicants", "in _applicants")), "error", "underscore"),
    ("wrongly", _swap('applicant["can_do_job"]', 'applicant["can_do"]'), "error", "key"),
    ("wrongly", _swap('applicant["gap"] > limit', 'applicant["gap"] >= limit'), "wrong", "strict"),
    ("wrongly", _s("wrongly = 0\nfor applicant in applicants:\n    if applicant[\"gap\"] > limit:\n"
                   "        wrongly = wrongly + 1\n"), "wrong", "everyone"),
    ("wrongly", _swap('== "yes"', '== "no"'), "wrong", "cannot"),
    ("wrongly", _swap('applicant["gap"] > limit', 'applicant["gap"] <= limit'), "wrong", "kept"),
    ("wrongly", _s("wrongly = 0\nfor applicant in applicants:\n    if applicant[\"can_do_job\"] == \"yes\":\n"
                   "        wrongly = wrongly + 1\n"), "wrong", "all"),
    ("wrongly", _s("wrongly = []\nfor applicant in applicants:\n    if applicant[\"gap\"] > limit:\n"
                   "        if applicant[\"can_do_job\"] == \"yes\":\n            wrongly.append(applicant[\"name\"])\n"),
     "wrong", "list"),
    ("wrongly", _swap("wrongly = wrongly + 1", "wrongly = wrongly + 2"), "wrong", None),
    ("wrongly", _s("wrongly = 3\n"), "almost", "other-value"),
    ("wrongly", _swap('applicant["gap"] > limit', 'applicant["gap"] > 6'), "almost", "other-value"),
    ("wrongly", _SOLUTION.replace("limit = 6", "limit = 12"), "wrong", "changed-line"),
    ("wrongly", _s("...\nwrongly = 3\n"), "wrong", "dots-left"),
    ("concerns", {}, "empty", None),
    ("concerns", _SORTED, "correct", None),
    ("concerns", {"gap": "screen", "medicines": "ask"}, "wrong", "unfinished"),
    ("concerns", {**_SORTED, "game": "adjust"}, "wrong", "wrong:game"),
    ("concerns", {**_SORTED, "register": "screen"}, "wrong", "wrong:register"),
    ("concerns", {**_SORTED, "illness": "screen"}, "wrong", "wrong:illness"),
]
