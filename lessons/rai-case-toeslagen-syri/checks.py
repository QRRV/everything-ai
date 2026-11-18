# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-case-toeslagen-syri: the risk score of each application on a small scorecard made up by the
course, sorting what went wrong in the childcare benefits scandal and SyRI, and the pictures of the chain from
application to label and of the learner's scores."""
import html
import re

from everything_ai.check import Code, CodeBox, Mistake, Sort
from everything_ai import yours

# A scorecard made up by the course, in the way of the Tax Administration's model: points per indicator, added up. Of
# the real model's few dozen indicators only two are public: nationality (Dutch or not) and the distance between home
# and childcare. "income change" is made up.
POINTS = {"distance": 2, "income change": 3, "not Dutch": 4}
# Four applications, made up: A and D did the same things, and so did B and C; only nationality differs.
APPLICATIONS = {
    "A": {"distance": False, "income change": True, "not Dutch": False},
    "B": {"distance": True, "income change": True, "not Dutch": False},
    "C": {"distance": True, "income change": True, "not Dutch": True},
    "D": {"distance": False, "income change": True, "not Dutch": True},
}
CHECKED_FROM = 6   # an application with at least this many points goes to a person for a check
# The worked example's application.
EXAMPLE = {"distance": True, "income change": False, "not Dutch": True}


def risk_score(application: dict, points: dict) -> int:
    """The sum of the points of the indicators that hold for the application."""
    return sum(points[i] for i in points if application[i])


def scores(applications: dict, points: dict) -> dict:
    return {name: risk_score(a, points) for name, a in applications.items()}


_UI = "font-family:var(--ea-font-ui)"


def picture(labels: dict) -> str:
    """The chain from an application to the label, as five numbered steps: the step of the model has a left border in
    the colour of predictions, the steps of the rules and the people after it one in grey. `labels` gives "title", "steps" (five texts),
    "model" and "rules" (the words of the key) and "alt"."""
    def card(i: int, words: str) -> str:
        edge = "var(--ea-pred)" if i == 1 else "var(--ea-ink-3)" if i > 1 else "var(--ea-line)"
        return (f'<div style="min-width:0;border:1px solid var(--ea-line);border-left:4px solid {edge};'
                f'border-radius:var(--ea-radius-small);padding:0.4rem 0.6rem;{_UI};font-size:0.85rem;color:var(--ea-ink);'
                f'line-height:1.35"><span style="font-weight:600">{i + 1}.</span> {html.escape(words)}</div>')

    cards = "".join(card(i, w) for i, w in enumerate(labels["steps"]))

    def swatch(edge: str) -> str:
        return (f'<svg width="16" height="12" aria-hidden="true"><rect x="1" y="1" width="4" height="10" '
                f'style="fill:{edge}"/><rect x="5" y="1" width="10" height="10" style="fill:none;stroke:var(--ea-line)"/></svg>')

    key = (f'<div style="{_UI};font-size:0.82rem;color:var(--ea-ink-2);display:flex;gap:1.2rem;flex-wrap:wrap;margin-top:0.5rem">'
           f'<span>{swatch("var(--ea-pred)")} {html.escape(labels["model"])}</span>'
           f'<span>{swatch("var(--ea-ink-3)")} {html.escape(labels["rules"])}</span></div>')
    body = (f'<div role="img" aria-label="{html.escape(labels["alt"])}"><div style="display:grid;'
            f'grid-template-columns:repeat(auto-fit,minmax(min(100%,150px),1fr));gap:0.6rem">{cards}</div>{key}</div>')
    return yours.figure(labels["title"], body)


def _number(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _scores_picture(seen) -> str:
    """The learner's score per application as a bar, with a dashed line at the right score and a line at the score
    from which an application is checked; the bars at or above it are marked."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value or not isinstance(mine.value, dict):
        return ""
    got, want = mine.value, scores(APPLICATIONS, POINTS)
    names = list(APPLICATIONS)
    top = 10
    panel = chart.Panel(x=(-0.6, len(names) - 0.4), y=(0, top), ylabel=seen.word({"en": "points", "nl": "punten"}),
                        yticks=[0, 2, 4, 6, 8, 10], xticks=list(range(len(names))), xtick_labels=names)
    panel.line([-0.6, len(names) - 0.4], [CHECKED_FROM] * 2, role="ink-2", width=1.5)
    for i, n in enumerate(names):
        v = _number(got.get(n))
        if v is not None and v > 0:
            panel.rect(i - 0.3, 0, i + 0.3, min(v, top), role="pred", solid=v >= CHECKED_FROM)
        panel.line([i - 0.38, i + 0.38], [want[n]] * 2, role="ink", width=2, dashed=True)
    panel.text(-0.55, CHECKED_FROM + 0.45, seen.word({"en": "checked from here", "nl": "gecontroleerd vanaf hier"}),
               "ink-2", anchor="start", halo=True)
    title = seen.word({"en": "Your score per application", "nl": "Jouw score per aanvraag"})
    words = {"en": ("your code, checked", "your code, not checked", "right"),
             "nl": ("jouw code, gecontroleerd", "jouw code, niet gecontroleerd", "goed")}[seen.lang]

    def box(solid: bool) -> str:
        fill = "fill:var(--ea-pred)" if solid else "fill:var(--ea-pred);fill-opacity:0.18"
        return (f'<svg width="14" height="12" aria-hidden="true"><rect x="1" y="1" width="12" height="10" style="{fill};'
                f'stroke:var(--ea-pred);stroke-width:1.2"/></svg>')

    dash = ('<svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" style="stroke:var(--ea-ink);'
            'stroke-width:2;stroke-dasharray:5 5"/></svg>')
    key = (f'<div style="{_UI};font-size:0.82rem;color:var(--ea-ink-2);display:flex;gap:1.2rem;flex-wrap:wrap;margin-top:0.3rem">'
           f'<span>{box(True)} {words[0]}</span><span>{box(False)} {words[1]}</span><span>{dash} {words[2]}</span></div>')
    return yours.figure(title, chart.figure([panel], alt=title, width=420, height=250) + key)


# The lesson is done when these are correct. "ruling" is the choice question in the lesson.
REQUIRED = ["score", "failures", "ruling"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _line(name: str, values) -> str:
    return f"{name} = {values!r}".replace("'", '"')


_POINTS_LINE = _line("points", POINTS)
_APPS = "applications = {\n" + "".join(f'    "{n}": {_line("", a)[3:]},\n' for n, a in APPLICATIONS.items()) + "}"
_HEAD = ("# points per indicator, made up by the course\n" + _POINTS_LINE + "\n"
         "# for each application, whether each indicator holds\n" + _APPS + "\n\n\ndef risk_score(application):\n")
_TAIL = "\n\n\nscores = {}\nfor name in applications:\n    scores[name] = risk_score(applications[name])\nscores"
_START = _HEAD + "    ..." + _TAIL + "\n"
_BODY = ("    score = 0\n    for indicator in points:\n        if application[indicator]:\n"
         "            score = score + points[indicator]\n    return score")
_SOLUTION = _HEAD + _BODY + _TAIL


def _s(body: str) -> str:
    return _HEAD + body + _TAIL


EXERCISES = {
    "score": CodeBox(
        name="score",
        picture=_scores_picture,
        start=_START,
        solution=_SOLUTION,
        keep=[_APPS],
        # The check also runs the box with a scorecard without points for nationality.
        others=[{"points": Code('{"distance": 2, "income change": 3, "not Dutch": 0}')}],
        first=[_DOTS],
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_points`. In the box the names have none: `points`, `application`, `score`, `indicator`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_points`. In het vak hebben de namen dat niet: `points`, `application`, `score`, `indicator`."),
            Mistake("no-return", matches=lambda got, text: isinstance(got, dict) and len(got) > 0 and all(v is None for v in got.values()),
                    en="Every score is `None`: your function gives nothing back. End it with `return score`, under the loop.",
                    nl="Elke score is `None`: je functie geeft niets terug. Sluit haar af met `return score`, onder de loop."),
            Mistake("return-in-loop", wrong=_s("    score = 0\n    for indicator in points:\n        if application[indicator]:\n"
                                               "            score = score + points[indicator]\n        return score"),
                    en="Your last line gives `{got}`: your function gives the score back after the first indicator. Put `return score` under the loop, not in it.",
                    nl="Je laatste regel geeft `{got}`: je functie geeft de score al na de eerste indicator terug. Zet `return score` onder de loop, niet erin."),
            Mistake("all-points", wrong=_s("    score = 0\n    for indicator in points:\n            score = score + points[indicator]\n    return score"),
                    en="Your last line gives `{got}`: every application gets all the points. Add only the points of an indicator that holds, with `if application[indicator]:`.",
                    nl="Je laatste regel geeft `{got}`: elke aanvraag krijgt alle punten. Tel alleen de punten op van een indicator die geldt, met `if application[indicator]:`."),
            Mistake("counted", wrong=_s("    score = 0\n    for indicator in points:\n        if application[indicator]:\n"
                                        "            score = score + 1\n    return score"),
                    en="Your last line gives `{got}`: how many indicators hold. Add the points of each indicator, from `points`.",
                    nl="Je laatste regel geeft `{got}`: hoeveel indicatoren gelden. Tel de punten van elke indicator op, uit `points`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the body of `risk_score`: start a score at 0, add the points of every indicator that holds for the application, and give the score back.",
                      "nl": "Vervang de `...` door de body van `risk_score`: begin een score bij 0, tel de punten op van elke indicator die voor de aanvraag geldt, en geef de score terug."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the lines of `applications` exactly as they were: put them back. The check tries other points by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `applications` nodig zoals ze waren: zet ze terug. De check probeert zelf andere punten."},
            "no_value": {"en": "The last line of the box shows nothing. Put `scores` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `scores` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does `risk_score` add, for each indicator that holds, its points from `points`, and give the sum back after the loop?",
                      "nl": "Je laatste regel geeft `{got}`. Telt `risk_score` voor elke indicator die geldt de punten uit `points` op, en geeft de functie de som na de loop terug?"},
            "other": {"en": "Right for these points. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you use `points` itself?",
                      "nl": "Goed voor deze punten. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruik je `points` zelf?"},
            "other_error": {"en": "Right for these points. But with {where}, your code stops with an error. Do you use `points` itself?",
                            "nl": "Goed voor deze punten. Maar met {where} stopt je code met een fout. Gebruik je `points` zelf?"},
            "correct": {"en": "`{got}`: C and D are checked, A and B are not. Yet A and D are the same in everything but nationality, and so are B and C. The check also tried a scorecard with no points for nationality: then A and D get 3 points, B and C 5, and nobody is checked.",
                        "nl": "`{got}`: C en D worden gecontroleerd, A en B niet. Toch zijn A en D in alles gelijk behalve de nationaliteit, en B en C ook. De check probeerde ook een scorekaart zonder punten voor nationaliteit: dan krijgen A en D 3 punten, B en C 5, en wordt niemand gecontroleerd."},
        },
    ),
    "failures": Sort(
        name="failures",
        groups={
            "hidden": {"en": "Nobody could see how it worked", "nl": "Niemand kon zien hoe het werkte"},
            "who": {"en": "Who was picked", "nl": "Wie er werd gekozen"},
            "cost": {"en": "What a small mistake cost", "nl": "Wat een kleine fout kostte"},
        },
        items={
            "staff": {"en": "**The staff.** The officials who checked an application did not know why the model had given it a high score.",
                      "nl": "**De medewerkers.** De ambtenaren die een aanvraag controleerden, wisten niet waarom het model die een hoge score had gegeven."},
            "court": {"en": "**The court.** SyRI's risk model and its indicators were secret, from the people it flagged and from the court.",
                      "nl": "**De rechtbank.** Het risicomodel van SyRI en de indicatoren waren geheim, voor de mensen die het aanwees en voor de rechtbank."},
            "nationality": {"en": "**The nationality.** The model gave points for whether an applicant had Dutch nationality or not.",
                            "nl": "**De nationaliteit.** Het model gaf punten voor de vraag of iemand de Nederlandse nationaliteit had of niet."},
            "neighborhoods": {"en": "**The neighborhoods.** SyRI was only used in neighborhoods that the government called problem areas.",
                              "nl": "**De wijken.** SyRI werd alleen gebruikt in wijken die de overheid probleemwijken noemde."},
            "signature": {"en": "**The signature.** A missing signature could mean paying back the benefit of a whole year.",
                          "nl": "**De handtekening.** Een ontbrekende handtekening kon betekenen dat ouders de toeslag van een heel jaar moesten terugbetalen."},
            "label": {"en": "**The label.** On top of paying back, parents could get the label *intent or gross negligence*, and with it no payment plan.",
                      "nl": "**Het label.** Bovenop het terugbetalen konden ouders het label *opzet/grove schuld* krijgen, en daarmee geen betalingsregeling."},
        },
        solution={"staff": "hidden", "court": "hidden", "nationality": "who", "neighborhoods": "who", "signature": "cost", "label": "cost"},
        nudge={
            "staff": {"en": "Could the officials explain to a parent why the application got a high score?", "nl": "Konden de ambtenaren een ouder uitleggen waarom de aanvraag een hoge score kreeg?"},
            "court": {"en": "What could the court not check?", "nl": "Wat kon de rechtbank niet controleren?"},
            "nationality": {"en": "Does this decide who is looked at, or what happens afterwards?", "nl": "Bepaalt dit wie er bekeken wordt, of wat er daarna gebeurt?"},
            "neighborhoods": {"en": "Who could be flagged at all, and who could not?", "nl": "Wie kon er überhaupt aangewezen worden, en wie niet?"},
            "signature": {"en": "Was the mistake large, and was the price?", "nl": "Was de fout groot, en de prijs?"},
            "label": {"en": "What happened to parents after the label, whatever they had done?", "nl": "Wat gebeurde er met ouders na het label, wat ze ook gedaan hadden?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the facts again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de feiten opnieuw uit."},
            "empty": {"en": "Put each fact under what went wrong.",
                      "nl": "Zet elk feit bij wat er misging."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The fact *{item}* belongs with something else that went wrong.",
                      "nl": "Het feit *{item}* hoort bij iets anders wat misging."},
            "correct": {"en": "All {n} are right. Each of the three would have done harm on its own; together, people were picked for reasons they could not see, and paid a high price for small mistakes.",
                        "nl": "Alle {n} kloppen. Elk van de drie had in zijn eentje al schade gedaan; samen werden mensen gekozen om redenen die ze niet konden zien, en betaalden ze een hoge prijs voor kleine fouten."},
        },
    ),
}

_SORTED = {"staff": "hidden", "court": "hidden", "nationality": "who", "neighborhoods": "who", "signature": "cost", "label": "cost"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("score", _START, "empty", None),
    ("score", _SOLUTION, "correct", None),
    ("score", _s("    score = 0\n    for indicator in application:\n        if application[indicator]:\n"
                 "            score = score + points[indicator]\n    return score"), "correct", None),
    ("score", _s("    score = 0\n    for indicator in points:\n        if application[indicator]:\n"
                 "            score = score + points[indicator]"), "wrong", "no-return"),
    ("score", _s("    score = 0\n    for indicator in points:\n        if application[indicator]:\n"
                 "            score = score + points[indicator]\n        return score"), "wrong", "return-in-loop"),
    ("score", _s("    score = 0\n    for indicator in points:\n            score = score + points[indicator]\n    return score"), "wrong", "all-points"),
    ("score", _s("    score = 0\n    for indicator in points:\n        if application[indicator]:\n"
                 "            score = score + 1\n    return score"), "wrong", "counted"),
    ("score", _s("    return 5"), "wrong", None),
    ("score", _s("    _score = 0\n    for _indicator in _points:\n        if application[_indicator]:\n"
                 "            _score = _score + _points[_indicator]\n    return _score"), "error", "underscore"),
    ("score", _SOLUTION.replace('"A": {"distance": False', '"A": {"distance": True'), "wrong", "changed-line"),
    ("score", _s("    return {'A': 3, 'B': 5, 'C': 9, 'D': 7}[[n for n in applications if applications[n] is application][0]]"),
     "almost", "other-value"),
    ("score", _s("    ...\n    return 0"), "wrong", "dots-left"),
    ("failures", {}, "empty", None),
    ("failures", _SORTED, "correct", None),
    ("failures", {"staff": "hidden", "label": "cost"}, "wrong", "unfinished"),
    ("failures", {**_SORTED, "staff": "who"}, "wrong", "wrong:staff"),
    ("failures", {**_SORTED, "neighborhoods": "hidden"}, "wrong", "wrong:neighborhoods"),
    ("failures", {**_SORTED, "label": "who"}, "wrong", "wrong:label"),
]
