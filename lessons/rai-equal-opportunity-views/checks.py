# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-equal-opportunity-views: who gets in when the best of each school are taken instead of
one bar for everyone, sorting claims by the view of equal opportunity they belong to, and comparing people by
their rank within their own group; with the pictures of the three views and of who gets in."""
import html
import re

from everything_ai.check import CodeBox, Mistake, Sort
from everything_ai import yours

# Made-up test scores of six students at each of two schools, from high to low. The narrow view sets one bar for
# everyone; the middle view, like the Texas rule for its state universities, takes the best of each school.
SCORES = {"North": [91, 88, 85, 82, 76, 70], "South": [83, 77, 74, 69, 65, 61]}
BAR = 80
TOP = 2


def narrow(scores: dict, bar: float) -> dict:
    """Who gets in with one bar for everyone, per school."""
    found = {}
    for school in scores:
        found[school] = [s for s in scores[school] if s >= bar]
    return found


def middle(scores: dict, top: int) -> dict:
    """Who gets in when the best `top` of each school get a place, per school."""
    return {school: scores[school][:top] for school in scores}


_UI = "font-family:var(--ea-font-ui)"


def _card(name: str, rows: list) -> str:
    inner = "".join(
        f'<div style="{_UI};font-size:0.78rem;color:var(--ea-ink-3);margin-top:0.35rem">{html.escape(label)}</div>'
        f'<div style="{_UI};font-size:0.84rem;color:var(--ea-ink);line-height:1.35">{html.escape(value)}</div>'
        for label, value in rows)
    return (f'<div style="min-width:0;border:1px solid var(--ea-line);border-radius:var(--ea-radius-small);'
            f'padding:0.45rem 0.65rem">'
            f'<div style="{_UI};font-size:0.88rem;font-weight:600;color:var(--ea-ink)">{html.escape(name)}</div>'
            f'{inner}</div>')


def picture(labels: dict) -> str:
    """The three views side by side. `labels` gives "title", "rows" (the three row names), "views" (per view a name
    and three texts, one per row) and "alt"."""
    cards = [_card(view["name"], list(zip(labels["rows"], view["texts"]))) for view in labels["views"]]
    body = ('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,180px),1fr));gap:0.8rem">'
            + "".join(cards) + "</div>")
    return yours.figure(labels["title"], body, alt=labels["alt"])


def _numbers(v) -> list | None:
    if not isinstance(v, list):
        return None
    for x in v:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return None
    return v


def _admitted_picture(seen) -> str:
    """Each school's scores as dots on one scale, filled for whoever gets in by the learner's code, with the one bar
    of the narrow view as a dashed line to compare."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value or type(mine.value) is not dict:
        return ""
    names = mine.names if isinstance(mine.names, dict) else {}
    scores = names.get("scores") if type(names.get("scores")) is dict else SCORES
    words = {"en": {"title": "Who gets in by your code", "xlabel": "test score", "bar": f"one bar: {BAR}",
                    "of": "of", "key": "filled: gets in, open: does not"},
             "nl": {"title": "Wie er door jouw code binnenkomt", "xlabel": "testscore", "bar": f"één lat: {BAR}",
                    "of": "van", "key": "gevuld: komt binnen, open: niet"}}[seen.lang]
    p = chart.Panel(x=(52, 98), y=(0, 3.5), xlabel=words["xlabel"], yticks=[],
                    xticks=[55, 65, 75, 85, 95], xtick_labels=["55", "65", "75", "85", "95"])
    p.line([BAR, BAR], [0.55, 2.95], role="ink-3", width=1, dashed=True)
    p.text(BAR, 3.12, words["bar"], "ink-3", halo=True)
    p.text(97.5, 0.2, words["key"], "ink-3", anchor="end")
    shown = []
    for i, school in enumerate(list(scores)[:2]):
        row = _numbers(scores[school])
        if row is None:
            return ""
        y = 2.35 - 1.45 * i
        got = mine.value.get(school)
        picked = _numbers(got) if got is not None else []
        if picked is None:
            return ""
        p.line([55, 95], [y, y], role="ink-3", width=1)
        inside = [s for s in row if s in picked and 52 <= s <= 98]
        outside = [s for s in row if s not in picked and 52 <= s <= 98]
        if outside:
            p.points(outside, [y] * len(outside), role="ink-2", r=5, hollow=True)
        if inside:
            p.points(inside, [y] * len(inside), role="ink", r=5)
        for s in row:
            if 52 <= s <= 98:
                p.text(s, y + 0.3, str(s), "ink-2")
        label = f'{html.escape(str(school))}: {len(inside)} {words["of"]} {len(row)}'
        p.text(52.6, y - 0.42, label, "ink-2", anchor="start")
        shown.append(label)
    title = words["title"]
    alt = f'{title}. ' + "; ".join(shown) + f'. {words["bar"]}.'
    return yours.figure(title, chart.figure([p], alt=alt, width=520, height=250))


# The lesson is done when these are correct. "rank" is the choice question in the lesson.
REQUIRED = ["admitted", "views", "rank"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _line(name: str, values) -> str:
    return f"{name} = {values!r}".replace("'", '"')


_SCORES_LINE = _line("scores", SCORES)
_TOP_LINE = f"top = {TOP}"
_HEAD = ("# made-up test scores of six students at each school, from high to low\n" + _SCORES_LINE + "\n"
         "# the middle view: the best of each school get a place\n" + _TOP_LINE + "\n\n")
_START = _HEAD + "...\nadmitted\n"
_BODY = "admitted = {}\nfor school in scores:\n    admitted[school] = scores[school][:top]\n"
_SOLUTION = _HEAD + _BODY + "admitted"


def _s(body: str) -> str:
    return _HEAD + body + "admitted"


def _swap(old: str, new: str) -> str:
    assert _BODY.count(old) == 1
    return _s(_BODY.replace(old, new))


def _is(value):
    return lambda got, text: type(got) is dict and got == value


EXERCISES = {
    "admitted": CodeBox(
        name="admitted",
        picture=_admitted_picture,
        start=_START,
        solution=_SOLUTION,
        keep=[_SCORES_LINE, _TOP_LINE],
        # The check also runs the box with three places per school, and with other scores.
        others=[{"top": 3}, {"scores": {"North": [90, 70, 60], "South": [95, 85, 50]}}],
        first=[_DOTS],
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_scores`. In the box the names have none: `scores`, `top`, `admitted`, `school`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_scores`. In het vak hebben de namen dat niet: `scores`, `top`, `admitted`, `school`."),
            Mistake("bar", matches=_is(narrow(SCORES, BAR)),
                    en="Your last line gives `{got}`: everyone with at least 80, the narrow view of the worked example. Here the best `top` of each school get in, whatever their score.",
                    nl="Je laatste regel geeft `{got}`: iedereen met minstens 80, de smalle opvatting uit het uitgewerkte voorbeeld. Hier komen de beste `top` van elke school binnen, wat hun score ook is."),
            Mistake("first", matches=_is({"North": [88, 85], "South": [77, 74]}),
                    en="Your last line gives `{got}`: the best of each school is missing. A list starts counting at 0, so `[:top]` starts at the first score.",
                    nl="Je laatste regel geeft `{got}`: de beste van elke school ontbreekt. Een lijst begint te tellen bij 0, dus `[:top]` begint bij de eerste score."),
            Mistake("lowest", matches=_is({"North": [76, 70], "South": [65, 61]}),
                    en="Your last line gives `{got}`: the lowest scores. The lists run from high to low, so the best come first: take the first `top`.",
                    nl="Je laatste regel geeft `{got}`: de laagste scores. De lijsten lopen van hoog naar laag, dus de besten staan vooraan: neem de eerste `top`."),
            Mistake("one", matches=lambda got, text: type(got) is dict and got == {"North": [85], "South": [74]}
                    or got == {"North": 85, "South": 74},
                    en="Your last line gives `{got}`: one score per school, the one at index `top`. A slice gives several: `[:top]` gives the first `top` scores.",
                    nl="Je laatste regel geeft `{got}`: één score per school, die op index `top`. Een slice geeft er meer: `[:top]` geeft de eerste `top` scores."),
            Mistake("list", matches=lambda got, text: isinstance(got, list),
                    en="Your last line gives a list, `{got}`, not a dictionary. `admitted` needs a list per school: start with `admitted = {{}}` and fill `admitted[school]` in the loop.",
                    nl="Je laatste regel geeft een lijst, `{got}`, geen dictionary. `admitted` heeft een lijst per school nodig: begin met `admitted = {{}}` en vul `admitted[school]` in de loop."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a loop over the schools, as in the worked example, that gives `admitted[school]` the first `top` scores of each school.",
                      "nl": "Vervang de `...` door een loop over de scholen, zoals in het uitgewerkte voorbeeld, die `admitted[school]` de eerste `top` scores van elke school geeft."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the lines of `scores` and `top` exactly as they were: put them back. The check tries other numbers by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `scores` en `top` nodig zoals ze waren: zet ze terug. De check probeert zelf andere getallen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `admitted` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `admitted` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does `admitted` get, for each school, the first `top` scores of that school's list?",
                      "nl": "Je laatste regel geeft `{got}`. Krijgt `admitted` voor elke school de eerste `top` scores uit de lijst van die school?"},
            "other": {"en": "Right for these numbers. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you use `scores` and `top` themselves, and not their numbers?",
                      "nl": "Goed voor deze getallen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruik je `scores` en `top` zelf, en niet hun getallen?"},
            "other_error": {"en": "Right for these numbers. But with {where}, your code stops with an error. Do you use `scores` and `top` themselves?",
                            "nl": "Goed voor deze getallen. Maar met {where} stopt je code met een fout. Gebruik je `scores` en `top` zelf?"},
            "correct": {"en": "`{{'North': [91, 88], 'South': [83, 77]}}`: two from each school. South's 77 gets in and North's 85 does not, although 85 is higher. With one bar of 80 it was four from North and one from South.",
                        "nl": "`{{'North': [91, 88], 'South': [83, 77]}}`: twee van elke school. De 77 van South komt binnen en de 85 van North niet, al is 85 hoger. Met één lat van 80 waren het er vier van North en één van South."},
        },
    ),
    "views": Sort(
        name="views",
        groups={
            "narrow": {"en": "The narrow view", "nl": "De smalle opvatting"},
            "middle": {"en": "The middle view", "nl": "De middenopvatting"},
            "broad": {"en": "The broad view", "nl": "De brede opvatting"},
        },
        items={
            "careers": {"en": "**Open to talents.** The place goes to whoever has the best qualifications now.",
                        "nl": "**Open voor talent.** De plek gaat naar wie nu de beste papieren heeft."},
            "same": {"en": "**Same score, same decision.** Two applicants who score the same are treated the same, whatever their group.",
                     "nl": "**Zelfde score, zelfde besluit.** Twee sollicitanten met dezelfde score worden hetzelfde behandeld, wat hun groep ook is."},
            "class": {"en": "**Top of the class.** The best 10 percent of every high school get a place at a state university.",
                      "nl": "**Beste van de klas.** De beste 10 procent van elke middelbare school krijgt een plek aan een staatsuniversiteit."},
            "injustice": {"en": "**Past injustice.** A decision maker discounts a gap in scores that comes from past injustice.",
                          "nl": "**Vroeger onrecht.** Wie beslist, telt een verschil in scores dat uit vroeger onrecht komt minder mee."},
            "schools": {"en": "**Better schools.** A country funds schools so that children with the same talent and effort get the same prospects.",
                        "nl": "**Betere scholen.** Een land betaalt scholen zo dat kinderen met hetzelfde talent en dezelfde inzet dezelfde vooruitzichten krijgen."},
            "born": {"en": "**Born anywhere.** The family a person is born into should not change their prospects; talent and willingness may.",
                     "nl": "**Overal geboren.** Het gezin waarin iemand geboren wordt, mag de vooruitzichten niet veranderen; talent en inzet wel."},
        },
        solution={"careers": "narrow", "same": "narrow", "class": "middle", "injustice": "middle", "schools": "broad",
                  "born": "broad"},
        nudge={
            "careers": {"en": "Does this look at people as they are now, or at where they started?",
                        "nl": "Kijkt dit naar mensen zoals ze nu zijn, of naar waar ze begonnen?"},
            "same": {"en": "Treating people who are alike now in the same way: which view is only that?",
                     "nl": "Mensen die nu gelijk zijn hetzelfde behandelen: welke opvatting is alleen dat?"},
            "class": {"en": "Is this one decision that takes the starting point into account, or the design of a whole society?",
                      "nl": "Is dit één besluit dat rekening houdt met het beginpunt, of de inrichting van een hele samenleving?"},
            "injustice": {"en": "Which view treats people who now look different the same, because of where the difference comes from?",
                          "nl": "Welke opvatting behandelt mensen die nu verschillen hetzelfde, vanwege waar het verschil vandaan komt?"},
            "schools": {"en": "Is this about a decision maker, or about the institutions of a society?",
                        "nl": "Gaat dit over wie beslist, of over de instellingen van een samenleving?"},
            "born": {"en": "Whose view lets talent and willingness count, but not the starting point in society?",
                     "nl": "Wiens opvatting laat talent en inzet meetellen, maar niet het beginpunt in de samenleving?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the claims again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de beweringen opnieuw uit."},
            "empty": {"en": "Sort each claim by the view it belongs to.",
                      "nl": "Deel elke bewering in bij de opvatting waar ze bij hoort."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The claim *{item}* belongs somewhere else.",
                      "nl": "De bewering *{item}* hoort ergens anders."},
            "correct": {"en": "All {n} are right. The narrow view starts the competition at the decision, the middle view a little earlier, and the broad view at birth.",
                        "nl": "Alle {n} kloppen. De smalle opvatting laat de wedstrijd beginnen bij het besluit, de middenopvatting iets eerder, en de brede bij de geboorte."},
        },
    ),
}

_SORTED = {"careers": "narrow", "same": "narrow", "class": "middle", "injustice": "middle", "schools": "broad",
           "born": "broad"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("admitted", _START, "empty", None),
    ("admitted", _SOLUTION, "correct", None),
    ("admitted", _s('admitted = {"North": scores["North"][:top], "South": scores["South"][:top]}\n'), "correct", None),
    ("admitted", _s("admitted = {}\nfor school in scores:\n    admitted[school] = []\n"
                    "    for score in scores[school][:top]:\n        admitted[school].append(score)\n"), "correct", None),
    ("admitted", _s(_BODY.replace("in scores", "in _scores")), "error", "underscore"),
    ("admitted", _s("admitted = {}\nfor school in scores:\n    admitted[school] = []\n"
                    "    for score in scores[school]:\n        if score >= 80:\n"
                    "            admitted[school].append(score)\n"), "wrong", "bar"),
    ("admitted", _swap("[:top]", "[1:top + 1]"), "wrong", "first"),
    ("admitted", _swap("[:top]", "[-top:]"), "wrong", "lowest"),
    ("admitted", _swap("[:top]", "[top]"), "wrong", "one"),
    ("admitted", _s("admitted = []\nfor school in scores:\n    admitted = scores[school][:top]\n"), "wrong", "list"),
    ("admitted", _swap("[:top]", "[:top + 1]"), "wrong", None),
    ("admitted", _s('admitted = {"North": [91, 88], "South": [83, 77]}\n'), "almost", "other-value"),
    ("admitted", _swap("[:top]", "[:2]"), "almost", "other-value"),
    ("admitted", _SOLUTION.replace("top = 2", "top = 3"), "wrong", "changed-line"),
    ("admitted", _s('...\nadmitted = {"North": [91, 88], "South": [83, 77]}\n'), "wrong", "dots-left"),
    ("views", {}, "empty", None),
    ("views", _SORTED, "correct", None),
    ("views", {"careers": "narrow", "born": "broad"}, "wrong", "unfinished"),
    ("views", {**_SORTED, "class": "broad"}, "wrong", "wrong:class"),
    ("views", {**_SORTED, "same": "middle"}, "wrong", "wrong:same"),
    ("views", {**_SORTED, "born": "middle"}, "wrong", "wrong:born"),
]
