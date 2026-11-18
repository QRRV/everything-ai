# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-discrimination-law: the three questions of objective justification as code, sorting
cases into direct discrimination, indirect discrimination and no distinction, and what a judgment of the Netherlands
Institute for Human Rights does; with the picture of how the law looks at a rule."""
import html
import re

from everything_ai.check import CodeBox, Mistake, Sort, run_code
from everything_ai import yours

# The three questions of objective justification (article 2 AWGB), as the Institute answered them about the code-98
# rule of the Dienst Toeslagen in judgment 2024-94: a legitimate aim, an appropriate means, but not a necessary one.
QUESTIONS = ["aim", "appropriate", "necessary"]
CODE_98 = {"aim": "yes", "appropriate": "yes", "necessary": "no"}
# A made-up rule for the worked example: it names no protected ground, but hits one group harder.
EXAMPLE = {"names_ground": "no", "hits_harder": "yes"}


def kind(rule: dict) -> str:
    """The first two questions of the picture, simplified: a rule that names a protected ground is direct
    discrimination; a neutral rule that hits one group harder is indirect; otherwise there is no distinction."""
    found = "no distinction"
    if rule["hits_harder"] == "yes":
        found = "indirect"
    if rule["names_ground"] == "yes":
        found = "direct"
    return found


def verdict(test: dict) -> str:
    """Indirect discrimination is allowed only when all three answers are yes."""
    found = "allowed"
    for question in QUESTIONS:
        if test[question] == "no":
            found = "forbidden"
    return found


def every_test() -> list[dict]:
    """The eight ways to answer the three questions, in a fixed order: all yes first, all no last."""
    tests = []
    for aim in ("yes", "no"):
        for appropriate in ("yes", "no"):
            for necessary in ("yes", "no"):
                tests.append({"aim": aim, "appropriate": appropriate, "necessary": necessary})
    return tests


_UI = "font-family:var(--ea-font-ui)"
_CODE = "font-family:var(--ea-font-code)"


def _card(number: str, question: str, inner: str) -> str:
    return (f'<div style="min-width:0;border:1px solid var(--ea-line);border-radius:var(--ea-radius-small);'
            f'padding:0.45rem 0.65rem">'
            f'<div style="{_UI};font-size:0.85rem;font-weight:600;color:var(--ea-ink);margin-bottom:0.25rem">'
            f'{number}. {html.escape(question)}</div>{inner}</div>')


def _way(words: str, strong: bool = False) -> str:
    weight = "font-weight:600;color:var(--ea-ink)" if strong else "color:var(--ea-ink-2)"
    return f'<div style="{_UI};font-size:0.82rem;{weight};line-height:1.35;margin-top:0.15rem">{html.escape(words)}</div>'


def picture(labels: dict) -> str:
    """How the law looks at a rule, in three steps. `labels` gives "title", "questions" (three texts), "ways" (two
    texts under each question: what yes and no lead to), "steps" (the three questions under question 3 as short words) and
    "alt"."""
    q, ways = labels["questions"], labels["ways"]
    chips = "".join(f'<span style="{_UI};font-size:0.8rem;color:var(--ea-ink);border:1px solid var(--ea-ink-3);'
                    f'border-radius:var(--ea-radius-small);padding:0.05rem 0.45rem;white-space:nowrap">'
                    f'{html.escape(s)}</span>' for s in labels["steps"])
    third = (f'<div style="display:flex;flex-wrap:wrap;gap:0.35rem;margin:0.2rem 0 0.1rem">{chips}</div>'
             + _way(ways[2][0], strong=True) + _way(ways[2][1], strong=True))
    cards = [_card("1", q[0], _way(ways[0][0], strong=True) + _way(ways[0][1])),
             _card("2", q[1], _way(ways[1][0], strong=True) + _way(ways[1][1])),
             _card("3", q[2], third)]
    body = (f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,190px),1fr));gap:0.8rem">{"".join(cards)}</div>')
    return yours.figure(labels["title"], body, alt=labels["alt"])


def _show(value) -> str:
    text = repr(value) if isinstance(value, str) else str(value)
    return html.escape(text if len(text) <= 24 else text[:22] + "…")


def _verdict_picture(seen) -> str:
    """The learner's rule run on all eight ways to answer the three questions, as a table beside what the law says;
    the row of the code-98 rule is marked."""
    mine = seen.mine
    if mine is None or not mine.has_value:
        return ""
    words = {"en": {"title": "What your code says for every answer", "law": "the law", "yours": "your verdict",
                    "same": "same", "differs": "differs", "error": "error", "case": "code 98",
                    "yes": "yes", "no": "no", "heads": ["aim", "appropriate", "necessary"]},
             "nl": {"title": "Wat jouw code zegt bij elk antwoord", "law": "de wet", "yours": "jouw oordeel",
                    "same": "gelijk", "differs": "anders", "error": "fout", "case": "code 98",
                    "yes": "ja", "no": "nee", "heads": ["doel", "passend", "noodzakelijk"]}}[seen.lang]
    cell = f'style="{_UI};font-size:0.82rem;padding:0.15rem 0.5rem;border-top:1px solid var(--ea-line);text-align:left"'
    head = "".join(f'<th {cell}>{html.escape(h)}</th>'
                   for h in words["heads"] + [words["yours"], words["law"], ""])
    rows = []
    for test in every_test():
        if test == CODE_98:
            run = mine
        else:
            run = run_code(seen.text.replace(_TEST_LINE, _line("test", test)), seen.names)
        want = verdict(test)
        if run.error is not None or run.timeout or not run.has_value:
            got, same = words["error"], False
        else:
            got, same = f'<code style="{_CODE}">{_show(run.value)}</code>', run.value == want
        answers = "".join(f'<td {cell}>{words[test[k]]}</td>' for k in QUESTIONS)
        mark = f' <span style="color:var(--ea-ink-2)">({words["case"]})</span>' if test == CODE_98 else ""
        weight = "" if same else ";font-weight:600"
        rows.append(f'<tr>{answers}<td {cell}>{got}</td><td {cell}><code style="{_CODE}">{_show(want)}</code></td>'
                    f'<td {cell[:-1]}{weight}">{words["same"] if same else words["differs"]}{mark}</td></tr>')
    table = (f'<table style="border-collapse:collapse;color:var(--ea-ink);max-width:100%">'
             f'<thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>')
    body = f'<div style="overflow-x:auto">{table}</div>'
    return yours.figure(words["title"], body)


# The lesson is done when these are correct. "college" is the choice question in the lesson.
REQUIRED = ["verdict", "kinds", "college"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _line(name: str, values) -> str:
    return f"{name} = {values!r}".replace("'", '"')


_TEST_LINE = _line("test", CODE_98)
_HEAD = "# the three questions of the Institute about the code-98 rule (judgment 2024-94)\n" + _TEST_LINE + "\n\n"
_START = _HEAD + "...\nverdict\n"
_BODY = ('verdict = "allowed"\nif test["aim"] == "no":\n    verdict = "forbidden"\n'
         'if test["appropriate"] == "no":\n    verdict = "forbidden"\nif test["necessary"] == "no":\n'
         '    verdict = "forbidden"\n')
_SOLUTION = _HEAD + _BODY + "verdict"


def _s(body: str) -> str:
    return _HEAD + body + "verdict"


def _swap(old: str, new: str) -> str:
    assert _BODY.count(old) == 1
    return _s(_BODY.replace(old, new))


def _odd_word(got) -> bool:
    return isinstance(got, str) and got != "allowed" and got != "forbidden"


EXERCISES = {
    "verdict": CodeBox(
        name="verdict",
        picture=_verdict_picture,
        start=_START,
        solution=_SOLUTION,
        keep=[_TEST_LINE],
        # The check also runs the box with other answers: all yes, and a no on each of the other two questions.
        others=[{"test": {"aim": "yes", "appropriate": "yes", "necessary": "yes"}},
                {"test": {"aim": "no", "appropriate": "yes", "necessary": "yes"}},
                {"test": {"aim": "yes", "appropriate": "no", "necessary": "yes"}}],
        first=[_DOTS],
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_rule`. In the box the names have none: `test` and `verdict`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_rule`. In het vak hebben de namen dat niet: `test` en `verdict`."),
            Mistake("key", error=lambda e: isinstance(e, KeyError),
                    en="Your code asks `test` for a key it does not have. Its keys are `\"aim\"`, `\"appropriate\"` and `\"necessary\"`, written exactly like that.",
                    nl="Je code vraagt `test` om een sleutel die het niet heeft. De sleutels zijn `\"aim\"`, `\"appropriate\"` en `\"necessary\"`, precies zo geschreven."),
            Mistake("allowed", matches=lambda got, text: got == "allowed",
                    en="Your last line gives `'allowed'`. But the code-98 rule failed one of the three questions. Which answer is `\"no\"`, and what should a no do to `verdict`?",
                    nl="Je laatste regel geeft `'allowed'`. Maar de regel met code 98 zakte voor een van de drie vragen. Welk antwoord is `\"no\"`, en wat moet een nee met `verdict` doen?"),
            Mistake("word", matches=lambda got, text: _odd_word(got),
                    en="Your last line gives `{got}`. Use exactly the words `\"allowed\"` and `\"forbidden\"`, in lowercase, so the check can compare them.",
                    nl="Je laatste regel geeft `{got}`. Gebruik precies de woorden `\"allowed\"` en `\"forbidden\"`, met kleine letters, zodat de check ze kan vergelijken."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with your lines: start `verdict` as `\"allowed\"`, and let every question answered `\"no\"` make it `\"forbidden\"`.",
                      "nl": "Vervang de `...` door je regels: begin `verdict` als `\"allowed\"`, en laat elke vraag met antwoord `\"no\"` er `\"forbidden\"` van maken."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line of `test` exactly as it was: put it back. The check tries other answers by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel van `test` nodig zoals die was: zet hem terug. De check probeert zelf andere antwoorden."},
            "no_value": {"en": "The last line of the box shows nothing. Put `verdict` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `verdict` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does `verdict` start as `\"allowed\"`, and does each question answered `\"no\"` make it `\"forbidden\"`?",
                      "nl": "Je laatste regel geeft `{got}`. Begint `verdict` als `\"allowed\"`, en maakt elke vraag met antwoord `\"no\"` er `\"forbidden\"` van?"},
            "other": {"en": "Right for code 98. But with {where}, your last line gives `{got}`, and it should give `{want}`. Does every one of the three questions answered `\"no\"` make `verdict` `\"forbidden\"`, and only a no?",
                      "nl": "Goed voor code 98. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Maakt elk van de drie vragen met antwoord `\"no\"` van `verdict` `\"forbidden\"`, en alleen een nee?"},
            "other_error": {"en": "Right for code 98. But with {where}, your code stops with an error. Does `verdict` get a value before the first `if`, also when every answer is yes?",
                            "nl": "Goed voor code 98. Maar met {where} stopt je code met een fout. Krijgt `verdict` een waarde voor de eerste `if`, ook als elk antwoord ja is?"},
            "correct": {"en": "`'forbidden'`: the code-98 rule served a legitimate aim with an appropriate means, but it was not necessary, as the Dienst Toeslagen could have checked the code before it stopped a benefit. One no is enough: the Institute found prohibited distinction on grounds of race.",
                        "nl": "`'forbidden'`: de regel met code 98 diende een legitiem doel met een passend middel, maar was niet noodzakelijk, want de Dienst Toeslagen had de code kunnen controleren voordat hij een toeslag stopzette. Eén nee is genoeg: het College vond verboden onderscheid op grond van ras."},
        },
    ),
    "kinds": Sort(
        name="kinds",
        groups={
            "direct": {"en": "Direct discrimination", "nl": "Directe discriminatie"},
            "indirect": {"en": "Indirect discrimination", "nl": "Indirecte discriminatie"},
            "none": {"en": "No distinction", "nl": "Geen onderscheid"},
        },
        items={
            "score": {"en": "**The lower score.** A hiring algorithm has a variable that gives women a lower recommendation.",
                      "nl": "**De lagere score.** Een algoritme voor werving heeft een variabele die vrouwen een lagere aanbeveling geeft."},
            "border": {"en": "**The border check.** Officers pick people for a check at the border partly by their skin color.",
                       "nl": "**De grenscontrole.** Agenten kiezen mensen voor een controle aan de grens mede op hun huidskleur."},
            "dutch": {"en": "**Perfect Dutch.** A job ad for a programmer asks for perfect Dutch.",
                      "nl": "**Perfect Nederlands.** Een vacature voor een programmeur vraagt om perfect Nederlands."},
            "match": {"en": "**The match.** A dating app learns from likes and gives users with a darker skin a lower chance of a match, without any data on origin.",
                      "nl": "**De match.** Een datingapp leert van likes en geeft gebruikers met een donkere huid een kleinere kans op een match, zonder gegevens over afkomst."},
            "students": {"en": "**Students only.** A supermarket's tool invites only pupils and students for jobs stacking shelves.",
                         "nl": "**Alleen scholieren.** Het hulpmiddel van een supermarkt nodigt alleen scholieren en studenten uit voor werk als vakkenvuller."},
            "test": {"en": "**The coding test.** A programming test for a programming job, which every group of applicants passes equally often.",
                     "nl": "**De programmeertoets.** Een programmeertoets voor een baan als programmeur, die elke groep sollicitanten even vaak haalt."},
        },
        solution={"score": "direct", "border": "direct", "dutch": "indirect", "match": "indirect", "students": "indirect",
                  "test": "none"},
        nudge={
            "score": {"en": "Does the rule itself name a protected ground?", "nl": "Noemt de regel zelf een discriminatiegrond?"},
            "border": {"en": "Is skin color named in the choice itself?", "nl": "Wordt huidskleur in de keuze zelf genoemd?"},
            "dutch": {"en": "The ad names no origin. Who does the rule hit harder?", "nl": "De vacature noemt geen afkomst. Wie raakt de regel harder?"},
            "match": {"en": "The app uses no data on origin. Look at the effect.", "nl": "De app gebruikt geen gegevens over afkomst. Kijk naar het effect."},
            "students": {"en": "The rule names no age. Who does it keep out?", "nl": "De regel noemt geen leeftijd. Wie houdt hij buiten?"},
            "test": {"en": "Does the test hit any group harder?", "nl": "Raakt de toets een groep harder?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the cases again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de gevallen opnieuw uit."},
            "empty": {"en": "Sort each case with the first two questions of the picture.",
                      "nl": "Deel elk geval in met de eerste twee vragen van het plaatje."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The case *{item}* belongs somewhere else.",
                      "nl": "Het geval *{item}* hoort ergens anders."},
            "correct": {"en": "All {n} are right. Whether the three indirect cases are also forbidden depends on the three questions: perfect Dutch can be needed for a teacher, but not for a programmer.",
                        "nl": "Alle {n} kloppen. Of de drie indirecte gevallen ook verboden zijn, hangt af van de drie vragen: perfect Nederlands kan nodig zijn voor een leraar, maar niet voor een programmeur."},
        },
    ),
}

_SORTED = {"score": "direct", "border": "direct", "dutch": "indirect", "match": "indirect", "students": "indirect",
           "test": "none"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("verdict", _START, "empty", None),
    ("verdict", _SOLUTION, "correct", None),
    ("verdict", _s('verdict = "allowed"\nfor question in test:\n    if test[question] == "no":\n'
                   '        verdict = "forbidden"\n'), "correct", None),
    ("verdict", _s('verdict = "forbidden"\nif test["aim"] == "yes":\n    if test["appropriate"] == "yes":\n'
                   '        if test["necessary"] == "yes":\n            verdict = "allowed"\n'), "correct", None),
    ("verdict", _s(_BODY.replace("test[", "_test[")), "error", "underscore"),
    ("verdict", _swap('test["appropriate"]', 'test["suitable"]'), "error", "key"),
    ("verdict", _swap('if test["necessary"] == "no":\n    verdict = "forbidden"\n', ""), "wrong", "allowed"),
    ("verdict", _s('verdict = "allowed"\n'), "wrong", "allowed"),
    ("verdict", _s('verdict = "Forbidden"\n'), "wrong", "word"),
    ("verdict", _s('verdict = "verboden"\n'), "wrong", "word"),
    ("verdict", _s("verdict = False\n"), "wrong", None),
    ("verdict", _s('verdict = "forbidden"\n'), "almost", "other-value"),
    ("verdict", _swap('if test["aim"] == "no":\n    verdict = "forbidden"\n', ""), "almost", "other-value"),
    ("verdict", _swap('if test["appropriate"] == "no":\n    verdict = "forbidden"\n', ""), "almost", "other-value"),
    ("verdict", _swap('verdict = "allowed"\n', ""), "almost", "other-value"),
    ("verdict", _s(_BODY.replace('== "no"', '== "yes"')), "almost", "other-value"),
    ("verdict", _SOLUTION.replace('"necessary": "no"', '"necessary": "yes"'), "wrong", "changed-line"),
    ("verdict", _s('...\nverdict = "forbidden"\n'), "wrong", "dots-left"),
    ("kinds", {}, "empty", None),
    ("kinds", _SORTED, "correct", None),
    ("kinds", {"score": "direct", "test": "none"}, "wrong", "unfinished"),
    ("kinds", {**_SORTED, "dutch": "none"}, "wrong", "wrong:dutch"),
    ("kinds", {**_SORTED, "border": "indirect"}, "wrong", "wrong:border"),
    ("kinds", {**_SORTED, "test": "indirect"}, "wrong", "wrong:test"),
]
