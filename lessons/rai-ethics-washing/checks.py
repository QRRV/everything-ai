# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-ethics-washing: how many of three tests (a check, an owner, a consequence) each promise of
a made-up company passes, sorting promises that cost nothing when broken from those someone would notice, and the
picture of the promises against the three tests."""
import html
import re

from everything_ai.check import Code, CodeBox, Mistake, Sort
from everything_ai import yours

# The promises of a company that builds AI, made up by the course. For each promise: 1 when it has a check (a way to
# find out that it was broken), an owner (someone who answers for it) or a consequence (something happens when it is
# broken), 0 when it does not.
PROMISES = ["fair AI", "bias test", "outside test", "ethics board", "complaints", "planet"]
CHECK = {"fair AI": 0, "bias test": 1, "outside test": 1, "ethics board": 0, "complaints": 1, "planet": 0}
OWNER = {"fair AI": 0, "bias test": 1, "outside test": 0, "ethics board": 1, "complaints": 1, "planet": 0}
CONSEQUENCE = {"fair AI": 0, "bias test": 0, "outside test": 0, "ethics board": 0, "complaints": 1, "planet": 0}
TESTS = {"check": CHECK, "owner": OWNER, "consequence": CONSEQUENCE}


def passed(check: dict, owner: dict, consequence: dict) -> dict:
    """How many of the three tests each promise passes."""
    return {p: check[p] + owner[p] + consequence[p] for p in check}


def words_only(check: dict, owner: dict, consequence: dict) -> list[str]:
    """The promises that pass none of the three tests."""
    return [p for p, n in passed(check, owner, consequence).items() if n == 0]


def picture(labels: dict) -> str:
    """The promises against the three tests: a filled dot where a promise has it, an open one where it does not, and
    "only words" beside a promise without any. `labels` gives "title", "tests" (three words), "words" and "alt"."""
    from everything_ai import chart
    rows = len(PROMISES)
    panel = chart.Panel(x=(-2.6, 5.6), y=(-0.6, rows - 0.4), title=labels["title"], yticks=[], xticks=[0, 1.5, 3],
                        xtick_labels=labels["tests"])
    counts = passed(CHECK, OWNER, CONSEQUENCE)
    for k, promise in enumerate(PROMISES):
        y = rows - 1 - k
        panel.text(-2.5, y - 0.12, promise, "ink-2", "start")
        for i, test in enumerate(TESTS.values()):
            panel.points([1.5 * i], [y], "data", r=6, hollow=not test[promise])
        if counts[promise] == 0:
            panel.text(3.45, y - 0.12, labels["words"], "ink", "start")
    return chart.figure([panel], alt=labels["alt"], width=540, height=32 * rows + 70)


def _passed_picture(seen) -> str:
    """Each promise with the learner's total, the right one where it differs, and "only words" at a total of 0."""
    mine = seen.mine
    if mine is None or not mine.has_value or not isinstance(mine.value, dict):
        return ""
    right = passed(CHECK, OWNER, CONSEQUENCE)
    rows = []
    for promise in PROMISES:
        got = mine.value.get(promise, "?")
        word = seen.word({"en": "only words", "nl": "alleen woorden"}) if got == 0 else ""
        mark = "" if got == right[promise] else seen.word({"en": f"(the course: {right[promise]})", "nl": f"(de course: {right[promise]})"})
        rows.append(f'<tr><td style="padding:0.15rem 0.8rem 0.15rem 0"><code>{promise}</code></td>'
                    f'<td style="padding:0.15rem 0.8rem;font-weight:600;color:var(--ea-ink)">{html.escape(str(got))}</td>'
                    f'<td style="padding:0.15rem 0.8rem;color:var(--ea-ink)">{word}</td>'
                    f'<td style="padding:0.15rem 0;color:var(--ea-ink-2)">{mark}</td></tr>')
    title = seen.word({"en": "How many tests each promise passes, by your code", "nl": "Hoeveel toetsen elke belofte haalt, volgens jouw code"})
    table = f'<table style="font-family:var(--ea-font-ui);font-size:0.9rem;border-collapse:collapse">{"".join(rows)}</table>'
    return yours.figure(title, table)


# The lesson is done when these are correct. "code-of-ethics" is the choice question in the lesson.
REQUIRED = ["passed", "notice", "code-of-ethics"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_RIGHT = passed(CHECK, OWNER, CONSEQUENCE)


def _too_low(got, text) -> bool:
    """A dictionary with the right promises whose totals are each at most the right one, and some lower: one or two
    of the three dictionaries left out."""
    return (isinstance(got, dict) and set(got) == set(_RIGHT)
            and all(isinstance(v, int) and not isinstance(v, bool) and v <= _RIGHT[p] for p, v in got.items())
            and got != _RIGHT)


def _line(name: str, values: dict) -> str:
    """`name = {...}` on one line, or over two inside the brackets when one line would be too wide for the box."""
    one = f"{name} = {values!r}".replace("'", '"')
    if len(one) <= 100:
        return one
    items = [f'"{k}": {v}' for k, v in values.items()]
    half = (len(items) + 1) // 2
    return f"{name} = {{" + ", ".join(items[:half]) + ",\n" + " " * (len(name) + 4) + ", ".join(items[half:]) + "}"


_LINES = (_line("check", CHECK), _line("owner", OWNER), _line("consequence", CONSEQUENCE))
_HEAD = ("# 1 when the promise has it, 0 when it does not\n" + "\n".join(_LINES)
         + "\n\npassed = {}\nfor promise in check:\n")
_START = _HEAD + "    ...\npassed\n"
_LINE = "    passed[promise] = check[promise] + owner[promise] + consequence[promise]\n"
_SOLUTION = _HEAD + _LINE + "passed"


def _p(body: str) -> str:
    return _HEAD + body + "passed"


EXERCISES = {
    "passed": CodeBox(
        name="passed",
        picture=_passed_picture,
        start=_START,
        solution=_SOLUTION,
        keep=[_LINES[0], _LINES[1]],
        # The check also runs the box as if the company had given one more promise a consequence.
        others=[{"consequence": Code('{"fair AI": 0, "bias test": 0, "outside test": 0, "ethics board": 0, "complaints": 1, "planet": 1}')}],
        first=[_DOTS],
        mistakes=[
            Mistake("dicts-added", error=lambda e: isinstance(e, TypeError) and "dict" in str(e),
                    en="Your code adds whole dictionaries. Add the values for this one promise: `check[promise]`, `owner[promise]` and `consequence[promise]`.",
                    nl="Je code telt hele dictionaries op. Tel de waarden voor deze ene belofte op: `check[promise]`, `owner[promise]` en `consequence[promise]`."),
            Mistake("appended", error=lambda e: isinstance(e, AttributeError) and "append" in str(e),
                    en="`passed` is a dictionary, and a dictionary has no `append`. Put a value under a key with `passed[promise] = ...`.",
                    nl="`passed` is een dictionary, en een dictionary heeft geen `append`. Zet een waarde onder een key met `passed[promise] = ...`."),
            Mistake("too-low", matches=_too_low,
                    en="Your last line gives `{got}`: some totals are too low, such as the one for `complaints`, which passes all three tests. Does your code add up all three dictionaries?",
                    nl="Je laatste regel geeft `{got}`: sommige totalen zijn te laag, zoals dat van `complaints`, dat alle drie de toetsen haalt. Telt je code alle drie de dictionaries op?"),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with one line that puts under `passed[promise]` the three tests of this promise added up.",
                      "nl": "Vervang de `...` door één regel die onder `passed[promise]` de drie toetsen van deze belofte opgeteld zet."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other consequences by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere gevolgen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `passed` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `passed` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does your loop put under each promise its three tests added up?",
                      "nl": "Je laatste regel geeft `{got}`. Zet je loop onder elke belofte haar drie toetsen opgeteld?"},
            "other": {"en": "Right for this company. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you add up the three dictionaries themselves?",
                      "nl": "Goed voor dit bedrijf. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Tel je de drie dictionaries zelf op?"},
            "other_error": {"en": "Right for this company. But with {where}, your code stops with an error. Do you add up the three dictionaries themselves?",
                            "nl": "Goed voor dit bedrijf. Maar met {where} stopt je code met een fout. Tel je de drie dictionaries zelf op?"},
            "correct": {"en": "`{got}`: fair AI and planet pass none of the three. Nobody would find out if they were broken, nobody answers for them, and nothing would happen: they are only words. The ethics board has an owner but no check or consequence; only the complaints promise passes all three. The check also tried other consequences.",
                        "nl": "`{got}`: fair AI en planet halen geen van de drie. Niemand zou merken als ze gebroken werden, niemand staat ervoor in, en er zou niets gebeuren: ze zijn alleen woorden. De ethische commissie heeft een eigenaar, maar geen controle of gevolg; alleen de belofte over klachten haalt alle drie. De check probeerde ook andere gevolgen."},
        },
    ),
    "notice": Sort(
        name="notice",
        groups={
            "nothing": {"en": "Costs nothing when broken", "nl": "Kost niets als ze gebroken wordt"},
            "notice": {"en": "Someone would notice", "nl": "Iemand zou het merken"},
        },
        items={
            "people": {"en": "**People first.** We put people first in everything we build.",
                       "nl": "**Mensen eerst.** We zetten mensen op de eerste plaats in alles wat we bouwen."},
            "rates": {"en": "**Error rates.** Every quarter we publish how often our model is wrong for each group of users.",
                      "nl": "**Foutpercentages.** Elk kwartaal publiceren we hoe vaak ons model zich vergist voor elke groep gebruikers."},
            "strive": {"en": "**We strive.** We strive for fairness in all our products.",
                       "nl": "**We streven.** We streven naar fairness in al onze producten."},
            "launch": {"en": "**No launch.** A model that fails the bias test does not launch until the head of the data team signs off a fix.",
                       "nl": "**Geen lancering.** Een model dat zakt voor de toets op bias, gaat pas live als het hoofd van het datateam een oplossing goedkeurt."},
            "secret": {"en": "**A board.** An ethics board advises us; its advice is not published and need not be followed.",
                       "nl": "**Een commissie.** Een ethische commissie adviseert ons; het advies wordt niet gepubliceerd en hoeft niet gevolgd te worden."},
            "appeal": {"en": "**Appeal.** Anyone can take a decision to a person, who must answer within 30 days.",
                       "nl": "**Bezwaar.** Iedereen kan met een beslissing naar een mens stappen, die binnen 30 dagen moet antwoorden."},
        },
        solution={"people": "nothing", "strive": "nothing", "secret": "nothing", "rates": "notice", "launch": "notice", "appeal": "notice"},
        nudge={
            "people": {"en": "How would anyone find out that this was broken?", "nl": "Hoe zou iemand ontdekken dat dit gebroken werd?"},
            "strive": {"en": "Striving cannot fail. What would you check?", "nl": "Streven kan niet mislukken. Wat zou je controleren?"},
            "secret": {"en": "The board has members. But who sees its advice, and what happens if it is ignored?", "nl": "De commissie heeft leden. Maar wie ziet het advies, en wat gebeurt er als het genegeerd wordt?"},
            "rates": {"en": "A published number can be compared, by anyone, every quarter.", "nl": "Een gepubliceerd getal kan iedereen elk kwartaal vergelijken."},
            "launch": {"en": "Is there a test, a person and something that happens?", "nl": "Is er een toets, een persoon en iets dat gebeurt?"},
            "appeal": {"en": "A person must answer within a set time. Would the user notice if nobody did?", "nl": "Een mens moet binnen een vaste termijn antwoorden. Zou de gebruiker merken als niemand dat deed?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the promises again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de beloften opnieuw uit."},
            "empty": {"en": "Put each promise with what happens when it is broken.",
                      "nl": "Zet elke belofte bij wat er gebeurt als ze gebroken wordt."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The promise *{item}* belongs on the other side.",
                      "nl": "De belofte *{item}* hoort aan de andere kant."},
            "correct": {"en": "All {n} are right. The promises someone would notice name something you can check: a published number, a test that stops a launch, a deadline for an answer. The others could be broken tomorrow without anyone finding out.",
                        "nl": "Alle {n} kloppen. De beloften die iemand zou merken, noemen iets wat je kunt controleren: een gepubliceerd getal, een toets die een lancering tegenhoudt, een termijn voor een antwoord. De andere kunnen morgen gebroken worden zonder dat iemand het ontdekt."},
        },
    ),
}

_SORTED = {"people": "nothing", "strive": "nothing", "secret": "nothing", "rates": "notice", "launch": "notice", "appeal": "notice"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("passed", _START, "empty", None),
    ("passed", _SOLUTION, "correct", None),
    ("passed", _p("    passed[promise] = consequence[promise] + owner[promise] + check[promise]\n"), "correct", None),
    ("passed", _p('    passed[promise] = {"fair AI": 0, "bias test": 2, "outside test": 1, "ethics board": 1, "complaints": 3, "planet": 0}[promise]\n'), "almost", "other-value"),
    ("passed", _p("    passed[promise] = check[promise] + owner[promise]\n"), "wrong", "too-low"),
    ("passed", _p("    passed[promise] = check[promise] + consequence[promise]\n"), "wrong", "too-low"),
    ("passed", _p("    passed[promise] = owner[promise]\n"), "wrong", "too-low"),
    ("passed", _p("    passed[promise] = check + owner + consequence\n"), "error", "dicts-added"),
    ("passed", _p("    passed.append(check[promise] + owner[promise] + consequence[promise])\n"), "error", "appended"),
    ("passed", _p("    passed[promise] = check[promise] + owner[promise] + 1\n"), "wrong", None),
    ("passed", _HEAD + "    ...\npassed = {}\npassed", "wrong", "dots-left"),
    ("notice", {}, "empty", None),
    ("notice", _SORTED, "correct", None),
    ("notice", {"people": "nothing", "rates": "notice"}, "wrong", "unfinished"),
    ("notice", {**_SORTED, "secret": "notice"}, "wrong", "wrong:secret"),
    ("notice", {**_SORTED, "appeal": "nothing"}, "wrong", "wrong:appeal"),
    ("notice", {**_SORTED, "strive": "notice"}, "wrong", "wrong:strive"),
]
