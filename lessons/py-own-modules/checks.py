# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson py-own-modules: a function saved as a module, and code that imports it.
And the module the lesson gives you (project/demo/shapes.py), shown in the lesson."""
import re
from pathlib import Path

from everything_ai import projects
from everything_ai.check import CodeBox, FunctionBox, Mistake, run_code, same_value

# The module the course puts in your project folder, without its licence lines, as the lesson shows it.
DEMO = "\n".join(line for line in (Path(__file__).parent / "project" / "demo" / "shapes.py").read_text(encoding="utf-8").splitlines()
                 if not line.startswith("# SPDX")).strip()


def _uses_own_mean(text) -> bool:
    """The code calls the learner's `mean`: `stats.mean(...)`, or `mean(...)` after `from tools.stats import mean`."""
    if re.search(r"\bstats\s*\.\s*mean\s*\(", text):
        return True
    return (re.search(r"^\s*from\s+tools\.stats\s+import\s+[^\n]*\bmean\b", text, re.MULTILINE) is not None
            and re.search(r"(?<![\w.])mean\s*\(", text) is not None)


def _lines(text) -> list[str]:
    return [line.rstrip() for line in str(text).rstrip("\n").splitlines()]


def same_lines(a, b) -> bool:
    return isinstance(a, str) and isinstance(b, str) and _lines(a) == _lines(b)


def _stats_file() -> Path:
    return projects.folder() / "tools" / "stats.py"


def _uses_tools(text) -> bool:
    return re.search(r"\btools\b", text) is not None


def _module_state() -> str:
    """How the learner's saved module is doing: "missing", "broken" (it stops with an error), "wrong" or "ok"."""
    if not _stats_file().exists():
        return "missing"
    run = run_code("from tools import stats\nstats.mean([4, 8, 6])")
    if run.error is not None or run.timeout:
        return "broken"
    return "ok" if same_value(run.value, 6.0) else "wrong"


_REDO = {"en": " Did you change that file? Delete it, and check exercise 1 again: its check then saves your `mean` anew.",
         "nl": " Heb je dat bestand veranderd? Verwijder het, en laat oefening 1 opnieuw nakijken: die check bewaart je `mean` dan opnieuw."}

_DOTS = Mistake("dots-left", syntax=r"^[^#\n]*\.\.\.",
                en="There is still a `...` in your code. Replace it with your own lines.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen regels.")

# The lesson is done when these are correct. "import-line" is the choice question in the lesson.
REQUIRED = ["mean", "import-line", "use-stats"]

EXERCISES = {
    "mean": FunctionBox(
        name="mean",
        function="mean",
        start="def mean(numbers):\n    ...\n",
        solution="def mean(numbers):\n    total = 0\n    for n in numbers:\n        total = total + n\n    return total / len(numbers)",
        visible=[([4, 8, 6],)],
        hidden=[([5],), ([1, 2],), ([-3, 3],), ([2.5, 3.5, 6],)],
        save_as=("tools", "stats"),
        python_values=True,
        mistakes=[
            _DOTS,
            Mistake("return-in-loop", wrong="def mean(numbers):\n    total = 0\n    for n in numbers:\n        total = total + n\n        return total / len(numbers)",
                    en="`{call}` gives `{got}`: your function stops after the first number. The `return` is indented under the loop, so it runs in the first round. Put it after the loop, at the same indent as `for`.",
                    nl="`{call}` geeft `{got}`: je functie stopt na het eerste getal. De `return` is ingesprongen onder de loop, dus hij draait in de eerste ronde. Zet hem na de loop, even ver ingesprongen als `for`."),
            Mistake("floor-divide", state="almost", wrong="def mean(numbers):\n    total = 0\n    for n in numbers:\n        total = total + n\n    return total // len(numbers)",
                    en="`{call}` gives `{got}`, and it should give `{want}`. `//` divides and drops what comes after the point. Divide with `/`.",
                    nl="`{call}` geeft `{got}`, en het zou `{want}` moeten zijn. `//` deelt en laat weg wat na de komma komt. Deel met `/`."),
            Mistake("no-divide", wrong="def mean(numbers):\n    total = 0\n    for n in numbers:\n        total = total + n\n    return total",
                    en="`{call}` gives `{got}`: the total, not the average. Divide it by the number of items, `len(numbers)`.",
                    nl="`{call}` geeft `{got}`: het totaal, niet het gemiddelde. Deel het door het aantal items, `len(numbers)`."),
            Mistake("fixed-three", state="almost", wrong="def mean(numbers):\n    total = 0\n    for n in numbers:\n        total = total + n\n    return total / 3",
                    en="`{call}` gives `{got}`: your function divides by 3, the length of the example. Divide by `len(numbers)`, so it works for every list.",
                    nl="`{call}` geeft `{got}`: je functie deelt door 3, de lengte van het voorbeeld. Deel door `len(numbers)`, dan werkt hij voor elke lijst."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the lines that work out the average: the total, divided by the number of items.",
                      "nl": "Vervang de `...` door de regels die het gemiddelde uitrekenen: het totaal, gedeeld door het aantal items."},
            "cleared": {"en": "There is no code in the box any more. Start again with `def mean(numbers):`, and under it, indented, your lines.",
                        "nl": "Er staat geen code meer in het vak. Begin opnieuw met `def mean(numbers):`, met daaronder, ingesprongen, je regels."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: (4 + 8 + 6) / 3. Add up the numbers in a loop, and after the loop, divide by `len(numbers)`.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: (4 + 8 + 6) / 3. Tel de getallen op in een loop, en deel na de loop door `len(numbers)`."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function use the list it gets, for every list?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Gebruikt je functie de lijst die hij krijgt, voor elke lijst?"},
            "correct": {"en": "Your function is right in all {n} cases. The check saved it in your project, as `{saved}`: exercise 3 imports it from there.",
                        "nl": "Je functie klopt in alle {n} gevallen. De check heeft hem in je project bewaard, als `{saved}`: oefening 3 haalt hem daar binnen."},
            "kept": {"en": "Your function is right in all {n} cases. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je functie klopt in alle {n} gevallen. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren? Verwijder het bestand en laat opnieuw nakijken."},
        },
    ),
    "use-stats": CodeBox(
        name="use-stats",
        start="from tools import stats\n\ntemps = [12, 15, 9]\n...",
        solution="from tools import stats\n\ntemps = [12, 15, 9]\nprint(stats.mean(temps))",
        outcome="stdout",
        expect="12.0\n",
        compare=same_lines,
        others=[{"temps": [1, 2]}],
        keep=["temps = [12, 15, 9]"],
        first=[
            Mistake("not-saved-yet", state="empty", matches=lambda got, text: _uses_tools(text) and _module_state() == "missing",
                    en="Your module is not in your project yet. The check of exercise 1 saves it, as `work/projects/tools/stats.py`, once your `mean` is right. Finish exercise 1 first; this check then runs again by itself.",
                    nl="Je module staat nog niet in je project. De check van oefening 1 bewaart hem, als `work/projects/tools/stats.py`, zodra je `mean` klopt. Maak eerst oefening 1 af; deze check draait daarna vanzelf opnieuw."),
            Mistake("module-broken", state="error", matches=lambda got, text: _uses_tools(text) and _module_state() == "broken",
                    en="Your module `work/projects/tools/stats.py` stops with an error when it is imported or used, so this box cannot use it." + _REDO["en"],
                    nl="Je module `work/projects/tools/stats.py` stopt met een fout als hij binnengehaald of gebruikt wordt, dus dit vak kan hem niet gebruiken." + _REDO["nl"]),
            Mistake("module-wrong", matches=lambda got, text: _uses_tools(text) and _module_state() == "wrong",
                    en="The `mean` in your module `work/projects/tools/stats.py` no longer gives the right average: for `[4, 8, 6]` it does not give `6.0`." + _REDO["en"],
                    nl="De `mean` in je module `work/projects/tools/stats.py` geeft niet meer het goede gemiddelde: voor `[4, 8, 6]` geeft hij geen `6.0`." + _REDO["nl"]),
        ],
        form=[Mistake("by-hand", matches=lambda got, text: not _uses_own_mean(text), state="almost",
                      en="Right, but you worked it out here again. The point is to use your own function from your project: `stats.mean(temps)`.",
                      nl="Goed, maar je rekent het hier opnieuw uit. Het gaat erom dat je je eigen functie uit je project gebruikt: `stats.mean(temps)`.")],
        mistakes=[
            _DOTS,
            Mistake("import-stats", syntax=r"^\s*import\s+stats\b",
                    en="`stats` is a module in the package `tools`, so Python cannot find it on its own. Import it from its package: `from tools import stats`.",
                    nl="`stats` is een module in het package `tools`, dus Python kan hem niet los vinden. Haal hem uit zijn package: `from tools import stats`."),
            Mistake("import-tools-stats", syntax=r"^\s*import\s+tools\s*\.\s*stats\b",
                    en="After `import tools.stats`, the module's whole name is `tools.stats`, so `stats` on its own is unknown. Import it with `from tools import stats`: then its name is `stats`.",
                    nl="Na `import tools.stats` is de hele naam van de module `tools.stats`, dus `stats` alleen is onbekend. Haal hem binnen met `from tools import stats`: dan heet hij `stats`."),
            Mistake("project-path", syntax=r"\bwork\s*\.\s*projects\b",
                    en="Python already looks in `work/projects/`, so that part is not in the name. Import the module from its package: `from tools import stats`.",
                    nl="Python kijkt al in `work/projects/`, dus dat deel hoort niet in de naam. Haal de module uit zijn package: `from tools import stats`."),
            Mistake("no-module-name", syntax=r"(?s)\A(?!.*^\s*from\s+tools\.stats\s+import\b).*?(?<![\w.])mean\s*\(",
                    en="`mean` lives in the module `stats`: write the module's name and a dot in front, `stats.mean(temps)`.",
                    nl="`mean` woont in de module `stats`: zet de naam van de module en een punt ervoor, `stats.mean(temps)`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a line that shows the average of `temps`, with `stats.mean`. Did you finish exercise 1? Its check saves `mean` in your project.",
                      "nl": "Vervang de `...` door een regel die het gemiddelde van `temps` laat zien, met `stats.mean`. Heb je oefening 1 af? De check daarvan bewaart `mean` in je project."},
            "cleared": {"en": "There is no code in the box any more. Type the lines back: `from tools import stats`, `temps = [12, 15, 9]` and `...`.",
                        "nl": "Er staat geen code meer in het vak. Typ de regels terug: `from tools import stats`, `temps = [12, 15, 9]` en `...`."},
            "wrong": {"en": "Your program shows:\n\n```\n{got}\n```\n\nThat is not yet `12.0`, the average of `temps`. Show `stats.mean(temps)` with `print`.",
                      "nl": "Je programma laat dit zien:\n\n```\n{got}\n```\n\nDat is nog niet `12.0`, het gemiddelde van `temps`. Laat `stats.mean(temps)` zien met `print`."},
            "other": {"en": "Right for these temperatures. But with {where} instead, your program shows `{got}`, and it should show `{want}`. Did you type the average, or the list, yourself? Use the name `temps`, and let `stats.mean` work it out.",
                      "nl": "Goed voor deze temperaturen. Maar met {where} laat je programma `{got}` zien, en dat moet `{want}` zijn. Heb je het gemiddelde of de lijst zelf getypt? Gebruik de naam `temps`, en laat `stats.mean` het uitrekenen."},
            "correct": {"en": "Your program shows `12.0`, worked out by your own function from your own module. It also works for other temperatures.",
                        "nl": "Je programma laat `12.0` zien, uitgerekend door je eigen functie uit je eigen module. Het werkt ook voor andere temperaturen."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
# The tests check every solution first, so the module is saved before these cases run; what use-stats says
# before that is tested in tests/test_projects.py.
_M = "def mean(numbers):\n"
_U = "from tools import stats\n\ntemps = [12, 15, 9]\n"
CASES = [
    ("mean", _M + "    ...\n", "empty", None),
    ("mean", _M + "    return sum(numbers) / len(numbers)", "correct", None),
    ("mean", _M + "    total = 0\n    for n in numbers:\n        total = total + n\n    return total", "wrong", "no-divide"),
    ("mean", _M + "    total = 0\n    for n in numbers:\n        total = total + n\n    return total / 3", "almost", "fixed-three"),
    ("mean", _M + "    total = 0\n    for n in numbers:\n        total = total + n\n        return total / len(numbers)", "wrong", "return-in-loop"),
    ("mean", _M + "    total = 0\n    for n in numbers:\n        total = total + n\n    return total // len(numbers)", "almost", "floor-divide"),
    ("mean", _M + "    total = 0\n    for n in numbers:\n        total = total + n\n    return total / len(numbers)", "correct", None),
    ("use-stats", _U + "...", "empty", None),
    ("use-stats", _U + "print(stats.mean(temps))", "correct", None),
    ("use-stats", _U + "average = stats.mean(temps)\nprint(average)", "correct", None),
    ("use-stats", _U + "print(sum(temps) / len(temps))", "almost", "by-hand"),
    ("use-stats", _U + "print(12.0)", "almost", "other-value"),
    ("use-stats", _U + "print(mean(temps))", "error", "no-module-name"),
    ("use-stats", "from tools.stats import mean\n\ntemps = [12, 15, 9]\nprint(mean(temps))", "correct", None),
    ("use-stats", "import tools.stats\n\ntemps = [12, 15, 9]\nprint(stats.mean(temps))", "error", "import-tools-stats"),
    ("use-stats", "from work.projects.tools import stats\n\ntemps = [12, 15, 9]\nprint(stats.mean(temps))", "error", "project-path"),
    ("use-stats", "import stats\ntemps = [12, 15, 9]\nprint(stats.mean(temps))", "error", "import-stats"),
    ("use-stats", _U + "stats.mean(temps)", "wrong", "nothing"),
]
