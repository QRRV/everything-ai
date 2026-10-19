# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson py-terminal: a short script written in a box, checked by what it prints, and saved
whole in the learner's project (work/projects/scripts/countdown.py) so they can run it from the terminal.
And the script the lesson gives (project/scripts/dice.py), shown in the lesson."""
import re
from pathlib import Path

from everything_ai.check import CodeBox, Mistake

# The script the course puts in the learner's project folder, without its licence lines, as the lesson shows it.
DICE = "\n".join(line for line in (Path(__file__).parent / "project" / "scripts" / "dice.py").read_text(encoding="utf-8").splitlines()
                 if not line.startswith("# SPDX")).strip()


def _lines(text) -> list[str]:
    return [line.rstrip() for line in str(text).rstrip("\n").splitlines()]


def same_lines(a, b) -> bool:
    return isinstance(a, str) and isinstance(b, str) and _lines(a) == _lines(b)


# The lesson is done when these are correct. "which-command" and "dice-result" are choice questions in the lesson.
REQUIRED = ["which-command", "dice-result", "countdown"]

_START = "# countdown.py: counts down from 3, then says go\n...\n"
_SOLUTION = "# countdown.py: counts down from 3, then says go\nfor i in range(3):\n    print(3 - i)\nprint(\"go\")"

EXERCISES = {
    "countdown": CodeBox(
        name="countdown",
        start=_START,
        solution=_SOLUTION,
        outcome="stdout",
        expect="3\n2\n1\ngo\n",
        compare=same_lines,
        save_as=("scripts", "countdown"),
        save_whole=True,
        first=[Mistake("dots-left", matches=lambda got, text: re.search(r"^\s*\.\.\.\s*$", text, re.MULTILINE) is not None,
                       en="There is still a `...` in your script. Replace it with your own lines.",
                       nl="Er staat nog een `...` in je script. Vervang die door je eigen regels.")],
        mistakes=[
            Mistake("go-in-loop", matches=lambda got, text: _lines(got)[:2] == ["3", "go"],
                    en="Your script prints `go` after every number: the `print(\"go\")` is indented under the loop. Put it at the left edge, after the loop, so that it runs once.",
                    nl="Je script print `go` na elk getal: de `print(\"go\")` is ingesprongen onder de loop. Zet hem helemaal links, na de loop, zodat hij één keer draait."),
            Mistake("counts-up", matches=lambda got, text: _lines(got)[:3] in (["0", "1", "2"], ["1", "2", "3"]),
                    en="Your script counts up, and it should count down: 3, then 2, then 1. With `for i in range(3):`, `3 - i` gives 3, 2 and 1.",
                    nl="Je script telt op, en het moet aftellen: 3, dan 2, dan 1. Met `for i in range(3):` geeft `3 - i` 3, 2 en 1."),
            Mistake("one-too-many", matches=lambda got, text: _lines(got)[:4] == ["3", "2", "1", "0"],
                    en="Your script prints a `0` too: the loop goes round one time too many. It should go round 3 times, for 3, 2 and 1: `range(3)`.",
                    nl="Je script print ook een `0`: de loop gaat één keer te veel rond. Hij moet 3 keer rondgaan, voor 3, 2 en 1: `range(3)`."),
            Mistake("one-lower", matches=lambda got, text: _lines(got)[:3] == ["2", "1", "0"],
                    en="Your script counts down from 2: 2, 1, 0. Each number is 1 too low. With `i` = 0, 1 and 2, which sum gives 3, 2 and 1?",
                    nl="Je script telt af vanaf 2: 2, 1, 0. Elk getal is 1 te laag. Met `i` = 0, 1 en 2: welke som geeft 3, 2 en 1?"),
            Mistake("empty-line", matches=lambda got, text: "" in _lines(got),
                    en="Your script prints an empty line too. Each `print` prints one line, and a `print()` with nothing in it prints an empty one. Leave it out.",
                    nl="Je script print ook een lege regel. Elke `print` print één regel, en een `print()` zonder iets erin print een lege. Laat die weg."),
            Mistake("go-spelled", state="almost",
                    matches=lambda got, text: _lines(got)[:3] == ["3", "2", "1"] and len(_lines(got)) == 4
                    and _lines(got)[3].strip().lower().strip("!.") == "go",
                    en="Almost: the last line should be exactly `go`, in small letters, with nothing after it.",
                    nl="Bijna: de laatste regel moet precies `go` zijn, in kleine letters, met niets erachter."),
            Mistake("terminal-command", syntax=r"^\s*(\$\s*)?(uv\s+run|cd\s|python3?\s)",
                    en="That is a command for the terminal, not Python: type it in your second terminal. In the box goes the script itself, in Python.",
                    nl="Dat is een commando voor de terminal, geen Python: typ het in je tweede terminal. In het vak komt het script zelf, in Python."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a script that prints 3, 2 and 1, one per line, and then `go`.",
                      "nl": "Vervang de `...` door een script dat 3, 2 en 1 print, één per regel, en daarna `go`."},
            "cleared": {"en": "There is no code in the box any more. Write a script that prints 3, 2 and 1, one per line, and then `go`.",
                        "nl": "Er staat geen code meer in het vak. Schrijf een script dat 3, 2 en 1 print, één per regel, en daarna `go`."},
            "wrong": {"en": "Your script prints:\n\n```\n{got}\n```\n\nand it should print `3`, `2`, `1` and `go`, one per line. Compare it line by line with 3, 2, 1, go: which line is different?",
                      "nl": "Je script print dit:\n\n```\n{got}\n```\n\nen het moet `3`, `2`, `1` en `go` printen, één per regel. Vergelijk het regel voor regel met 3, 2, 1, go: welke regel is anders?"},
            "nothing": {"en": "Your script prints nothing. A script shows only what it prints: use `print` for each line.",
                        "nl": "Je script print niets. Een script laat alleen zien wat het print: gebruik `print` voor elke regel."},
            "correct": {"en": "Your script is right: 3, 2, 1, go. The check saved it whole, as `{saved}`. Now run it in your terminal, from the course folder: `uv run python {saved}`.",
                        "nl": "Je script klopt: 3, 2, 1, go. De check heeft het in zijn geheel bewaard, als `{saved}`. Voer het nu uit in je terminal, vanuit de map van de course: `uv run python {saved}`."},
            "kept": {"en": "Your script is right. You changed `{saved}` yourself, so the check leaves that file as it is. Run it in your terminal: `uv run python {saved}`.",
                     "nl": "Je script klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Voer het uit in je terminal: `uv run python {saved}`."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("countdown", _START, "empty", None),
    ("countdown", _SOLUTION, "correct", None),
    ("countdown", "print(3)\nprint(2)\nprint(1)\nprint(\"go\")", "correct", None),
    ("countdown", "for i in range(3):\n    print(3 - i)\n    print(\"go\")", "wrong", "go-in-loop"),
    ("countdown", "for i in range(3):\n    print(i)\nprint(\"go\")", "wrong", "counts-up"),
    ("countdown", "for i in range(3):\n    print(3 - i)", "wrong", None),
    ("countdown", "for i in range(3):\n    print(3 - i)\nprint(\"go\")\n...", "wrong", "dots-left"),
    ("countdown", "for i in range(3):\n    print(f\"{3 - i}...\")\nprint(\"go\")", "wrong", None),
    ("countdown", "for i in range(1, 4):\n    print(i)\nprint(\"go\")", "wrong", "counts-up"),
    ("countdown", "for i in range(4):\n    print(3 - i)\nprint(\"go\")", "wrong", "one-too-many"),
    ("countdown", "for i in range(3):\n    print(2 - i)\nprint(\"go\")", "wrong", "one-lower"),
    ("countdown", "for i in range(3):\n    print(3 - i)\nprint()\nprint(\"go\")", "wrong", "empty-line"),
    ("countdown", "for i in range(3):\n    print(3 - i)\nprint(\"Go!\")", "almost", "go-spelled"),
    ("countdown", "uv run python work/projects/scripts/countdown.py", "error", "terminal-command"),
    ("countdown", "def countdown():\n    for i in range(3):\n        print(3 - i)\n    print(\"go\")", "wrong", "nothing"),
]
