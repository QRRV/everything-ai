# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson py-comprehensions: a new list in one line, with and without an if. And the parts in the picture."""
import ast
import re

from everything_ai.check import CodeBox, Mistake

# The picture: a loop that builds a list, and the same list in one line. The marks tie the parts together.
PICTURE_LOOP = [("above = []", ""), ("for t in temps:", "2"), ("    if t > 20:", "3"), ("        above.append(t - 20)", "1")]
PICTURE_LINE = [("above = [", ""), ("t - 20", "1"), (" ", ""), ("for t in temps", "2"), (" ", ""), ("if t > 20", "3"), ("]", "")]


def _lines(text) -> list[str]:
    return [line.rstrip() for line in str(text).rstrip("\n").splitlines()]


def same_lines(a, b) -> bool:
    return isinstance(a, str) and isinstance(b, str) and _lines(a) == _lines(b)


def _shows(*texts):
    """The program showed exactly one of these lines."""
    return lambda got, text: _lines(got) in [[t] for t in texts]


def _no_list_comprehension(got, text) -> bool:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    return not any(isinstance(node, ast.ListComp) for node in ast.walk(tree))


_DOTS = [Mistake("dots-left", syntax=r"^[^#\n]*\.\.\.",
                 en="There is still a `...` in your program. Replace it with your own code.",
                 nl="Er staat nog een `...` in je programma. Vervang die door je eigen code."),
         Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                 en="There is still a `...` in your program. Replace it with your own code.",
                 nl="Er staat nog een `...` in je programma. Vervang die door je eigen code.")]
_CURLY = Mistake("curly-quote", syntax=r"[“”‘’„]",
                 en="Your program has a curly quote, like `“`. Python only reads straight quotes, like `\"`: type the quotes again in the box.",
                 nl="Je programma heeft een gekruld aanhalingsteken, zoals `“`. Python leest alleen rechte, zoals `\"`: typ de aanhalingstekens opnieuw in het vak.")
_FOR_FIRST = Mistake("for-first", syntax=r"\[\s*for\b",
                     en="In a comprehension, what goes in the new list comes first, and `for` after it: `[... for ... in ...]`.",
                     nl="In een comprehension komt eerst wat er in de nieuwe lijst komt, en daarna `for`: `[... for ... in ...]`.")
_COLON = Mistake("colon", syntax=r"\[[^\n]*\bfor\b[^\n]*:",
                 en="A comprehension has no colon after the `for` part or the `if`, as a `for` line or an `if` line has: they end at the closing square bracket.",
                 nl="Een comprehension heeft geen dubbele punt na het deel met `for` of na de `if`, zoals een regel met `for` of `if` die wel heeft: ze eindigen bij de sluitende vierkante haak.")
_CALLS = [Mistake("len-after", syntax=r"\.\s*len\s*\(",
                  en="`len` goes in front, with the value between round brackets: `len(word)`.",
                  nl="`len` staat vooraan, met de waarde tussen ronde haakjes: `len(word)`."),
          Mistake("len-square", syntax=r"\blen\s*\[",
                  en="`len` takes round brackets, not square ones: `len(word)`.",
                  nl="`len` krijgt ronde haakjes, geen vierkante: `len(word)`.")]
_ROUND = Mistake("round", matches=lambda got, text: got.strip().startswith("<generator"),
                 en="Your program shows `<generator object ...>`: parentheses do not make a list. Use square brackets: `[... for ... in ...]`.",
                 nl="Je programma laat `<generator object ...>` zien: ronde haakjes maken geen lijst. Gebruik vierkante haken: `[... for ... in ...]`.")
_TWICE = Mistake("twice", matches=lambda got, text: len(_lines(got)) == 2 and _lines(got)[0] == _lines(got)[1],
                 en="Your program shows the list twice. Keep one `print`: the one at the end.",
                 nl="Je programma laat de lijst twee keer zien. Houd één `print`: die aan het eind.")
_NOTHINGS = Mistake("print-inside", matches=lambda got, text: "None" in got,
                    en="Your list holds `None`, Python's word for nothing: a `print` inside a comprehension shows a value, but gives nothing to put in the list. Put the value itself in front of `for`, and print the whole list once.",
                    nl="Je lijst bevat `None`, het woord van Python voor niets: een `print` in een comprehension laat een waarde zien, maar geeft niets om in de lijst te zetten. Zet de waarde zelf vóór `for`, en laat de hele lijst één keer zien.")


def _one_line(name: str) -> list[Mistake]:
    return [Mistake("loop", matches=_no_list_comprehension, state="almost",
                    en=f"Right, but with a loop. This lesson is about building `{name}` in one line, with a comprehension: `[... for ... in ...]`. Try it that way.",
                    nl=f"Goed, maar met een loop. Deze les gaat erover dat je `{name}` in één regel bouwt, met een comprehension: `[... for ... in ...]`. Probeer het zo.")]


# The lesson is done when these are correct. "dict-comp" is the choice question in the lesson.
REQUIRED = ["lengths", "dict-comp", "passed"]

_SCORES = 'scores = {"Ada": 7.5, "Bo": 4.0, "Cas": 6.0, "Dex": 5.5}'

EXERCISES = {
    "lengths": CodeBox(
        name="lengths",
        start='words = ["horse", "ox", "bird"]\nlengths = ...\nprint(lengths)',
        solution='words = ["horse", "ox", "bird"]\nlengths = [len(word) for word in words]\nprint(lengths)',
        outcome="stdout",
        expect="[5, 2, 4]\n",
        compare=same_lines,
        others=[{"words": ["a", "bb", "dddd", "c"]}],
        keep=['words = ["horse", "ox", "bird"]'],
        form=_one_line("lengths"),
        mistakes=_DOTS + [
            _CURLY, _FOR_FIRST, _COLON, *_CALLS, _ROUND, _TWICE, _NOTHINGS,
            Mistake("whole-list", matches=_shows("[3, 3, 3]"),
                    en="Each item is `3`: the length of the whole list, `len(words)`. Use the name after `for`, which is one word: `len(word)`.",
                    nl="Elk item is `3`: de lengte van de hele lijst, `len(words)`. Gebruik de naam na `for`, en dat is één woord: `len(word)`."),
            Mistake("copy", matches=_shows("['horse', 'ox', 'bird']"),
                    en="Your new list has the words themselves. In front of `for` comes what goes in the new list: here, the length of each word.",
                    nl="Je nieuwe lijst bevat de woorden zelf. Vóór `for` komt wat er in de nieuwe lijst komt: hier de lengte van elk woord."),
            Mistake("count", matches=_shows("3"),
                    en="Your program shows `3`, the number of words. The task asks for a list: the length of each word.",
                    nl="Je programma laat `3` zien, het aantal woorden. De opdracht vraagt om een lijst: de lengte van elk woord."),
            Mistake("dict", matches=lambda got, text: got.strip().startswith("{") and ":" in got,
                    en="Curly braces with `key: value` in front make a dictionary. The task asks for a list: use square brackets, `[... for word in words]`.",
                    nl="Accolades met vooraan `key: waarde` maken een dictionary. De opdracht vraagt om een lijst: gebruik vierkante haken, `[... for word in words]`."),
            Mistake("set", matches=lambda got, text: got.strip().startswith("{"),
                    en="With curly braces and only one value in front of `for`, Python makes a set: each value once, in no fixed order. A list needs square brackets: `[... for word in words]`.",
                    nl="Met accolades en maar één waarde vóór `for` maakt Python een set: elke waarde één keer, zonder vaste volgorde. Een lijst heeft vierkante haken nodig: `[... for word in words]`."),
            Mistake("lines", matches=lambda got, text: len(_lines(got)) > 1 and all(line.strip().isdigit() for line in _lines(got)),
                    en="Your program shows each length on its own line. Build one list with a comprehension, and show it once with `print(lengths)`.",
                    nl="Je programma laat elke lengte op een eigen regel zien. Bouw één lijst met een comprehension, en laat die één keer zien met `print(lengths)`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a comprehension: in square brackets, the length of one word, `for`, and the list.",
                      "nl": "Vervang de `...` door een comprehension: tussen vierkante haken de lengte van één woord, `for`, en de lijst."},
            "cleared": {"en": "There is no code in the box any more. Type the three lines back: the line with `words`, `lengths = ...` and `print(lengths)`.",
                        "nl": "Er staat geen code meer in het vak. Typ de drie regels terug: de regel met `words`, `lengths = ...` en `print(lengths)`."},
            "wrong": {"en": "Your program shows:\n\n```\n{got}\n```\n\nThat is not yet `[5, 2, 4]`. Look at steps 2 and 3 of the recipe: what goes in front of `for`, and which list comes after `in`?",
                      "nl": "Je programma laat dit zien:\n\n```\n{got}\n```\n\nDat is nog niet `[5, 2, 4]`. Kijk naar stap 2 en 3 van het recept: wat komt er vóór `for`, en welke lijst komt na `in`?"},
            "other": {"en": "Right for these words. But with {where} instead, your program shows `{got}`, and it should show `{want}`. Did you type the numbers yourself? Let the comprehension work them out from `words`.",
                      "nl": "Goed voor deze woorden. Maar met {where} laat je programma `{got}` zien, en dat moet `{want}` zijn. Typte je de getallen zelf? Laat de comprehension ze uitrekenen uit `words`."},
            "correct": {"en": "Your program shows `[5, 2, 4]`: the length of each word, in one line. It also works for another list.",
                        "nl": "Je programma laat `[5, 2, 4]` zien: de lengte van elk woord, in één regel. Het werkt ook voor een andere lijst."},
        },
    ),
    "passed": CodeBox(
        name="passed",
        start=_SCORES + "\npassed = ...\nprint(passed)",
        solution=_SCORES + "\npassed = [name for name in scores if scores[name] >= 5.5]\nprint(passed)",
        outcome="stdout",
        expect="['Ada', 'Cas', 'Dex']\n",
        compare=same_lines,
        others=[{"scores": {"Eva": 9.0, "Fay": 5.2, "Gus": 5.5}}],
        keep=[_SCORES],
        form=_one_line("passed"),
        mistakes=_DOTS + [
            _CURLY, _FOR_FIRST, _COLON, _ROUND, _TWICE, _NOTHINGS,
            Mistake("decimal-comma", syntax=r"[<>=]=?\s*\d+,\d",
                    en="In code, a decimal number has a point, not a comma: `5.5`.",
                    nl="In code heeft een kommagetal een punt, geen komma: `5.5`."),
            Mistake("one-equals", syntax=r"\bif\b[^\n]*[^=!<>]=[^=]",
                    en="In a condition, one `=` does not compare: it gives a name a value. To keep scores of 5.5 or more, use `>=`.",
                    nl="In een voorwaarde vergelijkt één `=` niet: die geeft een naam een waarde. Wil je scores van 5,5 of meer houden, gebruik dan `>=`."),
            Mistake("scores-call", syntax=r"scores\s*\(",
                    en="A value in a dictionary is looked up with square brackets: `scores[name]`.",
                    nl="Een waarde in een dictionary zoek je op met vierkante haken: `scores[name]`."),
            Mistake("if-in-front", syntax=r"\[(?:(?!\bfor\b)[^\n])*\bif\b[^\n]*\bfor\b",
                    en="The `if` goes at the end, after the `for` part: `[... for ... in ... if ...]`.",
                    nl="De `if` komt aan het eind, na het deel met `for`: `[... for ... in ... if ...]`."),
            Mistake("name-compared", syntax=r"\bfor\s+(\w+)\s+in\s+scores\b[^\n]*\bif\s+\1\s*[<>=]",
                    en="A loop over a dictionary gives its keys, the names, such as `\"Ada\"`, not the scores. The score is the value under the key: `scores[...]`, with the name after `for` between the brackets.",
                    nl="Een loop over een dictionary geeft zijn keys, de namen, zoals `\"Ada\"`, niet de scores. De score is de waarde onder de key: `scores[...]`, met de naam na `for` tussen de haken."),
            Mistake("greater", matches=_shows("['Ada', 'Cas']"),
                    en="Dex is missing. Dex has exactly 5.5, and that is a pass too: use `>=`, not `>`.",
                    nl="Dex ontbreekt. Dex heeft precies 5,5, en dat is ook voldoende: gebruik `>=`, niet `>`."),
            Mistake("failed", matches=_shows("['Bo']"),
                    en="Your list has the name of the one who did not pass. Keep the names whose score is 5.5 or more.",
                    nl="Je lijst bevat de naam van wie een onvoldoende had. Houd de namen waarvan de score 5,5 of meer is."),
            Mistake("equal-only", matches=_shows("['Dex']"),
                    en="Your list has only Dex, with exactly 5.5: `==` keeps only scores equal to 5.5. A pass is 5.5 or more: `>=`.",
                    nl="Je lijst bevat alleen Dex, met precies 5,5: `==` houdt alleen scores die gelijk zijn aan 5,5. Voldoende is 5,5 of meer: `>=`."),
            Mistake("all", matches=_shows("['Ada', 'Bo', 'Cas', 'Dex']"),
                    en="Your list has every name, also Bo, with 4.0. Add an `if` at the end, so only names with 5.5 or more stay.",
                    nl="Je lijst bevat elke naam, ook Bo, met 4,0. Zet een `if` aan het eind, zodat alleen namen met 5,5 of meer blijven."),
            Mistake("scores", matches=_shows("[7.5, 6.0, 5.5]"),
                    en="Your list has the scores, not the names. In front of `for` comes what goes in the list: the name.",
                    nl="Je lijst bevat de scores, niet de namen. Vóór `for` komt wat er in de lijst komt: de naam."),
            Mistake("booleans", matches=lambda got, text: "True" in got or "False" in got,
                    en="Your list has `True` and `False`: the condition is in front of `for`. In front goes what goes in the list, the name; the condition comes at the end, after `if`.",
                    nl="Je lijst bevat `True` en `False`: de voorwaarde staat vóór `for`. Vooraan komt wat er in de lijst komt, de naam; de voorwaarde komt aan het eind, na `if`."),
            Mistake("dict", matches=lambda got, text: got.strip().startswith("{"),
                    en="Your program shows curly braces. The task asks for a list of names: use square brackets, `[... for name in scores if ...]`.",
                    nl="Je programma laat accolades zien. De opdracht vraagt om een lijst met namen: gebruik vierkante haken, `[... for name in scores if ...]`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a comprehension: the names in `scores`, with an `if` at the end for a score of 5.5 or more.",
                      "nl": "Vervang de `...` door een comprehension: de namen in `scores`, met een `if` aan het eind voor een score van 5,5 of meer."},
            "cleared": {"en": "There is no code in the box any more. Type the three lines back: the line with `scores`, `passed = ...` and `print(passed)`.",
                        "nl": "Er staat geen code meer in het vak. Typ de drie regels terug: de regel met `scores`, `passed = ...` en `print(passed)`."},
            "wrong": {"en": "Your program shows:\n\n```\n{got}\n```\n\nThat is not yet `['Ada', 'Cas', 'Dex']`. Look at step 4 of the recipe: which names stay, and what does the `if` compare?",
                      "nl": "Je programma laat dit zien:\n\n```\n{got}\n```\n\nDat is nog niet `['Ada', 'Cas', 'Dex']`. Kijk naar stap 4 van het recept: welke namen blijven, en wat vergelijkt de `if`?"},
            "other": {"en": "Right for these scores. But with {where} instead, your program shows `{got}`, and it should show `{want}`. Is your `if` right for every score, without names you typed yourself? A pass is 5.5 or more.",
                      "nl": "Goed voor deze scores. Maar met {where} laat je programma `{got}` zien, en dat moet `{want}` zijn. Klopt je `if` voor elke score, zonder namen die je zelf typte? Voldoende is 5,5 of meer."},
            "other_error": {"en": "Right for these scores. But with {where} instead, your program stops with an error. Did you type one of the names, such as `\"Dex\"`, yourself? Use the name after `for` and `scores[...]`, so it works for every dictionary.",
                            "nl": "Goed voor deze scores. Maar met {where} stopt je programma met een fout. Typte je zelf een van de namen, zoals `\"Dex\"`? Gebruik de naam na `for` en `scores[...]`, dan werkt het voor elke dictionary."},
            "correct": {"en": "Your program shows `['Ada', 'Cas', 'Dex']`: everyone with 5.5 or more, also Dex with exactly 5.5. It also works for other scores.",
                        "nl": "Je programma laat `['Ada', 'Cas', 'Dex']` zien: iedereen met 5,5 of meer, ook Dex met precies 5,5. Het werkt ook voor andere scores."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_W = 'words = ["horse", "ox", "bird"]\n'
_S = _SCORES + "\n"
CASES = [
    ("lengths", _W + "lengths = ...\nprint(lengths)", "empty", None),
    ("lengths", "", "empty", None),
    ("lengths", _W + "lengths = [len(word) for word in words]\nprint(lengths)", "correct", None),
    ("lengths", _W + "lengths = [len(w) for w in words]\nprint(lengths)", "correct", None),
    ("lengths", _W + "lengths = []\nfor word in words:\n    lengths.append(len(word))\nprint(lengths)", "almost", "loop"),
    ("lengths", _W + "lengths = [len(words) for word in words]\nprint(lengths)", "wrong", "whole-list"),
    ("lengths", _W + "lengths = [word for word in words]\nprint(lengths)", "wrong", "copy"),
    ("lengths", _W + "lengths = len(words)\nprint(lengths)", "wrong", "count"),
    ("lengths", _W + "lengths = (len(word) for word in words)\nprint(lengths)", "wrong", "round"),
    ("lengths", _W + "lengths = {len(word) for word in words}\nprint(lengths)", "wrong", "set"),
    ("lengths", _W + "lengths = [for word in words len(word)]\nprint(lengths)", "error", "for-first"),
    ("lengths", _W + "lengths = [len(word) for word in words:]\nprint(lengths)", "error", "colon"),
    ("lengths", _W + "lengths = [5, 2, 4]\nprint(lengths)", "almost", "other-value"),
    ("lengths", _W + "lengths = {word: len(word) for word in words}\nprint(lengths)", "wrong", "dict"),
    ("lengths", _W + "lengths = [len(word) for word in words]\nprint(lengths)\nprint(lengths)", "wrong", "twice"),
    ("lengths", _W + "lengths = [print(len(word)) for word in words]\nprint(lengths)", "wrong", "print-inside"),
    ("lengths", _W + "for word in words:\n    print(len(word))", "wrong", "lines"),
    ("lengths", _W + "lengths = [word.len() for word in words]\nprint(lengths)", "error", "len-after"),
    ("lengths", _W + "lengths = [len[word] for word in words]\nprint(lengths)", "error", "len-square"),
    ("passed", _S + "passed = ...\nprint(passed)", "empty", None),
    ("passed", _S + "passed = [name for name in scores if scores[name] >= 5.5]\nprint(passed)", "correct", None),
    ("passed", _S + "passed = [n for n in scores if scores[n] >= 5.5]\nprint(passed)", "correct", None),
    ("passed", _S + "passed = []\nfor name in scores:\n    if scores[name] >= 5.5:\n        passed.append(name)\nprint(passed)", "almost", "loop"),
    ("passed", _S + "passed = [name for name in scores if scores[name] > 5.5]\nprint(passed)", "wrong", "greater"),
    ("passed", _S + "passed = [name for name in scores if scores[name] < 5.5]\nprint(passed)", "wrong", "failed"),
    ("passed", _S + "passed = [name for name in scores]\nprint(passed)", "wrong", "all"),
    ("passed", _S + "passed = [scores[name] for name in scores if scores[name] >= 5.5]\nprint(passed)", "wrong", "scores"),
    ("passed", _S + "passed = [scores[name] >= 5.5 for name in scores]\nprint(passed)", "wrong", "booleans"),
    ("passed", _S + "passed = [name for name in scores if name >= 5.5]\nprint(passed)", "error", "name-compared"),
    ("passed", _S + "passed = [name if scores[name] >= 5.5 for name in scores]\nprint(passed)", "error", "if-in-front"),
    ("passed", _S + "passed = {name: scores[name] for name in scores if scores[name] >= 5.5}\nprint(passed)", "wrong", "dict"),
    ("passed", _S + "passed = ['Ada', 'Cas', 'Dex']\nprint(passed)", "almost", "other-value"),
    ("passed", _S + "passed = [name for name in scores if scores[name] >= 5.5:]\nprint(passed)", "error", "colon"),
    ("passed", _S + "passed = [name for name in scores: if scores[name] >= 5.5]\nprint(passed)", "error", "colon"),
    ("passed", _S + "passed = [score for score in scores if score >= 5.5]\nprint(passed)", "error", "name-compared"),
    ("passed", _S + "passed = [name for name in scores if scores[name] > 5]\nprint(passed)", "almost", "other-value"),
    ("passed", _S + "passed = [name for name in scores if scores[name] >= 5]\nprint(passed)", "almost", "other-value"),
    ("passed", _S + "passed = [name for name in scores if scores[name] >= 5,5]\nprint(passed)", "error", "decimal-comma"),
    ("passed", _S + "passed = [name for name in scores if scores[name] == 5.5]\nprint(passed)", "wrong", "equal-only"),
    ("passed", _S + "passed = [name for name in scores if scores(name) >= 5.5]\nprint(passed)", "error", "scores-call"),
    ("passed", _S + "passed = [name for name in scores if scores[name] = 5.5]\nprint(passed)", "error", "one-equals"),
]
