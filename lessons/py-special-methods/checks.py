# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson py-special-methods: a pair of numbers that adds and shows itself, and a line you can
call, each a class written in a box and tried in scenarios."""
import re

from everything_ai.check import ClassBox, Mistake, same_value

# A `...` that is still there, found before the code runs: a method that is still `...` gives back nothing,
# and that should not be explained as some other mistake.
_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your class. Replace every `...` with your own code.",
                nl="Er staat nog een `...` in je class. Vervang elke `...` door je eigen code.")

_SYNTAX = [
    Mistake("str-print", syntax=r"def\s+__str__\s*\(\s*self\s*\)\s*:\s*\n\s+print\s*\(",
            en="`__str__` shows its text with `print`, and gives nothing back. `print` and an f-string need the text itself from `__str__`: write `return` instead of `print`.",
            nl="`__str__` laat zijn tekst zien met `print`, en geeft niets terug. `print` en een f-string hebben de tekst zelf nodig van `__str__`: schrijf `return` in plaats van `print`."),
    Mistake("colon-missing", syntax=r"^[ \t]*(class\b[^:\n]*|def\b[^:\n]*)$",
            en="The line `{line}` must end with a colon, `:`. The colon tells Python that the indented lines below belong to it.",
            nl="De regel `{line}` moet eindigen op een dubbele punt, `:`. Die vertelt Python dat de ingesprongen regels eronder erbij horen."),
]

# The lesson is done when these are correct. "add-order" is the choice question in the lesson.
REQUIRED = ["vec", "add-order", "line"]

_VEC_START = ("class Vec:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n\n"
              "    def __add__(self, other):\n        ...\n\n    def __str__(self):\n        ...")
_VEC = _VEC_START.replace("    def __add__(self, other):\n        ...", "    def __add__(self, other):\n        return Vec(self.x + other.x, self.y + other.y)") \
                 .replace("    def __str__(self):\n        ...", "    def __str__(self):\n        return f\"({self.x}, {self.y})\"")
_VEC_CHANGES_SELF = _VEC.replace("        return Vec(self.x + other.x, self.y + other.y)",
                                 "        self.x = self.x + other.x\n        self.y = self.y + other.y\n        return self")

_LINE_START = ("class Line:\n    def __init__(self, w, b):\n        self.w = w\n        self.b = b\n\n"
               "    def __call__(self, x):\n        ...\n\n    def __str__(self):\n        ...")
_LINE = _LINE_START.replace("    def __call__(self, x):\n        ...", "    def __call__(self, x):\n        return self.w * x + self.b") \
                   .replace("    def __str__(self):\n        ...", "    def __str__(self):\n        return f\"y = {self.w}x + {self.b}\"")

# __add__ gives back a list, a pair in parentheses or a text instead of a new Vec.
_ADD_NOT_VEC = r"def\s+__add__\s*\([^)]*\)\s*:\s*\n\s+return\s+[\[\(\"'f]"
_ADD_NOT_VEC_TEXT = {
    "en": "`__add__` should give back a new `Vec`, made with `Vec(...)`, just like `_Money(...)` in the worked example. Then the result has its own `x` and `y`, and `print` uses its `__str__`.",
    "nl": "`__add__` moet een nieuwe `Vec` teruggeven, gemaakt met `Vec(...)`, net als `_Money(...)` in het uitgewerkte voorbeeld. Dan heeft de uitkomst een eigen `x` en `y`, en gebruikt `print` zijn `__str__`.",
}

# No __str__ (yet, or misspelled): Python shows its own text for an object, such as <__main__.Vec object at 0x7f...>.
_NO_STR = Mistake("no-str", matches=lambda got, text: " object at 0x" in str(got),
                  en="Your object shows up as Python's own text for an object: its class, and where it is in memory. Python finds no `__str__` in your class: check its name, with two underscores on each side, and give back the text with `return`.",
                  nl="Je object verschijnt als de eigen tekst van Python voor een object: zijn class, en waar het in het geheugen staat. Python vindt geen `__str__` in je class: kijk de naam na, met twee lage streepjes aan elke kant, en geef de tekst terug met `return`.")


def _line_parts(got) -> bool:
    """The first scenario of `line` gives a list of two: what `line(3)` gives back, and its text."""
    return isinstance(got, list) and len(got) == 2


EXERCISES = {
    "vec": ClassBox(
        name="vec",
        start=_VEC_START,
        solution=_VEC,
        scenarios=[
            "u = Vec(1, 2)\nv = Vec(3, 4)\nf\"{u + v}\"",
            "f\"{Vec(5, 0)}\"",
            "s = Vec(1, 2) + Vec(3, 4)\n[s.x, s.y]",
            "u = Vec(1, 2)\nv = Vec(3, 4)\ns = u + v\n[u.x, u.y, v.x, v.y]",
            "f\"{Vec(2, -1) + Vec(-2, 1)}\"",
        ],
        first=[_DOTS],
        mistakes=_SYNTAX + [
            Mistake("no-self", syntax=r"^\s+return\s+[^\n]*((?<![\w.])(x|y)\b(?!\s*\()|(?<![\w.])(a|b|u|v)\s*\.)",
                    en="Inside `__add__` and `__str__`, the two pairs are `self` and `other`: `self.x` is the `x` of the object itself, and `other.x` that of the other one. A bare `x`, or a name such as `a`, means nothing there.",
                    nl="In `__add__` en `__str__` heten de twee paren `self` en `other`: `self.x` is de `x` van het object zelf, en `other.x` die van het andere. Een losse `x`, of een naam zoals `a`, betekent daar niets."),
            Mistake("add-not-vec", syntax=_ADD_NOT_VEC, en=_ADD_NOT_VEC_TEXT["en"], nl=_ADD_NOT_VEC_TEXT["nl"]),
            _NO_STR,
            Mistake("add-no-return", matches=lambda got, text: got == "None",
                    en="`u + v` gives back nothing, so the f-string shows `None`, Python's word for nothing. Give back a new pair with `return Vec(...)`, and leave `self` as it is.",
                    nl="`u + v` geeft niets terug, dus laat de f-string `None` zien, het woord van Python voor niets. Geef een nieuw paar terug met `return Vec(...)`, en laat `self` zoals het is."),
            Mistake("add-changes-self", state="almost", wrong=_VEC_CHANGES_SELF,
                    en="After these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: `u` itself has changed. Your `__add__` changes `self`. Leave `self` as it is, and give back a new `Vec` made from both: `return Vec(...)`.",
                    nl="Na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: `u` zelf is veranderd. Je `__add__` verandert `self`. Laat `self` zoals het is, en geef een nieuwe `Vec` terug, gemaakt van allebei: `return Vec(...)`."),
            Mistake("add-not-vec", matches=lambda got, text: re.search(_ADD_NOT_VEC, text) is not None,
                    en=_ADD_NOT_VEC_TEXT["en"], nl=_ADD_NOT_VEC_TEXT["nl"]),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: in `__add__`, give back a new `Vec`; in `__str__`, give back the text, such as `(4, 6)`.",
                      "nl": "Vervang de twee `...`: geef in `__add__` een nieuwe `Vec` terug, en in `__str__` de tekst, zoals `(4, 6)`."},
            "cleared": {"en": "There is no code in the box any more. Start again with `class Vec:`, then the methods `__init__`, `__add__` and `__str__`.",
                        "nl": "Er staat geen code meer in het vak. Begin opnieuw met `class Vec:`, en dan de methods `__init__`, `__add__` en `__str__`."},
            "wrong": {"en": "After these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Does `__add__` add the `x` of `other` to the `x` of `self`, and the same for `y`? And does `__str__` give back the two numbers between parentheses, with a comma and a space?",
                      "nl": "Na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Telt `__add__` de `x` van `other` op bij de `x` van `self`, en net zo voor `y`? En geeft `__str__` de twee getallen tussen haakjes terug, met een komma en een spatie?"},
            "almost": {"en": "Right for the example. But after these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Does `__add__` leave `self` and `other` as they were, and does `__str__` use the object's own `x` and `y`?",
                       "nl": "Goed voor het voorbeeld. Maar na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Laat `__add__` `self` en `other` zoals ze waren, en gebruikt `__str__` de eigen `x` en `y` van het object?"},
            "correct": {"en": "Your `Vec` is right in all {n} cases: `+` gives a new pair, the old ones stay as they were, and `print` shows it as `(x, y)`.",
                        "nl": "Je `Vec` klopt in alle {n} gevallen: `+` geeft een nieuw paar, de oude blijven zoals ze waren, en `print` laat het zien als `(x, y)`."},
        },
    ),
    "line": ClassBox(
        name="line",
        start=_LINE_START,
        solution=_LINE,
        scenarios=[
            "line = Line(2, 1)\n[line(3), f\"{Line(2, 1)}\"]",
            "f\"{Line(3, 5)}\"",
            "line = Line(0.5, -1)\n[line(0), line(4), line(10)]",
            "g = Line(2, 1)\nh = Line(-1, 0)\n[g(1), h(1), g(1)]",
        ],
        first=[_DOTS],
        mistakes=_SYNTAX + [
            Mistake("self-x", syntax=r"self\s*\.\s*x\b",
                    en="`x` is not one of the line's own numbers: `__call__` gets it in the call, as in `line(3)`. Use `x` without `self.` in front.",
                    nl="`x` is geen eigen getal van de lijn: `__call__` krijgt hem bij de aanroep, zoals in `line(3)`. Gebruik `x` zonder `self.` ervoor."),
            Mistake("no-self", syntax=r"^\s+return\s+[^\n]*(?<![\w.{])(w|b)\b(?!\s*\()|\{\s*(w|b)\s*\}",
                    en="Inside a method, a bare `w` is not the line's own number: that is `self.w`. The same goes for `b`: `self.b`.",
                    nl="In een method is een losse `w` niet het eigen getal van de lijn: dat is `self.w`. Net zo voor `b`: `self.b`."),
            Mistake("call-no-x", syntax=r"def\s+__call__\s*\(\s*self\s*\)",
                    en="`line(3)` gives 3 to `__call__`, so it needs a name for it after `self`, as in `def __call__(self, x):`.",
                    nl="`line(3)` geeft 3 aan `__call__`, dus heeft die er een naam voor nodig na `self`, zoals in `def __call__(self, x):`."),
            _NO_STR,
            Mistake("call-nothing", matches=lambda got, text: _line_parts(got) and got[0] is None,
                    en="`__call__` gives back nothing, `None`: it has no `return`, or it shows the result with `print`. Put `return` in front of the calculation.",
                    nl="`__call__` geeft niets terug, `None`: er staat geen `return` in, of hij laat de uitkomst zien met `print`. Zet `return` voor de berekening."),
            Mistake("call-text", matches=lambda got, text: _line_parts(got) and isinstance(got[0], str),
                    en="`__call__` gives back text, and not a number. Give back the calculation itself: `w` times `x` plus `b`, without an f-string.",
                    nl="`__call__` geeft tekst terug, en geen getal. Geef de berekening zelf terug: `w` keer `x` plus `b`, zonder f-string."),
            Mistake("no-b", wrong=_LINE.replace("        return self.w * x + self.b", "        return self.w * x"),
                    en="After these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: the `+ b` is missing. Add the line's own `b` to `w` times `x`.",
                    nl="Na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: de `+ b` ontbreekt. Tel de eigen `b` van de lijn op bij `w` keer `x`."),
            Mistake("str-format", matches=lambda got, text: _line_parts(got) and same_value(got[0], 7),
                    en="After these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. `__call__` is right, but the text of `__str__` is not yet: check the spaces, `y = ` in front, and `x` right after `w`.",
                    nl="Na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. `__call__` klopt, maar de tekst van `__str__` nog niet: kijk de spaties na, `y = ` vooraan, en `x` direct na `w`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: in `__call__`, give back `w` times `x` plus `b`; in `__str__`, give back the text, such as `y = 2x + 1`.",
                      "nl": "Vervang de twee `...`: geef in `__call__` `w` keer `x` plus `b` terug, en in `__str__` de tekst, zoals `y = 2x + 1`."},
            "cleared": {"en": "There is no code in the box any more. Start again with `class Line:`, then the methods `__init__`, `__call__` and `__str__`.",
                        "nl": "Er staat geen code meer in het vak. Begin opnieuw met `class Line:`, en dan de methods `__init__`, `__call__` en `__str__`."},
            "wrong": {"en": "After these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Does `__call__` give back `w` times `x` plus `b`, and does `__str__` give back the text `y = 2x + 1`?",
                      "nl": "Na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Geeft `__call__` `w` keer `x` plus `b` terug, en geeft `__str__` de tekst `y = 2x + 1` terug?"},
            "almost": {"en": "Right for the example. But after these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Does your class use the line's own `self.w` and `self.b`, and the `x` it gets?",
                       "nl": "Goed voor het voorbeeld. Maar na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Gebruikt je class de eigen `self.w` en `self.b` van de lijn, en de `x` die hij krijgt?"},
            "correct": {"en": "Your `Line` is right in all {n} cases: you call it like a function, two lines each keep their own `w` and `b`, and `print` shows its formula.",
                        "nl": "Je `Line` klopt in alle {n} gevallen: je roept hem aan als een functie, twee lijnen houden elk hun eigen `w` en `b`, en `print` laat de formule zien."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_ADD = "        return Vec(self.x + other.x, self.y + other.y)"
_STR = 'return f"({self.x}, {self.y})"'
_CALL = "return self.w * x + self.b"
_LSTR = 'return f"y = {self.w}x + {self.b}"'
CASES = [
    ("vec", _VEC_START, "empty", None),
    ("vec", "", "empty", None),
    ("vec", _VEC, "correct", None),
    ("vec", _VEC.replace(_ADD, "        new = Vec(self.x + other.x, self.y + other.y)\n        return new"), "correct", None),
    ("vec", _VEC.replace(_ADD, "        Vec(self.x + other.x, self.y + other.y)"), "wrong", "add-no-return"),
    ("vec", _VEC.replace(_ADD, "        self.x = self.x + other.x\n        self.y = self.y + other.y"), "wrong", "add-no-return"),
    ("vec", _VEC_CHANGES_SELF, "almost", "add-changes-self"),
    ("vec", _VEC.replace(_ADD, "        other.x = self.x + other.x\n        other.y = self.y + other.y\n        return other"), "almost", None),
    ("vec", _VEC.replace(_ADD, "        return [self.x + other.x, self.y + other.y]"), "wrong", "add-not-vec"),
    ("vec", _VEC.replace(_ADD, "        return (self.x + other.x, self.y + other.y)"), "error", "add-not-vec"),
    ("vec", _VEC.replace(_ADD, "        return Vec(a.x + b.x, a.y + b.y)"), "error", "no-self"),
    ("vec", _VEC.replace(_STR, 'return f"({x}, {y})"'), "error", "no-self"),
    ("vec", _VEC.replace(_STR, 'print(f"({self.x}, {self.y})")'), "error", "str-print"),
    ("vec", _VEC.replace(_STR, 'return f"{self.x}, {self.y}"'), "wrong", None),
    ("vec", _VEC.replace(_STR, "return self.x"), "error", None),
    ("vec", _VEC.replace(_ADD, "        return Vec(self.x + self.y, other.x + other.y)"), "wrong", None),
    ("vec", _VEC.replace(_ADD, "        ..."), "wrong", "dots-left"),
    ("vec", _VEC.replace("class Vec:", "class Vec"), "error", "colon-missing"),
    ("vec", _VEC.replace("__str__", "__string__"), "wrong", "no-str"),
    ("vec", _VEC.replace("__add__", "__plus__"), "error", None),
    ("vec", _VEC.replace("__init__", "__innit__"), "error", None),
    ("line", _LINE_START, "empty", None),
    ("line", _LINE, "correct", None),
    ("line", _LINE.replace(_CALL, "return self.b + self.w * x"), "correct", None),
    ("line", _LINE.replace(_CALL, "print(self.w * x + self.b)"), "wrong", "call-nothing"),
    ("line", _LINE.replace(_CALL, "self.w * x + self.b"), "wrong", "call-nothing"),
    ("line", _LINE.replace(_CALL, 'return f"{self.w * x + self.b}"'), "wrong", "call-text"),
    ("line", _LINE.replace(_CALL, "return self.w * x"), "wrong", "no-b"),
    ("line", _LINE.replace(_CALL, "return w * x + b"), "error", "no-self"),
    ("line", _LINE.replace(_CALL, "return self.w * self.x + self.b"), "error", "self-x"),
    ("line", _LINE.replace("def __call__(self, x):", "def __call__(self):"), "error", "call-no-x"),
    ("line", _LINE.replace("__call__", "_call_"), "error", None),
    ("line", _LINE.replace(_LSTR, 'return f"{self.w}x + {self.b}"'), "wrong", "str-format"),
    ("line", _LINE.replace(_LSTR, 'return f"y={self.w}x+{self.b}"'), "wrong", "str-format"),
    ("line", _LINE.replace(_LSTR, 'return "y = 2x + 1"'), "almost", None),
    ("line", _LINE.replace("    def __str__(self):\n        " + _LSTR, ""), "wrong", "no-str"),
]
