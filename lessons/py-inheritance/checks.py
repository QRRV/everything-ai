# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson py-inheritance: a subclass that overrides one method, and a subclass with its own
__init__ that calls the parent's with super(). Each is a class written in a box and tried in scenarios."""
import re

from everything_ai.check import ClassBox, Mistake

# A `...` that is still there, found before the code runs: a method that is still `...` gives back nothing,
# and that should not be explained as some other mistake.
_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_COLON = Mistake("colon-missing", syntax=r"^[ \t]*(class\b[^:\n]*|def\b[^:\n]*)$",
                 en="The line `{line}` must end with a colon, `:`. The colon tells Python that the indented lines below belong to it.",
                 nl="De regel `{line}` moet eindigen op een dubbele punt, `:`. Die vertelt Python dat de ingesprongen regels eronder erbij horen.")
_NO_SELF = Mistake("no-self", syntax=r"^\s+def\s+\w+\(\s*(\)|(?!self\b)\w)",
                   en="Every method gets the object itself as its first argument: in `{line}`, write `self` first between the parentheses.",
                   nl="Elke method krijgt het object zelf als eerste argument: zet in `{line}` `self` als eerste tussen de haakjes.")

# The lesson is done when these are correct. "which-switch" is the choice question in the lesson.
REQUIRED = ["greeter", "which-switch", "savings"]

_GREETER = ("class Greeter:\n    def __init__(self, name):\n        self.name = name\n\n"
            "    def greeting(self):\n        return \"Hello\"\n\n"
            "    def greet(self):\n        return f\"{self.greeting()}, {self.name}!\"\n\n\n")
_DUTCH_START = _GREETER + "class DutchGreeter(Greeter):\n    ..."
_DUTCH = _GREETER + "class DutchGreeter(Greeter):\n    def greeting(self):\n        return \"Hallo\""
_DUTCH_COPIED = _GREETER + ("class DutchGreeter(Greeter):\n    def greet(self):\n        return f\"Hallo, {self.name}!\"")

_ACCOUNT = ("class Account:\n    def __init__(self, owner):\n        self.owner = owner\n        self.balance = 0\n\n"
            "    def deposit(self, amount):\n        self.balance = self.balance + amount\n\n\n")
_SAVINGS_START = _ACCOUNT + ("class Savings(Account):\n    def __init__(self, owner, rate):\n        ...\n        self.rate = rate\n\n"
                             "    def add_interest(self):\n        ...")
_SAVINGS = _ACCOUNT + ("class Savings(Account):\n    def __init__(self, owner, rate):\n        super().__init__(owner)\n        self.rate = rate\n\n"
                       "    def add_interest(self):\n        self.deposit(self.balance * self.rate)")

def _same_parent(parent: str):
    """The parent class is still in the box exactly as the course gave it, spaces aside."""
    return lambda got, text: " ".join(parent.split()) in " ".join(text.split())


_HAS_SUPER = r"\bsuper\s*\("

EXERCISES = {
    "greeter": ClassBox(
        name="greeter",
        start=_DUTCH_START,
        solution=_DUTCH,
        scenarios=[
            "g = DutchGreeter(\"Ada\")\ng.greet()",
            "DutchGreeter(\"Bo\").greeting()",
            "Greeter(\"Ada\").greet()",
            "g = DutchGreeter(\"Cy\")\ng.name",
        ],
        first=[_DOTS],
        mistakes=[
            _COLON, _NO_SELF,
            Mistake("own-init", syntax=r"(?s)class\s+DutchGreeter\b.*def\s+__init__(?!.*\bsuper\s*\()",
                    en="`DutchGreeter` has its own `__init__`, so the `__init__` of `Greeter` no longer runs by itself, and `self.name` is never set. Leave `__init__` out of `DutchGreeter`: then it gets `Greeter`'s.",
                    nl="`DutchGreeter` heeft een eigen `__init__`, dus draait de `__init__` van `Greeter` niet meer vanzelf, en wordt `self.name` nooit gezet. Laat `__init__` weg uit `DutchGreeter`: dan krijgt hij die van `Greeter`."),
            Mistake("no-parent", syntax=r"^class\s+DutchGreeter\s*(\(\s*\))?\s*:",
                    en="`DutchGreeter` is built on `Greeter` only when you say so: put the parent class between parentheses, `class DutchGreeter(Greeter):`. Then it gets `__init__` and `greet` from `Greeter`.",
                    nl="`DutchGreeter` bouwt alleen op `Greeter` voort als je dat zegt: zet de parent class tussen haakjes, `class DutchGreeter(Greeter):`. Dan krijgt hij `__init__` en `greet` van `Greeter`."),
            Mistake("prints", matches=lambda got, text: isinstance(got, str) and got.startswith("None"),
                    en="After these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`: your `greeting` gives back nothing, `None`. It uses `print`, or it has no `return`. Give back the word with `return`.",
                    nl="Na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`: je `greeting` geeft niets terug, `None`. Hij gebruikt `print`, of er staat geen `return` in. Geef het woord terug met `return`."),
            Mistake("parent-changed", matches=lambda got, text: not _same_parent(_GREETER)(got, text),
                    en="After these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. You changed `Greeter` itself, so every greeter says something else now. Leave `Greeter` as it was, and change only `DutchGreeter`.",
                    nl="Na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Je hebt `Greeter` zelf veranderd, dus zegt elke greeter nu iets anders. Laat `Greeter` zoals hij was, en verander alleen `DutchGreeter`."),
            Mistake("greet-copied", state="almost", wrong=_DUTCH_COPIED,
                    en="Right for the example. But your `DutchGreeter` writes `greet` again, and its `greeting` still says `{got}`. It already gets `greet` from `Greeter`, and that `greet` asks `self.greeting()` for the word. Override only `greeting`: then `greet` uses your version by itself.",
                    nl="Goed voor het voorbeeld. Maar je `DutchGreeter` schrijft `greet` opnieuw, en zijn `greeting` zegt nog steeds `{got}`. Hij krijgt `greet` al van `Greeter`, en die vraagt `self.greeting()` om het woord. Overschrijf alleen `greeting`: dan gebruikt `greet` vanzelf jouw versie."),
        ],
        form=[Mistake("init-by-hand", state="almost", matches=lambda got, text: re.search(r"(?s)class\s+DutchGreeter\b.*def\s+__init__(?!.*\bsuper\s*\()", text) is not None,
                      en="Right, but `DutchGreeter` does not need its own `__init__`: it gets `Greeter`'s. Leave it out, and override only what is different.",
                      nl="Goed, maar `DutchGreeter` heeft geen eigen `__init__` nodig: hij krijgt die van `Greeter`. Laat hem weg, en overschrijf alleen wat anders is.")],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` under `class DutchGreeter(Greeter):` with one method that makes `DutchGreeter` say Hallo. Step 2 of the recipe shows how.",
                      "nl": "Vervang de `...` onder `class DutchGreeter(Greeter):` door één method die `DutchGreeter` Hallo laat zeggen. Stap 2 van het recept laat zien hoe."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z) until the classes are back, or open the lesson again: an empty box then starts with the classes again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z) tot de classes terug zijn, of open de les opnieuw: een leeg vak begint dan weer met de classes."},
            "wrong": {"en": "After these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Which method of `Greeter` gives the word, and does `DutchGreeter` override it, indented under `class DutchGreeter(Greeter):`?",
                      "nl": "Na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Welke method van `Greeter` geeft het woord, en overschrijft `DutchGreeter` die, ingesprongen onder `class DutchGreeter(Greeter):`?"},
            "almost": {"en": "Right for the example. But after these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Did you change only `DutchGreeter`, and only its `greeting`?",
                       "nl": "Goed voor het voorbeeld. Maar na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Heb je alleen `DutchGreeter` veranderd, en daarvan alleen `greeting`?"},
            "correct": {"en": "Your `DutchGreeter` is right in all {n} cases: it says Hallo, it gets `__init__` and `greet` from `Greeter`, and `Greeter` itself still says Hello.",
                        "nl": "Je `DutchGreeter` klopt in alle {n} gevallen: hij zegt Hallo, hij krijgt `__init__` en `greet` van `Greeter`, en `Greeter` zelf zegt nog steeds Hello."},
        },
    ),
    "savings": ClassBox(
        name="savings",
        start=_SAVINGS_START,
        solution=_SAVINGS,
        scenarios=[
            "s = Savings(\"Ada\", 0.1)\ns.deposit(100)\ns.add_interest()\ns.balance",
            "s = Savings(\"Bo\", 0.5)\n[s.owner, s.balance, s.rate]",
            "s = Savings(\"Bo\", 0.5)\ns.deposit(10)\ns.add_interest()\ns.add_interest()\ns.balance",
            "a = Account(\"Cy\")\na.deposit(5)\na.balance",
        ],
        first=[_DOTS],
        mistakes=[
            _COLON, _NO_SELF,
            Mistake("no-parent-savings", syntax=r"^class\s+Savings\s*(\(\s*\))?\s*:",
                    en="`Savings` is built on `Account` only when you say so: put the parent class between parentheses, `class Savings(Account):`. Then it gets `__init__` and `deposit` from `Account`.",
                    nl="`Savings` bouwt alleen op `Account` voort als je dat zegt: zet de parent class tussen haakjes, `class Savings(Account):`. Dan krijgt hij `__init__` en `deposit` van `Account`."),
            Mistake("super-no-parens", syntax=r"\bsuper\s*\.\s*__init__",
                    en="`super` needs its parentheses: `super().__init__(...)`. Then it stands for the parent class of this object.",
                    nl="`super` heeft zijn haakjes nodig: `super().__init__(...)`. Dan staat het voor de parent class van dit object."),
            Mistake("super-not-called", syntax=r"^\s*super\s*\(\s*\)\s*(\.\s*__init__)?\s*$",
                    en="`super()` alone does nothing yet: call the `__init__` of the parent class with it, `super().__init__(...)`, and pass the owner on.",
                    nl="`super()` alleen doet nog niets: roep er de `__init__` van de parent class mee aan, `super().__init__(...)`, en geef de eigenaar door."),
            Mistake("super-self", syntax=r"super\s*\(\s*\)\s*\.\s*__init__\s*\(\s*self\b",
                    en="`super()` already passes the object along as `self`. Pass only the owner: `super().__init__(owner)`.",
                    nl="`super()` geeft het object zelf al mee als `self`. Geef alleen de eigenaar mee: `super().__init__(owner)`."),
            Mistake("super-two", syntax=r"super\s*\(\s*\)\s*\.\s*__init__\s*\([^)\n]*,",
                    en="The `__init__` of `Account` takes only the owner. Pass on only `owner`; set `self.rate` yourself, on the next line.",
                    nl="De `__init__` van `Account` neemt alleen de eigenaar. Geef alleen `owner` door; zet `self.rate` zelf, op de regel daarna."),
            Mistake("super-empty", syntax=r"super\s*\(\s*\)\s*\.\s*__init__\s*\(\s*\)",
                    en="The `__init__` of `Account` needs the owner: pass it on, `super().__init__(owner)`.",
                    nl="De `__init__` van `Account` heeft de eigenaar nodig: geef die door, `super().__init__(owner)`."),
            Mistake("no-super", syntax=r"(?s)\A(?!.*\bsuper\s*\()(?!.*Account\s*\.\s*__init__\s*\()(?!(?:.*self\.balance\s*=\s*0\b){2}).*",
                    en="`Savings` has its own `__init__`, so the `__init__` of `Account` no longer runs by itself, and what it sets up, such as `balance`, is missing. Call it first, in the `__init__` of `Savings`, with `super()`.",
                    nl="`Savings` heeft een eigen `__init__`, dus draait de `__init__` van `Account` niet meer vanzelf, en ontbreekt wat die klaarzet, zoals `balance`. Roep hem eerst aan, in de `__init__` van `Savings`, met `super()`."),
            Mistake("no-self-value", syntax=r"^(?!\s*(def\b|self\.rate\s*=\s*rate\s*$))\s+[^\n#]*(?<![\w.])(balance|rate)\b",
                    en="Inside a method, `balance` or `rate` without `self.` in front is not the object's own value. Write `self.balance` and `self.rate`.",
                    nl="In een method is `balance` of `rate` zonder `self.` ervoor niet de eigen waarde van het object. Schrijf `self.balance` en `self.rate`."),
            Mistake("interest-only", wrong=_SAVINGS.replace("self.deposit(self.balance * self.rate)", "self.balance = self.balance * self.rate"),
                    en="After these lines:\n\n```python\n{scenario}\n```\n\nthe balance is `{got}`, and it should be `{want}`: `add_interest` sets the balance to the interest alone. Add the interest to what is already there.",
                    nl="Na deze regels:\n\n```python\n{scenario}\n```\n\nis het saldo `{got}`, en dat moet `{want}` zijn: `add_interest` zet het saldo op alleen de rente. Tel de rente op bij wat er al staat."),
            Mistake("parent-changed", matches=lambda got, text: not _same_parent(_ACCOUNT)(got, text),
                    en="After these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. You changed `Account` itself. Leave `Account` as it was, and change only `Savings`.",
                    nl="Na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Je hebt `Account` zelf veranderd. Laat `Account` zoals hij was, en verander alleen `Savings`."),
        ],
        form=[
            Mistake("parent-by-name", state="almost", matches=lambda got, text: re.search(r"Account\s*\.\s*__init__\s*\(", text) is not None,
                    en="Right: `Account.__init__(self, owner)` works too. But `super().__init__(owner)` does it without naming `Account`, and passes `self` for you. Use `super()`, as the exercise asks.",
                    nl="Goed: `Account.__init__(self, owner)` werkt ook. Maar `super().__init__(owner)` doet het zonder `Account` te noemen, en geeft `self` voor je mee. Gebruik `super()`, zoals de oefening vraagt."),
            Mistake("by-hand", state="almost", matches=lambda got, text: re.search(_HAS_SUPER, text) is None,
                    en="Right, but you wrote the lines of `Account`'s `__init__` again. `super().__init__(owner)` runs them for you, so a later change in `Account` counts for `Savings` too. Use it in the `__init__` of `Savings`.",
                    nl="Goed, maar je schrijft de regels van de `__init__` van `Account` opnieuw. `super().__init__(owner)` voert ze voor je uit, zodat een latere verandering in `Account` ook voor `Savings` telt. Gebruik het in de `__init__` van `Savings`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: in `__init__`, call the `__init__` of `Account` with `super()`; in `add_interest`, add the interest to the balance.",
                      "nl": "Vervang de twee `...`: roep in `__init__` met `super()` de `__init__` van `Account` aan; tel in `add_interest` de rente op bij het saldo."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z) until the classes are back, or open the lesson again: an empty box then starts with the classes again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z) tot de classes terug zijn, of open de les opnieuw: een leeg vak begint dan weer met de classes."},
            "wrong": {"en": "After these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: 100, plus 0.1 times 100 in interest. Does `add_interest` add `self.balance * self.rate` to the balance?",
                      "nl": "Na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: 100, plus 0,1 keer 100 aan rente. Telt `add_interest` `self.balance * self.rate` op bij het saldo?"},
            "almost": {"en": "Right for the example. But after these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Does your code use `owner`, `self.balance` and `self.rate`, and not the numbers or the name of the example?",
                       "nl": "Goed voor het voorbeeld. Maar na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Gebruikt je code `owner`, `self.balance` en `self.rate`, en niet de getallen of de naam van het voorbeeld?"},
            "correct": {"en": "Your `Savings` is right in all {n} cases: `super()` gives it an owner and a balance, `deposit` comes from `Account`, and the interest is added.",
                        "nl": "Je `Savings` klopt in alle {n} gevallen: `super()` geeft hem een eigenaar en een saldo, `deposit` komt van `Account`, en de rente wordt erbij opgeteld."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("greeter", _DUTCH_START, "empty", None),
    ("greeter", _DUTCH, "correct", None),
    ("greeter", _DUTCH.replace("return \"Hallo\"", "return 'Hallo'"), "correct", None),
    ("greeter", _DUTCH_COPIED, "almost", "greet-copied"),
    ("greeter", _DUTCH.replace("class DutchGreeter(Greeter):", "class DutchGreeter:"), "error", "no-parent"),
    ("greeter", _DUTCH.replace("def greeting(self):\n        return \"Hallo\"", "def greeting():\n        return \"Hallo\""), "error", "no-self"),
    ("greeter", _GREETER.replace("return \"Hello\"", "return \"Hallo\"") + "class DutchGreeter(Greeter):\n    def greeting(self):\n        return \"Hallo\"", "wrong", "parent-changed"),
    ("greeter", _DUTCH.replace("    def greeting(self):\n        return \"Hallo\"", "def greeting(self):\n    return \"Hallo\""), "error", None),
    ("greeter", _DUTCH.replace("return \"Hallo\"", "return \"Hallo, \""), "wrong", None),
    ("greeter", _DUTCH.replace("return \"Hallo\"", "print(\"Hallo\")"), "wrong", "prints"),
    ("greeter", _DUTCH.replace("class DutchGreeter(Greeter):", "class DutchGreeter():"), "error", "no-parent"),
    ("greeter", _DUTCH.replace("class DutchGreeter(Greeter):\n", "class DutchGreeter(Greeter):\n    def __init__(self, name):\n        self.lang = \"nl\"\n\n"), "error", "own-init"),
    ("greeter", _DUTCH.replace("class DutchGreeter(Greeter):\n", "class DutchGreeter(Greeter):\n    def __init__(self, name):\n        self.name = name\n\n"), "almost", "init-by-hand"),
    ("greeter", _DUTCH.replace("class DutchGreeter(Greeter):\n", "class DutchGreeter(Greeter):\n    def __init__(self, name):\n        super().__init__(name)\n\n"), "correct", None),
    ("savings", _SAVINGS_START, "empty", None),
    ("savings", "", "empty", None),
    ("savings", _SAVINGS, "correct", None),
    ("savings", _SAVINGS.replace("self.deposit(self.balance * self.rate)", "self.balance = self.balance + self.balance * self.rate"), "correct", None),
    ("savings", _SAVINGS.replace("super().__init__(owner)", "self.owner = owner\n        self.balance = 0"), "almost", "by-hand"),
    ("savings", _SAVINGS.replace("super().__init__(owner)", "self.owner = owner"), "error", "no-super"),
    ("savings", _SAVINGS.replace("        super().__init__(owner)\n", ""), "error", "no-super"),
    ("savings", _SAVINGS.replace("super().__init__(owner)", "super().__init__(\"Ada\")"), "almost", None),
    ("savings", _SAVINGS.replace("super().__init__(owner)", "Account.__init__(self, owner)"), "almost", "parent-by-name"),
    ("savings", _SAVINGS.replace("super().__init__(owner)", "super(Savings, self).__init__(owner)"), "correct", None),
    ("savings", _SAVINGS.replace("super().__init__(owner)", "super().__init__(owner, rate)"), "error", "super-two"),
    ("savings", _SAVINGS.replace("super().__init__(owner)", "super.__init__(owner)"), "error", "super-no-parens"),
    ("savings", _SAVINGS.replace("super().__init__(owner)", "super()"), "error", "super-not-called"),
    ("savings", _SAVINGS.replace("self.deposit(self.balance * self.rate)", "self.deposit(balance * rate)"), "error", "no-self-value"),
    ("savings", _SAVINGS.replace("self.deposit(self.balance * self.rate)", "self.deposit(self.balance * 0.1)"), "almost", None),
    ("savings", _SAVINGS.replace("class Savings(Account):", "class Savings:"), "error", "no-parent-savings"),
    ("savings", _SAVINGS.replace("self.balance = 0", "self.balance = 100", 1), "wrong", "parent-changed"),
    ("savings", _SAVINGS.replace("super().__init__(owner)", "self.owner = owner\n        self.balance = 0").replace("self.rate = rate", "self.rate = rat"), "error", None),
    ("savings", _SAVINGS.replace("super().__init__(owner)", "super().__init__(self, owner)"), "error", "super-self"),
    ("savings", _SAVINGS.replace("super().__init__(owner)", "super().__init__()"), "error", "super-empty"),
    ("savings", _SAVINGS.replace("self.deposit(self.balance * self.rate)", "self.balance = self.balance * self.rate"), "wrong", "interest-only"),
    ("savings", _SAVINGS.replace("self.deposit(self.balance * self.rate)", "self.balance = self.balance + self.rate"), "wrong", None),
    ("savings", _SAVINGS.replace("self.deposit(self.balance * self.rate)", "..."), "wrong", "dots-left"),
]
