# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-js-basics: two functions in JavaScript, which the learner's own browser runs in a web worker
(everything_ai.jsbox): the sum of two arrays pair by pair, and an async function that fetches a list of numbers as
JSON and adds them up; the picture of await, and the demo of the page and a web worker."""
import html
import re

from everything_ai import yours
from everything_ai.check import Mistake
from everything_ai.jsbox import UNDEFINED, Freeze, JsBox, assigned_const, missing_name, outcomes_picture, python_mistakes

# --- the picture of the idea: await ---------------------------------------------------------------------------------------

_AWAIT_WORDS = {
    "en": {"title": "What await does", "page": "the page", "server": "the server", "runs": "your code runs",
           "ask": "fetch(url)", "waits": "await: the page answers clicks and typing", "back": "the answer",
           "on": "your code goes on", "works": "works on the answer"},
    "nl": {"title": "Wat await doet", "page": "de pagina", "server": "de server", "runs": "je code draait",
           "ask": "fetch(url)", "waits": "await: de pagina reageert op klikken en typen", "back": "het antwoord",
           "on": "je code gaat verder", "works": "werkt aan het antwoord"},
}


def await_picture(lang: str) -> str:
    """Two lanes over time: the page runs the code, sends a request with fetch, keeps answering the user while the
    server works, and goes on with the answer when it comes."""
    w = _AWAIT_WORDS[lang]
    t = lambda x, y, s, c="var(--ea-ink)", a="start", size=12: (
        f'<text x="{x}" y="{y}" text-anchor="{a}" style="font-size:{size}px;font-family:var(--ea-font-ui);fill:{c}">{html.escape(s)}</text>')
    parts = [
        '<defs><marker id="ea-aw-head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
        '<path d="M0,0 L10,5 L0,10 z" style="fill:var(--ea-ink-2)"/></marker></defs>',
        t(0, 38, w["page"], "var(--ea-ink-2)"), t(0, 118, w["server"], "var(--ea-ink-2)"),
        '<line x1="90" y1="34" x2="700" y2="34" style="stroke:var(--ea-line);stroke-width:1"/>',
        '<line x1="90" y1="114" x2="700" y2="114" style="stroke:var(--ea-line);stroke-width:1"/>',
        '<rect x="96" y="24" width="150" height="20" rx="4" style="fill:var(--ea-inset);stroke:var(--ea-ink-2)"/>',
        t(171, 38, w["runs"], a="middle", size=11),
        '<rect x="246" y="24" width="300" height="20" rx="4" style="fill:none;stroke:var(--ea-ink-3);stroke-dasharray:4 3"/>',
        t(396, 38, w["waits"], "var(--ea-ink-2)", "middle", 11),
        '<rect x="546" y="24" width="150" height="20" rx="4" style="fill:var(--ea-inset);stroke:var(--ea-ink-2)"/>',
        t(621, 38, w["on"], a="middle", size=11),
        '<line x1="250" y1="46" x2="300" y2="104" style="stroke:var(--ea-ink-2);stroke-width:1.5" marker-end="url(#ea-aw-head)"/>',
        t(266, 80, w["ask"], "var(--ea-ink-2)", "end", 11),
        '<rect x="304" y="104" width="186" height="20" rx="4" style="fill:var(--ea-inset);stroke:var(--ea-ink-3)"/>',
        t(397, 118, w["works"], "var(--ea-ink-2)", "middle", 11),
        '<line x1="494" y1="104" x2="544" y2="48" style="stroke:var(--ea-ink-2);stroke-width:1.5" marker-end="url(#ea-aw-head)"/>',
        t(530, 80, w["back"], "var(--ea-ink-2)", "start", 11),
    ]
    svg = (f'<svg viewBox="0 0 710 136" role="img" style="width:100%;max-width:46rem;height:auto">'
           f'<title>{html.escape(w["title"])}</title>{"".join(parts)}</svg>')
    return yours.figure(w["title"], svg, alt=w["title"])


# --- exercise 1: dot ------------------------------------------------------------------------------------------------------

_START_DOT = '''function dot(a, b) {
  return 0;
}
'''
SOLUTION_DOT = '''function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}
'''
DOT_CASES = [([1, 2, 3], [4, 5, 6]), ([0.5, -1], [2, 3]), ([], []), ([2], [3]), ([1, 1, 1, 1], [0.25, 0.25, 0.25, 0.25])]

def _const() -> Mistake:
    return Mistake("const", error=assigned_const,
                   en="With `{call}`, your function stops: a name made with `const` keeps its value. Make a name that changes, such as a total or a counter, with `let`.",
                   nl="Met `{call}` stopt je functie: een naam die met `const` gemaakt is, houdt zijn waarde. Maak een naam die verandert, zoals een totaal of een teller, met `let`.")


def _python_names() -> list:
    return [
        Mistake("python-sum", error=missing_name("sum", "zip"),
                en="JavaScript has no `sum` or `zip` of its own: add up in a loop, with `total += ...`.",
                nl="JavaScript heeft geen eigen `sum` of `zip`: tel op in een loop, met `total += ...`."),
        Mistake("print", error=missing_name("print"),
                en="JavaScript has no `print`: it writes with `console.log`. But the check reads what your function gives back with `return`.",
                nl="JavaScript heeft geen `print`: het schrijft met `console.log`. Maar de check leest wat je functie teruggeeft met `return`."),
    ]


EXERCISES = {
    "dot": JsBox(
        name="dot",
        picture=outcomes_picture,
        function="dot",
        start=_START_DOT,
        solution=SOLUTION_DOT,
        visible=DOT_CASES[:1],
        hidden=DOT_CASES[1:],
        first=python_mistakes(_START_DOT),
        mistakes=[
            Mistake("len", error=missing_name("len"),
                    en="JavaScript has no `len`: an array has its length in `.length`, as in `a.length`.",
                    nl="JavaScript heeft geen `len`: een array heeft zijn lengte in `.length`, zoals in `a.length`."),
            Mistake("range", error=missing_name("range"),
                    en="JavaScript has no `range`: count with `for (let i = 0; i < a.length; i++) {{ ... }}`.",
                    nl="JavaScript heeft geen `range`: tel met `for (let i = 0; i < a.length; i++) {{ ... }}`."),
            *_python_names(),
            _const(),
            Mistake("last", wrong=SOLUTION_DOT.replace("total += ", "total = "),
                    en="`{call}` gives `{got}`: only the last pair. `total = ...` puts a new value in `total` each time; `total += ...` adds to it.",
                    nl="`{call}` geeft `{got}`: alleen het laatste paar. `total = ...` zet elke keer een nieuwe waarde in `total`; `total += ...` telt erbij op."),
            Mistake("past-end", wrong=SOLUTION_DOT.replace("i < a.length", "i <= a.length"),
                    en="`{call}` gives `NaN`, Not a Number: your loop goes one step past the end, where `a[i]` is `undefined`, and a sum with `undefined` is `NaN`. The last position is `a.length - 1`.",
                    nl="`{call}` geeft `NaN`, Not a Number: je loop gaat één stap voorbij het einde, waar `a[i]` gelijk is aan `undefined`, en een som met `undefined` is `NaN`. De laatste positie is `a.length - 1`."),
            Mistake("no-return", matches=lambda got, text: got == UNDEFINED,
                    en="`{call}` gives `undefined`: your function gives nothing back. End it with `return total;`.",
                    nl="`{call}` geeft `undefined`: je functie geeft niets terug. Sluit hem af met `return total;`."),
        ],
        texts={
            "empty": {"en": "`dot` gives 0 yet. Go through the positions of `a`, and add up `a[i] * b[i]`.",
                      "nl": "`dot` geeft nog 0. Loop door de posities van `a`, en tel `a[i] * b[i]` op."},
            "correct": {"en": "Your `dot` multiplies the arrays pair by pair and adds it all up, for all {n} pairs of arrays the check tried, two empty ones too. Your browser ran it, in a web worker.",
                        "nl": "Je `dot` vermenigvuldigt de arrays paar voor paar en telt alles op, voor alle {n} paren arrays die de check probeerde, ook twee lege. Je browser draaide hem, in een web worker."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: 1 × 4 + 2 × 5 + 3 × 6.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: 1 × 4 + 2 × 5 + 3 × 6."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`.",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn."},
        },
    ),
}


# --- exercise 2: sumFrom --------------------------------------------------------------------------------------------------

_START_SUM = '''async function sumFrom(url) {
  return 0;
}
'''
SOLUTION_SUM = '''async function sumFrom(url) {
  const response = await fetch(url);
  const numbers = await response.json();
  let total = 0;
  for (const number of numbers) {
    total += number;
  }
  return total;
}
'''
SUM_CASES = [("data:application/json,[1,2,3]",), ("data:application/json,[]",), ("data:application/json,[0.5,0.25,4]",),
             ("data:application/json,[-2,10]",)]


def _not_awaited_fetch(e) -> bool:
    return e.name == "TypeError" and ".json is not a function" in e.message


def _not_awaited_json(e) -> bool:
    """What a list that is still a promise does in a loop or with an array's own functions."""
    return e.name == "TypeError" and ("is not iterable" in e.message or "has no properties" in e.message
                                      or any(f".{f} is not a function" in e.message for f in ("reduce", "map", "forEach")))


EXERCISES["sumFrom"] = JsBox(
    name="sumFrom",
    picture=outcomes_picture,
    function="sumFrom",
    start=_START_SUM,
    solution=SOLUTION_SUM,
    visible=SUM_CASES[:1],
    hidden=SUM_CASES[1:],
    first=python_mistakes(_START_SUM) + [
        Mistake("no-async", state="error",
                matches=lambda got, text: re.search(r"\bawait\b", text) is not None and re.search(r"\basync\b", text) is None,
                en="`await` only works inside a function with `async` in front: keep `async function sumFrom(url) {`.",
                nl="`await` werkt alleen in een functie met `async` ervoor: houd `async function sumFrom(url) {`."),
    ],
    mistakes=[
        Mistake("await-fetch", error=_not_awaited_fetch,
                en="With `{call}`, your function stops: what `fetch(url)` gives is a promise, not the answer yet, and a promise has no `.json()`. Wait for the answer first, on a line of its own: `const response = await fetch(url);`.",
                nl="Met `{call}` stopt je functie: wat `fetch(url)` geeft, is een promise, nog niet het antwoord, en een promise heeft geen `.json()`. Wacht eerst op het antwoord, op een eigen regel: `const response = await fetch(url);`."),
        Mistake("await-json", error=_not_awaited_json,
                en="With `{call}`, your function stops: `response.json()` gives a promise too, and the list only comes once you wait for it. Put `await` before `response.json()`.",
                nl="Met `{call}` stopt je functie: `response.json()` geeft ook een promise, en de lijst komt pas als je erop wacht. Zet `await` vóór `response.json()`."),
        Mistake("for-in", matches=lambda got, text: isinstance(got, str),
                en="`{call}` gives the text `{got}`: your loop goes over the positions as text, `\"0\"`, `\"1\"`, and adds those up. That is `for ... in`; for the numbers themselves, use `for ... of`.",
                nl="`{call}` geeft de tekst `{got}`: je loop gaat langs de posities als tekst, `\"0\"`, `\"1\"`, en telt die op. Dat doet `for ... in`; voor de getallen zelf gebruik je `for ... of`."),
        *_python_names(),
        _const(),
        Mistake("first-only", wrong=SOLUTION_SUM.replace("  let total = 0;\n  for (const number of numbers) {\n    total += number;\n  }\n  return total;\n",
                                                          "  return numbers[0];\n"),
                en="`{call}` gives `{got}`: only the first number. Add them all up, with a loop over `numbers`.",
                nl="`{call}` geeft `{got}`: alleen het eerste getal. Tel ze allemaal op, met een loop over `numbers`."),
        Mistake("no-return", matches=lambda got, text: got == UNDEFINED,
                en="`{call}` gives `undefined`: your function gives nothing back. End it with `return total;`.",
                nl="`{call}` geeft `undefined`: je functie geeft niets terug. Sluit hem af met `return total;`."),
    ],
    texts={
        "empty": {"en": "`sumFrom` gives 0 yet. Fetch the JSON at `url`, wait for the list, and add its numbers up.",
                  "nl": "`sumFrom` geeft nog 0. Haal de JSON op van `url`, wacht op de lijst, en tel de getallen op."},
        "correct": {"en": "Your `sumFrom` waits for the answer and for its JSON, and adds up the list, for all {n} addresses the check tried, the empty list too.",
                    "nl": "Je `sumFrom` wacht op het antwoord en op de JSON ervan, en telt de lijst op, voor alle {n} adressen die de check probeerde, ook de lege lijst."},
        "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: 1 + 2 + 3.",
                  "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: 1 + 2 + 3."},
        "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`.",
                   "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn."},
    },
)

REQUIRED = ["dot", "sumFrom", "worker", "import"]

CASES = [
    ("dot", _START_DOT, "empty", None),
    ("dot", SOLUTION_DOT, "correct", None),
    ("dot", "function dot(a, b) {\n  return a.reduce((sum, x, i) => sum + x * b[i], 0);\n}\n", "correct", None),
    ("dot", "def dot(a, b):\n    return sum(x * y for x, y in zip(a, b))\n", "error", "python-def"),
    ("dot", SOLUTION_DOT.replace("i < a.length", "i < len(a)"), "error", "len"),
    ("dot", "function dot(a, b) {\n  let total = 0;\n  for (const i of range(a.length)) {\n    total += a[i] * b[i];\n  }\n  return total;\n}\n", "error", "range"),
    ("dot", SOLUTION_DOT.replace("total += ", "total = "), "wrong", "last"),
    ("dot", SOLUTION_DOT.replace("i < a.length", "i <= a.length"), "wrong", "past-end"),
    ("dot", SOLUTION_DOT.replace("  return total;\n", ""), "wrong", "no-return"),
    ("dot", SOLUTION_DOT.replace("total += a[i] * b[i]", "total += a[i] + b[i]"), "wrong", None),
    ("dot", SOLUTION_DOT.replace("let total = 0;", "let total = 1;").replace("total += a[i] * b[i]", "total += a[i] * b[i] - 1 / a.length"), "almost", None),
    ("dot", "function dot(a, b) {\n  return a.map(\n}\n", "error", None),
    ("dot", "function dot(a, b) {\n  while (true) {}\n}\n", "slow", None),
    ("dot", "function product(a, b) {\n  return 0;\n}\n", "wrong", "no-function"),
    ("dot", SOLUTION_DOT.replace("let total = 0;", "total = 0;"), "error", "no-let"),
    ("dot", SOLUTION_DOT.replace("let total", "const total"), "error", "const"),
    ("dot", "function dot(a, b) {\n  let total = 0\n  for i in range(len(a)):\n    total += a[i] * b[i]\n  return total\n}\n", "error", "python-block"),
    ("dot", SOLUTION_DOT[:-2], "error", "braces"),
    ("dot", "function dot(a, b) {\n  return sum(a);\n}\n", "error", "python-sum"),
    ("dot", SOLUTION_DOT.replace("return total;", "print(total);"), "error", "print"),
    ("sumFrom", _START_SUM, "empty", None),
    ("sumFrom", SOLUTION_SUM, "correct", None),
    ("sumFrom", "async function sumFrom(url) {\n  const numbers = await (await fetch(url)).json();\n  return numbers.reduce((a, b) => a + b, 0);\n}\n", "correct", None),
    ("sumFrom", SOLUTION_SUM.replace("await fetch(url)", "fetch(url)"), "error", "await-fetch"),
    ("sumFrom", SOLUTION_SUM.replace("await response.json()", "response.json()"), "error", "await-json"),
    ("sumFrom", SOLUTION_SUM.replace("async function", "function"), "error", "no-async"),
    ("sumFrom", SOLUTION_SUM.replace("  let total = 0;\n  for (const number of numbers) {\n    total += number;\n  }\n  return total;\n", "  return numbers[0];\n"), "wrong", "first-only"),
    ("sumFrom", SOLUTION_SUM.replace("  return total;\n", ""), "wrong", "no-return"),
    ("sumFrom", "def sumFrom(url):\n    return 0\n", "error", "python-def"),
    ("sumFrom", SOLUTION_SUM.replace("for (const number of numbers)", "for (const number in numbers)"), "wrong", "for-in"),
    ("sumFrom", SOLUTION_SUM.replace("  const response = await fetch(url);\n  const numbers = await response.json();\n",
                                     "  const numbers = await fetch(url).json();\n"), "error", "await-fetch"),
]


# --- the demo: the page, or a web worker -----------------------------------------------------------------------------------

def freeze(words: dict, lang: str) -> Freeze:
    """The same work on the page and in a web worker, with the longest pause of the page measured (jsbox.Freeze)."""
    return Freeze(words=words, locale={"en": "en-US", "nl": "nl-NL"}[lang])
