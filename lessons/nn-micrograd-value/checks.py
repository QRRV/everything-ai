# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nn-micrograd-value: the learner's own Value, with __mul__, relu and backward_step, saved
in the project as nn/value.py; and the backward pass of one student's loss with that Value. And the pictures: a
Value graph walked one backward step at a time, and five students with the line the learner's Value trained."""
import ast
import re

from everything_ai import draw, projects
from everything_ai.check import ClassBox, CodeBox, Code, Mistake, run_code

# --- the picture: d = a * b + a, with a = 2 and b = 3 ----------------------------------------------------------------

_GRADS = [[0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 1, 1], [4, 2, 1, 1]]   # a, b, c, d after each step


def grads(step: int) -> list[int]:
    return _GRADS[step]


def picture(step: int, labels: dict[str, str]):
    """The Value graph of d = a · b + a: each node with its data and its grad after `step` of the backward pass,
    and the local gradients its parents hold at the arrows. Drawn once for every step of the slider (L.explore)."""
    a, b, c, d = _GRADS[step]
    nodes = [("a", 0, 0, "a", "2", f"grad = {a}", None), ("b", 0, 1.25, "b", "3", f"grad = {b}", None),
             ("c", 1, 1, "c = a · b", "6", f"grad = {c}", None), ("d", 2, 0, "d = c + a", "8", f"grad = {d}", None)]
    edges = [("a", "c", "3"), ("b", "c", "2"), ("c", "d", "1"), ("a", "d", "1")]
    return draw.computational_graph(nodes, edges, title=labels["title"], alt=labels["alt"])


def steps_table(step: int, labels: dict[str, str]):
    """The grads of a, b, c and d after every step of the backward pass, one row per step, with the row of `step`
    in ink: the whole walk at once, so that you see which grad changes twice before you touch the slider."""
    rows = [[str(g) for g in row] for row in _GRADS]
    return draw.data_tables([(labels["label"], ["a", "b", "c", "d"], rows, {(step, j) for j in range(4)}, "")],
                            alt=labels["alt"])


# --- the demo: five students --------------------------------------------------------------------------------------

HOURS = [1, 2, 3, 4, 5]
GRADES = [3, 5, 6, 7, 9]
_TRAIN = """from nn.value import Value

xs = {xs}
ys = {ys}
w = Value(1)
b = Value(1)
for step in range({steps}):
    w.grad = 0
    b.grad = 0
    for x0, y0 in zip(xs, ys):
        m = w * Value(x0)
        y_pred = m + b
        neg = y_pred * Value(-1)
        r = Value(y0) + neg
        loss = r * r
        loss.grad = 1 / len(xs)
        loss.backward_step()
        r.backward_step()
        neg.backward_step()
        y_pred.backward_step()
        m.backward_step()
    w.data = w.data - 0.05 * w.grad
    b.data = b.data - 0.05 * b.grad
[w.data, b.data]"""


def _mse(w: float, b: float) -> float:
    return sum((y - (w * x + b)) ** 2 for x, y in zip(HOURS, GRADES)) / len(HOURS)


def trained():
    """For the demo: None when the learner's Value is not saved yet; False when it no longer gives the right slopes,
    or stops with an error; else (w, b, the mean squared error before, after) after 300 steps with that Value."""
    ok = value_ok()
    if ok is None:
        return None
    if not ok:
        return False
    run = run_code(_TRAIN.format(xs=HOURS, ys=GRADES, steps=300))
    if run.error is not None or run.timeout:
        return False
    try:
        w, b = (float(v) for v in run.value)
    except (TypeError, ValueError):
        return False
    return w, b, _mse(1, 1), _mse(w, b)


def fit_picture(w: float, b: float, labels: dict[str, str]) -> str:
    """The five students as points, and the line w · x + b through them."""
    from everything_ai import chart
    p = chart.Panel((0, 6), (0, 10), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    p.line([0, 6], [b, 6 * w + b], "pred", 2.2)
    p.points(HOURS, GRADES, "data", 5)
    return chart.figure([p], alt=labels["alt"], width=420, height=260)


# The lesson is done when these are correct. "order" is the choice question in the lesson.
REQUIRED = ["order", "value", "student"]


def _no_return(text: str) -> bool:
    """`__mul__` or `relu` gives nothing back: no `return` with a value in it, so `a * b` or `a.relu()` is None.
    A method that is still `...` is caught before this, by "dots-left"."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.FunctionDef) and node.name in ("__mul__", "relu")
               and not any(isinstance(sub, ast.Return) and sub.value is not None for sub in ast.walk(node))
               for node in ast.walk(tree))


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# --- exercise 2: the Value ------------------------------------------------------------------------------------------

_V_HEAD = ("class Value:\n    def __init__(self, data, parents=()):\n        self.data = data\n        self.grad = 0\n"
           "        self.parents = parents\n\n    def __add__(self, other):\n        return Value(self.data + other.data, [(self, 1), (other, 1)])\n\n")
_MUL = "    def __mul__(self, other):\n        return Value(self.data * other.data, [(self, other.data), (other, self.data)])\n\n"
_RELU = ("    def relu(self):\n        if self.data > 0:\n            return Value(self.data, [(self, 1)])\n"
         "        return Value(0, [(self, 0)])\n\n")
_STEP = ("    def backward_step(self):\n        for parent, local in self.parents:\n"
         "            parent.grad = parent.grad + local * self.grad")
_V_START = (_V_HEAD + "    def __mul__(self, other):\n        ...\n\n    def relu(self):\n        ...\n\n"
            "    def backward_step(self):\n        ...\n")
_V = _V_HEAD + _MUL + _RELU + _STEP


def _v(mul=_MUL, relu=_RELU, step=_STEP) -> str:
    return _V_HEAD + mul + relu + step


_SCENARIOS = [
    "a = Value(2)\nb = Value(3)\nc = a * b\nc.grad = 1\nc.backward_step()\n[c.data, c.grad, a.grad, b.grad]",
    "r = Value(2)\np = r * r\np.grad = 1\np.backward_step()\n[p.data, r.grad]",
    "a = Value(2)\nb = Value(3)\nc = a * b\nd = c + a\nd.grad = 1\nd.backward_step()\nc.backward_step()\n[d.data, a.grad, b.grad]",
    "a = Value(3)\nb = Value(-1)\nc = a * b\nc.grad = 5\nc.backward_step()\n[c.data, a.grad, b.grad]",
    "a = Value(1.5)\nh = a.relu()\nh.grad = 2\nh.backward_step()\n[h.data, a.grad]",
    "a = Value(-2)\nh = a.relu()\nh.grad = 2\nh.backward_step()\n[h.data, a.grad]",
    "a = Value(0)\nh = a.relu()\nh.grad = 2\nh.backward_step()\n[h.data, a.grad]",
]


def value_ok():
    """None when the learner's Value is not in the project yet; True when it gives the course's results in every
    scenario of exercise 2; False when it does not, or stops with an error."""
    if not (projects.folder() / "nn" / "value.py").exists():
        return None
    for scenario in _SCENARIOS:
        got = run_code("from nn.value import Value\n\n" + scenario)
        want = run_code(_V + "\n\n" + scenario)
        if got.error is not None or got.timeout or got.value != want.value:
            return False
    return True


# --- exercise 3: the student ----------------------------------------------------------------------------------------

_S_HEAD = ("from nn.value import Value\n\nx = Value(4)\ny = Value(7)\nw = Value(1)\nb = Value(1)\n\n# the forward pass\n"
           "m = w * x\ny_pred = m + b\nneg = y_pred * Value(-1)\nr = y + neg\nloss = r * r\n\n# the backward pass\n")
_S_END = "[w.grad, b.grad]"
_S_START = _S_HEAD + "...\n" + _S_END + "\n"
_ORDER = ["loss", "r", "neg", "y_pred", "m"]


def _s(order=_ORDER, start="loss.grad = 1\n") -> str:
    return _S_HEAD + start + "".join(f"{name}.backward_step()\n" for name in order) + _S_END


_S = _s()


def _steps(text: str) -> list[str]:
    """The values of the forward pass whose backward step the code takes, in order. A step on x, y, w or b changes nothing."""
    return [name for name in re.findall(r"^\s*(\w+)\.backward_step\(\)\s*$", text, re.MULTILINE) if name in _ORDER]


def _starts_at_one(text: str) -> bool:
    return re.search(r"^\s*loss\.grad\s*=\s*1(?:\.0*)?\s*$", text, re.MULTILINE) is not None


_REDO = {"en": " Did you change that file? Delete it, and check exercise 2 again: its check then saves your class again.",
         "nl": " Heb je dat bestand veranderd? Verwijder het, en laat oefening 2 opnieuw nakijken: die check bewaart je class dan opnieuw."}

EXERCISES = {
    "value": ClassBox(
        name="value",
        start=_V_START,
        solution=_V,
        scenarios=_SCENARIOS,
        save_as=("nn", "value"),
        first=[_DOTS,
               Mistake("no-return", matches=lambda got, text: _no_return(text),
                       en="`__mul__` or `relu` makes a new `Value`, but does not give it back: then `a * b` or `a.relu()` gives nothing, `None`. Put `return` in front of the new `Value`.",
                       nl="`__mul__` of `relu` maakt een nieuwe `Value`, maar geeft hem niet terug: dan geeft `a * b` of `a.relu()` niets, `None`. Zet `return` voor de nieuwe `Value`.")],
        mistakes=[
            Mistake("pair", syntax=r"for\s+(\w+)\s+in\s+self\.parents\s*:\s*\n\s*\1\.grad",
                    en="Python stops: each item of `parents` is a pair, `(parent, local)`, and a pair has no `grad`. Take the pair apart in the loop, as in the worked example: `for parent, local in self.parents:`.",
                    nl="Python stopt: elk item van `parents` is een paar, `(parent, local)`, en een paar heeft geen `grad`. Haal het paar uit elkaar in de loop, zoals in het uitgewerkte voorbeeld: `for parent, local in self.parents:`."),
            Mistake("replace", wrong=_v(step="    def backward_step(self):\n        for parent, local in self.parents:\n            parent.grad = local * self.grad"),
                    en="After these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: `backward_step` replaces the `grad` of a parent. A value that goes into two places gets a slope back from each: add it, `parent.grad = parent.grad + ...`.",
                    nl="Na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: `backward_step` vervangt de `grad` van een ouder. Een waarde die op twee plekken in gaat, krijgt van elk een helling terug: tel hem op, `parent.grad = parent.grad + ...`."),
            Mistake("own-number", wrong=_v(mul="    def __mul__(self, other):\n        return Value(self.data * other.data, [(self, self.data), (other, other.data)])\n\n"),
                    en="After these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: in `__mul__`, each parent has its own number as its local gradient. For a · b, a changes the product as fast as the number of b: `(self, other.data)` and `(other, self.data)`.",
                    nl="Na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: in `__mul__` heeft elke ouder zijn eigen getal als local gradient. Bij a · b verandert a het product zo snel als het getal van b: `(self, other.data)` en `(other, self.data)`."),
            Mistake("no-grad", wrong=_v(step="    def backward_step(self):\n        for parent, local in self.parents:\n            parent.grad = parent.grad + local"),
                    en="After these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: `backward_step` passes on only the local gradient. Multiply it by the value's own slope: `local * self.grad`.",
                    nl="Na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: `backward_step` geeft alleen de local gradient door. Vermenigvuldig die met de eigen helling van de waarde: `local * self.grad`."),
            Mistake("relu-number", wrong=_v(relu="    def relu(self):\n        if self.data > 0:\n            return Value(self.data, [(self, self.data)])\n        return Value(0, [(self, 0)])\n\n"),
                    en="After these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: above 0 your ReLU has its number as local gradient. ReLU passes its number on as it is, so its slope there is 1: `[(self, 1)]`.",
                    nl="Na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: boven 0 heeft je ReLU zijn getal als local gradient. ReLU geeft zijn getal door zoals het is, dus zijn helling is daar 1: `[(self, 1)]`."),
            Mistake("relu-pass", wrong=_v(relu="    def relu(self):\n        return Value(self.data, [(self, 1)])\n\n"),
                    en="After these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: your ReLU lets a negative number through. Below 0 it gives 0, with local gradient 0: `if self.data > 0:`, and otherwise `return Value(0, [(self, 0)])`.",
                    nl="Na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: je ReLU laat een negatief getal door. Onder 0 geeft hij 0, met local gradient 0: `if self.data > 0:`, en anders `return Value(0, [(self, 0)])`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the three `...`: `__mul__` and `relu` give back a new `Value` with its parents, and `backward_step` passes `grad` back to each parent.",
                      "nl": "Vervang de drie `...`: `__mul__` en `relu` geven een nieuwe `Value` terug met zijn ouders, en `backward_step` geeft `grad` terug aan elke ouder."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "wrong": {"en": "After these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Does `__mul__` give back the product, with, for each parent, the number of the other one as its local gradient? And does `backward_step` add its own `grad` times the local gradient to each parent's `grad`?",
                      "nl": "Na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Geeft `__mul__` het product terug, met voor elke ouder het getal van de andere als local gradient? En telt `backward_step` zijn eigen `grad` keer de local gradient op bij de `grad` van elke ouder?"},
            "almost": {"en": "Right for the example. But after these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Does `backward_step` add to what a parent already has, and does `relu` give 0, with local gradient 0, at 0 and below?",
                       "nl": "Goed voor het voorbeeld. Maar na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Telt `backward_step` op bij wat een ouder al heeft, en geeft `relu` 0, met local gradient 0, bij 0 en lager?"},
            "correct": {"en": "Your `Value` is right in all {n} cases: products, ReLU on both sides of 0, a value used twice, and two steps in a row. The check saved it in your project, as `{saved}`: exercise 3 imports it from there.",
                        "nl": "Je `Value` klopt in alle {n} gevallen: producten, ReLU aan beide kanten van 0, een waarde die twee keer meedoet, en twee stappen achter elkaar. De check heeft hem in je project bewaard, als `{saved}`: oefening 3 haalt hem daar binnen."},
            "kept": {"en": "Your `Value` is right in all {n} cases. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `Value` klopt in alle {n} gevallen. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren? Verwijder het bestand en laat opnieuw nakijken."},
        },
    ),
    "student": CodeBox(
        name="student",
        start=_S_START,
        solution=_S,
        keep=["from nn.value import Value", "x = Value(4)", "y = Value(7)", "w = Value(1)", "b = Value(1)", "m = w * x", "y_pred = m + b", "neg = y_pred * Value(-1)", "r = y + neg", "loss = r * r", _S_END],
        others=[{"w": Code("Value(2)")}, {"w": Code("Value(-0.5)"), "b": Code("Value(3)")}],
        first=[
            _DOTS,
            Mistake("not-saved-yet", state="empty", matches=lambda got, text: value_ok() is None,
                    en="Your `Value` is not in your project yet. The check of exercise 2 saves it, as `work/projects/nn/value.py`, once it is right. Finish exercise 2 first; this check then runs again by itself.",
                    nl="Je `Value` staat nog niet in je project. De check van oefening 2 bewaart hem, als `work/projects/nn/value.py`, zodra hij klopt. Maak eerst oefening 2 af; deze check draait daarna vanzelf opnieuw."),
            Mistake("bad-value", matches=lambda got, text: value_ok() is False,
                    en="Your backward pass may be right, but the `Value` in `work/projects/nn/value.py` no longer gives the right slopes, or stops with an error." + _REDO["en"],
                    nl="Je backward pass kan goed zijn, maar de `Value` in `work/projects/nn/value.py` geeft niet meer de goede hellingen, of stopt met een fout." + _REDO["nl"]),
            Mistake("typed", matches=lambda got, text: re.search(r"^\s*(w|b)\.grad\s*=", text, re.MULTILINE) is not None,
                    en="You set `w.grad` or `b.grad` yourself. Let the backward pass give them their slopes: set only `loss.grad`, and call `backward_step()`.",
                    nl="Je zet `w.grad` of `b.grad` zelf. Laat de backward pass ze hun helling geven: zet alleen `loss.grad`, en roep `backward_step()` aan."),
        ],
        mistakes=[
            Mistake("no-start", matches=lambda got, text: re.search(r"^\s*loss\.grad\s*=", text, re.MULTILINE) is None,
                    en="Your last line gives `{got}`: every `grad` starts at 0, and so does the one of `loss`. Set `loss.grad = 1` first: the slope of the loss with respect to itself.",
                    nl="Je laatste regel geeft `{got}`: elke `grad` begint bij 0, ook die van `loss`. Zet eerst `loss.grad = 1`: de helling van de loss naar zichzelf."),
            Mistake("once", matches=lambda got, text: _starts_at_one(text) and sorted(_steps(text)) != sorted(_ORDER),
                    en="Your last line gives `{got}`, and it should be `[-16, -4]`. Does every value of the forward pass take its `backward_step()` exactly once? A step left out passes nothing on; a step taken twice passes its slope on twice.",
                    nl="Je laatste regel geeft `{got}`, en dat moet `[-16, -4]` zijn. Zet elke waarde van de forward pass precies één keer zijn `backward_step()`? Een stap die ontbreekt, geeft niets door; een stap die twee keer gebeurt, geeft zijn helling twee keer door."),
            Mistake("order", matches=lambda got, text: _starts_at_one(text) and _steps(text) != _ORDER,
                    en="Your last line gives `{got}`, and it should be `[-16, -4]`. Every value takes its step once, but not in the right order: a value that passes its `grad` on before it has all of it passes on too little. Which value was made last, and which just before it?",
                    nl="Je laatste regel geeft `{got}`, en dat moet `[-16, -4]` zijn. Elke waarde zet één keer zijn stap, maar niet in de goede volgorde: een waarde die zijn `grad` doorgeeft voordat hij hem helemaal heeft, geeft te weinig door. Welke waarde is als laatste gemaakt, en welke vlak daarvoor?"),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the backward pass: `loss.grad = 1`, and then one `backward_step()` per value, from the loss back.",
                      "nl": "Vervang de `...` door de backward pass: `loss.grad = 1`, en dan één `backward_step()` per waarde, vanaf de loss terug."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug."},
            "no_value": {"en": "The last line of the box shows nothing. Put `[w.grad, b.grad]` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `[w.grad, b.grad]` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`, and it should be `[-16, -4]`. Do you set `loss.grad = 1` before the backward steps, and does every value take its step once, from the last one made back to the first?",
                      "nl": "Je laatste regel geeft `{got}`, en dat moet `[-16, -4]` zijn. Zet je `loss.grad = 1` vóór de backward steps, en zet elke waarde één keer zijn stap, van de laatst gemaakte terug naar de eerste?"},
            "other": {"en": "Right for these numbers. But with {where}, your last line gives `{got}`, and it should give `{want}`. Does your backward pass get the slopes from the graph, rather than from numbers you typed?",
                      "nl": "Goed voor deze getallen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Haalt je backward pass de hellingen uit de graaf, en niet uit getallen die je typte?"},
            "correct": {"en": "−16 and −4: the slopes of the lesson before, now from your own `Value`. The check also tried other numbers for `w` and `b`.",
                        "nl": "−16 en −4: de hellingen van de vorige les, nu uit je eigen `Value`. De check probeerde ook andere getallen voor `w` en `b`."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
# The cases of exercise 3 need the learner's Value in the project; the tests save the course's own there first.
CASES = [
    ("value", _V_START, "empty", None),
    ("value", _V, "correct", None),
    ("value", _v(relu="    def relu(self):\n        if self.data <= 0:\n            return Value(0, [(self, 0)])\n        return Value(self.data, [(self, 1)])\n\n"), "correct", None),
    ("value", _v(step="    def backward_step(self):\n        for pair in self.parents:\n            pair[0].grad += pair[1] * self.grad"), "correct", None),
    ("value", _v(step="    def backward_step(self):\n        for parent, local in self.parents:\n            parent.grad = local * self.grad"), "wrong", "replace"),
    ("value", _v(mul="    def __mul__(self, other):\n        return Value(self.data * other.data, [(self, self.data), (other, other.data)])\n\n"), "wrong", "own-number"),
    ("value", _v(step="    def backward_step(self):\n        for parent, local in self.parents:\n            parent.grad = parent.grad + local"), "wrong", "no-grad"),
    ("value", _v(relu="    def relu(self):\n        if self.data > 0:\n            return Value(self.data, [(self, self.data)])\n        return Value(0, [(self, 0)])\n\n"), "wrong", "relu-number"),
    ("value", _v(relu="    def relu(self):\n        return Value(self.data, [(self, 1)])\n\n"), "wrong", "relu-pass"),
    ("value", _v(step="    def backward_step(self):\n        for parent in self.parents:\n            parent.grad = parent.grad + self.grad"), "error", "pair"),
    ("value", _v(mul="    def __mul__(self, other):\n        Value(self.data * other.data, [(self, other.data), (other, self.data)])\n\n"), "wrong", "no-return"),
    ("value", _v(relu="    def relu(self):\n        return Value(max(0, self.data), [(self, 1 if self.data > 0 else 0)])\n\n"), "correct", None),
    ("student", _S, "correct", None),
    ("student", _S_START, "empty", None),
    ("student", _s(start=""), "wrong", "no-start"),
    ("student", _s(order=["m", "y_pred", "neg", "r", "loss"]), "wrong", "order"),
    ("student", _s(order=["loss", "r", "neg", "m", "y_pred"]), "wrong", "order"),
    ("student", _s(order=["loss", "r", "neg", "y_pred"]), "wrong", "once"),
    ("student", _s(order=["loss", "r", "neg", "y_pred", "m", "m"]), "wrong", "once"),
    ("student", _s(order=["loss", "r", "neg", "y_pred", "m", "x", "w", "b"]), "correct", None),
    ("student", _s(start="loss.grad = 2\n"), "wrong", None),
    ("student", _S.replace("w = Value(1)", "w = Value(2)"), "wrong", "changed-line"),
    ("student", _S.replace("loss.grad = 1\n", "loss.grad = 1\nw.grad = -16\n"), "wrong", "typed"),
]
