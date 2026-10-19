# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nn-micrograd-backward: the learner's own build_order and backward, saved in the project as
nn/backward.py, tried on graphs made with the learner's own Value; and the slopes of a neuron with two inputs with
one call to that backward. And the pictures: build_order walked one step at a time on a slider, and a network with a
hidden layer trained on XOR with the learner's own Value and backward."""
import re

from everything_ai import draw, projects
from everything_ai.check import ClassBox, CodeBox, Code, Mistake, run_code, with_time_limit, Timeout

# The Value of the lesson before: yours when it is there and gives the course's results, else the course's (template 1.3).
projects.builds_on(("nn", "value", "nn-micrograd-value", "value"))

# --- the picture: build_order on d = a * b + a ---------------------------------------------------------------------

ORDER = ["a", "b", "c", "d"]
CALLS = ["d", "c", "a", "b", "a"]   # the calls of build_order, in order: the last one finds `a` in seen and stops at once
CALLED = [3, 4, 4, 5, 5]            # how many of those calls each step of the slider has made


def picture(step: int, labels: dict[str, str]):
    """The Value graph of d = a · b + a, with the place in the list of each value that build_order has placed by `step`.
    Drawn once for every step of the slider (L.explore)."""
    def label(name: str, text: str) -> str:
        k = ORDER.index(name)
        return f"{text}, {labels['place']} {k + 1}" if k < step else text
    nodes = [("a", 0, 0, label("a", "a"), "2", None, None), ("b", 0, 1.25, label("b", "b"), "3", None, None),
             ("c", 1, 1, label("c", "c = a · b"), "6", None, None), ("d", 2, 0, label("d", "d = c + a"), "8", None, None)]
    edges = [("a", "c", None), ("b", "c", None), ("c", "d", None), ("a", "d", None)]
    return draw.computational_graph(nodes, edges, title=labels["title"], alt=labels["alt"])


# --- the course's own Value, to compare with -----------------------------------------------------------------------

_VALUE = projects.course_code("nn", "value")


# --- exercise 2: build_order and backward --------------------------------------------------------------------------

_B_ORDER_HEAD = "def build_order(v, seen, order):\n    if v in seen:\n        return\n    seen.add(v)\n"
_B_PARENTS = "    for parent, local in v.parents:\n        build_order(parent, seen, order)\n"
_B_BACK_HEAD = "    order.append(v)\n\n\ndef backward(loss):\n    order = []\n    build_order(loss, set(), order)\n    loss.grad = 1\n"
_B_LOOP = "    for v in reversed(order):\n        v.backward_step()"
_B_START = _B_ORDER_HEAD + "    ...\n" + _B_BACK_HEAD + "    ...\n"
_B = _B_ORDER_HEAD + _B_PARENTS + _B_BACK_HEAD + _B_LOOP


def _b(parents=_B_PARENTS, loop=_B_LOOP) -> str:
    return _B_ORDER_HEAD + parents + _B_BACK_HEAD + loop


_IMPORT = "from nn.value import Value\n"
_SCENARIOS = [
    _IMPORT + "a = Value(2)\nb = Value(3)\nc = a * b\nd = c + a\nbackward(d)\n[a.grad, b.grad, c.grad, d.grad]",
    _IMPORT + "x = Value(4)\ny = Value(7)\nw = Value(1)\nb = Value(1)\nr = y + (w * x + b) * Value(-1)\nloss = r * r\nbackward(loss)\n[w.grad, b.grad]",
    _IMPORT + "r = Value(3)\nloss = r * r\nbackward(loss)\nr.grad",
    # a graph with one topological order only, a b c d, so any right build_order gives these numbers
    _IMPORT + "a = Value(2)\nb = a + a\nc = b * a\nd = c + b\norder = []\nbuild_order(d, set(), order)\n"
              "numbers = []\nfor v in order:\n    numbers.append(v.data)\nnumbers",
    _IMPORT + "a = Value(-1)\nb = Value(2)\nh = (a * b).relu()\nout = h * Value(5) + b\nbackward(out)\n[a.grad, b.grad]",
    _IMPORT + "x = Value(1)\nfirst = x\nfor i in range(25):\n    x = x + x\nbackward(x)\nfirst.grad",
]


def backward_ok():
    """None when the learner's backward is not in the project; True when it gives the course's slopes; else False."""
    if not (projects.folder() / "nn" / "backward.py").exists():
        return None
    for scenario in _SCENARIOS:
        got = run_code("from nn.backward import build_order, backward\n" + scenario)
        want = run_code(_VALUE + "\n\n" + _B + "\n\n" + scenario.replace(_IMPORT, ""))
        if got.error is not None or got.timeout or got.value != want.value:
            return False
    return True


# --- exercise 3: a neuron with two inputs --------------------------------------------------------------------------

_N_HEAD = ("from nn.value import Value\nfrom nn.backward import backward\n\nx1 = Value(2)\nx2 = Value(1)\ny = Value(3)\n"
           "w1 = Value(0.5)\nw2 = Value(-1)\nb = Value(1)\n\n")
_N_END = "[w1.grad, w2.grad, b.grad]"
_N_START = _N_HEAD + "...\n" + _N_END + "\n"
_N_BODY = "z = w1 * x1 + w2 * x2 + b\ny_pred = z.relu()\nr = y_pred + y * Value(-1)\nloss = r * r\nbackward(loss)\n"
_N = _N_HEAD + _N_BODY + _N_END


def _n(body: str) -> str:
    return _N_HEAD + body + _N_END


# --- the demo: XOR with a hidden layer of 4 neurons ----------------------------------------------------------------

_X = [(0, 0), (0, 1), (1, 0), (1, 1)]
_Y = [0, 1, 1, 0]
_START = [0.5, -0.4, -0.3, 0.6, 0.7, 0.8, -0.6, -0.5,   # the weights of the 4 hidden neurons, two each
          0.1, 0.1, -0.2, 0.3,                          # their bias terms
          0.4, 0.5, -0.6, 0.3, 0.0]                     # the weights and the bias term of the output neuron


def _train(Value, backward, steps: int = 500, lr: float = 0.1):
    """A hidden layer of 4 ReLU neurons and a linear output on XOR, with the given Value and backward: the loss before
    and after, the outputs for the four corners, and the number of values in the graph of one step."""
    params = [Value(p) for p in _START]

    def model(x):
        out = params[16]
        for k in range(4):
            z = params[2 * k] * Value(x[0]) + params[2 * k + 1] * Value(x[1]) + params[8 + k]
            out = out + params[12 + k] * z.relu()
        return out

    def loss_now():
        loss = Value(0)
        for x, y in zip(_X, _Y):
            r = model(x) + Value(-y)
            loss = loss + r * r * Value(0.25)
        return loss

    before = loss_now().data
    for _ in range(steps):
        for p in params:
            p.grad = 0
        loss = loss_now()
        backward(loss)
        for p in params:
            p.data = p.data - lr * p.grad
    order, seen = [], set()

    def visit(v):
        if v in seen:
            return
        seen.add(v)
        for parent, _ in v.parents:
            visit(parent)
        order.append(v)
    visit(loss_now())
    return before, loss_now().data, [model(x).data for x in _X], len(order)


def trained():
    """For the demo: None when backward is not saved yet; False when the learner's backward is broken; else the loss
    before and after, the four outputs, the number of values per step and whether the Value was the learner's own."""
    if backward_ok() is None:
        return None
    if backward_ok() is not True:
        return False
    run = run_code("from nn.value import Value\nfrom nn.backward import backward\n")
    if run.error is not None:
        return False
    try:
        fit = with_time_limit(lambda: _train(run.names["Value"], run.names["backward"]), 30)
    except Exception:   # also Timeout
        return False
    own_value = all(module != "value" for _, module, _, _ in projects.stand_ins())
    return (*fit, own_value)


# The lesson is done when these are correct. "topo" is the choice question in the lesson.
REQUIRED = ["topo", "backward", "neuron"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

EXERCISES = {
    "backward": ClassBox(
        name="backward",
        start=_B_START,
        solution=_B,
        scenarios=_SCENARIOS,
        save_as=("nn", "backward"),
        first=[_DOTS],
        mistakes=[
            Mistake("self-call", matches=lambda got, text: re.search(r"^\s*build_order\(\s*v\s*,", text, re.MULTILINE) is not None,
                    en="After these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: in the loop, `build_order(v, ...)` calls itself with the same value. That value is already in `seen`, so the call stops at once, and the parents never get into the list. Call it on each parent: `build_order(parent, seen, order)`.",
                    nl="Na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: in de loop roept `build_order(v, ...)` zichzelf aan met dezelfde waarde. Die waarde zit al in `seen`, dus de aanroep stopt meteen, en de ouders komen nooit in de lijst. Roep hem aan op elke ouder: `build_order(parent, seen, order)`."),
            Mistake("pair", syntax=r"for\s+\w+\s+in\s+v\.parents\s*:\s*\n\s*build_order\(",
                    en="Python stops: the parents of a `Value` are pairs, `(parent, local)`. Take them apart in the loop, as in `backward_step()`: `for parent, local in v.parents:`.",
                    nl="Python stopt: de ouders van een `Value` zijn paren, `(parent, local)`. Haal ze uit elkaar in de loop, zoals in `backward_step()`: `for parent, local in v.parents:`."),
            Mistake("forwards", wrong=_b(loop="    for v in order:\n        v.backward_step()"),
                    en="After these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: only the parents of the loss get a slope, so only the loss passed its `grad` on. Does `backward` go through `order` from the start? Then every other value takes its step while its `grad` is still 0. Each value takes its step in `reversed(order)`.",
                    nl="Na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: alleen de ouders van de loss krijgen een helling, dus alleen de loss gaf zijn `grad` door. Loopt `backward` `order` van voren af door? Dan doet elke andere waarde zijn stap terwijl zijn `grad` nog 0 is. Laat elke waarde zijn stap doen in `reversed(order)`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: in `build_order`, a loop over the parents of `v` that calls `build_order` on each; in `backward`, a loop over `reversed(order)` that calls `backward_step()`.",
                      "nl": "Vervang de twee `...`: in `build_order` een loop over de ouders van `v` die `build_order` op elke ouder aanroept; in `backward` een loop over `reversed(order)` die `backward_step()` aanroept."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "slow": {"en": "Your code is still not done after 2 seconds. Does `build_order` stop at a value that is already in `seen`, so it visits each value once?",
                     "nl": "Je code is na 2 seconden nog niet klaar. Stopt `build_order` bij een waarde die al in `seen` zit, zodat hij elke waarde één keer bezoekt?"},
            "wrong": {"en": "After these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Does `build_order` do every parent first and only then append `v`, and does `backward` go through `reversed(order)`?",
                      "nl": "Na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Doet `build_order` eerst elke ouder en voegt hij pas daarna `v` toe, en loopt `backward` door `reversed(order)`?"},
            "almost": {"en": "Right for the first graph. But after these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Does every value get its backward step once, after all the values it went into?",
                       "nl": "Goed voor de eerste graaf. Maar na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Doet elke waarde zijn backward step één keer, na alle waarden waar hij in ging?"},
            "correct": {"en": "Your `backward` is right in all {n} cases, also on a graph of 26 values whose paths double 25 times. The check saved it in your project, as `{saved}`: exercise 3 imports it from there.",
                        "nl": "Je `backward` klopt in alle {n} gevallen, ook op een graaf van 26 waarden waarvan de paden 25 keer verdubbelen. De check heeft hem in je project bewaard, als `{saved}`: oefening 3 haalt hem daar binnen."},
            "kept": {"en": "Your `backward` is right in all {n} cases. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `backward` klopt in alle {n} gevallen. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren? Verwijder het bestand en laat opnieuw nakijken."},
        },
    ),
    "neuron": CodeBox(
        name="neuron",
        start=_N_START,
        solution=_N,
        keep=["from nn.value import Value", "from nn.backward import backward", _N_END],
        # z = -2, below 0, where every slope is 0; and z = 2 with another y, so a typed number for x1 or y shows
        others=[{"w1": Code("Value(-1)")}, {"x1": Code("Value(4)"), "y": Code("Value(-1)")}],
        first=[
            _DOTS,
            Mistake("not-saved-yet", state="empty", matches=lambda got, text: backward_ok() is None,
                    en="Your `backward` is not in your project yet. The check of exercise 2 saves it, as `work/projects/nn/backward.py`, once it is right. Finish exercise 2 first; this check then runs again by itself.",
                    nl="Je `backward` staat nog niet in je project. De check van oefening 2 bewaart hem, als `work/projects/nn/backward.py`, zodra hij klopt. Maak eerst oefening 2 af; deze check draait daarna vanzelf opnieuw."),
            Mistake("bad-backward", matches=lambda got, text: backward_ok() is False,
                    en="Your forward pass may be right, but your `backward` in `work/projects/nn/backward.py` no longer gives the right slopes. Did you change that file? Delete it, and check exercise 2 again.",
                    nl="Je forward pass kan goed zijn, maar je `backward` in `work/projects/nn/backward.py` geeft niet meer de goede hellingen. Heb je dat bestand veranderd? Verwijder het, en laat oefening 2 opnieuw nakijken."),
            Mistake("typed", matches=lambda got, text: re.search(r"^\s*(w1|w2|b)\.grad\s*=", text, re.MULTILINE) is not None,
                    en="You set a `grad` yourself. Let your `backward` give every value its slope: build `loss`, and call `backward(loss)`.",
                    nl="Je zet zelf een `grad`. Laat je `backward` elke waarde zijn helling geven: bouw `loss`, en roep `backward(loss)` aan."),
        ],
        mistakes=[
            Mistake("minus", error=lambda e: isinstance(e, TypeError) and ("for -:" in str(e) or "unary -" in str(e)),
                    en="Python stops: your `Value` has no `-`. It knows `+`, `*` and `relu()`, so write ŷ − y as `y_pred + y * Value(-1)`.",
                    nl="Python stopt: je `Value` kent geen `-`. Hij kent `+`, `*` en `relu()`, dus schrijf ŷ − y als `y_pred + y * Value(-1)`."),
            Mistake("power", error=lambda e: isinstance(e, TypeError) and "for ** or pow()" in str(e),
                    en="Python stops: your `Value` has no `**`. It knows `+`, `*` and `relu()`, so square a value by multiplying it by itself: `r * r`.",
                    nl="Python stopt: je `Value` kent geen `**`. Hij kent `+`, `*` en `relu()`, dus kwadrateer een waarde door hem met zichzelf te vermenigvuldigen: `r * r`."),
            Mistake("no-backward", matches=lambda got, text: re.search(r"^\s*backward\(", text, re.MULTILINE) is None,
                    en="Your last line gives `{got}`: every `grad` is still 0. Call `backward(loss)` after the forward pass.",
                    nl="Je laatste regel geeft `{got}`: elke `grad` is nog 0. Roep `backward(loss)` aan na de forward pass."),
            Mistake("no-square", wrong=_n("z = w1 * x1 + w2 * x2 + b\ny_pred = z.relu()\nloss = y_pred + y * Value(-1)\nbackward(loss)\n"),
                    en="Your last line gives `{got}`: you call `backward` on ŷ − y itself, not on its square. Make `loss = r * r`, and call `backward(loss)`.",
                    nl="Je laatste regel geeft `{got}`: je roept `backward` aan op ŷ − y zelf, niet op het kwadraat ervan. Maak `loss = r * r`, en roep `backward(loss)` aan."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the forward pass, ending in `loss`, and then `backward(loss)`.",
                      "nl": "Vervang de `...` door de forward pass, die eindigt in `loss`, en dan `backward(loss)`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug."},
            "no_value": {"en": "The last line of the box shows nothing. Put `[w1.grad, w2.grad, b.grad]` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `[w1.grad, w2.grad, b.grad]` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`, and it should be `[-8.0, -4.0, -4.0]`. Does your forward pass follow the formulas above, one operation at a time: each input times its weight, the bias term, ReLU on `z`, ŷ − y and then the square? And do you call `backward(loss)` once?",
                      "nl": "Je laatste regel geeft `{got}`, en dat moet `[-8.0, -4.0, -4.0]` zijn. Volgt je forward pass de formules hierboven, bewerking voor bewerking: elke input keer zijn weight, de bias term, ReLU op `z`, ŷ − y en dan het kwadraat? En roep je `backward(loss)` één keer aan?"},
            "other": {"en": "Right for these numbers. But with {where}, your last line gives `{got}`, and it should give `{want}`. Does your forward pass use the values themselves, with ReLU on `z`?",
                      "nl": "Goed voor deze getallen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruikt je forward pass de waarden zelf, met ReLU op `z`?"},
            "correct": {"en": "−8, −4 and −4, all three from one call to your own `backward`. The check also tried other numbers, one of them with `z` below 0, where every slope is 0.",
                        "nl": "−8, −4 en −4, alle drie uit één aanroep van je eigen `backward`. De check probeerde ook andere getallen, één ervan met `z` onder 0, waar elke helling 0 is."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("backward", _B_START, "empty", None),
    ("backward", _B, "correct", None),
    ("backward", _b(parents="    for pair in v.parents:\n        build_order(pair[0], seen, order)\n"), "correct", None),
    ("backward", _b(loop="    for v in order[::-1]:\n        v.backward_step()"), "correct", None),
    ("backward", _b(loop="    for v in order:\n        v.backward_step()"), "wrong", "forwards"),
    ("backward", _b(loop="    loss.backward_step()"), "wrong", "forwards"),
    ("backward", _b(parents="    for parent, local in v.parents:\n        build_order(v, seen, order)\n"), "wrong", "self-call"),
    ("backward", _b(parents="    for parent in v.parents:\n        build_order(parent, seen, order)\n"), "error", "pair"),
    # the value before its parents, and forwards: right slopes on the first three graphs, but not a topological order
    ("backward", _B_ORDER_HEAD + "    order.append(v)\n    for parent, local in v.parents:\n        build_order(parent, seen, order)\n"
                 + _B_BACK_HEAD.split("\n", 1)[1] + "    for v in order:\n        v.backward_step()", "almost", None),
    ("neuron", _N_START, "empty", None),
    ("neuron", _N, "correct", None),
    ("neuron", _n("loss = (w1 * x1 + w2 * x2 + b).relu() * Value(1) + y * Value(-1)\nloss = loss * loss\nbackward(loss)\n"), "correct", None),
    ("neuron", _n("z = w1 * x1 + w2 * x2 + b\ny_pred = z.relu()\nr = y_pred + y * Value(-1)\nloss = r * r\n"), "wrong", "no-backward"),
    ("neuron", _n("z = w1 * x1 + w2 * x2 + b\nr = z + y * Value(-1)\nloss = r * r\nbackward(loss)\n"), "almost", "other-value"),
    ("neuron", _n("z = w1 * x1 + w2 * x2 + b\ny_pred = z.relu()\nr = y_pred + Value(-3)\nloss = r * r\nbackward(loss)\n"), "almost", "other-value"),
    ("neuron", _n("z = w1 * x1 + w2 * x2 + b\ny_pred = z.relu()\nr = y_pred - y\nloss = r * r\nbackward(loss)\n"), "error", "minus"),
    ("neuron", _n("z = w1 * x1 + w2 * x2 + b\ny_pred = z.relu()\nr = y_pred + y * Value(-1)\nloss = r ** 2\nbackward(loss)\n"), "error", "power"),
    ("neuron", _n("z = w1 * x1 + w2 * x2 + b\ny_pred = z.relu()\nr = y_pred + y * Value(-1)\nloss = r * r\nbackward(r)\n"), "wrong", "no-square"),
    ("neuron", _n("z = w1 * x1 + w2 * x2 + b\ny_pred = z.relu()\nloss = y_pred + y * Value(-1)\nbackward(loss)\n"), "wrong", "no-square"),
    ("neuron", _N.replace("backward(loss)\n", "backward(loss)\nb.grad = -4\n"), "wrong", "typed"),
]
