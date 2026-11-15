# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-model-in-browser: the digit network of torch-mlp-digits as JSON text (as_json, in Python), and its
two layers in JavaScript, which the learner's own browser runs (layer and forward, everything_ai.jsbox); the picture of
the way from PyTorch to the page, and the demo: a digit the learner draws, read by their own code in their browser,
and by PyTorch beside it."""
import html
import json
import math
import re

import anywidget
import numpy as np
import torch
import traitlets
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from everything_ai import paths, yours
from everything_ai.check import FunctionBox, Mistake, kept, run_code
from everything_ai.jsbox import UNDEFINED, JsBox, missing_name, outcomes_picture, python_mistakes

DATA = "data/qmnist-small.npz"   # the first 6,000 training and 1,000 test images of QMNIST: see data/README.md
HIDDEN, LR, EPOCHS, BATCH = 64, 0.5, 10, 64   # as in the demo of torch-mlp-digits


# --- the network: the MLP of torch-mlp-digits, trained the same way ----------------------------------------------------

def _tensors():
    d = np.load(paths.ROOT / DATA)
    return (torch.tensor(d["train_images"], dtype=torch.float32) / 255, torch.tensor(d["train_labels"], dtype=torch.int64),
            torch.tensor(d["test_images"], dtype=torch.float32) / 255, torch.tensor(d["test_labels"], dtype=torch.int64))


class DigitsMLP(nn.Module):
    """The MLP of torch-mlp-digits: 784 pixels, a hidden layer with ReLU, ten logits."""
    def __init__(self, hidden):
        super().__init__()
        self.hidden = nn.Linear(784, hidden)
        self.out = nn.Linear(hidden, 10)

    def forward(self, x):
        return self.out(torch.relu(self.hidden(x)))

    def __repr__(self):   # how a check writes it in a call: as_json(DigitsMLP(3))
        return f"DigitsMLP({self.out.in_features})"


def _trained() -> dict:
    X, y, _, _ = _tensors()
    torch.manual_seed(0)
    model = DigitsMLP(HIDDEN)
    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH, shuffle=True)
    torch.manual_seed(2)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    for _ in range(EPOCHS):
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def network() -> DigitsMLP:
    """The course's digit network: DigitsMLP(64) trained on the 6,000 training digits as the demo of torch-mlp-digits
    trains it, for 10 epochs of mini-batches of 64 with a learning rate of 0.5; the same seeds give the same weights.
    Made once per process, in eval mode."""
    model = DigitsMLP(HIDDEN)
    model.load_state_dict(kept("build-model-in-browser:network", _trained))
    return model.eval()


def accuracy() -> float:
    """The share of the 1,000 test digits the course's network reads right."""
    _, _, Xt, yt = _tensors()
    with torch.no_grad():
        return (network()(Xt).argmax(dim=1) == yt).float().mean().item()


def test_digits(n: int = 1000) -> tuple[list, list]:
    """The first n test digits as lists of 784 numbers from 0 to 1, and their labels: for the tests."""
    _, _, Xt, yt = _tensors()
    return Xt[:n].tolist(), yt[:n].tolist()


def json_megabytes() -> float:
    """The size of the course's network as JSON text, in MB, as the course's as_json writes it."""
    return len(_as_json(network()).encode()) / 1e6


def _as_json(model) -> str:
    return json.dumps({"w1": model.hidden.weight.tolist(), "b1": model.hidden.bias.tolist(),
                       "w2": model.out.weight.tolist(), "b2": model.out.bias.tolist()})


# --- the picture of the idea: from PyTorch to the page ----------------------------------------------------------------

_WAY_WORDS = {
    "en": {"title": "From PyTorch to the page", "model": "DigitsMLP(64)", "model2": "in PyTorch", "json": "JSON text",
           "json2": "w1, b1, w2, b2", "page": "the page", "page2": "forward in JavaScript", "digit": "the digit",
           "dumps": "as_json", "fetch": "fetch", "runs": "in your browser", "once": "once, on your computer"},
    "nl": {"title": "Van PyTorch naar de pagina", "model": "DigitsMLP(64)", "model2": "in PyTorch", "json": "JSON-tekst",
           "json2": "w1, b1, w2, b2", "page": "de pagina", "page2": "forward in JavaScript", "digit": "het cijfer",
           "dumps": "as_json", "fetch": "fetch", "runs": "in je browser", "once": "één keer, op je computer"},
}


def way_picture(lang: str) -> str:
    """Three boxes: the network in PyTorch, its weights as JSON text, and the page that computes with them."""
    w = _WAY_WORDS[lang]
    t = lambda x, y, s, c="var(--ea-ink)", a="middle", size=12, font="var(--ea-font-ui)": (
        f'<text x="{x}" y="{y}" text-anchor="{a}" style="font-size:{size}px;font-family:{font};fill:{c}">{html.escape(s)}</text>')
    arrow = lambda x1, x2: (f'<line x1="{x1}" y1="62" x2="{x2}" y2="62" style="stroke:var(--ea-ink-2);stroke-width:1.5" '
                            f'marker-end="url(#ea-mib-head)"/>')
    parts = [
        '<defs><marker id="ea-mib-head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
        '<path d="M0,0 L10,5 L0,10 z" style="fill:var(--ea-ink-2)"/></marker></defs>',
        '<rect x="0" y="34" width="150" height="56" rx="6" style="fill:var(--ea-inset);stroke:var(--ea-param)"/>',
        t(75, 58, w["model"], font="var(--ea-font-code)"), t(75, 76, w["model2"], "var(--ea-ink-2)", size=11),
        arrow(152, 228), t(190, 54, w["dumps"], "var(--ea-ink-2)", size=11, font="var(--ea-font-code)"),
        '<rect x="230" y="34" width="150" height="56" rx="6" style="fill:var(--ea-inset);stroke:var(--ea-ink-2)"/>',
        t(305, 58, w["json"]), t(305, 76, w["json2"], "var(--ea-ink-2)", size=11, font="var(--ea-font-code)"),
        arrow(382, 458), t(420, 54, w["fetch"], "var(--ea-ink-2)", size=11, font="var(--ea-font-code)"),
        '<rect x="460" y="34" width="150" height="56" rx="6" style="fill:var(--ea-inset);stroke:var(--ea-ink-2)"/>',
        t(535, 58, w["page"]), t(535, 76, w["page2"], "var(--ea-ink-2)", size=11),
        arrow(612, 668), t(700, 66, w["digit"], a="middle"),
        t(190, 20, w["once"], "var(--ea-ink-3)", size=11), t(535, 20, w["runs"], "var(--ea-ink-3)", size=11),
    ]
    svg = (f'<svg viewBox="0 0 740 100" role="img" style="width:100%;max-width:46rem;height:auto">'
           f'<title>{html.escape(w["title"])}</title>{"".join(parts)}</svg>')
    return yours.figure(w["title"], svg, alt=w["title"])


# --- exercise 1: as_json ---------------------------------------------------------------------------------------------

class _NotText:
    """What a function gave that is not text, such as the dict itself: JSON is text."""
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, _NotText) and type(other.value) is type(self.value)

    def __repr__(self):   # the kind of value, as Python names it: int, NoneType
        return type(self.value).__name__


class _NotJson:
    """Text that json.loads cannot read, such as str() of a dict, with its single quotes."""
    def __init__(self, text):
        self.text = text

    def __eq__(self, other):
        return isinstance(other, _NotJson)

    def __repr__(self):   # the start of the text, as Python writes it
        return repr(self.text[:30] + "…" if len(self.text) > 30 else self.text)


def _read(value):
    if not isinstance(value, str):
        return _NotText(value)
    try:
        return json.loads(value)
    except ValueError:
        return _NotJson(value)


def _shape(v) -> str:
    if isinstance(v, list) and v and isinstance(v[0], list):
        return f"{len(v)} × {len(v[0])}"
    return str(len(v)) if isinstance(v, list) else type(v).__name__


def _show_json(v) -> str:
    """Weights as JSON, told by their keys and shapes: the numbers themselves are too many to show."""
    if isinstance(v, dict):
        return "{" + ", ".join(f'"{k}": {_shape(x)}' for k, x in v.items()) + "}"
    if isinstance(v, list):   # a list, told by the shape of each of its first items
        return "[" + ", ".join(_shape(x) for x in v[:4]) + (", …" if len(v) > 4 else "") + "]"
    return repr(v)


def _model(hidden: int, seed: int) -> DigitsMLP:
    torch.manual_seed(seed)
    return DigitsMLP(hidden)


_START_JSON = '''import json


def as_json(model):
    return ""
'''
SOLUTION_JSON = '''import json


def as_json(model):
    weights = {
        "w1": model.hidden.weight.tolist(),
        "b1": model.hidden.bias.tolist(),
        "w2": model.out.weight.tolist(),
        "b2": model.out.bias.tolist(),
    }
    return json.dumps(weights)
'''
JSON_CASES = [(_model(3, 0),), (_model(1, 1),), (_model(5, 2),)]   # 3, 1 and 5 hidden neurons: never 10 or 784
_KEYS = {"w1", "b1", "w2", "b2"}


def _turned(got) -> bool:
    """A weight table turned on its side: "w1" with 784 rows instead of 784 columns, or "w2" with 10 columns instead of
    10 rows."""
    if not (isinstance(got, dict) and set(got) == _KEYS):
        return False
    w1, w2 = got["w1"], got["w2"]
    rows = lambda t: len(t) if isinstance(t, list) else 0
    columns = lambda t: len(t[0]) if isinstance(t, list) and t and isinstance(t[0], list) else 0
    return (rows(w1) == 784 and columns(w1) != 784) or (columns(w2) == 10 and rows(w2) != 10)


def json_picture(seen) -> str:
    """The keys of the learner's JSON and the shape under each, beside what the page needs."""
    run = run_code(seen.text)
    fn = run.names.get("as_json") if run.error is None else None
    if not callable(fn):
        return ""
    try:
        got = _read(fn(_model(3, 0)))
    except Exception:
        return ""
    want = _read(_as_json(_model(3, 0)))
    words = {"en": ("What your as_json gives for DigitsMLP(3)", "key", "yours", "the page needs", "missing"),
             "nl": ("Wat je as_json geeft voor DigitsMLP(3)", "key", "van jou", "de pagina heeft nodig", "ontbreekt")}[seen.lang]
    if not isinstance(got, dict):
        return ""
    td = 'style="padding:0.2rem 1rem 0.2rem 0;font-family:var(--ea-font-code)'
    rows = []
    for k in list(want) + [k for k in got if k not in want]:
        mine = _shape(got[k]) if k in got else words[4]
        should = _shape(want[k]) if k in want else ""
        ok = k in got and k in want and _shape(got[k]) == _shape(want[k])
        colour = "var(--ea-ink)" if ok else "var(--ea-loss-text)"
        rows.append(f'<tr><td {td}">"{html.escape(k)}"</td><td {td};color:{colour}">{html.escape(mine)}</td>'
                    f'<td {td};color:var(--ea-ink-2)">{html.escape(should)}</td></tr>')
    head = "".join(f'<th style="text-align:left;font-weight:500;color:var(--ea-ink-2);padding:0 1rem 0.3rem 0">{h}</th>'
                   for h in words[1:4])
    body = (f'<table style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink);border-collapse:collapse">'
            f'<tr>{head}</tr>{"".join(rows)}</table>')
    return yours.figure(words[0], body, alt=words[0])


EXERCISES = {
    "as_json": FunctionBox(
        name="as_json",
        picture=json_picture,
        function="as_json",
        start=_START_JSON,
        solution=SOLUTION_JSON,
        visible=JSON_CASES[:1],
        hidden=JSON_CASES[1:],
        collect=_read,
        show=_show_json,
        mistakes=[
            Mistake("dict", matches=lambda got, text: isinstance(got, _NotText) and isinstance(got.value, dict),
                    en="`{call}` gives the dict itself. A page reads text: turn the dict into JSON text with `json.dumps`.",
                    nl="`{call}` geeft de dict zelf. Een pagina leest tekst: maak van de dict JSON-tekst met `json.dumps`."),
            Mistake("str", matches=lambda got, text: isinstance(got, _NotJson),
                    en="`{call}` gives text that is not JSON. `str` writes a dict the way Python does, with single quotes, and JSON needs double quotes: use `json.dumps`.",
                    nl="`{call}` geeft tekst die geen JSON is. `str` schrijft een dict zoals Python dat doet, met enkele aanhalingstekens, en JSON heeft dubbele nodig: gebruik `json.dumps`."),
            Mistake("tensor", error=lambda e: isinstance(e, TypeError) and "not JSON serializable" in str(e),
                    en="With `{call}`, `json.dumps` stops: it cannot write a tensor. `.tolist()` makes a tensor a list of Python numbers first, as in `model.hidden.weight.tolist()`.",
                    nl="Met `{call}` stopt `json.dumps`: een tensor kan het niet schrijven. `.tolist()` maakt van een tensor eerst een lijst Python-getallen, zoals in `model.hidden.weight.tolist()`."),
            Mistake("list", matches=lambda got, text: isinstance(got, list),
                    en="`{call}` gives JSON with a list, {got}. The page reads each tensor by its name: put them in a dict, with the keys `\"w1\"`, `\"b1\"`, `\"w2\"` and `\"b2\"`.",
                    nl="`{call}` geeft JSON met een lijst, {got}. De pagina leest elke tensor bij zijn naam: zet ze in een dict, met de keys `\"w1\"`, `\"b1\"`, `\"w2\"` en `\"b2\"`."),
            Mistake("keys", matches=lambda got, text: isinstance(got, dict) and set(got) != _KEYS,
                    en="`{call}` gives JSON with the keys {got}. The page reads `\"w1\"`, `\"b1\"`, `\"w2\"` and `\"b2\"`: the weights and biases of `model.hidden` and `model.out`.",
                    nl="`{call}` geeft JSON met de keys {got}. De pagina leest `\"w1\"`, `\"b1\"`, `\"w2\"` en `\"b2\"`: de weights en biases van `model.hidden` en `model.out`."),
            Mistake("transposed", matches=lambda got, text: _turned(got),
                    en="`{call}` gives {got}: a table of weights turned on its side. In PyTorch, row `j` of `model.hidden.weight` holds the 784 weights of hidden neuron `j`, and row `k` of `model.out.weight` those of output `k`: keep them that way, without `.T`.",
                    nl="`{call}` geeft {got}: een tabel met weights op zijn kant. In PyTorch heeft rij `j` van `model.hidden.weight` de 784 weights van hidden neuron `j`, en rij `k` van `model.out.weight` die van output `k`: houd ze zo, zonder `.T`."),
        ],
        texts={
            "empty": {"en": "`as_json` gives empty text yet. Put the four tensors in a dict, each made a list, and turn the dict into JSON text.",
                      "nl": "`as_json` geeft nog lege tekst. Zet de vier tensors in een dict, elk als lijst, en maak van de dict JSON-tekst."},
            "correct": {"en": "Your `as_json` writes both layers as JSON text, with the same numbers as the tensors, for all {n} networks the check tried.",
                        "nl": "Je `as_json` schrijft beide layers als JSON-tekst, met dezelfde getallen als de tensors, voor alle {n} netwerken die de check probeerde."},
            "wrong": {"en": "`{call}` gives {got}, and the page needs {want}, with the same numbers as the tensors.",
                      "nl": "`{call}` geeft {got}, en de pagina heeft {want} nodig, met dezelfde getallen als de tensors."},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should be {want}.",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en het zou {want} moeten zijn."},
        },
    ),
}


# --- exercise 2: layer, in JavaScript ------------------------------------------------------------------------------------

DOT = '''function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}
'''
_START_LAYER = DOT + '''
function layer(w, b, x) {
  return [];
}
'''
SOLUTION_LAYER = DOT + '''
function layer(w, b, x) {
  const out = [];
  for (let i = 0; i < w.length; i++) {
    out.push(dot(w[i], x) + b[i]);
  }
  return out;
}
'''
LAYER_CASES = [([[1, 2], [3, -1]], [0.5, -2], [2, 1]), ([[0.5, 0.5, 0.5]], [1], [2, 4, -2]),
               ([[-1, 0], [0, -1], [1, 1]], [0, 0, -3], [1, 2]), ([[2]], [0.25], [-3]), ([], [], [1, 2])]


def _counts_inputs(text: str) -> bool:
    """The loop in layer that goes over the neurons stops at x.length, the number of inputs: a counter that runs to
    x.length and picks a row of w or a bias from b. A loop to x.length inside one row, a dot product written out by
    hand, is right."""
    found = re.search(r"function\s+layer\b", text)
    if found is None:
        return False
    body = text[found.start():]
    for loop in re.finditer(r"for\s*\(\s*(?:let|var)?\s*([A-Za-z_$][\w$]*)\s*=[^;]*;\s*\1\s*<=?\s*x\.length", body):
        if re.search(r"(?<![\w$\]])[wb]\s*\[\s*" + re.escape(loop.group(1)) + r"\s*\]", body):
            return True
    return False


EXERCISES["layer"] = JsBox(
    name="layer",
    picture=outcomes_picture,
    function="layer",
    start=_START_LAYER,
    solution=SOLUTION_LAYER,
    visible=LAYER_CASES[:1],
    hidden=LAYER_CASES[1:],
    first=python_mistakes("function layer(w, b, x) {\n") + [
        Mistake("inputs", matches=lambda got, text: _counts_inputs(text),
                en="Your loop in `layer` counts the inputs, `x.length`. A layer gives one number per neuron, so it goes over the rows of `w`: `w.length`.",
                nl="Je loop in `layer` telt de inputs, `x.length`. Een layer geeft één getal per neuron, dus hij gaat langs de rijen van `w`: `w.length`."),
    ],
    mistakes=[
        Mistake("python-names", error=missing_name("len", "range"),
                en="With `{call}`, your function stops: `len` and `range` are Python. In JavaScript the number of rows is `w.length`, and a loop counts with `for (let i = 0; i < w.length; i++)`, as in `dot`.",
                nl="Met `{call}` stopt je functie: `len` en `range` zijn Python. In JavaScript is het aantal rijen `w.length`, en een loop telt met `for (let i = 0; i < w.length; i++)`, zoals in `dot`."),
        Mistake("append", error=lambda e: e.name == "TypeError" and ".append is not a function" in e.message,
                en="With `{call}`, your function stops: a JavaScript array has no `append`. It adds at the end with `push`, as in `out.push(...)`.",
                nl="Met `{call}` stopt je functie: een array in JavaScript heeft geen `append`. Die voegt achteraan toe met `push`, zoals in `out.push(...)`."),
        Mistake("no-bias", wrong=SOLUTION_LAYER.replace("dot(w[i], x) + b[i]", "dot(w[i], x)"),
                en="`{call}` gives `{got}`: the biases are missing. Each neuron adds its own bias, `b[i]`, to its sum.",
                nl="`{call}` geeft `{got}`: de biases ontbreken. Elk neuron telt zijn eigen bias, `b[i]`, bij zijn som op."),
        Mistake("relu", wrong=SOLUTION_LAYER.replace("out.push(dot(w[i], x) + b[i]);", "out.push(Math.max(0, dot(w[i], x) + b[i]));"),
                en="`{call}` gives `{got}`: your layer cuts off what is below 0. That is ReLU, and it comes after the layer, in `forward`; `layer` gives the sums as they are.",
                nl="`{call}` geeft `{got}`: je layer kapt af wat onder 0 is. Dat is ReLU, en dat komt na de layer, in `forward`; `layer` geeft de sommen zoals ze zijn."),
        Mistake("one-number", matches=lambda got, text: isinstance(got, (int, float)) and not isinstance(got, bool),
                en="`{call}` gives one number, `{got}`. A layer gives an array, with one number for each row of `w`.",
                nl="`{call}` geeft één getal, `{got}`. Een layer geeft een array, met één getal voor elke rij van `w`."),
        Mistake("no-return", matches=lambda got, text: got == UNDEFINED,
                en="`{call}` gives `undefined`: your function gives nothing back. End it with `return out;`.",
                nl="`{call}` geeft `undefined`: je functie geeft niets terug. Sluit hem af met `return out;`."),
    ],
    texts={
        "empty": {"en": "`layer` gives an empty array yet. Go over the rows of `w`, and for each, push `dot(w[i], x) + b[i]`.",
                  "nl": "`layer` geeft nog een lege array. Loop langs de rijen van `w`, en push voor elke rij `dot(w[i], x) + b[i]`."},
        "correct": {"en": "Your `layer` gives one sum per neuron, its bias added, for all {n} layers the check tried, one without neurons too.",
                    "nl": "Je `layer` geeft één som per neuron, met zijn bias erbij, voor alle {n} layers die de check probeerde, ook een zonder neuronen."},
        "wrong": {"en": "`{call}` gives `{got}`, and it should be `{want}`.", "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn."},
        "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should be `{want}`.",
                   "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn."},
    },
)


# --- exercise 3: forward, in JavaScript ----------------------------------------------------------------------------------

_START_FORWARD = '''function forward(net, pixels) {
  return [];
}
'''
SOLUTION_FORWARD = '''function forward(net, pixels) {
  const hidden = [];
  for (const v of layer(net.w1, net.b1, pixels)) {
    hidden.push(Math.max(0, v));
  }
  return layer(net.w2, net.b2, hidden);
}
'''
NET = {"w1": [[1, -1], [0.5, 2]], "b1": [0, -3], "w2": [[1, 1], [2, -1], [-1, 0]], "b2": [0, 0.5, -1]}
NET_ONE = {"w1": [[2, 2, 2]], "b1": [-1], "w2": [[1], [-1]], "b2": [0.25, 0]}
FORWARD_CASES = [(NET, [1, 2]), (NET, [0, 0]), (NET, [3, 1]), (NET_ONE, [0.5, 0, 0.25]), (NET_ONE, [0, 0, 0])]

EXERCISES["forward"] = JsBox(
    name="forward",
    picture=outcomes_picture,
    function="forward",
    start=_START_FORWARD,
    solution=SOLUTION_FORWARD,
    before=SOLUTION_LAYER,
    visible=FORWARD_CASES[:1],
    hidden=FORWARD_CASES[1:],
    first=python_mistakes(_START_FORWARD),
    mistakes=[
        Mistake("python-max", error=missing_name("max", "relu", "torch"),
                en="With `{call}`, your function stops: JavaScript has no `max` or `relu` of its own, and no `torch`. There, ReLU is `Math.max(0, v)`.",
                nl="Met `{call}` stopt je functie: JavaScript heeft geen eigen `max` of `relu`, en geen `torch`. Daar is ReLU `Math.max(0, v)`."),
        Mistake("again", error=lambda e: any(w in e.message for w in ("already been declared", "declare a const", "redeclaration")),
                en="The check puts the course's `dot` and `layer` before your code, and your code makes one of those names again with `const` or `let`. Leave them out, and call `layer` as it is.",
                nl="De check zet de `dot` en `layer` van de course vóór je code, en je code maakt een van die namen opnieuw met `const` of `let`. Laat ze weg, en roep `layer` aan zoals hij is."),
        Mistake("append", error=lambda e: e.name == "TypeError" and ".append is not a function" in e.message,
                en="With `{call}`, your function stops: a JavaScript array has no `append`. It adds at the end with `push`, as in `hidden.push(...)`.",
                nl="Met `{call}` stopt je functie: een array in JavaScript heeft geen `append`. Die voegt achteraan toe met `push`, zoals in `hidden.push(...)`."),
        Mistake("for-in", wrong=SOLUTION_FORWARD.replace("for (const v of", "for (const v in"),
                en="`{call}` gives `{got}`: `for (const v in ...)` goes through the positions of an array, as text, not through its values. Write `of`: `for (const v of ...)`.",
                nl="`{call}` geeft `{got}`: `for (const v in ...)` gaat langs de posities van een array, als tekst, niet langs de waarden. Schrijf `of`: `for (const v of ...)`."),
        Mistake("no-relu", wrong=SOLUTION_FORWARD.replace("hidden.push(Math.max(0, v));", "hidden.push(v);"),
                en="`{call}` gives `{got}`: ReLU is missing. A hidden neuron below 0 gives 0: `Math.max(0, v)`, as `max(0, v)` in Python.",
                nl="`{call}` geeft `{got}`: ReLU ontbreekt. Een hidden neuron onder 0 geeft 0: `Math.max(0, v)`, zoals `max(0, v)` in Python."),
        Mistake("relu-out", wrong=SOLUTION_FORWARD.replace("  return layer(net.w2, net.b2, hidden);\n",
                                                           "  const out = [];\n  for (const v of layer(net.w2, net.b2, hidden)) {\n    out.push(Math.max(0, v));\n  }\n  return out;\n"),
                en="`{call}` gives `{got}`: your ReLU cuts off the logits too. As in your `DigitsMLP`, ReLU comes only after the hidden layer; the output layer gives its numbers as they are.",
                nl="`{call}` geeft `{got}`: je ReLU kapt de logits ook af. Zoals in je `DigitsMLP` komt ReLU alleen na de hidden layer; de output layer geeft zijn getallen zoals ze zijn."),
        Mistake("hidden", wrong=SOLUTION_FORWARD.replace("  return layer(net.w2, net.b2, hidden);\n", "  return hidden;\n"),
                en="`{call}` gives `{got}`: what the hidden layer gives. The output layer comes after it, with `net.w2` and `net.b2`.",
                nl="`{call}` geeft `{got}`: wat de hidden layer geeft. Daarna komt de output layer, met `net.w2` en `net.b2`."),
        Mistake("no-return", matches=lambda got, text: got == UNDEFINED,
                en="`{call}` gives `undefined`: your function gives nothing back. End it with `return layer(net.w2, net.b2, hidden);`.",
                nl="`{call}` geeft `undefined`: je functie geeft niets terug. Sluit hem af met `return layer(net.w2, net.b2, hidden);`."),
    ],
    texts={
        "empty": {"en": "`forward` gives an empty array yet. First the hidden layer with ReLU, then the output layer.",
                  "nl": "`forward` geeft nog een lege array. Eerst de hidden layer met ReLU, dan de output layer."},
        "correct": {"en": "Your `forward` computes as your `DigitsMLP` does: the hidden layer, ReLU, the output layer, for all {n} networks and inputs the check tried.",
                    "nl": "Je `forward` rekent zoals je `DigitsMLP`: de hidden layer, ReLU, de output layer, voor alle {n} netwerken en inputs die de check probeerde."},
        "wrong": {"en": "`{call}` gives `{got}`, and it should be `{want}`.", "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn."},
        "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should be `{want}`.",
                   "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn."},
    },
)

REQUIRED = ["as_json", "layer", "forward", "builtin"]

CASES = [
    ("as_json", _START_JSON, "empty", None),
    ("as_json", SOLUTION_JSON, "correct", None),
    ("as_json", "import json\n\n\ndef as_json(model):\n    s = model.state_dict()\n    return json.dumps({\"w1\": s[\"hidden.weight\"].tolist(), \"b1\": s[\"hidden.bias\"].tolist(),\n                       \"w2\": s[\"out.weight\"].tolist(), \"b2\": s[\"out.bias\"].tolist()})\n", "correct", None),
    ("as_json", SOLUTION_JSON.replace("return json.dumps(weights)", "return weights"), "wrong", "dict"),
    ("as_json", SOLUTION_JSON.replace("return json.dumps(weights)", "return str(weights)"), "wrong", "str"),
    ("as_json", SOLUTION_JSON.replace(".tolist()", ""), "error", "tensor"),
    ("as_json", SOLUTION_JSON.replace('"w1"', '"weight1"'), "wrong", "keys"),
    ("as_json", SOLUTION_JSON.replace("model.hidden.weight.tolist()", "model.hidden.weight.T.tolist()"), "wrong", "transposed"),
    ("as_json", SOLUTION_JSON.replace("model.out.weight.tolist()", "model.out.weight.T.tolist()"), "wrong", "transposed"),
    ("as_json", SOLUTION_JSON.replace("weight.tolist()", "weight.T.tolist()"), "wrong", "transposed"),
    ("as_json", "import json\n\n\ndef as_json(model):\n    return json.dumps([model.hidden.weight.tolist(), model.hidden.bias.tolist(),\n                       model.out.weight.tolist(), model.out.bias.tolist()])\n", "wrong", "list"),
    ("as_json", "import json\n\n\ndef as_json(model):\n    return json.dumps(model.hidden.weight.tolist())\n", "wrong", "list"),
    ("as_json", "import json\n\n\ndef as_json(model):\n    return 7\n", "wrong", None),
    ("as_json", SOLUTION_JSON.replace("model.hidden.bias.tolist()", "model.out.bias.tolist()"), "wrong", None),
    ("layer", _START_LAYER, "empty", None),
    ("layer", SOLUTION_LAYER, "correct", None),
    ("layer", DOT + "\nfunction layer(w, b, x) {\n  return w.map((row, i) => dot(row, x) + b[i]);\n}\n", "correct", None),
    ("layer", SOLUTION_LAYER.replace("out.push(", "out.append("), "error", "append"),
    ("layer", SOLUTION_LAYER.replace("i < w.length", "i < len(w)"), "error", "python-names"),
    ("layer", SOLUTION_LAYER.replace("dot(w[i], x) + b[i]", "dot(w[i], x)"), "wrong", "no-bias"),
    ("layer", SOLUTION_LAYER.replace("out.push(dot(w[i], x) + b[i]);", "out.push(Math.max(0, dot(w[i], x) + b[i]));"), "wrong", "relu"),
    ("layer", SOLUTION_LAYER.replace("i < w.length", "i < x.length"), "wrong", "inputs"),
    ("layer", DOT + "\nfunction layer(w, b, x) {\n  const out = [];\n  for (let i = 0; i < w.length; i++) {\n    let sum = b[i];\n    for (let j = 0; j < x.length; j++) {\n      sum += w[i][j] * x[j];\n    }\n    out.push(sum);\n  }\n  return out;\n}\n", "correct", None),
    ("layer", DOT + "\nfunction layer(w, b, x) {\n  return dot(w[0], x) + b[0];\n}\n", "wrong", "one-number"),
    ("layer", SOLUTION_LAYER.replace("  return out;\n", ""), "wrong", "no-return"),
    ("layer", SOLUTION_LAYER.replace("dot(w[i], x) + b[i]", "dot(w[i], x) - b[i]"), "wrong", None),
    ("layer", SOLUTION_LAYER.replace("const out = [];", "out = [];"), "error", "no-let"),
    ("layer", DOT + "\ndef layer(w, b, x):\n    return [dot(row, x) + bias for row, bias in zip(w, b)]\n", "error", "python-def"),
    ("layer", SOLUTION_LAYER.replace("for (let i = 0; i < w.length; i++) {", "for i in range(len(w)):"), "error", "python-block"),
    ("layer", SOLUTION_LAYER[:-2], "error", "braces"),
    ("layer", DOT + "\nfunction layer(w, b, x) {\n  while (true) {}\n}\n", "slow", None),
    ("layer", SOLUTION_LAYER.replace("function layer(", "function dense("), "wrong", "no-function"),
    ("forward", _START_FORWARD, "empty", None),
    ("forward", SOLUTION_FORWARD, "correct", None),
    ("forward", "function forward(net, pixels) {\n  const hidden = layer(net.w1, net.b1, pixels).map(v => Math.max(0, v));\n  return layer(net.w2, net.b2, hidden);\n}\n", "correct", None),
    ("forward", SOLUTION_LAYER + "\n" + SOLUTION_FORWARD, "correct", None),
    ("forward", SOLUTION_FORWARD.replace("hidden.push(Math.max(0, v));", "hidden.push(v);"), "wrong", "no-relu"),
    ("forward", SOLUTION_FORWARD.replace("Math.max(0, v)", "max(0, v)"), "error", "python-max"),
    ("forward", SOLUTION_FORWARD.replace("  return layer(net.w2, net.b2, hidden);\n",
                                         "  const out = [];\n  for (const v of layer(net.w2, net.b2, hidden)) {\n    out.push(Math.max(0, v));\n  }\n  return out;\n"), "wrong", "relu-out"),
    ("forward", SOLUTION_FORWARD.replace("  return layer(net.w2, net.b2, hidden);\n", "  return hidden;\n"), "wrong", "hidden"),
    ("forward", SOLUTION_FORWARD.replace("  return layer(net.w2, net.b2, hidden);\n", "  layer(net.w2, net.b2, hidden);\n"), "wrong", "no-return"),
    ("forward", SOLUTION_FORWARD.replace("Math.max(0, v)", "Math.relu(v)"), "error", None),
    ("forward", SOLUTION_FORWARD.replace("for (const v of", "for (const v in"), "wrong", "for-in"),
    ("forward", SOLUTION_FORWARD.replace("hidden.push(", "hidden.append("), "error", "append"),
    ("forward", "const layer = (w, b, x) => w.map((row, i) => dot(row, x) + b[i]);\n\n" + SOLUTION_FORWARD, "error", "again"),
    ("forward", "def forward(net, pixels):\n    return []\n", "error", "python-def"),
]


# --- the demo: a digit you draw, read by your own code in your browser ----------------------------------------------------

class DrawDigit(anywidget.AnyWidget):
    """A square to draw a digit in. After every stroke the drawing becomes 28 × 28 pixels, the way QMNIST's digits
    were made: the digit scaled to fit 20 × 20 and moved so that its centre of mass is in the middle. The learner's own
    `code` (their layer and forward) runs in a web worker with the network `net` (JSON text) on those pixels; the widget
    shows the ten logits as bars and the largest, and sends back the pixels and the logits (`value`), so that Python can
    read the same pixels with PyTorch. `words` has "clear", "draw", "reads", "stopped" and "error"."""
    _esm = r"""
    const SIZE = 280;
    function render({ model, el }) {
      const w = model.get("words"), locale = model.get("locale") || undefined;
      const root = document.createElement("div"); root.className = "ea-w ea-draw";
      const pad = document.createElement("canvas"); pad.width = SIZE; pad.height = SIZE; pad.className = "ea-draw-pad";
      pad.setAttribute("aria-label", w.draw); pad.setAttribute("role", "img");
      const side = document.createElement("div"); side.className = "ea-draw-side";
      const seen = document.createElement("canvas"); seen.width = 28; seen.height = 28; seen.className = "ea-draw-seen";
      seen.setAttribute("aria-hidden", "true");   // the same digit, small: the line under it says what the code reads
      const reads = document.createElement("p"); reads.className = "ea-draw-reads"; reads.setAttribute("aria-live", "polite");
      const bars = document.createElement("div"); bars.className = "ea-draw-bars";
      const clear = document.createElement("button"); clear.textContent = w.clear;
      side.append(seen, reads, bars, clear); root.append(pad, side); el.appendChild(root);
      const ctx = pad.getContext("2d", { willReadFrequently: true });
      let drawing = false, worker = null, url = null, timer = null;
      function stopWorker() { clearTimeout(timer); if (worker) worker.terminate(); worker = null; if (url) URL.revokeObjectURL(url); url = null; }
      function forget() { bars.textContent = ""; model.set("value", {}); model.save_changes(); }   // no old reading stays
      reads.textContent = w.draw;
      // the ink is the page's own ink, in light and in dark; only how much ink a pixel has counts
      function stroke() {
        ctx.lineWidth = 20; ctx.lineCap = "round"; ctx.lineJoin = "round";
        ctx.strokeStyle = getComputedStyle(pad).color;
      }
      function at(e) { const r = pad.getBoundingClientRect(); return [(e.clientX - r.left) * SIZE / r.width, (e.clientY - r.top) * SIZE / r.height]; }
      pad.addEventListener("pointerdown", (e) => {
        drawing = true; pad.setPointerCapture(e.pointerId); stroke();
        const [x, y] = at(e); ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 0.1, y); ctx.stroke();
      });
      pad.addEventListener("pointermove", (e) => { if (!drawing) return; const [x, y] = at(e); ctx.lineTo(x, y); ctx.stroke(); });
      const up = () => { if (!drawing) return; drawing = false; read(); };
      pad.addEventListener("pointerup", up); pad.addEventListener("pointercancel", up);
      clear.addEventListener("click", () => {
        stopWorker(); ctx.clearRect(0, 0, SIZE, SIZE); seen.getContext("2d").clearRect(0, 0, 28, 28);
        forget(); reads.textContent = w.draw;
      });
      function pixels() {
        // the drawing as QMNIST made its digits: into a 20 x 20 box, then the centre of mass in the middle of 28 x 28
        const a = ctx.getImageData(0, 0, SIZE, SIZE).data;
        let x0 = SIZE, y0 = SIZE, x1 = -1, y1 = -1;
        for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) if (a[(y * SIZE + x) * 4 + 3] > 0) {
          if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
        }
        if (x1 < 0) return null;
        const bw = x1 - x0 + 1, bh = y1 - y0 + 1, scale = 20 / Math.max(bw, bh);
        const box = document.createElement("canvas"); box.width = 28; box.height = 28;
        const b = box.getContext("2d", { willReadFrequently: true }); b.imageSmoothingEnabled = true; b.imageSmoothingQuality = "high";
        const dw = bw * scale, dh = bh * scale;
        b.drawImage(pad, x0, y0, bw, bh, (28 - dw) / 2, (28 - dh) / 2, dw, dh);
        let d = b.getImageData(0, 0, 28, 28).data, m = 0, mx = 0, my = 0;
        for (let i = 0; i < 784; i++) { const v = d[i * 4 + 3] / 255; m += v; mx += v * (i % 28); my += v * Math.floor(i / 28); }
        const out = document.createElement("canvas"); out.width = 28; out.height = 28;
        const o = out.getContext("2d", { willReadFrequently: true });
        o.drawImage(box, Math.round(13.5 - mx / m), Math.round(13.5 - my / m));
        d = o.getImageData(0, 0, 28, 28).data;
        const p = []; for (let i = 0; i < 784; i++) p.push(d[i * 4 + 3] / 255);
        const s = seen.getContext("2d");
        s.clearRect(0, 0, 28, 28); s.fillStyle = getComputedStyle(seen).color;
        for (let i = 0; i < 784; i++) if (p[i] > 0) { s.globalAlpha = p[i]; s.fillRect(i % 28, Math.floor(i / 28), 1, 1); }
        s.globalAlpha = 1;
        return p;
      }
      function show(scores) {
        let best = 0; for (let k = 1; k < scores.length; k++) if (scores[k] > scores[best]) best = k;
        reads.textContent = w.reads.replace("{digit}", String(best));
        const lo = Math.min(0, ...scores), hi = Math.max(0, ...scores), span = (hi - lo) || 1;
        bars.textContent = "";
        scores.forEach((v, k) => {
          const row = document.createElement("div"); row.className = "ea-draw-row" + (k === best ? " ea-draw-best" : "");
          const label = document.createElement("span"); label.textContent = String(k);
          const track = document.createElement("span"); track.className = "ea-draw-track";
          const bar = document.createElement("span"); bar.className = "ea-draw-bar";
          bar.style.left = ((Math.min(v, 0) - lo) / span * 100) + "%"; bar.style.width = (Math.abs(v) / span * 100) + "%";
          const num = document.createElement("span"); num.className = "ea-draw-num"; num.textContent = v.toLocaleString(locale, { minimumFractionDigits: 1, maximumFractionDigits: 1 });
          track.appendChild(bar); row.append(label, track, num); bars.appendChild(row);
        });
      }
      function read() {
        const p = pixels(); if (!p) return;
        const code = model.get("code"), net = model.get("net");
        if (!code || !net) return;
        stopWorker();
        const body = code + "\n;self.onmessage = (e) => { let out; try { out = { scores: Array.from(forward(e.data.net, e.data.pixels), Number) }; }" +
                     " catch (err) { out = { error: String(err) }; } self.postMessage(out); };";
        url = URL.createObjectURL(new Blob([body], { type: "text/javascript" }));
        worker = new Worker(url);
        worker.onmessage = (e) => {
          stopWorker();
          if (e.data.error) { forget(); reads.textContent = w.error.replace("{error}", e.data.error); return; }
          show(e.data.scores);
          model.set("value", { pixels: p, scores: e.data.scores }); model.save_changes();
        };
        worker.onerror = (e) => { e.preventDefault(); stopWorker(); forget(); reads.textContent = w.error.replace("{error}", e.message || "?"); };
        timer = setTimeout(() => { stopWorker(); forget(); reads.textContent = w.stopped; }, 3000);
        worker.postMessage({ net: JSON.parse(net), pixels: p });
      }
      return () => { stopWorker(); };
    }
    export default { render };
    """
    _css = r"""
    .ea-draw { display: flex; flex-wrap: wrap; gap: 1.25rem; align-items: flex-start; margin: 0.5rem 0;
      font-family: var(--ea-font-ui, Inter, system-ui, sans-serif); color: var(--ea-ink); }
    .ea-draw-pad { width: 280px; max-width: 100%; aspect-ratio: 1; border: 1px solid var(--ea-line); border-radius: 8px;
      background: var(--ea-page); color: var(--ea-ink); touch-action: none; cursor: crosshair; }
    .ea-draw-side { flex: 1 1 14rem; min-width: 12rem; }
    .ea-draw-seen { width: 84px; height: 84px; image-rendering: pixelated; border: 1px solid var(--ea-line);
      background: var(--ea-page); color: var(--ea-ink); }
    .ea-draw-reads { margin: 0.4rem 0 0.6rem; font-size: 1rem; min-height: 1.4em; }
    .ea-draw-row { display: grid; grid-template-columns: 1rem 1fr 3rem; gap: 0.5rem; align-items: center; font-size: 12px;
      font-variant-numeric: tabular-nums; color: var(--ea-ink-2); margin: 2px 0; }
    .ea-draw-best { color: var(--ea-ink); font-weight: 600; }
    .ea-draw-track { position: relative; height: 10px; background: var(--ea-inset); border-radius: 2px; }
    .ea-draw-bar { position: absolute; top: 0; bottom: 0; background: var(--ea-ink-3); border-radius: 2px; }
    .ea-draw-best .ea-draw-bar { background: var(--ea-ink); }
    .ea-draw-num { text-align: right; }
    .ea-draw button { font: inherit; font-size: 0.9rem; margin-top: 0.6rem; padding: 0.35rem 0.8rem; border-radius: 6px;
      border: 1px solid var(--ea-line); background: var(--ea-page); color: var(--ea-ink); cursor: pointer; }
    .ea-draw button:hover { background: var(--ea-inset); }
    .ea-draw button:focus-visible { outline: 2px solid var(--ea-ink); outline-offset: 2px; }
    """
    code = traitlets.Unicode("").tag(sync=True)
    net = traitlets.Unicode("").tag(sync=True)
    locale = traitlets.Unicode("").tag(sync=True)   # how numbers are written: "en-US" or "nl-NL", as the lesson
    words = traitlets.Dict({}).tag(sync=True)
    value = traitlets.Dict({}).tag(sync=True)


def learner_json(text: str) -> str:
    """The course's network as JSON text, written by the learner's own as_json, which the check found right: the demo
    sends it to the page, as a site would put it next to its page."""
    return run_code(text).names["as_json"](network())


def draw(code: str, net: str, words: dict, lang: str) -> DrawDigit:
    """The pad of the demo, with the learner's JavaScript (`code`), the network as JSON text (`net`) and the words.
    The worker runs the code as a plain script, so an `export` goes, as in the check (jsbox.HARNESS)."""
    code = re.sub(r"(?m)^([ \t]*)export[ \t]+(default[ \t]+)?", r"\1", code)
    return DrawDigit(code=code, net=net, words=words, locale={"en": "en-US", "nl": "nl-NL"}[lang])


def less_than(gap: float) -> tuple[float, int]:
    """The smallest power of ten above `gap`, with its number of decimals: the logits of PyTorch (float32) and of
    JavaScript (float64) differ by less than 0.00001, and some digits by less than 0.0000001."""
    power = math.floor(math.log10(gap)) + 1 if gap > 0 else -9
    return 10.0 ** power, max(0, -power)


def torch_reads(pixels: list) -> tuple[int, list]:
    """What the course's network in PyTorch makes of the same 784 pixels: its digit and its ten logits."""
    with torch.no_grad():
        logits = network()(torch.tensor([pixels], dtype=torch.float32))[0]
    return int(logits.argmax()), [float(v) for v in logits]
