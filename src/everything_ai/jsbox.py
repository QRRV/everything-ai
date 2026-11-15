# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Code boxes for JavaScript. The learner writes a function in JavaScript in a code box; their own browser runs it,
in a web worker beside the page, on the cases of the check, together with the course's solution and every known wrong
version (`JsRun`, an anywidget under the box). Python then compares what came back, as check.FunctionBox does for
Python. The tests have no browser: they run the same harness in Node (`node_runs`).

A value goes from JavaScript to Python as JSON: numbers, strings, true and false, null and lists as themselves; an
object as {"$": "object", "items": ...}; undefined, NaN, Infinity, a promise, a function and an error each as a small
record of its own, which Python turns into `UNDEFINED`, a float, or a `JsError`. Messages show values the way
JavaScript writes them (`js_repr`)."""
import hashlib
import html
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable

import anywidget
import traitlets

from everything_ai.check import MISSING, CodeBox, Mistake, Result, is_empty_text

# --- the harness: the same JavaScript in the browser's worker and in Node ----------------------------------------------

HARNESS = r"""
function eaEncode(v, depth) {
  depth = depth || 0;
  if (v === undefined) return {"$": "undefined"};
  if (v === null) return null;
  if (typeof v === "number") return Number.isFinite(v) ? v : {"$": "number", "text": String(v)};
  if (typeof v === "string" || typeof v === "boolean") return v;
  if (typeof v === "bigint") return {"$": "other", "text": String(v) + "n"};
  if (typeof v === "function") return {"$": "function"};
  if (typeof v.then === "function") return {"$": "promise"};
  if (depth > 20) return {"$": "other", "text": "…"};
  if (Array.isArray(v)) return v.map(x => eaEncode(x, depth + 1));
  if (ArrayBuffer.isView(v)) return {"$": "typed", "type": v.constructor.name, "items": Array.from(v, x => eaEncode(x, depth + 1))};
  const proto = Object.getPrototypeOf(v);
  if (proto === Object.prototype || proto === null) {
    const items = {};
    for (const [k, x] of Object.entries(v)) items[k] = eaEncode(x, depth + 1);
    return {"$": "object", "items": items};
  }
  return {"$": "other", "text": String(v)};
}

function eaError(err) {
  if (!(err instanceof Error)) return {"$": "error", "name": "Thrown", "message": String(err), "line": null};
  const stack = String(err.stack || "");
  const found = stack.match(/<anonymous>:(\d+):\d+/) || stack.match(/> Function:(\d+):\d+/);
  const line = found ? Math.max(1, Number(found[1]) - 2) : null;   // new Function puts two lines before the code
  return {"$": "error", "name": err.name, "message": err.message, "line": line};
}

function eaWithin(promise, ms) {
  let timer;
  const late = new Promise((_, reject) => { timer = setTimeout(() => { const e = new Error("still waiting"); e.name = "EaTimeout"; reject(e); }, ms); });
  return Promise.race([promise, late]).finally(() => clearTimeout(timer));
}

async function eaRunOne(code, name, cases, ms) {
  let fn;
  try {
    // strict, as in a real module, and on the first line, so the line numbers of errors stay the learner's
    const body = "\"use strict\"; " + code.replace(/^([ \t]*)export[ \t]+(default[ \t]+)?/gm, "$1") +
      "\n;return typeof " + name + " === \"function\" ? " + name + " : undefined;";
    fn = new Function(body)();
  } catch (err) {
    return {"compile": eaError(err)};
  }
  if (fn === undefined) return {"missing": true};
  const results = [];
  for (const args of cases) {
    if (results.length && results[results.length - 1]?.name === "EaTimeout") {   // one wait that never ends is enough
      results.push(results[results.length - 1]);
      continue;
    }
    try {
      let v = fn(...structuredClone(args));
      if (v && typeof v.then === "function") v = await eaWithin(v, ms);
      results.push(eaEncode(v));
    } catch (err) {
      results.push(eaError(err));
    }
  }
  return {"results": results};
}

async function eaRunAll(job) {
  const runs = {};
  for (const [key, code] of Object.entries(job.variants)) runs[key] = await eaRunOne(code, job.name, job.cases, job.ms);
  return runs;
}
"""

_WORKER = HARNESS + r"""
self.onmessage = async (e) => { self.postMessage(await eaRunAll(e.data)); };
"""

_NODE = HARNESS + r"""
console.log = console.info = console.debug = console.warn = () => {};   // stdout carries the answer, nothing else
let input = "";
process.stdin.on("data", d => { input += d; });
process.stdin.on("end", async () => { process.stdout.write(JSON.stringify(await eaRunAll(JSON.parse(input)))); });
"""


def node_runs(job: dict, seconds: float = 10.0) -> dict:
    """What the browser's worker gives for `job`, worked out by Node instead: for the tests, which have no browser.
    A run that takes more than `seconds` in all gives {"timeout": True}, as the worker in the browser does."""
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("the checks of a JavaScript box need Node in the tests: https://nodejs.org")
    try:
        done = subprocess.run([node, "-e", _NODE], input=json.dumps(job), capture_output=True, text=True,
                              encoding="utf-8", timeout=seconds, check=True)
    except subprocess.TimeoutExpired:
        return {"timeout": True}
    return json.loads(done.stdout)


# --- values from JavaScript ---------------------------------------------------------------------------------------------

class _Undefined:
    def __repr__(self):
        return "undefined"

    def __eq__(self, other):
        return isinstance(other, _Undefined)

    def __hash__(self):
        return 0


UNDEFINED = _Undefined()


@dataclass(frozen=True)
class JsError:
    """An error a function stopped with in JavaScript, such as TypeError: data.reduce is not a function."""
    name: str
    message: str
    line: int | None = None

    def __str__(self):
        return f"{self.name}: {self.message}"


@dataclass(frozen=True)
class JsOther:
    """A value Python has no form for, such as a promise or a function, by what JavaScript calls it."""
    text: str

    def __repr__(self):
        return self.text


def decode(v):
    """A value as the harness sent it, in Python: see the module's docstring."""
    if isinstance(v, list):
        return [decode(x) for x in v]
    if not isinstance(v, dict):
        return v
    kind = v.get("$")
    if kind == "undefined":
        return UNDEFINED
    if kind == "number":
        return float(v["text"].replace("Infinity", "inf"))
    if kind == "object":
        return {k: decode(x) for k, x in v["items"].items()}
    if kind == "typed":
        return [decode(x) for x in v["items"]]
    if kind == "error":
        return JsError(v["name"], v["message"], v.get("line"))
    if kind == "promise":
        return JsOther("Promise")
    if kind == "function":
        return JsOther("function")
    return JsOther(v.get("text", "?"))


def js_repr(v) -> str:
    """A value as JavaScript writes it in code: [1, 2.5], "text", true, null, {a: 1}, undefined."""
    if v is None:
        return "null"
    if v is True or v is False:
        return "true" if v else "false"
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        return str(int(v)) if v.is_integer() and abs(v) < 1e15 else repr(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(js_repr(x) for x in v) + "]"
    if isinstance(v, dict):
        keys = (k if k.isidentifier() else json.dumps(k, ensure_ascii=False) for k in v)
        return "{" + ", ".join(f"{k}: {js_repr(x)}" for k, x in zip(keys, v.values())) + "}"
    return repr(v)


def _same(a, b, compare) -> bool:
    if isinstance(a, JsError) or isinstance(b, JsError):
        return isinstance(a, JsError) and isinstance(b, JsError) and a.name == b.name
    if isinstance(a, (_Undefined, JsOther)) or isinstance(b, (_Undefined, JsOther)):
        return a == b
    return compare(a, b)


# --- mistakes a learner who knows Python makes in JavaScript -------------------------------------------------------------

def _missing(error) -> str | None:
    """The name a ReferenceError says is not there, in the words of Chrome and Firefox (`len is not defined`) and of
    Safari (`Can't find variable: len`)."""
    if getattr(error, "name", "") != "ReferenceError":
        return None
    m = re.match(r"^(?:Can't find variable: ([\w$]+)|([\w$]+) is not defined)", error.message)
    return (m.group(1) or m.group(2)) if m else None


def missing_name(*words: str) -> Callable:
    """A test on a JsError: the code uses one of `words`, which JavaScript does not have, such as Python's `len`."""
    return lambda e: _missing(e) in words


def assigned_const(error) -> bool:
    """A test on a JsError: the code gives a name made with `const` a new value."""
    return getattr(error, "name", "") == "TypeError" and any(
        w in error.message for w in ("constant variable", "assignment to const", "readonly property"))


def _code_only(text: str) -> str:
    """The code without its strings and comments, so that braces and names in them do not count."""
    return re.sub(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|`(?:\\.|[^`\\])*`|//[^\n]*|/\*.*?\*/', '""', text, flags=re.S)


def undeclared(text: str) -> str | None:
    """The first name the code gives a value without ever making it with let, const or var, nor as a parameter:
    JavaScript in a module stops there, where Python would make the name."""
    code = _code_only(text)
    made = set(re.findall(r"\b(?:let|const|var|function)\s+([A-Za-z_$][\w$]*)", code))
    for group in re.findall(r"\(([^()]*)\)\s*(?:=>|\{)", code):
        made |= set(re.findall(r"[A-Za-z_$][\w$]*", group))
    for name in re.findall(r"^\s*([A-Za-z_$][\w$]*)\s*[-+*/%]?=(?!=)", code, flags=re.M):
        if name not in made and name not in ("let", "const", "var"):
            return name
    return None


def python_mistakes(start: str) -> list:
    """Mistakes the check sees in the text alone, before the browser runs anything: Python's def, Python's way of
    writing a loop or an if, a brace that is missing, and a name used without let."""
    first = start.splitlines()[0]
    return [
        Mistake("python-def", state="error", matches=lambda got, text: re.search(r"^\s*def\s", text, re.M) is not None,
                en=f"That is Python's `def`. In JavaScript a function starts with `function`, as in `{first}`, and its body is between braces.",
                nl=f"Dat is `def` van Python. In JavaScript begint een functie met `function`, zoals in `{first}`, en staat de body tussen accolades."),
        Mistake("python-block", state="error",
                matches=lambda got, text: re.search(r"^\s*(?:(?:for|if|while|elif|else)\b[^{(\n]*:\s*$|for\s+\w+\s+in\s)", _code_only(text), re.M) is not None,
                en="That is how Python writes a loop or an `if`, with a colon. In JavaScript the condition goes between `( )` and the block between braces, as in `for (let i = 0; i < a.length; i++) { ... }` or `if (x > 0) { ... }`.",
                nl="Zo schrijft Python een loop of een `if`, met een dubbele punt. In JavaScript staat de voorwaarde tussen `( )` en het blok tussen accolades, zoals in `for (let i = 0; i < a.length; i++) { ... }` of `if (x > 0) { ... }`."),
        Mistake("braces", state="error",
                matches=lambda got, text: _code_only(text).count("{") != _code_only(text).count("}"),
                en="A brace is missing: every `{` that opens a block needs its own `}` that closes it. Count them in your code.",
                nl="Er ontbreekt een accolade: elke `{` die een blok opent, heeft een eigen `}` nodig die het sluit. Tel ze in je code."),
        Mistake("no-let", state="error", matches=lambda got, text: undeclared(text) is not None,
                en="A new name needs `let` the first time it gets a value, as in `let total = 0;`: JavaScript in a module stops at a name it has never seen made.",
                nl="Een nieuwe naam heeft `let` nodig de eerste keer dat hij een waarde krijgt, zoals in `let total = 0;`: JavaScript in een module stopt bij een naam die nooit gemaakt is."),
    ]


# --- the exercise -------------------------------------------------------------------------------------------------------

_DEFAULTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "waiting": {"en": "Your browser runs your code, in a web worker beside the page …",
                "nl": "Je browser draait je code, in een web worker naast de pagina …"},
    "slow": {"en": "Your code was still not done after {s:g} seconds, so the browser stopped it. Does a loop keep going forever?",
             "nl": "Je code was na {s:g} seconden nog niet klaar, dus de browser stopte hem. Gaat een loop eindeloos door?"},
    "syntax": {"en": "The browser cannot read your code: {error}. Look for a missing bracket, quote or comma{where}.",
               "nl": "De browser kan je code niet lezen: {error}. Zoek een haakje, aanhalingsteken of komma dat ontbreekt{where}."},
    "top": {"en": "Your code stops with an error before the check can call your function: {error}{where}.",
            "nl": "Je code stopt met een fout voordat de check je functie kan aanroepen: {error}{where}."},
    "no_function": {"en": "Your box does not make a function `{function}`. Start it with `function {function}(...) {{`, with that exact name.",
                    "nl": "Je vak maakt geen functie `{function}`. Begin met `function {function}(...) {{`, met precies die naam."},
    "stops": {"en": "With `{call}`, your function stops with an error: {error}{where}.",
              "nl": "Met `{call}` stopt je functie met een fout: {error}{where}."},
    "waits": {"en": "With `{call}`, your function was still waiting after {s:g} seconds: does an `await` wait for something that never comes?",
              "nl": "Met `{call}` wachtte je functie na {s:g} seconden nog: wacht een `await` op iets dat nooit komt?"},
}
_WHERE = {"en": " (line {line})", "nl": " (regel {line})"}


@dataclass
class JsBox(CodeBox):
    """A function in JavaScript in a code box: `function` is its name, `visible` the case of the lesson's text and
    `hidden` the others, each a tuple of arguments that JSON can carry. A mistake's `wrong` is a wrong version in
    JavaScript, which the browser runs too; `error` is a test on a JsError, and `matches(got, text)` one on the
    outcome. Texts as in check.FunctionBox: empty, correct, wrong, almost, and any of _DEFAULTS to say it otherwise."""
    function: str = ""
    visible: list[tuple] = field(default_factory=list)
    hidden: list[tuple] = field(default_factory=list)
    seconds: float = 3.0          # for one call that waits (an await), on one case
    show: Callable[[Any], str] | None = None

    @property
    def limit(self) -> float:
        """Seconds for the whole run, every version on every case: then the browser, or Node, stops it."""
        return max(10.0, 3 * self.seconds)

    def text_of(self, key: str, **values) -> dict:
        t = self.texts.get(key) or _DEFAULTS[key]
        return {lang: s.format(**values) for lang, s in t.items()}

    def _where(self, error: JsError, lang: str) -> str:
        return _WHERE[lang].format(line=error.line) if error.line else ""

    def _error_text(self, key: str, error: JsError, **values) -> dict:
        t = self.texts.get(key) or _DEFAULTS[key]
        return {lang: s.format(error=str(error), where=self._where(error, lang), **values) for lang, s in t.items()}

    def job(self, code: str) -> dict:
        """What the browser runs: the learner's code, the solution and every wrong version, on every case."""
        variants = {"mine": code, "solution": self.solution}
        variants.update({f"wrong:{m.key}": m.wrong for m in self.mistakes if isinstance(m.wrong, str)})
        return {"name": self.function, "cases": [list(c) for c in self.visible + self.hidden], "variants": variants,
                "ms": int(self.seconds * 1000)}

    def token(self, code: str) -> str:
        return hashlib.sha256(json.dumps(self.job(code), sort_keys=True).encode()).hexdigest()[:16]

    def runs(self, value) -> dict | None:
        """What came back for what is in the box: from the browser (`value` is the runner's dict, for the code now in
        the box), or from Node (`value` is the code, in the tests); None while the browser has not run it yet."""
        if isinstance(value, str):
            return node_runs(self.job(value), seconds=self.limit)
        if isinstance(value, dict) and value.get("code") is not None and value.get("token") == self.token(value["code"]):
            return value.get("runs") or {}
        return None

    def _call(self, case) -> str:
        return f"{self.function}({', '.join(js_repr(a) for a in case)})"

    def _shown(self, v) -> str:
        return self.show(v) if self.show else js_repr(v)

    def check(self, value, inputs=None) -> Result:
        if value is MISSING or value is None:
            return Result("error", self.text_of("missing"))
        text = value if isinstance(value, str) else value.get("code") if isinstance(value, dict) else None
        if text is None:
            return Result("waiting", self.text_of("waiting"))
        if is_empty_text(text, self.start):
            cleared = not text.strip() and "cleared" in self.texts
            return Result("empty", self.message("cleared" if cleared else "empty"))
        for m in self.first:
            if m.matches(None, text):
                return Result(m.state, m.text, mistake=m.key)
        runs = self.runs(value)
        if runs is None:
            return Result("waiting", self.text_of("waiting"))
        if runs.get("timeout") or (isinstance(value, dict) and value.get("timeout")):
            return Result("slow", self.text_of("slow", s=self.limit))
        solution = (runs.get("solution") or {}).get("results")
        if runs.get("failed") or solution is None or any(isinstance(decode(r), JsError) for r in solution):
            # the course's own solution did not run in this browser: no verdict on the learner's code
            return Result("broken", {"en": "", "nl": ""})
        mine = runs.get("mine") or {}
        if "compile" in mine:
            error = decode(mine["compile"])
            key = "syntax" if error.name == "SyntaxError" else "top"
            for m in self.mistakes:
                if m.error is not None and m.error(error):
                    return Result("error", {lang: s.format(call="") for lang, s in m.text.items()}, mistake=m.key)
            return Result("error", self._error_text(key, error))
        if mine.get("missing"):
            return Result("wrong", self.text_of("no_function", function=self.function), mistake="no-function")
        got_all = [decode(r) for r in mine["results"]]
        want_all = [decode(r) for r in solution]
        cases = self.visible + self.hidden
        for i, (case, got, want) in enumerate(zip(cases, got_all, want_all)):
            call = self._call(case)
            if isinstance(got, JsError) and not isinstance(want, JsError):
                for m in self.mistakes:
                    if m.error is not None and m.error(got):
                        return Result("error", {lang: s.format(call=call) for lang, s in m.text.items()}, mistake=m.key)
                if got.name == "EaTimeout":
                    return Result("slow", self.text_of("waits", call=call, s=self.seconds))
                return Result("error", self._error_text("stops", got, call=call))
            if _same(got, want, self.compare):
                continue
            for m in self.mistakes:
                if m.error is not None:
                    continue
                if isinstance(m.wrong, str):
                    theirs = runs.get(f"wrong:{m.key}") or {}
                    wrong = [decode(r) for r in theirs.get("results", [])]
                    if len(wrong) == len(got_all) and all(_same(a, b, self.compare) for a, b in zip(got_all, wrong)):
                        return Result(m.state, self._filled(m.text, call, got, want), mistake=m.key)
                elif m.matches is not None and m.matches(got, text):
                    return Result(m.state, self._filled(m.text, call, got, want), mistake=m.key)
            key = "almost" if i >= len(self.visible) else "wrong"
            return Result(key, self.message(key, call=call, got=self._shown(got), want=self._shown(want)))
        return self._correct(text, n=len(cases))

    def _filled(self, text: dict, call: str, got, want) -> dict:
        return {lang: s.format(call=call, got=self._shown(got), want=self._shown(want)) for lang, s in text.items()}


_PICTURE_WORDS = {"en": ("What your {function} gives", "the check calls", "yours", "should be", "stops: {error}"),
                  "nl": ("Wat je {function} geeft", "de check roept aan", "van jou", "zou zijn", "stopt: {error}")}


def outcomes_picture(seen) -> str:
    """For a JsBox: every case of the check, with what the learner's function gave and what it should give, the
    differences in the colour of a mistake."""
    from everything_ai import yours
    ex = seen.exercise
    runs = ex.runs(seen.text) if isinstance(seen.text, (str, dict)) else None
    mine = (runs or {}).get("mine") or {}
    if "results" not in mine or "results" not in ((runs or {}).get("solution") or {}):
        return ""
    w = _PICTURE_WORDS[seen.lang]
    td = 'style="padding:0.2rem 1rem 0.2rem 0;font-family:var(--ea-font-code);vertical-align:top'
    rows = []
    for case, g, r in zip(ex.visible + ex.hidden, mine["results"], runs["solution"]["results"]):
        got, want = decode(g), decode(r)
        ok = _same(got, want, ex.compare)
        shown = w[4].format(error=str(got)) if isinstance(got, JsError) else ex._shown(got)
        colour = "var(--ea-ink)" if ok else "var(--ea-loss-text)"
        rows.append(f'<tr><td {td}">{html.escape(ex._call(case))}</td><td {td};color:{colour}">{html.escape(shown)}</td>'
                    f'<td {td};color:var(--ea-ink-2)">{html.escape(ex._shown(want))}</td></tr>')
    head = "".join(f'<th style="text-align:left;font-weight:500;color:var(--ea-ink-2);padding:0 1rem 0.3rem 0">{html.escape(h)}</th>'
                   for h in w[1:4])
    body = (f'<table style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink);border-collapse:collapse">'
            f'<tr>{head}</tr>{"".join(rows)}</table>')
    title = w[0].format(function=ex.function)
    return yours.figure(title, body, alt=title)


# --- the runner in the browser ------------------------------------------------------------------------------------------

class JsRun(anywidget.AnyWidget):
    """Runs a JsBox's job in a web worker in the learner's browser, and sends back what came out (`value`), with the
    job's token, so that the check knows it belongs to the code now in the box. A line says what it is doing."""
    _esm = r"""
    const WORKER = __WORKER__;
    function render({ model, el }) {
      const line = document.createElement("div"); line.className = "ea-jsrun";
      el.appendChild(line);
      let current = null;   // the run now going: a run that the box replaced finishes into nothing
      function say() {
        const v = model.get("value") || {}, w = model.get("words");
        if (v.token !== model.get("token")) line.textContent = w.running;
        else if (v.timeout) line.textContent = w.stopped;
        else line.textContent = w.ran.replace("{ms}", String(v.ms));
      }
      function stop(run) { clearTimeout(run.timer); if (run.worker) run.worker.terminate(); if (run.url) URL.revokeObjectURL(run.url); }
      function start() {
        const job = model.get("job"), token = model.get("token");
        const v = model.get("value") || {};
        if (current) { stop(current); current = null; }
        if (!job || !job.variants || v.token === token) { say(); return; }
        const run = { began: performance.now() };
        current = run;
        const finish = (out) => {
          if (current !== run) return;
          stop(run); current = null;
          model.set("value", Object.assign({ token, code: job.variants.mine, ms: Math.round(performance.now() - run.began) }, out));
          model.save_changes(); say();
        };
        run.timer = setTimeout(() => finish({ timeout: true }), model.get("limit") * 1000);
        try {
          run.url = URL.createObjectURL(new Blob([WORKER], { type: "text/javascript" }));
          run.worker = new Worker(run.url);
          run.worker.onmessage = (e) => finish({ runs: e.data });
          run.worker.onerror = (e) => { e.preventDefault(); finish({ failed: String(e.message || "the worker stopped") }); };
          run.worker.postMessage(job);
        } catch (err) {
          finish({ failed: String(err) });
        }
        say();
      }
      model.on("change:job", start);
      model.on("change:value", say);
      start();
      return () => { if (current) stop(current); current = null; };
    }
    export default { render };
    """.replace("__WORKER__", json.dumps(_WORKER))
    _css = r"""
    .ea-jsrun { font-family: var(--ea-font-ui, Inter, system-ui, sans-serif); font-size: 0.8rem; color: var(--ea-ink-3); margin: 0.2rem 0 0; }
    """
    job = traitlets.Dict({}).tag(sync=True)
    token = traitlets.Unicode("").tag(sync=True)
    limit = traitlets.Float(10.0).tag(sync=True)
    words = traitlets.Dict({}).tag(sync=True)
    value = traitlets.Dict({}).tag(sync=True)


# --- the demo of the page and a web worker --------------------------------------------------------------------------------

class Freeze(anywidget.AnyWidget):
    """The same work done twice: once on the page itself, once in a web worker beside it. A dot moves along a track
    from a second before the work until a second after it, and the widget measures the longest pause between two frames
    the page drew: on the page the dot stands still while the work runs, beside it the dot keeps moving. `words` has
    "page", "worker", "busy", "result" ({where}, {work} and {gap}, in ms), "where_page", "where_worker" and "frames"."""
    _esm = r"""
    const WORK = `function work(n) { let s = 0; for (let i = 0; i < n; i++) s += Math.sqrt(i); return s; }`;
    function render({ model, el }) {
      const w = model.get("words"), n = model.get("steps");
      const root = document.createElement("div"); root.className = "ea-w ea-freeze";
      const track = document.createElement("div"); track.className = "ea-freeze-track";
      const dot = document.createElement("div"); dot.className = "ea-freeze-dot"; track.appendChild(dot);
      const frames = document.createElement("div"); frames.className = "ea-freeze-frames";
      const row = document.createElement("div"); row.className = "ea-freeze-buttons";
      const onPage = document.createElement("button"); onPage.textContent = w.page;
      const inWorker = document.createElement("button"); inWorker.textContent = w.worker;
      row.append(onPage, inWorker);
      const out = document.createElement("p"); out.className = "ea-freeze-out"; out.setAttribute("aria-live", "polite");
      root.append(track, frames, row, out); el.appendChild(root);
      // the dot moves only during a run that the learner starts: a second before the work, during it, and a second after
      let last = 0, gap = 0, count = 0, measuring = false, moving = false, alive = true, worker = null;
      const locale = model.get("locale") || undefined;
      // the time of a frame comes from performance.now(): the time requestAnimationFrame gives can be from before
      // the page was busy, and would hide the pause
      function frame(now) {
        if (!alive || !moving) return;
        const t = performance.now();
        if (measuring) gap = Math.max(gap, t - last);
        last = t; count += 1;
        dot.style.left = (50 - 45 * Math.cos(now / 400)) + "%";
        frames.textContent = w.frames.replace("{n}", count.toLocaleString(locale));
        requestAnimationFrame(frame);
      }
      dot.style.left = "5%";
      const fmt = (x) => Math.round(x).toLocaleString(locale);
      function begin(work) {
        onPage.disabled = inWorker.disabled = true; out.textContent = w.busy; gap = 0; count = 0;
        moving = true; requestAnimationFrame(frame);
        setTimeout(() => { measuring = true; last = performance.now(); work(); }, 1000);
      }
      function end(where, ms) {
        // one more frame, so that a pause that lasts until the work is done is measured too
        requestAnimationFrame(() => {
          gap = Math.max(gap, performance.now() - last); measuring = false;
          out.textContent = w.result.replace("{where}", where).replace("{work}", fmt(ms)).replace("{gap}", fmt(gap));
          setTimeout(() => { moving = false; onPage.disabled = inWorker.disabled = false; }, 1000);
        });
      }
      onPage.addEventListener("click", () => begin(() => {
        const t = performance.now(); new Function("n", WORK + "; return work(n);")(n); end(w.where_page, performance.now() - t);
      }));
      inWorker.addEventListener("click", () => begin(() => {
        const url = URL.createObjectURL(new Blob([WORK + "; self.onmessage = (e) => { const t = performance.now(); work(e.data); self.postMessage(performance.now() - t); };"], { type: "text/javascript" }));
        worker = new Worker(url);
        const done = () => { worker.terminate(); worker = null; URL.revokeObjectURL(url); };
        worker.onmessage = (e) => { done(); end(w.where_worker, e.data); };
        worker.onerror = (e) => { e.preventDefault(); done(); measuring = false; out.textContent = String(e.message || "?");
                                  moving = false; onPage.disabled = inWorker.disabled = false; };
        worker.postMessage(n);
      }));
      return () => { alive = false; if (worker) worker.terminate(); };
    }
    export default { render };
    """
    _css = r"""
    .ea-freeze { max-width: 34rem; margin: 0.5rem 0; font-family: var(--ea-font-ui, Inter, system-ui, sans-serif); }
    .ea-freeze-track { position: relative; height: 22px; border-radius: 11px; background: var(--ea-inset); }
    .ea-freeze-dot { position: absolute; top: 3px; width: 16px; height: 16px; margin-left: -8px; border-radius: 50%; background: var(--ea-ink-2); }
    .ea-freeze-frames { font-size: 0.8rem; color: var(--ea-ink-3); margin: 0.25rem 0 0.5rem; font-variant-numeric: tabular-nums; }
    .ea-freeze-buttons { display: flex; flex-wrap: wrap; gap: 0.5rem; }
    .ea-freeze button { font: inherit; font-size: 0.9rem; padding: 0.35rem 0.8rem; border-radius: 6px; border: 1px solid var(--ea-line);
      background: var(--ea-page); color: var(--ea-ink); cursor: pointer; }
    .ea-freeze button:hover:not(:disabled) { background: var(--ea-inset); }
    .ea-freeze button:disabled { color: var(--ea-ink-3); cursor: default; }
    .ea-freeze button:focus-visible { outline: 2px solid var(--ea-ink); outline-offset: 2px; }
    .ea-freeze-out { margin: 0.6rem 0 0; color: var(--ea-ink); min-height: 1.4em; }
    """
    steps = traitlets.Int(300_000_000).tag(sync=True)
    locale = traitlets.Unicode("").tag(sync=True)   # how numbers are written: "en-US" or "nl-NL", as the lesson
    words = traitlets.Dict({}).tag(sync=True)
