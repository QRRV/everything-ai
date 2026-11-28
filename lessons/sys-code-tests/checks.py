# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-code-tests: tests for the code of the learner's project ops. test_one_row is a unit test
of `inputs` on one row whose answer is known; impossible is a model assertion that flags predictions that cannot be
right; test_overfit_one_batch trains a fresh DemandNet on 16 hours and asks the loss to come close to 0. A test is
right when it passes for the course's version of the code and fails for each version with a known bug, as a test
should on code with that bug. The demo trains three versions of train_epoch on the 16 hours and runs the learner's
test on each.
The data are the hourly counts of Capital Bikeshare in Washington, D.C., in 2011, from the Bike Sharing dataset of
Fanaee-T and Gama (2013, CC BY 4.0, see data/README.md); the bugs, the tests and the pictures are the course's own."""
import ast
import html
import importlib.abc
import importlib.util
import math
import re
import sys
import types
from dataclasses import dataclass, field

import torch

from everything_ai import chart, live, paths, projects, yours
from everything_ai.check import (MISSING, CodeBox, FunctionBox, Mistake, Result, Timeout, is_empty_text, run_code,
                                 with_time_limit)

LESSON = "sys-code-tests"
LANGS = ("en", "nl")

projects.provide("sys-project-tracked-model")   # ops/demand.py, which that lesson gives, even when only the checks load


def _course_demand():
    """The course's own copy of ops/demand.py: the checks swap it in, so they stay the same when you change yours."""
    path = paths.LESSONS / "sys-project-tracked-model" / "project" / "ops" / "demand.py"
    spec = importlib.util.spec_from_file_location("_course_ops_demand_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


demand = _course_demand()

HOURS_2011 = demand.read_hours()                       # 8,645 hours of 365 days
FEW = HOURS_2011.sample(16, random_state=0)            # the 16 hours of the test that overfits one batch
X16 = demand.inputs(FEW, demand.scaling(HOURS_2011))
Y16 = demand.targets(FEW)
STEPS = 400
LR = 0.03
DEMO_LIMIT = 60.0   # seconds the demo gets


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma and a thousands dot in Dutch, and a real minus sign."""
    out = f"{x:,.{digits}f}"
    out = out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out
    return out.replace("-", "−")


def _lang(lang: str) -> str:
    return lang if lang in LANGS else "en"


def _chart(panels: list, alt: str, lang: str, width: int = 330, height: int = 230) -> str:
    out = chart.figure(panels, alt=alt, width=width, height=height)
    return chart.with_decimal_comma(out) if lang == "nl" else out


# ======== the versions with a bug ========================================================================================

def _inputs(shift: int = 0, mean: str = "scale", hour_first: bool = True):
    """A version of `inputs` from ops/demand.py: the course's, or one with a bug."""
    def inputs(hours, scale):
        hour = torch.nn.functional.one_hot(torch.tensor((hours["hr"].to_numpy() + shift) % 24), 24).float()
        values = torch.tensor(hours[demand.FEATURES].to_numpy(dtype="float32"))
        center = {"scale": torch.tensor(scale["mean"]), "none": 0.0}.get(mean, values.mean(0))
        scaled = (values - center) / torch.tensor(scale["std"])
        return torch.cat([hour, scaled] if hour_first else [scaled, hour], dim=1)
    return inputs


def _train_epoch(column: bool = False, zero: bool = True, step: bool = True):
    """A version of `train_epoch` from ops/demand.py: the course's, or one with a bug."""
    def train_epoch(model, opt, x, y, batch):
        order = torch.randperm(len(x))
        for i in range(0, len(x), batch):
            part = order[i:i + batch]
            target = y[part].reshape(-1) if column else y[part]
            loss = ((model(x[part]) - target) ** 2).mean()
            if zero:
                opt.zero_grad()
            loss.backward()
            if step:
                opt.step()
    return train_epoch


INPUT_VERSIONS = [("right", demand.inputs), ("shift", _inputs(shift=1)), ("order", _inputs(hour_first=False)),
                  ("no-mean", _inputs(mean="none")), ("own-mean", _inputs(mean="own"))]
TRAIN_VERSIONS = [("right", demand.train_epoch), ("no-step", _train_epoch(step=False)),
                  ("column", _train_epoch(column=True)), ("no-zero-grad", _train_epoch(zero=False))]

VERSION_NAMES = {
    "en": {"right": "the right one", "shift": "hour 8 in column 9", "order": "features first",
           "no-mean": "mean not subtracted", "own-mean": "mean of its own hours",
           "no-step": "no opt.step()", "column": "y as a row", "no-zero-grad": "no opt.zero_grad()"},
    "nl": {"right": "de goede", "shift": "uur 8 in kolom 9", "order": "features eerst",
           "no-mean": "gemiddelde niet eraf", "own-mean": "gemiddelde van de eigen uren",
           "no-step": "geen opt.step()", "column": "y als rij", "no-zero-grad": "geen opt.zero_grad()"},
}


# ======== running the learner's tests ====================================================================================

def _tests_in(text: str) -> list:
    """The test functions the text defines, as pytest finds them: `def test_...():` at the top, without arguments."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    return [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test")
            and not n.args.args]


def _swapped(text: str, swap: dict) -> str:
    """The text, then the names in `swap` in place of the ones it imported, then a call of each test function."""
    lines = "".join(f"{name} = __swap__[{name!r}]\n" for name in swap)
    return text.rstrip("\n") + "\n" + lines + "".join(f"{name}()\n" for name in _tests_in(text))


def _base(names: tuple) -> dict:
    return {name: getattr(demand, name) for name in names}


class _Serve(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serves one module object under the name ops.demand to every import during a run."""

    def __init__(self, module):
        self.module = module

    def find_spec(self, name, path, target=None):
        return importlib.util.spec_from_loader(name, self) if name == "ops.demand" else None

    def create_module(self, spec):
        return self.module

    def exec_module(self, module):
        pass


def _swap_module(swap: dict):
    """For one run, ops.demand is the course's copy with the names in `swap`: also for a test that imports inside its
    function, or imports the module itself (`from ops import demand`, `import ops.demand as d`). run_code forgets the
    project's modules before it runs, so the copy is served by a finder at the front of sys.meta_path. Gives back a
    function that takes the finder away and forgets the copy, so that the lesson's own cells import your file again."""
    fake = types.ModuleType("ops.demand")
    fake.__dict__.update({k: v for k, v in vars(demand).items() if not k.startswith("__")})
    fake.__dict__.update(swap)
    finder = _Serve(fake)
    sys.meta_path.insert(0, finder)

    def back():
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        if sys.modules.get("ops.demand") is fake:
            del sys.modules["ops.demand"]
        package = sys.modules.get("ops")
        if package is not None and package.__dict__.get("demand") is fake:
            del package.__dict__["demand"]
    return back


def _run_tests(text: str, swap: dict):
    back = _swap_module(swap)
    try:
        return run_code(_swapped(text, swap), {"__swap__": swap})
    finally:
        back()


def _outside(text: str) -> str | None:
    """The first `assert` outside every function, on one line, or None: Python runs it before the tests."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assert):
            return _one_line(ast.get_source_segment(text, node) or "")
    return None


def _verdict(run) -> bool | None:
    """True when every test passed, False when one failed or stopped with an error, None when it took too long."""
    if run.timeout:
        return None
    return run.error is None


def _one_line(code: str) -> str:
    return " ".join(part.strip() for part in code.splitlines() if part.strip())


def _statement_at(text: str, line: int | None) -> str:
    """The whole statement that holds `line`, on one line: also an assert written over several lines."""
    if not line:
        return ""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text.splitlines()[line - 1].strip() if 0 < line <= len(text.splitlines()) else ""
    inner = [n for n in ast.walk(tree) if isinstance(n, ast.stmt) and not isinstance(n, ast.FunctionDef)
             and n.lineno <= line <= (n.end_lineno or n.lineno)]
    if not inner:
        return ""
    node = min(inner, key=lambda n: (n.end_lineno or n.lineno) - n.lineno)
    return _one_line(ast.get_source_segment(text, node) or "")


_LAST: dict = {}   # per exercise, the text the check ran last and how its tests did on each version: for the picture


@dataclass
class TestsBox(CodeBox):
    """The learner writes a test, `def test_...():` with lines that assert, of one function of ops/demand.py. The check
    runs it on the course's version of that function, where it must pass, and on each version with a known bug, where
    it must fail: by a failing assert, or by an error, as pytest counts it. Before it runs, the names the test imports
    from ops.demand become the course's copies, and the function under test the version tried. `versions` is a list of
    (key, function): the right one first. `bugs` gives per key of a bug the text when the test passes for it.
    `on_right` are mistakes in a line that fails for the right version, each with `matches(None, line)`. Texts:
    missing, empty, cleared, no_test, fails_right ({line}), slow, correct, kept."""
    function: str = ""
    swaps: tuple = ()
    versions: list = field(default_factory=list)
    bugs: dict = field(default_factory=dict)
    on_right: list = field(default_factory=list)

    def tries(self, text: str) -> list:
        """How the tests in `text` do on each version, as (key, verdict); worked out once per text."""
        last = _LAST.get(self.name)
        if last is not None and last[0] == text:
            return last[1]
        base = _base(self.swaps)
        out = []
        for key, fn in self.versions:
            out.append((key, _run_tests(text, {**base, self.function: fn})))
        if not any(run.timeout for _, run in out):   # a run that took too long is tried again next time
            _LAST[self.name] = (text, out)
        return out

    def check(self, text, inputs=None) -> Result:
        if text is MISSING or text is None:
            return Result("error", self.message("missing"))
        if is_empty_text(text, self.start):
            cleared = text.strip() != self.start.strip() and "cleared" in self.texts
            return Result("empty", self.message("cleared" if cleared else "empty"))
        for m in self.first:
            if m.matches(None, text):
                return Result(m.state, m.text, mistake=m.key)
        try:
            ast.parse(text)
            broken = False
        except SyntaxError:
            broken = True
        for m in self.mistakes if broken else []:   # a line such as `f(a=1)` or a quote in a comment is fine Python
            found = re.search(m.syntax, text, flags=re.MULTILINE) if m.syntax else None
            if found:
                line = text[text.rfind("\n", 0, found.start()) + 1:].split("\n", 1)[0].strip()
                return Result("error", {k: v.replace("{line}", line) for k, v in m.text.items()}, mistake=m.key)
        if not _tests_in(text):
            try:
                ast.parse(text)
            except SyntaxError:
                return Result("error", self._plain_error(run_code(text), text))
            return Result("wrong", self.message("no_test"), mistake="no-test")
        outside = _outside(text)
        if outside is not None:
            return Result("wrong", self.message("outside", line=outside), mistake="outside")
        runs = self.tries(text)
        good = runs[0][1]
        if good.timeout:
            return Result("slow", self.message("slow"))
        if isinstance(good.error, AssertionError):
            line = _statement_at(text, good.line)
            for m in self.on_right:
                if m.matches(None, line):
                    return Result("wrong", {k: v.replace("{line}", line) for k, v in m.text.items()}, mistake=m.key)
            return Result("wrong", self.message("fails_right", line=line), mistake="fails-right")
        if good.error is not None:
            return Result("error", self._plain_error(good, text))
        for key, bad in runs[1:]:
            if bad.timeout:
                return Result("slow", self.message("slow"))
            if bad.error is None:
                return Result("wrong", self.bugs[key], mistake="misses-" + key)
        return self._correct(text)


def tests_picture(seen) -> str:
    """The learner's test on each version of the function: passes or fails, in ink where that is what a good test does
    (pass on the right one, fail on one with a bug), and in the color of mistakes where it is not."""
    ex = seen.exercise
    if not isinstance(seen.text, str) or not _tests_in(seen.text):
        return ""
    lang = _lang(seen.lang)
    marks = {True: {"en": "passes", "nl": "slaagt"}, False: {"en": "fails", "nl": "faalt"},
             None: {"en": "too slow", "nl": "te traag"}}
    rows = []
    for key, run in ex.tries(seen.text):
        verdict = _verdict(run)
        good = verdict is (key == "right")
        color = "var(--ea-ink)" if good else "var(--ea-loss-text)"
        name = html.escape(VERSION_NAMES[lang][key])
        name = re.sub(r"(opt\.\w+\(\))", r'<code style="font-variant-ligatures:none">\1</code>', name)
        rows.append(f'<tr><td style="padding:0.1rem 1rem 0.1rem 0">{name}</td>'
                    f'<td style="padding:0.1rem 0;color:{color};font-weight:600">{marks[verdict][lang]}</td></tr>')
    head = {"en": [f"version of {ex.function}", "your test"], "nl": [f"versie van {ex.function}", "je test"]}[lang]
    table = ('<table style="font-size:0.85rem;border-collapse:collapse"><tr>'
             + "".join(f'<th style="text-align:left;font-weight:400;color:var(--ea-ink-2);padding:0 1rem 0.2rem 0">'
                       f'{html.escape(h)}</th>' for h in head)
             + f'</tr>{"".join(rows)}</table>')
    n = len(ex.versions)
    title = {"en": f"Your test on {n} versions of {ex.function}", "nl": f"Je test op {n} versies van {ex.function}"}[lang]
    return yours.figure(title, table)


_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then begins with the course's lines again.",
                "nl": "Alle code in het vak is weg. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
    "outside": {"en": "The line `{line}` is not indented, so it is not part of a test: Python runs it once, before the tests, and pytest does not count it as a test. Can you indent it like the other lines of the test?",
                "nl": "De regel `{line}` is niet ingesprongen, dus hoort de regel bij geen enkele test: Python voert de regel één keer uit, nog voordat de tests draaien, en pytest telt de regel niet als test. Kun je de regel net zo ver laten inspringen als de andere regels van de test?"},
}

_SYNTAX = [
    Mistake("curly-quote", syntax=r"[“”‘’„]",
            en="Your code has a curly quote, like `“`. Python only reads straight quotes, like `\"`: can you type the quotes again in the box?",
            nl="Je code heeft een gekruld aanhalingsteken, zoals `“`. Python leest alleen rechte, zoals `\"`: kun je de aanhalingstekens opnieuw typen in het vak?"),
    Mistake("one-equals", syntax=r"^[ \t]*assert\b(?![^\n]*(?:==|[<>!]=))[^\n]*[\w\])][ \t]*=[^=\n]*$",
            en="The line `{line}` has one `=`. That puts a value in a name; to ask whether two things are equal, Python uses `==`. Can you write `==`?",
            nl="De regel `{line}` heeft één `=`. Daarmee zet je een waarde in een naam; om te vragen of twee dingen gelijk zijn, gebruikt Python `==`. Kun je `==` schrijven?"),
]


def _kept(name: str) -> dict:
    return {"en": f"Your `{name}` is right. You changed `{{saved}}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
            "nl": f"Je `{name}` klopt. Je hebt `{{saved}}` zelf veranderd, dus laat de check dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."}


def _dots_left(got, text_) -> bool:
    try:
        tree = ast.parse(text_)
    except SyntaxError:
        return "..." in text_
    return any(isinstance(n, ast.Constant) and n.value is Ellipsis for n in ast.walk(tree))


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. Can you replace it with your own code?",
                nl="Er staat nog een `...` in je code. Kun je de `...` vervangen door je eigen code?")


# ======== exercise 1: a unit test of inputs, on one row ==================================================================

_O_START = ("import pandas as pd\n"
            "from ops.demand import FEATURES, inputs\n"
            "\n\n"
            "def test_one_row():\n"
            "    scale = {\"mean\": [3, 1, 0, 1, 0.5, 0.5, 0.5, 0.25, 3, 7],\n"
            "             \"std\": [2, 0.5, 0.25, 0.5, 0.25, 0.25, 0.25, 0.125, 1, 4]}\n"
            "    row = dict(zip(FEATURES, scale[\"mean\"]))\n"
            "    row[\"temp\"] = 0.75\n"
            "    row[\"hr\"] = 8\n"
            "    x = inputs(pd.DataFrame([row]), scale)\n"
            "    assert x.shape == (1, 34)\n")
_O = _O_START + "    assert x[0, 8] == 1\n    assert x[0, 28] == 1\n"

# ======== exercise 2: a model assertion ==================================================================================

# What the course's model of the lesson *Project: a demand model, tracked and packed* (h16-lr0.01) expects for each hour
# of Thursday 14 July 2011, to one decimal (tests/test_sys_code_tests.py works it out again).
DAY_PREDICTIONS = [45.7, 23.4, 7.4, -6.3, -10.8, 8.1, 101.0, 296.4, 431.2, 223.1, 143.5, 167.4, 205.5, 203.1, 186.4, 197.5,
                   288.8, 537.3, 517.2, 375.7, 268.5, 202.1, 151.4, 93.9]
NEGATIVE_2011 = 235   # hours of 2011 for which that model expects fewer than 0 bikes, all from midnight to 5 in the morning


class _Pred:
    """A tensor of predictions the check calls by the name the lesson gives it; the learner's code gets a copy."""

    def __init__(self, values, name: str):
        self.t, self.name = torch.tensor(values, dtype=torch.float32), name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return self.t.clone()


NAN = float("nan")
PRED = _Pred(DAY_PREDICTIONS, "PRED")
GAPS = _Pred([12.0, NAN, 40.5, -3.2, 0.0, 88.0], "GAPS")       # an hour without its humidity gives no number
COLUMN = _Pred([[5.0], [-1.0], [NAN], [300.0]], "COLUMN")       # a column, as the network gives it
ZEROS = _Pred([0.0, 0.0, 0.5], "ZEROS")                         # 0 bikes in an hour can happen

_I_HEAD = "import torch\n\n\ndef impossible(pred):\n"


def _i(body: str) -> str:
    return _I_HEAD + f"    return {body}\n"


_I_START = _i("...")
_I = _i("(pred < 0) | torch.isnan(pred)")
_I_NEG = _i("pred < 0")
_I_NAN = _i("torch.isnan(pred)")
_I_ZERO = _i("(pred <= 0) | torch.isnan(pred)")
_I_AND = _i("(pred < 0) & torch.isnan(pred)")
_I_COUNT = _i("((pred < 0) | torch.isnan(pred)).sum()")


def _mask_shown(v):
    """A mask in a check message: where it is True, or the value itself when it is not a mask."""
    if isinstance(v, torch.Tensor) and v.dtype == torch.bool:
        where = [i for i, b in enumerate(v.reshape(-1).tolist()) if b]
        n = v.numel()
        out = {}
        for lang in LANGS:
            places = ", ".join(str(i) for i in where[:8]) + (", …" if len(where) > 8 else "")
            if lang == "en":
                out[lang] = (f"a mask of {n} that is False everywhere" if not where else
                             f"a mask of {n} that is True at {'place' if len(where) == 1 else 'the places'} {places}")
            else:
                out[lang] = (f"een mask van {n} dat overal False is" if not where else
                             f"een mask van {n} dat True is op {'plek' if len(where) == 1 else 'de plekken'} {places}")
        return out
    shown = repr(v) if not isinstance(v, torch.Tensor) else ("tensor(" + ", ".join(
        str(round(float(x), 1)) for x in v.reshape(-1)[:6]) + (", …" if v.numel() > 6 else "") + ")")
    return {lang: f"`{shown[:70]}`" for lang in LANGS}


IMPOSSIBLE_TEXT = {
    "en": {"title": "What the model expects for each hour", "x": "hour of the day",
           "y": "bikes", "pred": "what the course's model expects", "flag": "flagged by your impossible",
           "alt": "Bars of what the course's model expects for each hour of Thursday 14 July 2011: two peaks, about 430 bikes at 8 in the morning and about 540 at 5 in the afternoon, and two small bars below 0 at 3 and 4 in the morning, −6.3 and −10.8 bikes. The hours your impossible flags are marked with a dot.",
           "none": "Your `impossible` flags none of the 24 hours.",
           "some": "Your `impossible` flags {n} of the 24 hours: {hours}."},
    "nl": {"title": "Wat het model voor elk uur verwacht", "x": "uur van de dag",
           "y": "fietsen", "pred": "wat het model van de course verwacht", "flag": "gemarkeerd door je impossible",
           "alt": "Staven van wat het model van de course verwacht voor elk uur van donderdag 14 juli 2011: twee pieken, ongeveer 430 fietsen om 8 uur 's ochtends en ongeveer 540 om 5 uur 's middags, en twee kleine staven onder 0 om 3 en 4 uur 's nachts, −6,3 en −10,8 fietsen. Op elk uur dat je impossible markeert, staat een stip.",
           "none": "Je `impossible` markeert geen van de 24 uren.",
           "some": "Je `impossible` markeert {n} van de 24 uren: {hours}."},
}


def impossible_picture(seen) -> str:
    """What the course's model expects for each hour of 14 July as bars, with a dot on each hour the learner's
    impossible flags."""
    lang = _lang(seen.lang)
    w = IMPOSSIBLE_TEXT[lang]
    fn = seen.mine.names.get("impossible") if seen.mine is not None else None
    if not callable(fn):
        return ""
    try:
        mask = with_time_limit(lambda: fn(PRED.t.clone()), 2.0)
        flagged = [i for i, b in enumerate(torch.as_tensor(mask).reshape(-1).tolist()) if b] \
            if isinstance(mask, torch.Tensor) and mask.numel() == 24 else None
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check says what is wrong with the function
        flagged = None
    p = chart.Panel((-0.5, 23.5), (-50, 600), title=w["title"], xlabel=w["x"], ylabel=w["y"], xticks=[0, 6, 12, 18, 23])
    for h, v in enumerate(DAY_PREDICTIONS):
        p.rect(h - 0.4, min(0.0, v), h + 0.4, max(0.0, v), "pred")
    if flagged:
        p.points(flagged, [max(DAY_PREDICTIONS[h], 0.0) + 30 for h in flagged], "loss", r=4.5)
    out = _chart([p], w["alt"], lang) + live.key_html([("pred", False, w["pred"]), ("loss", False, w["flag"])])
    if flagged is not None:
        joiner = " and " if lang == "en" else " en "
        hours = [str(h) for h in flagged]
        listed = hours[0] if len(hours) == 1 else ", ".join(hours[:-1]) + joiner + hours[-1] if hours else ""
        note = w["none"] if not flagged else w["some"].format(n=len(flagged), hours=listed)
        out += (f'<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem">'
                f'{html.escape(note).replace("`impossible`", "<code>impossible</code>")}</div>')
    title = {"en": "Your impossible on 14 July", "nl": "Je impossible op 14 juli"}[lang]
    return yours.figure(title, out)


# ======== exercise 3: overfit one batch ==================================================================================

_F_START = ("import torch\n"
            "from ops.demand import DemandNet, inputs, read_hours, scaling, targets, train_epoch\n"
            "\n\n"
            "def test_overfit_one_batch():\n"
            "    hours = read_hours()\n"
            "    few = hours.sample(16, random_state=0)\n"
            "    x, y = inputs(few, scaling(hours)), targets(few)\n"
            "    torch.manual_seed(0)\n"
            "    model = DemandNet(16)\n"
            "    opt = torch.optim.Adam(model.parameters(), lr=0.03)\n"
            "    first = ((model(x) - y) ** 2).mean().item()\n"
            "    for _ in range(400):\n"
            "        train_epoch(model, opt, x, y, 16)\n"
            "    last = ((model(x) - y) ** 2).mean().item()\n"
            "    assert ...\n")
_F = _F_START.replace("    assert ...\n", "    assert last < 0.01 * first\n")


# ======== the exercises ==================================================================================================

EXERCISES = {
    "one-row": TestsBox(
        name="one-row",
        function="inputs",
        swaps=("FEATURES", "inputs"),
        versions=INPUT_VERSIONS,
        start=_O_START,
        solution=_O,
        save_as=("ops", "test_inputs"),
        picture=tests_picture,
        mistakes=_SYNTAX,
        bugs={
            "shift": {"en": "Your test passes for an `inputs` with a bug: it puts hour 8 in column 9, one hour too far. The 24 hour columns come first. Which column must be 1 for hour 8, and does your test check it?",
                      "nl": "Je test slaagt voor een `inputs` met een bug: deze versie zet uur 8 in kolom 9, één uur te ver. De 24 kolommen van het uur komen eerst. Welke kolom moet 1 zijn voor uur 8, en controleert je test deze kolom?"},
            "order": {"en": "Your test passes for an `inputs` with a bug: it puts the 10 features first and the 24 hour columns after them, so column 8 holds a feature. Which column must be 1 for hour 8, and does your test check it?",
                      "nl": "Je test slaagt voor een `inputs` met een bug: deze versie zet de 10 features eerst en de 24 kolommen van het uur erachter, dus bevat kolom 8 een feature. Welke kolom moet 1 zijn voor uur 8, en controleert je test deze kolom?"},
            "no-mean": {"en": "Your test passes for an `inputs` with a bug: it divides by the standard deviation but does not subtract the mean first, so temp becomes 0.75 / 0.25 = 3. Temp is one standard deviation above its mean: in which column is it, what must be in that column, and does your test check it?",
                        "nl": "Je test slaagt voor een `inputs` met een bug: deze versie deelt door de standaardafwijking, maar trekt het gemiddelde er niet eerst af, dus wordt temp 0,75 / 0,25 = 3. Temp ligt één standaardafwijking boven het gemiddelde: in welke kolom staat temp, wat moet erin staan, en controleert je test deze kolom?"},
            "own-mean": {"en": "Your test passes for an `inputs` with a bug: it subtracts the mean of the hours it gets, not the mean in `scale`. With one row, every feature then becomes 0, temp too. Which column must be 1 besides the column of the hour, and does your test check it?",
                         "nl": "Je test slaagt voor een `inputs` met een bug: deze versie trekt het gemiddelde van de gegeven uren af, niet het gemiddelde in `scale`. Met één rij wordt dan elke feature 0, ook temp. Welke kolom moet behalve de kolom van het uur ook 1 zijn, en controleert je test deze kolom?"},
        },
        on_right=[
            Mistake("raw-temp", matches=lambda got, line: "0.75" in line,
                    en="The test `{line}` fails for the right `inputs`: `inputs` standardizes temp, so this row gets (0.75 − 0.5) / 0.25 in that column. What number is that?",
                    nl="De test `{line}` faalt voor de goede `inputs`: `inputs` standaardiseert temp, dus krijgt deze rij (0,75 − 0,5) / 0,25 in deze kolom. Welk getal is dat?"),
            Mistake("feature-index", matches=lambda got, line: re.search(r"\[\s*0\s*,\s*4\s*\]", line) is not None,
                    en="The test `{line}` fails for the right `inputs`: column 4 is the column of hour 4, which is 0 for hour 8. The 24 hour columns come first: in which column does temp land, the fifth feature?",
                    nl="De test `{line}` faalt voor de goede `inputs`: kolom 4 is de kolom van uur 4, en deze kolom is 0 voor uur 8. De 24 kolommen van het uur komen eerst: in welke kolom komt temp, de vijfde feature?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "The test in the box passes for the right `inputs`, and for all four versions with a bug too: each gives 34 numbers. Can you add lines with `assert` under the last line that say what must be in the column of hour 8 and in the column of temp?",
                      "nl": "De test in het vak slaagt voor de goede `inputs`, en ook voor alle vier versies met een bug: elke versie geeft 34 getallen. Kun je onder de laatste regel met `assert` vastleggen wat er in de kolom van uur 8 en in de kolom van temp moet staan?"},
            "no_test": {"en": "Your box has no function whose name starts with `test`, so pytest finds no test to run. Can you keep `def test_one_row():` and put your lines with `assert` inside it, indented?",
                        "nl": "Je vak heeft geen functie met een naam die begint met `test`, dus vindt pytest geen test om uit te voeren. Kun je `def test_one_row():` houden en je regels met `assert` erin zetten, ingesprongen?"},
            "fails_right": {"en": "The test `{line}` fails for the right `inputs` too, so the test itself is wrong. Work out by hand what that column holds for this row: is it the column you mean?",
                            "nl": "De test `{line}` faalt ook voor de goede `inputs`, dus klopt de test zelf niet. Reken met de hand uit wat deze kolom voor deze rij bevat: is het de kolom die je bedoelt?"},
            "slow": {"en": "Your test is still not done after a few seconds. Does something in it keep going forever?",
                     "nl": "Je test is na een paar seconden nog niet klaar. Gaat er iets in eindeloos door?"},
            "correct": {"en": "Right: your test passes for the right `inputs`, and fails for each of the four versions with a bug. It is now in your project, in `{saved}`, where pytest would find it by its name. Which of the four bugs would a test of the shape alone have caught?",
                        "nl": "Goed: je test slaagt voor de goede `inputs`, en faalt voor elk van de vier versies met een bug. De test staat nu in je project, in `{saved}`, waar pytest de test aan de naam zou vinden. Welke van de vier bugs had een test van alleen de shape gevangen?"},
            "kept": _kept("test_one_row"),
        },
    ),
    "impossible": FunctionBox(
        name="impossible",
        function="impossible",
        start=_I_START,
        solution=_I,
        save_as=("ops", "assertions"),
        visible=[(PRED,)],
        hidden=[(GAPS,), (COLUMN,), (ZEROS,)],
        first=[_DOTS],
        picture=impossible_picture,
        show=_mask_shown,
        mistakes=[
            Mistake("or", error=lambda e: isinstance(e, RuntimeError) and "ambiguous" in str(e),
                    en="With `{call}`, your function stops with an error: `or` asks for one True or False, and a tensor of 24 predictions holds 24 of them. Which operator combines two masks number by number?",
                    nl="Met `{call}` stopt je functie met een fout: `or` vraagt om één True of False, en een tensor met 24 voorspellingen bevat er 24. Welke operator combineert twee masks getal voor getal?"),
            Mistake("brackets", error=lambda e: "bitwise_or" in str(e),
                    en="With `{call}`, your function stops with an error: Python does `|` before `<`, so it first combines the mask of the NaNs with `pred` itself, and `|` cannot combine True and False with numbers. Does each comparison have parentheses of its own?",
                    nl="Met `{call}` stopt je functie met een fout: Python doet `|` eerder dan `<`, dus combineert Python eerst het mask van de NaN's met `pred` zelf, en `|` kan True en False niet met getallen combineren. Heeft elke vergelijking eigen haakjes?"),
            Mistake("brackets-order", matches=lambda got, text_: re.search(r"<\s*0\s*\|", text_) is not None, state="almost",
                    en="Right for `PRED`. But `{call}` gives {got}, and it should give {want}: Python does `|` before `<`, so your line compares `pred` with what `0` and the mask of the NaNs give together with `|`, not with `0`. Does each comparison have parentheses of its own?",
                    nl="Goed voor `PRED`. Maar `{call}` geeft {got}, en het moet {want} zijn: Python doet `|` eerder dan `<`, dus vergelijkt je regel `pred` met wat `0` en het mask van de NaN's samen met `|` geven, niet met `0`. Heeft elke vergelijking eigen haakjes?"),
            Mistake("nan-equal", matches=lambda got, text_: re.search(r"==\s*(float\(\s*['\"]nan['\"]\s*\)|torch\.nan|math\.nan|np\.nan|numpy\.nan)", text_) is not None,
                    state="almost",
                    en="Right for `PRED`. But `{call}` gives {got}, and it should give {want}: `pred == float(\"nan\")` is False everywhere, also where a prediction is not a number. NaN is equal to nothing, not even to itself. Which function of `torch` asks whether a number is NaN?",
                    nl="Goed voor `PRED`. Maar `{call}` geeft {got}, en het moet {want} zijn: `pred == float(\"nan\")` is overal False, ook waar een voorspelling geen getal is. NaN is aan niets gelijk, zelfs niet aan zichzelf. Welke functie van `torch` vraagt of een getal NaN is?"),
            Mistake("only-negative", wrong=_I_NEG, state="almost",
                    en="Right for `PRED`. But `{call}` gives {got}, and it should give {want}: an hour without its humidity gives a prediction that is not a number, NaN, and `pred < 0` is False for it. Which function of `torch` asks whether a number is NaN?",
                    nl="Goed voor `PRED`. Maar `{call}` geeft {got}, en het moet {want} zijn: een uur zonder luchtvochtigheid geeft een voorspelling die geen getal is, NaN, en `pred < 0` is daar False. Welke functie van `torch` vraagt of een getal NaN is?"),
            Mistake("zero", wrong=_I_ZERO, state="almost",
                    en="Right for `PRED`. But `{call}` gives {got}, and it should give {want}: your check flags 0 bikes too, and an hour without a single rented bike can happen, late at night. Which comparison flags fewer than 0 only?",
                    nl="Goed voor `PRED`. Maar `{call}` geeft {got}, en het moet {want} zijn: je check markeert ook 0 fietsen, en een uur zonder één gehuurde fiets kan gebeuren, laat in de nacht. Welke vergelijking markeert alleen minder dan 0?"),
            Mistake("only-nan", wrong=_I_NAN,
                    en="`{call}` gives {got}, and it should give {want}: your check flags only what is not a number, and the model expects −6.3 and −10.8 bikes at 3 and 4 in the morning. What else cannot be right?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je check markeert alleen wat geen getal is, en het model verwacht −6,3 en −10,8 fietsen om 3 en 4 uur 's nachts. Wat kan er nog meer niet kloppen?"),
            Mistake("and", wrong=_I_AND,
                    en="`{call}` gives {got}, and it should give {want}: `&` is True only where both are True, and no prediction is below 0 and NaN at once. Which operator is True where at least one of the two is True?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `&` is alleen True waar allebei True zijn, en geen voorspelling is tegelijk onder 0 en NaN. Welke operator is True waar minstens één van de twee True is?"),
            Mistake("count", wrong=_I_COUNT,
                    en="`{call}` gives {got}, and it should give {want}: you count the flagged hours, and the service needs to know which hours they are. Can you give back the mask itself, one True or False for each prediction?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je telt de gemarkeerde uren, en de service moet weten welke uren het zijn. Kun je het mask zelf teruggeven, één True of False per voorspelling?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with a mask of `pred`: True where a prediction cannot be right, False everywhere else.",
                      "nl": "Vervang de `...` door een mask van `pred`: True waar een voorspelling niet kan kloppen, en verder overal False."},
            "arguments": {"en": "The check calls `{call}`, with one value, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(pred)`?",
                          "nl": "De check roept `{call}` aan, met één waarde, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(pred)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is your mask True for every prediction below 0 and for every NaN, and False for the others?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Is je mask True voor elke voorspelling onder 0 en voor elke NaN, en False voor de andere?"},
            "almost": {"en": "Right for `PRED`. But `{call}` gives {got}, and it should give {want}. Does your function work for any tensor of predictions, also a column, and also one with NaN in it?",
                       "nl": "Goed voor `PRED`. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie voor elke tensor met voorspellingen, ook een kolom, en ook een tensor met NaN erin?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still begin its last line with `return`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Begint de laatste regel nog met `return`?"},
            "correct": {"en": "Right in all {n} cases: your check flags the hours 3 and 4 of 14 July, and every NaN. On all 8,645 hours of 2011 it would flag 235, all from midnight to 5 in the morning. Your function is now in your project, in `{saved}`. What should a service do with an hour it flags?",
                        "nl": "Goed in alle {n} gevallen: je check markeert de uren 3 en 4 van 14 juli, en elke NaN. Op alle 8.645 uren van 2011 zou de check er 235 markeren, allemaal van middernacht tot 5 uur 's ochtends. Je functie staat nu in je project, in `{saved}`. Wat moet een service doen met een uur dat de check markeert?"},
            "kept": _kept("impossible"),
        },
    ),
    "overfit": TestsBox(
        name="overfit",
        function="train_epoch",
        swaps=("DemandNet", "inputs", "read_hours", "scaling", "targets", "train_epoch"),
        versions=TRAIN_VERSIONS,
        start=_F_START,
        solution=_F,
        save_as=("ops", "test_training"),
        picture=tests_picture,
        first=[_DOTS],
        mistakes=_SYNTAX,
        bugs={
            "no-step": {"en": "Your test passes for a `train_epoch` with a bug: it never calls `opt.step()`, so the weights never change, and the loss stays at 36,930 from the first step to the last. Does your test ask the loss to fall?",
                        "nl": "Je test slaagt voor een `train_epoch` met een bug: deze versie roept `opt.step()` nooit aan, dus veranderen de weights nooit, en de loss blijft 36.930 van de eerste tot de laatste stap. Vraagt je test of de loss daalt?"},
            "column": {"en": "Your test passes for a `train_epoch` with a bug: it compares `model(x)` with `y` as a row, so the loss falls from 36,930 to 21,091 and stops there, since the network can then learn only the mean of the 16 hours. A network of 577 numbers can learn 16 hours by heart: how close to 0 must the loss come?",
                       "nl": "Je test slaagt voor een `train_epoch` met een bug: deze versie vergelijkt `model(x)` met `y` als rij, dus daalt de loss van 36.930 naar 21.091 en blijft daar, omdat het netwerk dan alleen het gemiddelde van de 16 uren kan leren. Een netwerk van 577 getallen kan 16 uren uit het hoofd leren: hoe dicht bij 0 moet de loss komen?"},
            "no-zero-grad": {"en": "Your test passes for a `train_epoch` with a bug: it never calls `opt.zero_grad()`, so the gradients of all earlier steps add up, and the loss jumps up and down, to 28,134 after the last step. How close to 0 must the loss come?",
                             "nl": "Je test slaagt voor een `train_epoch` met een bug: deze versie roept `opt.zero_grad()` nooit aan, dus tellen de gradients van alle eerdere stappen op, en springt de loss op en neer, naar 28.134 na de laatste stap. Hoe dicht bij 0 moet de loss komen?"},
        },
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` after `assert` with a condition: how far must `last` have fallen below `first`, after 400 steps on 16 hours?",
                      "nl": "Vervang de `...` na `assert` door een voorwaarde: hoe ver moet `last` onder `first` gedaald zijn, na 400 stappen op 16 uren?"},
            "no_test": {"en": "Your box has no function whose name starts with `test`, so pytest finds no test to run. Can you keep `def test_overfit_one_batch():` as it is?",
                        "nl": "Je vak heeft geen functie met een naam die begint met `test`, dus vindt pytest geen test om uit te voeren. Kun je `def test_overfit_one_batch():` laten zoals het was?"},
            "fails_right": {"en": "The test `{line}` fails for the right `train_epoch` too: there the loss falls from 36,930 to 0.99, close to 0 but not exactly 0. Which condition lets that pass, and still fails when the loss stops far above 0?",
                            "nl": "De test `{line}` faalt ook voor de goede `train_epoch`: daar daalt de loss van 36.930 naar 0,99, dicht bij 0 maar niet precies 0. Welke voorwaarde laat dat slagen, en faalt toch als de loss ver boven 0 blijft steken?"},
            "slow": {"en": "Your test is still not done after a few seconds. Does something in it keep going forever?",
                     "nl": "Je test is na een paar seconden nog niet klaar. Gaat er iets in eindeloos door?"},
            "correct": {"en": "Right: your test passes for the right `train_epoch`, and fails for each of the three versions with a bug, in about a second. It is now in your project, in `{saved}`. Would `assert last < first` have caught all three?",
                        "nl": "Goed: je test slaagt voor de goede `train_epoch`, en faalt voor elk van de drie versies met een bug, in ongeveer een seconde. De test staat nu in je project, in `{saved}`. Had `assert last < first` alle drie gevangen?"},
            "kept": _kept("test_overfit_one_batch"),
        },
    ),
}

# The lesson is done when these are correct.
REQUIRED = ["one-row", "impossible", "overfit", "regression"]


# ======== the course's runs on the 16 hours ==============================================================================

_RUNS: dict = {}


def overfit_run(train_epoch, show=None) -> dict:
    """A fresh DemandNet trained on the 16 hours as the test of exercise 3 does it, with the loss after every step
    and what the network says for each of the 16 hours every 10 steps."""
    torch.manual_seed(0)
    model = demand.DemandNet(16)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    with torch.no_grad():
        out = model(X16)
    losses = [float(((out - Y16) ** 2).mean())]
    frames = {0: [float(v) for v in out.reshape(-1)]}
    if show is not None:
        show(0, losses[0], frames[0])
    for step in range(1, STEPS + 1):
        train_epoch(model, opt, X16, Y16, 16)
        with torch.no_grad():
            out = model(X16)
        losses.append(float(((out - Y16) ** 2).mean()))
        if step % 10 == 0:
            frames[step] = [float(v) for v in out.reshape(-1)]
        if show is not None:
            show(step, losses[-1], frames.get(step))
    return {"losses": losses, "frames": frames}


def course_runs() -> dict:
    """The four versions of train_epoch on the 16 hours: the numbers the lesson's text gives."""
    if not _RUNS:
        for key, fn in TRAIN_VERSIONS:
            _RUNS[key] = overfit_run(fn)
    return _RUNS


# ======== the picture of the idea ========================================================================================

IDEA_TEXT = {
    "en": {"title": "Loss on 16 hours of 2011", "x": "step", "y": "loss",
           "right": "the right train_epoch", "column": "y as a row",
           "alt": "Two lines of the loss on the 16 hours over 400 steps, on a scale where every step up is ten times more. With the right train_epoch the loss falls from 36,930 to about 1. With y as a row it falls to 21,091 within the first 100 steps, and stays there."},
    "nl": {"title": "Loss op 16 uren van 2011", "x": "stap", "y": "loss",
           "right": "de goede train_epoch", "column": "y als rij",
           "alt": "Twee lijnen van de loss op de 16 uren in 400 stappen, op een schaal waarop elke stap omhoog tien keer zoveel is. Met de goede train_epoch daalt de loss van 36.930 naar ongeveer 1. Met y als rij daalt de loss in de eerste 100 stappen naar 21.091, en blijft daar."},
}


def idea_picture(lang: str) -> str:
    lang = _lang(lang)
    w = IDEA_TEXT[lang]
    runs = course_runs()
    p = chart.Panel((0, STEPS), (0.1, 100000), title=w["title"], xlabel=w["x"], ylabel=w["y"],
                    xticks=[0, 100, 200, 300, 400], ylog=True)
    steps = list(range(STEPS + 1))
    p.line(steps, [max(v, 0.1) for v in runs["column"]["losses"]], "loss", 1.4, dashed=True)
    p.line(steps, [max(v, 0.1) for v in runs["right"]["losses"]], "loss", 2.4)
    return _chart([p], w["alt"], lang) + live.key_html([("loss", False, w["right"]), ("loss", True, w["column"])])


# ======== the demo: three versions of train_epoch, and your test on each =================================================

DEMO_VERSIONS = ["right", "column", "no-zero-grad"]
LOW, HIGH = 0.0, 40000.0   # the y-axis of the live chart of the loss

DEMO_TEXT = {
    "en": {"title": "Loss on 16 hours: {name}", "x": "step", "y": "loss",
           "alt": "The loss on the 16 hours after each step of the version of train_epoch that trains now, with the versions already trained as thin dashed lines.",
           "progress": "Training with **{name}**: step {s} of {n}, loss {loss}.",
           "hours": "{name}, step {s}", "real": "bikes rented", "model": "what the network says",
           "hours_alt": "Bars of the bikes rented in each of the 16 hours, from the quietest to the busiest, and a dot for what the network says for each.",
           "hour": "the 16 hours, quietest first", "bikes": "bikes",
           "rewind": "After step {s}: {listed}.",
           "rewind_alt": "The loss of the three versions up to the chosen step, on a scale where every step up is ten times more, with a line at that step.",
           "curves": "Loss on 16 hours", "who": {"yours": "your test from exercise 3", "course": "the course's test"},
           "not_yet": "Your test from exercise 3 is not in your project yet: once its check says it is right, the demo runs your test.",
           "verdicts": "Then {who} ran on each version: {listed}.",
           "passes": "passes", "fails": "fails", "slow": "too slow",
           "slowed": "*The demo took more than a minute. Does something run longer than it needs?*"},
    "nl": {"title": "Loss op 16 uren: {name}", "x": "stap", "y": "loss",
           "alt": "De loss op de 16 uren na elke stap van de versie van train_epoch waarmee het netwerk nu traint, met de versies die al getraind zijn als dunne lijnen van streepjes.",
           "progress": "Training met **{name}**: stap {s} van {n}, loss {loss}.",
           "hours": "{name}, stap {s}", "real": "gehuurde fietsen", "model": "wat het netwerk zegt",
           "hours_alt": "Staven van de gehuurde fietsen in elk van de 16 uren, van het rustigste naar het drukste, en een stip voor wat het netwerk voor elk uur zegt.",
           "hour": "de 16 uren, rustigste eerst", "bikes": "fietsen",
           "rewind": "Na stap {s}: {listed}.",
           "rewind_alt": "De loss van de drie versies tot de gekozen stap, op een schaal waarop elke stap omhoog tien keer zoveel is, met een lijn bij deze stap.",
           "curves": "Loss op 16 uren", "who": {"yours": "je test uit oefening 3", "course": "de test van de course"},
           "not_yet": "Je test uit oefening 3 staat nog niet in je project: zegt de check dat de test goed is, dan draait de demo je eigen test.",
           "verdicts": "Daarna draaide {who} op elke versie: {listed}.",
           "passes": "slaagt", "fails": "faalt", "slow": "te traag",
           "slowed": "*De demo duurde langer dan een minuut. Draait er iets langer dan nodig is?*"},
}
STYLE = {"right": ("loss", False), "column": ("loss", True), "no-zero-grad": ("ink-3", False)}
_ORDER = sorted(range(16), key=lambda i: (float(Y16[i]), i))   # the 16 hours, quietest first


def version_name(key: str, lang: str) -> str:
    return VERSION_NAMES[_lang(lang)][key]


def live_labels(name: str, lang: str) -> dict:
    w = DEMO_TEXT[_lang(lang)]
    return {"title": w["title"].format(name=version_name(name, lang)), "x": w["x"], "y": w["y"], "alt": w["alt"]}


def progress_line(name: str, step: int, loss: float, lang: str) -> str:
    lang = _lang(lang)
    return DEMO_TEXT[lang]["progress"].format(name=version_name(name, lang), s=step, n=STEPS, loss=number(loss, 0, lang))


def hours_panel(preds: list, name: str, step: int, lang: str, title: bool = True) -> chart.Panel:
    """The bikes of the 16 hours as bars, quietest first, and what the network says for each as a dot. Without
    `title`, the panel names only the version: the rewind says the step under it."""
    w = DEMO_TEXT[_lang(lang)]
    real = [float(Y16[i]) for i in _ORDER]
    p = chart.Panel((0.5, 16.5), (0, 600), title=w["hours"].format(name=version_name(name, lang), s=step) if title else version_name(name, lang),
                    xlabel=w["hour"], ylabel=w["bikes"], xticks=[1, 4, 8, 12, 16])
    for k, v in enumerate(real, start=1):
        p.rect(k - 0.35, 0, k + 0.35, v, "data")
    if len(preds) == 16:
        p.points(list(range(1, 17)), [max(0.0, min(preds[i], 600.0)) for i in _ORDER], "pred", r=3.5)
    return p


def hours_picture(preds: list, name: str, step: int, lang: str) -> str:
    lang = _lang(lang)
    w = DEMO_TEXT[lang]
    return (_chart([hours_panel(preds, name, step, lang)], w["hours_alt"], lang, width=300, height=220)
            + live.key_html([("data", False, w["real"]), ("pred", False, w["model"])]))


def now_picture(state: dict, lang: str) -> str:
    """Beside the live chart: what the version that trains now says for each of the 16 hours."""
    return hours_picture(state["preds"], state["name"], state["step"], lang)


def demo(test_text: str | None, show=None) -> dict | str:
    """Three versions of train_epoch on the 16 hours, then the learner's test of exercise 3 on each (or the course's,
    when theirs is not in the project yet): a dict, or "slow"."""
    def go():
        out = {"runs": {}, "verdicts": {}, "who": "yours" if test_text else "course"}
        for key in DEMO_VERSIONS:
            fn = dict(TRAIN_VERSIONS)[key]
            done = [(k, list(enumerate(out["runs"][k]["losses"]))) for k in DEMO_VERSIONS if k in out["runs"]]

            def look(step, loss, preds, key=key, done=done):
                if show is not None:
                    show(key, step, loss, preds, done)

            out["runs"][key] = overfit_run(fn, show=look)
        text = test_text or _F
        base = _base(EXERCISES["overfit"].swaps)
        for key in DEMO_VERSIONS:
            out["verdicts"][key] = _verdict(_run_tests(text, {**base, "train_epoch": dict(TRAIN_VERSIONS)[key]}))
        return out
    try:
        return with_time_limit(go, DEMO_LIMIT)
    except Timeout:
        return "slow"


def saved_test() -> str | None:
    """The learner's test of exercise 3, as the check saved it in the project, or None while it is not there."""
    path = projects.folder() / "ops" / "test_training.py"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    return text if _tests_in(text) else None


def _curve_panel(r: dict, s: int, lang: str) -> chart.Panel:
    w = DEMO_TEXT[lang]
    p = chart.Panel((0, STEPS), (0.1, 100000), title=w["curves"], xlabel=w["x"], ylabel=w["y"],
                    xticks=[0, 100, 200, 300, 400], ylog=True)
    for key in DEMO_VERSIONS:
        losses = r["runs"].get(key, {}).get("losses", [])
        if losses:
            role, dashed = STYLE[key]
            p.line(list(range(len(losses))), [min(max(v, 0.1), 100000) for v in losses], role,
                   1.4 if dashed else 2.2, dashed=dashed)
    p.line([s, s], [0.1, 100000], "axis", 1.2)
    return p


def curve_key(lang: str) -> list:
    return [(STYLE[k][0], STYLE[k][1], version_name(k, lang)) for k in DEMO_VERSIONS]


def rewind_frame(r: dict, s: int, lang: str) -> tuple:
    """A stop of the rewind after the demo: the three curves with a line at step s, under them what each version said
    for the 16 hours after that step, and the loss of each then."""
    lang = _lang(lang)
    w = DEMO_TEXT[lang]
    s = max(0, min(int(s) // 10 * 10, STEPS))
    curves = _chart([_curve_panel(r, s, lang)], w["rewind_alt"], lang) + live.key_html(curve_key(lang))
    panels = [hours_panel(r["runs"][k]["frames"].get(s, []), k, s, lang, title=False) for k in DEMO_VERSIONS]
    hours = (_chart(panels, w["hours_alt"], lang, width=250, height=200)
             + live.key_html([("data", False, w["real"]), ("pred", False, w["model"])]))
    listed = ", ".join(f"{version_name(k, lang)} {number(r['runs'][k]['losses'][s], 0 if r['runs'][k]['losses'][s] >= 10 else 2, lang)}"
                       for k in DEMO_VERSIONS)
    return live.rewound(curves, [hours], width=330), w["rewind"].format(s=s, listed=listed)


def verdicts(r: dict, lang: str) -> str:
    lang = _lang(lang)
    w = DEMO_TEXT[lang]
    words = {True: w["passes"], False: w["fails"], None: w["slow"]}
    listed = "; ".join(f"{version_name(k, lang)}: **{words[r['verdicts'][k]]}**" for k in DEMO_VERSIONS)
    said = w["verdicts"].format(who=w["who"][r["who"]], listed=listed)
    return said + (" " + w["not_yet"] if r["who"] == "course" else "")


def slowed(lang: str) -> str:
    return DEMO_TEXT[_lang(lang)]["slowed"]


# ======== cases the tests run ============================================================================================

def _with(text: str, *lines: str) -> str:
    return text + "".join(f"    {line}\n" for line in lines)


CASES = [
    ("one-row", _O_START, "empty", None),
    ("one-row", "", "empty", None),
    ("one-row", _O, "correct", None),
    ("one-row", "import torch\n" + _with(_O_START, "expected = torch.zeros(34)", "expected[8] = 1", "expected[28] = 1",
                                          "assert torch.equal(x[0], expected)"), "correct", None),
    ("one-row", _with(_O_START, "assert x[0, 8] == 1", "assert x[0, 24 + FEATURES.index(\"temp\")] == 1"), "correct", None),
    ("one-row", _with(_O_START, "assert x[0, 8] == 1", "assert x[0, 24:].abs().sum() == 1"), "correct", None),
    ("one-row", _with(_O_START, "assert x[0, 8] == 1"), "wrong", "misses-no-mean"),
    ("one-row", _with(_O_START, "assert x[0, 28] == 1"), "wrong", "misses-shift"),
    ("one-row", _with(_O_START, "assert x[0, 9] == 0"), "wrong", "misses-order"),
    ("one-row", _with(_O_START, "assert x[0, 8] == 1", "assert x[0, 24] == 0"), "wrong", "misses-own-mean"),
    ("one-row", _with(_O_START, "assert x[0, 8] == 1", "assert x[0, 28] == 0.75"), "wrong", "raw-temp"),
    ("one-row", _with(_O_START, "assert x[0, 8] == 1", "assert x[0, 4] == 1"), "wrong", "feature-index"),
    ("one-row", _with(_O_START, "assert x[0, 9] == 1"), "wrong", "fails-right"),
    ("one-row", _with(_O_START, "assert x[0, 8] = 1"), "error", "one-equals"),
    ("one-row", _with(_O_START, "assert x[0, 8] == 1", "assert FEATURES[4] == “temp”"), "error", "curly-quote"),
    ("one-row", _with(_O_START, "assert x[0, 99] == 1"), "error", None),
    ("one-row", _O_START + "assert x[0, 8] == 1\n", "wrong", "outside"),
    ("one-row", _O.replace("from ops.demand import FEATURES, inputs\n", "").replace(
        "def test_one_row():\n", "def test_one_row():\n    from ops.demand import FEATURES, inputs\n"), "correct", None),
    ("one-row", _O.replace("from ops.demand import FEATURES, inputs", "from ops import demand").replace(
        "zip(FEATURES", "zip(demand.FEATURES").replace("x = inputs(", "x = demand.inputs("), "correct", None),
    ("one-row", "import torch\n" + _with(_O_START, "assert x[0, 8] == 1", "assert torch.isclose(x[0, 28], torch.tensor(1.0), atol=1e-6)",
                                          "# it’s temp, one standard deviation up"), "correct", None),
    ("one-row", _O.replace("def test_one_row():", "def check_one_row():"), "wrong", "no-test"),
    ("impossible", _I_START, "empty", None),
    ("impossible", "", "empty", None),
    ("impossible", _I, "correct", None),
    ("impossible", _i("(pred < 0) | (pred != pred)"), "correct", None),
    ("impossible", _i("~(pred >= 0)"), "correct", None),
    ("impossible", _i("pred.isnan() | (pred < 0)"), "correct", None),
    ("impossible", _i("(pred < 0) | (pred == float(\"nan\"))"), "almost", "nan-equal"),
    ("impossible", _I_NEG, "almost", "only-negative"),
    ("impossible", _I_ZERO, "almost", "zero"),
    ("impossible", _I_NAN, "wrong", "only-nan"),
    ("impossible", _I_AND, "wrong", "and"),
    ("impossible", _I_COUNT, "wrong", "count"),
    ("impossible", _i("(pred < 0) or torch.isnan(pred)"), "error", "or"),
    ("impossible", _i("torch.isnan(pred) | pred < 0"), "error", "brackets"),
    ("impossible", _i("pred < 0 | torch.isnan(pred)"), "almost", "brackets-order"),
    ("impossible", _i("(pred < 0) | ..."), "wrong", "dots-left"),
    ("impossible", _i("pred.clamp(min=0)"), "wrong", None),
    ("overfit", _F_START, "empty", None),
    ("overfit", "", "empty", None),
    ("overfit", _F, "correct", None),
    ("overfit", _F_START.replace("assert ...", "assert last < 100"), "correct", None),
    ("overfit", _F_START.replace("assert ...", "assert last < 0.5 * first"), "correct", None),
    ("overfit", _F_START.replace("assert ...", "assert last <= first"), "wrong", "misses-no-step"),
    ("overfit", _F_START.replace("assert ...", "assert last < first"), "wrong", "misses-column"),
    ("overfit", _F_START.replace("assert ...", "assert last < 0.6 * first"), "wrong", "misses-column"),
    ("overfit", _F_START.replace("assert ...", "assert model(x).std() > 1"), "wrong", "misses-no-zero-grad"),
    ("overfit", _F_START.replace("assert ...", "assert last == 0"), "wrong", "fails-right"),
    ("overfit", _F_START.replace("assert ...", "assert last = 0"), "error", "one-equals"),
    ("overfit", "import math\n" + _F_START.replace("assert ...", "assert math.isclose(last, 0, abs_tol=5)"), "correct", None),
    ("overfit", _F.replace("from ops.demand import", "import ops.demand as d\nfrom ops.demand import").replace(
        "        train_epoch(model", "        d.train_epoch(model"), "correct", None),
    ("overfit", _F_START.replace("    assert ...\n", "assert last < 0.01 * first\n"), "wrong", "outside"),
    ("overfit", _F_START.replace("assert ...", "assert last < 0.01 * first, “too high”"), "error", "curly-quote"),
    ("overfit", _F_START.replace("assert ...", "assert losses[-1] < 1"), "error", None),
    ("overfit", _F_START.replace("    assert ...\n", "    assert last < 0.01 * first\n    assert ...\n"), "wrong", "dots-left"),
    ("overfit", _F.replace("def test_overfit_one_batch():", "def overfit_one_batch():"), "wrong", "no-test"),
]
