# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-behavior-tests: behavioral tests of the learner's packed demand model, as CheckList
(Ribeiro and others, 2020) has them. test_quiet_night is a minimum functionality test, test_row_number an invariance
test and test_rain a directional expectation test. Each takes the fixture `expect`: a function from a table of hours to
what a model expects for each. A test is right when it passes for the course's model and fails for the versions with
a bug it is meant to catch: the hour in UTC, the temperatures in degrees, the weather the wrong way round, and a
version that learned the row number. Four of the five are the course's model with a bug in the code around it; the
course trains the course's model and the version with the row number here, as the lesson *Project: a demand model,
tracked and packed* trains h16-lr0.01.
The data are the hourly counts of Capital Bikeshare in Washington, D.C., in 2011, from the Bike Sharing dataset of
Fanaee-T and Gama (2013, CC BY 4.0, see data/README.md); the bugs, the tests and the pictures are the course's own."""
import ast
import html
import importlib.util
import re
from dataclasses import dataclass, field

import torch
from torch import nn

from everything_ai import chart, live, paths, projects, yours
from everything_ai.check import MISSING, CodeBox, Mistake, Result, is_empty_text, kept, run_code

LESSON = "sys-behavior-tests"
LANGS = ("en", "nl")

projects.provide("sys-project-tracked-model")   # ops/demand.py and ops/safetensors_io.py, which that lesson gives
projects.provide(LESSON)                        # ops/serving.py and ops/conftest.py, which this lesson gives


def _course_demand():
    """The course's own copy of ops/demand.py: the checks use it, so they stay the same when you change yours."""
    path = paths.LESSONS / "sys-project-tracked-model" / "project" / "ops" / "demand.py"
    spec = importlib.util.spec_from_file_location("_course_ops_demand_behavior", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


demand = _course_demand()

HOURS_2011 = demand.read_hours()                                  # 8,645 hours of 365 days
TRAIN_HOURS = HOURS_2011[~(HOURS_2011["day"] % 5 == 4)]           # the training days of the project: 6,912 hours
VAL_HOURS = HOURS_2011[HOURS_2011["day"] % 5 == 4]                # every fifth day: 1,733 hours
WATCH_DAY = 194                                                   # Thursday 14 July 2011, clear all day


def _lang(lang: str) -> str:
    return lang if lang in LANGS else "en"


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma and a thousands dot in Dutch, and a real minus sign."""
    out = f"{x:,.{digits}f}"
    out = out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out
    return out.replace("-", "−")


def _chart(panels: list, alt: str, lang: str, width: int = 330, height: int = 230) -> str:
    out = chart.figure(panels, alt=alt, width=width, height=height)
    return chart.with_decimal_comma(out) if lang == "nl" else out


# ======== the course's model, and a version that learned the row number ==================================================

ROW_FEATURES = demand.FEATURES + ["instant"]   # the version with the row number: the 10 features and instant


class _Wider(nn.Module):
    """DemandNet with room for one more feature: the same layers, made in the same order."""

    def __init__(self, n_in: int, hidden: int):
        super().__init__()
        self.hidden = nn.Linear(n_in, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.out(torch.relu(self.hidden(x)))


def _scale(features: list) -> dict:
    return {"mean": [round(float(TRAIN_HOURS[f].mean()), 6) for f in features],
            "std": [round(float(TRAIN_HOURS[f].std()), 6) for f in features]}


def _rows(hours, features: list, scale: dict):
    """What demand.inputs gives, for any list of features: the hour as 24 one-hot columns, then each feature
    standardized."""
    hour = nn.functional.one_hot(torch.tensor(hours["hr"].to_numpy()), 24).float()
    values = torch.tensor(hours[features].to_numpy(dtype="float32"))
    return torch.cat([hour, (values - torch.tensor(scale["mean"])) / torch.tensor(scale["std"])], dim=1)


def _model(key: str):
    """The course's model ("course") or the version with the row number ("row-number"), trained once, the way the
    course's train_version trains h16-lr0.01 in the lesson *Project: a demand model, tracked and packed*: seed 0, 16
    neurons, Adam with a learning rate of 0.01, 30 epochs in batches of 64 hours, on the training days. On one thread,
    so that the numbers are the ones the lesson's text gives. Kept once per Python process, for both languages."""
    return kept((LESSON, "model", key), lambda: _trained(key))


def _trained(key: str):
    features = demand.FEATURES if key == "course" else ROW_FEATURES
    scale = _scale(features)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        x, y = _rows(TRAIN_HOURS, features, scale), demand.targets(TRAIN_HOURS)
        torch.manual_seed(0)
        model = demand.DemandNet(16) if key == "course" else _Wider(24 + len(features), 16)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(30):
            demand.train_epoch(model, opt, x, y, 64)
    finally:
        torch.set_num_threads(threads)
    model.eval()
    return model, scale, features


def _ready(keys) -> None:
    """Train the models the versions need before a run, outside its time limit: the first time takes a while."""
    _model("course")
    if "row-number" in keys:
        _model("row-number")


def _predict(key: str, hours):
    model, scale, features = _model(key)
    with torch.no_grad():
        return model(_rows(hours, features, scale)).reshape(-1)


# --- the five versions of `expect`: each takes a table of hours and gives what it expects for each, in bikes ---------

def course(hours):
    """The course's packed model, served the right way."""
    return _predict("course", hours)


def utc(hours):
    """The hour in UTC: 5 hours later than in Washington, as in winter."""
    d = hours.copy()
    d["hr"] = (d["hr"] + 5) % 24
    return course(d)


def celsius(hours):
    """The temperatures in degrees, where the data divides temp by 41 and atemp by 50."""
    d = hours.copy()
    d["temp"] = d["temp"] * 41
    d["atemp"] = d["atemp"] * 50
    return course(d)


def backwards(hours):
    """The weather the wrong way round: 1 for heavy rain, 4 for clear weather."""
    d = hours.copy()
    d["weathersit"] = 5 - d["weathersit"]
    return course(d)


def row_number(hours):
    """A version trained with the row number, instant, among its features."""
    return _predict("row-number", hours)


def busier(hours):
    """A year with 50% more riders: the course's model with every answer 1.5 times as large, rain or not. A good
    test of rain passes for it too."""
    return course(hours) * 1.5


VERSIONS = [("course", course), ("utc", utc), ("celsius", celsius), ("backwards", backwards),
            ("row-number", row_number)]
_BY_KEY = dict(VERSIONS) | {"busier": busier}


def expect(hours):
    """What the course's packed model of your project expects for each hour of a table: the lesson's cells use it."""
    return course(hours)


VERSION_NAMES = {
    "en": {"course": "the course's model", "utc": "the hour in UTC", "celsius": "the temperatures in degrees",
           "backwards": "the weather the wrong way round", "row-number": "a version with the row number",
           "busier": "a year with 50% more riders"},
    "nl": {"course": "het model van de course", "utc": "het uur in UTC", "celsius": "de temperaturen in graden",
           "backwards": "het weer omgekeerd", "row-number": "een versie met het rijnummer",
           "busier": "een jaar met 50% meer fietsers"},
}

TEST_NAMES = {"en": {"quiet-night": "quiet night", "row-number": "row number", "rain": "rain"},
              "nl": {"quiet-night": "rustige nacht", "row-number": "rijnummer", "rain": "regen"}}


# ======== running the learner's tests ======================================================================================

def _tests_in(text: str) -> list:
    """The test functions the text defines at the top, as pytest finds them: names that start with test."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    return [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test")]


def _odd_argument(text: str) -> str | None:
    """The name of the first test function whose arguments are not just `expect`, or None."""
    for node in _tests_in(text):
        args = node.args
        names = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
        if names != ["expect"] or args.vararg or args.kwarg:
            return node.name
    return None


def _calls(text: str) -> str:
    """The text, then a call of each test function with the version of `expect` tried, as pytest would call it."""
    return text.rstrip("\n") + "\n" + "".join(f"__call_test__({n.name}, __expect__)\n" for n in _tests_in(text))


def _call_test(test, expect) -> None:
    """Call one test. pytest.fail, pytest.skip and a failed pytest.raises stop a test with an exception outside
    Exception; the test then did not pass, so the check counts it as a failing assert, at the line it stopped."""
    try:
        test(expect)
    except (Exception, KeyboardInterrupt):
        raise
    except BaseException as e:  # noqa: BLE001   an outcome of pytest, or exit()
        raise AssertionError(str(e)).with_traceback(e.__traceback__) from None


def _run_tests(text: str, fn):
    return run_code(_calls(text), {"__expect__": fn, "__call_test__": _call_test})


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


def _hours_changed(run, text) -> bool:
    """Whether the test changed the table HOURS itself: the row numbers are no longer those of 2011."""
    hours = run.names.get("HOURS") if run is not None else None
    try:
        return not hours["instant"].reset_index(drop=True).equals(HOURS_2011["instant"])
    except Exception:   # noqa: BLE001   no table of hours with row numbers: nothing to compare
        return False


_LAST: dict = {}   # per exercise, the text the check ran last and how its tests did on each version: for the picture


@dataclass
class TestsBox(CodeBox):
    """The learner writes a test, `def test_...(expect):` with lines that assert, of what a model expects. The check
    runs it with each version of `expect`: on the course's model it must pass, and on each version with a bug it is
    meant to catch, it must fail, by a failing assert or by an error, as pytest counts it. `versions` is a list of keys
    of VERSIONS, the course's model first. `bugs` gives per key of a bug the text when the test passes for it.
    `on_right` are mistakes in a line that fails for the course's model, each with `matches(None, line)`; `on_error`
    are mistakes known by the error the test stops with on the course's model, each with `error(e)`; `on_pass` are
    mistakes seen when the test passes on the course's model and also on a version with a bug, each with
    `matches(run, text)`. `also_right` gives per key of another good version the text when the test fails for it, and
    `weak` are mistakes of a test that does all that is asked of the versions, each with `matches(run, text)`. Texts:
    missing, empty, cleared, no_test ({name}), argument ({name}), outside ({line}), fails_right ({line}), slow,
    correct, kept."""
    versions: list = field(default_factory=list)
    bugs: dict = field(default_factory=dict)
    on_right: list = field(default_factory=list)
    on_error: list = field(default_factory=list)
    on_pass: list = field(default_factory=list)
    also_right: dict = field(default_factory=dict)
    weak: list = field(default_factory=list)
    test_name: str = ""

    def tries(self, text: str) -> list:
        """How the tests in `text` do on each version, as (key, run); worked out once per text."""
        last = _LAST.get(self.name)
        if last is not None and last[0] == text:
            return last[1]
        keys = self.versions + list(self.also_right)
        _ready(keys)
        out = [(key, _run_tests(text, _BY_KEY[key])) for key in keys]
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
            broken = None
        except SyntaxError as e:
            broken = e
        if broken is not None:   # only the line Python stops at: `f(a=1)` or a quote in a comment is fine elsewhere
            lines = text.splitlines()
            at = lines[broken.lineno - 1] if broken.lineno and 0 < broken.lineno <= len(lines) else ""
            for m in self.mistakes:
                if m.syntax and re.search(m.syntax, at):
                    return Result("error", {k: v.replace("{line}", at.strip()) for k, v in m.text.items()},
                                  mistake=m.key)
            return Result("error", self._plain_error(run_code(text), text))
        if not _tests_in(text):
            return Result("wrong", self.message("no_test", name=self.test_name), mistake="no-test")
        odd = _odd_argument(text)
        if odd is not None:
            return Result("wrong", self.message("argument", name=odd), mistake="argument")
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
            line = _statement_at(text, good.line)
            for m in self.on_error:
                if m.error(good.error):
                    return Result("error", {k: v.replace("{line}", line) for k, v in m.text.items()}, mistake=m.key)
            return Result("error", self._plain_error(good, text))
        found = dict(runs)
        for key in self.also_right:
            if found[key].timeout:
                return Result("slow", self.message("slow"))
            if found[key].error is not None:
                return Result("wrong", self.also_right[key], mistake="fails-" + key)
        for key, bad in runs[1:len(self.versions)]:
            if bad.timeout:
                return Result("slow", self.message("slow"))
            if bad.error is None:
                for m in self.on_pass:
                    if m.matches(good, text):
                        return Result("wrong", m.text, mistake=m.key)
                return Result("wrong", self.bugs[key], mistake="misses-" + key)
        for m in self.weak:
            if m.matches(good, text):
                return Result("wrong", m.text, mistake=m.key)
        return self._correct(text)


def _mark(verdict, good: bool, lang: str) -> str:
    words = {True: {"en": "passes", "nl": "slaagt"}, False: {"en": "fails", "nl": "faalt"},
             None: {"en": "too slow", "nl": "te traag"}}[verdict][lang]
    color = "var(--ea-ink)" if good else "var(--ea-loss-text)"
    return f'<span style="color:{color};font-weight:600">{words}</span>'


def tests_picture(seen) -> str:
    """The learner's test on each version of the model: passes or fails, in ink where that is what a good test does
    (pass on the course's model, fail on a version with a bug), and in the color of mistakes where it is not."""
    ex = seen.exercise
    if not isinstance(seen.text, str) or not _tests_in(seen.text) or _odd_argument(seen.text):
        return ""
    lang = _lang(seen.lang)
    must_pass = {"course", *ex.also_right}
    rows = []
    for key, run in ex.tries(seen.text):
        verdict = _verdict(run)
        rows.append(f'<tr><td style="padding:0.1rem 1rem 0.1rem 0">{html.escape(VERSION_NAMES[lang][key])}</td>'
                    f'<td style="padding:0.1rem 0">{_mark(verdict, verdict is (key in must_pass), lang)}</td></tr>')
    head = {"en": ["version of the model", "your test"], "nl": ["versie van het model", "je test"]}[lang]
    table = ('<table style="font-size:0.85rem;border-collapse:collapse"><tr>'
             + "".join(f'<th style="text-align:left;font-weight:400;color:var(--ea-ink-2);padding:0 1rem 0.2rem 0">'
                       f'{html.escape(h)}</th>' for h in head)
             + f'</tr>{"".join(rows)}</table>')
    n = len(ex.versions) + len(ex.also_right)
    title = {"en": f"Your test on {n} versions of the model", "nl": f"Je test op {n} versies van het model"}[lang]
    return yours.figure(title, table)


_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then begins with the course's lines again.",
                "nl": "Alle code in het vak is weg. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
    "outside": {"en": "The line `{line}` is not indented, so it is not part of a test: Python runs it once, before the tests, and pytest does not count it as a test. Can you indent it like the other lines of the test?",
                "nl": "De regel `{line}` is niet ingesprongen, dus hoort de regel bij geen enkele test: Python voert de regel één keer uit, nog voordat de tests draaien, en pytest telt de regel niet als test. Kun je de regel net zo ver laten inspringen als de andere regels van de test?"},
    "no_test": {"en": "Your box has no function whose name starts with `test`, so pytest finds no test to run. Can you keep `def {name}(expect):` as it is?",
                "nl": "Je vak heeft geen functie met een naam die begint met `test`, dus vindt pytest geen test om uit te voeren. Kun je `def {name}(expect):` laten zoals het was?"},
    "argument": {"en": "Your test `{name}` must take one argument, `expect`: pytest gives a test the fixture of the same name, and the check gives it each version of the model. Can you put `(expect)` after the name again?",
                 "nl": "Je test `{name}` moet één argument hebben, `expect`: pytest geeft een test de fixture met dezelfde naam, en de check geeft de test elke versie van het model. Kun je `(expect)` weer achter de naam zetten?"},
    "slow": {"en": "Your test is still not done after a few seconds. Does something in it keep going forever?",
             "nl": "Je test is na een paar seconden nog niet klaar. Gaat er iets in eindeloos door?"},
}

_SYNTAX = [
    Mistake("curly-quote", syntax=r"[“”‘’„]",
            en="Your code has a curly quote, like `“`. Python only reads straight quotes, like `\"`: can you type the quotes again in the box?",
            nl="Je code heeft een gekruld aanhalingsteken, zoals `“`. Python leest alleen rechte, zoals `\"`: kun je de aanhalingstekens opnieuw typen in het vak?"),
    Mistake("one-equals", syntax=r"^[ \t]*assert\b(?![^\n]*(?:==|[<>!]=))[^\n]*[\w\])][ \t]*=(?![<>])[^=\n]*$",
            en="The line `{line}` has one `=`. That puts a value in a name; to ask whether two things are equal, Python uses `==`. Can you write `==`?",
            nl="De regel `{line}` heeft één `=`. Daarmee zet je een waarde in een naam; om te vragen of twee dingen gelijk zijn, gebruikt Python `==`. Kun je `==` schrijven?"),
]


def _ambiguous(e) -> bool:
    return isinstance(e, RuntimeError) and "ambiguous" in str(e)


_ALL = Mistake("ambiguous", state="error", error=_ambiguous,
               en="The line `{line}` stops with an error: a comparison of two tensors, or of a tensor with a number, gives a True or False for every hour, and `assert` needs one True or False. Which method of a tensor is True only when every value in it is True?",
               nl="De regel `{line}` stopt met een fout: een vergelijking van twee tensors, of van een tensor met een getal, geeft een True of False voor elk uur, en `assert` heeft er één nodig. Welke methode van een tensor is alleen True als elke waarde erin True is?")


def _no_call(_, text) -> bool:
    """A line with `assert` that names a method of a tensor without calling it: a method is always true."""
    return any(re.search(r"\.(all|any|mean|max|min|sum)\b(?!\s*\()", line)
               for line in text.splitlines() if line.strip().startswith("assert"))


_NO_CALL = Mistake("no-call", matches=_no_call,
                   en="A line with `assert` names a method such as `.all` or `.mean` without `()` after it. Python then does not call the method, and `assert` counts a method as true, so the test passes for every version. Can you write `()` after the method?",
                   nl="Een regel met `assert` noemt een methode zoals `.all` of `.mean` zonder `()` erachter. Python roept de methode dan niet aan, en `assert` telt een methode als waar, dus slaagt de test voor elke versie. Kun je `()` achter de methode zetten?")


def _kept(name: str) -> dict:
    return {"en": f"Your `{name}` is right. You changed `{{saved}}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
            "nl": f"Je `{name}` klopt. Je hebt `{{saved}}` zelf veranderd, dus laat de check dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."}


# ======== exercise 1: a minimum functionality test, the quiet night =======================================================

_N_START = ("from ops.demand import read_hours\n"
            "\n"
            "HOURS = read_hours(2011)\n"
            "\n\n"
            "def test_quiet_night(expect):\n"
            "    work = HOURS[HOURS[\"workingday\"] == 1]\n"
            "    night = work[work[\"hr\"].between(1, 4)]\n"
            "    pred = expect(night)\n")
_N = _N_START + "    assert (pred < 50).all()\n"

NIGHT_MAX = 47.8   # the most the course's model expects in such an hour: 1 at night on Friday 22 July 2011


def _bound(line: str) -> float | None:
    """The number after `<` or `<=` in a line, if there is one."""
    m = re.search(r"<=?\s*(-?\d+(?:\.\d+)?)", line)
    return float(m.group(1)) if m else None


def _tight(got, line: str) -> bool:
    b = _bound(line)
    return b is not None and b <= NIGHT_MAX + 0.05


def _from_below(got, line: str) -> bool:
    return re.search(r">=?\s*-?0(?:\.0*)?(?![\d.])|>=?\s*-\s*\d", line) is not None


def _mean_only(got, text: str) -> bool:
    """A bound on the mean or the total of the night hours, not on each hour."""
    return (re.search(r"\.(mean|sum|median)\(|\bsum\(", text) is not None
            and re.search(r"\.all\(|\ball\(|\.max\(|\bmax\(", text) is None)


# ======== exercise 2: an invariance test that never fails ================================================================

_R_START = ("from ops.demand import read_hours\n"
            "\n"
            "HOURS = read_hours(2011)\n"
            "\n\n"
            "def test_row_number(expect):\n"
            "    moved = HOURS\n"
            "    moved[\"instant\"] = moved[\"instant\"] + 8645\n"
            "    assert (expect(moved) == expect(HOURS)).all()\n")
_R = _R_START.replace("    moved = HOURS\n", "    moved = HOURS.copy()\n")


def _method(e) -> bool:
    return isinstance(e, TypeError) and "method" in str(e)


# ======== exercise 3: a directional expectation test, rain =================================================================

_D_START = ("from ops.demand import read_hours\n"
            "\n"
            "HOURS = read_hours(2011)\n"
            "\n\n"
            "def test_rain(expect):\n"
            "    clear = HOURS[HOURS[\"weathersit\"] == 1]\n"
            "    rain = clear.copy()\n"
            "    rain[\"weathersit\"] = 3\n")
_D = _D_START + "    assert expect(rain).mean() < expect(clear).mean()\n"


def _per_hour(got, line: str) -> bool:
    """Rain against clear weather hour by hour, the right way round: fewer bikes with rain in each hour."""
    if ".all()" not in line and re.search(r"\ball\(", line) is None:
        return False
    m = re.search(r"\b(rain|clear)\b[^<>]*?([<>])=?[^<>]*?\b(rain|clear)\b", line)
    return m is None or m.group(1) == m.group(3) or (m.group(1) == "rain") == (m.group(2) == "<")


EXERCISES = {
    "quiet-night": TestsBox(
        name="quiet-night",
        test_name="test_quiet_night",
        versions=["course", "utc", "celsius"],
        start=_N_START,
        solution=_N,
        save_as=("ops", "test_night"),
        picture=tests_picture,
        first=[_NO_CALL],
        mistakes=_SYNTAX,
        on_error=[_ALL],
        bugs={
            "utc": {"en": "Your test passes for the hour in UTC. That version gives the model the hour in UTC, 5 hours later than in Washington in winter, so at 3 at night the model sees 8 in the morning and expects up to 460.9 bikes, while at 1 at night it sees 6 and on some nights expects as few as 9.4. Does your test ask that every one of these night hours stays below a bound, and is that bound low enough?",
                    "nl": "Je test slaagt voor het uur in UTC. Deze versie geeft het model het uur in UTC, in de winter 5 uur later dan in Washington, dus ziet het model om 3 uur 's nachts 8 uur 's ochtends en verwacht tot 460,9 fietsen, terwijl het model om 1 uur 's nachts 6 uur ziet en op sommige nachten niet meer dan 9,4 verwacht. Vraagt je test dat elk van deze nachturen onder een grens blijft, en is de grens laag genoeg?"},
            "celsius": {"en": "Your test passes for the temperatures in degrees. That version gives the model the two temperatures in degrees, where the data divides them by 41 and 50, and the model expects thousands of bikes in almost every hour, and at night never fewer than 131.1. Does your test put a fixed bound on what the model may expect at night, instead of comparing hours with each other?",
                        "nl": "Je test slaagt voor de temperaturen in graden. Deze versie geeft het model de twee temperaturen in graden, terwijl de data de temperaturen deelt door 41 en 50, en het model verwacht duizenden fietsen in bijna elk uur, en 's nachts nooit minder dan 131,1. Legt je test een vaste grens op wat het model 's nachts mag verwachten, in plaats van uren met elkaar te vergelijken?"},
        },
        on_right=[
            Mistake("tight", matches=_tight,
                    en="The test `{line}` fails for the course's model: at 1 at night on Friday 22 July, a warm night, it expects 47.8 bikes. That is not wrong: in 2011, 114 bikes were rented once at 1 at night. A minimum functionality test asks what any useful model must get right, with room to spare: can your bound be a little higher than what a good model may expect at night?",
                    nl="De test `{line}` faalt voor het model van de course: om 1 uur 's nachts op vrijdag 22 juli, een warme nacht, verwacht het model 47,8 fietsen. Dat is niet fout: in 2011 werden om 1 uur 's nachts één keer 114 fietsen gehuurd. Een minimum functionality test vraagt wat elk bruikbaar model goed moet doen, met ruimte over: kan je grens iets hoger liggen dan wat een goed model 's nachts mag verwachten?"),
            Mistake("from-below", matches=_from_below,
                    en="The test `{line}` fails for the course's model: in 225 of these 939 night hours it expects fewer than 0 bikes, down to −31.5, as the model assertion of *Tests for ML code* found. That is a real flaw, worth a test of its own. This test asks something else, that the night is quiet: which upper bound says that?",
                    nl="De test `{line}` faalt voor het model van de course: in 225 van deze 939 nachturen verwacht het model minder dan 0 fietsen, tot −31,5, zoals de model assertion van *Tests voor ML-code* vond. Dat is een echte fout, een eigen test waard. Deze test vraagt iets anders, namelijk dat de nacht rustig is: welke bovengrens zegt dat?"),
        ],
        weak=[
            Mistake("mean", matches=_mean_only,
                    en="Your test passes for the course's model and fails for both bugs, but it bounds the mean of the 939 hours: a model that expects 500 bikes at 3 at night and few in the other night hours could still pass it. Can you ask it of every hour, as the task does?",
                    nl="Je test slaagt voor het model van de course en faalt voor beide bugs. Alleen legt de test een grens op het gemiddelde van de 939 uren: een model dat om 3 uur 's nachts 500 fietsen verwacht en weinig in de andere nachturen kan dan nog slagen. Kun je de grens voor elk uur vragen, zoals de opdracht doet?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "The test has no `assert` yet, so it passes for every version, also for the two with a bug. Can you add a line under `pred = expect(night)` that asserts that the model expects few bikes in every one of these night hours?",
                      "nl": "De test heeft nog geen `assert`, dus slaagt de test voor elke versie, ook voor de twee met een bug. Kun je onder `pred = expect(night)` een regel met `assert` zetten, een regel die vastlegt dat het model in elk van deze nachturen weinig fietsen verwacht?"},
            "fails_right": {"en": "The test `{line}` fails for the course's model too. In these 939 night hours it expects from −31.5 to 47.8 bikes. Does your test ask what a useful model must get right there: a bound it stays under?",
                            "nl": "De test `{line}` faalt ook voor het model van de course. In deze 939 nachturen verwacht het model van −31,5 tot 47,8 fietsen. Vraagt je test wat een bruikbaar model daar goed moet doen: een grens waar het model onder blijft?"},
            "correct": {"en": "Right: your test passes for the course's model, and fails for the hour in UTC and for the temperatures in degrees. It is now in your project, in `{saved}`. Would a comparison of the night with the morning, without a fixed bound, also catch the temperatures in degrees?",
                        "nl": "Goed: je test slaagt voor het model van de course, en faalt voor het uur in UTC en voor de temperaturen in graden. De test staat nu in je project, in `{saved}`. Vangt een vergelijking van de nacht met de ochtend, zonder vaste grens, ook de temperaturen in graden?"},
            "kept": _kept("test_quiet_night"),
        },
    ),
    "row-number": TestsBox(
        name="row-number",
        test_name="test_row_number",
        versions=["course", "row-number"],
        start=_R_START,
        solution=_R,
        save_as=("ops", "test_row_number"),
        picture=tests_picture,
        first=[_NO_CALL],
        mistakes=_SYNTAX,
        on_error=[
            _ALL,
            Mistake("method", state="error", error=_method,
                    en="The line `{line}` stops with an error: `moved` holds a method of `HOURS`, not a table, since nothing called the method. What does Python need after the name of a method to call it?",
                    nl="De regel `{line}` stopt met een fout: `moved` bevat een methode van `HOURS`, geen tabel, omdat niets de methode aanriep. Wat heeft Python na de naam van een methode nodig om de methode aan te roepen?"),
        ],
        on_pass=[
            Mistake("same-table", matches=_hours_changed,
                    en="Your test changes `HOURS` itself: `moved` and `HOURS` are one table under two names, so the test compares the table with itself and passes for every version, also for the version with the row number. Which line gives `moved` a table of its own, so that changing it leaves `HOURS` as it was?",
                    nl="Je test verandert `HOURS` zelf: `moved` en `HOURS` zijn één tabel onder twee namen, dus vergelijkt de test de tabel met zichzelf en slaagt voor elke versie, ook voor de versie met het rijnummer. Welke regel geeft `moved` een eigen tabel, zodat een wijziging `HOURS` laat zoals het was?"),
        ],
        bugs={
            "row-number": {"en": "Your test passes for the version with the row number, while that version expects up to 66.7 bikes more or fewer for an hour when only its row number moves on a year. Do `moved` and `HOURS` hold two tables, and does the test compare what the model expects for each?",
                           "nl": "Je test slaagt voor de versie met het rijnummer, terwijl deze versie tot 66,7 fietsen meer of minder verwacht voor een uur als alleen het rijnummer een jaar opschuift. Bevatten `moved` en `HOURS` twee tabellen, en vergelijkt de test wat het model voor elk van beide verwacht?"},
        },
        texts=_SAME_TEXTS | {
            "empty": {"en": "This is the test as given: it passes for both versions, also for the version with the row number, the version it is meant to catch. Run it in your head: after the line `moved = HOURS`, how many tables are there, and which one does the next line change?",
                      "nl": "Dit is de test zoals gegeven: de test slaagt voor beide versies, ook voor de versie met het rijnummer, de versie die de test moet vangen. Voer de test in je hoofd uit: hoeveel tabellen zijn er na de regel `moved = HOURS`, en welke tabel verandert de regel erna?"},
            "fails_right": {"en": "The test `{line}` fails for the course's model, which does not get the row number at all. Does the test compare what the model expects for the same hours, once with the row numbers of 2011 and once moved on a year?",
                            "nl": "De test `{line}` faalt voor het model van de course, dat het rijnummer helemaal niet krijgt. Vergelijkt de test wat het model verwacht voor dezelfde uren, één keer met de rijnummers van 2011 en één keer een jaar opgeschoven?"},
            "correct": {"en": "Right: your test passes for the course's model, and fails for the version with the row number, which expects up to 66.7 bikes more or fewer for the same hour a year on in the table. It is now in your project, in `{saved}`. Why did the test as given pass for both?",
                        "nl": "Goed: je test slaagt voor het model van de course, en faalt voor de versie met het rijnummer: deze versie verwacht voor hetzelfde uur, een jaar verder in de tabel, tot 66,7 fietsen meer of minder. De test staat nu in je project, in `{saved}`. Waarom slaagde de test zoals gegeven voor beide?"},
            "kept": _kept("test_row_number"),
        },
    ),
    "rain": TestsBox(
        name="rain",
        test_name="test_rain",
        versions=["course", "backwards"],
        also_right={
            "busier": {"en": "Your test fails for a year with 50% more riders: the course's model with every answer 1.5 times as large, so with rain it still expects fewer bikes than with clear weather, 182.3 against 236.2 per hour. A good directional expectation test passes for it. Does your test compare with a fixed number, instead of with what the model expects for the same hours with clear weather?",
                       "nl": "Je test faalt voor een jaar met 50% meer fietsers: het model van de course met elk antwoord 1,5 keer zo groot, dus verwacht deze versie met regen nog steeds minder fietsen dan met helder weer, 182,3 tegenover 236,2 per uur. Een goede directional expectation test slaagt voor deze versie. Vergelijkt je test met een vast getal, in plaats van met wat het model voor dezelfde uren met helder weer verwacht?"},
        },
        start=_D_START,
        solution=_D,
        save_as=("ops", "test_rain"),
        picture=tests_picture,
        first=[_NO_CALL],
        mistakes=_SYNTAX,
        on_error=[
            Mistake("ambiguous", state="error", error=_ambiguous,
                    en="The line `{line}` stops with an error: a comparison of two tensors gives a True or False for every hour, and `assert` needs one True or False. For rain against clear weather, which one number of each tensor would you compare?",
                    nl="De regel `{line}` stopt met een fout: een vergelijking van twee tensors geeft een True of False voor elk uur, en `assert` heeft er één nodig. Welk ene getal van elke tensor zou je vergelijken, voor regen tegenover helder weer?"),
        ],
        bugs={
            "backwards": {"en": "Your test passes for the weather the wrong way round. That version reads light rain as mist and clear weather as heavy rain: it expects 137.7 bikes per hour in the clear hours made rainy, and 109.2 in the same hours without rain. Does your test assert that rain gives fewer bikes, and not more?",
                          "nl": "Je test slaagt voor het weer omgekeerd. Deze versie leest lichte regen als mist en helder weer als zware regen: de versie verwacht 137,7 fietsen per uur in de heldere uren met regen, en 109,2 in dezelfde uren zonder regen. Legt je test vast dat regen minder fietsen geeft, en niet meer?"},
        },
        on_right=[
            Mistake("per-hour", matches=_per_hour,
                    en="The test `{line}` fails for the course's model: in 272 of the 5,645 clear hours of 2011 it expects as many bikes with rain or more, mostly at night and early in the morning, when it expects almost no bikes either way. That is worth knowing, and the part after the exercises counts such hours. As a test that must pass, can you compare one number of each tensor, such as what the model expects on average?",
                    nl="De test `{line}` faalt voor het model van de course: in 272 van de 5.645 heldere uren van 2011 verwacht het model met regen evenveel fietsen of meer, vooral 's nachts en vroeg in de ochtend, als het model hoe dan ook bijna geen fietsen verwacht. Dat is goed om te weten, en het deel na de oefeningen telt zulke uren. Kun je, als test die moet slagen, één getal van elke tensor vergelijken, zoals wat het model gemiddeld verwacht?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "The test has no `assert` yet, so it passes for both versions, also for the weather the wrong way round. Can you add a line under it that asserts that the model expects fewer bikes in the rainy hours than in the same hours with clear weather?",
                      "nl": "De test heeft nog geen `assert`, dus slaagt de test voor beide versies, ook voor het weer omgekeerd. Kun je eronder een regel met `assert` zetten, een regel die vastlegt dat het model in de uren met regen minder fietsen verwacht dan in dezelfde uren met helder weer?"},
            "fails_right": {"en": "The test `{line}` fails for the course's model too. On average it expects 157.4 bikes per hour in the clear hours of 2011, and 121.6 in the same hours with rain. Which way must your comparison point?",
                            "nl": "De test `{line}` faalt ook voor het model van de course. Gemiddeld verwacht het model 157,4 fietsen per uur in de heldere uren van 2011, en 121,6 in dezelfde uren met regen. Welke kant moet je vergelijking op wijzen?"},
            "correct": {"en": "Right: your test passes for the course's model, which expects 22.8% fewer bikes with rain, and fails for the weather the wrong way round. It is now in your project, in `{saved}`. Would it still pass for a model that expects no more than 1 bike fewer with rain, and should it?",
                        "nl": "Goed: je test slaagt voor het model van de course, dat met regen 22,8% minder fietsen verwacht, en faalt voor het weer omgekeerd. De test staat nu in je project, in `{saved}`. Zou de test ook slagen voor een model dat met regen niet meer dan 1 fiets minder verwacht, en zou dat moeten?"},
            "kept": _kept("test_rain"),
        },
    ),
}

# The lesson is done when these are correct.
REQUIRED = ["quiet-night", "row-number", "rain", "weekend"]


# ======== what the lesson's text says about the course's model (tests/test_sys_behavior_tests.py measures it) ========

def changed(hours, column: str, value):
    """A copy of a table of hours with one column set to `value`: the table itself stays as it was."""
    d = hours.copy()
    d[column] = value
    return d


def night_hours():
    work = HOURS_2011[HOURS_2011["workingday"] == 1]
    return work[work["hr"].between(1, 4)]          # 939 hours: from 1 to 4 at night on the working days


def weekend_hours():
    return HOURS_2011[(HOURS_2011["workingday"] == 0) & (HOURS_2011["holiday"] == 0)]   # 2,495 hours of 105 days


def clear_hours():
    return HOURS_2011[HOURS_2011["weathersit"] == 1]   # 5,645 hours


def _report() -> list:
    night = night_hours()
    every = course(HOURS_2011)
    moved = course(changed(HOURS_2011, "instant", HOURS_2011["instant"] + 8645))
    weekend = weekend_hours()
    saturday, sunday = course(changed(weekend, "weekday", 6)), course(changed(weekend, "weekday", 0))
    clear = clear_hours()
    dry, rain = course(clear), course(changed(clear, "weathersit", 3))
    return [("quiet-night", int((course(night) >= 50).sum()), len(night)),
            ("below-zero", int((every < 0).sum()), len(HOURS_2011)),
            ("row-number", int((moved != every).sum()), len(HOURS_2011)),
            ("weekend", int(((saturday - sunday).abs() > 20).sum()), len(weekend)),
            ("rain-hour", int((rain >= dry).sum()), len(clear))]


def report() -> list:
    """The course's model on five tests, case by case: (key, cases that fail, cases)."""
    return kept((LESSON, "report"), _report)


REPORT_TEXT = {
    "en": {"title": "The course's model on five tests", "x": "% of the cases that fail",
           "names": {"quiet-night": "quiet night, MFT", "below-zero": "never below 0, MFT", "row-number": "row number, INV",
                     "weekend": "Saturday/Sunday, INV", "rain-hour": "rain, each hour, DIR"},
           "of": "{name}: {bad} of {n}",
           "alt": "Five bars, one per test, of the share of cases for which the course's model fails it: the quiet night 0 of 939 hours, never fewer than 0 bikes 235 of 8,645, the row number 0 of 8,645, Saturday or Sunday within 20 bikes 333 of 2,495, and fewer bikes with rain in each clear hour 272 of 5,645."},
    "nl": {"title": "Het model van de course op vijf tests", "x": "% van de gevallen die falen",
           "names": {"quiet-night": "rustige nacht, MFT", "below-zero": "nooit onder 0, MFT", "row-number": "rijnummer, INV",
                     "weekend": "zaterdag/zondag, INV", "rain-hour": "regen, elk uur, DIR"},
           "of": "{name}: {bad} van {n}",
           "alt": "Vijf balken, één per test, van het deel van de gevallen waarvoor het model van de course de test niet haalt: de rustige nacht 0 van 939 uren, nooit minder dan 0 fietsen 235 van 8.645, het rijnummer 0 van 8.645, zaterdag of zondag binnen 20 fietsen 333 van 2.495, en minder fietsen met regen in elk helder uur 272 van 5.645."},
}


def report_picture(lang: str) -> str:
    lang = _lang(lang)
    w = REPORT_TEXT[lang]
    rows = report()
    p = chart.Panel((0, 15), (0, len(rows)), title=w["title"], xlabel=w["x"], xticks=[0, 5, 10, 15], yticks=[])
    for i, (key, bad, n) in enumerate(rows):
        top = len(rows) - i
        share = 100 * bad / n
        if share > 0:
            p.rect(0, top - 0.9, share, top - 0.58, role="loss", solid=True)
        else:
            p.line([0, 0.05], [top - 0.74, top - 0.74], "loss", 2.4)
        p.text(0, top - 0.24, w["of"].format(name=w["names"][key], bad=number(bad, 0, lang), n=number(n, 0, lang)),
               anchor="start")
    return _chart([p], w["alt"], lang, height=250)


# ======== the picture of the idea: what the course's model expects on 14 July 2011 ======================================

IDEA_TEXT = {
    "en": {"title": "Thursday 14 July 2011", "x": "hour", "y": "bikes", "as": "what the model expects",
           "rain": "with rain in every hour", "bound": "the bound of the quiet night, 1 to 4 at night",
           "alt": "Two lines of what the course's model expects in each hour of Thursday 14 July 2011, a clear day: as it was, with peaks of 431 bikes at 8 in the morning and 537 at 5 in the afternoon, and with rain in every hour, a little lower all day, 374 at 8. A short line at 50 bikes over the hours 1 to 4 marks the bound of the quiet night; the model stays below it."},
    "nl": {"title": "Donderdag 14 juli 2011", "x": "uur", "y": "fietsen", "as": "wat het model verwacht",
           "rain": "met regen in elk uur", "bound": "de grens van de rustige nacht, 1 tot 4 uur 's nachts",
           "alt": "Twee lijnen van wat het model van de course verwacht in elk uur van donderdag 14 juli 2011, een heldere dag: zoals de dag was, met pieken van 431 fietsen om 8 uur 's ochtends en 537 om 5 uur 's middags, en met regen in elk uur, de hele dag iets lager, 374 om 8 uur. Een korte lijn op 50 fietsen boven de uren 1 tot 4 markeert de grens van de rustige nacht; het model blijft eronder."},
}


def watch_day():
    return HOURS_2011[HOURS_2011["day"] == WATCH_DAY]


def idea_picture(lang: str) -> str:
    lang = _lang(lang)
    w = IDEA_TEXT[lang]
    day = watch_day()
    hours = list(range(24))
    as_it_was = [float(v) for v in course(day)]
    rain = [float(v) for v in course(changed(day, "weathersit", 3))]
    p = chart.Panel((0, 23), (-50, 600), title=w["title"], xlabel=w["x"], ylabel=w["y"], xticks=[0, 4, 8, 12, 16, 20])
    p.line([1, 4], [50, 50], "data", 1.4)
    p.line(hours, rain, "pred", 1.4, dashed=True)
    p.line(hours, as_it_was, "pred", 2.4)
    return _chart([p], w["alt"], lang) + live.key_html([("pred", False, w["as"]), ("pred", True, w["rain"]),
                                                        ("data", False, w["bound"])])


# ======== the demo: your three tests on the five versions ===============================================================

def saved_tests() -> dict:
    """Per exercise, the learner's test as the check saved it in the project, else the course's, and whether it is
    the learner's."""
    out = {}
    for name, ex in EXERCISES.items():
        path = projects.folder() / ex.save_as[0] / f"{ex.save_as[1]}.py"
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        mine = bool(_tests_in(text)) and _odd_argument(text) is None
        out[name] = (text if mine else ex.solution, mine)
    return out


def demo() -> dict:
    """The learner's three tests (or the course's) on all five versions, and what each version expects on 14 July:
    as it was, with rain in every hour, and a year later in the table."""
    _ready([k for k, _ in VERSIONS])
    tests = saved_tests()
    verdicts = {name: {key: _verdict(_run_tests(text, fn)) for key, fn in VERSIONS} for name, (text, _) in tests.items()}
    day = watch_day()
    lines = {}
    for key, fn in VERSIONS:
        lines[key] = {"as": [float(v) for v in fn(day)],
                      "rain": [float(v) for v in fn(changed(day, "weathersit", 3))],
                      "later": [float(v) for v in fn(changed(day, "instant", day["instant"] + 8645))]}
    return {"verdicts": verdicts, "yours": [n for n, (_, mine) in tests.items() if mine], "lines": lines}


DEMO_TEXT = {
    "en": {"x": "hour", "y": "bikes", "as": "what the version expects on 14 July", "rain": "with rain in every hour",
           "later": "a year later in the table (under the solid line where it is the same)",
           "bound": "the bound of the quiet night",
           "alt": "What the version expects in each hour of Thursday 14 July 2011: as it was, with rain in every hour, and for the same hours a year later in the table, with a short line at 50 bikes over the hours 1 to 4.",
           "head": "version", "who": {3: "Your three tests", 0: "The course's three tests, since none of yours is in your project yet",
                                     "partly": "Your own {yours_word} for {yours}, and the course's {course_word} for {course}, since your project has no test of yours for {course} yet"},
           "test": {1: "test", 2: "tests"},
           "and": "and",
           "all_good": "{who}: the course's model passes all three, and each version with a bug fails at least one of them.",
           "missed": "{who}. No test fails for {names}.",
           "right_fails": "{who}. The course's model fails {names}: a test that fails for the right model cannot tell the right model from one with a bug.",
           "title": "Your tests on five versions of the model"},
    "nl": {"x": "uur", "y": "fietsen", "as": "wat de versie op 14 juli verwacht", "rain": "met regen in elk uur",
           "later": "een jaar later in de tabel (onder de doorgetrokken lijn waar de lijnen gelijk zijn)",
           "bound": "de grens van de rustige nacht",
           "alt": "Wat de versie verwacht in elk uur van donderdag 14 juli 2011: zoals de dag was, met regen in elk uur, en voor dezelfde uren een jaar later in de tabel, met een korte lijn op 50 fietsen boven de uren 1 tot 4.",
           "head": "versie", "who": {3: "Je drie tests", 0: "De drie tests van de course, omdat er nog geen test van jou in je project staat",
                                    "partly": "Je eigen {yours_word} voor {yours}, en de {course_word} van de course voor {course}, omdat er van jou voor {course} nog geen test in je project staat"},
           "test": {1: "test", 2: "tests"},
           "and": "en",
           "all_good": "{who}: het model van de course slaagt voor alle drie, en elke versie met een bug faalt voor minstens één ervan.",
           "missed": "{who}. Geen enkele test faalt voor {names}.",
           "right_fails": "{who}. Het model van de course faalt voor {names}: een test die faalt voor het goede model kan het goede model niet onderscheiden van een model met een bug.",
           "title": "Je tests op vijf versies van het model"},
}


def _listed(items: list, lang: str) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" {DEMO_TEXT[lang]['and']} " + items[-1]


def _who(r: dict, lang: str) -> str:
    w = DEMO_TEXT[lang]
    mine = r["yours"]
    if len(mine) in (0, 3):
        return w["who"][len(mine)]
    names = TEST_NAMES[lang]
    others = [n for n in EXERCISES if n not in mine]
    return w["who"]["partly"].format(yours=_listed([names[n] for n in mine], lang),
                                     course=_listed([names[n] for n in others], lang),
                                     yours_word=w["test"][len(mine)], course_word=w["test"][len(others)])


def verdicts(r: dict, lang: str) -> str:
    """The sentence under the table: whose tests ran, and whether every version with a bug failed one."""
    lang = _lang(lang)
    w = DEMO_TEXT[lang]
    v = r["verdicts"]
    who = _who(r, lang)
    names = TEST_NAMES[lang]
    failing = [names[n] for n in EXERCISES if v[n]["course"] is not True]
    if failing:
        return w["right_fails"].format(who=who, names=_listed(failing, lang))
    missed = [VERSION_NAMES[lang][k] for k, _ in VERSIONS[1:] if all(v[n][k] is True for n in EXERCISES)]
    if missed:
        return w["missed"].format(who=who, names=_listed(missed, lang))
    return w["all_good"].format(who=who)


def matrix_picture(r: dict, lang: str) -> str:
    """Your three tests on the five versions: in the color of mistakes where a test fails for the course's model,
    and along the row of a version with a bug that no test catches; in ink everywhere else."""
    lang = _lang(lang)
    w = DEMO_TEXT[lang]
    head = [w["head"]] + [TEST_NAMES[lang][n] for n in EXERCISES]
    cells = '<th style="text-align:left;font-weight:400;color:var(--ea-ink-2);padding:0 1rem 0.2rem 0">{}</th>'
    rows = ["<tr>" + "".join(cells.format(html.escape(h)) for h in head) + "</tr>"]
    for key, _ in VERSIONS:
        missed = key != "course" and all(r["verdicts"][n][key] is True for n in EXERCISES)
        row = f'<td style="padding:0.1rem 1rem 0.1rem 0">{html.escape(VERSION_NAMES[lang][key])}</td>'
        for n in EXERCISES:
            verdict = r["verdicts"][n][key]
            good = verdict is True if key == "course" else not missed
            row += f'<td style="padding:0.1rem 1rem 0.1rem 0">{_mark(verdict, good, lang)}</td>'
        rows.append(f"<tr>{row}</tr>")
    table = f'<table style="font-size:0.85rem;border-collapse:collapse">{"".join(rows)}</table>'
    return yours.figure(w["title"], table)


def version_frame(r: dict, v: float, lang: str) -> tuple:
    """For one version: what it expects on 14 July, as it was, with rain, and a year later in the table, and how
    your three tests did on it."""
    lang = _lang(lang)
    w = DEMO_TEXT[lang]
    key = VERSIONS[min(max(int(round(v)), 1), len(VERSIONS)) - 1][0]
    lines = r["lines"][key]
    hours = list(range(24))
    values = lines["as"] + lines["rain"] + lines["later"]
    top = max(600.0, max(values) * 1.08)
    low = min(-50.0, min(values) * 1.08)
    name = VERSION_NAMES[lang][key]
    p = chart.Panel((0, 23), (low, top), title=name[0].upper() + name[1:], xlabel=w["x"], ylabel=w["y"],
                    xticks=[0, 4, 8, 12, 16, 20])
    p.line([1, 4], [50, 50], "data", 1.4)
    p.line(hours, lines["later"], "loss", 2.4)
    p.line(hours, lines["rain"], "pred", 1.4, dashed=True)
    p.line(hours, lines["as"], "pred", 2.4)
    said = []
    for n in EXERCISES:
        verdict = r["verdicts"][n][key]
        word = {True: {"en": "passes", "nl": "slaagt"}, False: {"en": "**fails**", "nl": "**faalt**"},
                None: {"en": "too slow", "nl": "te traag"}}[verdict][lang]
        said.append(f"{TEST_NAMES[lang][n]}: {word}")
    sentence = f"**{name[0].upper() + name[1:]}.** " + "; ".join(said) + "."
    return _chart([p], w["alt"], lang), sentence


def demo_key(lang: str) -> str:
    w = DEMO_TEXT[_lang(lang)]
    return live.key_html([("pred", False, w["as"]), ("pred", True, w["rain"]), ("loss", False, w["later"]),
                          ("data", False, w["bound"])])


# ======== the cases the tests feed the checks ============================================================================

def _plus(start: str, *lines: str) -> str:
    return start + "".join(f"    {line}\n" for line in lines)


CASES = [
    ("quiet-night", _N_START, "empty", None),
    ("quiet-night", "", "empty", None),
    ("quiet-night", _N, "correct", None),
    ("quiet-night", _plus(_N_START, "assert (pred <= 100).all()"), "correct", None),
    ("quiet-night", _plus(_N_START, "assert pred.max() < 50"), "correct", None),
    ("quiet-night", _plus(_N_START, "assert all(p < 60 for p in pred)"), "correct", None),
    ("quiet-night", _plus(_N_START, "assert (pred < 50).all()  # it’s a quiet hour"), "correct", None),
    ("quiet-night", _plus(_N_START, "assert (pred < 50).any()"), "wrong", "misses-utc"),
    ("quiet-night", _plus(_N_START, "assert (pred < 1000).all()"), "wrong", "misses-utc"),
    ("quiet-night", _plus(_N_START, "assert pred.mean() < expect(work[work[\"hr\"] == 8]).mean()"), "wrong",
     "misses-celsius"),
    ("quiet-night", _plus(_N_START, "assert (pred < 40).all()"), "wrong", "tight"),
    ("quiet-night", _plus(_N_START, "assert (pred >= 0).all()"), "wrong", "from-below"),
    ("quiet-night", _plus(_N_START, "assert (pred > 50).all()"), "wrong", "fails-right"),
    ("quiet-night", _plus(_N_START, "assert pred < 50"), "error", "ambiguous"),
    ("quiet-night", _plus(_N_START, "assert (pred = 50).all()"), "error", "one-equals"),
    ("quiet-night", _plus(_N_START, "assert (pred < 50).all() == “True”"), "error", "curly-quote"),
    ("quiet-night", _plus(_N_START, "assert (pred < 50).every()"), "error", None),
    ("quiet-night", _N_START + "assert (expect(HOURS) < 50).all()\n", "wrong", "outside"),
    ("quiet-night", _N.replace("def test_quiet_night(expect):", "def test_quiet_night():"), "wrong", "argument"),
    ("quiet-night", _N.replace("def test_quiet_night(expect):", "def quiet_night(expect):"), "wrong", "no-test"),
    ("row-number", _R_START, "empty", None),
    ("row-number", "", "empty", None),
    ("row-number", _R, "correct", None),
    ("row-number", _R_START.replace("    moved = HOURS\n", "    moved = read_hours(2011)\n"), "correct", None),
    ("row-number", _R_START.replace("    moved = HOURS\n", "    moved = HOURS.copy(deep=True)\n"), "correct", None),
    ("row-number", _R.replace("assert (expect(moved) == expect(HOURS)).all()",
                              "assert (expect(moved) - expect(HOURS)).abs().max() < 0.01"), "correct", None),
    ("row-number", _R_START.replace("+ 8645", "* 2"), "wrong", "same-table"),
    ("row-number", _R.replace("== expect(HOURS)", "== expect(moved)"), "wrong", "misses-row-number"),
    ("row-number", _R.replace("+ 8645", "+ 0"), "wrong", "misses-row-number"),
    ("row-number", _R.replace("==", "!="), "wrong", "fails-right"),
    ("row-number", _R_START.replace("    moved = HOURS\n", "    moved = HOURS.copy\n"), "error", "method"),
    ("row-number", _R.replace("assert (expect(moved) == expect(HOURS)).all()", "assert expect(moved) == expect(HOURS)"),
     "error", "ambiguous"),
    ("row-number", _R.replace("assert (expect(moved) == expect(HOURS)).all()",
                              "assert (expect(moved) = expect(HOURS)).all()"), "error", "one-equals"),
    ("row-number", _R.replace("moved[\"instant\"]", "moved[“instant”]", 1), "error", "curly-quote"),
    ("row-number", _R.replace("moved[\"instant\"] + 8645", "moved[\"row\"] + 8645"), "error", None),
    ("row-number", _R + "assert (expect(HOURS) == expect(HOURS)).all()\n", "wrong", "outside"),
    ("row-number", _R.replace("(expect):", "(model):"), "wrong", "argument"),
    ("row-number", _R.replace("def test_row_number", "def row_number"), "wrong", "no-test"),
    ("rain", _D_START, "empty", None),
    ("rain", "", "empty", None),
    ("rain", _D, "correct", None),
    ("rain", _plus(_D_START, "assert expect(rain).sum() < expect(clear).sum()"), "correct", None),
    ("rain", _plus(_D_START, "assert expect(rain).mean() <= expect(clear).mean()"), "correct", None),
    ("rain", _plus(_D_START, "assert expect(clear).mean() - expect(rain).mean() > 10"), "correct", None),
    ("rain", _plus(_D_START, "assert expect(clear).mean() > 0"), "wrong", "misses-backwards"),
    ("rain", _plus(_D_START, "assert expect(rain).mean() > expect(clear).mean()"), "wrong", "fails-right"),
    ("rain", _plus(_D_START, "assert expect(rain).mean() < expect(rain).mean()"), "wrong", "fails-right"),
    ("rain", _plus(_D_START, "assert (expect(rain) < expect(clear)).all()"), "wrong", "per-hour"),
    ("rain", _plus(_D_START, "assert expect(rain) < expect(clear)"), "error", "ambiguous"),
    ("rain", _plus(_D_START, "assert expect(rain).mean() = expect(clear).mean()"), "error", "one-equals"),
    ("rain", _plus(_D_START, "assert expect(rain).mean() < expect(clear).mean()  # “less”"), "correct", None),
    ("rain", _D_START.replace("rain[\"weathersit\"] = 3", "rain[“weathersit”] = 3") + "    assert True\n", "error",
     "curly-quote"),
    ("rain", _plus(_D_START, "assert expect(rain).avg() < expect(clear).mean()"), "error", None),
    ("rain", _D_START + "assert expect(HOURS).mean() > 0\n", "wrong", "outside"),
    ("rain", _D.replace("def test_rain(expect):", "def test_rain(expect, hours):"), "wrong", "argument"),
    ("rain", _D.replace("def test_rain", "def rain_test"), "wrong", "no-test"),
    # found in review: a bound on the mean, a method without (), a fixed number, the wrong way round, pytest.fail
    ("quiet-night", _plus(_N_START, "assert pred.mean() < 50"), "wrong", "mean"),
    ("quiet-night", _plus(_N_START, "assert (pred < 50).all"), "wrong", "no-call"),
    ("quiet-night", _plus(_N_START, "assert (pred >= 0.0).all()"), "wrong", "from-below"),
    ("quiet-night", _plus(_N_START, "assert (pred =< 50).all()"), "error", None),
    ("row-number", _R.replace(".all()", ".all"), "wrong", "no-call"),
    ("row-number", _R_START.replace("expect(HOURS)).all()", "expect(read_hours(2011))).all()"), "correct", None),
    ("rain", _plus(_D_START, "assert expect(rain).mean() < 130"), "wrong", "fails-busier"),
    ("rain", _plus(_D_START, "assert expect(rain).mean < expect(clear).mean"), "wrong", "no-call"),
    ("rain", _plus(_D_START, "assert (expect(rain) > expect(clear)).all()"), "wrong", "fails-right"),
    ("rain", "import pytest\n" + _plus(_D_START, "pytest.fail(\"not written yet\")"), "wrong", "fails-right"),
    ("rain", _plus(_D_START, "assert (expect(clear) - expect(rain)).mean() > 0  # “fewer” with rain"), "correct", None),
]
