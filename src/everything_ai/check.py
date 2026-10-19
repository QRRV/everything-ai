# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""How exercises are checked (lesson template, chapter 4).

A check looks at the outcome, never at the form. It stops at the first thing that is not
right, so the learner hears one thing at a time, and every message says whether it is
right, why, and what to do now. The right answer and every known mistake are code: the
check computes them on the same input as the learner's, never from numbers typed in.

States: empty (nothing to check yet), error (error in your code), wrong (not yet),
almost, correct, slow (too slow) and broken (a mistake in the check itself).
"""
from __future__ import annotations

import ast
import contextlib
import copy
import inspect
import io
import math
import re
import signal
import sys
import textwrap
import traceback
import types
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

MISSING = object()            # what a check cell passes when the exercise's name does not exist
TIME_LIMIT = 2.0              # seconds the learner's code gets per test case

Text = dict  # {"en": "...", "nl": "..."}; may contain {placeholders} filled in per check


@dataclass
class Result:
    state: str                 # empty | error | wrong | almost | correct | slow | broken
    text: Text                 # the message in both languages
    mistake: str | None = None  # the known mistake that was recognized, for tests and statistics
    extra: str = ""             # HTML shown under the message, such as the learner's own chart


@dataclass
class Mistake:
    """A known mistake: what it looks like, and what to tell the learner.

    `wrong` is a wrong version of the solution (code text for a code box, a function for a
    function exercise); the check runs it on the same input and compares outcomes.
    `matches` is a test on the learner's outcome or text, for mistakes that are not one version.
    `syntax` is a regular expression on the learner's text, for a mistake Python cannot even read.
    `detail` works out extra {placeholders} for the message from the learner's outcome and text,
    such as the first line that is out of place.
    `error` is a test on the error the code stopped with, for a mistake that is known by its error.
    """
    key: str
    en: str
    nl: str
    wrong: Any = None
    matches: Callable[..., bool] | None = None
    syntax: str | None = None
    state: str = "wrong"
    detail: Callable[..., dict] | None = None
    error: Callable[[BaseException], bool] | None = None   # a test on the error the code stopped with; {call} in a function exercise

    @property
    def text(self) -> Text:
        return {"en": self.en, "nl": self.nl}


class Timeout(Exception):
    pass


def with_time_limit(fn, seconds: float = TIME_LIMIT):
    """Run fn, stopping it after `seconds`. Works where the lesson runs in the main thread
    (marimo edit, the tests); elsewhere it runs without a limit. Meanwhile PyTorch, if it is loaded, calculates with
    one thread: the small models of a check are as fast that way, and on a busy computer several threads that wait
    for each other can make a check ten times slower than its time limit allows."""
    def _raise(*_):
        raise Timeout()
    torch = sys.modules.get("torch")
    threads = torch.get_num_threads() if torch is not None else 1
    if threads > 1:
        torch.set_num_threads(1)
    try:
        try:
            old = signal.signal(signal.SIGALRM, _raise)
        except (ValueError, AttributeError):   # not the main thread, or no SIGALRM
            return fn()
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            return fn()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
    finally:
        if threads > 1:
            torch.set_num_threads(threads)


# --- running code from a code box ----------------------------------------------------

@dataclass
class Run:
    value: Any = None          # the value of the last line, if that line is an expression
    has_value: bool = False
    stdout: str = ""
    error: BaseException | None = None
    line: int | None = None    # the line of the learner's code where the error happened
    timeout: bool = False
    names: dict = field(default_factory=dict)   # the names the code made, such as a function it defined


def run_code(text: str, names: dict | None = None, keep_figures: bool = False) -> Run:
    """Run the learner's code the way a notebook cell does: everything, and keep the value of the last line.
    Charts the code opens with matplotlib are closed afterwards, unless `keep_figures` (a chart check reads them):
    otherwise the next chart in the lesson could be drawn on top of one."""
    from everything_ai import projects   # the learner's own modules, read fresh each time
    projects.on_path()
    projects.forget()
    projects.use_course_versions()   # where a module this lesson builds on is missing or broken (template 1.3)
    run = Run()
    # Python's own hints while it reads the code, such as "perhaps you missed a comma?", stay out of the lesson:
    # they are in English, and the check says it in plain words itself.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            tree = ast.parse(text, mode="exec")
        except SyntaxError as e:
            run.error, run.line = e, e.lineno
            return run
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = compile(ast.Expression(tree.body.pop().value), "<your code>", "eval")
        code = compile(tree, "<your code>", "exec")
    ns = {"__name__": "__learner__", **(names or {})}
    out = io.StringIO()

    def _go():
        with contextlib.redirect_stdout(out), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # NumPy's "divide by zero": the check explains, not Python
            warnings.simplefilter("ignore", UserWarning)      # PyTorch's warnings in English: the check explains them too
            exec(code, ns)
            if last is not None:
                run.value = eval(last, ns)
                run.has_value = True

    plt = sys.modules.get("matplotlib.pyplot")
    before = set(plt.get_fignums()) if plt is not None and not keep_figures else None
    try:
        with_time_limit(_go)
    except Timeout:
        run.timeout = True
    except Exception as e:  # the learner's mistake, explained in plain words by the exercise
        run.error = e
        frames = [f for f in _tb_frames(e) if f[0] == "<your code>"]
        run.line = frames[-1][1] if frames else (e.lineno if isinstance(e, SyntaxError) else None)
    run.stdout = out.getvalue()
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is not None and not keep_figures:
        for number in set(plt.get_fignums()) - (before or set()):
            plt.close(number)
    run.names = ns
    return run


def _tb_frames(e: BaseException):
    tb = e.__traceback__
    while tb is not None:
        yield tb.tb_frame.f_code.co_filename, tb.tb_lineno
        tb = tb.tb_next


def is_empty_text(text: str, start: str) -> bool:
    """Nothing filled in yet: the box still holds the skeleton, or only `...`."""
    stripped = "\n".join(l for l in text.strip().splitlines() if not l.strip().startswith("#")).strip()
    return text.strip() == start.strip() or stripped in ("", "...")


def is_skeleton(fn) -> bool:
    """True if a function's body is still only `...` (comments and a docstring allowed)."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (OSError, TypeError, SyntaxError):
        return False
    f = tree.body[0] if tree.body else None
    if not isinstance(f, ast.FunctionDef):
        return False
    body = [s for s in f.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                                      and isinstance(s.value.value, str))]
    return all(isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and s.value.value is Ellipsis
               for s in body)


def _numpy():
    """NumPy, if something has imported it already: a check never needs it otherwise."""
    return sys.modules.get("numpy")


def _torch():
    """PyTorch, if something has imported it already: a check never needs it otherwise."""
    return sys.modules.get("torch")


def same_value(a, b, rel: float = 1e-6) -> bool:
    """The same outcome, whatever the form: int or float, and numbers within a small margin.
    Text must match exactly; the values must be the same kind (the number 10 is not the text "10").
    A NumPy array is the same as another array, or a list, with the same shape and the same values, int or
    float (template 4.3); a lesson about the form itself compares with `same_array` instead. A PyTorch tensor is
    the same as another tensor, an array or a list with the same shape and values, and a tensor with no axes as
    the number it holds (see `_same_tensor`)."""
    pd = sys.modules.get("pandas")
    if pd is not None and any(isinstance(v, (pd.DataFrame, pd.Series)) for v in (a, b)):
        return _same_table(pd, a, b, rel)
    torch = _torch()
    if torch is not None and any(isinstance(v, torch.Tensor) for v in (a, b)):
        return _same_tensor(torch, a, b, rel)
    np = _numpy()
    if np is not None:
        a = a.item() if isinstance(a, np.generic) or isinstance(a, np.ndarray) and a.ndim == 0 else a
        b = b.item() if isinstance(b, np.generic) or isinstance(b, np.ndarray) and b.ndim == 0 else b
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return _same_array(np, a, b, rel, strict=False)
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=rel, abs_tol=1e-9) or math.isnan(a) and math.isnan(b)
    if isinstance(a, list) and isinstance(b, list) or type(a) is tuple and type(b) is tuple:
        # a list, a tuple, or the values a generator handed out: value by value, so arrays inside work too
        return len(a) == len(b) and all(same_value(x, y, rel) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def _same_tensor(torch, a, b, rel: float) -> bool:
    """PyTorch: a tensor is the same as another tensor, a NumPy array or a list with the same shape and the same
    values, int or float, on any device; a tensor with no axes is the same as the number it holds (template 4.3).
    A lesson about the form itself, such as `.item()`, gives its exercise its own `compare`. float32 keeps about 7
    digits, so a different but right order of operations may differ in the last of them."""
    np = _numpy()

    def tensor(v):
        if isinstance(v, torch.Tensor):
            return v.detach().cpu()
        if np is not None and isinstance(v, (np.ndarray, np.generic)):
            v = v.tolist() if v.dtype.kind in "biuf" else None
        if isinstance(v, (int, float, list, tuple)) and not isinstance(v, bool):
            try:
                return torch.tensor(v, dtype=torch.float64)
            except (TypeError, ValueError, RuntimeError):   # text, or rows of different lengths
                return None
        return None

    a, b = tensor(a), tensor(b)
    if a is None or b is None or a.shape != b.shape:
        return False
    if torch.float64 not in (a.dtype, b.dtype) or a.dtype != b.dtype:
        rel = max(rel, 1e-5)
    a, b = a.detach().cpu().double(), b.detach().cpu().double()
    return bool(torch.allclose(a, b, rtol=rel, atol=1e-9, equal_nan=True))


def _same_table(pd, a, b, rel: float) -> bool:
    """pandas: a DataFrame is the same as another with the same column names and the same values, row by row;
    a Series is the same as another Series, a list, a tuple or an array with the same values. The row labels (the
    index) do not count: a selection keeps the old ones, and either way the outcome is the same. Missing values
    (NaN, None, pd.NA) count as the same as each other."""
    def values(v):
        v = v.tolist() if hasattr(v, "tolist") and not isinstance(v, (str, list)) else v
        v = list(v) if isinstance(v, tuple) else v
        return [None if not isinstance(x, (list, tuple)) and pd.isna(x) else x for x in v] if isinstance(v, list) else v

    if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
        if not (isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame)) or a.shape != b.shape:
            return False
        return list(a.columns) == list(b.columns) and all(
            same_value(values(a.iloc[:, j]), values(b.iloc[:, j]), rel) for j in range(a.shape[1]))
    av, bv = values(a), values(b)
    return isinstance(av, list) and isinstance(bv, list) and same_value(av, bv, rel)


def same_array(a, b, rel: float = 1e-6) -> bool:
    """For a lesson about arrays themselves: only a NumPy array counts, with the same shape and values."""
    np = _numpy()
    return np is not None and _same_array(np, a, b, rel, strict=True)


def _same_array(np, a, b, rel: float, strict: bool) -> bool:
    if not strict:   # a list or tuple of numbers counts as the array it would make
        try:
            a = a if isinstance(a, np.ndarray) else np.asarray(a) if isinstance(a, (list, tuple)) else a
            b = b if isinstance(b, np.ndarray) else np.asarray(b) if isinstance(b, (list, tuple)) else b
        except ValueError:   # rows of different lengths make no array
            return False
    if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)) or a.shape != b.shape:
        return False
    number = "iuf"
    if a.dtype.kind in number and b.dtype.kind in number:
        return bool(np.allclose(a, b, rtol=rel, atol=1e-9, equal_nan=True))
    return a.dtype.kind == b.dtype.kind and bool(np.array_equal(a, b))


# --- exercises ------------------------------------------------------------------------

@dataclass
class Exercise:
    """The base of every exercise. `name` is what the learner makes (or the code box's key)."""
    name: str
    required: bool = True
    mistakes: list[Mistake] = field(default_factory=list)
    texts: dict[str, Text] = field(default_factory=dict)   # empty, correct, wrong, almost, error...

    def message(self, key: str, **values) -> Text:
        t = self.texts[key]
        return {lang: s.format(**_for_lang(values, lang)) for lang, s in t.items()}

    def check(self, got) -> Result:  # pragma: no cover - every kind implements it
        raise NotImplementedError


# Plain-language explanations of Python errors, for lessons before `py-errors` teaches tracebacks.
# They use no error names and only words a first-week learner knows.
PLAIN_ERRORS: dict[str, Text] = {
    "SyntaxError": {"en": "Python cannot read what is in the box.",
                    "nl": "Python kan niet lezen wat er in het vak staat."},
    "expected-indent": {"en": "After a line that ends with `:`, the next line must be indented: put spaces before it.",
                        "nl": "Na een regel die eindigt op `:` moet de volgende regel inspringen: zet er spaties voor."},
    "indent-mismatch": {"en": "A line has another number of spaces in front than the lines around it. Lines that belong together need the same number of spaces.",
                        "nl": "Een regel heeft een ander aantal spaties ervoor dan de regels eromheen. Regels die bij elkaar horen, hebben evenveel spaties nodig."},
    "unexpected-indent": {"en": "A line starts with a space, and Python does not expect that here. Start each line at the very left.",
                          "nl": "Een regel begint met een spatie, en dat verwacht Python hier niet. Begin elke regel helemaal links."},
    "NameError": {"en": "Python does not know the word `{word}`. Is it a typo, or are quotes missing around text?",
                  "nl": "Python kent het woord `{word}` niet. Zit er een typefout in, of missen er aanhalingstekens om een tekst?"},
    "TypeError": {"en": "Python cannot combine these two things, for example a number and a text.",
                  "nl": "Python kan deze twee dingen niet combineren, bijvoorbeeld een getal en een tekst."},
    "ZeroDivisionError": {"en": "You divide by zero, and nobody can do that, not even Python.",
                          "nl": "Je deelt door nul, en dat kan niemand, ook Python niet."},
    "IndexError": {"en": "You ask for a place that is not there: the places go from 0 up to the length minus 1.",
                   "nl": "Je vraagt om een plek die er niet is: de plekken lopen van 0 tot de lengte min 1."},
    "KeyError": {"en": "The dictionary has no key `{word}`. Check the spelling, or check first with `in` whether the key is there.",
                 "nl": "De dictionary heeft geen key `{word}`. Kijk de spelling na, of kijk eerst met `in` of de key er is."},
    "RecursionError": {"en": "The function keeps calling itself and never stops: Python gives up after about a thousand calls inside each other. Does every call get closer to a case that gives back an answer without calling itself again?",
                       "nl": "De functie blijft zichzelf aanroepen en stopt nooit: Python geeft het op na zo'n duizend aanroepen diep. Komt elke aanroep dichter bij een geval dat een antwoord teruggeeft zonder zichzelf opnieuw aan te roepen?"},
    "FileNotFoundError": {"en": "Python cannot find the file `{word}`. Check the name, and the folders in the path: a path starts at the folder Python runs in, and in the course that is the course folder, `everything-ai`.",
                          "nl": "Python kan het bestand `{word}` niet vinden. Kijk de naam na, en de mappen in het pad: een pad begint bij de map waarin Python draait, en in de course is dat de map van de course, `everything-ai`."},
    "ModuleNotFoundError": {"en": "Python cannot find a module called `{word}`. Check the spelling of the name after `import`, small letters included.",
                            "nl": "Python kan geen module vinden die `{word}` heet. Kijk de naam achter `import` goed na, ook de kleine letters."},
    "ImportError": {"en": "There is nothing called `{word}` in that module. Check the spelling, small letters included.",
                    "nl": "Die module heeft niets dat `{word}` heet. Kijk de spelling na, ook de kleine letters."},
    "closed-file": {"en": "The file is used after the indented lines under `with`, and by then it is closed again. Put the lines that read the file under `with`, indented.",
                    "nl": "Het bestand wordt gebruikt na de ingesprongen regels onder `with`, en dan is het al gesloten. Zet de regels die het bestand lezen onder `with`, ingesprongen."},
    "AttributeError": {"en": "There is nothing called `{word}` after that dot. Check the spelling, small letters included.",
                       "nl": "Na die punt bestaat niets dat `{word}` heet. Kijk de spelling na, ook de kleine letters."},
    "none": {"en": "Python gets `None` here, its word for nothing, and cannot work with it: something before this gave nothing back.",
             "nl": "Python krijgt hier `None`, zijn woord voor niets, en kan daar niets mee: iets hiervoor gaf niets terug."},
    "compare": {"en": "Python cannot compare these two things with `<` or `>`, for example a text and a number.",
                "nl": "Python kan deze twee dingen niet vergelijken met `<` of `>`, bijvoorbeeld een tekst en een getal."},
    "unbound": {"en": "`{word}` has no value yet on this line. Inside a function, give it its starting value before you use it, for example before the loop.",
                "nl": "`{word}` heeft op deze regel nog geen waarde. Geef hem in een functie zijn beginwaarde voordat je hem gebruikt, bijvoorbeeld vóór de loop."},
    "other": {"en": "Python stops with an error.",
              "nl": "Python stopt met een fout."},
}

# The same, when Python can say which line it is about: the message names that line.
PLAIN_ERRORS_AT: dict[str, Text] = {
    "SyntaxError": {"en": "Python cannot read the line `{where}`. Compare it sign by sign with the code in the lesson: one missing or extra sign is enough.",
                    "nl": "Python kan de regel `{where}` niet lezen. Vergelijk hem teken voor teken met de code in de les: één teken te veel of te weinig is genoeg."},
    "expected-indent": {"en": "Under the line `{where}` comes at least one indented line, because it ends with `:`. In the box, Enter after a colon indents the next line for you.",
                        "nl": "Onder de regel `{where}` hoort minstens één ingesprongen regel, omdat hij eindigt op `:`. In het vak laat Enter na een dubbele punt de volgende regel vanzelf inspringen."},
    "indent-mismatch": {"en": "The line `{where}` has another number of spaces in front than the lines around it. Lines that belong together need the same number of spaces.",
                        "nl": "De regel `{where}` heeft een ander aantal spaties ervoor dan de regels eromheen. Regels die bij elkaar horen, hebben evenveel spaties nodig."},
    "unexpected-indent": {"en": "The line `{where}` starts with a space, and Python does not expect that here. Remove the spaces before it, so it starts at the very left.",
                          "nl": "De regel `{where}` begint met een spatie, en dat verwacht Python hier niet. Haal de spaties ervoor weg, zodat hij helemaal links begint."},
}
_NONE_IN_FUNCTION: Text = {
    "en": "Does every way through your function end with `return` and a value, not with `print`?",
    "nl": "Eindigt elke weg door je functie met `return` en een waarde, niet met `print`?"}
_AT_LINE: Text = {"en": "It happens on the line `{where}`.", "nl": "Het gebeurt op de regel `{where}`."}
# A name that only gets its value on a later line: the order is the mistake, not the name.
_TOO_EARLY: Text = {
    "en": "On the line `{where}`, Python does not know `{word}` yet: a name exists only from the line that gives it a value. Move the line with `{word} =` up.",
    "nl": "Op de regel `{where}` kent Python `{word}` nog niet: een naam bestaat pas vanaf de regel die hem een waarde geeft. Zet de regel met `{word} =` hoger."}


_PRINT_LAST = {
    "en": "Your last line is a `print`. It shows a value, but gives nothing back to the check: its value is `None`. Make the last line just `{what}`, without `print`.",
    "nl": "Je laatste regel is een `print`. Die laat een waarde zien, maar geeft de check niets terug: zijn waarde is `None`. Maak de laatste regel gewoon `{what}`, zonder `print`.",
}


def uses_loop(text: str) -> bool:
    """The code has a loop of its own: `for`, `while` or a comprehension. Comments and strings do not count."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    return any(isinstance(n, (ast.For, ast.While, ast.comprehension)) for n in ast.walk(tree))


def _plain_kind(error: BaseException) -> str:
    """Which plain explanation fits an error. A TypeError only gets "cannot combine" when it is about combining."""
    kind, msg = type(error).__name__, str(error)
    if isinstance(error, IndentationError) and "unexpected indent" in msg:
        return "unexpected-indent"
    if isinstance(error, IndentationError) and "expected an indented block" in msg:
        return "expected-indent"
    if isinstance(error, TabError) or isinstance(error, IndentationError) and "does not match" in msg:
        return "indent-mismatch"
    if isinstance(error, SyntaxError):
        return "SyntaxError"
    if kind in ("TypeError", "AttributeError") and "NoneType" in msg:
        return "none"   # also `None.data`: a method gave nothing back, not a misspelt name
    if kind == "TypeError" and "not supported between" in msg:
        return "compare"
    if kind == "TypeError" and not ("operand" in msg or "concatenate" in msg):
        return "other"
    if kind == "UnboundLocalError":
        return "unbound"
    if kind == "ValueError" and "closed file" in msg:
        return "closed-file"
    return kind if kind in PLAIN_ERRORS else "other"


@dataclass
class CodeBox(Exercise):
    """The learner types code in a box in the lesson; the check runs it itself.

    This is the exercise form until `py-errors`: every error is told in plain words, without
    a traceback. `outcome` says what is compared: the value of the last line, or what is printed.
    """
    start: str = ""                    # what is in the box at the start
    solution: str = ""                 # the course's solution (shown as the last hint, used by the tests)
    outcome: str = "value"             # "value" or "stdout"
    expect: Any = MISSING              # the right outcome; computed from `solution` if not given
    compare: Callable[[Any, Any], bool] = same_value
    # Other values for names the box starts with, such as [{"temperature": 5}]: the check also runs the
    # learner's code with the line `temperature = ...` changed, and compares with the solution changed the same way.
    # A name from `names` gets the other value directly.
    others: list[dict] = field(default_factory=list)
    # Names from the lesson's own cells that the code in the box may use, as the lesson makes them.
    # The check cell passes their current values (`inputs`): the learner may change them.
    names: dict = field(default_factory=dict)
    # Mistakes in what the learner typed that count even when the outcome is right, such as making a
    # name again that a cell of the lesson already makes. Each has `matches(None, text)`.
    first: list[Mistake] = field(default_factory=list)
    # Lines of the start that must stay as they are, such as `temperature = -3`: the check tries other values itself.
    keep: list[str] = field(default_factory=list)
    # When the lesson is about the form: mistakes in how the code is written, checked only once the outcome
    # is right in every case, such as a function that should call itself. Each has `matches(None, text)`.
    form: list[Mistake] = field(default_factory=list)
    # (package, module): once the code is right, it is saved in the learner's project, as
    # work/projects/<package>/<module>.py, so later code can import it: its imports, functions, classes and
    # names, not the lines that try it out. Texts may use {saved}; "kept" is said when the learner changed
    # that file by hand, so the check leaves it as it is.
    save_as: tuple[str, str] | None = None
    save_whole: bool = False   # save every line: a script to run from the terminal, not a module to import
    # With digits, the floats a message shows are rounded to that many decimals, without float noise or e-notation,
    # such as a float32 loss turned into a Python number: 0.3333, not 0.3333333432674408.
    digits: int | None = None

    def _correct(self, text: str, **values) -> "Result":
        """Right. With `save_as`, the code is saved in the learner's project first; when they changed that file
        themselves, it is kept as it is, and the text "kept" (if there is one) says so."""
        if not self.save_as:
            return Result("correct", self.message("correct", **values))
        from everything_ai import projects
        path, kept = projects.save(*self.save_as, text, whole=self.save_whole)
        key = "kept" if kept and "kept" in self.texts else "correct"
        return Result("correct", self.message(key, saved=projects.shown(path), **values))

    def _outcome(self, run: Run):
        return run.value if self.outcome == "value" else run.stdout

    def expected(self, names: dict | None = None):
        if self.expect is not MISSING and not names:
            return self.expect
        return self._outcome(run_code(self.solution, names))

    def check(self, text, inputs: dict | None = None) -> Result:
        if text is MISSING or text is None:
            return Result("error", self.message("missing"))
        if is_empty_text(text, self.start):
            cleared = text.strip() != self.start.strip() and "cleared" in self.texts
            return Result("empty", self.message("cleared" if cleared else "empty"))
        names = dict(self.names if inputs is None else inputs)
        typed = {_same_code(line) for line in text.splitlines()}
        for line in self.keep:
            if _same_code(line) not in typed:
                return Result("wrong", self.message("keep", line=line) if "keep" in self.texts else {
                    "en": f"Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other values by itself.",
                    "nl": f"Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere waarden."},
                    mistake="changed-line")
        for m in self.first:
            if m.matches(None, text):
                return Result(m.state, m.text, mistake=m.key)
        run = run_code(text, names)
        if run.timeout:
            return Result("slow", self.message("slow") if "slow" in self.texts else {
                "en": "Your code is still not done after {s:g} seconds. Does something keep going forever?".format(s=TIME_LIMIT),
                "nl": "Je code is na {s:g} seconden nog niet klaar. Gaat er iets eindeloos door?".format(s=TIME_LIMIT)})
        if run.error is not None:
            for m in self.mistakes:
                if m.error is not None and m.error(run.error):
                    return Result("error", m.text, mistake=m.key)
                found = re.search(m.syntax, text, flags=re.MULTILINE) if m.syntax else None
                if found:   # {line} in its message is the line where the pattern was found
                    line = text[text.rfind("\n", 0, found.start()) + 1:].split("\n", 1)[0].strip()
                    return Result("error", _fill(m.text, line=line), mistake=m.key)
            return Result("error", self._plain_error(run, text))
        got = self._outcome(run)
        if self.outcome == "value" and not run.has_value:
            return Result("wrong", self.message("no_value"), mistake="no-value")
        # What a program printed is shown as it appears under the cell, not as Python code.
        shown = got.rstrip("\n") if self.outcome == "stdout" else _show_rounded(got, self.digits)
        if self.compare(got, self.expected(names)):
            for values in self.others:
                other = self._with_other(text, values, names)
                if other is not None:
                    return other
            for m in self.form:
                if m.matches(None, text):
                    return Result(m.state, m.text, mistake=m.key)
            return self._correct(text, got=shown)
        if self.outcome == "stdout" and not got.strip():
            return Result("wrong", self.message("nothing") if "nothing" in self.texts else {
                "en": "Your program shows nothing. Something only appears under the cell with `print`.",
                "nl": "Je programma laat niets zien. Er verschijnt pas iets onder de cel met `print`."}, mistake="nothing")
        last = [line.strip() for line in text.strip().splitlines() if line.strip()][-1:]
        shown_by_print = re.match(r"print\((.+)\)$", last[0]) if last else None
        if self.outcome == "value" and got is None and shown_by_print:   # it shows on screen, but gives nothing back
            return Result("wrong", _fill(_PRINT_LAST, what=shown_by_print.group(1).strip()), mistake="print-last")
        for m in self.mistakes:
            if m.syntax or m.error is not None:
                continue
            hit = (m.wrong is not None and self.compare(got, self._outcome(run_code(m.wrong, names)))
                   or m.matches is not None and m.matches(got, text))
            if hit:
                extra = m.detail(got, text) if m.detail else {}
                return Result(m.state, _fill(m.text, got=shown, **extra), mistake=m.key)
        return Result("wrong", self.message("wrong", got=shown))

    def _with_other(self, text: str, values: dict, names: dict) -> Result | None:
        """Run the learner's code and the solution with other values; None if they agree."""
        def change(code):
            for name, value in values.items():
                code = re.sub(rf"^([ \t]*){re.escape(name)}[ \t]*=(?!=).*$",
                              lambda m: f"{m.group(1)}{name} = {value!r}", code, count=1, flags=re.MULTILINE)
            return code
        other_names = {**names, **{k: v for k, v in values.items() if k in names}}
        run = run_code(change(text), other_names)
        want = self._outcome(run_code(change(self.solution), other_names))
        if run.error is None and not run.timeout and self.compare(self._outcome(run), want):
            return None
        where = ", ".join(f"`{name} = {_as_code(value)}`" for name, value in values.items())
        show = (lambda v: v.rstrip("\n")) if self.outcome == "stdout" else (lambda v: _show_rounded(v, self.digits))
        if run.error is not None or run.timeout:
            return Result("almost", self.message("other_error", where=where) if "other_error" in self.texts else {
                "en": f"Right for this value. But with {where}, your program stops with an error. Does it work for every value?",
                "nl": f"Goed voor deze waarde. Maar met {where} stopt je programma met een fout. Werkt het voor elke waarde?"},
                mistake="other-value")
        got = show(self._outcome(run))
        if self.outcome == "stdout" and not got.strip():
            got = {"en": "(nothing)", "nl": "(niets)"}
        else:
            got = {"en": got, "nl": got}
        if "other" in self.texts:
            return Result("almost", {lang: self.texts["other"][lang].format(where=where, got=got[lang], want=show(want))
                                     for lang in self.texts["other"]}, mistake="other-value")
        got_en, got_nl = got["en"], got["nl"]
        wanted = show(want)
        wanted = f"\n\n```\n{wanted}\n```\n\n" if "\n" in wanted else f" `{wanted}`. "   # an array over several lines
        return Result("almost", {
            "en": f"Right for this value. But with {where}, your program shows:\n\n```\n{got_en}\n```\n\nand it should show{wanted}Does it work for every value?",
            "nl": f"Goed voor deze waarde. Maar met {where} laat je programma dit zien:\n\n```\n{got_nl}\n```\n\nen het zou dit moeten laten zien:{wanted}Werkt het voor elke waarde?"},
            mistake="other-value")

    def _plain_error(self, run: Run, text: str) -> Text:
        kind = _plain_kind(run.error)
        word = ""
        if kind == "FileNotFoundError":
            word = str(getattr(run.error, "filename", "") or "?")
        if kind == "ModuleNotFoundError":
            word = str(getattr(run.error, "name", "") or "?")
        if kind == "ImportError":
            m = re.search(r"cannot import name '([^']+)'", str(run.error))
            word = m.group(1) if m else "?"
        if kind == "AttributeError":
            m = re.search(r"attribute '([^']+)'", str(run.error))
            word = m.group(1) if m else "?"
        if kind in ("NameError", "unbound"):
            m = re.search(r"(?:name|variable) '([^']+)'", str(run.error))
            word = m.group(1) if m else "?"
        lines = text.splitlines()
        where = ""
        at = run.line
        if kind == "expected-indent":   # Python names the line with the colon: "... after 'if' statement on line 2"
            m = re.search(r"on line (\d+)", str(run.error))
            at = int(m.group(1)) if m else at
        if at and 0 < at <= len(lines):
            where = lines[at - 1].strip()
            before = [line for line in lines[:at - 1] if line.strip()]
            if kind == "unexpected-indent" and before and before[-1][:1] in (" ", "\t"):
                kind = "indent-mismatch"   # inside a block: the spaces differ from the line above
        if kind == "KeyError":   # a key looked up with square brackets; anything else gets neutral words
            key = run.error.args[0] if run.error.args else None
            if "[" in where and not isinstance(key, slice):
                word = _as_code(key)
            else:
                kind = "other"
        if where and kind in PLAIN_ERRORS_AT:
            return _fill(PLAIN_ERRORS_AT[kind], where=where)
        if where and kind == "NameError" and any(re.match(rf"\s*{re.escape(word)}\s*=(?!=)", later)
                                                 for later in lines[run.line:]):
            return _fill(_TOO_EARLY, where=where, word=word)
        base = _fill(PLAIN_ERRORS[kind], word=word)
        if not where:
            return base
        return {lang: base[lang] + " " + _AT_LINE[lang].format(where=where) for lang in base}


@dataclass
class ClassBox(CodeBox):
    """The learner writes a class in a code box. The check runs the box, then each scenario: a few lines that
    make objects and use them, ending in an expression, whose value is compared with the same lines after the
    course's solution. The first scenario is the lesson's example; the others try more, such as two objects
    that must not share their data. Every error is told in plain words, as in a code box.
    Texts may use {scenario} (the lines), {got} and {want}; a mistake's `wrong` is a wrong version of the class.
    With `digits`, {got} and {want} show their floats rounded to that many decimals, without float noise or e-notation."""
    scenarios: list[str] = field(default_factory=list)

    def check(self, text, inputs=None) -> Result:
        if text is MISSING or text is None:
            return Result("error", self.message("missing"))
        if is_empty_text(text, self.start):
            cleared = not text.strip() and "cleared" in self.texts
            return Result("empty", self.message("cleared" if cleared else "empty"))
        for m in self.first:
            if m.matches(None, text):
                return Result(m.state, m.text, mistake=m.key)
        steps = [""] + self.scenarios   # first the class alone, then each scenario
        for i, scenario in enumerate(steps):
            code = text + "\n\n" + scenario if scenario else text
            run = run_code(code)
            if run.timeout:
                return Result("slow", self.message("slow") if "slow" in self.texts else {
                    "en": "Your code is still not done after {s:g} seconds. Does a loop keep going forever?".format(s=TIME_LIMIT),
                    "nl": "Je code is na {s:g} seconden nog niet klaar. Gaat een loop eindeloos door?".format(s=TIME_LIMIT)})
            if run.error is not None:
                for m in self.mistakes:
                    if m.error is not None and m.error(run.error):
                        return Result("error", m.text, mistake=m.key)
                    found = re.search(m.syntax, text, flags=re.MULTILINE) if m.syntax else None
                    if found:
                        line = text[text.rfind("\n", 0, found.start()) + 1:].split("\n", 1)[0].strip()
                        return Result("error", _fill(m.text, line=line), mistake=m.key)
                known = self._class_error(run, scenario)
                # the line Python names is the learner's, never one of the check's own lines after the class
                inner = run if run.line is None or run.line <= len(text.splitlines()) else Run(error=run.error)
                plain = known or self._plain_error(inner, text)
                if not scenario:
                    return Result("error", plain)
                # the plain words for "other" already say that it stops, also for a KeyError that is not a lookup
                stops = known is not None or not plain["en"].startswith(PLAIN_ERRORS["other"]["en"])
                what = ("class", "class") if re.search(r"^class\s", self.solution, re.MULTILINE) else ("code", "code")
                return Result("error", {"en": f"With these lines after your {what[0]}:\n\n```python\n{scenario}\n```\n\n" + ("Python stops with an error. " if stops else "") + plain["en"],
                                        "nl": f"Met deze regels na je {what[1]}:\n\n```python\n{scenario}\n```\n\n" + ("stopt Python met een fout. " if stops else "") + plain["nl"]})
            if not scenario:
                continue
            want = run_code(self.solution + "\n\n" + scenario).value
            if self.compare(run.value, want):
                continue
            values = dict(scenario=scenario, got=_as_rounded(run.value, self.digits), want=_as_rounded(want, self.digits))
            for m in self.mistakes:
                if m.syntax or m.error is not None:
                    continue
                if m.wrong is not None and self._same_everywhere(text, m.wrong):
                    return Result(m.state, _fill(m.text, **values), mistake=m.key)
                if m.wrong is None and m.matches is not None and m.matches(run.value, text):
                    return Result(m.state, _fill(m.text, **values), mistake=m.key)
            return Result("almost" if i > 1 else "wrong", self.message("almost" if i > 1 else "wrong", **values))
        for m in self.form:
            if m.matches(None, text):
                return Result(m.state, m.text, mistake=m.key)
        return self._correct(text, n=len(self.scenarios))

    def _class_error(self, run: Run, scenario: str) -> Text | None:
        """Errors that are about the class itself, told in its own words: a method that gets another number of
        values than its def line has names for, a method or attribute the objects do not have, no such class."""
        classes = re.findall(r"^class\s+(\w+)", self.solution, re.MULTILINE)
        cls = classes[-1] if classes else ""   # the class the learner writes: the last one, after any parent class
        msg = str(run.error)
        named = re.search(r"'(\w+)' object|(\w+)\(\)", msg)
        if named and (named.group(1) or named.group(2)) in classes:
            cls = named.group(1) or named.group(2)   # the error is about this class
        if isinstance(run.error, TypeError) and cls:
            m = re.search(r"__str__ returned non-string \(type (\w+)\)", msg)
            if m:   # say what it gives back in words the learner knows, not Python's name of its type
                what = {"NoneType": ("nothing, `None`: it has no `return`, or it uses `print`", "niets terug, `None`: er staat geen `return` in, of hij gebruikt `print`"),
                        "int": ("a number", "een getal"), "float": ("a number", "een getal"),
                        "list": ("a list", "een lijst")}.get(m.group(1), ("something else", "iets anders"))
                if m.group(1) == "NoneType":
                    return {"en": f"`__str__` must give back text, a string, and yours gives back {what[0]}. Give back the text with `return f\"...\"`.",
                            "nl": f"`__str__` moet tekst teruggeven, een string, en de jouwe geeft {what[1]}. Geef de tekst terug met `return f\"...\"`."}
                return {"en": f"`__str__` must give back text, a string, and yours gives back {what[0]}. Build the text with an f-string: `return f\"...\"`.",
                        "nl": f"`__str__` moet tekst teruggeven, een string, en de jouwe geeft {what[1]} terug. Maak de tekst met een f-string: `return f\"...\"`."}
            m = re.search(rf"unsupported operand type\(s\) for ([^:]+): '{cls}' and '{cls}'", msg)
            if m and m.group(1).strip() == "+":
                return {"en": f"Python does not know what `+` means for two `{cls}` objects: your class has no `__add__`. Check its name: two underscores on each side.",
                        "nl": f"Python weet niet wat `+` betekent voor twee `{cls}`-objecten: je class heeft geen `__add__`. Kijk de naam na: twee lage streepjes aan elke kant."}
            if f"{cls}() takes no arguments" in msg:
                return {"en": f"`{cls}(...)` gets values, but your class has no `__init__` that takes them. Check the spelling of `__init__`: two underscores on each side.",
                        "nl": f"`{cls}(...)` krijgt waarden, maar je class heeft geen `__init__` die ze aanneemt. Kijk de spelling van `__init__` na: twee lage streepjes aan elke kant."}
            if f"'{cls}' object is not callable" in msg:
                return {"en": f"Python does not know what calling a `{cls}` object means: your class has no `__call__`. Check its name: two underscores on each side.",
                        "nl": f"Python weet niet wat het aanroepen van een `{cls}`-object betekent: je class heeft geen `__call__`. Kijk de naam na: twee lage streepjes aan elke kant."}
        if isinstance(run.error, TypeError) and classes:   # about methods and self: only when the box holds a class
            m = re.search(r"([\w.]+)\(\) (?:takes \d+ positional arguments? but \d+ (?:was|were) given|missing \d+ required positional argument)", msg)
            if m:
                name = m.group(1).split(".")[-1]
                return {"en": f"The method `{name}` gets another number of values than its `def` line has names for. Python passes the object itself as `self`, so the `def` line starts with `self`, and then has one name for each value in the call.",
                        "nl": f"De method `{name}` krijgt een ander aantal waarden dan zijn `def`-regel namen heeft. Python geeft het object zelf mee als `self`, dus begint de `def`-regel met `self`, en heeft hij daarna één naam voor elke waarde in de aanroep."}
        if isinstance(run.error, AttributeError):
            m = re.search(r"'(\w+)' object has no attribute '(\w+)'", msg)
            if m and m.group(1) in classes:
                attr = m.group(2)
                if re.search(rf"\.{re.escape(attr)}\s*\(", scenario):
                    return {"en": f"Your class `{cls}` has no method `{attr}`. Write it indented under `class {cls}:`, as `def {attr}(self, ...):`, and check its spelling.",
                            "nl": f"Je class `{cls}` heeft geen method `{attr}`. Schrijf hem ingesprongen onder `class {cls}:`, als `def {attr}(self, ...):`, en kijk de spelling na."}
                return {"en": f"Your objects have no attribute `{attr}`. Does `__init__` give it its starting value, as in `self.{attr} = ...`? Check the spelling of `__init__` too: two underscores on each side.",
                        "nl": f"Je objecten hebben geen attribute `{attr}`. Geeft `__init__` het zijn beginwaarde, zoals in `self.{attr} = ...`? Kijk ook de spelling van `__init__` na: twee lage streepjes aan elke kant."}
        if isinstance(run.error, NameError) and scenario and getattr(run.error, "name", None) in classes:
            cls = run.error.name
            return {"en": f"The check makes objects with `{cls}(...)`, and your box has no class `{cls}`. Check the name after `class`, capital letter included.",
                    "nl": f"De check maakt objecten met `{cls}(...)`, en je vak heeft geen class `{cls}`. Kijk de naam achter `class` na, ook de hoofdletter."}
        return None

    def _same_everywhere(self, text: str, wrong: str) -> bool:
        """The learner's class and a wrong version give the same in every scenario."""
        for scenario in self.scenarios:
            a, b = run_code(text + "\n\n" + scenario), run_code(wrong + "\n\n" + scenario)
            if (a.error is None) != (b.error is None) or (a.error is None and not self.compare(a.value, b.value)):
                return False
        return True


# The same box for code that is not a class, such as a decorator: the box, then each scenario.
ScenarioBox = ClassBox


_KINDS: dict[str, Text] = {
    "line": {"en": "a line chart", "nl": "een lijngrafiek"},
    "scatter": {"en": "a scatter plot", "nl": "een scatter plot"},
    "bar": {"en": "a bar chart", "nl": "een staafdiagram"},
    "barh": {"en": "a bar chart lying on its side", "nl": "een liggend staafdiagram"},
}
_DRAW = {"line": "plt.plot", "scatter": "plt.scatter", "bar": "plt.bar"}


@dataclass
class PlotBox(CodeBox):
    """The learner draws a chart with matplotlib in a code box. The check runs the box without showing anything,
    reads back what the figure shows (plots.describe: the kind of chart, its x and y values, its title and axis
    labels) and compares that with the course's solution. A mistake's `matches` gets that description as `got`;
    its `error` gets the error the code stopped with. The learner's own chart is shown under the message.
    Texts may use {kind}, {want_kind}, {draw}, {got}, {want}, {got_x} and {want_x}; "nothing", "kind", "values",
    "title", "xlabel" and "ylabel" are the lesson's own words for those cases."""
    # The code the check compares with, when it should differ from the solution the lesson shows: such as the
    # course's own count, where the solution imports the learner's module, so a wrong module shows up.
    reference: str = ""
    # True: every chart of the solution's kind counts together, in any order, such as one plt.scatter per group;
    # each of the solution's points must be there, and more points, such as a mean drawn on top, may be.
    merge: bool = False

    @staticmethod
    def describe_run(code: str, picture: bool = False):
        """Run the code with a fresh figure; give back (run, description, and with `picture` the figure as HTML).
        plt.show() shows nothing here: it keeps the figure it would show and starts a new one, as in a cell, so
        lines after it land on a new chart. Whatever the code does to pyplot or its settings, such as
        `plt.title = "..."`, is undone afterwards."""
        import matplotlib
        import matplotlib.pyplot as plt   # in a lesson, marimo's own backend; elsewhere matplotlib picks one
        from everything_ai import plots
        saved = dict(vars(plt))
        shown = []

        def show(*args, **kwargs):
            shown.append(plt.gcf())
            plt.figure()

        plt.close("all")
        described, html_ = None, ""
        try:
            with matplotlib.rc_context():
                plt.show = show
                run = run_code(code, keep_figures=True)
                if run.error is None and not run.timeout:
                    fig = shown[0] if shown else plt.gcf()
                    described = plots.describe(fig)
                    if shown:   # title or labels set after plt.show() land on the next, empty chart
                        later = plots.describe(plt.gcf())
                        described["after_show"] = not later["charts"] and any(later[k] for k in ("title", "xlabel", "ylabel"))
                    if picture and described["charts"]:
                        html_ = plots.render(fig, alt=plots.alt_text(described))
        finally:
            for name in [n for n in vars(plt) if n not in saved]:
                delattr(plt, name)
            for name, value in saved.items():
                setattr(plt, name, value)
            plt.close("all")
        return run, described, html_

    def check(self, text, inputs=None) -> Result:
        if text is MISSING or text is None:
            return Result("error", self.message("missing"))
        if is_empty_text(text, self.start):
            cleared = not text.strip() and "cleared" in self.texts
            return Result("empty", self.message("cleared" if cleared else "empty"))
        for m in self.first:
            if m.matches(None, text):
                return Result(m.state, m.text, mistake=m.key)
        run, got, picture = self.describe_run(text, picture=True)
        if run.timeout:
            return Result("slow", {"en": "Your code is still not done after {s:g} seconds. Does a loop keep going forever?".format(s=TIME_LIMIT),
                                   "nl": "Je code is na {s:g} seconden nog niet klaar. Gaat een loop eindeloos door?".format(s=TIME_LIMIT)})
        if run.error is not None:
            for m in self.mistakes:
                if m.error is not None and m.error(run.error):
                    return Result("error", m.text, mistake=m.key)
                found = re.search(m.syntax, text, flags=re.MULTILINE) if m.syntax else None
                if found:
                    line = text[text.rfind("\n", 0, found.start()) + 1:].split("\n", 1)[0].strip()
                    return Result("error", _fill(m.text, line=line), mistake=m.key)
            return Result("error", self._plain_error(run, text))
        _, want, _ = self.describe_run(self.reference or self.solution)
        for m in self.mistakes:
            if not m.syntax and m.error is None and m.matches is not None and m.matches(got, text):
                return Result(m.state, _fill(m.text, got=got, want=want), mistake=m.key, extra=picture)
        if not got["charts"]:
            return Result("wrong", self.message("nothing"), mistake="nothing", extra=picture)
        mine, theirs = got["charts"][0], want["charts"][0]
        if self.merge:
            theirs = _merged(want["charts"], want["charts"][0]["kind"])
            mine = _merged(got["charts"], theirs["kind"]) if any(c["kind"] == theirs["kind"] for c in got["charts"]) else mine
        if mine["kind"] != theirs["kind"]:
            return Result("wrong", self.message("kind", kind=_KINDS[mine["kind"]], want_kind=_KINDS[theirs["kind"]],
                                                draw=_DRAW[theirs["kind"]]), mistake="kind", extra=picture)
        same = _covers(mine, theirs, self.compare) if self.merge else (self.compare(mine["x"], theirs["x"]) and self.compare(mine["y"], theirs["y"]))
        if not same:
            return Result("wrong", self.message("values", got=_as_python(mine["y"]), want=_as_python(theirs["y"]),
                                                got_x=_as_python(mine["x"]), want_x=_as_python(theirs["x"])), mistake="values", extra=picture)
        if got.get("after_show"):
            return Result("almost", {"en": "Your title or labels come after `plt.show()`, so they land on a new, empty chart: `plt.show()` finishes the chart. Move `plt.show()` to the last line.",
                                     "nl": "Je titel of labels komen na `plt.show()`, dus komen ze op een nieuwe, lege grafiek: `plt.show()` maakt de grafiek af. Zet `plt.show()` op de laatste regel."},
                          mistake="after-show", extra=picture)
        for part in ("title", "xlabel", "ylabel"):
            if want[part] and not got[part].strip():
                return Result("almost", self.message(part), mistake=f"no-{part}", extra=picture)
        for m in self.form:
            if m.matches(None, text):
                return Result(m.state, m.text, mistake=m.key, extra=picture)
        result = self._correct(text)
        result.extra = picture
        return result


def _merged(charts: list, kind: str) -> dict:
    """All charts of one kind as one, their points in a fixed order: by x as text, then by y (numbers before names,
    so a name up a y-axis next to a number does not stop the sorting)."""
    pairs = sorted(((x, y) for c in charts if c["kind"] == kind for x, y in zip(c["x"], c["y"])),
                   key=lambda p: (str(p[0]), isinstance(p[1], str), str(p[1]) if isinstance(p[1], str) else p[1]))
    return {"kind": kind, "x": [x for x, _ in pairs], "y": [y for _, y in pairs]}


def _covers(mine: dict, theirs: dict, compare) -> bool:
    """Every point of `theirs` is among the points of `mine`, each point used once."""
    left = list(zip(mine["x"], mine["y"]))
    for x, y in zip(theirs["x"], theirs["y"]):
        hit = next((i for i, (a, b) in enumerate(left) if compare([a], [x]) and compare([b], [y])), None)
        if hit is None:
            return False
        left.pop(hit)
    return True


class Handed(list):
    """The values a generator handed out, in order: a list that remembers it came from a generator."""


# A generator that hands out more than this keeps going forever, as far as a check is concerned.
MAX_HANDED = 10_000


def collect_generator(value):
    """A generator as the list of the values it hands out; anything else as it is. Each value is copied the
    moment it is handed out, so a list that the generator changes afterwards is shown as it was then."""
    if not isinstance(value, types.GeneratorType):
        return value
    out = Handed()
    for v in value:
        out.append(copy.deepcopy(v))
        if len(out) > MAX_HANDED:
            raise Timeout()   # it keeps handing out values: told like a loop that never ends
    return out


def handed_out(value) -> Text:
    """What a generator function gave, in words: its values one after the other, or nothing; a value given
    back with `return` instead is named as such."""
    if isinstance(value, Handed):
        if not value:
            return {"en": "nothing", "nl": "niets"}
        parts = [f"`{_as_python(v)}`" for v in value]
        if len(parts) == 1:
            return {"en": parts[0], "nl": parts[0]}
        return {"en": ", ".join(parts[:-1]) + " and " + parts[-1], "nl": ", ".join(parts[:-1]) + " en " + parts[-1]}
    return {"en": f"no values one at a time: it gives back `{_as_python(value)}` with `return`",
            "nl": f"geen waarden één voor één af: hij geeft `{_as_python(value)}` terug met `return`"}


def calls_itself(text: str, name: str) -> bool:
    """The function `name` in this code calls itself somewhere in its body."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == name
                       for c in ast.walk(node))
    return False


def _skeleton_def(text: str, name: str) -> bool:
    """The box still has the function `name` with only `...` in its body."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body = [s for s in node.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                                                and isinstance(s.value.value, str))]
            return all(isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and s.value.value is Ellipsis
                       for s in body)
    return False


@dataclass
class FunctionBox(CodeBox):
    """The learner writes a function in a code box; the check runs the box, then calls the function
    on the example from the lesson (`visible`) and on other cases (`hidden`), each a tuple of arguments,
    and compares with the course's solution, which is code text too. Every error is told in plain
    words, as in a code box. A mistake's `wrong` is a wrong version of the code text."""
    function: str = ""                 # the name of the function the box makes
    visible: list[tuple] = field(default_factory=list)
    hidden: list[tuple] = field(default_factory=list)
    # True: {got} and {want} show values the way Python shows them, `6.0` and not `6`, for a lesson where
    # the difference matters (as in BugFix).
    python_values: bool = False
    # What the check does with what the function gives back, before it compares: `collect_generator` turns a
    # generator into the list of values it hands out. It runs within the time limit, like the call itself.
    collect: Callable[[Any], Any] | None = None
    # How {got} and {want} are shown, such as `handed_out` for a generator; a value or one per language.
    show: Callable[[Any], Any] | None = None

    def _run(self, fn, case):
        with warnings.catch_warnings():   # NumPy's "divide by zero" and the like: the check explains, not Python
            warnings.simplefilter("ignore", RuntimeWarning)
            value = fn(*(_copy(a) for a in case))
        return self.collect(value) if self.collect else value

    def _shown(self, v):
        if self.show:
            return self.show(v)
        return _as_python(v) if self.python_values else _as_value(v)

    def _call_text(self, case) -> str:
        return f"{self.function}({', '.join(_as_code(a) for a in case)})"

    def _fn(self, code: str):
        run = run_code(code)
        return run.names.get(self.function) if run.error is None else None

    def check(self, text, inputs=None) -> Result:
        if text is MISSING or text is None:
            return Result("error", self.message("missing"))
        if is_empty_text(text, self.start) or _skeleton_def(text, self.function):
            cleared = not text.strip() and text.strip() != self.start.strip() and "cleared" in self.texts
            return Result("empty", self.message("cleared" if cleared else "empty"))
        for m in self.first:
            if m.matches(None, text):
                return Result(m.state, m.text, mistake=m.key)
        run = run_code(text)
        if run.timeout:
            return Result("slow", self.message("slow") if "slow" in self.texts else {
                "en": "Your code is still not done after {s:g} seconds. Does something keep going forever?".format(s=TIME_LIMIT),
                "nl": "Je code is na {s:g} seconden nog niet klaar. Gaat er iets eindeloos door?".format(s=TIME_LIMIT)})
        if run.error is not None:
            for m in self.mistakes:
                if m.error is not None and m.error(run.error):
                    return Result("error", m.text, mistake=m.key)
                found = re.search(m.syntax, text, flags=re.MULTILINE) if m.syntax else None
                if found:
                    line = text[text.rfind("\n", 0, found.start()) + 1:].split("\n", 1)[0].strip()
                    return Result("error", _fill(m.text, line=line), mistake=m.key)
            return Result("error", self._plain_error(run, text))
        fn = run.names.get(self.function)
        if not callable(fn):
            return Result("wrong", self.message("no_function", function=self.function) if "no_function" in self.texts else {
                "en": f"Your box does not make a function `{self.function}`. Start it with `def {self.function}(...):`, with that exact name.",
                "nl": f"Je vak maakt geen functie `{self.function}`. Begin met `def {self.function}(...):`, met precies die naam."},
                mistake="no-function")
        solution = self._fn(self.solution)
        lines = text.splitlines()
        cases = self.visible + self.hidden
        for i, case in enumerate(cases):
            call = self._call_text(case)
            out = io.StringIO()
            try:
                with contextlib.redirect_stdout(out):
                    got = with_time_limit(lambda: self._run(fn, case))
            except Timeout:
                return Result("slow", self.message("slow") if "slow" in self.texts else {
                    "en": f"With `{call}`, your function is still not done after {TIME_LIMIT:g} seconds. Does a loop keep going forever?",
                    "nl": f"Met `{call}` is je functie na {TIME_LIMIT:g} seconden nog niet klaar. Gaat een loop eindeloos door?"})
            except Exception as e:
                frames = [fr for fr in _tb_frames(e) if fr[0] == "<your code>"]
                if isinstance(e, TypeError) and not frames and "argument" in str(e):   # the call itself does not fit the def line
                    return Result("error", self.message("arguments", call=call, k=len(case)) if "arguments" in self.texts else {
                        "en": f"The check calls `{call}`, with {len(case)} value{'' if len(case) == 1 else 's'}, and the `def` line of your function has another number of names between its parentheses. It needs one name for each value, in the same order.",
                        "nl": f"De check roept `{call}` aan, met {len(case)} waarde{'' if len(case) == 1 else 'n'}, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Hij heeft één naam per waarde nodig, in dezelfde volgorde."},
                        mistake="arguments")
                inner = Run(error=e, line=frames[-1][1] if frames else None)
                for m in self.mistakes:
                    if m.error is not None and m.error(e):
                        return Result("error", _fill(m.text, call=call), mistake=m.key)
                    found = re.search(m.syntax, text, flags=re.MULTILINE) if m.syntax else None
                    if found:
                        line = text[text.rfind("\n", 0, found.start()) + 1:].split("\n", 1)[0].strip()
                        return Result("error", _fill(m.text, line=line), mistake=m.key)
                plain = self._plain_error(inner, text)
                if _plain_kind(e) == "other":   # nothing to explain but the line
                    plain = {lang: s.replace(PLAIN_ERRORS["other"][lang], "", 1).strip() for lang, s in plain.items()}
                if _plain_kind(e) == "none" and not inspect.isgeneratorfunction(fn):   # usually a way through it without return
                    plain = {lang: s + " " + _NONE_IN_FUNCTION[lang] for lang, s in plain.items()}
                return Result("error", {"en": f"With `{call}`, your function stops with an error. " + plain["en"],
                                        "nl": f"Met `{call}` stopt je functie met een fout. " + plain["nl"]})
            want = self._run(solution, case)
            if got is None and want is not None:
                printed = bool(out.getvalue().strip())
                key = "prints" if printed else "no_return"
                if key == "no_return" and not self.collect and re.search(r"^[ \t]*return[ \t]+\S", text, re.MULTILINE):
                    key = "unreached"   # there is a return with a value, but Python never gets to it
                default = {"prints": {"en": "Your function shows its result with `print`, but gives nothing back. Use `return` instead of `print`, so the result comes out of the function.",
                                      "nl": "Je functie laat de uitkomst zien met `print`, maar geeft niets terug. Gebruik `return` in plaats van `print`, dan komt de uitkomst uit de functie."},
                           "no_return": {"en": "Your function gives nothing back: it has no `return` with the result. Write `return` and the result as the last line of the function.",
                                         "nl": "Je functie geeft niets terug: er staat geen `return` met de uitkomst in. Schrijf `return` en de uitkomst als laatste regel van de functie."},
                           "unreached": {"en": f"With `{call}`, your function gives nothing back: Python never gets to a `return` with a value. Look at the spaces in front of each line: is every line where it belongs, and does every way through the function end with `return`?",
                                         "nl": f"Met `{call}` geeft je functie niets terug: Python komt nooit bij een `return` met een waarde. Kijk naar de spaties voor elke regel: staat elke regel waar hij hoort, en eindigt elke weg door de functie met `return`?"}}[key]
                return Result("wrong", self.message(key, call=call) if key in self.texts else default,
                              mistake="unreached" if key == "unreached" else key.replace("_", "-"))
            if self.compare(got, want):
                continue
            mine = None
            for m in self.mistakes:
                if m.syntax or m.error is not None:
                    continue
                if m.wrong is not None:
                    # a wrong version only names the mistake when it gives the same as the learner in every case
                    wrong_fn = self._fn(m.wrong)
                    mine = self._results(fn, cases) if mine is None else mine
                    if wrong_fn is not None and self._same_results(mine, self._results(wrong_fn, cases)):
                        return Result(m.state, _fill(m.text, call=call, got=self._shown(got), want=self._shown(want)), mistake=m.key)
                elif m.matches is not None and m.matches(got, text):
                    return Result(m.state, _fill(m.text, call=call, got=self._shown(got), want=self._shown(want)), mistake=m.key)
            if isinstance(got, Handed) and not got and want and "unreached" in self.texts:
                return Result("wrong", self.message("unreached", call=call), mistake="unreached")   # it never gets to a yield
            key = "almost" if i >= len(self.visible) else "wrong"
            return Result(key, self.message(key, call=call, got=self._shown(got), want=self._shown(want)))
        for m in self.form:
            if m.matches(None, text):
                return Result(m.state, m.text, mistake=m.key)
        return self._correct(text, n=len(cases))

    def _results(self, fn, cases) -> list:
        """What `fn` gives for each case; an error or a time-out counts as an outcome of its own."""
        out = []
        for case in cases:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    out.append(("value", with_time_limit(lambda: self._run(fn, case))))
            except Exception:   # also Timeout
                out.append(("error", None))
        return out

    def _same_results(self, a: list, b: list) -> bool:
        return all(x[0] == y[0] and (x[0] == "error" or self.compare(x[1], y[1])) for x, y in zip(a, b))


def _code_key(code: types.CodeType):
    """What a function does, without where it stands: its instructions, constants and names."""
    consts = tuple(_code_key(c) if isinstance(c, types.CodeType) else (type(c).__name__, repr(c)) for c in code.co_consts)
    return code.co_code, consts, code.co_names, code.co_varnames


def same_function(a, b) -> bool:
    """Two functions with the same code, whatever their comments, blank lines or place in a file."""
    try:
        return _code_key(a.__code__) == _code_key(b.__code__)
    except AttributeError:
        return False


def learner_traceback(e: BaseException, fn) -> str:
    """Python's traceback of an error in the learner's function, cut to the lines of the learner's own cell."""
    where = fn.__code__.co_filename
    lines = ["Traceback (most recent call last):"]
    last, repeats = None, 0
    for frame in traceback.extract_tb(e.__traceback__):
        if frame.filename != where:
            continue
        if (frame.lineno, frame.name) == last:   # a function that keeps calling itself: say so once, as Python does
            repeats += 1
            if repeats > 2:
                continue
        else:
            if repeats > 2:
                lines.append(f"  [Previous line repeated {repeats - 2} more times]")
            last, repeats = (frame.lineno, frame.name), 0
        lines.append(f"  Cell, line {frame.lineno}, in {frame.name}")
        if frame.line:
            lines.append(f"    {frame.line.strip()}")
    if repeats > 2:
        lines.append(f"  [Previous line repeated {repeats - 2} more times]")
    lines += "".join(traceback.TracebackException.from_exception(e).format_exception_only()).rstrip().splitlines()
    return "\n".join(lines)


@dataclass
class BugFix(Exercise):
    """Find the bug, from `py-errors` on. The lesson gives a function with a bug in an ordinary cell, and the
    learner fixes it there. The check calls the function on the lesson's example (`visible`) and other cases
    (`hidden`). An error is shown as Python's own traceback, cut to the learner's cell: reading it is the point.
    While the function is still the one the lesson gave, nothing counts yet: the message shows what goes wrong.
    `start` and `solution` are code text; a mistake's `wrong` is a wrong version, as code text too. When the
    solution itself stops with an error for a case, such as a ValueError it raises on purpose, the learner's
    function must stop with the same kind of error, with a message. The start may also be a skeleton (`...`):
    then empty_wrong says what to write. `form` holds mistakes in how it is written, checked once every case is
    right; each has `matches(None, fn)`.
    Texts: empty_error and error ({call}, {traceback}), empty_wrong, wrong and almost ({call}, {got}, {want}),
    not_raised ({call}, {got}, {kind}), no_message ({call}, {kind}), correct ({n})."""
    function: str = ""                 # the name of the function in the cell
    start: str = ""                    # the function with the bug, as the lesson's cell has it
    solution: str = ""                 # the function without the bug
    visible: list[tuple] = field(default_factory=list)
    hidden: list[tuple] = field(default_factory=list)
    compare: Callable[[Any, Any], bool] = same_value
    form: list[Mistake] = field(default_factory=list)

    def _call_text(self, case) -> str:
        return f"{self.function}({', '.join(_as_code(a) for a in case)})"

    def _fn(self, code: str):
        run = run_code(code)
        return run.names.get(self.function) if run.error is None else None

    @staticmethod
    def _call(fn, case):
        """(what fn gives, the error or None); the error is Timeout when it does not finish in time."""
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                return with_time_limit(lambda: fn(*(_copy(a) for a in case))), None
        except Exception as e:   # also Timeout
            return None, e

    @staticmethod
    def _printed(fn, case) -> str:
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                with_time_limit(lambda: fn(*(_copy(a) for a in case)))
        except Exception:
            pass
        return out.getvalue().strip()

    def _same_everywhere(self, fn, other, cases) -> bool:
        for case in cases:
            (a, ea), (b, eb) = self._call(fn, case), self._call(other, case)
            if (ea is None) != (eb is None) or (ea is None and not self.compare(a, b)) or (ea is not None and type(ea) is not type(eb)):
                return False
        return True

    def check(self, fn) -> Result:
        if fn is MISSING or not callable(fn):
            return Result("error", self.message("missing"))
        solution = self._fn(self.solution)
        cases = self.visible + self.hidden
        if same_function(fn, self._fn(self.start)):   # nothing changed yet: show what goes wrong with the example
            case = self.visible[0]
            got, e = self._call(fn, case)
            if e is not None:
                return Result("empty", self.message("empty_error", call=self._call_text(case), traceback=learner_traceback(e, fn)))
            want, _ = self._call(solution, case)
            return Result("empty", self.message("empty_wrong", call=self._call_text(case), got=_as_python(got), want=_as_python(want)))
        for i, case in enumerate(cases):
            call = self._call_text(case)
            got, e = self._call(fn, case)
            if isinstance(e, Timeout):
                return Result("slow", self.message("slow", call=call) if "slow" in self.texts else {
                    "en": f"With `{call}`, your function is still not done after {TIME_LIMIT:g} seconds. Does a loop keep going forever?",
                    "nl": f"Met `{call}` is je functie na {TIME_LIMIT:g} seconden nog niet klaar. Gaat een loop eindeloos door?"})
            want, we = self._call(solution, case)
            if we is not None:   # the right outcome is an error of this kind
                kind = type(we).__name__
                if e is None and isinstance(got, BaseException):   # `return ValueError(...)`: the error is given back, not raised
                    return Result("wrong", self.message("returned_error", call=call, kind=kind) if "returned_error" in self.texts else {
                        "en": f"With `{call}`, your function gives the error back with `return`, so it does not stop. Write `raise` instead of `return`.",
                        "nl": f"Met `{call}` geeft je functie de fout terug met `return`, dus stopt hij niet. Schrijf `raise` in plaats van `return`."},
                        mistake="returned-error")
                if e is None:
                    return Result("wrong", self.message("not_raised", call=call, got=_as_python(got), kind=kind), mistake="not-raised")
                if type(e) is type(we):
                    if str(e).strip().strip(".").strip("_").strip() == "":   # no message, or only the dots of an example
                        return Result("almost", self.message("no_message", call=call, kind=kind), mistake="no-message")
                    continue
            if e is not None:
                for m in self.mistakes:   # a known wrong version that stops in the same way, such as the bug itself
                    wrong = self._fn(m.wrong) if m.wrong is not None else None
                    if wrong is not None and self._same_everywhere(fn, wrong, cases):
                        return Result("error", _fill(m.text, call=call, traceback=learner_traceback(e, fn)), mistake=m.key)
                return Result("error", self.message("error", call=call, traceback=learner_traceback(e, fn)))
            if got is None and want is not None and self._printed(fn, case):
                return Result("wrong", self.message("prints", call=call) if "prints" in self.texts else {
                    "en": "Your function shows its result with `print`, but gives nothing back. Use `return` instead of `print`, so the result comes out of the function.",
                    "nl": "Je functie laat de uitkomst zien met `print`, maar geeft niets terug. Gebruik `return` in plaats van `print`, dan komt de uitkomst uit de functie."},
                    mistake="prints")
            if got is None and want is not None:
                return Result("wrong", self.message("no_return", call=call) if "no_return" in self.texts else {
                    "en": f"With `{call}`, your function gives nothing back: Python never gets to a `return` with a value. Does every way through it end with `return`?",
                    "nl": f"Met `{call}` geeft je functie niets terug: Python komt nooit bij een `return` met een waarde. Eindigt elke weg erdoorheen met `return`?"},
                    mistake="no-return")
            if self.compare(got, want):
                continue
            for m in self.mistakes:
                wrong = self._fn(m.wrong) if m.wrong is not None else None
                if wrong is not None and self._same_everywhere(fn, wrong, cases):
                    return Result(m.state, _fill(m.text, call=call, got=_as_python(got), want=_as_python(want)), mistake=m.key)
            key = "almost" if i >= len(self.visible) else "wrong"
            return Result(key, self.message(key, call=call, got=_as_python(got), want=_as_python(want)))
        for m in self.form:
            if m.matches(None, fn):
                return Result(m.state, m.text, mistake=m.key)
        return Result("correct", self.message("correct", n=len(cases)))


@dataclass
class Function(Exercise):
    """The learner writes a function; the check runs it on the example from the lesson,
    then on other cases. Each case is a tuple of arguments."""
    solution: Callable | None = None
    visible: list[tuple] = field(default_factory=list)   # the example from the lesson
    hidden: list[tuple] = field(default_factory=list)    # edge cases and other inputs
    compare: Callable[[Any, Any], bool] = same_value

    def check(self, fn) -> Result:
        if fn is MISSING or not callable(fn):
            return Result("error", self.message("missing"))
        if is_skeleton(fn):
            return Result("empty", self.message("empty"))
        cases = self.visible + self.hidden
        for i, case in enumerate(cases):
            args = tuple(_copy(a) for a in case)
            try:
                got = with_time_limit(lambda: fn(*args))
            except Timeout:
                return Result("slow", self.message("slow"))
            except Exception as e:
                return Result("error", self.message("error", error=f"{type(e).__name__}: {e}"))
            if got is None:
                return Result("wrong", self.message("no_return"), mistake="no-return")
            want = self.solution(*(_copy(a) for a in case))
            if self.compare(got, want):
                continue
            for m in self.mistakes:
                if m.wrong is not None and self.compare(got, m.wrong(*(_copy(a) for a in case))):
                    return Result(m.state, m.text, mistake=m.key)
            if i >= len(self.visible):
                return Result("almost", self.message("almost", case=_show(case), got=_show(got), want=_show(want)))
            return Result("wrong", self.message("wrong", case=_show(case), got=_show(got), want=_show(want)))
        return Result("correct", self.message("correct", n=len(cases)))


@dataclass
class Value(Exercise):
    """The learner sets something in the lesson, such as a slider, or makes a name in a cell; the
    check computes what it leads to (`outcome`) and compares that with the right outcome.
    Texts may use {got} (the outcome) and {value} (what the learner set).

    When the exercise uses names from the lesson's own cells, `inputs` holds their values as the
    lesson makes them, and the check cell passes the current ones: the learner may change them.
    `expect_from(*inputs)` gives the right outcome, and a mistake's `wrong(*inputs)` a wrong one."""
    start: Any = None
    solution: Any = None
    outcome: Callable[[Any], Any] = staticmethod(lambda v: v)
    expect: Any = MISSING
    compare: Callable[[Any, Any], bool] = same_value
    inputs: tuple = ()
    expect_from: Callable[..., Any] | None = None
    prose_numbers: bool = False   # {got} and {value} stand in running text: a decimal comma in Dutch

    def expected(self, inputs: tuple | None = None):
        if self.expect_from is not None:
            return self.expect_from(*(self.inputs if inputs is None else inputs))
        return self.expect if self.expect is not MISSING else self.outcome(self.solution)

    def check(self, value, inputs: tuple | None = None) -> Result:
        inputs = self.inputs if inputs is None else tuple(inputs)
        if value is MISSING:
            return Result("error", self.message("missing"))
        # a name that is still `...` is not filled in; None is "not set" only for something like a slider
        if value is Ellipsis or (value is None and self.start is None) or (value is not None and value == self.start):
            return Result("empty", self.message("empty"))
        got = self.outcome(value)
        shown = {"got": _show(got), "value": _show(value)}
        if self.prose_numbers:
            shown = {k: _prose(v) for k, v in shown.items()}
        if self.compare(got, self.expected(inputs)):
            return Result("correct", _fill(self.texts["correct"], **shown))
        for m in self.mistakes:
            hit = (m.wrong is not None and callable(m.wrong) and self.compare(got, m.wrong(*inputs))
                   or m.matches is not None and m.matches(got, value))
            if hit:
                return Result(m.state, _fill(m.text, **shown), mistake=m.key)
        return Result("wrong", _fill(self.texts["wrong"], **shown))


@dataclass
class Sort(Exercise):
    """The learner puts every item in one of the groups. With `nested`, the groups lie inside
    each other, widest first, and each item belongs in the narrowest group it fits in.

    The check looks at the items in their order and stops at the first one that is not right,
    so the learner hears about one item at a time. Texts may use {item} and {n}."""
    items: dict[str, Text] = field(default_factory=dict)      # item key -> how the item reads
    groups: dict[str, Text] = field(default_factory=dict)     # group key -> its name; widest first when nested
    solution: dict[str, str] = field(default_factory=dict)    # item key -> group key
    nudge: dict[str, Text] = field(default_factory=dict)      # item key -> a question that points the way, without the answer
    nested: bool = False
    start: dict = field(default_factory=dict)

    def check(self, value) -> Result:
        if value is MISSING or value is None:
            return Result("error", self.message("missing"))
        chosen = {k: v for k, v in dict(value).items() if v}
        if not chosen:
            return Result("empty", self.message("empty"))
        order = list(self.groups)
        for item, want in self.solution.items():
            got = chosen.get(item)
            if got is None or got == want:
                continue
            if self.nested and got in order and order.index(got) < order.index(want):
                key, state = "too_wide", "almost"
            elif self.nested:
                key, state = "too_narrow", "wrong"
            else:
                key, state = "wrong", "wrong"
            return Result(state, self._about(key, item), mistake=f"{key}:{item}")
        left = [k for k in self.solution if k not in chosen]
        if left:
            return Result("wrong", self.message("unfinished", n=len(left)), mistake="unfinished")
        return Result("correct", self.message("correct", n=len(self.solution)))

    def _about(self, key: str, item: str) -> Text:
        """The message for one item: what is off, then a question that points the way."""
        return {lang: self.texts[key][lang].format(item=_item_name(self.items[item][lang]))
                + " " + self.nudge[item][lang] for lang in self.texts[key]}


def _item_name(text: str) -> str:
    """An item that starts with a bold name ("**Spam filter.** Learned from ...") is called by that name."""
    m = re.match(r"\*\*(.+?)\.?\*\*", text)
    if m:
        return m.group(1)
    # Code stays as it is, so `_total` keeps its _ and `price * 2` its *; only emphasis around words goes.
    parts = re.split(r"(`[^`]*`)", text)
    return "".join(p if p.startswith("`") else re.sub(r"[*_]", "", p) for p in parts).rstrip(".")


def _copy(a):
    """The learner's code gets its own copy of the test data, so it cannot damage the check."""
    import copy
    return copy.deepcopy(a)


def _same_code(line: str) -> str:
    """A line of code without its spaces, and with one kind of quote: `x = 'a'` and `x="a"` are the same."""
    return re.sub(r"\s+", "", line).replace("'", '"')


class Code(str):
    """A value for `others` written as code, such as `np.array([[1, 2], [3, 4]])`: it goes into the line as it is,
    so a box can also be tried with another array."""

    def __repr__(self):
        return str(self)


def _as_code(value) -> str:
    """A value the way the lessons write it in code: strings with double quotes, also inside a list."""
    if isinstance(value, Code):
        return str(value)
    if isinstance(value, str):   # a backslash, a quote and a newline the way code writes them, so a newline shows as \n
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    if isinstance(value, list) and type(value).__repr__ is list.__repr__:   # a list of a lesson's own kind can name itself
        return "[" + ", ".join(_as_code(v) for v in value) + "]"
    if type(value) is dict:   # a plain one: a dict of a lesson's own kind keeps its own way of showing itself
        return "{" + ", ".join(f"{_as_code(k)}: {_as_code(v)}" for k, v in value.items()) + "}"
    return repr(value)


def _as_python(v) -> str:
    """A value the way Python shows it, for lessons that read Python's own output: `8.0`, not `8`.
    A NumPy number as the plain number it is; an array as a cell shows it, on one line."""
    np, torch = _numpy(), _torch()
    v = _plain(v)
    if np is not None and isinstance(v, np.ndarray) or torch is not None and isinstance(v, torch.Tensor):
        return " ".join(repr(v).split())
    return _as_code(v) if isinstance(v, (str, list)) else repr(v)


def _as_value(v) -> str:
    """A value a function gave back, the way the lessons write it: strings and lists as code, numbers as numbers."""
    return _as_code(v) if isinstance(v, (str, list)) or type(v) is dict else _show(v)


def _as_rounded(v, digits: int | None) -> str:
    """A value the way Python shows it; with `digits`, its floats, also inside a list or tuple, rounded to that many
    decimals and never in e-notation: 0.000096, not 9.6e-05."""
    v = _plain(v)
    if digits is None:
        return _as_python(v)
    if isinstance(v, float):
        text = f"{round(v, digits) + 0.0:.{digits}f}".rstrip("0")   # + 0.0: a tiny negative number shows as 0.0
        return text + "0" if text.endswith(".") else text
    if isinstance(v, (list, tuple)) and not isinstance(v, Handed):
        inner = ", ".join(_as_rounded(x, digits) for x in v)
        return f"[{inner}]" if isinstance(v, list) else f"({inner}{',' if len(v) == 1 else ''})"
    return _as_python(v)


def _show_rounded(v, digits: int | None) -> str:
    """As `_show`; with `digits`, its floats, also inside a list or tuple, rounded to that many decimals, without
    trailing zeros and never in e-notation."""
    v = _plain(v)
    if digits is None:
        return _show(v)
    if isinstance(v, float):
        return f"{round(v, digits) + 0.0:.{digits}f}".rstrip("0").rstrip(".")
    if isinstance(v, (list, tuple)) and not isinstance(v, Handed):
        inner = ", ".join(_show_rounded(x, digits) for x in v)
        return f"[{inner}]" if isinstance(v, list) else f"({inner}{',' if len(v) == 1 else ''})"
    return _show(v)


def _plain(v):
    """NumPy numbers inside a list or tuple as the plain numbers they are, so they show as `6`, not `np.int64(6)`."""
    np = _numpy()
    if np is not None and isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)) and not isinstance(v, Handed):
        return type(v)(_plain(x) for x in v)
    return v


def _show(v) -> str:
    v = _plain(v)
    pd = sys.modules.get("pandas")
    if pd is not None and isinstance(v, (pd.DataFrame, pd.Series)):   # as print shows it
        return str(v)
    np, torch = _numpy(), _torch()
    if np is not None and isinstance(v, np.ndarray):   # as a cell shows it: array([[1, 2], [3, 4]])
        return repr(v)
    if torch is not None and isinstance(v, torch.Tensor):   # as a cell shows it: tensor([[1., 2.], [3., 4.]])
        return " ".join(repr(v).split())
    if np is not None and isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, float) and v.is_integer():
        return str(int(v)) if abs(v) < 1e15 else str(v)
    return str(v)


def _prose(text: str):
    """A number shown in running text: a real minus sign, and in Dutch a decimal comma. Anything else stays as it is."""
    try:
        number = float(text)
    except ValueError:
        return text
    if "e" in text.lower():   # Python writes 0.00005 as 5e-05; running text writes it out
        text = f"{number:.12f}".rstrip("0").rstrip(".")
    text = text.replace("-", "\u2212")
    return {"en": text, "nl": text.replace(".", ",")}


def _for_lang(values: dict, lang: str) -> dict:
    """A value may differ per language, as a dict such as {"en": "nothing", "nl": "niets"}."""
    return {k: (v[lang] if isinstance(v, dict) and lang in v else v) for k, v in values.items()}


def _fill(text: Text, **values) -> Text:
    out = {}
    for lang, s in text.items():
        try:
            out[lang] = s.format(**_for_lang(values, lang))
        except (KeyError, IndexError, ValueError):
            out[lang] = s
    return out
