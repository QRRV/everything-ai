# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Your projects: the code you write and keep, as modules in work/projects/<package>/<module>.py (template 1.3).

Lessons import from there, so the folder is on Python's path in every lesson and every check. A check that
says your code is right can save it there; a lesson can give you starting files (lessons/<id>/project/).

A saved file is yours: the check saves over it only while it is still exactly what the check wrote. Once you
change it, the check leaves it alone. The folder comes last on Python's path, so a package of yours never
hides a module Python already has, such as `random` or `numpy`.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib
import importlib.util
import io
import os
import random
import shutil
import sys
import types
from pathlib import Path

from everything_ai import paths

HEADER = ("# Your own code: the Everything AI course saved it here when your check was right.\n"
          "# Je eigen code: de Everything AI-course heeft hem hier bewaard toen je check goed was.\n"
          "# Change it as you like; once you do, the check no longer saves over it.\n"
          "# Pas hem gerust aan; zodra je dat doet, bewaart de check er niet meer overheen.\n")
_MARK = "# check: "   # the last header line: a fingerprint of the code the check wrote

# Top-level lines that are kept when code is saved: what a module is for. Other lines, such as
# `print(mean([4, 8, 6]))` to try a function out, would run every time the module is imported.
_KEEP = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)


def folder() -> Path:
    return paths.WORK / "projects"


def shown(path: Path) -> str:
    """A project file the way the lessons write it: work/projects/tools/stats.py."""
    return "work/" + path.relative_to(paths.WORK).as_posix()


def on_path() -> None:
    """Make `from tools import stats` find work/projects/tools/stats.py. Last on the path, so that it never
    hides a module Python finds elsewhere."""
    folder().mkdir(parents=True, exist_ok=True)
    if str(folder()) not in sys.path:
        sys.path.append(str(folder()))


def forget() -> None:
    """Make the next import read the project files again, so a changed file counts at once."""
    root = str(folder()) + os.sep
    packages = {p.name for p in folder().iterdir() if p.is_dir()} if folder().is_dir() else set()
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None) or ""
        # a package folder of yours without __init__.py has no file, only a path
        where = [file] if file else [str(p) for p in (getattr(module, "__path__", None) or [])]
        # also a module without a file inside one of your packages, such as a copy of the course's version a check made
        if (any(w.startswith(root) for w in where) or name in _STAND_INS
                or ("." in name and not file and name.split(".")[0] in packages)):
            del sys.modules[name]
    _STAND_INS.clear()
    # Python's saved bytecode trusts a file's time to the second and its size: a file changed within the same
    # second to one of the same size would still run the old code.
    for cache in folder().rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    importlib.invalidate_caches()


def module_code(text: str) -> str:
    """The lines of `text` worth keeping in a module: imports, functions, classes and names. Other top-level
    lines, such as a `print` that tries the function out, are left out."""
    lines = text.strip().splitlines()
    try:
        tree = ast.parse(text.strip())
    except SyntaxError:
        return text.strip()
    drop = set()
    for node in tree.body:
        if not isinstance(node, _KEEP):
            drop.update(range(node.lineno - 1, node.end_lineno))
    kept = [line for i, line in enumerate(lines) if i not in drop]
    out = []
    for line in kept:   # no more than one empty line in a row
        if line.strip() or (out and out[-1].strip()):
            out.append(line)
    return "\n".join(out).strip()


def _fingerprint(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


def _changed_by_hand(target: Path) -> bool:
    """The file is no longer exactly what the check wrote."""
    text = target.read_text(encoding="utf-8")
    head, sep, body = text.partition("\n" + _MARK)
    if not sep:
        return True
    mark, _, code = body.partition("\n")
    return mark.strip() != _fingerprint(code.strip())


def save(package: str, module: str, text: str, whole: bool = False) -> tuple[Path, bool]:
    """Save code as work/projects/<package>/<module>.py, in a package that can be imported. With `whole`,
    every line is kept: a script that you run from the terminal, not a module that others import.
    Gives back the file, and whether it was kept as it is because you changed it yourself."""
    on_path()
    pkg = folder() / package
    pkg.mkdir(parents=True, exist_ok=True)
    init = pkg / "__init__.py"
    if not init.exists():
        init.write_text("", encoding="utf-8")
    target = pkg / f"{module}.py"
    if target.exists() and _changed_by_hand(target):
        return target, True
    code = text.strip() if whole else module_code(text)
    new = HEADER + _MARK + _fingerprint(code) + "\n" + code + "\n"
    if not target.exists() or target.read_text(encoding="utf-8") != new:
        target.write_text(new, encoding="utf-8")
        # Python trusts a cached .pyc when size and time (in whole seconds) match: drop it, so the next
        # import reads the new code even when it was written in the same second with the same size.
        Path(importlib.util.cache_from_source(str(target))).unlink(missing_ok=True)
    forget()
    return target, False


def provide(lesson_id: str) -> None:
    """Copy the files a lesson gives you into your project folder, once: your own changes are never overwritten."""
    source = paths.LESSONS / lesson_id / "project"
    if not source.is_dir():
        return
    for f in source.rglob("*.py"):
        target = folder() / f.relative_to(source)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, target)


# --- the course's version, for a lesson that builds on an earlier one (template 1.3) ------------------------------
# A later lesson uses your version of a project module. When yours is missing, or fails a short contract test, the
# lesson uses the course's own version instead, without touching your file, and says which lesson to go back to.
# The course's version is the solution of the exercise that saves the module; the contract test is that exercise's
# own scenarios, which must give with yours what they give with the course's version.

_BUILDS_ON: dict[tuple[str, str], tuple[str, str]] = {}   # (package, module) -> (lesson, exercise) that saves it
_STAND_INS: set[str] = set()   # names in sys.modules that hold the course's version, not a file of yours
_CHECKS: dict[str, object] = {}
_PASSED: dict[tuple, bool] = {}   # contract tests already run, by the code they ran on


def builds_on(*modules: tuple[str, str, str, str]) -> None:
    """Say which project modules this lesson uses, each as (package, module, the lesson that saves it, the exercise
    there), in the order they build on each other: the module the others import first. A lesson's checks say it
    once; it replaces what an earlier lesson in the same process said."""
    _BUILDS_ON.clear()
    for package, module, lesson, exercise in modules:
        _BUILDS_ON[(package, module)] = (lesson, exercise)


def _exercise(package: str, module: str):
    lesson, exercise = _BUILDS_ON[(package, module)]
    if lesson not in _CHECKS:
        spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson.replace('-', '_')}",
                                                      paths.LESSONS / lesson / "checks.py")
        mod = importlib.util.module_from_spec(spec)
        ours = dict(_BUILDS_ON)
        try:
            spec.loader.exec_module(mod)   # its own builds_on, if it has one, says what *that* lesson builds on
        finally:
            _BUILDS_ON.clear()
            _BUILDS_ON.update(ours)
        _CHECKS[lesson] = mod
    return _CHECKS[lesson].EXERCISES[exercise]


def course_code(package: str, module: str) -> str:
    """The course's own version of a module that this lesson builds on."""
    return module_code(_exercise(package, module).solution)


def _value(code: str, names: dict):
    """Run `code` in `names`, and give back the value of its last line."""
    tree = ast.parse(code, mode="exec")
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    exec(compile(tree, "<contract>", "exec"), names)
    return eval(compile(ast.Expression(last.value), "<contract>", "eval"), names) if last is not None else None


def _passes(package: str, module: str, mod) -> bool:
    """The contract test: every scenario of the exercise that saves the module gives with yours what it gives with
    the course's version. For a function exercise, every case it checks (`visible` and `hidden`) does."""
    from everything_ai.check import with_time_limit
    exercise = _exercise(package, module)
    course = {"__name__": "__course__"}
    seed = random.getstate()   # the scenarios may seed `random`: the lesson's own random numbers stay as they were
    try:
        exec(compile(course_code(package, module), "<course>", "exec"), course)
        for scenario in getattr(exercise, "scenarios", []):
            got = with_time_limit(lambda: _value(scenario, {**vars(mod), "__name__": "__contract__"}))
            want = _value(scenario, {**course})
            if not exercise.compare(got, want):
                return False
        name = getattr(exercise, "function", "")
        if name:   # a function exercise has no scenarios: its own cases are the contract
            fn = getattr(mod, name, None)
            if not callable(fn):
                return False
            with contextlib.redirect_stdout(io.StringIO()):
                for case in list(exercise.visible) + list(exercise.hidden):
                    got = with_time_limit(lambda: exercise._run(fn, case))
                    if not exercise.compare(got, exercise._run(course[name], case)):
                        return False
    except Exception:   # also Timeout
        return False
    finally:
        random.setstate(seed)
    return True


def state(package: str, module: str) -> str:
    """"yours" when work/projects/<package>/<module>.py is there and passes its contract test, else "missing" or
    "broken". Modules it imports that are not yours are the course's version already."""
    path = folder() / package / f"{module}.py"
    if not path.exists():
        return "missing"
    key = (package, module, path.read_text(encoding="utf-8"), tuple(sorted(_STAND_INS)))
    if key not in _PASSED:
        from everything_ai.check import with_time_limit
        try:
            # your code runs here, before the code of the box: within the time limit, and what it prints stays out
            with contextlib.redirect_stdout(io.StringIO()):
                mod = with_time_limit(lambda: importlib.import_module(f"{package}.{module}"))
                _PASSED[key] = f"{package}.{module}" not in _STAND_INS and _passes(package, module, mod)
        except Exception:   # also Timeout
            _PASSED[key] = False
    return "yours" if _PASSED[key] else "broken"


def use_course_versions() -> list[tuple[str, str, str, str]]:
    """Where a module this lesson builds on is missing or broken, put the course's version in its place for this
    run, so code of yours that imports it gets the course's. Gives back what it replaced: (package, module,
    "missing" or "broken", the lesson where you write it)."""
    from everything_ai.check import with_time_limit
    out = []
    for (package, module), (lesson, _) in list(_BUILDS_ON.items()):
        st = state(package, module)
        if st == "yours":
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                pkg = with_time_limit(lambda: importlib.import_module(package))
        except Exception:   # also Timeout
            pkg = types.ModuleType(package)
            pkg.__path__ = []
            sys.modules[package] = pkg
            _STAND_INS.add(package)
        name = f"{package}.{module}"
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        _STAND_INS.add(name)
        exec(compile(course_code(package, module), f"<course {name}>", "exec"), mod.__dict__)
        setattr(pkg, module, mod)
        out.append((package, module, st, lesson))
    return out


def stand_ins() -> list[tuple[str, str, str, str]]:
    """The modules this lesson builds on that the lesson takes from the course right now, as use_course_versions."""
    on_path()
    forget()
    out = use_course_versions()
    forget()
    return out
