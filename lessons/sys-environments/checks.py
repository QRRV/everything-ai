# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-environments: the same packages everywhere. Code that runs on one computer can stop on
another because a package has another version there. A virtual environment keeps the packages of one project apart,
and a lock file records the exact version of every package, so every computer installs the same. The learner lists
the packages whose versions differ between two computers, and picks the newest release a version range allows, as an
installer does; the demo installs NumPy on a chosen day with and without a pin.

The versions of the laptop, the server and the colleague are a scenario of the course, with versions that exist on
PyPI; the NumPy releases and their dates come from PyPI (see RELEASES)."""
import html
import math
import re
import tomllib

from everything_ai import check, paths, yours
from everything_ai.check import FunctionBox, Mistake

LESSON = "sys-environments"
LANGS = ("en", "nl")

# The worked example and the guess: the course's own files. pyproject.toml names the packages the course uses
# itself; uv.lock pins every package uv installs for it, the course itself included.
PYPROJECT = paths.ROOT / "pyproject.toml"
LOCK_FILE = paths.ROOT / "uv.lock"


def _named() -> list:
    """The packages the course's pyproject.toml names, without their versions."""
    with open(PYPROJECT, "rb") as f:
        needs = tomllib.load(f)["project"]["dependencies"]
    return [re.split(r"[=<>~!;\[ ]", need, maxsplit=1)[0] for need in needs]


def _locked() -> int:
    """How many packages uv.lock pins: one [[package]] table each."""
    with open(LOCK_FILE, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip() == "[[package]]")


NAMED = _named()
LOCKED = _locked()

# Exercise 1: the packages on two computers, as `pip freeze` or `uv pip list` would show them.
LAPTOP = {"numpy": "1.26.4", "pandas": "2.2.2", "scikit-learn": "1.5.1", "matplotlib": "3.9.2", "joblib": "1.4.2"}
SERVER = {"numpy": "2.0.1", "pandas": "2.2.2", "scikit-learn": "1.5.1", "matplotlib": "3.9.2"}
COLLEAGUE = {"numpy": "1.26.3", "pandas": "2.2.2", "scikit-learn": "1.5.1", "matplotlib": "3.9.2", "joblib": "1.4.2"}


def differences(mine, theirs):
    found = []
    for name in mine:
        if name not in theirs:
            found.append(name)
        elif mine[name] != theirs[name]:
            found.append(name)
    return found


# Exercise 2 and the demo: every final NumPy release on PyPI from 1.24.0 to 2.5.5, read from pypi.org/pypi/numpy/json on
# 2026-11-27, each with the day (UTC) its first file came out. 2.4.0 (2025-12-20) is left out: PyPI marks it yanked
# ("Backward compatibility bug"), and an installer skips a yanked release unless a line pins it exactly.
RELEASES_TEXT = """
1.24.0 2022-12-18
1.24.1 2022-12-26
1.24.2 2023-02-05
1.24.3 2023-04-22
1.25.0 2023-06-17
1.24.4 2023-06-26
1.25.1 2023-07-08
1.25.2 2023-07-31
1.26.0 2023-09-16
1.26.1 2023-10-14
1.26.2 2023-11-12
1.26.3 2024-01-02
1.26.4 2024-02-05
2.0.0 2024-06-16
2.0.1 2024-07-21
2.1.0 2024-08-18
2.0.2 2024-08-26
2.1.1 2024-09-03
2.1.2 2024-10-05
2.1.3 2024-11-02
2.2.0 2024-12-08
2.2.1 2024-12-21
2.2.2 2025-01-18
2.2.3 2025-02-13
2.2.4 2025-03-16
2.2.5 2025-04-19
2.2.6 2025-05-17
2.3.0 2025-06-07
2.3.1 2025-06-21
2.3.2 2025-07-24
2.3.3 2025-09-09
2.3.4 2025-10-15
2.3.5 2025-11-16
2.4.1 2026-01-10
2.4.2 2026-01-31
2.4.3 2026-03-09
2.4.4 2026-03-29
2.4.5 2026-05-15
2.4.6 2026-05-18
2.5.0 2026-06-21
2.5.1 2026-07-04
2.5.2 2026-08-09
2.5.3 2026-09-06
2.5.4 2026-10-10
2.5.5 2026-11-15
"""
RELEASES = [(tuple(int(p) for p in v.split(".")), d) for v, d in (line.split() for line in RELEASES_TEXT.strip().splitlines())]
RELEASED = sorted(v for v, _ in RELEASES)     # from the oldest version to the newest, as exercise 2 gets them


def resolve(released, low, high):
    chosen = None
    for version in released:
        if version >= low and version < high:
            chosen = version
    return chosen


def shown(version) -> str:
    """A version as people write it: (1, 26, 4) as 1.26.4."""
    return ".".join(str(p) for p in version) if isinstance(version, tuple) else str(version)


RANGE = ((1, 24, 0), (2, 0, 0))        # numpy>=1.24,<2: the range of the check of exercise 2
RANGE_2 = ((2, 0, 0), (3, 0, 0))       # numpy>=2.0,<3: the second range in the picture of exercise 2
PICK = resolve(RELEASED, *RANGE)       # (1, 26, 4): the last NumPy 1
PICK_2 = resolve(RELEASED, *RANGE_2)   # the newest NumPy 2

# The demo: an install on the first day of each month, from January 2023 on (1.24.1, the newest release on that
# day, came out in December 2022).
FIRST = (2023, 1)
MONTHS = []
_y, _m = FIRST
while (_y, _m) <= (2026, 11):
    MONTHS.append((_y, _m))
    _y, _m = (_y + 1, 1) if _m == 12 else (_y, _m + 1)
PINNED = (1, 24, 1)
LOCKED_ON = (2023, 1)                  # the day of `uv lock` in the demo: 1 January 2023


def available(year: int, month: int) -> list:
    """The releases out before the first day of this month, from the oldest version to the newest, as `resolve`
    needs them: 2.0.2 came out after 2.1.0."""
    return sorted(v for v, d in RELEASES if d < f"{year}-{month:02d}-01")


def installs(year: int, month: int) -> dict:
    """What each line of the demo installs on the first day of this month."""
    out = available(year, month)
    return {"bare": resolve(out, (0, 0, 0), (99, 0, 0)),
            "range": resolve(out, *RANGE),
            "pinned": PINNED if PINNED in out else None,
            "lock": resolve(available(*LOCKED_ON), (0, 0, 0), (99, 0, 0))}


FIRST_DRIFT = next(i for i, (y, m) in enumerate(MONTHS) if installs(y, m)["range"] != PINNED)   # March 2023
FIRST_TWO = next(i for i, (y, m) in enumerate(MONTHS) if installs(y, m)["bare"] >= (2, 0, 0))     # July 2024


# ======== words and drawing ===============================================================================================

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem;max-width:var(--ea-measure)"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem;max-width:var(--ea-measure)"
_SVG = "width:100%;max-width:330px;height:auto;font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums"


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma and a thousands dot in Dutch, and a real minus sign."""
    out = f"{x:,.{digits}f}"
    out = out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out
    return out.replace("-", "−")


def short(x: float, digits: int, lang: str) -> str:
    """A number with at most `digits` decimals, without zeros at its end: 0.3, 0.73, 1."""
    out = f"{round(x, digits) + 0.0:.{digits}f}".rstrip("0").rstrip(".") if digits else f"{round(x):d}"
    out = "0" if out in ("-0", "") else out
    return (out.replace(".", ",") if lang == "nl" else out).replace("-", "−")


def _marked(text_: str) -> str:
    """Escaped text in which `name` is set as code and *word* in italics."""
    out = []
    for k, part in enumerate(html.escape(text_).split("`")):
        if k % 2:
            out.append(f"<code>{part}</code>")
        else:
            out.append("".join(f"<i>{p}</i>" if j % 2 else p for j, p in enumerate(part.split("*"))))
    return "".join(out)


def _note(text_: str, style: str = _UI) -> str:
    return f'<div style="{style}">{_marked(text_)}</div>'


def _t(x: float, y: float, text_: str, size: float = 12.5, weight: int = 400, fill: str = "var(--ea-ink)",
       anchor: str = "start", halo: bool = False) -> str:
    rim = ";paint-order:stroke;stroke:var(--ea-page);stroke-width:4px;stroke-linejoin:round" if halo else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" style="font-size:{size}px;font-weight:{weight};'
            f'fill:{fill}{rim}">{html.escape(text_)}</text>')


def _line(x1: float, y1: float, x2: float, y2: float, ink: str = "var(--ea-line)", width: float = 1.0,
          dash: str = "") -> str:
    dashed = f";stroke-dasharray:{dash}" if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'style="stroke:{ink};stroke-width:{width}{dashed}"/>')


def _rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "") -> str:
    edge = f";stroke:{stroke};stroke-width:1" if stroke else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" style="fill:{fill}{edge}"/>'


def _box(x: float, y: float, w: float, h: float, stroke: str = "var(--ea-ink-3)", width: float = 1.2) -> str:
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="5" '
            f'style="fill:var(--ea-page);stroke:{stroke};stroke-width:{width}"/>')


def _arrow(x1: float, y1: float, x2: float, y2: float, ink: str = "var(--ea-ink-2)", width: float = 1.3) -> str:
    """A line from (x1, y1) to (x2, y2) with a small head at its end."""
    a = math.atan2(y2 - y1, x2 - x1)
    h1 = (x2 - 6 * math.cos(a - 0.45), y2 - 6 * math.sin(a - 0.45))
    h2 = (x2 - 6 * math.cos(a + 0.45), y2 - 6 * math.sin(a + 0.45))
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" style="stroke:{ink};stroke-width:{width}"/>'
            f'<path d="M{h1[0]:.1f},{h1[1]:.1f} L{x2:.1f},{y2:.1f} L{h2[0]:.1f},{h2[1]:.1f}" '
            f'style="fill:none;stroke:{ink};stroke-width:{width}"/>')


def _svg(height: float, alt: str, parts: list) -> str:
    return (f'<svg viewBox="0 0 330 {height:.0f}" role="img" aria-label="{html.escape(alt)}" style="{_SVG}">'
            f'{"".join(parts)}</svg>')


# ======== the picture of the idea =========================================================================================

IDEA_TEXT = {
    "en": {"title": "One line, two computers", "without": "Without a lock file", "with": "With a lock file",
           "need": "pyproject.toml", "march": "install, March 2024", "july": "install, July 2024",
           "lock": "uv lock, once", "sync": "uv sync, everywhere", "laptop": "your laptop", "server": "the server",
           "locked": "uv.lock: numpy 1.26.4", "all": "and every other package, exact",
           "differs": "differs", "same": "the same everywhere",
           "alt": "Above, the line numpy installed on two days: version 1.26.4 on the laptop in March 2024 and version 2.0.0 on the server in July 2024, which differs. Below, uv lock writes the exact version 1.26.4 into uv.lock once, and uv sync installs that version on the laptop and on the server: the same everywhere."},
    "nl": {"title": "Eén regel, twee computers", "without": "Zonder lock file", "with": "Met een lock file",
           "need": "pyproject.toml", "march": "installatie, maart 2024", "july": "installatie, juli 2024",
           "lock": "uv lock, één keer", "sync": "uv sync, overal", "laptop": "je laptop", "server": "de server",
           "locked": "uv.lock: numpy 1.26.4", "all": "en elk ander package, exact",
           "differs": "anders", "same": "overal hetzelfde",
           "alt": "Bovenaan: de regel numpy, geïnstalleerd op twee dagen: versie 1.26.4 op de laptop in maart 2024 en versie 2.0.0 op de server in juli 2024, een andere versie. Onderaan schrijft uv lock de exacte versie 1.26.4 één keer in uv.lock, en installeert uv sync deze versie op de laptop en op de server: overal hetzelfde."},
}


def _mono(x: float, y: float, text_: str, size: float = 10.5, fill: str = "var(--ea-ink)", anchor: str = "start") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" style="font-family:var(--ea-font-code);font-variant-ligatures:none;'
            f'font-size:{size}px;fill:{fill}">{html.escape(text_)}</text>')


def idea_picture(lang: str) -> str:
    """The line numpy installed on two days without a lock file, and through uv.lock with one."""
    w = IDEA_TEXT[lang]
    parts = [_t(0, 14, w["title"], 12.5, 600)]
    # without a lock file: two installs, two versions
    y = 36.0
    parts.append(_t(0, y, w["without"], 11.5, 600, "var(--ea-ink-2)"))
    top = y + 10
    parts.append(_box(0, top + 14, 92, 34))
    parts.append(_t(46, top + 27, w["need"], 9.5, 600, "var(--ea-ink-2)", "middle"))
    parts.append(_mono(46, top + 41, "numpy", 10.5, anchor="middle"))
    for k, (when, where, version, bad) in enumerate(((w["march"], w["laptop"], "1.26.4", False),
                                                      (w["july"], w["server"], "2.0.0", True))):
        by = top + k * 40
        parts.append(_arrow(94, top + 31, 192, by + 15))
        parts.append(_t(143, by + 7 + (0 if k == 0 else 26), when, 9, 400, "var(--ea-ink-2)", "middle"))
        stroke = "var(--ea-loss)" if bad else "var(--ea-ink-3)"
        parts.append(_box(194, by, 134, 30, stroke, 1.6 if bad else 1.2))
        parts.append(_t(200, by + 12, where, 9.5, 400, "var(--ea-ink-2)"))
        parts.append(_mono(200, by + 25, f"numpy {version}", 10.5, "var(--ea-loss-text)" if bad else "var(--ea-ink)"))
        if bad:
            parts.append(_t(322, by + 25, w["differs"], 9.5, 600, "var(--ea-loss-text)", "end"))
    # with a lock file: resolved once, installed everywhere
    y = top + 96
    parts.append(_t(0, y, w["with"], 11.5, 600, "var(--ea-ink-2)"))
    top = y + 10
    parts.append(_box(0, top + 14, 92, 34))
    parts.append(_t(46, top + 27, w["need"], 9.5, 600, "var(--ea-ink-2)", "middle"))
    parts.append(_mono(46, top + 41, "numpy", 10.5, anchor="middle"))
    parts.append(_arrow(46, top + 50, 46, top + 72))
    parts.append(_t(52, top + 65, w["lock"], 9, 400, "var(--ea-ink-2)"))
    parts.append(_box(0, top + 74, 150, 34, "var(--ea-ink)", 1.4))
    parts.append(_mono(6, top + 88, w["locked"], 10.5))
    parts.append(_t(6, top + 101, w["all"], 9, 400, "var(--ea-ink-2)"))
    for k, where in enumerate((w["laptop"], w["server"])):
        by = top + 52 + k * 40
        parts.append(_arrow(152, top + 91, 192, by + 15))
        parts.append(_box(194, by, 134, 30))
        parts.append(_t(200, by + 12, where, 9.5, 400, "var(--ea-ink-2)"))
        parts.append(_mono(200, by + 25, "numpy 1.26.4", 10.5))
    parts.append(_t(172, top + 44, w["sync"], 9, 400, "var(--ea-ink-2)", "middle"))
    parts.append(_t(328, top + 140, w["same"], 10.5, 600, "var(--ea-ink)", "end"))
    height = top + 148
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(w["alt"])}" style="margin:0.25rem 0 0.75rem;'
            f'max-width:min(100%,330px)"><svg viewBox="0 0 330 {height:.0f}" aria-hidden="true" style="{_SVG}">'
            f'{"".join(parts)}</svg></figure>')


# ======== the pictures of the learner's work ==============================================================================

PICTURE_SECONDS = 2   # what the learner's function gets for all the calls of one picture together
_STOPPED = object()   # what _call gives for a function that stopped with an error


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _call(make_):
    """make_() within a time limit; _STOPPED when it stops with an error or runs out of time."""
    try:
        return check.with_time_limit(make_, PICTURE_SECONDS)
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _strings(v) -> list | None:
    """v as a list of package names, or None: never anything else."""
    if isinstance(v, (list, tuple)) and len(v) <= 40 and all(isinstance(x, str) for x in v):
        return list(v)
    return None


def _version(v):
    """v as a version tuple of whole numbers, None for nothing, or _STOPPED for anything else."""
    if v is None:
        return None
    if isinstance(v, tuple) and 1 <= len(v) <= 4 and all(isinstance(p, int) and not isinstance(p, bool) and 0 <= p < 1000 for p in v):
        return v
    return _STOPPED


def table_svg(tables: list, lang: str, alt: str) -> str:
    """For each pair of computers, every package with its two versions, and the packages the learner's function
    lists, marked in the color of mistakes: a difference is a mistake waiting to happen."""
    w = _YOURS_TEXT[lang]["differences"]
    X1, X2, X3 = 96.0, 166.0, 238.0
    parts = []
    y = 4.0
    for title, names_, mine, theirs, found in tables:
        parts.append(_t(0, y + 12, title, 11.5, 600))
        y += 30
        for x, head in ((0, w["package"]), (X1, names_[0]), (X2, names_[1]), (X3, w["yours"])):
            parts.append(_t(x, y, head, 9.5, 600, "var(--ea-ink-2)"))
        parts.append(_line(0, y + 5, 328, y + 5, "var(--ea-line)"))
        y += 6
        rows = list(mine) + [n for n in theirs if n not in mine]
        for name in rows:
            flagged = name in found
            ink = "var(--ea-loss-text)" if flagged else "var(--ea-ink)"
            parts.append(_mono(0, y + 12, name, 10, ink))
            parts.append(_mono(X1, y + 12, mine.get(name, "–"), 10, ink))
            parts.append(_mono(X2, y + 12, theirs.get(name, "–"), 10, ink))
            if flagged:
                parts.append(_rect(X3, y + 4, 8, 8, "var(--ea-loss)"))
                parts.append(_t(X3 + 12, y + 12, w["listed"], 9.5, 600, "var(--ea-loss-text)"))
            y += 16
        y += 14
    return _svg(y, alt, parts)


def strip_svg(rows: list, lang: str, alt: str) -> str:
    """For each range, every NumPy release as a dot from old to new, the releases the range allows on a band, and the
    release the learner's function picks, in ink when it is the newest allowed and in the color of mistakes when not."""
    w = _YOURS_TEXT[lang]["resolve"]
    X0, X1 = 6.0, 322.0
    step = (X1 - X0) / max(len(RELEASED) - 1, 1)
    parts = []
    y = 4.0
    for title, low, high, mine in rows:
        right = resolve(RELEASED, low, high)
        parts.append(_mono(0, y + 12, title, 10.5))
        y += 30
        allowed = [i for i, v in enumerate(RELEASED) if low <= v < high]
        if allowed:
            parts.append(_rect(X0 + allowed[0] * step - 4, y - 7, (allowed[-1] - allowed[0]) * step + 8, 14,
                               "var(--ea-inset)", "var(--ea-line)"))
        for i, v in enumerate(RELEASED):
            parts.append(f'<circle cx="{X0 + i * step:.1f}" cy="{y:.1f}" r="2.4" style="fill:var(--ea-ink-3)"/>')
        for i, v in enumerate(RELEASED):
            if i == 0 or RELEASED[i - 1][:2] != v[:2]:   # the first release of each minor version
                parts.append(_t(max(X0 + i * step, 10), y + 18 + (9 if v[0] == 1 and v[1] % 2 else 0), f"{v[0]}.{v[1]}", 8.5,
                                400, "var(--ea-ink-2)", "middle"))
        if mine in RELEASED:
            i = RELEASED.index(mine)
            ink = "var(--ea-ink)" if mine == right else "var(--ea-loss)"
            parts.append(f'<circle cx="{X0 + i * step:.1f}" cy="{y:.1f}" r="5" style="fill:none;stroke:{ink};stroke-width:2"/>')
        said = w["none"] if mine is None else w["mine"].format(v=shown(mine))
        good = mine == right
        parts.append(_t(1, y + 44, said, 11, 600, "var(--ea-ink)" if good else "var(--ea-loss-text)"))
        y += 62
    return _svg(y, alt, parts)


_YOURS_TEXT = {
    "en": {"differences": {"title": "Your differences on two pairs of computers",
                           "intro": "Every package with its version on both computers, and the packages your `differences(mine, theirs)` lists, marked in the color of mistakes.",
                           "laptop_server": "Your laptop and the server", "laptop_colleague": "Your laptop and a colleague",
                           "laptop": "laptop", "server": "server", "colleague": "colleague",
                           "package": "package", "yours": "your function", "listed": "differs",
                           "alt": "Two tables of packages with their versions on two computers, with the packages your function lists as different marked.",
                           "right": "Your function lists every package that is missing on the other computer or has another version there, and nothing else.",
                           "wrong": "Your function's list is not yet right for every pair: compare the marked rows with the versions next to them.",
                           "none": "Your `differences` stops with an error, takes too long, or does not give a list of names, so nothing can be drawn yet."},
           "resolve": {"title": "Your resolve on two ranges",
                       "intro": "Every NumPy release on PyPI as a dot, from old to new. The band holds the releases a range allows, and the ring is the release your `resolve` picks.",
                       "mine": "your function: {v}", "none": "your function: None",
                       "alt": "The NumPy releases as dots from old to new, for two ranges, with the allowed releases on a band and the release your function picks circled.",
                       "right": "Your function picks the newest release each range allows, as an installer does: a range gives a different version as new releases come out.",
                       "wrong": "Your function does not pick the newest allowed release each time: its pick is circled in the color of mistakes.",
                       "fail": "Your `resolve` stops with an error, takes too long, or does not give a version or `None`, so nothing can be drawn yet."}},
    "nl": {"differences": {"title": "Je differences op twee paren computers",
                           "intro": "Elk package met de versie op beide computers, en de packages die jouw `differences(mine, theirs)` noemt, gemarkeerd in de kleur van fouten.",
                           "laptop_server": "Je laptop en de server", "laptop_colleague": "Je laptop en een collega",
                           "laptop": "laptop", "server": "server", "colleague": "collega",
                           "package": "package", "yours": "je functie", "listed": "anders",
                           "alt": "Twee tabellen met packages en de versies op twee computers, met de packages die je functie als anders noemt gemarkeerd.",
                           "right": "Je functie noemt elk package dat op de andere computer ontbreekt of daar een andere versie heeft, en verder niets.",
                           "wrong": "De lijst van je functie klopt nog niet voor elk paar: vergelijk de gemarkeerde regels met de versies ernaast.",
                           "none": "Je `differences` stopt met een fout, duurt te lang of geeft geen lijst met namen, dus is er nog geen afbeelding."},
           "resolve": {"title": "Je resolve bij twee bereiken",
                       "intro": "Elke release van NumPy op PyPI als een stip, van oud naar nieuw. De band bevat de releases die een bereik toestaat, en de ring is de release die jouw `resolve` kiest.",
                       "mine": "je functie: {v}", "none": "je functie: None",
                       "alt": "De releases van NumPy als stippen van oud naar nieuw, voor twee bereiken, met de toegestane releases op een band en de release die je functie kiest omcirkeld.",
                       "right": "Je functie kiest de nieuwste release die elk bereik toestaat, net als een installer: een bereik geeft een andere versie zodra er nieuwe releases uitkomen.",
                       "wrong": "Je functie kiest niet elke keer de nieuwste toegestane release: de keuze heeft een ring in de kleur van fouten.",
                       "fail": "Je `resolve` stopt met een fout, duurt te lang of geeft geen versie of `None`, dus is er nog geen afbeelding."}},
}


def differences_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["differences"]
    fn = _fn(seen)
    pairs = [(w["laptop_server"], (w["laptop"], w["server"]), LAPTOP, SERVER),
             (w["laptop_colleague"], (w["laptop"], w["colleague"]), LAPTOP, COLLEAGUE)]
    tables = []
    for title, names_, mine, theirs in pairs:
        v = _call(lambda a=mine, b=theirs: fn(dict(a), dict(b))) if fn is not None else _STOPPED
        found = _strings(v) if v is not _STOPPED else None
        if found is None:
            return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
        tables.append((title, names_, mine, theirs, found))
    right = all(t[4] == differences(t[2], t[3]) for t in tables)
    return yours.figure(w["title"], _note(w["intro"]) + table_svg(tables, lang, w["alt"]) +
                        _note(w["right"] if right else w["wrong"], _UI if right else _LOSS_UI), alt=w["title"])


def resolve_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["resolve"]
    fn = _fn(seen)
    rows = []
    for title, (low, high) in (("numpy>=1.24,<2", RANGE), ("numpy>=2.0,<3", RANGE_2)):
        v = _call(lambda a=low, b=high: fn(list(RELEASED), a, b)) if fn is not None else _STOPPED
        mine = _version(v) if v is not _STOPPED else _STOPPED
        if mine is _STOPPED:
            return yours.figure(w["title"], _note(w["intro"]) + _note(w["fail"], _LOSS_UI), alt=w["title"])
        rows.append((title, low, high, mine))
    right = all(m == resolve(RELEASED, low, high) for _, low, high, m in rows)
    return yours.figure(w["title"], _note(w["intro"]) + strip_svg(rows, lang, w["alt"]) +
                        _note(w["right"] if right else w["wrong"], _UI if right else _LOSS_UI), alt=w["title"])


# ======== the demo: one line, installed on a chosen day ===================================================================

MONTH_NAMES = {
    "en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"],
    "nl": ["januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus", "september", "oktober",
           "november", "december"],
}

DEMO_TEXT = {
    "en": {"title": "Installed on 1 {date}", "lock": "uv.lock, made on 1 January 2023",
           "old": "a package built for NumPy 1.x, beside the line numpy:", "runs": "runs",
           "stops": "stops with an ImportError", "differs": "differs", "axis": "NumPy releases on PyPI",
           "foot": "Each install takes the newest release a line allows on that day.",
           "alt": "The NumPy releases on a time line, and what four lines install on 1 {date}: numpy gives {v}, numpy>=1.24,<2 gives {r}, and numpy==1.24.1 and the lock file give 1.24.1. A package built for NumPy 1.x {old}.",
           "says": {
               "same": "On **1 {date}**, every line installs NumPy 1.24.1, and the package built for NumPy 1.x runs.",
               "drift": "On **1 {date}**, the line `numpy` without a version and the range both install NumPy {v}, and the pin and the lock file still give 1.24.1. Install on two different days, and you now get two different versions.",
               "split": "On **1 {date}**, the line `numpy` without a version installs NumPy {v}, and the package built for NumPy 1.x stops with an error. The range installs {r}, and the pin and the lock file still give 1.24.1.",
           }},
    "nl": {"title": "Geïnstalleerd op 1 {date}", "lock": "uv.lock, gemaakt op 1 januari 2023",
           "old": "een package gebouwd voor NumPy 1.x, naast de regel numpy:", "runs": "werkt",
           "stops": "stopt met een ImportError", "differs": "anders", "axis": "releases van NumPy op PyPI",
           "foot": "Elke regel krijgt de nieuwste release die op deze dag toegestaan is.",
           "alt": "De releases van NumPy op een tijdlijn, en wat vier regels installeren op 1 {date}: numpy geeft {v}, numpy>=1.24,<2 geeft {r}, en numpy==1.24.1 en de lock file geven 1.24.1. Een package gebouwd voor NumPy 1.x {old}.",
           "says": {
               "same": "Op **1 {date}** installeert elke regel NumPy 1.24.1, en werkt het package dat voor NumPy 1.x gebouwd is.",
               "drift": "Op **1 {date}** installeren de regel `numpy` zonder versie en het bereik allebei NumPy {v}, en geven de vaste versie en de lock file nog steeds 1.24.1. Installeer je op twee verschillende dagen, dan krijg je nu twee verschillende versies.",
               "split": "Op **1 {date}** installeert de regel `numpy` zonder versie NumPy {v}, en stopt het package dat voor NumPy 1.x gebouwd is met een fout. Het bereik installeert {r}, en de vaste versie en de lock file geven nog steeds 1.24.1.",
           }},
}


def date(k: int, lang: str) -> str:
    """The k-th month of the demo as a date without its day: March 2023, maart 2023."""
    y, m = MONTHS[k]
    return f"{MONTH_NAMES[lang][m - 1]} {y}"


def say(k: int, lang: str) -> str:
    w = DEMO_TEXT[lang]["says"]
    got = installs(*MONTHS[k])
    if got["range"] == got["lock"]:
        return w["same"].format(date=date(k, lang))
    if got["bare"] < (2, 0, 0):
        return w["drift"].format(date=date(k, lang), v=shown(got["bare"]))
    return w["split"].format(date=date(k, lang), v=shown(got["bare"]), r=shown(got["range"]))


def frame(value: float, lang: str) -> tuple[str, str]:
    """The releases on a time line up to the day of the install, what each of four lines installs that day, and
    whether a package built for NumPy 1.x still runs beside the line without a version."""
    k = min(max(int(round(value)), 0), len(MONTHS) - 1)
    w = DEMO_TEXT[lang]
    y_, m_ = MONTHS[k]
    got = installs(y_, m_)
    parts = [_t(0, 14, w["title"].format(date=date(k, lang)), 12, 600)]
    # the time line: one month per step, a dot per release, the first release of each minor version named
    X0, X1 = 6.0, 322.0
    step = (X1 - X0) / (len(MONTHS) - 1)

    def x_of(day: str) -> float:
        y, m, d = (int(p) for p in day.split("-"))
        months = (y - FIRST[0]) * 12 + (m - FIRST[1]) + (d - 1) / 31
        return X0 + months * step

    ty = 52.0
    parts.append(_t(0, ty - 22, w["axis"], 9.5, 400, "var(--ea-ink-2)"))
    parts.append(_line(X0, ty, X1, ty, "var(--ea-line)", 1))
    now = f"{y_}-{m_:02d}-01"
    shown_minor = set()
    for v, day in sorted(RELEASES, key=lambda r: r[1]):
        if day < f"{FIRST[0]}-{FIRST[1]:02d}-01":
            continue
        out = day < now
        parts.append(f'<circle cx="{x_of(day):.1f}" cy="{ty:.1f}" r="2.4" '
                     f'style="fill:{"var(--ea-ink)" if out else "var(--ea-ink-3)"}"/>')
        if v[:2] not in shown_minor:
            shown_minor.add(v[:2])
            parts.append(_t(x_of(day), ty - 6, f"{v[0]}.{v[1]}", 8.5, 400, "var(--ea-ink-2)", "middle"))
    xn = X0 + k * step
    parts.append(_line(xn, ty - 4, xn, ty + 8, "var(--ea-ink)", 1.6))
    day_ = f"1 {date(k, lang)}"
    half = len(day_) * 2.7   # about half the width of the date at 9 px
    if xn - half < 0:
        parts.append(_t(0, ty + 19, day_, 9, 600, "var(--ea-ink)"))
    elif xn + half > 328:
        parts.append(_t(328, ty + 19, day_, 9, 600, "var(--ea-ink)", "end"))
    else:
        parts.append(_t(xn, ty + 19, day_, 9, 600, "var(--ea-ink)", "middle"))
    # what each line installs that day
    y = ty + 44
    lines = [("numpy", got["bare"], True), ("numpy>=1.24,<2", got["range"], True),
             (f"numpy=={shown(PINNED)}", got["pinned"], True), (w["lock"], got["lock"], False)]
    for text_, version, mono in lines:
        bad = version != got["lock"]
        ink = "var(--ea-loss-text)" if bad else "var(--ea-ink)"
        parts.append(_mono(0, y, text_, 10.5) if mono else _t(0, y, text_, 10.5, 400, "var(--ea-ink)"))
        parts.append(_t(196, y, "→", 10.5, 400, "var(--ea-ink-2)"))
        parts.append(f'<text x="212" y="{y:.1f}" style="font-family:var(--ea-font-code);font-size:11px;font-weight:600;'
                     f'fill:{ink}">{html.escape(shown(version))}</text>')
        if bad:
            parts.append(_t(328, y, w["differs"], 9.5, 600, "var(--ea-loss-text)", "end"))
        y += 22
    stops = got["bare"] >= (2, 0, 0)
    y += 8
    parts.append(_line(0, y - 14, 328, y - 14, "var(--ea-line)", 1))
    parts.append(_t(0, y, w["old"], 10.5, 400, "var(--ea-ink)"))
    parts.append(_t(0, y + 17, w["stops"] if stops else w["runs"], 11, 600,
                    "var(--ea-loss-text)" if stops else "var(--ea-ink)"))
    parts.append(_t(0, y + 38, w["foot"], 9.5, 400, "var(--ea-ink-2)"))
    alt = w["alt"].format(date=date(k, lang), v=shown(got["bare"]), r=shown(got["range"]),
                          old=w["stops"] if stops else w["runs"])
    return _svg(y + 46, alt, parts), say(k, lang)


# ======== the exercises ===================================================================================================

def _dots_left(got, text_) -> bool:
    """A `...` of the start code left in the box: an Ellipsis in the code, not three dots inside a string."""
    import ast
    try:
        tree = ast.parse(text_)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Constant) and node.value is Ellipsis for node in ast.walk(tree))


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. Can you replace it with your own code?",
                nl="Er staat nog een `...` in je code. Kun je de `...` vervangen door je eigen code?")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then begins with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


class _Listed(list):
    """A list of the lesson that the check shows by its name: C.APPS."""

    def __init__(self, items, name: str):
        super().__init__(items)
        self._name = name

    def __repr__(self) -> str:
        return self._name

    def __deepcopy__(self, memo):
        return list(self)

class _Named(dict):
    """A dictionary of the lesson that the check shows by its name: C.LAPTOP."""

    def __init__(self, items, name: str):
        super().__init__(items)
        self._name = name

    def __repr__(self) -> str:
        return self._name

    def __deepcopy__(self, memo):
        return dict(self)


LAPTOP_CASE = _Named(LAPTOP, "C.LAPTOP")
SERVER_CASE = _Named(SERVER, "C.SERVER")
RELEASED_CASE = _Listed(RELEASED, "C.RELEASED")

# exercise 1: differences
_D_HEAD = ("def differences(mine, theirs):\n"
           "    found = []\n"
           "    for name in mine:\n"
           "        if name not in theirs:\n"
           "            found.append(name)\n")
_D_END = ("            found.append(name)\n"
          "    return found\n")


def _d(condition: str) -> str:
    return _D_HEAD + f"        elif {condition}:\n" + _D_END


_D_START = _d("...")
_D = _d("mine[name] != theirs[name]")
_D_SAME = _d("mine[name] == theirs[name]")
_D_KEY = _d("mine[name] not in theirs")
_D_FIRST = _d("mine[name][0] != theirs[name][0]")

# exercise 2: resolve
_R_HEAD = ("def resolve(released, low, high):\n"
           "    chosen = None\n"
           "    for version in released:\n")
_R_END = ("            chosen = version\n"
          "    return chosen\n")


def _r(condition: str) -> str:
    return _R_HEAD + f"        if {condition}:\n" + _R_END


_R_START = _r("...")
_R = _r("version >= low and version < high")
_R_UPPER = _r("version >= low and version <= high")
_R_OPEN = _r("version > low and version < high")
_R_HIGH = _r("version < high")

EXERCISES = {
    "differences": FunctionBox(
        name="differences",
        function="differences",
        start=_D_START,
        solution=_D,
        visible=[(LAPTOP_CASE, SERVER_CASE)],
        hidden=[(dict(LAPTOP), dict(COLLEAGUE)), (dict(SERVER), dict(LAPTOP)), ({}, dict(LAPTOP)),
                ({"numpy": "2.1.0"}, {"numpy": "2.1.1"}), ({"numpy": "1.26.4", "joblib": "1.4.2"}, {"numpy": "1.26.4"})],
        first=[_DOTS],
        picture=differences_picture,
        mistakes=[
            Mistake("same", wrong=_D_SAME,
                    en="`{call}` gives {got}, and it should give {want}: your function lists the packages whose versions are the same on both computers. Which comparison is true when two versions differ?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie noemt de packages die op beide computers dezelfde versie hebben. Welke vergelijking is waar als twee versies verschillen?"),
            Mistake("key", wrong=_D_KEY,
                    en="`{call}` gives {got}, and it should give {want}: `mine[name]` is a version, such as `\"1.26.4\"`, and `in theirs` looks among the names, so a version is never there. Which version does `theirs` have for this `name`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `mine[name]` is een versie, zoals `\"1.26.4\"`, en `in theirs` zoekt tussen de namen, dus staat een versie er nooit tussen. Welke versie heeft `theirs` voor deze `name`?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with a condition that is true when the package has another version on the other computer.",
                      "nl": "Vervang de `...` door een voorwaarde die waar is als het package op de andere computer een andere versie heeft."},
            "arguments": {"en": "The check calls `{call}`, with two dictionaries of versions, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(mine, theirs)`?",
                          "nl": "De check roept `{call}` aan, met twee dictionaries met versies, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(mine, theirs)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function list each package with another version on the other computer?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Noemt je functie elk package met een andere versie op de andere computer?"},
            "almost": {"en": "Right for `C.LAPTOP` and `C.SERVER`. But `{call}` gives {got}, and it should give {want}. Does your condition compare the whole version, up to its last number?",
                       "nl": "Goed voor `C.LAPTOP` en `C.SERVER`. Maar `{call}` geeft {got}, en het moet {want} zijn. Vergelijkt je voorwaarde de hele versie, tot en met het laatste getal?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return found`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return found`?"},
            "correct": {"en": "Right in all {n} cases. Both packages your function found can make code stop on the server: numpy through NumPy 2, and joblib because scikit-learn needs it. Which step of the recipe prevents both?",
                        "nl": "Goed in alle {n} gevallen. Beide packages die je functie vond kunnen code op de server laten stoppen: numpy door NumPy 2, en joblib omdat scikit-learn joblib nodig heeft. Welke stap van het recept voorkomt allebei?"},
        },
    ),
    "resolve": FunctionBox(
        name="resolve",
        function="resolve",
        start=_R_START,
        solution=_R,
        visible=[(RELEASED_CASE, RANGE[0], RANGE[1])],
        hidden=[(list(RELEASED), (2, 0, 0), (3, 0, 0)), (list(RELEASED), (3, 0, 0), (4, 0, 0)),
                (list(RELEASED), (1, 24, 0), (1, 25, 0)),
                ([(1, 24, 0)], (1, 24, 0), (1, 25, 0)), ([(1, 9, 0), (1, 10, 0), (2, 0, 0)], (1, 0, 0), (2, 0, 0)),
                ([(0, 9, 0), (1, 0, 0)], (1, 5, 0), (2, 0, 0))],
        first=[_DOTS],
        picture=resolve_picture,
        mistakes=[
            Mistake("upper", wrong=_R_UPPER,
                    en="`{call}` gives {got}, and it should give {want}: your condition lets `high` itself in, and `<2` means below 2.0.0. Which comparison with `high` leaves `high` out?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je voorwaarde laat `high` zelf toe, en `<2` betekent onder 2.0.0. Welke vergelijking met `high` laat `high` zelf weg?"),
            Mistake("open", wrong=_R_OPEN,
                    en="`{call}` gives {got}, and it should give {want}: your condition leaves `low` itself out, and `>=1.24` allows 1.24.0. Which comparison with `low` lets `low` in?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je voorwaarde laat `low` zelf weg, en `>=1.24` staat 1.24.0 toe. Welke vergelijking met `low` laat `low` toe?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with a condition that is true when `version` is allowed: at least `low`, and below `high`.",
                      "nl": "Vervang de `...` door een voorwaarde die waar is als `version` toegestaan is: minstens `low`, en onder `high`."},
            "arguments": {"en": "The check calls `{call}`, with a list of versions and two versions, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(released, low, high)`?",
                          "nl": "De check roept `{call}` aan, met een lijst versies en twee versies, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(released, low, high)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function keep the newest version that is at least `low` and below `high`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Houdt je functie de nieuwste versie die minstens `low` is en onder `high` blijft?"},
            "almost": {"en": "Right for `C.RELEASED` from 1.24 to 2. But `{call}` gives {got}, and it should give {want}. Does your condition check both ends of the range?",
                       "nl": "Goed voor `C.RELEASED` van 1.24 tot 2. Maar `{call}` geeft {got}, en het moet {want} zijn. Controleert je voorwaarde beide kanten van het bereik?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return chosen`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return chosen`?"},
            "correct": {"en": "Right in all {n} cases. Which version does the same range give a year from now, and which does `numpy>=2.0,<3` give?",
                        "nl": "Goed in alle {n} gevallen. Welke versie geeft hetzelfde bereik over een jaar, en welke geeft `numpy>=2.0,<3`?"},
        },
    ),
}

# The lesson is done when these are correct. "lock" and "projects" are the choice questions after the demo.
REQUIRED = ["differences", "resolve", "lock", "projects"]

CASES = [
    ("differences", _D_START, "empty", None),
    ("differences", _D, "correct", None),
    ("differences", _d("mine[name] != ..."), "wrong", "dots-left"),
    ("differences", _D_SAME, "wrong", "same"),
    ("differences", _D_KEY, "wrong", "key"),
    ("differences", _D_FIRST, "almost", None),
    ("differences", _d("theirs[name] != mine[name]"), "correct", None),
    ("differences", _d("not mine[name] == theirs[name]"), "correct", None),
    ("differences", _D.replace("    return found\n", "    print(found)\n"), "wrong", "prints"),
    ("resolve", _R_START, "empty", None),
    ("resolve", _R, "correct", None),
    ("resolve", _r("version >= low and ..."), "wrong", "dots-left"),
    ("resolve", _R_UPPER, "wrong", "upper"),
    ("resolve", _R_OPEN, "wrong", "open"),
    ("resolve", _R_HIGH, "almost", None),
    ("resolve", _r("version >= low and version[0] == low[0]"), "almost", None),
    ("resolve", _r("low <= version < high"), "correct", None),
    ("resolve", _r("version < high and version >= low"), "correct", None),
    ("resolve", _R.replace("    return chosen\n", "    print(chosen)\n"), "wrong", "prints"),
]
