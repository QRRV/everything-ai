# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-model-versions: which model is this? A model file gets its version from a hash of its
bytes, so one name can never hide two models; a model registry keeps per model its versions, numbered in the order
they were registered, with the run that made each one. The learner writes `version`, the first 12 characters of the
SHA-256 hash of a file, and `register`, which gives a new number only to bytes the registry has not seen yet; the
demo changes one weight of a model by less and less, and shows what that does to the hash.

The model is a digits model of one layer, 784 pixels to 10 logits, made after torch.manual_seed(0) and not trained:
only its file matters here. Its files are written with PyTorch, so their bytes, and so their hashes, can differ
between versions of PyTorch; no hash is written into the lesson's text."""
import copy
import hashlib
import html
import io
import json

import torch

from everything_ai import check, paths, yours
from everything_ai.check import ClassBox, Code, FunctionBox, Mistake

LESSON = "sys-model-versions"
LANGS = ("en", "nl")

# ======== the course's model and its files ================================================================================
# WEIGHT is the weight that the worked example and the demo change. It lies between 1/64 and 1/32, where float32 keeps
# steps of about 0.0000000019: a change of 0.000000001 still changes it, a change of 0.0000000001 is too small for it.
SEED, PIXELS, DIGITS = 0, 784, 10
WEIGHT = (0, 1)
CHANGE = 0.000001                 # the change of the worked example, the guess and the file SECOND
DECIMALS = list(range(1, 11))     # the demo: a change of 0.1, 0.01, ..., 0.0000000001


def make_model() -> torch.nn.Linear:
    torch.manual_seed(SEED)
    return torch.nn.Linear(PIXELS, DIGITS)


def changed(model: torch.nn.Module, change: float) -> torch.nn.Module:
    """A copy of the model with `change` added to the weight WEIGHT."""
    out = copy.deepcopy(model)
    with torch.no_grad():
        out.weight[WEIGHT] += change
    return out


def saved_bytes(model: torch.nn.Module) -> bytes:
    """What torch.save writes for the model's state_dict to an open file: the folder inside the file is then `archive`."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


PARAMETERS = sum(p.numel() for p in make_model().parameters())   # 7,850


# ======== the course's solutions ==========================================================================================

def version(path) -> str:
    with open(path, "rb") as f:
        data = f.read()
    return hashlib.sha256(data).hexdigest()[:12]


def register(registry, name, file_version, run):
    if name not in registry:
        registry[name] = []
    for entry in registry[name]:
        if entry["version"] == file_version:
            return entry["number"]
    number = len(registry[name]) + 1
    registry[name].append({"number": number, "version": file_version, "run": run})
    return number


def model_path(name: str) -> str:
    """For the worked example: a file in the learner's work folder, work/models/<name>."""
    folder = paths.WORK / "models"
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder / name)


class _Path(Code):
    """A file of the exercise. The check shows it by its name, `C.FIRST`; Python opens the file at its path."""

    def __new__(cls, path: str, name: str):
        out = super().__new__(cls, path)
        out.name = name
        return out

    def __str__(self) -> str:
        return f"C.{self.name}"

    def __repr__(self) -> str:
        return f"C.{self.name}"

    def __deepcopy__(self, memo):
        _restore()   # the check copies the arguments before each call: the files are as the course made them
        return self


FILE_NAMES = {"FIRST": "first.pt", "COPY": "first-copy.pt", "SECOND": "second.pt", "HELLO": "hello.txt"}
_FOLDER = paths.WORK / "models" / "check"
# The four files of exercise 1: the model, a copy of its file under another name, the model with one weight changed
# by CHANGE, and a text file of six bytes. Both models are saved through saved_bytes, so the folder inside the zip
# archive has the same name in both files, and only the changed weight tells them apart.
_CONTENT = {"FIRST": saved_bytes(make_model()), "SECOND": saved_bytes(changed(make_model(), CHANGE)), "HELLO": b"hello\n"}
_CONTENT["COPY"] = _CONTENT["FIRST"]


def _restore() -> None:
    """Write each file of exercise 1 again when it is missing or changed, as by a function that opened it to write."""
    _FOLDER.mkdir(parents=True, exist_ok=True)
    for key, data in _CONTENT.items():
        path = _FOLDER / FILE_NAMES[key]
        try:
            same = path.read_bytes() == data
        except OSError:
            same = False
        if not same:
            path.unlink(missing_ok=True)
            path.write_bytes(data)


_restore()
FIRST, COPY, SECOND, HELLO = (_Path(str(_FOLDER / FILE_NAMES[k]), k) for k in ("FIRST", "COPY", "SECOND", "HELLO"))


# ======== drawing =========================================================================================================

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem;max-width:var(--ea-measure)"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem;max-width:var(--ea-measure)"
_SVG = "width:100%;max-width:330px;height:auto;font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums"


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma and a thousands dot in Dutch, and a real minus sign."""
    out = f"{x:,.{digits}f}"
    out = out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out
    return out.replace("-", "−")


def change_text(k: int, lang: str) -> str:
    """A change of 10 to the power -k, written out: 0.000001 for k = 6."""
    return number(10.0 ** -k, k, lang)


def _marked(text_: str) -> str:
    """Escaped text in which `name` is set as code and *word* in italics."""
    out = []
    for k, part in enumerate(html.escape(text_).split("`")):
        if k % 2:
            out.append(f'<code style="white-space:nowrap">{part}</code>')
        else:
            out.append("".join(f"<i>{p}</i>" if j % 2 else p for j, p in enumerate(part.split("*"))))
    return "".join(out)


def _note(text_: str, style: str = _UI) -> str:
    return f'<div style="{style}">{_marked(text_)}</div>'


def _t(x: float, y: float, text_: str, size: float = 12.5, weight: int = 400, fill: str = "var(--ea-ink)",
       anchor: str = "start") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" style="font-size:{size}px;font-weight:{weight};'
            f'fill:{fill}">{html.escape(text_)}</text>')


def _mono(x: float, y: float, text_: str, size: float = 10.5, fill: str = "var(--ea-ink)", anchor: str = "start",
          weight: int = 400) -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" style="font-family:var(--ea-font-code);'
            f'font-variant-ligatures:none;font-size:{size}px;font-weight:{weight};fill:{fill}">{html.escape(text_)}</text>')


def _spans(x: float, y: float, chars: list, size: float = 10.5) -> str:
    """Code text in which each piece has its own ink and weight: chars is a list of (text, ink, weight)."""
    inner = "".join(f'<tspan style="fill:{ink};font-weight:{weight}">{html.escape(c)}</tspan>' for c, ink, weight in chars)
    return (f'<text x="{x:.1f}" y="{y:.1f}" style="font-family:var(--ea-font-code);font-variant-ligatures:none;'
            f'font-size:{size}px">{inner}</text>')


def _line(x1: float, y1: float, x2: float, y2: float, ink: str = "var(--ea-line)", width: float = 1.0) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" style="stroke:{ink};stroke-width:{width}"/>'


def _rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "") -> str:
    edge = f";stroke:{stroke};stroke-width:1" if stroke else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" style="fill:{fill}{edge}"/>'


def _box(x: float, y: float, w: float, h: float, stroke: str = "var(--ea-ink-3)", width: float = 1.2) -> str:
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="5" '
            f'style="fill:var(--ea-page);stroke:{stroke};stroke-width:{width}"/>')


def _arrow(x1: float, y1: float, x2: float, y2: float, ink: str = "var(--ea-ink-2)", width: float = 1.3) -> str:
    """A line from (x1, y1) to (x2, y2) with a small head at its end."""
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    h1 = (x2 - 6 * math.cos(a - 0.45), y2 - 6 * math.sin(a - 0.45))
    h2 = (x2 - 6 * math.cos(a + 0.45), y2 - 6 * math.sin(a + 0.45))
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" style="stroke:{ink};stroke-width:{width}"/>'
            f'<path d="M{h1[0]:.1f},{h1[1]:.1f} L{x2:.1f},{y2:.1f} L{h2[0]:.1f},{h2[1]:.1f}" '
            f'style="fill:none;stroke:{ink};stroke-width:{width}"/>')


def _svg(height: float, alt: str, parts: list) -> str:
    return (f'<svg viewBox="0 0 330 {height:.0f}" role="img" aria-label="{html.escape(alt)}" style="{_SVG}">'
            f'{"".join(parts)}</svg>')


def _short(text_: str, n: int) -> str:
    return text_ if len(text_) <= n else text_[:n - 1] + "…"


# ======== the picture of the idea =========================================================================================

IDEA_TEXT = {
    "en": {"title": "From a file to a version in the registry",
           "file": "the file", "hash": "SHA-256 of its bytes", "version": "version: the first 12 characters",
           "changed": "one weight + 0.000001",
           "registry": "registry: digits", "number": "number", "column": "version", "run": "run", "alias": "alias production → 1",
           "alt": "Two model files, first.pt and second.pt, which differ in one weight. The SHA-256 hash of each file's bytes is 64 characters long, and the first 12 are its version: the two versions have nothing in common. The registry of the model digits holds both, as number 1 and number 2, each with its version and the run that made it; the alias production points to number 1."},
    "nl": {"title": "Van bestand naar versie in de registry",
           "file": "het bestand", "hash": "SHA-256 van de bytes", "version": "versie: de eerste 12 tekens",
           "changed": "één weight + 0,000001",
           "registry": "registry: digits", "number": "nummer", "column": "versie", "run": "run", "alias": "alias production → 1",
           "alt": "Twee modelbestanden, first.pt en second.pt, met één weight verschil. De SHA-256-hash van de bytes van elk bestand is 64 tekens lang, en de eerste 12 zijn de versie: de twee versies hebben niets gemeen. De registry van het model digits bevat beide, als nummer 1 en nummer 2, elk met de versie en de run die het model maakte; de alias production wijst naar nummer 1."},
}
IDEA_RUNS = ("first-try", "second-try")


def idea_picture(lang: str) -> str:
    """Two files, their hashes and versions, and the registry that holds both."""
    w = IDEA_TEXT[lang]
    _restore()
    parts = [_t(0, 14, w["title"], 12.5, 600)]
    y = 32.0
    rows = []
    for k, (path, name) in enumerate(((FIRST, "first.pt"), (SECOND, "second.pt"))):
        with open(path, "rb") as f:
            full = sha256(f.read())
        rows.append(full[:12])
        parts.append(_box(0, y, 330, 54))
        parts.append(_mono(8, y + 16, name, 10.5, weight=600))
        parts.append(_t(322, y + 16, w["file"] if k == 0 else w["changed"], 9, 400, "var(--ea-ink-2)", "end"))
        parts.append(_t(8, y + 31, w["hash"], 9, 400, "var(--ea-ink-2)"))
        parts.append(_spans(8, y + 45, [(full[:12], "var(--ea-ink)", 700), (full[12:40] + "…", "var(--ea-ink-3)", 400)], 10))
        y += 62
    parts.append(_t(0, y + 4, w["version"], 9, 400, "var(--ea-ink-2)"))
    y += 18
    parts.append(_arrow(40, y - 8, 40, y + 4))
    y += 10
    parts.append(_box(0, y, 330, 70))
    parts.append(_t(8, y + 16, w["registry"], 10, 600))
    parts.append(_t(8, y + 32, w["number"], 9, 400, "var(--ea-ink-2)"))
    parts.append(_t(70, y + 32, w["column"], 9, 400, "var(--ea-ink-2)"))
    parts.append(_t(190, y + 32, w["run"], 9, 400, "var(--ea-ink-2)"))
    for k, (v, run) in enumerate(zip(rows, IDEA_RUNS)):
        parts.append(_mono(8, y + 47 + 14 * k, str(k + 1), 10))
        parts.append(_mono(70, y + 47 + 14 * k, v, 10, weight=600))
        parts.append(_mono(190, y + 47 + 14 * k, run, 10, "var(--ea-ink-2)"))
    parts.append(_t(322, y + 16, w["alias"], 9, 600, "var(--ea-ink)", "end"))
    height = y + 78
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(w["alt"])}" style="margin:0.25rem 0 0.75rem;'
            f'max-width:min(100%,330px)"><svg viewBox="0 0 330 {height:.0f}" aria-hidden="true" style="{_SVG}">'
            f'{"".join(parts)}</svg></figure>')


# ======== the pictures of the learner's work ==============================================================================

PICTURE_SECONDS = 2   # what the learner's function gets for all the calls of one picture together
_STOPPED = object()   # what _call gives for a function that stopped with an error


def _fn(seen, name: str):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(name)
    return fn if callable(fn) else None


def _call(make_):
    """make_() within a time limit; _STOPPED when it stops with an error or runs out of time."""
    try:
        return check.with_time_limit(make_, PICTURE_SECONDS)
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _shown(v, n: int = 20) -> str:
    return _short(v if isinstance(v, str) else repr(v), n)


_YOURS_TEXT = {
    "en": {
        "version": {"title": "Your versions of three files",
                    "intro": "The picture calls your `version` on three files of the exercise: `C.FIRST`; `C.COPY`, a copy of its file under another name; and `C.SECOND`, the same model with one weight changed by 0.000001.",
                    "file": "file", "yours": "your version", "course": "the course's",
                    "fail": "Your `version` stops on these files, so the picture has no version to show yet. The check above says why.",
                    "right": "Same bytes, same version: the copy gets the version of `first.pt`, under another name. One weight changed by 0.000001, and the version is another one.",
                    "names": "Your versions follow the names of the files: the copy gets another version, though its bytes are the same. What holds the bytes?",
                    "wrong": "Your versions are not those of the course. Are they the first 12 characters of the SHA-256 hash of each file's bytes?",
                    "alt": "Three files, first.pt, first-copy.pt and second.pt, each with the version your function gives and the version of the course."},
        "register": {"title": "Your register on three files",
                     "intro": "The picture registers three files as the model `digits`, all from the run `first-try`: `first.pt`, its copy, and `second.pt`, with one weight changed. Your `register` gives a number for each, and fills the registry.",
                     "call": "call", "gave": "gave", "want": "should be", "number": "number", "column": "version", "run": "run",
                     "fail": "Your `register` stops on these calls, so the picture has no registry to show yet. The check above says why.",
                     "right": "The copy has the bytes of `first.pt`, so it gets number 1 again; `second.pt` is another model, so it gets number 2, though its run has the same name.",
                     "wrong": "Your registry differs from the course's in the rows marked. Which files hold the same bytes, and which hold another model?",
                     "alt": "Three calls of your register, each with the number it gave and the number it should give, and the rows of the registry your function filled."},
    },
    "nl": {
        "version": {"title": "Je versies van drie bestanden",
                    "intro": "De afbeelding roept je `version` aan op drie bestanden uit de oefening: `C.FIRST`; `C.COPY`, een kopie van dat bestand onder een andere naam; en `C.SECOND`, hetzelfde model met één weight 0,000001 veranderd.",
                    "file": "bestand", "yours": "jouw versie", "course": "van de course",
                    "fail": "Je `version` stopt bij deze bestanden, dus valt er nog geen versie te laten zien. De check hierboven zegt waarom.",
                    "right": "Dezelfde bytes, dezelfde versie: de kopie krijgt de versie van `first.pt`, onder een andere naam. Eén weight 0,000001 veranderd, en de versie is een andere.",
                    "names": "Je versies volgen de namen van de bestanden: de kopie krijgt een andere versie, terwijl de bytes hetzelfde zijn. Waarin staan de bytes?",
                    "wrong": "Je versies verschillen van de versies van de course. Zijn het de eerste 12 tekens van de SHA-256-hash van de bytes van elk bestand?",
                    "alt": "Drie bestanden, first.pt, first-copy.pt en second.pt, elk met de versie die je functie geeft en de versie van de course."},
        "register": {"title": "Je register bij drie bestanden",
                     "intro": "De afbeelding registreert drie bestanden als het model `digits`, alle drie uit de run `first-try`: `first.pt`, de kopie, en `second.pt`, met één weight veranderd. Je `register` geeft voor elk bestand een nummer, en vult de registry.",
                     "call": "aanroep", "gave": "gaf", "want": "moet zijn", "number": "nummer", "column": "versie", "run": "run",
                     "fail": "Je `register` stopt bij deze aanroepen, dus valt er nog geen registry te laten zien. De check hierboven zegt waarom.",
                     "right": "De kopie heeft de bytes van `first.pt`, dus krijgt de kopie weer nummer 1; `second.pt` is een ander model, dus krijgt `second.pt` nummer 2, ook al heeft de run dezelfde naam.",
                     "wrong": "Je registry verschilt in de gemarkeerde rijen van de registry van de course. Welke bestanden bevatten dezelfde bytes, en welke een ander model?",
                     "alt": "Drie aanroepen van je register, elk met het nummer dat je functie gaf en het nummer dat het moet zijn, en de rijen van de registry die je functie vulde."},
    },
}

_SHOWN_FILES = ("FIRST", "COPY", "SECOND")


def version_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["version"]
    fn = _fn(seen, "version")
    paths_ = {"FIRST": FIRST, "COPY": COPY, "SECOND": SECOND}
    _restore()
    got = _call(lambda: [fn(paths_[k]) for k in _SHOWN_FILES]) if fn is not None else _STOPPED
    if got is _STOPPED:
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["fail"], _LOSS_UI), alt=w["alt"])
    want = [version(paths_[k]) for k in _SHOWN_FILES]
    parts = [_t(0, 12, w["file"], 9, 400, "var(--ea-ink-2)"), _t(112, 12, w["yours"], 9, 400, "var(--ea-ink-2)"),
             _t(232, 12, w["course"], 9, 400, "var(--ea-ink-2)")]
    for k, key in enumerate(_SHOWN_FILES):
        y = 32 + 20 * k
        ok = got[k] == want[k]
        parts.append(_line(0, y - 14, 330, y - 14, "var(--ea-line)", 0.8))
        parts.append(_mono(0, y, FILE_NAMES[key], 10))
        parts.append(_mono(112, y, _shown(got[k], 16), 10, "var(--ea-ink)" if ok else "var(--ea-loss-text)", weight=600))
        parts.append(_mono(232, y, want[k], 10, "var(--ea-ink-2)"))
    svg = _svg(32 + 20 * len(_SHOWN_FILES) - 6, w["alt"], parts)
    if got == want:
        note = _note(w["right"])
    elif isinstance(got[0], str) and got[0] != got[1] and want[0] == want[1]:
        note = _note(w["names"], _LOSS_UI)
    else:
        note = _note(w["wrong"], _LOSS_UI)
    return yours.figure(w["title"], _note(w["intro"]) + svg + note, alt=w["alt"])


def _history() -> list:
    """The three calls of the picture: the model digits, from the run first-try, for FIRST, COPY and SECOND."""
    return [("digits", version(p), "first-try") for p in (FIRST, COPY, SECOND)]


def register_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["register"]
    fn = _fn(seen, "register")
    _restore()
    calls = _history()

    def run(f):
        reg = {}
        return [f(reg, *c) for c in calls], reg

    got = _call(lambda: run(fn)) if fn is not None else _STOPPED
    if got is _STOPPED:
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["fail"], _LOSS_UI), alt=w["alt"])
    numbers, reg = got
    want_numbers, want_reg = run(register)
    parts = [_t(0, 12, w["call"], 9, 400, "var(--ea-ink-2)"), _t(150, 12, w["gave"], 9, 400, "var(--ea-ink-2)"),
             _t(232, 12, w["want"], 9, 400, "var(--ea-ink-2)")]
    for k, (name, v, run_) in enumerate(calls):
        y = 30 + 18 * k
        ok = numbers[k] == want_numbers[k]
        parts.append(_mono(0, y, f"{FILE_NAMES[_SHOWN_FILES[k]]}  {v[:6]}…", 10))
        parts.append(_mono(150, y, _shown(numbers[k], 10), 10, "var(--ea-ink)" if ok else "var(--ea-loss-text)", weight=600))
        parts.append(_mono(232, y, str(want_numbers[k]), 10, "var(--ea-ink-2)"))
    y = 30 + 18 * len(calls) + 12
    parts.append(_line(0, y - 10, 330, y - 10, "var(--ea-line)", 0.8))
    parts.append(_t(0, y + 4, "registry[\"digits\"]", 9, 600, "var(--ea-ink-2)"))
    rows = reg.get("digits") if isinstance(reg, dict) else None
    rows = rows if isinstance(rows, list) else []
    want_rows = want_reg["digits"]
    parts.append(_t(0, y + 20, w["number"], 9, 400, "var(--ea-ink-2)"))
    parts.append(_t(70, y + 20, w["column"], 9, 400, "var(--ea-ink-2)"))
    parts.append(_t(190, y + 20, w["run"], 9, 400, "var(--ea-ink-2)"))
    shown = rows[:5]
    for k, row in enumerate(shown):
        yy = y + 36 + 15 * k
        ok = k < len(want_rows) and row == want_rows[k]
        ink = "var(--ea-ink)" if ok else "var(--ea-loss-text)"
        if isinstance(row, dict):
            parts.append(_mono(0, yy, _shown(row.get("number"), 6), 10, ink))
            parts.append(_mono(70, yy, _shown(row.get("version"), 14), 10, ink, weight=600))
            parts.append(_mono(190, yy, _shown(row.get("run"), 18), 10, ink))
        else:
            parts.append(_mono(0, yy, _shown(row, 40), 10, ink))
    height = y + 36 + 15 * max(len(shown), 1)
    svg = _svg(height, w["alt"], parts)
    right = numbers == want_numbers and rows == want_rows
    note = _note(w["right"]) if right else _note(w["wrong"], _LOSS_UI)
    return yours.figure(w["title"], _note(w["intro"]) + svg + note, alt=w["alt"])


# ======== the demo: one weight, less and less ===========================================================================

def _demo_data() -> dict:
    model = make_model()
    before = model.weight[WEIGHT].item()
    base = sha256(saved_bytes(model))
    out = {"before": before, "base": base, "after": {}}
    for k in DECIMALS:
        m = changed(model, 10.0 ** -k)
        out["after"][k] = (m.weight[WEIGHT].item(), sha256(saved_bytes(m)))
    return out


def demo_data() -> dict:
    return check.kept((LESSON, "demo"), _demo_data)


def differing(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


DEMO_TEXT = {
    "en": {"weight": "weight [0, 1]", "change": "change", "before": "hash before", "after": "hash after",
           "differ": "{n} of 64 characters differ",
           "alt": "One weight of the model, before and after a change of {change}, and the SHA-256 hash of the saved file before and after, with the {n} characters that differ in bold.",
           "says": {
               "differ": "A change of **{change}** to one of the {params} weights and biases: the weight goes from {before} to {after}, and **{n}** of the 64 characters of the hash differ.",
               "same": "A change of **{change}** is too small for float32 to add to this weight: the weight stays {before}, the file is byte for byte the same, and so is its hash: **0** of the 64 characters differ.",
           }},
    "nl": {"weight": "weight [0, 1]", "change": "verandering", "before": "hash eerst", "after": "hash daarna",
           "differ": "{n} van de 64 tekens verschillen",
           "alt": "Eén weight van het model, voor en na een verandering van {change}, en de SHA-256-hash van het opgeslagen bestand voor en na, met de {n} tekens die verschillen vet.",
           "says": {
               "differ": "Een verandering van **{change}** in een van de {params} weights en biases: de weight gaat van {before} naar {after}, en **{n}** van de 64 tekens van de hash verschillen.",
               "same": "Een verandering van **{change}** is te klein om in float32 bij deze weight op te tellen: de weight blijft {before}, het bestand is byte voor byte hetzelfde, en de hash dus ook: **0** van de 64 tekens verschillen.",
           }},
}


def _weight(v: float, lang: str) -> str:
    return number(v, 12, lang)


def say(k: int, lang: str) -> str:
    d = demo_data()
    after, h = d["after"][k]
    n = differing(d["base"], h)
    w = DEMO_TEXT[lang]["says"]["differ" if n else "same"]
    return w.format(change=change_text(k, lang), params=number(PARAMETERS, 0, lang), before=_weight(d["before"], lang),
                    after=_weight(after, lang), n=n)


def frame(value: float, lang: str) -> tuple[str, str]:
    """The weight before and after a change of 10 to the power -k, and the two hashes, the characters that differ bold."""
    k = min(max(int(round(value)), DECIMALS[0]), DECIMALS[-1])
    d = demo_data()
    w = DEMO_TEXT[lang]
    after, h = d["after"][k]
    n = differing(d["base"], h)
    parts = [_t(0, 14, w["weight"], 10, 600),
             _mono(0, 32, _weight(d["before"], lang), 11),
             _mono(118, 32, "→", 11, "var(--ea-ink-2)"),
             _mono(140, 32, _weight(after, lang), 11, weight=600 if after != d["before"] else 400),
             _t(0, 50, f'{w["change"]}: {change_text(k, lang)}', 9.5, 400, "var(--ea-ink-2)")]
    y = 74
    parts.append(_t(0, y, w["before"], 9.5, 400, "var(--ea-ink-2)"))
    parts.append(_t(176, y, w["after"], 9.5, 400, "var(--ea-ink-2)"))
    for row in range(4):
        a, b = d["base"][16 * row:16 * row + 16], h[16 * row:16 * row + 16]
        yy = y + 18 + 15 * row
        parts.append(_mono(0, yy, a, 11, "var(--ea-ink-2)"))
        parts.append(_spans(176, yy, [(c, "var(--ea-ink)" if c != o else "var(--ea-ink-3)", 700 if c != o else 400)
                                      for c, o in zip(b, a)], 11))
    y += 18 + 15 * 4 + 6
    parts.append(_t(0, y, w["differ"].format(n=n), 10, 600))
    alt = w["alt"].format(change=change_text(k, lang), n=n)
    return _svg(y + 8, alt, parts), say(k, lang)


# ======== the exercises ===================================================================================================

def _dots_left(got, text_) -> bool:
    import ast
    try:
        tree = ast.parse(text_)
    except SyntaxError:
        return "..." in text_
    return any(isinstance(n, ast.Constant) and n.value is Ellipsis for n in ast.walk(tree))


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. Can you replace it with your own code?",
                nl="Er staat nog een `...` in je code. Kun je de `...` vervangen door je eigen code?")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then begins with the course's lines again.",
                "nl": "Alle code in het vak is weg. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# version: the box holds `import hashlib` and the function, which reads the file's bytes into `data`.
_V_HEAD = ("import hashlib\n"
           "\n\n"
           "def version(path):\n"
           "    with open(path, \"rb\") as f:\n"
           "        data = f.read()\n")


def _v(line: str) -> str:
    return _V_HEAD + "    return " + line + "\n"


_V_START = _v("...")
_V = _v("hashlib.sha256(data).hexdigest()[:12]")
_V_NAME = _v("hashlib.sha256(path.encode()).hexdigest()[:12]")
_V_FULL = _v("hashlib.sha256(data).hexdigest()")
_V_MD5 = _v("hashlib.md5(data).hexdigest()[:12]")
_V_BYTES = _v("hashlib.sha256(data).digest()[:12]")
_V_BUILTIN = _v("hash(data)")
_V_LAST = _v("hashlib.sha256(data).hexdigest()[-12:]")

# register: the box holds the function; each scenario fills a registry and ends with what it gave.
_R_HEAD = ("def register(registry, name, file_version, run):\n"
           "    if name not in registry:\n"
           "        registry[name] = []\n"
           "    for entry in registry[name]:\n")
_R_END = ("            return entry[\"number\"]\n"
          "    number = len(registry[name]) + 1\n"
          "    registry[name].append({\"number\": number, \"version\": file_version, \"run\": run})\n"
          "    return number\n")


def _r(condition: str) -> str:
    return _R_HEAD + f"        if {condition}:\n" + _R_END


_R_START = _r("...")
_R = _r("entry[\"version\"] == file_version")
_R_RUN = _r("entry[\"run\"] == run")
_R_NEVER = _r("entry == file_version")
_R_NAME = _r("entry[\"version\"] == name")
_R_AND_RUN = _r("entry[\"version\"] == file_version and entry[\"run\"] == run")

_R_SCENARIOS = [
    'registry = {}\n'
    'a = register(registry, "digits", "e3551ea36113", "first-try")\n'
    'b = register(registry, "digits", "3988980e633e", "first-try")\n'
    'c = register(registry, "digits", "e3551ea36113", "first-try")\n'
    'a, b, c',
    'registry = {}\n'
    'register(registry, "digits", "e3551ea36113", "first-try")\n'
    'register(registry, "digits", "e3551ea36113", "first-try")\n'
    'registry',
    'registry = {"digits": [{"number": 1, "version": "e3551ea36113", "run": "first-try"}]}\n'
    'register(registry, "letters", "5891b5b522d5", "abc")\n'
    'registry',
    'registry = {"digits": [{"number": 1, "version": "e3551ea36113", "run": "first-try"},\n'
    '                       {"number": 2, "version": "3988980e633e", "run": "second-try"}]}\n'
    'register(registry, "digits", "e3551ea36113", "third-try"), len(registry["digits"])',
    'registry = {"digits": [{"number": 1, "version": "e3551ea36113", "run": "first-try"},\n'
    '                       {"number": 2, "version": "3988980e633e", "run": "second-try"}]}\n'
    'register(registry, "digits", "3988980e633e", "second-try"), len(registry["digits"])',
]

_F_EN = "With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives {got}, and it should give {want}"
_F_NL = "Met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel {got}, en het moet {want} zijn"

EXERCISES = {
    "version": FunctionBox(
        name="version",
        function="version",
        start=_V_START,
        solution=_V,
        visible=[(FIRST,)],
        hidden=[(COPY,), (SECOND,), (HELLO,)],
        first=[_DOTS],
        picture=version_picture,
        mistakes=[
            Mistake("name", wrong=_V_NAME,
                    en="`{call}` gives {got}, and it should give {want}: your version is the hash of the text of `path`, the name of the file, not of its bytes. A file that is saved over keeps its name, and so its version. Which name holds the bytes of the file?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je versie is de hash van de tekst van `path`, de naam van het bestand, niet van de bytes. Als je een bestand overschrijft, houdt het bestand de naam, en dus de versie. Welke naam bevat de bytes van het bestand?"),
            Mistake("full", wrong=_V_FULL,
                    en="`{call}` gives {got}, and it should give {want}: your version has all 64 characters of the hash. The course's versions are the first 12, as `[:12]` keeps them. Can you keep only those?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je versie heeft alle 64 tekens van de hash. De versies van de course zijn de eerste 12, zoals `[:12]` doet. Kun je alleen deze tekens houden?"),
            Mistake("md5", wrong=_V_MD5,
                    en="`{call}` gives {got}, and it should give {want}: your version comes from MD5, the hash DVC uses. The course's versions come from SHA-256. Which function of `hashlib` gives that one?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je versie komt uit MD5, de hash van DVC. De versies van de course komen uit SHA-256. Welke functie van `hashlib` geeft deze hash?"),
            Mistake("bytes", wrong=_V_BYTES,
                    en="`{call}` gives {got}, and it should give {want}: `.digest()` gives the hash as bytes. Which method gives it as text, in hex?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `.digest()` geeft de hash als bytes. Welke methode geeft de hash als tekst, in hex?"),
            Mistake("builtin", wrong=_V_BUILTIN,
                    en="`{call}` gives {got}, and it should give {want}: `hash` is Python's own hash, a number, and for bytes and text it differs every time Python starts. A version has to stay the same on every computer. Which module gives SHA-256?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `hash` is de eigen hash van Python, een getal, en bij bytes en tekst verschilt dat getal elke keer dat Python start. Een versie moet op elke computer hetzelfde blijven. Welke module geeft SHA-256?"),
            Mistake("closed", error=lambda e: isinstance(e, ValueError) and "closed file" in str(e),
                    en="With `{call}`, your function reads from `f` after the `with` block, and there the file is closed again. Its bytes are already in `data`: can you use those?",
                    nl="Met `{call}` leest je functie uit `f` na het `with`-blok, en daar is het bestand alweer dicht. De bytes staan al in `data`: kun je deze bytes gebruiken?"),
            Mistake("last", wrong=_V_LAST,
                    en="`{call}` gives {got}, and it should give {want}: your version has the last 12 characters of the hash. The course's versions are the first 12. Which slice gives those?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je versie heeft de laatste 12 tekens van de hash. De versies van de course zijn de eerste 12. Welke slice geeft deze tekens?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with what the function gives back: the first 12 characters of the SHA-256 hash of `data`, in hex.",
                      "nl": "Vervang de `...` door wat de functie teruggeeft: de eerste 12 tekens van de SHA-256-hash van `data`, in hex."},
            "arguments": {"en": "The check calls `{call}`, with the path of one file, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(path)`?",
                          "nl": "De check roept `{call}` aan, met het pad van één bestand, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(path)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is your version the first 12 characters of the SHA-256 hash of the bytes in `data`, in hex?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Is je versie de eerste 12 tekens van de SHA-256-hash van de bytes in `data`, in hex?"},
            "almost": {"en": "Right for `C.FIRST`. But `{call}` gives {got}, and it should give {want}. Does your version come from the bytes of the file at `path`, whatever file it is?",
                       "nl": "Goed voor `C.FIRST`. Maar `{call}` geeft {got}, en het moet {want} zijn. Komt je versie uit de bytes van het bestand op `path`, welk bestand het ook is?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it end with `return` and the version?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie met `return` en de versie?"},
            "correct": {"en": "Right in all {n} cases. The copy of `first.pt` gets the same version under another name, and one weight changed by 0.000001 gives another version. What would the version of a copy of `first.pt` be on a colleague's computer?",
                        "nl": "Goed in alle {n} gevallen. De kopie van `first.pt` krijgt dezelfde versie onder een andere naam, en één weight 0,000001 veranderd geeft een andere versie. Wat zou de versie van een kopie van `first.pt` op de computer van een collega zijn?"},
        },
    ),
    "register": ClassBox(
        name="register",
        start=_R_START,
        solution=_R,
        scenarios=_R_SCENARIOS,
        first=[_DOTS],
        picture=register_picture,
        mistakes=[
            Mistake("by-run", wrong=_R_RUN,
                    en=_F_EN + ": your function finds a version by the name of its run. A model trained again under the same run name then gets the number of the old model, and the new one never gets into the registry. What tells two models apart for sure?",
                    nl=_F_NL + ": je functie zoekt een versie op de naam van de run. Een opnieuw getraind model met dezelfde runnaam krijgt dan het nummer van het oude model, en het nieuwe model komt nooit in de registry. Wat onderscheidt twee modellen zeker?"),
            Mistake("and-run", wrong=_R_AND_RUN,
                    en=_F_EN + ": your comparison also asks for the same run. The same bytes are the same model, whatever run registers them again. What does the version alone tell you?",
                    nl=_F_NL + ": je vergelijking vraagt ook om dezelfde run. Dezelfde bytes zijn hetzelfde model, welke run de bytes ook opnieuw registreert. Wat zegt de versie alleen al?"),
            Mistake("never", wrong=_R_NEVER,
                    en=_F_EN + ": the same file gets a new number every time it is registered, so your comparison is never `True` for it. `entry` is a dict with a number, a version and a run: which part of it do you compare, and with what?",
                    nl=_F_NL + ": hetzelfde bestand krijgt bij elke registratie een nieuw nummer, dus is je vergelijking daarbij nooit `True`. `entry` is een dict met een nummer, een versie en een run: welk deel vergelijk je, en waarmee?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with a comparison that is `True` when `entry` holds the version of the file you register.",
                      "nl": "Vervang de `...` door een vergelijking die `True` is als `entry` de versie bevat van het bestand dat je registreert."},
            "wrong": {"en": _F_EN + ". Does your comparison find an entry with the same version as `file_version`, so that the same bytes keep their number?",
                      "nl": _F_NL + ". Vindt je vergelijking een entry met dezelfde versie als `file_version`, zodat dezelfde bytes het nummer houden?"},
            "almost": {"en": "Right for the first lines. But " + _F_EN[0].lower() + _F_EN[1:] + ". Does your comparison look at the version of each entry, for any model and any registry?",
                       "nl": "Goed voor de eerste regels. Maar " + _F_NL[0].lower() + _F_NL[1:] + ". Kijkt je vergelijking naar de versie van elke entry, bij elk model en elke registry?"},
            "correct": {"en": "Right in all {n} cases. The same bytes keep their number, and every new model gets the next one, whatever its run is called. Where would you write down which number is in production?",
                        "nl": "Goed in alle {n} gevallen. Dezelfde bytes houden het nummer, en elk nieuw model krijgt het volgende, hoe de run ook heet. Waar zou je opschrijven welk nummer in productie is?"},
        },
    ),
}

# The lesson is done when these are correct. "pointer" and "alias" are the choice questions at the end.
REQUIRED = ["version", "register", "pointer", "alias"]

CASES = [
    ("version", _V_START, "empty", None),
    ("version", _V, "correct", None),
    ("version", _v("hashlib.sha256(data).hexdigest()[0:12]"), "correct", None),
    ("version", _v("hashlib.new(\"sha256\", data).hexdigest()[:12]"), "correct", None),
    ("version", _v("hashlib.sha256(...).hexdigest()[:12]"), "wrong", "dots-left"),
    ("version", _V_NAME, "wrong", "name"),
    ("version", _V_FULL, "wrong", "full"),
    ("version", _V_MD5, "wrong", "md5"),
    ("version", _V_BYTES, "wrong", "bytes"),
    ("version", _V_BUILTIN, "wrong", "builtin"),
    ("version", _V_LAST, "wrong", "last"),
    ("version", _v("hashlib.sha1(data).hexdigest()[:12]"), "wrong", None),
    ("version", _v("\"" + "0" * 12 + "\""), "wrong", None),
    ("version", _v("\"" + version(FIRST) + "\""), "almost", None),
    ("version", _V.replace("    return ", "    print(").replace("[:12]\n", "[:12])\n"), "wrong", "prints"),
    ("register", _R_START, "empty", None),
    ("register", _R, "correct", None),
    ("register", _r("file_version == entry[\"version\"]"), "correct", None),
    ("register", _r("entry[\"version\"] == ..."), "wrong", "dots-left"),
    ("register", _R_RUN, "wrong", "by-run"),
    ("register", _R_NEVER, "wrong", "never"),
    ("register", _r("entry[\"number\"] == file_version"), "wrong", "never"),
    ("register", _R_NAME, "wrong", "never"),
    ("register", _R_AND_RUN, "wrong", "and-run"),
    ("register", _r("entry[\"version\"] == file_version and entry[\"number\"] == 1"), "almost", None),
    ("version", _V.replace("    return hashlib.sha256(data)", "    return hashlib.sha256(f.read())"), "error", "closed"),
]
