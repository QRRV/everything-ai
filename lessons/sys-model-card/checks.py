# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-model-card: the course's digit reader (from the lesson torch-pretrained) measured per group
of writers and per condition, as a model card should report it (Mitchell and others, 2019), and the parts of a card.
The 2,000 test digits come from QMNIST (Yadav and Bottou, 2019; BSD-3-Clause, see data/README.md): the 1,000 test
digits of qmnist-small.npz, all written by American high-school students (NIST series 4), and 1,000 written by
employees of the American government (series 0 and 1), from qmnist-writers.npz. No writer of a test digit wrote a
training digit of the reader. The conditions (turned, moved, dark on light) and the pictures are the course's own."""
import base64
import functools
import html
import io
import math
import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from everything_ai import hub, paths, yours
from everything_ai.check import CodeBox, Mistake, Result, Sort, run_code
from everything_ai.models import DigitCNN

REPO = "QRRV/everything-ai-qmnist-cnn"   # the course's digit reader on Hugging Face, as in torch-pretrained
HUB_REPOS = [REPO]                       # the lesson downloads from this repository: the header says so
SMALL = "data/qmnist-small.npz"          # its first 1,000 test digits: students
WRITERS = "data/qmnist-writers.npz"      # who wrote them, and 1,000 test digits by employees
BOUND = 0.9                              # what the intended use of the lesson asks for: 90 in 100 digits right
LANGS = ("en", "nl")

# The lesson is done when these are correct. "sections" is the sort, "system-card" the choice question in the lesson.
REQUIRED = ["per-group", "sections", "system-card"]

# The conditions of exercise 1, as keys in the lesson's code, and what the pictures call them.
CONDITIONS = ["original", "turned +15", "turned -15", "moved 3", "inverted"]
CONDITION_WORDS = {
    "en": {"original": "as written", "turned +15": "turned 15° counterclockwise", "turned -15": "turned 15° clockwise",
           "moved 3": "moved 3 pixels to the right", "inverted": "dark on light"},
    "nl": {"original": "zoals geschreven", "turned +15": "15° tegen de klok in gedraaid",
           "turned -15": "15° met de klok mee gedraaid", "moved 3": "3 pixels naar rechts geschoven",
           "inverted": "donker op licht"},
}
GROUP_WORDS = {"en": {"students": "students", "employees": "employees", "all": "together"},
               "nl": {"students": "scholieren", "employees": "medewerkers", "all": "samen"}}


def _lang(lang: str) -> str:
    return lang if lang in LANGS else "en"


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma and a thousands dot in Dutch, and a real minus sign."""
    out = f"{x:,.{digits}f}"
    out = out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out
    return out.replace("-", "−")


def percent(x: float, lang: str) -> str:
    return number(100 * x, 1, lang) + "%"


# --- the reader and the digits -------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def reader() -> DigitCNN:
    """The course's digit reader from the hub, as torch-pretrained loads it (or the course's copy without a
    connection), in eval mode."""
    model = DigitCNN()
    model.load_state_dict(torch.load(hub_file("model.pt")[0], weights_only=True))
    return model.eval()


def test_digits():
    """2,000 test digits as a tensor of shape (2000, 784) with pixels from 0 to 1, their digits, and the NIST series
    of each digit's writer: the first 1,000 by students (series 4), the next 1,000 by employees (series 0 and 1).
    Each call gives its own copies, so a change in place in a lesson leaves the check's digits as they are."""
    return tuple(t.clone() for t in _test_digits())


@functools.lru_cache(maxsize=1)
def _test_digits():
    small, writers = np.load(paths.ROOT / SMALL), np.load(paths.ROOT / WRITERS)
    images = np.concatenate([small["test_images"], writers["employee_images"]])
    digits = np.concatenate([small["test_labels"], writers["employee_labels"]])
    series = np.concatenate([np.full(1000, 4), writers["employee_series"]])
    return (torch.tensor(images, dtype=torch.float32) / 255, torch.tensor(digits, dtype=torch.int64),
            torch.tensor(series, dtype=torch.int64))


@functools.lru_cache(maxsize=1)
def writer_counts() -> dict:
    """How many of the 1,000 test digits of each group each writer wrote: {group: [count per writer]}."""
    w = np.load(paths.ROOT / WRITERS)
    return {group: sorted(np.unique(w[key], return_counts=True)[1].tolist())
            for group, key in (("students", "student_writers"), ("employees", "employee_writers"))}


def turn(images: torch.Tensor, degrees: float) -> torch.Tensor:
    """The images turned by `degrees` around their middle: counterclockwise for a positive number, clockwise for a
    negative one. What falls outside is lost, and what comes in from outside is black."""
    a = math.radians(degrees)
    theta = torch.tensor([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0]], dtype=torch.float32)
    x = images.reshape(-1, 1, 28, 28)
    grid = F.affine_grid(theta.repeat(len(x), 1, 1), list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False).reshape(images.shape)


def move(images: torch.Tensor, pixels: int) -> torch.Tensor:
    """The images moved `pixels` to the right, with black coming in on the left."""
    x = images.reshape(-1, 28, 28)
    out = torch.zeros_like(x)
    out[:, :, pixels:] = x[:, :, :28 - pixels]
    return out.reshape(images.shape)


def hub_file(filename: str):
    """A file of the digit reader's repository: its path, and "hub" or "course" for where it came from."""
    return hub.fetch(REPO, filename)


def _read_right(images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return reader()(images).argmax(dim=1) == test_digits()[1]


def changed(images: torch.Tensor) -> dict:
    """The five conditions of exercise 1, as the lesson's cell makes them."""
    return {"original": images, "turned +15": turn(images, 15), "turned -15": turn(images, -15),
            "moved 3": move(images, 3), "inverted": 1 - images}


@functools.lru_cache(maxsize=1)
def _right_when() -> dict:
    return {name: _read_right(x) for name, x in changed(test_digits()[0]).items()}


def default_names() -> dict:
    """The names the code of exercise 1 gets, as the lesson makes them."""
    return {"right_when": {k: v.clone() for k, v in _right_when().items()}, "series": test_digits()[2]}


def _shuffled(names: dict) -> dict:
    """The same digits in another order: code that relies on the employees being the last 1,000 goes wrong."""
    order = torch.randperm(len(names["series"]), generator=torch.Generator().manual_seed(7))
    return {"right_when": {k: v[order] for k, v in names["right_when"].items()}, "series": names["series"][order]}


def accuracy(right: torch.Tensor, mask: torch.Tensor) -> float:
    return right[mask].float().mean().item()


@functools.lru_cache(maxsize=1)
def numbers() -> dict:
    """The course's table, and the other ways of splitting that the mistakes of exercise 1 give."""
    _, digits, series = test_digits()
    rw = _right_when()
    masks = {"students": series == 4, "employees": series != 4, "series-0": series == 0, "series-1": series == 1,
             "all": torch.ones_like(series, dtype=torch.bool)}
    out = {k: {c: accuracy(r, m) for c, r in rw.items()} for k, m in masks.items()}
    out["count"] = {c: r[series != 4].sum().item() for c, r in rw.items()}
    out["per-digit"] = {d: accuracy(rw["original"], digits == d) for d in range(10)}
    return out


# --- pictures --------------------------------------------------------------------------------------------------------

W = 330


def _svg(height: float, alt: str, parts: list[str]) -> str:
    return (f'<svg viewBox="0 0 {W} {height:.0f}" role="img" aria-label="{html.escape(alt)}" '
            f'style="width:100%;max-width:{W}px;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(alt)}</title>' + "".join(parts) + "</svg>")


def _bar(y: float, label: str, value, color: str, lang: str) -> tuple[list[str], float]:
    """One bar from 0 to 100%: the share right in the group's color, the share wrong in the color of mistakes, and
    the number after it. A value that is not a number between 0 and 1 is written out instead of drawn."""
    x0, x1 = 86, W - 46
    parts = [f'<text x="{x0 - 6}" y="{y + 10}" text-anchor="end" style="font-size:12px;fill:var(--ea-ink-2)">'
             f'{html.escape(label)}</text>']
    try:
        v = float(value)
    except (TypeError, ValueError, OverflowError):
        v = None
    if v is None or not 0 <= v <= 1:
        shown = html.escape(_short(value) if v is None else _short(number(v, 3, lang)))
        parts.append(f'<text x="{x0}" y="{y + 10}" style="font-size:12px;fill:var(--ea-loss-text)">{shown}</text>')
        return parts, y + 18
    mid = x0 + (x1 - x0) * v
    parts.append(f'<rect x="{x0}" y="{y}" width="{mid - x0:.1f}" height="13" style="fill:{color}"/>'
                 f'<rect x="{mid:.1f}" y="{y}" width="{x1 - mid:.1f}" height="13" style="fill:var(--ea-loss);opacity:0.75"/>'
                 f'<text x="{x1 + 5}" y="{y + 10.5}" style="font-size:12px;fill:var(--ea-ink)">{percent(v, lang)}</text>')
    return parts, y + 18


def _short(value) -> str:
    """A value as text, cut off after 40 characters: a list of 1,000 values stays one line."""
    text = str(value)
    return text if len(text) <= 40 else text[:40] + "…"


def _group_color(key: str) -> str:
    return {"students": "var(--ea-group-1)", "employees": "var(--ea-group-2)"}.get(key, "var(--ea-ink-3)")


def bars(blocks: list, alt: str, lang: str, bound: bool = True) -> str:
    """Blocks of bars, each with a heading: [(heading, [(group key, value), ...]), ...]. With `bound`, a line at 90%,
    what the intended use asks for."""
    lang = _lang(lang)
    words = GROUP_WORDS[lang]
    x0, x1 = 86, W - 46
    parts, y = [], 4
    if bound:
        xb = x0 + (x1 - x0) * BOUND
        label = {"en": "90%: what the use asks", "nl": "90%: wat het gebruik vraagt"}[lang]
        parts.append(f'<text x="{xb:.1f}" y="12" text-anchor="end" style="font-size:11.5px;fill:var(--ea-ink-2)">'
                     f'{html.escape(label)} ↓</text>')
        y = 20
    top = y
    for heading, rows in blocks:
        parts.append(f'<text x="0" y="{y + 12}" style="font-size:12.5px;font-weight:600;fill:var(--ea-ink)">'
                     f'{html.escape(heading)}</text>')
        y += 18
        for key, value in rows:
            drawn, y = _bar(y, words.get(key, str(key)), value, _group_color(key), lang)
            parts += drawn
        y += 6
    if bound:
        parts.append(f'<line x1="{xb:.1f}" y1="{top}" x2="{xb:.1f}" y2="{y - 4}" '
                     f'style="stroke:var(--ea-ink);stroke-width:1.2;stroke-dasharray:3 3"/>')
    return _svg(y, alt, parts)


def _figure(svg: str, caption: str = "") -> str:
    note = (f'<figcaption style="font-family:var(--ea-font-ui);font-size:0.85rem;color:var(--ea-ink-2);margin-top:0.2rem">'
            f'{html.escape(caption)}</figcaption>') if caption else ""
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:min(100%,{W}px)">{svg}{note}</figure>')


def _alt_rows(blocks: list, lang: str) -> str:
    words = GROUP_WORDS[_lang(lang)]
    out = []
    for heading, rows in blocks:
        shown = []
        for k, v in rows:
            try:
                shown.append(f"{words.get(k, k)} {_short(percent(float(v), lang))}")
            except (TypeError, ValueError, OverflowError):
                shown.append(f"{words.get(k, k)} {_short(v)}")
        out.append(f"{heading}: " + ", ".join(shown))
    return "; ".join(out) + "."


def split_picture(lang: str) -> str:
    """The picture of the idea: all 2,000 digits, then per group, then per group under one condition."""
    lang = _lang(lang)
    n = numbers()
    heads = {"en": ["All 2,000 digits", "Per group of writers", "Per group, turned 15° clockwise"],
             "nl": ["Alle 2.000 cijfers", "Per groep schrijvers", "Per groep, 15° met de klok mee gedraaid"]}[lang]
    blocks = [(heads[0], [("all", n["all"]["original"])]),
              (heads[1], [("students", n["students"]["original"]), ("employees", n["employees"]["original"])]),
              (heads[2], [("students", n["students"]["turned -15"]), ("employees", n["employees"]["turned -15"])])]
    title = {"en": "The digit reader, per group and per condition", "nl": "De cijferlezer, per groep en per omstandigheid"}[lang]
    alt = title + ". " + _alt_rows(blocks, lang)
    return (f'<div style="font-family:var(--ea-font-ui);font-size:0.92rem;font-weight:600;color:var(--ea-ink);'
            f'margin:0.4rem 0 0.2rem">{html.escape(title)}</div>' + _figure(bars(blocks, alt, lang)))


def _table_blocks(table, lang: str) -> list | None:
    """The learner's table as blocks of bars, or None when it is not a dict of dicts."""
    if not isinstance(table, dict) or not table or not all(isinstance(v, dict) for v in table.values()):
        return None
    names = CONDITION_WORDS[_lang(lang)]
    return [(names.get(c, str(c)), list(rows.items())[:3]) for c, rows in list(table.items())[:6]]


def table_picture(seen) -> str:
    """Under the check of exercise 1: the learner's table as bars, per condition and per group, with the 90% the
    intended use asks for."""
    lang = _lang(seen.lang)
    if "right_when" in seen.names:
        run = seen.mine
    else:   # a picture asked for before any check: the names are made here, as the check makes them
        run = run_code(seen.text, default_names()) if isinstance(seen.text, str) else None
        run = run if run is not None and run.error is None and not run.timeout else None
    blocks = _table_blocks(run.value if run is not None else None, lang)
    if blocks is None:
        return ""
    title = {"en": "Your table: the reader per condition and per group",
             "nl": "Jouw tabel: de lezer per omstandigheid en per groep"}[lang]
    alt = title + ". " + _alt_rows(blocks, lang)
    return yours.figure(title, _figure(bars(blocks, alt, lang)), alt=alt)


def _png(pixels: np.ndarray, scale: int = 2) -> str:
    """A digit as a small PNG, white on black as the reader sees it."""
    from PIL import Image
    img = Image.fromarray((np.clip(pixels, 0, 1) * 255).astype(np.uint8).reshape(28, 28))
    buf = io.BytesIO()
    img.resize((28 * scale, 28 * scale), Image.NEAREST).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


SAMPLE = [0, 1, 2, 1004, 1005, 1006]   # three digits by students, three by employees


@functools.lru_cache(maxsize=32)
def _turned(degrees: int) -> tuple:
    images, _, series = test_digits()
    right = _read_right(turn(images, degrees))
    return accuracy(right, series == 4), accuracy(right, series != 4), turn(images[SAMPLE], degrees).numpy()


def turn_frame(degrees: int, lang: str) -> tuple[str, str]:
    """The demo: six digits turned by `degrees`, and the accuracy per group for all 2,000 turned that far."""
    lang = _lang(lang)
    students, employees, pixels = _turned(int(degrees))
    words = GROUP_WORDS[lang]
    size = 46
    parts = []
    for k, px in enumerate(pixels):
        x = 4 + k * (size + 8) + (10 if k >= 3 else 0)
        parts.append(f'<image x="{x}" y="18" width="{size}" height="{size}" href="{_png(px)}"/>')
    parts.append(f'<text x="4" y="12" style="font-size:12px;fill:var(--ea-ink-2)">{html.escape(words["students"])}</text>'
                 f'<text x="{4 + 3 * (size + 8) + 10}" y="12" style="font-size:12px;fill:var(--ea-ink-2)">'
                 f'{html.escape(words["employees"])}</text>')
    head = {"en": "All 2,000 digits, turned this far", "nl": "Alle 2.000 cijfers, zo ver gedraaid"}[lang]
    blocks = [(head, [("students", students), ("employees", employees)])]
    alt = _turn_said(degrees, students, employees, lang)
    chart = bars(blocks, alt, lang)
    inner = re.search(r"<title>.*?</title>(.*)</svg>", chart, re.S).group(1)
    height = float(re.search(r'viewBox="0 0 \d+ ([\d.]+)"', chart).group(1))
    body = parts + [f'<g transform="translate(0,72)">{inner}</g>']
    return _svg(72 + height, alt, body), alt


def _turn_said(degrees: int, students: float, employees: float, lang: str) -> str:
    d = abs(int(degrees))
    if lang == "nl":
        how = "niet gedraaid" if d == 0 else (f"{d}° tegen de klok in gedraaid" if degrees > 0 else f"{d}° met de klok mee gedraaid")
        return f"Cijfers {how}: de scholieren {percent(students, lang)} goed, de medewerkers {percent(employees, lang)}."
    how = "not turned" if d == 0 else (f"turned {d}° counterclockwise" if degrees > 0 else f"turned {d}° clockwise")
    return f"Digits {how}: students {percent(students, lang)} right, employees {percent(employees, lang)}."


# --- the card of exercise 2 --------------------------------------------------------------------------------------------

SECTION_ORDER = ["for", "not-for", "data", "limits"]


def card_picture(seen) -> str:
    """Under the check of exercise 2: the parts of the card, with the items where the learner put them."""
    lang = _lang(seen.lang)
    chosen = {k: v for k, v in dict(seen.text or {}).items() if v} if isinstance(seen.text, dict) else {}
    ex = EXERCISES["sections"]
    if not chosen:
        return ""
    rows = []
    for g in SECTION_ORDER:
        items = [k for k in ex.items if chosen.get(k) == g]
        names = "".join(f'<li style="margin:0.1rem 0">{html.escape(_name(ex.items[k][lang]))}</li>' for k in items)
        empty = {"en": "(nothing yet)", "nl": "(nog niets)"}[lang]
        body = f'<ul style="margin:0.1rem 0 0.4rem 1.1rem;padding:0">{names}</ul>' if items else (
            f'<div style="color:var(--ea-ink-3);margin:0.1rem 0 0.4rem">{empty}</div>')
        rows.append(f'<div style="font-weight:600;color:var(--ea-ink)">{html.escape(ex.groups[g][lang])}</div>{body}')
    title = {"en": "Your card: the parts as you filled them", "nl": "Jouw card, zoals jij de delen vulde"}[lang]
    box = ('<div style="font-family:var(--ea-font-ui);font-size:0.88rem;border:1.2px solid var(--ea-ink-3);'
           'border-radius:8px;padding:0.55rem 0.75rem;max-width:min(100%,30rem);background:var(--ea-page)">'
           + "".join(rows) + "</div>")
    return yours.figure(title, box, alt=title)


def _name(text: str) -> str:
    m = re.match(r"\*\*(.+?)\.?\*\*", text)
    return m.group(1) if m else re.sub(r"[*_`]", "", text)


# --- exercise 1: per group and per condition ---------------------------------------------------------------------------

_START = '''students = series == 4
table = {}
for condition in right_when:
    right = right_when[condition]
    table[condition] = {
        "students": right[students].float().mean().item(),
        "employees": ...,
    }
table
'''
_LINE = '        "employees": right[series != 4].float().mean().item(),\n'
SOLUTION = _START.replace('        "employees": ...,\n', _LINE)


def _with(line: str) -> str:
    return _START.replace('        "employees": ...,\n', line if line.endswith("\n") else line + "\n")


def _employees(got) -> dict | None:
    """The employees' number per condition in the learner's table, when it is one."""
    if not isinstance(got, dict) or not got:
        return None
    out = {}
    for c, rows in got.items():
        if not isinstance(rows, dict) or "employees" not in rows:
            return None
        try:
            out[c] = float(rows["employees"])
        except (TypeError, ValueError, OverflowError):
            return None
    return out


def _like(kind: str):
    """A test: the employees' numbers are those of another way of splitting, for every condition."""
    def test(got, text) -> bool:
        mine, want = _employees(got), numbers()[kind]
        return (mine is not None and mine.keys() == want.keys()
                and all(abs(mine[c] - want[c]) < 1e-6 for c in want))
    return test


def _students_off(got, text) -> bool:
    if not isinstance(got, dict) or set(got) != set(CONDITIONS):
        return False
    try:
        return any(abs(float(got[c]["students"]) - numbers()["students"][c]) > 1e-6 for c in CONDITIONS)
    except (TypeError, ValueError, OverflowError, KeyError):
        return False


def _keys_off(got, text) -> bool:
    if not isinstance(got, dict):
        return False
    return set(got) != set(CONDITIONS) or any(not isinstance(v, dict) or set(v) != {"students", "employees"}
                                              for v in got.values())


def _no_digits(got, text) -> bool:
    mine = _employees(got)
    return mine is not None and any(math.isnan(v) for v in mine.values())


def _counts(got, text) -> bool:
    mine = _employees(got)
    return mine is not None and any(v > 1 for v in mine.values())


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. What comes in its place, on the line for the employees?",
                nl="Er staat nog een `...` in je code. Wat komt er op deze plek, op de regel voor de medewerkers?")


@dataclass
class TableBox(CodeBox):
    """The code box of exercise 1. Its names, the reader's results per condition and the series of each digit, are
    made the first time a check needs them, not when the lesson opens; the check also runs the code on the same
    digits in another order."""

    def check(self, text, inputs=None) -> Result:
        names = dict(inputs) if inputs is not None else default_names()
        self.names, self.others = names, [_shuffled(names)]
        return super().check(text, names)


_TEXTS_1 = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "empty": {"en": "Replace the `...` on the line for the employees: their accuracy under this condition, with a mask as on the line for the students.",
              "nl": "Vervang de `...` op de regel voor de medewerkers: de accuracy onder deze omstandigheid, met een mask zoals op de regel voor de scholieren."},
    "cleared": {"en": "The box has no code in it any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
    "no_value": {"en": "The last line of the box shows nothing. Is `table` on the last line, so the check can see it?",
                 "nl": "De laatste regel van het vak laat niets zien. Staat `table` op de laatste regel, zodat de check het kan zien?"},
    "wrong": {"en": "Your table is not the course's table yet. Does the line for the employees take the mean of `right` over the digits of series 0 and 1 together?",
              "nl": "Je tabel is nog niet de tabel van de course. Neemt de regel voor de medewerkers het gemiddelde van `right` over de cijfers van serie 0 en 1 samen?"},
    "other": {"en": "Right for these 2,000 digits. But the check also put the same digits in another order, and then your numbers for the employees change. Does your mask come from `series`, whatever the order of the digits?",
              "nl": "Goed voor deze 2.000 cijfers. De check zette dezelfde cijfers ook in een andere volgorde, en dan veranderen je getallen voor de medewerkers. Komt je mask uit `series`, wat de volgorde van de cijfers ook is?"},
    "correct": {"en": "Right. As written, the reader gets 92.9% of the students' digits right and 96.0% of the employees'. Turned 15° clockwise, it is the other way round: 86.0% against 82.7%. Under every other condition, both groups fall below the 90% of the picture, except the employees' digits turned counterclockwise.",
                "nl": "Goed. Zoals geschreven leest de lezer 92,9% van de cijfers van de scholieren goed en 96,0% van de medewerkers. 15° met de klok mee gedraaid is het andersom: 86,0% tegen 82,7%. Onder elke andere omstandigheid zakken beide groepen onder de 90% van het plaatje, behalve de cijfers van de medewerkers, tegen de klok in gedraaid."},
}

_MISTAKES_1 = [
    _DOTS,
    Mistake("float", error=lambda e: isinstance(e, RuntimeError) and "infer output dtype" in str(e),
            en="`.mean()` cannot take the mean of True and False as they are. Does your line have `.float()` before `.mean()`, as the line for the students has?",
            nl="`.mean()` kan het gemiddelde van True en False niet zomaar nemen. Staat er op je regel `.float()` voor `.mean()`, zoals op de regel voor de scholieren?"),
    Mistake("or-masks", error=lambda e: isinstance(e, RuntimeError) and "ambiguous" in str(e),
            en="Python asked here whether a whole mask is True, and a mask of 2,000 values has no single answer. That happens with `or`, `and` or `in` on a mask, and with `|` between comparisons without brackets. Which one comparison of `series` with 4 is True for series 0 and series 1 alike?",
            nl="Python vroeg hier of een heel mask True is, en een mask van 2.000 waarden heeft geen enkel antwoord. Dat gebeurt met `or`, `and` of `in` op een mask, en met `|` tussen vergelijkingen zonder haakjes. Welke ene vergelijking van `series` met 4 is True voor serie 0 en voor serie 1?"),
    Mistake("keys", matches=_keys_off,
            en="The check looks for the five conditions of `right_when`, and under each the keys `\"students\"` and `\"employees\"`, as in the start code. Are they all there, spelled the same?",
            nl="De check zoekt de vijf omstandigheden van `right_when`, en onder elke omstandigheid de keys `\"students\"` en `\"employees\"`, zoals in de startcode. Staan alle omstandigheden en keys erin, op dezelfde manier geschreven?"),
    Mistake("students-changed", matches=_students_off,
            en="The numbers for the students are not the ones the start code gave. Did `students` or the line for the students change by accident?",
            nl="De getallen voor de scholieren zijn niet meer de getallen die de startcode gaf. Is `students` of de regel voor de scholieren per ongeluk veranderd?"),
    Mistake("no-digits", matches=_no_digits,
            en="Your numbers for the employees are `nan`: your mask keeps no digit at all. Which series did the employees write?",
            nl="Je getallen voor de medewerkers zijn `nan`: je mask houdt geen enkel cijfer over. Welke series schreven de medewerkers?"),
    Mistake("count", matches=_counts,
            en="Your numbers for the employees are above 1, but a share runs from 0 to 1. Is it a count of digits read right, or a percentage? How does your line differ from the line for the students?",
            nl="Je getallen voor de medewerkers zijn groter dan 1, terwijl een deel van 0 tot 1 loopt. Is het een aantal goed gelezen cijfers, of een percentage? Waarin verschilt je regel van de regel voor de scholieren?"),
    Mistake("students-twice", matches=_like("students"),
            en="The employees get the same numbers as the students. Is the mask on their line the same as on the line above?",
            nl="De medewerkers krijgen dezelfde getallen als de scholieren. Is het mask op de regel van de medewerkers hetzelfde als op de regel erboven?"),
    Mistake("series-0", matches=_like("series-0"),
            en="Your numbers for the employees are those of series 0 alone. Which series did the other employees write, and which mask takes both series?",
            nl="Je getallen voor de medewerkers zijn de getallen van alleen serie 0. Welke serie schreven de andere medewerkers, en welk mask neemt beide series?"),
    Mistake("series-1", matches=_like("series-1"),
            en="Your numbers for the employees are those of series 1 alone. Which series did the other employees write, and which mask takes both series?",
            nl="Je getallen voor de medewerkers zijn de getallen van alleen serie 1. Welke serie schreven de andere medewerkers, en welk mask neemt beide series?"),
    Mistake("all", matches=_like("all"),
            en="Your numbers for the employees are those of all 2,000 digits. Which mask keeps only the digits the employees wrote?",
            nl="Je getallen voor de medewerkers zijn de getallen van alle 2.000 cijfers. Welk mask houdt alleen de cijfers van de medewerkers over?"),
]


# --- exercise 2: the parts of the card ---------------------------------------------------------------------------------

_SECTIONS = Sort(
    name="sections",
    picture=card_picture,
    groups={
        "for": {"en": "What it is for", "nl": "Waarvoor het is"},
        "not-for": {"en": "What it is not for", "nl": "Waarvoor het niet is"},
        "data": {"en": "What it learned from and was tested on", "nl": "Waarvan het leerde en waarop het getest is"},
        "limits": {"en": "Known limits", "nl": "Bekende grenzen"},
    },
    items={
        "forms": {"en": "**Forms.** Reading the digits in the boxes of scanned forms, one digit per box, each made white on black of 28 by 28 pixels first.",
                  "nl": "**Formulieren.** De cijfers lezen in de vakjes van gescande formulieren, één cijfer per vakje, elk eerst wit op zwart gemaakt, 28 bij 28 pixels."},
        "lessons": {"en": "**Lessons.** Seeing how a small network for images works, in a course such as this one.",
                    "nl": "**Lessen.** Zien hoe een klein netwerk voor beelden werkt, in een course zoals deze."},
        "money": {"en": "**Money.** Deciding alone how much money to pay out, from an amount written by hand, with nobody looking.",
                  "nl": "**Geld.** Alleen beslissen hoeveel geld er uitbetaald wordt, op basis van een met de hand geschreven bedrag, zonder dat iemand meekijkt."},
        "streets": {"en": "**Street photos.** Reading house numbers in photos of streets.",
                    "nl": "**Straatfoto's.** Huisnummers lezen op foto's van straten."},
        "train": {"en": "**Training digits.** 6,000 digits of QMNIST: 3,000 written by employees of the American government, 3,000 by American high-school students, 279 writers in all.",
                  "nl": "**Trainingscijfers.** 6.000 cijfers uit QMNIST: 3.000 geschreven door medewerkers van de Amerikaanse overheid, 3.000 door Amerikaanse scholieren, 279 schrijvers in totaal."},
        "test": {"en": "**Test digits.** 1,000 digits by 247 students and 1,000 by 233 employees, all writers who wrote no training digit.",
                 "nl": "**Testcijfers.** 1.000 cijfers van 247 scholieren en 1.000 van 233 medewerkers, allemaal schrijvers zonder een enkel trainingscijfer."},
        "turned": {"en": "**Turned.** Turned 15° clockwise, the reader gets fewer than 90 in 100 digits right, in both groups.",
                   "nl": "**Gedraaid.** 15° met de klok mee gedraaid leest de lezer minder dan 90 van de 100 cijfers goed, in beide groepen."},
        "dark": {"en": "**Dark on light.** Dark digits on a light background: fewer than 20 in 100 right.",
                 "nl": "**Donker op licht.** Donkere cijfers op een lichte achtergrond: minder dan 20 van de 100 goed."},
    },
    solution={"forms": "for", "lessons": "for", "money": "not-for", "streets": "not-for", "train": "data",
              "test": "data", "turned": "limits", "dark": "limits"},
    nudge={
        "forms": {"en": "Is this a task the reader was made for, with inputs like its training digits?",
                  "nl": "Is dit een taak waarvoor de lezer gemaakt is, met invoer zoals de trainingscijfers?"},
        "lessons": {"en": "The course made the reader for its lessons. Is this a use it has in mind, or one it warns against?",
                    "nl": "De course maakte de lezer voor de lessen. Is dit een gebruik dat de course voor ogen heeft, of een gebruik waarvoor de course waarschuwt?"},
        "money": {"en": "About 7 in 100 students' digits go wrong. Should the reader decide about money alone?",
                  "nl": "Ongeveer 7 van de 100 cijfers van scholieren gaan fout. Mag de lezer alleen over geld beslissen, zonder dat iemand meekijkt?"},
        "streets": {"en": "Do photos of streets look like the white digits on black the reader learned from?",
                    "nl": "Lijken foto's van straten op de witte cijfers op zwart waarvan de lezer leerde?"},
        "train": {"en": "Is this about what the reader may be used for, or about the digits behind it?",
                  "nl": "Gaat dit over waarvoor de lezer gebruikt mag worden, of over de cijfers erachter?"},
        "test": {"en": "Is this a use or a limit, or does it say on which digits the numbers were measured?",
                 "nl": "Is dit een gebruik of een grens, of zegt het op welke cijfers de getallen gemeten zijn?"},
        "turned": {"en": "Is this a use, or a condition under which the numbers fall below what the use asks for?",
                   "nl": "Is dit een gebruik, of een omstandigheid waaronder de getallen zakken onder wat het gebruik vraagt?"},
        "dark": {"en": "Is this about the data behind the reader, or about inputs on which it fails?",
                 "nl": "Gaat dit over de data achter de lezer, of over invoer waarop de lezer faalt?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the parts again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de delen opnieuw uit."},
        "empty": {"en": "Put each line in the part of the card where it belongs.",
                  "nl": "Zet elke regel in het deel van de card waar de regel hoort."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "The line *{item}* belongs in another part of the card.",
                  "nl": "De regel *{item}* hoort in een ander deel van de card."},
        "correct": {"en": "All {n} are right. A card like this tells someone what the reader is for and what not, which digits stand behind its numbers, and where it fails, before they build on it.",
                    "nl": "Alle {n} kloppen. Met zo'n card weet iemand waarvoor de lezer is en waarvoor niet, welke cijfers achter de getallen staan, en waar de lezer faalt, nog voor iemand erop bouwt."},
    },
)


EXERCISES = {
    "per-group": TableBox(
        name="per-group",
        picture=table_picture,
        start=_START,
        solution=SOLUTION,
        digits=3,
        mistakes=_MISTAKES_1,
        texts=_TEXTS_1,
    ),
    "sections": _SECTIONS,
}

_SOLVED = dict(_SECTIONS.solution)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("per-group", _START, "empty", None),
    ("per-group", "", "empty", None),
    ("per-group", SOLUTION, "correct", None),
    ("per-group", _with('        "employees": right[series < 4].float().mean().item(),'), "correct", None),
    ("per-group", _with('        "employees": right[series <= 1].float().mean().item(),'), "correct", None),
    ("per-group", _with('        "employees": right[~students].float().mean().item(),'), "correct", None),
    ("per-group", _with('        "employees": right[series != 4].float().mean(),'), "correct", None),
    ("per-group", _with('        "employees": right[series != 4].sum().item() / (series != 4).sum().item(),'),
     "correct", None),
    ("per-group", _with('        "employees": right[1000:].float().mean().item(),'), "almost", "other-value"),
    ("per-group", _with('        "employees": 0.96,'), "wrong", None),
    ("per-group", _START.replace("students = series == 4\n", "students = series == 4\nn = 2\n"), "wrong", "dots-left"),
    ("per-group", _with('        "employees": right[series != 4].mean().item(),'), "error", "float"),
    ("per-group", _with('        "employees": right[series == 0 or series == 1].float().mean().item(),'), "error",
     "or-masks"),
    ("per-group", _with('        "employee": right[series != 4].float().mean().item(),'), "wrong", "keys"),
    ("per-group", _with('        "employees": right[series != 4].float().mean().item(),').replace(
        "right[students]", "right[series == 0]"), "wrong", "students-changed"),
    ("per-group", _with('        "employees": right[series != 4].sum().item(),'), "wrong", "count"),
    ("per-group", _with('        "employees": right[series != 4].float().mean().item() * 100,'), "wrong", "count"),
    ("per-group", _with('        "employees": right[series > 4].float().mean().item(),'), "wrong", "no-digits"),
    ("per-group", _with('        "employees": right[series in [0, 1]].float().mean().item(),'), "error", "or-masks"),
    ("per-group", _with('        "employees": right[students].float().mean().item(),'), "wrong", "students-twice"),
    ("per-group", _with('        "employees": right[series == 0].float().mean().item(),'), "wrong", "series-0"),
    ("per-group", _with('        "employees": right[series == 1].float().mean().item(),'), "wrong", "series-1"),
    ("per-group", _with('        "employees": right.float().mean().item(),'), "wrong", "all"),
    ("per-group", _with('        "employees": right[series != 4].float().mean().item(),').replace("\ntable\n", "\n"),
     "wrong", "no-value"),
    ("sections", {}, "empty", None),
    ("sections", _SOLVED, "correct", None),
    ("sections", {"forms": "for", "money": "not-for"}, "wrong", "unfinished"),
    ("sections", {**_SOLVED, "lessons": "not-for"}, "wrong", "wrong:lessons"),
    ("sections", {**_SOLVED, "test": "limits"}, "wrong", "wrong:test"),
    ("sections", {**_SOLVED, "turned": "data"}, "wrong", "wrong:turned"),
    ("sections", {**_SOLVED, "streets": "for"}, "wrong", "wrong:streets"),
]
