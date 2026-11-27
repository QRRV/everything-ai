# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-model-package: packing a model. A model you hand over is its weights, its settings and its
preprocessing; the weights go in a safetensors file, which the learner writes by hand: 8 bytes with the length of a
JSON header, the header with the dtype, shape and place of each tensor, then the numbers. The learner writes `header`,
which gives each tensor its place among the bytes, and `read_tensor`, which takes one tensor back out of the bytes of
a file; the demo walks through the bytes of the course's digit reader as a safetensors file.

The model is the course's digit reader, data/course-qmnist-cnn.pt, the state_dict that the lesson *Using a pretrained
model* loaded from the hub; the course keeps a copy in data/, so nothing is downloaded here. The safetensors files are
written by this module in the layout the format describes; the library safetensors itself is not installed."""
import copy
import html
import json

import numpy as np
import torch

from everything_ai import models, paths, yours
from everything_ai.check import FunctionBox, Mistake

LESSON = "sys-model-package"
LANGS = ("en", "nl")

# ======== the course's model and its safetensors file =====================================================================

READER = models.DIGIT_CNN                    # data/course-qmnist-cnn.pt: the digit reader's state_dict
STATE_DICT = torch.load(READER, weights_only=True)
PARAMETERS = sum(t.numel() for t in STATE_DICT.values())   # 51,648
NUMBER_BYTES = PARAMETERS * 4                              # 206,592: float32 takes 4 bytes per number


def header(state):
    out = {}
    start = 0
    for name, t in state.items():
        end = start + t.numel() * 4
        out[name] = {"dtype": "F32", "shape": list(t.shape), "data_offsets": [start, end]}
        start = end
    return out


def header_text(state, library: bool = False) -> bytes:
    """The header as the file holds it: JSON in UTF-8, with spaces after it until its length is a multiple of 8. With
    library=True as the library safetensors writes it: the tensors sorted by name, and JSON without spaces."""
    text = json.dumps(header(state), separators=(",", ":") if library else None).encode()
    while len(text) % 8:
        text += b" "
    return text


def to_safetensors(state, library: bool = False) -> bytes:
    """A safetensors file of float32 tensors: 8 bytes with the length of the header (little-endian), the header, and
    the numbers of each tensor in the order of the header. Every tensor goes in as float32, 4 bytes per number."""
    state = dict(sorted(state.items())) if library else state
    text = header_text(state, library)
    numbers = b"".join(t.detach().contiguous().numpy().astype("<f4").tobytes() for t in state.values())
    return len(text).to_bytes(8, "little") + text + numbers


def read_tensor(data, name):
    n = int.from_bytes(data[:8], "little")
    head = json.loads(data[8:8 + n])
    start, end = head[name]["data_offsets"]
    numbers = data[8 + n + start:8 + n + end]
    return torch.frombuffer(bytearray(numbers), dtype=torch.float32).reshape(head[name]["shape"])


def package_path(name: str) -> str:
    """For the worked example: a file in the learner's work folder, work/package/<name>."""
    folder = paths.WORK / "package"
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder / name)


class _State(dict):
    """A state_dict of the exercise. The check shows it by its name, such as `C.STATE`; the function gets its tensors."""

    def __init__(self, tensors, shown: str):
        super().__init__(tensors)
        self.shown = shown

    def __repr__(self) -> str:
        return self.shown

    def __deepcopy__(self, memo):
        return _State({k: v.clone() for k, v in self.items()}, self.shown)


class _File(bytes):
    """The bytes of a safetensors file of the exercise, shown by name, such as `C.FILE`."""

    def __new__(cls, data: bytes, shown: str):
        out = super().__new__(cls, data)
        out.shown = shown
        return out

    def __repr__(self) -> str:
        return self.shown

    def __deepcopy__(self, memo):
        return self


STATE = _State(STATE_DICT, "C.STATE")
_SMALL_TENSORS = {"w": torch.arange(6.0).reshape(2, 3), "b": torch.tensor([0.5, -0.5])}
SMALL = _State(_SMALL_TENSORS, '{"w": torch.arange(6.0).reshape(2, 3), "b": torch.tensor([0.5, -0.5])}')
CUBE = _State({"k": torch.ones(2, 2, 2)}, '{"k": torch.ones(2, 2, 2)}')
DOUBLE = _State({"x": torch.tensor([1.0, 2.0], dtype=torch.float64)},
                '{"x": torch.tensor([1.0, 2.0], dtype=torch.float64)}')
NONE = _State({}, "{}")
FILE = _File(to_safetensors(STATE_DICT), "C.FILE")
SMALL_FILE = _File(to_safetensors(_SMALL_TENSORS), "C.SMALL_FILE")
SORTED = _File(to_safetensors(STATE_DICT, library=True), "C.SORTED")   # the same weights as the library writes them
HEADER_BYTES = len(header_text(STATE_DICT))          # 824: 819 bytes of JSON and 5 spaces
START = 8 + HEADER_BYTES                             # 832: where the numbers begin in C.FILE

# How many of the 1,000 test digits of QMNIST the reader gets right, with the pixels as it learned them (0 to 1, ink at
# 1), from 0 to 255, and the other way around (1 minus each pixel): what tests/test_sys_model_package.py measures.
READ_RIGHT = {"learned": 929, "raw": 927, "inverted": 157}


def reads_right(kind: str) -> int:
    data = np.load(paths.ROOT / "data" / "qmnist-small.npz")
    x = torch.tensor(data["test_images"], dtype=torch.float32)
    y = torch.tensor(data["test_labels"]).long()
    x = {"learned": x / 255, "raw": x, "inverted": 1 - x / 255}[kind]
    model = models.DigitCNN()
    model.load_state_dict(STATE_DICT)
    model.eval()
    with torch.no_grad():
        return int((model(x).argmax(1) == y).sum())


# ======== drawing =========================================================================================================

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem;max-width:var(--ea-measure)"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem;max-width:var(--ea-measure)"
_SVG = "width:100%;max-width:330px;height:auto;font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums"


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma and a thousands dot in Dutch, and a real minus sign."""
    out = f"{x:,.{digits}f}"
    out = out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out
    return out.replace("-", "−")


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


def _rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "", opacity: float = 1.0) -> str:
    edge = f";stroke:{stroke};stroke-width:1" if stroke else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" '
            f'style="fill:{fill};fill-opacity:{opacity:.2f}{edge}"/>')


def _svg(height: float, alt: str, parts: list) -> str:
    return (f'<svg viewBox="0 0 330 {height:.0f}" role="img" aria-label="{html.escape(alt)}" style="{_SVG}">'
            f'{"".join(parts)}</svg>')


def _short(text_: str, n: int) -> str:
    return text_ if len(text_) <= n else text_[:n - 1] + "…"


# ======== the picture of the idea =========================================================================================

IDEA_TEXT = {
    "en": {"title": "A model package: two files in a folder",
           "weights": "the weights", "config": "the settings and the preprocessing", "length": "8 bytes: length of the header",
           "header": "JSON header: per tensor dtype, shape, place", "numbers": "the numbers, float32, one tensor after another",
           "alt": "A folder with two files. model.safetensors holds 8 bytes with the length of the header, a JSON header with the dtype, shape and place of each tensor, and the numbers of all tensors one after another. config.json holds the class of the model, how the pixels go in, the labels and the version of the weights."},
    "nl": {"title": "Een modelpakket: twee bestanden in een map",
           "weights": "de weights", "config": "de instellingen en de voorbewerking", "length": "8 bytes: lengte van de header",
           "header": "JSON-header: per tensor dtype, shape, plek", "numbers": "de getallen, float32, de ene tensor na de andere",
           "alt": "Een map met twee bestanden. model.safetensors bevat 8 bytes met de lengte van de header, een JSON-header met het dtype, de shape en de plek van elke tensor, en de getallen van alle tensors achter elkaar. config.json bevat de class van het model, hoe de pixels erin gaan, de labels en de versie van de weights."},
}
IDEA_CONFIG = {lang: ['"class": "DigitCNN",', pixels, '"labels": [0, 1, ..., 9],', '"weights": "model.safetensors",',
                      '"version": "…"']
               for lang, pixels in (("en", '"pixels": "28 by 28, 0 to 1, ink 1",'),
                                    ("nl", '"pixels": "28 bij 28, 0 tot 1, inkt 1",'))}


def idea_picture(lang: str) -> str:
    """The folder of a model package: model.safetensors in its three parts, and config.json beside it."""
    w = IDEA_TEXT[lang]
    parts = [_t(0, 14, w["title"], 12.5, 600)]
    y = 30.0
    parts.append(_rect(0, y, 330, 112, "var(--ea-page)", "var(--ea-ink-3)"))
    parts.append(_mono(8, y + 16, "model.safetensors", 10.5, weight=600))
    parts.append(_t(322, y + 16, w["weights"], 9, 400, "var(--ea-ink-2)", "end"))
    rows = [(w["length"], "var(--ea-ink-3)", f"{HEADER_BYTES}"),
            (w["header"], "var(--ea-ink-3)", '{"conv1.weight": {...}, ...}'),
            (w["numbers"], "var(--ea-param)", f"{number(NUMBER_BYTES, 0, lang)} bytes")]
    yy = y + 26
    for k, (label, ink, sample) in enumerate(rows):
        h = 22
        parts.append(_rect(8, yy, 6, h, ink, opacity=0.9 if k == 2 else 0.6))
        parts.append(_t(20, yy + 9, label, 9, 400, "var(--ea-ink-2)"))
        parts.append(_mono(20, yy + 20, sample, 9.5))
        yy += h + 6
    y += 122
    parts.append(_rect(0, y, 330, 92, "var(--ea-page)", "var(--ea-ink-3)"))
    parts.append(_mono(8, y + 16, "config.json", 10.5, weight=600))
    parts.append(_t(322, y + 16, w["config"], 9, 400, "var(--ea-ink-2)", "end"))
    for k, line in enumerate(IDEA_CONFIG[lang]):
        parts.append(_mono(14, y + 32 + 13 * k, line, 9.5, "var(--ea-ink-2)"))
    height = y + 100
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
    from everything_ai import check
    try:
        return check.with_time_limit(make_, PICTURE_SECONDS)
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _shown(v, n: int = 20) -> str:
    return _short(v if isinstance(v, str) else repr(v), n)


_YOURS_TEXT = {
    "en": {
        "header": {"title": "Your header for the digit reader",
                   "intro": "The picture calls your `header(C.STATE)` and shows the place of each of the 10 tensors among the bytes of the numbers: where it begins, and where the next one may begin.",
                   "tensor": "tensor", "yours": "your offsets", "course": "the course's",
                   "fail": "Your `header` stops on `C.STATE`, or gives no dict, so the picture has no places to show yet. The check above says why.",
                   "right": "Each tensor begins where the one before ends, and the last ends at byte {end}: {n} numbers of 4 bytes, with no hole and no overlap.",
                   "wrong": "The offsets in purple differ from the course's. Does each tensor take 4 bytes per number, and begin where the one before ends?",
                   "alt": "The 10 tensors of the digit reader, each with the offsets your header gives and the offsets of the course."},
        "read": {"title": "The numbers you read",
                 "intro": "The picture calls your `read_tensor(C.SORTED, \"conv1.weight\")`, where `conv1.weight` comes after `conv1.bias`, and draws the 72 numbers it gives as 8 grids of 3 by 3, the shape `[8, 1, 3, 3]`: orange for a positive weight, gray for a negative one, and stronger for a larger one.",
                 "yours": "yours", "course": "the course's",
                 "fail": "Your `read_tensor` stops on `C.SORTED`, or gives no tensor of shape [8, 1, 3, 3], so the picture has nothing to draw yet. The check above says why.",
                 "right": "The 8 grids as the digit reader learned them, out of the bytes of the file: each one looks for its own small stroke in the pixels.",
                 "wrong": "These numbers are not the weights of the reader. Do the bytes you take begin after the 8 bytes and the header?",
                 "alt": "The 72 numbers of conv1.weight as 8 grids of 3 by 3, as your function read them from the file, beside the numbers of the course."},
    },
    "nl": {
        "header": {"title": "Je header voor de cijferlezer",
                   "intro": "De afbeelding roept je `header(C.STATE)` aan en laat de plek van elk van de 10 tensors tussen de bytes van de getallen zien: waar de tensor begint, en waar de volgende mag beginnen.",
                   "tensor": "tensor", "yours": "jouw offsets", "course": "van de course",
                   "fail": "Je `header` stopt bij `C.STATE`, of geeft geen dict, dus valt er nog geen plek te laten zien. De check hierboven zegt waarom.",
                   "right": "Elke tensor begint waar de vorige eindigt, en de laatste eindigt bij byte {end}: {n} getallen van 4 bytes, zonder gat en zonder overlap.",
                   "wrong": "De paarse offsets verschillen van de offsets van de course. Neemt elke tensor 4 bytes per getal, en begint elke tensor waar de vorige eindigt?",
                   "alt": "De 10 tensors van de cijferlezer, elk met de offsets uit je header en de offsets van de course."},
        "read": {"title": "De getallen die je las",
                 "intro": "De afbeelding roept je `read_tensor(C.SORTED, \"conv1.weight\")` aan, waarin `conv1.weight` na `conv1.bias` komt, en tekent de 72 getallen als 8 roosters van 3 bij 3, de shape `[8, 1, 3, 3]`: oranje voor een positieve weight, grijs voor een negatieve, en sterker voor een grotere.",
                 "yours": "van jou", "course": "van de course",
                 "fail": "Je `read_tensor` stopt bij `C.SORTED`, of geeft geen tensor met shape [8, 1, 3, 3], dus valt er nog niets te tekenen. De check hierboven zegt waarom.",
                 "right": "De 8 roosters zoals de cijferlezer deze getallen leerde, uit de bytes van het bestand: elk rooster zoekt een eigen klein streepje in de pixels.",
                 "wrong": "Deze getallen zijn niet de weights van de cijferlezer. Beginnen de bytes die je neemt na de 8 bytes en de header?",
                 "alt": "De 72 getallen van conv1.weight als 8 roosters van 3 bij 3, zoals je functie de getallen uit het bestand las, naast de getallen van de course."},
    },
}


def _offsets(h, name):
    """The data_offsets of one tensor in a header the learner made, or None when it has none."""
    try:
        got = h[name]["data_offsets"]
        return [int(got[0]), int(got[1])] if len(got) == 2 else None
    except Exception:   # noqa: BLE001   any other shape of header: nothing to show for this tensor
        return None


def header_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["header"]
    fn = _fn(seen, "header")
    got = _call(lambda: fn(copy.deepcopy(STATE))) if fn is not None else _STOPPED
    if got is _STOPPED or not isinstance(got, dict):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["fail"], _LOSS_UI), alt=w["alt"])
    want = header(STATE_DICT)
    parts = [_t(0, 12, w["tensor"], 9, 400, "var(--ea-ink-2)"), _t(112, 12, w["yours"], 9, 400, "var(--ea-ink-2)"),
             _t(222, 12, w["course"], 9, 400, "var(--ea-ink-2)")]
    right = True
    for k, name in enumerate(want):
        y = 30 + 16 * k
        mine = _offsets(got, name)
        ok = mine == want[name]["data_offsets"]
        right = right and ok
        parts.append(_line(0, y - 12, 330, y - 12, "var(--ea-line)", 0.6))
        parts.append(_mono(0, y, name, 9.5))
        shown = "—" if mine is None else f"[{number(mine[0], 0, lang)}, {number(mine[1], 0, lang)}]"
        parts.append(_mono(112, y, _short(shown, 18), 9.5, "var(--ea-ink)" if ok else "var(--ea-loss-text)", weight=600))
        course = want[name]["data_offsets"]
        parts.append(_mono(222, y, f"[{number(course[0], 0, lang)}, {number(course[1], 0, lang)}]", 9.5, "var(--ea-ink-2)"))
    svg = _svg(30 + 16 * len(want) - 6, w["alt"], parts)
    note = (_note(w["right"].format(end=number(NUMBER_BYTES, 0, lang), n=number(PARAMETERS, 0, lang))) if right
            else _note(w["wrong"], _LOSS_UI))
    return yours.figure(w["title"], _note(w["intro"]) + svg + note, alt=w["alt"])


def _grids(t, x0: float, y0: float, scale: float) -> list:
    """The 72 numbers of conv1.weight as 8 grids of 3 by 3 squares: orange for a positive weight, gray for a negative one."""
    parts = []
    cell = 9.0
    for k in range(8):
        kx = x0 + k * (3 * cell + 9)
        for i in range(3):
            for j in range(3):
                v = float(t[k, 0, i, j])
                ink = "var(--ea-param)" if v >= 0 else "var(--ea-ink-2)"
                a = min(abs(v) / scale, 1.0) if np.isfinite(v) else 1.0
                parts.append(_rect(kx + j * cell, y0 + i * cell, cell - 1, cell - 1, ink, opacity=0.08 + 0.92 * a))
    return parts


def read_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["read"]
    fn = _fn(seen, "read_tensor")
    got = _call(lambda: fn(SORTED, "conv1.weight")) if fn is not None else _STOPPED
    ok_shape = isinstance(got, torch.Tensor) and tuple(got.shape) == (8, 1, 3, 3)
    if got is _STOPPED or not ok_shape:
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["fail"], _LOSS_UI), alt=w["alt"])
    want = STATE_DICT["conv1.weight"]
    scale = float(want.abs().max())
    got = got.detach().float()
    parts = [_t(0, 12, w["yours"], 9, 400, "var(--ea-ink-2)")] + _grids(got, 0, 18, scale)
    parts += [_t(0, 62, w["course"], 9, 400, "var(--ea-ink-2)")] + _grids(want, 0, 68, scale)
    svg = _svg(98, w["alt"], parts)
    right = bool(torch.allclose(got, want, atol=1e-6, equal_nan=False))
    note = _note(w["right"]) if right else _note(w["wrong"], _LOSS_UI)
    return yours.figure(w["title"], _note(w["intro"]) + svg + note, alt=w["alt"])


# ======== the demo: the bytes of the digit reader ========================================================================

_HEAD = header(STATE_DICT)
_JSON = header_text(STATE_DICT)


def stops() -> list:
    """The bytes the slider stops at: the length, the header (its start and its middle), the first byte of each tensor,
    a byte in the middle of the largest one, and the last byte of the file."""
    out = [0, 8, 8 + HEADER_BYTES // 2]
    for name, h in _HEAD.items():
        out.append(START + h["data_offsets"][0])
        if name == "hidden.weight":
            out.append(START + (h["data_offsets"][0] + h["data_offsets"][1]) // 2 // 4 * 4)
    out.append(len(FILE) - 1)
    return sorted(set(out))


STOPS = stops()


def part(k: int) -> tuple:
    """What byte k of C.FILE belongs to: ("length",), ("header", i) with i the place in the header, ("padding",) for
    the spaces after the JSON, or ("tensor", name, index, value) with the number the byte is part of."""
    if k < 8:
        return ("length",)
    if k < START:
        i = k - 8
        return ("padding",) if i >= len(_JSON.rstrip(b" ")) else ("header", i)
    offset = k - START
    for name, h in _HEAD.items():
        begin, end = h["data_offsets"]
        if begin <= offset < end:
            flat = (offset - begin) // 4
            t = STATE_DICT[name]
            index = [int(i) for i in np.unravel_index(flat, tuple(t.shape))]
            return ("tensor", name, index, float(t.flatten()[flat]))
    raise ValueError(k)


DEMO_TEXT = {
    "en": {"title": "byte {k} of {size}", "parts": {"length": "length", "header": "header", "numbers": "numbers"},
           "around": "the 16 bytes from byte {a}",
           "alt": "The file of the digit reader as a bar of {size} bytes: 8 bytes of length, {head} bytes of header, then the numbers of the 10 tensors. A mark shows byte {k}, and below it the 16 bytes around it in hex.",
           "says": {
               "length": "Byte **{k}** is one of the first 8 bytes: together they hold the length of the header, **{head}**, with the lowest byte first.",
               "header": "Byte **{k}** is in the JSON header, character {i} of {head}: `{around}`.",
               "padding": "Byte **{k}** is one of the spaces after the JSON, which bring the header to {head} bytes, a multiple of 8.",
               "tensor": "Byte **{k}** is part of **{name}**: the number at index `{index}`, **{value}**, in the 4 bytes from byte {first}.",
           }},
    "nl": {"title": "byte {k} van {size}", "parts": {"length": "lengte", "header": "header", "numbers": "getallen"},
           "around": "de 16 bytes vanaf byte {a}",
           "alt": "Het bestand van de cijferlezer als een balk van {size} bytes: 8 bytes lengte, {head} bytes header, dan de getallen van de 10 tensors. Een markering wijst byte {k} aan, en daaronder staan de 16 bytes eromheen in hex.",
           "says": {
               "length": "Byte **{k}** is een van de eerste 8 bytes: samen bevatten deze bytes de lengte van de header, **{head}**, met de laagste byte eerst.",
               "header": "Byte **{k}** zit in de JSON-header, teken {i} van {head}: `{around}`.",
               "padding": "Byte **{k}** is een van de spaties na de JSON, waarmee de header op {head} bytes komt, een veelvoud van 8.",
               "tensor": "Byte **{k}** hoort bij **{name}**: het getal op index `{index}`, **{value}**, in de 4 bytes vanaf byte {first}.",
           }},
}


def say(k: int, lang: str) -> str:
    w = DEMO_TEXT[lang]["says"]
    p = part(k)
    head = number(HEADER_BYTES, 0, lang)
    kk = number(k, 0, lang)
    if p[0] in ("length", "padding"):
        return w[p[0]].format(k=kk, head=head)
    if p[0] == "header":
        i = p[1]
        around = _JSON[max(0, i - 8):i + 9].decode().replace("`", "'")
        return w["header"].format(k=kk, i=number(i + 1, 0, lang), head=head, around=around)
    _, name, index, value = p
    t = STATE_DICT[name]
    flat = int(np.ravel_multi_index(index, tuple(t.shape)))
    first = START + _HEAD[name]["data_offsets"][0] + 4 * flat
    return w["tensor"].format(k=kk, name=name, index=index, value=number(value, 4, lang), first=number(first, 0, lang))


def frame(value: float, lang: str) -> tuple[str, str]:
    """The file as a bar with its parts, a mark at byte k, and the 16 bytes around it in hex."""
    k = min(STOPS, key=lambda s: abs(s - value))
    w = DEMO_TEXT[lang]
    size = len(FILE)
    parts = [_t(0, 14, w["title"].format(k=number(k, 0, lang), size=number(size, 0, lang)), 10.5, 600)]
    # the bar: the length and the header are too small to see at this scale, so they get a fixed width
    short, fixed = 8.0, 44.0

    def x_of(byte: int) -> float:
        if byte < 8:
            return byte / 8 * short
        if byte < START:
            return short + (byte - 8) / HEADER_BYTES * (fixed - short)
        return fixed + (byte - START) / (size - START) * (330 - fixed)

    y = 26
    parts.append(_rect(0, y, short, 18, "var(--ea-ink-2)", opacity=0.9))
    parts.append(_rect(short, y, fixed - short, 18, "var(--ea-ink-3)", opacity=0.45))
    for j, (name, h) in enumerate(_HEAD.items()):
        a, b = x_of(START + h["data_offsets"][0]), x_of(START + h["data_offsets"][1])
        parts.append(_rect(a, y, b - a, 18, "var(--ea-param)", opacity=0.35 if j % 2 else 0.6))
    parts.append(_t(0, y + 30, w["parts"]["header"], 8.5, 400, "var(--ea-ink-2)"))
    parts.append(_t(fixed + 4, y + 30, w["parts"]["numbers"], 8.5, 400, "var(--ea-ink-2)"))
    mx = x_of(k)
    tip = min(max(mx, 5.0), 325.0)   # the mark stays whole at either end of the bar
    parts.append(f'<path d="M{tip - 5:.1f},{y - 6} L{tip + 5:.1f},{y - 6} L{mx:.1f},{y:.1f} Z" style="fill:var(--ea-ink)"/>')
    parts.append(_line(mx, y, mx, y + 18, "var(--ea-ink)", 1.4))
    # the 16 bytes from a, in hex, byte k in bold
    a = max(0, min(k - 7, size - 16))
    y2 = y + 52
    parts.append(_t(0, y2, w["around"].format(a=number(a, 0, lang)), 9, 400, "var(--ea-ink-2)"))
    chars = []
    for i in range(a, a + 16):
        chars.append((f"{FILE[i]:02x}", "var(--ea-ink)" if i == k else "var(--ea-ink-3)", 700 if i == k else 400))
        chars.append((" ", "var(--ea-ink-3)", 400))
    parts.append(_spans(0, y2 + 16, chars, 10.5))
    p = part(k)
    tag = {"length": w["parts"]["length"], "header": w["parts"]["header"], "padding": w["parts"]["header"]}.get(p[0])
    parts.append(_mono(0, y2 + 34, tag if tag else f"{p[1]} {p[2]}", 10, weight=600))
    alt = w["alt"].format(size=number(size, 0, lang), head=number(HEADER_BYTES, 0, lang), k=number(k, 0, lang))
    return _svg(y2 + 42, alt, parts), say(k, lang)


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

# header: the loop is there; the learner writes where a tensor ends.
_H_HEAD = ("def header(state):\n"
           "    out = {}\n"
           "    start = 0\n"
           "    for name, t in state.items():\n")
_H_END = ("        out[name] = {\"dtype\": \"F32\", \"shape\": list(t.shape), \"data_offsets\": [start, end]}\n"
          "        start = end\n"
          "    return out\n")


def _h(line: str) -> str:
    return _H_HEAD + "        end = " + line + "\n" + _H_END


_H_START = _h("...")
_H = _h("start + t.numel() * 4")
_H_NUMBERS = _h("start + t.numel()")
_H_ALONE = _h("t.numel() * 4")
_H_INCLUSIVE = _h("start + t.numel() * 4 - 1")
_H_FIRST_AXIS = _h("start + len(t) * 4")
_H_DOUBLE = _h("start + t.numel() * 8")
_H_MEMORY = _h("start + t.nbytes")

# read_tensor: the header is read; the learner takes the bytes of one tensor.
_R_HEAD = ("import json\n"
           "import torch\n"
           "\n\n"
           "def read_tensor(data, name):\n"
           "    n = int.from_bytes(data[:8], \"little\")\n"
           "    header = json.loads(data[8:8 + n])\n"
           "    start, end = header[name][\"data_offsets\"]\n")
_R_END = "    return torch.frombuffer(bytearray(numbers), dtype=torch.float32).reshape(header[name][\"shape\"])\n"


def _r(line: str) -> str:
    return _R_HEAD + "    numbers = " + line + "\n" + _R_END


_R_START = _r("...")
_R = _r("data[8 + n + start:8 + n + end]")
_R_FROM_FILE = _r("data[start:end]")
_R_NO_HEADER = _r("data[8 + start:8 + end]")
_R_NO_LENGTH = _r("data[n + start:n + end]")


def _header_shown(v) -> str:
    """A header in a check message: its first two tensors, not all 10 of the digit reader."""
    if isinstance(v, dict):
        try:
            items = list(v.items())
            text_ = json.dumps(dict(items[:2]))
            return f"`{text_[:-1] + ', ...}' if len(items) > 2 else text_}`"
        except (TypeError, ValueError):   # something JSON cannot write: shown as Python shows it
            pass
    return f"`{_short(repr(v), 200)}`"


def _tensor_shown(v) -> str:
    """A tensor in a check message: its first numbers and its shape, not all 72 or 50,176 of them."""
    if isinstance(v, torch.Tensor):
        flat = v.detach().flatten().tolist()
        first = ", ".join(f"{x:.4g}" for x in flat[:3]) + (", ..." if len(flat) > 3 else "")
        return f"`tensor([{first}])`, shape `{list(v.shape)}`"
    return f"`{v!r}`"


EXERCISES = {
    "header": FunctionBox(
        name="header",
        function="header",
        start=_H_START,
        solution=_H,
        visible=[(STATE,)],
        hidden=[(SMALL,), (CUBE,), (DOUBLE,), (NONE,)],
        first=[_DOTS],
        picture=header_picture,
        show=_header_shown,
        mistakes=[
            Mistake("numbers", wrong=_H_NUMBERS,
                    en="`{call}` gives {got}, and it should give {want}: your offsets count numbers, and `data_offsets` counts bytes. How many bytes does one float32 take?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je offsets tellen getallen, en `data_offsets` telt bytes. Hoeveel bytes neemt één float32?"),
            Mistake("alone", wrong=_H_ALONE,
                    en="`{call}` gives {got}, and it should give {want}: each tensor of yours ends at its own size, as if it began at byte 0, so after the first one an end is lower than its begin. Where does a tensor begin, and where does it end then?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke tensor eindigt bij jou na het eigen aantal bytes, alsof de tensor bij byte 0 begint, dus ligt na de eerste tensor het eind lager dan het begin. Waar begint een tensor, en waar eindigt de tensor dan?"),
            Mistake("inclusive", wrong=_H_INCLUSIVE,
                    en="`{call}` gives {got}, and it should give {want}: your end is the last byte of the tensor. In safetensors the end is the first byte after it, as in a slice `data[begin:end]`, so the next tensor begins at that end. Can you leave out the `- 1`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je eind is de laatste byte van de tensor. In safetensors is het eind de eerste byte erna, zoals bij een slice `data[begin:end]`, dus begint de volgende tensor bij dat eind. Kun je de `- 1` weglaten?"),
            Mistake("first-axis", wrong=_H_FIRST_AXIS,
                    en="`{call}` gives {got}, and it should give {want}: `len(t)` is the length of the first axis only, 8 for `conv1.weight` of shape `[8, 1, 3, 3]`, which holds 72 numbers. Which method of a tensor counts all its numbers?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `len(t)` is alleen de lengte van de eerste as, 8 bij `conv1.weight` met shape `[8, 1, 3, 3]`, met 72 getallen. Welke methode van een tensor telt alle getallen?"),
            Mistake("memory", wrong=_H_MEMORY, state="almost",
                    en="Right for `C.STATE`. But `{call}` gives {got}, and it should give {want}: `x` is float64 in memory, 8 bytes per number, and the file holds every tensor as float32, as the header says with `\"F32\"`. Does your end count the bytes in the file?",
                    nl="Goed voor `C.STATE`. Maar `{call}` geeft {got}, en het moet {want} zijn: `x` is float64 in het geheugen, 8 bytes per getal, en het bestand bevat elke tensor als float32, zoals de header zegt met `\"F32\"`. Telt je eind de bytes in het bestand?"),
            Mistake("double", wrong=_H_DOUBLE,
                    en="`{call}` gives {got}, and it should give {want}: 8 bytes per number is float64. The header says `\"F32\"`: how many bytes does a float32 take?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: 8 bytes per getal is float64. De header zegt `\"F32\"`: hoeveel bytes neemt een float32?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with where the tensor `t` ends: its begin `start`, plus the bytes its numbers take.",
                      "nl": "Vervang de `...` door waar de tensor `t` eindigt: het begin `start`, plus de bytes van de getallen."},
            "arguments": {"en": "The check calls `{call}`, with one state_dict, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(state)`?",
                          "nl": "De check roept `{call}` aan, met één state_dict, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(state)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does each tensor end at its begin plus 4 bytes for each of its numbers?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Eindigt elke tensor bij het begin plus 4 bytes voor elk getal?"},
            "almost": {"en": "Right for `C.STATE`. But `{call}` gives {got}, and it should give {want}. Does your end take 4 bytes for each number of `t`, whatever its shape?",
                       "nl": "Goed voor `C.STATE`. Maar `{call}` geeft {got}, en het moet {want} zijn. Neemt je eind 4 bytes voor elk getal van `t`, welke shape de tensor ook heeft?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return out`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return out`?"},
            "correct": {"en": "Right in all {n} cases. The 10 tensors of the digit reader take bytes 0 to 206,592, one after another. Where in the file do these bytes begin?",
                        "nl": "Goed in alle {n} gevallen. De 10 tensors van de cijferlezer nemen de bytes 0 tot 206.592, de ene na de andere. Waar in het bestand beginnen deze bytes?"},
        },
    ),
    "read-tensor": FunctionBox(
        name="read-tensor",
        function="read_tensor",
        start=_R_START,
        solution=_R,
        visible=[(FILE, "conv1.weight")],
        hidden=[(SORTED, "conv1.weight"), (SORTED, "out.bias"), (FILE, "hidden.weight"), (SMALL_FILE, "b"),
                (SMALL_FILE, "w")],
        first=[_DOTS],
        picture=read_picture,
        show=_tensor_shown,
        mistakes=[
            Mistake("empty", error=lambda e: isinstance(e, ValueError) and "buffer length (0)" in str(e),
                    en="With `{call}`, your slice of `data` is empty, so `torch.frombuffer` gets no bytes to read. Your begin skips the 8 bytes and the header: does your end skip them too?",
                    nl="Met `{call}` is je slice van `data` leeg, dus krijgt `torch.frombuffer` geen bytes om te lezen. Je begin slaat de 8 bytes en de header over: slaat je eind deze bytes ook over?"),
            Mistake("length", error=lambda e: ("multiple of element size" in str(e) or "is invalid for input of size" in str(e)),
                    en="With `{call}`, your slice holds another number of bytes than the tensor: it takes `end - start` bytes, 4 for each number. Do your begin and your end both skip the 8 bytes and the header, and both add the offset of the tensor?",
                    nl="Met `{call}` bevat je slice een ander aantal bytes dan de tensor: de tensor neemt `end - start` bytes, 4 voor elk getal. Slaan je begin en je eind allebei de 8 bytes en de header over, en tellen allebei de offset van de tensor erbij op?"),
            Mistake("from-file", wrong=_R_FROM_FILE,
                    en="`{call}` gives {got}, and it should give {want}: your bytes are counted from the start of the file, but `data_offsets` counts from the first byte after the header. What comes before that byte?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je bytes tellen vanaf het begin van het bestand, maar `data_offsets` telt vanaf de eerste byte na de header. Wat komt er voor deze byte?"),
            Mistake("no-header", wrong=_R_NO_HEADER,
                    en="`{call}` gives {got}, and it should give {want}: you skip the first 8 bytes, but not the header after them. How long is the header, and which name holds that length?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je slaat de eerste 8 bytes over, maar niet de header erna. Hoe lang is de header, en welke naam bevat deze lengte?"),
            Mistake("no-length", wrong=_R_NO_LENGTH,
                    en="`{call}` gives {got}, and it should give {want}: you skip the header, but not the 8 bytes before it that hold its length. Where does the header begin?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je slaat de header over, maar niet de 8 bytes voor de header, met de lengte. Waar begint de header?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the bytes of the tensor `name`: its `start` and `end` count from the first byte after the 8 bytes and the header.",
                      "nl": "Vervang de `...` door de bytes van de tensor `name`: de `start` en het `end` tellen vanaf de eerste byte na de 8 bytes en de header."},
            "arguments": {"en": "The check calls `{call}`, with the bytes of a file and the name of a tensor, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(data, name)`?",
                          "nl": "De check roept `{call}` aan, met de bytes van een bestand en de naam van een tensor, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(data, name)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Do the bytes you take begin after the 8 bytes and the header?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Beginnen de bytes die je neemt na de 8 bytes en de header?"},
            "almost": {"en": "Right for `C.FILE`. But `{call}` gives {got}, and it should give {want}. `C.SORTED` holds the same weights with a shorter header. Does your slice use the length `n` of the header of the file it gets?",
                       "nl": "Goed voor `C.FILE`. Maar `{call}` geeft {got}, en het moet {want} zijn. `C.SORTED` bevat dezelfde weights met een kortere header. Gebruikt je slice de lengte `n` van de header van het bestand dat je functie krijgt?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with the `return` line?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met de regel met `return`?"},
            "correct": {"en": "Right in all {n} cases. You read the weights of the digit reader without running anything that came with the file. What does the file not tell you about how to use them?",
                        "nl": "Goed in alle {n} gevallen. Je las de weights van de cijferlezer zonder iets uit te voeren wat met het bestand meekwam. Wat vertelt het bestand je niet over hoe je deze weights gebruikt?"},
        },
    ),
}

# The lesson is done when these are correct. "pickle" and "package" are the choice questions at the end.
REQUIRED = ["header", "read-tensor", "pickle", "package"]

CASES = [
    ("header", _H_START, "empty", None),
    ("header", _H, "correct", None),
    ("header", _h("start + 4 * t.numel()"), "correct", None),
    ("header", _h("start + t.nelement() * 4"), "correct", None),
    ("header", _h("start + ... * 4"), "wrong", "dots-left"),
    ("header", _H_NUMBERS, "wrong", "numbers"),
    ("header", _H_ALONE, "wrong", "alone"),
    ("header", _H_INCLUSIVE, "wrong", "inclusive"),
    ("header", _H_FIRST_AXIS, "wrong", "first-axis"),
    ("header", _H_DOUBLE, "wrong", "double"),
    ("header", _h("start + 4"), "wrong", None),
    ("header", _H_MEMORY, "almost", "memory"),
    ("header", _h("start + t.numel() * t.element_size()"), "almost", "memory"),
    ("header", _h("start + (t.numel() * 4 if t.dim() != 3 else 4)"), "almost", None),
    ("read-tensor", _R_START, "empty", None),
    ("read-tensor", _R, "correct", None),
    ("read-tensor", _r("data[8 + n:][start:end]"), "correct", None),
    ("read-tensor", _r("data[8 + n + start:...]"), "wrong", "dots-left"),
    ("read-tensor", _R_FROM_FILE, "wrong", "from-file"),
    ("read-tensor", _R_NO_HEADER, "wrong", "no-header"),
    ("read-tensor", _R_NO_LENGTH, "wrong", "no-length"),
    ("read-tensor", _r(f"data[{START} + start:{START} + end]"), "almost", None),
    ("read-tensor", _r("data[8 + 824 + start:8 + 824 + end]"), "almost", None),
    ("read-tensor", _r("data[8 + n + start:8 + n + start + 4]"), "error", "length"),
    ("read-tensor", _r("data[8 + n + start:end]"), "error", "empty"),
    ("read-tensor", _r("data[8 + n + start:8 + n + end - 1]"), "error", "length"),
    ("read-tensor", _r("data[8 + n + start:]"), "error", "length"),
    ("read-tensor", _r("data[8 + n:8 + n + end]"), "error", "length"),
]
