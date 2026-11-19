# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-pickle-safetensors: a small model that a stranger offers the bike shop Spoke & Chain, as
four files the course makes itself, in memory: the state_dict as torch.save writes it, the same with one object more
whose __reduce__ makes pickle call print while the file is loaded (harmless: it only prints a line), the whole model,
and the same numbers as a safetensors file. The pickle part of a .pt file is read with pickletools, which lists what a pickle
would import without running anything; a safetensors file is read by hand: 8 bytes with the length of a JSON header,
the header, then the numbers."""
import contextlib
import copy
import html
import io
import json
import pickle
import pickletools
import re
import zipfile

import torch   # imported once, when the lesson opens

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

# --- the stranger's model and its files --------------------------------------------------------------------------------

NOTE = "This line ran while the file was loaded."   # what the hidden object prints, and all it does


class Surprise:
    """An object that pickle rebuilds by calling print(NOTE): what __reduce__ returns is what unpickling calls."""

    def __reduce__(self):
        return (print, (NOTE,))


def stranger_model() -> torch.nn.Linear:
    """The stranger's model: one layer from 4 numbers about a review to 3 topics, the same every time."""
    torch.manual_seed(4417)
    return torch.nn.Linear(4, 3)


def saved(obj) -> bytes:
    """What torch.save writes for obj, as bytes."""
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()


def to_safetensors(tensors: dict, metadata: dict | None = None) -> bytes:
    """A safetensors file of float32 tensors, as the format describes it: 8 bytes with the length N of the header,
    little-endian; N bytes of JSON, padded with spaces to a multiple of 8; then the numbers of each tensor in turn."""
    header, buffer = {}, b""
    if metadata is not None:
        header["__metadata__"] = metadata
    for name, t in tensors.items():
        raw = t.detach().to(torch.float32).contiguous().numpy().tobytes()
        header[name] = {"dtype": "F32", "shape": list(t.shape), "data_offsets": [len(buffer), len(buffer) + len(raw)]}
        buffer += raw
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    return len(text).to_bytes(8, "little") + text + buffer


def pickle_part(data: bytes) -> bytes:
    """The pickle inside a file that torch.save wrote: torch.save writes a zip, and its data.pkl is the pickle."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        name = next(n for n in z.namelist() if n.endswith("/data.pkl"))
        return z.read(name)


_TEXTS = {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE", "SHORT_BINSTRING", "BINSTRING", "STRING"}


def _module(name: str) -> str:
    """Pickle protocol 2, which torch.save uses, writes Python 2's name __builtin__; pickle reads it as builtins."""
    return "builtins." + name[len("__builtin__."):] if name.startswith("__builtin__.") else name


def imports(data: bytes) -> list:
    """What a pickle would import, as "module.name", in the order it comes, read with pickletools.genops: that reads
    the instructions one by one and runs none of them. GLOBAL holds the name itself; STACK_GLOBAL takes the last two
    texts, which may come back from the memo. data is a pickle, or a file that torch.save wrote: a zip with one."""
    pkl = pickle_part(data) if data[:2] == b"PK" else data
    found, stack, memo = [], [], {}
    for op, arg, _ in pickletools.genops(pkl):
        if op.name in _TEXTS:
            stack.append(arg)
        elif op.name == "MEMOIZE":
            memo[len(memo)] = stack[-1] if stack else None
        elif op.name in ("PUT", "BINPUT", "LONG_BINPUT"):
            memo[arg] = stack[-1] if stack else None
        elif op.name in ("GET", "BINGET", "LONG_BINGET"):
            stack.append(memo.get(arg))
        elif op.name in ("GLOBAL", "INST"):
            found.append(_module(arg.replace(" ", ".")))
            stack.append(None)
        elif op.name == "STACK_GLOBAL":
            found.append(_module(f"{stack[-2]}.{stack[-1]}"))
            stack.append(None)
        else:
            stack.append(None)
    return found


def instructions(pkl: bytes) -> list:
    """The instructions of a pickle as (place, name, argument), for the picture."""
    return [(pos, op.name, arg) for op, arg, pos in pickletools.genops(pkl)]


_MODEL = stranger_model()
STATE = {k: v.detach().clone() for k, v in _MODEL.state_dict().items()}   # weight (3, 4) and bias (3,)
STATE_PT = saved(_MODEL.state_dict())                                       # trail-sorter.pt
SURPRISE_PT = saved({"weight": STATE["weight"], "bias": STATE["bias"], "note": Surprise()})   # trail-sorter-v2.pt
WHOLE_PT = saved(_MODEL)                                                    # the whole model, class and all
SAFE = to_safetensors(STATE)                                                # trail-sorter.safetensors
FILES = {"state": "trail-sorter.pt", "surprise": "trail-sorter-v2.pt", "whole": "trail-sorter-model.pt",
         "safe": "trail-sorter.safetensors"}
DATA = {"state": STATE_PT, "surprise": SURPRISE_PT, "whole": WHOLE_PT, "safe": SAFE}
ALLOWED = ["collections.OrderedDict", "torch._utils._rebuild_tensor_v2", "torch.FloatStorage"]   # what a state_dict needs


# --- the course's own versions of the exercises ------------------------------------------------------------------------

def read_header(data):
    # the JSON header of a safetensors file as a dictionary; data holds the bytes of the whole file
    n = int.from_bytes(data[:8], "little")
    return json.loads(data[8:8 + n])


def flagged(names, allowed):
    # the imports in names that are not on the allow-list, each once, in the order they come
    found = []
    for name in names:
        if name not in allowed and name not in found:
            found.append(name)
    return found


def no_holes(spans, size):
    # True when spans, [start, stop] pairs sorted by start, fill a byte buffer of size bytes exactly
    end = 0
    for start, stop in spans:
        if start != end:
            return False
        end = stop
    return end == size


def spans(header: dict) -> list:
    """The data_offsets of the tensors in a header, sorted by start."""
    return sorted(v["data_offsets"] for k, v in header.items() if k != "__metadata__")


def buffer_size(data: bytes) -> int:
    """The number of bytes after the header of a safetensors file."""
    return len(data) - 8 - int.from_bytes(data[:8], "little")


# --- loading, for the worked example and the demo --------------------------------------------------------------------

def load(data: bytes, weights_only: bool) -> dict:
    """What happens when torch.load reads data: what it printed while loading, and the names it gave back, or the
    last line of its error. Only ever for the course's own files: weights_only=False runs what a pickle asks."""
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            got = torch.load(io.BytesIO(data), weights_only=weights_only)
        return {"printed": out.getvalue().strip(), "names": list(got) if isinstance(got, dict) else [type(got).__name__],
                "error": None}
    except Exception as e:   # noqa: BLE001   the picture says what stopped it
        return {"printed": out.getvalue().strip(), "names": None, "error": _error_line(e)}


def _error_line(e: Exception) -> str:
    """The line of torch's error that says why: the Unsupported global, else the first line."""
    lines = [line.strip() for line in str(e).splitlines() if line.strip()]
    for line in lines:
        if line.startswith(("WeightsUnpickler error: Unsupported", "Unsupported")):
            return line.split(" was not an allowed global", 1)[0].replace("WeightsUnpickler error: ", "")
    return f"{type(e).__name__}: {lines[0] if lines else ''}"[:160]


def worked_example() -> list:
    """The lines the worked example prints, for the lesson's test."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        print("imports:", imports(SURPRISE_PT))
        loaded = torch.load(io.BytesIO(SURPRISE_PT), weights_only=False)
        print("loaded:", list(loaded), "note:", loaded["note"])
        try:
            torch.load(io.BytesIO(SURPRISE_PT), weights_only=True)
        except pickle.UnpicklingError as e:
            print("refused:", _error_line(e))
    return out.getvalue().splitlines()


# --- the files the checks of read_header use -------------------------------------------------------------------------

def _deep_state() -> dict:
    """A model with three layers, so that its header is longer than 255 bytes: one byte cannot hold its length."""
    torch.manual_seed(29)
    net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 8), torch.nn.ReLU(),
                              torch.nn.Linear(8, 3))
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


TAGGED = to_safetensors(STATE, metadata={"format": "pt"})   # the same numbers, with a __metadata__ key
EMPTY = to_safetensors({})                                   # no tensors: the header is {} and seven spaces
DEEP = to_safetensors(_deep_state())                         # six tensors


def header_length(data: bytes) -> int:
    """The number in the first 8 bytes of a safetensors file."""
    return int.from_bytes(data[:8], "little")


# --- the pictures ----------------------------------------------------------------------------------------------------
# Every name and every value of the learner is shown through html.escape.

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.15rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;vertical-align:top")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"
_MONO = "font-family:var(--ea-font-code);font-size:12px;overflow-wrap:anywhere"
_WHOLE = "white-space:nowrap"   # a file name, an import or an instruction, never broken in the middle
_LINE = ("font-family:var(--ea-font-code);font-size:12.5px;line-height:1.5;white-space:pre-wrap;overflow-wrap:anywhere;"
         "background:var(--ea-inset);border:1px solid var(--ea-line);border-radius:4px;padding:0.3rem 0.5rem;margin:0.15rem 0")


def _note(text: str) -> str:
    return f'<div style="{_UI}">{html.escape(text)}</div>'


def _mono(text: str) -> str:
    return f'<span style="{_MONO}">{html.escape(text)}</span>'


def _table(rows: list, head: list) -> str:
    """A table with a head row; each row is a list of (html, style)."""
    top = "".join(f'<th style="{_HEAD}">{html.escape(h)}</th>' for h in head)
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{";" + style if style else ""}">{cell}</td>'
                                    for cell, style in row) + "</tr>" for row in rows)
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.3rem 0">'
            f'<tr>{top}</tr>{body}</table></div>')


def _many(n: int, forms: tuple) -> str:
    """forms[0] for one, forms[1] for more, and forms[2], when there is one, for none; each with {n} for the number."""
    if n == 0 and len(forms) > 2:
        return forms[2].format(n=n)
    return (forms[0] if n == 1 else forms[1]).format(n=n)


def _number(n: int, lang: str) -> str:
    """A whole number as the lesson writes it: 1,893 in English; in Dutch a dot only from five digits on."""
    if lang == "nl":
        return f"{n:,}".replace(",", ".") if n >= 10000 else str(n)
    return f"{n:,}"


_LAYOUT_WORDS = {
    "en": {"title": "Two files with the same 15 numbers",
           "pt": "trail-sorter.pt, {size} bytes: a zip with these files in it",
           "head": ["file in the zip", "bytes", "what it holds"],
           "pkl": "the pickle: {k} instructions that rebuild the state_dict, and the imports they need",
           "data0": "the 12 numbers of weight, 4 bytes each", "data1": "the 3 numbers of bias",
           "rest": "the version, the byte order, an id and other settings, in {k} small files",
           "st": "trail-sorter.safetensors, {size} bytes: three parts",
           "len": "8 bytes", "len_says": "the number {n}: the length of the header",
           "header": "{n} bytes of JSON", "header_says": "the header: for each tensor its dtype, shape and place",
           "buffer": "{n} bytes", "buffer_says": "the numbers: weight from byte 0 to 48, bias from 48 to 60",
           "text": "The header itself:",
           "alt": "trail-sorter.pt is a zip of {size_pt} bytes whose data.pkl holds {k} pickle instructions and whose "
                  "data/0 and data/1 hold the numbers. trail-sorter.safetensors is {size_st} bytes: 8 bytes with the "
                  "number {n}, a JSON header of {n} bytes, and 60 bytes of numbers."},
    "nl": {"title": "Twee bestanden met dezelfde 15 getallen",
           "pt": "trail-sorter.pt, {size} bytes: een zip met deze bestanden erin",
           "head": ["bestand in de zip", "bytes", "wat erin staat"],
           "pkl": "de pickle: {k} instructies die de state_dict weer opbouwen, en de imports die ze nodig hebben",
           "data0": "de 12 getallen van weight, 4 bytes per getal", "data1": "de 3 getallen van bias",
           "rest": "de versie, de bytevolgorde, een id en andere instellingen, in {k} kleine bestanden",
           "st": "trail-sorter.safetensors, {size} bytes: drie delen",
           "len": "8 bytes", "len_says": "het getal {n}: de lengte van de header",
           "header": "{n} bytes JSON", "header_says": "de header: per tensor het dtype, de shape en de plek",
           "buffer": "{n} bytes", "buffer_says": "de getallen: weight van byte 0 tot 48, bias van 48 tot 60",
           "text": "De header zelf:",
           "alt": "trail-sorter.pt is een zip van {size_pt} bytes waarvan data.pkl {k} pickle-instructies bevat en "
                  "data/0 en data/1 de getallen. trail-sorter.safetensors is {size_st} bytes: 8 bytes met het getal "
                  "{n}, een JSON-header van {n} bytes, en 60 bytes getallen."},
}


def _members(data: bytes) -> list:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return [(i.filename, i.file_size) for i in z.infolist()]


def layout_picture(lang: str) -> str:
    """What is inside trail-sorter.pt and trail-sorter.safetensors: the same numbers, kept in two ways."""
    w = _LAYOUT_WORDS[lang]
    members = _members(STATE_PT)
    k = len(instructions(pickle_part(STATE_PT)))
    rows, rest = [], 0
    for name, size in members:
        short = name.split("/", 1)[1]
        if short == "data.pkl":
            rows.append([(_mono(short), ""), (str(size), ""), (html.escape(w["pkl"].format(k=k)), _OFF)])
        elif short in ("data/0", "data/1"):
            rows.append([(_mono(short), ""), (str(size), ""), (html.escape(w["data0" if short == "data/0" else "data1"]), "")])
        else:
            rest += 1
    others = sum(size for name, size in members if name.split("/", 1)[1] not in ("data.pkl", "data/0", "data/1"))
    rows.append([("&hellip;", ""), (str(others), ""), (html.escape(w["rest"].format(k=rest)), "")])
    n = header_length(SAFE)
    part = ("flex:1 1 0;min-width:7rem;padding:0.3rem 0.45rem;border:1px solid var(--ea-line);border-radius:4px;"
            "font-family:var(--ea-font-ui);font-size:0.82rem")
    parts = [(w["len"], w["len_says"].format(n=n), ""),
             (w["header"].format(n=n), w["header_says"], ""),
             (w["buffer"].format(n=len(SAFE) - 8 - n), w["buffer_says"], ";background:var(--ea-param-wash)")]
    bar = ('<div style="display:flex;flex-wrap:wrap;gap:0.3rem;margin:0.3rem 0">'
           + "".join(f'<div style="{part}{extra}"><b>{html.escape(a)}</b><br>{html.escape(b)}</div>'
                     for a, b, extra in parts) + "</div>")
    body = (_note(w["pt"].format(size=_number(len(STATE_PT), lang))) + _table(rows, w["head"])
            + _note(w["st"].format(size=len(SAFE))) + bar
            + _note(w["text"]) + f'<div style="{_LINE}">{html.escape(SAFE[8:8 + n].decode().rstrip())}</div>')
    alt = w["alt"].format(size_pt=_number(len(STATE_PT), lang), k=k, size_st=len(SAFE), n=n)
    return yours.figure(w["title"], body, alt=alt)


_V2_WORDS = {
    "en": {"title": "The end of the pickle in trail-sorter-v2.pt, instruction by instruction",
           "head": ["place", "instruction", "argument", "what pickle does"],
           "GLOBAL": "imports print from builtins (Python 2 called it __builtin__)",
           "BINUNICODE": "puts the text on the stack", "TUPLE1": "makes a tuple of it: the arguments",
           "REDUCE": "calls print with those arguments", "SETITEMS": "puts the keys and values in the dictionary",
           "STOP": "the end of the pickle", "other": "keeps a value for later",
           "after": "pickletools read these instructions without running one of them. The dashed instructions are the "
                    "import and the call.",
           "alt": "The last instructions of the pickle in trail-sorter-v2.pt: {rows}."},
    "nl": {"title": "Het einde van de pickle in trail-sorter-v2.pt, instructie voor instructie",
           "head": ["plek", "instructie", "argument", "wat pickle doet"],
           "GLOBAL": "importeert print uit builtins (Python 2 noemde dat __builtin__)",
           "BINUNICODE": "zet de tekst op de stack", "TUPLE1": "maakt er een tuple van: de argumenten",
           "REDUCE": "roept print aan met die argumenten", "SETITEMS": "zet de keys en values in de dictionary",
           "STOP": "het einde van de pickle", "other": "bewaart een waarde voor later",
           "after": "pickletools las deze instructies zonder er een uit te voeren. De gestreepte instructies zijn de "
                    "import en de aanroep.",
           "alt": "De laatste instructies van de pickle in trail-sorter-v2.pt: {rows}."},
}


def note_instructions() -> list:
    """The instructions of trail-sorter-v2.pt from the key "note" on: the text, the import of print and the call."""
    found = instructions(pickle_part(SURPRISE_PT))
    start = next(i for i, (_, name, arg) in enumerate(found) if name in _TEXTS and arg == "note")
    return found[start + 1:]


def v2_picture(lang: str) -> str:
    """The instructions that pickle runs for the object under "note" in trail-sorter-v2.pt."""
    w = _V2_WORDS[lang]
    rows, said = [], []
    for place, name, arg in note_instructions():
        what = w.get(name, w["other"])
        shown = "" if arg is None else repr(arg)
        rows.append([(str(place), ""), (_mono(name), _WHOLE + (";" + _OFF if name in ("GLOBAL", "REDUCE") else "")),
                     (_mono(shown), ""), (html.escape(what), "")])
        said.append(f"{place} {name} {shown}".strip())
    body = _table(rows, w["head"]) + _note(w["after"])
    return yours.figure(w["title"], body, alt=w["alt"].format(rows="; ".join(said)))


_FILES_WORDS = {
    "en": {"title": "The stranger's four files, before and while they load",
           "head": ["file", "imports, read with pickletools", "torch.load, weights_only=True"],
           "holds": {"state": "the state_dict", "surprise": "the state_dict and one object more",
                     "whole": "the whole model", "safe": "the same numbers as safetensors"},
           "flag": "not on the allow-list", "none": "no pickle: a header and numbers",
           "loads": "loads {names}", "refused": "refuses: {why}", "cannot": "cannot read it: {why}",
           "after": "The allow-list holds the three imports the state_dict of this lesson needs. Dashed: what is not "
                    "on it, and what torch.load refuses.",
           "alt": "The four files with their imports and what torch.load with weights_only=True does: {rows}."},
    "nl": {"title": "De vier bestanden van de onbekende, voor en tijdens het laden",
           "head": ["bestand", "imports, gelezen met pickletools", "torch.load, weights_only=True"],
           "holds": {"state": "de state_dict", "surprise": "de state_dict en nog één object",
                     "whole": "het hele model", "safe": "dezelfde getallen als safetensors"},
           "flag": "niet op de allow-list", "none": "geen pickle: een header en getallen",
           "loads": "laadt {names}", "refused": "weigert: {why}", "cannot": "kan het niet lezen: {why}",
           "after": "De allow-list bevat de drie imports die de state_dict van deze les nodig heeft. Gestreept: wat er "
                    "niet op staat, en wat torch.load weigert.",
           "alt": "De vier bestanden met hun imports en wat torch.load met weights_only=True doet: {rows}."},
}


def _imports_cell(names: list, allowed: list) -> str:
    return "<br>".join(f'<span style="{_MONO};{_WHOLE}{";" + _OFF + ";padding:0 0.15rem" if n not in allowed else ""}">'
                       f'{html.escape(n)}</span>' for n in names)


def files_picture(lang: str) -> str:
    """Each of the stranger's files: what it holds, what its pickle would import, and what weights_only=True does."""
    w = _FILES_WORDS[lang]
    rows, said = [], []
    for key, name in FILES.items():
        data = DATA[key]
        if key == "safe":
            found, cell = [], html.escape(w["none"])
        else:
            found = imports(pickle_part(data))
            cell = _imports_cell(found, ALLOWED)
        out = load(data, True)
        refused = out["error"] is not None and key != "safe"
        outcome = (w["loads"].format(names=", ".join(out["names"])) if out["error"] is None
                   else w["refused" if refused else "cannot"].format(why=out["error"]))
        rows.append([(_mono(name) + "<br>" + html.escape(w["holds"][key]), ""), (cell, ""),
                     (_mono(outcome), _OFF if refused else "")])
        flagged_here = [n for n in found if n not in ALLOWED]
        said.append(f"{name}, {w['holds'][key]}: {', '.join(found) or w['none']}"
                    f"{' (' + w['flag'] + ': ' + ', '.join(flagged_here) + ')' if flagged_here else ''}; {outcome}")
    body = _table(rows, w["head"]) + _note(w["after"])
    return yours.figure(w["title"], body, alt=w["alt"].format(rows="; ".join(said)))


# --- the pictures under the checks -----------------------------------------------------------------------------------

def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error


def _call(fn, *args):
    """The learner's function on copies of args within the time limit of a check; _STOPPED when it stops with an
    error, and the Timeout passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(lambda: fn(*(copy.deepcopy(a) for a in args)))
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


class Named:
    """A value the check calls by the name the lesson gives it, such as SAFE, so that a message stays short. The
    learner's code gets a copy of the value itself."""

    def __init__(self, value, name: str):
        self.value, self.name = value, name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


SAFE_N = Named(SAFE, "SAFE")
HEADER_FILES = [SAFE_N, Named(TAGGED, "TAGGED"), Named(EMPTY, "EMPTY"), Named(DEEP, "DEEP")]

_HEADER_WORDS = {
    "en": {"title": "Your read_header on four safetensors files",
           "head": ["file", "first 8 bytes", "your read_header", "the course's"], "error": "error",
           "dict": "a dictionary: {keys}", "empty": "an empty dictionary",
           "fine": "The same as the course's read_header on all {k} files.",
           "differs": "It differs from the course's on {k} of the {m} files: they have a dashed frame.",
           "alt": "Four safetensors files with the length in their first 8 bytes, what your read_header gives and "
                  "what the course's gives: {rows}. {after}"},
    "nl": {"title": "Jouw read_header op vier safetensors-bestanden",
           "head": ["bestand", "eerste 8 bytes", "jouw read_header", "die van de course"], "error": "fout",
           "dict": "een dictionary: {keys}", "empty": "een lege dictionary",
           "fine": "Hetzelfde als read_header van de course bij alle {k} bestanden.",
           "differs": "Het verschilt van die van de course bij {k} van de {m} bestanden: die hebben een gestreept "
                      "kader.",
           "alt": "Vier safetensors-bestanden met de lengte in hun eerste 8 bytes, wat jouw read_header geeft en wat "
                  "die van de course geeft: {rows}. {after}"},
}


def _short(v, w) -> str:
    """A value read_header gave, in a few words: a dictionary by its keys, anything else as Python shows it."""
    if v is _STOPPED:
        return w["error"]
    if isinstance(v, dict):
        return w["dict"].format(keys=", ".join(str(k) for k in v)) if v else w["empty"]
    text = repr(v)
    return text if len(text) <= 48 else text[:45] + "..."


def header_picture(seen) -> str:
    """The learner's read_header on each file of the check beside the course's."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _HEADER_WORDS[seen.lang]
    rows, said, k = [], [], 0
    for named in HEADER_FILES:
        got = _call(fn, named)
        want = read_header(named.value)
        differs = got is _STOPPED or got != want
        k += differs
        n = header_length(named.value)
        rows.append([(_mono(named.name), _WHOLE), (str(n), ""), (html.escape(_short(got, w)), _OFF if differs else ""),
                     (html.escape(_short(want, w)), "")])
        said.append(f"{named.name}, {n}: {_short(got, w)}; {_short(want, w)}")
    after = w["differs"].format(k=k, m=len(HEADER_FILES)) if k else w["fine"].format(k=len(HEADER_FILES))
    body = _table(rows, w["head"]) + _note(after)
    return yours.figure(w["title"], body, alt=w["alt"].format(rows="; ".join(said), after=after))


V2_IMPORTS = imports(pickle_part(SURPRISE_PT))
STATE_IMPORTS = imports(pickle_part(STATE_PT))
MODEL_IMPORTS = imports(pickle_part(WHOLE_PT))
ALLOWED_N = Named(ALLOWED, "ALLOWED")
FLAG_CASES = [
    (Named(V2_IMPORTS, "V2_IMPORTS"), ALLOWED_N),
    (Named(STATE_IMPORTS, "STATE_IMPORTS"), ALLOWED_N),
    (Named(MODEL_IMPORTS, "MODEL_IMPORTS"), ALLOWED_N),
    (["builtins.print", "collections.OrderedDict", "builtins.print"], ALLOWED_N),
    (["builtins.print"], []),
    ([], ALLOWED_N),
    (ALLOWED_N, ALLOWED_N),
]

_FLAG_WORDS = {
    "en": {"title": "Your flagged on the stranger's files, and on lists of the check",
           "head": ["names", "allow-list", "your flagged", "the course's"], "error": "error",
           "count": ("{n} name", "{n} names", "no names"),
           "fine": "The same as the course's flagged on all {k} lists.",
           "differs": "It differs from the course's on {k} of the {m} lists: they have a dashed frame.",
           "alt": "Lists of imports with an allow-list, what your flagged gives and what the course's gives: {rows}. "
                  "{after}"},
    "nl": {"title": "Jouw flagged op de bestanden van de onbekende, en op lijsten van de check",
           "head": ["namen", "allow-list", "jouw flagged", "die van de course"], "error": "fout",
           "count": ("{n} naam", "{n} namen", "geen namen"),
           "fine": "Hetzelfde als flagged van de course bij alle {k} lijsten.",
           "differs": "Het verschilt van die van de course bij {k} van de {m} lijsten: die hebben een gestreept kader.",
           "alt": "Lijsten met imports en een allow-list, wat jouw flagged geeft en wat die van de course geeft: "
                  "{rows}. {after}"},
}


def _names_text(v, w) -> str:
    if isinstance(v, Named):
        return f"{v.name} ({_many(len(v.value), w['count'])})"
    return repr(v)


def _plain(v):
    return v.value if isinstance(v, Named) else v


def flagged_picture(seen) -> str:
    """The learner's flagged on each list of the check beside the course's."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _FLAG_WORDS[seen.lang]
    rows, said, k = [], [], 0
    for names, allowed in FLAG_CASES:
        got = _call(fn, names, allowed)
        want = flagged(_plain(names), _plain(allowed))
        differs = got is _STOPPED or got != want
        k += differs
        shown = w["error"] if got is _STOPPED else repr(got)
        rows.append([(_mono(_names_text(names, w)), ""), (_mono(_names_text(allowed, w)), ""),
                     (_mono(shown), _OFF if differs else ""), (_mono(repr(want)), "")])
        said.append(f"{_names_text(names, w)} {_names_text(allowed, w)}: {shown}; {want!r}")
    after = w["differs"].format(k=k, m=len(FLAG_CASES)) if k else w["fine"].format(k=len(FLAG_CASES))
    body = _table(rows, w["head"]) + _note(after)
    return yours.figure(w["title"], body, alt=w["alt"].format(rows="; ".join(said), after=after))


HOLE_CASES = [
    ([[0, 48], [48, 60]], 60),
    ([[0, 48], [52, 64]], 64),
    ([[0, 48], [40, 60]], 60),
    ([[0, 48], [48, 60]], 64),
    ([[4, 48], [48, 60]], 60),
    ([[0, 48], [44, 56], [60, 64]], 64),
    ([[0, 12], [12, 60], [60, 60]], 60),
    ([], 0),
    ([], 10),
]

_HOLE_WORDS = {
    "en": {"title": "Your no_holes on byte buffers",
           "head": ["spans", "size", "the buffer", "your no_holes", "the course's"], "error": "error",
           "fine": "The same as the course's no_holes on all {k} buffers.",
           "differs": "It differs from the course's on {k} of the {m} buffers: they have a dashed frame.",
           "key": "In each bar, the filled parts are bytes that a tensor claims; a dashed part is a hole, an overlap, "
                  "or bytes after the last tensor.",
           "empty": "no bytes",
           "alt": "Byte buffers with their spans, what your no_holes gives and what the course's gives: {rows}. "
                  "{after}"},
    "nl": {"title": "Jouw no_holes op buffers van bytes",
           "head": ["spans", "size", "de buffer", "jouw no_holes", "die van de course"], "error": "fout",
           "fine": "Hetzelfde als no_holes van de course bij alle {k} buffers.",
           "differs": "Het verschilt van die van de course bij {k} van de {m} buffers: die hebben een gestreept kader.",
           "key": "In elke balk zijn de gevulde stukken bytes die bij een tensor horen; een gestreept stuk is een gat, "
                  "een overlap, of bytes na de laatste tensor.",
           "empty": "geen bytes",
           "alt": "Buffers van bytes met hun spans, wat jouw no_holes geeft en wat die van de course geeft: {rows}. "
                  "{after}"},
}


def _bar(spans: list, size: int, w) -> str:
    """A bar of the buffer: a filled block per span, and a dashed block for each hole, overlap or byte left over."""
    scale = max([size] + [stop for _, stop in spans])
    if scale == 0:
        return html.escape(w["empty"])
    covered = [0] * scale
    for start, stop in spans:
        for b in range(max(start, 0), min(stop, scale)):
            covered[b] += 1
    marks, b = [], 0
    while b < scale:   # runs of bytes that no span claims, or that more than one claims, or that lie past size
        bad = covered[b] != 1 or b >= size
        if bad:
            e = b
            while e < scale and (covered[e] != 1 or e >= size):
                e += 1
            marks.append((b, e))
            b = e
        else:
            b += 1
    pct = lambda x: f"{100 * x / scale:.3f}%"   # noqa: E731
    blocks = "".join(f'<div style="position:absolute;top:2px;bottom:2px;left:{pct(s)};width:{pct(e - s)};'
                     f'background:var(--ea-param-wash);border:1px solid var(--ea-param);box-sizing:border-box"></div>'
                     for s, e in spans if e > s)
    blocks += "".join(f'<div style="position:absolute;top:0;bottom:0;left:{pct(s)};width:{pct(e - s)};{_OFF}"></div>'
                      for s, e in marks)
    return (f'<div style="position:relative;height:16px;min-width:9rem;border:1px solid var(--ea-line);'
            f'box-sizing:border-box">{blocks}</div>')


def holes_picture(seen) -> str:
    """The learner's no_holes on each buffer of the check beside the course's, with a bar of each buffer."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _HOLE_WORDS[seen.lang]
    rows, said, k = [], [], 0
    for spans_, size in HOLE_CASES:
        got = _call(fn, spans_, size)
        want = no_holes(spans_, size)
        differs = got is _STOPPED or got != want
        k += differs
        shown = w["error"] if got is _STOPPED else repr(got)
        rows.append([(_mono(repr(spans_)), ""), (str(size), ""), (_bar(spans_, size, w), ""),
                     (html.escape(shown), _OFF if differs else ""), (repr(want), "")])
        said.append(f"{spans_!r} {size}: {shown}; {want}")
    after = w["differs"].format(k=k, m=len(HOLE_CASES)) if k else w["fine"].format(k=len(HOLE_CASES))
    body = _table(rows, w["head"]) + _note(w["key"]) + _note(after)
    return yours.figure(w["title"], body, alt=w["alt"].format(rows="; ".join(said), after=after))


# --- the demo: look before you load ----------------------------------------------------------------------------------

def mine(box_text, correct: bool, name: str):
    """The learner's own function `name` when its exercise is right, else None: the demo then uses the course's."""
    if not correct or not isinstance(box_text, str):
        return None
    from everything_ai.check import run_code
    run_ = run_code(box_text, {})
    fn = run_.names.get(name) if run_.error is None else None
    return fn if callable(fn) else None


def _use(fn, course):
    """A function for the demo: the learner's, unless it stops with an error; then the course's."""
    def use(*args):
        if fn is not None:
            got = _call(fn, *args)
            if got is not _STOPPED:
                return got
        return course(*args)
    return use


_LOOK_WORDS = {
    "en": {"title": "Look before you load: {name}",
           "imports": "Step 1, look: pickletools reads {k}, and runs nothing.",
           "flagged": "Your flagged: {names}.", "clean": "Every import is on the allow-list.",
           "header": "Step 1, look: the first 8 bytes say {n}, and read_header reads the header: {keys}.",
           "holes": "no_holes on its spans and its {size} bytes of numbers: {verdict}.",
           "head": ["way", "printed while loading", "what came back"],
           "nothing": "nothing", "error": "error: {why}",
           "imports_n": ("{n} import", "{n} imports"),
           "own": "The course runs weights_only=False here only because it made these files itself; on a "
                  "stranger's file, torch.load would then run whatever the file asks.",
           "whose": "With your {names}; the course's for the rest, until yours are right.",
           "whose_none": "With the course's functions, until yours are right.", "and": " and ",
           "alt": "{name}: {look} Loading it: {rows}."},
    "nl": {"title": "Kijk voor je laadt: {name}",
           "imports": "Stap 1, kijken: pickletools leest {k}, en voert niets uit.",
           "flagged": "Jouw flagged: {names}.", "clean": "Elke import staat op de allow-list.",
           "header": "Stap 1, kijken: de eerste 8 bytes zeggen {n}, en read_header leest de header: {keys}.",
           "holes": "no_holes op de spans en de {size} bytes getallen: {verdict}.",
           "head": ["manier", "geprint tijdens het laden", "wat er terugkwam"],
           "nothing": "niets", "error": "fout: {why}",
           "imports_n": ("{n} import", "{n} imports"),
           "own": "De course gebruikt weights_only=False hier alleen omdat hij deze bestanden zelf heeft gemaakt; "
                  "bij het bestand van een onbekende zou torch.load dan uitvoeren wat het bestand vraagt.",
           "whose": "Met jouw {names}; die van de course voor de rest, tot de jouwe kloppen.",
           "whose_none": "Met de functies van de course, tot de jouwe kloppen.", "and": " en ",
           "alt": "{name}: {look} Het laden: {rows}."},
}


def look_picture(key: str, lang: str, header_fn=None, flagged_fn=None, holes_fn=None) -> str:
    """For the chosen file of the stranger: first a look that runs nothing, then what torch.load does in both ways."""
    w = _LOOK_WORDS[lang]
    key = key if key in DATA else "surprise"
    data, name = DATA[key], FILES[key]
    if key == "safe":
        header = _use(header_fn, read_header)(data)
        keys = ", ".join(str(k) for k in header) if isinstance(header, dict) else repr(header)
        look = w["header"].format(n=header_length(data), keys=keys)
        try:
            ok = _use(holes_fn, no_holes)(spans(read_header(data)), buffer_size(data))
        except Exception:   # noqa: BLE001
            ok = None
        look += " " + w["holes"].format(size=buffer_size(data), verdict=repr(ok))
        top = _note(look)
    else:
        found = imports(pickle_part(data))
        flags = _use(flagged_fn, flagged)(found, ALLOWED)
        look = w["imports"].format(k=_many(len(found), w["imports_n"])) + " "
        look += w["flagged"].format(names=", ".join(map(str, flags))) if flags else w["clean"]
        verdict = w["flagged"].format(names=", ".join(map(str, flags))) if flags else w["clean"]
        top = (_note(w["imports"].format(k=_many(len(found), w["imports_n"]))) + _imports_cell(found, ALLOWED)
               + _note(verdict))
    rows, said = [], []
    for only in (True, False):
        out = load(data, only)
        way = f"torch.load(..., weights_only={only})"
        back = ", ".join(out["names"]) if out["error"] is None else w["error"].format(why=out["error"])
        printed = out["printed"] or w["nothing"]
        rows.append([(_mono(way), _WHOLE), (_mono(printed), _OFF if out["printed"] else ""),
                     (_mono(back), _OFF if out["error"] else "")])
        said.append(f"{way}: {printed}; {back}")
    mine_ = [n for n, f in (("read_header", header_fn), ("flagged", flagged_fn), ("no_holes", holes_fn)) if f]
    joined = mine_[0] if len(mine_) == 1 else ", ".join(mine_[:-1]) + w["and"] + mine_[-1] if mine_ else ""
    whose = w["whose"].format(names=joined) if mine_ else w["whose_none"]
    body = top + _table(rows, w["head"]) + _note(w["own"]) + _note(whose)
    return yours.figure(w["title"].format(name=name), body,
                        alt=w["alt"].format(name=name, look=look, rows="; ".join(said)))


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["read-header", "flagged", "no-holes", "flag", "repository"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _names_error(e) -> bool:
    return isinstance(e, NameError) and re.search(r"'(SAFE|TAGGED|EMPTY|DEEP|ALLOWED|V2_IMPORTS|STATE_IMPORTS|"
                                                  r"MODEL_IMPORTS|C)'", str(e)) is not None


_NAMES = Mistake("lesson-names", error=_names_error,
                 en="Your function uses a name of the lesson, such as `SAFE` or `ALLOWED`, and the check does not give it that name: use the names between the parentheses of the `def` line.",
                 nl="Je functie gebruikt een naam van de les, zoals `SAFE` of `ALLOWED`, en de check geeft die naam niet mee: gebruik de namen tussen de haakjes van de `def`-regel.")


def _show(v) -> str:
    """A value for a check message, as a code span."""
    return f"`{v!r}`"


def _show_header(v):
    """What read_header gave, for a message: a dictionary by its keys, anything else as Python shows it."""
    if isinstance(v, dict):
        if not v:
            return {"en": "an empty dictionary", "nl": "een lege dictionary"}
        keys = ", ".join(f"`{k}`" for k in v)
        return {"en": f"a dictionary with the keys {keys}", "nl": f"een dictionary met de keys {keys}"}
    text = repr(v)
    return f"`{text if len(text) <= 60 else text[:57] + '...'}`"


# exercise 1: read_header
_H_HEAD = ("import json\n"
           "\n"
           "def read_header(data):\n"
           "    # the JSON header of a safetensors file as a dictionary; data holds the bytes of the whole file\n")
_H_N = "    n = int.from_bytes(data[:8], \"little\")\n"
_H_RET = "    return json.loads(data[8:8 + n])\n"
_H_START = _H_HEAD + "    n = ...\n" + "    return ...\n"
_H = _H_HEAD + _H_N + _H_RET
_H_BIG = _H.replace('"little"', '"big"')
_H_FIRST = _H.replace("int.from_bytes(data[:8], \"little\")", "data[0]")
_H_TO_END = _H.replace("data[8:8 + n]", "data[8:]")
_H_NO_OFFSET = _H.replace("data[8:8 + n]", "data[8:n]")
_H_FROM_START = _H.replace("data[8:8 + n]", "data[:n]")
_H_RAW = _H.replace("json.loads(data[8:8 + n])", "data[8:8 + n]")
_H_LENGTH = _H.replace("json.loads(data[8:8 + n])", "n")
_H_NO_INT = _H.replace("int.from_bytes(data[:8], \"little\")", "data[:8]")

# exercise 2: flagged
_F_HEAD = ("def flagged(names, allowed):\n"
           "    # the imports in names that are not on the allow-list, each once, in the order they come\n"
           "    found = []\n"
           "    for name in names:\n")
_F_IF = "        if name not in allowed and name not in found:\n"
_F_ADD = "            found.append(name)\n"
_F_RET = "    return found\n"
_F_START = _F_HEAD + "        ...\n" + _F_RET
_F = _F_HEAD + _F_IF + _F_ADD + _F_RET
_F_INVERTED = _F.replace("if name not in allowed and", "if name in allowed and")
_F_TWICE = _F.replace(" and name not in found", "")
_F_FIRST = _F_HEAD + _F_IF + _F_ADD + "            return found\n" + _F_RET

# exercise 3: no_holes
_N_HEAD = ("def no_holes(spans, size):\n"
           "    # True when spans, [start, stop] pairs sorted by start, fill a byte buffer of size bytes exactly\n"
           "    end = 0\n"
           "    for start, stop in spans:\n")
_N_IF = ("        if start != end:\n"
         "            return False\n")
_N_END = "        end = stop\n"
_N_RET = "    return end == size\n"
_N_START = _N_HEAD + "        ...\n" + _N_END + "    return ...\n"
_N = _N_HEAD + _N_IF + _N_END + _N_RET
_N_NO_END = _N.replace("    return end == size\n", "    return True\n")
_N_HOLES_ONLY = _N.replace("if start != end:", "if start > end:")
_N_LENGTHS = ("def no_holes(spans, size):\n"
              "    total = 0\n"
              "    for start, stop in spans:\n"
              "        total = total + stop - start\n"
              "    return total == size\n")

EXERCISES = {
    "read-header": FunctionBox(
        name="read-header",
        picture=header_picture,
        function="read_header",
        start=_H_START,
        solution=_H,
        visible=[(SAFE_N,)],
        hidden=[(f,) for f in HEADER_FILES[1:]],
        show=_show_header,
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("no-json", error=lambda e: isinstance(e, NameError) and "'json'" in str(e),
                    en="Your code uses `json`, and the line `import json` at the top of the box is gone. Put it back: without it, Python does not know the name.",
                    nl="Je code gebruikt `json`, en de regel `import json` bovenaan het vak is weg. Zet hem terug: zonder die regel kent Python de naam niet."),
            Mistake("big", syntax=r"[\"']big[\"']",
                    en="Your code reads the first 8 bytes with `\"big\"`, the highest byte first, so `n` becomes a huge number and the slice runs on into the numbers, which are not JSON. The format puts the lowest byte first.",
                    nl="Je code leest de eerste 8 bytes met `\"big\"`, de hoogste byte eerst, dus `n` wordt een enorm getal en het stuk loopt door tot in de getallen, die geen JSON zijn. Het formaat zet de laagste byte eerst."),
            Mistake("no-order", syntax=r"int\.from_bytes\(\s*data\s*\[[^\]]*\]\s*\)",
                    en="Your code calls `int.from_bytes` without a byte order, and then Python reads the highest byte first, so `n` becomes a huge number and the slice runs on into the numbers, which are not JSON. The format puts the lowest byte first.",
                    nl="Je code roept `int.from_bytes` aan zonder bytevolgorde, en dan leest Python de hoogste byte eerst, dus `n` wordt een enorm getal en het stuk loopt door tot in de getallen, die geen JSON zijn. Het formaat zet de laagste byte eerst."),
            Mistake("no-int", syntax=r"^\s*n\s*=\s*data\s*\[\s*0?\s*:\s*8\s*\]\s*$",
                    en="`data[:8]` is 8 bytes, not a number yet, so Python cannot count with it. `int.from_bytes` turns the 8 bytes into the number they hold.",
                    nl="`data[:8]` is 8 bytes, nog geen getal, dus Python kan er niet mee rekenen. `int.from_bytes` maakt van de 8 bytes het getal dat erin staat."),
            Mistake("first-byte", syntax=r"data\s*\[\s*0\s*\]",
                    en="Your code takes the length from `data[0]`, one byte, and one byte holds at most 255. The header of `DEEP` is " + str(header_length(DEEP)) + " bytes long, so its length takes two bytes. Read the whole of the first 8 bytes as one number.",
                    nl="Je code haalt de lengte uit `data[0]`, één byte, en in één byte past hoogstens 255. De header van `DEEP` is " + str(header_length(DEEP)) + " bytes lang, dus de lengte neemt twee bytes in. Lees de eerste 8 bytes samen als één getal."),
            Mistake("to-end", syntax=r"data\s*\[\s*8\s*:\s*\]",
                    en="Your code gives `json.loads` everything after the first 8 bytes, and after the header come the numbers, which are not JSON. Give it only the header: that is what `n` is for.",
                    nl="Je code geeft `json.loads` alles na de eerste 8 bytes, en na de header komen de getallen, die geen JSON zijn. Geef het alleen de header: daar is `n` voor."),
            Mistake("no-offset", syntax=r"data\s*\[\s*8\s*:\s*n\s*\]",
                    en="Your slice stops at byte `n`, but the header does not start at byte 0: the first 8 bytes hold its length. So `json.loads` gets a header with its end cut off.",
                    nl="Je stuk stopt bij byte `n`, maar de header begint niet bij byte 0: de eerste 8 bytes bevatten de lengte. Dus krijgt `json.loads` een header waarvan het eind is afgeknipt."),
            Mistake("from-start", syntax=r"data\s*\[\s*0?\s*:\s*(?:8\s*\+\s*)?n\s*(?:\+\s*8\s*)?\]",
                    en="Your code gives `json.loads` the bytes from the start of the file, and the first 8 bytes are the length of the header, not JSON. The header starts at byte 8.",
                    nl="Je code geeft `json.loads` de bytes vanaf het begin van het bestand, en de eerste 8 bytes zijn de lengte van de header, geen JSON. De header begint bij byte 8."),
            Mistake("raw", matches=lambda got, text: isinstance(got, (bytes, bytearray, str)),
                    en="`{call}` gives {got}: the header as bytes, not yet a dictionary. `json.loads` turns JSON into a dictionary.",
                    nl="`{call}` geeft {got}: de header als bytes, nog geen dictionary. `json.loads` maakt van JSON een dictionary."),
            Mistake("length", matches=lambda got, text: isinstance(got, int),
                    en="`{call}` gives {got}: that is the length of the header, not the header itself. Give back the header, read as JSON.",
                    nl="`{call}` geeft {got}: dat is de lengte van de header, niet de header zelf. Geef de header terug, gelezen als JSON."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: first `n`, the number in the first 8 bytes, with the lowest byte first; then give back the header that follows them, read as JSON.",
                      "nl": "Vervang de twee `...`: eerst `n`, het getal in de eerste 8 bytes, met de laagste byte eerst; geef daarna de header terug die erop volgt, gelezen als JSON."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The first 8 bytes hold the length of the header, lowest byte first; the header itself is JSON.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De eerste 8 bytes bevatten de lengte van de header, met de laagste byte eerst; de header zelf is JSON."},
            "almost": {"en": "Right for `read_header(SAFE)`. But `{call}` gives {got}, and it should give {want}. Does your code use the length it read, for every file?",
                       "nl": "Goed voor `read_header(SAFE)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je code de lengte die ze las, bij elk bestand?"},
            "correct": {"en": "Your `read_header` is right on all {n} files. It reads bytes and JSON, and nothing else: no line of the file can run.",
                        "nl": "Je `read_header` klopt bij alle {n} bestanden. Hij leest bytes en JSON, en verder niets: geen regel uit het bestand kan draaien."},
        },
    ),
    "flagged": FunctionBox(
        name="flagged",
        picture=flagged_picture,
        function="flagged",
        start=_F_START,
        solution=_F,
        visible=[FLAG_CASES[0]],
        hidden=FLAG_CASES[1:],
        show=_show,
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("inverted", wrong=_F_INVERTED,
                    en="`{call}` gives {got}, and it should give {want}: your `flagged` keeps the names that are on the allow-list. It should flag the others.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw `flagged` houdt de namen die op de allow-list staan. Het moet juist de andere aanwijzen."),
            Mistake("first-only", wrong=_F_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: your `flagged` stops at the first name it flags. Is there a `return` inside the loop?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw `flagged` stopt bij de eerste naam die het aanwijst. Staat er een `return` in de loop?"),
            Mistake("twice", wrong=_F_TWICE,
                    en="`{call}` gives {got}, and it should give {want}: `builtins.print` comes twice in the names, and your `flagged` flags it twice. Each name should be flagged once.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `builtins.print` staat twee keer in de namen, en jouw `flagged` wijst het twee keer aan. Elke naam hoort één keer aangewezen te worden."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` in the loop: flag `name` when it is not on the allow-list, and flag each name only once.",
                      "nl": "Vervang de `...` in de loop: wijs `name` aan als die niet op de allow-list staat, en wijs elke naam maar één keer aan."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Flag every name that is not on the allow-list, once, in the order the names come.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Wijs elke naam aan die niet op de allow-list staat, één keer, in de volgorde van de namen."},
            "almost": {"en": "Right for `flagged(V2_IMPORTS, ALLOWED)`. But `{call}` gives {got}, and it should give {want}. Look at every name, and at names that come twice.",
                       "nl": "Goed voor `flagged(V2_IMPORTS, ALLOWED)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Kijk naar elke naam, en naar namen die twee keer voorkomen."},
            "correct": {"en": "Your `flagged` is right on all {n} lists. On the stranger's files, it flags `builtins.print` in trail-sorter-v2.pt, three names in trail-sorter-model.pt, and nothing in trail-sorter.pt.",
                        "nl": "Je `flagged` klopt bij alle {n} lijsten. Op de bestanden van de onbekende wijst het `builtins.print` aan in trail-sorter-v2.pt, drie namen in trail-sorter-model.pt, en niets in trail-sorter.pt."},
        },
    ),
    "no-holes": FunctionBox(
        name="no-holes",
        picture=holes_picture,
        function="no_holes",
        start=_N_START,
        solution=_N,
        visible=[HOLE_CASES[0]],
        hidden=HOLE_CASES[1:],
        show=_show,
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("no-end", wrong=_N_NO_END,
                    en="`{call}` gives {got}, and it should give {want}: the spans stop at byte 60, and the buffer has 64 bytes. Four bytes that no tensor claims are a hole too, at the end.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de spans stoppen bij byte 60, en de buffer heeft 64 bytes. Vier bytes die bij geen enkele tensor horen, zijn ook een gat, aan het eind."),
            Mistake("holes-only", wrong=_N_HOLES_ONLY,
                    en="`{call}` gives {got}, and it should give {want}: the second span starts at byte 40, before the first one stops at 48. Your check should also refuse bytes that two tensors claim.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de tweede span begint bij byte 40, voordat de eerste stopt bij 48. Je controle moet ook bytes weigeren die bij twee tensors horen."),
            Mistake("lengths", wrong=_N_LENGTHS,
                    en="`{call}` gives {got}, and it should give {want}: the lengths of the spans add up to 64, but the second span overlaps the first by 4 bytes, and bytes 56 to 60 are a hole. A sum of lengths cannot see where each span lies.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de lengtes van de spans tellen op tot 64, maar de tweede span overlapt de eerste met 4 bytes, en de bytes 56 tot 60 zijn een gat. Een som van lengtes ziet niet waar elke span ligt."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: in the loop, give back False as soon as a span leaves a hole or overlaps the one before it; after the loop, give back whether the spans reach exactly `size`.",
                      "nl": "Vervang de twee `...`: geef in de loop False terug zodra een span een gat laat of de vorige overlapt; geef na de loop terug of de spans precies tot `size` komen."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Every byte of the buffer must belong to exactly one span: no holes, no overlaps, nothing left over at the end.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Elke byte van de buffer moet bij precies één span horen: geen gaten, geen overlap, niets over aan het eind."},
            "almost": {"en": "Right for `no_holes([[0, 48], [48, 60]], 60)`. But `{call}` gives {got}, and it should give {want}. Look at the bar of that buffer in the picture under the check.",
                       "nl": "Goed voor `no_holes([[0, 48], [48, 60]], 60)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Kijk naar de balk van die buffer in het plaatje onder de check."},
            "correct": {"en": "Your `no_holes` is right on all {n} buffers. The safetensors README asks for a buffer with no holes, so that no bytes can hide in the file outside the tensors; your check also refuses overlaps, so each byte belongs to exactly one tensor.",
                        "nl": "Je `no_holes` klopt bij alle {n} buffers. De README van safetensors vraagt om een buffer zonder gaten, zodat er geen bytes buiten de tensors in het bestand kunnen schuilen; jouw controle weigert ook overlap, dus elke byte hoort bij precies één tensor."},
        },
    ),
}

CASES = [
    ("read-header", _H_START, "empty", None),
    ("read-header", _H, "correct", None),
    ("read-header", "import json\n\ndef read_header(data):\n    n = int.from_bytes(data[0:8], byteorder=\"little\")\n    return json.loads(data[8:n + 8].decode())\n", "correct", None),
    ("read-header", _H_BIG, "error", "big"),
    ("read-header", _H.replace(", \"little\")", ")"), "error", "no-order"),
    ("read-header", _H_NO_INT, "error", "no-int"),
    ("read-header", _H_FIRST, "error", "first-byte"),
    ("read-header", _H_TO_END, "error", "to-end"),
    ("read-header", _H_NO_OFFSET, "error", "no-offset"),
    ("read-header", _H_FROM_START, "error", "from-start"),
    ("read-header", _H_RAW, "wrong", "raw"),
    ("read-header", _H_LENGTH, "wrong", "length"),
    ("read-header", _H.replace("import json\n", ""), "error", "no-json"),
    ("read-header", "import json\n\ndef read_header(data):\n    return json.loads(SAFE[8:128])\n", "error", "lesson-names"),
    ("read-header", "import json\n\ndef read_header(data):\n    return {}\n", "wrong", None),
    ("flagged", _F_START, "empty", None),
    ("flagged", _F, "correct", None),
    ("flagged", "def flagged(names, allowed):\n    found = []\n    for name in names:\n        if name in allowed or name in found:\n            continue\n        found.append(name)\n    return found\n", "correct", None),
    ("flagged", _F_INVERTED, "wrong", "inverted"),
    ("flagged", _F_FIRST, "wrong", "first-only"),
    ("flagged", _F_TWICE, "wrong", "twice"),
    ("flagged", "def flagged(names, allowed):\n    return []\n", "wrong", None),
    ("flagged", "def flagged(names, allowed):\n    return [\"builtins.print\"]\n", "almost", None),
    ("flagged", "def flagged(names, allowed):\n    return [n for n in names if n not in ALLOWED]\n", "error", "lesson-names"),
    ("no-holes", _N_START, "empty", None),
    ("no-holes", _N, "correct", None),
    ("no-holes", "def no_holes(spans, size):\n    end = 0\n    for start, stop in spans:\n        if start == end:\n            end = stop\n        else:\n            return False\n    return end == size\n", "correct", None),
    ("no-holes", _N_NO_END, "wrong", "no-end"),
    ("no-holes", _N_HOLES_ONLY, "wrong", "holes-only"),
    ("no-holes", _N_LENGTHS, "wrong", "lengths"),
    ("no-holes", "def no_holes(spans, size):\n    return False\n", "wrong", None),
    ("no-holes", "def no_holes(spans, size):\n    return True\n", "almost", None),
    ("no-holes", "def no_holes(spans, size):\n    return SAFE == size\n", "error", "lesson-names"),
]
