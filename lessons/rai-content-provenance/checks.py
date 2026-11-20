# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-content-provenance: signing and checking with a toy key, and reading a picture's label.
The key is the toy example of Rivest, Shamir and Adleman (1978): p = 47, q = 59, n = 2773, e = 17, d = 157. The
pictures are the course's own pixel drawings, and a label is a toy version of a C2PA manifest: who signed, the
picture's hash below n, and the signature of that hash."""
import contextlib
import copy
import hashlib
import html
import io
import json
import math
import re

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

# --- the toy keys ----------------------------------------------------------------------------------------------------

# "Consider the case p = 47, q = 59, n = p · q = 47 · 59 = 2773, and d = 157." "Therefore e = 17, the multiplicative
# inverse (mod 2668) of d = 157." (Rivest, Shamir and Adleman, 1978, p. 10)
P, Q = 47, 59
N = P * Q   # 2773: the modulus, part of the public key
E = 17      # the public exponent
D = 157     # the private exponent: only the signer has it

DESK = "photo-desk.example"       # a made-up news desk that signs its pictures with the key of the paper
MILL = "pixel-mill.example"       # a made-up maker of pictures with a key of its own: p = 43, q = 67
MILL_KEY = (5, 2881)
MILL_D = 1109
DOCK = "harbor-office.example"    # one more made-up signer, for the check's own cases: p = 53, q = 71
DOCK_KEY = (3, 3763)
DOCK_D = 2427
KEYS = {DESK: (E, N)}             # the trust list: the public keys of the signers you trust


def toy_hash(picture: bytes, n: int) -> int:
    """The SHA-256 of the picture's bytes as one whole number, and its remainder after dividing by n: a fingerprint
    small enough for a toy key. A real signer signs the whole SHA-256 with a key of 2048 bits or more."""
    return int.from_bytes(hashlib.sha256(picture).digest(), "big") % n


# --- the pictures: the course's own drawings, 12 pixels wide and 9 high, one byte per pixel ----------------------------

WIDTH, HEIGHT = 12, 9
_SHADES = {".": 0, "~": 120, "#": 230}
_BOAT = ["............", ".....#......", ".....##.....", ".....###....", ".....####...", "...#######..",
         "....#####...", "~~~~~~~~~~~~", "~~~~~~~~~~~~"]
_LIGHTHOUSE = ["............", "......#.....", ".....###....", "......#.....", "......#.....", ".....###....",
               ".....###....", "~~~~~###~~~~", "~~~~~~~~~~~~"]
EDIT = 1 * WIDTH + 5   # the pixel at the top of the sail, which the edit makes dark


def _drawing(rows: list) -> bytes:
    return bytes(_SHADES[c] for row in rows for c in row)


BOAT = _drawing(_BOAT)
EDITED = BOAT[:EDIT] + bytes([0]) + BOAT[EDIT + 1:]
LIGHTHOUSE = _drawing(_LIGHTHOUSE)


def sign_picture(picture: bytes, signer: str, d: int, n: int) -> dict:
    """A toy label: who signed, the picture's hash below n, and the signature of that hash with the private
    exponent d."""
    h = toy_hash(picture, n)
    return {"signer": signer, "hash": h, "signature": pow(h, d, n)}


BOAT_LABEL = sign_picture(BOAT, DESK, D, N)
MILL_LABEL = sign_picture(LIGHTHOUSE, MILL, MILL_D, MILL_KEY[1])
FORGED_LABEL = dict(BOAT_LABEL, hash=toy_hash(EDITED, N))   # the hash of the edited picture, the boat's signature

# numbers that photo-desk.example is said to have signed, each with its signature: the paper's 920, 921 with the
# signature of 920, 300 with a signature one higher than its own, the boat's hash, and the edited boat's hash with
# the boat's signature
SIGNED = [(920, 192), (921, 192), (300, pow(300, D, N) + 1), (BOAT_LABEL["hash"], BOAT_LABEL["signature"]),
          (FORGED_LABEL["hash"], BOAT_LABEL["signature"])]


def genuine(pairs: list, e: int, n: int) -> list:
    """The numbers whose signature checks with the public key (e, n), in the order of pairs."""
    return [m for m, s in pairs if pow(s, e, n) == m]


def verdict(picture: bytes, manifest, keys: dict) -> str:
    """What a picture's label says: no credentials, an unknown signer, a changed picture, a bad signature, or who
    signed it."""
    if manifest is None:
        return "no credentials"
    signer = manifest["signer"]
    if signer not in keys:
        return "unknown signer"
    e, n = keys[signer]
    if toy_hash(picture, n) != manifest["hash"]:
        return "changed"
    if pow(manifest["signature"], e, n) != manifest["hash"]:
        return "bad signature"
    return "signed by " + signer


class Named:
    """A value the check gives the learner's function, shown in a message by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


def _plain(v):
    return v.value if isinstance(v, Named) else v


SIGNED_N = Named("SIGNED", SIGNED)
BOAT_N = Named("BOAT", BOAT)
EDITED_N = Named("EDITED", EDITED)
LIGHTHOUSE_N = Named("LIGHTHOUSE", LIGHTHOUSE)
BOAT_LABEL_N = Named("BOAT_LABEL", BOAT_LABEL)
MILL_LABEL_N = Named("MILL_LABEL", MILL_LABEL)
FORGED_LABEL_N = Named("FORGED_LABEL", FORGED_LABEL)
KEYS_N = Named("KEYS", KEYS)

# the five pictures of Exercise 3: what each is, with its label
SCENES = [(BOAT_N, BOAT_LABEL_N, "signed"), (EDITED_N, BOAT_LABEL_N, "edited"), (BOAT_N, None, "stripped"),
          (LIGHTHOUSE_N, MILL_LABEL_N, "mill"), (EDITED_N, FORGED_LABEL_N, "forged")]

# --- pictures --------------------------------------------------------------------------------------------------------
# Every value is what the learner's code gives: all of it is shown through html.escape.

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.15rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;vertical-align:middle")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"
_MONO = "font-family:var(--ea-font-code);font-size:12px;overflow-wrap:anywhere"
_WHOLE = "white-space:nowrap"


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


def pixels(picture: bytes, cell: int = 6, mark: int | None = None, alt: str = "") -> str:
    """A drawing of 12 by 9 pixels in the ink colour, darker where the byte is larger; a dashed frame marks one pixel."""
    parts = [f'<rect x="0" y="0" width="{WIDTH * cell}" height="{HEIGHT * cell}" fill="none" '
             f'stroke="var(--ea-line)" stroke-width="1"/>']
    for i, value in enumerate(picture[:WIDTH * HEIGHT]):
        x, y = (i % WIDTH) * cell, (i // WIDTH) * cell
        if value:
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="var(--ea-ink)" '
                         f'fill-opacity="{value / 255:.2f}"/>')
    if mark is not None:
        x, y = (mark % WIDTH) * cell, (mark // WIDTH) * cell
        parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="none" stroke="var(--ea-loss)" '
                     f'stroke-width="1.5" stroke-dasharray="2 1.5"/>')
    label = f' role="img" aria-label="{html.escape(alt)}"' if alt else ' aria-hidden="true"'
    return (f'<svg width="{WIDTH * cell}" height="{HEIGHT * cell}" viewBox="0 0 {WIDTH * cell} {HEIGHT * cell}"{label} '
            f'style="display:block">{"".join(parts)}</svg>')


def _quoted(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def label(v) -> str:
    """A value in a few words: a Named value by its name, otherwise as Python writes it, cut short."""
    if isinstance(v, Named):
        return v.name
    if isinstance(v, str):
        return _quoted(v)
    if isinstance(v, (bytes, bytearray)):
        return f"<{len(v)} bytes>"
    if isinstance(v, int) and not isinstance(v, bool) and abs(v) >= 10 ** 12:
        return f"≈ 10^{_digits(v) - 1}"   # Python refuses to write a number of more than 4300 digits
    if isinstance(v, dict):
        text = "{" + ", ".join(f"{_quoted(str(k))}: {label(x)}" for k, x in v.items()) + "}"
    elif type(v) is list:
        text = "[" + ", ".join(label(x) for x in v) + "]"
    else:
        text = repr(v)
    return text if len(text) <= 80 else text[:77] + "..."


_LABEL_WORDS = {
    "en": {"title": "The boat of photo-desk.example and its label",
           "head": ["in the label", "in this lesson", "in the words of C2PA"],
           "rows": [("signer", "who signed: photo-desk.example", "the signer, whose private key made the claim signature"),
                    ("hash", "toy_hash(BOAT, n)", "hard binding: one or more cryptographic hashes of the asset"),
                    ("signature", "pow(hash, d, n), with d known only to the signer", "claim signature"),
                    ("(none)", "no statements about the picture", "assertions, such as which camera or tool made it")],
           "alt": "A drawing of a sailboat on the sea, 12 by 9 pixels, with its label: {rows}.",
           "after": "A real manifest also signs its assertions, and uses SHA-256 and a key of 2048 bits or more."},
    "nl": {"title": "De boot van photo-desk.example en zijn label",
           "head": ["in het label", "in deze les", "in de woorden van C2PA"],
           "rows": [("signer", "wie tekende: photo-desk.example", "de ondertekenaar, die met de eigen private key de claim signature maakte"),
                    ("hash", "toy_hash(BOAT, n)", "hard binding: een of meer cryptografische hashes van het bestand"),
                    ("signature", "pow(hash, d, n), met d alleen bekend bij de ondertekenaar", "claim signature"),
                    ("(geen)", "geen uitspraken over het plaatje", "assertions, zoals welke camera of tool het maakte")],
           "alt": "Een tekening van een zeilboot op zee, 12 bij 9 pixels, met zijn label: {rows}.",
           "after": "Een echt manifest ondertekent ook zijn assertions, en gebruikt SHA-256 en een key van 2048 bits of meer."},
}


def label_picture(lang: str) -> str:
    """The boat with its toy label, each part next to the C2PA part it stands for."""
    w = _LABEL_WORDS[lang]
    rows = [[(_mono(part), _WHOLE), (html.escape(toy), ""), (html.escape(c2pa), "")] for part, toy, c2pa in w["rows"]]
    said = "; ".join(f"{part}: {toy} ({c2pa})" for part, toy, c2pa in w["rows"])
    drawing = pixels(BOAT, cell=10)
    body = (f'<div style="display:flex;gap:1rem;align-items:flex-start;flex-wrap:wrap">{drawing}'
            f'<div style="flex:1;min-width:16rem">{_table(rows, w["head"])}</div></div>' + _note(w["after"]))
    return yours.figure(w["title"], body, alt=w["alt"].format(rows=said))


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


_CHECK_WORDS = {
    "en": {"yes": "yes", "no": "no",
           "genuine": {"title": "Your genuine on SIGNED, with the key of photo-desk.example",
                       "head": ["number", "signature", "checking gives", "in yours", "in the course's"],
                       "fine": "Your genuine gives the same numbers as the course's: {want}.",
                       "differs": "Your genuine gives {got}, the course's {want}; rows where they differ have a "
                                  "dashed frame.",
                       "other": "Your genuine does not give a list: the check above says what it gives.",
                       "stops": "Your genuine stops with an error: the check above says where."},
           "sign": {"title": "Your sign_picture on the boat, with the key of photo-desk.example",
                    "head": ["", "yours", "the course's"],
                    "fine": "The same label as the course's.",
                    "differs": "It differs from the course's in {k} of the 3 parts: they have a dashed frame.",
                    "other": "Your sign_picture does not give a dict: the check above says what it gives.",
                    "stops": "Your sign_picture stops with an error: the check above says where."},
           "verdict": {"title": "Your verdict on five pictures, with KEYS as the trust list",
                       "head": ["picture", "its label", "yours", "the course's"],
                       "labels": {"signed": "BOAT_LABEL", "edited": "BOAT_LABEL",
                                  "stripped": "none: a platform removed it", "mill": "MILL_LABEL, of pixel-mill.example",
                                  "forged": "FORGED_LABEL: the hash of EDITED, the signature of BOAT"},
                       "fine": "The same as the course's for all five pictures.",
                       "differs": "It differs from the course's for {k} of the five pictures: they have a dashed "
                                  "frame.",
                       "error": "error"},
           "alt": "{title}: {rows}. {after}"},
    "nl": {"yes": "ja", "no": "nee",
           "genuine": {"title": "Jouw genuine op SIGNED, met de key van photo-desk.example",
                       "head": ["getal", "handtekening", "controleren geeft", "in de jouwe", "in die van de course"],
                       "fine": "Jouw genuine geeft dezelfde getallen als die van de course: {want}.",
                       "differs": "Jouw genuine geeft {got}, die van de course {want}; rijen waar ze verschillen "
                                  "hebben een gestreept kader.",
                       "other": "Jouw genuine geeft geen lijst: de check hierboven zegt wat hij geeft.",
                       "stops": "Jouw genuine stopt met een fout: de check hierboven zegt waar."},
           "sign": {"title": "Jouw sign_picture op de boot, met de key van photo-desk.example",
                    "head": ["", "de jouwe", "die van de course"],
                    "fine": "Hetzelfde label als dat van de course.",
                    "differs": "Het verschilt van dat van de course in {k} van de 3 delen: die hebben een gestreept "
                               "kader.",
                    "other": "Jouw sign_picture geeft geen dict: de check hierboven zegt wat hij geeft.",
                    "stops": "Jouw sign_picture stopt met een fout: de check hierboven zegt waar."},
           "verdict": {"title": "Jouw verdict op vijf plaatjes, met KEYS als vertrouwenslijst",
                       "head": ["plaatje", "zijn label", "de jouwe", "die van de course"],
                       "labels": {"signed": "BOAT_LABEL", "edited": "BOAT_LABEL",
                                  "stripped": "geen: een platform haalde het weg", "mill": "MILL_LABEL, van pixel-mill.example",
                                  "forged": "FORGED_LABEL: de hash van EDITED, de handtekening van BOAT"},
                       "fine": "Bij alle vijf plaatjes hetzelfde als het verdict van de course.",
                       "differs": "Bij {k} van de vijf plaatjes anders dan het verdict van de course: die hebben een "
                                  "gestreept kader.",
                       "error": "fout"},
           "alt": "{title}: {rows}. {after}"},
}


def genuine_picture(seen) -> str:
    """Each number of SIGNED with its signature and what checking gives, and whether the learner's list and the
    course's hold it."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _CHECK_WORDS[seen.lang]
    words = w["genuine"]
    got = _call(fn, SIGNED_N, E, N)
    if got is _STOPPED:
        return yours.figure(words["title"], _note(words["stops"]))
    if not isinstance(got, list):
        return yours.figure(words["title"], _note(words["other"]))
    want = genuine(SIGNED, E, N)
    rows, said = [], []
    for m, s in SIGNED:   # every number is in SIGNED once
        mine_, course_ = m in got, m in want
        yes_no = (w["yes"] if mine_ else w["no"], w["yes"] if course_ else w["no"])
        rows.append([(_mono(str(m)), ""), (_mono(str(s)), ""), (_mono(str(pow(s, E, N))), ""),
                     (html.escape(yes_no[0]), _OFF if mine_ != course_ else ""), (html.escape(yes_no[1]), "")])
        said.append(f"{m}, {s}: {pow(s, E, N)}, {yes_no[0]}, {yes_no[1]}")
    after = (words["fine"] if got == want else words["differs"]).format(got=label(got), want=label(want))
    return yours.figure(words["title"], _table(rows, words["head"]) + _note(after),
                        alt=w["alt"].format(title=words["title"], rows="; ".join(said), after=after))


def sign_picture_picture(seen) -> str:
    """The boat, and the learner's label for it beside the course's."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _CHECK_WORDS[seen.lang]
    words = w["sign"]
    got = _call(fn, BOAT_N, DESK, D, N)
    if got is _STOPPED:
        return yours.figure(words["title"], _note(words["stops"]))
    if not isinstance(got, dict):
        return yours.figure(words["title"], _note(words["other"]))
    rows, said, k = [], [], 0
    for part in ("signer", "hash", "signature"):
        mine_ = label(got[part]) if part in got else "-"
        course_ = label(BOAT_LABEL[part])
        differs = part not in got or type(got[part]) is not type(BOAT_LABEL[part]) or got[part] != BOAT_LABEL[part]
        k += differs
        rows.append([(_mono(part), _WHOLE), (_mono(mine_), _OFF if differs else ""), (_mono(course_), "")])
        said.append(f"{part}: {mine_}, {course_}")
    after = words["differs"].format(k=k) if k else words["fine"]
    body = (f'<div style="display:flex;gap:1rem;align-items:flex-start;flex-wrap:wrap">{pixels(BOAT, cell=8)}'
            f'<div style="flex:1;min-width:14rem">{_table(rows, words["head"])}</div></div>' + _note(after))
    return yours.figure(words["title"], body, alt=w["alt"].format(title=words["title"], rows="; ".join(said),
                                                                   after=after))


def verdict_picture(seen) -> str:
    """The five pictures of Exercise 3 with their labels, and the learner's verdict beside the course's."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _CHECK_WORDS[seen.lang]
    words = w["verdict"]
    rows, said, k = [], [], 0
    for picture, manifest, kind in SCENES:
        got = _call(fn, picture, manifest, KEYS_N)
        want = verdict(_plain(picture), _plain(manifest), KEYS)
        differs = got is _STOPPED or got != want
        k += differs
        shown = words["error"] if got is _STOPPED else (got if isinstance(got, str) else label(got))
        mark = EDIT if kind in ("edited", "forged") else None
        drawing = (f'<div style="display:flex;gap:0.5rem;align-items:center">{pixels(_plain(picture), cell=5, mark=mark)}'
                   f'{_mono(repr(picture))}</div>')
        rows.append([(drawing, _WHOLE), (html.escape(words["labels"][kind]), ""),
                     (_mono(shown), _OFF if differs else ""), (_mono(want), "")])
        said.append(f"{picture!r}, {words['labels'][kind]}: {shown}, {want}")
    after = words["differs"].format(k=k) if k else words["fine"]
    return yours.figure(words["title"], _table(rows, words["head"]) + _note(after),
                        alt=w["alt"].format(title=words["title"], rows="; ".join(said), after=after))


# --- the demo: change one pixel ----------------------------------------------------------------------------------------

_DEMO_WORDS = {
    "en": {"title": "The boat with pixel {i} changed",
           "kept": "The label of photo-desk.example stays with the picture.",
           "stripped": "A platform removed the label when the picture was shared.",
           "hashes": "toy_hash of this picture: {now}; the hash in the label: {then}.",
           "verdict": "verdict: {v}"},
    "nl": {"title": "De boot met pixel {i} veranderd",
           "kept": "Het label van photo-desk.example blijft bij het plaatje.",
           "stripped": "Een platform haalde het label weg toen het plaatje gedeeld werd.",
           "hashes": "toy_hash van dit plaatje: {now}; de hash in het label: {then}.",
           "verdict": "verdict: {v}"},
}


def changed_boat(i: int) -> bytes:
    """The boat with pixel i flipped: dark where it was light, and light where it was dark."""
    i = max(0, min(int(i), WIDTH * HEIGHT - 1))
    return BOAT[:i] + bytes([0 if BOAT[i] else 230]) + BOAT[i + 1:]


def demo_picture(i: int, stripped: bool, lang: str) -> str:
    """The boat with one pixel changed, its hash next to the label's, and what verdict says."""
    w = _DEMO_WORDS[lang]
    i = max(0, min(int(i), WIDTH * HEIGHT - 1))
    picture = changed_boat(i)
    manifest = None if stripped else BOAT_LABEL
    lines = [w["stripped"] if stripped else w["kept"]]
    if not stripped:
        lines.append(w["hashes"].format(now=toy_hash(picture, N), then=BOAT_LABEL["hash"]))
    lines.append(w["verdict"].format(v=verdict(picture, manifest, KEYS)))
    body = (f'<div style="display:flex;gap:1rem;align-items:center;flex-wrap:wrap">{pixels(picture, cell=12, mark=i)}'
            f'<div>{"".join(_note(x) for x in lines)}</div></div>')
    return yours.figure(w["title"].format(i=i), body, alt=w["title"].format(i=i) + ". " + " ".join(lines))


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["genuine", "sign-picture", "verdict", "no-label", "watermark"]

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
    return isinstance(e, NameError) and re.search(
        r"'(SIGNED|BOAT|EDITED|LIGHTHOUSE|BOAT_LABEL|MILL_LABEL|FORGED_LABEL|KEYS|N|E|D|C)'", str(e)) is not None


_NAMES = Mistake("lesson-names", error=_names_error,
                 en="Your function uses a name of the lesson, such as `KEYS`, `BOAT` or `N`, and the check does not give it that name: use the names between the parentheses of the `def` line.",
                 nl="Je functie gebruikt een naam van de les, zoals `KEYS`, `BOAT` of `N`, en de check geeft die naam niet mee: gebruik de namen tussen de haakjes van de `def`-regel.")

BOX_NAMES = {"toy_hash": toy_hash}   # the course's fingerprint, which Exercises 2 and 3 use


def _digits(x: int) -> int:
    """How many digits a whole number has, without writing it out: Python refuses to write more than 4300."""
    k = int(abs(x).bit_length() * math.log10(2)) + 1
    return k - 1 if 10 ** (k - 1) > abs(x) else k


def _written(v, lang: str) -> str:
    """A value the way the lessons write it in code, with a number of 13 digits or more shown by its length."""
    if isinstance(v, bool) or v is None:
        return repr(v)
    if isinstance(v, int) and abs(v) >= 10 ** 12:
        return {"en": f"<a number of {_digits(v)} digits>", "nl": f"<een getal van {_digits(v)} cijfers>"}[lang]
    if isinstance(v, str):
        return _quoted(v)
    if type(v) is dict:
        return "{" + ", ".join(f"{_written(k, lang)}: {_written(x, lang)}" for k, x in v.items()) + "}"
    if type(v) is list:
        return "[" + ", ".join(_written(x, lang) for x in v) + "]"
    return repr(v)


def _show(v) -> dict:
    """How {got} and {want} are shown: a signature without `n` grows to hundreds of digits."""
    return {lang: _written(v, lang) for lang in ("en", "nl")}


_DOCK_PAIRS = [(10, pow(10, DOCK_D, DOCK_KEY[1])), (500, pow(500, DOCK_D, DOCK_KEY[1])),
               (11, pow(10, DOCK_D, DOCK_KEY[1]))]
GENUINE_CASES = [(SIGNED_N, E, N), ([], E, N), ([(1, 1)], E, N), ([(741, 2)], E, N), ([(920, 948)], E, N),
                 (_DOCK_PAIRS, DOCK_KEY[0], DOCK_KEY[1]), ([(920, 192), (920, 192)], E, N)]
SIGN_CASES = [(BOAT_N, DESK, D, N), (LIGHTHOUSE_N, MILL, MILL_D, MILL_KEY[1]), (EDITED_N, DESK, D, N),
              (b"", DOCK, DOCK_D, DOCK_KEY[1]), (bytes([1, 2, 3]), "a.example", D, N)]
_BOTH = {DESK: (E, N), MILL: MILL_KEY}
_DOCK_PICTURE = bytes([7] * 20)
_DOCK_LABEL = sign_picture(_DOCK_PICTURE, DOCK, DOCK_D, DOCK_KEY[1])
VERDICT_CASES = [(BOAT_N, BOAT_LABEL_N, KEYS_N), (EDITED_N, BOAT_LABEL_N, KEYS_N), (BOAT_N, None, KEYS_N),
                 (LIGHTHOUSE_N, MILL_LABEL_N, KEYS_N), (EDITED_N, FORGED_LABEL_N, KEYS_N),
                 (LIGHTHOUSE_N, MILL_LABEL_N, _BOTH), (_DOCK_PICTURE, _DOCK_LABEL, {DOCK: DOCK_KEY}),
                 (LIGHTHOUSE_N, BOAT_LABEL_N, KEYS_N)]

# exercise 1: genuine
_GE_HEAD = ("def genuine(pairs, e, n):\n"
            "    found = []\n"
            "    for m, s in pairs:\n")
_GE_BODY = ("        if pow(s, e, n) == m:\n"
            "            found.append(m)\n")
_GE_TAIL = "    return found\n"
_GE_START = _GE_HEAD + "        ...\n" + _GE_TAIL
_GE = _GE_HEAD + _GE_BODY + _GE_TAIL
_GE_MESSAGE = _GE.replace("if pow(s, e, n) == m:", "if pow(m, e, n) == s:")
_GE_NOT = _GE.replace("== m:", "!= m:")
_GE_SIGNATURES = _GE.replace("found.append(m)", "found.append(s)")
_GE_WHOLE = _GE.replace("pow(s, e, n)", "pow(s, e)")

# exercise 2: sign_picture
_S_START = "def sign_picture(picture, signer, d, n):\n    ...\n"
_S = ("def sign_picture(picture, signer, d, n):\n"
      "    h = toy_hash(picture, n)\n"
      "    return {\"signer\": signer, \"hash\": h, \"signature\": pow(h, d, n)}\n")
_S_PLAIN = _S.replace("pow(h, d, n)", "h")
_S_WHOLE = _S.replace("pow(h, d, n)", "pow(h, d)")
_S_SWAPPED = _S.replace("pow(h, d, n)", "pow(d, h, n)")

# exercise 3: verdict
_V_HEAD = ("def verdict(picture, manifest, keys):\n"
           "    if manifest is None:\n"
           "        return \"no credentials\"\n"
           "    signer = manifest[\"signer\"]\n")
_V_BODY = ("    if signer not in keys:\n"
           "        return \"unknown signer\"\n"
           "    e, n = keys[signer]\n"
           "    if toy_hash(picture, n) != manifest[\"hash\"]:\n"
           "        return \"changed\"\n"
           "    if pow(manifest[\"signature\"], e, n) != manifest[\"hash\"]:\n"
           "        return \"bad signature\"\n")
_V_TAIL = "    return \"signed by \" + signer\n"
_V_START = _V_HEAD + "    ...\n" + _V_TAIL
_V = _V_HEAD + _V_BODY + _V_TAIL
_V_NO_SIGNATURE = _V.replace("    if pow(manifest[\"signature\"], e, n) != manifest[\"hash\"]:\n"
                             "        return \"bad signature\"\n", "")
_V_NO_HASH = _V.replace("    if toy_hash(picture, n) != manifest[\"hash\"]:\n        return \"changed\"\n", "")
_V_PICTURE = _V_NO_HASH.replace("!= manifest[\"hash\"]:\n        return \"bad signature\"",
                                "!= toy_hash(picture, n):\n        return \"bad signature\"")


EXERCISES = {
    "genuine": FunctionBox(
        name="genuine",
        picture=genuine_picture,
        function="genuine",
        start=_GE_START,
        solution=_GE,
        visible=[GENUINE_CASES[0]],
        hidden=GENUINE_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("message", wrong=_GE_MESSAGE,
                    en="`{call}` gives {got}, and it should give {want}: your `genuine` raises the number `m` to the power `e` and compares that with the signature. With the public key, that locks a number so that only the key's owner can read it; it does not check a signature. See the worked example.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw `genuine` verheft het getal `m` tot de macht `e` en vergelijkt dat met de handtekening. Met de public key versleutel je zo een getal, zodat alleen de eigenaar van de key het kan lezen; een handtekening controleer je er niet mee. Zie het uitgewerkte voorbeeld."),
            Mistake("not", wrong=_GE_NOT,
                    en="`{call}` gives {got}, and it should give {want}: your `genuine` keeps the numbers whose signature does not check. The exercise asks for the others.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw `genuine` houdt de getallen waarvan de handtekening niet klopt. De oefening vraagt om de andere."),
            Mistake("signatures", wrong=_GE_SIGNATURES,
                    en="`{call}` gives {got}, and it should give {want}: your `genuine` keeps the signatures. The exercise asks for the numbers they sign.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw `genuine` houdt de handtekeningen. De oefening vraagt om de getallen die ze ondertekenen."),
            Mistake("whole", wrong=_GE_WHOLE,
                    en="`{call}` gives {got}, and it should give {want}: your `genuine` leaves `n` out, so the power grows far past `n` and almost never equals the number. In the worked example, checking 192 gives 920, a number below `n`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw `genuine` laat `n` weg, dus de macht groeit tot ver boven `n` en is bijna nooit gelijk aan het getal. In het uitgewerkte voorbeeld geeft controleren van 192 het getal 920, een getal onder `n`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: keep `m` in `found` when its signature `s` checks with the key `e` and `n`.",
                      "nl": "Vervang de `...`: houd `m` in `found` als zijn handtekening `s` klopt met de key `e` en `n`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. A number belongs in the list when checking its signature with `e` and `n` gives the number back.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Een getal hoort in de lijst als controleren van zijn handtekening met `e` en `n` het getal teruggeeft."},
            "almost": {"en": "Right for `genuine(SIGNED, 17, 2773)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `pairs`, `e` and `n` it gets?",
                       "nl": "Goed voor `genuine(SIGNED, 17, 2773)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `pairs`, `e` en `n` die ze krijgt?"},
            "correct": {"en": "Your `genuine` is right in all {n} cases. Of the five numbers in `SIGNED`, two check: 920, the paper's example, and 1110, the hash of the boat. 921 fails with the signature of 920, 300 with a signature one higher than its own, and 2357 with the boat's signature: a signature fits one number and one key.",
                        "nl": "Je `genuine` klopt in alle {n} gevallen. Van de vijf getallen in `SIGNED` kloppen er twee: 920, het voorbeeld uit het paper, en 1110, de hash van de boot. 921 faalt met de handtekening van 920, 300 met een handtekening die één hoger is dan de zijne, en 2357 met de handtekening van de boot: een handtekening past bij één getal en één key."},
        },
    ),
    "sign-picture": FunctionBox(
        name="sign-picture",
        picture=sign_picture_picture,
        function="sign_picture",
        start=_S_START,
        solution=_S,
        visible=[SIGN_CASES[0]],
        hidden=SIGN_CASES[1:],
        names=BOX_NAMES,
        show=_show,
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("plain", wrong=_S_PLAIN,
                    en="`{call}` gives {got}, and it should give {want}: in your label, the `\"signature\"` is the same number as the `\"hash\"`. Anyone could have written that, so it says nothing about who signed.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: in jouw label is de `\"signature\"` hetzelfde getal als de `\"hash\"`. Dat had iedereen kunnen schrijven, dus het zegt niets over wie tekende."),
            Mistake("whole", wrong=_S_WHOLE,
                    en="`{call}` gives {got}, and it should give {want}: your `\"signature\"` is a number far larger than `n`. In the worked example, the signature of 920 is 192, a number below `n`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw `\"signature\"` is een getal dat veel groter is dan `n`. In het uitgewerkte voorbeeld is de handtekening van 920 het getal 192, een getal onder `n`."),
            Mistake("swapped", wrong=_S_SWAPPED,
                    en="`{call}` gives {got}, and it should give {want}: your `\"signature\"` raises `d` to the power of the hash. Compare the order of the numbers in your `pow` with the worked example.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw `\"signature\"` verheft `d` tot de macht van de hash. Vergelijk de volgorde van de getallen in je `pow` met het uitgewerkte voorbeeld."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: compute the picture's hash with `toy_hash(picture, n)`, sign it with `d`, and give back a dict with `\"signer\"`, `\"hash\"` and `\"signature\"`.",
                      "nl": "Vervang de `...`: bereken de hash van het plaatje met `toy_hash(picture, n)`, onderteken hem met `d`, en geef een dict terug met `\"signer\"`, `\"hash\"` en `\"signature\"`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The label holds the signer, the picture's hash below `n`, and the signature of that hash with `d`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Het label bevat de ondertekenaar, de hash van het plaatje onder `n`, en de handtekening van die hash met `d`."},
            "almost": {"en": "Right for the boat. But `{call}` gives {got}, and it should give {want}. Does your function use the `picture`, `signer`, `d` and `n` it gets?",
                       "nl": "Goed voor de boot. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `picture`, `signer`, `d` en `n` die ze krijgt?"},
            "correct": {"en": "Your `sign_picture` is right in all {n} cases. The boat's hash below 2773 is 1110, and its signature is 1378: anyone can check that with `e` = 17, but only whoever holds `d` = 157 could make it.",
                        "nl": "Je `sign_picture` klopt in alle {n} gevallen. De hash van de boot onder 2773 is 1110, en zijn handtekening is 1378: iedereen kan dat controleren met `e` = 17, maar alleen wie `d` = 157 heeft, kon hem maken."},
        },
    ),
    "verdict": FunctionBox(
        name="verdict",
        picture=verdict_picture,
        function="verdict",
        start=_V_START,
        solution=_V,
        visible=[VERDICT_CASES[0]],
        hidden=VERDICT_CASES[1:],
        names=BOX_NAMES,
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("no-signature", wrong=_V_NO_SIGNATURE,
                    en="`{call}` gives {got}, and it should give {want}: the hash in this label matches the picture, but its signature does not check, and your `verdict` does not notice. Anyone can write a hash into a label; only the key's owner can make a signature that checks.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de hash in dit label past bij het plaatje, maar de handtekening klopt niet, en jouw `verdict` merkt dat niet. Iedereen kan een hash in een label schrijven; alleen de eigenaar van de key kan een handtekening maken die klopt."),
            Mistake("no-hash", wrong=_V_NO_HASH,
                    en="`{call}` gives {got}, and it should give {want}: this picture is not the one that was signed. Its own hash differs from the hash in the label, and your `verdict` does not compare the two.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dit plaatje is niet het plaatje dat ondertekend werd. Zijn eigen hash verschilt van de hash in het label, en jouw `verdict` vergelijkt die twee niet."),
            Mistake("picture", wrong=_V_PICTURE,
                    en="`{call}` gives {got}, and it should give {want}: the signature in this label is fine, but the picture changed since it was signed. Your `verdict` calls that a bad signature; it should say first that the picture changed.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de handtekening in dit label is in orde, maar het plaatje veranderde nadat het ondertekend werd. Jouw `verdict` noemt dat een foute handtekening; hij moet eerst zeggen dat het plaatje veranderd is."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: look up the signer in `keys`, compare the picture's hash with the label's, and check the signature.",
                      "nl": "Vervang de `...`: zoek de ondertekenaar op in `keys`, vergelijk de hash van het plaatje met die in het label, en controleer de handtekening."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. In this order: a signer not in `keys` is unknown, a picture whose hash differs from the label's has changed, and a signature that does not give the label's hash back is bad.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. In deze volgorde: een ondertekenaar die niet in `keys` staat is onbekend, een plaatje met een andere hash dan die in het label is veranderd, en een handtekening die de hash in het label niet teruggeeft is fout."},
            "almost": {"en": "Right for the boat with its own label. But `{call}` gives {got}, and it should give {want}. Does your function use the `picture`, `manifest` and `keys` it gets?",
                       "nl": "Goed voor de boot met zijn eigen label. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `picture`, `manifest` en `keys` die ze krijgt?"},
            "correct": {"en": "Your `verdict` is right in all {n} cases. Of the five pictures in the table, only one gets \"signed by\": the lighthouse is signed and unchanged too, but by a signer not in `KEYS`. Even that verdict says only who vouches for the picture and that it has not changed since; it does not say that what the picture shows happened.",
                        "nl": "Je `verdict` klopt in alle {n} gevallen. Van de vijf plaatjes in de tabel krijgt er maar één \"signed by\": ook de vuurtoren is ondertekend en onveranderd, maar door een ondertekenaar die niet in `KEYS` staat. Zelfs dat verdict zegt alleen wie voor het plaatje instaat en dat het sindsdien niet veranderd is; het zegt niet dat wat het plaatje laat zien echt gebeurde."},
        },
    ),
}

CASES = [
    ("genuine", _GE_START, "empty", None),
    ("genuine", _GE, "correct", None),
    ("genuine", "def genuine(pairs, e, n):\n    return [m for m, s in pairs if s ** e % n == m]\n", "correct", None),
    ("genuine", _GE_MESSAGE, "wrong", "message"),
    ("genuine", _GE_NOT, "wrong", "not"),
    ("genuine", _GE_SIGNATURES, "wrong", "signatures"),
    ("genuine", _GE_WHOLE, "wrong", "whole"),
    ("genuine", "def genuine(pairs, e, n):\n    return [920, 1110]\n", "almost", None),
    ("genuine", "def genuine(pairs, e, n):\n    return SIGNED\n", "error", "lesson-names"),
    ("sign-picture", _S_START, "empty", None),
    ("sign-picture", _S, "correct", None),
    ("sign-picture", "def sign_picture(picture, signer, d, n):\n    h = toy_hash(picture, n)\n    return dict(signer=signer, hash=h, signature=h ** d % n)\n", "correct", None),
    ("sign-picture", _S_PLAIN, "wrong", "plain"),
    ("sign-picture", _S_WHOLE, "wrong", "whole"),
    ("sign-picture", _S_SWAPPED, "wrong", "swapped"),
    ("sign-picture", "def sign_picture(picture, signer, d, n):\n    return {\"signer\": \"photo-desk.example\", \"hash\": 1110, \"signature\": 1378}\n", "almost", None),
    ("sign-picture", "def sign_picture(picture, signer, d, n):\n    return BOAT\n", "error", "lesson-names"),
    ("verdict", _V_START, "empty", None),
    ("verdict", _V, "correct", None),
    ("verdict", "def verdict(picture, manifest, keys):\n    if manifest is None:\n        return \"no credentials\"\n    signer = manifest[\"signer\"]\n    if signer not in keys:\n        return \"unknown signer\"\n    e, n = keys[signer]\n    h = manifest[\"hash\"]\n    if h != toy_hash(picture, n):\n        return \"changed\"\n    if h == pow(manifest[\"signature\"], e, n):\n        return \"signed by \" + signer\n    return \"bad signature\"\n", "correct", None),
    ("verdict", _V_NO_SIGNATURE, "wrong", "no-signature"),
    ("verdict", _V_NO_HASH, "wrong", "no-hash"),
    ("verdict", _V_PICTURE, "wrong", "picture"),
    ("verdict", "def verdict(picture, manifest, keys):\n    return \"signed by photo-desk.example\"\n", "almost", None),
    ("verdict", "def verdict(picture, manifest, keys):\n    return KEYS\n", "error", "lesson-names"),
]
