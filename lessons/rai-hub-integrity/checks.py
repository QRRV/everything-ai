# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-hub-integrity: a small hub that the course makes itself, in memory, with one repository
of a made-up organization, cog-and-crank/trail-sorter, as it is on four days: on day 1 its first commit, on day 30 an
update on the branch main, on day 60 nothing (the organization was deleted), and on day 61 a repository of the same
name from someone else, with a tag v1 of its own and a model in which one number differs. A repository is a
dictionary with its branches, its tags and its commits; a commit id is a SHA-1 hash of the commit, and every file is
checked by its SHA-256, with hashlib. Nothing is downloaded."""
import contextlib
import copy
import hashlib
import html
import io
import json
import random
import re
import struct

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

# --- the small hub ---------------------------------------------------------------------------------------------------

ORG = "cog-and-crank"            # a made-up organization
REPO = f"{ORG}/trail-sorter"
MODEL = "model.safetensors"
CARD = "README.md"
TOPICS = ["ride", "parts", "service"]   # what the model sorts bike reviews into


def to_safetensors(tensors: dict) -> bytes:
    """A safetensors file of float32 numbers: 8 bytes with the length of the JSON header, lowest byte first, the
    header padded with spaces to a multiple of 8, then the numbers. `tensors` maps a name to (shape, numbers)."""
    header, buffer = {}, b""
    for name, (shape, values) in tensors.items():
        raw = struct.pack(f"<{len(values)}f", *values)
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [len(buffer), len(buffer) + len(raw)]}
        buffer += raw
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    return len(text).to_bytes(8, "little") + text + buffer


def _layer(seed: int) -> dict:
    """The 12 weights and 3 biases of a layer from 4 numbers about a review to the 3 topics, the same every time."""
    rng = random.Random(seed)
    return {"weight": ([3, 4], [round(rng.uniform(-0.5, 0.5), 4) for _ in range(12)]),
            "bias": ([3], [round(rng.uniform(-0.5, 0.5), 4) for _ in range(3)])}


def numbers(data: bytes) -> list:
    """The 15 numbers of one of these model files, as float32 turns them into Python floats."""
    n = int.from_bytes(data[:8], "little")
    return list(struct.unpack(f"<{(len(data) - 8 - n) // 4}f", data[8 + n:]))


_FIRST = _layer(1301)
_NEW = {"weight": _FIRST["weight"], "bias": ([3], _FIRST["bias"][1][:2] + [round(_FIRST["bias"][1][2] + 2.0, 4)])}
MODEL_FIRST = to_safetensors(_FIRST)           # model.safetensors of day 1
MODEL_UPDATE = to_safetensors(_layer(1330))    # of day 30: trained again, every number new
MODEL_NEW = to_safetensors(_NEW)               # of day 61: day 1's numbers, with the bias of "service" 2 larger
CARD_FIRST = (b"# trail-sorter\n\nA small model by cog-and-crank that sorts bike reviews into three topics: "
              b"ride, parts and service.\n")
CARD_UPDATE = CARD_FIRST + b"\nVersion 2: trained on more reviews.\n"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def commit_id(parent, message: str, files: dict) -> str:
    """A commit id as the small hub makes it: the SHA-1 of the commit before it, its message and the SHA-256 of each
    file, so that an id always names the same files. (Git hashes a commit in its own way; the idea is the same.)"""
    commit = {"parent": parent, "message": message, "files": {name: sha256(data) for name, data in files.items()}}
    return hashlib.sha1(json.dumps(commit, sort_keys=True).encode()).hexdigest()


_FILES_FIRST = {MODEL: MODEL_FIRST, CARD: CARD_FIRST}
_FILES_UPDATE = {MODEL: MODEL_UPDATE, CARD: CARD_UPDATE}
_FILES_NEW = {MODEL: MODEL_NEW, CARD: CARD_FIRST}   # the new owner copied the model card byte for byte
FIRST_ID = commit_id(None, "First version", _FILES_FIRST)
UPDATE_ID = commit_id(FIRST_ID, "Trained on more reviews", _FILES_UPDATE)
NEW_ID = commit_id(None, "First version", _FILES_NEW)   # the same message, one number different: another id
REPOS = {
    1: {"branches": {"main": FIRST_ID}, "tags": {"v1": FIRST_ID}, "commits": {FIRST_ID: _FILES_FIRST}},
    30: {"branches": {"main": UPDATE_ID}, "tags": {"v1": FIRST_ID},
         "commits": {FIRST_ID: _FILES_FIRST, UPDATE_ID: _FILES_UPDATE}},
    60: None,   # the organization was deleted, and its repositories with it
    61: {"branches": {"main": NEW_ID}, "tags": {"v1": NEW_ID}, "commits": {NEW_ID: _FILES_NEW}},
}
DAYS = list(REPOS)


# --- the course's own solutions ----------------------------------------------------------------------------------------

def commit_of(repo, revision):
    # the commit id that revision names in repo: a branch, a tag or a commit id; None when it names none
    if revision in repo["branches"]:
        return repo["branches"][revision]
    if revision in repo["tags"]:
        return repo["tags"][revision]
    if revision in repo["commits"]:
        return revision
    return None


def pin_now(repo, revision, name):
    # a pin for the file name: the commit that revision names today, and the SHA-256 of the file in it
    commit = commit_of(repo, revision)
    data = repo["commits"][commit][name]
    return {"revision": commit, "sha256": hashlib.sha256(data).hexdigest()}


def load_pinned(repo, pin, name):
    # the bytes of the file name at the pinned revision; None when they are not exactly the pinned file
    if repo is None:
        return None
    commit = commit_of(repo, pin["revision"])
    if commit is None:
        return None
    data = repo["commits"][commit][name]
    if hashlib.sha256(data).hexdigest() != pin["sha256"]:
        return None
    return data


def why_refused(repo, pin, name) -> str:
    """Why load_pinned gives None: "gone" (no repository), "commit" (the revision names no commit) or "sha256"."""
    if repo is None:
        return "gone"
    commit = commit_of(repo, pin["revision"])
    return "commit" if commit is None else "sha256"


def same_places(a: str, b: str) -> int:
    """In how many places two digests have the same digit."""
    return sum(x == y for x, y in zip(a, b))


def worked_example() -> list:
    """What the worked example prints, line by line."""
    return [sha256(b"hello"), sha256(b"hello!"), f"{len(MODEL_FIRST)} {sha256(MODEL_FIRST)}",
            f"{len(MODEL_NEW)} {sha256(MODEL_NEW)}"]


# --- names the check uses ----------------------------------------------------------------------------------------------

class Named:
    """A value the check calls by the name the lesson gives it, such as DAY_30, so that a message stays short. The
    learner's code gets a copy of the value itself."""

    def __init__(self, value, name: str):
        self.value, self.name = value, name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


def _plain(v):
    return v.value if isinstance(v, Named) else v


DAY_1, DAY_30, DAY_60, DAY_61 = (Named(REPOS[d], f"DAY_{d}") for d in DAYS)
FIRST, UPDATE, NEW_OWNER = Named(FIRST_ID, "FIRST"), Named(UPDATE_ID, "UPDATE"), Named(NEW_ID, "NEW_OWNER")
ID_NAMES = {FIRST_ID: "FIRST", UPDATE_ID: "UPDATE", NEW_ID: "NEW_OWNER"}
PIN_DAY_1 = Named(pin_now(REPOS[1], "main", MODEL), "PIN_DAY_1")                       # made with pin_now on day 1
PIN_MAIN = Named({"revision": "main", "sha256": sha256(MODEL_FIRST)}, "PIN_MAIN")     # by hand, on a branch
PIN_V1 = Named({"revision": "v1", "sha256": sha256(MODEL_FIRST)}, "PIN_V1")           # by hand, on a tag
PIN_CARD = Named({"revision": "v1", "sha256": sha256(CARD_FIRST)}, "PIN_CARD")        # the model card, on the tag
PINS = {p.name: p.value for p in (PIN_DAY_1, PIN_MAIN, PIN_V1, PIN_CARD)}

COMMIT_CASES = [(DAY_1, "main"), (DAY_30, "main"), (DAY_30, "v1"), (DAY_30, FIRST), (DAY_30, "v2"), (DAY_61, "v1"),
                (DAY_61, FIRST), (DAY_61, NEW_OWNER)]
PIN_CASES = [(DAY_1, "main", MODEL), (DAY_30, "main", MODEL), (DAY_30, "v1", CARD), (DAY_30, UPDATE, CARD),
             (DAY_61, "v1", MODEL)]
LOAD_CASES = [(DAY_30, PIN_DAY_1, MODEL), (DAY_1, PIN_MAIN, MODEL), (DAY_30, PIN_MAIN, MODEL), (DAY_30, PIN_V1, MODEL),
              (DAY_60, PIN_DAY_1, MODEL), (DAY_61, PIN_DAY_1, MODEL), (DAY_61, PIN_V1, MODEL),
              (DAY_61, PIN_CARD, CARD)]

_LABELS = {
    MODEL_FIRST: {"en": "the model of day 1", "nl": "het model van dag 1"},
    MODEL_UPDATE: {"en": "the model of day 30", "nl": "het model van dag 30"},
    MODEL_NEW: {"en": "the new owner's model", "nl": "het model van de nieuwe eigenaar"},
    CARD_FIRST: {"en": "README.md of day 1", "nl": "README.md van dag 1"},
    CARD_UPDATE: {"en": "README.md of day 30", "nl": "README.md van dag 30"},
}


def _quoted(text: str) -> str:
    """A text in double quotes, as the lesson writes it in code."""
    return json.dumps(text, ensure_ascii=False) if len(text) <= 60 else json.dumps(text[:57], ensure_ascii=False) + "..."


def label(v, lang: str) -> str:
    """A value in a few words: a file of the hub by what it is, a commit id by its name, a pin by its two keys."""
    v = _plain(v)
    if isinstance(v, (bytes, bytearray)):
        found = _LABELS.get(bytes(v))
        return found[lang] if found else (f"{len(v)} bytes")
    if isinstance(v, str):
        return ID_NAMES.get(v) or _quoted(v)
    if isinstance(v, dict) and set(v) == {"revision", "sha256"}:
        rev, digest = v["revision"], v["sha256"]
        digest = digest[:10] + "..." if isinstance(digest, str) and len(digest) > 10 else digest
        return '{"revision": ' + label(rev, lang) + ', "sha256": ' + label(digest, lang) + "}"
    text = repr(v)
    return text if len(text) <= 60 else text[:57] + "..."


def _show(v):
    """A value for a check message: as a code span, or a file by what it is, in each language."""
    v = _plain(v)
    if isinstance(v, (bytes, bytearray)):
        return {lang: label(v, lang) for lang in ("en", "nl")}
    return f"`{label(v, 'en')}`"


# --- pictures: the hub over 61 days ------------------------------------------------------------------------------------
# Every name and every value of the learner is shown through html.escape.

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.15rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;vertical-align:top")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"
_MONO = "font-family:var(--ea-font-code);font-size:12px;overflow-wrap:anywhere"
_WHOLE = "white-space:nowrap"   # a name or a commit, never broken in the middle


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


def _commit(cid: str) -> str:
    """A commit by its name and its first 12 digits."""
    return f"{ID_NAMES[cid]} ({cid[:12]}...)"


_HUB_WORDS = {
    "en": {"title": "The repository cog-and-crank/trail-sorter on four days",
           "head": ["day", "what happened", "main", "v1", "SHA-256 of model.safetensors under main"],
           "events": {1: "cog-and-crank uploads the model and its card: the commit FIRST",
                      30: "cog-and-crank uploads an update: the commit UPDATE",
                      60: "the organization cog-and-crank is deleted, and the repository is gone",
                      61: "someone else registers the name cog-and-crank and uploads trail-sorter: the commit "
                          "NEW_OWNER"},
           "none": "no repository",
           "after": "On day 61 the model is {size} bytes, as on day 1, with the same header; one of its 15 numbers "
                    "differs, and with it the SHA-256 and the commit id. Dashed: what someone else put there.",
           "alt": "The repository on four days: {rows}. {after}"},
    "nl": {"title": "De repository cog-and-crank/trail-sorter op vier dagen",
           "head": ["dag", "wat er gebeurde", "main", "v1", "SHA-256 van model.safetensors onder main"],
           "events": {1: "cog-and-crank uploadt het model en zijn card: de commit FIRST",
                      30: "cog-and-crank uploadt een update: de commit UPDATE",
                      60: "de organisatie cog-and-crank wordt verwijderd, en de repository is weg",
                      61: "iemand anders registreert de naam cog-and-crank en uploadt trail-sorter: de commit "
                          "NEW_OWNER"},
           "none": "geen repository",
           "after": "Op dag 61 is het model {size} bytes, net als op dag 1, met dezelfde header; een van de 15 "
                    "getallen is anders, en daarmee ook de SHA-256 en de commit id. Gestreept: wat iemand anders "
                    "daar neerzette.",
           "alt": "De repository op vier dagen: {rows}. {after}"},
}


def hub_picture(lang: str) -> str:
    """The repository on day 1, 30, 60 and 61: what happened, what main and v1 name, and the SHA-256 of the model."""
    w = _HUB_WORDS[lang]
    rows, said = [], []
    for day in DAYS:
        repo = REPOS[day]
        other = day == 61
        if repo is None:
            cells = [w["none"], "", ""]
        else:
            main, tag = repo["branches"]["main"], repo["tags"]["v1"]
            cells = [_commit(main), _commit(tag), sha256(repo["commits"][main][MODEL])[:16] + "..."]
        style = _OFF if other else ""
        rows.append([(str(day), ""), (html.escape(w["events"][day]), "")]
                    + [(_mono(c), _WHOLE + (";" + style if style and c else "")) for c in cells])
        said.append(f"{day}: {w['events'][day]}; " + ", ".join(c for c in cells if c))
    after = w["after"].format(size=len(MODEL_NEW))
    body = _table(rows, w["head"]) + _note(after)
    return yours.figure(w["title"], body, alt=w["alt"].format(rows="; ".join(said), after=after))


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


def _arg(a) -> str:
    """An argument of a case the way the check message writes it: a name of the lesson, or a text in double quotes."""
    return f'"{a}"' if isinstance(a, str) else repr(a)


def _same(got, want) -> bool:
    return got is not _STOPPED and type(got) is type(want) and got == want


_CHECK_WORDS = {
    "en": {"error": "error",
           "commit": {"title": "Your commit_of on the hub", "count": "repositories and revisions",
                      "head": ["repository", "revision", "your commit_of", "the course's"]},
           "pin": {"title": "Your pin_now on the hub", "count": "pins",
                   "head": ["repository", "revision", "file", "your pin", "the course's"]},
           "load": {"title": "Your load_pinned on the hub", "count": "repositories and pins",
                    "head": ["repository", "pin", "file", "your load_pinned", "the course's"]},
           "fine": "The same as the course's on all {k} {what}.",
           "differs": "It differs from the course's on {k} of the {m} {what}: they have a dashed frame.",
           "alt": "{title}: {rows}. {after}"},
    "nl": {"error": "fout",
           "commit": {"title": "Jouw commit_of op de hub", "count": "repositories en revisions",
                      "head": ["repository", "revision", "jouw commit_of", "die van de course"]},
           "pin": {"title": "Jouw pin_now op de hub", "count": "pins",
                   "head": ["repository", "revision", "bestand", "jouw pin", "die van de course"]},
           "load": {"title": "Jouw load_pinned op de hub", "count": "repositories en pins",
                    "head": ["repository", "pin", "bestand", "jouw load_pinned", "die van de course"]},
           "fine": "Hetzelfde als die van de course bij alle {k} {what}.",
           "differs": "Het verschilt van die van de course bij {k} van de {m} {what}: die hebben een gestreept kader.",
           "alt": "{title}: {rows}. {after}"},
}


def _case_picture(seen, kind: str, cases: list, course) -> str:
    """The learner's function on each case of the check beside the course's: one row per case."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _CHECK_WORDS[seen.lang]
    words = w[kind]
    rows, said, k = [], [], 0
    for case in cases:
        got = _call(fn, *case)
        want = course(*(_plain(a) for a in case))
        differs = not _same(got, want)
        k += differs
        shown = w["error"] if got is _STOPPED else label(got, seen.lang)
        right = label(want, seen.lang)
        args = [_arg(a) for a in case]
        rows.append([(_mono(a), _WHOLE) for a in args]
                    + [(_mono(shown), _OFF if differs else ""), (_mono(right), "")])
        said.append(f"{', '.join(args)}: {shown}; {right}")
    m = len(cases)
    after = (w["differs"].format(k=k, m=m, what=words["count"]) if k
             else w["fine"].format(k=m, what=words["count"]))
    body = _table(rows, words["head"]) + _note(after)
    return yours.figure(words["title"], body, alt=w["alt"].format(title=words["title"], rows="; ".join(said),
                                                                   after=after))


def commit_picture(seen) -> str:
    return _case_picture(seen, "commit", COMMIT_CASES, commit_of)


def pin_picture(seen) -> str:
    return _case_picture(seen, "pin", PIN_CASES, pin_now)


def load_picture(seen) -> str:
    return _case_picture(seen, "load", LOAD_CASES, load_pinned)


# --- the demo: your pin over 61 days -----------------------------------------------------------------------------------

BOX_NAMES = {"commit_of": commit_of}   # what the boxes of Exercise 2 and 3 may use: the course's commit_of


def mine(box_text, correct: bool, name: str):
    """The learner's own function `name` when its exercise is right, else None: the demo then uses the course's."""
    if not correct or not isinstance(box_text, str):
        return None
    from everything_ai.check import run_code
    run_ = run_code(box_text, dict(BOX_NAMES))
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


_DEMO_WORDS = {
    "en": {"title": "What Spoke & Chain gets on each day, with a pin made on day {day}",
           "head": ["way", "day 1", "day 30", "day 60", "day 61"],
           "ways": {"main": 'by name: revision="main"', "v1": 'by tag: revision="v1"', "pin": "your pin of day {day}"},
           "gone": "nothing: no repository", "early": "no pin yet", "refused": "refused",
           "why": {"commit": "refused: no such commit", "sha256": "refused: another SHA-256"},
           "late": "This pin holds the new owner's model, and load_pinned lets it through on day 61.",
           "kept": "Your pin gives {what} for as long as the hub has that commit, and refuses everything else.",
           "dashed": "Dashed: the new owner's model.",
           "whose": "With your {names}; the course's for the rest, until yours are right.",
           "whose_none": "With the course's functions, until yours are right.", "and": " and ",
           "alt": "{title}: {rows}. {note}"},
    "nl": {"title": "Wat Spoke & Chain elke dag krijgt, met een pin van dag {day}",
           "head": ["manier", "dag 1", "dag 30", "dag 60", "dag 61"],
           "ways": {"main": 'op naam: revision="main"', "v1": 'op tag: revision="v1"', "pin": "jouw pin van dag {day}"},
           "gone": "niets: geen repository", "early": "nog geen pin", "refused": "geweigerd",
           "why": {"commit": "geweigerd: zo'n commit is er niet", "sha256": "geweigerd: een andere SHA-256"},
           "late": "Deze pin bevat het model van de nieuwe eigenaar, en load_pinned laat het op dag 61 door.",
           "kept": "Jouw pin geeft {what} zolang de hub die commit heeft, en weigert al het andere.",
           "dashed": "Gestreept: het model van de nieuwe eigenaar.",
           "whose": "Met jouw {names}; die van de course voor de rest, tot de jouwe kloppen.",
           "whose_none": "Met de functies van de course, tot de jouwe kloppen.", "and": " en ",
           "alt": "{title}: {rows}. {note}"},
}


def demo_rows(pin_day: int, commit_fn=None, pin_fn=None, load_fn=None) -> dict:
    """What the shop gets on each day by name, by tag and with a pin made on pin_day: the bytes, None when refused,
    "gone" when there is no repository, and "early" before the pin was made."""
    commit_ = _use(commit_fn, commit_of)
    pin_ = _use(pin_fn, pin_now)
    load_ = _use(load_fn, load_pinned)
    pin = pin_(REPOS[pin_day], "main", MODEL)
    out = {"main": {}, "v1": {}, "pin": {}}
    for day in DAYS:
        repo = REPOS[day]
        for way in ("main", "v1"):
            if repo is None:
                out[way][day] = "gone"
            else:
                commit = commit_(repo, way)
                out[way][day] = repo["commits"][commit][MODEL] if commit in repo["commits"] else None
        if day < pin_day:
            out["pin"][day] = "early"
        elif repo is None:
            out["pin"][day] = "gone"
        else:
            out["pin"][day] = load_(repo, pin, MODEL)
    out["pin_value"] = pin
    return out


def demo_picture(pin_day, lang: str, commit_fn=None, pin_fn=None, load_fn=None) -> str:
    """For a pin made on pin_day with pin_now on main: what the shop gets on each day, by name, by tag and by pin."""
    w = _DEMO_WORDS[lang]
    pin_day = pin_day if pin_day in (1, 30, 61) else 1
    got = demo_rows(pin_day, commit_fn, pin_fn, load_fn)
    rows, said = [], []
    for way in ("main", "v1", "pin"):
        name = w["ways"][way].format(day=pin_day)
        cells = []
        for day in DAYS:
            v = got[way][day]
            if v == "gone":
                text = w["gone"]
            elif v == "early":
                text = w["early"]
            elif v is None:
                text = w["why"][why_refused(REPOS[day], got["pin_value"], MODEL)] if way == "pin" else w["refused"]
            else:
                text = label(v, lang)
            text = re.sub(r"\b(day|dag) (\d)", "\\1\u00a0\\2", text)   # never "day" at the end of a line, "1" on the next
            cells.append((html.escape(text), _OFF if v == MODEL_NEW else ""))
        rows.append([(_mono(name), _WHOLE)] + cells)
        said.append(name + ": " + ", ".join(f"{d} {html.unescape(c)}" for d, (c, _) in zip(DAYS, cells)))
    note = w["late"] if pin_day == 61 else w["kept"].format(what=label(got["pin"][pin_day], lang))
    mine_ = [n for n, f in (("commit_of", commit_fn), ("pin_now", pin_fn), ("load_pinned", load_fn)) if f]
    joined = mine_[0] if len(mine_) == 1 else ", ".join(mine_[:-1]) + w["and"] + mine_[-1] if mine_ else ""
    whose = w["whose"].format(names=joined) if mine_ else w["whose_none"]
    title = w["title"].format(day=pin_day)
    body = _table(rows, w["head"]) + _note(note) + _note(w["dashed"]) + _note(whose)
    return yours.figure(title, body, alt=w["alt"].format(title=title, rows="; ".join(said), note=note))


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["commit-of", "pin-now", "load-pinned", "match", "pipeline"]

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
    return isinstance(e, NameError) and re.search(r"'(DAY_1|DAY_30|DAY_60|DAY_61|FIRST|UPDATE|NEW_OWNER|PIN_DAY_1|"
                                                  r"PIN_MAIN|PIN_V1|PIN_CARD|C)'", str(e)) is not None


_NAMES = Mistake("lesson-names", error=_names_error,
                 en="Your function uses a name of the lesson, such as `DAY_30` or `FIRST`, and the check does not give it that name: use the names between the parentheses of the `def` line.",
                 nl="Je functie gebruikt een naam van de les, zoals `DAY_30` of `FIRST`, en de check geeft die naam niet mee: gebruik de namen tussen de haakjes van de `def`-regel.")


def _uses(word: str):
    """True when the code never names the key `word` in quotes, such as "tags"."""
    return lambda got, text: re.search(rf"[\"']{word}[\"']", text) is None


# exercise 1: commit_of
_C_HEAD = ("def commit_of(repo, revision):\n"
           "    # the commit id that revision names in repo: a branch, a tag or a commit id; None when it names none\n")
_C_BODY = ("    if revision in repo[\"branches\"]:\n"
           "        return repo[\"branches\"][revision]\n"
           "    if revision in repo[\"tags\"]:\n"
           "        return repo[\"tags\"][revision]\n"
           "    if revision in repo[\"commits\"]:\n"
           "        return revision\n")
_C_START = _C_HEAD + "    ...\n"
_C = _C_HEAD + _C_BODY + "    return None\n"
_C_ECHO = _C_HEAD + _C_BODY + "    return revision\n"
_C_NO_TAGS = _C.replace("    if revision in repo[\"tags\"]:\n        return repo[\"tags\"][revision]\n", "")
_C_NO_COMMITS = _C.replace("    if revision in repo[\"commits\"]:\n        return revision\n", "")
_C_TAG_VALUE = _C.replace("        return repo[\"tags\"][revision]\n", "        return revision\n")

# exercise 2: pin_now
_P_HEAD = ("import hashlib\n"
           "\n"
           "def pin_now(repo, revision, name):\n"
           "    # a pin for the file name: the commit that revision names today, and the SHA-256 of the file in it\n")
_P_COMMIT = "    commit = commit_of(repo, revision)\n"
_P_DATA = "    data = repo[\"commits\"][commit][name]\n"
_P_RET = "    return {\"revision\": commit, \"sha256\": hashlib.sha256(data).hexdigest()}\n"
_P_START = _P_HEAD + "    commit = ...\n" + _P_DATA + "    return {\"revision\": commit, \"sha256\": ...}\n"
_P = _P_HEAD + _P_COMMIT + _P_DATA + _P_RET
_P_BRANCH = _P.replace("commit = commit_of(repo, revision)", "commit = revision")
_P_KEEP = _P.replace("{\"revision\": commit,", "{\"revision\": revision,")
_P_NAME = _P.replace("hashlib.sha256(data)", "hashlib.sha256(name.encode())")
_P_BYTES = _P.replace(".hexdigest()", ".digest()")
_P_OBJECT = _P.replace(".hexdigest()", "")
_P_METHOD = _P.replace(".hexdigest()", ".hexdigest")

# exercise 3: load_pinned
_L_HEAD = ("import hashlib\n"
           "\n"
           "def load_pinned(repo, pin, name):\n"
           "    # the bytes of the file name at the pinned revision; None when they are not exactly the pinned file\n"
           "    if repo is None:\n"
           "        return None\n"
           "    commit = commit_of(repo, pin[\"revision\"])\n")
_L_NONE = ("    if commit is None:\n"
           "        return None\n")
_L_DATA = "    data = repo[\"commits\"][commit][name]\n"
_L_CHECK = ("    if hashlib.sha256(data).hexdigest() != pin[\"sha256\"]:\n"
            "        return None\n")
_L_RET = "    return data\n"
_L_START = _L_HEAD + "    ...\n" + _L_DATA + "    ...\n" + _L_RET
_L = _L_HEAD + _L_NONE + _L_DATA + _L_CHECK + _L_RET
_L_NO_NONE = _L_HEAD + _L_DATA + _L_CHECK + _L_RET
_L_NO_CHECK = _L_HEAD + _L_NONE + _L_DATA + _L_RET
_L_EQUAL = _L.replace("hexdigest() != pin", "hexdigest() == pin")
_L_BYTES = _L.replace(".hexdigest()", ".digest()")
_L_REVISION = _L.replace("!= pin[\"sha256\"]", "!= pin[\"revision\"]")   # refuses every file


def _is_bytes_pin(got, text) -> bool:
    return isinstance(got, dict) and isinstance(got.get("sha256"), (bytes, bytearray))


def _is_object_pin(got, text) -> bool:
    v = got.get("sha256") if isinstance(got, dict) else None
    return v is not None and not isinstance(v, (str, bytes, bytearray)) and hasattr(v, "hexdigest")


def _is_method_pin(got, text) -> bool:
    v = got.get("sha256") if isinstance(got, dict) else None
    return callable(v) and not hasattr(v, "hexdigest")


EXERCISES = {
    "commit-of": FunctionBox(
        name="commit-of",
        picture=commit_picture,
        function="commit_of",
        start=_C_START,
        solution=_C,
        visible=[COMMIT_CASES[0]],
        hidden=COMMIT_CASES[1:],
        show=_show,
        first=[_DOTS,
               Mistake("no-tags", matches=_uses("tags"),
                       en="Your function never looks at `repo[\"tags\"]`. A revision such as `\"v1\"` is a tag: a name for a commit, just as a branch is, and `commit_of` should find the commit it names.",
                       nl="Je functie kijkt nergens naar `repo[\"tags\"]`. Een revision zoals `\"v1\"` is een tag: een naam voor een commit, net als een branch, en `commit_of` hoort de commit te vinden die hij noemt."),
               Mistake("no-commits", matches=_uses("commits"),
                       en="Your function never looks at `repo[\"commits\"]`. A revision can also be a commit id itself, such as `FIRST`: then it names that commit, when the repository has it.",
                       nl="Je functie kijkt nergens naar `repo[\"commits\"]`. Een revision kan ook zelf een commit id zijn, zoals `FIRST`: dan noemt hij die commit, als de repository hem heeft.")],
        mistakes=[
            _NAMES,
            Mistake("echo", wrong=_C_ECHO,
                    en="`{call}` gives {got}, and it should give `None`: `\"v2\"` names nothing in this repository, neither a branch nor a tag nor a commit. Give back a revision itself only when it is one of the commits.",
                    nl="`{call}` geeft {got}, en dat moet `None` zijn: `\"v2\"` noemt niets in deze repository, geen branch, geen tag en geen commit. Geef een revision alleen zelf terug als het een van de commits is."),
            Mistake("tag-value", wrong=_C_TAG_VALUE,
                    en="`{call}` gives {got}, and it should give {want}: `\"v1\"` is the name of the tag, and the commit it names is in `repo[\"tags\"]`, under that name.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `\"v1\"` is de naam van de tag, en de commit die hij noemt staat in `repo[\"tags\"]`, onder die naam."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back the commit id that `revision` names, whether it is a branch, a tag or a commit id itself; and `None` when it names none.",
                      "nl": "Vervang de `...`: geef de commit id terug die `revision` noemt, of het nu een branch, een tag of zelf een commit id is; en `None` als hij er geen noemt."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Look in the branches, the tags and the commits of `repo`, in that order.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Kijk in de branches, de tags en de commits van `repo`, in die volgorde."},
            "almost": {"en": "Right for `commit_of(DAY_1, \"main\")`. But `{call}` gives {got}, and it should give {want}. Look at that row in the picture under the check.",
                       "nl": "Goed voor `commit_of(DAY_1, \"main\")`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Kijk naar die rij in het plaatje onder de check."},
            "unreached": {"en": "`{call}` gives `None`, and that revision does name a commit in this repository. A revision can be a branch, a tag or a commit id: does your function look at all three?",
                          "nl": "`{call}` geeft `None`, en die revision noemt wel een commit in deze repository. Een revision kan een branch, een tag of een commit id zijn: kijkt je functie naar alle drie?"},
            "correct": {"en": "Your `commit_of` is right on all {n} repositories and revisions. On day 61, `\"v1\"` names the new owner's commit, and `FIRST` names nothing any more.",
                        "nl": "Je `commit_of` klopt bij alle {n} repositories en revisions. Op dag 61 noemt `\"v1\"` de commit van de nieuwe eigenaar, en `FIRST` noemt niets meer."},
        },
    ),
    "pin-now": FunctionBox(
        name="pin-now",
        picture=pin_picture,
        function="pin_now",
        start=_P_START,
        solution=_P,
        names=BOX_NAMES,
        visible=[PIN_CASES[0]],
        hidden=PIN_CASES[1:],
        show=_show,
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("branch-name", error=lambda e: isinstance(e, KeyError) and bool(e.args) and e.args[0] in ("main", "v1"),
                    en="`{call}` stops with a KeyError on the revision itself. A branch or a tag is a name that points at a commit, not a commit id, and a revision can be any of the three. `commit_of` finds the commit that a revision names today: that is the commit the pin needs.",
                    nl="`{call}` stopt met een KeyError op de revision zelf. Een branch of een tag is een naam die naar een commit wijst, geen commit id, en een revision kan elk van de drie zijn. `commit_of` vindt de commit die een revision vandaag noemt: die commit heeft de pin nodig."),
            Mistake("keep-revision", wrong=_P_KEEP,
                    en="`{call}` gives {got}, and it should give {want}: your pin keeps `\"main\"`, the branch, and a branch moves on. By day 30 it names another commit, and the pin would follow it. Keep the commit it names today.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je pin bewaart `\"main\"`, de branch, en een branch gaat mee. Op dag 30 noemt hij een andere commit, en de pin zou meegaan. Bewaar de commit die hij vandaag noemt."),
            Mistake("name-hash", wrong=_P_NAME,
                    en="`{call}` gives {got}, and it should give {want}: your pin holds the SHA-256 of the name `\"model.safetensors\"`, and the name stays the same whatever is in the file.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je pin bevat de SHA-256 van de naam `\"model.safetensors\"`, en de naam blijft hetzelfde, wat er ook in het bestand staat."),
            Mistake("bytes", matches=_is_bytes_pin,
                    en="`{call}` gives a pin whose `\"sha256\"` is 32 bytes. A pin keeps the SHA-256 the way the worked example printed it, as 64 hexadecimal digits: the course's pins in `everything_ai/hub.py` do too.",
                    nl="`{call}` geeft een pin waarvan `\"sha256\"` 32 bytes is. Een pin bewaart de SHA-256 zoals het uitgewerkte voorbeeld hem printte, als 64 hexadecimale cijfers: de pins van de course in `everything_ai/hub.py` ook."),
            Mistake("object", matches=_is_object_pin,
                    en="`{call}` gives a pin whose `\"sha256\"` is the hash object of `hashlib`, not yet its digits. A pin keeps the SHA-256 the way the worked example printed it.",
                    nl="`{call}` geeft een pin waarvan `\"sha256\"` het hash-object van `hashlib` is, nog niet de cijfers. Een pin bewaart de SHA-256 zoals het uitgewerkte voorbeeld hem printte."),
            Mistake("method", matches=_is_method_pin,
                    en="`{call}` gives a pin whose `\"sha256\"` is a method, not its result: the parentheses after it are missing.",
                    nl="`{call}` geeft een pin waarvan `\"sha256\"` een methode is, niet de uitkomst ervan: de haakjes erachter ontbreken."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: `commit`, the commit id that `revision` names today, and the SHA-256 of the file, as 64 hexadecimal digits.",
                      "nl": "Vervang de twee `...`: `commit`, de commit id die `revision` vandaag noemt, en de SHA-256 van het bestand, als 64 hexadecimale cijfers."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. A pin holds the commit that the revision names today, and the SHA-256 of the file's bytes.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Een pin bevat de commit die de revision vandaag noemt, en de SHA-256 van de bytes van het bestand."},
            "almost": {"en": "Right for `pin_now(DAY_1, \"main\", \"model.safetensors\")`. But `{call}` gives {got}, and it should give {want}. Look at that row in the picture under the check.",
                       "nl": "Goed voor `pin_now(DAY_1, \"main\", \"model.safetensors\")`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Kijk naar die rij in het plaatje onder de check."},
            "correct": {"en": "Your `pin_now` is right on all {n} pins. A pin made on day 1 keeps `FIRST`, whatever `main` names later.",
                        "nl": "Je `pin_now` klopt bij alle {n} pins. Een pin van dag 1 bewaart `FIRST`, wat `main` later ook noemt."},
        },
    ),
    "load-pinned": FunctionBox(
        name="load-pinned",
        picture=load_picture,
        function="load_pinned",
        start=_L_START,
        solution=_L,
        names=BOX_NAMES,
        visible=[LOAD_CASES[0]],
        hidden=LOAD_CASES[1:],
        show=_show,
        first=[_DOTS,
               Mistake("bytes", matches=lambda got, text: re.search(r"\.digest\(\)", text) is not None and "fromhex" not in text,
                       en="`digest()` gives the SHA-256 as 32 bytes, and the pin holds it as 64 hexadecimal digits: the two are never equal, so your function would refuse every file. Write the SHA-256 as in Exercise 2.",
                       nl="`digest()` geeft de SHA-256 als 32 bytes, en de pin bevat hem als 64 hexadecimale cijfers: die twee zijn nooit gelijk, dus je functie zou elk bestand weigeren. Schrijf de SHA-256 zoals in oefening 2."),
               Mistake("object", matches=lambda got, text: re.search(r"hashlib\.sha256\([^()]*\)\s*[!=]=|[!=]=\s*hashlib\.sha256\([^()]*\)\s*:", text) is not None,
                       en="`hashlib.sha256(data)` is the hash object of `hashlib`, not yet its digits, and the pin holds 64 hexadecimal digits: the two are never equal, so your function would refuse every file. Write the SHA-256 as in Exercise 2.",
                       nl="`hashlib.sha256(data)` is het hash-object van `hashlib`, nog niet de cijfers, en de pin bevat 64 hexadecimale cijfers: die twee zijn nooit gelijk, dus je functie zou elk bestand weigeren. Schrijf de SHA-256 zoals in oefening 2."),
               Mistake("method", matches=lambda got, text: re.search(r"\.hexdigest\b(?!\s*\()", text) is not None,
                       en="`hexdigest` without parentheses is the method itself, not the 64 digits it gives: it is never equal to the pin, so your function would refuse every file. The parentheses after it are missing.",
                       nl="`hexdigest` zonder haakjes is de methode zelf, niet de 64 cijfers die hij geeft: die is nooit gelijk aan de pin, dus je functie zou elk bestand weigeren. De haakjes erachter ontbreken.")],
        mistakes=[
            _NAMES,
            Mistake("no-none", error=lambda e: isinstance(e, KeyError) and e.args == (None,),
                    en="`{call}` stops with a KeyError: the pin's commit is not in this repository, so `commit_of` gives `None`, and `repo[\"commits\"]` has no key `None`. Give back `None` before that line.",
                    nl="`{call}` stopt met een KeyError: de commit van de pin staat niet in deze repository, dus `commit_of` geeft `None`, en `repo[\"commits\"]` heeft geen key `None`. Geef vóór die regel `None` terug."),
            Mistake("no-check", wrong=_L_NO_CHECK,
                    en="`{call}` gives {got}, and it should give {want}: `main` has moved on to the update of day 30, and that file is not the pinned one. The revision alone does not say which file you get.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `main` is meegegaan naar de update van dag 30, en dat bestand is niet het vastgelegde. De revision alleen zegt niet welk bestand je krijgt."),
            ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: give back `None` when the revision names no commit, and when the file is not exactly the pinned file.",
                      "nl": "Vervang de twee `...`: geef `None` terug als de revision geen commit noemt, en als het bestand niet precies het vastgelegde bestand is."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Give back the file only when its commit is there and its SHA-256 is the pinned one.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Geef het bestand alleen terug als de commit er is en de SHA-256 de vastgelegde is."},
            "almost": {"en": "Right for `load_pinned(DAY_30, PIN_DAY_1, \"model.safetensors\")`. But `{call}` gives {got}, and it should give {want}. Look at that row in the picture under the check.",
                       "nl": "Goed voor `load_pinned(DAY_30, PIN_DAY_1, \"model.safetensors\")`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Kijk naar die rij in het plaatje onder de check."},
            "unreached": {"en": "`{call}` gives `None`, and it should give the file: the pin's revision names a commit with exactly the pinned file in it. Does your function let that file through?",
                          "nl": "`{call}` geeft `None`, en het moet het bestand geven: de revision van de pin noemt een commit met precies het vastgelegde bestand erin. Laat je functie dat bestand door?"},
            "correct": {"en": "Your `load_pinned` is right on all {n} repositories and pins. It refuses the update, the new owner's model and a missing commit, and lets through the model card of day 61: the same bytes as on day 1.",
                        "nl": "Je `load_pinned` klopt bij alle {n} repositories en pins. Hij weigert de update, het model van de nieuwe eigenaar en een ontbrekende commit, en laat de model card van dag 61 door: dezelfde bytes als op dag 1."},
        },
    ),
}

CASES = [
    ("commit-of", _C_START, "empty", None),
    ("commit-of", _C, "correct", None),
    ("commit-of", "def commit_of(repo, revision):\n    for kind in (\"branches\", \"tags\"):\n        if revision in repo[kind]:\n            return repo[kind][revision]\n    if revision in repo[\"commits\"]:\n        return revision\n", "correct", None),
    ("commit-of", "def commit_of(repo, revision):\n    if revision in repo[\"commits\"]:\n        return revision\n    return repo[\"branches\"].get(revision, repo[\"tags\"].get(revision))\n", "correct", None),
    ("commit-of", _C_NO_TAGS, "wrong", "no-tags"),
    ("commit-of", _C_NO_COMMITS, "wrong", "no-commits"),
    ("commit-of", _C_ECHO, "wrong", "echo"),
    ("commit-of", _C_TAG_VALUE, "wrong", "tag-value"),
    ("commit-of", "def commit_of(repo, revision):\n    # repo[\"tags\"], repo[\"commits\"]\n    return repo[\"branches\"][\"main\"]\n", "almost", None),
    ("commit-of", "def commit_of(repo, revision):\n    # repo[\"tags\"], repo[\"commits\"]\n    return repo[\"branches\"][revision]\n", "error", None),
    ("commit-of", "def commit_of(repo, revision):\n    # repo[\"tags\"], repo[\"commits\"]\n    return DAY_1[\"branches\"][revision]\n", "error", "lesson-names"),
    ("pin-now", _P_START, "empty", None),
    ("pin-now", _P, "correct", None),
    ("pin-now", "import hashlib\n\ndef pin_now(repo, revision, name):\n    c = commit_of(repo, revision)\n    return {\"revision\": c, \"sha256\": hashlib.sha256(repo[\"commits\"][c][name]).hexdigest()}\n", "correct", None),
    ("pin-now", _P_BRANCH, "error", "branch-name"),
    ("pin-now", _P.replace("commit = commit_of(repo, revision)", "commit = repo[\"branches\"][revision]"), "error", "branch-name"),
    ("pin-now", _P_KEEP, "wrong", "keep-revision"),
    ("pin-now", _P_NAME, "wrong", "name-hash"),
    ("pin-now", _P_BYTES, "wrong", "bytes"),
    ("pin-now", _P_OBJECT, "wrong", "object"),
    ("pin-now", _P_METHOD, "wrong", "method"),
    ("pin-now", "def pin_now(repo, revision, name):\n    return {\"revision\": FIRST, \"sha256\": \"\"}\n", "error", "lesson-names"),
    ("pin-now", "import hashlib\n\ndef pin_now(repo, revision, name):\n    return {\"revision\": commit_of(repo, \"main\"), \"sha256\": hashlib.sha256(repo[\"commits\"][commit_of(repo, \"main\")][\"model.safetensors\"]).hexdigest()}\n", "almost", None),
    ("load-pinned", _L_START, "empty", None),
    ("load-pinned", _L, "correct", None),
    ("load-pinned", "import hashlib\n\ndef load_pinned(repo, pin, name):\n    if repo is None or commit_of(repo, pin[\"revision\"]) is None:\n        return None\n    data = repo[\"commits\"][commit_of(repo, pin[\"revision\"])][name]\n    if hashlib.sha256(data).hexdigest() == pin[\"sha256\"]:\n        return data\n    return None\n", "correct", None),
    ("load-pinned", _L_NO_NONE, "error", "no-none"),
    ("load-pinned", _L_NO_CHECK, "wrong", "no-check"),
    ("load-pinned", _L_BYTES, "wrong", "bytes"),
    ("load-pinned", _L.replace(".hexdigest()", ""), "wrong", "object"),
    ("load-pinned", _L.replace("if hashlib.sha256(data).hexdigest() != pin[\"sha256\"]", "if pin[\"sha256\"] != hashlib.sha256(data)"), "wrong", "object"),
    ("load-pinned", _L.replace(".hexdigest()", ".hexdigest"), "wrong", "method"),
    ("load-pinned", _L.replace(".hexdigest() != pin[\"sha256\"]", ".digest() != bytes.fromhex(pin[\"sha256\"])"), "correct", None),
    ("load-pinned", _L_REVISION, "wrong", "unreached"),
    ("load-pinned", _L_EQUAL, "wrong", "unreached"),
    ("load-pinned", _L.replace("pin[\"sha256\"]", "PIN_DAY_1[\"sha256\"]"), "error", "lesson-names"),
]
