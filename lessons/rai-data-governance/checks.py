# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-data-governance: the catalog of the made-up library from rai-privacy-by-design, with who
looks after each dataset and which roles may read it, and the library's staff. The three functions the learner
writes: may_read (may a staff member read a dataset, by role), holding (the datasets that hold a member's card, for
an erasure request) and no_owner (those of them that nobody owns); with the pictures of what they give. All names
and numbers are made up."""
import copy
import html
import re

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, run_code, with_time_limit


class Named(list):
    """A list the check calls by the name the lesson gives it, such as CATALOG, so that a message stays short. The
    learner's code gets its own copy, a plain list."""

    def __init__(self, values, name: str):
        super().__init__(values)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(list(self), memo)


class Entry(dict):
    """One dataset or one staff member, which the check calls by its place, such as CATALOG[0]. The learner's code
    gets its own copy, a plain dict."""

    def __init__(self, values: dict, name: str):
        super().__init__(values)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(dict(self), memo)


# the members of the library, by card number, as in rai-privacy-by-design; three of them left and were deleted from
# the member file, so the files made after that hold only the seven who stay
MEMBER_NAMES = {"L-1001": "Anouk de Vries", "L-1002": "Bram Hendriks", "L-1003": "Yara El Amrani",
                "L-1004": "Joris van Dam", "L-1005": "Lina Postma", "L-1006": "Sem Brouwer", "L-1007": "Mila Koster",
                "L-1008": "Tim de Graaf", "L-1009": "Nora Willems", "L-1010": "Ravi Sitaram"}
MEMBER_CARDS = {name: card for card, name in MEMBER_NAMES.items()}   # for the demo: a name gives the card
_ALL = list(MEMBER_NAMES)
_STAY = ["L-1001", "L-1002", "L-1004", "L-1005", "L-1007", "L-1008", "L-1010"]

# the staff and their roles; Mees Peters is a new volunteer with no role yet
_STAFF = [("Noor Jansen", ["desk"]), ("Daan Smit", ["desk", "lending"]), ("Fenna Bakker", ["lending"]),
          ("Ilse de Wit", ["data"]), ("Omar Haddad", ["data"]), ("Sanne Visser", ["communications"]),
          ("Lotte Dekker", ["communications"]), ("Kees Mulder", ["it"]), ("Mees Peters", [])]
STAFF = Named([Entry({"name": name, "roles": roles}, f"STAFF[{i}]") for i, (name, roles) in enumerate(_STAFF)],
              "STAFF")

# the catalog: every dataset, its owner, steward and custodian (None when nobody has that role), the cards of the
# members whose data it holds, and the roles that may read it. "recommender" is the model trained on recommend-train:
# it holds the cards of the members whose rows it was trained on. The backup was made in October 2026, before the
# members who left were deleted; the survey of 2024 sits in a shared folder that nobody took charge of.
FIELDS = ["name", "owner", "steward", "custodian", "cards", "access"]
_CATALOG = [
    ("members", "Fenna Bakker", "Daan Smit", "Kees Mulder", _STAY, ["desk", "lending"]),
    ("loans", "Fenna Bakker", "Daan Smit", "Kees Mulder", _STAY, ["desk", "lending"]),
    ("newsletter", "Sanne Visser", "Lotte Dekker", "Kees Mulder", ["L-1002", "L-1010"], ["communications"]),
    ("recommend-train", "Ilse de Wit", "Omar Haddad", "Kees Mulder", _STAY, ["data"]),
    ("recommender", "Ilse de Wit", "Omar Haddad", "Kees Mulder", _STAY, ["data"]),
    ("backup-2026-10", None, None, "Kees Mulder", _ALL, ["it"]),
    ("survey-2024", None, None, "Kees Mulder", ["L-1004", "L-1007", "L-1009"], ["communications", "lending"]),
    ("opening-hours", "Fenna Bakker", "Noor Jansen", "Kees Mulder", [],
     ["desk", "lending", "data", "communications", "it"]),
]
CATALOG = Named([Entry(dict(zip(FIELDS, copy.deepcopy(row))), f"CATALOG[{i}]") for i, row in enumerate(_CATALOG)],
                "CATALOG")

JORIS = "L-1004"   # the member who asks the library to erase all data

# --- the course's own functions -----------------------------------------------------------------------------------------

_R_HEAD = ("def may_read(person, dataset):\n"
           "    # person[\"roles\"] lists the roles of a staff member; dataset[\"access\"] the roles that may read it\n"
           "    for role in person[\"roles\"]:\n")
_R_START = _R_HEAD + "        ...\n    return False\n"
_R = _R_HEAD + "        if role in dataset[\"access\"]:\n            return True\n    return False\n"
may_read = run_code(_R).names["may_read"]

_H_HEAD = ("def holding(catalog, card):\n"
           "    # each dataset has a \"name\", and in \"cards\" the card numbers of the members whose data it holds\n"
           "    names = []\n"
           "    for dataset in catalog:\n")
_H_START = _H_HEAD + "        ...\n    return names\n"
_H = _H_HEAD + "        if card in dataset[\"cards\"]:\n            names.append(dataset[\"name\"])\n    return names\n"
holding = run_code(_H).names["holding"]

_N_HEAD = ("def no_owner(catalog, card):\n"
           "    # dataset[\"owner\"] is None when nobody owns the dataset\n"
           "    names = []\n"
           "    for dataset in catalog:\n")
_N_START = _N_HEAD + "        ...\n    return names\n"
_N = _N_HEAD + ("        if card in dataset[\"cards\"]:\n"
                "            if dataset[\"owner\"] is None:\n"
                "                names.append(dataset[\"name\"])\n") + "    return names\n"
no_owner = run_code(_N).names["no_owner"]


def readable_pairs() -> int:
    """How many (staff member, dataset) pairs may_read allows."""
    return sum(may_read(dict(p), dict(d)) for p in STAFF for d in CATALOG)


# --- formatting -------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CODE = "font-family:var(--ea-font-code);font-size:0.8rem"
_LABEL = "font-family:var(--ea-font-code);font-size:0.72rem;color:var(--ea-ink-2)"


def _code(text: str) -> str:
    return f'<span style="{_CODE}">{html.escape(text)}</span>'


def _note(parts: list) -> str:
    return f'<div style="{_UI}">{html.escape(" ".join(p for p in parts if p))}</div>'


def _fn(seen):
    mine = seen.mine
    if mine is None:
        return None
    fn = mine.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _shown(v) -> str:
    """A value a function gave back, the way Python shows it, kept short."""
    try:
        text = repr(v)
    except Exception:
        text = "?"
    return text if len(text) <= 60 else text[:57] + "..."


_STOPPED = object()   # what a function gave back when it stopped with an error


def _call(fn, *args):
    try:
        return with_time_limit(lambda: fn(*copy.deepcopy(args)))
    except Exception:
        return _STOPPED


def _plain(entry) -> dict:
    return copy.deepcopy(dict(entry))


def _catalog() -> list:
    return [_plain(d) for d in CATALOG]


def _square(filled: bool, wrong: bool) -> str:
    fill = "background:var(--ea-ink)" if filled else "background:transparent"
    edge = "border:1.5px dashed var(--ea-loss)" if wrong else "border:1px solid var(--ea-line)"
    return f'<div style="width:0.9rem;height:0.9rem;border-radius:2px;{fill};{edge}"></div>'


def _dash(wrong: bool) -> str:
    frame = "border:1.5px dashed var(--ea-loss);border-radius:2px;" if wrong else ""
    return (f'<div style="width:1rem;height:1rem;display:flex;align-items:center;justify-content:center;{frame}">'
            f'<span style="color:var(--ea-ink-3)">–</span></div>')


# --- the picture under exercise 1: may_read -----------------------------------------------------------------------------

_ACCESS_WORDS = {
    "en": {"title": "Your may_read: who may read which dataset",
           "said": "Each row is a dataset, each column a staff member. A filled square: your may_read lets that person "
                   "read that dataset, {k} of the {n} pairs.",
           "none": "no role",
           "fine": "It gives the same answers as the course's may_read.",
           "wrong": "A dashed purple frame marks a square where it differs from the course's may_read.",
           "off": "Your may_read gives {got} for {call}; the picture draws True or False.",
           "stops": "Your may_read stops with an error on {call}: the check above says where."},
    "nl": {"title": "Jouw may_read: wie welke dataset mag lezen",
           "said": "Elke rij is een dataset, elke kolom een medewerker. Een gevuld vakje: jouw may_read laat die "
                   "persoon die dataset lezen, {k} van de {n} paren.",
           "none": "geen rol",
           "fine": "Hij geeft dezelfde antwoorden als de may_read van de course.",
           "wrong": "Een gestreept paars kader wijst een vakje aan waar hij verschilt van de may_read van de course.",
           "off": "Jouw may_read geeft {got} voor {call}; het plaatje tekent True of False.",
           "stops": "Jouw may_read stopt met een fout bij {call}: de check hierboven zegt waar."},
}


def initials(name: str) -> str:
    """Noor Jansen gives NJ, Ilse de Wit gives IW: the first letter of the first and the last word."""
    words = name.split()
    return words[0][0] + words[-1][0]


def access_grid(fn) -> tuple:
    """What `fn` (a may_read) answers for every dataset and staff member: (rows, allowed, wrong), rows being one list per
    dataset with (allowed, differs from the course's) per staff member. When `fn` stops or gives no bool: ("stops" or
    "off", the call, what it gave)."""
    rows, allowed, wrong = [], 0, 0
    for j, dataset in enumerate(CATALOG):
        row = []
        for i, person in enumerate(STAFF):
            got = _call(fn, _plain(person), _plain(dataset))
            call = f"may_read(STAFF[{i}], CATALOG[{j}])"
            if got is _STOPPED:
                return ("stops", call, None)
            if not isinstance(got, bool):
                return ("off", call, got)
            differs = got != may_read(_plain(person), _plain(dataset))
            row.append((got, differs))
            allowed += got
            wrong += differs
        rows.append(row)
    return rows, allowed, wrong


def access_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _ACCESS_WORDS[seen.lang]
    got = access_grid(fn)
    if isinstance(got[0], str):
        what, call, value = got
        note = w["stops"].format(call=call) if what == "stops" else w["off"].format(got=_shown(value), call=call)
        return yours.figure(w["title"], _note([note]))
    rows, allowed, wrong = got
    head = "".join(f'<th style="{_LABEL};font-weight:400;padding:0 0.2rem;text-align:center">'
                   f'{html.escape(initials(p["name"]))}</th>' for p in STAFF)
    body = []
    for dataset, row in zip(CATALOG, rows):
        cells = "".join(f'<td style="padding:0.15rem 0.2rem"><div style="display:flex;justify-content:center">'
                        f'{_square(ok, differs)}</div></td>' for ok, differs in row)
        body.append(f'<tr><td style="{_LABEL};padding:0.1rem 0.6rem 0.1rem 0;white-space:nowrap">'
                    f'{html.escape(dataset["name"])}</td>{cells}</tr>')
    table = (f'<table style="border-collapse:collapse;margin-top:0.4rem"><thead><tr><th></th>{head}</tr></thead>'
             f'<tbody>{"".join(body)}</tbody></table>')
    legend = " · ".join(f'{initials(p["name"])} {p["name"]} ({", ".join(p["roles"]) or w["none"]})' for p in STAFF)
    notes = [w["said"].format(k=allowed, n=len(STAFF) * len(CATALOG)), w["wrong"] if wrong else w["fine"]]
    return yours.figure(w["title"], table + f'<div style="{_UI}">{html.escape(legend)}</div>' + _note(notes))


# --- the pictures under exercises 2 and 3: the catalog for Joris van Dam's card -----------------------------------------

_HOLD_WORDS = {
    "en": {"title": "Your holding: where the data of Joris van Dam is",
           "said": "Filled: the {k} datasets your holding gives for L-1004, the card of Joris van Dam. Beside each, its "
                   "owner.",
           "fine": "The same datasets as the course's holding.",
           "wrong": "A dashed purple frame marks a dataset where it differs from the course's holding.",
           "off": "Your holding gives {got}; the picture draws a list of dataset names.",
           "stops": "Your holding stops with an error on CATALOG: the check above says where."},
    "nl": {"title": "Jouw holding: waar de gegevens van Joris van Dam staan",
           "said": "Gevuld: de {k} datasets die jouw holding geeft voor L-1004, de kaart van Joris van Dam. Ernaast de "
                   "eigenaar van elk.",
           "fine": "Dezelfde datasets als bij de holding van de course.",
           "wrong": "Een gestreept paars kader wijst een dataset aan waar hij verschilt van de holding van de course.",
           "off": "Jouw holding geeft {got}; het plaatje tekent een lijst namen van datasets.",
           "stops": "Jouw holding stopt met een fout bij CATALOG: de check hierboven zegt waar."},
}

_OWNER_WORDS = {
    "en": {"title": "Your no_owner: the data of Joris van Dam that nobody owns",
           "said": "Filled: the {k} datasets your no_owner gives for L-1004. An empty square: a dataset that holds "
                   "L-1004 but is not in that list; a dash: a dataset without L-1004.",
           "fine": "The same datasets as the course's no_owner.",
           "wrong": "A dashed purple frame marks a dataset where it differs from the course's no_owner.",
           "off": "Your no_owner gives {got}; the picture draws a list of dataset names.",
           "stops": "Your no_owner stops with an error on CATALOG: the check above says where."},
    "nl": {"title": "Jouw no_owner: de gegevens van Joris van Dam die van niemand zijn",
           "said": "Gevuld: de {k} datasets die jouw no_owner geeft voor L-1004. Een leeg vakje: een dataset met "
                   "L-1004 die niet in die lijst staat; een streepje: een dataset zonder L-1004.",
           "fine": "Dezelfde datasets als bij de no_owner van de course.",
           "wrong": "Een gestreept paars kader wijst een dataset aan waar hij verschilt van de no_owner van de course.",
           "off": "Jouw no_owner geeft {got}; het plaatje tekent een lijst namen van datasets.",
           "stops": "Jouw no_owner stopt met een fout bij CATALOG: de check hierboven zegt waar."},
}

_ROW = ("display:flex;gap:0.6rem;align-items:center;padding:0.2rem 0.5rem;margin:0.15rem 0;border-radius:4px;"
        "font-family:var(--ea-font-ui);font-size:0.85rem;")


def _owner_text(owner) -> str:
    return _code("None") if owner is None else html.escape(str(owner))


def _dataset_row(mark: str, dataset: dict, faded: bool, wrong: bool) -> str:
    style = _ROW + ("color:var(--ea-ink-3);" if faded else "color:var(--ea-ink);")
    style += "border:1.5px dashed var(--ea-loss)" if wrong else "border:1px solid var(--ea-line)"
    return (f'<div style="{style}"><div style="flex:0 0 1rem;display:flex;justify-content:center">{mark}</div>'
            f'<div style="flex:3;min-width:0;overflow-wrap:anywhere">{_code(dataset["name"])}</div>'
            f'<div style="flex:3;min-width:0;overflow-wrap:anywhere">{_owner_text(dataset["owner"])}</div></div>')


def _head() -> str:
    cells = "".join(f'<div style="flex:{f}">{c}</div>' for c, f in (("dataset", 3), ("owner", 3)))
    return (f'<div style="display:flex;gap:0.6rem;padding:0 0.5rem;{_LABEL};border-bottom:2px solid var(--ea-ink-3);'
            f'margin-bottom:0.2rem"><div style="flex:0 0 1rem"></div>{cells}</div>')


def _names_of(fn, words: dict):
    """What `fn` gives for the whole catalog and Joris van Dam's card, or a figure that says why it draws nothing."""
    got = _call(fn, _catalog(), JORIS)
    if got is _STOPPED:
        return yours.figure(words["title"], _note([words["stops"]]))
    if not isinstance(got, list):
        return yours.figure(words["title"], _note([words["off"].format(got=_shown(got))]))
    return got


def holding_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _HOLD_WORDS[seen.lang]
    got = _names_of(fn, w)
    if isinstance(got, str):
        return got
    want = holding(_catalog(), JORIS)
    rows, found, wrong_any = [_head()], 0, 0
    for dataset in CATALOG:
        has = dataset["name"] in got
        wrong = has != (dataset["name"] in want)
        found += has
        wrong_any += wrong
        rows.append(_dataset_row(_square(has, False), dataset, not has, wrong))
    notes = [w["said"].format(k=found), w["wrong"] if wrong_any else w["fine"]]
    return yours.figure(w["title"], f'<div style="max-width:26rem">{"".join(rows)}</div>' + _note(notes))


def owner_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _OWNER_WORDS[seen.lang]
    got = _names_of(fn, w)
    if isinstance(got, str):
        return got
    want = no_owner(_catalog(), JORIS)
    holds = holding(_catalog(), JORIS)
    rows, found, wrong_any = [_head()], 0, 0
    for dataset in CATALOG:
        name = dataset["name"]
        has = name in got
        wrong = has != (name in want)
        found += has
        wrong_any += wrong
        mark = _square(has, False) if has or name in holds else _dash(False)
        rows.append(_dataset_row(mark, dataset, name not in holds, wrong))
    notes = [w["said"].format(k=found), w["wrong"] if wrong_any else w["fine"]]
    return yours.figure(w["title"], f'<div style="max-width:26rem">{"".join(rows)}</div>' + _note(notes))


# --- the demo: whose data, and where ------------------------------------------------------------------------------------

_WHOSE_WORDS = {
    "en": {"title": "Where {who} finds the data of {name}",
           "yours": "your holding", "course": "the course's holding",
           "said": "{k} of the {n} datasets hold data of {name}; {rest}",
           "said1": "1 of the {n} datasets holds data of {name}; {rest}",
           "many": "{m} of them have no owner.", "one": "1 of them has no owner.", "none": "each of them has an owner.",
           "it": "it has no owner.", "it-owned": "it has an owner.",
           "nothing": "No dataset holds data of {name}.",
           "off": "{who} gives {got}; the picture draws a list of dataset names.",
           "stops": "{who} stops with an error."},
    "nl": {"title": "Waar {who} de gegevens van {name} vindt",
           "yours": "jouw holding", "course": "de holding van de course",
           "said": "{k} van de {n} datasets bevatten gegevens van {name}; {rest}",
           "said1": "1 van de {n} datasets bevat gegevens van {name}; {rest}",
           "it": "die heeft geen eigenaar.", "it-owned": "die heeft een eigenaar.",
           "many": "{m} daarvan hebben geen eigenaar.", "one": "1 daarvan heeft geen eigenaar.",
           "none": "elk daarvan heeft een eigenaar.",
           "nothing": "Geen enkele dataset bevat gegevens van {name}.",
           "off": "{who} geeft {got}; het plaatje tekent een lijst namen van datasets.",
           "stops": "{who} stopt met een fout."},
}


def mine(box_text, correct: bool):
    """The learner's own holding when exercise 2 is right, else None: the demo then uses the course's."""
    if not correct or not isinstance(box_text, str):
        return None
    run = run_code(box_text)
    fn = run.names.get("holding") if run.error is None else None
    return fn if callable(fn) else None


def whose_picture(card, fn, lang: str) -> str:
    """Where `fn` (the learner's holding, or None for the course's) finds the data of the member with this card, with
    who looks after each of those datasets."""
    w = _WHOSE_WORDS[lang]
    who = w["yours"] if fn is not None else w["course"]
    fn = fn or holding
    name = MEMBER_NAMES.get(card, str(card))
    title = w["title"].format(who=who, name=name)
    got = _call(fn, _catalog(), card)
    if got is _STOPPED:
        return yours.figure(title, _note([w["stops"].format(who=who.capitalize())]))
    if not isinstance(got, list):
        return yours.figure(title, _note([w["off"].format(who=who.capitalize(), got=_shown(got))]))
    found = [d for d in CATALOG if d["name"] in got]
    if not found:
        return yours.figure(title, _note([w["nothing"].format(name=name)]))
    head = "".join(f'<th style="{_LABEL};font-weight:400;text-align:left;padding:0 0.8rem 0.2rem 0">{c}</th>'
                   for c in ("dataset", "owner", "steward", "custodian"))
    body = "".join(
        f'<tr><td style="padding:0.15rem 0.8rem 0.15rem 0">{_code(d["name"])}</td>'
        + "".join(f'<td style="padding:0.15rem 0.8rem 0.15rem 0">{_owner_text(d[key])}</td>'
                  for key in ("owner", "steward", "custodian")) + "</tr>"
        for d in found)
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;font-family:var(--ea-font-ui);'
             f'font-size:0.85rem;color:var(--ea-ink)"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')
    m = sum(d["owner"] is None for d in found)
    if len(found) == 1:
        said = w["said1"].format(n=len(CATALOG), name=name, rest=w["it"] if m else w["it-owned"])
    else:
        rest = w["many"].format(m=m) if m > 1 else w["one"] if m == 1 else w["none"]
        said = w["said"].format(k=len(found), n=len(CATALOG), name=name, rest=rest)
    return yours.figure(title, table + _note([said]))


# --- the exercises ----------------------------------------------------------------------------------------------------

REQUIRED = ["may-read", "holding", "no-owner", "deadline", "model"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# exercise 1: may_read
_R_NEVER = _R_HEAD + "        if person[\"roles\"] in dataset[\"access\"]:\n            return True\n    return False\n"
_R_FIRST = _R_HEAD + ("        if role in dataset[\"access\"]:\n            return True\n"
                      "        else:\n            return False\n    return False\n")
_R_KEY = _R_HEAD + "        if role in dataset[\"roles\"]:\n            return True\n    return False\n"

# exercise 2: holding
_H_DATASET = _H_HEAD + "        if card in dataset[\"cards\"]:\n            names.append(dataset)\n    return names\n"
_H_NOT = _H_HEAD + "        if card not in dataset[\"cards\"]:\n            names.append(dataset[\"name\"])\n    return names\n"
_H_FIRST = _H_HEAD + "        if card in dataset[\"cards\"]:\n            return dataset[\"name\"]\n    return names\n"
_H_KEYS = _H_HEAD + "        if card in dataset:\n            names.append(dataset[\"name\"])\n    return names\n"
_H_INSIDE = _H_HEAD + ("        if card in dataset[\"cards\"]:\n            names.append(dataset[\"name\"])\n"
                       "        return names\n")

# exercise 3: no_owner
_N_HOLDING = _N_HEAD + "        if card in dataset[\"cards\"]:\n            names.append(dataset[\"name\"])\n    return names\n"
_N_ANY = _N_HEAD + "        if dataset[\"owner\"] is None:\n            names.append(dataset[\"name\"])\n    return names\n"
_N_OWNED = _N_HEAD + ("        if card in dataset[\"cards\"]:\n"
                      "            if dataset[\"owner\"] is not None:\n"
                      "                names.append(dataset[\"name\"])\n") + "    return names\n"
_N_INSIDE = _N_HEAD + ("        if card in dataset[\"cards\"]:\n"
                       "            if dataset[\"owner\"] is None:\n"
                       "                names.append(dataset[\"name\"])\n") + "        return names\n"


def _key_error(e: BaseException) -> bool:
    return isinstance(e, KeyError)


S, K = STAFF, CATALOG
_ROSA = {"name": "Rosa Kok", "roles": ["it", "data"]}

EXERCISES = {
    "may-read": FunctionBox(
        name="may-read",
        picture=access_picture,
        function="may_read",
        start=_R_START,
        solution=_R,
        visible=[(S[0], K[0])],
        hidden=[(S[0], K[3]), (S[1], K[6]), (S[8], K[7]), (S[7], K[5]), (S[4], K[0]), (_ROSA, K[3]), (S[6], K[2])],
        first=[_DOTS],
        mistakes=[
            Mistake("key", error=_key_error,
                    en="With `{call}`, your function stops with an error: it looks up a key that is not there. A person has `\"name\"` and `\"roles\"`; a dataset has its roles under `\"access\"`.",
                    nl="Met `{call}` stopt je functie met een fout: hij zoekt een key op die er niet is. Een persoon heeft `\"name\"` en `\"roles\"`; een dataset heeft zijn rollen onder `\"access\"`."),
            Mistake("never", wrong=_R_NEVER,
                    en="`{call}` gives {got}, and it should give {want}. Your function never gives `True`. Look up one role at a time: a whole list of roles is never one of the roles in the dataset's list.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie geeft nooit `True`. Zoek één rol tegelijk op: een hele lijst rollen is nooit een van de rollen in de lijst van de dataset."),
            Mistake("first", wrong=_R_FIRST,
                    en="`{call}` gives {got}, and it should give {want}. Does your function give `False` after the first role? A person may read the dataset through any of their roles: give back `False` only after the loop, when no role matched.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Geeft je functie `False` na de eerste rol? Iemand mag de dataset lezen via elk van de eigen rollen: geef pas na de loop `False` terug, als geen enkele rol paste."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: in the loop, give back `True` as soon as one of the person's roles may read the dataset.",
                      "nl": "Vervang de `...`: geef in de loop `True` terug zodra een van de rollen van de persoon de dataset mag lezen."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Give back `True` when at least one of the roles in `person[\"roles\"]` is in the dataset's access list, and `False` when none is.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Geef `True` terug als minstens één van de rollen in `person[\"roles\"]` in de toegangslijst van de dataset staat, en `False` als er geen een in staat."},
            "almost": {"en": "Right for `may_read(STAFF[0], CATALOG[0])`. But `{call}` gives {got}, and it should give {want}. Does your function use the `person` and the `dataset` it gets?",
                       "nl": "Goed voor `may_read(STAFF[0], CATALOG[0])`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `person` en de `dataset` die hij krijgt?"},
            "correct": {"en": "Your `may_read` is right in all {n} cases. The picture runs it for every staff member and every dataset.",
                        "nl": "Je `may_read` klopt in alle {n} gevallen. Het plaatje laat hem lopen voor elke medewerker en elke dataset."},
        },
    ),
    "holding": FunctionBox(
        name="holding",
        picture=holding_picture,
        function="holding",
        start=_H_START,
        solution=_H,
        visible=[(CATALOG, JORIS)],
        hidden=[(CATALOG, "L-1003"), (CATALOG, "L-1002"), (CATALOG, "L-9999"), ([], JORIS), ([K[7], K[5]], "L-1010")],
        first=[_DOTS],
        mistakes=[
            Mistake("dataset", wrong=_H_DATASET,
                    en="`{call}` gives whole datasets, with their owners and access lists. Give back only their names. Which key of a dataset holds its name?",
                    nl="`{call}` geeft hele datasets, met hun eigenaren en toegangslijsten. Geef alleen hun namen terug. Onder welke key staat de naam van een dataset?"),
            Mistake("not", wrong=_H_NOT,
                    en="`{call}` gives {got}: the datasets that do not hold this card. Add a name when the card is among the dataset's cards, not when it is not.",
                    nl="`{call}` geeft {got}: de datasets die deze kaart niet bevatten. Voeg een naam toe als de kaart tussen de kaarten van de dataset staat, niet als die er niet tussen staat."),
            Mistake("first", wrong=_H_FIRST,
                    en="`{call}` gives {got}: only the first dataset that holds the card. A request to erase has to reach every copy: collect all the names in `names`, and give back the list after the loop.",
                    nl="`{call}` geeft {got}: alleen de eerste dataset met de kaart. Een verzoek om te wissen moet elke kopie bereiken: verzamel alle namen in `names`, en geef de lijst na de loop terug."),
            Mistake("keys", wrong=_H_KEYS,
                    en="`{call}` gives {got}: no dataset at all. `card in dataset` asks whether the card is a key of the dict. The card numbers are in a list inside the dataset: under which key?",
                    nl="`{call}` geeft {got}: geen enkele dataset. `card in dataset` vraagt of de kaart een key van de dict is. De kaartnummers staan in een lijst in de dataset: onder welke key?"),
            Mistake("inside", wrong=_H_INSIDE,
                    en="`{call}` gives {got}, and it should give {want}. Is your `return` inside the loop? Then the function stops after the first dataset. Put it after the loop, at the level of `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Staat je `return` in de loop? Dan stopt de functie na de eerste dataset. Zet hem na de loop, op de hoogte van `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: in the loop, add the name of a dataset to `names` when its cards include `card`.",
                      "nl": "Vervang de `...`: voeg in de loop de naam van een dataset toe aan `names` als `card` tussen zijn kaarten staat."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Give back the names of the datasets whose `\"cards\"` include `card`, in the order of the catalog.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Geef de namen terug van de datasets waarvan `\"cards\"` de kaart `card` bevat, in de volgorde van de catalog."},
            "almost": {"en": "Right for `holding(CATALOG, \"L-1004\")`. But `{call}` gives {got}, and it should give {want}. Does your function use the `catalog` and the `card` it gets?",
                       "nl": "Goed voor `holding(CATALOG, \"L-1004\")`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `catalog` en de `card` die hij krijgt?"},
            "correct": {"en": "Your `holding` is right in all {n} cases. The picture runs it for the card of Joris van Dam.",
                        "nl": "Je `holding` klopt in alle {n} gevallen. Het plaatje laat hem lopen voor de kaart van Joris van Dam."},
        },
    ),
    "no-owner": FunctionBox(
        name="no-owner",
        picture=owner_picture,
        function="no_owner",
        start=_N_START,
        solution=_N,
        visible=[(CATALOG, JORIS)],
        hidden=[(CATALOG, "L-1001"), (CATALOG, "L-1003"), (CATALOG, "L-9999"), ([K[0], K[1]], JORIS), ([K[6]], "L-1007"),
                ([K[5], K[4]], "L-1005")],
        first=[_DOTS],
        mistakes=[
            Mistake("holding", wrong=_N_HOLDING,
                    en="`{call}` gives {got}: every dataset that holds the card. Of those, keep only the ones without an owner.",
                    nl="`{call}` geeft {got}: elke dataset met de kaart. Bewaar daarvan alleen die zonder eigenaar."),
            Mistake("any", wrong=_N_ANY,
                    en="`{call}` gives {got}, and it should give {want}: you also give datasets without an owner that do not hold this card. Ask about the card first, and inside that, about the owner.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je geeft ook datasets zonder eigenaar die deze kaart niet bevatten. Vraag eerst naar de kaart, en daarbinnen naar de eigenaar."),
            Mistake("owned", wrong=_N_OWNED,
                    en="`{call}` gives {got}: the datasets that hold the card and do have an owner. Keep the ones whose owner is `None`.",
                    nl="`{call}` geeft {got}: de datasets met de kaart die wel een eigenaar hebben. Bewaar die waarvan de eigenaar `None` is."),
            Mistake("inside", wrong=_N_INSIDE,
                    en="`{call}` gives {got}, and it should give {want}. Is your `return` inside the loop? Then the function stops after the first dataset. Put it after the loop, at the level of `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Staat je `return` in de loop? Dan stopt de functie na de eerste dataset. Zet hem na de loop, op de hoogte van `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: in the loop, add the name of a dataset to `names` when it holds `card` and has no owner.",
                      "nl": "Vervang de `...`: voeg in de loop de naam van een dataset toe aan `names` als die `card` bevat en geen eigenaar heeft."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Give back the names of the datasets that hold `card` and whose `\"owner\"` is `None`, in the order of the catalog.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Geef de namen terug van de datasets die `card` bevatten en waarvan `\"owner\"` `None` is, in de volgorde van de catalog."},
            "almost": {"en": "Right for `no_owner(CATALOG, \"L-1004\")`. But `{call}` gives {got}, and it should give {want}. Does your function use the `catalog` and the `card` it gets?",
                       "nl": "Goed voor `no_owner(CATALOG, \"L-1004\")`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `catalog` en de `card` die hij krijgt?"},
            "correct": {"en": "Your `no_owner` is right in all {n} cases. The picture runs it for the card of Joris van Dam.",
                        "nl": "Je `no_owner` klopt in alle {n} gevallen. Het plaatje laat hem lopen voor de kaart van Joris van Dam."},
        },
    ),
}

CASES = [
    ("may-read", _R_START, "empty", None),
    ("may-read", _R, "correct", None),
    ("may-read", "def may_read(person, dataset):\n    for role in dataset[\"access\"]:\n        if role in person[\"roles\"]:\n            return True\n    return False\n", "correct", None),
    ("may-read", "def may_read(person, dataset):\n    found = False\n    for role in person[\"roles\"]:\n        if role in dataset[\"access\"]:\n            found = True\n    return found\n", "correct", None),
    ("may-read", _R_KEY, "error", "key"),
    ("may-read", _R_NEVER, "wrong", "never"),
    ("may-read", _R_FIRST, "wrong", "first"),
    ("may-read", _R_HEAD + "        return role in dataset[\"access\"]\n    return False\n", "wrong", "first"),
    ("may-read", _R_HEAD + "        if role in dataset[\"access\"]:\n            return False\n    return True\n", "wrong", None),
    ("may-read", "def may_read(person, dataset):\n    return True\n", "almost", None),
    ("holding", _H_START, "empty", None),
    ("holding", _H, "correct", None),
    ("holding", "def holding(catalog, card):\n    names = []\n    for d in catalog:\n        if card in d[\"cards\"]:\n            names = names + [d[\"name\"]]\n    return names\n", "correct", None),
    ("holding", _H_DATASET, "wrong", "dataset"),
    ("holding", _H_NOT, "wrong", "not"),
    ("holding", _H_FIRST, "wrong", "first"),
    ("holding", _H_KEYS, "wrong", "keys"),
    ("holding", _H_INSIDE, "wrong", "inside"),
    ("holding", _H_HEAD + "        names.append(dataset[\"name\"])\n    return names\n", "wrong", None),
    ("holding", "def holding(catalog, card):\n    return [\"members\", \"loans\", \"recommend-train\", \"recommender\", \"backup-2026-10\", \"survey-2024\"]\n", "almost", None),
    ("no-owner", _N_START, "empty", None),
    ("no-owner", _N, "correct", None),
    ("no-owner", _N_HEAD + "        if dataset[\"owner\"] is None:\n            if card in dataset[\"cards\"]:\n                names.append(dataset[\"name\"])\n    return names\n", "correct", None),
    ("no-owner", _N_HEAD + "        if card in dataset[\"cards\"]:\n            if dataset[\"owner\"] == None:\n                names.append(dataset[\"name\"])\n    return names\n", "correct", None),
    ("no-owner", _N_HOLDING, "wrong", "holding"),
    ("no-owner", _N_ANY, "wrong", "any"),
    ("no-owner", _N_OWNED, "wrong", "owned"),
    ("no-owner", _N_INSIDE, "wrong", "inside"),
    ("no-owner", "def no_owner(catalog, card):\n    return [\"backup-2026-10\", \"survey-2024\"]\n", "almost", None),
]
