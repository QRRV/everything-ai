# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-privacy-by-design: the member file of a made-up library, and a pipeline that keeps only
what a purpose needs and lets go of members who left. The three functions the learner writes: minimize (one member:
only the keys the purpose needs), to_delete (the card numbers of the members whose membership ended) and values_kept
(how many values the pipeline keeps for a purpose); with the pictures of what they give. All names, numbers and
addresses are made up."""
import copy
import html
import re

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, run_code, with_time_limit


class Named(list):
    """A list the check calls by the name the lesson gives it, such as MEMBERS, so that a message stays short. The
    learner's code gets its own copy, a plain list."""

    def __init__(self, values, name: str):
        super().__init__(values)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(list(self), memo)


class Member(dict):
    """One member of the file, which the check calls by its place, such as MEMBERS[0]. The learner's code gets its own
    copy, a plain dict."""

    def __init__(self, values: dict, name: str):
        super().__init__(values)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(dict(self), memo)


# the keys of the member file, in order
FIELDS = Named(["card", "name", "email", "phone", "birth_date", "postcode", "books", "newsletter", "ended"], "FIELDS")

# the made-up library's member file: ten members; "books" holds the kinds of books they borrowed, "newsletter" whether
# they asked for it, and "ended" the month their membership ended, or None while they are still a member. Lina Postma
# left the phone number out: the form does not need it.
_ROWS = [
    ("L-1001", "Anouk de Vries", "anouk.devries@example.org", "0612340001", "1958-04-02", "3511 KL", ["crime", "cooking"],
     False, None),
    ("L-1002", "Bram Hendriks", "bram.hendriks@example.org", "0612340002", "1963-11-20", "3512 AB", ["history"], True,
     None),
    ("L-1003", "Yara El Amrani", "yara.elamrani@example.org", "0612340003", "1971-06-09", "3512 CD", ["comics", "fantasy"],
     False, "2019-03"),
    ("L-1004", "Joris van Dam", "joris.vandam@example.org", "0612340004", "1979-01-30", "3513 EF", ["travel", "crime"],
     False, None),
    ("L-1005", "Lina Postma", "lina.postma@example.org", None, "1984-08-17", "3511 KL", ["poetry"], False, None),
    ("L-1006", "Sem Brouwer", "sem.brouwer@example.org", "0612340006", "1988-12-05", "3514 GH", ["fantasy", "science"],
     True, "2023-11"),
    ("L-1007", "Mila Koster", "mila.koster@example.org", "0612340007", "1992-03-14", "3512 AB",
     ["cooking", "travel", "history"], False, None),
    ("L-1008", "Tim de Graaf", "tim.degraaf@example.org", "0612340008", "1997-07-22", "3513 EF", ["children"], False,
     None),
    ("L-1009", "Nora Willems", "nora.willems@example.org", "0612340009", "2003-10-01", "3511 KL", ["science"], False,
     "2025-06"),
    ("L-1010", "Ravi Sitaram", "ravi.sitaram@example.org", "0612340010", "2009-05-27", "3514 GH", ["crime", "history"],
     True, None),
]


def _member(row, i: int) -> Member:
    values = {key: value for key, value in zip(FIELDS, row) if not (key == "phone" and value is None)}
    return Member(values, f"MEMBERS[{i}]")


MEMBERS = Named([_member(row, i) for i, row in enumerate(_ROWS)], "MEMBERS")

# what each purpose needs: lending (with a reminder by e-mail when a book is late) and the recommendation model
PURPOSES = {"lending": ["card", "name", "email"], "recommend": ["books"]}
LENDING = Named(PURPOSES["lending"], 'PURPOSES["lending"]')
RECOMMEND = Named(PURPOSES["recommend"], 'PURPOSES["recommend"]')

# --- the course's own functions -----------------------------------------------------------------------------------------

_M_HEAD = ("def minimize(record, needed):\n"
           "    # record is one member as a dict; needed is a list of the keys the purpose needs\n"
           "    kept = {}\n"
           "    for key in record:\n")
_M_START = _M_HEAD + "        ...\n    return kept\n"
_M = _M_HEAD + "        if key in needed:\n            kept[key] = record[key]\n    return kept\n"
minimize = run_code(_M).names["minimize"]

_D_HEAD = ("def to_delete(members):\n"
           "    # \"ended\" holds the month a membership ended, or None while the person is still a member\n"
           "    cards = []\n"
           "    for member in members:\n")
_D_START = _D_HEAD + "        ...\n    return cards\n"
_D = _D_HEAD + "        if member[\"ended\"] is not None:\n            cards.append(member[\"card\"])\n    return cards\n"
to_delete = run_code(_D).names["to_delete"]

_V_HEAD = ("def values_kept(members, needed):\n"
           "    # minimize is ready to use\n"
           "    count = 0\n"
           "    for member in members:\n")
_V_START = _V_HEAD + "        ...\n    return count\n"
_V = _V_HEAD + ("        if member[\"ended\"] is None:\n"
                "            count = count + len(minimize(member, needed))\n") + "    return count\n"
_KEPT_NAMES = {"minimize": minimize}
values_kept = run_code(_V, _KEPT_NAMES).names["values_kept"]


def file_values() -> int:
    """How many values the whole file holds: every key of every member."""
    return sum(len(member) for member in MEMBERS)


def decade(birth_date: str) -> str:
    """The decade of a birth date, as the worked example computes it: "1958-04-02" gives "1950s"."""
    return birth_date[:3] + "0s"


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


def _plain(member) -> dict:
    return copy.deepcopy(dict(member))


# --- the picture under exercise 1: minimize, for lending --------------------------------------------------------------

_MIN_WORDS = {
    "en": {"title": "Your minimize on the member file, for lending",
           "said": "Each column is a member, each row a key of the file. A filled square is a value your minimize keeps "
                   "for lending: {k} of the {n} values in the file. An empty square is a value it leaves out; a dash, "
                   "a phone number the member never gave.",
           "fine": "It keeps the same values as the course's minimize.",
           "wrong": "A dashed purple frame marks a square where it differs from the course's minimize.",
           "off": "Your minimize gives {got} for {who}; the picture draws a dict.",
           "stops": "Your minimize stops with an error on {who}: the check above says where."},
    "nl": {"title": "Jouw minimize op het ledenbestand, voor het uitlenen",
           "said": "Elke kolom is een lid, elke rij een key van het bestand. Een gevuld vakje is een waarde die jouw "
                   "minimize voor het uitlenen bewaart: {k} van de {n} waarden in het bestand. Een leeg vakje is een "
                   "waarde die hij weglaat; een streepje, een telefoonnummer dat het lid nooit gaf.",
           "fine": "Hij bewaart dezelfde waarden als de minimize van de course.",
           "wrong": "Een gestreept paars kader wijst een vakje aan waar hij verschilt van de minimize van de course.",
           "off": "Jouw minimize geeft {got} voor {who}; het plaatje tekent een dict.",
           "stops": "Jouw minimize stopt met een fout bij {who}: de check hierboven zegt waar."},
}


def _square(kind: str, wrong: bool) -> str:
    if kind == "none":
        inner = '<span style="color:var(--ea-ink-3)">–</span>'
        frame = "border:1.5px dashed var(--ea-loss);border-radius:2px;" if wrong else ""
        return (f'<div style="width:1rem;height:1rem;display:flex;align-items:center;justify-content:center;{frame}">'
                f'{inner}</div>')
    fill = "background:var(--ea-ink)" if kind == "kept" else "background:transparent"
    edge = "border:1.5px dashed var(--ea-loss)" if wrong else "border:1px solid var(--ea-line)"
    return f'<div style="width:0.9rem;height:0.9rem;border-radius:2px;{fill};{edge}"></div>'


def matrix(fn, needed) -> tuple:
    """What `fn` (a minimize) keeps of each member for `needed`: (rows, kept, wrong), rows being one list per key of
    the file with ("kept" | "left" | "none", differs from the course's) per member. When `fn` stops or gives no dict:
    ("stops" or "off", the member's name, what it gave)."""
    rows = {key: [] for key in FIELDS}
    kept = wrong = 0
    for member in MEMBERS:
        got = _call(fn, _plain(member), list(needed))
        if got is _STOPPED:
            return ("stops", repr(member), None)
        if not isinstance(got, dict):
            return ("off", repr(member), got)
        want = minimize(_plain(member), list(needed))
        for key in FIELDS:
            if key not in member:
                rows[key].append(("none", key in got))
                wrong += key in got
                continue
            has = key in got and got[key] == member[key]
            differs = has != (key in want) or (key in got and not has)
            rows[key].append(("kept" if has else "left", differs))
            kept += has
            wrong += differs
    return rows, kept, wrong


def _matrix_html(rows: dict) -> str:
    head = "".join(f'<th style="{_LABEL};font-weight:400;padding:0 0.2rem;text-align:center">{i + 1}</th>'
                   for i in range(len(MEMBERS)))
    body = []
    for key in FIELDS:
        cells = "".join(f'<td style="padding:0.15rem 0.2rem"><div style="display:flex;justify-content:center">'
                        f'{_square(kind, wrong)}</div></td>' for kind, wrong in rows[key])
        body.append(f'<tr><td style="{_LABEL};padding:0.1rem 0.6rem 0.1rem 0;white-space:nowrap">{html.escape(key)}</td>'
                    f'{cells}</tr>')
    return (f'<table style="border-collapse:collapse;margin-top:0.4rem"><thead><tr><th></th>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


def minimize_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _MIN_WORDS[seen.lang]
    got = matrix(fn, LENDING)
    if isinstance(got[0], str):
        what, who, value = got
        note = w["stops"].format(who=who) if what == "stops" else w["off"].format(got=_shown(value), who=who)
        return yours.figure(w["title"], _note([note]))
    rows, kept, wrong = got
    notes = [w["said"].format(k=kept, n=file_values()), w["wrong"] if wrong else w["fine"]]
    return yours.figure(w["title"], _matrix_html(rows) + _note(notes))


# --- the picture under exercise 2: to_delete --------------------------------------------------------------------------

_DEL_WORDS = {
    "en": {"title": "Your to_delete on the member file",
           "head": ["card", "name", "ended"],
           "member": "member",
           "said": "Struck through in gray: the {k} members whose card numbers your to_delete gives back; their data "
                   "goes. {r} members stay.",
           "fine": "The same members as the course's to_delete.",
           "wrong": "A dashed purple frame marks a member where it differs from the course's to_delete.",
           "off": "Your to_delete gives {got}; the picture draws a list of card numbers.",
           "stops": "Your to_delete stops with an error on MEMBERS: the check above says where."},
    "nl": {"title": "Jouw to_delete op het ledenbestand",
           "head": ["card", "name", "ended"],
           "member": "lid",
           "said": "Grijs doorgestreept: de {k} leden van wie jouw to_delete het kaartnummer teruggeeft; hun gegevens "
                   "gaan weg. {r} leden blijven.",
           "fine": "Dezelfde leden als bij de to_delete van de course.",
           "wrong": "Een gestreept paars kader wijst een lid aan waar hij verschilt van de to_delete van de course.",
           "off": "Jouw to_delete geeft {got}; het plaatje tekent een lijst kaartnummers.",
           "stops": "Jouw to_delete stopt met een fout bij MEMBERS: de check hierboven zegt waar."},
}

_ROW = ("display:flex;gap:0.6rem;align-items:baseline;padding:0.2rem 0.5rem;margin:0.15rem 0;border-radius:4px;"
        "font-family:var(--ea-font-ui);font-size:0.85rem;")


def _member_row(cells: list, gone: bool, wrong: bool) -> str:
    style = _ROW + ("color:var(--ea-ink-3);text-decoration:line-through;" if gone else "color:var(--ea-ink);")
    style += "border:1.5px dashed var(--ea-loss)" if wrong else "border:1px solid var(--ea-line)"
    parts = "".join(f'<div style="flex:{f};min-width:0;overflow-wrap:anywhere">{c}</div>' for c, f in zip(cells, (2, 4, 2)))
    return f'<div style="{style}">{parts}</div>'


def delete_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _DEL_WORDS[seen.lang]
    got = _call(fn, [_plain(m) for m in MEMBERS])
    if got is _STOPPED:
        return yours.figure(w["title"], _note([w["stops"]]))
    if not isinstance(got, list):
        return yours.figure(w["title"], _note([w["off"].format(got=_shown(got))]))
    want = to_delete([_plain(m) for m in MEMBERS])
    head = "".join(f'<div style="flex:{f}">{html.escape(c)}</div>' for c, f in zip(w["head"], (2, 4, 2)))
    rows = [f'<div style="display:flex;gap:0.6rem;padding:0 0.5rem;{_LABEL};border-bottom:2px solid var(--ea-ink-3);'
            f'margin-bottom:0.2rem">{head}</div>']
    gone_count = wrong_any = 0
    for member in MEMBERS:
        gone = member["card"] in got
        wrong = gone != (member["card"] in want)
        gone_count += gone
        wrong_any += wrong
        ended = "None" if member["ended"] is None else f'"{member["ended"]}"'
        rows.append(_member_row([_code(member["card"]), html.escape(member["name"]), _code(ended)], gone, wrong))
    notes = [w["said"].format(k=gone_count, r=len(MEMBERS) - gone_count), w["wrong"] if wrong_any else w["fine"]]
    return yours.figure(w["title"], f'<div style="max-width:28rem">{"".join(rows)}</div>' + _note(notes))


# --- the picture under exercise 3: values_kept ------------------------------------------------------------------------

_KEPT_WORDS = {
    "en": {"title": "Your values_kept: what would show if the pipeline's data leaked",
           "file": "the whole file",
           "said": "The first bar is the whole file as it is: every value. The others are what your values_kept "
                   "gives for the calls beside them.",
           "off": "Your values_kept gives {got} for {which}; the bars draw a whole number from 0 to {n}.",
           "stops": "Your values_kept stops with an error on {which}: the check above says where."},
    "nl": {"title": "Jouw values_kept: wat zichtbaar wordt als de gegevens van de pipeline lekken",
           "file": "het hele bestand",
           "said": "De eerste balk is het hele bestand zoals het is: elke waarde. De andere zijn wat jouw "
                   "values_kept geeft voor de aanroepen ernaast.",
           "off": "Jouw values_kept geeft {got} voor {which}; de balken tekenen een geheel getal van 0 tot {n}.",
           "stops": "Jouw values_kept stopt met een fout bij {which}: de check hierboven zegt waar."},
}


def _bar(label: str, value: int, total: int, style: str) -> str:
    return (f'<div style="display:flex;align-items:center;gap:0.6rem;margin:0.35rem 0;font-family:var(--ea-font-ui);'
            f'font-size:0.85rem;color:var(--ea-ink)"><div style="flex:0 0 13rem;overflow-wrap:anywhere">{label}</div>'
            f'<div style="flex:1 1 auto;max-width:16rem;height:0.9rem;border:1px solid var(--ea-line);border-radius:3px">'
            f'<div style="width:{100 * value / total:.1f}%;height:100%;{style}"></div></div>'
            f'<div style="flex:0 0 2.5rem">{_code(str(value))}</div></div>')


def kept_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w, total = _KEPT_WORDS[seen.lang], file_values()
    bars = [_bar(html.escape(w["file"]), total, total, "background:var(--ea-ink-3)")]
    for needed, which in ((FIELDS, "FIELDS"), (LENDING, 'PURPOSES["lending"]'), (RECOMMEND, 'PURPOSES["recommend"]')):
        got = _call(fn, [_plain(m) for m in MEMBERS], list(needed))
        call = f"values_kept(MEMBERS, {which})"
        if got is _STOPPED:
            return yours.figure(w["title"], _note([w["stops"].format(which=call)]))
        if isinstance(got, bool) or not isinstance(got, int) or not 0 <= got <= total:
            return yours.figure(w["title"], _note([w["off"].format(got=_shown(got), which=call, n=total)]))
        bars.append(_bar(_code(call), got, total, "background:var(--ea-ink-2)"))
    return yours.figure(w["title"], "".join(bars) + _note([w["said"]]))


# --- the demo: a purpose of the learner's own --------------------------------------------------------------------------

_TRY_WORDS = {
    "en": {"title": "What {who} keeps of the members who stay",
           "yours": "your minimize", "course": "the course's minimize",
           "said": "{k} of the {n} values in the file; the {d} members whose membership ended are gone.",
           "nothing": "Nothing: with no keys chosen, the pipeline keeps no personal data at all.",
           "off": "{who} gives {got}; the picture draws a dict.",
           "stops": "{who} stops with an error."},
    "nl": {"title": "Wat {who} bewaart van de leden die blijven",
           "yours": "jouw minimize", "course": "de minimize van de course",
           "said": "{k} van de {n} waarden in het bestand; de {d} leden van wie het lidmaatschap stopte, zijn weg.",
           "nothing": "Niets: zonder gekozen keys bewaart de pipeline helemaal geen persoonsgegevens.",
           "off": "{who} geeft {got}; het plaatje tekent een dict.",
           "stops": "{who} stopt met een fout."},
}


def mine(box_text, correct: bool):
    """The learner's own minimize when exercise 1 is right, else None: the demo then uses the course's."""
    if not correct or not isinstance(box_text, str):
        return None
    run = run_code(box_text)
    fn = run.names.get("minimize") if run.error is None else None
    return fn if callable(fn) else None


def _value_text(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def try_picture(needed, fn, lang: str) -> str:
    """What `fn` (the learner's minimize, or None for the course's) keeps of the members who stay, for the keys the
    learner chose."""
    w = _TRY_WORDS[lang]
    who = w["yours"] if fn is not None else w["course"]
    fn = fn or minimize
    needed = [key for key in FIELDS if key in list(needed or [])]
    gone = set(to_delete([_plain(m) for m in MEMBERS]))
    lines, k = [], 0
    for member in MEMBERS:
        if member["card"] in gone:
            continue
        got = _call(fn, _plain(member), list(needed))
        if got is _STOPPED:
            return yours.figure(w["title"].format(who=who), _note([w["stops"].format(who=who.capitalize())]))
        if not isinstance(got, dict):
            return yours.figure(w["title"].format(who=who),
                                _note([w["off"].format(who=who.capitalize(), got=_shown(got))]))
        k += len(got)
        if got:
            parts = " · ".join(f'<span style="{_LABEL}">{html.escape(str(key))}</span> {html.escape(_value_text(v))}'
                               for key, v in got.items())
            lines.append(f'<div style="border-left:2px solid var(--ea-line);padding:0.1rem 0 0.1rem 0.6rem;'
                         f'margin:0.3rem 0;line-height:1.6">{parts}</div>')
    body = (f'<div style="font-family:var(--ea-font-ui);font-size:0.85rem;color:var(--ea-ink);overflow-wrap:anywhere">'
            f'{"".join(lines)}</div>')
    notes = [w["said"].format(k=k, n=file_values(), d=len(gone))] if k else [w["nothing"]]
    return yours.figure(w["title"].format(who=who), body + _note(notes))


# --- the exercises ----------------------------------------------------------------------------------------------------

REQUIRED = ["minimize", "to-delete", "values-kept", "minimum", "default"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# exercise 1: minimize
_M_NOT = _M_HEAD + "        if key not in needed:\n            kept[key] = record[key]\n    return kept\n"
_M_INSIDE = _M_HEAD + "        if key in needed:\n            kept[key] = record[key]\n        return kept\n"
_M_KEY = _M_HEAD + "        if key in needed:\n            kept[key] = key\n    return kept\n"
_M_NEEDED = ("def minimize(record, needed):\n    kept = {}\n    for key in needed:\n        kept[key] = record[key]\n"
             "    return kept\n")
_M_APPEND = _M_HEAD + "        if key in needed:\n            kept.append(record[key])\n    return kept\n"

# exercise 2: to_delete
_D_STAY = _D_HEAD + "        if member[\"ended\"] is None:\n            cards.append(member[\"card\"])\n    return cards\n"
_D_MEMBER = _D_HEAD + "        if member[\"ended\"] is not None:\n            cards.append(member)\n    return cards\n"
_D_INSIDE = _D_HEAD + ("        if member[\"ended\"] is not None:\n            cards.append(member[\"card\"])\n"
                       "        return cards\n")
_D_ENDED = _D_HEAD + "        if member[\"ended\"] is not None:\n            cards.append(member[\"ended\"])\n    return cards\n"

# exercise 3: values_kept
_V_ALL = _V_HEAD + "        count = count + len(minimize(member, needed))\n    return count\n"
_V_MEMBERS = _V_HEAD + "        if member[\"ended\"] is None:\n            count = count + 1\n    return count\n"
_V_NEEDED = _V_HEAD + "        if member[\"ended\"] is None:\n            count = count + len(needed)\n    return count\n"
_V_GONE = _V_HEAD + ("        if member[\"ended\"] is not None:\n"
                     "            count = count + len(minimize(member, needed))\n") + "    return count\n"
_V_INSIDE = _V_HEAD + ("        if member[\"ended\"] is None:\n"
                       "            count = count + len(minimize(member, needed))\n") + "        return count\n"


def _key_error(e: BaseException) -> bool:
    return isinstance(e, KeyError)


def _append_error(e: BaseException) -> bool:
    return isinstance(e, AttributeError) and "append" in str(e)


M = MEMBERS

EXERCISES = {
    "minimize": FunctionBox(
        name="minimize",
        picture=minimize_picture,
        function="minimize",
        start=_M_START,
        solution=_M,
        visible=[(M[0], LENDING)],
        hidden=[(M[1], ["email"]), (M[4], ["name", "phone"]), (M[2], []), (M[0], FIELDS), (M[6], ["books", "card"]),
                (M[9], ["newsletter", "postcode"])],
        first=[_DOTS],
        mistakes=[
            Mistake("needed", error=_key_error,
                    en="With `{call}`, your function stops with an error: it looks up a key that this member does not have. Go through the keys of `record`, as the start does, and keep a key when it is `in needed`.",
                    nl="Met `{call}` stopt je functie met een fout: hij zoekt een key op die dit lid niet heeft. Loop door de keys van `record`, zoals de start doet, en bewaar een key als die `in needed` staat."),
            Mistake("append", error=_append_error,
                    en="With `{call}`, your function stops with an error: `kept` is a dict, and a dict has no `append`. Put a value in it under its key, with `=`, as in [Dictionaries](/course/open/py-dicts?lang=en).",
                    nl="Met `{call}` stopt je functie met een fout: `kept` is een dict, en een dict heeft geen `append`. Zet er een waarde in onder zijn key, met `=`, zoals in [Dictionaries](/course/open/py-dicts?lang=nl)."),
            Mistake("not", wrong=_M_NOT,
                    en="`{call}` gives {got}: everything the purpose does not need. Keep a key when it is `in needed`, not when it is not.",
                    nl="`{call}` geeft {got}: alles wat het doel niet nodig heeft. Bewaar een key als die `in needed` staat, niet als die er niet in staat."),
            Mistake("key", wrong=_M_KEY,
                    en="`{call}` gives {got}: under each key, the key itself. Put the member's own value under it.",
                    nl="`{call}` geeft {got}: onder elke key de key zelf. Zet er de eigen waarde van het lid onder."),
            Mistake("inside", wrong=_M_INSIDE,
                    en="`{call}` gives {got}, and it should give {want}. Is your `return` inside the loop? Then the function stops after the first key. Put it after the loop, at the level of `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Staat je `return` in de loop? Dan stopt de functie na de eerste key. Zet hem na de loop, op de hoogte van `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: in the loop, put a key and its value in `kept` when the purpose needs that key.",
                      "nl": "Vervang de `...`: zet in de loop een key met zijn waarde in `kept` als het doel die key nodig heeft."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Give back a dict with only the keys of `record` that are in `needed`, each with its value.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Geef een dict terug met alleen de keys van `record` die in `needed` staan, elk met zijn waarde."},
            "almost": {"en": "Right for `minimize(MEMBERS[0], PURPOSES[\"lending\"])`. But `{call}` gives {got}, and it should give {want}. Does your function use the `record` and the `needed` it gets?",
                       "nl": "Goed voor `minimize(MEMBERS[0], PURPOSES[\"lending\"])`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie het `record` en de `needed` die hij krijgt?"},
            "correct": {"en": "Your `minimize` is right in all {n} cases. The picture runs it on every member, for lending.",
                        "nl": "Je `minimize` klopt in alle {n} gevallen. Het plaatje laat hem lopen op elk lid, voor het uitlenen."},
        },
    ),
    "to-delete": FunctionBox(
        name="to-delete",
        picture=delete_picture,
        function="to_delete",
        start=_D_START,
        solution=_D,
        visible=[(MEMBERS,)],
        hidden=[([M[0], M[1]],), ([M[2]],), ([],), ([M[8], M[3], M[5]],)],
        first=[_DOTS],
        mistakes=[
            Mistake("stay", wrong=_D_STAY,
                    en="`{call}` gives {got}: the members who stay. A membership ended when `\"ended\"` holds a month, so when it is not `None`.",
                    nl="`{call}` geeft {got}: de leden die blijven. Een lidmaatschap is gestopt als `\"ended\"` een maand bevat, dus als het niet `None` is."),
            Mistake("member", wrong=_D_MEMBER,
                    en="`{call}` gives whole members, with their names and e-mail addresses. Give back only their card numbers. Which key of a member holds the card number?",
                    nl="`{call}` geeft hele leden, met hun namen en e-mailadressen. Geef alleen hun kaartnummers terug. Onder welke key staat het kaartnummer van een lid?"),
            Mistake("ended", wrong=_D_ENDED,
                    en="`{call}` gives {got}: the months the memberships ended. Give back the card numbers. Which key of a member holds the card number?",
                    nl="`{call}` geeft {got}: de maanden waarin de lidmaatschappen stopten. Geef de kaartnummers terug. Onder welke key staat het kaartnummer van een lid?"),
            Mistake("inside", wrong=_D_INSIDE,
                    en="`{call}` gives {got}, and it should give {want}. Is your `return` inside the loop? Then the function stops after the first member. Put it after the loop, at the level of `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Staat je `return` in de loop? Dan stopt de functie na het eerste lid. Zet hem na de loop, op de hoogte van `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: in the loop, add the card number of a member to `cards` when the membership ended.",
                      "nl": "Vervang de `...`: voeg in de loop het kaartnummer van een lid toe aan `cards` als het lidmaatschap gestopt is."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Give back the card numbers of the members whose `\"ended\"` is not `None`, in the order of the list.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Geef de kaartnummers terug van de leden bij wie `\"ended\"` niet `None` is, in de volgorde van de lijst."},
            "almost": {"en": "Right for `to_delete(MEMBERS)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `members` it gets?",
                       "nl": "Goed voor `to_delete(MEMBERS)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `members` die hij krijgt?"},
            "correct": {"en": "Your `to_delete` is right in all {n} cases. The picture shows which members it lets go.",
                        "nl": "Je `to_delete` klopt in alle {n} gevallen. Het plaatje laat zien welke leden hij laat gaan."},
        },
    ),
    "values-kept": FunctionBox(
        name="values-kept",
        picture=kept_picture,
        function="values_kept",
        start=_V_START,
        solution=_V,
        names=_KEPT_NAMES,
        visible=[(MEMBERS, RECOMMEND)],
        hidden=[(MEMBERS, LENDING), (MEMBERS, FIELDS), ([], ["books"]), ([M[4]], ["name", "phone"]), (MEMBERS, []),
                ([M[2], M[5]], ["name"])],
        first=[_DOTS],
        mistakes=[
            Mistake("all", wrong=_V_ALL,
                    en="`{call}` gives {got}, and it should give {want}. Your count includes the members whose membership ended, but their data is deleted. Count only the members whose `\"ended\"` is `None`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je telt ook de leden van wie het lidmaatschap gestopt is, maar hun gegevens zijn gewist. Tel alleen de leden bij wie `\"ended\"` `None` is."),
            Mistake("members", wrong=_V_MEMBERS,
                    en="`{call}` gives {got}: the number of members who stay. Count their values instead: for each member, the number of keys that `minimize` keeps.",
                    nl="`{call}` geeft {got}: het aantal leden dat blijft. Tel juist hun waarden: per lid het aantal keys dat `minimize` bewaart."),
            Mistake("needed", wrong=_V_NEEDED,
                    en="`{call}` gives {got}, and it should give {want}. You count every key in `needed`, also for a member who never gave that value, such as a phone number. Count what `minimize` keeps.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je telt elke key in `needed`, ook bij een lid dat die waarde nooit gaf, zoals een telefoonnummer. Tel wat `minimize` bewaart."),
            Mistake("gone", wrong=_V_GONE,
                    en="`{call}` gives {got}: the values of the members who left. Count the members who stay: those whose `\"ended\"` is `None`.",
                    nl="`{call}` geeft {got}: de waarden van de leden die weg zijn. Tel de leden die blijven: die bij wie `\"ended\"` `None` is."),
            Mistake("inside", wrong=_V_INSIDE,
                    en="`{call}` gives {got}, and it should give {want}. Is your `return` inside the loop? Then the function stops after the first member. Put it after the loop, at the level of `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Staat je `return` in de loop? Dan stopt de functie na het eerste lid. Zet hem na de loop, op de hoogte van `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: for each member who stays, add the number of values that `minimize` keeps to `count`.",
                      "nl": "Vervang de `...`: tel voor elk lid dat blijft het aantal waarden dat `minimize` bewaart op bij `count`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Skip the members whose membership ended; for the others, add up how many keys `minimize` keeps.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Sla de leden over van wie het lidmaatschap gestopt is; tel voor de anderen op hoeveel keys `minimize` bewaart."},
            "almost": {"en": "Right for `values_kept(MEMBERS, PURPOSES[\"recommend\"])`. But `{call}` gives {got}, and it should give {want}. Does your function use the `members` and the `needed` it gets?",
                       "nl": "Goed voor `values_kept(MEMBERS, PURPOSES[\"recommend\"])`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `members` en de `needed` die hij krijgt?"},
            "correct": {"en": "Your `values_kept` is right in all {n} cases. The picture compares it with the whole file.",
                        "nl": "Je `values_kept` klopt in alle {n} gevallen. Het plaatje zet hem naast het hele bestand."},
        },
    ),
}

CASES = [
    ("minimize", _M_START, "empty", None),
    ("minimize", _M, "correct", None),
    ("minimize", "def minimize(record, needed):\n    kept = {}\n    for key in needed:\n        if key in record:\n            kept[key] = record[key]\n    return kept\n", "correct", None),
    ("minimize", "def minimize(record, needed):\n    kept = {}\n    for key, value in record.items():\n        if key in needed:\n            kept[key] = value\n    return kept\n", "correct", None),
    ("minimize", _M_NEEDED, "error", "needed"),
    ("minimize", _M_APPEND, "error", "append"),
    ("minimize", _M_NOT, "wrong", "not"),
    ("minimize", _M_KEY, "wrong", "key"),
    ("minimize", _M_INSIDE, "wrong", "inside"),
    ("minimize", _M_HEAD + "        kept[key] = record[key]\n    return kept\n", "wrong", None),
    ("minimize", "def minimize(record, needed):\n    return {\"card\": \"L-1001\", \"name\": \"Anouk de Vries\", \"email\": \"anouk.devries@example.org\"}\n", "almost", None),
    ("to-delete", _D_START, "empty", None),
    ("to-delete", _D, "correct", None),
    ("to-delete", _D_HEAD + "        if member[\"ended\"] != None:\n            cards.append(member[\"card\"])\n    return cards\n", "correct", None),
    ("to-delete", _D_HEAD + "        if member[\"ended\"]:\n            cards.append(member[\"card\"])\n    return cards\n", "correct", None),
    ("to-delete", _D_STAY, "wrong", "stay"),
    ("to-delete", _D_MEMBER, "wrong", "member"),
    ("to-delete", _D_ENDED, "wrong", "ended"),
    ("to-delete", _D_INSIDE, "wrong", "inside"),
    ("to-delete", "def to_delete(members):\n    return [\"L-1003\", \"L-1006\", \"L-1009\"]\n", "almost", None),
    ("values-kept", _V_START, "empty", None),
    ("values-kept", _V, "correct", None),
    ("values-kept", _V_HEAD + "        if member[\"ended\"] is None:\n            count += len(minimize(member, needed))\n    return count\n", "correct", None),
    ("values-kept", _V_ALL, "wrong", "all"),
    ("values-kept", _V_MEMBERS, "wrong", "members"),
    ("values-kept", _V_NEEDED, "wrong", "needed"),
    ("values-kept", _V_GONE, "wrong", "gone"),
    ("values-kept", _V_INSIDE, "wrong", "inside"),
    ("values-kept", "def values_kept(members, needed):\n    return 7\n", "almost", None),
]
