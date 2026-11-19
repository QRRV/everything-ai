# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-pseudonymization: the made-up language-learning app with its twelve users (USERS), the same
rows after pseudonymization (SHARED, with CODES), the sign-up list of a language café (CAFE) and three facts about one
user (KNOWN). code_of gives the first 8 characters of the SHA-256 hash of a text. The three functions the learner
writes: pseudonymize (a copy of a row with the e-mail address replaced by its code), unmask (which codes belong to
addresses on a list) and candidates (the codes of the rows that match every known fact); with the pictures of what the
learner's functions give, and the figure of rows before and after pseudonymization."""
import hashlib
import html
import re

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake


class NamedList(list):
    """A list the check calls by its name, such as CODES, so that a message stays short."""

    def __init__(self, items, name: str):
        super().__init__(items)
        self.name = name

    def __repr__(self):
        return self.name


class NamedDict(dict):
    """A dict the check calls by its name, such as USERS[0], so that a message stays short."""

    def __init__(self, items, name: str):
        super().__init__(items)
        self.name = name

    def __repr__(self):
        return self.name


def code_of(text):
    """The first 8 characters of the SHA-256 hash of the text: the same text always gives the same code."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


# --- the made-up language-learning app: twelve users, each with an e-mail address and what the app knows ------------
_USERS = [
    {"email": "ana@mail.example", "language": "Spanish", "joined": 2022, "hour": 21, "streak": 15,
     "goal": "travel through Peru"},
    {"email": "bo@mail.example", "language": "Japanese", "joined": 2023, "hour": 7, "streak": 41,
     "goal": "a new job in Osaka"},
    {"email": "chen@mail.example", "language": "Japanese", "joined": 2023, "hour": 22, "streak": 3,
     "goal": "films without subtitles"},
    {"email": "dara@mail.example", "language": "Japanese", "joined": 2021, "hour": 7, "streak": 120,
     "goal": "talk with a partner's family"},
    {"email": "eli@mail.example", "language": "Spanish", "joined": 2023, "hour": 7, "streak": 9,
     "goal": "move to Valencia"},
    {"email": "fenna@mail.example", "language": "Italian", "joined": 2022, "hour": 12, "streak": 60,
     "goal": "read Italian novels"},
    {"email": "gus@mail.example", "language": "French", "joined": 2023, "hour": 21, "streak": 2,
     "goal": "pass a school exam"},
    {"email": "hana@mail.example", "language": "Spanish", "joined": 2021, "hour": 18, "streak": 300,
     "goal": "teach Spanish one day"},
    {"email": "ivo@mail.example", "language": "French", "joined": 2022, "hour": 7, "streak": 25,
     "goal": "a year of study in Lyon"},
    {"email": "jas@mail.example", "language": "Italian", "joined": 2023, "hour": 22, "streak": 5,
     "goal": "talk with grandparents"},
    {"email": "kim@mail.example", "language": "Spanish", "joined": 2023, "hour": 12, "streak": 33,
     "goal": "follow football in Spanish"},
    {"email": "lou@mail.example", "language": "French", "joined": 2021, "hour": 18, "streak": 88,
     "goal": "a job in Brussels"},
]
USERS = NamedList([NamedDict(u, f"USERS[{i}]") for i, u in enumerate(_USERS)], "USERS")
KEYS = ["email", "language", "joined", "hour", "streak", "goal"]


def pseudonymize(row):
    new = {}
    for key in row:
        if key == "email":
            new["code"] = code_of(row[key])
        else:
            new[key] = row[key]
    return new


# --- what the research team gets: the same rows, each address replaced by its code -----------------------------------
SHARED = NamedList([pseudonymize(u) for u in _USERS], "SHARED")
CODES = NamedList([row["code"] for row in SHARED], "CODES")

# --- the sign-up list of a language café: ten addresses, four of them of users of the app ----------------------------
CAFE = NamedList(["ana@mail.example", "mo@mail.example", "chen@mail.example", "pim@mail.example", "rosa@mail.example",
                  "ivo@mail.example", "sky@mail.example", "teun@mail.example", "lou@mail.example", "yara@mail.example"],
                 "CAFE")

# --- three facts a colleague knows about Bo, a user of the app --------------------------------------------------------
KNOWN = NamedDict({"language": "Japanese", "joined": 2023, "hour": 7}, "KNOWN")


def unmask(codes, emails):
    found = {}
    for email in emails:
        if code_of(email) in codes:
            found[code_of(email)] = email
    return found


def candidates(known, rows):
    found = []
    for row in rows:
        fits = True
        for key in known:
            if row[key] != known[key]:
                fits = False
        if fits:
            found.append(row["code"])
    return found


# --- the pictures ---------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_BOX = "padding:0.25rem 0.5rem;margin:0.2rem 0;border-radius:4px;"
_PLAIN = "border:1px solid var(--ea-line);color:var(--ea-ink)"
_WRONG = "border:1.5px dashed var(--ea-loss);color:var(--ea-loss-text)"
_HEAVY = "border:2px solid var(--ea-ink);color:var(--ea-ink);font-weight:600"


def _code(text) -> str:
    return f'<code style="font-family:var(--ea-font-code);font-size:0.8rem">{html.escape(str(text))}</code>'


def _row(cells: list, style: str, widths: list) -> str:
    parts = "".join(f'<div style="flex:{w};min-width:0;overflow-wrap:anywhere">{c}</div>' for c, w in zip(cells, widths))
    return f'<div style="display:flex;gap:0.6rem;align-items:baseline;{_BOX}{style}">{parts}</div>'


def _head(cells: list, widths: list) -> str:
    parts = "".join(f'<div style="flex:{w};min-width:0">{html.escape(c)}</div>' for c, w in zip(cells, widths))
    return (f'<div style="display:flex;gap:0.6rem;font-weight:600;color:var(--ea-ink);padding:0 0.5rem;'
            f'border-bottom:2px solid var(--ea-ink-3);margin-bottom:0.2rem">{parts}</div>')


def _table(head: list, rows: list, widths: list) -> str:
    return (f'<div style="font-family:var(--ea-font-ui);font-size:0.85rem;max-width:46rem">'
            f'{_head(head, widths)}{"".join(rows)}</div>')


def _fn(seen):
    mine = seen.mine
    if mine is None:
        return None
    fn = mine.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _call(fn, *args):
    import copy
    from everything_ai import check as checking
    return checking.with_time_limit(lambda: fn(*(copy.deepcopy(a) for a in args)))


_SHARED_WORDS = {
    "en": {"title": "Before and after pseudonymization",
           "app": "in the app", "team": "for the research team",
           "key": "Left: the first three users of the made-up app, each with an e-mail address. Right: the same rows "
                  "after pseudonymization. The address is replaced by its code, the first 8 characters of its SHA-256 "
                  "hash; the language, the year they joined, the hour they usually practice, the streak and the goal "
                  "stay as they were."},
    "nl": {"title": "Voor en na pseudonimisering",
           "app": "in de app", "team": "voor het onderzoeksteam",
           "key": "Links: de eerste drie gebruikers van de verzonnen app, elk met een e-mailadres. Rechts: dezelfde "
                  "rijen na pseudonimisering. Het adres is vervangen door zijn code, de eerste 8 tekens van de "
                  "SHA-256-hash; de taal, het jaar van aanmelden, het uur waarop iemand meestal oefent, de reeks en "
                  "het doel blijven zoals ze waren."},
}


def _card(row: dict, mark: str) -> str:
    lines = "".join(f'<div>{html.escape(k)}: {_marked(row[k]) if k == mark else _code(row[k])}</div>' for k in row)
    return f'<div style="{_BOX}{_PLAIN};padding:0.35rem 0.6rem">{lines}</div>'


def _marked(value) -> str:
    return f'<span style="display:inline-block;padding:0 0.25rem;border-radius:3px;{_HEAVY}">{_code(value)}</span>'


def shared_picture(lang: str) -> str:
    """The figure of The idea: three rows of USERS beside the same rows of SHARED, the address and its code marked."""
    w = _SHARED_WORDS[lang]
    left = "".join(_card(USERS[i], "email") for i in range(3))
    right = "".join(_card(SHARED[i], "code") for i in range(3))
    cols = (f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;font-family:var(--ea-font-ui);'
            f'font-size:0.83rem;max-width:46rem;color:var(--ea-ink)">'
            f'<div><div style="font-weight:600">{html.escape(w["app"])}</div>{left}</div>'
            f'<div><div style="font-weight:600">{html.escape(w["team"])} →</div>{right}</div></div>')
    return yours.figure(w["title"], cols + f'<div style="{_UI}">{html.escape(w["key"])}</div>')


_PSEUDO_WORDS = {
    "en": {"title": "Your pseudonymize on the twelve users",
           "key": "What your pseudonymize gives for each row of USERS: what the research team would get.",
           "wrong": "A dashed frame marks where it differs from the course's pseudonymize: {k} of the 12 rows.",
           "bad": "no dictionary"},
    "nl": {"title": "Jouw pseudonymize op de twaalf gebruikers",
           "key": "Wat jouw pseudonymize voor elke rij van USERS geeft: wat het onderzoeksteam zou krijgen.",
           "wrong": "Een gestippeld kader wijst aan waar hij verschilt van de pseudonymize van de course: bij {k} van "
                    "de 12 rijen.",
           "bad": "geen dictionary"},
}


def pseudonymize_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _PSEUDO_WORDS[seen.lang]
    rows, k = [], 0
    for user in USERS:
        try:
            got = _call(fn, user)
        except Exception:
            got = None
        wrong = got != pseudonymize(user)
        k += wrong
        if isinstance(got, dict):
            text = ", ".join(f'{html.escape(str(key))}: {_code(value)}' for key, value in got.items())
        else:
            text = html.escape(w["bad"])
        rows.append(_row([text], _WRONG if wrong else _PLAIN, [1]))
    note = w["key"] + (" " + w["wrong"].format(k=k) if k else "")
    body = f'<div style="font-family:var(--ea-font-ui);font-size:0.83rem;max-width:46rem">{"".join(rows)}</div>'
    return yours.figure(w["title"], body + f'<div style="{_UI}">{html.escape(note)}</div>')


_UNMASK_WORDS = {
    "en": {"title": "Your unmask on the café's list",
           "head": ["address on the list", "its code", "in SHARED", "your unmask"],
           "yes": "yes", "no": "no", "found": "found", "none": "—",
           "key": "Each address on the café's list, its code, and whether that code is one of the twelve in SHARED. "
                  "The last column says whether your unmask gives this address back under its code.",
           "said": "In all, the dictionary your unmask gives back has {n} pairs.",
           "wrong": "A dashed frame marks where it differs from the course's unmask: {k} of the 10 addresses.",
           "other": "It differs from the course's unmask in pairs that belong to no address on the list."},
    "nl": {"title": "Jouw unmask op de lijst van het café",
           "head": ["adres op de lijst", "de code", "in SHARED", "jouw unmask"],
           "yes": "ja", "no": "nee", "found": "gevonden", "none": "—",
           "key": "Elk adres op de lijst van het café, de code ervan, en of die code een van de twaalf in SHARED is. "
                  "De laatste kolom zegt of jouw unmask dit adres onder zijn code teruggeeft.",
           "said": "In totaal heeft de dictionary die jouw unmask teruggeeft {n} paren.",
           "wrong": "Een gestippeld kader wijst aan waar hij verschilt van de unmask van de course: bij {k} van de 10 "
                    "adressen.",
           "other": "Hij verschilt van de unmask van de course in paren die bij geen adres op de lijst horen."},
}


def unmask_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _UNMASK_WORDS[seen.lang]
    try:
        got = _call(fn, CODES, CAFE)
    except Exception:
        return ""
    if not isinstance(got, dict):
        return ""
    want = unmask(CODES, CAFE)
    rows, k = [], 0
    for email in CAFE:
        code = code_of(email)
        mine = got.get(code) == email
        wrong = mine != (want.get(code) == email)
        k += wrong
        rows.append(_row([_code(email), _code(code), w["yes"] if code in CODES else w["no"],
                          w["found"] if mine else w["none"]], _WRONG if wrong else _PLAIN, [5, 3, 2, 2]))
    note = w["key"] + " " + w["said"].format(n=len(got))
    if k:
        note += " " + w["wrong"].format(k=k)
    elif got != want:
        note += " " + w["other"]
    return yours.figure(w["title"], _table(w["head"], rows, [5, 3, 2, 2]) + f'<div style="{_UI}">{html.escape(note)}</div>')


_CANDIDATES_WORDS = {
    "en": {"title": "Your candidates, one fact at a time",
           "head": ["what is known", "rows your candidates keeps"],
           "nothing": "nothing yet",
           "key": "Your candidates on the twelve rows of SHARED, first with none of the facts of KNOWN, then with one "
                  "more each time.",
           "one": "With all three facts your candidates keeps one row, and so singles it out. Whoever knows these "
                  "three facts about Bo can now read everything else in that row:",
           "wrong": "A dashed frame marks where it differs from the course's candidates."},
    "nl": {"title": "Jouw candidates, één feit tegelijk",
           "head": ["wat bekend is", "rijen die jouw candidates houdt"],
           "nothing": "nog niets",
           "key": "Jouw candidates op de twaalf rijen van SHARED, eerst met geen van de feiten van KNOWN, daarna elke "
                  "keer met één feit meer.",
           "one": "Met alle drie de feiten houdt jouw candidates één rij over, en pikt die er dus uit. Wie deze drie "
                  "feiten over Bo kent, kan nu al het andere in die rij lezen:",
           "wrong": "Een gestippeld kader wijst aan waar hij verschilt van de candidates van de course."},
}


def candidates_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _CANDIDATES_WORDS[seen.lang]
    rows, k, last = [], 0, None
    facts = list(KNOWN)
    for n in range(len(facts) + 1):
        known = {key: KNOWN[key] for key in facts[:n]}
        try:
            got = _call(fn, known, SHARED)
        except Exception:
            got = None
        wrong = got != candidates(known, SHARED)
        k += wrong
        said = (f'<span style="display:inline-block;height:0.7rem;width:{4 * len(got)}%;min-width:2px;'
                f'background:var(--ea-ink-3);border-radius:2px;margin-right:0.4rem"></span>{len(got)}'
                if isinstance(got, list) else html.escape(repr(got)))
        what = ", ".join(f"{key}: {_code(value)}" for key, value in known.items()) or html.escape(w["nothing"])
        rows.append(_row([what, said], _WRONG if wrong else _PLAIN, [5, 4]))
        last = got
    note = w["key"] + (" " + w["wrong"] if k else "")
    out = _table(w["head"], rows, [5, 4]) + f'<div style="{_UI}">{html.escape(note)}</div>'
    if last == candidates(KNOWN, SHARED):
        row = SHARED[CODES.index(last[0])]
        out += (f'<div style="{_UI}">{html.escape(w["one"])}</div>'
                f'<div style="font-family:var(--ea-font-ui);font-size:0.83rem;max-width:22rem;color:var(--ea-ink)">'
                f'{_card(row, "goal")}</div>')
    return yours.figure(w["title"], out)


# --- the exercises ----------------------------------------------------------------------------------------------------

REQUIRED = ["pseudonymize", "unmask", "candidates", "recipient", "deleted"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# exercise 1: pseudonymize
_P_HEAD = ("def pseudonymize(row):\n"
           "    # a copy of the row, with the e-mail address replaced by its code under the key \"code\"\n"
           "    new = {}\n"
           "    for key in row:\n")
_P_TAIL = "    return new\n"
_P_START = _P_HEAD + "        ...\n" + _P_TAIL
_P_IF = "        if key == \"email\":\n"
_P_ELSE = "        else:\n            new[key] = row[key]\n"
_P = _P_HEAD + _P_IF + "            new[\"code\"] = code_of(row[key])\n" + _P_ELSE + _P_TAIL
_P_RAW = _P_HEAD + _P_IF + "            new[\"code\"] = row[key]\n" + _P_ELSE + _P_TAIL
_P_KEPT = _P_HEAD + "        new[key] = row[key]\n" + _P_IF + "            new[\"code\"] = code_of(row[key])\n" + _P_TAIL
_P_ONLY = _P_HEAD + _P_IF + "            new[\"code\"] = code_of(row[key])\n" + _P_TAIL
_P_KEY = _P_HEAD + _P_IF + "            new[\"email\"] = code_of(row[key])\n" + _P_ELSE + _P_TAIL

# exercise 2: unmask
_U_HEAD = ("def unmask(codes, emails):\n"
           "    # the codes that belong to one of the addresses: each code as a key, with its address as the value\n"
           "    found = {}\n"
           "    for email in emails:\n")
_U_TAIL = "    return found\n"
_U_START = _U_HEAD + "        ...\n" + _U_TAIL
_U = _U_HEAD + "        if code_of(email) in codes:\n            found[code_of(email)] = email\n" + _U_TAIL
_U_ALL = _U_HEAD + "        found[code_of(email)] = email\n" + _U_TAIL
_U_REVERSE = _U_HEAD + "        if code_of(email) in codes:\n            found[email] = code_of(email)\n" + _U_TAIL
_U_RAW = _U_HEAD + "        if email in codes:\n            found[code_of(email)] = email\n" + _U_TAIL

# exercise 3: candidates
_C_HEAD = ("def candidates(known, rows):\n"
           "    # the codes of the rows that match every fact in known\n"
           "    found = []\n"
           "    for row in rows:\n")
_C_TAIL = "    return found\n"
_C_START = _C_HEAD + "        ...\n" + _C_TAIL
_C_LOOP = ("        for key in known:\n"
           "            if row[key] != known[key]:\n"
           "                fits = False\n")
_C = _C_HEAD + "        fits = True\n" + _C_LOOP + "        if fits:\n            found.append(row[\"code\"])\n" + _C_TAIL
_C_ROW = _C_HEAD + "        fits = True\n" + _C_LOOP + "        if fits:\n            found.append(row)\n" + _C_TAIL
_C_ANY = _C_HEAD + ("        fits = False\n"
                    "        for key in known:\n"
                    "            if row[key] == known[key]:\n"
                    "                fits = True\n") + "        if fits:\n            found.append(row[\"code\"])\n" + _C_TAIL
_C_RESET = ("def candidates(known, rows):\n"
            "    # the codes of the rows that match every fact in known\n"
            "    found = []\n"
            "    fits = True\n"
            "    for row in rows:\n") + _C_LOOP + "        if fits:\n            found.append(row[\"code\"])\n" + _C_TAIL

_ROW_A = {"email": "a@mail.example", "language": "Dutch"}
_ROW_B = {"city": "Oslo", "email": "b@mail.example", "hour": 9}
_ROW_C = {"email": "c@mail.example"}
_PLAIN_ROWS = [{"code": "x1", "a": 1, "b": 2}, {"code": "x2", "a": 1, "b": 3}, {"code": "x3", "a": 2, "b": 2}]

EXERCISES = {
    "pseudonymize": FunctionBox(
        name="pseudonymize",
        picture=pseudonymize_picture,
        function="pseudonymize",
        start=_P_START,
        solution=_P,
        names={"code_of": code_of},
        visible=[(USERS[0],)],
        hidden=[(USERS[1],), (USERS[11],), (_ROW_A,), (_ROW_B,), (_ROW_C,)],
        first=[_DOTS],
        mistakes=[
            Mistake("raw", wrong=_P_RAW,
                    en="`{call}` gives {got}, and it should give {want}. Under `\"code\"` your copy holds the e-mail address itself. Put its code there: `code_of(...)` of the address.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Onder `\"code\"` staat in je kopie het e-mailadres zelf. Zet daar de code van het adres: `code_of(...)`."),
            Mistake("kept", wrong=_P_KEPT,
                    en="`{call}` gives {got}, and it should give {want}. Your copy has the code, and still the e-mail address too. The code replaces the address: leave the key `\"email\"` out of the copy.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je kopie heeft de code, en ook nog het e-mailadres. De code vervangt het adres: laat de key `\"email\"` weg uit de kopie."),
            Mistake("only", wrong=_P_ONLY,
                    en="`{call}` gives {got}, and it should give {want}. Your copy holds only the code; the other keys of the row, such as `\"language\"`, are missing. Copy every key other than `\"email\"` as it is.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je kopie bevat alleen de code; de andere keys van de rij, zoals `\"language\"`, ontbreken. Neem elke key behalve `\"email\"` over zoals hij is."),
            Mistake("key", wrong=_P_KEY,
                    en="`{call}` gives {got}, and it should give {want}. Your copy has the code under the key `\"email\"`. Put it under the key `\"code\"`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je kopie heeft de code onder de key `\"email\"`. Zet hem onder de key `\"code\"`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with an `if` and an `else`: for the key `\"email\"`, put the code of the address under `\"code\"` in `new`; copy every other key as it is.",
                      "nl": "Vervang de `...` door een `if` en een `else`: zet voor de key `\"email\"` de code van het adres onder `\"code\"` in `new`; neem elke andere key over zoals hij is."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Go over the keys of the row: the address becomes its code under `\"code\"`, every other key stays as it is.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Ga de keys van de rij langs: het adres wordt de code ervan onder `\"code\"`, elke andere key blijft zoals hij is."},
            "almost": {"en": "Right for `pseudonymize(USERS[0])`. But `{call}` gives {got}, and it should give {want}. Does your function use the `row` it gets, with all its keys?",
                       "nl": "Goed voor `pseudonymize(USERS[0])`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `row` die hij krijgt, met al zijn keys?"},
            "correct": {"en": "Your `pseudonymize` is right in all {n} cases. The picture shows what it gives for all twelve users.",
                        "nl": "Je `pseudonymize` klopt in alle {n} gevallen. Het plaatje laat zien wat hij voor alle twaalf gebruikers geeft."},
        },
    ),
    "unmask": FunctionBox(
        name="unmask",
        picture=unmask_picture,
        function="unmask",
        start=_U_START,
        solution=_U,
        names={"code_of": code_of},
        visible=[(CODES, CAFE)],
        hidden=[(CODES, CAFE[:3]), (CODES[:1], CAFE), ([], CAFE), (CODES, []),
                ([code_of("x@mail.example"), "ffffffff"], ["y@mail.example", "x@mail.example"])],
        first=[_DOTS],
        mistakes=[
            Mistake("all", wrong=_U_ALL,
                    en="`{call}` gives {got}, and it should give {want}. Your dictionary also holds addresses whose code is not in `codes`. Add an address only when its code is one of `codes`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je dictionary bevat ook adressen waarvan de code niet in `codes` staat. Voeg een adres alleen toe als de code ervan een van `codes` is."),
            Mistake("reverse", wrong=_U_REVERSE,
                    en="`{call}` gives {got}, and it should give {want}. Your dictionary has the address as the key and the code as the value; it should be the other way around: the code as the key.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je dictionary heeft het adres als key en de code als waarde; het moet andersom: de code als key."),
            Mistake("raw", wrong=_U_RAW,
                    en="`{call}` gives {got}, and it should give {want}. No address gets in. `codes` holds codes, not addresses: compare the code of each address, `code_of(email)`, with `codes`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Er komt geen enkel adres in. `codes` bevat codes, geen adressen: vergelijk de code van elk adres, `code_of(email)`, met `codes`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with two lines: when the code of `email` is in `codes`, put the address in `found` under its code.",
                      "nl": "Vervang de `...` door twee regels: als de code van `email` in `codes` staat, zet het adres dan onder zijn code in `found`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Go over the addresses; for each one whose code is in `codes`, add the code as a key with the address as its value.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Ga de adressen langs; voeg voor elk adres waarvan de code in `codes` staat de code toe als key, met het adres als waarde."},
            "almost": {"en": "Right for `unmask(CODES, CAFE)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `codes` and the `emails` it gets?",
                       "nl": "Goed voor `unmask(CODES, CAFE)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `codes` en de `emails` die hij krijgt?"},
            "correct": {"en": "Your `unmask` is right in all {n} cases. The picture goes over the café's list, address by address.",
                        "nl": "Je `unmask` klopt in alle {n} gevallen. Het plaatje gaat de lijst van het café langs, adres voor adres."},
        },
    ),
    "candidates": FunctionBox(
        name="candidates",
        picture=candidates_picture,
        function="candidates",
        start=_C_START,
        solution=_C,
        visible=[(KNOWN, SHARED)],
        hidden=[({}, SHARED), ({"language": "Japanese"}, SHARED), ({"hour": 7}, SHARED), ({"language": "Dutch"}, SHARED),
                ({"joined": 2023, "hour": 7}, SHARED), (KNOWN, []), ({"a": 1}, _PLAIN_ROWS), ({"b": 2, "a": 1}, _PLAIN_ROWS)],
        first=[_DOTS],
        mistakes=[
            Mistake("row", wrong=_C_ROW,
                    en="`{call}` gives {got}, and it should give {want}. Your list holds whole rows; it should hold their codes: `row[\"code\"]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je lijst bevat hele rijen; hij moet hun codes bevatten: `row[\"code\"]`."),
            Mistake("any", wrong=_C_ANY,
                    en="`{call}` gives {got}, and it should give {want}. Your function also keeps a row that matches only some of the facts. A row stays only when it matches every fact in `known`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie houdt ook een rij die maar een deel van de feiten klopt. Een rij blijft alleen als hij bij elk feit in `known` klopt."),
            Mistake("reset", wrong=_C_RESET,
                    en="`{call}` gives {got}, and it should give {want}. Once one row does not fit, your function lets no later row through. Start every row again with `fits = True`, inside the loop over the rows.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Zodra één rij niet past, laat je functie geen enkele latere rij meer door. Begin elke rij opnieuw met `fits = True`, binnen de loop over de rijen."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with your own lines: start with `fits = True`, set `fits` to `False` when the row differs from a fact in `known`, and add the code of the row to `found` when it still fits.",
                      "nl": "Vervang de `...` door je eigen regels: begin met `fits = True`, zet `fits` op `False` als de rij van een feit in `known` verschilt, en voeg de code van de rij toe aan `found` als hij dan nog past."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Go over the rows, and keep the code of each row that matches every fact in `known`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Ga de rijen langs, en houd de code van elke rij die bij elk feit in `known` klopt."},
            "almost": {"en": "Right for `candidates(KNOWN, SHARED)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `known` and the `rows` it gets?",
                       "nl": "Goed voor `candidates(KNOWN, SHARED)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `known` en de `rows` die hij krijgt?"},
            "correct": {"en": "Your `candidates` is right in all {n} cases. The picture adds the facts about Bo one at a time.",
                        "nl": "Je `candidates` klopt in alle {n} gevallen. Het plaatje voegt de feiten over Bo één voor één toe."},
        },
    ),
}

CASES = [
    ("pseudonymize", _P_START, "empty", None),
    ("pseudonymize", _P_HEAD + _P_IF + "            ...\n" + _P_ELSE + _P_TAIL, "wrong", "dots-left"),
    ("pseudonymize", _P, "correct", None),
    ("pseudonymize", _P_HEAD + ("        if key != \"email\":\n"
                                "            new[key] = row[key]\n"
                                "    new[\"code\"] = code_of(row[\"email\"])\n") + _P_TAIL, "correct", None),
    ("pseudonymize", _P_HEAD + ("        new[key] = row[key]\n"
                                "    new[\"code\"] = code_of(new.pop(\"email\"))\n") + _P_TAIL, "correct", None),
    ("pseudonymize", _P_RAW, "wrong", "raw"),
    ("pseudonymize", _P_KEPT, "wrong", "kept"),
    ("pseudonymize", _P_ONLY, "wrong", "only"),
    ("pseudonymize", _P_KEY, "wrong", "key"),
    ("pseudonymize", _P_HEAD + "        new[key] = row[key]\n" + _P_TAIL, "wrong", None),
    ("unmask", _U_START, "empty", None),
    ("unmask", _U_HEAD + "        if code_of(email) in codes:\n            ...\n" + _U_TAIL, "wrong", "dots-left"),
    ("unmask", _U, "correct", None),
    ("unmask", _U_HEAD + ("        code = code_of(email)\n"
                          "        if code in codes:\n"
                          "            found[code] = email\n") + _U_TAIL, "correct", None),
    ("unmask", _U_ALL, "wrong", "all"),
    ("unmask", _U_REVERSE, "wrong", "reverse"),
    ("unmask", _U_RAW, "wrong", "raw"),
    ("unmask", _U_HEAD + "        if code_of(email) not in codes:\n            found[code_of(email)] = email\n" + _U_TAIL,
     "wrong", None),
    ("candidates", _C_START, "empty", None),
    ("candidates", _C_HEAD + "        fits = True\n" + _C_LOOP + "        ...\n" + _C_TAIL, "wrong", "dots-left"),
    ("candidates", _C, "correct", None),
    ("candidates", _C_HEAD + ("        count = 0\n"
                              "        for key in known:\n"
                              "            if row[key] == known[key]:\n"
                              "                count = count + 1\n"
                              "        if count == len(known):\n"
                              "            found.append(row[\"code\"])\n") + _C_TAIL, "correct", None),
    ("candidates", _C_ROW, "wrong", "row"),
    ("candidates", _C_ANY, "wrong", "any"),
    ("candidates", _C_RESET, "wrong", "reset"),
    ("candidates", _C_HEAD + "        found.append(row[\"code\"])\n" + _C_TAIL, "wrong", None),
]
