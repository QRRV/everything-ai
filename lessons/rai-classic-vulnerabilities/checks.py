# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-classic-vulnerabilities: SQL injection on the website of the made-up library from the
module privacy, the fix with placeholders, and how the library keeps its members' passwords. The SQL runs on a small
database that lives only in memory and is made fresh for every call, so nothing touches the learner's files. The
three things the learner writes: an attack string that makes the library's loans search give every loan, loans_due
(the same search with placeholders) and check_login (a login against salted password hashes); with the pictures of
what they give, and the demo that puts any text into the search both ways. Members, loans and passwords are made up;
the books are old novels."""
import copy
import hashlib
import html
import re
import sqlite3

from everything_ai import yours
from everything_ai.check import CodeBox, FunctionBox, Mistake, run_code, with_time_limit


class Called(dict):
    """A dict the check calls by the name the lesson gives it, such as USERS, so that a message stays short. The
    learner's code gets its own copy, a plain dict."""

    def __init__(self, values: dict, name: str):
        super().__init__(values)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(dict(self), memo)


# --- the library's database ---------------------------------------------------------------------------------------------

MEMBER_NAMES = {"L-1001": "Anouk de Vries", "L-1002": "Bram Hendriks", "L-1003": "Yara El Amrani",
                "L-1004": "Joris van Dam", "L-1005": "Lina Postma", "L-1006": "Sem Brouwer", "L-1007": "Mila Koster",
                "L-1008": "Tim de Graaf", "L-1009": "Nora Willems", "L-1010": "Ravi Sitaram"}
JORIS = "L-1004"

# the catalog: old novels, each with its author
BOOKS = [("Pride and Prejudice", "Jane Austen"), ("Frankenstein", "Mary Shelley"), ("Max Havelaar", "Multatuli"),
         ("Alice's Adventures in Wonderland", "Lewis Carroll"), ("Treasure Island", "Robert Louis Stevenson"),
         ("Gulliver's Travels", "Jonathan Swift")]

# the loans: card, title, date it is due back
LOANS = [("L-1001", "Frankenstein", "2026-12-01"), ("L-1002", "Treasure Island", "2026-11-24"),
         ("L-1004", "Max Havelaar", "2026-11-26"), ("L-1004", "Pride and Prejudice", "2026-12-08"),
         ("L-1005", "Gulliver's Travels", "2026-12-03"), ("L-1007", "Alice's Adventures in Wonderland", "2026-11-30"),
         ("L-1007", "Frankenstein", "2026-12-10"), ("L-1008", "Pride and Prejudice", "2026-11-27"),
         ("L-1010", "Max Havelaar", "2026-12-04")]


def run(sql, values=None):
    """Runs one SQL statement on the library's database and gives back the first column of every row it finds, as a
    list. `values` fills the placeholders: a list for `?`, a dict for `:name`. The database lives only in memory and
    is made fresh for every call, so a statement cannot change anything that stays."""
    con = sqlite3.connect(":memory:")
    try:
        con.execute("CREATE TABLE books (title TEXT, author TEXT)")
        con.execute("CREATE TABLE loans (card TEXT, title TEXT, due TEXT)")
        con.executemany("INSERT INTO books VALUES (?, ?)", BOOKS)
        con.executemany("INSERT INTO loans VALUES (?, ?, ?)", LOANS)
        rows = con.execute(sql) if values is None else con.execute(sql, values)
        return [row[0] for row in rows]
    finally:
        con.close()


def glued(card) -> str:
    """The SQL of the library's loans search, with the card glued into it: what loans_for runs."""
    return f"SELECT title FROM loans WHERE card = '{card}'"


def loans_for(card):
    """The library's loans search, as its website has it: the card goes straight into the SQL."""
    return run(f"SELECT title FROM loans WHERE card = '{card}'")


def safe_loans(card):
    """The same search with a placeholder: the card goes to the database as data."""
    return run("SELECT title FROM loans WHERE card = ?", [card])


def loans_due(card, before):
    return run("SELECT title FROM loans WHERE card = ? AND due < ?", [card, before])


ATTACK = "x' OR 'a'='a"
ALL_TITLES = [title for _, title, _ in LOANS]


# --- passwords ----------------------------------------------------------------------------------------------------------

def slow_hash(password, salt):
    """The hash of a password with a salt (a hex text), as the library keeps it. PBKDF2 with SHA-256; with 1,000
    rounds, so that the checks stay quick: OWASP asks for 600,000 or more."""
    if not isinstance(password, str) or not isinstance(salt, str):
        raise TypeError("slow_hash needs a password and a salt, both texts")
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt) if salt else b"", 1000, dklen=8).hex()


# the members' passwords, which the library does not keep: only the course knows them, to make the hashes and to try
# logins with; Lina and Mila chose the same one
_PASSWORDS = {"L-1001": "rainy tuesday at the dunes", "L-1004": "seven herons over the dike",
              "L-1005": "pancakes with apple syrup", "L-1007": "pancakes with apple syrup",
              "L-1010": "blue tram to the harbor"}
_SALTS = {"L-1001": "3f9a1c07", "L-1004": "b25e7d10", "L-1005": "07c4e9a2", "L-1007": "e1d03b6f", "L-1010": "5a8f2c91"}
USERS = Called({card: Called({"salt": _SALTS[card], "hash": slow_hash(pw, _SALTS[card])}, f'USERS["{card}"]')
                for card, pw in _PASSWORDS.items()}, "USERS")
JORIS_PASSWORD = _PASSWORDS[JORIS]


def check_login(users, card, password):
    if card in users:
        user = users[card]
        return slow_hash(password, user["salt"]) == user["hash"]
    return False


# logins the picture tries: the right password, one capital letter different, Lina's password on Lina's own account,
# Joris's password on Mila's account, and a card without an account
TRIES = [(JORIS, JORIS_PASSWORD), (JORIS, "Seven herons over the dike"), ("L-1005", _PASSWORDS["L-1005"]),
         ("L-1007", JORIS_PASSWORD), ("L-1003", _PASSWORDS["L-1005"])]


# --- formatting -------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CODE = "font-family:var(--ea-font-code);font-size:0.8rem;color:var(--ea-ink)"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.78rem;color:var(--ea-ink-2)"
_CELL = "padding:0.15rem 0.5rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--ea-line)"


def _note(parts: list) -> str:
    return f'<div style="{_UI}">{html.escape(" ".join(p for p in parts if p))}</div>'


def _shown(v) -> str:
    """A value the way Python shows it, kept short."""
    try:
        text = repr(v)
    except Exception:
        text = "?"
    return text if len(text) <= 60 else text[:57] + "..."


_STOPPED = object()   # what a call gave back when it stopped with an error


def _call(fn, *args):
    try:
        return with_time_limit(lambda: fn(*copy.deepcopy(args)))
    except Exception:
        return _STOPPED


def _frame(wrong: bool) -> str:
    return "outline:1.5px dashed var(--ea-loss);outline-offset:-1px" if wrong else ""


def _square(filled: bool) -> str:
    fill = "background:var(--ea-ink)" if filled else "background:transparent"
    return (f'<span style="display:inline-block;width:0.8rem;height:0.8rem;border-radius:2px;{fill};'
            f'border:1px solid var(--ea-line);vertical-align:-0.1rem"></span>')


def _table(head: list, rows: list) -> str:
    th = "".join(f'<th style="{_CELL};{_LABEL};font-weight:600">{html.escape(h)}</th>' for h in head)
    body = "".join(f'<tr style="{style}">' + "".join(f'<td style="{_CELL}">{c}</td>' for c in cells) + "</tr>"
                   for cells, style in rows)
    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;'
            f'font-family:var(--ea-font-ui);font-size:0.85rem;color:var(--ea-ink);margin-top:0.3rem">'
            f'<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>')


def _sql_html(text) -> str:
    """The glued SQL of the loans search, with the typed text on a grey wash; from its first quote on, the typed text
    is read as SQL, which a dashed purple line under it marks."""
    text = str(text)
    cut = text.find("'")
    value, code = (text, "") if cut < 0 else (text[:cut], text[cut:])
    wash = "background:var(--ea-data-wash);border-radius:2px;padding:0 0.1rem"
    mark = "border-bottom:2px dashed var(--ea-loss)"
    typed = (f'<span style="{wash}">{html.escape(value)}'
             + (f'<span style="{mark}">{html.escape(code)}</span>' if code else "") + "</span>")
    return (f'<div style="{_CODE};overflow-wrap:anywhere;margin:0.2rem 0">'
            f'SELECT title FROM loans WHERE card = \'{typed}\'</div>')


def _rows_of(card) -> set:
    """Which of the nine loans the glued search finds for card, by their row numbers (1 to 9, in the order of LOANS):
    the same WHERE as loans_for, so that two loans of one title are told apart."""
    return set(run(f"SELECT rowid FROM loans WHERE card = '{card}'"))


def _loans_table(rows_found: set, lang: str, wrong: bool = False) -> str:
    """The nine loans, with a filled square for each loan a search gave back."""
    w = _WORDS[lang]
    rows = []
    for i, (card, title, due) in enumerate(LOANS, start=1):
        on = i in rows_found
        rows.append(([_square(on), html.escape(card), html.escape(MEMBER_NAMES[card]), html.escape(title),
                      html.escape(due)], "" if on else "color:var(--ea-ink-3)"))
    return f'<div style="{_frame(wrong)};max-width:36rem">' + _table(["", w["card"], w["member"], w["title"], w["due"]],
                                                                      rows) + "</div>"


_WORDS = {
    "en": {"card": "card", "member": "member", "title": "title", "due": "due", "salt": "salt", "hash": "hash",
           "typed": "password typed", "yours": "your check_login", "course": "the course's", "none": "no account"},
    "nl": {"card": "kaart", "member": "lid", "title": "titel", "due": "terug op", "salt": "salt", "hash": "hash",
           "typed": "getypt wachtwoord", "yours": "jouw check_login", "course": "die van de course",
           "none": "geen account"},
}


# --- the picture in The idea: what the database reads ---------------------------------------------------------------------

_IDEA_WORDS = {
    "en": {"title": "What the database reads",
           "normal": "Joris types a card number: one value between quotes.",
           "attack": "Someone types x' OR 'a'='a: the quote ends the value, and the rest is read as SQL.",
           "found": "{n} of the 9 loans.", "all": "All 9 loans.",
           "said": "Grey: the text that was typed. Dashed purple line: the part of it the database reads as SQL."},
    "nl": {"title": "Wat de database leest",
           "normal": "Joris typt een kaartnummer: één waarde tussen aanhalingstekens.",
           "attack": "Iemand typt x' OR 'a'='a: het aanhalingsteken sluit de waarde af, en de rest wordt als SQL gelezen.",
           "found": "{n} van de 9 uitleningen.", "all": "Alle 9 uitleningen.",
           "said": "Grijs: de getypte tekst. Gestreepte paarse lijn: het deel ervan dat de database als SQL leest."},
}


def idea_picture(lang: str) -> str:
    w = _IDEA_WORDS[lang]
    parts = []
    for key, text in (("normal", JORIS), ("attack", ATTACK)):
        n = len(loans_for(text))
        parts.append(f'<div style="margin:0.35rem 0 0.6rem"><div style="{_LABEL}">{html.escape(w[key])}</div>'
                     f'{_sql_html(text)}<div style="{_LABEL}">{html.escape(w["all"] if n == len(LOANS) else w["found"].format(n=n))}</div></div>')
    return yours.figure(w["title"], "".join(parts) + _note([w["said"]]))


# --- the picture under exercise 1: the attack -------------------------------------------------------------------------------

_ATTACK_WORDS = {
    "en": {"title": "Your attack on the loans search",
           "said": "The SQL loans_for makes of your text, and the loans it gives back, filled.",
           "all": "All 9 loans, of {m} members: the search no longer looks at the card.",
           "some": "{n} of the 9 loans.",
           "own": "Only the loans of {who}: what that member is meant to see.",
           "none": "No loans: your text stayed one value between the quotes, a card no member has.",
           "stops": "The database cannot read this SQL and stops with an error: the check above says more.",
           "off": "attack is {got}; the picture needs a text."},
    "nl": {"title": "Jouw aanval op de zoekfunctie voor uitleningen",
           "said": "De SQL die loans_for van je tekst maakt, en de uitleningen die het teruggeeft, gevuld.",
           "all": "Alle 9 uitleningen, van {m} leden: de zoekfunctie kijkt niet meer naar de kaart.",
           "some": "{n} van de 9 uitleningen.",
           "own": "Alleen de uitleningen van {who}: wat dat lid hoort te zien.",
           "none": "Geen uitleningen: je tekst bleef één waarde tussen de aanhalingstekens, een kaart die geen lid heeft.",
           "stops": "De database kan deze SQL niet lezen en stopt met een fout: de check hierboven zegt meer.",
           "off": "attack is {got}; het plaatje heeft een tekst nodig."},
}


def _members_of(found: list) -> int:
    left, cards = list(found), set()
    for card, title, _ in LOANS:
        if title in left:
            left.remove(title)
            cards.add(card)
    return len(cards)


def attack_picture(seen) -> str:
    run_ = seen.run
    w = _ATTACK_WORDS[seen.lang]
    text = run_.names.get("attack") if run_ is not None else None
    if text is None:
        return ""
    if not isinstance(text, str):
        return yours.figure(w["title"], _note([w["off"].format(got=_shown(text))]))
    sql = _sql_html(text)
    found = _call(loans_for, text)
    if found is _STOPPED:
        return yours.figure(w["title"], sql + _note([w["stops"]]))
    if found == ALL_TITLES:
        said = w["all"].format(m=_members_of(found))
    elif not found:
        said = w["none"]
    else:
        owners = {card for card, _, _ in LOANS if card == text}
        said = (w["own"].format(who=MEMBER_NAMES[text]) if owners and found == loans_for(text) and "'" not in text
                else w["some"].format(n=len(found)))
    return yours.figure(w["title"], sql + _loans_table(_rows_of(text), seen.lang) + _note([w["said"], said]))


# --- the picture under exercise 2: loans_due ---------------------------------------------------------------------------------

_DUE_CALLS = [(JORIS, "2026-12-01"), (ATTACK, "2026-12-31"), (JORIS, "2026-12-01' OR 'a'='a")]
_DUE_WORDS = {
    "en": {"title": "Your loans_due, with a normal card and with two attacks",
           "call": "call", "gives": "your loans_due gives", "want": "the course's",
           "said": "Each row is one call and the titles it gives back. With placeholders, an attack is only a strange "
                   "card or date: it finds nothing, or only what the real card has.",
           "fine": "The same titles as the course's loans_due.",
           "wrong": "A dashed purple frame marks a call where it differs from the course's loans_due.",
           "stops": "Your loans_due stops with an error on {call}: the check above says where.",
           "off": "Your loans_due gives {got} for {call}; the picture draws a list of titles."},
    "nl": {"title": "Jouw loans_due, met een gewone kaart en met twee aanvallen",
           "call": "aanroep", "gives": "jouw loans_due geeft", "want": "die van de course",
           "said": "Elke rij is één aanroep en de titels die hij teruggeeft. Met placeholders is een aanval alleen een "
                   "vreemde kaart of datum: hij vindt niets, of alleen wat de echte kaart heeft.",
           "fine": "Dezelfde titels als bij de loans_due van de course.",
           "wrong": "Een gestreept paars kader wijst een aanroep aan waar hij verschilt van de loans_due van de course.",
           "stops": "Jouw loans_due stopt met een fout bij {call}: de check hierboven zegt waar.",
           "off": "Jouw loans_due geeft {got} voor {call}; het plaatje tekent een lijst titels."},
}


def _fn(seen, name: str):
    mine = seen.mine
    if mine is None:
        return None
    fn = mine.names.get(name)
    return fn if callable(fn) else None


def _call_text(name: str, args: tuple) -> str:
    return f"{name}(" + ", ".join(_shown(a) if not isinstance(a, Called) else repr(a) for a in args) + ")"


def due_picture(seen) -> str:
    fn = _fn(seen, "loans_due")
    if fn is None:
        return ""
    w = _DUE_WORDS[seen.lang]
    rows, wrong_any = [], False
    for args in _DUE_CALLS:
        call = _call_text("loans_due", args)
        got = _call(fn, *args)
        if got is _STOPPED:
            return yours.figure(w["title"], _note([w["stops"].format(call=call)]))
        if not isinstance(got, list):
            return yours.figure(w["title"], _note([w["off"].format(got=_shown(got), call=call)]))
        want = loans_due(*args)
        wrong = got != want
        wrong_any |= wrong
        rows.append(([f'<code style="{_CODE}">{html.escape(call)}</code>', html.escape(_shown(got)),
                      html.escape(_shown(want))], _frame(wrong)))
    table = _table([w["call"], w["gives"], w["want"]], rows)
    return yours.figure(w["title"], table + _note([w["said"], w["wrong"] if wrong_any else w["fine"]]))


# --- the picture beside exercise 3: what the library stores, and the one under it: logins -------------------------------------

_USERS_WORDS = {
    "en": {"title": "USERS: what the library stores for each account",
           "said": "No passwords: a salt and a hash for each card. Lina (L-1005) and Mila (L-1007) chose the same "
                   "password, yet their hashes differ, because their salts do."},
    "nl": {"title": "USERS: wat de bibliotheek per account bewaart",
           "said": "Geen wachtwoorden: een salt en een hash per kaart. Lina (L-1005) en Mila (L-1007) kozen hetzelfde "
                   "wachtwoord, en toch verschillen hun hashes, omdat hun salts verschillen."},
}


def users_picture(lang: str) -> str:
    w, t = _USERS_WORDS[lang], _WORDS[lang]
    rows = [([html.escape(card), html.escape(MEMBER_NAMES[card]), f'<code style="{_CODE}">{user["salt"]}</code>',
              f'<code style="{_CODE}">{user["hash"]}</code>'], "") for card, user in USERS.items()]
    return yours.figure(w["title"], _table([t["card"], t["member"], t["salt"], t["hash"]], rows) + _note([w["said"]]))


_LOGIN_WORDS = {
    "en": {"title": "Your check_login on five tries",
           "said": "Each row is one login: the card, the password typed, and what your check_login says.",
           "fine": "The same answers as the course's check_login.",
           "wrong": "A dashed purple frame marks a login where it differs from the course's check_login.",
           "stops": "Your check_login stops with an error on {call}: the check above says where.",
           "off": "Your check_login gives {got} for {call}; the picture draws True or False."},
    "nl": {"title": "Jouw check_login bij vijf pogingen",
           "said": "Elke rij is één keer inloggen: de kaart, het getypte wachtwoord, en wat jouw check_login zegt.",
           "fine": "Dezelfde antwoorden als bij de check_login van de course.",
           "wrong": "Een gestreept paars kader wijst een poging aan waar hij verschilt van de check_login van de course.",
           "stops": "Jouw check_login stopt met een fout bij {call}: de check hierboven zegt waar.",
           "off": "Jouw check_login geeft {got} voor {call}; het plaatje tekent True of False."},
}


def login_picture(seen) -> str:
    fn = _fn(seen, "check_login")
    if fn is None:
        return ""
    w, t = _LOGIN_WORDS[seen.lang], _WORDS[seen.lang]
    rows, wrong_any = [], False
    for card, password in TRIES:
        call = _call_text("check_login", (USERS, card, password))
        got = _call(fn, USERS, card, password)
        if got is _STOPPED:
            return yours.figure(w["title"], _note([w["stops"].format(call=call)]))
        if not isinstance(got, bool):
            return yours.figure(w["title"], _note([w["off"].format(got=_shown(got), call=call)]))
        want = check_login(USERS, card, password)
        wrong = got != want
        wrong_any |= wrong
        who = MEMBER_NAMES[card] if card in USERS else f'{MEMBER_NAMES[card]} ({t["none"]})'
        rows.append(([html.escape(card), html.escape(who), f'<code style="{_CODE}">{html.escape(password)}</code>',
                      str(got)], _frame(wrong)))
    table = _table([t["card"], t["member"], t["typed"], t["yours"]], rows)
    return yours.figure(w["title"], table + _note([w["said"], w["wrong"] if wrong_any else w["fine"]]))


# --- the demo: any text, glued in and with a placeholder ----------------------------------------------------------------------

_DEMO_WORDS = {
    "en": {"title": "Your text in the loans search, both ways",
           "glued": "Glued into the SQL, as loans_for does:",
           "bound": "With a placeholder, as in Exercise 2: the SQL stays the same, and your text goes along as data:",
           "found": "{n} of the 9 loans: {titles}",
           "nothing": "No loans.",
           "error": "The database cannot read this SQL and stops with an error: {error}",
           "two": "sqlite3 runs one statement at a time and refuses this one: {error}"},
    "nl": {"title": "Jouw tekst in de zoekfunctie voor uitleningen, op twee manieren",
           "glued": "In de SQL geplakt, zoals loans_for doet:",
           "bound": "Met een placeholder, zoals in oefening 2: de SQL blijft hetzelfde, en je tekst gaat mee als gegevens:",
           "found": "{n} van de 9 uitleningen: {titles}",
           "nothing": "Geen uitleningen.",
           "error": "De database kan deze SQL niet lezen en stopt met een fout: {error}",
           "two": "sqlite3 voert één opdracht tegelijk uit en weigert deze: {error}"},
}


def _found_text(w: dict, found: list) -> str:
    return w["found"].format(n=len(found), titles=", ".join(found)) if found else w["nothing"]


def demo_picture(text, lang: str) -> str:
    w = _DEMO_WORDS[lang]
    text = "" if text is None else str(text)
    try:
        said = _found_text(w, loans_for(text))
    except sqlite3.ProgrammingError as e:
        said = w["two"].format(error=e)
    except sqlite3.Error as e:
        said = w["error"].format(error=e)
    first = (f'<div style="{_LABEL}">{html.escape(w["glued"])}</div>{_sql_html(text)}'
             f'<div style="{_UI};margin-bottom:0.7rem">{html.escape(said)}</div>')
    wash = "background:var(--ea-data-wash);border-radius:2px;padding:0 0.1rem"
    second = (f'<div style="{_LABEL}">{html.escape(w["bound"])}</div>'
              f'<div style="{_CODE};overflow-wrap:anywhere;margin:0.2rem 0">SELECT title FROM loans WHERE card = ?'
              f'&nbsp;&nbsp;&nbsp;[<span style="{wash}">{html.escape(repr(text))}</span>]</div>'
              f'<div style="{_UI}">{html.escape(_found_text(w, safe_loans(text)))}</div>')
    return yours.figure(w["title"], first + second)


# --- the exercises ----------------------------------------------------------------------------------------------------

REQUIRED = ["attack", "loans-due", "login", "secret", "access"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# exercise 1: the attack
_K_START = ("# loans_for glues the text you give it into its SQL; make it give back every loan\n"
            "attack = \"...\"\n"
            "loans_for(attack)\n")
_K = ("# loans_for glues the text you give it into its SQL; make it give back every loan\n"
      "attack = \"x' OR 'a'='a\"\n"
      "loans_for(attack)\n")


def _own_card(got, text) -> bool:
    return isinstance(got, list) and bool(got) and any(got == loans_for(c) for c in MEMBER_NAMES) and got != ALL_TITLES


def _is_two(e: BaseException) -> bool:
    return isinstance(e, (sqlite3.ProgrammingError, sqlite3.Warning)) and "one statement" in str(e)


# exercise 2: loans_due
_D_HEAD = ("def loans_due(card, before):\n"
           "    # the titles of the loans of card that are due back before the date before, such as \"2026-12-01\"\n")
_D_START = _D_HEAD + "    ...\n"
_D = _D_HEAD + "    return run(\"SELECT title FROM loans WHERE card = ? AND due < ?\", [card, before])\n"
_D_GLUED = _D_HEAD + "    return run(f\"SELECT title FROM loans WHERE card = '{card}' AND due < '{before}'\")\n"
_D_ORDER = _D_HEAD + "    return run(\"SELECT title FROM loans WHERE card = ? AND due < ?\", [before, card])\n"
_D_ON = _D_HEAD + "    return run(\"SELECT title FROM loans WHERE card = ? AND due <= ?\", [card, before])\n"
_D_QUOTED = _D_HEAD + "    return run(\"SELECT title FROM loans WHERE card = '?' AND due < '?'\", [card, before])\n"
_D_ONE = _D_HEAD + "    return run(\"SELECT title FROM loans WHERE card = ?\", [card])\n"
_D_TEXT = _D_HEAD + "    return run(\"SELECT title FROM loans WHERE card = ? AND due < ?\", card)\n"


def _is_quoted(e: BaseException) -> bool:
    return isinstance(e, sqlite3.ProgrammingError) and "uses 0" in str(e)


def _is_bindings(e: BaseException) -> bool:
    return isinstance(e, sqlite3.ProgrammingError) and "bindings" in str(e) and "uses 0" not in str(e)


def _is_sql_error(e: BaseException) -> bool:
    return isinstance(e, sqlite3.OperationalError)


# exercise 3: check_login
_L_HEAD = ("def check_login(users, card, password):\n"
           "    # users[card] has the \"salt\" and the \"hash\" of the password, never the password itself\n")
_L_START = _L_HEAD + "    ...\n"
_L = _L_HEAD + ("    if card in users:\n"
                "        user = users[card]\n"
                "        return slow_hash(password, user[\"salt\"]) == user[\"hash\"]\n"
                "    return False\n")
_L_PLAIN = _L_HEAD + ("    if card in users:\n"
                      "        return password == users[card][\"hash\"]\n"
                      "    return False\n")
_L_NOSALT = _L_HEAD + ("    if card in users:\n"
                       "        return slow_hash(password, \"\") == users[card][\"hash\"]\n"
                       "    return False\n")
_L_KNOWN = _L_HEAD + "    return card in users\n"
_L_KEY = _L_HEAD + "    return slow_hash(password, users[card][\"salt\"]) == users[card][\"hash\"]\n"


def _key_error(e: BaseException) -> bool:
    return isinstance(e, KeyError)


def _code(text) -> str:
    """The code without its comments."""
    return "\n".join(line.split("#", 1)[0] for line in str(text).splitlines())


def _plain_compare(got, text) -> bool:
    return re.search(r"\bpassword\s*==|==\s*password\b", _code(text)) is not None and "slow_hash" not in _code(text)


def _no_salt(got, text) -> bool:
    calls = re.findall(r"slow_hash\(([^)]*)\)", _code(text))
    return bool(calls) and all("salt" not in c for c in calls)


def _type_error(e: BaseException) -> bool:
    return isinstance(e, TypeError) and "slow_hash" in str(e) or isinstance(e, TypeError) and "positional" in str(e)


_NO_USERS = Called({}, "{}")

EXERCISES = {
    "attack": CodeBox(
        name="attack",
        picture=attack_picture,
        start=_K_START,
        solution=_K,
        names={"loans_for": loans_for},
        first=[_DOTS],
        mistakes=[
            Mistake("two", error=_is_two,
                    en="sqlite3 runs one statement at a time, so it refuses a `;` with a second statement after it. Stay inside the one `SELECT`: close the card's quote and add a condition with `OR` that is always true.",
                    nl="sqlite3 voert één opdracht tegelijk uit, dus weigert het een `;` met een tweede opdracht erachter. Blijf binnen die ene `SELECT`: sluit het aanhalingsteken van de kaart af en voeg met `OR` een voorwaarde toe die altijd waar is."),
            Mistake("syntax", error=_is_sql_error,
                    en="The database cannot read the SQL your text makes. loans_for puts a `'` before your text and another one after it: count the quotes. The last part of your text has to leave one quote open, for that last `'` to close.",
                    nl="De database kan de SQL die je tekst maakt niet lezen. loans_for zet een `'` voor je tekst en nog een erna: tel de aanhalingstekens. Het laatste stuk van je tekst moet één aanhalingsteken openlaten, zodat die laatste `'` het afsluit."),
            Mistake("own", matches=_own_card,
                    en="`loans_for(attack)` gives {got}: the loans of one card, what that member is meant to see. Your text has to get out of the quotes: start it with a `'` to close the card, then add a condition with `OR` that is always true.",
                    nl="`loans_for(attack)` geeft {got}: de uitleningen van één kaart, wat dat lid hoort te zien. Je tekst moet buiten de aanhalingstekens komen: begin met een `'` om de kaart af te sluiten, en voeg dan met `OR` een voorwaarde toe die altijd waar is."),
            Mistake("none", matches=lambda got, text: got == [],
                    en="`loans_for(attack)` gives no loans: the database read your whole text as one card number, and no member has that card. Close the card with a `'` first; what comes after it is read as SQL.",
                    nl="`loans_for(attack)` geeft geen uitleningen: de database las je hele tekst als één kaartnummer, en geen lid heeft die kaart. Sluit de kaart eerst af met een `'`; wat daarna komt, wordt als SQL gelezen."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` between the double quotes with your text for the search.",
                      "nl": "Vervang de `...` tussen de dubbele aanhalingstekens door je tekst voor de zoekfunctie."},
            "no_value": {"en": "The last line has to be `loans_for(attack)`, so that the check sees what the search gives back.",
                         "nl": "De laatste regel moet `loans_for(attack)` zijn, zodat de check ziet wat de zoekfunctie teruggeeft."},
            "wrong": {"en": "`loans_for(attack)` gives {got}. That is not every loan yet: the condition you add after the card has to be true for every row.",
                      "nl": "`loans_for(attack)` geeft {got}. Dat zijn nog niet alle uitleningen: de voorwaarde die je na de kaart toevoegt, moet voor elke rij waar zijn."},
            "correct": {"en": "Right: `loans_for(attack)` gives all 9 loans, of 7 members. The picture shows the SQL your text made.",
                        "nl": "Goed: `loans_for(attack)` geeft alle 9 uitleningen, van 7 leden. Het plaatje laat de SQL zien die je tekst maakte."},
        },
    ),
    "loans-due": FunctionBox(
        name="loans-due",
        picture=due_picture,
        function="loans_due",
        names={"run": run},
        start=_D_START,
        solution=_D,
        visible=[(JORIS, "2026-12-01")],
        hidden=[(JORIS, "2026-12-31"), ("L-1007", "2026-12-31"), ("L-1007", "2026-11-30"), ("L-1003", "2026-12-31"),
                (ATTACK, "2026-12-31"), (JORIS, "2026-12-01' OR 'a'='a")],
        first=[_DOTS],
        mistakes=[
            Mistake("quoted", error=_is_quoted,
                    en="With `{call}`, the database finds no placeholder to fill. Is there a `'` around a `?`? Then it is a text with a question mark in it. Write the `?` without quotes: the database knows the value is a text.",
                    nl="Met `{call}` vindt de database geen placeholder om in te vullen. Staat er een `'` om een `?`? Dan is het een tekst met een vraagteken erin. Schrijf de `?` zonder aanhalingstekens: de database weet dat de waarde een tekst is."),
            Mistake("bindings", error=_is_bindings,
                    en="With `{call}`, the number of values does not match the number of `?`. `run` gets the values as a list, one for each `?`, in the same order: a text on its own counts as one value per letter.",
                    nl="Met `{call}` klopt het aantal waarden niet met het aantal `?`. `run` krijgt de waarden als een lijst, één voor elke `?`, in dezelfde volgorde: een losse tekst telt als één waarde per letter."),
            Mistake("syntax", error=_is_sql_error,
                    en="With `{call}`, the database cannot read your SQL. The table is `loans`, its columns are `card`, `title` and `due`, and two conditions are joined with `AND`.",
                    nl="Met `{call}` kan de database je SQL niet lezen. De tabel heet `loans`, de kolommen `card`, `title` en `due`, en twee voorwaarden zet je aan elkaar met `AND`."),
            Mistake("glued", wrong=_D_GLUED,
                    en="`{call}` gives {got}, and it should give {want}. Your function glues the card and the date into the SQL, so the attack in the call rewrites it. Put a `?` where each value goes, and give the values to `run` separately.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie plakt de kaart en de datum in de SQL, dus de aanval in de aanroep herschrijft die. Zet een `?` waar elke waarde komt, en geef de waarden los aan `run`."),
            Mistake("order", wrong=_D_ORDER,
                    en="`{call}` gives {got}, and it should give {want}. The values fill the `?` in order: the first value goes to the first `?`. Is the date in the place of the card?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. De waarden vullen de `?` op volgorde: de eerste waarde gaat naar de eerste `?`. Staat de datum op de plek van de kaart?"),
            Mistake("on", wrong=_D_ON,
                    en="`{call}` gives {got}, and it should give {want}. A loan due back on that very date is not due before it: use `<`, not `<=`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Een uitlening die precies op die datum terug moet, hoeft niet vóór die datum terug: gebruik `<`, niet `<=`."),
            Mistake("one", wrong=_D_ONE,
                    en="`{call}` gives {got}, and it should give {want}. Your function gives every loan of the card. Add a second condition, on `due`, with its own `?`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie geeft elke uitlening van de kaart. Voeg een tweede voorwaarde toe, op `due`, met een eigen `?`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: write the SQL with a `?` for the card and one for the date, and give `run` the values.",
                      "nl": "Vervang de `...`: schrijf de SQL met een `?` voor de kaart en een voor de datum, en geef `run` de waarden."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The search wants the titles of the loans of that card that are due back before that date.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De zoekfunctie wil de titels van de uitleningen van die kaart die voor die datum terug moeten."},
            "almost": {"en": "Right for `loans_due(\"L-1004\", \"2026-12-01\")`. But `{call}` gives {got}, and it should give {want}. Does your function use the card and the date it gets?",
                       "nl": "Goed voor `loans_due(\"L-1004\", \"2026-12-01\")`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de kaart en de datum die hij krijgt?"},
            "correct": {"en": "Your `loans_due` is right in all {n} cases, also with the two attacks. The picture shows three of them.",
                        "nl": "Je `loans_due` klopt in alle {n} gevallen, ook met de twee aanvallen. Het plaatje laat er drie zien."},
        },
    ),
    "login": FunctionBox(
        name="login",
        picture=login_picture,
        function="check_login",
        names={"slow_hash": slow_hash},
        start=_L_START,
        solution=_L,
        visible=[(USERS, JORIS, JORIS_PASSWORD)],
        hidden=[(USERS, JORIS, "Seven herons over the dike"), (USERS, "L-1005", _PASSWORDS["L-1005"]),
                (USERS, "L-1007", _PASSWORDS["L-1007"]), (USERS, "L-1007", JORIS_PASSWORD),
                (USERS, "L-1003", _PASSWORDS["L-1005"]), (USERS, "L-1010", ""), (_NO_USERS, JORIS, JORIS_PASSWORD)],
        first=[_DOTS],
        mistakes=[
            Mistake("key", error=_key_error,
                    en="With `{call}`, your function stops with an error: it looks up a card that has no account. Check with `in` first whether the card is in `users`, and give back `False` when it is not.",
                    nl="Met `{call}` stopt je functie met een fout: hij zoekt een kaart op die geen account heeft. Kijk eerst met `in` of de kaart in `users` staat, en geef `False` terug als dat niet zo is."),
            Mistake("args", error=_type_error,
                    en="With `{call}`, `slow_hash` stops: it needs two texts, the password and the salt, in that order.",
                    nl="Met `{call}` stopt `slow_hash`: die heeft twee teksten nodig, het wachtwoord en de salt, in die volgorde."),
            Mistake("plain", matches=_plain_compare,
                    en="`{call}` gives {got}, and it should give {want}. Your function compares the typed password with the stored hash, and a password never looks like its hash. Hash the typed password first, with `slow_hash`, and compare the two hashes.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie vergelijkt het getypte wachtwoord met de bewaarde hash, en een wachtwoord lijkt nooit op zijn hash. Hash eerst het getypte wachtwoord, met `slow_hash`, en vergelijk de twee hashes."),
            Mistake("nosalt", matches=_no_salt,
                    en="`{call}` gives {got}, and it should give {want}. Your function hashes the password without the salt of the account, so its hash never matches. Give `slow_hash` the salt that is stored with that card.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie hasht het wachtwoord zonder de salt van het account, dus de hash klopt nooit. Geef `slow_hash` de salt die bij die kaart bewaard is."),
            Mistake("known", wrong=_L_KNOWN,
                    en="`{call}` gives {got}, and it should give {want}. Your function lets in anyone who types a card that has an account, whatever the password. Check the password too.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie laat iedereen binnen die een kaart met een account typt, welk wachtwoord ook. Controleer ook het wachtwoord."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: look up the account of the card, hash the typed password with its salt, and compare with the stored hash.",
                      "nl": "Vervang de `...`: zoek het account van de kaart op, hash het getypte wachtwoord met de salt ervan, en vergelijk met de bewaarde hash."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. A login is right when the card has an account and the hash of the typed password, made with that account's salt, is the stored hash.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Inloggen klopt als de kaart een account heeft en de hash van het getypte wachtwoord, gemaakt met de salt van dat account, de bewaarde hash is."},
            "almost": {"en": "Right for the right password of Joris. But `{call}` gives {got}, and it should give {want}. Does your function use the card and the password it gets?",
                       "nl": "Goed voor het juiste wachtwoord van Joris. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de kaart en het wachtwoord die hij krijgt?"},
            "correct": {"en": "Your `check_login` is right in all {n} cases. The picture tries five logins with it.",
                        "nl": "Je `check_login` klopt in alle {n} gevallen. Het plaatje probeert er vijf keer mee in te loggen."},
        },
    ),
}

CASES = [
    ("attack", _K_START, "empty", None),
    ("attack", _K, "correct", None),
    ("attack", "attack = \"' OR '1'='1\"\nloans_for(attack)\n", "correct", None),
    ("attack", "attack = \"x' OR 1=1 --\"\nloans_for(attack)\n", "correct", None),
    ("attack", "attack = \"L-1004' OR card <> 'L-1004\"\nloans_for(attack)\n", "correct", None),
    ("attack", "attack = \"L-1004\"\nloans_for(attack)\n", "wrong", "own"),
    ("attack", "attack = \"L-1004' OR card = 'L-1007\"\nloans_for(attack)\n", "wrong", None),
    ("attack", "attack = \"x OR a = a\"\nloans_for(attack)\n", "wrong", "none"),
    ("attack", "attack = \"x' OR 'a'='a'\"\nloans_for(attack)\n", "error", "syntax"),
    ("attack", "attack = \"x'; SELECT title FROM loans; --\"\nloans_for(attack)\n", "error", "two"),
    ("attack", "attack = \"x' OR 'a'='a\"\n", "wrong", "no-value"),
    ("loans-due", _D_START, "empty", None),
    ("loans-due", _D, "correct", None),
    ("loans-due", _D_HEAD + "    return run(\"SELECT title FROM loans WHERE card = :card AND due < :before\", {\"card\": card, \"before\": before})\n", "correct", None),
    ("loans-due", _D_HEAD + "    sql = \"SELECT title FROM loans WHERE due < ? AND card = ?\"\n    return run(sql, [before, card])\n", "correct", None),
    ("loans-due", _D_GLUED, "wrong", "glued"),
    ("loans-due", _D_ORDER, "wrong", "order"),
    ("loans-due", _D_ON, "wrong", "on"),
    ("loans-due", _D_ONE, "wrong", "one"),
    ("loans-due", _D_QUOTED, "error", "quoted"),
    ("loans-due", _D_TEXT, "error", "bindings"),
    ("loans-due", _D_HEAD + "    return run(\"SELECT title FROM loan WHERE card = ? AND due < ?\", [card, before])\n", "error", "syntax"),
    ("loans-due", _D_HEAD + "    return [\"Max Havelaar\"]\n", "almost", None),
    ("loans-due", _D_HEAD + "    return run(\"SELECT title FROM loans WHERE card = ? AND due > ?\", [card, before])\n", "wrong", None),
    ("login", _L_START, "empty", None),
    ("login", _L, "correct", None),
    ("login", _L_HEAD + "    if card in users:\n        return slow_hash(password, users[card][\"salt\"]) == users[card][\"hash\"]\n    else:\n        return False\n", "correct", None),
    ("login", _L_KEY, "error", "key"),
    ("login", _L_HEAD + "    if card in users:\n        return slow_hash(password) == users[card][\"hash\"]\n    return False\n", "error", "args"),
    ("login", _L_PLAIN, "wrong", "plain"),
    ("login", _L_NOSALT, "wrong", "nosalt"),
    ("login", _L_KNOWN, "wrong", "known"),
    ("login", _L_HEAD + "    return True\n", "almost", None),
    ("login", _L_HEAD + "    return False\n", "wrong", None),
]
