# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-gdpr: the six lawful bases of Article 6(1) GDPR (BASES), and three made-up tables of a
made-up language-learning app: a register of what it does with personal data (REGISTER), what it collected data for
and what it uses them for (USES), and posts on its forum (POSTS). The three functions the learner writes: has_basis
(for each activity, whether its basis is one of the six), new_purposes (the data used for another purpose than the one
they were collected for) and erase (the posts without the ones by or about one person); with the pictures of what
the learner's functions give, and the figure of the six bases."""
import html
import re

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake


class NamedList(list):
    """A list the check calls by its name, such as REGISTER, so that a message stays short."""

    def __init__(self, items, name: str):
        super().__init__(items)
        self.name = name

    def __repr__(self):
        return self.name


# --- the six lawful bases of Article 6(1), in its order (a) to (f) ---------------------------------------------------
BASES = NamedList(["consent", "contract", "legal obligation", "vital interests", "public task",
                   "legitimate interests"], "BASES")
DUTCH = {"consent": "toestemming", "contract": "overeenkomst", "legal obligation": "wettelijke verplichting",
         "vital interests": "vitaal belang", "public task": "taak van algemeen belang",
         "legitimate interests": "gerechtvaardigd belang"}

# --- a made-up language-learning app: what it does with personal data, and the basis its register names ---------------
REGISTER = NamedList([
    {"activity": "send lesson reminders", "basis": "consent"},
    {"activity": "bill subscriptions", "basis": "contract"},
    {"activity": "keep invoices for the tax office", "basis": "legal obligation"},
    {"activity": "detect fraud in payments", "basis": "legitimate interests"},
    {"activity": "train a model on posts scraped from other sites", "basis": "public data"},
    {"activity": "share progress with advertisers", "basis": "none"},
    {"activity": "train the tutor bot on users' chats", "basis": "unknown"},
], "REGISTER")

# --- what the same app collected data for, and what it uses them for ---------------------------------------------------
USES = NamedList([
    {"data": "email addresses", "collected_for": "lesson reminders", "used_for": "lesson reminders"},
    {"data": "voice recordings", "collected_for": "pronunciation feedback", "used_for": "training a speech model"},
    {"data": "lesson scores", "collected_for": "showing progress", "used_for": "showing progress"},
    {"data": "chats with the tutor bot", "collected_for": "answering questions", "used_for": "training the tutor bot"},
    {"data": "payment details", "collected_for": "billing", "used_for": "detecting fraud"},
], "USES")

# --- seven made-up posts on the app's forum, each with its writer and the people it names ------------------------------
POSTS = NamedList([
    {"id": 1, "author": "sam", "mentions": [], "text": "Does anyone know a good podcast in Spanish?"},
    {"id": 2, "author": "noor", "mentions": ["sam"], "text": "Try the one about cooking, sam."},
    {"id": 3, "author": "kai", "mentions": [], "text": "Day 30 of my streak!"},
    {"id": 4, "author": "sam", "mentions": ["kai"], "text": "Well done, kai!"},
    {"id": 5, "author": "ash", "mentions": [], "text": "I keep mixing up ser and estar."},
    {"id": 6, "author": "lior", "mentions": ["sam", "ash"], "text": "sam and ash, shall we practice on Friday?"},
    {"id": 7, "author": "remi", "mentions": ["noor"], "text": "Thanks for the tip, noor."},
], "POSTS")


# --- the course's own versions of the three functions ---------------------------------------------------------------

def has_basis(register, bases):
    """For each activity in the register: True when its basis is one of the bases."""
    result = {}
    for row in register:
        result[row["activity"]] = row["basis"] in bases
    return result


def new_purposes(uses):
    """The data that are used for another purpose than the one they were collected for."""
    found = []
    for use in uses:
        if use["used_for"] != use["collected_for"]:
            found.append(use["data"])
    return found


def erase(posts, name):
    """The posts without the ones by `name` and the ones that mention `name`."""
    kept = []
    for post in posts:
        if post["author"] != name and name not in post["mentions"]:
            kept.append(post)
    return kept


# --- the pictures ---------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_BOX = "padding:0.25rem 0.5rem;margin:0.2rem 0;border-radius:4px;"
_PLAIN = "border:1px solid var(--ea-line);color:var(--ea-ink)"
_WRONG = "border:1.5px dashed var(--ea-loss);color:var(--ea-loss-text)"


def _basis_name(basis: str, lang: str) -> str:
    return basis if lang == "en" or basis not in DUTCH else f"{DUTCH[basis]} ({basis})"


def _join(items: list, lang: str) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + {"en": " and ", "nl": " en "}[lang] + items[-1]


_BASES_WORDS = {
    "en": {"title": "The six lawful bases of the GDPR (Article 6(1))",
           "names": {"consent": "consent of the person", "contract": "needed for a contract with the person",
                     "legal obligation": "needed for a legal obligation",
                     "vital interests": "needed to protect someone's vital interests, such as their life",
                     "public task": "needed for a task in the public interest or for official authority",
                     "legitimate interests": "needed for a legitimate interest, unless the person's interests or rights weigh more"},
           "steps": ["1. Is there a legitimate interest?", "2. Is the processing necessary for it?",
                     "3. Does it hold up against the person's interests and rights?"],
           "test": "The three-step test for legitimate interests, as the EDPB sets it out",
           "key": "Processing is lawful only if at least one of the six applies. There is no hierarchy between them: "
                  "consent is one basis, not the default."},
    "nl": {"title": "De zes grondslagen van de AVG (artikel 6, lid 1)",
           "names": {"consent": "toestemming van de persoon", "contract": "nodig voor een overeenkomst met de persoon",
                     "legal obligation": "nodig voor een wettelijke verplichting",
                     "vital interests": "nodig om iemands vitale belangen te beschermen, zoals het leven",
                     "public task": "nodig voor een taak van algemeen belang of voor openbaar gezag",
                     "legitimate interests": "nodig voor een gerechtvaardigd belang, tenzij de belangen of rechten van de persoon zwaarder wegen"},
           "steps": ["1. Is er een gerechtvaardigd belang?", "2. Is de verwerking daarvoor nodig?",
                     "3. Houdt het stand tegenover de belangen en rechten van de persoon?"],
           "test": "De toets in drie stappen voor gerechtvaardigd belang, zoals de EDPB die beschrijft",
           "key": "Verwerken mag alleen als ten minste één van de zes van toepassing is. Er is geen rangorde tussen "
                  "de zes: toestemming is één grondslag, niet de standaard."},
}


def bases_picture(lang: str) -> str:
    """The course's figure of the six lawful bases and the three-step test for legitimate interests, for The idea."""
    w = _BASES_WORDS[lang]
    boxes = "".join(
        f'<div style="flex:1 1 10rem;min-width:9rem;max-width:15rem;{_BOX}{_PLAIN}">'
        f'<div style="font-weight:600">({"abcdef"[i]}) {html.escape(DUTCH[b] if lang == "nl" else b)}</div>'
        f'<div style="color:var(--ea-ink-3);font-size:0.8rem">{html.escape(w["names"][b])}</div></div>'
        for i, b in enumerate(BASES))
    steps = "".join(f'<div style="{_BOX}{_PLAIN}">{html.escape(s)}</div>' for s in w["steps"])
    body = (f'<div style="display:flex;gap:0.5rem;flex-wrap:wrap;font-family:var(--ea-font-ui);font-size:0.85rem">'
            f'{boxes}</div>'
            f'<div style="font-family:var(--ea-font-ui);font-size:0.85rem;margin-top:0.6rem;max-width:32rem">'
            f'<div style="font-weight:600;color:var(--ea-ink);border-bottom:2px solid var(--ea-ink-3);'
            f'margin-bottom:0.3rem">(f) {html.escape(w["test"])}</div>{steps}</div>'
            f'<div style="{_UI}">{html.escape(w["key"])}</div>')
    return yours.figure(w["title"], body)


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


def _row(cells: list, wrong: bool, widths: list) -> str:
    parts = "".join(f'<div style="flex:{w};min-width:0">{c}</div>' for c, w in zip(cells, widths))
    return (f'<div style="display:flex;gap:0.6rem;align-items:baseline;{_BOX}{_WRONG if wrong else _PLAIN}">'
            f'{parts}</div>')


def _head(cells: list, widths: list) -> str:
    parts = "".join(f'<div style="flex:{w};min-width:0">{html.escape(c)}</div>' for c, w in zip(cells, widths))
    return (f'<div style="display:flex;gap:0.6rem;font-weight:600;color:var(--ea-ink);padding:0 0.5rem;'
            f'border-bottom:2px solid var(--ea-ink-3);margin-bottom:0.2rem">{parts}</div>')


def _table(head: list, rows: list, widths: list) -> str:
    return (f'<div style="font-family:var(--ea-font-ui);font-size:0.85rem;max-width:44rem">'
            f'{_head(head, widths)}{"".join(rows)}</div>')


_BASIS_WORDS = {
    "en": {"title": "Your has_basis on the made-up register",
           "head": ["activity", "basis in the register", "your has_basis"],
           "yes": "a basis of Article 6(1)", "no": "no basis of Article 6(1)", "none": "not in your dictionary",
           "key": "Each activity of REGISTER with what your has_basis gives for it.",
           "said": "It says {n} of the 7 activities name one of the six bases.",
           "wrong": "A dashed frame marks the {k} where it differs from the course's has_basis."},
    "nl": {"title": "Jouw has_basis op het verzonnen register",
           "head": ["activiteit", "grondslag in het register", "jouw has_basis"],
           "yes": "een grondslag uit artikel 6, lid 1", "no": "geen grondslag uit artikel 6, lid 1",
           "none": "niet in je dictionary",
           "key": "Elke activiteit van REGISTER met wat jouw has_basis ervoor geeft.",
           "said": "Volgens hem noemen {n} van de 7 activiteiten een van de zes grondslagen.",
           "wrong": "Een gestippeld kader wijst de {k} aan waar hij verschilt van de has_basis van de course."},
}


def basis_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _BASIS_WORDS[seen.lang]
    try:
        got = _call(fn, REGISTER, BASES)
    except Exception:
        return ""
    if not isinstance(got, dict):
        return ""
    want = has_basis(REGISTER, BASES)
    rows, n, k = [], 0, 0
    for row in REGISTER:
        a = row["activity"]
        v = got.get(a)
        said = w["none"] if a not in got else w["yes"] if v is True else w["no"] if v is False else repr(v)
        n += v is True
        wrong = v is not want[a]
        k += wrong
        rows.append(_row([html.escape(a), html.escape(_basis_name(row["basis"], seen.lang)), html.escape(said)],
                         wrong, [5, 4, 4]))
    note = w["key"] + " " + w["said"].format(n=n) + (" " + w["wrong"].format(k=k) if k else "")
    return yours.figure(w["title"], _table(w["head"], rows, [5, 4, 4]) + f'<div style="{_UI}">{html.escape(note)}</div>')


_PURPOSE_WORDS = {
    "en": {"title": "Your new_purposes on the made-up uses",
           "head": ["data", "collected for", "used for", "your new_purposes"],
           "yes": "new purpose", "no": "",
           "key": "Each row of USES; the last column marks the data your new_purposes gives back.",
           "said": "It gives back {n} of the 5.",
           "wrong": "A dashed frame marks the {k} where it differs from the course's new_purposes."},
    "nl": {"title": "Jouw new_purposes op het verzonnen gebruik",
           "head": ["data", "verzameld voor", "gebruikt voor", "jouw new_purposes"],
           "yes": "nieuw doel", "no": "",
           "key": "Elke rij van USES; de laatste kolom wijst de data aan die jouw new_purposes teruggeeft.",
           "said": "Hij geeft er {n} van de 5 terug.",
           "wrong": "Een gestippeld kader wijst de {k} aan waar hij verschilt van de new_purposes van de course."},
}


def purpose_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _PURPOSE_WORDS[seen.lang]
    try:
        got = _call(fn, USES)
    except Exception:
        return ""
    if not isinstance(got, list) or not all(isinstance(x, str) for x in got):
        return ""
    want = new_purposes(USES)
    rows, k = [], 0
    for use in USES:
        mine = use["data"] in got
        wrong = mine != (use["data"] in want)
        k += wrong
        rows.append(_row([html.escape(use[c]) for c in ("data", "collected_for", "used_for")]
                         + [html.escape(w["yes"] if mine else w["no"])], wrong, [4, 4, 4, 3]))
    n = sum(use["data"] in got for use in USES)
    note = w["key"] + " " + w["said"].format(n=n) + (" " + w["wrong"].format(k=k) if k else "")
    return yours.figure(w["title"], _table(w["head"], rows, [4, 4, 4, 3])
                        + f'<div style="{_UI}">{html.escape(note)}</div>')


_ERASE_WORDS = {
    "en": {"title": "Your erase on the forum posts, for sam",
           "kept": "kept", "gone": "taken out",
           "by": "by {a}", "names": "names {m}",
           "key": "The seven posts of POSTS, sorted by what your erase(POSTS, \"sam\") does with them.",
           "said": "It keeps {n} of the 7.",
           "wrong": "A dashed frame marks the {k} it treats differently from the course's erase."},
    "nl": {"title": "Jouw erase op de forumberichten, voor sam",
           "kept": "bewaard", "gone": "eruit gehaald",
           "by": "van {a}", "names": "noemt {m}",
           "key": "De zeven berichten van POSTS, verdeeld naar wat jouw erase(POSTS, \"sam\") ermee doet.",
           "said": "Hij bewaart er {n} van de 7.",
           "wrong": "Een gestippeld kader wijst de {k} aan die hij anders behandelt dan de erase van de course."},
}


def erase_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _ERASE_WORDS[seen.lang]
    try:
        got = _call(fn, POSTS, "sam")
    except Exception:
        return ""
    if not isinstance(got, list) or not all(isinstance(p, dict) and "id" in p for p in got):
        return ""
    ids = {p["id"] for p in got}
    want = {p["id"] for p in erase(POSTS, "sam")}
    cols, k = [], 0
    for title, keep in ((w["kept"], True), (w["gone"], False)):
        items = []
        for p in POSTS:
            if (p["id"] in ids) != keep:
                continue
            wrong = (p["id"] in want) != keep
            k += wrong
            about = w["by"].format(a=p["author"]) + (", " + w["names"].format(m=_join(p["mentions"], seen.lang))
                                                     if p["mentions"] else "")
            items.append(f'<div style="{_BOX}{_WRONG if wrong else _PLAIN}"><div>{p["id"]}. '
                         f'{html.escape(p["text"])}</div><div style="font-size:0.78rem;color:var(--ea-ink-3)">'
                         f'{html.escape(about)}</div></div>')
        cols.append(f'<div style="flex:1 1 14rem;min-width:12rem;max-width:22rem"><div style="font-weight:600;'
                    f'color:var(--ea-ink);border-bottom:2px solid var(--ea-ink-3);margin-bottom:0.3rem">'
                    f'{html.escape(title)}</div>{"".join(items)}</div>')
    note = w["key"] + " " + w["said"].format(n=len(ids & {p["id"] for p in POSTS})) + (
        " " + w["wrong"].format(k=k) if k else "")
    return yours.figure(w["title"], f'<div style="display:flex;gap:0.8rem;flex-wrap:wrap;'
                                    f'font-family:var(--ea-font-ui);font-size:0.85rem">{"".join(cols)}</div>'
                                    f'<div style="{_UI}">{html.escape(note)}</div>')


# --- the exercises ----------------------------------------------------------------------------------------------------

REQUIRED = ["basis", "purpose", "controller", "erase", "blog"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# exercise 1: has_basis
_B_HEAD = ("def has_basis(register, bases):\n"
           "    # for each activity in the register: True when its basis is one of the bases\n"
           "    result = {}\n"
           "    for row in register:\n")
_B_TAIL = "    return result\n"
_B_START = _B_HEAD + "        ...\n" + _B_TAIL
_B = _B_HEAD + '        result[row["activity"]] = row["basis"] in bases\n' + _B_TAIL
_B_NOT = _B_HEAD + '        result[row["activity"]] = row["basis"] not in bases\n' + _B_TAIL
_B_KEY = _B_HEAD + '        result[row["basis"]] = row["basis"] in bases\n' + _B_TAIL
_B_VALUE = _B_HEAD + '        result[row["activity"]] = row["basis"]\n' + _B_TAIL
_B_ONLY = _B_HEAD + '        if row["basis"] in bases:\n            result[row["activity"]] = True\n' + _B_TAIL

# exercise 2: new_purposes
_P_HEAD = ("def new_purposes(uses):\n"
           "    # the data that are used for another purpose than the one they were collected for\n"
           "    found = []\n"
           "    for use in uses:\n")
_P_TAIL = "    return found\n"
_P_START = _P_HEAD + "        ...\n" + _P_TAIL
_P_IF = '        if use["used_for"] != use["collected_for"]:\n'
_P = _P_HEAD + _P_IF + '            found.append(use["data"])\n' + _P_TAIL
_P_SAME = _P.replace("!=", "==")
_P_ROW = _P_HEAD + _P_IF + "            found.append(use)\n" + _P_TAIL
_P_PURPOSE = _P_HEAD + _P_IF + '            found.append(use["used_for"])\n' + _P_TAIL
_P_EARLY = _P_HEAD + _P_IF + '            found.append(use["data"])\n            return found\n' + _P_TAIL

# exercise 4: erase
_E_HEAD = ("def erase(posts, name):\n"
           "    # the posts without the ones by name and the ones that mention name\n"
           "    kept = []\n"
           "    for post in posts:\n")
_E_TAIL = "    return kept\n"
_E_START = _E_HEAD + "        ...\n" + _E_TAIL
_E_APPEND = "            kept.append(post)\n"
_E = _E_HEAD + '        if post["author"] != name and name not in post["mentions"]:\n' + _E_APPEND + _E_TAIL
_E_AUTHOR = _E_HEAD + '        if post["author"] != name:\n' + _E_APPEND + _E_TAIL
_E_MENTIONS = _E_HEAD + '        if name not in post["mentions"]:\n' + _E_APPEND + _E_TAIL
_E_OR = _E_HEAD + '        if post["author"] != name or name not in post["mentions"]:\n' + _E_APPEND + _E_TAIL
_E_GONE = _E_HEAD + '        if post["author"] == name or name in post["mentions"]:\n' + _E_APPEND + _E_TAIL

_TINY = [{"id": 1, "author": "a", "mentions": ["a"], "text": "hi"},
         {"id": 2, "author": "b", "mentions": [], "text": "yes"},
         {"id": 3, "author": "c", "mentions": ["b", "a"], "text": "no"}]


def show_posts(v):
    """Posts as the check shows them: by their ids, such as `the posts with id 3, 5 and 7`, so that a message stays
    short; anything else the way the lessons write it in code."""
    if isinstance(v, list) and all(isinstance(p, dict) and "id" in p for p in v):
        if not v:
            return {"en": "no posts at all", "nl": "helemaal geen berichten"}
        ids = [str(p["id"]) for p in v]
        one = len(ids) == 1
        return {"en": ("the post with id " if one else "the posts with id ") + _join(ids, "en"),
                "nl": ("het bericht met id " if one else "de berichten met id ") + _join(ids, "nl")}
    from everything_ai import check as checking
    return checking._as_value(v)


EXERCISES = {
    "basis": FunctionBox(
        name="basis",
        picture=basis_picture,
        function="has_basis",
        start=_B_START,
        solution=_B,
        visible=[(REGISTER, BASES)],
        hidden=[(REGISTER, ["consent", "contract"]),
                ([{"activity": "a", "basis": "x"}, {"activity": "b", "basis": "y"}], ["y"]),
                ([], BASES)],
        first=[_DOTS],
        mistakes=[
            Mistake("not", wrong=_B_NOT,
                    en="`{call}` gives {got}, and it should give {want}. Your dictionary gives `True` to the activities whose basis is not in `bases`, and `False` to the others: the other way round.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je dictionary geeft `True` aan de activiteiten waarvan de grondslag niet in `bases` staat, en `False` aan de andere: andersom."),
            Mistake("key", wrong=_B_KEY,
                    en="`{call}` gives {got}, and it should give {want}. The keys of your dictionary are the bases; they should be the activities: `result[row[\"activity\"]]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. De keys van je dictionary zijn de grondslagen; dat moeten de activiteiten zijn: `result[row[\"activity\"]]`."),
            Mistake("value", wrong=_B_VALUE,
                    en="`{call}` gives {got}, and it should give {want}. Your dictionary gives each activity the basis from the register; it should give `True` or `False`: whether `row[\"basis\"]` is in `bases`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je dictionary geeft elke activiteit de grondslag uit het register; hij moet `True` of `False` geven: of `row[\"basis\"]` in `bases` staat."),
            Mistake("only", wrong=_B_ONLY,
                    en="`{call}` gives {got}, and it should give {want}. Your dictionary leaves out the activities without a basis: they belong in it too, with `False`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je dictionary laat de activiteiten zonder grondslag weg: die horen er ook in, met `False`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with one line that stores in `result`, under `row[\"activity\"]`, whether `row[\"basis\"]` is in `bases`.",
                      "nl": "Vervang de `...` door één regel die in `result`, onder `row[\"activity\"]`, bewaart of `row[\"basis\"]` in `bases` staat."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. For each row, store under its activity whether its basis is in `bases`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Bewaar voor elke rij onder de activiteit of de grondslag in `bases` staat."},
            "almost": {"en": "Right for `has_basis(REGISTER, BASES)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `register` and the `bases` it gets?",
                       "nl": "Goed voor `has_basis(REGISTER, BASES)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie het `register` en de `bases` die hij krijgt?"},
            "correct": {"en": "Your `has_basis` is right in all {n} cases. The picture puts what it says beside each activity of the register.",
                        "nl": "Je `has_basis` klopt in alle {n} gevallen. Het plaatje zet wat hij zegt naast elke activiteit van het register."},
        },
    ),
    "purpose": FunctionBox(
        name="purpose",
        picture=purpose_picture,
        function="new_purposes",
        start=_P_START,
        solution=_P,
        visible=[(USES,)],
        hidden=[([{"data": "a", "collected_for": "x", "used_for": "y"},
                  {"data": "b", "collected_for": "z", "used_for": "z"},
                  {"data": "c", "collected_for": "x", "used_for": "z"}],),
                ([],)],
        first=[_DOTS],
        mistakes=[
            Mistake("same", wrong=_P_SAME,
                    en="`{call}` gives {got}, and it should give {want}. Your list holds the data used for the purpose they were collected for; you need the others: `!=`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je lijst bevat de data die gebruikt worden voor het doel waarvoor ze verzameld zijn; je hebt de andere nodig: `!=`."),
            Mistake("row", wrong=_P_ROW,
                    en="`{call}` gives a list of whole rows, and it should give {want}. Add only which data it is: `use[\"data\"]`.",
                    nl="`{call}` geeft een lijst met hele rijen, en dat moet {want} zijn. Voeg alleen toe om welke data het gaat: `use[\"data\"]`."),
            Mistake("purpose", wrong=_P_PURPOSE,
                    en="`{call}` gives {got}, and it should give {want}. Your list holds the new purposes; it should hold the data that have them: `use[\"data\"]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je lijst bevat de nieuwe doelen; hij moet de data bevatten die ze hebben: `use[\"data\"]`."),
            Mistake("early", wrong=_P_EARLY,
                    en="`{call}` gives {got}, and it should give {want}. Your function stops at the first data with a new purpose, but there can be more. Let the loop go over all uses, and give back `found` after it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie stopt bij de eerste data met een nieuw doel, maar er kunnen er meer zijn. Laat de loop alle rijen langsgaan, en geef `found` daarna terug."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with two lines: when `use[\"used_for\"]` differs from `use[\"collected_for\"]`, add `use[\"data\"]` to `found`.",
                      "nl": "Vervang de `...` door twee regels: als `use[\"used_for\"]` verschilt van `use[\"collected_for\"]`, voeg `use[\"data\"]` dan toe aan `found`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Go over the uses, and add the data of each use whose purpose changed.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Ga de rijen langs, en voeg de data toe van elke rij waarvan het doel veranderd is."},
            "almost": {"en": "Right for `new_purposes(USES)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `uses` it gets?",
                       "nl": "Goed voor `new_purposes(USES)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `uses` die hij krijgt?"},
            "correct": {"en": "Your `new_purposes` is right in all {n} cases. The picture marks the data it gives back.",
                        "nl": "Je `new_purposes` klopt in alle {n} gevallen. Het plaatje wijst de data aan die hij teruggeeft."},
        },
    ),
    "erase": FunctionBox(
        name="erase",
        picture=erase_picture,
        function="erase",
        start=_E_START,
        solution=_E,
        show=show_posts,
        visible=[(POSTS, "sam")],
        hidden=[(POSTS, "kai"), (POSTS, "nobody"), (_TINY, "a"), (_TINY, "b")],
        first=[_DOTS],
        mistakes=[
            Mistake("author", wrong=_E_AUTHOR,
                    en="`{call}` gives {got}, and it should give {want}. Your function takes out the posts by `name`, but keeps the posts by others that mention `name`. Keep a post only when it is not by `name` and its `\"mentions\"` do not hold `name` either.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie haalt de berichten van `name` eruit, maar bewaart de berichten van anderen die `name` noemen. Bewaar een bericht alleen als het niet van `name` is en de `\"mentions\"` `name` ook niet bevatten."),
            Mistake("mentions", wrong=_E_MENTIONS,
                    en="`{call}` gives {got}, and it should give {want}. Your function takes out the posts that mention `name`, but keeps the posts by `name`. Keep a post only when its `\"author\"` is not `name` either.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie haalt de berichten eruit die `name` noemen, maar bewaart de berichten van `name` zelf. Bewaar een bericht alleen als de `\"author\"` ook niet `name` is."),
            Mistake("or", wrong=_E_OR,
                    en="`{call}` gives {got}, and it should give {want}. Your function keeps a post as soon as one of the two holds: it is not by `name`, or it does not mention `name`. A post stays only when both hold at once. That is `and`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie bewaart een bericht zodra een van de twee waar is: het is niet van `name`, of het noemt `name` niet. Een bericht blijft alleen als allebei tegelijk waar is. Dat is `and`."),
            Mistake("gone", wrong=_E_GONE,
                    en="`{call}` gives {got}, and it should give {want}. Your list holds the posts by or about `name`: the ones to take out. Keep the others.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je lijst bevat de berichten van of over `name`: die moeten er juist uit. Bewaar de andere."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with two lines: when `post[\"author\"]` is not `name` and `name` is not in `post[\"mentions\"]`, add `post` to `kept`.",
                      "nl": "Vervang de `...` door twee regels: als `post[\"author\"]` niet `name` is en `name` niet in `post[\"mentions\"]` staat, voeg `post` dan toe aan `kept`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Keep each post that is not by `name` and does not mention `name`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Bewaar elk bericht dat niet van `name` is en `name` niet noemt."},
            "almost": {"en": "Right for `erase(POSTS, \"sam\")`. But `{call}` gives {got}, and it should give {want}. Does your function use the `posts` and the `name` it gets?",
                       "nl": "Goed voor `erase(POSTS, \"sam\")`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `posts` en de `name` die hij krijgt?"},
            "correct": {"en": "Your `erase` is right in all {n} cases. The picture sorts the seven posts into the ones it keeps and the ones it takes out.",
                        "nl": "Je `erase` klopt in alle {n} gevallen. Het plaatje verdeelt de zeven berichten in de berichten die hij bewaart en die hij eruit haalt."},
        },
    ),
}

CASES = [
    ("basis", _B_START, "empty", None),
    ("basis", _B_HEAD + "        result[row[\"activity\"]] = ...\n" + _B_TAIL, "wrong", "dots-left"),
    ("basis", _B, "correct", None),
    ("basis", _B_HEAD + ('        if row["basis"] in bases:\n'
                         '            result[row["activity"]] = True\n'
                         '        else:\n'
                         '            result[row["activity"]] = False\n') + _B_TAIL, "correct", None),
    ("basis", _B_HEAD + '        result[row["activity"]] = bases.count(row["basis"]) > 0\n' + _B_TAIL, "correct", None),
    ("basis", _B_NOT, "wrong", "not"),
    ("basis", _B_KEY, "wrong", "key"),
    ("basis", _B_VALUE, "wrong", "value"),
    ("basis", _B_ONLY, "wrong", "only"),
    ("basis", _B_HEAD + '        result[row["activity"]] = True\n' + _B_TAIL, "wrong", None),
    ("purpose", _P_START, "empty", None),
    ("purpose", _P_HEAD + _P_IF + "            ...\n" + _P_TAIL, "wrong", "dots-left"),
    ("purpose", _P, "correct", None),
    ("purpose", _P_HEAD + ('        if use["collected_for"] == use["used_for"]:\n'
                           '            continue\n'
                           '        found = found + [use["data"]]\n') + _P_TAIL, "correct", None),
    ("purpose", _P_SAME, "wrong", "same"),
    ("purpose", _P_ROW, "wrong", "row"),
    ("purpose", _P_PURPOSE, "wrong", "purpose"),
    ("purpose", _P_EARLY, "wrong", "early"),
    ("purpose", _P_HEAD + _P_IF.replace("used_for", "data", 1) + '            found.append(use["data"])\n' + _P_TAIL,
     "wrong", None),
    ("erase", _E_START, "empty", None),
    ("erase", _E_HEAD + '        if post["author"] != name:\n            ...\n' + _E_TAIL, "wrong", "dots-left"),
    ("erase", _E, "correct", None),
    ("erase", _E_HEAD + ('        if post["author"] == name or name in post["mentions"]:\n'
                         '            continue\n'
                         '        kept.append(post)\n') + _E_TAIL, "correct", None),
    ("erase", _E_HEAD + ('        if not (post["author"] == name or name in post["mentions"]):\n'
                         + _E_APPEND) + _E_TAIL, "correct", None),
    ("erase", _E_AUTHOR, "wrong", "author"),
    ("erase", _E_MENTIONS, "wrong", "mentions"),
    ("erase", _E_OR, "wrong", "or"),
    ("erase", _E_GONE, "wrong", "gone"),
    ("erase", _E_HEAD + '        kept.append(post["text"])\n' + _E_TAIL, "wrong", None),
]
