# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-what-is-privacy: Solove's taxonomy of privacy problems (TAXONOMY), three made-up tables
about the same made-up people (TABLES) and a made-up survey (SURVEY), and the three functions the learner writes:
group_of (the group of a kind of privacy problem), facts (what the tables say together about one person) and
share_who_shared (of the people who say they are concerned, the share who shared their data); with the pictures of
what the learner's functions give."""
import html
import re

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake


class NamedDict(dict):
    """A dict the check calls by its name, such as TAXONOMY, so that a message stays short."""

    def __init__(self, items, name: str):
        super().__init__(items)
        self.name = name

    def __repr__(self):
        return self.name


class NamedList(list):
    """A list the check calls by its name, such as TABLES."""

    def __init__(self, items, name: str):
        super().__init__(items)
        self.name = name

    def __repr__(self):
        return self.name


# --- Solove's taxonomy (2006): four groups of activities that cause privacy problems, sixteen kinds ------------------
TAXONOMY = NamedDict({
    "collection": ["surveillance", "interrogation"],
    "processing": ["aggregation", "identification", "insecurity", "secondary use", "exclusion"],
    "dissemination": ["breach of confidentiality", "disclosure", "exposure", "increased accessibility", "blackmail",
                      "appropriation", "distortion"],
    "invasion": ["intrusion", "decisional interference"],
}, "TAXONOMY")
KINDS = [k for kinds in TAXONOMY.values() for k in kinds]
DUTCH = {
    "collection": "verzamelen", "processing": "verwerken", "dissemination": "verspreiden", "invasion": "binnendringen",
    "surveillance": "observeren", "interrogation": "ondervragen",
    "aggregation": "aggregatie", "identification": "identificeren", "insecurity": "onveilig bewaren",
    "secondary use": "hergebruik voor een ander doel", "exclusion": "buitensluiten",
    "breach of confidentiality": "vertrouwelijkheid schenden", "disclosure": "onthullen", "exposure": "te kijk zetten",
    "increased accessibility": "toegankelijker maken", "blackmail": "chanteren", "appropriation": "identiteit misbruiken",
    "distortion": "vertekenen",
    "intrusion": "rust verstoren", "decisional interference": "ingrijpen in beslissingen",
}

# --- three made-up tables of three made-up apps, about the same made-up people, by e-mail address --------------------
RUNS = {"sam@example.org": "runs from Harbor Street at 6:30 on weekdays",
        "noor@example.org": "runs in the dunes on Sunday mornings",
        "kai@example.org": "runs around the lake after work",
        "ash@example.org": "runs to the station at 7:15 on weekdays"}
PHARMACY = {"sam@example.org": "buys allergy tablets every April",
            "lior@example.org": "buys reading glasses every two years",
            "kai@example.org": "buys bandages and sports tape every month",
            "ash@example.org": "buys sleeping tablets every week"}
FORUM = {"sam@example.org": "writes about night shifts at the hospital",
         "noor@example.org": "writes about moving house in May",
         "remi@example.org": "writes about learning to drive",
         "ash@example.org": "writes about looking for a new job"}
TABLES = NamedList([RUNS, PHARMACY, FORUM], "TABLES")
APPS = {"en": ["running app", "pharmacy card", "forum"], "nl": ["hardloopapp", "apotheekpas", "forum"]}
PEOPLE = ["sam@example.org", "noor@example.org", "kai@example.org", "lior@example.org", "ash@example.org",
          "remi@example.org"]

# --- a made-up survey: how concerned each of 20 people says they are about privacy (1 to 5), and whether they signed
# up for a store card with their name and address for a discount -------------------------------------------------------
_ANSWERS = [(5, True), (3, True), (2, True), (4, True), (1, True), (4, False), (3, False), (5, True), (2, True),
            (3, True), (4, True), (1, True), (2, False), (5, True), (3, True), (4, True), (1, True), (2, True),
            (5, True), (3, True)]
SURVEY = NamedList([{"concern": c, "shared": s} for c, s in _ANSWERS], "SURVEY")


# --- the course's own versions of the three functions ---------------------------------------------------------------

def group_of(kind, taxonomy):
    """The group of the taxonomy that a kind of privacy problem belongs to, or None."""
    for group, kinds in taxonomy.items():
        if kind in kinds:
            return group
    return None


def facts(key, tables):
    """What all tables together say about one person."""
    found = []
    for table in tables:
        if key in table:
            found.append(table[key])
    return found


def share_who_shared(people, least):
    """Of the people whose concern is at least `least`, the share who shared their data."""
    count = 0
    shared = 0
    for person in people:
        if person["concern"] >= least:
            count = count + 1
            if person["shared"]:
                shared = shared + 1
    return shared / count


def share_below(people, least):
    """Of the people whose concern is below `least`, the share who shared their data."""
    below = [p for p in people if p["concern"] < least]
    return sum(1 for p in below if p["shared"]) / len(below)


# --- the pictures ---------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"


def number(x: float, lang: str, digits: int = 3) -> str:
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def _name(key: str, lang: str) -> str:
    return key if lang == "en" else f"{DUTCH[key]} ({key})"


def _columns(placed: dict, lang: str, wrong: set = frozenset(), extra: list | None = None) -> str:
    """Solove's four groups as columns, each with the kinds placed in it; a kind in `wrong` in the colour that counts
    mistakes, with a dashed frame. `extra` is a fifth column for kinds placed nowhere."""
    cols = []
    groups = list(TAXONOMY) + (["none"] if extra else [])
    for g in groups:
        kinds = extra if g == "none" else placed.get(g, [])
        title = {"en": "no group", "nl": "geen groep"}[lang] if g == "none" else _name(g, lang)
        items = "".join(
            f'<div style="padding:0.15rem 0.4rem;margin:0.15rem 0;border-radius:4px;'
            + ('border:1.5px dashed var(--ea-loss);color:var(--ea-loss-text)' if k in wrong
               else 'border:1px solid var(--ea-line);color:var(--ea-ink)')
            + f'">{html.escape(_name(k, lang))}</div>' for k in kinds)
        cols.append(f'<div style="flex:1 1 9rem;min-width:8rem;max-width:14rem"><div style="font-weight:600;color:var(--ea-ink);'
                    f'border-bottom:2px solid var(--ea-ink-3);margin-bottom:0.3rem">{html.escape(title)}</div>'
                    f'{items}</div>')
    return (f'<div style="display:flex;gap:0.6rem;flex-wrap:wrap;font-family:var(--ea-font-ui);font-size:0.85rem">'
            f'{"".join(cols)}</div>')


def taxonomy_picture(lang: str) -> str:
    """The course's figure of Solove's taxonomy, for The idea."""
    words = {"en": {"title": "Solove's taxonomy of privacy problems (2006)",
                    "key": "Four groups of activities, sixteen kinds. Collection, processing and dissemination move "
                           "information further from the control of the person it is about; an invasion comes toward the person, and "
                           "need not involve information."},
             "nl": {"title": "De taxonomie van privacyproblemen van Solove (2006)",
                    "key": "Vier groepen activiteiten, zestien soorten. Verzamelen, verwerken en verspreiden brengen "
                           "informatie verder buiten de zeggenschap van de persoon over wie ze gaat; binnendringen komt naar de persoon "
                           "toe, en hoeft niet over informatie te gaan."}}[lang]
    return yours.figure(words["title"], _columns(dict(TAXONOMY), lang)
                        + f'<div style="{_UI}">{html.escape(words["key"])}</div>')


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


_GROUP_WORDS = {
    "en": {"title": "Your group_of on Solove's sixteen kinds",
           "key": "Each kind in the column of the group your group_of gives for it; under no group, the kinds for "
                  "which it gives no group of the taxonomy.",
           "said": "Your group_of puts {n} of the 16 kinds in Solove's group.",
           "wrong": "A dashed frame marks the {k} it puts elsewhere."},
    "nl": {"title": "Jouw group_of op de zestien soorten van Solove",
           "key": "Elke soort in de kolom van de groep die jouw group_of ervoor geeft; onder geen groep de soorten "
                  "waarvoor hij geen groep van de taxonomie geeft.",
           "said": "Jouw group_of zet {n} van de 16 soorten in de groep van Solove.",
           "wrong": "Een gestippeld kader wijst de {k} aan die hij ergens anders zet."},
}


def group_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _GROUP_WORDS[seen.lang]
    placed, nowhere, wrong = {}, [], set()
    for kind in KINDS:
        try:
            g = _call(fn, kind, TAXONOMY)
        except Exception:
            return ""
        if isinstance(g, str) and g in TAXONOMY:
            placed.setdefault(g, []).append(kind)
        else:
            nowhere.append(kind)
        if not (isinstance(g, str) and g == group_of(kind, TAXONOMY)):
            wrong.add(kind)
    n = len(KINDS) - len(wrong)
    note = w["key"] + " " + w["said"].format(n=n) + (" " + w["wrong"].format(k=len(wrong)) if wrong else "")
    return yours.figure(w["title"], _columns(placed, seen.lang, wrong, nowhere or None)
                        + f'<div style="{_UI}">{html.escape(note)}</div>')


def _rows_svg(rows: list, max_n: int, alt: str, lang: str, digits: int | None = None, x_label: str = "",
              ticks: list | None = None) -> str:
    """Bars from 0, one per row (label, value, role), each with its value."""
    from everything_ai import chart
    n = len(rows)
    ticks = ticks or list(range(0, max_n + 1))
    panel = chart.Panel(x=(0, max_n), y=(0, 1.05 * n), xlabel=x_label, yticks=[], xticks=ticks,
                        xtick_labels=[f"{t:g}" for t in ticks])
    for i, (label, value, role) in enumerate(rows):
        y = 1.05 * (n - 1 - i) + 0.1
        shown = str(value) if digits is None else number(value, lang, digits)
        panel.text(0.002 * max_n, y + 0.62, f"{label}: {shown}", "ink", anchor="start")
        if value > 0:
            panel.rect(0, y, min(value, max_n), y + 0.36, role=role, solid=True)
    svg = chart.figure([panel], alt=alt, width=520, height=70 + 56 * n)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


_FACTS_WORDS = {
    "en": {"title": "Your facts on the three tables",
           "key": "For each of the six made-up people: how many facts your facts finds about them in the three tables "
                  "together. Each table alone holds at most one fact per person.",
           "sam": "About sam@example.org, it finds: {facts}.",
           "right": "The course's facts finds about sam@example.org: {facts}.",
           "xlabel": "facts found",
           "alt": "Your facts finds {n} facts about the six people."},
    "nl": {"title": "Jouw facts op de drie tabellen",
           "key": "Voor elk van de zes verzonnen mensen: hoeveel feiten jouw facts over die persoon vindt in de drie "
                  "tabellen samen. Elke tabel apart bevat hooguit één feit per persoon.",
           "sam": "Over sam@example.org vindt hij: {facts}.",
           "right": "De facts van de course vindt over sam@example.org: {facts}.",
           "xlabel": "gevonden feiten",
           "alt": "Jouw facts vindt {n} feiten over de zes mensen."},
}


def facts_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _FACTS_WORDS[seen.lang]
    got = {}
    for key in PEOPLE:
        try:
            v = _call(fn, key, TABLES)
        except Exception:
            return ""
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            return ""
        got[key] = v
    total = sum(len(v) for v in got.values())
    rows = [(key.split("@")[0], len(got[key]), "pred") for key in PEOPLE]
    svg = _rows_svg(rows, 3, w["alt"].format(n=total), seen.lang, x_label=w["xlabel"])
    note = w["key"] + " " + w["sam"].format(facts="; ".join(got["sam@example.org"]) or "—")
    if got["sam@example.org"] != facts("sam@example.org", TABLES):
        note += " " + w["right"].format(facts="; ".join(facts("sam@example.org", TABLES)))
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(note)}</div>')


_SHARE_WORDS = {
    "en": {"title": "Your share_who_shared on the made-up survey",
           "rows": ["concern {least} or more: share who shared (yours)", "concern below {least}: share who shared"],
           "xlabel": "share who signed up for the store card",
           "key": "The course's made-up survey of 20 people, with least = {least}. The second bar is the course's.",
           "right": "The right share for concern {least} or more: {want}.",
           "alt": "Your share: {got}; below {least}: {below}."},
    "nl": {"title": "Jouw share_who_shared op de verzonnen enquête",
           "rows": ["zorgscore {least} of meer: aandeel aangemeld (jouw functie)", "zorgscore onder {least}: aandeel aangemeld"],
           "xlabel": "deel dat zich aanmeldde voor de klantenkaart",
           "key": "De verzonnen enquête van de course onder 20 mensen, met least = {least}. De tweede balk is die van "
                  "de course.",
           "right": "Het goede aandeel bij zorgscore {least} of meer: {want}.",
           "alt": "Jouw aandeel: {got}; onder {least}: {below}."},
}


def share_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _SHARE_WORDS[seen.lang]
    least = 4
    try:
        v = _call(fn, SURVEY, least)
    except Exception:
        return ""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1:
        return ""
    v = float(v)
    want, below = share_who_shared(SURVEY, least), share_below(SURVEY, least)
    rows = [(w["rows"][0].format(least=least), v, "pred"), (w["rows"][1].format(least=least), below, "data")]
    svg = _rows_svg(rows, 1, w["alt"].format(got=number(v, seen.lang), least=least, below=number(below, seen.lang)),
                    seen.lang, digits=3, x_label=w["xlabel"], ticks=[0, 0.25, 0.5, 0.75, 1])
    note = w["key"].format(least=least)
    if abs(v - want) > 1e-9:
        note += " " + w["right"].format(least=least, want=number(want, seen.lang))
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(note)}</div>')


# --- the exercises ----------------------------------------------------------------------------------------------------

REQUIRED = ["group", "facts", "views", "share", "headline"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# exercise 1: group_of
_G_HEAD = ("def group_of(kind, taxonomy):\n"
           "    # the group of the taxonomy that a kind of privacy problem belongs to, or None\n"
           "    for group, kinds in taxonomy.items():\n")
_G_TAIL = "    return None\n"
_G_START = _G_HEAD + "        ...\n" + _G_TAIL
_G = _G_HEAD + "        if kind in kinds:\n            return group\n" + _G_TAIL
_G_FIRST = _G_HEAD + "        return group\n" + _G_TAIL
_G_KINDS = _G_HEAD + "        if kind in kinds:\n            return kinds\n" + _G_TAIL
_G_ELSE = _G_HEAD + "        if kind in kinds:\n            return group\n        else:\n            return None\n" + _G_TAIL

# exercise 2: facts
_F_HEAD = ("def facts(key, tables):\n"
           "    # what all tables together say about one person\n"
           "    found = []\n"
           "    for table in tables:\n")
_F_TAIL = "    return found\n"
_F_START = _F_HEAD + "        ...\n" + _F_TAIL
_F = _F_HEAD + "        if key in table:\n            found.append(table[key])\n" + _F_TAIL
_F_EARLY = _F_HEAD + "        if key in table:\n            found.append(table[key])\n            return found\n" + _F_TAIL
_F_TABLE = _F_HEAD + "        if key in table:\n            found.append(table)\n" + _F_TAIL
_F_KEY = _F_HEAD + "        if key in table:\n            found.append(key)\n" + _F_TAIL

# exercise 3: share_who_shared
_S_HEAD = ("def share_who_shared(people, least):\n"
           "    # of the people with a concern of least or more, the share who shared their data\n"
           "    count = 0\n"
           "    shared = 0\n"
           "    for person in people:\n")
_S_TAIL = "    return shared / count\n"
_S_START = _S_HEAD + "        ...\n" + _S_TAIL
_S_BODY = ('        if person["concern"] >= least:\n'
           '            count = count + 1\n'
           '            if person["shared"]:\n'
           '                shared = shared + 1\n')
_S = _S_HEAD + _S_BODY + _S_TAIL
_S_ALL = _S_HEAD + ('        count = count + 1\n'
                    '        if person["shared"]:\n'
                    '            shared = shared + 1\n') + _S_TAIL
_S_STRICT = _S.replace('person["concern"] >= least', 'person["concern"] > least')
_S_SWAPPED = _S_HEAD + ('        if person["shared"]:\n'
                        '            count = count + 1\n'
                        '            if person["concern"] >= least:\n'
                        '                shared = shared + 1\n') + _S_TAIL
_S_COUNT = _S.replace("    return shared / count\n", "    return shared\n")

_MINI = {"a": ["x"], "b": ["y", "z"]}


def show_share(v):
    """A share as the check shows it: three decimals, so that 0.875 stays 0.875 and 2/3 is 0.667."""
    if isinstance(v, float):
        return {"en": f"{v:.3f}", "nl": f"{v:.3f}".replace(".", ",")}
    return {"en": repr(v), "nl": repr(v)}


EXERCISES = {
    "group": FunctionBox(
        name="group",
        picture=group_picture,
        function="group_of",
        start=_G_START,
        solution=_G,
        visible=[("aggregation", TAXONOMY)],
        hidden=[("surveillance", TAXONOMY), ("decisional interference", TAXONOMY), ("profiling", TAXONOMY),
                ("z", _MINI), ("w", _MINI)],
        first=[_DOTS],
        mistakes=[
            Mistake("first", wrong=_G_FIRST,
                    en="`{call}` gives {got}, and it should give {want}. Your function gives the first group of the dictionary, whatever the kind. Give back a group only when `kind in kinds`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie geeft de eerste groep van de dictionary, welke soort het ook is. Geef een groep alleen terug als `kind in kinds`."),
            Mistake("kinds", wrong=_G_KINDS,
                    en="`{call}` gives {got}, and it should give {want}. Your function gives the list of kinds of the group; give the name of the group: `return group`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie geeft de lijst met soorten van de groep; geef de naam van de groep: `return group`."),
        ],
        texts=_SAME_TEXTS | {
            "unreached": {"en": "With `{call}`, your function gives back `None`: it does not give the group that has the kind. Does your `if` give back `group` when `kind in kinds`? A `return` inside the loop ends the function at once, so whether no group has the kind is clear only after the loop, where the `return None` already is.",
                          "nl": "Met `{call}` geeft je functie `None` terug: hij geeft niet de groep die de soort heeft. Geeft je `if` de `group` terug als `kind in kinds`? Een `return` in de loop beëindigt de functie meteen, dus of geen enkele groep de soort heeft, weet je pas na de loop, waar de `return None` al staat."},
            "empty": {"en": "Replace the `...` with two lines: when `kind` is in `kinds`, give back `group`.",
                      "nl": "Vervang de `...` door twee regels: als `kind` in `kinds` staat, geef dan `group` terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Go over the groups, and give back the name of the group whose list holds `kind`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Ga de groepen langs, en geef de naam terug van de groep waarvan de lijst `kind` bevat."},
            "almost": {"en": "Right for `group_of(\"aggregation\", TAXONOMY)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `kind` and the `taxonomy` it gets?",
                       "nl": "Goed voor `group_of(\"aggregation\", TAXONOMY)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `kind` en de `taxonomy` die hij krijgt?"},
            "correct": {"en": "Your `group_of` is right in all {n} cases. The picture puts Solove's sixteen kinds in the groups your function gives.",
                        "nl": "Je `group_of` klopt in alle {n} gevallen. Het plaatje zet de zestien soorten van Solove in de groepen die je functie geeft."},
        },
    ),
    "facts": FunctionBox(
        name="facts",
        picture=facts_picture,
        function="facts",
        start=_F_START,
        solution=_F,
        visible=[("sam@example.org", TABLES)],
        hidden=[("noor@example.org", TABLES), ("lior@example.org", TABLES), ("nobody@example.org", TABLES),
                ("a", [{"a": "x"}, {"b": "y"}, {"a": "z"}])],
        first=[_DOTS],
        mistakes=[
            Mistake("early", wrong=_F_EARLY,
                    en="`{call}` gives {got}, and it should give {want}. Your function gives only the first fact it finds, but a person can be in more than one table. Give back `found` after the loop.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie geeft alleen het eerste feit dat hij vindt, maar een persoon kan in meer dan één tabel staan. Geef `found` na de loop terug."),
            Mistake("table", wrong=_F_TABLE,
                    en="`{call}` gives a list of whole tables, and it should give {want}. Add only what the table says about `key`: `table[key]`.",
                    nl="`{call}` geeft een lijst met hele tabellen, en dat moet {want} zijn. Voeg alleen toe wat de tabel over `key` zegt: `table[key]`."),
            Mistake("key", wrong=_F_KEY,
                    en="`{call}` gives {got}, and it should give {want}. Your list holds the email address once for each table it is in; add what the table says about it: `table[key]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je lijst bevat het e-mailadres één keer voor elke tabel waarin het staat; voeg toe wat de tabel erover zegt: `table[key]`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with two lines: when `key` is in `table`, add `table[key]` to `found`.",
                      "nl": "Vervang de `...` door twee regels: als `key` in `table` staat, voeg `table[key]` dan toe aan `found`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Go over the tables, and add what each table that has `key` says about it.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Ga de tabellen langs, en voeg toe wat elke tabel die `key` heeft erover zegt."},
            "almost": {"en": "Right for `facts(\"sam@example.org\", TABLES)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `key` and the `tables` it gets?",
                       "nl": "Goed voor `facts(\"sam@example.org\", TABLES)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `key` en de `tables` die hij krijgt?"},
            "correct": {"en": "Your `facts` is right in all {n} cases. The picture counts what it finds about each of the six people.",
                        "nl": "Je `facts` klopt in alle {n} gevallen. Het plaatje telt wat hij vindt over elk van de zes mensen."},
        },
    ),
    "share": FunctionBox(
        name="share",
        picture=share_picture,
        function="share_who_shared",
        start=_S_START,
        solution=_S,
        show=show_share,
        visible=[(SURVEY, 4)],
        hidden=[(SURVEY, 5), (SURVEY, 2),
                ([{"concern": 4, "shared": False}, {"concern": 2, "shared": True}, {"concern": 5, "shared": True}], 4)],
        first=[_DOTS],
        mistakes=[
            Mistake("all", wrong=_S_ALL,
                    en="`{call}` gives {got}, and it should give {want}. Your function gives the share of all people who shared; count only the people whose `\"concern\"` is at least `least`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie geeft het aandeel van alle mensen dat zich aanmeldde; tel alleen de mensen van wie de `\"concern\"` minstens `least` is."),
            Mistake("strict", wrong=_S_STRICT,
                    en="`{call}` gives {got}, and it should give {want}. Your function leaves out the people whose concern is exactly `least`: at least means `>=`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie laat de mensen weg van wie de zorgscore precies `least` is: minstens is `>=`."),
            Mistake("swapped", wrong=_S_SWAPPED,
                    en="`{call}` gives {got}, and it should give {want}. Your function gives, of the people who shared, the share who are concerned. The question is the other way round: of the concerned people, the share who shared.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie geeft van de mensen die zich aanmeldden het aandeel dat zich zorgen maakt. De vraag is andersom: van de mensen die zich zorgen maken, het aandeel dat zich aanmeldde."),
            Mistake("count", wrong=_S_COUNT,
                    en="`{call}` gives {got}, and it should give {want}. You give back how many of them shared; give back the share: `shared / count`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je geeft terug hoeveel van hen zich aanmeldden; geef het aandeel terug: `shared / count`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with four lines: when `person[\"concern\"]` is at least `least`, add 1 to `count`, and when that person also shared, add 1 to `shared`.",
                      "nl": "Vervang de `...` door vier regels: als `person[\"concern\"]` minstens `least` is, tel dan 1 op bij `count`, en als die persoon zich ook aanmeldde, 1 bij `shared`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Count in `count` the people whose concern is at least `least`, and in `shared` those of them who shared.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Tel in `count` de mensen van wie de zorgscore minstens `least` is, en in `shared` wie van hen zich aanmeldde."},
            "almost": {"en": "Right for `share_who_shared(SURVEY, 4)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `people` and the `least` it gets?",
                       "nl": "Goed voor `share_who_shared(SURVEY, 4)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `people` en de `least` die hij krijgt?"},
            "correct": {"en": "Your `share_who_shared` is right in all {n} cases. The picture puts it beside the share among the less concerned.",
                        "nl": "Je `share_who_shared` klopt in alle {n} gevallen. Het plaatje zet hem naast het deel bij de mensen die zich minder zorgen maken."},
        },
    ),
}

CASES = [
    ("group", _G_START, "empty", None),
    ("group", _G_HEAD + "        if kind in kinds:\n            ...\n" + _G_TAIL, "wrong", "dots-left"),
    ("group", _G, "correct", None),
    ("group", _G_HEAD + "        for k in kinds:\n            if k == kind:\n                return group\n" + _G_TAIL,
     "correct", None),
    ("group", _G_HEAD + "        if kinds.count(kind) > 0:\n            return group\n" + _G_TAIL, "correct", None),
    ("group", _G_FIRST, "wrong", "first"),
    ("group", _G_KINDS, "wrong", "kinds"),
    ("group", _G_ELSE, "wrong", "unreached"),
    ("group", _G_HEAD + "        if kind not in kinds:\n            return group\n" + _G_TAIL, "wrong", None),
    ("facts", _F_START, "empty", None),
    ("facts", _F_HEAD + "        if key in table:\n            ...\n" + _F_TAIL, "wrong", "dots-left"),
    ("facts", _F, "correct", None),
    ("facts", _F_HEAD + "        if key in table:\n            found = found + [table[key]]\n" + _F_TAIL, "correct", None),
    ("facts", _F_HEAD + "        if key not in table:\n            continue\n        found.append(table[key])\n" + _F_TAIL,
     "correct", None),
    ("facts", _F_EARLY, "wrong", "early"),
    ("facts", _F_TABLE, "wrong", "table"),
    ("facts", _F_KEY, "wrong", "key"),
    ("facts", _F_HEAD + "        if key not in table:\n            found.append(key)\n" + _F_TAIL, "wrong", None),
    ("share", _S_START, "empty", None),
    ("share", _S_HEAD + '        if person["concern"] >= least:\n            ...\n' + _S_TAIL, "wrong", "dots-left"),
    ("share", _S, "correct", None),
    ("share", _S_HEAD + ('        if least <= person["concern"]:\n'
                         '            count += 1\n'
                         '            if person["shared"] == True:\n'
                         '                shared += 1\n') + _S_TAIL, "correct", None),
    ("share", _S_ALL, "wrong", "all"),
    ("share", _S_STRICT, "wrong", "strict"),
    ("share", _S_SWAPPED, "wrong", "swapped"),
    ("share", _S_COUNT, "wrong", "count"),
    ("share", _S_HEAD + ('        if person["concern"] >= least:\n'
                         '            count = count + 1\n'
                         '            if not person["shared"]:\n'
                         '                shared = shared + 1\n') + _S_TAIL, "wrong", None),
]
