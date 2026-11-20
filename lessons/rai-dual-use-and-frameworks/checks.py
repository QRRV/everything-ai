# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-dual-use-and-frameworks: what attackers were seen doing with AI, and the rules for secure
AI. The data come from the sources the lesson quotes: the four areas of the NCSC/CISA guidelines (November 2023), the
five phases of ETSI EN 304 223 (December 2025), the dates from which Article 15 of the AI Act applies after the
Digital Omnibus (Article 113, as at 27 July 2026), and Google's counts of the state-backed groups that used Gemini
(January 2025)."""
import contextlib
import copy
import html
import io
import json
import re

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

# --- the sources as data ---------------------------------------------------------------------------------------------

# "four key areas within the AI system development life cycle: secure design, secure development, secure deployment,
# and secure operation and maintenance" (Guidelines for secure AI system development, p. 5)
NCSC_AREAS = ["secure design", "secure development", "secure deployment", "secure operation and maintenance"]
# "13 principles and requirements across five phases: secure design, secure development, secure deployment, secure
# maintenance, and secure end of life" (ETSI press release, 14 January 2026)
ETSI_PHASES = ["secure design", "secure development", "secure deployment", "secure maintenance", "secure end of life"]

# Article 113 (c) as amended: the rules for high-risk AI systems, which hold Article 15, apply from these days
ANNEX_III = "2027-12-02"
ANNEX_I = "2028-08-02"
ACT = "2026-08-02"   # the Act "shall apply from 2 August 2026", Article 15 later

# "Over 20 China-backed groups", "Over 10 Iran-backed groups", "Nine North Korea-backed groups" and "Three Russia-backed
# groups" (Google Threat Intelligence Group, 29 January 2025)
GROUPS = {"China": "Over 20", "Iran": "Over 10", "North Korea": "Nine", "Russia": "Three"}
NUMBERS = {"Nine": 9, "Three": 3}


def uncovered(phases: list, areas: list) -> list:
    """The phases that no area covers, in the order of phases: an area covers a phase when the phase's name without
    'secure ' is part of the area's name."""
    return [p for p in phases if not any(p[7:] in a for a in areas)]


def article_15(kind: str, day: str) -> bool:
    """Whether Article 15 applies on day (year-month-day) to a system of this kind."""
    if kind == "Annex III":
        return day >= ANNEX_III
    if kind == "Annex I":
        return day >= ANNEX_I
    return False


def at_least(groups: dict, numbers: dict) -> int:
    """The fewest groups that counts such as 'Over 20' (at least 21) and 'Nine' allow."""
    total = 0
    for count in groups.values():
        total += int(count[5:]) + 1 if count[:5] == "Over " else numbers[count]
    return total


def same_names(areas: list, phases: list) -> list:
    """The areas whose name is also a phase's name: the worked example."""
    return [a for a in areas if a in phases]


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


ETSI = Named("ETSI_PHASES", ETSI_PHASES)
NCSC = Named("NCSC_AREAS", NCSC_AREAS)
GROUPS_N = Named("GROUPS", GROUPS)
NUMBERS_N = Named("NUMBERS", NUMBERS)

# --- pictures --------------------------------------------------------------------------------------------------------
# Every value is what the learner's code gives: all of it is shown through html.escape.

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.15rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;vertical-align:top")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"
_MONO = "font-family:var(--ea-font-code);font-size:12px;overflow-wrap:anywhere"
_LAW = "font-weight:600;color:var(--ea-ink)"
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


def _bar(k: int, n: int) -> str:
    """k of n as a short grey bar with k beside it."""
    width = 100 * k / n if n else 0
    return (f'<span style="display:inline-flex;align-items:center;gap:0.4rem;width:100%;min-width:6rem">'
            f'<span style="flex:1;height:0.6rem;background:var(--ea-line);border-radius:3px;min-width:3rem">'
            f'<span style="display:block;height:100%;width:{width:.1f}%;background:var(--ea-ink-3);border-radius:3px"></span></span>'
            f'<span style="min-width:1.4rem;text-align:right">{k}</span></span>')


def _quoted(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def label(v) -> str:
    """A value in a few words: a Named value by its name, otherwise as Python writes it, cut short."""
    if isinstance(v, Named):
        return v.name
    if isinstance(v, str):
        return _quoted(v)
    if isinstance(v, list) and all(isinstance(x, str) for x in v):
        text = "[" + ", ".join(_quoted(x) for x in v) + "]"
    elif isinstance(v, dict):
        text = "{" + ", ".join(f"{_quoted(str(k))}: {label(x)}" for k, x in v.items()) + "}"
    else:
        text = repr(v)
    return text if len(text) <= 80 else text[:77] + "..."


_DOCS = {
    "en": {"title": "Five documents on secure AI, and which of them is law",
           "head": ["document", "by", "when", "in its own words", "law?"],
           "rows": [("Secure AI Framework (SAIF)", "Google", "June 2023", "a conceptual framework", "no"),
                    ("Guidelines for secure AI system development", "agencies from 18 countries, among them the UK NCSC and the US CISA",
                     "November 2023", "recommends guidelines", "no"),
                    ("Code of Practice for the Cyber Security of AI", "UK government", "January 2025", "voluntary", "no"),
                    ("ETSI EN 304 223", "ETSI", "December 2025", "baseline security requirements", "no, a standard"),
                    ("AI Act, Article 15", "European Union", "applies from 2 December 2027 or 2 August 2028",
                     "shall", "yes, for high-risk AI systems")],
           "after": "Only the last one is law, and only for high-risk AI systems, from the day in its row.",
           "alt": "{title}. {rows}. {after}"},
    "nl": {"title": "Vijf documenten over veilige AI, en welke daarvan wet is",
           "head": ["document", "van", "wanneer", "in eigen woorden", "wet?"],
           "rows": [("Secure AI Framework (SAIF)", "Google", "juni 2023", "een denkkader", "nee"),
                    ("Guidelines for secure AI system development", "instanties uit 18 landen, waaronder het Britse NCSC en "
                     "het Amerikaanse CISA", "november 2023", "beveelt richtlijnen aan", "nee"),
                    ("Code of Practice for the Cyber Security of AI", "de Britse overheid", "januari 2025", "vrijwillig",
                     "nee"),
                    ("ETSI EN 304 223", "ETSI", "december 2025", "basiseisen voor beveiliging", "nee, een norm"),
                    ("AI Act, artikel 15", "Europese Unie", "geldt vanaf 2 december 2027 of 2 augustus 2028", "moet",
                     "ja, voor AI-systemen met een hoog risico")],
           "after": "Alleen het laatste is wet, en alleen voor AI-systemen met een hoog risico, vanaf de dag in zijn rij.",
           "alt": "{title}. {rows}. {after}"},
}


def documents_picture(lang: str) -> str:
    """The five documents of the lesson, with who made them, when, how they call themselves, and whether they are law."""
    w = _DOCS[lang]
    rows, said = [], []
    for row in w["rows"]:
        law = row[4] not in ("no", "nee") and not row[4].startswith(("no,", "nee,"))
        rows.append([(html.escape(row[0]), _LAW if law else ""), (html.escape(row[1]), ""),
                     (html.escape(row[2]), ""), (html.escape(row[3]), ""), (html.escape(row[4]), _LAW if law else "")])
        said.append(", ".join(row))
    return yours.figure(w["title"], _table(rows, w["head"]) + _note(w["after"]),
                        alt=w["alt"].format(title=w["title"], rows="; ".join(said), after=w["after"]))


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


def _same(got, want) -> bool:
    return got is not _STOPPED and type(got) is type(want) and got == want


_CHECK_WORDS = {
    "en": {"error": "error", "yes": "yes", "no": "no",
           "uncovered": {"title": "Your uncovered on the phases of ETSI EN 304 223",
                         "head": ["phase", "an area that covers it", "in yours", "in the course's"],
                         "none": "none",
                         "fine": "Your uncovered gives the same phases as the course's: {want}.",
                         "differs": "Your uncovered gives {got}, the course's {want}; rows where they differ have a "
                                    "dashed frame.",
                         "other": "Your uncovered does not give a list: the check above says what it gives.",
                         "stops": "Your uncovered stops with an error: the check above says where."},
           "article": {"title": "Your article_15 on three kinds of system and five days",
                       "head": ["kind"], "other": "not high-risk",
                       "fine": "The same as the course's on all {m} days and kinds.",
                       "differs": "It differs from the course's in {k} of the {m} cells: they have a dashed frame.",
                       "stops": "Your article_15 stops with an error in {k} of the {m} cells."},
           "groups": {"title": "Your at_least on Google's counts",
                      "head": ["country", "Google's words", "at least"],
                      "mine": "Your at_least gives {got}; the course's gives {want}.",
                      "stops": "Your at_least stops with an error: the check above says where."},
           "alt": "{title}: {rows}. {after}"},
    "nl": {"error": "fout", "yes": "ja", "no": "nee",
           "uncovered": {"title": "Jouw uncovered op de fasen van ETSI EN 304 223",
                         "head": ["fase", "een gebied dat hem dekt", "in de jouwe", "in die van de course"],
                         "none": "geen",
                         "fine": "Jouw uncovered geeft dezelfde fasen als die van de course: {want}.",
                         "differs": "Jouw uncovered geeft {got}, die van de course {want}; rijen waar ze verschillen "
                                    "hebben een gestreept kader.",
                         "other": "Jouw uncovered geeft geen lijst: de check hierboven zegt wat hij geeft.",
                         "stops": "Jouw uncovered stopt met een fout: de check hierboven zegt waar."},
           "article": {"title": "Jouw article_15 op drie soorten systemen en vijf dagen",
                       "head": ["soort"], "other": "geen hoog risico",
                       "fine": "Hetzelfde als die van de course op alle {m} dagen en soorten.",
                       "differs": "Het verschilt van die van de course in {k} van de {m} cellen: die hebben een "
                                  "gestreept kader.",
                       "stops": "Jouw article_15 stopt met een fout in {k} van de {m} cellen."},
           "groups": {"title": "Jouw at_least op de tellingen van Google",
                      "head": ["land", "de woorden van Google", "minstens"],
                      "mine": "Jouw at_least geeft {got}; die van de course geeft {want}.",
                      "stops": "Jouw at_least stopt met een fout: de check hierboven zegt waar."},
           "alt": "{title}: {rows}. {after}"},
}


def uncovered_picture(seen) -> str:
    """Each phase of ETSI with the area of the guidelines that covers it, and whether the learner's list and the
    course's hold it."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _CHECK_WORDS[seen.lang]
    words = w["uncovered"]
    got = _call(fn, ETSI, NCSC)
    if got is _STOPPED:
        return yours.figure(words["title"], _note(words["stops"]))
    if not isinstance(got, list):
        return yours.figure(words["title"], _note(words["other"]))
    want = uncovered(ETSI_PHASES, NCSC_AREAS)
    rows, said = [], []
    for phase in ETSI_PHASES:
        area = next((a for a in NCSC_AREAS if phase[7:] in a), words["none"])
        mine_, course_ = phase in got, phase in want
        yes_no = (w["yes"] if mine_ else w["no"], w["yes"] if course_ else w["no"])
        rows.append([(_mono(phase), _WHOLE), (_mono(area), ""), (html.escape(yes_no[0]), _OFF if mine_ != course_ else ""),
                     (html.escape(yes_no[1]), "")])
        said.append(f"{phase}: {area}, {yes_no[0]}, {yes_no[1]}")
    after = (words["fine"] if got == want else words["differs"]).format(got=label(got), want=label(want))
    return yours.figure(words["title"], _table(rows, words["head"]) + _note(after),
                        alt=w["alt"].format(title=words["title"], rows="; ".join(said), after=after))


PICTURE_KINDS = ["Annex III", "Annex I", "other"]
PICTURE_DAYS = [ACT, "2027-12-01", ANNEX_III, "2028-08-01", ANNEX_I]


def article_picture(seen) -> str:
    """The learner's article_15 on three kinds of system and five days, a dashed frame where it differs from the
    course's."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _CHECK_WORDS[seen.lang]
    words = w["article"]
    rows, said, k, stops = [], [], 0, 0
    for kind in PICTURE_KINDS:
        shown_kind = words["other"] if kind == "other" else kind
        row = [(html.escape(shown_kind), _WHOLE)]
        for day in PICTURE_DAYS:
            got = _call(fn, kind, day)
            want = article_15(kind, day)
            differs = not _same(got, want)
            k += differs
            stops += got is _STOPPED
            text = w["error"] if got is _STOPPED else (w["yes"] if got is True else w["no"] if got is False else label(got))
            row.append((html.escape(text), _OFF if differs else ""))
            said.append(f"{shown_kind}, {day}: {text}")
        rows.append(row)
    m = len(PICTURE_KINDS) * len(PICTURE_DAYS)
    if stops:
        after = words["stops"].format(k=stops, m=m)
    else:
        after = words["differs"].format(k=k, m=m) if k else words["fine"].format(m=m)
    return yours.figure(words["title"], _table(rows, words["head"] + PICTURE_DAYS) + _note(after),
                        alt=w["alt"].format(title=words["title"], rows="; ".join(said), after=after))


def groups_picture(seen) -> str:
    """Google's counts with the fewest groups each allows, and the learner's at_least beside the course's."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _CHECK_WORDS[seen.lang]
    words = w["groups"]
    got = _call(fn, GROUPS_N, NUMBERS_N)
    want = at_least(GROUPS, NUMBERS)
    rows, said = [], []
    for country, count in GROUPS.items():
        least = at_least({country: count}, NUMBERS)
        rows.append([(html.escape(country), _WHOLE), (_mono(count), _WHOLE), (_bar(least, 21), "")])
        said.append(f"{country}: {count}, {least}")
    if got is _STOPPED:
        after = words["stops"]
    else:
        after = words["mine"].format(got=label(got), want=want)
    return yours.figure(words["title"], _table(rows, words["head"]) + _note(after),
                        alt=w["alt"].format(title=words["title"], rows="; ".join(said), after=after))


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["uncovered", "article-15", "at-least", "dual-use", "reports"]

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
    return isinstance(e, NameError) and re.search(r"'(ETSI_PHASES|NCSC_AREAS|GROUPS|NUMBERS|C)'", str(e)) is not None


_NAMES = Mistake("lesson-names", error=_names_error,
                 en="Your function uses a name of the lesson, such as `GROUPS` or `NCSC_AREAS`, and the check does not give it that name: use the names between the parentheses of the `def` line.",
                 nl="Je functie gebruikt een naam van de les, zoals `GROUPS` of `NCSC_AREAS`, en de check geeft die naam niet mee: gebruik de namen tussen de haakjes van de `def`-regel.")

UNCOVERED_CASES = [(ETSI, NCSC), (["secure design", "secure testing"], ["secure design and testing"]),
                   (["secure design"], []), ([], NCSC), (["secure end of life"], ["secure end of life"]),
                   (["secure x", "secure y", "secure z"], ["secure y"]), (NCSC, ETSI)]
ARTICLE_CASES = [("Annex III", ANNEX_III), ("Annex III", "2026-09-01"), ("Annex I", ANNEX_III), ("Annex I", ANNEX_I),
                 ("not high-risk", "2030-01-01"), ("Annex III", "2028-01-15"), ("Annex I", ACT),
                 ("Annex I", "2031-05-20"), ("Annex III", "2027-12-01")]
GROUPS_CASES = [(GROUPS_N, NUMBERS_N), ({"A": "Over 5"}, {}), ({"A": "Two"}, {"Two": 2}), ({}, {}),
                ({"A": "Over 0", "B": "One"}, {"One": 1}), ({"X": "Over 100", "Y": "Over 1"}, {})]

# exercise 1: uncovered
_U_HEAD = ("def uncovered(phases, areas):\n"
           "    # the phases that no area covers, in the order of phases\n"
           "    found = []\n"
           "    for phase in phases:\n"
           "        rest = phase[7:]   # the name without \"secure \"\n")
_U_BODY = ("        covered = False\n"
           "        for area in areas:\n"
           "            if rest in area:\n"
           "                covered = True\n"
           "        if not covered:\n"
           "            found.append(phase)\n")
_U_TAIL = "    return found\n"
_U_START = _U_HEAD + "        ...\n" + _U_TAIL
_U = _U_HEAD + _U_BODY + _U_TAIL
_U_EXACT = _U.replace("if rest in area:", "if phase == area:")
_U_COVERED = _U.replace("if not covered:", "if covered:")
_U_ONCE = _U_HEAD.replace("    found = []\n", "    found = []\n    covered = False\n") + _U_BODY.replace(
    "        covered = False\n", "") + _U_TAIL
_U_EACH = _U_HEAD + ("        for area in areas:\n"
                     "            if rest not in area:\n"
                     "                found.append(phase)\n") + _U_TAIL

# exercise 2: article_15
_A_HEAD = ("def article_15(kind, day):\n"
           "    # True when Article 15 applies to a system of this kind on this day\n")
_A_BODY = ("    if kind == \"Annex III\":\n"
           "        return day >= \"2027-12-02\"\n"
           "    if kind == \"Annex I\":\n"
           "        return day >= \"2028-08-02\"\n"
           "    return False\n")
_A_START = _A_HEAD + "    ...\n"
_A = _A_HEAD + _A_BODY
_A_ACT = _A.replace("\"2027-12-02\"", "\"2026-08-02\"").replace("\"2028-08-02\"", "\"2026-08-02\"")
_A_ALL = _A_HEAD + "    return day >= \"2027-12-02\"\n"
_A_SWAPPED = _A.replace("\"2027-12-02\"", "\"X\"").replace("\"2028-08-02\"", "\"2027-12-02\"").replace("\"X\"", "\"2028-08-02\"")
_A_AFTER = _A.replace(">=", ">")

# exercise 3: at_least
_G_HEAD = ("def at_least(groups, numbers):\n"
           "    # the fewest groups that the counts allow\n"
           "    total = 0\n"
           "    for country, count in groups.items():\n")
_G_BODY = ("        if count[:5] == \"Over \":\n"
           "            total = total + int(count[5:]) + 1\n"
           "        else:\n"
           "            total = total + numbers[count]\n")
_G_TAIL = "    return total\n"
_G_START = _G_HEAD + "        ...\n" + _G_TAIL
_G = _G_HEAD + _G_BODY + _G_TAIL
_G_OVER = _G.replace("int(count[5:]) + 1", "int(count[5:])")
_G_WORDS = _G_HEAD + ("        if count[:5] == \"Over \":\n"
                      "            total = total + int(count[5:]) + 1\n") + _G_TAIL
_G_PLUS = _G.replace("numbers[count]", "numbers[count] + 1")


EXERCISES = {
    "uncovered": FunctionBox(
        name="uncovered",
        picture=uncovered_picture,
        function="uncovered",
        start=_U_START,
        solution=_U,
        visible=[UNCOVERED_CASES[0]],
        hidden=UNCOVERED_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("exact", wrong=_U_EXACT,
                    en="`{call}` gives {got}, and it should give {want}: your `uncovered` compares whole names, so `secure maintenance` counts as not covered, though `maintenance` is part of `secure operation and maintenance`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw `uncovered` vergelijkt hele namen, dus `secure maintenance` telt als niet gedekt, terwijl `maintenance` wel deel is van `secure operation and maintenance`."),
            Mistake("covered", wrong=_U_COVERED,
                    en="`{call}` gives {got}, and it should give {want}: your `uncovered` gives the phases that an area does cover. The exercise asks for the others.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw `uncovered` geeft de fasen die een gebied wel dekt. De oefening vraagt om de andere."),
            Mistake("once", wrong=_U_ONCE,
                    en="`{call}` gives {got}, and it should give {want}: `covered` is set to `False` once, before the loop over the phases, so after the first covered phase every phase counts as covered. Set it to `False` again for each phase.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `covered` wordt één keer op `False` gezet, vóór de loop over de fasen, dus na de eerste gedekte fase telt elke fase als gedekt. Zet hem voor elke fase opnieuw op `False`."),
            Mistake("each", wrong=_U_EACH,
                    en="`{call}` gives {got}, and it should give {want}: your `uncovered` adds a phase for every area that does not cover it. A phase is uncovered only when no area at all covers it: look at all the areas first, then decide once.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw `uncovered` voegt een fase toe voor elk gebied dat hem niet dekt. Een fase is alleen niet gedekt als geen enkel gebied hem dekt: bekijk eerst alle gebieden, en beslis dan één keer."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: look through the areas for one that holds `rest`, and add the phase to `found` when none does.",
                      "nl": "Vervang de `...`: zoek in de gebieden naar een dat `rest` bevat, en voeg de fase aan `found` toe als er geen is."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. A phase is uncovered when no area holds its name without `secure `.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Een fase is niet gedekt als geen gebied zijn naam zonder `secure ` bevat."},
            "almost": {"en": "Right for `uncovered(ETSI_PHASES, NCSC_AREAS)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `phases` and `areas` it gets?",
                       "nl": "Goed voor `uncovered(ETSI_PHASES, NCSC_AREAS)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `phases` en `areas` die ze krijgt?"},
            "correct": {"en": "Your `uncovered` is right in all {n} cases. Of ETSI's five phases, only `secure end of life` has no area in the guidelines of 2023: the standard adds what happens to data and models when a system is retired. ETSI's principle 13 is `Ensure proper data and model disposal`.",
                        "nl": "Je `uncovered` klopt in alle {n} gevallen. Van de vijf fasen van ETSI heeft alleen `secure end of life` geen gebied in de richtlijnen van 2023: de norm voegt toe wat er met data en modellen gebeurt als een systeem stopt. Principe 13 van ETSI gaat over het op de juiste manier afvoeren van data en modellen."},
        },
    ),
    "article-15": FunctionBox(
        name="article-15",
        picture=article_picture,
        function="article_15",
        start=_A_START,
        solution=_A,
        visible=[ARTICLE_CASES[0]],
        hidden=ARTICLE_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("act", wrong=_A_ACT,
                    en="`{call}` gives {got}, and it should give {want}: the AI Act shall apply from 2 August 2026, but after the Digital Omnibus the rules for high-risk AI systems, with Article 15, apply later: from 2 December 2027 for Annex III, and from 2 August 2028 for Annex I.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de AI Act geldt vanaf 2 augustus 2026, maar na de Digital Omnibus gelden de regels voor AI-systemen met een hoog risico, met artikel 15, later: vanaf 2 december 2027 voor bijlage III, en vanaf 2 augustus 2028 voor bijlage I."),
            Mistake("all", wrong=_A_ALL,
                    en="`{call}` gives {got}, and it should give {want}: your `article_15` looks only at the day. The kind matters too: Annex I has its own day, and a system that is not high-risk never falls under Article 15.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw `article_15` kijkt alleen naar de dag. De soort telt ook: bijlage I heeft een eigen dag, en een systeem zonder hoog risico valt nooit onder artikel 15."),
            Mistake("swapped", wrong=_A_SWAPPED,
                    en="`{call}` gives {got}, and it should give {want}: the two days are swapped. Annex III comes first, from 2 December 2027; Annex I follows, from 2 August 2028.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de twee dagen zijn omgewisseld. Bijlage III komt eerst, vanaf 2 december 2027; bijlage I volgt, vanaf 2 augustus 2028."),
            Mistake("after", wrong=_A_AFTER,
                    en="`{call}` gives {got}, and it should give {want}: Article 15 applies from its first day on, and that day counts too. Your `article_15` starts a day late.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: artikel 15 geldt vanaf zijn eerste dag, en die dag telt mee. Jouw `article_15` begint een dag te laat."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give the day from which Article 15 applies for each annex, and `False` for any other kind.",
                      "nl": "Vervang de `...`: geef voor elke bijlage de dag vanaf wanneer artikel 15 geldt, en `False` voor elke andere soort."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Annex III from 2 December 2027, Annex I from 2 August 2028, both days included, and never for a system that is not high-risk.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Bijlage III vanaf 2 december 2027, bijlage I vanaf 2 augustus 2028, beide dagen meegeteld, en nooit voor een systeem zonder hoog risico."},
            "almost": {"en": "Right for `article_15(\"Annex III\", \"2027-12-02\")`. But `{call}` gives {got}, and it should give {want}. Does your function use the `kind` and `day` it gets?",
                       "nl": "Goed voor `article_15(\"Annex III\", \"2027-12-02\")`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `kind` en `day` die ze krijgt?"},
            "correct": {"en": "Your `article_15` is right in all {n} cases. Of the 15 cells in the picture, 4 say yes: none before 2 December 2027, and never for a system that is not high-risk.",
                        "nl": "Je `article_15` klopt in alle {n} gevallen. Van de 15 cellen in het plaatje zeggen er 4 ja: geen enkele vóór 2 december 2027, en nooit voor een systeem zonder hoog risico."},
        },
    ),
    "at-least": FunctionBox(
        name="at-least",
        picture=groups_picture,
        function="at_least",
        start=_G_START,
        solution=_G,
        visible=[GROUPS_CASES[0]],
        hidden=GROUPS_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("over", wrong=_G_OVER,
                    en="`{call}` gives {got}, and it should give {want}: your `at_least` reads `Over 20` as 20. More than 20 groups is at least 21.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw `at_least` leest `Over 20` als 20. Meer dan 20 groepen is minstens 21."),
            Mistake("words", wrong=_G_WORDS,
                    en="`{call}` gives {got}, and it should give {want}: your `at_least` counts only the counts that start with `Over `. A count such as `Nine` is in `numbers`: add `numbers[count]` for those.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw `at_least` telt alleen de tellingen die met `Over ` beginnen. Een telling als `Nine` staat in `numbers`: tel daarvoor `numbers[count]` op."),
            Mistake("plus", wrong=_G_PLUS,
                    en="`{call}` gives {got}, and it should give {want}: your `at_least` adds 1 to `Nine` and `Three` too. Only `Over` means more than the number; `Nine` is exactly 9.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw `at_least` telt ook bij `Nine` en `Three` 1 op. Alleen `Over` betekent meer dan het getal; `Nine` is precies 9."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: for a count that starts with `Over `, add the fewest groups it allows to `total`; for another count, add its number from `numbers`.",
                      "nl": "Vervang de `...`: tel voor een telling die met `Over ` begint het kleinste aantal groepen dat hij toelaat op bij `total`; tel voor een andere telling het getal uit `numbers` op."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. `Over 20` allows 21 groups at the fewest, and `Nine` exactly 9.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. `Over 20` laat minstens 21 groepen toe, en `Nine` precies 9."},
            "almost": {"en": "Right for `at_least(GROUPS, NUMBERS)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `groups` and `numbers` it gets?",
                       "nl": "Goed voor `at_least(GROUPS, NUMBERS)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `groups` en `numbers` die ze krijgt?"},
            "correct": {"en": "Your `at_least` is right in all {n} cases. Google's counts allow at least 44 groups backed by these four states, and its report speaks of APT groups from more than 20 countries: these four are only part of them.",
                        "nl": "Je `at_least` klopt in alle {n} gevallen. De tellingen van Google laten minstens 44 groepen toe die door deze vier staten gesteund worden, en het rapport spreekt van APT-groepen uit meer dan 20 landen: deze vier zijn er maar een deel van."},
        },
    ),
}

CASES = [
    ("uncovered", _U_START, "empty", None),
    ("uncovered", _U, "correct", None),
    ("uncovered", "def uncovered(phases, areas):\n    return [p for p in phases if not any(p[7:] in a for a in areas)]\n", "correct", None),
    ("uncovered", _U_EXACT, "wrong", "exact"),
    ("uncovered", _U_COVERED, "wrong", "covered"),
    ("uncovered", _U_ONCE, "wrong", "once"),
    ("uncovered", _U_EACH, "wrong", "each"),
    ("uncovered", "def uncovered(phases, areas):\n    return [\"secure end of life\"]\n", "almost", None),
    ("uncovered", "def uncovered(phases, areas):\n    return ETSI_PHASES\n", "error", "lesson-names"),
    ("article-15", _A_START, "empty", None),
    ("article-15", _A, "correct", None),
    ("article-15", "def article_15(kind, day):\n    first = {\"Annex III\": \"2027-12-02\", \"Annex I\": \"2028-08-02\"}\n    return kind in first and day >= first[kind]\n", "correct", None),
    ("article-15", _A_ACT, "wrong", "act"),
    ("article-15", _A_ALL, "wrong", "all"),
    ("article-15", _A_SWAPPED, "wrong", "swapped"),
    ("article-15", _A_AFTER, "wrong", "after"),
    ("article-15", "def article_15(kind, day):\n    return True\n", "almost", None),
    ("article-15", "def article_15(kind, day):\n    return C\n", "error", "lesson-names"),
    ("at-least", _G_START, "empty", None),
    ("at-least", _G, "correct", None),
    ("at-least", "def at_least(groups, numbers):\n    return sum(int(c[5:]) + 1 if c.startswith(\"Over \") else numbers[c] for c in groups.values())\n", "correct", None),
    ("at-least", _G_OVER, "wrong", "over"),
    ("at-least", _G_WORDS, "wrong", "words"),
    ("at-least", _G_PLUS, "wrong", "plus"),
    ("at-least", "def at_least(groups, numbers):\n    return 44\n", "almost", None),
    ("at-least", "def at_least(groups, numbers):\n    return len(GROUPS)\n", "error", "lesson-names"),
]
