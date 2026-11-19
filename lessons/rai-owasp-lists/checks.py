# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-owasp-lists: the three editions of the OWASP Top 10 for LLM Applications (2023, 2025 and
2026) and the list for agents, as the course writes them down from OWASP's own documents, and a first review of the
made-up library's assistant with the list of 2026. The three things the learner writes: where (the id of one entry in
every edition), moves (how many places each entry rose between two editions) and findings (the lines of a review,
each with its id and edition); with the pictures of what they give. The review questions and the library's answers
are the course's own, made up for this lesson."""
import copy
import html
import re

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, with_time_limit


class Called(dict):
    """A dict the check calls by the name the lesson gives it, such as EDITIONS, so that a message stays short. The
    learner's code gets its own copy, a plain dict."""

    def __init__(self, values: dict, name: str):
        super().__init__(values)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(dict(self), memo)


# --- the lists, as OWASP's documents name them ------------------------------------------------------------------------

# OWASP Top 10 for LLM Applications: version 1.1 (October 2023), the 2025 edition (November 2024) and the 2026 edition
_2023 = ["Prompt Injection", "Insecure Output Handling", "Training Data Poisoning", "Model Denial of Service",
         "Supply Chain Vulnerabilities", "Sensitive Information Disclosure", "Insecure Plugin Design", "Excessive Agency",
         "Overreliance", "Model Theft"]
_2025 = ["Prompt Injection", "Sensitive Information Disclosure", "Supply Chain", "Data and Model Poisoning",
         "Improper Output Handling", "Excessive Agency", "System Prompt Leakage", "Vector and Embedding Weaknesses",
         "Misinformation", "Unbounded Consumption"]
_2026 = ["Prompt Injection", "Sensitive Information Disclosure", "Excessive Agency", "Supply Chain",
         "Data and Model Poisoning", "Unbounded Consumption", "Misinformation", "Hidden Context Exposure",
         "Vector and Embedding Weaknesses", "Improper Output Handling"]


def _ids(prefix: str, names: list) -> dict:
    return {f"{prefix}{i:02d}": name for i, name in enumerate(names, start=1)}


EDITIONS = Called({2023: _ids("LLM", _2023), 2025: _ids("LLM", _2025), 2026: _ids("LLM", _2026)}, "EDITIONS")
for _year in (2023, 2025, 2026):
    EDITIONS[_year] = Called(EDITIONS[_year], f"EDITIONS[{_year}]")

# OWASP Top 10 for Agentic Applications for 2026 (December 2025)
AGENTS = Called(_ids("ASI", ["Agent Goal Hijack", "Tool Misuse and Exploitation", "Identity and Privilege Abuse",
                             "Agentic Supply Chain Vulnerabilities", "Unexpected Code Execution (RCE)",
                             "Memory & Context Poisoning", "Insecure Inter-Agent Communication", "Cascading Failures",
                             "Human-Agent Trust Exploitation", "Rogue Agents"]), "AGENTS")

RENAMED = {"Hidden Context Exposure": "System Prompt Leakage"}   # 2026 name: the 2025 name it replaced

# a made-up list of three office risks in two editions, for the hidden cases: other ids, other years
MADE_UP = Called({2024: Called(_ids("OFF", ["Phishing", "Weak Passwords", "Lost Laptops"]), "MADE_UP[2024]"),
                  2025: Called(_ids("OFF", ["Weak Passwords", "Lost Laptops", "Phishing", "Shadow IT"]),
                               "MADE_UP[2025]")}, "MADE_UP")


# --- the review of the library's assistant ----------------------------------------------------------------------------

# one question for each entry of 2026, the course's own words, as the lesson's table shows them; True means: yes, look
# closer
QUESTIONS = {
    "en": {"LLM01": "Does text that the app did not write reach the model, such as what members type?",
           "LLM02": "Can the model see data that not everyone who talks to it may see?",
           "LLM03": "Can the model do things, such as change data, without a person checking first?",
           "LLM04": "Does the app use a model, data or software that others made?",
           "LLM05": "Does the app train a model further on data that others can change?",
           "LLM06": "Can one user ask as much as they like, without a limit?",
           "LLM07": "Do people act on its answers without a source to check them against?",
           "LLM08": "Is there anything in its hidden instructions that may not leak?",
           "LLM09": "Does the app look things up in documents that others can add or change?",
           "LLM10": "Does its output go on to other software without a check, such as into SQL?"},
    "nl": {"LLM01": "Komt er tekst bij het model die de app niet zelf schreef, zoals wat leden typen?",
           "LLM02": "Kan het model data zien die niet iedereen mag zien die ermee praat?",
           "LLM03": "Kan het model dingen doen, zoals data veranderen, zonder dat iemand eerst kijkt?",
           "LLM04": "Gebruikt de app een model, data of software die anderen maakten?",
           "LLM05": "Traint de app een model verder op data die anderen kunnen veranderen?",
           "LLM06": "Kan één gebruiker zoveel vragen als die wil, zonder grens?",
           "LLM07": "Doen mensen iets met de antwoorden zonder bron om ze na te kijken?",
           "LLM08": "Staat er iets in de verborgen instructies dat niet mag uitlekken?",
           "LLM09": "Zoekt de app dingen op in documenten die anderen kunnen toevoegen of veranderen?",
           "LLM10": "Gaat de output zonder controle door naar andere software, zoals in SQL?"},
}

# the library's assistant: an open-weights model from a hub, the loans of every member in view, it renews and cancels
# loans by itself with SQL it writes, the key to the catalog in its hidden instructions, no limit on questions; it
# names the catalog record each answer comes from, nobody else can add documents, and it is not trained further
ANSWERS = Called({"LLM01": True, "LLM02": True, "LLM03": True, "LLM04": True, "LLM05": False, "LLM06": True,
                  "LLM07": False, "LLM08": True, "LLM09": False, "LLM10": True}, "ANSWERS")


def _answers_2025(answers: dict) -> dict:
    """The same answers under the ids of 2025, entry by entry, with System Prompt Leakage for Hidden Context
    Exposure."""
    by_name = {RENAMED.get(EDITIONS[2026][code], EDITIONS[2026][code]): yes for code, yes in answers.items()}
    return {code: by_name[name] for code, name in EDITIONS[2025].items()}


ANSWERS_2025 = Called(_answers_2025(ANSWERS), "ANSWERS_2025")
# a second made-up app for the hidden cases: a museum guide that writes texts about paintings
GUIDE = Called({"LLM01": True, "LLM02": False, "LLM03": False, "LLM04": True, "LLM05": True, "LLM06": False,
                "LLM07": True, "LLM08": False, "LLM09": True, "LLM10": False}, "GUIDE")


# --- the course's own versions of the three functions -------------------------------------------------------------

def where(name: str, editions: dict) -> dict:
    found = {}
    for year, entries in editions.items():
        found[year] = None
        for code, entry in entries.items():
            if entry == name:
                found[year] = code
    return found


def moves(old: dict, new: dict) -> dict:
    risen = {}
    for new_code, name in new.items():
        for old_code, old_name in old.items():
            if old_name == name:
                risen[name] = int(old_code[3:]) - int(new_code[3:])
    return risen


def findings(answers: dict, entries: dict, edition) -> list:
    return [f"{code}:{edition} {entries[code]}" for code, yes in answers.items() if yes]


# --- formatting -------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.78rem;color:var(--ea-ink-2)"
_CODE = "font-family:var(--ea-font-code);font-size:0.78rem"
_CELL = "padding:0.15rem 0.5rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--ea-line)"
_MARK = "background:var(--ea-data-wash);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"


def _note(parts: list) -> str:
    return f'<div style="{_UI}">{html.escape(" ".join(p for p in parts if p))}</div>'


def _table(head: list, rows: list) -> str:
    """head: the column titles; rows: lists of (html, style) cells."""
    th = "".join(f'<th style="{_CELL};{_LABEL};font-weight:600">{html.escape(str(h))}</th>' for h in head)
    body = "".join("<tr>" + "".join(f'<td style="{_CELL};{style}">{cell}</td>' for cell, style in row) + "</tr>"
                   for row in rows)
    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;'
            f'font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink);margin-top:0.3rem">'
            f'<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>')


def _cut(text: str, n: int = 80) -> str:
    text = re.sub(r"\s+", " ", text)
    return text if len(text) <= n else text[:n - 3] + "..."


def _short(v) -> str:
    try:
        text = repr(v)
    except Exception:
        text = "?"
    return _cut(text, 60)


_STOPPED = object()   # what a call gave back when it stopped with an error


def _call(fn, *args):
    try:
        return with_time_limit(lambda: fn(*copy.deepcopy(args)))
    except Exception:
        return _STOPPED


def _fn(seen, name: str):
    mine = seen.mine
    if mine is None:
        return None
    fn = mine.names.get(name)
    return fn if callable(fn) else None


def _editions_table(editions: dict, marked: dict, off: set) -> str:
    """The editions side by side, one column each, a row for each place; marked: year -> the id to mark; off: the
    years whose column gets a dashed purple frame."""
    years = list(editions)
    places = max(len(entries) for entries in editions.values())
    rows = []
    for i in range(places):
        row = []
        for year in years:
            codes = list(editions[year])
            code = codes[i] if i < len(codes) else None
            text = f'<span style="{_CODE};color:var(--ea-ink-3)">{code}</span> {html.escape(editions[year][code])}' \
                if code else ""
            style = (_MARK if code is not None and marked.get(year) == code else "") + (";" + _OFF if year in off else "")
            row.append((text, style))
        rows.append(row)
    return _table(years, rows)


# --- the picture in The idea: three editions, one entry followed ------------------------------------------------------

_IDEA_WORDS = {
    "en": {"title": "Ten entries, three editions",
           "said": "Each column is one edition of OWASP's list for LLM applications, from LLM01 at the top to LLM10. "
                   "The shaded cells are Excessive Agency: LLM08 in 2023, LLM06 in 2025 and LLM03 in 2026. Prompt "
                   "Injection is LLM01 in all three."},
    "nl": {"title": "Tien risico's, drie edities",
           "said": "Elke kolom is één editie van de lijst van OWASP voor LLM-applicaties, van LLM01 bovenaan tot LLM10. "
                   "De gekleurde cellen zijn Excessive Agency: LLM08 in 2023, LLM06 in 2025 en LLM03 in 2026. Prompt "
                   "Injection is in alle drie LLM01."},
}


def idea_picture(lang: str) -> str:
    w = _IDEA_WORDS[lang]
    marked = where("Excessive Agency", EDITIONS)
    return yours.figure(w["title"], _editions_table(EDITIONS, marked, set()) + _note([w["said"]]))


# --- the picture under exercise 1: where -------------------------------------------------------------------------------

_WHERE_NAME = "Supply Chain"
_WHERE_WORDS = {
    "en": {"title": "Your where for Supply Chain",
           "said": "Your where(\"Supply Chain\", EDITIONS) gives {got}. The shaded cells are the ids it gives; a "
                   "column without one got None.",
           "fine": "The same as the course's where.",
           "wrong": "A dashed purple frame marks each edition where it differs from the course's where, which gives "
                    "{want}.",
           "stops": "Your where stops with an error on where(\"Supply Chain\", EDITIONS): the check above says where.",
           "off": "Your where gives {got} for where(\"Supply Chain\", EDITIONS), not a dict from year to id, so there "
                  "is nothing to shade."},
    "nl": {"title": "Jouw where voor Supply Chain",
           "said": "Jouw where(\"Supply Chain\", EDITIONS) geeft {got}. De gekleurde cellen zijn de ids die ze geeft; "
                   "een kolom zonder gekleurde cel kreeg None.",
           "fine": "Hetzelfde als de where van de course.",
           "wrong": "Een gestreept paars kader wijst elke editie aan waar ze verschilt van de where van de course, die "
                    "{want} geeft.",
           "stops": "Jouw where stopt met een fout bij where(\"Supply Chain\", EDITIONS): de check hierboven zegt waar.",
           "off": "Jouw where geeft {got} voor where(\"Supply Chain\", EDITIONS), geen dict van jaar naar id, dus er "
                  "is niets om te kleuren."},
}


def _dict_text(d: dict) -> str:
    """A dict the way the check's messages write it: text between double quotes."""
    def one(v):
        return f'"{v}"' if isinstance(v, str) else repr(v)
    return "{" + ", ".join(f"{one(k)}: {one(v)}" for k, v in d.items()) + "}"


def where_picture(seen) -> str:
    fn = _fn(seen, "where")
    if fn is None:
        return ""
    w = _WHERE_WORDS[seen.lang]
    got = _call(fn, _WHERE_NAME, EDITIONS)
    if got is _STOPPED:
        return yours.figure(w["title"], _note([w["stops"]]))
    if not isinstance(got, dict):
        return yours.figure(w["title"], _note([w["off"].format(got=_short(got))]))
    want = where(_WHERE_NAME, EDITIONS)
    off = {year for year in EDITIONS if year not in got or got[year] != want[year]}
    said = w["said"].format(got=_cut(_dict_text(got)))
    return yours.figure(w["title"], _editions_table(EDITIONS, got, off)
                        + _note([said, w["wrong"].format(want=_dict_text(want)) if off else w["fine"]]))


# --- the picture under exercise 2: moves -------------------------------------------------------------------------------

_MOVES_WORDS = {
    "en": {"title": "Your moves from 2025 to 2026",
           "head": ["2026", "entry", "2025", "your moves"],
           "said": "Each row is one entry of 2026, with its id in 2025 and what your moves(EDITIONS[2025], "
                   "EDITIONS[2026]) gives for it: a bar to the right for places it rose, to the left for places it fell.",
           "none": "not in your dict",
           "fine": "The same as the course's moves. Hidden Context Exposure has no id in 2025: by name, it is new.",
           "wrong": "A dashed purple frame marks each row where your moves differs from the course's.",
           "stops": "Your moves stops with an error on moves(EDITIONS[2025], EDITIONS[2026]): the check above says where.",
           "off": "Your moves gives {got} for moves(EDITIONS[2025], EDITIONS[2026]), not a dict from name to places, "
                  "so there is nothing to draw."},
    "nl": {"title": "Jouw moves van 2025 naar 2026",
           "head": ["2026", "risico", "2025", "jouw moves"],
           "said": "Elke rij is één risico van 2026, met de id in 2025 en wat jouw moves(EDITIONS[2025], "
                   "EDITIONS[2026]) ervoor geeft: een balk naar rechts voor plaatsen omhoog, naar links voor plaatsen "
                   "omlaag.",
           "none": "niet in je dict",
           "fine": "Hetzelfde als de moves van de course. Hidden Context Exposure heeft geen id in 2025: op naam is "
                   "dat risico nieuw.",
           "wrong": "Een gestreept paars kader wijst elke rij aan waar jouw moves verschilt van die van de course.",
           "stops": "Jouw moves stopt met een fout bij moves(EDITIONS[2025], EDITIONS[2026]): de check hierboven zegt "
                    "waar.",
           "off": "Jouw moves geeft {got} voor moves(EDITIONS[2025], EDITIONS[2026]), geen dict van naam naar "
                  "plaatsen, dus er is niets om te tekenen."},
}


def _bar(v) -> str:
    """A bar from the middle: to the right for a rise, to the left for a fall, 0.5rem per place, with the number."""
    if not isinstance(v, int) or isinstance(v, bool):
        return html.escape(_short(v))
    size = f"{min(abs(v), 9) * 0.5}rem"
    bar = f'<span style="display:inline-block;height:0.6rem;width:{size};background:var(--ea-ink-2);' \
          f'vertical-align:middle"></span>'
    left = bar if v < 0 else ""
    right = bar if v > 0 else ""
    number = f"+{v}" if v > 0 else str(v).replace("-", "−")
    return (f'<span style="display:inline-block;width:4.6rem;text-align:right">{left}</span>'
            f'<span style="display:inline-block;width:1px;height:0.9rem;background:var(--ea-axis);'
            f'vertical-align:middle"></span>'
            f'<span style="display:inline-block;width:4.6rem;text-align:left">{right}</span>'
            f'<span style="{_CODE};margin-left:0.4rem">{number}</span>')


def moves_picture(seen) -> str:
    fn = _fn(seen, "moves")
    if fn is None:
        return ""
    w = _MOVES_WORDS[seen.lang]
    got = _call(fn, EDITIONS[2025], EDITIONS[2026])
    if got is _STOPPED:
        return yours.figure(w["title"], _note([w["stops"]]))
    if not isinstance(got, dict):
        return yours.figure(w["title"], _note([w["off"].format(got=_short(got))]))
    want = moves(EDITIONS[2025], EDITIONS[2026])
    old = {name: code for code, name in EDITIONS[2025].items()}
    rows, wrong = [], False
    for code, name in EDITIONS[2026].items():
        mine = got.get(name)
        differs = (name in got) != (name in want) or (name in want and got.get(name) != want[name])
        wrong |= differs
        shown = _bar(mine) if name in got else html.escape(w["none"]) if name in want else ""
        style = _OFF if differs else ""
        rows.append([(f'<span style="{_CODE}">{code}</span>', style), (html.escape(name), style),
                     (f'<span style="{_CODE}">{old.get(name, "")}</span>', style), (shown, style)])
    return yours.figure(w["title"], _table(w["head"], rows) + _note([w["said"], w["wrong"] if wrong else w["fine"]]))


# --- the picture under exercise 3: findings ---------------------------------------------------------------------------

_FIND_WORDS = {
    "en": {"title": "Your findings for the library's assistant",
           "head": ["id", "entry", "answer", "your line"],
           "yes": "yes", "no": "no",
           "said": "Each row is one entry of 2026, with the library's answer to its question. The last column "
                   "is the line your findings(ANSWERS, EDITIONS[2026], 2026) gives for that entry.",
           "fine": "{n} lines, the same as the course's findings.",
           "wrong": "A dashed purple frame marks each row where your lines differ from the course's.",
           "extra": "Your findings also gives lines that belong to no entry: {extra}.",
           "stops": "Your findings stops with an error on findings(ANSWERS, EDITIONS[2026], 2026): the check above "
                    "says where.",
           "off": "Your findings gives {got} for findings(ANSWERS, EDITIONS[2026], 2026), not a list of lines, so "
                  "there is nothing to put in the table."},
    "nl": {"title": "Jouw findings voor de assistent van de bibliotheek",
           "head": ["id", "risico", "antwoord", "jouw regel"],
           "yes": "ja", "no": "nee",
           "said": "Elke rij is één risico van 2026, met het antwoord van de bibliotheek op de vraag erbij. De laatste "
                   "kolom is de regel die jouw findings(ANSWERS, EDITIONS[2026], 2026) voor dat risico geeft.",
           "fine": "{n} regels, hetzelfde als de findings van de course.",
           "wrong": "Een gestreept paars kader wijst elke rij aan waar jouw regels verschillen van die van de course.",
           "extra": "Jouw findings geeft ook regels die bij geen risico horen: {extra}.",
           "stops": "Jouw findings stopt met een fout bij findings(ANSWERS, EDITIONS[2026], 2026): de check hierboven "
                    "zegt waar.",
           "off": "Jouw findings geeft {got} voor findings(ANSWERS, EDITIONS[2026], 2026), geen lijst met regels, dus "
                  "er is niets om in de tabel te zetten."},
}


def _code_of(line) -> str | None:
    """The id a line starts with, such as LLM01 in "LLM01:2026 Prompt Injection"."""
    found = re.match(r"\s*(LLM\d\d)", str(line))
    return found.group(1) if found else None


def findings_picture(seen) -> str:
    fn = _fn(seen, "findings")
    if fn is None:
        return ""
    w = _FIND_WORDS[seen.lang]
    got = _call(fn, ANSWERS, EDITIONS[2026], 2026)
    if got is _STOPPED:
        return yours.figure(w["title"], _note([w["stops"]]))
    if not isinstance(got, (list, tuple)):
        return yours.figure(w["title"], _note([w["off"].format(got=_short(got))]))
    want = findings(ANSWERS, EDITIONS[2026], 2026)
    mine: dict = {}
    extra = []
    for line in got:
        code = _code_of(line)
        if code in EDITIONS[2026]:
            mine.setdefault(code, []).append(str(line))
        else:
            extra.append(_short(line))
    rows, wrong = [], False
    for code in EDITIONS[2026]:
        lines = mine.get(code, [])
        right = [x for x in want if x.startswith(code + ":")]
        differs = lines != right
        wrong |= differs
        style = _OFF if differs else ""
        answer = w["yes"] if ANSWERS[code] else w["no"]
        shown = "<br>".join(f'<span style="{_CODE}">{html.escape(x)}</span>' for x in lines)
        rows.append([(f'<span style="{_CODE}">{code}</span>', style), (html.escape(EDITIONS[2026][code]), style),
                     (answer, (_MARK if ANSWERS[code] else "") + (";" + style if style else "")), (shown, style)])
    wrong |= bool(extra)
    notes = [w["said"], w["wrong"] if wrong else w["fine"].format(n=len(want))]
    if extra:
        notes.append(w["extra"].format(extra=", ".join(extra[:3])))
    return yours.figure(w["title"], _table(w["head"], rows) + _note(notes))


# --- the exercises ----------------------------------------------------------------------------------------------------

REQUIRED = ["where", "moves", "findings", "vote", "lists"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _lesson_name(e) -> bool:
    return isinstance(e, NameError) and re.search(r"'(EDITIONS|ANSWERS|ANSWERS_2025|GUIDE|MADE_UP|AGENTS|C)'",
                                                  str(e)) is not None


_NAMES = Mistake("lesson-names", error=_lesson_name,
                 en="With `{call}`, your function uses a name of the lesson, such as `EDITIONS` or `ANSWERS`, and the box does not know it. Use the names between the parentheses of the `def` line: the check calls your function with other lists too.",
                 nl="Met `{call}` gebruikt je functie een naam uit de les, zoals `EDITIONS` of `ANSWERS`, en die kent het vak niet. Gebruik de namen tussen de haakjes van de `def`-regel: de check roept je functie ook aan met andere lijsten.")

# exercise 1: where
_W_HEAD = ("def where(name, editions):\n"
           "    # for each year in editions: the id of the entry called name, or None when that edition has none\n")
_W_START = _W_HEAD + "    ...\n"


def _w(body: str) -> str:
    return _W_HEAD + body


_W_BODY = ("    found = {}\n"
           "    for year, entries in editions.items():\n"
           "        found[year] = None\n"
           "        for code, entry in entries.items():\n"
           "            if entry == name:\n"
           "                found[year] = code\n"
           "    return found\n")
_W = _w(_W_BODY)
_W_NONE = _w(_W_BODY.replace("        found[year] = None\n", ""))
_W_ELSE = _w(_W_BODY.replace("        found[year] = None\n", "").replace(
    "                found[year] = code\n", "                found[year] = code\n            else:\n"
                                            "                found[year] = None\n"))
_W_NAME = _w(_W_BODY.replace("found[year] = code", "found[year] = entry"))
_W_PART = _w(_W_BODY.replace("if entry == name:", "if name in entry:"))
_W_RETURN = _w(_W_BODY.replace("    return found\n", "        return found\n"))


def _a_list(got, text) -> bool:
    return isinstance(got, (list, tuple))


# exercise 2: moves
_M_HEAD = ("def moves(old, new):\n"
           "    # for each name on both lists: how many places it rose from old to new, negative when it fell\n")
_M_START = _M_HEAD + "    ...\n"


def _m(body: str) -> str:
    return _M_HEAD + body


_M_BODY = ("    risen = {}\n"
           "    for new_code, name in new.items():\n"
           "        for old_code, old_name in old.items():\n"
           "            if old_name == name:\n"
           "                risen[name] = int(old_code[3:]) - int(new_code[3:])\n"
           "    return risen\n")
_M = _m(_M_BODY)
_M_SIGN = _m(_M_BODY.replace("int(old_code[3:]) - int(new_code[3:])", "int(new_code[3:]) - int(old_code[3:])"))
_M_CODE = _m(_M_BODY.replace("if old_name == name:", "if old_code == new_code:"))
_M_KEYS = _m(_M_BODY.replace("risen[name] =", "risen[new_code] ="))
_M_TEXT = _m(_M_BODY.replace("int(old_code[3:]) - int(new_code[3:])", "old_code[3:] - new_code[3:]"))
_M_WHOLE = _m(_M_BODY.replace("int(old_code[3:]) - int(new_code[3:])", "int(old_code) - int(new_code)"))


def _texts_minus(e) -> bool:
    return isinstance(e, TypeError) and "unsupported operand type(s) for -: 'str' and 'str'" in str(e)


def _letters_int(e) -> bool:
    return isinstance(e, ValueError) and re.search(r"invalid literal for int\(\).*'[A-Z]{3}\d\d'", str(e)) is not None


# exercise 3: findings
_F_HEAD = ("def findings(answers, entries, edition):\n"
           "    # one line for each entry whose answer is True, such as \"LLM01:2026 Prompt Injection\"\n")
_F_START = _F_HEAD + "    ...\n"


def _f(body: str) -> str:
    return _F_HEAD + body


_F_BODY = ("    lines = []\n"
           "    for code, yes in answers.items():\n"
           "        if yes:\n"
           "            lines.append(f\"{code}:{edition} {entries[code]}\")\n"
           "    return lines\n")
_F = _f(_F_BODY)
_F_ALL = _f(_F_BODY.replace("        if yes:\n            lines.append", "        lines.append"))
_F_NOT = _f(_F_BODY.replace("if yes:", "if not yes:"))
_F_NOEDITION = _f(_F_BODY.replace('f"{code}:{edition} {entries[code]}"', 'f"{code} {entries[code]}"'))
_F_CODES = _f(_F_BODY.replace('lines.append(f"{code}:{edition} {entries[code]}")', "lines.append(code)"))
_F_SPACE = _f(_F_BODY.replace('f"{code}:{edition} {entries[code]}"', 'f"{code}: {edition} {entries[code]}"'))


def _same_lines(got, want) -> bool:
    """The same lines, in any order."""
    if not isinstance(got, (list, tuple)):
        return False
    try:
        return sorted(str(x) for x in got) == sorted(want) and all(isinstance(x, str) for x in got)
    except TypeError:
        return False


def _moves_show(v) -> str:
    """{got} and {want} of moves: the first three names and how many there are; the picture shows them all."""
    if not isinstance(v, dict):
        return _short(v)
    def one(x):
        return f'"{x}"' if isinstance(x, str) else repr(x)
    first = ", ".join(f"{one(k)}: {one(x)}" for k, x in list(v.items())[:3])
    more = f", ... ({len(v)})" if len(v) > 3 else ""
    return "{" + first + more + "}"


def _lines_show(v) -> str:
    """{got} and {want}: the first two lines and how many there are."""
    if not isinstance(v, (list, tuple)):
        return _short(v)
    first = ", ".join(_short(x) if not isinstance(x, str) else '"' + x + '"' for x in list(v)[:2])
    more = f", ... ({len(v)})" if len(v) > 2 else ""
    return f"[{first}{more}]"


EXERCISES = {
    "where": FunctionBox(
        name="where",
        picture=where_picture,
        function="where",
        start=_W_START,
        solution=_W,
        visible=[(_WHERE_NAME, EDITIONS)],
        hidden=[("System Prompt Leakage", EDITIONS), ("Model Theft", EDITIONS), ("Prompt Injection", EDITIONS),
                ("Phishing", MADE_UP), ("Shadow IT", MADE_UP)],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("list", matches=_a_list,
                    en="`{call}` gives a list, and it should give a dict from year to id, such as `{{2023: None, 2025: \"LLM03\", ...}}`: with only the ids, you cannot see which one belongs to which edition.",
                    nl="`{call}` geeft een lijst, en het moet een dict van jaar naar id zijn, zoals `{{2023: None, 2025: \"LLM03\", ...}}`: met alleen de ids zie je niet welke bij welke editie hoort."),
            Mistake("no-none", wrong=_W_NONE,
                    en="`{call}` gives `{got}`, and it should give `{want}`. An edition without an entry by that name is missing from your dict. Give it `None`, so that every year of `editions` is in the dict, also when the name is not on that list.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Een editie zonder risico met die naam ontbreekt in je dict. Geef die editie `None`, zodat elk jaar van `editions` in de dict staat, ook als de naam niet op die lijst staat."),
            Mistake("else", wrong=_W_ELSE,
                    en="`{call}` gives `{got}`, and it should give `{want}`. Your `else` puts `None` back for every entry after the one that matched, so only a match at LLM10 stays. Give each edition `None` once, before the loop over its entries.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Je `else` zet `None` terug bij elk risico na het risico dat klopte, dus blijft alleen een treffer bij LLM10 staan. Geef elke editie één keer `None`, vóór de loop over de risico's."),
            Mistake("name", wrong=_W_NAME,
                    en="`{call}` gives `{got}`: the names, and it should give the ids, `{want}`. In `entries`, the id is the key and the name the value.",
                    nl="`{call}` geeft `{got}`: de namen, en het moeten de ids zijn, `{want}`. In `entries` is de id de key en de naam de waarde."),
            Mistake("part", wrong=_W_PART,
                    en="`{call}` gives `{got}`, and it should give `{want}`. Your function also finds an entry whose name only contains `name`, such as Supply Chain Vulnerabilities in 2023. A renamed entry is another name: compare the whole name.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Je functie vindt ook een risico waarvan de naam `name` alleen bevat, zoals *Supply Chain Vulnerabilities* in 2023. Een risico met een nieuwe naam heet anders: vergelijk de hele naam."),
            Mistake("return-early", wrong=_W_RETURN,
                    en="`{call}` gives only one edition: `{got}`. The `return` stands inside the loop over the editions, so the function stops after the first year. Put `return found` after the loops, as far in as `found = {{}}`.",
                    nl="`{call}` geeft maar één editie: `{got}`. De `return` staat in de loop over de edities, dus stopt de functie na het eerste jaar. Zet `return found` na de loops, even ver ingesprongen als `found = {{}}`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: make a dict, and for each edition put in the id of the entry called `name`, or `None`.",
                      "nl": "Vervang de `...`: maak een dict, en zet er voor elke editie de id in van het risico dat `name` heet, of `None`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. For each year: the id whose name is exactly `name`, or `None` when that edition has no such entry.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Voor elk jaar: de id waarvan de naam precies `name` is, of `None` als die editie zo'n risico niet heeft."},
            "almost": {"en": "Right for `where(\"Supply Chain\", EDITIONS)`. But `{call}` gives `{got}`, and it should give `{want}`. Does your function use the `name` and `editions` it gets?",
                       "nl": "Goed voor `where(\"Supply Chain\", EDITIONS)`. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Gebruikt je functie de `name` en `editions` die ze krijgt?"},
            "correct": {"en": "Your `where` is right in all {n} cases. Supply Chain is LLM03 in 2025 and LLM04 in 2026; 2023 has no entry by that name. The picture shades what it found.",
                        "nl": "Je `where` klopt in alle {n} gevallen. *Supply Chain* is LLM03 in 2025 en LLM04 in 2026; 2023 heeft geen risico met die naam. Het plaatje kleurt wat je functie vond."},
        },
    ),
    "moves": FunctionBox(
        name="moves",
        picture=moves_picture,
        function="moves",
        start=_M_START,
        solution=_M,
        visible=[(EDITIONS[2025], EDITIONS[2026])],
        hidden=[(EDITIONS[2023], EDITIONS[2025]), (EDITIONS[2026], EDITIONS[2025]), (MADE_UP[2024], MADE_UP[2025])],
        show=_moves_show,
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("texts", error=_texts_minus,
                    en="With `{call}`, your function subtracts two pieces of text, such as \"10\" and \"06\". Python cannot subtract text: make whole numbers of them with `int(...)` first.",
                    nl="Met `{call}` trekt je functie twee stukken tekst van elkaar af, zoals \"10\" en \"06\". Tekst aftrekken kan Python niet: maak er eerst hele getallen van met `int(...)`."),
            Mistake("whole-id", error=_letters_int,
                    en="With `{call}`, your function gives a whole id, such as LLM10, to `int`, and `int` cannot read the letters. Give it only the digits: the characters from the fourth on, `[3:]`.",
                    nl="Met `{call}` geeft je functie een hele id, zoals LLM10, aan `int`, en `int` kan de letters niet lezen. Geef alleen de cijfers: de tekens vanaf de vierde, `[3:]`."),
            Mistake("sign", wrong=_M_SIGN,
                    en="`{call}` gives every move the wrong way round: Unbounded Consumption went from LLM10 in 2025 to LLM06 in 2026, so it rose 4 places, and your function says −4. A rise is the place on the old list minus the place on the new one.",
                    nl="`{call}` draait elke stap om: *Unbounded Consumption* ging van LLM10 in 2025 naar LLM06 in 2026, dus 4 plaatsen omhoog, en je functie zegt −4. Omhoog is de plaats op de oude lijst min de plaats op de nieuwe."),
            Mistake("same-id", wrong=_M_CODE,
                    en="`{call}` gives 0 for every entry: your function looks up the same id on the old list. The same id can be a different entry in another edition, as LLM03 shows. Look up the same name.",
                    nl="`{call}` geeft 0 voor elk risico: je functie zoekt dezelfde id op de oude lijst op. Dezelfde id kan in een andere editie een ander risico zijn, zoals LLM03 laat zien. Zoek dezelfde naam op."),
            Mistake("id-keys", wrong=_M_KEYS,
                    en="`{call}` gives `{got}`: the ids of the new list as keys. Use the names as keys: an id can mean another entry in another edition.",
                    nl="`{call}` geeft `{got}`: de ids van de nieuwe lijst als keys. Gebruik de namen als keys: een id kan in een andere editie een ander risico zijn."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: for each entry of `new`, find the same name in `old`, and keep how many places it rose.",
                      "nl": "Vervang de `...`: zoek voor elk risico van `new` dezelfde naam in `old` op, en bewaar hoeveel plaatsen die omhoog ging."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. For each name on both lists: its place on `old` minus its place on `new`; the place is the number in the id.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Voor elke naam op beide lijsten: de plaats op `old` min de plaats op `new`; de plaats is het getal in de id."},
            "almost": {"en": "Right for `moves(EDITIONS[2025], EDITIONS[2026])`. But `{call}` gives `{got}`, and it should give `{want}`. Does your function use the `old` and `new` it gets?",
                       "nl": "Goed voor `moves(EDITIONS[2025], EDITIONS[2026])`. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Gebruikt je functie de `old` en `new` die ze krijgt?"},
            "correct": {"en": "Your `moves` is right in all {n} cases. From 2025 to 2026, Unbounded Consumption rose 4 places and Improper Output Handling fell 5. The picture draws every move.",
                        "nl": "Je `moves` klopt in alle {n} gevallen. Van 2025 naar 2026 ging *Unbounded Consumption* 4 plaatsen omhoog en *Improper Output Handling* 5 omlaag. Het plaatje tekent elke stap omhoog en omlaag."},
        },
    ),
    "findings": FunctionBox(
        name="findings",
        picture=findings_picture,
        function="findings",
        start=_F_START,
        solution=_F,
        visible=[(ANSWERS, EDITIONS[2026], 2026)],
        hidden=[(ANSWERS_2025, EDITIONS[2025], 2025), (GUIDE, EDITIONS[2026], 2026)],
        compare=_same_lines,
        show=_lines_show,
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("all", wrong=_F_ALL,
                    en="`{call}` gives a line for every entry, also for those whose answer is False. A review writes down only the entries where the answer is yes: keep only those.",
                    nl="`{call}` geeft een regel voor elk risico, ook voor de risico's met het antwoord False. Een review schrijft alleen de risico's op waar het antwoord ja is: houd alleen die over."),
            Mistake("not", wrong=_F_NOT,
                    en="`{call}` gives the entries whose answer is False, and it should give those whose answer is True: a yes means look closer.",
                    nl="`{call}` geeft de risico's met het antwoord False, en het moeten de risico's met het antwoord True zijn: een ja betekent beter kijken."),
            Mistake("no-edition", wrong=_F_NOEDITION,
                    en="`{call}` gives `{got}`: the lines leave out the edition. Write the year right after the id, with a colon, as in LLM01:2026: an id alone can mean another entry in another edition.",
                    nl="`{call}` geeft `{got}`: in de regels ontbreekt de editie. Schrijf het jaar direct na de id, met een dubbele punt, zoals in LLM01:2026: een id alleen kan in een andere editie een ander risico zijn."),
            Mistake("codes", wrong=_F_CODES,
                    en="`{call}` gives `{got}`: only the ids. Each line needs the edition and the name of the entry too, as in LLM01:2026 Prompt Injection, so that whoever reads it knows which risk is meant.",
                    nl="`{call}` geeft `{got}`: alleen de ids. Elke regel heeft ook de editie en de naam van het risico nodig, zoals in LLM01:2026 *Prompt Injection*, zodat wie het leest weet welk risico bedoeld is."),
            Mistake("space", wrong=_F_SPACE,
                    en="`{call}` gives `{got}`: there is a space between the colon and the year. Write the id and the edition without a space, as OWASP does: LLM01:2026.",
                    nl="`{call}` geeft `{got}`: er staat een spatie tussen de dubbele punt en het jaar. Schrijf de id en de editie zonder spatie, zoals OWASP doet: LLM01:2026."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: go through `answers`, and for each entry whose answer is True add a line with its id, the edition and its name.",
                      "nl": "Vervang de `...`: loop door `answers`, en zet voor elk risico met het antwoord True een regel in de lijst met de id, de editie en de naam."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. One line for each entry whose answer is True: id, colon, edition, a space and the name of the entry.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Eén regel voor elk risico met het antwoord True: id, dubbele punt, editie, een spatie en de naam van het risico."},
            "almost": {"en": "Right for `findings(ANSWERS, EDITIONS[2026], 2026)`. But `{call}` gives `{got}`, and it should give `{want}`. Does your function use the `answers`, `entries` and `edition` it gets?",
                       "nl": "Goed voor `findings(ANSWERS, EDITIONS[2026], 2026)`. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Gebruikt je functie de `answers`, `entries` en `edition` die ze krijgt?"},
            "correct": {"en": "Your `findings` is right in all {n} cases: 7 findings for the library's assistant, each with its id and edition. The picture puts them next to the library's answers.",
                        "nl": "Je `findings` klopt in alle {n} gevallen: 7 bevindingen voor de assistent van de bibliotheek, elk met id en editie. Het plaatje zet ze naast de antwoorden van de bibliotheek."},
        },
    ),
}

CASES = [
    ("where", _W_START, "empty", None),
    ("where", _W, "correct", None),
    ("where", _W_HEAD + "    found = {}\n    for year in editions:\n        found[year] = None\n"
              "        for code in editions[year]:\n            if editions[year][code] == name:\n"
              "                found[year] = code\n    return found\n", "correct", None),
    ("where", _W_NONE, "wrong", "no-none"),
    ("where", _W_ELSE, "wrong", "else"),
    ("where", _W_NAME, "wrong", "name"),
    ("where", _W_PART, "wrong", "part"),
    ("where", _W_RETURN, "wrong", "return-early"),
    ("where", _W_HEAD + "    found = []\n    for year, entries in editions.items():\n"
              "        for code, entry in entries.items():\n            if entry == name:\n"
              "                found.append(code)\n    return found\n", "wrong", "list"),
    ("where", _W_HEAD + "    return {2023: None, 2025: \"LLM03\", 2026: \"LLM04\"}\n", "almost", None),
    ("where", _W_HEAD + "    entries = EDITIONS[2026]\n    return {}\n", "error", "lesson-names"),
    ("where", _W_HEAD + "    return {}\n", "wrong", None),
    ("moves", _M_START, "empty", None),
    ("moves", _M, "correct", None),
    ("moves", _M_HEAD + "    places = {}\n    for code, name in old.items():\n        places[name] = int(code[3:])\n"
              "    risen = {}\n    for code, name in new.items():\n        if name in places:\n"
              "            risen[name] = places[name] - int(code[3:])\n    return risen\n", "correct", None),
    ("moves", _M_SIGN, "wrong", "sign"),
    ("moves", _M_CODE, "wrong", "same-id"),
    ("moves", _M_KEYS, "wrong", "id-keys"),
    ("moves", _M_TEXT, "error", "texts"),
    ("moves", _M_WHOLE, "error", "whole-id"),
    ("moves", _M_HEAD + "    return " + repr(moves(EDITIONS[2025], EDITIONS[2026])) + "\n", "almost", None),
    ("moves", _M_HEAD + "    return {}\n", "wrong", None),
    ("findings", _F_START, "empty", None),
    ("findings", _F, "correct", None),
    ("findings", _F_HEAD + "    lines = []\n    for code in entries:\n        if answers[code]:\n"
                 "            lines.append(code + \":\" + str(edition) + \" \" + entries[code])\n    return lines\n",
     "correct", None),
    ("findings", _F_ALL, "wrong", "all"),
    ("findings", _F_NOT, "wrong", "not"),
    ("findings", _F_NOEDITION, "wrong", "no-edition"),
    ("findings", _F_CODES, "wrong", "codes"),
    ("findings", _F_SPACE, "wrong", "space"),
    ("findings", _F_HEAD + "    return " + repr(findings(ANSWERS, EDITIONS[2026], 2026)) + "\n", "almost", None),
    ("findings", _F_HEAD + "    entries = EDITIONS[2026]\n    return []\n", "error", "lesson-names"),
    ("findings", _F_HEAD + "    return []\n", "wrong", None),
]
