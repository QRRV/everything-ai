# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-kinds-of-ai-risk: sorting risks of AI into the three categories of the International AI
Safety Report, and finding which domains of the MIT AI Risk Repository a list of risks leaves out. The cases are made
up by the course, each a short example of a kind of risk the report names."""
import contextlib
import copy
import html
import io
import re

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

# --- the course's cases ----------------------------------------------------------------------------------------------

# Made-up cases, one line each: someone meant to cause harm (harm_meant), and the AI system did what its developer or
# user intended (worked). Each matches an example the International AI Safety Report (2025) lists under a category.
RISKS = [
    {"text": "Someone has a chat model write hundreds of fake bad reviews of a rival's shop.",
     "harm_meant": True, "worked": True},
    {"text": "A chatbot on a bike shop's website promises a refund the shop does not offer.",
     "harm_meant": False, "worked": False},
    {"text": "Translators get fewer jobs as more clients use machine translation.",
     "harm_meant": False, "worked": True},
    {"text": "A model that sorts job applications ranks those of women lower, although nobody wanted that.",
     "harm_meant": False, "worked": False},
    {"text": "Someone uses a cloned voice of a manager to have staff transfer money.",
     "harm_meant": True, "worked": True},
    {"text": "Many apps build on the models of a few providers, which gain a lot of power over the market.",
     "harm_meant": False, "worked": True},
    {"text": "An AI agent asked to tidy up a folder deletes files that were still needed.",
     "harm_meant": False, "worked": False},
    {"text": "The data centers that train and run AI models use more and more electricity.",
     "harm_meant": False, "worked": True},
    {"text": "Someone has a chat model write thousands of posts that push one side before an election.",
     "harm_meant": True, "worked": True},
]
RISKS_NL = [
    "Iemand laat een chat model honderden valse slechte reviews schrijven over de winkel van een concurrent.",
    "Een chatbot op de website van een fietsenwinkel belooft geld terug dat de winkel niet geeft.",
    "Vertalers krijgen minder opdrachten nu meer klanten machinevertaling gebruiken.",
    "Een model dat sollicitaties sorteert, zet die van vrouwen lager, hoewel niemand dat wilde.",
    "Iemand gebruikt een gekloonde stem van een manager om personeel geld te laten overmaken.",
    "Veel apps bouwen op de modellen van een paar aanbieders, die zo veel macht over de markt krijgen.",
    "Een AI-agent die een map moest opruimen, verwijdert bestanden die nog nodig waren.",
    "De datacenters die AI-modellen trainen en draaien, gebruiken steeds meer stroom.",
    "Iemand laat een chat model duizenden berichten schrijven die voor een verkiezing één kant op duwen.",
]
# what the report calls the example each case stands for (2025, executive summary, pp. 18 to 20)
REPORT_EXAMPLES = ["fake content", "reliability issues", "labour market risks", "bias", "fake content",
                   "market concentration", "reliability issues", "environmental risks",
                   "manipulation of public opinion"]

# The seven domains of the Domain Taxonomy of the MIT AI Risk Repository (Slattery and others, v3, 2026, p. 22)
DOMAINS = ["Discrimination & toxicity", "Privacy & security", "Misinformation", "Malicious actors & misuse",
           "Human-computer interaction", "Socioeconomic & environmental harms", "AI system safety, failures & limitations"]
DOMAINS_NL = ["Discriminatie en toxiciteit", "Privacy en security", "Misinformatie", "Kwaadwillenden en misbruik",
              "Interactie tussen mens en computer", "Sociaal-economische schade en milieuschade",
              "Veiligheid, falen en beperkingen van AI-systemen"]
# the risks a made-up team that builds a chatbot for a bike shop wrote down, by domain
CHECKLIST = ["Privacy & security", "Misinformation", "Discrimination & toxicity"]

KINDS = ["malicious use", "malfunction", "systemic"]
# what comes closest to each of the three among the kinds of Zwetsloot and Dafoe (2019); the two sortings are not the same
OTHER_NAMES = {"malicious use": "misuse", "malfunction": "accident", "systemic": "structure"}


def iasr_category(case: dict) -> str:
    """The course's own Exercise 1: did someone mean harm, and else, did the system do what was intended?"""
    if case["harm_meant"]:
        return "malicious use"
    elif case["worked"]:
        return "systemic"
    return "malfunction"


def missing(covered: list, domains: list) -> list:
    """The course's own Exercise 2: the domains, in their order, that the list `covered` leaves out."""
    left = []
    for domain in domains:
        if domain not in covered:
            left.append(domain)
    return left


class Named:
    """A value the check gives the learner's function, shown in a message by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


RISKS_N = [Named(f"RISKS[{i}]", r) for i, r in enumerate(RISKS)]
DOMAINS_N = Named("DOMAINS", DOMAINS)
CHECKLIST_N = Named("CHECKLIST", CHECKLIST)


def case_text(i: int, lang: str) -> str:
    return RISKS_NL[i] if lang == "nl" else RISKS[i]["text"]


def case_options(lang: str) -> dict:
    """The cases for the demo's dropdown: a label in the lesson's language for each index."""
    return {f"{i + 1}. {case_text(i, lang)}": i for i in range(len(RISKS))}


# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.15rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;color:var(--ea-ink);vertical-align:top")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"


def _note(text: str) -> str:
    return f'<div style="{_UI}">{html.escape(text)}</div>'


def _word(text: str) -> str:
    """A short value as Python writes it, kept on one line in a table cell."""
    return f'<span style="font-family:var(--ea-font-code);font-size:12px;white-space:nowrap">{html.escape(text)}</span>'


def _table(rows: list, head: list) -> str:
    """A table with a head row; each row is a list of (html, style)."""
    top = "".join(f'<th style="{_HEAD}">{html.escape(h)}</th>' for h in head)
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{";" + style if style else ""}">{cell}</td>'
                                    for cell, style in row) + "</tr>" for row in rows)
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.3rem 0">'
            f'<tr>{top}</tr>{body}</table></div>')


_REPORT_WORDS = {
    "en": {"title": "The three categories of the International AI Safety Report (2025)",
           "heads": ["Malicious use", "Malfunctions", "Systemic risks"],
           "what": ["Someone uses AI to cause harm on purpose.",
                    "AI does not do what its developer or user intended.",
                    "Harm from how AI changes society, not directly from what AI can do."],
           "items": [["harm to individuals through fake content", "manipulation of public opinion", "cyber offence",
                      "biological and chemical attacks"],
                     ["reliability issues", "bias", "loss of control"],
                     ["labour market risks", "global AI R&D divide", "market concentration and single points of failure",
                      "environmental risks", "privacy risks", "copyright infringements"]],
           "note": "The headings of the report's executive summary, pp. 17 to 20. Its 2026 edition keeps the three "
                   "categories, but no longer addresses issues such as bias, the environment, privacy and copyright."},
    "nl": {"title": "De drie categorieën van het International AI Safety Report (2025)",
           "heads": ["Kwaadwillig gebruik", "Falen", "Systeemrisico's"],
           "what": ["Iemand gebruikt AI om met opzet schade te doen.",
                    "AI doet niet wat de ontwikkelaar of de gebruiker bedoelde.",
                    "Schade doordat AI de samenleving verandert, niet rechtstreeks door wat AI kan."],
           "items": [["schade aan personen door nepinhoud", "beïnvloeding van de publieke opinie", "cyberaanvallen",
                      "biologische en chemische aanvallen"],
                     ["onbetrouwbaarheid", "bias", "verlies van controle"],
                     ["risico's voor de arbeidsmarkt", "wereldwijde kloof in AI-onderzoek", "marktconcentratie en kwetsbare knooppunten",
                      "risico's voor het milieu", "risico's voor privacy", "inbreuk op auteursrecht"]],
           "note": "De kopjes van de samenvatting van het rapport, blz. 17 tot 20. De editie van 2026 houdt dezelfde drie "
                   "categorieën, maar behandelt onderwerpen als bias, milieu, privacy en auteursrecht niet meer."},
}


def report_picture(lang: str) -> str:
    """The three categories of the report, side by side, with the examples it lists under each."""
    w = _REPORT_WORDS[lang]
    cols = []
    for head, what, items in zip(w["heads"], w["what"], w["items"]):
        lis = "".join(f'<li>{html.escape(x)}</li>' for x in items)
        cols.append(f'<div style="flex:1 1 12rem;min-width:11rem;border:1px solid var(--ea-line);border-radius:6px;'
                    f'padding:0.5rem 0.6rem;font-family:var(--ea-font-ui);font-size:0.85rem;color:var(--ea-ink)">'
                    f'<div style="font-weight:600">{html.escape(head)}</div>'
                    f'<div style="color:var(--ea-ink-2);margin:0.2rem 0 0.3rem">{html.escape(what)}</div>'
                    f'<ul style="margin:0;padding-left:1.1rem">{lis}</ul></div>')
    body = f'<div style="display:flex;flex-wrap:wrap;gap:0.5rem">{"".join(cols)}</div>' + _note(w["note"])
    alt = w["title"] + ". " + " ".join(f"{h}: {', '.join(i)}." for h, i in zip(w["heads"], w["items"]))
    return yours.figure(w["title"], body, alt=alt)


_TREE_WORDS = {
    "en": {"title": "Two questions sort a case",
           "q1": "Did someone mean to cause harm?", "q2": "Did the system do what was intended?",
           "yes": "yes", "no": "no",
           "leaves": {"malicious use": "malicious use", "malfunction": "malfunction", "systemic": "systemic risk"},
           "other": ["closest to {name}", "(Zwetsloot and Dafoe)"]},
    "nl": {"title": "Twee vragen delen een geval in",
           "q1": "Wilde iemand schade doen?", "q2": "Deed het systeem wat bedoeld was?",
           "yes": "ja", "no": "nee",
           "leaves": {"malicious use": "kwaadwillig gebruik", "malfunction": "falen", "systemic": "systeemrisico"},
           "other": ["het dichtst bij {name}", "(Zwetsloot en Dafoe)"]},
}
_OTHER_NL = {"misuse": "misbruik", "accident": "ongeluk", "structure": "structuur"}


def _tree(path: str | None, lang: str) -> str:
    """The two questions as a tree; `path` is the category a case ends in, drawn in ink, the rest in grey."""
    w = _TREE_WORDS[lang]
    W, H = 560, 262
    on = "var(--ea-ink)"
    off = "var(--ea-ink-3)" if path else "var(--ea-ink-2)"   # with no case on it, the whole tree is plain to read

    def colour(*cats):
        return on if path in cats else off

    def box(x, y, text, stroke, bold=False):
        return (f'<rect x="{x - 125}" y="{y - 16}" width="250" height="32" rx="6" style="fill:none;stroke:{stroke};'
                f'stroke-width:{2 if stroke == on else 1}"/><text x="{x}" y="{y + 5}" style="font-size:13px;fill:{stroke};'
                f'text-anchor:middle;font-weight:{600 if bold else 400}">{html.escape(text)}</text>')

    def leaf(x, y, cat):
        name = OTHER_NAMES[cat] if lang == "en" else _OTHER_NL[OTHER_NAMES[cat]]
        c = colour(cat)
        return (f'<text x="{x}" y="{y}" style="font-size:13px;fill:{c};text-anchor:middle;font-weight:'
                f'{700 if path == cat else 500}">{html.escape(w["leaves"][cat])}</text>'
                + "".join(f'<text x="{x}" y="{y + 16 + 14 * k}" style="font-size:11px;fill:{c};text-anchor:middle">'
                          f'{html.escape(part.format(name=name))}</text>' for k, part in enumerate(w["other"])))

    def line(x1, y1, x2, y2, c, label, lx, ly):
        return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" style="stroke:{c};stroke-width:{2 if c == on else 1}"/>'
                f'<text x="{lx}" y="{ly}" style="font-size:11px;fill:{c};text-anchor:middle">{html.escape(label)}</text>')

    p = [box(280, 30, w["q1"], colour(*KINDS) if path else off, bold=True),
         line(220, 46, 120, 120, colour("malicious use"), w["yes"], 150, 80),
         line(340, 46, 400, 104, colour("malfunction", "systemic"), w["no"], 385, 72),
         leaf(110, 140, "malicious use"),
         box(400, 120, w["q2"], colour("malfunction", "systemic"), bold=True),
         line(340, 136, 300, 196, colour("malfunction"), w["no"], 305, 166),
         line(460, 136, 490, 196, colour("systemic"), w["yes"], 492, 166),
         leaf(290, 216, "malfunction"), leaf(470, 216, "systemic")]
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(w["title"])}" style="width:100%;'
            f'max-width:{W}px;height:auto;display:block;font-family:var(--ea-font-ui)">{"".join(p)}</svg>')


_DEMO_WORDS = {
    "en": {"title": "Case {i}", "meant": "Someone meant to cause harm: {v}.", "worked": "The system did what was intended: {v}.",
           "yes": "yes", "no": "no", "lands": "The report's example: {example}. Category: {cat}.",
           "examples": {"fake content": "fake content", "reliability issues": "reliability issues",
                        "labour market risks": "labour market risks", "bias": "bias",
                        "market concentration": "market concentration", "environmental risks": "environmental risks",
                        "manipulation of public opinion": "manipulation of public opinion"}},
    "nl": {"title": "Geval {i}", "meant": "Iemand wilde schade doen: {v}.", "worked": "Het systeem deed wat bedoeld was: {v}.",
           "yes": "ja", "no": "nee", "lands": "Het voorbeeld van het rapport: {example}. Categorie: {cat}.",
           "examples": {"fake content": "nepinhoud", "reliability issues": "onbetrouwbaarheid",
                        "labour market risks": "risico's voor de arbeidsmarkt", "bias": "bias",
                        "market concentration": "marktconcentratie", "environmental risks": "risico's voor het milieu",
                        "manipulation of public opinion": "beïnvloeding van de publieke opinie"}},
}


def demo_picture(i: int, lang: str) -> str:
    """One case of the course: its two answers, and the path it takes through the two questions."""
    i = max(0, min(int(i), len(RISKS) - 1))
    w, t = _DEMO_WORDS[lang], _TREE_WORDS[lang]
    case = RISKS[i]
    cat = iasr_category(case)
    lines = [w["meant"].format(v=w["yes"] if case["harm_meant"] else w["no"]),
             w["worked"].format(v=w["yes"] if case["worked"] else w["no"]),
             w["lands"].format(example=w["examples"][REPORT_EXAMPLES[i]], cat=t["leaves"][cat])]
    title = w["title"].format(i=i + 1)
    body = (f'<div style="font-family:var(--ea-font-ui);font-size:0.9rem;color:var(--ea-ink);margin-bottom:0.3rem">'
            f'{html.escape(case_text(i, lang))}</div>' + _tree(cat, lang) + "".join(_note(x) for x in lines))
    return yours.figure(title, body, alt=title + ". " + case_text(i, lang) + " " + " ".join(lines))


def tree_picture(lang: str) -> str:
    """The two questions with no case on them, for the recipe."""
    return yours.figure(_TREE_WORDS[lang]["title"], _tree(None, lang), alt=_TREE_WORDS[lang]["title"] + ". "
                        + _TREE_WORDS[lang]["q1"] + " " + _TREE_WORDS[lang]["q2"])


_MIT_WORDS = {
    "en": {"title": "The seven domains of the MIT AI Risk Repository",
           "note": "1,725 risks from 74 documents, sorted into 7 domains and 24 subdomains (version of March 2026). "
                   "One document covers 8 of the 24 subdomains on average, from 1 to 20."},
    "nl": {"title": "De zeven domeinen van de MIT AI Risk Repository",
           "note": "1725 risico's uit 74 documenten, ingedeeld in 7 domeinen en 24 subdomeinen (versie van maart 2026). "
                   "Eén document dekt gemiddeld 8 van de 24 subdomeinen, van 1 tot 20."},
}


def _tiles(lang: str, marks: dict | None = None) -> str:
    """The seven domains as tiles; `marks` maps a domain to (label, style) drawn under its name."""
    names = DOMAINS_NL if lang == "nl" else DOMAINS
    tiles = []
    for key, name in zip(DOMAINS, names):
        label, style = (marks or {}).get(key, ("", ""))
        tiles.append(f'<div style="flex:1 1 9.5rem;min-width:9rem;border:1px solid var(--ea-line);border-radius:6px;'
                     f'padding:0.4rem 0.5rem;font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink);{style}">'
                     f'<div>{html.escape(name)}</div>'
                     + (f'<div style="color:var(--ea-ink-2);font-size:0.75rem;margin-top:0.2rem">{html.escape(label)}</div>'
                        if label else "") + '</div>')
    return f'<div style="display:flex;flex-wrap:wrap;gap:0.4rem">{"".join(tiles)}</div>'


def mit_picture(lang: str) -> str:
    """The seven domains of the repository, with its numbers."""
    w = _MIT_WORDS[lang]
    names = DOMAINS_NL if lang == "nl" else DOMAINS
    return yours.figure(w["title"], _tiles(lang) + _note(w["note"]), alt=w["title"] + ": " + "; ".join(names) + ". " + w["note"])


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


def _shown(v, lang: str) -> str:
    """What the learner's function gave, for a cell of a table, as Python writes it."""
    if v is _STOPPED:
        return {"en": "error", "nl": "fout"}[lang]
    text = repr(v)
    return text if len(text) <= 40 else text[:37] + "..."


_CHECK_WORDS = {
    "en": {"category": {"title": "Your iasr_category on the nine cases",
                        "head": ["case", "harm meant", "worked", "yours", "the course's"],
                        "fine": "The same as the course's for all nine cases: {counts}.",
                        "differs": "It differs from the course's for {k} of the nine cases: they have a dashed frame."},
           "missing": {"title": "Your missing(CHECKLIST, DOMAINS)",
                       "listed": "on the team's list", "yours": "yours: missing", "course": "the course's: missing",
                       "fine": "The same as the course's: the team's list leaves out 4 of the 7 domains.",
                       "differs": "It differs from the course's for {k} of the seven domains: they have a dashed frame.",
                       "not_list": "Your missing gives {got}, not a list of domains: the tiles show the course's answer."},
           "yes": "yes", "no": "no", "alt": "{title}: {rows}. {after}"},
    "nl": {"category": {"title": "Jouw iasr_category op de negen gevallen",
                        "head": ["geval", "schade bedoeld", "werkte", "de jouwe", "die van de course"],
                        "fine": "Bij alle negen gevallen hetzelfde als die van de course: {counts}.",
                        "differs": "Bij {k} van de negen gevallen anders dan die van de course: die hebben een gestreept kader."},
           "missing": {"title": "Jouw missing(CHECKLIST, DOMAINS)",
                       "listed": "op de lijst van het team", "yours": "de jouwe: ontbreekt",
                       "course": "die van de course: ontbreekt",
                       "fine": "Hetzelfde als die van de course: de lijst van het team laat 4 van de 7 domeinen weg.",
                       "differs": "Bij {k} van de zeven domeinen anders dan die van de course: die hebben een gestreept kader.",
                       "not_list": "Jouw missing geeft {got}, geen lijst van domeinen: de tegels tonen het antwoord van de course."},
           "yes": "ja", "no": "nee", "alt": "{title}: {rows}. {after}"},
}


def category_picture(seen) -> str:
    """The nine cases, with the learner's category beside the course's."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _CHECK_WORDS[seen.lang]
    words = w["category"]
    rows, said, k = [], [], 0
    for i, case in enumerate(RISKS):
        got, want = _call(fn, case), iasr_category(case)
        differs = got != want
        k += differs
        rows.append([(html.escape(case_text(i, seen.lang)), ""),
                     (html.escape(w["yes"] if case["harm_meant"] else w["no"]), ""),
                     (html.escape(w["yes"] if case["worked"] else w["no"]), ""),
                     (_word(_shown(got, seen.lang)), _OFF if differs else ""), (_word(repr(want)), "")])
        said.append(f"{i + 1}: {_shown(got, seen.lang)}, {want!r}")
    counts = ", ".join(f"{sum(iasr_category(c) == kind for c in RISKS)} {kind!r}" for kind in KINDS)
    after = words["differs"].format(k=k) if k else words["fine"].format(counts=counts)
    return yours.figure(words["title"], _table(rows, words["head"]) + _note(after),
                        alt=w["alt"].format(title=words["title"], rows="; ".join(said), after=after))


def missing_picture(seen) -> str:
    """The seven domains: which the team listed, which the learner's missing gives, and which the course's gives."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _CHECK_WORDS[seen.lang]
    words = w["missing"]
    got, want = _call(fn, CHECKLIST, DOMAINS), missing(CHECKLIST, DOMAINS)
    marks, k = {}, 0
    if isinstance(got, list):
        for d in DOMAINS:
            labels = [words["listed"]] if d in CHECKLIST else []
            if d in got:
                labels.append(words["yours"])
            if d in want:
                labels.append(words["course"])
            differs = (d in got) != (d in want)
            k += differs
            marks[d] = (" · ".join(labels), _OFF if differs else "")
        after = words["differs"].format(k=k) if k else words["fine"]
    else:
        for d in DOMAINS:
            marks[d] = (words["listed"] if d in CHECKLIST else words["course"], "")
        after = words["not_list"].format(got=_shown(got, seen.lang))
    names = DOMAINS_NL if seen.lang == "nl" else DOMAINS
    said = "; ".join(f"{n}: {marks[d][0]}" for d, n in zip(DOMAINS, names))
    return yours.figure(words["title"], _tiles(seen.lang, marks) + _note(after),
                        alt=w["alt"].format(title=words["title"], rows=said, after=after))


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["iasr-category", "missing", "uber", "overlap"]

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
    return isinstance(e, NameError) and re.search(r"'(RISKS|DOMAINS|CHECKLIST|C)'", str(e)) is not None


_NAMES = Mistake("lesson-names", error=_names_error,
                 en="Your function uses a name of the lesson, such as `RISKS`, `DOMAINS` or `C`, and the check does not give it that name: use the names between the parentheses of the `def` line.",
                 nl="Je functie gebruikt een naam van de les, zoals `RISKS`, `DOMAINS` of `C`, en de check geeft die naam niet mee: gebruik de namen tussen de haakjes van de `def`-regel.")

# exercise 1: iasr_category; the last case is someone who meant harm with a system that did not do what was asked
CATEGORY_CASES = [(r,) for r in RISKS_N] + [({"harm_meant": True, "worked": False},)]
_C_START = "def iasr_category(case):\n    ...\n"
_C = ("def iasr_category(case):\n"
      "    if case[\"harm_meant\"]:\n"
      "        return \"malicious use\"\n"
      "    elif case[\"worked\"]:\n"
      "        return \"systemic\"\n"
      "    return \"malfunction\"\n")
_C_ORDER = ("def iasr_category(case):\n"
            "    if case[\"worked\"]:\n"
            "        return \"systemic\"\n"
            "    elif case[\"harm_meant\"]:\n"
            "        return \"malicious use\"\n"
            "    return \"malfunction\"\n")
_C_SWAP = _C.replace("\"systemic\"", "\"MALF\"").replace("return \"malfunction\"", "return \"systemic\"").replace("\"MALF\"", "\"malfunction\"")
_C_TWO = ("def iasr_category(case):\n"
          "    if case[\"harm_meant\"]:\n"
          "        return \"malicious use\"\n"
          "    return \"malfunction\"\n")

# exercise 2: missing
MISSING_CASES = [(CHECKLIST_N, DOMAINS_N), ([], DOMAINS_N), (DOMAINS_N, DOMAINS_N), (["b"], ["a", "b", "c"]),
                 (["c", "x"], ["a", "b", "c"])]
_M_START = "def missing(covered, domains):\n    ...\n"
_M = ("def missing(covered, domains):\n"
      "    left = []\n"
      "    for domain in domains:\n"
      "        if domain not in covered:\n"
      "            left.append(domain)\n"
      "    return left\n")
_M_COVERED = _M.replace("if domain not in covered", "if domain in covered")
_M_COUNT = _M.replace("return left", "return len(left)")
_M_SWAP = ("def missing(covered, domains):\n"
           "    left = []\n"
           "    for domain in covered:\n"
           "        if domain not in domains:\n"
           "            left.append(domain)\n"
           "    return left\n")


EXERCISES = {
    "iasr-category": FunctionBox(
        name="iasr-category",
        picture=category_picture,
        function="iasr_category",
        start=_C_START,
        solution=_C,
        visible=[CATEGORY_CASES[0]],
        hidden=CATEGORY_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("order", wrong=_C_ORDER,
                    en="`{call}` gives {got}, and it should give {want}: someone meant to cause harm there, and the system did what that person wanted. Your function asks first whether the system worked; ask first whether someone meant harm, since misuse can use a system that works fine.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: daar wilde iemand schade doen, en het systeem deed wat die persoon wilde. Jouw functie vraagt eerst of het systeem werkte; vraag eerst of iemand schade wilde doen, want misbruik kan een systeem gebruiken dat prima werkt."),
            Mistake("swap", wrong=_C_SWAP,
                    en="`{call}` gives {got}, and it should give {want}: there nobody meant harm, and the system did not do what was intended. That is a malfunction; a systemic risk comes from systems that do work as intended.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: daar wilde niemand schade doen, en deed het systeem niet wat bedoeld was. Dat is falen; een systeemrisico komt van systemen die wel doen wat bedoeld is."),
            Mistake("two", wrong=_C_TWO,
                    en="`{call}` gives {got}, and it should give {want}: there nobody meant harm, and the system did what was intended, yet harm comes from how it is used across society. Your function knows only two kinds; ask the second question too.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: daar wilde niemand schade doen, en deed het systeem wat bedoeld was, maar komt er toch schade van hoe het in de samenleving gebruikt wordt. Jouw functie kent maar twee soorten; stel ook de tweede vraag."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: ask whether someone meant to cause harm, and else whether the system did what was intended.",
                      "nl": "Vervang de `...`: vraag of iemand schade wilde doen, en anders of het systeem deed wat bedoeld was."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The function gives \"malicious use\", \"malfunction\" or \"systemic\", from the values under the keys `\"harm_meant\"` and `\"worked\"`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De functie geeft \"malicious use\", \"malfunction\" of \"systemic\", op grond van de waarden onder de keys `\"harm_meant\"` en `\"worked\"`."},
            "almost": {"en": "Right for `RISKS[0]`. But `{call}` gives {got}, and it should give {want}. Does your function look at the `case` it gets?",
                       "nl": "Goed voor `RISKS[0]`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Kijkt je functie naar de `case` die ze krijgt?"},
            "correct": {"en": "Your `iasr_category` is right in all {n} cases. Of the course's nine, three are malicious use, three malfunctions and three systemic risks. The model that ranks women lower is a malfunction for the report of 2025: bias sits next to reliability issues and loss of control.",
                        "nl": "Je `iasr_category` klopt in alle {n} gevallen. Van de negen van de course zijn er drie kwaadwillig gebruik, drie falen en drie systeemrisico's. Het model dat vrouwen lager zet, is voor het rapport van 2025 falen: bias staat naast onbetrouwbaarheid en verlies van controle."},
        },
    ),
    "missing": FunctionBox(
        name="missing",
        picture=missing_picture,
        function="missing",
        start=_M_START,
        solution=_M,
        visible=[MISSING_CASES[0]],
        hidden=MISSING_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("covered", wrong=_M_COVERED,
                    en="`{call}` gives {got}, and it should give {want}: those are the domains the list covers. The exercise asks for the ones it leaves out.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat zijn de domeinen die de lijst wel dekt. De oefening vraagt om de domeinen die ze weglaat."),
            Mistake("count", wrong=_M_COUNT,
                    en="`{call}` gives {got}, and it should give {want}: that is how many domains are missing. The exercise asks for the domains themselves, as a list.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is hoeveel domeinen er ontbreken. De oefening vraagt om de domeinen zelf, als lijst."),
            Mistake("swap", wrong=_M_SWAP,
                    en="`{call}` gives {got}, and it should give {want}: your loop goes over `covered`, so it finds what the list holds beyond the domains. Go over `domains`, and keep each one that is not in `covered`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw loop gaat over `covered`, dus hij vindt wat de lijst bevat buiten de domeinen. Ga over `domains`, en houd elk domein dat niet in `covered` staat."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: go over the domains, and keep each one that the list `covered` does not hold.",
                      "nl": "Vervang de `...`: ga de domeinen langs, en houd elk domein dat niet in de lijst `covered` staat."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The function gives the domains of `domains`, in their order, that are not in `covered`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De functie geeft de domeinen van `domains`, in hun volgorde, die niet in `covered` staan."},
            "almost": {"en": "Right for the team's list. But `{call}` gives {got}, and it should give {want}. Does your function use the `covered` and `domains` it gets?",
                       "nl": "Goed voor de lijst van het team. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `covered` en `domains` die ze krijgt?"},
            "correct": {"en": "Your `missing` is right in all {n} cases. The team thought of privacy, misinformation and discrimination, and left out 4 of the 7 domains, among them malicious actors and the safety of the system itself. A taxonomy is a checklist: it shows what nobody thought of.",
                        "nl": "Je `missing` klopt in alle {n} gevallen. Het team dacht aan privacy, misinformatie en discriminatie, en liet 4 van de 7 domeinen weg, waaronder kwaadwillenden en de veiligheid van het systeem zelf. Een taxonomie is een checklist: ze laat zien waar niemand aan dacht."},
        },
    ),
}

CASES = [
    ("iasr-category", _C_START, "empty", None),
    ("iasr-category", _C, "correct", None),
    ("iasr-category", "def iasr_category(case):\n    if case['harm_meant']:\n        return 'malicious use'\n    if not case['worked']:\n        return 'malfunction'\n    return 'systemic'\n", "correct", None),
    ("iasr-category", _C_ORDER, "wrong", "order"),
    ("iasr-category", _C_SWAP, "wrong", "swap"),
    ("iasr-category", _C_TWO, "wrong", "two"),
    ("iasr-category", "def iasr_category(case):\n    return 'malicious use'\n", "almost", None),
    ("iasr-category", "def iasr_category(case):\n    return RISKS[0]\n", "error", "lesson-names"),
    ("missing", _M_START, "empty", None),
    ("missing", _M, "correct", None),
    ("missing", "def missing(covered, domains):\n    return [d for d in domains if d not in covered]\n", "correct", None),
    ("missing", _M_COVERED, "wrong", "covered"),
    ("missing", _M_COUNT, "wrong", "count"),
    ("missing", _M_SWAP, "wrong", "swap"),
    ("missing", f"def missing(covered, domains):\n    return {missing(CHECKLIST, DOMAINS)!r}\n", "almost", None),
    ("missing", "def missing(covered, domains):\n    return DOMAINS\n", "error", "lesson-names"),
]
