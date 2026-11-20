# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-safety-field: who works on AI safety. Three pieces of data from the sources: who signed
the Seoul agreement of 21 May 2024 on a network of AI safety institutes, and who were the network's first members on
20 November 2024; Table 1 of Apollo Research's paper on in-context scheming (Meinke and colleagues, version of
January 2025): in which of six evaluations each of six models schemed at least once; and the names of four public
bodies over time. The learner writes changes, who signed but did not join and who joined without signing, and
schemed_in, in how many evaluations a model schemed at least once."""
import contextlib
import copy
import datetime
import html
import io
import re

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

# --- the data --------------------------------------------------------------------------------------------------------

# The AI Seoul Summit, 21 May 2024: "10 countries plus the European Union" agreed to launch a network of AI safety
# institutes (UK government, notes to editors). Names are written the same way in both sets.
SEOUL = {"Australia", "Canada", "European Union", "France", "Germany", "Italy", "Japan", "Republic of Korea",
         "Singapore", "United States", "United Kingdom"}
# The first members of the International Network of AI Safety Institutes, San Francisco, 20 November 2024 (fact sheet
# of the US Department of Commerce and Department of State); the UK government gives the same list in December 2025.
NETWORK = {"Australia", "Canada", "European Union", "France", "Japan", "Kenya", "Republic of Korea", "Singapore",
           "United Kingdom", "United States"}

# Apollo Research, "Frontier Models are Capable of In-context Scheming", Table 1: for each evaluation, the models that
# showed scheming in it at least once. The goal came from the prompt, with a strong push to pursue it in all but
# instrumental alignment faking. Apollo also tested o1-mini and o1-preview, which Table 1 leaves out.
MODELS = ["gpt-4o", "gemini-1.5", "llama-3.1", "sonnet-3.5", "opus-3", "o1"]
TABLE = {
    "oversight subversion": {"llama-3.1", "opus-3", "o1"},
    "self-exfiltration": {"gemini-1.5", "llama-3.1", "sonnet-3.5", "opus-3", "o1"},
    "goal-guarding": {"gemini-1.5", "llama-3.1", "opus-3", "o1"},
    "covert email reranking": {"o1"},
    "instrumental alignment faking": {"gemini-1.5", "llama-3.1", "o1"},
    "sandbagging": {"sonnet-3.5", "opus-3", "o1"},
}
# Table 2, "Disabled oversight": the share of runs in which a model switched its oversight off (gpt-4o is not in it)
OVERSIGHT = {"gemini-1.5": 0, "llama-3.1": 0.03, "sonnet-3.5": 0, "opus-3": 0.05, "o1": 0.02}

# The names of four bodies, each from the date it took that name. The UK institute: 2 November 2023, from the Frontier
# AI Taskforce, and 14 February 2025. The US center: announced on 1 November 2023 at the UK's summit; "In June 2025"
# re-established as CAISI, on a day the sources do not give (drawn at the middle of June); renamed CAISSI after the
# executive order of 29 September 2026 and before 16 October 2026 (drawn at the order). The EU AI Office: the decision
# of 24 January 2024 entered into force on 21 February 2024. The network: agreed in Seoul on 21 May 2024, first met on
# 20 November 2024, renamed on 9 December 2025.
BODIES = ["UK", "US", "EU", "network"]
NAMES = {
    "UK": [((2023, 11, 2), "AI Safety Institute"), ((2025, 2, 14), "AI Security Institute")],
    "US": [((2023, 11, 1), "U.S. AI Safety Institute"),
           ((2025, 6, 15), "Center for AI Standards and Innovation (CAISI)"),
           ((2026, 9, 29), "Center for Advancing Innovation and Standards for Super Intelligence (CAISSI)")],
    "EU": [((2024, 2, 21), "European AI Office")],
    "network": [((2024, 11, 20), "International Network of AI Safety Institutes"),
                ((2025, 12, 9), "International Network for Advanced AI Measurement, Evaluation and Science")],
}
SHORT = {"AI Safety Institute": "AI Safety Institute", "AI Security Institute": "AI Security Institute",
         "U.S. AI Safety Institute": "U.S. AI Safety Institute",
         "Center for AI Standards and Innovation (CAISI)": "CAISI",
         "Center for Advancing Innovation and Standards for Super Intelligence (CAISSI)": "CAISSI",
         "European AI Office": "European AI Office",
         "International Network of AI Safety Institutes": "Network of AI Safety Institutes",
         "International Network for Advanced AI Measurement, Evaluation and Science":
             "Network for Advanced AI Measurement, Evaluation and Science"}
SEOUL_DAY = (2024, 5, 21)
MONTHS = [2, 8, 14, 20, 26, 32, 36]   # the demo: months after November 2023, each the first day of its month


def changes(before, after):
    """Who is in `before` and not in `after`, and who is in `after` and not in `before`, each as a sorted list: what
    the learner writes in Exercise 2."""
    return sorted(before - after), sorted(after - before)


def schemed_in(table, model):
    """In how many evaluations of `table` the model schemed at least once: what the learner writes in Exercise 1."""
    return sum(1 for evaluation in table if model in table[evaluation])


def day(months: int) -> tuple:
    """The first day of the month that is `months` months after November 2023."""
    total = 2023 * 12 + 10 + int(months)
    return total // 12, total % 12 + 1, 1


def name_on(body: str, date: tuple):
    """The name the body had on that date, or None before it existed."""
    name = None
    for start, new in NAMES[body]:
        if start <= date:
            name = new
    return name


def with_safety(date: tuple) -> list:
    """The bodies whose name on that date has the word Safety in it."""
    return [b for b in BODIES if name_on(b, date) and "Safety" in name_on(b, date)]


# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.15rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;color:var(--ea-ink);vertical-align:top")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"
_CODE = "font-family:var(--ea-font-code);font-size:0.78rem"


def _note(text: str) -> str:
    return f'<div style="{_UI}">{html.escape(text)}</div>'


def _table(rows: list, head: list) -> str:
    """A table with a head row; each row is (its cells, a style of the row), and each cell is (html, style)."""
    top = "".join(f'<th style="{_HEAD}">{h}</th>' for h in head)
    body = "".join(f'<tr style="{style}">' + "".join(f'<td style="{_CELL}{";" + s if s else ""}">{cell}</td>'
                                                    for cell, s in cells_) + "</tr>" for cells_, style in rows)
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.3rem 0">'
            f'<tr>{top}</tr>{body}</table></div>')


def _code(text: str) -> str:
    return f'<span style="{_CODE}">{html.escape(text)}</span>'


_AREAS = {
    "en": {"title": "The three areas of the Singapore Consensus (8 May 2025)",
           "areas": [("1. Assessment", "how severe and how likely a harm is, before and after release",
                      ["Audit techniques and benchmarks", "Downstream impact assessment and forecasting",
                       "Secure evaluation infrastructure", "System safety assessment",
                       "Metrology for AI risk assessment", "Dangerous capability and propensity assessment",
                       "Loss-of-control risk assessment"]),
                     ("2. Development", "build the system so that it does what was specified, and check that it does",
                      ["Specification and validation: defining the system's purpose",
                       "Design and implementation: building the system",
                       "Verification: assessing if the system works as specified"]),
                     ("3. Control", "watch the system and the world around it after release, and step in",
                      ["AI system monitoring", "AI ecosystem monitoring", "Societal resilience research"])],
           "note": "Each list gives the research areas under that heading, in the document's own order. The areas "
                   "overlap: research on one can help another."},
    "nl": {"title": "De drie gebieden van de Singapore Consensus (8 mei 2025)",
           "areas": [("1. Beoordeling", "hoe ernstig en hoe waarschijnlijk een schade is, voor en na de release",
                      ["Audittechnieken en benchmarks", "Gevolgen verderop beoordelen en voorspellen",
                       "Een veilige infrastructuur voor evaluaties", "De veiligheid van het systeem beoordelen",
                       "Goed meten bij risicobeoordeling", "Gevaarlijke vaardigheden en neigingen beoordelen",
                       "Het risico op verlies van controle beoordelen"]),
                     ("2. Ontwikkeling", "het systeem zo bouwen dat het doet wat is vastgelegd, en nagaan of het dat doet",
                      ["Specificatie en validatie: vastleggen waar het systeem voor is",
                       "Ontwerp en implementatie: het systeem bouwen",
                       "Verificatie: nagaan of het systeem werkt zoals vastgelegd"]),
                     ("3. Controle", "het systeem en de wereld eromheen na de release in de gaten houden, en ingrijpen",
                      ["AI-systemen in de gaten houden", "Het ecosysteem rond AI in de gaten houden",
                       "Onderzoek naar de weerbaarheid van de samenleving"])],
           "note": "Elke lijst geeft de onderzoeksgebieden onder dat kopje, in de volgorde van het document. De gebieden "
                   "overlappen: onderzoek voor het ene kan het andere helpen."},
}


def areas_picture(lang: str) -> str:
    """The three areas of the Singapore Consensus, each with what it is about and its research areas."""
    w = _AREAS[lang]
    boxes = []
    for i, (name, what, items) in enumerate(w["areas"], start=1):
        lis = "".join(f'<li style="margin:0.1rem 0">{i}.{k} {html.escape(item)}</li>'
                      for k, item in enumerate(items, start=1))
        boxes.append(f'<div style="flex:1 1 12rem;border:1px solid var(--ea-line);border-radius:6px;padding:0.5rem 0.7rem;'
                     f'font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink)">'
                     f'<div style="font-weight:600">{html.escape(name)}</div>'
                     f'<div style="color:var(--ea-ink-2);margin:0.2rem 0 0.3rem">{html.escape(what)}</div>'
                     f'<ul style="margin:0;padding-left:1.1rem;list-style:none;text-indent:-1.1rem">{lis}</ul></div>')
    body = f'<div style="display:flex;flex-wrap:wrap;gap:0.6rem;margin:0.3rem 0">{"".join(boxes)}</div>'
    return yours.figure(w["title"], body + _note(w["note"]), alt=w["title"])


_MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
_MONTHS_NL = ["januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus", "september", "oktober",
              "november", "december"]


def date_words(date: tuple, lang: str) -> str:
    y, m, d = date
    return f"{d} {(_MONTHS_EN if lang == 'en' else _MONTHS_NL)[m - 1]} {y}"


_DEMO_WORDS = {
    "en": {"title": "Four public bodies and their names, on {date}",
           "rows": {"UK": "UK institute", "US": "US center", "EU": "EU", "network": "network"},
           "agreed": "agreed in Seoul",
           "said": {"UK": "the British institute is called the {name}", "US": "the American center the {name}",
                    "EU": "the EU has its {name}", "network": "the network is called the {name}"},
           "missing": {"EU": "the EU AI Office does not exist yet", "network": "the network does not exist yet"},
           "start": "On {date}, ", "sep": ", ", "and": " and ",
           "count": {0: "None of the names has the word Safety in it.", 1: "One of the names has the word Safety in it.",
                     2: "Two of the names have the word Safety in them.",
                     3: "Three of the names have the word Safety in them."},
           "note": "The name in bold, on the bar that stands out, is the one in force on that date. The American "
                   "center became CAISI in June 2025, on "
                   "a day the sources do not give, and CAISSI after an order of 29 September 2026."},
    "nl": {"title": "Vier overheidsinstellingen en hun namen, op {date}",
           "rows": {"UK": "Brits instituut", "US": "VS-centrum", "EU": "EU", "network": "netwerk"},
           "agreed": "afgesproken in Seoul",
           "said": {"UK": "heet het Britse instituut het {name}", "US": "het Amerikaanse centrum het {name}",
                    "EU": "heeft de EU het {name}", "network": "heet het netwerk het {name}"},
           "missing": {"EU": "bestaat het AI-bureau van de EU nog niet", "network": "bestaat het netwerk nog niet"},
           "start": "Op {date} ", "sep": ", ", "and": " en ",
           "count": {0: "Geen van de namen bevat het woord Safety.", 1: "Eén van de namen bevat het woord Safety.",
                     2: "Twee van de namen bevatten het woord Safety.", 3: "Drie van de namen bevatten het woord Safety."},
           "note": "De vetgedrukte naam, op de balk die opvalt, is de naam die op die datum gold. Het Amerikaanse "
                   "centrum werd CAISI in juni "
                   "2025, op een dag die de bronnen niet geven, en CAISSI na een besluit van 29 september 2026."},
}
_X0, _X1 = datetime.date(2023, 10, 1).toordinal(), datetime.date(2027, 1, 1).toordinal()


def _label(parts: list, x: float, y: float, text: str, room: float, style: str) -> None:
    """A label above a bar, starting at x; on two lines of about equal length when it does not fit in `room`
    pixels, and on as many lines as it needs when two are not enough."""
    lines = [text]
    if len(text) * 5.7 > room and " " in text:
        words = text.split(" ")
        best = min(range(1, len(words)), key=lambda k: abs(len(" ".join(words[:k])) - len(" ".join(words[k:]))))
        lines = [" ".join(words[:best]), " ".join(words[best:])]
        if max(len(line) for line in lines) * 5.7 > room:
            lines = [words[0]]
            for word in words[1:]:
                if len(lines[-1] + " " + word) * 5.7 > room:
                    lines.append(word)
                else:
                    lines[-1] += " " + word
    for k, line in enumerate(lines):
        dy = (k - len(lines) + 1) * 12
        parts.append(f'<text x="{x:.1f}" y="{y + dy:.1f}" style="{style}">{html.escape(line)}</text>')


def demo_frame(months, lang: str) -> tuple[str, str]:
    """The demo: the four bodies over time, and their names on the first day of the month `months` months after
    November 2023."""
    months = int(min(MONTHS, key=lambda m: abs(m - float(months))))
    w = _DEMO_WORDS[lang]
    date = day(months)
    W, left, right, top, row = 640, 96, 628, 34, 52
    H = top + row * len(BODIES) + 26

    def at(d: tuple) -> float:
        return left + (datetime.date(*d).toordinal() - _X0) / (_X1 - _X0) * (right - left)

    parts = []
    base = H - 24
    parts.append(f'<line x1="{left}" y1="{base}" x2="{right}" y2="{base}" style="stroke:var(--ea-axis)"/>')
    for y in (2024, 2025, 2026, 2027):
        x = at((y, 1, 1))
        parts.append(f'<line x1="{x:.1f}" y1="{base}" x2="{x:.1f}" y2="{base + 5}" style="stroke:var(--ea-axis)"/>')
        parts.append(f'<text x="{x:.1f}" y="{base + 17}" style="font-size:11px;fill:var(--ea-ink-2);'
                     f'text-anchor:middle">{y}</text>')
    halo = ";paint-order:stroke;stroke:var(--ea-page);stroke-width:3px;stroke-linejoin:round"
    label = "font-size:10.5px;fill:var(--ea-ink-2)" + halo
    names = []                                # the labels, drawn last so the date's line runs behind them
    for i, body in enumerate(BODIES):
        y = top + row * i + row - 14          # the top of this row's bar
        parts.append(f'<text x="{left - 8}" y="{y + 7:.1f}" style="font-size:12px;fill:var(--ea-ink);'
                     f'text-anchor:end">{html.escape(w["rows"][body])}</text>')
        if body == "network":                 # agreed in Seoul, before its first meeting
            x0, x1 = at(SEOUL_DAY), at(NAMES[body][0][0])
            parts.append(f'<rect x="{x0:.1f}" y="{y:.1f}" width="{x1 - x0:.1f}" height="9" style="fill:none;'
                         f'stroke:var(--ea-ink-3);stroke-dasharray:3 2"/>')
            _label(names, x0, y - 4, w["agreed"], x1 - x0, label)
        history = NAMES[body]
        now = name_on(body, date)
        for k, (start, name) in enumerate(history):
            x0 = at(start)
            x1 = at(history[k + 1][0]) if k + 1 < len(history) else right
            fill = "var(--ea-ink)" if name == now else "var(--ea-line)"
            parts.append(f'<rect x="{x0:.1f}" y="{y:.1f}" width="{x1 - x0:.1f}" height="9" style="fill:{fill};'
                         f'stroke:var(--ea-page);stroke-width:1"/>')
            _label(names, x0 + 1, y - 4, SHORT[name], x1 - x0 - 2,
                   "font-size:10.5px;fill:var(--ea-ink);font-weight:600" + halo if name == now else label)
    x = at(date)
    parts.append(f'<line x1="{x:.1f}" y1="{top - 14}" x2="{x:.1f}" y2="{base}" style="stroke:var(--ea-ink-2);'
                 f'stroke-width:1.5;stroke-dasharray:4 3"/>')
    parts += names
    anchor = "end" if x > right - 80 else ("start" if x < left + 80 else "middle")
    parts.append(f'<text x="{x:.1f}" y="{top - 18}" style="font-size:11px;fill:var(--ea-ink-2);text-anchor:{anchor};'
                 f'paint-order:stroke;stroke:var(--ea-page);stroke-width:4px;stroke-linejoin:round">'
                 f'{html.escape(date_words(date, lang))}</text>')
    title = w["title"].format(date=date_words(date, lang))
    svg = (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}" style="width:{W}px;'
           f'max-width:100%;height:auto;display:block;font-family:var(--ea-font-ui)">{"".join(parts)}</svg>')
    pieces = []
    for body in BODIES:
        name = name_on(body, date)
        pieces.append(w["said"][body].format(name=name) if name else w["missing"][body])
    first = w["start"].format(date=date_words(date, lang)) + w["sep"].join(pieces[:-1]) + w["and"] + pieces[-1] + "."
    said = first + " " + w["count"][len(with_safety(date))]
    return yours.figure(title, svg + _note(w["note"]), alt=title + ". " + said), said


# --- the pictures under the checks -----------------------------------------------------------------------------------

def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _all gives for a function that stopped with an error


def _all(fn, calls: list):
    """The learner's function on copies of each list of arguments in `calls`, all within one time limit of a check;
    _STOPPED when it stops with an error, and the Timeout passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(lambda: [fn(*(copy.deepcopy(a) for a in args)) for args in calls])
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _shown(v) -> str:
    text = repr(v)
    return text if len(text) <= 60 else text[:57] + "..."


_CHANGES_WORDS = {
    "en": {"title": "Who signed in Seoul, and who joined the network",
           "head": ["", "the course's changes", "yours"],
           "rows": ["signed in Seoul, not a first member", "first member, did not sign in Seoul"],
           "both": "In both: {names}. In neither: the Netherlands, among others.",
           "stopped": "error", "shape": "not two lists",
           "note": "A dashed row is one where your changes gives something other than the course's."},
    "nl": {"title": "Wie tekende in Seoul, en wie werd lid van het netwerk",
           "head": ["", "changes van de course", "de jouwe"],
           "rows": ["tekende in Seoul, geen lid van het eerste uur", "lid van het eerste uur, tekende niet in Seoul"],
           "both": "In allebei: {names}. In geen van beide: onder meer Nederland.",
           "stopped": "fout", "shape": "geen twee lijsten",
           "note": "Een gestippelde rij is er een waar jouw changes iets anders geeft dan die van de course."},
}


def changes_picture(seen) -> str:
    """The course's two lists for the Seoul signatories and the network's first members beside the learner's."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _CHANGES_WORDS[seen.lang]
    want = changes(SEOUL, NETWORK)
    got = _all(fn, [(SEOUL, NETWORK)])
    mine = None if got is _STOPPED else got[0]
    rows = []
    for k in range(2):
        if mine is None:
            shown, off = w["stopped"], True
        elif not (isinstance(mine, tuple) and len(mine) == 2):
            shown, off = (w["shape"] if k == 0 else ""), True
        else:
            shown, off = _code(_shown(mine[k])), mine[k] != want[k]
        rows.append(([(html.escape(w["rows"][k]), ""), (_code(repr(want[k])), ""), (shown, "font-weight:600")],
                     _OFF if off else ""))
    both = w["both"].format(names=", ".join(sorted(SEOUL & NETWORK)))
    return yours.figure(w["title"], _table(rows, [html.escape(h) for h in w["head"]]) + _note(both) + _note(w["note"]),
                        alt=w["title"])


_SCHEMED_WORDS = {
    "en": {"title": "Apollo's Table 1, with your count under it",
           "corner": "evaluation", "course": "the course's schemed_in", "yours": "yours", "stopped": "error",
           "note": "✓: the model schemed in that evaluation at least once, with a goal from the prompt; in all but "
                   "instrumental alignment faking with a strong push to pursue it. A dashed cell is one where your schemed_in gives something other than the course's."},
    "nl": {"title": "Tabel 1 van Apollo, met jouw telling eronder",
           "corner": "evaluatie", "course": "schemed_in van de course", "yours": "de jouwe", "stopped": "fout",
           "note": "✓: het model deed in die evaluatie minstens één keer aan scheming, met een doel uit de prompt; "
                   "behalve bij instrumental alignment faking met een sterke duw om het na te streven. Een gestippelde cel is er een waar jouw schemed_in iets anders geeft "
                   "dan die van de course."},
}


def schemed_picture(seen) -> str:
    """Table 1 of Apollo's paper, with the course's count per model and the learner's under it."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _SCHEMED_WORDS[seen.lang]
    got = _all(fn, [(TABLE, m) for m in MODELS])
    rows = [([(html.escape(e), "")] + [("✓" if m in TABLE[e] else "×", "text-align:center") for m in MODELS], "")
            for e in TABLE]
    rows.append(([(html.escape(w["course"]), "")] + [(str(schemed_in(TABLE, m)), "text-align:center") for m in MODELS],
                 ""))
    cells = [(html.escape(w["yours"]), "")]
    for i, m in enumerate(MODELS):
        if got is _STOPPED:
            cells.append((w["stopped"], "text-align:center;font-weight:600;" + _OFF))
        else:
            v = got[i]
            ok = isinstance(v, int) and not isinstance(v, bool) and v == schemed_in(TABLE, m)
            cells.append((html.escape(_shown(v)), "text-align:center;font-weight:600" + ("" if ok else ";" + _OFF)))
    rows.append((cells, ""))
    head = [html.escape(w["corner"])] + [_code(m) for m in MODELS]
    return yours.figure(w["title"], _table(rows, head) + _note(w["note"]), alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["schemed_in", "changes", "headline", "area"]

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
    return isinstance(e, NameError) and re.search(r"'(SEOUL|NETWORK|TABLE|MODELS|C)'", str(e)) is not None


_NAMES = Mistake("lesson-names", error=_names_error,
                 en="Your function uses a name of the lesson, such as `SEOUL`, `TABLE` or `C`, and the check does not give it that name: use the names between the parentheses of the `def` line.",
                 nl="Je functie gebruikt een naam van de les, zoals `SEOUL`, `TABLE` of `C`, en de check geeft die naam niet mee: gebruik de namen tussen de haakjes van de `def`-regel.")


class Named:
    """A value the check gives the learner's function, shown in a message by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


# exercise 2: changes; the first case is the Seoul signatories against the network's first members
CHANGES_CASES = [(Named("SEOUL", SEOUL), Named("NETWORK", NETWORK)),
                 ({"a", "b", "c"}, {"b", "c", "d"}),
                 ({"x", "y"}, {"x", "y"}),
                 (set(), {"q", "p"}),
                 ({"n", "m", "o"}, {"o"})]
_C_START = "def changes(before, after):\n    ...\n"
_C = ("def changes(before, after):\n"
      "    left = sorted(before - after)\n"
      "    joined = sorted(after - before)\n"
      "    return left, joined\n")
_C_SWAP = _C.replace("    return left, joined\n", "    return joined, left\n")
_C_SETS = _C.replace("sorted(before - after)", "before - after").replace("sorted(after - before)", "after - before")
_C_SAME = _C.replace("sorted(after - before)", "sorted(before - after)")
_C_ONE = _C.replace("    return left, joined\n", "    return left\n")

# exercise 1: schemed_in; the first case is opus-3, which schemed in four of the six evaluations
SCHEMED_CASES = [(Named("TABLE", TABLE), "opus-3"),
                 (Named("TABLE", TABLE), "o1"),
                 (Named("TABLE", TABLE), "gpt-4o"),
                 (Named("TABLE", TABLE), "sonnet-3.5"),
                 ({"a": {"m"}, "b": {"m", "n"}, "c": set()}, "m"),
                 ({"a": {"m"}, "b": {"m", "n"}, "c": set()}, "n"),
                 ({}, "m")]
_S_START = "def schemed_in(table, model):\n    ...\n"
_S = ("def schemed_in(table, model):\n"
      "    count = 0\n"
      "    for evaluation in table:\n"
      "        if model in table[evaluation]:\n"
      "            count = count + 1\n"
      "    return count\n")
_S_ALL = _S.replace("        if model in table[evaluation]:\n            count = count + 1\n",
                    "        count = count + 1\n")
_S_SIZES = _S.replace("        if model in table[evaluation]:\n            count = count + 1\n",
                      "        count = count + len(table[evaluation])\n")
_S_EARLY = _S.replace("    return count\n", "        return count\n")
_S_NOT = _S.replace("if model in table[evaluation]", "if model not in table[evaluation]")


EXERCISES = {
    "changes": FunctionBox(
        name="changes",
        picture=changes_picture,
        function="changes",
        start=_C_START,
        solution=_C,
        visible=[CHANGES_CASES[0]],
        hidden=CHANGES_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("swapped", wrong=_C_SWAP,
                    en="`{call}` gives {got}, and it should give {want}: the two lists are the other way around. First who is in `before` and not in `after`, then who is in `after` and not in `before`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de twee lijsten staan andersom. Eerst wie in `before` zit en niet in `after`, dan wie in `after` zit en niet in `before`."),
            Mistake("sets", wrong=_C_SETS,
                    en="`{call}` gives {got}, and it should give {want}: those are sets, which have no fixed order. Put each difference in `sorted(...)` to get a list in order.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat zijn sets, en die hebben geen vaste volgorde. Zet elk verschil in `sorted(...)` voor een lijst op volgorde."),
            Mistake("same", wrong=_C_SAME,
                    en="`{call}` gives {got}, and it should give {want}: both lists are who is in `before` and not in `after`. The second is the other way around: who is in `after` and not in `before`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: beide lijsten zijn wie in `before` zit en niet in `after`. De tweede is andersom: wie in `after` zit en niet in `before`."),
            Mistake("one", wrong=_C_ONE,
                    en="`{call}` gives {got}, and it should give {want}: return both lists, with a comma between them.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: geef beide lijsten terug, met een komma ertussen."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: the sorted list of who is in `before` and not in `after`, then the sorted list of who is in `after` and not in `before`, and return both.",
                      "nl": "Vervang de `...`: de gesorteerde lijst van wie in `before` zit en niet in `after`, dan de gesorteerde lijst van wie in `after` zit en niet in `before`, en geef ze allebei terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. First the sorted list of who is in `before` and not in `after`, then the sorted list of who is in `after` and not in `before`, with a comma between them after `return`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Eerst de gesorteerde lijst van wie in `before` zit en niet in `after`, dan de gesorteerde lijst van wie in `after` zit en niet in `before`, met een komma ertussen na `return`."},
            "almost": {"en": "Right for Seoul and the network. But `{call}` gives {got}, and it should give {want}. Does your function use the `before` and `after` it gets?",
                       "nl": "Goed voor Seoul en het netwerk. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `before` en `after` die ze krijgt?"},
            "correct": {"en": "Your `changes` is right in all {n} cases. Under the check, your lists stand beside the course's.",
                        "nl": "Je `changes` klopt in alle {n} gevallen. Onder de check staan jouw lijsten naast die van de course."},
        },
    ),
    "schemed_in": FunctionBox(
        name="schemed_in",
        picture=schemed_picture,
        function="schemed_in",
        start=_S_START,
        solution=_S,
        visible=[SCHEMED_CASES[0]],
        hidden=SCHEMED_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("all", wrong=_S_ALL,
                    en="`{call}` gives {got}, and it should give {want}: that is the number of evaluations. Count only those in whose set the model is.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is het aantal evaluaties. Tel alleen die waarin het model in de set zit."),
            Mistake("sizes", wrong=_S_SIZES,
                    en="`{call}` gives {got}, and it should give {want}: you add up how many models schemed in each evaluation. Add 1 for each evaluation in which this model did.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt op hoeveel modellen er in elke evaluatie aan scheming deden. Tel 1 op voor elke evaluatie waarin dit model dat deed."),
            Mistake("early", wrong=_S_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your `return` is inside the loop, so the function stops after the first evaluation. Move `return` back, level with `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `return` staat in de loop, dus de functie stopt na de eerste evaluatie. Zet `return` terug, op gelijke hoogte met `for`."),
            Mistake("not", wrong=_S_NOT,
                    en="`{call}` gives {got}, and it should give {want}: you count the evaluations in which the model did not scheme. Leave out the `not`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt de evaluaties waarin het model niet aan scheming deed. Laat de `not` weg."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: start a count at 0, go through the evaluations in `table`, add 1 for each one in whose set the model is, and return the count.",
                      "nl": "Vervang de `...`: begin een telling bij 0, loop door de evaluaties in `table`, tel 1 op voor elke evaluatie waarin het model in de set zit, en geef de telling terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Count the evaluations in `table` whose set of models has `model` in it.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Tel de evaluaties in `table` waarvan de set modellen `model` bevat."},
            "almost": {"en": "Right for opus-3. But `{call}` gives {got}, and it should give {want}. Does your function use the `table` and `model` it gets?",
                       "nl": "Goed voor opus-3. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `table` en het `model` die ze krijgt?"},
            "correct": {"en": "Your `schemed_in` is right in all {n} cases. Under the check, your count stands under Apollo's table.",
                        "nl": "Je `schemed_in` klopt in alle {n} gevallen. Onder de check staat jouw telling onder de tabel van Apollo."},
        },
    ),
}

CASES = [
    ("changes", _C_START, "empty", None),
    ("changes", _C, "correct", None),
    ("changes", "def changes(before, after):\n    return sorted(before - after), sorted(after - before)\n", "correct", None),
    ("changes", "def changes(before, after):\n    left = []\n    for x in sorted(before):\n        if x not in after:\n            left.append(x)\n    joined = []\n    for x in sorted(after):\n        if x not in before:\n            joined.append(x)\n    return left, joined\n", "correct", None),
    ("changes", _C_SWAP, "wrong", "swapped"),
    ("changes", _C_SETS, "wrong", "sets"),
    ("changes", _C_SAME, "wrong", "same"),
    ("changes", _C_ONE, "wrong", "one"),
    ("changes", f"def changes(before, after):\n    return {changes(SEOUL, NETWORK)!r}\n", "almost", None),
    ("changes", "def changes(before, after):\n    return sorted(SEOUL - after), []\n", "error", "lesson-names"),
    ("schemed_in", _S_START, "empty", None),
    ("schemed_in", _S, "correct", None),
    ("schemed_in", "def schemed_in(table, model):\n    found = []\n    for evaluation in table:\n        if model in table[evaluation]:\n            found.append(evaluation)\n    return len(found)\n", "correct", None),
    ("schemed_in", _S_ALL, "wrong", "all"),
    ("schemed_in", _S_SIZES, "wrong", "sizes"),
    ("schemed_in", _S_EARLY, "wrong", "early"),
    ("schemed_in", _S_NOT, "wrong", "not"),
    ("schemed_in", "def schemed_in(table, model):\n    return 4\n", "almost", None),
    ("schemed_in", "def schemed_in(table, model):\n    return len(TABLE)\n", "error", "lesson-names"),
]
