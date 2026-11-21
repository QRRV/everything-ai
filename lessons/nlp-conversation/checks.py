# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-conversation: measure the gap or overlap at each change of speaker in a conversation
(gaps), label a turn with one of three dialogue acts by three rules (label), sort replies by the kind of evidence of
understanding they give, after Clark and Schaefer (1989), and tell an answer from a backchannel. The conversation of
Sam and Kim with its times, the labels by hand, the lists of backchannels, Sam's request in the demo with its pauses,
the replies of the sort and the exchanges of the choice are the course's own."""
import ast
import contextlib
import copy
import html
import io
import re

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, with_time_limit

# --- the conversation ------------------------------------------------------------------------------------------------

# Each stretch of talk: who speaks, from when, to when, in milliseconds. The same in both languages.
TALK = [("Sam", 0, 1500), ("Kim", 1700, 3000), ("Sam", 2950, 3500), ("Kim", 3600, 3900), ("Sam", 3900, 5200),
        ("Sam", 5800, 6500), ("Kim", 6600, 6900), ("Sam", 6950, 8400), ("Kim", 8700, 9000), ("Sam", 9100, 9600)]
SAID = {
    "en": ["When does the last train to Utrecht leave?", "At 11:40, from platform 4", "Platform 4?", "Yeah",
           "So I have to leave home at eleven", "or a bit before", "Mhm", "Could you send me the times?", "Okay",
           "Thanks"],
    "nl": ["Hoe laat gaat de laatste trein naar Utrecht?", "Om 23.40 uur, van spoor 4", "Spoor 4?", "Ja",
           "Dus ik moet om elf uur van huis", "of iets eerder", "Mhm", "Kun je me de tijden sturen?", "Oké",
           "Bedankt"],
}
BACKCHANNELS = {"en": ["mhm", "uh-huh", "yeah", "right", "okay"], "nl": ["mhm", "hm-hm", "ja", "oké", "precies"]}
# What each stretch does, labeled by hand, the course's own, in the style of Switchboard's tags.
BY_HAND = ["question", "answer", "question", "answer", "statement", "statement", "backchannel", "request", "accept",
           "thanks"]
LABELS = ("backchannel", "question", "statement")


def gaps(talk):
    out = []
    for i in range(1, len(talk)):
        if talk[i][0] != talk[i - 1][0]:
            out.append(talk[i][1] - talk[i - 1][2])
    return out


def label(text, backchannels):
    if text.lower() in backchannels:
        return "backchannel"
    if "?" in text:
        return "question"
    return "statement"


def speakers(talk):
    return [stretch[0] for stretch in talk]


def lengths(talk):
    out = []
    for i in range(len(talk)):
        out.append(talk[i][2] - talk[i][1])
    return out


def changes(talk) -> list:
    """The places i at which the speaker changes, with talk[i] the first stretch of the new turn."""
    return [i for i in range(1, len(talk)) if talk[i][0] != talk[i - 1][0]]


# --- the demo: Sam asks a program for a train, with two pauses to think --------------------------------------------

ASKS = {
    "en": [("I'd like a train", 0, 1100), ("to, uh", 1500, 2200), ("Utrecht", 3100, 3700),
           ("tomorrow morning", 3950, 5100)],
    "nl": [("Ik wil een trein", 0, 1100), ("naar, eh", 1500, 2200), ("Utrecht", 3100, 3700),
           ("morgenochtend", 3950, 5100)],
}
PEOPLE_MS = 208   # the mean offset before a response to a yes/no question, over ten languages (Stivers et al. 2009)


def takes_turn(wait: int, asks: list) -> tuple:
    """A program that takes the turn after a silence of `wait` ms: (the index of the last stretch it lets Sam say,
    the time at which it starts to talk)."""
    for i in range(len(asks) - 1):
        if asks[i + 1][1] - asks[i][2] >= wait:
            return i, asks[i][2] + wait
    return len(asks) - 1, asks[-1][2] + wait


# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = ";color:var(--ea-loss-text)"
_CODE = ";font-family:var(--ea-font-code);font-size:0.78rem"
_SVG_TEXT = "font-family:var(--ea-font-ui);font-size:12px;fill:var(--ea-ink-2)"


def _note(text: str) -> str:
    """A line under a picture; a word between asterisks is set in italics, as in the lesson's text."""
    marked = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", html.escape(text))
    return f'<div style="{_UI}">{marked}</div>'


def _bare(text: str) -> str:
    """The same line without its asterisks, for the alt text."""
    return text.replace("*", "")


def _table(head: list, rows: list) -> str:
    """rows: (cells, off); a cell is (text, is_code)."""
    top = "".join(f'<th style="{_HEAD}">{html.escape(h)}</th>' for h in head)
    body = ""
    for cells, off in rows:
        body += "<tr>" + "".join(
            f'<td style="{_CELL}{_CODE if code else ""}{_OFF if off else ""}">{html.escape(text)}</td>'
            for text, code in cells) + "</tr>"
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.3rem 0">'
            f'<tr>{top}</tr>{body}</table></div>')


def _code(value) -> str:
    """A value the way Python shows it, with the strings in double quotes."""
    if isinstance(value, str):
        return '"' + value + '"'
    if isinstance(value, tuple):
        return "(" + ", ".join(_code(v) for v in value) + ")"
    if isinstance(value, list):
        return "[" + ", ".join(_code(v) for v in value) + "]"
    return repr(value)


def _short(value, limit: int = 44) -> str:
    text = _code(value)
    return text if len(text) <= limit else text[:limit - 1] + "…"


_EVIDENCE = {
    "en": {"title": "Five kinds of evidence of understanding, from weakest to strongest",
           "head": ["Kind", "What Sam does after Kim says: The museum opens at ten."],
           "rows": [("continued attention", "keeps looking at Kim, and listens on"),
                    ("the relevant next contribution", "And when does it close?"),
                    ("acknowledgment", "Mhm."),
                    ("demonstration", "So not before ten."),
                    ("display", "At ten.")],
           "legend": "After Clark and Schaefer (1989), who call the order rough; the replies are the course's own. "
                     "Only the last two show *what* Sam understood: in Sam's own words, or in Kim's."},
    "nl": {"title": "Vijf tekens van begrip, van zwakst naar sterkst",
           "head": ["Soort", "Wat Sam doet nadat Kim zegt: Het museum gaat om tien uur open."],
           "rows": [("blijvende aandacht", "blijft naar Kim kijken en luistert verder"),
                    ("de passende volgende bijdrage", "En hoe laat gaat het dicht?"),
                    ("ontvangstbevestiging", "Mhm."),
                    ("demonstratie", "Dus niet vóór tienen."),
                    ("letterlijke herhaling", "Om tien uur.")],
           "legend": "Naar Clark en Schaefer (1989), volgens wie de volgorde maar ruwweg geldt; de antwoorden zijn van de course "
                     "zelf. Alleen de laatste twee laten zien *wat* Sam begreep: in de eigen woorden van Sam, of in "
                     "die van Kim."},
}


def evidence_picture(lang: str) -> str:
    w = _EVIDENCE[lang]
    rows = [([(kind, False), (reply, False)], False) for kind, reply in w["rows"]]
    alt = f'{w["title"]}. ' + "; ".join(f"{k}: {r}" for k, r in w["rows"]) + f'. {_bare(w["legend"])}'
    return yours.figure(w["title"], _table(w["head"], rows) + _note(w["legend"]), alt=alt)


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error


def _call(fn, *args):
    """The learner's function within the time limit of a check; _STOPPED when it stops with an error, and the Timeout
    passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(lambda: fn(*(copy.deepcopy(a) for a in args)))
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _ms(value) -> str:
    """A gap as the picture shows it: +200, -50, 0; anything else as Python shows it."""
    if type(value) is int:
        return f"{value:+d}" if value else "0"
    return _short(value, 12)


_GAPS_WORDS = {
    "en": {"title": "Your gaps, at work: the conversation of Sam and Kim",
           "axis": "seconds",
           "fine": "Your gaps are right at each of the {n} changes of speaker. One is an overlap: Sam starts *Platform "
                   "4?* before Kim is done. The longest silence, {pause} ms within Sam's turn, is no gap at all.",
           "differs": "Where a number is in purple, your gaps differ from the right ones: {want}.",
           "count": "Your list has {got} values, and the conversation has {n} changes of speaker.",
           "not_list": "Your gaps gives {got}, not a list of numbers.",
           "error": "Your function stops with an error on TALK.",
           "slow": "Your function takes too long."},
    "nl": {"title": "Jouw gaps aan het werk: het gesprek van Sam en Kim",
           "axis": "seconden",
           "fine": "Je gaps klopt bij elk van de {n} sprekerswisselingen. Eén keer is er overlap: Sam begint aan "
                   "*Spoor 4?* voordat Kim klaar is. De langste stilte, {pause} ms binnen de beurt van Sam, telt "
                   "niet mee: Sam houdt de beurt.",
           "differs": "Waar een getal paars is, wijkt je gaps af van de goede overgangstijden: {want}.",
           "count": "Je lijst heeft {got} waarden, en het gesprek heeft {n} sprekerswisselingen.",
           "not_list": "Je gaps geeft {got}, geen lijst met getallen.",
           "error": "Je functie stopt met een fout bij TALK.",
           "slow": "Je functie doet er te lang over."},
}


def _timeline(lang: str, values: list | None) -> tuple:
    """The conversation as two rows of stretches, with `values` written at the changes of speaker: the svg and
    whether every value is right."""
    said, want, at = SAID[lang], gaps(TALK), changes(TALK)
    left, right_edge, width = 46, 12, 760
    end = max(s[2] for s in TALK)
    scale = (width - left - right_edge) / end
    lane = {"Sam": 26, "Kim": 96}
    bar_h = 22
    height, axis = 164, 128
    parts = []
    for name, y in lane.items():
        parts.append(f'<text x="4" y="{y + 15}" style="font-family:var(--ea-font-ui);font-size:13px;font-weight:600;'
                     f'fill:var(--ea-ink)">{name}</text>')
    for k, (who, start, stop) in enumerate(TALK):
        x, w = left + start * scale, max(2.0, (stop - start) * scale)
        parts.append(f'<rect x="{x:.1f}" y="{lane[who]}" width="{w:.1f}" height="{bar_h}" rx="4" '
                     f'style="fill:var(--ea-inset);stroke:var(--ea-ink-2);stroke-width:1.2">'
                     f'<title>{html.escape(who + ": " + said[k])}</title></rect>')
    all_right = values is not None and len(values) == len(want)
    for k, i in enumerate(at):
        prev_end, new_start = TALK[i - 1][2], TALK[i][1]
        mid = left + (prev_end + new_start) / 2 * scale
        if values is None:
            text, off = _ms(want[k]), False
        elif k < len(values):
            ok = type(values[k]) is int and values[k] == want[k]
            text, off = _ms(values[k]), not ok
            all_right = all_right and ok
        else:
            text, off = "?", True
        y = 64 if k % 2 == 0 else 80
        fill = "var(--ea-loss-text)" if off else "var(--ea-ink)"
        parts.append(f'<line x1="{mid:.1f}" y1="{lane["Sam"] + bar_h + 2}" x2="{mid:.1f}" y2="{lane["Kim"] - 2}" '
                     f'style="stroke:var(--ea-line);stroke-width:1;stroke-dasharray:2 3"/>')
        parts.append(f'<text x="{mid:.1f}" y="{y}" style="font-family:var(--ea-font-code);font-size:11px;'
                     f'font-weight:600;fill:{fill};text-anchor:middle;paint-order:stroke;stroke:var(--ea-page);'
                     f'stroke-width:4px">{html.escape(text)}</text>')
    for sec in range(0, end // 1000 + 1):
        x = left + sec * 1000 * scale
        parts.append(f'<line x1="{x:.1f}" y1="{axis}" x2="{x:.1f}" y2="{axis + 5}" '
                     f'style="stroke:var(--ea-ink-3);stroke-width:1"/>')
        parts.append(f'<text x="{x:.1f}" y="{axis + 17}" style="{_SVG_TEXT};font-size:11px;text-anchor:middle">'
                     f'{sec}</text>')
    parts.append(f'<line x1="{left}" y1="{axis}" x2="{width - right_edge}" y2="{axis}" '
                 f'style="stroke:var(--ea-ink-3);stroke-width:1"/>')
    parts.append(f'<text x="{width - right_edge}" y="{height - 3}" style="{_SVG_TEXT};font-size:11px;'
                 f'text-anchor:end">{_GAPS_WORDS[lang]["axis"]}</text>')
    return parts, width, height, all_right


def gaps_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    lang, w = seen.lang, _GAPS_WORDS[seen.lang]
    want = gaps(TALK)
    try:
        got = _call(fn, TALK)
    except Timeout:
        return yours.figure(w["title"], _note(w["slow"]), alt=w["slow"])
    if got is _STOPPED:
        return yours.figure(w["title"], _note(w["error"]), alt=w["error"])
    if not isinstance(got, list):
        line = w["not_list"].format(got=_short(got))
        return yours.figure(w["title"], _note(line), alt=line)
    parts, width, height, all_right = _timeline(lang, got)
    pause = TALK[5][1] - TALK[4][2]
    if all_right:
        after = w["fine"].format(n=len(want), pause=pause)
    else:
        after = w["differs"].format(want=_code(want))
        if len(got) != len(want):
            after += " " + w["count"].format(got=len(got), n=len(want))
    alt = (f'{w["title"]}. ' + "; ".join(f"{who} {a}-{b} ms: {SAID[lang][k]}" for k, (who, a, b) in enumerate(TALK))
           + f". {_short(got, 80)}. {_bare(after)}")
    svg = (f'<svg viewBox="0 0 {width} {height}" role="img" style="width:100%;max-width:{width}px;height:auto">'
           f'<title>{html.escape(alt)}</title>{"".join(parts)}</svg>')
    return yours.figure(w["title"], svg + _note(after), alt=alt)


_DEMO_WORDS = {
    "en": {"title": "When is Sam done? A program that waits for {wait} ms of silence",
           "pause": "{ms} ms", "starts": "program starts",
           "cut": "The program starts talking after *{said}*, before Sam is done.",
           "waits": "The program lets Sam finish, and starts {wait} ms after the end. In the study of Stivers and "
                    "colleagues, people responded to a yes/no question after {people} ms on average.",
           "how": "Each bar is a stretch of Sam's talk; the numbers between the bars are the pauses. The shaded box is "
                  "the silence the program waits for."},
    "nl": {"title": "Wanneer is Sam klaar? Een programma dat {wait} ms stilte afwacht",
           "pause": "{ms} ms", "starts": "programma begint",
           "cut": "Het programma begint te praten na *{said}*, voordat Sam klaar is.",
           "waits": "Het programma laat Sam uitpraten, en begint {wait} ms na het einde. In het onderzoek van "
                    "Stivers en collega's reageerden mensen gemiddeld na {people} ms op een ja-neevraag.",
           "how": "Elke balk is een stuk van wat Sam zegt; de getallen tussen de balken zijn de pauzes. Het grijze "
                  "vak is de stilte die het programma afwacht."},
}


def demo_picture(wait: int, lang: str) -> str:
    """Sam's request with its pauses, and where a program that waits `wait` ms for silence starts to talk."""
    w, asks = _DEMO_WORDS[lang], ASKS[lang]
    wait = max(50, min(3000, int(wait)))
    last, start = takes_turn(wait, asks)
    left, width, height = 8, 760, 92
    end = asks[-1][2] + 1500
    scale = (width - 2 * left) / end
    top, bar_h = 34, 26
    parts = []
    for k, (said, a, b) in enumerate(asks):
        x, bw = left + a * scale, (b - a) * scale
        stroke = "stroke:var(--ea-ink);stroke-width:2" if k <= last else "stroke:var(--ea-line);stroke-width:1.2"
        parts.append(f'<rect x="{x:.1f}" y="{top}" width="{bw:.1f}" height="{bar_h}" rx="5" '
                     f'style="fill:var(--ea-inset);{stroke}"/>')
        parts.append(f'<text x="{x + bw / 2:.1f}" y="{top + 17}" style="font-family:var(--ea-font-ui);font-size:12px;'
                     f'fill:var(--ea-ink);text-anchor:middle">{html.escape(said)}</text>')
        if k + 1 < len(asks):
            nxt = asks[k + 1][1]
            parts.append(f'<text x="{left + (b + nxt) / 2 * scale:.1f}" y="{top + bar_h + 16}" '
                         f'style="{_SVG_TEXT};font-size:11px;text-anchor:middle">'
                         f'{w["pause"].format(ms=nxt - b)}</text>')
    quiet_from = asks[last][2]
    parts.append(f'<rect x="{left + quiet_from * scale:.1f}" y="{top - 6}" width="{wait * scale:.1f}" '
                 f'height="{bar_h + 12}" style="fill:var(--ea-ink-3);opacity:0.22"/>')
    sx = left + start * scale
    parts.append(f'<line x1="{sx:.1f}" y1="{top - 14}" x2="{sx:.1f}" y2="{top + bar_h + 6}" '
                 f'style="stroke:var(--ea-ink);stroke-width:2"/>')
    anchor = "end" if sx > width - 120 else "start"
    shift = -5 if anchor == "end" else 5
    parts.append(f'<text x="{sx + shift:.1f}" y="{top - 16}" style="font-family:var(--ea-font-ui);font-size:12px;'
                 f'font-weight:600;fill:var(--ea-ink);text-anchor:{anchor}">{html.escape(w["starts"])}</text>')
    if last < len(asks) - 1:
        line = w["cut"].format(said=" ".join(s for s, _, _ in asks[:last + 1]))
    else:
        line = w["waits"].format(wait=wait, people=PEOPLE_MS)
    title = w["title"].format(wait=wait)
    alt = f"{title}. " + " ".join(s for s, _, _ in asks) + f". {_bare(line)} {_bare(w['how'])}"
    svg = (f'<svg viewBox="0 0 {width} {height}" role="img" style="width:100%;max-width:{width}px;height:auto">'
           f'<title>{html.escape(alt)}</title>{"".join(parts)}</svg>')
    return yours.figure(title, svg + _note(line) + _note(w["how"]), alt=alt)


_LABEL_WORDS = {
    "en": {"title": "Your label, at work: a dialogue act for each turn",
           "head": ["Turn", "Yours", "Rules", "By hand"],
           "fine": "Your label follows the three rules on every turn. The rules agree with the labels by hand on "
                   "{agree} of the {n} turns; *statement* for every turn would agree on {flat}.",
           "differs": "Where a row is in purple, your label differs from what the three rules give.",
           "error": "Your function stops with an error at: {what}.",
           "slow": "Your function takes too long."},
    "nl": {"title": "Jouw label aan het werk: een dialogue act bij elke beurt",
           "head": ["Beurt", "Jouw antwoord", "Regels", "Met de hand"],
           "fine": "Je label volgt de drie regels bij elke beurt. De regels komen bij {agree} van de {n} beurten overeen "
                   "met de labels die de course met de hand gaf; *statement* bij elke beurt zou er {flat} goed hebben.",
           "differs": "Waar een rij paars is, wijkt je label af van wat de drie regels geven.",
           "error": "Je functie stopt met een fout bij: {what}.",
           "slow": "Je functie doet er te lang over."},
}


def label_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    lang, w = seen.lang, _LABEL_WORDS[seen.lang]
    said, chans = SAID[lang], BACKCHANNELS[lang]
    rows, end, all_right = [], "", True
    for k, text in enumerate(said):
        turn = f"{TALK[k][0]}: {text}"
        try:
            got = _call(fn, text, chans)
        except Timeout:
            end = w["slow"]
            break
        if got is _STOPPED:
            end = w["error"].format(what=turn)
            break
        rule = label(text, chans)
        right = type(got) is str and got == rule
        all_right = all_right and right
        shown = got if type(got) is str else _short(got)
        rows.append(([(turn, False), (shown, True), (rule, True), (BY_HAND[k], True)], not right))
    agree = sum(label(t, chans) == h for t, h in zip(said, BY_HAND))
    flat = sum(h == "statement" for h in BY_HAND)
    after = end or (w["fine"].format(agree=agree, n=len(said), flat=flat) if all_right else w["differs"])
    alt = f'{w["title"]}. ' + "; ".join(" | ".join(c[0] for c in r[0]) for r in rows) + f". {_bare(after)}"
    return yours.figure(w["title"], _table(w["head"], rows) + _note(after), alt=alt)


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["gaps", "label", "evidence", "yeah"]


class Named:
    """A value the check gives the learner's function, shown in a message by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: "..." in "\n".join(
                    line.split("#")[0] for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# exercise 1: gaps
GAPS_CASES = [
    (Named("TALK", TALK),),
    ([("A", 0, 1000), ("B", 1200, 2000)],),
    ([("A", 0, 1000), ("B", 900, 2000)],),
    ([("A", 0, 1000), ("A", 1500, 2000), ("B", 2300, 3000)],),
    ([("A", 0, 1000)],),
    ([],),
    ([("A", 0, 500), ("B", 500, 900), ("A", 1100, 1600), ("B", 1500, 2000)],),
    ([("B", 0, 400), ("B", 600, 900)],),
]
_G_START = "def gaps(talk):\n    ...\n"
_G = ("def gaps(talk):\n"
      "    out = []\n"
      "    for i in range(1, len(talk)):\n"
      "        if talk[i][0] != talk[i - 1][0]:\n"
      "            out.append(talk[i][1] - talk[i - 1][2])\n"
      "    return out\n")
_G_PAUSES = ("def gaps(talk):\n"
             "    out = []\n"
             "    for i in range(1, len(talk)):\n"
             "        out.append(talk[i][1] - talk[i - 1][2])\n"
             "    return out\n")
_G_SAME = _G.replace("!=", "==")
_G_SIGN = _G.replace("talk[i][1] - talk[i - 1][2]", "talk[i - 1][2] - talk[i][1]")
_G_STARTS = _G.replace("talk[i][1] - talk[i - 1][2]", "talk[i][1] - talk[i - 1][1]")
_G_ZERO = _G.replace("range(1, len(talk))", "range(len(talk))")
_G_EARLY = _G.replace("    return out\n", "        return out\n")
_G_EARLY_IF = _G.replace("    return out\n", "            return out\n")

# exercise 2: label
_B = Named("BACKCHANNELS", BACKCHANNELS["en"])
LABEL_CASES = [
    ("Mhm", _B),
    ("11:40?", ["mhm", "okay"]),
    ("Platform 4", ["mhm", "okay"]),
    ("okay", ["mhm", "okay"]),
    ("YEAH", ["yeah", "right"]),
    ("Right?", ["yeah", "right"]),
    ("Thanks", ["mhm", "okay"]),
    ("Yeah", ["uh-huh", "mhm"]),
    ("Ja", ["ja", "hm-hm"]),
]
_L_START = "def label(text, backchannels):\n    ...\n"
_L = ("def label(text, backchannels):\n"
      "    if text.lower() in backchannels:\n"
      "        return \"backchannel\"\n"
      "    if \"?\" in text:\n"
      "        return \"question\"\n"
      "    return \"statement\"\n")
_L_CAPS = _L.replace("text.lower() in", "text in")
_L_EQUAL = _L.replace("text.lower() in backchannels", "text.lower() == backchannels")
_L_NO_QUESTION = _L.replace("    if \"?\" in text:\n        return \"question\"\n", "")


def _not_a_label(got, text) -> bool:
    """The learner's function gives a text that is not one of the three labels, such as a Dutch word."""
    return isinstance(got, str) and got not in LABELS


def _list_unused(got, text) -> bool:
    """The function does not use its second argument, the list: it looks the turn up in a list of its own."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "label" and len(node.args.args) >= 2:
            name = node.args.args[1].arg
            used = any(isinstance(n, ast.Name) and n.id == name for part in node.body for n in ast.walk(part))
            return not used
    return False


EXERCISES = {
    "gaps": FunctionBox(
        name="gaps",
        picture=gaps_picture,
        function="gaps",
        start=_G_START,
        solution=_G,
        visible=GAPS_CASES[:1],
        hidden=GAPS_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("pauses", wrong=_G_PAUSES,
                    en="`{call}` gives {got}, and it should give {want}: your list also holds the pause between two stretches of the same speaker. Such a pause lies within one turn. Which pairs of stretches does the exercise ask about?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je lijst bevat ook de pauze tussen twee stukken van dezelfde spreker. Zo'n pauze valt binnen één beurt. Naar welke paren stukken vraagt de opdracht?"),
            Mistake("same", wrong=_G_SAME,
                    en="`{call}` gives {got}, and it should give {want}: your list holds the pauses within one speaker's turn, and leaves out the rest. Look at the comparison in your `if`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je lijst bevat de pauzes binnen de beurt van één spreker, en laat de rest weg. Kijk naar de vergelijking in je `if`."),
            Mistake("sign", wrong=_G_SIGN,
                    en="`{call}` gives {got}, and it should give {want}: the numbers have the opposite sign. A silence should come out positive and an overlap negative. Which do you subtract from which?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de getallen hebben het omgekeerde teken. Een stilte moet positief uitkomen en overlap negatief. Wat trek je van wat af?"),
            Mistake("starts", wrong=_G_STARTS,
                    en="`{call}` gives {got}, and it should give {want}: your function measures from the start of the stretch before. The gap begins where that stretch ends, at place 2.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie meet vanaf het begin van het stuk ervoor. De overgangstijd begint waar dat stuk eindigt, op plaats 2."),
            Mistake("from-zero", wrong=_G_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: your list starts with a number too many. With `i` at 0, `i - 1` is -1, and Python then takes the last stretch of the list. The first stretch has nothing before it: start at 1.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je lijst begint met een getal te veel. Met `i` op 0 is `i - 1` gelijk aan -1, en Python neemt dan het laatste stuk van de lijst. Het eerste stuk heeft niets ervoor: begin bij 1."),
            Mistake("early", wrong=_G_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your function stops after the first pair of stretches. Is the `return` inside the loop? Put it after the loop.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie stopt na het eerste paar stukken. Staat de `return` in de loop? Zet de regel na de loop."),
            Mistake("early-if", wrong=_G_EARLY_IF,
                    en="`{call}` gives {got}, and it should give {want}: your function stops after the first pair of stretches. Is the `return` inside the loop? Put it after the loop.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie stopt na het eerste paar stukken. Staat de `return` in de loop? Zet de regel na de loop."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back the gap in milliseconds at each change of speaker in `talk`.",
                      "nl": "Vervang de `...`: geef bij elke sprekerswisseling in `talk` de overgangstijd terug, in milliseconden."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Where does the speaker change, and what is the new start minus the end before it?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Waar wisselt de spreker, en wat is het nieuwe begin min het einde ervoor?"},
            "almost": {"en": "Right for TALK. But `{call}` gives {got}, and it should give {want}. Does your function work for any conversation, also a short one?",
                       "nl": "Goed voor TALK. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elk gesprek, ook een kort?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back. Is there a `return` after the loop, also for a conversation without any change of speaker?",
                          "nl": "Met `{call}` geeft je functie niets terug. Staat er een `return` na de loop, ook voor een gesprek zonder sprekerswisseling?"},
            "correct": {"en": "Your `gaps` is right in all {n} cases. Look at the picture: the conversation of Sam and Kim, with your numbers at each change of speaker.",
                        "nl": "Je `gaps` klopt in alle {n} gevallen. Kijk naar de afbeelding: het gesprek van Sam en Kim, met jouw getallen bij elke sprekerswisseling."},
        },
    ),
    "label": FunctionBox(
        name="label",
        picture=label_picture,
        function="label",
        start=_L_START,
        solution=_L,
        visible=LABEL_CASES[:1],
        hidden=LABEL_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("not-a-label", matches=_not_a_label,
                    en="`{call}` gives {got}, and that is not one of the three labels. Give back `\"backchannel\"`, `\"question\"` or `\"statement\"`, written exactly like that.",
                    nl="`{call}` geeft {got}, en dat is geen van de drie labels. Geef `\"backchannel\"`, `\"question\"` of `\"statement\"` terug, precies zo geschreven, ook in de Nederlandse les."),
            Mistake("list-unused", matches=_list_unused,
                    en="`{call}` gives {got}, and it should give {want}: your function does not use the list it gets. The check gives it lists of its own: look the turn up in `backchannels`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie doet niets met de lijst die de check meegeeft. De check geeft de functie eigen lijsten, met Engelse woorden, ook in de Nederlandse les: zoek de beurt op in `backchannels`."),
            Mistake("caps", wrong=_L_CAPS,
                    en="`{call}` gives {got}, and it should give {want}. The words in `backchannels` are written in small letters, and a turn can start with a capital. Does your function look the turn up in the same form as the list?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. De woorden in `backchannels` staan in kleine letters, en een beurt kan met een hoofdletter beginnen. Zoekt je functie de beurt op in dezelfde vorm als de lijst?"),
            Mistake("equal", wrong=_L_EQUAL,
                    en="`{call}` gives {got}, and it should give {want}: your function never gives `\"backchannel\"`. A turn is one text, and `backchannels` a list of them: do you ask whether the list holds the turn?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft nooit `\"backchannel\"`. Een beurt is één tekst, en `backchannels` een lijst teksten: vraag je of de lijst de beurt bevat?"),
            Mistake("no-question", wrong=_L_NO_QUESTION,
                    en="`{call}` gives {got}, and it should give {want}: your function never gives `\"question\"`. Does it look for the question mark, and does it get to that rule before the last `return`?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft nooit `\"question\"`. Kijkt je functie naar het vraagteken, en komt de functie bij die regel vóór de laatste `return`?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back `\"backchannel\"`, `\"question\"` or `\"statement\"` for the turn `text`.",
                      "nl": "Vervang de `...`: geef `\"backchannel\"`, `\"question\"` of `\"statement\"` terug voor de beurt `text`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which of the three rules fits this turn first?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Welke van de drie regels past als eerste bij deze beurt?"},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function work for any turn and any list it gets?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke beurt en elke lijst die de check meegeeft?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back. What should it give for a turn that is not in the list and has no question mark?",
                          "nl": "Met `{call}` geeft je functie niets terug. Wat moet de functie geven voor een beurt die niet in de lijst staat en geen vraagteken heeft?"},
            "correct": {"en": "Your `label` is right in all {n} cases. Look at the picture: each turn of the conversation, with your label, and a label by hand next to it.",
                        "nl": "Je `label` klopt in alle {n} gevallen. Kijk naar de afbeelding: elke beurt van het gesprek, met jouw label, en ernaast een label dat de course met de hand gaf."},
        },
    ),
    "evidence": Sort(
        name="evidence",
        groups={
            "ack": {"en": "Only signals: go on, I follow", "nl": "Geeft alleen aan: ga door, ik volg je"},
            "demo": {"en": "Says in other words what was understood", "nl": "Zegt in andere woorden wat er begrepen is"},
            "display": {"en": "Repeats the words that were said", "nl": "Herhaalt de woorden die gezegd zijn"},
        },
        items={
            "uh-huh": {"en": "**Uh-huh.**", "nl": "**Hm-hm.**"},
            "got-it": {"en": "**Got it.**", "nl": "**Helder.**"},
            "quarter": {"en": "**A quarter past eight, then.**", "nl": "**Kwart over acht dus.**"},
            "soon": {"en": "**So in a quarter of an hour.**", "nl": "**Dus over een kwartier.**"},
            "both": {"en": "**Eight fifteen, platform 7.**", "nl": "**Acht uur vijftien, spoor 7.**"},
            "platform": {"en": "**Platform 7.**", "nl": "**Spoor 7.**"},
        },
        solution={"uh-huh": "ack", "got-it": "ack", "quarter": "demo", "soon": "demo", "both": "display",
                  "platform": "display"},
        nudge={
            "uh-huh": {"en": "Could Sam say this without having heard the time at all?",
                       "nl": "Kan Sam dit zeggen zonder de tijd gehoord te hebben?"},
            "got-it": {"en": "Which part of what Kim said comes back in this reply?",
                     "nl": "Welk deel van wat Kim zei komt terug in dit antwoord?"},
            "quarter": {"en": "Are these Kim's words for the time, or Sam's own?",
                        "nl": "Zijn dit de woorden van Kim voor de tijd, of die van Sam zelf?"},
            "soon": {"en": "Did Kim say how long Sam has, or does Sam work that out?",
                     "nl": "Zei Kim hoeveel tijd Sam nog heeft, of maakt Sam dat zelf op?"},
            "both": {"en": "Where did Sam get exactly these words?",
                     "nl": "Waar haalt Sam precies deze woorden vandaan?"},
            "platform": {"en": "Is this how Sam would put it, or how Kim put it?",
                         "nl": "Zegt Sam het hier op een eigen manier, of zoals Kim het zei?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the replies again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de antwoorden opnieuw uit."},
            "empty": {"en": "Put each bold reply under what it shows of what Sam understood.",
                      "nl": "Zet elk vetgedrukt antwoord bij wat het laat zien van wat Sam begreep."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} belongs under another group.",
                      "nl": "{item} hoort bij een andere groep."},
            "correct": {"en": "All {n} are right. Clark and Schaefer order these kinds roughly from weakest to strongest: an acknowledgment, a demonstration, a display. *Uh-huh* and *Got it* would fit any sentence; the other four show what Sam took from this one.",
                        "nl": "Alle {n} kloppen. Clark en Schaefer zetten deze soorten ruwweg van zwakst naar sterkst: een ontvangstbevestiging, een demonstratie, een letterlijke herhaling. *Hm-hm* en *Helder* passen bij elke zin; de andere vier laten zien wat Sam uit deze zin haalde."},
        },
    ),
}

CASES = [
    ("gaps", _G_START, "empty", None),
    ("gaps", _G, "correct", None),
    ("gaps", "def gaps(talk):\n    return [talk[i][1] - talk[i - 1][2] for i in range(1, len(talk)) if talk[i][0] != talk[i - 1][0]]\n", "correct", None),
    ("gaps", "def gaps(talk):\n    out = []\n    for i in range(len(talk) - 1):\n        if talk[i + 1][0] != talk[i][0]:\n            out.append(talk[i + 1][1] - talk[i][2])\n    return out\n", "correct", None),
    ("gaps", _G_PAUSES, "wrong", "pauses"),
    ("gaps", _G_SAME, "wrong", "same"),
    ("gaps", _G_SIGN, "wrong", "sign"),
    ("gaps", _G_STARTS, "wrong", "starts"),
    ("gaps", _G_ZERO, "wrong", "from-zero"),
    ("gaps", _G_EARLY, "wrong", "early"),
    ("gaps", _G_EARLY_IF, "wrong", "early-if"),
    ("gaps", _G.replace("    return out\n", ""), "wrong", "no-return"),
    ("gaps", "def gaps(talk):\n    return [200, -50, 100, 0, 100, 50, 300, 100]\n", "almost", None),
    ("label", _L_START, "empty", None),
    ("label", _L, "correct", None),
    ("label", "def label(text, backchannels):\n    if \"?\" in text:\n        return \"question\"\n    if text.lower() in backchannels:\n        return \"backchannel\"\n    return \"statement\"\n", "correct", None),
    ("label", "def label(text, backchannels):\n    if text.lower() in backchannels:\n        return \"backchannel\"\n    elif \"?\" in text:\n        return \"question\"\n    else:\n        return \"statement\"\n", "correct", None),
    ("label", _L_CAPS, "wrong", "caps"),
    ("label", _L_EQUAL, "wrong", "equal"),
    ("label", _L_NO_QUESTION, "wrong", "no-question"),
    ("label", _L.replace("    return \"statement\"\n", ""), "wrong", "unreached"),
    ("label", _L.replace("\"statement\"", "\"mededeling\""), "wrong", "not-a-label"),
    ("label", _L.replace("in backchannels", "in [\"mhm\", \"uh-huh\", \"yeah\", \"right\", \"okay\"]"), "wrong", "list-unused"),
    ("label", _L.replace("in backchannels", "in [\"mhm\", \"hm-hm\", \"ja\", \"oké\", \"precies\"]"), "wrong", "list-unused"),
    ("label", "def label(text, backchannels):\n    if text.lower() in backchannels:\n        return \"backchannel\"\n    return \"question\"\n", "almost", None),
    ("evidence", {}, "empty", None),
    ("evidence", {"uh-huh": "ack", "got-it": "ack", "quarter": "demo", "soon": "demo", "both": "display",
                  "platform": "display"}, "correct", None),
    ("evidence", {"uh-huh": "ack", "quarter": "demo"}, "wrong", "unfinished"),
    ("evidence", {"quarter": "display"}, "wrong", "wrong:quarter"),
    ("evidence", {"uh-huh": "ack", "got-it": "ack", "quarter": "demo", "soon": "demo", "both": "demo",
                  "platform": "display"}, "wrong", "wrong:both"),
]
