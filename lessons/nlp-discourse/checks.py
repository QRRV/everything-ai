# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-discourse: count the different words two sentences share (overlap), mark each mention of
a text new or old (status), sort ties between two sentences by kind, after the chapters of Halliday and Hasan (1976),
and find two sentences that share a word but do not hang together. The text about Kim's bike with its words and its
mentions, the shuffled orders, the pairs of the sort and the pairs of the choice are the course's own."""
import contextlib
import copy
import html
import io
import itertools
import random
import re

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, with_time_limit

# --- the text --------------------------------------------------------------------------------------------------------

TEXT = {
    "en": ["Kim wanted a cheap bike for the trip to work.", "The bike in the window was cheap.",
           "Kim bought the bike and rode it home.", "Close to home, the chain came off.",
           "Kim fixed the chain with a tool from the shed.", "That night, a storm hit the town.",
           "The storm blew the roof off the shed.", "The next morning, Kim rode past the roof lying in the street."],
    "nl": ["Kim wilde een goedkope fiets voor de rit naar het werk.", "De fiets in de etalage was goedkoop.",
           "Kim kocht de fiets en reed ermee naar huis.", "Vlak bij huis liep de ketting eraf.",
           "Kim zette de ketting terug met gereedschap uit de schuur.", "Die nacht trok een storm over de stad.",
           "De storm blies het dak van de schuur.", "De volgende ochtend reed Kim langs het dak dat op straat lag."],
}
# The words of each sentence that carry content, in small letters and in their base form.
WORDS = {
    "en": [["kim", "want", "cheap", "bike", "trip", "work"], ["bike", "window", "cheap"],
           ["kim", "buy", "bike", "ride", "home"], ["close", "home", "chain", "come"],
           ["kim", "fix", "chain", "tool", "shed"], ["night", "storm", "hit", "town"],
           ["storm", "blow", "roof", "shed"], ["next", "morning", "kim", "ride", "roof", "lie", "street"]],
    "nl": [["kim", "willen", "goedkoop", "fiets", "rit", "werk"], ["fiets", "etalage", "goedkoop"],
           ["kim", "kopen", "fiets", "rijden", "huis"], ["vlak", "huis", "ketting", "lopen"],
           ["kim", "terugzetten", "ketting", "gereedschap", "schuur"], ["nacht", "storm", "trekken", "stad"],
           ["storm", "blazen", "dak", "schuur"], ["volgend", "ochtend", "rijden", "kim", "dak", "liggen", "straat"]],
}
# The things each sentence mentions, in order; "it" in sentence 3 is the bike again.
MENTIONS = {
    "en": [["Kim", "bike", "work"], ["bike", "window"], ["Kim", "bike", "bike", "home"], ["home", "chain"],
           ["Kim", "chain", "tool", "shed"], ["night", "storm", "town"], ["storm", "roof", "shed"],
           ["morning", "Kim", "roof", "street"]],
    "nl": [["Kim", "fiets", "werk"], ["fiets", "etalage"], ["Kim", "fiets", "fiets", "huis"], ["huis", "ketting"],
           ["Kim", "ketting", "gereedschap", "schuur"], ["nacht", "storm", "stad"], ["storm", "dak", "schuur"],
           ["ochtend", "Kim", "dak", "straat"]],
}
SHUFFLE = [6, 3, 1, 0, 7, 2, 5, 4]   # the shuffled order of the status picture
LABELS = ("new", "old")


def overlap(a, b):
    both = [w for w in a if w in b]
    return len(set(both))


def status(mentions):
    seen = []
    out = []
    for m in mentions:
        if m in seen:
            out.append("old")
        else:
            out.append("new")
            seen.append(m)
    return out


def flat(text):
    out = []
    for sentence in text:
        for m in sentence:
            out.append(m)
    return out


def mentions_of(thing, text):
    return [i for i in range(len(text)) if thing in text[i]]


def scores(words, order) -> list:
    """The overlap at each gap between neighbors, with the sentences in `order`."""
    return [overlap(words[order[i - 1]], words[order[i]]) for i in range(1, len(order))]


def split_back(labels: list, text: list) -> list:
    """A flat list of labels, cut into one list per sentence of `text`."""
    out, k = [], 0
    for sentence in text:
        out.append(labels[k:k + len(sentence)])
        k += len(sentence)
    return out


def starts_old(mentions, order) -> int:
    """How many sentences after the first start with something old, with the sentences in `order`."""
    text = [mentions[i] for i in order]
    rows = split_back(status(flat(text)), text)
    return sum(row[0] == "old" for row in rows[1:])


def all_orders(lang: str = "en") -> dict:
    """Over every order of the eight sentences: how often the written order scores higher, the same or lower,
    on shared words and on sentences that start with something old."""
    words, mentions = WORDS[lang], MENTIONS[lang]
    order = list(range(len(words)))
    base = (sum(scores(words, order)), starts_old(mentions, order))
    out = {"words": [0, 0, 0], "old": [0, 0, 0], "n": 0}
    for other in itertools.permutations(order):
        out["n"] += 1
        for key, mine, theirs in (("words", base[0], sum(scores(words, other))),
                                  ("old", base[1], starts_old(mentions, other))):
            out[key][0 if mine > theirs else 1 if mine == theirs else 2] += 1
    return out


def demo_order(seed: int) -> list:
    order = list(range(len(TEXT["en"])))
    random.Random(int(seed)).shuffle(order)
    return order


# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_LINE = "font-family:var(--ea-font-ui);font-size:0.85rem;color:var(--ea-ink);margin:0.1rem 0"
_NUM = "display:inline-block;min-width:1.4rem;color:var(--ea-ink-3);font-family:var(--ea-font-code);font-size:0.78rem"


def _note(text: str) -> str:
    """A line under a picture; a word between asterisks is set in italics, as in the lesson's text."""
    marked = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", html.escape(text))
    return f'<div style="{_UI}">{marked}</div>'


def _bare(text: str) -> str:
    """The same line without its asterisks, for the alt text."""
    return text.replace("*", "")


def _code(value) -> str:
    """A value the way Python shows it, with the strings in double quotes."""
    if isinstance(value, str):
        return '"' + value + '"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_code(v) for v in value) + "]"
    return repr(value)


def _short(value, limit: int = 44) -> str:
    text = _code(value)
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _numbered(lang: str, order: list) -> str:
    return "".join(f'<div style="{_LINE}"><span style="{_NUM}">{i + 1}</span>{html.escape(TEXT[lang][i])}</div>'
                   for i in order)


_GRID = {
    "en": {"title": "Who and what comes back: one row per thing, one column per sentence",
           "thing": "Thing",
           "legend": "After the entity grid of Barzilay and Lapata (2008), without the roles; the text is the course's "
                     "own. Kim comes back all through the text, the bike in the first three sentences, and the storm "
                     "in two sentences in a row."},
    "nl": {"title": "Wie en wat er terugkomt: een rij per ding, een kolom per zin",
           "thing": "Ding",
           "legend": "Naar het entity grid van Barzilay en Lapata (2008), zonder de rollen; de tekst is van de course "
                     "zelf. Kim komt door de hele tekst terug, de fiets in de eerste drie zinnen en de storm in twee "
                     "zinnen na elkaar."},
}


def grid_picture(lang: str) -> str:
    w, mentions = _GRID[lang], MENTIONS[lang]
    things = []
    for m in flat(mentions):
        if m not in things:
            things.append(m)
    head = "".join(f'<th style="{_HEAD};text-align:center">{i + 1}</th>' for i in range(len(mentions)))
    body = ""
    for thing in things:
        cells = "".join(f'<td style="{_CELL};text-align:center">{"●" if thing in sentence else ""}</td>'
                        for sentence in mentions)
        body += f'<tr><td style="{_CELL}">{html.escape(thing)}</td>{cells}</tr>'
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.4rem 0">'
             f'<tr><th style="{_HEAD}">{w["thing"]}</th>{head}</tr>{body}</table></div>')
    alt = (f'{w["title"]}. ' + " ".join(TEXT[lang]) + " "
           + "; ".join(f"{t}: {', '.join(str(i + 1) for i in mentions_of(t, mentions))}" for t in things)
           + f". {_bare(w['legend'])}")
    return yours.figure(w["title"], _numbered(lang, range(len(mentions))) + table + _note(w["legend"]), alt=alt)


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


_OVERLAP_WORDS = {
    "en": {"title": "Your overlap, at work: the words that neighbors share",
           "fine": "Your counts are right at all {n} gaps. The count drops to 0 where the text turns to the storm; "
                   "the last sentence picks up the roof again.",
           "differs": "Where a count is in purple, your overlap differs from the right count: {want}.",
           "error": "Your function stops with an error at the gap after sentence {k}.",
           "slow": "Your function takes too long."},
    "nl": {"title": "Jouw overlap aan het werk: de woorden die buren delen",
           "fine": "Je aantallen kloppen bij alle {n} overgangen. Het aantal zakt naar 0 waar de tekst overgaat op de "
                   "storm; de laatste zin pakt het dak weer op.",
           "differs": "Waar een aantal paars is, wijkt je overlap af van het goede aantal: {want}.",
           "error": "Je functie stopt met een fout bij de overgang na zin {k}.",
           "slow": "Je functie doet er te lang over."},
}


def _bar(value, off: bool) -> str:
    color = "var(--ea-loss)" if off else "var(--ea-ink-3)"
    text = "var(--ea-loss-text)" if off else "var(--ea-ink-2)"
    width = 18 * max(0, min(value, 8)) if type(value) is int else 0
    shown = str(value) if type(value) is int else _short(value, 24)
    return (f'<span style="display:inline-block;width:{width}px;height:0.55rem;background:{color};'
            f'border-radius:2px;vertical-align:middle;margin-right:0.35rem"></span>'
            f'<span style="font-family:var(--ea-font-code);font-size:0.78rem;color:{text};margin-right:0.5rem">'
            f'{html.escape(shown)}</span>')


def overlap_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    lang, w = seen.lang, _OVERLAP_WORDS[seen.lang]
    words, text = WORDS[lang], TEXT[lang]
    rows, alt_rows, end, all_right, want = [], [], "", True, scores(words, list(range(len(words))))
    for i, sentence in enumerate(text):
        if i:
            try:
                got = _call(fn, words[i - 1], words[i])
            except Timeout:
                end = w["slow"]
                break
            if got is _STOPPED:
                end = w["error"].format(k=i)
                break
            shared = sorted(set(x for x in words[i - 1] if x in words[i]))
            off = not (type(got) is int and got == want[i - 1])
            all_right = all_right and not off
            rows.append(f'<div style="{_UI};margin:0.05rem 0 0.05rem 1.4rem">{_bar(got, off)}'
                        f'<i>{html.escape(", ".join(shared))}</i></div>')
            alt_rows.append(f"{_short(got, 24)} ({', '.join(shared) or '-'})")
        rows.append(f'<div style="{_LINE}"><span style="{_NUM}">{i + 1}</span>{html.escape(sentence)}</div>')
    after = end or (w["fine"].format(n=len(want)) if all_right else w["differs"].format(want=_code(want)))
    alt = f'{w["title"]}. ' + " ".join(text) + " " + "; ".join(alt_rows) + f". {_bare(after)}"
    return yours.figure(w["title"], "".join(rows) + _note(after), alt=alt)


_STATUS_WORDS = {
    "en": {"title": "Your status, at work: what is new and what is old",
           "written": "As written", "shuffled": "Shuffled",
           "key": "Filled: new. Outlined: old.",
           "fine": "Your labels are right. As written, {a} of the {n} sentences after the first start with something "
                   "old; shuffled, {b}.",
           "differs": "Where a thing has a purple edge, your label differs from the right one.",
           "error": "Your function stops with an error on this text.",
           "slow": "Your function takes too long.",
           "not_list": "Your status gives {got}, not a list with a label for each mention.",
           "count": "The text has {n} mentions, and the length of your list is {got}."},
    "nl": {"title": "Jouw status aan het werk: wat nieuw is en wat oud",
           "written": "Zoals geschreven", "shuffled": "Door elkaar",
           "key": "Gevuld: nieuw. Omlijnd: oud.",
           "fine": "Je labels kloppen. Zoals geschreven beginnen {a} van de {n} zinnen na de eerste met iets ouds; "
                   "door elkaar {b}.",
           "differs": "Waar een ding een paarse rand heeft, wijkt je label af van het goede.",
           "error": "Je functie stopt met een fout bij deze tekst.",
           "slow": "Je functie doet er te lang over.",
           "not_list": "Je status geeft {got}, geen lijst met een label bij elke mention.",
           "count": "De tekst heeft {n} mentions, en je lijst heeft lengte {got}."},
}


def _chip(thing: str, label, right: str) -> str:
    off = label != right
    if label == "new":
        look = "background:var(--ea-ink);color:var(--ea-page);border:1.5px solid var(--ea-ink)"
    elif label == "old":
        look = "background:transparent;color:var(--ea-ink-2);border:1.5px solid var(--ea-ink-3)"
    else:
        look = "background:transparent;color:var(--ea-loss-text);border:1.5px dashed var(--ea-loss)"
    if off:
        look += ";box-shadow:0 0 0 2px var(--ea-loss)"
    return (f'<span style="display:inline-block;padding:0.02rem 0.45rem;margin:0.12rem 0.2rem;border-radius:999px;'
            f'font-family:var(--ea-font-ui);font-size:0.78rem;{look}">{html.escape(thing)}</span>')


def _panel(lang: str, order: list, labels: list, head: str) -> tuple:
    """The sentences in `order` with their mentions as chips: the html, and whether every label is right."""
    text = [MENTIONS[lang][i] for i in order]
    right = split_back(status(flat(text)), text)
    mine = split_back(labels, text)
    rows, all_right = "", True
    for k, i in enumerate(order):
        chips = "".join(_chip(t, mine[k][j], right[k][j]) for j, t in enumerate(text[k]))
        all_right = all_right and mine[k] == right[k]
        rows += (f'<div style="{_LINE}"><span style="{_NUM}">{i + 1}</span>{chips}</div>')
    return (f'<div style="flex:1 1 15rem;min-width:13rem"><div style="{_HEAD};border:none;padding-left:0">{head}</div>'
            f'{rows}</div>'), all_right


def _labels(fn, lang: str, order: list, w: dict):
    """The learner's labels for the mentions with the sentences in `order`, or a line that says what went wrong."""
    text = [MENTIONS[lang][i] for i in order]
    mentions = flat(text)
    try:
        got = _call(fn, mentions)
    except Timeout:
        return None, w["slow"]
    if got is _STOPPED:
        return None, w["error"]
    if not isinstance(got, list):
        return None, w["not_list"].format(got=_short(got))
    if len(got) != len(mentions):
        return None, w["count"].format(got=len(got), n=len(mentions))
    return got, ""


def status_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    lang, w = seen.lang, _STATUS_WORDS[seen.lang]
    written = list(range(len(TEXT[lang])))
    panels, all_right = [], True
    for order, head in ((written, w["written"]), (SHUFFLE, w["shuffled"])):
        labels, problem = _labels(fn, lang, order, w)
        if problem:
            return yours.figure(w["title"], _note(problem), alt=f'{w["title"]}. {problem}')
        panel, right = _panel(lang, order, labels, head)
        panels.append(panel)
        all_right = all_right and right
    n = len(written) - 1
    after = (w["fine"].format(a=starts_old(MENTIONS[lang], written), n=n, b=starts_old(MENTIONS[lang], SHUFFLE))
             if all_right else w["differs"])
    body = f'<div style="display:flex;flex-wrap:wrap;gap:0.4rem 1.5rem">{"".join(panels)}</div>'
    alt = (f'{w["title"]}. ' + " ".join(TEXT[lang]) + " " + _bare(w["key"]) + " " + _bare(after))
    return yours.figure(w["title"], body + _note(w["key"]) + _note(after), alt=alt)


_DEMO_WORDS = {
    "en": {"title": "Shuffle {seed}: the same eight sentences in another order",
           "head": ["", "Shared words between neighbors", "Sentences that start with something old"],
           "rows": ["As written", "This order"],
           "key": "Filled: new. Outlined: old. The numbers are the places of the sentences as written."},
    "nl": {"title": "Volgorde {seed}: dezelfde acht zinnen in een andere volgorde",
           "head": ["", "Gedeelde woorden tussen buren", "Zinnen die beginnen met iets ouds"],
           "rows": ["Zoals geschreven", "Deze volgorde"],
           "key": "Gevuld: nieuw. Omlijnd: oud. De getallen zijn de plaatsen van de zinnen zoals geschreven."},
}


def demo_picture(seed: int, lang: str) -> str:
    w = _DEMO_WORDS[lang]
    seed = max(1, min(99, int(seed)))
    order, written = demo_order(seed), list(range(len(TEXT[lang])))
    text = [MENTIONS[lang][i] for i in order]
    labels = split_back(status(flat(text)), text)
    rows = ""
    for k, i in enumerate(order):   # one row per sentence: its place as written, the sentence, its mentions
        chips = "".join(_chip(t, labels[k][j], labels[k][j]) for j, t in enumerate(text[k]))
        rows += (f'<tr><td style="{_CELL};vertical-align:top;color:var(--ea-ink-3);font-family:var(--ea-font-code)">'
                 f'{i + 1}</td><td style="{_CELL};vertical-align:top">{html.escape(TEXT[lang][i])}</td>'
                 f'<td style="{_CELL};min-width:12rem">{chips}</td></tr>')
    sentences = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.2rem 0">{rows}'
                 f'</table></div>')
    cells = []
    for name, o in zip(w["rows"], (written, order)):
        cells.append(f'<tr><td style="{_CELL}">{name}</td><td style="{_CELL}">{sum(scores(WORDS[lang], o))}</td>'
                     f'<td style="{_CELL}">{starts_old(MENTIONS[lang], o)}</td></tr>')
    head = "".join(f'<th style="{_HEAD}">{h}</th>' for h in w["head"])
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.4rem 0">'
             f'<tr>{head}</tr>{"".join(cells)}</table></div>')
    title = w["title"].format(seed=seed)
    body = sentences
    alt = (f"{title}. " + " ".join(TEXT[lang][i] for i in order) + " "
           + "; ".join(f"{name}: {sum(scores(WORDS[lang], o))}, {starts_old(MENTIONS[lang], o)}"
                       for name, o in zip(w["rows"], (written, order))) + ". " + _bare(w["key"]))
    return yours.figure(title, body + table + _note(w["key"]), alt=alt)


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["overlap", "status", "ties", "bank"]


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

# exercise 1: overlap; the words of the cases besides the lesson's are the same in English and Dutch
OVERLAP_CASES = [
    (Named("WORDS[0]", WORDS["en"][0]), Named("WORDS[1]", WORDS["en"][1])),
    (["kim", "sam", "pizza"], ["pizza", "taxi"]),
    (["storm", "storm", "park"], ["storm"]),
    (["tram"], ["taxi"]),
    ([], ["kim"]),
    (["film", "radio", "hotel"], ["hotel", "radio", "film"]),
    (["taxi", "park"], ["park", "park", "taxi"]),
]
_O_START = "def overlap(a, b):\n    ...\n"
_O = ("def overlap(a, b):\n"
      "    both = [w for w in a if w in b]\n"
      "    return len(set(both))\n")
_O_TWICE = _O.replace("len(set(both))", "len(both)")
_O_ALL = _O.replace("len(set(both))", "len(set(a + b))")
_O_SELF = _O.replace("if w in b]", "if w in a]")


def _not_a_number(got, text) -> bool:
    return isinstance(got, (list, set, tuple, str))


# exercise 2: status
STATUS_CASES = [
    (Named("flat(MENTIONS)", flat(MENTIONS["en"])),),
    (["sam", "kim", "sam"],),
    (["taxi"],),
    ([],),
    (["park", "park", "park"],),
    (["tram", "taxi", "radio"],),
    (["kim", "sam", "kim", "sam"],),
]
_S_START = "def status(mentions):\n    ...\n"
_S = ("def status(mentions):\n"
      "    seen = []\n"
      "    out = []\n"
      "    for m in mentions:\n"
      "        if m in seen:\n"
      "            out.append(\"old\")\n"
      "        else:\n"
      "            out.append(\"new\")\n"
      "            seen.append(m)\n"
      "    return out\n")
_S_FORGETS = _S.replace("            seen.append(m)\n", "")
_S_TOO_SOON = ("def status(mentions):\n"
               "    seen = []\n"
               "    out = []\n"
               "    for m in mentions:\n"
               "        seen.append(m)\n"
               "        if m in seen:\n"
               "            out.append(\"old\")\n"
               "        else:\n"
               "            out.append(\"new\")\n"
               "    return out\n")
_S_SWAPPED = _S.replace("\"old\"", "\"X\"").replace("\"new\"", "\"old\"").replace("\"X\"", "\"new\"")
_S_EARLY = _S.replace("    return out\n", "        return out\n")


def _not_labels(got, text) -> bool:
    return isinstance(got, list) and any(x not in LABELS for x in got)


EXERCISES = {
    "overlap": FunctionBox(
        name="overlap",
        picture=overlap_picture,
        function="overlap",
        start=_O_START,
        solution=_O,
        visible=OVERLAP_CASES[:1],
        hidden=OVERLAP_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("not-a-number", matches=_not_a_number,
                    en="`{call}` should give a number: how many different words the two sentences share. What tells you how many values a list or a set holds?",
                    nl="`{call}` moet een getal geven: hoeveel verschillende woorden de twee zinnen delen. Wat vertelt je hoeveel waarden een lijst of een set bevat?"),
            Mistake("twice", wrong=_O_TWICE,
                    en="`{call}` gives {got}, and it should give {want}: a word that is twice in the first sentence counts twice in your function. The exercise counts different words. What keeps each value once?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een woord dat twee keer in de eerste zin staat, telt bij jouw functie twee keer. De opdracht telt verschillende woorden. Wat houdt elke waarde maar één keer over?"),
            Mistake("all-words", wrong=_O_ALL,
                    en="`{call}` gives {got}, and it should give {want}: your function counts all the different words of both sentences. Which words does the exercise ask for?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt alle verschillende woorden van beide zinnen. Naar welke woorden vraagt de opdracht?"),
            Mistake("self", wrong=_O_SELF,
                    en="`{call}` gives {got}, and it should give {want}: your function counts the different words of the first sentence. In which list should each word be looked up?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt de verschillende woorden van de eerste zin. In welke lijst moet je elk woord opzoeken?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back how many different words `a` and `b` have in common.",
                      "nl": "Vervang de `...`: geef terug hoeveel verschillende woorden `a` en `b` gemeen hebben."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which words do the two lists share, and how many different ones are there?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Welke woorden delen de twee lijsten, en hoeveel verschillende zijn dat?"},
            "almost": {"en": "Right for the first two sentences. But `{call}` gives {got}, and it should give {want}. Does your function work for any two lists of words?",
                       "nl": "Goed voor de eerste twee zinnen. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke twee lijsten met woorden?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back: it has a `return`, but Python does not get to it. Is the `return` the last line, outside any `if`?",
                          "nl": "Met `{call}` geeft je functie niets terug: er staat een `return` in, maar Python komt er niet aan toe. Is de `return` de laatste regel, buiten elke `if`?"},
            "correct": {"en": "Your `overlap` is right in all {n} cases. Look at the picture: the text, with your count at each gap between two sentences.",
                        "nl": "Je `overlap` klopt in alle {n} gevallen. Kijk naar de afbeelding: de tekst, met jouw aantal bij elke overgang tussen twee zinnen."},
        },
    ),
    "status": FunctionBox(
        name="status",
        picture=status_picture,
        function="status",
        start=_S_START,
        solution=_S,
        visible=STATUS_CASES[:1],
        hidden=STATUS_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("not-labels", matches=_not_labels,
                    en="With `{call}`, your list holds something else than `\"new\"` and `\"old\"`. Give back a list with `\"new\"` or `\"old\"` for each mention, written exactly like that, also in the Dutch lesson.",
                    nl="Met `{call}` bevat je lijst iets anders dan `\"new\"` en `\"old\"`. Geef een lijst terug met `\"new\"` of `\"old\"` bij elke mention, precies zo geschreven, ook in de Nederlandse les."),
            Mistake("forgets", wrong=_S_FORGETS,
                    en="With `{call}`, every mention comes out `\"new\"`, also one that came before. Does your function remember the mentions it has seen?",
                    nl="Met `{call}` komt elke mention eruit als `\"new\"`, ook een die al eerder kwam. Onthoudt je functie de mentions die al voorbijkwamen?"),
            Mistake("too-soon", wrong=_S_TOO_SOON,
                    en="With `{call}`, every mention comes out `\"old\"`, also the first time. Does your function look in `seen` before it adds the mention?",
                    nl="Met `{call}` komt elke mention eruit als `\"old\"`, ook de eerste keer. Kijkt je functie in `seen` voordat de mention erbij komt?"),
            Mistake("swapped", wrong=_S_SWAPPED,
                    en="With `{call}`, the two labels are the wrong way round: the first mention comes out `\"old\"`. A mention the text has seen before is old.",
                    nl="Met `{call}` zijn de twee labels omgedraaid: de eerste mention komt eruit als `\"old\"`. Een mention die de tekst al eerder had, is oud: `\"old\"`."),
            Mistake("early", wrong=_S_EARLY,
                    en="`{call}` gives {got}: your function stops after the first mention. Is the `return` inside the loop? Put it after the loop.",
                    nl="`{call}` geeft {got}: je functie stopt na de eerste mention. Staat de `return` in de loop? Zet de regel na de loop."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back `\"new\"` or `\"old\"` for each mention in `mentions`.",
                      "nl": "Vervang de `...`: geef `\"new\"` of `\"old\"` terug bij elke mention in `mentions`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Has the text mentioned this thing before, or is this the first time?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Heeft de tekst dit ding al eerder genoemd, of is dit de eerste keer?"},
            "almost": {"en": "Right for the text about Kim. But `{call}` gives {got}, and it should give {want}. Does your function work for any list of mentions, also a short one?",
                       "nl": "Goed voor de tekst over Kim. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke lijst mentions, ook een korte?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back. Is there a `return` after the loop, also for an empty list?",
                          "nl": "Met `{call}` geeft je functie niets terug. Staat er een `return` na de loop, ook voor een lege lijst?"},
            "correct": {"en": "Your `status` is right in all {n} cases. Look at the picture: each mention as new or old, in the text as written and shuffled.",
                        "nl": "Je `status` klopt in alle {n} gevallen. Kijk naar de afbeelding: elke mention als nieuw of oud, in de tekst zoals geschreven en door elkaar."},
        },
    ),
    "ties": Sort(
        name="ties",
        groups={
            "ref": {"en": "A small word stands in for something named before",
                    "nl": "Een klein woord staat voor iets wat eerder genoemd is"},
            "conj": {"en": "Says with one word how the second follows on the first",
                     "nl": "Zegt met één woord hoe de tweede zin op de eerste volgt"},
            "lex": {"en": "Repeats a word, or uses a related word",
                    "nl": "Herhaalt een woord, of gebruikt een woord dat er in betekenis bij hoort"},
        },
        items={
            "museum": {"en": "**Sam called the museum. It was closed.**", "nl": "**Sam belde het museum. Het was dicht.**"},
            "glove": {"en": "**Kim lost a glove. Nobody has seen it since.**",
                      "nl": "**Kim verloor een handschoen. Niemand heeft die sindsdien gezien.**"},
            "icy": {"en": "**The road was icy. So Sam stayed home.**", "nl": "**De weg was glad. Dus bleef Sam thuis.**"},
            "tickets": {"en": "**The tickets were cheap. But we stayed home.**",
                        "nl": "**De kaartjes waren goedkoop. Maar we bleven thuis.**"},
            "cake": {"en": "**Sam baked a cake. The cake was gone by noon.**",
                     "nl": "**Sam bakte een taart. De taart was om twaalf uur op.**"},
            "engine": {"en": "**Kim bought a new car. The engine is very quiet.**",
                       "nl": "**Kim kocht een nieuwe auto. De motor is heel stil.**"},
        },
        solution={"museum": "ref", "glove": "ref", "icy": "conj", "tickets": "conj", "cake": "lex", "engine": "lex"},
        nudge={
            "museum": {"en": "What was closed: is it named again in the second sentence?",
                       "nl": "Wat was er dicht: wordt dat in de tweede zin opnieuw genoemd?"},
            "glove": {"en": "What has nobody seen, and is it named again, or only pointed at?",
                      "nl": "Wat heeft niemand gezien, en wordt het opnieuw genoemd, of alleen aangewezen?"},
            "icy": {"en": "Which word tells you that the second sentence is a result of the first?",
                    "nl": "Welk woord vertelt dat de tweede zin een gevolg is van de eerste?"},
            "tickets": {"en": "Does any word come back, or does one small word set the second sentence against the first?",
                        "nl": "Komt er een woord terug, of zet één klein woord de tweede zin tegenover de eerste?"},
            "cake": {"en": "Which word of the first sentence do you meet again?",
                     "nl": "Welk woord uit de eerste zin kom je opnieuw tegen?"},
            "engine": {"en": "Is there a word that points back, or a word that belongs with one in the first sentence?",
                       "nl": "Is er een woord dat terugverwijst, of een woord dat hoort bij een woord in de eerste zin?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the pairs again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de paren opnieuw uit."},
            "empty": {"en": "Put each bold pair under the tie that holds its two sentences together.",
                      "nl": "Zet elk vetgedrukt paar bij de band die de twee zinnen bij elkaar houdt."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} belongs under another group.",
                      "nl": "{item} hoort bij een andere groep."},
            "correct": {"en": "All {n} are right. These are three of the five kinds of ties in Halliday and Hasan's book: reference, conjunction and lexical cohesion. *The engine* is also something a reader infers from *a new car*: it is new in the text, and yet expected.",
                        "nl": "Alle {n} kloppen. Dit zijn drie van de vijf soorten banden uit het boek van Halliday en Hasan: verwijzing, conjunctie en lexicale cohesie. *De motor* is ook iets wat een lezer afleidt uit *een nieuwe auto*: nieuw in de tekst, en toch verwacht."},
        },
    ),
}

CASES = [
    ("overlap", _O_START, "empty", None),
    ("overlap", _O, "correct", None),
    ("overlap", "def overlap(a, b):\n    return len(set([w for w in b if w in a]))\n", "correct", None),
    ("overlap", "def overlap(a, b):\n    both = []\n    for w in a:\n        if w in b:\n            both.append(w)\n    return len(set(both))\n", "correct", None),
    ("overlap", _O_TWICE, "wrong", "twice"),
    ("overlap", _O_ALL, "wrong", "all-words"),
    ("overlap", _O_SELF, "wrong", "self"),
    ("overlap", "def overlap(a, b):\n    return sorted(set([w for w in a if w in b]))\n", "wrong", "not-a-number"),
    ("overlap", _O.replace("    return len(set(both))\n", "    len(set(both))\n"), "wrong", "no-return"),
    ("overlap", "def overlap(a, b):\n    return 2\n", "almost", None),
    ("status", _S_START, "empty", None),
    ("status", _S, "correct", None),
    ("status", "def status(mentions):\n    out = []\n    for i in range(len(mentions)):\n        if mentions[i] in mentions[:i]:\n            out.append(\"old\")\n        else:\n            out.append(\"new\")\n    return out\n", "correct", None),
    ("status", _S_FORGETS, "wrong", "forgets"),
    ("status", _S_TOO_SOON, "wrong", "too-soon"),
    ("status", _S_SWAPPED, "wrong", "swapped"),
    ("status", _S_EARLY, "wrong", "early"),
    ("status", _S.replace("\"old\"", "\"oud\"").replace("\"new\"", "\"nieuw\""), "wrong", "not-labels"),
    ("status", _S.replace("    return out\n", ""), "wrong", "no-return"),
    ("status", "def status(mentions):\n    return " + repr(status(flat(MENTIONS["en"]))) + "\n", "almost", None),
    ("ties", {}, "empty", None),
    ("ties", {"museum": "ref", "glove": "ref", "icy": "conj", "tickets": "conj", "cake": "lex", "engine": "lex"},
     "correct", None),
    ("ties", {"museum": "ref", "icy": "conj"}, "wrong", "unfinished"),
    ("ties", {"engine": "ref"}, "wrong", "wrong:engine"),
    ("ties", {"museum": "ref", "glove": "ref", "icy": "conj", "tickets": "lex", "cake": "lex", "engine": "lex"},
     "wrong", "wrong:tickets"),
]
