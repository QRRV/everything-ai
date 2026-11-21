# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-pragmatics: find the words on a scale that a listener takes to be false when a speaker
says a weaker one (implicated), find the one thing a phrase such as *the cat that sleeps* refers to, or None when
there is not exactly one (the_one), sort utterances by what the speaker does with them, and choose the sentence that
takes back what was meant without taking back what was said. The world of four animals comes from the lesson
nlp-meaning-and-truth; the scales other than all, most, many and some, the six cats of the demo, its reading of
*most* as more than half, and the sentences of the sort and the choice are the course's own."""
import contextlib
import copy
import html
import io
import re

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, with_time_limit

# --- the scales ------------------------------------------------------------------------------------------------------

# Strongest word first. SCALE is Horn's scale of quantifiers (via the SEP entry on implicature); HEAT and OFTEN are the
# course's own examples.
SCALES = {
    "en": {"SCALE": ["all", "most", "many", "some"], "HEAT": ["hot", "warm"], "OFTEN": ["always", "often", "sometimes"]},
    "nl": {"SCALE": ["alle", "de meeste", "veel", "sommige"], "HEAT": ["heet", "warm"], "OFTEN": ["altijd", "vaak", "soms"]},
}
_TO_NL = {e: n for key in SCALES["en"] for e, n in zip(SCALES["en"][key], SCALES["nl"][key])}


def implicated(scale, word):
    for i in range(len(scale)):
        if scale[i] == word:
            return scale[:i]
    return []


# --- the world of nlp-meaning-and-truth ------------------------------------------------------------------------------

WORD = {
    "en": {"cat": "cat", "dog": "dog", "mouse": "mouse", "unicorn": "unicorn", "sleeps": "sleeps", "chases": "chases"},
    "nl": {"cat": "kat", "dog": "hond", "mouse": "muis", "unicorn": "eenhoorn", "sleeps": "slaapt",
           "chases": "achtervolgt"},
}
THINGS = ["Tom", "Mia", "Rex", "Pip"]
KIND = {"Tom": "cat", "Mia": "cat", "Rex": "dog", "Pip": "mouse"}


def _world(sleeps: list, chases: list, lang: str) -> dict:
    w = WORD[lang]
    out = {w[k]: [t for t in THINGS if KIND[t] == k] for k in ("cat", "dog", "mouse")}
    out[w["sleeps"]] = list(sleeps)
    out[w["chases"]] = [tuple(p) for p in chases]
    return out


WORLD = {lang: _world(["Mia", "Rex"], [("Tom", "Pip"), ("Rex", "Tom")], lang) for lang in WORD}
NIGHT = {lang: _world(["Tom", "Mia", "Rex"], [("Pip", "Tom")], lang) for lang in WORD}


def says_some(things, group):
    inside = [thing for thing in things if thing in group]
    return len(inside) > 0


def means_some(things, group):
    inside = [thing for thing in things if thing in group]
    return len(inside) > 0 and len(inside) < len(things)


def the_one(things, group):
    inside = [thing for thing in things if thing in group]
    if len(inside) == 1:
        return inside[0]
    return None


# --- the demo: six cats, three words ---------------------------------------------------------------------------------

CATS = ["Tom", "Mia", "Bo", "Lot", "Kiki", "Max"]
DEMO_WORDS = {"en": ["all", "most", "some"], "nl": ["alle", "de meeste", "sommige"]}
DEMO_LABELS = {"en": ["All cats sleep", "Most cats sleep", "Some cats sleep"],
               "nl": ["Alle katten slapen", "De meeste katten slapen", "Sommige katten slapen"]}


def says(k: int, count: int, total: int = len(CATS)) -> bool:
    """What demo word k says in logic, with `count` of `total` cats asleep: all, more than half (the course's own
    reading of *most*), at least one."""
    return [count == total, count * 2 > total, count > 0][k]


def means(k: int, count: int, total: int = len(CATS)) -> bool:
    """What a listener takes demo word k to mean: what it says, and the stronger words false."""
    return says(k, count, total) and not any(says(j, count, total) for j in range(k))


def best_word(count: int, total: int = len(CATS)):
    """The strongest demo word that is true: what a speaker who keeps to Quantity says; None when none is."""
    for k in range(3):
        if says(k, count, total):
            return k
    return None


# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = ";color:var(--ea-loss-text)"
_CODE = ";font-family:var(--ea-font-code);font-size:0.78rem"
_CAPTION = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin:0 0 0.2rem"


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


def _in_dutch(value):
    if isinstance(value, str):
        return _TO_NL.get(value, value)
    if isinstance(value, list):
        return [_in_dutch(v) for v in value]
    return value


def show_words(value) -> dict:
    """How {got} and {want} show a list of words: the words of the scales in the learner's language."""
    return {"en": _code(value), "nl": _code(_in_dutch(value))}


_SCALE_WORDS = {
    "en": {"title": "A scale, strongest word first: saying some",
           "stronger": "stronger", "said": "said", "false": "taken as false",
           "legend": "Someone who says *some cats sleep* could have said *many*, *most* or *all*, and did not. A "
                     "listener takes those stronger words to be false: *some*, but not *many*, not *most* and not *all*."},
    "nl": {"title": "Een schaal, het sterkste woord eerst: wie sommige zegt",
           "stronger": "sterker", "said": "gezegd", "false": "opgevat als onwaar",
           "legend": "Wie zegt dat *sommige* katten slapen, had *veel*, *de meeste* of *alle* kunnen zeggen, en deed "
                     "dat niet. Een luisteraar vat die sterkere woorden op als onwaar: *sommige*, maar niet *veel*, niet "
                     "*de meeste* en niet *alle*."},
}


def idea_picture(lang: str) -> str:
    """Horn's scale as four boxes, the said word with a bold border, the stronger ones struck through."""
    w, scale = _SCALE_WORDS[lang], SCALES[lang]["SCALE"]
    box_w, box_h, gap, left, top = 128, 46, 22, 10, 40
    width = left * 2 + len(scale) * box_w + (len(scale) - 1) * gap
    height = top + box_h + 46
    parts = [f'<line x1="{left + 8}" y1="18" x2="{width - left - 8}" y2="18" style="stroke:var(--ea-ink-3);'
             f'stroke-width:1.2"/>',
             f'<path d="M{left + 2},18 L{left + 12},13 L{left + 12},23 z" style="fill:var(--ea-ink-3)"/>',
             f'<text x="{width / 2}" y="12" style="font-family:var(--ea-font-ui);font-size:12px;fill:var(--ea-ink-3);'
             f'text-anchor:middle">{html.escape(w["stronger"])}</text>']
    said = len(scale) - 1
    for i, word in enumerate(scale):
        x = left + i * (box_w + gap)
        bold = i == said
        stroke = "stroke:var(--ea-ink);stroke-width:2.4" if bold else "stroke:var(--ea-line);stroke-width:1.2"
        parts.append(f'<rect x="{x}" y="{top}" width="{box_w}" height="{box_h}" rx="8" '
                     f'style="fill:var(--ea-inset);{stroke}"/>')
        parts.append(f'<text x="{x + box_w / 2}" y="{top + 28}" style="font-family:var(--ea-font-ui);font-size:15px;'
                     f'font-weight:600;fill:var(--ea-ink);text-anchor:middle">{html.escape(word)}</text>')
        if i < said:
            half = 4.6 * len(word) + 6
            parts.append(f'<line x1="{x + box_w / 2 - half:.1f}" y1="{top + 23}" x2="{x + box_w / 2 + half:.1f}" '
                         f'y2="{top + 23}" style="stroke:var(--ea-ink-2);stroke-width:1.6"/>')
        label = w["said"] if bold else w["false"] if i < said else ""
        if label:
            parts.append(f'<text x="{x + box_w / 2}" y="{top + box_h + 18}" style="font-family:var(--ea-font-ui);'
                         f'font-size:12px;fill:var(--ea-ink-2);text-anchor:middle">{html.escape(label)}</text>')
    alt = f'{w["title"]}: {", ".join(scale)}. {_bare(w["legend"])}'
    svg = (f'<svg viewBox="0 0 {width} {height}" role="img" style="width:100%;max-width:{width}px;height:auto">'
           f'<title>{html.escape(alt)}</title>{"".join(parts)}</svg>')
    return yours.figure(w["title"], svg + _note(w["legend"]), alt=alt)


_PRESUP_WORDS = {
    "en": {"title": "What a sentence says, and what it takes for granted",
           "head": ["Sentence", "Says about now", "Takes for granted"],
           "rows": [("Tom has stopped chasing Pip.", "Tom does not chase Pip.", "Tom chased Pip before."),
                    ("Tom has not stopped chasing Pip.", "Tom still chases Pip.", "Tom chased Pip before.")],
           "legend": "The word *not* turns around what the sentence says, and leaves what it takes for granted as "
                     "it was: the negation test."},
    "nl": {"title": "Wat een zin zegt, en wat die zin als vanzelfsprekend aanneemt",
           "head": ["Zin", "Zegt over nu", "Neemt aan"],
           "rows": [("Tom is gestopt met Pip achtervolgen.", "Tom achtervolgt Pip niet.", "Tom achtervolgde Pip eerst."),
                    ("Tom is niet gestopt met Pip achtervolgen.", "Tom achtervolgt Pip nog.",
                     "Tom achtervolgde Pip eerst.")],
           "legend": "Het woord *niet* draait om wat de zin zegt, en laat wat de zin aanneemt ongemoeid: de "
                     "negatietest."},
}


def presup_picture(lang: str) -> str:
    w = _PRESUP_WORDS[lang]
    rows = [([(a, False), (b, False), (c, False)], False) for a, b, c in w["rows"]]
    alt = f'{w["title"]}. ' + " ".join(" ".join(r) for r in w["rows"]) + f' {_bare(w["legend"])}'
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


def _short(value, limit: int = 44) -> str:
    text = _code(value)
    return text if len(text) <= limit else text[:limit - 1] + "…"


_IMPLICATED_WORDS = {
    "en": {"title": "Your implicated, at work: what a listener takes to be false",
           "head": ["Said", "Yours", "Right"],
           "said": "{w} cats sleep",
           "fine": "Your implicated is right for each word on the scale. The weaker the word, the more stronger words "
                   "a listener takes to be false; for *all*, there is nothing stronger.",
           "differs": "Where a row is in purple, your implicated gives other words.",
           "error": "Your function stops with an error at: {what}.",
           "slow": "Your function takes too long."},
    "nl": {"title": "Jouw implicated aan het werk: wat een luisteraar opvat als onwaar",
           "head": ["Gezegd", "Jouw antwoord", "Goed"],
           "said": "{w} katten slapen",
           "fine": "Je implicated klopt bij elk woord op de schaal. Hoe zwakker het woord, hoe meer sterkere woorden "
                   "een luisteraar opvat als onwaar; bij *alle* is er niets sterkers.",
           "differs": "Waar een rij paars is, geeft je implicated andere woorden.",
           "error": "Je functie stopt met een fout bij: {what}.",
           "slow": "Je functie doet er te lang over."},
}


def implicated_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    lang, w = seen.lang, _IMPLICATED_WORDS[seen.lang]
    scale = SCALES[lang]["SCALE"]
    rows, end, all_right = [], "", True
    for word in reversed(scale):
        said = w["said"].format(w=word)
        said = said[0].upper() + said[1:]
        try:
            got = _call(fn, scale, word)
        except Timeout:
            end = w["slow"]
            break
        if got is _STOPPED:
            end = w["error"].format(what=said)
            break
        want = implicated(scale, word)
        right = type(got) is list and got == want
        all_right = all_right and right
        rows.append(([(said, False), (_short(got), True), (_code(want), True)], not right))
    after = end or (w["fine"] if all_right else w["differs"])
    alt = f'{w["title"]}. ' + "; ".join(" | ".join(c[0] for c in r[0]) for r in rows) + f". {_bare(after)}"
    return yours.figure(w["title"], _table(w["head"], rows) + _note(after), alt=alt)


_ONE_PHRASES = {
    "en": [("the cat that sleeps", "cat", "sleeps"), ("the dog", "dog", None), ("the cat", "cat", None),
           ("the mouse that sleeps", "mouse", "sleeps"), ("the unicorn", "unicorn", None)],
    "nl": [("de kat die slaapt", "cat", "sleeps"), ("de hond", "dog", None), ("de kat", "cat", None),
           ("de muis die slaapt", "mouse", "sleeps"), ("de eenhoorn", "unicorn", None)],
}
_ONE_WORDS = {
    "en": {"title": "Your the_one, at work: what each phrase refers to",
           "head": ["Phrase", "Yours", "Right", "… chases Tom"],
           "true": "true", "false": "false", "none": "no truth value",
           "fine": "Your the_one finds the one thing where there is one. Where there is not, the sentence with the "
                   "phrase can lack a truth value, as Strawson argued; Russell would call it false.",
           "differs": "Where a row is in purple, your the_one gives something else. The last column follows your "
                      "answer.",
           "error": "Your function stops with an error at: {what}.",
           "slow": "Your function takes too long."},
    "nl": {"title": "Jouw the_one aan het werk: waar elke woordgroep naar verwijst",
           "head": ["Woordgroep", "Jouw antwoord", "Goed", "… achtervolgt Tom"],
           "true": "waar", "false": "onwaar", "none": "geen waarheidswaarde",
           "fine": "Je the_one vindt het ene ding waar er één is. Waar dat niet zo is, kan de zin met de woordgroep "
                   "geen waarheidswaarde hebben, zoals Strawson betoogde; volgens Russell is de zin dan onwaar.",
           "differs": "Waar een rij paars is, geeft je the_one iets anders. De laatste kolom volgt jouw antwoord.",
           "error": "Je functie stopt met een fout bij: {what}.",
           "slow": "Je functie doet er te lang over."},
}


def the_one_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    lang, w = seen.lang, _ONE_WORDS[seen.lang]
    world, word = WORLD[lang], WORD[lang]
    rows, end, all_right = [], "", True
    for phrase, noun, rest in _ONE_PHRASES[lang]:
        things = world.get(word[noun], [])
        group = world[word[rest]] if rest else list(THINGS)
        try:
            got = _call(fn, things, group)
        except Timeout:
            end = w["slow"]
            break
        if got is _STOPPED:
            end = w["error"].format(what=phrase)
            break
        want = the_one(things, group)
        right = type(got) is type(want) and got == want
        all_right = all_right and right
        if got is None:
            truth = w["none"]
        else:
            truth = w["true"] if (got, "Tom") in world[word["chases"]] else w["false"]
        rows.append(([(phrase, False), (_short(got), True), (_code(want), True), (truth, False)], not right))
    after = end or (w["fine"] if all_right else w["differs"])
    alt = f'{w["title"]}. ' + "; ".join(" | ".join(c[0] for c in r[0]) for r in rows) + f". {after}"
    return yours.figure(w["title"], _table(w["head"], rows) + _note(after), alt=alt)


_DEMO_TEXT = {
    "en": {"title": "Six cats: what each sentence says, and what a listener takes it to mean",
           "head": ["Sentence", "Says (logic)", "Meant (listener)"],
           "true": "true", "false": "false", "sleeps": "sleeps",
           "best": "A speaker who keeps to the maxim of Quantity says: {s}.",
           "none": "None of the three is true: a speaker would say that no cat sleeps.",
           "grid": "{s}, for 0 to 6 sleeping cats",
           "rows": ["says", "meant"],
           "how": "Each square is a number of sleeping cats: 0 on the left, 6 on the right. A filled square is "
                  "a number for which the sentence is true; the ringed square is the number you chose. *Most* is read "
                  "here as more than half."},
    "nl": {"title": "Zes katten: wat elke zin zegt, en wat een luisteraar eruit opmaakt",
           "head": ["Zin", "Zegt (logica)", "Bedoeld (luisteraar)"],
           "true": "waar", "false": "onwaar", "sleeps": "slaapt",
           "best": "Een spreker die zich aan de maxime van kwantiteit houdt, zegt: {s}.",
           "none": "Geen van de drie is waar: een spreker zou zeggen dat er geen kat slaapt.",
           "grid": "{s}, bij 0 tot en met 6 slapende katten",
           "rows": ["zegt", "bedoeld"],
           "how": "Elk vakje is een aantal slapende katten: links 0, rechts 6. Een gevuld vakje is een aantal "
                  "waarbij de zin waar is; het vakje met de ring is het aantal dat je koos. *De meeste* betekent hier "
                  "meer dan de helft."},
}


def _cats_svg(count: int, lang: str) -> str:
    w = _DEMO_TEXT[lang]
    card_w, card_h, gap, left, top = 92, 40, 12, 6, 6
    width = left * 2 + len(CATS) * card_w + (len(CATS) - 1) * gap
    height = top + card_h + 24
    parts = []
    for i, cat in enumerate(CATS):
        x = left + i * (card_w + gap)
        asleep = i < count
        stroke = "stroke:var(--ea-ink);stroke-width:2.4" if asleep else "stroke:var(--ea-line);stroke-width:1.2"
        parts.append(f'<rect x="{x}" y="{top}" width="{card_w}" height="{card_h}" rx="8" '
                     f'style="fill:var(--ea-inset);{stroke}"/>')
        parts.append(f'<text x="{x + card_w / 2}" y="{top + 25}" style="font-family:var(--ea-font-ui);font-size:14px;'
                     f'font-weight:600;fill:var(--ea-ink);text-anchor:middle">{cat}</text>')
        if asleep:
            parts.append(f'<text x="{x + card_w / 2}" y="{top + card_h + 16}" style="font-family:var(--ea-font-code);'
                         f'font-size:12px;fill:var(--ea-ink-2);text-anchor:middle">{html.escape(w["sleeps"])}</text>')
    alt = ", ".join(CATS[:count]) or "-"
    return (f'<svg viewBox="0 0 {width} {height}" role="img" style="width:100%;max-width:{width}px;height:auto">'
            f'<title>{html.escape(w["sleeps"] + ": " + alt)}</title>{"".join(parts)}</svg>')


def demo_picture(count: int, k: int, lang: str) -> str:
    """The six cats with `count` asleep, what the three sentences say and mean, and the counts for sentence k."""
    w, labels = _DEMO_TEXT[lang], DEMO_LABELS[lang]
    count = max(0, min(len(CATS), int(count)))
    rows = []
    for j, label in enumerate(labels):
        rows.append(([(label, False), (w["true"] if says(j, count) else w["false"], False),
                      (w["true"] if means(j, count) else w["false"], False)], False))
    best = best_word(count)
    line = w["none"] if best is None else w["best"].format(s=labels[best][0].lower() + labels[best][1:])
    squares = ""
    for r, fn in enumerate((says, means)):
        cells = "".join(
            f'<div style="width:22px;height:22px;border-radius:4px;border:1px solid var(--ea-line);'
            f'background:{"var(--ea-ink)" if fn(k, c) else "var(--ea-inset)"};'
            f'{"outline:3px solid var(--ea-ink-2);outline-offset:2px;" if c == count else ""}"></div>'
            for c in range(len(CATS) + 1))
        squares += (f'<div style="display:flex;align-items:center;gap:5px;margin:0.25rem 0 0.25rem 4px">'
                    f'<span style="width:4.8rem;font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2)">'
                    f'{html.escape(w["rows"][r])}</span>{cells}</div>')
    head = w["grid"].format(s=labels[k])
    says_at = [c for c in range(len(CATS) + 1) if says(k, c)]
    means_at = [c for c in range(len(CATS) + 1) if means(k, c)]
    alt = (f'{w["title"]}. {w["sleeps"]}: {count}. '
           + "; ".join(" | ".join(c[0] for c in r[0]) for r in rows)
           + f'. {line} {head}: {w["rows"][0]} {says_at}, {w["rows"][1]} {means_at}. {_bare(w["how"])}')
    body = (_cats_svg(count, lang) + _table(w["head"], rows) + _note(line)
            + f'<p style="{_CAPTION};margin-top:0.8rem">{html.escape(head)}</p>' + squares + _note(w["how"]))
    return yours.figure(w["title"], body, alt=alt)


def demo_options(lang: str) -> dict:
    return {label: i for i, label in enumerate(DEMO_LABELS[lang])}


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["implicated", "the_one", "acts", "cancel"]


class Named:
    """A value the check gives the learner's function, shown in a message by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


def _named(name: str) -> Named:
    """SCALE, HEAT[1] and the like, as in the lesson's cell, with the English words."""
    key, _, rest = name.partition("[")
    value = SCALES["en"][key]
    return Named(name, value[int(rest[:-1])] if rest else value)


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

# exercise 1: implicated
IMPLICATED_CASES = [(_named(s), _named(w)) for s, w in (
    ("SCALE", "SCALE[3]"), ("SCALE", "SCALE[0]"), ("SCALE", "SCALE[2]"), ("HEAT", "HEAT[1]"), ("OFTEN", "OFTEN[1]"),
    ("OFTEN", "OFTEN[2]"), ("SCALE", "HEAT[1]"), ("HEAT", "HEAT[0]"))]
_I_START = "def implicated(scale, word):\n    ...\n"
_I = ("def implicated(scale, word):\n"
      "    for i in range(len(scale)):\n"
      "        if scale[i] == word:\n"
      "            return scale[:i]\n"
      "    return []\n")
_I_WEAKER = _I.replace("return scale[:i]", "return scale[i + 1:]")
_I_WITH = _I.replace("return scale[:i]", "return scale[:i + 1]")
_I_OTHERS = ("def implicated(scale, word):\n"
             "    return [w for w in scale if w != word]\n")
_I_EMPTY = ("def implicated(scale, word):\n"
            "    for i in range(len(scale)):\n"
            "        if scale[i] == word:\n"
            "            return scale[:i]\n"
            "        else:\n"
            "            return []\n")
_NL_ONLY = {w for key in SCALES["nl"] for w in SCALES["nl"][key]} - {w for key in SCALES["en"] for w in SCALES["en"][key]}


def _dutch_words(got, text) -> bool:
    """The learner's answer holds Dutch words of the scales, written into the code: the check gives English ones."""
    return isinstance(got, list) and any(isinstance(w, str) and w in _NL_ONLY for w in got)


_I_ALL = ("def implicated(scale, word):\n"
          "    stronger = []\n"
          "    for w in scale:\n"
          "        if w == word:\n"
          "            return stronger\n"
          "        stronger.append(w)\n"
          "    return stronger\n")

# exercise 2: the_one
_ANIMALS = ["Tom", "Mia", "Rex", "Pip"]
ONE_CASES = [
    (["Tom", "Mia"], ["Mia", "Rex"]),
    (["Rex"], _ANIMALS),
    (["Tom", "Mia"], _ANIMALS),
    ([], _ANIMALS),
    (["Tom", "Mia"], ["Tom", "Mia", "Rex"]),
    (["Pip"], []),
    (["Tom", "Mia", "Pip"], ["Pip", "Rex"]),
]
_O_START = "def the_one(things, group):\n    ...\n"
_O = ("def the_one(things, group):\n"
      "    inside = [thing for thing in things if thing in group]\n"
      "    if len(inside) == 1:\n"
      "        return inside[0]\n"
      "    return None\n")
_O_FIRST = _O.replace("if len(inside) == 1:", "if len(inside) > 0:")
_O_LIST = _O.replace("return inside[0]", "return inside")
_O_FALSE = _O.replace("return None", "return False")


EXERCISES = {
    "implicated": FunctionBox(
        name="implicated",
        picture=implicated_picture,
        function="implicated",
        start=_I_START,
        solution=_I,
        visible=IMPLICATED_CASES[:1],
        hidden=IMPLICATED_CASES[1:],
        first=[_DOTS],
        show=show_words,
        mistakes=[
            Mistake("written", matches=_dutch_words,
                    en="`{call}` gives {got}: words written into your code. The check gives your function scales of its own, with English words also in the Dutch lesson. Build the answer from `scale`, so that it works for any scale.",
                    nl="`{call}` geeft {got}: woorden die in je code staan. De check geeft je functie eigen schalen, met Engelse woorden, ook in de Nederlandse les. Bouw het antwoord uit `scale`, zodat het voor elke schaal werkt."),
            Mistake("weaker", wrong=_I_WEAKER,
                    en="`{call}` gives {got}, and it should give {want}: your function gives the words after `word`, the weaker ones. Which end of the scale holds the stronger words?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft de woorden na `word`, de zwakkere. Aan welke kant van de schaal staan de sterkere woorden?"),
            Mistake("empty", wrong=_I_EMPTY,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back an empty list for every word, also for a word on the scale. Is the `return []` inside the loop? Then it runs as soon as the first word is not `word`. Put it after the loop.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft bij elk woord een lege lijst terug, ook bij een woord op de schaal. Staat de `return []` in de loop? Dan gebeurt die al zodra het eerste woord niet `word` is. Zet de regel na de loop."),
            Mistake("with-word", wrong=_I_WITH,
                    en="`{call}` gives {got}, and it should give {want}: your list also holds `word` itself. What the speaker said is not taken to be false; only the stronger words are.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je lijst bevat ook `word` zelf. Wat de spreker zei, wordt niet opgevat als onwaar; alleen de sterkere woorden."),
            Mistake("others", wrong=_I_OTHERS,
                    en="`{call}` gives {got}, and it should give {want}: your function gives every other word on the scale. Only the stronger words are taken to be false; a weaker word is still true when a stronger one is.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft alle andere woorden van de schaal. Alleen de sterkere woorden worden opgevat als onwaar; een zwakker woord is nog steeds waar als een sterker woord waar is."),
            Mistake("not-on-scale", wrong=_I_ALL,
                    en="Right for each word on the scale. But `{call}` gives {got}: for a word that is not on the scale, your function gives the whole scale. Such a word says nothing about this scale: give back an empty list.",
                    nl="Goed voor elk woord op de schaal. Maar `{call}` geeft {got}: bij een woord dat niet op de schaal staat, geeft je functie de hele schaal. Zo'n woord zegt niets over deze schaal: geef een lege lijst terug."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back the words on the scale that are stronger than `word`.",
                      "nl": "Vervang de `...`: geef de woorden van de schaal terug die sterker zijn dan `word`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which words on the scale are stronger than the one that was said?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Welke woorden van de schaal zijn sterker dan het woord dat gezegd werd?"},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function work for any scale and any word it gets?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke schaal en elk woord dat ze krijgt?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back: the loop ends without a `return`. Does your function find `word`? And what should it give back after the loop?",
                          "nl": "Met `{call}` geeft je functie niets terug: de loop eindigt zonder `return`. Vindt je functie `word`? En wat moet je functie na de loop teruggeven?"},
            "correct": {"en": "Your `implicated` is right in all {n} cases, also for a word that is not on the scale. Look at the picture: each word of the scale, and what a listener takes to be false.",
                        "nl": "Je `implicated` klopt in alle {n} gevallen, ook bij een woord dat niet op de schaal staat. Kijk naar de afbeelding: elk woord van de schaal, en wat een luisteraar opvat als onwaar."},
        },
    ),
    "the_one": FunctionBox(
        name="the_one",
        picture=the_one_picture,
        function="the_one",
        start=_O_START,
        solution=_O,
        visible=ONE_CASES[:1],
        hidden=ONE_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("first", wrong=_O_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: with more than one such thing, your function gives the first. *The cat* takes for granted that there is exactly one cat; with two, it refers to nothing.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: bij meer dan één zo'n ding geeft je functie het eerste. *De kat* neemt aan dat er precies één kat is; bij twee verwijst de woordgroep naar niets."),
            Mistake("list", wrong=_O_LIST,
                    en="`{call}` gives {got}, and it should give {want}: your function gives a list with one thing in it. Give back the thing itself, not a list.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft een lijst met één ding erin. Geef het ding zelf terug, geen lijst."),
            Mistake("false", wrong=_O_FALSE,
                    en="`{call}` gives {got}, and it should give {want}. `False` is a truth value, and `the_one` gives a thing: when there is not exactly one, give back `None`, no thing at all.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. `False` is een waarheidswaarde, en `the_one` geeft een ding: als er niet precies één is, geef dan `None` terug, helemaal geen ding."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back the one thing of `things` that is also in `group`, or `None` when there is not exactly one.",
                      "nl": "Vervang de `...`: geef het ene ding uit `things` terug dat ook in `group` zit, of `None` als er niet precies één is."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. How many things of the first list are in the second?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Hoeveel dingen uit de eerste lijst zitten in de tweede?"},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. What should your function give when there are two such things, or none?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Wat moet je functie geven bij twee zulke dingen, of bij geen enkel?"},
            "unreached": {"en": "With `{call}`, your function gives `None`, and there is exactly one thing of the first list in the second. Does your function find it?",
                          "nl": "Met `{call}` geeft je functie `None`, en er zit precies één ding uit de eerste lijst in de tweede. Vindt je functie dat ding?"},
            "correct": {"en": "Your `the_one` is right in all {n} cases. Look at the picture: the phrases of the world, and a sentence with each.",
                        "nl": "Je `the_one` klopt in alle {n} gevallen. Kijk naar de afbeelding: de woordgroepen van de wereld, en een zin met elk ervan."},
        },
    ),
    "acts": Sort(
        name="acts",
        groups={
            "info": {"en": "Asks for information", "nl": "Vraagt om informatie"},
            "do": {"en": "Asks the listener to do something besides answering", "nl": "Vraagt de luisteraar iets anders te doen dan antwoorden"},
            "says": {"en": "Does what it says, by saying it", "nl": "Doet wat het zegt, door het te zeggen"},
        },
        items={
            "bus": {"en": "**Do you know when the last bus leaves?**", "nl": "**Weet jij hoe laat de laatste bus gaat?**"},
            "car": {"en": "**Where did you park the car?**", "nl": "**Waar heb je de auto geparkeerd?**"},
            "window": {"en": "**Could you close the window?**", "nl": "**Zou je het raam dicht kunnen doen?**"},
            "music": {"en": "**Would you mind turning the music down?**", "nl": "**Zou je de muziek wat zachter willen zetten?**"},
            "promise": {"en": "**I promise to bring the book back on Monday.**",
                        "nl": "**Ik beloof dat ik het boek maandag terugbreng.**"},
            "bet": {"en": "**I bet you ten euros that it rains tomorrow.**", "nl": "**Ik wed om tien euro dat het morgen regent.**"},
        },
        solution={"bus": "info", "car": "info", "window": "do", "music": "do", "promise": "says", "bet": "says"},
        nudge={
            "bus": {"en": "Would *yes* alone be a good answer, or does the speaker want the time?",
                    "nl": "Is alleen *ja* een goed antwoord, of wil de spreker de tijd weten?"},
            "car": {"en": "What does the speaker want to know?",
                    "nl": "Wat wil de spreker weten?"},
            "window": {"en": "Would *yes, I could* alone be a good answer, or does the speaker want the window closed?",
                       "nl": "Is alleen *ja, dat zou ik kunnen* een goed antwoord, of wil de spreker dat het raam dichtgaat?"},
            "music": {"en": "Does the speaker want to know whether you mind, or want quieter music?",
                      "nl": "Wil de spreker weten of je dat wilt, of wil de spreker zachtere muziek?"},
            "promise": {"en": "Once the sentence is said, has the speaker already done something, or only told something?",
                        "nl": "Heeft de spreker, zodra de zin gezegd is, al iets gedaan, of alleen iets verteld?"},
            "bet": {"en": "Once the sentence is said, is there a bet, or only news about one?",
                    "nl": "Is er, zodra de zin gezegd is, een weddenschap, of alleen nieuws over een weddenschap?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the sentences again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de zinnen opnieuw uit."},
            "empty": {"en": "Put each bold sentence under what the speaker does with it.",
                      "nl": "Zet elke vetgedrukte zin bij wat de spreker ermee doet."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} belongs under another group.",
                      "nl": "{item} hoort bij een andere groep."},
            "correct": {"en": "All {n} are right. Three of them are questions to answer with yes or no, and none of the three wants only a yes: one wants information, two want an act. And two sentences do what they say: once they are said, the promise and the bet are made.",
                        "nl": "Alle {n} kloppen. Drie ervan zijn ja-neevragen, en geen van de drie wil alleen een ja: één wil informatie, twee willen een handeling. En twee zinnen doen wat ze zeggen: zodra ze gezegd zijn, is de belofte gedaan en de weddenschap aangegaan."},
        },
    ),
}

CASES = [
    ("implicated", _I_START, "empty", None),
    ("implicated", _I, "correct", None),
    ("implicated", "def implicated(scale, word):\n    stronger = []\n    for w in scale:\n        if w == word:\n            return stronger\n        stronger.append(w)\n    return []\n", "correct", None),
    ("implicated", "def implicated(scale, word):\n    if word in scale:\n        return scale[:scale.index(word)]\n    return []\n", "correct", None),
    ("implicated", _I_WEAKER, "wrong", "weaker"),
    ("implicated", _I_WITH, "wrong", "with-word"),
    ("implicated", _I_OTHERS, "wrong", "others"),
    ("implicated", _I_ALL, "wrong", "not-on-scale"),
    ("implicated", _I_EMPTY, "wrong", "empty"),
    ("implicated", "def implicated(scale, word):\n    return []\n", "wrong", "empty"),
    ("implicated", "def implicated(scale, word):\n    return [\"alle\", \"de meeste\", \"veel\"]\n", "wrong", "written"),
    ("implicated", "def implicated(scale, word):\n    for i in range(len(scale)):\n        if scale[i] == \"word\":\n            return scale[:i]\n", "wrong", "unreached"),
    ("implicated", _I.replace("    return []\n", ""), "wrong", "unreached"),
    ("implicated", "def implicated(scale, word):\n    return [\"all\", \"most\", \"many\"]\n", "almost", None),
    ("the_one", _O_START, "empty", None),
    ("the_one", _O, "correct", None),
    ("the_one", _O.replace("    return None\n", ""), "correct", None),
    ("the_one", "def the_one(things, group):\n    for thing in things:\n        if thing in group and len([t for t in things if t in group]) == 1:\n            return thing\n    return None\n", "correct", None),
    ("the_one", _O_FIRST, "wrong", "first"),
    ("the_one", _O_LIST, "wrong", "list"),
    ("the_one", _O_FALSE, "wrong", "false"),
    ("the_one", _O.replace("[thing for thing in things if thing in group]", "things"), "wrong", "unreached"),
    ("the_one", "def the_one(things, group):\n    return \"Mia\"\n", "almost", None),
    ("acts", {}, "empty", None),
    ("acts", {"bus": "info", "car": "info", "window": "do", "music": "do", "promise": "says", "bet": "says"},
     "correct", None),
    ("acts", {"bus": "info", "window": "do"}, "wrong", "unfinished"),
    ("acts", {"bus": "do"}, "wrong", "wrong:bus"),
    ("acts", {"bus": "info", "car": "info", "window": "info", "music": "do", "promise": "says", "bet": "says"},
     "wrong", "wrong:window"),
]
