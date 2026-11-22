# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-edit-distance: count the insertions, deletions and substitutions in an alignment of two
words (count_edits), make the start of the table, with the distances from and to the empty start (start_table), and
fill the rest of the table to find the edit distance (edit_distance); and sort pairs of words by how many edits apart
they are.

Every edit costs 1, as in the Levenshtein distance, unless a picture says otherwise; the demo can also give a
substitution the cost 2, as Jurafsky and Martin do in the table of their chapter *Words and Tokens*. The words, the
sentences and every number in the pictures are the course's own; the pairs *intention* and *execution* (Jurafsky and
Martin, after Kruskal 1983) and *cats* and *fast* (Manning, Raghavan and Schütze) come from the books the lesson
quotes, and their distances are computed by the course."""
import contextlib
import copy
import functools
import html
import io

import numpy as np

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, with_time_limit

# --- the course's solutions ------------------------------------------------------------------------------------------

GAP = "-"   # the mark of a gap in an alignment: a letter or a word without a partner


def count_edits(top, bottom):
    """The course's solution of Exercise 1: how many insertions, deletions and substitutions an alignment holds. A gap
    in `top` is an insertion (the letter of `bottom` comes in), a gap in `bottom` a deletion (the letter of `top`
    goes), and two different letters a substitution."""
    counts = {"insert": 0, "delete": 0, "substitute": 0}
    for i in range(len(top)):
        if top[i] == "-":
            counts["insert"] = counts["insert"] + 1
        elif bottom[i] == "-":
            counts["delete"] = counts["delete"] + 1
        elif top[i] != bottom[i]:
            counts["substitute"] = counts["substitute"] + 1
    return counts


def start_table(a, b):
    """The course's solution of Exercise 2: a table of whole numbers with a row for each first part of `a`, from no
    letters to all of them, and a column for each first part of `b`; the first column holds 0, 1, 2, ... (deleting
    every letter) and the first row too (inserting every letter); the rest is 0."""
    D = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
    for i in range(len(a) + 1):
        D[i, 0] = i
    for j in range(len(b) + 1):
        D[0, j] = j
    return D


def edit_distance(a, b):
    """The course's solution of Exercise 3: the edit distance between `a` and `b`, every edit costing 1. It works for
    strings, and for lists of words too."""
    D = start_table(a, b)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 1
            if a[i - 1] == b[j - 1]:
                cost = 0
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + cost)
    return D[len(a), len(b)]


# --- the table, the way back and the number of ways ------------------------------------------------------------------

def full_table(a, b, sub: int = 1):
    """The whole table of edit distances between the first parts of `a` and `b`; a substitution costs `sub`."""
    D = start_table(a, b)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else sub
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + cost)
    return D


def distance(a, b, sub: int = 1) -> int:
    return int(full_table(a, b, sub)[len(a), len(b)])


def backtrace(a, b, sub: int = 1) -> tuple:
    """One cheapest alignment of `a` and `b`, found from the last cell back to the first: (steps, cells), with steps
    as (top, bottom, kind) in reading order, kind "same", "sub", "del" or "ins", and cells the (i, j) of the path.
    Where two ways back give the same value, the step diagonally goes first, then up, then to the left."""
    D = full_table(a, b, sub)
    i, j, steps, cells = len(a), len(b), [], [(len(a), len(b))]
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (0 if a[i - 1] == b[j - 1] else sub):
            steps.append((a[i - 1], b[j - 1], "same" if a[i - 1] == b[j - 1] else "sub"))
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            steps.append((a[i - 1], GAP, "del"))
            i -= 1
        else:
            steps.append((GAP, b[j - 1], "ins"))
            j -= 1
        cells.append((i, j))
    return steps[::-1], set(cells)


def cheapest_ways(a, b, sub: int = 1) -> int:
    """How many different paths through the table give the edit distance: the cheapest alignments."""
    D = full_table(a, b, sub)
    N = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    N[0][0] = 1
    for i in range(len(a) + 1):
        for j in range(len(b) + 1):
            if i == 0 and j == 0:
                continue
            n = 0
            if i > 0 and D[i, j] == D[i - 1, j] + 1:
                n += N[i - 1][j]
            if j > 0 and D[i, j] == D[i, j - 1] + 1:
                n += N[i][j - 1]
            if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (0 if a[i - 1] == b[j - 1] else sub):
                n += N[i - 1][j - 1]
            N[i][j] = n
    return N[len(a)][len(b)]


@functools.cache
def all_ways(m: int, n: int) -> int:
    """How many ways there are to line up a word of m letters with a word of n letters, when each step pairs two
    letters or gives one letter a gap: every path through the table from the first cell to the last (the Delannoy
    number)."""
    if m == 0 or n == 0:
        return 1
    return all_ways(m - 1, n) + all_ways(m, n - 1) + all_ways(m - 1, n - 1)


def kinds(top, bottom) -> list:
    """The kind of each place of an alignment: "same", "sub", "del" or "ins"."""
    return ["ins" if t == GAP else "del" if u == GAP else "same" if t == u else "sub" for t, u in zip(top, bottom)]


# the alignments of the lesson: the cheapest of phone and power, the one of Jurafsky and Martin for intention and
# execution, and the one that deletes every letter of phone and types power
CHEAPEST = ("phone-", "p-ower")
SLP = ("inte-ntion", "-execution")
ALL_OVER = ("phone-----", "-----power")
# what someone said and what a program heard, for the word error rate: two deletions, one substitution, one insertion
# (not as many insertions as deletions, so a count_edits that swaps the two shows it in the picture)
SAID = {"en": ["just", "set", "a", "timer", "for", "ten", "minutes"],
        "nl": ["zet", "even", "een", "timer", "voor", "tien", "minuten"]}
HEARD = {"en": ["set", "timer", "for", "tea", "minutes", "please"],
         "nl": ["zet", "timer", "voor", "twee", "minuten", "alsjeblieft"]}
SAID_ALIGNED = {"en": ["just", "set", "a", "timer", "for", "ten", "minutes", GAP],
                "nl": ["zet", "even", "een", "timer", "voor", "tien", "minuten", GAP]}
HEARD_ALIGNED = {"en": [GAP, "set", GAP, "timer", "for", "tea", "minutes", "please"],
                 "nl": ["zet", GAP, GAP, "timer", "voor", "twee", "minuten", "alsjeblieft"]}


def wer(counts: dict, said: list) -> float:
    """The word error rate in percent: all edits, against the number of words that were said."""
    return 100 * (counts["insert"] + counts["delete"] + counts["substitute"]) / len(said)


# --- the look of the pictures ----------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);margin-top:0.45rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums"
_CHIP = ("display:inline-block;margin:0.12rem 0.2rem 0.12rem 0;padding:0 0.3rem;border-radius:3px;"
         "font-family:var(--ea-font-code);font-size:0.82rem;line-height:1.5;white-space:pre")
_CHIP_BLUE = _CHIP + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
_CHIP_PURPLE = _CHIP + ";background:var(--ea-loss-wash);color:var(--ea-loss-text)"
_CHIP_DATA = _CHIP + ";border:1px solid var(--ea-line);color:var(--ea-ink)"
# a cell of the table: a fixed square with a number in it
_SQ = ("width:1.75rem;height:1.55rem;text-align:center;vertical-align:middle;font-family:var(--ea-font-code);"
       "font-size:0.8rem;font-variant-numeric:tabular-nums;border:1px solid var(--ea-line);color:var(--ea-ink)")
_SQ_HEAD = _SQ + ";border-color:transparent;color:var(--ea-ink-2);font-weight:600"
_SQ_PATH = _SQ + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
_SQ_END = _SQ_PATH + ";font-weight:700;border-color:var(--ea-pred)"
_SQ_WRONG = _SQ + ";background:var(--ea-loss-wash);color:var(--ea-loss-text)"
_SQ_REST = _SQ + ";color:var(--ea-ink-3)"
# a place of an alignment: one letter or word above another
_PLACE = ("padding:0.1rem 0.3rem;text-align:center;font-family:var(--ea-font-code);font-size:0.82rem;"
          "border:1px solid var(--ea-line);color:var(--ea-ink);white-space:pre")
_PLACE_EDIT = _PLACE + ";background:var(--ea-loss-wash);color:var(--ea-loss-text)"
_PLACE_KIND = "padding:0 0.3rem;text-align:center;font-family:var(--ea-font-ui);font-size:0.75rem;color:var(--ea-loss-text)"

KIND_MARK = {"en": {"del": "d", "ins": "i", "sub": "s", "same": ""},
             "nl": {"del": "w", "ins": "i", "sub": "v", "same": ""}}


def _note(text: str) -> str:
    """A line under a picture; a word between asterisks is set in italics, as in the lesson's text."""
    parts = html.escape(text).split("*")
    marked = "".join(f"<i>{p}</i>" if k % 2 else p for k, p in enumerate(parts))
    return f'<div style="{_UI}">{marked}</div>'


def _label(text: str) -> str:
    parts = html.escape(text).split("*")
    marked = "".join(f"<i>{p}</i>" if k % 2 else p for k, p in enumerate(parts))
    return f'<div style="{_LABEL}">{marked}</div>'


def _num(n, lang: str) -> str:
    """A whole number with a separator for thousands: 1,683 in English, 1.683 in Dutch."""
    text = f"{round(n):,}"
    return text.replace(",", ".") if lang == "nl" else text


def _pct(x: float, lang: str) -> str:
    """A percentage with one decimal: 50.0% in English, 50,0% in Dutch."""
    text = f"{x:.1f}%"
    return text.replace(".", ",") if lang == "nl" else text


def _chip(text: str, style: str = _CHIP_DATA) -> str:
    return f'<span style="{style}">{html.escape(text)}</span>'


def _table(rows_: list, head: list, numbers: tuple = ()) -> str:
    """A small table; rows hold HTML that is already escaped; numbers are the columns set to the right."""
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{_NUMBER if j in numbers else ""}">{c}</td>'
                                    for j, c in enumerate(row)) + "</tr>" for row in rows_)
    return (f'<table style="border-collapse:collapse;margin:0.3rem 0 0.2rem"><thead><tr>{top}</tr></thead>'
            f"<tbody>{body}</tbody></table>")


def _shown_letter(x) -> str:
    """A letter as a table shows it: a space as ␣."""
    return "␣" if x == " " else str(x)


def grid(a, b, values, path=frozenset(), wrong=frozenset(), rest=frozenset(), end: bool = True) -> str:
    """The table as squares: a header row with # and the letters of b, a header column with # and the letters of a.
    `values` gives the number of each cell (or None for an empty square); path cells are blue, the last one framed
    when `end`; wrong cells purple; rest cells grey."""
    head = f'<td style="{_SQ_HEAD}"></td><td style="{_SQ_HEAD}">#</td>' + "".join(
        f'<td style="{_SQ_HEAD}">{html.escape(_shown_letter(x))}</td>' for x in b)
    rows_ = [f"<tr>{head}</tr>"]
    for i in range(len(a) + 1):
        letter = "#" if i == 0 else _shown_letter(a[i - 1])
        cells = f'<td style="{_SQ_HEAD}">{html.escape(letter)}</td>'
        for j in range(len(b) + 1):
            v = values(i, j)
            text = "" if v is None else html.escape(str(v))
            if (i, j) in wrong:
                style = _SQ_WRONG
            elif (i, j) in path:
                style = _SQ_END if end and (i, j) == (len(a), len(b)) else _SQ_PATH
            elif (i, j) in rest:
                style = _SQ_REST
            else:
                style = _SQ
            cells += f'<td style="{style}">{text}</td>'
        rows_.append(f"<tr>{cells}</tr>")
    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;margin:0.3rem 0 0.2rem">'
            f'{"".join(rows_)}</table></div>')


def alignment_html(top, bottom, lang: str) -> str:
    """An alignment as three rows: the top word, the bottom word, and a mark under each edit."""
    places = kinds(top, bottom)
    rows_ = ""
    for line in (top, bottom):
        rows_ += "<tr>" + "".join(f'<td style="{_PLACE if k == "same" else _PLACE_EDIT}">{html.escape(_shown_letter(x))}</td>'
                                  for x, k in zip(line, places)) + "</tr>"
    rows_ += "<tr>" + "".join(f'<td style="{_PLACE_KIND}">{KIND_MARK[lang][k]}</td>' for k in places) + "</tr>"
    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:separate;border-spacing:2px;'
            f'margin:0.2rem 0">{rows_}</table></div>')


def _alignment_alt(top, bottom, lang: str) -> str:
    return " | ".join(f"{t}/{u}" for t, u in zip(top, bottom))


_MARKS_NOTE = {"en": "Purple: an edit, marked d (delete), s (substitute) or i (insert); a - is a gap.",
               "nl": "Paars: een bewerking, met w (weghalen), v (vervangen) of i (invoegen) eronder; een - is een gat."}


# --- the idea's pictures -----------------------------------------------------------------------------------------------

_IDEA_WORDS = {
    "en": {"title": "Two ways to line up phone and power",
           "cheap": "The cheapest way: {n} edits", "over": "Delete every letter, then type the other word: {n} edits",
           "note": "The edit distance is the number of edits of the cheapest way: *phone* and *power* are {n} edits "
                   "apart.",
           "alt": "{title}. {rows}. {marks} {note}"},
    "nl": {"title": "Twee manieren om phone en power uit te lijnen",
           "cheap": "De goedkoopste manier: {n} bewerkingen",
           "over": "Elke letter weghalen en dan het andere woord typen: {n} bewerkingen",
           "note": "De edit distance is het aantal bewerkingen van de goedkoopste manier: *phone* en *power* liggen "
                   "{n} bewerkingen uit elkaar.",
           "alt": "{title}. {rows}. {marks} {note}"},
}


def idea_picture(lang: str) -> str:
    w = _IDEA_WORDS[lang]
    body, alts = "", []
    for key, (top, bottom) in (("cheap", CHEAPEST), ("over", ALL_OVER)):
        n = sum(1 for k in kinds(top, bottom) if k != "same")
        label = w[key].format(n=n)
        body += _label(label) + alignment_html(top, bottom, lang)
        alts.append(label + ": " + _alignment_alt(top, bottom, lang))
    note = w["note"].format(n=distance("phone", "power"))
    body += _note(_MARKS_NOTE[lang]) + _note(note)
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows="; ".join(alts),
                                                             marks=_MARKS_NOTE[lang], note=note.replace("*", "")))


_TABLE_WORDS = {
    "en": {"title": "The table of phone and power",
           "note": "The cell in row i and column j holds the edit distance between the first i letters of *phone* and the first j "
                   "letters of *power*; # is the empty start, no letters yet. Blue: the path back from the last "
                   "cell, which holds the distance, {d}. Below: the alignment that the path gives.",
           "alt": "{title}. Rows: {rows}. {note} {line}"},
    "nl": {"title": "De tabel van phone en power",
           "note": "De cel in rij i en kolom j bevat de edit distance tussen de eerste i letters van *phone* en de eerste j "
                   "letters van *power*; # is het lege begin, nog geen letters. Blauw: de weg terug vanuit de laatste "
                   "cel, met de afstand, {d}. Daaronder: de alignment die de weg geeft.",
           "alt": "{title}. Rijen: {rows}. {note} {line}"},
}


def table_picture(lang: str, a: str = "phone", b: str = "power") -> str:
    w = _TABLE_WORDS[lang]
    D = full_table(a, b)
    steps, cells = backtrace(a, b)
    top, bottom = [s[0] for s in steps], [s[1] for s in steps]
    note = w["note"].format(d=int(D[len(a), len(b)]))
    body = grid(a, b, lambda i, j: int(D[i, j]), path=cells) + alignment_html(top, bottom, lang)
    body += _note(note) + _note(_MARKS_NOTE[lang])
    rows_ = "; ".join(("#" if i == 0 else a[i - 1]) + ": " + " ".join(str(int(v)) for v in D[i]) for i in range(len(a) + 1))
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows=rows_, note=note.replace("*", ""),
                                                             line=_alignment_alt(top, bottom, lang)))


# --- running the learner's function ----------------------------------------------------------------------------------

def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error
PICTURE_LIMIT = 10.0   # seconds the learner's function gets for everything a picture asks of it


def _call(make, seconds: float = PICTURE_LIMIT):
    """make() within a time limit, with what it prints thrown away; _STOPPED when it stops with an error, and the
    Timeout passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(make, seconds)
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _run_or_say(make, w: dict, ok) -> tuple:
    """(what make() gives, None), or (None, a figure that says why there is no picture): an error, too slow, or a
    result that `ok` does not accept."""
    try:
        got = _call(make)
    except Timeout:
        text = w["slow"].format(s=PICTURE_LIMIT)
        return None, yours.figure(w["title"], _note(text), alt=text)
    if got is _STOPPED:
        return None, yours.figure(w["title"], _note(w["stopped"]), alt=w["stopped"])
    if not ok(got):
        return None, yours.figure(w["title"], _note(w["odd"]), alt=w["odd"])
    return got, None


def _whole(v) -> bool:
    """A whole number, also a NumPy one or a float such as 3.0."""
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, np.integer)):
        return True
    return isinstance(v, (float, np.floating)) and float(v).is_integer()


_STOPPED_TEXT = {"en": "Your function stopped with an error on one of the cases, so the picture stays empty.",
                 "nl": "Je functie stopte met een fout op een van de gevallen, dus de afbeelding blijft leeg."}
_SLOW_TEXT = {"en": "Your function took more than {s:g} seconds for all cases, so the picture stays empty.",
              "nl": "Je functie deed er langer dan {s:g} seconden over voor alle gevallen, dus de afbeelding blijft leeg."}


# --- the picture under exercise 1: your counts on four alignments ---------------------------------------------------

_COUNT_WORDS = {
    "en": {"title": "Your count_edits on four alignments",
           "rows": ["*phone* and *power*, the cheapest way", "*intention* and *execution*, as Jurafsky and Martin "
                    "line them up", "*phone* and *power*, every letter deleted and typed again",
                    "What was said, above, and what a program heard, below"],
           "counts": "insert {i}, delete {d}, substitute {s}: {t} edits",
           "course": "the course counts {c}",
           "wer": "word error rate: 100 × {t} / {n} words said = {p}",
           "note": "Blue: what your function counts. Purple: a count that differs from the course's, with the course's "
                   "count beside it. The last alignment holds words, not letters: your function counts those too.",
           "stopped": _STOPPED_TEXT["en"], "slow": _SLOW_TEXT["en"],
           "odd": "Your function does not give back a dictionary with the counts \"insert\", \"delete\" and "
                  "\"substitute\" for every alignment, so the picture stays empty.",
           "alt": "{title}. {rows}. {note}"},
    "nl": {"title": "Jouw count_edits op vier alignments",
           "rows": ["*phone* en *power*, de goedkoopste manier", "*intention* en *execution*, uitgelijnd zoals bij "
                    "Jurafsky en Martin", "*phone* en *power*, elke letter weggehaald en opnieuw getypt",
                    "Wat iemand zei, boven, en wat een programma hoorde, onder"],
           "counts": "invoegen {i}, weghalen {d}, vervangen {s}: {t} bewerkingen",
           "course": "de course telt {c}",
           "wer": "word error rate: 100 × {t} / {n} gezegde woorden = {p}",
           "note": "Blauw: wat je functie telt. Paars: een aantal dat anders is dan dat van de course, met het aantal van "
                   "de course ernaast. De laatste alignment bevat woorden, geen letters: die telt je functie ook.",
           "stopped": _STOPPED_TEXT["nl"], "slow": _SLOW_TEXT["nl"],
           "odd": "Je functie geeft niet voor elke alignment een dictionary terug met de aantallen \"insert\", "
                  "\"delete\" en \"substitute\", dus de afbeelding blijft leeg.",
           "alt": "{title}. {rows}. {note}"},
}


def count_cases(lang: str) -> list:
    """The four alignments of the picture as (top, bottom): three of letters, as strings, and one of words."""
    return [CHEAPEST, SLP, ALL_OVER, (list(SAID_ALIGNED[lang]), list(HEARD_ALIGNED[lang]))]


def _counts_ok(got) -> bool:
    return isinstance(got, list) and all(
        isinstance(c, dict) and all(k in c and _whole(c[k]) for k in ("insert", "delete", "substitute")) for c in got)


def _counts_text(c: dict, w: dict) -> str:
    i, d, s = int(c["insert"]), int(c["delete"]), int(c["substitute"])
    return w["counts"].format(i=i, d=d, s=s, t=i + d + s)


def counts_picture(seen) -> str:
    lang = seen.lang
    w = _COUNT_WORDS[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    cases = count_cases(lang)
    got, said = _run_or_say(lambda: [fn(copy.copy(t), copy.copy(u)) for t, u in cases], w, _counts_ok)
    if said:
        return said
    body, alts = "", []
    for k, ((top, bottom), mine) in enumerate(zip(cases, got)):
        want = count_edits(top, bottom)
        right = all(int(mine[x]) == want[x] for x in want)
        line = _chip(_counts_text(mine, w), _CHIP_BLUE if right else _CHIP_PURPLE)
        alt = w["rows"][k].replace("*", "") + ": " + _counts_text(mine, w)
        if not right:
            line += " " + _chip(w["course"].format(c=_counts_text(want, w)))
            alt += ", " + w["course"].format(c=_counts_text(want, w))
        if k == 3:
            said_words = [x for x in top if x != GAP]
            t = int(mine["insert"]) + int(mine["delete"]) + int(mine["substitute"])
            rate = w["wer"].format(t=t, n=len(said_words), p=_pct(100 * t / len(said_words), lang))
            line += " " + _chip(rate, _CHIP_BLUE if right else _CHIP_PURPLE)
            alt += "; " + rate
        body += _label(w["rows"][k]) + alignment_html(top, bottom, lang) + f"<div>{line}</div>"
        alts.append(alt)
    body += _note(_MARKS_NOTE[lang]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows="; ".join(alts), note=w["note"]))


# --- the picture under exercise 2: your start of the table ------------------------------------------------------------

_START_WORDS = {
    "en": {"title": "Your start_table",
           "pair": "start_table({a}, {b})",
           "shape": "Your table has the shape {got}; for these words it needs {want}: a row for each first part of "
                    "the first word, from no letters to all of them, and a column for each first part of the second.",
           "note": "The first column says how many deletions turn the first letters of the first word into nothing; "
                   "the first row, how many insertions turn nothing into the first letters of the second word. Purple: "
                   "a number that differs from the course's start. Grey: cells that Exercise 3 fills in.",
           "stopped": _STOPPED_TEXT["en"], "slow": _SLOW_TEXT["en"],
           "odd": "Your function does not give back a table of numbers with rows and columns for every pair of "
                  "words, so the picture stays empty.",
           "alt": "{title}. {rows}. {note}"},
    "nl": {"title": "Jouw start_table",
           "pair": "start_table({a}, {b})",
           "shape": "Je tabel heeft de shape {got}; voor deze woorden is {want} nodig: een rij voor elk beginstuk van het "
                    "eerste woord, van geen letters tot allemaal, en een kolom voor elk beginstuk van het tweede.",
           "note": "De eerste kolom zegt hoeveel keer weghalen de eerste letters van het eerste woord in niets "
                   "verandert; de eerste rij hoeveel keer invoegen niets in de eerste letters van het tweede woord "
                   "verandert. Paars: een getal dat anders is dan in de start van de course. Grijs: cellen die "
                   "oefening 3 invult.",
           "stopped": _STOPPED_TEXT["nl"], "slow": _SLOW_TEXT["nl"],
           "odd": "Je functie geeft niet voor elk paar woorden een tabel met getallen in rijen en kolommen terug, dus "
                  "de afbeelding blijft leeg.",
           "alt": "{title}. {rows}. {note}"},
}
START_PAIRS = [("phone", "power"), ("cat", "cast")]


def _as_table(v):
    """A learner's table as a two-dimensional NumPy array of numbers, or None."""
    try:
        arr = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return None
    return arr if arr.ndim == 2 else None


def _int_text(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def start_picture(seen) -> str:
    lang = seen.lang
    w = _START_WORDS[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    got, said = _run_or_say(lambda: [fn(a, b) for a, b in START_PAIRS], w,
                            lambda g: all(_as_table(t) is not None for t in g))
    if said:
        return said
    body, alts = "", []
    for (a, b), mine in zip(START_PAIRS, got):
        arr, want = _as_table(mine), start_table(a, b)
        body += _label(w["pair"].format(a=f'"{a}"', b=f'"{b}"'))
        alt = w["pair"].format(a=a, b=b) + ": "
        if arr.shape != want.shape:
            text = w["shape"].format(got=tuple(arr.shape), want=tuple(want.shape))
            body += _note(text)
            alts.append(alt + text)
            continue
        rest = {(i, j) for i in range(1, len(a) + 1) for j in range(1, len(b) + 1)}
        wrong = {(i, j) for i in range(len(a) + 1) for j in range(len(b) + 1) if arr[i, j] != want[i, j]}
        body += grid(a, b, lambda i, j: _int_text(arr[i, j]), wrong=wrong, rest=rest - wrong, end=False)
        alts.append(alt + "; ".join(" ".join(_int_text(x) for x in row) for row in arr))
    body += _note(w["note"])
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows="; ".join(alts), note=w["note"]))


# --- the picture under exercise 3: your distances ----------------------------------------------------------------------

_DISTANCE_WORDS = {
    "en": {"title": "Your edit_distance",
           "grid": "Your edit_distance on every first part of *phone* and of *power*",
           "pairs": "Your edit_distance on other pairs",
           "head": ["pair", "your distance"],
           "course": "the course: {d}",
           "said": "what was said and what a program heard, as lists of words",
           "wer": "word error rate: 100 × {t} / {n} words said = {p}",
           "note": "Blue: the path back through the table, where your numbers are right. Purple: a number that "
                   "differs from the course's. *car* and *cat* are close in letters and far in meaning; *car* and "
                   "*automobile* the other way round.",
           "stopped": _STOPPED_TEXT["en"], "slow": _SLOW_TEXT["en"],
           "odd": "Your function does not give back a whole number for every pair, so the picture stays empty.",
           "alt": "{title}. Table: {rows}. {pairs}. {note}"},
    "nl": {"title": "Jouw edit_distance",
           "grid": "Je edit_distance op elk beginstuk van *phone* en van *power*",
           "pairs": "Je edit_distance op andere paren",
           "head": ["paar", "jouw afstand"],
           "course": "de course: {d}",
           "said": "wat iemand zei en wat een programma hoorde, als lijsten woorden",
           "wer": "word error rate: 100 × {t} / {n} gezegde woorden = {p}",
           "note": "Blauw: de weg terug door de tabel, waar je getallen kloppen. Paars: een getal dat anders is dan "
                   "dat van de course. *car* (auto) en *cat* (kat) liggen dicht bij elkaar in letters en ver uit "
                   "elkaar in betekenis; *car* en *automobile* andersom.",
           "stopped": _STOPPED_TEXT["nl"], "slow": _SLOW_TEXT["nl"],
           "odd": "Je functie geeft niet voor elk paar een geheel getal terug, dus de afbeelding blijft leeg.",
           "alt": "{title}. Tabel: {rows}. {pairs}. {note}"},
}
DISTANCE_PAIRS = [("intention", "execution"), ("cats", "fast"), ("resturant", "restaurant"), ("langauge", "language"),
                  ("car", "cat"), ("car", "automobile")]
GRID_PAIR = ("phone", "power")


def distance_picture(seen) -> str:
    lang = seen.lang
    w = _DISTANCE_WORDS[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    a, b = GRID_PAIR
    said, heard = SAID[lang], HEARD[lang]

    def run():
        table_ = [[fn(a[:i], b[:j]) for j in range(len(b) + 1)] for i in range(len(a) + 1)]
        pairs = [fn(x, y) for x, y in DISTANCE_PAIRS]
        return table_, pairs, fn(list(said), list(heard))

    got, say = _run_or_say(run, w, lambda g: all(_whole(v) for row in g[0] for v in row)
                           and all(_whole(v) for v in g[1]) and _whole(g[2]))
    if say:
        return say
    mine, pairs, words = got
    D = full_table(a, b)
    _, cells = backtrace(a, b)
    wrong = {(i, j) for i in range(len(a) + 1) for j in range(len(b) + 1) if int(mine[i][j]) != D[i, j]}
    body = _label(w["grid"]) + grid(a, b, lambda i, j: int(mine[i][j]), path=cells - wrong, wrong=wrong,
                                    end=(len(a), len(b)) not in wrong)
    rows_, alts = [], []
    for (x, y), d in zip(DISTANCE_PAIRS, pairs):
        want = distance(x, y)
        cell = _chip(str(int(d)), _CHIP_BLUE if int(d) == want else _CHIP_PURPLE)
        if int(d) != want:
            cell += " " + _chip(w["course"].format(d=want))
        rows_.append([_chip(x) + " " + _chip(y), cell])
        alts.append(f"{x}, {y}: {int(d)}")
    t, want_t = int(words), distance(said, heard)
    rate = w["wer"].format(t=t, n=len(said), p=_pct(100 * t / len(said), lang))
    cell = _chip(str(t), _CHIP_BLUE if t == want_t else _CHIP_PURPLE) + " " + _chip(
        rate, _CHIP_BLUE if t == want_t else _CHIP_PURPLE)
    rows_.append([html.escape(w["said"]), cell])
    alts.append(w["said"] + ": " + rate)
    body += _label(w["pairs"]) + _table(rows_, w["head"]) + _note(w["note"])
    grid_alt = "; ".join(" ".join(str(int(v)) for v in row) for row in mine)
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows=grid_alt, pairs="; ".join(alts),
                                                             note=w["note"].replace("*", "")))


# --- the demo: the table of your own words -----------------------------------------------------------------------------

_DEMO_WORDS = {
    "en": {"title": "The table of your words",
           "distance": "edit distance: {d}",
           "ways": "cheapest ways to line them up: {k} of {n}",
           "cost": {1: "every edit costs 1", 2: "a substitution costs 2, an insertion or a deletion 1"},
           "note": "Blue: the path back from the last cell; where several ways are cheapest, the picture takes the "
                   "step up and to the left first, then up, then to the left. Capitals, accents and spaces count as letters.",
           "empty": "Type a word in each box.",
           "long": "A word is longer than {k} characters; try a shorter one.",
           "alt": "{title}: {a} and {b}, {cost}. {distance}. {ways}. Rows: {rows}. Alignment: {line}. {note}"},
    "nl": {"title": "De tabel van jouw woorden",
           "distance": "edit distance: {d}",
           "ways": "goedkoopste manieren om de woorden uit te lijnen: {k} van de {n}",
           "cost": {1: "elke bewerking kost 1", 2: "vervangen kost 2, invoegen of weghalen 1"},
           "note": "Blauw: de weg terug vanuit de laatste cel; waar meer manieren het goedkoopst zijn, neemt de "
                   "afbeelding eerst de stap naar linksboven, dan omhoog, dan naar links. Hoofdletters, accenten en spaties "
                   "tellen als letters.",
           "empty": "Typ in elk vak een woord.",
           "long": "Een woord is langer dan {k} tekens; probeer een korter woord.",
           "alt": "{title}: {a} en {b}, {cost}. {distance}. {ways}. Rijen: {rows}. Alignment: {line}. {note}"},
}
DEMO_LONGEST = 20


def demo_picture(a, b, cost, lang: str) -> str:
    w = _DEMO_WORDS[lang]
    a, b = str(a or "").strip(), str(b or "").strip()
    if not a and not b:
        return yours.figure(w["title"], _note(w["empty"]), alt=w["empty"])
    if len(a) > DEMO_LONGEST or len(b) > DEMO_LONGEST:
        msg = w["long"].format(k=DEMO_LONGEST)
        return yours.figure(w["title"], _note(msg), alt=msg)
    sub = 2 if str(cost) == "2" else 1
    D = full_table(a, b, sub)
    steps, cells = backtrace(a, b, sub)
    top, bottom = [s[0] for s in steps], [s[1] for s in steps]
    d = int(D[len(a), len(b)])
    dist = w["distance"].format(d=d)
    ways = w["ways"].format(k=_num(cheapest_ways(a, b, sub), lang), n=_num(all_ways(len(a), len(b)), lang))
    body = (_label(w["cost"][sub]) + grid(a, b, lambda i, j: int(D[i, j]), path=cells)
            + (alignment_html(top, bottom, lang) if steps else "")
            + f"<div>{_chip(dist, _CHIP_BLUE)} {_chip(ways)}</div>" + _note(_MARKS_NOTE[lang]) + _note(w["note"]))
    rows_ = "; ".join(("#" if i == 0 else a[i - 1]) + ": " + " ".join(str(int(v)) for v in D[i]) for i in range(len(a) + 1))
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], a=a, b=b, cost=w["cost"][sub],
                                                             distance=dist, ways=ways, rows=rows_,
                                                             line=_alignment_alt(top, bottom, lang), note=w["note"]))


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["count-edits", "start-table", "edit-distance", "how-many-edits"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: "..." in "\n".join(
                    line.split("#")[0] for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang het door je eigen code.")


def _value(v) -> str:
    """A short value the way the lessons write it: strings in double quotes, also inside a list or a dictionary; an
    array as its list of rows."""
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_value(p) for p in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_value(k)}: {_value(x)}" for k, x in v.items()) + "}"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return repr(v)


class Named:
    """A value of the check's own that a message shows as Python, with strings in double quotes."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


def _lit(v) -> Named:
    return Named(_value(v), v)


def _a_number(got, text) -> bool:
    return _whole(got)


def _a_table(got, text) -> bool:
    return getattr(got, "ndim", 0) == 2 or isinstance(got, list) and bool(got) and isinstance(got[0], list)


def _index_error(e) -> bool:
    return isinstance(e, IndexError)


# exercise 1: count_edits. Besides the cheapest alignment of phone and power: the alignment of Jurafsky and Martin,
# one alignment with no edits, one with only an insertion, two letters swapped, a gap on each side, nothing at all,
# kitten and sitting with two substitutions and an insertion, and an alignment of words.
COUNT_CASES = [
    (_lit("phone-"), _lit("p-ower")),
    (_lit("inte-ntion"), _lit("-execution")),
    (_lit("cat"), _lit("cat")),
    (_lit("ca-t"), _lit("cast")),
    (_lit("form"), _lit("from")),
    (_lit("-a"), _lit("b-")),
    (_lit(""), _lit("")),
    (_lit("kitten-"), _lit("sitting")),
    (_lit(["set", "a", "timer"]), _lit(["set", "-", "timer"])),
]
_C_START = "def count_edits(top, bottom):\n    ...\n"
_C = ("def count_edits(top, bottom):\n"
      "    counts = {\"insert\": 0, \"delete\": 0, \"substitute\": 0}\n"
      "    for i in range(len(top)):\n"
      "        if top[i] == \"-\":\n"
      "            counts[\"insert\"] = counts[\"insert\"] + 1\n"
      "        elif bottom[i] == \"-\":\n"
      "            counts[\"delete\"] = counts[\"delete\"] + 1\n"
      "        elif top[i] != bottom[i]:\n"
      "            counts[\"substitute\"] = counts[\"substitute\"] + 1\n"
      "    return counts\n")
_C_SWAPPED = (_C.replace('if top[i] == "-"', 'if TOP_GAP').replace('elif bottom[i] == "-"', 'elif BOTTOM_GAP')
              .replace("TOP_GAP", 'bottom[i] == "-"').replace("BOTTOM_GAP", 'top[i] == "-"'))
_C_MATCHES = _C.replace("        elif top[i] != bottom[i]:\n", "        else:\n")
_C_GAP_FIRST = ("def count_edits(top, bottom):\n"
                "    counts = {\"insert\": 0, \"delete\": 0, \"substitute\": 0}\n"
                "    for i in range(len(top)):\n"
                "        if top[i] != bottom[i]:\n"
                "            counts[\"substitute\"] = counts[\"substitute\"] + 1\n"
                "        elif top[i] == \"-\":\n"
                "            counts[\"insert\"] = counts[\"insert\"] + 1\n"
                "        elif bottom[i] == \"-\":\n"
                "            counts[\"delete\"] = counts[\"delete\"] + 1\n"
                "    return counts\n")
_C_RETURN_IN = _C.replace("    return counts\n", "        return counts\n")
_C_TOTAL = ("def count_edits(top, bottom):\n"
            "    n = 0\n"
            "    for i in range(len(top)):\n"
            "        if top[i] != bottom[i]:\n"
            "            n = n + 1\n"
            "    return n\n")

# exercise 2: start_table. Besides phone and power: words of different lengths both ways round, an empty word on
# each side, two empty words, and lists of words.
START_CASES = [
    (_lit("phone"), _lit("power")),
    (_lit("cat"), _lit("cast")),
    (_lit("intention"), _lit("exe")),
    (_lit(""), _lit("abc")),
    (_lit("abc"), _lit("")),
    (_lit(""), _lit("")),
    (_lit(["set", "a", "timer"]), _lit(["set", "timer"])),
]
_S_START = "def start_table(a, b):\n    ...\n"
_S = ("def start_table(a, b):\n"
      "    D = np.zeros((len(a) + 1, len(b) + 1), dtype=int)\n"
      "    for i in range(len(a) + 1):\n"
      "        D[i, 0] = i\n"
      "    for j in range(len(b) + 1):\n"
      "        D[0, j] = j\n"
      "    return D\n")
_S_SMALL = ("def start_table(a, b):\n"
            "    D = np.zeros((len(a), len(b)), dtype=int)\n"
            "    for i in range(len(a)):\n"
            "        D[i, 0] = i\n"
            "    for j in range(len(b)):\n"
            "        D[0, j] = j\n"
            "    return D\n")
_S_SWAPPED = ("def start_table(a, b):\n"
              "    D = np.zeros((len(b) + 1, len(a) + 1), dtype=int)\n"
              "    for i in range(len(b) + 1):\n"
              "        D[i, 0] = i\n"
              "    for j in range(len(a) + 1):\n"
              "        D[0, j] = j\n"
              "    return D\n")
_S_COLUMN = _S.replace("    for j in range(len(b) + 1):\n        D[0, j] = j\n", "")
_S_ROW = _S.replace("    for i in range(len(a) + 1):\n        D[i, 0] = i\n", "")

# exercise 3: edit_distance. Besides phone and power: the pair of Jurafsky and Martin, the pair of Manning and
# colleagues, a missing letter, two letters swapped, the same word, an empty word on each side, two empty words, one
# letter that differs, and lists of words.
DISTANCE_CASES = [
    (_lit("phone"), _lit("power")),
    (_lit("intention"), _lit("execution")),
    (_lit("cats"), _lit("fast")),
    (_lit("resturant"), _lit("restaurant")),
    (_lit("langauge"), _lit("language")),
    (_lit("same"), _lit("same")),
    (_lit(""), _lit("abc")),
    (_lit("abc"), _lit("")),
    (_lit(""), _lit("")),
    (_lit("a"), _lit("b")),
    (_lit(["the", "cat", "sat"]), _lit(["the", "hat", "sat", "down"])),
]
_E_START = "def edit_distance(a, b):\n    ...\n"
_E = ("def edit_distance(a, b):\n"
      "    D = start_table(a, b)\n"
      "    for i in range(1, len(a) + 1):\n"
      "        for j in range(1, len(b) + 1):\n"
      "            cost = 1\n"
      "            if a[i - 1] == b[j - 1]:\n"
      "                cost = 0\n"
      "            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + cost)\n"
      "    return D[len(a), len(b)]\n")
_E_INDEX = _E.replace("if a[i - 1] == b[j - 1]:", "if a[i] == b[j]:")
_E_TWO = _E.replace("            cost = 1\n", "            cost = 2\n")
_E_MAX = _E.replace("min(", "max(")
_E_NO_ONE = _E.replace("min(D[i - 1, j] + 1, D[i, j - 1] + 1,", "min(D[i - 1, j], D[i, j - 1],")
_E_FROM_0 = _E.replace("range(1, len(a) + 1)", "range(len(a) + 1)").replace("range(1, len(b) + 1)", "range(len(b) + 1)")
_E_EARLY = _E.replace("range(1, len(a) + 1)", "range(1, len(a))").replace("range(1, len(b) + 1)", "range(1, len(b))")
_E_RETURN_IN = _E.replace("    return D[len(a), len(b)]\n", "            return D[len(a), len(b)]\n")
_E_COST_FLIPPED = (_E.replace("            cost = 1\n", "            cost = 0\n")
                   .replace("                cost = 0\n", "                cost = 1\n"))
_E_TABLE = _E.replace("    return D[len(a), len(b)]\n", "    return D\n")

EXERCISES = {
    "count-edits": FunctionBox(
        name="count-edits",
        picture=counts_picture,
        function="count_edits",
        start=_C_START,
        solution=_C,
        visible=COUNT_CASES[:1],
        hidden=COUNT_CASES[1:],
        first=[_DOTS],
        show=_value,
        mistakes=[
            Mistake("a-number", matches=_a_number,
                    en="`{call}` gives {got}: your function gives back one number. Here it should give back a dictionary with three counts, {want}: how many insertions, deletions and substitutions, each under its own key.",
                    nl="`{call}` geeft {got}: je functie geeft één getal terug. Hier moet de functie een dictionary met drie aantallen teruggeven, {want}: hoeveel keer invoegen, weghalen en vervangen, elk onder een eigen key."),
            Mistake("swapped", wrong=_C_SWAPPED,
                    en="`{call}` gives {got}, and it should give {want}: your function swaps insertions and deletions. A `\"-\"` in the top line means that the letter below has no partner above: it comes in, an insertion. A `\"-\"` in the bottom line is a deletion.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie verwisselt invoegen en weghalen. Een `\"-\"` in de bovenste regel betekent dat de letter eronder boven geen partner heeft: die letter komt erbij, invoegen. Een `\"-\"` in de onderste regel is weghalen."),
            Mistake("matches", wrong=_C_MATCHES,
                    en="`{call}` gives {got}, and it should give {want}: your function counts every place without a gap as a substitution, also where both lines have the same letter. Count a substitution only where the two letters differ.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt elke plek zonder gat als vervangen, ook waar beide regels dezelfde letter hebben. Tel vervangen alleen waar de twee letters verschillen."),
            Mistake("gap-first", wrong=_C_GAP_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: a gap and a letter also differ, so your first question counts every gap as a substitution, and the questions about `\"-\"` never get a turn. Ask about the gaps first, and about different letters last.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een gat en een letter verschillen ook, dus je eerste vraag telt elk gat als vervangen, en de vragen over `\"-\"` komen nooit aan de beurt. Vraag eerst naar de gaten, en als laatste naar verschillende letters."),
            Mistake("return-in-loop", wrong=_C_RETURN_IN,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back the counts after the first place it looks at. Put `return counts` after the loop, at the same depth as `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft de aantallen al terug na de eerste plek die je functie bekijkt. Zet `return counts` na de loop, even ver ingesprongen als `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: start with a count of 0 for each kind of edit, go over the places of the alignment, add one to the right count at each edit, and give back the counts.",
                      "nl": "Vervang de `...`: begin met een aantal van 0 voor elke soort bewerking, ga de plekken van de alignment langs, tel bij elke bewerking één op bij het goede aantal, en geef de aantallen terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function count an insertion where the top line has a `\"-\"`, a deletion where the bottom line has one, and a substitution where two different letters stand one above the other?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Telt je functie invoegen waar de bovenste regel een `\"-\"` heeft, weghalen waar de onderste regel er een heeft, en vervangen waar twee verschillende letters boven elkaar staan?"},
            "almost": {"en": "Right for the alignment of *phone* and *power*. But `{call}` gives {got}, and it should give {want}. Does your function work for every alignment: one without edits, an empty one, one of words?",
                       "nl": "Goed voor de alignment van *phone* en *power*. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke alignment: een zonder bewerkingen, een lege, een met woorden?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back. Does it end with `return counts`, after the loop?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie met `return counts`, na de loop?"},
            "correct": {"en": "Your `count_edits` is right in all {n} cases. Look at the picture: your function on four alignments, the last one of words, with the word error rate it gives.",
                        "nl": "Je `count_edits` klopt in alle {n} gevallen. Kijk naar de afbeelding: je functie op vier alignments, de laatste met woorden, met de word error rate die eruit volgt."},
        },
    ),
    "start-table": FunctionBox(
        name="start-table",
        picture=start_picture,
        function="start_table",
        start=_S_START,
        solution=_S,
        names={"np": np},
        visible=START_CASES[:1],
        hidden=START_CASES[1:],
        first=[_DOTS],
        show=_value,
        mistakes=[
            Mistake("too-small", error=_index_error,
                    en="With `{call}`, your function stops with an error: a loop reaches a row or a column that your table does not have. Does the table have a row for no letters at all, before the first letter, and a column for that too? And do the rows go with `a` and the columns with `b`, in the table and in the loops?",
                    nl="Met `{call}` stopt je functie met een fout: een loop komt bij een rij of kolom die je tabel niet heeft. Heeft de tabel een rij voor helemaal geen letters, vóór de eerste letter, en ook zo'n kolom? En horen de rijen bij `a` en de kolommen bij `b`, in de tabel en in de loops?"),
            Mistake("small", wrong=_S_SMALL,
                    en="`{call}` gives {got}, and it should give {want}: your table has a row for each letter of `a`, but the table also needs a row for no letters at all, the empty start, and a column for that too.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je tabel heeft een rij voor elke letter van `a`, maar de tabel heeft ook een rij nodig voor helemaal geen letters, het lege begin, en ook zo'n kolom."),
            Mistake("swapped", wrong=_S_SWAPPED,
                    en="`{call}` gives {got}, and it should give {want}: your table has a row for each first part of `b` and a column for each first part of `a`. The first word goes down the rows, the second along the columns.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je tabel heeft een rij voor elk beginstuk van `b` en een kolom voor elk beginstuk van `a`. Het eerste woord loopt langs de rijen, het tweede langs de kolommen."),
            Mistake("column-only", wrong=_S_COLUMN,
                    en="`{call}` gives {got}, and it should give {want}: your function fills the first column, but leaves the first row at 0. The first row says how many insertions turn nothing into the first letters of `b`: 0, 1, 2, and so on.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie vult de eerste kolom, maar laat de eerste rij op 0 staan. De eerste rij zegt hoeveel keer invoegen niets in de eerste letters van `b` verandert: 0, 1, 2, enzovoort."),
            Mistake("row-only", wrong=_S_ROW,
                    en="`{call}` gives {got}, and it should give {want}: your function fills the first row, but leaves the first column at 0. The first column says how many deletions turn the first letters of `a` into nothing: 0, 1, 2, and so on.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie vult de eerste rij, maar laat de eerste kolom op 0 staan. De eerste kolom zegt hoeveel keer weghalen de eerste letters van `a` in niets verandert: 0, 1, 2, enzovoort."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: make a table of zeros with `np.zeros`, a row for each first part of `a` and a column for each first part of `b`; fill the first column and the first row with 0, 1, 2, ...; and give back the table.",
                      "nl": "Vervang de `...`: maak een tabel met nullen met `np.zeros`, een rij voor elk beginstuk van `a` en een kolom voor elk beginstuk van `b`; vul de eerste kolom en de eerste rij met 0, 1, 2, ...; en geef de tabel terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your table have a row for each first part of `a`, from no letters to all of them, and a column for each first part of `b`, with 0, 1, 2, ... down the first column and along the first row?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Heeft je tabel een rij voor elk beginstuk van `a`, van geen letters tot allemaal, en een kolom voor elk beginstuk van `b`, met 0, 1, 2, ... in de eerste kolom en in de eerste rij?"},
            "almost": {"en": "Right for *phone* and *power*. But `{call}` gives {got}, and it should give {want}. Does your function work for every pair: words of different lengths, an empty word?",
                       "nl": "Goed voor *phone* en *power*. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elk paar: woorden van verschillende lengte, een leeg woord?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back. Does it end with `return D`, after both loops?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie met `return D`, na beide loops?"},
            "correct": {"en": "Your `start_table` is right in all {n} cases. Look at the picture: the start of the table for two pairs of words, with the cells that Exercise 3 fills in.",
                        "nl": "Je `start_table` klopt in alle {n} gevallen. Kijk naar de afbeelding: het begin van de tabel voor twee paren woorden, met de cellen die oefening 3 invult."},
        },
    ),
    "edit-distance": FunctionBox(
        name="edit-distance",
        picture=distance_picture,
        function="edit_distance",
        start=_E_START,
        solution=_E,
        names={"start_table": start_table, "np": np},
        visible=DISTANCE_CASES[:1],
        hidden=DISTANCE_CASES[1:],
        first=[_DOTS],
        show=_value,
        mistakes=[
            Mistake("index", error=_index_error,
                    en="With `{call}`, your function stops with an error: in the last column, `j` is the length of `b`, and in the last row, `i` is the length of `a`, and a string has no letter at that place. Row `i` of the table is about the first `i` letters of `a`, so its new letter is the one before place `i`; the same goes for column `j` and `b`.",
                    nl="Met `{call}` stopt je functie met een fout: in de laatste kolom is `j` de lengte van `b`, en in de laatste rij is `i` de lengte van `a`, en op die plek heeft een string geen letter. Rij `i` van de tabel gaat over de eerste `i` letters van `a`, dus de nieuwe letter is de letter vóór plek `i`; voor kolom `j` en `b` geldt hetzelfde."),
            Mistake("a-table", matches=_a_table,
                    en="`{call}` gives the whole table, and it should give one number, {want}: the edit distance is in the last cell, at the last row and the last column.",
                    nl="`{call}` geeft de hele tabel, en dat moet één getal zijn, {want}: de edit distance staat in de laatste cel, in de laatste rij en de laatste kolom."),
            Mistake("cost-two", wrong=_E_TWO,
                    en="`{call}` gives {got}, and it should give {want}: your function gives a substitution the cost 2, as in Jurafsky and Martin's table. In this exercise every edit costs 1.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie laat vervangen 2 kosten, zoals in de tabel van Jurafsky en Martin. In deze oefening kost elke bewerking 1."),
            Mistake("cost-flipped", wrong=_E_COST_FLIPPED,
                    en="`{call}` gives {got}, and it should give {want}: your function makes two equal letters cost 1 and two different letters cost 0. Keeping a letter is free; a substitution costs 1.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie laat twee gelijke letters 1 kosten en twee verschillende letters 0. Een letter houden is gratis; vervangen kost 1."),
            Mistake("max", wrong=_E_MAX,
                    en="`{call}` gives {got}, and it should give {want}: your function takes the largest of the three ways into a cell. The edit distance is the cheapest way: take the smallest, with `min`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie neemt de grootste van de drie wegen naar een cel. De edit distance is de goedkoopste weg: neem de kleinste, met `min`."),
            Mistake("no-plus-one", wrong=_E_NO_ONE,
                    en="`{call}` gives {got}, and it should give {want}: your function takes the cell above and the cell to the left without adding 1. A step down is a deletion and a step to the right an insertion, and each costs 1.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie neemt de cel erboven en de cel links ervan zonder 1 op te tellen. Een stap omlaag is weghalen en een stap naar rechts invoegen, en elk kost 1."),
            Mistake("from-zero", wrong=_E_FROM_0,
                    en="`{call}` gives {got}, and it should give {want}: your loops start at row 0 and column 0, and so write over the start of the table. Those cells are already right; start both loops at 1.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je loops beginnen bij rij 0 en kolom 0, en schrijven zo over het begin van de tabel heen. Die cellen kloppen al; laat beide loops bij 1 beginnen."),
            Mistake("stops-early", wrong=_E_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your loops stop one row or one column early, so the last cell keeps its 0. `range(1, n)` stops before `n`, and the last row is row `len(a)`, the last column column `len(b)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je loops stoppen een rij of een kolom te vroeg, dus de laatste cel houdt de 0. `range(1, n)` stopt vóór `n`, en de laatste rij is rij `len(a)`, de laatste kolom kolom `len(b)`."),
            Mistake("return-in-loop", wrong=_E_RETURN_IN,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back the last cell after the first cell it fills, when the last cell is still empty. Put the `return` after both loops, at the same depth as the first `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft de laatste cel al terug na de eerste cel die je functie invult, als de laatste cel nog leeg is. Zet de `return` na beide loops, even ver ingesprongen als de eerste `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: start with `D = start_table(a, b)`, fill every other cell, row by row, with the smallest of the three ways into it, and give back the last cell.",
                      "nl": "Vervang de `...`: begin met `D = start_table(a, b)`, vul elke andere cel, rij na rij, met de kleinste van de drie wegen ernaartoe, en geef de laatste cel terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function fill each cell with the smallest of three: the cell above plus 1, the cell to the left plus 1, and the cell up and to the left plus 0 for the same letter or 1 for a different one?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Vult je functie elke cel met de kleinste van drie: de cel erboven plus 1, de cel links plus 1, en de cel linksboven plus 0 voor dezelfde letter of 1 voor een andere?"},
            "almost": {"en": "Right for *phone* and *power*. But `{call}` gives {got}, and it should give {want}. Does your function work for every pair: the same word, an empty word, lists of words?",
                       "nl": "Goed voor *phone* en *power*. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elk paar: hetzelfde woord, een leeg woord, lijsten woorden?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back. Does it end with a `return` of the last cell, after both loops?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie met een `return` van de laatste cel, na beide loops?"},
            "correct": {"en": "Your `edit_distance` is right in all {n} cases. Look at the picture: your function on every first part of *phone* and *power*, which makes the whole table, and on other pairs, also of words.",
                        "nl": "Je `edit_distance` klopt in alle {n} gevallen. Kijk naar de afbeelding: je functie op elk beginstuk van *phone* en *power*, wat de hele tabel geeft, en op andere paren, ook van woorden."},
        },
    ),
    "how-many-edits": Sort(
        name="how-many-edits",
        groups={
            "one": {"en": "One edit", "nl": "Eén bewerking"},
            "two": {"en": "Two edits", "nl": "Twee bewerkingen"},
            "three": {"en": "Three edits", "nl": "Drie bewerkingen"},
        },
        items={
            "insert": {"en": "**From *cat* to *cast*.**", "nl": "**Van *kat* naar *kast*.**"},
            "accent": {"en": "**From *café* to *cafe*.**", "nl": "**Van *café* naar *cafe*.**"},
            "capital": {"en": "**From *Edit* to *edit*.**", "nl": "**Van *Boek* naar *boek*.**"},
            "swap": {"en": "**From *form* to *from*.**", "nl": "**Van *zeil* naar *ziel*.**"},
            "ends": {"en": "**From *flaw* to *lawn*.**", "nl": "**Van *slang* naar *langs*.**"},
            "vowels": {"en": "**From *book* to *back*.**", "nl": "**Van *boot* naar *baat*.**"},
            "kitten": {"en": "**From *kitten* to *sitting*.**", "nl": "**Van *zon* naar *maan*.**"},
            "days": {"en": "**From *Sunday* to *Saturday*.**", "nl": "**Van *lopen* naar *liep*.**"},
            "animals": {"en": "**From *cat* to *dog*.**", "nl": "**Van *rood* naar *groen*.**"},
        },
        solution={"insert": "one", "accent": "one", "capital": "one", "swap": "two", "ends": "two", "vowels": "two",
                  "kitten": "three", "days": "three", "animals": "three"},
        nudge={
            "insert": {"en": "Line the two words up: which letter of *cast* has no partner in *cat*?",
                       "nl": "Lijn de twee woorden uit: welke letter van *kast* heeft geen partner in *kat*?"},
            "accent": {"en": "Is *é* the same character as *e*, for a program that compares letters exactly?",
                       "nl": "Is *é* hetzelfde teken als *e*, voor een programma dat letters precies vergelijkt?"},
            "capital": {"en": "Is *E* the same character as *e*, for a program that compares letters exactly?",
                        "nl": "Is *B* hetzelfde teken als *b*, voor een programma dat letters precies vergelijkt?"},
            "swap": {"en": "Two letters change places. Can a single substitution do that, or does each of the two letters need a change of its own?",
                     "nl": "Twee letters wisselen van plaats. Lukt dat met één keer vervangen, of heeft elk van de twee letters een eigen verandering nodig?"},
            "ends": {"en": "What happens at the front of *flaw*, and what at the end of *lawn*?",
                     "nl": "Wat gebeurt er vooraan bij *slang*, en wat achteraan bij *langs*?"},
            "vowels": {"en": "Line the words up letter by letter: at how many places do the letters differ?",
                       "nl": "Lijn de woorden letter voor letter uit: op hoeveel plekken verschillen de letters?"},
            "kitten": {"en": "Which letters of *kitten* must change, and what must come in at the end?",
                       "nl": "Hoeveel letters van *zon* passen bij een letter van *maan*, en wat moet er nog bij?"},
            "days": {"en": "*Sunday* and *Saturday* share *S* at the start and *day* at the end. What happens between them?",
                     "nl": "*lopen* en *liep* beginnen allebei met een *l*. Wat moet er daarna veranderen, weg of erbij?"},
            "animals": {"en": "Do *cat* and *dog* share a single letter in the same place?",
                        "nl": "Welke letters van *rood* staan ook in *groen*, in dezelfde volgorde, en wat moet er met de rest gebeuren?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the pairs again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de paren opnieuw uit."},
            "empty": {"en": "Put each bold pair under its edit distance.",
                      "nl": "Zet elk vetgedrukt paar bij de edit distance."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item}: this pair goes somewhere else.",
                      "nl": "{item}: dit paar hoort ergens anders."},
            "correct": {"en": "All {n} are right. An accent and a capital each count as a substitution, so normalize first where they should not count. Two letters that swap places cost two edits here; the Damerau-Levenshtein distance counts such a swap as one. And words that look nothing alike can still be only three edits apart.",
                        "nl": "Alle {n} kloppen. Een accent en een hoofdletter tellen elk als een keer vervangen, dus normaliseer eerst waar die niet mogen tellen. Twee letters die van plaats wisselen, kosten hier twee bewerkingen; de Damerau-Levenshtein distance telt zo'n verwisseling als één. En woorden die helemaal niet op elkaar lijken, kunnen toch maar drie bewerkingen uit elkaar liggen."},
        },
    ),
}

CASES = [
    ("count-edits", _C_START, "empty", None),
    ("count-edits", _C, "correct", None),
    ("count-edits", "def count_edits(top, bottom):\n    counts = {\"insert\": 0, \"delete\": 0, \"substitute\": 0}\n    for x, y in zip(top, bottom):\n        if x == \"-\":\n            counts[\"insert\"] += 1\n        elif y == \"-\":\n            counts[\"delete\"] += 1\n        elif x != y:\n            counts[\"substitute\"] += 1\n    return counts\n", "correct", None),
    ("count-edits", "def count_edits(top, bottom):\n    ins = sum(1 for x in top if x == \"-\")\n    dels = sum(1 for y in bottom if y == \"-\")\n    subs = sum(1 for x, y in zip(top, bottom) if x != y and \"-\" not in (x, y))\n    return {\"substitute\": subs, \"delete\": dels, \"insert\": ins}\n", "correct", None),
    ("count-edits", _C_TOTAL, "wrong", "a-number"),
    ("count-edits", _C_SWAPPED, "wrong", "swapped"),
    ("count-edits", _C_MATCHES, "wrong", "matches"),
    ("count-edits", _C_GAP_FIRST, "wrong", "gap-first"),
    ("count-edits", _C_RETURN_IN, "wrong", "return-in-loop"),
    ("count-edits", _C.replace("    return counts\n", ""), "wrong", "no-return"),
    ("count-edits", "def count_edits(top, bottom):\n    return {\"insert\": 1, \"delete\": 1, \"substitute\": 1}\n", "almost", None),
    ("start-table", _S_START, "empty", None),
    ("start-table", _S, "correct", None),
    ("start-table", "def start_table(a, b):\n    D = np.zeros((len(a) + 1, len(b) + 1))\n    D[:, 0] = np.arange(len(a) + 1)\n    D[0, :] = np.arange(len(b) + 1)\n    return D\n", "correct", None),
    ("start-table", "def start_table(a, b):\n    D = [[0] * (len(b) + 1) for i in range(len(a) + 1)]\n    for i in range(len(a) + 1):\n        D[i][0] = i\n    for j in range(len(b) + 1):\n        D[0][j] = j\n    return D\n", "correct", None),
    ("start-table", _S_SMALL, "wrong", "small"),
    ("start-table", _S_SMALL.replace("range(len(a))", "range(len(a) + 1)"), "error", "too-small"),
    ("start-table", _S_SWAPPED, "wrong", "swapped"),
    ("start-table", _S.replace("for i in range(len(a) + 1)", "for i in range(len(b) + 1)"), "error", "too-small"),
    ("start-table", _S_COLUMN, "wrong", "column-only"),
    ("start-table", _S_ROW, "wrong", "row-only"),
    ("start-table", _S.replace("    return D\n", ""), "wrong", "no-return"),
    ("start-table", "def start_table(a, b):\n    return np.array([[0, 1, 2, 3, 4, 5], [1, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0], [4, 0, 0, 0, 0, 0], [5, 0, 0, 0, 0, 0]])\n", "almost", None),
    ("edit-distance", _E_START, "empty", None),
    ("edit-distance", _E, "correct", None),
    ("edit-distance", "def edit_distance(a, b):\n    D = start_table(a, b)\n    for i in range(1, len(a) + 1):\n        for j in range(1, len(b) + 1):\n            if a[i - 1] == b[j - 1]:\n                D[i, j] = D[i - 1, j - 1]\n            else:\n                D[i, j] = 1 + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])\n    return D[-1, -1]\n", "correct", None),
    ("edit-distance", "def edit_distance(a, b):\n    row = list(range(len(b) + 1))\n    for i in range(1, len(a) + 1):\n        new = [i]\n        for j in range(1, len(b) + 1):\n            new.append(min(row[j] + 1, new[j - 1] + 1, row[j - 1] + (a[i - 1] != b[j - 1])))\n        row = new\n    return row[-1]\n", "correct", None),
    ("edit-distance", _E_INDEX, "error", "index"),
    ("edit-distance", _E_TABLE, "wrong", "a-table"),
    ("edit-distance", _E_TWO, "wrong", "cost-two"),
    ("edit-distance", _E_COST_FLIPPED, "wrong", "cost-flipped"),
    ("edit-distance", _E_MAX, "wrong", "max"),
    ("edit-distance", _E_NO_ONE, "wrong", "no-plus-one"),
    ("edit-distance", _E_FROM_0, "wrong", "from-zero"),
    ("edit-distance", _E_RETURN_IN, "wrong", "return-in-loop"),
    ("edit-distance", _E_EARLY, "wrong", "stops-early"),
    ("edit-distance", _E.replace("range(1, len(a) + 1)", "range(1, len(a))"), "wrong", "stops-early"),
    ("edit-distance", _E.replace("    return D[len(a), len(b)]\n", ""), "wrong", "no-return"),
    ("edit-distance", "def edit_distance(a, b):\n    return 3\n", "almost", None),
    ("how-many-edits", {}, "empty", None),
    ("how-many-edits", {"insert": "one", "accent": "one", "capital": "one", "swap": "two", "ends": "two",
                        "vowels": "two", "kitten": "three", "days": "three", "animals": "three"}, "correct", None),
    ("how-many-edits", {"insert": "one", "kitten": "three"}, "wrong", "unfinished"),
    ("how-many-edits", {"swap": "one"}, "wrong", "wrong:swap"),
    ("how-many-edits", {"insert": "one", "accent": "two", "capital": "one", "swap": "two", "ends": "two",
                        "vowels": "two", "kitten": "three", "days": "three", "animals": "three"}, "wrong", "wrong:accent"),
    ("how-many-edits", {"insert": "one", "accent": "one", "capital": "one", "swap": "two", "ends": "two",
                        "vowels": "two", "kitten": "three", "days": "two", "animals": "three"}, "wrong", "wrong:days"),
]
