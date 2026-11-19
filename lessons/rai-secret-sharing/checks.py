# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-secret-sharing: secrets split into shares that add up to them modulo M = 100,000, as in
additive secret sharing. Five made-up companies each keep a number secret (SECRETS) and split it into three shares,
one for each of three parties (SHARES: SHARES[i][j] is share j of company i, and party j holds it). The three
functions the learner writes: share (split a secret, given the random numbers), reveal (add shares up modulo M) and
party_totals (what each party adds up from the shares it holds); with the pictures of what they give. The demo draws
the last share of two secrets, made with the learner's own share, for random numbers from ranges of different
widths."""
import copy
import html
import math
import random
import re

from everything_ai import chart, yours
from everything_ai.check import FunctionBox, Mistake, with_time_limit

M = 100000       # every share is a whole number from 0 to M - 1
PARTIES = 3      # the parties of exercise 3, each holding one share of every company
SEED = 2018


class Named(list):
    """A list the check calls by the name the lesson gives it, such as SHARES, so that a message stays short. The
    learner's code gets its own copy, a plain list."""

    def __init__(self, values, name: str):
        super().__init__(values)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(list(self), memo)


def share(secret, randoms, M):
    shares = []
    left = secret
    for r in randoms:
        shares.append(r)
        left = left - r
    shares.append(left % M)
    return shares


def reveal(shares, M):
    total = 0
    for s in shares:
        total = total + s
    return total % M


def party_totals(table, M):
    totals = []
    for j in range(len(table[0])):
        total = 0
        for row in table:
            total = total + row[j]
        totals.append(total % M)
    return totals


# what five made-up companies pay their women employees in a year, in thousands of dollars
SECRETS = Named([3120, 870, 12450, 5300, 760], "SECRETS")


def _shares() -> list:
    rng = random.Random(SEED)
    return [share(s, [rng.randint(0, M - 1) for _ in range(PARTIES - 1)], M) for s in SECRETS]


SHARES = Named(_shares(), "SHARES")


# --- formatting -------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_BOX = "padding:0.25rem 0.5rem;margin:0.2rem 0;border-radius:4px;"
_PLAIN = "border:1px solid var(--ea-line);color:var(--ea-ink)"
_WRONG = "border:1.5px dashed var(--ea-loss);color:var(--ea-loss-text)"
_CODE = "font-family:var(--ea-font-code);font-size:0.8rem"


def grouped(n: int, lang: str) -> str:
    """A whole number as the lesson's language writes it: 25,000 in English, 25.000 in Dutch (a dot from five
    digits), with a real minus sign."""
    text = f"{abs(n):,}"
    if lang == "nl":
        text = text.replace(",", ".") if abs(n) >= 10000 else str(abs(n))
    return ("−" if n < 0 else "") + text


def _row(cells: list, style: str, widths: list) -> str:
    parts = "".join(f'<div style="flex:{w};min-width:0;overflow-wrap:anywhere">{c}</div>' for c, w in zip(cells, widths))
    return f'<div style="display:flex;gap:0.6rem;align-items:baseline;{_BOX}{style}">{parts}</div>'


def _head(cells: list, widths: list) -> str:
    parts = "".join(f'<div style="flex:{w};min-width:0;overflow-wrap:anywhere">{html.escape(c)}</div>'
                    for c, w in zip(cells, widths))
    return (f'<div style="display:flex;gap:0.6rem;font-weight:600;color:var(--ea-ink);padding:0 0.5rem;'
            f'border-bottom:2px solid var(--ea-ink-3);margin-bottom:0.2rem">{parts}</div>')


def _table(head: list, rows: list, widths: list, width: str = "30rem") -> str:
    return (f'<div style="font-family:var(--ea-font-ui);font-size:0.85rem;max-width:{width};flex:1 1 18rem">'
            f'{_head(head, widths)}{"".join(rows)}</div>')


def _code(text: str) -> str:
    return f'<span style="{_CODE}">{html.escape(text)}</span>'


def _fn(seen):
    mine = seen.mine
    if mine is None:
        return None
    fn = mine.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _limited(work):
    return with_time_limit(work)


def _whole(v):
    """A number that is a whole number, as an int, or None: 24367 and 24367.0 count, 24367.5 and True do not."""
    if hasattr(v, "item") and not isinstance(v, (list, tuple)):   # a NumPy number
        try:
            v = v.item()
        except Exception:
            return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if isinstance(v, float) and not (math.isfinite(v) and v == int(v)):
        return None
    return int(v)


def _wholes(v):
    """A list of whole numbers, as ints, or None."""
    if not isinstance(v, (list, tuple)):
        return None
    out = [_whole(x) for x in v]
    return None if any(x is None for x in out) else out


def _shown(v) -> str:
    """A value a function gave back, the way Python shows it, kept short."""
    try:
        text = repr(v)
    except Exception:
        text = "?"
    return text if len(text) <= 60 else text[:57] + "..."


# --- the clock ----------------------------------------------------------------------------------------------------------

_CX, _CY, _TRACK = 150, 128, 96


def _at(v: float, r: float) -> tuple:
    a = 2 * math.pi * (v % M) / M
    return _CX + r * math.sin(a), _CY - r * math.cos(a)


def clock(steps: list, lang: str, key: str, mark: int | None = None, mark_label: str = "") -> str:
    """Shares as a walk round a clock of M marks: from 0 at the top, clockwise, each share an arc on a ring of its
    own, from the outer ring inward; where the walk ends, a dot on the outer track with the total modulo M. With
    `mark`, a value that differs from that end, an open dot with `mark_label` beside it. `steps` are whole numbers
    from 0 to M - 1, at most five."""
    parts = [f'<circle cx="{_CX}" cy="{_CY}" r="{_TRACK}" style="fill:none;stroke:var(--ea-line);stroke-width:1.5"/>']
    for q in range(4):   # a tick and its number at 0, M/4, M/2 and 3M/4
        v = int(q * M / 4)
        x0, y0 = _at(v, _TRACK - 4)
        x1, y1 = _at(v, _TRACK + 4)
        parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" style="stroke:var(--ea-ink-3);stroke-width:1.5"/>')
        tx, ty = _at(v, _TRACK + 9)
        anchor = {0: "middle", 1: "start", 2: "middle", 3: "end"}[q]
        dy = {0: -2, 1: 4, 2: 12, 3: 4}[q]
        parts.append(f'<text x="{tx:.1f}" y="{ty + dy:.1f}" text-anchor="{anchor}" style="fill:var(--ea-ink-3);font-size:11px">'
                     f'{html.escape(grouped(v, lang))}</text>')
    pos = 0
    rings = [_TRACK - 14 - 12 * k for k in range(len(steps))]
    for k, step in enumerate(steps):
        r = rings[k]
        if k > 0:   # step inward to the next ring, where the walk goes on
            xa, ya = _at(pos, rings[k - 1])
            xb, yb = _at(pos, r)
            parts.append(f'<line x1="{xa:.1f}" y1="{ya:.1f}" x2="{xb:.1f}" y2="{yb:.1f}" style="stroke:var(--ea-ink-3);stroke-width:1.2"/>')
        if step > 0:
            x0, y0 = _at(pos, r)
            x1, y1 = _at(pos + step, r)
            large = 1 if step > M / 2 else 0
            parts.append(f'<path d="M{x0:.1f},{y0:.1f} A{r},{r} 0 {large} 1 {x1:.1f},{y1:.1f}" '
                         f'marker-end="url(#ea-ss-{key})" style="fill:none;stroke:var(--ea-ink-2);stroke-width:2.4"/>')
        pos += step
    xs, ys = _at(0, rings[0] if rings else _TRACK)
    parts.append(f'<circle cx="{xs:.1f}" cy="{ys:.1f}" r="3.2" style="fill:var(--ea-ink-2)"/>')
    end = pos % M
    x0, y0 = _at(pos, rings[-1] if rings else _TRACK)
    x1, y1 = _at(pos, _TRACK)
    parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" style="stroke:var(--ea-ink);stroke-width:1.2;stroke-dasharray:3 3"/>')
    parts.append(f'<circle cx="{x1:.1f}" cy="{y1:.1f}" r="4.5" style="fill:var(--ea-ink)"/>')

    def label(v, words, bold):
        a = 2 * math.pi * (v % M) / M
        tx, ty = _at(v, _TRACK + 13)
        anchor = "start" if math.sin(a) > 0.25 else "end" if math.sin(a) < -0.25 else "middle"
        ty += 12 if math.cos(a) < -0.5 else -4 if math.cos(a) > 0.5 else 4
        weight = "font-weight:600;" if bold else ""
        return (f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="{anchor}" style="fill:var(--ea-ink);font-size:12px;{weight}'
                f'paint-order:stroke;stroke:var(--ea-page);stroke-width:3px">{html.escape(words)}</text>')
    parts.append(label(end, grouped(end, lang), True))
    if mark is not None and mark != end and 0 <= mark < M:
        xm, ym = _at(mark, _TRACK)
        parts.append(f'<circle cx="{xm:.1f}" cy="{ym:.1f}" r="4.5" style="fill:var(--ea-page);stroke:var(--ea-loss);stroke-width:1.8"/>')
        parts.append(label(mark, mark_label, False))
    marker = (f'<defs><marker id="ea-ss-{key}" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="7" markerHeight="7" '
              f'orient="auto"><path d="M0,0 L8,4 L0,8 z" style="fill:var(--ea-ink-2)"/></marker></defs>')
    return (f'<svg viewBox="0 0 300 262" role="img" aria-hidden="true" style="flex:0 1 300px;max-width:300px;min-width:0;'
            f'height:auto;font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums">{marker}{"".join(parts)}</svg>')


def _side(svg: str, table: str) -> str:
    return f'<div style="display:flex;flex-wrap:wrap;gap:0.8rem;align-items:center">{svg}{table}</div>'


# --- the pictures under the checks --------------------------------------------------------------------------------------

_STOPPED = object()   # what a function gave back when it stopped with an error
SHARE_CALLS = [(4200, [71320, 8513]), (4200, [5000, 90000]), (70000, [20000])]

_SHARE_WORDS = {
    "en": {"title": "Your share(secret, randoms, M) on a clock of 100,000",
           "head": ["secret", "your shares", "sum", "% M"],
           "walk": "Each arc is one of your shares of 4,200 in share(4200, [71320, 8513], 100000), from 0 at the top "
                   "and from the outer ring inward. Together they go round the clock {turns} and end at {end}.",
           "once": "once", "times": "{n} times", "none": "less than once",
           "off": "Your share(4200, [71320, 8513], 100000) gives {got}, which the clock cannot draw: it draws one to "
                  "five whole numbers from 0 to 99,999.",
           "table": "The table runs your share on three secrets with other random numbers.",
           "fine": "The shares look nothing like the secret, and modulo M they add up to it.",
           "stops": "Your share stops with an error on share(4200, [71320, 8513], 100000): the check above says "
                    "where.",
           "wrong": "A dashed frame marks where your shares differ from the course's."},
    "nl": {"title": "Jouw share(secret, randoms, M) op een klok van 100.000",
           "head": ["geheim", "jouw shares", "som", "% M"],
           "walk": "Elke boog is een van jouw shares van 4200 in share(4200, [71320, 8513], 100000), vanaf 0 bovenaan "
                   "en van de buitenste ring naar binnen. Samen gaan ze {turns} de klok rond en eindigen ze bij {end}.",
           "once": "één keer", "times": "{n} keer", "none": "minder dan één keer",
           "off": "Jouw share(4200, [71320, 8513], 100000) geeft {got}, en dat kan de klok niet tekenen: die tekent "
                  "één tot vijf hele getallen van 0 tot 99.999.",
           "table": "De tabel laat jouw share lopen op drie geheimen met andere willekeurige getallen.",
           "fine": "De shares lijken in niets op het geheim, en modulo M tellen ze er precies toe op.",
           "stops": "Jouw share stopt met een fout bij share(4200, [71320, 8513], 100000): de check hierboven zegt "
                    "waar.",
           "wrong": "Een gestreept kader wijst aan waar jouw shares verschillen van die van de course."},
}


def _turns(total: int, w: dict) -> str:
    n = int(total / M)
    return w["none"] if n == 0 else w["once"] if n == 1 else w["times"].format(n=n)


def share_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w, lang = _SHARE_WORDS[seen.lang], seen.lang
    rows, notes, wrong_any, first, first_got = [], [], False, None, None
    widths = [2, 5, 2, 2]
    for k, (secret, randoms) in enumerate(SHARE_CALLS):
        try:
            got = _limited(lambda: fn(secret, list(randoms), M))
        except Exception:
            got = _STOPPED
        mine = None if got is _STOPPED else _wholes(got)
        want = share(secret, list(randoms), M)
        wrong = mine != want
        wrong_any |= wrong
        if k == 0:
            first, first_got = mine, got
        total = sum(mine) if mine is not None else None
        rows.append(_row([_code(str(secret)), _code("–" if got is _STOPPED else _shown(got)),
                          _code("–" if total is None else str(total)), _code("–" if total is None else str(total % M))],
                         _WRONG if wrong else _PLAIN, widths))
    table = _table(w["head"], rows, widths)
    if first is not None and 0 < len(first) <= 5 and all(0 <= v < M for v in first):
        body = _side(clock(first, lang, "share"), table)
        notes.append(w["walk"].format(turns=_turns(sum(first), w), end=grouped(sum(first) % M, lang)))
    else:
        body = table
        notes.append(w["stops"] if first_got is _STOPPED else w["off"].format(got=_shown(first_got)))
    notes.append(w["table"])
    notes.append(w["wrong"] if wrong_any else w["fine"])
    return yours.figure(w["title"], body + f'<div style="{_UI}">{html.escape(" ".join(notes))}</div>')


REVEAL_CALLS = [([71320, 8513, 24367], 4200), ([71320, 8513], None), ([5000, 90000, 9200], 4200), ([20000, 50000], 70000)]

_REVEAL_WORDS = {
    "en": {"title": "Your reveal(shares, M) on all shares of a secret, and on all but one",
           "head": ["shares", "sum", "your reveal", "secret"],
           "missing": "4200, one share missing",
           "walk": "The clock walks the three shares of 4,200 and ends at {end}; your reveal of them gives {mine}.",
           "short": "Without the last share, your reveal gives {b}: with one share missing, every value from 0 to "
                    "99,999 is just as likely to be the secret.",
           "stops": "Your reveal stops with an error on reveal([71320, 8513, 24367], 100000): the check above says "
                    "where.",
           "off": "Your reveal gives {got} on reveal([71320, 8513, 24367], 100000), not a whole number, so it has no "
                  "place on the clock.",
           "yours": "your reveal",
           "wrong": "A dashed frame marks where your reveal differs from the course's."},
    "nl": {"title": "Jouw reveal(shares, M) op alle shares van een geheim, en op alle op één na",
           "head": ["shares", "som", "jouw reveal", "geheim"],
           "missing": "4200, één share ontbreekt",
           "walk": "De klok loopt de drie shares van 4200 af en eindigt bij {end}; jouw reveal ervan geeft {mine}.",
           "short": "Zonder de laatste share geeft jouw reveal {b}: als er één share ontbreekt, is elke waarde van 0 "
                    "tot 99.999 even waarschijnlijk het geheim.",
           "stops": "Jouw reveal stopt met een fout bij reveal([71320, 8513, 24367], 100000): de check hierboven zegt "
                    "waar.",
           "off": "Jouw reveal geeft {got} bij reveal([71320, 8513, 24367], 100000), geen heel getal, dus het heeft "
                  "geen plek op de klok.",
           "yours": "jouw reveal",
           "wrong": "Een gestreept kader wijst aan waar jouw reveal verschilt van die van de course."},
}


def reveal_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w, lang = _REVEAL_WORDS[seen.lang], seen.lang
    rows, got_all, raw, wrong_any = [], [], [], False
    widths = [5, 2, 2, 3]
    for shares, secret in REVEAL_CALLS:
        try:
            got = _limited(lambda: fn(list(shares), M))
        except Exception:
            got = _STOPPED
        mine = None if got is _STOPPED else _whole(got)
        got_all.append(mine)
        raw.append(got)
        wrong = mine != reveal(shares, M)
        wrong_any |= wrong
        rows.append(_row([_code(str(shares)), _code(str(sum(shares))),
                          _code("–" if got is _STOPPED else _shown(got)),
                          _code(str(secret)) if secret is not None else html.escape(w["missing"])],
                         _WRONG if wrong else _PLAIN, widths))
    table = _table(w["head"], rows, widths)
    first = REVEAL_CALLS[0][0]
    mark = got_all[0]
    body = _side(clock(first, lang, "reveal", mark=mark, mark_label=w["yours"]), table)
    notes = []
    if mark is None:
        notes.append(w["stops"] if raw[0] is _STOPPED else w["off"].format(got=_shown(raw[0])))
    else:
        notes.append(w["walk"].format(end=grouped(sum(first) % M, lang), mine=grouped(mark, lang)))
    if got_all[1] is not None:
        notes.append(w["short"].format(b=grouped(got_all[1], lang)))
    if wrong_any:
        notes.append(w["wrong"])
    return yours.figure(w["title"], body + f'<div style="{_UI}">{html.escape(" ".join(notes))}</div>')


_TOTALS_WORDS = {
    "en": {"title": "Your party_totals(table, M) on SHARES: what each party adds up",
           "head": ["company", "secret", "party 0", "party 1", "party 2"],
           "company": "company {i}", "totals": "your totals",
           "key": "Each company knows only its own secret, and party j sees only column j: five numbers that each "
                  "look like random noise.",
           "sum": "Your totals added up, modulo 100,000: {a}. The five secrets add up to {b}.",
           "count": "Your party_totals gives back {n} values, where there are 3 parties.",
           "stops": "Your party_totals stops with an error on SHARES: the check above says where.",
           "wrong": "A dashed frame marks where your totals differ from the course's."},
    "nl": {"title": "Jouw party_totals(table, M) op SHARES: wat elke partij optelt",
           "head": ["bedrijf", "geheim", "partij 0", "partij 1", "partij 2"],
           "company": "bedrijf {i}", "totals": "jouw totalen",
           "key": "Elk bedrijf kent alleen het eigen geheim, en partij j ziet alleen kolom j: vijf getallen die elk "
                  "op willekeurige ruis lijken.",
           "sum": "Jouw totalen opgeteld, modulo 100.000: {a}. De vijf geheimen tellen op tot {b}.",
           "count": "Jouw party_totals geeft {n} waarden terug, terwijl er 3 partijen zijn.",
           "stops": "Jouw party_totals stopt met een fout op SHARES: de check hierboven zegt waar.",
           "wrong": "Een gestreept kader wijst aan waar jouw totalen verschillen van die van de course."},
}


def totals_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w, lang = _TOTALS_WORDS[seen.lang], seen.lang
    try:
        got = _limited(lambda: fn(copy.deepcopy(SHARES), M))
        stopped = False
    except Exception:
        got, stopped = None, True
    mine = _wholes(got)
    want = party_totals(SHARES, M)
    widths = [3, 2, 2, 2, 2]
    rows = [_row([html.escape(w["company"].format(i=i)), _code(str(SECRETS[i]))] + [_code(str(v)) for v in row],
                 _PLAIN, widths) for i, row in enumerate(SHARES)]
    fits = mine is not None and len(mine) == PARTIES
    cells = [_code(str(mine[j])) if fits else _code("–") for j in range(PARTIES)]
    wrong = mine != want
    rows.append(_row([f'<b>{html.escape(w["totals"])}</b>', ""] + cells, _WRONG if wrong else _PLAIN, widths))
    notes = [w["key"]]
    if stopped:
        notes.append(w["stops"])
    elif mine is not None and not fits:
        notes.append(w["count"].format(n=len(mine)))
    elif fits:
        notes.append(w["sum"].format(a=grouped(sum(mine) % M, lang), b=grouped(sum(SECRETS), lang)))
    if wrong and not stopped:
        notes.append(w["wrong"])
    return yours.figure(w["title"], _table(w["head"], rows, widths, width="36rem") +
                        f'<div style="{_UI}">{html.escape(" ".join(notes))}</div>')


# --- the demo: what one share gives away -------------------------------------------------------------------------------

LOW, HIGH = 20000, 70000          # the two secrets the demo shares
DOTS = 200                        # sharings of each secret
WIDTHS = [1000, 10000, 30000, 50000, 60000, 80000, 100000]   # W: the random number lies from 0 to W - 1


def mine(box_text, correct: bool):
    """The learner's own share when exercise 1 is right, else None: the demo then uses the course's."""
    if not correct or not isinstance(box_text, str):
        return None
    from everything_ai.check import run_code
    run = run_code(box_text, {})
    fn = run.names.get("share") if run.error is None else None
    return fn if callable(fn) else None


def leak_runs(fn=None) -> dict | None:
    """For every W: the last share of LOW and of HIGH in DOTS sharings each, made with `fn` (the learner's share, or
    the course's), the random number drawn from 0 to W - 1. None when `fn` stops with an error, gives no whole number
    from 0 to M - 1 as its last share, or takes more than ten seconds."""
    fn = fn or share
    out = {}

    def work():
        for width in WIDTHS:
            runs = {}
            for secret, seed in ((LOW, 1), (HIGH, 2)):
                rng = random.Random(1000 * seed + width)
                lasts = []
                for _ in range(DOTS):
                    got = fn(secret, [rng.randint(0, width - 1)], M)
                    last = _whole(got[-1]) if isinstance(got, (list, tuple)) and got else None
                    if last is None or not 0 <= last < M:
                        raise ValueError("no share")
                    lasts.append(last)
                runs[secret] = lasts
            out[width] = runs

    try:
        with_time_limit(work, 10)
    except Exception:
        return None
    return out


def telling(runs: dict, width: int) -> int:
    """How many of the 2 · DOTS last shares only one of the two secrets can give: a last share v of one secret is
    out of reach of the other secret s when (s - v) % M is W or more."""
    lasts = runs[width]
    return (sum(1 for v in lasts[LOW] if (HIGH - v) % M >= width) +
            sum(1 for v in lasts[HIGH] if (LOW - v) % M >= width))


_LEAK_WORDS = {
    "en": {"x": "the last share, in thousands", "low": "last shares of 20,000", "high": "last shares of 70,000",
           "alt": "Dots: the last share of 20,000, filled, and of 70,000, open, in 200 sharings each, with the random "
                  "number from 0 to {w}.",
           "said": "W = {w}: the random number lies from 0 to {w1}. In {p} of the 400 sharings, the last share alone "
                   "says which of the two secrets was shared."},
    "nl": {"x": "de laatste share, in duizendtallen", "low": "laatste shares van 20.000", "high": "laatste shares van 70.000",
           "alt": "Stippen: de laatste share van 20.000, gevuld, en van 70.000, open, bij elk 200 keer verdelen, met het "
                  "willekeurige getal van 0 tot {w}.",
           "said": "W = {w}: het willekeurige getal ligt van 0 tot {w1}. Bij {p} van de 400 keer verdelen zegt de "
                   "laatste share alleen al welk van de twee geheimen verdeeld werd."},
}


def leak_frame(runs: dict, width: int, lang: str) -> tuple:
    """The picture of the demo for one W, and the sentence under it."""
    w = _LEAK_WORDS[lang]
    lasts = runs[width]
    panel = chart.Panel(x=(0, M / 1000), y=(0, 2.3), xlabel=w["x"], yticks=[],
                        xticks=[0, 25, 50, 75, 100])
    jitter = random.Random(7)
    for secret, y, hollow in ((LOW, 1.6, False), (HIGH, 0.5, True)):
        ys = [y + jitter.uniform(-0.24, 0.24) for _ in lasts[secret]]
        panel.points([v / 1000 for v in lasts[secret]], ys, role="ink-2", r=2.6, hollow=hollow)
        panel.line([secret / 1000, secret / 1000], [y - 0.32, y + 0.32], role="ink", width=1.6, dashed=True)
    panel.text(1, 2.08, w["low"], role="ink-2", anchor="start", halo=True)
    panel.text(1, 0.98, w["high"], role="ink-2", anchor="start", halo=True)
    figure = chart.figure([panel], alt=w["alt"].format(w=grouped(width - 1, lang)), width=460, height=240)
    p = telling(runs, width) / (2 * DOTS)
    percent = f"{100 * p:.0f}%"
    return figure, w["said"].format(w=grouped(width, lang), w1=grouped(width - 1, lang), p=percent)


# --- the exercises ----------------------------------------------------------------------------------------------------

REQUIRED = ["share", "reveal", "party-totals", "two-companies", "collude"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# exercise 1: share
_S_HEAD = ("def share(secret, randoms, M):\n"
           "    # the random numbers are the first shares; the last share is what is left of secret, modulo M\n"
           "    shares = []\n"
           "    left = secret\n"
           "    for r in randoms:\n"
           "        shares.append(r)\n")
_S_START = _S_HEAD + "        left = ...\n    shares.append(...)\n    return shares\n"
_S_LOOP = "        left = left - r\n"
_S = _S_HEAD + _S_LOOP + "    shares.append(left % M)\n    return shares\n"
_S_NO_MOD = _S_HEAD + _S_LOOP + "    shares.append(left)\n    return shares\n"
_S_PLUS = _S_HEAD + "        left = left + r\n    shares.append(left % M)\n    return shares\n"
_S_RESTART = _S_HEAD + "        left = secret - r\n    shares.append(left % M)\n    return shares\n"
_S_SECRET = _S_HEAD + _S_LOOP + "    shares.append(secret % M)\n    return shares\n"
_S_INSIDE = _S_HEAD + _S_LOOP + "        shares.append(left % M)\n    return shares\n"

# exercise 2: reveal
_R_HEAD = ("def reveal(shares, M):\n"
           "    # add all shares up, then take the remainder after dividing by M\n"
           "    total = 0\n"
           "    for s in shares:\n")
_R_START = _R_HEAD + "        total = ...\n    return ...\n"
_R_LOOP = "        total = total + s\n"
_R = _R_HEAD + _R_LOOP + "    return total % M\n"
_R_NO_MOD = _R_HEAD + _R_LOOP + "    return total\n"
_R_LAST = _R_HEAD + "        total = s\n    return total % M\n"
_R_INSIDE = _R_HEAD + _R_LOOP + "        return total % M\n"

# exercise 3: party_totals
_T_HEAD = ("def party_totals(table, M):\n"
           "    # table[i][j] is share j of company i, and party j holds it\n"
           "    totals = []\n"
           "    for j in range(len(table[0])):\n"
           "        total = 0\n"
           "        for row in table:\n")
_T_START = _T_HEAD + "            total = ...\n        totals.append(...)\n    return totals\n"
_T_LOOP = "            total = total + row[j]\n"
_T = _T_HEAD + _T_LOOP + "        totals.append(total % M)\n    return totals\n"
_T_NO_MOD = _T_HEAD + _T_LOOP + "        totals.append(total)\n    return totals\n"
_T_ALL = _T_HEAD + "            total = total + sum(row)\n        totals.append(total % M)\n    return totals\n"
_T_FIRST = _T_HEAD + "            total = total + row[0]\n        totals.append(total % M)\n    return totals\n"
_T_ROWS = ("def party_totals(table, M):\n"
           "    # table[i][j] is share j of company i, and party j holds it\n"
           "    totals = []\n"
           "    for row in table:\n"
           "        total = 0\n"
           "        for j in range(len(row)):\n"
           "            total = total + row[j]\n"
           "        totals.append(total % M)\n"
           "    return totals\n")

EXERCISES = {
    "share": FunctionBox(
        name="share",
        picture=share_picture,
        function="share",
        start=_S_START,
        solution=_S,
        visible=[(4200, [71320, 8513], M)],
        hidden=[(4200, [5000, 90000], M), (70000, [20000], M), (5, [], 10), (0, [3], 10),
                (9999, [99999, 99999, 99999], M), (12, [4, 3], 1000)],
        first=[_DOTS],
        mistakes=[
            Mistake("no-mod", wrong=_S_NO_MOD,
                    en="`{call}` gives {got}, and it should give {want}. Your last share is below 0, and a share is a whole number from 0 to M − 1. Keep the remainder after dividing by `M`, with `%`, as in the worked example.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je laatste share ligt onder 0, en een share is een heel getal van 0 tot M − 1. Houd de rest bij deling door `M` over, met `%`, zoals in het uitgewerkte voorbeeld."),
            Mistake("plus", wrong=_S_PLUS,
                    en="`{call}` gives {got}, and it should give {want}. Your last share adds the random numbers to the secret. Take them off instead, so that all shares together come back to the secret.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je laatste share telt de willekeurige getallen bij het geheim op. Haal ze er juist af, zodat alle shares samen weer op het geheim uitkomen."),
            Mistake("restart", wrong=_S_RESTART,
                    en="`{call}` gives {got}, and it should give {want}. Your last share is the secret minus only the last random number. Take every random number off: each time round the loop, take `r` off what is left so far, `left`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je laatste share is het geheim min alleen het laatste willekeurige getal. Haal elk willekeurig getal eraf: haal elke keer in de loop `r` af van wat er tot nu toe over is, `left`."),
            Mistake("secret", wrong=_S_SECRET,
                    en="`{call}` gives {got}: your last share is the secret itself, and whoever holds it knows the secret. The last share is what is left once the random numbers are taken off, `left`, modulo `M`.",
                    nl="`{call}` geeft {got}: je laatste share is het geheim zelf, en wie die heeft, kent het geheim. De laatste share is wat er over is als de willekeurige getallen eraf zijn, `left`, modulo `M`."),
            Mistake("inside", wrong=_S_INSIDE,
                    en="`{call}` gives {got}, and it should give {want}. After each random number there is an extra share: is the line that adds the last share inside the loop? Put it after the loop, at the level of `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Na elk willekeurig getal staat een extra share: staat de regel die de laatste share toevoegt in de loop? Zet die na de loop, op de hoogte van `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: in the loop, take each random number off `left`; after the loop, add what is left, modulo `M`, to `shares`.",
                      "nl": "Vervang de twee `...`: haal in de loop elk willekeurig getal af van `left`; voeg na de loop wat er over is, modulo `M`, aan `shares` toe."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The shares are the random numbers, in their order, and after them what is left of `secret` once all of them are taken off, modulo `M`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De shares zijn de willekeurige getallen, in hun volgorde, en daarna wat er van `secret` over is als ze er allemaal af zijn, modulo `M`."},
            "almost": {"en": "Right for `share(4200, [71320, 8513], 100000)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `secret`, the `randoms` and the `M` it gets?",
                       "nl": "Goed voor `share(4200, [71320, 8513], 100000)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `secret`, de `randoms` en de `M` die hij krijgt?"},
            "correct": {"en": "Your `share` is right in all {n} cases. The picture walks your shares round the clock.",
                        "nl": "Je `share` klopt in alle {n} gevallen. Het plaatje laat jouw shares de klok rond lopen."},
        },
    ),
    "reveal": FunctionBox(
        name="reveal",
        picture=reveal_picture,
        function="reveal",
        start=_R_START,
        solution=_R,
        visible=[([71320, 8513, 24367], M)],
        hidden=[([5000, 90000, 9200], M), ([20000, 50000], M), ([5], 10), ([3, 7], 10), ([], 10),
                ([99999, 99999, 99999, 10002], M), ([4, 3, 5], 1000)],
        first=[_DOTS],
        mistakes=[
            Mistake("no-mod", wrong=_R_NO_MOD,
                    en="`{call}` gives {got}: the shares added up. That sum can be M or more; the secret is its remainder after dividing by `M`, with `%`.",
                    nl="`{call}` geeft {got}: de shares opgeteld. Die som kan M of meer zijn; het geheim is de rest bij deling door `M`, met `%`."),
            Mistake("last", wrong=_R_LAST,
                    en="`{call}` gives {got}: the last share alone, not all shares added up. Each time round the loop, add the share `s` to the total so far.",
                    nl="`{call}` geeft {got}: alleen de laatste share, niet alle shares opgeteld. Tel elke keer in de loop de share `s` op bij het totaal tot nu toe."),
            Mistake("inside", wrong=_R_INSIDE,
                    en="`{call}` gives {got}: the first share alone. Is your `return` inside the loop? Then the function stops after the first share. Put it after the loop, at the level of `for`.",
                    nl="`{call}` geeft {got}: alleen de eerste share. Staat je `return` in de loop? Dan stopt de functie na de eerste share. Zet hem na de loop, op de hoogte van `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: in the loop, add each share to `total`; after the loop, give back the remainder of `total` after dividing by `M`.",
                      "nl": "Vervang de twee `...`: tel in de loop elke share op bij `total`; geef na de loop de rest van `total` bij deling door `M` terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Add up all shares, then take the remainder after dividing by `M`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Tel alle shares op, en neem dan de rest bij deling door `M`."},
            "almost": {"en": "Right for `reveal([71320, 8513, 24367], 100000)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `shares` and the `M` it gets?",
                       "nl": "Goed voor `reveal([71320, 8513, 24367], 100000)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `shares` en de `M` die hij krijgt?"},
            "correct": {"en": "Your `reveal` is right in all {n} cases. The picture runs it on all shares of a secret, and on all but one.",
                        "nl": "Je `reveal` klopt in alle {n} gevallen. Het plaatje laat hem lopen op alle shares van een geheim, en op alle op één na."},
        },
    ),
    "party-totals": FunctionBox(
        name="party-totals",
        picture=totals_picture,
        function="party_totals",
        start=_T_START,
        solution=_T,
        visible=[(SHARES, M)],
        hidden=[([[1, 2, 3], [4, 5, 6]], 100), ([[1, 2], [3, 4]], 10), ([[7, 8, 9]], 10), ([[9, 9], [9, 9], [9, 9]], 10),
                ([[5, 1, 2, 3], [9, 9, 9, 9], [0, 4, 0, 4]], 1000)],
        first=[_DOTS],
        mistakes=[
            Mistake("rows", wrong=_T_ROWS,
                    en="`{call}` gives {got}: the secrets of the five companies! Your totals add up the shares of one company, a row, and only that company itself holds all of them. Party j holds a column: share j of every company.",
                    nl="`{call}` geeft {got}: de geheimen van de vijf bedrijven! Je totalen tellen de shares van één bedrijf op, een rij, en alleen dat bedrijf zelf heeft die allemaal. Partij j heeft een kolom: share j van elk bedrijf."),
            Mistake("all", wrong=_T_ALL,
                    en="`{call}` gives {got}: every party gets the sum of all secrets. Your code adds whole rows, every share of a company, but a party holds only one share of each company: the one in column j.",
                    nl="`{call}` geeft {got}: elke partij krijgt de som van alle geheimen. Je code telt hele rijen op, elke share van een bedrijf, maar een partij heeft maar één share van elk bedrijf: die in kolom j."),
            Mistake("first", wrong=_T_FIRST,
                    en="`{call}` gives {got}: every party adds up the first share of each company. Party j adds up the share in column j.",
                    nl="`{call}` geeft {got}: elke partij telt de eerste share van elk bedrijf op. Partij j telt de share in kolom j op."),
            Mistake("no-mod", wrong=_T_NO_MOD,
                    en="`{call}` gives {got}, and it should give {want}: the right sums, but a party's total is a share of the sum, so a whole number from 0 to M − 1 as well. Keep the remainder after dividing by `M`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de goede sommen, maar het totaal van een partij is een share van de som, dus ook een heel getal van 0 tot M − 1. Houd de rest bij deling door `M` over."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: in the loop over the rows, add the share in column j to `total`; after it, add `total`, modulo `M`, to `totals`.",
                      "nl": "Vervang de twee `...`: tel in de loop over de rijen de share in kolom j op bij `total`; voeg daarna `total`, modulo `M`, aan `totals` toe."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. For each party j, add up share j of every company, then take the remainder after dividing by `M`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Tel voor elke partij j share j van elk bedrijf op, en neem dan de rest bij deling door `M`."},
            "almost": {"en": "Right for `party_totals(SHARES, 100000)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `table` and the `M` it gets?",
                       "nl": "Goed voor `party_totals(SHARES, 100000)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `table` en de `M` die hij krijgt?"},
            "correct": {"en": "Your `party_totals` is right in all {n} cases. The picture shows what each party sees, and your totals added up.",
                        "nl": "Je `party_totals` klopt in alle {n} gevallen. Het plaatje laat zien wat elke partij ziet, en jouw totalen opgeteld."},
        },
    ),
}

CASES = [
    ("share", _S_START, "empty", None),
    ("share", _S, "correct", None),
    ("share", "def share(secret, randoms, M):\n    return randoms + [(secret - sum(randoms)) % M]\n", "correct", None),
    ("share", _S_HEAD + "        left -= r\n    shares.append(left % M)\n    return shares\n", "correct", None),
    ("share", "def share(secret, randoms, M):\n    randoms.append((secret - sum(randoms)) % M)\n    return randoms\n", "correct", None),
    ("share", _S_NO_MOD, "wrong", "no-mod"),
    ("share", _S_PLUS, "wrong", "plus"),
    ("share", _S_RESTART, "wrong", "restart"),
    ("share", "def share(secret, randoms, M):\n    return randoms + [(secret - randoms[-1]) % M]\n", "wrong", None),
    ("share", _S_SECRET, "wrong", "secret"),
    ("share", "def share(secret, randoms, M):\n    return randoms + [secret]\n", "wrong", "secret"),
    ("share", _S_INSIDE, "wrong", "inside"),
    ("share", _S_HEAD + _S_LOOP + "    shares.append(left % (M - 1))\n    return shares\n", "wrong", None),
    ("reveal", _R_START, "empty", None),
    ("reveal", _R, "correct", None),
    ("reveal", "def reveal(shares, M):\n    return sum(shares) % M\n", "correct", None),
    ("reveal", _R_HEAD + "        total += s\n        total = total % M\n    return total\n", "correct", None),
    ("reveal", _R_NO_MOD, "wrong", "no-mod"),
    ("reveal", "def reveal(shares, M):\n    return sum(shares)\n", "wrong", "no-mod"),
    ("reveal", _R_LAST, "wrong", "last"),
    ("reveal", _R_INSIDE, "wrong", "inside"),
    ("reveal", _R_HEAD + _R_LOOP + "    return total % (M + 1)\n", "wrong", None),
    ("party-totals", _T_START, "empty", None),
    ("party-totals", _T, "correct", None),
    ("party-totals", _T_HEAD + "            total += row[j]\n        totals.append(total % M)\n    return totals\n", "correct", None),
    ("party-totals", _T_ROWS, "wrong", "rows"),
    ("party-totals", _T_ALL, "wrong", "all"),
    ("party-totals", _T_FIRST, "wrong", "first"),
    ("party-totals", _T_NO_MOD, "wrong", "no-mod"),
    ("party-totals", _T_HEAD + _T_LOOP + "            totals.append(total % M)\n    return totals\n", "wrong", None),
]
