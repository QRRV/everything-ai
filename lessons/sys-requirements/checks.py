# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-requirements: reliable, scalable, maintainable, adaptable. The four requirements a system
in production is judged on, and how a wish becomes a number with a target: a service level indicator (SLI), a service
level objective (SLO) and the error budget it leaves. The learner measures an SLI and works out what is left of an
error budget; the demo follows the spam service through one busy day with one to six servers.

The answer times, the outages and the messages of the busy day are invented by the course; the lesson's texts say
which facts come from Google's SRE book, Huyen and the companies they name."""
import html
import math

from everything_ai import check, yours
from everything_ai.check import FunctionBox, Mistake

LESSON = "sys-requirements"
LANGS = ("en", "nl")

REQUIREMENTS = ("reliability", "scalability", "maintainability", "adaptability")

# Exercise 1: the seconds 20 messages of a busy morning waited for their score, and the limit of the SLI.
TIMES = [2, 3, 2, 4, 75, 3, 2, 5, 3, 2, 60, 4, 3, 2, 140, 3, 2, 4, 3, 2]
LIMIT = 60          # seconds
SLO_SHARE = 0.99    # the SLO of the spam service: 99% of messages within the limit

# Exercise 2 and the worked example: the outages of one month of 30 days, in minutes, and the SLO for availability.
OUTAGES = [12, 5, 20]
AVAILABILITY = 0.999
DAYS = 30
OVERSPENT = [50]    # a month in which one outage takes more than the whole budget


def share_within(times, limit):
    good = 0
    for seconds in times:
        if seconds <= limit:
            good = good + 1
    return good / len(times)


def minutes_left(availability, days, outages):
    budget = (1 - availability) * days * 24 * 60
    for minutes in outages:
        budget = budget - minutes
    return budget


MINUTES = DAYS * 24 * 60                                     # 43,200 minutes in 30 days
DOWN = sum(OUTAGES)                                          # 37 minutes down
UP_SHARE = (MINUTES - DOWN) / MINUTES                        # 0.99914
BUDGET = (1 - AVAILABILITY) * MINUTES                        # 43.2 minutes
LEFT = minutes_left(AVAILABILITY, DAYS, OUTAGES)             # 6.2 minutes
GOOD = sum(1 for t in TIMES if t <= LIMIT)                   # 18 of 20
SHARE = share_within(TIMES, LIMIT)                           # 0.9

# The picture of the idea: what each SLO allows in 30 days (Google's SRE book, appendix A, gives the same per month).
LADDER = (0.99, 0.995, 0.999, 0.9995, 0.9999)


def allowed_minutes(availability: float, days: int = DAYS) -> float:
    return (1 - availability) * days * 24 * 60


# ======== the busy day of the demo ========================================================================================
# The spam service now scores the mail of many cities. Messages a minute over one day: quiet at night, a peak at 9:00,
# and at 17:00 a storm warning, after which 1,400 messages a minute come in for 20 minutes. Each server scores 300.

PER_SERVER = 300          # messages a minute one server scores
STORM = (17 * 60, 17 * 60 + 20)
STORM_RATE = 1400
WAIT_LIMIT = 1.0          # minutes: the SLO of the demo is 99% of messages within 1 minute
SERVERS = (1, 2, 3, 4, 5, 6)


def arrivals() -> list:
    """Messages a minute for each of the 1,440 minutes of the day."""
    out = []
    for m in range(24 * 60):
        h = m / 60
        if h < 6:
            a = 60
        elif h < 9:
            a = 60 + (h - 6) / 3 * 440        # up to 500 at 9:00
        elif h < 12:
            a = 500 - (h - 9) / 3 * 150       # 350 at noon
        elif m < STORM[0]:
            a = 350
        elif m < STORM[1]:
            a = STORM_RATE
        elif h < 22:
            a = 350 - (h - 17) / 5 * 250
        else:
            a = 100 - (h - 22) / 2 * 40
        out.append(round(a))
    return out


ARRIVALS = arrivals()
TOTAL = sum(ARRIVALS)                       # 351,757 messages in the day
AVERAGE = TOTAL / len(ARRIVALS)             # 244 messages a minute


def run(servers: int) -> dict:
    """The day with this many servers: for each minute, the wait of a message that comes in then (the queue in front
    of it divided by what the servers score in a minute), and the share of the day's messages within the limit."""
    cap = servers * PER_SERVER
    queue = 0
    waits = []
    good = 0
    for a in ARRIVALS:
        wait = queue / cap
        waits.append(wait)
        if wait <= WAIT_LIMIT:
            good = good + a
        queue = max(0, queue + a - cap)
    late_minutes = [m for m, w in enumerate(waits) if w > WAIT_LIMIT]
    return {"servers": servers, "cap": cap, "waits": waits, "share": good / TOTAL, "longest": max(waits),
            "late_from": late_minutes[0] if late_minutes else None,
            "late_until": late_minutes[-1] + 1 if late_minutes else None}


RUNS = {s: run(s) for s in SERVERS}
FIRST_MET = min(s for s in SERVERS if RUNS[s]["share"] >= SLO_SHARE)    # 5 servers


# ======== words and drawing ===============================================================================================

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem;max-width:var(--ea-measure)"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem;max-width:var(--ea-measure)"
_SVG = "width:100%;max-width:330px;height:auto;font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums"


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma and a thousands dot in Dutch, and a real minus sign."""
    out = f"{x:,.{digits}f}"
    out = out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out
    return out.replace("-", "−")


def short(x: float, digits: int, lang: str) -> str:
    """A number with at most `digits` decimals, without zeros at its end: 6.2, 52.56, 0.9."""
    out = f"{round(x, digits) + 0.0:.{digits}f}".rstrip("0").rstrip(".") if digits else f"{round(x):d}"
    out = "0" if out in ("-0", "") else out
    return (out.replace(".", ",") if lang == "nl" else out).replace("-", "−")


def clock(minute: int) -> str:
    return f"{minute // 60}:{minute % 60:02d}"


def _marked(text_: str) -> str:
    """Escaped text in which `name` is set as code and *word* in italics."""
    out = []
    for k, part in enumerate(html.escape(text_).split("`")):
        if k % 2:
            out.append(f"<code>{part}</code>")
        else:
            out.append("".join(f"<i>{p}</i>" if j % 2 else p for j, p in enumerate(part.split("*"))))
    return "".join(out)


def _note(text_: str, style: str = _UI) -> str:
    return f'<div style="{style}">{_marked(text_)}</div>'


def _t(x: float, y: float, text_: str, size: float = 12.5, weight: int = 400, fill: str = "var(--ea-ink)",
       anchor: str = "start", halo: bool = False) -> str:
    rim = ";paint-order:stroke;stroke:var(--ea-page);stroke-width:4px;stroke-linejoin:round" if halo else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" style="font-size:{size}px;font-weight:{weight};'
            f'fill:{fill}{rim}">{html.escape(text_)}</text>')


def _line(x1: float, y1: float, x2: float, y2: float, ink: str = "var(--ea-line)", width: float = 1.0,
          dash: str = "") -> str:
    dashed = f";stroke-dasharray:{dash}" if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'style="stroke:{ink};stroke-width:{width}{dashed}"/>')


def _rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "") -> str:
    edge = f";stroke:{stroke};stroke-width:1" if stroke else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" style="fill:{fill}{edge}"/>'


def _svg(height: float, alt: str, parts: list) -> str:
    return (f'<svg viewBox="0 0 330 {height:.0f}" role="img" aria-label="{html.escape(alt)}" style="{_SVG}">'
            f'{"".join(parts)}</svg>')


def percent(share: float, digits: int, lang: str) -> str:
    return short(share * 100, digits, lang) + "%"


# ======== the picture of the idea =========================================================================================

IDEA_TEXT = {
    "en": {"title": "Allowed downtime in 30 days, per SLO", "minutes": "{m} minutes", "hours": "{h} hours",
           "foot": "Each extra nine leaves a tenth of the downtime.",
           "alt": "Five bars for an SLO of 99%, 99.5%, 99.9%, 99.95% and 99.99%: in 30 days they allow 7.2 hours, 3.6 hours, 43.2 minutes, 21.6 minutes and 4.32 minutes of downtime."},
    "nl": {"title": "Toegestane downtime in 30 dagen, per SLO", "minutes": "{m} minuten", "hours": "{h} uur",
           "foot": "Elke extra negen laat een tiende van de downtime over.",
           "alt": "Vijf balken voor een SLO van 99%, 99,5%, 99,9%, 99,95% en 99,99%: in 30 dagen staan deze SLO's 7,2 uur, 3,6 uur, 43,2 minuten, 21,6 minuten en 4,32 minuten downtime toe."},
}


def duration(minutes: float, lang: str) -> str:
    w = IDEA_TEXT[lang]
    if minutes >= 120:
        return w["hours"].format(h=short(minutes / 60, 2, lang))
    return w["minutes"].format(m=short(minutes, 2, lang))


def idea_picture(lang: str) -> str:
    """What each SLO of the ladder allows in 30 days, as bars on one scale."""
    w = IDEA_TEXT[lang]
    parts = [_t(0, 14, w["title"], 12.5, 600)]
    X0, X1 = 58.0, 230.0
    most = allowed_minutes(LADDER[0])
    y = 32.0
    for a in LADDER:
        m = allowed_minutes(a)
        width = (X1 - X0) * m / most
        parts.append(_t(X0 - 8, y + 10, percent(a, 2, lang), 11.5, 600, "var(--ea-ink)", "end"))
        parts.append(_rect(X0, y, max(width, 1.5), 13, "var(--ea-loss)"))
        parts.append(_t(X0 + max(width, 1.5) + 6, y + 10, duration(m, lang), 11, 400, "var(--ea-ink-2)"))
        y += 24
    parts.append(_t(0, y + 8, w["foot"], 11, 400, "var(--ea-ink-2)"))
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(w["alt"])}" style="margin:0.25rem 0 0.75rem;'
            f'max-width:min(100%,330px)"><svg viewBox="0 0 330 {y + 14:.0f}" aria-hidden="true" style="{_SVG}">'
            f'{"".join(parts)}</svg></figure>')


# ======== the pictures of the learner's work ==============================================================================

PICTURE_SECONDS = 2   # what the learner's function gets for all the calls of one picture together
_STOPPED = object()   # what _call gives for a function that stopped with an error


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _call(make_):
    """make_() within a time limit; _STOPPED when it stops with an error or runs out of time."""
    try:
        return check.with_time_limit(make_, PICTURE_SECONDS)
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _real(v, low: float, high: float):
    """v as a finite number from low to high, or None: never from text, True and False, or a nan."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < low or v > high:
        return None
    return float(v)


def times_svg(times: list, limit: float, mine: float, lang: str, alt: str) -> str:
    """A bar per message for the seconds it waited, a dashed line at the limit, the late ones in the color of
    mistakes, and the share the learner's function gives."""
    w = _YOURS_TEXT[lang]["share"]
    L_, R_, T_, B_ = 34.0, 322.0, 26.0, 116.0
    top = 150.0
    n = len(times)
    slot = (R_ - L_) / n

    def py(v):
        return B_ - (B_ - T_) * min(v, top) / top

    parts = []
    for v in (0, 50, 100, 150):
        parts.append(_line(L_, py(v), R_, py(v)))
        parts.append(_t(L_ - 6, py(v) + 4, number(v, 0, lang), 10, 400, "var(--ea-ink-3)", "end"))
    parts.append(_t(L_ - 6, 12, w["seconds"], 10, 400, "var(--ea-ink-3)", "start"))
    for i, v in enumerate(times):
        late = v > limit
        x = L_ + i * slot + 2
        parts.append(_rect(x, py(v), slot - 4, B_ - py(v), "var(--ea-loss)" if late else "var(--ea-ink-3)"))
    parts.append(_line(L_, py(limit), R_, py(limit), "var(--ea-ink)", 1.4, "5 3"))
    parts.append(_t(L_ + 5 * slot + 2, py(limit) - 5, w["limit"].format(s=number(limit, 0, lang)), 11, 600, "var(--ea-ink)",
                    "start", halo=True))
    good = sum(1 for v in times if v <= limit)
    parts.append(_t(L_, B_ + 18, w["good"].format(g=good, n=n), 11.5, 600, "var(--ea-ink)"))
    parts.append(_t(L_, B_ + 34, w["mine"].format(v=short(mine, 3, lang)), 11.5, 600, "var(--ea-ink)"))
    return _svg(B_ + 42, alt, parts)


def budget_svg(rows: list, lang: str, alt: str) -> str:
    """A bar per month for the error budget, the outages taken from it in the color of mistakes, and what the
    learner's function says is left, marked under the bar."""
    w = _YOURS_TEXT[lang]["budget"]
    L_, R_ = 4.0, 300.0
    most = max(max(BUDGET, sum(outages)) for outages, _ in rows)
    scale = (R_ - L_) / most
    parts = []
    y = 4.0
    for outages, mine in rows:
        parts.append(_t(L_, y + 12, w["month"].format(o=" + ".join(number(o, 0, lang) for o in outages)), 11.5, 600))
        by = y + 20
        parts.append(_rect(L_, by, BUDGET * scale, 16, "var(--ea-page)", "var(--ea-ink)"))
        x = L_
        for o in outages:
            parts.append(_rect(x, by, o * scale - 1, 16, "var(--ea-loss)"))
            x += o * scale
        parts.append(_line(L_ + BUDGET * scale, by - 4, L_ + BUDGET * scale, by + 20, "var(--ea-ink)", 1.4))
        parts.append(_t(L_ + BUDGET * scale, by - 6, w["budget"].format(b=short(BUDGET, 1, lang)), 10.5, 400,
                        "var(--ea-ink-2)", "end" if BUDGET * scale > 200 else "middle", halo=True))
        parts.append(_t(L_, by + 32, w["mine"].format(v=short(mine, 2, lang)), 11.5, 600,
                        "var(--ea-loss-text)" if mine < 0 else "var(--ea-ink)"))
        y = by + 44
    return _svg(y, alt, parts)


_YOURS_TEXT = {
    "en": {"share": {"title": "Your share_within on the busy morning",
                     "intro": "Your `share_within(C.TIMES, 60)`: a bar for each of the 20 messages, as high as the seconds it waited for its score. The messages over the limit were not on time.",
                     "seconds": "seconds", "limit": "limit: {s} seconds", "good": "within the limit: {g} of {n} messages",
                     "mine": "your function: {v}",
                     "alt": "Twenty bars for the seconds each message waited, most of them a few seconds, two above the dashed line at 60 seconds, and the share your function gives.",
                     "slo": "The SLO of the spam service asks for 0.99: {v} is below it.",
                     "met": "The SLO of the spam service asks for 0.99, and {v} meets it.",
                     "none": "Your `share_within` stops with an error, takes too long, or does not give a share from 0 to 1, so nothing can be drawn yet."},
           "budget": {"title": "Your minutes_left for two months",
                      "intro": "Your `minutes_left(0.999, 30, ...)` for the outages of `C.OUTAGES`, and for a month with one outage of 50 minutes. The box is the error budget; the outages fill it from the left.",
                      "month": "outages: {o} minutes", "budget": "budget {b}", "mine": "your function: {v} minutes left",
                      "alt": "Two bars of the error budget of 43.2 minutes, with the outages of each month taken from it, and the minutes your function says are left under each bar.",
                      "over": "In the second month, the outage takes more than the whole budget: the minutes left are negative, and the SLO is broken.",
                      "none": "Your `minutes_left` stops with an error, takes too long, or does not give a number of minutes, so nothing can be drawn yet."}},
    "nl": {"share": {"title": "Je share_within op de drukke ochtend",
                     "intro": "Je `share_within(C.TIMES, 60)`: een balk voor elk van de 20 berichten, zo hoog als de seconden die het bericht op de score wachtte. De berichten boven de grens kwamen niet op tijd.",
                     "seconds": "seconden", "limit": "grens: {s} seconden", "good": "binnen de grens: {g} van {n} berichten",
                     "mine": "je functie: {v}",
                     "alt": "Twintig balken voor de seconden die elk bericht wachtte, de meeste een paar seconden, twee boven de lijn van streepjes bij 60 seconden, en het deel dat je functie geeft.",
                     "slo": "De SLO van de spamdienst vraagt 0,99: {v} ligt eronder.",
                     "met": "De SLO van de spamdienst vraagt 0,99, en {v} haalt de SLO.",
                     "none": "Je `share_within` stopt met een fout, duurt te lang, of geeft geen deel van 0 tot 1, dus is er nog geen afbeelding."},
           "budget": {"title": "Je minutes_left voor twee maanden",
                      "intro": "Je `minutes_left(0.999, 30, ...)` voor de storingen van `C.OUTAGES`, en voor een maand met één storing van 50 minuten. Het vak is het error budget; de storingen vullen het vak vanaf links.",
                      "month": "storingen: {o} minuten", "budget": "budget {b}", "mine": "je functie: nog {v} minuten",
                      "alt": "Twee balken van het error budget van 43,2 minuten, met de storingen van elke maand eraf, en onder elke balk de minuten die er volgens je functie over zijn.",
                      "over": "In de tweede maand duurt de storing langer dan het hele budget: er blijft een negatief aantal minuten over, en de SLO is niet gehaald.",
                      "none": "Je `minutes_left` stopt met een fout, duurt te lang, of geeft geen aantal minuten, dus is er nog geen afbeelding."}},
}


def share_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["share"]
    fn = _fn(seen)
    v = _call(lambda: fn(list(TIMES), LIMIT)) if fn is not None else None
    got = _real(v, 0, 1) if v is not _STOPPED else None
    if fn is None or got is None:
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    svg = times_svg(TIMES, LIMIT, got, lang, w["alt"])
    verdict = (w["met"] if got >= SLO_SHARE else w["slo"]).format(v=short(got, 3, lang))
    return yours.figure(w["title"], _note(w["intro"]) + svg + _note(verdict), alt=w["title"])


def budget_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["budget"]
    fn = _fn(seen)
    got = []
    for outages in (OUTAGES, OVERSPENT):
        v = _call(lambda o=outages: fn(AVAILABILITY, DAYS, list(o))) if fn is not None else None
        got.append(_real(v, -100_000, 100_000) if v is not _STOPPED else None)
    if fn is None or any(g is None for g in got):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    svg = budget_svg([(OUTAGES, got[0]), (OVERSPENT, got[1])], lang, w["alt"])
    return yours.figure(w["title"], _note(w["intro"]) + svg + _note(w["over"]), alt=w["title"])


# ======== the demo: one busy day with more servers ========================================================================

DEMO_TEXT = {
    "en": {"title": "{n} servers: {c} messages a minute", "title1": "1 server: {c} messages a minute",
           "in": "messages a minute", "wait": "wait in minutes", "limit": "1 minute",
           "share": "within 1 minute: {p} of the messages", "met": "SLO of 99%: met", "missed": "SLO of 99%: missed",
           "longest": "longest wait: {m} minutes", "clipped": "up to {m}",
           "alt": "{n} servers: the messages a minute over one day with a dashed line at what the servers score, and under it the wait of a message over the day, with the minutes over the limit of 1 minute in the color of mistakes.",
           "says": {
               1: "**1 server** scores 300 messages a minute, more than the 244 of the average minute. But from 7:39, more messages come in than it can score, and at midnight more than 18,000 messages are still waiting: only 12.7% of the messages get a score within 1 minute.",
               2: "**2 servers** keep up all day, until the storm warning at 17:00: 1,400 messages a minute for 20 minutes. The queue grows to waits of 26.7 minutes, and 12.3% of the day's messages miss the limit.",
               3: "**3 servers** still lose to the storm: the longest wait is 11.1 minutes, and 91.3% of the messages get a score within 1 minute.",
               4: "**4 servers** score 1,200 messages a minute, still less than the 1,400 of the storm. The longest wait is 3.3 minutes, and 94.4% of the messages are within the limit: the SLO of 99% is still missed.",
               5: "**5 servers** score 1,500 messages a minute, more than the storm brings: no message waits, and the SLO is met, on a day whose average minute needs less than 1 server.",
               6: "**6 servers** meet the SLO too, with room for a storm that is a bit bigger. The sixth server costs as much as the others, and on this day it never has a queue to score.",
           }},
    "nl": {"title": "{n} servers: {c} berichten per minuut", "title1": "1 server: {c} berichten per minuut",
           "in": "berichten per minuut", "wait": "wachttijd in minuten", "limit": "1 minuut",
           "share": "binnen 1 minuut: {p} van de berichten", "met": "SLO van 99%: gehaald", "missed": "SLO van 99%: niet gehaald",
           "longest": "langste wachttijd: {m} minuten", "clipped": "tot {m}",
           "alt": "{n} servers: de berichten per minuut over één dag met een lijn van streepjes bij wat de servers scoren, en daaronder de wachttijd van een bericht over de dag, met de minuten boven de grens van 1 minuut in de kleur van fouten.",
           "says": {
               1: "**1 server** scoort 300 berichten per minuut, meer dan de 244 van de gemiddelde minuut. Toch komen er vanaf 7:39 meer berichten binnen dan de server kan scoren, en om middernacht wachten er nog meer dan 18.000 berichten: slechts 12,7% van de berichten krijgt binnen 1 minuut een score.",
               2: "**2 servers** houden het de hele dag bij, tot de stormwaarschuwing om 17:00: 20 minuten lang 1.400 berichten per minuut. De wachtrij groeit tot wachttijden van 26,7 minuten, en 12,3% van de berichten van de dag haalt de grens niet.",
               3: "**3 servers** houden de storm nog steeds niet bij: de langste wachttijd is 11,1 minuten, en 91,3% van de berichten krijgt binnen 1 minuut een score.",
               4: "**4 servers** scoren 1.200 berichten per minuut, nog steeds minder dan de 1.400 van de storm. De langste wachttijd is 3,3 minuten, en 94,4% van de berichten blijft binnen de grens: de SLO van 99% is nog steeds niet gehaald.",
               5: "**5 servers** scoren 1.500 berichten per minuut, meer dan de storm brengt: geen bericht wacht, en de SLO is gehaald, op een dag waarvan de gemiddelde minuut minder dan 1 server nodig heeft.",
               6: "**6 servers** halen de SLO ook, met ruimte voor een storm die iets groter is. De zesde server kost evenveel als de andere, en heeft op deze dag nooit een wachtrij om te scoren.",
           }},
}


def frame(servers: int, lang: str) -> tuple[str, str]:
    """The busy day with this many servers: the messages a minute and what the servers score, the wait of a message
    over the day, and the share of messages within the limit."""
    w = DEMO_TEXT[lang]
    servers = max(SERVERS[0], min(SERVERS[-1], int(servers)))
    r = RUNS[servers]
    X0, X1 = 34.0, 322.0
    n = len(ARRIVALS)

    def px(m):
        return X0 + (X1 - X0) * m / n

    title = (w["title1"] if servers == 1 else w["title"]).format(n=servers, c=number(r["cap"], 0, lang))
    parts = [_t(0, 14, title, 12.5, 600)]
    # the messages a minute, and what the servers score
    T1, B1, TOP1 = 34.0, 120.0, 1800.0

    def py1(v):
        return B1 - (B1 - T1) * min(v, TOP1) / TOP1

    parts.append(_t(X0 - 6, T1 - 8, w["in"], 10, 400, "var(--ea-ink-3)", "start"))
    for v in (0, 600, 1200, 1800):
        parts.append(_line(X0, py1(v), X1, py1(v)))
        parts.append(_t(X0 - 6, py1(v) + 4, number(v, 0, lang), 10, 400, "var(--ea-ink-3)", "end"))
    points = [f"{px(0):.1f},{B1:.1f}"]
    for m in range(0, n, 5):
        points.append(f"{px(m):.1f},{py1(max(ARRIVALS[m:m + 5])):.1f}")
    points.append(f"{px(n):.1f},{py1(ARRIVALS[-1]):.1f}")
    points.append(f"{px(n):.1f},{B1:.1f}")
    parts.append(f'<polygon points="{" ".join(points)}" style="fill:var(--ea-line);stroke:var(--ea-ink-3);stroke-width:1"/>')
    parts.append(_line(X0, py1(r["cap"]), X1, py1(r["cap"]), "var(--ea-ink)", 1.4, "5 3"))
    # the wait of a message over the day
    T2, B2, TOP2 = 150.0, 214.0, 10.0

    def py2(v):
        return B2 - (B2 - T2) * min(v, TOP2) / TOP2

    parts.append(_t(X0 - 6, T2 - 8, w["wait"], 10, 400, "var(--ea-ink-3)", "start"))
    for v in (0, 5, 10):
        parts.append(_line(X0, py2(v), X1, py2(v)))
        parts.append(_t(X0 - 6, py2(v) + 4, number(v, 0, lang), 10, 400, "var(--ea-ink-3)", "end"))
    for m0 in range(0, n, 2):
        wv = max(r["waits"][m0:m0 + 2])
        if wv > 0:
            late = wv > WAIT_LIMIT
            parts.append(_rect(px(m0), py2(wv), px(m0 + 2) - px(m0), B2 - py2(wv),
                               "var(--ea-loss)" if late else "var(--ea-ink-3)"))
    parts.append(_line(X0, py2(WAIT_LIMIT), X1, py2(WAIT_LIMIT), "var(--ea-ink)", 1.2, "4 3"))
    parts.append(_t(X1, py2(WAIT_LIMIT) - 4, w["limit"], 10, 400, "var(--ea-ink-2)", "end", halo=True))
    if r["longest"] > TOP2:
        top_at = max(range(n), key=lambda m: r["waits"][m])
        parts.append(_t(min(max(px(top_at), X0 + 30), X1 - 30), T2 - 2, w["clipped"].format(m=short(r["longest"], 1, lang)),
                        10, 600, "var(--ea-loss-text)", "middle", halo=True))
    for h in (0, 6, 12, 18, 24):
        anchor = "start" if h == 0 else ("end" if h == 24 else "middle")
        parts.append(_t(px(h * 60), B2 + 13, clock(h * 60), 10, 400, "var(--ea-ink-3)", anchor))
    met = r["share"] >= SLO_SHARE
    parts.append(_t(0, B2 + 32, w["share"].format(p=percent(r["share"], 1, lang)), 11.5, 600, "var(--ea-ink)"))
    parts.append(_t(0, B2 + 48, w["met"] if met else w["missed"], 11.5, 600,
                    "var(--ea-ink)" if met else "var(--ea-loss-text)"))
    parts.append(_t(0, B2 + 64, w["longest"].format(m=short(r["longest"], 1, lang)), 11, 400, "var(--ea-ink-2)"))
    return _svg(B2 + 70, w["alt"].format(n=servers), parts), w["says"][servers]


# ======== the exercises ===================================================================================================

def _dots_left(got, text_) -> bool:
    """A `...` of the start code left in the box: an Ellipsis in the code, not three dots inside a string."""
    import ast
    try:
        tree = ast.parse(text_)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Constant) and node.value is Ellipsis for node in ast.walk(tree))


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. Can you replace it with your own code?",
                nl="Er staat nog een `...` in je code. Kun je de `...` vervangen door je eigen code?")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then begins with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


class _Listed(list):
    """A list of the lesson that the check shows by its name: C.TIMES."""

    def __init__(self, items, name: str):
        super().__init__(items)
        self._name = name

    def __repr__(self) -> str:
        return self._name

    def __deepcopy__(self, memo):
        return list(self)


TIMES_CASE = _Listed(TIMES, "C.TIMES")
OUTAGES_CASE = _Listed(OUTAGES, "C.OUTAGES")


def _shown(digits: int):
    """How {got} and {want} show a number in the check's text: at most `digits` decimals, a decimal comma in Dutch."""
    def show(v):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            return check._as_value(v)
        return {lang: short(v, digits, lang) for lang in LANGS}
    return show


# exercise 1: share_within
_S_HEAD = ("def share_within(times, limit):\n"
           "    good = 0\n"
           "    for seconds in times:\n"
           "        if seconds <= limit:\n"
           "            good = good + 1\n")


def _s(back: str) -> str:
    return _S_HEAD + f"    return {back}\n"


_S_START = _s("...")
_S = _s("good / len(times)")
_S_COUNT = _s("good")
_S_LIMIT = _s("good / limit")
_S_TWENTY = _s("good / 20")

# exercise 2: minutes_left
_M_HEAD = "def minutes_left(availability, days, outages):\n"
_M_END = ("    for minutes in outages:\n"
          "        budget = budget - minutes\n"
          "    return budget\n")


def _m(share: str) -> str:
    return _M_HEAD + f"    budget = ({share}) * days * 24 * 60\n" + _M_END


_M_START = _M_HEAD + "    budget = ...\n" + _M_END
_M = _m("1 - availability")
_M_UP = _m("availability")
_M_PERCENT = _m("100 - availability")
_M_FIXED = _m("0.001")
_M_DAYS = _M_HEAD + "    budget = (1 - availability) * days\n" + _M_END

EXERCISES = {
    "share": FunctionBox(
        name="share",
        function="share_within",
        start=_S_START,
        solution=_S,
        visible=[(TIMES_CASE, LIMIT)],
        hidden=[([3, 61], 60), ([60, 60, 61], 60), ([10], 5), ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5), (list(TIMES), 2)],
        first=[_DOTS],
        picture=share_picture,
        show=_shown(3),
        mistakes=[
            Mistake("count", wrong=_S_COUNT,
                    en="`{call}` gives {got}, and it should give {want}: that is the number of messages within the limit, not their share. What do you divide it by to get a number from 0 to 1?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: dat is het aantal berichten binnen de grens, niet het deel. Waardoor deel je het om een getal van 0 tot 1 te krijgen?"),
            Mistake("limit", wrong=_S_LIMIT,
                    en="`{call}` gives {got}, and it should give {want}: your function divides by the limit in seconds. The share is out of how many messages?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie deelt door de grens in seconden. Van hoeveel berichten is het een deel?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the share of messages within the limit, worked out from `good` and `times`.",
                      "nl": "Vervang de `...` door het deel van de berichten binnen de grens, uitgerekend uit `good` en `times`."},
            "arguments": {"en": "The check calls `{call}`, with a list of seconds and a limit, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(times, limit)`?",
                          "nl": "De check roept `{call}` aan, met een lijst seconden en een grens, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(times, limit)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function divide the messages within the limit by all the messages?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Deelt je functie de berichten binnen de grens door alle berichten?"},
            "almost": {"en": "Right for the 20 messages of `C.TIMES`. But `{call}` gives {got}, and it should give {want}. What if a list has more or fewer messages?",
                       "nl": "Goed voor de 20 berichten van `C.TIMES`. Maar `{call}` geeft {got}, en het moet {want} zijn. Wat als een lijst meer of minder berichten heeft?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with a `return`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met een `return`?"},
            "correct": {"en": "Right in all {n} cases: 18 of the 20 messages are within 60 seconds, a share of 0.9. How many of the 20 would have to be within the limit to meet the SLO of 0.99?",
                        "nl": "Goed in alle {n} gevallen: 18 van de 20 berichten blijven binnen 60 seconden, een deel van 0,9. Hoeveel van de 20 zouden binnen de grens moeten blijven om de SLO van 0,99 te halen?"},
        },
    ),
    "budget": FunctionBox(
        name="budget",
        function="minutes_left",
        start=_M_START,
        solution=_M,
        visible=[(AVAILABILITY, DAYS, OUTAGES_CASE)],
        hidden=[(0.9999, 365, []), (0.99, 7, [60]), (0.999, 30, list(OVERSPENT)), (0.995, 30, [100, 100, 16]),
                (0.9995, 30, [1, 2, 3])],
        first=[_DOTS],
        picture=budget_picture,
        show=_shown(2),
        mistakes=[
            Mistake("uptime", wrong=_M_UP,
                    en="`{call}` gives {got}, and it should give {want}: your budget is the minutes the service should be up, almost the whole period. Which share of the time may the service be down?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je budget is het aantal minuten dat de dienst moet werken, bijna de hele periode. Welk deel van de tijd mag de dienst plat liggen?"),
            Mistake("percent", wrong=_M_PERCENT,
                    en="`{call}` gives {got}, and it should give {want}: `availability` is a share such as `0.999`, not a percentage. In a share, what number stands for the whole time?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `availability` is een deel zoals `0.999`, geen percentage. Welk getal staat in een deel voor de hele tijd?"),
            Mistake("days", wrong=_M_DAYS,
                    en="`{call}` gives {got}, and it should give {want}: your budget is in days, and the outages are in minutes. How many minutes does a day have?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je budget staat in dagen, en de storingen staan in minuten. Hoeveel minuten heeft een dag?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the budget in minutes at the start, worked out from `availability` and `days`.",
                      "nl": "Vervang de `...` door het budget in minuten aan het begin, uitgerekend uit `availability` en `days`."},
            "arguments": {"en": "The check calls `{call}`, with an SLO, a number of days and a list of outages, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(availability, days, outages)`?",
                          "nl": "De check roept `{call}` aan, met een SLO, een aantal dagen en een lijst storingen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(availability, days, outages)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your budget start as the share of the time the service may be down, times all the minutes of the period?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Begint je budget als het deel van de tijd dat de dienst plat mag liggen, keer alle minuten van de periode?"},
            "almost": {"en": "Right for the month of the worked example. But `{call}` gives {got}, and it should give {want}. Does your budget change when `availability` or `days` changes?",
                       "nl": "Goed voor de maand van het uitgewerkte voorbeeld. Maar `{call}` geeft {got}, en het moet {want} zijn. Verandert je budget als `availability` of `days` verandert?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return budget`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return budget`?"},
            "correct": {"en": "Right in all {n} cases: the budget of the month is 43.2 minutes, and 6.2 are left. In the picture, how many minutes too many was the service down in the month with one outage of 50 minutes?",
                        "nl": "Goed in alle {n} gevallen: het budget van de maand is 43,2 minuten, en er zijn er nog 6,2 over. Hoeveel minuten te veel lag de dienst in de afbeelding plat in de maand met één storing van 50 minuten?"},
        },
    ),
}

# The lesson is done when these are correct. "which" and "slo" are the choice questions after the demo.
REQUIRED = ["share", "budget", "which", "slo"]

CASES = [
    ("share", _S_START, "empty", None),
    ("share", _S, "correct", None),
    ("share", _s("good / ..."), "wrong", "dots-left"),
    ("share", _S_COUNT, "wrong", "count"),
    ("share", _S_LIMIT, "wrong", "limit"),
    ("share", _S_TWENTY, "almost", None),
    ("share", _s("good / len(times) * 1.0"), "correct", None),
    ("share", _s("1 - (len(times) - good) / len(times)"), "correct", None),
    ("share", _S.replace("    return good / len(times)\n", "    print(good / len(times))\n"), "wrong", "prints"),
    ("budget", _M_START, "empty", None),
    ("budget", _M, "correct", None),
    ("budget", _m("1 - ..."), "wrong", "dots-left"),
    ("budget", _M_UP, "wrong", "uptime"),
    ("budget", _M_PERCENT, "wrong", "percent"),
    ("budget", _M_FIXED, "almost", None),
    ("budget", _M_DAYS, "wrong", "days"),
    ("budget", _M_HEAD + "    budget = 0.001 * 43200\n" + _M_END, "almost", None),
    ("budget", _m("-availability + 1"), "correct", None),
    ("budget", _m("(100 - availability * 100) / 100"), "correct", None),
]
