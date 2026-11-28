# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-deployment-patterns: four ways to put a model to use. A made-up bike-share app with 10,000
members, whose batch makes a suggestion for each of them at 2 at night, and one simulated day of the app being opened;
a stream of rentals at one station of the same company; four patterns to sort applications into; and how often the
batch runs, against one model call for every time the app is opened.

The day and the stream are simulations with a fixed seed, made by the course: the lesson says so where it gives their
numbers."""
import html
import math
import random
from functools import cache

from everything_ai import check, yours
from everything_ai.check import FunctionBox, Mistake, Sort

LESSON = "sys-deployment-patterns"
LANGS = ("en", "nl")

SEED = 2026
MEMBERS = 10_000      # members before the night: the batch makes a suggestion for each of them
NEW = 60              # members who join during the day and open the app at once
BATCH_HOUR = 2.0      # the batch runs at 2 at night
# (kind of member, share of the members, chance to open the app on the day)
KINDS = [("daily", 0.09, 0.8), ("weekly", 0.22, 0.18), ("rare", 0.69, 0.02)]
WINDOW = 15           # minutes: the stream looks at the rentals of the last quarter of an hour
ALERT = 10            # rentals in the window from which the rule sends a truck to the station
BURST = 1050          # 17:30, a minute of the stream with a rental in it
CROWD = 1038          # 17:18, when a crowd at the station begins to rent bikes


# ======== the simulated day of the app ===================================================================================

def _clip(h: float) -> float:
    return min(max(h, 5.0), 23.99)


@cache
def _day() -> tuple:
    """The opens of one day, as (hour, member) in order of time, and the hour each new member joined."""
    rng = random.Random(SEED)
    opens, joined = [], {}
    cut1 = int(MEMBERS * KINDS[0][1])
    cut2 = cut1 + int(MEMBERS * KINDS[1][1])
    for m in range(1, MEMBERS + 1):
        kind = 0 if m <= cut1 else (1 if m <= cut2 else 2)
        if rng.random() >= KINDS[kind][2]:
            continue
        if kind == 0:   # to work and back, and now and then once more
            hours = [rng.gauss(8.0, 0.7), rng.gauss(17.5, 1.0)]
            if rng.random() < 0.3:
                hours.append(rng.uniform(10, 22))
        else:
            n = 2 if rng.random() < 0.2 else 1
            hours = [rng.gauss(12.5, 2.0) if rng.random() < 0.4 else rng.gauss(18.0, 2.5) for _ in range(n)]
        opens += [(round(_clip(h), 2), m) for h in hours]
    for j in range(NEW):
        m = MEMBERS + 1 + j
        h = round(rng.uniform(8, 22), 2)
        joined[m] = h
        opens.append((h, m))
        if rng.random() < 0.4:
            opens.append((round(_clip(h + rng.uniform(0.5, 5)), 2), m))
    opens.sort()
    return tuple(opens), dict(joined)


class _Named(list):
    """A list from the lesson's cells: the check's call shows it by its name, and the learner's function gets a copy."""

    def __init__(self, name: str, values):
        super().__init__(values)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return list(self)


def batch_members() -> list:
    """The members the batch makes a suggestion for at 2 at night: 1 to 10,000."""
    return list(range(1, MEMBERS + 1))


def opened_members() -> list:
    """The member of every time the app was opened during the day, in order of time, with repeats."""
    return [m for _, m in _day()[0]]


def open_hours() -> list:
    """The hour of every time the app was opened, in the same order: 7.5 is half past seven."""
    return [h for h, _ in _day()[0]]


BATCH = _Named("C.BATCH", batch_members())
OPENED = _Named("C.OPENED", opened_members())


def unused(batch, opened):
    seen = set(opened)
    used = 0
    for user in batch:
        if user in seen:
            used = used + 1
    return len(batch) - used


def schedule(runs_per_day: int) -> dict:
    """The batch run `runs_per_day` times a day, from 2 at night on at equal steps, on the simulated day: the
    suggestions it makes, how many of them a member sees, how old a suggestion is when it is seen, and the opens that
    find no suggestion, since the member joined after the last run."""
    opens, joined = _day()
    runs = sorted((BATCH_HOUR + i * 24 / runs_per_day) % 24 for i in range(runs_per_day))
    made = sum(MEMBERS + sum(1 for h in joined.values() if h < r) for r in runs)
    read, ages, uncovered = set(), [], 0
    for h, m in opens:
        r = max(x for x in runs if x <= h)
        if m > MEMBERS and joined[m] >= r:
            uncovered += 1
            continue
        read.add((r, m))
        ages.append(h - r)
    return {"runs": runs_per_day, "made": made, "used": len(read), "unused": made - len(read),
            "age": sum(ages) / len(ages), "uncovered": uncovered, "opens": len(opens)}


RUNS = [1, 2, 4, 8, 24]   # the stops of the demo's slider


@cache
def facts() -> dict:
    """The numbers the lesson's texts give about the day."""
    opens, joined = _day()
    members = [m for _, m in opens]
    once = schedule(1)
    return {"opens": len(opens), "openers": len(set(members)), "used": once["used"], "unused": once["unused"],
            "share_unused": once["unused"] / MEMBERS, "new": len(joined),
            "new_opens": sum(1 for m in members if m > MEMBERS), "age": once["age"],
            "hourly": schedule(24)}


# ======== the stream of one station =======================================================================================

@cache
def _station() -> tuple:
    """The minute of every rental at station 31 during one day, from a rate per minute that peaks in the morning
    and in the evening, with a crowd at the station from 17:18 to 17:30."""
    rng = random.Random(SEED + 1)
    times = []
    for minute in range(24 * 60):
        h = minute / 60
        rate = (0.003 + 0.25 * math.exp(-((h - 8.2) / 0.8) ** 2) + 0.32 * math.exp(-((h - 17.6) / 1.1) ** 2)
                + 0.06 * math.exp(-((h - 13) / 2.5) ** 2))
        if CROWD <= minute <= BURST:
            rate += 0.7
        for _ in range(4):
            if rng.random() < rate / 4:
                times.append(minute)
    if BURST not in times:
        times.append(BURST)
    return tuple(sorted(times))


def station_times() -> list:
    """The minute after midnight of every rental at station 31 on the day, in order: 480 is 8:00."""
    return list(_station())


TIMES = _Named("C.TIMES", station_times())


def recent(times, now, minutes):
    count = 0
    for t in times:
        if t > now - minutes and t <= now:
            count = count + 1
    return count


@cache
def stream_facts() -> dict:
    times = _station()
    after = [(t, recent(times, t, WINDOW)) for t in times]
    alerts = [t for t, n in after if n >= ALERT]
    return {"rentals": len(times), "at_burst": recent(times, BURST, WINDOW),
            "morning_max": max(n for t, n in after if t < 12 * 60), "first_alert": alerts[0], "last_alert": alerts[-1],
            "alerts": len(set(alerts))}


def clock(minute: int) -> str:
    return f"{minute // 60}:{minute % 60:02d}"


# ======== numbers and small pieces of pictures ============================================================================

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem;max-width:var(--ea-measure)"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem;max-width:var(--ea-measure)"
_SVG = "width:100%;max-width:330px;height:auto;font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums"


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma and a thousands dot in Dutch."""
    out = f"{x:,.{digits}f}"
    return out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out


def _lang(lang) -> str:
    return lang if lang in LANGS else "en"


def _marked(text_: str) -> str:
    """Escaped text in which `name` is set as code."""
    return "".join(f"<code>{part}</code>" if k % 2 else part for k, part in enumerate(html.escape(text_).split("`")))


def _note(text_: str, style: str = _UI) -> str:
    return f'<div style="{style}">{_marked(text_)}</div>'


def _t(x: float, y: float, text_: str, size: float = 12.5, weight: int = 400, fill: str = "var(--ea-ink)",
       anchor: str = "start") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" style="font-size:{size}px;font-weight:{weight};'
            f'fill:{fill}">{html.escape(text_)}</text>')


def _svg(height: float, alt: str, parts: list) -> str:
    return (f'<svg viewBox="0 0 330 {height:.0f}" role="img" aria-label="{html.escape(alt)}" style="{_SVG}">'
            f'{"".join(parts)}</svg>')


# ======== the picture of the idea ========================================================================================

IDEA_TEXT = {
    "en": {"title": "When and where the model makes its prediction",
           "lanes": [("batch", "all members at once, at 2 at night"), ("online", "the moment someone asks"),
                     ("streaming", "after every event, nobody asks"), ("on the device", "on the phone itself, no network needed")],
           "made": "prediction made", "needed": "someone looks", "event": "event", "alert": "alert",
           "alt": "Four timelines over one day. Batch: one block at 2 at night, and later the moments someone looks. "
                  "Online: a prediction made at each moment someone asks. Streaming: a prediction after every event, "
                  "and an alert in the evening. On the device: predictions made on the phone, at each moment."},
    "nl": {"title": "Wanneer en waar het model voorspelt",
           "lanes": [("batch", "alle leden tegelijk, om 2 uur 's nachts"), ("online", "op het moment dat iemand vraagt"),
                     ("streaming", "na elke gebeurtenis, niemand vraagt"), ("op het apparaat", "op de telefoon zelf, zonder netwerk")],
           "made": "voorspelling gemaakt", "needed": "iemand kijkt", "event": "gebeurtenis", "alert": "alarm",
           "alt": "Vier tijdlijnen over één dag. Batch: één blok om 2 uur 's nachts, en later de momenten waarop iemand "
                  "kijkt. Online: een voorspelling op elk moment dat iemand vraagt. Streaming: een voorspelling na elke "
                  "gebeurtenis, en een alarm in de avond. Op het apparaat: voorspellingen op de telefoon, op elk moment."},
}
_LOOKS = [8.1, 12.6, 17.4, 21.0]
_EVENTS = [7.6, 8.0, 8.4, 12.2, 16.9, 17.2, 17.35, 17.5]


def _x(hour: float) -> float:
    return 12 + 306 * hour / 24


def idea_picture(lang: str) -> str:
    lang = _lang(lang)
    w = IDEA_TEXT[lang]
    parts = [_t(0, 15, w["title"], 14, 600)]
    y = 34
    made = lambda x, yy: f'<circle cx="{x:.1f}" cy="{yy:.1f}" r="4" style="fill:var(--ea-ink)"/>'   # noqa: E731
    looks = lambda x, yy: f'<circle cx="{x:.1f}" cy="{yy:.1f}" r="6.5" style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.4"/>'   # noqa: E731
    for k, (name, caption) in enumerate(w["lanes"]):
        line_y = y + 26
        parts.append(_t(0, y + 10, name, 12.5, 600) + _t(330, y + 10, caption, 11.5, 400, "var(--ea-ink-2)", "end"))
        parts.append(f'<line x1="{_x(0):.1f}" y1="{line_y}" x2="{_x(24):.1f}" y2="{line_y}" style="stroke:var(--ea-line);stroke-width:1.2"/>')
        if k == 0:
            parts.append(f'<rect x="{_x(BATCH_HOUR) - 4:.1f}" y="{line_y - 9}" width="8" height="18" rx="2" style="fill:var(--ea-ink)"/>')
            parts += [looks(_x(h), line_y) for h in _LOOKS]
        elif k in (1, 3):
            parts += [looks(_x(h), line_y) + made(_x(h), line_y) for h in _LOOKS]
            if k == 3:   # the phone around the predictions
                parts += [f'<rect x="{_x(h) - 8:.1f}" y="{line_y - 12}" width="16" height="24" rx="4" '
                          f'style="fill:none;stroke:var(--ea-ink-3);stroke-width:1"/>' for h in _LOOKS]
        else:
            for h in _EVENTS:
                parts.append(f'<line x1="{_x(h):.1f}" y1="{line_y - 8}" x2="{_x(h):.1f}" y2="{line_y + 8}" style="stroke:var(--ea-ink-3);stroke-width:1.2"/>')
                parts.append(made(_x(h) + 3, line_y))
            ax, ay = _x(17.5) + 3, line_y - 13
            parts.append(f'<path d="M{ax - 6:.1f},{ay:.1f} L{ax + 6:.1f},{ay:.1f} L{ax:.1f},{ay + 9:.1f} Z" style="fill:var(--ea-ink)"/>')
        y += 50
    for h in (0, 6, 12, 18, 24):
        parts.append(_t(_x(h) + (4 if h == 24 else 0), y + 4, f"{h}:00", 10.5, 400, "var(--ea-ink-3)", "end" if h == 24 else "middle"))
    y += 22
    parts.append(made(8, y) + _t(16, y + 4, w["made"], 11.5, 400, "var(--ea-ink-2)"))
    parts.append(looks(178, y) + _t(189, y + 4, w["needed"], 11.5, 400, "var(--ea-ink-2)"))
    y += 20
    parts.append(f'<line x1="8" y1="{y - 7}" x2="8" y2="{y + 7}" style="stroke:var(--ea-ink-3);stroke-width:1.2"/>'
                 + _t(16, y + 4, w["event"], 11.5, 400, "var(--ea-ink-2)"))
    parts.append(f'<path d="M172,{y - 5} L184,{y - 5} L178,{y + 4} Z" style="fill:var(--ea-ink)"/>'
                 + _t(189, y + 4, w["alert"], 11.5, 400, "var(--ea-ink-2)"))
    return _svg(y + 10, w["alt"], parts)


# ======== the pictures of the learner's work =============================================================================

_STOPPED = object()   # what _call gives for a function that stopped with an error
PICTURE_SECONDS = 3   # what the learner's function gets for all the calls of one picture together


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


def _whole(v, high: int) -> int | None:
    """v as a whole number from 0 to high, or None: never from text or True and False."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v != int(v) or not 0 <= v <= high:
        return None
    return int(v)


YOURS_TEXT = {
    "en": {"unused": {"title": "Your unused on the day of the app",
                      "intro": "The 10,000 suggestions of the night, as your `unused(batch, opened)` counts them.",
                      "used": "seen: {n}", "unused": "never seen: {n} ({p}%)",
                      "new": "The {m} members who joined during the day opened the app {o} times and had no suggestion at all.",
                      "alt": "One bar for the 10,000 suggestions of the night: the part a member saw, and the part your function says nobody saw.",
                      "none": "Your `unused` stops with an error, takes too long, or does not give a whole number from 0 to 10,000, so nothing can be drawn yet."},
           "recent": {"title": "Your recent after every rental at station 31",
                      "intro": "After every rental of the day, your `recent(times, now, 15)` counts the rentals of the last 15 minutes. From {a} rentals on, the rule sends a truck.",
                      "axis": "rentals in the last 15 minutes", "line": "{a}: send a truck",
                      "first": "Your counts send the first truck at {t}, {k} minutes after the evening rush at the station begins, at {c}.",
                      "early": "Your counts send the first truck at {t}, before the evening rush at the station begins, at {c}.",
                      "never": "Your counts never reach {a}, so the rule sends no truck at all.",
                      "alt": "A dot for every rental of the day: the count of your function after it, over the hours of the day, with a line at the count from which a truck is sent.",
                      "none": "Your `recent` stops with an error, takes too long, or does not give a whole number for every rental, so nothing can be drawn yet."}},
    "nl": {"unused": {"title": "Je unused op de dag van de app",
                      "intro": "De 10.000 suggesties van de nacht, geteld door je `unused(batch, opened)`.",
                      "used": "gezien: {n}", "unused": "nooit gezien: {n} ({p}%)",
                      "new": "De {m} leden die overdag lid werden, openden de app {o} keer en hadden helemaal geen suggestie.",
                      "alt": "Eén balk voor de 10.000 suggesties van de nacht: het deel dat een lid zag, en het deel waarvan je functie zegt dat niemand het zag.",
                      "none": "Je `unused` stopt met een fout, duurt te lang, of geeft geen heel getal van 0 tot 10.000, dus is er nog geen afbeelding."},
           "recent": {"title": "Je recent na elke verhuring bij station 31",
                      "intro": "Na elke verhuring van de dag telt je `recent(times, now, 15)` de verhuringen van de laatste 15 minuten. Vanaf {a} verhuringen stuurt de regel een vrachtwagen.",
                      "axis": "verhuringen in de laatste 15 minuten", "line": "{a}: stuur een vrachtwagen",
                      "first": "Met je tellingen gaat de eerste vrachtwagen om {t}, {k} minuten nadat de avondspits bij het station begint, om {c}.",
                      "early": "Met je tellingen gaat de eerste vrachtwagen al om {t}, nog voordat de avondspits bij het station begint, om {c}.",
                      "never": "Je tellingen komen nooit op {a}, dus stuurt de regel helemaal geen vrachtwagen.",
                      "alt": "Een stip voor elke verhuring van de dag: de telling van je functie erna, over de uren van de dag, met een lijn bij de telling waarbij een vrachtwagen gaat.",
                      "none": "Je `recent` stopt met een fout, duurt te lang, of geeft niet voor elke verhuring een heel getal, dus is er nog geen afbeelding."}},
}


def unused_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["unused"]
    fn = _fn(seen)
    got = _call(lambda: fn(list(BATCH), list(OPENED))) if fn is not None else None
    n = _whole(got, MEMBERS) if got is not _STOPPED else None
    if n is None:
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    W = 330.0
    seen_w = W * (MEMBERS - n) / MEMBERS
    parts = [f'<rect x="0" y="4" width="{W:.1f}" height="22" rx="2" style="fill:var(--ea-line)"/>',
             f'<rect x="0" y="4" width="{seen_w:.1f}" height="22" rx="2" style="fill:var(--ea-ink)"/>',
             _t(0, 44, w["used"].format(n=number(MEMBERS - n, 0, lang)), 12.5, 600),
             _t(330, 44, w["unused"].format(n=number(n, 0, lang), p=number(100 * n / MEMBERS, 1, lang)), 12.5, 600,
                "var(--ea-ink-2)", "end")]
    f = facts()
    body = (_note(w["intro"]) + _svg(52, w["alt"], parts)
            + _note(w["new"].format(m=f["new"], o=f["new_opens"])))
    return yours.figure(w["title"], body, alt=w["title"])


def recent_svg(counts: list, lang: str, alt: str) -> str:
    """A dot per rental: the count after it, over the hours of the day, with the line of the rule."""
    w = YOURS_TEXT[lang]["recent"]
    L, R, TOP, H = 26, 324, 22, 150
    top = max(max(counts), ALERT + 2)
    xs = lambda minute: L + (R - L) * minute / (24 * 60)   # noqa: E731
    ys = lambda n: TOP + H - H * n / top                    # noqa: E731
    parts = [_t(L, 12, w["axis"], 11.5, 400, "var(--ea-ink-2)")]
    step = 5 if top <= 25 else (10 if top <= 50 else 20)
    for n in range(0, top + 1, step):
        parts.append(f'<line x1="{L}" y1="{ys(n):.1f}" x2="{R}" y2="{ys(n):.1f}" style="stroke:var(--ea-line);stroke-width:0.8"/>'
                     + _t(L - 5, ys(n) + 4, str(n), 10.5, 400, "var(--ea-ink-3)", "end"))
    parts.append(f'<line x1="{L}" y1="{ys(ALERT):.1f}" x2="{R}" y2="{ys(ALERT):.1f}" '
                 f'style="stroke:var(--ea-ink);stroke-width:1.3;stroke-dasharray:5 3"/>'
                 + _t(L + 4, ys(ALERT) - 5, w["line"].format(a=ALERT), 11, 600))
    for t, n in zip(TIMES, counts):
        over = n >= ALERT
        parts.append(f'<circle cx="{xs(t):.1f}" cy="{ys(n):.1f}" r="{3 if over else 2.2}" '
                     f'style="fill:{"var(--ea-ink)" if over else "var(--ea-ink-3)"}"/>')
    for h in (0, 6, 12, 18, 24):
        parts.append(_t(xs(h * 60), TOP + H + 16, f"{h}:00", 10.5, 400, "var(--ea-ink-3)", "end" if h == 24 else "middle"))
    return _svg(TOP + H + 22, alt, parts)


def recent_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["recent"]
    fn = _fn(seen)
    got = _call(lambda: [fn(list(TIMES), t, WINDOW) for t in TIMES]) if fn is not None else None
    counts = [_whole(v, len(TIMES)) for v in got] if isinstance(got, list) else [None]
    if any(c is None for c in counts):
        return yours.figure(w["title"], _note(w["intro"].format(a=ALERT)) + _note(w["none"], _LOSS_UI), alt=w["title"])
    over = [t for t, n in zip(TIMES, counts) if n >= ALERT]
    if over:
        start = min(t for t in TIMES if t >= CROWD)
        said = (w["first"].format(t=clock(over[0]), k=over[0] - start, c=clock(start)) if over[0] >= start
                else w["early"].format(t=clock(over[0]), c=clock(start)))
    else:
        said = w["never"].format(a=ALERT)
    body = _note(w["intro"].format(a=ALERT)) + recent_svg(counts, lang, w["alt"]) + _note(said)
    return yours.figure(w["title"], body, alt=w["title"])


# ======== the demo: how often the batch runs =============================================================================

DEMO_TEXT = {
    "en": {"title": "The batch {k} a day, against online",
           "times": {1: "once", 2: "twice", 4: "4 times", 8: "8 times", 24: "24 times"},
           "made": "model calls in the day", "age": "how old a suggestion is when seen, on average",
           "batch": "batch", "online": "online", "seen": "{n} seen", "hours": "{h} hours", "zero": "0: made when asked",
           "say": "The batch runs {k} a day and makes {m} suggestions, of which members see {u}: {p}%. A suggestion is "
                  "{h} hours old on average when it is seen, and {x} opens by new members find none. Online makes "
                  "{o} calls, one for each time the app is opened, each while a member waits.",
           "alt": "Two pairs of bars: the model calls of the day for the batch and for online, with the part of the batch "
                  "that a member saw; and the age of a suggestion when it is seen."},
    "nl": {"title": "De batch {k} per dag, tegenover online",
           "times": {1: "één keer", 2: "twee keer", 4: "4 keer", 8: "8 keer", 24: "24 keer"},
           "made": "modelaanroepen op de dag", "age": "hoe oud een suggestie gemiddeld is bij het bekijken",
           "batch": "batch", "online": "online", "seen": "{n} gezien", "hours": "{h} uur", "zero": "0: gemaakt bij de vraag",
           "say": "De batch draait {k} per dag en maakt {m} suggesties, en daarvan zien leden er {u}: {p}%. Een suggestie "
                  "is gemiddeld {h} uur oud als een lid de suggestie ziet, en {x} keer opent een nieuw lid de app zonder "
                  "suggestie. Online doet {o} aanroepen, één voor elke keer dat de app opengaat, en bij elke aanroep "
                  "wacht een lid.",
           "alt": "Twee paar balken: de modelaanroepen van de dag voor de batch en voor online, met het deel van de batch "
                  "dat een lid zag; en de leeftijd van een suggestie als iemand de suggestie ziet."},
}


def frame(runs_per_day: int, lang: str) -> tuple[str, str]:
    lang = _lang(lang)
    w = DEMO_TEXT[lang]
    k = int(runs_per_day)
    s = schedule(k)
    most = schedule(RUNS[-1])["made"]
    W = 200.0
    parts = [_t(0, 15, w["title"].format(k=w["times"][k]), 14, 600), _t(0, 40, w["made"], 12, 600, "var(--ea-ink-2)")]
    y = 50
    for name, total, used in ((w["batch"], s["made"], s["used"]), (w["online"], s["opens"], s["opens"])):
        width = max(2.0, W * total / most)
        parts.append(_t(0, y + 12, name, 12.5))
        parts.append(f'<rect x="62" y="{y + 1}" width="{width:.1f}" height="14" rx="2" style="fill:var(--ea-line)"/>')
        parts.append(f'<rect x="62" y="{y + 1}" width="{max(1.5, W * used / most):.1f}" height="14" rx="2" style="fill:var(--ea-ink)"/>')
        parts.append(_t(62 + width + 5, y + 12, number(total, 0, lang), 12, 600))
        y += 22
    parts.append(_t(62, y + 8, w["seen"].format(n=number(s["used"], 0, lang)), 11, 400, "var(--ea-ink-2)"))
    y += 30
    parts.append(_t(0, y, w["age"], 12, 600, "var(--ea-ink-2)"))
    y += 10
    for name, age in ((w["batch"], s["age"]), (w["online"], 0.0)):
        width = W * age / 24
        parts.append(_t(0, y + 12, name, 12.5))
        if age > 0:
            parts.append(f'<rect x="62" y="{y + 1}" width="{max(2.0, width):.1f}" height="14" rx="2" style="fill:var(--ea-ink-3)"/>')
        label = w["hours"].format(h=number(age, 1, lang)) if age > 0 else w["zero"]
        parts.append(_t(62 + max(2.0, width) + 5, y + 12, label, 12, 600))
        y += 22
    say = w["say"].format(k=w["times"][k], m=number(s["made"], 0, lang), u=number(s["used"], 0, lang),
                          p=number(100 * s["used"] / s["made"], 1, lang), h=number(s["age"], 1, lang), x=s["uncovered"],
                          o=number(s["opens"], 0, lang))
    return _svg(y + 4, w["alt"], parts), say


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

# exercise 1: unused
_U_HEAD = ("def unused(batch, opened):\n"
           "    seen = set(opened)\n"
           "    used = 0\n"
           "    for user in batch:\n")
_U_END = "    return len(batch) - used\n"


def _u(body: str) -> str:
    return _U_HEAD + body + _U_END


_U_START = _u("        ...\n")
_U = _u("        if user in seen:\n            used = used + 1\n")
_U_EVERY = _u("        used = used + 1\n")
_U_REPEATS = ("def unused(batch, opened):\n"
              "    members = set(batch)\n"
              "    used = 0\n"
              "    for user in opened:\n"
              "        if user in members:\n"
              "            used = used + 1\n"
              "    return len(batch) - used\n")
_U_ALL_SEEN = _u("        pass\n").replace("return len(batch) - used", "return len(batch) - len(seen)")
_U_OPENS = _u("        pass\n").replace("return len(batch) - used", "return len(batch) - len(opened)")
_U_SEEN_LOOP = _u("        used = len(seen)\n")      # the same mistakes, made inside the loop
_U_OPENS_LOOP = _u("        used = len(opened)\n")

# a small day: the worked example, and days with nobody, only new members, or only repeats
_SMALL = (["a", "b", "c", "d", "e"], ["b", "d", "f", "b"])

# exercise 2: recent
_R_HEAD = ("def recent(times, now, minutes):\n"
           "    count = 0\n"
           "    for t in times:\n")
_R_END = ("            count = count + 1\n"
          "    return count\n")


def _r(condition: str) -> str:
    return _R_HEAD + f"        if {condition}:\n" + _R_END


_R_START = _R_HEAD + "        ...\n    return count\n"
_R = _r("t > now - minutes and t <= now")
_R_BEFORE = _r("t <= now")
_R_AFTER = _r("t > now - minutes")
_R_FROM = _r("t >= now - minutes and t <= now")
_R_NOT_NOW = _r("t > now - minutes and t < now")

PATTERNS = Sort(
    name="patterns",
    groups={
        "batch": {"en": "Batch", "nl": "Batch"},
        "online": {"en": "Online", "nl": "Online"},
        "streaming": {"en": "Streaming", "nl": "Streaming"},
        "device": {"en": "On the device", "nl": "Op het apparaat"},
    },
    items={
        "trucks": {"en": "**Trucks at night.** For every station, how many bikes it needs at 8 in the morning, so that trucks can move bikes during the night.",
                   "nl": "**Vrachtwagens 's nachts.** Voor elk station hoeveel fietsen er om 8 uur 's ochtends nodig zijn, zodat vrachtwagens 's nachts fietsen kunnen verplaatsen."},
        "email": {"en": "**Monday email.** A tip for each member in the email that goes to all members on Monday morning.",
                  "nl": "**Mail op maandag.** Een tip voor elk lid in de mail die maandagochtend naar alle leden gaat."},
        "price": {"en": "**Price of a ride.** The price a rider sees after typing where the ride goes, before pressing Rent.",
                  "nl": "**Prijs van een rit.** De prijs die een fietser ziet na het intypen van de bestemming, nog voordat de fietser op Huur drukt."},
        "chat": {"en": "**Help desk.** A suggested answer for the help desk, while a rider waits in the chat.",
                 "nl": "**Helpdesk.** Een voorgesteld antwoord voor de helpdesk, terwijl een fietser in de chat wacht."},
        "moving": {"en": "**A bike on the move.** An alert as soon as a bike that nobody rented starts to move, from the positions all bikes send every minute.",
                   "nl": "**Een fiets die rijdt.** Een alarm zodra een fiets die niemand huurde gaat rijden, uit de posities die alle fietsen elke minuut sturen."},
        "empty": {"en": "**Station almost empty.** After every rental, whether the station will be empty within half an hour, so that a truck can go there.",
                  "nl": "**Station bijna leeg.** Na elke verhuring of het station binnen een halfuur leeg is, zodat er een vrachtwagen heen kan."},
        "voice": {"en": "**Voice.** The app understands \"end my ride\", spoken into the phone, without a recording leaving the phone.",
                  "nl": "**Stem.** De app verstaat \"beëindig mijn rit\", ingesproken in de telefoon, zonder dat er een opname de telefoon verlaat."},
        "fall": {"en": "**A fall.** Within a second of a fall from the bike, the app notices it from the phone's motion sensors, even where the phone has no network.",
                 "nl": "**Een val.** Binnen een seconde na een val van de fiets merkt de app de val op uit de bewegingssensoren van de telefoon, ook waar geen netwerk is."},
    },
    solution={"trucks": "batch", "email": "batch", "price": "online", "chat": "online", "moving": "streaming",
              "empty": "streaming", "voice": "device", "fall": "device"},
    nudge={
        "trucks": {"en": "Is everything the prediction needs known the evening before, and does anyone wait for it at the moment it is made?",
                   "nl": "Is alles wat de voorspelling nodig heeft de avond van tevoren al bekend, en wacht er iemand op het moment dat de voorspelling gemaakt wordt?"},
        "email": {"en": "The email goes to every member at the same time. When must the tips be ready, and does a tip depend on what a member does at that moment?",
                  "nl": "De mail gaat op hetzelfde moment naar elk lid. Wanneer moeten de tips klaar zijn, en hangt een tip af van wat een lid op dat moment doet?"},
        "price": {"en": "Can the price be made before the rider typed where the ride goes? And who waits for it?",
                  "nl": "Kan de prijs gemaakt worden voordat de fietser de bestemming intypte? En wie wacht erop?"},
        "chat": {"en": "The answer depends on the question the rider just asked. Who waits, and what kind of request is it?",
                 "nl": "Het antwoord hangt af van de vraag die de fietser net stelde. Wie wacht er, en wat voor verzoek is dit?"},
        "moving": {"en": "Does anyone ask for this alert? What sets the model off: a request, or the positions that keep coming in?",
                   "nl": "Vraagt iemand om dit alarm? Wat zet het model in gang: een verzoek, of de posities die blijven binnenkomen?"},
        "empty": {"en": "What happens after every rental, without anyone asking? And how quickly must the truck know?",
                  "nl": "Wat gebeurt er na elke verhuring, zonder dat iemand iets vraagt? En hoe snel moet de vrachtwagen het weten?"},
        "voice": {"en": "Where may the recording go? And what does that say about where the model must run?",
                  "nl": "Waar mag de opname heen? En wat zegt dat over de plek waar het model moet draaien?"},
        "fall": {"en": "What if the phone has no network at the moment of the fall? Where must the model be then?",
                 "nl": "Wat als er geen netwerk is op het moment van de val? Waar moet het model dan zijn?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the patterns again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de patronen opnieuw uit."},
        "empty": {"en": "Put each use of a model under the pattern that fits it best.",
                  "nl": "Zet elke toepassing van een model onder het patroon dat het best past."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "*{item}* fits another pattern better.",
                  "nl": "*{item}* past beter bij een ander patroon."},
        "correct": {"en": "All {n} are right. Batch where everything is known in advance and the answer may wait; online where someone waits for an answer to new input; streaming where events keep coming and someone must act on them; on the device where there may be no network, or the data must stay with the user.",
                    "nl": "Alle {n} kloppen. Batch waar alles vooraf bekend is en het antwoord mag wachten; online waar iemand wacht op een antwoord op nieuwe invoer; streaming waar gebeurtenissen blijven binnenkomen en iemand erop moet reageren; op het apparaat waar misschien geen netwerk is, of waar de data bij de gebruiker moet blijven."},
    },
)

EXERCISES = {
    "unused": FunctionBox(
        name="unused",
        function="unused",
        start=_U_START,
        solution=_U,
        visible=[(BATCH, OPENED)],
        hidden=[_SMALL, ([1, 2, 3], []), ([1, 2, 3], [4, 4, 5]), ([], [1]), ([1, 2], [2, 1, 2, 1])],
        first=[_DOTS],
        picture=unused_picture,
        mistakes=[
            Mistake("every", wrong=_U_EVERY,
                    en="`{call}` gives {got}: no suggestion left over, as if every member opened the app. `used` goes up for every member of `batch`. For which members should it go up?",
                    nl="`{call}` geeft {got}: geen suggestie over, alsof elk lid de app opende. `used` gaat omhoog voor elk lid van `batch`. Voor welke leden hoort `used` omhoog te gaan?"),
            Mistake("repeats", wrong=_U_REPEATS,
                    en="`{call}` gives {got}, and it should give {want}: too few, since a member who opened the app twice counts twice. A suggestion is seen or not, however often. Which list has every member with a suggestion once?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: te weinig, want een lid dat de app twee keer opende, telt twee keer. Een suggestie is gezien of niet, hoe vaak ook. Welke lijst heeft elk lid met een suggestie één keer?"),
            Mistake("all-seen", wrong=_U_ALL_SEEN,
                    en="`{call}` gives {got}, and it should give {want}. `len(seen)` counts every member who opened the app, including the new members who joined during the day and had no suggestion. Which members of `seen` had one?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn. `len(seen)` telt elk lid dat de app opende, ook de nieuwe leden die overdag lid werden en geen suggestie hadden. Welke leden van `seen` hadden er een?"),
            Mistake("opens", wrong=_U_OPENS,
                    en="`{call}` gives {got}, and it should give {want}: `len(opened)` is the number of times the app was opened, with repeats and new members. How many different members of `batch` opened it?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `len(opened)` is het aantal keer dat de app openging, met herhalingen en nieuwe leden. Hoeveel verschillende leden van `batch` openden de app?"),
            Mistake("all-seen-loop", wrong=_U_SEEN_LOOP,
                    en="`{call}` gives {got}, and it should give {want}. `len(seen)` counts every member who opened the app, including the new members who joined during the day and had no suggestion. Which members of `seen` had one?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn. `len(seen)` telt elk lid dat de app opende, ook de nieuwe leden die overdag lid werden en geen suggestie hadden. Welke leden van `seen` hadden er een?"),
            Mistake("opens-loop", wrong=_U_OPENS_LOOP,
                    en="`{call}` gives {got}, and it should give {want}: `len(opened)` is the number of times the app was opened, with repeats and new members. How many different members of `batch` opened it?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `len(opened)` is het aantal keer dat de app openging, met herhalingen en nieuwe leden. Hoeveel verschillende leden van `batch` openden de app?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with two lines: when `user` opened the app, `used` goes up by 1.",
                      "nl": "Vervang de `...` door twee regels: als `user` de app opende, gaat `used` 1 omhoog."},
            "arguments": {"en": "The check calls `{call}`, with two lists, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(batch, opened)`?",
                          "nl": "De check roept `{call}` aan, met twee lijsten, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(batch, opened)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does `used` go up once for every member of `batch` who is in `seen`, and for nobody else?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Gaat `used` één keer omhoog voor elk lid van `batch` dat in `seen` zit, en voor niemand anders?"},
            "almost": {"en": "Right for the day of the app. But `{call}` gives {got}, and it should give {want}. Does your function count with the lists it gets, and not with a fixed number?",
                       "nl": "Goed voor de dag van de app. Maar `{call}` geeft {got}, en het moet {want} zijn. Telt je functie met de lijsten die de functie krijgt, en niet met een vast getal?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return len(batch) - used`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return len(batch) - used`?"},
            "correct": {"en": "Right in all {n} cases. Of the 10,000 suggestions of the night, 8,684 were never seen: 86.8% of the work of the batch was for nobody. In the picture: what did the new members of the day get from the batch?",
                        "nl": "Goed in alle {n} gevallen. Van de 10.000 suggesties van de nacht zijn er 8.684 nooit gezien: 86,8% van het werk van de batch was voor niemand. In de afbeelding: wat kregen de nieuwe leden van deze dag van de batch?"},
        },
    ),
    "recent": FunctionBox(
        name="recent",
        function="recent",
        start=_R_START,
        solution=_R,
        visible=[(TIMES, BURST, WINDOW)],
        hidden=[([600, 605, 610, 615, 620], 620, 15), ([100, 200], 150, 30), ([5, 10], 10, 60), ([], 50, 15),
                ([700, 710, 720], 705, 10)],
        first=[_DOTS],
        picture=recent_picture,
        mistakes=[
            Mistake("before", wrong=_R_BEFORE,
                    en="`{call}` gives {got}, and it should give {want}: every rental of the day up to `now`, even those from hours ago. Which rentals are too old to count?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke verhuring van de dag tot `now`, ook de verhuringen van uren geleden. Welke verhuringen zijn te oud om mee te tellen?"),
            Mistake("after", wrong=_R_AFTER,
                    en="`{call}` gives {got}, and it should give {want}: it counts the rentals after `now` too, which have not happened yet at that moment. A stream only has what came in so far. Which rentals lie after `now`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de functie telt ook de verhuringen na `now`, en op dat moment zijn deze verhuringen nog niet gebeurd. Een stream heeft alleen wat er tot nu toe binnenkwam. Welke verhuringen liggen na `now`?"),
            Mistake("not-now", wrong=_R_NOT_NOW,
                    en="`{call}` gives {got}, and it should give {want}: one too few. The rental at `now` itself, the one that just came in, is left out. Should a rental at exactly `now` count?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: één te weinig. De verhuring op `now` zelf valt weg: de verhuring die net binnenkwam. Hoort een verhuring op precies `now` mee te tellen?"),
            Mistake("from", wrong=_R_FROM, state="almost",
                    en="Right for 17:30. But `{call}` gives {got}, and it should give {want}: a rental exactly `minutes` minutes before `now` counts too, so the window is one minute longer. Should a rental of exactly 15 minutes ago still count?",
                    nl="Goed voor 17:30. Maar `{call}` geeft {got}, en het moet {want} zijn: een verhuring van precies `minutes` minuten voor `now` telt ook mee, dus is het venster een minuut langer. Hoort een verhuring van precies 15 minuten geleden nog mee te tellen?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with two lines: for a rental after `now - minutes`, and at `now` or before, `count` goes up by 1.",
                      "nl": "Vervang de `...` door twee regels: voor een verhuring na `now - minutes`, en op `now` of eerder, gaat `count` 1 omhoog."},
            "arguments": {"en": "The check calls `{call}`, with a list of minutes and two numbers, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(times, now, minutes)`?",
                          "nl": "De check roept `{call}` aan, met een lijst minuten en twee getallen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(times, now, minutes)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does `count` go up only for a rental after `now - minutes`, and at `now` or before?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Gaat `count` alleen omhoog voor een verhuring na `now - minutes`, en op `now` of eerder?"},
            "almost": {"en": "Right for 17:30. But `{call}` gives {got}, and it should give {want}. Does your function use `now` and `minutes`, and not fixed numbers?",
                       "nl": "Goed voor 17:30. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je functie `now` en `minutes`, en geen vaste getallen?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return count`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return count`?"},
            "correct": {"en": "Right in all {n} cases. At 17:30, 17 bikes left station 31 in the last 15 minutes; in the morning, never more than 6 in a quarter of an hour. In the picture: from what time does the rule send a truck?",
                        "nl": "Goed in alle {n} gevallen. Om 17:30 vertrokken er in de laatste 15 minuten 17 fietsen van station 31; 's ochtends nooit meer dan 6 in een kwartier. In de afbeelding: vanaf hoe laat stuurt de regel een vrachtwagen?"},
        },
    ),
    "patterns": PATTERNS,
}

# The lesson is done when these are correct. "hourly" is the choice question after the demo.
REQUIRED = ["unused", "recent", "patterns", "hourly"]

_SORTED = dict(PATTERNS.solution)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("unused", _U_START, "empty", None),
    ("unused", "", "empty", None),
    ("unused", _U, "correct", None),
    ("unused", _u("        if user in opened:\n            used = used + 1\n"), "correct", None),
    ("unused", _u("        if user in seen:\n            used += 1\n"), "correct", None),
    ("unused", _U_EVERY, "wrong", "every"),
    ("unused", _U_REPEATS, "wrong", "repeats"),
    ("unused", _U_ALL_SEEN, "wrong", "all-seen"),
    ("unused", _U_OPENS, "wrong", "opens"),
    ("unused", _U_SEEN_LOOP, "wrong", "all-seen-loop"),
    ("unused", _U_OPENS_LOOP, "wrong", "opens-loop"),
    ("unused", _u("        if user in seen:\n            ...\n"), "wrong", "dots-left"),
    ("unused", _U.replace("    return len(batch) - used\n", ""), "wrong", "no-return"),
    ("unused", _U.replace("def unused(batch, opened):", "def unused(batch):"), "error", "arguments"),
    ("unused", "def unused(batch, opened):\n    return 8684\n", "almost", None),
    ("unused", _u("        if user in seen:\n            used = used + 2\n"), "wrong", None),
    ("unused", "x = 1\n", "wrong", "no-function"),
    ("recent", _R_START, "empty", None),
    ("recent", "", "empty", None),
    ("recent", _R, "correct", None),
    ("recent", _r("now - minutes < t <= now"), "correct", None),
    ("recent", _r("t <= now and t > now - minutes"), "correct", None),
    ("recent", _R_BEFORE, "wrong", "before"),
    ("recent", _R_AFTER, "wrong", "after"),
    ("recent", _R_NOT_NOW, "wrong", "not-now"),
    ("recent", _R_FROM, "almost", "from"),
    ("recent", _r("t > now - minutes and ..."), "wrong", "dots-left"),
    ("recent", _R.replace("    return count\n", ""), "wrong", "no-return"),
    ("recent", _R.replace("def recent(times, now, minutes):", "def recent(times, now):"), "error", "arguments"),
    ("recent", "def recent(times, now, minutes):\n    return 17\n", "almost", None),
    ("recent", _r("t > 1035 and t <= 1050"), "almost", None),
    ("patterns", {}, "empty", None),
    ("patterns", _SORTED, "correct", None),
    ("patterns", {"trucks": "batch", "email": "batch"}, "wrong", "unfinished"),
    ("patterns", _SORTED | {"trucks": "streaming"}, "wrong", "wrong:trucks"),
    ("patterns", _SORTED | {"email": "online"}, "wrong", "wrong:email"),
    ("patterns", _SORTED | {"price": "batch"}, "wrong", "wrong:price"),
    ("patterns", _SORTED | {"chat": "streaming"}, "wrong", "wrong:chat"),
    ("patterns", _SORTED | {"moving": "online"}, "wrong", "wrong:moving"),
    ("patterns", _SORTED | {"empty": "batch"}, "wrong", "wrong:empty"),
    ("patterns", _SORTED | {"voice": "online"}, "wrong", "wrong:voice"),
    ("patterns", _SORTED | {"fall": "streaming"}, "wrong", "wrong:fall"),
]
