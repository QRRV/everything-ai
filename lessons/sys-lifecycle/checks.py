# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-lifecycle: the life of an AI system, from the first idea to switching it off. The eight
stages of ISO/IEC 22989, the roles of the parties around a system, and the life of the course's spam service in a city,
month by month: the learner adds up the months per stage, and finds the month the monitoring calls for a new look.

The history of the service is invented by the course, to show how a life cycle goes; the stage and role names are the
clause titles of ISO/IEC 22989:2022, which the course does not copy further."""
import html
import math

from everything_ai import check, yours
from everything_ai.check import FunctionBox, Mistake

LESSON = "sys-lifecycle"
LANGS = ("en", "nl")

# The stages of ISO/IEC 22989:2022, clause 6.2, in its order.
STAGES = ("inception", "design and development", "verification and validation", "deployment",
          "operation and monitoring", "continuous validation", "re-evaluation", "retirement")
STAGE_NAMES = {
    "en": {s: s for s in STAGES},
    "nl": {"inception": "idee", "design and development": "ontwerp en ontwikkeling",
           "verification and validation": "verificatie en validatie", "deployment": "deployment",
           "operation and monitoring": "gebruik en monitoring", "continuous validation": "doorlopende validatie",
           "re-evaluation": "herbeoordeling", "retirement": "uitfasering"},
}

# The roles of ISO/IEC 22989:2022, clause 5.19, and who has them around the course's spam service.
ROLES = ("AI provider", "AI producer", "AI customer", "AI partner", "AI subject", "relevant authorities")
ROLE_NAMES = {
    "en": {r: r for r in ROLES},
    "nl": {"AI provider": "AI-aanbieder", "AI producer": "AI-producent", "AI customer": "AI-klant",
           "AI partner": "AI-partner", "AI subject": "AI-subject", "relevant authorities": "relevante autoriteiten"},
}
WHO = {
    "en": {"AI provider": "the company", "AI producer": "the company", "AI customer": "the city",
           "AI partner": "a labeling firm", "AI subject": "the residents", "relevant authorities": "the privacy regulator"},
    "nl": {"AI provider": "het bedrijf", "AI producer": "het bedrijf", "AI customer": "de gemeente",
           "AI partner": "een labelbureau", "AI subject": "de inwoners", "relevant authorities": "de privacytoezichthouder"},
}

# The life of the course's spam service in the mailbox of a city, as (stage, months): invented by the course.
HISTORY = [
    ("inception", 1),
    ("design and development", 3),
    ("verification and validation", 1),
    ("deployment", 1),
    ("operation and monitoring", 14),
    ("re-evaluation", 1),
    ("design and development", 2),
    ("verification and validation", 1),
    ("deployment", 1),
    ("operation and monitoring", 9),
    ("re-evaluation", 1),
    ("retirement", 1),
]
GOAL = 0.9   # the accuracy the city and the company agreed on at the start
# The accuracy of each month of the first period of operation (months 7 to 20 of its life), on messages checked by hand.
ACCURACY = [0.95, 0.95, 0.94, 0.95, 0.94, 0.93, 0.93, 0.92, 0.92, 0.91, 0.91, 0.91, 0.89, 0.88]
# The same for the second period (months 26 to 34), with version 2 of the service.
ACCURACY_2 = [0.96, 0.96, 0.95, 0.96, 0.95, 0.95, 0.94, 0.95, 0.94]
RETRAINS = (12, 17, 30)   # the months in which feedback trained the model again on new marks


# --- the functions of the exercises, as the course writes them ----------------------------------------------------------

def months_per_stage(history):
    totals = {}
    for stage, months in history:
        if stage not in totals:
            totals[stage] = 0
        totals[stage] = totals[stage] + months
    return totals


def first_month_below(accuracies, goal):
    month = 0
    for accuracy in accuracies:
        month = month + 1
        if accuracy < goal:
            return month
    return None


# --- the life, month by month -------------------------------------------------------------------------------------------

def months() -> list:
    """Each month of the life as a dict: its number from 1, its stage, the version that answers, and the accuracy of
    that month when the service is in operation."""
    out, n = [], 0
    ops = [list(ACCURACY), list(ACCURACY_2)]
    period = -1
    for stage, length in HISTORY:
        if stage == "operation and monitoring":
            period += 1
        for k in range(length):
            n += 1
            acc = ops[period][k] if stage == "operation and monitoring" else None
            version = "spam-1" if n <= 25 else "spam-2"   # spam-2 answers from its deployment on
            out.append({"month": n, "stage": stage, "accuracy": acc, "version": version,
                        "retrain": n in RETRAINS, "first": k == 0})
    return out


LIFE = months()
TOTAL_MONTHS = len(LIFE)                               # 36
OPERATION_MONTHS = months_per_stage(HISTORY)["operation and monitoring"]   # 23
ALERT_MONTH = 6 + first_month_below(ACCURACY, GOAL)    # 19: the month of the life in which the accuracy drops below the goal

# Who does something in each stage, as roles.
ACTIVE = {
    "inception": ("AI customer", "AI provider", "relevant authorities"),
    "design and development": ("AI producer", "AI partner"),
    "verification and validation": ("AI producer", "AI customer"),
    "deployment": ("AI provider", "AI customer"),
    "operation and monitoring": ("AI customer", "AI provider", "AI subject"),
    "continuous validation": ("AI producer",),
    "re-evaluation": ("AI customer", "AI provider"),
    "retirement": ("AI provider", "AI customer", "AI subject"),
}


# ======== words and drawing ===============================================================================================

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem;max-width:var(--ea-measure)"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem;max-width:var(--ea-measure)"
_SVG = "width:100%;max-width:330px;height:auto;font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums"


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma and a thousands dot in Dutch."""
    out = f"{x:,.{digits}f}"
    return out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out


def stage_name(stage: str, lang: str) -> str:
    return STAGE_NAMES[lang].get(stage, stage)


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


def _arrow(x1: float, y1: float, x2: float, y2: float, ink: str = "var(--ea-ink-2)") -> str:
    """A line from (x1, y1) to (x2, y2) with a small head at its end."""
    a = math.atan2(y2 - y1, x2 - x1)
    h1 = (x2 - 6 * math.cos(a - 0.45), y2 - 6 * math.sin(a - 0.45))
    h2 = (x2 - 6 * math.cos(a + 0.45), y2 - 6 * math.sin(a + 0.45))
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" style="stroke:{ink};stroke-width:1.3"/>'
            f'<path d="M{h1[0]:.1f},{h1[1]:.1f} L{x2:.1f},{y2:.1f} L{h2[0]:.1f},{h2[1]:.1f}" '
            f'style="fill:none;stroke:{ink};stroke-width:1.3"/>')


def _stage_box(x: float, y: float, w: float, h: float, label: str, strong: bool = False) -> str:
    stroke = "var(--ea-ink)" if strong else "var(--ea-ink-3)"
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="5" '
            f'style="fill:var(--ea-page);stroke:{stroke};stroke-width:{1.8 if strong else 1.2}"/>'
            + _t(x + w / 2, y + h / 2 + 4.5, label, 12, 600 if strong else 400, "var(--ea-ink)", "middle"))


IDEA_TEXT = {
    "en": {"back": "back to design", "end": "or retire",
           "alongside": "alongside, for a system that keeps learning",
           "alt": "The eight stages of the life cycle of an AI system according to ISO/IEC 22989, from top to bottom: inception, design and development, verification and validation, deployment, operation and monitoring, re-evaluation and retirement. Continuous validation runs beside operation and monitoring. From re-evaluation an arrow goes back up to design and development, and one goes down to retirement."},
    "nl": {"back": "terug naar ontwerp", "end": "of uitfaseren",
           "alongside": "ernaast, bij een systeem dat blijft leren",
           "alt": "De acht fasen van de levenscyclus van een AI-systeem volgens ISO/IEC 22989, van boven naar beneden: idee, ontwerp en ontwikkeling, verificatie en validatie, deployment, gebruik en monitoring, herbeoordeling en uitfasering. Doorlopende validatie loopt naast gebruik en monitoring. Vanaf herbeoordeling gaat een pijl terug omhoog naar ontwerp en ontwikkeling, en een pijl omlaag naar uitfasering."},
}


def idea_picture(lang: str) -> str:
    """The stages from top to bottom, continuous validation beside operation, and the two ways out of re-evaluation."""
    w = IDEA_TEXT[lang]
    X, W, H, GAP = 34.0, 186.0, 26.0, 14.0
    order = ["inception", "design and development", "verification and validation", "deployment",
             "operation and monitoring", "re-evaluation", "retirement"]
    y = {s: 4 + i * (H + GAP) for i, s in enumerate(order)}
    parts = []
    for i, s in enumerate(order):
        parts.append(_stage_box(X, y[s], W, H, stage_name(s, lang), strong=s == "operation and monitoring"))
        if i < len(order) - 1:
            parts.append(_arrow(X + W / 2, y[s] + H, X + W / 2, y[order[i + 1]] - 1))
    # continuous validation beside operation and monitoring
    cx, cw = 232.0, 96.0
    below = _wrap(w["alongside"], 14)
    cy = y["operation and monitoring"] + H / 2 - (36 + 12 * len(below)) / 2
    parts.append(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{cw:.1f}" height="{36 + 12 * len(below):.1f}" rx="5" '
                 f'style="fill:var(--ea-page);stroke:var(--ea-ink-3);stroke-width:1.2;stroke-dasharray:4 3"/>')
    name = stage_name("continuous validation", lang).split(" ", 1)
    parts.append(_t(cx + 8, cy + 16, name[0], 12, 600))
    parts.append(_t(cx + 8, cy + 30, name[1] if len(name) > 1 else "", 12, 600))
    for k, line in enumerate(below):
        parts.append(_t(cx + 8, cy + 44 + 12 * k, line, 10.5, 400, "var(--ea-ink-2)"))
    parts.append(f'<line x1="{X + W:.1f}" y1="{y["operation and monitoring"] + H / 2:.1f}" x2="{cx:.1f}" '
                 f'y2="{y["operation and monitoring"] + H / 2:.1f}" style="stroke:var(--ea-ink-3);stroke-width:1.2;stroke-dasharray:3 3"/>')
    # back from re-evaluation to design and development, on the left
    ry, dy = y["re-evaluation"] + H / 2, y["design and development"] + H / 2
    parts.append(f'<path d="M{X:.1f},{ry:.1f} L{X - 22:.1f},{ry:.1f} L{X - 22:.1f},{dy:.1f}" '
                 f'style="fill:none;stroke:var(--ea-ink);stroke-width:1.3"/>')
    parts.append(_arrow(X - 22, dy, X - 1, dy, "var(--ea-ink)"))
    parts.append(f'<text transform="translate({X - 27:.1f},{(ry + dy) / 2:.1f}) rotate(-90)" text-anchor="middle" '
                 f'style="font-size:10.5px;fill:var(--ea-ink-2)">{html.escape(w["back"])}</text>')
    parts.append(_t(X + W / 2 + 8, y["retirement"] - 3.5, w["end"], 10.5, 400, "var(--ea-ink-2)"))
    height = y["retirement"] + H + 4
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(w["alt"])}" style="margin:0.25rem 0 0.75rem;'
            f'max-width:min(100%,330px)"><svg viewBox="0 0 330 {height:.0f}" aria-hidden="true" style="{_SVG}">'
            f'{"".join(parts)}</svg></figure>')


def _wrap(text_: str, width: int) -> list:
    """Words in lines of at most about width characters."""
    lines, line = [], ""
    for word in text_.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + ([line] if line else [])


def bars_svg(rows: list, title: str, unit: tuple, lang: str, alt: str) -> str:
    """One bar per row, (label, value, strong), from 0 to its value, with its label above it and its value at its end;
    unit is the word for one and for more."""
    BAR, ROW = 13, 34
    W = 322 - 7.4 * max(len(f"{number(v, 1, lang)} {unit[1]}") for _, v, _ in rows)
    top = max(v for _, v, _ in rows) or 1
    parts, y = [], 0
    if title:
        parts.append(_t(0, 15, title, 14, 600))
        y = 28
    for label, value, strong in rows:
        width = max(2.0, W * value / top) if value > 0 else 0.0
        fill = "var(--ea-ink)" if strong else "var(--ea-ink-3)"
        shown = number(value, 0 if value == int(value) else 1, lang)
        parts.append(_t(0, y + 12, label, 12.5, 600 if strong else 400, "var(--ea-ink)" if strong else "var(--ea-ink-2)")
                     + f'<rect x="0" y="{y + 16}" width="{width:.1f}" height="{BAR}" rx="2" style="fill:{fill}"/>'
                     + _t(width + 6, y + 27, f"{shown} {unit[0] if value == 1 else unit[1]}", 12.5, 600))
        y += ROW
    return f'<svg viewBox="0 0 330 {y + 4}" role="img" aria-label="{html.escape(alt)}" style="{_SVG}">{"".join(parts)}</svg>'


def accuracy_svg(values: list, goal: float, mine, lang: str, alt: str, first_month: int = 1) -> str:
    """The accuracy per month as a line with dots, the goal as a dashed line, and one month marked."""
    w = _YOURS_TEXT[lang]["below"]
    L_, R_, T_, B_ = 34.0, 322.0, 12.0, 120.0
    lo, hi = 0.85, 1.0
    n = max(len(values), 1)

    def px(i):
        return L_ + (R_ - L_) * (i + 0.5) / n

    def py(v):
        return B_ - (B_ - T_) * (min(max(v, lo), hi) - lo) / (hi - lo)

    parts = []
    for v in (0.85, 0.9, 0.95, 1.0):
        parts.append(_t(L_ - 6, py(v) + 4, number(v, 2, lang), 10.5, 400, "var(--ea-ink-3)", "end"))
        parts.append(f'<line x1="{L_:.1f}" y1="{py(v):.1f}" x2="{R_:.1f}" y2="{py(v):.1f}" style="stroke:var(--ea-line);stroke-width:1"/>')
    parts.append(f'<line x1="{L_:.1f}" y1="{py(goal):.1f}" x2="{R_:.1f}" y2="{py(goal):.1f}" '
                 f'style="stroke:var(--ea-ink);stroke-width:1.4;stroke-dasharray:5 3"/>')
    parts.append(_t(L_ + 4, py(goal) + 14, w["goal"].format(g=number(goal, 2, lang)), 11, 600, "var(--ea-ink)", "start", halo=True))
    pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(values))
    if values:
        parts.append(f'<polyline points="{pts}" style="fill:none;stroke:var(--ea-ink-3);stroke-width:1.6"/>')
    for i, v in enumerate(values):
        strong = mine == i + 1
        r = 4.5 if strong else 2.6
        fill = "var(--ea-ink)" if strong else "var(--ea-ink-3)"
        parts.append(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="{r}" style="fill:{fill}"/>')
    if isinstance(mine, int) and 1 <= mine <= len(values):
        x = px(mine - 1)
        parts.append(_t(min(max(x, L_ + 40), R_ - 40), B_ + 30, w["mine"].format(m=mine), 11.5, 600, "var(--ea-ink)",
                        "middle"))
        parts.append(f'<line x1="{x:.1f}" y1="{py(values[mine - 1]) + 6:.1f}" x2="{x:.1f}" y2="{B_ + 3:.1f}" '
                     f'style="stroke:var(--ea-ink);stroke-width:1"/>')
    for i in range(len(values)):
        if (first_month + i) % 2 == 1 or n <= 10:
            parts.append(_t(px(i), B_ + 14, str(first_month + i), 10, 400, "var(--ea-ink-3)", "middle"))
    return f'<svg viewBox="0 0 330 {B_ + 38:.0f}" role="img" aria-label="{html.escape(alt)}" style="{_SVG}">{"".join(parts)}</svg>'


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


def _amount(v) -> float | None:
    """v as a number of months, 0 or more, or None: never from text or True and False."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
        return None
    return float(v)


_YOURS_TEXT = {
    "en": {"months": {"title": "Your months_per_stage on the life of the service",
                      "intro": "Your `months_per_stage` on the 12 entries of `C.HISTORY`: the 36 months of the course's spam service in the mailbox of a city.",
                      "chart": "Months per stage",
                      "alt": "One bar per stage of the life cycle, with the months your function adds up for it; the bar of operation and monitoring stands out.",
                      "unit": ("month", "months"),
                      "share": "operation and monitoring: {m} of {t} months, {p}%",
                      "alongside": "Continuous validation runs alongside operation and monitoring, so the history gives it no months of its own.",
                      "none": "Your `months_per_stage` stops with an error, takes too long, or does not give a dict from stages to numbers of months, so nothing can be drawn yet."},
           "below": {"title": "Your first_month_below on the accuracy of the service",
                     "intro": "Your `first_month_below` on `C.ACCURACY`: the accuracy in each month of the first period of operation, with the goal of {g}.",
                     "alt": "The accuracy of the service in each month of operation as a line with dots, the goal as a dashed line across it, and the month your function gives marked under the line.",
                     "goal": "goal {g}", "mine": "your month: {m}",
                     "found": "Your function gives month {m}: month {life} of the life of the service, after {d} months of operation without an alert.",
                     "nothing": "Your function gives None: no month below the goal.",
                     "none": "Your `first_month_below` stops with an error, takes too long, or gives something that is not a month of the list or None, so nothing can be drawn yet."}},
    "nl": {"months": {"title": "Je months_per_stage op het leven van de dienst",
                      "intro": "Je `months_per_stage` op de 12 regels van `C.HISTORY`: de 36 maanden van de spamdienst van de course in de mailbox van een gemeente.",
                      "chart": "Maanden per fase",
                      "alt": "Eén balk per fase van de levenscyclus, met de maanden die je functie voor de fase optelt; de balk van gebruik en monitoring valt op.",
                      "unit": ("maand", "maanden"),
                      "share": "gebruik en monitoring: {m} van de {t} maanden, {p}%",
                      "alongside": "Doorlopende validatie loopt naast gebruik en monitoring, dus geeft de geschiedenis deze fase geen eigen maanden.",
                      "none": "Je `months_per_stage` stopt met een fout, duurt te lang, of geeft geen dict van fasen naar aantallen maanden, dus is er nog geen afbeelding."},
           "below": {"title": "Je first_month_below op de accuracy van de dienst",
                     "intro": "Je `first_month_below` op `C.ACCURACY`: de accuracy in elke maand van de eerste periode van gebruik, met het doel van {g}.",
                     "alt": "De accuracy van de dienst in elke maand van gebruik als een lijn met punten, het doel als een lijn van streepjes erdoorheen, en de maand die je functie geeft gemarkeerd onder de lijn.",
                     "goal": "doel {g}", "mine": "jouw maand: {m}",
                     "found": "Je functie geeft maand {m}: maand {life} van het leven van de dienst, na {d} maanden gebruik zonder alarm.",
                     "nothing": "Je functie geeft None: geen maand onder het doel.",
                     "none": "Je `first_month_below` stopt met een fout, duurt te lang, of geeft iets dat geen maand uit de lijst is en geen None, dus is er nog geen afbeelding."}},
}


def months_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["months"]
    fn = _fn(seen)
    got = _call(lambda: fn([tuple(h) for h in HISTORY])) if fn is not None else None
    ok = isinstance(got, dict) and all(isinstance(k, str) for k in got) and all(_amount(v) is not None for v in got.values())
    if not ok:
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    rows = [(stage_name(s, lang), _amount(got[s]), s == "operation and monitoring") for s in STAGES if s in got]
    rows += [(str(k), _amount(v), False) for k, v in got.items() if k not in STAGES]
    if not rows:
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    body = _note(w["intro"]) + bars_svg(rows, w["chart"], w["unit"], lang, w["alt"])
    total = sum(v for _, v, _ in rows)
    op = _amount(got.get("operation and monitoring", 0)) or 0
    if total > 0:
        body += _note(w["share"].format(m=number(op, 0 if op == int(op) else 1, lang),
                                        t=number(total, 0 if total == int(total) else 1, lang),
                                        p=number(100 * op / total, 0, lang)))
    if "continuous validation" not in got:
        body += _note(w["alongside"])
    return yours.figure(w["title"], body, alt=w["title"])


def below_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["below"]
    intro = w["intro"].format(g=number(GOAL, 2, lang))
    fn = _fn(seen)
    got = _call(lambda: fn(list(ACCURACY), GOAL)) if fn is not None else None
    ok = got is None and fn is not None or (isinstance(got, int) and not isinstance(got, bool) and 1 <= got <= len(ACCURACY))
    if fn is None or got is _STOPPED or not ok:
        return yours.figure(w["title"], _note(intro) + _note(w["none"], _LOSS_UI), alt=w["title"])
    svg = accuracy_svg(ACCURACY, GOAL, got, lang, w["alt"], first_month=1)
    says = w["nothing"] if got is None else w["found"].format(m=got, life=6 + got, d=got - 1)
    return yours.figure(w["title"], _note(intro) + svg + _note(says), alt=w["title"])


# ======== the demo: the life of the service, month by month ===============================================================

DEMO_TEXT = {
    "en": {"title": "Month {m} of {n}", "now": "now: {stage}", "version": "version {v}",
           "who": "who: {who}", "accuracy": "accuracy {a}", "goal": "goal {g}",
           "alt": "Month {m} of the life of the course's spam service: a row per stage with the months the service spent in it, a line at the current month, the accuracy in the months of operation, and who does something now.",
           "says": {
               "inception": "The city wants fewer spam messages in its mailbox for residents and asks the company for a filter. Together they agree on what a mistake costs: a real question from a resident in the spam folder is worse than a spam message in the inbox. They also agree on a goal: an accuracy of at least {g}. The privacy rules say what the service may do with the messages of residents.",
               "design and development": "The company builds the service: collection, verification, features, the model and serving. A labeling firm labels messages as spam or ok, so that the model has something to learn from.",
               "design and development 2": "Back to design. The company looks at the messages the service got wrong: spammers now write words the model never saw. It collects new labeled messages and changes the features; this becomes version spam-2.",
               "verification and validation": "The company checks that the service does what was agreed (verification): the accuracy on test messages is above the goal. Then staff of the city try it on a week of real mail, to check that it helps them (validation).",
               "deployment": "The service goes into the mailbox of the city. Staff learn where the spam folder is, and how to mark a message the service got wrong.",
               "operation and monitoring": "The service labels every message as spam or ok. Each month, staff check a sample of messages by hand: the accuracy of this month is {a}.",
               "alert": "The accuracy is {a}, below the goal of {g}, for the first time: the monitoring raises an alert.",
               "retrain": " Feedback trained the model again on new marks; before the new model answers, it is checked on the test messages: continuous validation.",
               "re-evaluation": "The city and the company look again at the service as a whole. The service is still needed, but no longer good enough: back to design and development.",
               "re-evaluation 2": "A new look: the city moves to a mail provider that filters spam itself, so the service is not needed anymore. Its next stage is retirement.",
               "retirement": "The company switches the service off. The city tells residents that their mail is now filtered elsewhere, and the stored messages and marks are deleted as agreed at the start.",
           }},
    "nl": {"title": "Maand {m} van {n}", "now": "nu: {stage}", "version": "versie {v}",
           "who": "wie: {who}", "accuracy": "accuracy {a}", "goal": "doel {g}",
           "alt": "Maand {m} van het leven van de spamdienst van de course: een rij per fase met de maanden die de dienst in de fase doorbracht, een lijn bij de huidige maand, de accuracy in de maanden van gebruik, en wie nu iets doet.",
           "says": {
               "inception": "De gemeente wil minder spam in de mailbox voor inwoners en vraagt het bedrijf om een filter. Samen spreken de gemeente en het bedrijf af wat een fout kost: een echte vraag van een inwoner in de spammap is erger dan een spambericht in de inbox. Er komt ook een doel: een accuracy van minstens {g}. De privacyregels zeggen wat de dienst met de berichten van inwoners mag doen.",
               "design and development": "Het bedrijf bouwt de dienst: verzamelen, controle, features, het model en serving. Een labelbureau labelt berichten als spam of ok, zodat het model iets heeft om van te leren.",
               "design and development 2": "Terug naar ontwerp. Het bedrijf bekijkt de berichten die de dienst fout had: spammers schrijven nu woorden die het model nooit zag. Het bedrijf verzamelt nieuwe gelabelde berichten en verandert de features; dat wordt versie spam-2.",
               "verification and validation": "Het bedrijf kijkt of de dienst doet wat afgesproken is (verificatie): de accuracy op testberichten ligt boven het doel. Daarna proberen medewerkers van de gemeente de dienst een week op echte mail, om te zien of de dienst de medewerkers helpt (validatie).",
               "deployment": "De dienst gaat in de mailbox van de gemeente. Medewerkers leren waar de spammap is, en hoe je een bericht markeert dat de dienst fout had.",
               "operation and monitoring": "De dienst geeft elk bericht het label spam of ok. Elke maand kijken medewerkers een steekproef van berichten met de hand na: de accuracy van deze maand is {a}.",
               "alert": "De accuracy is {a}, voor het eerst onder het doel van {g}: de monitoring geeft een alarm.",
               "retrain": " Feedback trainde het model opnieuw op nieuwe markeringen; voordat het nieuwe model antwoordt, wordt het op de testberichten gecontroleerd: doorlopende validatie.",
               "re-evaluation": "De gemeente en het bedrijf bekijken de dienst als geheel opnieuw. De dienst is nog nodig, alleen niet meer goed genoeg: terug naar ontwerp en ontwikkeling.",
               "re-evaluation 2": "Een nieuwe blik: de gemeente stapt over op een mailprovider die zelf spam filtert, dus is de dienst niet meer nodig. De volgende fase is uitfasering.",
               "retirement": "Het bedrijf zet de dienst uit. De gemeente vertelt inwoners dat de mail nu ergens anders gefilterd wordt, en de bewaarde berichten en markeringen worden gewist, zoals bij het begin afgesproken.",
           }},
}


def say(m: int, lang: str) -> str:
    """What happens in month m."""
    w = DEMO_TEXT[lang]["says"]
    month = LIFE[max(1, min(TOTAL_MONTHS, int(m))) - 1]
    stage = month["stage"]
    g = number(GOAL, 2, lang)
    if stage == "operation and monitoring":
        a = number(month["accuracy"], 2, lang)
        out = w["alert"].format(a=a, g=g) if month["month"] == ALERT_MONTH else w["operation and monitoring"].format(a=a)
        if month["retrain"]:
            out += w["retrain"]
        return out
    later = month["month"] > 21 and stage in ("design and development", "re-evaluation")   # after the first re-evaluation
    return w[stage + " 2" if later else stage].format(g=g)


def frame(m: int, lang: str) -> tuple[str, str]:
    """Month m of the life (1 to 36): a row per stage, the accuracy so far, and who does something now."""
    w = DEMO_TEXT[lang]
    m = max(1, min(TOTAL_MONTHS, int(m)))
    month = LIFE[m - 1]
    LX, RX = 158.0, 326.0
    step = (RX - LX) / TOTAL_MONTHS
    parts = [_t(0, 14, w["title"].format(m=m, n=TOTAL_MONTHS), 13, 600),
             _t(RX, 14, w["version"].format(v=month["version"]), 11.5, 400, "var(--ea-ink-2)", "end")]
    top, row = 26.0, 17.0
    for i, s in enumerate(STAGES):
        y = top + i * row
        now = s == month["stage"] or (s == "continuous validation" and month["retrain"])
        parts.append(_t(0, y + 11.5, stage_name(s, lang), 10.5, 600 if now else 400,
                        "var(--ea-ink)" if now else "var(--ea-ink-2)"))
        parts.append(f'<line x1="{LX:.1f}" y1="{y + 8:.1f}" x2="{RX:.1f}" y2="{y + 8:.1f}" style="stroke:var(--ea-line);stroke-width:1"/>')
        for mo in LIFE:
            here = mo["stage"] == s or (s == "continuous validation" and mo["retrain"])
            if not here:
                continue
            fill = "var(--ea-ink-3)" if mo["month"] < m else ("var(--ea-ink)" if mo["month"] == m else "var(--ea-line)")
            parts.append(f'<rect x="{LX + (mo["month"] - 1) * step + 0.5:.1f}" y="{y + 3:.1f}" width="{step - 1:.1f}" '
                         f'height="10" rx="1.5" style="fill:{fill}"/>')
    x_now = LX + (m - 0.5) * step
    bottom = top + len(STAGES) * row
    parts.append(f'<line x1="{x_now:.1f}" y1="{top - 2:.1f}" x2="{x_now:.1f}" y2="{bottom + 82:.1f}" '
                 f'style="stroke:var(--ea-ink);stroke-width:1.2"/>')
    # the accuracy of the months of operation so far, on the same months
    cy0, cy1, lo, hi = bottom + 12, bottom + 78, 0.85, 1.0

    def py(v):
        return cy1 - (cy1 - cy0) * (min(max(v, lo), hi) - lo) / (hi - lo)

    parts.append(_t(0, cy0 + 10, w["accuracy"].format(a=""), 11, 400, "var(--ea-ink-2)"))
    parts.append(f'<line x1="{LX:.1f}" y1="{py(GOAL):.1f}" x2="{RX:.1f}" y2="{py(GOAL):.1f}" '
                 f'style="stroke:var(--ea-ink);stroke-width:1.2;stroke-dasharray:5 3"/>')
    parts.append(_t(LX - 6, py(GOAL) + 4, w["goal"].format(g=number(GOAL, 2, lang)), 10.5, 400, "var(--ea-ink-2)", "end"))
    seen = [mo for mo in LIFE if mo["accuracy"] is not None and mo["month"] <= m]
    for mo in seen:
        strong = mo["month"] == m
        parts.append(f'<circle cx="{LX + (mo["month"] - 0.5) * step:.1f}" cy="{py(mo["accuracy"]):.1f}" '
                     f'r="{3.6 if strong else 2.2}" style="fill:{"var(--ea-ink)" if strong else "var(--ea-ink-3)"}"/>')
    # who does something now
    y = cy1 + 22
    parts.append(_t(0, y, w["now"].format(stage=stage_name(month["stage"], lang)), 12, 600))
    who = [f'{ROLE_NAMES[lang][r]} ({WHO[lang][r]})' for r in ACTIVE[month["stage"]]]
    lines = _wrap(w["who"].format(who=", ".join(who)), 52)
    for k, line in enumerate(lines):
        parts.append(_t(0, y + 17 + 15 * k, line, 11.5, 400, "var(--ea-ink-2)"))
    height = y + 17 + 15 * len(lines)
    svg = (f'<svg viewBox="0 0 330 {height:.0f}" role="img" aria-label="{html.escape(w["alt"].format(m=m))}" '
           f'style="{_SVG}">{"".join(parts)}</svg>')
    return svg, say(m, lang)


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
    """A list of the lesson that the check shows by its name: C.HISTORY or C.ACCURACY."""

    def __init__(self, items, name: str):
        super().__init__(items)
        self._name = name

    def __repr__(self) -> str:
        return self._name

    def __deepcopy__(self, memo):
        return list(self)


HISTORY_CASE = _Listed([tuple(h) for h in HISTORY], "C.HISTORY")
ACCURACY_CASE = _Listed(ACCURACY, "C.ACCURACY")

# exercise 1: months_per_stage
_M_HEAD = ("def months_per_stage(history):\n"
           "    totals = {}\n"
           "    for stage, months in history:\n"
           "        if stage not in totals:\n"
           "            totals[stage] = 0\n")
_M_END = "    return totals\n"


def _m(value: str) -> str:
    return _M_HEAD + f"        totals[stage] = {value}\n" + _M_END


_M_START = _m("...")
_M = _m("totals[stage] + months")
_M_REPLACE = _m("months")
_M_COUNT = _m("totals[stage] + 1")
_M_FIXED = ("def months_per_stage(history):\n"
            "    return {'inception': 1, 'design and development': 5, 'verification and validation': 2,\n"
            "            'deployment': 2, 'operation and monitoring': 23, 're-evaluation': 2, 'retirement': 1}\n")

# exercise 2: first_month_below
_B_HEAD = ("def first_month_below(accuracies, goal):\n"
           "    month = 0\n"
           "    for accuracy in accuracies:\n"
           "        month = month + 1\n")
_B_END = ("            return month\n"
          "    return None\n")


def _b(condition: str) -> str:
    return _B_HEAD + f"        if {condition}:\n" + _B_END


_B_START = _b("...")
_B = _b("accuracy < goal")
_B_ABOVE = _b("accuracy > goal")
_B_AT = _b("accuracy <= goal")
_B_FIXED = _b("accuracy < 0.9")

EXERCISES = {
    "months": FunctionBox(
        name="months",
        function="months_per_stage",
        start=_M_START,
        solution=_M,
        visible=[(HISTORY_CASE,)],
        hidden=[([],), ([("inception", 2)],), ([("deployment", 1), ("deployment", 1), ("deployment", 1)],),
                ([("inception", 0.5), ("design and development", 2), ("inception", 1.5)],),
                ([("operation and monitoring", 12), ("re-evaluation", 1), ("operation and monitoring", 6),
                  ("retirement", 2)],)],
        first=[_DOTS],
        picture=months_picture,
        mistakes=[
            Mistake("replace", wrong=_M_REPLACE,
                    en="`{call}` gives {got}, and it should give {want}: for a stage that comes back, your function keeps only the months of its last entry. Does the new value add the months of this entry to what `totals[stage]` already holds?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: bij een fase die terugkomt, houdt je functie alleen de maanden van de laatste regel. Telt de nieuwe waarde de maanden van deze regel op bij wat `totals[stage]` al bevat?"),
            Mistake("count", wrong=_M_COUNT,
                    en="`{call}` gives {got}, and it should give {want}: the number of entries of each stage, not its months. Which name holds the months of this entry?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het aantal regels van elke fase, niet de maanden. Welke naam bevat de maanden van deze regel?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the new value of `totals[stage]`: what it held so far, plus the `months` of this entry.",
                      "nl": "Vervang de `...` door de nieuwe waarde van `totals[stage]`: wat de waarde tot nu toe was, plus de `months` van deze regel."},
            "arguments": {"en": "The check calls `{call}`, with a list of entries, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(history)`?",
                          "nl": "De check roept `{call}` aan, met een lijst met regels, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(history)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function add up the months of every entry under its stage, also when a stage comes back?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Telt je functie de maanden van elke regel op bij de eigen fase, ook als een fase terugkomt?"},
            "almost": {"en": "Right for the life of the service. But `{call}` gives {got}, and it should give {want}. Does your function add up the entries of the `history` it gets, also halves of a month, and not give a fixed dict?",
                       "nl": "Goed voor het leven van de dienst. Maar `{call}` geeft {got}, en het moet {want} zijn. Telt je functie de regels op van de `history` die de functie krijgt, ook halve maanden, en geeft de functie geen vaste dict?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return totals`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return totals`?"},
            "correct": {"en": "Right in all {n} cases: the service spent 23 of its 36 months in operation and monitoring. In the picture, how many months went to inception, design and development, verification and validation, and deployment together?",
                        "nl": "Goed in alle {n} gevallen: de dienst was 23 van de 36 maanden in gebruik en monitoring. Hoeveel maanden gingen er in de afbeelding samen naar idee, ontwerp en ontwikkeling, verificatie en validatie, en deployment?"},
        },
    ),
    "below": FunctionBox(
        name="below",
        function="first_month_below",
        start=_B_START,
        solution=_B,
        visible=[(ACCURACY_CASE, GOAL)],
        hidden=[([0.92, 0.90, 0.85], 0.9), ([0.95, 0.96], 0.9), ([], 0.9), ([0.80, 0.95], 0.9),
                ([0.85, 0.82, 0.79], 0.8)],
        first=[_DOTS],
        picture=below_picture,
        mistakes=[
            Mistake("above", wrong=_B_ABOVE,
                    en="`{call}` gives {got}, and it should give {want}: the first month above the goal. Which comparison says that `accuracy` is below `goal`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de eerste maand boven het doel. Welke vergelijking zegt dat `accuracy` onder `goal` ligt?"),
            Mistake("fixed-goal", wrong=_B_FIXED,
                    en="`{call}` gives {got}, and it should give {want}: your function compares with 0.9, and this call has another goal. Which name holds the goal of the call?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie vergelijkt met 0,9, en deze aanroep heeft een ander doel. Welke naam bevat het doel van de aanroep?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the condition: the `accuracy` of this month is below the `goal`.",
                      "nl": "Vervang de `...` door de voorwaarde: de `accuracy` van deze maand ligt onder het `goal`."},
            "arguments": {"en": "The check calls `{call}`, with a list of accuracies and a goal, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(accuracies, goal)`?",
                          "nl": "De check roept `{call}` aan, met een lijst met accuracy's en een doel, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(accuracies, goal)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your condition let through only a month whose `accuracy` is below `goal`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Laat je voorwaarde alleen een maand door met een `accuracy` onder `goal`?"},
            "almost": {"en": "Right for the accuracy of the service. But `{call}` gives {got}, and it should give {want}. Is a month exactly at the goal below it?",
                       "nl": "Goed voor de accuracy van de dienst. Maar `{call}` geeft {got}, en het moet {want} zijn. Ligt een maand precies op het doel eronder?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still give back `month` inside the loop?",
                          "nl": "Met `{call}` geeft je functie niets terug. Geeft je functie in de loop nog `month` terug?"},
            "unreached": {"en": "With `{call}`, your function gives back `None` from the `return None` after the loop: inside the loop, it never gives back a `month`. Does your condition become true for a month whose `accuracy` is below `goal`, and does `return month` still follow it?",
                          "nl": "Met `{call}` geeft je functie `None` terug, van de `return None` na de loop: in de loop geeft je functie nooit een `month` terug. Wordt je voorwaarde waar bij een maand met een `accuracy` onder `goal`, en volgt `return month` er nog op?"},
            "correct": {"en": "Right in all {n} cases: month 13 of the first period of operation, month 19 of the life of the service. In the picture, could the accuracy of month 12 already have warned the team?",
                        "nl": "Goed in alle {n} gevallen: maand 13 van de eerste periode van gebruik, maand 19 van het leven van de dienst. Had de accuracy van maand 12 het team in de afbeelding al kunnen waarschuwen?"},
        },
    ),
}

# The lesson is done when these are correct. "flu" and "subject" are the choice questions after the demo.
REQUIRED = ["months", "below", "flu", "subject"]

CASES = [
    ("months", _M_START, "empty", None),
    ("months", _M, "correct", None),
    ("months", _m("totals[stage] + ..."), "wrong", "dots-left"),
    ("months", _M_REPLACE, "wrong", "replace"),
    ("months", _M_COUNT, "wrong", "count"),
    ("months", _M.replace("    return totals\n", ""), "wrong", "no-return"),
    # other right answers: the same sum written another way
    ("months", _m("months + totals[stage]"), "correct", None),
    ("months", "def months_per_stage(history):\n    totals = {}\n    for stage, months in history:\n"
               "        totals[stage] = totals.get(stage, 0) + months\n    return totals\n", "correct", None),
    # a fixed dict: right for the life of the service, wrong for every other history
    ("months", _M_FIXED, "almost", None),
    ("below", _B_START, "empty", None),
    ("below", _B, "correct", None),
    ("below", _b("accuracy < ..."), "wrong", "dots-left"),
    ("below", _B_ABOVE, "wrong", "above"),
    ("below", _B_FIXED, "wrong", "fixed-goal"),
    ("below", _B.replace("            return month\n", "            pass\n"), "wrong", "unreached"),
    ("below", _b("accuracy == goal"), "wrong", "unreached"),
    ("below", _b("goal > accuracy"), "correct", None),
    # a month exactly at the goal counted as below: right for the service, wrong for [0.92, 0.90, 0.85]
    ("below", _B_AT, "almost", None),
]
