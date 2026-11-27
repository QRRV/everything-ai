# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-ml-teams: who builds an ML system. The roles around a model in production (data
scientist, ML engineer, MLOps engineer, platform engineer), and four ways a company can organize them: teams by
function, one team per product, one data scientist from start to finish, and a data scientist with a shared platform.
The learner counts the handoffs in a plan, and how many models a platform needs before it pays off.

The plans and their days are invented by the course, to show where the work gets stuck in each way; the lesson's texts
say which facts come from Uber, Stitch Fix and the literature."""
import html
import math

from everything_ai import check, yours
from everything_ai.check import FunctionBox, Mistake

LESSON = "sys-ml-teams"
LANGS = ("en", "nl")

# The roles of the lesson. The names are English jargon in Dutch too.
ROLES = ("data scientist", "ML engineer", "MLOps engineer", "platform engineer")

# The steps of one new version of the spam model, and what someone does in each.
STEPS = ("data", "model", "package", "deploy")
STEP_NAMES = {"en": {"data": "data", "model": "model", "package": "package", "deploy": "deploy"},
              "nl": {"data": "data", "model": "model", "package": "pakket", "deploy": "deploy"}}
STEP_DOING = {"en": {"data": "collecting and labeling messages", "model": "training the model",
                     "package": "packaging it as a service", "deploy": "deploying it with monitoring"},
              "nl": {"data": "berichten verzamelen en labelen", "model": "het model trainen",
                     "package": "er een dienst van maken", "deploy": "deployen met monitoring"}}

# The four ways to organize the work, and the plan of one new version in each, as (step, who, days): invented.
WAYS = ("by function", "per product", "start to finish", "platform")
WAY_NAMES = {"en": {"by function": "teams by function", "per product": "one team per product",
                    "start to finish": "one data scientist from start to finish",
                    "platform": "a data scientist with a platform"},
             "nl": {"by function": "teams per functie", "per product": "één team per product",
                    "start to finish": "één data scientist van begin tot eind",
                    "platform": "een data scientist met een platform"}}
PLANS = {
    "by function": [("data", "data scientist", 10), ("model", "data scientist", 10), ("package", "ML engineer", 5),
                    ("deploy", "MLOps engineer", 3)],
    "per product": [("data", "data scientist", 10), ("model", "data scientist", 10), ("package", "ML engineer", 5),
                    ("deploy", "MLOps engineer", 3)],
    "start to finish": [("data", "data scientist", 10), ("model", "data scientist", 10),
                        ("package", "data scientist", 10), ("deploy", "data scientist", 8)],
    "platform": [("data", "data scientist", 10), ("model", "data scientist", 10), ("package", "data scientist", 3),
                 ("deploy", "data scientist", 1)],
}
# Days the work waits each time it passes to someone else: in the queue of another team, or of a colleague in the same team.
WAIT = {"by function": 10, "per product": 2, "start to finish": 0, "platform": 0}
PLAN = PLANS["by function"]
# The same plan, but the ML engineer sends the model back once: it used a feature that serving cannot compute.
BACK_AND_FORTH = [("data", "data scientist", 10), ("model", "data scientist", 10), ("package", "ML engineer", 5),
                  ("model", "data scientist", 3), ("package", "ML engineer", 2), ("deploy", "MLOps engineer", 3)]

# The platform: two platform engineers for about six months, before the first model uses it.
PLATFORM_DAYS = 240


def handoffs(plan):
    count = 0
    previous = None
    for step, who, days in plan:
        if previous is not None and who != previous:
            count = count + 1
        previous = who
    return count


def models_to_pay_off(platform_days, saved):
    models = 0
    gained = 0
    while gained < platform_days:
        models = models + 1
        gained = gained + saved
    return models


def timeline(way: str) -> list:
    """The plan of a way as segments (kind, step, who, start, end) in days, with a wait before each handoff."""
    out, t, previous = [], 0, None
    for step, who, days in PLANS[way]:
        if previous is not None and who != previous:
            out.append(("wait", step, who, t, t + WAIT[way]))
            t += WAIT[way]
        out.append(("work", step, who, t, t + days))
        t += days
        previous = who
    return out


TOTALS = {w: timeline(w)[-1][4] for w in WAYS}                         # 48, 32, 38, 24 days
WORK = {w: sum(d for _, _, d in PLANS[w]) for w in WAYS}               # 28, 28, 38, 24 days of work
WAITED = {w: TOTALS[w] - WORK[w] for w in WAYS}                        # 20, 4, 0, 0 days of waiting
SAVED = WORK["start to finish"] - WORK["platform"]                     # 14 days of a data scientist per model
PAYS_OFF = models_to_pay_off(PLATFORM_DAYS, SAVED)                     # 18 models
LAST_DAY = max(TOTALS.values())                                        # 48


# ======== words and drawing ===============================================================================================

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem;max-width:var(--ea-measure)"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem;max-width:var(--ea-measure)"
_SVG = "width:100%;max-width:330px;height:auto;font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums"


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma and a thousands dot in Dutch."""
    out = f"{x:,.{digits}f}"
    return out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out


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


def _arrow(x1: float, y1: float, x2: float, y2: float, ink: str = "var(--ea-ink-2)", width: float = 1.3) -> str:
    """A line from (x1, y1) to (x2, y2) with a small head at its end."""
    a = math.atan2(y2 - y1, x2 - x1)
    h1 = (x2 - 6 * math.cos(a - 0.45), y2 - 6 * math.sin(a - 0.45))
    h2 = (x2 - 6 * math.cos(a + 0.45), y2 - 6 * math.sin(a + 0.45))
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" style="stroke:{ink};stroke-width:{width}"/>'
            f'<path d="M{h1[0]:.1f},{h1[1]:.1f} L{x2:.1f},{y2:.1f} L{h2[0]:.1f},{h2[1]:.1f}" '
            f'style="fill:none;stroke:{ink};stroke-width:{width}"/>')


def _box(x: float, y: float, w: float, h: float, strong: bool = False, dashed: bool = False) -> str:
    stroke = "var(--ea-ink)" if strong else "var(--ea-ink-3)"
    dash = ";stroke-dasharray:4 3" if dashed else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="5" '
            f'style="fill:var(--ea-page);stroke:{stroke};stroke-width:{1.6 if strong else 1.2}{dash}"/>')


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


def way_name(way: str, lang: str) -> str:
    return WAY_NAMES[lang].get(way, way)


def step_name(step: str, lang: str) -> str:
    return STEP_NAMES[lang].get(step, str(step))


# ======== the picture of the idea =========================================================================================

IDEA_TEXT = {
    "en": {"steps": {"data scientist": "data, model", "ML engineer": "package", "MLOps engineer": "deploy",
                     "all": "data, model, package, deploy"},
           "wait": "waits", "team": "one team", "slow": "package and deploy take longer",
           "platform": "ML platform", "platform below": "built and kept by platform engineers, used by every team",
           "fast": "package and deploy with the platform's tools",
           "alt": "Four ways to organize the work on a model. Teams by function: data scientist, ML engineer and MLOps engineer in three teams, with a queue where the work waits between them. One team per product: the same three roles inside one team, with short waits. One data scientist from start to finish: one person does all four steps, and packaging and deploying take longer. A data scientist with a platform: one person does all four steps on top of an ML platform built by platform engineers."},
    "nl": {"steps": {"data scientist": "data, model", "ML engineer": "pakket", "MLOps engineer": "deploy",
                     "all": "data, model, pakket, deploy"},
           "wait": "wacht", "team": "één team", "slow": "pakket en deploy duren langer",
           "platform": "ML-platform", "platform below": "gebouwd en onderhouden door platform engineers, voor elk team",
           "fast": "pakket en deploy met de tools van het platform",
           "alt": "Vier manieren om het werk aan een model in te richten. Teams per functie: data scientist, ML engineer en MLOps engineer in drie teams, met een wachtrij waar het werk tussen de teams wacht. Eén team per product: dezelfde drie rollen in één team, met korte wachttijden. Eén data scientist van begin tot eind: één persoon doet alle vier de stappen, en pakket en deploy duren langer. Een data scientist met een platform: één persoon doet alle vier de stappen, op een ML-platform dat platform engineers bouwen."},
}


def _queue(x: float, y: float) -> str:
    """Three small stacked cards: work that waits."""
    return "".join(f'<rect x="{x + 2 * k:.1f}" y="{y - 2 * k:.1f}" width="9" height="7" rx="1" '
                   f'style="fill:var(--ea-page);stroke:var(--ea-ink-3);stroke-width:1"/>' for k in range(3))


def idea_picture(lang: str) -> str:
    """The four ways as small diagrams, from top to bottom."""
    w = IDEA_TEXT[lang]
    parts, y = [], 0.0
    widths = {"data scientist": 90.0, "ML engineer": 80.0, "MLOps engineer": 90.0}
    for way in WAYS:
        parts.append(_t(0, y + 13, way_name(way, lang), 12.5, 600))
        top = y + 22
        if way in ("by function", "per product"):
            gap = (330 - sum(widths.values()) - (12 if way == "per product" else 0)) / 2
            x = 6.0 if way == "per product" else 0.0
            if way == "per product":
                parts.append(f'<rect x="1" y="{top - 4:.1f}" width="328" height="50" rx="8" '
                             f'style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.2;stroke-dasharray:5 3"/>')
                parts.append(_t(326, top - 7, w["team"], 10.5, 400, "var(--ea-ink-2)", "end", halo=True))
            for k, role in enumerate(("data scientist", "ML engineer", "MLOps engineer")):
                bw = widths[role]
                parts.append(_box(x, top, bw, 42))
                parts.append(_t(x + bw / 2, top + 17, role, 10.5, 600, "var(--ea-ink)", "middle"))
                parts.append(_t(x + bw / 2, top + 32, w["steps"][role], 10.5, 400, "var(--ea-ink-2)", "middle"))
                if k < 2:
                    gx = x + bw
                    if way == "by function":
                        parts.append(_queue(gx + gap / 2 - 7, top + 12))
                        parts.append(_arrow(gx + 2, top + 26, gx + gap - 2, top + 26))
                        parts.append(_t(gx + gap / 2, top + 40, w["wait"], 9.5, 400, "var(--ea-ink-2)", "middle"))
                    else:
                        parts.append(_arrow(gx + 2, top + 21, gx + gap - 2, top + 21))
                    x = gx + gap
            y = top + 54
        elif way == "start to finish":
            parts.append(_box(0, top, 330, 54))
            parts.append(_t(165, top + 17, "data scientist", 10.5, 600, "var(--ea-ink)", "middle"))
            parts.append(_t(165, top + 32, w["steps"]["all"], 10.5, 400, "var(--ea-ink-2)", "middle"))
            parts.append(_t(165, top + 46, w["slow"], 10.5, 400, "var(--ea-ink-2)", "middle"))
            y = top + 66
        else:
            parts.append(_box(0, top, 330, 54, strong=True))
            parts.append(_t(165, top + 17, "data scientist", 10.5, 600, "var(--ea-ink)", "middle"))
            parts.append(_t(165, top + 32, w["steps"]["all"], 10.5, 400, "var(--ea-ink-2)", "middle"))
            parts.append(_t(165, top + 46, w["fast"], 10.5, 400, "var(--ea-ink-2)", "middle"))
            py = top + 62
            parts.append(f'<rect x="0" y="{py:.1f}" width="330" height="40" rx="5" '
                         f'style="fill:var(--ea-line);stroke:var(--ea-ink-3);stroke-width:1.2"/>')
            parts.append(_t(165, py + 16, w["platform"], 11.5, 600, "var(--ea-ink)", "middle"))
            parts.append(_t(165, py + 31, w["platform below"], 10.5, 400, "var(--ea-ink-2)", "middle"))
            y = py + 44
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(w["alt"])}" style="margin:0.25rem 0 0.75rem;'
            f'max-width:min(100%,330px)"><svg viewBox="0 0 330 {y:.0f}" aria-hidden="true" style="{_SVG}">'
            f'{"".join(parts)}</svg></figure>')


# ======== drawing a plan and a payoff =====================================================================================

def plan_svg(plan: list, title: str, mine, lang: str, alt: str) -> str:
    """A lane per person in the plan, a box per step in order, and an arrow with a number at each handoff."""
    w = _YOURS_TEXT[lang]["handoffs"]
    people = []
    for _, who, _ in plan:
        if who not in people:
            people.append(who)
    LBL, LANE, top = 52.0, 30.0, 26.0
    n = max(len(plan), 1)
    cw = (330 - LBL) / n
    parts = [_t(0, 14, title, 12, 600)]
    for r, who in enumerate(people):
        y = top + r * LANE
        words = who.split(" ", 1)
        parts.append(_t(0, y + 13, words[0], 10, 600, "var(--ea-ink-2)"))
        parts.append(_t(0, y + 25, words[1] if len(words) > 1 else "", 10, 400, "var(--ea-ink-2)"))
        parts.append(f'<line x1="{LBL:.1f}" y1="{y + LANE:.1f}" x2="330" y2="{y + LANE:.1f}" style="stroke:var(--ea-line);stroke-width:1"/>')
    boxes, arrows = [], 0
    for i, (step, who, _) in enumerate(plan):
        r = people.index(who)
        x, y = LBL + i * cw + 3, top + r * LANE + 5
        boxes.append((x, y, cw - 6))
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw - 6:.1f}" height="20" rx="3" '
                     f'style="fill:var(--ea-page);stroke:var(--ea-ink-3);stroke-width:1.2"/>')
        parts.append(_t(x + (cw - 6) / 2, y + 14, step_name(step, lang), 9.5, 400, "var(--ea-ink)", "middle"))
    for i in range(1, len(plan)):
        (x0, y0, b0), (x1, y1, _) = boxes[i - 1], boxes[i]
        if plan[i][1] == plan[i - 1][1]:
            parts.append(f'<line x1="{x0 + b0:.1f}" y1="{y0 + 10:.1f}" x2="{x1:.1f}" y2="{y1 + 10:.1f}" '
                         f'style="stroke:var(--ea-ink-3);stroke-width:1.2"/>')
        else:
            arrows += 1
            ya, yb = (y0 + 20, y1) if y1 > y0 else (y0, y1 + 20)
            parts.append(_arrow(x0 + b0 - 4, ya, x1 + 4, yb, "var(--ea-ink)", 1.4))
            parts.append(_t((x0 + b0 + x1) / 2 + 7, (ya + yb) / 2 + 2, str(arrows), 10.5, 600, "var(--ea-ink)", "middle", halo=True))
    y = top + len(people) * LANE + 18
    says = w["mine"].format(n=mine) if mine is not None else ""
    parts.append(_t(0, y, f'{says}{w["arrows"].format(m=arrows)}', 11.5, 600))
    return f'<svg viewBox="0 0 330 {y + 6:.0f}" role="img" aria-label="{html.escape(alt)}" style="{_SVG}">{"".join(parts)}</svg>'


def payoff_svg(cost: float, saved: float, mine: int, lang: str, alt: str) -> str:
    """The days the data scientists save after each model, the cost of the platform as a line, and one model marked."""
    w = _YOURS_TEXT[lang]["payoff"]
    N = max(24, mine + 3)
    L_, R_, T_, B_ = 34.0, 322.0, 10.0, 130.0
    top = max(cost * 1.5, saved * N * 1.05, 1)

    def px(n):
        return L_ + (R_ - L_) * n / N

    def py(v):
        return B_ - (B_ - T_) * min(v, top) / top

    parts = []
    step = 120 if top > 240 else 60
    v = 0
    while v <= top:
        parts.append(_t(L_ - 6, py(v) + 4, number(v, 0, lang), 10, 400, "var(--ea-ink-3)", "end"))
        parts.append(f'<line x1="{L_:.1f}" y1="{py(v):.1f}" x2="{R_:.1f}" y2="{py(v):.1f}" style="stroke:var(--ea-line);stroke-width:1"/>')
        v += step
    parts.append(f'<line x1="{L_:.1f}" y1="{py(cost):.1f}" x2="{R_:.1f}" y2="{py(cost):.1f}" '
                 f'style="stroke:var(--ea-ink);stroke-width:1.4;stroke-dasharray:5 3"/>')
    parts.append(_t(L_ + 4, py(cost) - 6, w["cost"].format(c=number(cost, 0, lang)), 11, 600, "var(--ea-ink)", "start", halo=True))
    for n in range(1, N + 1):
        strong = n == mine
        fill = "var(--ea-ink)" if strong else ("var(--ea-ink-3)" if n < mine else "var(--ea-line)")
        stroke = "" if n <= mine else ";stroke:var(--ea-ink-3);stroke-width:0.8"
        parts.append(f'<circle cx="{px(n):.1f}" cy="{py(saved * n):.1f}" r="{4.2 if strong else 2.4}" style="fill:{fill}{stroke}"/>')
    if mine >= 1:
        x = px(mine)
        parts.append(f'<line x1="{x:.1f}" y1="{py(saved * mine) + 6:.1f}" x2="{x:.1f}" y2="{B_ + 3:.1f}" '
                     f'style="stroke:var(--ea-ink);stroke-width:1"/>')
        parts.append(_t(min(max(x, L_ + 50), R_ - 50), B_ + 30, w["mine"].format(m=mine), 11.5, 600, "var(--ea-ink)", "middle"))
    for n in range(0, N + 1, 6):
        parts.append(_t(px(n), B_ + 14, str(n), 10, 400, "var(--ea-ink-3)", "middle"))
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


def _count(v, most: int):
    """v as a whole number from 0 to most, or None: never from text, floats or True and False."""
    if isinstance(v, bool) or not isinstance(v, int) or v < 0 or v > most:
        return None
    return v


_YOURS_TEXT = {
    "en": {"handoffs": {"title": "Your handoffs on two plans",
                        "intro": "Your `handoffs` on `C.PLAN`, and on `C.BACK_AND_FORTH`: the same plan, but the ML engineer sends the model back once, because it uses a feature that serving cannot compute.",
                        "plan": "C.PLAN, teams by function", "back": "C.BACK_AND_FORTH, with a step back",
                        "mine": "your function: {n}; ", "arrows": "arrows to someone else: {m}",
                        "alt": "A lane per person and a box per step of the plan, in order, with a numbered arrow each time the work passes to someone else.",
                        "none": "Your `handoffs` stops with an error, takes too long, or does not give a whole number of handoffs, so nothing can be drawn yet."},
           "payoff": {"title": "Your models_to_pay_off for the platform",
                      "intro": "Your `models_to_pay_off(240, 14)`: a platform of 240 days of work, and 14 days of a data scientist saved on each model.",
                      "cost": "platform: {c} days", "mine": "your function: {m} models",
                      "alt": "Dots for the days the data scientists have saved after each model, rising by 14 a model, a dashed line at the 240 days the platform cost, and the number of models your function gives marked.",
                      "says": "After {m} models, the data scientists have saved {g} days, and the platform cost {c}.",
                      "none": "Your `models_to_pay_off` stops with an error, takes too long, or does not give a whole number of models, so nothing can be drawn yet."}},
    "nl": {"handoffs": {"title": "Je handoffs op twee plannen",
                        "intro": "Je `handoffs` op `C.PLAN`, en op `C.BACK_AND_FORTH`: hetzelfde plan, alleen stuurt de ML engineer het model één keer terug, omdat het een feature gebruikt die serving niet kan berekenen.",
                        "plan": "C.PLAN, teams per functie", "back": "C.BACK_AND_FORTH, met een stap terug",
                        "mine": "je functie: {n}; ", "arrows": "pijlen naar iemand anders: {m}",
                        "alt": "Een baan per persoon en een vak per stap van het plan, op volgorde, met een genummerde pijl elke keer dat het werk naar iemand anders gaat.",
                        "none": "Je `handoffs` stopt met een fout, duurt te lang, of geeft geen geheel aantal overdrachten, dus is er nog geen afbeelding."},
           "payoff": {"title": "Je models_to_pay_off voor het platform",
                      "intro": "Je `models_to_pay_off(240, 14)`: een platform van 240 dagen werk, en bij elk model 14 dagen van een data scientist bespaard.",
                      "cost": "platform: {c} dagen", "mine": "je functie: {m} modellen",
                      "alt": "Punten voor de dagen die de data scientists na elk model bespaard hebben, 14 erbij per model, een lijn van streepjes bij de 240 dagen die het platform kostte, en het aantal modellen dat je functie geeft gemarkeerd.",
                      "says": "Na {m} modellen hebben de data scientists {g} dagen bespaard, en het platform kostte {c}.",
                      "none": "Je `models_to_pay_off` stopt met een fout, duurt te lang, of geeft geen geheel aantal modellen, dus is er nog geen afbeelding."}},
}


def handoffs_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["handoffs"]
    fn = _fn(seen)
    got = []
    for plan in (PLAN, BACK_AND_FORTH):
        v = _call(lambda p=plan: fn([tuple(s) for s in p])) if fn is not None else None
        got.append(_count(v, 100) if v is not _STOPPED else None)
    if fn is None or any(g is None for g in got):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    body = (_note(w["intro"]) + plan_svg(PLAN, w["plan"], got[0], lang, w["alt"])
            + plan_svg(BACK_AND_FORTH, w["back"], got[1], lang, w["alt"]))
    return yours.figure(w["title"], body, alt=w["title"])


def payoff_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["payoff"]
    fn = _fn(seen)
    v = _call(lambda: fn(PLATFORM_DAYS, SAVED)) if fn is not None else None
    got = _count(v, 60) if v is not _STOPPED else None
    if fn is None or got is None:
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    svg = payoff_svg(PLATFORM_DAYS, SAVED, got, lang, w["alt"])
    says = w["says"].format(m=got, g=number(got * SAVED, 0, lang), c=number(PLATFORM_DAYS, 0, lang))
    return yours.figure(w["title"], _note(w["intro"]) + svg + _note(says), alt=w["title"])


# ======== the demo: one new version, day by day, in the four ways =========================================================

DEMO_TEXT = {
    "en": {"title": "Day {d} of {n}", "days": "{n} days",
           "work": "now: {doing} ({who})", "wait": "now: waiting for the {who}, day {k} of {n}",
           "done": "done on day {d}: the new version answers",
           "alt": "Day {d}: four bars, one for each way to organize the work, with the days of work filled in and the days of waiting as dashed boxes, a line at the current day, and under each bar what happens now.",
           "same": "Day {d}: in all four ways, the data scientist is {doing}.",
           "waited": "Day {d}: so far, the work has waited {a} days with teams by function, and {b} days in one team per product.",
           "says": {
               1: "Day 1: four companies start on the same job, a new version of their spam model, each organized in one of the four ways. For the first 20 days they look alike: a data scientist collects and labels new messages, then trains the model.",
               21: "Day 21: the model is ready, and the four ways part. With teams by function, the model goes into the queue of the ML engineering team. In one team per product, the ML engineer of the same team can start after 2 days. The other two data scientists start packaging the model themselves.",
               24: "Day 24: with a platform, the new version answers. The data scientist packaged it in 3 days and deployed it in 1, with the tools of the platform.",
               31: "Day 31: with teams by function, the ML engineer starts after 10 days in the queue. Nobody worked on the model in those days, and the data scientist has started on something else.",
               32: "Day 32: in one team per product, the new version answers: two handoffs of 2 days each, inside one team.",
               38: "Day 38: the data scientist from start to finish is done. Nothing was handed off, but packaging and deploying took 18 days, where the specialists need 8.",
               46: "Day 46: with teams by function, the MLOps engineer starts, after a second wait of 10 days.",
               48: "Day 48: with teams by function, the new version answers at last: 28 days of work and 20 days of waiting.",
           }},
    "nl": {"title": "Dag {d} van {n}", "days": "{n} dagen",
           "work": "nu: {doing} ({who})", "wait": "nu: wachten op de {who}, dag {k} van {n}",
           "done": "klaar op dag {d}: de nieuwe versie antwoordt",
           "alt": "Dag {d}: vier balken, één voor elke manier om het werk in te richten, met de dagen werk ingekleurd en de dagen wachten als vakken met een rand van streepjes, een lijn bij de huidige dag, en onder elke balk wat er nu gebeurt.",
           "same": "Dag {d}: bij alle vier de manieren is de data scientist bezig met {doing}.",
           "waited": "Dag {d}: tot nu toe heeft het werk {a} dagen gewacht bij teams per functie, en {b} dagen bij één team per product.",
           "says": {
               1: "Dag 1: vier bedrijven beginnen aan dezelfde klus, een nieuwe versie van het eigen spammodel, elk ingericht op een van de vier manieren. De eerste 20 dagen lijken de vier bedrijven op elkaar: een data scientist verzamelt en labelt nieuwe berichten en traint daarna het model.",
               21: "Dag 21: het model is klaar, en de vier manieren gaan uiteen. Bij teams per functie gaat het model in de wachtrij van het team van ML engineers. Bij één team per product kan de ML engineer van hetzelfde team na 2 dagen beginnen. De andere twee data scientists maken zelf een dienst van het model.",
               24: "Dag 24: met een platform antwoordt de nieuwe versie. De data scientist maakte er in 3 dagen een dienst van en deployde de dienst in 1 dag, met de tools van het platform.",
               31: "Dag 31: bij teams per functie begint de ML engineer, na 10 dagen in de wachtrij. In deze 10 dagen werkte niemand aan het model, en de data scientist is aan iets anders begonnen.",
               32: "Dag 32: bij één team per product antwoordt de nieuwe versie: twee overdrachten van elk 2 dagen, binnen één team.",
               38: "Dag 38: de data scientist van begin tot eind is klaar. Er waren geen overdrachten, en toch kostte een dienst maken en deployen 18 dagen, waar de specialisten 8 dagen nodig hebben.",
               46: "Dag 46: bij teams per functie begint de MLOps engineer, na een tweede wachttijd van 10 dagen.",
               48: "Dag 48: bij teams per functie antwoordt de nieuwe versie eindelijk: 28 dagen werk en 20 dagen wachten.",
           }},
}


def _at(way: str, d: int):
    """The segment of a way on day d (1 is the first day), or None when the way is done."""
    for seg in timeline(way):
        if seg[3] < d <= seg[4]:
            return seg
    return None


def _waited(way: str, d: int) -> int:
    return sum(max(0, min(d, b) - a) for kind, _, _, a, b in timeline(way) if kind == "wait")


def say(d: int, lang: str) -> str:
    """What happens on day d."""
    w = DEMO_TEXT[lang]
    d = max(1, min(LAST_DAY, int(d)))
    if d in w["says"]:
        return w["says"][d]
    if d <= 20:
        return w["same"].format(d=d, doing=STEP_DOING[lang][_at("platform", d)[1]])
    return w["waited"].format(d=d, a=_waited("by function", d), b=_waited("per product", d))


def frame(d: int, lang: str) -> tuple[str, str]:
    """Day d (1 to 48): a bar per way with the days of work and of waiting, a line at the day, and what happens now."""
    w = DEMO_TEXT[lang]
    d = max(1, min(LAST_DAY, int(d)))
    X0, X1 = 4.0, 326.0
    S = (X1 - X0) / LAST_DAY
    parts = [_t(0, 14, w["title"].format(d=d, n=LAST_DAY), 13, 600)]
    for k in range(0, LAST_DAY + 1, 12):
        anchor = "start" if k == 0 else ("end" if k == LAST_DAY else "middle")
        parts.append(_t(X0 + k * S + (2 if k == LAST_DAY else 0), 30, str(k), 10, 400, "var(--ea-ink-3)", anchor))
    top, LANE = 40.0, 54.0
    x_now = X0 + (d - 0.5) * S
    bottom = top + (len(WAYS) - 1) * LANE + 17 + 12
    parts.append(f'<line x1="{x_now:.1f}" y1="{top - 4:.1f}" x2="{x_now:.1f}" y2="{bottom + 2:.1f}" '
                 f'style="stroke:var(--ea-ink);stroke-width:1.2"/>')
    for i, way in enumerate(WAYS):
        y = top + i * LANE
        running = d < TOTALS[way]
        parts.append(_t(0, y + 11, way_name(way, lang), 11.5, 600 if running else 400,
                        "var(--ea-ink)" if running else "var(--ea-ink-2)", halo=True))
        parts.append(_t(330, y + 11, w["days"].format(n=TOTALS[way]), 11, 400, "var(--ea-ink-2)", "end", halo=True))
        by = y + 17
        for kind, step, who, a, b in timeline(way):
            if kind == "wait":
                ink = "var(--ea-ink-3)" if a < d else "var(--ea-line)"
                parts.append(f'<rect x="{X0 + a * S + 0.5:.1f}" y="{by + 0.5:.1f}" width="{(b - a) * S - 1:.1f}" height="11" '
                             f'rx="1.5" style="fill:none;stroke:{ink};stroke-width:1;stroke-dasharray:3 2"/>')
                continue
            spans = [(a, min(b, d - 1), "var(--ea-ink-3)"), (max(a, d - 1), min(b, d), "var(--ea-ink)"),
                     (max(a, d), b, "var(--ea-line)")]
            for s0, s1, fill in spans:
                if s1 > s0:
                    parts.append(f'<rect x="{X0 + s0 * S:.1f}" y="{by:.1f}" width="{(s1 - s0) * S:.1f}" height="12" '
                                 f'style="fill:{fill}"/>')
            parts.append(f'<line x1="{X0 + b * S:.1f}" y1="{by:.1f}" x2="{X0 + b * S:.1f}" y2="{by + 12:.1f}" '
                         f'style="stroke:var(--ea-page);stroke-width:1.2"/>')
        seg = _at(way, d)
        if seg is None or d == TOTALS[way]:
            status, weight, ink = w["done"].format(d=TOTALS[way]), 600, "var(--ea-ink)"
        elif seg[0] == "wait":
            status, weight, ink = w["wait"].format(who=seg[2], k=d - seg[3], n=seg[4] - seg[3]), 400, "var(--ea-ink-2)"
        else:
            status, weight, ink = w["work"].format(doing=STEP_DOING[lang][seg[1]], who=seg[2]), 400, "var(--ea-ink-2)"
        parts.append(_t(0, by + 27, status, 11, weight, ink, halo=True))
    height = top + len(WAYS) * LANE - 4
    svg = (f'<svg viewBox="0 0 330 {height:.0f}" role="img" aria-label="{html.escape(w["alt"].format(d=d))}" '
           f'style="{_SVG}">{"".join(parts)}</svg>')
    return svg, say(d, lang)


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
    """A list of the lesson that the check shows by its name: C.PLAN."""

    def __init__(self, items, name: str):
        super().__init__(items)
        self._name = name

    def __repr__(self) -> str:
        return self._name

    def __deepcopy__(self, memo):
        return list(self)


PLAN_CASE = _Listed([tuple(s) for s in PLAN], "C.PLAN")

# exercise 1: handoffs
_H_HEAD = ("def handoffs(plan):\n"
           "    count = 0\n"
           "    previous = None\n"
           "    for step, who, days in plan:\n")
_H_END = ("            count = count + 1\n"
          "        previous = who\n"
          "    return count\n")


def _h(condition: str) -> str:
    return _H_HEAD + f"        if previous is not None and {condition}:\n" + _H_END


_H_START = _h("...")
_H = _h("who != previous")
_H_SAME = _h("who == previous")
_H_STEP = _h("step != previous")
_H_PEOPLE = ("def handoffs(plan):\n"
             "    people = []\n"
             "    for step, who, days in plan:\n"
             "        if who not in people:\n"
             "            people.append(who)\n"
             "    return max(len(people) - 1, 0)\n")

# exercise 2: models_to_pay_off
_P_HEAD = ("def models_to_pay_off(platform_days, saved):\n"
           "    models = 0\n"
           "    gained = 0\n")
_P_END = ("        models = models + 1\n"
          "        gained = gained + saved\n"
          "    return models\n")


def _p(condition: str) -> str:
    return _P_HEAD + f"    while {condition}:\n" + _P_END


_P_START = _p("...")
_P = _p("gained < platform_days")
_P_NEVER = _p("gained > platform_days")
_P_MODELS = _p("models < platform_days")
_P_FIXED = _p("gained < 240")
_P_AT = _p("gained <= platform_days")

EXERCISES = {
    "handoffs": FunctionBox(
        name="handoffs",
        function="handoffs",
        start=_H_START,
        solution=_H,
        visible=[(PLAN_CASE,)],
        hidden=[([],), ([("data", "data scientist", 10)],), ([tuple(s) for s in PLANS["start to finish"]],),
                ([tuple(s) for s in BACK_AND_FORTH],),
                ([("data", "data engineer", 4), ("model", "data scientist", 6), ("data", "data engineer", 2),
                  ("model", "data scientist", 3)],)],
        first=[_DOTS],
        picture=handoffs_picture,
        mistakes=[
            Mistake("same", wrong=_H_SAME,
                    en="`{call}` gives {got}, and it should give {want}: your function counts the steps that stay with the same person. Which comparison says that `who` is not the same person as `previous`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie telt de stappen die bij dezelfde persoon blijven. Welke vergelijking zegt dat `who` iemand anders is dan `previous`?"),
            Mistake("step", wrong=_H_STEP,
                    en="`{call}` gives {got}, and it should give {want}: your function compares the name of a step with a person, and those always differ. Which name holds the person of this step?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie vergelijkt de naam van een stap met een persoon, en de twee verschillen altijd. Welke naam bevat de persoon van deze stap?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the condition: the person of this step, `who`, is not the same as the person of the step before, `previous`.",
                      "nl": "Vervang de `...` door de voorwaarde: de persoon van deze stap, `who`, is iemand anders dan de persoon van de vorige stap, `previous`."},
            "arguments": {"en": "The check calls `{call}`, with one plan, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(plan)`?",
                          "nl": "De check roept `{call}` aan, met één plan, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(plan)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function count each step whose `who` differs from the `who` of the step before?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Telt je functie elke stap waarvan `who` verschilt van de `who` van de vorige stap?"},
            "almost": {"en": "Right for `C.PLAN`. But `{call}` gives {got}, and it should give {want}. What if the work goes back to someone it was with before?",
                       "nl": "Goed voor `C.PLAN`. Maar `{call}` geeft {got}, en het moet {want} zijn. Wat als het werk teruggaat naar iemand bij wie het al eerder was?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return count`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return count`?"},
            "correct": {"en": "Right in all {n} cases: `C.PLAN` has 2 handoffs, and the plan with a step back has 4. How many days does the work wait in the plan with a step back, at 10 days a handoff?",
                        "nl": "Goed in alle {n} gevallen: `C.PLAN` heeft 2 overdrachten, en het plan met een stap terug heeft er 4. Hoeveel dagen wacht het werk in het plan met een stap terug, bij 10 dagen per overdracht?"},
        },
    ),
    "payoff": FunctionBox(
        name="payoff",
        function="models_to_pay_off",
        start=_P_START,
        solution=_P,
        visible=[(PLATFORM_DAYS, SAVED)],
        hidden=[(100, 10), (0, 14), (10, 14), (50, 7), (240, 4)],
        first=[_DOTS],
        picture=payoff_picture,
        mistakes=[
            Mistake("never", wrong=_P_NEVER,
                    en="`{call}` gives {got}, and it should give {want}: at the start, `gained` is 0, so your condition is false and the loop never runs. Should the loop go on while `gained` is still below `platform_days`, or above it?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: aan het begin is `gained` 0, dus is je voorwaarde onwaar en draait de loop nooit. Moet de loop doorgaan zolang `gained` nog onder `platform_days` ligt, of erboven?"),
            Mistake("models", wrong=_P_MODELS,
                    en="`{call}` gives {got}, and it should give {want}: your loop goes on until the number of models reaches the days of the platform. Which name holds the days saved so far?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je loop gaat door tot het aantal modellen het aantal dagen van het platform haalt. Welke naam bevat de dagen die tot nu toe bespaard zijn?"),
            Mistake("fixed-cost", wrong=_P_FIXED,
                    en="`{call}` gives {got}, and it should give {want}: your loop compares with 240, and this call has another platform. Which name holds the days of the platform of the call?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je loop vergelijkt met 240, en deze aanroep heeft een ander platform. Welke naam bevat de dagen van het platform van de aanroep?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the condition of the loop: the days saved so far, `gained`, are still fewer than `platform_days`.",
                      "nl": "Vervang de `...` door de voorwaarde van de loop: de dagen die tot nu toe bespaard zijn, `gained`, zijn nog minder dan `platform_days`."},
            "arguments": {"en": "The check calls `{call}`, with the days of the platform and the days saved per model, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(platform_days, saved)`?",
                          "nl": "De check roept `{call}` aan, met de dagen van het platform en de bespaarde dagen per model, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(platform_days, saved)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your loop add a model as long as `gained` is still below `platform_days`, and stop as soon as it is not?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Telt je loop een model op zolang `gained` nog onder `platform_days` ligt, en stopt de loop zodra dat niet meer zo is?"},
            "almost": {"en": "Right for the platform of the lesson. But `{call}` gives {got}, and it should give {want}. When `gained` is exactly as large as `platform_days`, has the platform paid off?",
                       "nl": "Goed voor het platform van de les. Maar `{call}` geeft {got}, en het moet {want} zijn. Als `gained` precies even groot is als `platform_days`, heeft het platform zich dan terugbetaald?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return models`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return models`?"},
            "correct": {"en": "Right in all {n} cases: the platform pays off from the 18th model on. In the picture, how many days have the data scientists saved after 24 models, the last dot, and how many more is that than the 240 the platform cost?",
                        "nl": "Goed in alle {n} gevallen: het platform betaalt zich terug vanaf het 18e model. Hoeveel dagen hebben de data scientists na 24 modellen bespaard, bij het laatste punt in de afbeelding, en hoeveel is dat meer dan de 240 die het platform kostte?"},
        },
    ),
}

# The lesson is done when these are correct. "role" and "stuck" are the choice questions after the demo.
REQUIRED = ["handoffs", "payoff", "role", "stuck"]

CASES = [
    ("handoffs", _H_START, "empty", None),
    ("handoffs", _H, "correct", None),
    ("handoffs", _h("who != ..."), "wrong", "dots-left"),
    ("handoffs", _H_SAME, "wrong", "same"),
    ("handoffs", _H_STEP, "wrong", "step"),
    ("handoffs", _H_PEOPLE, "almost", None),
    ("handoffs", _h("previous != who"), "correct", None),
    ("handoffs", _h("not who == previous"), "correct", None),
    ("handoffs", _H.replace("    return count\n", "    print(count)\n"), "wrong", "prints"),
    ("payoff", _P_START, "empty", None),
    ("payoff", _P, "correct", None),
    ("payoff", _p("gained < ..."), "wrong", "dots-left"),
    ("payoff", _P_NEVER, "wrong", "never"),
    ("payoff", _P_MODELS, "wrong", "models"),
    ("payoff", _P_FIXED, "wrong", "fixed-cost"),
    ("payoff", _P_AT, "almost", None),
    ("payoff", _p("platform_days > gained"), "correct", None),
    ("payoff", _p("not gained >= platform_days"), "correct", None),
]
