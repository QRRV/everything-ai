# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-framing: framing the problem for production. The same wish can become a classification,
a regression or a ranking, and the choice decides what you can change later without training again. With two goals,
one model per goal and a weight in code let a team change the balance at once. The learner picks the app with the
highest score, as in Huyen's example of the next app, and combines the scores of two models with a weight; the demo
moves that weight over the feed of the city's app.

The views, the apps, the scores and the posts are invented by the course; the lesson's texts say which facts come
from Huyen, Google and YouTube."""
import html
import math

from everything_ai import check, yours
from everything_ai.check import FunctionBox, Mistake

LESSON = "sys-framing"
LANGS = ("en", "nl")

# The worked example: the views a regression model predicts for 8 new videos in their first week. From the
# threshold on, a video goes into the expensive cache, as in Google's course on problem framing.
VIEWS = [12, 95, 47, 51, 230, 8, 64, 33]
THRESHOLDS = (50, 80)
CACHED = {t: sum(1 for v in VIEWS if v >= t) for t in THRESHOLDS}     # 4 and 2

# Exercise 1: one score per app, from a model that gets the user, the moment and one app, and gives how likely the
# user opens that app next (Huyen's example, framed as a regression). App names read the same in English and Dutch.
APPS = ["Mail", "Chat", "Radio", "Camera", "Bank", "Taxi", "Podcast", "Game"]
SCORES = [0.12, 0.64, 0.31, 0.08, 0.22, 0.45, 0.27, 0.05]
NEW_APP = "Bus"          # an app the user installs after the model was trained
NEW_SCORE = 0.81


def next_app(apps, scores):
    best = 0
    for i in range(len(apps)):
        if scores[i] > scores[best]:
            best = i
    return apps[best]


PICK = next_app(APPS, SCORES)                                   # Chat
PICK_NEW = next_app(APPS + [NEW_APP], SCORES + [NEW_SCORE])     # Bus

# Exercise 2 and the demo: the feed of the city's app. For each post, a quality model gives how trustworthy it is,
# and an engagement model how likely a user reacts to it, both from 0 to 1. Three posts are misleading.
POSTS = ("bank", "food", "bike", "budget", "storm", "water", "library", "soup")
QUALITY = [0.10, 0.05, 0.80, 0.95, 0.70, 0.22, 0.86, 0.62]
ENGAGEMENT = [0.95, 0.88, 0.56, 0.20, 0.74, 0.81, 0.36, 0.61]
MISLEADING = {"bank", "food", "water"}
ALPHA = 0.3            # the weight of quality in the check of exercise 2
ALPHA_HIGH = 0.8       # the second weight in the picture of exercise 2
TITLES = {
    "en": {"bank": "Ten tricks banks hide", "food": "One food cures it all",
           "bike": "New bike lanes on Monday", "budget": "Where city money goes",
           "storm": "Last night's storm", "water": "What is in tap water?",
           "library": "Library open on Sundays", "soup": "Soup in 20 minutes"},
    "nl": {"bank": "Wat banken verzwijgen", "food": "Eén voedsel geneest alles",
           "bike": "Nieuwe fietspaden open", "budget": "Waar het stadsgeld blijft",
           "storm": "De storm van vannacht", "water": "Wat zit er in kraanwater?",
           "library": "Bieb ook open op zondag", "soup": "Soep in 20 minuten"},
}


def combined(quality, engagement, alpha):
    scores = []
    for i in range(len(quality)):
        scores.append(alpha * quality[i] + (1 - alpha) * engagement[i])
    return scores


def ranked(scores: list) -> list:
    """The posts from the highest score to the lowest."""
    return [POSTS[i] for i in sorted(range(len(POSTS)), key=lambda i: -scores[i])]


ALPHAS = [k / 10 for k in range(11)]
FEEDS = {a: ranked(combined(QUALITY, ENGAGEMENT, a)) for a in ALPHAS}


def misleading_top3(order: list) -> int:
    return sum(1 for p in order[:3] if p in MISLEADING)


def mean_top3(order: list, values: list) -> float:
    return sum(values[POSTS.index(p)] for p in order[:3]) / 3


CLEAN_FROM = min(a for a in ALPHAS if misleading_top3(FEEDS[a]) == 0)      # 0.4: no misleading post in the top 3


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
    """A number with at most `digits` decimals, without zeros at its end: 0.3, 0.73, 1."""
    out = f"{round(x, digits) + 0.0:.{digits}f}".rstrip("0").rstrip(".") if digits else f"{round(x):d}"
    out = "0" if out in ("-0", "") else out
    return (out.replace(".", ",") if lang == "nl" else out).replace("-", "−")


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


def _box(x: float, y: float, w: float, h: float, stroke: str = "var(--ea-ink-3)", width: float = 1.2) -> str:
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="5" '
            f'style="fill:var(--ea-page);stroke:{stroke};stroke-width:{width}"/>')


def _arrow(x1: float, y1: float, x2: float, y2: float, ink: str = "var(--ea-ink-2)", width: float = 1.3) -> str:
    """A line from (x1, y1) to (x2, y2) with a small head at its end."""
    a = math.atan2(y2 - y1, x2 - x1)
    h1 = (x2 - 6 * math.cos(a - 0.45), y2 - 6 * math.sin(a - 0.45))
    h2 = (x2 - 6 * math.cos(a + 0.45), y2 - 6 * math.sin(a + 0.45))
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" style="stroke:{ink};stroke-width:{width}"/>'
            f'<path d="M{h1[0]:.1f},{h1[1]:.1f} L{x2:.1f},{y2:.1f} L{h2[0]:.1f},{h2[1]:.1f}" '
            f'style="fill:none;stroke:{ink};stroke-width:{width}"/>')


def _svg(height: float, alt: str, parts: list) -> str:
    return (f'<svg viewBox="0 0 330 {height:.0f}" role="img" aria-label="{html.escape(alt)}" style="{_SVG}">'
            f'{"".join(parts)}</svg>')


# ======== the picture of the idea =========================================================================================

IDEA_TEXT = {
    "en": {"title": "Two goals for one feed", "one": "One model, one combined loss", "two": "Two models, one weight in code",
           "posts": "posts", "feed": "feed", "model": "model", "trained": "trained on",
           "loss": "α × quality loss", "loss2": "+ (1 − α) × engagement loss",
           "for": "model for", "quality": "quality", "engagement": "engagement",
           "code": "α × quality", "code2": "+ (1 − α) × engagement", "in_code": "in code",
           "again": "A new α: train the model again.", "now": "A new α: change one number, train nothing.",
           "foot": "An orange edge marks what training learns.",
           "alt": "Two ways to serve a feed with two goals. Above, one model trained on one combined loss with the weight α inside it: a new α means training again. Below, a quality model and an engagement model, and their scores combined with the weight α in code: a new α means changing one number."},
    "nl": {"title": "Twee doelen voor één feed", "one": "Eén model, één samengestelde loss", "two": "Twee modellen, één weging in code",
           "posts": "posts", "feed": "feed", "model": "model", "trained": "getraind op",
           "loss": "α × loss op kwaliteit", "loss2": "+ (1 − α) × loss op engagement",
           "for": "model voor", "quality": "kwaliteit", "engagement": "engagement",
           "code": "α × kwaliteit", "code2": "+ (1 − α) × engagement", "in_code": "in code",
           "again": "Een nieuwe α: het model opnieuw trainen.", "now": "Een nieuwe α: één getal aanpassen, niets trainen.",
           "foot": "Een oranje rand laat zien wat training leert.",
           "alt": "Twee manieren om een feed met twee doelen te maken. Boven één model, getraind op één samengestelde loss met de weging α erin: een nieuwe α betekent opnieuw trainen. Onder een kwaliteitsmodel en een engagementmodel, en de scores van beide samengevoegd met de weging α in code: een nieuwe α betekent één getal aanpassen."},
}


def idea_picture(lang: str) -> str:
    """One model on one combined loss beside two models whose scores a weight in code combines."""
    w = IDEA_TEXT[lang]
    param = "var(--ea-param)"
    parts = [_t(0, 14, w["title"], 12.5, 600)]
    # one model, one combined loss
    y = 34.0
    parts.append(_t(0, y, w["one"], 11.5, 600, "var(--ea-ink-2)"))
    top = y + 10
    parts.append(_box(0, top + 12, 44, 26))
    parts.append(_t(22, top + 29, w["posts"], 10.5, 400, "var(--ea-ink)", "middle"))
    parts.append(_arrow(46, top + 25, 62, top + 25))
    parts.append(_box(64, top, 200, 50, param, 1.8))
    parts.append(_t(164, top + 15, f'{w["model"]}, {w["trained"]}', 10.5, 600, "var(--ea-ink)", "middle"))
    parts.append(_t(164, top + 29, w["loss"], 10, 400, "var(--ea-ink)", "middle"))
    parts.append(_t(164, top + 42, w["loss2"], 10, 400, "var(--ea-ink)", "middle"))
    parts.append(_arrow(266, top + 25, 282, top + 25))
    parts.append(_box(284, top + 12, 44, 26))
    parts.append(_t(306, top + 29, w["feed"], 10.5, 400, "var(--ea-ink)", "middle"))
    parts.append(_t(0, top + 68, w["again"], 11, 600, "var(--ea-loss-text)"))
    # two models, one weight in code
    y = top + 98
    parts.append(_t(0, y, w["two"], 11.5, 600, "var(--ea-ink-2)"))
    top = y + 10
    parts.append(_box(0, top + 23, 44, 26))
    parts.append(_t(22, top + 40, w["posts"], 10.5, 400, "var(--ea-ink)", "middle"))
    for k, name in enumerate(("quality", "engagement")):
        by = top + k * 40
        parts.append(_arrow(46, top + 36, 58, by + 16))
        parts.append(_box(60, by, 80, 32, param, 1.8))
        parts.append(_t(100, by + 13, w["for"], 9.5, 400, "var(--ea-ink-2)", "middle"))
        parts.append(_t(100, by + 26, w[name], 10.5, 600, "var(--ea-ink)", "middle"))
        parts.append(_arrow(142, by + 16, 154, top + 36))
    parts.append(_box(156, top + 8, 126, 56, "var(--ea-ink)", 1.4))
    parts.append(_t(219, top + 25, w["code"], 10, 400, "var(--ea-ink)", "middle"))
    parts.append(_t(219, top + 39, w["code2"], 10, 400, "var(--ea-ink)", "middle"))
    parts.append(_t(219, top + 55, w["in_code"], 9.5, 600, "var(--ea-ink-2)", "middle"))
    parts.append(_arrow(284, top + 36, 294, top + 36))
    parts.append(_box(295, top + 23, 34, 26))
    parts.append(_t(312, top + 40, w["feed"], 10, 400, "var(--ea-ink)", "middle"))
    parts.append(_t(0, top + 92, w["now"], 11, 600, "var(--ea-ink)"))
    parts.append(_t(0, top + 112, w["foot"], 10.5, 400, "var(--ea-ink-2)"))
    height = top + 120
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(w["alt"])}" style="margin:0.25rem 0 0.75rem;'
            f'max-width:min(100%,330px)"><svg viewBox="0 0 330 {height:.0f}" aria-hidden="true" style="{_SVG}">'
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


def apps_svg(rows: list, lang: str, alt: str) -> str:
    """For each list of apps, a bar per app as long as its score, and the app the learner's function picks, marked
    in ink when it has the highest score and in the color of mistakes when it does not."""
    w = _YOURS_TEXT[lang]["next"]
    L_, R_ = 64.0, 288.0
    parts = []
    y = 4.0
    for title, apps, scores, mine in rows:
        parts.append(_t(0, y + 12, title, 11.5, 600))
        y += 22
        best = next_app(apps, scores)
        for app, score in zip(apps, scores):
            picked = app == mine
            fill = ("var(--ea-ink)" if app == best else "var(--ea-loss)") if picked else "var(--ea-ink-3)"
            parts.append(_t(L_ - 6, y + 9, app, 10.5, 600 if picked else 400, "var(--ea-ink)", "end"))
            parts.append(_rect(L_, y, (R_ - L_) * score, 11, fill))
            parts.append(_t(L_ + (R_ - L_) * score + 4, y + 9, short(score, 2, lang), 10, 400, "var(--ea-ink-2)"))
            y += 15
        shown = w["mine"].format(a=mine) if mine in apps else w["other"].format(a=str(mine)[:24])
        parts.append(_t(L_, y + 12, shown, 11.5, 600, "var(--ea-ink)" if mine == best else "var(--ea-loss-text)"))
        y += 26
    return _svg(y, alt, parts)


def feeds_svg(columns: list, lang: str, alt: str) -> str:
    """For each weight, the posts in the order of the learner's scores, the misleading ones in the color of mistakes,
    and how many misleading posts made the top 3."""
    w = _YOURS_TEXT[lang]["combined"]
    parts = []
    width = 330 / len(columns)
    for k, (alpha, order) in enumerate(columns):
        x = k * width
        parts.append(_t(x, 14, w["alpha"].format(a=short(alpha, 1, lang)), 11.5, 600))
        y = 30.0
        for rank, post in enumerate(order, start=1):
            bad = post in MISLEADING
            if rank == 4:
                parts.append(_line(x, y - 4, x + width - 10, y - 4, "var(--ea-ink-3)", 1, "3 3"))
            title = TITLES[lang][post]
            title = title if len(title) <= 25 else title[:24] + "…"
            parts.append(_t(x, y + 8, f"{rank}. {title}", 10, 600 if bad else 400,
                            "var(--ea-loss-text)" if bad else "var(--ea-ink)"))
            y += 16
        parts.append(_t(x, y + 12, w["top3"].format(k=misleading_top3(order)), 10.5, 600,
                        "var(--ea-loss-text)" if misleading_top3(order) else "var(--ea-ink)"))
    return _svg(30 + 16 * len(POSTS) + 20, alt, parts)


_YOURS_TEXT = {
    "en": {"next": {"title": "Your next_app before and after a new app",
                    "intro": "Your `next_app(C.APPS, C.SCORES)`, and the same call after the user installs Bus, with a score of 0.81 from the same model. A bar per app, as long as its score.",
                    "today": "8 apps", "after": "after installing Bus", "mine": "your function: {a}",
                    "other": "your function: {a}, not one of the apps",
                    "alt": "Bars for the scores of the 8 apps, and of the 9 apps after Bus is installed, with the app your function picks marked.",
                    "right": "Bus gets a score like any other app, and your function picks it: the model needed no new output and no training.",
                    "wrong": "Your function does not pick the app with the highest score each time: the bar of its pick is in the color of mistakes.",
                    "none": "Your `next_app` stops with an error or takes too long, so nothing can be drawn yet."},
           "combined": {"title": "Your combined at two weights",
                        "intro": "The posts in the order of your `combined(C.QUALITY, C.ENGAGEMENT, alpha)`, for `alpha` 0.3 and 0.8, from the highest score down. Misleading posts are in the color of mistakes.",
                        "alpha": "alpha = {a}", "top3": "misleading in the top 3: {k}",
                        "alt": "The 8 posts in the order your function gives them for alpha 0.3 and for alpha 0.8, with the misleading posts marked and their count in the top 3.",
                        "after": "Between the two orders, no model was trained again: only `alpha` changed.",
                        "same": "Both orders are the same, so `alpha` changes nothing in your function: does every score use `alpha`?",
                        "none": "Your `combined` stops with an error, takes too long, or does not give a list of 8 numbers, so nothing can be drawn yet."}},
    "nl": {"next": {"title": "Je next_app voor en na een nieuwe app",
                    "intro": "Je `next_app(C.APPS, C.SCORES)`, en dezelfde aanroep nadat de gebruiker Bus installeert, met een score van 0,81 van hetzelfde model. Een balk per app, zo lang als de score.",
                    "today": "8 apps", "after": "na het installeren van Bus", "mine": "je functie: {a}",
                    "other": "je functie: {a}, geen van de apps",
                    "alt": "Balken voor de scores van de 8 apps, en van de 9 apps nadat Bus geïnstalleerd is, met de app die je functie kiest gemarkeerd.",
                    "right": "Bus krijgt een score zoals elke andere app, en je functie kiest Bus: het model had geen nieuwe output en geen training nodig.",
                    "wrong": "Je functie kiest niet elke keer de app met de hoogste score: de balk van de gekozen app heeft de kleur van fouten.",
                    "none": "Je `next_app` stopt met een fout of duurt te lang, dus is er nog geen afbeelding."},
           "combined": {"title": "Je combined bij twee wegingen",
                        "intro": "De posts in de volgorde van je `combined(C.QUALITY, C.ENGAGEMENT, alpha)`, voor `alpha` 0,3 en 0,8, van de hoogste score omlaag. Misleidende posts hebben de kleur van fouten.",
                        "alpha": "alpha = {a}", "top3": "misleidend in de top 3: {k}",
                        "alt": "De 8 posts in de volgorde die je functie geeft voor alpha 0,3 en voor alpha 0,8, met de misleidende posts gemarkeerd en het aantal misleidende posts in de top 3.",
                        "after": "Tussen de twee volgordes is geen model opnieuw getraind: alleen `alpha` veranderde.",
                        "same": "Beide volgordes zijn gelijk, dus verandert `alpha` niets in je functie: gebruikt elke score `alpha`?",
                        "none": "Je `combined` stopt met een fout, duurt te lang, of geeft geen lijst van 8 getallen, dus is er nog geen afbeelding."}},
}


def next_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["next"]
    fn = _fn(seen)
    lists = [(w["today"], list(APPS), list(SCORES)), (w["after"], APPS + [NEW_APP], SCORES + [NEW_SCORE])]
    got = []
    for _, apps, scores in lists:
        v = _call(lambda a=apps, s=scores: fn(list(a), list(s))) if fn is not None else _STOPPED
        got.append(None if v is _STOPPED else v)
    if fn is None or any(g is None for g in got):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    rows = [(title, apps, scores, mine) for (title, apps, scores), mine in zip(lists, got)]
    svg = apps_svg(rows, lang, w["alt"])
    right = got[0] == PICK and got[1] == PICK_NEW
    return yours.figure(w["title"], _note(w["intro"]) + svg + _note(w["right"] if right else w["wrong"],
                                                                     _UI if right else _LOSS_UI), alt=w["title"])


def combined_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["combined"]
    fn = _fn(seen)
    columns = []
    for alpha in (ALPHA, ALPHA_HIGH):
        v = _call(lambda a=alpha: fn(list(QUALITY), list(ENGAGEMENT), a)) if fn is not None else _STOPPED
        scores = v if isinstance(v, (list, tuple)) and len(v) == len(POSTS) else None
        scores = [_real(s, -1e9, 1e9) for s in scores] if scores is not None else None
        if scores is None or any(s is None for s in scores):
            return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
        columns.append((alpha, ranked(scores)))
    same = columns[0][1] == columns[1][1]
    note = _note(w["same"], _LOSS_UI) if same else _note(w["after"])
    return yours.figure(w["title"], _note(w["intro"]) + feeds_svg(columns, lang, w["alt"]) + note, alt=w["title"])


# ======== the demo: one weight over the feed ==============================================================================

DEMO_TEXT = {
    "en": {"title": "α = {a}: quality weighs {a}, engagement {b}", "quality": "quality × α",
           "engagement": "engagement × (1 − α)", "misleading": "misleading post",
           "top3": "misleading posts in the top 3: {k}", "means": "top 3: quality {q}, engagement {e} on average",
           "trained": "models trained again for this α: 0",
           "alt": "The 8 posts of the feed in the order of their combined score at alpha {a}, each with a bar made of its quality times alpha and its engagement times 1 minus alpha, and the misleading posts marked.",
           "says": {
               "zero": "At **α = 0**, only engagement counts: the top 3 are the three misleading posts, which many people react to.",
               "all": "At **α = {a}**, all of the top 3 are still misleading. The top 3 have a quality of {q} and an engagement of {e} on average.",
               "mid": "At **α = {a}**, {k} of the top 3 {verb} misleading. The top 3 have a quality of {q} and an engagement of {e} on average.",
               "clean": "At **α = {a}**, no misleading post is left in the top 3, while the top 3 have an engagement of {e} on average.",
               "one": "At **α = 1**, only quality counts: *Where city money goes* leads, though few people react to it. The top 3 have an engagement of {e} on average.",
           }},
    "nl": {"title": "α = {a}: kwaliteit weegt {a}, engagement {b}", "quality": "kwaliteit × α",
           "engagement": "engagement × (1 − α)", "misleading": "misleidende post",
           "top3": "misleidende posts in de top 3: {k}", "means": "top 3: gemiddelde kwaliteit {q}, engagement {e}",
           "trained": "opnieuw getrainde modellen voor deze α: 0",
           "alt": "De 8 posts van de feed in de volgorde van de samengestelde score bij alpha {a}, elk met een balk van de kwaliteit keer alpha en het engagement keer 1 min alpha, en de misleidende posts gemarkeerd.",
           "says": {
               "zero": "Bij **α = 0** telt alleen engagement: de top 3 bestaat uit de drie misleidende posts, waar veel mensen op reageren.",
               "all": "Bij **α = {a}** is de hele top 3 nog misleidend. De top 3 heeft gemiddeld een kwaliteit van {q} en een engagement van {e}.",
               "mid": "Bij **α = {a}** {verb} {k} van de top 3 misleidend. De top 3 heeft gemiddeld een kwaliteit van {q} en een engagement van {e}.",
               "clean": "Bij **α = {a}** staat er geen misleidende post meer in de top 3, en heeft de top 3 gemiddeld nog een engagement van {e}.",
               "one": "Bij **α = 1** telt alleen kwaliteit: *Waar het stadsgeld blijft* staat bovenaan, al reageren er weinig mensen op. De top 3 heeft gemiddeld een engagement van {e}.",
           }},
}


def say(a: float, lang: str) -> str:
    w = DEMO_TEXT[lang]["says"]
    order = FEEDS[a]
    k = misleading_top3(order)
    q, e = short(mean_top3(order, QUALITY), 2, lang), short(mean_top3(order, ENGAGEMENT), 2, lang)
    if a == 0:
        return w["zero"]
    if a == 1:
        return w["one"].format(e=e)
    if k == 0:
        return w["clean"].format(a=short(a, 1, lang), e=e)
    if k == 3:
        return w["all"].format(a=short(a, 1, lang), q=q, e=e)
    verb = {"en": "is" if k == 1 else "are", "nl": "is" if k == 1 else "zijn"}[lang]
    return w["mid"].format(a=short(a, 1, lang), k=k, verb=verb, q=q, e=e)


def frame(alpha: float, lang: str) -> tuple[str, str]:
    """The feed at this weight: the posts in the order of their combined score, each bar made of the part from
    quality and the part from engagement, the misleading posts marked, and what the top 3 hold."""
    a = round(alpha, 1)
    w = DEMO_TEXT[lang]
    order = FEEDS[a]
    scores = combined(QUALITY, ENGAGEMENT, a)
    parts = [_t(0, 14, w["title"].format(a=short(a, 1, lang), b=short(1 - a, 1, lang)), 12, 600)]
    X0, X1 = 190.0, 300.0
    y = 30.0
    for rank, post in enumerate(order, start=1):
        i = POSTS.index(post)
        bad = post in MISLEADING
        if rank == 4:
            parts.append(_line(0, y - 3, 326, y - 3, "var(--ea-ink-3)", 1, "3 3"))
        title = TITLES[lang][post]
        title = title if len(title) <= 30 else title[:29] + "…"
        if bad:
            parts.append(_rect(0, y + 2, 7, 7, "var(--ea-loss)"))
        parts.append(_t(11, y + 9, f"{rank}. {title}", 10, 600 if bad else 400,
                        "var(--ea-loss-text)" if bad else "var(--ea-ink)"))
        qpart = (X1 - X0) * a * QUALITY[i]
        epart = (X1 - X0) * (1 - a) * ENGAGEMENT[i]
        parts.append(_rect(X0, y + 1, qpart, 10, "var(--ea-ink)"))
        parts.append(_rect(X0 + qpart, y + 1, epart, 10, "var(--ea-ink-3)"))
        parts.append(_t(X0 + qpart + epart + 4, y + 10, number(scores[i], 3, lang), 9.5, 400, "var(--ea-ink-2)"))
        y += 18
    y += 6
    parts.append(_rect(0, y, 9, 9, "var(--ea-ink)"))
    parts.append(_t(13, y + 8, w["quality"], 10, 400, "var(--ea-ink-2)"))
    parts.append(_rect(88, y, 9, 9, "var(--ea-ink-3)"))
    parts.append(_t(101, y + 8, w["engagement"], 10, 400, "var(--ea-ink-2)"))
    parts.append(_rect(222, y + 1, 7, 7, "var(--ea-loss)"))
    parts.append(_t(233, y + 8, w["misleading"], 10, 400, "var(--ea-ink-2)"))
    k = misleading_top3(order)
    parts.append(_t(0, y + 28, w["top3"].format(k=k), 11.5, 600, "var(--ea-loss-text)" if k else "var(--ea-ink)"))
    parts.append(_t(0, y + 44, w["means"].format(q=short(mean_top3(order, QUALITY), 2, lang),
                                                  e=short(mean_top3(order, ENGAGEMENT), 2, lang)), 11, 400, "var(--ea-ink-2)"))
    parts.append(_t(0, y + 60, w["trained"], 11, 600, "var(--ea-ink)"))
    return _svg(y + 66, w["alt"].format(a=short(a, 1, lang)), parts), say(a, lang)


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
    """A list of the lesson that the check shows by its name: C.APPS."""

    def __init__(self, items, name: str):
        super().__init__(items)
        self._name = name

    def __repr__(self) -> str:
        return self._name

    def __deepcopy__(self, memo):
        return list(self)


APPS_CASE = _Listed(APPS, "C.APPS")
SCORES_CASE = _Listed(SCORES, "C.SCORES")
QUALITY_CASE = _Listed(QUALITY, "C.QUALITY")
ENGAGEMENT_CASE = _Listed(ENGAGEMENT, "C.ENGAGEMENT")


def _shown_scores(v):
    """How {got} and {want} show a list of scores: at most 3 decimals each, with a decimal comma in Dutch."""
    if isinstance(v, list) and all(not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x) for x in v):
        return {lang: "[" + ("; " if lang == "nl" else ", ").join(short(x, 3, lang) for x in v) + "]" for lang in LANGS}
    return check._as_value(v)


# exercise 1: next_app
_N_HEAD = ("def next_app(apps, scores):\n"
           "    best = 0\n"
           "    for i in range(len(apps)):\n")
_N_END = ("            best = i\n"
          "    return apps[best]\n")


def _n(condition: str) -> str:
    return _N_HEAD + f"        if {condition}:\n" + _N_END


_N_START = _n("...")
_N = _n("scores[i] > scores[best]")
_N_LOWEST = _n("scores[i] < scores[best]")
_N_POSITION = _n("scores[i] > best")
_N_FIXED = _n("scores[i] > 0.6")

# exercise 2: combined
_C_HEAD = ("def combined(quality, engagement, alpha):\n"
           "    scores = []\n"
           "    for i in range(len(quality)):\n")
_C_END = "    return scores\n"


def _c(score: str) -> str:
    return _C_HEAD + f"        scores.append({score})\n" + _C_END


_C_START = _c("...")
_C = _c("alpha * quality[i] + (1 - alpha) * engagement[i]")
_C_PLAIN = _c("quality[i] + engagement[i]")
_C_SWAPPED = _c("alpha * engagement[i] + (1 - alpha) * quality[i]")
_C_FULL = _c("alpha * quality[i] + engagement[i]")
_C_BRACKETS = _c("alpha * quality[i] + 1 - alpha * engagement[i]")
_C_FIXED = _c("0.3 * quality[i] + 0.7 * engagement[i]")

EXERCISES = {
    "next": FunctionBox(
        name="next",
        function="next_app",
        start=_N_START,
        solution=_N,
        visible=[(APPS_CASE, SCORES_CASE)],
        hidden=[(APPS + [NEW_APP], SCORES + [NEW_SCORE]), (["Mail", "Radio", "Bank"], [0.9, 0.2, 0.4]),
                (["Chat", "Taxi"], [0.3, 0.35]), (["Radio"], [0.5]),
                (["Game", "Podcast", "Camera", "Bank"], [0.2, 0.1, 0.95, 0.7])],
        first=[_DOTS],
        picture=next_picture,
        mistakes=[
            Mistake("lowest", wrong=_N_LOWEST,
                    en="`{call}` gives {got}, and it should give {want}: your function keeps the app with the lowest score. Which way should the comparison go to keep the highest?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie houdt de app met de laagste score. Welke kant op moet de vergelijking om de hoogste te houden?"),
            Mistake("position", wrong=_N_POSITION,
                    en="`{call}` gives {got}, and it should give {want}: your condition compares a score with `best`, and `best` is a position in the list, not a score. Which score belongs to the app at position `best`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je voorwaarde vergelijkt een score met `best`, en `best` is een plek in de lijst, geen score. Welke score hoort bij de app op plek `best`?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with a condition that is true when app `i` has a higher score than the best app so far.",
                      "nl": "Vervang de `...` door een voorwaarde die waar is als app `i` een hogere score heeft dan de beste app tot nu toe."},
            "arguments": {"en": "The check calls `{call}`, with a list of apps and a list of scores, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(apps, scores)`?",
                          "nl": "De check roept `{call}` aan, met een lijst apps en een lijst scores, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(apps, scores)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function keep the app with the highest score?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Houdt je functie de app met de hoogste score?"},
            "almost": {"en": "Right for the 8 apps of `C.APPS`. But `{call}` gives {got}, and it should give {want}. Does your condition compare with the best app so far, whatever the scores are?",
                       "nl": "Goed voor de 8 apps van `C.APPS`. Maar `{call}` geeft {got}, en het moet {want} zijn. Vergelijkt je voorwaarde met de beste app tot nu toe, wat de scores ook zijn?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return apps[best]`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return apps[best]`?"},
            "correct": {"en": "Right in all {n} cases, also after the user installed Bus: one more app is one more score, and the model stays the same. Could the model with one output per app ever have picked Bus?",
                        "nl": "Goed in alle {n} gevallen, ook nadat de gebruiker Bus installeerde: één app meer is één score meer, en het model blijft hetzelfde. Had het model met één output per app Bus ooit kunnen kiezen?"},
        },
    ),
    "combined": FunctionBox(
        name="combined",
        function="combined",
        start=_C_START,
        solution=_C,
        visible=[(QUALITY_CASE, ENGAGEMENT_CASE, ALPHA)],
        hidden=[(list(QUALITY), list(ENGAGEMENT), 0.0), (list(QUALITY), list(ENGAGEMENT), 1.0), ([0.5], [0.9], 0.5),
                ([0.2, 0.8], [0.6, 0.4], 0.75), ([], [], 0.4)],
        first=[_DOTS],
        picture=combined_picture,
        show=_shown_scores,
        mistakes=[
            Mistake("plain", wrong=_C_PLAIN,
                    en="`{call}` gives {got}, and it should give {want}: your scores add quality and engagement without weights, so changing `alpha` changes nothing. Where does `alpha` go?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je scores tellen kwaliteit en engagement op zonder weging, dus verandert een andere `alpha` niets. Waar hoort `alpha`?"),
            Mistake("swapped", wrong=_C_SWAPPED,
                    en="`{call}` gives {got}, and it should give {want}: in your scores, `alpha` is the weight of engagement. Which of the two goals does `alpha` weigh?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: in je scores is `alpha` de weging van engagement. Welk van de twee doelen weegt `alpha`?"),
            Mistake("brackets", wrong=_C_BRACKETS,
                    en="`{call}` gives {got}, and it should give {want}: Python multiplies before it subtracts, so `1 - alpha * engagement[i]` takes the product `alpha * engagement[i]` away from 1. Which part should the parentheses hold?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: Python vermenigvuldigt voordat Python aftrekt, dus trekt `1 - alpha * engagement[i]` het product `alpha * engagement[i]` van 1 af. Wat hoort er tussen de haakjes?"),
            Mistake("full", wrong=_C_FULL,
                    en="`{call}` gives {got}, and it should give {want}: engagement keeps its full weight, whatever `alpha` is. When quality gets `alpha`, what weight is left for engagement?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: engagement telt helemaal mee, wat `alpha` ook is. Als kwaliteit `alpha` krijgt, welke weging blijft er dan over voor engagement?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the score of post `i`: its quality weighted by `alpha`, plus its engagement weighted by what is left.",
                      "nl": "Vervang de `...` door de score van post `i`: de kwaliteit keer `alpha`, plus het engagement keer wat er van de weging overblijft."},
            "arguments": {"en": "The check calls `{call}`, with two lists of scores and a weight, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(quality, engagement, alpha)`?",
                          "nl": "De check roept `{call}` aan, met twee lijsten scores en een weging, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(quality, engagement, alpha)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is each score `alpha` times the quality plus the rest of the weight times the engagement?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Is elke score `alpha` keer de kwaliteit plus de rest van de weging keer het engagement?"},
            "almost": {"en": "Right for `alpha` 0.3. But `{call}` gives {got}, and it should give {want}. Does your score change when `alpha` changes?",
                       "nl": "Goed voor `alpha` 0,3. Maar `{call}` geeft {got}, en het moet {want} zijn. Verandert je score als `alpha` verandert?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return scores`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return scores`?"},
            "correct": {"en": "Right in all {n} cases. In the picture, how many misleading posts are in the top 3 at `alpha` 0.3, and at 0.8?",
                        "nl": "Goed in alle {n} gevallen. Hoeveel misleidende posts staan er in de afbeelding in de top 3 bij `alpha` 0,3, en bij 0,8?"},
        },
    ),
}

# The lesson is done when these are correct. "framing" and "goals" are the choice questions after the demo.
REQUIRED = ["next", "combined", "framing", "goals"]

CASES = [
    ("next", _N_START, "empty", None),
    ("next", _N, "correct", None),
    ("next", _n("scores[i] > ..."), "wrong", "dots-left"),
    ("next", _N_LOWEST, "wrong", "lowest"),
    ("next", _N_POSITION, "wrong", "position"),
    ("next", _N_FIXED, "almost", None),
    ("next", _n("scores[i] >= scores[best]"), "correct", None),
    ("next", _n("scores[best] < scores[i]"), "correct", None),
    ("next", _N.replace("    return apps[best]\n", "    print(apps[best])\n"), "wrong", "prints"),
    ("combined", _C_START, "empty", None),
    ("combined", _C, "correct", None),
    ("combined", _c("alpha * quality[i] + ..."), "wrong", "dots-left"),
    ("combined", _C_PLAIN, "wrong", "plain"),
    ("combined", _C_SWAPPED, "wrong", "swapped"),
    ("combined", _C_FULL, "wrong", "full"),
    ("combined", _C_BRACKETS, "wrong", "brackets"),
    ("combined", _C_FIXED, "almost", None),
    ("combined", _c("engagement[i] + alpha * (quality[i] - engagement[i])"), "correct", None),
    ("combined", _c("(1 - alpha) * engagement[i] + alpha * quality[i]"), "correct", None),
    ("combined", _C.replace("    return scores\n", "    print(scores)\n"), "wrong", "prints"),
]
