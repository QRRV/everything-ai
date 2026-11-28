# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-cloud: renting someone else's computer. Eight made-up GPU offers, each with the country of
its data center and the country of its company, for a hospital that keeps its data in the EU; what a job costs on each
offer; eight ways to run software, to sort into your own computers, IaaS, PaaS and SaaS; and, in the demo, how fast an
H100 has to be to cost less for a job than an A100.

The offers are made up by the course, with prices like the list prices of November 2026: the lesson says so where it
gives their numbers. The prices of Lambda in the guess and the demo were read from its price list on 28 November 2026."""
import html
import math
from functools import cache

from everything_ai import check, yours
from everything_ai.check import FunctionBox, Mistake, Sort

LESSON = "sys-cloud"
LANGS = ("en", "nl")

# the 27 countries of the European Union in November 2026, as their two-letter codes (ISO 3166)
EU_CODES = ("AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "ES", "FI", "FR", "GR", "HR", "HU", "IE", "IT", "LT", "LU",
            "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK")

# (name, GPU, GB of memory, dollars per hour, how many times as fast as the A100 with 40 GB on the job,
#  country of the data center, country of the company): made up, with prices like those of November 2026
_TABLE = [
    ("A", "H100", 80, 6.88, 2.5, "US", "US"),
    ("B", "H100", 80, 7.20, 2.5, "DE", "US"),
    ("C", "A100", 40, 1.99, 1.0, "US", "US"),
    ("D", "A100", 80, 3.40, 1.0, "NL", "NL"),
    ("E", "H100", 80, 4.90, 2.5, "FI", "FI"),
    ("F", "L4", 24, 0.95, 0.4, "FR", "FR"),
    ("G", "H100", 80, 4.10, 2.5, "CA", "FR"),
    ("H", "A100", 40, 2.60, 1.0, "CH", "CH"),
]
HOURS = 30    # the job of the guess: 30 hours on the A100 with 40 GB
MEMORY = 40   # GB the job needs on the GPU

# Lambda's list prices on 28 November 2026, in dollars per GPU hour, for the guess and the demo
LAMBDA = {"A100 40 GB": 1.99, "A100 80 GB": 3.19, "H100 80 GB": 4.59}
GUESS_SPEED = 2.5   # how many times as fast the H100 is on the job of the guess: an example, not a measurement
AWS_MACHINE = 55.04  # AWS's list price per hour on 28 November 2026 for p5.48xlarge, a machine with eight H100s
SPEEDS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5]   # the stops of the demo's slider


class _Named(list):
    """A list from the lesson: the check's call shows it by its name, and the learner's function gets a copy."""

    def __init__(self, name: str, values):
        super().__init__(values)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return [dict(v) if isinstance(v, dict) else v for v in self]


class _NamedSet(frozenset):
    """The set of EU countries: shown by its name, and the learner's function gets a plain copy."""

    def __new__(cls, name: str, values):
        made = super().__new__(cls, values)
        made.name = name
        return made

    def __init__(self, *args):
        pass

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return set(self)


def offer_list() -> list:
    """The eight offers as dictionaries, the way the learner's functions get them."""
    return [{"name": n, "gpu": g, "memory": m, "price": p, "speed": s, "country": c, "home": h}
            for n, g, m, p, s, c, h in _TABLE]


OFFERS = _Named("C.OFFERS", offer_list())
EU = _NamedSet("C.EU", EU_CODES)


def _some(*names: str) -> _Named:
    """Some of the offers, shown as the code that picks them from C.OFFERS."""
    picked = [o for o in offer_list() if o["name"] in names]
    index = [k for k, o in enumerate(OFFERS) if o["name"] in names]
    return _Named("[" + ", ".join(f"C.OFFERS[{k}]" for k in index) + "]", picked)


def allowed(offers, eu):
    names = []
    for offer in offers:
        if offer["country"] in eu and offer["home"] in eu:
            names.append(offer["name"])
    return names


def cheapest(offers, hours, memory):
    best = None
    best_cost = None
    for offer in offers:
        if offer["memory"] >= memory:
            cost = hours / offer["speed"] * offer["price"]
            if best_cost is None or cost < best_cost:
                best = offer["name"]
                best_cost = cost
    return best


def job_cost(offer: dict, hours: float = HOURS) -> float:
    return hours / offer["speed"] * offer["price"]


def by_name(name: str) -> dict:
    return next(o for o in offer_list() if o["name"] == name)


@cache
def facts() -> dict:
    """The numbers the lesson's texts give."""
    offers = offer_list()
    fit = [o for o in offers if o["memory"] >= MEMORY]
    hour = min(fit, key=lambda o: o["price"])
    rule = allowed(offers, EU)
    a100, h100 = LAMBDA["A100 40 GB"], LAMBDA["H100 80 GB"]
    return {"costs": {o["name"]: round(job_cost(o), 2) for o in fit}, "best": cheapest(offers, HOURS, MEMORY),
            "per_hour": hour["name"], "allowed": rule,
            "best_allowed": cheapest([o for o in offers if o["name"] in rule], HOURS, MEMORY),
            "guess_a100": round(HOURS * a100, 2), "guess_h100": round(HOURS / GUESS_SPEED * h100, 2),
            "guess_hours": HOURS / GUESS_SPEED, "ratio": h100 / a100,
            "aws_per_gpu": AWS_MACHINE / 8}


def demo_costs(speed: float) -> dict:
    """What the job of the guess costs on Lambda's three GPUs, with the H100 `speed` times as fast as the A100."""
    return {"A100 40 GB": (HOURS, HOURS * LAMBDA["A100 40 GB"]), "A100 80 GB": (HOURS, HOURS * LAMBDA["A100 80 GB"]),
            "H100 80 GB": (HOURS / speed, HOURS / speed * LAMBDA["H100 80 GB"])}


# ======== numbers and small pieces of pictures ============================================================================

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem;max-width:var(--ea-measure)"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem;max-width:var(--ea-measure)"
_SVG = "width:100%;max-width:330px;height:auto;font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums"


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma and a thousands dot in Dutch."""
    out = f"{x:,.{digits}f}"
    return out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out


def short(x: float, lang: str) -> str:
    """A number with only the decimals it needs: 2.5, 12, 13.3; in Dutch 2,5 and 13,3."""
    out = f"{round(x, 1):g}" if abs(x - round(x, 2)) > 1e-9 else f"{x:g}"
    return out.replace(".", ",") if lang == "nl" else out


def money(x: float, lang: str) -> str:
    """Dollars in a picture: $59.70, in Dutch $59,70."""
    return "$" + number(x, 2, lang)


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


def _names(names: list, lang: str) -> str:
    """D, E and F; in Dutch D, E en F."""
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + (" en " if lang == "nl" else " and ") + names[-1]


# ======== the picture of the idea: who looks after what ===================================================================

IDEA_TEXT = {
    "en": {"title": "Who looks after what",
           "columns": ["your own", "IaaS", "PaaS", "SaaS"],
           "rows": ["the app people use", "your code and model", "Python and libraries", "operating system",
                    "computers and GPUs", "building and power"],
           "you": "you look after it", "them": "the provider looks after it",
           "alt": "A table of six layers, from the building and the computers at the bottom to the app people use at "
                  "the top, in four columns. Your own: you look after all six. IaaS: the provider looks after the "
                  "building and the computers, you the rest. PaaS: the provider looks after everything up to Python "
                  "and its libraries, you after your code and the app. SaaS: the provider looks after all six."},
    "nl": {"title": "Wie zorgt voor wat",
           "columns": ["eigen", "IaaS", "PaaS", "SaaS"],
           "rows": ["de app voor mensen", "je code en model", "Python en libraries", "besturingssysteem",
                    "computers en GPU's", "gebouw en stroom"],
           "you": "dat doe je zelf", "them": "dat doet de aanbieder",
           "alt": "Een tabel van zes lagen, van het gebouw en de computers onderaan tot de app voor mensen bovenaan, "
                  "in vier kolommen. Eigen: je zorgt zelf voor alle zes. IaaS: de aanbieder zorgt voor het gebouw en de "
                  "computers, jij voor de rest. PaaS: de aanbieder zorgt voor alles tot en met Python en de libraries, "
                  "jij voor je code en de app. SaaS: de aanbieder zorgt voor alle zes."},
}
# how many layers, from the top, you look after yourself in each column
_YOURS = [6, 4, 2, 0]


def idea_picture(lang: str) -> str:
    lang = _lang(lang)
    w = IDEA_TEXT[lang]
    left, width, gap, row_h = 146, 44, 2, 22
    parts = [_t(0, 15, w["title"], 14, 600)]
    top = 46
    for k, name in enumerate(w["columns"]):
        x = left + k * (width + gap)
        parts.append(_t(x + width / 2, top - 8, name, 11.5, 600, "var(--ea-ink)", "middle"))
    for r, row in enumerate(w["rows"]):
        y = top + r * (row_h + gap)
        parts.append(_t(left - 6, y + 15, row, 11.5, 400, "var(--ea-ink-2)", "end"))
        for k in range(4):
            x = left + k * (width + gap)
            mine = r < _YOURS[k]
            style = "fill:var(--ea-ink)" if mine else "fill:var(--ea-inset);stroke:var(--ea-line);stroke-width:1"
            parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{row_h}" rx="2" style="{style}"/>')
    y = top + 6 * (row_h + gap) + 14
    parts.append(f'<rect x="0" y="{y - 10}" width="14" height="12" rx="2" style="fill:var(--ea-ink)"/>'
                 + _t(20, y, w["you"], 11.5, 400, "var(--ea-ink-2)"))
    parts.append(f'<rect x="130" y="{y - 10}" width="14" height="12" rx="2" '
                 f'style="fill:var(--ea-inset);stroke:var(--ea-line);stroke-width:1"/>'
                 + _t(150, y, w["them"], 11.5, 400, "var(--ea-ink-2)"))
    return _svg(y + 8, w["alt"], parts)


# ======== the pictures of the learner's work =============================================================================

_STOPPED = object()   # what _call gives for a function that stopped with an error
PICTURE_SECONDS = 3   # what the learner's function gets for each call of a picture


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


_ALL = [o["name"] for o in offer_list()]


def _name_list(got) -> list | None:
    """got as a list of names of offers, or None: never a string, a set or names the offers do not have."""
    if not isinstance(got, (list, tuple)) or not all(isinstance(v, str) and v in _ALL for v in got):
        return None
    return list(got)


def _one_name(got):
    """got as the name of one offer, None for none at all, or _STOPPED when it is neither."""
    if got is None:
        return None
    return got if isinstance(got, str) and got in _ALL else _STOPPED


COUNTRY = {"en": {"US": "the United States", "DE": "Germany", "NL": "the Netherlands", "FI": "Finland", "FR": "France",
                  "CA": "Canada", "CH": "Switzerland"},
           "nl": {"US": "de Verenigde Staten", "DE": "Duitsland", "NL": "Nederland", "FI": "Finland", "FR": "Frankrijk",
                  "CA": "Canada", "CH": "Zwitserland"}}

TABLE_TEXT = {"en": {"intro": "The eight offers in `C.OFFERS`, with the key of each column:",
                     "head": ["Offer", "GPU", "Memory", "Dollars per hour", "Speed", "Data center", "Company"]},
              "nl": {"intro": "De acht aanbiedingen in `C.OFFERS`, met de key van elke kolom:",
                     "head": ["Aanbieding", "GPU", "Geheugen", "Dollar per uur", "Snelheid", "Datacenter", "Bedrijf"]}}


def offers_table(lang: str) -> str:
    """The eight offers as a Markdown table, for the text of exercise 1."""
    w = TABLE_TEXT[lang]
    keys = ["name", "gpu", "memory", "price", "speed", "country", "home"]
    head = [f'{h}, `"{k}"`' for h, k in zip(w["head"], keys)]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for o in offer_list():
        rows.append(f'| {o["name"]} | {o["gpu"]} | {o["memory"]} GB | {number(o["price"], 2, lang)} | '
                    f'{short(o["speed"], lang)} | {o["country"]} | {o["home"]} |')
    return w["intro"] + "\n\n" + "\n".join(rows)


YOURS_TEXT = {
    "en": {"allowed": {"title": "Your allowed on the eight offers",
                       "intro": "Every offer, placed by the two questions of the hospital's rule, with the country of its "
                                "data center. Filled: in the list of your `allowed(C.OFFERS, C.EU)`.",
                       "columns": ["data center in the EU", "data center outside"],
                       "rows": [["company", "from the EU"], ["company", "from outside"]],
                       "in": "in your list", "out": "not in your list",
                       "said": "Your list: {names}.", "empty": "Your list is empty: no offer may hold the scans.",
                       "rule": "The rule allows only the top left box, where both answers are yes.",
                       "alt": "A grid of four boxes: data center in the EU or outside, against company from the EU or "
                              "from outside. Every offer is in one box, filled when your function keeps it.",
                       "none": "Your `allowed` stops with an error, takes too long, or does not give a list of names "
                               "of offers, so nothing can be drawn yet."},
           "cheapest": {"title": "Your cheapest on the job of the guess",
                        "intro": "What the job of 30 hours on the A100 with 40 GB costs on every offer. Filled: the "
                                 "offer your `cheapest(C.OFFERS, 30, 40)` gives back.",
                        "small": "{gb} GB: too little memory",
                        "pick": "Your function picks {x}: {c} dollars for the job, in {h} hours on its {gpu}.",
                        "nothing": "Your function picks no offer at all, while seven have enough memory.",
                        "small_pick": "Your function picks {x}, whose {gb} GB is too little memory for the job, so the job cannot run there.",
                        "rule_out": "Under the hospital's rule, only {names} are left, and your function picks {y}, which is not one of them.",
                        "hour": "Of the offers with enough memory, {x} has the cheapest hour, at {p} dollars, and there the job costs {c} dollars.",
                        "rule": "Under the hospital's rule, only {names} are left, and your function picks {y} from them.",
                        "rule_none": "Under the hospital's rule, only {names} are left, and your function picks none of them.",
                        "alt": "A bar for every offer with enough memory: what the job costs there. The bar of the "
                               "offer your function picks is filled.",
                        "none": "Your `cheapest` stops with an error, takes too long, or does not give the name of an "
                                "offer or `None`, so nothing can be drawn yet."}},
    "nl": {"allowed": {"title": "Je allowed op de acht aanbiedingen",
                       "intro": "Elke aanbieding volgens de twee vragen van de regel van het ziekenhuis, met het land "
                                "van het datacenter. Gevuld: in de lijst van je `allowed(C.OFFERS, C.EU)`.",
                       "columns": ["datacenter in de EU", "datacenter erbuiten"],
                       "rows": [["bedrijf", "uit de EU"], ["bedrijf", "van buiten"]],
                       "in": "in je lijst", "out": "niet in je lijst",
                       "said": "Je lijst: {names}.", "empty": "Je lijst is leeg: geen enkele aanbieding mag de scans bewaren.",
                       "rule": "De regel staat alleen het vak linksboven toe, waar beide antwoorden ja zijn.",
                       "alt": "Een rooster van vier vakken: datacenter in de EU of erbuiten, tegenover bedrijf uit de "
                              "EU of van buiten. Elke aanbieding staat in één vak, gevuld als je functie de aanbieding houdt.",
                       "none": "Je `allowed` stopt met een fout, duurt te lang, of geeft geen lijst met namen van "
                               "aanbiedingen, dus is er nog geen afbeelding."},
           "cheapest": {"title": "Je cheapest op de job van de gok",
                        "intro": "Wat de job van 30 uur op de A100 met 40 GB kost op elke aanbieding. Gevuld: de "
                                 "aanbieding die je `cheapest(C.OFFERS, 30, 40)` teruggeeft.",
                        "small": "{gb} GB: te weinig geheugen",
                        "pick": "Je functie kiest {x}: {c} dollar voor de job, in {h} uur op een {gpu}.",
                        "nothing": "Je functie kiest helemaal geen aanbieding, terwijl er zeven genoeg geheugen hebben.",
                        "small_pick": "Je functie kiest {x}, met {gb} GB te weinig geheugen voor de job, dus kan de job daar niet draaien.",
                        "rule_out": "Onder de regel van het ziekenhuis blijven alleen {names} over, en je functie kiest {y}, en {y} hoort daar niet bij.",
                        "hour": "Van de aanbiedingen met genoeg geheugen heeft {x} het goedkoopste uur, met {p} dollar, en daar kost de job {c} dollar.",
                        "rule": "Onder de regel van het ziekenhuis blijven alleen {names} over, en daaruit kiest je functie {y}.",
                        "rule_none": "Onder de regel van het ziekenhuis blijven alleen {names} over, en daaruit kiest je functie niets.",
                        "alt": "Een balk voor elke aanbieding met genoeg geheugen: wat de job daar kost. Gevuld: de balk "
                               "van de aanbieding die je functie kiest.",
                        "none": "Je `cheapest` stopt met een fout, duurt te lang, of geeft niet de naam van een "
                                "aanbieding of `None`, dus is er nog geen afbeelding."}},
}


def _chip(x: float, y: float, offer: dict, filled: bool) -> str:
    style = "fill:var(--ea-ink)" if filled else "fill:var(--ea-inset);stroke:var(--ea-line);stroke-width:1"
    ink = "var(--ea-page)" if filled else "var(--ea-ink)"
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="38" height="20" rx="4" style="{style}"/>'
            + _t(x + 19, y + 14, f'{offer["name"]} {offer["country"]}', 11.5, 600, ink, "middle"))


def allowed_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["allowed"]
    fn = _fn(seen)
    got = _call(lambda: fn(offer_list(), set(EU_CODES))) if fn is not None else None
    names = _name_list(got) if got is not _STOPPED else None
    if names is None:
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    left, col, row_h, top = 72, 128, 52, 34
    parts = []
    for k, head in enumerate(w["columns"]):
        parts.append(_t(left + k * (col + 2) + col / 2, top - 10, head, 11.5, 600, "var(--ea-ink)", "middle"))
    for r, (one, two) in enumerate(w["rows"]):
        y = top + r * (row_h + 2)
        parts.append(_t(left - 6, y + 22, one, 11.5, 400, "var(--ea-ink-2)", "end")
                     + _t(left - 6, y + 36, two, 11.5, 400, "var(--ea-ink-2)", "end"))
        for k in range(2):
            x = left + k * (col + 2)
            parts.append(f'<rect x="{x}" y="{y}" width="{col}" height="{row_h}" rx="3" '
                         f'style="fill:none;stroke:var(--ea-line);stroke-width:1"/>')
            box = [o for o in offer_list() if (o["home"] in EU) == (r == 0) and (o["country"] in EU) == (k == 0)]
            for j, o in enumerate(box):
                parts.append(_chip(x + 3 + j * 42, y + 16, o, o["name"] in names))
    y = top + 2 * (row_h + 2) + 18
    parts.append(f'<rect x="0" y="{y - 11}" width="16" height="13" rx="3" style="fill:var(--ea-ink)"/>'
                 + _t(22, y, w["in"], 11.5, 400, "var(--ea-ink-2)"))
    parts.append(f'<rect x="150" y="{y - 11}" width="16" height="13" rx="3" '
                 f'style="fill:var(--ea-inset);stroke:var(--ea-line);stroke-width:1"/>'
                 + _t(172, y, w["out"], 11.5, 400, "var(--ea-ink-2)"))
    said = w["said"].format(names=_names(names, lang)) if names else w["empty"]
    body = _note(w["intro"]) + _svg(y + 8, w["alt"], parts) + _note(said + " " + w["rule"])
    return yours.figure(w["title"], body, alt=w["title"])


def cheapest_svg(pick, lang: str, alt: str) -> str:
    """A bar per offer: what the job of the guess costs there, the learner's pick filled."""
    w = YOURS_TEXT[lang]["cheapest"]
    x0, W, row = 96, 176, 24
    top = max(job_cost(o) for o in offer_list() if o["memory"] >= MEMORY)
    parts = []
    for k, o in enumerate(offer_list()):
        y = 6 + k * row
        parts.append(_t(0, y + 13, o["name"], 12.5, 600) + _t(16, y + 13, f'{o["gpu"]} {o["memory"]} GB', 11.5, 400,
                                                                  "var(--ea-ink-2)"))
        if o["memory"] < MEMORY:
            parts.append(_t(x0, y + 13, w["small"].format(gb=o["memory"]), 11.5, 400, "var(--ea-ink-3)"))
            continue
        cost = job_cost(o)
        width = max(2.0, W * cost / top)
        mine = o["name"] == pick
        style = "fill:var(--ea-ink)" if mine else "fill:var(--ea-line)"
        parts.append(f'<rect x="{x0}" y="{y + 2}" width="{width:.1f}" height="15" rx="2" style="{style}"/>')
        parts.append(_t(x0 + width + 5, y + 14, money(cost, lang), 11.5, 600 if mine else 400))
    return _svg(6 + len(_TABLE) * row + 2, alt, parts)


def cheapest_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["cheapest"]
    fn = _fn(seen)
    got = _call(lambda: fn(offer_list(), HOURS, MEMORY)) if fn is not None else _STOPPED
    pick = _one_name(got) if got is not _STOPPED else _STOPPED
    rule = allowed(offer_list(), EU)
    left = [o for o in offer_list() if o["name"] in rule]
    if pick is not _STOPPED:
        under = _call(lambda: fn([dict(o) for o in left], HOURS, MEMORY))
        under = _one_name(under) if under is not _STOPPED else _STOPPED
    if pick is _STOPPED or under is _STOPPED:
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    f = facts()
    if pick is None:
        said = w["nothing"]
    elif by_name(pick)["memory"] < MEMORY:
        said = w["small_pick"].format(x=pick, gb=by_name(pick)["memory"])
    else:
        o = by_name(pick)
        said = w["pick"].format(x=pick, c=number(job_cost(o), 2, lang), h=short(HOURS / o["speed"], lang), gpu=o["gpu"])
    hour = by_name(f["per_hour"])
    said += " " + w["hour"].format(x=hour["name"], p=number(hour["price"], 2, lang), c=number(job_cost(hour), 2, lang))
    names = _names(rule, lang)
    if under is None:
        said += " " + w["rule_none"].format(names=names)
    else:
        said += " " + w["rule" if under in rule else "rule_out"].format(names=names, y=under)
    body = _note(w["intro"]) + cheapest_svg(pick, lang, w["alt"]) + _note(said)
    return yours.figure(w["title"], body, alt=w["title"])


# ======== the demo: how fast the H100 has to be ==========================================================================

DEMO_TEXT = {
    "en": {"title": "The H100 {s}",
           "speed": {"one": "as fast as the A100", "more": "{s} times as fast as the A100"},
           "short": {"one": "as fast", "more": "{s} times as fast"},
           "row": "{p} an hour, {h} hours",
           "say": "With the H100 {s}, the job takes {h} hours on it and costs {c} dollars: {cmp} the {a} dollars of "
                  "the A100 with 40 GB. The A100 with 80 GB costs {b} dollars at every speed.",
           "more": "more than", "less": "less than",
           "alt": "Three bars: what the job of 30 hours costs on Lambda's A100 with 40 GB, A100 with 80 GB and H100, "
                  "with the H100 as fast as the slider says."},
    "nl": {"title": "De H100 {s}",
           "speed": {"one": "even snel als de A100", "more": "{s} keer zo snel als de A100"},
           "short": {"one": "even snel", "more": "{s} keer zo snel"},
           "row": "{p} per uur, {h} uur",
           "say": "Op een H100 die {t} is als de A100 duurt de job {h} uur en kost de job {c} dollar: {cmp} de {a} "
                  "dollar van de A100 met 40 GB. De A100 met 80 GB kost bij elke snelheid {b} dollar.",
           "more": "meer dan", "less": "minder dan",
           "alt": "Drie balken: wat de job van 30 uur kost op de A100 met 40 GB, de A100 met 80 GB en de H100 van "
                  "Lambda, met de H100 zo snel als de schuif zegt."},
}


def speed_text(speed: float, lang: str) -> str:
    w = DEMO_TEXT[_lang(lang)]["speed"]
    return w["one"] if speed == 1 else w["more"].format(s=short(speed, lang))


def frame(speed: float, lang: str) -> tuple[str, str]:
    lang = _lang(lang)
    w = DEMO_TEXT[lang]
    costs = demo_costs(speed)
    top = max(demo_costs(SPEEDS[0])[k][1] for k in LAMBDA)
    W = 230.0
    parts = [_t(0, 15, w["title"].format(s=speed_text(speed, lang)), 13, 600)]
    y = 34
    for name, (hours, cost) in costs.items():
        parts.append(_t(0, y + 10, name, 12.5, 600)
                     + _t(330, y + 10, w["row"].format(p=money(LAMBDA[name], lang), h=short(hours, lang)), 11.5, 400,
                          "var(--ea-ink-2)", "end"))
        width = max(2.0, W * cost / top)
        style = "fill:var(--ea-ink)" if name.startswith("H100") else "fill:var(--ea-ink-3)"
        parts.append(f'<rect x="0" y="{y + 16}" width="{width:.1f}" height="15" rx="2" style="{style}"/>')
        parts.append(_t(width + 6, y + 28, money(cost, lang), 12, 600))
        y += 46
    a, b = costs["A100 40 GB"][1], costs["A100 80 GB"][1]
    hours, c = costs["H100 80 GB"]
    t = w["short"]["one"] if speed == 1 else w["short"]["more"].format(s=short(speed, lang))
    say = w["say"].format(s=speed_text(speed, lang), t=t, h=short(hours, lang), c=number(c, 2, lang),
                          cmp=w["less"] if c < a else w["more"], a=number(a, 2, lang), b=number(b, 2, lang))
    return _svg(y, w["alt"], parts), say


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

# exercise 1: allowed
_A_HEAD = ("def allowed(offers, eu):\n"
           "    names = []\n"
           "    for offer in offers:\n")
_A_END = ("            names.append(offer[\"name\"])\n"
          "    return names\n")


def _a(condition: str) -> str:
    return _A_HEAD + f"        if {condition}:\n" + _A_END


_A_START = _a("...")
_A = _a('offer["country"] in eu and offer["home"] in eu')
_A_COUNTRY = _a('offer["country"] in eu')
_A_HOME = _a('offer["home"] in eu')
_A_OR = _a('offer["country"] in eu or offer["home"] in eu')

# exercise 2: cheapest
_C_HEAD = ("def cheapest(offers, hours, memory):\n"
           "    best = None\n"
           "    best_cost = None\n"
           "    for offer in offers:\n"
           "        if offer[\"memory\"] >= memory:\n")
_C_END = ("            if best_cost is None or cost < best_cost:\n"
          "                best = offer[\"name\"]\n"
          "                best_cost = cost\n"
          "    return best\n")


def _c(expression: str) -> str:
    return _C_HEAD + f"            cost = {expression}\n" + _C_END


_C_START = _c("...")
_C = _c('hours / offer["speed"] * offer["price"]')
_C_HOUR = _c('offer["price"]')
_C_TIMES = _c('hours * offer["speed"] * offer["price"]')
_C_FAST = _c('hours / offer["speed"]')

# three made-up GPUs, each the cheapest in another way: per hour P, with the speed the wrong way round Q, per job R
_PQR = [{"name": "P", "memory": 48, "price": 1.0, "speed": 3.0}, {"name": "Q", "memory": 48, "price": 2.0, "speed": 1.0},
        {"name": "R", "memory": 48, "price": 2.5, "speed": 10.0}]
# a made-up offer with its data center and its company in two different EU countries
_IRISH = [{"name": "I", "gpu": "H100", "memory": 80, "price": 5.0, "speed": 2.5, "country": "IE", "home": "DE"}]


def _bare_and(got, text_) -> bool:
    """An `and` with a side that compares nothing, such as `offer["country"] in eu and offer["home"]`."""
    import ast
    try:
        tree = ast.parse(text_)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And)
               and any(isinstance(v, (ast.Subscript, ast.Name, ast.Constant)) for v in node.values)
               for node in ast.walk(tree))

KINDS = Sort(
    name="kinds",
    groups={
        "own": {"en": "Your own", "nl": "Eigen"},
        "iaas": {"en": "IaaS", "nl": "IaaS"},
        "paas": {"en": "PaaS", "nl": "PaaS"},
        "saas": {"en": "SaaS", "nl": "SaaS"},
    },
    items={
        "pc": {"en": "**Your PC.** A network of this course, trained on the processor or the graphics card of your own computer.",
               "nl": "**Je pc.** Een netwerk van deze course, getraind op de processor of de grafische kaart van je eigen computer."},
        "room": {"en": "**A server room.** A hospital buys four GPU servers and puts them in a room of its own building, with its own staff to run them.",
                 "nl": "**Een serverruimte.** Een ziekenhuis koopt vier GPU-servers en zet de servers in een ruimte van het eigen gebouw, met eigen personeel dat de servers draaiende houdt."},
        "vm": {"en": "**A virtual machine with a GPU.** Rented by the hour; the team installs the drivers, Python and PyTorch, and runs its own training script on it.",
               "nl": "**Een virtuele machine met een GPU.** Per uur gehuurd; het team installeert de drivers, Python en PyTorch, en draait er het eigen trainingsscript op."},
        "disk": {"en": "**A rented disk.** Storage for the team's virtual machine, paid per gigabyte per month; the team sets it up, and decides what goes on it and who may read it.",
                 "nl": "**Een gehuurde schijf.** Opslag bij de virtuele machine van het team, betaald per gigabyte per maand; het team richt de schijf in, en bepaalt wat erop komt en wie erbij mag."},
        "train": {"en": "**A training service.** The team hands over its training script and data; the provider starts the GPUs, runs the script and stops the GPUs when it is done.",
                  "nl": "**Een trainingsdienst.** Het team levert het trainingsscript en de data aan; de aanbieder start de GPU's, draait het script en zet de GPU's uit als het script klaar is."},
        "serve": {"en": "**A model service.** The team uploads its trained model; the provider runs it behind a web address, and adds computers when the requests grow.",
                  "nl": "**Een modeldienst.** Het team uploadt het getrainde model; de aanbieder draait het model achter een webadres, en zet er computers bij als het aantal verzoeken groeit."},
        "chat": {"en": "**A chat assistant.** Every staff member gets an account, paid per person per month; nobody at the company sees a model or a server.",
                 "nl": "**Een chatassistent.** Elke medewerker krijgt een account, betaald per persoon per maand; niemand bij het bedrijf ziet een model of een server."},
        "transcribe": {"en": "**Transcription.** A website turns recordings of meetings into text, paid per minute of sound; the company only uploads the recordings.",
                       "nl": "**Transcriptie.** Een website maakt tekst van opnames van vergaderingen, betaald per minuut geluid; het bedrijf uploadt alleen de opnames."},
    },
    solution={"pc": "own", "room": "own", "vm": "iaas", "disk": "iaas", "train": "paas", "serve": "paas",
              "chat": "saas", "transcribe": "saas"},
    nudge={
        "pc": {"en": "Who owns the computer the network trains on, and who rents anything out?",
               "nl": "Van wie is de computer waarop het netwerk traint, en verhuurt iemand hier iets?"},
        "room": {"en": "The hospital buys the servers and runs them in its own building. Is anything rented here?",
                 "nl": "Het ziekenhuis koopt de servers en draait de servers in het eigen gebouw. Wordt hier iets gehuurd?"},
        "vm": {"en": "Who installs the drivers, Python and PyTorch: the team or the provider? What does the provider give, beyond the bare machine?",
               "nl": "Wie installeert de drivers, Python en PyTorch: het team of de aanbieder? Wat geeft de aanbieder, naast de kale machine?"},
        "disk": {"en": "Is this finished software, a platform that runs code, or a bare building block that the team fills itself?",
                 "nl": "Is dit kant-en-klare software, een platform dat code draait, of een kale bouwsteen die het team zelf vult?"},
        "train": {"en": "The team brings code and data, and never sees a machine. Who looks after the computers and the software under the script?",
                  "nl": "Het team brengt code en data, en ziet nooit een machine. Wie zorgt voor de computers en de software onder het script?"},
        "serve": {"en": "The model is the team's own, and the provider runs it. Which layer brings your own code or model, and nothing else?",
                  "nl": "Het model is van het team zelf, en de aanbieder draait het model. Bij welke laag breng je je eigen code of model, en verder niets?"},
        "chat": {"en": "Does the company bring any code or model of its own here, or does it only use what is ready?",
                 "nl": "Brengt het bedrijf hier eigen code of een eigen model, of gebruikt het bedrijf alleen wat klaar is?"},
        "transcribe": {"en": "The company uploads recordings and gets text back. Whose software does the work, and who looks after it?",
                       "nl": "Het bedrijf uploadt opnames en krijgt tekst terug. Van wie is de software die het werk doet, en wie houdt de software draaiende?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the eight ways again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de acht manieren opnieuw uit."},
        "empty": {"en": "Put each way of running software under your own computers, IaaS, PaaS or SaaS.",
                  "nl": "Zet elke manier om software te draaien onder eigen computers, IaaS, PaaS of SaaS."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "*{item}* fits another column better.",
                  "nl": "*{item}* past beter bij een andere kolom."},
        "correct": {"en": "All {n} are right. Your own: you buy and run everything. IaaS: you rent bare computers and storage, and install the rest. PaaS: you bring code or a model, and the provider runs it. SaaS: you use finished software.",
                    "nl": "Alle {n} kloppen. Eigen: je koopt en draait alles zelf. IaaS: je huurt kale computers en opslag, en installeert de rest. PaaS: je brengt code of een model, en de aanbieder draait het. SaaS: je gebruikt kant-en-klare software."},
    },
)

EXERCISES = {
    "allowed": FunctionBox(
        name="allowed",
        function="allowed",
        start=_A_START,
        solution=_A,
        visible=[(OFFERS, EU)],
        hidden=[([], EU), (_some("D"), EU), (_some("B", "G", "H"), EU), (OFFERS, {"FR"}), (_IRISH, EU)],
        first=[_DOTS],
        picture=allowed_picture,
        mistakes=[
            Mistake("bare-and", matches=_bare_and,
                    en="`{call}` gives {got}, and it should give {want}. `and` joins two whole questions, each with its own comparison: a side such as `offer[\"home\"]` on its own only asks whether there is a country at all, and every offer has one. Does each side of your `and` ask about `eu`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn. `and` verbindt twee hele vragen, elk met een eigen vergelijking: een kant met alleen `offer[\"home\"]` vraagt niets anders dan of er een land staat, en elke aanbieding heeft er een. Vraagt elke kant van je `and` iets over `eu`?"),
            Mistake("country-only", wrong=_A_COUNTRY,
                    en="`{call}` gives {got}, and it should give {want}. B is in your list: its data center is in Germany, but the company is from the United States, so US law can reach the data too. What does the rule ask about the company?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn. B staat in je lijst: het datacenter staat in Duitsland, maar het bedrijf komt uit de Verenigde Staten, dus kan de Amerikaanse wet ook bij de data. Wat vraagt de regel over het bedrijf?"),
            Mistake("home-only", wrong=_A_HOME,
                    en="`{call}` gives {got}, and it should give {want}. G is in your list: the company is from France, but its data center is in Canada. Where does the rule want the data to be?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn. G staat in je lijst: het bedrijf komt uit Frankrijk, maar het datacenter staat in Canada. Waar wil de regel de data hebben?"),
            Mistake("or", wrong=_A_OR,
                    en="`{call}` gives {got}, and it should give {want}: B and G are in your list, and each has only its data center or only its company in the EU. Must one answer be yes, or both?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: B en G staan in je lijst, en bij elk zit alleen het datacenter of alleen het bedrijf in de EU. Moet één antwoord ja zijn, of allebei?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the condition of the rule: the data center in the EU, and the company from the EU.",
                      "nl": "Vervang de `...` door de voorwaarde van de regel: het datacenter in de EU, en het bedrijf uit de EU."},
            "arguments": {"en": "The check calls `{call}`, with a list of offers and a set of countries, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(offers, eu)`?",
                          "nl": "De check roept `{call}` aan, met een lijst aanbiedingen en een set landen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(offers, eu)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does an offer get in only when the country of its data center and the country of its company are both in `eu`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Komt een aanbieding er alleen in als het land van het datacenter en het land van het bedrijf allebei in `eu` zitten?"},
            "almost": {"en": "Right for the eight offers. But `{call}` gives {got}, and it should give {want}. Does your function look at the offers and the set it gets, and not at fixed names?",
                       "nl": "Goed voor de acht aanbiedingen. Maar `{call}` geeft {got}, en het moet {want} zijn. Kijkt je functie naar de aanbiedingen en de set die de functie krijgt, en niet naar vaste namen?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return names`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return names`?"},
            "correct": {"en": "Right in all {n} cases. Of the eight offers, the rule leaves three: D in the Netherlands, E in Finland and F in France. B has its data center in Frankfurt, but belongs to a US company. In the picture: which box of the grid is left?",
                        "nl": "Goed in alle {n} gevallen. Van de acht aanbiedingen laat de regel er drie over: D in Nederland, E in Finland en F in Frankrijk. B heeft het datacenter in Frankfurt, maar is van een Amerikaans bedrijf. In de afbeelding: welk vak van het rooster blijft over?"},
        },
    ),
    "cheapest": FunctionBox(
        name="cheapest",
        function="cheapest",
        start=_C_START,
        solution=_C,
        visible=[(OFFERS, HOURS, MEMORY)],
        hidden=[(_some("C", "E"), 30, 40), (_some("A", "D"), 10, 80), (_some("A", "C"), 30, 40), (_PQR, 12, 40), ([], 10, 40),
                (_some("F"), 30, 40)],
        first=[_DOTS],
        picture=cheapest_picture,
        mistakes=[
            Mistake("per-hour", wrong=_C_HOUR,
                    en="`{call}` gives {got}, the offer with enough memory and the lowest price per hour, and it should give {want}. But a faster GPU finishes the job in fewer hours. How many hours does the job take on each offer?",
                    nl="`{call}` geeft {got}, de aanbieding met genoeg geheugen en de laagste prijs per uur, en het moet {want} zijn. Maar een snellere GPU is in minder uren klaar met de job. Hoeveel uur duurt de job op elke aanbieding?"),
            Mistake("times-speed", wrong=_C_TIMES,
                    en="`{call}` gives {got}, and it should give {want}: in your cost, a faster GPU takes more hours, so a fast GPU looks dear. Does a GPU that is twice as fast need twice the hours, or half?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: in je kosten duurt de job op een snellere GPU langer, dus lijkt een snelle GPU duur. Heeft een GPU die twee keer zo snel is twee keer zoveel uren nodig, of de helft?"),
            Mistake("fastest", wrong=_C_FAST,
                    en="`{call}` gives {got}, the fastest GPU whatever an hour on it costs, and it should give {want}. What does an hour on each offer cost?",
                    nl="`{call}` geeft {got}, de snelste GPU, wat een uur erop ook kost, en het moet {want} zijn. Wat kost een uur op elke aanbieding?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with what the job costs on `offer`: the hours it takes there, times the price per hour.",
                      "nl": "Vervang de `...` door wat de job kost op `offer`: de uren die de job daar duurt, keer de prijs per uur."},
            "arguments": {"en": "The check calls `{call}`, with a list of offers and two numbers, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(offers, hours, memory)`?",
                          "nl": "De check roept `{call}` aan, met een lijst aanbiedingen en twee getallen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(offers, hours, memory)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is `cost` the hours the job takes on `offer`, times the price of an hour there?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Is `cost` het aantal uren dat de job op `offer` duurt, keer de prijs van een uur daar?"},
            "almost": {"en": "Right for the job of the guess. But `{call}` gives {got}, and it should give {want}. Does your function work with the offers, `hours` and `memory` it gets, and not with a fixed name?",
                       "nl": "Goed voor de job van de gok. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie met de aanbiedingen, `hours` en `memory` die de functie krijgt, en niet met een vaste naam?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return best`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return best`?"},
            "correct": {"en": "Right in all {n} cases. The job costs least on G: 49.20 dollars, in 12 hours on an H100. C, with the cheapest hour of the offers with enough memory, 1.99 dollars, costs 59.70 dollars for the job. In the picture: which offer does your function pick under the hospital's rule?",
                        "nl": "Goed in alle {n} gevallen. De job kost het minst op G: 49,20 dollar, in 12 uur op een H100. C, met 1,99 dollar het goedkoopste uur van de aanbiedingen met genoeg geheugen, kost 59,70 dollar voor de job. In de afbeelding: welke aanbieding kiest je functie onder de regel van het ziekenhuis?"},
        },
    ),
    "kinds": KINDS,
}

# The lesson is done when these are correct. "colab" is the choice question before the demo.
REQUIRED = ["allowed", "cheapest", "kinds", "colab"]

_SORTED = dict(KINDS.solution)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("allowed", _A_START, "empty", None),
    ("allowed", "", "empty", None),
    ("allowed", _A, "correct", None),
    ("allowed", _a('offer["home"] in eu and offer["country"] in eu'), "correct", None),
    ("allowed", _a('eu.issuperset([offer["country"], offer["home"]])'), "correct", None),
    ("allowed", _A_COUNTRY, "wrong", "country-only"),
    ("allowed", _A_HOME, "wrong", "home-only"),
    ("allowed", _A_OR, "wrong", "or"),
    ("allowed", _a('offer["country"] not in eu and offer["home"] not in eu'), "wrong", None),
    ("allowed", _a('offer["country"] in eu and offer["home"] == offer["country"]'), "almost", None),
    ("allowed", _a('offer["country"] in eu and offer["home"]'), "wrong", "bare-and"),
    ("allowed", _a('offer["country"] and offer["home"] in eu'), "wrong", "bare-and"),
    ("allowed", "def allowed(offers, eu):\n    return [\"D\", \"E\", \"F\"]\n", "almost", None),
    ("allowed", _a('offer["land"] in eu'), "error", None),
    ("allowed", _A.replace("def allowed(offers, eu):", "def allowed(offers):"), "error", "arguments"),
    ("allowed", _A.replace("    return names\n", ""), "wrong", "no-return"),
    ("allowed", _A.replace("    return names\n", "    print(names)\n"), "wrong", "prints"),
    ("allowed", _A.replace("    return names\n", "    extra = ...\n    return names\n"), "wrong", "dots-left"),
    ("allowed", "x = 1\n", "wrong", "no-function"),
    ("cheapest", _C_START, "empty", None),
    ("cheapest", "", "empty", None),
    ("cheapest", _C, "correct", None),
    ("cheapest", _c('offer["price"] * hours / offer["speed"]'), "correct", None),
    ("cheapest", _c('offer["price"] / offer["speed"]'), "correct", None),
    ("cheapest", _C_HOUR, "wrong", "per-hour"),
    ("cheapest", _c('hours * offer["price"]'), "wrong", "per-hour"),
    ("cheapest", _C_TIMES, "wrong", "times-speed"),
    ("cheapest", _C_FAST, "wrong", "fastest"),
    ("cheapest", _c('offer["speed"] / offer["price"]'), "wrong", None),
    ("cheapest", _c('hours / offer["speed"] + offer["price"]'), "almost", None),
    ("cheapest", "def cheapest(offers, hours, memory):\n    return \"G\"\n", "almost", None),
    ("cheapest", _c('hours / offer["speed"] * offer["prise"]'), "error", None),
    ("cheapest", _C.replace("def cheapest(offers, hours, memory):", "def cheapest(offers, hours):"), "error", "arguments"),
    ("cheapest", _C.replace("    return best\n", ""), "wrong", "no-return"),
    ("cheapest", _C.replace("    return best\n", "    print(best)\n"), "wrong", "prints"),
    ("cheapest", _C.replace("    return best\n", "    extra = ...\n    return best\n"), "wrong", "dots-left"),
    ("cheapest", "x = 1\n", "wrong", "no-function"),
    ("kinds", {}, "empty", None),
    ("kinds", _SORTED, "correct", None),
    ("kinds", {"pc": "own", "room": "own"}, "wrong", "unfinished"),
    ("kinds", _SORTED | {"pc": "saas"}, "wrong", "wrong:pc"),
    ("kinds", _SORTED | {"room": "iaas"}, "wrong", "wrong:room"),
    ("kinds", _SORTED | {"vm": "paas"}, "wrong", "wrong:vm"),
    ("kinds", _SORTED | {"disk": "saas"}, "wrong", "wrong:disk"),
    ("kinds", _SORTED | {"train": "iaas"}, "wrong", "wrong:train"),
    ("kinds", _SORTED | {"serve": "saas"}, "wrong", "wrong:serve"),
    ("kinds", _SORTED | {"chat": "paas"}, "wrong", "wrong:chat"),
    ("kinds", _SORTED | {"transcribe": "paas"}, "wrong", "wrong:transcribe"),
]
