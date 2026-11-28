# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-containers: a box that runs anywhere. Two Dockerfiles for a service like that of the
project ops, one that copies everything first and one in order; how many steps Docker builds again when a file
changes; how much of a new image a server still has to download; what goes in an image, what comes from the server
and what is given when a container starts; and, in the demo, both Dockerfiles side by side for each changed file.

The seconds and megabytes of the steps are examples made up by the course, of the size a laptop and a service with
PyTorch, pandas and a small model give: the lesson says so where it gives them. Docker does not run in the course:
the cache rule is the one of Docker's documentation, "Docker build cache", read on 28 November 2026."""
import hashlib
import html
from functools import cache

from everything_ai import check, yours
from everything_ai.check import FunctionBox, Mistake, Sort

LESSON = "sys-containers"
LANGS = ("en", "nl")


class _Named(list):
    """A list from the lesson: the check's call shows it by its name, and the learner's function gets a copy."""

    def __init__(self, name: str, values):
        super().__init__(values)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return [{k: (list(x) if isinstance(x, list) else x) for k, x in v.items()} for v in self]


class _NamedSet(frozenset):
    """A set from the lesson: shown by its name, and the learner's function gets a plain copy."""

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


# the files of the service, as in the project ops: the packages, the model package, the code, and the experiment log
FILES = ["requirements.txt", "model/", "service.py", "runs.jsonl"]

# what each step takes on a laptop, in seconds, and what its layer weighs to download, in MB: examples by the course
SECONDS = {"FROM python:3.12-slim": 0.0, "WORKDIR /app": 0.1, "COPY requirements.txt .": 0.1,
           "RUN pip install -r requirements.txt": 95.0, "COPY model/ model/": 0.4, "COPY service.py .": 0.1,
           "COPY . .": 0.5}
MB = {"FROM python:3.12-slim": 45.0, "WORKDIR /app": 0.0, "COPY requirements.txt .": 0.01,
      "RUN pip install -r requirements.txt": 380.0, "COPY model/ model/": 3.0, "COPY service.py .": 0.02,
      "COPY . .": 3.05}
CMD = 'CMD ["python", "service.py"]'   # the last line of both Dockerfiles: it makes no layer


def quick_steps() -> list:
    """The Dockerfile in order: what changes least first, the code of the service last."""
    return [{"line": "FROM python:3.12-slim", "files": []},
            {"line": "WORKDIR /app", "files": []},
            {"line": "COPY requirements.txt .", "files": ["requirements.txt"]},
            {"line": "RUN pip install -r requirements.txt", "files": []},
            {"line": "COPY model/ model/", "files": ["model/"]},
            {"line": "COPY service.py .", "files": ["service.py"]}]


def slow_steps() -> list:
    """The Dockerfile that copies the whole project before it installs the packages."""
    return [{"line": "FROM python:3.12-slim", "files": []},
            {"line": "WORKDIR /app", "files": []},
            {"line": "COPY . .", "files": list(FILES)},
            {"line": "RUN pip install -r requirements.txt", "files": []}]


QUICK = _Named("C.QUICK", quick_steps())
SLOW = _Named("C.SLOW", slow_steps())


def rebuilt(steps, changed):
    for i in range(len(steps)):
        for name in steps[i]["files"]:
            if name in changed:
                return len(steps) - i
    return 0


def seconds(steps: list, n: int) -> float:
    """What building the last n steps again takes."""
    return round(sum(SECONDS[s["line"]] for s in steps[len(steps) - n:]) if n else 0.0, 2)


def image(steps: list, changed: set) -> list:
    """The layers of the image after a build in which `changed` changed: a layer built again gets a new id, since
    its files are new, and so does every layer on top of it; a layer from the cache keeps its id."""
    first = len(steps) - rebuilt(steps, changed)
    layers, below = [], ""
    for i, s in enumerate(steps):
        build = 2 if i >= first else 1
        below = hashlib.sha256(f"{below}\n{s['line']}\n{build}".encode()).hexdigest()[:12]
        layers.append({"id": below, "line": s["line"], "mb": MB[s["line"]]})
    return layers


def to_download(layers, have):
    mb = 0
    for layer in layers:
        if layer["id"] not in have:
            mb = mb + layer["mb"]
    return mb


def _ids(*images) -> set:
    return {layer["id"] for one in images for layer in one}


# yesterday's versions of both images are on the server; today service.py changed, and both images are built again
ON_SERVER = _NamedSet("C.ON_SERVER", _ids(image(quick_steps(), set()), image(slow_steps(), set())))
QUICK_IMAGE = _Named("C.QUICK_IMAGE", image(quick_steps(), {"service.py"}))
SLOW_IMAGE = _Named("C.SLOW_IMAGE", image(slow_steps(), {"service.py"}))

DEMO_VALUES = [1, 2, 3, 4]
DEMO_FILES = {1: "runs.jsonl", 2: "service.py", 3: "model/", 4: "requirements.txt"}


@cache
def facts() -> dict:
    """The numbers the texts give."""
    out = {}
    for name in FILES:
        for label, steps in (("quick", quick_steps()), ("slow", slow_steps())):
            n = rebuilt(steps, {name})
            new = image(steps, {name})
            out[(label, name)] = {"n": n, "seconds": seconds(steps, n), "mb": round(to_download(new, ON_SERVER), 2)}
    out["all_quick"] = round(sum(MB[s["line"]] for s in quick_steps()), 2)
    out["all_slow"] = round(sum(MB[s["line"]] for s in slow_steps()), 2)
    return out


_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem;max-width:var(--ea-measure)"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem;max-width:var(--ea-measure)"
_SVG = "width:100%;max-width:330px;height:auto;font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums"
_MONO = "var(--ea-font-code)"
_FILLED = "fill:var(--ea-ink)"
_EMPTY = "fill:var(--ea-inset);stroke:var(--ea-line);stroke-width:1"


def number(x: float, lang: str) -> str:
    """A number with only the decimals it needs, at most two: 95.6, 0.02, 383.05; in Dutch with a decimal comma."""
    out = f"{round(x, 2):g}" if abs(x) < 1e6 else f"{x:.0f}"
    return out.replace(".", ",") if lang == "nl" else out


def _lang(lang) -> str:
    return lang if lang in LANGS else "en"


def _marked(text_: str) -> str:
    """Escaped text in which `name` is set as code."""
    return "".join(f"<code>{part}</code>" if k % 2 else part for k, part in enumerate(html.escape(text_).split("`")))


def _note(text_: str, style: str = _UI) -> str:
    return f'<div style="{style}">{_marked(text_)}</div>'


def _t(x: float, y: float, text_: str, size: float = 12.5, weight: int = 400, fill: str = "var(--ea-ink)",
       anchor: str = "start", font: str | None = None) -> str:
    family = f"font-family:{font};" if font else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" style="{family}font-size:{size}px;'
            f'font-weight:{weight};fill:{fill}">{html.escape(text_)}</text>')


def _svg(height: float, alt: str, parts: list) -> str:
    return (f'<svg viewBox="0 0 330 {height:.0f}" role="img" aria-label="{html.escape(alt)}" style="{_SVG}">'
            f'{"".join(parts)}</svg>')


def _box(x: float, y: float, w: float, h: float, filled: bool) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="3" style="{_FILLED if filled else _EMPTY}"/>'


def _legend(y: float, filled_text: str, empty_text: str, second_x: float = 160) -> str:
    return (_box(0, y - 11, 16, 13, True) + _t(22, y, filled_text, 11.5, 400, "var(--ea-ink-2)")
            + _box(second_x, y - 11, 16, 13, False) + _t(second_x + 22, y, empty_text, 11.5, 400, "var(--ea-ink-2)"))


# ======== the picture of the idea: virtual machines and containers ========================================================

IDEA_TEXT = {
    "en": {"vm": "Virtual machines", "ct": "Containers",
           "own": ["service", "Python and|packages", "own|operating|system"],
           "vm_base": ["hypervisor", "operating system", "server"],
           "ct_base": ["Docker", "operating system and kernel", "server"],
           "packed": "packed in the image", "shared": "on the server",
           "alt": "Two stacks side by side. Two virtual machines on a server each bring an operating system of their "
                  "own, with Python, packages and the service on top. Two containers on a server share its operating "
                  "system and kernel, and each brings only Python, packages and the service."},
    "nl": {"vm": "Virtuele machines", "ct": "Containers",
           "own": ["dienst", "Python en|packages", "eigen|besturings-|systeem"],
           "vm_base": ["hypervisor", "besturingssysteem", "server"],
           "ct_base": ["Docker", "besturingssysteem met kernel", "server"],
           "packed": "verpakt in het image", "shared": "op de server",
           "alt": "Twee stapels naast elkaar. Twee virtuele machines op een server brengen elk een eigen "
                  "besturingssysteem mee, met Python, packages en de dienst erop. Twee containers op een server delen "
                  "het besturingssysteem en de kernel, en brengen elk alleen Python, packages en de dienst mee."},
}


def _cell(x: float, y: float, w: float, h: float, label: str, filled: bool) -> str:
    """A box with a label of one or two lines, split at |."""
    lines = label.split("|")
    ink = "var(--ea-page)" if filled else "var(--ea-ink)"
    out = _box(x, y, w, h, filled)
    first = y + h / 2 + 3.5 - 6 * (len(lines) - 1)
    for k, line in enumerate(lines):
        out += _t(x + w / 2, first + 12 * k, line, 10, 600 if filled else 400, ink, "middle")
    return out


def idea_picture(lang: str) -> str:
    w = IDEA_TEXT[_lang(lang)]
    parts = []
    stack_w, gap, h = 160, 10, 40
    tall = 3 + 3   # three own rows and three shared rows in the tallest stack
    for s, (title, own_rows, base) in enumerate(((w["vm"], 3, w["vm_base"]), (w["ct"], 2, w["ct_base"]))):
        x0 = s * (stack_w + gap)
        parts.append(_t(x0 + stack_w / 2, 14, title, 12.5, 600, "var(--ea-ink)", "middle"))
        bottom = 24 + tall * (h + 3)
        for r, label in enumerate(base[::-1]):   # the server at the bottom
            y = bottom - (r + 1) * (h + 3)
            parts.append(_cell(x0, y, stack_w, h, label, False))
        top_of_base = bottom - len(base) * (h + 3)
        col_w = (stack_w - 4) / 2
        for c in range(2):
            for r in range(own_rows):
                y = top_of_base - (own_rows - r) * (h + 3)
                parts.append(_cell(x0 + c * (col_w + 4), y, col_w, h, w["own"][r], True))
    y = 24 + tall * (h + 3) + 16
    parts.append(_legend(y, w["packed"], w["shared"], 172))
    return _svg(y + 6, w["alt"], parts)


# ======== the pictures under the checks ===================================================================================

_STOPPED = object()   # what _call gives for a function that stopped with an error
PICTURE_SECONDS = 3   # what the learner's function gets for each call of a picture

YOURS_TEXT = {
    "en": {"rebuilt": {"title": "Your rebuilt on both Dockerfiles",
                       "intro": "For each file that changes: the steps your `rebuilt` says Docker builds again, filled, "
                                "and the seconds they take in this example.",
                       "quick": "In order, `C.QUICK`", "slow": "Everything first, `C.SLOW`",
                       "key": "Steps of `C.QUICK`: {q}. Steps of `C.SLOW`: {s}.",
                       "again": "built again", "cache": "from the cache",
                       "none": "Your `rebuilt` stops with an error, takes too long, or gives something other than a "
                               "number of steps, so the picture waits.",
                       "right": "Your function is right for all eight builds. A change to service.py builds one step "
                                "again in `C.QUICK`, and two in `C.SLOW`, with the install of 95 seconds.",
                       "off": "For a change to {f} in {d}, your function says {n}, and Docker builds {m} again.",
                       "steps": {"one": "1 step", "more": "{n} steps"},
                       "alt": "Two grids: for each of four files, the steps of a Dockerfile in a row, filled when "
                              "the learner's function says Docker builds them again, with the seconds beside them."},
           "download": {"title": "Your to_download after each change",
                        "intro": "Yesterday's version of both images is on the server. For each file that changes "
                                 "today: what your `to_download` says the server downloads of the new image.",
                        "slow": "everything first", "quick": "in order",
                        "none": "Your `to_download` stops with an error, takes too long, or gives something other "
                                "than a number, so the picture waits.",
                        "said": "After a change to service.py, your function says {a} MB for the image that copies "
                                "everything first, and {b} MB for the image in order.",
                        "right": "That is right: only the layers with a new id are downloaded.",
                        "off": "The server downloads {a} MB and {b} MB: the layers whose id it does not have yet.",
                        "alt": "Bars for four changed files, two each: the megabytes the learner's function says the "
                               "server downloads of the image that copies everything first, and of the image in order."}},
    "nl": {"rebuilt": {"title": "Je rebuilt op de twee Dockerfiles",
                       "intro": "Voor elk bestand dat verandert: gevuld de stappen die Docker volgens je `rebuilt` "
                                "opnieuw bouwt, en de seconden die de stappen in dit voorbeeld kosten.",
                       "quick": "Op volgorde, `C.QUICK`", "slow": "Alles eerst, `C.SLOW`",
                       "key": "Stappen van `C.QUICK`: {q}. Stappen van `C.SLOW`: {s}.",
                       "again": "opnieuw gebouwd", "cache": "uit de cache",
                       "none": "Je `rebuilt` stopt met een fout, duurt te lang, of geeft geen aantal stappen, dus is "
                               "er nog geen afbeelding.",
                       "right": "Je functie klopt voor alle acht builds. Een verandering in service.py bouwt in "
                                "`C.QUICK` één stap opnieuw, en in `C.SLOW` twee, met de installatie van 95 seconden.",
                       "off": "Bij een verandering in {f} in {d} zegt je functie {n}, en bouwt Docker er {m} opnieuw.",
                       "steps": {"one": "1 stap", "more": "{n} stappen"},
                       "alt": "Twee roosters: voor elk van vier bestanden de stappen van een Dockerfile op een rij, "
                              "gevuld als Docker de stap volgens de functie van de leerling opnieuw bouwt, met de "
                              "seconden ernaast."},
           "download": {"title": "Je to_download na elke verandering",
                        "intro": "De versie van gisteren van beide images staat op de server. Voor elk bestand dat "
                                 "vandaag verandert: wat de server volgens je `to_download` van het nieuwe image "
                                 "downloadt.",
                        "slow": "alles eerst", "quick": "op volgorde",
                        "none": "Je `to_download` stopt met een fout, duurt te lang, of geeft geen getal, dus is er "
                                "nog geen afbeelding.",
                        "said": "Na een verandering in service.py zegt je functie {a} MB voor het image dat alles "
                                "eerst kopieert, en {b} MB voor het image op volgorde.",
                        "right": "Dat klopt: alleen de lagen met een nieuwe id worden gedownload.",
                        "off": "De server downloadt {a} MB en {b} MB: de lagen waarvan de server de id nog niet heeft.",
                        "alt": "Balken voor vier veranderde bestanden, twee per bestand: de megabytes die de server "
                               "volgens de functie van de leerling downloadt van het image dat alles eerst kopieert, "
                               "en van het image op volgorde."}},
}


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


def _count(got, steps: list):
    """got as a number of steps of `steps`, or _STOPPED when it is not one."""
    if isinstance(got, bool) or not isinstance(got, int) or not 0 <= got <= len(steps):
        return _STOPPED
    return got


def _steps_text(n: int, lang: str) -> str:
    w = YOURS_TEXT[lang]["rebuilt"]["steps"]
    return w["one"] if n == 1 else w["more"].format(n=n)


def _key(steps: list) -> str:
    return ", ".join(f"{k} `{s['line']}`" for k, s in enumerate(steps))


def rebuilt_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["rebuilt"]
    fn = _fn(seen)
    blocks = (("quick", quick_steps()), ("slow", slow_steps()))
    counts = {}
    for label, steps in blocks:
        for name in FILES:
            got = _call(lambda: fn([dict(s, files=list(s["files"])) for s in steps], {name})) if fn else _STOPPED
            counts[(label, name)] = _count(got, steps) if got is not _STOPPED else _STOPPED
    if any(v is _STOPPED for v in counts.values()):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    parts, y = [], 0
    x0, box, gap = 112, 21, 3
    for label, steps in blocks:
        y += 16
        parts.append(_t(0, y, w[label].replace("`", ""), 12, 600))
        y += 14
        for k in range(len(steps)):
            parts.append(_t(x0 + k * (box + gap) + box / 2, y, str(k), 10.5, 400, "var(--ea-ink-3)", "middle"))
        y += 6
        for name in FILES:
            n = counts[(label, name)]
            parts.append(_t(0, y + 15, name, 10.5, 400, "var(--ea-ink-2)", font=_MONO))
            for k in range(len(steps)):
                parts.append(_box(x0 + k * (box + gap), y + 2, box, 18, k >= len(steps) - n))
            sec = number(seconds(steps, n), lang) + " s"
            parts.append(_t(x0 + 6 * (box + gap) + 4, y + 15, sec, 11.5, 600 if n else 400))
            y += 23
        y += 4
    y += 14
    parts.append(_legend(y, w["again"], w["cache"]))
    wrong = [(label, name) for label, steps in blocks for name in FILES
             if counts[(label, name)] != rebuilt(steps, {name})]
    if not wrong:
        said = w["right"]
    else:
        label, name = wrong[0]
        steps = dict(blocks)[label]
        said = w["off"].format(f=name, d=f"`{'C.QUICK' if label == 'quick' else 'C.SLOW'}`",
                               n=_steps_text(counts[(label, name)], lang), m=rebuilt(steps, {name}))
    key = w["key"].format(q=_key(quick_steps()), s=_key(slow_steps()))
    body = _note(w["intro"]) + _svg(y + 6, w["alt"], parts) + _note(key) + _note(said)
    return yours.figure(w["title"], body, alt=w["title"])


def _megabytes(got):
    if isinstance(got, bool) or not isinstance(got, (int, float)) or got != got or got < 0:
        return _STOPPED
    return float(got)


def download_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["download"]
    fn = _fn(seen)
    got = {}
    for name in FILES:
        for label, steps in (("slow", slow_steps()), ("quick", quick_steps())):
            new = image(steps, {name})
            mb = _call(lambda: fn([dict(layer) for layer in new], set(ON_SERVER))) if fn else _STOPPED
            got[(label, name)] = _megabytes(mb) if mb is not _STOPPED else _STOPPED
    if any(v is _STOPPED for v in got.values()):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    top = max(max(got.values()), facts()["all_slow"])
    x0, W = 112, 150
    parts, y = [], 4
    for name in FILES:
        parts.append(_t(0, y + 14, name, 10.5, 400, "var(--ea-ink-2)", font=_MONO))
        for k, label in enumerate(("slow", "quick")):
            mb = got[(label, name)]
            width = max(2.0, W * min(mb, top) / top) if top else 2.0
            style = "fill:var(--ea-ink)" if label == "slow" else "fill:var(--ea-ink-3)"
            yy = y + 3 + k * 15
            parts.append(f'<rect x="{x0}" y="{yy:.1f}" width="{width:.1f}" height="12" rx="2" style="{style}"/>')
            parts.append(_t(x0 + width + 5, yy + 10.5, number(mb, lang) + " MB", 11, 400))
        y += 38
    y += 12
    parts.append(f'<rect x="0" y="{y - 10}" width="16" height="12" rx="2" style="fill:var(--ea-ink)"/>'
                 + _t(22, y, w["slow"], 11.5, 400, "var(--ea-ink-2)")
                 + f'<rect x="160" y="{y - 10}" width="16" height="12" rx="2" style="fill:var(--ea-ink-3)"/>'
                 + _t(182, y, w["quick"], 11.5, 400, "var(--ea-ink-2)"))
    a, b = got[("slow", "service.py")], got[("quick", "service.py")]
    said = w["said"].format(a=number(a, lang), b=number(b, lang))
    right = all(abs(got[k] - facts()[k]["mb"]) < 1e-6 for k in got)
    if right:
        said += " " + w["right"]
    else:
        f = facts()
        said += " " + w["off"].format(a=number(f[("slow", "service.py")]["mb"], lang),
                                      b=number(f[("quick", "service.py")]["mb"], lang))
    body = _note(w["intro"]) + _svg(y + 6, w["alt"], parts) + _note(said)
    return yours.figure(w["title"], body, alt=w["title"])


# ======== the demo: both Dockerfiles for each changed file ================================================================

SHORT = {"RUN pip install -r requirements.txt": "RUN pip install …"}

DEMO_TEXT = {
    "en": {"title": "{f} changes",
           "slow": "Everything first", "quick": "In order",
           "row": "{s} s, {mb} MB",
           "steps": {"none": "nothing built again", "one": "1 step built again", "more": "{n} steps built again"},
           "count": {"none": "nothing", "one": "1 step", "more": "{n} steps"},
           "say": "A change to {f}: with everything copied first, Docker builds {a} again, in {sa} seconds, and a "
                  "server downloads {ma} MB. In order, {b}, in {sb} seconds, and {mb} MB.",
           "alt": "Two Dockerfiles side by side, one that copies everything first and one in order, with the steps "
                  "that Docker builds again filled, for the file the slider chooses."},
    "nl": {"title": "{f} verandert",
           "slow": "Alles eerst", "quick": "Op volgorde",
           "row": "{s} s, {mb} MB",
           "steps": {"none": "niets opnieuw gebouwd", "one": "1 stap opnieuw", "more": "{n} stappen opnieuw"},
           "count": {"none": "niets", "one": "1 stap", "more": "{n} stappen"},
           "say": "Een verandering in {f}: met alles eerst gekopieerd bouwt Docker {a} opnieuw, in {sa} seconden, en "
                  "downloadt een server {ma} MB. Op volgorde: {b}, in {sb} seconden, en {mb} MB.",
           "alt": "Twee Dockerfiles naast elkaar, de ene kopieert alles eerst en de andere werkt op volgorde, met "
                  "gevuld de stappen die Docker opnieuw bouwt, voor het bestand dat de schuif kiest."},
}


def _demo_steps(n: int, lang: str, key: str = "steps") -> str:
    w = DEMO_TEXT[lang][key]
    return w["none"] if n == 0 else w["one"] if n == 1 else w["more"].format(n=n)


def frame(value: float, lang: str) -> tuple[str, str]:
    lang = _lang(lang)
    w = DEMO_TEXT[lang]
    name = DEMO_FILES[int(value)]
    f = facts()
    parts = [_t(0, 15, w["title"].format(f=name), 13, 600)]
    col, gap = 158, 14
    bottom = 0
    for c, (label, steps) in enumerate((("slow", slow_steps()), ("quick", quick_steps()))):
        x = c * (col + gap)
        parts.append(_t(x, 38, w[label], 12, 600))
        n = rebuilt(steps, {name})
        y = 46
        for k, s in enumerate(steps):
            again = k >= len(steps) - n
            parts.append(_box(x, y, col, 20, again)
                         + _t(x + 6, y + 14, SHORT.get(s["line"], s["line"]), 10, 600 if again else 400,
                              "var(--ea-page)" if again else "var(--ea-ink)", font=_MONO))
            y += 23
        fact = f[(label, name)]
        parts.append(_t(x, y + 12, _demo_steps(n, lang), 11.5, 600)
                     + _t(x, y + 27, w["row"].format(s=number(fact["seconds"], lang), mb=number(fact["mb"], lang)),
                          11.5, 400, "var(--ea-ink-2)"))
        bottom = max(bottom, y + 34)
    a, b = f[("slow", name)], f[("quick", name)]
    say = w["say"].format(f=f"`{name}`", a=_demo_steps(a["n"], lang, "count"), sa=number(a["seconds"], lang),
                          ma=number(a["mb"], lang), b=_demo_steps(b["n"], lang, "count"), sb=number(b["seconds"], lang),
                          mb=number(b["mb"], lang))
    return _svg(bottom, w["alt"], parts), say


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

# exercise 1: rebuilt
_R_HEAD = ("def rebuilt(steps, changed):\n"
           "    for i in range(len(steps)):\n"
           "        for name in steps[i][\"files\"]:\n"
           "            if name in changed:\n")
_R_END = "    return 0\n"


def _r(expression: str) -> str:
    return _R_HEAD + f"                return {expression}\n" + _R_END


_R_START = _r("...")
_R = _r("len(steps) - i")
_R_ONE = _r("1")
_R_CACHED = _r("i")
_R_SHORT = _r("len(steps) - i - 1")
_R_ALL = _r("len(steps)")

# a made-up Dockerfile in which one step copies two files
_TWO = [{"line": "FROM python:3.12-slim", "files": []}, {"line": "COPY train.py data.csv .", "files": ["train.py", "data.csv"]},
        {"line": "RUN python train.py", "files": []}]

# exercise 2: to_download
_D_HEAD = ("def to_download(layers, have):\n"
           "    mb = 0\n"
           "    for layer in layers:\n")
_D_END = ("            mb = mb + layer[\"mb\"]\n"
          "    return mb\n")


def _d(condition: str) -> str:
    return _D_HEAD + f"        if {condition}:\n" + _D_END


_D_START = _d("...")
_D = _d('layer["id"] not in have')
_D_IN = _d('layer["id"] in have')

_ABC = [{"id": "a1", "line": "FROM python:3.12-slim", "mb": 10.0}, {"id": "b2", "line": "COPY a.py .", "mb": 5.0},
        {"id": "c3", "line": "COPY b.py .", "mb": 2.0}]


def _whole_layer(e: BaseException) -> bool:
    """`layer not in have`: a dictionary cannot be looked up in a set."""
    return isinstance(e, TypeError) and "unhashable" in str(e) and "dict" in str(e)


def _mb_shown(v):
    """A number of megabytes the way the lesson writes it: at most two decimals, and a decimal comma in Dutch."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return repr(v)
    return {"en": number(v, "en"), "nl": number(v, "nl")}


WHERE = Sort(
    name="where",
    groups={
        "image": {"en": "In the image", "nl": "In het image"},
        "host": {"en": "From the server", "nl": "Van de server"},
        "start": {"en": "Given at the start", "nl": "Bij de start meegegeven"},
    },
    items={
        "python": {"en": "**Python 3.12.** The Python that runs `service.py`.",
                   "nl": "**Python 3.12.** De Python die `service.py` draait."},
        "packages": {"en": "**The packages.** PyTorch, pandas and the rest, at the versions of `requirements.txt`.",
                     "nl": "**De packages.** PyTorch, pandas en de rest, in de versies van `requirements.txt`."},
        "model": {"en": "**The model.** The folder `model/`, with the weights and the model card.",
                  "nl": "**Het model.** De map `model/`, met de weights en de model card."},
        "kernel": {"en": "**The kernel.** The core of the operating system, which starts programs and hands out memory.",
                   "nl": "**De kernel.** De kern van het besturingssysteem: het deel dat programma's start en geheugen uitdeelt."},
        "driver": {"en": "**The NVIDIA driver.** The software through which programs reach the GPU of the computer.",
                   "nl": "**De NVIDIA-driver.** De software waarmee programma's de GPU van de computer bereiken."},
        "key": {"en": "**An API key.** The secret with which the service calls a paid service of another company.",
                "nl": "**Een API-key.** Het geheim waarmee de dienst een betaalde dienst van een ander bedrijf aanroept."},
        "port": {"en": "**The port.** The number at which the outside world reaches the service, such as 8000.",
                 "nl": "**De poort.** Het nummer waarop de buitenwereld de dienst bereikt, zoals 8000."},
        "logs": {"en": "**A folder for the log.** Where `requests.jsonl` goes, so the log is still there when the container is removed.",
                 "nl": "**Een map voor de log.** Waar `requests.jsonl` terechtkomt, zodat de log er nog is als de container verwijderd wordt."},
    },
    solution={"python": "image", "packages": "image", "model": "image", "kernel": "host", "driver": "host",
              "key": "start", "port": "start", "logs": "start"},
    nudge={
        "python": {"en": "Which line of the Dockerfile brings Python? Is that part of the image or of the server?",
                   "nl": "Welke regel van de Dockerfile brengt Python mee? Hoort deze regel bij het image of bij de server?"},
        "packages": {"en": "Which step of the Dockerfile installs them, and where do its files end up?",
                     "nl": "Welke stap van de Dockerfile installeert de packages, en waar komen de bestanden terecht?"},
        "model": {"en": "The Dockerfile has a COPY line for `model/`. Where does a COPY put its files?",
                  "nl": "De Dockerfile heeft een COPY-regel voor `model/`. Waar zet een COPY de bestanden neer?"},
        "kernel": {"en": "Does a container bring an operating system of its own, as a virtual machine does?",
                   "nl": "Brengt een container een eigen besturingssysteem mee, zoals een virtuele machine?"},
        "driver": {"en": "The driver belongs to the GPU of one particular computer. Can an image know which GPU it will meet?",
                   "nl": "De driver hoort bij de GPU van één bepaalde computer. Kan een image weten welke GPU het tegenkomt?"},
        "key": {"en": "Everyone who has the image can read every layer of it. Where should a secret be, then?",
                "nl": "Iedereen met het image kan elke laag ervan lezen. Waar hoort een geheim dan?"},
        "port": {"en": "The same image can run on many servers, each with numbers of its own. When is the number chosen?",
                 "nl": "Hetzelfde image kan op veel servers draaien, elk met eigen nummers. Wanneer kies je het nummer?"},
        "logs": {"en": "What a container writes is gone when the container is removed. Where must the log go to stay?",
                 "nl": "Wat een container schrijft is weg als de container verwijderd wordt. Waar moet de log heen om te blijven?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the eight parts again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de acht onderdelen opnieuw uit."},
        "empty": {"en": "Put each part under the image, the server, or what is given at the start.",
                  "nl": "Zet elk onderdeel onder het image, de server, of wat bij de start wordt meegegeven."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "*{item}* fits another column better.",
                  "nl": "*{item}* past beter bij een andere kolom."},
        "correct": {"en": "All {n} are right. In the image: what the service needs to run, the same on every server. From the server: the kernel and the driver, which belong to that computer. Given at the start: what differs per server or must stay out of the image, with `docker run`.",
                    "nl": "Alle {n} kloppen. In het image: wat de dienst nodig heeft om te draaien, op elke server hetzelfde. Van de server: de kernel en de driver, want de kernel en de driver horen bij de computer zelf. Bij de start meegegeven: wat per server verschilt of buiten het image moet blijven, met `docker run`."},
    },
)

EXERCISES = {
    "rebuilt": FunctionBox(
        name="rebuilt",
        function="rebuilt",
        start=_R_START,
        solution=_R,
        visible=[(QUICK, {"requirements.txt"})],
        hidden=[(QUICK, {"service.py"}), (QUICK, {"runs.jsonl"}), (SLOW, {"runs.jsonl"}), (QUICK, {"model/", "service.py"}),
                (SLOW, set()), ([], {"service.py"}), (_TWO, {"data.csv"})],
        first=[_DOTS],
        picture=rebuilt_picture,
        mistakes=[
            Mistake("only-one", wrong=_R_ONE,
                    en="`{call}` gives {got}, only the step that copies the changed file, and it should give {want}. But each layer sits on the layers before it. What happens to the steps after the changed one?",
                    nl="`{call}` geeft {got}, alleen de stap die het veranderde bestand kopieert, en het moet {want} zijn. Maar elke laag ligt op de eerdere lagen. Wat gebeurt er met de stappen na de veranderde stap?"),
            Mistake("cached", wrong=_R_CACHED,
                    en="`{call}` gives {got}, the steps before step `i`, which come from the cache, and it should give {want}. How many steps are left from step `i` on?",
                    nl="`{call}` geeft {got}: het aantal stappen uit de cache, tot stap `i`. Het moet {want} zijn. Hoeveel stappen blijven er over vanaf stap `i`?"),
            Mistake("one-short", wrong=_R_SHORT,
                    en="`{call}` gives {got}, one less than it should: {want}. Is the step that copies the changed file built again itself?",
                    nl="`{call}` geeft {got}, één minder dan het moet zijn: {want}. Wordt de stap die het veranderde bestand kopieert zelf opnieuw gebouwd?"),
            Mistake("all", wrong=_R_ALL,
                    en="`{call}` gives {got}, every step of the Dockerfile, and it should give {want}. The steps before step `i` copy nothing that changed. Where do their layers come from?",
                    nl="`{call}` geeft {got}, elke stap van de Dockerfile, en het moet {want} zijn. De stappen tot stap `i` kopiëren niets wat veranderd is. Waar komen de lagen van deze stappen vandaan?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with how many steps Docker builds again when step `i` is the first to copy a changed file.",
                      "nl": "Vervang de `...` door het aantal stappen dat Docker opnieuw bouwt als stap `i` als eerste een veranderd bestand kopieert."},
            "arguments": {"en": "The check calls `{call}`, with a list of steps and a set of file names, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(steps, changed)`?",
                          "nl": "De check roept `{call}` aan, met een lijst stappen en een set bestandsnamen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(steps, changed)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your number count the steps from step `i` up to the last one?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Telt je getal de stappen vanaf stap `i` tot en met de laatste?"},
            "almost": {"en": "Right for `C.QUICK` with a new requirements.txt. But `{call}` gives {got}, and it should give {want}. Does your number come from `steps` and `i`, and not from a fixed number?",
                       "nl": "Goed voor `C.QUICK` met een nieuwe requirements.txt. Maar `{call}` geeft {got}, en het moet {want} zijn. Komt je getal uit `steps` en `i`, en niet uit een vast getal?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does every way through it end with `return` and a number?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt elke weg door je functie met `return` en een getal?"},
            "correct": {"en": "Right in all {n} cases. With a new requirements.txt, Docker builds 4 of the 6 steps of `C.QUICK` again, the install included; with a new service.py, only the last one. In the picture: which file costs the most seconds in each Dockerfile?",
                        "nl": "Goed in alle {n} gevallen. Met een nieuwe requirements.txt bouwt Docker 4 van de 6 stappen van `C.QUICK` opnieuw, de installatie ook; met een nieuwe service.py alleen de laatste. In de afbeelding: welk bestand kost in elke Dockerfile de meeste seconden?"},
        },
    ),
    "to_download": FunctionBox(
        name="to_download",
        function="to_download",
        start=_D_START,
        solution=_D,
        visible=[(SLOW_IMAGE, ON_SERVER), (QUICK_IMAGE, ON_SERVER)],
        hidden=[([], ON_SERVER), (QUICK_IMAGE, set()), (_ABC, {"b2"})],
        first=[_DOTS],
        picture=download_picture,
        show=_mb_shown,
        mistakes=[
            Mistake("in-have", wrong=_D_IN,
                    en="`{call}` gives {got}, the layers the server has already, and it should give {want}. Which layers does the server still lack?",
                    nl="`{call}` geeft {got}, de lagen die de server al heeft, en het moet {want} zijn. Welke lagen mist de server nog?"),
            Mistake("whole-layer", error=_whole_layer, state="error",
                    en="With `{call}`, your function stops with an error: Python cannot look up a whole dictionary in a set. `layer` is a dictionary, and `have` holds ids. Which key of `layer` holds its id?",
                    nl="Met `{call}` stopt je functie met een fout: Python kan geen hele dictionary opzoeken in een set. `layer` is een dictionary, en `have` bevat ids. Welke key van `layer` bevat de id?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the question whether the server still lacks `layer`.",
                      "nl": "Vervang de `...` door de vraag of de server `layer` nog mist."},
            "arguments": {"en": "The check calls `{call}`, with a list of layers and a set of ids, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(layers, have)`?",
                          "nl": "De check roept `{call}` aan, met een lijst lagen en een set ids, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(layers, have)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your total count a layer only when the server does not have its id yet?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Telt je totaal een laag alleen mee als de server de id ervan nog niet heeft?"},
            "almost": {"en": "Right for the two images of the check. But `{call}` gives {got}, and it should give {want}. Does your function add up the layers it gets, and not fixed numbers?",
                       "nl": "Goed voor de twee images van de check. Maar `{call}` geeft {got}, en het moet {want} zijn. Telt je functie de gekregen lagen op, en geen vaste getallen?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return mb`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return mb`?"},
            "correct": {"en": "Right in all {n} cases. After a change to service.py, the server downloads 383.05 MB of the image that copies everything first, and 0.02 MB of the image in order: only the layer with the new code. In the picture: which change makes the image in order heavy to download too?",
                        "nl": "Goed in alle {n} gevallen. Na een verandering in service.py downloadt de server 383,05 MB van het image dat alles eerst kopieert, en 0,02 MB van het image op volgorde: alleen de laag met de nieuwe code. In de afbeelding: bij welke verandering is ook het image op volgorde zwaar om te downloaden?"},
        },
    ),
    "where": WHERE,
}

# The lesson is done when these are correct. "platform" is the choice question before the demo.
REQUIRED = ["rebuilt", "to_download", "where", "platform"]

_SORTED = dict(WHERE.solution)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("rebuilt", _R_START, "empty", None),
    ("rebuilt", "", "empty", None),
    ("rebuilt", _R, "correct", None),
    ("rebuilt", _r("-i + len(steps)"), "correct", None),
    ("rebuilt", _r("len(steps[i:])"), "correct", None),
    ("rebuilt", _R_ONE, "wrong", "only-one"),
    ("rebuilt", _R_CACHED, "wrong", "cached"),
    ("rebuilt", _R_SHORT, "wrong", "one-short"),
    ("rebuilt", _R_ALL, "wrong", "all"),
    ("rebuilt", _r("i + 1"), "wrong", None),
    ("rebuilt", _r("len(steps) - 2"), "almost", None),
    ("rebuilt", "def rebuilt(steps, changed):\n    return 4\n", "almost", None),
    ("rebuilt", _r("len(step) - i"), "error", None),
    ("rebuilt", _R.replace("def rebuilt(steps, changed):", "def rebuilt(steps):"), "error", "arguments"),
    ("rebuilt", _R.replace("                return len(steps) - i\n", "                count = len(steps) - i\n").replace("    return 0\n", ""),
     "wrong", "no-return"),
    ("rebuilt", _R.replace("                return len(steps) - i\n", "                print(len(steps) - i)\n").replace("    return 0\n", ""),
     "wrong", "prints"),
    ("rebuilt", _R.replace("    return 0\n", "    extra = ...\n    return 0\n"), "wrong", "dots-left"),
    ("rebuilt", "x = 1\n", "wrong", "no-function"),
    ("to_download", _D_START, "empty", None),
    ("to_download", "", "empty", None),
    ("to_download", _D, "correct", None),
    ("to_download", _d('not layer["id"] in have'), "correct", None),
    ("to_download", _d('have.isdisjoint([layer["id"]])'), "correct", None),
    ("to_download", _D_IN, "wrong", "in-have"),
    ("to_download", _d('layer not in have'), "error", "whole-layer"),
    ("to_download", _d('layer["mb"] not in have'), "wrong", None),
    ("to_download", _d('layer["id"] not in have and layer["mb"] > 0.01'), "almost", None),
    ("to_download", "def to_download(layers, have):\n    return 383.05 if len(layers) == 4 else 0.02\n", "almost", None),
    ("to_download", _d('layer["name"] not in have'), "error", None),
    ("to_download", _D.replace("def to_download(layers, have):", "def to_download(layers):"), "error", "arguments"),
    ("to_download", _D.replace("    return mb\n", ""), "wrong", "no-return"),
    ("to_download", _D.replace("    return mb\n", "    print(mb)\n"), "wrong", "prints"),
    ("to_download", _D.replace("    return mb\n", "    extra = ...\n    return mb\n"), "wrong", "dots-left"),
    ("to_download", "x = 1\n", "wrong", "no-function"),
    ("where", {}, "empty", None),
    ("where", _SORTED, "correct", None),
    ("where", {"python": "image", "packages": "image"}, "wrong", "unfinished"),
    ("where", _SORTED | {"python": "host"}, "wrong", "wrong:python"),
    ("where", _SORTED | {"packages": "start"}, "wrong", "wrong:packages"),
    ("where", _SORTED | {"model": "start"}, "wrong", "wrong:model"),
    ("where", _SORTED | {"kernel": "image"}, "wrong", "wrong:kernel"),
    ("where", _SORTED | {"driver": "image"}, "wrong", "wrong:driver"),
    ("where", _SORTED | {"key": "image"}, "wrong", "wrong:key"),
    ("where", _SORTED | {"port": "image"}, "wrong", "wrong:port"),
    ("where", _SORTED | {"logs": "image"}, "wrong", "wrong:logs"),
]
