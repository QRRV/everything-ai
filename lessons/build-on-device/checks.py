# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-on-device: how long a model takes to download, from its size in MB and a connection in Mbit/s, and
the most bits per weight at which a model's weights fit in the memory a device has; the picture of what goes over the
network, the pictures of the learner's two functions, the lesson's numbers, and the demo with both functions for the
learner's own computer."""
import html
import re

from everything_ai import system, yours
from everything_ai.check import FunctionBox, Mistake, run_code

QWEN_MB = 1506            # what the course downloads for Qwen3-0.6B, its weights and its tokenizer (chat.size_mb())
DIGIT_WEIGHTS = 50_890    # the digit model of torch-mlp-digits, its MLP: 784 * 64 + 64 + 64 * 10 + 10
BITS = (32, 16, 8, 4)


def _nr(x: float, lang: str, digits: int = 1) -> str:
    text = f"{x:,.{digits}f}"
    return text.translate(str.maketrans(",.", ".,")) if lang == "nl" else text


def gb(params: float, bits: int) -> float:
    return params * bits / 8 / 1e9


def seconds(megabytes: float, megabits_per_second: float) -> float:
    return megabytes * 8 / megabits_per_second


def best(params: float, memory_gb: float):
    for bits in BITS:
        if gb(params, bits) <= memory_gb:
            return bits
    return None


def duration(s: float, lang: str) -> str:
    """A time in seconds as a reader says it: 0.03 s, 45 s, 4 min, 2.5 h."""
    if s != s or s in (float("inf"), float("-inf")):
        return str(s)
    if abs(s) < 1:
        return _nr(s, lang, 2) + " s"
    if abs(s) < 120:
        return _nr(s, lang, 0) + " s"
    if abs(s) < 2 * 3600:
        return _nr(s / 60, lang, 0) + " min"
    return _nr(s / 3600, lang, 1) + " " + ("h" if lang == "en" else "uur")


# --- the picture of the idea: what goes over the network ----------------------------------------------------------------

_NET_WORDS = {
    "en": {"title": "Two places for the model", "server": "A model on a server", "device": "A model on the device",
           "yours": "your device", "host": "server", "model": "model", "ask": ("your question,", "with your data"),
           "answer": "the answer", "every": "with every question", "once": "the model, once", "hub": "a server or hub",
           "inside": "question and answer stay here"},
    "nl": {"title": "Twee plekken voor het model", "server": "Een model op een server", "device": "Een model op het apparaat",
           "yours": "je apparaat", "host": "server", "model": "model", "ask": ("je vraag,", "met je data"),
           "answer": "het antwoord", "every": "bij elke vraag", "once": "het model, één keer", "hub": "een server of hub",
           "inside": "vraag en antwoord blijven hier"},
}


def _box(x, y, w, h, label, stroke="var(--ea-ink-2)", dash=""):
    d = f";stroke-dasharray:{dash}" if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" style="fill:none;stroke:{stroke};stroke-width:1.5{d}"/>'
            f'<text x="{x + w / 2}" y="{y + 18}" text-anchor="middle" style="font-size:12px;font-family:var(--ea-font-ui);fill:var(--ea-ink-2)">{html.escape(label)}</text>')


def _model(x, y, label):
    return (f'<rect x="{x}" y="{y}" width="64" height="30" rx="4" style="fill:none;stroke:var(--ea-param);stroke-width:2"/>'
            f'<text x="{x + 32}" y="{y + 19}" text-anchor="middle" style="font-size:12px;font-family:var(--ea-font-ui);fill:var(--ea-ink)">{html.escape(label)}</text>')


def _arrow(x1, y1, x2, y2, role, dash="", panel="l"):
    d = f";stroke-dasharray:{dash}" if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" style="stroke:var(--ea-{role});stroke-width:1.8{d}" '
            f'marker-end="url(#ea-od-{panel}-head-{role})"/>')


def _text(x, y, words, colour="var(--ea-ink)", anchor="middle", size=12):
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" style="font-size:{size}px;font-family:var(--ea-font-ui);'
            f'fill:{colour}">{html.escape(words)}</text>')


def _heads(panel: str) -> str:
    return "".join(f'<marker id="ea-od-{panel}-head-{c}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
                   f'<path d="M0,0 L10,5 L0,10 z" style="fill:var(--ea-{c})"/></marker>' for c in ("data", "pred", "ink-2"))


def network_picture(lang: str) -> str:
    """Two panels: with the model on a server, every question goes over the network with its data and every answer
    comes back; with the model on the device, only the model comes over, once, and question and answer stay there."""
    w = _NET_WORDS[lang]
    left = (f"<defs>{_heads('l')}</defs>" + _text(170, 16, w["server"], size=13)
            + _box(10, 40, 90, 120, w["yours"]) + _box(240, 40, 90, 120, w["host"]) + _model(253, 84, w["model"])
            + _text(170, 60, w["ask"][0], "var(--ea-data-text)", size=11) + _text(170, 74, w["ask"][1], "var(--ea-data-text)", size=11)
            + _arrow(102, 82, 236, 82, "data")
            + _arrow(236, 124, 102, 124, "pred") + _text(170, 142, w["answer"], "var(--ea-pred-text)", size=11)
            + _text(170, 182, w["every"], "var(--ea-ink-2)", size=11))
    right = (f"<defs>{_heads('r')}</defs>" + _text(170, 16, w["device"], size=13)
             + _box(200, 30, 130, 40, w["hub"], dash="4 3")
             + _arrow(220, 72, 150, 98, "ink-2", dash="4 3", panel="r") + _text(214, 96, w["once"], "var(--ea-ink-2)", "start", 11)
             + _box(40, 100, 200, 80, w["yours"]) + _model(108, 126, w["model"])
             + _arrow(62, 141, 104, 141, "data", panel="r") + _arrow(176, 141, 218, 141, "pred", panel="r")
             + _text(140, 196, w["inside"], "var(--ea-ink-2)", size=11))
    svgs = "".join(f'<svg viewBox="0 0 340 204" role="img" style="flex:1 1 300px;max-width:340px;height:auto">'
                   f'<title>{html.escape(t)}</title>{body}</svg>' for t, body in ((w["server"], left), (w["device"], right)))
    body = f'<div style="display:flex;flex-wrap:wrap;gap:12px 28px;align-items:flex-start">{svgs}</div>'
    return yours.figure(w["title"], body, alt=w["title"])


# --- exercise 1: how long the download takes ----------------------------------------------------------------------------

_START_DOWNLOAD = '''def download_seconds(megabytes, megabits_per_second):
    return 0
'''
SOLUTION_DOWNLOAD = '''def download_seconds(megabytes, megabits_per_second):
    return megabytes * 8 / megabits_per_second
'''
DOWNLOAD_CASES = [(QWEN_MB, 50), (0.2, 50), (4000, 50), (4000, 500), (QWEN_MB, 10), (1, 8)]
# the sizes of the picture: the digit model in float32, Qwen3-0.6B as the course downloads it, and a model of
# 8 billion weights at 4 bits
_SIZES = [(DIGIT_WEIGHTS * 4 / 1e6, {"en": "the digit model", "nl": "het cijfermodel"}),
          (QWEN_MB, {"en": "Qwen3-0.6B", "nl": "Qwen3-0.6B"}),
          (gb(8e9, 4) * 1000, {"en": "8 billion weights, 4 bits", "nl": "8 miljard weights, 4 bits"})]
SPEEDS = (10, 100)


def _learner(seen, name: str):
    run = seen.mine
    fn = run.names.get(name) if run is not None else None
    return fn if callable(fn) else None


def _cell(text: str, ok: bool, right: str, should: str) -> str:
    colour = "var(--ea-ink)" if ok else "var(--ea-loss-text)"
    extra = "" if ok else f'<br><span style="color:var(--ea-ink-2)">{html.escape(should)} {html.escape(right)}</span>'
    return f'<td style="font-family:var(--ea-font-code);color:{colour};padding:0.2rem 0.8rem 0.2rem 0;vertical-align:top">{html.escape(text)}{extra}</td>'


def _table(head: list[str], rows: list[str]) -> str:
    th = "".join(f'<th style="text-align:left;font-weight:500;color:var(--ea-ink-2);padding:0 0.8rem 0.3rem 0">{html.escape(h)}</th>' for h in head)
    return (f'<table style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink);border-collapse:collapse">'
            f'<tr>{th}</tr>{"".join(rows)}</table>')


def download_picture(seen) -> str:
    """What the learner's download_seconds says for three sizes, at 10 and at 100 Mbit/s, beside what it should say."""
    fn = _learner(seen, "download_seconds")
    if fn is None:
        return ""
    lang = seen.lang
    w = {"en": ("Your download_seconds", "model", "should be", "error"),
         "nl": ("Je download_seconds", "model", "zou zijn", "fout")}[lang]
    rows = []
    for megabytes, name in _SIZES:
        cells = [f'<td style="padding:0.2rem 0.8rem 0.2rem 0;vertical-align:top">{html.escape(name[lang])}, '
                 f'<span style="font-family:var(--ea-font-code)">{_nr(megabytes, lang, 1 if megabytes < 10 else 0)} MB</span></td>']
        for speed in SPEEDS:
            right = seconds(megabytes, speed)
            try:
                got = float(fn(megabytes, speed))
                text, ok = duration(got, lang), abs(got - right) <= 1e-6 * right
            except Exception:   # the learner's code
                text, ok = w[3], False
            cells.append(_cell(text, ok, duration(right, lang), w[2]))
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return yours.figure(w[0], _table([w[1]] + [f"{s} Mbit/s" for s in SPEEDS], rows), alt=w[0])


# --- exercise 2: the most bits that fit ---------------------------------------------------------------------------------

_START_BITS = '''def best_bits(params, memory_gb):
    return None
'''
SOLUTION_BITS = '''def best_bits(params, memory_gb):
    for bits in [32, 16, 8, 4]:
        if params * bits / 8 / 1_000_000_000 <= memory_gb:
            return bits
    return None
'''
# no case where the weights take the memory exactly: whether that fits is not what the exercise is about
BITS_CASES = [(600_000_000, 1), (600_000_000, 5), (3_000_000_000, 2), (8_000_000_000, 24), (70_000_000_000, 24),
              (70_000_000_000, 80), (8_000_000_000, 2), (DIGIT_WEIGHTS, 0.001)]
GRID_MODELS = [(600_000_000, {"en": "0.6 billion", "nl": "0,6 miljard"}), (3e9, {"en": "3 billion", "nl": "3 miljard"}),
               (8e9, {"en": "8 billion", "nl": "8 miljard"}), (70e9, {"en": "70 billion", "nl": "70 miljard"})]
GRID_MEMORY = (2, 5, 24, 80)


class Nothing:
    """What best_bits gives when no number of bits fits, None, as the check holds it: a None that is not taken for a
    function that forgot its return (check.FunctionBox), and that shows as None."""
    def __eq__(self, other):
        return isinstance(other, Nothing)

    def __hash__(self):
        return 0

    def __repr__(self):
        return "None"

    __str__ = __repr__


NOTHING = Nothing()


def _held(value):
    return NOTHING if value is None else value


def _bits_text(value) -> str:
    return "None" if value is None or isinstance(value, Nothing) else f"{value}"


def bits_picture(seen) -> str:
    """What the learner's best_bits picks for four models in four sizes of memory, beside what it should pick."""
    fn = _learner(seen, "best_bits")
    if fn is None:
        return ""
    lang = seen.lang
    w = {"en": ("Your best_bits", "model", "should be", "error"), "nl": ("Je best_bits", "model", "zou zijn", "fout")}[lang]
    rows = []
    for params, name in GRID_MODELS:
        cells = [f'<td style="padding:0.2rem 0.8rem 0.2rem 0;vertical-align:top">{html.escape(name[lang])} weights</td>']
        for memory in GRID_MEMORY:
            right = best(params, memory)
            try:
                got = fn(params, memory)
                ok = got == right and (got is None) == (right is None)
                text = _bits_text(got)
            except Exception:   # the learner's code
                text, ok = w[3], False
            cells.append(_cell(text, ok, _bits_text(right), w[2]))
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return yours.figure(w[0], _table([w[1]] + [f"{m} GB" for m in GRID_MEMORY], rows), alt=w[0])


_COMMON = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
                "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."},
}

EXERCISES = {
    "download_seconds": FunctionBox(
        name="download_seconds",
        picture=download_picture,
        function="download_seconds",
        start=_START_DOWNLOAD,
        solution=SOLUTION_DOWNLOAD,
        visible=DOWNLOAD_CASES[:1],
        hidden=DOWNLOAD_CASES[1:],
        mistakes=[
            Mistake("no-8", wrong=SOLUTION_DOWNLOAD.replace(" * 8", ""),
                    en="`{call}` gives `{got}`, eight times too fast: the size is in megabytes and the connection in megabits. A byte has 8 bits.",
                    nl="`{call}` geeft `{got}`, acht keer te snel: de grootte is in megabytes en de verbinding in megabits. Een byte heeft 8 bits."),
            Mistake("div-8", wrong=SOLUTION_DOWNLOAD.replace("megabytes * 8 / megabits_per_second", "megabytes / 8 / megabits_per_second"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. A megabyte is 8 megabits: a file has more megabits than megabytes, so multiply by 8.",
                    nl="`{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Een megabyte is 8 megabits: een bestand heeft meer megabits dan megabytes, dus vermenigvuldig met 8."),
            Mistake("upside-down", wrong=SOLUTION_DOWNLOAD.replace("megabytes * 8 / megabits_per_second", "megabits_per_second / (megabytes * 8)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. A bigger file should take longer, and yours goes faster. Which of the two belongs on top?",
                    nl="`{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Een groter bestand zou langer moeten duren, en bij jou gaat het sneller. Welke van de twee hoort bovenaan?"),
            Mistake("minutes", wrong=SOLUTION_DOWNLOAD.replace("megabits_per_second\n", "megabits_per_second / 60\n"),
                    en="`{call}` gives `{got}`: those are minutes. The function gives seconds.",
                    nl="`{call}` geeft `{got}`: dat zijn minuten. De functie geeft seconden."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "`download_seconds` gives 0 yet. How many megabits is a file of `megabytes` megabytes, and how many seconds do they take at `megabits_per_second`?",
                      "nl": "`download_seconds` geeft nog 0. Hoeveel megabits is een bestand van `megabytes` megabytes, en hoeveel seconden kosten die bij `megabits_per_second`?"},
            "correct": {"en": "Right: your `download_seconds` turns megabytes into megabits, and divides by the speed, for all {n} downloads the check tried.",
                        "nl": "Goed: je `download_seconds` maakt van megabytes megabits, en deelt door de snelheid, voor alle {n} downloads die de check probeerde."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: 1,506 MB is 12,048 megabits, at 50 megabits per second.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: 1.506 MB is 12.048 megabits, bij 50 megabits per seconde."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`.",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn."},
        },
    ),
    "best_bits": FunctionBox(
        name="best_bits",
        picture=bits_picture,
        function="best_bits",
        start=_START_BITS,
        solution=SOLUTION_BITS,
        visible=BITS_CASES[:1],
        hidden=BITS_CASES[1:],
        collect=_held,
        mistakes=[
            Mistake("fewest", wrong=SOLUTION_BITS.replace("[32, 16, 8, 4]", "[4, 8, 16, 32]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your function stops at the fewest bits that fit. More bits keep the weights more precise; which number of bits should it try first?",
                    nl="`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: je functie stopt bij de minste bits die passen. Meer bits houden de weights preciezer; welk aantal bits moet hij eerst proberen?"),
            Mistake("bits-as-bytes", wrong=SOLUTION_BITS.replace(" / 8 ", " "),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your weights take eight times too much memory. `bits` are bits, and a byte has 8.",
                    nl="`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: je weights nemen acht keer te veel geheugen in. `bits` zijn bits, en een byte heeft er 8."),
            Mistake("prints", matches=lambda got, text: isinstance(got, Nothing) and "print(" in text and not re.search(r"return\s+bits", text),
                    en="`{call}` gives `None`: your function shows the bits with `print`, but gives nothing back. Use `return bits` instead of `print`, so the bits come out of the function.",
                    nl="`{call}` geeft `None`: je functie laat de bits zien met `print`, maar geeft niets terug. Gebruik `return bits` in plaats van `print`, dan komen de bits uit de functie."),
            Mistake("zero", wrong=SOLUTION_BITS.replace("    return None\n", "    return 0\n"),
                    en="`{call}` gives `{got}`. When no number of bits fits, the function gives `None`: nothing fits.",
                    nl="`{call}` geeft `{got}`. Als geen aantal bits past, geeft de functie `None`: niets past."),
            Mistake("first-only", wrong='''def best_bits(params, memory_gb):
    for bits in [32, 16, 8, 4]:
        if params * bits / 8 / 1_000_000_000 <= memory_gb:
            return bits
        return None
''',
                    en="`{call}` gives `{got}`, and it should give `{want}`. Your function gives up after 32 bits: is `return None` inside the loop? It belongs after it, once all four have been tried.",
                    nl="`{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Je functie geeft het op na 32 bits: staat `return None` in de loop? Hij hoort erna, als ze alle vier geprobeerd zijn."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "`best_bits` gives `None` yet, also for Qwen3-0.6B in 1 GB. Try the numbers of bits one by one, and give back the first at which the weights fit.",
                      "nl": "`best_bits` geeft nog `None`, ook voor Qwen3-0.6B in 1 GB. Probeer de aantallen bits een voor een, en geef het eerste terug waarbij de weights passen."},
            "correct": {"en": "Right: your `best_bits` gives the most bits that fit, or `None` when even 4 bits do not, for all {n} models the check tried.",
                        "nl": "Goed: je `best_bits` geeft de meeste bits die passen, of `None` als zelfs 4 bits niet passen, voor alle {n} modellen die de check probeerde."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: in `float32` the weights take 2.4 GB, in `bfloat16` 1.2 GB, at 8 bits 0.6 GB.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: in `float32` nemen de weights 2,4 GB in, in `bfloat16` 1,2 GB, met 8 bits 0,6 GB."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`.",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn."},
        },
    ),
}

REQUIRED = ["download_seconds", "best_bits", "device", "tool"]

CASES = [
    ("download_seconds", _START_DOWNLOAD, "empty", None),
    ("download_seconds", SOLUTION_DOWNLOAD, "correct", None),
    ("download_seconds", "def download_seconds(megabytes, megabits_per_second):\n    return 8 * megabytes / megabits_per_second\n", "correct", None),
    ("download_seconds", SOLUTION_DOWNLOAD.replace(" * 8", ""), "wrong", "no-8"),
    ("download_seconds", SOLUTION_DOWNLOAD.replace("megabytes * 8 / megabits_per_second", "megabytes / 8 / megabits_per_second"), "wrong", "div-8"),
    ("download_seconds", SOLUTION_DOWNLOAD.replace("megabytes * 8 / megabits_per_second", "megabits_per_second / (megabytes * 8)"), "wrong", "upside-down"),
    ("download_seconds", SOLUTION_DOWNLOAD.replace("megabits_per_second\n", "megabits_per_second / 60\n"), "wrong", "minutes"),
    ("download_seconds", SOLUTION_DOWNLOAD.replace(" * 8", " * 8 + 1"), "wrong", None),
    ("best_bits", _START_BITS, "empty", None),
    ("best_bits", SOLUTION_BITS, "correct", None),
    ("best_bits", '''def best_bits(params, memory_gb):
    answer = None
    for bits in [4, 8, 16, 32]:
        if params * bits / 8 / 1e9 <= memory_gb:
            answer = bits
    return answer
''', "correct", None),
    ("best_bits", SOLUTION_BITS.replace("[32, 16, 8, 4]", "[4, 8, 16, 32]"), "wrong", "fewest"),
    ("best_bits", SOLUTION_BITS.replace(" / 8 ", " "), "wrong", "bits-as-bytes"),
    ("best_bits", SOLUTION_BITS.replace("    return None\n", "    return 0\n"), "wrong", "zero"),
    ("best_bits", '''def best_bits(params, memory_gb):
    for bits in [32, 16, 8, 4]:
        if params * bits / 8 / 1_000_000_000 <= memory_gb:
            return bits
        return None
''', "wrong", "first-only"),
    ("best_bits", SOLUTION_BITS.replace("[32, 16, 8, 4]", "[32, 16, 4]"), "wrong", None),
    ("best_bits", SOLUTION_BITS.replace("            return bits\n", "            print(bits)\n            break\n"), "wrong", "prints"),
]


# --- the demo: what fits on your own computer ---------------------------------------------------------------------------

# each with the bits it was made with: more add nothing
DEMO_MODELS = [(DIGIT_WEIGHTS, 32, {"en": "the digit model", "nl": "het cijfermodel"}),
               (600_000_000, 16, {"en": "Qwen3-0.6B", "nl": "Qwen3-0.6B"}),
               (3_000_000_000, 16, {"en": "a model of 3 billion", "nl": "een model van 3 miljard"}),
               (8_000_000_000, 16, {"en": "a model of 8 billion", "nl": "een model van 8 miljard"}),
               (70_000_000_000, 16, {"en": "a model of 70 billion", "nl": "een model van 70 miljard"})]


def memory_gb() -> float | None:
    """The memory of this computer in GB, billions of bytes, as the system tells Python (on WSL2: the part Windows
    gives it), or None."""
    return system._ram_gb()


def learner(text, name: str):
    """The learner's function from the box, when it runs; None otherwise."""
    if not text:
        return None
    run = run_code(text)
    fn = run.names.get(name) if run.error is None else None
    return fn if callable(fn) else None


def demo(download_fn, bits_fn, memory: float, speed: float, words: dict, lang: str) -> str:
    """How much memory this computer has, and a table: for each model of DEMO_MODELS, the bits that the learner's
    best_bits picks for half of `memory`, at most the bits the model was made with, the size of the weights at those
    bits, and how long the learner's
    download_seconds says they take at `speed`. `words` has "memory" ({memory}), "head" (four, with {room} and
    {speed}), "none" and "failed" ({why})."""
    room = memory / 2
    rows = []
    for params, made, name in DEMO_MODELS:
        try:
            bits = bits_fn(params, room)
            if bits is None:
                cells = [words["none"], "", ""]
            else:
                bits = min(bits, made)
                size = gb(params, bits)
                megabytes = size * 1000
                shown = f"{_nr(megabytes, lang)} MB" if size < 0.1 else f"{_nr(size, lang)} GB"
                cells = [f"{bits} bits", shown,
                         duration(float(download_fn(megabytes, speed)), lang)]
        except Exception as e:   # the learner's code
            cells = [words["failed"].format(why=type(e).__name__), "", ""]
        tds = "".join(f'<td style="padding:0.25rem 1rem 0.25rem 0;font-family:var(--ea-font-code)">{html.escape(c)}</td>' for c in cells)
        rows.append(f'<tr><td style="padding:0.25rem 1rem 0.25rem 0">{html.escape(name[lang])}</td>{tds}</tr>')
    head = [h.format(room=_nr(room, lang), speed=_nr(speed, lang, 0)) for h in words["head"]]
    line = f'<p style="margin:0 0 0.6rem">{html.escape(words["memory"].format(memory=_nr(memory, lang)))}</p>'
    return line + _table(head, rows)
