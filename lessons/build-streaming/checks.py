# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-streaming: reading one line of a stream of server-sent events, and a function that asks for a
stream and collects its pieces, against a stand-in of the server; the course's recorded streams of Qwen3 through its
own server (data/course-apis-qwen.json), with the second at which each piece came in; and the pictures."""
import html
import json
import time
import urllib.request

from everything_ai import apicheck, chat, paths, serve, yours
from everything_ai.apicheck import URL, ApiBox, Case
from everything_ai.check import Code, FunctionBox, Mistake, run_code

STORY = "Write a short story of five sentences about a lighthouse keeper."
RECORD = {"story": {"messages": [{"role": "user", "content": STORY}], "max_tokens": 200, "seeds": 3}}
DATA = paths.ROOT / "data" / "course-apis-qwen.json"


def course() -> list:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-streaming"]["story"]["answers"]


def findings() -> dict:
    """Seed 0 of the course's streams: seconds to the first piece and to the end, the pieces and tokens; and the
    slowest first piece and quickest end of the three runs."""
    runs = course()
    a = runs[0]
    return {"first": a["first"], "seconds": a["seconds"], "pieces": a["pieces"], "tokens": a["tokens"],
            "first_max": max(r["first"] for r in runs), "seconds_min": min(r["seconds"] for r in runs), "runs": len(runs)}


def _n(x: float, lang: str, digits: int = 1) -> str:
    s = f"{x:.{digits}f}"
    return s.replace(".", ",") if lang == "nl" else s


def numbers(lang: str) -> dict:
    """The findings as the lesson's text writes them: seconds with one decimal, in the lesson's language."""
    return {k: _n(v, lang) if isinstance(v, float) else str(v) for k, v in findings().items()}


_TIME_WORDS = {
    "en": {"title": "When the pieces of the answer came in", "stream": "with streaming: every piece as it is written",
           "whole": "without: the whole answer at once", "first": "first piece after {s} s", "end": "the end after {s} s",
           "axis": "seconds after the request", "nothing": "nothing to see"},
    "nl": {"title": "Wanneer de stukjes van het antwoord binnenkwamen", "stream": "met streaming: elk stukje zodra het geschreven is",
           "whole": "zonder: het hele antwoord in één keer", "first": "eerste stukje na {s} s", "end": "het einde na {s} s",
           "axis": "seconden na het request", "nothing": "niets te zien"},
}


def timeline_picture(lang: str) -> str:
    """Seed 0 of the course's streams on a time line: a tick for every piece at the second it came in, in the colour
    of what a model writes; under it, the same answer without streaming, which shows nothing until the end."""
    w, a = _TIME_WORDS[lang], course()[0]
    W, L, R = 720, 20, 20
    end = a["seconds"]
    scale = (W - L - R) / end

    def x(s):
        return L + s * scale
    parts = [f'<text x="{L}" y="18" style="font-size:12px;font-family:var(--ea-font-ui);fill:var(--ea-ink)">{html.escape(w["stream"])}</text>']
    for t in a["times"]:
        parts.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="28" y2="50" style="stroke:var(--ea-pred-text);stroke-width:1.2"/>')
    parts.append(f'<text x="{x(a["first"]) + 4:.1f}" y="64" style="font-size:11px;font-family:var(--ea-font-ui);fill:var(--ea-ink-2)">'
                 f'{html.escape(w["first"].format(s=_n(a["first"], lang)))}</text>')
    parts.append(f'<text x="{L}" y="94" style="font-size:12px;font-family:var(--ea-font-ui);fill:var(--ea-ink)">{html.escape(w["whole"])}</text>')
    parts.append(f'<line x1="{L}" x2="{x(end):.1f}" y1="114" y2="114" style="stroke:var(--ea-line);stroke-width:6;stroke-linecap:round"/>')
    parts.append(f'<text x="{(L + x(end)) / 2:.1f}" y="110" text-anchor="middle" style="font-size:11px;font-family:var(--ea-font-ui);fill:var(--ea-ink-3)">'
                 f'{html.escape(w["nothing"])}</text>')
    parts.append(f'<rect x="{x(end) - 5:.1f}" y="104" width="10" height="20" rx="2" style="fill:var(--ea-pred-text)"/>')
    parts.append(f'<text x="{x(end):.1f}" y="140" text-anchor="end" style="font-size:11px;font-family:var(--ea-font-ui);fill:var(--ea-ink-2)">'
                 f'{html.escape(w["end"].format(s=_n(end, lang)))}</text>')
    step = next(k for k in (1, 2, 5, 10, 20, 60) if end / k <= 12)
    for s in range(0, int(end) + 1, step):
        parts.append(f'<line x1="{x(s):.1f}" x2="{x(s):.1f}" y1="150" y2="155" style="stroke:var(--ea-ink-3)"/>')
        parts.append(f'<text x="{x(s):.1f}" y="168" text-anchor="middle" style="font-size:11px;font-family:var(--ea-font-ui);fill:var(--ea-ink-2)">{s}</text>')
    parts.append(f'<line x1="{L}" x2="{x(end):.1f}" y1="150" y2="150" style="stroke:var(--ea-ink-3)"/>')
    parts.append(f'<text x="{(L + x(end)) / 2:.1f}" y="186" text-anchor="middle" style="font-size:11px;font-family:var(--ea-font-ui);fill:var(--ea-ink-2)">{html.escape(w["axis"])}</text>')
    svg = (f'<svg viewBox="0 0 {W} 194" role="img" style="width:100%;max-width:46rem;height:auto"><title>{html.escape(w["title"])}</title>'
           f'{"".join(parts)}</svg>')
    return yours.figure(w["title"], svg, alt=w["title"])


_LINE_NOTES = {
    "en": {"role": "the first event: the role, no text yet", "content": "a piece of the answer", "finish": "the last piece: why it stopped",
           "usage": "the tokens, when you ask for them", "done": "the end of the stream"},
    "nl": {"role": "het eerste event: de rol, nog geen tekst", "content": "een stukje van het antwoord", "finish": "het laatste stukje: waarom het stopte",
           "usage": "de tokens, als je erom vraagt", "done": "het einde van de stream"},
}


def _note(line: str, lang: str) -> str:
    n = _LINE_NOTES[lang]
    if line == "data: [DONE]":
        return n["done"]
    chunk = json.loads(line[len("data: "):])
    if not chunk["choices"]:
        return n["usage"]
    c = chunk["choices"][0]
    if c.get("finish_reason"):
        return n["finish"]
    return n["role"] if "role" in c["delta"] else n["content"]


def lines_picture(lang: str) -> str:
    """The first and last lines of seed 0's stream as the course's server sent them, each with what it is."""
    lines = course()[0]["lines"]
    title = {"en": "The start and the end of the stream", "nl": "Het begin en het einde van de stream"}[lang]
    rows = []
    for i, line in enumerate(lines):
        if i == 6:
            rows.append('<div style="color:var(--ea-ink-3);font-family:var(--ea-font-code);font-size:12px">⋮</div>')
        piece = "content" in line and '"role"' not in line and '"finish_reason": null' in line and '"content": ""' not in line
        colour = "var(--ea-pred-text)" if piece else "var(--ea-ink)"
        rows.append(f'<div style="padding:0.1rem 0;border-top:1px solid var(--ea-line)"><div style="font-family:var(--ea-font-code);font-size:11.5px;'
                    f'overflow-wrap:anywhere;color:{colour}">{html.escape(line)}</div><div style="font-family:var(--ea-font-ui);font-size:11px;'
                    f'color:var(--ea-ink-2)">↑ {html.escape(_note(line, lang))}</div></div>')
    return yours.figure(title, "".join(rows), alt=title)


# --- exercise 1: one line of the stream -------------------------------------------------------------------------------

class Line(Code):
    """A line of a stream, written in a check's message the way Python writes a string: in single quotes, so the
    double quotes of its JSON need no backslashes."""

    def __str__(self):
        return str.__repr__(self)


def _line(delta: dict, finish=None) -> str:
    """A line of a stream in short: a real one also has the id of the answer, the model and the time."""
    return Line("data: " + json.dumps({"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}, ensure_ascii=False))


_START_PIECE = '''import json


def piece(line):
    chunk = json.loads(line)
'''
SOLUTION_PIECE = '''import json


def piece(line):
    if line[:6] != "data: " or line.strip() == "data: [DONE]":
        return ""
    chunk = json.loads(line[6:])
    delta = chunk["choices"][0]["delta"]
    if "content" in delta:
        return delta["content"]
    return ""
'''
EXAMPLE = _line({"content": " keeper"})
# the lines as they come out of a response: each ends in "\n", and an empty line follows every event
PIECE_CASES = [(EXAMPLE,)] + [(Line(line),) for line in (
    "\n", "data: [DONE]\n", _line({"role": "assistant", "content": ""}) + "\n", _line({}, "stop") + "\n", _line({"content": "Every night,"}) + "\n",
    _line({"content": "\n\n"}) + "\n", _line({"content": "é"}) + "\n", "")]


def piece_picture(seen) -> str:
    """Each line of a short stream with what the learner's piece makes of it."""
    run = seen.mine
    fn = run.names.get("piece") if run is not None else None
    if not callable(fn):
        return ""
    w = {"en": ("What your piece makes of each line", "gives", "stops with"), "nl": ("Wat je piece van elke regel maakt", "geeft", "stopt met")}[seen.lang]
    rows = []
    for (line,) in PIECE_CASES[:5]:
        try:
            got = repr(fn(line))
            colour = "var(--ea-pred-text)" if got not in ("''", "None") else "var(--ea-ink-2)"
            said = f'{w[1]} <span style="color:{colour}">{html.escape(got)}</span>'
        except Exception as e:   # the learner's code
            said = f'{w[2]} <span style="color:var(--ea-loss-text)">{html.escape(type(e).__name__)}</span>'
        rows.append(f'<div style="padding:0.15rem 0;border-top:1px solid var(--ea-line)"><div style="font-family:var(--ea-font-code);font-size:11.5px;'
                    f'color:var(--ea-ink);overflow-wrap:anywhere">{html.escape(str.__repr__(line))}</div>'
                    f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)">→ {said}</div></div>')
    return yours.figure(w[0], "".join(rows), alt=w[0])


def _not_json(e) -> bool:
    return isinstance(e, json.JSONDecodeError)


def _done(e) -> bool:
    return isinstance(e, json.JSONDecodeError) and e.doc.strip() == "[DONE]"


def _no_content(e) -> bool:
    return isinstance(e, KeyError) and e.args[:1] == ("content",)


# --- exercise 2: the whole stream ----------------------------------------------------------------------------------------

class NoPiece(Exception):
    """The learner's piece is not right yet."""


def no_piece(line):
    raise NoPiece()


def learner_piece(text):
    """The learner's piece from exercise 1, when it is right; else None."""
    from everything_ai.check import MISSING
    if text is MISSING or text is None or EXERCISES["piece"].check(text).state != "correct":
        return None
    return run_code(text).names.get("piece")


def course_piece():
    return run_code(SOLUTION_PIECE).names["piece"]


def stream_inputs(piece_text) -> dict:
    """What exercise 2 runs with: the learner's piece from exercise 1, or one that says to finish it first."""
    return {"piece": learner_piece(piece_text) or no_piece}


_START_STREAM = '''import json
import urllib.request


def stream(url, question):
    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": question}], "max_tokens": 200}
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    pieces = []
    with urllib.request.urlopen(request) as response:
        ...
    return pieces
'''
SOLUTION_STREAM = '''import json
import urllib.request


def stream(url, question):
    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": question}], "max_tokens": 200,
            "stream": True}
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    pieces = []
    with urllib.request.urlopen(request) as response:
        for line in response:
            text = piece(line.decode())
            if text:
                pieces.append(text)
    return pieces
'''
ANSWER = "The keeper climbs the stairs every night."
WANT = ["The", " keeper", " climbs", " the", " stairs", " every", " night."]


def _judge_stream(got, running):
    sent = [e["body"] for e in running.log if e["method"] == "POST" and isinstance(e["body"], dict)]
    if not sent:
        return "nothing-sent"
    if not sent[-1].get("stream"):
        return "no-stream"
    if got == WANT:
        return None
    if got == "".join(WANT):
        return "joined"
    if isinstance(got, list) and "" in got and [p for p in got if p] == WANT:
        return "empties"
    if got == []:
        return "none-found"
    return "other"


ASKED = "Write one sentence about a lighthouse keeper."
STREAM_CASES = [Case(lambda: serve.StandIn({ASKED: ANSWER}), (URL, ASKED), _judge_stream)]


def stream_picture(seen) -> str:
    words = {"en": {"title": "What your stream sent, and what came back", "request": "Your code sent: {method} {path}",
                    "headers": "Headers:", "answer": "The server sent back: status {status}",
                    "stream": "a stream, with {n} pieces of the answer", "none": "Your function sent nothing to the server yet."},
             "nl": {"title": "Wat je stream stuurde, en wat er terugkwam", "request": "Je code stuurde: {method} {path}",
                    "headers": "Headers:", "answer": "De server stuurde terug: status {status}",
                    "stream": "een stream, met {n} stukjes van het antwoord", "none": "Je functie stuurde nog niets naar de server."}}[seen.lang]
    return apicheck.exchange(seen, "stream", STREAM_CASES[0], words)


_COMMON = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
                "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."},
}

EXERCISES = {
    "piece": FunctionBox(
        name="piece",
        picture=piece_picture,
        function="piece",
        start=_START_PIECE,
        solution=SOLUTION_PIECE,
        visible=[(EXAMPLE,)],
        hidden=PIECE_CASES[1:],
        mistakes=[
            Mistake("done", error=_done,
                    en="With `{call}`, `json.loads` gets `[DONE]`, which is not JSON. A line from a stream still ends in `\\n`, and `line.strip()` takes it off: compare `line.strip()` with `\"data: [DONE]\"`, and give back `\"\"` for that line before `json.loads`.",
                    nl="Met `{call}` krijgt `json.loads` `[DONE]`, en dat is geen JSON. Een regel uit een stream eindigt nog op `\\n`, en `line.strip()` haalt die eraf: vergelijk `line.strip()` met `\"data: [DONE]\"`, en geef voor die regel `\"\"` terug vóór `json.loads`."),
            Mistake("not-json", error=_not_json,
                    en="With `{call}`, `json.loads` gets text that is not JSON. Only what comes after `data: ` is JSON; an empty line and `data: [DONE]` have none, so give back `\"\"` for those before `json.loads`.",
                    nl="Met `{call}` krijgt `json.loads` tekst die geen JSON is. Alleen wat na `data: ` komt is JSON; een lege regel en `data: [DONE]` hebben die niet, dus geef daarvoor `\"\"` terug vóór `json.loads`."),
            Mistake("no-content", error=_no_content,
                    en="With `{call}`, the delta has no `\"content\"`: the last event of a stream only says why it stopped. Look first whether `\"content\"` is in the delta, and give back `\"\"` when it is not.",
                    nl="Met `{call}` heeft de delta geen `\"content\"`: het laatste event van een stream zegt alleen waarom hij stopte. Kijk eerst of `\"content\"` in de delta zit, en geef `\"\"` terug als dat niet zo is."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "`piece` reads the line as JSON, and gives nothing back yet. Mind the `data: ` in front: only the part after it is JSON.",
                      "nl": "`piece` leest de regel als JSON, en geeft nog niets terug. Let op de `data: ` ervoor: alleen het deel erna is JSON."},
            "correct": {"en": "Right: your `piece` takes the text out of every line with a piece, and gives `\"\"` for every other line, the empty ones and `data: [DONE]` too: {n} lines, as they come out of a response.",
                        "nl": "Goed: je `piece` haalt de tekst uit elke regel met een stukje, en geeft `\"\"` voor elke andere regel, ook voor de lege en voor `data: [DONE]`: {n} regels, zoals ze uit een response komen."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. The text is in `[\"choices\"][0][\"delta\"][\"content\"]`.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn. De tekst staat in `[\"choices\"][0][\"delta\"][\"content\"]`."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`.",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn."},
        },
    ),
    "stream": ApiBox(
        name="stream",
        picture=stream_picture,
        function="stream",
        start=_START_STREAM,
        solution=SOLUTION_STREAM,
        cases=STREAM_CASES,
        names={"piece": course_piece()},
        mistakes=[
            Mistake("no-piece", error=lambda e: isinstance(e, NoPiece),
                    en="This exercise uses your `piece` from exercise 1. Finish exercise 1 first; this check then runs again by itself.",
                    nl="Deze oefening gebruikt je `piece` uit oefening 1. Maak eerst oefening 1 af; deze check draait daarna vanzelf opnieuw."),
            Mistake("bytes", error=lambda e: isinstance(e, TypeError) and "bytes" in str(e),
                    en="With `{call}`, `piece` got bytes: every line of the response is bytes. Give it `line.decode()`.",
                    nl="Met `{call}` kreeg `piece` bytes: elke regel van het antwoord is bytes. Geef hem `line.decode()`."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "`stream` sends the request, and reads nothing yet. Ask for a stream in the body, then go through the lines of `response`, and keep the pieces `piece` finds.",
                      "nl": "`stream` verstuurt het request, en leest nog niets. Vraag in de body om een stream, ga dan door de regels van `response`, en bewaar de stukjes die `piece` vindt."},
            "correct": {"en": "Right: your `stream` asks for a stream, and collects every piece, in order, without the empty ones.",
                        "nl": "Goed: je `stream` vraagt om een stream, en verzamelt elk stukje, op volgorde, zonder de lege."},
            "nothing-sent": {"en": "With `{call}`, your function sent nothing to the server.",
                             "nl": "Met `{call}` stuurde je functie niets naar de server."},
            "no-stream": {"en": "With `{call}`, the body did not ask for a stream, so the server sent the whole answer at once, as one JSON. Add `\"stream\": True` to the body.",
                          "nl": "Met `{call}` vroeg de body niet om een stream, dus stuurde de server het hele antwoord in één keer, als één JSON. Voeg `\"stream\": True` toe aan de body."},
            "joined": {"en": "With `{call}`, your function gives the whole text, `{got}`. Give back the list of the pieces, each as it came in.",
                       "nl": "Met `{call}` geeft je functie de hele tekst, `{got}`. Geef de lijst van de stukjes terug, elk zoals het binnenkwam."},
            "empties": {"en": "With `{call}`, your list also holds empty strings, from the lines without a piece. Keep only the pieces with text.",
                        "nl": "Met `{call}` bevat je lijst ook lege strings, van de regels zonder stukje. Bewaar alleen de stukjes met tekst."},
            "none-found": {"en": "With `{call}`, the server sent a stream, but your list stayed empty. Does each line go through `piece`, as text and not as bytes, and do you keep what it gives?",
                           "nl": "Met `{call}` stuurde de server een stream, maar je lijst bleef leeg. Gaat elke regel door `piece`, als tekst en niet als bytes, en bewaar je wat hij geeft?"},
            "other": {"en": "With `{call}`, your function gives `{got}`; the stand-in sent the pieces of *The keeper climbs the stairs every night.*, one word at a time.",
                      "nl": "Met `{call}` geeft je functie `{got}`; de stand-in stuurde de stukjes van *The keeper climbs the stairs every night.*, één woord tegelijk."},
        },
    ),
}
REQUIRED = ["piece", "stream", "when"]

CASES = [
    ("piece", _START_PIECE, "empty", None),
    ("piece", SOLUTION_PIECE, "correct", None),
    ("piece", SOLUTION_PIECE.replace('    if line[:6] != "data: " or line.strip() == "data: [DONE]":\n        return ""\n', "").replace("line[6:]", "line"), "error", "not-json"),
    ("piece", SOLUTION_PIECE.replace(' or line.strip() == "data: [DONE]"', ""), "error", "done"),
    ("piece", SOLUTION_PIECE.replace('line.strip() == "data: [DONE]"', 'line == "data: [DONE]"'), "error", "done"),
    ("piece", SOLUTION_PIECE.replace('    if "content" in delta:\n        return delta["content"]\n    return ""\n', '    return delta["content"]\n'), "error", "no-content"),
    ("piece", SOLUTION_PIECE.replace('return delta["content"]', 'return delta'), "wrong", None),
    ("stream", _START_STREAM, "empty", None),
    ("stream", SOLUTION_STREAM, "correct", None),
    ("stream", SOLUTION_STREAM.replace(',\n            "stream": True}', "}"), "wrong", "no-stream"),
    ("stream", SOLUTION_STREAM.replace("            if text:\n                pieces.append(text)", "            pieces.append(text)"), "wrong", "empties"),
    ("stream", SOLUTION_STREAM.replace("    return pieces", '    return "".join(pieces)'), "wrong", "joined"),
    ("stream", SOLUTION_STREAM.replace("piece(line.decode())", "piece(line)"), "wrong", "none-found"),
]


# --- the demo: your piece, live, against the course's server --------------------------------------------------------------

def downloaded() -> bool:
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    chat.download_or_stop(button, texts)


def demo(piece_fn, question: str, show, words: dict, lang: str) -> str:
    """A stream from the course's server with Qwen3, each line through the learner's piece: the text so far is shown
    by `show(html)` at most five times a second. At the end: the seconds to the first piece and to the end."""
    running = serve.model_server()
    body = {"model": serve.MODEL, "messages": [{"role": "user", "content": question}], "max_tokens": 200, "stream": True}
    request = urllib.request.Request(running.url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    start, first, text, shown = time.perf_counter(), None, "", 0.0
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            for line in response:
                problem = serve.stream_error(line.decode())
                if problem is not None:   # the server stopped the stream with an error
                    said = words["memory"] if problem == serve.MEMORY else words["failed"].format(why=problem)
                    return f"<p><em>{html.escape(said)}</em></p>"
                got = piece_fn(line.decode())
                if got:
                    first = first if first is not None else time.perf_counter() - start
                    text += got
                    if time.perf_counter() - shown > 0.2:
                        shown = time.perf_counter()
                        show(chat.answer_picture(text, words["writing"], words["title"]))
    except Exception as e:   # the learner's piece, against the real server; it may have run out of memory
        said = words["memory"] if serve.out_of_memory(e) else words["failed"].format(why=f"{type(e).__name__}: {e}")
        return f"<p><em>{html.escape(said)}</em></p>"
    end = time.perf_counter() - start
    note = words["done"].format(first=_n(first if first is not None else end, lang), seconds=_n(end, lang))
    return chat.answer_picture(text, note, words["title"])
