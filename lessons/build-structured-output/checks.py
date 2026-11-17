# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-structured-output: a function that checks a JSON answer against a small JSON schema (the keys it
must have and the type of each value), and one that asks again in JSON mode with the answer and what was wrong with
it, against a stand-in of the server; the course's recorded runs (data/course-dev-prompts-qwen.json): a bike's details
asked as JSON eight times without and eight times with JSON mode, where every answer in JSON mode was checked against
the schema and asked again while it did not fit; their pictures, and the demo."""
import copy
import html
import json

from everything_ai import chat, paths, serve, yours
from everything_ai.apicheck import URL, ApiBox, Case, call
from everything_ai.check import FunctionBox, Mistake, run_code

DOWNLOADS = {"Qwen3-0.6B": chat.FILES}   # what the demo downloads, for the line under the header

BIKE_TEXT = "Our new Cityline is a grey city bike with seven gears, for 649 euros."
ASK_BIKE = "Give the name, color and price of this bike as JSON, with the keys name, color and price.\n\n"
QUESTION = ASK_BIKE + BIKE_TEXT
SCHEMA = {"type": "object",
          "properties": {"name": {"type": "string"}, "color": {"type": "string"}, "price": {"type": "number"}},
          "required": ["name", "color", "price"]}
JSON_MODE = {"type": "json_object"}
TRIES = 3
RECORD = {
    "plain": {"messages": [{"role": "user", "content": QUESTION}], "max_tokens": 80, "seeds": 8},
    "json-mode": {"messages": [{"role": "user", "content": QUESTION}], "max_tokens": 80, "response_format": JSON_MODE,
                  "retry": TRIES, "seeds": 8},
}
DATA = paths.ROOT / "data" / "course-dev-prompts-qwen.json"

# --- exercise 1: problems ----------------------------------------------------------------------------------------------

START_PROBLEMS = '''import json

TYPES = {"string": str, "number": (int, float)}
WORDS = {"string": "a string", "number": "a number"}


def problems(text, schema):
    try:
        value = json.loads(text)
    except ValueError:
        return ["the answer is not JSON"]
    if not isinstance(value, dict):
        return ["the answer is not a JSON object"]
    found = []
    for key in schema["required"]:
        if key not in value:
            found.append(key + " is missing")
    return found
'''
SOLUTION_PROBLEMS = START_PROBLEMS.replace('''    return found
''', '''    for key, rule in schema["properties"].items():
        if key in value and not isinstance(value[key], TYPES[rule["type"]]):
            found.append(key + " must be " + WORDS[rule["type"]])
    return found
''')

BAD = '{"name": "Cityline", "color": "grey", "price": "649"}'
GOOD = '{"name": "Cityline", "color": "grey", "price": 649}'
BOOK_TEXT = "The novel Northwind has 312 pages."
BOOK_QUESTION = "Give the title and the number of pages of this book as JSON, with the keys title and pages.\n\n" + BOOK_TEXT
BOOK_SCHEMA = {"type": "object", "properties": {"title": {"type": "string"}, "pages": {"type": "number"}},
               "required": ["title", "pages"]}
BOOK_BAD = '{"title": "Northwind"}'
BOOK_GOOD = '{"title": "Northwind", "pages": 312}'
FENCED = '```json\n{"name": "Cityline", "color": "grey", "price": 649}\n```'
GEARS_SCHEMA = {"type": "object", "properties": {**SCHEMA["properties"], "gears": {"type": "number"}}, "required": SCHEMA["required"]}
PROBLEM_CASES = [
    (BAD, SCHEMA),
    (GOOD, SCHEMA),
    ('{"name": "Cityline", "price": 649}', SCHEMA),
    ('{"name": "Cityline", "color": 7, "price": 649.5}', SCHEMA),
    ('{"title": "Northwind", "pages": "312"}', BOOK_SCHEMA),
    (FENCED, SCHEMA),
    ('{"name": "Cityline", "color": "grey", "price": 649, "gears": 7}', SCHEMA),   # a key the schema does not name
    ('{"name": "Cityline", "color": "grey", "price": 649, "gears": "seven"}', GEARS_SCHEMA),   # a key that is not required
]


_COURSE: dict = {}   # the course's solution of exercise 1, run once (not with run_code: a lesson that builds on it loads this file)
exec(compile(SOLUTION_PROBLEMS, "<course problems>", "exec"), _COURSE)


def course_problems(text: str, schema: dict) -> list:
    """What the course's solution of exercise 1 finds in an answer."""
    return _COURSE["problems"](text, schema)


def fix_message(found: list) -> str:
    """The message that asks again, with the problems in it: as in the course's solution of exercise 2."""
    return "Fix these problems: " + "; ".join(found) + ". Answer with the whole JSON again."


def follow_up(text: str):
    """For the recorder: the message that asks again when the answer does not fit SCHEMA, else None."""
    found = course_problems(text, SCHEMA)
    return fix_message(found) if found else None


# --- the recorded runs ---------------------------------------------------------------------------------------------------

def course() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-structured-output"]


def answers(key: str) -> list:
    return course()[key]["answers"]


def is_json(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def inside(text: str):
    """The JSON in a code block with backticks, read: what an app would get if it took the backticks off."""
    lines = text.strip().split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return json.loads("\n".join(lines))


def findings() -> dict:
    """What the recorded runs show: without JSON mode, how many answers were in a code block and could be read by
    json.loads, and in how many of them the price was a text; in JSON mode, how many first answers json.loads could
    read and how many fit the schema, what was wrong with the others, how many tries each run needed, and the worked
    example (the first run)."""
    plain, mode = answers("plain"), answers("json-mode")
    first = [a["tries"][0]["text"] for a in mode]
    out = {
        "runs": len(plain),
        "plain_fenced": sum(a["text"].startswith("```") for a in plain),
        "plain_json": sum(is_json(a["text"]) for a in plain),
        "plain_finish": {a["finish"] for a in plain},
        "plain_text_price": sum(isinstance(inside(a["text"])["price"], str) for a in plain),
        "mode_json": sum(is_json(t) for t in first),
        "mode_fit": sum(course_problems(t, SCHEMA) == [] for t in first),
        "mode_problems": sorted({p for t in first for p in course_problems(t, SCHEMA)}),
        "tries": [len(a["tries"]) for a in mode],
        "final_fit": sum(course_problems(a["tries"][-1]["text"], SCHEMA) == [] for a in mode),
        "finish": {t["finish"] for a in mode for t in a["tries"]},
    }
    out["again"] = sum(n > 1 for n in out["tries"])
    worked = mode[0]["tries"]
    out["worked"] = {"first": worked[0]["text"], "found": course_problems(worked[0]["text"], SCHEMA),
                     "second": worked[1]["text"] if len(worked) > 1 else None,
                     "second_found": course_problems(worked[1]["text"], SCHEMA) if len(worked) > 1 else None,
                     "value": json.loads(worked[-1]["text"])}
    return out


# --- the pictures of the recorded runs -------------------------------------------------------------------------------

_CELL = "padding:0.18rem 0.5rem;border-bottom:1px solid var(--ea-line);vertical-align:top;font-size:13px"
_HEAD = f"{_CELL};color:var(--ea-ink-2);font-weight:500;text-align:left;font-family:var(--ea-font-ui)"


def _label(text: str) -> str:
    return f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2);margin:0.2rem 0">{html.escape(text)}</div>'


def _flat(text: str, mark: str | None = None) -> str:
    """Text the model wrote, on one line: each line break shown as ↵, the spaces at the start of a line left out, and
    `mark` (a part of one line) on the wash of a mistake."""
    parts = []
    for line in text.split("\n"):
        shown = html.escape(line.strip())
        if mark and html.escape(mark) in shown:
            shown = shown.replace(html.escape(mark), f'<span style="background:var(--ea-loss-wash)">{html.escape(mark)}</span>')
        parts.append(shown)
    joined = '<span style="color:var(--ea-ink-3)">↵</span>'.join(parts)
    return f'<span style="color:var(--ea-pred-text);font-family:var(--ea-font-code);font-size:13px">{joined}</span>'


def _mine(text: str) -> str:
    """Text your app wrote."""
    return f'<span style="color:var(--ea-ink);font-family:var(--ea-font-code);font-size:13px;font-weight:600">{html.escape(text)}</span>'


def _note(text: str, mistake: bool) -> str:
    color = "var(--ea-loss-text)" if mistake else "var(--ea-ink-2)"
    return f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:{color};font-weight:600">{html.escape(text)}</div>'


def _price_mark(text: str) -> str | None:
    """The price as a text in an answer, such as `"price": "649"`, to mark; None when the price is a number."""
    try:
        value = json.loads(text)
    except ValueError:
        try:
            value = inside(text)
        except ValueError:
            return None
    price = value.get("price") if isinstance(value, dict) else None
    return f'"price": "{price}"' if isinstance(price, str) else None


def _table(heads: list, rows: list) -> str:
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%"><thead><tr>'
            + "".join(f'<th style="{_HEAD}">{h}</th>' for h in heads) + f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


def _count_row(span: int, text: str) -> str:
    return (f'<tr><td style="{_CELL};border-bottom:none"></td><td colspan="{span}" style="{_CELL};border-bottom:none;'
            f'font-family:var(--ea-font-ui);color:var(--ea-ink);font-weight:600">{html.escape(text)}</td></tr>')


_PLAIN_WORDS = {
    "en": {"title": "The bike's details as JSON, eight times without JSON mode", "question": "The question: " + ASK_BIKE.strip() + " (and the text about the bike)",
           "seed": "run", "answer": "the answer", "not": "json.loads: not JSON", "count": "json.loads could read {n} of {k}"},
    "nl": {"title": "De gegevens van de fiets als JSON, acht keer zonder JSON mode", "question": "De vraag: " + ASK_BIKE.strip() + " (en de tekst over de fiets)",
           "seed": "run", "answer": "het antwoord", "not": "json.loads: geen JSON", "count": "json.loads kon er {n} van de {k} lezen"},
}


def plain_picture(lang: str) -> str:
    """The eight answers without JSON mode, each on one line (a price that is a text marked), and what json.loads made of it."""
    w = _PLAIN_WORDS[lang]
    rows = []
    for i, a in enumerate(answers("plain")):
        ok = is_json(a["text"])
        rows.append(f'<tr><td style="{_CELL};color:var(--ea-ink-2);font-family:var(--ea-font-ui)">{i + 1}</td>'
                    f'<td style="{_CELL}">{_flat(a["text"], _price_mark(a["text"]))}{"" if ok else _note(w["not"], True)}</td></tr>')
    plain = answers("plain")
    rows.append(_count_row(1, w["count"].format(n=sum(is_json(a["text"]) for a in plain), k=len(plain))))
    return yours.figure(w["title"], _label(w["question"]) + _table([w["seed"], w["answer"]], rows), alt=w["title"])


_MODE_WORDS = {
    "en": {"title": "The same question, eight times with JSON mode", "seed": "run", "answer": "the answer",
           "number": "json.loads reads it; price is the number {v}", "text": "json.loads reads it, but price is the text \"{v}\"",
           "count": "json.loads could read {n} of {k}; the price was a number in {m}"},
    "nl": {"title": "Dezelfde vraag, acht keer met JSON mode", "seed": "run", "answer": "het antwoord",
           "number": "json.loads leest het; price is het getal {v}", "text": "json.loads leest het, maar price is de tekst \"{v}\"",
           "count": "json.loads kon er {n} van de {k} lezen; de prijs was een getal in {m}"},
}


def mode_picture(lang: str) -> str:
    """The first answer of each run in JSON mode, each on one line (a price that is a text marked), and what json.loads made of it."""
    w = _MODE_WORDS[lang]
    rows, first = [], [a["tries"][0]["text"] for a in answers("json-mode")]
    for i, text in enumerate(first):
        price = json.loads(text).get("price")
        note = _note(w["text"].format(v=price), True) if isinstance(price, str) else _note(w["number"].format(v=price), False)
        rows.append(f'<tr><td style="{_CELL};color:var(--ea-ink-2);font-family:var(--ea-font-ui)">{i + 1}</td>'
                    f'<td style="{_CELL}">{_flat(text, _price_mark(text))}{note}</td></tr>')
    numbers = sum(not isinstance(json.loads(t).get("price"), str) for t in first)
    rows.append(_count_row(1, w["count"].format(n=sum(is_json(t) for t in first), k=len(first), m=numbers)))
    return yours.figure(w["title"], _table([w["seed"], w["answer"]], rows), alt=w["title"])


_AGAIN_WORDS = {
    "en": {"title": "Asked again with what was wrong: the first run, and all eight", "user": "user", "assistant": "assistant",
           "question": "the question about the bike", "fits": "fits", "seed": "run", "try": "try {n}",
           "count": "fits the schema in the end: {n} of {k}"},
    "nl": {"title": "Opnieuw gevraagd met wat er mis was: de eerste run, en alle acht", "user": "user", "assistant": "assistant",
           "question": "de vraag over de fiets", "fits": "past", "seed": "run", "try": "poging {n}",
           "count": "past uiteindelijk bij het schema: {n} van de {k}"},
}


def again_picture(lang: str) -> str:
    """The conversation of the first run in JSON mode: the question, the first answer, the message with the problems
    (bold: your app wrote it), and the second answer; then, for all eight runs, what the check found in each try."""
    w = _AGAIN_WORDS[lang]
    runs = answers("json-mode")
    tries = runs[0]["tries"]

    def row(role: str, inner: str) -> str:
        return (f'<div style="border-left:3px solid var(--ea-line);padding:0.1rem 0.5rem;margin:0.15rem 0">'
                f'<span style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2);display:inline-block;min-width:5.5rem">{role}</span>{inner}</div>')
    talk = [row(w["user"], f'<span style="font-family:var(--ea-font-ui);font-size:13px;color:var(--ea-ink-2)">{html.escape(w["question"])}</span>')]
    for i, t in enumerate(tries):
        talk.append(row(w["assistant"], _flat(t["text"], _price_mark(t["text"]))))
        found = course_problems(t["text"], SCHEMA)
        if found and i + 1 < len(tries):
            talk.append(row(w["user"], _mine(fix_message(found))))
    most = max(len(a["tries"]) for a in runs)
    rows = []
    for i, a in enumerate(runs):
        cells = []
        for n in range(most):
            if n < len(a["tries"]):
                found = course_problems(a["tries"][n]["text"], SCHEMA)
                cells.append(f'<td style="{_CELL}">{_note("; ".join(found) if found else w["fits"], bool(found))}</td>')
            else:
                cells.append(f'<td style="{_CELL};color:var(--ea-ink-3)">–</td>')
        rows.append(f'<tr><td style="{_CELL};color:var(--ea-ink-2);font-family:var(--ea-font-ui)">{i + 1}</td>{"".join(cells)}</tr>')
    fit = sum(course_problems(a["tries"][-1]["text"], SCHEMA) == [] for a in runs)
    rows.append(_count_row(most, w["count"].format(n=fit, k=len(runs))))
    table = _table([w["seed"]] + [w["try"].format(n=n + 1) for n in range(most)], rows)
    return yours.figure(w["title"], "".join(talk) + '<div style="height:0.6rem"></div>' + table, alt=w["title"])


# --- the picture of exercise 1 ----------------------------------------------------------------------------------------------

_PROBLEMS_WORDS = {
    "en": {"title": "What your problems found in three answers", "mine": "your problems:", "want": "should find:", "error": "stops with {e}"},
    "nl": {"title": "Wat je problems vond in drie antwoorden", "mine": "je problems:", "want": "zou moeten vinden:", "error": "stopt met {e}"},
}


def problems_picture(seen) -> str:
    """Three answers checked against the bike's schema: what the learner's problems found in each, and, where that
    differs, what it should find, marked."""
    w = _PROBLEMS_WORDS[seen.lang]
    mine = seen.mine
    fn = mine.names.get("problems") if mine is not None else None
    if not callable(fn):
        return ""
    blocks = []
    for text, schema in PROBLEM_CASES[:3]:
        want = seen.remember(("problems", text), lambda text=text, schema=schema: course_problems(text, schema))
        try:
            got = fn(text, copy.deepcopy(schema))
        except Exception as e:   # the learner's code; the check says what went wrong
            got, shown = None, w["error"].format(e=type(e).__name__)
        else:
            shown = repr(got)
        off = got != want
        edge = "var(--ea-loss)" if off else "var(--ea-line)"
        blocks.append(f'<div style="border-left:3px solid {edge};padding:0.2rem 0.5rem;margin:0.3rem 0">'
                      f'<div>{_flat(text)}</div>'
                      f'<div style="font-family:var(--ea-font-code);font-size:13px;color:var(--ea-ink)">'
                      f'<span style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)">{w["mine"]}</span> {html.escape(shown)}</div>'
                      + (f'<div style="font-family:var(--ea-font-code);font-size:13px;color:var(--ea-ink)">'
                         f'<span style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)">{w["want"]}</span> {html.escape(repr(want))}</div>' if off else "")
                      + "</div>")
    return yours.figure(w["title"], "".join(blocks), alt=w["title"])


# --- exercise 2: ask_json, which asks again ------------------------------------------------------------------------------

_ASK = '''import json
import urllib.request


def ask(url, messages):
    body = {"model": "qwen3-0.6b", "messages": messages, "max_tokens": 80,
            "response_format": {"type": "json_object"}}
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        answer = json.loads(response.read())
    return answer["choices"][0]["message"]["content"]


'''
START_ASK = _ASK + '''def ask_json(url, messages, schema, tries):
    for attempt in range(tries):
        text = ask(url, messages)
        found = problems(text, schema)
        if found == []:
            return json.loads(text)
    return None
'''
SOLUTION_ASK = START_ASK.replace('''            return json.loads(text)
''', '''            return json.loads(text)
        fix = "Fix these problems: " + "; ".join(found) + ". Answer with the whole JSON again."
        messages = messages + [{"role": "assistant", "content": text},
                               {"role": "user", "content": fix}]
''')
BIKE_ASKED = [{"role": "user", "content": QUESTION}]
BOOK_ASKED = [{"role": "system", "content": "Answer with JSON only."}, {"role": "user", "content": BOOK_QUESTION}]
_BIKE_ARG, _BOOK_ARG = [], []   # the lists the check's calls get as messages: filled anew before each call


def _braced(text: str) -> str:
    """A stand-in's default answer is filled in with {question}: its own braces doubled."""
    return text.replace("{", "{{").replace("}", "}}")


def _fresh(arg: list, asked: list, backend):
    """The stand-in for a case, after the list the call gets is put back the way the case asks it."""
    def make():
        arg[:] = copy.deepcopy(asked)
        return backend()
    return make


def _sent(log: list) -> list:
    """The requests in a server's log that sent JSON."""
    return [e for e in log if e["method"] == "POST" and isinstance(e["body"], dict)]


def _content(entry) -> str:
    try:
        return entry["answer"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""


def _judge_ask(arg: list, asked: list, schema: dict, posts: int, want):
    def judge(got, running):
        sent = _sent(running.log)
        if any(e["body"].get("response_format") != JSON_MODE for e in sent):
            return "json-mode"
        if len(sent) >= 2:
            first, second = sent[0]["body"].get("messages"), sent[1]["body"].get("messages")
            said = _content(sent[0])
            if second == first:
                return "no-fix"
            if not isinstance(second, list) or not isinstance(first, list) or second[:len(first)] != first:
                return "lost"
            added = [m for m in second[len(first):] if isinstance(m, dict)]
            answer = {"role": "assistant", "content": said}
            if answer not in added:
                return "no-answer"
            fixes = [i for i, m in enumerate(added) if m.get("role") == "user"]
            found = course_problems(said, schema)
            if not fixes or not all(p in str(added[fixes[-1]].get("content")) for p in found):
                return "no-problems"
            if fixes[-1] < added.index(answer):
                return "order"
        if len(sent) != posts:
            return "tries"
        if want is None:
            if got is not None:
                return "not-none"
        elif got != want:
            return "return"
        if arg != asked:
            return "almost"
        return None
    return judge


ASK_CASES = [
    Case(_fresh(_BIKE_ARG, BIKE_ASKED, lambda: serve.StandIn({QUESTION: BAD}, default=_braced(GOOD))),
         (URL, _BIKE_ARG, SCHEMA, TRIES), _judge_ask(_BIKE_ARG, BIKE_ASKED, SCHEMA, 2, json.loads(GOOD)),
         shown_args=("url", {"en": "[the question about the bike]", "nl": "[de vraag over de fiets]"}, "SCHEMA", TRIES), share=True),
    Case(_fresh(_BOOK_ARG, BOOK_ASKED, lambda: serve.StandIn(default=_braced(BOOK_BAD))),
         (URL, _BOOK_ARG, BOOK_SCHEMA, 2), _judge_ask(_BOOK_ARG, BOOK_ASKED, BOOK_SCHEMA, 2, None),
         shown_args=("url", {"en": "[the question about the book]", "nl": "[de vraag over het boek]"}, "BOOK_SCHEMA", 2), share=True),
    Case(_fresh(_BOOK_ARG, BOOK_ASKED, lambda: serve.StandIn(default=_braced(BOOK_GOOD))),
         (URL, _BOOK_ARG, BOOK_SCHEMA, TRIES), _judge_ask(_BOOK_ARG, BOOK_ASKED, BOOK_SCHEMA, 1, json.loads(BOOK_GOOD)),
         shown_args=("url", {"en": "[the question about the book]", "nl": "[de vraag over het boek]"}, "BOOK_SCHEMA", TRIES), share=True),
]


def _messages_html(messages) -> str:
    """The messages of a request, one row each: the role, then the content (what the model wrote in its colour, any
    other text only up to its first line break)."""
    rows = []
    for m in messages if isinstance(messages, list) else []:
        role = html.escape(str(m.get("role"))) if isinstance(m, dict) else ""
        content = str(m.get("content")) if isinstance(m, dict) else repr(m)
        if role == "assistant":
            inner = _flat(content)
        else:
            first = content.split("\n")[0] + (" …" if "\n" in content else "")
            inner = f'<span style="font-family:var(--ea-font-code);font-size:13px;color:var(--ea-ink)">{html.escape(first)}</span>'
        rows.append(f'<div style="border-left:3px solid var(--ea-line);padding:0.1rem 0.5rem;margin:0.15rem 0">'
                    f'<span style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2);display:inline-block;min-width:5.5rem">{role}</span>{inner}</div>')
    return "".join(rows)


_TRIES_WORDS = {
    "en": {"title": "What your ask_json sent, and what came back", "request": "Request {n}, with {k} messages:",
           "came": "came back:", "found": "problems found: {p}", "gave": "Your function gave back: {v}",
           "error": "Your function stopped with {e}.", "none": "Your function sent nothing to the server yet."},
    "nl": {"title": "Wat je ask_json stuurde, en wat er terugkwam", "request": "Request {n}, met {k} berichten:",
           "came": "kwam terug:", "found": "gevonden problemen: {p}", "gave": "Je functie gaf terug: {v}",
           "error": "Je functie stopte met {e}.", "none": "Je functie stuurde nog niets naar de server."},
}


def tries_picture(seen) -> str:
    """The requests the learner's ask_json sent for the bike's question against the stand-in, whose first answer has
    the price as a text: the messages of each, what came back and what the course's check found in it, and what the
    function gave back."""
    w = _TRIES_WORDS[seen.lang]
    run = seen.run
    fn = run.names.get("ask_json") if run is not None and run.error is None else None
    if not callable(fn):
        return ""
    got, error, running = call(fn, ASK_CASES[0])
    sent = _sent(running.log)
    if not sent:
        return yours.figure(w["title"], _label(w["none"]), alt=w["title"])
    body = []
    for n, e in enumerate(sent):
        messages = e["body"].get("messages")
        body.append(_label(w["request"].format(n=n + 1, k=len(messages) if isinstance(messages, list) else 0)))
        body.append(_messages_html(messages))
        said = _content(e)
        found = course_problems(said, SCHEMA)
        body.append(f'<div style="margin:0.1rem 0 0.5rem 0.6rem">{_label(w["came"])}{_flat(said, _price_mark(said))}'
                    f'{_note(w["found"].format(p=repr(found)), bool(found))}</div>')
    body.append(_label(w["error"].format(e=type(error).__name__) if error is not None else w["gave"].format(v=repr(got))))
    return yours.figure(w["title"], "".join(body), alt=w["title"])


def _not_bytes(e) -> bool:
    return isinstance(e, TypeError) and "bytes" in str(e)


_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}

EXERCISES = {
    "problems": FunctionBox(
        name="problems",
        picture=problems_picture,
        function="problems",
        start=START_PROBLEMS,
        solution=SOLUTION_PROBLEMS,
        visible=PROBLEM_CASES[:1],
        hidden=PROBLEM_CASES[1:],
        save_as=("app", "schema"),   # Project: your assistant, prompted well builds on it
        mistakes=[
            Mistake("key-error", error=lambda e: isinstance(e, KeyError),
                    en="With `{call}`, your function stops with a KeyError, Python's error for a key that is not in a dictionary. Which key does your loop look up: is it in `value`, in `TYPES` or in `WORDS`? A key of the schema that is not in the answer is already in `found`; check whether it is in `value` before you look at its value.",
                    nl="Met `{call}` stopt je functie met een KeyError, de fout van Python voor een key die niet in een dictionary staat. Welke key zoekt je loop op: staat die in `value`, in `TYPES` of in `WORDS`? Een key van het schema die niet in het antwoord staat, staat al in `found`; kijk of hij in `value` staat voordat je naar zijn waarde kijkt."),
            Mistake("type-word", wrong=SOLUTION_PROBLEMS.replace('" must be " + WORDS[rule["type"]]', '" must be " + rule["type"]'),
                    en="`{call}` gives `{got}`: the type as the schema writes it. Write it in words, with `WORDS[rule[\"type\"]]`: *price must be a number*, so the model understands the problem when it hears it.",
                    nl="`{call}` geeft `{got}`: het type zoals het schema het schrijft. Schrijf het in woorden, met `WORDS[rule[\"type\"]]`: *price must be a number*, zodat het model het probleem begrijpt als het dat hoort."),
            Mistake("inverted", wrong=SOLUTION_PROBLEMS.replace("if key in value and not isinstance(", "if key in value and isinstance("),
                    en="`{call}` gives `{got}`: your function names the values that do have the right type. A value is a problem when `isinstance` gives `False`: put `not` in front of it.",
                    nl="`{call}` geeft `{got}`: je functie noemt de waarden die wél het goede type hebben. Een waarde is een probleem als `isinstance` `False` geeft: zet er `not` voor."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`problems` still checks only whether the keys are there. Add a loop over `schema[\"properties\"]` that checks the type of each value that is there.",
                      "nl": "`problems` kijkt nog alleen of de keys er zijn. Voeg een loop over `schema[\"properties\"]` toe die het type controleert van elke waarde die er is."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. For each key in `schema[\"properties\"]` that is in the answer, check its value with `isinstance` and the type in `TYPES`, and name it in `found` when it does not fit.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Controleer voor elke key in `schema[\"properties\"]` die in het antwoord staat de waarde met `isinstance` en het type in `TYPES`, en noem hem in `found` als hij niet past."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for an answer without one of the keys, for a price with a decimal point, and for another schema?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Werkt je functie ook voor een antwoord zonder een van de keys, voor een prijs met een decimale punt, en voor een ander schema?"},
            "no_function": {"en": "Your box does not make a function `problems`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `problems`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `problems` is right in all {n} cases: what is not JSON, what is missing, and each value of the wrong type, in words the model can act on. The check kept it as a file, `{saved}`.",
                        "nl": "Je `problems` klopt in alle {n} gevallen: wat geen JSON is, wat ontbreekt, en elke waarde van het verkeerde type, in woorden waar het model iets mee kan. De check heeft hem als bestand bewaard, `{saved}`."},
            "kept": {"en": "Your `problems` is right. You changed `{saved}` yourself, so the check left it as it is.",
                     "nl": "Je `problems` is goed. Je hebt `{saved}` zelf veranderd, dus de check liet het staan zoals het is."},
        },
    ),
    "ask_json": ApiBox(
        name="ask_json",
        picture=tries_picture,
        function="ask_json",
        start=START_ASK,
        solution=SOLUTION_ASK,
        cases=ASK_CASES,
        names={"problems": _COURSE["problems"]},
        mistakes=[
            Mistake("data-text", error=_not_bytes,
                    en="urllib sends bytes, not text. Keep `.encode()` after `json.dumps(body)`, as the box had it.",
                    nl="urllib verstuurt bytes, geen tekst. Laat `.encode()` achter `json.dumps(body)` staan, zoals het in het vak stond."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "After an answer that does not fit, `ask_json` asks the same question again, so the model never hears what was wrong. Before the next try, add the answer and a message with the problems to the conversation.",
                      "nl": "Na een antwoord dat niet past, stelt `ask_json` dezelfde vraag opnieuw, dus hoort het model nooit wat er mis was. Voeg vóór de volgende poging het antwoord en een bericht met de problemen aan het gesprek toe."},
            "correct": {"en": "Your `ask_json` gives the model its answer and the problems in it, and asks again: for all {n} calls of the check, as often as needed and no more.",
                        "nl": "Je `ask_json` geeft het model zijn antwoord en de problemen erin, en vraagt opnieuw: bij alle {n} aanroepen van de check, zo vaak als nodig en niet vaker."},
            "json-mode": {"en": "With `{call}`, a request went without `\"response_format\": {{\"type\": \"json_object\"}}`. Keep JSON mode on in `ask`, as the box had it.",
                          "nl": "Met `{call}` ging er een request zonder `\"response_format\": {{\"type\": \"json_object\"}}`. Laat JSON mode aan staan in `ask`, zoals het in het vak stond."},
            "no-fix": {"en": "With `{call}`, your function sent the same messages a second time, so the model heard nothing new. Before the next try, make `messages` the conversation with two messages added: the answer, and one with the problems.",
                       "nl": "Met `{call}` stuurde je functie dezelfde berichten een tweede keer, dus hoorde het model niets nieuws. Maak `messages` vóór de volgende poging het gesprek met twee berichten erbij: het antwoord, en een met de problemen."},
            "lost": {"en": "With `{call}`, the second request no longer starts with the conversation the function got. Keep all of `messages`, and add the two new messages after it: without the question, the model does not know what to fix.",
                     "nl": "Met `{call}` begint het tweede request niet meer met het gesprek dat de functie kreeg. Houd heel `messages`, en zet de twee nieuwe berichten erachter: zonder de vraag weet het model niet wat het moet herstellen."},
            "no-answer": {"en": "With `{call}`, the second request lacks the answer that did not fit. Add it as a message with the role `assistant` and `text` as its content, so the model sees what it wrote.",
                          "nl": "Met `{call}` mist het tweede request het antwoord dat niet paste. Voeg het toe als bericht met de rol `assistant` en `text` als content, zodat het model ziet wat het schreef."},
            "no-problems": {"en": "With `{call}`, the second request has no message of the user that names every problem `problems` found. Put them in it, joined with `\"; \".join(found)`.",
                            "nl": "Met `{call}` heeft het tweede request geen bericht van de gebruiker dat elk probleem noemt dat `problems` vond. Zet ze erin, samengevoegd met `\"; \".join(found)`."},
            "order": {"en": "With `{call}`, your message with the problems comes before the answer it is about. Keep the order of a conversation: first the assistant's answer, then your message.",
                      "nl": "Met `{call}` komt je bericht met de problemen vóór het antwoord waar het over gaat. Houd de volgorde van een gesprek aan: eerst het antwoord van de assistant, dan je bericht."},
            "tries": {"en": "With `{call}`, your function asked another number of times than it should: once more after each answer that does not fit, and at most `tries` times in all.",
                      "nl": "Met `{call}` vroeg je functie een ander aantal keren dan zou moeten: nog een keer na elk antwoord dat niet past, en hooguit `tries` keer in totaal."},
            "not-none": {"en": "With `{call}`, no answer fitted in the tries, and your function gives back `{got}`. After the last try, give back `None`, so the code that calls it can see that nothing fitted.",
                         "nl": "Met `{call}` paste geen enkel antwoord in de pogingen, en geeft je functie `{got}` terug. Geef na de laatste poging `None` terug, zodat de code die hem aanroept kan zien dat niets paste."},
            "return": {"en": "With `{call}`, your function gives back `{got}`. When an answer fits, give back what `json.loads` makes of it: a dictionary.",
                       "nl": "Met `{call}` geeft je functie `{got}` terug. Als een antwoord past, geef dan terug wat `json.loads` ervan maakt: een dictionary."},
            "almost": {"en": "Your function asks again the right way, but it also changed the list it got: the answer that did not fit and your message are now in the conversation of whoever called it. Make a new list with `+`, as in `messages = messages + [...]`, instead of `append` or `+=`.",
                       "nl": "Je functie vraagt op de goede manier opnieuw, maar hij veranderde ook de lijst die hij kreeg: het antwoord dat niet paste en je bericht staan nu in het gesprek van wie hem aanriep. Maak een nieuwe lijst met `+`, zoals `messages = messages + [...]`, in plaats van `append` of `+=`."},
        },
    ),
}

REQUIRED = ["problems", "ask_json", "guarantee"]

_FIX_LINE = '        fix = "Fix these problems: " + "; ".join(found) + ". Answer with the whole JSON again."\n'
_ADD = ('        messages = messages + [{"role": "assistant", "content": text},\n'
        '                               {"role": "user", "content": fix}]\n')

CASES = [
    ("problems", START_PROBLEMS, "empty", None),
    ("problems", "", "empty", None),
    ("problems", SOLUTION_PROBLEMS, "correct", None),
    ("problems", START_PROBLEMS.replace('''    return found
''', '''    for key in value:
        rule = schema["properties"].get(key)
        if rule is not None and not isinstance(value[key], TYPES[rule["type"]]):
            found.append(key + " must be " + WORDS[rule["type"]])
    return found
'''), "correct", None),
    ("problems", SOLUTION_PROBLEMS.replace("if key in value and not isinstance(", "if not isinstance("), "error", "key-error"),
    ("problems", SOLUTION_PROBLEMS.replace('" must be " + WORDS[rule["type"]]', '" must be " + rule["type"]'), "wrong", "type-word"),
    ("problems", SOLUTION_PROBLEMS.replace("if key in value and not isinstance(", "if key in value and isinstance("), "wrong", "inverted"),
    ("problems", SOLUTION_PROBLEMS.replace('TYPES = {"string": str, "number": (int, float)}', 'TYPES = {"string": str, "number": int}'), "almost", None),
    ("problems", SOLUTION_PROBLEMS.replace('for key, rule in schema["properties"].items():\n        if key in value and not isinstance(value[key], TYPES[rule["type"]]):',
                                           'for key in ["price"]:\n        if key in value and not isinstance(value[key], (int, float)):')
     .replace('" must be " + WORDS[rule["type"]]', '" must be a number"'), "almost", None),
    ("ask_json", START_ASK, "empty", None),
    ("ask_json", SOLUTION_ASK, "correct", None),
    ("ask_json", SOLUTION_ASK.replace(_FIX_LINE, '        fix = "These keys are wrong: " + ", ".join(found)\n'), "correct", None),
    ("ask_json", SOLUTION_ASK.replace(',\n            "response_format": {"type": "json_object"}}', "}"), "wrong", "json-mode"),
    ("ask_json", SOLUTION_ASK.replace("        messages = messages + [", "        more = messages + ["), "wrong", "no-fix"),
    ("ask_json", SOLUTION_ASK.replace("        messages = messages + [", "        messages = ["), "wrong", "lost"),
    ("ask_json", SOLUTION_ASK.replace(_ADD, '        messages = messages + [{"role": "user", "content": fix}]\n'), "wrong", "no-answer"),
    ("ask_json", SOLUTION_ASK.replace(_FIX_LINE, '        fix = "Please fix the JSON."\n'), "wrong", "no-problems"),
    ("ask_json", SOLUTION_ASK.replace(_ADD, '        messages = messages + [{"role": "user", "content": fix},\n'
                                            '                               {"role": "assistant", "content": text}]\n'), "wrong", "order"),
    ("ask_json", SOLUTION_ASK.replace("range(tries)", "range(tries + 1)"), "wrong", "tries"),
    ("ask_json", SOLUTION_ASK.replace("    return None\n", "    return json.loads(text)\n"), "wrong", "not-none"),
    ("ask_json", SOLUTION_ASK.replace("            return json.loads(text)\n", "            return text\n"), "wrong", "return"),
    ("ask_json", SOLUTION_ASK.replace(_ADD, '        messages.append({"role": "assistant", "content": text})\n'
                                            '        messages.append({"role": "user", "content": fix})\n'), "almost", "almost"),
    ("ask_json", SOLUTION_ASK.replace("json.dumps(body).encode()", "json.dumps(body)"), "error", "data-text"),
]


# --- the demo: your ask_json, against the course's own server ---------------------------------------------------------------

def downloaded() -> bool:
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    chat.download_or_stop(button, texts)


def learner_ask_json(text):
    """The learner's ask_json from exercise 2, when it is right; else None."""
    from everything_ai.check import MISSING
    ex = EXERCISES["ask_json"]
    if text is MISSING or text is None or ex.check(text).state != "correct":
        return None
    return run_code(text, dict(ex.names)).names.get("ask_json")


def demo(ask_json, text: str, words: dict, lang: str) -> str:
    """The bike question about the learner's text, sent with the learner's ask_json to the course's server with Qwen3
    behind it, in JSON mode, with the bike's schema and three tries: each answer and what the check found in it, and
    what ask_json gave back."""
    try:
        running = serve.model_server()
        before = len(running.log)
        got = ask_json(running.url, [{"role": "user", "content": ASK_BIKE + text}], copy.deepcopy(SCHEMA), TRIES)
    except Exception as e:   # the learner's code, against the real server; it may have run out of memory
        said = words["memory"] if serve.out_of_memory(e) else words["failed"].format(why=f"{type(e).__name__}: {e}")
        return f"<p><em>{html.escape(said)}</em></p>"
    body = []
    for n, e in enumerate(_sent(running.log[before:])):
        said = _content(e)
        found = course_problems(said, SCHEMA)
        body.append(_label(words["try"].format(n=n + 1)) + f'<div>{_flat(said, _price_mark(said))}</div>'
                    + _note(words["found"].format(p="; ".join(found)) if found else words["fits"], bool(found)))
    body.append(_label(words["gave"]) + f'<div style="font-family:var(--ea-font-code);font-size:13px;color:var(--ea-ink)">{html.escape(repr(got))}</div>')
    return yours.figure(words["title"], "".join(body), alt=words["title"])
