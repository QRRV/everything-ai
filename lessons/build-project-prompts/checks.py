# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-project-prompts: app/prompts.py, the assistant's system prompt and templates in a file of their own,
with messages_for, which makes the first messages of a task; app/answers.py, with ask_json, which asks through the
client of build-project-call, checks the answer with problems (app/schema.py, which build-structured-output saves) and
asks again with what was wrong, both against a stand-in of the server; the course's recorded runs
(data/course-dev-prompts-qwen.json): the event in three messages as JSON, with and without the rule about JSON in the
system prompt; the pictures, and the demo with the learner's own modules."""
import html
import json
import sys

from everything_ai import apicheck, chat, paths, projects, serve, yours
from everything_ai.apicheck import URL, ApiBox, Case
from everything_ai.check import FunctionBox, Mistake, run_code

DOWNLOADS = {"Qwen3-0.6B": chat.FILES}   # what the demo downloads, for the line under the header

# ask_json uses the learner's chat of build-project-call and problems of build-structured-output; where one is missing
# or does not pass, the course's version stands in (template 1.3). Their app/prompts.py of exercise 1 is not among
# them: it is theirs to change, so ask_json and the demo always use it as it is.
projects.builds_on(("app", "client", "build-project-call", "chat"), ("app", "schema", "build-structured-output", "problems"))

FOOTBALL = "Football training moves to Tuesday, 7 pm."
MESSAGES = {"lunch": "Can we have lunch on Thursday at 12? Kim",
            "football": FOOTBALL,
            "book-club": "Reminder: the book club meets on Sunday at 3 in the afternoon."}
RAIN = "Yesterday it rained all day, so we stayed in, baked bread and played cards."
PLAIN = "You are a friendly assistant. Answer briefly."   # the system prompt without its rule about JSON
TRIES = 3
SEEDS = 8

# --- exercise 1: app/prompts.py ------------------------------------------------------------------------------------------

START_PROMPTS = '''SYSTEM = ("You are a friendly assistant. Answer briefly. "
          "When a task asks for JSON, answer with the JSON only, without a code block.")

TEMPLATES = {
    "event": {
        "version": 1,
        "text": ("Give the event in this message as JSON with the keys what, day and hour. "
                 "The hour is a number from 0 to 23.\\n\\n{text}"),
        "schema": {"type": "object",
                   "properties": {"what": {"type": "string"}, "day": {"type": "string"},
                                  "hour": {"type": "number"}},
                   "required": ["what", "day", "hour"]},
    },
    "title": {
        "version": 1,
        "text": "Make a title of at most five words for this text, as JSON with the key title.\\n\\n{text}",
        "schema": {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]},
    },
}


def messages_for(name, text):
'''
SOLUTION_PROMPTS = START_PROMPTS + '''    template = TEMPLATES[name]
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": template["text"].format(text=text)}]
'''
_COURSE_PROMPTS: dict = {}
exec(compile(SOLUTION_PROMPTS, "<course prompts>", "exec"), _COURSE_PROMPTS)
SYSTEM, TEMPLATES = _COURSE_PROMPTS["SYSTEM"], _COURSE_PROMPTS["TEMPLATES"]
SCHEMA = TEMPLATES["event"]["schema"]


def course_messages(name: str, text: str) -> list:
    """The messages the course's messages_for makes."""
    return _COURSE_PROMPTS["messages_for"](name, text)


PROMPT_CASES = [("event", FOOTBALL), ("title", RAIN), ("event", "Dinner at {Luigi's} on Friday at 8.")]

# --- app/schema.py: problems, which build-structured-output saves ---------------------------------------------------------

_COURSE_SCHEMA: dict = {}
exec(compile(projects.course_code("app", "schema"), "<course schema>", "exec"), _COURSE_SCHEMA)


def course_problems(text: str, schema: dict) -> list:
    return _COURSE_SCHEMA["problems"](text, schema)


def fix_message(found: list) -> str:
    return "Fix these problems: " + "; ".join(found) + ". Answer with the whole JSON again."


def follow_up(text: str):
    """What the course's ask_json says after an answer about an event: the problems in it, or None when it fits."""
    found = course_problems(text, SCHEMA)
    return fix_message(found) if found else None


# --- the course's runs ---------------------------------------------------------------------------------------------------

RECORD = {}
for _key, _text in MESSAGES.items():
    _asked = course_messages("event", _text)
    RECORD["rule-" + _key] = {"messages": _asked, "max_tokens": 80, "retry": TRIES, "seeds": SEEDS}
    RECORD["plain-" + _key] = {"messages": [{"role": "system", "content": PLAIN}, _asked[1]], "max_tokens": 80,
                               "retry": TRIES, "seeds": SEEDS}
DATA = paths.ROOT / "data" / "course-dev-prompts-qwen.json"


def course() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-project-prompts"]


def answers(mode: str, key: str) -> list:
    """The recorded runs of one message, with ("rule") or without ("plain") the rule about JSON: each {"seed", "tries"}."""
    return course()[f"{mode}-{key}"]["answers"]


def fits(text: str) -> bool:
    return course_problems(text, SCHEMA) == []


def fenced(text: str) -> bool:
    return text.strip().startswith("```")


def findings() -> dict:
    """What the recorded runs show, per mode: how many answers fit at the first try, per message and in all; how many
    of those that did not were in a code block; how many fit in the end, and the most tries; and the hours the answers
    about the book club gave (3 in the afternoon is 15)."""
    out = {}
    for mode in ("plain", "rule"):
        runs = [a for key in MESSAGES for a in answers(mode, key)]
        firsts = [a["tries"][0]["text"] for a in runs]
        out[mode] = {
            "runs": len(runs),
            "first": {key: sum(fits(a["tries"][0]["text"]) for a in answers(mode, key)) for key in MESSAGES},
            "first_all": sum(fits(t) for t in firsts),
            "fenced": sum(fenced(t) for t in firsts if not fits(t)),
            "problems": sorted({p for t in firsts for p in course_problems(t, SCHEMA)}),
            "final": sum(fits(a["tries"][-1]["text"]) for a in runs),
            "again": sum(len(a["tries"]) > 1 for a in runs),
            "tries": max(len(a["tries"]) for a in runs),
            "hours": [json.loads(a["tries"][-1]["text"])["hour"] for a in answers(mode, "book-club") if fits(a["tries"][-1]["text"])],
        }
    first = answers("rule", "football")[0]["tries"]
    out["worked"] = {"text": first[0]["text"], "tries": len(first), "value": json.loads(first[0]["text"])}
    return out


# --- pictures ------------------------------------------------------------------------------------------------------------

def _mono(text: str, colour: str = "var(--ea-ink)") -> str:
    return (f'<span style="font-family:var(--ea-font-code);font-size:12px;color:{colour};white-space:pre-wrap;'
            f'overflow-wrap:anywhere">{html.escape(text)}</span>')


def _ui(text: str, colour: str = "var(--ea-ink-2)", size: int = 12, weight: int = 400) -> str:
    return (f'<span style="font-family:var(--ea-font-ui);font-size:{size}px;color:{colour};font-weight:{weight}">'
            f'{html.escape(text)}</span>')


_FILES_WORDS = {
    "en": {"title": "Your assistant's files after today", "today": "new today", "given": "from JSON you can rely on",
           "earlier": "from Project: your assistant answers", "uses": "uses",
           "client": "chat: sends a conversation, gives back the text",
           "prompts": "SYSTEM; TEMPLATES, each with a version and a schema; messages_for",
           "schema": "problems: what is wrong with an answer",
           "answers": "ask_json: asks, checks, and asks again"},
    "nl": {"title": "De bestanden van je assistent na vandaag", "today": "nieuw vandaag", "given": "uit JSON waar je op kunt bouwen",
           "earlier": "uit Project: je assistent antwoordt", "uses": "gebruikt",
           "client": "chat: stuurt een gesprek, geeft de tekst terug",
           "prompts": "SYSTEM; TEMPLATES, elk met een versie en een schema; messages_for",
           "schema": "problems: wat er mis is met een antwoord",
           "answers": "ask_json: vraagt, controleert, en vraagt opnieuw"},
}


def files_picture(lang: str) -> str:
    """The four files of the package app after this lesson, what each holds, and which file ask_json uses. Ink only:
    the picture is about the code, not about what a model makes."""
    w = _FILES_WORDS[lang]

    def box(name: str, what: str, note: str, now: bool) -> str:
        border = "2px solid var(--ea-ink)" if now else "1px solid var(--ea-line)"
        return (f'<div style="flex:1 1 13rem;min-width:0;border:{border};border-radius:8px;padding:0.45rem 0.6rem">'
                f'<div>{_mono("app/" + name + ".py")}</div><div style="margin-top:0.15rem">{_ui(what, "var(--ea-ink)")}</div>'
                f'<div style="margin-top:0.15rem">{_ui(note, size=11)}</div></div>')

    top = box("answers", w["answers"], w["today"], True)
    used = "".join([box("client", w["client"], w["earlier"], False), box("prompts", w["prompts"], w["today"], True),
                    box("schema", w["schema"], w["given"], False)])
    arrow = f'<div style="margin:0.3rem 0 0.3rem 1rem">{_ui("↓ " + w["uses"], size=12)}</div>'
    body = (f'<div style="display:flex">{top}</div>{arrow}<div style="display:flex;flex-wrap:wrap;gap:0.5rem">{used}</div>')
    return yours.figure(w["title"], body, alt=w["title"])


_RUNS_WORDS = {
    "en": {"title": "The event as JSON: every run, try by try", "plain": "without the rule about JSON",
           "rule": "with the rule about JSON", "first": "fit at the first try: {n} of {k}", "final": "in the end: {n} of {k}",
           "fits": "fits the schema", "not": "does not fit", "keys": {"lunch": "lunch", "football": "football", "book-club": "book club"}},
    "nl": {"title": "De afspraak als JSON: elke run, poging na poging", "plain": "zonder de regel over JSON",
           "rule": "met de regel over JSON", "first": "paste bij de eerste poging: {n} van de {k}", "final": "uiteindelijk: {n} van de {k}",
           "fits": "past bij het schema", "not": "past niet", "keys": {"lunch": "lunch", "football": "voetbal", "book-club": "leesclub"}},
}


def _square(ok: bool) -> str:
    colour = "var(--ea-pred)" if ok else "var(--ea-loss)"
    return f'<span style="display:block;width:12px;height:12px;border-radius:3px;background:{colour};margin:1px 0"></span>'


def runs_picture(lang: str) -> str:
    """The recorded runs, without and with the rule about JSON in the system prompt: a column for each run, a square
    for each try, in the colour of what a model makes when it fits the schema and of a mistake when it does not."""
    w, f = _RUNS_WORDS[lang], findings()
    cols = []
    for mode in ("plain", "rule"):
        rows = []
        for key in MESSAGES:
            runs = "".join(f'<span style="display:inline-flex;flex-direction:column;margin-right:3px;vertical-align:top">'
                           f'{"".join(_square(fits(t["text"])) for t in a["tries"])}</span>' for a in answers(mode, key))
            rows.append(f'<div style="display:flex;gap:0.5rem;align-items:flex-start;padding:0.2rem 0;border-top:1px solid var(--ea-line)">'
                        f'<span style="min-width:4.5rem">{_ui(w["keys"][key])}</span><span>{runs}</span></div>')
        k = f[mode]["runs"]
        head = (f'<div>{_ui(w[mode], "var(--ea-ink)", 13, 600)}</div>'
                f'<div>{_ui(w["first"].format(n=f[mode]["first_all"], k=k) + " · " + w["final"].format(n=f[mode]["final"], k=k))}</div>')
        cols.append(f'<div style="flex:1 1 16rem;min-width:0">{head}{"".join(rows)}</div>')
    key = (f'<div style="margin-top:0.4rem;display:flex;gap:1rem;align-items:center">'
           f'<span style="display:inline-flex;gap:0.3rem;align-items:center">{_square(True)}{_ui(w["fits"], size=11)}</span>'
           f'<span style="display:inline-flex;gap:0.3rem;align-items:center">{_square(False)}{_ui(w["not"], size=11)}</span></div>')
    return yours.figure(w["title"], f'<div style="display:flex;flex-wrap:wrap;gap:1rem">{"".join(cols)}</div>{key}', alt=w["title"])


_BOOK_WORDS = {
    "en": {"title": "The book club, with the rule: what the model gave as the hour", "run": "run",
           "note": "The message says 3 in the afternoon: hour 15. Every answer fits the schema."},
    "nl": {"title": "De leesclub, met de regel: wat het model als uur gaf", "run": "run",
           "note": "Het bericht zegt 3 uur 's middags: uur 15. Elk antwoord past bij het schema."},
}


def book_picture(lang: str) -> str:
    """The answers about the book club, with the rule about JSON: the JSON each run gave, its hour marked as a mistake
    when it is not 15. The answers are text the model wrote."""
    w = _BOOK_WORDS[lang]
    rows = []
    for a in answers("rule", "book-club"):
        text = a["tries"][-1]["text"]
        value = json.loads(text) if fits(text) else None
        hour = value["hour"] if value else None
        mark = "" if hour == 15 else ";background:var(--ea-loss-wash)"
        rows.append(f'<div style="display:flex;gap:0.5rem;align-items:baseline;padding:0.15rem 0;border-top:1px solid var(--ea-line)">'
                    f'<span style="min-width:3.2rem">{_ui(w["run"] + " " + str(a["seed"] + 1), size=11)}</span>'
                    f'<span style="font-family:var(--ea-font-code);font-size:12px;color:var(--ea-pred-text);padding:0 0.2rem;border-radius:3px{mark}">'
                    f'{html.escape(" ".join(text.split()))}</span></div>')
    body = "".join(rows) + f'<div style="margin-top:0.3rem">{_ui(w["note"], size=12)}</div>'
    return yours.figure(w["title"], body, alt=w["title"])


def _message_blocks(messages, limit: int = 120) -> str:
    """Messages as a column of blocks with their role: the user's and the system's in ink, the assistant's in the
    colour of what a model writes."""
    out = []
    for m in messages if isinstance(messages, list) else []:
        role = str(m.get("role")) if isinstance(m, dict) else "?"
        content = str(m.get("content")) if isinstance(m, dict) else repr(m)
        content = content if len(content) <= limit else content[:limit - 1] + "…"
        colour = "var(--ea-pred-text)" if role == "assistant" else "var(--ea-ink)"
        out.append(f'<div style="display:flex;gap:0.5rem;align-items:baseline;margin:0.15rem 0">'
                   f'<span style="min-width:5.2rem;font-family:var(--ea-font-code);font-size:11px;color:var(--ea-ink-2)">{html.escape(role)}</span>'
                   f'<span style="flex:1;min-width:0;border:1.5px solid {colour};border-radius:5px;padding:0.15rem 0.4rem">{_mono(content, colour)}</span></div>')
    return "".join(out)


def prompts_picture(seen) -> str:
    """The messages the learner's messages_for makes for the football message, block by block, beside what the
    course's makes when it differs."""
    run = seen.run
    fn = run.names.get("messages_for") if run is not None and run.error is None else None
    if not callable(fn):
        return ""
    w = {"en": ("What your messages_for makes for the football message", "It should make:", "Your function stopped: {why}"),
         "nl": ("Wat je messages_for maakt voor het bericht over voetbal", "Hij hoort te maken:", "Je functie stopte: {why}")}[seen.lang]
    want = course_messages("event", FOOTBALL)
    try:
        got = fn("event", FOOTBALL)
    except Exception as e:   # the learner's code
        return yours.figure(w[0], _ui(w[2].format(why=f"{type(e).__name__}: {e}")), alt=w[0])
    if not isinstance(got, list):
        got = [{"role": "?", "content": repr(got)}]
    body = _message_blocks(got, 400)
    if got != want:
        body += f'<div style="margin-top:0.4rem">{_ui(w[1])}</div>' + _message_blocks(want, 400)
    return yours.figure(w[0], body, alt=w[0])


# --- exercise 2: app/answers.py --------------------------------------------------------------------------------------------

START_ASK = '''import json

from app.client import chat
from app.prompts import TEMPLATES, messages_for
from app.schema import problems


def ask_json(url, name, text, tries=3):
    messages = messages_for(name, text)
    schema = TEMPLATES[name]["schema"]
    for attempt in range(tries):
        answer = chat(url, messages)
    return None
'''
SOLUTION_ASK = START_ASK.replace('''        answer = chat(url, messages)
''', '''        answer = chat(url, messages)
        found = problems(answer, schema)
        if found == []:
            return json.loads(answer)
        fix = "Fix these problems: " + "; ".join(found) + ". Answer with the whole JSON again."
        messages = messages + [{"role": "assistant", "content": answer}, {"role": "user", "content": fix}]
''')
FENCED = '```json\n{"what": "Football training", "day": "Tuesday", "hour": 19}\n```'
GOOD = '{"what": "Football training", "day": "Tuesday", "hour": 19}'
TEXT_HOUR = '{"what": "Lunch", "day": "Thursday", "hour": "noon"}'
TITLE = '{"title": "A rainy day at home"}'


def _braced(text: str) -> str:
    """A stand-in's default answer is filled in with {question}: its own braces doubled."""
    return text.replace("{", "{{").replace("}", "}}")


def _sent(log: list) -> list:
    """The requests in a server's log that sent JSON."""
    return [e for e in log if e["method"] == "POST" and isinstance(e["body"], dict)]


def _content(entry) -> str:
    try:
        return entry["answer"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""


def _in_prompts(e: BaseException) -> bool:
    """Whether an error came from the learner's own app/prompts.py, as Python imported it."""
    where = str(getattr(e, "filename", "") or "") if isinstance(e, SyntaxError) else ""
    tb = e.__traceback__
    while tb is not None:
        where += "\n" + tb.tb_frame.f_code.co_filename
        tb = tb.tb_next
    return any(line.replace("\\", "/").endswith("app/prompts.py") for line in where.split("\n"))


class _Then(serve.StandIn):
    """A stand-in whose first answer is `first`, and every later answer `then`, whatever the messages say."""

    def __init__(self, first: str, then: str):
        super().__init__()
        self.first, self.then, self.asked = first, then, 0

    def answer(self, request: dict, heard) -> dict:
        self.default = _braced(self.first if self.asked == 0 else self.then)
        self.asked += 1
        return super().answer(request, heard)


def _asked(name: str, text: str) -> tuple[list, dict]:
    """The first messages of a task and the schema of its answer, as the learner's app/prompts.py, which their code
    just imported, makes them; the course's when it does not run."""
    mod = sys.modules.get("app.prompts")
    try:
        return mod.messages_for(name, text), mod.TEMPLATES[name]["schema"]
    except Exception:   # the learner's module
        return course_messages(name, text), TEMPLATES[name]["schema"]


def _expected(answers: list, schema: dict, tries: int) -> tuple:
    """How many requests the course's ask_json sends when the stand-in answers `answers` in turn, with the schema of
    the learner's own template, and what it gives back: the JSON of the first answer that fits, or None."""
    for k in range(tries):
        if course_problems(answers[min(k, len(answers) - 1)], schema) == []:
            return k + 1, json.loads(answers[min(k, len(answers) - 1)])
    return tries, None


def _judge_ask(name: str, text: str, answers: list, tries: int = 3):
    def judge(got, running):
        asked, schema = _asked(name, text)
        posts, want = _expected(answers, schema, tries)
        sent = _sent(running.log)
        if not sent:
            return "not-to-url"
        if sent[0]["body"].get("messages") != asked:
            return "first"
        if len(sent) >= 2:
            first, second = sent[0]["body"].get("messages"), sent[1]["body"].get("messages")
            said = _content(sent[0])
            if second == first:
                return "no-fix"
            if not isinstance(second, list) or second[:len(first)] != first:
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
        if len(sent) < posts and want is not None and got == want:
            return "unchecked"
        if len(sent) != posts:
            return "tries"
        if want is None:
            return None if got is None else "not-none"
        if isinstance(got, str) and got == _content(sent[-1]):
            return "text"
        return None if got == want else "return"
    return judge


ASK_CASES = [   # with the course's prompts: 2 requests and the event; 2 requests and None; 1 request and the title
    Case(lambda: _Then(FENCED, GOOD), (URL, "event", FOOTBALL), _judge_ask("event", FOOTBALL, [FENCED, GOOD])),
    Case(lambda: serve.StandIn(default=_braced(TEXT_HOUR)),
         (URL, "event", "Lunch on Thursday at noon.", 2), _judge_ask("event", "Lunch on Thursday at noon.", [TEXT_HOUR], 2)),
    Case(lambda: serve.StandIn(default=_braced(TITLE)), (URL, "title", RAIN), _judge_ask("title", RAIN, [TITLE]),
         shown_args=("url", '"title"', {"en": "[the text about the rain]", "nl": "[de tekst over de regen]"})),
]


def tries_picture(seen) -> str:
    """The requests the learner's ask_json sent for the football message, against the stand-in that first answers in a
    code block: a row of message blocks for each, and what the function gave back."""
    run = seen.run
    fn = run.names.get("ask_json") if run is not None and run.error is None else None
    if not callable(fn):
        return ""
    got, error, running = apicheck.call(fn, ASK_CASES[0])
    sent = _sent(running.log)
    w = {"en": ("What your ask_json sent for the football message", "Request {k}, with {n} messages:",
                "It gave back:", "It stopped: {why}", "It sent nothing yet."),
         "nl": ("Wat je ask_json stuurde voor het bericht over voetbal", "Request {k}, met {n} berichten:",
                "Hij gaf terug:", "Hij stopte: {why}", "Hij stuurde nog niets.")}[seen.lang]
    parts = []
    for k, e in enumerate(sent[:TRIES + 1]):
        messages = e["body"].get("messages") or []
        parts.append(f'<div style="margin-top:0.3rem">{_ui(w[1].format(k=k + 1, n=len(messages)))}</div>' + _message_blocks(messages, 90))
    if not sent:
        parts.append(_ui(w[4]))
    end = w[3].format(why=f"{type(error).__name__}: {error}") if error is not None else w[2]
    parts.append(f'<div style="margin-top:0.4rem">{_ui(end)}</div>')
    if error is None:
        parts.append(f'<div>{_mono(repr(got), "var(--ea-pred-text)")}</div>')
    return yours.figure(w[0], "".join(parts), alt=w[0])


_COMMON = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
                "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."},
}

EXERCISES = {
    "messages_for": FunctionBox(
        name="messages_for",
        picture=prompts_picture,
        function="messages_for",
        start=START_PROMPTS,
        solution=SOLUTION_PROMPTS,
        visible=[PROMPT_CASES[0]],
        hidden=PROMPT_CASES[1:],
        save_as=("app", "prompts"),
        mistakes=[
            Mistake("dict-format", error=lambda e: isinstance(e, AttributeError) and "format" in str(e),
                    en="With `{call}`, your function calls `.format` on the whole entry of the template, a dictionary. The text of the template is in its `\"text\"`: `template[\"text\"].format(text=text)`.",
                    nl="Met `{call}` roept je functie `.format` aan op de hele ingang van de template, een dictionary. De tekst van de template staat in zijn `\"text\"`: `template[\"text\"].format(text=text)`."),
            Mistake("no-system", wrong=SOLUTION_PROMPTS.replace('''    return [{"role": "system", "content": SYSTEM},
            {"role": "user"''', '''    return [{"role": "user"'''),
                    en="With `{call}`, the system prompt is not among the messages. The first message is `{{\"role\": \"system\", \"content\": SYSTEM}}`, then the filled template as the user's.",
                    nl="Met `{call}` staat de system prompt niet tussen de berichten. Het eerste bericht is `{{\"role\": \"system\", \"content\": SYSTEM}}`, daarna de ingevulde template als bericht van de gebruiker."),
            Mistake("order", wrong=SOLUTION_PROMPTS.replace('''    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": template["text"].format(text=text)}]''', '''    return [{"role": "user", "content": template["text"].format(text=text)},
            {"role": "system", "content": SYSTEM}]'''),
                    en="With `{call}`, the system prompt comes after the user's message. It goes first: the model reads it before anything else.",
                    nl="Met `{call}` komt de system prompt na het bericht van de gebruiker. Hij hoort vooraan: het model leest hem vóór al het andere."),
            Mistake("unfilled", wrong=SOLUTION_PROMPTS.replace('template["text"].format(text=text)', 'template["text"]'),
                    en="With `{call}`, the user's message is the template with its field still in it. Fill it in with `.format(text=text)`.",
                    nl="Met `{call}` is het bericht van de gebruiker de template met het veld er nog in. Vul het in met `.format(text=text)`."),
            Mistake("one-template", wrong=SOLUTION_PROMPTS.replace('TEMPLATES[name]\n', 'TEMPLATES["event"]\n'), state="almost",
                    en="With `{call}`, your function uses the template of the event, whatever `name` it gets. Take the template that `name` says: `TEMPLATES[name]`.",
                    nl="Met `{call}` gebruikt je functie de template van de afspraak, welke `name` hij ook krijgt. Neem de template die `name` noemt: `TEMPLATES[name]`."),
            Mistake("text-only", matches=lambda got, text: isinstance(got, str),
                    en="With `{call}`, your function gives back a text, `{got}`. It gives back the list of messages: the system's, then the user's.",
                    nl="Met `{call}` geeft je functie een tekst terug, `{got}`. Hij geeft de lijst met berichten terug: dat van het systeem, en dan dat van de gebruiker."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "`messages_for` has only its `def` line. Take the template that `name` says, and give back a list of two messages: the system prompt, and the filled template as the user's.",
                      "nl": "`messages_for` heeft alleen zijn `def`-regel. Neem de template die `name` noemt, en geef een lijst met twee berichten terug: de system prompt, en de ingevulde template als bericht van de gebruiker."},
            "wrong": {"en": "With `{call}`, your function gives `{got}`, and the course's gives `{want}`. Did you leave `SYSTEM` and `TEMPLATES` as they are?",
                      "nl": "Met `{call}` geeft je functie `{got}`, en die van de course `{want}`. Heb je `SYSTEM` en `TEMPLATES` gelaten zoals ze zijn?"},
            "almost": {"en": "Right for the football message. With `{call}`, your function gives `{got}`, and the course's gives `{want}`.",
                       "nl": "Goed voor het bericht over voetbal. Met `{call}` geeft je functie `{got}`, en die van de course `{want}`."},
            "no_function": {"en": "Your box does not make a function `messages_for` any more. The last lines start with `def messages_for(name, text):`.",
                            "nl": "Je vak maakt geen functie `messages_for` meer. De laatste regels beginnen met `def messages_for(name, text):`."},
            "correct": {"en": "Your `messages_for` makes the first messages of every task: the system prompt, and the filled template. The check saved the prompts and the function in your project, as `{saved}`.",
                        "nl": "Je `messages_for` maakt de eerste berichten van elke taak: de system prompt, en de ingevulde template. De check heeft de prompts en de functie in je project bewaard, als `{saved}`."},
            "kept": {"en": "Your `messages_for` is right. You changed `{saved}` yourself, so the check left it as it is.",
                     "nl": "Je `messages_for` is goed. Je hebt `{saved}` zelf veranderd, dus de check liet het staan zoals het is."},
        },
    ),
    "ask_json": ApiBox(
        name="ask_json",
        picture=tries_picture,
        function="ask_json",
        start=START_ASK,
        solution=SOLUTION_ASK,
        cases=ASK_CASES,
        save_as=("app", "answers"),
        mistakes=[
            Mistake("prompts-error", error=lambda e: _in_prompts(e),
                    en="With `{call}`, Python stops inside your own `work/projects/app/prompts.py`, which the box uses. Does your box ask for a task that `TEMPLATES` does not have? And is the file still whole, with the tasks \"event\" and \"title\"? Repair it there, or delete the file and run the check of exercise 1 again, which then saves its box once more.",
                    nl="Met `{call}` stopt Python binnen je eigen `work/projects/app/prompts.py`, die het vak gebruikt. Vraagt je vak om een taak die `TEMPLATES` niet heeft? En is het bestand nog heel, met de taken \"event\" en \"title\"? Herstel het daar, of gooi het bestand weg en voer de check van oefening 1 nog eens uit, die dan zijn vak opnieuw bewaart."),
            Mistake("no-prompts", error=lambda e: isinstance(e, ModuleNotFoundError) and "app.prompts" in str(e),
                    en="Your `messages_for` is not in your project yet. The check of exercise 1 saves it, as `work/projects/app/prompts.py`, once it is right. Finish exercise 1 first; this check then runs again by itself.",
                    nl="Je `messages_for` staat nog niet in je project. De check van oefening 1 bewaart hem, als `work/projects/app/prompts.py`, zodra hij goed is. Maak eerst oefening 1 af; deze check draait daarna vanzelf opnieuw."),
            Mistake("prompts-name", error=lambda e: isinstance(e, ImportError) and "cannot import name" in str(e) and "app.prompts" in str(e),
                    en="Your `work/projects/app/prompts.py` has no `TEMPLATES` or no `messages_for`, which the box imports. Did you give one of them another name there? Give it its name back.",
                    nl="Je `work/projects/app/prompts.py` heeft geen `TEMPLATES` of geen `messages_for`, die het vak importeert. Heb je daar een van de twee een andere naam gegeven? Geef hem zijn naam terug."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "The loop asks, but does nothing with the answer yet. After `chat`: find the problems with `problems(answer, schema)`, give back the JSON when there are none, and else ask again with what was wrong.",
                      "nl": "De loop vraagt, maar doet nog niets met het antwoord. Na `chat`: zoek de problemen met `problems(answer, schema)`, geef de JSON terug als er geen zijn, en vraag anders opnieuw met wat er mis was."},
            "not-to-url": {"en": "With `{call}`, your function sent nothing to the server. Does it call `chat(url, messages)` in the loop?",
                           "nl": "Met `{call}` stuurde je functie niets naar de server. Roept hij `chat(url, messages)` aan in de loop?"},
            "first": {"en": "With `{call}`, the first request did not carry what `messages_for(name, text)` makes. Start from those messages, as the start of the function does.",
                      "nl": "Met `{call}` had het eerste request niet wat `messages_for(name, text)` maakt. Begin met die berichten, zoals het begin van de functie doet."},
            "no-fix": {"en": "With `{call}`, the second request was the same as the first: the model hears nothing about what was wrong. Add its answer and a message with the problems to `messages` before you ask again.",
                       "nl": "Met `{call}` was het tweede request hetzelfde als het eerste: het model hoort niets over wat er mis was. Voeg zijn antwoord en een bericht met de problemen toe aan `messages` voordat je opnieuw vraagt."},
            "lost": {"en": "With `{call}`, the second request lost messages of the first. Keep the messages so far, and add the answer and the problems at the end.",
                     "nl": "Met `{call}` miste het tweede request berichten van het eerste. Houd de berichten tot nu toe, en voeg het antwoord en de problemen achteraan toe."},
            "no-answer": {"en": "With `{call}`, the second request does not carry the answer the model gave. Add it as `{{\"role\": \"assistant\", \"content\": answer}}`: then the model knows what its problems are about.",
                          "nl": "Met `{call}` heeft het tweede request het antwoord van het model niet. Voeg het toe als `{{\"role\": \"assistant\", \"content\": answer}}`: dan weet het model waar de problemen over gaan."},
            "no-problems": {"en": "With `{call}`, your message after the answer does not name the problems that `problems` found. Put them in it, such as with `\"; \".join(found)`.",
                            "nl": "Met `{call}` noemt je bericht na het antwoord de problemen niet die `problems` vond. Zet ze erin, bijvoorbeeld met `\"; \".join(found)`."},
            "order": {"en": "With `{call}`, the problems come before the answer they are about. First the assistant's answer, then the user's message with the problems.",
                      "nl": "Met `{call}` komen de problemen vóór het antwoord waar ze over gaan. Eerst het antwoord van de assistant, dan het bericht van de gebruiker met de problemen."},
            "unchecked": {"en": "With `{call}`, your function gave back JSON from an answer that does not fit as it came: `problems` finds something in it. Give back the JSON of an answer only when `problems` finds nothing in it, and ask again otherwise.",
                          "nl": "Met `{call}` gaf je functie JSON terug uit een antwoord dat niet past zoals het kwam: `problems` vindt er iets in. Geef de JSON van een antwoord alleen terug als `problems` er niets in vindt, en vraag anders opnieuw."},
            "tries": {"en": "With `{call}`, your function sent another number of requests than it should. It asks until an answer fits, and at most `tries` times.",
                      "nl": "Met `{call}` stuurde je functie een ander aantal requests dan hij hoort. Hij vraagt tot een antwoord past, en hooguit `tries` keer."},
            "not-none": {"en": "With `{call}`, no answer fit, and your function gave back `{got}`. When no try fits, it gives back `None`: the last line of the start does that.",
                         "nl": "Met `{call}` paste geen enkel antwoord, en je functie gaf `{got}` terug. Als geen poging past, geeft hij `None` terug: de laatste regel van het begin doet dat."},
            "text": {"en": "With `{call}`, your function gives back the text of the answer, `{got}`. Give back what `json.loads` reads from it: a dictionary the rest of your app can use.",
                     "nl": "Met `{call}` geeft je functie de tekst van het antwoord terug, `{got}`. Geef terug wat `json.loads` ervan leest: een dictionary die de rest van je app kan gebruiken."},
            "return": {"en": "With `{call}`, your function gives back `{got}`, and not the JSON of the answer that fit.",
                       "nl": "Met `{call}` geeft je functie `{got}` terug, en niet de JSON van het antwoord dat paste."},
            "correct": {"en": "Your `ask_json` asks with your prompts through your client, checks each answer, and asks again with what was wrong. The check saved it in your project, as `{saved}`.",
                        "nl": "Je `ask_json` vraagt met je prompts via je client, controleert elk antwoord, en vraagt opnieuw met wat er mis was. De check heeft hem in je project bewaard, als `{saved}`."},
            "kept": {"en": "Your `ask_json` is right. You changed `{saved}` yourself, so the check left it as it is.",
                     "nl": "Je `ask_json` is goed. Je hebt `{saved}` zelf veranderd, dus de check liet het staan zoals het is."},
        },
    ),
}

REQUIRED = ["messages_for", "ask_json", "new-version"]

CASES = [
    ("messages_for", START_PROMPTS, "empty", None),
    ("messages_for", SOLUTION_PROMPTS, "correct", None),
    ("messages_for", SOLUTION_PROMPTS.replace('template["text"].format(text=text)', 'template.format(text=text)'), "error", "dict-format"),
    ("messages_for", EXERCISES["messages_for"].mistakes[1].wrong, "wrong", "no-system"),
    ("messages_for", EXERCISES["messages_for"].mistakes[2].wrong, "wrong", "order"),
    ("messages_for", EXERCISES["messages_for"].mistakes[3].wrong, "wrong", "unfilled"),
    ("messages_for", EXERCISES["messages_for"].mistakes[4].wrong, "almost", "one-template"),
    ("messages_for", SOLUTION_PROMPTS.replace('''    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": template["text"].format(text=text)}]''', '''    return template["text"].format(text=text)'''), "wrong", "text-only"),
    ("messages_for", SOLUTION_PROMPTS.replace("Answer briefly.", "Answer in one sentence."), "wrong", None),
    ("ask_json", START_ASK, "empty", None),
    ("ask_json", SOLUTION_ASK.replace("import TEMPLATES, messages_for", "import TEMPLATES, messages_for, PROMPTS"), "error", "prompts-name"),
    ("ask_json", SOLUTION_ASK.replace("from app.prompts import", "from app.prompts.old import"), "error", "no-prompts"),
    ("ask_json", SOLUTION_ASK.replace("messages_for(name, text)", 'messages_for("events", text)'), "error", "prompts-error"),
    ("ask_json", SOLUTION_ASK, "correct", None),
    ("ask_json", SOLUTION_ASK.replace('        answer = chat(url, messages)\n',
                                      '        answer = "{}"\n'), "wrong", "not-to-url"),
    ("ask_json", SOLUTION_ASK.replace('''        messages = messages + [{"role": "assistant", "content": answer}, {"role": "user", "content": fix}]
''', ""), "wrong", "no-fix"),
    ("ask_json", SOLUTION_ASK.replace('[{"role": "assistant", "content": answer}, {"role": "user", "content": fix}]',
                                      '[{"role": "user", "content": fix}]'), "wrong", "no-answer"),
    ("ask_json", SOLUTION_ASK.replace('[{"role": "assistant", "content": answer}, {"role": "user", "content": fix}]',
                                      '[{"role": "user", "content": fix}, {"role": "assistant", "content": answer}]'), "wrong", "order"),
    ("ask_json", SOLUTION_ASK.replace('"Fix these problems: " + "; ".join(found) + ". Answer with the whole JSON again."',
                                      '"Answer with the whole JSON again."'), "wrong", "no-problems"),
    ("ask_json", SOLUTION_ASK.replace("messages = messages + [", "messages = [{\"role\": \"system\", \"content\": \"JSON\"}] + ["), "wrong", "lost"),
    ("ask_json", SOLUTION_ASK.replace("    messages = messages_for(name, text)\n", "    messages = [{\"role\": \"user\", \"content\": text}]\n"), "wrong", "first"),
    ("ask_json", SOLUTION_ASK.replace("return json.loads(answer)", "return answer"), "wrong", "text"),
    ("ask_json", SOLUTION_ASK.replace("        if found == []:\n            return json.loads(answer)\n",
                                      "        return json.loads(answer.strip('`').removeprefix('json'))\n"), "wrong", "unchecked"),
    ("ask_json", SOLUTION_ASK.replace("    return None\n", "    return {}\n"), "wrong", "not-none"),
    ("ask_json", SOLUTION_ASK.replace("for attempt in range(tries):", "for attempt in range(tries + 1):"), "wrong", "tries"),
    ("ask_json", SOLUTION_ASK.replace("return json.loads(answer)", "return json.loads(answer)[\"what\"]"), "wrong", "return"),
]


# --- the demo: your assistant turns a message into an event -----------------------------------------------------------------

def downloaded() -> bool:
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    chat.download_or_stop(button, texts)


def learner_ask_json():
    """ask_json from the learner's work/projects/app/answers.py, and None; or None, and the error that stopped the import,
    such as one in their app/prompts.py. Modules it imports that are missing or do not pass are the course's (builds_on)."""
    run = run_code("from app.answers import ask_json")
    if run.error is not None:
        return None, f"{type(run.error).__name__}: {run.error}"
    return run.names.get("ask_json"), None


def version_of(name: str):
    """The version of a template in the learner's app/prompts.py, or the course's when theirs does not run."""
    run = run_code("from app.prompts import TEMPLATES")
    templates = run.names.get("TEMPLATES") if run.error is None else None
    try:
        return templates[name]["version"]
    except (TypeError, KeyError):
        return TEMPLATES[name]["version"]


def event_card(value, words: dict) -> str:
    """The event as the rest of an app would show it: a card with the day, the hour and what, in the colour of what a
    model makes. Values that are not as the schema says are shown as they came."""
    def get(key):
        return value.get(key) if isinstance(value, dict) else None
    hour = get("hour")
    when = f"{get('day')}, {int(hour):02d}:00" if isinstance(hour, (int, float)) and not isinstance(hour, bool) and 0 <= hour < 24 \
        else f"{get('day')}, {hour}"
    return (f'<div style="display:inline-block;min-width:14rem;border:1.5px solid var(--ea-pred-text);border-radius:10px;padding:0.5rem 0.8rem;margin-top:0.3rem">'
            f'<div>{_ui(words["card"], size=11)}</div>'
            f'<div style="font-family:var(--ea-font-ui);font-size:17px;font-weight:600;color:var(--ea-pred-text)">{html.escape(str(get("what")))}</div>'
            f'<div style="font-family:var(--ea-font-ui);font-size:14px;color:var(--ea-pred-text)">{html.escape(when)}</div></div>')


def demo(fn, text: str, words: dict, lang: str) -> str:
    """The learner's ask_json for the template "event", with a message of the learner's, against the course's server with
    Qwen3: each request it sent with the answer, and what it gave back, as JSON and as a card."""
    import time
    running = serve.model_server()
    before = len(running.log)
    start = time.perf_counter()
    try:
        got = fn(running.url, "event", text)
    except Exception as e:   # the learner's code, against the real server; it may have run out of memory
        said = words["memory"] if serve.out_of_memory(e) else words["failed"].format(why=f"{type(e).__name__}: {e}")
        return f"<p><em>{html.escape(said)}</em></p>"
    seconds = f"{time.perf_counter() - start:.1f}".replace(".", "," if lang == "nl" else ".")
    sent = _sent(running.log[before:])
    schema = _asked("event", text)[1]   # the schema of the learner's own template, as their ask_json checks it
    course = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]
    whose = words["course"].format(files=", ".join(course)) if course else words["yours"]
    parts = [f'<div>{_ui(words["prompt"].format(version=version_of("event"), n=len(sent), seconds=seconds) + " " + whose)}</div>']
    for k, e in enumerate(sent):
        said = _content(e)
        found = course_problems(said, schema)
        verdict = words["fits"] if not found else words["found"].format(p="; ".join(found))
        parts.append(f'<div style="margin-top:0.3rem">{_ui(words["try"].format(n=k + 1) + " " + verdict, "var(--ea-ink)" if not found else "var(--ea-loss-text)")}</div>'
                     f'<div>{_mono(said, "var(--ea-pred-text)")}</div>')
    if got is None:
        parts.append(f'<div style="margin-top:0.4rem">{_ui(words["none"], "var(--ea-ink)")}</div>')
    else:
        parts.append(f'<div style="margin-top:0.4rem">{_ui(words["gave"])}</div><div>{_mono(repr(got), "var(--ea-pred-text)")}</div>')
        parts.append(event_card(got, words))
    return yours.figure(words["title"], "".join(parts), alt=words["title"])
