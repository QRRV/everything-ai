# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-project-call: app/client.py, a function that sends a whole conversation to the API and gives back
the text of the answer, and a conversation of several turns with it, both against a stand-in of the server; the
course's recorded runs (data/course-apis-qwen.json): does Qwen3 know a name from earlier in the conversation; and the
pictures of the conversations."""
import html
import json

from everything_ai import apicheck, chat, paths, serve, yours
from everything_ai.apicheck import URL, ApiBox, Case
from everything_ai.check import Mistake, run_code

FIRST = "My name is Sam. Suggest a dinner for tonight, in one sentence."
ASK = "What is my name?"
SEEDS = 8
RECORD = {
    "with": {"turns": [FIRST, ASK], "max_tokens": 60, "seeds": SEEDS},
    "without": {"messages": [{"role": "user", "content": ASK}], "max_tokens": 60, "seeds": SEEDS},
}
DATA = paths.ROOT / "data" / "course-apis-qwen.json"


def course() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-project-call"]


def says_sam(text: str) -> bool:
    return "sam" in text.lower()


def findings() -> dict:
    """What the recorded runs show: how many of the answers to "What is my name?" name Sam, with the conversation
    before it and without; how many of those with it give Sam as the model's own name ("I am Sam.", "My name is
    Sam."), and how many say "Your name"."""
    R = course()
    named = [a["turns"][1]["text"] for a in R["with"]["answers"]]
    alone = [a["text"] for a in R["without"]["answers"]]
    return {"with": sum(says_sam(t) for t in named), "without": sum(says_sam(t) for t in alone),
            "own": sum(t.strip().lower().startswith(("i am sam", "my name is sam")) for t in named),
            "your": sum("your name" in t.lower() for t in named),
            "tokens_first": R["with"]["answers"][0]["turns"][0]["prompt_tokens"],
            "tokens_second": R["with"]["answers"][0]["turns"][1]["prompt_tokens"]}


_RUNS_WORDS = {
    "en": {"title": "What is my name? Asked eight times each way", "with": "with the conversation before it",
           "without": "the question alone", "names": "names Sam", "run": "run"},
    "nl": {"title": "Wat is mijn naam? Elke manier acht keer gevraagd", "with": "met het gesprek ervoor",
           "without": "de vraag alleen", "names": "noemt Sam", "run": "run"},
}


def runs_picture(lang: str) -> str:
    """The recorded answers to "What is my name?", side by side: with the first turn sent along, and without it. The
    answers are text the model wrote."""
    w, R = _RUNS_WORDS[lang], course()
    cols = [(w["with"], [a["turns"][1]["text"] for a in R["with"]["answers"]]),
            (w["without"], [a["text"] for a in R["without"]["answers"]])]
    parts = []
    for title, texts in cols:
        n = sum(says_sam(t) for t in texts)
        rows = "".join(
            f'<div style="display:flex;gap:0.5rem;align-items:baseline;padding:0.15rem 0;border-top:1px solid var(--ea-line)">'
            f'<span style="font-family:var(--ea-font-ui);font-size:11px;color:var(--ea-ink-3);min-width:3.2rem">{w["run"]} {i + 1}</span>'
            f'<span style="font-family:var(--ea-font-code);font-size:12px;color:var(--ea-pred-text)">{html.escape(_short(t))}</span></div>'
            for i, t in enumerate(texts))
        parts.append(f'<div style="flex:1 1 18rem;min-width:0"><div style="font-family:var(--ea-font-ui);font-size:13px;font-weight:600;'
                     f'color:var(--ea-ink);margin-bottom:0.2rem">{html.escape(title)}: {n}/{len(texts)} {html.escape(w["names"])}</div>{rows}</div>')
    body = f'<div style="display:flex;flex-wrap:wrap;gap:1rem">{"".join(parts)}</div>'
    return yours.figure(w["title"], body, alt=w["title"])


def _short(text: str, n: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n - 1] + "…"


# --- exercise 1: chat in app/client.py ---------------------------------------------------------------------------------

_START_CHAT = '''import json
import urllib.request


def chat(url, messages, max_tokens=200):
'''
SOLUTION_CHAT = '''import json
import urllib.request


def chat(url, messages, max_tokens=200):
    body = {"model": "qwen3-0.6b", "messages": messages, "max_tokens": max_tokens}
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        answer = json.loads(response.read())
    return answer["choices"][0]["message"]["content"]
'''
TALK = [{"role": "user", "content": "My name is Sam."}, {"role": "assistant", "content": "Hello Sam!"},
        {"role": "user", "content": ASK}]
LONG = "one two three four five six seven eight nine ten"


def _sent(running) -> list:
    return [e["body"] for e in running.log if e["method"] == "POST" and isinstance(e["body"], dict)]


def _judge_chat(messages, max_tokens, want, given=True):
    def judge(got, running):
        sent = _sent(running)
        if not sent:
            return "nothing-sent"
        body = sent[-1]
        if body.get("messages") != messages:
            last = messages[-1:]
            return "only-last" if body.get("messages") == last or body.get("messages") == last[0].get("content") else "messages"
        if body.get("max_tokens") != max_tokens:
            return "max-tokens" if given else "default"
        if isinstance(got, dict):
            return "whole" if "choices" in got else "message"
        return None if got == want else "other"
    return judge


CHAT_CASES = [
    Case(lambda: serve.StandIn({ASK: "Your name is Sam."}), (URL, TALK), _judge_chat(TALK, 200, "Your name is Sam.", given=False)),
    Case(lambda: serve.StandIn({"Count to ten.": LONG}), (URL, [{"role": "user", "content": "Count to ten."}], 4),
         _judge_chat([{"role": "user", "content": "Count to ten."}], 4, "one two three four")),
]


# --- exercise 2: a conversation of several turns -------------------------------------------------------------------------

_START_TALK = '''from app.client import chat


def talk(url, said):
    messages = []
    answers = []
    for text in said:
        ...
    return answers
'''
SOLUTION_TALK = '''from app.client import chat


def talk(url, said):
    messages = []
    answers = []
    for text in said:
        messages.append({"role": "user", "content": text})
        answer = chat(url, messages)
        messages.append({"role": "assistant", "content": answer})
        answers.append(answer)
    return answers
'''
SAID = ["Hi, I am Sam.", "I like pasta.", "What do I like?"]
REPLIES = {"Hi, I am Sam.": "Hello Sam!", "I like pasta.": "Pasta is a fine choice.", "What do I like?": "You like pasta."}


def _judge_talk(got, running):
    sent = _sent(running)
    if len(sent) < len(SAID):
        return "too-few" if sent else "nothing-sent"
    want = [REPLIES[s] for s in SAID]
    for k, body in enumerate(sent[:len(SAID)]):
        messages = body.get("messages") or []
        users = [m.get("content") for m in messages if m.get("role") == "user"]
        helpers = [m.get("content") for m in messages if m.get("role") == "assistant"]
        if users != SAID[:k + 1]:
            return "forgets-user" if len(users) < k + 1 else "users"
        if helpers != want[:k]:
            return "forgets-answers"
    if got != want:
        return "answers"
    return None


TALK_CASES = [Case(lambda: serve.StandIn(REPLIES), (URL, SAID), _judge_talk)]


def talk_picture(seen) -> str:
    """The requests the learner's talk sent, one row each: a block for every message in it, the user's in ink and the
    assistant's in the colour of what a model writes. The rows grow: every request carries the whole conversation."""
    run = seen.run
    fn = run.names.get("talk") if run is not None and run.error is None else None
    if not callable(fn):
        return ""
    _got, _error, running = apicheck.call(fn, TALK_CASES[0])
    sent = _sent(running)
    w = {"en": ("What your talk sent, request by request", "request", "the user", "the assistant",
                "Your talk sent nothing yet.", "messages"),
         "nl": ("Wat je talk stuurde, request na request", "request", "de gebruiker", "de assistant",
                "Je talk stuurde nog niets.", "berichten")}[seen.lang]
    if not sent:
        return yours.figure(w[0], f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)">{w[4]}</div>', alt=w[0])
    rows = []
    for k, body in enumerate(sent[:6]):
        blocks = []
        for m in body.get("messages") or []:
            mine = m.get("role") == "assistant"
            colour = "var(--ea-pred-text)" if mine else "var(--ea-ink)"
            blocks.append(f'<span title="{html.escape(str(m.get("role")))}" style="display:inline-block;padding:0.1rem 0.4rem;margin:0.1rem;'
                          f'border:1.5px solid {colour};border-radius:5px;font-family:var(--ea-font-code);font-size:11px;color:{colour}">'
                          f'{html.escape(_short(str(m.get("content")), 26))}</span>')
        n = len(body.get("messages") or [])
        rows.append(f'<div style="display:flex;gap:0.5rem;align-items:center;flex-wrap:wrap;padding:0.2rem 0;border-top:1px solid var(--ea-line)">'
                    f'<span style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2);min-width:7.5rem">{w[1]} {k + 1}: {n} {w[5]}</span>'
                    f'{"".join(blocks)}</div>')
    key = (f'<div style="font-family:var(--ea-font-ui);font-size:11px;color:var(--ea-ink-2);margin-top:0.3rem">'
           f'<span style="color:var(--ea-ink)">▭ {w[2]}</span> · <span style="color:var(--ea-pred-text)">▭ {w[3]}</span></div>')
    return yours.figure(w[0], "".join(rows) + key, alt=w[0])


def chat_picture(seen) -> str:
    words = {"en": {"title": "What your chat sent, and what came back", "request": "Your code sent: {method} {path}",
                    "headers": "Headers:", "answer": "The server sent back: status {status}",
                    "stream": "a stream with {n} pieces", "none": "Your function sent nothing to the server yet."},
             "nl": {"title": "Wat je chat stuurde, en wat er terugkwam", "request": "Je code stuurde: {method} {path}",
                    "headers": "Headers:", "answer": "De server stuurde terug: status {status}",
                    "stream": "een stream met {n} stukjes", "none": "Je functie stuurde nog niets naar de server."}}[seen.lang]
    return apicheck.exchange(seen, "chat", CHAT_CASES[0], words)


def _no_client(e) -> bool:
    return isinstance(e, ImportError) and "app" in str(e)


def _not_bytes(e) -> bool:
    return isinstance(e, TypeError) and "bytes" in str(e)


def _no_default(e) -> bool:
    return isinstance(e, TypeError) and "missing 1 required positional argument: 'max_tokens'" in str(e)


_COMMON = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
                "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."},
    "nothing-sent": {"en": "With `{call}`, your function sent nothing to the server. Does it make a request and open it with `urlopen`?",
                     "nl": "Met `{call}` stuurde je functie niets naar de server. Maakt hij een request en opent hij dat met `urlopen`?"},
}

EXERCISES = {
    "chat": ApiBox(
        name="chat",
        picture=chat_picture,
        function="chat",
        start=_START_CHAT,
        solution=SOLUTION_CHAT,
        cases=CHAT_CASES,
        save_as=("app", "client"),
        mistakes=[
            Mistake("data-text", error=_not_bytes,
                    en="urllib sends bytes, not text: `json.dumps(body)` is text. Turn it into bytes with `.encode()`.",
                    nl="urllib verstuurt bytes, geen tekst: `json.dumps(body)` is tekst. Maak er bytes van met `.encode()`."),
            Mistake("no-default", error=_no_default,
                    en="With `{call}`, the call gives no `max_tokens`, and your `chat` needs one. Write `max_tokens=200` in the `def` line: a call that leaves it out then gets 200.",
                    nl="Met `{call}` geeft de aanroep geen `max_tokens`, en je `chat` heeft er een nodig. Schrijf `max_tokens=200` in de `def`-regel: een aanroep zonder krijgt dan 200."),
            Mistake("key", error=lambda e: isinstance(e, KeyError),
                    en="With `{call}`, a key you ask for is not in the answer. The text is in `[\"choices\"][0][\"message\"][\"content\"]`.",
                    nl="Met `{call}` staat een key waar je om vraagt niet in het antwoord. De tekst staat in `[\"choices\"][0][\"message\"][\"content\"]`."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "`chat` has only its `def` line. Write it as your `ask` of [Calling a model over HTTP](/course/open/build-call-api?lang=en), with the whole `messages` and `max_tokens` in the body.",
                      "nl": "`chat` heeft alleen zijn `def`-regel. Schrijf hem zoals je `ask` van [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl), met de hele `messages` en `max_tokens` in de body."},
            "correct": {"en": "Right: your `chat` sends the whole conversation and `max_tokens`, and gives back the text. The check saved it in your project, as `{saved}`.",
                        "nl": "Goed: je `chat` stuurt het hele gesprek en `max_tokens`, en geeft de tekst terug. De check heeft hem in je project bewaard, als `{saved}`."},
            "kept": {"en": "Right. You changed `{saved}` yourself, so the check left it as it is.",
                     "nl": "Goed. Je hebt `{saved}` zelf veranderd, dus de check liet het staan zoals het is."},
            "only-last": {"en": "With `{call}`, your request carried only the last message. Send the whole list `messages`: the model keeps nothing between requests.",
                          "nl": "Met `{call}` had je request alleen het laatste bericht. Stuur de hele lijst `messages`: het model onthoudt niets tussen requests."},
            "messages": {"en": "With `{call}`, the messages in your request are not the `messages` your function got. Put that list in the body as it is.",
                         "nl": "Met `{call}` zijn de berichten in je request niet de `messages` die je functie kreeg. Zet die lijst zoals hij is in de body."},
            "default": {"en": "With `{call}`, the call gives no `max_tokens`, so `chat` should ask for 200; your request asked for something else. Write `max_tokens=200` in the `def` line, and `max_tokens` in the body.",
                        "nl": "Met `{call}` geeft de aanroep geen `max_tokens`, dus `chat` hoort om 200 te vragen; je request vroeg om iets anders. Schrijf `max_tokens=200` in de `def`-regel, en `max_tokens` in de body."},
            "max-tokens": {"en": "With `{call}`, your request did not ask for the `max_tokens` your function got. Put `max_tokens` in the body, not a fixed number.",
                           "nl": "Met `{call}` vroeg je request niet om de `max_tokens` die je functie kreeg. Zet `max_tokens` in de body, geen vast getal."},
            "whole": {"en": "With `{call}`, your function gives back the whole JSON of the answer. Give back only the text.",
                      "nl": "Met `{call}` geeft je functie de hele JSON van het antwoord terug. Geef alleen de tekst terug."},
            "message": {"en": "With `{call}`, your function gives back the message, `{got}`. One step further in: its `[\"content\"]`.",
                        "nl": "Met `{call}` geeft je functie het bericht terug, `{got}`. Nog één stap verder naar binnen: de `[\"content\"]` ervan."},
            "other": {"en": "With `{call}`, your function gives `{got}`, and not the text the stand-in answered.",
                      "nl": "Met `{call}` geeft je functie `{got}`, en niet de tekst die de stand-in antwoordde."},
        },
    ),
    "talk": ApiBox(
        name="talk",
        picture=talk_picture,
        function="talk",
        start=_START_TALK,
        solution=SOLUTION_TALK,
        cases=TALK_CASES,
        mistakes=[
            Mistake("no-client", error=_no_client,
                    en="Your `chat` is not in your project yet. The check of exercise 1 saves it, as `work/projects/app/client.py`, once it is right. Finish exercise 1 first; this check then runs again by itself.",
                    nl="Je `chat` staat nog niet in je project. De check van oefening 1 bewaart hem, als `work/projects/app/client.py`, zodra hij goed is. Maak eerst oefening 1 af; deze check draait daarna vanzelf opnieuw."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "The loop does nothing yet. For each `text`: add it to `messages` as the user's, ask `chat`, and add the answer as the assistant's, and to `answers`.",
                      "nl": "De loop doet nog niets. Voor elke `text`: voeg hem als bericht van de gebruiker aan `messages` toe, vraag `chat`, en voeg het antwoord toe als bericht van de assistant, en aan `answers`."},
            "correct": {"en": "Right: every request carries the whole conversation so far, the user's turns and the assistant's answers, so the third question can be answered.",
                        "nl": "Goed: elk request draagt het hele gesprek tot nu toe, de beurten van de gebruiker en de antwoorden van de assistant, zodat de derde vraag te beantwoorden is."},
            "too-few": {"en": "With `{call}`, your function sent fewer requests than there are turns in `said`. One request for each `text`.",
                        "nl": "Met `{call}` stuurde je functie minder requests dan er beurten in `said` zijn. Eén request voor elke `text`."},
            "forgets-user": {"en": "With `{call}`, a request lacked the user's earlier turns. Keep one list `messages` for the whole conversation, made before the loop, and add each `text` to it.",
                             "nl": "Met `{call}` miste een request de eerdere beurten van de gebruiker. Houd één lijst `messages` voor het hele gesprek, gemaakt vóór de loop, en voeg elke `text` eraan toe."},
            "users": {"en": "With `{call}`, the user's messages in your requests are not the turns of `said`, in order.",
                      "nl": "Met `{call}` zijn de berichten van de gebruiker in je requests niet de beurten van `said`, op volgorde."},
            "forgets-answers": {"en": "With `{call}`, your requests lack the assistant's answers. After each answer, add `{{\"role\": \"assistant\", \"content\": answer}}` to `messages`: the model only knows what it said from what you send.",
                                "nl": "Met `{call}` missen je requests de antwoorden van de assistant. Voeg na elk antwoord `{{\"role\": \"assistant\", \"content\": answer}}` toe aan `messages`: het model weet alleen wat het zei uit wat jij meestuurt."},
            "answers": {"en": "With `{call}`, the requests are right, but your function gives back `{got}`: the list of the answers, in order.",
                        "nl": "Met `{call}` kloppen de requests, maar je functie geeft `{got}` terug: de lijst van de antwoorden, op volgorde."},
        },
    ),
}

REQUIRED = ["chat", "talk", "why-all"]

CASES = [
    ("chat", _START_CHAT, "empty", None),
    ("chat", SOLUTION_CHAT, "correct", None),
    ("chat", SOLUTION_CHAT.replace('"messages": messages', '"messages": messages[-1:]'), "wrong", "only-last"),
    ("chat", SOLUTION_CHAT.replace('"max_tokens": max_tokens', '"max_tokens": 200'), "wrong", "max-tokens"),
    ("chat", SOLUTION_CHAT.replace("max_tokens=200", "max_tokens=50"), "wrong", "default"),
    ("chat", SOLUTION_CHAT.replace("max_tokens=200", "max_tokens"), "error", "no-default"),
    ("chat", SOLUTION_CHAT.replace('return answer["choices"][0]["message"]["content"]', 'return answer'), "wrong", "whole"),
    ("chat", SOLUTION_CHAT.replace("json.dumps(body).encode()", "json.dumps(body)"), "error", "data-text"),
    ("talk", _START_TALK, "empty", None),
    ("talk", SOLUTION_TALK, "correct", None),
    ("talk", SOLUTION_TALK.replace('        messages.append({"role": "assistant", "content": answer})\n', ""), "wrong", "forgets-answers"),
    ("talk", SOLUTION_TALK.replace("        messages.append({\"role\": \"user\", \"content\": text})\n        answer = chat(url, messages)",
                                   "        answer = chat(url, [{\"role\": \"user\", \"content\": text}])"), "wrong", "forgets-user"),
    ("talk", SOLUTION_TALK.replace("        answers.append(answer)\n", ""), "wrong", "answers"),
]


# --- the demo: a conversation with Qwen3 through your own client ---------------------------------------------------------

def downloaded() -> bool:
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    chat.download_or_stop(button, texts)


def learner_chat():
    """`chat` from the learner's work/projects/app/client.py, or None when it is not there or does not run."""
    run = run_code("from app.client import chat")
    return run.names.get("chat") if run.error is None else None


def conversation_picture(messages: list, note: str, title: str) -> str:
    """A conversation as it stands: the user's turns in ink, the assistant's in the colour of what a model writes."""
    rows = []
    for m in messages:
        mine = m["role"] == "assistant"
        rows.append(f'<div style="display:flex;justify-content:{"flex-start" if mine else "flex-end"};margin:0.25rem 0">'
                    f'<div style="max-width:80%;padding:0.35rem 0.6rem;border-radius:8px;border:1.5px solid '
                    f'{"var(--ea-pred-text)" if mine else "var(--ea-line)"};background:{"transparent" if mine else "var(--ea-inset)"};'
                    f'font-family:var(--ea-font-code);font-size:13px;white-space:pre-wrap;color:{"var(--ea-pred-text)" if mine else "var(--ea-ink)"}">'
                    f'{html.escape(m["content"])}</div></div>')
    rows.append(f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2);margin-top:0.3rem">{html.escape(note)}</div>')
    return yours.figure(title, "".join(rows), alt=title)


def demo_turn(fn, messages: list, text: str, words: dict, lang: str):
    """One turn with the learner's chat against the course's server: the conversation after it, and the picture."""
    import time
    running = serve.model_server()
    asked = messages + [{"role": "user", "content": text}]
    start = time.perf_counter()
    try:
        answer = fn(running.url, asked)
    except Exception as e:   # the learner's code, against the real server; it may have run out of memory
        said = words["memory"] if serve.out_of_memory(e) else words["failed"].format(why=f"{type(e).__name__}: {e}")
        return messages, f"<p><em>{html.escape(said)}</em></p>"
    after = asked + [{"role": "assistant", "content": str(answer)}]
    seconds = f"{time.perf_counter() - start:.1f}".replace(".", "," if lang == "nl" else ".")
    return after, conversation_picture(after, words["done"].format(seconds=seconds, n=len(asked)), words["title"])

