# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-call-api: the body of a request to a model's API, and a function that sends it with urllib and
reads the answer from its JSON, against a stand-in of the server; the course's recorded call through its own server
(data/course-apis-qwen.json), and the pictures of a request and its answer."""
import html
import json

from everything_ai import apicheck, chat, paths, serve, yours
from everything_ai.apicheck import URL, ApiBox, Case
from everything_ai.check import CodeBox, Mistake

DOWNLOADS = {"Qwen3-0.6B": chat.FILES}   # what the demo downloads, for the line under the header

QUESTION = "Name three colors of the rainbow."
RECORD = {"rainbow": {"messages": [{"role": "user", "content": QUESTION}], "max_tokens": 50, "seeds": 1, "raw": True}}
DATA = paths.ROOT / "data" / "course-apis-qwen.json"


def recorded() -> dict:
    """The course's call: the answer of the server to RECORD["rainbow"], as the JSON it sent."""
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-call-api"]["rainbow"]["answers"][0]


def request_body() -> dict:
    """The body of the course's call."""
    ask = RECORD["rainbow"]
    return {"model": serve.MODEL, "messages": ask["messages"], "max_tokens": ask["max_tokens"]}


# --- the picture of a request and its answer --------------------------------------------------------------------------

_WORDS = {
    "en": {"title": "One request to the API, and its answer", "request": "Your code sends: POST /v1/chat/completions",
           "headers": "Header: Content-Type: application/json", "answer": "The server sends back: status 200",
           "content": "the text of the answer", "usage": "the tokens: in, out and together",
           "finish": "stop: the model ended the answer itself", "body": "the body, as JSON"},
    "nl": {"title": "Eén request aan de API, en het antwoord", "request": "Je code stuurt: POST /v1/chat/completions",
           "headers": "Header: Content-Type: application/json", "answer": "De server stuurt terug: status 200",
           "content": "de tekst van het antwoord", "usage": "de tokens: erin, eruit en samen",
           "finish": "stop: het model beëindigde het antwoord zelf", "body": "de body, als JSON"},
}


def _pre(text: str, marks: dict | None = None, said: str | None = None) -> str:
    """Code-like text; each line that contains a key of `marks` gets that note beside it, in grey, and the line with
    the text the model wrote, `said`, the colour of what a model writes."""
    out = []
    model_line = None if said is None else '"content": ' + json.dumps(said, ensure_ascii=False)
    for line in text.split("\n"):
        note = next((n for k, n in (marks or {}).items() if k in line), "")
        colour = "var(--ea-pred-text)" if model_line is not None and model_line in line else "var(--ea-ink)"
        out.append(f'<span style="color:{colour}">{html.escape(line)}</span>'
                   + (f'<span style="color:var(--ea-ink-2);font-family:var(--ea-font-ui)">   ← {html.escape(note)}</span>' if note else ""))
    return ('<pre style="margin:0.2rem 0 0.7rem;padding:0.55rem 0.7rem;background:var(--ea-inset);border-radius:6px;'
            'font-family:var(--ea-font-code);font-size:12px;line-height:1.45;white-space:pre-wrap;overflow-wrap:anywhere">'
            + "\n".join(out) + "</pre>")


def _label(text: str) -> str:
    return f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)">{html.escape(text)}</div>'


def call_picture(lang: str) -> str:
    """The course's call: the request its code sent, and the JSON the server sent back, with notes at the parts that
    matter."""
    w = _WORDS[lang]
    raw = recorded()["raw"]
    shown = {**raw, "id": raw["id"], "created": raw["created"]}
    body = (_label(w["request"]) + _label(w["headers"]) + _pre(json.dumps(request_body(), ensure_ascii=False, indent=2))
            + _label(w["answer"])
            + _pre(json.dumps(shown, ensure_ascii=False, indent=2), {'"content"': w["content"], '"finish_reason"': w["finish"],
                                                                      '"total_tokens"': w["usage"]},
                   said=raw["choices"][0]["message"]["content"]))
    return yours.figure(w["title"], body, alt=w["title"])


# --- exercise 1: the body ---------------------------------------------------------------------------------------------

_START_BODY = 'question = "Name three colors of the rainbow."\nbody = {}\nbody\n'
_SOLUTION_BODY = ('question = "Name three colors of the rainbow."\n'
                  'body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": question}], "max_tokens": 50}\n'
                  'body\n')


def _messages(got):
    return got.get("messages") if isinstance(got, dict) else None


def body_picture(seen) -> str:
    """The learner's body as the JSON text that goes over the line."""
    run = seen.mine
    if run is None or not isinstance(run.value, dict):
        return ""
    title = {"en": "Your body, as JSON text", "nl": "Je body, als JSON-tekst"}[seen.lang]
    try:
        text = json.dumps(run.value, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return ""
    return yours.figure(title, _pre(text), alt=title)


# --- exercise 2: ask ----------------------------------------------------------------------------------------------------

_START_ASK = '''import json
import urllib.request


def ask(url, question):
    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": question}], "max_tokens": 50}
'''
_SOLUTION_ASK = '''import json
import urllib.request


def ask(url, question):
    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": question}], "max_tokens": 50}
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        answer = json.loads(response.read())
    return answer["choices"][0]["message"]["content"]
'''
ASKED = {"Name three colors of the rainbow.": "Red, orange and yellow.", "What is 2 + 2?": "2 + 2 is 4."}


def _judge(question):
    want = ASKED[question]

    def judge(got, running):
        if got == want:
            return None
        if isinstance(got, dict) and "choices" in got:
            return "whole"
        if isinstance(got, dict) and "content" in got:
            return "message"
        if isinstance(got, bytes) or (isinstance(got, str) and got.lstrip().startswith("{")):
            return "text"
        if got is None:
            return "none"
        return "other"
    return judge


def _stand_in():
    return serve.StandIn(ASKED)


ASK_CASES = [Case(_stand_in, (URL, q), _judge(q)) for q in ASKED]


def ask_picture(seen) -> str:
    words = {"en": {"title": "What your ask sent, and what came back", "request": "Your code sent: {method} {path}",
                    "headers": "Headers:", "answer": "The server sent back: status {status}",
                    "stream": "a stream with {n} pieces", "none": "Your function sent nothing to the server yet."},
             "nl": {"title": "Wat je ask stuurde, en wat er terugkwam", "request": "Je code stuurde: {method} {path}",
                    "headers": "Headers:", "answer": "De server stuurde terug: status {status}",
                    "stream": "een stream met {n} stukjes", "none": "Je functie stuurde nog niets naar de server."}}[seen.lang]
    return apicheck.exchange(seen, "ask", ASK_CASES[0], words)


def _not_bytes(e) -> bool:
    return isinstance(e, TypeError) and "bytes" in str(e)


def _list_by_name(e) -> bool:
    return isinstance(e, TypeError) and "list indices must be integers" in str(e)


EXERCISES = {
    "body": CodeBox(
        name="body",
        picture=body_picture,
        start=_START_BODY,
        solution=_SOLUTION_BODY,
        others=[{"question": "What is 2 + 2?"}],
        keep=['question = "Name three colors of the rainbow."'],
        mistakes=[
            Mistake("key-message", matches=lambda got, text: isinstance(got, dict) and "message" in got and "messages" not in got,
                    en="Your body has a key `\"message\"`; the API wants `\"messages\"`, with an s: a list, also when it holds one message.",
                    nl="Je body heeft een key `\"message\"`; de API wil `\"messages\"`, met een s: een lijst, ook als er één bericht in zit."),
            Mistake("messages-text", matches=lambda got, text: isinstance(_messages(got), str),
                    en="`\"messages\"` is a string now. The API wants a list of messages, each a dictionary with `\"role\"` and `\"content\"`, as in a conversation in [Base model or chat model?](/course/open/lm-base-vs-chat?lang=en).",
                    nl="`\"messages\"` is nu een string. De API wil een lijst berichten, elk een dictionary met `\"role\"` en `\"content\"`, zoals een gesprek in [Base model of chat model?](/course/open/lm-base-vs-chat?lang=nl)."),
            Mistake("no-list", matches=lambda got, text: isinstance(_messages(got), dict),
                    en="`\"messages\"` is one dictionary now. Put it in a list, between `[` and `]`: the API always takes a list, also for one message.",
                    nl="`\"messages\"` is nu één dictionary. Zet hem in een lijst, tussen `[` en `]`: de API neemt altijd een lijst, ook voor één bericht."),
            Mistake("role", matches=lambda got, text: isinstance(_messages(got), list) and _messages(got)[:1]
                    and isinstance(_messages(got)[0], dict) and _messages(got)[0].get("role") != "user",
                    en="Your message has the role `{got_role}`. A question from you, the person, has the role `\"user\"`.",
                    nl="Je bericht heeft de rol `{got_role}`. Een vraag van jou, de persoon, heeft de rol `\"user\"`.",
                    detail=lambda got, text: {"got_role": repr(_messages(got)[0].get("role"))}),
            Mistake("max-tokens-text", matches=lambda got, text: isinstance(got, dict) and isinstance(got.get("max_tokens"), str),
                    en="`\"max_tokens\"` is the string `\"50\"` now; the API wants the number `50`, without quotes.",
                    nl="`\"max_tokens\"` is nu de string `\"50\"`; de API wil het getal `50`, zonder aanhalingstekens."),
            Mistake("no-model", matches=lambda got, text: isinstance(got, dict) and "model" not in got and "messages" in got,
                    en="Almost: the body says which messages and how many tokens, but not which model. Add `\"model\": \"qwen3-0.6b\"`: a server can have several.",
                    nl="Bijna: de body zegt welke berichten en hoeveel tokens, maar niet welk model. Voeg `\"model\": \"qwen3-0.6b\"` toe: een server kan er meer hebben.",
                    state="almost"),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "`body` is still an empty dictionary. Give it three keys: `\"model\"`, `\"messages\"` and `\"max_tokens\"`.",
                      "nl": "`body` is nog een lege dictionary. Geef hem drie keys: `\"model\"`, `\"messages\"` en `\"max_tokens\"`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
                        "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."},
            "correct": {"en": "Right: `{got}`. `json.dumps(body)` turns it into the text that goes to the server.",
                        "nl": "Goed: `{got}`. `json.dumps(body)` maakt er de tekst van die naar de server gaat."},
            "wrong": {"en": "Your box gives `{got}`. Look at the body in the picture above: which key, or which value, is different?",
                      "nl": "Je vak geeft `{got}`. Kijk naar de body in het plaatje hierboven: welke key, of welke waarde, is anders?"},
            "no_value": {"en": "Your box gives nothing back. Keep `body` as the last line, on its own.",
                         "nl": "Je vak geeft niets terug. Laat `body` als laatste regel staan, los."},
        },
    ),
    "ask": ApiBox(
        name="ask",
        picture=ask_picture,
        function="ask",
        start=_START_ASK,
        solution=_SOLUTION_ASK,
        cases=ASK_CASES,
        mistakes=[
            Mistake("data-text", error=_not_bytes,
                    en="urllib sends bytes, not text: `json.dumps(body)` is text. Turn it into bytes with `.encode()`, as in the recipe.",
                    nl="urllib verstuurt bytes, geen tekst: `json.dumps(body)` is tekst. Maak er bytes van met `.encode()`, zoals in het recept."),
            Mistake("list-by-name", error=_list_by_name,
                    en="With `{call}`, `\"choices\"` gives a list, and a list takes a number between the brackets, not a name. Take its first element with `[0]`, then `[\"message\"]`.",
                    nl="Met `{call}` geeft `\"choices\"` een lijst, en een lijst neemt een getal tussen de haken, geen naam. Neem het eerste element met `[0]`, en dan `[\"message\"]`."),
            Mistake("key", error=lambda e: isinstance(e, KeyError),
                    en="With `{call}`, a key you ask for is not in the answer. Follow the JSON in the picture from the outside in: `[\"choices\"]`, `[0]`, `[\"message\"]`, `[\"content\"]`.",
                    nl="Met `{call}` staat een key waar je om vraagt niet in het antwoord. Volg de JSON in het plaatje van buiten naar binnen: `[\"choices\"]`, `[0]`, `[\"message\"]`, `[\"content\"]`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "`ask` makes the body, and sends nothing yet. Under it: turn the body into bytes, make a `Request` to `url + \"/chat/completions\"`, open it, and return the text of the answer.",
                      "nl": "`ask` maakt de body, en verstuurt nog niets. Daaronder: maak er bytes van, maak een `Request` naar `url + \"/chat/completions\"`, open hem, en geef de tekst van het antwoord terug."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
                        "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."},
            "http-405": {"en": "With `{call}`, your request reached the server as a GET, and the server said: \"{message}\" A `Request` is a POST only when it gets `data=`.",
                         "nl": "Met `{call}` kwam je request als GET bij de server aan, en de server zei: \"{message}\" Een `Request` is pas een POST als hij `data=` krijgt."},
            "http-404": {"en": "With `{call}`, the server answered 404: \"{message}\" The endpoint is `url + \"/chat/completions\"`: `url` already ends in `/v1`.",
                         "nl": "Met `{call}` antwoordde de server 404: \"{message}\" Het endpoint is `url + \"/chat/completions\"`: `url` eindigt al op `/v1`."},
            "http-404-model": {"en": "With `{call}`, the server answered 404: \"{message}\" Keep the `model` in the body as the box had it at the start.",
                               "nl": "Met `{call}` antwoordde de server 404: \"{message}\" Laat het `model` in de body staan zoals het aan het begin in het vak stond."},
            "correct": {"en": "Right: your `ask` sends the question, and gives back only the text of the answer, for both questions the check asked.",
                        "nl": "Goed: je `ask` stuurt de vraag, en geeft alleen de tekst van het antwoord terug, voor beide vragen die de check stelde."},
            "whole": {"en": "With `{call}`, your function gives back the whole JSON of the answer. Give back only the text: `[\"choices\"][0][\"message\"][\"content\"]`.",
                      "nl": "Met `{call}` geeft je functie de hele JSON van het antwoord terug. Geef alleen de tekst terug: `[\"choices\"][0][\"message\"][\"content\"]`."},
            "message": {"en": "With `{call}`, your function gives back the message, `{got}`. One step further in: its `[\"content\"]`.",
                        "nl": "Met `{call}` geeft je functie het bericht terug, `{got}`. Nog één stap verder naar binnen: de `[\"content\"]` ervan."},
            "text": {"en": "With `{call}`, your function gives back the answer as JSON text, not as Python. Read it with `json.loads`, then take the text out.",
                     "nl": "Met `{call}` geeft je functie het antwoord terug als JSON-tekst, niet als Python. Lees het met `json.loads`, en haal er dan de tekst uit."},
            "none": {"en": "With `{call}`, your function gives nothing back. End it with `return` and the text of the answer.",
                     "nl": "Met `{call}` geeft je functie niets terug. Eindig met `return` en de tekst van het antwoord."},
            "other": {"en": "With `{call}`, your function gives `{got}`. The stand-in of the server answered something else: do you send the `question` the function gets?",
                      "nl": "Met `{call}` geeft je functie `{got}`. De stand-in van de server antwoordde iets anders: stuur je de `question` die de functie krijgt?"},
        },
    ),
}

REQUIRED = ["body", "ask", "finish"]

CASES = [
    ("body", _START_BODY, "empty", None),
    ("body", "", "empty", None),
    ("body", _SOLUTION_BODY, "correct", None),
    ("body", _SOLUTION_BODY.replace('"max_tokens": 50', '"max_tokens": "50"'), "wrong", "max-tokens-text"),
    ("body", _SOLUTION_BODY.replace('[{"role": "user", "content": question}]', '{"role": "user", "content": question}'), "wrong", "no-list"),
    ("body", _SOLUTION_BODY.replace('[{"role": "user", "content": question}]', 'question'), "wrong", "messages-text"),
    ("body", _SOLUTION_BODY.replace('"messages"', '"message"'), "wrong", "key-message"),
    ("body", _SOLUTION_BODY.replace('"role": "user"', '"role": "human"'), "wrong", "role"),
    ("body", _SOLUTION_BODY.replace('"model": "qwen3-0.6b", ', ''), "almost", "no-model"),
    ("body", _SOLUTION_BODY.replace('"content": question', '"content": "Name three colors of the rainbow."'), "almost", "other-value"),
    ("ask", _START_ASK, "empty", None),
    ("ask", _SOLUTION_ASK, "correct", None),
    ("ask", _SOLUTION_ASK.replace('"model": "qwen3-0.6b"', '"model": "gpt-4o"'), "error", "http-404-model"),
    ("ask", _SOLUTION_ASK.replace("json.dumps(body).encode()", "json.dumps(body)"), "error", "data-text"),
    ("ask", _SOLUTION_ASK.replace(", data=data,", ","), "error", "http-405"),
    ("ask", _SOLUTION_ASK.replace('url + "/chat/completions"', 'url + "/v1/chat/completions"'), "error", "http-404"),
    ("ask", _SOLUTION_ASK.replace('return answer["choices"][0]["message"]["content"]', 'return answer'), "wrong", "whole"),
    ("ask", _SOLUTION_ASK.replace('return answer["choices"][0]["message"]["content"]', 'return answer["choices"][0]["message"]'), "wrong", "message"),
    ("ask", _SOLUTION_ASK.replace('answer["choices"][0]["message"]', 'answer["choices"]["message"]'), "error", "list-by-name"),
    ("ask", _SOLUTION_ASK.replace('answer["choices"][0]', 'answer["choice"][0]'), "error", "key"),
    ("ask", _SOLUTION_ASK.replace("answer = json.loads(response.read())\n    return answer[\"choices\"][0][\"message\"][\"content\"]",
                                  "return response.read()"), "wrong", "text"),
    ("ask", _SOLUTION_ASK.replace('return answer["choices"][0]["message"]["content"]', 'answer["choices"][0]["message"]["content"]'), "wrong", "none"),
    ("ask", _SOLUTION_ASK.replace('"content": question', '"content": "What is 2 + 2?"'), "wrong", "other"),
]


# --- the demo: your ask, against the course's own server with Qwen3 ----------------------------------------------------

def downloaded() -> bool:
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    chat.download_or_stop(button, texts)


def learner_ask(text):
    """The learner's ask from exercise 2, when it is right; else None."""
    from everything_ai.check import MISSING
    if text is MISSING or text is None or EXERCISES["ask"].check(text).state != "correct":
        return None
    from everything_ai.check import run_code
    return run_code(text).names.get("ask")


def demo(ask, question: str, words: dict, lang: str) -> str:
    """The learner's ask, sent to the course's server with Qwen3 behind it: the answer, and how long it took."""
    import time
    running = serve.model_server()
    start = time.perf_counter()
    try:
        text = ask(running.url, question)
    except Exception as e:   # the learner's code, against the real server; it may have run out of memory
        said = words["memory"] if serve.out_of_memory(e) else words["failed"].format(why=f"{type(e).__name__}: {e}")
        return f"<p><em>{html.escape(said)}</em></p>"
    seconds = f"{time.perf_counter() - start:.1f}".replace(".", "," if lang == "nl" else ".")
    return chat.answer_picture(str(text), words["done"].format(seconds=seconds, url=running.url), words["title"])
