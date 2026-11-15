# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-system-prompt: a function that puts a system prompt at the start of a conversation, in place of
the one it had, and one that sends a response prefill and gives back the whole JSON text, against a stand-in of the
server; the course's recorded runs (data/course-dev-prompts-qwen.json): a question with and without a system prompt,
a bike shop's system prompt against two questions, JSON with and without a response prefill, and the worked example;
their pictures, and the demo."""
import copy
import html
import json
import time

from everything_ai import apicheck, chat, paths, serve, yours
from everything_ai.apicheck import URL, ApiBox, Case
from everything_ai.check import FunctionBox, Mistake, run_code, with_time_limit

DOWNLOADS = {"Qwen3-0.6B": chat.FILES}   # what the demo downloads, for the line under the header

QUESTION = "What is a neural network?"
BRIEF = "Answer in one sentence of at most 20 words."
SHOP = ("You are the assistant of a bike shop. Answer only questions about bikes. "
        "To any other question, answer: Sorry, I can only help with bikes.")
BIKE = "How often should I pump up my tires?"
POEM = "Write a short poem about the sea."
CITY = "Give the capital of France as JSON, with the key city."
START = '{"city": "'
TITLE = "Make a title of at most five words for the text the user sends."
STORY = "Yesterday it rained all day, so we stayed in, baked bread and played cards."
TITLE_START = '{"title": "'
RECORD = {
    "plain": {"messages": [{"role": "user", "content": QUESTION}], "max_tokens": 200, "seeds": 4},
    "brief": {"messages": [{"role": "system", "content": BRIEF}, {"role": "user", "content": QUESTION}], "max_tokens": 200, "seeds": 4},
    "shop-bike": {"messages": [{"role": "system", "content": SHOP}, {"role": "user", "content": BIKE}], "max_tokens": 80, "seeds": 4},
    "shop-poem": {"messages": [{"role": "system", "content": SHOP}, {"role": "user", "content": POEM}], "max_tokens": 80, "seeds": 4},
    "json": {"messages": [{"role": "user", "content": CITY}], "max_tokens": 40, "seeds": 4},
    "json-prefill": {"messages": [{"role": "user", "content": CITY}, {"role": "assistant", "content": START}], "max_tokens": 40, "seeds": 4},
    "worked": {"messages": [{"role": "system", "content": TITLE}, {"role": "user", "content": STORY},
                            {"role": "assistant", "content": TITLE_START}], "max_tokens": 30, "seeds": 1},
}
DATA = paths.ROOT / "data" / "course-dev-prompts-qwen.json"


def course() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-system-prompt"]


def answers(key: str) -> list:
    return course()[key]["answers"]


def refused(text: str) -> bool:
    """An answer that is the refusal the bike shop's system prompt asks for."""
    return "only help with bikes" in text


def json_state(text: str, key: str = "city") -> str:
    """What json.loads makes of an answer: "usable" (JSON with the key), "key" (JSON without it) or "not-json"."""
    try:
        value = json.loads(text)
    except ValueError:
        return "not-json"
    return "usable" if isinstance(value, dict) and key in value else "key"


def findings() -> dict:
    """What the recorded runs show: the tokens and endings with and without the short system prompt, its words and
    the prompt tokens it adds, how often the bike shop's model refused each question, and how many JSON answers were
    usable with and without the response prefill."""
    plain, brief = answers("plain"), answers("brief")
    out = {
        "plain_tokens": [a["tokens"] for a in plain], "plain_finish": {a["finish"] for a in plain},
        "brief_tokens": [a["tokens"] for a in brief], "brief_finish": {a["finish"] for a in brief},
        "brief_words": [len(a["text"].split()) for a in brief],
        "prompt_tokens": (plain[0]["prompt_tokens"], brief[0]["prompt_tokens"]),
        "bike_refused": sum(refused(a["text"]) for a in answers("shop-bike")),
        "poem_refused": sum(refused(a["text"]) for a in answers("shop-poem")),
        "runs": len(plain),
    }
    out["json"] = [json_state(a["text"]) for a in answers("json")]
    out["fenced"] = sum(a["text"].startswith("```") for a in answers("json"))
    out["capital"] = sum('"capital"' in a["text"] for a in answers("json"))
    out["prefill"] = [json_state(START + a["text"]) for a in answers("json-prefill")]
    worked = answers("worked")[0]
    out["worked"] = {"text": worked["text"], "joined": TITLE_START + worked["text"], "finish": worked["finish"]}
    return out


# --- the pictures of the recorded runs -------------------------------------------------------------------------------

_CELL = "padding:0.18rem 0.5rem;border-bottom:1px solid var(--ea-line);vertical-align:top;font-size:13px"
_HEAD = f"{_CELL};color:var(--ea-ink-2);font-weight:500;text-align:left;font-family:var(--ea-font-ui)"


def _label(text: str) -> str:
    return f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2);margin:0.2rem 0">{html.escape(text)}</div>'


def _short(text: str, n: int = 90) -> str:
    one = " ".join(text.split())
    return one if len(one) <= n else one[:n - 1].rstrip() + "…"


def _model(text: str) -> str:
    """Text the model wrote, with a line break shown as ↵."""
    return (f'<span style="color:var(--ea-pred-text);font-family:var(--ea-font-code);white-space:pre-wrap">'
            + html.escape(text).replace("\n", '<span style="color:var(--ea-ink-3)">↵</span>\n') + "</span>")


def _mine(text: str) -> str:
    """Text the request wrote: a prefill."""
    return f'<span style="color:var(--ea-ink);font-family:var(--ea-font-code);white-space:pre-wrap;font-weight:600">{html.escape(text)}</span>'


_BRIEF_WORDS = {
    "en": {"title": "The same question, four times without and four times with a system prompt",
           "question": "The question: " + QUESTION, "plain": "no system prompt", "brief": "system prompt: " + BRIEF,
           "seed": "run", "tokens": "tokens written", "finish": "finish_reason", "answer": "how the answer starts"},
    "nl": {"title": "Dezelfde vraag, vier keer zonder en vier keer met een system prompt",
           "question": "De vraag: " + QUESTION, "plain": "geen system prompt", "brief": "system prompt: " + BRIEF,
           "seed": "run", "tokens": "geschreven tokens", "finish": "finish_reason", "answer": "hoe het antwoord begint"},
}


def brief_picture(lang: str) -> str:
    """The four answers to the question without a system prompt and the four with the short one: for each, a bar for
    the tokens it wrote (out of max_tokens 200), its finish_reason, and how it starts."""
    w = _BRIEF_WORDS[lang]
    top = RECORD["plain"]["max_tokens"]
    rows = []
    for key in ("plain", "brief"):
        rows.append(f'<tr><td colspan="4" style="{_CELL};font-family:var(--ea-font-ui);color:var(--ea-ink);font-weight:600;padding-top:0.5rem">'
                    f'{html.escape(w[key])}</td></tr>')
        for i, a in enumerate(answers(key)):
            bar = (f'<div style="display:flex;align-items:center;gap:0.4rem"><div style="height:10px;width:{100 * a["tokens"] / top:.1f}%;'
                   f'min-width:2px;max-width:7rem;background:var(--ea-pred)"></div>'
                   f'<span style="font-family:var(--ea-font-ui);color:var(--ea-ink)">{a["tokens"]}</span></div>')
            rows.append(f'<tr><td style="{_CELL};color:var(--ea-ink-2);font-family:var(--ea-font-ui)">{i + 1}</td>'
                        f'<td style="{_CELL};width:8rem">{bar}</td>'
                        f'<td style="{_CELL};font-family:var(--ea-font-code);color:var(--ea-ink)">"{a["finish"]}"</td>'
                        f'<td style="{_CELL}">{_model(_short(a["text"], 70))}</td></tr>')
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%"><thead><tr>'
             f'<th style="{_HEAD}">{w["seed"]}</th><th style="{_HEAD}">{w["tokens"]}</th><th style="{_HEAD}">{w["finish"]}</th>'
             f'<th style="{_HEAD}">{w["answer"]}</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    return yours.figure(w["title"], _label(w["question"]) + table, alt=w["title"])


_SHOP_WORDS = {
    "en": {"title": "A bike shop's system prompt, against two questions", "system": "The system prompt: " + SHOP,
           "seed": "run", "kept": "followed", "broke": "not followed"},
    "nl": {"title": "De system prompt van een fietsenwinkel, tegen twee vragen", "system": "De system prompt: " + SHOP,
           "seed": "run", "kept": "gevolgd", "broke": "niet gevolgd"},
}


def shop_picture(lang: str) -> str:
    """The bike shop's four answers to a question about bikes and four to a poem: for each, how it starts, and whether
    it did what the system prompt says (answer about bikes, refuse anything else). Not followed is marked as a mistake."""
    w = _SHOP_WORDS[lang]

    def tag(kept: bool) -> str:
        color = "var(--ea-ink-2)" if kept else "var(--ea-loss-text)"
        return f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:{color};font-weight:600">{w["kept"] if kept else w["broke"]}</div>'
    rows = []
    for i, (bike, poem) in enumerate(zip(answers("shop-bike"), answers("shop-poem"))):
        rows.append(f'<tr><td style="{_CELL};color:var(--ea-ink-2);font-family:var(--ea-font-ui)">{i + 1}</td>'
                    f'<td style="{_CELL}">{_model(_short(bike["text"], 64))}{tag(not refused(bike["text"]))}</td>'
                    f'<td style="{_CELL}">{_model(_short(poem["text"], 64))}{tag(refused(poem["text"]))}</td></tr>')
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%"><thead><tr>'
             f'<th style="{_HEAD}">{w["seed"]}</th><th style="{_HEAD}">{html.escape(BIKE)}</th><th style="{_HEAD}">{html.escape(POEM)}</th>'
             f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    return yours.figure(w["title"], _label(w["system"]) + table, alt=w["title"])


_PROMPT_WORDS = {
    "en": {"title": "What the model gets, with a response prefill", "yours": "your prefill", "model": "what the model wrote after it",
           "note": "Simple markers, as in the lesson about base models and chat models; Qwen3's own markers look different, and work the same way."},
    "nl": {"title": "Wat het model krijgt, met een response prefill", "yours": "je prefill", "model": "wat het model erna schreef",
           "note": "Eenvoudige markeringen, zoals in de les over base models en chat models; die van Qwen3 zelf zien er anders uit, en werken net zo."},
}


def prompt_picture(lang: str) -> str:
    """The conversation as the one text the model gets, with simple markers: the question, the assistant's marker, the
    prefill (bold) and what Qwen3 wrote after it (in the colour of model output), the first recorded run."""
    w = _PROMPT_WORDS[lang]
    written = answers("json-prefill")[0]["text"]
    marker = 'color:var(--ea-ink-2);font-family:var(--ea-font-code)'
    text = (f'<div style="font-family:var(--ea-font-code);font-size:13px;line-height:1.6;padding:0.4rem 0.6rem;background:var(--ea-inset);border-radius:4px">'
            f'<div style="{marker}">USER:</div><div style="color:var(--ea-ink)">{html.escape(CITY)}</div><div>&nbsp;</div>'
            f'<div style="{marker}">ASSISTANT:</div><div>{_mine(START)}{_model(written)}</div></div>')
    legend = (f'<div style="font-family:var(--ea-font-ui);font-size:12px;margin-top:0.3rem">{_mine("{…")} '
              f'<span style="color:var(--ea-ink-2)">{w["yours"]}</span> &nbsp; {_model("…}")} '
              f'<span style="color:var(--ea-ink-2)">{w["model"]}</span></div>')
    return yours.figure(w["title"], text + legend + _label(w["note"]), alt=w["title"])


_JSON_WORDS = {
    "en": {"title": "JSON with the key city: four times without and four times with a response prefill",
           "question": "The question: " + CITY, "without": "without prefill", "with": 'with the prefill {"city": "',
           "usable": "json.loads: city is in it", "key": "json.loads: no key city", "not-json": "json.loads: not JSON",
           "count": "usable: {n} of {k}", "seed": "run"},
    "nl": {"title": "JSON met de key city: vier keer zonder en vier keer met een response prefill",
           "question": "De vraag: " + CITY, "without": "zonder prefill", "with": 'met de prefill {"city": "',
           "usable": "json.loads: city staat erin", "key": "json.loads: geen key city", "not-json": "json.loads: geen JSON",
           "count": "bruikbaar: {n} van {k}", "seed": "run"},
}


def json_picture(lang: str) -> str:
    """The four JSON answers without a prefill and the four with one (the prefill bold, joined with what the model
    wrote): for each, what json.loads made of it, and under each column how many were usable."""
    w = _JSON_WORDS[lang]

    def outcome(state: str) -> str:
        color = "var(--ea-ink-2)" if state == "usable" else "var(--ea-loss-text)"
        return f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:{color};font-weight:600">{w[state]}</div>'
    plain, filled = answers("json"), answers("json-prefill")
    rows = []
    for i, (a, b) in enumerate(zip(plain, filled)):
        rows.append(f'<tr><td style="{_CELL};color:var(--ea-ink-2);font-family:var(--ea-font-ui)">{i + 1}</td>'
                    f'<td style="{_CELL}">{_model(a["text"])}{outcome(json_state(a["text"]))}</td>'
                    f'<td style="{_CELL}">{_mine(START)}{_model(b["text"])}{outcome(json_state(START + b["text"]))}</td></tr>')
    counts = [sum(json_state(t) == "usable" for t in texts)
              for texts in ([a["text"] for a in plain], [START + b["text"] for b in filled])]
    rows.append(f'<tr><td style="{_CELL};border-bottom:none"></td>'
                + "".join(f'<td style="{_CELL};border-bottom:none;font-family:var(--ea-font-ui);color:var(--ea-ink);font-weight:600">'
                          f'{w["count"].format(n=n, k=len(plain))}</td>' for n in counts) + "</tr>")
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%"><thead><tr>'
             f'<th style="{_HEAD}">{w["seed"]}</th><th style="{_HEAD}">{w["without"]}</th><th style="{_HEAD}">{html.escape(w["with"])}</th>'
             f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    return yours.figure(w["title"], _label(w["question"]) + table, alt=w["title"])


# --- exercise 1: with_system ---------------------------------------------------------------------------------------------

START_WITH = '''def with_system(messages, system):
    new = []
    for message in messages:
        new.append(message)
    return new
'''
SOLUTION_WITH = '''def with_system(messages, system):
    new = [{"role": "system", "content": system}]
    for message in messages:
        if message["role"] != "system":
            new.append(message)
    return new
'''
TALK = [{"role": "user", "content": "Hi, I am Sam."}, {"role": "assistant", "content": "Hello Sam!"},
        {"role": "user", "content": QUESTION}]
FRENCH = [{"role": "system", "content": "Answer in French."}] + TALK
WITH_CASES = [(TALK, BRIEF), (FRENCH, BRIEF), ([], BRIEF)]


def _changes_its_list(_, text) -> bool:
    """The function gives the right list, but changes the list it got."""
    fn = run_code(text).names.get("with_system")
    if not callable(fn):
        return False
    talk = copy.deepcopy(FRENCH)
    try:
        with_time_limit(lambda: fn(talk, BRIEF))
    except Exception:   # the learner's code; the check itself already said what is wrong with it
        return False
    return talk != FRENCH


_CONV_WORDS = {
    "en": {"title": "What your with_system made of a conversation with a system prompt", "got": "with_system got, with the system prompt \"{s}\":",
           "made": "Your function gave back:", "want": "It should give back:", "none": "nothing yet", "same": "Right: the list it should give."},
    "nl": {"title": "Wat je with_system maakte van een gesprek met een system prompt", "got": "with_system kreeg, met de system prompt \"{s}\":",
           "made": "Je functie gaf terug:", "want": "Hij zou moeten teruggeven:", "none": "nog niets", "same": "Goed: de lijst die hij moet geven."},
}


def _messages_html(messages, want=None) -> str:
    """A list of messages, one row each: the role, then the content; a row that differs from the same row of `want`
    is marked."""
    if not isinstance(messages, list):
        return f'<div style="font-family:var(--ea-font-code);font-size:13px;color:var(--ea-ink)">{html.escape(repr(messages))}</div>'
    rows = []
    for i, m in enumerate(messages):
        off = want is not None and (i >= len(want) or m != want[i])
        edge = "var(--ea-loss)" if off else "var(--ea-line)"
        if isinstance(m, dict) and set(m) == {"role", "content"}:
            inner = (f'<span style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2);display:inline-block;min-width:5.5rem">{html.escape(str(m["role"]))}</span>'
                     f'<span style="font-family:var(--ea-font-code);font-size:13px;color:var(--ea-ink)">{html.escape(str(m["content"]))}</span>')
        else:
            inner = f'<span style="font-family:var(--ea-font-code);font-size:13px;color:var(--ea-ink)">{html.escape(repr(m))}</span>'
        rows.append(f'<div style="border-left:3px solid {edge};padding:0.1rem 0.5rem;margin:0.15rem 0">{inner}</div>')
    if want is not None and len(want) > len(messages):
        rows.append('<div style="border-left:3px solid var(--ea-loss);padding:0.1rem 0.5rem;margin:0.15rem 0">&nbsp;</div>')
    return "".join(rows) or f'<div style="font-family:var(--ea-font-code);font-size:13px;color:var(--ea-ink)">[]</div>'


def conversation_picture(seen) -> str:
    """The conversation that already has a system prompt, what the learner's with_system made of it, and when that
    differs, what it should make, with the rows that differ marked."""
    w = _CONV_WORDS[seen.lang]
    conversation, system = WITH_CASES[1]
    mine = seen.mine
    fn = mine.names.get("with_system") if mine is not None else None
    if not callable(fn):
        return ""
    want = seen.remember("conversation", lambda: run_code(SOLUTION_WITH).names["with_system"](copy.deepcopy(conversation), system))
    try:
        got = with_time_limit(lambda: fn(copy.deepcopy(conversation), system))
    except Exception:   # the learner's code; the check says what went wrong
        return ""
    body = _label(w["got"].format(s=system)) + _messages_html(conversation) + _label(w["made"]) + _messages_html(got, want)
    body += _label(w["same"]) if got == want else _label(w["want"]) + _messages_html(want)
    return yours.figure(w["title"], body, alt=w["title"])


# --- exercise 2: ask_json with a response prefill --------------------------------------------------------------------------

_CLIENT = '''    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        answer = json.loads(response.read())
'''
START_JSON = ('import json\nimport urllib.request\n\n\ndef ask_json(url, question, start):\n'
              '    messages = [{"role": "user", "content": question}]\n'
              '    body = {"model": "qwen3-0.6b", "messages": messages, "max_tokens": 40}\n'
              + _CLIENT + '    return answer["choices"][0]["message"]["content"]\n')
SOLUTION_JSON = (START_JSON.replace('    messages = [{"role": "user", "content": question}]\n',
                                    '    messages = [{"role": "user", "content": question},\n'
                                    '                {"role": "assistant", "content": start}]\n')
                 .replace('    return answer["choices"][0]["message"]["content"]\n',
                          '    return start + answer["choices"][0]["message"]["content"]\n'))
PLANET = "Give the largest planet as JSON, with the key planet."
JSONS = {CITY: '{"city": "Paris"}', PLANET: '{"planet": "Jupiter"}'}


def _sent(running) -> list:
    return [e["body"] for e in running.log if e["method"] == "POST" and isinstance(e["body"], dict)]


def _judge_json(question, start):
    want = JSONS[question]

    def judge(got, running):
        sent = _sent(running)
        messages = (sent[-1] if sent else {}).get("messages") or []
        messages = [m for m in messages if isinstance(m, dict)]
        last = messages[-1] if messages else {}
        if last.get("content") == start and last.get("role") != "assistant":
            return "prefill-role"
        if "assistant" not in [m.get("role") for m in messages]:
            return "no-prefill"
        if last.get("role") != "assistant":
            return "prefill-first"
        if last.get("content") != start:
            return "other-start"
        if isinstance(got, dict):
            return "whole" if "choices" in got else "parsed"
        if got == want[len(start):]:
            return "only-rest"
        return None if got == want else "other"
    return judge


JSON_CASES = [Case(lambda: serve.StandIn(JSONS), (URL, CITY, START), _judge_json(CITY, START)),
              Case(lambda: serve.StandIn(JSONS), (URL, PLANET, '{"planet": "'), _judge_json(PLANET, '{"planet": "'))]

_EXCHANGE = {"en": {"title": "What your ask_json sent, and what came back", "request": "Your code sent: {method} {path}",
                    "headers": "Headers:", "answer": "The server sent back: status {status}",
                    "stream": "a stream with {n} pieces", "none": "Your function sent nothing to the server yet."},
             "nl": {"title": "Wat je ask_json stuurde, en wat er terugkwam", "request": "Je code stuurde: {method} {path}",
                    "headers": "Headers:", "answer": "De server stuurde terug: status {status}",
                    "stream": "een stream met {n} stukjes", "none": "Je functie stuurde nog niets naar de server."}}


def json_exchange_picture(seen) -> str:
    return apicheck.exchange(seen, "ask_json", JSON_CASES[0], _EXCHANGE[seen.lang])


def _not_bytes(e) -> bool:
    return isinstance(e, TypeError) and "bytes" in str(e)


_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}

EXERCISES = {
    "with_system": FunctionBox(
        name="with_system",
        picture=conversation_picture,
        function="with_system",
        start=START_WITH,
        solution=SOLUTION_WITH,
        visible=[WITH_CASES[0]],
        hidden=WITH_CASES[1:],
        mistakes=[
            Mistake("at-end", wrong=SOLUTION_WITH.replace('    new = [{"role": "system", "content": system}]\n', "    new = []\n")
                    .replace("    return new\n", '    new.append({"role": "system", "content": system})\n    return new\n'),
                    en="`{call}` gives the system prompt as the last message. The system prompt goes first: the model reads the conversation from the start, and the instructions hold for all that follows.",
                    nl="`{call}` geeft de system prompt als laatste bericht. De system prompt komt eerst: het model leest het gesprek vanaf het begin, en de instructies gelden voor alles wat volgt."),
            Mistake("kept-old", state="almost", wrong=START_WITH.replace("    new = []\n", '    new = [{"role": "system", "content": system}]\n'),
                    en="Right for the example. But `{call}` gives `{got}`: two system prompts, the new one and the old one, which say different things. Leave out the messages of the conversation that have the role `system`.",
                    nl="Goed voor het voorbeeld. Maar `{call}` geeft `{got}`: twee system prompts, de nieuwe en de oude, die iets anders zeggen. Laat de berichten van het gesprek met de rol `system` weg."),
            Mistake("users-only", wrong=SOLUTION_WITH.replace('message["role"] != "system"', 'message["role"] == "user"'),
                    en="`{call}` leaves out the answers of the assistant too. The model needs them: it keeps nothing itself, so the conversation must go in whole. Leave out only the messages with the role `system`.",
                    nl="`{call}` laat ook de antwoorden van de assistant weg. Het model heeft ze nodig: het onthoudt zelf niets, dus het gesprek moet er helemaal in. Laat alleen de berichten met de rol `system` weg."),
            Mistake("only-system", wrong='def with_system(messages, system):\n    return [{"role": "system", "content": system}]\n',
                    en="`{call}` gives only the system prompt, without the conversation. Keep every message of the conversation after it, except an old system prompt.",
                    nl="`{call}` geeft alleen de system prompt, zonder het gesprek. Houd elk bericht van het gesprek erachter, behalve een oude system prompt."),
            Mistake("text-key", wrong=SOLUTION_WITH.replace('{"role": "system", "content": system}', '{"role": "system", "text": system}'),
                    en="`{call}` gives a message with the role `system` and the key `\"text\"`. Every message has the keys `\"role\"` and `\"content\"`, the system prompt too.",
                    nl="`{call}` geeft een bericht met de rol `system` en de key `\"text\"`. Elk bericht heeft de keys `\"role\"` en `\"content\"`, de system prompt ook."),
        ],
        form=[Mistake("changed", state="almost", matches=_changes_its_list,
                      en="Your function gives the right list, but it also changed the list it got. An app keeps that list as the conversation, and sends it again with every new message: changed in place, it would get a new system prompt each time. Build a new list, as the box did with `new`, and leave `messages` as it was.",
                      nl="Je functie geeft de goede lijst, maar hij veranderde ook de lijst die hij kreeg. Een app bewaart die lijst als het gesprek, en stuurt hem bij elk nieuw bericht opnieuw: als hij zelf verandert, krijgt hij elke keer een nieuwe system prompt. Maak een nieuwe lijst, zoals het vak deed met `new`, en laat `messages` zoals hij was.")],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`with_system` still gives back the conversation as it got it. Make the system prompt its first message, and leave out a system prompt the conversation already had.",
                      "nl": "`with_system` geeft het gesprek nog terug zoals hij het kreeg. Maak de system prompt het eerste bericht, en laat een system prompt die het gesprek al had weg."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. First a message with the role `system` and the system prompt as its content, then the messages of the conversation.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Eerst een bericht met de rol `system` en de system prompt als content, dan de berichten van het gesprek."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for a conversation that already has a system prompt, and for an empty one?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Werkt je functie voor een gesprek dat al een system prompt heeft, en voor een leeg gesprek?"},
            "no_function": {"en": "Your box does not make a function `with_system`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `with_system`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `with_system` is right in all {n} cases: the system prompt first, an old one left out, and the rest of the conversation as it was.",
                        "nl": "Je `with_system` klopt in alle {n} gevallen: de system prompt eerst, een oude weggelaten, en de rest van het gesprek zoals het was."},
        },
    ),
    "ask_json": ApiBox(
        name="ask_json",
        picture=json_exchange_picture,
        function="ask_json",
        start=START_JSON,
        solution=SOLUTION_JSON,
        cases=JSON_CASES,
        mistakes=[
            Mistake("data-text", error=_not_bytes,
                    en="urllib sends bytes, not text. Keep `.encode()` after `json.dumps(body)`, as the box had it.",
                    nl="urllib verstuurt bytes, geen tekst. Laat `.encode()` achter `json.dumps(body)` staan, zoals het in het vak stond."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`ask_json` sends only the question yet, and gives back only what the model wrote. Send `start` as a response prefill, and give back `start` and what the model wrote after it.",
                      "nl": "`ask_json` stuurt nog alleen de vraag, en geeft alleen terug wat het model schreef. Stuur `start` mee als response prefill, en geef `start` en wat het model erna schreef terug."},
            "correct": {"en": "Your `ask_json` sends `start` as the last message, of the assistant, and joins it with what the model wrote after it: the whole JSON text, for both calls the check made.",
                        "nl": "Je `ask_json` stuurt `start` als laatste bericht, van de assistant, en zet het vóór wat het model erna schreef: de hele JSON-tekst, bij beide aanroepen van de check."},
            "no-prefill": {"en": "With `{call}`, your messages end with the question, so the model begins its answer itself. Add a last message with the role `assistant` and `start` as its content.",
                           "nl": "Met `{call}` eindigen je berichten met de vraag, dus begint het model zijn antwoord zelf. Voeg een laatste bericht toe met de rol `assistant` en `start` als content."},
            "prefill-first": {"en": "With `{call}`, the assistant's message comes before the question. A prefill must be the last message: the model goes on from the end of the conversation.",
                              "nl": "Met `{call}` komt het bericht van de assistant vóór de vraag. Een prefill moet het laatste bericht zijn: het model gaat verder vanaf het einde van het gesprek."},
            "prefill-role": {"en": "With `{call}`, `start` is in the last message, but with another role than `assistant`. The start of the answer is the assistant's: give that message the role `assistant`.",
                             "nl": "Met `{call}` staat `start` in het laatste bericht, maar met een andere rol dan `assistant`. Het begin van het antwoord is van de assistant: geef dat bericht de rol `assistant`."},
            "other-start": {"en": "With `{call}`, the assistant's message has another content than `start`. Send the `start` that the function gets, as it is.",
                            "nl": "Met `{call}` heeft het bericht van de assistant een andere content dan `start`. Stuur de `start` die de functie krijgt, zoals hij is."},
            "only-rest": {"en": "With `{call}`, your function gives back `{got}`: only what the model wrote after the prefill. The answer holds only that, so put `start` in front of it: `start + ...`.",
                          "nl": "Met `{call}` geeft je functie `{got}` terug: alleen wat het model na de prefill schreef. Het antwoord bevat alleen dat, dus zet `start` ervoor: `start + ...`."},
            "parsed": {"en": "With `{call}`, your function reads the JSON and gives back a dictionary. Give back the text, `start` joined with what the model wrote, so whoever calls it can see what came back.",
                       "nl": "Met `{call}` leest je functie de JSON en geeft een dictionary terug. Geef de tekst terug, `start` samen met wat het model schreef, zodat wie hem aanroept kan zien wat er terugkwam."},
            "whole": {"en": "With `{call}`, your function gives back the whole JSON of the answer. Give back `start` joined with the text in `answer[\"choices\"][0][\"message\"][\"content\"]`.",
                      "nl": "Met `{call}` geeft je functie de hele JSON van het antwoord terug. Geef `start` terug, samen met de tekst in `answer[\"choices\"][0][\"message\"][\"content\"]`."},
            "other": {"en": "With `{call}`, the request is right, but your function gives back `{got}`. Give back `start` and then what the model wrote after it.",
                      "nl": "Met `{call}` klopt het request, maar je functie geeft `{got}` terug. Geef `start` terug en daarachter wat het model erna schreef."},
        },
    ),
}

REQUIRED = ["with_system", "ask_json", "secret"]

CASES = [
    ("with_system", START_WITH, "empty", None),
    ("with_system", "", "empty", None),
    ("with_system", SOLUTION_WITH, "correct", None),
    ("with_system", '''def with_system(messages, system):
    rest = messages
    if len(messages) > 0:
        if messages[0]["role"] == "system":
            rest = messages[1:]
    return [{"role": "system", "content": system}] + rest
''', "correct", None),
    ("with_system", SOLUTION_WITH.replace('    new = [{"role": "system", "content": system}]\n', "    new = []\n")
     .replace("    return new\n", '    new.append({"role": "system", "content": system})\n    return new\n'), "wrong", "at-end"),
    ("with_system", START_WITH.replace("    new = []\n", '    new = [{"role": "system", "content": system}]\n'), "almost", "kept-old"),
    ("with_system", SOLUTION_WITH.replace('message["role"] != "system"', 'message["role"] == "user"'), "wrong", "users-only"),
    ("with_system", 'def with_system(messages, system):\n    return [{"role": "system", "content": system}]\n', "wrong", "only-system"),
    ("with_system", SOLUTION_WITH.replace('"content": system}', '"text": system}'), "wrong", "text-key"),
    ("with_system", '''def with_system(messages, system):
    if len(messages) > 0:
        if messages[0]["role"] == "system":
            messages.pop(0)
    messages.insert(0, {"role": "system", "content": system})
    return messages
''', "almost", "changed"),
    ("ask_json", START_JSON, "empty", None),
    ("ask_json", SOLUTION_JSON, "correct", None),
    ("ask_json", START_JSON.replace('    return answer["choices"][0]["message"]["content"]\n',
                                    '    return start + answer["choices"][0]["message"]["content"]\n'), "wrong", "no-prefill"),
    ("ask_json", SOLUTION_JSON.replace('[{"role": "user", "content": question},\n                {"role": "assistant", "content": start}]',
                                       '[{"role": "assistant", "content": start},\n                {"role": "user", "content": question}]'), "wrong", "prefill-first"),
    ("ask_json", SOLUTION_JSON.replace('{"role": "assistant", "content": start}', '{"role": "user", "content": start}'), "wrong", "prefill-role"),
    ("ask_json", SOLUTION_JSON.replace('{"role": "assistant", "content": start}', '{"role": "assistant", "content": "{"}'), "wrong", "other-start"),
    ("ask_json", SOLUTION_JSON.replace("    return start + answer", "    return answer"), "wrong", "only-rest"),
    ("ask_json", SOLUTION_JSON.replace('    return start + answer["choices"][0]["message"]["content"]\n',
                                       '    return json.loads(start + answer["choices"][0]["message"]["content"])\n'), "wrong", "parsed"),
    ("ask_json", SOLUTION_JSON.replace('    return start + answer["choices"][0]["message"]["content"]\n', "    return answer\n"), "wrong", "whole"),
    ("ask_json", SOLUTION_JSON.replace("json.dumps(body).encode()", "json.dumps(body)"), "error", "data-text"),
    ("ask_json", SOLUTION_JSON.replace('    return start + answer["choices"][0]["message"]["content"]\n',
                                       '    return start + answer["choices"][0]["message"]["content"].upper()\n'), "wrong", "other"),
]


# --- the demo: your with_system, against the course's own server -------------------------------------------------------

def downloaded() -> bool:
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    chat.download_or_stop(button, texts)


def learner_with_system(text):
    """The learner's with_system from exercise 1, when it is right; else None."""
    from everything_ai.check import MISSING
    if text is MISSING or text is None or EXERCISES["with_system"].check(text).state != "correct":
        return None
    return run_code(text).names.get("with_system")


def demo_messages(with_system, system: str, question: str, start: str) -> list:
    """The messages the demo sends: the question, after the system prompt through the learner's with_system when the
    box has one, and the prefill as the last message when there is one."""
    messages = [{"role": "user", "content": question}]
    if system.strip():
        messages = with_system(messages, system)
    if start:
        messages = messages + [{"role": "assistant", "content": start}]
    return messages


def _post(url: str, body: dict) -> dict:
    import urllib.request
    request = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def demo(with_system, system: str, question: str, start: str, words: dict, lang: str) -> str:
    """The question sent to the course's server with Qwen3 behind it, with the learner's with_system and the prefill:
    the messages that went, the answer (the prefill bold, then what the model wrote), its tokens, and its seconds."""
    try:
        messages = demo_messages(with_system, system, question, start)
        running = serve.model_server()
        t = time.perf_counter()
        answer = _post(running.url, {"model": serve.MODEL, "messages": messages, "max_tokens": 200})
    except Exception as e:   # the learner's code, against the real server; it may have run out of memory
        said = words["memory"] if serve.out_of_memory(e) else words["failed"].format(why=f"{type(e).__name__}: {e}")
        return f"<p><em>{html.escape(said)}</em></p>"
    seconds = f"{time.perf_counter() - t:.1f}".replace(".", "," if lang == "nl" else ".")
    choice, usage = answer["choices"][0], answer.get("usage", {})
    body = (_label(words["sent"]) + _messages_html(messages) + _label(words["answer"])
            + f'<div style="padding:0.3rem 0.5rem;background:var(--ea-inset);border-radius:4px">{_mine(start)}{_model(choice["message"]["content"])}</div>'
            + _label(words["done"].format(given=usage.get("prompt_tokens", 0), written=usage.get("completion_tokens", 0),
                                          finish=choice["finish_reason"], seconds=seconds)))
    return yours.figure(words["title"], body, alt=words["title"])
