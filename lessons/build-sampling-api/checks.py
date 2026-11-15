# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-sampling-api: a function that sends temperature and seed with a request, and one that ends the
answer at the first line break with a stop sequence and gives back the text and its finish_reason, both against a
stand-in of the server; the course's recorded runs (data/course-dev-prompts-qwen.json): the same question eight times
at three settings, and one answer ended three ways; their pictures, and the demo."""
import html
import json

from everything_ai import apicheck, chat, paths, serve, yours
from everything_ai.apicheck import URL, ApiBox, Case
from everything_ai.check import Mistake

DOWNLOADS = {"Qwen3-0.6B": chat.FILES}   # what the demo downloads, for the line under the header

NAME = "Suggest a name for a small bakery. Answer with the name only."
FRUITS = "Name three fruits, one per line."
ADVISED = {"temperature": 0.7, "top_p": 0.8}
SEEDS = 8
RECORD = {
    "greedy": {"messages": [{"role": "user", "content": NAME}], "max_tokens": 12, "temperature": 0, "seeds": SEEDS},
    "advised": {"messages": [{"role": "user", "content": NAME}], "max_tokens": 12, **ADVISED, "seeds": SEEDS},
    "hot": {"messages": [{"role": "user", "content": NAME}], "max_tokens": 12, "temperature": 1.5, "top_p": 1.0, "seeds": SEEDS},
    "again": {"messages": [{"role": "user", "content": NAME}], "max_tokens": 12, **ADVISED, "seeds": 3},
    "lines": {"messages": [{"role": "user", "content": FRUITS}], "max_tokens": 40, **ADVISED, "seeds": 1},
    "first": {"messages": [{"role": "user", "content": FRUITS}], "max_tokens": 40, **ADVISED, "stop": ["\n"], "seeds": 1},
    "short": {"messages": [{"role": "user", "content": FRUITS}], "max_tokens": 3, **ADVISED, "seeds": 1},
}
DATA = paths.ROOT / "data" / "course-dev-prompts-qwen.json"


def course() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-sampling-api"]


def texts(key: str) -> list:
    return [a["text"] for a in course()[key]["answers"]]


def sentence(text: str) -> bool:
    """An answer that is a sentence about a name, not the name only."""
    return text.rstrip().endswith(".") or " named " in text


def findings() -> dict:
    """What the recorded runs show: how many different names each setting gave in eight runs, how many answers at the
    highest temperature were a sentence and not the name only, whether the seeds 0 to 2 gave the same names again,
    and the three endings of the fruit answer."""
    R = course()
    out = {k: len(set(texts(k))) for k in ("greedy", "advised", "hot")}
    out["sentences"] = sum(sentence(t) for t in texts("hot"))
    out["repeated"] = texts("again") == texts("advised")[:3]
    for k in ("lines", "first", "short"):
        a = R[k]["answers"][0]
        out[k] = {"text": a["text"], "tokens": a["tokens"], "finish": a["finish"]}
    return out


# --- the pictures of the recorded runs -------------------------------------------------------------------------------

_RUNS_WORDS = {
    "en": {"title": "The same question, eight times at each setting", "seed": "seed", "names": "{n} different",
           "one": "1 different", "cols": ["temperature 0", "temperature 0.7, top_p 0.8", "temperature 1.5, top_p 1.0"],
           "question": "The question: " + NAME},
    "nl": {"title": "Dezelfde vraag, acht keer bij elke instelling", "seed": "seed", "names": "{n} verschillend",
           "one": "1 verschillend", "cols": ["temperature 0", "temperature 0,7, top_p 0,8", "temperature 1,5, top_p 1,0"],
           "question": "De vraag: " + NAME},
}

_CELL = "padding:0.18rem 0.5rem;border-bottom:1px solid var(--ea-line);vertical-align:top;font-size:13px"


def _label(text: str) -> str:
    return f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2);margin:0.2rem 0">{html.escape(text)}</div>'


def runs_picture(lang: str) -> str:
    """The recorded answers to the bakery question: a column per setting, a row per seed, and under each column how
    many different answers it has. The answers are text the model wrote."""
    w = _RUNS_WORDS[lang]
    cols = [texts(k) for k in ("greedy", "advised", "hot")]
    head = (f'<th style="{_CELL};color:var(--ea-ink-2);font-weight:500;text-align:left;font-family:var(--ea-font-ui)">{w["seed"]}</th>'
            + "".join(f'<th style="{_CELL};color:var(--ea-ink-2);font-weight:500;text-align:left;font-family:var(--ea-font-ui)">{html.escape(c)}</th>'
                      for c in w["cols"]))
    rows = []
    for i in range(SEEDS):
        rows.append(f'<tr><td style="{_CELL};color:var(--ea-ink-2);font-family:var(--ea-font-ui)">{i}</td>'
                    + "".join(f'<td style="{_CELL};color:var(--ea-pred-text);font-family:var(--ea-font-code);white-space:pre-wrap">'
                              f'{html.escape(c[i])}</td>' for c in cols) + "</tr>")
    counts = [len(set(c)) for c in cols]
    rows.append(f'<tr><td style="{_CELL};border-bottom:none"></td>'
                + "".join(f'<td style="{_CELL};border-bottom:none;color:var(--ea-ink);font-family:var(--ea-font-ui);font-weight:600">'
                          f'{w["one"] if n == 1 else w["names"].format(n=n)}</td>' for n in counts) + "</tr>")
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%">'
             f'<thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    return yours.figure(w["title"], _label(w["question"]) + table, alt=w["title"])


_ENDS_WORDS = {
    "en": {"title": "One answer, ended three ways", "question": "The question: " + FRUITS,
           "rows": ["max_tokens 40, nothing else", 'stop ["\\n"]', "max_tokens 3"], "setting": "in the body",
           "answer": "the answer", "finish": "finish_reason", "tokens": "tokens written"},
    "nl": {"title": "Eén antwoord, op drie manieren beëindigd", "question": "De vraag: " + FRUITS,
           "rows": ["max_tokens 40, verder niets", 'stop ["\\n"]', "max_tokens 3"], "setting": "in de body",
           "answer": "het antwoord", "finish": "finish_reason", "tokens": "geschreven tokens"},
}


def ends_picture(lang: str) -> str:
    """The fruit answer three ways: as the model ended it, at the stop sequence "\\n", and at max_tokens 3, each with
    its finish_reason and the tokens the model wrote. A line break in an answer shows as ↵ and a new line."""
    w, f = _ENDS_WORDS[lang], findings()
    head = f"{_CELL};color:var(--ea-ink-2);font-weight:500;font-family:var(--ea-font-ui)"
    th, th_right = f'style="{head};text-align:left"', f'style="{head};text-align:right"'
    rows = []
    for label, key in zip(w["rows"], ("lines", "first", "short")):
        a = f[key]
        shown = html.escape(a["text"]).replace("\n", '<span style="color:var(--ea-ink-3)">↵</span>\n')
        rows.append(f'<tr><td style="{_CELL};font-family:var(--ea-font-code);color:var(--ea-ink)">{html.escape(label)}</td>'
                    f'<td style="{_CELL};font-family:var(--ea-font-code);color:var(--ea-pred-text);white-space:pre-wrap">{shown}</td>'
                    f'<td style="{_CELL};font-family:var(--ea-font-code);color:var(--ea-ink)">"{a["finish"]}"</td>'
                    f'<td style="{_CELL};font-family:var(--ea-font-ui);color:var(--ea-ink);text-align:right">{a["tokens"]}</td></tr>')
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%"><thead><tr>'
             f'<th {th}>{w["setting"]}</th><th {th}>{w["answer"]}</th><th {th}>{w["finish"]}</th>'
             f'<th {th_right}>{w["tokens"]}</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    return yours.figure(w["title"], _label(w["question"]) + table, alt=w["title"])


# --- exercise 1: ask with temperature and seed ------------------------------------------------------------------------

_CLIENT = '''    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        answer = json.loads(response.read())
'''
START_ASK = ('import json\nimport urllib.request\n\n\ndef ask(url, question, temperature, seed):\n'
             '    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": question}], "max_tokens": 12}\n'
             + _CLIENT + '    return answer["choices"][0]["message"]["content"]\n')
SOLUTION_ASK = START_ASK.replace('"max_tokens": 12}', '"max_tokens": 12,\n            "temperature": temperature, "seed": seed}')
FLOWERS = "Suggest a name for a flower shop. Answer with the name only."
ASKED = {NAME: "Bakeria", FLOWERS: "Petal and Stem"}


def _sent(running) -> list:
    return [e["body"] for e in running.log if e["method"] == "POST" and isinstance(e["body"], dict)]


def _judge_ask(question, temperature, seed):
    want = ASKED[question]

    def judge(got, running):
        sent = _sent(running)
        body = sent[-1] if sent else {}
        if "temperature" not in body:
            return "temp-key" if "temp" in body else "no-temperature"
        if body["temperature"] != temperature:
            return "fixed-temperature"
        if "seed" not in body:
            return "no-seed"
        if body["seed"] != seed:
            return "fixed-seed"
        if isinstance(got, dict):
            return "whole"
        return None if got == want else "other"
    return judge


ASK_CASES = [Case(lambda: serve.StandIn(ASKED), (URL, NAME, 0.7, 3), _judge_ask(NAME, 0.7, 3)),
             Case(lambda: serve.StandIn(ASKED), (URL, FLOWERS, 0, 1), _judge_ask(FLOWERS, 0, 1))]

_EXCHANGE = {"en": {"title": "What your {fn} sent, and what came back", "request": "Your code sent: {method} {path}",
                    "headers": "Headers:", "answer": "The server sent back: status {status}",
                    "stream": "a stream with {n} pieces", "none": "Your function sent nothing to the server yet."},
             "nl": {"title": "Wat je {fn} stuurde, en wat er terugkwam", "request": "Je code stuurde: {method} {path}",
                    "headers": "Headers:", "answer": "De server stuurde terug: status {status}",
                    "stream": "een stream met {n} stukjes", "none": "Je functie stuurde nog niets naar de server."}}


def ask_picture(seen) -> str:
    words = {**_EXCHANGE[seen.lang], "title": _EXCHANGE[seen.lang]["title"].format(fn="ask")}
    return apicheck.exchange(seen, "ask", ASK_CASES[0], words)


# --- exercise 2: first_line with a stop sequence ------------------------------------------------------------------------

START_FIRST = ('import json\nimport urllib.request\n\n\ndef first_line(url, question):\n'
               '    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": question}], "max_tokens": 20}\n'
               + _CLIENT + '    return answer["choices"][0]["message"]["content"]\n')
SOLUTION_FIRST = (START_FIRST.replace('"max_tokens": 20}', '"max_tokens": 20,\n            "stop": ["\\n"]}')
                  .replace('    return answer["choices"][0]["message"]["content"]\n',
                           '    choice = answer["choices"][0]\n    return choice["message"]["content"], choice["finish_reason"]\n'))
LONG_ASK = "Tell me about apples."
LONG = ("Apples grow on trees in cool places all over the world and come in thousands of kinds that differ in color, "
        "taste and the time of year they are ready to pick")
LISTED = {FRUITS: "Banana\nApple\nCherry", LONG_ASK: LONG}


def _judge_first(question, want):
    def judge(got, running):
        sent = _sent(running)
        body = sent[-1] if sent else {}
        stop = body.get("stop")
        stops = [stop] if isinstance(stop, str) else stop if isinstance(stop, list) else []
        pair = tuple(got) if isinstance(got, (tuple, list)) else None
        if "\n" not in stops:
            return "cut-yourself" if pair == want or got == want[0] else "no-stop"
        if body.get("max_tokens") != 20:
            return "max-tokens"
        if isinstance(got, str):
            return "only-text"
        if pair == want[::-1]:
            return "order"
        return None if pair == want else "other"
    return judge


FIRST_CASES = [Case(lambda: serve.StandIn(LISTED), (URL, FRUITS), _judge_first(FRUITS, ("Banana", "stop"))),
               Case(lambda: serve.StandIn(LISTED), (URL, LONG_ASK),
                    _judge_first(LONG_ASK, (" ".join(LONG.split(" ")[:20]), "length")))]


def first_picture(seen) -> str:
    words = {**_EXCHANGE[seen.lang], "title": _EXCHANGE[seen.lang]["title"].format(fn="first_line")}
    return apicheck.exchange(seen, "first_line", FIRST_CASES[0], words)


_COMMON = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
                "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."},
    "http-400": {"en": "With `{call}`, the server answered 400: \"{message}\" Look at that field in your body.",
                 "nl": "Met `{call}` antwoordde de server 400: \"{message}\" Kijk naar dat veld in je body."},
    "whole": {"en": "With `{call}`, your function gives back the whole JSON of the answer. Keep the last line as it was.",
              "nl": "Met `{call}` geeft je functie de hele JSON van het antwoord terug. Laat de laatste regel zoals hij was."},
}


def _not_bytes(e) -> bool:
    return isinstance(e, TypeError) and "bytes" in str(e)


def _finish_key(e) -> bool:
    return isinstance(e, KeyError) and "finish_reason" in str(e)


EXERCISES = {
    "ask": ApiBox(
        name="ask",
        picture=ask_picture,
        function="ask",
        start=START_ASK,
        solution=SOLUTION_ASK,
        cases=ASK_CASES,
        mistakes=[
            Mistake("data-text", error=_not_bytes,
                    en="urllib sends bytes, not text. Keep `.encode()` after `json.dumps(body)`, as the box had it.",
                    nl="urllib verstuurt bytes, geen tekst. Laat `.encode()` achter `json.dumps(body)` staan, zoals het in het vak stond."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "The body has no `\"temperature\"` and no `\"seed\"` yet. Add both, with the values `ask` gets.",
                      "nl": "De body heeft nog geen `\"temperature\"` en geen `\"seed\"`. Voeg ze allebei toe, met de waarden die `ask` krijgt."},
            "correct": {"en": "Right: your `ask` sends the `temperature` and the `seed` it gets, also a temperature of 0, for both calls the check made.",
                        "nl": "Goed: je `ask` stuurt de `temperature` en de `seed` die hij krijgt, ook een temperature van 0, bij beide aanroepen van de check."},
            "temp-key": {"en": "With `{call}`, your body has a key `\"temp\"`. The API knows only the whole word, `\"temperature\"`; a server may skip a key it does not know, without a word.",
                         "nl": "Met `{call}` heeft je body een key `\"temp\"`. De API kent alleen het hele woord, `\"temperature\"`; een server kan een key die hij niet kent zonder iets te zeggen overslaan."},
            "no-temperature": {"en": "With `{call}`, your body has no `\"temperature\"`, so the server takes its own. Put `\"temperature\": temperature` in the body.",
                               "nl": "Met `{call}` heeft je body geen `\"temperature\"`, dus neemt de server zijn eigen. Zet `\"temperature\": temperature` in de body."},
            "fixed-temperature": {"en": "With `{call}`, your body sends another temperature than the one `ask` got. Send the `temperature` argument itself, not a fixed number.",
                                  "nl": "Met `{call}` stuurt je body een andere temperature dan `ask` kreeg. Stuur het argument `temperature` zelf, geen vast getal."},
            "no-seed": {"en": "With `{call}`, the temperature is right, but your body has no `\"seed\"`. Add `\"seed\": seed`.",
                        "nl": "Met `{call}` klopt de temperature, maar je body heeft geen `\"seed\"`. Voeg `\"seed\": seed` toe."},
            "fixed-seed": {"en": "With `{call}`, your body sends another seed than the one `ask` got. Send the `seed` argument itself.",
                           "nl": "Met `{call}` stuurt je body een andere seed dan `ask` kreeg. Stuur het argument `seed` zelf."},
            "other": {"en": "With `{call}`, the request is right, but your function gives back `{got}`. Keep the last line as it was: the text of the answer.",
                      "nl": "Met `{call}` klopt het request, maar je functie geeft `{got}` terug. Laat de laatste regel zoals hij was: de tekst van het antwoord."},
        },
    ),
    "first_line": ApiBox(
        name="first_line",
        picture=first_picture,
        function="first_line",
        start=START_FIRST,
        solution=SOLUTION_FIRST,
        cases=FIRST_CASES,
        mistakes=[
            Mistake("data-text", error=_not_bytes,
                    en="urllib sends bytes, not text. Keep `.encode()` after `json.dumps(body)`, as the box had it.",
                    nl="urllib verstuurt bytes, geen tekst. Laat `.encode()` achter `json.dumps(body)` staan, zoals het in het vak stond."),
            Mistake("finish-place", error=_finish_key,
                    en="With `{call}`, `finish_reason` is not where your code looks. It sits in the choice, next to the message: `answer[\"choices\"][0][\"finish_reason\"]`.",
                    nl="Met `{call}` staat `finish_reason` niet waar je code zoekt. Hij staat in de choice, naast het bericht: `answer[\"choices\"][0][\"finish_reason\"]`."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "`first_line` asks without a stop sequence yet, and gives back only the text. Add `\"stop\"` to the body, and give back the text and the `finish_reason`.",
                      "nl": "`first_line` vraagt nog zonder stop sequence, en geeft alleen de tekst terug. Voeg `\"stop\"` toe aan de body, en geef de tekst en de `finish_reason` terug."},
            "correct": {"en": "Right: the server ends the answer at the first line break, and your function says why it ended: `\"stop\"` at the line break, `\"length\"` at 20 tokens.",
                        "nl": "Goed: de server beëindigt het antwoord bij de eerste regelovergang, en je functie zegt waarom het eindigde: `\"stop\"` bij de regelovergang, `\"length\"` bij 20 tokens."},
            "cut-yourself": {"en": "With `{call}`, your function gives the right first line, but it cut the text itself: the request had no `\"stop\"`, so the model wrote every line, and every token of them counts. Put `\"stop\": [\"\\n\"]` in the body.",
                             "nl": "Met `{call}` geeft je functie de goede eerste regel, maar hij knipte de tekst zelf: het request had geen `\"stop\"`, dus schreef het model alle regels, en elk token daarvan telt. Zet `\"stop\": [\"\\n\"]` in de body."},
            "no-stop": {"en": "With `{call}`, your request has no stop sequence with a line break, so the answer goes on after the first line. Add `\"stop\"` to the body, with a list that holds `\"\\n\"`.",
                        "nl": "Met `{call}` heeft je request geen stop sequence met een regelovergang, dus gaat het antwoord door na de eerste regel. Voeg `\"stop\"` toe aan de body, met een lijst waar `\"\\n\"` in staat."},
            "max-tokens": {"en": "With `{call}`, the stop sequence is right, but `\"max_tokens\"` is no longer 20. Keep it at 20.",
                           "nl": "Met `{call}` klopt de stop sequence, maar `\"max_tokens\"` is niet meer 20. Laat het op 20 staan."},
            "only-text": {"en": "With `{call}`, your function gives back only the text, `{got}`. Give back two things, the text and the `finish_reason` of the choice: `return text, finish`.",
                          "nl": "Met `{call}` geeft je functie alleen de tekst terug, `{got}`. Geef twee dingen terug, de tekst en de `finish_reason` van de choice: `return text, finish`."},
            "order": {"en": "With `{call}`, your function gives `{got}`: the two are the other way round. First the text, then the `finish_reason`.",
                      "nl": "Met `{call}` geeft je functie `{got}`: de twee staan andersom. Eerst de tekst, dan de `finish_reason`."},
            "other": {"en": "With `{call}`, the request is right, but your function gives back `{got}`. Give back the text of the answer and its `finish_reason`, in that order.",
                      "nl": "Met `{call}` klopt het request, maar je functie geeft `{got}` terug. Geef de tekst van het antwoord en zijn `finish_reason` terug, in die volgorde."},
        },
    ),
}

REQUIRED = ["ask", "first_line", "names"]

CASES = [
    ("ask", START_ASK, "empty", None),
    ("ask", "", "empty", None),
    ("ask", SOLUTION_ASK, "correct", None),
    ("ask", SOLUTION_ASK.replace('"temperature": temperature, ', ""), "wrong", "no-temperature"),
    ("ask", SOLUTION_ASK.replace('"temperature": temperature', '"temp": temperature'), "wrong", "temp-key"),
    ("ask", SOLUTION_ASK.replace('"temperature": temperature', '"temperature": 0.7'), "wrong", "fixed-temperature"),
    ("ask", SOLUTION_ASK.replace(', "seed": seed', ""), "wrong", "no-seed"),
    ("ask", SOLUTION_ASK.replace('"seed": seed', '"seed": 3'), "wrong", "fixed-seed"),
    ("ask", SOLUTION_ASK.replace('"temperature": temperature', '"temperature": str(temperature)'), "error", "http-400"),
    ("ask", SOLUTION_ASK.replace('return answer["choices"][0]["message"]["content"]', "return answer"), "wrong", "whole"),
    ("ask", SOLUTION_ASK.replace("json.dumps(body).encode()", "json.dumps(body)"), "error", "data-text"),
    ("ask", SOLUTION_ASK.replace('return answer["choices"][0]["message"]["content"]', 'return answer["choices"][0]["message"]["content"].upper()'), "wrong", "other"),
    ("first_line", START_FIRST, "empty", None),
    ("first_line", SOLUTION_FIRST, "correct", None),
    ("first_line", SOLUTION_FIRST.replace('["\\n"]', '"\\n"'), "correct", None),
    ("first_line", SOLUTION_FIRST.replace(',\n            "stop": ["\\n"]}', "}").replace(
        'choice["message"]["content"], ', 'choice["message"]["content"].split("\\n")[0], '), "wrong", "cut-yourself"),
    ("first_line", SOLUTION_FIRST.replace(',\n            "stop": ["\\n"]}', "}"), "wrong", "no-stop"),
    ("first_line", SOLUTION_FIRST.replace('"max_tokens": 20', '"max_tokens": 40'), "wrong", "max-tokens"),
    ("first_line", SOLUTION_FIRST.replace(', choice["finish_reason"]', ""), "wrong", "only-text"),
    ("first_line", SOLUTION_FIRST.replace('choice["message"]["content"], choice["finish_reason"]',
                                          'choice["finish_reason"], choice["message"]["content"]'), "wrong", "order"),
    ("first_line", SOLUTION_FIRST.replace('choice["finish_reason"]', 'answer["finish_reason"]'), "error", "finish-place"),
    ("first_line", SOLUTION_FIRST.replace('choice["message"]["content"], ', 'choice["message"]["content"].upper(), '), "wrong", "other"),
    ("first_line", SOLUTION_FIRST.replace('"stop": ["\\n"]', '"stop": ["\\n", "\\n\\n", ".", ",", ";"]'), "error", "http-400"),
]


# --- the demo: your ask, three times against the course's own server ---------------------------------------------------

def downloaded() -> bool:
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    chat.download_or_stop(button, texts)


def learner_ask(text):
    """The learner's ask from exercise 1, when it is right; else None."""
    from everything_ai.check import MISSING, run_code
    if text is MISSING or text is None or EXERCISES["ask"].check(text).state != "correct":
        return None
    return run_code(text).names.get("ask")


def demo(ask, question: str, temperature: float, words: dict, lang: str) -> str:
    """The learner's ask, sent three times to the course's server with Qwen3 behind it, at `temperature` with the
    seeds 0, 1 and 2: the three answers, and how long they took together."""
    import time
    running = serve.model_server()
    start, answers = time.perf_counter(), []
    try:
        for seed in range(3):
            answers.append(str(ask(running.url, question, temperature, seed)))
    except Exception as e:   # the learner's code, against the real server; it may have run out of memory
        said = words["memory"] if serve.out_of_memory(e) else words["failed"].format(why=f"{type(e).__name__}: {e}")
        return f"<p><em>{html.escape(said)}</em></p>"
    seconds = f"{time.perf_counter() - start:.1f}".replace(".", "," if lang == "nl" else ".")
    rows = "".join(f'<tr><td style="{_CELL};color:var(--ea-ink-2);font-family:var(--ea-font-ui)">{words["seed"]} {i}</td>'
                   f'<td style="{_CELL};color:var(--ea-pred-text);font-family:var(--ea-font-code);white-space:pre-wrap">{html.escape(a)}</td></tr>'
                   for i, a in enumerate(answers))
    shown_t = f"{temperature:g}".replace(".", "," if lang == "nl" else ".")
    body = (f'<table style="border-collapse:collapse;width:100%"><tbody>{rows}</tbody></table>'
            + _label(words["done"].format(seconds=seconds, t=shown_t, different=len(set(answers)))))
    return yours.figure(words["title"], body, alt=words["title"])
