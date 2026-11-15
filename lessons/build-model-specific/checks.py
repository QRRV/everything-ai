# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-model-specific: a function that finds the examples that worked before and fail now, and one that
runs every example with given settings against a stand-in of the server; the course's recorded runs
(data/course-dev-prompts-qwen.json): three messages with an event, each asked four times the normal way, with
thinking and 80 tokens, with thinking and 1000 tokens, and with thinking and Qwen3's own switch /no_think; their
pictures, and the demo."""
import html
import json

from everything_ai import apicheck, chat, paths, serve, yours
from everything_ai.apicheck import URL, ApiBox, Case
from everything_ai.check import FunctionBox, Mistake, run_code, with_time_limit

DOWNLOADS = {"Qwen3-0.6B": chat.FILES}   # what the demo downloads, for the line under the header

SYSTEM = ("You are a friendly assistant. Answer briefly. "
          "When a task asks for JSON, answer with the JSON only, without a code block.")
EVENT = ("Give the event in this message as JSON with the keys what, day and hour. "
         "The hour is a number from 0 to 23.\n\n{text}")
MESSAGES = {"lunch": "Can we have lunch on Thursday at 12? Kim",
            "football": "Football training moves to Tuesday, 7 pm.",
            "book-club": "Reminder: the book club meets on Sunday at 3 in the afternoon."}
RIGHT = {"lunch": ("thursday", 12), "football": ("tuesday", 19), "book-club": ("sunday", 15)}   # the day and hour of each
NO_THINK = " /no_think"   # Qwen3's own switch in a message: no thinking for this answer
SEEDS = 4
RECORD = {}
for _key, _text in MESSAGES.items():
    _asked = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": EVENT.format(text=_text)}]
    RECORD["normal-" + _key] = {"messages": _asked, "max_tokens": 80, "seeds": SEEDS}
    RECORD["think-short-" + _key] = {"messages": _asked, "max_tokens": 80, "seeds": SEEDS, "think": True}
    RECORD["think-" + _key] = {"messages": _asked, "max_tokens": 1000, "seeds": SEEDS, "think": True}
    RECORD["no-think-" + _key] = {"messages": [_asked[0], {"role": "user", "content": _asked[1]["content"] + NO_THINK}],
                                  "max_tokens": 80, "seeds": SEEDS, "think": True}
DATA = paths.ROOT / "data" / "course-dev-prompts-qwen.json"
MODES = ("normal", "think-short", "think", "no-think")


def course() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-model-specific"]


def answers(mode: str, key: str) -> list:
    return course()[f"{mode}-{key}"]["answers"]


def messages_for(text):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": EVENT.format(text=text)}]


OK = '''def ok(name, answer):
    start, end = answer.find("{"), answer.rfind("}")
    if start < 0 or end < start:
        return False
    try:
        event = json.loads(answer[start:end + 1])
    except ValueError:
        return False
    if not isinstance(event, dict):
        return False
    day, hour = RIGHT[name]
    return str(event.get("day", "")).strip().lower() == day and event.get("hour") == hour
'''
ok = run_code(OK, {"json": json, "RIGHT": RIGHT}).names["ok"]
ok.__doc__ = """Whether an answer gives the event of example `name` right: JSON somewhere in it, with the right day and hour."""


def findings() -> dict:
    """What the recorded runs show, per mode: per example how many of its four answers were right, all right answers,
    how often the answer ran out of tokens, and the most tokens of an answer; and the hour each mode gave each example."""
    out = {"seeds": SEEDS, "examples": len(MESSAGES)}
    for mode in MODES:
        per = {key: [ok(key, a["text"]) for a in answers(mode, key)] for key in MESSAGES}
        every = [a for key in MESSAGES for a in answers(mode, key)]
        out[mode] = {"right": {key: sum(v) for key, v in per.items()}, "all": sum(sum(v) for v in per.values()),
                     "cut": sum(a.get("finish") == "length" for a in every),
                     "tokens": max(a["tokens"] for a in every),
                     "hours": {key: sorted({_hour(a["text"]) for a in answers(mode, key)}, key=str) for key in MESSAGES}}
    return out


def _hour(text: str):
    """The hour in the JSON of an answer, or None."""
    start, end = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[start:end + 1]).get("hour") if 0 <= start < end else None
    except (ValueError, AttributeError):
        return None


def number(x: float, lang: str, digits: int = 1) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "\x00").replace(".", ",").replace("\x00", ".") if lang == "nl" else text


# --- the pictures of the recorded runs -------------------------------------------------------------------------------

_LABEL = "font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)"


def _label(text: str) -> str:
    return f'<div style="{_LABEL};margin:0.2rem 0">{html.escape(text)}</div>'


_MODE_WORDS = {
    "en": {"title": "The same prompt, four ways of asking Qwen3: right answers out of {n}",
           "normal": "the normal way", "think-short": "thinking, max_tokens 80", "think": "thinking, max_tokens 1000",
           "no-think": "thinking, with /no_think in the message", "right": "{k} of {n} right", "cut": "{c} ran out of tokens",
           "slow": "up to {t} tokens an answer"},
    "nl": {"title": "Dezelfde prompt, vier manieren om Qwen3 te vragen: goede antwoorden van de {n}",
           "normal": "de gewone manier", "think-short": "thinking, max_tokens 80", "think": "thinking, max_tokens 1000",
           "no-think": "thinking, met /no_think in het bericht", "right": "{k} van de {n} goed", "cut": "{c} hadden te weinig tokens",
           "slow": "tot {t} tokens per antwoord"},
}


def modes_picture(lang: str) -> str:
    """A row per way of asking: a mark per answer, right in the colour of model output and wrong as a mistake, with
    how many ran out of tokens, and the most tokens an answer took (the seconds were measured beside other work, so the
    lesson leaves them out)."""
    w = _MODE_WORDS[lang]
    f = findings()
    n = SEEDS * len(MESSAGES)
    rows = []
    for mode in MODES:
        marks = "".join(f'<span style="display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:2px;'
                        f'background:var(--ea-{"pred" if ok(key, a["text"]) else "loss"})"></span>'
                        for key in MESSAGES for a in answers(mode, key))
        m = f[mode]
        notes = [w["right"].format(k=m["all"], n=n)] + ([w["cut"].format(c=m["cut"])] if m["cut"] else [])
        notes.append(w["slow"].format(t=m["tokens"]))
        rows.append(f'<div style="display:grid;grid-template-columns:minmax(0,1.5fr) minmax(0,2fr);gap:0.6rem;align-items:center;margin:0.35rem 0">'
                    f'<div style="font-family:var(--ea-font-ui);font-size:13px;color:var(--ea-ink)">{html.escape(w[mode])}</div>'
                    f'<div>{marks}<div style="{_LABEL}">{html.escape(" · ".join(notes))}</div></div></div>')
    return yours.figure(w["title"].format(n=n), "".join(rows), alt=w["title"].format(n=n))


_ANSWER_WORDS = {"en": {"title": "What Qwen3 answered to the lunch message, {mode}", "empty": "(no answer: the tokens ran out while it was thinking)"},
                 "nl": {"title": "Wat Qwen3 antwoordde op het lunchbericht, {mode}", "empty": "(geen antwoord: de tokens waren op tijdens het denken)"}}


def answer_picture(lang: str, mode: str) -> str:
    """The first recorded answer to the lunch message in one way of asking, as it came back."""
    w = _ANSWER_WORDS[lang]
    a = answers(mode, "lunch")[0]
    text = a["text"].strip() or w["empty"]
    color = "var(--ea-pred-text)" if ok("lunch", a["text"]) else "var(--ea-loss-text)"
    title = w["title"].format(mode=_MODE_WORDS[lang][mode])
    return yours.figure(title, f'<pre style="font-family:var(--ea-font-code);font-size:12px;color:{color};white-space:pre-wrap;margin:0">{html.escape(text)}</pre>', alt=title)


# --- exercise 1: worse ---------------------------------------------------------------------------------------------------------------

START_WORSE = '''def worse(old, new):
    names = []
    for name in old:
        names.append(name)
    return names
'''
SOLUTION_WORSE = '''def worse(old, new):
    names = []
    for name in old:
        if old[name] and not new[name]:
            names.append(name)
    return names
'''
WORSE_CASES = [({"lunch": True, "football": True, "book-club": False}, {"lunch": False, "football": True, "book-club": False}),
               ({"lunch": False, "football": True, "book-club": True}, {"lunch": True, "football": False, "book-club": True}),
               ({"a": True, "b": False, "c": True}, {"c": False, "b": True, "a": False}),
               ({"a": True, "b": False}, {"a": True, "b": False}),
               ({}, {})]

_WORSE_WORDS = {
    "en": {"title": "What your worse found for the second case", "name": "example", "old": "before", "new": "now",
           "yours": "in your list", "pass": "right", "fail": "wrong", "want": "It should find: {want}"},
    "nl": {"title": "Wat je worse vond voor het tweede geval", "name": "voorbeeld", "old": "eerst", "new": "nu",
           "yours": "in je lijst", "pass": "goed", "fail": "fout", "want": "Hij zou moeten vinden: {want}"},
}


def worse_picture(seen) -> str:
    """The second case as a table: each example right or wrong before and now, and whether the learner's worse put it
    in its list; the names that were right before and are wrong now are the ones it should find."""
    w = _WORSE_WORDS[seen.lang]
    mine = seen.mine
    fn = mine.names.get("worse") if mine is not None else None
    if not callable(fn):
        return ""
    old, new = WORSE_CASES[1]
    want = worse(dict(old), dict(new))
    try:
        got = with_time_limit(lambda: fn(dict(old), dict(new)))
    except Exception:   # the learner's code; the check says what went wrong
        return ""
    got = got if isinstance(got, list) else []

    def cell(passed):
        return (f'<td style="text-align:center;font-family:var(--ea-font-ui);font-size:12px;padding:0.15rem 0.5rem;'
                f'color:var(--ea-{"pred-text" if passed else "loss-text"})">{w["pass"] if passed else w["fail"]}</td>')
    rows = []
    for name in old:
        mark = "✓" if name in got else ""
        good = (name in got) == (name in want)
        rows.append(f'<tr><td style="font-family:var(--ea-font-code);font-size:12px;padding:0.15rem 0.5rem;color:var(--ea-ink)">{html.escape(name)}</td>'
                    f'{cell(old[name])}{cell(new[name])}<td style="text-align:center;font-size:13px;color:var(--ea-{"ink" if good else "loss-text"})">{mark}</td></tr>')
    head = "".join(f'<th style="{_LABEL};font-weight:500;padding:0 0.5rem">{w[k]}</th>' for k in ("name", "old", "new", "yours"))
    return yours.figure(w["title"], f'<table style="border-collapse:collapse"><tr>{head}</tr>{"".join(rows)}</table>'
                        + _label(w["want"].format(want=repr(want))), alt=w["title"])


worse = run_code(SOLUTION_WORSE).names["worse"]


# --- exercise 2: run_examples ----------------------------------------------------------------------------------------------------

THINK = {"chat_template_kwargs": {"enable_thinking": True}}
SAYS = {   # what the stand-in answers the normal way: the hour of the book club wrong, as Qwen3 did
    "lunch": '{"what": "lunch", "day": "Thursday", "hour": 12}',
    "football": '{"what": "football training", "day": "Tuesday", "hour": 19}',
    "book-club": '{"what": "book club", "day": "Sunday", "hour": 3}',
}
SAYS_THINKING = dict(SAYS, **{"football": '{"what": "football training", "day": "Tuesday", "hour": 7}'})   # as Qwen3 did, thinking
_BY_TEXT = {EVENT.format(text=text): key for key, text in MESSAGES.items()}


class _ByMode(serve.StandIn):
    """A stand-in that answers each example the normal way, or, when the request asks for thinking, after a thought of
    300 words: with max_tokens below that, the tokens run out before the answer starts, as with Qwen3."""

    def answer(self, request: dict, heard) -> dict:
        key = _BY_TEXT.get(request["messages"][-1].get("content", "") if request.get("messages") else "")
        thinking = bool((request.get("chat_template_kwargs") or {}).get("enable_thinking"))
        limit = request.get("max_tokens") or 256
        if key is None:
            text = "The stand-in has no answer to this message."
        elif thinking and limit < 300:
            return {"text": "", "tool_calls": [], "finish": "length", "prompt_tokens": serve._words(request["messages"]), "tokens": int(limit)}
        else:
            text = (SAYS_THINKING if thinking else SAYS)[key]
        self.answers = {request["messages"][-1].get("content", ""): text}
        return super().answer(request, heard)


START_RUN = ('import json\nimport urllib.request\n\n\ndef run_examples(url, settings, texts):\n'
             '    results = {}\n'
             '    for name in texts:\n'
             '        body = {"model": "qwen3-0.6b", "messages": messages_for(texts[name]), "max_tokens": 80}\n'
             '        data = json.dumps(body).encode()\n'
             '        request = urllib.request.Request(url + "/chat/completions", data=data,\n'
             '                                         headers={"Content-Type": "application/json"})\n'
             '        with urllib.request.urlopen(request) as response:\n'
             '            answer = json.loads(response.read())\n'
             '        results[name] = answer["choices"][0]["message"]["content"]\n'
             '    return results\n')
SOLUTION_RUN = (START_RUN
                .replace('"max_tokens": 80}\n', '"max_tokens": 80}\n        for key in settings:\n            body[key] = settings[key]\n')
                .replace('        results[name] = answer["choices"][0]["message"]["content"]\n',
                         '        results[name] = ok(name, answer["choices"][0]["message"]["content"])\n'))


def _posts(log: list) -> list:
    return [e["body"] for e in log if e["method"] == "POST" and isinstance(e["body"], dict)]


def _judge_run(settings: dict, think: bool):
    want = {key: ok(key, (SAYS_THINKING if think else SAYS)[key]) if not (think and settings.get("max_tokens", 80) < 300) else False
            for key in MESSAGES}

    def judge(got, running):
        sent = _posts(running.log)
        if len(sent) != len(MESSAGES):
            return "count"
        if any(b.get(k) != v for b in sent for k, v in settings.items()):
            return "settings"
        if sorted(json.dumps(b.get("messages"), sort_keys=True) for b in sent) != sorted(json.dumps(messages_for(t), sort_keys=True) for t in MESSAGES.values()):
            return "messages"
        if isinstance(got, dict) and any(isinstance(v, str) for v in got.values()):
            return "text"
        return None if got == want else "other"
    return judge


RUN_CASES = [Case(lambda: _ByMode(), (URL, {}, dict(MESSAGES)), _judge_run({}, False)),
             Case(lambda: _ByMode(), (URL, dict(THINK), dict(MESSAGES)), _judge_run(THINK, True)),
             Case(lambda: _ByMode(), (URL, dict(THINK, max_tokens=1000), dict(MESSAGES)), _judge_run(dict(THINK, max_tokens=1000), True))]

_EXCHANGE = {"en": {"title": "What your run_examples sent with thinking switched on, and what came back", "request": "Your code sent: {method} {path}",
                    "headers": "Headers:", "answer": "The server sent back: status {status}",
                    "stream": "a stream with {n} pieces", "none": "Your function sent nothing to the server yet."},
             "nl": {"title": "Wat je run_examples stuurde met thinking aan, en wat er terugkwam", "request": "Je code stuurde: {method} {path}",
                    "headers": "Headers:", "answer": "De server stuurde terug: status {status}",
                    "stream": "een stream met {n} stukjes", "none": "Je functie stuurde nog niets naar de server."}}


def run_picture(seen) -> str:
    return apicheck.exchange(seen, "run_examples", RUN_CASES[1], _EXCHANGE[seen.lang])


def _not_bytes(e) -> bool:
    return isinstance(e, TypeError) and "bytes" in str(e)


_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}

EXERCISES = {
    "worse": FunctionBox(
        name="worse",
        picture=worse_picture,
        function="worse",
        start=START_WORSE,
        solution=SOLUTION_WORSE,
        visible=[WORSE_CASES[0]],
        hidden=WORSE_CASES[1:],
        mistakes=[
            Mistake("now-wrong", wrong=SOLUTION_WORSE.replace("if old[name] and not new[name]:", "if not new[name]:"),
                    en="`{call}` gives `{got}`: every example that is wrong now, also the ones that were already wrong before. Those did not get worse. Take only the names that were right in `old` as well.",
                    nl="`{call}` geeft `{got}`: elk voorbeeld dat nu fout is, ook de voorbeelden die eerst al fout waren. Die zijn niet slechter geworden. Neem alleen de namen die in `old` ook goed waren."),
            Mistake("changed", wrong=SOLUTION_WORSE.replace("if old[name] and not new[name]:", "if old[name] != new[name]:"),
                    en="`{call}` gives `{got}`: every example that changed, also the ones that got better. Take only the names that were right before and are wrong now.",
                    nl="`{call}` geeft `{got}`: elk voorbeeld dat veranderde, ook de voorbeelden die beter werden. Neem alleen de namen die eerst goed waren en nu fout zijn."),
            Mistake("better", wrong=SOLUTION_WORSE.replace("if old[name] and not new[name]:", "if new[name] and not old[name]:"),
                    en="`{call}` gives `{got}`: the examples that got better, wrong before and right now. Turn it around: right in `old`, wrong in `new`.",
                    nl="`{call}` geeft `{got}`: de voorbeelden die beter werden, eerst fout en nu goed. Draai het om: goed in `old`, fout in `new`."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`worse` still gives back every name. Keep only the names that were right in `old` and are wrong in `new`.",
                      "nl": "`worse` geeft nog elke naam terug. Houd alleen de namen over die in `old` goed waren en in `new` fout zijn."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: the names that are `True` in `old` and `False` in `new`, in the order of `old`.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: de namen die in `old` `True` zijn en in `new` `False`, in de volgorde van `old`."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function follow the order of `old`, and does it work with no examples at all?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Volgt je functie de volgorde van `old`, en werkt hij zonder voorbeelden?"},
            "no_function": {"en": "Your box does not make a function `worse`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `worse`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `worse` is right in all {n} cases: the examples that were right before and are wrong now, and nothing else.",
                        "nl": "Je `worse` klopt in alle {n} gevallen: de voorbeelden die eerst goed waren en nu fout zijn, en verder niets."},
        },
    ),
    "run_examples": ApiBox(
        name="run_examples",
        picture=run_picture,
        function="run_examples",
        start=START_RUN,
        solution=SOLUTION_RUN,
        names={"messages_for": messages_for, "ok": ok},
        cases=RUN_CASES,
        mistakes=[
            Mistake("data-text", error=_not_bytes,
                    en="urllib sends bytes, not text. Keep `.encode()` after `json.dumps(body)`, as the box had it.",
                    nl="urllib verstuurt bytes, geen tekst. Laat `.encode()` achter `json.dumps(body)` staan, zoals het in het vak stond."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`run_examples` sends every example, but without the settings, and it keeps the answer itself. Put each setting in the body, and keep `ok(name, ...)` of the answer.",
                      "nl": "`run_examples` stuurt elk voorbeeld, maar zonder de settings, en bewaart het antwoord zelf. Zet elke setting in de body, en bewaar `ok(name, ...)` van het antwoord."},
            "correct": {"en": "Your `run_examples` sends every example with the settings in its body, and keeps for each name whether the answer was right.",
                        "nl": "Je `run_examples` stuurt elk voorbeeld met de settings in de body, en bewaart per naam of het antwoord goed was."},
            "count": {"en": "With `{call}`, your function sent another number of requests than there are examples. Send one request per name in `texts`.",
                      "nl": "Met `{call}` stuurde je functie een ander aantal requests dan er voorbeelden zijn. Stuur één request per naam in `texts`."},
            "settings": {"en": "With `{call}`, a body does not have every setting with its value. After the line with `body = ...`, go through `settings` and put each key with its value in `body`; a setting such as `max_tokens` then replaces the 80.",
                         "nl": "Met `{call}` heeft een body niet elke setting met zijn waarde. Loop na de regel met `body = ...` door `settings` en zet elke key met zijn waarde in `body`; een setting als `max_tokens` vervangt dan de 80."},
            "messages": {"en": "With `{call}`, the messages are not `messages_for(texts[name])`. Keep the messages as the box had them, so that only the settings change.",
                         "nl": "Met `{call}` zijn de berichten niet `messages_for(texts[name])`. Houd de berichten zoals ze in het vak stonden, zodat alleen de settings veranderen."},
            "text": {"en": "With `{call}`, the requests are right, but your function keeps the answers themselves. Keep `ok(name, ...)` of each answer: `True` or `False`.",
                     "nl": "Met `{call}` kloppen de requests, maar je functie bewaart de antwoorden zelf. Bewaar `ok(name, ...)` van elk antwoord: `True` of `False`."},
            "other": {"en": "With `{call}`, the requests are right, but your function gives back `{got}`. Give back a dictionary with for each name `ok(name, ...)` of its answer.",
                      "nl": "Met `{call}` kloppen de requests, maar je functie geeft `{got}` terug. Geef een dictionary terug met per naam `ok(name, ...)` van zijn antwoord."},
        },
    ),
}

REQUIRED = ["worse", "run_examples", "migrate"]

CASES = [
    ("worse", START_WORSE, "empty", None),
    ("worse", "", "empty", None),
    ("worse", SOLUTION_WORSE, "correct", None),
    ("worse", "def worse(old, new):\n    return [name for name in old if old[name] and not new[name]]\n", "correct", None),
    ("worse", SOLUTION_WORSE.replace("if old[name] and not new[name]:", "if not new[name]:"), "wrong", "now-wrong"),
    ("worse", SOLUTION_WORSE.replace("if old[name] and not new[name]:", "if old[name] != new[name]:"), "wrong", "changed"),
    ("worse", SOLUTION_WORSE.replace("if old[name] and not new[name]:", "if new[name] and not old[name]:"), "wrong", "better"),
    ("worse", SOLUTION_WORSE.replace("for name in old:", "for name in new:"), "almost", None),
    ("run_examples", START_RUN, "empty", None),
    ("run_examples", SOLUTION_RUN, "correct", None),
    ("run_examples", SOLUTION_RUN.replace("        for key in settings:\n            body[key] = settings[key]\n", ""), "wrong", "settings"),
    ("run_examples", SOLUTION_RUN.replace('        body = {"model": "qwen3-0.6b", "messages": messages_for(texts[name]), "max_tokens": 80}\n        for key in settings:\n            body[key] = settings[key]\n',
                                          '        body = {}\n        for key in settings:\n            body[key] = settings[key]\n        body["model"] = "qwen3-0.6b"\n        body["messages"] = messages_for(texts[name])\n        body["max_tokens"] = 80\n'), "wrong", "settings"),
    ("run_examples", SOLUTION_RUN.replace("    for name in texts:\n", "    for name in list(texts)[:1]:\n"), "wrong", "count"),
    ("run_examples", SOLUTION_RUN.replace("messages_for(texts[name])", "messages_for(name)"), "wrong", "messages"),
    ("run_examples", START_RUN.replace('"max_tokens": 80}\n', '"max_tokens": 80}\n        for key in settings:\n            body[key] = settings[key]\n'), "wrong", "text"),
    ("run_examples", SOLUTION_RUN.replace("    return results\n", "    return list(results.values())\n"), "wrong", "other"),
    ("run_examples", SOLUTION_RUN.replace("json.dumps(body).encode()", "json.dumps(body)"), "error", "data-text"),
]


# --- the demo: your run_examples and worse, against the course's own server ------------------------------------------------------

def downloaded() -> bool:
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    chat.download_or_stop(button, texts)


def learner_functions(worse_text, run_text):
    """The learner's worse and run_examples from exercises 1 and 2, when both are right; else None."""
    from everything_ai.check import MISSING
    for name, text in (("worse", worse_text), ("run_examples", run_text)):
        if text is MISSING or text is None or EXERCISES[name].check(text).state != "correct":
            return None
    return (run_code(worse_text).names["worse"],
            run_code(run_text, {"messages_for": messages_for, "ok": ok}).names["run_examples"])


DEMO_SETTINGS = {"think-short": dict(THINK), "no-think": dict(THINK)}


def demo(worse_fn, run_examples, mode: str, words: dict, lang: str) -> str:
    """The learner's run_examples against the course's server with Qwen3 behind it: the three examples the normal way,
    then in the chosen way; with /no_think the switch goes in each message. Then the learner's worse on the two."""
    import time
    running = serve.model_server()
    texts = dict(MESSAGES)
    new_texts = {k: t + NO_THINK for k, t in texts.items()} if mode == "no-think" else texts
    start = time.perf_counter()
    try:
        old = run_examples(running.url, {}, texts)
        new = run_examples(running.url, DEMO_SETTINGS[mode], new_texts)
        lost = worse_fn(old, new)
    except Exception as e:   # the learner's code, against the real server; it may have run out of memory
        said = words["memory"] if serve.out_of_memory(e) else words["failed"].format(why=f"{type(e).__name__}: {e}")
        return f"<p><em>{html.escape(said)}</em></p>"
    rows = "".join(f'<tr><td style="font-family:var(--ea-font-code);font-size:12px;padding:0.15rem 0.5rem">{html.escape(k)}</td>'
                   + "".join(f'<td style="font-family:var(--ea-font-ui);font-size:12px;padding:0.15rem 0.5rem;color:var(--ea-{"pred-text" if r.get(k) else "loss-text"})">{words["right"] if r.get(k) else words["wrong"]}</td>' for r in (old, new))
                   + "</tr>" for k in texts)
    head = "".join(f'<th style="{_LABEL};font-weight:500;padding:0 0.5rem">{h}</th>' for h in ("", words["old"], words["new"]))
    seconds = number(time.perf_counter() - start, lang, 0)
    return yours.figure(words["title"], f'<table style="border-collapse:collapse"><tr>{head}</tr>{rows}</table>'
                        + f'<div style="font-family:var(--ea-font-ui);font-size:14px;font-weight:600;color:var(--ea-ink);margin-top:0.4rem">{html.escape(words["lost"].format(names=lost, seconds=seconds))}</div>', alt=words["title"])
