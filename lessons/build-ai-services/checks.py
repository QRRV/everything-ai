# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-ai-services: a function that reads the label out of a model's answer, and one that asks the
course's server to sort a customer's message into one of six labels, against a stand-in; the course's run of Qwen3 on
the 120 test questions of data/banking77-small.json (data/course-apis-qwen.json), the table of what it answered, and
the demo with the learner's own function."""
import html
import json

from everything_ai import apicheck, chat, paths, serve, yours
from everything_ai.apicheck import URL, ApiBox, Case
from everything_ai.check import MISSING, FunctionBox, Mistake, run_code

DOWNLOADS = {"Qwen3-0.6B": chat.FILES}   # what the demo downloads, for the line under the header

SYSTEM = ("Sort the customer's message into one of these labels, and answer with the label only.\n"
          "card_arrival: the new card has not arrived yet\n"
          "lost_or_stolen_card: the card is lost or stolen\n"
          "exchange_rate: a question about exchange rates\n"
          "card_payment_wrong_exchange_rate: a card payment used the wrong exchange rate\n"
          "pin_blocked: the PIN is blocked\n"
          "request_refund: the customer wants money back for something")
RECORD = {"test": {"labels": "test", "system": SYSTEM, "max_tokens": 20}}
DATA = paths.ROOT / "data" / "course-apis-qwen.json"
BANKING = paths.ROOT / "data" / "banking77-small.json"


def banking() -> dict:
    """The questions of data/banking77-small.json: "labels", and "train" and "test" as [label, text] pairs."""
    return json.loads(BANKING.read_text(encoding="utf-8"))


LABELS = tuple(banking()["labels"])


def recorded() -> list:
    """The course's run: Qwen3's answer to each test question, in the order of the file, with its seconds and tokens."""
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-ai-services"]["test"]["answers"]


def _course_parse(answer, labels):
    best = None
    for label in labels:
        if label in answer.lower() and (best is None or len(label) > len(best)):
            best = label
    return best


def table(parse=_course_parse) -> dict:
    """What `parse` makes of Qwen3's answers: for each true label, how often each label (or None) came out, and the
    number right of all."""
    rows = {label: {} for label in LABELS}
    right = 0
    for (true, _), a in zip(banking()["test"], recorded()):
        got = parse(a["text"], list(LABELS))
        got = got if got in LABELS else None
        rows[true][got] = rows[true].get(got, 0) + 1
        right += got == true
    return {"rows": rows, "right": right, "n": len(recorded())}


def run_facts() -> dict:
    """The numbers the lesson gives about the course's run: right, of all, and per answer the seconds and tokens."""
    a = recorded()
    t = table()
    return {"right": t["right"], "n": t["n"], "seconds": sum(x["seconds"] for x in a) / len(a),
            "prompt_tokens": sum(x["prompt_tokens"] for x in a) / len(a), "tokens": sum(x["tokens"] for x in a) / len(a),
            "exact": sum(x["text"] in LABELS for x in a)}


def _nr(x: float, lang: str, digits: int = 1) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


_TABLE = {
    "en": {"title": "What Qwen3 answered to the 120 test questions", "true": "the question is about", "said": "Qwen3 answered",
           "none": "no label", "right": "{right} of {n} right: {pct}%",
           "alt": "A table with a row for each of the six labels a question is about and a column for each label Qwen3 answered; the number in each cell counts the questions. The cells where the two are the same hold the right answers."},
    "nl": {"title": "Wat Qwen3 antwoordde op de 120 testvragen", "true": "de vraag gaat over", "said": "Qwen3 antwoordde",
           "none": "geen label", "right": "{right} van de {n} goed: {pct}%",
           "alt": "Een tabel met een rij voor elk van de zes labels waar een vraag over gaat en een kolom voor elk label dat Qwen3 antwoordde; het getal in elke cel telt de vragen. De cellen waar die twee gelijk zijn, bevatten de goede antwoorden."},
}


def table_html(t: dict, lang: str, title: str | None = None) -> str:
    """The table of `t` (from table()): a row per true label, a column per answered label, and one for no label when
    any; the right answers on the diagonal, the mistakes in the colour of mistakes."""
    w = _TABLE[lang]
    cols = list(LABELS) + ([None] if any(None in r for r in t["rows"].values()) else [])
    e = html.escape
    th = 'style="font-family:var(--ea-font-code);font-size:11px;font-weight:400;color:var(--ea-ink-2);padding:2px 4px;text-align:{a};{x}"'
    head = "".join(f'<th {th.format(a="center", x="writing-mode:vertical-rl;transform:rotate(180deg);height:9.5rem;vertical-align:bottom")}>'
                   f'{e(c if c is not None else w["none"])}</th>' for c in cols)
    body = ""
    for true in LABELS:
        cells = ""
        for c in cols:
            n = t["rows"][true].get(c, 0)
            colour = ("var(--ea-ink)" if c == true else "var(--ea-loss-text)") if n else "var(--ea-ink-2)"
            weight = "600" if n and c == true else "400"
            back = "background:var(--ea-inset);" if c == true else ""
            cells += (f'<td style="text-align:center;padding:3px 6px;font-family:var(--ea-font-ui);font-size:13px;'
                      f'color:{colour};font-weight:{weight};{back}">{n if n else "·"}</td>')
        body += f'<tr><th {th.format(a="right", x="")}>{e(true)}</th>{cells}</tr>'
    pct = _nr(100 * t["right"] / t["n"], lang)
    out = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.2rem 0">'
           f'<tr><th style="font-family:var(--ea-font-ui);font-size:11px;font-weight:400;color:var(--ea-ink-2);text-align:right;vertical-align:bottom;padding:2px 4px">'
           f'{e(w["true"])} ↓ · {e(w["said"])} →</th>{head}</tr>{body}</table></div>'
           f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2);margin-top:0.3rem">'
           f'{e(w["right"].format(right=t["right"], n=t["n"], pct=pct))}</div>')
    return yours.figure(title or w["title"], out, alt=w["alt"])


def table_picture(lang: str) -> str:
    """The course's run, read with the course's parse_label."""
    return table_html(table(), lang)


# --- exercise 1: the label in an answer ---------------------------------------------------------------------------------

_START_PARSE = '''def parse_label(answer, labels):
    best = None
    for label in labels:
        ...
    return best
'''
SOLUTION_PARSE = '''def parse_label(answer, labels):
    best = None
    for label in labels:
        if label in answer.lower() and (best is None or len(label) > len(best)):
            best = label
    return best
'''
_FIRST = '''def parse_label(answer, labels):
    best = None
    for label in labels:
        if label in answer.lower():
            return label
    return best
'''
_LAST = '''def parse_label(answer, labels):
    best = None
    for label in labels:
        if label in answer.lower():
            best = label
    return best
'''
_SAME = '''def parse_label(answer, labels):
    best = None
    for label in labels:
        if label == answer.lower():
            best = label
    return best
'''
_CASE = SOLUTION_PARSE.replace("label in answer.lower()", "label in answer")
_EARLY = SOLUTION_PARSE.replace("            best = label\n    return best", "            best = label\n        return best")
_THREE = ["exchange_rate", "card_payment_wrong_exchange_rate", "pin_blocked"]


def _kept(value):
    """What parse_label gave, kept in a tuple: None is one of its right answers, not a function without return."""
    return ("gave", value)


def _shown(value):
    return value[1] if isinstance(value, tuple) and len(value) == 2 and value[0] == "gave" else value
_TWO = ["exchange_rate", "pin_blocked"]


def parse_picture(seen) -> str:
    """The learner's parse_label on Qwen3's 120 answers of the course's run: the table it makes of them."""
    run = seen.mine
    fn = run.names.get("parse_label") if run is not None else None
    if not callable(fn):
        return ""
    title = {"en": "Your parse_label on Qwen3's 120 answers", "nl": "Je parse_label op de 120 antwoorden van Qwen3"}[seen.lang]
    try:
        t = table(lambda answer, labels: fn(answer, labels))
    except Exception:   # the learner's code
        return ""
    return table_html(t, seen.lang, title)


# --- exercise 2: sort a message through the API -------------------------------------------------------------------------

class NoParse(Exception):
    """The learner's parse_label is not right yet."""


def no_parse(answer, labels):
    raise NoParse()


def learner_parse(text):
    """The learner's parse_label from exercise 1, when it is right; else None."""
    if text is MISSING or text is None or EXERCISES["parse_label"].check(text).state != "correct":
        return None
    return run_code(text).names.get("parse_label")


def label_inputs(parse_text) -> dict:
    """What exercise 2 runs with: the system prompt, the labels, and the learner's parse_label from exercise 1, or one
    that says to finish it first."""
    return {"SYSTEM": SYSTEM, "LABELS": list(LABELS), "parse_label": learner_parse(parse_text) or no_parse}


_START_LABEL = '''import json
import urllib.request


def label(url, message):
    messages = ...
    body = {"model": "qwen3-0.6b", "messages": messages, "max_tokens": 20, "temperature": 0}
    request = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        answer = json.loads(response.read())
    return ...
'''
SOLUTION_LABEL = '''import json
import urllib.request


def label(url, message):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": message}]
    body = {"model": "qwen3-0.6b", "messages": messages, "max_tokens": 20, "temperature": 0}
    request = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        answer = json.loads(response.read())
    return parse_label(answer["choices"][0]["message"]["content"], LABELS)
'''
_ASKED = {"My new card still hasn't come. How long does it take?": ("card_arrival", "card_arrival"),
          "The rate on my card payment abroad was not the real one.": ("Label: card_payment_wrong_exchange_rate.",
                                                                       "card_payment_wrong_exchange_rate")}


def _stand_in():
    return serve.StandIn(answers={m: said for m, (said, _) in _ASKED.items()})


def _judge(message: str, want: str, said: str):
    def judge(got, running):
        try:
            sent = running.log[-1]["body"]["messages"]
        except (IndexError, KeyError, TypeError):
            return "other"
        roles = [m.get("role") for m in sent if isinstance(m, dict)]
        if "system" not in roles:
            return "no-system"
        if next(m for m in sent if m.get("role") == "system").get("content") != SYSTEM:
            return "other-system"
        if not sent or sent[-1].get("role") != "user" or sent[-1].get("content") != message:
            return "no-message"
        if got == want:
            return None
        if got == said:
            return "not-parsed"
        if isinstance(got, dict):
            return "whole"
        if got is None:
            return "none"
        return "other"
    return judge


LABEL_CASES = [Case(_stand_in, (URL, m), _judge(m, want, said)) for m, (said, want) in _ASKED.items()]


def label_picture(seen) -> str:
    words = {"en": {"title": "What your label sent, and what came back", "request": "Your code sent: {method} {path}",
                    "headers": "Headers:", "answer": "The server sent back: status {status}",
                    "stream": "a stream with {n} pieces", "none": "Your function sent nothing to the server yet."},
             "nl": {"title": "Wat je label stuurde, en wat er terugkwam", "request": "Je code stuurde: {method} {path}",
                    "headers": "Headers:", "answer": "De server stuurde terug: status {status}",
                    "stream": "een stream met {n} stukjes", "none": "Je functie stuurde nog niets naar de server."}}[seen.lang]
    return apicheck.exchange(seen, "label", LABEL_CASES[1], words)


_COMMON = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
                "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."},
}

EXERCISES = {
    "parse_label": FunctionBox(
        name="parse_label",
        picture=parse_picture,
        function="parse_label",
        start=_START_PARSE,
        solution=SOLUTION_PARSE,
        collect=_kept,
        show=_shown,
        visible=[("card_payment_wrong_exchange_rate", _THREE)],
        hidden=[("Exchange_Rate", _TWO), ("The label is: pin_blocked.", _TWO), ("I am not sure.", _TWO),
                ("Label: card_payment_wrong_exchange_rate", ["card_payment_wrong_exchange_rate", "exchange_rate"])],
        mistakes=[
            Mistake("first", wrong=_FIRST,
                    en="`{call}` gives `{got}`. That label is in the answer, but only as a piece of a longer one: your function stops at the first label it finds. Go through all labels, and keep the longest one that is in the answer.",
                    nl="`{call}` geeft `{got}`. Dat label staat in het antwoord, maar alleen als stuk van een langer label: je functie stopt bij het eerste label dat hij vindt. Loop alle labels langs, en houd het langste dat in het antwoord staat."),
            Mistake("last", wrong=_LAST,
                    en="`{call}` gives `{got}`, and it should give `{want}`. Your function keeps the last label it finds, and here the shorter one comes last. Keep a label only when it is longer than the one you have.",
                    nl="`{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Je functie houdt het laatste label dat hij vindt, en hier komt het kortere als laatste. Houd een label alleen als het langer is dan het label dat je al hebt."),
            Mistake("same", wrong=_SAME,
                    en="`{call}` gives `{got}`: your function only finds a label when the answer is that label and nothing else. A model often writes more around it. Look whether the label is *in* the answer, with `in`.",
                    nl="`{call}` geeft `{got}`: je functie vindt een label alleen als het antwoord precies dat label is en verder niets. Een model schrijft er vaak meer omheen. Kijk of het label *in* het antwoord staat, met `in`."),
            Mistake("prints", matches=lambda got, text: _shown(got) is None and "print(" in text,
                    en="`{call}` gives `None`: your function shows the label with `print`, but gives nothing back. Keep it in `best`, which the function gives back at the end.",
                    nl="`{call}` geeft `None`: je functie laat het label zien met `print`, maar geeft niets terug. Bewaar het in `best`, dat de functie aan het eind teruggeeft."),
            Mistake("case", wrong=_CASE,
                    en="`{call}` gives `{got}`: the answer has capitals, and the labels have none. Compare with the answer in small letters, `answer.lower()`.",
                    nl="`{call}` geeft `{got}`: het antwoord heeft hoofdletters, en de labels niet. Vergelijk met het antwoord in kleine letters, `answer.lower()`."),
            Mistake("early-return", wrong=_EARLY,
                    en="`{call}` gives `{got}`: the `return` is inside the loop, so your function stops after the first label. Move it back, so it comes after the loop.",
                    nl="`{call}` geeft `{got}`: de `return` staat in de loop, dus je functie stopt na het eerste label. Zet hem terug, zodat hij na de loop komt."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "Replace the `...` in the loop: when a label is in the answer and longer than `best`, it becomes the new `best`.",
                      "nl": "Vervang de `...` in de loop: als een label in het antwoord staat en langer is dan `best`, wordt het de nieuwe `best`."},
            "correct": {"en": "Right: your `parse_label` finds the longest label in an answer, with or without capitals and words around it, and gives `None` when there is none. The table under the check uses it on Qwen3's 120 answers.",
                        "nl": "Goed: je `parse_label` vindt het langste label in een antwoord, met of zonder hoofdletters en woorden eromheen, en geeft `None` als er geen is. De tabel onder de check gebruikt hem op de 120 antwoorden van Qwen3."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`.",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn."},
        },
    ),
    "label": ApiBox(
        name="label",
        picture=label_picture,
        function="label",
        start=_START_LABEL,
        solution=SOLUTION_LABEL,
        cases=LABEL_CASES,
        names={"SYSTEM": SYSTEM, "LABELS": list(LABELS), "parse_label": run_code(SOLUTION_PARSE).names["parse_label"]},
        mistakes=[
            Mistake("no-parse", error=lambda e: isinstance(e, NoParse),
                    en="This exercise uses your `parse_label` from exercise 1. Finish exercise 1 first; this check then runs again by itself.",
                    nl="Deze oefening gebruikt je `parse_label` uit oefening 1. Maak eerst oefening 1 af; deze check draait daarna vanzelf opnieuw."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "Replace the two `...`: the messages, with the system prompt and the customer's message, and what the function gives back, the label.",
                      "nl": "Vervang de twee `...`: de messages, met de system prompt en het bericht van de klant, en wat de functie teruggeeft, het label."},
            "correct": {"en": "Right: your `label` sends the system prompt and the message, and reads the label out of the answer with your `parse_label`, also when the model writes more around it.",
                        "nl": "Goed: je `label` stuurt de system prompt en het bericht, en leest het label uit het antwoord met je `parse_label`, ook als het model er meer omheen schrijft."},
            "no-system": {"en": "With `{call}`, the server got no message with the role `\"system\"`, so the model does not know the labels. Put the system prompt `SYSTEM` first, as a message with that role.",
                          "nl": "Met `{call}` kreeg de server geen bericht met de rol `\"system\"`, dus het model kent de labels niet. Zet de system prompt `SYSTEM` vooraan, als bericht met die rol."},
            "other-system": {"en": "With `{call}`, the system message is not the system prompt `SYSTEM` that lists the labels. Use `SYSTEM` as its content.",
                             "nl": "Met `{call}` is het bericht met de rol `\"system\"` niet de system prompt `SYSTEM` met de labels erin. Gebruik `SYSTEM` als content."},
            "no-message": {"en": "With `{call}`, the last message the server got is not the customer's `message`, with the role `\"user\"`. Put it after the system prompt.",
                           "nl": "Met `{call}` is het laatste bericht dat de server kreeg niet het `message` van de klant, met de rol `\"user\"`. Zet het na de system prompt."},
            "not-parsed": {"en": "With `{call}`, your function gives back `{got}`, the whole text of the answer. Give the text to your `parse_label`, with `LABELS`, so that only the label comes back.",
                           "nl": "Met `{call}` geeft je functie `{got}` terug, de hele tekst van het antwoord. Geef de tekst aan je `parse_label`, met `LABELS`, zodat alleen het label terugkomt."},
            "whole": {"en": "With `{call}`, your function gives back the whole JSON of the answer. Take the text out first, under `\"choices\"`, then read the label out of it.",
                      "nl": "Met `{call}` geeft je functie de hele JSON van het antwoord terug. Haal er eerst de tekst uit, onder `\"choices\"`, en lees daar het label uit."},
            "none": {"en": "With `{call}`, your function gives back nothing. End with `return` and the label.",
                     "nl": "Met `{call}` geeft je functie niets terug. Eindig met `return` en het label."},
            "other": {"en": "With `{call}`, your function gives `{got}`, not the label that is in the server's answer.",
                      "nl": "Met `{call}` geeft je functie `{got}`, niet het label dat in het antwoord van de server staat."},
        },
    ),
}

REQUIRED = ["parse_label", "label", "own-labels", "mixed"]

CASES = [
    ("parse_label", _START_PARSE, "empty", None),
    ("parse_label", SOLUTION_PARSE, "correct", None),
    ("parse_label", _FIRST, "wrong", "first"),
    ("parse_label", _LAST, "wrong", "last"),
    ("parse_label", _SAME, "wrong", "same"),
    ("parse_label", _CASE, "wrong", "case"),
    ("parse_label", '''def parse_label(answer, labels):
    best = None
    for label in labels:
        if label in answer.lower() and (best is None or len(label) > len(best)):
            best = label
        return best
''', "wrong", "early-return"),
    ("parse_label", '''def parse_label(answer, labels):
    best = None
    for label in labels:
        if label in answer.lower() and (best is None or len(label) > len(best)):
            print(label)
    return best
''', "wrong", "prints"),
    ("label", _START_LABEL, "empty", None),
    ("label", SOLUTION_LABEL, "correct", None),
    ("label", SOLUTION_LABEL.replace('{"role": "system", "content": SYSTEM}, ', ""), "wrong", "no-system"),
    ("label", SOLUTION_LABEL.replace('"content": SYSTEM}', '"content": "Sort the message."}'), "wrong", "other-system"),
    ("label", SOLUTION_LABEL.replace('{"role": "user", "content": message}', '{"role": "user", "content": "message"}'), "wrong", "no-message"),
    ("label", SOLUTION_LABEL.replace('return parse_label(answer["choices"][0]["message"]["content"], LABELS)',
                                     'return answer["choices"][0]["message"]["content"]'), "wrong", "not-parsed"),
    ("label", SOLUTION_LABEL.replace('return parse_label(answer["choices"][0]["message"]["content"], LABELS)', "return answer"), "wrong", "whole"),
    ("label", SOLUTION_LABEL.replace("return parse_label(", "parse_label("), "wrong", "none"),
]


# --- the demo: the learner's label, against the course's server with Qwen3 -----------------------------------------------

def downloaded() -> bool:
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    chat.download_or_stop(button, texts)


def learner_label(text, parse_text):
    """The learner's label from exercise 2 with their parse_label, when both are right; else None."""
    inputs = label_inputs(parse_text)
    if inputs["parse_label"] is no_parse or text is MISSING or text is None:
        return None
    if EXERCISES["label"].check(text, inputs).state != "correct":
        return None
    return run_code(text, inputs).names.get("label")


def demo(label, message: str, words: dict, lang: str) -> str:
    """The learner's label for a message, sent to the course's server with Qwen3 behind it: the label, the text the
    model wrote, and how long it took."""
    import time
    running = serve.model_server()
    before = len(running.log)
    start = time.perf_counter()
    try:
        got = label(running.url, message)
    except Exception as e:   # the learner's code, against the real server
        said = words["memory"] if serve.out_of_memory(e) else words["failed"].format(why=f"{type(e).__name__}: {e}")
        return f"<p><em>{html.escape(said)}</em></p>"
    took = _nr(time.perf_counter() - start, lang)
    texts = [entry["answer"]["choices"][0]["message"]["content"] for entry in running.log[before:]
             if isinstance(entry.get("answer"), dict) and entry["answer"].get("choices")]
    wrote = texts[-1] if texts else ""
    shown = words["label"].format(label=got) if got is not None else words["no_label"]
    body = (f'<div style="font-family:var(--ea-font-code);font-size:14px;color:var(--ea-pred-text);font-weight:600">{html.escape(shown)}</div>'
            f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2);margin-top:0.25rem">'
            f'{html.escape(words["wrote"].format(text=wrote, seconds=took))}</div>')
    return yours.figure(words["title"], body, alt=words["title"])
