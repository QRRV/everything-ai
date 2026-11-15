# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-embeddings-api: a function that asks the course's server for the embeddings of a list of texts,
and one that finds the text that best fits a question with it, both against a stand-in of the server; the course's
recorded runs of Granite Embedding (data/course-embeddings-granite.json), their pictures, and the demo with the
learner's own functions."""
import html
import json
import math
import re

import numpy as np

from everything_ai import apicheck, lesson as lesson_helpers, paths, serve, yours
from everything_ai.apicheck import URL, ApiBox, Case
from everything_ai.check import MISSING, Mistake, run_code

DATA = paths.ROOT / "data" / "course-embeddings-granite.json"
REPO = "ibm-granite/granite-embedding-107m-multilingual"   # everything_ai.embed.REPO, without importing PyTorch
FILES = [(REPO, "model.safetensors"), (REPO, "tokenizer.json")]
DOWNLOADS = {"Granite Embedding": FILES}   # what the demo downloads, for the line under the header

# Four questions of customers, in English and the same four in Dutch
ENGLISH = ["How do I change my password?", "When will my package arrive?", "Can I pay with a gift card?",
           "I want to return my shoes."]
DUTCH = ["Hoe wijzig ik mijn wachtwoord?", "Wanneer komt mijn pakket aan?", "Kan ik met een cadeaubon betalen?",
         "Ik wil mijn schoenen terugsturen."]
# The help pages of a made-up web shop, three in English and three in Dutch
PAGES = ["You can return an order within 30 days, in its box.",
         "Delivery takes two to four working days.",
         "Pay by card, by bank transfer or with a gift card.",
         "Je wachtwoord wijzig je onder Instellingen, bij Account.",
         "Een bestelling annuleren kan zolang het pakket nog niet verzonden is.",
         "De klantenservice is bereikbaar van maandag tot en met vrijdag, van 9 tot 17 uur."]
QUESTIONS = ["How do I change my password?", "Hoe lang duurt de bezorging?", "Kan ik iemand bellen?"]
WORKED = [QUESTIONS[0], PAGES[3], PAGES[1]]
SUN = "The sun is a star."   # a text about something else altogether
# The two calls of exercise 2: a question, and each text with about the dot product Granite gives it with the question
# (tests/test_build_embeddings_api.py compares them with the recorded runs "match0" and "match1")
_MATCHES = [
    ("How do I change my password?", {"Pay by card, by bank transfer or with a gift card.": 0.52,
                                      "Je wachtwoord wijzig je onder Instellingen, bij Account.": 0.69,
                                      "Delivery takes two to four working days.": 0.49}),
    ("Kan ik mijn bestelling nog annuleren?", {"De klantenservice is bereikbaar van maandag tot en met vrijdag.": 0.61,
                                               "You can return an order within 30 days, in its box.": 0.70,
                                               "Delivery takes two to four working days.": 0.61,
                                               "Een bestelling annuleren kan zolang het pakket nog niet verzonden is.": 0.86}),
]

# What data/tools/record_course_embeddings.py asks the course's server
EMBED = {
    "pairs": {"texts": ENGLISH + DUTCH},
    "worked": {"texts": WORKED},
    "search": {"texts": QUESTIONS + PAGES},
    "speed": {"speed": [QUESTIONS[0]] + PAGES, "repeat": 7},
    "sun": {"texts": [QUESTIONS[0], SUN]},
    **{f"match{k}": {"texts": [q] + list(scores)} for k, (q, scores) in enumerate(_MATCHES)},
}


def recorded(key: str) -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-embeddings-api"][key]


def _nr(x: float, lang: str, digits: int = 3) -> str:
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def pair_scores() -> list[list[float]]:
    """The dot product of each English question (rows) with each Dutch one (columns)."""
    dots = recorded("pairs")["dots"]
    n = len(ENGLISH)
    return [[dots[i][n + j] for j in range(n)] for i in range(n)]


def pair_facts(lang: str) -> dict:
    """The numbers the text gives about the pairs: the lowest and highest score of a question with its translation,
    and of a question with another one."""
    s = pair_scores()
    same = [s[i][i] for i in range(len(s))]
    other = [s[i][j] for i in range(len(s)) for j in range(len(s)) if i != j]
    return {"same_low": _nr(min(same), lang, 2), "same_high": _nr(max(same), lang, 2),
            "other_low": _nr(min(other), lang, 2), "other_high": _nr(max(other), lang, 2)}


def search_scores() -> list[list[float]]:
    """The dot product of each of the three questions with each help page."""
    dots, n = recorded("search")["dots"], len(QUESTIONS)
    return [[dots[i][n + j] for j in range(len(PAGES))] for i in range(n)]


def worked_facts(lang: str) -> dict:
    r = recorded("worked")
    return {"tokens": r["tokens"], "prompt_tokens": r["prompt_tokens"], "dutch": _nr(r["dots"][0][1], lang),
            "english": _nr(r["dots"][0][2], lang), "length": _nr(r["lengths"][0], lang, 4)}


def speed_facts(lang: str) -> dict:
    r = recorded("speed")
    return {"together": _nr(r["together"], lang), "apart": _nr(r["apart"], lang),
            "times": _nr(r["apart"] / r["together"], lang, 1), "n": len(EMBED["speed"]["speed"])}


# --- the pictures ---------------------------------------------------------------------------------------------------

def _shade(score: float) -> float:
    """How strongly a score colors its cell: nothing at 0.4 and below, 0.6 at 1."""
    return round(max(0.0, min(1.0, (score - 0.4) / 0.6)) * 0.6, 3)


_PAIRS = {
    "en": {"title": "Four questions in English and the same four in Dutch: the dot product of every two",
           "rows": "English", "cols": "Dutch",
           "note": "Granite gives every embedding length 1, so each number is the cosine similarity of two questions.",
           "alt": "A table of four English questions against the same four in Dutch. The four cells where a question meets its own translation have the highest dot products; all other cells are lower."},
    "nl": {"title": "Vier vragen in het Engels en dezelfde vier in het Nederlands: het inproduct van elk tweetal",
           "rows": "Engels", "cols": "Nederlands",
           "note": "Granite geeft elke embedding lengte 1, dus elk getal is de cosine similarity van twee vragen.",
           "alt": "Een tabel van vier Engelse vragen tegen dezelfde vier in het Nederlands. De vier cellen waar een vraag haar eigen vertaling tegenkomt hebben de hoogste inproducten; alle andere cellen zijn lager."},
}


def pairs_picture(lang: str) -> str:
    w, e = _PAIRS[lang], html.escape
    s = pair_scores()
    small = "font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)"
    legend = "".join(f'<div style="{small};margin:0.1rem 0"><b style="color:var(--ea-ink)">{k + 1}</b>&nbsp; '
                     f'<span style="font-family:var(--ea-font-code);color:var(--ea-ink)">{e(a)}</span> &nbsp;·&nbsp; '
                     f'<span style="font-family:var(--ea-font-code);color:var(--ea-ink)">{e(b)}</span></div>'
                     for k, (a, b) in enumerate(zip(ENGLISH, DUTCH)))
    cell = "padding:0.35rem 0.5rem;text-align:center;font-family:var(--ea-font-code);font-size:13px;color:var(--ea-ink)"
    head = (f'<tr><th style="{small};padding:0.2rem 0.5rem;text-align:left">{e(w["rows"])} ↓ &nbsp; {e(w["cols"])} →</th>'
            + "".join(f'<th style="{small};padding:0.2rem 0.5rem">{j + 1}</th>' for j in range(len(DUTCH))) + "</tr>")
    rows = "".join(
        f'<tr><th style="{small};padding:0.2rem 0.5rem;text-align:left">{i + 1}</th>'
        + "".join(f'<td style="{cell};position:relative;border:1px solid var(--ea-line)">'
                  f'<span style="position:absolute;inset:0;background:var(--ea-pred);opacity:{_shade(v)}"></span>'
                  f'<span style="position:relative;{"font-weight:600" if i == j else ""}">{_nr(v, lang)}</span></td>'
                  for j, v in enumerate(row)) + "</tr>"
        for i, row in enumerate(s))
    body = (legend + f'<table style="border-collapse:collapse;margin:0.6rem 0 0.4rem">{head}{rows}</table>'
            + f'<div style="{small}">{e(w["note"])}</div>')
    return yours.figure(w["title"], body, alt=w["alt"])


def _bars(question: str, texts: list, scores: list[float], lang: str, picked: int | None = None, mark: str = "") -> str:
    """A question, and under it every text with a bar for its score; the highest in the color of what a model makes,
    and text number `picked` with `mark` under it."""
    e = html.escape
    best = max(range(len(scores)), key=lambda k: scores[k])
    rows = []
    for k, (page, v) in enumerate(zip(texts, scores)):
        colour = "var(--ea-pred)" if k == best else "var(--ea-ink-3)"
        note = (f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-pred-text)">{e(mark)}</div>'
                if picked == k and mark else "")
        rows.append(
            f'<div style="font-family:var(--ea-font-code);font-size:12px;color:var(--ea-ink);overflow-wrap:anywhere">{e(page)}{note}</div>'
            f'<div style="align-self:center;height:10px;background:var(--ea-inset);border-radius:3px">'
            f'<div style="width:{max(0.0, v) * 100:.1f}%;height:10px;background:{colour};border-radius:3px"></div></div>'
            f'<div style="align-self:center;font-family:var(--ea-font-code);font-size:12px;color:var(--ea-ink);'
            f'{"font-weight:600" if k == best else ""}">{_nr(v, lang)}</div>')
    return (f'<div style="margin:0.3rem 0 0.8rem"><div style="font-family:var(--ea-font-code);font-size:13px;'
            f'color:var(--ea-ink);margin-bottom:0.3rem">{e(question)}</div>'
            f'<div style="display:grid;grid-template-columns:minmax(0,1fr) minmax(70px,140px) 3.2rem;gap:0.25rem 0.6rem">'
            + "".join(rows) + "</div></div>")


_SEARCH = {
    "en": {"title": "Three questions against six help pages: the score of each page",
           "alt": "Three questions, each with six help pages and a bar for the dot product of the question with each page. The highest bar of each question is colored."},
    "nl": {"title": "Drie vragen tegen zes hulppagina's: de score van elke pagina",
           "alt": "Drie vragen, elk met zes hulppagina's en een balk voor het inproduct van de vraag met elke pagina. De hoogste balk van elke vraag is gekleurd."},
}


def search_picture(lang: str) -> str:
    w = _SEARCH[lang]
    body = "".join(_bars(q, PAGES, s, lang) for q, s in zip(QUESTIONS, search_scores()))
    return yours.figure(w["title"], body, alt=w["alt"])


_TRIM = {"en": {"data": "… {n} characters in all", "numbers": "… {n} numbers in all", "thousands": ","},
         "nl": {"data": "… {n} tekens in totaal", "numbers": "… {n} getallen in totaal", "thousands": "."}}

_WORKED = {
    "en": {"title": "The course's request for three embeddings", "request": "The course's code sent: POST /v1/embeddings",
           "answer": "The server sent back: status 200", "input": "three texts in one list", "e0": "the question",
           "e1": "the Dutch page", "e2": "the English page", "usage": "the tokens of the three texts"},
    "nl": {"title": "Het request van de course voor drie embeddings", "request": "De code van de course stuurde: POST /v1/embeddings",
           "answer": "De server stuurde terug: status 200", "input": "drie teksten in één lijst", "e0": "de vraag",
           "e1": "de Nederlandse pagina", "e2": "de Engelse pagina", "usage": "de tokens van de drie teksten"},
}


def worked_request() -> dict:
    return {"model": serve.EMBEDDER, "input": WORKED}


def worked_answer(lang: str) -> dict:
    """The answer of the worked example as the server sent it, with each embedding cut after four numbers."""
    r = recorded("worked")
    more = _TRIM[lang]["numbers"].format(n=serve.DIMS)
    return {"object": "list", "model": serve.EMBEDDER,
            "data": [{"object": "embedding", "index": i, "embedding": start + [more]} for i, start in enumerate(r["start"])],
            "usage": {"prompt_tokens": r["prompt_tokens"], "total_tokens": r["prompt_tokens"]}}


def _compact(text: str) -> str:
    """JSON with indent, with every list that holds no list or dictionary on one line."""
    return re.sub(r"\[\n\s*([^\[\]{}]*?)\n\s*\]", lambda m: "[" + re.sub(r",\n\s*", ", ", m.group(1)) + "]", text)


def _pre(text: str, marks: dict, model_key: str | None = None) -> str:
    """Code-like text; each line that contains a key of `marks` gets that note beside it, in grey, and each line with
    `model_key` the colour of what a model makes."""
    out = []
    for line in text.split("\n"):
        note = next((n for k, n in marks.items() if k in line), "")
        colour = "var(--ea-pred-text)" if model_key and model_key in line else "var(--ea-ink)"
        out.append(f'<span style="color:{colour}">{html.escape(line)}</span>'
                   + (f'<span style="color:var(--ea-ink-2);font-family:var(--ea-font-ui)">   ← {html.escape(note)}</span>' if note else ""))
    return ('<pre style="margin:0.2rem 0 0.7rem;padding:0.55rem 0.7rem;background:var(--ea-inset);border-radius:6px;'
            'font-family:var(--ea-font-code);font-size:12px;line-height:1.45;white-space:pre-wrap;overflow-wrap:anywhere">'
            + "\n".join(out) + "</pre>")


def _label(text: str) -> str:
    return f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)">{html.escape(text)}</div>'


def worked_picture(lang: str) -> str:
    w = _WORKED[lang]
    request = _compact(json.dumps(worked_request(), ensure_ascii=False, indent=2))
    answer = _compact(json.dumps(worked_answer(lang), ensure_ascii=False, indent=2))
    body = (_label(w["request"]) + _pre(request, {'"input"': w["input"]})
            + _label(w["answer"])
            + _pre(answer, {'"index": 0': w["e0"], '"index": 1': w["e1"], '"index": 2': w["e2"], '"usage"': w["usage"]},
                   model_key='"embedding": ['))
    return yours.figure(w["title"], body, alt=w["title"])


# --- exercise 1: the embeddings of a list of texts ---------------------------------------------------------------------

_START_EMBED = '''import json
import urllib.request


def embed(url, texts):
    body = {"model": "granite-embedding-107m-multilingual", "input": ...}
    request = urllib.request.Request(url + ..., data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        answer = json.loads(response.read())
    return ...
'''
SOLUTION_EMBED = '''import json
import urllib.request


def embed(url, texts):
    body = {"model": "granite-embedding-107m-multilingual", "input": texts}
    request = urllib.request.Request(url + "/embeddings", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        answer = json.loads(response.read())
    return [item["embedding"] for item in answer["data"]]
'''
_EMBEDDED = [["How do I change my password?", "Hoe wijzig ik mijn wachtwoord?"],
             ["Delivery takes two to four working days.", "Wanneer komt mijn pakket aan?", "Can I pay with a gift card?"]]


def _posts(running) -> list:
    return [e for e in running.log if e["method"] == "POST" and isinstance(e["body"], dict)]


def _judge_embed(texts: list):
    def judge(got, running):
        posts = _posts(running)
        sent = posts[0]["body"].get("input") if posts else None
        if sent != texts:
            return "one-text" if sent in (texts[0], [texts[0]]) else "other-input"
        answer = posts[0]["answer"]
        want = [d["embedding"] for d in answer["data"]] if isinstance(answer, dict) and "data" in answer else None
        if want is not None and got == want:
            numbers = all(isinstance(v, list) and len(v) == serve.DIMS and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v) for v in got)
            return None if numbers else "not-numbers"
        if got == answer:
            return "whole"
        if got is None:
            return "none"
        if isinstance(got, dict) and "embedding" in got:
            return "one-item"
        if isinstance(got, list) and got and isinstance(got[0], dict):
            return "data"
        if want and got in want:
            return "first"
        return "other"
    return judge


EMBED_CASES = [Case(serve.StandIn, (URL, texts), _judge_embed(texts)) for texts in _EMBEDDED]


def embed_picture(seen) -> str:
    words = {"en": {"title": "What your embed sent, and what came back", "request": "Your code sent: {method} {path}",
                    "headers": "Headers:", "answer": "The server sent back: status {status}",
                    "stream": "a stream with {n} pieces", "none": "Your function sent nothing to the server yet.", **_TRIM["en"]},
             "nl": {"title": "Wat je embed stuurde, en wat er terugkwam", "request": "Je code stuurde: {method} {path}",
                    "headers": "Headers:", "answer": "De server stuurde terug: status {status}",
                    "stream": "een stream met {n} stukjes", "none": "Je functie stuurde nog niets naar de server.", **_TRIM["nl"]}}[seen.lang]
    return apicheck.exchange(seen, "embed", EMBED_CASES[0], words)


# --- exercise 2: the text that best fits a question --------------------------------------------------------------------

class NoEmbed(Exception):
    """The learner's embed is not right yet."""


def no_embed(url, texts):
    raise NoEmbed()


def learner_embed(text):
    """The learner's embed from exercise 1, when it is right; else None."""
    if text is MISSING or text is None or EXERCISES["embed"].check(text).state != "correct":
        return None
    return run_code(text).names.get("embed")


def best_inputs(embed_text) -> dict:
    """What exercise 2 runs with: the learner's embed from exercise 1, or one that says to finish it first."""
    return {"embed": learner_embed(embed_text) or no_embed}


_START_BEST = '''import numpy as np


def best_match(url, question, texts):
    vectors = ...
    q = np.array(vectors[0])
    best, best_score = None, -2
    for i in range(len(texts)):
        score = ...
        if score > best_score:
            best, best_score = texts[i], score
    return best
'''
SOLUTION_BEST = '''import numpy as np


def best_match(url, question, texts):
    vectors = embed(url, [question] + texts)
    q = np.array(vectors[0])
    best, best_score = None, -2
    for i in range(len(texts)):
        score = np.dot(q, vectors[i + 1])
        if score > best_score:
            best, best_score = texts[i], score
    return best
'''
# The stand-in's embeddings for the texts of the two calls: made so that each text has the dot product with the question
# that _MATCHES gives it, and the right text scores highest
def made_vectors(question: str, scores: dict, seed: int) -> dict:
    """Embeddings of length 1 for a question and its texts, with exactly the given dot products with the question."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(serve.DIMS)
    q /= np.linalg.norm(q)
    out = {question: q}
    for text, s in scores.items():
        r = rng.standard_normal(serve.DIMS)
        r -= (r @ q) * q
        r /= np.linalg.norm(r)
        out[text] = s * q + math.sqrt(1 - s * s) * r
    return {t: [float(x) for x in v] for t, v in out.items()}


MATCH_VECTORS = [made_vectors(q, scores, seed) for seed, (q, scores) in enumerate(_MATCHES)]


def _judge_best(question: str, scores: dict):
    texts = list(scores)
    want = max(texts, key=lambda t: scores[t])

    def judge(got, running):
        posts = _posts(running)
        if len(posts) > 1:
            return "almost" if got == want else "many"
        sent = posts[0]["body"].get("input") if posts else None
        if sent != [question] + texts:
            if sent == texts + [question]:
                return "question-last"
            if isinstance(sent, list) and question not in sent:
                return "no-question"
            return "other-input"
        if got == want:
            return None
        if got is None:
            return "none"
        if got == texts[0]:
            return "off-by-one"
        if got == min(texts, key=lambda t: scores[t]):
            return "worst"
        return "other"
    return judge


def _backend(k: int):
    return lambda: serve.StandIn(vectors=MATCH_VECTORS[k])


BEST_CASES = [Case(_backend(k), (URL, q, list(scores)), _judge_best(q, scores)) for k, (q, scores) in enumerate(_MATCHES)]


def match_scores(k: int = 0) -> list[float]:
    """The dot product of the question of call k with each of its texts, with the stand-in's embeddings."""
    q, scores = _MATCHES[k]
    v = MATCH_VECTORS[k]
    return [float(np.dot(v[q], v[t])) for t in scores]


_MATCH = {"en": {"title": "What your best_match picked", "mark": "← your best_match picked this one",
                 "nothing": "Your best_match gave back {got}, none of the texts.", "error": "Your best_match stops with an error.",
                 "note": "The bars are the dot products of the stand-in's embeddings: the question with each text."},
          "nl": {"title": "Wat je best_match koos", "mark": "← je best_match koos deze",
                 "nothing": "Je best_match gaf {got} terug, geen van de teksten.", "error": "Je best_match stopt met een fout.",
                 "note": "De balken zijn de inproducten van de embeddings van de stand-in: de vraag met elke tekst."}}


def match_picture(seen) -> str:
    """The texts of the first call with their scores, and the one the learner's best_match picked."""
    w = _MATCH[seen.lang]
    run = seen.run
    fn = run.names.get("best_match") if run is not None and run.error is None else None
    if not callable(fn):
        return ""
    case = BEST_CASES[0]
    got, error, _running = apicheck.call(fn, case)
    if isinstance(error, NoEmbed):
        return ""
    if error is not None:
        return yours.figure(w["title"], f'<p style="color:var(--ea-loss-text)">{html.escape(w["error"])}</p>', alt=w["title"])
    question, texts = case.args[1], case.args[2]
    picked = texts.index(got) if got in texts else None
    body = _bars(question, texts, match_scores(0), seen.lang, picked, w["mark"])
    if picked is None:
        body += f'<p style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-loss-text)">{html.escape(w["nothing"].format(got=repr(got)[:80]))}</p>'
    body += _label(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


_COMMON = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
                "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."},
}

EXERCISES = {
    "embed": ApiBox(
        name="embed",
        picture=embed_picture,
        function="embed",
        start=_START_EMBED,
        solution=SOLUTION_EMBED,
        cases=EMBED_CASES,
        texts={
            **_COMMON,
            "empty": {"en": "Replace the three `...`: the texts as the input, the endpoint for embeddings, and the list of embeddings.",
                      "nl": "Vervang de drie `...`: de teksten als input, het endpoint voor embeddings, en de lijst met embeddings."},
            "correct": {"en": "Right: your `embed` sends all texts in one request to `/embeddings`, and gives back one embedding per text, in order.",
                        "nl": "Goed: je `embed` stuurt alle teksten in één request naar `/embeddings`, en geeft één embedding per tekst terug, op volgorde."},
            "http-400-input": {"en": "With `{call}`, the server answered with status 400: \"{message}\" Put the list `texts` the function gets under `\"input\"`.",
                               "nl": "Met `{call}` antwoordde de server met status 400: \"{message}\" Zet de lijst `texts` die de functie krijgt onder `\"input\"`."},
            "http-400": {"en": "With `{call}`, the server answered with status 400: \"{message}\" Does your request go to the endpoint for embeddings, with the texts under `\"input\"`?",
                         "nl": "Met `{call}` antwoordde de server met status 400: \"{message}\" Gaat je request naar het endpoint voor embeddings, met de teksten onder `\"input\"`?"},
            "http-404": {"en": "With `{call}`, the server answered with status 404: \"{message}\" Is the address right? The endpoint for embeddings is `/embeddings`, after the `url` the function gets.",
                         "nl": "Met `{call}` antwoordde de server met status 404: \"{message}\" Klopt het adres? Het endpoint voor embeddings is `/embeddings`, na de `url` die de functie krijgt."},
            "http-404-model": {"en": "With `{call}`, the server answered with status 404: \"{message}\" Keep the name of the model as it was.",
                               "nl": "Met `{call}` antwoordde de server met status 404: \"{message}\" Laat de naam van het model staan zoals hij was."},
            "one-text": {"en": "With `{call}`, your function sent only the first text. Put the whole list `texts` under `\"input\"`: one request gives the embeddings of all of them.",
                         "nl": "Met `{call}` stuurde je functie alleen de eerste tekst. Zet de hele lijst `texts` onder `\"input\"`: één request geeft de embeddings van allemaal."},
            "other-input": {"en": "With `{call}`, the server did not get the list `texts` under `\"input\"`. Send the texts the function gets, as they are.",
                            "nl": "Met `{call}` kreeg de server niet de lijst `texts` onder `\"input\"`. Stuur de teksten die de functie krijgt, zoals ze zijn."},
            "whole": {"en": "With `{call}`, your function gives back the whole JSON of the answer. Give back only the embeddings: one list of numbers for each item under `\"data\"`.",
                      "nl": "Met `{call}` geeft je functie de hele JSON van het antwoord terug. Geef alleen de embeddings terug: één lijst met getallen voor elk item onder `\"data\"`."},
            "one-item": {"en": "With `{call}`, your function gives back one item under `\"data\"`: a dictionary with the embedding of one text. Give back a list with the `\"embedding\"` of every item.",
                         "nl": "Met `{call}` geeft je functie één item onder `\"data\"` terug: een dictionary met de embedding van één tekst. Geef een lijst terug met de `\"embedding\"` van elk item."},
            "not-numbers": {"en": "With `{call}`, your function gives back the embeddings as text, not as lists of numbers: the request asked for another `\"encoding_format\"`. Leave that out, so that each embedding comes as a list of 384 numbers you can compute with.",
                            "nl": "Met `{call}` geeft je functie de embeddings als tekst terug, niet als lijsten met getallen: het request vroeg om een ander `\"encoding_format\"`. Laat dat weg, zodat elke embedding komt als een lijst van 384 getallen waarmee je kunt rekenen."},
            "data": {"en": "With `{call}`, your function gives back the items under `\"data\"`, each a dictionary with `\"object\"`, `\"index\"` and `\"embedding\"`. Give back only the `\"embedding\"` of each item.",
                     "nl": "Met `{call}` geeft je functie de items onder `\"data\"` terug, elk een dictionary met `\"object\"`, `\"index\"` en `\"embedding\"`. Geef alleen de `\"embedding\"` van elk item terug."},
            "first": {"en": "With `{call}`, your function gives back one list of numbers: the embedding of one text. Give back a list with the embedding of every text, one for each item under `\"data\"`.",
                      "nl": "Met `{call}` geeft je functie één lijst met getallen terug: de embedding van één tekst. Geef een lijst terug met de embedding van elke tekst, één voor elk item onder `\"data\"`."},
            "none": {"en": "With `{call}`, your function gives back nothing. End with `return` and the list of embeddings.",
                     "nl": "Met `{call}` geeft je functie niets terug. Eindig met `return` en de lijst met embeddings."},
            "other": {"en": "With `{call}`, your function gives `{got}`, not the embeddings the server sent back.",
                      "nl": "Met `{call}` geeft je functie `{got}`, niet de embeddings die de server terugstuurde."},
        },
    ),
    "best_match": ApiBox(
        name="best_match",
        picture=match_picture,
        function="best_match",
        start=_START_BEST,
        solution=SOLUTION_BEST,
        cases=BEST_CASES,
        names={"embed": run_code(SOLUTION_EMBED).names["embed"]},
        mistakes=[
            Mistake("no-embed", error=lambda e: isinstance(e, NoEmbed),
                    en="This exercise uses your `embed` from exercise 1. Finish exercise 1 first; this check then runs again by itself.",
                    nl="Deze oefening gebruikt je `embed` uit oefening 1. Maak eerst oefening 1 af; deze check draait daarna vanzelf opnieuw."),
            Mistake("short", error=lambda e: isinstance(e, IndexError),
                    en="Your function asks `vectors` for a place it does not have. `vectors` holds one embedding for each text you sent: is the question in the list too, in front of the texts?",
                    nl="Je functie vraagt `vectors` om een plek die er niet is. `vectors` bevat één embedding voor elke tekst die je stuurde: staat de vraag ook in de lijst, vóór de teksten?"),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "Replace the two `...`: the embeddings of the question and the texts, in one request, and the score of each text.",
                      "nl": "Vervang de twee `...`: de embeddings van de vraag en de teksten, in één request, en de score van elke tekst."},
            "correct": {"en": "Right: your `best_match` embeds the question and the texts in one request, and picks the text whose embedding points most the same way as the question's.",
                        "nl": "Goed: je `best_match` haalt de embeddings van de vraag en de teksten in één request op, en kiest de tekst waarvan de embedding het meest dezelfde kant op wijst als die van de vraag."},
            "almost": {"en": "Right text with `{call}`. But your function sent more than one request to the server. Put the question and all the texts in one list, so that one request gives all the embeddings: that is quicker, and a server counts fewer requests.",
                       "nl": "Goede tekst met `{call}`. Maar je functie stuurde meer dan één request naar de server. Zet de vraag en alle teksten in één lijst, zodat één request alle embeddings geeft: dat is sneller, en een server telt minder requests."},
            "many": {"en": "With `{call}`, your function sent more than one request, and gives `{got}`, not the text that fits best. Send the question and the texts in one list, the question first.",
                     "nl": "Met `{call}` stuurde je functie meer dan één request, en geeft hij `{got}`, niet de tekst die het best past. Stuur de vraag en de teksten in één lijst, de vraag eerst."},
            "question-last": {"en": "With `{call}`, the question comes last in the list your function sent, so `vectors[0]` is the embedding of the first text, not of the question. Put the question first.",
                              "nl": "Met `{call}` staat de vraag achteraan in de lijst die je functie stuurde, dus `vectors[0]` is de embedding van de eerste tekst, niet van de vraag. Zet de vraag vooraan."},
            "no-question": {"en": "With `{call}`, your function did not send the question. Without its embedding there is nothing to compare the texts with: put the question in the list, first.",
                            "nl": "Met `{call}` stuurde je functie de vraag niet mee. Zonder haar embedding is er niets om de teksten mee te vergelijken: zet de vraag in de lijst, vooraan."},
            "other-input": {"en": "With `{call}`, the server did not get the question and the texts as one list. Use your `embed` with the question first and then the texts.",
                            "nl": "Met `{call}` kreeg de server de vraag en de teksten niet als één lijst. Gebruik je `embed` met eerst de vraag en dan de teksten."},
            "off-by-one": {"en": "With `{call}`, your function gives the first text, `{got}`, while another fits better. Which vector belongs to `texts[i]`? The question is in front, at place 0.",
                           "nl": "Met `{call}` geeft je functie de eerste tekst, `{got}`, terwijl een andere beter past. Welke vector hoort bij `texts[i]`? De vraag staat vooraan, op plek 0."},
            "worst": {"en": "With `{call}`, your function gives `{got}`, the text that fits the question worst. The higher the dot product, the more two embeddings point the same way: keep the text with the highest score. A distance works the other way round: there the smallest is the closest.",
                      "nl": "Met `{call}` geeft je functie `{got}`, de tekst die het slechtst bij de vraag past. Hoe hoger het inproduct, hoe meer twee embeddings dezelfde kant op wijzen: houd de tekst met de hoogste score over. Een afstand werkt andersom: daar is de kleinste het dichtstbij."},
            "none": {"en": "With `{call}`, your function gives `None`: `best` was never set to a text. Does each `score` come out as a number, and does `if score > best_score:` still keep the highest?",
                     "nl": "Met `{call}` geeft je functie `None`: `best` werd nooit een tekst. Komt er voor elke `score` een getal uit, en houdt `if score > best_score:` nog steeds de hoogste over?"},
            "other": {"en": "With `{call}`, your function gives `{got}`, not the text that fits the question best. Is the score the dot product of the question's vector with that text's vector?",
                      "nl": "Met `{call}` geeft je functie `{got}`, niet de tekst die het best bij de vraag past. Is de score het inproduct van de vector van de vraag met de vector van die tekst?"},
        },
    ),
}

REQUIRED = ["embed", "best_match", "no-clear-match", "upload"]

CASES = [
    ("embed", _START_EMBED, "empty", None),
    ("embed", SOLUTION_EMBED, "correct", None),
    ("embed", SOLUTION_EMBED.replace('"input": texts', '"input": texts[0]'), "wrong", "one-text"),
    ("embed", SOLUTION_EMBED.replace('"input": texts', '"input": ", ".join(texts)'), "wrong", "other-input"),
    ("embed", SOLUTION_EMBED.replace('"input": texts', '"input": None'), "error", "http-400-input"),
    ("embed", SOLUTION_EMBED.replace('"/embeddings"', '"/chat/completions"'), "error", "http-400"),
    ("embed", SOLUTION_EMBED.replace('"/embeddings"', '"/embedding"'), "error", "http-404"),
    ("embed", SOLUTION_EMBED.replace('"granite-embedding-107m-multilingual"', '"text-embedding-3-small"'), "error", "http-404-model"),
    ("embed", SOLUTION_EMBED.replace('return [item["embedding"] for item in answer["data"]]', "return answer"), "wrong", "whole"),
    ("embed", SOLUTION_EMBED.replace('return [item["embedding"] for item in answer["data"]]', 'return answer["data"]'), "wrong", "data"),
    ("embed", SOLUTION_EMBED.replace('return [item["embedding"] for item in answer["data"]]', 'return answer["data"][0]["embedding"]'), "wrong", "first"),
    ("embed", SOLUTION_EMBED.replace('return [item["embedding"] for item in answer["data"]]', 'return answer["data"][0]'), "wrong", "one-item"),
    ("embed", SOLUTION_EMBED.replace('"input": texts', '"input": texts, "encoding_format": "base64"'), "wrong", "not-numbers"),
    ("embed", SOLUTION_EMBED.replace('return [item["embedding"] for item in answer["data"]]', '[item["embedding"] for item in answer["data"]]'), "wrong", "none"),
    ("embed", SOLUTION_EMBED.replace('return [item["embedding"] for item in answer["data"]]', 'return [item["index"] for item in answer["data"]]'), "wrong", "other"),
    ("best_match", _START_BEST, "empty", None),
    ("best_match", SOLUTION_BEST, "correct", None),
    ("best_match", SOLUTION_BEST.replace("score = np.dot(q, vectors[i + 1])",
                                         "score = np.dot(q, vectors[i + 1]) / (np.linalg.norm(q) * np.linalg.norm(vectors[i + 1]))"), "correct", None),
    ("best_match", SOLUTION_BEST.replace("    vectors = embed(url, [question] + texts)\n    q = np.array(vectors[0])\n",
                                         "    vectors = [None] + embed(url, texts)\n    q = np.array(embed(url, [question])[0])\n"), "almost", "almost"),
    ("best_match", SOLUTION_BEST.replace("embed(url, [question] + texts)", "embed(url, texts) + embed(url, [question])"), "wrong", "many"),
    ("best_match", SOLUTION_BEST.replace("embed(url, [question] + texts)", "embed(url, texts + [question])"), "wrong", "question-last"),
    ("best_match", SOLUTION_BEST.replace("embed(url, [question] + texts)", "embed(url, texts)"), "error", "short"),
    ("best_match", SOLUTION_BEST.replace("embed(url, [question] + texts)", "embed(url, texts)").replace("vectors[i + 1]", "vectors[i]"), "wrong", "no-question"),
    ("best_match", SOLUTION_BEST.replace("embed(url, [question] + texts)", "embed(url, [question] + texts + [question])"), "wrong", "other-input"),
    ("best_match", SOLUTION_BEST.replace("vectors[i + 1]", "vectors[i]"), "wrong", "off-by-one"),
    ("best_match", SOLUTION_BEST.replace("score = np.dot(q, vectors[i + 1])", "score = -np.dot(q, vectors[i + 1])"), "wrong", "worst"),
    ("best_match", SOLUTION_BEST.replace("score = np.dot(q, vectors[i + 1])", "score = np.linalg.norm(q - np.array(vectors[i + 1]))"), "wrong", "worst"),
    ("best_match", SOLUTION_BEST.replace("if score > best_score:", "if score < best_score:"), "wrong", "none"),
    ("best_match", SOLUTION_BEST.replace("for i in range(len(texts)):", "for i in range(len(texts) - 1):"), "wrong", "other"),
]


# --- the demo: the learner's best_match, against the course's server with Granite --------------------------------------

def size_mb() -> int:
    """The size of Granite's two files, in MB."""
    from everything_ai import hub
    return round(sum(hub.PINNED[pin]["size"] for pin in FILES) / 1e6)


def downloaded() -> bool:
    from everything_ai import hub
    return all(hub.pinned(*pin) is not None for pin in FILES)


def download_or_stop(button, texts: dict) -> None:
    lesson_helpers.pins_or_stop(button, texts, FILES)


def learner_best_match(text, embed_text):
    """The learner's best_match from exercise 2 with their embed, when both are right; else None."""
    inputs = best_inputs(embed_text)
    if inputs["embed"] is no_embed or text is MISSING or text is None:
        return None
    if EXERCISES["best_match"].check(text, inputs).state != "correct":
        return None
    return run_code(text, inputs).names.get("best_match")


def demo(best_match, question: str, words: dict, lang: str) -> str:
    """The learner's best_match for `question` against the six help pages, sent to the course's server: every page's
    score, from the embeddings the server sent back, the page the learner's function picked, and the time."""
    import time
    running = serve.model_server()
    before = len(running.log)
    start = time.perf_counter()
    try:
        got = best_match(running.url, question, list(PAGES))
    except Exception as e:   # the learner's code, against the real server
        said = words["memory"] if serve.out_of_memory(e) else words["failed"].format(why=f"{type(e).__name__}: {e}")
        return f"<p><em>{html.escape(said)}</em></p>"
    took = time.perf_counter() - start
    answers = [entry["answer"] for entry in running.log[before:]
               if entry.get("path", "").rstrip("/") == "/v1/embeddings" and isinstance(entry.get("answer"), dict)
               and len(entry["answer"].get("data", [])) == len(PAGES) + 1]
    body = ""
    if answers:
        v = np.array([d["embedding"] for d in sorted(answers[-1]["data"], key=lambda d: d["index"])])
        scores = [float(x) for x in v[1:] @ v[0]]
        picked = PAGES.index(got) if got in PAGES else None
        body = _bars(question, PAGES, scores, lang, picked, words["mark"])
    body += _label(words["done"].format(got=got, seconds=_nr(took, lang, 2)))
    return yours.figure(words["title"], body, alt=words["title"])


def no_clear_facts(lang: str) -> dict:
    """The numbers of exercise 3: the scores of the question about calling, and the clear match of the first question."""
    s = search_scores()
    calling = sorted(s[2], reverse=True)
    first = sorted(s[0], reverse=True)
    return {"low": _nr(min(s[2]), lang), "high": _nr(calling[0], lang), "gap": _nr(calling[0] - calling[1], lang),
            "clear": _nr(first[0], lang), "next": _nr(first[1], lang),
            "best": PAGES[max(range(len(PAGES)), key=lambda k: s[2][k])]}
