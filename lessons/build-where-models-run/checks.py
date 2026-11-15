# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-where-models-run: one function that reaches a model wherever it runs, by its base URL and its
model's name, against two stand-ins with other names (everything_ai.serve.StandIn with `model`); how much memory the
weights of a model take; the course's measured speed of Qwen3-0.6B on one thread, in float32 and in bfloat16
(data/course-apis-qwen.json); the pictures; and the demo, the learner's function against the course's own server."""
import html
import json
import time

from everything_ai import apicheck, chat, paths, serve, yours
from everything_ai.apicheck import URL, ApiBox, Case
from everything_ai.check import MISSING, FunctionBox, Mistake, run_code

DOWNLOADS = {"Qwen3-0.6B": chat.FILES}   # what the demo downloads, for the line under the header

STORY = "Write a short story of five sentences about a lighthouse keeper."
RECORD = {
    "speed32": {"speed": "float32", "messages": [{"role": "user", "content": STORY}], "max_tokens": 100},
    "speed16": {"speed": "bfloat16", "messages": [{"role": "user", "content": STORY}], "max_tokens": 100},
}
DATA = paths.ROOT / "data" / "course-apis-qwen.json"
QUESTION = "Name three colors of the rainbow."
ANSWERS = {"qwen3-0.6b": "Red, orange and yellow.", "llama3.2:1b": "Red, green and blue."}


def speeds() -> dict:
    """Tokens per second of Qwen3-0.6B on one thread of the course's computer, in float32 and in bfloat16."""
    part = json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-where-models-run"]
    return {"float32": part["speed32"]["answers"][0]["per_second"], "bfloat16": part["speed16"]["answers"][0]["per_second"]}


def file_sizes() -> dict:
    """Qwen3-0.6B's weights, from the shapes of its tensors (data/qwen3-0.6b-base.json), and the numbers in the chat
    model's download, which stores the table of token vectors a second time, as the layer that picks the next token
    (lm_head.weight, the same values; everything_ai.qwen3 leaves it out)."""
    import math
    from everything_ai import qwen3
    tensors = qwen3.header()["tensors"]
    weights = sum(math.prod(t["shape"]) for t in tensors.values())
    table = math.prod(tensors["model.embed_tokens.weight"]["shape"])
    return {"weights": weights, "table": table, "file": weights + table}


def _thousands(x: int, lang: str) -> str:
    s = f"{x:,}"
    return s.replace(",", ".") if lang == "nl" else s


def numbers(lang: str) -> dict:
    """The speeds and the sizes as the lesson's text writes them, in the lesson's language."""
    f = file_sizes()
    return {**{k: _n(v, lang) for k, v in speeds().items()}, **{k: _thousands(v, lang) for k, v in f.items()},
            "weights_gb": _n(gb(f["weights"], 16), lang), "file_gb": _n(gb(f["file"], 16), lang)}


def machine() -> str:
    return json.loads(DATA.read_text(encoding="utf-8"))["machine"]


def _n(x, lang: str, digits: int = 1) -> str:
    s = f"{x:.{digits}f}"
    return s.replace(".", ",") if lang == "nl" else s


def gb(params: float, bits: int) -> float:
    return params * bits / 8 / 1e9


_SIZE_WORDS = {
    "en": {"title": "How much memory the weights take", "kind": ["float32, 32 bits", "bfloat16, 16 bits", "4 bits"],
           "models": [("Qwen3-0.6B", 0.6e9), ("a model of 8 billion", 8e9), ("a model of 70 billion", 70e9)], "laptop": "16 GB: the memory of a good laptop"},
    "nl": {"title": "Hoeveel geheugen de weights innemen", "kind": ["float32, 32 bits", "bfloat16, 16 bits", "4 bits"],
           "models": [("Qwen3-0.6B", 0.6e9), ("een model van 8 miljard", 8e9), ("een model van 70 miljard", 70e9)], "laptop": "16 GB: het geheugen van een goede laptop"},
}


def sizes_picture(lang: str) -> str:
    """For three sizes of model, the memory of their weights at 32, 16 and 4 bits per weight, on a log-free bar that
    is cut at 160 GB, with a line at 16 GB."""
    w = _SIZE_WORDS[lang]
    W, L, R = 720, 190, 70
    top = 160.0
    scale = (W - L - R) / top
    parts, y = [], 18
    for name, params in w["models"]:
        parts.append(f'<text x="0" y="{y + 12}" style="font-size:12px;font-family:var(--ea-font-ui);fill:var(--ea-ink)">{html.escape(name)}</text>')
        for kind, bits in zip(w["kind"], (32, 16, 4)):
            size = gb(params, bits)
            width = min(size, top) * scale
            parts.append(f'<text x="{L - 8}" y="{y + 11}" text-anchor="end" style="font-size:11px;font-family:var(--ea-font-ui);fill:var(--ea-ink-2)">{html.escape(kind)}</text>')
            parts.append(f'<rect x="{L}" y="{y + 2}" width="{max(width, 1.5):.1f}" height="11" rx="2" style="fill:var(--ea-ink-3)"/>')
            label = f"{_n(size, lang)} GB" + (" →" if size > top else "")
            parts.append(f'<text x="{L + min(width, (W - L - R)) + 6:.1f}" y="{y + 11}" style="font-size:11px;font-family:var(--ea-font-code);fill:var(--ea-ink)">{html.escape(label)}</text>')
            y += 16
        y += 12
    x16 = L + 16 * scale
    parts.append(f'<line x1="{x16:.1f}" x2="{x16:.1f}" y1="10" y2="{y - 6}" style="stroke:var(--ea-ink);stroke-dasharray:3 3"/>')
    parts.append(f'<text x="{x16 + 4:.1f}" y="{y + 8}" style="font-size:11px;font-family:var(--ea-font-ui);fill:var(--ea-ink-2)">{html.escape(w["laptop"])}</text>')
    svg = f'<svg viewBox="0 0 {W} {y + 16}" role="img" style="width:100%;max-width:46rem;height:auto"><title>{html.escape(w["title"])}</title>{"".join(parts)}</svg>'
    return yours.figure(w["title"], svg, alt=w["title"])


# --- exercise 1: one function, any place --------------------------------------------------------------------------------

_START_ASK = '''import json
import urllib.request


def ask(base_url, model, question):
    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": question}], "max_tokens": 50}
    data = json.dumps(body).encode()
    request = urllib.request.Request("http://127.0.0.1:8765/v1/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        answer = json.loads(response.read())
    return answer["choices"][0]["message"]["content"]
'''
SOLUTION_ASK = _START_ASK.replace('"model": "qwen3-0.6b"', '"model": model').replace(
    '"http://127.0.0.1:8765/v1/chat/completions"', 'base_url + "/chat/completions"')


def _judge(model):
    def judge(got, running):
        return None if got == ANSWERS[model] else "other"
    return judge


ASK_CASES = [Case(lambda model=model, answer=answer: serve.StandIn({QUESTION: answer}, model=model), (URL, model, QUESTION), _judge(model))
             for model, answer in ANSWERS.items()]


def ask_picture(seen) -> str:
    """The learner's ask against both stand-ins: where each request went, which model it asked for, what came back."""
    run = seen.run
    fn = run.names.get("ask") if run is not None and run.error is None else None
    if not callable(fn):
        return ""
    w = {"en": {"title": "Your ask, at two places", "place": "A server with {model}", "sent": "your request went to {where}, for the model {asked}",
                "back": "status {status}: {said}", "nowhere": "your request did not reach this server"},
         "nl": {"title": "Je ask, op twee plekken", "place": "Een server met {model}", "sent": "je request ging naar {where}, voor het model {asked}",
                "back": "status {status}: {said}", "nowhere": "je request bereikte deze server niet"}}[seen.lang]
    rows = []
    for (model, _answer), case in zip(ANSWERS.items(), ASK_CASES):
        _got, _error, running = apicheck.call(fn, case)
        posts = [e for e in running.log if e["method"] == "POST"]
        lines = [f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink);font-weight:600">{html.escape(w["place"].format(model=model))}</div>']
        if not posts:
            lines.append(f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-loss-text)">{html.escape(w["nowhere"])}</div>')
        else:
            e = posts[0]
            asked = e["body"].get("model") if isinstance(e["body"], dict) else "?"
            lines.append(f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)">'
                         f'{html.escape(w["sent"].format(where=running.url.split("/v1")[0] + e["path"], asked=asked))}</div>')
            answer = e.get("answer") or {}
            said = (answer.get("choices") or [{}])[0].get("message", {}).get("content") if e["status"] == 200 else (answer.get("error") or {}).get("message", "")
            colour = "var(--ea-pred-text)" if e["status"] == 200 else "var(--ea-loss-text)"
            lines.append(f'<div style="font-family:var(--ea-font-code);font-size:12px;color:{colour}">{html.escape(w["back"].format(status=e["status"], said=said))}</div>')
        rows.append(f'<div style="padding:0.3rem 0;border-top:1px solid var(--ea-line)">{"".join(lines)}</div>')
    return yours.figure(w["title"], "".join(rows), alt=w["title"])


def learner_ask(text):
    """The learner's ask from exercise 1, when it is right; else None."""
    if text is MISSING or text is None or EXERCISES["ask"].check(text).state != "correct":
        return None
    return run_code(text).names.get("ask")


# --- exercise 2: the memory of the weights ------------------------------------------------------------------------------

_START_GB = '''def weights_gb(params, bits):
    return 0
'''
SOLUTION_GB = '''def weights_gb(params, bits):
    return params * bits / 8 / 1_000_000_000
'''
GB_CASES = [(600_000_000, 32), (600_000_000, 16), (8_000_000_000, 4), (70_000_000_000, 16), (70_000_000_000, 4), (1_000_000_000, 8)]


def gb_picture(seen) -> str:
    """What the learner's weights_gb says for Qwen3-0.6B and for a model of 70 billion, at 32, 16 and 4 bits."""
    run = seen.mine
    fn = run.names.get("weights_gb") if run is not None else None
    if not callable(fn):
        return ""
    w = {"en": ("Your weights_gb", "Qwen3-0.6B", "70 billion", "bits", "should be", "error"),
         "nl": ("Je weights_gb", "Qwen3-0.6B", "70 miljard", "bits", "zou zijn", "fout")}[seen.lang]
    rows = []
    for label, params in ((w[1], 600_000_000), (w[2], 70_000_000_000)):
        for bits in (32, 16, 4):
            right = gb(params, bits)
            try:
                got = float(fn(params, bits))
                mine, ok = _n(got, seen.lang) + " GB", abs(got - right) <= 1e-6 * right
            except Exception:   # the learner's code
                mine, ok = w[5], False
            colour = "var(--ea-ink)" if ok else "var(--ea-loss-text)"
            rows.append(f'<tr><td style="padding:0.1rem 0.6rem 0.1rem 0">{html.escape(label)}</td><td>{bits} {w[3]}</td>'
                        f'<td style="font-family:var(--ea-font-code);color:{colour};padding-left:0.6rem">{html.escape(mine)}</td>'
                        f'<td style="font-family:var(--ea-font-code);color:var(--ea-ink-2);padding-left:0.6rem">{html.escape(w[4])} {_n(right, seen.lang)} GB</td></tr>')
    body = f'<table style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink);border-collapse:collapse">{"".join(rows)}</table>'
    return yours.figure(w[0], body, alt=w[0])


_COMMON = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
                "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."},
}

EXERCISES = {
    "ask": ApiBox(
        name="ask",
        picture=ask_picture,
        function="ask",
        url_name="base_url",
        start=_START_ASK,
        solution=SOLUTION_ASK,
        cases=ASK_CASES,
        texts={
            **_COMMON,
            "empty": {"en": "`ask` still always goes to the course's server, for `\"qwen3-0.6b\"`. Use the `base_url` and the `model` it gets instead.",
                      "nl": "`ask` gaat nog altijd naar de server van de course, voor `\"qwen3-0.6b\"`. Gebruik in plaats daarvan de `base_url` en het `model` die hij krijgt."},
            "correct": {"en": "Right: your `ask` reached both servers, each at its own address and with its own model. Only `base_url` and `model` changed; the rest of the call stayed the same.",
                        "nl": "Goed: je `ask` bereikte beide servers, elk op zijn eigen adres en met zijn eigen model. Alleen `base_url` en `model` veranderden; de rest van de aanroep bleef hetzelfde."},
            "http-404-model": {"en": "With `{call}`, the server answered 404: \"{message}\" Put the `model` your function gets in the body.",
                               "nl": "Met `{call}` antwoordde de server 404: \"{message}\" Zet het `model` dat je functie krijgt in de body."},
            "http-404": {"en": "With `{call}`, the server answered 404: \"{message}\" The address is `base_url + \"/chat/completions\"`: `base_url` already ends in `/v1`.",
                         "nl": "Met `{call}` antwoordde de server 404: \"{message}\" Het adres is `base_url + \"/chat/completions\"`: `base_url` eindigt al op `/v1`."},
            "unreachable": {"en": "With `{call}`, your function could not reach a server ({why}). The address must start with the `base_url` your function gets: `base_url + \"/chat/completions\"`.",
                            "nl": "Met `{call}` kon je functie geen server bereiken ({why}). Het adres moet beginnen met de `base_url` die je functie krijgt: `base_url + \"/chat/completions\"`."},
            "other": {"en": "With `{call}`, your function gave `{got}`. Give back the text of the answer, as before.",
                      "nl": "Met `{call}` gaf je functie `{got}`. Geef de tekst van het antwoord terug, zoals eerst."},
        },
    ),
    "weights_gb": FunctionBox(
        name="weights_gb",
        picture=gb_picture,
        function="weights_gb",
        start=_START_GB,
        solution=SOLUTION_GB,
        visible=GB_CASES[:1],
        hidden=GB_CASES[1:],
        mistakes=[
            Mistake("bits", wrong=SOLUTION_GB.replace(" / 8", ""),
                    en="`{call}` gives `{got}`, eight times too much: `bits` are bits, and a byte has 8. Divide by 8 for the bytes.",
                    nl="`{call}` geeft `{got}`, acht keer te veel: `bits` zijn bits, en een byte heeft er 8. Deel door 8 voor de bytes."),
            Mistake("bytes", wrong=SOLUTION_GB.replace(" / 1_000_000_000", ""),
                    en="`{call}` gives `{got}`: that is the number of bytes. A gigabyte is a billion bytes: divide by `1_000_000_000`.",
                    nl="`{call}` geeft `{got}`: dat is het aantal bytes. Een gigabyte is een miljard bytes: deel door `1_000_000_000`."),
            Mistake("gib", wrong=SOLUTION_GB.replace(" / 1_000_000_000", " / 1024 ** 3"),
                    en="`{call}` gives `{got}`, a little less than `{want}`: you divided by `1024 ** 3`. That makes a gibibyte, GiB, which some programs show as GB; a gigabyte is a billion bytes: divide by `1_000_000_000`.",
                    nl="`{call}` geeft `{got}`, iets minder dan `{want}`: je deelde door `1024 ** 3`. Dat geeft een gibibyte, GiB, die sommige programma's als GB tonen; een gigabyte is een miljard bytes: deel door `1_000_000_000`."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "`weights_gb` gives 0 yet. Each weight takes `bits` bits; how many bytes is that for `params` weights, and how many gigabytes?",
                      "nl": "`weights_gb` geeft nog 0. Elke weight neemt `bits` bits in; hoeveel bytes is dat voor `params` weights, en hoeveel gigabyte?"},
            "correct": {"en": "Right: your `weights_gb` turns weights and bits into gigabytes, for all {n} models the check tried.",
                        "nl": "Goed: je `weights_gb` maakt van weights en bits gigabytes, voor alle {n} modellen die de check probeerde."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: 600 million weights of 4 bytes each.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: 600 miljoen weights van elk 4 bytes."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`.",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn."},
        },
    ),
}

REQUIRED = ["ask", "weights_gb", "where"]

CASES = [
    ("ask", _START_ASK, "empty", None),
    ("ask", SOLUTION_ASK, "correct", None),
    ("ask", _START_ASK.replace('"http://127.0.0.1:8765/v1/chat/completions"', 'base_url + "/chat/completions"'), "error", "http-404-model"),
    ("ask", SOLUTION_ASK.replace('base_url + "/chat/completions"', 'base_url + "/v1/chat/completions"'), "error", "http-404"),
    ("ask", _START_ASK.replace('"model": "qwen3-0.6b"', '"model": model').replace('"http://127.0.0.1:8765/v1/chat/completions"', '"http://127.0.0.1:9/v1/chat/completions"'), "error", "unreachable"),
    ("weights_gb", _START_GB, "empty", None),
    ("weights_gb", SOLUTION_GB, "correct", None),
    ("weights_gb", SOLUTION_GB.replace(" / 8", ""), "wrong", "bits"),
    ("weights_gb", SOLUTION_GB.replace(" / 1_000_000_000", ""), "wrong", "bytes"),
    ("weights_gb", SOLUTION_GB.replace(" / 1_000_000_000", " / 1024 ** 3"), "wrong", "gib"),
]


# --- the demo: your ask, at the course's own server -----------------------------------------------------------------------

def downloaded() -> bool:
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    chat.download_or_stop(button, texts)


def demo(ask, question: str, words: dict, lang: str) -> str:
    """The learner's ask, pointed at the course's own server with Qwen3 behind it: a local runtime on this computer."""
    running = serve.model_server()
    start = time.perf_counter()
    try:
        text = ask(running.url, serve.MODEL, question)
    except Exception as e:   # the learner's code, against the real server; it may have run out of memory
        said = words["memory"] if serve.out_of_memory(e) else words["failed"].format(why=f"{type(e).__name__}: {e}")
        return f"<p><em>{html.escape(said)}</em></p>"
    seconds = _n(time.perf_counter() - start, lang)
    return chat.answer_picture(str(text), words["done"].format(url=running.url, seconds=seconds), words["title"])
