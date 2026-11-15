# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-generation-hubs: a function that asks the course's server for an image and gives back its bytes,
against a stand-in of the server, and a pipeline around the course's digit reader; the pictures, the measured numbers
of the lesson, and the demo with the learner's own function against the course's server."""
import base64
import functools
import html
import io
import json
import math
import re

import numpy as np

from everything_ai import apicheck, drawing, paths, serve, yours
from everything_ai.apicheck import URL, ApiBox, Case
from everything_ai.check import MISSING, FunctionBox, Mistake, Timeout, run_code, with_time_limit

WORKED = "two red circles and a blue square"   # the prompt of the worked example
PNG = b"\x89PNG\r\n\x1a\n"                       # how every PNG file starts
DIGITS = paths.ROOT / "data" / "qmnist-small.npz"
READER = paths.ROOT / "data" / "course-qmnist-cnn.pt"   # the course's digit reader, as in torch-pretrained
SHOWN = [0, 36, 111, 62]   # the test digits of exercise 2: a sure one, two less sure, and one the reader gets wrong


def _nr(x: float, lang: str, digits: int = 3) -> str:
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def _label(text: str) -> str:
    return f'<p style="margin:6px 0;font-size:13px;color:var(--ea-ink-2)">{html.escape(text)}</p>'


# --- exercise 1: an image through the API ------------------------------------------------------------------------------

_START_IMAGE = '''import base64
import json
import urllib.request


def image_bytes(url, prompt):
    body = {"model": "course-shapes", "prompt": ..., "size": "256x256"}
    request = urllib.request.Request(url + ..., data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        answer = json.loads(response.read())
    return ...
'''
SOLUTION_IMAGE = '''import base64
import json
import urllib.request


def image_bytes(url, prompt):
    body = {"model": "course-shapes", "prompt": prompt, "size": "256x256"}
    request = urllib.request.Request(url + "/images/generations", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        answer = json.loads(response.read())
    return base64.b64decode(answer["data"][0]["b64_json"])
'''
_PROMPTS = [WORKED, "drie gele driehoeken en een paars vierkant"]


def _posts(running) -> list:
    return [e for e in running.log if e["method"] == "POST" and isinstance(e["body"], dict)]


def _judge_image(prompt: str):
    def judge(got, running):
        posts = _posts(running)
        sent = posts[0]["body"].get("prompt") if posts else None
        if sent != prompt:
            return "other-prompt"
        answer = posts[0]["answer"]
        items = answer.get("data") if isinstance(answer, dict) else None
        if not items:
            return "other"
        want = base64.b64decode(items[0]["b64_json"])
        if got == want:
            return None
        if got is None:
            return "none"
        if got == items[0]["b64_json"]:
            return "text"
        if got == answer:
            return "whole"
        if got == items:
            return "data"
        if got == items[0]:
            return "item"
        return "other"
    return judge


IMAGE_CASES = [Case(serve.StandIn, (URL, p), _judge_image(p)) for p in _PROMPTS]

_TRIM = {"en": {"data": "… {n} characters in all", "numbers": "… {n} numbers in all", "thousands": ","},
         "nl": {"data": "… {n} tekens in totaal", "numbers": "… {n} getallen in totaal", "thousands": "."}}


def png_html(data, width: int = 160, alt: str = "") -> str:
    """An image from the bytes of a PNG, `width` pixels wide."""
    src = "data:image/png;base64," + base64.b64encode(data).decode()
    return (f'<img src="{src}" alt="{html.escape(alt)}" width="{width}" height="{width}" '
            'style="display:block;border-radius:6px;border:1px solid var(--ea-line)">')


def image_picture(seen) -> str:
    """What the learner's image_bytes sent and got back, and the image their function gave back, if it is a PNG."""
    words = {"en": {"title": "What your image_bytes sent, and what came back", "request": "Your code sent: {method} {path}",
                    "headers": "Headers:", "answer": "The server sent back: status {status}", "stream": "a stream with {n} pieces",
                    "none": "Your function sent nothing to the server yet.", "gave": "What your function gave back, shown as an image:",
                    "bytes": "{n} bytes, starting with {start}", **_TRIM["en"]},
             "nl": {"title": "Wat je image_bytes stuurde, en wat er terugkwam", "request": "Je code stuurde: {method} {path}",
                    "headers": "Headers:", "answer": "De server stuurde terug: status {status}", "stream": "een stream met {n} stukjes",
                    "none": "Je functie stuurde nog niets naar de server.", "gave": "Wat je functie teruggaf, als afbeelding:",
                    "bytes": "{n} bytes, beginnend met {start}", **_TRIM["nl"]}}[seen.lang]
    out = apicheck.exchange(seen, "image_bytes", IMAGE_CASES[0], words)
    run = seen.run
    fn = run.names.get("image_bytes") if run is not None and run.error is None else None
    if out and callable(fn):
        got, _error, _running = apicheck.call(fn, IMAGE_CASES[0])
        if isinstance(got, bytes) and got.startswith(PNG):
            n = f"{len(got):,}".replace(",", words["thousands"])
            out += yours.figure(words["gave"], png_html(got, alt=WORKED) + _label(words["bytes"].format(n=n, start=repr(got[:8]))),
                                alt=words["gave"])
    return out


def worked(lang: str) -> dict:
    """The worked example, sent to a stand-in of the course's server: the request, the answer as it came back, and
    the bytes of the image in it."""
    import urllib.request
    running = serve.start(serve.StandIn())
    try:
        body = {"model": serve.IMAGER, "prompt": WORKED, "size": "256x256"}
        request = urllib.request.Request(running.url + "/images/generations", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            answer = json.loads(response.read())
    finally:
        running.stop()
    return {"request": body, "answer": answer, "png": base64.b64decode(answer["data"][0]["b64_json"])}


_WORKED = {
    "en": {"title": "The course's request for an image", "request": "The course's code sent: POST /v1/images/generations",
           "answer": "The server sent back: status 200", "prompt": "the description", "b64": "the image, in base64",
           "revised": "what the server drew", "decoded": "base64.b64decode gives {n} bytes, starting with {start}: a PNG",
           "alt": "Two red circles and a blue square on a light background, drawn by the course's server"},
    "nl": {"title": "Het request van de course voor een afbeelding", "request": "De code van de course stuurde: POST /v1/images/generations",
           "answer": "De server stuurde terug: status 200", "prompt": "de beschrijving", "b64": "de afbeelding, in base64",
           "revised": "wat de server tekende", "decoded": "base64.b64decode geeft {n} bytes, beginnend met {start}: een PNG",
           "alt": "Twee rode cirkels en een blauw vierkant op een lichte achtergrond, getekend door de server van de course"},
}


def _pre(text: str, marks: dict, model_key: str | None = None) -> str:
    """Code-like text; each line that contains a key of `marks` gets that note beside it, in grey, and the line with
    `model_key` the colour of what a model makes."""
    out = []
    for line in text.split("\n"):
        note = next((n for k, n in marks.items() if k in line), "")
        colour = "var(--ea-pred-text)" if model_key and model_key in line else "var(--ea-ink)"
        out.append(f'<span style="color:{colour}">{html.escape(line)}</span>'
                   + (f'<span style="color:var(--ea-ink-2);font-family:var(--ea-font-ui)">   ← {html.escape(note)}</span>' if note else ""))
    return ('<pre style="margin:4px 0 10px;font-family:var(--ea-font-code);font-size:12.5px;line-height:1.5;'
            'white-space:pre-wrap;overflow-wrap:anywhere">' + "\n".join(out) + "</pre>")


def worked_picture(lang: str) -> str:
    w, ex = _WORKED[lang], worked(lang)
    answer = apicheck.trimmed(ex["answer"], _TRIM[lang])
    body = (_label(w["request"]) + _pre(json.dumps(ex["request"], indent=2), {'"prompt"': w["prompt"]})
            + _label(w["answer"]) + _pre(json.dumps(answer, indent=2, ensure_ascii=False),
                                         {'"b64_json"': w["b64"], '"revised_prompt"': w["revised"]}, '"b64_json"'))
    n = f"{len(ex['png']):,}".replace(",", _TRIM[lang]["thousands"])
    body += _label(w["decoded"].format(n=n, start=repr(ex["png"][:8]))) + png_html(ex["png"], 200, w["alt"])
    return yours.figure(w["title"], body, alt=w["title"])


# --- a video: a job, not an answer ------------------------------------------------------------------------------------

_VIDEO = {"en": {"title": "A video as a job: start it, ask, download", "prompt": "a red ball rolls across a table",
                 "wait": "… 10 seconds later …", "file": "the video, an MP4 file", "sent": "→ what your program sends", "back": "← what the server sends back"},
          "nl": {"title": "Een video als job: starten, vragen, downloaden", "prompt": "een rode bal rolt over een tafel",
                 "wait": "… 10 seconden later …", "file": "de video, een MP4-bestand", "sent": "→ wat je programma stuurt", "back": "← wat de server terugstuurt"}}


def video_picture(lang: str) -> str:
    """The requests of a video job and what comes back: a POST that starts the job, the status asked again and again,
    and the video downloaded once the job has succeeded. A made-up form, not a run: each provider has its own paths and
    words for the status, and the course's server makes no videos."""
    w = _VIDEO[lang]
    steps = [("POST /v1/videos  " + json.dumps({"prompt": w["prompt"]}, ensure_ascii=False),
              json.dumps({"id": "video_abc", "status": "queued"})),
             None,
             ("GET /v1/videos/video_abc", json.dumps({"id": "video_abc", "status": "running", "progress": 40})),
             None,
             ("GET /v1/videos/video_abc", json.dumps({"id": "video_abc", "status": "succeeded", "url": "https://…/video_abc.mp4"}, ensure_ascii=False)),
             ("GET https://…/video_abc.mp4", w["file"])]
    rows = []
    for step in steps:
        if step is None:
            rows.append(f'<div style="margin:2px 0 6px 18px;font-size:12px;color:var(--ea-ink-3)">{html.escape(w["wait"])}</div>')
            continue
        sent, back = step
        rows.append('<div style="font-family:var(--ea-font-code);font-size:12.5px;line-height:1.5;overflow-wrap:anywhere">'
                    f'<div style="color:var(--ea-ink)">→ {html.escape(sent)}</div>'
                    f'<div style="color:var(--ea-pred-text);margin-bottom:4px">← {html.escape(back)}</div></div>')
    legend = _label(f"{w['sent']}; {w['back']}")
    return yours.figure(w["title"], "".join(rows) + legend, alt=w["title"])


# --- exercise 2: a pipeline around the digit reader --------------------------------------------------------------------

class Pixels(list):
    """The 784 pixels of a test digit, from 0 to 255, row by row; a check writes it as `digits[i]`."""

    def __init__(self, values, index: int):
        super().__init__(values)
        self.index = index

    def __repr__(self):
        return f"digits[{self.index}]"


@functools.lru_cache(maxsize=1)
def _test():
    d = np.load(DIGITS)
    return d["test_images"], d["test_labels"]


def digits(i: int) -> Pixels:
    images, _ = _test()
    return Pixels([int(v) for v in images[i]], i)


def label(i: int) -> int:
    return int(_test()[1][i])


@functools.lru_cache(maxsize=1)
def reader():
    """The course's digit reader (everything_ai.models.DigitCNN) with the course's weights, in eval mode; a check writes
    it as `reader`."""
    import torch
    from everything_ai.models import DigitCNN

    class Reader(DigitCNN):
        def __repr__(self):
            return "reader"
    model = Reader()
    model.load_state_dict(torch.load(READER, weights_only=True))
    return model.eval()


_START_DIGIT = '''import torch


def read_digit(model, pixels):
    x = ...
    model.eval()
    with torch.no_grad():
        p = ...
    digit = int(p.argmax())
    return digit, float(p[digit])
'''
SOLUTION_DIGIT = '''import torch


def read_digit(model, pixels):
    x = torch.tensor(pixels, dtype=torch.float32) / 255
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1)[0]
    digit = int(p.argmax())
    return digit, float(p[digit])
'''


def shown(value) -> str:
    """How a check writes what a read_digit gave: a probability with three decimals."""
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], float):
        return f"({value[0]!r}, {value[1]:.3f})"
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + "..."


def reads(fn, indices=SHOWN, seconds: float = 3) -> list:
    """What `fn`, a read_digit, gives for the test digits; None for a digit where it stops with an error or is not done
    after `seconds`, and for every digit after one that was not done."""
    out, stopped = [], False
    for i in indices:
        if stopped:
            out.append(None)
            continue
        try:
            out.append(with_time_limit(lambda: fn(reader(), digits(i)), seconds))
        except Exception as e:   # the learner's code; also Timeout
            out.append(None)
            stopped = isinstance(e, Timeout)
    return out


def digit_png(i: int) -> bytes:
    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray(np.array(digits(i), dtype=np.uint8).reshape(28, 28)).save(buffer, format="PNG")
    return buffer.getvalue()


def _digit_img(i: int, size: int = 56) -> str:
    src = "data:image/png;base64," + base64.b64encode(digit_png(i)).decode()
    return (f'<img src="{src}" alt="" width="{size}" height="{size}" '
            'style="display:block;image-rendering:pixelated;border-radius:4px">')


_DIGIT_WORDS = {"en": {"title": "What your read_digit says about four test digits", "true": "written: {d}",
                       "says": "read_digit gives ({d}, {p})", "error": "read_digit stops with an error",
                       "other": "read_digit gives {got}"},
                "nl": {"title": "Wat je read_digit zegt over vier testcijfers", "true": "geschreven: {d}",
                       "says": "read_digit geeft ({d}, {p})", "error": "read_digit stopt met een fout",
                       "other": "read_digit geeft {got}"}}


def _bar(p: float) -> str:
    width = max(0.0, min(1.0, p)) * 100
    return ('<div style="height:8px;background:var(--ea-line);border-radius:4px;margin-top:3px">'
            f'<div style="height:8px;width:{width:.1f}%;background:var(--ea-pred);border-radius:4px"></div></div>')


def digits_html(results: list, lang: str, indices=SHOWN) -> str:
    w = _DIGIT_WORDS[lang]
    cells = []
    for i, r in zip(indices, results):
        lines = [f'<div style="font-size:12px;color:var(--ea-ink-2)">{html.escape(w["true"].format(d=label(i)))}</div>']
        if r is None:
            lines.append(f'<div style="font-size:12px;color:var(--ea-ink-2)">{html.escape(w["error"])}</div>')
        elif (isinstance(r, tuple) and len(r) == 2 and isinstance(r[0], int) and isinstance(r[1], float)):
            lines.append(f'<div style="font-size:12.5px;color:var(--ea-pred-text);font-family:var(--ea-font-code)">'
                         f'{html.escape(w["says"].format(d=r[0], p=f"{r[1]:.3f}"))}</div>' + _bar(r[1]))   # code: a point in both languages
        else:
            lines.append(f'<div style="font-size:12px;color:var(--ea-ink-2)">{html.escape(w["other"].format(got=repr(r)[:40]))}</div>')
        cells.append('<div style="display:flex;gap:10px;align-items:center;min-width:200px;flex:1">'
                     f'{_digit_img(i)}<div style="flex:1">{"".join(lines)}</div></div>')
    return '<div style="display:flex;flex-wrap:wrap;gap:12px 20px">' + "".join(cells) + "</div>"


def digit_picture(seen) -> str:
    fn = seen.run.names.get("read_digit") if seen.run is not None and seen.run.error is None else None
    if not callable(fn):
        return ""
    w = _DIGIT_WORDS[seen.lang]
    return yours.figure(w["title"], digits_html(reads(fn), seen.lang), alt=w["title"])


# --- the numbers the lesson states --------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def measured() -> dict:
    """The digit reader on the 1,000 test digits, with the pixels divided by 255 and without: how many it reads right,
    and what it says about the test digits of exercise 2."""
    import torch
    images, labels = _test()
    x = torch.tensor(images, dtype=torch.float32)
    with torch.no_grad():
        right = torch.softmax(reader()(x / 255), dim=1)
        raw = torch.softmax(reader()(x), dim=1)
    y = torch.tensor(labels)
    return {"correct": int((right.argmax(1) == y).sum()), "correct_raw": int((raw.argmax(1) == y).sum()),
            "unsure": int((right.max(1).values < 0.9).sum()), "least_raw": float(raw.max(1).values.min()),
            "shown": {i: (int(right[i].argmax()), float(right[i].max())) for i in SHOWN},
            "shown_raw": {i: (int(raw[i].argmax()), float(raw[i].max())) for i in SHOWN}}


_PIPE = {"en": {"title": "A pipeline: test digit 36 in, an answer out", "in": "784 pixels, 0 to 255",
                "prep": "÷ 255: pixels from 0 to 1", "model": "the digit reader: 10 logits",
                "post": "softmax: 10 probabilities", "out": "({d}, {p})", "prep_note": "preparing",
                "model_note": "the model", "post_note": "the answer"},
         "nl": {"title": "Een pipeline: testcijfer 36 erin, een antwoord eruit", "in": "784 pixels, 0 tot 255",
                "prep": "÷ 255: pixels van 0 tot 1", "model": "de cijferlezer: 10 logits",
                "post": "softmax: 10 kansen", "out": "({d}, {p})", "prep_note": "voorbereiden",
                "model_note": "het model", "post_note": "het antwoord"}}


def pipeline_picture(lang: str) -> str:
    """The steps of a pipeline around the digit reader, for test digit 36: preparing, the model, the answer."""
    w = _PIPE[lang]
    d, p = measured()["shown"][36]
    step = ('<div style="padding:6px 9px;border:1px solid var(--ea-line);border-radius:8px;'
            'font-size:13px;color:var(--ea-ink)">{text}<div style="font-size:12px;color:var(--ea-ink-2);margin-top:2px">{note}</div></div>')
    answer = (f'<div style="font-family:var(--ea-font-code);font-size:14px;color:var(--ea-pred-text)">'
              f'{html.escape(w["out"].format(d=d, p=f"{p:.3f}"))}</div>')   # what read_digit gives: code, with a point
    piece = ('<div style="display:flex;gap:6px;align-items:center">'
             '<div style="font-size:16px;color:var(--ea-ink-3)">→</div>{}</div>')
    digit = ('<div style="display:flex;flex-direction:column;align-items:center;gap:2px">' + _digit_img(36, 56)
             + f'<div style="font-size:12px;color:var(--ea-ink-2)">{html.escape(w["in"])}</div></div>')
    parts = [digit,
             piece.format(step.format(text=html.escape(w["prep"]), note=html.escape(w["prep_note"]))),
             piece.format(step.format(text=html.escape(w["model"]), note=html.escape(w["model_note"]))),
             piece.format(step.format(text=html.escape(w["post"]), note=html.escape(w["post_note"]))),
             piece.format(answer)]
    body = '<div style="display:flex;flex-wrap:wrap;gap:8px 6px;align-items:center">' + "".join(parts) + "</div>"
    return yours.figure(w["title"], body, alt=w["title"])


def facts(lang: str) -> dict:
    m = measured()
    d36, p36 = m["shown"][36]
    d111, p111 = m["shown"][111]
    d62, p62 = m["shown"][62]
    sep = "." if lang == "nl" else ","
    return {"correct": m["correct"], "correct_raw": m["correct_raw"], "total": f"1{sep}000",
            "wrong_raw": 1000 - m["correct_raw"], "unsure": m["unsure"], "least_raw": _nr(m["least_raw"], lang, 4),
            "d36": d36, "p36": _nr(p36, lang), "p36_raw": _nr(m["shown_raw"][36][1], lang),
            "d111": d111, "p111": _nr(p111, lang), "d111_raw": m["shown_raw"][111][0],
            "d62": d62, "p62": _nr(p62, lang), "label62": label(62)}


# --- the exercises ------------------------------------------------------------------------------------------------------

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_COMMON = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
                "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."},
}

EXERCISES = {
    "image": ApiBox(
        name="image",
        picture=image_picture,
        function="image_bytes",
        start=_START_IMAGE,
        solution=SOLUTION_IMAGE,
        cases=IMAGE_CASES,
        first=[_DOTS],
        texts={
            **_COMMON,
            "empty": {"en": "Replace the three `...`: the description, the endpoint for images, and the bytes of the image.",
                      "nl": "Vervang de drie `...`: de beschrijving, het endpoint voor afbeeldingen, en de bytes van de afbeelding."},
            "correct": {"en": "Right: your `image_bytes` sends the prompt to `/images/generations`, takes the first image out of the answer, and turns its base64 back into the bytes of a PNG.",
                        "nl": "Goed: je `image_bytes` stuurt de prompt naar `/images/generations`, haalt de eerste afbeelding uit het antwoord, en maakt van de base64 weer de bytes van een PNG."},
            "http-400-prompt": {"en": "With `{call}`, the server answered with status 400: \"{message}\" Put the `prompt` the function gets under `\"prompt\"`.",
                                "nl": "Met `{call}` antwoordde de server met status 400: \"{message}\" Zet de `prompt` die de functie krijgt onder `\"prompt\"`."},
            "http-400-model": {"en": "With `{call}`, the server answered with status 400: \"{message}\" The model and the endpoint go together: `course-shapes` at `/images/generations`.",
                               "nl": "Met `{call}` antwoordde de server met status 400: \"{message}\" Het model en het endpoint horen bij elkaar: `course-shapes` op `/images/generations`."},
            "http-400": {"en": "With `{call}`, the server answered with status 400: \"{message}\" Change the part of your request that the server names, back to what the start had.",
                         "nl": "Met `{call}` antwoordde de server met status 400: \"{message}\" Zet het deel van je request dat de server noemt terug zoals het in het begin stond."},
            "http-404": {"en": "With `{call}`, the server answered with status 404: \"{message}\" Is the address right? The endpoint for images is `/images/generations`, after the `url` the function gets.",
                         "nl": "Met `{call}` antwoordde de server met status 404: \"{message}\" Klopt het adres? Het endpoint voor afbeeldingen is `/images/generations`, na de `url` die de functie krijgt."},
            "http-404-model": {"en": "With `{call}`, the server answered with status 404: \"{message}\" Keep the name of the model as it was.",
                               "nl": "Met `{call}` antwoordde de server met status 404: \"{message}\" Laat de naam van het model staan zoals hij was."},
            "other-prompt": {"en": "With `{call}`, the server did not get under `\"prompt\"` the prompt the function got. Send it as it is.",
                             "nl": "Met `{call}` kreeg de server onder `\"prompt\"` niet de prompt die de functie kreeg. Stuur hem zoals hij is."},
            "text": {"en": "With `{call}`, your function gives back the image as text in base64. Turn that text back into bytes: the module `base64` has a function for it.",
                     "nl": "Met `{call}` geeft je functie de afbeelding terug als tekst in base64. Maak van die tekst weer bytes: de module `base64` heeft daar een functie voor."},
            "whole": {"en": "With `{call}`, your function gives back the whole JSON of the answer. Give back only the bytes of the first image under `\"data\"`.",
                      "nl": "Met `{call}` geeft je functie de hele JSON van het antwoord terug. Geef alleen de bytes van de eerste afbeelding onder `\"data\"` terug."},
            "data": {"en": "With `{call}`, your function gives back the list under `\"data\"`, one item per image. Take the first item, and from it the image under `\"b64_json\"`.",
                     "nl": "Met `{call}` geeft je functie de lijst onder `\"data\"` terug, één item per afbeelding. Neem het eerste item, en daaruit de afbeelding onder `\"b64_json\"`."},
            "item": {"en": "With `{call}`, your function gives back the first item under `\"data\"`, a dictionary. The image is in it under `\"b64_json\"`, as text in base64: give back its bytes.",
                     "nl": "Met `{call}` geeft je functie het eerste item onder `\"data\"` terug, een dictionary. De afbeelding staat erin onder `\"b64_json\"`, als tekst in base64: geef de bytes ervan terug."},
            "none": {"en": "With `{call}`, your function gives back nothing. End with `return` and the bytes of the image.",
                     "nl": "Met `{call}` geeft je functie niets terug. Eindig met `return` en de bytes van de afbeelding."},
            "other": {"en": "With `{call}`, your function gives `{got}`, not the bytes of the image the server sent back.",
                      "nl": "Met `{call}` geeft je functie `{got}`, niet de bytes van de afbeelding die de server terugstuurde."},
        },
    ),
    "read_digit": FunctionBox(
        name="read_digit",
        picture=digit_picture,
        function="read_digit",
        start=_START_DIGIT,
        solution=SOLUTION_DIGIT,
        show=shown,
        visible=[(reader(), digits(36))],
        hidden=[(reader(), digits(i)) for i in SHOWN if i != 36],
        first=[_DOTS],
        mistakes=[
            Mistake("raw", wrong=SOLUTION_DIGIT.replace(" / 255\n", "\n"),
                    en="With `{call}`, your function gives `{got}`: the reader is sure, where it should say `{want}`. The model learned on pixels from 0 to 1, as its model card says, and `pixels` go from 0 to 255: prepare them first.",
                    nl="Met `{call}` geeft je functie `{got}`: de lezer is zeker, waar hij `{want}` zou moeten zeggen. Het model leerde op pixels van 0 tot 1, zoals zijn model card zegt, en `pixels` lopen van 0 tot 255: bereid ze eerst voor."),
            Mistake("floor", wrong=SOLUTION_DIGIT.replace("/ 255", "// 255"),
                    en="With `{call}`, your function gives `{got}`, where `{want}` was expected. `//` divides and rounds down to a whole number, so every pixel becomes 0 or 1. Divide with `/`.",
                    nl="Met `{call}` geeft je functie `{got}`, waar `{want}` werd verwacht. `//` deelt en rondt af naar beneden op een geheel getal, dus elke pixel wordt 0 of 1. Deel met `/`."),
            Mistake("axis", wrong=SOLUTION_DIGIT.replace("dim=1", "dim=0"),
                    en="With `{call}`, your function gives `{got}`: every probability came out as 1. `model(x)` has shape `(1, 10)`, one row with the 10 logits of the digit: which axis does softmax have to go along?",
                    nl="Met `{call}` geeft je functie `{got}`: elke kans kwam uit op 1. `model(x)` heeft vorm `(1, 10)`, één rij met de 10 logits van het cijfer: langs welke as moet softmax gaan?"),
            Mistake("nan", matches=lambda got, text: isinstance(got, tuple) and len(got) == 2 and isinstance(got[1], float) and math.isnan(got[1]),
                    en="With `{call}`, your function gives `{got}`: its probability is no number, `nan`. The reader's largest logit for this digit is about 104, and `torch.exp(104.0)` is too large for a float32. Take the largest logit off first, as in [Many classes: softmax](/course/open/ml-softmax?lang=en), or use `torch.softmax`, which does that for you.",
                    nl="Met `{call}` geeft je functie `{got}`: de kans is geen getal, `nan`. De grootste logit van de lezer voor dit cijfer is ongeveer 104, en `torch.exp(104.0)` is te groot voor een float32. Haal eerst de grootste logit eraf, zoals in [Veel klassen: softmax](/course/open/ml-softmax?lang=nl), of gebruik `torch.softmax`, die dat voor je doet."),
            Mistake("logits", wrong=SOLUTION_DIGIT.replace("torch.softmax(model(x), dim=1)[0]", "model(x)[0]"),
                    en="With `{call}`, your function gives `{got}`: the right digit, but its score is a logit, not a probability from 0 to 1. Turn the 10 logits into probabilities first.",
                    nl="Met `{call}` geeft je functie `{got}`: het goede cijfer, maar de score is een logit, geen kans van 0 tot 1. Maak eerst kansen van de 10 logits."),
            Mistake("tensor", error=lambda e: (isinstance(e, TypeError) and ("unsupported operand" in str(e) or "Tensor input" in str(e))
                                               or isinstance(e, AttributeError) and "reshape" in str(e)),
                    en="With `{call}`, your function stops with an error: `pixels` is a list of whole numbers, and the model takes a tensor of floats. Make that tensor first, then prepare it.",
                    nl="Met `{call}` stopt je functie met een fout: `pixels` is een lijst gehele getallen, en het model wil een tensor met floats. Maak eerst die tensor, en bereid hem dan voor."),
            Mistake("dtype", error=lambda e: isinstance(e, RuntimeError) and ("should be the same" in str(e) or "scalar type" in str(e)),
                    en="With `{call}`, your function stops with an error: the model calculates with floats of type `torch.float32`, and your tensor holds other numbers, such as whole numbers. Make it a tensor of `torch.float32`, and divide so that the result stays a float.",
                    nl="Met `{call}` stopt je functie met een fout: het model rekent met floats van het type `torch.float32`, en je tensor bevat andere getallen, zoals gehele getallen. Maak er een tensor van `torch.float32` van, en deel zo dat de uitkomst een float blijft."),
            Mistake("axis-range", error=lambda e: isinstance(e, IndexError) and "Dimension out of range" in str(e),
                    en="With `{call}`, your function asks for an axis the tensor does not have. `model(x)` has shape `(1, 10)`, so axes 0 and 1; after `[0]`, only axis 0 is left. Along which axis lie the 10 logits?",
                    nl="Met `{call}` vraagt je functie om een as die de tensor niet heeft. `model(x)` heeft vorm `(1, 10)`, dus as 0 en as 1; na `[0]` is alleen as 0 over. Langs welke as liggen de 10 logits?"),
            Mistake("no-dim", error=lambda e: isinstance(e, TypeError) and "softmax" in str(e) and "dim" in str(e),
                    en="With `{call}`, your function stops with an error: `torch.softmax` needs to know along which axis it goes, as `dim=`. The 10 logits of the digit lie along axis 1.",
                    nl="Met `{call}` stopt je functie met een fout: `torch.softmax` moet weten langs welke as hij gaat, als `dim=`. De 10 logits van het cijfer liggen langs as 1."),
            Mistake("row", error=lambda e: isinstance(e, IndexError) and "Dimension out of range" not in str(e),
                    en="Your function asks `p` for a place it does not have. `model(x)` gives one row of 10 numbers, shape `(1, 10)`: take that row with `[0]`, so that `p[digit]` is one probability.",
                    nl="Je functie vraagt `p` om een plek die er niet is. `model(x)` geeft één rij van 10 getallen, vorm `(1, 10)`: neem die rij met `[0]`, zodat `p[digit]` één kans is."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "Replace the two `...`: the pixels prepared for the model, and the 10 probabilities of the digit.",
                      "nl": "Vervang de twee `...`: de pixels voorbereid voor het model, en de 10 kansen van het cijfer."},
            "correct": {"en": "Right: your `read_digit` is a pipeline. It prepares the pixels as the model learned them, runs the reader, and turns its 10 logits into a digit with its probability.",
                        "nl": "Goed: je `read_digit` is een pipeline. Hij bereidt de pixels voor zoals het model ze leerde, laat de lezer rekenen, en maakt van zijn 10 logits een cijfer met zijn kans."},
            "wrong": {"en": "With `{call}`, your function gives `{got}`, and `{want}` was expected.",
                      "nl": "Met `{call}` geeft je functie `{got}`, en `{want}` werd verwacht."},
        },
    ),
}

REQUIRED = ["image", "read_digit", "video-job", "hub-task"]

CASES = [
    ("image", _START_IMAGE, "empty", None),
    ("image", SOLUTION_IMAGE, "correct", None),
    ("image", SOLUTION_IMAGE.replace('"prompt": prompt', '"prompt": None'), "error", "http-400-prompt"),
    ("image", SOLUTION_IMAGE.replace('"prompt": prompt', '"prompt": "a circle"'), "wrong", "other-prompt"),
    ("image", SOLUTION_IMAGE.replace('"/images/generations"', '"/chat/completions"'), "error", "http-400"),
    ("image", SOLUTION_IMAGE.replace('"/images/generations"', '"/images"'), "error", "http-404"),
    ("image", SOLUTION_IMAGE.replace('"course-shapes"', '"qwen3-0.6b"'), "error", "http-400-model"),
    ("image", SOLUTION_IMAGE.replace('"256x256"', '"100x100"'), "error", "http-400"),
    ("image", SOLUTION_IMAGE.replace('url + "/images/generations"', 'url + ...'), "wrong", "dots-left"),
    ("image", SOLUTION_IMAGE.replace('"course-shapes"', '"dall-e-3"'), "error", "http-404-model"),
    ("image", SOLUTION_IMAGE.replace('base64.b64decode(answer["data"][0]["b64_json"])', 'answer["data"][0]["b64_json"]'), "wrong", "text"),
    ("image", SOLUTION_IMAGE.replace('base64.b64decode(answer["data"][0]["b64_json"])', "answer"), "wrong", "whole"),
    ("image", SOLUTION_IMAGE.replace('base64.b64decode(answer["data"][0]["b64_json"])', 'answer["data"]'), "wrong", "data"),
    ("image", SOLUTION_IMAGE.replace('base64.b64decode(answer["data"][0]["b64_json"])', 'answer["data"][0]'), "wrong", "item"),
    ("image", SOLUTION_IMAGE.replace('    return base64.b64decode(answer["data"][0]["b64_json"])', '    base64.b64decode(answer["data"][0]["b64_json"])'), "wrong", "none"),
    ("image", SOLUTION_IMAGE.replace('base64.b64decode(answer["data"][0]["b64_json"])', 'answer["data"][0]["revised_prompt"]'), "wrong", "other"),
    ("read_digit", _START_DIGIT, "empty", None),
    ("read_digit", SOLUTION_DIGIT, "correct", None),
    ("read_digit", SOLUTION_DIGIT.replace("torch.softmax(model(x), dim=1)[0]", "torch.softmax(model(x)[0], dim=0)"), "correct", None),
    ("read_digit", SOLUTION_DIGIT.replace("torch.tensor(pixels, dtype=torch.float32) / 255", "torch.tensor(pixels) / 255.0"), "correct", None),
    ("read_digit", SOLUTION_DIGIT.replace(" / 255\n", "\n"), "wrong", "raw"),
    ("read_digit", SOLUTION_DIGIT.replace("dim=1", "dim=0"), "wrong", "axis"),
    ("read_digit", SOLUTION_DIGIT.replace("torch.softmax(model(x), dim=1)[0]", "model(x)[0]"), "wrong", "logits"),
    ("read_digit", SOLUTION_DIGIT.replace("torch.softmax(model(x), dim=1)[0]", "torch.softmax(model(x), dim=1)"), "error", "row"),
    ("read_digit", SOLUTION_DIGIT.replace("torch.tensor(pixels, dtype=torch.float32) / 255", "pixels / 255"), "error", "tensor"),
    ("read_digit", SOLUTION_DIGIT.replace("x = torch.tensor(pixels, dtype=torch.float32) / 255", "import numpy as np\n    x = np.array(pixels) / 255"), "error", "tensor"),
    ("read_digit", SOLUTION_DIGIT.replace("torch.softmax(model(x), dim=1)[0]", "torch.softmax(model(x))[0]"), "error", "no-dim"),
    ("read_digit", SOLUTION_DIGIT.replace("torch.softmax(model(x), dim=1)[0]", "model(x).softmax(dim=-1)[0]"), "correct", None),
    ("read_digit", SOLUTION_DIGIT.replace("torch.softmax(model(x), dim=1)[0]", "..."), "wrong", "dots-left"),
    ("read_digit", SOLUTION_DIGIT.replace("torch.tensor(pixels, dtype=torch.float32) / 255", "torch.tensor(pixels)"), "error", "dtype"),
    ("read_digit", SOLUTION_DIGIT.replace("torch.tensor(pixels, dtype=torch.float32) / 255", "torch.tensor(pixels) // 255"), "error", "dtype"),
    ("read_digit", SOLUTION_DIGIT.replace("/ 255", "// 255"), "wrong", "floor"),
    ("read_digit", SOLUTION_DIGIT.replace("torch.tensor(pixels, dtype=torch.float32) / 255", "pixels"), "error", "tensor"),
    ("read_digit", SOLUTION_DIGIT.replace("dim=1", "dim=2"), "error", "axis-range"),
    ("read_digit", SOLUTION_DIGIT.replace("torch.softmax(model(x), dim=1)[0]", "torch.softmax(model(x)[0], dim=1)"), "error", "axis-range"),
    ("read_digit", SOLUTION_DIGIT.replace("torch.softmax(model(x), dim=1)[0]", "(torch.exp(model(x)) / torch.exp(model(x)).sum())[0]"), "wrong", "nan"),
    ("read_digit", SOLUTION_DIGIT.replace("        p = torch.softmax(model(x), dim=1)[0]",
                                          "        z = model(x)\n        e = torch.exp(z - z.max())\n        p = (e / e.sum())[0]"), "correct", None),
]


# --- the demo: the learner's image_bytes against the course's server ----------------------------------------------------

def learner_image(text):
    """The learner's image_bytes from exercise 1, when it is right; else None."""
    if text is MISSING or text is None or EXERCISES["image"].check(text).state != "correct":
        return None
    return run_code(text).names.get("image_bytes")


def demo(image_bytes, prompt: str, words: dict, lang: str) -> str:
    """The learner's image_bytes with their own prompt, sent to the course's server: the image their function gave
    back, the shapes the server read in the prompt, and the time. `words` has "title", "failed" ({why}), "drawn"
    ({drawn}), "done" ({seconds}) and "not_png"."""
    import time
    running = serve.model_server()   # the endpoint for images needs no download
    start = time.perf_counter()
    try:
        got = image_bytes(running.url, prompt)
    except Exception as e:   # the learner's code, against the real server; a prompt it cannot draw gives status 400
        why = apicheck.server_said(e) if hasattr(e, "code") else f"{type(e).__name__}: {e}"
        return f"<p><em>{html.escape(words['failed'].format(why=why))}</em></p>"
    took = time.perf_counter() - start
    if not (isinstance(got, bytes) and got.startswith(PNG)):
        return f"<p><em>{html.escape(words['not_png'])}</em></p>"
    body = (png_html(got, 256, prompt) + _label(words["drawn"].format(drawn=drawing.said(drawing.read(prompt), lang)))
            + _label(words["done"].format(seconds=_nr(took, lang, 2))))
    return yours.figure(words["title"], body, alt=words["title"])
