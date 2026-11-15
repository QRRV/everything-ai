# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-multimodal-apis: a function that turns an image file into a data URL, and one that asks the
course's server about an image, with its hosted tool, against a stand-in of the server; the course's recorded questions
about three of its photos (data/course-apis-qwen.json), the picture of a request with an image, and the demo with the
learner's own function."""
import base64
import html
import io
import json

from everything_ai import apicheck, detr, lesson as lesson_helpers, paths, qwen3, serve, yours
from everything_ai.apicheck import URL, ApiBox, Case
from everything_ai.check import MISSING, FunctionBox, Mistake, run_code

DEMO_FILES = [*qwen3.CHAT_FILES, detr.FILE]   # what the demo downloads, for the line under the header
DOWNLOAD_LINE = {
    "en": f"Works on every computer. The chat model Qwen3-0.6B and the detector DETR are downloaded once from Hugging Face, {lesson_helpers.download_size(DEMO_FILES, 'en')}, when you click their button in the lesson; after that, the lesson works without a connection. The exercises work without them too.",
    "nl": f"Werkt op elke computer. Het chat model Qwen3-0.6B en de detector DETR worden één keer gedownload van Hugging Face, {lesson_helpers.download_size(DEMO_FILES, 'nl')}, als je in de les op de knop ervoor klikt; daarna werkt de les zonder verbinding. De oefeningen werken ook zonder die twee.",
}

PHOTOS = {"coffee": "What is on the table?", "chelsea": "What animal is this?", "fruit": "What fruit is in the picture?"}
RECORD = {name: {"image": name, "question": q, "max_tokens": 60, "seeds": 1} for name, q in PHOTOS.items()}
DATA = paths.ROOT / "data" / "course-apis-qwen.json"
SHOWN = "data/photos/fruit-sticker.png"   # the image of exercise 1's example and of exercise 2's first case
LABELS = "data/photos/fruit-labels.png"   # the other image of exercise 1, a small grayscale PNG
CELL = "data/photos/cell.png"             # and the other image of exercise 2


def recorded(name: str) -> dict:
    """The course's question about data/photos/<name>.png: the answer, what the hosted tool found, and the time."""
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-multimodal-apis"][name]["answers"][0]


def _nr(x: float, lang: str, digits: int = 1) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def _count(n: int, lang: str) -> str:
    return f"{n:,}".replace(",", ".") if lang == "nl" else f"{n:,}"


def size_facts(lang: str) -> dict:
    """The numbers the texts give about exercise 1's image: its bytes, and the characters of its base64 and its data URL."""
    data = (paths.ROOT / SHOWN).read_bytes()
    text = base64.b64encode(data).decode()
    return {"bytes": _count(len(data), lang), "chars": _count(len(text), lang),
            "url": _count(len("data:image/png;base64," + text), lang), "start": text[:24]}


def found_words(name: str) -> str:
    """What the hosted tool found on a photo, as the server told the model: "cup, dining table, spoon"."""
    runs = recorded(name)["hosted_tools"]
    return serve.named([(f["kind"], f["score"]) for f in runs[0]["found"]])


def seconds(name: str, lang: str) -> str:
    return _nr(recorded(name)["seconds"], lang)


_WORKED = {
    "en": {"title": "The course's question about the coffee photo", "request": "The course's code sent: POST /v1/chat/completions",
           "answer": "The server sent back: status 200", "parts": "the content: a list of two parts",
           "image": "the image, as a data URL", "seed": "so the course gets the same answer again", "tool": "ask for the server's hosted tool", "content": "the text of the answer",
           "found": "what the hosted tool found"},
    "nl": {"title": "De vraag van de course over de koffiefoto", "request": "De code van de course stuurde: POST /v1/chat/completions",
           "answer": "De server stuurde terug: status 200", "parts": "de content: een lijst met twee delen",
           "image": "het beeld, als data-URL", "seed": "zodat de course hetzelfde antwoord terugkrijgt", "tool": "vraag om de hosted tool van de server", "content": "de tekst van het antwoord",
           "found": "wat de hosted tool vond"},
}


def worked_request(lang: str) -> dict:
    """The body of the worked example, with its data URL cut short as the pictures cut it."""
    data = (paths.ROOT / "data" / "photos" / "coffee.png").read_bytes()
    body = {"model": serve.MODEL, "messages": [{"role": "user", "content": [
        {"type": "text", "text": PHOTOS["coffee"]},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}}]}],
        "max_tokens": 60, "seed": 0, "tools": [{"type": serve.DETECT}]}
    return apicheck.trimmed(body, _TRIM[lang])


def worked_answer() -> dict:
    """The answer of the worked example, as the server sent it, without its id and time."""
    a = recorded("coffee")
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": a["text"]}, "finish_reason": a["finish"]}],
            "usage": {"prompt_tokens": a["prompt_tokens"], "completion_tokens": a["tokens"],
                      "total_tokens": a["prompt_tokens"] + a["tokens"]},
            "hosted_tools": a["hosted_tools"]}


def _pre(text: str, marks: dict, said: str | None = None) -> str:
    """Code-like text; each line that contains a key of `marks` gets that note beside it, in grey, and the line with
    the text the model wrote, `said`, the colour of what a model writes."""
    out = []
    model_line = None if said is None else '"content": ' + json.dumps(said, ensure_ascii=False)
    for line in text.split("\n"):
        note = next((n for k, n in marks.items() if k in line), "")
        colour = "var(--ea-pred-text)" if model_line is not None and model_line in line else "var(--ea-ink)"
        out.append(f'<span style="color:{colour}">{html.escape(line)}</span>'
                   + (f'<span style="color:var(--ea-ink-2);font-family:var(--ea-font-ui)">   ← {html.escape(note)}</span>' if note else ""))
    return ('<pre style="margin:0.2rem 0 0.7rem;padding:0.55rem 0.7rem;background:var(--ea-inset);border-radius:6px;'
            'font-family:var(--ea-font-code);font-size:12px;line-height:1.45;white-space:pre-wrap;overflow-wrap:anywhere">'
            + "\n".join(out) + "</pre>")


def _label(text: str) -> str:
    return f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)">{html.escape(text)}</div>'


def worked_picture(lang: str) -> str:
    """The worked example: the request the course's code sent about the coffee photo, and the answer, with notes."""
    w = _WORKED[lang]
    answer = worked_answer()
    body = (_label(w["request"])
            + _pre(json.dumps(worked_request(lang), ensure_ascii=False, indent=2),
                   {'"content": [': w["parts"], '"url"': w["image"], '"seed"': w["seed"], '"tools"': w["tool"]})
            + _label(w["answer"])
            + _pre(json.dumps(answer, ensure_ascii=False, indent=2), {'"content": "': w["content"], '"hosted_tools"': w["found"]},
                   said=answer["choices"][0]["message"]["content"]))
    return yours.figure(w["title"], body, alt=w["title"])


_TRIM = {"en": {"data": "… {n} characters in all", "numbers": "… {n} numbers in all", "thousands": ","},
         "nl": {"data": "… {n} tekens in totaal", "numbers": "… {n} getallen in totaal", "thousands": "."}}


# --- the picture: a request with an image, and what the course's server does with it -----------------------------------

_FLOW = {
    "en": {"title": "A question about an image, through the course's server", "program": "your program",
           "message": "one message, two parts", "text": "text: the question", "image": "image_url: data:image/png;base64,…",
           "server": "the course's server", "tool": "hosted tool: the detector", "words": "\"cup, dining table, spoon\"",
           "model": "Qwen3: reads text only", "answer": "the answer",
           "note": "The detector runs at the server, not in your program: you only ask for it in the request.",
           "alt": "Your program sends one message with a text part and an image part to the course's server. There the hosted tool, the detector, turns the image into words, and Qwen3 reads the question with those words and writes the answer, which goes back to your program."},
    "nl": {"title": "Een vraag over een beeld, via de server van de course", "program": "je programma",
           "message": "één bericht, twee delen", "text": "text: de vraag", "image": "image_url: data:image/png;base64,…",
           "server": "de server van de course", "tool": "hosted tool: de detector", "words": "\"cup, dining table, spoon\"",
           "model": "Qwen3: leest alleen tekst", "answer": "het antwoord",
           "note": "De detector draait op de server, niet in je programma: je vraagt er alleen om in het request.",
           "alt": "Je programma stuurt één bericht met een tekstdeel en een beelddeel naar de server van de course. Daar maakt de hosted tool, de detector, woorden van het beeld, en Qwen3 leest de vraag met die woorden en schrijft het antwoord, dat terug naar je programma gaat."},
}


def flow_picture(lang: str) -> str:
    w = _FLOW[lang]
    e = html.escape

    def box(x, y, width, height, text, sub="", ink="var(--ea-ink)"):
        out = (f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="6" fill="var(--ea-inset)" stroke="var(--ea-line)"/>'
               f'<text x="{x + width / 2}" y="{y + (18 if sub else height / 2 + 4)}" text-anchor="middle" font-size="12" fill="{ink}">{e(text)}</text>')
        if sub:
            out += f'<text x="{x + width / 2}" y="{y + 36}" text-anchor="middle" font-size="11" fill="var(--ea-ink-2)">{e(sub)}</text>'
        return out

    def arrow(x1, y1, x2, y2):
        return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="var(--ea-ink-2)" stroke-width="1.2" '
                f'marker-end="url(#ea-mm-arrow)"/>')
    svg = (f'<svg viewBox="0 0 640 250" width="100%" role="img" aria-label="{e(w["alt"])}" '
           f'style="max-width:640px;font-family:var(--ea-font-ui)">'
           '<defs><marker id="ea-mm-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">'
           '<path d="M0,0 L8,4 L0,8 z" fill="var(--ea-ink-2)"/></marker></defs>'
           + box(10, 20, 150, 140, w["program"])
           + f'<text x="85" y="62" text-anchor="middle" font-size="11" fill="var(--ea-ink-2)">{e(w["message"])}</text>'
           + f'<rect x="22" y="72" width="126" height="28" rx="4" fill="var(--ea-inset)"/>'
           + f'<text x="85" y="90" text-anchor="middle" font-size="11" fill="var(--ea-ink)">{e(w["text"])}</text>'
           + f'<rect x="22" y="106" width="126" height="40" rx="4" fill="var(--ea-inset)"/>'
           + f'<text x="85" y="122" text-anchor="middle" font-size="11" fill="var(--ea-ink)">image_url:</text>'
           + f'<text x="85" y="138" text-anchor="middle" font-size="11" fill="var(--ea-ink)">data:image/png;base64,…</text>'
           + arrow(162, 90, 228, 90)
           + f'<rect x="230" y="10" width="400" height="160" rx="8" fill="none" stroke="var(--ea-line)" stroke-dasharray="4 3"/>'
           + f'<text x="430" y="28" text-anchor="middle" font-size="12" fill="var(--ea-ink-2)">{e(w["server"])}</text>'
           + box(246, 50, 160, 56, w["tool"], "detr-resnet-50")
           + arrow(408, 78, 448, 78)
           + f'<text x="530" y="56" text-anchor="middle" font-size="11" fill="var(--ea-ink-2)">{e(w["words"])}</text>'
           + box(450, 66, 168, 40, w["model"])
           + arrow(534, 108, 534, 150)
           + f'<text x="534" y="166" text-anchor="middle" font-size="12" fill="var(--ea-pred-text)">{e(w["answer"])}</text>'
           + f'<path d="M 470 162 C 330 200 200 200 90 164" fill="none" stroke="var(--ea-ink-2)" stroke-width="1.2" marker-end="url(#ea-mm-arrow)"/>'
           + f'<text x="320" y="236" text-anchor="middle" font-size="12" fill="var(--ea-ink-2)">{e(w["note"])}</text>'
           + '</svg>')
    return yours.figure(w["title"], svg, alt=w["alt"])


def _thumb(path, width: int = 150) -> str:
    """A small JPEG of an image file, as an <img> with a data URL: the page stays light."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    img.thumbnail((width, width))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=85)
    return (f'<img src="data:image/jpeg;base64,{base64.b64encode(out.getvalue()).decode()}" alt="" '
            f'style="width:{img.width}px;height:{img.height}px;border-radius:4px;flex:none">')


_RUNS = {
    "en": {"title": "The course's three questions, with the hosted tool", "found": "the detector found: {found}",
           "none": "the detector found nothing it knows", "time": "{s} seconds",
           "alt": "Three of the course's photos, each with the question, what the server's detector found, and Qwen3's answer."},
    "nl": {"title": "De drie vragen van de course, met de hosted tool", "found": "de detector vond: {found}",
           "none": "de detector vond niets wat hij kent", "time": "{s} seconden",
           "alt": "Drie foto's van de course, elk met de vraag, wat de detector van de server vond, en het antwoord van Qwen3."},
}


def _run_row(path, question: str, found: str, answer: str, note: str, lang: str) -> str:
    w = _RUNS[lang]
    said = w["found"].format(found=found) if found else w["none"]
    return (f'<div style="display:flex;gap:0.7rem;align-items:flex-start;margin:0.3rem 0 0.6rem">{_thumb(path)}'
            f'<div style="min-width:0"><div style="font-family:var(--ea-font-code);font-size:13px;color:var(--ea-ink)">{html.escape(question)}</div>'
            f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2);margin:0.15rem 0">{html.escape(said)}</div>'
            f'<div style="font-family:var(--ea-font-code);font-size:13px;line-height:1.45;color:var(--ea-pred-text);white-space:pre-wrap;overflow-wrap:anywhere">{html.escape(answer)}</div>'
            f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)">{html.escape(note)}</div></div></div>')


def recorded_picture(lang: str) -> str:
    """The course's recorded question about each photo: the photo, what the detector found, the answer and the time."""
    w = _RUNS[lang]
    rows = [_run_row(paths.ROOT / "data" / "photos" / f"{name}.png", q, found_words(name), recorded(name)["text"],
                     w["time"].format(s=seconds(name, lang)), lang) for name, q in PHOTOS.items()]
    return yours.figure(w["title"], "".join(rows), alt=w["alt"])


# --- exercise 1: an image file as a data URL ----------------------------------------------------------------------------

_START_URL = '''import base64


def data_url(path):
    with open(path, "rb") as f:
        data = f.read()
    text = ...
    return ...
'''
SOLUTION_URL = '''import base64


def data_url(path):
    with open(path, "rb") as f:
        data = f.read()
    text = base64.b64encode(data).decode()
    return "data:image/png;base64," + text
'''


def _short(value):
    """How a check message shows a data URL: its start and its length."""
    if isinstance(value, str) and len(value) > 70:
        return {"en": f"{value[:48]}… ({len(value):,} characters)", "nl": f"{value[:48]}… ({_count(len(value), 'nl')} tekens)"}
    return value


def url_picture(seen) -> str:
    """The learner's data URL of the example image, as a browser shows it, with its length."""
    run = seen.mine
    fn = run.names.get("data_url") if run is not None else None
    if not callable(fn):
        return ""
    w = {"en": ("Your data URL, as a browser shows it", "{bytes} bytes in the file became {chars} characters of text.",
                "This is not a data URL of a PNG image yet, so a browser shows no image.", "Your function stops with an error."),
         "nl": ("Je data-URL, zoals een browser hem laat zien", "{bytes} bytes in het bestand werden {chars} tekens tekst.",
                "Dit is nog geen data-URL van een PNG-beeld, dus een browser laat geen beeld zien.", "Je functie stopt met een fout.")}[seen.lang]
    try:
        got = fn(SHOWN)
    except Exception:   # the learner's code
        return yours.figure(w[0], f'<p style="color:var(--ea-loss-text)">{html.escape(w[3])}</p>', alt=w[0])
    if not isinstance(got, str) or not got.startswith("data:image/png;base64,"):
        return yours.figure(w[0], f'<p style="color:var(--ea-ink-2)">{html.escape(w[2])}</p>', alt=w[0])
    size = len((paths.ROOT / SHOWN).read_bytes())
    note = w[1].format(bytes=_count(size, seen.lang), chars=_count(len(got), seen.lang))
    body = (f'<div style="display:flex;gap:0.8rem;align-items:center"><img src="{html.escape(got, quote=True)}" alt="" '
            f'style="max-width:150px;border:1px solid var(--ea-line);border-radius:4px;background:var(--ea-inset)">'
            f'<div><div style="font-family:var(--ea-font-code);font-size:12px;color:var(--ea-ink);overflow-wrap:anywhere">{html.escape(got[:56])}…</div>'
            f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2);margin-top:0.2rem">{html.escape(note)}</div></div></div>')
    return yours.figure(w[0], body, alt=w[0])


# --- exercise 2: a question about an image, with the hosted tool ------------------------------------------------------------

class NoDataUrl(Exception):
    """The learner's data_url is not right yet."""


def no_data_url(path):
    raise NoDataUrl()


def learner_data_url(text):
    """The learner's data_url from exercise 1, when it is right; else None."""
    if text is MISSING or text is None or EXERCISES["data_url"].check(text).state != "correct":
        return None
    return run_code(text).names.get("data_url")


def photo_inputs(url_text) -> dict:
    """What exercise 2 runs with: the learner's data_url from exercise 1, or one that says to finish it first."""
    return {"data_url": learner_data_url(url_text) or no_data_url}


_START_PHOTO = '''import json
import urllib.request


def ask_photo(url, path, question):
    content = ...
    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": content}],
            "max_tokens": 60, "tools": ...}
    request = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        answer = json.loads(response.read())
    return answer["choices"][0]["message"]["content"]
'''
SOLUTION_PHOTO = '''import json
import urllib.request


def ask_photo(url, path, question):
    content = [{"type": "text", "text": question},
               {"type": "image_url", "image_url": {"url": data_url(path)}}]
    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": content}],
            "max_tokens": 60, "tools": [{"type": "detect_objects"}]}
    request = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        answer = json.loads(response.read())
    return answer["choices"][0]["message"]["content"]
'''
_ASKED = [(SHOWN, "What is in this picture?"), (CELL, "What do you see?")]


def _parts(running) -> list | None:
    """The parts of the content of the first message the server got, when it is a list; else None."""
    try:
        content = running.log[0]["body"]["messages"][0]["content"]
    except (IndexError, KeyError, TypeError):
        return None
    return content if isinstance(content, list) else None


def _judge(path: str, question: str):
    want = (paths.ROOT / path).read_bytes()

    def judge(got, running):
        last = running.log[-1] if running.log else {}
        sent = (((last.get("answer") or {}).get("choices") or [{}])[0].get("message") or {}).get("content") \
            if isinstance(last.get("answer"), dict) else None
        parts = _parts(running)
        if parts is None:
            return "text-only"
        texts = [p.get("text") for p in parts if isinstance(p, dict) and p.get("type") == "text"]
        images = [p["image_url"]["url"] for p in parts if isinstance(p, dict) and p.get("type") == "image_url"]
        if question not in texts:
            return "no-question"
        if not images or base64.b64decode(images[0].split(";base64,", 1)[1]) != want:
            return "other-image"
        if got == sent:
            return None
        if isinstance(got, dict):
            return "whole"
        if got is None:
            return "none"
        return "other"
    return judge


PHOTO_CASES = [Case(serve.StandIn, (URL, p, q), _judge(p, q)) for p, q in _ASKED]


def photo_picture(seen) -> str:
    words = {"en": {"title": "What your ask_photo sent, and what came back", "request": "Your code sent: {method} {path}",
                    "headers": "Headers:", "answer": "The server sent back: status {status}",
                    "stream": "a stream with {n} pieces", "none": "Your function sent nothing to the server yet.", **_TRIM["en"]},
             "nl": {"title": "Wat je ask_photo stuurde, en wat er terugkwam", "request": "Je code stuurde: {method} {path}",
                    "headers": "Headers:", "answer": "De server stuurde terug: status {status}",
                    "stream": "een stream met {n} stukjes", "none": "Je functie stuurde nog niets naar de server.", **_TRIM["nl"]}}[seen.lang]
    return apicheck.exchange(seen, "ask_photo", PHOTO_CASES[0], words)


_COMMON = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
                "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."},
}

EXERCISES = {
    "data_url": FunctionBox(
        name="data_url",
        picture=url_picture,
        function="data_url",
        start=_START_URL,
        solution=SOLUTION_URL,
        visible=[(SHOWN,)],
        hidden=[(LABELS,)],
        show=_short,
        mistakes=[
            Mistake("no-decode", wrong=SOLUTION_URL.replace("base64.b64encode(data).decode()", "str(base64.b64encode(data))"),
                    en="`{call}` gives `{got}`: there is a `b'` after the comma. `base64.b64encode` gives bytes, and `str()` of bytes writes them as Python shows them, with `b'` and `'`. Turn the bytes into text with `.decode()`.",
                    nl="`{call}` geeft `{got}`: er staat een `b'` na de komma. `base64.b64encode` geeft bytes, en `str()` van bytes schrijft ze zoals Python ze laat zien, met `b'` en `'`. Maak er tekst van met `.decode()`."),
            Mistake("no-prefix", wrong=SOLUTION_URL.replace('return "data:image/png;base64," + text', "return text"),
                    en="`{call}` gives `{got}`: the base64 of the image, but no data URL yet. Put the start of a data URL in front: what kind of file it is, and that base64 follows.",
                    nl="`{call}` geeft `{got}`: de base64 van het beeld, maar nog geen data-URL. Zet het begin van een data-URL ervoor: wat voor bestand het is, en dat er base64 volgt."),
            Mistake("jpeg", wrong=SOLUTION_URL.replace("image/png", "image/jpeg"),
                    en="`{call}` gives `{got}`. The file is a PNG, so its data URL says `image/png`: the type tells the reader how to read the bytes.",
                    nl="`{call}` geeft `{got}`. Het bestand is een PNG, dus de data-URL zegt `image/png`: het type vertelt de lezer hoe hij de bytes moet lezen."),
            Mistake("no-comma", wrong=SOLUTION_URL.replace('"data:image/png;base64,"', '"data:image/png;base64"'),
                    en="`{call}` gives `{got}`. Between `base64` and the text of the image comes a comma: `data:image/png;base64,` and then the text.",
                    nl="`{call}` geeft `{got}`. Tussen `base64` en de tekst van het beeld staat een komma: `data:image/png;base64,` en dan de tekst."),
            Mistake("path-text", wrong=SOLUTION_URL.replace("base64.b64encode(data)", "base64.b64encode(path.encode())"),
                    en="`{call}` gives `{got}`: that is the base64 of the file's name, not of the image. Encode the bytes you read from the file, `data`.",
                    nl="`{call}` geeft `{got}`: dat is de base64 van de naam van het bestand, niet van het beeld. Codeer de bytes die je uit het bestand las, `data`."),
            Mistake("text-mode", error=lambda e: isinstance(e, UnicodeDecodeError),
                    en="Python stops while it reads the file: it tries to read the bytes of an image as letters. Open the file with `\"rb\"`, so that `f.read()` gives the bytes as they are.",
                    nl="Python stopt terwijl hij het bestand leest: hij probeert de bytes van een beeld als letters te lezen. Open het bestand met `\"rb\"`, zodat `f.read()` de bytes geeft zoals ze zijn."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "Replace the two `...`: first the base64 of `data` as text, then the data URL.",
                      "nl": "Vervang de twee `...`: eerst de base64 van `data` als tekst, dan de data-URL."},
            "correct": {"en": "Right: your `data_url` turns both images into a data URL that a browser, and a multimodal API, can read.",
                        "nl": "Goed: je `data_url` maakt van beide beelden een data-URL die een browser, en een multimodale API, kan lezen."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn."},
            "almost": {"en": "Right for the first image. But `{call}` gives `{got}`, and it should give `{want}`.",
                       "nl": "Goed voor het eerste beeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn."},
        },
    ),
    "ask_photo": ApiBox(
        name="ask_photo",
        picture=photo_picture,
        function="ask_photo",
        start=_START_PHOTO,
        solution=SOLUTION_PHOTO,
        cases=PHOTO_CASES,
        names={"data_url": run_code(SOLUTION_URL).names["data_url"]},
        mistakes=[
            Mistake("no-data-url", error=lambda e: isinstance(e, NoDataUrl),
                    en="This exercise uses your `data_url` from exercise 1. Finish exercise 1 first; this check then runs again by itself.",
                    nl="Deze oefening gebruikt je `data_url` uit oefening 1. Maak eerst oefening 1 af; deze check draait daarna vanzelf opnieuw."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "Replace the two `...`: the content, a list with a text part and an image part, and the tools, with the server's hosted tool.",
                      "nl": "Vervang de twee `...`: de content, een lijst met een tekstdeel en een beelddeel, en de tools, met de hosted tool van de server."},
            "correct": {"en": "Right: your `ask_photo` sends the question and the image as two parts of one message, asks for the hosted tool, and gives back the answer, for both images.",
                        "nl": "Goed: je `ask_photo` stuurt de vraag en het beeld als twee delen van één bericht, vraagt om de hosted tool, en geeft het antwoord terug, voor beide beelden."},
            "text-only": {"en": "With `{call}`, the content your function sent is one string, so the server got no image part. Make the content a list of parts: one of type `text`, one of type `image_url`.",
                          "nl": "Met `{call}` is de content die je functie stuurde één string, dus de server kreeg geen beelddeel. Maak van de content een lijst met delen: één van type `text`, één van type `image_url`."},
            "no-question": {"en": "With `{call}`, the server got no part of type `text` with the question. Put the `question` the function gets in a part `{{\"type\": \"text\", \"text\": question}}`.",
                            "nl": "Met `{call}` kreeg de server geen deel van type `text` met de vraag. Zet de `question` die de functie krijgt in een deel `{{\"type\": \"text\", \"text\": question}}`."},
            "other-image": {"en": "With `{call}`, the server got no image, or another one than the file at `path`. Send `data_url(path)`, with the `path` the function gets.",
                            "nl": "Met `{call}` kreeg de server geen beeld, of een ander dan het bestand op `path`. Stuur `data_url(path)`, met de `path` die de functie krijgt."},
            "whole": {"en": "With `{call}`, your function gives back the whole JSON of the answer. Give back only the text: under `\"choices\"`, the first, its `\"message\"`, its `\"content\"`.",
                      "nl": "Met `{call}` geeft je functie de hele JSON van het antwoord terug. Geef alleen de tekst terug: onder `\"choices\"` de eerste, zijn `\"message\"`, zijn `\"content\"`."},
            "none": {"en": "With `{call}`, your function gives back nothing. End with `return` and the text of the answer.",
                     "nl": "Met `{call}` geeft je functie niets terug. Eindig met `return` en de tekst van het antwoord."},
            "other": {"en": "With `{call}`, your function gives `{got}`, not the text the server sent back.",
                      "nl": "Met `{call}` geeft je functie `{got}`, niet de tekst die de server terugstuurde."},
        },
    ),
}

REQUIRED = ["data_url", "ask_photo", "fruit", "realtime"]

CASES = [
    ("data_url", _START_URL, "empty", None),
    ("data_url", SOLUTION_URL, "correct", None),
    ("data_url", EXERCISES["data_url"].mistakes[0].wrong, "wrong", "no-decode"),
    ("data_url", EXERCISES["data_url"].mistakes[1].wrong, "wrong", "no-prefix"),
    ("data_url", EXERCISES["data_url"].mistakes[2].wrong, "wrong", "jpeg"),
    ("data_url", EXERCISES["data_url"].mistakes[3].wrong, "wrong", "no-comma"),
    ("data_url", EXERCISES["data_url"].mistakes[4].wrong, "wrong", "path-text"),
    ("data_url", SOLUTION_URL.replace('open(path, "rb")', "open(path)"), "error", "text-mode"),
    ("ask_photo", _START_PHOTO, "empty", None),
    ("ask_photo", SOLUTION_PHOTO, "correct", None),
    ("ask_photo", SOLUTION_PHOTO.replace(', "tools": [{"type": "detect_objects"}]', ""), "error", "http"),
    ("ask_photo", SOLUTION_PHOTO.replace('"image_url": {"url": data_url(path)}', '"image_url": data_url(path)'), "error", "http"),
    ("ask_photo", SOLUTION_PHOTO.replace('{"type": "image_url", "image_url"', '{"type": "image", "image_url"'), "error", "http"),
    ("ask_photo", SOLUTION_PHOTO.replace("data_url(path)}", "path}"), "error", "http"),
    ("ask_photo", SOLUTION_PHOTO.replace('''    content = [{"type": "text", "text": question},
               {"type": "image_url", "image_url": {"url": data_url(path)}}]''', '''    content = question + " " + data_url(path)'''), "wrong", "text-only"),
    ("ask_photo", SOLUTION_PHOTO.replace('{"type": "text", "text": question},\n               ', ""), "wrong", "no-question"),
    ("ask_photo", SOLUTION_PHOTO.replace("data_url(path)", 'data_url("data/photos/fruit-sticker.png")'), "wrong", "other-image"),
    ("ask_photo", SOLUTION_PHOTO.replace(''',
               {"type": "image_url", "image_url": {"url": data_url(path)}}]''', "]"), "wrong", "other-image"),
    ("ask_photo", SOLUTION_PHOTO.replace('return answer["choices"][0]["message"]["content"]', "return answer"), "wrong", "whole"),
    ("ask_photo", SOLUTION_PHOTO.replace('return answer["choices"][0]["message"]["content"]', 'answer["choices"][0]["message"]["content"]'), "wrong", "none"),
]


# --- the demo: the learner's ask_photo, against the course's server with the detector and Qwen3 -------------------------------

FILES = list(qwen3.CHAT_FILES) + [detr.FILE]
DEMO_PHOTOS = {"en": {"a cup of coffee": "coffee", "a cat": "chelsea", "a drawing of fruit": "fruit", "a cell under a microscope": "cell"},
               "nl": {"een kopje koffie": "coffee", "een kat": "chelsea", "een tekening van fruit": "fruit", "een cel onder een microscoop": "cell"}}


def size_mb() -> int:
    """The size of the demo's two models with the tokenizer, in MB: the button says it in GB."""
    from everything_ai import hub
    return round(sum(hub.PINNED[pin]["size"] for pin in FILES) / 1e6)


def downloaded() -> bool:
    from everything_ai import hub
    return all(hub.pinned(*pin) is not None for pin in FILES)


def download_or_stop(button, texts: dict) -> None:
    lesson_helpers.pins_or_stop(button, texts, FILES)


def learner_ask_photo(text, url_text):
    """The learner's ask_photo from exercise 2 with their data_url, when both are right; else None."""
    inputs = photo_inputs(url_text)
    if inputs["data_url"] is no_data_url or text is MISSING or text is None:
        return None
    if EXERCISES["ask_photo"].check(text, inputs).state != "correct":
        return None
    return run_code(text, inputs).names.get("ask_photo")


def demo(ask_photo, name: str, question: str, words: dict, lang: str) -> str:
    """The learner's ask_photo about data/photos/<name>.png, sent to the course's server: the photo, what the hosted
    tool found, the answer, and how long it took."""
    import time
    running = serve.model_server()
    path = f"data/photos/{name}.png"
    before = len(running.log)
    start = time.perf_counter()
    try:
        text = ask_photo(running.url, path, question)
    except Exception as e:   # the learner's code, against the real server
        said = words["memory"] if serve.out_of_memory(e) else words["failed"].format(why=f"{type(e).__name__}: {e}")
        return f"<p><em>{html.escape(said)}</em></p>"
    took = time.perf_counter() - start
    runs = [r for entry in running.log[before:] if isinstance(entry.get("answer"), dict)
            for r in entry["answer"].get("hosted_tools", [])]
    found = serve.named([(f["kind"], f["score"]) for f in runs[0]["found"]]) if runs else ""
    row = _run_row(paths.ROOT / path, question, found, str(text), words["done"].format(seconds=_nr(took, lang)), lang)
    return yours.figure(words["title"], row, alt=words["title"])
