# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-prompt-templates: a template with a JSON example in it, whose own braces must be doubled for
.format, and a function that counts in a log how many answers of each version of a template fit; the course's
recorded runs (data/course-dev-prompts-qwen.json): two versions of the bike template, each filled with three texts
about bikes and asked eight times in JSON mode; their pictures, and the demo with a version of your own."""
import html
import json

from everything_ai import chat, paths, serve, yours
from everything_ai.check import FunctionBox, Mistake, run_code, with_time_limit

DOWNLOADS = {"Qwen3-0.6B": chat.FILES}   # what the demo downloads, for the line under the header

NAME = "bike-details"
TEXTS = {
    "cityline": "Our new Cityline is a grey city bike with seven gears, for 649 euros.",
    "trailhawk": "The Trailhawk is a green mountain bike with 21 gears. Price: 1199 euros.",
    "pixie": "Meet the Pixie: a red children's bike without gears, yours for 189 euros.",
}
ASK = "Give the name, color and price of this bike as JSON, with the keys name, color and price."
TEMPLATES = {
    1: ASK + "\n\n{text}",
    2: ASK + ' The price is a number, without quotes, such as {{"price": 100}}.\n\n{text}',
}
JSON_MODE = {"type": "json_object"}
SEEDS = 8
RECORD = {f"v{v}-{key}": {"messages": [{"role": "user", "content": template.format(text=text)}], "max_tokens": 80,
                          "response_format": JSON_MODE, "seeds": SEEDS}
          for v, template in TEMPLATES.items() for key, text in TEXTS.items()}
DATA = paths.ROOT / "data" / "course-dev-prompts-qwen.json"


def fits(text: str) -> bool:
    """Whether an answer is what the app needs: JSON that json.loads reads, a dictionary with name, color and price,
    name and color strings, and the price a number."""
    try:
        value = json.loads(text)
    except ValueError:
        return False
    if not isinstance(value, dict) or not all(k in value for k in ("name", "color", "price")):
        return False
    price = value["price"]
    return (isinstance(value["name"], str) and isinstance(value["color"], str)
            and isinstance(price, (int, float)) and not isinstance(price, bool))


# --- the recorded runs ---------------------------------------------------------------------------------------------------

def course() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-prompt-templates"]


def answers(version: int, key: str) -> list:
    return course()[f"v{version}-{key}"]["answers"]


def log() -> list:
    """The course's runs as an app would log them: for each answer, the template and its version, the text it was
    filled with, the seed, the answer, and whether it fits."""
    return [{"prompt": NAME, "version": v, "text": key, "seed": a["seed"], "answer": a["text"], "fits": fits(a["text"])}
            for v in TEMPLATES for key in TEXTS for a in answers(v, key)]


def value(text: str):
    """What json.loads makes of an answer that fits."""
    return json.loads(text)


def findings() -> dict:
    """How many answers of each version fit, for each text and in all, and what the other answers had wrong."""
    out = {"per_text": {v: {key: sum(fits(a["text"]) for a in answers(v, key)) for key in TEXTS} for v in TEMPLATES}}
    out["total"] = {v: sum(out["per_text"][v].values()) for v in TEMPLATES}
    out["asked"] = {v: SEEDS * len(TEXTS) for v in TEMPLATES}
    wrong = [a["text"] for key in TEXTS for a in answers(1, key) if not fits(a["text"])]   # the answers of version 1 that do not fit
    out["text_prices"] = sum(isinstance(json.loads(t).get("price"), str) for t in wrong)
    out["wrong"] = len(wrong)
    return out


# --- the pictures ---------------------------------------------------------------------------------------------------------

_CELL = "padding:0.18rem 0.5rem;border-bottom:1px solid var(--ea-line);vertical-align:middle;font-size:13px"
_HEAD = f"{_CELL};color:var(--ea-ink-2);font-weight:500;text-align:left;font-family:var(--ea-font-ui)"


def _label(text: str) -> str:
    return f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2);margin:0.2rem 0">{html.escape(text)}</div>'


def _code(inner: str) -> str:
    return (f'<div style="font-family:var(--ea-font-code);font-size:13px;line-height:1.5;padding:0.4rem 0.6rem;'
            f'background:var(--ea-inset);border-radius:4px;white-space:pre-wrap;color:var(--ea-ink)">{inner}</div>')


def _filled(text: str) -> str:
    """What code filled in: a value in a field."""
    return f'<span style="color:var(--ea-data);font-weight:600">{html.escape(text)}</span>'


def _field(name: str) -> str:
    return f'<span style="color:var(--ea-data);font-weight:600">{{{html.escape(name)}}}</span>'


_FILL_WORDS = {
    "en": {"title": "One template, three prompts", "template": "The template, with the field {text}:",
           "filled": "TEMPLATE.format(text=...) with each text about a bike gives:"},
    "nl": {"title": "Eén template, drie prompts", "template": "De template, met het veld {text}:",
           "filled": "TEMPLATE.format(text=...) met elke tekst over een fiets geeft:"},
}


def fill_picture(lang: str) -> str:
    """Version 1 of the bike template, with its field, and the three prompts it makes: the fixed words in ink, what
    code filled in (in the colour of data)."""
    w = _FILL_WORDS[lang]
    head, _ = TEMPLATES[1].split("{text}")
    body = _label(w["template"]) + _code(html.escape(head) + _field("text")) + _label(w["filled"])
    for text in TEXTS.values():
        body += f'<div style="margin:0.25rem 0">{_code(html.escape(head) + _filled(text))}</div>'
    return yours.figure(w["title"], body, alt=w["title"])


_LOG_WORDS = {
    "en": {"title": "Which version gave which answers", "version": "version {v}", "text": "text", "runs": "the eight runs",
           "count": "{n} of {k}", "total": "version {v}: {n} of {k} answers fit", "fits": "fits", "not": "does not fit"},
    "nl": {"title": "Welke versie welke antwoorden gaf", "version": "versie {v}", "text": "tekst", "runs": "de acht runs",
           "count": "{n} van {k}", "total": "versie {v}: {n} van de {k} antwoorden passen", "fits": "past", "not": "past niet"},
}


def log_picture(lang: str) -> str:
    """The course's log of both versions: for each version and text, a square for each of the eight runs (an answer
    that fits in the colour of model output, one that does not in the colour of mistakes), how many fit, and under
    it each version's total."""
    w = _LOG_WORDS[lang]
    rows = []
    entries = log()
    for v in TEMPLATES:
        rows.append(f'<tr><td colspan="3" style="{_CELL};font-family:var(--ea-font-ui);color:var(--ea-ink);font-weight:600;padding-top:0.5rem">'
                    f'{html.escape(w["version"].format(v=v))}</td></tr>')
        for key in TEXTS:
            mine = [e for e in entries if e["version"] == v and e["text"] == key]
            squares = "".join(f'<span title="{w["fits"] if e["fits"] else w["not"]}" style="display:inline-block;width:12px;height:12px;'
                              f'margin-right:3px;border-radius:2px;background:{"var(--ea-pred)" if e["fits"] else "var(--ea-loss)"}"></span>'
                              for e in mine)
            n = sum(e["fits"] for e in mine)
            rows.append(f'<tr><td style="{_CELL};font-family:var(--ea-font-code);color:var(--ea-ink)">{key}</td>'
                        f'<td style="{_CELL}">{squares}</td>'
                        f'<td style="{_CELL};font-family:var(--ea-font-ui);color:var(--ea-ink)">{w["count"].format(n=n, k=len(mine))}</td></tr>')
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%"><thead><tr>'
             f'<th style="{_HEAD}">{w["text"]}</th><th style="{_HEAD}">{w["runs"]}</th><th style="{_HEAD}"></th>'
             f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    legend = (f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2);margin-top:0.3rem">'
              f'<span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:var(--ea-pred);vertical-align:middle"></span> {w["fits"]} &nbsp; '
              f'<span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:var(--ea-loss);vertical-align:middle"></span> {w["not"]}</div>')
    totals = "".join(_label(w["total"].format(v=v, n=sum(e["fits"] for e in entries if e["version"] == v),
                                              k=sum(e["version"] == v for e in entries))) for v in TEMPLATES)
    return yours.figure(w["title"], table + legend + totals, alt=w["title"])


# --- exercise 1: the braces of the template ------------------------------------------------------------------------------

START_PROMPT = '''TEMPLATE = ("Give the name, color and price of this bike as JSON, with the keys name, color and price. "
            'The price is a number, without quotes, such as {"price": 100}.\\n\\n{text}')


def bike_prompt(text):
    return TEMPLATE.format(text=text)
'''
SOLUTION_PROMPT = START_PROMPT.replace('{"price": 100}', '{{"price": 100}}')
PROMPT_CASES = [(TEXTS["pixie"],), (TEXTS["cityline"],), ("A bike called {Zoom}, for 300 euros.",)]


def _prompt_error(kind: str):
    def match(e) -> bool:
        if kind == "close":
            return isinstance(e, ValueError) and "Single '}'" in str(e)
        if kind == "open":
            return isinstance(e, ValueError) and "Single '{'" in str(e)
        return isinstance(e, KeyError) and "price" in str(e)
    return match


_PROMPT_WORDS = {
    "en": {"title": "The prompt your bike_prompt makes", "error": "bike_prompt stops with {e}", "none": "nothing yet",
           "same": "Right: the example keeps its braces, and the text about the bike is in the field.",
           "want": "It should make:"},
    "nl": {"title": "De prompt die je bike_prompt maakt", "error": "bike_prompt stopt met {e}", "none": "nog niets",
           "same": "Goed: het voorbeeld houdt zijn accolades, en de tekst over de fiets staat in het veld.",
           "want": "Hij zou moeten maken:"},
}


def prompt_picture(seen) -> str:
    """The prompt the learner's bike_prompt makes of the Pixie's text, with the text in the colour of data; when it
    stops with an error, the error; when it differs, the prompt it should make."""
    w = _PROMPT_WORDS[seen.lang]
    mine = seen.mine
    fn = mine.names.get("bike_prompt") if mine is not None else None
    if not callable(fn):
        return ""
    text = PROMPT_CASES[0][0]
    want = seen.remember("prompt", lambda: run_code(SOLUTION_PROMPT).names["bike_prompt"](text))

    def shown(prompt: str) -> str:
        return _code(html.escape(prompt).replace(html.escape(text), _filled(text)))
    try:
        got = with_time_limit(lambda: fn(text))
    except Exception as e:   # the learner's code; the check says what went wrong
        body = _label(w["error"].format(e=f"{type(e).__name__}: {e}")) + _label(w["want"]) + shown(want)
        return yours.figure(w["title"], body, alt=w["title"])
    if not isinstance(got, str):
        return ""
    body = shown(got) + (_label(w["same"]) if got == want else _label(w["want"]) + shown(want))
    return yours.figure(w["title"], body, alt=w["title"])


# --- exercise 2: what fit, per version --------------------------------------------------------------------------------------

START_TALLY = '''def fits_per_version(log):
    counts = {}
    for entry in log:
        version = entry["version"]
    return counts
'''
SOLUTION_TALLY = START_TALLY.replace('''        version = entry["version"]
''', '''        version = entry["version"]
        if version not in counts:
            counts[version] = 0
        if entry["fits"]:
            counts[version] = counts[version] + 1
''')
SMALL_LOG = [{"prompt": NAME, "version": 1, "fits": True}, {"prompt": NAME, "version": 1, "fits": False},
             {"prompt": NAME, "version": 2, "fits": True}, {"prompt": NAME, "version": 2, "fits": True}]
NONE_FIT = [{"prompt": NAME, "version": 3, "fits": False}, {"prompt": NAME, "version": 4, "fits": True},
            {"prompt": NAME, "version": 3, "fits": False}]


def _course_log() -> list:
    """The course's log without the answers' texts: the check's longest case."""
    return [{k: e[k] for k in ("prompt", "version", "text", "seed", "fits")} for e in log()]


_TALLY_WORDS = {
    "en": {"title": "What your fits_per_version counts in the course's log", "version": "version {v}",
           "mine": "yours: {n}", "want": "should be: {n}", "error": "fits_per_version stops with {e}"},
    "nl": {"title": "Wat je fits_per_version telt in het log van de course", "version": "versie {v}",
           "mine": "van jou: {n}", "want": "zou moeten zijn: {n}", "error": "fits_per_version stopt met {e}"},
}


def tally_picture(seen) -> str:
    """For each version in the course's log, a bar of the answers that fit by the learner's fits_per_version, out of
    the answers of that version, and where it differs, what it should count."""
    w = _TALLY_WORDS[seen.lang]
    mine = seen.mine
    fn = mine.names.get("fits_per_version") if mine is not None else None
    if not callable(fn):
        return ""
    entries = _course_log()
    want = seen.remember("tally", lambda: run_code(SOLUTION_TALLY).names["fits_per_version"](_course_log()))
    try:
        got = with_time_limit(lambda: fn([dict(e) for e in entries]))
    except Exception as e:   # the learner's code; the check says what went wrong
        return yours.figure(w["title"], _label(w["error"].format(e=type(e).__name__)), alt=w["title"])
    if not isinstance(got, dict):
        return ""
    rows = []
    for v in TEMPLATES:
        total = sum(e["version"] == v for e in entries)
        n = got.get(v)
        ok = n == want[v]
        width = 100 * n / total if isinstance(n, (int, float)) and total and 0 <= n <= total else 0
        rows.append(f'<div style="display:flex;align-items:center;gap:0.5rem;margin:0.25rem 0;font-family:var(--ea-font-ui);font-size:13px">'
                    f'<span style="min-width:5rem;color:var(--ea-ink)">{html.escape(w["version"].format(v=v))}</span>'
                    f'<span style="display:inline-block;width:10rem;height:12px;background:var(--ea-inset);border-radius:2px">'
                    f'<span style="display:block;height:12px;width:{width:.1f}%;background:var(--ea-pred);border-radius:2px"></span></span>'
                    f'<span style="color:{"var(--ea-ink)" if ok else "var(--ea-loss-text)"}">{html.escape(w["mine"].format(n=n))} / {total}</span>'
                    + ("" if ok else f'<span style="color:var(--ea-ink-2)">{html.escape(w["want"].format(n=want[v]))}</span>') + "</div>")
    return yours.figure(w["title"], "".join(rows), alt=w["title"])


_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}

EXERCISES = {
    "bike_prompt": FunctionBox(
        name="bike_prompt",
        picture=prompt_picture,
        function="bike_prompt",
        start=START_PROMPT,
        solution=SOLUTION_PROMPT,
        visible=PROMPT_CASES[:1],
        hidden=PROMPT_CASES[1:],
        mistakes=[
            Mistake("single-close", error=_prompt_error("close"),
                    en="With `{call}`, `.format` stops: *Single '}}' encountered*. You doubled the opening brace of the example; double its closing brace too, `}}}}`.",
                    nl="Met `{call}` stopt `.format`: *Single '}}' encountered*. Je hebt de openingsaccolade van het voorbeeld verdubbeld; verdubbel de sluitende ook, `}}}}`."),
            Mistake("single-open", error=_prompt_error("open"),
                    en="With `{call}`, `.format` stops: *Single '{{' encountered*. Each brace that should stay in the prompt is written twice: `{{{{` and `}}}}`.",
                    nl="Met `{call}` stopt `.format`: *Single '{{' encountered*. Elke accolade die in de prompt moet blijven, schrijf je dubbel: `{{{{` en `}}}}`."),
            Mistake("key-error", error=_prompt_error("key"),
                    en="With `{call}`, `.format` stops with a KeyError: it reads `{{\"price\": 100}}` as a field named `\"price\"`, and gets no value for it. A brace that should stay in the prompt is written twice.",
                    nl="Met `{call}` stopt `.format` met een KeyError: het leest `{{\"price\": 100}}` als een veld met de naam `\"price\"`, en krijgt daar geen waarde voor. Een accolade die in de prompt moet blijven, schrijf je dubbel."),
            Mistake("field-doubled", wrong=SOLUTION_PROMPT.replace("\\n\\n{text}", "\\n\\n{{text}}"),
                    en="`{call}` gives a prompt that still ends with the name of the field in braces, not with the text about the bike. The field keeps single braces: only the braces of the example are doubled.",
                    nl="`{call}` geeft een prompt die nog eindigt met de naam van het veld tussen accolades, niet met de tekst over de fiets. Het veld houdt enkele accolades: alleen de accolades van het voorbeeld worden dubbel."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`bike_prompt` still stops with a KeyError: `.format` reads the braces of the example as a field. Write each brace of the example twice, so that `.format` leaves one of each in the prompt.",
                      "nl": "`bike_prompt` stopt nog met een KeyError: `.format` leest de accolades van het voorbeeld als een veld. Schrijf elke accolade van het voorbeeld dubbel, zodat `.format` er telkens één in de prompt laat staan."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Keep the words of the template as they were, and change only the braces of the example.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Laat de woorden van de template zoals ze waren, en verander alleen de accolades van het voorbeeld."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Keep the words of the template as they were, and change only the braces of the example.",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Laat de woorden van de template zoals ze waren, en verander alleen de accolades van het voorbeeld."},
            "no_function": {"en": "Your box does not make a function `bike_prompt`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `bike_prompt`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your template is right for all {n} texts: the example keeps its braces, the text goes in the field, and braces in the text itself stay as they are.",
                        "nl": "Je template klopt voor alle {n} teksten: het voorbeeld houdt zijn accolades, de tekst komt in het veld, en accolades in de tekst zelf blijven zoals ze zijn."},
        },
    ),
    "fits_per_version": FunctionBox(
        name="fits_per_version",
        picture=tally_picture,
        function="fits_per_version",
        start=START_TALLY,
        solution=SOLUTION_TALLY,
        visible=[(SMALL_LOG,)],
        hidden=[(NONE_FIT,), ([],)],
        mistakes=[
            Mistake("all-answers", wrong=SOLUTION_TALLY.replace('''        if entry["fits"]:
            counts[version] = counts[version] + 1
''', '''        counts[version] = counts[version] + 1
'''),
                    en="`{call}` gives `{got}`: it counts every answer of a version. Count only the answers with `entry[\"fits\"]` `True`.",
                    nl="`{call}` geeft `{got}`: het telt elk antwoord van een versie. Tel alleen de antwoorden waarbij `entry[\"fits\"]` `True` is."),
            Mistake("left-out", wrong='''def fits_per_version(log):
    counts = {}
    for entry in log:
        version = entry["version"]
        if entry["fits"]:
            if version in counts:
                counts[version] = counts[version] + 1
            else:
                counts[version] = 1
    return counts
''', state="almost",
                    en="Right for the example. But `{call}` gives `{got}`: a version none of whose answers fit is missing. Give every version in the log a count, also when it is 0, so the log shows that it was tried.",
                    nl="Goed voor het voorbeeld. Maar `{call}` geeft `{got}`: een versie waarvan geen enkel antwoord paste, ontbreekt. Geef elke versie in het log een telling, ook als die 0 is, zodat te zien is dat hij geprobeerd is."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`fits_per_version` still gives back an empty dictionary. In the loop, give each version a count, starting at 0, and add 1 for every answer of it that fits.",
                      "nl": "`fits_per_version` geeft nog een lege dictionary terug. Geef in de loop elke versie een telling die begint bij 0, en tel er 1 bij op voor elk antwoord ervan dat past."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: for each version, how many of its answers have `\"fits\"` `True`.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: voor elke versie hoeveel van zijn antwoorden `\"fits\"` `True` hebben."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for a version none of whose answers fit, and for an empty log?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Werkt je functie ook voor een versie waarvan geen enkel antwoord past, en voor een leeg log?"},
            "no_function": {"en": "Your box does not make a function `fits_per_version`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `fits_per_version`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `fits_per_version` is right in all {n} cases: every version in the log, with how many of its answers fit, 0 included.",
                        "nl": "Je `fits_per_version` klopt in alle {n} gevallen: elke versie in het log, met hoeveel van zijn antwoorden passen, 0 ook."},
        },
    ),
}

REQUIRED = ["bike_prompt", "fits_per_version", "new-version"]

CASES = [
    ("bike_prompt", START_PROMPT, "empty", None),
    ("bike_prompt", "", "empty", None),
    ("bike_prompt", SOLUTION_PROMPT, "correct", None),
    ("bike_prompt", START_PROMPT.replace('{"price": 100}', '{{"price": 100}'), "error", "single-close"),
    ("bike_prompt", START_PROMPT.replace('{"price": 100}', '{"price": 100}}'), "error", "key-error"),
    ("bike_prompt", START_PROMPT.replace('{"price": 100}', '{{"price": 100}}').replace("\\n\\n{text}", "\\n\\n{text}{"), "error", "single-open"),
    ("bike_prompt", SOLUTION_PROMPT.replace("\\n\\n{text}", "\\n\\n{{text}}"), "wrong", "field-doubled"),
    ("bike_prompt", START_PROMPT.replace(", such as {\"price\": 100}", ""), "wrong", None),
    ("fits_per_version", START_TALLY, "empty", None),
    ("fits_per_version", "", "empty", None),
    ("fits_per_version", SOLUTION_TALLY, "correct", None),
    ("fits_per_version", '''def fits_per_version(log):
    counts = {}
    for entry in log:
        version = entry["version"]
        counts[version] = counts.get(version, 0) + (1 if entry["fits"] else 0)
    return counts
''', "correct", None),
    ("fits_per_version", SOLUTION_TALLY.replace('''        if entry["fits"]:
            counts[version] = counts[version] + 1
''', '''        counts[version] = counts[version] + 1
'''), "wrong", "all-answers"),
    ("fits_per_version", '''def fits_per_version(log):
    counts = {}
    for entry in log:
        version = entry["version"]
        if entry["fits"]:
            if version in counts:
                counts[version] = counts[version] + 1
            else:
                counts[version] = 1
    return counts
''', "almost", "left-out"),
]


# --- the demo: a version of your own, against the course's own server -----------------------------------------------------

def downloaded() -> bool:
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    chat.download_or_stop(button, texts)


def _post(url: str, body: dict) -> dict:
    import urllib.request
    request = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def demo(template: str, words: dict, lang: str) -> str:
    """The learner's template filled with each of the three texts with .format, and sent in JSON mode with seed 0 to
    the course's server with Qwen3 behind it: each prompt's answer, and whether it fits. When .format stops, what it
    said."""
    try:
        prompts = {key: template.format(text=text) for key, text in TEXTS.items()}
    except (KeyError, ValueError, IndexError, AttributeError) as e:
        return f'<p><em>{html.escape(words["format"].format(why=f"{type(e).__name__}: {e}"))}</em></p>'
    rows = []
    try:
        running = serve.model_server()
        for key, prompt in prompts.items():
            answer = _post(running.url, {"model": serve.MODEL, "messages": [{"role": "user", "content": prompt}],
                                         "max_tokens": 80, "seed": 0, "response_format": JSON_MODE})
            text = answer["choices"][0]["message"]["content"]
            ok = fits(text)
            rows.append(f'<tr><td style="{_CELL};font-family:var(--ea-font-code);color:var(--ea-ink)">{key}</td>'
                        f'<td style="{_CELL};font-family:var(--ea-font-code);color:var(--ea-pred-text)">{html.escape(" ".join(text.split()))}</td>'
                        f'<td style="{_CELL};font-family:var(--ea-font-ui);color:{"var(--ea-ink-2)" if ok else "var(--ea-loss-text)"};font-weight:600">'
                        f'{words["fits"] if ok else words["not"]}</td></tr>')
    except Exception as e:   # the real server; it may have run out of memory
        said = words["memory"] if serve.out_of_memory(e) else words["failed"].format(why=f"{type(e).__name__}: {e}")
        return f"<p><em>{html.escape(said)}</em></p>"
    table = f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%"><tbody>{"".join(rows)}</tbody></table></div>'
    return yours.figure(words["title"], table, alt=words["title"])
