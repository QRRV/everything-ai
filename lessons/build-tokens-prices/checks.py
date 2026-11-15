# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-tokens-prices: what one call costs from its tokens, and the most a call can cost before it is
sent, with GPT-2's tokenizer from the course's own copy (everything_ai.gpt2_tokenizer); the course's counts with
Qwen3's tokenizer and two answers of Qwen3 through its own server (data/course-apis-qwen.json); the pictures; and the
demo, which counts a prompt with Qwen3's tokenizer and then sends it."""
import html
import json
import time
import urllib.request

from everything_ai import chat, gpt2_tokenizer, paths, serve, yours
from everything_ai.check import FunctionBox, Mistake

gpt2_tokenizer.offline()   # before any tiktoken.get_encoding("gpt2"), also in the learner's code

import tiktoken  # noqa: E402  (after offline(): tiktoken reads its cache folder when it loads a file)

ENC = tiktoken.get_encoding("gpt2")
SHORT = "In one sentence: what is a token?"
LONG = "Explain what a token is, with three examples."
TEXTS = [
    "The weather is lovely today, so we are going for a walk in the park.",
    "Het weer is vandaag heerlijk, dus we gaan wandelen in het park.",
    "for i in range(10):\n    print(i * i)",
    "3.14159265358979",
    SHORT,
]
RECORD = {
    "count": {"count": TEXTS},
    "short": {"messages": [{"role": "user", "content": SHORT}], "max_tokens": 600, "seeds": 1},
    "long": {"messages": [{"role": "user", "content": LONG}], "max_tokens": 600, "seeds": 1},
}
PRICE_IN, PRICE_OUT = 3, 15   # made-up prices, in dollars per million tokens
DATA = paths.ROOT / "data" / "course-apis-qwen.json"


def _part() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-tokens-prices"]


def counts() -> list:
    """For each text: (text, words, tokens with GPT-2's tokenizer, tokens with Qwen3's)."""
    qwen = {a["text"]: a["tokens"] for a in _part()["count"]["answers"]}
    return [(t, len(t.split()), len(ENC.encode(t)), qwen[t]) for t in TEXTS]


def answers() -> dict:
    """The course's two answers: tokens in and out, seconds, and what each cost at the made-up prices."""
    out = {}
    for key in ("short", "long"):
        a = _part()[key]["answers"][0]
        out[key] = {"in": a["prompt_tokens"], "out": a["tokens"], "seconds": a["seconds"], "finish": a["finish"], "text": a["text"],
                    "cost": a["prompt_tokens"] * PRICE_IN / 1_000_000 + a["tokens"] * PRICE_OUT / 1_000_000}
    return out


def findings() -> dict:
    """The numbers the lesson's text names."""
    c, a = counts(), answers()
    return {"en_qwen": c[0][3], "nl_qwen": c[1][3], "en_gpt2": c[0][2], "nl_gpt2": c[1][2], "words_en": c[0][1], "words_nl": c[1][1],
            "number_gpt2": c[3][2], "number_qwen": c[3][3],
            "short_text": c[4][3], "short_in": a["short"]["in"], "template": a["short"]["in"] - c[4][3],
            "short_out": a["short"]["out"], "long_out": a["long"]["out"], "times": round(a["long"]["out"] / a["short"]["out"]),
            "short_seconds": a["short"]["seconds"], "long_seconds": a["long"]["seconds"],
            "short_cost": a["short"]["cost"], "long_cost": a["long"]["cost"]}


def _n(x, lang: str, digits: int = 1) -> str:
    s = f"{x:.{digits}f}"
    return s.replace(".", ",") if lang == "nl" else s


def numbers(lang: str) -> dict:
    """The findings as the lesson's text writes them, in the lesson's language."""
    f = findings()
    out = {k: str(v) for k, v in f.items()}
    for k in ("short_seconds", "long_seconds"):
        out[k] = _n(f[k], lang)
    for k in ("short_cost", "long_cost"):
        out[k] = _n(f[k], lang, 6)
    return out


_COUNT_WORDS = {
    "en": {"title": "The same texts, counted by two tokenizers", "words": "words", "gpt2": "GPT-2's tokenizer", "qwen": "Qwen3's tokenizer",
           "texts": ["English sentence", "The same in Dutch", "Two lines of Python", "A number", "The question of the course"]},
    "nl": {"title": "Dezelfde teksten, geteld door twee tokenizers", "words": "woorden", "gpt2": "tokenizer van GPT-2", "qwen": "tokenizer van Qwen3",
           "texts": ["Engelse zin", "Hetzelfde in het Nederlands", "Twee regels Python", "Een getal", "De vraag van de course"]},
}


def counts_picture(lang: str) -> str:
    """Per text: the words, and the tokens of each tokenizer, as bars; the text itself under its name."""
    w = _COUNT_WORDS[lang]
    rows = counts()
    top = max(max(g, q) for _t, _w, g, q in rows)
    out = []
    for name, (text, words, g, q) in zip(w["texts"], rows):
        bars = []
        for label, value, colour in ((w["gpt2"], g, "var(--ea-ink-3)"), (w["qwen"], q, "var(--ea-ink)")):
            bars.append(f'<div style="display:flex;align-items:center;gap:0.4rem;margin:0.1rem 0"><div style="width:9.5rem;font-family:var(--ea-font-ui);'
                        f'font-size:11px;color:var(--ea-ink-2)">{html.escape(label)}</div><div style="flex:1"><div style="width:{100 * value / top:.1f}%;'
                        f'height:11px;background:{colour};border-radius:2px"></div></div><div style="width:2.2rem;text-align:right;font-family:var(--ea-font-code);'
                        f'font-size:12px;color:var(--ea-ink)">{value}</div></div>')
        shown = text.replace("\n", " ⏎ ")
        out.append(f'<div style="padding:0.35rem 0;border-top:1px solid var(--ea-line)"><div style="font-family:var(--ea-font-ui);font-size:12px;'
                   f'color:var(--ea-ink)">{html.escape(name)} <span style="color:var(--ea-ink-2)">· {words} {html.escape(w["words"])}</span></div>'
                   f'<div style="font-family:var(--ea-font-code);font-size:11.5px;color:var(--ea-ink-2);margin:0.1rem 0 0.2rem;overflow-wrap:anywhere">'
                   f'{html.escape(shown)}</div>{"".join(bars)}</div>')
    return yours.figure(w["title"], "".join(out), alt=w["title"])


_ANSWER_WORDS = {
    "en": {"title": "Two questions to Qwen3, and their bill", "short": "Short: “{q}”", "long": "Long: “{q}”", "in": "tokens in",
           "out": "tokens out", "seconds": "{s} seconds", "cost": "at 3 and 15 dollars per million: ${c}"},
    "nl": {"title": "Twee vragen aan Qwen3, en hun rekening", "short": "Kort: “{q}”", "long": "Lang: “{q}”", "in": "tokens erin",
           "out": "tokens eruit", "seconds": "{s} seconden", "cost": "tegen 3 en 15 dollar per miljoen: ${c}"},
}


def answers_picture(lang: str) -> str:
    """The course's two answers: tokens in and tokens out as bars, with the seconds and what each cost; tokens out in
    the colour of what a model writes."""
    w, a = _ANSWER_WORDS[lang], answers()
    top = max(max(x["in"], x["out"]) for x in a.values())
    out = []
    for key, q in (("short", SHORT), ("long", LONG)):
        x = a[key]
        bars = []
        for label, value, colour in ((w["in"], x["in"], "var(--ea-ink-3)"), (w["out"], x["out"], "var(--ea-pred-text)")):
            bars.append(f'<div style="display:flex;align-items:center;gap:0.4rem;margin:0.1rem 0"><div style="width:6.5rem;font-family:var(--ea-font-ui);'
                        f'font-size:11px;color:var(--ea-ink-2)">{html.escape(label)}</div><div style="flex:1"><div style="width:{100 * value / top:.1f}%;'
                        f'height:11px;background:{colour};border-radius:2px"></div></div><div style="width:2.5rem;text-align:right;font-family:var(--ea-font-code);'
                        f'font-size:12px;color:var(--ea-ink)">{value}</div></div>')
        note = w["seconds"].format(s=_n(x["seconds"], lang)) + " · " + w["cost"].format(c=_n(x["cost"], lang, 6))
        out.append(f'<div style="padding:0.35rem 0;border-top:1px solid var(--ea-line)"><div style="font-family:var(--ea-font-ui);font-size:12px;'
                   f'color:var(--ea-ink)">{html.escape(w[key].format(q=q))}</div>{"".join(bars)}<div style="font-family:var(--ea-font-ui);font-size:11px;'
                   f'color:var(--ea-ink-2)">{html.escape(note)}</div></div>')
    return yours.figure(w["title"], "".join(out), alt=w["title"])


# --- exercise 1: what a call costs --------------------------------------------------------------------------------------

_START_COST = '''def cost(tokens_in, tokens_out, price_in, price_out):
    return 0
'''
SOLUTION_COST = '''def cost(tokens_in, tokens_out, price_in, price_out):
    return tokens_in * price_in / 1_000_000 + tokens_out * price_out / 1_000_000
'''
_SHORT = answers()["short"]   # the course's short call: its tokens in and out are the example of exercise 1
COST_CASES = [(_SHORT["in"], _SHORT["out"], 3, 15), (1000, 500, 3, 15), (0, 0, 3, 15), (1_000_000, 0, 3, 15), (0, 1_000_000, 3, 15), (2500, 800, 0.5, 1.5)]


def cost_picture(seen) -> str:
    """What the learner's cost says for the course's two answers, beside what they cost."""
    run = seen.mine
    fn = run.names.get("cost") if run is not None else None
    if not callable(fn):
        return ""
    w = {"en": ("Your cost for the course's two answers", "short answer", "long answer", "your cost", "should be", "error"),
         "nl": ("Je cost voor de twee antwoorden van de course", "kort antwoord", "lang antwoord", "jouw cost", "zou zijn", "fout")}[seen.lang]
    a = answers()
    rows = []
    for key, label in (("short", w[1]), ("long", w[2])):
        x = a[key]
        try:
            got = float(fn(x["in"], x["out"], PRICE_IN, PRICE_OUT))
            mine = "$" + _n(got, seen.lang, 6)
            ok = abs(got - x["cost"]) <= 1e-9 + 1e-6 * abs(x["cost"])
        except Exception:   # the learner's code
            mine, ok = w[5], False
        colour = "var(--ea-ink)" if ok else "var(--ea-loss-text)"
        rows.append(f'<tr><td style="padding:0.15rem 0.6rem 0.15rem 0">{html.escape(label)}</td><td style="font-family:var(--ea-font-code)">{x["in"]} + {x["out"]}</td>'
                    f'<td style="font-family:var(--ea-font-code);color:{colour};padding-left:0.6rem">{html.escape(mine)}</td>'
                    f'<td style="font-family:var(--ea-font-code);color:var(--ea-ink-2);padding-left:0.6rem">${_n(x["cost"], seen.lang, 6)}</td></tr>')
    head = (f'<tr style="color:var(--ea-ink-2)"><td></td><td>tokens</td><td style="padding-left:0.6rem">{html.escape(w[3])}</td>'
            f'<td style="padding-left:0.6rem">{html.escape(w[4])}</td></tr>')
    body = f'<table style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink);border-collapse:collapse">{head}{"".join(rows)}</table>'
    return yours.figure(w[0], body, alt=w[0])


# --- exercise 2: the most a call can cost, before you send it ----------------------------------------------------------

_HEAD = 'import tiktoken\n\nenc = tiktoken.get_encoding("gpt2")\n\n\n'
_START_MOST = _HEAD + '''def most(prompt, max_tokens, price_in, price_out):
    tokens_in = 0
    tokens_out = 0
    return 0
'''
SOLUTION_MOST = _HEAD + '''def most(prompt, max_tokens, price_in, price_out):
    tokens_in = len(enc.encode(prompt))
    tokens_out = max_tokens
    return tokens_in * price_in / 1_000_000 + tokens_out * price_out / 1_000_000
'''
MOST_CASES = [
    ("Name three colors of the rainbow.", 50, 3, 15),
    ("", 100, 3, 15),
    (LONG, 600, 0.5, 1.5),
    (TEXTS[1] * 20, 0, 3, 15),
    (TEXTS[2], 200, 1, 4),
    ("Tokenization is surprisingly important.", 10, 3, 15),
]


def most_picture(seen) -> str:
    """The learner's most for the example, part by part: the prompt's tokens, the answer's most, and the sum."""
    run = seen.mine
    fn = run.names.get("most") if run is not None else None
    if not callable(fn):
        return ""
    w = {"en": ("Your most, for the example", "the prompt: {n} tokens with GPT-2's tokenizer", "the answer: at most {n} tokens",
                "your most: {got}", "error"),
         "nl": ("Je most, voor het voorbeeld", "de prompt: {n} tokens met de tokenizer van GPT-2", "het antwoord: hooguit {n} tokens",
                "jouw most: {got}", "fout")}[seen.lang]
    prompt, max_tokens, p_in, p_out = MOST_CASES[0]
    try:
        got = "$" + _n(float(fn(prompt, max_tokens, p_in, p_out)), seen.lang, 6)
    except Exception:   # the learner's code
        got = w[4]
    lines = [w[1].format(n=len(ENC.encode(prompt))), w[2].format(n=max_tokens), w[3].format(got=got)]
    body = "".join(f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink);padding:0.1rem 0">{html.escape(t)}</div>' for t in lines)
    return yours.figure(w[0], body, alt=w[0])


_COMMON = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
                "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."},
}

EXERCISES = {
    "cost": FunctionBox(
        name="cost",
        picture=cost_picture,
        function="cost",
        start=_START_COST,
        solution=SOLUTION_COST,
        visible=COST_CASES[:1],
        hidden=COST_CASES[1:],
        mistakes=[
            Mistake("per-million", wrong=SOLUTION_COST.replace(" / 1_000_000", ""),
                    en="`{call}` gives `{got}`, a million times too much: the prices are per million tokens. Divide by `1_000_000`.",
                    nl="`{call}` geeft `{got}`, een miljoen keer te veel: de prijzen zijn per miljoen tokens. Deel door `1_000_000`."),
            Mistake("swapped", wrong=SOLUTION_COST.replace("tokens_in * price_in", "tokens_in * price_out").replace("tokens_out * price_out", "tokens_out * price_in"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. The tokens that go in cost `price_in`, the ones that come out `price_out`.",
                    nl="`{call}` geeft `{got}`, en het zou `{want}` moeten zijn. De tokens die erin gaan kosten `price_in`, die eruit komen `price_out`."),
            Mistake("one-price", wrong=SOLUTION_COST.replace("tokens_out * price_out", "tokens_out * price_in"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. The tokens that come out have their own price, `price_out`.",
                    nl="`{call}` geeft `{got}`, en het zou `{want}` moeten zijn. De tokens die eruit komen hebben hun eigen prijs, `price_out`."),
            Mistake("both-prices", wrong='''def cost(tokens_in, tokens_out, price_in, price_out):
    return (tokens_in + tokens_out) * (price_in + price_out) / 1_000_000
''',
                    en="`{call}` gives `{got}`: every token at both prices. Each kind of token has one price: tokens in times `price_in`, plus tokens out times `price_out`.",
                    nl="`{call}` geeft `{got}`: elk token tegen beide prijzen. Elke soort token heeft één prijs: tokens erin keer `price_in`, plus tokens eruit keer `price_out`."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "`cost` gives 0 yet. The tokens in times their price, plus the tokens out times theirs, per million.",
                      "nl": "`cost` geeft nog 0. De tokens erin keer hun prijs, plus de tokens eruit keer de hunne, per miljoen."},
            "correct": {"en": "Right: your `cost` prices the tokens in and the tokens out each at their own price per million, for all {n} calls the check tried.",
                        "nl": "Goed: je `cost` rekent de tokens erin en de tokens eruit elk tegen hun eigen prijs per miljoen, voor alle {n} aanroepen die de check probeerde."},
            "wrong": {"en": f"`{{call}}` gives `{{got}}`, and it should give `{{want}}`: {_SHORT['in']} tokens at 3 dollars per million, plus {_SHORT['out']} at 15.",
                      "nl": f"`{{call}}` geeft `{{got}}`, en het zou `{{want}}` moeten zijn: {_SHORT['in']} tokens tegen 3 dollar per miljoen, plus {_SHORT['out']} tegen 15."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`.",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn."},
        },
    ),
    "most": FunctionBox(
        name="most",
        picture=most_picture,
        function="most",
        start=_START_MOST,
        solution=SOLUTION_MOST,
        visible=MOST_CASES[:1],
        hidden=MOST_CASES[1:],
        mistakes=[
            Mistake("characters", wrong=SOLUTION_MOST.replace("len(enc.encode(prompt))", "len(prompt)"),
                    en="`{call}` gives `{got}`: `len(prompt)` counts characters, and a token is often a whole word, or a piece of one. Count with the tokenizer: `enc.encode(prompt)`.",
                    nl="`{call}` geeft `{got}`: `len(prompt)` telt tekens, en een token is vaak een heel woord, of een stuk ervan. Tel met de tokenizer: `enc.encode(prompt)`."),
            Mistake("words", wrong=SOLUTION_MOST.replace("len(enc.encode(prompt))", "len(prompt.split())"),
                    en="`{call}` gives `{got}`: `split` counts words, and the tokenizer cuts some words in more pieces, and a full stop is a token too. Count with `enc.encode(prompt)`.",
                    nl="`{call}` geeft `{got}`: `split` telt woorden, en de tokenizer knipt sommige woorden in meer stukken, en een punt is ook een token. Tel met `enc.encode(prompt)`."),
            Mistake("no-answer", wrong=SOLUTION_MOST.replace("tokens_out = max_tokens", "tokens_out = 0"),
                    en="`{call}` gives `{got}`: only the prompt. The answer costs too, and it has at most `max_tokens` tokens.",
                    nl="`{call}` geeft `{got}`: alleen de prompt. Het antwoord kost ook geld, en het heeft hooguit `max_tokens` tokens."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "`most` gives 0 yet. Count the prompt's tokens with `enc`, take the most the answer can have, and price both as in exercise 1.",
                      "nl": "`most` geeft nog 0. Tel de tokens van de prompt met `enc`, neem het meeste dat het antwoord kan hebben, en reken beide zoals in oefening 1."},
            "correct": {"en": "Right: your `most` counts the prompt with the tokenizer, takes `max_tokens` for the answer, and prices both, for all {n} prompts the check tried.",
                        "nl": "Goed: je `most` telt de prompt met de tokenizer, neemt `max_tokens` voor het antwoord, en rekent beide, voor alle {n} prompts die de check probeerde."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. The prompt has as many tokens as `enc.encode(prompt)` gives back.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn. De prompt heeft zoveel tokens als `enc.encode(prompt)` teruggeeft."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`.",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn."},
        },
    ),
}

REQUIRED = ["cost", "most", "cheaper"]

CASES = [
    ("cost", _START_COST, "empty", None),
    ("cost", SOLUTION_COST, "correct", None),
    ("cost", SOLUTION_COST.replace(" / 1_000_000", ""), "wrong", "per-million"),
    ("cost", SOLUTION_COST.replace("tokens_out * price_out", "tokens_out * price_in"), "wrong", "one-price"),
    ("cost", SOLUTION_COST.replace("tokens_in * price_in", "tokens_in * price_out").replace("tokens_out * price_out", "tokens_out * price_in"), "wrong", "swapped"),
    ("cost", "def cost(tokens_in, tokens_out, price_in, price_out):\n    return (tokens_in + tokens_out) * (price_in + price_out) / 1_000_000\n", "wrong", "both-prices"),
    ("most", _START_MOST, "empty", None),
    ("most", SOLUTION_MOST, "correct", None),
    ("most", SOLUTION_MOST.replace("len(enc.encode(prompt))", "len(prompt)"), "wrong", "characters"),
    ("most", SOLUTION_MOST.replace("len(enc.encode(prompt))", "len(prompt.split())"), "wrong", "words"),
    ("most", SOLUTION_MOST.replace("tokens_out = max_tokens", "tokens_out = 0"), "wrong", "no-answer"),
]


# --- the demo: count with Qwen3's tokenizer, then send ------------------------------------------------------------------

def downloaded() -> bool:
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    chat.download_or_stop(button, texts)


def demo(prompt: str, max_tokens: int, words: dict, lang: str) -> str:
    """The prompt counted with Qwen3's tokenizer and with GPT-2's, then sent to the course's server with Qwen3: what
    the server counted in and out, and what that costs at the made-up prices."""
    from everything_ai import qwen3
    enc = qwen3.tokenizer()
    counted = len(qwen3.encode(enc, prompt))
    running = serve.model_server()
    body = {"model": serve.MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
    request = urllib.request.Request(running.url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            answer = json.loads(response.read())
    except Exception as e:   # the server ran out of memory, or could not be reached
        said = words["memory"] if serve.out_of_memory(e) else words["failed"].format(why=f"{type(e).__name__}: {e}")
        return f"<p><em>{html.escape(said)}</em></p>"
    seconds = time.perf_counter() - start
    usage = answer["usage"]
    template = findings()["template"]
    worst = (counted + template) * PRICE_IN / 1_000_000 + max_tokens * PRICE_OUT / 1_000_000
    real = usage["prompt_tokens"] * PRICE_IN / 1_000_000 + usage["completion_tokens"] * PRICE_OUT / 1_000_000
    note = words["done"].format(counted=counted, gpt2=len(ENC.encode(prompt)), template=template, sent=usage["prompt_tokens"], out=usage["completion_tokens"],
                                most=max_tokens, worst=_n(worst, lang, 6), real=_n(real, lang, 6), seconds=_n(seconds, lang))
    return chat.answer_picture(answer["choices"][0]["message"]["content"] or "", note, words["title"])
