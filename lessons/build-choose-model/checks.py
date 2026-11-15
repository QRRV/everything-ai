# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-choose-model: what a right answer costs, from a list of runs; a router that sends sums to the
thinking way and every other question to the normal way; the course's measurement of Qwen3-0.6B in both ways on
eighteen questions (data/course-apis-qwen.json); the pictures; and the demo, the learner's router in front of the
course's own server."""
import html
import json
import re
import time
import urllib.request

from everything_ai import chat, paths, serve, yours
from everything_ai.check import MISSING, FunctionBox, Mistake, kept, run_code

# Eighteen questions: ten sums in words, eight short questions of fact or language. Each with what a right answer has:
# a number, which must be the last number in the answer, or words, one of which must be in it.
QUESTIONS = [
    ("A shop sells pens at 3 euros each. Tom buys 7 pens and pays with a 50 euro note. How much change does Tom get?", 29),
    ("A book has 320 pages. Mia reads 25 pages a day. After how many days has Mia read more than half of the book?", 7),
    ("What is 17 times 24?", 408),
    ("A class has 28 students, and 3 out of every 4 have a bike. How many students do not have a bike?", 7),
    ("A parent is 3 times as old as their child. In 12 years, the parent will be twice as old as the child. How old is the child now?", 12),
    ("A recipe for 4 people uses 300 grams of flour. How many grams of flour do you need for 10 people?", 750),
    ("What is 15% of 240?", 36),
    ("A rectangle is 12 cm long and its area is 96 square cm. What is its perimeter in cm?", 40),
    ("If 5 machines make 5 parts in 5 minutes, how many minutes do 100 machines need to make 100 parts?", 5),
    ("What is 1001 minus 487?", 514),
    ("What is the capital of Japan?", ["tokyo"]),
    ("Translate 'good morning' into French.", ["bonjour"]),
    ("What is the opposite of 'cold'?", ["hot", "warm"]),
    ("Which planet is known as the Red Planet?", ["mars"]),
    ("What color do you get when you mix blue and yellow?", ["green"]),
    ("What is the largest ocean on Earth?", ["pacific"]),
    ("Who wrote 'Romeo and Juliet'?", ["shakespeare"]),
    ("What is the plural of 'mouse'?", ["mice"]),
]
FAST_SEEDS, THINK_SEEDS = 4, 2
FAST_TOKENS, THINK_TOKENS = 300, 2000
PRICE_OUT = 15   # a made-up price, in dollars per million output tokens
DATA = paths.ROOT / "data" / "course-apis-qwen.json"

RECORD = {}
for _i, (_q, _right) in enumerate(QUESTIONS):
    RECORD[f"fast:{_i}"] = {"messages": [{"role": "user", "content": _q}], "max_tokens": FAST_TOKENS, "seeds": FAST_SEEDS}
    RECORD[f"think:{_i}"] = {"messages": [{"role": "user", "content": _q}], "max_tokens": THINK_TOKENS, "seeds": THINK_SEEDS, "think": True}


_NUMBER = r"\d[\d,]*(?:\.\d+)?"


def final_number(text: str):
    """The number an answer gives as its answer: the last number in its last \\boxed{...} or, without one, in its last
    bold **...** with a number; else its last number. None when it has no number."""
    for pattern in (r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\*\*([^*]+)\*\*"):
        spans = [s for s in re.findall(pattern, text) if re.search(_NUMBER, s)]
        if spans:
            return float(re.findall(_NUMBER, spans[-1])[-1].replace(",", ""))
    numbers = re.findall(_NUMBER, text)
    return float(numbers[-1].replace(",", "")) if numbers else None


def is_right(text: str, right) -> bool:
    """Whether an answer has what a right answer has: the number as its answer (final_number), or one of the words."""
    if isinstance(right, int):
        return final_number(text) == right
    return any(re.search(rf"\b{w}\b", text.lower()) for w in right)


def is_sum(i: int) -> bool:
    return isinstance(QUESTIONS[i][1], int)


# --- the course's measurement -----------------------------------------------------------------------------------------

def course() -> dict:
    return kept(("build-choose-model", "course"), lambda: json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-choose-model"])


def runs(way: str, i: int) -> list:
    """The runs of question i in one way ("fast" or "think"), each as {"right": ..., "tokens": ..., "seconds": ...}."""
    right = QUESTIONS[i][1]
    return [{"right": is_right(a["text"], right), "tokens": a["tokens"], "seconds": a["seconds"]}
            for a in course()[f"{way}:{i}"]["answers"]]


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs)


def table() -> list:
    """For every question: how often each way was right, and its mean tokens and seconds."""
    out = []
    for i, (q, _) in enumerate(QUESTIONS):
        row = {"question": q, "sum": is_sum(i)}
        for way in ("fast", "think"):
            rs = runs(way, i)
            row[way] = {"right": sum(r["right"] for r in rs), "runs": len(rs), "share": _mean(r["right"] for r in rs),
                        "tokens": _mean(r["tokens"] for r in rs), "seconds": _mean(r["seconds"] for r in rs)}
        out.append(row)
    return out


def cost_per_right(rs: list, price_out: float = PRICE_OUT):
    """The course's own: what the output tokens of all runs cost, per right answer; None when none is right."""
    right = sum(r["right"] for r in rs)
    return None if right == 0 else sum(r["tokens"] for r in rs) * price_out / 1_000_000 / right


def by_router(choose) -> dict:
    """What a router does on the course's eighteen questions, with the measured runs of the way it picks for each: the
    right answers to expect (the share right of that way, added up), and the mean tokens and seconds."""
    rows = table()
    ways = [choose(i) for i in range(len(rows))]
    return {"ways": ways,
            "right": sum(r[w]["share"] for r, w in zip(rows, ways)),
            "tokens": sum(r[w]["tokens"] for r, w in zip(rows, ways)),
            "seconds": sum(r[w]["seconds"] for r, w in zip(rows, ways))}


def findings() -> dict:
    """The numbers the lesson's text names."""
    rows = table()
    fast = [r for i in range(len(rows)) for r in runs("fast", i)]
    think = [r for i in range(len(rows)) for r in runs("think", i)]
    sums = [i for i in range(len(rows)) if rows[i]["sum"]]
    facts = [i for i in range(len(rows)) if not rows[i]["sum"]]
    allfast, allthink = by_router(lambda i: "fast"), by_router(lambda i: "think")
    digits = by_router(lambda i: "think" if is_sum(i) else "fast")
    return {
        "questions": len(rows), "sums": len(sums), "facts": len(facts),
        "fast_runs": len(fast), "think_runs": len(think),
        "fast_right": sum(r["right"] for r in fast), "think_right": sum(r["right"] for r in think),
        "fast_tokens": _mean(r["tokens"] for r in fast), "think_tokens": _mean(r["tokens"] for r in think),
        "fast_seconds": _mean(r["seconds"] for r in fast), "think_seconds": _mean(r["seconds"] for r in think),
        "fast_cost": cost_per_right(fast), "think_cost": cost_per_right(think),
        "think_sum_tokens": _mean(rows[i]["think"]["tokens"] for i in sums),
        "think_fact_tokens": _mean(rows[i]["think"]["tokens"] for i in facts),
        "fast_sum_tokens": _mean(rows[i]["fast"]["tokens"] for i in sums),
        "fast_fact_tokens": _mean(rows[i]["fast"]["tokens"] for i in facts),
        "wrong_fast": [i for i in range(len(rows)) if rows[i]["fast"]["right"] < rows[i]["fast"]["runs"]],
        "wrong_think": [i for i in range(len(rows)) if rows[i]["think"]["right"] < rows[i]["think"]["runs"]],
        "better_think": [i for i in range(len(rows)) if rows[i]["think"]["share"] > rows[i]["fast"]["share"]],
        "worse_think": [i for i in range(len(rows)) if rows[i]["think"]["share"] < rows[i]["fast"]["share"]],
        "think_no_answer": sum(a["finish"] == "length" and not a["text"].strip()
                               for i in range(len(rows)) for a in course()[f"think:{i}"]["answers"]),
        "allfast": allfast, "allthink": allthink, "router": digits,
    }


def _n(x, lang: str, digits: int = 1) -> str:
    s = f"{x:,.{digits}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else s


_TIMES = {"en": {1: "once", 2: "twice", 3: "three times", 4: "four times", 5: "five times"},
          "nl": {1: "één keer", 2: "twee keer", 3: "drie keer", 4: "vier keer", 5: "vijf keer"}}
_WORDS = {"en": "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
                "seventeen eighteen nineteen twenty".split(),
          "nl": "nul een twee drie vier vijf zes zeven acht negen tien elf twaalf dertien veertien vijftien zestien "
                "zeventien achttien negentien twintig".split()}


def _count(n: int, lang: str) -> str:
    """A count in words up to twenty, as the lesson writes it; above that in digits."""
    return _WORDS[lang][n] if n <= 20 else _n(n, lang, 0)


def measure_text(lang: str) -> str:
    """How the course measured, with the numbers of RECORD."""
    n_sums = sum(is_sum(i) for i in range(len(QUESTIONS)))
    fast, think = _TIMES[lang][FAST_SEEDS], _TIMES[lang][THINK_SEEDS]
    if lang == "nl":
        return (f"### De meting van de course\n\nDe course stelde Qwen3-0.6B via zijn eigen server {_count(len(QUESTIONS), lang)} vragen waarvan het "
                f"antwoord vaststaat: {_count(n_sums, lang)} sommen in woorden, en {_count(len(QUESTIONS) - n_sums, lang)} korte vragen over feiten of taal. Elke vraag ging {fast} op de "
                f"normale manier, met hooguit {_n(FAST_TOKENS, lang, 0)} tokens, en {think} in thinking mode, met hooguit {_n(THINK_TOKENS, lang, 0)}, "
                f"omdat nadenken veel meer tokens kost. Code beoordeelde elk antwoord: goed als het getal dat het als antwoord geeft het goede is, "
                f"het laatste getal in een kader of in vet, of anders het laatste getal; of als het goede woord erin staat. Code beoordeelt snel en "
                f"elke keer hetzelfde, maar leest niet of de rest van een antwoord klopt.")
    return (f"### The course's measurement\n\nThe course asked Qwen3-0.6B {_count(len(QUESTIONS), lang)} questions with a known answer, through its "
            f"own server: {_count(n_sums, lang)} sums in words, and {_count(len(QUESTIONS) - n_sums, lang)} short questions of fact or language. Each question went {fast} the normal way, "
            f"with at most {_n(FAST_TOKENS, lang, 0)} tokens, and {think} in thinking mode, with at most {_n(THINK_TOKENS, lang, 0)}, because "
            f"thinking takes many more. Code marked every answer: right when the number it gives as its answer is the right one, the last number "
            f"in a box or in bold, or else its last number; or when it has the right word. Code marks fast and the same every time, but it does "
            f"not read whether the rest of an answer makes sense.")


# --- the picture of the measurement --------------------------------------------------------------------------------

_TABLE_WORDS = {
    "en": {"title": "Qwen3-0.6B on eighteen questions, in two ways", "question": "Question", "fast": "normal",
           "think": "thinking", "right": "right", "tokens": "mean tokens", "sums": "Sums in words", "facts": "Facts and language",
           "all": "All eighteen"},
    "nl": {"title": "Qwen3-0.6B op achttien vragen, op twee manieren", "question": "Vraag", "fast": "normaal",
           "think": "thinking", "right": "goed", "tokens": "tokens, gemiddeld", "sums": "Sommen in woorden", "facts": "Feiten en taal",
           "all": "Alle achttien"},
}


def _dots(right: int, n: int) -> str:
    return "".join(f'<span style="color:{"var(--ea-pred-text)" if k < right else "var(--ea-loss-text)"}">{"●" if k < right else "○"}</span>'
                   for k in range(n))


def _short(text: str, n: int = 58) -> str:
    return text if len(text) <= n else text[:n - 1] + "…"


def table_picture(lang: str) -> str:
    """Every question with its runs in both ways: a filled dot for a right answer, an open one for a wrong one, and
    the mean tokens of each way."""
    w, rows = _TABLE_WORDS[lang], table()
    td = 'style="padding:0.12rem 0.5rem;border-top:1px solid var(--ea-line)"'
    num = 'style="padding:0.12rem 0.5rem;border-top:1px solid var(--ea-line);text-align:right;font-family:var(--ea-font-code)"'
    head = (f'<tr style="color:var(--ea-ink-2)"><th style="text-align:left;padding:0.12rem 0.5rem">{w["question"]}</th>'
            f'<th style="padding:0.12rem 0.5rem">{w["fast"]}</th><th style="padding:0.12rem 0.5rem;text-align:right">{w["tokens"]}</th>'
            f'<th style="padding:0.12rem 0.5rem">{w["think"]}</th><th style="padding:0.12rem 0.5rem;text-align:right">{w["tokens"]}</th></tr>')
    body = []
    for group, want in ((w["sums"], True), (w["facts"], False)):
        body.append(f'<tr><td colspan="5" style="padding:0.5rem 0.5rem 0.1rem;font-weight:600;color:var(--ea-ink)">{group}</td></tr>')
        for r in rows:
            if r["sum"] != want:
                continue
            f, t = r["fast"], r["think"]
            body.append(f'<tr><td {td}>{html.escape(_short(r["question"]))}</td>'
                        f'<td {td}>{_dots(f["right"], f["runs"])}</td><td {num}>{_n(f["tokens"], lang, 0)}</td>'
                        f'<td {td}>{_dots(t["right"], t["runs"])}</td><td {num}>{_n(t["tokens"], lang, 0)}</td></tr>')
    F = findings()
    body.append(f'<tr style="font-weight:600"><td {td}>{w["all"]}</td>'
                f'<td {td}>{F["fast_right"]}/{F["fast_runs"]} {w["right"]}</td><td {num}>{_n(F["fast_tokens"], lang, 0)}</td>'
                f'<td {td}>{F["think_right"]}/{F["think_runs"]} {w["right"]}</td><td {num}>{_n(F["think_tokens"], lang, 0)}</td></tr>')
    table_html = (f'<div style="overflow-x:auto"><table style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink);'
                  f'border-collapse:collapse;min-width:34rem">{head}{"".join(body)}</table></div>')
    return yours.figure(w["title"], table_html, alt=w["title"])


# --- the texts with the measured numbers ------------------------------------------------------------------------------

WORKED_Q = 10   # "What is the capital of Japan?", for the worked example


def _list(items: list, lang: str) -> str:
    """'a', 'a and b', 'a, b and c'; Dutch 'a en b'."""
    word = {"en": "and", "nl": "en"}[lang]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" {word} " + items[-1]


def _times(x: float, lang: str) -> str:
    return _n(x, lang, 0) if x >= 10 else _n(x, lang)


def _misses(F: dict, lang: str) -> str:
    """Which questions a way got wrong, and how often, from the measurement."""
    rows = table()
    wrong = sorted(set(F["wrong_fast"]) | set(F["wrong_think"]))
    if not wrong:
        return {"en": "Both ways got every question right in every run: on these questions, thinking bought nothing.",
                "nl": "Beide manieren hadden elke vraag in elke run goed: op deze vragen leverde thinking niets op."}[lang]
    items = []
    for i in wrong:
        f, t = rows[i]["fast"], rows[i]["think"]
        q = html.escape(_short(rows[i]["question"], 70))
        items.append({"en": f"*{q}* (normal {f['right']} of {f['runs']}, thinking {t['right']} of {t['runs']})",
                      "nl": f"*{q}* (normaal {f['right']} van de {f['runs']}, thinking {t['right']} van de {t['runs']})"}[lang])
    return {"en": f"A way went wrong only on {_list(items, lang)}.",
            "nl": f"Alleen bij {_list(items, lang)} ging een manier de mist in."}[lang]


def numbers(lang: str) -> dict:
    """The findings as the lesson's text writes them, in the lesson's language."""
    F = findings()
    out = {k: str(v) for k, v in F.items() if isinstance(v, int)}
    for k in ("fast_tokens", "think_tokens", "think_sum_tokens", "think_fact_tokens", "fast_sum_tokens", "fast_fact_tokens"):
        out[k] = _n(F[k], lang, 0)
    for k in ("fast_seconds", "think_seconds"):
        out[k] = _n(F[k], lang)
    for k in ("fast_cost", "think_cost"):
        out[k] = _n(F[k], lang, 6)
    out["token_ratio"] = _times(F["think_tokens"] / F["fast_tokens"], lang)
    out["seconds_ratio"] = _times(F["think_seconds"] / F["fast_seconds"], lang)
    out["cost_ratio"] = _times(F["think_cost"] / F["fast_cost"], lang)
    n = F["questions"]
    for key in ("allfast", "allthink", "router"):
        out[key + "_right"] = _n(F[key]["right"], lang)
        out[key + "_tokens"] = _n(F[key]["tokens"] / n, lang, 0)
    return out


def _share(F: dict, lang: str) -> str:
    """How the share right of thinking compares with that of the normal way."""
    a, b = F["think_right"] * F["fast_runs"], F["fast_right"] * F["think_runs"]
    key = "same" if a == b else "more" if a > b else "less"
    return {"en": {"same": "the same share", "more": "a larger share", "less": "a smaller share"},
            "nl": {"same": "naar verhouding even vaak", "more": "naar verhouding vaker", "less": "naar verhouding minder vaak"}}[lang][key]


def _no_answer(F: dict, lang: str) -> str:
    """The thinking runs that spent all their tokens on the thought, and so gave no answer: how many of its wrong runs
    they are."""
    k, wrong = F["think_no_answer"], F["think_runs"] - F["think_right"]
    if k == 0:
        return ""
    if lang == "nl":
        some = f"De {k} foute runs van thinking gebruikten" if k == wrong else f"Van de {wrong} foute runs van thinking gebruikten er {k}"
        return (f"{some} alle {_n(THINK_TOKENS, lang, 0)} tokens voor de gedachte, zonder ooit een antwoord te geven. Met een hogere "
                f"`max_tokens` was er misschien een antwoord gekomen, voor nog meer geld. ")
    some = f"All {k} wrong runs of thinking" if k == wrong else f"Of thinking's {wrong} wrong runs, {k}"
    return (f"{some} spent all {_n(THINK_TOKENS, lang, 0)} tokens on the thought, and gave no answer at all. With a larger `max_tokens`, an "
            f"answer might have come, at an even higher cost. ")


def measured_after(lang: str) -> str:
    """The text under the picture of the measurement."""
    F, n = findings(), numbers(lang)
    if lang == "nl":
        return (f"De normale manier had {n['fast_right']} van de {n['fast_runs']} antwoorden goed, met gemiddeld {n['fast_tokens']} output tokens "
                f"per antwoord; thinking {n['think_right']} van de {n['think_runs']}, {_share(F, lang)}, met {n['think_tokens']}: {n['token_ratio']} keer zoveel tokens, "
                f"en het duurde {n['seconds_ratio']} keer zo lang. {_misses(F, lang)} {_no_answer(F, lang)}Tegen een verzonnen prijs van {PRICE_OUT} dollar per miljoen output tokens "
                f"kostte een goed antwoord op de normale manier {n['fast_cost']} dollar, en met thinking {n['think_cost']} dollar: {n['cost_ratio']} keer zoveel.")
    return (f"The normal way got {n['fast_right']} of {n['fast_runs']} answers right, with {n['fast_tokens']} output tokens per answer on average; "
            f"thinking got {n['think_right']} of {n['think_runs']}, {_share(F, lang)}, with {n['think_tokens']}: {n['token_ratio']} times as many tokens, and it took "
            f"{n['seconds_ratio']} times as long. {_misses(F, lang)} {_no_answer(F, lang)}At a made-up price of {PRICE_OUT} dollars per million output tokens, a right answer cost "
            f"{n['fast_cost']} dollars the normal way, and {n['think_cost']} dollars thinking: {n['cost_ratio']} times as much.")


def _questions(items: list, lang: str) -> str:
    return _list([f"*{html.escape(_short(QUESTIONS[i][0], 44))}*" for i in items], lang)


def _verdict(lang: str) -> str:
    """What the measurement says about the rule: the questions where thinking did better and worse than the normal way."""
    F = findings()
    best = max(F["allfast"]["right"], F["allthink"]["right"])
    if F["router"]["right"] >= best:
        return {"en": " So on these questions, the rule kept the right answers of the best way, for fewer tokens than always thinking.",
                "nl": " Op deze vragen hield de regel dus de goede antwoorden van de beste manier, voor minder tokens dan altijd thinking."}[lang]
    better = _questions(F["better_think"], lang) if F["better_think"] else {"en": "no question", "nl": "geen enkele vraag"}[lang]
    worse = _questions(F["worse_think"], lang) if F["worse_think"] else {"en": "no question", "nl": "geen enkele vraag"}[lang]
    if lang == "nl":
        return (f"\n\nOp deze vragen deed de regel het dus slechter dan elke vraag dezelfde manier op te sturen. Thinking deed het beter dan de normale manier bij "
                f"{better}, en slechter bij {worse}. Waar thinking helpt, hangt af van het model en de vraag, niet van een cijfer: een router is zo "
                f"goed als de meting erachter.")
    return (f"\n\nSo on these questions, the rule did worse than sending every question the same way. Thinking did better than the normal way on {better}, "
            f"and worse on {worse}. Where thinking helps depends on the model and the question, not on a digit: a router is only as good as the "
            f"measurement behind it.")


def routing_text(lang: str) -> str:
    """The text on routing, with what the rule of exercise 2 would have done with the course's questions."""
    n = numbers(lang)
    if lang == "nl":
        return (f"### Routing\n\nThinking kostte het meest bij de sommen: gemiddeld {n['think_sum_tokens']} tokens per antwoord, tegen "
                f"{n['think_fact_tokens']} bij de feiten. Een router beslist per vraag welke manier hij neemt. De eenvoudigste is een regel: een vraag "
                f"met een cijfer erin gaat naar thinking, elke andere de normale manier. Op de vragen van de course had je met die regel {n['router_right']} "
                f"goede antwoorden van de {n['questions']} te verwachten gehad, met gemiddeld {n['router_tokens']} tokens per vraag; met altijd thinking "
                f"{n['allthink_right']} ({n['allthink_tokens']} tokens), en met altijd de normale manier {n['allfast_right']} ({n['allfast_tokens']} tokens). Een vraag die een "
                f"manier maar in een deel van de runs goed had, telt hier mee voor dat deel." + _verdict(lang))
    return (f"### Routing\n\nThinking cost the most on the sums: {n['think_sum_tokens']} tokens per answer on average, against "
            f"{n['think_fact_tokens']} on the facts. A router decides for each question which way it takes. The simplest router is a rule: a "
            f"question with a digit in it goes to thinking, every other one the normal way. On the course's questions, that rule would have "
            f"given {n['router_right']} right answers of {n['questions']} to expect, with {n['router_tokens']} tokens per question on average; always thinking "
            f"{n['allthink_right']} ({n['allthink_tokens']} tokens), and always the normal way {n['allfast_right']} ({n['allfast_tokens']} tokens). A "
            f"question that a way got right in only some of its runs counts here for that share." + _verdict(lang))


def worked() -> dict:
    """Seed 0 of the course's thinking run on the question of the worked example."""
    a = course()[f"think:{WORKED_Q}"]["answers"][0]
    fast = table()[WORKED_Q]["fast"]
    return {"text": a["text"].strip(), "chars": len(a["reasoning"]), "tokens": a["tokens"],
            "fast_tokens": fast["tokens"], "fast_right": fast["right"], "fast_runs": fast["runs"]}


_WORKED_CODE = '''import json
import urllib.request

url = "http://127.0.0.1:8765/v1"
body = {"model": "qwen3-0.6b",
        "messages": [{"role": "user", "content": "What is the capital of Japan?"}],
        "max_tokens": 2000,
        "chat_template_kwargs": {"enable_thinking": True}}
request = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
with urllib.request.urlopen(request, timeout=600) as response:
    answer = json.loads(response.read())
message = answer["choices"][0]["message"]
'''


def worked_text(lang: str) -> str:
    """The worked example: one question in thinking mode, with what the course's run gave."""
    w = worked()
    text = w["text"].replace("`", "'")
    tokens, chars, fast = w["tokens"], _n(w["chars"], lang, 0), _n(w["fast_tokens"], lang, 0)
    ratio = _times(w["tokens"] / w["fast_tokens"], lang)
    if lang == "nl":
        return (f"### Uitgewerkt voorbeeld\n\nDezelfde vraag als in de meting, in thinking mode, met het request van "
                f"[Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl) en één instelling erbij:\n\n```python\n{_WORKED_CODE}```\n\n"
                f"1. **De schakelaar:** `\"chat_template_kwargs\"` geeft instellingen door aan het chat template van het model, en "
                f"`\"enable_thinking\": True` zet de thinking van Qwen3 aan. De server van de course antwoordt op de normale manier, tot je thinking "
                f"aanzet; bij vLLM en in het chat template van Qwen3 zelf staat thinking juist aan, tenzij je `\"enable_thinking\": False` meestuurt. "
                f"Andere API's hebben er een eigen instelling voor, zoals een *reasoning effort*.\n"
                f"2. **Het antwoord:** `message[\"content\"]` is alleen het antwoord; in de run van de course: `{text}`\n"
                f"3. **De gedachte:** `message[\"reasoning_content\"]` bevat de redenering, {chars} tekens in die run; op de normale manier is dat veld er niet.\n"
                f"4. **De rekening:** `answer[\"usage\"][\"completion_tokens\"]` was {tokens}: de gedachte en het antwoord samen. Op de normale "
                f"manier kostte dezelfde vraag gemiddeld {fast} tokens, {ratio} keer zo weinig, en was het antwoord in {w['fast_right']} van de "
                f"{w['fast_runs']} runs goed.")
    return (f"### Worked example\n\nThe same question as in the measurement, in thinking mode, with the request of "
            f"[Calling a model over HTTP](/course/open/build-call-api?lang=en) and one setting more:\n\n```python\n{_WORKED_CODE}```\n\n"
            f"1. **The switch:** `\"chat_template_kwargs\"` passes settings on to the model's chat template, and `\"enable_thinking\": True` "
            f"switches Qwen3's thinking on. The course's server answers the normal way until you switch thinking on; on vLLM, and in Qwen3's "
            f"own chat template, thinking is on unless you send `\"enable_thinking\": False`. Other APIs have a setting of their own for it, such as a "
            f"*reasoning effort*.\n"
            f"2. **The answer:** `message[\"content\"]` is only the answer; in the course's run: `{text}`\n"
            f"3. **The thought:** `message[\"reasoning_content\"]` holds the reasoning, {chars} characters in that run; the normal way has no such field.\n"
            f"4. **The bill:** `answer[\"usage\"][\"completion_tokens\"]` was {tokens}: the thought and the answer together. Asked the normal "
            f"way, the same question took {fast} tokens on average, {ratio} times fewer, and its answer was right in {w['fast_right']} of its "
            f"{w['fast_runs']} runs.")


# --- the picture of routing ----------------------------------------------------------------------------------------

_ROUTE_WORDS = {
    "en": {"title": "A router in front of two ways", "question": "a question", "router": "route(question)",
           "fast": "the normal way", "fast_note": "few tokens, quick", "think": "the thinking way", "think_note": "many tokens, slow",
           "digit": "with a digit", "other": "any other"},
    "nl": {"title": "Een router voor twee manieren", "question": "een vraag", "router": "route(question)",
           "fast": "de normale manier", "fast_note": "weinig tokens, snel", "think": "de thinking-manier", "think_note": "veel tokens, langzaam",
           "digit": "met een cijfer", "other": "elke andere"},
}


def router_picture(lang: str) -> str:
    """A question goes into the router, which sends it one of two ways; ink only, because nothing here is a role of a
    model."""
    w = _ROUTE_WORDS[lang]
    box = 'style="fill:var(--ea-ink);fill-opacity:0.04;stroke:var(--ea-ink-3);stroke-width:1.2"'
    big = 'style="font-size:14px;font-weight:650;fill:var(--ea-ink);text-anchor:middle"'
    small = 'style="font-size:12px;fill:var(--ea-ink-2);text-anchor:middle"'
    line = 'style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round"'
    parts = [
        f'<rect x="0" y="62" width="150" height="56" rx="8" {box}/><text x="75" y="95" {big}>{html.escape(w["question"])}</text>',
        f'<rect x="230" y="62" width="170" height="56" rx="8" {box}/><text x="315" y="95" {big} font-family="var(--ea-font-code)">{html.escape(w["router"])}</text>',
        f'<rect x="510" y="4" width="210" height="64" rx="8" {box}/><text x="615" y="31" {big}>{html.escape(w["think"])}</text>'
        f'<text x="615" y="52" {small}>{html.escape(w["think_note"])}</text>',
        f'<rect x="510" y="112" width="210" height="64" rx="8" {box}/><text x="615" y="139" {big}>{html.escape(w["fast"])}</text>'
        f'<text x="615" y="160" {small}>{html.escape(w["fast_note"])}</text>',
        f'<path d="M156 90 H224 M218 85 L224 90 L218 95" {line}/>',
        f'<path d="M406 82 C450 82 460 36 504 36 M498 31 L504 36 L498 41" {line}/>',
        f'<path d="M406 98 C450 98 460 144 504 144 M498 139 L504 144 L498 149" {line}/>',
        f'<text x="446" y="24" {small}>{html.escape(w["digit"])}</text>',
        f'<text x="446" y="168" {small}>{html.escape(w["other"])}</text>',  # above and below the curves, clear of them
    ]
    svg = f'<svg viewBox="0 0 720 180" role="img" aria-label="{html.escape(w["title"])}" style="width:100%;max-width:720px;font-family:var(--ea-font-ui)">{"".join(parts)}</svg>'
    return yours.figure(w["title"], svg, alt=w["title"])


# --- exercise 1: what a right answer costs ---------------------------------------------------------------------------

_START_COST = '''def cost_per_right(runs, price_out):
    return 0
'''
SOLUTION_COST = '''def cost_per_right(runs, price_out):
    tokens = 0
    right = 0
    for run in runs:
        tokens = tokens + run["tokens"]
        if run["right"]:
            right = right + 1
    if right == 0:
        return None
    return tokens * price_out / 1_000_000 / right
'''


def _r(right: bool, tokens: int) -> dict:
    return {"right": right, "tokens": tokens}


EXAMPLE_RUNS = [_r(True, 100), _r(False, 60), _r(True, 40)]
EXAMPLE_PRICE = 15
COST_CASES = [
    (EXAMPLE_RUNS, EXAMPLE_PRICE),
    ([_r(True, 1906), _r(True, 1435)], 15),
    ([_r(False, 300), _r(False, 300)], 15),
    ([_r(True, 10)], 15),
    ([_r(True, 500), _r(False, 700), _r(False, 900)], 0.6),
    ([], 15),
]


def example(lang: str) -> dict:
    """The example of exercise 1 as its text names it: the runs' tokens, the one that was wrong, and what it costs."""
    return {"n": len(EXAMPLE_RUNS), "tokens": _list([str(r["tokens"]) for r in EXAMPLE_RUNS], lang),
            "wrong": [r["tokens"] for r in EXAMPLE_RUNS if not r["right"]][0], "price": EXAMPLE_PRICE,
            "want": repr(cost_per_right(EXAMPLE_RUNS, EXAMPLE_PRICE))}


def cost_picture(seen) -> str:
    """What the learner's cost_per_right says for the course's runs of both ways, over all eighteen questions."""
    run = seen.mine
    fn = run.names.get("cost_per_right") if run is not None else None
    if not callable(fn):
        return ""
    w = {"en": ("Your cost_per_right on the course's runs", "normal", "thinking", "runs", "should be", "error", "dollars per right answer"),
         "nl": ("Je cost_per_right op de runs van de course", "normaal", "thinking", "runs", "zou zijn", "fout", "dollar per goed antwoord")}[seen.lang]
    rows = []
    for label, way in ((w[1], "fast"), (w[2], "think")):
        rs = [r for i in range(len(QUESTIONS)) for r in runs(way, i)]
        right = cost_per_right(rs)
        try:
            got = fn([dict(r) for r in rs], PRICE_OUT)
            ok = got is not None and abs(float(got) - right) <= 1e-9 + 1e-6 * right
            mine = f"{float(got):.6f}" if got is not None else "None"
        except Exception:   # the learner's code
            mine, ok = w[5], False
        colour = "var(--ea-ink)" if ok else "var(--ea-loss-text)"
        rows.append(f'<tr><td style="padding:0.1rem 0.6rem 0.1rem 0">{html.escape(label)}, {len(rs)} {w[3]}</td>'
                    f'<td style="font-family:var(--ea-font-code);color:{colour};padding-left:0.6rem">{html.escape(mine)}</td>'
                    f'<td style="font-family:var(--ea-font-code);color:var(--ea-ink-2);padding-left:0.6rem">{html.escape(w[4])} {right:.6f}</td></tr>')
    body = (f'<table style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink);border-collapse:collapse">{"".join(rows)}</table>'
            f'<div style="font-family:var(--ea-font-ui);font-size:11px;color:var(--ea-ink-2);margin-top:0.3rem">{html.escape(w[6])}, {PRICE_OUT} $ / 1M</div>')
    return yours.figure(w[0], body, alt=w[0])


# --- exercise 2: a router -------------------------------------------------------------------------------------------

_START_ROUTE = '''def route(question):
    return "fast"
'''
SOLUTION_ROUTE = '''def route(question):
    for ch in question:
        if ch in "0123456789":
            return "think"
    return "fast"
'''
_EARLY_ROUTE = SOLUTION_ROUTE.replace('            return "think"\n', '            return "think"\n        else:\n            return "fast"\n')
_SWAPPED_ROUTE = SOLUTION_ROUTE.replace('return "think"', 'return "X"').replace('return "fast"', 'return "think"').replace('return "X"', 'return "fast"')
ROUTE_CASES = [
    ("What is 17 times 24?",),
    ("What is the capital of Japan?",),
    ("How many legs do 3 spiders have?",),
    ("Translate 'thank you' into Spanish.",),
    ("What is 2 + 2?",),
    ("Who painted the Mona Lisa?",),
    ("A train leaves at 9 and arrives at 11. How long is the trip?",),
    ("Name a color of the rainbow.",),
    ("12 eggs cost 3 euros. What does one egg cost?",),
]


def route_picture(seen) -> str:
    """Where the learner's route sends the course's eighteen questions, and what that would have cost, with the
    measured runs of each way: beside always the normal way and always the thinking way."""
    run = seen.mine
    fn = run.names.get("route") if run is not None else None
    if not callable(fn):
        return ""
    w = {"en": ("Your router on the course's eighteen questions", "sums to thinking", "facts to thinking", "Your router",
                "Always normal", "Always thinking", "right, to expect", "tokens per question", "error", "of"),
         "nl": ("Je router op de achttien vragen van de course", "sommen naar thinking", "feiten naar thinking", "Je router",
                "Altijd normaal", "Altijd thinking", "goed, te verwachten", "tokens per vraag", "fout", "van")}[seen.lang]
    ways = []
    for q, _ in QUESTIONS:
        try:
            got = fn(q)
        except Exception:   # the learner's code
            got = None
        ways.append(got if got in ("fast", "think") else None)
    if None in ways:
        return ""
    mine = by_router(lambda i: ways[i])
    F = findings()
    n = len(QUESTIONS)
    sums_think = sum(1 for i in range(n) if is_sum(i) and ways[i] == "think")
    facts_think = sum(1 for i in range(n) if not is_sum(i) and ways[i] == "think")
    rows = []
    for label, res in ((w[3], mine), (w[4], F["allfast"]), (w[5], F["allthink"])):
        rows.append(f'<tr><td style="padding:0.1rem 0.8rem 0.1rem 0;font-weight:{600 if res is mine else 400}">{html.escape(label)}</td>'
                    f'<td style="font-family:var(--ea-font-code);padding:0.1rem 0.8rem 0.1rem 0">{_n(res["right"], seen.lang)} {w[9]} {n}</td>'
                    f'<td style="font-family:var(--ea-font-code);text-align:right">{_n(res["tokens"] / n, seen.lang, 0)}</td></tr>')
    head = (f'<tr style="color:var(--ea-ink-2)"><th></th><th style="text-align:left;padding-right:0.8rem">{html.escape(w[6])}</th>'
            f'<th style="text-align:right">{html.escape(w[7])}</th></tr>')
    sent = (f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2);margin-bottom:0.3rem">'
            f'{sums_think}/{F["sums"]} {html.escape(w[1])} · {facts_think}/{F["facts"]} {html.escape(w[2])}</div>')
    body = sent + f'<table style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink);border-collapse:collapse">{head}{"".join(rows)}</table>'
    return yours.figure(w[0], body, alt=w[0])


_COMMON = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
                "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."},
}

EXERCISES = {
    "cost_per_right": FunctionBox(
        name="cost_per_right",
        picture=cost_picture,
        function="cost_per_right",
        start=_START_COST,
        solution=SOLUTION_COST,
        visible=COST_CASES[:1],
        hidden=COST_CASES[1:],
        mistakes=[
            Mistake("per-answer", wrong=SOLUTION_COST.replace("/ right\n", "/ len(runs)\n"),
                    en="`{call}` gives `{got}`: that is what one answer costs, right or wrong. Divide by the number of right answers, not by the number of runs.",
                    nl="`{call}` geeft `{got}`: dat kost één antwoord, goed of fout. Deel door het aantal goede antwoorden, niet door het aantal runs."),
            Mistake("per-million", wrong=SOLUTION_COST.replace(" / 1_000_000", ""),
                    en="`{call}` gives `{got}`, a million times too much: the price is per million tokens. Divide by `1_000_000`.",
                    nl="`{call}` geeft `{got}`, een miljoen keer te veel: de prijs is per miljoen tokens. Deel door `1_000_000`."),
            Mistake("right-tokens", wrong=SOLUTION_COST.replace('        tokens = tokens + run["tokens"]\n        if run["right"]:\n            right = right + 1\n',
                                                                 '        if run["right"]:\n            tokens = tokens + run["tokens"]\n            right = right + 1\n'),
                    en="`{call}` gives `{got}`: you only count the tokens of the right answers. The wrong ones cost tokens too, and you pay for them.",
                    nl="`{call}` geeft `{got}`: je telt alleen de tokens van de goede antwoorden. De foute kosten ook tokens, en die betaal je ook."),
            Mistake("none-right", error=lambda e: isinstance(e, ZeroDivisionError),
                    en="`{call}` stops with a `ZeroDivisionError`: none of these runs is right, and you divide by 0. Give back `None` when no run is right.",
                    nl="`{call}` stopt met een `ZeroDivisionError`: geen van deze runs is goed, en je deelt door 0. Geef `None` terug als geen run goed is."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "`cost_per_right` gives 0 yet. Add up the tokens of all runs and count the right ones; what do those tokens cost, per right answer?",
                      "nl": "`cost_per_right` geeft nog 0. Tel de tokens van alle runs op en tel de goede; wat kosten die tokens, per goed antwoord?"},
            "correct": {"en": "Right: your `cost_per_right` gives what a right answer costs, for all {n} lists of runs the check tried, also the one without a right answer.",
                        "nl": "Goed: je `cost_per_right` geeft wat een goed antwoord kost, voor alle {n} lijsten met runs die de check probeerde, ook die zonder goed antwoord."},
            "wrong": {"en": f"`{{call}}` gives `{{got}}`, and it should give `{{want}}`: {sum(r['tokens'] for r in EXAMPLE_RUNS)} tokens at "
                            f"{EXAMPLE_PRICE} dollars per million, for {sum(r['right'] for r in EXAMPLE_RUNS)} right answers.",
                      "nl": f"`{{call}}` geeft `{{got}}`, en het zou `{{want}}` moeten zijn: {sum(r['tokens'] for r in EXAMPLE_RUNS)} tokens tegen "
                            f"{EXAMPLE_PRICE} dollar per miljoen, voor {sum(r['right'] for r in EXAMPLE_RUNS)} goede antwoorden."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`.",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn."},
        },
    ),
    "route": FunctionBox(
        name="route",
        picture=route_picture,
        function="route",
        start=_START_ROUTE,
        solution=SOLUTION_ROUTE,
        visible=ROUTE_CASES[:1],
        hidden=ROUTE_CASES[1:],
        mistakes=[
            Mistake("always-think", wrong='def route(question):\n    return "think"\n',
                    en="`{call}` gives `{got}`: your `route` sends every question to thinking, so every question pays its price. Only a question with a digit goes there.",
                    nl="`{call}` geeft `{got}`: je `route` stuurt elke vraag naar thinking, dus betaalt elke vraag de prijs ervan. Alleen een vraag met een cijfer gaat daarheen."),
            Mistake("early-fast", wrong=_EARLY_ROUTE,
                    en="`{call}` gives `{got}`: your `route` decides at the first character. The `return \"fast\"` in the loop stops it there, before it reaches a digit. Give back `\"fast\"` only after the loop, when no digit came.",
                    nl="`{call}` geeft `{got}`: je `route` beslist al bij het eerste teken. De `return \"fast\"` in de loop stopt hem daar, voordat hij bij een cijfer komt. Geef `\"fast\"` pas na de loop terug, als er geen cijfer kwam."),
            Mistake("swapped", wrong=_SWAPPED_ROUTE,
                    en="`{call}` gives `{got}`: the two ways are the wrong way round. A question with a digit goes to `\"think\"`, any other to `\"fast\"`.",
                    nl="`{call}` geeft `{got}`: de twee manieren staan andersom. Een vraag met een cijfer gaat naar `\"think\"`, elke andere naar `\"fast\"`."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "`route` sends every question the normal way yet, also `\"What is 17 times 24?\"`. Look at the characters of the question: is there a digit among them?",
                      "nl": "`route` stuurt nog elke vraag de normale manier op, ook `\"What is 17 times 24?\"`. Kijk naar de tekens van de vraag: zit er een cijfer tussen?"},
            "correct": {"en": "Right: your `route` sends a question with a digit to `\"think\"` and every other one to `\"fast\"`, for all {n} questions the check tried. Under it, what that router would have done with the course's questions.",
                        "nl": "Goed: je `route` stuurt een vraag met een cijfer naar `\"think\"` en elke andere naar `\"fast\"`, voor alle {n} vragen die de check probeerde. Eronder wat die router met de vragen van de course gedaan zou hebben."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: the question has the digits 1, 7, 2 and 4.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: de vraag heeft de cijfers 1, 7, 2 en 4."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`.",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn."},
        },
    ),
}

REQUIRED = ["cost_per_right", "route", "support"]

CASES = [
    ("cost_per_right", _START_COST, "empty", None),
    ("cost_per_right", SOLUTION_COST, "correct", None),
    ("cost_per_right", SOLUTION_COST.replace("/ right\n", "/ len(runs)\n"), "wrong", "per-answer"),
    ("cost_per_right", SOLUTION_COST.replace(" / 1_000_000", ""), "wrong", "per-million"),
    ("cost_per_right", SOLUTION_COST.replace("    if right == 0:\n        return None\n", ""), "error", "none-right"),
    ("route", _START_ROUTE, "empty", None),
    ("route", SOLUTION_ROUTE, "correct", None),
    ("route", 'def route(question):\n    return "think"\n', "wrong", "always-think"),
    ("route", _SWAPPED_ROUTE, "wrong", "swapped"),
    ("route", _EARLY_ROUTE, "wrong", "early-fast"),
]


# --- the demo: your router in front of the course's own server --------------------------------------------------------

def default_question(lang: str) -> str:
    """What the course measured on the question that stands in the demo's box at first, the one of the worked example."""
    row = table()[WORKED_Q]
    f, t = row["fast"], row["think"]
    if lang == "nl":
        return (f"In het vak staat de vraag van het uitgewerkte voorbeeld. Er staat geen cijfer in, dus je `route` stuurt hem de normale manier op. "
                f"In de meting van de course was die manier hier {f['right']} van de {f['runs']} keer goed, en thinking {t['right']} van de "
                f"{t['runs']} keer. Probeer ook een som, en vergelijk de tokens en de seconden.")
    return (f"The box holds the question of the worked example. It has no digit, so your `route` sends it the normal way. In the course's "
            f"measurement, that way got it right {f['right']} of {f['runs']} times here, and thinking {t['right']} of {t['runs']} times. Try a sum "
            f"too, and compare the tokens and the seconds.")


def downloaded() -> bool:
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    chat.download_or_stop(button, texts)


def learner_route(text):
    """The learner's route from exercise 2, when it is right; else None."""
    if text is MISSING or text is None or EXERCISES["route"].check(text).state != "correct":
        return None
    return run_code(text).names.get("route")


def ask(url: str, question: str, way: str) -> dict:
    """One question to the course's server, the normal way or the thinking way, as the course measured: the answer,
    the thought, the output tokens and the seconds."""
    body = {"model": serve.MODEL, "messages": [{"role": "user", "content": question}],
            "max_tokens": THINK_TOKENS if way == "think" else FAST_TOKENS}
    if way == "think":
        body["chat_template_kwargs"] = {"enable_thinking": True}
    request = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1800) as response:
        answer = json.loads(response.read())
    message = answer["choices"][0]["message"]
    return {"text": message["content"] or "", "thought": message.get("reasoning_content") or "",
            "tokens": answer["usage"]["completion_tokens"], "seconds": time.perf_counter() - start,
            "finish": answer["choices"][0]["finish_reason"]}


def demo(route, question: str, words: dict, lang: str) -> str:
    """The learner's route picks the way; the course's server, with Qwen3 behind it, answers that way."""
    try:
        way = route(question)
    except Exception as e:   # the learner's code
        return f"<p><em>{html.escape(words['failed'].format(why=type(e).__name__ + ': ' + str(e)))}</em></p>"
    if way not in ("fast", "think"):
        return f"<p><em>{html.escape(words['other'].format(way=repr(way)))}</em></p>"
    running = serve.model_server()
    try:
        a = ask(running.url, question, way)
    except Exception as e:   # the server ran out of memory, or could not be reached
        said = words["memory"] if serve.out_of_memory(e) else words["server"].format(why=f"{type(e).__name__}: {e}")
        return f"<p><em>{html.escape(said)}</em></p>"
    if not a["text"].strip() and a["thought"]:   # the thought used every token, and no answer came: the end of the thought
        note = words["no_answer"].format(tokens=a["tokens"], seconds=_n(a["seconds"], lang))
        return chat.answer_picture("… " + a["thought"][-600:].lstrip(), note, words["title_" + way])
    note = words["done_" + way].format(tokens=a["tokens"], seconds=_n(a["seconds"], lang))
    return chat.answer_picture(a["text"], note, words["title_" + way])
