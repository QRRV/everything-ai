# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-self-consistency: a function that takes the answer that comes back most often, and one that asks
the same question several times with sampling and takes that answer, against a stand-in of the server; the course's
recorded runs (data/course-dev-prompts-qwen.json): twelve sums in words, each asked nine times with sampling, and how
often one answer, and the most common of three, five or nine answers, was right; their pictures, and the demo."""
import html
import itertools
import json
import re

from everything_ai import apicheck, chat, paths, serve, yours
from everything_ai.apicheck import URL, ApiBox, Case
from everything_ai.check import FunctionBox, Mistake, run_code, with_time_limit

DOWNLOADS = {"Qwen3-0.6B": chat.FILES}   # what the demo downloads, for the line under the header

QUESTIONS = [
    ("A pen costs 3 euros. Mia buys 4 pens and pays with a 20-euro note. How many euros of change does Mia get?", 8),
    ("A train leaves at 9:40 and arrives at 11:15. How many minutes does the trip take?", 95),
    ("There are 5 boxes with 12 apples each. 17 of the apples are rotten. How many apples are good?", 43),
    ("Tom is 3 times as old as Jo, who is 9. How old will Tom be in 4 years?", 31),
    ("A recipe needs 250 grams of flour for 10 cookies. How many grams of flour are needed for 24 cookies?", 600),
    ("A rectangle is 7 cm wide and 12 cm long. What is its perimeter in cm?", 38),
    ("Sam reads 18 pages a day of a book with 300 pages. How many pages are left after 12 days?", 84),
    ("A bus has 40 seats. 107 children ride in 3 buses. How many seats stay empty?", 13),
    ("Lisa has 65 euros and saves 15 euros a week. A bike costs 200 euros. After how many weeks can Lisa buy it?", 9),
    ("A farm has only chickens and cows: 10 animals with 28 legs in total. How many cows are there?", 4),
    ("A tank holds 60 liters and is one quarter full. How many liters must be added to fill it?", 45),
    ("Three friends pay a bill of 85 euros. One pays 10 euros more than each of the other two, who pay the same. "
     "How many euros does each of the other two pay?", 25),
]
STEPS = "{question}\nThink step by step. On the last line, write only the answer as a number, after Answer:"
SEEDS = 9
RECORD = {f"q{i}": {"messages": [{"role": "user", "content": STEPS.format(question=q)}], "max_tokens": 300, "seeds": SEEDS}
          for i, (q, _) in enumerate(QUESTIONS)}
DATA = paths.ROOT / "data" / "course-dev-prompts-qwen.json"
KS = (1, 3, 5, 9)   # how many answers a vote takes


def course() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-self-consistency"]


def answers(i: int) -> list:
    return course()[f"q{i}"]["answers"]


ANSWER_OF = '''def answer_of(text):
    found = re.findall(r"Answer:\\s*\\$?(-?\\d[\\d,]*(?:\\.\\d+)?)", text)
    numbers = found or re.findall(r"-?\\d[\\d,]*(?:\\.\\d+)?", text)
    if not numbers:
        return None
    x = float(numbers[-1].replace(",", ""))
    return int(x) if x == int(x) else x
'''
answer_of = run_code(ANSWER_OF, {"re": re}).names["answer_of"]
answer_of.__doc__ = """The number an answer gives as its answer: the last number after Answer:, else its last number; None
when it has no number. A whole number as an int."""

SOLUTION_MAJORITY = '''def majority(answers):
    counts = {}
    for answer in answers:
        if answer is not None:
            if answer in counts:
                counts[answer] = counts[answer] + 1
            else:
                counts[answer] = 1
    best = None
    for answer in counts:
        if best is None or counts[answer] > counts[best]:
            best = answer
    return best
'''
majority = run_code(SOLUTION_MAJORITY).names["majority"]


def numbers(i: int) -> list:
    """The answer of each run of question i, as answer_of reads it."""
    return [answer_of(a["text"]) for a in answers(i)]


def vote_right(i: int, k: int) -> float:
    """How often the most common answer of k runs of question i is right, over every way to pick k of its runs: the
    share of those picks whose majority is the right number."""
    got, right = numbers(i), QUESTIONS[i][1]
    picks = list(itertools.combinations(range(len(got)), k))
    return sum(majority([got[j] for j in pick]) == right for pick in picks) / len(picks)


def findings() -> dict:
    """What the recorded runs show: per question how many of its runs were right; per k how many questions the vote of
    k runs gets right on average over all picks (out of the twelve), and the vote of all nine; the sum of the worked
    example (where the vote of nine is right with the fewest right runs) with its counts; and the tokens."""
    right = [sum(n == QUESTIONS[i][1] for n in numbers(i)) for i in range(len(QUESTIONS))]
    per_k = {k: sum(vote_right(i, k) for i in range(len(QUESTIONS))) for k in KS}
    voted = [i for i in range(len(QUESTIONS)) if majority(numbers(i)) == QUESTIONS[i][1]]
    example = min(voted, key=lambda i: right[i]) if voted else 0   # the sum where the vote helped most
    stuck = [i for i in range(len(QUESTIONS)) if majority(numbers(i)) != QUESTIONS[i][1]]   # where even nine votes are wrong
    counts = {}
    for n in numbers(example):
        if n is not None:
            counts[n] = counts.get(n, 0) + 1
    return {"runs": SEEDS, "questions": len(QUESTIONS), "right": right, "all_right": sum(right),
            "example": example, "example_counts": sorted(counts.items(), key=lambda kv: -kv[1]),
            "stuck": stuck, "stuck_vote": [(majority(numbers(i)), numbers(i).count(majority(numbers(i)))) for i in stuck],
            "per_k": per_k, "nine": [majority(numbers(i)) == QUESTIONS[i][1] for i in range(len(QUESTIONS))],
            "tokens": [a["tokens"] for i in range(len(QUESTIONS)) for a in answers(i)],
            "none": sum(n is None for i in range(len(QUESTIONS)) for n in numbers(i))}


def number(x: float, lang: str, digits: int = 1) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "\x00").replace(".", ",").replace("\x00", ".") if lang == "nl" else text


# --- the pictures of the recorded runs -------------------------------------------------------------------------------

_LABEL = "font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)"


def _label(text: str) -> str:
    return f'<div style="{_LABEL};margin:0.2rem 0">{html.escape(text)}</div>'


_GRID_WORDS = {
    "en": {"title": "Twelve sums, nine runs each: the number of each answer", "q": "sum", "right": "right",
           "vote": "most common", "legend_right": "right", "legend_wrong": "wrong", "legend_none": "no number"},
    "nl": {"title": "Twaalf sommen, elk negen runs: het getal van elk antwoord", "q": "som", "right": "goed",
           "vote": "meest gegeven", "legend_right": "goed", "legend_wrong": "fout", "legend_none": "geen getal"},
}


def grid_picture(lang: str) -> str:
    """A row per question: the number each of its nine runs gave, right in the colour of model output and wrong as a
    mistake, then the right number and the most common of the nine, marked when it is wrong."""
    w = _GRID_WORDS[lang]
    rows = []
    for i, (_, right) in enumerate(QUESTIONS):
        cells = []
        for n in numbers(i):
            if n is None:
                style, text = "background:var(--ea-inset);color:var(--ea-ink-3)", "–"
            elif n == right:
                style, text = "background:var(--ea-pred-wash);color:var(--ea-pred-text)", str(n)
            else:
                style, text = "background:var(--ea-loss-wash);color:var(--ea-loss-text);font-weight:600", str(n)
            cells.append(f'<td style="{style};font-family:var(--ea-font-code);font-size:12px;text-align:center;padding:0.1rem 0.25rem;border:1px solid var(--ea-page)">{html.escape(text)}</td>')
        vote = majority(numbers(i))
        vote_style = "color:var(--ea-ink)" if vote == right else "color:var(--ea-loss-text);font-weight:600"
        rows.append(f'<tr><td style="{_LABEL};padding-right:0.4rem">{i + 1}</td>{"".join(cells)}'
                    f'<td style="font-family:var(--ea-font-code);font-size:12px;padding:0 0.5rem;color:var(--ea-ink-2)">{right}</td>'
                    f'<td style="font-family:var(--ea-font-code);font-size:12px;{vote_style}">{html.escape(str(vote))}</td></tr>')
    head = (f'<tr><th style="{_LABEL};font-weight:500;text-align:left">{w["q"]}</th>'
            + "".join(f'<th style="{_LABEL};font-weight:500">{j + 1}</th>' for j in range(SEEDS))
            + f'<th style="{_LABEL};font-weight:500;padding:0 0.5rem">{w["right"]}</th><th style="{_LABEL};font-weight:500;text-align:left">{w["vote"]}</th></tr>')
    table = f'<div style="overflow-x:auto"><table style="border-collapse:collapse">{head}{"".join(rows)}</table></div>'
    legend = "".join(f'<span style="display:inline-flex;align-items:center;gap:0.25rem;margin-right:0.8rem;{_LABEL}">'
                     f'<span style="display:inline-block;width:12px;height:12px;border-radius:2px;{style}"></span>{w[key]}</span>'
                     for key, style in (("legend_right", "background:var(--ea-pred-wash);border:1px solid var(--ea-pred)"),
                                        ("legend_wrong", "background:var(--ea-loss-wash);border:1px solid var(--ea-loss)"),
                                        ("legend_none", "background:var(--ea-inset);border:1px solid var(--ea-ink-3)")))
    return yours.figure(w["title"], table + f'<div style="margin-top:0.3rem">{legend}</div>', alt=w["title"])


_VOTE_WORDS = {
    "en": {"title": "More answers per question, more questions right", "k": "{k} answer per question|{k} answers per question",
           "right": "{n} of 12 right", "calls": "{k}× the calls and the tokens",
           "note": "For 1, 3 and 5 answers: the average over every way to pick that many of the nine runs."},
    "nl": {"title": "Meer antwoorden per vraag, meer vragen goed", "k": "{k} antwoord per vraag|{k} antwoorden per vraag",
           "right": "{n} van de 12 goed", "calls": "{k}× de aanroepen en de tokens",
           "note": "Bij 1, 3 en 5 antwoorden: het gemiddelde over elke manier om er zoveel uit de negen runs te kiezen."},
}


def vote_picture(lang: str) -> str:
    """A bar per k: how many of the twelve questions the most common of k answers got right, with what it costs."""
    w = _VOTE_WORDS[lang]
    f = findings()
    rows = []
    for k in KS:
        n = f["per_k"][k]
        one, many = w["k"].split("|")
        rows.append(f'<div style="display:grid;grid-template-columns:minmax(0,1.6fr) minmax(0,3fr);gap:0.6rem;align-items:center;margin:0.3rem 0">'
                    f'<div style="font-family:var(--ea-font-ui);font-size:13px;color:var(--ea-ink)">{(one if k == 1 else many).format(k=k)}'
                    f'<div style="{_LABEL}">{w["calls"].format(k=k)}</div></div>'
                    f'<div style="display:flex;align-items:center;gap:0.4rem"><div style="height:12px;width:{75 * n / len(QUESTIONS):.1f}%;'
                    f'background:var(--ea-pred);border-radius:2px"></div>'
                    f'<span style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink)">{w["right"].format(n=number(n, lang, 0 if n == int(n) else 1))}</span></div></div>')
    return yours.figure(w["title"], "".join(rows) + _label(w["note"]), alt=w["title"])


# --- exercise 1: majority -----------------------------------------------------------------------------------------------------

START_MAJORITY = '''def majority(answers):
    counts = {}
    for answer in answers:
        if answer in counts:
            counts[answer] = counts[answer] + 1
        else:
            counts[answer] = 1
    return answers[0]
'''
MAJORITY_CASES = [([8, 8, 12, 8, None],), ([95, 85, 85],), ([30, None, None, 31],), ([600, 640, 640, 600],),
                  ([None, None],), ([],)]

_VOTES_WORDS = {
    "en": {"title": "The votes your majority counted", "answers": "The answers: {a}", "count": "{n}×",
           "said": "Your function: {got}", "want": "The most common answer: {want}; None is no answer, and a tie goes to the answer that came first."},
    "nl": {"title": "De stemmen die je majority telde", "answers": "De antwoorden: {a}", "count": "{n}×",
           "said": "Je functie: {got}", "want": "Het meest gegeven antwoord: {want}; None is geen antwoord, en bij gelijkspel wint het antwoord dat eerst kwam."},
}


def votes_picture(seen) -> str:
    """The answers of the third case, with None in them and a tie: a bar per answer for how often it came, what the
    learner's majority gave back, and what it should give back."""
    w = _VOTES_WORDS[seen.lang]
    mine = seen.mine
    fn = mine.names.get("majority") if mine is not None else None
    if not callable(fn):
        return ""
    votes = list(MAJORITY_CASES[2][0])
    want = majority(list(votes))
    try:
        got = with_time_limit(lambda: fn(list(votes)))
    except Exception:   # the learner's code; the check says what went wrong
        return ""
    bars = []
    for value in dict.fromkeys(v for v in votes if v is not None):
        n = votes.count(value)
        bars.append(f'<div style="display:flex;align-items:center;gap:0.4rem;margin:0.1rem 0"><span style="font-family:var(--ea-font-code);font-size:12px;min-width:2.5rem;color:var(--ea-ink)">{value}</span>'
                    f'<div style="height:10px;width:{n * 2}rem;background:var(--ea-pred)"></div><span style="{_LABEL}">{w["count"].format(n=n)}</span></div>')
    color = "var(--ea-ink)" if got == want else "var(--ea-loss-text)"
    body = (_label(w["answers"].format(a=repr(votes))) + "".join(bars)
            + f'<div style="font-family:var(--ea-font-ui);font-size:13px;color:{color};margin-top:0.3rem">{html.escape(w["said"].format(got=repr(got)))}</div>'
            + _label(w["want"].format(want=repr(want))))
    return yours.figure(w["title"], body, alt=w["title"])


# --- exercise 2: ask_many --------------------------------------------------------------------------------------------------------

PENS = QUESTIONS[0][0]
TRAIN = QUESTIONS[1][0]
BY_SEED = {   # what the stand-in answers each question, per seed: the answer of seed 0 is wrong, the majority right
    PENS: ["4 pens cost 12 euros, so the change is 20 - 12 = 6.\nAnswer: 6", "4 × 3 = 12, and 20 - 12 = 8.\nAnswer: 8",
           "Mia pays 12 euros and gets 8 back.\nAnswer: 8", "The pens cost 4 × 3 = 12 euros. Change: 8.\nAnswer: 8",
           "20 - 4 = 16, 16 - 3 = 13.\nAnswer: 13"],
    TRAIN: ["From 9:40 to 11:15 is 1 hour and 35 minutes.\nAnswer: 85", "1 hour 35 minutes is 95 minutes.\nAnswer: 95",
            "20 minutes to 10:00, then 75 more.\nAnswer: 95"],
}


class _BySeed(serve.StandIn):
    """A stand-in that answers each question with the answer for the request's seed; greedy (temperature 0), it
    always gives the answer of seed 0, as a model does."""

    def answer(self, request: dict, heard) -> dict:
        question = request["messages"][-1].get("content", "") if request.get("messages") else ""
        texts = BY_SEED.get(question, ["The stand-in has no answer to this question."])
        seed = request.get("seed") or 0
        greedy = request.get("temperature") == 0
        text = texts[0 if greedy else seed % len(texts)]
        self.answers = {question: text.replace("{", "{{").replace("}", "}}")}
        return super().answer(request, heard)


SOLUTION_ASK = ('import json\nimport urllib.request\n\n\ndef ask_many(url, question, n):\n'
                '    answers = []\n'
                '    for i in range(n):\n'
                '        messages = [{"role": "user", "content": question}]\n'
                '        body = {"model": "qwen3-0.6b", "messages": messages, "max_tokens": 300,\n'
                '                "temperature": 0.7, "seed": i}\n'
                '        data = json.dumps(body).encode()\n'
                '        request = urllib.request.Request(url + "/chat/completions", data=data,\n'
                '                                         headers={"Content-Type": "application/json"})\n'
                '        with urllib.request.urlopen(request) as response:\n'
                '            answer = json.loads(response.read())\n'
                '        answers.append(answer_of(answer["choices"][0]["message"]["content"]))\n'
                '    return majority(answers)\n')
START_ASK = ('import json\nimport urllib.request\n\n\ndef ask_many(url, question, n):\n'
             '    answers = []\n'
             '    messages = [{"role": "user", "content": question}]\n'
             '    body = {"model": "qwen3-0.6b", "messages": messages, "max_tokens": 300,\n'
             '            "temperature": 0.7, "seed": 0}\n'
             '    data = json.dumps(body).encode()\n'
             '    request = urllib.request.Request(url + "/chat/completions", data=data,\n'
             '                                     headers={"Content-Type": "application/json"})\n'
             '    with urllib.request.urlopen(request) as response:\n'
             '        answer = json.loads(response.read())\n'
             '    answers.append(answer_of(answer["choices"][0]["message"]["content"]))\n'
             '    return majority(answers)\n')


def _posts(log: list) -> list:
    return [e["body"] for e in log if e["method"] == "POST" and isinstance(e["body"], dict)]


def _judge_ask(question: str, n: int):
    want = majority([answer_of(t) for t in [BY_SEED[question][i % len(BY_SEED[question])] for i in range(n)]])

    def judge(got, running):
        sent = _posts(running.log)
        if len(sent) != n:
            return "one" if len(sent) == 1 else "count"
        if any(b.get("temperature") == 0 for b in sent):
            return "greedy"
        if len({json.dumps(b.get("seed")) for b in sent}) < n:
            return "same-seed"
        if isinstance(got, list):
            return "list"
        return None if got == want else "other"
    return judge


ASK_CASES = [Case(lambda: _BySeed(), (URL, PENS, 5), _judge_ask(PENS, 5)),
             Case(lambda: _BySeed(), (URL, TRAIN, 3), _judge_ask(TRAIN, 3))]

_ASKS_WORDS = {"en": {"title": "What your ask_many sent for the pens, five times", "sent": "Request {k}: seed {seed}, temperature {t}",
                      "number": "the number in the answer: {n}", "gave": "Your function gave back: {got}", "none": "Your function sent nothing to the server yet."},
               "nl": {"title": "Wat je ask_many stuurde voor de pennen, vijf keer", "sent": "Request {k}: seed {seed}, temperature {t}",
                      "number": "het getal in het antwoord: {n}", "gave": "Je functie gaf terug: {got}", "none": "Je functie stuurde nog niets naar de server."}}


def asks_picture(seen) -> str:
    """The requests the learner's ask_many sent for the pens with n 5: the seed and temperature of each, and the number
    in what came back; then what the function gave back."""
    w = _ASKS_WORDS[seen.lang]
    run = seen.run
    fn = run.names.get("ask_many") if run is not None and run.error is None else None
    if not callable(fn):
        return ""
    got, error, running = apicheck.call(fn, ASK_CASES[0])
    rows = []
    for k, e in enumerate([e for e in running.log if e["method"] == "POST" and isinstance(e["body"], dict)][:9]):
        try:
            said = e["answer"]["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            said = ""
        rows.append(f'<div style="font-family:var(--ea-font-ui);font-size:13px;color:var(--ea-ink)">{html.escape(w["sent"].format(k=k + 1, seed=e["body"].get("seed"), t=e["body"].get("temperature")))}'
                    f' <span style="color:var(--ea-pred-text)">· {html.escape(w["number"].format(n=answer_of(said)))}</span></div>')
    if not rows:
        rows.append(_label(w["none"]))
    end = f'{type(error).__name__}: {error}' if error is not None else repr(got)
    return yours.figure(w["title"], "".join(rows) + _label(w["gave"].format(got=end)), alt=w["title"])


def _not_bytes(e) -> bool:
    return isinstance(e, TypeError) and "bytes" in str(e)


_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}

EXERCISES = {
    "majority": FunctionBox(
        name="majority",
        picture=votes_picture,
        function="majority",
        start=START_MAJORITY,
        solution=SOLUTION_MAJORITY,
        visible=[MAJORITY_CASES[0]],
        hidden=MAJORITY_CASES[1:],
        mistakes=[
            Mistake("counts-none", wrong=SOLUTION_MAJORITY.replace("        if answer is not None:\n            if answer in counts:\n                counts[answer] = counts[answer] + 1\n            else:\n                counts[answer] = 1\n",
                                                                   "        if answer in counts:\n            counts[answer] = counts[answer] + 1\n        else:\n            counts[answer] = 1\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your function counts `None` as an answer too, and that upsets the count. `None` means the answer had no number, so it is no vote. Skip it with `if answer is not None:` before you count.",
                    nl="`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: je functie telt `None` ook als antwoord, en dat gooit de telling in de war. `None` betekent dat het antwoord geen getal had, dus het is geen stem. Sla het over met `if answer is not None:` voordat je telt."),
            Mistake("last-tie", state="almost", wrong=SOLUTION_MAJORITY.replace("counts[answer] > counts[best]", "counts[answer] >= counts[best]"),
                    en="Right when one answer has the most votes. But with a tie, `{call}` gives `{got}`: the answer that came last. Let a later answer win only with more votes, `>`, so that a tie goes to the answer that came first.",
                    nl="Goed als één antwoord de meeste stemmen heeft. Maar bij gelijkspel geeft `{call}` `{got}`: het antwoord dat het laatst kwam. Laat een later antwoord alleen winnen met meer stemmen, `>`, zodat bij gelijkspel het antwoord wint dat eerst kwam."),
            Mistake("count", wrong=SOLUTION_MAJORITY.replace("    return best\n", "    return counts[best] if best is not None else None\n"),
                    en="`{call}` gives `{got}`: how often the most common answer came, not the answer itself. Give back the answer, the key in `counts`.",
                    nl="`{call}` geeft `{got}`: hoe vaak het meest gegeven antwoord kwam, niet het antwoord zelf. Geef het antwoord terug, de key in `counts`."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`majority` counts the answers, but still gives back the first one. Skip `None`, and give back the answer with the most votes.",
                      "nl": "`majority` telt de antwoorden, maar geeft nog het eerste terug. Sla `None` over, en geef het antwoord met de meeste stemmen terug."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: the answer that came most often, without `None`, and with a tie the one that came first.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: het antwoord dat het vaakst kwam, zonder `None`, en bij gelijkspel het antwoord dat eerst kwam."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work with a tie, with only `None`, and with an empty list?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Werkt je functie bij gelijkspel, met alleen `None`, en met een lege lijst?"},
            "no_return": {"en": "With `{call}`, your function gives `None`. Does it count `None` as a vote? Leave it out with `if answer is not None:`. And does every way through your function end with `return` and an answer?",
                          "nl": "Met `{call}` geeft je functie `None`. Telt hij `None` als stem? Laat het weg met `if answer is not None:`. En eindigt elke weg door je functie met `return` en een antwoord?"},
            "unreached": {"en": "With `{call}`, your function gives `None`. Does it count `None` as a vote? Leave it out with `if answer is not None:`. And does every way through your function end with `return` and an answer?",
                          "nl": "Met `{call}` geeft je functie `None`. Telt hij `None` als stem? Laat het weg met `if answer is not None:`. En eindigt elke weg door je functie met `return` en een antwoord?"},
            "no_function": {"en": "Your box does not make a function `majority`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `majority`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `majority` is right in all {n} cases: the answer with the most votes, `None` left out, and a tie to the answer that came first.",
                        "nl": "Je `majority` klopt in alle {n} gevallen: het antwoord met de meeste stemmen, `None` weggelaten, en bij gelijkspel het antwoord dat eerst kwam."},
        },
    ),
    "ask_many": ApiBox(
        name="ask_many",
        picture=asks_picture,
        function="ask_many",
        start=START_ASK,
        solution=SOLUTION_ASK,
        names={"answer_of": answer_of, "majority": majority},
        cases=ASK_CASES,
        mistakes=[
            Mistake("data-text", error=_not_bytes,
                    en="urllib sends bytes, not text. Keep `.encode()` after `json.dumps(body)`, as the box had it.",
                    nl="urllib verstuurt bytes, geen tekst. Laat `.encode()` achter `json.dumps(body)` staan, zoals het in het vak stond."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`ask_many` still asks once, so the majority of one answer is just that answer. Ask `n` times, each time with another seed.",
                      "nl": "`ask_many` vraagt nog één keer, dus de meerderheid van één antwoord is gewoon dat antwoord. Vraag `n` keer, elke keer met een andere seed."},
            "correct": {"en": "Your `ask_many` asks `n` times, each with its own seed and with sampling, and gives back the most common number: 8 for the pens, where the first answer alone said 6.",
                        "nl": "Je `ask_many` vraagt `n` keer, elk met een eigen seed en met sampling, en geeft het meest gegeven getal terug: 8 voor de pennen, waar het eerste antwoord alleen 6 zei."},
            "one": {"en": "With `{call}`, your function sent one request. Ask `n` times: put the request in a loop, `for i in range(n):`.",
                    "nl": "Met `{call}` stuurde je functie één request. Vraag `n` keer: zet het request in een loop, `for i in range(n):`."},
            "count": {"en": "With `{call}`, your function sent another number of requests than `n`. Let the loop run exactly `n` times, with `range(n)`.",
                      "nl": "Met `{call}` stuurde je functie een ander aantal requests dan `n`. Laat de loop precies `n` keer lopen, met `range(n)`."},
            "greedy": {"en": "With `{call}`, a request has temperature 0: greedy, the same answer every time, so the vote has nothing to choose from. Keep `\"temperature\": 0.7`.",
                       "nl": "Met `{call}` heeft een request temperature 0: greedy, elke keer hetzelfde antwoord, dus de stemming heeft niets te kiezen. Houd `\"temperature\": 0.7`."},
            "same-seed": {"en": "With `{call}`, two requests have the same seed, so they get the same answer: a vote of copies. Give each request its own seed, such as `\"seed\": i`.",
                          "nl": "Met `{call}` hebben twee requests dezelfde seed, dus ze krijgen hetzelfde antwoord: een stemming van kopieën. Geef elk request een eigen seed, zoals `\"seed\": i`."},
            "list": {"en": "With `{call}`, your function gives back the list of numbers. Give back the most common one, with `majority(answers)`.",
                     "nl": "Met `{call}` geeft je functie de lijst getallen terug. Geef het meest gegeven getal terug, met `majority(answers)`."},
            "other": {"en": "With `{call}`, the requests are right, but your function gives back `{got}`. Give back `majority(answers)`, the most common number of the answers.",
                      "nl": "Met `{call}` kloppen de requests, maar je functie geeft `{got}` terug. Geef `majority(answers)` terug, het meest gegeven getal van de antwoorden."},
        },
    ),
}

REQUIRED = ["majority", "ask_many", "worth"]

CASES = [
    ("majority", START_MAJORITY, "empty", None),
    ("majority", "", "empty", None),
    ("majority", SOLUTION_MAJORITY, "correct", None),
    ("majority", '''def majority(answers):
    counts = {}
    for answer in answers:
        if answer is not None:
            counts[answer] = counts.get(answer, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)
''', "correct", None),
    ("majority", SOLUTION_MAJORITY.replace("        if answer is not None:\n            if answer in counts:\n                counts[answer] = counts[answer] + 1\n            else:\n                counts[answer] = 1\n",
                                           "        if answer in counts:\n            counts[answer] = counts[answer] + 1\n        else:\n            counts[answer] = 1\n"), "wrong", "counts-none"),
    ("majority", SOLUTION_MAJORITY.replace("counts[answer] > counts[best]", "counts[answer] >= counts[best]"), "almost", "last-tie"),
    ("majority", SOLUTION_MAJORITY.replace("    return best\n", "    return counts[best] if best is not None else None\n"), "wrong", "count"),
    ("majority", SOLUTION_MAJORITY.replace("    best = None\n", "    best = None\n    if len(answers) == 2:\n        return 0\n"), "almost", None),
    ("majority", '''def majority(answers):
    counts = {}
    for answer in answers:
        counts[answer] = counts.get(answer, 0) + 1
    return max(counts, key=counts.get) if counts else None
''', "wrong", "unreached"),
    ("ask_many", START_ASK, "empty", None),
    ("ask_many", SOLUTION_ASK, "correct", None),
    ("ask_many", START_ASK.replace('"seed": 0', '"seed": 1'), "wrong", "one"),
    ("ask_many", SOLUTION_ASK.replace("range(n)", "range(n + 1)"), "wrong", "count"),
    ("ask_many", SOLUTION_ASK.replace('"temperature": 0.7', '"temperature": 0'), "wrong", "greedy"),
    ("ask_many", SOLUTION_ASK.replace('"seed": i', '"seed": 0'), "wrong", "same-seed"),
    ("ask_many", SOLUTION_ASK.replace("    return majority(answers)\n", "    return answers\n"), "wrong", "list"),
    ("ask_many", SOLUTION_ASK.replace("    return majority(answers)\n", "    return answers[0]\n"), "wrong", "other"),
    ("ask_many", SOLUTION_ASK.replace("json.dumps(body).encode()", "json.dumps(body)"), "error", "data-text"),
]


# --- the demo: your ask_many, against the course's own server ----------------------------------------------------------------

def downloaded() -> bool:
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    chat.download_or_stop(button, texts)


def learner_ask_many(majority_text, ask_text):
    """The learner's ask_many from exercise 2 with their own majority from exercise 1, when both are right; else None."""
    from everything_ai.check import MISSING
    for name, text in (("majority", majority_text), ("ask_many", ask_text)):
        if text is MISSING or text is None or EXERCISES[name].check(text).state != "correct":
            return None
    mine = run_code(majority_text).names["majority"]
    return run_code(ask_text, {"answer_of": answer_of, "majority": mine}).names["ask_many"]


def demo(ask_many, question: str, n: int, words: dict, lang: str) -> str:
    """The learner's ask_many with a question of the learner's, asked n times through the course's server with Qwen3
    behind it, in the words of the lesson's prompt: the number of each answer, and what the vote gave back."""
    import time
    running = serve.model_server()
    before = len(running.log)
    start = time.perf_counter()
    try:
        got = ask_many(running.url, STEPS.format(question=question), n)
    except Exception as e:   # the learner's code, against the real server; it may have run out of memory
        said = words["memory"] if serve.out_of_memory(e) else words["failed"].format(why=f"{type(e).__name__}: {e}")
        return f"<p><em>{html.escape(said)}</em></p>"
    rows = []
    for k, e in enumerate(e for e in running.log[before:] if e["method"] == "POST"):
        try:
            said = e["answer"]["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            said = ""
        last = said.strip().splitlines()[-1] if said.strip() else ""
        rows.append(f'<div style="font-family:var(--ea-font-ui);font-size:13px;color:var(--ea-ink)">{html.escape(words["run"].format(k=k + 1, n=answer_of(said)))}'
                    f' <span style="font-family:var(--ea-font-code);font-size:12px;color:var(--ea-pred-text)">{html.escape(last[:80])}</span></div>')
    seconds = number(time.perf_counter() - start, lang, 0)
    return yours.figure(words["title"], "".join(rows) + f'<div style="font-family:var(--ea-font-ui);font-size:14px;font-weight:600;color:var(--ea-pred-text);margin-top:0.4rem">{html.escape(words["vote"].format(got=got, seconds=seconds))}</div>', alt=words["title"])
