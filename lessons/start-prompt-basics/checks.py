# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson start-prompt-basics, and its data: three questions, each asked to Qwen3-0.6B (the chat model)
in seven ways, eight times each with another seed (data/course-prompts-qwen.json, made by
data/tools/record_course_prompts_qwen.py with this file), and how the lesson reads an answer: the last number in it."""
import html
import json
import re

from everything_ai.check import Sort

COURSE = "data/course-prompts-qwen.json"
SEEDS = 8
MAX_TOKENS = 300

# The questions, with their answers. Made up for the course, in the style of school sums.
QUESTIONS = {
    "bakery": ("A bakery sells a loaf of bread for 2.50 euros. On Monday it sold 36 loaves, and on Tuesday 12 more than "
               "on Monday. How much money did the bakery take in on those two days together?", 210),
    "pencils": ("Lena has 3 boxes with 12 pencils each. Lena gives 7 pencils to a friend and then buys a pack of 5. "
                "How many pencils does Lena have now?", 34),
    "cinema": ("A cinema has 12 rows of 18 seats. For a show, 157 tickets are sold. How many seats stay empty?", 59),
}
# Two worked examples for the few-shot prompts: question, answer, and the steps to it.
EXAMPLES = [("Tom has 4 bags with 6 apples each. Tom eats 3 apples. How many apples are left?", 21,
             "4 bags with 6 apples is 4 × 6 = 24 apples. Tom eats 3, so 24 − 3 = 21."),
            ("A ticket costs 8 euros. How much do 5 tickets and a 3-euro program cost together?", 43,
             "5 tickets cost 5 × 8 = 40 euros. With the program: 40 + 3 = 43.")]
ROLE = "You are a patient math teacher for children of ten."
WAYS = ["plain", "delimit", "number", "examples", "examples-steps", "role", "steps"]


def messages(way: str, question: str) -> list:
    """The messages of one way of asking `question`."""
    if way == "plain":
        return [{"role": "user", "content": question}]
    if way == "delimit":
        return [{"role": "user", "content": f"Answer the question between the <question> tags.\n\n<question>\n{question}\n</question>"}]
    if way == "number":
        return [{"role": "user", "content": f"{question}\n\nReply with only the final number, nothing else."}]
    if way == "examples":
        shots = "".join(f"Question: {q}\nAnswer: {a}\n\n" for q, a, _ in EXAMPLES)
        return [{"role": "user", "content": f"{shots}Question: {question}\nAnswer:"}]
    if way == "examples-steps":
        shots = "".join(f"Question: {q}\nAnswer: {steps} The answer is {a}.\n\n" for q, a, steps in EXAMPLES)
        return [{"role": "user", "content": f"{shots}Question: {question}\nAnswer:"}]
    if way == "role":
        return [{"role": "system", "content": ROLE}, {"role": "user", "content": question}]
    if way == "steps":
        return [{"role": "user", "content": f"{question}\n\nWork it out step by step. Then give the final answer on the last line, as: Answer: <number>"}]
    raise ValueError(way)


def read(text: str):
    """The answer in a text: its last number, without thousands separators, as an int when it is whole; None when
    there is no number."""
    found = re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))
    if not found:
        return None
    x = float(found[-1])
    return int(x) if x == int(x) else x


def course() -> dict:
    from everything_ai import paths
    return json.loads((paths.ROOT / COURSE).read_text(encoding="utf-8"))


def downloaded() -> bool:
    from everything_ai import chat
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    from everything_ai import chat
    chat.download_or_stop(button, texts)


def ask_live(prompt: str, show, words: dict, lang: str) -> str:
    """The demo: the learner's prompt as one user message to Qwen3, sampled with seed 0, with what it writes so far
    shown by `show(html)` at most five times a second. The picture of the answer, or the message of what went wrong."""
    from everything_ai import chat
    a = chat.answer_live([{"role": "user", "content": prompt}],
                         lambda text: show(_answer_picture(text or "", words["loading"] if text is None else words["writing"], words)),
                         max_tokens=MAX_TOKENS, seed=0)
    if isinstance(a, str):
        return f'<p><em>{html.escape(words[a])}</em></p>'
    got = read(a["message"]["content"])
    note = words["done"].format(tokens=a["tokens"], seconds=number(a["seconds"], 1, lang), read=words["nothing"] if got is None else number(got, 0 if isinstance(got, int) else 2, lang))
    return _answer_picture(a["message"]["content"], note, words)


def _answer_picture(text: str, note: str, words: dict) -> str:
    from everything_ai import yours
    body = (f'<div style="font-family:var(--ea-font-code);font-size:13px;line-height:1.45;white-space:pre-wrap;overflow-wrap:anywhere;'
            f'color:var(--ea-pred-text)">{html.escape(text)}</div>'
            f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2);margin-top:0.3rem">{html.escape(note)}</div>')
    return yours.figure(words["title"], body, alt=words["title"])


def number(x: float, digits: int, lang: str) -> str:
    text_ = f"{x:,.{digits}f}"
    if lang == "nl":
        text_ = text_.replace(",", "_").replace(".", ",").replace("_", ".")
    return text_


# --- what the course's answers show ---------------------------------------------------------------------------------

_WAY = {"plain": {"en": "the question as it is", "nl": "de vraag zoals hij is"},
        "delimit": {"en": "between delimiters", "nl": "tussen scheidingstekens"},
        "number": {"en": "only the number", "nl": "alleen het getal"},
        "examples": {"en": "two examples, answers only", "nl": "twee voorbeelden, alleen antwoorden"},
        "examples-steps": {"en": "two examples with their steps", "nl": "twee voorbeelden met hun stappen"},
        "role": {"en": "a role in the system message", "nl": "een rol in het system-bericht"},
        "steps": {"en": "step by step", "nl": "stap voor stap"}}
_WHO = {"system": {"en": "system", "nl": "system"}, "user": {"en": "you", "nl": "jij"}}


def way_name(way: str, lang: str) -> str:
    return _WAY[way][lang]


def results(q: str, way: str, R: dict = None) -> list:
    """The number the lesson reads in each of the course's answers to question `q` asked in `way`, seed after seed."""
    R = R or course()
    return [read(a["text"]) for a in R["answers"][q][way]]


def right(q: str, way: str, R: dict = None) -> int:
    """In how many of the course's runs the answer to `q`, asked in `way`, was right."""
    return sum(x == QUESTIONS[q][1] for x in results(q, way, R))


def tokens(way: str, R: dict = None, qs=None) -> float:
    """The mean number of tokens of an answer asked in `way`, over the questions `qs` (all three when None)."""
    R = R or course()
    got = [a["tokens"] for q in (qs or QUESTIONS) for a in R["answers"][q][way]]
    return sum(got) / len(got)


def _typical(q: str, way: str, R: dict) -> dict:
    """An answer with the result the model wrote most often, a wrong one where a wrong and the right result are as
    common: the first such answer, by seed."""
    got = results(q, way, R)
    common = max(got, key=lambda x: (got.count(x), x != QUESTIONS[q][1]))
    return R["answers"][q][way][got.index(common)]


def _prompt_html(way: str, question: str, lang: str) -> str:
    rows = []
    for m in messages(way, question):
        who = _WHO[m["role"]][lang]
        rows.append(f'<div style="display:flex;gap:0.6rem;padding:0.3rem 0;border-top:1px solid var(--ea-line)">'
                    f'<span style="flex:0 0 3.6rem;font-family:var(--ea-font-ui);font-size:0.8rem;font-weight:600;color:var(--ea-ink-2)">{html.escape(who)}</span>'
                    f'<span style="flex:1 1 auto;font-family:var(--ea-font-code);font-size:0.8rem;line-height:1.45;white-space:pre-wrap;'
                    f'overflow-wrap:anywhere;color:var(--ea-ink)">{html.escape(m["content"])}</span></div>')
    return "".join(rows)


def way_frame(i: int, q: str, lang: str) -> tuple:
    """Way i of the seven on the slider: the prompt, the number read in each of the eight answers (right ones in the
    colour of what a model writes, wrong ones in the colour of mistakes), and one answer with the most common result."""
    R = course()
    way = WAYS[i - 1]
    question, answer = QUESTIONS[q]
    got = results(q, way, R)
    k = right(q, way, R)
    t = tokens(way, R, [q])
    chips = "".join(
        f'<span style="display:inline-block;min-width:2.6rem;text-align:center;margin:0 0.3rem 0.3rem 0;padding:0.15rem 0.35rem;'
        f'border:1px solid {"var(--ea-pred)" if x == answer else "var(--ea-loss)"};border-radius:4px;font-family:var(--ea-font-code);'
        f'font-size:0.85rem;color:{"var(--ea-pred-text)" if x == answer else "var(--ea-loss-text)"}">'
        f'{"✓" if x == answer else "✗"} {html.escape("–" if x is None else number(x, 0 if isinstance(x, int) else 2, lang))}</span>'
        for x in got)
    one = _typical(q, way, R)
    words = {"en": {"prompt": f"Way {i} of {len(WAYS)}: {way_name(way, lang)}", "read": f"The number read in each of the {len(got)} answers, seed after seed",
                    "one": f"The answer of seed {one['seed']}, {one['tokens']} tokens"},
             "nl": {"prompt": f"Manier {i} van {len(WAYS)}: {way_name(way, lang)}", "read": f"Het getal dat in elk van de {len(got)} antwoorden staat, seed na seed",
                    "one": f"Het antwoord van seed {one['seed']}, {one['tokens']} tokens"}}[lang]
    label = 'style="font-family:var(--ea-font-ui);font-size:0.85rem;font-weight:600;color:var(--ea-ink-2);margin:0.6rem 0 0.25rem"'
    body = (f'<div {label}>{html.escape(words["prompt"])}</div>{_prompt_html(way, question, lang)}'
            f'<div {label}>{html.escape(words["read"])}</div><div>{chips}</div>'
            f'<div {label}>{html.escape(words["one"])}</div>'
            f'<div style="max-height:13rem;overflow-y:auto;font-family:var(--ea-font-code);font-size:0.8rem;line-height:1.45;white-space:pre-wrap;'
            f'overflow-wrap:anywhere;color:var(--ea-pred-text);border-left:2px solid var(--ea-line);padding-left:0.6rem">{html.escape(one["text"])}</div>')
    if lang == "en":
        sentence = f"**Way {i}, {way_name(way, lang)}:** right in {k} of {len(got)} runs. Its answers had {number(t, 0, lang)} tokens on average."
    else:
        sentence = f"**Manier {i}, {way_name(way, lang)}:** goed in {k} van de {len(got)} runs. De antwoorden hadden gemiddeld {number(t, 0, lang)} tokens."
    return body, sentence


def number_reveal(lang: str) -> str:
    """What the guess about only the number shows, for the bakery: how often each way was right, and why."""
    R = course()
    n, p, k = R["seeds"], right("bakery", "plain", R), right("bakery", "number", R)
    t_plain, t_number = number(tokens("plain", R, ["bakery"]), 0, lang), number(tokens("number", R, ["bakery"]), 0, lang)
    seed0 = R["answers"]["bakery"]["number"][0]
    assert "36 + 12 = 48" in seed0["text"] and read(seed0["text"]) == 120, "the example below is no longer in the course's run"
    assert p > k and tokens("number", R, ["bakery"]) < tokens("plain", R, ["bakery"])
    if lang == "en":
        return (f"**The question as it is was right in {p} of {n} runs; with only the number, in {k} of {n}.** Even asked for only the number, "
                f"Qwen3 still wrote a few steps, {t_number} tokens on average against {t_plain}, and in those short answers it slipped: "
                f"seed 0, for one, takes 36 + 12 = 48 loaves as the loaves of both days together, and ends at 120 euros. The steps a model "
                f"writes out are where it works the sum out.")
    return (f"**De vraag zoals hij is, was goed in {p} van de {n} runs; met alleen het getal in {k} van de {n}.** Ook gevraagd om alleen het getal "
            f"schreef Qwen3 nog een paar stappen, gemiddeld {t_number} tokens tegen {t_plain}, en in die korte antwoorden ging het mis: seed 0 "
            f"bijvoorbeeld neemt 36 + 12 = 48 broden als de broden van de twee dagen samen, en komt uit op 120 euro. In de stappen die een model "
            f"uitschrijft, rekent het de som uit.")


def table(lang: str) -> str:
    """How often each way was right, for each of the three questions, and how long its answers were."""
    R = course()
    head = {"en": ["Way", "Bakery", "Pencils", "Cinema", "Tokens per answer"],
            "nl": ["Manier", "Bakkerij", "Potloden", "Bioscoop", "Tokens per antwoord"]}[lang]
    lines = ["| " + " | ".join(head) + " |", "|---|---|---|---|---|"]
    for way in WAYS:
        lines.append(f"| {way_name(way, lang)} | " + " | ".join(str(right(q, way, R)) for q in QUESTIONS) + f" | {number(tokens(way, R), 0, lang)} |")
    note = {"en": f"*How many of the {R['seeds']} answers were right, for each question; and the mean number of tokens of an answer.*",
            "nl": f"*Hoeveel van de {R['seeds']} antwoorden goed waren, per vraag; en het gemiddelde aantal tokens van een antwoord.*"}[lang]
    return "\n".join(lines) + "\n\n" + note



def ways_after(lang: str) -> str:
    """What the table of the seven ways shows: the ways that were always right, the three with the shortest answers and
    their mistakes, the examples that answered the first example, and the role."""
    R = course()
    seeds, n = R["seeds"], R["seeds"] * len(QUESTIONS)
    always = [w for w in WAYS if all(right(q, w, R) == seeds for q in QUESTIONS)]
    short = sorted(WAYS, key=lambda w: tokens(w, R))[:3]
    wrong = sum(seeds - right(q, w, R) for w in short for q in QUESTIONS)
    bakery = sum(seeds - right("bakery", w, R) for w in short)
    misread = [x for w in short for x in results("bakery", w, R) if x != QUESTIONS["bakery"][1]]
    answers = [(a["text"], read(a["text"])) for q in QUESTIONS for a in R["answers"][q]["examples"]]
    tom = sum(x == EXAMPLES[0][1] for _, x in answers)
    ticket = sum(x == EXAMPLES[1][1] for _, x in answers)
    assert all("apple" in t.lower() for t, x in answers if x == EXAMPLES[0][1]) and all("ticket" in t.lower() for t, x in answers if x == EXAMPLES[1][1])
    with_steps = sum(right(q, "examples-steps", R) for q in QUESTIONS)
    assert always == ["plain", "role", "steps"] and short == ["number", "examples", "delimit"], "the text below no longer fits the course's runs"
    assert max(set(misread), key=misread.count) == 120 and sum(right(q, "role", R) for q in QUESTIONS) == sum(right(q, "plain", R) for q in QUESTIONS)
    if lang == "en":
        return (f"**Three ways were right in all {n} runs: the question as it is, a role in the system message, and step by step.** The three ways "
                f"with the shortest answers, only the number, two examples with answers only, and between delimiters, were wrong {wrong} times, "
                f"{bakery} of them on the bakery. Most often, Qwen3 took the 48 loaves of Tuesday as the loaves of both days, and ended at 120 euros. "
                f"The examples with answers only went wrong in another way too: {tom + ticket} times, Qwen3 answered one of the examples instead, "
                f"{tom} times the first, *Tom has 4 bags…*, with 21, and {'once' if ticket == 1 else f'{ticket} times'} the second, with 43. With their steps, the examples were right in {with_steps} of {n} runs. And the role made no difference: with it, Qwen3 was right in "
                f"all {n} runs, just as without it. So, for a sum, give the model room for its steps, and a fixed place for its "
                f"answer, such as *Answer:* on the last line.")
    return (f"**Drie manieren waren in alle {n} runs goed: de vraag zoals hij is, een rol in het system-bericht, en stap voor stap.** De drie "
            f"manieren met de kortste antwoorden, alleen het getal, twee voorbeelden met alleen antwoorden, en tussen scheidingstekens, waren "
            f"{wrong} keer fout, {bakery} keer bij de bakkerij. Het vaakst zag Qwen3 de 48 broden van dinsdag aan voor de broden van beide dagen, en "
            f"kwam het uit op 120 euro. De voorbeelden met alleen antwoorden gingen ook nog op een andere manier mis: {tom + ticket} keer beantwoordde "
            f"Qwen3 een van de voorbeelden, {tom} keer het eerste, *Tom has 4 bags…*, met 21, en {'één' if ticket == 1 else ticket} keer het tweede, met 43. Met hun stappen erbij waren de voorbeelden in {with_steps} van de {n} runs goed. En de "
            f"rol maakte geen verschil: ermee was Qwen3 in alle {n} runs goed, net als zonder. Dus geef het model bij een som ruimte voor "
            f"zijn stappen, en een vaste plek voor zijn antwoord, zoals *Answer:* op de laatste regel.")

# The lesson is done when these are correct. "sum-prompt" and "message-role" are choice questions in the lesson.
REQUIRED = ["parts", "sum-prompt", "message-role"]

_CONTEXT = {"en": "Is this something the model should do, or something it needs to know?",
            "nl": "Is dit iets wat het model moet doen, of iets wat het moet weten?"}
_FORM = {"en": "Does this line say what to do, or how the answer must be written?",
         "nl": "Zegt deze regel wat het model moet doen, of hoe het antwoord eruit moet zien?"}

EXERCISES = {
    "parts": Sort(
        name="parts",
        groups={
            "instruction": {"en": "Instruction", "nl": "Instructie"},
            "context": {"en": "Context", "nl": "Context"},
            "example": {"en": "Example", "nl": "Voorbeeld"},
            "form": {"en": "Form of the answer", "nl": "Vorm van het antwoord"},
        },
        items={
            "landlord": {"en": "The message is from my landlord.", "nl": "Het bericht is van mijn verhuurder."},
            "short": {"en": "Use at most two sentences.", "nl": "Gebruik hooguit twee zinnen."},
            "translate": {"en": "Translate the message between the tags into Dutch.", "nl": "Vertaal het bericht tussen de tags naar het Engels."},
            "example": {"en": "English: Good morning! Dutch: Goedemorgen!", "nl": "Nederlands: Goedemorgen! Engels: Good morning!"},
            "message": {"en": "`<message>` The plumber comes on Tuesday between 8 and 12, so please leave the door open. `</message>`",
                        "nl": "`<bericht>` De loodgieter komt dinsdag tussen 8 en 12 uur, laat de deur dus open. `</bericht>`"},
            "only": {"en": "Reply with the translation only.", "nl": "Geef alleen de vertaling."},
        },
        solution={"landlord": "context", "short": "form", "translate": "instruction", "example": "example", "message": "context", "only": "form"},
        nudge={"landlord": _CONTEXT, "message": _CONTEXT, "short": _FORM, "only": _FORM,
               "translate": {"en": "Which line says what the model should do with the message?",
                             "nl": "Welke regel zegt wat het model met het bericht moet doen?"},
               "example": {"en": "Does this line ask for something, or show what a finished answer looks like?",
                           "nl": "Vraagt deze regel iets, of laat hij zien hoe een af antwoord eruitziet?"}},
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the six lines again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de zes regels opnieuw uit."},
            "empty": {"en": "Pick a group for each of the six lines.",
                      "nl": "Kies voor elk van de zes regels een groep."},
            "unfinished": {"en": "Everything you picked so far is right. Pick a group for the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe koos, klopt. Kies ook voor de rest een groep (nog {n})."},
            "wrong": {"en": "“{item}” is in the wrong group.",
                      "nl": "‘{item}’ staat in de verkeerde groep."},
            "correct": {"en": "All {n} are right. One instruction, two pieces of context, one example and two lines about the form: every part has its own lines, so the model can tell them apart.",
                        "nl": "Alle {n} kloppen. Eén instructie, twee stukken context, één voorbeeld en twee regels over de vorm: elk deel heeft zijn eigen regels, zodat het model ze uit elkaar kan houden."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner chose, state, known mistake).
_RIGHT = {"landlord": "context", "short": "form", "translate": "instruction", "example": "example", "message": "context", "only": "form"}
CASES = [
    ("parts", {}, "empty", None),
    ("parts", _RIGHT, "correct", None),
    ("parts", {"translate": "instruction", "message": "context"}, "wrong", "unfinished"),
    ("parts", {**_RIGHT, "landlord": "instruction"}, "wrong", "wrong:landlord"),
    ("parts", {**_RIGHT, "only": "instruction"}, "wrong", "wrong:only"),
    ("parts", {**_RIGHT, "example": "context", "message": "example"}, "wrong", "wrong:example"),
]
