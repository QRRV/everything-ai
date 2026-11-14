# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson start-prompt-chains, and its data: Qwen3-0.6B (the chat model) on four small experiments, each
eight times with another seed (data/course-chains-qwen.json, made by data/tools/record_course_chains_qwen.py with this
file): one prompt against a chain of two for a shopping sum; checking its own answers to the sums of start-prompt-basics;
Chain-of-Verification for a list of prime numbers; and answering from a source, with the number of the sentence."""
import html
import json
import re

from everything_ai.check import Sort

COURSE = "data/course-chains-qwen.json"
SEEDS = 8
MAX_TOKENS = 300
ONE_TOKENS = 2 * MAX_TOKENS     # the one prompt may write as much as the two prompts of the chain together

# --- 1. one prompt or a chain of two -----------------------------------------------------------------------------------
RECIPE = "A pancake recipe for 4 people needs 200 g flour, 500 ml milk, 2 eggs and 40 g butter."
PEOPLE = 12
PRICES = "Flour costs 1 euro per 100 g, milk 1 euro per 500 ml, an egg 0.50 euros, and butter 1 euro per 40 g."
STEPS = "Work it out step by step."
LAST_LINE = "Then give the final answer on the last line, as: Answer: <euros>"
AMOUNTS = "Then write the line Amounts: and under it the amount of each ingredient, one per line."
COST = 15              # for 12 people: 600 g flour 6 euros, 1500 ml milk 3, 6 eggs 3, 120 g butter 3
SCALED = {"flour": 600, "milk": 1500, "eggs": 6, "butter": 120}


def one_prompt() -> list:
    return [{"role": "user", "content": f"{RECIPE} I make pancakes for {PEOPLE} people. {PRICES} How much do the ingredients cost? {STEPS} {LAST_LINE}"}]


def chain_first() -> list:
    return [{"role": "user", "content": f"{RECIPE} How much of each do I need for {PEOPLE} people? {STEPS} {AMOUNTS}"}]


def amounts_of(text: str) -> str:
    """What the first prompt of the chain hands to the second: the lines after its last line Amounts:, or all of it
    when it wrote no such line."""
    parts = re.split(r"(?i)amounts:", text)
    rest = parts[-1].strip(" *\n") if len(parts) > 1 else ""
    return rest or text.strip()


def chain_second(amounts: str) -> list:
    return [{"role": "user", "content": f"I need:\n{amounts}\n\n{PRICES} How much do these ingredients cost together? {STEPS} {LAST_LINE}"}]


def scaled_right(text: str) -> bool:
    """Whether the amounts a text hands on are the amounts for 12 people: 600 g flour, 1500 ml (or 1.5 l) milk, 6 eggs
    and 120 g butter, each named on one line with its number."""
    lines = amounts_of(text).lower().splitlines()

    def has(word, *numbers):
        return any(word in line and any(re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", line.replace(",", "")) for n in numbers) for line in lines)
    return has("flour", "600") and has("milk", "1500", "1.5") and has("egg", "6") and has("butter", "120")


# --- 2. checking its own answer (self-refine) --------------------------------------------------------------------------
CHECK = "Check your answer step by step. If it is wrong, fix it. Give the final answer on the last line, as: Answer: <number>"


def markdown_safe(text: str) -> str:
    """A prompt as text in Markdown: with < and > written out, so that <number> shows and is not taken for HTML."""
    return text.replace("<", "&lt;").replace(">", "&gt;")


def check_own(question: str, answer: str) -> list:
    """The second prompt of Self-Refine: the question, the model's own first answer, and the request to check it."""
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}, {"role": "user", "content": CHECK}]


REFINE_WAYS = ("plain", "number")     # the ways of start-prompt-basics whose first answers get checked
SUMS = {"bakery": {"en": "Bakery", "nl": "Bakkerij"}, "pencils": {"en": "Pencils", "nl": "Potloden"},
        "cinema": {"en": "Cinema", "nl": "Bioscoop"}}
_ASKED = {"plain": {"en": "as it is", "nl": "zoals hij is"}, "number": {"en": "only the number", "nl": "alleen het getal"}}


def refine_rows(R: dict = None) -> list:
    """Every first answer of experiment 2 and its check: (sum, way, seed, right before, right after)."""
    R = R or course()
    return [(q, way, r["seed"], read(r["first"]) == entry["answer"], read(r["checked"]["text"]) == entry["answer"])
            for q, entry in R["refine"].items() for way in REFINE_WAYS for r in entry[way]]


def refine_counts(R: dict = None, q: str = None, way: str = None) -> dict:
    """Of the first answers of experiment 2 (of one sum and way, or all): how many there were, how many were right
    before the check and after it, how many wrong ones the check fixed, and how many right ones it broke."""
    rows = [r for r in refine_rows(R) if (q is None or r[0] == q) and (way is None or r[1] == way)]
    return {"n": len(rows), "before": sum(r[3] for r in rows), "after": sum(r[4] for r in rows),
            "wrong": sum(not r[3] for r in rows), "fixed": sum(r[4] and not r[3] for r in rows), "broken": sum(r[3] and not r[4] for r in rows)}


def refine_question(lang: str) -> str:
    c = refine_counts()
    if lang == "en":
        return (f"The course gave Qwen3-0.6B {c['n']} of its own answers to three sums back, with one more message: *Check your answer step by "
                f"step. If it is wrong, fix it.* {c['wrong']} of those answers were wrong. How many of them did it fix?")
    return (f"De course gaf Qwen3-0.6B {c['n']} van zijn eigen antwoorden op drie sommen terug, met nog één bericht: *Check your answer step "
            f"by step. If it is wrong, fix it.* (controleer je antwoord stap voor stap, en verbeter het als het fout is). {c['wrong']} van die "
            f"antwoorden waren fout. Hoeveel daarvan verbeterde het?")


def refine_reveal(lang: str) -> str:
    """What the guess about checking its own answer shows."""
    R = course()
    c = refine_counts(R)
    rows = [r for r in refine_rows(R) if not r[3]]
    entries = {(q, way, r["seed"]): r for q, e in R["refine"].items() for way in REFINE_WAYS for r in e[way]}
    kept = sum(read(entries[r[:3]]["first"]) == read(entries[r[:3]]["checked"]["text"]) for r in rows if not r[4])
    assert {(r[0], r[1]) for r in rows} == {("bakery", "number")} and c["broken"] == 0 and kept == c["wrong"] - c["fixed"] > 0, "the text below no longer fits the course's runs"
    if lang == "en":
        return (f"**It fixed {c['fixed']} of the {c['wrong']}, and none of the {c['before']} right answers became wrong.** All {c['wrong']} wrong "
                f"answers were to the bakery, asked for only the number. In its check, Qwen3 worked the sum out again, step by step this time, "
                f"and in {c['fixed']} of them it came to {R['refine']['bakery']['answer']} euros. In the other {kept}, it came to the same wrong "
                f"number as before: a check by the same model can miss what the model missed the first time.")
    return (f"**Het verbeterde er {c['fixed']} van de {c['wrong']}, en geen van de {c['before']} goede antwoorden werd fout.** Alle {c['wrong']} foute "
            f"antwoorden hoorden bij de bakkerij, gevraagd om alleen het getal. In zijn controle rekende Qwen3 de som opnieuw uit, nu stap voor "
            f"stap, en bij {c['fixed']} ervan kwam het op {R['refine']['bakery']['answer']} euro. Bij de andere {kept} kwam het op hetzelfde foute "
            f"getal als eerst: een controle door hetzelfde model kan missen wat het model de eerste keer ook miste.")


def refine_after(lang: str) -> str:
    """What the table of experiment 2 shows."""
    c = refine_counts()
    assert c["broken"] == 0 and 0 < refine_counts(way="number")["fixed"] == c["fixed"] < c["wrong"], "the text below no longer fits the course's runs"
    if lang == "en":
        return (f"The check helped where the first answer was short: asked for only the number, Qwen3 had written few steps, and the check "
                f"asked for them. The {c['before']} answers that were already right stayed right. Self-Refine works "
                f"when the second prompt adds what the first one lacked, here the steps; but the model can also keep its own mistake, as it did "
                f"{_TIMES['en'].get(c['wrong'] - c['fixed'], str(c['wrong'] - c['fixed']) + ' times')} here.")
    return (f"De controle hielp waar het eerste antwoord kort was: gevraagd om alleen het getal had Qwen3 weinig stappen geschreven, en de "
            f"controle vroeg erom. De {c['before']} antwoorden die al goed waren, bleven goed. Self-Refine werkt als "
            f"de tweede prompt toevoegt wat de eerste miste, hier de stappen; maar het model kan ook zijn eigen fout houden, zoals het hier "
            f"{_TIMES['nl'].get(c['wrong'] - c['fixed'], str(c['wrong'] - c['fixed']) + ' keer')} deed.")


def refine_table(lang: str) -> str:
    """How many first answers were right before the check and after it, for each sum and way of asking."""
    R = course()
    head = {"en": ["Sum", "Asked", "Right before the check", "Right after it"], "nl": ["Som", "Gevraagd", "Goed vóór de controle", "Goed erna"]}[lang]
    lines = ["| " + " | ".join(head) + " |", "|---|---|---|---|"]
    for q in R["refine"]:
        for way in REFINE_WAYS:
            c = refine_counts(R, q, way)
            lines.append(f"| {SUMS[q][lang]} | {_ASKED[way][lang]} | {c['before']} of {c['n']} | {c['after']} of {c['n']} |".replace(" of ", " of " if lang == "en" else " van "))
    return "\n".join(lines)


# --- 3. a chain of verification ----------------------------------------------------------------------------------------
LOW, HIGH = 80, 100
PRIMES = {83, 89, 97}          # the only primes between 80 and 100, so a draft of five has to be wrong somewhere
CHECK_TOKENS = 500             # a check that works it out step by step may write this much


def draft_primes() -> list:
    return [{"role": "user", "content": f"Name five prime numbers between {LOW} and {HIGH}. Reply with the list only."}]


def verify(n: int) -> list:
    """A fresh check of one claim of the draft, as a plain question."""
    return [{"role": "user", "content": f"Is {n} a prime number between {LOW} and {HIGH}? Answer yes or no on the last line."}]


def verify_steps(n: int) -> list:
    """The same check, worked out step by step first."""
    return [{"role": "user", "content": f"Is {n} a prime number between {LOW} and {HIGH}? {STEPS} Then answer yes or no on the last line."}]


def listed(text: str) -> list:
    """The numbers a draft names, from 10 up and without the ends of the range it may repeat, from small to large."""
    return sorted({int(x) for x in re.findall(r"\d+", text) if int(x) >= 10 and int(x) not in (LOW, HIGH)})


def right_claim(n: int) -> bool:
    return n in PRIMES


def said_yes(text: str) -> bool:
    """Whether the last line of an answer says yes."""
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return bool(lines) and bool(re.search(r"\byes\b", lines[-1], re.I)) and not re.search(r"\bno\b", lines[-1], re.I)


# --- 4. answering from a source, with the sentence named (grounding and citation) ---------------------------------
SOURCE = ("[1] The library in Elm Street is open from Monday to Friday from 9:00 to 18:00.\n"
          "[2] On Saturday it opens at 10:00 and closes at 14:00.\n"
          "[3] It is closed on Sunday and on public holidays.\n"
          "[4] Books can be borrowed for three weeks.")
ASK_SATURDAY = "When does the library in Elm Street close on Saturday?"
ASK_MAIL = "Can I return books to the library in Elm Street by mail?"
NOT_SAID = "The text does not say."


def grounded(question: str) -> list:
    return [{"role": "user", "content": "Answer the question using only the text between the <source> tags, and end your answer "
             f"with the number of the sentence you used, like [1]. If the text does not answer the question, write: {NOT_SAID}\n\n"
             f"<source>\n{SOURCE}\n</source>\n\nQuestion: {question}"}]


def ungrounded(question: str) -> list:
    return [{"role": "user", "content": question}]


def says_two(text: str) -> bool:
    """Whether an answer gives 14:00 as the closing time: 14:00, 14.00, 2 pm or 2:00 pm."""
    return bool(re.search(r"\b14[:.]00\b|\b2(?::00)?\s*(?:p\.?m\.?)", text, re.I))


def read(text: str):
    found = re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))
    if not found:
        return None
    x = float(found[-1])
    return int(x) if x == int(x) else x


def course() -> dict:
    from everything_ai import paths
    return json.loads((paths.ROOT / COURSE).read_text(encoding="utf-8"))


def cited(text: str) -> list:
    """The numbers of the sentences an answer names, such as [2], in the order it names them first."""
    out = []
    for n in re.findall(r"\[(\d+)\]", text):
        if int(n) not in out:
            out.append(int(n))
    return out


def number(x: float, digits: int, lang: str) -> str:
    text_ = f"{x:,.{digits}f}"
    if lang == "nl":
        text_ = text_.replace(",", "_").replace(".", ",").replace("_", ".")
    return text_


# --- what the course's runs show ---------------------------------------------------------------------------------

CHAIN_SHOWN = VERIFY_SHOWN = SOURCE_SHOWN = 1    # the run each slider opens at


def _mark(ok: bool) -> str:
    colour = "var(--ea-pred-text)" if ok else "var(--ea-loss-text)"
    return f'<span style="color:{colour};font-weight:600">{"✓" if ok else "✗"}</span>'


def _chip(text: str, ok: bool) -> str:
    border, colour = ("var(--ea-pred)", "var(--ea-pred-text)") if ok else ("var(--ea-loss)", "var(--ea-loss-text)")
    return (f'<span style="display:inline-block;min-width:2.2rem;text-align:center;margin:0 0.3rem 0.3rem 0;padding:0.1rem 0.35rem;'
            f'border:1px solid {border};border-radius:4px;font-family:var(--ea-font-code);font-size:0.85rem;color:{colour}">{html.escape(text)}</span>')


def _box(label: str, text: str, note: str = "", height: str = "11rem") -> str:
    """One answer of the model under its label: the label, what the lesson read in it, and the text, which scrolls."""
    return (f'<div style="font-family:var(--ea-font-ui);font-size:0.85rem;font-weight:600;color:var(--ea-ink-2);margin:0.6rem 0 0.25rem">'
            f'{html.escape(label)}{" · " + note if note else ""}</div>'
            f'<div style="max-height:{height};overflow-y:auto;font-family:var(--ea-font-code);font-size:0.8rem;line-height:1.45;white-space:pre-wrap;'
            f'overflow-wrap:anywhere;color:var(--ea-pred-text);border-left:2px solid var(--ea-line);padding-left:0.6rem">{html.escape(text)}</div>')


def _shown(x, lang: str) -> str:
    if x is None:
        return "–"
    return number(x, 0 if x == int(x) else 2, lang)


def chain_frame(i: int, lang: str) -> tuple:
    """Run i of experiment 1: the answer to the one prompt, and the two answers of the chain, each with what the lesson
    read in it."""
    run = course()["chain"][i - 1]
    one, cost = read(run["one"]["text"]), read(run["second"]["text"])
    scaled = scaled_right(run["first"]["text"])
    w = {"en": {"one": "One prompt", "first": "The chain, prompt 1: the amounts", "second": "The chain, prompt 2: the cost of the amounts of prompt 1",
                "read": "the last number: {x} euros", "scaled": "the amounts for 12 people: right", "unscaled": "the amounts for 12 people: not right"},
         "nl": {"one": "Eén prompt", "first": "De keten, prompt 1: de hoeveelheden", "second": "De keten, prompt 2: de kosten van de hoeveelheden uit prompt 1",
                "read": "het laatste getal: {x} euro", "scaled": "de hoeveelheden voor 12 mensen: goed", "unscaled": "de hoeveelheden voor 12 mensen: niet goed"}}[lang]
    cut = {k: cut_off(run[k], lang) for k in ("one", "first", "second")}
    body = (_box(w["one"], run["one"]["text"], _mark(one == COST) + " " + html.escape(w["read"].format(x=_shown(one, lang)) + cut["one"]))
            + _box(w["first"], run["first"]["text"], _mark(scaled) + " " + html.escape(w["scaled" if scaled else "unscaled"] + cut["first"]))
            + _box(w["second"], run["second"]["text"], _mark(cost == COST) + " " + html.escape(w["read"].format(x=_shown(cost, lang)) + cut["second"])))
    if lang == "en":
        said = (f"**Run {i}:** one prompt ends in {_shown(one, lang)} euros{cut['one']}, {'right' if one == COST else 'wrong'}. In the chain, the "
                f"amounts are {'right' if scaled else 'not right'}{cut['first']}, and the cost ends in {_shown(cost, lang)} euros{cut['second']}, "
                f"{'right' if cost == COST else 'wrong'}.")
    else:
        said = (f"**Run {i}:** één prompt eindigt op {_shown(one, lang)} euro{cut['one']}, {'goed' if one == COST else 'fout'}. In de keten zijn de "
                f"hoeveelheden {'goed' if scaled else 'niet goed'}{cut['first']}, en eindigen de kosten op {_shown(cost, lang)} euro{cut['second']}, "
                f"{'goed' if cost == COST else 'fout'}.")
    return body, said


def cut_off(answer: dict, lang: str) -> str:
    """Words that say an answer stopped at its limit of tokens, before it was done; empty when it was done."""
    if answer["finish"] != "length":
        return ""
    limit = answer.get("max_tokens", MAX_TOKENS)
    return {"en": f" (cut off at {limit} tokens)", "nl": f" (afgebroken na {limit} tokens)"}[lang]


def chain_counts(R: dict = None) -> dict:
    """Of the runs of experiment 1: how often the one prompt was right, the chain's amounts, the chain's cost; and the
    chain's cost when its amounts were right, and when they were not."""
    R = R or course()
    rows = R["chain"]
    scaled = [scaled_right(r["first"]["text"]) for r in rows]
    cost = [read(r["second"]["text"]) == COST for r in rows]
    return {"n": len(rows), "one": sum(read(r["one"]["text"]) == COST for r in rows), "scaled": sum(scaled), "chain": sum(cost),
            "chain_when_scaled": sum(c for c, s in zip(cost, scaled) if s), "chain_when_not": sum(c for c, s in zip(cost, scaled) if not s)}


def flour(text: str):
    """The grams of flour among the amounts a text hands on: the number on its line with flour, or None."""
    for line in amounts_of(text).lower().splitlines():
        if "flour" in line:
            return read(re.sub(r"\s*(?:g|grams?)\b.*", "", line.split("flour", 1)[1]) or line)
    return None


_TIMES = {"en": {1: "once", 2: "twice", 3: "three times"}, "nl": {1: "één keer", 2: "twee keer", 3: "drie keer"}}


def chain_after(lang: str) -> str:
    """What the runs of experiment 1 show: where the one prompt went wrong, the chain, and where the chain went wrong."""
    R = course()
    c = chain_counts(R)
    ones = [read(r["one"]["text"]) for r in R["chain"]]
    four, twelve = ones.count(5), ones.count(60)      # the recipe for 4 people priced; the recipe times 12 priced
    rest = [x for x in ones if x not in (5, 60, COST)]
    rows = [r for r in R["chain"] if not scaled_right(r["first"]["text"])]
    times12 = sum(flour(r["first"]["text"]) == 2400 for r in rows)
    one_person = sum(flour(r["first"]["text"]) == 50 for r in rows)
    assert c["chain_when_not"] == 0 and c["chain_when_scaled"] == c["scaled"] and times12 + one_person == len(rows), "the text below no longer fits the course's runs"
    assert four and twelve and len(rest) >= 2 and all(r["one"]["finish"] == "stop" for r in R["chain"])
    assert all(not scaled_right(r["first"]["text"]) for r in R["chain"] if "length" in (r["first"]["finish"], r["second"]["finish"]))
    t = _TIMES[lang]
    if lang == "en":
        return (f"**One prompt was right in {c['one']} of {c['n']} runs; the chain in {c['chain']}.** The one prompt slipped in different places: "
                f"{t[four]} it priced the recipe for 4 people, 5 euros, as if the 12 people were not there; {t[twelve]} it multiplied the amounts "
                f"by 12 instead of by 3, 60 euros; and the other wrong runs ended at {_list([_shown(x, lang) for x in rest], lang)} euros. To find "
                f"where, you have to read the whole answer. The chain was right in exactly the runs where prompt 1 got the amounts for 12 people "
                f"right. In {times12} of the others, prompt 1 multiplied the recipe by 12 instead of by 3, so 2,400 g of flour, and in "
                f"{one_person} it worked out the amounts for one person; prompt 2 then priced amounts that were already wrong. That mistake was "
                f"there to see in the answer of prompt 1, before prompt 2 ran: that is what a chain gives you. Each prompt asks less, and between "
                f"the two you, or your program, can look.")
    return (f"**Eén prompt was goed in {c['one']} van de {c['n']} runs; de keten in {c['chain']}.** Met één prompt ging het telkens ergens anders "
            f"mis: {t[four]} gaf het model de prijs van het recept voor 4 mensen, 5 euro, alsof die 12 mensen er niet waren; {t[twelve]} "
            f"vermenigvuldigde het de hoeveelheden met 12 in plaats van met 3, 60 euro; en de andere foute runs kwamen uit op "
            f"{_list([_shown(x, lang) for x in rest], lang)} euro. Om te zien waar het misging, moet je het hele antwoord lezen. De keten was goed "
            f"in precies de runs waarin prompt 1 de hoeveelheden voor 12 mensen goed had. In {times12} van de andere vermenigvuldigde prompt 1 "
            f"het recept met 12 in plaats van met 3, dus 2.400 g bloem, en in {one_person} rekende het de hoeveelheden voor één persoon uit; "
            f"prompt 2 rekende daarna de prijs uit van hoeveelheden die al fout waren. Die fout was te zien in het antwoord van prompt 1, vóór "
            f"prompt 2 liep: dat geeft een keten je. Elke prompt vraagt minder, en tussen de twee kun jij, of je programma, kijken.")


def kept(run: dict, how: str) -> list:
    """The numbers of a draft that got a yes from their check: asked plainly ("checks") or worked out ("steps")."""
    return [n for n in listed(run["draft"]["text"]) if said_yes(run[how][str(n)]["text"])]


def verify_frame(i: int, lang: str) -> tuple:
    """Run i of experiment 3: the draft, the numbers it names, both checks of each, and the two final lists."""
    run = course()["verify"][i - 1]
    named = listed(run["draft"]["text"])
    w = {"en": {"draft": "The draft", "named": f"The numbers it names (✓ a prime between {LOW} and {HIGH}, ✗ not)",
                "checks": "Each number in a fresh prompt: asked plainly, and worked out first", "plain": "plainly", "steps": "worked out",
                "kept_plain": "The final list of the plain checks", "kept_steps": "The final list of the checks worked out", "yes": "yes", "no": "no"},
         "nl": {"draft": "De eerste versie", "named": f"De getallen die het noemt (✓ een priemgetal tussen {LOW} en {HIGH}, ✗ niet)",
                "checks": "Elk getal in een nieuwe prompt: gewoon gevraagd, en eerst uitgerekend", "plain": "gewoon", "steps": "uitgerekend",
                "kept_plain": "De eindlijst van de gewone controles", "kept_steps": "De eindlijst van de uitgerekende controles", "yes": "ja", "no": "nee"}}[lang]
    label = 'style="font-family:var(--ea-font-ui);font-size:0.85rem;font-weight:600;color:var(--ea-ink-2);margin:0.6rem 0 0.25rem"'

    def check(how, n, height):
        answer = run[how][str(n)]
        yes = said_yes(answer["text"])
        head = (f'{_mark(yes == right_claim(n))} {w["steps" if how == "steps" else "plain"]}: {w["yes"] if yes else w["no"]}'
                f'{html.escape(cut_off(answer, lang))}')
        return (f'<div style="margin-top:0.2rem">{head}</div>'
                f'<div style="font-family:var(--ea-font-code);font-size:0.78rem;color:var(--ea-pred-text);white-space:pre-wrap;overflow-wrap:anywhere;'
                f'max-height:{height};overflow-y:auto;border-left:2px solid var(--ea-line);padding-left:0.5rem">{html.escape(answer["text"])}</div>')
    rows = "".join(
        f'<div style="padding:0.3rem 0;border-top:1px solid var(--ea-line);font-size:0.85rem">'
        f'<div style="font-family:var(--ea-font-code);font-weight:600">{n}</div>{check("checks", n, "2.6rem")}{check("steps", n, "4.5rem")}</div>'
        for n in named)
    plain, steps = kept(run, "checks"), kept(run, "steps")
    chips = lambda ns: "".join(_chip(str(n), right_claim(n)) for n in ns) or "–"
    body = (_box(w["draft"], run["draft"]["text"], height="6rem")
            + f'<div {label}>{html.escape(w["named"])}</div><div>{chips(named)}</div>'
            + f'<div {label}>{html.escape(w["checks"])}</div>{rows}'
            + f'<div {label}>{html.escape(w["kept_plain"])}</div><div>{chips(plain)}</div>'
            + f'<div {label}>{html.escape(w["kept_steps"])}</div><div>{chips(steps)}</div>')
    wrong = [n for n in named if not right_claim(n)]
    if lang == "en":
        bad = f"of which {_list(wrong, lang)} {'is' if len(wrong) == 1 else 'are'} wrong" if wrong else "and none of them is wrong"
        said = (f"**Run {i}:** the draft names {_list(named, lang)}, {bad}. The plain checks keep {_list(plain, lang)}; the checks worked out "
                f"keep {_list(steps, lang)}.")
    else:
        bad = f"waarvan {_list(wrong, lang)} fout {'is' if len(wrong) == 1 else 'zijn'}" if wrong else "en geen ervan is fout"
        said = (f"**Run {i}:** de eerste versie noemt {_list(named, lang)}, {bad}. De gewone controles houden {_list(plain, lang)}; de "
                f"uitgerekende controles houden {_list(steps, lang)}.")
    return body, said


def _list(numbers: list, lang: str) -> str:
    if not numbers:
        return {"en": "no number", "nl": "geen getal"}[lang]
    if len(numbers) == 1:
        return str(numbers[0])
    return ", ".join(map(str, numbers[:-1])) + {"en": " and ", "nl": " en "}[lang] + str(numbers[-1])


def verify_counts(R: dict = None) -> dict:
    """Of the runs of experiment 3: the wrong numbers in the drafts, how many of them each kind of check threw out, how
    many right numbers it threw out, the final lists without a wrong number, and the drafts that left out a prime."""
    R = R or course()
    out = {"n": 0, "wrong": 0, "right": 0, "plain_caught": 0, "steps_caught": 0, "plain_lost": 0, "steps_lost": 0,
           "plain_clean": 0, "steps_clean": 0, "missed": 0}
    for run in R["verify"]:
        named = listed(run["draft"]["text"])
        plain, steps = set(kept(run, "checks")), set(kept(run, "steps"))
        wrong = [n for n in named if not right_claim(n)]
        out["n"] += 1
        out["wrong"] += len(wrong)
        out["right"] += len(named) - len(wrong)
        out["plain_caught"] += sum(n not in plain for n in wrong)
        out["steps_caught"] += sum(n not in steps for n in wrong)
        out["plain_lost"] += sum(right_claim(n) and n not in plain for n in named)
        out["steps_lost"] += sum(right_claim(n) and n not in steps for n in named)
        out["plain_clean"] += not any(n in plain for n in wrong)
        out["steps_clean"] += not any(n in steps for n in wrong)
        out["missed"] += bool(PRIMES - set(named))
    return out


def verify_after(lang: str) -> str:
    """What the runs of experiment 3 show: the plain checks, the checks worked out, and what no check can do."""
    R = course()
    c = verify_counts(R)
    wrong = [(run, n) for run in R["verify"] for n in listed(run["draft"]["text"]) if not right_claim(n)]
    caught = [(run, n) for run, n in wrong if n not in kept(run, "steps")]
    cut = sum(run["steps"][str(n)]["finish"] == "length" for run, n in caught)
    said_no = len(caught) - cut
    assert c["plain_caught"] == c["plain_lost"] == c["steps_lost"] == 0 and {n for _, n in caught} == {101} and said_no and cut, "the text below no longer fits the course's runs"
    assert {n for _, n in wrong} == {101, 103, 107} and all(n > HIGH for _, n in wrong)
    assert all(re.search(r"greater than 100|not\*\* between|between 100 and 101", run["steps"][str(n)]["text"]) for run, n in caught if run["steps"][str(n)]["finish"] == "stop")
    t = _TIMES[lang]
    total = c["wrong"] + c["right"]
    if lang == "en":
        return (f"**The plain checks said yes to all {total} numbers, the {c['wrong']} wrong ones too.** A small model answers a bare yes-or-no "
                f"question with yes far too easily. Worked out first, the checks threw out {len(caught)} of the {c['wrong']} wrong numbers, all of "
                f"them 101, and none of the right ones. {t[said_no].capitalize()} the steps saw that 101 is greater than 100, and said no; {t[cut]} "
                f"the check used up its {CHECK_TOKENS} tokens, still dividing 101, before it said yes or no. The other wrong numbers kept their yes: "
                f"the steps showed, rightly, that 101, 103 and 107 are prime, and then took them to lie between {LOW} and {HIGH}. And no check could "
                f"add 83, which {c['missed']} of the {c['n']} drafts left out: a check throws claims out, it does not add the ones that are missing. "
                f"So a check catches a mistake only in what it really looks at: read the steps of a check, not just its last line.")
    return (f"**De gewone controles zeiden ja tegen alle {total} getallen, ook tegen de {c['wrong']} foute.** Een klein model beantwoordt een kale "
            f"ja-of-neevraag veel te makkelijk met ja. Met eerst uitrekenen gooiden de controles {len(caught)} van de {c['wrong']} foute getallen "
            f"eruit, allemaal 101, en geen van de goede. {t[said_no].capitalize()} zagen de stappen dat 101 groter is dan 100, en zeiden ze nee; "
            f"{t[cut]} had de controle haar {CHECK_TOKENS} tokens op terwijl ze 101 nog aan het delen was, voordat ze ja of nee zei. De andere "
            f"foute getallen hielden hun ja: de stappen lieten terecht zien dat 101, 103 en 107 priem zijn, en namen dan aan dat ze tussen {LOW} "
            f"en {HIGH} liggen. En geen controle kon 83 toevoegen, dat {c['missed']} van de {c['n']} eerste versies weglieten: een controle gooit "
            f"beweringen eruit, maar voegt geen ontbrekende toe. Een controle vangt dus alleen een fout in wat ze echt bekijkt: lees de stappen "
            f"van een controle, niet alleen haar laatste regel.")


def last_cited(text: str):
    """The number of the last sentence an answer names, such as 2 for [2], or None: where the prompt asks for it, at
    the end."""
    found = re.findall(r"\[(\d+)\]", text)
    return int(found[-1]) if found else None


def no_access(text: str) -> bool:
    """Whether an answer says the model has no access to the information."""
    return bool(re.search(r"(?:don't|do not) have access", text, re.I))


def source_frame(i: int, lang: str) -> tuple:
    """Run i of experiment 4: the answer without the source, with it, and to the question it does not answer."""
    run = course()["source"][i - 1]
    without, with_, mail = run["without"]["text"], run["with"]["text"], run["not_said"]["text"]
    w = {"en": {"without": "Without the source", "with": "With the source", "not_said": "With the source, a question it does not answer",
                "two": "says 14:00", "other": "does not say 14:00", "ends": "ends with {c}", "none": "names no sentence",
                "says": "says the text does not say", "not": "does not say that"},
         "nl": {"without": "Zonder de bron", "with": "Met de bron", "not_said": "Met de bron, een vraag waar die geen antwoord op geeft",
                "two": "zegt 14.00 uur", "other": "zegt geen 14.00 uur", "ends": "eindigt met {c}", "none": "noemt geen zin",
                "says": "zegt dat de tekst het niet zegt", "not": "zegt dat niet"}}[lang]

    def ends(text):
        c = last_cited(text)
        return w["ends"].format(c=f"[{c}]") if c else w["none"]
    body = (_box(w["without"], without, _mark(says_two(without)) + " " + html.escape(w["two"] if says_two(without) else w["other"]), height="8rem")
            + _box(w["with"], with_, _mark(says_two(with_)) + " " + html.escape(w["two"] if says_two(with_) else w["other"]) + " · "
                   + _mark(last_cited(with_) == 2) + " " + html.escape(ends(with_)), height="8rem")
            + _box(w["not_said"], mail, _mark(not_said(mail)) + " " + html.escape(w["says"] if not_said(mail) else w["not"]) + " · "
                   + _mark(last_cited(mail) is None) + " " + html.escape(ends(mail)), height="8rem"))
    if lang == "en":
        said = (f"**Run {i}:** without the source the answer {w['two'] if says_two(without) else w['other']}; with it, it "
                f"{w['two'] if says_two(with_) else w['other']} and {ends(with_)}; to the question about mail, it "
                f"{w['says'] if not_said(mail) else w['not']}, and {ends(mail)}.")
    else:
        said = (f"**Run {i}:** zonder de bron noemt het antwoord {'' if says_two(without) else 'geen '}14.00 uur; met de bron noemt het "
                f"{'' if says_two(with_) else 'geen '}14.00 uur, en {ends(with_).replace('eindigt', 'eindigt het').replace('noemt', 'noemt het')}; "
                f"op de vraag over de post {'zegt het dat de tekst het niet zegt' if not_said(mail) else 'zegt het dat niet'}, en "
                f"{ends(mail).replace('eindigt', 'eindigt het').replace('noemt', 'noemt het')}.")
    return body, said


def not_said(text: str) -> bool:
    """Whether an answer says that the source does not answer the question."""
    return bool(re.search(r"does not say|doesn't say|does not (?:mention|contain|provide|specify)|no information|not (?:mentioned|stated|specified)", text, re.I))


def source_counts(R: dict = None) -> dict:
    """Of the runs of experiment 4: without the source, the answers with 14:00 and those that say the model has no
    access; with it, the answers with 14:00 and those that end with [2] or [1]; and the answers to the question about
    mail that say the text does not say, and those that name a sentence all the same."""
    R = R or course()
    rows = R["source"]
    return {"n": len(rows), "without": sum(says_two(r["without"]["text"]) for r in rows), "no_access": sum(no_access(r["without"]["text"]) for r in rows),
            "with": sum(says_two(r["with"]["text"]) for r in rows), "end2": sum(last_cited(r["with"]["text"]) == 2 for r in rows),
            "end1": sum(last_cited(r["with"]["text"]) == 1 for r in rows), "not_said": sum(not_said(r["not_said"]["text"]) for r in rows),
            "not_said_named": sum(not_said(r["not_said"]["text"]) and last_cited(r["not_said"]["text"]) is not None for r in rows)}


def source_after(lang: str) -> str:
    """What the runs of experiment 4 show."""
    R = course()
    c = source_counts(R)
    made_up = c["n"] - c["no_access"] - c["without"]
    assert c["without"] == 0 and c["with"] == c["n"] and c["end2"] + c["end1"] == c["n"] and c["not_said"] == c["not_said_named"] == c["n"], "the text below no longer fits the course's runs"
    assert all(last_cited(r["not_said"]["text"]) == 1 for r in R["source"]) and any("5:00 PM" in r["without"]["text"] for r in R["source"])
    assert all(r["with"]["text"].startswith("[2]") for r in R["source"]) and "like [1]" in grounded(ASK_SATURDAY)[0]["content"]
    assert all(re.search(r"\d:\d\d", r["without"]["text"]) for r in R["source"] if not no_access(r["without"]["text"]))
    if lang == "en":
        return (f"**Without the source, Qwen3 never said 14:00: {c['no_access']} times it said it had no access to such information, and "
                f"{made_up} times it made up a closing time, such as 5:00 PM.** With the source, it said 14:00 every time: it copied sentence [2], "
                f"number and all. At the end it named the sentence it used, as asked, but that was [2] in only {c['end2']} of the {c['n']} runs, "
                f"and [1] in the others. To the question about mail, it wrote *{NOT_SAID}* every time, as asked, and then put [1] after it, a "
                f"sentence that says nothing about mail. [1] is the example in the prompt, *like [1]*: as with Tom's apples in "
                f"[Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en), Qwen3 copied the example. So grounding gave the right answer, and the citation let you check it in a moment, "
                f"and see where the citation itself was wrong.")
    return (f"**Zonder de bron noemde Qwen3 nooit 14.00 uur: {c['no_access']} keer zei het dat het geen toegang heeft tot zulke informatie, en {made_up} keer "
            f"verzon het een sluitingstijd, zoals 5:00 PM (17.00 uur).** Met de bron noemde het elke keer 14.00 uur: het nam zin [2] over, met "
            f"nummer en al. Aan het eind noemde het de zin die het gebruikte, zoals gevraagd, maar dat was maar in {c['end2']} van de {c['n']} runs "
            f"[2], en in de andere [1]. Op de vraag over de post schreef het elke keer *{NOT_SAID}* (de tekst zegt het niet), zoals gevraagd, en "
            f"zette het er [1] achter, een zin die niets over de post zegt. [1] is het voorbeeld uit de prompt, *like [1]*: net als met de appels van Tom in "
            f"[Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl) nam Qwen3 het voorbeeld over. Grounding gaf dus het goede antwoord, en met de citatie kon je het in "
            f"een oogopslag nakijken, en zien waar de citatie zelf niet klopte.")


# --- the demo: the learner's own prompt ---------------------------------------------------------------------------

def downloaded() -> bool:
    from everything_ai import chat
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    from everything_ai import chat
    chat.download_or_stop(button, texts)


def ask_live(prompt: str, show, words: dict, lang: str) -> str:
    """The demo: the learner's prompt as one user message to Qwen3, sampled with seed 0, with what it writes so far
    shown by `show(html)` at most five times a second. The picture of the answer and the sentences it names, or the
    message of what went wrong."""
    from everything_ai import chat
    a = chat.answer_live([{"role": "user", "content": prompt}],
                         lambda text: show(_answer_picture(text or "", words["loading"] if text is None else words["writing"], words)),
                         max_tokens=MAX_TOKENS, seed=0)
    if isinstance(a, str):
        return f'<p><em>{html.escape(words[a])}</em></p>'
    named = cited(a["message"]["content"])
    note = words["done"].format(tokens=a["tokens"], seconds=number(a["seconds"], 1, lang),
                                cited=", ".join(f"[{n}]" for n in named) if named else words["nothing"])
    return _answer_picture(a["message"]["content"], note, words)


def _answer_picture(text: str, note: str, words: dict) -> str:
    from everything_ai import yours
    body = (f'<div style="font-family:var(--ea-font-code);font-size:13px;line-height:1.45;white-space:pre-wrap;overflow-wrap:anywhere;'
            f'color:var(--ea-pred-text)">{html.escape(text)}</div>'
            f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2);margin-top:0.3rem">{html.escape(note)}</div>')
    return yours.figure(words["title"], body, alt=words["title"])


# The lesson is done when these are correct. "cite" and "fresh" are choice questions in the lesson.
REQUIRED = ["habits", "cite", "fresh"]

_STEPS = {"en": "Are there several prompts here, each doing one step of the task?",
          "nl": "Zijn hier meerdere prompts, die elk één stap van de taak doen?"}
_OWN = {"en": "Who looks at the answer here: the same model, at what it wrote itself?",
        "nl": "Wie kijkt hier naar het antwoord: hetzelfde model, naar wat het zelf schreef?"}

EXERCISES = {
    "habits": Sort(
        name="habits",
        groups={
            "chaining": {"en": "Prompt chaining", "nl": "Prompt chaining"},
            "refine": {"en": "Self-Refine", "nl": "Self-Refine"},
            "verify": {"en": "Chain-of-Verification", "nl": "Chain-of-Verification"},
            "grounding": {"en": "Grounding, with a citation", "nl": "Grounding, met een citatie"},
        },
        items={
            "outline": {"en": "First ask for an outline of a talk; then, in a new prompt, for the talk from that outline.",
                        "nl": "Vraag eerst om een opzet voor een praatje; daarna, in een nieuwe prompt, om het praatje bij die opzet."},
            "again": {"en": "After its answer, ask: *Read your answer again. Is every step right? If not, fix it.*",
                      "nl": "Vraag na het antwoord: *Lees je antwoord nog eens. Klopt elke stap? Zo niet, verbeter het.*"},
            "years": {"en": "A draft names five years. Ask about each in a new chat, without the draft, such as *In which year did the Berlin Wall fall?*, and keep only the years that match.",
                      "nl": "Een eerste versie noemt vijf jaartallen. Vraag elk ervan na in een nieuwe chat, zonder de eerste versie, zoals *In welk jaar viel de Berlijnse Muur?*, en houd alleen de jaartallen die kloppen."},
            "manual": {"en": "Paste the manual of a washing machine, and ask: *Using only this text, what does error E21 mean? Name the section.*",
                       "nl": "Plak de handleiding van een wasmachine erin, en vraag: *Wat betekent fout E21, alleen volgens deze tekst? Noem de paragraaf.*"},
            "calendar": {"en": "First let the model list the dates in an email; then, in a second prompt, turn that list into entries for a calendar.",
                         "nl": "Laat het model eerst de datums uit een e-mail op een rij zetten; maak daarna, in een tweede prompt, van die lijst afspraken voor je agenda."},
            "judge": {"en": "Let the model judge its own summary for what is missing, and then write a better one.",
                      "nl": "Laat het model zijn eigen samenvatting beoordelen op wat er ontbreekt, en dan een betere schrijven."},
        },
        solution={"outline": "chaining", "again": "refine", "years": "verify", "manual": "grounding", "calendar": "chaining", "judge": "refine"},
        nudge={"outline": _STEPS, "calendar": _STEPS, "again": _OWN, "judge": _OWN,
               "years": {"en": "Is each year asked about on its own, without the draft in view?",
                         "nl": "Wordt elk jaartal op zichzelf nagevraagd, zonder de eerste versie erbij?"},
               "manual": {"en": "Where does the answer come from here: from what the model remembers, or from a text it was given?",
                          "nl": "Waar komt het antwoord hier vandaan: uit wat het model onthouden heeft, of uit een tekst die het kreeg?"}},
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the six ways again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de zes manieren opnieuw uit."},
            "empty": {"en": "Pick a group for each of the six.",
                      "nl": "Kies voor elk van de zes een groep."},
            "unfinished": {"en": "Everything you picked so far is right. Pick a group for the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe koos, klopt. Kies ook voor de rest een groep (nog {n})."},
            "wrong": {"en": "“{item}” is in the wrong group.",
                      "nl": "‘{item}’ staat in de verkeerde groep."},
            "correct": {"en": "All {n} are right. Two chains, two times the model looking at its own work, one fresh check per claim, and one answer from a given text: each habit gives you a place to catch a mistake.",
                        "nl": "Alle {n} kloppen. Twee ketens, twee keer kijkt het model naar zijn eigen werk, één losse controle per bewering, en één antwoord uit een gegeven tekst: elke gewoonte geeft je een plek om een fout te vangen."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner chose, state, known mistake).
_RIGHT = {"outline": "chaining", "again": "refine", "years": "verify", "manual": "grounding", "calendar": "chaining", "judge": "refine"}
CASES = [
    ("habits", {}, "empty", None),
    ("habits", _RIGHT, "correct", None),
    ("habits", {"outline": "chaining", "manual": "grounding"}, "wrong", "unfinished"),
    ("habits", {**_RIGHT, "years": "refine"}, "wrong", "wrong:years"),
    ("habits", {**_RIGHT, "judge": "verify"}, "wrong", "wrong:judge"),
    ("habits", {**_RIGHT, "calendar": "refine", "again": "chaining"}, "wrong", "wrong:again"),
]
