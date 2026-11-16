# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-grounded-answers: the sources an answer names in square brackets, and whether every number in it is
in the sources it names; the course's recorded runs (data/course-rag-qwen.json): the 28 questions about the course,
each with the three points search by meaning finds, asked with the plain system prompt of build-rag-first and with a
grounded one; their pictures, and the demo with Qwen3 and the learner's own system prompt."""
import functools
import html
import json
import re

from everything_ai import chat, paths, summaries, yours
from everything_ai.check import MISSING, FunctionBox, Mistake, run_code, with_time_limit

DOWNLOADS = {"Qwen3-0.6B": chat.FILES}   # what the demo downloads, for the line under the header

POINTS = summaries.points("en")   # the knowledge base: every point of every summary, in English
VECTORS = summaries.vectors("points", "en")
QUESTIONS = summaries.questions()
ANSWERABLE = [q for q in QUESTIONS if q["kind"] != "none"]
NONE = [q for q in QUESTIONS if q["kind"] == "none"]
ABOUT = summaries.about()
K = 3
SEEDS = 2
SYSTEM = "You answer questions about the course Everything AI. Answer in one sentence."   # the system prompt of build-rag-first
GROUNDED = ("You answer questions about the course Everything AI. Answer in one sentence, using only the numbered sources. "
            "End the sentence with the number of the source it comes from, in square brackets, such as [2]. "
            "If the sources do not answer the question, answer only: I don't know.")
with_sources = run_code('''def with_sources(question, passages):
    if len(passages) == 0:
        return question
    text = "Use these sources to answer the question.\\n\\n"
    n = 1
    for passage in passages:
        text = text + f"[{n}] {passage}\\n"
        n = n + 1
    return text + "\\nQuestion: " + question
''').names["with_sources"]   # the solution of build-rag-idea


def found(q: dict) -> list:
    """The K points search by meaning finds for a question, with the course's embeddings of the question."""
    return summaries.meaning_search(summaries.vectors("questions", "en")[QUESTIONS.index(q)], VECTORS, POINTS, K)


def _messages(q: dict, way: str) -> list:
    return [{"role": "system", "content": SYSTEM if way == "plain" else GROUNDED},
            {"role": "user", "content": with_sources(q["text"]["en"], found(q))}]


RECORD = {f'{q["id"]}-{way}': {"messages": _messages(q, way), "max_tokens": 80, "seeds": SEEDS}
          for q in QUESTIONS for way in ("plain", "grounded")}
DATA = paths.ROOT / "data" / "course-rag-qwen.json"


@functools.cache
def course() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-grounded-answers"]


def answers(qid: str, way: str) -> list:
    return [a["text"] for a in course()[f"{qid}-{way}"]["answers"]]


def question(qid: str) -> dict:
    return next(q for q in QUESTIONS if q["id"] == qid)


def hit(q: dict) -> bool:
    """Whether search by meaning puts the point that answers the question among the K it finds."""
    return q["gold"] is not None and summaries.text_of(q["gold"]) in found(q)


def lesson_of(text: str) -> dict:
    return ABOUT[POINTS.index(text)]


def refuses(answer: str) -> bool:
    """Whether an answer says what the grounded prompt asks for when the sources do not answer: I don't know."""
    return "i don't know" in answer.lower()


def good(answer: str, q: dict) -> bool:
    """A right answer to a question a point answers; to a question no point answers, one that says it cannot answer."""
    if q["kind"] == "none":
        return refuses(answer) or summaries.says_unknown(answer)
    return summaries.right(answer, q)


def number(x: float, lang: str, digits: int = 1) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


@functools.cache
def findings() -> dict:
    """For the 24 questions a point answers, two runs each: how many answers were right with the plain prompt and
    with the grounded one, how many named a source, how many grounded answers said I don't know; for the 4 questions
    no point answers, how many answers said they cannot answer; and how the course's supported judged the grounded
    answers to the 24, against whether they were right."""
    f = {"questions": len(ANSWERABLE), "runs": SEEDS * len(ANSWERABLE), "none": len(NONE), "none_runs": SEEDS * len(NONE),
         "hits": sum(hit(q) for q in ANSWERABLE)}
    for way in ("plain", "grounded"):
        f[way] = sum(good(a, q) for q in ANSWERABLE for a in answers(q["id"], way))
        f[way + "_none"] = sum(good(a, q) for q in NONE for a in answers(q["id"], way))
        f[way + "_cites"] = sum(len(summaries.cited(a)) > 0 for q in ANSWERABLE for a in answers(q["id"], way))
    f["refused"] = sum(refuses(a) for q in ANSWERABLE for a in answers(q["id"], "grounded"))
    f["none_cites"] = sum(len(summaries.cited(a)) > 0 for q in NONE for a in answers(q["id"], "grounded"))
    for q in ANSWERABLE:
        for a in answers(q["id"], "grounded"):
            key = ("right" if summaries.right(a, q) else "other") + ("_supported" if summaries.supported(a, found(q)) else "_unsupported")
            f[key] = f.get(key, 0) + 1
    return f


def worked() -> dict:
    """The worked example: the question about Sam, its three sources, an answer with the plain prompt and one with the
    grounded prompt, what cited and numbers find in the grounded one, and the numbers of the source it names."""
    q = question("q05")
    passages = found(q)
    grounded = answers("q05", "grounded")[0]
    n = summaries.cited(grounded)
    return {"question": q["text"]["en"], "passages": passages, "plain": answers("q05", "plain")[0], "grounded": grounded,
            "cited": n, "numbers": sorted(summaries.numbers(grounded)), "source": passages[n[0] - 1],
            "source_numbers": sorted(summaries.numbers(passages[n[0] - 1])), "supported": summaries.supported(grounded, passages),
            "lesson": lesson_of(passages[n[0] - 1])["lesson"]}


_LABEL = "font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)"
_CODE = "font-family:var(--ea-font-code);font-size:12px"


def _swatch(color: str, label: str) -> str:
    return (f'<span style="display:inline-flex;align-items:center;gap:0.25rem;margin-right:0.8rem;{_LABEL}">'
            f'<span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:var(--ea-{color})"></span>{html.escape(label)}</span>')


def _square(ok: bool, text: str) -> str:
    return (f'<span title="{html.escape(" ".join(text.split())[:160])}" style="display:inline-block;width:12px;height:12px;'
            f'margin-right:2px;border-radius:2px;background:var(--ea-{"pred" if ok else "loss"})"></span>')


_RUNS_WORDS = {
    "en": {"title": "Qwen3's answers with the same three sources: with the plain prompt, and with the grounded one. Two runs each",
           "question": "question", "found": "right point", "plain": "plain", "grounded": "grounded", "yes": "found", "no": "not found",
           "good": "right, or says it cannot answer where no point answers", "bad": "not right", "count": "{k} of {n}",
           "none": "Questions no point answers", "meaning": "asks it in other words"},
    "nl": {"title": "De antwoorden van Qwen3 met dezelfde drie bronnen: met de gewone prompt, en met de grounded prompt. Elk twee runs",
           "question": "vraag", "found": "goede punt", "plain": "gewoon", "grounded": "grounded", "yes": "gevonden", "no": "niet gevonden",
           "good": "goed, of zegt dat het niet kan antwoorden waar geen punt antwoordt", "bad": "niet goed", "count": "{k} van de {n}",
           "none": "Vragen die geen punt beantwoordt", "meaning": "vraagt het in andere woorden"},
}


def runs_picture(lang: str) -> str:
    """A row per question: whether search by meaning found the point that answers it, as input data, then a square per
    run with the plain prompt and with the grounded one, in the color of model output when the answer is good and as a
    mistake when not; the questions no point answers apart, where a good answer says it cannot answer."""
    w = _RUNS_WORDS[lang]
    f = findings()

    def row(q: dict) -> str:
        mark = f' <span style="{_LABEL}">({html.escape(w["meaning"])})</span>' if q["kind"] == "meaning" else ""
        if q["kind"] == "none":
            found_cell = f'<span style="{_LABEL}">–</span>'
        else:
            yes = hit(q)
            found_cell = (f'<span style="{_LABEL}"><span style="display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:middle;'
                          f'{"background:var(--ea-data)" if yes else "border:1px dashed var(--ea-line)"}"></span> {html.escape(w["yes"] if yes else w["no"])}</span>')
        cells = "".join(f'<span>{"".join(_square(good(a, q), a) for a in answers(q["id"], way))}</span>' for way in ("plain", "grounded"))
        return f'<span style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink)">{html.escape(q["text"][lang])}{mark}</span>{found_cell}{cells}'

    def foot(k_plain: int, k_grounded: int, n: int) -> str:
        return (f'<span></span><span></span><span style="{_LABEL}">{html.escape(w["count"].format(k=k_plain, n=n))}</span>'
                f'<span style="{_LABEL}">{html.escape(w["count"].format(k=k_grounded, n=n))}</span>')
    head = "".join(f'<span style="{_LABEL}">{html.escape(w[k])}</span>' for k in ("question", "found", "plain", "grounded"))
    gap = f'<span style="grid-column:1 / -1;{_LABEL};margin-top:0.4rem;border-top:1px solid var(--ea-line);padding-top:0.3rem">{html.escape(w["none"])}</span>'
    grid = (f'<div style="display:grid;grid-template-columns:minmax(0,1fr) auto auto auto;gap:0.3rem 0.8rem;align-items:center">'
            f'{head}{"".join(row(q) for q in ANSWERABLE)}{foot(f["plain"], f["grounded"], f["runs"])}'
            f'{gap}{"".join(row(q) for q in NONE)}{foot(f["plain_none"], f["grounded_none"], f["none_runs"])}</div>')
    legend = _swatch("data", w["yes"]) + _swatch("pred", w["good"]) + _swatch("loss", w["bad"])
    return yours.figure(w["title"], grid + f'<div style="margin-top:0.4rem">{legend}</div>', alt=w["title"])


# --- exercise 1: cited -----------------------------------------------------------------------------------------------------

START_CITED = '''def cited(answer):
    found = []
    for part in answer.split("[")[1:]:
        inside = part.split("]")[0]
        found.append(inside)
    return found
'''
SOLUTION_CITED = START_CITED.replace('''        found.append(inside)
''', '''        if inside.isdigit() and int(inside) not in found:
            found.append(int(inside))
''')
AS_TEXT = START_CITED.replace('''        found.append(inside)
''', '''        if inside.isdigit() and inside not in found:
            found.append(inside)
''')
TWICE = START_CITED.replace('''        found.append(inside)
''', '''        if inside.isdigit():
            found.append(int(inside))
''')
NO_DIGITS = START_CITED.replace('''        found.append(inside)
''', '''        if int(inside) not in found:
            found.append(int(inside))
''')
CITED_EXAMPLE = answers("q05", "grounded")[0]
EXAMPLES_CITED = [("q05", 0), ("q17", 0), ("q18", 1), ("q10", 0), ("q25", 0), ("q28", 0)]   # grounded answers: (question, seed)

_CITED_WORDS = {
    "en": {"title": "What your cited finds in six of Qwen3's grounded answers", "yours": "your cited:", "course": "should be {want}",
           "none": "Your function gave back {got}, not a list."},
    "nl": {"title": "Wat je cited vindt in zes grounded antwoorden van Qwen3", "yours": "je cited:", "course": "zou {want} moeten zijn",
           "none": "Je functie gaf {got} terug, geen lijst."},
}


def _marked(answer: str) -> str:
    """An answer in code font, with every part in square brackets marked as input data."""
    out, last = [], 0
    for m in re.finditer(r"\[[^\[\]]*\]", answer):
        out.append(html.escape(answer[last:m.start()]))
        out.append(f'<span style="background:var(--ea-data-wash);border:1px solid var(--ea-data);border-radius:2px">{html.escape(m.group())}</span>')
        last = m.end()
    out.append(html.escape(answer[last:]))
    return "".join(out)


def cited_picture(seen) -> str:
    """The learner's cited on six recorded grounded answers: each answer with its brackets marked as input data, and
    what the function found, as a mistake where it is not what the course's cited finds."""
    w = _CITED_WORDS[seen.lang]
    mine = seen.mine
    fn = mine.names.get("cited") if mine is not None else None
    if not callable(fn):
        return ""
    rows = []
    for qid, seed in EXAMPLES_CITED:
        answer = answers(qid, "grounded")[seed]
        try:
            got = with_time_limit(lambda: fn(answer))
        except Exception:   # the learner's code; the check says what went wrong
            return ""
        if not isinstance(got, list):
            return yours.figure(w["title"], f'<div style="{_LABEL}">{html.escape(w["none"].format(got=repr(got)[:80]))}</div>', alt=w["title"])
        want = summaries.cited(answer)
        ok = got == want
        verdict = (f'<span style="{_LABEL}">{html.escape(w["yours"])}</span> <span style="{_CODE};'
                   f'color:var(--ea-{"ink" if ok else "loss-text"})">{html.escape(repr(got))}</span>'
                   + ("" if ok else f' <span style="{_LABEL}">{html.escape(w["course"].format(want=repr(want)))}</span>'))
        rows.append(f'<div style="display:grid;grid-template-columns:minmax(0,1fr) auto;gap:0.2rem 0.8rem;align-items:baseline;margin:3px 0;'
                    f'padding:0.1rem 0.4rem;border-left:3px solid var(--ea-{"line" if ok else "loss"})">'
                    f'<span style="{_CODE};color:var(--ea-pred-text);overflow-wrap:anywhere">{_marked(answer.strip())}</span><span>{verdict}</span></div>')
    return yours.figure(w["title"], "".join(rows), alt=w["title"])


# --- exercise 2: supported -------------------------------------------------------------------------------------------------

class Named(list):
    """The lesson's list of sources, written in the check's messages by its name."""

    def __init__(self, items, name: str):
        super().__init__(items)
        self.name = name

    def __repr__(self):
        return self.name


SHOP = Named(["The shop opens at 9 on Saturday.", "A full service costs 85 euros.", "Brake pads cost 12 euros per wheel.",
              "A rental bike costs 15 euros a day."], "SHOP")
START_SUPPORTED = '''def supported(answer, passages):
    found = cited(answer)
    if len(found) == 0:
        return False
    text = ""
    for n in found:
        text = text + " " + passages[n - 1]
    return True
'''
SOLUTION_SUPPORTED = START_SUPPORTED.replace('''    for n in found:
        text = text + " " + passages[n - 1]
    return True
''', '''    for n in found:
        if n < 1 or n > len(passages):
            return False
        text = text + " " + passages[n - 1]
    for number in numbers(answer):
        if number not in numbers(text):
            return False
    return True
''')
SUPPORTED_NAMES = {"cited": summaries.cited, "numbers": summaries.numbers, "SHOP": SHOP}
NO_RANGE = SOLUTION_SUPPORTED.replace('''        if n < 1 or n > len(passages):
            return False
''', "")
NO_ZERO = SOLUTION_SUPPORTED.replace("n < 1 or n > len(passages)", "n > len(passages)")
AS_SUBSTRING = SOLUTION_SUPPORTED.replace("if number not in numbers(text):", "if number not in text:")
ANY_NUMBER = SOLUTION_SUPPORTED.replace('''    for number in numbers(answer):
        if number not in numbers(text):
            return False
    return True
''', '''    for number in numbers(answer):
        if number in numbers(text):
            return True
    return False
''')
ALL_SOURCES = SOLUTION_SUPPORTED.replace('''    for number in numbers(answer):
        if number not in numbers(text):''', '''    for number in numbers(answer):
        if number not in numbers(" ".join(passages)):''')
EXAMPLES_SUPPORTED = [("q01", 0), ("q23", 0), ("q18", 0), ("q24", 0)]   # grounded answers: (question, seed)

_SUPPORTED_WORDS = {
    "en": {"title": "What your supported says about Qwen3's {n} grounded answers to the questions a point answers",
           "verdict": "your supported", "right": "right", "other": "not right", "examples": "Four of them:",
           "none": "Your function gave back {got}, not True or False."},
    "nl": {"title": "Wat je supported zegt over de {n} grounded antwoorden van Qwen3 op de vragen die een punt beantwoordt",
           "verdict": "je supported", "right": "goed", "other": "niet goed", "examples": "Vier ervan:",
           "none": "Je functie gaf {got} terug, geen True of False."},
}


def judged(fn) -> list | None:
    """For every grounded answer to a question a point answers: (question, answer, whether it is right, what `fn`
    says about it with its three sources); None when `fn` fails or gives something other than True or False."""
    out = []
    for q in ANSWERABLE:
        passages = found(q)
        for a in answers(q["id"], "grounded"):
            try:
                said = with_time_limit(lambda: fn(a, list(passages)))
            except Exception:   # the learner's code; the check says what went wrong
                return None
            if not isinstance(said, bool):
                return [(q, a, None, said)]
            out.append((q, a, summaries.right(a, q), said))
    return out


def supported_picture(seen) -> str:
    """The learner's supported on all recorded grounded answers to the questions a point answers: how many it says
    True and False of, split by whether the answer is right, with the right ones in the color of model output and the
    others as mistakes; then four of the answers with what it said."""
    w = _SUPPORTED_WORDS[seen.lang]
    mine = seen.mine
    fn = mine.names.get("supported") if mine is not None else None
    if not callable(fn):
        return ""
    rows = judged(fn)
    if rows is None:
        return ""
    title = w["title"].format(n=SEEDS * len(ANSWERABLE))
    if rows and rows[0][2] is None:
        return yours.figure(title, f'<div style="{_LABEL}">{html.escape(w["none"].format(got=repr(rows[0][3])[:80]))}</div>', alt=title)
    count = {(r, s): sum(1 for _, _, right, said in rows if right == r and said == s) for r in (True, False) for s in (True, False)}
    cell = "padding:0.15rem 0.6rem;text-align:right;font-family:var(--ea-font-ui);font-size:13px"
    table = (f'<table style="border-collapse:collapse;margin:0.2rem 0 0.5rem">'
             f'<tr><td style="{_LABEL};padding:0.15rem 0.6rem">{html.escape(w["verdict"])}</td>'
             f'<td style="{cell};color:var(--ea-pred-text)">{html.escape(w["right"])}</td><td style="{cell};color:var(--ea-loss-text)">{html.escape(w["other"])}</td></tr>'
             + "".join(f'<tr><td style="{_CODE};padding:0.15rem 0.6rem;color:var(--ea-ink)">{s}</td>'
                       f'<td style="{cell};color:var(--ea-ink);border-top:1px solid var(--ea-line)">{count[(True, s)]}</td>'
                       f'<td style="{cell};color:var(--ea-ink);border-top:1px solid var(--ea-line)">{count[(False, s)]}</td></tr>' for s in (True, False))
             + "</table>")
    examples = []
    for qid, seed in EXAMPLES_SUPPORTED:
        q = question(qid)
        a = answers(qid, "grounded")[seed]
        said = next(s for qq, aa, _, s in rows if qq is q and aa == a)
        right = summaries.right(a, q)
        examples.append(f'<div style="margin:3px 0;padding:0.1rem 0.4rem;border-left:3px solid var(--ea-{"pred" if right else "loss"})">'
                        f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)"><em>{html.escape(q["text"]["en"])}</em></div>'
                        f'<div style="display:grid;grid-template-columns:minmax(0,1fr) auto;gap:0.2rem 0.8rem;align-items:baseline">'
                        f'<span style="{_CODE};color:var(--ea-pred-text);overflow-wrap:anywhere">{_marked(a.strip())}</span>'
                        f'<span style="{_LABEL}">{html.escape(w["right"] if right else w["other"])} · <span style="{_CODE};color:var(--ea-ink)">{said}</span></span></div></div>')
    body = table + f'<div style="{_LABEL}">{html.escape(w["examples"])}</div>' + "".join(examples)
    return yours.figure(title, body, alt=title)


def _index_error(e) -> bool:
    return isinstance(e, IndexError)


def _int_error(e) -> bool:
    return isinstance(e, ValueError) and "int()" in str(e)


_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}

EXERCISES = {
    "cited": FunctionBox(
        name="cited",
        picture=cited_picture,
        function="cited",
        save_as=("app", "citations"),   # Project: your assistant reads your documents builds on it
        start=START_CITED,
        solution=SOLUTION_CITED,
        visible=[(CITED_EXAMPLE,)],
        hidden=[("",), ("I don't know.",), ("The shop opens at 9 [3], and closes at 5 [1]. [3]",), ("See [a] and [12].",),
                ("A list such as [1, 2] names no source; [2] does.",)],
        mistakes=[
            Mistake("as-text", wrong=AS_TEXT,
                    en="`{call}` gives `{got}`: texts, not numbers. Make a whole number of each with `int(inside)`, so that you can use it as the number of a source.",
                    nl="`{call}` geeft `{got}`: teksten, geen getallen. Maak van elk een heel getal met `int(inside)`, zodat je het als nummer van een bron kunt gebruiken."),
            Mistake("twice", wrong=TWICE,
                    en="`{call}` gives `{got}`: a source the answer names twice is in the list twice. Add a number only when it is not in `found` yet: `int(inside) not in found`.",
                    nl="`{call}` geeft `{got}`: een bron die het antwoord twee keer noemt, staat twee keer in de lijst. Voeg een getal alleen toe als het nog niet in `found` zit: `int(inside) not in found`."),
            Mistake("no-digits", error=_int_error,
                    en="With `{call}`, `int` gets a text that is not a whole number, and stops. Check first with `inside.isdigit()` that the part holds only digits.",
                    nl="Met `{call}` krijgt `int` een tekst die geen heel getal is, en stopt. Kijk eerst met `inside.isdigit()` of het stuk alleen cijfers heeft."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`cited` gives back every text between `[` and `]` as it is: as text, also when it is not a number, and twice when the answer names a source twice. Keep a part only when it is a number, as a whole number, and only once.",
                      "nl": "`cited` geeft elke tekst tussen `[` en `]` terug zoals hij is: als tekst, ook als het geen getal is, en twee keer als het antwoord een bron twee keer noemt. Houd een stuk alleen als het een getal is, als heel getal, en maar één keer."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: the numbers in square brackets, as whole numbers, in order and each once.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: de getallen tussen vierkante haken, als hele getallen, op volgorde en elk één keer."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for an answer without brackets, for a source named twice, and for brackets with something other than one number in them?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Werkt je functie voor een antwoord zonder haken, voor een bron die twee keer genoemd wordt, en voor haken met iets anders dan één getal erin?"},
            "no_function": {"en": "Your box does not make a function `cited`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `cited`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `cited` is right in all {n} cases: the numbers of the sources an answer names, in order and each once. The check kept it as a file, `{saved}`.",
                        "nl": "Je `cited` klopt in alle {n} gevallen: de nummers van de bronnen die een antwoord noemt, op volgorde en elk één keer. De check heeft hem als bestand bewaard, `{saved}`."},
            "kept": {"en": "Your `cited` is right. You changed `{saved}` yourself, so the check left it as it is.",
                     "nl": "Je `cited` is goed. Je hebt `{saved}` zelf veranderd, dus de check liet het staan zoals het is."},
        },
    ),
    "supported": FunctionBox(
        name="supported",
        picture=supported_picture,
        function="supported",
        start=START_SUPPORTED,
        solution=SOLUTION_SUPPORTED,
        names=SUPPORTED_NAMES,
        visible=[("A full service costs 85 euros [2].", SHOP)],
        hidden=[("A full service costs 95 euros [2].", SHOP), ("A full service costs 85 euros [1].", SHOP),
                ("A full service costs 85 euros.", SHOP), ("The shop opens on Saturday [1].", SHOP),
                ("Brake pads cost 12 euros [1].", SHOP), ("A rental bike costs 1 euro an hour [4].", SHOP),
                ("Brake pads cost 12 euros [3], a service 85 [2].", SHOP), ("Brake pads cost 12 euros [5].", SHOP),
                ("A rental bike costs 15 euros a day [0].", SHOP)],
        mistakes=[
            Mistake("no-range", error=_index_error,
                    en="With `{call}`, the answer names a source that is not there, and `passages[n - 1]` stops. Give back `False` for a number below 1 or above `len(passages)`, before you take the passage.",
                    nl="Met `{call}` noemt het antwoord een bron die er niet is, en stopt `passages[n - 1]`. Geef `False` terug voor een nummer onder de 1 of boven `len(passages)`, voordat je de passage pakt."),
            Mistake("zero", wrong=NO_ZERO, state="almost",
                    en="Almost. `{call}` gives `{got}`: `[0]` names no source, but `passages[0 - 1]` is the last one. Give back `False` for a number below 1 too.",
                    nl="Bijna. `{call}` geeft `{got}`: `[0]` noemt geen bron, maar `passages[0 - 1]` is de laatste. Geef ook `False` terug voor een nummer onder de 1."),
            Mistake("substring", wrong=AS_SUBSTRING,
                    en="`{call}` gives `{got}`: `\"1\"` is in `\"15\"` as a piece of text, so a number that is not in the source still counts. Compare with the numbers of the text: `numbers(text)`.",
                    nl="`{call}` geeft `{got}`: `\"1\"` zit als stukje tekst in `\"15\"`, dus telt een getal dat niet in de bron staat toch. Vergelijk met de getallen van de tekst: `numbers(text)`."),
            Mistake("any-number", wrong=ANY_NUMBER,
                    en="`{call}` gives `{got}`: your function says `True` as soon as one number is in the sources. Every number of the answer must be in them: give back `False` at the first that is not, and `True` after the loop.",
                    nl="`{call}` geeft `{got}`: je functie zegt `True` zodra één getal in de bronnen staat. Elk getal van het antwoord moet erin staan: geef `False` terug bij het eerste dat er niet in staat, en `True` na de loop."),
            Mistake("all-sources", wrong=ALL_SOURCES,
                    en="`{call}` gives `{got}`: you look in all sources, also those the answer does not name. Look only in `text`, the sources it names.",
                    nl="`{call}` geeft `{got}`: je kijkt in alle bronnen, ook in die het antwoord niet noemt. Kijk alleen in `text`, de bronnen die het noemt."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`supported` says `True` for every answer that names a source, whatever the numbers in it. Give back `False` when it names a source that is not there, or holds a number that is not in the sources it names.",
                      "nl": "`supported` zegt `True` voor elk antwoord dat een bron noemt, wat de getallen erin ook zijn. Geef `False` terug als het een bron noemt die er niet is, of een getal heeft dat niet in de bronnen staat die het noemt."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: `True` only when the answer names a source, every source it names is there, and every number in it is in the sources it names.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: alleen `True` als het antwoord een bron noemt, elke bron die het noemt er is, en elk getal erin in de bronnen staat die het noemt."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for a number that is not in the source, for a source that is not there, and for an answer without numbers?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Werkt je functie voor een getal dat niet in de bron staat, voor een bron die er niet is, en voor een antwoord zonder getallen?"},
            "no_function": {"en": "Your box does not make a function `supported`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `supported`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `supported` is right in all {n} cases: `True` only when every number of the answer is in the sources it names.",
                        "nl": "Je `supported` klopt in alle {n} gevallen: alleen `True` als elk getal van het antwoord in de bronnen staat die het noemt."},
        },
    ),
}

REQUIRED = ["cited", "supported", "named-source", "refused"]

CASES = [
    ("cited", START_CITED, "empty", None),
    ("cited", "", "empty", None),
    ("cited", SOLUTION_CITED, "correct", None),
    ("cited", '''def cited(answer):
    found = []
    for part in answer.split("[")[1:]:
        inside = part.split("]")[0]
        if inside.isdigit():
            n = int(inside)
            if n not in found:
                found.append(n)
    return found
''', "correct", None),
    ("cited", AS_TEXT, "wrong", "as-text"),
    ("cited", TWICE, "wrong", "twice"),
    ("cited", NO_DIGITS, "error", "no-digits"),
    ("cited", SOLUTION_CITED.replace('answer.split("[")[1:]', 'answer.split("[")[2:]'), "almost", None),
    ("cited", SOLUTION_CITED.replace("    return found\n", "    return sorted(found)\n"), "almost", None),
    ("supported", START_SUPPORTED, "empty", None),
    ("supported", "", "empty", None),
    ("supported", SOLUTION_SUPPORTED, "correct", None),
    ("supported", SOLUTION_SUPPORTED.replace("if n < 1 or n > len(passages):", "if not 1 <= n <= len(passages):"), "correct", None),
    ("supported", SOLUTION_SUPPORTED.replace('''    for number in numbers(answer):
        if number not in numbers(text):
            return False
    return True
''', '''    return len(numbers(answer) - numbers(text)) == 0
'''), "correct", None),
    ("supported", START_SUPPORTED.replace("    return True\n", '''    for number in numbers(answer):
        if number not in numbers(text):
            return False
    return True
'''), "error", "no-range"),
    ("supported", NO_ZERO, "almost", "zero"),
    ("supported", AS_SUBSTRING, "wrong", "substring"),
    ("supported", ANY_NUMBER, "wrong", "any-number"),
    ("supported", ALL_SOURCES, "wrong", "all-sources"),
    ("supported", SOLUTION_SUPPORTED.replace("    return True\n", "    return False\n"), "wrong", None),
]


# --- the demo: Qwen3 with your system prompt, and your checks -----------------------------------------------------------

def downloaded() -> bool:
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    chat.download_or_stop(button, texts)


def learner_checks(cited_text, supported_text):
    """The learner's cited and supported, the second with their own cited, when both are right; else None."""
    for name, text in (("cited", cited_text), ("supported", supported_text)):
        if text is MISSING or text is None or EXERCISES[name].check(text).state != "correct":
            return None
    mine = run_code(cited_text).names["cited"]
    return mine, run_code(supported_text, dict(SUPPORTED_NAMES, cited=mine)).names["supported"]


def demo_messages(qid: str, system: str) -> list:
    """The messages the demo sends: the system prompt when there is one, then the question with its three sources."""
    q = question(qid)
    user = {"role": "user", "content": with_sources(q["text"]["en"], found(q))}
    return ([{"role": "system", "content": system.strip()}] if system.strip() else []) + [user]


def _demo_html(passages: list, text: str, note: str, lang: str, words: dict) -> str:
    sources = "".join(f'<div style="{_CODE};background:var(--ea-data-wash);border-left:3px solid var(--ea-data);padding:0.1rem 0.4rem;margin:2px 0;'
                      f'white-space:pre-wrap">[{i}] {html.escape(p)} <span style="{_LABEL}">({html.escape(lesson_of(p)["title"][lang])})</span></div>'
                      for i, p in enumerate(passages, start=1))
    return (f'<div style="{_LABEL};margin-top:0.2rem">{html.escape(words["sources"])}</div>{sources}'
            f'<div style="{_LABEL};margin-top:0.4rem">{html.escape(words["answer"])}</div>'
            f'<div style="font-family:var(--ea-font-code);font-size:13px;white-space:pre-wrap;overflow-wrap:anywhere;color:var(--ea-pred-text)">{html.escape(text)}</div>'
            f'<div style="{_LABEL};margin-top:0.3rem">{html.escape(note)}</div>')


def demo(checks, qid: str, system: str, show, words: dict, lang: str) -> str:
    """The question with its three sources to Qwen3, after the learner's system prompt, with the answer so far shown by
    `show(html)`; then what the learner's cited and supported say about the answer."""
    cited, supported = checks
    q = question(qid)
    passages = found(q)
    title = words["title"].format(question=q["text"]["en"])
    a = chat.answer_live(demo_messages(qid, system),
                         lambda text: show(yours.figure(title, _demo_html(passages, text or "", words["loading"] if text is None else words["writing"], lang, words), alt=title)),
                         max_tokens=80, seed=0)
    if isinstance(a, str):
        return f'<p><em>{html.escape(words[a])}</em></p>'
    text = a["message"]["content"].strip()
    try:
        names, said = with_time_limit(lambda: (cited(text), supported(text, list(passages))))
        verdict = words["checked"].format(cited=repr(names), supported=said)
    except Exception as e:   # the learner's code, on an answer it has not seen before
        verdict = words["failed"].format(why=f"{type(e).__name__}: {e}")
    done = words["done"].format(tokens=a["tokens"], seconds=number(a["seconds"], lang, 1))
    return yours.figure(title, _demo_html(passages, text, done + " " + verdict, lang, words), alt=title)
