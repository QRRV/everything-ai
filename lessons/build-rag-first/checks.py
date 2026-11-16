# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-rag-first: the words of a text, a search for the points that share the most words with a
question, and a whole RAG that puts the three best in front of the question and sends it to the server, against a
stand-in; the course's recorded runs (data/course-rag-qwen.json): the 28 questions about the course, asked alone and
with RAG; their pictures, and the demo."""
import functools
import html
import json

from everything_ai import apicheck, chat, paths, serve, summaries, yours
from everything_ai.apicheck import URL, ApiBox, Case
from everything_ai.check import FunctionBox, Mistake, run_code, with_time_limit

DOWNLOADS = {"Qwen3-0.6B": chat.FILES}   # what the demo downloads, for the line under the header

SYSTEM = "You answer questions about the course Everything AI. Answer in one sentence."
POINTS = summaries.points("en")   # the knowledge base: every point of every summary, in English
K = 3
SEEDS = 2
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
QUESTIONS = summaries.questions()


def found(q: dict) -> list:
    """The K points the course's word search finds for a question."""
    return summaries.search(q["text"]["en"], POINTS, K)


def _messages(q: dict, way: str) -> list:
    user = q["text"]["en"] if way == "alone" else with_sources(q["text"]["en"], found(q))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


RECORD = {f'{q["id"]}-{way}': {"messages": _messages(q, way), "max_tokens": 60, "seeds": SEEDS}
          for q in QUESTIONS for way in ("alone", "rag")}
DATA = paths.ROOT / "data" / "course-rag-qwen.json"
ABOUT = summaries.about()
ANSWERABLE = [q for q in QUESTIONS if q["kind"] != "none"]


@functools.cache
def course() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-rag-first"]


def answers(qid: str, way: str) -> list:
    return course()[f"{qid}-{way}"]["answers"]


def question(qid: str) -> dict:
    return next(q for q in QUESTIONS if q["id"] == qid)


def hit(q: dict) -> bool:
    """Whether the word search puts the point that answers the question among the K it finds."""
    return q["gold"] is not None and summaries.text_of(q["gold"]) in found(q)


def lesson_of(text: str) -> dict:
    return ABOUT[POINTS.index(text)]


def lesson_count() -> int:
    return summaries.lessons()


def worked() -> dict:
    """The worked example: the words of the question, the three points the search finds with how many words each
    shares, and how many of all points share each number of words."""
    q = question("q11")["text"]["en"]
    top = summaries.search(q, POINTS, K)
    counts = {}
    for text in POINTS:
        n = summaries.overlap(q, text)
        counts[n] = counts.get(n, 0) + 1
    common = max(counts, key=counts.get)
    return {"question": q, "asked": sorted(words(q)), "top": top, "shares": [summaries.overlap(q, t) for t in top],
            "shared": sorted(words(q) & words(top[0])), "common": common, "many": counts[common], "zero": counts.get(0, 0),
            "lesson": lesson_of(top[0])["lesson"]}


@functools.cache
def findings() -> dict:
    """For the 24 questions a point answers: how often the search found that point, per kind; how many answers were
    right alone and with RAG, and with RAG when the point was found and when it was not; and for the 4 questions no
    point answers, how many answers said the sources do not say."""
    hits = [q for q in ANSWERABLE if hit(q)]
    rag = {q["id"]: sum(summaries.right(a["text"], q) for a in answers(q["id"], "rag")) for q in ANSWERABLE}
    none = [q for q in QUESTIONS if q["kind"] == "none"]
    return {"questions": len(ANSWERABLE), "hits": len(hits),
            "words_hits": sum(q["kind"] == "words" for q in hits), "words": sum(q["kind"] == "words" for q in ANSWERABLE),
            "meaning_hits": sum(q["kind"] == "meaning" for q in hits), "meaning": sum(q["kind"] == "meaning" for q in ANSWERABLE),
            "alone": sum(summaries.right(a["text"], q) for q in ANSWERABLE for a in answers(q["id"], "alone")),
            "rag": sum(rag.values()), "runs": SEEDS * len(ANSWERABLE),
            "rag_hit": sum(rag[q["id"]] for q in hits), "hit_runs": SEEDS * len(hits),
            "rag_miss": sum(rag[q["id"]] for q in ANSWERABLE if not hit(q)), "miss_runs": SEEDS * (len(ANSWERABLE) - len(hits)),
            "none": len(none), "none_runs": SEEDS * len(none),
            "none_unknown_alone": sum(summaries.says_unknown(a["text"]) for q in none for a in answers(q["id"], "alone")),
            "none_unknown_rag": sum(summaries.says_unknown(a["text"]) for q in none for a in answers(q["id"], "rag"))}


_LABEL = "font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)"


def _swatch(color: str, label: str) -> str:
    return (f'<span style="display:inline-flex;align-items:center;gap:0.25rem;margin-right:0.8rem;{_LABEL}">'
            f'<span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:var(--ea-{color})"></span>{html.escape(label)}</span>')


def _square(ok: bool, text: str) -> str:
    return (f'<span title="{html.escape(" ".join(text.split())[:160])}" style="display:inline-block;width:12px;height:12px;'
            f'margin-right:2px;border-radius:2px;background:var(--ea-{"pred" if ok else "loss"})"></span>')


_RUNS_WORDS = {
    "en": {"title": "Did the search find the right point, and is Qwen3's answer right? Two runs each",
           "question": "question", "found": "point found", "alone": "without sources", "rag": "with RAG", "yes": "found", "no": "not found",
           "right": "right answer", "not": "not right", "count": "{k} of {n}", "meaning": "asks it in other words"},
    "nl": {"title": "Vond de zoekfunctie het goede punt, en is het antwoord van Qwen3 goed? Elk twee runs",
           "question": "vraag", "found": "punt gevonden", "alone": "zonder bronnen", "rag": "met RAG", "yes": "gevonden", "no": "niet gevonden",
           "right": "goed antwoord", "not": "niet goed", "count": "{k} van de {n}", "meaning": "vraagt het in andere woorden"},
}


def runs_picture(lang: str) -> str:
    """A row per question a point answers: whether the word search found that point among its three, as input data,
    then a square per run alone and with RAG, in the color of model output when right and as a mistake when not."""
    w = _RUNS_WORDS[lang]
    f = findings()
    head = "".join(f'<span style="{_LABEL}">{html.escape(w[k])}</span>' for k in ("question", "found", "alone", "rag"))
    rows = []
    for q in ANSWERABLE:
        mark = f' <span style="{_LABEL}">({html.escape(w["meaning"])})</span>' if q["kind"] == "meaning" else ""
        yes = hit(q)
        found_cell = (f'<span style="{_LABEL}"><span style="display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:middle;'
                      f'{"background:var(--ea-data)" if yes else "border:1px dashed var(--ea-line)"}"></span> {html.escape(w["yes"] if yes else w["no"])}</span>')
        cells = "".join(f'<span>{"".join(_square(summaries.right(a["text"], q), a["text"]) for a in answers(q["id"], way))}</span>' for way in ("alone", "rag"))
        rows.append(f'<span style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink)">{html.escape(q["text"][lang])}{mark}</span>{found_cell}{cells}')
    foot = (f'<span></span><span style="{_LABEL}">{html.escape(w["count"].format(k=f["hits"], n=f["questions"]))}</span>'
            f'<span style="{_LABEL}">{html.escape(w["count"].format(k=f["alone"], n=f["runs"]))}</span>'
            f'<span style="{_LABEL}">{html.escape(w["count"].format(k=f["rag"], n=f["runs"]))}</span>')
    grid = (f'<div style="display:grid;grid-template-columns:minmax(0,1fr) auto auto auto;gap:0.3rem 0.8rem;align-items:center">'
            f'{head}{"".join(rows)}{foot}</div>')
    legend = _swatch("data", w["yes"]) + _swatch("pred", w["right"]) + _swatch("loss", w["not"])
    return yours.figure(w["title"], grid + f'<div style="margin-top:0.4rem">{legend}</div>', alt=w["title"])


# --- exercise 1: words ---------------------------------------------------------------------------------------------------

START_WORDS = '''def words(text):
    return set(text.split())
'''
SOLUTION_WORDS = '''def words(text):
    return set([word.strip(".,:;!?()") for word in text.lower().split()])
'''
NO_LOWER = SOLUTION_WORDS.replace("text.lower().split()", "text.split()")
NO_STRIP = SOLUTION_WORDS.replace('word.strip(".,:;!?()")', "word")
AS_LIST = SOLUTION_WORDS.replace('    return set([word.strip(".,:;!?()") for word in text.lower().split()])', '    return [word.strip(".,:;!?()") for word in text.lower().split()]')
words = run_code(SOLUTION_WORDS).names["words"]
WORDS_QUESTION = question("q11")["text"]["en"]
WORDS_POINT = summaries.text_of(question("q11")["gold"])

_WORDS_WORDS = {
    "en": {"title": "The words your words finds in the question and in the point that answers it", "question": "question:",
           "point": "point:", "shared": "in both", "only": "in one", "dirty": "still has a capital or punctuation",
           "count": "{k} words in both", "none": "Your function gave back {got}, not a set."},
    "nl": {"title": "De woorden die je words vindt in de vraag en in het punt dat hem beantwoordt", "question": "vraag:",
           "point": "punt:", "shared": "in allebei", "only": "in één", "dirty": "heeft nog een hoofdletter of leesteken",
           "count": "{k} woorden in allebei", "none": "Je functie gaf {got} terug, geen set."},
}


def _chips(items, shared: set) -> str:
    out = []
    for word in sorted(items, key=str):
        word = str(word)
        dirty = word != word.lower() or word.strip(".,:;!?()") != word
        style = ("background:var(--ea-loss-wash);color:var(--ea-loss-text);border:1px solid var(--ea-loss)" if dirty else
                 "background:var(--ea-data-wash);color:var(--ea-ink);border:1px solid var(--ea-data)" if word in shared else
                 "background:transparent;color:var(--ea-ink-2);border:1px dashed var(--ea-line)")
        out.append(f'<span style="display:inline-block;padding:0 0.25rem;margin:1px;border-radius:2px;font-family:var(--ea-font-code);font-size:12px;{style}">{html.escape(word)}</span>')
    return "".join(out)


def words_picture(seen) -> str:
    """The learner's words of the worked example's question and of its point, as chips: a word in both as input data,
    a word with a capital or punctuation still on it as a mistake."""
    w = _WORDS_WORDS[seen.lang]
    mine = seen.mine
    fn = mine.names.get("words") if mine is not None else None
    if not callable(fn):
        return ""
    try:
        asked, point = with_time_limit(lambda: (fn(WORDS_QUESTION), fn(WORDS_POINT)))
    except Exception:   # the learner's code; the check says what went wrong
        return ""
    if not isinstance(asked, (set, list, tuple)) or not isinstance(point, (set, list, tuple)):
        return yours.figure(w["title"], f'<div style="{_LABEL}">{html.escape(w["none"].format(got=repr(asked)[:80]))}</div>', alt=w["title"])
    shared = set(asked) & set(point)
    body = (f'<div style="margin:0.2rem 0"><span style="{_LABEL};margin-right:0.4rem">{html.escape(w["question"])}</span>{_chips(set(asked), shared)}</div>'
            f'<div style="margin:0.2rem 0"><span style="{_LABEL};margin-right:0.4rem">{html.escape(w["point"])}</span>{_chips(set(point), shared)}</div>'
            f'<div style="margin-top:0.3rem">{_swatch("data", w["shared"])}{_swatch("loss", w["dirty"])}'
            f'<span style="{_LABEL}">{html.escape(w["count"].format(k=len(shared)))}</span></div>')
    return yours.figure(w["title"], body, alt=w["title"])


# --- exercise 2: search --------------------------------------------------------------------------------------------------

START_SEARCH = '''def search(question, points, k):
    found = []
    for point in points:
        found.append((0, point))
    return [point for shared, point in sorted(found, reverse=True)[:k]]
'''
SOLUTION_SEARCH = '''def search(question, points, k):
    asked = words(question)
    found = []
    for point in points:
        found.append((len(asked & words(point)), point))
    return [point for shared, point in sorted(found, reverse=True)[:k]]
'''
LONGEST = SOLUTION_SEARCH.replace("len(asked & words(point))", "len(words(point))")
FEWEST = SOLUTION_SEARCH.replace("sorted(found, reverse=True)", "sorted(found)")
PAIRS = SOLUTION_SEARCH.replace("    return [point for shared, point in sorted(found, reverse=True)[:k]]\n", "    return sorted(found, reverse=True)[:k]\n")
SMALL = ["The shop opens at nine on Saturday.", "A full service costs 85 euros and takes two working days.",
         "Brake pads cost 12 euros per wheel.", "Rental bikes come with a lock and a helmet, and cost 15 euros a day."]
EXAMPLES = ["q11", "q13", "q18"]   # the picture: found, found, and a question in other words that the search misses

_SEARCH_WORDS = {
    "en": {"title": "The three points your search finds among all {n}, for three questions", "shared": "{k} shared",
           "gold": "the point that answers it", "other": "another point", "missing": "The point that answers it is not among them."},
    "nl": {"title": "De drie punten die je search vindt tussen alle {n}, voor drie vragen", "shared": "{k} gedeeld",
           "gold": "het punt dat hem beantwoordt", "other": "een ander punt", "missing": "Het punt dat hem beantwoordt zit er niet bij."},
}


def search_picture(seen) -> str:
    """The learner's search on all points for three questions: per question the three it finds, with how many words
    each shares, the point that answers it as input data."""
    w = _SEARCH_WORDS[seen.lang]
    mine = seen.mine
    fn = mine.names.get("search") if mine is not None else None
    if not callable(fn):
        return ""
    blocks = []
    for qid in EXAMPLES:
        q = question(qid)
        try:
            got = with_time_limit(lambda: fn(q["text"]["en"], list(POINTS), K), 10)
        except Exception:   # the learner's code; the check says what went wrong
            return ""
        if not isinstance(got, list):
            return ""
        gold = summaries.text_of(q["gold"])
        rows = []
        for text in got[:K]:
            text = text if isinstance(text, str) else repr(text)
            is_gold = text == gold
            style = ("background:var(--ea-data-wash);border-left:3px solid var(--ea-data)" if is_gold else "border-left:3px solid var(--ea-line)")
            shared = len(summaries.words(q["text"]["en"]) & summaries.words(text))
            rows.append(f'<div style="margin:2px 0;padding:0.1rem 0.4rem;{style};font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink);'
                        f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis"><span style="color:var(--ea-ink-2)">'
                        f'{html.escape(w["shared"].format(k=shared))}</span> {html.escape(text)}</div>')
        if gold not in got[:K]:
            rows.append(f'<div style="{_LABEL};color:var(--ea-loss-text)">{html.escape(w["missing"])}</div>')
        blocks.append(f'<div style="margin:0.3rem 0 0.5rem"><div style="font-family:var(--ea-font-ui);font-size:13px;color:var(--ea-ink)">'
                      f'<em>{html.escape(q["text"]["en"])}</em></div>{"".join(rows)}</div>')
    legend = _swatch("data", w["gold"])
    title = w["title"].format(n=number(len(POINTS), seen.lang, 0))
    return yours.figure(title, "".join(blocks) + f'<div>{legend}</div>', alt=title)


# --- exercise 3: rag -------------------------------------------------------------------------------------------------------

START_RAG = '''import json
import urllib.request


def rag(url, question):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
    body = {"model": "qwen3-0.6b", "messages": messages, "max_tokens": 60}
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        answer = json.loads(response.read())
    return answer["choices"][0]["message"]["content"]
'''
SOLUTION_RAG = START_RAG.replace('''    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
''', '''    passages = search(question, POINTS, 3)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": with_sources(question, passages)}]
''')
RAG_NAMES = {"SYSTEM": SYSTEM, "POINTS": POINTS, "search": summaries.search, "with_sources": with_sources}
RAG_QUESTIONS = [question("q11")["text"]["en"], question("q27")["text"]["en"]]


def _prompt(q: str) -> str:
    return with_sources(q, summaries.search(q, POINTS, K))


def _rag_stand_in():
    answers = {}
    for q in RAG_QUESTIONS:
        answers[_prompt(q)] = "The stand-in's answer from the sources."
        answers[q] = "The stand-in's answer without sources."
    return serve.StandIn(answers, default="The stand-in got a prompt it did not expect.")


def _posts(log: list) -> list:
    return [e["body"] for e in log if e["method"] == "POST" and isinstance(e["body"], dict)]


def _judge_rag(q: str):
    def judge(got, running):
        sent = _posts(running.log)
        messages = [m for m in (sent[-1].get("messages") or []) if isinstance(m, dict)] if sent else []
        user = next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), None)
        if user == q:
            return "question-only"
        if isinstance(user, str) and user.startswith(summaries_instruction()) and user != _prompt(q):
            return "passages"
        if not any(m.get("role") == "system" and m.get("content") == SYSTEM for m in messages):
            return "no-system"
        if got == "The stand-in's answer from the sources.":
            return None
        if isinstance(got, dict):
            return "whole"
        return "other"
    return judge


def summaries_instruction() -> str:
    return "Use these sources to answer the question."


RAG_CASES = [Case(_rag_stand_in, (URL, q), _judge_rag(q)) for q in RAG_QUESTIONS]


def rag_picture(seen) -> str:
    words = {"en": {"title": "What your rag sent, and what came back", "request": "Your code sent: {method} {path}",
                    "headers": "Headers:", "answer": "The server sent back: status {status}",
                    "stream": "a stream with {n} pieces", "none": "Your function sent nothing to the server yet."},
             "nl": {"title": "Wat je rag stuurde, en wat er terugkwam", "request": "Je code stuurde: {method} {path}",
                    "headers": "Headers:", "answer": "De server stuurde terug: status {status}",
                    "stream": "een stream met {n} stukjes", "none": "Je functie stuurde nog niets naar de server."}}[seen.lang]
    return apicheck.exchange(seen, "rag", RAG_CASES[0], words)


def _not_bytes(e) -> bool:
    return isinstance(e, TypeError) and "bytes" in str(e)


_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}

EXERCISES = {
    "words": FunctionBox(
        name="words",
        picture=words_picture,
        function="words",
        start=START_WORDS,
        solution=SOLUTION_WORDS,
        visible=[(WORDS_QUESTION,)],
        hidden=[("",), ("The model, the MODEL (and the model).",), ("Ready? Yes!",), ("top-5 error: 15.3%",)],
        mistakes=[
            Mistake("list", wrong=AS_LIST,
                    en="`{call}` gives `{got}`: a list, where a word can appear twice. Make a set of it with `set(...)`, so that every word counts once.",
                    nl="`{call}` geeft `{got}`: een lijst, waarin een woord twee keer kan staan. Maak er een set van met `set(...)`, zodat elk woord één keer telt."),
            Mistake("no-lower", wrong=NO_LOWER,
                    en="`{call}` gives `{got}`: words with a capital stay apart from the same words without one. Make the text lower case first, with `text.lower()`.",
                    nl="`{call}` geeft `{got}`: woorden met een hoofdletter blijven los van dezelfde woorden zonder. Maak de tekst eerst klein, met `text.lower()`."),
            Mistake("no-strip", wrong=NO_STRIP,
                    en="`{call}` gives `{got}`: punctuation still sticks to some words. Take it off both ends of each word, with `word.strip(\".,:;!?()\")`.",
                    nl="`{call}` geeft `{got}`: aan sommige woorden zitten nog leestekens. Haal ze van beide uiteinden van elk woord, met `word.strip(\".,:;!?()\")`."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`words` splits the text at the spaces, but keeps capitals and punctuation, so *Cache?* and *cache* are two different words. Make the text lower case, and take `.,:;!?()` off both ends of every word.",
                      "nl": "`words` splitst de tekst bij de spaties, maar houdt hoofdletters en leestekens, dus *Cache?* en *cache* zijn twee verschillende woorden. Maak de tekst klein, en haal `.,:;!?()` van beide uiteinden van elk woord."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: the set of the words, in lower case, with `.,:;!?()` taken off both ends of each.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: de set van de woorden, klein geschreven, met `.,:;!?()` van beide uiteinden van elk woord."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for an empty text, and for words with capitals, commas or brackets?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Werkt je functie voor een lege tekst, en voor woorden met hoofdletters, komma's of haakjes?"},
            "no_function": {"en": "Your box does not make a function `words`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `words`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `words` is right in all {n} cases: the set of the words, in lower case and without punctuation at their ends.",
                        "nl": "Je `words` klopt in alle {n} gevallen: de set van de woorden, klein geschreven en zonder leestekens aan de uiteinden."},
        },
    ),
    "search": FunctionBox(
        name="search",
        picture=search_picture,
        function="search",
        start=START_SEARCH,
        solution=SOLUTION_SEARCH,
        names={"words": words},
        visible=[("How much does a full service cost?", SMALL, 2)],
        hidden=[("How much does a full service cost?", SMALL, 0), ("Do rental bikes come with a lock?", SMALL, 1),
                ("When does the shop open on Saturday?", SMALL, 10), ("Brake pads?", SMALL, 3)],
        mistakes=[
            Mistake("longest", wrong=LONGEST,
                    en="`{call}` gives `{got}`: the points with the most words, whatever the question asks. Count only the words a point shares with the question: `len(asked & words(point))`.",
                    nl="`{call}` geeft `{got}`: de punten met de meeste woorden, wat de vraag ook vraagt. Tel alleen de woorden die een punt met de vraag deelt: `len(asked & words(point))`."),
            Mistake("fewest", wrong=FEWEST,
                    en="`{call}` gives `{got}`: the points that share the fewest words come first. Keep `reverse=True` in `sorted`, so that the most come first.",
                    nl="`{call}` geeft `{got}`: de punten die de minste woorden delen, komen eerst. Laat `reverse=True` in `sorted` staan, zodat de meeste eerst komen."),
            Mistake("pairs", wrong=PAIRS,
                    en="`{call}` gives `{got}`: pairs of a number and a point. Give back the points alone, as the last line of the box did.",
                    nl="`{call}` geeft `{got}`: paren van een getal en een punt. Geef alleen de punten terug, zoals de laatste regel van het vak deed."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`search` gives every point the score 0, so `sorted` puts them in reverse alphabetical order. Give each point the number of words it shares with the question instead.",
                      "nl": "`search` geeft elk punt de score 0, dus zet `sorted` ze in omgekeerde alfabetische volgorde. Geef elk punt in plaats daarvan het aantal woorden dat het met de vraag deelt."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: the `k` points that share the most words with the question, most first.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: de `k` punten die de meeste woorden met de vraag delen, de meeste eerst."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work with `k` 0, with `k` 1, and with a `k` larger than the number of points?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Werkt je functie met `k` 0, met `k` 1, en met een `k` groter dan het aantal punten?"},
            "no_function": {"en": "Your box does not make a function `search`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `search`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `search` is right in all {n} cases: the `k` points that share the most words with the question.",
                        "nl": "Je `search` klopt in alle {n} gevallen: de `k` punten die de meeste woorden met de vraag delen."},
        },
    ),
    "rag": ApiBox(
        name="rag",
        picture=rag_picture,
        function="rag",
        start=START_RAG,
        solution=SOLUTION_RAG,
        names=RAG_NAMES,
        cases=RAG_CASES,
        mistakes=[
            Mistake("data-text", error=_not_bytes,
                    en="urllib sends bytes, not text. Keep `.encode()` after `json.dumps(body)`, as the box had it.",
                    nl="urllib verstuurt bytes, geen tekst. Laat `.encode()` achter `json.dumps(body)` staan, zoals het in het vak stond."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`rag` sends the question alone, after the system prompt. Find the three points with `search(question, POINTS, 3)`, and send `with_sources(question, passages)` as the user's message.",
                      "nl": "`rag` stuurt alleen de vraag, na de system prompt. Vind de drie punten met `search(question, POINTS, 3)`, en stuur `with_sources(question, passages)` als het bericht van de user."},
            "question-only": {"en": "With `{call}`, the server got the question without the sources. Send `with_sources(question, passages)` as the user's message, with the three points `search` finds.",
                              "nl": "Met `{call}` kreeg de server de vraag zonder de bronnen. Stuur `with_sources(question, passages)` als het bericht van de user, met de drie punten die `search` vindt."},
            "passages": {"en": "With `{call}`, the sources in the prompt are not the three points `search(question, POINTS, 3)` finds for the question. Give `with_sources` exactly what that call gives back.",
                         "nl": "Met `{call}` zijn de bronnen in de prompt niet de drie punten die `search(question, POINTS, 3)` voor de vraag vindt. Geef `with_sources` precies wat die aanroep teruggeeft."},
            "no-system": {"en": "With `{call}`, the system prompt is gone from the messages. Keep the system message with `SYSTEM` first, as the box had it.",
                          "nl": "Met `{call}` is de system prompt uit de berichten verdwenen. Laat het system-bericht met `SYSTEM` vooraan staan, zoals het in het vak stond."},
            "whole": {"en": "With `{call}`, your function gives back the whole answer of the server. Give back only the text: `answer[\"choices\"][0][\"message\"][\"content\"]`.",
                      "nl": "Met `{call}` geeft je functie het hele antwoord van de server terug. Geef alleen de tekst terug: `answer[\"choices\"][0][\"message\"][\"content\"]`."},
            "other": {"en": "With `{call}`, your function gives back `{got}`. Give back the text of the model's answer.",
                      "nl": "Met `{call}` geeft je functie `{got}` terug. Geef de tekst van het antwoord van het model terug."},
            "correct": {"en": "Your `rag` finds the three points that fit the question best, puts them in the prompt after the system prompt, and gives back the model's answer: a whole RAG.",
                        "nl": "Je `rag` vindt de drie punten die het best bij de vraag passen, zet ze na de system prompt in de prompt, en geeft het antwoord van het model terug: een hele RAG."},
        },
    ),
}

REQUIRED = ["words", "search", "rag"]

CASES = [
    ("words", START_WORDS, "empty", None),
    ("words", "", "empty", None),
    ("words", SOLUTION_WORDS, "correct", None),
    ("words", "def words(text):\n    out = set()\n    for word in text.lower().split():\n        out.add(word.strip(\".,:;!?()\"))\n    return out\n", "correct", None),
    ("words", AS_LIST, "wrong", "list"),
    ("words", NO_LOWER, "wrong", "no-lower"),
    ("words", NO_STRIP, "wrong", "no-strip"),
    ("words", SOLUTION_WORDS.replace('".,:;!?()"', '".,?"'), "almost", None),
    ("search", START_SEARCH, "empty", None),
    ("search", SOLUTION_SEARCH, "correct", None),
    ("search", SOLUTION_SEARCH.replace("        found.append((len(asked & words(point)), point))\n",
                                       "        shared = 0\n        for word in words(point):\n            if word in asked:\n"
                                       "                shared = shared + 1\n        found.append((shared, point))\n"), "correct", None),
    ("search", LONGEST, "wrong", "longest"),
    ("search", FEWEST, "wrong", "fewest"),
    ("search", PAIRS, "wrong", "pairs"),
    ("search", SOLUTION_SEARCH.replace("[:k]", "[:k + 1]"), "wrong", None),
    ("rag", START_RAG, "empty", None),
    ("rag", SOLUTION_RAG, "correct", None),
    ("rag", START_RAG.replace('"max_tokens": 60', '"max_tokens": 50'), "wrong", "question-only"),
    ("rag", SOLUTION_RAG.replace("search(question, POINTS, 3)", "search(question, POINTS, 5)"), "wrong", "passages"),
    ("rag", SOLUTION_RAG.replace('    messages = [{"role": "system", "content": SYSTEM},\n                {"role": "user"',
                                 '    messages = [{"role": "user"'), "wrong", "no-system"),
    ("rag", SOLUTION_RAG.replace('    return answer["choices"][0]["message"]["content"]\n', "    return answer\n"), "wrong", "whole"),
    ("rag", SOLUTION_RAG.replace('    return answer["choices"][0]["message"]["content"]\n', "    return passages\n"), "wrong", "other"),
    ("rag", SOLUTION_RAG.replace("json.dumps(body).encode()", "json.dumps(body)"), "error", "data-text"),
]


# --- the demo: your rag, against the course's own server --------------------------------------------------------------------

def downloaded() -> bool:
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    chat.download_or_stop(button, texts)


def learner_rag(words_text, search_text, rag_text):
    """The learner's rag with their own words and search, when all three are right; else None. It gives back the
    function and the learner's search, so that the demo can show what it found."""
    from everything_ai.check import MISSING
    for name, text in (("words", words_text), ("search", search_text), ("rag", rag_text)):
        if text is MISSING or text is None or EXERCISES[name].check(text).state != "correct":
            return None
    mine_words = run_code(words_text).names["words"]
    mine_search = run_code(search_text, {"words": mine_words}).names["search"]
    names = dict(RAG_NAMES, search=mine_search)
    return run_code(rag_text, names).names["rag"], mine_search


def demo(learner, question_text: str, words: dict, lang: str) -> str:
    """The learner's rag on a question, through the course's server with Qwen3 behind it: the three points their
    search found, with the lesson of each, and the answer."""
    import time
    rag, search = learner
    question_text = " ".join(question_text.split())
    if not question_text:
        return f"<p><em>{html.escape(words['empty'])}</em></p>"
    running = serve.model_server()
    start = time.perf_counter()
    try:
        passages = search(question_text, list(POINTS), K)
        text = rag(running.url, question_text)
    except Exception as e:   # the learner's code, against the real server; it may have run out of memory
        said = words["memory"] if serve.out_of_memory(e) else words["failed"].format(why=f"{type(e).__name__}: {e}")
        return f"<p><em>{html.escape(said)}</em></p>"
    seconds = time.perf_counter() - start
    code = "font-family:var(--ea-font-code);font-size:13px;white-space:pre-wrap"
    sources = "".join(f'<div style="{code};background:var(--ea-data-wash);border-left:3px solid var(--ea-data);padding:0.1rem 0.4rem;margin:2px 0">'
                      f'[{i}] {html.escape(p)} <span style="{_LABEL}">({html.escape(lesson_of(p)["title"][lang])})</span></div>'
                      for i, p in enumerate(passages, start=1))
    body = (f'<div style="{_LABEL};margin-top:0.2rem">{html.escape(words["sources"])}</div>{sources}'
            f'<div style="{_LABEL};margin-top:0.4rem">{html.escape(words["answer"].format(seconds=number(seconds, lang, 0)))}</div>'
            f'<div style="{code};color:var(--ea-pred-text)">{html.escape(str(text).strip())}</div>')
    title = words["title"].format(question=question_text)
    return yours.figure(title, body, alt=title)


def number(x: float, lang: str, digits: int = 1) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text
