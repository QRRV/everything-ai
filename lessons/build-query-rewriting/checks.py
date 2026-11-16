# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-query-rewriting: let the model change a question before the search. The course's recorded runs of
Qwen3 (data/course-rag-qwen.json), for the 24 questions about the course that a point answers: a follow-up question
rewritten so that it stands on its own, a passage written to search with (HyDE) for a short question, and three other
versions of that short question; and Granite Embedding's embeddings of all of them (data/course-rewrites-granite.npz).
How often search by meaning puts the right point among the three best with each; the pictures, and the search to try
with the learner's own functions."""
import functools
import html
import json

import numpy as np

from everything_ai import paths, summaries, yours
from everything_ai.check import MISSING, FunctionBox, Mistake, run_code, with_time_limit

POINTS = summaries.points("en")   # the knowledge base: every point of every summary, in English
VECTORS = summaries.vectors("points", "en")
QUESTIONS = summaries.questions()
ANSWERABLE = [q for q in QUESTIONS if q["kind"] != "none"]
ABOUT = summaries.about()
K = 3
RRF_K = 60
WORKED = "q02"   # the follow-up of the worked example
DATA = paths.ROOT / "data" / "course-rag-qwen.json"
EMBEDDINGS = paths.ROOT / "data" / "course-rewrites-granite.npz"

# A follow-up question in a chat, after an earlier one that names what it is about: (the earlier question, the
# follow-up). The follow-up alone leaves out the name; the course wrote both, for each question a point answers.
FOLLOW = {
    "q01": ("The course trains a small CNN on the digits, doesn't it?", "How many parameters does it have?"),
    "q02": ("What does the counted bigram model of Tiny Shakespeare do?", "What is its perplexity?"),
    "q03": ("Which network won ImageNet's contest in 2012?", "What top-5 error did it reach?"),
    "q04": ("What is the downloaded digit reader?", "How many of the 1,000 test digits does it read right?"),
    "q05": ("Why does Qwen3 forget Sam when the earlier turn is left out of the request?", "In how many of the 8 runs did it still name Sam?"),
    "q06": ("What does nanoGPT train on, OpenWebText?", "How many GB are its ids?"),
    "q07": ("How does GPT-2 make its tokens?", "How many rules did it learn?"),
    "q08": ("What is the popularity baseline for recommending products?", "For what share of the customers does it find something?"),
    "q09": ("What does PCA do to the penguins?", "On two components, how many of the 342 does k-means put with their own species?"),
    "q10": ("What does nn.Conv2d(3, 8, 3) make?", "How many parameters does it have?"),
    "q11": ("What does prompt caching save on a long system prompt?", "How many seconds did a request take without it?"),
    "q12": ("How did Qwen3 sort the emails the plain way?", "How many of the 24 answers followed the email?"),
    "q13": ("What does DBSCAN do with the 400 made-up points?", "How many strays are among them?"),
    "q14": ("What does an opening of 3 by 3 do to the apples?", "How many specks does it remove?"),
    "q15": ("Where does the Grad-CAM of Egyptian cat look on the collage?", "What share of it lies on the cat half?"),
    "q16": ("What does compute_optimal do?", "How many tokens per parameter does it use?"),
    "q17": ("What is a SHA-256 checksum for?", "How many hex digits does it have?"),
    "q18": ("How does a model's loss change with its size?", "What shape does that curve take with both axes on a log scale?"),
    "q19": ("What has to fit in memory while a network trains?", "Apart from the weights themselves, what else?"),
    "q20": ("In reinforcement learning, an agent takes a step. What then?", "What tells it how good that step was?"),
    "q21": ("How does a classifier that looks at the closest stored examples work?", "How does it pick a class?"),
    "q22": ("How uncertain is the average of a sample?", "How much more data do you need to halve that?"),
    "q23": ("Does a tokenizer cut every language into pieces the same way?", "Does a sentence in Dutch need more or fewer than the same one in English?"),
    "q24": ("What is bfloat16?", "What happens when you add a thousandth to one in it?"),
}
# Two of the chats in Dutch, for the Dutch lesson's text; the course searched in English.
FOLLOW_NL = {
    "q02": ("Wat doet het getelde bigram-model van Tiny Shakespeare?", "Wat is zijn perplexity?"),
    "q18": ("Hoe verandert de loss van een model met zijn grootte?", "Welke vorm heeft die kromme als beide assen logaritmisch zijn?"),
}
# The same questions, short, as someone types them into a search box; the course wrote them.
SHORT = {
    "q01": "small CNN parameters", "q02": "bigram perplexity", "q03": "AlexNet error", "q04": "digit reader accuracy",
    "q05": "Qwen3 remembers Sam", "q06": "OpenWebText size", "q07": "GPT-2 token rules", "q08": "popularity baseline",
    "q09": "penguins PCA k-means", "q10": "Conv2d parameters", "q11": "request time without cache", "q12": "sorting emails",
    "q13": "DBSCAN strays", "q14": "opening specks", "q15": "Grad-CAM cat", "q16": "tokens per parameter",
    "q17": "SHA-256 length", "q18": "loss and model size", "q19": "memory during training", "q20": "agent feedback",
    "q21": "nearest examples classifier", "q22": "halve the uncertainty", "q23": "Dutch tokens", "q24": "bfloat16 tiny addition",
}
REWRITE = ("You rewrite questions for a search engine. Given a conversation, write the last question again so that it "
           "stands on its own, with the names it refers to. Write only the question.")
HYDE = ("Write a short passage from the summary of a course on AI that answers the question. Two sentences. "
        "If you do not know a detail, make it up.")
VERSIONS = ("Write three other ways to ask this question about a course on AI, each on its own line. "
            "Write only the questions.")


def rewrite_messages(qid: str) -> list:
    earlier, follow = FOLLOW[qid]
    return [{"role": "system", "content": REWRITE}, {"role": "user", "content": f"User: {earlier}\nUser: {follow}"}]


def hyde_messages(qid: str) -> list:
    return [{"role": "system", "content": HYDE}, {"role": "user", "content": SHORT[qid]}]


def versions_messages(qid: str) -> list:
    return [{"role": "system", "content": VERSIONS}, {"role": "user", "content": SHORT[qid]}]


RECORD = {}
for _q in ANSWERABLE:
    RECORD[f'{_q["id"]}-rewrite'] = {"messages": rewrite_messages(_q["id"]), "max_tokens": 60, "temperature": 0, "seeds": 1}
    RECORD[f'{_q["id"]}-hyde'] = {"messages": hyde_messages(_q["id"]), "max_tokens": 90, "temperature": 0, "seeds": 1}
    RECORD[f'{_q["id"]}-versions'] = {"messages": versions_messages(_q["id"]), "max_tokens": 90, "temperature": 0, "seeds": 1}


@functools.cache
def course() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-query-rewriting"]


def written(qid: str, kind: str) -> str:
    """What Qwen3 wrote for a question: kind "rewrite", "hyde" or "versions"."""
    return course()[f"{qid}-{kind}"]["answers"][0]["text"].strip()


SOLUTION_VERSIONS = '''def versions(text):
    found = []
    for line in text.split("\\n"):
        line = line.strip()
        if line.startswith("- "):
            line = line[2:]
        elif len(line) > 1 and line[0].isdigit() and line[1] == ".":
            line = line[2:]
        line = line.strip()
        if line != "":
            found.append(line)
    return found
'''
versions = run_code(SOLUTION_VERSIONS).names["versions"]


def to_embed() -> list:
    """Every text the course embeds with Granite for this lesson, each once, in a fixed order: the follow-ups, each
    after its earlier question, the rewrites, the short questions, the passages of HyDE and the versions, as the
    course's versions() reads them."""
    texts = []
    for q in ANSWERABLE:
        earlier, follow = FOLLOW[q["id"]]
        texts += [follow, earlier + " " + follow, written(q["id"], "rewrite"), SHORT[q["id"]], written(q["id"], "hyde")]
        texts += versions(written(q["id"], "versions"))
    return list(dict.fromkeys(texts))


rrf = run_code('''def rrf(rankings, k):
    scores = {}
    for ranking in rankings:
        for place in range(len(ranking)):
            text = ranking[place]
            if text not in scores:
                scores[text] = 0
            scores[text] = scores[text] + 1 / (k + place + 1)
    return scores
''').names["rrf"]   # the solution of build-hybrid-search
meaning_search = summaries.meaning_search   # the solution of build-retrieval-embeddings


def question(qid: str) -> dict:
    return next(q for q in QUESTIONS if q["id"] == qid)


def gold(q: dict) -> str:
    return summaries.text_of(q["gold"])


def number(x: float, lang: str, digits: int = 0) -> str:
    s = f"{x:,.{digits}f}"
    return s if lang == "en" else s.replace(",", " ").replace(".", ",").replace(" ", ".")


def title_of(text: str, lang: str) -> str:
    """The title of the lesson a point comes from."""
    return ABOUT[POINTS.index(text)]["title"][lang]


@functools.cache
def _embeddings() -> dict:
    with np.load(EMBEDDINGS) as data:
        return dict(zip(json.loads(str(data["texts"])), data["vectors"].astype(np.float32)))


class Unknown(KeyError):
    """What embed raises for a text the course recorded no embedding for, so that the check can tell it from any other
    KeyError in the learner's code."""


def embed(text):
    """Granite Embedding's embedding of a text the course recorded for this lesson; Unknown, a KeyError, for any other
    text."""
    try:
        return _embeddings()[text]
    except KeyError:
        raise Unknown("The course has no embedding for this text.") from None


@functools.cache
def ranking(text: str) -> tuple:
    """All points in the order search by meaning gives them for a recorded text."""
    return tuple(meaning_search(embed(text), VECTORS, POINTS, len(POINTS)))


def merged(texts) -> list:
    """All points in the order of the merged rankings of several texts: the highest RRF score first."""
    scores = rrf([list(ranking(t)) for t in texts], RRF_K)
    return [t for s, t in sorted([(s, t) for t, s in scores.items()], reverse=True)]


WAYS = ("full", "follow", "joined", "rewrite", "short", "hyde", "multi")


def asked(qid: str) -> dict:
    """What the course searched with for a question, for each way: a text, or for "multi" the short question with
    its versions. "full" is the question as the course wrote it in full."""
    earlier, follow = FOLLOW[qid]
    return {"full": question(qid)["text"]["en"], "follow": follow, "joined": earlier + " " + follow,
            "rewrite": written(qid, "rewrite"), "short": SHORT[qid], "hyde": written(qid, "hyde"),
            "multi": [SHORT[qid]] + versions(written(qid, "versions"))}


@functools.cache
def places(qid: str) -> dict:
    """The place (from 1) of the right point of a question in the search with each way of asking."""
    q, out = question(qid), {}
    for way, text in asked(qid).items():
        if way == "full":
            order = meaning_search(summaries.vectors("questions", "en")[QUESTIONS.index(q)], VECTORS, POINTS, len(POINTS))
        elif way == "multi":
            order = merged(text)
        else:
            order = list(ranking(text))
        out[way] = order.index(gold(q)) + 1
    return out


def _median(values: list) -> float:
    v = sorted(values)
    return (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2 if len(v) % 2 == 0 else v[len(v) // 2]


@functools.cache
def hybrid_worked() -> dict:
    """The range of the BM25 scores and of the dot products of all points for the question of the worked example of
    build-hybrid-search, as that lesson gives them."""
    q = question("q20")
    index = summaries.bm25_index(POINTS)
    asked = sorted(summaries.words(q["text"]["en"]))
    idfs = {w: index["idf"].get(w, 0.0) for w in asked}
    b = [summaries.bm25(asked, ws, idfs, index["average"]) for ws in index["words"]]
    m = VECTORS @ summaries.vectors("questions", "en")[QUESTIONS.index(q)]
    return {"bm25_low": min(b), "bm25_high": max(b), "meaning_low": float(m.min()), "meaning_high": float(m.max())}


@functools.cache
def findings() -> dict:
    """For the 24 questions a point answers: how often each way of asking puts the right point among the three best;
    for how many the short question with its versions puts it higher and lower than the short question alone; which
    rewrites only copied the earlier question and the follow-up; and the median seconds Qwen3 took to write a
    rewrite, a passage and three versions."""
    f = {"questions": len(ANSWERABLE)}
    for way in WAYS:
        f[way] = sum(places(q["id"])[way] <= K for q in ANSWERABLE)
    f["multi_better"] = sum(places(q["id"])["multi"] < places(q["id"])["short"] for q in ANSWERABLE)
    f["multi_worse"] = sum(places(q["id"])["multi"] > places(q["id"])["short"] for q in ANSWERABLE)
    f["copied"] = [q["id"] for q in ANSWERABLE if written(q["id"], "rewrite") == " ".join(FOLLOW[q["id"]])]
    for kind in ("rewrite", "hyde", "versions"):
        f[f"seconds_{kind}"] = _median([course()[f'{q["id"]}-{kind}']["answers"][0]["seconds"] for q in ANSWERABLE])
    return f


def worked() -> dict:
    earlier, follow = FOLLOW[WORKED]
    alone = list(ranking(follow))
    return {"earlier": earlier, "follow": follow, "earlier_nl": FOLLOW_NL[WORKED][0], "follow_nl": FOLLOW_NL[WORKED][1],
            "rewrite": written(WORKED, "rewrite"), "places": places(WORKED),
            "first": alone[0], "gold": gold(question(WORKED))}


DRIFT = "q14"   # HyDE: a passage about something else
LOST = "q18"    # a rewrite that left out what the follow-up is about


def request_code(qid: str) -> str:
    """The request of the worked example's rewrite as Python code."""
    earlier, follow = FOLLOW[qid]
    return f'''```python
request = {{
    "model": "qwen3-0.6b",
    "messages": [
        {{"role": "system", "content": "You rewrite questions for a search engine. Given a conversation, write the last "
                                      "question again so that it stands on its own, with the names it refers to. "
                                      "Write only the question."}},
        {{"role": "user", "content": "User: {earlier}\\n"
                                    "User: {follow}"}},
    ],
    "max_tokens": 60,
    "temperature": 0,
}}
```'''


_LABEL = "font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)"
_CODE = "font-family:var(--ea-font-code);font-size:12px"


def _swatch(style: str, label: str) -> str:
    return (f'<span style="display:inline-flex;align-items:center;gap:0.25rem;margin-right:0.8rem;{_LABEL}">'
            f'<span style="display:inline-block;width:11px;height:11px;border-radius:2px;{style}"></span>{html.escape(label)}</span>')


_BARS_WORDS = {
    "en": {"title": "How often search by meaning puts the right point among the three best, for {n} questions, by the way it is asked",
           "full": "the question in full", "follow": "the follow-up alone", "joined": "the follow-up after the earlier question",
           "rewrite": "Qwen3's rewrite of the follow-up", "short": "the short question", "hyde": "Qwen3's passage (HyDE)",
           "multi": "the short question and its three versions", "of": "{a} of {n}",
           "group_follow": "A follow-up in a chat", "group_short": "A short question", "model": "written by Qwen3",
           "line": "the line: the question in full"},
    "nl": {"title": "Hoe vaak zoeken op betekenis het goede punt bij de drie beste zet, voor {n} vragen, naar de manier van vragen",
           "full": "de vraag helemaal", "follow": "de vervolgvraag alleen", "joined": "de vervolgvraag na de eerdere vraag",
           "rewrite": "de herschrijving van Qwen3", "short": "de korte vraag", "hyde": "de passage van Qwen3 (HyDE)",
           "multi": "de korte vraag en zijn drie versies", "of": "{a} van de {n}",
           "group_follow": "Een vervolgvraag in een chat", "group_short": "Een korte vraag", "model": "geschreven door Qwen3",
           "line": "de lijn: de vraag helemaal"},
}


def bars_picture(lang: str) -> str:
    """A bar per way of asking: how often the right point is among the three best. The ways Qwen3 wrote in the color
    of model output, the others as ink; the question in full as the line to compare with."""
    w = _BARS_WORDS[lang]
    f = findings()
    n = f["questions"]

    def bar(way: str) -> str:
        color = "var(--ea-pred)" if way in ("rewrite", "hyde", "multi") else "var(--ea-ink-3)"
        width = 100 * f[way] / n
        return (f'<span style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink)">{html.escape(w[way])}</span>'
                f'<span style="display:block;position:relative;height:12px;background:var(--ea-inset);border-radius:2px">'
                f'<span style="position:absolute;left:0;top:0;bottom:0;width:{width:.1f}%;background:{color};border-radius:2px"></span>'
                f'<span style="position:absolute;top:-2px;bottom:-2px;left:{100 * f["full"] / n:.1f}%;border-left:2px solid var(--ea-axis)"></span></span>'
                f'<span style="{_CODE};color:var(--ea-ink)">{html.escape(w["of"].format(a=f[way], n=n))}</span>')

    def group(label: str, ways: tuple) -> str:
        return (f'<div style="{_LABEL};margin:0.5rem 0 0.2rem;font-weight:600">{html.escape(label)}</div>'
                f'<div style="display:grid;grid-template-columns:minmax(0,1fr) minmax(0,10rem) auto;gap:0.3rem 0.6rem;align-items:center">'
                f'{"".join(bar(x) for x in ways)}</div>')
    body = group(w["group_follow"], ("full", "follow", "joined", "rewrite")) + group(w["group_short"], ("full", "short", "hyde", "multi"))
    legend = _swatch("background:var(--ea-pred)", w["model"]) + _swatch("border-left:2px solid var(--ea-axis);border-radius:0;width:0", w["line"])
    title = w["title"].format(n=n)
    return yours.figure(title, body + f'<div style="margin-top:0.3rem">{legend}</div>', alt=title)


# --- exercise 1: versions ------------------------------------------------------------------------------------------------

class Named(str):
    """A text, written in the check's messages by its name."""

    def __new__(cls, text: str, name: str):
        out = super().__new__(cls, text)
        out.name = name
        return out

    def __reduce__(self):   # so that the check can copy it
        return (Named, (str(self), self.name))

    def __repr__(self):
        return self.name


class Listed(list):
    """A list of texts, written in the check's messages by its name."""

    def __init__(self, items, name: str):
        super().__init__(items)
        self.name = name

    def __repr__(self):
        return self.name


def written_named(qid: str) -> Named:
    return Named(written(qid, "versions"), f'WRITTEN["{qid}"]')


VERSIONS_SHOWN = "q07"   # the list of the exercise's example
START_VERSIONS = '''def versions(text):
    found = []
    for line in text.split("\\n"):
        found.append(line)
    return found
'''
KEEPS_DASH = '''def versions(text):
    found = []
    for line in text.split("\\n"):
        line = line.strip()
        if line != "":
            found.append(line)
    return found
'''
NO_STRIP = '''def versions(text):
    found = []
    for line in text.split("\\n"):
        if line.startswith("- "):
            line = line[2:]
        elif len(line) > 1 and line[0].isdigit() and line[1] == ".":
            line = line[2:]
        if line.strip() != "":
            found.append(line)
    return found
'''
KEEPS_EMPTY = SOLUTION_VERSIONS.replace('''        if line != "":
            found.append(line)''', '''        found.append(line)''')
DASH_ONLY = SOLUTION_VERSIONS.replace('''        elif len(line) > 1 and line[0].isdigit() and line[1] == ".":
            line = line[2:]
''', "")
NUMBERED = Named("1. What does the opening of 3 by 3 remove?\n2. How many specks are left on the apples?\n3. Which closing fills the holes?", "NUMBERED")
BLANK_LINES = Named("\n- How big is OpenWebText?  \n\n- How many GB are the ids of nanoGPT?\n\n", "BLANK_LINES")
PLAIN = Named("How long does a request take without the cache?\nWhat does prompt caching save?", "PLAIN")
ONE = Named("  - What is bfloat16?  ", "ONE")


_VERSIONS_WORDS = {
    "en": {"title": "What your versions reads from Qwen3's list of three versions of the short question “{short}”",
           "written": "what Qwen3 wrote, line by line", "yours": "what your versions gives", "course": "the course reads {n}",
           "all": "In Qwen3's {n} lists your versions reads {a} questions; the course's reads {b}.",
           "none": "Your function gave back {got}, not a list of texts."},
    "nl": {"title": "Wat je versions leest uit de lijst van Qwen3 met drie versies van de korte vraag “{short}”",
           "written": "wat Qwen3 schreef, regel voor regel", "yours": "wat je versions geeft", "course": "de course leest er {n}",
           "all": "In de {n} lijsten van Qwen3 leest je versions {a} vragen; die van de course leest er {b}.",
           "none": "Je functie gaf {got} terug, geen lijst met teksten."},
}


def versions_picture(seen) -> str:
    """Qwen3's list of the exercise's example, line by line with its spaces made visible, beside what the learner's
    versions gives: each text as model output that the course reads too, as a mistake where it differs."""
    w = _VERSIONS_WORDS[seen.lang]
    mine = seen.mine
    fn = mine.names.get("versions") if mine is not None else None
    if not callable(fn):
        return ""
    raw = written(VERSIONS_SHOWN, "versions")
    title = w["title"].format(short=SHORT[VERSIONS_SHOWN])
    try:
        got = with_time_limit(lambda: fn(raw))
    except Exception:   # the learner's code; the check says what went wrong
        return ""
    if not isinstance(got, list) or not all(isinstance(t, str) for t in got):
        return yours.figure(title, f'<div style="{_LABEL}">{html.escape(w["none"].format(got=repr(got)[:80]))}</div>', alt=title)
    want = versions(raw)
    lines = "".join(f'<div style="{_CODE};color:var(--ea-data-text);white-space:pre-wrap;overflow-wrap:anywhere">{html.escape(repr(line))}</div>'
                    for line in raw.split("\n"))
    items = "".join(f'<div style="{_CODE};white-space:pre-wrap;overflow-wrap:anywhere;color:var(--ea-{"pred-text" if t in want else "loss-text"})">{html.escape(repr(t))}</div>'
                    for t in got)
    body = (f'<div style="display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:0.3rem 1rem">'
            f'<div><div style="{_LABEL};margin-bottom:0.2rem">{html.escape(w["written"])}</div>{lines}</div>'
            f'<div><div style="{_LABEL};margin-bottom:0.2rem">{html.escape(w["yours"])}</div>{items}</div></div>')
    total, ok = 0, True
    for q in ANSWERABLE:
        try:
            out = with_time_limit(lambda: fn(written(q["id"], "versions")))
        except Exception:   # the learner's code
            ok = False
            break
        if not isinstance(out, list):
            ok = False
            break
        total += len(out)
    if ok:
        body += f'<div style="{_LABEL};margin-top:0.4rem">{html.escape(w["all"].format(n=len(ANSWERABLE), a=total, b=sum(len(versions(written(q["id"], "versions"))) for q in ANSWERABLE)))}</div>'
    return yours.figure(title, body, alt=title)


# --- exercise 2: multi_search --------------------------------------------------------------------------------------------

MULTI_SHOWN = "q23"   # the short question of the exercise's example
START_MULTI = '''def multi_search(texts, n):
    rankings = []
    for text in texts:
        rankings.append(meaning_search(embed(text), VECTORS, POINTS, len(POINTS)))
    return rankings[0][:n]
'''
SOLUTION_MULTI = START_MULTI.replace('''    return rankings[0][:n]
''', '''    scores = rrf(rankings, 60)
    best = sorted([(score, text) for text, score in scores.items()], reverse=True)
    return [text for score, text in best[:n]]
''')
LOWEST_MULTI = SOLUTION_MULTI.replace("reverse=True)", ")")
LAST_ONLY = SOLUTION_MULTI.replace("rrf(rankings, 60)", "rrf([rankings[-1]], 60)")
AVERAGED = '''def multi_search(texts, n):
    total = 0
    for text in texts:
        total = total + embed(text)
    return meaning_search(total / len(texts), VECTORS, POINTS, n)
'''
MULTI_NAMES = {"embed": embed, "meaning_search": meaning_search, "rrf": rrf, "POINTS": POINTS, "VECTORS": VECTORS}


def queries(qid: str) -> Listed:
    return Listed(asked(qid)["multi"], f'QUERIES["{qid}"]')


_MULTI_WORDS = {
    "en": {"title": "The three best your multi_search gives for “{short}” and its three versions",
           "lesson": "point of the lesson", "by": "place by each of the {n} texts", "right": "the right point",
           "all": "With your multi_search, the short question and its versions put the right point among the three best for {a} of the {n} questions; the short question alone, {b}.",
           "none": "Your function gave back {got}, not a list of points."},
    "nl": {"title": "De drie beste die je multi_search geeft voor “{short}” en zijn drie versies",
           "lesson": "punt van de les", "by": "plaats bij elk van de {n} teksten", "right": "het goede punt",
           "all": "Met je multi_search zetten de korte vraag en zijn versies het goede punt bij de drie beste voor {a} van de {n} vragen; de korte vraag alleen voor {b}.",
           "none": "Je functie gaf {got} terug, geen lijst met punten."},
}


def multi_picture(seen) -> str:
    """The three points the learner's multi_search gives for the exercise's example, each with its place by each of
    the four texts, the right point marked as input data; and over all questions how often it puts the right point
    among the three best, against the short question alone."""
    w = _MULTI_WORDS[seen.lang]
    mine = seen.mine
    fn = mine.names.get("multi_search") if mine is not None else None
    if not callable(fn):
        return ""
    texts = asked(MULTI_SHOWN)["multi"]
    title = w["title"].format(short=SHORT[MULTI_SHOWN])
    try:
        got = with_time_limit(lambda: fn(queries(MULTI_SHOWN), K))
    except Exception:   # the learner's code; the check says what went wrong
        return ""
    if not isinstance(got, list) or not all(isinstance(t, str) and t in POINTS for t in got):
        return yours.figure(title, f'<div style="{_LABEL}">{html.escape(w["none"].format(got=repr(got)[:80]))}</div>', alt=title)
    g = gold(question(MULTI_SHOWN))
    rows = []
    for t in got[:K]:
        mark = "border-left:3px solid var(--ea-data);padding-left:0.3rem" if t == g else "padding-left:calc(0.3rem + 3px)"
        where = ", ".join(str(ranking(x).index(t) + 1) for x in texts)
        rows.append(f'<span style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink);{mark};overflow-wrap:anywhere">'
                    f'<em>{html.escape(title_of(t, seen.lang))}</em>: {html.escape(t[:150])}{"…" if len(t) > 150 else ""}</span>'
                    f'<span style="{_CODE};color:var(--ea-ink-2)">{where}</span>')
    head = f'<span style="{_LABEL}">{html.escape(w["lesson"])}</span><span style="{_LABEL}">{html.escape(w["by"].format(n=len(texts)))}</span>'
    body = f'<div style="display:grid;grid-template-columns:minmax(0,1fr) auto;gap:0.25rem 0.8rem;align-items:baseline">{head}{"".join(rows)}</div>'
    n = 0
    for q in ANSWERABLE:
        try:
            out = with_time_limit(lambda: fn(queries(q["id"]), K))
        except Exception:   # the learner's code
            n = None
            break
        if not isinstance(out, list):
            n = None
            break
        n += gold(q) in out[:K]
    if n is not None:
        body += f'<div style="{_LABEL};margin-top:0.4rem">{html.escape(w["all"].format(a=n, n=len(ANSWERABLE), b=findings()["short"]))}</div>'
    body += f'<div style="margin-top:0.3rem">{_swatch("background:var(--ea-data)", w["right"])}</div>'
    return yours.figure(title, body, alt=title)


_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}


def _show_texts(v):
    """A list of texts as the check's messages show it: the start of each."""
    if isinstance(v, list) and all(isinstance(t, str) for t in v):
        return [t if len(t) <= 50 else t[:50] + "..." for t in v]
    return v


def _key_error(e) -> bool:
    return isinstance(e, Unknown)


EXERCISES = {
    "versions": FunctionBox(
        name="versions",
        picture=versions_picture,
        function="versions",
        start=START_VERSIONS,
        solution=SOLUTION_VERSIONS,
        show=_show_texts,
        visible=[(written_named(VERSIONS_SHOWN),)],
        hidden=[(written_named("q11"),), (NUMBERED,), (BLANK_LINES,), (PLAIN,), (ONE,), (Named("", '""'),)],
        mistakes=[
            Mistake("keeps-dash", wrong=KEEPS_DASH,
                    en="`{call}` gives `{got}`: each text still starts with the `- ` of the list. Cut it off: `line[2:]` when the line starts with `\"- \"`.",
                    nl="`{call}` geeft `{got}`: elke tekst begint nog met de `- ` van de lijst. Knip die eraf: `line[2:]` als de regel met `\"- \"` begint."),
            Mistake("spaces", wrong=NO_STRIP,
                    en="`{call}` gives `{got}`: the texts keep the spaces around them, such as the two at the end of each line Qwen3 wrote. `line.strip()` takes them off, before and after you cut off the `- `.",
                    nl="`{call}` geeft `{got}`: de teksten houden de spaties eromheen, zoals de twee aan het eind van elke regel die Qwen3 schreef. `line.strip()` haalt ze weg, voor en na het afknippen van de `- `."),
            Mistake("keeps-empty", wrong=KEEPS_EMPTY,
                    en="`{call}` gives `{got}`: an empty line becomes an empty text, and a search with it finds nothing useful. Add a text only when it is not `\"\"`.",
                    nl="`{call}` geeft `{got}`: een lege regel wordt een lege tekst, en een zoekfunctie vindt daarmee niets bruikbaars. Voeg een tekst alleen toe als hij niet `\"\"` is."),
            Mistake("dash-only", wrong=DASH_ONLY, state="almost",
                    en="Almost. `{call}` gives `{got}`: a numbered list keeps its `1.` in front. Cut off a digit with a point too, as the task says.",
                    nl="Bijna. `{call}` geeft `{got}`: een genummerde lijst houdt zijn `1.` vooraan. Knip ook een cijfer met een punt eraf, zoals de opdracht zegt."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`versions` gives every line as it is, with the `- ` in front, the spaces around it and empty lines too. Strip each line, cut off the `- ` or the number, and skip what is empty.",
                      "nl": "`versions` geeft elke regel zoals hij is, met de `- ` vooraan, de spaties eromheen en ook lege regels. Strip elke regel, knip de `- ` of het nummer eraf, en sla over wat leeg is."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: each question on its own, without the `- ` or the number in front, without spaces around it, and no empty texts.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: elke vraag apart, zonder de `- ` of het nummer vooraan, zonder spaties eromheen, en geen lege teksten."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for a numbered list, for empty lines, for lines without a `- `, and for an empty text?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Werkt je functie voor een genummerde lijst, voor lege regels, voor regels zonder `- `, en voor een lege tekst?"},
            "no_function": {"en": "Your box does not make a function `versions`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `versions`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `versions` is right in all {n} cases: each question on its own, as a search can use it.",
                        "nl": "Je `versions` klopt in alle {n} gevallen: elke vraag apart, zoals een zoekfunctie hem kan gebruiken."},
        },
    ),
    "multi_search": FunctionBox(
        name="multi_search",
        picture=multi_picture,
        function="multi_search",
        start=START_MULTI,
        solution=SOLUTION_MULTI,
        show=_show_texts,
        names=MULTI_NAMES,
        visible=[(queries(MULTI_SHOWN), K)],
        hidden=[(queries("q11"), K), (queries("q02"), 1), (queries("q09"), 5),
                (Listed([SHORT["q20"], written("q20", "hyde")], 'SHORT_AND_HYDE["q20"]'), K),
                (Listed([SHORT["q18"]], 'ONLY_SHORT["q18"]'), 2)],
        mistakes=[
            Mistake("lowest-first", wrong=LOWEST_MULTI,
                    en="`{call}` gives the points with the lowest RRF score first. `sorted` puts the smallest first; with `reverse=True` the highest comes first.",
                    nl="`{call}` geeft de punten met de laagste RRF-score eerst. `sorted` zet de kleinste vooraan; met `reverse=True` komt de hoogste eerst."),
            Mistake("last-only", wrong=LAST_ONLY,
                    en="`{call}` gives the best of the last text only. Give `rrf` all rankings: `rrf(rankings, 60)`.",
                    nl="`{call}` geeft alleen de beste van de laatste tekst. Geef `rrf` alle ranglijsten: `rrf(rankings, 60)`."),
            Mistake("averaged", wrong=AVERAGED,
                    en="`{call}` gives `{got}`: you search with the average of the embeddings. That is another method, and it gives another order. Here every text gets its own ranking, and `rrf` merges them.",
                    nl="`{call}` geeft `{got}`: je zoekt met het gemiddelde van de embeddings. Dat is een andere methode, en die geeft een andere volgorde. Hier krijgt elke tekst een eigen ranglijst, en `rrf` voegt ze samen."),
            Mistake("unknown", error=_key_error,
                    en="With `{call}`, `embed` gets a text the course has no embedding for. Give `embed` each text of `texts` as it is.",
                    nl="Met `{call}` krijgt `embed` een tekst waar de course geen embedding van heeft. Geef `embed` elke tekst van `texts` zoals hij is."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`multi_search` makes a ranking for every text, but gives the best of the first only. Merge the rankings with `rrf`, and give the `n` points with the highest score.",
                      "nl": "`multi_search` maakt een ranglijst voor elke tekst, maar geeft alleen de beste van de eerste. Voeg de ranglijsten samen met `rrf`, en geef de `n` punten met de hoogste score."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: the `n` points with the highest RRF score over the rankings of all texts, with `k = 60`, the highest first.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: de `n` punten met de hoogste RRF-score over de ranglijsten van alle teksten, met `k = 60`, de hoogste eerst."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for another `n`, for two texts, and for one?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Werkt je functie voor een andere `n`, voor twee teksten, en voor één?"},
            "no_function": {"en": "Your box does not make a function `multi_search`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `multi_search`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `multi_search` is right in all {n} cases: one ranking per text, merged with reciprocal rank fusion.",
                        "nl": "Je `multi_search` klopt in alle {n} gevallen: één ranglijst per tekst, samengevoegd met reciprocal rank fusion."},
        },
    ),
}

REQUIRED = ["versions", "multi_search", "hyde-drift", "lost-subject"]

CASES = [
    ("versions", START_VERSIONS, "empty", None),
    ("versions", "", "empty", None),
    ("versions", SOLUTION_VERSIONS, "correct", None),
    ("versions", START_VERSIONS.replace('''        found.append(line)
''', '''        line = line.strip().removeprefix("- ")
        if line[:1].isdigit() and line[1:2] == ".":
            line = line[2:].strip()
        if line:
            found.append(line)
'''), "correct", None),
    ("versions", KEEPS_DASH, "wrong", "keeps-dash"),
    ("versions", NO_STRIP, "wrong", "spaces"),
    ("versions", KEEPS_EMPTY, "wrong", "keeps-empty"),
    ("versions", DASH_ONLY, "almost", "dash-only"),
    ("versions", SOLUTION_VERSIONS.replace('text.split("\\n")', 'text.split("  \\n")'), "almost", None),
    ("multi_search", START_MULTI, "empty", None),
    ("multi_search", "", "empty", None),
    ("multi_search", SOLUTION_MULTI, "correct", None),
    ("multi_search", START_MULTI.replace('''    return rankings[0][:n]
''', '''    scores = rrf(rankings, 60)
    return sorted(scores, key=scores.get, reverse=True)[:n]
'''), "correct", None),
    ("multi_search", LOWEST_MULTI, "wrong", "lowest-first"),
    ("multi_search", LAST_ONLY, "wrong", "last-only"),
    ("multi_search", AVERAGED, "wrong", "averaged"),
    ("multi_search", START_MULTI.replace("embed(text)", "embed(text.lower())"), "error", "unknown"),
    ("multi_search", SOLUTION_MULTI.replace("best = sorted(", 'scores["total"] = scores["total"] + 1\n    best = sorted('), "error", None),
    ("multi_search", SOLUTION_MULTI.replace("best[:n]", "best[:3]"), "almost", None),
]


# --- the search to try ---------------------------------------------------------------------------------------------------

def question_options(lang: str) -> dict:
    return {q["text"][lang]: q["id"] for q in ANSWERABLE}


def learner_multi(versions_text, multi_text):
    """The learner's versions and multi_search, when both are right; else None."""
    for name, text in (("versions", versions_text), ("multi_search", multi_text)):
        if text is MISSING or text is None or EXERCISES[name].check(text).state != "correct":
            return None
    return run_code(versions_text).names["versions"], run_code(multi_text, dict(MULTI_NAMES)).names["multi_search"]


_TRY_WORDS = {
    "en": {"title": "Asked six ways: {question}", "way": "asked as", "text": "what the search got", "place": "place of the right point",
           "follow": "the follow-up alone", "joined": "after the earlier question", "rewrite": "Qwen3's rewrite",
           "short": "the short question", "hyde": "Qwen3's passage (HyDE)", "multi": "short question and versions, with your functions",
           "failed": "Your functions stopped: {why}"},
    "nl": {"title": "Op zes manieren gevraagd: {question}", "way": "gevraagd als", "text": "wat de zoekfunctie kreeg", "place": "plaats van het goede punt",
           "follow": "de vervolgvraag alleen", "joined": "na de eerdere vraag", "rewrite": "de herschrijving van Qwen3",
           "short": "de korte vraag", "hyde": "de passage van Qwen3 (HyDE)", "multi": "korte vraag en versies, met je functies",
           "failed": "Je functies stopten: {why}"},
}


def tried(fns, qid: str, lang: str) -> str:
    """For a question: each way of asking, the text the search got and the place of the right point; the last way with
    the learner's own versions and multi_search."""
    w = _TRY_WORDS[lang]
    versions_fn, multi_fn = fns
    q, a, p = question(qid), asked(qid), places(qid)
    try:
        mine = with_time_limit(lambda: versions_fn(written(qid, "versions")))
        found = with_time_limit(lambda: multi_fn([SHORT[qid]] + list(mine), len(POINTS)))
        place = found.index(gold(q)) + 1 if gold(q) in found else None
    except Exception as e:   # the learner's code
        return f'<p><em>{html.escape(w["failed"].format(why=f"{type(e).__name__}: {e}"))}</em></p>'
    rows = []
    for way in ("follow", "joined", "rewrite", "short", "hyde", "multi"):
        text = " | ".join([SHORT[qid]] + list(mine)) if way == "multi" else a[way]
        at = place if way == "multi" else p[way]
        color = "pred-text" if way in ("rewrite", "hyde", "multi") else "ink"
        rows.append(f'<span style="{_LABEL}">{html.escape(w[way])}</span>'
                    f'<span style="{_CODE};color:var(--ea-{color});overflow-wrap:anywhere">{html.escape(text)}</span>'
                    f'<span style="{_CODE};color:var(--ea-ink);text-align:right;{"font-weight:600" if at is not None and at <= K else ""}">{"" if at is None else at}</span>')
    head = "".join(f'<span style="{_LABEL}">{html.escape(w[k])}</span>' for k in ("way", "text", "place"))
    body = f'<div style="display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:0.3rem 0.8rem;align-items:baseline">{head}{"".join(rows)}</div>'
    title = w["title"].format(question=q["text"][lang])
    return yours.figure(title, body, alt=title)
