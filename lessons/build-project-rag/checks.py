# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-project-rag: app/library.py, with make_library, which embeds the chunks of documents through the
learner's embed of build-embeddings-api, BATCH at a time, and keeps them with their embeddings; app/rag.py, with ask,
which searches the library with meaning_search of build-retrieval-embeddings, asks the model through chat of
build-project-call with with_sources of build-rag-idea and a grounded system prompt, and keeps the sources the answer
names with cited of build-grounded-answers; both against a stand-in of the server. The course's recorded runs
(data/course-rag-qwen.json): the questions about the handbook of Spoke & Chain, with the course's ask and without the
handbook; the pictures, and the demo with the learner's own modules on the documents in work/projects/app/documents/."""
import functools
import html
import json
import shutil
import time

import numpy as np

from everything_ai import apicheck, chat, embed, lesson as lesson_helpers, paths, projects, serve, summaries, yours
from everything_ai.apicheck import URL, ApiBox, Case
from everything_ai.check import Mistake, run_code

FILES = list(chat.FILES) + [(embed.REPO, name) for name in embed.FILES]   # what the demo downloads
DOWNLOADS = {"Qwen3-0.6B": FILES}
DOWNLOAD_LINE = {
    "en": f"Works on every computer. The chat model Qwen3-0.6B and the embedding model Granite Embedding are downloaded once from Hugging Face, {lesson_helpers.download_size(FILES, 'en')}, when you click their button in the lesson; after that, the lesson works without a connection. The exercises work without them too.",
    "nl": f"Werkt op elke computer. Het chat model Qwen3-0.6B en het embedding model Granite Embedding worden één keer gedownload van Hugging Face, {lesson_helpers.download_size(FILES, 'nl')}, als je in de les op de knop ervoor klikt; daarna werkt de les zonder verbinding. De oefeningen werken ook zonder die twee.",
}

# ask uses the learner's modules of five earlier lessons, and the demo cuts documents with their chunk; where one is
# missing or does not pass, the course's version stands in (template 1.3). Their app/library.py of exercise 1 is not
# among them: the demo uses it as it is.
projects.builds_on(("app", "client", "build-project-call", "chat"), ("app", "embeddings", "build-embeddings-api", "embed"),
                   ("app", "meaning", "build-retrieval-embeddings", "meaning_search"),
                   ("app", "sources", "build-rag-idea", "with_sources"), ("app", "citations", "build-grounded-answers", "cited"),
                   ("app", "chunks", "build-chunking", "chunk"))

ABOUT = "the bike shop Spoke & Chain"
SYSTEM = "You answer questions about the bike shop Spoke & Chain. Answer in one sentence."   # without the handbook
GROUNDED = ("You answer questions about {about}. Answer in one sentence, using only the numbered sources. "
            "End the sentence with the number of the source it comes from, in square brackets, such as [2]. "
            "If the sources do not answer the question, answer only: I don't know.")
K = 3
SEEDS = 2
MAX_TOKENS = 200   # what the learner's chat of build-project-call asks for when the call gives none
BATCH = 64         # texts in one request for embeddings, as the course embedded its summaries
with_sources = run_code('''def with_sources(question, passages):
    if len(passages) == 0:
        return question
    text = "Use these sources to answer the question.\\n\\n"
    n = 1
    for passage in passages:
        text = text + f"[{n}] {passage}\\n"
        n = n + 1
    return text + "\\nQuestion: " + question
''').names["with_sources"]

CHUNKS = summaries.handbook_chunks()
VECTORS = summaries.handbook_vectors("chunks")
QUESTIONS = summaries.handbook_questions()
ANSWERABLE = [q for q in QUESTIONS if q["kind"] != "none"]
NONE = summaries.handbook_questions("none")


def question(qid: str) -> dict:
    return next(q for q in QUESTIONS if q["id"] == qid)


def question_vector(q: dict):
    return summaries.handbook_vectors("questions")[QUESTIONS.index(q)]


def found(q: dict) -> list:
    """The K chunks of the handbook whose embeddings have the highest dot product with the question's."""
    return summaries.meaning_search(question_vector(q), VECTORS, CHUNKS, K)


def messages(q: dict, way: str) -> list:
    """What the course sent for a question: "plain", the question alone after a system prompt about the shop; "yours",
    what the course's ask sends, the grounded system prompt and the question with the chunks search by meaning finds."""
    if way == "plain":
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q["text"]["en"]}]
    return [{"role": "system", "content": GROUNDED.format(about=ABOUT)}, {"role": "user", "content": with_sources(q["text"]["en"], found(q))}]


RECORD = {f"{q['id']}-{way}": {"messages": messages(q, way), "max_tokens": MAX_TOKENS, "seeds": SEEDS}
          for q in QUESTIONS for way in ("plain", "yours")}
DATA = paths.ROOT / "data" / "course-rag-qwen.json"


@functools.cache
def course() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-project-rag"]


def answers(qid: str, way: str) -> list:
    return [a["text"] for a in course()[f"{qid}-{way}"]["answers"]]


def holds(chunk: str, q: dict) -> bool:
    """Whether a chunk holds the words of the handbook that answer the question."""
    return q["fact"] is not None and q["fact"] in chunk


def hit(q: dict) -> bool:
    """Whether search by meaning puts a chunk that holds the answer among the K it finds."""
    return any(holds(c, q) for c in found(q))


def refuses(answer: str) -> bool:
    return "i don't know" in answer.lower()


def good(answer: str, q: dict) -> bool:
    """A right answer to a question the handbook answers; to one it does not answer, one that says it cannot answer."""
    if q["kind"] == "none":
        return refuses(answer) or summaries.says_unknown(answer)
    return summaries.right(answer, q)


def named(answer: str, passages: list) -> list:
    """The course's named: the passages an answer cites, in its order, for numbers that are a passage."""
    return [passages[n - 1] for n in summaries.cited(answer) if 1 <= n <= len(passages)]


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


@functools.cache
def findings() -> dict:
    """For the questions the handbook answers, two runs each: how many answers were right without the handbook and with
    the course's ask, and in how many runs the course's ask named a chunk that holds the answer; for the questions
    it does not answer, how many answers said they cannot answer; and in how many questions search by meaning found
    a chunk that holds the answer."""
    f = {"questions": len(ANSWERABLE), "runs": SEEDS * len(ANSWERABLE), "none": len(NONE), "none_runs": SEEDS * len(NONE),
         "hits": sum(hit(q) for q in ANSWERABLE), "chunks": len(CHUNKS), "words": len(summaries.handbook().split())}
    for way in ("plain", "yours"):
        f[way] = sum(good(a, q) for q in ANSWERABLE for a in answers(q["id"], way))
        f[way + "_none"] = sum(good(a, q) for q in NONE for a in answers(q["id"], way))
    f["refused"] = sum(refuses(a) for q in ANSWERABLE for a in answers(q["id"], "yours"))
    f["missed"] = [q["id"] for q in ANSWERABLE if not hit(q)]
    f["missed_refused"] = sum(refuses(a) for q in ANSWERABLE if not hit(q) for a in answers(q["id"], "yours"))
    f["right_found"] = sum(good(a, q) for q in ANSWERABLE if hit(q) for a in answers(q["id"], "yours"))
    f["runs_found"] = SEEDS * f["hits"]
    for q in ANSWERABLE:   # whether an answer named a chunk that holds the answer, for right and for other answers
        for a in answers(q["id"], "yours"):
            key = ("right" if good(a, q) else "other") + ("_named" if any(holds(c, q) for c in named(a, found(q))) else "_unnamed")
            f[key] = f.get(key, 0) + 1
    # search by words: whether BM25 of build-bm25 puts a chunk that holds the answer first, where search by meaning missed
    f["bm25_first"] = [qid for qid in f["missed"] if holds(summaries.bm25_search(question(qid)["text"]["en"], CHUNKS, 1)[0], question(qid))]
    return f


def batch_seconds() -> float:
    """How long the course took to embed the 1,266 English points of its summaries, 64 at a time, on its computer."""
    return summaries.vectors_about()["seconds"]["points"]


# --- pictures ----------------------------------------------------------------------------------------------------------

_LABEL = "font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)"


def _mono(text: str, colour: str = "var(--ea-ink)") -> str:
    return (f'<span style="font-family:var(--ea-font-code);font-size:12px;color:{colour};white-space:pre-wrap;'
            f'overflow-wrap:anywhere">{html.escape(text)}</span>')


def _ui(text: str, colour: str = "var(--ea-ink-2)", size: int = 12, weight: int = 400) -> str:
    return (f'<span style="font-family:var(--ea-font-ui);font-size:{size}px;color:{colour};font-weight:{weight}">'
            f'{html.escape(text)}</span>')


_FILES_WORDS = {
    "en": {"title": "Your assistant's files after today", "today": "new today", "uses": "uses",
           "library": "make_library: embeds the chunks, BATCH at a time", "rag": "ask: searches, asks, and names the sources",
           "client": ("chat", "Project: your assistant answers"), "embeddings": ("embed", "Embeddings and documents through an API"),
           "meaning": ("meaning_search", "Search by meaning"), "sources": ("with_sources", "Look it up first: what RAG is"),
           "citations": ("cited", "Answers from the sources"), "chunks": ("chunk", "Cutting documents into pieces"),
           "demo": "the demo cuts your documents with it"},
    "nl": {"title": "De bestanden van je assistent na vandaag", "today": "nieuw vandaag", "uses": "gebruikt",
           "library": "make_library: maakt de embeddings van de chunks, BATCH tegelijk", "rag": "ask: zoekt, vraagt, en noemt de bronnen",
           "client": ("chat", "Project: je assistent antwoordt"), "embeddings": ("embed", "Embeddings en documenten via een API"),
           "meaning": ("meaning_search", "Zoeken op betekenis"), "sources": ("with_sources", "Eerst opzoeken: wat RAG is"),
           "citations": ("cited", "Antwoorden uit de bronnen"), "chunks": ("chunk", "Documenten in stukken knippen"),
           "demo": "de demo knipt je documenten ermee"},
}


def files_picture(lang: str) -> str:
    """The files of the package app this lesson uses: the two new ones, and the six earlier ones with the lesson each
    came from. Ink only: the picture is about the code, not about what a model makes."""
    w = _FILES_WORDS[lang]

    def box(name: str, what: str, note: str, now: bool) -> str:
        border = "2px solid var(--ea-ink)" if now else "1px solid var(--ea-line)"
        return (f'<div style="flex:1 1 11rem;min-width:0;border:{border};border-radius:8px;padding:0.4rem 0.55rem">'
                f'<div>{_mono("app/" + name + ".py")}</div><div style="margin-top:0.15rem">{_ui(what, "var(--ea-ink)")}</div>'
                f'<div style="margin-top:0.1rem">{_ui(note, size=11)}</div></div>')

    top = box("library", w["library"], w["today"], True) + box("rag", w["rag"], w["today"], True)
    used = "".join(box(name, w[name][0], w[name][1], False) for name in ("embeddings", "meaning", "sources", "client", "citations"))
    chunks = box("chunks", w["chunks"][0], w["chunks"][1] + "; " + w["demo"], False)
    arrow = f'<div style="margin:0.3rem 0 0.3rem 1rem">{_ui("↓ " + w["uses"], size=12)}</div>'
    body = (f'<div style="display:flex;flex-wrap:wrap;gap:0.5rem">{top}</div>{arrow}'
            f'<div style="display:flex;flex-wrap:wrap;gap:0.5rem">{used}{chunks}</div>')
    return yours.figure(w["title"], body, alt=w["title"])


def _swatch(color: str, label: str) -> str:
    return (f'<span style="display:inline-flex;align-items:center;gap:0.25rem;margin-right:0.8rem;{_LABEL}">'
            f'<span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:var(--ea-{color})"></span>{html.escape(label)}</span>')


def _square(ok: bool, text: str) -> str:
    return (f'<span title="{html.escape(" ".join(text.split())[:160])}" style="display:inline-block;width:12px;height:12px;'
            f'margin-right:2px;border-radius:2px;background:var(--ea-{"pred" if ok else "loss"})"></span>')


_RUNS_WORDS = {
    "en": {"title": "Qwen3's answers about the handbook of Spoke & Chain: without the handbook, and with your ask. Two runs each",
           "question": "question", "found": "chunk with the answer", "plain": "without", "yours": "your ask", "yes": "found",
           "no": "not found", "good": "right, or says it cannot answer where the handbook does not say", "bad": "not right",
           "count": "{k} of {n}", "none": "Questions the handbook does not answer", "meaning": "asks it in other words"},
    "nl": {"title": "De antwoorden van Qwen3 over het handboek van Spoke & Chain: zonder het handboek, en met je ask. Elk twee runs",
           "question": "vraag", "found": "chunk met het antwoord", "plain": "zonder", "yours": "je ask", "yes": "gevonden",
           "no": "niet gevonden", "good": "goed, of zegt dat het niet kan antwoorden waar het handboek het niet zegt", "bad": "niet goed",
           "count": "{k} van de {n}", "none": "Vragen die het handboek niet beantwoordt", "meaning": "vraagt het in andere woorden"},
}


def runs_picture(lang: str) -> str:
    """A row per question: whether search by meaning found a chunk that holds the answer, as input data, then a square
    per run without the handbook and with the course's ask, in the color of model output when the answer is good and
    as a mistake when not; the questions the handbook does not answer apart."""
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
        cells = "".join(f'<span>{"".join(_square(good(a, q), a) for a in answers(q["id"], way))}</span>' for way in ("plain", "yours"))
        return f'<span style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink)">{html.escape(q["text"][lang])}{mark}</span>{found_cell}{cells}'

    def foot(k_plain: int, k_yours: int, n: int) -> str:
        return (f'<span></span><span></span><span style="{_LABEL}">{html.escape(w["count"].format(k=k_plain, n=n))}</span>'
                f'<span style="{_LABEL}">{html.escape(w["count"].format(k=k_yours, n=n))}</span>')
    head = "".join(f'<span style="{_LABEL}">{html.escape(w[k])}</span>' for k in ("question", "found", "plain", "yours"))
    gap = f'<span style="grid-column:1 / -1;{_LABEL};margin-top:0.4rem;border-top:1px solid var(--ea-line);padding-top:0.3rem">{html.escape(w["none"])}</span>'
    grid = (f'<div style="display:grid;grid-template-columns:minmax(0,1fr) auto auto auto;gap:0.3rem 0.8rem;align-items:center">'
            f'{head}{"".join(row(q) for q in ANSWERABLE)}{foot(f["plain"], f["yours"], f["runs"])}'
            f'{gap}{"".join(row(q) for q in NONE)}{foot(f["plain_none"], f["yours_none"], f["none_runs"])}</div>')
    legend = _swatch("data", w["yes"]) + _swatch("pred", w["good"]) + _swatch("loss", w["bad"])
    return yours.figure(w["title"], grid + f'<div style="margin-top:0.4rem">{legend}</div>', alt=w["title"])


def _source_blocks(passages: list, marked: list, where: list | None = None, limit: int = 260) -> str:
    """The sources as numbered blocks, the ones the answer names marked as data; `where`, the file of each."""
    parts = []
    for n, p in enumerate(passages, 1):
        on = p in marked
        text = p if len(p) <= limit else p[:limit - 1] + "…"
        head = f"[{n}]" + (f" {where[n - 1]}" if where else "")
        parts.append(f'<div style="margin-top:0.25rem;padding:0.3rem 0.5rem;border-radius:6px;border-left:3px solid var(--ea-{"data" if on else "line"});'
                     f'background:var(--ea-{"data-wash" if on else "inset"})">{_ui(head, "var(--ea-ink)" if on else "var(--ea-ink-2)", 12, 600 if on else 400)} '
                     f'{_mono(text, "var(--ea-ink)" if on else "var(--ea-ink-2)")}</div>')
    return "".join(parts)


# --- exercise 1: app/library.py ------------------------------------------------------------------------------------------

START_LIBRARY = '''import numpy as np

from app.embeddings import embed

BATCH = 64


def make_library(url, chunks, about):
    vectors = embed(url, chunks)
    return {"about": about, "chunks": chunks, "vectors": np.array(vectors)}
'''
SOLUTION_LIBRARY = START_LIBRARY.replace('''    vectors = embed(url, chunks)
''', '''    vectors = []
    for start in range(0, len(chunks), BATCH):
        vectors = vectors + embed(url, chunks[start:start + BATCH])
''')
LOST = SOLUTION_LIBRARY.replace("vectors = vectors + embed(", "vectors = embed(")
NESTED = SOLUTION_LIBRARY.replace("vectors = vectors + embed(url, chunks[start:start + BATCH])",
                                  "vectors.append(embed(url, chunks[start:start + BATCH]))")
LONG = [f"Part {i + 1} of a long handbook, cut into chunks." for i in range(150)]


def _embedded(log: list) -> list:
    """The lists of texts in the requests for embeddings the server got."""
    return [e["body"].get("input") for e in log
            if e["method"] == "POST" and e["path"].endswith("/embeddings") and isinstance(e["body"], dict)]


def _judge_library(chunks: list, about: str):
    want = np.array([serve.made_up(t) for t in chunks])

    def judge(got, running):
        sent = _embedded(running.log)
        if any(isinstance(s, list) and len(s) > BATCH for s in sent):
            return "one-request"
        if any(not isinstance(s, list) for s in sent):
            return "one-text"
        if len(sent) > -(-len(chunks) // BATCH):   # more requests than full batches of BATCH need
            return "small-batches"
        if not isinstance(got, dict) or "vectors" not in got:
            return "return"
        vectors = got["vectors"]
        if not isinstance(vectors, np.ndarray):
            return "not-array"
        if vectors.ndim == 3:
            return "nested"
        if vectors.ndim == 2 and len(vectors) < len(chunks):
            return "lost"
        if vectors.shape != want.shape or not np.allclose(vectors, want, atol=1e-5):
            return "vectors"
        if got.get("about") != about or got.get("chunks") != chunks or set(got) != {"about", "chunks", "vectors"}:
            return "return"
        return None
    return judge


LIBRARY_CASES = [
    Case(serve.StandIn, (URL, LONG, ABOUT), _judge_library(LONG, ABOUT),
         shown_args=("url", {"en": "[150 chunks]", "nl": "[150 chunks]"}, repr(ABOUT))),
    Case(serve.StandIn, (URL, LONG[:BATCH], "a handbook"), _judge_library(LONG[:BATCH], "a handbook"),
         shown_args=("url", {"en": "[64 chunks]", "nl": "[64 chunks]"}, '"a handbook"')),
    Case(serve.StandIn, (URL, LONG[:BATCH + 1], "a handbook"), _judge_library(LONG[:BATCH + 1], "a handbook"),
         shown_args=("url", {"en": "[65 chunks]", "nl": "[65 chunks]"}, '"a handbook"')),
]

_LIBRARY_WORDS = {
    "en": {"title": "What your make_library sent for 150 chunks", "request": "request {k}: {n} texts",
           "one": "request {k}: 1 text", "more": "and {n} more requests", "gave": "The library it gave back: {chunks} chunks, and vectors of shape {shape}.", "stopped": "It stopped: {why}",
           "none": "It sent nothing yet."},
    "nl": {"title": "Wat je make_library stuurde voor 150 chunks", "request": "request {k}: {n} teksten",
           "one": "request {k}: 1 tekst", "more": "en nog {n} requests", "gave": "De library die hij teruggaf: {chunks} chunks, en vectors met vorm {shape}.", "stopped": "Hij stopte: {why}",
           "none": "Hij stuurde nog niets."},
}


def library_picture(seen) -> str:
    """The requests the learner's make_library sent for the 150 chunks of the first case, a bar per request as long as
    the texts it carried, against the line of BATCH; and the shape of the vectors it gave back."""
    run = seen.run
    fn = run.names.get("make_library") if run is not None and run.error is None else None
    if not callable(fn):
        return ""
    w = _LIBRARY_WORDS[seen.lang]
    got, error, running = apicheck.call(fn, LIBRARY_CASES[0])
    sent = _embedded(running.log)
    rows = []
    for k, s in enumerate(sent[:12]):
        n = len(s) if isinstance(s, list) else 1   # a text on its own is one text
        over = n > BATCH
        width = min(100.0, 100.0 * n / len(LONG))
        label = w["one"].format(k=k + 1) if n == 1 else w["request"].format(k=k + 1, n=n)
        rows.append(f'<div style="display:grid;grid-template-columns:9rem minmax(0,1fr);gap:0.5rem;align-items:center;margin-top:0.2rem">'
                    f'{_ui(label, "var(--ea-loss-text)" if over else "var(--ea-ink-2)")}'
                    f'<div style="position:relative;height:12px;background:var(--ea-inset);border-radius:3px">'
                    f'<div style="width:{width:.1f}%;height:12px;border-radius:3px;background:var(--ea-{"loss" if over else "data"})"></div>'
                    f'<div style="position:absolute;top:-2px;bottom:-2px;left:{100.0 * BATCH / len(LONG):.1f}%;border-left:1.5px dashed var(--ea-ink-3)"></div></div></div>')
    if len(sent) > 12:
        rows.append(f'<div style="margin-top:0.2rem">{_ui(w["more"].format(n=len(sent) - 12))}</div>')
    if not sent:
        rows.append(f'<div>{_ui(w["none"])}</div>')
    if error is not None:
        end = w["stopped"].format(why=f"{type(error).__name__}: {error}")
    elif isinstance(got, dict) and isinstance(got.get("vectors"), np.ndarray):
        end = w["gave"].format(chunks=len(got.get("chunks") or []), shape=tuple(got["vectors"].shape))
    else:
        end = w["stopped"].format(why=repr(got)[:120])
    legend = f'<div style="margin-top:0.3rem">{_ui("┆ BATCH = 64", size=11)}</div>'
    body = "".join(rows) + legend + f'<div style="margin-top:0.4rem">{_ui(end, "var(--ea-ink)")}</div>'
    return yours.figure(w["title"], body, alt=w["title"])


# --- exercise 2: app/rag.py ------------------------------------------------------------------------------------------------

START_ASK = '''import numpy as np

from app.citations import cited
from app.client import chat
from app.embeddings import embed
from app.meaning import meaning_search
from app.sources import with_sources

GROUNDED = ("You answer questions about {about}. Answer in one sentence, using only the numbered sources. "
            "End the sentence with the number of the source it comes from, in square brackets, such as [2]. "
            "If the sources do not answer the question, answer only: I don't know.")


def ask(url, library, question, k=3):
    q = np.array(embed(url, [question])[0])
    passages = meaning_search(q, library["vectors"], library["chunks"], k)
    answer = chat(url, [{"role": "user", "content": question}])
    named = []
    return {"answer": answer, "sources": passages, "named": named}
'''
SOLUTION_ASK = START_ASK.replace('''    answer = chat(url, [{"role": "user", "content": question}])
    named = []
''', '''    messages = [{"role": "system", "content": GROUNDED.format(about=library["about"])},
                {"role": "user", "content": with_sources(question, passages)}]
    answer = chat(url, messages)
    named = []
    for n in cited(answer):
        if n >= 1 and n <= len(passages):
            named.append(passages[n - 1])
''')
assert SOLUTION_ASK.count("GROUNDED.format") == 1 and START_ASK.split("GROUNDED = ")[1].split(")\n")[0].count("{about}") == 1


class _Fixed(serve.StandIn):
    """A stand-in that answers `text` to any question, and gives the question the embedding `vector`."""

    def __init__(self, text: str, question: str = "", vector=None):
        super().__init__(vectors={question: list(map(float, vector))} if vector is not None else None)
        self.text = text

    def answer(self, request: dict, heard) -> dict:
        self.default = self.text.replace("{", "{{").replace("}", "}}")
        return super().answer(request, heard)


def _library(about: str, chunks: list, vectors=None) -> dict:
    return {"about": about, "chunks": list(chunks), "vectors": np.array(vectors if vectors is not None else [serve.made_up(c) for c in chunks])}


def _chats(log: list) -> list:
    return [e for e in log if e["method"] == "POST" and e["path"].endswith("/chat/completions") and isinstance(e["body"], dict)]


def _judge_ask(library: dict, q: str, text: str, k: int, vector=None):
    qv = np.array(vector if vector is not None else serve.made_up(q), dtype=np.float32)
    passages = summaries.meaning_search(qv, library["vectors"], library["chunks"], k)
    want_messages = [{"role": "system", "content": GROUNDED.format(about=library["about"])},
                     {"role": "user", "content": with_sources(q, passages)}]
    want_named = named(text, passages)

    def judge(got, running):
        sent = _chats(running.log)
        if not sent:
            return "no-chat"
        msgs = sent[0]["body"].get("messages")
        if msgs == [{"role": "user", "content": q}]:
            return "question-alone"
        if not isinstance(msgs, list) or len(msgs) != 2 or not isinstance(msgs[0], dict) or msgs[0].get("role") != "system":
            return "no-system"
        system = str(msgs[0].get("content"))
        if "{about}" in system:
            return "unformatted"
        if system != want_messages[0]["content"]:
            return "fixed-about" if ABOUT in system and library["about"] != ABOUT else "system"
        user = msgs[1].get("content") if isinstance(msgs[1], dict) else None
        if user == q:
            return "no-sources"
        if msgs[1] != want_messages[1]:
            return "sources"
        if not isinstance(got, dict) or got.get("answer") != text or got.get("sources") != passages:
            return "return"
        if got.get("named") != want_named:
            if got.get("named") == passages:
                return "named-all"
            if got.get("named") == summaries.cited(text):
                return "named-numbers"
            return "named"
        return None
    return judge


def _ask_case(library: dict, q: str, text: str, k: int | None = None, vector=None, shown=None) -> Case:
    args = (URL, library, q) if k is None else (URL, library, q, k)
    return Case(lambda: _Fixed(text, q, vector), args, _judge_ask(library, q, text, K if k is None else k, vector), shown_args=shown)


SHOWN = "h07"   # the question of the first case and of the worked example
OTHER = ["The small CNN of the course has 9,098 parameters.", "Perplexity is the exponent of the mean loss per token.",
         "A bigram model predicts the next character from the one before it.", "Dropout zeroes a random part of a layer while training."]


def _first_case() -> Case:
    q = question(SHOWN)
    library = _library(ABOUT, CHUNKS, VECTORS)
    return _ask_case(library, q["text"]["en"], answers(SHOWN, "yours")[0], vector=question_vector(q),
                     shown=("url", {"en": "[the library of the handbook]", "nl": "[de library van het handboek]"}, repr(q["text"]["en"])))


ASK_CASES = [
    _first_case(),
    _ask_case(_library("the course Everything AI", OTHER), "What does dropout do?", "I don't know.",
              shown=("url", {"en": "[a library about the course]", "nl": "[een library over de course]"}, '"What does dropout do?"')),
    _ask_case(_library("the course Everything AI", OTHER), "How many parameters does the small CNN have?",
              "It has 9,098 parameters [2], see also [0] and [4]. [2]", 2,
              shown=("url", {"en": "[a library about the course]", "nl": "[een library over de course]"},
                     '"How many parameters does the small CNN have?"', "2")),
]

_ASK_WORDS = {
    "en": {"title": "What your ask did with: {q}", "sources": "The sources it found, the ones the answer names marked:",
           "answer": "The answer:", "named": "named: {names}", "none": "named: none", "stopped": "It stopped: {why}",
           "sent": "The messages it sent the model: {roles}."},
    "nl": {"title": "Wat je ask deed met: {q}", "sources": "De bronnen die hij vond, de bronnen die het antwoord noemt gemarkeerd:",
           "answer": "Het antwoord:", "named": "named: {names}", "none": "named: geen", "stopped": "Hij stopte: {why}",
           "sent": "De berichten die hij het model stuurde: {roles}."},
}


def ask_picture(seen) -> str:
    """The learner's ask on the first case, the question about the deposit against the course's recorded answer: the
    messages it sent, the sources it found with the named ones marked as data, the answer as model output."""
    run = seen.run
    fn = run.names.get("ask") if run is not None and run.error is None else None
    if not callable(fn):
        return ""
    w = _ASK_WORDS[seen.lang]
    q = question(SHOWN)
    got, error, running = apicheck.call(fn, ASK_CASES[0])
    title = w["title"].format(q=q["text"]["en"])
    if error is not None:
        return yours.figure(title, _ui(w["stopped"].format(why=f"{type(error).__name__}: {error}"), "var(--ea-ink)"), alt=title)
    sent = _chats(running.log)
    msgs = sent[0]["body"].get("messages") if sent else []
    roles = ", ".join(str(m.get("role")) for m in msgs if isinstance(m, dict)) if isinstance(msgs, list) else "?"
    parts = [f'<div>{_ui(w["sent"].format(roles=roles))}</div>']
    if isinstance(got, dict):
        passages = got.get("sources") if isinstance(got.get("sources"), list) else []
        marked = got.get("named") if isinstance(got.get("named"), list) else []
        parts.append(f'<div style="margin-top:0.4rem">{_ui(w["sources"])}</div>' + _source_blocks([str(p) for p in passages], marked))
        parts.append(f'<div style="margin-top:0.4rem">{_ui(w["answer"])} {_mono(str(got.get("answer")), "var(--ea-pred-text)")}</div>')
        numbers = [str(passages.index(m) + 1) for m in marked if m in passages]
        parts.append(f'<div>{_ui(w["named"].format(names=", ".join("[" + n + "]" for n in numbers)) if numbers else w["none"])}</div>')
    else:
        parts.append(f'<div>{_mono(repr(got)[:200])}</div>')
    return yours.figure(title, "".join(parts), alt=title)


def worked() -> dict:
    """The worked example: the question about the deposit, its three chunks, the messages the course's ask sends, the
    recorded answer of seed 0, what cited finds in it and the chunk it names."""
    q = question(SHOWN)
    passages = found(q)
    answer = answers(SHOWN, "yours")[0]
    return {"question": q["text"]["en"], "passages": passages, "answer": answer, "cited": summaries.cited(answer),
            "named": named(answer, passages), "messages": messages(q, "yours"), "plain": answers(SHOWN, "plain")[0]}


_COMMON = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
                "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."},
}


def _inhomogeneous(e: BaseException) -> bool:
    return isinstance(e, ValueError) and "inhomogeneous" in str(e)


def _module_missing(name: str):
    return lambda e: isinstance(e, ModuleNotFoundError) and f"app.{name}" in str(e)


EXERCISES = {
    "make_library": ApiBox(
        name="make_library",
        picture=library_picture,
        function="make_library",
        start=START_LIBRARY,
        solution=SOLUTION_LIBRARY,
        cases=LIBRARY_CASES,
        save_as=("app", "library"),
        mistakes=[
            Mistake("nested", error=_inhomogeneous,
                    en="With `{call}`, `np.array` cannot make a table of your list: each item is the list of one batch, and the last batch is shorter. Add the embeddings of a batch one by one with `+`: `vectors = vectors + embed(...)`. `append` adds the whole list as one item.",
                    nl="Met `{call}` kan `np.array` geen tabel maken van je lijst: elk item is de lijst van één batch, en de laatste batch is korter. Voeg de embeddings van een batch één voor één toe met `+`: `vectors = vectors + embed(...)`. `append` voegt de hele lijst toe als één item."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "The box sends all chunks in one request. Send them `BATCH` at a time, one request per batch, and add the embeddings of each batch to one list.",
                      "nl": "Het vak stuurt alle chunks in één request. Stuur ze `BATCH` tegelijk, één request per batch, en voeg de embeddings van elke batch toe aan één lijst."},
            "no_function": {"en": "Your box does not make a function `make_library` any more. The last lines start with `def make_library(url, chunks, about):`.",
                            "nl": "Je vak maakt geen functie `make_library` meer. De laatste regels beginnen met `def make_library(url, chunks, about):`."},
            "one-request": {"en": "With `{call}`, your function sent a request with more than `BATCH` texts. Send the chunks `BATCH` at a time, with `chunks[start:start + BATCH]`, and keep `BATCH = 64`.",
                            "nl": "Met `{call}` stuurde je functie een request met meer dan `BATCH` teksten. Stuur de chunks `BATCH` tegelijk, met `chunks[start:start + BATCH]`, en laat `BATCH = 64` staan."},
            "one-text": {"en": "With `{call}`, your function gave `embed` a text on its own, not a list of texts, and so sent a request per chunk. Give `embed` a list of `BATCH` chunks at a time: the slice `chunks[start:start + BATCH]` is a list too.",
                         "nl": "Met `{call}` gaf je functie `embed` een losse tekst, geen lijst teksten, en stuurde hij dus een request per chunk. Geef `embed` telkens een lijst van `BATCH` chunks: het stuk `chunks[start:start + BATCH]` is ook een lijst."},
            "small-batches": {"en": "With `{call}`, your function sent more requests than it needs: a request takes up to `BATCH` texts, so fill each one. Let `start` go up by `BATCH`, and send `chunks[start:start + BATCH]`.",
                              "nl": "Met `{call}` stuurde je functie meer requests dan nodig: een request neemt tot `BATCH` teksten aan, dus vul ze elk. Laat `start` telkens `BATCH` hoger worden, en stuur `chunks[start:start + BATCH]`."},
            "lost": {"en": "With `{call}`, the library has fewer embeddings than chunks: each batch replaces the embeddings of the one before. Add them to the list instead: `vectors = vectors + embed(...)`.",
                     "nl": "Met `{call}` heeft de library minder embeddings dan chunks: elke batch vervangt de embeddings van de vorige. Voeg ze juist toe aan de lijst: `vectors = vectors + embed(...)`."},
            "nested": {"en": "With `{call}`, the vectors have a list per batch: a table of tables. Add the embeddings of a batch one by one with `+`, not as one item with `append`.",
                       "nl": "Met `{call}` hebben de vectors een lijst per batch: een tabel van tabellen. Voeg de embeddings van een batch één voor één toe met `+`, niet als één item met `append`."},
            "not-array": {"en": "With `{call}`, `\"vectors\"` is not a NumPy array. Keep `np.array(vectors)` in the last line: `meaning_search` takes rows of an array.",
                          "nl": "Met `{call}` is `\"vectors\"` geen NumPy-array. Laat `np.array(vectors)` in de laatste regel staan: `meaning_search` neemt rijen van een array."},
            "vectors": {"en": "With `{call}`, the embeddings are not those of the chunks, in their order. Send each chunk once, in order: the slices `chunks[start:start + BATCH]` with `start` going up by `BATCH` from 0.",
                        "nl": "Met `{call}` zijn de embeddings niet die van de chunks, op hun volgorde. Stuur elke chunk één keer, op volgorde: de stukken `chunks[start:start + BATCH]` met `start` vanaf 0 telkens `BATCH` hoger."},
            "return": {"en": "With `{call}`, your function gives back `{got}`. It gives back a dictionary with `\"about\"`, `\"chunks\"` and `\"vectors\"`, as the last line of the start does.",
                       "nl": "Met `{call}` geeft je functie `{got}` terug. Hij geeft een dictionary terug met `\"about\"`, `\"chunks\"` en `\"vectors\"`, zoals de laatste regel van het begin doet."},
            "correct": {"en": "Your `make_library` embeds the chunks `BATCH` at a time and keeps them with their embeddings, in all {n} cases. The check saved it in your project, as `{saved}`.",
                        "nl": "Je `make_library` maakt de embeddings van de chunks `BATCH` tegelijk en bewaart ze samen, in alle {n} gevallen. De check heeft hem in je project bewaard, als `{saved}`."},
            "kept": {"en": "Your `make_library` is right. You changed `{saved}` yourself, so the check left it as it is.",
                     "nl": "Je `make_library` is goed. Je hebt `{saved}` zelf veranderd, dus de check liet het staan zoals het is."},
        },
    ),
    "ask": ApiBox(
        name="ask",
        picture=ask_picture,
        function="ask",
        start=START_ASK,
        solution=SOLUTION_ASK,
        cases=ASK_CASES,
        save_as=("app", "rag"),
        mistakes=[
            Mistake("index", error=lambda e: isinstance(e, IndexError),
                    en="With `{call}`, the answer names a source with a number that is not one of the passages, and `passages[n - 1]` stops with an IndexError. Keep only the numbers from 1 to `len(passages)`.",
                    nl="Met `{call}` noemt het antwoord een bron met een nummer dat geen van de passages is, en `passages[n - 1]` stopt met een IndexError. Houd alleen de nummers van 1 tot en met `len(passages)`."),
            Mistake("no-module", error=lambda e: isinstance(e, ModuleNotFoundError) and "app." in str(e),
                    en="Python cannot find a module of your project that the box imports. The first lines import `app.citations`, `app.client`, `app.embeddings`, `app.meaning` and `app.sources`: did you change a name there? Put the lines back as they were.",
                    nl="Python vindt een module van je project niet die het vak importeert. De eerste regels importeren `app.citations`, `app.client`, `app.embeddings`, `app.meaning` en `app.sources`: heb je daar een naam veranderd? Zet de regels terug zoals ze waren."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "The box sends the model the question alone, and names no source. Make the two messages: the system prompt `GROUNDED` filled in with the library's `\"about\"`, and the question with the passages from `with_sources`. Then keep in `named` the passages the answer cites.",
                      "nl": "Het vak stuurt het model alleen de vraag, en noemt geen bron. Maak de twee berichten: de system prompt `GROUNDED` ingevuld met de `\"about\"` van de library, en de vraag met de passages van `with_sources`. Houd daarna in `named` de passages die het antwoord noemt."},
            "no_function": {"en": "Your box does not make a function `ask` any more. The last lines start with `def ask(url, library, question, k=3):`.",
                            "nl": "Je vak maakt geen functie `ask` meer. De laatste regels beginnen met `def ask(url, library, question, k=3):`."},
            "no-chat": {"en": "With `{call}`, your function asked the model nothing. Does it still call `chat(url, messages)`?",
                        "nl": "Met `{call}` vroeg je functie het model niets. Roept hij nog `chat(url, messages)` aan?"},
            "question-alone": {"en": "With `{call}`, the model got the question alone: no rules, and no passages. Send it the two messages, the system prompt and the question with its sources.",
                               "nl": "Met `{call}` kreeg het model alleen de vraag: geen regels, en geen passages. Stuur het de twee berichten, de system prompt en de vraag met zijn bronnen."},
            "no-system": {"en": "With `{call}`, the request did not start with the system prompt, or did not have exactly two messages. First `{{\"role\": \"system\", ...}}` with `GROUNDED`, then the user's message.",
                          "nl": "Met `{call}` begon het request niet met de system prompt, of had het niet precies twee berichten. Eerst `{{\"role\": \"system\", ...}}` met `GROUNDED`, dan het bericht van de gebruiker."},
            "unformatted": {"en": "With `{call}`, the system prompt still has its field in it: the word about between curly braces. Fill it in: `GROUNDED.format(about=library[\"about\"])`.",
                            "nl": "Met `{call}` staat het veld nog in de system prompt: het woord about tussen accolades. Vul het in: `GROUNDED.format(about=library[\"about\"])`."},
            "fixed-about": {"en": "With `{call}`, the system prompt is about the bike shop, but this library is about something else. Fill in what the library says it is about: `library[\"about\"]`.",
                            "nl": "Met `{call}` gaat de system prompt over de fietsenwinkel, maar deze library gaat over iets anders. Vul in waar de library zegt dat hij over gaat: `library[\"about\"]`."},
            "system": {"en": "With `{call}`, the system prompt is not `GROUNDED` filled in with the library's `\"about\"`. Leave `GROUNDED` as it is, and use `GROUNDED.format(about=library[\"about\"])`.",
                       "nl": "Met `{call}` is de system prompt niet `GROUNDED` ingevuld met de `\"about\"` van de library. Laat `GROUNDED` staan zoals het is, en gebruik `GROUNDED.format(about=library[\"about\"])`."},
            "no-sources": {"en": "With `{call}`, the user's message is the question without the passages: the model cannot answer from them. Make it with `with_sources(question, passages)`.",
                           "nl": "Met `{call}` is het bericht van de gebruiker de vraag zonder de passages: het model kan er dan niet uit antwoorden. Maak het met `with_sources(question, passages)`."},
            "sources": {"en": "With `{call}`, the user's message is not what `with_sources(question, passages)` makes of the question and the passages your search found.",
                        "nl": "Met `{call}` is het bericht van de gebruiker niet wat `with_sources(question, passages)` maakt van de vraag en de passages die je zoekfunctie vond."},
            "return": {"en": "With `{call}`, your function gives back `{got}`. It gives back the answer, the passages it found as `\"sources\"`, and `\"named\"`, as the last line of the start does.",
                       "nl": "Met `{call}` geeft je functie `{got}` terug. Hij geeft het antwoord terug, de passages die hij vond als `\"sources\"`, en `\"named\"`, zoals de laatste regel van het begin doet."},
            "named-all": {"en": "With `{call}`, `named` holds all passages, also the ones the answer does not cite. Keep only the passages whose numbers `cited(answer)` finds.",
                          "nl": "Met `{call}` heeft `named` alle passages, ook die het antwoord niet noemt. Houd alleen de passages waarvan `cited(answer)` het nummer vindt."},
            "named-numbers": {"en": "With `{call}`, `named` holds the numbers the answer cites. Keep the passages themselves: number `n` is `passages[n - 1]`, because the sources are numbered from 1.",
                              "nl": "Met `{call}` heeft `named` de nummers die het antwoord noemt. Houd de passages zelf: nummer `n` is `passages[n - 1]`, omdat de bronnen vanaf 1 genummerd zijn."},
            "named": {"en": "With `{call}`, your `named` is not the passages the answer cites, in its order. Keep `passages[n - 1]` for every `n` from `cited(answer)` from 1 to `len(passages)`.",
                      "nl": "Met `{call}` is je `named` niet de passages die het antwoord noemt, op zijn volgorde. Houd `passages[n - 1]` voor elke `n` uit `cited(answer)` van 1 tot en met `len(passages)`."},
            "correct": {"en": "Your `ask` searches your library, asks with the grounded prompt and the sources, and keeps the sources the answer names, in all {n} cases. The check saved it in your project, as `{saved}`.",
                        "nl": "Je `ask` zoekt in je library, vraagt met de grounded prompt en de bronnen, en houdt de bronnen die het antwoord noemt, in alle {n} gevallen. De check heeft hem in je project bewaard, als `{saved}`."},
            "kept": {"en": "Your `ask` is right. You changed `{saved}` yourself, so the check left it as it is.",
                     "nl": "Je `ask` is goed. Je hebt `{saved}` zelf veranderd, dus de check liet het staan zoals het is."},
        },
    ),
}

REQUIRED = ["make_library", "ask", "new-document"]

CASES = [
    ("make_library", START_LIBRARY, "empty", None),
    ("make_library", SOLUTION_LIBRARY, "correct", None),
    ("make_library", START_LIBRARY.replace("    vectors = embed(url, chunks)\n", "    vectors = embed(url, chunks[:BATCH])\n"), "wrong", "lost"),
    ("make_library", LOST, "wrong", "lost"),
    ("make_library", NESTED, "error", "nested"),
    ("make_library", SOLUTION_LIBRARY.replace("BATCH = 64", "BATCH = 100"), "wrong", "one-request"),
    ("make_library", SOLUTION_LIBRARY.replace('"vectors": np.array(vectors)', '"vectors": vectors'), "wrong", "not-array"),
    ("make_library", SOLUTION_LIBRARY.replace("    for start in range(0, len(chunks), BATCH):\n        vectors = vectors + embed(url, chunks[start:start + BATCH])\n",
                                              "    for chunk in chunks:\n        vectors = vectors + embed(url, chunk)\n"), "wrong", "one-text"),
    ("make_library", SOLUTION_LIBRARY.replace("range(0, len(chunks), BATCH):\n        vectors = vectors + embed(url, chunks[start:start + BATCH])",
                                              "range(0, len(chunks), 10):\n        vectors = vectors + embed(url, chunks[start:start + 10])"), "wrong", "small-batches"),
    ("make_library", SOLUTION_LIBRARY.replace("vectors + embed(url, chunks[start:start + BATCH])", "embed(url, chunks[start:start + BATCH]) + vectors"), "wrong", "vectors"),
    ("make_library", SOLUTION_LIBRARY.replace('return {"about": about, ', 'return {'), "wrong", "return"),
    ("ask", START_ASK, "empty", None),
    ("ask", SOLUTION_ASK, "correct", None),
    ("ask", SOLUTION_ASK.replace('''    answer = chat(url, messages)''', '''    answer = chat(url, messages[1:])'''), "wrong", "no-system"),
    ("ask", SOLUTION_ASK.replace("    answer = chat(url, messages)\n", '    answer = chat(url, [{"role": "user", "content": question}])\n'), "wrong", "question-alone"),
    ("ask", SOLUTION_ASK.replace('GROUNDED.format(about=library["about"])', "GROUNDED"), "wrong", "unformatted"),
    ("ask", SOLUTION_ASK.replace('GROUNDED.format(about=library["about"])', 'GROUNDED.format(about="the bike shop Spoke & Chain")'), "wrong", "fixed-about"),
    ("ask", SOLUTION_ASK.replace('GROUNDED.format(about=library["about"])', 'GROUNDED.format(about=question)'), "wrong", "system"),
    ("ask", SOLUTION_ASK.replace("with_sources(question, passages)}]", "question}]"), "wrong", "no-sources"),
    ("ask", SOLUTION_ASK.replace("with_sources(question, passages)}]", "with_sources(question, passages[:1])}]"), "wrong", "sources"),
    ("ask", SOLUTION_ASK.replace("        if n >= 1 and n <= len(passages):\n            named.append(passages[n - 1])\n",
                                 "        named.append(passages[n - 1])\n"), "error", "index"),
    ("ask", SOLUTION_ASK.replace("            named.append(passages[n - 1])", "            named.append(n)"), "wrong", "named-numbers"),
    ("ask", SOLUTION_ASK.replace("    named = []\n    for n in cited(answer):\n        if n >= 1 and n <= len(passages):\n            named.append(passages[n - 1])\n",
                                 "    named = passages\n"), "wrong", "named-all"),
    ("ask", SOLUTION_ASK.replace("n >= 1 and n <= len(passages)", "n <= len(passages)"), "wrong", "named"),
    ("ask", SOLUTION_ASK.replace('"sources": passages', '"passages": passages'), "wrong", "return"),
    ("ask", SOLUTION_ASK.replace("from app.meaning import", "from app.meanings import"), "error", "no-module"),
    ("ask", SOLUTION_ASK.replace("    answer = chat(url, messages)\n", "    answer = \"\"\n"), "wrong", "no-chat"),
]


# --- the demo: your assistant reads your documents ---------------------------------------------------------------------------

SIZE, OVERLAP = summaries.HANDBOOK_SIZE, summaries.HANDBOOK_OVERLAP   # how the demo cuts your documents


def downloaded() -> bool:
    from everything_ai import hub
    return all(hub.pinned(*pin) is not None for pin in FILES)


def download_or_stop(button, texts: dict) -> None:
    lesson_helpers.pins_or_stop(button, texts, FILES)


def documents_folder():
    """work/projects/app/documents/: made the first time, with the course's handbook of Spoke & Chain in it."""
    folder = projects.folder() / "app" / "documents"
    if not folder.exists():
        folder.mkdir(parents=True)
        shutil.copyfile(summaries.HANDBOOK, folder / summaries.HANDBOOK.name)
    return folder


def documents() -> list:
    """The .md and .txt files in the documents folder, as (name, text), by name."""
    out = []
    for path in sorted(documents_folder().iterdir()):
        if path.is_file() and path.suffix.lower() in (".md", ".txt"):
            out.append((path.name, path.read_text(encoding="utf-8", errors="replace")))
    return out


def learner(module: str, name: str):
    """A function from the learner's work/projects/app/<module>.py, and None; or None, and the error that stopped the
    import. Modules it imports that are missing or do not pass are the course's (builds_on)."""
    run = run_code(f"from app.{module} import {name}")
    if run.error is not None:
        return None, f"{type(run.error).__name__}: {run.error}"
    return run.names.get(name), None


_LIBRARIES: dict = {}   # the libraries the demo made, by what they were made from: made again only when that changes


def _library_key(docs: list, about: str, make_library, chunk) -> tuple:
    return (tuple(docs), about, getattr(make_library, "__code__", None), getattr(chunk, "__code__", None))


def demo(make_library, ask, chunk, docs: list, about: str, question_text: str, show, words: dict, lang: str) -> str:
    """The learner's assistant on their documents against the course's server: the documents cut with their chunk into
    a library by their make_library (kept until the documents, the about or the code change), then their ask with
    the question; the answer, and the sources it found with the ones it names marked, each with its file."""
    running = serve.model_server()
    key = _library_key(docs, about, make_library, chunk)
    try:
        if key not in _LIBRARIES:
            chunks, where = [], []
            for name, text in docs:
                for c in chunk(text, SIZE, OVERLAP):
                    chunks.append(c)
                    where.append(name)
            show(words["embedding"].format(n=len(chunks), d=len(docs)))
            start = time.perf_counter()
            library = make_library(running.url, chunks, about)
            _LIBRARIES[key] = (library, where, time.perf_counter() - start)
        library, where, seconds = _LIBRARIES[key]
        show(words["asking"])
        start = time.perf_counter()
        out = ask(running.url, library, question_text)
        asked = time.perf_counter() - start
    except Exception as e:   # the learner's code, against the real server; it may have run out of memory
        said = words["memory"] if serve.out_of_memory(e) else words["failed"].format(why=f"{type(e).__name__}: {e}")
        return f"<p><em>{html.escape(said)}</em></p>"
    if not isinstance(out, dict) or not isinstance(out.get("sources"), list):
        return f"<p><em>{html.escape(words['failed'].format(why=repr(out)[:200]))}</em></p>"
    chunks = list(library.get("chunks") or []) if isinstance(library, dict) else []
    files = [where[chunks.index(p)] if p in chunks and len(where) == len(chunks) else "?" for p in out["sources"]]
    marked = out.get("named") if isinstance(out.get("named"), list) else []
    course = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]
    whose = words["course"].format(files=", ".join(course)) if course else words["yours"]
    head = words["library"].format(n=len(chunks), d=len(docs), seconds=number(seconds, lang, 1), asked=number(asked, lang, 1))
    numbers = [str(out["sources"].index(m) + 1) for m in marked if m in out["sources"]]
    parts = [f'<div>{_ui(head + " " + whose)}</div>',
             f'<div style="margin-top:0.4rem">{_ui(words["answer"])}</div>',
             f'<div style="font-family:var(--ea-font-ui);font-size:15px;color:var(--ea-pred-text)">{html.escape(str(out.get("answer")))}</div>',
             f'<div style="margin-top:0.2rem">{_ui(words["named"].format(names=", ".join("[" + n + "]" for n in numbers)) if numbers else words["none"])}</div>',
             f'<div style="margin-top:0.4rem">{_ui(words["sources"])}</div>' + _source_blocks([str(p) for p in out["sources"]], marked, files, 400)]
    return yours.figure(words["title"].format(question=question_text), "".join(parts), alt=words["title"].format(question=question_text))
