# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-retrieval-embeddings. Route: course/route/lessen/bouwen.yaml. Checks, the embeddings and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-retrieval-embeddings", "en")
    C = L.checks   # the knowledge base, Granite's embeddings of it, the checks, and the demo with your retrieve


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Embeddings and documents through an API](/course/open/build-embeddings-api?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_embed():
    quick_embed = L.choice(
        "quick-embed",
        question="Granite Embedding gives every embedding length 1. What is the dot product of two of its embeddings?",
        options={
            "distance": "The distance between the two texts: the smaller, the closer",
            "words": "The number of words the two texts share, divided by their length",
            "cosine": "Their cosine similarity: the higher, the closer in meaning",
        },
        correct="cosine",
        why={
            "cosine": "The cosine similarity is the dot product divided by the two lengths; when both lengths are 1, nothing changes. The higher it is, the more the two embeddings point the same way.",
            "distance": "A distance is smallest for the closest texts; the dot product of two embeddings of length 1 is largest for them. See [Embeddings and documents through an API](/course/open/build-embeddings-api?lang=en).",
            "words": "An embedding is a list of 384 numbers, not of words. See [Embeddings and documents through an API](/course/open/build-embeddings-api?lang=en).",
        },
    )
    quick_embed
    return (quick_embed,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The word search of [Your first RAG](/course/open/build-rag-first?lang=en) found the right point for only {words} of the {meaning} questions that ask in other words, and BM25, of [Search by words: BM25](/course/open/build-bm25?lang=en), also for only {bm25}. People ask in their own words, not in those of your documents, and sometimes in another language. In [Embeddings and documents through an API](/course/open/build-embeddings-api?lang=en), Granite found a Dutch help page for an English question. In this lesson you search all {points} points of the course that way: by meaning.

    **An everyday example.** You ask a librarian for *a book on why the sky is blue*, and get one called *Light in the atmosphere*. Not one of your words is in its title, but the librarian knows what you mean; the index at the back of a book can only point to words. That is only a comparison: the search below compares lists of numbers.
    """.format(words=C.findings()["search_meaning"], bm25=C.findings()["bm25_meaning"], meaning=C.findings()["meaning_questions"],
               points=C.number(len(C.POINTS), L.lang, 0)))
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **dense retrieval** | retrieval by meaning: compare the embedding of the question with the embedding of every text, and keep the texts with the highest cosine similarity |
    | **cross-lingual retrieval** | retrieval across languages: a question in one language finds texts in another |
    | `vectors @ q` | for an array `vectors` with one embedding per row and an embedding `q`: the dot product of every row with `q`, one number per row |

    *Dense*, because every one of the 384 numbers of an embedding has a value for every text. A list that counts each of the {vocab} different words in the points would, for one point, be zero almost everywhere: no point has more than {most} different words.

    The embeddings of the points do not depend on the question, so the course made them once: {points} embeddings of 384 numbers, the rows of one array. That took {all} seconds on the course's machine. A question then costs one request with one text: {one} seconds. Both come from the same model, Granite Embedding, because the embeddings of two models do not fit together. And because Granite gives a text and its translation embeddings close together, a question in Dutch finds the points too, though they are in English.

    `np.array([[0.6, 0.8], [1.0, 0.0]]) @ np.array([0.8, 0.6])` gives an array with 0.96 and 0.8: the dot product of each row with the embedding.
    """.format(vocab=C.number(C.vocabulary(), L.lang, 0), most=C.most_words(), points=C.number(len(C.POINTS), L.lang, 0),
               all=C.number(C.SECONDS["points"], L.lang, 1), one=C.number(C.SECONDS["question"], L.lang, 3)))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: search by meaning

    1. **Once:** embed every point, and keep the embeddings as the rows of one array.
    2. **Question:** embed the question with the same model: one request with one text.
    3. **Score:** take the dot product of every row with the question's embedding; at length 1, that is the cosine similarity.
    4. **Keep:** the `k` points with the highest score, the highest first.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _w = C.worked()
    mo.md(r"""
    ### Worked example

    *{question}*

    1. **Once:** the course embedded all {points} points: an array of {points} rows of 384 numbers.
    2. **Question:** one request gives the 384 numbers of the question.
    3. **Score:** `vectors @ q` gives {points} scores, from {low} to {s0}.
    4. **Keep:** the highest, {s0}, is a point of *{t0}*: *{p0}* It answers the question: the nearest examples vote. Then {s1}, a point of *{t1}*, and {s2}, of *{t2}*.

    That point shares only {shared} with the question, so BM25 put it at place {bm25}. BM25's first was the point of *{t1}*, second here.
    """.format(question=_w["question"], points=C.number(len(C.POINTS), L.lang, 0), low=C.number(_w["low"], L.lang, 3),
               s0=C.number(_w["scores"][0], L.lang, 3), s1=C.number(_w["scores"][1], L.lang, 3), s2=C.number(_w["scores"][2], L.lang, 3),
               t0=C.title_of(_w["top"][0], L.lang), t1=C.title_of(_w["top"][1], L.lang), t2=C.title_of(_w["top"][2], L.lang), p0=_w["top"][0],
               shared=", ".join(["*" + w + "*" for w in _w["shared"][:-1]]) + " and *" + _w["shared"][-1] + "*", bm25=_w["bm25_rank"]))
    return


@app.cell(hide_code=True)
def text_measured():
    mo.md(r"""
    ### What the course measured

    For each of the {questions} questions a point answers, the course searched all {points} points in two ways, with BM25 and by meaning, and noted the place of the right point. Then it did the same with the questions in Dutch, against the same English points. The picture marks the {meaning} questions that ask in other words:
    """.format(questions=C.findings()["questions"], points=C.number(len(C.POINTS), L.lang, 0), meaning=C.findings()["meaning_questions"]))
    return


@app.cell(hide_code=True)
def picture_ranks():
    mo.Html(C.ranks_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_runs():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        By meaning, the right point was in the top three for {meaning} of the {questions} questions; with BM25, for {bm25}. The difference is in the questions in other words: {mm} of {nm} by meaning, {bm} with BM25. On the questions that share words with their point, BM25 was a little better: {bw} of {nw} in the top three against {mw}, and first {bwf} times against {mwf}.

        With the questions in Dutch, meaning found {mnl} of the {questions}, BM25 {bnl}. BM25 can only go by the words a Dutch question shares with an English point: numbers, names, and English words that Dutch uses too, such as *cache* and *request*. *{sha}* shares none, because Dutch writes *SHA-256-checksum* as one word: BM25 put its point at place {sha_bm25}, search by meaning first.
        """.format(meaning=_f["meaning"], questions=_f["questions"], bm25=_f["bm25"], mm=_f["meaning_meaning"], nm=_f["meaning_questions"],
                   bm=_f["bm25_meaning"], bw=_f["bm25_words"], nw=_f["words_questions"], mw=_f["meaning_words"], bwf=_f["bm25_words_first"],
                   mwf=_f["meaning_words_first"], mnl=_f["meaning_nl"], bnl=_f["bm25_nl"], sha=C.question("q17")["text"]["nl"],
                   sha_bm25=C.number(C.ranks()["q17"][("bm25", "nl")], L.lang, 0))),
        L.watch_out("Meaning is not always better. *{agent}* is about reinforcement learning, but by meaning its three best were points of the two lessons on coding agents, where an agent is something else; BM25 put the right point first. A later lesson combines search by words and by meaning, because each finds points the other misses.".format(agent=C.question("q20")["text"]["en"])),
    ])
    return


@app.cell(hide_code=True)
def text_search():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The points whose embeddings fit best

        `meaning_search(q, vectors, points, k)` should give the `k` points whose embeddings have the highest dot product with `q`, the highest first. `q` is the embedding of the question, a NumPy array; `vectors` has one embedding per row, in the order of `points`, so `vectors[i]` is the embedding of `points[i]`. The box is the search of [Your first RAG](/course/open/build-rag-first?lang=en), with pairs of a score and a point, but every score is 0. When it is right, the check also keeps it as a file, `work/projects/app/meaning.py`: [Project: your assistant reads your documents](/course/open/build-project-rag?lang=en) uses it.
        """),
        L.task("Give each point the dot product of its embedding with `q` as its score, instead of 0."),
    ])
    return


@app.cell(hide_code=True)
def exercise_search():
    box_search = L.code_box("meaning_search", lines=5)
    box_search
    return (box_search,)


@app.cell(hide_code=True)
def check_search(box_search):
    result_search = L.check("meaning_search", box_search.value)
    result_search
    return (result_search,)


@app.cell(hide_code=True)
def hints_search():
    L.hints(
        "meaning_search",
        [
            "The score of point `i` compares its own embedding, row `i` of `vectors`, with the embedding of the question.",
            "`vectors[i] @ q` is the dot product of row `i` with `q`: one number, the score of point `i`.",
            "In the loop: `found.append((vectors[i] @ q, points[i]))`.",
        ],
        why="""```python
def meaning_search(q, vectors, points, k):
    found = []
    for i in range(len(points)):
        found.append((vectors[i] @ q, points[i]))
    return [point for score, point in sorted(found, reverse=True)[:k]]
```

`scores = vectors @ q` before the loop gives all scores at once, one per row, and `scores[i]` in the loop gives the same pairs. Only the score changed from the word search: the rest of the search stays as it was.""",
    )
    return


@app.cell(hide_code=True)
def text_retrieve():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Embed the points once

        `retrieve(url, question, k)` should give the `k` points that fit a question best, by meaning. The box does it the way `best_match` of [Embeddings and documents through an API](/course/open/build-embeddings-api?lang=en) did: it sends the question together with all {points} points, in one request, for every question. That works, but it embeds the whole knowledge base again for each question: all points took {all} seconds on the course's machine, one question {one}. The course embedded the points once: `VECTORS` holds their embeddings, a row per point, in the order of `POINTS`. `embed` of that lesson, `meaning_search`, `VECTORS` and `POINTS` are given. The check runs your function against a stand-in.
        """.format(points=C.number(len(C.POINTS), L.lang, 0), all=C.number(C.SECONDS["points"], L.lang, 1), one=C.number(C.SECONDS["question"], L.lang, 3))),
        L.task("Embed only the question, and search `VECTORS` with its embedding."),
    ])
    return


@app.cell(hide_code=True)
def exercise_retrieve():
    box_retrieve = L.code_box("retrieve", lines=7)
    box_retrieve
    return (box_retrieve,)


@app.cell(hide_code=True)
def check_retrieve(box_retrieve):
    result_retrieve = L.check("retrieve", box_retrieve.value)
    result_retrieve
    return (result_retrieve,)


@app.cell(hide_code=True)
def hints_retrieve():
    L.hints(
        "retrieve",
        [
            "The embeddings of the points are ready in `VECTORS`; for each question, only the question itself still needs one.",
            "`embed(url, [question])` gives a list with one embedding; `[0]` takes it out, and `np.array(...)` makes an array of it.",
            "`q = np.array(embed(url, [question])[0])`, and then `return meaning_search(q, VECTORS, POINTS, k)`.",
        ],
        why="""```python
import numpy as np


def retrieve(url, question, k):
    q = np.array(embed(url, [question])[0])
    return meaning_search(q, VECTORS, POINTS, k)
```

A RAG system embeds its documents once, when they come in, and keeps the embeddings; each question then costs one short request. When a document changes, only its own embeddings are made again.""",
    )
    return


@app.cell(hide_code=True)
def text_high_score():
    mo.md(r"""
    ### Exercise 3. A high score

    No point of the course says how many people finished it.
    """)
    return


@app.cell(hide_code=True)
def choice_high_score_cell():
    _f = C.findings()
    choice_high_score = L.choice(
        "high-score",
        question="For `{question}`, search by meaning still finds three points, the best with a score of {high}. What does that score tell you?".format(
            question=C.question(_f["none_question"])["text"]["en"], high=C.number(_f["none_high"], L.lang, 3)),
        options={
            "answers": "That the best point answers the question, because {high} is a high score".format(high=C.number(_f["none_high"], L.lang, 3)),
            "nothing": "On its own, nothing: right points often score lower",
            "below": "That Granite understood the question badly, because the score is below 0.8",
        },
        correct="nothing",
        why={
            "nothing": "Search by meaning always gives the `k` highest, also when no point answers the question. The right point of {below} of the {questions} questions scored lower than {high}, among them {mbelow} of the {mn} that ask in other words. The model has to see for itself whether the sources answer the question, in [Answers from the sources](/course/open/build-grounded-answers?lang=en).".format(
                below=_f["gold_below"], questions=_f["questions"], high=C.number(_f["none_high"], L.lang, 3), mbelow=_f["meaning_below"], mn=_f["meaning_questions"]),
            "answers": "The best point is about the course's training run on its CPU, not about people. And do right points always score higher than {high}?".format(high=C.number(_f["none_high"], L.lang, 3)),
            "below": "Is there a score above which a point is right? The right point of the question about memory while a network learns scored {low}, and was second.".format(low=C.number(C.gold_score("q19"), L.lang, 3)),
        },
    )
    choice_high_score
    return (choice_high_score,)


@app.cell(hide_code=True)
def text_dutch():
    mo.md(r"""
    ### Exercise 4. A question in Dutch

    Look again at the question about the checksum, in the picture of what the course measured.
    """)
    return


@app.cell(hide_code=True)
def choice_dutch_cell():
    _f = C.findings()
    choice_dutch = L.choice(
        "dutch-question",
        question="In Dutch, the question about the checksum is `{sha}`. BM25 puts the English point about the checksum at place {place}, search by meaning first. Why does BM25 miss it?".format(
            sha=C.question("q17")["text"]["nl"], place=C.number(C.ranks()["q17"][("bm25", "nl")], L.lang, 0)),
        options={
            "rare": "SHA-256 is too rare a word for BM25 to give it any weight",
            "translate": "BM25 first translates the question into English, and gets hextekens wrong",
            "words": "The question shares no word with the point, so every point scores 0",
        },
        correct="words",
        why={
            "words": "BM25 counts only the words a point shares with the question, as they are written. *SHA-256-checksum?* is one word in Dutch, and *hoeveel*, *hextekens*, *heeft* and *een* are in no English point, so all {points} points score 0 and their order says nothing. Granite gives a text and its translation embeddings close together, so search by meaning found the point first. Over all {questions} questions in Dutch: {mnl} by meaning, {bnl} with BM25.".format(
                points=C.number(len(C.POINTS), L.lang, 0), questions=_f["questions"], mnl=_f["meaning_nl"], bnl=_f["bm25_nl"]),
            "rare": "BM25 gives a rare word more weight, not less: that is what the idf does. But is *SHA-256-checksum?* a word of the English point?",
            "translate": "BM25 knows no language: it counts words as they are written. Which words does the Dutch question share with the English point?",
        },
    )
    choice_dutch
    return (choice_dutch,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Search the course with your own retrieve

    Once exercises 1 and 2 are right, you can search the course with your own functions: the course starts its server on your computer, with Granite behind it, and shows beside what your `retrieve` finds what BM25 finds for the same question. Ask in English, in Dutch, or in another of Granite's twelve languages, and try a question in your own words.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Granite Embedding (223 MB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*The demo needs Granite Embedding, which is not on your computer yet. The button downloads it once from Hugging Face, at a fixed version: `ibm-granite/granite-embedding-107m-multilingual`, 223 MB, under the Apache-2.0 license.*",
        "busy": "**Downloading** {done} of {total} MB …",
        "no_net": "*The download did not work: Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: a file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is not enough room on your disk. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
    })
    mo.md("*Granite Embedding is downloaded.*")
    return


@app.cell(hide_code=True)
def settings_cell():
    question_text = mo.ui.text(value=C.question("q22")["text"][L.lang], label="Your question", full_width=True)
    search_button = mo.ui.run_button(label="Search with your retrieve")
    mo.vstack([question_text, search_button])
    return question_text, search_button


@app.cell(hide_code=True)
def demo_answer(box_retrieve, box_search, question_text, search_button):
    _fn = C.learner_retrieve(box_search.value, box_retrieve.value)
    if not C.downloaded():
        _out = mo.md("*Download Granite Embedding first, with the button above.*")
    elif _fn is None:
        _out = mo.md("*Finish exercises 1 and 2 first: the demo searches with your own `meaning_search` and `retrieve`.*")
    elif not search_button.value:
        _out = mo.md("*Press the button to search.*")
    else:
        mo.output.replace(mo.md("*Your `retrieve` sends the question; the server loads Granite and works out its embedding …*"))
        _words = {"title": "Your retrieve: {question}", "mine": "By meaning, with your retrieve:", "bm25": "By words, with BM25:",
                  "done": "{seconds} seconds for your retrieve, with loading Granite the first time.", "empty": "Type a question first.",
                  "failed": "Your retrieve stopped with an error against the real server: {why}",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.demo(_fn, question_text.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The points your `retrieve` finds go into the prompt as in [Your first RAG](/course/open/build-rag-first?lang=en). [Answers from the sources](/course/open/build-grounded-answers?lang=en) asks Qwen3 to answer only from them, to name the point it used, and to say so when they do not answer the question, which the score cannot tell. The points here are short; a longer document is first cut into pieces, as in [Cutting documents into pieces](/course/open/build-chunking?lang=en), and every piece gets its own embedding.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_dutch, choice_high_score, result_retrieve, result_search):
    L.wrap_up(result_search, result_retrieve, choice_high_score, choice_dutch)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
