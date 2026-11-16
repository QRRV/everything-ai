# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-query-rewriting. Route: course/route/lessen/bouwen.yaml. Checks, the recorded rewrites and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-query-rewriting", "en")
    C = L.checks   # the course's points, Qwen3's recorded rewrites, passages and versions, their embeddings, the checks and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Words and meaning together](/course/open/build-hybrid-search?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_scale():
    _h = C.hybrid_worked()
    quick_scale = L.choice(
        "quick-scale",
        question="Why does a hybrid search merge the two rankings, rather than add the BM25 score of a point to the dot product of the embeddings?",
        options={
            "scale": "The two scores are on different scales",
            "slow": "Adding up scores is slower than merging rankings",
            "sign": "A BM25 score can be negative, a dot product cannot",
        },
        correct="scale",
        why={
            "scale": "For the worked example of that lesson, BM25 gave the points from {bl} to {bh}, the dot product from {ml} to {mh}: in a sum BM25 decides nearly alone. RRF only looks at the places.".format(
                bl=C.number(_h["bm25_low"], L.lang, 0), bh=C.number(_h["bm25_high"], L.lang, 0), ml=C.number(_h["meaning_low"], L.lang, 2), mh=C.number(_h["meaning_high"], L.lang, 2)),
            "slow": "Both are quick: a sum is one addition per point. The problem is the scales: BM25 gave the points from {bl} to {bh}, the dot product from {ml} to {mh}. See [Words and meaning together](/course/open/build-hybrid-search?lang=en).".format(
                bl=C.number(_h["bm25_low"], L.lang, 0), bh=C.number(_h["bm25_high"], L.lang, 0), ml=C.number(_h["meaning_low"], L.lang, 2), mh=C.number(_h["meaning_high"], L.lang, 2)),
            "sign": "The other way round: a BM25 score is never below 0, and a dot product can be. The problem is the scales: BM25 gave the points from {bl} to {bh}, the dot product from {ml} to {mh}. See [Words and meaning together](/course/open/build-hybrid-search?lang=en).".format(
                bl=C.number(_h["bm25_low"], L.lang, 0), bh=C.number(_h["bm25_high"], L.lang, 0), ml=C.number(_h["meaning_low"], L.lang, 2), mh=C.number(_h["meaning_high"], L.lang, 2)),
        },
    )
    quick_scale
    return (quick_scale,)


@app.cell(hide_code=True)
def text_why():
    _f = C.findings()
    _w = C.worked()
    mo.md(r"""
    ## Why this matters

    In a chat, the next question often leans on the one before. *{earlier}* And then: *{follow}* On its own, the follow-up does not say what "its" is. Search by meaning put the right point among the three best for {full} of the {n} questions about the course when they were asked in full, and for {alone} when only the follow-up was asked. A short question, as people type into a search box, did not do much better: {short}.

    **An everyday example.** You walk into a library and ask: "How many pages does it have?" The librarian first asks which book you mean, and only then goes to look. A search engine cannot ask back; something has to put the book in the question first.
    """.format(earlier=_w["earlier"], follow=_w["follow"], earlier_nl=_w["earlier_nl"], follow_nl=_w["follow_nl"], full=_f["full"], n=_f["questions"], alone=_f["follow"], short=_f["short"]))
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **query transformation** | changing the question before the search, often by letting a model write the text to search with |
    | **standalone question** | a follow-up written again so that it says itself what it is about, with the names it refers to |
    | **HyDE** | hypothetical document embeddings: let the model write a passage that could answer the question, and search with the embedding of that passage |
    | **multi-query retrieval** | let the model write other versions of the question, search with each, and merge the rankings, here with RRF |

    A search compares the question with the points. A follow-up leaves out what it is about, and a short question says little, so the model writes a better text to search with first. For a follow-up that is the question on its own: the model gets the chat, and writes the last question again. HyDE, proposed in 2022 by Gao, Ma, Lin and Callan, starts from the other side: a passage that answers looks more like the right point than the question does, even when its details are made up. Multi-query retrieval asks the same thing several ways, so that a point that comes high for most of them rises; RAG-Fusion merges such rankings with RRF, as this lesson does. A question with two parts can also be split, with a search for each part.

    Each of these costs a call to the model before the search can start, and the model writes from what it knows. Whether the search gets better, you only see by measuring.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: ask it differently

    1. **Look:** what is wrong with the question as it comes in: does it lean on the chat, or is it short?
    2. **Write:** let the model write the text to search with: the question on its own, a passage that answers, or versions of the question.
    3. **Search:** search with what the model wrote; with several versions, merge the rankings with RRF.
    4. **Measure:** compare with the search on the question as it came in, on questions whose right point you know.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _w = C.worked()
    mo.vstack([
        mo.md(r"""
        ### Worked example

        1. **Look:** *{follow}* leans on the question before it: *{earlier}* Searching with it alone put the right point at place {alone}; first came a point of *{first}*: *{first_text}*
        2. **Write:** the course sends Qwen3 the chat, with a system prompt that asks for the last question on its own.
        """.format(follow=_w["follow"], earlier=_w["earlier"], follow_nl=_w["follow_nl"], earlier_nl=_w["earlier_nl"], alone=_w["places"]["follow"], first=C.title_of(_w["first"], L.lang), first_text=_w["first"])),
        mo.md(C.request_code(C.WORKED)),
        mo.md(r"""
        With `"temperature": 0` the model takes the most likely token each time, so it writes the same on every run. Qwen3 wrote: *{rewrite}*

        3. **Search:** with that, search by meaning puts the right point at place {rewritten}: *{gold}*
        4. **Measure:** one question proves little. The course did the same for all {n} questions a point answers.
        """.format(rewrite=_w["rewrite"], rewritten=_w["places"]["rewrite"], gold=_w["gold"], n=C.findings()["questions"])),
    ])
    return


@app.cell(hide_code=True)
def text_measured():
    _f = C.findings()
    mo.md(r"""
    ### What the course measured

    For each of the {n} questions a point answers, the course wrote a follow-up after an earlier question, and a short question as people type into a search box. Qwen3 rewrote each follow-up, and for each short question wrote a passage (HyDE) and three versions. Search by meaning, with Granite's embeddings, searched with each:
    """.format(n=_f["questions"]))
    return


@app.cell(hide_code=True)
def picture_bars():
    mo.Html(C.bars_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_runs():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        Qwen3's rewrite lifted the follow-ups from {follow} to {rewrite}. But the follow-up simply after the earlier question, without the model, got {joined}. In {copied} of the {n} rewrites Qwen3 only copied the two questions one after the other, and where it really wrote a new question, it sometimes left out what the question was about.

        For the short questions, Qwen3 made things worse: its passages got {hyde}, the short question with its three versions {multi}, against {short} for the short question alone. Qwen3 knows little about this course, so what it wrote drifted to AI in general.

        Each way also cost a call to the model before the search: on the course's computer a rewrite took about {s_rewrite} seconds, a passage {s_hyde}, three versions {s_versions}.
        """.format(follow=_f["follow"], rewrite=_f["rewrite"], joined=_f["joined"], copied=len(_f["copied"]), n=_f["questions"], hyde=_f["hyde"],
                   multi=_f["multi"], short=_f["short"], s_rewrite=C.number(_f["seconds_rewrite"], L.lang, 1),
                   s_hyde=C.number(_f["seconds_hyde"], L.lang, 1), s_versions=C.number(_f["seconds_versions"], L.lang, 1))),
        L.watch_out("Query transformation can make a search worse, and a small model that does not know your subject does that easily. Measure it on your own questions against the question as it came in, and try the cheapest thing first: here that was the earlier question before the follow-up."),
    ])
    return


@app.cell(hide_code=True)
def text_versions():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Read the versions

        `versions(text)` gets what the model wrote when it was asked for other versions of a question, and should give each version as a text of its own. Models write such a list as lines that start with `- `, or numbered as `1.`, `2.` and `3.`, often with spaces around them, and sometimes with empty lines in between. Qwen3 wrote this for the short question *{short}*:

        `{raw}`

        For that, `versions` should give three questions, without the `- ` and the spaces. The check also tries a numbered list, empty lines and lines without a `- `.
        """.format(short=C.SHORT[C.VERSIONS_SHOWN], raw=repr(C.written(C.VERSIONS_SHOWN, "versions")))),
        L.task("For each line: strip it, cut off a `- ` in front or a digit with a point, strip it again, and add it to `found` only when it is not empty."),
    ])
    return


@app.cell(hide_code=True)
def exercise_versions():
    box_versions = L.code_box("versions", lines=12)
    box_versions
    return (box_versions,)


@app.cell(hide_code=True)
def check_versions(box_versions):
    result_versions = L.check("versions", box_versions.value)
    result_versions
    return (result_versions,)


@app.cell(hide_code=True)
def hints_versions():
    L.hints(
        "versions",
        [
            "Take each line apart in three steps: the spaces around it, then what the list puts in front, then the spaces after that. An empty line ends up as `\"\"`.",
            "`line.startswith(\"- \")` says whether the line starts with a dash, and `line[2:]` cuts off the first two characters. For a number: `line[0].isdigit() and line[1] == \".\"`, once you know the line has at least two characters.",
            "In the loop: `line = line.strip()`, then `if line.startswith(\"- \"):` with `line = line[2:]`, an `elif len(line) > 1 and line[0].isdigit() and line[1] == \".\":` with the same, then `line = line.strip()` again, and `if line != \"\":` with `found.append(line)`.",
        ],
        why="""```python
def versions(text):
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
```

What a model writes is text, not a list. What it puts in front of a line and the spaces around it differ from model to model and from answer to answer, so the code that reads it has to forgive them.""",
    )
    return


@app.cell(hide_code=True)
def text_multi():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Search with every version

        `multi_search(texts, n)` gets several texts that ask the same thing, and should give the `n` points with the highest RRF score over their rankings, the highest first. Given are:

        - `embed(text)`: Granite's embedding of a text, from the course's recorded embeddings;
        - `meaning_search(q, vectors, points, k)`: the one of [Search by meaning](/course/open/build-retrieval-embeddings?lang=en);
        - `rrf(rankings, k)`: the one of [Words and meaning together](/course/open/build-hybrid-search?lang=en);
        - `POINTS` and `VECTORS`: the points of the course and their embeddings.

        The box makes a ranking of all points for every text, but gives the first `n` of the first ranking. `QUERIES["{qid}"]` is the short question *{short}* with Qwen3's three versions; for it, `multi_search` should give the right point first. The check also tries other questions, another `n`, two texts and one.
        """.format(qid=C.MULTI_SHOWN, short=C.SHORT[C.MULTI_SHOWN])),
        L.task("Merge the rankings with `rrf(rankings, 60)`, sort the points by their score with the highest first, and give the first `n`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_multi():
    box_multi = L.code_box("multi_search", lines=8)
    box_multi
    return (box_multi,)


@app.cell(hide_code=True)
def check_multi(box_multi):
    result_multi = L.check("multi_search", box_multi.value)
    result_multi
    return (result_multi,)


@app.cell(hide_code=True)
def hints_multi():
    _f = C.findings()
    L.hints(
        "multi_search",
        [
            "It is the merge of [Words and meaning together](/course/open/build-hybrid-search?lang=en) again, now with a ranking for every text instead of one for words and one for meaning.",
            "`scores = rrf(rankings, 60)` gives a dict of point and score. Pairs `(score, text)` sort by the score.",
            "Replace the last line by three: `scores = rrf(rankings, 60)`, then `best = sorted([(score, text) for text, score in scores.items()], reverse=True)`, and return `[text for score, text in best[:n]]`.",
        ],
        why="""```python
def multi_search(texts, n):
    rankings = []
    for text in texts:
        rankings.append(meaning_search(embed(text), VECTORS, POINTS, len(POINTS)))
    scores = rrf(rankings, 60)
    best = sorted([(score, text) for text, score in scores.items()], reverse=True)
    return [text for score, text in best[:n]]
```

A point that comes high for several versions gets a high score, also when no single version puts it first. Whether that helps depends on the versions: with Qwen3's, the right point came higher than with the short question alone for {better} questions, and lower for {worse}.""".format(better=_f["multi_better"], worse=_f["multi_worse"]),
    )
    return


@app.cell(hide_code=True)
def text_drift():
    _d = C.places(C.DRIFT)
    mo.md(r"""
    ### Exercise 3. A passage that leads astray

    For the short question *{short}*, Qwen3 wrote this passage to search with: *{passage}* The search put the right point at place {hyde}; with the short question itself it came at place {alone}: *{gold}*
    """.format(short=C.SHORT[C.DRIFT], passage=C.written(C.DRIFT, "hyde"), hyde=C.number(_d["hyde"], L.lang), alone=_d["short"],
               gold=C.gold(C.question(C.DRIFT))))
    return


@app.cell(hide_code=True)
def choice_drift_cell():
    _d = C.places(C.DRIFT)
    choice_drift = L.choice(
        "hyde-drift",
        question="Why did the passage send the search so far off?",
        options={
            "long": "Two sentences are too long: the embedding of a long text says less",
            "unknown": "Qwen3 did not know this course and wrote about something else",
            "words": "The passage should be searched with BM25, as its words are rare",
        },
        correct="unknown",
        why={
            "unknown": "Nothing in the passage is about an opening or about specks: Qwen3 wrote about AI in general. HyDE writes from what the model knows, so it helps only when the model knows the subject well enough to write something close to the answer.",
            "long": "The points of the course are about as long, and Granite embeds them well. Read what Qwen3 wrote: nothing in it is about an opening or about specks. Qwen3 does not know this course, so it wrote about AI in general.",
            "words": "BM25 would not find it either: the passage shares hardly a word with the right point. Nothing in it is about an opening or about specks. Qwen3 does not know this course, so it wrote about AI in general.",
        },
    )
    choice_drift
    return (choice_drift,)


@app.cell(hide_code=True)
def text_lost():
    _l = C.places(C.LOST)
    mo.md(r"""
    ### Exercise 4. A rewrite that lost something

    *{earlier}* And then: *{follow}* Qwen3 rewrote the follow-up as *{rewrite}* With that, the search put the right point at place {rewritten}; the follow-up after the earlier question found it at place {joined}.
    """.format(earlier=C.FOLLOW[C.LOST][0], follow=C.FOLLOW[C.LOST][1], earlier_nl=C.FOLLOW_NL[C.LOST][0], follow_nl=C.FOLLOW_NL[C.LOST][1], rewrite=C.written(C.LOST, "rewrite"), rewritten=_l["rewrite"], joined=_l["joined"]))
    return


@app.cell(hide_code=True)
def choice_lost_cell():
    _l = C.places(C.LOST)
    choice_lost = L.choice(
        "lost-subject",
        question="What went wrong in the rewrite?",
        options={
            "rare": "The words log scale are too rare for search by meaning",
            "answer": "It should also hold the answer to the earlier question",
            "lost": "It left out what the curve is: loss against size",
        },
        correct="lost",
        why={
            "lost": "The rewrite says the curve, but not of what. The follow-up after the earlier question still had the loss and the size, and found the right point at place {joined}. A rewrite has to bring in what the follow-up points back to: check that it does, on chats of your own.".format(joined=_l["joined"]),
            "rare": "The follow-up after the earlier question has the same words, and found the right point at place {joined}. Compare the rewrite with the earlier question: the loss and the size are gone.".format(joined=_l["joined"]),
            "answer": "The search needs to know what the question is about, not its answer: the follow-up after the earlier question found the right point at place {joined}. Compare the rewrite with the earlier question: the loss and the size are gone.".format(joined=_l["joined"]),
        },
    )
    choice_lost
    return (choice_lost,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Six ways to ask

    Once exercises 1 and 2 are right, pick a question. You see what the search got for each way of asking, and where it put the right point; the last row reads Qwen3's versions with your `versions` and searches with your `multi_search`.
    """)
    return


@app.cell(hide_code=True)
def settings_cell():
    question_pick = mo.ui.dropdown(options=C.question_options(L.lang), value=C.question("q18")["text"][L.lang], label="Question", full_width=True)
    question_pick
    return (question_pick,)


@app.cell(hide_code=True)
def demo_ways(box_multi, box_versions, question_pick):
    _fns = C.learner_multi(box_versions.value, box_multi.value)
    if _fns is None:
        _out = mo.md("*Finish exercises 1 and 2 first: the last row uses your own `versions` and `multi_search`.*")
    else:
        _out = mo.Html(C.tried(_fns, question_pick.value, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** A RAG in a chat needs the earlier questions in its search, as here, and the reranker of [A second look: reranking](/course/open/build-reranking?lang=en) can then order what comes back. Whether a change helps, you only know by measuring; a later lesson measures a RAG in two parts, the search and the answer.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_drift, choice_lost, result_multi, result_versions):
    L.wrap_up(result_versions, result_multi, choice_drift, choice_lost)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
