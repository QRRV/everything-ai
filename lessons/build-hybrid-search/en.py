# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-hybrid-search. Route: course/route/lessen/bouwen.yaml. Checks, the two searches and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-hybrid-search", "en")
    C = L.checks   # the course's points with their embeddings, BM25, the checks and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Search by meaning](/course/open/build-retrieval-embeddings?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_meaning():
    _f = C.findings()
    quick_meaning = L.choice(
        "quick-meaning",
        question="Which question does search by meaning find a point for better than BM25?",
        options={
            "names": "A question that asks for a rare name or number from the point",
            "long": "A question that is much longer than the point it should find",
            "other-words": "A question that says what the point says, in other words",
        },
        correct="other-words",
        why={
            "other-words": "Embeddings of texts that mean the same lie close together, whatever their words. BM25 only counts the words a question shares with a point: of the {n} questions in other words, search by meaning found {m}, BM25 {b}.".format(
                n=_f["meaning_questions"], m=_f[("en", "meaning", "meaning")], b=_f[("en", "bm25", "meaning")]),
            "names": "That is where BM25 is strong: a rare word that both have gives a high score. On the questions that share words with their point, BM25 was a little better. See [Search by meaning](/course/open/build-retrieval-embeddings?lang=en).",
            "long": "The length does not decide it: the embedding of any text has 384 numbers. What counts is whether the question and the point mean the same. See [Search by meaning](/course/open/build-retrieval-embeddings?lang=en).",
        },
    )
    quick_meaning
    return (quick_meaning,)


@app.cell(hide_code=True)
def text_why():
    _f = C.findings()
    mo.md(r"""
    ## Why this matters

    In [Search by meaning](/course/open/build-retrieval-embeddings?lang=en), search by meaning put the right point among the three best for {m} of the {n} questions, BM25 for {b}. But they did not miss the same ones. *{q20}* BM25 put its point first, search by meaning only at place {q20_m}. *{q18}* Search by meaning put its point first, BM25 at place {q18_b}. Names, codes and numbers are found by their words; the same idea in other words is found by its meaning. A search that uses both can find both.

    **An everyday example.** Two friends each give you their ten favorite restaurants, in order. One gives stars, the other goes by a feeling, so you cannot add up what they say; but you can see where a restaurant stands on each list, and go by that. That is only a comparison: the search below merges two rankings of points.
    """.format(m=_f[("en", "meaning")], n=_f["questions"], b=_f[("en", "bm25")], q20=C.question(C.WORKED)["text"]["en"],
               q20_m=C.places(C.WORKED)["meaning"], q18=C.question(C.LOST)["text"]["en"], q18_b=C.places(C.LOST)["bm25"],
               q20_nl=C.question(C.WORKED)["text"]["nl"], q18_nl=C.question(C.LOST)["text"]["nl"]))
    return


@app.cell(hide_code=True)
def text_idea():
    _w = C.worked()
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **hybrid search** | searching with words, such as with BM25, and by meaning, and merging the two rankings into one |
    | **rank** | the place of a text in a ranking: 1 for the first, 2 for the second |
    | **reciprocal rank fusion** (RRF) | merging rankings by giving every text the sum of `1 / (k + rank)` over the rankings, and sorting on that sum |
    | `k` | the number that RRF adds to every rank; 60 is the usual value |

    Why not add up the two scores? They are not on the same scale. For the question of the worked example, BM25 gave the points scores from {bl} to {bh}, the dot product of the embeddings from {ml} to {mh}.

    Reciprocal rank fusion, proposed by Cormack, Clarke and Büttcher in 2009, looks only at the places. Every ranking gives each text `1 / (k + rank)`, and the parts add up. With `k = 60`, place 1 gives 1/61 = {p1}, place 10 gives 1/70 = {p10}. A text that is in only one ranking gets only that one part. Elasticsearch, OpenSearch, Weaviate and Qdrant can do hybrid search with RRF built in.
    """.format(bl=C.number(_w["bm25_low"], L.lang, 0), bh=C.number(_w["bm25_high"], L.lang, 2), ml=C.number(_w["meaning_low"], L.lang, 3),
               mh=C.number(_w["meaning_high"], L.lang, 3), p1=C.number(1 / 61, L.lang, 4), p10=C.number(1 / 70, L.lang, 4)))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: hybrid search

    1. **Search twice:** rank the points with BM25, and by meaning.
    2. **Score:** for every ranking, give each point `1 / (k + rank)`, and add the parts up.
    3. **Keep:** the `n` points with the highest sum, the highest first.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _w = C.worked()
    mo.md(r"""
    ### Worked example

    *{question}*

    1. **Search twice:** BM25 puts first a point of *{w0}*: *{gold}* It answers the question. By meaning, the three best are points of the two lessons on coding agents, models that work on your code for you: *{m0}* and *{m1}*; the right point comes at place {gm}.
    2. **Score:** the right point gets 1/(60 + {gb}) + 1/(60 + {gm}) = {gs}. The first by meaning is at place {fb} by BM25: *{first}* It gets 1/(60 + {fb}) + 1/(60 + {fm}) = {fs}. A point of *{h1}*, at place {hb} by BM25 and {hm} by meaning, gets {hs}.
    3. **Keep:** the right point first, with {gs}, then the point of *{h1}* and the first by meaning.
    """.format(question=_w["question"]["text"]["en"], question_nl=_w["question"]["text"]["nl"], w0=C.title_of(_w["words"][0], L.lang), gold=_w["gold"],
               m0=C.title_of(_w["meaning"][0], L.lang), m1=C.title_of(_w["meaning"][1], L.lang), gb=_w["gold_places"][0],
               gm=_w["gold_places"][1], gs=C.number(_w["gold_score"], L.lang, 4), first=_w["first"], fb=_w["first_places"][0],
               fm=_w["first_places"][1], fs=C.number(_w["first_score"], L.lang, 4), h1=C.title_of(_w["hybrid"][1], L.lang),
               hb=_w["hybrid_places"][1][0], hm=_w["hybrid_places"][1][1], hs=C.number(_w["hybrid_scores"][1], L.lang, 4)))
    return


@app.cell(hide_code=True)
def text_measured():
    mo.md(r"""
    ### What the course measured

    For each of the {n} questions a point answers, the course merged BM25's ranking of all {points} points and the ranking by meaning, with `k = 60`, and looked whether the right point was among the three best. Then it did the same with the questions in Dutch, against the same English points:
    """.format(n=C.findings()["questions"], points=C.number(len(C.POINTS), L.lang)))
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
        Asked in English, the merged search won {won} questions from search by meaning, such as the one of the worked example, and lost {lost}: two questions where search by meaning put the right point first and BM25 far down, at places {b18} and {b21}.

        Asked in Dutch, BM25 found the right point for only {b_nl} of the {n}: it can only go by the words a Dutch question shares with an English point. The merged search found {h_nl}, fewer than the {m_nl} of search by meaning: the weak places of BM25 pulled right points down. With `k = 1`, place 1 gives 1/2 and place 10 gives 1/11, so a first place counts much more. Then the merged search found {s_nl} in Dutch, and {s_en} in English, as many as with `k = 60`.
        """.format(won=len(_f["won"]), lost=len(_f["lost"]), b18=C.places(_f["lost"][0])["bm25"], b21=C.places(_f["lost"][1])["bm25"],
                   b_nl=_f[("nl", "bm25")], n=_f["questions"], h_nl=_f[("nl", "hybrid")], m_nl=_f[("nl", "meaning")],
                   s_nl=_f[("nl", "small")], s_en=_f[("en", "small")])),
        L.watch_out("Merging is not better by itself: on the same points, it helped the questions in English and hurt those in Dutch. Measure it on questions of your own, asked the way your users ask. When one search is much weaker, a smaller `k`, or leaving that search out, can do better."),
    ])
    return


@app.cell(hide_code=True)
def text_rrf():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Merge rankings

        `rrf(rankings, k)` should give a dict with, for every text in the rankings, the sum of `1 / (k + rank)` over the rankings it is in. `rankings` is a list of rankings, each a list of texts, the best first. The box already gives every text the score 0. `rrf([["A", "B", "C"], ["C", "A", "D"]], 60)` should give, rounded, `{"A": 0.0325, "B": 0.0161, "C": 0.0323, "D": 0.0159}`: `"A"` is first in one ranking and second in the other, so it gets 1/61 + 1/62. The check also shows what your function gives the points for the question of the worked example.
        """),
        L.task("Add `1 / (k + rank)` to the score of a text each time it is in a ranking, with rank 1 for the first."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rrf():
    box_rrf = L.code_box("rrf", lines=9)
    box_rrf
    return (box_rrf,)


@app.cell(hide_code=True)
def check_rrf(box_rrf):
    result_rrf = L.check("rrf", box_rrf.value)
    result_rrf
    return (result_rrf,)


@app.cell(hide_code=True)
def hints_rrf():
    L.hints(
        "rrf",
        [
            "Each time a text is in a ranking, its score grows by `1 / (k + rank)`. The rank of `ranking[place]` is one more than `place`: the first text has `place` 0 and rank 1.",
            "Add one line in the inner loop, after the `if` and at the same depth as the `if`, so that it runs for every text: `scores[text] = scores[text] + ...`.",
            "The new line: `scores[text] = scores[text] + 1 / (k + ___)`, with the rank in the gap.",
        ],
        why="""```python
def rrf(rankings, k):
    scores = {}
    for ranking in rankings:
        for place in range(len(ranking)):
            text = ranking[place]
            if text not in scores:
                scores[text] = 0
            scores[text] = scores[text] + 1 / (k + place + 1)
    return scores
```

A text gets a part from every ranking it is in; a text in only one ranking gets only that part.""",
    )
    return


@app.cell(hide_code=True)
def text_hybrid():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Search with both

        `hybrid_search(question, q, n)` should give the `n` points with the highest score of `rrf` over the ranking of BM25 and the ranking by meaning, the highest first. `question` is the text of the question and `q` its embedding, made by Granite; a check message writes it as `QVECTORS[i]`, the embedding of question `i` of the course's list. Given are:

        - `bm25_search(question, texts, k)`: the `k` texts with the highest BM25 score, the highest first;
        - `meaning_search(q, vectors, points, k)`: the `k` points whose embeddings have the highest dot product with `q`, as in [Search by meaning](/course/open/build-retrieval-embeddings?lang=en);
        - `rrf`: the one of exercise 1;
        - `POINTS` and `VECTORS`: the {points} points of the course and their embeddings.

        The box ranks all points both ways, and gives the first of search by meaning alone. For the question of the worked example, `hybrid_search` should give the three points of step 3, the right point first. The check also tries other questions, and other `n`.
        """.format(points=C.number(len(C.POINTS), L.lang))),
        L.task("Merge `words` and `meaning` with `rrf`, `k = 60`, and give the `n` points with the highest score."),
    ])
    return


@app.cell(hide_code=True)
def exercise_hybrid():
    box_hybrid = L.code_box("hybrid_search", lines=7)
    box_hybrid
    return (box_hybrid,)


@app.cell(hide_code=True)
def check_hybrid(box_hybrid):
    result_hybrid = L.check("hybrid_search", box_hybrid.value)
    result_hybrid
    return (result_hybrid,)


@app.cell(hide_code=True)
def hints_hybrid():
    L.hints(
        "hybrid_search",
        [
            "Give `rrf` both rankings, and then take the `n` texts of its dict with the highest score.",
            "`rrf([words, meaning], 60)` gives a dict from text to score. A list of pairs `(score, text)`, sorted with `reverse=True`, has the highest score first.",
            "`best = sorted([(score, text) for text, score in ___], reverse=True)`, then give back the texts of `best[:n]`.",
        ],
        why="""```python
def hybrid_search(question, q, n):
    words = bm25_search(question, POINTS, len(POINTS))
    meaning = meaning_search(q, VECTORS, POINTS, len(POINTS))
    scores = rrf([words, meaning], 60)
    best = sorted([(score, text) for text, score in scores.items()], reverse=True)
    return [text for score, text in best[:n]]
```

Both searches rank all points, so every point gets a part from each. A dict keeps its texts in the order they came in, here that of BM25; only sorting on the score gives the merged order.""",
    )
    return


@app.cell(hide_code=True)
def text_add():
    mo.md(r"""
    ### Exercise 3. Add the scores?

    A colleague merges the two searches another way: add the BM25 score and the dot product of every point, and sort on the sum.
    """)
    return


@app.cell(hide_code=True)
def choice_add_cell():
    _w = C.worked()
    _s = C.summed()
    _n = C.findings()["questions"]
    choice_add = L.choice(
        "add-scores",
        question="What order does that sum give?",
        options={
            "bm25": "Almost BM25's own order, since its scores are far larger",
            "meaning": "Almost the order by meaning, since its scores lie close together",
            "same": "The same order as reciprocal rank fusion gives on both lists",
        },
        correct="bm25",
        why={
            "bm25": "For the question of the worked example, BM25 gave scores from 0 to {bh}, the dot product from {ml} to {mh}: one small step in BM25 outweighs all of meaning. The course tried it: for {same} of the {n} questions, the three best of the sum were exactly BM25's three best. Places are on the same scale for both searches; that is why RRF uses them.".format(
                bh=C.number(_w["bm25_high"], L.lang, 2), ml=C.number(_w["meaning_low"], L.lang, 3), mh=C.number(_w["meaning_high"], L.lang, 3),
                same=_s["same"], n=_n),
            "meaning": "It is the other way round: scores that lie close together hardly change a sum. BM25's scores ran from 0 to {bh}, so they decide: for {same} of the {n} questions, the three best of the sum were exactly BM25's three best.".format(
                bh=C.number(_w["bm25_high"], L.lang, 2), same=_s["same"], n=_n),
            "same": "RRF looks only at the places, which are on the same scale for both searches. A sum is decided by the largest numbers, those of BM25: for {same} of the {n} questions, the three best of the sum were exactly BM25's three best.".format(
                same=_s["same"], n=_n),
        },
    )
    choice_add
    return (choice_add,)


@app.cell(hide_code=True)
def text_first():
    _l = C.lost()
    mo.md(r"""
    ### Exercise 4. Not in first place

    *{question}* Search by meaning put the right point first: *{gold}* BM25 put it at place {b}, and the merged search, with `k = 60`, only at place {h}. The merged first was a point of *{t}*, at place {fb} by BM25 and {fm} by meaning: *{first}*
    """.format(question=C.question(C.LOST)["text"]["en"], question_nl=C.question(C.LOST)["text"]["nl"], gold=C.gold(C.question(C.LOST)), b=_l["places"]["bm25"], h=_l["places"]["hybrid"],
               t=C.title_of(_l["first"], L.lang), fb=_l["first_places"][0], fm=_l["first_places"][1], first=_l["first"]))
    return


@app.cell(hide_code=True)
def choice_first_cell():
    _l = C.lost()
    choice_first = L.choice(
        "first-place",
        question="Why did the merged search put that point above the right one?",
        options={
            "bm25": "BM25 weighs more in the sum, because its scores are many times higher",
            "both": "Both lists put it fairly high, and with k = 60 that counts for more",
            "count": "RRF counts how often a point is in the lists, and adds that up",
        },
        correct="both",
        why={
            "both": "With `k = 60`, a first place gives 1/61 and an eighth 1/68: not much less. Places {fb} and {fm} gave {fs}, places 1 and {b} gave {s}. With `k = 1`, a first place gives 1/2 and an eighth 1/9: then the right point came at place {small}.".format(
                fb=_l["first_places"][0], fm=_l["first_places"][1], fs=C.number(_l["first_score"], L.lang, 4), b=_l["places"]["bm25"],
                s=C.number(_l["score"], L.lang, 4), small=_l["places"]["small"]),
            "bm25": "RRF never sees the scores of BM25, only the places: every ranking adds `1 / (k + rank)`. Places {fb} and {fm} gave {fs}, places 1 and {b} gave {s}.".format(
                fb=_l["first_places"][0], fm=_l["first_places"][1], fs=C.number(_l["first_score"], L.lang, 4), b=_l["places"]["bm25"],
                s=C.number(_l["score"], L.lang, 4)),
            "count": "Every point is in both rankings, so a count would give each the same. RRF adds `1 / (k + rank)`: places {fb} and {fm} gave {fs}, places 1 and {b} gave {s}.".format(
                fb=_l["first_places"][0], fm=_l["first_places"][1], fs=C.number(_l["first_score"], L.lang, 4), b=_l["places"]["bm25"],
                s=C.number(_l["score"], L.lang, 4)),
        },
    )
    choice_first
    return (choice_first,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Search the course with your own hybrid_search

    Once exercises 1 and 2 are right, pick a question. The course searches its points with your `hybrid_search` and your `rrf`, beside the three best of BM25 and of search by meaning, each point with its two places. The questions are in the language of the lesson.
    """)
    return


@app.cell(hide_code=True)
def settings_cell():
    question_pick = mo.ui.dropdown(options=C.question_options(L.lang), value=C.question("q23")["text"][L.lang], label="Question", full_width=True)
    question_pick
    return (question_pick,)


@app.cell(hide_code=True)
def demo_search(box_hybrid, box_rrf, question_pick):
    _fn = C.learner_search(box_rrf.value, box_hybrid.value)
    if _fn is None:
        _out = mo.md("*Finish exercises 1 and 2 first: the search uses your own `rrf` and `hybrid_search`.*")
    else:
        _out = mo.Html(C.searched(_fn, question_pick.value, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The search in [Your first RAG](/course/open/build-rag-first?lang=en) can now be a hybrid one: the merged points go into the prompt, as in [Answers from the sources](/course/open/build-grounded-answers?lang=en). A fast search, by words, by meaning or both, gives a few candidates; a later lesson lets a model read each of them together with the question, and put them in a better order.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_add, choice_first, result_hybrid, result_rrf):
    L.wrap_up(result_rrf, result_hybrid, choice_add, choice_first)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
