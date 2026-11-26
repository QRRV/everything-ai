# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ir-ndcg. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ir-ndcg", "en")
    C = L.checks   # the course's thirteen documents with grades, the rankings of XQuAD and SciFact, and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [Did the search work? P@k, MAP and MRR](/course/open/ir-precision-at-k?lang=en) and [Exponents and logarithms](/course/open/math-exp-log?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_ap():
    quick_ap = L.choice(
        "quick-ap",
        question="A query has two relevant documents, and a ranking puts them at places 1 and 4. What are the reciprocal rank and the average precision of this ranking?",
        options={
            "last": "RR 0.25 and AP 0.75",
            "mean": "RR 1 and AP 0.625, the mean of 1 divided by each of the two places",
            "right": "RR 1 and AP 0.75, from the precisions 1 and 0.5",
            "added": "RR 1 and AP 1.5, the precisions 1 and 0.5 added up",
        },
        correct="right",
        why={
            "right": "The reciprocal rank looks only at the first relevant document: 1 / 1 = 1. The average precision takes the precision at each relevant document, 1 at place 1 and 2 / 4 = 0.5 at place 4, and divides by the 2 relevant documents: 0.75. Both count the two documents as equally relevant; this lesson lets one document count more than another.",
            "last": "Which relevant document does the reciprocal rank look at: the first or the last? See [Did the search work? P@k, MAP and MRR](/course/open/ir-precision-at-k?lang=en).",
            "mean": "At place 4, the precision is not 1 divided by 4: how many of the first four documents are relevant? See [Did the search work? P@k, MAP and MRR](/course/open/ir-precision-at-k?lang=en).",
            "added": "The precisions at the two relevant documents are 1 and 0.5, but what does the average precision divide by? See [Did the search work? P@k, MAP and MRR](/course/open/ir-precision-at-k?lang=en).",
        },
    )
    quick_ap
    return (quick_ap,)


@app.cell(hide_code=True)
def quick_check_log():
    quick_log = L.choice(
        "quick-log",
        question="What do `np.log2(4)` and `np.log2(1)` give?",
        options={
            "power": "16 and 2: 2 to the power of 4, and 2 to the power of 1",
            "right": "2 and 0: 2 to the power of 2 is 4, and 2 to the power of 0 is 1",
            "one": "2 and 1",
            "natural": "1.386 and 0: the log base e of 4, and the log base e of 1, as `np.log` gives them",
        },
        correct="right",
        why={
            "right": "`np.log2(x)` asks: 2 to what power gives x? 2 to the power of 2 is 4, and 2 to the power of 0 is 1. This lesson divides the value of a document by such a log: at place 1 by the log base 2 of 2, which is 1, and lower down by more.",
            "power": "Those are 2 to the power of 4 and of 1. The log goes the other way: 2 to what power gives 4? See [Exponents and logarithms](/course/open/math-exp-log?lang=en).",
            "one": "2 to the power of 2 is 4: right. But 2 to what power gives 1? See [Exponents and logarithms](/course/open/math-exp-log?lang=en).",
            "natural": "`np.log2` takes the log base 2, not base e: 2 to what power gives 4? See [Exponents and logarithms](/course/open/math-exp-log?lang=en).",
        },
    )
    quick_log
    return (quick_log,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Did the search work? P@k, MAP and MRR](/course/open/ir-precision-at-k?lang=en), a document was relevant or not. For the need about red wine and the heart, documents 6 and 9 were relevant, each as relevant as the other. But document 6 answers the question head-on: it compares red and white wine for the risk of a heart attack. Document 9 compares them too, alongside beer, in an overview of alcohol in general. And document 7, about heart-healthy food with now and then a glass of red wine, does not answer the question, but it is closer than document 8, about a walk through an old town. Measured as relevant or not, the order 6, 7, 8, 10, 9 and the order 9, 7, 8, 10, 6 get the same average precision: 0.7.

    Kalervo Järvelin and Jaana Kekäläinen wrote in 2002: "Since all documents are not of equal relevance to their users, highly relevant documents should be identified and ranked first for presentation." Their measure, nDCG, starts from a grade for each document, counts the grades of a ranking from the top down, each a little less the lower it stands, and compares the outcome with the best order there is. Many search benchmarks report it: the TREC Deep Learning track of 2019 measured with nDCG@10, the benchmark BEIR does so on 18 test collections, and the retrieval score of MTEB in [Choosing a sentence model: reading MTEB](/course/open/nlp-mteb?lang=en) is nDCG@10 too.

    **An everyday example.** You ask three people in a strange town for the way to the station. The first tells you the way, the second says it is somewhere near the market, and the third talks about the weather. Counted as helpful or not, the first two are as good as each other; but you would want the full answer first, and the half answer still before the weather.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_place = L.guess(
        "guess",
        question="nDCG counts a relevant document less the lower it stands, but less steeply than the reciprocal rank, which gives place 10 a tenth of what it gives place 1. How much does nDCG give a document at place 10, against the same document at place 1?",
        options={
            "tenth": "a tenth, as the reciprocal rank does",
            "right": "a bit less than a third",
            "half": "half",
            "most": "nine tenths: lower places count almost as much",
        },
        reveal="A bit less than a third: 0.289. nDCG divides the value of the document at place 10 by the log base 2 of 11, which is 3.459, and at place 1 by the log base 2 of 2, which is 1. Järvelin and Kekäläinen chose the log because people who keep looking reach the lower places too. In their words, with the log base 2, \"a document at the position 1024 would still get one tenth of its face value\".",
    )
    guess_place
    return (guess_place,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **Graded relevance.** In [Did the search work? P@k, MAP and MRR](/course/open/ir-precision-at-k?lang=en), a relevance judgment said relevant or not relevant. With graded relevance, it gives each document a grade. Järvelin and Kekäläinen start from the view that "highly relevant documents are more valuable than marginally relevant documents". The TREC Deep Learning track of 2019, organized by Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos and Ellen Voorhees, had assessors of NIST judge passages of web pages with four grades:

    | Grade | In TREC Deep Learning 2019 |
    |---|---|
    | 3 | "Perfectly relevant: The passage is dedicated to the query and contains the exact answer." |
    | 2 | "Highly relevant: The passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information." |
    | 1 | "Related: The passage seems related to the query but does not answer it." |
    | 0 | "Irrelevant: The passage has nothing to do with the query." |

    This lesson grades the course's thirteen documents on this scale. For need 3, red wine and the heart, document 6 gets 3, document 9 gets 2, document 7 gets 1, and the others 0. For need 1, a pet python to buy, document 1, *Ball pythons for sale*, gets 3, document 3, about a warm tank and where to buy one, gets 2, and document 5, about caring for a python at home, gets 1. For need 2, learning Python, document 0, *Python for beginners*, gets 3, and document 4, about a new version of Python, gets 1. The documents with grade 2 or 3 are the relevant documents of [Search: needs and relevance](/course/open/ir-what-is-search?lang=en).

    **Gain.** The ranked documents "are turned to gained value lists by replacing document IDs by their relevance scores", as Järvelin and Kekäläinen write: each document of the ranking is replaced by its gain, here its grade, and a document without a grade gets 0. The gains of the first k places added up are the cumulative gain (Järvelin and Kekäläinen write cumulated gain). It grows with each document that has a grade, but it does not care where among the first k that document comes.

    **Discount.** Järvelin and Kekäläinen again: "the greater the ranked position of a relevant document, the less valuable it is for the user, because the less likely it is that the user will ever examine the document". So the gain at each place is divided by a discount that grows with the place. They chose the log of the place, a discount that "progressively reduces the document score as its rank increases but not too steeply (e.g., as division by rank) to allow for user persistence in examining further documents". The usual form divides the gain at place i by log₂(i + 1): place 1 by 1, place 2 by 1.585, place 3 by 2, and place 10 by 3.459. That is the form of trec_eval, the program that TREC and BEIR compute their numbers with. In the paper of 2002, with the base 2 of its examples, the gain at place i is divided by log₂ i from place 2 on, so the first two places both count in full.

    **DCG.** The gains of the first k places, each divided by its discount and then added up, are the discounted cumulative gain, DCG@k. For need 3, BM25 gives documents 6, 7, 8, 10 and 9, with the gains 3, 1, 0, 0 and 2: DCG@5 = 3 / 1 + 1 / 1.585 + 2 / 2.585 = 4.405.

    **The ideal order.** Is 4.405 good? That depends on the grades the query has. Järvelin and Kekäläinen compare with the best order there is: all the graded documents of the query, the highest grade first. "Note that the ideal vector is based on the recall base of the search topic rather than on the result of some IR technique": the ideal order holds every document with a grade, also those that no ranking finds. Its DCG@k is the ideal DCG, IDCG@k. For need 3, the ideal order is 3, 2, 1: IDCG@5 = 3 / 1 + 2 / 1.585 + 1 / 2 = 4.762.

    **nDCG.** The DCG divided by the ideal DCG is the normalized DCG, nDCG@k: 1 for a ranking in the ideal order, and 0 for a ranking without a graded document among its first k. For need 3: 4.405 / 4.762 = 0.925. Because each query gets a number from 0 to 1, queries with many and with few relevant documents can share one mean. The TREC Deep Learning track took nDCG@10 as its main measure, "since it makes use of our 4-level judgments and focuses on the first results that users will see".

    **The gain of a grade.** trec_eval takes the grade itself as the gain: "Gain values are the relevance values in the qrels file." Chris Burges and colleagues at Microsoft, who in 2005 trained a model to rank the results of a web search engine on ratings from 1 to 5, took 2 to the power of the rating, minus 1, as the gain: a perfect result then counts far more than a fair one. Both are called nDCG, so a paper should say which gain it uses. This lesson takes the grade itself, as trec_eval and BEIR do; in the Try it, you can switch.

    **One relevant document.** With one relevant document, of grade 1, at place p, the ideal DCG is 1, and nDCG@k is 1 / log₂(p + 1), or 0 when p comes after k. The reciprocal rank gives 1 / p: at place 2 nDCG gives 0.631 where the reciprocal rank gives 0.5, and at place 10 0.289 against 0.1.

    For a query, with the gain $g_i$ at place $i$:

    $$\text{DCG@}k = \sum_{i=1}^{k} \frac{g_i}{\log_2(i + 1)} \qquad \text{nDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}$$

    where IDCG@k is the DCG@k of the ideal order of all the grades of the query; nDCG@k is 0 when the IDCG@k is 0. The mean over the queries is the mean nDCG@k.

    **The three needs.** The picture shows the three needs with the results of BM25, each with its gain, its discount and the gain divided by the discount. For need 1, BM25 misses document 1, of grade 3, and puts document 5 second and document 3 fourth: a DCG@5 of 1 / 1.585 + 2 / 2.322 = 1.492, against an ideal 4.762, so an nDCG@5 of 0.313. For need 2, document 0 comes first, as in the ideal order, but document 4, of grade 1, comes fifth instead of second: an nDCG@5 of 0.933, where the average precision, which sees only document 0, was 1. For need 3, the order 6, 7, 8, 10, 9 gets 0.925, and the order 9, 7, 8, 10, 6, with the same average precision, would get 0.796. Over the three needs, the mean nDCG@5 is 0.724, against a MAP of 0.608.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.md(r"""
    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **graded relevance** | relevance judgments with grades, such as 0 to 3, instead of only relevant or not relevant |
    | **gain** | what a document of a ranking is worth: here its grade, or 0 without one |
    | **cumulative gain** | the gains of the first k places, added up |
    | **discount** | what the gain at a place is divided by: log₂(place + 1), 1 at place 1 and 3.459 at place 10 |
    | **discounted cumulative gain**, DCG@k | the gains of the first k places, each divided by its discount, added up |
    | **ideal DCG**, IDCG@k | the DCG@k of the ideal order: all the grades of the query, the highest first |
    | **normalized DCG**, nDCG@k | the DCG@k divided by the IDCG@k: a number from 0 to 1 |
    | `sorted(..., reverse=True)` | a sorted list from high to low, where `sorted` alone goes from low to high |
    | $g_i$ | the gain at place $i$ |
    | $\log_2$ | the log base 2: $\log_2 8 = 3$, because $2^3 = 8$ |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To measure a ranking with graded relevance:

    1. Give each document a grade for the query, such as 0 to 3; a document without a grade gets 0.
    2. Turn the ranking into its gains: the grade of each document, from the top down.
    3. DCG@k: divide the gain at place i by log₂(i + 1), and add up the first k.
    4. Ideal DCG: put all the grades of the query from high to low, and take the DCG@k of that order.
    5. nDCG@k: the DCG@k divided by the ideal DCG, or 0 when the ideal DCG is 0.
    6. Take the mean over all queries.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `C.small_results(L.lang, need, how)` gives the results of the cosine (`"cosine"`) or of BM25 (`"bm25"`) for the query of a need, best first; `C.GRADES` holds the course's grades of each need, and `C.judged(need)` all the grades of a need. `C.gains` turns a ranking into its gains, and `C.dcg` and `C.ndcg` are the course's measures. `C.rankings`, `C.relevant` and `C.reciprocal_rank` are those of [Did the search work? P@k, MAP and MRR](/course/open/ir-precision-at-k?lang=en). First the three needs: the results, their gains, the DCG@5 and the nDCG@5. Then the mean nDCG@10 over all the questions of XQuAD, where the own paragraph has grade 1, next to the MRR:
    """)
    return


@app.cell
def example_cell():
    for _need in ("buy", "learn", "wine"):
        for _how in ("cosine", "bm25"):
            _results = C.small_results(L.lang, _need, _how)
            _gains = C.gains(_results, C.GRADES[_need])
            print(_need, _how, _results, _gains, round(C.dcg(_gains, 5), 3),
                  round(C.ndcg(_gains, C.judged(_need), 5), 3))
    for _how in ("cosine", "bm25"):
        _rankings = C.rankings(L.lang, _how)
        _relevants = C.relevant(L.lang)
        _n = len(_rankings)
        _ndcg = sum([C.ndcg(C.gains(_rankings[_q], {_relevants[_q][0]: 1}), [1], 10) for _q in range(_n)]) / _n
        _mrr = sum([C.reciprocal_rank(_rankings[_q], _relevants[_q]) for _q in range(_n)]) / _n
        print(_how, _n, round(_ndcg, 3), round(_mrr, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The first six lines are the three needs, for the cosine and for BM25. For need 3, both rankings give the same five documents, with the gains 3, 1, 0, 0 and 2, and an nDCG@5 of 0.925. For need 1, the cosine gets 0.42 and BM25 0.313: both miss document 1, of grade 3, but the cosine puts document 5, of grade 1, first and document 3, of grade 2, third, where BM25 has them second and fourth.

    The last two lines are the means over all 1,190 questions of XQuAD: an nDCG@10 of 0.886 for BM25 and 0.804 for the cosine, each a little above its MRR, 0.862 and 0.764. With one relevant paragraph of grade 1, a second place gets 0.631 instead of the 0.5 of the reciprocal rank.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_dcg():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. dcg: discounted cumulative gain

        Write `dcg`, which gets `gains`, the gains of a ranking as a list, from the top place down, and `k`. It gives back the DCG@k: each of the first `k` gains divided by its discount, and added up. In the start code, `total` starts at 0, `top = gains[:k]` keeps the first `k` gains, or all of them when the list is shorter, and `for place in range(len(top))` goes through their places as Python counts them, from 0. Replace the `...` with the discount of the place: `np.log2` of the place counted from 1, plus 1. The box already knows `np`.

        The check calls `dcg([3, 2, 3, 0, 0, 1, 2, 2, 3, 0], 3)`, which gives `5.762`: 3 / 1 + 2 / 1.585 + 3 / 2, the first three gains of the example of Järvelin and Kekäläinen. And `dcg([1, 0, 3], 5)`, which gives `2.5`: the list holds three gains, and the function counts what it has. Then cases of its own: k = 1, an empty list, and the course's gains.
        """),
        L.task("Divide each gain by the log base 2 of its place counted from 1, plus 1."),
    ])
    return


@app.cell(hide_code=True)
def exercise_dcg():
    box_dcg = L.code_box("dcg", lines=6)
    box_dcg
    return (box_dcg,)


@app.cell(hide_code=True)
def check_dcg(box_dcg):
    result_dcg = L.check("dcg", box_dcg.value)
    result_dcg
    return (result_dcg,)


@app.cell(hide_code=True)
def hints_dcg():
    L.hints(
        "dcg",
        [
            "Each gain `top[place]` is divided by its discount before it goes into `total`; the discount is `np.log2` of the place counted from 1, plus 1.",
            "`place` counts from 0, so the place counted from 1 is one more than `place`, and the log is taken of one more than that: at the first place the log base 2 of 2, which is 1.",
            "`        total = total + top[place] / np.log2(___)`",
        ],
        why="`total = total + top[place] / np.log2(place + 2)`: the first place is divided by the log base 2 of 2, which is 1, the second by 1.585, the tenth by 3.459.",
    )
    return


@app.cell(hide_code=True)
def text_after_dcg():
    mo.md(r"""
    In the picture, your function gives a gain of 1 at place 2 the value 0.631, and at place 10 the value 0.289, where the reciprocal rank gives 0.5 and 0.1: the log falls fast over the first places, and slowly after. For the three needs, your DCG@5 of the results of BM25 is 1.492, 3.387 and 4.405, and that of the ideal orders 4.762, 3.631 and 4.762. Need 2 has no document of grade 2, so its ideal order, 3 and 1, has a lower DCG: a DCG alone says little until it is compared with the ideal DCG of its own query.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ndcg():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. ndcg: compare with the ideal order

        Write `ndcg`, which gets `gains`, the gains of a ranking as in exercise 1, `judged`, a list of all the grades of the query in the relevance judgments, in any order, and `k`. It gives back the nDCG@k: the DCG@k of the gains, divided by `best`, the DCG@k of the ideal order. In the start code, `if best == 0` gives back 0 for a query without a document of grade 1 or more, and the last line divides. Replace the `...` with the ideal order: all the grades in `judged`, the highest first. The box already knows `dcg` of exercise 1.

        The check calls `ndcg([0, 3, 1], [2, 3, 1], 3)`, which gives `0.502`: the ideal order 3, 2, 1 has a DCG@3 of 4.762, and the ranking gets 2.393. And `ndcg([1, 0], [3, 1], 2)`, which gives `0.275`: the ranking misses the document of grade 3, which the ideal order still puts first. Then cases of its own: a query without relevant documents, an empty ranking, k = 1, and the course's gains.
        """),
        L.task("Give dcg all the grades of the query, from high to low."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ndcg():
    box_ndcg = L.code_box("ndcg", lines=5)
    box_ndcg
    return (box_ndcg,)


@app.cell(hide_code=True)
def check_ndcg(box_ndcg):
    result_ndcg = L.check("ndcg", box_ndcg.value)
    result_ndcg
    return (result_ndcg,)


@app.cell(hide_code=True)
def hints_ndcg():
    L.hints(
        "ndcg",
        [
            "`best` is the `dcg` of the ideal order at `k`: all the grades in `judged`, put in order by `sorted`, the highest first.",
            "`sorted` puts the lowest first; with `reverse=True` after the list, it puts the highest first: `sorted([1, 3, 2], reverse=True)` gives `[3, 2, 1]`.",
            "`    best = dcg(sorted(___), k)`",
        ],
        why="`best = dcg(sorted(judged, reverse=True), k)`: the ideal order holds every graded document of the query, also those the ranking misses, so only the best ranking there is gets 1.",
    )
    return


@app.cell(hide_code=True)
def text_after_ndcg():
    mo.md(r"""
    In the picture, your function measures the cosine and BM25 on three test collections. On the thirteen documents, BM25 gets a mean nDCG@5 of 0.724 and the cosine 0.662: closer together than their MAP, 0.608 and 0.456. For need 2, where BM25 puts document 0 first and the cosine second, the average precision is 1 against 0.5, and the nDCG@5 0.933 against 0.640: the log costs a second place less than the reciprocal rank does. On XQuAD, the nDCG@10 is a little above the MRR for both rankings. With SciFact downloaded, BM25 gets 0.662 and the cosine 0.580. BEIR reports 0.665 for its BM25, which uses k1 = 0.9 and b = 0.4 and searches the title as a field of its own, where the course's BM25 uses k1 = 1.2 and b = 0.75.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_dcg_all():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. dcg_all: every query at once

        Write `dcg_all`, which gets `gains`, a NumPy array with one row of gains for each query, at least `k` columns wide, and `k`, and gives back an array with the DCG@k of each row, without a loop. In the start code, `gains[:, :k]` keeps the first `k` columns of every row, the division by `discounts` divides each column by its own number, as in [Broadcasting](/course/open/np-broadcasting?lang=en), and `.sum(axis=1)` adds up each row. Replace the `...` with `discounts`: an array with the discount of each of the first `k` places, `np.log2` of 2, 3 and so on up to `k + 1`.

        The check calls `dcg_all(np.array([[3, 2, 3, 0], [1, 0, 3, 2]]), 3)`, which gives `np.array([5.762, 2.5])`: the first row starts as the example of exercise 1. And `dcg_all(np.array([[0, 1], [2, 0]]), 2)`, which gives `np.array([0.631, 2.0])`. Then cases of its own: k = 1, a k as long as the rows, and the course's gains of XQuAD.
        """),
        L.task("Make the discounts of the first k places as one array."),
    ])
    return


@app.cell(hide_code=True)
def exercise_dcg_all():
    box_dcg_all = L.code_box("dcg_all", lines=4)
    box_dcg_all
    return (box_dcg_all,)


@app.cell(hide_code=True)
def check_dcg_all(box_dcg_all):
    result_dcg_all = L.check("dcg_all", box_dcg_all.value)
    result_dcg_all
    return (result_dcg_all,)


@app.cell(hide_code=True)
def hints_dcg_all():
    L.hints(
        "dcg_all",
        [
            "`discounts` holds the discount of each of the first `k` places: `np.log2` of the place counted from 1, plus 1, for all places at once.",
            "`np.arange(a, b)` gives the whole numbers from `a` up to `b`, without `b` itself: `np.arange(2, 5)` gives 2, 3 and 4. `np.log2` of an array takes the log of each number in it.",
            "`    discounts = np.log2(___)`",
        ],
        why="`discounts = np.log2(np.arange(2, k + 2))`: the logs of 2 up to k + 1, so place 1 is divided by 1 and place k by the log base 2 of k + 1.",
    )
    return


@app.cell(hide_code=True)
def text_after_dcg_all():
    mo.md(r"""
    In the picture, BM25 gets a mean nDCG@1 of 0.800, nDCG@3 of 0.866, nDCG@5 of 0.879 and nDCG@10 of 0.886; the cosine gets 0.666, 0.766, 0.788 and 0.804. The nDCG@1 is the precision at 1: with one relevant paragraph, only the first place counts, and in full. With each larger k, a few more own paragraphs count, each for less. The perfect ranking gets 1 for every k, because its DCG is the ideal DCG.

    ### Try it: grade the documents yourself

    Grades are judgments, and people judge differently. Pick a need, and give each document a grade from 0 to 3 with its slider; the sliders start at the course's grades. The picture shows the results of the cosine and of BM25 with your grades, their DCG@5 and nDCG@5, next to the nDCG@5 with the course's grades and the average precision, which counts grades 2 and 3 as relevant. Under the need, pick the gain: the grade itself, or 2 to the power of the grade, minus 1.

    Start with need 3, and give document 9 a 3 as well: why does the nDCG@5 of both rankings go down, while the rankings stay the same? Then put document 9 back at 2, pick the other gain, and compare need 3 with need 1: where does the nDCG@5 go up, and where down?
    """)
    return


@app.cell(hide_code=True)
def try_choose():
    _needs = C.try_options(L.lang)
    try_need = mo.ui.dropdown(options=_needs, value=next(t for t, n in _needs.items() if n == C.TRY_NEED), label="Need")
    try_gain = mo.ui.radio(options=C.GAIN_OPTIONS[L.lang], value=next(iter(C.GAIN_OPTIONS[L.lang])), label="Gain")
    mo.vstack([try_need, try_gain])
    return try_gain, try_need


@app.cell(hide_code=True)
def try_grades_cell(try_need):
    _sliders = C.try_sliders(L.lang, try_need.value)
    try_grades = mo.ui.array([mo.ui.slider(0, 3, step=1, value=v, show_value=True) for _t, v in _sliders])
    mo.vstack([mo.hstack([mo.md(_t), try_grades[_j]], widths=[3, 1], align="center")
               for _j, (_t, _v) in enumerate(_sliders)], gap=0.25).style({"max-width": "44rem"})
    return (try_grades,)


@app.cell(hide_code=True)
def try_look(try_gain, try_grades, try_need):
    mo.Html(C.try_picture(L.lang, try_need.value, try_grades.value, try_gain.value))
    return


@app.cell(hide_code=True)
def text_after_try():
    mo.md(r"""
    With document 9 at 3, the DCG@5 of both rankings goes up, from 4.405 to 4.791, but the ideal order goes up more: 3, 3, 1 gives an IDCG@5 of 5.393, so the nDCG@5 goes down from 0.925 to 0.888. Two documents of grade 3 now belong at the top, and the rankings have one of them fifth. With the gain 2 to the power of the grade, minus 1, a 3 counts for 7 and a 1 for 1. For need 3, where both rankings put document 6 first, the nDCG@5 goes up from 0.925 to 0.936; for need 1, where both rankings miss document 1, of grade 3, the nDCG@5 of BM25 goes down from 0.313 to 0.205. The steeper gain rewards finding the best document more, and punishes missing it more.

    Grades do not fix every limit of the judgments: a document that nobody judged counts as 0, even when it would help. Burges and colleagues wrote of $N_i$, the nDCG of query $i$: "also note that $N_i = 1$ is an unlikely event, even for a perfect ranker, since some unlabeled documents may in fact be highly relevant."

    ### Exercise 4. Up, down or the same?

    Sort each change below by what it does to the nDCG@10 of a query, with the grade as the gain.
    """)
    return


@app.cell(hide_code=True)
def exercise_changes():
    sort_changes = L.sort("changes")
    sort_changes
    return (sort_changes,)


@app.cell(hide_code=True)
def check_changes(sort_changes):
    result_changes = L.check("changes", sort_changes)
    result_changes
    return (result_changes,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The rankings measured here are those of [Ranking by cosine: the vector space model](/course/open/ir-vector-space?lang=en) and [BM25: the classic ranking](/course/open/ir-bm25?lang=en), and nDCG takes the measures of [Did the search work? P@k, MAP and MRR](/course/open/ir-precision-at-k?lang=en) from relevant or not to grades. The retrieval score of MTEB in [Choosing a sentence model: reading MTEB](/course/open/nlp-mteb?lang=en) is the nDCG@10 of this lesson, times 100. The next lesson, *Snippets: showing the results*, looks at what a page of results shows of each document; and *From ten links to one answer* looks at what changes when a search engine writes one answer instead of a ranked list.

    **An everyday example.** A playlist puts your favorite song first, a song you like a little further down, and songs you skip at the end. The gain is how much you like each song, the discount says that you may stop listening before the end, and nDCG compares the playlist with the best order of the same songs.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_dcg, result_ndcg, result_dcg_all, result_changes):
    L.wrap_up(result_dcg, result_ndcg, result_dcg_all, result_changes)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
