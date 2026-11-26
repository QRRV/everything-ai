# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ir-precision-at-k. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ir-precision-at-k", "en")
    C = L.checks   # the course's thirteen documents, the rankings of XQuAD, SciFact and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en) and [BM25: the classic ranking](/course/open/ir-bm25?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_precision():
    quick_precision = L.choice(
        "quick-precision",
        question="A search returns 8 documents, and 6 of them are relevant. The collection holds 10 relevant documents in all. What are the precision and the recall of this search?",
        options={
            "swap": "precision 0.6 and recall 0.75",
            "found": "precision 0.8 and recall 0.6, from 8 of 10 and 6 of 10",
            "right": "precision 0.75 and recall 0.6, from 6 of 8 and 6 of 10",
            "same": "both 0.75, because precision and recall count the same 6 relevant documents",
        },
        correct="right",
        why={
            "right": "Precision divides the 6 relevant results by the 8 results: 0.75. Recall divides them by the 10 relevant documents: 0.6. This lesson takes precision into a ranked list: the precision of the first k results.",
            "swap": "Which number counts the results, and which the relevant documents: does precision look at what the search returned, or at what it should have returned? See [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).",
            "found": "The search returns 8 documents, but are all 8 relevant? What does precision divide, and by what? See [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).",
            "same": "Both count the 6 relevant results, but do they divide by the same number? See [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).",
        },
    )
    quick_precision
    return (quick_precision,)


@app.cell(hide_code=True)
def quick_check_tune():
    quick_tune = L.choice(
        "quick-tune",
        question="In *BM25: the classic ranking*, k1 and b were tuned on the 809 training claims of SciFact, and then measured on the 300 test claims. Why not tune them on the test claims as well?",
        options={
            "few": "300 claims are too few to try all 651 settings of the grid on",
            "right": "a setting picked on the test claims looks better there than it will on new claims",
            "none": "the test claims have no relevant abstracts to tune on",
            "slow": "tuning takes too long on the test claims, because their abstracts are much longer than those of the training claims",
        },
        correct="right",
        why={
            "right": "A setting picked by looking at the test claims fits those claims, and promises more than it delivers on new ones: tuned on the training claims, BM25 put a relevant abstract first for 53.7% of the test claims, and the best setting on the test claims themselves got 56.0%. This lesson is about the test itself: fixed queries with known relevant documents, and the numbers that score a ranking on them.",
            "few": "Trying 651 settings works on any number of claims. What goes wrong when the claims you tune on are the claims you measure on? See [BM25: the classic ranking](/course/open/ir-bm25?lang=en).",
            "none": "The test claims have relevant abstracts: the lesson measured with them. What goes wrong when you tune on them too? See [BM25: the classic ranking](/course/open/ir-bm25?lang=en).",
            "slow": "Both sets of claims search the same 5,183 abstracts. What goes wrong when the claims you tune on are the claims you measure on? See [BM25: the classic ranking](/course/open/ir-bm25?lang=en).",
        },
    )
    quick_tune
    return (quick_tune,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Ranking by cosine: the vector space model](/course/open/ir-vector-space?lang=en) and [BM25: the classic ranking](/course/open/ir-bm25?lang=en), one number decided which ranking did better: the share of the questions of XQuAD whose own paragraph comes first. BM25 gets 80.0% there, the cosine 66.6%. That number looks only at the first place: a ranking that puts the own paragraph second gets as little for it as a ranking that puts it last. And it works only because each question has one paragraph it was written from. A claim of SciFact can have five relevant abstracts, and a search engine shows ten results at a time.

    Search has been measured in a fixed way since the late 1950s. Christopher Manning, Prabhakar Raghavan and Hinrich Schütze write in *Introduction to Information Retrieval* (2008): "To measure ad hoc IR effectiveness in the standard way, we need a test collection consisting of three things": documents, information needs written as queries, and relevance judgments, one for each pair of a query and a document. With those three, a ranking gets a number, and two rankings can be compared on the same test. This lesson adds three numbers that look at more than the first place.

    **An everyday example.** A teacher grades every student with the same exam and the same answer key. The exam is the queries, the answer key holds the relevance judgments, and the way the points are counted decides who looks best.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_p10 = L.guess(
        "guess",
        question="BM25 puts the own paragraph among its first ten paragraphs for 96.3% of the 1,190 questions. Its precision at 10 is the share of relevant paragraphs among those first ten, averaged over the questions. What is it?",
        options={
            "same": "about 0.96, as high as the share",
            "eight": "about 0.8, the share with the own paragraph first",
            "half": "about 0.5",
            "right": "just under 0.1",
        },
        reveal="Just under 0.1: 0.096. Each question has one relevant paragraph, so at most one of the first ten paragraphs is relevant, and even a perfect ranking gets a precision at 10 of 0.1. The 96.3% is another number: the share of the questions with the own paragraph among the first ten, the recall at 10. What a measure can reach depends on how many relevant documents each query has.",
    )
    guess_p10
    return (guess_p10,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **The Cranfield paradigm.** The book: "Rigorous formal testing of IR systems was first completed in the Cranfield experiments, beginning in the late 1950s." The Cranfield collection "contains 1,398 abstracts of aerodynamics journal articles, a set of 225 queries, and exhaustive relevance judgments of all (query, document) pairs". Measuring this way, every system on the same fixed test collection, is called the Cranfield paradigm. TREC, the evaluation that the American institute NIST has run since 1992, works the same way, with far larger collections.

    **Relevance judgments.** A judgment says whether a document is relevant to the need behind a query: in this lesson relevant or not relevant, nothing in between. The book: "Relevance is assessed relative to an information need, not a query." In [Search: needs and relevance](/course/open/ir-what-is-search?lang=en), two of the three needs had the same query, *python*, and each need had its own relevant documents. Thirteen documents can be judged for every need, as in Cranfield, but "For large modern collections, it is usual for relevance to be assessed only for a subset of the documents for each query": TREC judges the documents that the systems under test put among their first results. The two test collections of this module are judged in their own way. In XQuAD, the one relevant paragraph of a question is the paragraph it was written from; in SciFact, the relevant abstracts of a claim are the abstracts that the sentence it was written from cites. Every other document counts as not relevant, also when it would help.

    **Precision at k.** Cut the ranking after k documents, and take the precision of what is left: the number of relevant documents among the first k, divided by k. The share this module has used so far, the questions with the own paragraph first, is the precision at 1. The precision at 10 says how much of a page of ten results helps. The book names what it is good for, and what not: "It has the advantage of not requiring any estimate of the size of the set of relevant documents; the disadvantages that it is the least stable of the commonly used evaluation measures and that it does not average well because the total number of relevant documents for a query has a strong influence on precision at k."

    **Reciprocal rank.** Take 1 divided by the place of the first relevant document: 1 for the first place, 0.5 for the second, 0.333 for the third, and 0 when no relevant document comes. Its mean over the queries is the mean reciprocal rank, MRR. Ellen Voorhees and Dawn Tice describe how the question answering track of TREC-8, in 1999, scored five answers to each question: "An individual question received a score equal to the reciprocal of the rank at which the first correct response was returned, or 0 if none of the five responses contained a correct answer." MRR rewards a second place, but looks at nothing after the first relevant document. Voorhees and Tice: "Question answering systems are given no credit for retrieving multiple (different) correct answers."

    **Average precision.** Go down the ranking, and at the place of each relevant document, take the precision up to that place. Add these precisions up, and divide by the number of relevant documents of the query. A relevant document that the ranking does not hold adds nothing, but still counts in that number. The book: "When a relevant document is not retrieved at all, the precision value in the above equation is taken to be 0." Its mean over the queries is the mean average precision, MAP. The book: "Among evaluation measures, MAP has been shown to have especially good discrimination and stability." With one relevant document, the average precision is the precision at its place, 1 divided by that place: the reciprocal rank.

    For a query with the set $R$ of relevant documents:

    $$\text{P@}k = \frac{\text{relevant documents among the first } k}{k} \qquad \text{RR} = \frac{1}{\text{place of the first relevant document}} \qquad \text{AP} = \frac{1}{|R|} \sum_{d \in R} \text{P@place}(d)$$

    where a relevant document $d$ that the ranking does not hold adds 0 to AP. MRR and MAP are the means of RR and AP over the queries.

    **The three needs.** The picture shows the three needs of the course's thirteen documents, each with the results of BM25 for its query, and the precision at each place. For need 1, a pet python to buy, document 3 comes fourth, and document 1, *Ball pythons for sale*, is not a result at all: it says *pythons*, not *python*. The reciprocal rank is 0.25, and the average precision 0.125: the precision at place 4, 0.25, divided by the two relevant documents. For need 2, learning Python, document 0 comes first: reciprocal rank and average precision 1. For the need about red wine, document 6 comes first and document 9 fifth: reciprocal rank 1, average precision (1 + 0.4) / 2 = 0.7. Over the three needs, the MRR is 0.75 and the MAP 0.608.
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
    | **test collection** | documents, queries and relevance judgments: the fixed test that rankings are measured on |
    | **Cranfield paradigm** | measuring search systems on a fixed test collection, as the Cranfield experiments did from the late 1950s |
    | **relevance judgment** | the decision whether a document is relevant to the need behind a query, made by people or by how the collection was built; here relevant or not relevant |
    | **precision at k**, P@k | the share of relevant documents among the first k of a ranking |
    | **reciprocal rank**, RR | 1 divided by the place of the first relevant document, or 0 when none comes |
    | **mean reciprocal rank**, MRR | the mean of the reciprocal rank over the queries |
    | **average precision**, AP | the precision at the place of each relevant document, added up and divided by the number of relevant documents |
    | **mean average precision**, MAP | the mean of the average precision over the queries |
    | $R$, $\lvert R \rvert$ | the set of relevant documents of a query, and how many there are |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To measure a ranking on a test collection:

    1. Rank the documents for each query, and keep the first results, such as 100.
    2. Look up the relevant documents of each query in the relevance judgments.
    3. Precision at k: count the relevant documents among the first k, and divide by k.
    4. Reciprocal rank: 1 divided by the place of the first relevant document, or 0 when none comes.
    5. Average precision: at each relevant document, the precision up to its place; add these up, and divide by the number of relevant documents.
    6. Take the mean of each over all queries: the mean P@k, the MRR and the MAP.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `C.rankings(L.lang, "bm25")` gives, for each question of XQuAD, the first 100 paragraphs of BM25, best first, and `"cosine"` those of the cosine, as in the lessons before; paragraphs with the same score are in the order of their number. `C.relevant(L.lang)` gives, for each question, a list with its one relevant paragraph: the own paragraph. `C.precision_at_k`, `C.reciprocal_rank` and `C.average_precision` are the course's measures, and `C.question_place(C.EXAMPLE, L.lang)` gives the place of the example question among the questions. First the question *Prior to the arrival of the French, the area now known as Jacksonville was previously inhabited by what people?*, then the means over all the questions:
    """)
    return


@app.cell
def example_cell():
    _k = C.question_place(C.EXAMPLE, L.lang)
    _relevant = C.relevant(L.lang)[_k]
    for _how in ("cosine", "bm25"):
        _ranking = C.rankings(L.lang, _how)[_k]
        print(_how, _ranking[:5], _relevant, C.precision_at_k(_ranking, _relevant, 1),
              C.precision_at_k(_ranking, _relevant, 5), round(C.reciprocal_rank(_ranking, _relevant), 3),
              round(C.average_precision(_ranking, _relevant), 3))
    for _how in ("cosine", "bm25"):
        _rankings = C.rankings(L.lang, _how)
        _relevants = C.relevant(L.lang)
        _n = len(_rankings)
        _p1 = sum([C.precision_at_k(_rankings[_q], _relevants[_q], 1) for _q in range(_n)]) / _n
        _p10 = sum([C.precision_at_k(_rankings[_q], _relevants[_q], 10) for _q in range(_n)]) / _n
        _mrr = sum([C.reciprocal_rank(_rankings[_q], _relevants[_q]) for _q in range(_n)]) / _n
        _map = sum([C.average_precision(_rankings[_q], _relevants[_q]) for _q in range(_n)]) / _n
        print(_how, _n, round(_p1, 3), round(_p10, 3), round(_mrr, 3), round(_map, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The first two lines are the question about Jacksonville: the first five paragraphs, the relevant paragraph, the precision at 1 and at 5, the reciprocal rank and the average precision. Both rankings put paragraph 1382 first and the own paragraph, 1380, second: the precision at 1 is 0, at 5 it is 0.2, and the reciprocal rank and the average precision are both 0.5. Paragraph 1382 says that in the 16th century "the region was inhabited by the Mocama, a coastal subgroup of the Timucua people": it answers the question too, but the test collection judges only the own paragraph relevant.

    The last two lines are the means over all 1,190 questions: the precision at 1 and at 10, the MRR and the MAP. BM25 gets 0.8 at 1, the 80.0% of [BM25: the classic ranking](/course/open/ir-bm25?lang=en), 0.096 at 10, and an MRR of 0.862, against 0.764 for the cosine. The MAP is the same as the MRR for both rankings: each question has one relevant paragraph.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_precision_at_k():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. precision_at_k: the first k results

        Write `precision_at_k`, which gets `ranking`, the documents of a ranking as a list, best first, `relevant`, a list of the relevant documents of the query, and `k`. It gives back the precision at k: the number of relevant documents among the first `k` of the ranking, divided by `k`. In the start code, `found` starts at 0, and the loop adds 1 to it for each relevant `doc`. The `return` line already divides by `k`, also when the ranking holds fewer than `k` documents: a place without a document counts as not relevant. Replace the `...` with the documents that the loop goes through: the first `k` of `ranking`.

        The check calls `precision_at_k([7, 3, 1, 9, 4], [7, 4], 2)`, which gives `0.5`, and `precision_at_k([5, 2], [2, 8], 4)`, which gives `0.25`: the ranking holds two documents, and the division is by 4. Then cases of its own: k = 1, no relevant documents, an empty ranking, and the course's rankings.
        """),
        L.task("Let the loop go through the first k documents of the ranking."),
    ])
    return


@app.cell(hide_code=True)
def exercise_precision_at_k():
    box_precision_at_k = L.code_box("precision_at_k", lines=6)
    box_precision_at_k
    return (box_precision_at_k,)


@app.cell(hide_code=True)
def check_precision_at_k(box_precision_at_k):
    result_precision_at_k = L.check("precision_at_k", box_precision_at_k.value)
    result_precision_at_k
    return (result_precision_at_k,)


@app.cell(hide_code=True)
def hints_precision_at_k():
    L.hints(
        "precision_at_k",
        [
            "The loop should go through the first `k` documents of `ranking`; `doc` is then each of those documents in turn.",
            "A slice gives the start of a list: `numbers[:3]` gives its first three numbers.",
            "`    for doc in ranking[___]:`",
        ],
        why="`for doc in ranking[:k]:`: the loop sees only the first k documents, and each relevant one adds 1 to `found`.",
    )
    return


@app.cell(hide_code=True)
def text_after_precision_at_k():
    mo.md(r"""
    In the picture, BM25 gets a precision at 1 of 0.800 and at 10 of 0.096, and a perfect ranking, with the own paragraph always first, gets 1 and 0.1. With one relevant paragraph per question, the precision at k cannot be higher than 1 divided by k: here the precision at 10 says more about the test collection than about the ranking. The share of the questions with the own paragraph among the first 10, 96.3% for BM25, is the recall at 10, here 10 times the precision at 10.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_reciprocal_rank():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. reciprocal_rank: the first relevant document

        Write `reciprocal_rank`, which gets `ranking` and `relevant` as in exercise 1. It gives back 1 divided by the place of the first relevant document, counted from 1, or 0 when the ranking holds no relevant document. In the start code, `for place in range(len(ranking))` goes through the places of the ranking as Python counts them, from 0. At the first relevant document, the function gives back its value at once; when the loop ends without one, the last line gives back 0. Replace the `...` with the reciprocal rank: 1 divided by the place, where the first place is 1, not 0.

        The check calls `reciprocal_rank([4, 8, 5, 2, 6], [6, 2])`, which gives `0.25`: document 2 is the first relevant document, at place 4. And `reciprocal_rank([5, 1], [9])`, which gives `0`. Then cases of its own: a relevant document at the first place, an empty ranking, and the course's rankings.
        """),
        L.task("Give back 1 divided by the place of the document, counted from 1."),
    ])
    return


@app.cell(hide_code=True)
def exercise_reciprocal_rank():
    box_reciprocal_rank = L.code_box("reciprocal_rank", lines=5)
    box_reciprocal_rank
    return (box_reciprocal_rank,)


@app.cell(hide_code=True)
def check_reciprocal_rank(box_reciprocal_rank):
    result_reciprocal_rank = L.check("reciprocal_rank", box_reciprocal_rank.value)
    result_reciprocal_rank
    return (result_reciprocal_rank,)


@app.cell(hide_code=True)
def hints_reciprocal_rank():
    L.hints(
        "reciprocal_rank",
        [
            "`place` counts from 0, so the place of the document counted from 1 is one more than `place`.",
            "Divide 1 by that number, and put the number in parentheses, so that Python adds before it divides.",
            "`            return 1 / ___`",
        ],
        why="`return 1 / (place + 1)`: the first place gives 1, the second 0.5, the third 0.333.",
    )
    return


@app.cell(hide_code=True)
def text_after_reciprocal_rank():
    mo.md(r"""
    In the picture, BM25 puts the own paragraph first for 952 questions, second for 97 and third for 34, and 8 own paragraphs are not among the first 100. Its MRR is 0.862, against a precision at 1 of 0.800: before the mean is taken, each question with the own paragraph second adds 0.5, where P@1 gives it 0, and each third place 0.333. For the cosine the gap is larger, 0.764 against 0.666, because it puts more own paragraphs second: 130.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_average_precision():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. average_precision: every relevant document

        Write `average_precision`, which gets `ranking` and `relevant` as before. It gives back the average precision: at the place of each relevant document, the precision of the ranking up to that place, added up and divided by the number of relevant documents. In the start code, `total` starts at 0, the loop `for place in range(len(ranking))` finds each relevant document, and the `return` line divides by `len(relevant)`, so a relevant document that the ranking misses adds 0. Replace the `...` with what each relevant document adds: `precision_at_k` of exercise 1, which gets the ranking, the relevant documents and k, in that order, with k set to the place of the document, counted from 1. The box already knows `precision_at_k`.

        The check calls `average_precision([2, 5, 1, 7], [2, 7])`, which gives `0.75`: the precision 1 at place 1 and 0.5 at place 4, divided by 2. And `average_precision([9, 4, 6], [4, 8])`, which gives `0.25`: the precision 0.5 at place 2, divided by 2, because document 8 is missing. Then cases of its own: every relevant document at the top, every relevant document missing, an empty ranking, and the course's rankings.
        """),
        L.task("Let each relevant document add the precision of the ranking up to its own place."),
    ])
    return


@app.cell(hide_code=True)
def exercise_average_precision():
    box_average_precision = L.code_box("average_precision", lines=6)
    box_average_precision
    return (box_average_precision,)


@app.cell(hide_code=True)
def check_average_precision(box_average_precision):
    result_average_precision = L.check("average_precision", box_average_precision.value)
    result_average_precision
    return (result_average_precision,)


@app.cell(hide_code=True)
def hints_average_precision():
    L.hints(
        "average_precision",
        [
            "Each relevant document adds `precision_at_k` of the ranking up to its place: the new `total` is the old `total` plus that precision.",
            "`precision_at_k` gets the ranking, the relevant documents, and for k the place counted from 1, which is one more than `place`, as in exercise 2.",
            "`            total = total + precision_at_k(___)`",
        ],
        why="`total = total + precision_at_k(ranking, relevant, place + 1)`: each relevant document adds the precision of the ranking up to its place, and the last line divides by all relevant documents, found or not.",
    )
    return


@app.cell(hide_code=True)
def text_after_average_precision():
    mo.md(r"""
    For the three needs, your function gives an average precision of 0.125, 1 and 0.7, and a MAP of 0.608, where the MRR is 0.75. On XQuAD, your MAP is the same as the MRR: 0.862 for BM25, because with one relevant paragraph the average precision is 1 divided by its place. MAP and MRR only differ where queries have more than one relevant document: of the 300 test claims of SciFact, 23 have two to five.

    ### Try it: MRR and MAP on SciFact

    SciFact, from [BM25: the classic ranking](/course/open/ir-bm25?lang=en), is a test collection of the same kind: 5,183 abstracts, and for each claim the abstracts that its sentence cites as its relevant abstracts. Of the 300 test claims, 277 have one relevant abstract, and 23 have two to five. Download SciFact, pick one of those 23 claims, and move k. The picture shows the first k abstracts of the cosine and of BM25 for the claim, the places of all its relevant abstracts, and its precision at k, reciprocal rank (RR) and average precision (AP). Below are the means over all 300 test claims, next to a perfect ranking.

    Start with claim 133: both rankings put a relevant abstract first. Which ranking finds the other four sooner, and which measures show it, and which do not? Then pick claim 971, and compare its reciprocal rank with its average precision.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download SciFact, 8.2 MB")
    return (get_button,)


@app.cell(hide_code=True)
def scifact_cell(get_button):
    sci_claims = C.scifact_or_stop(get_button, {
        "ask": "*SciFact is not on your computer yet. The button downloads it once from the copy of BEIR on Hugging Face, 8.2 MB: the claims under the license CC BY 4.0 and the abstracts under ODC-By 1.0. The rest of the lesson works without it.*",
        "busy": "**Downloading** {done} of {total} MB …",
        "no_net": "*The download did not work: Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: a file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: your disk has no room for 8.2 MB. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
    })
    _options = C.try_options()
    try_claim = mo.ui.dropdown(options=_options, value=next(t for t, c in _options.items() if c == C.CLAIM), label="Claim")
    try_k = mo.ui.slider(C.TRY_K[0], C.TRY_K[1], step=1, value=C.TRY_K[2], label="k", show_value=True)
    mo.hstack([try_claim, try_k], justify="start", gap=2)
    return sci_claims, try_claim, try_k


@app.cell(hide_code=True)
def try_look(sci_claims, try_claim, try_k):
    mo.Html(C.try_picture(L.lang, try_claim.value, try_k.value))
    return


@app.cell(hide_code=True)
def text_after_try():
    mo.md(r"""
    For claim 133, both rankings get a reciprocal rank of 1. But the cosine finds the other relevant abstracts at places 3, 7, 10 and 75, and BM25 at 10, 25 and 31, and the fifth not among the first 100: an average precision of 0.512 against 0.290. For claim 971 it is the other way around: the cosine has its first relevant abstract at place 2 and BM25 at place 3, but BM25 has all four among its first 7. The reciprocal rank is 0.5 against 0.333, the average precision 0.427 against 0.501. Over all 300 test claims, BM25 does better on each measure: an MRR of 0.632 against 0.550, and a MAP of 0.620 against 0.536. With k = 10, even the perfect ranking gets a mean precision at 10 of only 0.113, because most claims have one relevant abstract.

    Two limits remain. The judgments miss documents that would help: for 34 of the 238 questions where BM25 puts another paragraph first, that paragraph holds the text of an answer too, as paragraph 1382 does for the question about Jacksonville. And people judge differently: Voorhees and Tice found that "assessors have legitimate differences of opinion as to correctness even for fact-based, short-answer questions." The book reports what such differences did to the order of the systems at TREC, in a study by Voorhees (2000): "although the numbers change, the rankings are quite stable."

    ### Exercise 4. Which measure fits?

    Sort each statement below: it fits precision at k, MRR or MAP.
    """)
    return


@app.cell(hide_code=True)
def exercise_measures():
    sort_measures = L.sort("measures")
    sort_measures
    return (sort_measures,)


@app.cell(hide_code=True)
def check_measures(sort_measures):
    result_measures = L.check("measures", sort_measures)
    result_measures
    return (result_measures,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The rankings measured here are those of [Ranking by cosine: the vector space model](/course/open/ir-vector-space?lang=en) and [BM25: the classic ranking](/course/open/ir-bm25?lang=en), and the precision of [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en) now counts in a ranked list. In the domain *Building with AI*, [Is the RAG right?](/course/open/build-rag-eval?lang=en) measures how well the search of a RAG system finds the right texts. The next lesson, *Graded relevance: nDCG*, lets a document be more or less relevant instead of only relevant or not; and *From ten links to one answer* looks at what changes when a search engine writes one answer instead of a ranked list.

    **An everyday example.** A friend gives you a list of restaurants. The precision at 3 asks how many of the first three tips were good; the reciprocal rank asks how far down the list the first good one was; the average precision asks whether all the good restaurants came early, the last one too.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_precision_at_k, result_reciprocal_rank, result_average_precision, result_measures):
    L.wrap_up(result_precision_at_k, result_reciprocal_rank, result_average_precision, result_measures)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
