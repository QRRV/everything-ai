# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ir-bm25. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ir-bm25", "en")
    C = L.checks   # the course's thirteen documents, the paragraphs and questions of SQuAD, SciFact and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [Ranking by cosine: the vector space model](/course/open/ir-vector-space?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_twice():
    quick_twice = L.choice(
        "quick-twice",
        question="In *Ranking by cosine: the vector space model*, document 6 was written out twice in a row. What happened to its cosine with the query *red wine heart*?",
        options={
            "double": "it doubled, like the dot product",
            "right": "it stayed the same: 0.476",
            "half": "it halved",
            "one": "it rose to 1, because the document then holds every word of the query twice",
        },
        correct="right",
        why={
            "right": "Every weight doubles, so the dot product doubles, and so does the length of the document vector: the cosine stays 0.476. This lesson asks a different question: what should the second and the third *wine* add?",
            "double": "The dot product doubled, from 1.482 to 2.965. What happened to the length of the document vector, and so to the division? See [Ranking by cosine: the vector space model](/course/open/ir-vector-space?lang=en).",
            "half": "A longer document has a longer vector, but its dot product with the query grows too. How do the two changes compare? See [Ranking by cosine: the vector space model](/course/open/ir-vector-space?lang=en).",
            "one": "The cosine is 1 only when the two vectors point the same way. Did writing the document out twice change the way its vector points? See [Ranking by cosine: the vector space model](/course/open/ir-vector-space?lang=en).",
        },
    )
    quick_twice
    return (quick_twice,)


@app.cell(hide_code=True)
def quick_check_slope():
    quick_slope = L.choice(
        "quick-slope",
        question="In the same lesson, pivoted length normalization divided the dot product by a mix of the length and a pivot. With slope 0, every document is divided by the same number. Which order does that give the documents?",
        options={
            "cosine": "the order of the cosine",
            "right": "the order of the dot product",
            "short": "the shortest document first",
            "number": "the order of their number, as with AND in an inverted index",
        },
        correct="right",
        why={
            "right": "Dividing every score by the same number keeps their order: slope 0 gives the order of the dot product, which lets long documents win. BM25 has a setting for how much the length counts too, and at 0 the length does not count at all.",
            "cosine": "The cosine divides by the length of the document vector. Is that the same number for every document? See [Ranking by cosine: the vector space model](/course/open/ir-vector-space?lang=en).",
            "short": "Dividing every dot product by the same number keeps the larger one larger. What decides the order then? See [Ranking by cosine: the vector space model](/course/open/ir-vector-space?lang=en).",
            "number": "A ranking orders the documents by a score. Which score is left when every document is divided by the same number? See [Ranking by cosine: the vector space model](/course/open/ir-vector-space?lang=en).",
        },
    )
    quick_slope
    return (quick_slope,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Ranking by cosine: the vector space model](/course/open/ir-vector-space?lang=en), the cosine put the own paragraph first for 66.6% of the 1,190 questions of XQuAD. Take the question *Where did the discharge from glaciers go in Europe in the last Ice Age?* The cosine puts first a paragraph about the Rhine during the last Ice Age: it says *ice* four times, and *last* and *age* twice each, but not where the water of the melting glaciers went. The own paragraph, with the answer, *the Rhine*, holds *discharge* once, *glaciers* once, *Europe* three times and *ice* once, and comes second. In a TF-IDF vector, a word that occurs four times weighs four times as much as a word that occurs once. But a paragraph that says *ice* four times is not four times as much about ice, and a paragraph that holds *discharge* and *glaciers* at all is already a good sign.

    BM25 ranks documents with the same two ingredients, the counts of the words of the query and their IDF, and changes two things: each repeat of a word adds less than the one before, and a setting of its own decides how much the length of a document counts. Christopher Manning, Prabhakar Raghavan and Hinrich Schütze write in *Introduction to Information Retrieval* (2008) that BM25 weighting is "often called Okapi weighting, after the system in which it was first implemented", and of such weightings: "Especially in the TREC evaluations, they performed well and were widely adopted by many groups." It is still the ranking that newer ways to search are measured against: BEIR, a set of test collections (queries, each with the documents known to be relevant to it) by Nandan Thakur and colleagues (2021), reports: "Our results show BM25 is a robust baseline".

    **An everyday example.** The first time a friend says a restaurant is good, you learn a lot; the fifth time, hardly anything new. BM25 lets the fourth *ice* count for less than the first.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_bm25 = L.guess(
        "guess",
        question="The cosine put the own paragraph first for 66.6% of the 1,190 questions. BM25 ranks the same 2,067 paragraphs with the same IDF, but lets each repeat of a word add less than the one before, with the settings that the book calls reasonable. For what share of the questions does it put the own paragraph first?",
        options={
            "fewer": "less often than the cosine: about six in ten",
            "same": "about as often as the cosine: two in three",
            "right": "more than seven in ten",
            "most": "nearly always: more than nine in ten",
        },
        reveal="More than seven in ten: 80.0%, against 66.6% for the cosine. When each word adds its IDF only once, however often it occurs, BM25 already gets 75.5%, more than the cosine. The gain is largest for the questions whose own paragraph is in the longest group: from 53.3% with the cosine to 72.3%.",
    )
    guess_bm25
    return (guess_bm25,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **The same ingredients.** BM25 scores a document by the words of the query that it holds. Each of them adds its IDF, log10(N / df) as in [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en), times a weight that grows with its count in the document. A word that is not in the document adds nothing, so, as with the cosine, only the postings lists of the words of the query matter.

    **Saturation.** With TF-IDF, the weight of a word grows as fast as its count: ten times *ice* weighs ten times as much as once. BM25 lets the count saturate:

    $$\frac{(k_1 + 1) \cdot \text{tf}}{k_1 + \text{tf}}$$

    For a count of 1 this is exactly 1. Each next repeat adds less than the one before, and however often the word occurs, the weight stays below $k_1 + 1$: with $k_1 = 1.2$, four times *ice* gives 1.69 instead of 4. Stephen Robertson and Hugo Zaragoza, in *The Probabilistic Relevance Framework: BM25 and Beyond* (2009), give it its name: "We refer to this behaviour as saturation." The lower $k_1$, the stronger the saturation. The book: "A k1 value of 0 corresponds to a binary model (no term frequency)": a word then adds its IDF once, however often it occurs. A large $k_1$ lets nearly every repeat count in full again.

    **Length.** Robertson and Zaragoza give two reasons why one document is longer than another: "Some authors are simply more verbose than others, using more words to say the same thing." And: "Some authors have more to say: they may write a single document containing or covering more ground." For the first kind, the counts should be scaled down by the length; for the second, not. BM25 mixes the two with a second setting, $b$, from 0 to 1, and multiplies $k_1$ by

    $$(1 - b) + b \cdot \frac{L_d}{L_{\text{ave}}}$$

    where $L_d$ is the number of words of the document and $L_{\text{ave}}$ the average over the collection. A document of average length gets 1, and keeps the saturation above. A longer document gets more than 1, so a word needs more repeats to reach the same weight, and a shorter one gets less than 1. The book: "b = 1 corresponds to fully scaling the term weight by the document length", and "b = 0 corresponds to no length normalization." Unlike the cosine, BM25 does not divide the score by a length: the length only sets how fast the counts saturate, and a word still adds less than $k_1 + 1$ times its IDF.

    **The whole score.** Together, the BM25 score of a document $d$ for a query $q$ is

    $$\text{BM25}(q, d) = \sum_{t \in q} \text{idf}_t \cdot \frac{(k_1 + 1) \cdot \text{tf}_{t,d}}{k_1 \cdot \left((1 - b) + b \cdot \frac{L_d}{L_{\text{ave}}}\right) + \text{tf}_{t,d}}$$

    where $\text{tf}_{t,d}$ is the count of word $t$ in document $d$, and a word that is not in $d$ adds nothing. This is the formula of the book. The book adds a third setting, $k_3$, for words that occur more than once in a long query; the questions here are short, and this lesson counts each word of the query once.

    **The settings.** The book: "The tuning parameters of these formulas should ideally be set to optimize performance on a development test collection." And: "In the absence of such optimization, experiments have shown reasonable values are to set k1 and k3 to a value between 1.2 and 2 and b = 0.75." This lesson uses $k_1 = 1.2$ and $b = 0.75$. On the paragraphs, each of the two settings helps: with $k_1 = 0$ the share of questions whose own paragraph comes first drops from 80.0% to 75.5%, and with $b = 0$ to 76.9%. With $k_1 = 5$, where the repeats count almost in full again, it drops to 73.4%. The gain over the cosine is largest for the questions whose own paragraph is in the longest group: from 53.3% to 72.3%, and from 71.1% to 82.6% for the other questions.

    For the question about the glaciers, BM25 puts the own paragraph first: 11.02 against 10.59 for the paragraph about the Ice Age. There, the four times *ice* add 3.98, where their TF-IDF weight was 9.10. The picture shows the question in both rankings, what each of its words adds in the two paragraphs, and all the questions ranked in six ways.
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
    | **BM25** | a ranking that adds up, for each word of the query that a document holds, its IDF times a weight that saturates with the count of the word and is scaled by the length of the document |
    | **saturation** | the way the weight of a word grows less with each repeat, and stays below $k_1 + 1$ |
    | **term weight** | what one word adds in one document before its IDF: its count, saturated and scaled by the length of the document |
    | $k_1$ | the setting for how strongly the counts saturate: at 0 each word counts once, and the higher, the more the repeats count |
    | $b$ | the setting for how much the length of a document counts against the average: at 0 not at all, at 1 fully |
    | $L_d$, $L_{\text{ave}}$ | the number of words of document $d$, and the average number of words of the documents of the collection |
    | **tune** | to pick settings such as $k_1$ and $b$ by trying them on queries with known relevant documents, and then to measure the result on other queries |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To rank documents for a query with BM25:

    1. Count the words of each document, and take the average number of words over the collection.
    2. Give each word its IDF, log10(N / df).
    3. For each word of the query, once, that a document holds: saturate its count with $k_1$, with $k_1$ scaled by the length of the document against the average with $b$, and multiply by its IDF.
    4. Add these parts up: the BM25 score of the document.
    5. Sort the documents by their score, the highest first.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `C.SMALL_IDF` holds the IDF of each word of the course's thirteen documents, `C.SMALL_AVERAGE` their average number of words, and `C.DOCS` the documents as lists of words. `C.saturation` gives the saturation of a count, `C.term_weight` the weight of a word with a count in a document of a length, and `C.bm25` the BM25 score of a document; here always with $k_1 = 1.2$ and $b = 0.75$. The query is *red wine heart*:
    """)
    return


@app.cell
def example_cell():
    _idf = C.SMALL_IDF[L.lang]
    _average = C.SMALL_AVERAGE[L.lang]
    _query = C.QUERIES[L.lang]["wine"]
    print(round(_average, 3), [round(C.saturation(_count, 1.2), 3) for _count in (1, 2, 3, 4)])
    for _doc in (6, 7, 8):
        _words = C.DOCS[L.lang][_doc]
        _counts = [_words.count(_word) for _word in _query]
        _parts = [_idf[_word] * C.term_weight(_words.count(_word), len(_words), _average, 1.2, 0.75)
                  for _word in _query]
        _score = C.bm25(_query, _words, _idf, _average, 1.2, 0.75)
        print(_doc, len(_words), _counts, [round(_part, 3) for _part in _parts], round(_score, 3))
    _twice = C.DOCS[L.lang][6] * 2
    print(round(C.bm25(_query, _twice, _idf, _average, 1.2, 0.75), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The documents have 16 words on average, and the first line shows the saturation of the counts 1 to 4: a word three times gives 1.571, not 3. Each next line gives a document, its number of words, the counts of *red*, *wine* and *heart*, what each of them adds, and the BM25 score. Document 6 holds *wine* three times, and with 23 words it is longer than average: *wine* adds 0.735 there, against 0.525 in document 7, where it occurs once. The TF-IDF score of *TF-IDF: weighing the words* gave document 6 more than twice the score of document 7, 2.974 against 1.439; BM25 gives 1.714 against 1.476. Documents 7 and 8 hold each word once, and document 8 is longer, 20 words against 15, so each word adds less there. The last line is document 6 written out twice in a row: its score grows from 1.714 to only 1.788, where the TF-IDF score would double.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_saturation():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. saturation: each repeat adds less

        Write `saturation`, which gets `tf`, the count of a word in a document, and `k1`, the setting. It gives back the saturated count of *The idea*: the count times $k_1 + 1$, divided by $k_1$ plus the count. With only `+`, `*` and `/`, the function also works when `tf` is a NumPy array of counts. Replace the `...` with the formula.

        The check calls `saturation(1, 1.2)`, which gives `1.0`, and `saturation(3, 2.0)`, which gives `1.8`. Then cases of its own: other counts and settings, a setting of 0, a count of 0, and a NumPy array of counts.
        """),
        L.task("Give the saturated count: the count times k1 + 1, divided by k1 plus the count."),
    ])
    return


@app.cell(hide_code=True)
def exercise_saturation():
    box_saturation = L.code_box("saturation", lines=3)
    box_saturation
    return (box_saturation,)


@app.cell(hide_code=True)
def check_saturation(box_saturation):
    result_saturation = L.check("saturation", box_saturation.value)
    result_saturation
    return (result_saturation,)


@app.cell(hide_code=True)
def hints_saturation():
    L.hints(
        "saturation",
        [
            "The formula has a part above the line, `tf * (k1 + 1)`, and a part below it: the count `tf` plus the setting `k1`.",
            "Divide by the two together: put them in parentheses, so that Python adds them before it divides.",
            "`    return tf * (k1 + 1) / (___)`",
        ],
        why="`return tf * (k1 + 1) / (tf + k1)`: a count of 1 gives exactly 1, and however large the count, the result stays below k1 + 1.",
    )
    return


@app.cell(hide_code=True)
def text_after_saturation():
    mo.md(r"""
    In the picture, with $k_1 = 1.2$, a word that occurs 20 times adds 2.075, against 1 for once: the count itself would say 20. With $k_1 = 0.5$, the weight hardly grows after a few repeats, and with $k_1 = 2.0$ it keeps growing for longer.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_term_weight():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. term_weight: the length of the document

        Write `term_weight`, which gets `tf`, the count of a word in a document, `length`, the number of words of the document, `average`, the average number of words of the documents, and the two settings `k1` and `b`. It gives back the weight of the word: the saturation of exercise 1, with $k_1$ multiplied by the scale of *The idea*, in which 1 counts with the share `1 - b` and the length against the average with the share `b`. The start code names that number `scale`, and its `return` line already holds the saturation with `k1 * scale`. Replace the `...` with the scale. With only `+`, `-`, `*` and `/`, the function also works when `tf` and `length` are NumPy arrays: the picture below calls it once, for every word in every paragraph.

        The check calls `term_weight(2, 20, 20.0, 2.0, 0.75)`, a document of average length, which gives `1.5`, and `term_weight(2, 60, 20.0, 2.0, 0.5)`, a document three times as long, which gives `1.0`. Then cases of its own: b = 0, b = 1, k1 = 0, a short document, and NumPy arrays.
        """),
        L.task("Scale k1 by a mix of 1 and the length against the average, in which the length has the share b."),
    ])
    return


@app.cell(hide_code=True)
def exercise_term_weight():
    box_term_weight = L.code_box("term_weight", lines=4)
    box_term_weight
    return (box_term_weight,)


@app.cell(hide_code=True)
def check_term_weight(box_term_weight):
    result_term_weight = L.check("term_weight", box_term_weight.value)
    result_term_weight
    return (result_term_weight,)


@app.cell(hide_code=True)
def hints_term_weight():
    L.hints(
        "term_weight",
        [
            "`scale` is a mix: 1 counts with the share `1 - b`, and the length of the document against the average, `length / average`, with the share `b`.",
            "The second part of the mix is the length divided by the average, multiplied by its share, the setting b.",
            "`    scale = 1 - b + ___`",
        ],
        why="`scale = 1 - b + b * length / average`: a document of average length gets 1, whatever b is; with b = 0 the length does not count, with b = 1 fully.",
    )
    return


@app.cell(hide_code=True)
def text_after_term_weight():
    mo.md(r"""
    In the examples of the check, the document of average length gets the scale 1 and the saturation of exercise 1: 1.5; the document three times as long gets the scale 2, and the word, twice in it, adds only 1.0. In the picture, a word that occurs once adds 1.271 in a paragraph of 60 words, 1.0 in a paragraph of average length, and 0.711 in a paragraph of 250 words. With the course's term weights, BM25 puts the own paragraph first for 80.0% of the questions; the picture shows what yours give.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_bm25():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. bm25: the score of a document

        Write `bm25`, which gets `query`, the words of a query as a list that holds each word once, `doc`, a document as a list of words, `idf`, a dictionary with the IDF of each word, `average`, the average number of words of the documents, and the settings `k1` and `b`. It gives back the BM25 score of the document. In the start code, `score` starts at 0, and the loop goes through each `word` of `query`; when `doc` holds the word, the word adds its part to `score`. Replace the `...` with the new `score`: the score so far, plus the IDF of the word times its term weight. The function `term_weight` of exercise 2 gives that weight for the count of the word in the document, the number of words of the document, the average and the two settings, in that order; the box already knows it. `doc.count(word)` gives how often `word` occurs in `doc`, and `len(doc)` the number of words of `doc`.

        The check calls `bm25(["red", "wine"], ["red", "wine", "wine", "cheese"], {"red": 0.5, "wine": 1.0, "cheese": 2.0}, 4.0, 1.2, 0.75)`, a document of average length, which gives `1.875`, and `bm25(["python", "snake"], ["python", "tank"], {"python": 1.0, "snake": 2.0, "tank": 1.5}, 2.0, 1.2, 0.75)`, which gives `1.0`: *snake* is not in the document. Then cases of its own: an empty query, k1 = 0, b = 0, b = 1, a word with IDF 0, and the query *red wine heart* with documents 6 and 9 of the course, and *rode wijn hart* with document 6 of the Dutch lesson.
        """),
        L.task("Add up, for each word of the query in the document, its IDF times its term weight."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bm25():
    box_bm25 = L.code_box("bm25", lines=6)
    box_bm25
    return (box_bm25,)


@app.cell(hide_code=True)
def check_bm25(box_bm25):
    result_bm25 = L.check("bm25", box_bm25.value)
    result_bm25
    return (result_bm25,)


@app.cell(hide_code=True)
def hints_bm25():
    L.hints(
        "bm25",
        [
            "Each round adds the part of `word` to `score`: its IDF, `idf[word]`, times what `term_weight` gives for this word in `doc`.",
            "`term_weight` gets five values: the count of the word in the document, the number of words of the document, then `average`, `k1` and `b`.",
            "`            score = score + idf[word] * ___`",
        ],
        why="`score = score + idf[word] * term_weight(doc.count(word), len(doc), average, k1, b)`: each word of the query that the document holds adds its IDF times its term weight.",
    )
    return


@app.cell(hide_code=True)
def text_after_bm25():
    mo.md(r"""
    For the question about the glaciers, your function puts the own paragraph first, where the cosine put it second. For the question about the Parliament of Victoria, it puts a longer paragraph about Victoria first, of 176 words: it holds *Victoria* four times, and also *what*, a word of the question that says nothing about the answer. The own paragraph comes second, where the cosine put it first: no ranking gets every question right.

    ### Try it: tuning k1 and b on SciFact

    The book says the settings "should ideally be set to optimize performance on a development test collection": queries whose relevant documents are known. SciFact, by David Wadden and colleagues (2020), holds such queries: "expert-written scientific claims paired with evidence-containing abstracts". Experts wrote each claim from a sentence of a paper that cites other papers. In the copy that BEIR uses, the collection is 5,183 abstracts of biomedical papers, the queries are the claims, and the abstracts that the sentence cites are the relevant abstracts of the claim. The claims come in two sets: 809 training claims to tune on, and 300 test claims to measure on. The abstracts have 223.2 words on average, from 38 to 1,575.

    Tune on the training claims only. A setting picked by looking at the test claims looks better on those claims than it will on new ones. Robertson and Zaragoza warn that tuned settings carry over badly too: "the optimised parameters will be specific to the collection evaluated and may not work well for other collections."

    Download SciFact, then move the sliders. First find a setting that puts the relevant abstract first for more training claims than $k_1 = 1.2$ and $b = 0.75$ do; then look at what your setting does on the test claims. The picture also shows one test claim, and the first three abstracts with your setting.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download SciFact, 8.2 MB")
    return (get_button,)


@app.cell(hide_code=True)
def scifact_cell(get_button):
    sci_numbers = C.scifact_or_stop(get_button, {
        "ask": "*SciFact is not on your computer yet. The button downloads it once from the copy of BEIR on Hugging Face, 8.2 MB: the claims under the license CC BY 4.0 and the abstracts under ODC-By 1.0. The rest of the lesson works without it.*",
        "busy": "**Downloading** {done} of {total} MB …",
        "no_net": "*The download did not work: Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: a file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: your disk has no room for 8.2 MB. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
    })
    try_k1 = mo.ui.slider(0.0, 3.0, step=0.1, value=C.TRY_START[0], label="k1", show_value=True)
    try_b = mo.ui.slider(0.0, 1.0, step=0.05, value=C.TRY_START[1], label="b", show_value=True)
    mo.hstack([try_k1, try_b], justify="start", gap=2)
    return sci_numbers, try_b, try_k1


@app.cell(hide_code=True)
def try_look(sci_numbers, try_k1, try_b):
    mo.Html(C.try_picture(L.lang, try_k1.value, try_b.value))
    return


@app.cell(hide_code=True)
def text_after_try():
    mo.md(r"""
    Tuned on the training claims, the best setting is $k_1 = 1.1$ and $b = 0.65$: 57.4% of the training claims, against 56.5% for the usual setting, and 53.7% of the test claims, against 53.3%. Tuning gains little here, because the usual setting was already close. The best setting on the test claims themselves, $k_1 = 0.7$ and $b = 1.0$, gives 56.0% there: picked by looking at the test, it promises more than tuning on the training claims delivers. Every setting of the sliders beats the cosine, which puts the relevant abstract first for 44.7% of the test claims.

    ### Exercise 4. What does k1 do, and what does b do?

    Sort each statement below: it is about the setting $k_1$, or about the setting $b$.
    """)
    return


@app.cell(hide_code=True)
def exercise_settings():
    sort_settings = L.sort("settings")
    sort_settings
    return (sort_settings,)


@app.cell(hide_code=True)
def check_settings(sort_settings):
    result_settings = L.check("settings", sort_settings)
    result_settings
    return (result_settings,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** BM25 uses the IDF of [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en) and the postings lists of [The inverted index](/course/open/ir-inverted-index?lang=en), and where [Ranking by cosine: the vector space model](/course/open/ir-vector-space?lang=en) divided by the length of a vector, BM25 lets the length set how fast the counts saturate. In the domain *Building with AI*, [Search by words: BM25](/course/open/build-bm25?lang=en) uses BM25 to find the texts that an assistant answers from. The next lesson, *Search engines for one domain*, ranks documents with fields, such as a title and a price; and *Did the search work? P@k, MAP and MRR* measures a ranking on SciFact in more ways than the first place alone.

    **An everyday example.** A cook tastes a soup and adds salt: the first pinch changes a lot, the fifth hardly anything, and a big pot needs more salt than a small bowl for the same taste. BM25 treats the repeats of a word like the pinches, and the length of a document like the size of the pot.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_saturation, result_term_weight, result_bm25, result_settings):
    L.wrap_up(result_saturation, result_term_weight, result_bm25, result_settings)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
