# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ir-vector-space. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ir-vector-space", "en")
    C = L.checks   # the course's thirteen documents, the paragraphs and questions of SQuAD, their vectors and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    A question about [The inverted index](/course/open/ir-inverted-index?lang=en), and one about [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_order():
    quick_order = L.choice(
        "quick-order",
        question="In *The inverted index*, `search_and` gave the paragraphs that hold every word of a query. In which order did it give them?",
        options={
            "best": "the paragraph that fits the query best first",
            "right": "in order of their number, from small to large",
            "short": "the shortest paragraph first",
            "most": "the paragraph in which the words of the query occur most often first, then the others",
        },
        correct="right",
        why={
            "right": "The walks keep the numbers in the order of the postings lists: from small to large. That order says nothing about how well a paragraph fits the query. This lesson ranks the paragraphs, the best first.",
            "best": "To know which paragraph fits best, the search needs a score for each paragraph. Does a postings list hold one? See [The inverted index](/course/open/ir-inverted-index?lang=en).",
            "short": "A postings list holds the numbers of the paragraphs, nothing about their length. In which order are those numbers? See [The inverted index](/course/open/ir-inverted-index?lang=en).",
            "most": "The index of that lesson keeps which paragraphs hold a word, not how often. In which order does a walk keep the numbers? See [The inverted index](/course/open/ir-inverted-index?lang=en).",
        },
    )
    quick_order
    return (quick_order,)


@app.cell(hide_code=True)
def quick_check_weight():
    quick_weight = L.choice(
        "quick-weight",
        question="In *TF-IDF: weighing the words*, what weight does a word get in a document when it occurs in every document of the collection?",
        options={
            "count": "its count in the document",
            "right": "0, however often it occurs there",
            "half": "half its count: the IDF is 0.5",
            "most": "the highest weight of the document, because such a word occurs most often",
        },
        correct="right",
        why={
            "right": "The IDF is log10(N / df), and for a word in every document N / df is 1, and the log of 1 is 0. A weight of 0 adds nothing to a score, also in this lesson.",
            "count": "The count is the term frequency. What is it multiplied by, and what is that number for a word in every document? See [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en).",
            "half": "The IDF is log10(N / df). What is N / df for a word in every document, and what is its log? See [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en).",
            "most": "In that lesson, *the* occurs 158 times in *A Mad Tea-Party*. What weight did it get there? See [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en).",
        },
    )
    quick_weight
    return (quick_weight,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    XQuAD, a selection from SQuAD by Mikel Artetxe, Sebastian Ruder and Dani Yogatama, holds 1,190 questions about 240 of the 2,067 paragraphs of the lesson [The inverted index](/course/open/ir-inverted-index?lang=en), each written by a person about one paragraph: its own paragraph. Other paragraphs may help with a question too, but this lesson counts only the own one. Put each question to the inverted index, with AND: it finds anything at all for 21 of the 1,190 questions, and the own paragraph for 20. A question nearly always holds a word that its paragraph does not: the question about the Parliament of Victoria below holds *what*, *document* and *formed*, and its own paragraph holds none of the three. And even when AND finds paragraphs, it gives them in order of their number, not of how well they fit.

    [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en) scored a document by adding up the weights of the search words in it. That score ranks every paragraph for every question, and the own paragraph comes first for 44.2% of the questions. But look at where the first places go. Put the paragraphs in five groups of about the same size by length: the longest group, from 157 to 634 words, holds the own paragraph of 25.2% of the questions, and gets 52.9% of the first places. For the question *What document formed the Parliament of Victoria?*, the first paragraph has 512 words about the law of the European Union: it says *parliament* eleven times and *Victoria* not once. The own paragraph, 90 words with the answer, the Victoria Constitution Act 1855, comes fifth. A long paragraph holds more different words and repeats them more often, so it collects more weight, whatever it is about.

    This lesson treats a question and a paragraph as two vectors, and compares the directions in which they point, not how much weight they hold. That is the vector space model, one of the classic ways to rank documents in information retrieval.

    **An everyday example.** A big pot of soup and a small bowl of the same soup taste the same: the amount differs, the mix does not. The cosine compares the mix of words, not the amount.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_cosine = L.guess(
        "guess",
        question="By the TF-IDF score, the own paragraph comes first for 44.2% of the 1,190 questions. Now turn each question into a vector of TF-IDF weights too, and rank the paragraphs by the cosine of the two vectors, from *Length, distance and cosine similarity*. For what share of the questions does the own paragraph come first then?",
        options={
            "same": "about as often as before, because dividing by a length changes no order",
            "fewer": "less often: about one in three",
            "right": "about two in three",
            "most": "about nine in ten",
        },
        reveal="About two in three: 66.6%. With the cosine, the longest group gets 22.5% of the first places, and the first paragraphs have 126.9 words on average, against 189.7 by the TF-IDF score; the own paragraphs have 129.9. Part of the gain comes from the question: weighing its words by their IDF as well already lifts the share to 54.9%, before any division.",
    )
    guess_cosine
    return (guess_cosine,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **A document as a vector of weights.** In [A document as a vector](/course/open/nlp-document-vectors?lang=en), a document became a vector of counts, one number for each word. Weigh each count by its IDF, and each of the 2,067 paragraphs becomes a vector of 23,298 numbers, one for each different word of the paragraphs, almost all 0. Christopher Manning, Prabhakar Raghavan and Hinrich Schütze write in *Introduction to Information Retrieval* (2008): "The representation of a set of documents as vectors in a common vector space is known as the vector space model". They name the SMART system of Gerard Salton, described in 1971, which "was perhaps the first to view a document as a vector of weights."

    **The query is a vector too.** In the book's words, "we can also view a query as a vector". The question *What document formed the Parliament of Victoria?* gets a weight for each of its words, its count times its IDF among the paragraphs: *document* 2.54, *victoria* 1.78, *formed* 1.68, *parliament* 1.54, *what* 1.36, *of* 0.02, and *the*, which is in 2,046 of the 2,067 paragraphs, 0.004. A word of the query that no paragraph holds has no IDF, and is left out.

    **The dot product.** Multiply the weights of each word that the query vector and a paragraph vector both hold, and add the products up: their dot product, from [Many features: the dot product](/course/open/math-dot-product?lang=en). The TF-IDF score of *TF-IDF: weighing the words* is a dot product too, with a query vector that holds the count of each word of the query and no IDF. Weighing the words of the query by their IDF as well lets a rare word of the question count for more: the own paragraph then comes first for 54.9% of the questions. A word that is not in the query adds nothing to the dot product, so only the postings lists of the words of the query matter: an index that keeps a weight beside each posting gives every dot product without opening a paragraph.

    **The cosine.** The dot product still grows with the length of a paragraph: write a paragraph out twice, and every weight doubles, and so does its dot product with any query. The book, on comparing two documents: "To compensate for the effect of document length", the standard way is to compute their cosine similarity, from [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en). It divides the dot product by the lengths of both vectors:

    $$\cos(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\lVert \mathbf{q} \rVert \, \lVert \mathbf{d} \rVert}$$

    In the book's words, "the denominator is the product of their Euclidean lengths": the norm of each vector, the square root of its squares added up. Dividing by $\lVert \mathbf{d} \rVert$ takes away the edge of a long paragraph. The length of the query vector is the same for every paragraph, so it changes no order; dividing by both lengths keeps the cosine at most 1, and because no weight is below 0, it is never below 0 either. This is **length normalization**: a score divided by the length of the document vector.

    **Pivoted length normalization.** Amit Singhal, Chris Buckley and Mandar Mitra (1996) put documents into groups by length, and compared, for each group, how often its documents were relevant with how often the cosine retrieved them. Long documents came up less often than they were relevant. The book puts their finding this way: "cosine normalization has a tendency to distort the computed relevance vis-à-vis the true relevance, at the expense of longer documents." Their fix divides the dot product by a mix of the length of the document and a fixed number, the **pivot**:

    $$\text{slope} \cdot \lVert \mathbf{d} \rVert + (1 - \text{slope}) \cdot \text{pivot}$$

    with a **slope** below 1. The book writes piv for the pivot, "where piv is the cosine normalization value at which the two curves intersect": the length at which the curve of how often documents are relevant crosses the curve of how often the cosine retrieves them. A document with a vector longer than the pivot gets a divisor smaller than its own length, so a higher score than when its dot product is divided by its own length, as in the cosine; a shorter one gets a larger divisor. Slope 1 gives the order of the cosine, and slope 0 divides every document by the same number, which leaves the order of the dot product.

    The paragraphs of this lesson show the pattern of Singhal and colleagues: with the cosine, the longest group gets 22.5% of the first places and holds the own paragraph of 25.2% of the questions, and when the own paragraph is in the longest group, it comes first for 53.3% of the questions, against 71.1% for the other questions. Singhal and colleagues set the pivot to the average of the old divisors, and so does this lesson: with the average length of the paragraph vectors, 23.2, as the pivot and slope 0.8, the longest group gets 27.7% of the first places, and its questions 58.0%; the other questions drop to 68.3%, and all questions together to 65.7%. On these paragraphs, the pivot moves first places from short paragraphs to long ones, but does not put the own paragraph first more often. The book warns: "pivoted document length normalization is not appropriate for all applications".

    The picture shows the question about the Parliament of Victoria in the three rankings, and where the first places of all questions go.
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
    | **vector space model** | documents and queries as vectors of weights, with one dimension for each word, ranked by how close the vector of a document is to that of the query |
    | **query vector** | the query as a vector of weights, in the same space as the documents: the count of each word of the query times its IDF |
    | **length normalization** | a score divided by the length of the document vector, so that a long document does not win by its length alone; the cosine does it |
    | **pivoted length normalization** | dividing by a mix of the length of the document vector and a fixed pivot, so that long documents lose less than with the cosine |
    | **pivot**, **slope** | the fixed number of pivoted length normalization, and the share of the length in the mix: slope 1 gives the order of the cosine |
    | $\mathbf{q} \cdot \mathbf{d}$ | the dot product of the query vector $\mathbf{q}$ and the document vector $\mathbf{d}$ |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To rank documents for a query with the vector space model:

    1. Turn each document into a vector: each word's count times its IDF.
    2. Turn the query into a vector the same way, with the IDF of the documents; leave out a word that no document holds.
    3. Compute the dot product of the query vector with each document vector. Only the words of the query count, so the postings lists of those words are enough.
    4. Divide each dot product by the length of the query vector and the length of the document vector: the cosine.
    5. Sort the documents by their cosine, the highest first.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `C.SMALL_IDF` holds the IDF of each word of the course's thirteen documents, and `C.SMALL_VECTORS` their TF-IDF vectors, each a dictionary from words to weights. `C.vector` turns a list of words into a vector with a given IDF, `C.dot_product` gives the dot product of two vectors, `C.norm` the length of a vector, and `C.cosine` the cosine of two vectors. The query is *red wine heart*:
    """)
    return


@app.cell
def example_cell():
    _idf = C.SMALL_IDF[L.lang]
    _query = C.vector(C.QUERIES[L.lang]["wine"], _idf)
    print({_word: round(_weight, 3) for _word, _weight in _query.items()})
    for _doc in (6, 7, 8):
        _vector = C.SMALL_VECTORS[L.lang][_doc]
        _dot = C.dot_product(_query, _vector)
        print(_doc, round(_dot, 3), round(C.norm(_vector), 3), round(C.cosine(_query, _vector), 3))
    _twice = C.vector(C.DOCS[L.lang][6] * 2, _idf)
    print(round(C.dot_product(_query, _twice), 3), round(C.norm(_twice), 3), round(C.cosine(_query, _twice), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    *Red* is in five of the thirteen documents and gets 0.415; *wine* and *heart* are in four and get 0.512 each. Each line gives a document, its dot product with the query vector, the length of its vector and the cosine. Documents 7 and 8 hold *red*, *wine* and *heart* once each, so their dot products are the same, 0.696. Document 8 holds more words besides, so its vector is longer, 3.439 against 3.199, and its cosine lower: 0.243 against 0.261. The last line is document 6 written out twice: the dot product doubles, from 1.482 to 2.965, the length doubles, from 3.735 to 7.47, and the cosine stays 0.476.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_norm():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. norm: the length of a vector of weights

        Write `norm`, which gets `weights`, a vector kept as a dictionary from words to weights, such as `C.SMALL_VECTORS["en"][6]`, the vector of document 6. It gives back the length of the vector: the square root of the squares of the weights added up, the norm $\lVert \mathbf{x} \rVert$ of *Length, distance and cosine similarity*. A word that is not in the dictionary has weight 0 and adds nothing, so only the words that are there count. In the start code, `total` starts at 0, the loop goes through each `word` of `weights`, and `** 0.5` takes the square root at the end. Replace the `...` with the new `total`.

        The check calls `norm({"red": 3.0, "wine": 4.0})`, which gives `5.0`, and `norm({"python": 1.0, "snake": 2.0, "tank": 2.0})`, which gives `3.0`. Then cases of its own: an empty vector, a weight of 0, four weights of 1, a single weight, and the vectors of document 6 in English and document 10 in Dutch.
        """),
        L.task("Give the length of a vector kept as a dictionary: the square root of the squares of its weights added up."),
    ])
    return


@app.cell(hide_code=True)
def exercise_norm():
    box_norm = L.code_box("norm", lines=6)
    box_norm
    return (box_norm,)


@app.cell(hide_code=True)
def check_norm(box_norm):
    result_norm = L.check("norm", box_norm.value)
    result_norm
    return (result_norm,)


@app.cell(hide_code=True)
def hints_norm():
    L.hints(
        "norm",
        [
            "Each round adds one weight to `total`. `weights[word]` is the weight of `word`, and the norm adds up the squares of the weights, not the weights themselves.",
            "The new total is the old `total` plus the weight of this word squared; in Python, `** 2` squares a number.",
            "`        total = total + ___`",
        ],
        why="`total = total + weights[word] ** 2`: each word adds the square of its weight, and the square root of the total is the length.",
    )
    return


@app.cell(hide_code=True)
def text_after_norm():
    mo.md(r"""
    In the picture, the vector of a paragraph in the longest group is about twice as long as one in the shortest: 32.37 against 16.42 on average. A long paragraph holds more different words and repeats them, and each adds to the length. The cosine divides by that length.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_cosine():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. cosine: the dot product divided by both lengths

        Write `cosine`, which gets two vectors kept as dictionaries, `q`, the query vector, and `d`, the vector of a document, and gives back their cosine. In the start code, the `if` gives `0.0` when one of the two vectors has length 0, such as a query whose words are in no document: the cosine is then not defined, and the function gives 0.0 instead. The loop goes through each `word` of `q`, and when `d` holds the word too, it adds the product of the two weights, `q[word] * d[word]`, to `dot`: the dot product. Replace the `...` with the cosine. The box already knows the function `norm` of exercise 1.

        The check calls `cosine({"red": 1.0, "wine": 1.0}, {"red": 2.0, "wine": 2.0})`, two vectors that point the same way, which gives `1.0`, and `cosine({"wine": 3.0, "heart": 4.0}, {"wine": 1.0})`, which gives `0.6`. Then cases of its own: vectors without a word in common, an empty query, a query whose only weight is 0, a document with a word that is not in the query, two vectors that are the same, and the query *red wine heart* with documents 6 and 8, and *rode wijn hart* with document 7 of the Dutch lesson.
        """),
        L.task("Give the cosine of two vectors: their dot product divided by the length of each."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cosine():
    box_cosine = L.code_box("cosine", lines=9)
    box_cosine
    return (box_cosine,)


@app.cell(hide_code=True)
def check_cosine(box_cosine):
    result_cosine = L.check("cosine", box_cosine.value)
    result_cosine
    return (result_cosine,)


@app.cell(hide_code=True)
def hints_cosine():
    L.hints(
        "cosine",
        [
            "`dot` holds the dot product, and `norm` gives the length of a vector. The cosine divides `dot` by the length of `q` and the length of `d`, multiplied together.",
            "Divide the dot product by one number: the length of the query vector times the length of the document vector, in parentheses, so that Python divides by both at once.",
            "`    return dot / (___)`",
        ],
        why="`return dot / (norm(q) * norm(d))`: the dot product divided by the product of the two lengths.",
    )
    return


@app.cell(hide_code=True)
def text_after_cosine():
    mo.md(r"""
    For the question about the Parliament of Victoria, the paragraph about the law of the European Union has the highest dot product, 26.2, against 16.7 for the own paragraph. But its vector is 51.0 long, and that of the own paragraph 18.6: 26.2 divided by 51.0 is smaller than 16.7 divided by 18.6, and the cosine puts the own paragraph first. For the question about the Energiprojekt AB engine, the TF-IDF score puts a paragraph of 204 words first, and the cosine the own paragraph, of 92 words.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pivoted():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. pivoted: dividing by a mix of the length and the pivot

        Write `pivoted`, which gets `dot`, the dot product of a query vector and a document vector, `length`, the length of the document vector, `pivot`, the fixed number from *The idea*, and `slope`, a number from 0 to 1. It gives back the dot product divided by the divisor of Singhal and colleagues, in which the length counts with the share `slope` and the pivot with the share `1 - slope`. The start code names that number `divisor`, and divides `dot` by it. Replace the `...` with the divisor. With only `+`, `-`, `*` and `/`, the function also works when `dot` and `length` are NumPy arrays: the picture below calls it with all the questions and all 2,067 paragraphs at once.

        The check calls `pivoted(5.6, 30.0, 20.0, 0.8)`, a document vector longer than the pivot, which gives `0.2`, and `pivoted(3.0, 10.0, 20.0, 0.8)`, a shorter one, which gives `0.25`. Then cases of its own: slope 1, slope 0, a length equal to the pivot, and NumPy arrays with one row and with two rows.
        """),
        L.task("Divide the dot product by a mix of the length and the pivot, in which the length has the share slope."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pivoted():
    box_pivoted = L.code_box("pivoted", lines=4)
    box_pivoted
    return (box_pivoted,)


@app.cell(hide_code=True)
def check_pivoted(box_pivoted):
    result_pivoted = L.check("pivoted", box_pivoted.value)
    result_pivoted
    return (result_pivoted,)


@app.cell(hide_code=True)
def hints_pivoted():
    L.hints(
        "pivoted",
        [
            "`divisor` is a mix of two numbers: `pivot` counts with the share `(1 - slope)`, and the length of the document with the share `slope`.",
            "The second part of the mix is the length of the document vector, multiplied by its share, the slope.",
            "`    divisor = (1 - slope) * pivot + ___`",
        ],
        why="`divisor = (1 - slope) * pivot + slope * length`: with slope 1 the divisor is the length, as in the cosine; with slope 0 it is the pivot for every document.",
    )
    return


@app.cell(hide_code=True)
def text_after_pivoted():
    mo.md(r"""
    In the examples of the check, the document of length 30 gets the divisor 28: a score of 0.2, where dividing by its length would give 0.187. The document of length 10 gets the divisor 12: 0.25, where its length would give 0.3. In the picture, slope 0.8 gives the longest group of paragraphs 27.7% of the first places: more than the 25.2% of the questions whose own paragraph it holds, where the cosine gave it 22.5%, and it finds the own paragraph first for fewer questions than the cosine.

    ### Try it: the slope

    Move the slider. At 1, the slope gives the cosine; at 0, the order of the dot product. The picture ranks the 2,067 paragraphs for all the questions with that slope, and shows how often the own paragraph comes first, and where the first places go. Find the slope at which the share of the first places of the longest group comes closest to its share of the own paragraphs. Then slide to 0, and watch the long paragraphs take the first places again.
    """)
    return


@app.cell(hide_code=True)
def try_slide():
    try_slope = mo.ui.slider(0.0, 1.0, step=0.05, value=C.TRY_START, label="Slope", show_value=True)
    try_slope
    return (try_slope,)


@app.cell(hide_code=True)
def try_look(try_slope):
    mo.Html(C.try_picture(L.lang, try_slope.value))
    return


@app.cell(hide_code=True)
def text_exercise_changes():
    mo.md(r"""
    ### Exercise 4. What changes the cosine?

    Take the query *red wine heart* and document 6 of the course, *Red wine, white wine and the heart: does the color of the wine change the risk of a heart attack? What studies found.* Sort each change below: the cosine of the query vector and the vector of document 6 stays the same, or it changes. The IDF of each word stays as it is.
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

    **How this connects.** The weights are those of [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en), and the cosine that of [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en), now with vectors kept as dictionaries, because almost every weight is 0. Only the words of the query count in a dot product, so the index of [The inverted index](/course/open/ir-inverted-index?lang=en), with a weight beside each posting, gives every score from the postings lists of the query. The next lesson, *BM25: the classic ranking*, uses an IDF as well, and changes two things: each repeat of a word adds less than the one before, and a setting of its own decides how much the length of a document counts.

    **An everyday example.** A photo printed as a stamp and the same photo printed as a poster show the same picture: only the size differs. The cosine sees the picture, not the size.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_norm, result_cosine, result_pivoted, result_changes):
    L.wrap_up(result_norm, result_cosine, result_pivoted, result_changes)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
