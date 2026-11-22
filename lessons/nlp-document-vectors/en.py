# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-document-vectors. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-document-vectors", "en")
    C = L.checks   # the course's words, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    A question about [Spam or not: naive Bayes](/course/open/lm-naive-bayes?lang=en), and one about [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_bag():
    quick_bag = L.choice(
        "quick-bag",
        question="Naive Bayes looks at a text as a bag of words: which words it has, and how often. Which change to a text leaves its bag of words as it was?",
        options={
            "add": "Adding one more word",
            "right": "Putting its words in another order",
            "twice": "Writing one of its words a second time",
        },
        correct="right",
        why={
            "right": "A bag of words keeps each word with its count, and forgets where it stood: *dog bites man* and *man bites dog* give the same bag. This lesson turns such a bag into a list of numbers with a fixed length, and so inherits that blind spot.",
            "add": "One more word adds a word to the bag, or adds 1 to its count. See [Spam or not: naive Bayes](/course/open/lm-naive-bayes?lang=en).",
            "twice": "The count of that word goes up by 1, and the bag keeps counts. See [Spam or not: naive Bayes](/course/open/lm-naive-bayes?lang=en).",
        },
    )
    quick_bag
    return (quick_bag,)


@app.cell(hide_code=True)
def quick_check_cosine():
    quick_cosine = L.choice(
        "quick-cosine",
        question="What is the cosine similarity of `[1, 2]` and `[2, 4]`?",
        options={
            "zero": "0, since the vectors differ",
            "half": "0.5, since one is half the other",
            "right": "1, as they point the same way",
            "two": "2, the factor between them",
        },
        correct="right",
        why={
            "right": "`[2, 4]` is `[1, 2]` times 2: twice as long, in the same direction, and the cosine similarity looks only at the direction. The dot product is 10, the lengths are √5 and √20, and 10 / (√5 × √20) = 10 / 10 = 1. This lesson uses it to compare texts of different lengths.",
            "zero": "A cosine similarity of 0 means that the vectors are at right angles, such as `[1, 0]` and `[0, 1]`. These two point the same way. See [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en).",
            "half": "The cosine similarity divides the dot product by both lengths, so a factor between the vectors drops out. See [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en).",
            "two": "The cosine similarity lies between −1 and 1, whatever the lengths. See [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en).",
        },
    )
    quick_cosine
    return (quick_cosine,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A text can have three words or three thousand, but most methods that compare or sort things want each thing as a list of numbers of one fixed length. This lesson gives a text such a list: one count for each word of a fixed list of words. Jurafsky and Martin describe it in *Speech and Language Processing*: "In the vector space model of information retrieval (Salton, 1971) a document is represented as a vector of counts of the words it contains." Information retrieval is the field of search, and the vector space model is its name for this way of turning documents into vectors; the idea comes from there. The textbook *Introduction to Information Retrieval* by Christopher Manning, Prabhakar Raghavan and Hinrich Schütze says that the SMART system of Gerard Salton and colleagues at Cornell "was perhaps the first to view a document as a vector of weights." A paper by Salton, A. Wong and C. S. Yang from 1975 carries the name in its title: *A vector space model for automatic indexing*. With such lists, two texts can be compared with the cosine similarity, and a new text can get the label of the text it is most like.

    The texts of this lesson are 3,000 short sentences from reviews, collected by Dimitrios Kotzias and colleagues for a paper in 2015 and published by the UCI Machine Learning Repository as *Sentiment Labelled Sentences*: 1,000 from Amazon, about products such as phones and headsets, 1,000 from IMDb, about films, and 1,000 from Yelp, about restaurants. Each site gives 500 positive and 500 negative sentences; the makers tried to choose sentences with a clearly positive or negative tone. The lesson never shows the 85 sentences with swear words, insults, slurs, or crude or sexual remarks, but counts them like the rest.

    **An everyday example.** The order form of a bakery lists every bread the bakery makes, each with a box for how many. Every customer fills in the same form, so any two orders can be compared box by box, and most boxes stay empty.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="The lesson keeps 2,400 of the sentences for training. Together they use 4,518 different words. Write each sentence as a list of 4,518 counts, one for each word. What share of all the numbers in those 2,400 lists is 0?",
        options={
            "half": "About half",
            "ninety": "About 90%",
            "right": "About 99.8%",
            "more": "About 99.99%",
        },
        reveal="**About 99.8%**: of the 10,843,200 numbers, only 26,188 are not 0, so 99.76% are 0. A training sentence has 10.9 different words on average and 56 at most, out of 4,518. Two sentences have no word at all: both are *10/10*, and the words of this lesson are runs of letters. 2,625 of the 4,518 words occur only once in all 2,400 sentences.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### A vocabulary and a vector

    The **vocabulary** is a fixed list of words, each with its own number, its column. In this lesson it is every word of the 2,400 training sentences, in small letters and in alphabetical order: *a* is column 0, *good* column 1,707, *the* column 3,948. A **document vector** holds one count for each word of the vocabulary: how often that word occurs in the document. Here a document is one sentence. Jurafsky and Martin call this a bag of words: "we are representing a text document as if it were a bag of words, that is, an unordered set of words with their position ignored, keeping only their frequency in the document." Manning and colleagues give an example: "Thus, the document Mary is quicker than John is, in this view, identical to the document John is quicker than Mary."

    ### The document-term matrix

    Put the vectors of all documents under each other, and you get a table with one row per document and one column per word: the **document-term matrix**. The documentation of scikit-learn, a Python library for machine learning, puts it like this: "A corpus of documents can thus be represented by a matrix with one row per document and one column per token (e.g. word) occurring in the corpus." Both textbooks draw it the other way round, with a row per word and a column per document, and call it a *term-document matrix*. The picture shows the matrix of four sentences, over only the twelve words they use.
    """)
    return


@app.cell(hide_code=True)
def idea_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Mostly zeros: sparse

    A sentence uses a handful of the 4,518 words, so almost every number in its vector is 0. The scikit-learn documentation says that such a matrix has "many feature values that are zeros (typically more than 99% of them)". A vector or matrix like that is **sparse**. Storing all those zeros wastes memory, so a **sparse matrix** keeps only the numbers that are not 0, each with its place. The documentation of SciPy, the library whose sparse matrices scikit-learn gives back, says that "only the locations where data exists are recorded". In plain Python, a dictionary from column to count does the same: *The mic is great.* becomes `{1731: 1, 2086: 1, 2475: 1, 3948: 1}`, four numbers instead of 4,518. This lesson calls such a dictionary a **sparse row**.

    ### Comparing with the cosine similarity

    Two sentences are alike when their vectors point in about the same direction. Jurafsky and Martin: "By far the most common similarity metric is the cosine of the angle between the vectors." The dot product alone will not do: "This raw dot product, however, has a problem as a similarity metric: it favors long vectors." Manning and colleagues make the same point for documents: "Two documents with very similar content can have a significant vector difference simply because one is much longer than the other." The cosine similarity divides the length out. Counts are never negative, and in the words of Jurafsky and Martin, "the cosine for these vectors ranges from 0–1": 1 when two sentences use the same words in the same proportions, 0 when they share no word.

    With sparse rows, the dot product only needs the columns that both rows hold: in every other column one of the two counts is 0, and so is the product.

    ### A new sentence gets the label of the most similar one

    This lesson labels a new sentence in the simplest way the vectors allow: find the training sentence with the highest cosine similarity, and give the new sentence its label. To try that out, the lesson holds out every fifth sentence, 600 in all: they play no part in the vocabulary or in the training rows. A word of a new sentence that is not in the vocabulary has no column, so it is left out. scikit-learn does the same: "words that were not seen in the training corpus will be completely ignored".
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **vocabulary** | a fixed list of words, each with its own column: here every word of the training sentences, in alphabetical order |
        | **document vector** | a list with one count for each word of the vocabulary: how often the word occurs in the document |
        | **document-term matrix** | the document vectors under each other: one row per document, one column per word |
        | **sparse** | mostly zeros |
        | **sparse matrix** | a matrix stored as only its numbers that are not 0, each with its place |
        | **sparse row** | in this lesson: a dictionary from column to count, with only the counts that are not 0 |
        """),
        L.watch_out("A high cosine similarity means that two sentences use the same words in about the same proportions, not that they mean the same. *the*, *was* and *on* count as much as *great* or *rude*, and *not* is just one more word."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Cut each training sentence into words, in small letters.
    2. Make the vocabulary: every word that occurs, in alphabetical order, with its column: 0, 1, 2, and so on.
    3. Turn each sentence into a vector: for each of its words, add 1 at the column of that word. Store only the counts that are not 0, as a sparse row.
    4. Turn a new sentence into a sparse row in the same way; a word outside the vocabulary is left out.
    5. Measure the cosine similarity of the new row with each training row, and give the new sentence the label of the training sentence with the highest one. When every cosine similarity is 0, give no answer.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The code below makes the vocabulary of three short documents, each already cut into words: every word goes into the list once, the list is sorted, and each word gets its place in the sorted list as its column. The last line shows how the course cuts a sentence into words, with its `words`.
    """)
    return


@app.cell
def example_cell():
    docs = [["the", "food", "was", "good"], ["the", "service", "was", "slow"], ["good", "food", "good", "service"]]
    vocabulary = []
    for doc in docs:
        for word in doc:
            if word not in vocabulary:
                vocabulary.append(word)
    vocabulary = sorted(vocabulary)
    example_columns = {}
    for i in range(len(vocabulary)):
        example_columns[vocabulary[i]] = i
    print(vocabulary)
    print(example_columns)
    print(C.words("The food was GOOD!"))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The vocabulary has six words, from *food* in column 0 to *was* in column 5. The third document, *good food good service*, becomes the vector `[1, 2, 1, 0, 0, 0]`: *food* once, *good* twice, *service* once, and no *slow*, *the* or *was*. As a sparse row, that is `{0: 1, 1: 2, 2: 1}`. `words` gives the words of a text in small letters and drops everything that is not a letter, such as *!*; an apostrophe inside a word stays, as in *don't*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_vectorize():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. vectorize: a sentence as a vector

        `small`, the course's `SMALL`, is the vocabulary of the four sentences in the picture, twelve words, and `columns` the vocabulary of the lesson, 4,518 words, from the course's `columns`; both are dictionaries from word to column. Write `vectorize(words, columns)`: give back a list with one number for each column of `columns`, 0 at the start, and for each word of `words` that `columns` holds, add 1 at the place `columns[word]`. A word that `columns` lacks is left out.

        For `vectorize(["the", "mic", "is", "great"], small)`, the result is `[0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]`: *great* is column 2, *is* 3, *mic* 4 and *the* 9. Replace the `...`. The check calls that and `vectorize(["crust", "is", "not", "good"], small)`, then cases of its own: a word that comes twice, a word outside the vocabulary, no words at all, only words outside it, a word three times in a longer sentence, a vocabulary of one word, and a sentence over the whole vocabulary of the lesson.
        """),
        L.task("Turn a list of words into a list of counts, one for each word of a vocabulary."),
    ])
    return


@app.cell
def columns_cell():
    small = C.SMALL
    columns = C.columns()
    print(small)
    print(len(columns), columns["good"], columns["the"])
    print(C.words("The mic is great."))
    return


@app.cell(hide_code=True)
def exercise_vectorize():
    box_vectorize = L.code_box("vectorize", lines=6)
    box_vectorize
    return (box_vectorize,)


@app.cell(hide_code=True)
def check_vectorize(box_vectorize):
    result_vectorize = L.check("vectorize", box_vectorize.value)
    result_vectorize
    return (result_vectorize,)


@app.cell(hide_code=True)
def hints_vectorize():
    L.hints(
        "vectorize",
        [
            "Start with `vector = [0 for column in columns]`: a 0 for each key of `columns`. Go over the words with `for word in words:`, ask with `if word in columns:` whether the word has a column, and add 1 to the number at that column. After the loop, give back `vector`.",
            "The column of *mic* is `small[\"mic\"]`, which is 4, so its count is the number at place 4 of `vector`. How do you write that place for any word?",
            r"`            vector[columns[word]] = ___ + 1`",
        ],
        why=r"`vector = [0 for column in columns]`; `for word in words:` and inside it `if word in columns:` with `vector[columns[word]] = vector[columns[word]] + 1`; after the loop, `return vector`.",
    )
    return


@app.cell(hide_code=True)
def text_after_vectorize():
    mo.md(r"""
    In the picture, your function turned the four sentences into lists of 4,518 numbers, and only two to four of each are not 0. The map shows your vectors of the first 60 training sentences from far away: a dot for each number that is not 0, on rows that are almost empty. Your function made 4,518 numbers for each sentence to say at most four things. A sparse row says the same with four. Here are the four sentences of the picture as sparse rows over `small`, the course's `SMALL_ROWS`, and *The mic is great.* as a sparse row over the vocabulary of the lesson, made with the course's `sparse_row`. `norm` gives the length of a sparse row: the square root of the sum of the squares of its counts.
    """)
    return


@app.cell
def rows_cell():
    small_rows = C.SMALL_ROWS
    norm = C.norm
    print(small_rows)
    print(C.sparse_row(C.words("The mic is great.")))
    print(norm(small_rows[0]), norm(small_rows[2]), norm({}))
    return


@app.cell(hide_code=True)
def text_exercise_cosine():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. sparse_cosine: how alike are two rows?

        Write `sparse_cosine(a, b)`, where `a` and `b` are sparse rows. For every column of `a` that `b` holds too, multiply the two counts, and add up those products: that is the dot product, since every other product is 0. When the sum is 0, give back `0.0`; otherwise divide it by `norm(a) * norm(b)`.

        For `sparse_cosine({2: 1, 3: 1, 4: 1, 9: 1}, {0: 1, 1: 1, 3: 1, 5: 1})`, the rows of *The mic is great.* and *Crust is not good.*, the result is 0.25: they share only column 3, *is*, which adds 1 × 1, and both lengths are 2. Replace the `...`. The check calls that and `sparse_cosine({7: 1, 10: 2}, {6: 1, 8: 1, 10: 1, 11: 1})`, then cases of its own: a row with itself, a row against the same row times three, rows with nothing in common, an empty row, two empty rows, counts above 1 in both rows, and two sentences over the vocabulary of the lesson.
        """),
        L.task("Measure the cosine similarity of two sparse rows, using only the columns they share."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cosine():
    box_cosine = L.code_box("sparse-cosine", lines=8)
    box_cosine
    return (box_cosine,)


@app.cell(hide_code=True)
def check_cosine(box_cosine):
    result_cosine = L.check("sparse-cosine", box_cosine.value)
    result_cosine
    return (result_cosine,)


@app.cell(hide_code=True)
def hints_cosine():
    L.hints(
        "sparse-cosine",
        [
            "Start with `dot = 0` and go over the columns with `for column in a:`. When `column in b`, add the product of the two counts to `dot`. After the loop, give back `0.0` when `dot` is 0, and otherwise `dot` divided by `norm(a) * norm(b)`.",
            "In column 10, *very*, the first row has 2 and the second has 1. How much does that column add to the dot product, and how do you write that for any column?",
            r"`            dot = dot + ___`",
        ],
        why=r"`dot = 0`; `for column in a:` and inside it `if column in b:` with `dot = dot + a[column] * b[column]`; after the loop, `if dot == 0:` with `return 0.0`, and then `return dot / (norm(a) * norm(b))`.",
    )
    return


@app.cell(hide_code=True)
def text_after_cosine():
    mo.md(r"""
    For the second example, *Very, very sad.* and *Service was very prompt.*, your function gives 0.447: they share only *very*, which adds 2 × 1, and the lengths are √5 and 2. In the picture, your function compared the held-out sentence *The delivery was on time.* with all 2,400 training sentences. The most similar one is *The bus boy on the other hand was so rude.*, at 0.516: it shares *the*, *was* and *on*, not one word about delivery or time. *The shipping time was also very quick!*, which is about the same thing, comes third, at 0.507. The most similar sentence is negative, so the new sentence would get the wrong label. Short, common words such as *the* and *was* are in so many sentences that they decide much of the cosine similarity.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_similar():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. most_similar: the nearest training sentence

        `rows` holds the sparse rows of the 2,400 training sentences, from the course's `training_rows`, and `labels` their labels, from `training_labels`, in the same order: 1 for positive, 0 for negative. Write `most_similar(row, rows)`: measure `cosine(row, rows[i])` for each place `i` of `rows`, and give back the place with the highest cosine similarity; the first such place when several are equal, and None when no cosine similarity is above 0. The check gives your function the course's `cosine`, which does the same as your `sparse_cosine`.

        For `most_similar({8: 1, 10: 1, 3: 1, 1: 1}, small_rows)`, the row of the words *service*, *very*, *is* and *good*, the result is 1: its cosine similarity with *Crust is not good.* is 0.5, the same as with *Service was very prompt.*, and *Crust is not good.* comes first. Replace the `...`. The check calls that and `most_similar({7: 1}, small_rows)`, the row of *sad*, then cases of its own: a row with nothing in common with any row, two rows that tie, the best row last, an empty list of rows, an empty row, and a held-out sentence against the 2,400 training rows.
        """),
        L.task("Find the place of the row with the highest cosine similarity, or None when no row shares a word."),
    ])
    return


@app.cell
def training_cell():
    cosine = C.cosine
    rows = C.training_rows()
    labels = C.training_labels()
    print(len(rows), labels[:10])
    print(rows[0])
    return


@app.cell(hide_code=True)
def text_after_training():
    mo.md(r"""
    The first training sentence, *So there is no way for me to plug it in here in the US unless I go by a converter.*, is negative. Its sparse row holds 20 columns, column 1,991 twice: *in*. Its vector would hold 4,498 zeros besides.
    """)
    return


@app.cell(hide_code=True)
def exercise_similar():
    box_similar = L.code_box("most-similar", lines=9)
    box_similar
    return (box_similar,)


@app.cell(hide_code=True)
def check_similar(box_similar):
    result_similar = L.check("most-similar", box_similar.value)
    result_similar
    return (result_similar,)


@app.cell(hide_code=True)
def hints_similar():
    L.hints(
        "most-similar",
        [
            "Start with `best = None` and with `best_value` at a value that only a row with a word in common can beat, and go over the places with `for i in range(len(rows)):`. Measure `value = cosine(row, rows[i])`; when it is higher than `best_value`, put `i` in `best` and `value` in `best_value`. After the loop, give back `best`.",
            "Start with `best_value = -1` in your head, and take a row that shares no word with `row`: its cosine similarity is 0. Does `value > best_value` let that row in? And should it?",
            r"`    best_value = ___`",
        ],
        why=r"`best = None` and `best_value = 0`; `for i in range(len(rows)):` with `value = cosine(row, rows[i])`, and `if value > best_value:` with `best = i` and `best_value = value`; after the loop, `return best`.",
    )
    return


@app.cell(hide_code=True)
def text_after_similar():
    mo.md(r"""
    `labels[most_similar(row, rows)]` is then the label for a new sentence. In the picture, your function found the most similar training sentence for twelve held-out sentences, and its label is right for eight of them. *The kids are annoying.* finds *The kids are very cool too.*, and *Setup went very smoothly.* finds *Very, very sad.*: the words they share are not the words that carry the tone.

    ### How good is it?

    On all 600 held-out sentences, the label of the most similar training sentence is right for 397: 66.2%. A coin would get about half right. Leave out the stop words, such as *the* and *was*, with the list from [Normalizing: case, accents and stop words](/course/open/nlp-normalize?lang=en), and 418 come out right: 69.7%. Naive Bayes, on the same split and the same counts, gets 497 right: 82.8%. It learns from all 2,400 sentences together how much each word says about the label, while the most similar sentence decides alone.

    Count vectors can go to any method that takes lists of numbers, but plain counts are a poor start. Manning and colleagues write: "Unweighted and unnormalized counts should not be used in vector space classification." The cosine similarity takes care of the length; the weight of each word is still missing. Jurafsky and Martin give counts from four plays by Shakespeare: *good* occurs 114 times in *As You Like It*, 80 in *Twelfth Night*, 62 in *Julius Caesar* and 89 in *Henry V*, more than *battle*, *fool* or *wit* in every play. They point out that the two comedies look more alike than the other two plays, with more *fool* and *wit* than *battle*. The course computed the cosine similarity from these four counts, and plain counts hide that difference: *As You Like It* gets 0.950 with the comedy *Twelfth Night*, but also 0.949 with *Henry V* and 0.945 with *Julius Caesar*, which are not comedies, because *good* decides almost everything. They go on: "In fact, in IR, we don’t use raw word counts like [1 114 36 20] for As You Like It". IR is short for information retrieval.
    """)
    return


@app.cell(hide_code=True)
def compare_picture():
    mo.Html(C.compare_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: your own sentence

    Type a sentence in English. The picture shows its sparse row over the vocabulary of the lesson, the words that have no column, the five most similar training sentences, and the label of the first. Try a sentence with *not*, one word, or a sentence in Dutch.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_text = mo.ui.text_area(value=C.DEMO_TEXT, label="Your sentence", full_width=True)
    demo_text
    return (demo_text,)


@app.cell(hide_code=True)
def demo_look(demo_text):
    mo.Html(C.demo_picture(demo_text.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    For the sentence in the box, the top two tie: *The food was very good.* and *The service was extremely slow.* both have a cosine similarity of 0.744 with it, one positive and one negative. The first one in the list wins, so the answer is *positive*, by the order of the sentences in the file. A sentence that mixes praise and complaint has no single most similar sentence that fits it.
    """)
    return


@app.cell(hide_code=True)
def text_explore():
    mo.md(r"""
    ### Try it: other vocabularies

    This lesson made its vocabulary from every word in small letters, and counted how often each word occurs. Change those choices, and see how many columns the vocabulary gets, how sparse the rows are, and how many of the 600 held-out sentences get the right label.
    """)
    return


@app.cell(hide_code=True)
def explore_controls():
    explore_lower = mo.ui.checkbox(value=True, label="Every word in small letters")
    explore_stop = mo.ui.checkbox(value=False, label="Leave out stop words")
    explore_binary = mo.ui.checkbox(value=False, label="0 or 1 instead of a count")
    explore_rare = mo.ui.checkbox(value=False, label="Leave out words seen only once")
    mo.vstack([explore_lower, explore_stop, explore_binary, explore_rare])
    return explore_binary, explore_lower, explore_rare, explore_stop


@app.cell(hide_code=True)
def explore_look(explore_binary, explore_lower, explore_rare, explore_stop):
    mo.Html(C.explore_picture(explore_lower.value, explore_stop.value, explore_binary.value, explore_rare.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_explore():
    mo.md(r"""
    Without small letters, *Great* and *great* get two columns: 5,149 columns, and 64.7% right. Leaving out the stop words helps most, 69.7%, with 4,364 columns. Leaving out the words seen only once cuts the vocabulary to 1,893 columns, and the score stays at 66.2%: a word that occurs in one training sentence can only match that one sentence. None of these choices, alone or together, moves the score by more than 21 sentences; the gap to naive Bayes stays.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Exercise 4. What comes out?

    Each line is a pair of texts. Put it under the cosine similarity of their two rows, over the vocabulary of this lesson.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_comes_out = L.sort("what-comes-out")
    sort_comes_out
    return (sort_comes_out,)


@app.cell(hide_code=True)
def check_sort(sort_comes_out):
    result_sort = L.check("what-comes-out", sort_comes_out)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Spam or not: naive Bayes](/course/open/lm-naive-bayes?lang=en) kept a bag of words for each class; this lesson wrote a bag of words for each sentence as a vector over a fixed vocabulary, stored it as a sparse row, and compared two such rows with the cosine similarity from [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en). Small letters and a list of stop words, as in [Normalizing: case, accents and stop words](/course/open/nlp-normalize?lang=en), change which columns there are. Plain counts let common words such as *the* decide too much. The next lesson of the domain, *TF-IDF: weighing the words*, gives each word a weight by how many documents it occurs in.

    **An everyday example.** A news site that shows *related articles* under a story can find them in this way: turn every article into a vector of word counts, and show the articles with the highest cosine similarity.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_cosine, result_similar, result_sort, result_vectorize):
    L.wrap_up(result_vectorize, result_cosine, result_similar, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
