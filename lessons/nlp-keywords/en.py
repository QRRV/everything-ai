# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-keywords. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-keywords", "en")
    C = L.checks   # the words of eight books, the 251 chapters of nlp-tf-idf, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_idf():
    quick_idf = L.choice(
        "quick-idf",
        question=r"In *TF-IDF: weighing the words*, a word that occurs in $\text{df}$ of the $N$ documents gets the IDF $\log_{10}(N / \text{df})$. With the 251 chapters as documents, what is the IDF of a word that occurs in all of them, such as *the*?",
        options={
            "one": "1",
            "n": "251, the number of chapters",
            "right": "0",
            "max": "About 2.4: the highest IDF there is, for the most common word",
        },
        correct="right",
        why={
            "right": r"$\log_{10}(251 / 251) = \log_{10} 1 = 0$: a word in every chapter says nothing about any one of them. This lesson takes the words of a chapter with the highest TF-IDF as its keywords, and then looks at pairs of words.",
            "one": r"The IDF looks at the share of the chapters that hold the word. What is 251 / 251, and what is the log of that? See [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en).",
            "n": r"251 / 251 is 1, and the IDF takes the log of it. What is $\log_{10} 1$? See [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en).",
            "max": r"About 2.4 is $\log_{10} 251$, the IDF of a word that occurs in only one chapter. Which word gets the lowest IDF? See [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en).",
        },
    )
    quick_idf
    return (quick_idf,)


@app.cell(hide_code=True)
def quick_check_weight():
    quick_weight = L.choice(
        "quick-weight",
        question="A word occurs 4 times in a document, and in 10 of the 1,000 documents of a collection. What is its TF-IDF weight in that document, with the IDF of *TF-IDF: weighing the words*?",
        options={
            "times": "400: 4 times 1,000 / 10",
            "right": "8",
            "idf": "2",
            "share": "0.04: the count 4 times the share of the documents that hold the word, 10 / 1,000",
        },
        correct="right",
        why={
            "right": r"The IDF is $\log_{10}(1000 / 10) = \log_{10} 100 = 2$, and the weight is the count times the IDF: 4 × 2 = 8. The keywords of a chapter in this lesson are its words with the highest weight.",
            "times": "1,000 / 10 is 100. What does the IDF do with that 100 before it is multiplied by the count? See [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en).",
            "idf": "2 is the IDF of the word. What does TF-IDF multiply it by? See [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en).",
            "share": r"TF-IDF multiplies the count by $\log_{10}(N / \text{df})$. Which way round does that quotient go, and what does the log do with it? See [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en).",
        },
    )
    quick_weight
    return (quick_weight,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    What is a text about? The words it uses most often say little: *the*, *and* and *of* are at the top of nearly every text in English. In *TF-IDF: weighing the words*, you weighed the words of a chapter against the other chapters, so that a word in every chapter got nothing. The words with the highest weight say what the chapter is about: its keywords. But many of the things a text is about have a name of two words, such as *Mock Turtle*, *Baker Street* or *natural philosophy*, and in a fixed expression such as *of course*, the two words belong together as well. One by one, their words say less than the pair.

    In 1990, Kenneth Church and Patrick Hanks used a measure from information theory to find such pairs by counting, in a paper about lexicography, the making of dictionaries. Which words go together is part of what a dictionary has to say about a word. This lesson counts the pairs of neighboring words in eight books, five in English and three in Dutch, and finds their terms.

    **An everyday example.** The keyboard of your phone suggests the next word while you type: after *Merry*, it offers *Christmas*. It has counted which words tend to follow which, and *Merry Christmas* is a pair that occurs far more often than its two words would by chance.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_chance = L.guess(
        "guess",
        question="*Alice* has 26,776 words, and *of* is followed by *the* 133 times. If the same words stood in a random order, how often would *of* be followed by *the*?",
        options={
            "same": "About 130, as now: a random order keeps every word",
            "right": "About 30",
            "few": "About 3",
            "none": "About 0.1: in a random order, a pair of two given words hardly ever comes back",
        },
        reveal="**About 32 times.** *of* occurs 515 times and *the* 1,651 times. In a random order, the word after an *of* is *the* with a chance of 1,651 in 26,776, so the 515 times *of* give about 515 × 1,651 / 26,776 = 31.8 pairs *of the*. The real 133 is about four times as many: *of the* is a frequent pair mostly because *of* and *the* are frequent words. *mock turtle* would occur 0.12 times by chance, and occurs 54 times: about 450 times as many.",
    )
    guess_chance
    return (guess_chance,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    A **keyword** of a document is a word that says what it is about. The simplest way of **keyword extraction** is the one of *TF-IDF: weighing the words*: take the words of the document with the highest TF-IDF weight, words that occur often in this document and in few others. For the chapter *A Mad Tea-Party* of *Alice*, that gives *hatter*, *alice*, *dormouse*, *hare* and *march* at the top.

    *march* and *hare* are not two keywords, though: together they are the name of the March Hare. A **collocation** is a pair of words that occur together more often than chance, such as *mock turtle*, *of course* or *strong tea*. Some collocations are names, some are fixed expressions, and many are **terms**: the words that a field uses for its own things, such as *natural philosophy* in *Frankenstein* or *quarter deck* in *Moby-Dick*.

    How often is *more often than chance*? Count the words of a text: $N$ words in all, of which $c(x)$ are the word $x$ and $c(y)$ the word $y$, and $c(x, y)$ pairs in which $x$ is directly followed by $y$. The chance that a word is $x$ is $P(x) = c(x) / N$, and the chance that a pair is $x$ followed by $y$ is $P(x, y) = c(x, y) / N$. Two words are **independent** when the one tells you nothing about the other: then the chance of the pair is the chance of the one times the chance of the other, $P(x) \cdot P(y)$. For independent words, as in a random order, the pair would occur about

    $$\text{chance count} = N \cdot P(x) \cdot P(y) = \frac{c(x) \cdot c(y)}{N}$$

    times. The **pointwise mutual information** of the pair, PMI for short, compares the real count with the chance count:

    $$\text{PMI}(x, y) = \log_2 \frac{P(x, y)}{P(x) \cdot P(y)} = \log_2 \frac{c(x, y)}{c(x) \cdot c(y) / N}$$

    Church and Hanks: "Informally, mutual information compares the probability of observing x and y together (the joint probability) with the probabilities of observing x and y independently (chance)." And: "If there is a genuine association between x and y, then the joint probability P(x,y) will be much larger than chance P(x) P(y), and consequently I(x,y) >> 0." They write $I(x, y)$ for what is now called PMI, and count both the words and the pairs out of $N$, the size of the corpus, as this lesson does.

    The log with base 2 turns *twice as often as chance* into 1, four times into 2, and 64 times into 6. A PMI of 0 means exactly as often as chance, and a pair that occurs less often than chance gets a PMI below 0.

    The picture shows both: the keywords of *A Mad Tea-Party* by TF-IDF, and six pairs of all of *Alice* with their counts, their chance counts and their PMI.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    *of the* occurs about four times as often as chance, a PMI of 2.07; *mock turtle* and *march hare* occur hundreds of times as often. The last pair, *abide figures*, from the Duchess's *I never could abide figures!*, occurs only once, and still gets the highest PMI of the six. Exercises 3 and 4 come back to that.
    """)
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.md(r"""
    ### New words and symbols

    | Word | Meaning |
    |---|---|
    | **keyword** | a word that says what a document is about; here, one of its words with the highest TF-IDF |
    | **keyword extraction** | finding the keywords of a document by computer |
    | **collocation** | a pair of words that occur together more often than chance, such as *mock turtle* or *of course* |
    | **term** | a word or a fixed group of words that names one thing in a field, such as *natural philosophy* |
    | **independent** | two words are independent when the one tells you nothing about the other: $P(x, y) = P(x) \cdot P(y)$ |
    | **chance count** | how often a pair would occur if its words were independent: $c(x) \cdot c(y) / N$ |
    | **pointwise mutual information** (PMI) | $\log_2$ of how many times as often as chance a pair occurs: 0 for as often, 1 for twice as often, below 0 for less often |
    | `{("mock", "turtle"): 54}` | a tuple can be a key of a dictionary, a list cannot |
    | `x, y = pair` | takes a tuple of two values apart into two names |
    | `sorted(scores, reverse=True)` | sorts a list from large to small; tuples are compared by their first element, and on a tie by the next one |
    | `C.book_words(key)` | the words of a book in small letters, in their order, such as `C.book_words("alice")`; the keys of the eight books are in `C.BOOKS` |
    | `C.word_counts(key)`, `C.book_pairs(key)` | how often each word, and each pair of neighboring words, occurs in a book |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Count the words of the text, and the pairs of neighboring words.
    2. For each pair, work out its chance count: $c(x) \cdot c(y) / N$.
    3. Its PMI is $\log_2$ of its real count divided by its chance count.
    4. Keep the pairs that occur at least a minimum number of times, and put the highest PMI first: those are the terms of the text.
    5. For keywords of one word, take the words with the highest TF-IDF.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Church and Hanks give an example from 44,344,077 words of news of the Associated Press from 1988: *set* occurs 13,046 times in it, *off* 20,693 times, and *set off* 463 times. The code works out the chance count of *set off*, how many times as often it occurs, and its PMI. Then it sorts three rows of PMI and pair from *Alice* from the highest PMI down.
    """)
    return


@app.cell
def example_cell():
    import numpy as np

    _n = 44_344_077
    _chance = 13_046 * 20_693 / _n
    print(round(_chance, 1))
    print(round(463 / _chance, 1))
    print(round(np.log2(463 / _chance), 2))

    _scores = [(2.07, ("of", "the")), (8.80, ("mock", "turtle")), (4.12, ("said", "alice"))]
    print(sorted(_scores, reverse=True))
    return (np,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    By chance, *set off* would occur about 6 times; it occurs 76 times as often, and $\log_2 76.1 = 6.25$. Church and Hanks write that it is about 6.2, and what that means: "the joint probability is more than 2^6 = 64 times larger than chance." `sorted` with `reverse=True` puts the rows from the highest PMI down: it compares the first element of each tuple, the PMI, and would only look at the pairs on a tie.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pairs():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. pair_counts: the pairs of a text

        Write `pair_counts(words)`, where `words` is a list of words. Give back a dictionary from each pair of neighboring words, a tuple of a word and the word after it, to how often that pair occurs in `words`. The loop goes over every place `i`, counting from 0, that has a word after it. Replace the two `...`: the pair at place `i`, and the lines that add 1 to its count in `pairs`, or put in 1 for a pair that `pairs` does not hold yet.

        The check calls `pair_counts(song)`, with the words `["beau", "ootiful", "soo", "oop", "beau", "ootiful", "soo", "oop"]` from the song of the Mock Turtle, *Beau—ootiful Soo—oop!* twice (in the lesson: `C.SONG`). It gives `{("beau", "ootiful"): 2, ("ootiful", "soo"): 2, ("soo", "oop"): 2, ("oop", "beau"): 1}`. Then cases of its own: an empty list; `["alice"]`; `["ha", "ha", "ha"]`; and all the words of *Alice* and of *Max Havelaar* (in the lesson: `C.book_words("alice")` and `C.book_words("havelaar")`).
        """),
        L.task("Count the pairs of neighboring words in a text."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pairs():
    box_pairs = L.code_box("pair_counts", lines=10)
    box_pairs
    return (box_pairs,)


@app.cell(hide_code=True)
def check_pairs(box_pairs):
    result_pairs = L.check("pair_counts", box_pairs.value)
    result_pairs
    return (result_pairs,)


@app.cell(hide_code=True)
def hints_pairs():
    L.hints(
        "pair_counts",
        [
            "In the loop, `pair` is a tuple of two words of `words`: the one at place `i` and the one after it. Then `if pair in pairs:` adds 1 to its count, and `else:` puts in 1.",
            'Round brackets with a comma between make a tuple, such as `("mock", "turtle")`. Which place in `words` holds the word after the one at place `i`, and how do you add 1 to a value in a dictionary?',
            "`        pair = (words[i], ___)`",
        ],
        why='`pair = (words[i], words[i + 1])`, then `if pair in pairs:` with `pairs[pair] = pairs[pair] + 1` under it, and `else:` with `pairs[pair] = 1`. The example gives `{("beau", "ootiful"): 2, ("ootiful", "soo"): 2, ("soo", "oop"): 2, ("oop", "beau"): 1}`.',
    )
    return


@app.cell(hide_code=True)
def text_after_pairs():
    mo.md(r"""
    *Alice* has 26,776 words and 14,752 different pairs, and 11,234 of those occur only once. The most frequent pairs, *said the*, *of the* and *said alice*, are made of the most frequent words.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pmi():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. pmi: against chance

        Write `pmi(pair, pairs, counts, n)`, where `pair` is a tuple of two words, `pairs` the counts of the pairs of a text, as your `pair_counts` gives them, `counts` the counts of its words, and `n` its number of words. Give back the PMI of the pair: $\log_2$ of its count divided by its chance count, with `np.log2`. The first line takes the pair apart into `x` and `y`. Replace the two `...`.

        The check calls `pmi(("mock", "turtle"), alice_pairs, alice_counts, n_alice)`, with the pairs and the words of *Alice* and its number of words, 26,776 (in the lesson: `C.book_pairs("alice")`, `C.word_counts("alice")` and `len(C.book_words("alice"))`). It gives about 8.80. Then cases of its own: *of the*, *said alice*, *abide figures* and *twinkle twinkle* in *Alice*; the pair `("a", "b")` of the words *a b a b*, with the pairs `{("a", "b"): 2, ("b", "a"): 1}`, the counts `{"a": 2, "b": 2}` and 4 words, which gives 1.0; and *and the* in *Moby-Dick*.
        """),
        L.task("Work out how many times as often as chance a pair occurs, as a PMI."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pmi():
    box_pmi = L.code_box("pmi", lines=8)
    box_pmi
    return (box_pmi,)


@app.cell(hide_code=True)
def check_pmi(box_pmi):
    result_pmi = L.check("pmi", box_pmi.value)
    result_pmi
    return (result_pmi,)


@app.cell(hide_code=True)
def hints_pmi():
    L.hints(
        "pmi",
        [
            "`chance` is the count of `x` in `counts` times the count of `y`, divided by `n`. Then `return` gives `np.log2` of the count of the pair in `pairs`, divided by `chance`.",
            "For *mock turtle*: 57 times *mock*, 57 times *turtle* and 26,776 words give a chance count of 0.12, and the pair occurs 54 times. How do you write the count of the word `x` from the dictionary `counts`?",
            "`    chance = ___ / n`",
        ],
        why="`chance = counts[x] * counts[y] / n` and `return np.log2(pairs[pair] / chance)`. The example gives 8.80: *mock turtle* occurs 445 times as often as chance.",
    )
    return


@app.cell(hide_code=True)
def text_after_pmi():
    mo.md(r"""
    Church and Hanks give a rule of thumb: "As a very rough rule of thumb, we have observed that pairs with I(x, y) > 3 tend to be interesting, and pairs with smaller I(x, y) are generally not." In your picture, *of the* (2.07) and *and the* (0.61) are below 3, and *the queen* (3.95) and *said alice* (4.12) above it. In *Moby-Dick*, *and the* occurs 372 times, where chance would give 432: a PMI of −0.22, less often than chance.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_terms():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. terms: the terms of a book

        Write `terms(pairs, counts, n, least, k)`, with `pairs`, `counts` and `n` as in exercise 2. Give back the `k` pairs with the highest PMI among the pairs that occur at least `least` times, as a list of tuples `(PMI, pair)`, from the highest PMI down; on a tie in PMI, `sorted` with `reverse=True` puts the pair that comes later in the alphabet first. In the box, `pmi` is the course's own, which works like yours from exercise 2. Replace the two `...`: in the loop, the lines that add the row of a pair that occurs often enough to `rows`, and after the loop, the first `k` rows from the highest PMI down.

        The check calls `terms(alice_pairs, alice_counts, n_alice, 10, 5)`: the terms of *Alice* among the pairs that occur at least 10 times. It gives the rows of *march hare* (9.62), *mock turtle* (8.80), *your majesty* (8.75), *white rabbit* (8.65) and *minute or* (7.63), in that order, each a tuple such as `(9.62…, ("march", "hare"))`. Then cases of its own: in *Alice*, at least 1 with `k` 3, at least 20 with `k` 5, at least 1,000 with `k` 5 (no pair occurs that often) and at least 10 with `k` 0; the pairs of *a b a b* from exercise 2 with at least 1 and `k` 2; and at least 20 with `k` 5 in *Moby-Dick* and in *Max Havelaar*.
        """),
        L.task("Find the terms of a book: the pairs with the highest PMI that occur often enough."),
    ])
    return


@app.cell(hide_code=True)
def exercise_terms():
    box_terms = L.code_box("terms", lines=8)
    box_terms
    return (box_terms,)


@app.cell(hide_code=True)
def check_terms(box_terms):
    result_terms = L.check("terms", box_terms.value)
    result_terms
    return (result_terms,)


@app.cell(hide_code=True)
def hints_terms():
    L.hints(
        "terms",
        [
            "In the loop, an `if` keeps the pairs whose count in `pairs` is at least `least`, and `rows.append` adds a tuple of the PMI of the pair, from `pmi`, and the pair. After the loop, `return` gives the first `k` rows of `rows`, sorted with `reverse=True`.",
            "A tuple `(PMI, pair)` is sorted by its PMI first. Which call puts the rows from large to small, and how do you take the first `k` items of a list?",
            "`    return ___[:k]`",
        ],
        why="In the loop: `if pairs[pair] >= least:` with `rows.append((pmi(pair, pairs, counts, n), pair))` under it; after the loop, `return sorted(rows, reverse=True)[:k]`. The example gives *march hare*, *mock turtle*, *your majesty*, *white rabbit* and *minute or*.",
    )
    return


@app.cell(hide_code=True)
def text_after_terms():
    mo.md(r"""
    Your picture shows how much the minimum matters. With at least 1, the top is all pairs that occur once, such as *wooden spades* and *tittered audibly*. With at least 3, *yer honour*, *mary ann* and *beau ootiful* lead. With at least 10 and 20, the names and fixed expressions of the book come up: *march hare*, *mock turtle*, *white rabbit*, *did not*, *went on*. *minute or* comes from *in a minute or two*: with neighbors only, a fixed expression of three words or more shows up as its pairs. Church and Hanks used a minimum too: "Since the association ratio becomes unstable when the counts are very small, we will not discuss word pairs with f(x,y) ≤ 5." They call it a "simple but arbitrary threshold". They also counted *x* followed by *y* within five words, not only as neighbors, and write: "Smaller window sizes will identify fixed expressions (idioms such as bread and butter)".
    """)
    return


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it: the terms of a book

    Choose a book, a minimum count and an order. The table shows the twelve pairs with the highest PMI, or the most frequent pairs, among those that occur at least that often, each with its count, its chance count and its PMI. In which book do names fill the top, and in which fixed expressions? What do the most frequent pairs have in common, and how high is their PMI? Which minimum gives the best list for each book? The three Dutch books are from the 19th century and keep their old spelling, such as *mensch* for *mens*.
    """)
    return


@app.cell(hide_code=True)
def try_controls():
    try_book = mo.ui.dropdown(options=C.book_options(L.lang), value=C.book_label("alice", L.lang), label="Book")
    try_least = mo.ui.slider(1, 30, value=1, label="At least", show_value=True)
    try_order = mo.ui.radio(options={"highest PMI": "pmi", "most frequent": "count"}, value="highest PMI", label="Order", inline=True)
    mo.hstack([try_book, try_least, try_order], justify="start", gap=2, wrap=True)
    return try_book, try_least, try_order


@app.cell(hide_code=True)
def try_look(try_book, try_least, try_order):
    mo.Html(C.try_picture(try_book.value, try_least.value, try_order.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_once():
    mo.md(r"""
    ### Exercise 4. Pairs that occur once

    Without a minimum, the top of the PMI is full of pairs that occur once, in every book: in *Alice*, the 71 pairs with the highest PMI all occur once, such as *wooden spades*, and each gets 14.71, far more than *march hare* with 9.62.
    """)
    return


@app.cell(hide_code=True)
def choice_once_cell():
    why_once = L.choice(
        "why_once",
        question="Why do these pairs that occur once get the highest PMI?",
        options={
            "rare": "PMI rewards rare pairs on purpose: Church and Hanks wanted the rare pairs, since a dictionary needs them most",
            "right": "Its two words occur once each, so its chance count is 1 / N and the quotient N, the largest there is",
            "round": "Rounding: with counts of 1, the log cannot be worked out exactly",
            "log": "The log of a count of 1 is 0, and PMI turns that around into the highest value",
        },
        correct="right",
        why={
            "right": r"For *abide figures*, $c(x) = c(y) = c(x, y) = 1$, so the chance count is $1 \cdot 1 / 26{,}776$, and the pair occurs 26,776 times as often as chance: $\log_2 26{,}776 = 14.71$, the highest PMI a pair of *Alice* can get. One accident is enough for it. A minimum count, as in your `terms`, keeps such accidents out of the top.",
            "rare": "Church and Hanks did the opposite: they left out pairs with small counts. What does a count of 1 do to the quotient of PMI?",
            "round": "The numbers are exact: 1 · 26,776 / (1 · 1) is 26,776. What is the largest quotient a pair can get?",
            "log": "PMI takes the log of the quotient of the real count and the chance count, not of the count alone. What is that quotient for a pair whose words occur once each?",
        },
    )
    why_once
    return (why_once,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en), you weighed the words of a document against the other documents; here, the words with the highest weight became its keywords. Then you looked at pairs: your `pair_counts` counted the pairs of neighboring words, your `pmi` compared the count of a pair with the count that chance would give, and your `terms` kept the pairs that occur often enough and put the highest PMI first. The words of a text are not independent: *mock* is nearly always followed by *turtle*. PMI measures how far a pair is from independence, and a minimum count keeps pairs that occur once by accident out of the top. The next lesson, *Extraction with an LLM*, has an LLM take the facts out of a text.

    **An everyday example.** A shop that wants to suggest what goes with a product can count which products land in the same basket more often than chance. Bread and milk are in many baskets together, but mostly because both are in so many baskets; pasta and grated cheese together say more.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_pairs, result_pmi, result_terms, why_once):
    L.wrap_up(result_pairs, result_pmi, result_terms, why_once)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
