# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-tf-idf. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    from everything_ai.lesson import lesson

    L = lesson("nlp-tf-idf", "en")
    C = L.checks   # the course's words, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    A question about [A document as a vector](/course/open/nlp-document-vectors?lang=en), and one about [Exponents and logarithms](/course/open/math-exp-log?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_common():
    quick_common = L.choice(
        "quick-common",
        question="Among 2,400 training sentences, the most similar one to *The delivery was on time.* turned out to be *The bus boy on the other hand was so rude.* What do these two sentences have in common?",
        options={
            "tone": "A negative tone, which the cosine similarity can pick up",
            "right": "*the*, *was* and *on*, words found in many sentences",
            "topic": "The topic: both are about service",
        },
        correct="right",
        why={
            "right": "They share *the*, *was* and *on*, and not one word about a delivery or time. Words like these are in a large share of all sentences, so they say little about any one sentence, yet plain counts let them decide much of the cosine similarity. This lesson gives each word a weight that is low when the word is in many documents.",
            "tone": "The cosine similarity of counts sees only which words two sentences share, not their tone, and *The delivery was on time.* is a positive sentence. See [A document as a vector](/course/open/nlp-document-vectors?lang=en).",
            "topic": "One is about a delivery, the other about a bus boy, and they share no word about either. See [A document as a vector](/course/open/nlp-document-vectors?lang=en).",
        },
    )
    quick_common
    return (quick_common,)


@app.cell(hide_code=True)
def quick_check_log():
    quick_log = L.choice(
        "quick-log",
        question="What does `np.log10(1000)` give?",
        options={
            "e": "6.908, the power of e that gives 1000",
            "hundred": "100.0, which is 1000 divided by 10",
            "right": "3.0, since 10 × 10 × 10 = 1000",
            "small": "0.001, one divided by 1000",
        },
        correct="right",
        why={
            "right": "`np.log10` asks which power of 10 gives the number: $10^3 = 1000$, so it gives `3.0`. Each factor 10 adds 1: `np.log10(100)` is 2 and `np.log10(10)` is 1. This lesson uses that to turn *in ten times as many documents* into *1 lower*.",
            "e": "6.908 is `np.log(1000)`, the natural log, with base $e$. `np.log10` has base 10. See [Exponents and logarithms](/course/open/math-exp-log?lang=en).",
            "hundred": "A logarithm gives an exponent, not a quotient. See [Exponents and logarithms](/course/open/math-exp-log?lang=en).",
            "small": "0.001 is $10^{-3}$, one divided by 1000; the logarithm of 1000 is the exponent itself. See [Exponents and logarithms](/course/open/math-exp-log?lang=en).",
        },
    )
    quick_log
    return (quick_log,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [A document as a vector](/course/open/nlp-document-vectors?lang=en), common words such as *the* and *was* decided much of the cosine similarity. A search shows the problem more sharply. Cut the five English books of the course, *Alice's Adventures in Wonderland*, *Pride and Prejudice*, *The Adventures of Sherlock Holmes*, *Frankenstein* and *Moby-Dick*, into their 251 chapters, and search them for *the cat* by adding up how often each chapter has the two words. First comes *The Adventure of the Speckled Band*, a Sherlock Holmes story of 9,882 words, with *the* 598 times and *cat* 3 times. *Pig and Pepper*, the chapter of *Alice* where the Cheshire Cat turns up, has *cat* 22 times, but *the* only 150 times, and comes 39th. *the* is in every chapter, so it says nothing about which chapter you are looking for, yet it counts as much as *cat*.

    This lesson gives each word a weight: high when the word is in few chapters, low when it is in many, and 0 when it is in all of them. Jurafsky and Martin write in *Speech and Language Processing*: "Two term weighting schemes are common: tf-idf and a variant of tf-idf called BM25." For the second part of that weight they cite a paper by Karen Spärck Jones from 1972, and so do Christopher Manning, Prabhakar Raghavan and Hinrich Schütze in *Introduction to Information Retrieval*, who describe the paper as "detailed experiments showing the use of inverse document frequency in term weighting." They add that Hans Peter Luhn's papers of 1957 and 1958 "may be thought of as anticipating tf–idf and related weighting schemes."

    **An everyday example.** In a cookbook, almost every recipe has salt, and only a few have saffron. A recipe that asks for saffron tells you much more about that recipe than one that asks for salt, though salt comes up far more often.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="The 251 chapters together use 22,858 different words. How many of those words occur in every one of the 251 chapters?",
        options={
            "right": "About 5",
            "fifty": "About 50",
            "hundreds": "About 500",
            "thousands": "About 5,000",
        },
        reveal="**5**: *a*, *it*, *of*, *that* and *the*. *and* misses one chapter: chapter 122 of *Moby-Dick*, *Midnight Aloft.—Thunder and Lightning*, has *and* only in its title, and the course leaves the headings out. The chapter is a short scene of 47 words: Tashtego, high up on a yard of the ship in a thunderstorm, wants rum instead of thunder. At the other end, 9,400 of the 22,858 words, 41.1%, occur in one chapter only.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Term frequency and document frequency

    The **term frequency** of a word in a document is how often the word occurs in it: the count of [A document as a vector](/course/open/nlp-document-vectors?lang=en). Manning and colleagues: "This weighting scheme is referred to as term frequency". The **document frequency** of a word is the number of documents it occurs in, however often it occurs in each. Jurafsky and Martin: "The document frequency $\text{df}_t$ of a term t is the number of documents it occurs in." Both textbooks say term where this lesson says word, and t stands for any term. Here a document is one chapter, so the document frequency of *hatter* is 3: the Hatter is at the table in chapter 7 of *Alice* and a witness in chapter 11, and chapter 6 only mentions the Hatter: the Cheshire Cat tells Alice where a Hatter lives.

    Why count documents, and not every time a word occurs? Manning and colleagues compare two words in a collection of news stories, Reuters-RCV1: *try* occurs 10,422 times and *insurance* 10,440 times, about as often, but *try* is spread over 8,760 documents and *insurance* over 3,997. The books of the course have such a pair as well: *alice* occurs 397 times, in 15 chapters, and *life* 399 times, in 157 chapters. A chapter with *alice* in it is probably about Alice; a chapter with *life* in it can be about anything. The total count, which Manning and colleagues call the **collection frequency**, cannot tell the two apart; the document frequency can.

    ### Inverse document frequency

    The **inverse document frequency**, IDF for short, turns the document frequency into a weight. With $N$ documents, a word $w$ that occurs in $\text{df}(w)$ of them gets

    $$\text{idf}(w) = \log_{10} \frac{N}{\text{df}(w)}$$

    Jurafsky and Martin give this formula and say what it does: "The fewer documents in which a term occurs, the higher this weight; the lowest weight of 0 is assigned to terms that occur in every document." The logarithm keeps the weights small: a word in one chapter of 251 gets $\log_{10} 251 = 2.40$, not 251, and a word in ten times as many chapters gets 1 less. Manning and colleagues use base 10 in their example as well, and add that "the precise base of the logarithm is not material to ranking": another base multiplies every weight by the same number. The picture shows the IDF of some words of the books.
    """)
    return


@app.cell(hide_code=True)
def idea_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### TF-IDF

    **TF-IDF** multiplies the two. The weight of a word $w$ in a document $d$ is its term frequency times its inverse document frequency:

    $$\text{tf-idf}(w, d) = \text{tf}(w, d) \times \log_{10} \frac{N}{\text{df}(w)}$$

    That is how Manning and colleagues define it, with the count itself as the term frequency. In their words, the weight is

    1. "highest when t occurs many times within a small number of documents (thus lending high discriminating power to those documents);"
    2. "lower when the term occurs fewer times in a document, or occurs in many documents (thus offering a less pronounced relevance signal);"
    3. "lowest when the term occurs in virtually all documents."

    A word that a document lacks has count 0, so its weight in that document is 0 too.

    ### A search, scored

    To search, give each document a **score**: add up the TF-IDF weights of the search words in that document, and put the documents with the highest score first. A search word that a document lacks adds nothing. Manning and colleagues build the score in this way: "We can refine this idea so that we add up not the number of occurrences of each query term t in d, but instead the tf–idf weight of each term in d." For *the cat*, *the* adds 0 to every chapter, so *cat* decides: in *Pig and Pepper*, 22 times *cat* with IDF 1.50 gives 32.9, the highest score of all 251 chapters.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **term frequency** (TF) | how often a word occurs in one document |
        | **document frequency** (df) | the number of documents a word occurs in |
        | **collection frequency** | how often a word occurs in all documents together |
        | **inverse document frequency** (IDF) | $\log_{10}(N / \text{df})$: high for a word in few documents, 0 for a word in all of them |
        | **TF-IDF** | the weight of a word in a document: its term frequency times its inverse document frequency |
        | **score** | in this lesson: the sum of the TF-IDF weights of the search words in one document |
        | $N$ | the number of documents |
        """),
        L.watch_out("The IDF depends on what counts as a document. Take the five books as five documents, and *life*, which is in every book, gets 0, like *the*; with chapters as documents it gets 0.20. And a word that occurs once by accident, such as a misspelling, gets the highest IDF of all."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Cut each document into words, in small letters.
    2. Count for each word the number of documents it occurs in: its document frequency.
    3. Turn each document frequency into an inverse document frequency: $\log_{10}$ of the quotient $N / \text{df}$.
    4. Multiply the count of each word in a document by the IDF of the word: its TF-IDF weight in that document.
    5. For a search, add up the weights of the search words in each document, and put the documents with the highest score first.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The code below turns five document frequencies into inverse document frequencies, for $N = 251$: it prints each document frequency, the quotient $N / \text{df}$, and its logarithm with base 10.
    """)
    return


@app.cell
def example_cell():
    n = 251
    for frequency in [251, 176, 15, 3, 1]:
        print(frequency, round(n / frequency, 2), round(np.log10(n / frequency), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    A word in all 251 chapters gets 0, since $251 / 251 = 1$ and $\log_{10} 1 = 0$. *said*, in 176 chapters, gets 0.154; *alice*, in 15, gets 1.224; *hatter*, in 3, gets 1.923; and a word in one chapter, such as *steelkilt*, the name of a sailor in a story told in *Moby-Dick*, gets the most: 2.4. From 251 chapters down to 1, the quotient grows 251 times, and its log with base 10 only from 0 to 2.4.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_df():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. document_frequency: in how many documents?

        `small_docs`, the course's `SMALL_DOCS`, holds four tiny documents, each a list of words, and `chapters` the 251 chapters, from the course's `chapter_words`. `chapter_label` gives the name of a chapter by its place. Write `document_frequency(docs)`: give back a dictionary from each word to the number of documents of `docs` it occurs in. A word that occurs three times in one document counts once for that document.

        For `document_frequency(small_docs)`, *the* gets 4, *hatter* and *tea* get 2, and every other word gets 1: *tea* occurs four times, but in two documents. Replace the `...`. The check calls that and `document_frequency([["whale", "whale", "whale"], ["white", "whale"], ["ship"]])`, then cases of its own: no documents at all, an empty document, a word in every document, the same document twice, and the 251 chapters.
        """),
        L.task("Count for each word the number of documents it occurs in."),
    ])
    return


@app.cell
def chapters_cell():
    small_docs = C.SMALL_DOCS
    chapters = C.chapter_words()
    for doc in small_docs:
        print(doc)
    print(len(chapters), len(chapters[6]), chapters[6][:10])
    print(C.chapter_label(6, L.lang))
    return


@app.cell(hide_code=True)
def exercise_df():
    box_df = L.code_box("document-frequency", lines=9)
    box_df
    return (box_df,)


@app.cell(hide_code=True)
def check_df(box_df):
    result_df = L.check("document-frequency", box_df.value)
    result_df
    return (result_df,)


@app.cell(hide_code=True)
def hints_df():
    L.hints(
        "document-frequency",
        [
            "Start with `df = {}`, and go over the documents with `for doc in docs:`. Inside, go over the different words of the document, each once. For each such `word`, add 1 to `df[word]`, or put in 1 when `df` does not hold the word yet. After the loops, give back `df`.",
            "*tea* occurs three times in the fourth document of `small_docs`. How much should that document add to the document frequency of *tea*, and which kind of collection, made from a list, holds each of its items once?",
            r"`        for word in ___:`",
        ],
        why=r"`df = {}`; `for doc in docs:` and inside it `for word in set(doc):`, with `if word in df:` and `df[word] = df[word] + 1`, and `else:` with `df[word] = 1`; after the loops, `return df`.",
    )
    return


@app.cell(hide_code=True)
def text_after_df():
    mo.md(r"""
    In the picture, your function counted the document frequency of every word of the 251 chapters. Many words are rare: 9,400 of the 22,858 occur in one chapter only, and 14,871 in at most three. Here are the course's document frequencies of the small documents, `SMALL_DF`, and of the chapters, from `chapter_df`, and the counts of chapter 7 of *Alice*, *A Mad Tea-Party*, from `chapter_counts`. The last two hold the most common words first.
    """)
    return


@app.cell
def counts_cell():
    small_df = C.SMALL_DF
    df = C.chapter_df()
    counts = C.chapter_counts(6)
    print(small_df)
    print(df["the"], df["hatter"], df["tea"])
    print(counts["the"], counts["hatter"], counts["tea"])
    return


@app.cell(hide_code=True)
def text_exercise_weights():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. tf_idf: the weights of one document

        Write `tf_idf(counts, df, n)`, where `counts` is a dictionary from each word of one document to its count, `df` the document frequencies of all words, and `n` the number of documents. Give back a dictionary from each word of `counts` to its weight: its count times `np.log10(n / df[word])`.

        For `tf_idf({"the": 2, "hatter": 1, "poured": 1, "tea": 1}, small_df, 4)`, the counts of the first small document, the result is `{"the": 0.0, "hatter": 0.30103, "poured": 0.60206, "tea": 0.30103}`: *the* is in all four documents, *poured* in one, so it gets $\log_{10} 4 = 0.602$, and *hatter* and *tea* are in two. Replace the `...`. The check calls that and `tf_idf({"tea": 3, "said": 1, "the": 1, "hatter": 1}, small_df, 4)`, then cases of its own: no words at all, a word in every document, another `n`, and *A Mad Tea-Party* over the 251 chapters.
        """),
        L.task("Weigh each count of one document by the inverse document frequency of its word."),
    ])
    return


@app.cell(hide_code=True)
def exercise_weights():
    box_weights = L.code_box("tf-idf", lines=5)
    box_weights
    return (box_weights,)


@app.cell(hide_code=True)
def check_weights(box_weights):
    result_weights = L.check("tf-idf", box_weights.value)
    result_weights
    return (result_weights,)


@app.cell(hide_code=True)
def hints_weights():
    L.hints(
        "tf-idf",
        [
            "Start with `weights = {}`, and go over the words with `for word in counts:`. For each `word`, put in `weights[word]` the count of the word times its IDF, `np.log10(n / df[word])`. After the loop, give back `weights`.",
            "*tea* occurs 3 times in the fourth small document, and is in 2 of the 4 documents, so its IDF is $\\log_{10}(4 / 2) = 0.301$. What do you multiply that by, and how do you write the count of any word of `counts`?",
            r"`        weights[word] = ___ * np.log10(n / df[word])`",
        ],
        why=r"`weights = {}`; `for word in counts:` with `weights[word] = counts[word] * np.log10(n / df[word])`; after the loop, `return weights`.",
    )
    return


@app.cell(hide_code=True)
def text_after_weights():
    mo.md(r"""
    In the picture, your function weighed *A Mad Tea-Party*. By count, its top ten are *the*, *said*, *and*, *i*, *it*, *alice*, *you*, *a*, *to* and *of*: words of any chapter, apart from *alice*. By your weights, they are *hatter*, *alice*, *dormouse*, *hare*, *march*, *twinkle*, *treacle*, *tea*, *said* and *muchness*. The Hatter, the Dormouse and the March Hare sit at the table, the Hatter sings *Twinkle, twinkle, little bat*, the Dormouse tells a story of three sisters who live at the bottom of a treacle well, and *muchness* comes up in *much of a muchness*. *the*, 158 times in the chapter, gets 0.

    Here are the course's weights of the four small documents, `SMALL_WEIGHTS`, and of all 251 chapters, from `chapter_weights`, in the order of `chapters`. Each is a list of dictionaries, one dictionary for each document.
    """)
    return


@app.cell
def weights_cell():
    small_weights = C.SMALL_WEIGHTS
    weights = C.chapter_weights()
    print(small_weights[3])
    print(len(weights), round(weights[6]["hatter"], 2), weights[6]["the"])
    return


@app.cell(hide_code=True)
def text_exercise_score():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. score: a search

        Write `score(query, weights)`, where `query` is a list of search words and `weights` the weights of one document: add up the weights of the search words, and give back the sum. A search word that `weights` does not hold adds nothing.

        For `score(["hatter", "tea"], small_weights[3])`, the result is about 1.204: *tea* has weight 0.903 in the fourth document and *hatter* 0.301. Replace the `...`. The check calls that and `score(["the", "queen"], small_weights[1])`, then cases of its own: a search word the document lacks, no search words at all, a word with weight 0, a mix of these, and *the cheshire cat* against chapter 6 of *Alice*, *Pig and Pepper*.
        """),
        L.task("Add up the weights of the search words in one document."),
    ])
    return


@app.cell(hide_code=True)
def exercise_score():
    box_score = L.code_box("score", lines=6)
    box_score
    return (box_score,)


@app.cell(hide_code=True)
def check_score(box_score):
    result_score = L.check("score", box_score.value)
    result_score
    return (result_score,)


@app.cell(hide_code=True)
def hints_score():
    L.hints(
        "score",
        [
            "Start with `total = 0`, and go over the search words with `for word in query:`. When the document has a weight for the `word`, add that weight to `total`. After the loop, give back `total`.",
            "*cat* is not in the first small document. What happens with `small_weights[0][\"cat\"]`, and how do you ask a dictionary first whether it holds a key?",
            r"`        if word in ___:`",
        ],
        why=r"`total = 0`; `for word in query:` with `if word in weights:` and `total = total + weights[word]`; after the loop, `return total`.",
    )
    return


@app.cell(hide_code=True)
def text_after_score():
    mo.md(r"""
    In the picture, your function scored the 251 chapters for three searches. For *the cheshire cat*, plain counts put *The Adventure of the Speckled Band* first, with its 598 times *the*; your scores put *Pig and Pepper* first and *The Queen’s Croquet-Ground*, where the Cat turns up again, second. *who stole the tarts* finds chapter 11 of *Alice*, *Who Stole the Tarts?*, and *the red headed league* finds the Sherlock Holmes story *The Red-Headed League*: the course cuts *Red-Headed* into the words *red* and *headed*.

    ### How good is it?

    TF-IDF has many variants. Manning and colleagues doubt the count itself: "It seems unlikely that twenty occurrences of a term in a document truly carry twenty times the significance of a single occurrence." Jurafsky and Martin take the log of the count as well: "We usually use the log10 of the word frequency, rather than the raw count." Their term frequency is $1 + \log_{10}$ of the count, and 0 for a count of 0: a word that occurs 10 times gets 2, and 100 times 3. scikit-learn's `TfidfVectorizer` keeps the count by default, but takes the natural log and gives every word $\log \frac{1 + N}{1 + \text{df}} + 1$, so that a word in every document keeps a small weight; then it divides the vector of each document by its length. They are all TF-IDF; this lesson uses the form of Manning and colleagues, with base 10.

    TF-IDF also does much of what a list of stop words does. Jurafsky and Martin: "In modern IR systems, the use of stop lists is much less common, partly due to improved efficiency and partly because much of their function is already handled by IDF weighting, which downweights function words that occur in every document." IR is short for information retrieval, the field of search.

    The score of this lesson adds weights up, so a long chapter that has a search word many times can beat a short chapter that is all about it. Two later lessons of the domain deal with that: *Ranking by cosine: the vector space model* divides by the length of each document, and *BM25: the classic ranking* lets each extra occurrence of a word count less.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: search the books

    Type a few search words; a word you type twice counts once. The picture shows the IDF of each search word, the five chapters with the highest score, the weight of each search word in each of them, and the chapter that plain counts would put first. Try *white whale*, a name from one of the books, *the* alone, or a word that is in no chapter.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_query = mo.ui.text(value=C.DEMO_QUERY, label="Search words", full_width=True)
    demo_query
    return (demo_query,)


@app.cell(hide_code=True)
def demo_look(demo_query):
    mo.Html(C.demo_picture(demo_query.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    For *a ship in the ice*, *a* and *the* add 0 and *in* almost nothing, so *ice* and *ship* decide. Chapter 24 of *Frankenstein*, where the story ends in the ice of the far north, has *ice* 18 times and comes first, with 18.84; letter 4 of the same book, also in the ice, comes second. Chapter 135 of *Moby-Dick*, *The Chase.—Third Day*, comes third, with *ship* 19 times and *ice* 3 times. Plain counts would put *The Adventure of the Speckled Band* first: 1,030 times *a*, *in* and *the*, and not one *ship* or *ice*.
    """)
    return


@app.cell(hide_code=True)
def text_explore():
    mo.md(r"""
    ### Try it: other weights

    Pick a chapter, and see its ten words with the highest weight: with or without the IDF, and with the count itself or with $1 + \log_{10}$ of the count, the term frequency of Jurafsky and Martin.
    """)
    return


@app.cell(hide_code=True)
def explore_controls():
    explore_chapter = mo.ui.dropdown(options=C.chapter_options(L.lang), value=C.chapter_label(6, L.lang), label="Chapter", full_width=True)
    explore_idf = mo.ui.checkbox(value=True, label="Times the IDF")
    explore_log = mo.ui.checkbox(value=False, label="1 + log10 of the count, instead of the count")
    mo.vstack([explore_chapter, explore_idf, explore_log])
    return explore_chapter, explore_idf, explore_log


@app.cell(hide_code=True)
def explore_look(explore_chapter, explore_idf, explore_log):
    mo.Html(C.explore_picture(explore_chapter.value, explore_idf.value, explore_log.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_explore():
    mo.md(r"""
    Without the IDF, the top ten of *A Mad Tea-Party* are the words with the highest counts, from *the*, 158 times, on. With the IDF and the count itself, the characters come first. With $1 + \log_{10}$ of the count, a word that occurs 32 times weighs only $1 + \log_{10} 32 = 2.5$ times as much as a word that occurs once, so rare words climb: *dormouse* goes to the top, and *muchness*, 3 times in this chapter and in no other, comes sixth.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Exercise 4. What weight?

    Each line gives a word, how often it occurs in a chapter, and in how many of the 251 chapters it occurs; each line is about a chapter of its own. Put each word under its TF-IDF weight in its chapter.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_weight = L.sort("what-weight")
    sort_weight
    return (sort_weight,)


@app.cell(hide_code=True)
def check_sort(sort_weight):
    result_sort = L.check("what-weight", sort_weight)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [A document as a vector](/course/open/nlp-document-vectors?lang=en) counted the words of each document; this lesson weighed each count by the inverse document frequency of its word, a logarithm as in [Exponents and logarithms](/course/open/math-exp-log?lang=en), so that a word in every document counts for nothing and a rare word counts most. Adding up those weights for the words of a search puts the chapters that are about the search first. The next lesson of the domain, *Logistic regression on words*, puts TF-IDF vectors into a classifier that learns a weight for each word from labeled examples.

    **An everyday example.** A search box on a website that sorts its results by how well they match can work in this way: the pages that have the rare words of your search, and have them often, come first.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_df, result_score, result_sort, result_weights):
    L.wrap_up(result_df, result_weights, result_score, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
