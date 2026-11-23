# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-word-embeddings. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-word-embeddings", "en")
    C = L.checks   # the course's vectors, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions, about [Embeddings: meaning as coordinates](/course/open/ml-embeddings-idea?lang=en) and [A word by the company it keeps](/course/open/nlp-distributional-hypothesis?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_cosine():
    quick_cosine = L.choice(
        "quick-cosine",
        question="What is the cosine similarity of $(3, 4)$ and $(6, 8)$?",
        options={
            "dot": "50",
            "right": "1",
            "half": "0.5",
        },
        correct="right",
        why={
            "right": "$(6, 8)$ is $(3, 4)$ twice as long, so the two point exactly the same way, and the lengths drop out: $50 / (5 \\cdot 10) = 1$. In this lesson every word gets such a vector, and the cosine similarity says how alike two words are.",
            "dot": "50 is the dot product, $3 \\cdot 6 + 4 \\cdot 8$. The cosine similarity divides it by the two norms, 5 and 10. See [Embeddings: meaning as coordinates](/course/open/ml-embeddings-idea?lang=en).",
            "half": "One vector is twice as long as the other, but the cosine similarity looks only at the direction, and that is the same. See [Embeddings: meaning as coordinates](/course/open/ml-embeddings-idea?lang=en).",
        },
    )
    quick_cosine
    return (quick_cosine,)


@app.cell(hide_code=True)
def quick_check_common():
    quick_common = L.choice(
        "quick-common",
        question="In *A word by the company it keeps*, rows of plain counts gave even target words that have nothing to do with each other much of their rows in common. Which words did that?",
        options={
            "pair": "hot, cold",
            "book": "white, sperm, whale",
            "right": "the, and, of",
        },
        correct="right",
        why={
            "right": "The most common words stand near almost every word, so they fill every row: more than half of 3,000 random pairs had a cosine similarity above 0.45. The word embeddings of this lesson are learned from such counts over billions of words, in a way that gives very frequent pairs of words less say.",
            "pair": "*hot* and *cold* had similar rows because they keep the same company, but that is two words, not all rows. See [A word by the company it keeps](/course/open/nlp-distributional-hypothesis?lang=en).",
            "book": "*white* found *sperm* because the rows knew only five books, one of them *Moby-Dick*. That is one row, not all of them. See [A word by the company it keeps](/course/open/nlp-distributional-hypothesis?lang=en).",
        },
    )
    quick_common
    return (quick_common,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The rows of the lesson before had thousands of columns, almost all of them 0, and the most common words decided much of every cosine similarity. Around 2013, fast methods appeared that give every word a short list of numbers instead, learned from billions of words. These vectors find related words, and something more. In 2013 Tomas Mikolov, Wen-tau Yih and Geoffrey Zweig wrote that the relation between male and female words is learned by itself: "with the induced vector representations, “King - Man + Woman” results in a vector very close to “Queen.”" The example became famous. Many search engines and translation systems, and every language model of today, start by turning words, or pieces of words, into vectors. This lesson works with real ones, finds the nearest neighbors of words, and checks how well such sums really work.

    **An everyday example.** Someone who knows that Paris is to France as Rome is to Italy can answer *Madrid is to …?* without ever having learned that pair.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_king = L.guess(
        "guess",
        question="The course has vectors of 50 numbers for the 50,000 most common English words. Take the vector of *king*, subtract the vector of *man* and add the vector of *woman*. Which of the 50,000 words has the vector with the highest cosine similarity with the result?",
        options={
            "queen": "queen",
            "right": "king",
            "princess": "princess",
            "woman": "woman",
        },
        reveal="**king**: with a cosine similarity of 0.886, the nearest vector is that of *king* itself. *queen* comes second, with 0.861, then *daughter*, *prince* and *throne*; *princess* comes sixth, and *woman* only 118th. The famous result leaves out the three words of the question, and then *queen* wins. This lesson shows why that step is needed, and what it says about the sums.",
    )
    guess_king
    return (guess_king,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Word embeddings

    A **word embedding** is an [embedding](/course/open/ml-embeddings-idea?lang=en) of a word: a short list of numbers, learned from a large amount of text, so that words that occur in similar contexts get similar vectors. The rows of [A word by the company it keeps](/course/open/nlp-distributional-hypothesis?lang=en) were sparse: a column for every context word, almost every count 0. A word embedding is **dense**: a few dozen to a few hundred numbers, hardly any of them 0, and none of them stands for one context word. Daniel Jurafsky and James H. Martin, who call the number of numbers d, write in *Speech and Language Processing*: "These d dimensions don’t have a clear interpretation." Only how the vectors compare counts. And: "It turns out that dense vectors work better in every NLP task than sparse vectors."

    ### GloVe

    **GloVe**, short for Global Vectors, came in 2014 from Jeffrey Pennington, Richard Socher and Christopher Manning. It still starts from a co-occurrence matrix of a whole corpus, as in the lesson before; in their words, GloVe "trains on global word-word co-occurrence counts". They trained it on Wikipedia of 2014 and Gigaword 5, a large collection of news, together 6 billion tokens in small letters, with a window of ten words on each side, and published vectors for 400,000 words. This lesson uses their smallest vectors, of 50 numbers per word, for the 50,000 most common words of only the letters a to z. The other famous method of those years is *word2vec*, by Mikolov and colleagues in 2013, which learns the vectors by predicting a word from the words around it, or the words around it from the word.

    The picture shows the 50 numbers of three words. Not one of them means anything on its own, but most numbers of *frog* and *toad* have the same sign.
    """)
    return


@app.cell(hide_code=True)
def idea_numbers():
    mo.Html(C.numbers_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Neighbors and analogies

    The **nearest neighbors** of a word are, as before, the words whose vectors have the highest cosine similarity with its vector. An **analogy** is a question of the form *a is to b as c is to what?*; this lesson writes it as a : b = c : ?, for example *apple : tree = grape : ?*, where the answer is *vine*. The **parallelogram model** answers it with the vectors: take the step from *a* to *b*, b − a, add it to *c*, and look for the word nearest to b − a + c. Jurafsky and Martin trace the idea back to the psychologists Rumelhart and Abrahamson in 1973, and give the example of this lesson: "For example, the result of the expression king − man + woman is a vector close to queen."
    """)
    return


@app.cell(hide_code=True)
def idea_parallelogram():
    mo.Html(C.parallelogram_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        As the guess showed, the nearest word to b − a + c is often a word of the question itself. Jurafsky and Martin, who write the question as a : b = a* : b*: "the closest value returned by the parallelogram algorithm in word2vec or GloVe embedding spaces is usually not in fact b* but one of the 3 input words or their morphological variants", "so these must be explicitly excluded". Mikolov and colleagues did so in the paper of 2013 that introduced word2vec; they write about their search for the answer: "(we discard the input question words during this search)".

        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **word embedding** | a short, dense vector of numbers for a word, learned from a large amount of text, so that words in similar contexts get similar vectors |
        | **dense** | of a vector: a few dozen to a few hundred numbers, hardly any of them 0; the opposite of sparse |
        | **GloVe** | Global Vectors (2014): word embeddings learned from the co-occurrence counts of a whole corpus |
        | **word2vec** | word embeddings learned by predicting the words around a word, or a word from the words around it (2013) |
        | **analogy** | a question of the form a : b = c : ?, *a is to b as c is to what?* |
        | **parallelogram model** | answering an analogy with the word nearest to b − a + c |
        | `E @ v` | for a table `E` and a vector `v`: the dot product of every row of `E` with `v`, one number per row |
        | `np.linalg.norm(A, axis=...)` | the norm along one axis of a table: one norm per row or one per column, depending on the axis, as with `np.sum(A, axis=...)` |
        """),
        L.watch_out("Similar vectors still mean similar company, not the same meaning: the nearest neighbors of *hot* are *cool* and *cold*. And the vectors learned what their texts say, news and Wikipedia, also about people."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Load the vectors: a list of words and a table `E` with one row per word, and a dictionary from each word to the number of its row.
    2. For the neighbors of a word, compute the cosine similarity of its row with every row, and take the highest, leaving out the word itself.
    3. For an analogy a : b = c : ?, compute v = b − a + c from their rows.
    4. Compute the cosine similarity of every row with v.
    5. Leave out a, b and c, and take the word with the highest cosine similarity.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The code below loads the vectors with the course's `glove`, which gives the 50,000 words as a list and their vectors as the table `E`, row `i` for word `i`. The course's `rows` makes the dictionary `index` from each word to the number of its row. The code shows the first five numbers of *frog*, then compares *frog* with *toad* and with *car*, as in [Embeddings: meaning as coordinates](/course/open/ml-embeddings-idea?lang=en), one pair at a time.
    """)
    return


@app.cell
def example_cell():
    import numpy as np

    words, E = C.glove()
    index = C.rows(words)
    print(len(words), E.shape)
    print(words[:8])
    frog = E[index["frog"]]
    print(np.round(frog[:5], 3))
    for other in ["toad", "car"]:
        row = E[index[other]]
        print(other, round(float(np.dot(frog, row) / (np.linalg.norm(frog) * np.linalg.norm(row))), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `E` has 50,000 rows of 50 numbers. The words come from the most common to the least common, so the list starts with *the*, *of*, *to* and *and*. The first five numbers of *frog* are 0.61, −0.208, −0.72, 0.893 and 0.325. Its cosine similarity with *toad* is 0.791, with *car* 0.042. To find the nearest neighbors of *frog*, this loop would compare it with all 50,000 rows, one at a time. Exercise 1 does them all at once.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_cosines():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. cosines: every row at once

        Write `cosines(E, v)`: give back the cosine similarity of every row of the table `E` with the vector `v`, one number per row, as one array. `E @ v` gives the dot product of every row with `v`. Divide it by the norm of each row and by the norm of `v`.

        For `cosines(np.array([[1.0, 0.0], [10.0, 1.0], [0.5, 0.5], [-1.0, 0.0]]), np.array([1.0, 0.0]))` the result is `[1, 0.995, 0.707, -1]`, rounded: the table of Exercise 2 in [Embeddings: meaning as coordinates](/course/open/ml-embeddings-idea?lang=en), now all rows in one go. Replace the `...`. The check calls that and `cosines(np.array([[3.0, 4.0], [0.0, 2.0], [-6.0, -8.0]]), np.array([2.0, 0.0]))`, then cases of its own: tables with 2 and 3 numbers per row, a table of one row, a `v` with norm 5, and two vectors of GloVe against its first 2,000 rows.
        """),
        L.task("Give back the cosine similarity of every row of E with v, as one array."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cosines():
    box_cosines = L.code_box("cosines", lines=5)
    box_cosines
    return (box_cosines,)


@app.cell(hide_code=True)
def check_cosines(box_cosines):
    result_cosines = L.check("cosines", box_cosines.value)
    result_cosines
    return (result_cosines,)


@app.cell(hide_code=True)
def hints_cosines():
    L.hints(
        "cosines",
        [
            "Three pieces: `E @ v` gives the dot product of every row with `v`, `np.linalg.norm(v)` the norm of `v`, and `np.linalg.norm(E, axis=...)` a norm for every row, with the right axis. After `return`, divide the dot products by the norms of the rows times the norm of `v`, with parentheses around that product.",
            "`E` has one row per word and one column per number. In *Sums and means along an axis*, `np.sum` added up along one axis or the other. Which axis gives one result per row, so that every word gets its own norm?",
            r"`    return E @ v / (np.linalg.norm(E, ___) * np.linalg.norm(v))`",
        ],
        why=r"`return E @ v / (np.linalg.norm(E, axis=1) * np.linalg.norm(v))`: `axis=1` runs along each row, so every row gets its own norm. In the example the four cosine similarities are 1, 0.995, 0.707 and −1.",
    )
    return


@app.cell(hide_code=True)
def text_after_cosines():
    mo.md(r"""
    In the picture, your function found the nearest neighbors of five words among all 50,000. *frog* finds *snake*, *ape*, *toad*, *monkey*, *spider* and *lizard*: animals. *coffee* finds *drink*, *drinks*, *wine*, *tea* and *beer*, and *computer* finds *computers*, *software* and *technology*. In the rows of the lesson before, *said* found *replied* and *asked*, words of the same kind; here *said* finds *told*, *says* and *asked*, but also *spokesman*, a word that goes with it in the news. The neighbors are related words of all sorts: words of the same kind, other forms of the same word, and words of the same topic.

    The vectors still keep company, not meaning. The nearest neighbors of *hot* are *cool* and *cold*, as the rows of the lesson before found. And a word gets one vector for all its meanings, learned mostly from news: the nearest neighbors of *apple* are *blackberry*, *chips*, *iphone* and *microsoft*, not a fruit among them. The lesson *One vector per word, or one per use* takes that up.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_analogy():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. analogy: a is to b as c is to what?

        Write `analogy(E, words, index, a, b, c)`: give back the answer to a : b = c : ?, by the parallelogram model. Compute `v = E[index[b]] - E[index[a]] + E[index[c]]`, and `sims`, the cosine similarity of every row with `v`, with `cosines`. The check gives your function the course's `cosines`, which does the same as yours. Then leave out the three words of the question: give each of them a number in `sims` so low that `np.argmax` passes over it. Last, give back the word with the highest cosine similarity.

        For `analogy(E, words, index, "man", "king", "woman")` the result is `"queen"`, with the vectors of GloVe. Replace the `...`. The check calls that and `analogy(E, words, index, "france", "paris", "italy")`, which should give `"rome"`, then cases of its own: three small tables of the course's own, and six more questions of GloVe.
        """),
        L.task("Give back the word nearest to b − a + c, leaving out a, b and c."),
    ])
    return


@app.cell(hide_code=True)
def exercise_analogy():
    box_analogy = L.code_box("analogy", lines=9)
    box_analogy
    return (box_analogy,)


@app.cell(hide_code=True)
def check_analogy(box_analogy):
    result_analogy = L.check("analogy", box_analogy.value)
    result_analogy
    return (result_analogy,)


@app.cell(hide_code=True)
def hints_analogy():
    L.hints(
        "analogy",
        [
            "First `v = E[index[b]] - E[index[a]] + E[index[c]]`, then `sims = cosines(E, v)`. Then `for word in [a, b, c]:`, with inside it a new number for `sims[index[word]]`, so that `np.argmax` cannot pick that word. Last, `return words[np.argmax(sims)]`.",
            "A cosine similarity lies between −1 and 1. Which number for the three words of the question makes every other word come out higher, whatever its cosine similarity?",
            r"`        sims[index[word]] = ___`",
        ],
        why=r"`sims[index[word]] = -2`, or any other number below −1: then every other word beats the three words of the question (−1 works here too, but a word pointing exactly the other way would then tie). In full: `v = E[index[b]] - E[index[a]] + E[index[c]]`, `sims = cosines(E, v)`, the loop `for word in [a, b, c]:` with `sims[index[word]] = -2` under it, and `return words[np.argmax(sims)]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_analogy():
    mo.md(r"""
    In the picture, the last column shows the words nearest to b − a + c with nothing left out. In three of the six questions, a word of the question itself comes first: *king*, *grape* and *bad*. Leaving them out, your function finds *queen*, *rome*, *vine* and *worse*. *big : bigger = small : ?* gives *larger*, with a cosine similarity of 0.904; *smaller* comes third, with 0.865, after *large*. *netherlands : amsterdam = germany : ?* gives *frankfurt*, 0.880, before *berlin*, 0.856. Mikolov and colleagues counted only the exact word as right: "synonyms are thus counted as mistakes".

    *man : doctor = woman : ?* gives *nurse*. With nothing left out, the nearest word is *doctor* itself, 0.896 against 0.842. Malvina Nissim, Rik van Noord and Rob van der Goot pointed out in 2020 that leaving out the question's words forces an answer other than *doctor*: "it is not exactly fair to claim that the embedding space is biased because it does not return doctor". That *nurse* comes right after *doctor* still shows something of what the texts say about women and men. Measuring that fairly is the subject of *Bias in word embeddings*.
    """)
    return


@app.cell(hide_code=True)
def text_groups():
    mo.md(r"""
    ### The questions of word2vec

    Mikolov and colleagues made a set of 19,544 analogy questions to measure word embeddings: "we define a comprehensive test set that contains five types of semantic questions, and nine types of syntactic questions." The fourteen groups are about meaning, such as capitals and their countries, and about grammar, such as plurals and past tenses. A question counts as right only when the answer is exactly the word meant. The course ran its analogy, the same as yours, on every question whose four words are among the 50,000 words of GloVe: 17,692 questions.

    ### Exercise 3. Which groups work?

    Each line names a group of the questions of word2vec, with the first pair of the group. Put it under the heading that says how often the course's analogy answers the questions of that group right with the vectors of GloVe.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_groups = L.sort("groups")
    sort_groups
    return (sort_groups,)


@app.cell(hide_code=True)
def check_sort(sort_groups):
    result_sort = L.check("groups", sort_groups)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_after_sort():
    mo.md(r"""
    The picture shows all fourteen groups, from the best to the worst.
    """)
    return


@app.cell(hide_code=True)
def groups_picture():
    mo.Html(C.groups_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_after_groups():
    mo.md(r"""
    Of all 17,692 questions, the course's analogy answers 47.9% right. Pennington and colleagues report 71.7% for their vectors of 300 numbers from the same corpus; the vectors of this lesson have only 50. With nothing left out, only 32.5% is right: in 46.1% of the questions, the nearest word is *b* or *c* itself, and *a* never. Tal Linzen noted in 2016 that leaving out the three words is itself a shortcut built into the solver: "However, such heuristics that are baked into an analogy solver, while likely to improve its performance, call into question the interpretation of the success of the analogy solver as evidence for the geometric organization of the underlying semantic space."

    Linzen also proposed a check, which the gray bars show: take the nearest neighbor of *c* alone, and use *a* and *b* only to leave them out. That alone answers 14.8% of the questions right, and for the -ing forms, such as *code : coding = dance : ?*, 40.6%, against 41.3% for the parallelogram. *dancing* is simply the nearest neighbor of *dance*; for this question, the parallelogram even answers *electronica*. For the -ing forms, the step from *a* to *b* adds almost nothing.

    The groups that work have common words and a relation that the texts state over and over: countries and their nationalities, capitals, *boy* and *girl*. Anna Rogers, Aleksandr Drozd and Bofang Li wrote in 2017: "Most models score well on morphological inflections – because morphological forms of the same word are highly distributionally similar and are likely to be close." Names of currencies are rare words. And Rogers and colleagues found that "the offsets of different word vector pairs appear to not be so regular, even for relatively straightforward linguistic relations". In the vectors of this lesson too, the steps of different pairs point in rather different directions for words and their opposites, and for adjectives and their adverbs with *-ly*, much more than for countries and their nationalities. Jurafsky and Martin sum it up: "embedding spaces perform well if the task involves frequent words, small distances, and certain relations (like relating countries with their capitals or verbs/nouns with their inflected forms)". On a harder set of 99,200 questions in 40 groups, Anna Gladkova, Aleksandr Drozd and Satoshi Matsuoka found in 2016 that GloVe answered 28.5% right on average: "Our experiments show that derivational and lexicographic relations remain a major challenge."
    """)
    return


@app.cell(hide_code=True)
def text_dutch():
    mo.md(r"""
    ### The same in Dutch

    For Dutch the course uses the vectors of *fastText*, which Edouard Grave and colleagues published in 2018 for 157 languages, learned from Common Crawl, a large collection of web pages, and from Wikipedia: 300 numbers per word, with a window of five words. fastText builds the vector of a word from pieces of five letters, so words with the same pieces get related vectors. The file for Dutch lists two million words, from the most to the least common, and is 1.3 GB. The course downloads only its first 20 MB: 30,355 complete lines, of which it keeps the 28,209 words of letters only. fastText keeps capitals, so *koning* and *Koning* are two words with two vectors. The Dutch questions are the course's own: 185 word pairs in ten of the groups of word2vec. Every two different pairs of a group, in both orders, make a question, as in most groups of word2vec: 3,584 questions.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download the Dutch vectors, 20 MB")
    return (get_button,)


@app.cell(hide_code=True)
def dutch_cell(get_button):
    C.download_or_stop(get_button, {
        "ask": "*The Dutch vectors of fastText are not on your computer yet. The button downloads the first 20 MB of the file once from the site of fastText, under the license CC BY-SA 3.0. The rest of the lesson works without them.*",
        "busy": "**Downloading** {done} of {total} MB …",
        "no_net": "*The download did not work: the site of fastText could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: the file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is no room on your disk for 20 MB. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
    })
    mo.Html(C.dutch_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_after_dutch():
    mo.md(r"""
    The Dutch neighbors look much like the English ones: *kikker* finds *kikkers*, *schildpad* and *hagedis*, *koffie* finds *thee*, and *zei* finds *antwoordde*, *zegt* and *sprak*. Of the 3,584 Dutch questions, 70.2% are answered right with the question's words left out. With nothing left out, 24.2% is: in 72.9% of the questions, the nearest word is *b* or *c* itself. *man : koning = vrouw : ?* gives *koningen*, the plural, before *koningin*; *Frankrijk : Parijs = Italië : ?* gives *Milaan* before *Rome*. Words and their opposites work worst here too, 21.4%; Stéphan Tulkens, Chris Emmery and Walter Daelemans also found opposites, and currencies, consistently hard for their Dutch word embeddings in 2016.
    """)
    return


@app.cell(hide_code=True)
def text_explore():
    mo.md(r"""
    ### Try it: any word, or any three

    Type one word to see its eight nearest neighbors, or three words, a b c, for the analogy a : b = c : ?. The picture shows the answer leaving out the three words, and the nearest words with nothing left out. Try *slow slower fast*, *japan tokyo france* or *france french germany*, then words of your own. Choose fastText for Dutch, once it is downloaded; there, capitals count, as in *Frankrijk Parijs Duitsland*.
    """)
    return


@app.cell(hide_code=True)
def explore_controls():
    explore_text = mo.ui.text(value="frog", label="Word, or three words")
    explore_kind = mo.ui.dropdown(options={"GloVe (English)": "glove", "fastText (Dutch)": "fasttext"},
                                  value="GloVe (English)", label="Vectors")
    mo.hstack([explore_text, explore_kind], justify="start", gap=1.5)
    return explore_kind, explore_text


@app.cell(hide_code=True)
def explore_look(explore_kind, explore_text):
    mo.Html(C.explore_picture(explore_text.value, explore_kind.value, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [A word by the company it keeps](/course/open/nlp-distributional-hypothesis?lang=en) counted the company of every word in long, sparse rows. GloVe learns short, dense word embeddings from such counts over billions of words, and you compared them all at once with the cosine similarity of [Embeddings: meaning as coordinates](/course/open/ml-embeddings-idea?lang=en). The nearest neighbors are related words of all sorts. The parallelogram answers about half of word2vec's questions, but only with the three words of the question left out, and best for common words and relations close by. The next lesson of the domain, *One vector per word, or one per use*, asks what to do with a word such as *bank*, which gets one vector for all its meanings.

    **An everyday example.** A shop's search box that also finds *sneakers* when you type *trainers* can use word embeddings: the two words keep the same company in millions of texts.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_analogy, result_cosines, result_sort):
    L.wrap_up(result_cosines, result_analogy, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
