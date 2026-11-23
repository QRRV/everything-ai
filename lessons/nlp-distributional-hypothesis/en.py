# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-distributional-hypothesis. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-distributional-hypothesis", "en")
    C = L.checks   # the course's words, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [A document as a vector](/course/open/nlp-document-vectors?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_dot():
    quick_dot = L.choice(
        "quick-dot",
        question="Two sparse rows hold counts by word: `{\"cup\": 2, \"tea\": 1}` and `{\"cup\": 3, \"milk\": 4}`. What is their dot product?",
        options={
            "pairs": "10",
            "right": "6",
            "add": "5",
        },
        correct="right",
        why={
            "right": "Only *cup* is in both rows, so only *cup* adds to the dot product: 2 × 3 = 6. *tea* and *milk* each have a 0 in the other row. This lesson compares rows like these, with words as their keys.",
            "pairs": "10 pairs the first count with the first and the second with the second: 2 × 3 + 1 × 4. A sparse row pairs counts by their key, and *tea* and *milk* are different keys. See [A document as a vector](/course/open/nlp-document-vectors?lang=en).",
            "add": "5 adds the two counts of *cup*; the dot product multiplies them. See [A document as a vector](/course/open/nlp-document-vectors?lang=en).",
        },
    )
    quick_dot
    return (quick_dot,)


@app.cell(hide_code=True)
def quick_check_common():
    quick_common = L.choice(
        "quick-common",
        question="In that lesson, the held-out sentence *The delivery was on time.* found its most similar training sentence by the cosine similarity of plain counts. Which words did the two sentences share?",
        options={
            "topic": "delivery, time",
            "right": "the, was, on",
            "tone": "great, rude",
        },
        correct="right",
        why={
            "right": "Its most similar sentence was *The bus boy on the other hand was so rude.*: not one word about delivery or time. Short, common words occur in so many sentences that they decide much of the cosine similarity of plain counts. In this lesson every word near a word counts, *the* too: watch what that does to the rows.",
            "topic": "The sentence about the same thing, *The shipping time was also very quick!*, came only third. See [A document as a vector](/course/open/nlp-document-vectors?lang=en).",
            "tone": "*The delivery was on time.* has no word of praise or complaint in it. See [A document as a vector](/course/open/nlp-document-vectors?lang=en).",
        },
    )
    quick_common
    return (quick_common,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A dictionary explains a word with other words. A computer has no dictionary in its head, but it can count. This lesson gives every word a list of numbers, a vector, by counting which words stand near it in a few books, and words with similar meanings turn out to get similar vectors. The idea is older than the computers that could test it. In 1954 the linguist Zellig Harris wrote in *Distributional Structure*: "In other words, difference of meaning correlates with difference of distribution." By the distribution of a word, Harris meant "the sum of all its environments": every place where it occurs, with the words around it. The example Harris gave: *oculist* and *eye-doctor*, two words for a doctor of the eyes, occur in almost the same environments, *oculist* and *lawyer* in only some of the same, and that second pair differs in meaning, "the amount of meaning difference corresponding roughly to the amount of difference in their environments."

    In 1957 the linguist J. R. Firth wrote a sentence that is quoted to this day: "You shall know a word by the company it keeps!" Firth meant something narrower than this lesson: the words that habitually go with a word, its collocations, as *silly* goes with *ass* in *You silly ass!*. For Firth, that company was one of the meanings of a word. The textbook *Speech and Language Processing* by Daniel Jurafsky and James H. Martin puts the idea and its name like this: "Words that occur in similar contexts tend to have similar meanings. This link between similarity in how words are distributed and similarity in what they mean is called the distributional hypothesis." The word embeddings of later lessons, and the language models built on them, rest on the same idea.

    **An everyday example.** On a train, two people talk about a padel court, rackets, a match on Sunday, and a partner who serves too hard. You have never played padel, but you already know that it is a game played with rackets, by pairs.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In the five English books of the course, every word that occurs at least 20 times and is not a stop word gets a row: how often each word stands within two places of it. Which word has the row most like the row of *hot*?",
        options={
            "warm": "warm",
            "fire": "fire",
            "right": "cold",
            "summer": "summer",
        },
        reveal="**cold**: of the 2,423 other words with a row, *cold* has the row most like that of *hot*, with a cosine similarity of 0.767. *warm* comes 92nd, *summer* 370th and *fire* 444th. *hot* and *cold* mean the opposite, but they keep the same company: the words they share most are *and*, *a*, *of*, *was*, *the* and *in*, and both stand near *supper*. A row records where a word stands, not what it says.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Context and window

    The **context** of a word is the text around it. This lesson takes a small piece of it: the **window**, the words within a fixed number of places before and after the word. The number is the size of the window. With a window of size 2, the window of *tea* in *we drink tea with milk* holds *we*, *drink*, *with* and *milk*. Jurafsky and Martin take a larger one in their example: "a context window around the word, let's say of 4 words to the left and 4 words to the right". The word whose window you take is the **target word**, and the words in its window are its **context words**. A target word and a word in its window **co-occur**.

    ### The co-occurrence matrix

    Count, over a whole text, how often each word stands in a window of each target word, and you get a table with one row per target word and one column per context word: the **co-occurrence matrix**. Jurafsky and Martin describe one kind of it: "each row in the matrix represents a word in the vocabulary and each column represents how often each other word in the vocabulary appears nearby." Each row is a vector for its word. As with the document vectors of [A document as a vector](/course/open/nlp-document-vectors?lang=en), almost every count is 0, so this lesson stores each row as a sparse row: a dictionary from context word to count, with the context word as the key of its column. The picture shows the matrix of four target words in six short sentences of the course's own.
    """)
    return


@app.cell(hide_code=True)
def idea_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Similar company, similar rows

    *tea* and *coffee* stand between the same words: *we drink … with*, *a cup of …*. Their rows share four of their six columns, and the cosine similarity of the two rows is 0.67. *tea* and *car* share no column, so theirs is 0. Over a few books, the rows get thousands of columns, and the **nearest neighbors** of a word are the words whose rows have the highest cosine similarity with its row. Describing what words mean by the contexts they occur in, as vectors like these, is **distributional semantics**.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **distributional hypothesis** | the idea that words that occur in similar contexts tend to have similar meanings |
        | **distributional semantics** | describing what words mean by the contexts they occur in, usually as vectors |
        | **context** | the text around a word |
        | **window** | the words within a fixed number of places before and after a word; with size 2, two words on each side |
        | **target word** | a word whose windows are counted; it gets a row |
        | **context word** | a word in a window of a target word; it gets a column |
        | **co-occur** | of a target word and a context word: one stands in a window of the other |
        | **co-occurrence matrix** | a table with a row per target word and a column per context word, holding how often the two co-occur |
        | **nearest neighbors** | the words whose rows have the highest cosine similarity with the row of a word |
        | `max(a, b)` | the larger of two numbers: `max(3, 5)` gives 5; `min(a, b)` gives the smaller |
        | `a + b` | for two lists: one list with first the items of `a`, then the items of `b`: `["x"] + ["y", "z"]` gives `["x", "y", "z"]` |
        """),
        L.watch_out("Similar rows mean that two words keep the same company, not that they mean the same. Words that stand in the same places get similar rows, whatever they mean."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Cut the text into words, in small letters.
    2. Choose the target words: here every word that occurs at least 20 times and is not a stop word.
    3. Go over the places of the text. At a target word, take its window: the words within `size` places before it and after it.
    4. In the row of that target word, add 1 to the count of every word in the window.
    5. Compare two rows with the cosine similarity. The nearest neighbors of a word are the words whose rows have the highest cosine similarity with its row.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The code below takes the six sentences of the picture, from the course's `toy_sentences`, and the four target words, from the course's `toy_targets`. For each place where *tea* stands, it shows the place, the two words before it and the two words after it. Then it shows the row of *tea* from the course's `toy_rows`, and compares it with two other rows with the course's `cosine`, which does the same as `sparse_cosine` of [A document as a vector](/course/open/nlp-document-vectors?lang=en).
    """)
    return


@app.cell
def example_cell():
    sentences = C.toy_sentences(L.lang)
    small_targets = C.toy_targets(L.lang)   # tea, coffee, car, bike
    first = small_targets[0]
    for sentence in sentences:
        for i in range(len(sentence)):
            if sentence[i] == first:
                print(i, sentence[i - 2:i], sentence[i + 1:i + 3])
    toy = C.toy_rows(L.lang)
    print(toy[first])
    print(C.cosine(toy[first], toy[small_targets[1]]), C.cosine(toy[first], toy[small_targets[2]]))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    *tea* stands at place 2 of the first sentence and at place 4 of the third. There, `sentence[5:7]` is empty: a slice that runs past the end of a list stops at the end. The row of *tea* adds up both windows: six words, each once. Its cosine similarity with the row of *coffee* is 0.667, and with the row of *car* 0.0. In this example *tea* never stands at place 0 or 1, so `i - 2` is never below 0. Exercise 1 makes a window for any place and any size.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_window():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. window: the words around a place

        Write `window(words, i, size)`: give back the words within `size` places of place `i` of the list `words`, without the word at place `i` itself, in their order: first the words before it, then the words after it. Near the start or the end of the list, the window holds fewer words.

        For `window(["a", "b", "c", "d", "e", "f", "g"], 3, 2)` the result is `["b", "c", "e", "f"]`: *d* stands at place 3, with *b* and *c* before it and *e* and *f* after it. Replace the `...`. The check calls that and `window(["a", "b", "c", "d", "e", "f", "g"], 0, 2)`, then cases of its own: place 1, the last place, a window that starts at place 0 exactly, sizes 1, 0 and 3, a window wider than the list on both sides, a list of one word, a word that comes twice, a list of twelve words, and the first words of two books. Watch out near the start: in a slice, a negative number counts from the end of the list, so `words[-1:1]` is empty.
        """),
        L.task("Give back the words within size places of place i, without the word at place i itself."),
    ])
    return


@app.cell(hide_code=True)
def exercise_window():
    box_window = L.code_box("window", lines=4)
    box_window
    return (box_window,)


@app.cell(hide_code=True)
def check_window(box_window):
    result_window = L.check("window", box_window.value)
    result_window
    return (result_window,)


@app.cell(hide_code=True)
def hints_window():
    L.hints(
        "window",
        [
            "Take two slices of `words` and put them together with `+`: the slice of the words before place `i` ends at `i`, and the slice of the words after it starts at `i + 1`. Each holds at most `size` words. Give back the two together, after `return`.",
            "With `i` as 1 and `size` as 2, `i - size` is −1, and `words[-1:1]` is empty, while the window should hold the word at place 0. At which place should the first slice start, for any `i` and `size`?",
            r"`    return words[___:i] + words[i + 1:i + size + 1]`",
        ],
        why=r"`return words[max(0, i - size):i] + words[i + 1:i + size + 1]`: `max(0, i - size)` is `i - size`, or 0 when that is below 0. A slice that runs past the end of the list stops at the end.",
    )
    return


@app.cell(hide_code=True)
def text_after_window():
    mo.md(r"""
    In the picture, your function took the window at four places of the first words of *Pride and Prejudice*, the novel by Jane Austen from 1813: *it is a truth universally acknowledged that a single man*. At place 0 the window holds only *is* and *a*, at place 9 only *a* and *single*. With a window of size 2, every word of the books gets four context words, except the first two and the last two.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rows():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. cooccurrence: a row for every target word

        `words` holds every word of the five English books of the course, in small letters, one book after another, from the course's `book_words`. `targets` holds their target words, from the course's `targets`: the 2,424 words that occur at least 20 times and are not stop words, as a set. Write `cooccurrence(words, size, targets)`: give back a dictionary with a row for every target word. A row is a sparse row: a dictionary from each word that stands in a window of the target to how often it stands there. Go over the places of `words`; at a target word, add 1 to the count of every word in its window, in the row of that target. The start code already makes an empty row for every target, so a target that does not occur keeps an empty row. The check gives your function the course's `window`, which does the same as yours.

        For `cooccurrence(["a", "b", "c", "a", "b"], 2, ["a", "c"])` the result is `{"a": {"b": 3, "c": 2}, "c": {"a": 2, "b": 2}}`: the first *a* has *b* and *c* in its window, the second *a* has *b*, *c* and *b*, and *c* has *a*, *b*, *a* and *b*. Replace the `...`. The check calls that and `cooccurrence(["a", "b", "a"], 1, ["a", "z"])`, then cases of its own: a word next to itself, no words, no targets, sizes 1 and 0, the targets as a set, a word twice with a window wider than the list, the first words of two books, and the first 500 words of *Alice's Adventures in Wonderland*.
        """),
        L.task("Give every target word a sparse row: how often each word stands in its windows."),
    ])
    return


@app.cell
def books_cell():
    window = C.window
    words = C.book_words(L.lang)
    targets = C.targets(L.lang)
    print(len(words), len(targets))
    print(words[:12])
    print(sorted(targets)[:8])
    return


@app.cell(hide_code=True)
def text_after_books():
    mo.md(r"""
    The books have 551,664 words, and they start with the title page of *Alice's Adventures in Wonderland*; its first word, *illustration*, is where Project Gutenberg notes a picture. The words are runs of letters, and an apostrophe inside a word stays, so *alice’s* is one word. The most common target words are *one*, *said* and *now*.
    """)
    return


@app.cell(hide_code=True)
def exercise_rows():
    box_rows = L.code_box("cooccurrence", lines=13)
    box_rows
    return (box_rows,)


@app.cell(hide_code=True)
def check_rows(box_rows):
    result_rows = L.check("cooccurrence", box_rows.value)
    result_rows
    return (result_rows,)


@app.cell(hide_code=True)
def hints_rows():
    L.hints(
        "cooccurrence",
        [
            "Go over the places with `for i in range(len(words)):` and take `word = words[i]`. When `word in targets`, go over the words of its window with `for context in window(words, i, size):`. When `context in rows[word]`, add 1 to its count in that row; otherwise start it at 1. After the loop, give back `rows`.",
            "In `cooccurrence([\"a\", \"b\", \"a\"], 1, [\"a\", \"z\"])`, *b* stands in the windows of both *a*'s. When the second *a* comes, the count of *b* in the row of *a* is 1 already: what should it become, and how do you write that for any `word` and `context`?",
            r"`                    rows[word][context] = ___`",
        ],
        why=r"Keep the start code, which makes an empty row for every target. Then `for i in range(len(words)):` with `word = words[i]`, and `if word in targets:` with `for context in window(words, i, size):` under it; there `if context in rows[word]:` with `rows[word][context] = rows[word][context] + 1`, and `else:` with `rows[word][context] = 1`. After the loop, `return rows`.",
    )
    return


@app.cell(hide_code=True)
def text_after_rows():
    mo.md(r"""
    Your function counted, for each of the 2,424 target words, every word within two places of it: 337,034 counts that are not 0. Look at the neighbors in the picture. *said* finds *replied*, *asked*, *continued*, *cried*, *answered* and *added*: all words for what someone says. *eyes* finds *face*, *hands*, *arms* and *lips*, *ran* finds *walked*, and *three* finds *two* and *four*. Nobody told the rows what these words mean; the words stand in the same places. Some neighbors are odd. *father* finds *friend*, *cousin* and *companion* first, not *mother*: *my* is the biggest count in all four rows, as in *my father* and *my friend*. *ran* finds *though* second.

    The biggest counts in the row of *door* are *the*, 246 times, then *and*, *at*, *of*, *a* and *was*. The same few common words fill every row, and so every cosine similarity in the picture is above 0.75.
    """)
    return


@app.cell(hide_code=True)
def text_explore():
    mo.md(r"""
    ### Try it: another word, another window

    Type a word, and choose the size of the window. The picture shows how often the word occurs in the books, the biggest counts in its row, and its eight nearest neighbors. Try *door* with each window size: from which size on does *open* stand among its eight nearest neighbors? Then try *said*, *sea*, or a word of your own.
    """)
    return


@app.cell(hide_code=True)
def explore_controls():
    explore_word = mo.ui.text(value=C.EXPLORE_WORD[L.lang], label="Word")
    explore_size = mo.ui.dropdown(options=["1", "2", "5", "10"], value="2", label="Window size")
    mo.hstack([explore_word, explore_size], justify="start", gap=1.5)
    return explore_size, explore_word


@app.cell(hide_code=True)
def explore_look(explore_size, explore_word):
    mo.Html(C.explore_picture(explore_word.value, explore_size.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_explore():
    mo.md(r"""
    With a window of 2, the nearest neighbors of *door* are *window*, *sea*, *sun*, *cottage* and *ship*: nouns that, like *door*, keep company with *the* most of all. *open* stands in the row of *door* 23 times, but comes 213th among its neighbors: *open* stands next to *door*, not in the same places. With a window of 5 it comes 12th, and with a window of 10 third, after *window* and *room*, with *entered* seventh: words that go with *door* come closer. Jurafsky and Martin describe the same: "Shorter context windows tend to lead to representations that are a bit more syntactic, since the information is coming from immediately nearby words." And: "When vectors are computed from long context windows, the highest cosine words to a target word w tend to be words that are topically related but not similar." The size, they write, "is generally between 1 and 10 words on each side of the target word". With a window of 10, the rows fill up with even more common words: the eight nearest neighbors of *door* all lie between 0.958 and 0.971.

    ### What the rows miss

    Words with opposite meanings keep the same company. Of all the rows, the one most like the row of *hot* is the row of *cold*; for *young* it is the row of *old*, and for *day* the row of *night*. An antonym is a word with the opposite meaning. Peter Turney and Patrick Pantel, in an overview from 2010: "We might not usually think that antonyms are similar, but antonyms have a high degree of attributional similarity (hot and cold are kinds of temperature, black and white are kinds of colour, loud and quiet are kinds of sound)." Nikola Mrkšić and colleagues point out in 2016 that even better methods generally fail to tell synonyms, words with the same meaning, from antonyms: "For example, words such as east and west or expensive and inexpensive appear in near-identical contexts, which means that distributional models produce very similar word vectors for such words."

    Words can also be related without being similar. Jurafsky and Martin: "Coffee is not similar to cup; they share practically no features". And: "But coffee and cup are clearly related". A short window tends to find words of the same kind, a long one words of the same topic, as *door* and *open* showed.

    The rows know only these books. The nearest neighbor of *white* is *sperm*, from the sperm whales of *Moby-Dick*, a novel of 216,543 words, the longest of the five.

    And plain counts let the most common words decide. Pick 3,000 pairs of target words at random, and more than half of them have a cosine similarity above 0.45: *the*, *and* and *of* give even unrelated words much of their rows in common. Turney and Pantel describe the way out: "The idea of weighting is to give more weight to surprising events and less weight to expected events." Their example: to tell how alike *mouse* and *rat* are, "the contexts dissect and exterminate are more discriminative of their similarity than the contexts have and like."
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Exercise 3. Near or far?

    Each line names two words from the English books. Put it under the heading that says where the first word stands among the neighbors of the second, with the rows of this lesson: a window of 2 and plain counts.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_neighbors = L.sort("neighbors")
    sort_neighbors
    return (sort_neighbors,)


@app.cell(hide_code=True)
def check_sort(sort_neighbors):
    result_sort = L.check("neighbors", sort_neighbors)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_after_sort():
    mo.md(r"""
    Jurafsky and Martin give the two kinds of company a name: "Two words have first-order co-occurrence (sometimes called syntagmatic association) if they are typically nearby each other. Thus wrote is a first-order associate of book or poem. Two words have second-order co-occurrence (sometimes called paradigmatic association) if they have similar neighbors. Thus wrote is a second-order associate of words like said or remarked." The rows of this lesson find the second kind: words with similar neighbors.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [A document as a vector](/course/open/nlp-document-vectors?lang=en) gave every sentence a sparse row of word counts and compared two rows with the cosine similarity. This lesson gave every word such a row, with the words around it as its columns: a co-occurrence matrix, built on the distributional hypothesis. Words of the same kind get similar rows, and so do words of opposite meaning; plain counts let *the*, *and* and *of* decide too much. The next lesson of the domain, *Word embeddings: neighbors and analogies*, works with word embeddings: short vectors of a few dozen to a few hundred numbers, learned from far more text.

    **An everyday example.** A search engine that also finds pages about a *sofa* when you type *couch* can learn from text alone that the two words keep the same company.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_rows, result_sort, result_window):
    L.wrap_up(result_window, result_rows, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
