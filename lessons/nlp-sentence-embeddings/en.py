# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-sentence-embeddings. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-sentence-embeddings", "en")
    C = L.checks   # GloVe, the course's pairs of sentences with BERT-Tiny's and Granite's vectors, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions, about [One vector per word, or one per use](/course/open/nlp-static-vs-contextual?lang=en) and [Attention: every word looks at the others](/course/open/attn-idea?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_contextual():
    quick_contextual = L.choice(
        "quick-contextual",
        question="In *One vector per word, or one per use*, BERT-Tiny read the course's 44 sentences with *bank*. After its second block, how many different vectors did *bank* get?",
        options={
            "one": "1, the same in every sentence",
            "two": "2, one per meaning",
            "right": "44, one per sentence",
            "numbers": "128, one per number",
        },
        correct="right",
        why={
            "right": "A contextual model gives every occurrence its own vector, built from the words around it: 44 sentences, 44 vectors. This lesson goes one step further, to one vector for a whole sentence.",
            "one": "A static embedding such as GloVe does that: one vector for *bank* in every sentence. BERT-Tiny gives every occurrence its own. See [One vector per word, or one per use](/course/open/nlp-static-vs-contextual?lang=en).",
            "two": "BERT-Tiny does not know the meanings in advance: it gives every occurrence its own vector, built from the words around it, 44 in all. The vectors of the river sentences lie closer to each other than to those of the money sentences, but each is a vector of its own.",
            "numbers": "Each vector has 128 numbers, but that is its length, not how many vectors there are: each of the 44 occurrences gets a vector of its own.",
        },
    )
    quick_contextual
    return (quick_contextual,)


@app.cell(hide_code=True)
def quick_check_attention():
    quick_attention = L.choice(
        "quick-attention",
        question="In *Attention: every word looks at the others*, what does attention give each position?",
        options={
            "own": "Its own vector, unchanged",
            "right": "A weighted average of the vectors of all positions",
            "before": "The sum of the vectors of the positions before it, in order",
            "max": "The largest number of each column",
        },
        correct="right",
        why={
            "right": "The weights depend on how well two positions match, and they add up to 1. So every vector after a block holds something of the whole sentence. This lesson takes the mean of those vectors once more, for one vector per sentence.",
            "own": "Then a block would change nothing. Attention mixes in the vectors of the other positions, each with a weight. See [Attention: every word looks at the others](/course/open/attn-idea?lang=en).",
            "before": "Attention takes a weighted average, with weights that add up to 1, and not a sum. In that lesson, every word looked at all the others.",
            "max": "Attention does not pick one number: it takes a weighted average of the vectors of all positions, with weights that depend on how well they match.",
        },
    )
    quick_attention
    return (quick_attention,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Search by meaning compares texts, and a question and the text that answers it can share few words. With one vector per sentence, a computer compares two texts with one cosine similarity, as it compared two words in [Word embeddings: neighbors and analogies](/course/open/nlp-word-embeddings?lang=en). Nils Reimers and Iryna Gurevych, who made Sentence-BERT, put a number on it in 2019. BERT reads two sentences together and then scores the pair, so finding the most similar pair among 10,000 sentences takes 49,995,000 runs of BERT: about 65 hours on a V100, a fast graphics card of that time. With one vector for each sentence, it takes about 5 seconds to compute the 10,000 vectors, and about 0.01 seconds to compare them.

    The simplest vector for a sentence is the mean of the vectors of its words. This lesson shows what such a mean loses, and what a model trained for sentences keeps.

    **An everyday example.** A smoothie of a banana and a strawberry tastes the same whichever fruit went into the blender first. A mean of word vectors is a smoothie of the words.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_one = L.guess(
        "guess",
        question="Granite Embedding is a model trained to give each sentence one vector. Which sentence does it put closer to *the film was good*: *i enjoyed the movie*, the same meaning in other words, or *the film was not good*, the opposite meaning?",
        options={
            "far": "*i enjoyed the movie*, by far",
            "little": "*i enjoyed the movie*, by a little",
            "same": "Both the same",
            "right": "*the film was not good*",
        },
        reveal="**The sentence with *not*:** *the film was not good* gets a cosine similarity of 0.925 with *the film was good*, and *i enjoyed the movie* 0.830. The pair with four words in common comes out closer, though *not* turns the meaning around. The idea shows where that comes from, and its watch-out comes back to *not*.",
    )
    guess_one
    return (guess_one,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### One vector for a sentence

    A **sentence embedding** is one vector for a whole sentence: as many numbers for a sentence of three words as for one of thirty, so that any two sentences can be compared with their cosine similarity, as in [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en). Sentences with the same meaning should get vectors that point nearly the same way. This lesson takes three roads to it: the mean of the word vectors, a transformer with pooling, and a transformer trained on pairs of sentences.

    ### The mean of the word vectors

    Take the GloVe vector of every word of the sentence, and take their mean: for each of the 50 places, add up the numbers of all words and divide by the number of words. *the dog bit the man* and *the man bit the dog* have the same five words, so their means are the same, number for number, and their cosine similarity is 1: the mean has lost the order of the words. A word that GloVe lacks is left out of the mean; a sentence without a single word in GloVe has no mean.
    """)
    return


@app.cell(hide_code=True)
def idea_mean():
    mo.Html(C.idea_mean_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_pairs():
    mo.md(r"""
    *not* is one word of five in *the film was not good*, and its vector shifts the mean only a little: the cosine similarity with *the film was good* is 0.990. *i enjoyed the movie*, the same meaning in other words, gets 0.949. Words such as *the* and *was* count as much as any other word in a mean, and most sentences have such words: even *the train leaves at nine*, about something else, gets 0.813.
    """)
    return


@app.cell(hide_code=True)
def idea_pairs():
    mo.Html(C.idea_pairs_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_pooling():
    mo.md(r"""
    ### A transformer and pooling

    A transformer such as BERT-Tiny gives every piece of a sentence its own vector, built with attention from all the other pieces, as in *One vector per word, or one per use*. For one vector per sentence, a model combines those vectors into one: **pooling**. **Mean pooling** takes their mean; **CLS pooling** takes only the vector of *[CLS]*, the piece before the sentence.

    A model reads sentences in a **batch**: a group of sentences in one array, as a model learned from a few examples at a time in [Mini-batches and epochs](/course/open/torch-batches?lang=en). In that array every sentence needs as many places, so a shorter sentence is filled up with the piece *[PAD]* to the length of the longest. A **mask** says which places hold a real piece: 1 for *[CLS]*, the pieces of the sentence and *[SEP]*, 0 for *[PAD]*. The mask keeps the *[PAD]* places out of attention, so they do not change the real pieces. They still get a vector of their own, and those vectors have to stay out of the mean. The course sent its 32 sentences through BERT-Tiny in one batch: every sentence has 10 places, of which 6 to 10 are real, so BERT-Tiny gives each sentence a table of 10 rows.

    Reimers and Gurevych tried both ways on BERT itself: "The most commonly used approach is to average the BERT output layer (known as BERT embeddings) or by using the output of the first token (the [CLS] token). As we will show, this common practice yields rather bad sentence embeddings, often worse than averaging GloVe embeddings (Pennington et al., 2014)." They used seven sets of pairs of sentences that people had scored for how alike they are. A score of 100 means that the cosine similarities put the pairs in exactly the order the people did. The mean of BERT's vectors got 54.81 on average, CLS pooling 29.19, and the mean of the GloVe vectors 61.32.

    ### Trained on pairs

    What helps is training on pairs of sentences. Sentence-BERT sends each sentence of a pair through the same BERT, with the same weights, one sentence at a time, pools the vectors of each, and learns from the pair. Reimers and Gurevych trained it on **NLI**, natural language inference: pairs of sentences that people labeled with whether the second follows from the first, contradicts it, or neither, 570,000 pairs in SNLI and 430,000 in MultiNLI. After one pass over those pairs, mean pooling of the trained model got 74.89 on the same seven sets. In that training, the model learned to put each pair under one of the three labels, and then the way of pooling mattered little; in their words: "When trained with the classification objective function on NLI data, the pooling strategy has a rather minor impact." Their work grew into **Sentence Transformers**, a Python library to compute and train sentence embeddings, with a family of models.

    Granite Embedding, the course's embedding model, also learned from pairs, with **contrastive training**. The team at IBM that made it describes a training that "brings the embeddings of a query closer to those of relevant passages, and pushes them further away from non-relevant ones." Among its pairs are titles with their texts, questions with their answers, and questions that ask the same thing. Granite uses CLS pooling, "which we find to perform better than mean-pooling strategy commonly used in other works," in the words of the same team. It gives each sentence 384 numbers, in 12 languages, Dutch among them.
    """)
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **sentence embedding** | one vector for a whole sentence, as long for every sentence, so that sentences can be compared with their cosine similarity |
        | **pooling** | combining the vectors of the pieces of a sentence into one vector |
        | **mean pooling** | pooling by the mean of the vectors of the real pieces |
        | **CLS pooling** | pooling by the vector of *[CLS]* alone |
        | **batch** | a group of sentences that a model reads together, in one array |
        | ***[PAD]*** | the piece that fills a shorter sentence up to the length of the longest in a batch |
        | **mask** | 1 for every real piece of a sentence in a batch, 0 for every *[PAD]* place |
        | **NLI** | natural language inference: pairs of sentences labeled with whether the second follows from the first, contradicts it, or neither |
        | **contrastive training** | training that pulls the vectors of texts that belong together closer, and pushes the vectors of other texts away |
        | **Sentence Transformers** | a Python library, with a family of models, to compute and train sentence embeddings |
        | `np.zeros(n)` | an array of `n` zeros: `np.zeros(3)` gives `array([0., 0., 0.])` |
        | `E.shape[1]` | the number of columns of the table `E`: 50 for GloVe |
        | `tokens * mask` | with `mask` a column of one number per row: every row of `tokens` times the number of `mask` in that row |
        """),
        L.watch_out("Training on ordinary pairs does not solve *not*. Granite puts *the film was not good* closer to *the film was good*, 0.925, than *i enjoyed the movie*, 0.830. Miriam Anschütz, Diego Miguel Lozano and Georg Groh, who measured in 2023 how several models handle negation, write: \"Large language models underestimate the impact of negations on how much they change the meaning of a sentence.\" They trained a sentence model further on pairs of sentences with and without a negation, and it did better on them."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Cut the sentence into words, or into the pieces of a model.
    2. For the mean of the word vectors: add up the vectors of the words that the table has, and divide by how many there are. Without such a word, the sentence has no mean.
    3. For a transformer: send the sentences in a batch, filled up with *[PAD]*, with their mask. For mean pooling, take the mean of the rows where the mask is 1; for CLS pooling, take the row of *[CLS]*.
    4. Compare two sentence embeddings with their cosine similarity. A model trained on pairs puts sentences with the same meaning closer than a mean does; try *not* on your own sentences before you trust it.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The code takes GloVe, as in *Word embeddings: neighbors and analogies*: `C.glove()` gives the list `words` and the table `E`, with a vector of 50 numbers for each of the 50,000 words, and `C.rows(words)` gives the dict `row` from each word to its row. `mean_of` takes the mean of the vectors of the words of a sentence, along axis 0, as in [Sums and means along an axis](/course/open/np-axis?lang=en): one number for each of the 50 places. `cosine` is the cosine similarity of two vectors. The code prints the length of the mean of *the film was good*, and its cosine similarity with two other sentences.
    """)
    return


@app.cell
def example_cell():
    import numpy as np

    words, E = C.glove()
    row = C.rows(words)

    def mean_of(sentence):
        return np.mean([E[row[w]] for w in sentence], axis=0)

    def cosine(a, b):
        return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

    good = mean_of(["the", "film", "was", "good"])
    print(good.shape)
    print(cosine(good, mean_of(["the", "film", "was", "not", "good"])))
    print(cosine(good, mean_of(["i", "enjoyed", "the", "movie"])))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The mean has 50 numbers, as every vector of GloVe. *not* moves it little: 0.990, against 0.949 for the same meaning in other words. `mean_of` looks up every word with `row[w]`, so a word that GloVe lacks, such as *o'clock*, stops it with a KeyError. The `mean_vector` of exercise 2 leaves such words out.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The same mean, or another?

    Each line holds two sentences, and GloVe has every word of them. Put the pair under *Exactly the same mean* when the means of the GloVe vectors of their words come out equal, number for number, and under *Another mean* when they differ, if only a little.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_same = L.sort("same-mean")
    sort_same
    return (sort_same,)


@app.cell(hide_code=True)
def check_sort(sort_same):
    result_sort = L.check("same-mean", sort_same)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_after_sort():
    mo.md(r"""
    The order never counts: *the bus hit the van* and *the van hit the bus* get the same mean, and so do *not cold but warm* and *not warm but cold*, which say the opposite. A sentence said twice keeps the share of every word: *rain* and *rain rain*. But *hot* twice among five words weighs more than once among four, and *a* is another word than *the*: those means differ, if only a little, with cosine similarities of 0.990 and 0.991. *the soup is not hot* still gets 0.983 with *the soup is hot*, and *the soup is warm* 0.972.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_mean():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. mean_vector: the mean of a sentence's word vectors

        Write `mean_vector(words, E, row)`: `words` is the list of words of a sentence, `E` a table of word vectors, and `row` a dict from each word to its row in `E`. Give back the mean of the vectors of the words that have a row, each word counted as often as it comes, or `None` when none of the words has one.

        The box starts `total` as `np.zeros(E.shape[1])`, a vector of zeros as long as a row of `E`, and `count` at 0; replace the `...`. The check first uses a small table, `small_E = np.array([[1, 0], [0, 1], [2, 2]])` with `small_row = {"a": 0, "b": 1, "c": 2}`: `mean_vector(["a", "b"], small_E, small_row)` should give `[0.5, 0.5]`, and `mean_vector(["a", "zz"], small_E, small_row)` should give `[1, 0]`, because *zz* has no row. Then cases of its own: a sentence without a word that has a row, one that starts with a word without a row, a word that comes twice, and two sentences with the `E` and `row` of GloVe.
        """),
        L.task("Give back the mean of the vectors of the words that have a row, or None without such a word."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mean():
    box_mean = L.code_box("mean_vector", lines=13)
    box_mean
    return (box_mean,)


@app.cell(hide_code=True)
def check_mean(box_mean):
    result_mean = L.check("mean_vector", box_mean.value)
    result_mean
    return (result_mean,)


@app.cell(hide_code=True)
def hints_mean():
    L.hints(
        "mean_vector",
        [
            "Go over the words with `for w in words:`. For a word that has a row, add its vector `E[row[w]]` to `total`, and add 1 to `count`. After the loop: when `count` is 0, give back `None`; otherwise, give back the mean.",
            "*zz* has no row, and `row[w]` stops with a KeyError for it. What can you ask the dict `row` about `w` before you look `w` up?",
            r"`        if ___:`",
        ],
        why="`for w in words:` with `if w in row:` under it, and under that `total = total + E[row[w]]` and `count = count + 1`; after the loop, `if count == 0:` with `return None`, and last `return total / count`. For `[\"a\", \"zz\"]`: *a* adds `[1, 0]` and counts 1, *zz* is skipped, so the mean is `[1, 0]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_mean():
    mo.md(r"""
    In the picture, your function took the mean of every sentence of the course's 16 pairs, in four kinds. The same words in another order get 1.000, and the pairs with *not* added 0.991 on average. The pairs with the same meaning in other words, 0.954 on average, hardly come out above the pairs with the same words and another meaning, 0.942: *we sat on the bank of the river* and *we put money in the bank* get 0.949, as much as *the film was good* and *i enjoyed the movie*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pooling():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. mean_pooling: the mean of the real rows

        Write `mean_pooling(tokens, mask)`: `tokens` is the table that a transformer gives a sentence in a batch, one row per place, and `mask` a column with 1 for every real piece and 0 for every *[PAD]* place. Give back the mean of the rows of `tokens` where `mask` is 1. One line does it, without a loop: `tokens * mask` keeps every real row and makes every *[PAD]* row 0.

        For `mean_pooling(np.array([[1, 2], [3, 4], [9, 9]]), np.array([[1], [1], [0]]))` the result is `[2, 3]`, the mean of the first two rows: the third is a *[PAD]* place. The check calls that and a table without *[PAD]*, then cases of its own: more *[PAD]* places, a 0 between two 1s, wider rows, and two sentences of the course in BERT-Tiny, with `C.tiny_tokens(i)` and `C.tiny_mask(i)`: the table of sentence `i` of the 32, after the second block, and its mask.
        """),
        L.task("Give back the mean of the rows of tokens where mask is 1."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pooling():
    box_pooling = L.code_box("mean_pooling", lines=6)
    box_pooling
    return (box_pooling,)


@app.cell(hide_code=True)
def check_pooling(box_pooling):
    result_pooling = L.check("mean_pooling", box_pooling.value)
    result_pooling
    return (result_pooling,)


@app.cell(hide_code=True)
def hints_pooling():
    L.hints(
        "mean_pooling",
        [
            "`tokens * mask` keeps the real rows and makes every *[PAD]* row 0. Add up the rows of that table with `.sum(axis=0)`, and divide by the number of real pieces.",
            "Every real piece has a 1 in `mask`, and every *[PAD]* place a 0. Which number, computed from `mask`, is the number of real pieces?",
            r"`    return (tokens * mask).sum(axis=0) / ___`",
        ],
        why="`return (tokens * mask).sum(axis=0) / mask.sum()`. For the example: `tokens * mask` is `[[1, 2], [3, 4], [0, 0]]`, its sum along axis 0 is `[4, 6]`, and `mask.sum()` is 2, so the mean is `[2, 3]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_pooling():
    mo.md(r"""
    In the picture, all three columns put the four kinds in the same order: the same words in another order highest, then *not* added, then the same meaning in other words, and the same words with another meaning lowest. They differ in how far apart the last two kinds come out. With the mean of GloVe, the same meaning in other words gets 0.954 on average and the same words with another meaning 0.942; with BERT-Tiny and your mean pooling, 0.897 and 0.857; with Granite, trained on pairs, 0.899 and 0.777. Not every pair follows: in Granite, *the film was long* still gets 0.851 with *the film was good*, more than *the engine failed to turn over* with *the car would not start*, 0.818. And Granite too gives the pairs with *not* added a high cosine similarity, 0.949 on average.
    """)
    return


@app.cell(hide_code=True)
def text_explore_glove():
    mo.md(r"""
    ### Try it: two sentences with GloVe

    Type two English sentences. The picture shows the mean of the GloVe vectors of each, with the words it used and the words that GloVe lacks, and the cosine similarity of the two means. Try the same words in another order, add *not*, or say the same thing in other words.
    """)
    return


@app.cell(hide_code=True)
def explore_glove_controls():
    explore_first = mo.ui.text(value="the shop opens at nine o'clock", label="First sentence", full_width=True)
    explore_second = mo.ui.text(value="the store is open at nine", label="Second sentence", full_width=True)
    mo.vstack([explore_first, explore_second])
    return explore_first, explore_second


@app.cell(hide_code=True)
def explore_glove(explore_first, explore_second):
    mo.Html(C.glove_picture(explore_first.value, explore_second.value, L.lang))
    return


@app.cell(hide_code=True)
def text_explore_granite():
    mo.md(r"""
    ### Try it: your sentences in Granite

    Granite Embedding runs on your computer once it is downloaded. Type up to six sentences, one per line, in English, Dutch or another of its languages: the picture compares every two, the most alike first. Try a sentence and its translation, a sentence with *not*, and the same meaning in other words.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download Granite Embedding (223 MB)")
    return (get_button,)


@app.cell(hide_code=True)
def granite_download(get_button):
    C.granite_or_stop(get_button, {
        "ask": "*This part needs Granite Embedding, which is not on your computer yet. The button downloads it once from Hugging Face, at a fixed version: `ibm-granite/granite-embedding-107m-multilingual`, 223 MB, under the Apache-2.0 license. The exercises and the rest of the lesson work without it.*",
        "busy": "**Downloading** {done} of {total} MB …",
        "no_net": "*The download did not work: Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: a file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is not enough room on your disk. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
    })
    granite_here = True
    mo.md("*Granite Embedding is downloaded.*")
    return (granite_here,)


@app.cell(hide_code=True)
def explore_granite_controls():
    explore_lines = mo.ui.text_area(value="the film was good\ni enjoyed the movie\nthe film was not good\nde film was goed", label="Your sentences, one per line", rows=6, full_width=True)
    explore_lines
    return (explore_lines,)


@app.cell(hide_code=True)
def explore_granite(explore_lines, granite_here):
    _ = granite_here
    mo.Html(C.granite_picture(explore_lines.value, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [One vector per word, or one per use](/course/open/nlp-static-vs-contextual?lang=en) gave every occurrence of a word its own vector; here a whole sentence gets one. The mean of the word vectors of [Word embeddings: neighbors and analogies](/course/open/nlp-word-embeddings?lang=en) loses the order of the words completely, and *not* nearly so. Mean pooling of a transformer's vectors, with a mask that keeps *[PAD]* out, does not do much better by itself; a model trained on pairs of sentences, such as Granite, puts sentences with the same meaning closer, though *not* stays hard. The next lesson of the domain, *Same meaning? Similarity and paraphrases*, measures how well cosine similarity follows what people judge, and where a paraphrase begins.

    **An everyday example.** Bread and pancakes are both made of flour, eggs and milk. A list of ingredients is a mean: the method, the order in which things happen, makes the dish.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_mean, result_pooling, result_sort):
    L.wrap_up(result_sort, result_mean, result_pooling)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
