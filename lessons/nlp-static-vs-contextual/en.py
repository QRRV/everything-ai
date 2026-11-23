# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-static-vs-contextual. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-static-vs-contextual", "en")
    C = L.checks   # the course's sentences and vectors, BERT-Tiny, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions, about [Word embeddings: neighbors and analogies](/course/open/nlp-word-embeddings?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_apple():
    quick_apple = L.choice(
        "quick-apple",
        question="In *Word embeddings: neighbors and analogies*, how many vectors does GloVe have for the word *apple*?",
        options={
            "senses": "Two: one for the fruit, one for the company",
            "right": "One",
            "uses": "One for every sentence it occurs in",
        },
        correct="right",
        why={
            "right": "GloVe gives every word one vector, learned from all the sentences the word occurs in. The nearest neighbors of *apple* were *blackberry*, *chips*, *iphone* and *microsoft*: not a fruit among them. This lesson looks at a model that gives every use of a word its own vector.",
            "senses": "GloVe knows nothing of meanings: it learns one vector per word, from all its sentences together. The nearest neighbors of *apple* were *blackberry*, *chips*, *iphone* and *microsoft*. See [Word embeddings: neighbors and analogies](/course/open/nlp-word-embeddings?lang=en).",
            "uses": "That is what the models of this lesson do. GloVe learns one vector per word, from all its sentences together, and looks it up in a table. See [Word embeddings: neighbors and analogies](/course/open/nlp-word-embeddings?lang=en).",
        },
    )
    quick_apple
    return (quick_apple,)


@app.cell(hide_code=True)
def quick_check_matmul():
    quick_matmul = L.choice(
        "quick-matmul",
        question="`E` is a table with one row per word, `v` a vector with as many numbers as a row. What does `E @ v` give?",
        options={
            "one": "One number: the dot product of E and v",
            "cos": "The cosine similarity of every row with v, one per row",
            "right": "The dot product of every row of E with v",
        },
        correct="right",
        why={
            "right": "`E @ v` gives one dot product per row. Divided by the norm of each row and by the norm of `v`, as in your `cosines`, they become cosine similarities. This lesson compares the vectors of a word in different sentences in that way.",
            "one": "`@` of a table and a vector gives one number for every row of the table, not one number in all. See [Word embeddings: neighbors and analogies](/course/open/nlp-word-embeddings?lang=en).",
            "cos": "Those come only after dividing by the norms. `E @ v` alone gives the dot products, which grow with the lengths of the rows. See [Word embeddings: neighbors and analogies](/course/open/nlp-word-embeddings?lang=en).",
        },
    )
    quick_matmul
    return (quick_matmul,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    GloVe gives a word one vector, whatever it means in a sentence. The nearest neighbors of *bank* are *banks*, *securities*, *banking*, *investment* and *exchange*: all about money. *money* is the 98th neighbor, and *river* only the 2,181st. The bank of a river has to make do with the same vector as the bank with your money.

    In 2018 Matthew Peters and colleagues presented ELMo, a model that gives a word a new vector every time it is used, computed from the sentence around it. They wrote that in their model "each token is assigned a representation that is a function of the entire input sentence." The language models of today work this way inside too. This lesson takes a small model of that kind, BERT-Tiny, and measures whether the bank of a river and the bank with your money really get different vectors.

    **An everyday example.** Someone who hears *the bank was steep and muddy* does not think of money, not even for a moment: the rest of the sentence settles what *bank* means.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_across = L.guess(
        "guess",
        question="The course wrote 44 sentences with *bank*: 22 about a river and 22 about money. At the end of its computation, BERT-Tiny gives *bank* a vector of 128 numbers in each of them. Take the cosine similarity of *bank* in a river sentence with *bank* in a money sentence, for all 22 times 22 pairs, and average. What comes out, roughly?",
        options={
            "low": "0.12",
            "right": "0.83",
            "neg": "−0.40",
            "half": "0.45",
        },
        reveal="**0.83**: *bank* in a sentence about a river and *bank* in a sentence about money still point almost the same way. Within the river sentences the mean is 0.87, within the money sentences 0.84: a little higher, but not by much. This lesson shows that so small a difference can still be enough to tell the two meanings apart.",
    )
    guess_across
    return (guess_across,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Static and contextual embeddings

    A **static embedding** gives a word one fixed vector, the same in every sentence. GloVe and word2vec are static. Daniel Jurafsky and James H. Martin write in *Speech and Language Processing*: "Word2vec embeddings are static embeddings, meaning that the method learns one fixed embedding for each word in the vocabulary." A **contextual embedding** is a vector for one use of a word, computed from the sentence it stands in, so the same word gets another vector in every sentence. In their words, where w stands for any word: "with contextual embeddings, each word w will be represented by a different vector each time it appears in a different context."

    ### BERT-Tiny

    BERT is a model that Jacob Devlin and colleagues at Google described in 2019. Its name stands for Bidirectional Encoder Representations from Transformers: it looks at the words on both sides of every word. It learned from books and from English Wikipedia, together 3.3 billion words, by filling in words that were hidden; a later lesson, *BERT: filling in the blank*, shows how. **BERT-Tiny** is the smallest of 24 small versions that Iulia Turc and colleagues described in 2019 and released in 2020: 4.4 million parameters, of which 3.9 million are the vectors of its 30,522 pieces. It knows only English, in small letters.

    BERT-Tiny first cuts a sentence into pieces, its tokens, as the tokenizers of [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en) do: *kayak* becomes *kay* and *##ak*, where *##* means that the piece continues a word. It puts the piece *[CLS]* before the sentence and *[SEP]* after it. Every piece gets a vector of 128 numbers: the vector of the piece itself, plus a vector for its place in the sentence, plus one more that is the same for every piece of a single sentence. Then come two **blocks**. A block computes a new vector for every piece from the vectors of all pieces of the sentence, with attention, as in *Attention: every word looks at the others*. So before the first block, the vector of *bank* depends only on the piece *bank* and its place; after each block, on the whole sentence.

    The picture shows what that does to the 44 sentences of the course.
    """)
    return


@app.cell(hide_code=True)
def idea_uses():
    mo.Html(C.uses_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.vstack([
        mo.md(r"""
        ### Self-similarity

        How alike are the uses of a word? In 2019 Kawin Ethayarajh measured that with the **self-similarity**: the mean cosine similarity of every two different uses of a word. Ethayarajh writes, where w is a word, ℓ a layer (here: the vectors after one block) and n the number of different sentences the word stands in: "the self-similarity of a word w in layer ℓ is the average cosine similarity between its contextualized representations across its n unique contexts." For a static embedding it is 1: every use has the same vector. The lower it is, the more the sentence counts.

        ### Guessing the meaning from examples

        If the uses of *bank* in a river sentence lie closer to each other than to the uses in a money sentence, the vectors can guess the meaning of *bank* in a new sentence. Take sentences whose meaning you know, average the vectors of *bank* per meaning, and choose for the new sentence the meaning whose mean vector has the highest cosine similarity with its vector. Jurafsky and Martin describe this guess, with large sets of sentences labeled with their meaning; Peters and colleagues used it for ELMo.

        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **static embedding** | one fixed vector per word, the same in every sentence, such as GloVe and word2vec |
        | **contextual embedding** | a vector for one use of a word, computed from the whole sentence it stands in |
        | **BERT-Tiny** | a small model (2019, released 2020) that gives every piece of an English sentence a contextual embedding of 128 numbers, after two blocks |
        | **block** | a step of such a model: a new vector for every piece, computed from the vectors of all pieces of the sentence |
        | **[CLS]**, **[SEP]** | the pieces BERT puts before and after a sentence |
        | **self-similarity** | the mean cosine similarity of every two different uses of a word; 1 when every use has the same vector |
        | `pieces.index(x)` | the place of the first `x` in the list `pieces`, counted from 0 |
        | `np.mean(A, axis=...)` | the mean along one axis of a table: one mean per column or one per row, depending on the axis, as with `np.sum(A, axis=...)` |
        """),
        L.watch_out("Different vectors for different uses do not mean that the model knows meanings. It never saw a list of meanings: it learned to fill in hidden words, and every vector carries some of its whole sentence."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Cut the sentence into pieces, and find the place of the word among them.
    2. Run the model, and take the vector at that place: before the first block, and after each block.
    3. Compare two uses of the word with the cosine similarity. For all uses of a word, take the mean over every two different uses: the self-similarity.
    4. To guess the meaning of a new use, average the vectors of the examples of each meaning, and take the meaning whose mean has the highest cosine similarity with the new vector.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The button below downloads BERT-Tiny. The code then takes *bank* in two sentences of the course: one about a river, one about money. The course's `bert.pieces` gives the pieces of a sentence, with *[CLS]* and *[SEP]*. `bert.layers` gives three tables, with one row of 128 numbers per piece: before the first block, after the first block and after the second. `bank_vectors` finds *bank* among the pieces and gives back its three rows.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download BERT-Tiny, 18 MB")
    return (get_button,)


@app.cell(hide_code=True)
def bert_cell(get_button):
    bert = C.bert_or_stop(get_button, {
        "ask": "*BERT-Tiny is not on your computer yet. The button downloads it once from Hugging Face, at a fixed version: `google/bert_uncased_L-2_H-128_A-2`, 18 MB, under the Apache-2.0 license. The worked example and the part where you try it yourself need it; the exercises do not.*",
        "busy": "**Downloading** {done} of {total} MB …",
        "no_net": "*The download did not work: Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: a file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is not enough room on your disk. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
    })
    mo.md("*BERT-Tiny is downloaded.*")
    return (bert,)


@app.cell
def example_cell(bert):
    import numpy as np


    def bank_vectors(sentence):
        pieces = bert.pieces(sentence)
        i = pieces.index("bank")
        return [layer[i] for layer in bert.layers(sentence)]


    river = "We had a picnic on the bank of the river."
    money = "I opened an account at the bank downtown."
    print(bert.pieces(river))
    print(bert.pieces(money))
    river_vectors = bank_vectors(river)
    money_vectors = bank_vectors(money)
    for k in range(3):
        a, b = river_vectors[k], money_vectors[k]
        print(k, round(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The first sentence has 13 pieces, the second 11; *bank* stands at place 7 in both, counted from 0, with *[CLS]* at place 0. The last three lines give, after the number of the step (0 before the first block, 1 and 2 after each block), the cosine similarity of the two vectors of *bank*: 1.0 at step 0, 0.968 at step 1 and 0.9 at step 2. Before the first block, the two are exactly the same. After each block they differ more: the sentences have gone into the vectors.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_same():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Why the same vector?
    """)
    return


@app.cell(hide_code=True)
def choice_same_cell():
    choice_same = L.choice(
        "same-start",
        question="Before the first block, *bank* has exactly the same vector in *We had a picnic on the bank of the river.* and in *I opened an account at the bank downtown.*: a cosine similarity of 1. Why?",
        options={
            "length": "Both sentences have the same number of pieces",
            "right": "It is the same piece at the same place",
            "place": "Before the blocks, every piece has one vector, whatever its place",
        },
        correct="right",
        why={
            "right": "Before the first block, the vector of a piece is made from the piece and its place, and nothing else. *bank* stands at place 7 in both sentences, so the two vectors are the same. The course wrote its 44 sentences in pairs like this, one about a river and one about money with *bank* at the same place, so that only the blocks can pull the two meanings apart.",
            "length": "The first sentence has 13 pieces, the second 11. The length does not count before the first block: only the piece and its place do.",
            "place": "The place counts too: the same piece at another place gets another vector. Here *bank* stands at place 7 in both sentences.",
        },
    )
    choice_same
    return (choice_same,)


@app.cell(hide_code=True)
def text_exercise_self():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. self_similarity: how alike are the uses?

        Write `self_similarity(rows)`: give back the mean cosine similarity of every two different rows of the table `rows`. Take every pair of two different rows once, compute the cosine similarity of the two, and divide the sum by the number of pairs.

        For `self_similarity(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))` the result is 0.4714, rounded: the three pairs have cosine similarities 0, 0.707 and 0.707. Replace the `...`. The check calls that and `self_similarity(np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]))`, which should give 1, then cases of its own: two rows, four rows of three numbers, three rows of which two point opposite ways, and the vectors of *bank* in sentences of the course, after the second block and before the first. `C.bank_rows(2, "river", "example")` gives the table of the 14 river sentences to learn from, after the second block: one row per sentence.
        """),
        L.task("Give back the mean cosine similarity of every two different rows of rows."),
    ])
    return


@app.cell(hide_code=True)
def exercise_self():
    box_self = L.code_box("self_similarity", lines=10)
    box_self
    return (box_self,)


@app.cell(hide_code=True)
def check_self(box_self):
    result_self = L.check("self_similarity", box_self.value)
    result_self
    return (result_self,)


@app.cell(hide_code=True)
def hints_self():
    L.hints(
        "self_similarity",
        [
            "Start with `total = 0` and `count = 0`. Then two loops: `for i in range(len(rows)):`, and inside it a loop over `j`, so that each pair of two different rows comes once. Inside both, add the cosine similarity of `rows[i]` and `rows[j]` to `total`, and 1 to `count`. After the loops, `return total / count`.",
            "Row 0 with row 1 is the same pair as row 1 with row 0, and row 1 with row 1 is not a pair at all. When `i` is 1, which is the first `j` that gives a new pair?",
            r"`        for j in range(___, len(rows)):`",
        ],
        why=r"`for j in range(i + 1, len(rows)):`: `j` starts just after `i`, so every pair comes once and no row meets itself. In full: `total = 0` and `count = 0`; `for i in range(len(rows)):` with that loop inside it; inside both, `a, b = rows[i], rows[j]`, `total += a @ b / (np.linalg.norm(a) * np.linalg.norm(b))` and `count += 1`; then `return total / count`. For the example, the three pairs give 0, 0.707 and 0.707, and the mean is 0.4714.",
    )
    return


@app.cell(hide_code=True)
def text_after_self():
    mo.md(r"""
    In the picture, your function measured the 44 uses of *bank*. In GloVe the self-similarity is 1: every use has the same vector. In BERT-Tiny it is 0.90 before the first block, 0.92 after the first and 0.84 after the second. Before the first block it is below 1 only because *bank* stands at different places in different pairs.

    That looks high, but look at the gray bars: two different words in two different sentences, such as *picnic* and *account*, already have a mean cosine similarity of 0.24 before the first block, 0.53 after the first and 0.57 after the second. After the blocks, all vectors point roughly the same way. Ethayarajh found that in larger models too: the vectors are "anisotropic, occupying a narrow cone in the vector space." So a self-similarity says little on its own, and Ethayarajh compares it with two random words from different sentences. Here, the uses of *bank* lie 0.66 above the gray bar before the first block, 0.39 after the first and 0.27 after the second: after every block, the sentence counts more. Ethayarajh, about BERT, ELMo and GPT-2: "While representations of the same word in different contexts still have a greater cosine similarity than those of two different words, this self-similarity is much lower in upper layers."
    """)
    return


@app.cell(hide_code=True)
def text_exercise_guess():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. guess_sense: river or money?

        Write `guess_sense(v, river, money)`. `v` is the vector of *bank* in a new sentence; `river` and `money` are tables with the vectors of *bank* in example sentences of each meaning, one row per sentence. Make the mean row of each table, with one number per column: the mean of that column. Give back `"river"` when `v` has a higher cosine similarity with the mean row of `river` than with the mean row of `money`, and `"money"` otherwise.

        For `guess_sense(np.array([1.0, 0.0]), np.array([[1.0, 0.2], [1.0, -0.2], [1.0, 0.0]]), np.array([[0.0, 1.0], [0.2, 1.0], [-0.1, 1.0]]))` the result is `"river"`. Replace the `...`. The check calls that and the same with `np.array([0.1, 1.0])` as `v`, which should give `"money"`, then cases of its own: small tables where the dot product, the distance, the single nearest example or the mean of the cosine similarities would give the other meaning, and four new sentences of the course after the second block, from `C.bank_rows(2, None, "new")`, with the 14 example sentences of each meaning.
        """),
        L.task("Give back the meaning whose mean row has the higher cosine similarity with v."),
    ])
    return


@app.cell(hide_code=True)
def exercise_guess():
    box_guess = L.code_box("guess_sense", lines=9)
    box_guess
    return (box_guess,)


@app.cell(hide_code=True)
def check_guess(box_guess):
    result_guess = L.check("guess_sense", box_guess.value)
    result_guess
    return (result_guess,)


@app.cell(hide_code=True)
def hints_guess():
    L.hints(
        "guess_sense",
        [
            "Four steps: `r = np.mean(river, axis=...)`, with the right axis, and `m` the same for `money`; then the cosine similarity of `v` with `r` and of `v` with `m`, as in your `self_similarity`; last, an `if` that gives back `\"river\"` when the first is higher, and `\"money\"` otherwise.",
            "`river` has one row per sentence and one column per number, 128 for BERT-Tiny. The mean row needs one number per column, as `v` has. In *Sums and means along an axis*, the axis said which way the sum or mean runs. Along which axis does `np.mean` then run: down the rows, or along each row?",
            r"`    r = np.mean(river, axis=___)`",
        ],
        why=r"`r = np.mean(river, axis=0)`: `axis=0` runs down the rows, so every column gets its mean, and `r` has as many numbers as `v`. In full: `r` and `m` that way; `to_river = v @ r / (np.linalg.norm(v) * np.linalg.norm(r))` and `to_money` the same with `m`; then `if to_river > to_money:` with `return \"river\"` under it, and `return \"money\"`. In the example, `r` is `[1, 0]` and `m` is about `[0.033, 1]`, so `v` lies nearest to the river.",
    )
    return


@app.cell(hide_code=True)
def text_after_guess():
    mo.md(r"""
    In the picture, your function guessed *bank* in the 16 new sentences, with the 14 example pairs. With GloVe, *bank* has one vector, so the two mean rows are the same, and your function gives the same answer 16 times: 8 right, as many as a coin gets on average. Before the first block it is 8 too: every new sentence has its pair with *bank* at the same place, so both get the same vector, and the examples of the two meanings have the same mean. After the first block and after the second, 15 of the 16 are right. The one miss is *We walked along the bank until sunset.*: no water in it, and it goes to money.

    Average all uses of a word, whatever their meaning, and you have one vector per word again: a static embedding made from a contextual model. Rishi Bommasani, Kelly Davis and Claire Cardie did that in 2020 with BERT and English Wikipedia: the mean over many sentences worked best, and the more sentences, the better. With enough sentences, their vectors beat GloVe and word2vec on tests of how alike two words are. A static vector such as GloVe's carries the meanings of a word too: Sanjeev Arora and colleagues showed in 2018 that, under their model of how text comes about, it is close to a weighted sum of a vector per meaning, each weighted by how often that meaning occurs.

    BERT-Tiny learned only English. A Dutch sentence it cuts into odd pieces: *De bank staat bij het water.* becomes *de*, *bank*, *st*, *##aa*, *##t*, *bi*, *##j*, *het*, *water* and the period. Dutch has a *bank* of its own: one to sit on, and one for money. For Dutch there are models of the same kind: BERTje (2019), learned from 2.4 billion Dutch tokens, and RobBERT (2020), with 117 million parameters against the 4.4 million of BERT-Tiny. A later lesson, *Language models for Dutch*, comes back to them.
    """)
    return


@app.cell(hide_code=True)
def text_explore():
    mo.md(r"""
    ### Try it: one word in two sentences

    Type two sentences and a word that stands in both. The picture shows the pieces, and the cosine similarity of the vectors of the word in the two sentences, before the first block and after each block. Peters and colleagues showed the word *play*: in GloVe its nearest neighbors are mostly about sport, while ELMo found a sentence about sport for a play in a game and a sentence about the theater for a play on Broadway. The two sentences below start with that. Try *bank*, *bat* or *light* too, and put the word at the same place in both sentences, to see what the blocks alone do.
    """)
    return


@app.cell(hide_code=True)
def explore_controls():
    explore_first = mo.ui.text(value="The team made a brilliant play in the last minute.", label="Sentence 1", full_width=True)
    explore_second = mo.ui.text(value="We saw a play by Shakespeare at the theater.", label="Sentence 2", full_width=True)
    explore_word = mo.ui.text(value="play", label="Word")
    mo.vstack([explore_first, explore_second, explore_word])
    return explore_first, explore_second, explore_word


@app.cell(hide_code=True)
def explore_look(bert, explore_first, explore_second, explore_word):
    bert   # BERT-Tiny is downloaded: this cell waits for the button above
    mo.Html(C.explore_picture(explore_first.value, explore_second.value, explore_word.value, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Word embeddings: neighbors and analogies](/course/open/nlp-word-embeddings?lang=en), GloVe gave every word one vector: a static embedding. BERT-Tiny gives every use its own vector, a contextual embedding, computed in two blocks from the whole sentence. Before its first block, *bank* depends only on its piece and its place; after the blocks, the river and money uses come apart enough that the mean of 14 examples per meaning guesses the meaning in 15 of 16 new sentences. The lead of *bank* over two different words shrinks after every block. The next lesson of the domain, *Senses and relations between words*, asks what the meanings of a word are, and how words relate to each other.

    **An everyday example.** A translation app that turns *bank* into Dutch has to choose between *oever* and *bank*: only the rest of the sentence can tell which.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_same, result_guess, result_self):
    L.wrap_up(choice_same, result_self, result_guess)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
