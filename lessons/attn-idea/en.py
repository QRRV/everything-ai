# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson attn-idea. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("attn-idea", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [The forward pass in NumPy, with shapes](/course/open/nn-forward-numpy?lang=en) and [Many classes: softmax](/course/open/ml-softmax?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="`X` has shape (4, 2): 4 words with 2 numbers each, one word per row. What shape does `X @ X.T` have?",
        options={
            "small": "(2, 2): one number for every pair of the 2 numbers",
            "right": "(4, 4): one number for every pair of words",
            "same": "(4, 2): the same shape as `X`",
        },
        correct="right",
        why={
            "small": "That would be `X.T @ X`: (2, 4) @ (4, 2). See [The forward pass in NumPy, with shapes](/course/open/nn-forward-numpy?lang=en).",
            "same": "(4, 2) @ (2, 4) gives (4, 4): the inner numbers, 2 and 2, are equal, and the outer numbers remain. See [The forward pass in NumPy, with shapes](/course/open/nn-forward-numpy?lang=en).",
            "right": "(4, 2) @ (2, 4) gives (4, 4): row $i$, column $j$ is the dot product of word $i$ with word $j$. Today, such dot products say how well two words match.",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def quick_check_equal():
    quick_equal = L.choice(
        "quick-equal",
        question="Two classes get the same logit, and a third one a much lower one. What does the softmax give the first two?",
        options={
            "right": "The same probability each, a little under one half",
            "one": "Both 1: they are both the largest",
            "logit": "Each its own logit back",
        },
        correct="right",
        why={
            "one": "The softmax probabilities add up to 1, so two of them cannot both be 1. See [Many classes: softmax](/course/open/ml-softmax?lang=en).",
            "logit": "The softmax turns logits into probabilities between 0 and 1 that add up to 1. See [Many classes: softmax](/course/open/ml-softmax?lang=en).",
            "right": "Equal logits give equal probabilities, and the low third one takes only a little, so each gets a little under one half. Today the softmax turns matches into weights in just this way.",
        },
    )
    quick_equal
    return (quick_equal,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    An embedding gives a word one list of numbers, whatever the sentence. But "bank" next to "river" means something different from "bank" next to "account". A model that reads text needs the meaning a word has *here*, in this sentence. So every word must be able to look at the other words, and take in most from those that fit it best.

    **An everyday example.** When you read a word that can mean two things, you look at the words around it to know which, and you look most at the words that help: "river" tells you more about "bank" than "the".
    """)
    return


@app.cell(hide_code=True)
def predict_bank():
    guess_bank = L.guess(
        "bank-looks",
        question="In \"the river bank\", \"bank\" takes a weighted average of the embeddings of all three words, with more weight for the words whose embedding matches its own best, by the dot product. With the made-up embeddings of this lesson, the (0.5, 0.5), river (0, 2) and bank (1, 1), which word gets the most weight?",
        options={
            "river": "\"river\", by far: more than \"the\" or \"bank\"",
            "equal": "All three about equally",
            "right": "\"bank\" itself, just as much as \"river\"",
        },
        reveal="**\"bank\" itself, just as much as \"river\".** The dot product of \"bank\", (1, 1), with \"river\", (0, 2), is 2, and with itself also 2; with \"the\" it is 1. So \"river\" and \"bank\" get the same weight, 0.42 each, and \"the\" 0.16: the worked example below works it out. A word always matches its own embedding well. Still, the new vector of \"bank\" moves toward water, because \"river\" pulls it that way. In a later lesson, real attention gives every word separate vectors for looking and for being found.",
    )
    guess_bank
    return (guess_bank,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Attention** lets every word, at every place in the sentence, take a weighted average of the embeddings of all words, its own included. The weights depend on how well two words match, measured with the dot product of their embeddings. The softmax turns the dot products into weights that are positive and add up to 1: the attention weights. For word $i$, with embeddings $\mathbf{x}_j$:
        """),
        L.formula(r"\begin{aligned} s_{j} &= [[hidden:\mathbf{x}_i]] \cdot [[hidden:\mathbf{x}_j]] \\ [[hidden:a_{j}]] &= \frac{e^{s_{j}}}{\sum_k e^{s_{k}}} \\ [[hidden:\mathbf{h}_i]] &= \sum_j [[hidden:a_{j}]] \, [[hidden:\mathbf{x}_j]] \end{aligned}"),
        mo.md(r"""
        $\mathbf{h}_i$ is the new vector of word $i$. Every word does this at the same time, and its new vector goes on to the next layer. Attention is the heart of the **transformer**, the kind of network behind almost every large language model: layer after layer of attention, each followed by a small MLP. The real thing has a few more parts, which later lessons build one by one.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **attention** | | every word takes a weighted average of the embeddings of all words, with more weight for the words that match it best | `weights @ X` |
        | $\mathbf{h}_i$ | h i | the new vector of word $i$ after attention | `weights @ X` |
        | **transformer** | | a network made of layers of attention, each followed by a small MLP | |
        """),
        L.watch_out("In everyday language, attention is something you pay. Here, it is a calculation, a weighted average; the model does not notice anything. Nor are the attention weights learned, like the weights of a layer: the softmax works them out anew for every sentence. And the embeddings in this lesson are made up by the course, with 2 numbers: how much a word is about money, and how much about water. A real model learns its embeddings, with hundreds of numbers."),
        mo.md(r"""
        The picture: the made-up embeddings of "river", "bank" and "account" as arrows, and the new vector of "bank" after attention in "the river bank" and in "the bank account". The same embedding of "bank" turns toward water in one sentence, and toward money in the other.
        """),
        mo.Html(C.bank_picture({"title": "Bank, before and after attention", "x": "money", "y": "water",
                                "river": "river", "bank": "bank", "account": "account", "in_river": "next to river", "in_account": "next to account",
                                "alt": "Arrows from the origin: river points straight up, account straight to the right, bank diagonally between them. After attention, bank next to river points more upward, and bank next to account more to the right."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    The new vector of one word, after attention:

    1. The scores: the dot product of the word's embedding with the embedding of every word in the sentence, its own included.
    2. The attention weights: the softmax of the scores.
    3. The new vector: every embedding times its weight, all added up.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    "bank" in "the river bank", with the made-up embeddings of the picture, one word per row of `_X`. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_attention():
    import numpy as np

    _words = ["the", "river", "bank"]
    _X = np.array([[0.5, 0.5], [0.0, 2.0], [1.0, 1.0]])
    _i = 2
    # 1. the scores: the dot product of the embedding of word _i with every row of _X
    _scores = _X @ _X[_i]
    print(_words[_i], _scores)
    # 2. the attention weights: the softmax of the scores
    _weights = np.exp(_scores) / np.sum(np.exp(_scores))
    print(np.round(_weights, 3), np.sum(_weights))
    # 3. the new vector: every row of _X times its weight, added up
    _h = _weights @ _X
    print(np.round(_h, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `_X @ _X[_i]` gives the dot product of "bank" with every row: 1 with "the", 2 with "river" and 2 with itself. The softmax turns them into the weights 0.155, 0.422 and 0.422, which add up to 1. `_weights @ _X` multiplies every row of `_X` by its weight and adds them up. Its shapes are (3,) @ (3, 2): the 3s fit, and (2,) remains, one vector of 2 numbers. With the weights rounded: $0.155 \cdot (0.5, 0.5) + 0.422 \cdot (0, 2) + 0.422 \cdot (1, 1) \approx (0.5, 1.345)$. The embedding of "bank" was $(1, 1)$; its new vector has moved toward water.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_weights():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Weights that add up to 1
    """)
    return


@app.cell(hide_code=True)
def choice_weights_cell():
    choice_weights = L.choice(
        "weights-sum",
        question="The attention weights of one word always add up to 1. What does that give the new vector?",
        options={
            "right": "It is an average: no larger than the largest embedding, however long the sentence",
            "largest": "It is the embedding of the word with the largest weight: the best match takes it all",
            "zero": "Its numbers add up to 1 as well: just like the weights",
        },
        correct="right",
        why={
            "largest": "All words count, each by its weight; the largest weight counts most, not alone.",
            "zero": "The weights add up to 1, not the numbers of the new vector: in the example, they add up to 1.845.",
            "right": "Weights that add up to 1 make a weighted average. Without the softmax, a sentence of 100 words would add up 100 embeddings, and the new vector would grow with the length of the sentence.",
        },
    )
    choice_weights
    return (choice_weights,)


@app.cell(hide_code=True)
def text_exercise_attend():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The new vector of any word

        Make `attend(X, i)` give the new vector of word `i` after attention, by the recipe. `X` is a NumPy array with one embedding per row, one row per word; `i` is the number of the row. Replace the three `...`, one per step, as in the worked example. The check calls `attend(np.array([[1, 0], [0, 1], [1, 1]]), 0)`, which should give `np.array([0.8446, 0.5777])`, and tries other words and other sentences, also with 3 numbers per word, and a sentence of one word.
        """),
        L.task("Make `attend` give the new vector of word `i` after attention."),
    ])
    return


@app.cell(hide_code=True)
def exercise_attend():
    box_attend = L.code_box("attend", lines=10)
    box_attend
    return (box_attend,)


@app.cell(hide_code=True)
def check_attend(box_attend):
    result_attend = L.check("attend", box_attend.value)
    result_attend
    return (result_attend,)


@app.cell(hide_code=True)
def hints_attend():
    L.hints(
        "attend",
        [
            "The same three steps as in the worked example, with `X` and `i` instead of `_X` and `_i`.",
            "Step 1 is one `@`, of `X` and row `i` of `X`. Step 2 is the softmax of those scores, as in [Many classes: softmax](/course/open/ml-softmax?lang=en).",
            "`scores = X @ X[i]`, `weights = np.exp(scores) / np.sum(___)`, and `return weights @ ___`.",
        ],
        why="`scores = X @ X[i]`; `weights = np.exp(scores) / np.sum(np.exp(scores))`; `return weights @ X`. In the example, the scores are 1, 0 and 1, the weights 0.4223, 0.1554 and 0.4223, and the new vector is $0.4223 \\cdot (1, 0) + 0.1554 \\cdot (0, 1) + 0.4223 \\cdot (1, 1) = (0.8446, 0.5777)$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_transformer():
    mo.md(r"""
    ### Exercise 3. Three building blocks
    """)
    return


@app.cell(hide_code=True)
def choice_transformer_cell():
    choice_transformer = L.choice(
        "transformer",
        question="Which of these is a transformer?",
        options={
            "rnn": "A network that reads the words one by one and carries a memory from each word to the next",
            "cnn": "A network that slides one small filter along the sentence, a few words at a time",
            "right": "A network in which every word looks at all other words at once, layer after layer",
        },
        correct="right",
        why={
            "rnn": "That is a recurrent network, from [Recurrence: a network with memory](/course/open/rnn-memory-idea?lang=en): it has to pass everything along, word by word.",
            "cnn": "That is a convolution, from [Convolution: one small filter slides over the image](/course/open/cnn-convolution-idea?lang=en), on a sentence instead of an image: each place sees only its neighbors.",
            "right": "In a transformer, every word takes in every other word in one step, with attention. Far-apart words meet as easily as neighbors, and all places are worked out at the same time.",
        },
    )
    choice_transformer
    return (choice_transformer,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The embeddings of [Embeddings: meaning as coordinates](/course/open/ml-embeddings-idea?lang=en) give every word one fixed list of numbers; attention gives it a new one that depends on the sentence. The scores are dot products, as in the matrix products of [The forward pass in NumPy, with shapes](/course/open/nn-forward-numpy?lang=en), and the weights come from [Many classes: softmax](/course/open/ml-softmax?lang=en). Attention began in 2014 as an addition to recurrent neural networks that translate, so that they could look back at every word of the sentence; the transformer of 2017 kept only the attention. At the intermediate level, you build real attention step by step, and then a transformer of your own. The next lesson, *Learning from rewards*, is about another way to learn: from rewards instead of from labels.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_weights, result_attend, choice_transformer):
    L.wrap_up(choice_weights, result_attend, choice_transformer)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
