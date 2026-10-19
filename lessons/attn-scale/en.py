# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson attn-scale. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("attn-scale", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Queries, keys and values](/course/open/attn-qkv?lang=en) and [Vanishing and exploding gradients](/course/open/nn-vanishing-exploding?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_value():
    quick_value = L.choice(
        "quick-value",
        question="In self-attention, the new vector of a word is a weighted average. Of what?",
        options={
            "query": "Of the queries of the words",
            "key": "Of the keys of the words",
            "right": "Of the values of the words",
        },
        correct="right",
        why={
            "query": "The query only decides where the word itself looks. See [Queries, keys and values](/course/open/attn-qkv?lang=en).",
            "key": "The key decides how much other words take from it, not what they take. See [Queries, keys and values](/course/open/attn-qkv?lang=en).",
            "right": "Queries and keys only make the scores. Today you look at the size of those scores.",
        },
    )
    quick_value
    return (quick_value,)


@app.cell(hide_code=True)
def quick_check_saturation():
    quick_saturation = L.choice(
        "quick-saturation",
        question="Where an activation function saturates, such as the sigmoid at $z = 10$, what reaches the layers before it on the way back?",
        options={
            "right": "Almost no gradient: its slope there is almost 0",
            "full": "The whole gradient: saturation only changes the forward pass",
            "more": "A larger gradient, as the input of the function is large",
        },
        correct="right",
        why={
            "full": "On the way back, the gradient is multiplied by the slope, and in the flat part the slope is almost 0. See [Vanishing and exploding gradients](/course/open/nn-vanishing-exploding?lang=en).",
            "more": "A large input lands in the flat part of the curve, where the slope is smallest. See [Vanishing and exploding gradients](/course/open/nn-vanishing-exploding?lang=en).",
            "right": "Today the softmax of attention turns out to saturate in the same way, when its scores get large.",
        },
    )
    quick_saturation
    return (quick_saturation,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A dot product adds up one product for every number of the two vectors. Real queries and keys have many numbers: 64 in GPT-2. The longer they are, the larger the scores get, and large scores make the softmax very sharp: almost all weight goes to one word. That looks decisive, but it all but stops learning: the softmax then saturates, like a sigmoid far from 0, and hardly any gradient passes back to the queries and keys. The model can no longer learn where to look. The fix is one division.
    """)
    return


@app.cell(hide_code=True)
def predict_spread():
    guess_spread = L.guess(
        "spread",
        question="A query and a key each have $d_k$ random numbers with mean 0 and standard deviation 1. With 4 numbers, their dot products spread about 2 around 0. How far do they spread with 64 numbers?",
        options={
            "same": "About 2: the numbers themselves are no larger",
            "linear": "About 32: 16 times as many numbers, 16 times the spread",
            "right": "About 8: 16 times as many numbers, 4 times the spread",
        },
        reveal="**About 8.** The spread grows with the square root of the number of products: $\\sqrt{64} = 8$. The products have mean 0, so many cancel out: their variances add up, not their sizes. With 1,024 numbers, the spread is about 32.",
    )
    guess_spread
    return (guess_spread,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Say the numbers of a query and a key are random and independent, so that knowing one tells you nothing about another, each with mean 0 and variance 1, the variance of [Expected value and variance](/course/open/prob-expectation?lang=en). Two rules from probability, which this lesson takes as given: the product of two such numbers, $q_i k_i$, again has mean 0 and variance 1; and the variance of a sum of independent numbers is the sum of their variances. So the score $\mathbf{q} \cdot \mathbf{k}$, a sum of $d_k$ products, has variance $d_k$, and a standard deviation of $\sqrt{d_k}$. Dividing every score by $\sqrt{d_k}$ brings the spread back to 1, whatever $d_k$ is. That is **scaled dot-product attention**, the attention of the transformer, from *Attention Is All You Need* (Vaswani and others, 2017):
        """),
        L.formula(r"[[hidden:\mathbf{H}]] = \mathrm{softmax}\!\left(\frac{[[hidden:Q]] [[hidden:K]]^\top}{\sqrt{[[hyper:d_k]]}}\right) [[hidden:V]]"),
        mo.md(r"""
        The softmax still goes along every row. Real queries and keys are not random, but they start out close to random, and the division keeps the scores in a range where the softmax still lets gradients through.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **scaled dot-product attention** | | attention whose scores are divided by $\sqrt{d_k}$ before the softmax | `torch.softmax(Q @ K.T / math.sqrt(d_k), dim=1) @ V` |
        | $\sqrt{d_k}$ | square root of d k | the spread of the dot product of two random vectors of $d_k$ numbers with spread 1 | `math.sqrt(d_k)` |
        """),
        L.watch_out("A sharp softmax is not a sign that the model knows where to look: with long, unscaled keys it is sharp from the start, before the model has learned anything, and then hardly learns. The scale is not a learned weight either: $\\sqrt{d_k}$ is fixed by the length of the keys."),
        mo.md(r"""
        The picture: random queries, and 16 random keys for each, numbers with spread 1; for each key length, the mean of the largest of the 16 attention weights, over 2,000 queries, seed 0. Without scaling, the largest weight climbs from 0.43 to 0.96: almost one-hot. Scaled, it stays at about 0.25. The gradient of the softmax shrinks with the sharpness: without scaling, it is 6 times smaller at 1,024 numbers than at 4; scaled, it hardly changes.
        """),
        mo.Html(C.sharpness_picture({"title": "The largest of 16 attention weights", "x": "length of the keys, d_k", "y": "mean largest weight",
                                     "plain": "not scaled", "scaled": "divided by √d_k", "even": "all 16 equal: 1/16", "decimal": ".",
                                     "alt": "Two lines against the key length 4, 16, 64, 256 and 1,024. Not scaled, the mean largest attention weight climbs from 0.43 to 0.96. Divided by the square root of d_k, it stays about 0.25. A dashed line at 1/16 marks all weights equal."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Scaled dot-product attention, with one query, key and value per row:

    1. The key length $d_k$: the number of columns of `Q`.
    2. The scores, every query with every key, divided by $\sqrt{d_k}$.
    3. The weights: the softmax of every row.
    4. The new vectors: the weights times the values.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    One random query against 4 random keys of 1,024 numbers each. `_rng.normal(size=(4, 1024))` asks the generator of [Random numbers and seeds](/course/open/np-random?lang=en) for random numbers with mean 0 and standard deviation 1, in an array of that shape. The comments with `#` say which step of the recipe each part is; step 4, the weights times the values, goes as in [Queries, keys and values](/course/open/attn-qkv?lang=en). Step 2 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_scale():
    import math
    import numpy as np

    _rng = np.random.default_rng(1)
    _q = _rng.normal(size=(1, 1024))
    _K = _rng.normal(size=(4, 1024))
    # 1. the key length
    _d_k = _q.shape[1]
    # 2. the scores
    _S = _q @ _K.T  # complete: divided by math.sqrt(_d_k)
    print(np.round(_S, 2))
    # 3. the weights: the softmax of every row
    _S = _S - _S.max(axis=1, keepdims=True)
    _A = np.exp(_S) / np.sum(np.exp(_S), axis=1, keepdims=True)
    print(np.round(_A, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    As it is, the scores run from about −27 to 52, and the softmax gives the largest score all the weight: 1, and the other three 0, to three decimals. A small change in a score changes nothing there: the gradient is about 0. Completed, every score is divided by $\sqrt{1024} = 32$: the scores become −0.86, 1.61, −0.8 and −0.41, and the weights 0.065, 0.765, 0.069 and 0.101. The best key still gets the most, but a small change in a score now changes the weights, so the gradient has something to work with.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_divide():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Dividing too much
    """)
    return


@app.cell(hide_code=True)
def choice_divide_cell():
    choice_divide = L.choice(
        "divide-too-much",
        question="With keys of 1,024 numbers, you divide the scores by $d_k = 1024$ instead of by $\\sqrt{d_k} = 32$. What happens to the attention weights?",
        options={
            "sharper": "They get sharper: one key gets almost all the weight",
            "right": "They get almost even: every key gets about the same weight",
            "same": "Nothing: the softmax gives the same weights for any division",
        },
        correct="right",
        why={
            "sharper": "Dividing by more makes the scores smaller, not larger: their spread becomes $32 / 1024 = 1/32$. Small scores give a flat softmax.",
            "same": "The softmax does not change when you add the same number to every score, but dividing does change it: it squeezes the scores together.",
            "right": "The scores then spread only $1/32$: all of them close to 0, so $e$ to the power of each is about 1, and each of the keys gets about the same share. The model would barely see a difference between keys.",
        },
    )
    choice_divide
    return (choice_divide,)


@app.cell(hide_code=True)
def text_exercise_spread():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Measure the spread

        Complete `score_spread(d_k)`: it makes 10,000 random queries and 10,000 random keys of `d_k` numbers, with mean 0 and spread 1, takes the dot product of every query with the key in the same row, and gives the standard deviation of those scores. Replace the two `...`. The check calls `score_spread(64)`, which should give about `8.08`, and tries other lengths.
        """),
        L.task("Make `score_spread` measure how far the scores of random queries and keys spread."),
    ])
    return


@app.cell(hide_code=True)
def exercise_spread():
    box_spread = L.code_box("score-spread", lines=11)
    box_spread
    return (box_spread,)


@app.cell(hide_code=True)
def check_spread(box_spread):
    result_spread = L.check("score-spread", box_spread.value)
    result_spread
    return (result_spread,)


@app.cell(hide_code=True)
def hints_spread():
    L.hints(
        "score-spread",
        [
            "`q * k` multiplies the two arrays number by number, row by row. A dot product adds up the products of one row: a sum along `axis=1`.",
            "The spread of a column of numbers is its standard deviation, as in [Expected value and variance](/course/open/prob-expectation?lang=en): `np.std`.",
        ],
        why="Every row gives one score, a sum of `d_k` products, and the standard deviation over the 10,000 scores is about $\\sqrt{d_k}$: 8.08 for 64, close to 8, as the spread of a measurement over random numbers is never exact.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_scaled():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Scaled attention

        Complete `scaled_attention(Q, K, V)`: one query, key and value per row; it gives the new vectors of scaled dot-product attention. The first lines turn the lists into tensors and find $d_k$; steps 3 and 4 are given. Replace the `...` following step 2 of the recipe. The check calls `scaled_attention` with 3 words whose queries and keys have 4 numbers, which should give `[[0.9841, 0.1332], [0.1332, 0.9841], [0.6667, 0.6667]]`, and tries other lengths.
        """),
        L.task("Make `scaled_attention` divide the scores by the square root of the key length."),
    ])
    return


@app.cell(hide_code=True)
def exercise_scaled():
    box_scaled = L.code_box("scaled-attention", lines=12)
    box_scaled
    return (box_scaled,)


@app.cell(hide_code=True)
def check_scaled(box_scaled):
    result_scaled = L.check("scaled-attention", box_scaled.value)
    result_scaled
    return (result_scaled,)


@app.cell(hide_code=True)
def hints_scaled():
    L.hints(
        "scaled-attention",
        [
            "Step 2 of the completed worked example, with `Q`, `K` and `d_k`.",
            "`math.sqrt(d_k)` is the square root; the scores are every query with every key, as in [Queries, keys and values](/course/open/attn-qkv?lang=en).",
        ],
        why="With 4 numbers per key, every score is divided by 2: the scores 8, 0 and 4 of the first word become 4, 0 and 2, and its weights 0.87, 0.02 and 0.12 instead of 0.98, 0.00 and 0.02.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The scores of [Queries, keys and values](/course/open/attn-qkv?lang=en) grow like $\sqrt{d_k}$, because their variances add up; too large, they make the softmax saturate, as the sigmoid did in [Vanishing and exploding gradients](/course/open/nn-vanishing-exploding?lang=en). Dividing them by $\sqrt{d_k}$ gives scaled dot-product attention. In the next lesson, *The causal mask*, a language model learns to predict the next word without looking ahead: every word may only take from the words before it.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_divide, result_spread, result_scaled):
    L.wrap_up(choice_divide, result_spread, result_scaled)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
