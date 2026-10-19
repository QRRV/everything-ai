# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson attn-scores. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("attn-scores", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Attention: every word looks at the others](/course/open/attn-idea?lang=en) and [Floating point and the log-sum-exp trick](/course/open/num-floats-stability?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_new_vector():
    quick_new_vector = L.choice(
        "quick-new-vector",
        question="In attention, how does a word decide how much to take from another word?",
        options={
            "right": "From the dot product of their embeddings, through a softmax",
            "best": "From how close the two words are in the sentence",
            "sum": "From a weight the model learned for that pair of words",
        },
        correct="right",
        why={
            "best": "Attention looks at the embeddings, not at the places: a word at the other end of the sentence can count just as much. See [Attention: every word looks at the others](/course/open/attn-idea?lang=en).",
            "sum": "The attention weights are not learned: the softmax works them out anew for every sentence, from the dot products. See [Attention: every word looks at the others](/course/open/attn-idea?lang=en).",
            "right": "Today you work out those weights for all words of a sentence at once, as one matrix.",
        },
    )
    quick_new_vector
    return (quick_new_vector,)


@app.cell(hide_code=True)
def quick_check_shift():
    quick_shift = L.choice(
        "quick-shift",
        question="Before taking e to the power of a list of numbers, you subtract the largest of them from every number. What does that change in the softmax of the list?",
        options={
            "smaller": "The probabilities get smaller: all numbers went down",
            "largest": "The largest number gets probability 0: it became 0",
            "right": "Nothing: only the overflow of exp is gone",
        },
        correct="right",
        why={
            "smaller": "Every $e^{z}$ is divided by the same $e^{m}$, above and below the line, so the fractions stay the same. See [Floating point and the log-sum-exp trick](/course/open/num-floats-stability?lang=en).",
            "largest": "$e^0$ is 1, not 0: the largest number gets the largest share, as before. See [Floating point and the log-sum-exp trick](/course/open/num-floats-stability?lang=en).",
            "right": "$e^{z_j - m} / \\sum_k e^{z_k - m}$ is $e^{z_j} / \\sum_k e^{z_k}$, as $e^{-m}$ cancels above and below the line, and no $e$ to the power of more than 0 is left to overflow.",
        },
    )
    quick_shift
    return (quick_shift,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Attention: every word looks at the others](/course/open/attn-idea?lang=en), you worked out the new vector of one word. A language model does that for every token of its text at once, and does it again in every layer: for the smallest GPT-2, with up to 1,024 tokens, at every one of its 12 layers. One word at a time in a Python loop is far too slow; written with matrices, it is two matrix multiplications and a softmax. And real embeddings hold hundreds of numbers, so their dot products can run into the hundreds or thousands; from about 710 on, $e$ to the power of a score overflows, and the softmax gives nan.
    """)
    return


@app.cell(hide_code=True)
def predict_count():
    guess_count = L.guess(
        "count",
        question="A text of 1,024 tokens goes through attention once, as in this lesson: every token takes a weighted average of all tokens. How many attention weights does that take?",
        options={
            "per-token": "1,024: one weight for every token",
            "right": "1,048,576: one for every pair of tokens",
            "neighbors": "3,072: one for every token and its two neighbors",
        },
        reveal="**1,048,576**, which is 1,024 × 1,024. Every token has a weight for every token, itself included, so the weights fill a matrix with a row and a column per token. Twice as much text gives four times as many weights: one reason why long texts make a transformer slow and use a lot of memory.",
    )
    guess_count
    return (guess_count,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Put the embeddings of the $n$ words of a sentence in a matrix $\mathbf{X}$, one row per word, $d$ numbers each: shape $(n, d)$. Then $\mathbf{X}\mathbf{X}^\top$ holds the dot product of every word with every word, the **attention scores**: number $(i, j)$ is $\mathbf{x}_i \cdot \mathbf{x}_j$, and the shape is $(n, n)$. The softmax of every row gives the **attention weights** $\mathbf{A}$: row $i$ says how much word $i$ takes from each word, and adds up to 1. One more matrix multiplication gives the new vectors of all words, $\mathbf{H}$, of shape $(n, d)$:
        """),
        L.formula(r"\begin{aligned} [[hidden:\mathbf{S}]] &= [[hidden:\mathbf{X}]] [[hidden:\mathbf{X}]]^\top \\ [[hidden:\mathbf{A}]] &= \mathrm{softmax}([[hidden:\mathbf{S}]]) \text{, per row} \\ [[hidden:\mathbf{H}]] &= [[hidden:\mathbf{A}]] [[hidden:\mathbf{X}]] \end{aligned}"),
        mo.md(r"""
        Row $i$ of $\mathbf{A}\mathbf{X}$ is $\sum_j A_{ij} \mathbf{x}_j$: the weighted average of [Attention: every word looks at the others](/course/open/attn-idea?lang=en), for word $i$. Before the softmax, subtract the largest score of every row from that row, as with the log-sum-exp in [Floating point and the log-sum-exp trick](/course/open/num-floats-stability?lang=en): the weights stay the same, and no $e$ to the power of a score can overflow any more. In NumPy, `S.max(axis=1, keepdims=True)` gives the largest number of every row as a column, so that every row subtracts its own.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **attention score** | | how well two words match: the dot product of their embeddings | `X @ X.T` |
        | **attention weight** | | how much one word takes from another: the softmax of a row of scores, so each row is positive and adds up to 1 | `A[i, j]` |
        | $\mathbf{S}$ | capital S | the attention scores, one row and one column per word: shape $(n, n)$ | `S` |
        | $\mathbf{A}$ | capital A | the attention weights, one row and one column per word: shape $(n, n)$ | `A` |
        | $\mathbf{H}$ | capital H | the new vectors, one row per word: shape $(n, d)$ | `A @ X` |
        """),
        L.watch_out("$\\mathbf{S}$ is symmetric, as $\\mathbf{x}_i \\cdot \\mathbf{x}_j = \\mathbf{x}_j \\cdot \\mathbf{x}_i$, but $\\mathbf{A}$ is not: every row is divided by its own sum. And the softmax goes along every row, `axis=1`: over the whole matrix, or down the columns, the weights of a word no longer add up to 1."),
        mo.md(r"""
        The picture: the attention weights of "the river bank", with the made-up embeddings of [Attention: every word looks at the others](/course/open/attn-idea?lang=en), whose two numbers say how much a word is about money and how much about water. Every row adds up to 1. "river" takes 0.84 from itself, "the" takes 0.38 from "river", but "river" takes only 0.04 from "the".
        """),
        C.weights_picture({"title": "Attention weights of \"the river bank\"", "rows": "word", "cols": "takes from", "decimal": ".",
                           "alt": "A matrix of 3 by 3 attention weights. the takes 0.23 from the, 0.38 from river and 0.38 from bank; river takes 0.04, 0.84 and 0.11; bank takes 0.16, 0.42 and 0.42."}),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Attention for all words at once, with one embedding per row of `X`:

    1. The scores: `X @ X.T`, the dot product of every word with every word.
    2. Subtract from every row its largest score: `S.max(axis=1, keepdims=True)`.
    3. The weights: e to the power of every score, divided by the sum of its row.
    4. The new vectors: the weights times `X`, one row per word.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    "the river bank" with the made-up embeddings, each 20 times as large: as with embeddings of many numbers, the scores run into the hundreds. The comments with `#` say which step of the recipe each part is. Step 2 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_scores():
    import numpy as np

    _X = 20 * np.array([[0.5, 0.5], [0.0, 2.0], [1.0, 1.0]])
    # 1. the scores of every word with every word
    _S = _X @ _X.T
    print(_S)
    # 2. the largest score of every row subtracted
    _S = _S  # complete: _S - _S.max(axis=1, keepdims=True)
    # 3. the weights: the softmax of every row
    _A = np.exp(_S) / np.sum(np.exp(_S), axis=1, keepdims=True)
    print(np.round(_A, 3))
    # 4. the new vectors
    _H = _A @ _X
    print(np.round(_H, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The scores run up to 1,600, for "river" with itself. As it is, $e^{1600}$ and $e^{800}$ overflow to `inf`, NumPy warns about it, and `inf / inf` is `nan`: the weights of "river" and "bank" become `nan` where a score overflowed, and their new vectors become `nan` entirely. Only "the", whose largest score is 400, still works. Completed, every row subtracts its largest score first: the weights become `[0. 0.5 0.5]`, `[0. 1. 0.]` and `[0. 0.5 0.5]`, and the new vectors `[10. 30.]`, `[0. 40.]` and `[10. 30.]`. Scores this large make the softmax very sharp: almost all weight goes to the best matches. The lesson after next, *Scaling the scores*, makes them smaller on purpose.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_symmetric():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Symmetric or not
    """)
    return


@app.cell(hide_code=True)
def choice_symmetric_cell():
    choice_symmetric = L.choice(
        "symmetric",
        question="The scores are symmetric: $S_{ij} = S_{ji}$. Are the attention weights symmetric too, $A_{ij} = A_{ji}$?",
        options={
            "yes": "Yes: the softmax keeps every pair of scores in step",
            "same-word": "Only on the diagonal, where a word meets itself",
            "right": "No: every row is divided by its own sum",
        },
        correct="right",
        why={
            "yes": "$A_{ij}$ is $e^{S_{ij}}$ divided by the sum of row $i$, and $A_{ji}$ the same number divided by the sum of row $j$. In the picture, \"the\" takes 0.38 from \"river\", \"river\" 0.04 from \"the\".",
            "same-word": "The diagonal compares a weight with itself: $A_{ii} = A_{ii}$ says nothing. Off the diagonal, row $i$ and row $j$ usually have different sums, so $A_{ij}$ and $A_{ji}$ differ: in the picture, 0.38 and 0.04.",
            "right": "$A_{ij}$ and $A_{ji}$ share the score $S_{ij}$, but row $i$ and row $j$ have their own sums. A word with strong matches elsewhere takes less from each word.",
        },
    )
    choice_symmetric
    return (choice_symmetric,)


@app.cell(hide_code=True)
def text_exercise_softmax_rows():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A softmax for every row

        Complete `softmax_rows(S)`: it gives the softmax of every row of the matrix `S`, without overflow, also for scores of a thousand or more. Replace the two `...` following steps 2 and 3 of the recipe. The check calls `softmax_rows([[1, 2, 3], [1000, 1000, 1001]])`, which should give `[[0.09, 0.2447, 0.6652], [0.2119, 0.2119, 0.5761]]`, and tries other matrices, also ones that are not square.
        """),
        L.task("Make `softmax_rows` give the softmax of every row of a matrix, without overflow."),
    ])
    return


@app.cell(hide_code=True)
def exercise_softmax_rows():
    box_softmax_rows = L.code_box("softmax-rows", lines=9)
    box_softmax_rows
    return (box_softmax_rows,)


@app.cell(hide_code=True)
def check_softmax_rows(box_softmax_rows):
    result_softmax_rows = L.check("softmax-rows", box_softmax_rows.value)
    result_softmax_rows
    return (result_softmax_rows,)


@app.cell(hide_code=True)
def hints_softmax_rows():
    L.hints(
        "softmax-rows",
        [
            "Steps 2 and 3 of the completed worked example, with `S` instead of `_S`.",
            "The largest number of every row, as a column: `S.max(axis=1, keepdims=True)`. Without `keepdims`, NumPy lines the flat result up with the columns instead of the rows.",
        ],
        why="Subtracting the row's own largest score leaves every fraction the same, and leaves $e^0 = 1$ as the largest exp, so nothing overflows. The second row of the example, 1,000, 1,000 and 1,001, becomes −1, −1 and 0: the same weights as for 0, 0 and 1.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_attend_all():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. All words at once

        Complete `attend_all(X)`: `X` holds one embedding per row; the function gives the new vectors of all words, one per row, by the recipe. Steps 2 and 3 are given. Replace the two `...` following steps 1 and 4. The check calls `attend_all([[1, 0], [0, 1], [1, 1]])`, the sentence of the exercise in [Attention: every word looks at the others](/course/open/attn-idea?lang=en), which should give `[[0.8446, 0.5777], [0.5777, 0.8446], [0.7881, 0.7881]]`, and tries other sentences, also of one word and with 3 numbers per word.
        """),
        L.task("Make `attend_all` give the new vectors of all words at once."),
    ])
    return


@app.cell(hide_code=True)
def exercise_attend_all():
    box_attend_all = L.code_box("self-attention", lines=11)
    box_attend_all
    return (box_attend_all,)


@app.cell(hide_code=True)
def check_attend_all(box_attend_all):
    result_attend_all = L.check("self-attention", box_attend_all.value)
    result_attend_all
    return (result_attend_all,)


@app.cell(hide_code=True)
def hints_attend_all():
    L.hints(
        "self-attention",
        [
            "Step 1 gives a matrix with a row and a column for every word, so its shape is $(n, n)$: which order of `X` and `X.T` gives that?",
            "Step 4: row $i$ of the result is the weighted average with the weights of row $i$ of `A`. Which matrix multiplication of `A` and `X` takes, for every row of `A`, the rows of `X`?",
        ],
        why="`X @ X.T` has shape $(n, d) \\times (d, n) = (n, n)$; `A @ X` has shape $(n, n) \\times (n, d) = (n, d)$: one new vector per word, as long as an embedding. Row 0 of the example is what `attend` gave for word 0.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The weighted average of [Attention: every word looks at the others](/course/open/attn-idea?lang=en), for all words at once with the matrix multiplication of [A whole batch at once: matrix multiplication](/course/open/math-matrix-multiply?lang=en), and kept from overflowing with the trick of [Floating point and the log-sum-exp trick](/course/open/num-floats-stability?lang=en). So far a word compares its embedding with the embeddings of the others, and passes on its embedding as it is. In the next lesson, *Queries, keys and values*, every word gets three learned vectors instead: one to look with, one to be found by, and one to pass on.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_symmetric, result_softmax_rows, result_attend_all):
    L.wrap_up(choice_symmetric, result_softmax_rows, result_attend_all)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
