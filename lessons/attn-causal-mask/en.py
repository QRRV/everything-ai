# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson attn-causal-mask. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("attn-causal-mask", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Scaling the scores](/course/open/attn-scale?lang=en) and [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_scale():
    quick_scale = L.choice(
        "quick-scale",
        question="Why does scaled dot-product attention divide its scores by $\\sqrt{d_k}$?",
        options={
            "sum": "So that the weights of every row add up to 1",
            "right": "So that the scores do not grow with the key length, and the softmax does not saturate",
            "speed": "So that the matrix multiplication is faster",
        },
        correct="right",
        why={
            "sum": "The softmax already makes every row add up to 1, with or without the division. See [Scaling the scores](/course/open/attn-scale?lang=en).",
            "speed": "The division makes nothing faster; it keeps the scores at a spread of about 1. See [Scaling the scores](/course/open/attn-scale?lang=en).",
            "right": "Today you change the scores once more, before the same softmax: some of them become minus infinity.",
        },
    )
    quick_scale
    return (quick_scale,)


@app.cell(hide_code=True)
def quick_check_next_token():
    quick_next_token = L.choice(
        "quick-next-token",
        question="What does a language model give back for the text so far?",
        options={
            "right": "A probability for every possible next token",
            "text": "The rest of the text, all at once",
            "one": "The one token that comes next, without probabilities",
        },
        correct="right",
        why={
            "text": "It predicts one token at a time; writing more means asking again with the new token added. See [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en).",
            "one": "It gives a probability for every possible next token; picking one is a separate step. See [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en).",
            "right": "And to train it, every place of a text is an example: the tokens so far, and the token that comes next.",
        },
    )
    quick_next_token
    return (quick_next_token,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A GPT learns from every place of a text at once: at position $i$, it predicts token $i + 1$. With the self-attention of the previous lessons, position $i$ takes from every position, including $i + 1$, the very token it has to predict. Training is then too easy: the model learns to copy the answer from the next position. And when it writes, there is no next position yet. Every position may only use itself and what came before it.
    """)
    return


@app.cell(hide_code=True)
def predict_zeros():
    guess_zeros = L.guess(
        "zeros",
        question="A text of 4 tokens goes through attention in which every position may only take from itself and the positions before it. Of the 4 × 4 = 16 attention weights, how many are 0?",
        options={
            "four": "4: one forbidden token for every position",
            "eight": "8: half of them",
            "right": "6: everything above the diagonal",
        },
        reveal="**6.** Position 1 may not take from positions 2, 3 and 4, position 2 not from 3 and 4, position 3 not from 4: 3 + 2 + 1 = 6, everything above the diagonal. Position 4 takes from all four. For $n$ tokens, $n(n - 1)/2$ weights are 0: for 1,024 tokens, almost half.",
    )
    guess_zeros
    return (guess_zeros,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        The **causal mask** hides the future of every position. Before the softmax, every score of a later position is set to $-\infty$: $e^{-\infty} = 0$, so its weight is exactly 0, and the softmax shares the whole weight of the row among the positions that remain. With $M_{ij} = 0$ for $j \le i$ and $-\infty$ for $j > i$:
        """),
        L.formula(r"[[hidden:\mathbf{H}]] = \mathrm{softmax}\!\left(\frac{[[hidden:Q]] [[hidden:K]]^\top}{\sqrt{[[hyper:d_k]]}} + M\right) [[hidden:V]]"),
        mo.md(r"""
        The first position only sees itself, and the last sees everything. A transformer with this mask is a *decoder*, like GPT. In PyTorch, `torch.tril(torch.ones(n, n))` gives an n by n matrix with ones on and below the diagonal, and zeros above it; `== 0` turns the zeros into `True`: the future. `S.masked_fill(future, float("-inf"))` gives a copy of `S` with $-\infty$ wherever `future` is `True`; `float("-inf")` is minus infinity as a Python number.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **causal mask** | | sets the scores of later positions to $-\infty$ before the softmax, so every position only takes from itself and the positions before it | `S.masked_fill(future, float("-inf"))` |
        | $M$ | M | the mask: 0 where a position may look, $-\infty$ where it may not | |
        | `torch.tril` | torch tril | the lower triangle of a matrix: the numbers on and below the diagonal, zeros above it | `torch.tril(A)` |
        | `masked_fill` | masked fill | a copy of a tensor with one value wherever a mask of `True` and `False` is `True` | `S.masked_fill(future, float("-inf"))` |
        """),
        L.watch_out("Mask the scores before the softmax, not the weights after it: weights set to 0 afterwards leave rows that no longer add up to 1. The mask also does not hide a word from itself: the diagonal stays, or the first position would see nothing at all."),
        mo.md(r"""
        The picture: the attention weights of "the cat sat down", with made-up embeddings of 2 numbers as queries, keys and values, and the causal mask. Above the diagonal every weight is 0. "the" takes everything from itself; "down" takes from all four words, the same weights as without the mask, as there is no future after it.
        """),
        C.mask_picture({"title": "Causal attention of \"the cat sat down\"", "rows": "word", "cols": "takes from", "decimal": ".",
                        "alt": "A matrix of 4 by 4 attention weights with zeros above the diagonal. the takes 1 from itself; cat takes 0.33 from the and 0.67 from itself; sat takes 0.25, 0.25 and 0.50; down takes 0.31, 0.15, 0.22 and 0.31."}),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Causal attention, with one query, key and value per row, for $n$ positions:

    1. The scores: every query with every key, divided by $\sqrt{d_k}$.
    2. The future: `True` above the diagonal of an n by n matrix, where `torch.tril` leaves zeros in a matrix of ones.
    3. The scores of the future set to $-\infty$, with `masked_fill`.
    4. The softmax of every row, and the weights times the values.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    "the cat sat down", with the made-up embeddings of the picture. The comments with `#` say which step of the recipe each part is. Step 3 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_mask():
    import math
    import torch

    _X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.5]])
    _n, _d_k = _X.shape
    # 1. the scores
    _S = _X @ _X.T / math.sqrt(_d_k)
    # 2. the future
    _future = torch.tril(torch.ones(_n, _n)) == 0
    print(_future)
    # 3. the scores of the future set to minus infinity
    _S = _S  # complete: _S.masked_fill(_future, float("-inf"))
    # 4. the weights
    print(torch.softmax(_S, dim=1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `_future` is `True` above the diagonal: 6 places. As it is, every row has 4 weights above 0: "the" takes 0.313 from "sat", which comes after it. Completed, the scores above the diagonal are $-\infty$, and the weights there are 0: "the" takes 1 from itself, "cat" 0.330 from "the" and 0.670 from itself, "sat" 0.248, 0.248 and 0.503. The last row, "down", is the same with and without the mask.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_infinity():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Why minus infinity
    """)
    return


@app.cell(hide_code=True)
def choice_infinity_cell():
    choice_infinity = L.choice(
        "minus-infinity",
        question="Why does the mask set the scores of the future to $-\\infty$, and not to 0?",
        options={
            "zero-ok": "0 would work as well: a score of 0 means no match",
            "right": "A score of 0 still gets $e^0 = 1$ in the softmax, so the future would still count",
            "faster": "$-\\infty$ makes the softmax faster to compute",
        },
        correct="right",
        why={
            "zero-ok": "A score of 0 is an ordinary score: next to scores around 0, it gets about as much weight as the others.",
            "faster": "The softmax costs the same; what changes is that $e^{-\\infty} = 0$ makes those weights exactly 0.",
            "right": "Only $e^{-\\infty} = 0$ gives a weight of exactly 0, whatever the other scores are. A very negative number, such as $-10^9$, works almost as well.",
        },
    )
    choice_infinity
    return (choice_infinity,)


@app.cell(hide_code=True)
def text_exercise_future():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The future of every position

        Complete `future_mask(n)`: it gives a tensor of n by n with `True` where column `j` comes after row `i`, and `False` on and below the diagonal. Replace the `...` following step 2 of the recipe. The check calls `future_mask(3)`, which should give `[[False, True, True], [False, False, True], [False, False, False]]`, and tries other sizes.
        """),
        L.task("Make `future_mask` mark the future of every position with True."),
    ])
    return


@app.cell(hide_code=True)
def exercise_future():
    box_future = L.code_box("future-mask", lines=6)
    box_future
    return (box_future,)


@app.cell(hide_code=True)
def check_future(box_future):
    result_future = L.check("future-mask", box_future.value)
    result_future
    return (result_future,)


@app.cell(hide_code=True)
def hints_future():
    L.hints(
        "future-mask",
        [
            "`torch.tril` keeps the ones on and below the diagonal; above it are the zeros, and the zeros are the future.",
            "Step 2 of the worked example, with `n` instead of `_n`.",
        ],
        why="`torch.tril(torch.ones(n, n))` has ones where a position may look, itself and before; `== 0` turns the rest, above the diagonal, into `True`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_causal():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Causal attention

        Complete `causal_attention(Q, K, V)`: one query, key and value per row; it gives the new vectors of scaled dot-product attention with the causal mask. The first lines make tensors of the lists and work out the scaled scores; step 4 is given. Replace the `...` following steps 2 and 3 of the recipe. The check calls `causal_attention` with 3 positions whose queries and keys are `[[2, 0], [0, 2], [1, 1]]` and whose values are `[[1, 0], [0, 1], [0.5, 0.5]]`, which should give `[[1, 0], [0.0558, 0.9442], [0.5, 0.5]]`, and tries other sizes.
        """),
        L.task("Make `causal_attention` hide the future of every position before the softmax."),
    ])
    return


@app.cell(hide_code=True)
def exercise_causal():
    box_causal = L.code_box("causal-attention", lines=11)
    box_causal
    return (box_causal,)


@app.cell(hide_code=True)
def check_causal(box_causal):
    result_causal = L.check("causal-attention", box_causal.value)
    result_causal
    return (result_causal,)


@app.cell(hide_code=True)
def hints_causal():
    L.hints(
        "causal-attention",
        [
            "Steps 2 and 3 of the completed worked example in one line, with `S` and `n`.",
            "`masked_fill` gets the mask of the future and the value it puts there: minus infinity, `float(\"-inf\")`.",
        ],
        why="The first position only sees itself, so it keeps its own value, `[1, 0]`. The second takes from the first a little: its score with itself is 2.83, with the first 0, so 0.944 of its weight goes to itself.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The scaled dot-product attention of [Scaling the scores](/course/open/attn-scale?lang=en), with the future hidden, so that a model can learn next-token prediction of [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en) from every place of a text at once, without seeing the answers. In the next lesson, *Multi-head attention*, several smaller attentions run side by side, each free to look for something else.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_infinity, result_future, result_causal):
    L.wrap_up(choice_infinity, result_future, result_causal)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
