# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson attn-positions. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("attn-positions", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Multi-head attention](/course/open/attn-multi-head?lang=en) and [Embedding layers](/course/open/nn-embedding-layer?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_projection():
    quick_projection = L.choice(
        "quick-projection",
        question="What does the output projection of multi-head attention do?",
        options={
            "split": "It splits the vectors into heads before the attention",
            "right": "It mixes the outputs of the heads, after they are put side by side",
            "scale": "It divides the scores of every head by the square root of its length",
        },
        correct="right",
        why={
            "split": "The heads take their columns straight from $Q$, $K$ and $V$; the projection comes after them. See [Multi-head attention](/course/open/attn-multi-head?lang=en).",
            "scale": "That is the scaling inside every head, before its softmax. See [Multi-head attention](/course/open/attn-multi-head?lang=en).",
            "right": "A linear layer from $d$ to $d$. Today another kind of vector is added, before the attention: one for every place.",
        },
    )
    quick_projection
    return (quick_projection,)


@app.cell(hide_code=True)
def quick_check_embedding():
    quick_embedding = L.choice(
        "quick-embedding",
        question="How many parameters does `nn.Embedding(50, 8)` have?",
        options={
            "right": "400: a row of 8 numbers for each of the 50 ids",
            "sum": "58: 50 ids and 8 numbers",
            "bias": "408: 400 weights and 8 bias terms",
        },
        correct="right",
        why={
            "sum": "It is a table: every id has a row of its own, of 8 learned numbers. See [Embedding layers](/course/open/nn-embedding-layer?lang=en).",
            "bias": "An embedding layer only looks rows up; it has no bias terms. See [Embedding layers](/course/open/nn-embedding-layer?lang=en).",
            "right": "50 × 8 = 400. Today a second table like it gives every place in the text a row of its own.",
        },
    )
    quick_embedding
    return (quick_embedding,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    "dog bites man" and "man bites dog" are made of the same three words, and mean something else. A model that should understand text needs to know where every word is. But attention compares contents: the scores of a word depend on the embeddings of the words, not on where they stand.
    """)
    return


@app.cell(hide_code=True)
def predict_swap():
    guess_swap = L.guess(
        "swap",
        question="Attention without a mask and without any position information: you swap the first and the last word of a sentence. What happens to the new vectors?",
        options={
            "all": "They all change, as every word sees another order",
            "right": "They swap places too, and nothing else changes",
            "none": "Nothing: every row stays exactly as it was",
        },
        reveal="**They swap places too, and nothing else changes.** Every word compares itself with the same words as before and averages the same embeddings: only the rows come out in another order. So \"dog\" gets the same new vector in \"dog bites man\" and in \"man bites dog\": attention treats a sentence as a bag of words. With the causal mask it is not much better: in one layer, a word sees which words came before it, but not in which order.",
    )
    guess_swap
    return (guess_swap,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Give every place a vector of its own, and add it to the embedding of the token there: $\mathbf{x}_t = \mathbf{e}_\text{token} + \mathbf{p}_t$. The same token at another place now has another vector, so its scores and its new vector depend on where it stands. This is **positional encoding**, and there are two common kinds.

        The original transformer of 2017 used a **sinusoidal positional encoding**: fixed waves of $\sin$ and $\cos$, which go up and down between $-1$ and $1$, starting at $\sin 0 = 0$ and $\cos 0 = 1$, and repeat every $2\pi \approx 6.28$. Column $2j$ of place $t$ gets a sine, and column $2j + 1$ a cosine of the same angle, each pair a little slower than the pair before it:
        """),
        L.formula(r"\begin{aligned} [[hidden:p_{t,\,2j}]] &= \sin\frac{t}{10000^{2j/d}} \\ [[hidden:p_{t,\,2j+1}]] &= \cos\frac{t}{10000^{2j/d}} \end{aligned}"),
        mo.md(r"""
        No two places get the same row. GPT-2 instead has a **learned position embedding**: an embedding layer of 1,024 rows, one for every place up to its longest text. In PyTorch that is a second `nn.Embedding(block, d)`, with `block` the most tokens the model reads at once, and `torch.arange(n)` gives the places 0, 1, ..., n − 1 to look up. Many newer models do not add anything to the embeddings, but change the scores, according to how far apart two places are: ALiBi subtracts a penalty that grows with the distance, and RoPE turns the queries and keys by an angle that depends on the place.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **positional encoding** | | a vector for every place, added to the embedding of the token there, so attention can tell places apart | `tok(ids) + pos(places)` |
        | **sinusoidal positional encoding** | | fixed positions from sine and cosine waves, slower in later columns | `math.sin(angle)` |
        | **learned position embedding** | | a learned table with a row for every place, as GPT-2 has | `nn.Embedding(block, d)` |
        | $\mathbf{p}_t$ | p t | the position vector of place $t$ | `pos(torch.arange(n))[t]` |
        | `torch.arange(n)` | torch arange | the whole numbers 0, 1, ..., n − 1, as a tensor | `torch.arange(3)` |
        """),
        L.watch_out("The position is added, not put next to the token: the same $d$ numbers carry what a token is and where it stands, and training sorts that out. And a learned table has a fixed length: GPT-2 has no row for place 1,024 or later, so it reads at most 1,024 tokens at once."),
        mo.md(r"""
        The picture: three columns of the sinusoids for $d = 16$, over 64 places. Column 0 is a fast wave, column 4 (dashed) one ten times slower, column 8 (the thin line) a hundred times slower. Every place has its own mix of the 16 waves, a bit like the hands of a clock.
        """),
        mo.Html(C.waves_picture({"title": "Sinusoids for d = 16", "x": "place t", "y": "value", "fast": "column 0: fast", "slow": "column 8: slow",
                                 "decimal": ".",
                                 "alt": "Three waves over 64 places: column 0 goes up and down about ten times, column 4 about once, and column 8 rises slowly from 0 to about 0.6."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Positions for a text of $n$ tokens and embeddings of $d$ numbers:

    1. Sinusoids: the angle of place $t$ in columns $i$ and $i + 1$, for every even $i$, is $t / 10000^{i/d}$.
    2. The sine of that angle in column $i$, the cosine in column $i + 1$.
    3. Learned instead: a table of places next to the table of tokens, both of $d$ numbers.
    4. Every token: the embedding of the token plus the embedding of its place.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    "dog bites man" as the ids 2, 7 and 5, and "man bites dog" as 5, 7 and 2, with a small attention without a mask; `torch.no_grad()` because nothing is trained here. It prints how much the new vector of "dog" differs between the two sentences. Step 4 is not finished: run the cell as it is, then complete it where the comments say, and run it again.
    """)
    return


@app.cell
def example_positions():
    import math
    import torch
    from torch import nn

    def _attention(x):
        A = torch.softmax(x @ x.T / math.sqrt(x.shape[1]), dim=1)
        return A @ x

    torch.manual_seed(0)
    # 3. a table of tokens, and next to it a table of places
    _tok = nn.Embedding(10, 4)
    _pos = nn.Embedding(8, 4)
    _dog_bites_man = torch.tensor([2, 7, 5])
    _man_bites_dog = torch.tensor([5, 7, 2])
    with torch.no_grad():
        # 4. every token: the embedding of the token plus that of its place
        _x = _tok(_dog_bites_man)  # complete: + _pos(torch.arange(3))
        _y = _tok(_man_bites_dog)  # complete: + _pos(torch.arange(3))
        print(_attention(_x)[0] - _attention(_y)[2])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    As it is, the difference is `[0., 0., 0., 0.]`: "dog" gets exactly the same new vector at the start of one sentence and at the end of the other. Completed, with the places added, the difference is `[0.4660, -0.3531, -0.2565, -0.0249]`: the vector of "dog" now depends on where it stands, here at place 0 or at place 2.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_too_long():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. A text that is too long
    """)
    return


@app.cell(hide_code=True)
def choice_too_long_cell():
    choice_too_long = L.choice(
        "too-long",
        question="A model has learned position embeddings for 1,024 places, as GPT-2. It gets a text of 1,500 tokens. What happens?",
        options={
            "learns": "It learns rows for the new places as it reads them",
            "again": "It starts again at place 0 after place 1,023",
            "right": "It has no rows for place 1,024 and later",
        },
        correct="right",
        why={
            "learns": "A model learns only while it is trained, not while it reads. The table has 1,024 rows, and that is all it has.",
            "again": "Then two tokens would get the same place, and the model would mix them up. The table simply ends at 1,024.",
            "right": "Looking up row 1,024 of a table of 1,024 rows stops with an error. So it reads at most 1,024 tokens at once: a longer text is cut, or read in parts. This is the context window of the model.",
        },
    )
    choice_too_long
    return (choice_too_long,)


@app.cell(hide_code=True)
def text_exercise_sinusoids():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Sinusoids

        Complete `sinusoids(n, d)`: it gives the sinusoidal positions of $n$ places with $d$ numbers each, as a NumPy array of $n$ by $d$, by steps 1 and 2 of the recipe. Replace the two `...`. The check calls `sinusoids(3, 4)`, which should give `[[0.0, 1.0, 0.0, 1.0], [0.8415, 0.5403, 0.01, 1.0], [0.9093, -0.4161, 0.02, 0.9998]]`, and tries other sizes.
        """),
        L.task("Make `sinusoids` give every place its sine and cosine waves."),
    ])
    return


@app.cell(hide_code=True)
def exercise_sinusoids():
    box_sinusoids = L.code_box("sinusoids", lines=12)
    box_sinusoids
    return (box_sinusoids,)


@app.cell(hide_code=True)
def check_sinusoids(box_sinusoids):
    result_sinusoids = L.check("sinusoids", box_sinusoids.value)
    result_sinusoids
    return (result_sinusoids,)


@app.cell(hide_code=True)
def hints_sinusoids():
    L.hints(
        "sinusoids",
        [
            "The angle is the place divided by a number that grows with the column: `10000 ** (i / d)`, 1 for column 0.",
            "`math.sin(angle)` goes into `pe[t, i]`, and `math.cos(angle)` into the column after it.",
        ],
        why="With $d = 4$, columns 0 and 1 turn with the angle $t$, and columns 2 and 3 with $t / 100$: place 1 gives $\\sin 1 \\approx 0.8415$ and $\\cos 1 \\approx 0.5403$, then $\\sin 0.01 \\approx 0.01$ and $\\cos 0.01 \\approx 1$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_tokens_positions():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Tokens and places

        Complete the class `TokensAndPositions(vocab, d, block)`: in `__init__` a table `self.tok` with a row of $d$ numbers for every token of the vocabulary, and then a table `self.pos` with a row for every place up to `block`; in `forward` the embeddings of a tensor of token ids `ids`, one text, plus those of their places. The check makes the module after `torch.manual_seed(2)` and gives it token 3, token 1 and token 3 again: the first and the last row should now differ.
        """),
        L.task("Make `TokensAndPositions` add the embedding of every place to the embedding of its token."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tokens_positions():
    box_tokens_positions = L.code_box("tokens-and-positions", lines=12)
    box_tokens_positions
    return (box_tokens_positions,)


@app.cell(hide_code=True)
def check_tokens_positions(box_tokens_positions):
    result_tokens_positions = L.check("tokens-and-positions", box_tokens_positions.value)
    result_tokens_positions
    return (result_tokens_positions,)


@app.cell(hide_code=True)
def hints_tokens_positions():
    L.hints(
        "tokens-and-positions",
        [
            "Two `nn.Embedding` layers, as in [Embedding layers](/course/open/nn-embedding-layer?lang=en): one of `vocab` rows and one of `block` rows, both with `d` numbers.",
            "The places of a text of `len(ids)` tokens are `torch.arange(len(ids))`; look them up in `self.pos`, and add the result to `self.tok(ids)`.",
        ],
        why="Both tables give one row of $d$ numbers per token, so they can be added number by number. Token 3 gets the same token row at place 0 and at place 2, but another place row, so its vector differs.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The tables of [Embedding layers](/course/open/nn-embedding-layer?lang=en), now also for places, give the multi-head attention of [Multi-head attention](/course/open/attn-multi-head?lang=en) what it could not see: where every token stands. Your `TokensAndPositions` is the input of a GPT. In the next lesson, *The transformer block*, attention gets a small MLP after it, both with layer normalization and a residual connection: the block that a GPT stacks again and again.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_too_long, result_sinusoids, result_tokens_positions):
    L.wrap_up(choice_too_long, result_sinusoids, result_tokens_positions)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
