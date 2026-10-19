# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson gpt-count. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("gpt-count", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Assemble your GPT](/course/open/gpt-assemble?lang=en) and [Too many weights to nudge one by one](/course/open/nn-why-backprop?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_vocab():
    quick_vocab = L.choice(
        "quick-vocab",
        question="Which two parts of your GPT grow with the size of the vocabulary?",
        options={
            "places": "The table of places and the output head",
            "right": "The table of tokens and the output head",
            "blocks": "The table of tokens and the blocks",
        },
        correct="right",
        why={
            "places": "The table of places has a row for every place, `block` of them, whatever the vocabulary. See [Assemble your GPT](/course/open/gpt-assemble?lang=en).",
            "blocks": "A block works on vectors of d numbers; it does not know how many tokens there are. See [Assemble your GPT](/course/open/gpt-assemble?lang=en).",
            "right": "A row for every token in the table, and a score for every token from the head. Today you count how big they are.",
        },
    )
    quick_vocab
    return (quick_vocab,)


@app.cell(hide_code=True)
def quick_check_flops():
    quick_flops = L.choice(
        "quick-flops",
        question="A computer twice as fast does the same forward pass of a network. What happens to the number of FLOPs of that forward pass?",
        options={
            "half": "It halves: the computer needs half the time",
            "double": "It doubles: the computer does twice as many per second",
            "right": "It stays the same: FLOPs count the work, not the time",
        },
        correct="right",
        why={
            "half": "The time halves, but the network needs as many multiplications and additions as before. See [Too many weights to nudge one by one](/course/open/nn-why-backprop?lang=en).",
            "double": "Twice as many per second is the speed of the computer; the forward pass itself needs the same calculations. See [Too many weights to nudge one by one](/course/open/nn-why-backprop?lang=en).",
            "right": "Today you count them for your GPT: per token, for the parameters and for attention.",
        },
    )
    quick_flops
    return (quick_flops,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Before you train a model, you want to know two things: how big it is, and what one token costs. Together they decide how long training takes, and what every answer of the model costs. GPT-3 has 175 billion parameters: reading one token costs it about 350 billion FLOPs. You can work out both numbers before you run anything, from the sizes of the model: count where the parameters sit, count about 2 FLOPs per parameter for every token, and add what attention costs, which grows with the length of the text.
    """)
    return


@app.cell(hide_code=True)
def predict_share():
    guess_share = L.guess(
        "blocks-share",
        question="A GPT with the sizes of the smallest GPT-2, a vocabulary of 50,257 tokens, $d = 768$ and 12 blocks, has an output head of its own. What share of its parameters do you think sits in the 12 blocks?",
        options={
            "most": "About 95%: the blocks are the model",
            "right": "About half",
            "few": "About 10%",
        },
        reveal="**About half**: 85 million of 163 million. The table of tokens and the output head each have 50,257 × 768, about 38.6 million, a quarter of the model each. GPT-2 itself shares one matrix between the table of tokens and the output head, and has 124 million parameters, 68% of them in the blocks.",
    )
    guess_share
    return (guess_share,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        The **parameter count** of your GPT, part by part, for a vocabulary of $|\mathcal{V}|$ tokens, vectors of $d$ numbers, $L$ blocks and texts of at most `block` tokens:

        - the table of tokens, $|\mathcal{V}| \cdot d$, and the table of places, `block` $\cdot\, d$;
        - every block: attention, $4d^2 + d$ (query, key, value and the output projection, which has bias terms); two layer normalizations, $2d$ each (a $\gamma$ and a $\beta$ for every number); and the MLP, $8d^2 + 5d$ (from $d$ to $4d$ and back, with bias terms);
        - the last layer normalization, $2d$, and the output head, $d \cdot |\mathcal{V}| + |\mathcal{V}|$.

        For a big $d$, a block is about $12d^2$, and the whole model:
        """),
        L.formula(r"\begin{gathered} N \approx 12\,L\,d^2 + 2\,|\mathcal{V}|\,d \\ \text{FLOPs} \approx 2N + 2\,L\,n\,d \end{gathered}"),
        mo.md(r"""
        **FLOPs per token.** As in [Too many weights to nudge one by one](/course/open/nn-why-backprop?lang=en), a forward pass uses every parameter once per token, for a multiplication and an addition: about $2N$ FLOPs. Attention adds a part that grows with the length $n$ of the text. In every block, a token at place $t$ compares its query with the keys of the $t$ tokens up to it, and mixes their values: about $4\,t\,d$ FLOPs. Through the causal mask, a token of a text of $n$ tokens looks on average at about $n/2$ of them, so attention costs about $2\,L\,n\,d$ per token, the second part of the formula. Your attention of [The causal mask](/course/open/attn-causal-mask?lang=en) still computes the scores of the later tokens and only then hides them, so it does about twice that: the formula counts the work a token needs. Training also needs the backward pass, which costs about twice the forward pass: about three times the forward FLOPs per token, about $6N$ for a short text.

        ### New words and symbols

        | Word or symbol | What it means |
        |---|---|
        | **parameter count** | how many numbers a model learns, part by part: 124 million in GPT-2 |
        | **FLOPs per token** | the calculations for one token: about $2N$ and attention forward, three times that to train |
        | $n$ | the number of tokens of the text, at most `block` |
        """),
        L.watch_out("Not every parameter costs FLOPs: looking up a row of a table multiplies nothing. The rule of $2N$ counts the tables anyway, so for a small model with a big vocabulary it guesses too high. And $6N$ is per token: a mini-batch of 64 windows of 256 tokens is 16,384 tokens, at $6N$ each."),
        mo.md(r"""
        The picture: where the parameters sit, as a share of each model. In your small GPT, almost everything is in the blocks; with the sizes of GPT-2, the table of tokens and the output head each take a quarter.
        """),
        mo.Html(C.parts_picture({"title": "Where the parameters sit", "small": "your GPT of the last lesson: {n}", "gpt2": "the sizes of GPT-2: {n}",
                                 "tok": "table of tokens", "pos": "table of places", "blocks": "blocks", "head": "output head", "lang": "en",
                                 "alt": "Two groups of four bars, one for each part. Your GPT of the last lesson, 211,777 parameters: the table of tokens 2%, the table of places 2%, the blocks 94%, the output head 2%. The sizes of GPT-2, 163,059,793 parameters: the table of tokens 24%, the table of places under 1%, the blocks 52%, the output head 24%."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    The size of a GPT, and what one token costs it:

    1. The tables: $|\mathcal{V}| \cdot d$ for the tokens, `block` $\cdot\, d$ for the places.
    2. Every block: attention $4d^2 + d$, two layer normalizations of $2d$, and the MLP $8d^2 + 5d$; times $L$.
    3. At the end: $2d$ for the last layer normalization, and $d \cdot |\mathcal{V}| + |\mathcal{V}|$ for the output head.
    4. One token forward: about $2N$, and $2\,L\,n\,d$ for attention in a text of $n$ tokens.
    5. Training on that token: about three times the FLOPs of step 4.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    GPT-2 small: its parameters, and the FLOPs of training on one token of a text of 1,024 tokens. GPT-2 shares one matrix between its table of tokens and its output head, so its end is only the last layer normalization. The comments with `#` say which step of the recipe each part is. Step 4 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_count():
    _V, _d, _L, _block = 50257, 768, 12, 1024
    # 1. the tables
    _tables = _V * _d + _block * _d
    # 2. the blocks: attention, two layer normalizations, the MLP
    _blocks = _L * ((4 * _d * _d + _d) + 2 * (2 * _d) + (8 * _d * _d + 5 * _d))
    # 3. the end: GPT-2's output head is its table of tokens
    _end = 2 * _d
    _N = _tables + _blocks + _end
    print(_tables, _blocks, _end, _N)
    # 4. one token forward, in a text of 1,024 tokens
    _forward = 2 * _N  # complete: + 2 * _L * _block * _d, for attention
    print(_forward)
    # 5. training on that token: forward, and backward at about twice the cost
    print(3 * _forward)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    GPT-2 small has 124,412,160 parameters by this count, 39 million in the tables and 85 million in the blocks: 27,648 fewer than GPT-2 itself, as its queries, keys and values have bias terms too. As it is, the cell counts $2N$: 248,824,320 FLOPs to read one token, and three times that to train on it. Completed, attention adds 18,874,368 for a text of 1,024 tokens, 7.6% more: 803,096,064 FLOPs to train on one token.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_flops():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. One token, forward
    """)
    return


@app.cell(hide_code=True)
def choice_flops_cell():
    choice_flops = L.choice(
        "forward-flops",
        question="The GPT you will train on a book has about 10.8 million parameters. About how many FLOPs does it need to read one token, forward?",
        options={
            "right": "About 22 million: 2 per parameter",
            "one": "About 10.8 million: one per parameter",
            "six": "About 65 million: 6 per parameter",
        },
        correct="right",
        why={
            "one": "Every parameter costs two FLOPs per token: a multiplication and an addition.",
            "six": "6 per parameter is training, forward and backward. Reading a token is the forward pass alone.",
            "right": "2 × 10.8 million, and a little more for attention: 2 · 6 · 256 · 384, about 1.2 million, for windows of 256 tokens. Training costs three times that per token.",
        },
    )
    choice_flops
    return (choice_flops,)


@app.cell(hide_code=True)
def text_exercise_params():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Count your GPT

        Complete `gpt_params(vocab, d, n_layers, block)`: it gives how many parameters your GPT of [Assemble your GPT](/course/open/gpt-assemble?lang=en) has, steps 1 to 3 of the recipe. Your GPT has an output head of its own, with bias terms. The check calls `gpt_params(65, 64, 4, 64)`, which should give `211777`, as PyTorch counts your GPT of the last lesson, and tries other sizes.
        """),
        L.task("Make `gpt_params` count the parameters of your GPT, part by part."),
    ])
    return


@app.cell(hide_code=True)
def exercise_params():
    box_params = L.code_box("gpt-params", lines=11)
    box_params
    return (box_params,)


@app.cell(hide_code=True)
def check_params(box_params):
    result_params = L.check("gpt-params", box_params.value)
    result_params
    return (result_params,)


@app.cell(hide_code=True)
def hints_params():
    L.hints(
        "gpt-params",
        [
            "`embeddings` and `one_block`: steps 1 and 2 of the worked example, in the names of this function, and for one block.",
            "`end` is not that of GPT-2: your GPT has an output head of its own, with a weight for every pair of one of the d numbers and one of the vocab tokens, and a bias term for every token.",
        ],
        why="A table has a row of d numbers for every token or every place. A linear layer has a weight for every pair of an input and an output, and, with bias terms, one per output; only query, key and value have none. A layer normalization has a γ and a β for every number.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_flops3():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Training on one token

        Complete `train_flops(n_params, n_layers, n, d)`: it gives the FLOPs of training a GPT of `n_params` parameters, with `n_layers` blocks of `d` numbers, on one token of a text of `n` tokens, steps 4 and 5 of the recipe. The check calls `train_flops(1_000_000, 4, 128, 128)`, which should give `6393216`, and tries other GPTs. In Python, `1_000_000` is a million: the `_` only makes it easier to read.
        """),
        L.task("Make `train_flops` count the FLOPs of training on one token: parameters and attention, forward and backward."),
    ])
    return


@app.cell(hide_code=True)
def exercise_flops3():
    box_flops = L.code_box("train-flops", lines=6)
    box_flops
    return (box_flops,)


@app.cell(hide_code=True)
def check_flops3(box_flops):
    result_flops = L.check("train-flops", box_flops.value)
    result_flops
    return (result_flops,)


@app.cell(hide_code=True)
def hints_flops3():
    L.hints(
        "train-flops",
        [
            "`forward`: step 4 of the completed worked example, in the names of this function.",
            "Training is the forward pass, and a backward pass of about twice its FLOPs. How many forward passes is that together?",
        ],
        why="In the backward pass, every multiplication of the forward pass gives two: one for the gradient of each of the two numbers it multiplied, such as a weight and what came in. So the backward pass costs about twice the forward pass, and training about three times.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your functions can do

    The cell below counts three GPTs with your `gpt_params`: your GPT of the last lesson, `GPT(65, 64, 4, 4, 64)`; the one you will train on a book, `GPT(65, 384, 6, 6, 256)`; and one with the sizes of GPT-2. PyTorch counts the first two, made by your `GPT` of [Assemble your GPT](/course/open/gpt-assemble?lang=en); the third would take some 650 MB of memory, so it is left out. Then your `train_flops` counts what training on one token costs the GPT for the book, and one with GPT-2's sizes for longer and longer texts, with a table of places as long as the text. It works once exercises 2 and 3 are right.
    """)
    return


@app.cell(hide_code=True)
def demo_count(box_params, box_flops, result_params, result_flops):
    _ok = result_params.state == "correct" and result_flops.state == "correct"
    if not _ok:
        _out = mo.md("*Finish exercises 2 and 3 first: this cell uses your `gpt_params` and your `train_flops`.*")
    else:
        _result = C.demo(box_params.value, box_flops.value)
        if isinstance(_result, str):
            _out = mo.md({
                "slow": "*Your GPT takes too long to make here: more than a minute.*",
                "gpt": "*Your `GPT` in `work/projects/llm/gpt.py` stops with an error. Did you change its file? Delete it, and check exercise 2 of [Assemble your GPT](/course/open/gpt-assemble?lang=en) again.*",
            }[_result])
        else:
            _rows = []
            for _name, _r in zip(["last lesson", "for the book", "GPT-2's sizes"], _result["rows"]):
                _pt = "left out" if _r["pytorch"] is None else C.grouped(_r["pytorch"], L.lang)
                _rows.append(f"| {_name} | {C.grouped(_r['yours'], L.lang)} | {_pt} |")
            _long = [f"| {C.grouped(_g['n'], L.lang)} | {C.grouped(_g['flops'], L.lang)} | {C.number(100 * _g['attention'], 0, L.lang)}% |" for _g in _result["gpt2"]]
            _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + "."
            _book = _result["book"]
            _out = mo.md("| GPT | your count | PyTorch |\n|---|---|---|\n" + "\n".join(_rows) + f"""

Training your GPT for the book on one token of a window of 256 costs **{C.grouped(_book['flops'], L.lang)} FLOPs**, {C.number(100 * _book['attention'], 0, L.lang)}% of them for attention. With GPT-2's other sizes and windows as long as the text, attention takes over:

| text of | FLOPs to train on one token | attention |
|---|---|---|
""" + "\n".join(_long) + f"""

For a short text, about 6 FLOPs per parameter is close; for a long one, attention can cost more than all the parameters together. That is why long texts are expensive.{_who}""")
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The FLOPs of [Too many weights to nudge one by one](/course/open/nn-why-backprop?lang=en), now for every token a GPT reads or learns from, and the parts of your GPT of [Assemble your GPT](/course/open/gpt-assemble?lang=en), counted one by one. Next: *Project: train your GPT on a book*, where your GPT of 10.8 million parameters learns from Tiny Shakespeare, at about 68 million FLOPs for every token it trains on.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_flops, result_params, result_flops):
    L.wrap_up(choice_flops, result_params, result_flops)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
