# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson lm-weight-tying. Route: course/route/lessen/taal.yaml. Checks: checks.py.
# termen:allow convolution

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("lm-weight-tying", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Assemble your GPT](/course/open/gpt-assemble?lang=en) and [Embedding layers](/course/open/nn-embedding-layer?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_logits():
    quick_logits = L.choice(
        "quick-logits",
        question="Where do the logits of your GPT come from?",
        options={
            "embedding": "From the embedding of the tokens, `self.tok`",
            "block": "From the attention of the last block",
            "right": "From `self.head`, the output head",
        },
        correct="right",
        why={
            "embedding": "`self.tok` is at the start: it turns ids into vectors. See [Assemble your GPT](/course/open/gpt-assemble?lang=en).",
            "block": "A block gives the residual stream, d numbers per place; it still has to become a score for every token. See [Assemble your GPT](/course/open/gpt-assemble?lang=en).",
            "right": "`nn.Linear(d, vocab)` turns the d numbers of every place into a logit for every token. That head is half of this lesson.",
        },
    )
    quick_logits
    return (quick_logits,)


@app.cell(hide_code=True)
def quick_check_lookup():
    quick_lookup = L.choice(
        "quick-lookup",
        question="`tok = nn.Embedding(vocab, d)`. What does `tok(torch.tensor([7]))` give?",
        options={
            "column": "Column 7 of `tok.weight`",
            "right": "Row 7 of `tok.weight`",
            "times": "Every weight of `tok.weight` times 7",
        },
        correct="right",
        why={
            "column": "The table has a row of d numbers for every token, shape (vocab, d). See [Embedding layers](/course/open/nn-embedding-layer?lang=en).",
            "times": "An embedding layer looks a row up; it does not multiply by the id. See [Embedding layers](/course/open/nn-embedding-layer?lang=en).",
            "right": "One row per token, shape (vocab, d). The other half of this lesson.",
        },
    )
    quick_lookup
    return (quick_lookup,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Your GPT has two tables with something for every token of the vocabulary: the embedding at the start, which turns a token into a vector, and the output head at the end, which turns a vector into a score for every token. With a vocabulary of 50,000 tokens, both are large. **Weight tying** uses one table for both ends. GPT-2 does it, and so do many small models.
    """)
    return


@app.cell(hide_code=True)
def predict_gpt2():
    guess_gpt2 = L.guess(
        "untied-gpt2",
        question="GPT-2 small, with $d = 768$ and a vocabulary of 50,257 tokens, has 124 million parameters, with its head tied to its embedding. How many would it have with a head of its own?",
        options={
            "same": "About 125 million",
            "right": "About 163 million",
            "double": "About 250 million",
        },
        reveal=f"**About 163 million:** the head's weight has 50,257 × 768 = {C.grouped(C.VOCAB * 768, 'en')} numbers, almost a quarter of the untied model. The twelve blocks together hold about 85 million.",
    )
    guess_gpt2
    return (guess_gpt2,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Two tables of the same shape.** The embedding `nn.Embedding(vocab, d)` keeps a table $\mathbf{E}$ of shape (vocab, d): row $v$ is the vector of token $v$. The head `nn.Linear(d, vocab)` keeps its weight with a row for every output, as every `nn.Linear` does: shape (vocab, d) as well. Its logit for token $v$ is the dot product of the last vector $\mathbf{h}$ with row $v$, plus a bias term.

        **One table.** Weight tying takes $\mathbf{E}$ for both. Then the logit of token $v$ is the dot product of $\mathbf{h}$ with the token's own embedding $\mathbf{e}_v$:
        """),
        L.formula(r"[[pred:z_v]] = [[hidden:\mathbf{h}]] \cdot [[param:\mathbf{e}_v]] + [[param:b_v]]"),
        mo.md(r"""
        For all tokens at once: $\mathbf{h} \mathbf{E}^\top$. The model gives token $v$ a high logit when its last vector points the same way as the embedding of $v$. In PyTorch, one line in `__init__` does it: `self.head.weight = self.tok.weight`. The head and the embedding now hold the same `nn.Parameter`; `model.parameters()` gives it once, and the optimizer changes it once per step.

        **Two gradients, added up.** The table is used twice, as a convolution uses its filter at every place in [Convolution: one small filter slides over the image](/course/open/cnn-convolution-idea?lang=en): weight sharing. So its gradient is the sum of the gradient from the start and the one from the end.

        **Why it can help.** Without tying, a training step gives a gradient to the embedding rows of the tokens in the batch only. The head gets one for every row: softmax with cross-entropy gives every logit a gradient, as in [The gradient of softmax with cross-entropy](/course/open/ml-softmax-gradient?lang=en). Tied, a rare token's row learns at every step too. Press and Wolf (2017) found that tied language models reached a lower loss on new text, with fewer parameters.

        The picture: the share of the parameters that tying saves, for the four sizes of GPT-2. The larger the model, the more of it sits in the blocks, and the less tying saves.
        """),
        mo.Html(C.saved_picture({"title": "What tying saves in GPT-2", "x": "share of the untied model's parameters (%)",
                                 "row": "{name}, {n} million untied", "lang": L.lang,
                                 "alt": "Four horizontal bars. Tying saves 24% of GPT-2 small, 163 million parameters untied; 13% of GPT-2 medium, 406 million; 8% of GPT-2 large, 838 million; and 5% of GPT-2 XL, 1,638 million."})),
        mo.md(r"""
        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **weight tying** | one table for the embedding and the output head, so that the logit of a token is the dot product with its own embedding | `self.head.weight = self.tok.weight` |
        | $\mathbf{E}$, $\mathbf{e}_v$ | the embedding table, shape (vocab, d), and its row for token $v$ | `self.tok.weight`, `self.tok.weight[v]` |
        | $\mathbf{h} \mathbf{E}^\top$ | the logits of a tied head for every token, without the bias terms | |
        """),
        L.watch_out("Tie the parameter, not a copy. `nn.Parameter(self.tok.weight.clone())` gives the head the same numbers to start with, but as a table of its own: PyTorch counts it, and training moves the two apart. And `a is b` is `True` only when `a` and `b` are one and the same object: that is how you check that a model is tied."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To tie the head of a language model to its embedding:

    1. Make the embedding `nn.Embedding(vocab, d)` and the head `nn.Linear(d, vocab)`, as before: both weights have shape (vocab, d).
    2. Give the head the embedding's parameter: `self.head.weight = self.tok.weight`.
    3. Check it: `model.head.weight is model.tok.weight` gives `True`, and `model.parameters()` holds the table once.
    4. Train as before: the gradient of the table is the sum of the gradients from both ends.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A vocabulary of 4 tokens with $d = 2$, one step: after token 1, token 3 should follow. Untied, with two tables that start with the same numbers; then tied, with one. The comments with `#` say which step of the recipe each part is. The sum is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_grad():
    import torch
    from torch import nn

    _start = [[0.5, -1.0], [1.0, 0.5], [-0.5, 0.5], [0.0, 1.0]]
    _ids, _target = torch.tensor([1]), torch.tensor([3])
    _loss_fn = nn.CrossEntropyLoss()
    # 1. untied: an embedding table and a head table, with the same numbers to start with
    _E_in = torch.tensor(_start, requires_grad=True)
    _E_out = torch.tensor(_start, requires_grad=True)
    _loss_fn(_E_in[_ids] @ _E_out.T, _target).backward()
    print("embedding:", _E_in.grad)
    print("head:", _E_out.grad)
    # 2. tied: one table at both ends
    _E = torch.tensor(_start, requires_grad=True)
    _loss_fn(_E[_ids] @ _E.T, _target).backward()
    print("tied:", _E.grad)
    # 4. the gradient of the tied table, from the two parts
    _sum = _E_in.grad  # complete: _E_in.grad + _E_out.grad
    print("sum:", _sum)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The embedding gets a gradient in row 1 only: token 1 was the only input. The head gets one in every row, because softmax spreads the gradient over every logit. As it is, the sum is only the embedding's part, with zeros in rows 0, 2 and 3. Completed, it is the same as the tied gradient, row for row: in row 1, $0.5205 + 0.5045 = 1.0251$ (with rounding), and the other rows come from the head alone.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_logits():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The logits of a tied head

        Complete `tied_logits(h, E)`: for the vectors in the rows of `h`, shape (n, d), and the embedding table `E`, shape (vocab, d), give the logits of a tied head without bias terms, shape (n, vocab). The check calls `tied_logits([[1.0, 0.0, 2.0]], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.5, -1.0, 2.0]])`, which should give `[[1.0, 0.0, 3.0, 4.5]]`, and tries more vectors and other tables.
        """),
        L.task("Make `tied_logits` give the dot product of every vector of `h` with every row of `E`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_logits():
    box_logits = L.code_box("tied-logits", lines=6)
    box_logits
    return (box_logits,)


@app.cell(hide_code=True)
def check_logits(box_logits):
    result_logits = L.check("tied-logits", box_logits.value)
    result_logits
    return (result_logits,)


@app.cell(hide_code=True)
def hints_logits():
    L.hints(
        "tied-logits",
        [
            "A matrix product of `h` with something gives a row for every vector of `h`. Which shape must that something have, to give a column for every token?",
            "`E` has a row for every token, shape (vocab, d); you need (d, vocab): a column for every token.",
        ],
        why="`return h @ E.T`. For the example, the vector (1, 0, 2) against the four rows gives 1, 0, 1 + 2 = 3 and 0.5 + 4 = 4.5.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_class():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A tied model

        `TiedLM` is the smallest language model with both tables: an embedding and a head, with nothing in between. Replace the `...` in `__init__` with the two layers, as in your GPT, and the line that ties them. The check makes `TiedLM(10, 4)` and asks whether `model.head.weight is model.tok.weight`; then it counts the tensors of parameters, and checks that after a training step the head and the embedding are still equal.
        """),
        L.task("Make `TiedLM` use one table for its embedding and its head."),
    ])
    return


@app.cell(hide_code=True)
def exercise_class():
    box_class = L.code_box("tied-lm", lines=12)
    box_class
    return (box_class,)


@app.cell(hide_code=True)
def check_class(box_class):
    result_class = L.check("tied-lm", box_class.value)
    result_class
    return (result_class,)


@app.cell(hide_code=True)
def hints_class():
    L.hints(
        "tied-lm",
        [
            "Steps 1 and 2 of the recipe. Both weights have shape (vocab, d), so nothing needs transposing.",
            "Assign to `self.head.weight` the parameter of the embedding itself, not a calculation with it.",
        ],
        why="`self.tok = nn.Embedding(vocab, d)`, `self.head = nn.Linear(d, vocab)` and `self.head.weight = self.tok.weight`. The other way round, `self.tok.weight = self.head.weight`, ties them too: then the table starts with the head's numbers.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Your model at GPT-2's size

    The cell below makes your `TiedLM` with GPT-2's vocabulary and $d = 768$, only the shapes, and counts its parameters as an optimizer sees them.
    """)
    return


@app.cell(hide_code=True)
def demo_counts(box_class, result_class):
    _both = C.counts(box_class.value) if result_class.state == "correct" else None
    mo.md(
        f"Your `TiedLM({C.VOCAB}, 768)` has {C.grouped(_both[0], L.lang)} parameters: one table and {C.grouped(C.VOCAB, L.lang)} bias terms. Untied, it would have {C.grouped(_both[1], L.lang)}: almost half of it would be a second table."
        if _both else "When your `TiedLM` is right, its parameters are counted here."
    )
    return


@app.cell(hide_code=True)
def text_exercise_rows():
    mo.md(r"""
    ### Exercise 3. Which rows learn
    """)
    return


@app.cell(hide_code=True)
def choice_rows_cell():
    choice_rows = L.choice(
        "rows-grad",
        question="An untied model takes a training step on a batch in which only the tokens \"the\" and \"cat\" occur as inputs. Which rows get a gradient?",
        options={
            "right": "Embedding: the rows of \"the\" and \"cat\"; head: every row",
            "both-two": "Embedding and head alike: only the two rows of \"the\" and \"cat\"",
            "both-all": "Embedding and head: every row",
        },
        correct="right",
        why={
            "both-two": "The head gives a logit for every token, and softmax with cross-entropy gives every logit a gradient: every row of the head gets one.",
            "both-all": "The embedding only looks up the rows of the tokens in the batch; the other rows are not used in this step, and get no gradient.",
            "right": "As in the worked example. Tied, the table gets both, so the row of a token that is seldom an input still learns from the head at every step.",
        },
    )
    choice_rows
    return (choice_rows,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The embedding of [Embedding layers](/course/open/nn-embedding-layer?lang=en) and the output head of [Assemble your GPT](/course/open/gpt-assemble?lang=en), now one table, with the weight sharing of [Convolution: one small filter slides over the image](/course/open/cnn-convolution-idea?lang=en): one parameter used in two places, and a gradient from each. Your own GPT reads only 65 characters, so tying would save it just 65 × d numbers, and it would change how its head starts: with the numbers of `nn.Embedding`, drawn with a standard deviation of 1, its first loss would be far above 4.17, the loss of guessing evenly. Tying pays off with a vocabulary of tens of thousands of tokens, as in GPT-2, which starts all its weights small.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_rows, result_class, result_logits):
    L.wrap_up(result_logits, result_class, choice_rows)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
