# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nn-embedding-layer. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-embedding-layer", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Project: the same bigram model, learned with gradient descent](/course/open/lm-bigram-neural?lang=en) and [Layers as modules: nn.Linear](/course/open/torch-nn-module?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rows():
    quick_rows = L.choice(
        "quick-rows",
        question="`W` is the 65 × 65 table of the bigram project, and `x = torch.tensor([3, 3, 10])`. What is `W[x]`?",
        options={
            "one": "One row: row 3, the first id of the tensor; the others are left out",
            "right": "Three rows: row 3, row 3 again and row 10, shape (3, 65)",
            "sum": "One row: the sum of rows 3, 3 and 10",
        },
        correct="right",
        why={
            "one": "Indexing with a tensor of ids gives one row for every id in it. See [Project: the same bigram model, learned with gradient descent](/course/open/lm-bigram-neural?lang=en).",
            "sum": "Nothing is added up: every id gives its own row. See [Project: the same bigram model, learned with gradient descent](/course/open/lm-bigram-neural?lang=en).",
            "right": "Every id gives its own row, so an id that comes twice gives the same row twice. Today, such a table gets fewer columns and its own layer.",
        },
    )
    quick_rows
    return (quick_rows,)


@app.cell(hide_code=True)
def quick_check_params():
    quick_params = L.choice(
        "quick-params",
        question="How many parameters does `nn.Linear(8, 65)` have?",
        options={
            "weights": "520: one weight for every pair of an input and a neuron",
            "plus": "73: 8 inputs and 65 neurons",
            "right": "585: 8 weights for each of the 65 neurons, plus 65 bias terms",
        },
        correct="right",
        why={
            "weights": "Each of the 65 neurons also has a bias term. See [Layers as modules: nn.Linear](/course/open/torch-nn-module?lang=en).",
            "plus": "Each neuron has a weight for every input, not one in all. See [Layers as modules: nn.Linear](/course/open/torch-nn-module?lang=en).",
            "right": "`weight` has shape (65, 8), 520 numbers, and `bias` 65 more. In exercise 3, such a layer comes after an embedding layer.",
        },
    )
    quick_params
    return (quick_params,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In the bigram project, every character had its own row of 65 numbers, and every row learned on its own, even for `!` and `?`, which are followed by almost the same characters. That does not scale: a model that looks at the 2 characters before needs a row for every pair, 4,225 rows; for 3 characters, 274,625. An **embedding layer** gives every character a short row of numbers instead, and lets the rest of the network compute with it. Characters that behave alike can then get rows that are alike, and share what they learn.
    """)
    return


@app.cell(hide_code=True)
def predict_dims():
    guess_dims = L.guess(
        "how-many-numbers",
        question="The bigram model now gets an embedding layer with a few numbers per character, and then `nn.Linear` from those numbers to 65 logits. Trained on all pairs of Tiny Shakespeare, the table of 65 × 65 of the bigram project reaches a loss of 2.4526. Of these three, which is the fewest numbers per character with which the embedding layer comes within 0.01 of that?",
        options={
            "two": "2: a vowel or not, and a capital or not",
            "right": "16: a quarter of the 65",
            "all": "65: one for every possible next character, as in the table",
        },
        reveal="**16.** With 16 numbers per character, the loss is 2.4543; with 8 it is 2.4839, with 2 still 2.7341. 16 numbers per character make 2,145 parameters with the linear layer, half of the table's 4,225. In such short rows of numbers, characters that behave alike end up close together: in this run, the four nearest to `.` by cosine similarity are `?`, `;`, `:` and `!`. With another seed, the four change, but they are nearly always punctuation.",
    )
    guess_dims
    return (guess_dims,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        An **embedding layer** is a table that the network learns: the **embedding matrix** $E$, one row of $d$ numbers for each of the $\lvert\mathcal{V}\rvert$ ids, such as the 65 characters. For id $x$ it gives row $x$, its embedding. That is what a one-hot vector times the table gives as well, as in the bigram project, but without multiplying by 64 zeros:
        """),
        L.formula(r"[[hidden:\mathbf{e}]] = [[param:E]][x] = \mathrm{onehot}(x) \, [[param:E]]"),
        mo.md(r"""
        On the way back, only the rows that were looked up get a gradient, and a row that was looked up at several places gets all their gradients added up. For row $v$ of $E$, the sum runs over the places $k$ where the id $x_k$ is $v$:
        """),
        L.formula(r"[[grad:\frac{\partial \mathcal{L}}{\partial E_v}]] = \sum_{k \,:\, x_k = v} [[grad:\frac{\partial \mathcal{L}}{\partial \mathbf{e}_k}]]"),
        mo.md(r"""
        With gradient descent, the rows of ids that did not come up stay as they are in that step.

        In PyTorch, the layer is `nn.Embedding(vocab, d)`. Its `weight` is $E$, of shape $(\lvert\mathcal{V}\rvert, d)$, with random numbers at the start. It takes a tensor of ids of any shape, of whole numbers, and gives one row of $d$ numbers per id: ids of shape (3,) give shape (3, $d$). The bigram model with an embedding layer is then `nn.Linear(d, vocab)` after `nn.Embedding(vocab, d)`: from an id to its embedding, and from the embedding to $\lvert\mathcal{V}\rvert$ logits for the next character.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **embedding layer** | | a layer that looks up a row of a learned table for every id | `nn.Embedding(vocab, d)` |
        | **embedding matrix** | | the table of that layer: one row of $d$ numbers per id | `emb.weight` |
        | $\lvert\mathcal{V}\rvert$, $d$ | the vocabulary size, d | the number of ids, and the number of numbers in an embedding | `num_embeddings`, `embedding_dim` |
        """),
        L.watch_out("`nn.Embedding` takes ids, whole numbers from 0 to $\\lvert\\mathcal{V}\\rvert - 1$, not one-hot vectors or floats: with an id of $\\lvert\\mathcal{V}\\rvert$ or more, Python stops with an error. And the embeddings start as random numbers; they get their meaning only from training."),
        mo.md(r"""
        The picture: the bigram model with an embedding layer of 1 to 32 numbers per character, trained on all pairs of characters of Tiny Shakespeare, and its loss after training. The dashed line is the table of 65 × 65 of the bigram project: the best a bigram model can do. From 16 numbers on, the embedding layer is almost as good as the table.
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_dims():
    mo.Html(C.dims_picture({"title": "Loss by size of the embedding", "x": "numbers per character", "y": "loss after training", "table": "table of 65 × 65",
                            "alt": "The loss after training against the number of numbers per character, 1 to 32, on a log scale: 2.92 with 1, 2.73 with 2, 2.59 with 4, 2.48 with 8, and 2.45 with 16 and 32, on the dashed line of the table of 65 by 65."}))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    An embedding layer:

    1. Number the things: every character, word or token gets an id from 0 to $\lvert\mathcal{V}\rvert - 1$.
    2. Make the table: $\lvert\mathcal{V}\rvert$ rows of $d$ numbers, random at first; in PyTorch `nn.Embedding(vocab, d)`.
    3. Look up: for ids `x`, the rows `E[x]`, one per id.
    4. Train: on the way back, each row gets the gradients of all places where its id came up, added up; then the optimizer takes its step.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A table of 4 ids with 2 numbers each, in NumPy, and the ids 2, 0 and 2. The comments with `#` say which step of the recipe each part is. Step 4 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_lookup():
    import numpy as np

    # 2. the table: 4 ids, 2 numbers each
    _E = np.array([[0.1, 0.2], [0.5, -0.3], [-0.4, 0.8], [0.0, 1.0]])
    _ids = np.array([2, 0, 2])
    # 3. look up: one row per id, the same as one-hot rows times the table
    print(_E[_ids])
    print(np.eye(4)[_ids] @ _E)
    # 4. the gradient of the table, from the gradients of the three looked-up rows
    _grad_out = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    _grad_E = np.zeros(_E.shape)
    for _k in range(len(_ids)):
        # complete: add to the row, instead of overwriting it
        _grad_E[_ids[_k]] = _grad_out[_k]
    print(_grad_E)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `_E[_ids]` gives rows 2, 0 and 2: $(-0.4, 0.8)$, $(0.1, 0.2)$ and $(-0.4, 0.8)$ again, and the one-hot rows times the table give the same. Rows 1 and 3 were not looked up, so their gradient is 0. Id 2 comes up twice, with the gradients $(1, 0)$ and $(2, 2)$. As it is, the loop overwrites row 2, and only the last one stays: $(2, 2)$. Completed, with `_grad_E[_ids[_k]] = _grad_E[_ids[_k]] + _grad_out[_k]`, row 2 gets both: $(3, 2)$.
    """)
    return


@app.cell(hide_code=True)
def text_torch_step():
    mo.md(r"""
    ### One rung up: the same in PyTorch

    `nn.Embedding` does all of this itself, the adding up included. Here with its own random table: `.backward()` of the sum of each looked-up row times its gradient gives every row of the table its gradients, added up, as in the loop.
    """)
    return


@app.cell
def example_torch():
    import torch
    from torch import nn

    torch.manual_seed(0)
    _emb = nn.Embedding(4, 2)
    _rows = _emb(torch.tensor([2, 0, 2]))
    print(_emb.weight.shape, _rows.shape)
    (_rows * torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])).sum().backward()
    print(_emb.weight.grad)
    return


@app.cell(hide_code=True)
def text_after_torch():
    mo.md(r"""
    The table has shape (4, 2), and the three looked-up rows shape (3, 2). The gradient of the table is the one of the completed NumPy example: $(0, 1)$ for row 0, $(3, 2)$ for row 2, and 0 for rows 1 and 3.

    | NumPy | PyTorch |
    |---|---|
    | `E = np.array(...)`, shape (4, 2) | `emb = nn.Embedding(4, 2)`, `emb.weight` |
    | `E[ids]` | `emb(ids)` |
    | the loop that adds `grad_out[k]` to row `ids[k]` | `.backward()`, into `emb.weight.grad` |
    """)
    return


@app.cell(hide_code=True)
def text_exercise_lookup():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Why look up?
    """)
    return


@app.cell(hide_code=True)
def choice_lookup_cell():
    choice_lookup = L.choice(
        "why-lookup",
        question="A one-hot vector times the embedding matrix gives the same row as looking it up. Why does `nn.Embedding` look rows up?",
        options={
            "right": "It gives the same row without the work: no multiplying by all the zeros",
            "better": "Looking up gives a better row: the multiplication adds rounding errors",
            "grad": "One-hot vectors have no gradient, so the table could not learn",
        },
        correct="right",
        why={
            "better": "Multiplying by 0s and one 1 gives exactly the row, without rounding errors: both ways give the same numbers.",
            "grad": "The table gets the same gradient either way; the one-hot vectors themselves do not need one.",
            "right": "With 50,000 different tokens, a one-hot vector holds 49,999 zeros, and multiplying by them is work for nothing. Looking up is the same row, at a fraction of the cost.",
        },
    )
    choice_lookup
    return (choice_lookup,)


@app.cell(hide_code=True)
def text_exercise_step():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. One step on the table

        Write `sgd_step(E, ids, grad_out, lr)`: one step of gradient descent on the embedding table `E`, shape $(\lvert\mathcal{V}\rvert, d)$, after the rows of `ids` were looked up and got the gradients `grad_out`, one row per id. First make the gradient of the table, by step 4 of the recipe; then give back `E - lr * grad_E`. The check calls `sgd_step(np.array([[0, 0], [1, 1], [2, 2]]), np.array([0, 0]), np.array([[1, 0], [0, 1]]), 0.5)`, which should give `np.array([[-0.5, -0.5], [1, 1], [2, 2]])`, and tries other tables and ids.
        """),
        L.task("Make `sgd_step` take one step of gradient descent on an embedding table."),
    ])
    return


@app.cell(hide_code=True)
def exercise_step():
    box_step = L.code_box("sgd-step", lines=10)
    box_step
    return (box_step,)


@app.cell(hide_code=True)
def check_step(box_step):
    result_step = L.check("sgd-step", box_step.value)
    result_step
    return (result_step,)


@app.cell(hide_code=True)
def hints_step():
    L.hints(
        "sgd-step",
        [
            "The loop of the worked example, completed: for every `k`, row `ids[k]` of `grad_E` gets `grad_out[k]` added to it.",
            "Then the step itself: the table minus the learning rate times its gradient. Rows that were not looked up have a gradient of 0, so they stay.",
        ],
        why="`for k in range(len(ids)): grad_E[ids[k]] = grad_E[ids[k]] + grad_out[k]`, then `return E - lr * grad_E`. In the example, row 0 is looked up twice and gets $(1, 0) + (0, 1) = (1, 1)$; with a learning rate of 0.5, it goes from $(0, 0)$ to $(-0.5, -0.5)$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_model():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The bigram model with an embedding layer

        Write a class `CharBigram(nn.Module)`: `CharBigram(vocab, dim)` first makes `self.emb`, an `nn.Embedding` with `vocab` rows of `dim` numbers, and then `self.out`, an `nn.Linear` from `dim` to `vocab`, in that order, so that they get the same random starting numbers as the check's model. `forward(x)` looks up the ids in `x` and gives the rows to the linear layer: one row of `vocab` logits per id. The check makes `CharBigram(5, 2)` with a fixed seed and compares its logits for the ids 0, 3 and 3; then it makes `CharBigram(65, 8)` and compares the shapes of its parameters and of its logits for ids in a table of 2 by 2.
        """),
        L.task("Make `CharBigram` give the logits of the next character with an embedding layer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_model():
    box_model = L.code_box("char-bigram", lines=12)
    box_model
    return (box_model,)


@app.cell(hide_code=True)
def check_model(box_model):
    result_model = L.check("char-bigram", box_model.value)
    result_model
    return (result_model,)


@app.cell(hide_code=True)
def hints_model():
    L.hints(
        "char-bigram",
        [
            "Both layers are attributes made in `__init__`, after `super().__init__()`, as in [Layers as modules: nn.Linear](/course/open/torch-nn-module?lang=en): the embedding layer gets `vocab` and `dim`, the linear layer goes from `dim` to `vocab`.",
            "`forward` is one line, read from the inside out: first the embedding layer on `x`, then the linear layer on what comes out.",
        ],
        why="`__init__`: `self.emb = nn.Embedding(vocab, dim)` and then `self.out = nn.Linear(dim, vocab)`. `forward`: `return self.out(self.emb(x))`. `CharBigram(65, 8)` has $65 \\cdot 8 = 520$ numbers in its table and 585 in its linear layer: 1,105 parameters.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The rows `W[x]` of [Project: the same bigram model, learned with gradient descent](/course/open/lm-bigram-neural?lang=en) were already a lookup in a learned table; an embedding layer makes those rows short and gives them a layer of their own, built like the layers of [Layers as modules: nn.Linear](/course/open/torch-nn-module?lang=en). What they learn is what [Embeddings: meaning as coordinates](/course/open/ml-embeddings-idea?lang=en) showed: things that behave alike get rows that are alike. In the next lesson, *A reusable training loop*, you write the training loop once, so that every model after this one can use it. Later, a language model that looks at several characters at once puts their embeddings side by side, and a transformer starts with an embedding layer.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_lookup, result_step, result_model):
    L.wrap_up(choice_lookup, result_step, result_model)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
