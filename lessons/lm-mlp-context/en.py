# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson lm-mlp-context. Route: course/route/lessen/taal.yaml. Checks: checks.py.
# termen:allow initialization

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("lm-mlp-context", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Embedding layers](/course/open/nn-embedding-layer?lang=en) and [Project: the same bigram model, learned with gradient descent](/course/open/lm-bigram-neural?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="`nn.Embedding(65, 10)` gets a tensor of ids of shape (32, 3). What shape comes out?",
        options={
            "flat": "(32, 30): the 3 embeddings of a row side by side",
            "right": "(32, 3, 10): one row of 10 numbers for every id",
            "same": "(32, 3): one number for every id",
        },
        correct="right",
        why={
            "flat": "Side by side is a step of its own, which you take today; the layer itself puts a new axis of 10 at the end. See [Embedding layers](/course/open/nn-embedding-layer?lang=en).",
            "same": "Every id gets a whole row of the embedding matrix, 10 numbers. See [Embedding layers](/course/open/nn-embedding-layer?lang=en).",
            "right": "The shape of the ids stays, and an axis of 10 comes at the end. Today, the 3 embeddings of each row are glued side by side, into 30 numbers.",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def quick_check_bigram():
    quick_bigram = L.choice(
        "quick-bigram",
        question="How many characters before it does the bigram model of the bigram project use to predict the next character?",
        options={
            "two": "Two: the character before it and the one before that",
            "all": "All of them, from the start of the text",
            "right": "One: only the last character",
        },
        correct="right",
        why={
            "two": "A bigram is a pair: the last character and the next one. See [Project: the same bigram model, learned with gradient descent](/course/open/lm-bigram-neural?lang=en).",
            "all": "Its only input is the last character, which picks one row of the table. See [Project: the same bigram model, learned with gradient descent](/course/open/lm-bigram-neural?lang=en).",
            "right": "After \"th\", \"ch\" and \"sh\" it only sees \"h\", so it gives all three the same probabilities for the next character. Today, the model looks further back.",
        },
    )
    quick_bigram
    return (quick_bigram,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The bigram model sees only the last character. After "th" it sees "h", and an "h" after "c" or "s" is followed by other characters than one after "t". A model that sees the last few tokens can do much better, but counting does not scale: with 3 characters of context, a table would need a row for each of $65^3 = 274{,}625$ combinations, of which only 11,556 ever occur in Tiny Shakespeare. In 2003, Yoshua Bengio and his colleagues gave every token an embedding, put the embeddings of the last few tokens side by side, and let an MLP give the probabilities of the next token: a **neural language model**. Tokens that behave alike get embeddings that are close, so what the model learns about one combination carries over to similar ones.
    """)
    return


@app.cell(hide_code=True)
def predict_window():
    guess_window = L.guess(
        "three-or-eight",
        question="The model of this lesson trains for 1,000 steps of 128 examples on Tiny Shakespeare: once looking back 3 characters, once 8. Which one ends with the lower loss on text it did not train on?",
        options={
            "eight": "8, clearly: it sees more of the text",
            "right": "About the same: 3 even a little lower",
            "three": "3, clearly: 8 characters are too many to learn",
        },
        reveal="**About the same, with 3 a little lower.** Over 4 seeds, 3 characters end at 2.20 to 2.22, and 8 at 2.22 to 2.24. In the MLP, every place of the window has its own weights, so it has to learn what an \"e\" two places back means apart from what it means five places back: more context is more to learn. With 8 times as much training, 8,000 steps, 8 gets a little ahead: 1.94 to 1.95 against 1.96 to 1.99.",
    )
    guess_window
    return (guess_window,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        The model looks back a fixed number of tokens, $k$: its **context window**. For the next token at place $t$, it takes the ids of the $k$ tokens before it, looks up their embeddings, $d$ numbers each, and glues them side by side into one vector of $k \cdot d$ numbers. A hidden layer with ReLU follows, and then the **output head**: a linear layer from the hidden layer to one logit for every token of the vocabulary, 65 for Tiny Shakespeare. As in [Project: the same bigram model, learned with gradient descent](/course/open/lm-bigram-neural?lang=en), the softmax of the logits gives the probabilities, and cross-entropy with the token that really comes next gives the loss. Bengio and his colleagues used tanh, the S-curve of [Initialization](/course/open/nn-initialization?lang=en), where this lesson uses ReLU; with tanh the story of the picture below is the same, with a loss a little higher.
        """),
        L.formula(r"[[hidden:\mathbf{x}]] = \big[\, [[hidden:\mathbf{e}_{t-k}]] ; \dots ; [[hidden:\mathbf{e}_{t-1}]] \,\big]"),
        L.formula(r"[[pred:\mathbf{z}]] = [[param:\mathbf{W}_2]] \, \mathrm{ReLU}([[param:\mathbf{W}_1]] [[hidden:\mathbf{x}]] + [[param:\mathbf{b}_1]]) + [[param:\mathbf{b}_2]]"),
        mo.md(r"""
        In PyTorch, `nn.Embedding` turns ids of shape (B, k) into embeddings of shape (B, k, d), and `e.reshape(len(x), -1)` glues the $k$ embeddings of each row into one row of $k \cdot d$ numbers. The window is fixed: the model cannot see anything before it, however much text there is. And $\mathbf{W}_1$ has $k \cdot d$ columns, a block of $d$ for every place in the window, so a longer window means more weights, and each place is learned on its own.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **context window** | | the fixed number of tokens a model looks back | `k` |
        | **neural language model** | | a neural network that gives the probabilities of the next token | `MLPLanguageModel(65, 3, 10, 128)` |
        | **output head** | | the last layer: from the network's last vector to one logit for every token of the vocabulary | `nn.Linear(hidden, vocab)` |
        | $\mathbf{x}$ | x | the $k$ embeddings of the window side by side, $k \cdot d$ numbers | `e.reshape(len(x), -1)` |
        | $\mathbf{z}$ | z | the logits, one for every token of the vocabulary | `model(x)` |
        """),
        L.watch_out("Glue the embeddings side by side; do not add them up. Added up, \"ab\" and \"ba\" give the same vector, and the model no longer knows which token came where. And the first $k$ tokens of a text have fewer than $k$ tokens before them: this lesson simply makes no examples for them."),
        mo.md(r"""
        The picture: the loss on text the model did not train on, the last tenth of Tiny Shakespeare, after the same training with the Trainer, for 1, 3 and 8 characters of context, in one run, with seed 0. From 1 to 3 the loss drops a lot; from 3 to 8 it does not drop any more: with this much training, the model cannot make use of the extra characters yet.
        """),
        mo.Html(C.context_picture({"title": "Loss on unseen text", "x": "characters of context, k", "y": "validation loss", "number": L.number,
                                   "alt": "The validation loss after 1,000 steps of training, against the number of characters of context: 2.51 with 1, 2.20 with 3, and 2.22 with 8."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    A neural language model with a context window of $k$ tokens:

    1. Make the examples: for every place $t$ from $k$ on, the $k$ ids before it, and the id at $t$ as the label.
    2. Look up the embeddings: ids of shape (B, k) give shape (B, k, d).
    3. Glue the $k$ embeddings of each example side by side: shape (B, k · d).
    4. A hidden layer with ReLU, then the output head: one logit for every token of the vocabulary.
    5. Train with cross-entropy against the label: your Trainer.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Steps 2 and 3 for two windows of 3 characters, "the" and "he ", with an embedding layer of 10 numbers per character. `C.encode` gives the ids of the characters of a text. The comments with `#` say which step of the recipe each part is. Step 3 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_glue():
    import torch
    from torch import nn

    torch.manual_seed(0)
    _emb = nn.Embedding(65, 10)
    _x = torch.tensor([C.encode("the"), C.encode("he ")])
    print(_x)
    # 2. look up: a row of 10 numbers for every id
    _e = _emb(_x)
    print(_e.shape)
    # 3. glue the 3 embeddings of each window side by side
    _glued = _e  # complete: _e.reshape(len(_x), -1)
    print(_glued.shape)
    # numbers 10 to 19 of the first row: the embedding of its second character, "h"
    print(_glued[0, 10:20].tolist() == _e[0, 1].tolist())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    "the" has the ids 58, 46 and 43, and "he " 46, 43 and 1: the space has id 1. The lookup gives shape (2, 3, 10). As it is, `_glued` is still that shape, and `_glued[0, 10:20]` asks for places 10 to 19 of an axis of 3: nothing, so the comparison prints `False`. Completed, `_glued` has shape (2, 30), and numbers 10 to 19 of its first row are the embedding of "h": `True`. The embedding of "h" stands in the middle of the first row and at the start of the second: the place in the window is the place in the row.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_window():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. What the window sees
    """)
    return


@app.cell(hide_code=True)
def choice_window_cell():
    choice_window = L.choice(
        "window",
        question="A model with a context window of 3 characters predicts the last \"e\" of \"to be or not to be\". Which characters can it use?",
        options={
            "all": "All of \"to be or not to b\": the window only says which ones count most",
            "right": "Only \"o\", the space and \"b\": the 3 just before it",
            "first": "The first 3, \"to \": the window starts at the beginning",
        },
        correct="right",
        why={
            "all": "The model gets only the k ids of the window as its input; everything before it is simply not there.",
            "first": "The window moves along: for every place, it holds the k tokens just before that place.",
            "right": "The window holds the 3 tokens just before the place. That the same \"to b\" came before, earlier in the text, the model cannot see: the window is fixed.",
        },
    )
    choice_window
    return (choice_window,)


@app.cell(hide_code=True)
def text_exercise_contexts():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The examples of a window

        Complete `contexts(ids, k)`: it gives the training examples of a list of ids for a context window of `k`, as two lists: `X`, the windows of `k` ids, and `Y`, the id that follows each window. Replace the three `...`, by step 1 of the recipe. The check calls `contexts([5, 1, 4, 2, 7], 2)`, which should give `([[5, 1], [1, 4], [4, 2]], [4, 2, 7])`, and tries other windows and lists. When it is right, the check saves it in your project, as `work/projects/llm/contexts.py`.
        """),
        L.task("Make `contexts` give the windows of `k` ids and the id that follows each."),
    ])
    return


@app.cell(hide_code=True)
def exercise_contexts():
    box_contexts = L.code_box("contexts", lines=8)
    box_contexts
    return (box_contexts,)


@app.cell(hide_code=True)
def check_contexts(box_contexts):
    result_contexts = L.check("contexts", box_contexts.value)
    result_contexts
    return (result_contexts,)


@app.cell(hide_code=True)
def hints_contexts():
    L.hints(
        "contexts",
        [
            "The first place with `k` ids before it is `k`, and the last place is `len(ids) - 1`, which is a label too.",
            "The window before place `t` is a slice of `ids` that stops just before `t` and is `k` long.",
        ],
        why="For `contexts([5, 1, 4, 2, 7], 2)`, `t` goes from 2 to 4. At `t` = 2, the window is `ids[0:2]`, `[5, 1]`, and the label `ids[2]`, 4; at `t` = 4, the window is `[4, 2]` and the label 7: a list of 5 ids gives 5 − 2 = 3 examples.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_model():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The model

        Complete the class `MLPLanguageModel`: `MLPLanguageModel(vocab, k, dim, hidden)` is the model of the recipe, for a vocabulary of `vocab` tokens and a context window of `k`. Replace the two `...`:

        - In `__init__`, three layers, made in this order so that they get the same random starting values as in the course's solution: `self.emb`, an `nn.Embedding` with `dim` numbers per token; `self.hidden`, an `nn.Linear` from the `k` embeddings side by side to `hidden` neurons; and `self.out`, the output head.
        - In `forward(x)`, with `x` of shape (B, k): steps 2 to 4 of the recipe.

        The check makes small models after `torch.manual_seed(0)` and compares their logits, and whether the order of the tokens in a window matters. When it is right, the check saves it as `work/projects/llm/mlp_lm.py`.
        """),
        L.task("Make `MLPLanguageModel` give one logit for every token, from the last `k` tokens side by side."),
    ])
    return


@app.cell(hide_code=True)
def exercise_model():
    box_model = L.code_box("mlp-lm", lines=16)
    box_model
    return (box_model,)


@app.cell(hide_code=True)
def check_model(box_model):
    result_model = L.check("mlp-lm", box_model.value)
    result_model
    return (result_model,)


@app.cell(hide_code=True)
def hints_model():
    L.hints(
        "mlp-lm",
        [
            "The input of the hidden layer is the glued row of the worked example: `k * dim` numbers. The output head goes from `hidden` to `vocab`.",
            "In `forward`: `self.emb(x)`, then the reshape of the completed worked example with `x` instead of `_x`, then `torch.relu` of the hidden layer, then `self.out`.",
        ],
        why="`self.emb(x)` gives (B, k, dim), the reshape (B, k * dim), the hidden layer (B, hidden), and the output head (B, vocab): one logit for every token of the vocabulary, for every example. The ReLU in between keeps the two linear layers from collapsing into one.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The bigram model of [Project: the same bigram model, learned with gradient descent](/course/open/lm-bigram-neural?lang=en) looked up one row for the last character; this model looks up the embeddings of [Embedding layers](/course/open/nn-embedding-layer?lang=en) for the last $k$, and an MLP makes the logits. It trains with your Trainer from [A reusable training loop](/course/open/nn-trainer?lang=en), and your llm project now has `contexts.py` and `mlp_lm.py`. Its window is fixed, and it learns every place of the window on its own: recurrent networks and then attention, later in the course, use the same weights at every place, and look much further back. In the next lesson, *Perplexity*, you turn the loss into a number you can picture: how many tokens the model is choosing between.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_window, result_contexts, result_model):
    L.wrap_up(choice_window, result_contexts, result_model)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
