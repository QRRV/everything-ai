# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson gpt-assemble. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("gpt-assemble", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [The transformer block](/course/open/attn-transformer-block?lang=en) and [Project: make a deep network train](/course/open/nn-deep-net-project?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_stack():
    quick_stack = L.choice(
        "quick-stack",
        question="A stack of 12 transformer blocks gets a mini-batch of shape (B, n, d). What shape comes out of the last block?",
        options={
            "wide": "(B, n, 12d): every block adds d numbers",
            "right": "(B, n, d): the shape that went in",
            "one": "(B, d): one vector per text",
        },
        correct="right",
        why={
            "wide": "A block adds its results to the residual stream, number by number; the stream keeps its d numbers. See [The transformer block](/course/open/attn-transformer-block?lang=en).",
            "one": "Every token keeps its own vector, through every block. See [The transformer block](/course/open/attn-transformer-block?lang=en).",
            "right": "Every block gives back the shape it gets, so they can be stacked. Today you stack them into a GPT.",
        },
    )
    quick_stack
    return (quick_stack,)


@app.cell(hide_code=True)
def quick_check_modulelist():
    quick_modulelist = L.choice(
        "quick-modulelist",
        question="Why does a network keep a list of layers in an `nn.ModuleList`, and not in a plain Python list?",
        options={
            "right": "So that PyTorch sees their parameters, and an optimizer trains them",
            "faster": "Because a plain list is too slow to hold the layers of a network",
            "order": "So that the layers run in the right order",
        },
        correct="right",
        why={
            "faster": "Speed has nothing to do with it: a plain list hides the layers from `model.parameters()`. See [Project: make a deep network train](/course/open/nn-deep-net-project?lang=en).",
            "order": "A plain list keeps its order too. The difference is that PyTorch does not look inside it. See [Project: make a deep network train](/course/open/nn-deep-net-project?lang=en).",
            "right": "Today an `nn.ModuleList` holds the transformer blocks of your GPT.",
        },
    )
    quick_modulelist
    return (quick_modulelist,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        All the parts are there. Your tokenizer turns a text into token ids, your embeddings of [Telling attention where things are](/course/open/attn-positions?lang=en) give every token and every place a vector, and your blocks of [The transformer block](/course/open/attn-transformer-block?lang=en) let the tokens look at the ones before them. A GPT is those parts in a row, with an output head at the end that gives, at every place, a score for every possible next token. The smallest GPT-2 is built that way: 12 blocks with $d = 768$, and a vocabulary of 50,257 tokens. In this project step you assemble your own GPT, and prepare its training examples.
        """),
        draw.project_steps(["Count letters and pairs", "A bigram model by counting", "The same model, learned by a neural network",
                            "Your own tokenizer", "Your own GPT", "Pretrain a small LLM", "Teach it to follow instructions and preferences"],
                           4, here_label="← today", title="Where your project stands: your own LLM"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_loss():
    guess_loss = L.guess(
        "untrained-loss",
        question="Your GPT for Tiny Shakespeare, with its 65 characters, gets windows of text before any training, with random weights. Which loss do you expect on the next characters?",
        options={
            "zero": "About 0: it has all the weights it needs",
            "right": "About 4.2: as good as guessing evenly among 65",
            "big": "About 65: one for every character it gets wrong",
        },
        reveal="**About 4.2.** Guessing evenly among 65 characters gives each a probability of 1/65, and a cross-entropy of $\\ln 65 = 4.17$. Random weights give logits that are close to equal, so the loss is close to that; the demo below finds 4.40. Training has to bring it down from there.",
    )
    guess_loss
    return (guess_loss,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **GPT** is a **decoder-only transformer**: only causal attention, so every token looks back and never ahead. The transformer of 2017 also had an encoder, which read a whole input sentence at once for a translation; a GPT has no such part. From token ids to logits, it goes:
        """),
        L.formula(r"\begin{aligned} [[hidden:\mathbf{X}]] &= [[param:E]][\text{ids}] + [[param:P]][0, \dots, n-1] \\ [[hidden:\mathbf{X}]] &\leftarrow \mathrm{Block}([[hidden:\mathbf{X}]]) \\ [[pred:\text{logits}]] &= \mathrm{LN}([[hidden:\mathbf{X}]])\,[[param:\mathbf{W}]] + [[param:\mathbf{b}]] \end{aligned}"),
        mo.md(r"""
        $E$ is the table of tokens and $P$ the table of places, `block` rows long: a GPT reads at most `block` tokens at once. The second line runs once for every block, `n_layers` times, and every block has weights of its own: `nn.ModuleList([Block(d, h) for _ in range(n_layers)])` makes a new `Block` for every place in the list. Inside the blocks, only copies of the stream were normalized, so a last layer normalization comes before the output head, a linear layer from $d$ to the size of the vocabulary. The logits have shape $(B, n, \text{vocab})$: a score for every possible next token, at every place of every text.

        That is also how a GPT learns: a window of `block` token ids goes in, and the target at every place is the token after it, so the targets are the same window, one id further. One window gives `block` examples at once, and the loss is the cross-entropy over all places.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **decoder-only transformer** | a transformer of causal blocks only, without an encoder | `GPT(vocab, d, h, n_layers, block)` |
        | **GPT** | a decoder-only transformer that predicts the next token; generative pretrained transformer | `model(ids)` |
        | `block` | the most tokens a GPT reads at once: the rows of its table of places | `GPT(..., block)` |
        """),
        L.watch_out("The places to look up are those of this text, `torch.arange(n)` with `n = ids.shape[1]`, not all `block` of them: a text can be shorter than `block`, and then the shapes would not fit. And a window and its targets have the same length: the target of the last place is the id right after the window."),
        mo.md(r"""
        The picture: your GPT from top to bottom, with the shape of what comes out of every part. The ids are data, the residual stream runs from the embeddings through all blocks, and the output head gives the logits.
        """),
        mo.Html(C.gpt_picture({"title": "Your GPT", "ids": "token ids", "embed": "tokens + places", "block1": "block 1", "blockL": "last block",
                               "norm": "layer normalization", "head": "output head: the logits",
                               "alt": "From top to bottom: token ids of shape (B, n); tokens plus places, (B, n, d); block 1 to the last block, each (B, n, d); layer normalization, (B, n, d); and the output head with the logits, (B, n, vocab)."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    A GPT with a vocabulary of `vocab` tokens, vectors of $d$ numbers, $h$ heads, `n_layers` blocks, and texts of at most `block` tokens:

    1. The ids of B texts of n tokens, a tensor of shape (B, n), with n at most `block`.
    2. The embeddings of the tokens plus those of their places: the residual stream, (B, n, d).
    3. Every block in turn, then layer normalization and the output head: the logits, (B, n, vocab).
    4. Training examples: windows of `block` ids side by side, as long as one more id follows, each with the same window one further as targets.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Steps 1 to 3 for 2 texts of 3 tokens, with a vocabulary of 10 and $d = 8$. The blocks keep the shape, so they are left out here. The first text has token 1 at place 0 and at place 2; the cell prints how much their logits differ. Step 2 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_gpt():
    import torch
    from torch import nn

    torch.manual_seed(0)
    # 1. 2 texts of 3 ids each
    _ids = torch.tensor([[1, 2, 1], [4, 5, 6]])
    _n = _ids.shape[1]
    _tok, _pos = nn.Embedding(10, 8), nn.Embedding(6, 8)
    _norm, _head = nn.LayerNorm(8), nn.Linear(8, 10)
    with torch.no_grad():
        # 2. the tokens plus their places: the residual stream
        _x = _tok(_ids)  # complete: + _pos(torch.arange(_n))
        print(_x.shape)
        # 3. (the blocks keep the shape) layer normalization and the output head: the logits
        _logits = _head(_norm(_x))
        print(_logits.shape)
        print((_logits[0, 0] - _logits[0, 2]).abs().max())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The stream has shape (2, 3, 8), and the logits (2, 3, 10): 10 scores at each of 3 places of 2 texts. As it is, the logits of token 1 at place 0 and at place 2 differ by `tensor(0.)`: without its place, token 1 has the same vector at both places, and so the same logits. Completed, they differ by up to 1.0345.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The shape of the logits
    """)
    return


@app.cell(hide_code=True)
def choice_shape_cell():
    choice_shape = L.choice(
        "logit-shape",
        question="Your GPT with a vocabulary of 65 characters gets 32 windows of 64 ids. What shape do its logits have?",
        options={
            "window": "(32, 65): one guess per window, for the id after it",
            "place": "(32, 64): one number per place",
            "right": "(32, 64, 65): a guess for the next id at every place",
        },
        correct="right",
        why={
            "window": "That was your model of [A neural language model that looks further back](/course/open/lm-mlp-context?lang=en). A GPT guesses at every place at once.",
            "place": "One number cannot say which of 65 characters comes next: every place needs a score for each.",
            "right": "32 × 64 = 2,048 guesses from one forward pass, each with its own target: one pass trains on all of them at once.",
        },
    )
    choice_shape
    return (choice_shape,)


@app.cell(hide_code=True)
def text_exercise_gpt():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Your GPT

        Complete `GPT(vocab, d, h, n_layers, block)`: in `__init__` its five parts, in the order the data passes them; in `forward`, steps 2 and 3 of the recipe, for ids of shape (B, n). Its first lines import your `Block` from your project. The check makes it after `torch.manual_seed(0)` and gives it 2 texts of 3 tokens; then it tries other sizes, and counts the tensors of parameters PyTorch sees. When it is right, the check saves it in your project, as `work/projects/llm/gpt.py`.
        """),
        L.task("Make `GPT` turn token ids into logits: embeddings, every block, layer normalization, output head."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gpt():
    box_gpt = L.code_box("gpt", lines=18)
    box_gpt
    return (box_gpt,)


@app.cell(hide_code=True)
def check_gpt(box_gpt):
    result_gpt = L.check("gpt", box_gpt.value)
    result_gpt
    return (result_gpt,)


@app.cell(hide_code=True)
def hints_gpt():
    L.hints(
        "gpt",
        [
            "`__init__`: the five parts of the idea, in the order of the data: a table with a row for every token and one with a row for every place, each row `d` numbers; `n_layers` new blocks, in a list PyTorch can see into; a layer normalization of `d` numbers; and the output head, from `d` numbers to a score for every token.",
            "`forward`: step 2 of the completed worked example, with `self.tok`, `self.pos` and `n`; then every block in turn on the stream, `x = ...`; then step 3.",
        ],
        why="The blocks run one after the other, each on what the one before gave, along the residual stream. Only after the last block does the stream get its last normalization and become logits.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_examples():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Training examples

        Complete `gpt_examples(ids, block)`: it gives the training examples of a list of token ids, step 4 of the recipe, as two lists: `X`, the windows of `block` ids, and `Y`, the same windows one id further. It is the `contexts` of [A neural language model that looks further back](/course/open/lm-mlp-context?lang=en), with a whole window as target. The check calls `gpt_examples([5, 6, 7, 8, 9, 10, 11], 3)`, which should give `([[5, 6, 7], [8, 9, 10]], [[6, 7, 8], [9, 10, 11]])`, and tries other lists. When it is right, the check saves it as `work/projects/llm/gpt_data.py`.
        """),
        L.task("Make `gpt_examples` cut a list of ids into windows side by side, each with its targets one id further."),
    ])
    return


@app.cell(hide_code=True)
def exercise_examples():
    box_examples = L.code_box("gpt-examples", lines=10)
    box_examples
    return (box_examples,)


@app.cell(hide_code=True)
def check_examples(box_examples):
    result_examples = L.check("gpt-examples", box_examples.value)
    result_examples
    return (result_examples,)


@app.cell(hide_code=True)
def hints_examples():
    L.hints(
        "gpt-examples",
        [
            "The windows start at 0, `block`, `2 * block`, and so on: a `range` with a step. The last start must leave room for the window and one id after it.",
            "A window is `ids[i:i + block]`; its targets are the same slice, with both ends one further.",
        ],
        why="`range(0, len(ids) - block, block)` stops before a start where the window would reach the last id, so every window has a full target: with 6 ids and windows of 3, a window starting at place 3 would end on the last id, and its last place would have no target.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    The cell below uses your project. It turns all of Tiny Shakespeare into character ids and makes your training examples with windows of 64. Then it makes your GPT for its 65 characters, with $d = 64$, 4 heads and 4 blocks, after `torch.manual_seed(0)`, and lets it guess the next characters of 8 windows, before any training. It works once exercises 2 and 3 are done.
    """)
    return


@app.cell(hide_code=True)
def demo_gpt(result_gpt, result_examples):
    result_gpt, result_examples  # run again when exercise 2 or 3 saves your module
    _result = C.demo()
    if _result is None:
        _out = mo.md("*Finish exercises 2 and 3 first: then your `GPT` and your `gpt_examples` are in your project, and this cell uses them.*")
    elif isinstance(_result, str):
        _out = mo.md({
            "slow": "*Your GPT takes too long here: more than a minute. Does your `GPT` or your `gpt_examples` do more work than it needs, such as a loop that runs too often?*",
            "gpt": "*`GPT` or `gpt_examples` in `work/projects/llm/` stops with an error. Did you change its file? Delete it, and check exercise 2 or 3 again: its check then saves your code again.*",
        }[_result])
    else:
        _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", as the note under *Where your project stands* says."
        _out = mo.md(f"""Tiny Shakespeare gives **{C.grouped(_result['windows'], L.lang)}** windows of 64 characters. Your GPT has {C.grouped(_result['params'], L.lang)} parameters, and for 8 windows its logits have shape {tuple(_result['shape'])}.

Before any training, its loss on those windows is **{C.number(_result['loss'], 2, L.lang)}**; guessing evenly among {_result['chars']} characters gives {C.number(_result['guess_loss'], 2, L.lang)}. Your GPT knows nothing yet, and it has everything it needs to learn.{_who}""")
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The embeddings of tokens and places of [Telling attention where things are](/course/open/attn-positions?lang=en), your blocks of [The transformer block](/course/open/attn-transformer-block?lang=en) in an `nn.ModuleList` as in [Project: make a deep network train](/course/open/nn-deep-net-project?lang=en), and the examples of [A neural language model that looks further back](/course/open/lm-mlp-context?lang=en), now with a target at every place: your own GPT. Your project `llm` now has `gpt.py` and `gpt_data.py`. Next: *Counting parameters and FLOPs per token*, where the parameters of your GPT sit and what one token costs; then you train it on a book.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_shape, result_gpt, result_examples):
    L.wrap_up(choice_shape, result_gpt, result_examples)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
