# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson attn-transformer-block. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow transpose

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("attn-transformer-block", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Telling attention where things are](/course/open/attn-positions?lang=en) and [Layer normalization](/course/open/nn-layernorm?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_place():
    quick_place = L.choice(
        "quick-place",
        question="A GPT-2 has learned position embeddings. Where does the place of a token come in?",
        options={
            "right": "Once, before the first layer: added to the embedding of the token",
            "scores": "In every attention head, added to its scores",
            "mask": "Nowhere: the causal mask already gives the order",
        },
        correct="right",
        why={
            "scores": "Some newer models change the scores, such as ALiBi. GPT-2 adds a learned vector for every place to the token embeddings. See [Telling attention where things are](/course/open/attn-positions?lang=en).",
            "mask": "The causal mask says which tokens came before, not in which order. See [Telling attention where things are](/course/open/attn-positions?lang=en).",
            "right": "Today you build the block that gets those vectors, and passes them on to the next block.",
        },
    )
    quick_place
    return (quick_place,)


@app.cell(hide_code=True)
def quick_check_prenorm():
    quick_prenorm = L.choice(
        "quick-prenorm",
        question="In a pre-norm block, `x + f(norm(x))`, what goes along the direct path, from the input straight to the output?",
        options={
            "normalized": "The normalized x",
            "right": "x itself, unchanged",
            "f": "What f makes of the normalized x",
        },
        correct="right",
        why={
            "normalized": "The normalization is only on the way into f. The direct path carries x as it came in. See [Layer normalization](/course/open/nn-layernorm?lang=en).",
            "f": "That is the other path, the one through f, which is added to x at the end. See [Layer normalization](/course/open/nn-layernorm?lang=en).",
            "right": "Today a block has two of these in a row: one with attention as f, one with an MLP.",
        },
    )
    quick_prenorm
    return (quick_prenorm,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        A GPT is one block, again and again: the smallest GPT-2 stacks 12 of them, GPT-3 stacks 96. Each block has attention, as in [Multi-head attention](/course/open/attn-multi-head?lang=en), and an MLP, each with layer normalization and a residual connection. In this project step you build that block into your project, for a mini-batch of texts at once, as a GPT trains. It is the first part of your own GPT; your own tokenizer comes next on the route, and then the whole GPT.
        """),
        draw.project_steps(["Count letters and pairs", "A bigram model by counting", "The same model, learned by a neural network",
                            "Your own tokenizer", "Your own GPT", "Pretrain a small LLM", "Teach it to follow instructions and preferences"],
                           4, here_label="← today: its block", title="Where your project stands: your own LLM"),
    ])
    return


@app.cell(hide_code=True)
def predict_params():
    guess_params = L.guess(
        "params",
        question="One block of GPT-2 has $d = 768$. Its attention has four matrices of 768 by 768; its MLP takes every vector to 3,072 numbers and back to 768, with one matrix each way. Which part has the most parameters?",
        options={
            "attention": "The attention: four matrices against two",
            "right": "The MLP: about twice as many as the attention",
            "equal": "Both about as many",
        },
        reveal="**The MLP: about twice as many as the attention.** The attention has four matrices of 768 by 768, for the queries, keys, values and the output projection: 2,359,296 weights, and 3,072 bias terms. The MLP has two matrices of 768 by 3,072: 4,718,592 weights, and 3,840 bias terms. The two layer normalizations have 3,072 together. One block has 7,087,872 parameters.",
    )
    guess_params
    return (guess_params,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **transformer block** gets the vectors of $n$ tokens, of $d$ numbers each, and gives back $n$ new vectors of the same size. It has two parts, each with a layer normalization before it and a residual connection around it, as your `PreNormBlock` of [Layer normalization](/course/open/nn-layernorm?lang=en): first multi-head causal attention, then an MLP, a hidden layer for the vector of every token:
        """),
        L.formula(r"\begin{aligned} [[hidden:\mathbf{X}]] &\leftarrow [[hidden:\mathbf{X}]] + \mathrm{MultiHead}\big(\mathrm{LN}_1([[hidden:\mathbf{X}]])\big) \\ [[hidden:\mathbf{H}]] &= \mathrm{ReLU}\big(\mathrm{LN}_2([[hidden:\mathbf{X}]])\,[[param:\mathbf{W}_1]] + [[param:\mathbf{b}_1]]\big) \\ [[hidden:\mathbf{X}]] &\leftarrow [[hidden:\mathbf{X}]] + [[hidden:\mathbf{H}]]\,[[param:\mathbf{W}_2]] + [[param:\mathbf{b}_2]] \end{aligned}"),
        mo.md(r"""
        The MLP takes every vector to $4d$ numbers, $\mathbf{H}$, and back to $d$, as in GPT-2. GPT-2 uses GELU there, a smoother ReLU; here it is ReLU. Each part adds to $\mathbf{X}$ and never replaces it, so the vector of every token runs straight through all blocks: the **residual stream**. Each part reads a normalized copy of it and writes what it found back, by adding. Every block of a GPT has weights of its own.

        A GPT trains on a mini-batch of $B$ texts at once, a tensor of shape $(B, n, d)$. `nn.Linear` and `nn.LayerNorm` work on the last axis, whatever comes before it, so they take a mini-batch as they are. Attention needs a change: every text may only compare its own tokens. `@` on two tensors of 3 axes multiplies them text by text, and `k.transpose(1, 2)` swaps axes 1 and 2, so that every text gets the transpose of its own keys. `q @ k.transpose(1, 2)` then gives $B$ matrices of scores of $n$ by $n$, one per text. The columns of a head are `q[:, :, a:b]`. The softmax and the heads side by side go along axis 2, and the mask of $n$ by $n$ applies to every text at once, by broadcasting.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **transformer block** | | multi-head causal attention and an MLP, each with layer normalization before it and a residual connection around it | `Block(d, h)` |
        | **residual stream** | | the vectors of the tokens on their way through all blocks; every part of every block adds to them | `x = x + ...` |
        | $(B, n, d)$ | B by n by d | the shape of a mini-batch of texts: texts, tokens, numbers per token | `x.shape` |
        | `transpose(1, 2)` | transpose 1 2 | swaps axes 1 and 2 of a tensor: in a mini-batch, the transpose of every text | `k.transpose(1, 2)` |
        """),
        L.watch_out("Normalize a copy, not the stream: in `x = x + self.attention(self.norm1(x))`, `x` itself goes on unchanged. And `.T` is not the transpose of a mini-batch: on a tensor of 3 axes it turns all axes around, $(B, n, d)$ into $(d, n, B)$."),
        mo.md(r"""
        The picture: one block. The residual stream runs straight down; each part takes a normalized copy of it and adds its result back, at ⊕.
        """),
        mo.Html(C.block_picture({"title": "One transformer block", "input": "x: n vectors of d numbers", "output": "to the next block",
                                 "norm": "layer normalization", "attention": "multi-head causal attention", "mlp": "MLP: d → 4d → d",
                                 "stream": "residual stream",
                                 "alt": "The residual stream runs from x, n vectors of d numbers, straight down to the next block. Two branches leave it: layer normalization, then multi-head causal attention, added back to the stream; then layer normalization, then an MLP from d to 4d to d numbers, added back again."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    A transformer block for a mini-batch $x$ of shape $(B, n, d)$, with $h$ heads:

    1. A mini-batch $x$ of shape $(B, n, d)$; its queries, keys and values, of the same shape, and a mask of $n$ by $n$.
    2. Head `i`, for every text at once: its scores `q[:, :, a:b] @ k[:, :, a:b].transpose(1, 2)`, scaled and masked, the softmax along axis 2, times its values `v[:, :, a:b]`.
    3. The outputs of the heads side by side along axis 2, then the output projection.
    4. The block: $x$ plus the attention of the normalized $x$; then $x$ plus the MLP of the normalized $x$: from $d$ to $4d$ numbers, ReLU, and back to $d$.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Two texts of 3 tokens, with 4 numbers each: the scores of one head for both texts at once, and the MLP half of a block. The comments with `#` say which step of the recipe each part is. Step 4 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_block():
    import math
    import torch
    from torch import nn

    torch.manual_seed(0)
    # 1. a mini-batch of 2 texts of 3 tokens, 4 numbers each
    _x = torch.randn(2, 3, 4)
    _query, _key = nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False)
    _norm, _lin1, _lin2 = nn.LayerNorm(4), nn.Linear(4, 16), nn.Linear(16, 4)
    with torch.no_grad():
        # 2. the scores of one head of 4 numbers, for both texts at once
        _q, _k = _query(_x), _key(_x)
        _S = _q @ _k.transpose(1, 2) / math.sqrt(4)
        print(_k.shape, _k.transpose(1, 2).shape, _S.shape)
        # 4. the MLP half of a block: x plus the MLP of the normalized x
        _y = _lin2(torch.relu(_lin1(_norm(_x))))  # complete: _x + in front
        print(_x[0, 0])
        print(_y[0, 0])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The keys have shape (2, 3, 4); swapped, (2, 4, 3); and the scores (2, 3, 3): a matrix of 3 by 3 for each text. As it is, `_y[0, 0]` is `[-0.1175, 0.0982, -0.0675, -0.3728]`: only what the MLP makes of the first token, and its own vector `[-1.1258, -1.1524, -0.2506, -0.4339]` is gone. Completed, it is `[-1.2433, -1.0542, -0.3181, -0.8067]`: the vector of the token, plus that small change. That is the residual stream.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_no_attention():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. A GPT without attention
    """)
    return


@app.cell(hide_code=True)
def choice_no_attention_cell():
    choice_no_attention = L.choice(
        "no-attention",
        question="You take the attention out of every block of a GPT, and keep the layer normalizations and the MLPs. What can it still predict the next token from?",
        options={
            "before": "From all tokens before it: the residual stream carries them along",
            "few": "From the last few tokens, as the neural language model did",
            "right": "Only from the token itself and its place",
        },
        correct="right",
        why={
            "before": "The residual stream of a token only carries what is added to that token's own vector. Without attention, nothing of other tokens is ever added to it.",
            "few": "That model of [A neural language model that looks further back](/course/open/lm-mlp-context?lang=en) put the last few embeddings side by side before its MLP. The MLP of a block gets the vector of one token at a time.",
            "right": "Nothing else reaches its vector: layer normalization and the MLP work on the vector of every token by itself. Attention is the only place where the tokens of a text exchange information.",
        },
    )
    choice_no_attention
    return (choice_no_attention,)


@app.cell(hide_code=True)
def text_exercise_attention():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Attention for a mini-batch

        Complete `CausalSelfAttention(d, h)`: the multi-head attention of [Multi-head attention](/course/open/attn-multi-head?lang=en), for a mini-batch `x` of shape $(B, n, d)$. `__init__`, the mask and the loop over the heads are given; replace the two `...` by steps 2 and 3 of the recipe. The check makes it after `torch.manual_seed(0)` and gives it 2 texts of 3 tokens; then it tries 3 heads and longer texts. When it is right, the check saves it in your project, as `work/projects/llm/attention.py`.
        """),
        L.task("Make `CausalSelfAttention` run causal attention in every head, for every text of a mini-batch at once."),
    ])
    return


@app.cell(hide_code=True)
def exercise_attention():
    box_attention = L.code_box("causal-self-attention", lines=26)
    box_attention
    return (box_attention,)


@app.cell(hide_code=True)
def check_attention(box_attention):
    result_attention = L.check("causal-self-attention", box_attention.value)
    result_attention
    return (result_attention,)


@app.cell(hide_code=True)
def hints_attention():
    L.hints(
        "causal-self-attention",
        [
            "The lines of causal attention of [The causal mask](/course/open/attn-causal-mask?lang=en), with `[:, :, a:b]` for the columns of head `i`, `transpose(1, 2)` instead of `.T`, and axis 2 instead of axis 1.",
            "Step 2 of the worked example gives the scores, divided by `math.sqrt(self.d_head)`; `S.masked_fill(future, float(\"-inf\"))` hides the future in every text at once. After the loop, `torch.cat(outputs, dim=2)` goes through `self.out`.",
        ],
        why="The scores have shape $(B, n, n)$ and the mask $(n, n)$: PyTorch uses the same mask for every text. `@` multiplies text by text, so the texts of a mini-batch never mix.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_block(result_attention):
    result_attention  # made again after the check of exercise 2, which saves your attention
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Your transformer block

        Complete `Block(d, h)`: in `__init__` the five layers, in the order the data passes them; in `forward`, step 4 of the recipe. Its first lines import your `CausalSelfAttention` from your project. The check makes the block after `torch.manual_seed(0)` and gives it 2 texts of 3 tokens with 8 numbers each; then it sends a text through the block twice, and looks at the shapes of the parameters. When it is right, the check saves it as `work/projects/llm/block.py`.
        """),
        L.course_versions(),
        L.task("Make `Block` add the attention and then the MLP, each of a normalized copy, to the residual stream."),
    ])
    return


@app.cell(hide_code=True)
def exercise_block():
    box_block = L.code_box("block", lines=16)
    box_block
    return (box_block,)


@app.cell(hide_code=True)
def check_block(box_block, result_attention):
    result_attention  # check again when exercise 2 saves your attention
    result_block = L.check("block", box_block.value)
    result_block
    return (result_block,)


@app.cell(hide_code=True)
def hints_block():
    L.hints(
        "block",
        [
            "`__init__` is that of your `PreNormBlock` of [Layer normalization](/course/open/nn-layernorm?lang=en), twice: `nn.LayerNorm(d)` and `CausalSelfAttention(d, h)`, then `nn.LayerNorm(d)` and the two `nn.Linear` layers of the MLP, `d` to `4 * d` and back.",
            "`forward` has two lines like `x = x + self.attention(self.norm1(x))`: the second with the MLP of the worked example, `self.lin2(torch.relu(self.lin1(...)))`, of `self.norm2(x)`; then `return x`.",
        ],
        why="The block gives back the shape it gets, $(B, n, d)$, so blocks can be stacked: what comes out of one goes into the next, along the residual stream.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    The cell below uses your project. It stacks 4 of your blocks, with $d = 32$ in 4 heads, made after `torch.manual_seed(0)`, and sends 2 texts of 8 random vectors through them. Then it changes the last token of the first text, and looks at what changes. It also makes one block of GPT-2's size, $d = 768$ in 12 heads, and counts its parameters. It works once exercises 2 and 3 are done.
    """)
    return


@app.cell(hide_code=True)
def demo_blocks(result_attention, result_block):
    result_attention, result_block  # run again when exercise 2 or 3 saves your module
    _result = C.demo()
    if _result is None:
        _out = mo.md("*Finish exercises 2 and 3 first: then your `CausalSelfAttention` and your `Block` are in your project, and this cell uses them.*")
    elif isinstance(_result, str):
        _out = mo.md({
            "slow": "*Your blocks take too long here: more than half a minute. Does your `Block` or your `CausalSelfAttention` do more work than it needs, such as a loop that runs too often?*",
            "block": "*`Block` or `CausalSelfAttention` in `work/projects/llm/` stops with an error. Did you change its file? Delete it, and check exercise 2 or 3 again: its check then saves your code again.*",
        }[_result])
    else:
        _by = lambda v: "not at all" if v == 0 else "by up to " + C.number(v, 3, L.lang)
        _ahead = ("your blocks never look ahead" if _result["before"] == 0
                  else "so your blocks look ahead: look at the mask of your `CausalSelfAttention`")
        _mixed = ("and the texts of a mini-batch do not mix" if _result["other"] == 0
                  else "so the texts of the mini-batch mix: look at the axes in your `CausalSelfAttention`")
        _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", as the note at exercise 3 says."
        _out = mo.md(f"""Out of your 4 blocks comes shape {tuple(_result['shape'])}, the shape that went in. Changing the last token of the first text changes its own new vector by up to {C.number(_result['last'], 3, L.lang)}, those of the 7 tokens before it {_by(_result['before'])}, and those of the second text {_by(_result['other'])}: {_ahead}, {_mixed}.

One block of GPT-2's size has **{C.grouped(_result['params'], L.lang)}** parameters, so the 12 blocks of the smallest GPT-2 have {C.grouped(12 * _result['params'], L.lang)}. GPT-2 itself has 2,304 more per block: bias terms on its queries, keys and values.{_who}""")
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The multi-head attention of [Multi-head attention](/course/open/attn-multi-head?lang=en), now for a mini-batch, and the pre-norm residual block of [Layer normalization](/course/open/nn-layernorm?lang=en), twice: that is the transformer block. The positions of [Telling attention where things are](/course/open/attn-positions?lang=en) go in at the start, and the residual stream carries them through every block. Your project `llm` now has `attention.py` and `block.py`. In *Assemble your GPT*, embeddings, a stack of your blocks and an output head become your GPT. First, on the main route, you build your own tokenizer, starting with *Byte pair encoding: build your own tokenizer*.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_no_attention, result_attention, result_block):
    L.wrap_up(choice_no_attention, result_attention, result_block)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
