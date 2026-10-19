# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson attn-multi-head. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("attn-multi-head", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [The causal mask](/course/open/attn-causal-mask?lang=en) and [Scaling the scores](/course/open/attn-scale?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_mask():
    quick_mask = L.choice(
        "quick-mask",
        question="With the causal mask, which positions can the third token of a text take from?",
        options={
            "right": "The first three: itself and the two before it",
            "all": "All positions: the mask only changes the scores",
            "before": "The first two: only the ones before it",
        },
        correct="right",
        why={
            "all": "The masked scores become minus infinity, so the weights of later positions are exactly 0. See [The causal mask](/course/open/attn-causal-mask?lang=en).",
            "before": "The diagonal stays: every position may take from itself. See [The causal mask](/course/open/attn-causal-mask?lang=en).",
            "right": "Today every head has its own causal attention, with the same mask.",
        },
    )
    quick_mask
    return (quick_mask,)


@app.cell(hide_code=True)
def quick_check_scale():
    quick_scale = L.choice(
        "quick-scale",
        question="Queries and keys of 16 numbers each: by what number does scaled attention divide the scores?",
        options={
            "sixteen": "16: the length of a key",
            "right": "4: the square root of 16",
            "none": "Nothing: only long keys need it",
        },
        correct="right",
        why={
            "sixteen": "It divides by the square root, $\\sqrt{d_k}$: dividing by $d_k$ would make the scores too small. See [Scaling the scores](/course/open/attn-scale?lang=en).",
            "none": "The scores spread about $\\sqrt{16} = 4$; dividing keeps that spread at 1 for any length. See [Scaling the scores](/course/open/attn-scale?lang=en).",
            "right": "Today attention is split into heads, narrower than the whole vector, so each divides by the square root of its own length.",
        },
    )
    quick_scale
    return (quick_scale,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    One attention gives every position one set of weights: one way of looking at the text. But a word needs several things at once. In "the black cat sat on the mat because it was tired", "it" needs to find what it refers to, "sat" its subject, and every word the words just before it. One softmax has to share its weight among all of that. Real transformers run several smaller attentions side by side: the smallest GPT-2 has 12 in every layer.
    """)
    return


@app.cell(hide_code=True)
def predict_weights():
    guess_weights = L.guess(
        "weights-per-pair",
        question="A sentence goes through attention with 4 heads side by side. How many attention weights does one position give to one other position?",
        options={
            "one": "1: the heads share one softmax",
            "right": "4: one for every head",
            "sixteen": "16: one for every pair of heads",
        },
        reveal="**4: one for every head.** Every head computes its own scores, its own softmax and its own average, so for every pair of positions there are 4 weights, which can be quite different. That is what the heads are for: four ways of looking at the same text at the same time.",
    )
    guess_weights
    return (guess_weights,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Multi-head attention** runs $h$ **attention heads** side by side. Each head has its own queries, keys and values of $d_\text{head} = d / h$ numbers, and its own causal scaled attention. In practice the matrices of all heads sit side by side in one matrix of $d$ by $d$: $Q = \mathbf{X}\mathbf{W}_Q$ has $d$ columns, and every head takes its own block of $d_\text{head}$ columns, of $Q$, $K$ and $V$ alike: head 1 the first block, head 2 the next. The outputs of the heads are put side by side again, $n$ by $d$, and an **output projection** $\mathbf{W}_O$, a linear layer from $d$ to $d$, mixes them:
        """),
        L.formula(r"\begin{aligned} [[hidden:\mathbf{H}_i]] &= \mathrm{softmax}\!\left(\frac{[[hidden:Q_i]] [[hidden:K_i]]^\top}{\sqrt{[[hyper:d_\text{head}]]}} + M\right) [[hidden:V_i]] \\ [[hidden:\mathbf{Y}]] &= [\,[[hidden:\mathbf{H}_1]] \; \cdots \; [[hidden:\mathbf{H}_h]]\,] \, [[param:\mathbf{W}_O]] + [[param:\mathbf{b}_O]] \end{aligned}"),
        mo.md(r"""
        Splitting adds no parameters: the matrices are as large as for one head of $d$ numbers, and the multiplications take as much work. What changes is that every head has its own scores and its own softmax, $h$ of them instead of one, so it can give its weight to its own words. The smallest GPT-2 has $d = 768$ in 12 heads of 64. In PyTorch, `torch.cat(outputs, dim=1)` puts a list of tensors side by side, column after column.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **attention head** | | one causal attention on its own block of columns of $Q$, $K$ and $V$ | `causal_attention(q[:, a:b], k[:, a:b], v[:, a:b])` |
        | **multi-head attention** | | $h$ heads side by side, their outputs put next to each other and mixed by an output projection | `self.out(torch.cat(outputs, dim=1))` |
        | **output projection** | | a linear layer from $d$ to $d$ after the heads, which mixes their outputs | `nn.Linear(d, d)` |
        | $d_\text{head}$ | d head | the number of columns of one head: $d / h$ | `d_head` |
        | `torch.cat` | torch cat | puts a list of tensors together, along one axis | `torch.cat(outputs, dim=1)` |
        """),
        L.watch_out("More heads do not mean more parameters: the same matrices of $d$ by $d$ are only cut into narrower heads. But more heads make every head narrower: with too many heads, each compares only a few numbers. Nobody tells a head what to look for either: what each head does comes out of training."),
        mo.md(r"""
        The picture: two heads of "the cat sat down", with made-up queries and keys of one number each, and the causal mask. Head 1 looks for a thing: from "cat" on, every word takes most from "cat". Head 2 looks for an action: from "sat" on, every word takes most from "sat". In one attention, these two would have to share one row of weights.
        """),
        mo.vstack(C.heads_pictures({"one": "Head 1: looks for a thing", "two": "Head 2: looks for an action", "rows": "word", "cols": "takes from", "decimal": ".",
                                    "one_alt": "Causal attention weights of head 1 for the cat sat down: from cat on, every word takes most from cat: 0.95, 0.91 and 0.87.",
                                    "two_alt": "Causal attention weights of head 2: cat takes 0.5 from the and 0.5 from itself; sat takes 0.91 from itself, and down 0.87 from sat."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Multi-head attention with $h$ heads, for queries, keys and values of $d$ numbers per position:

    1. The width of a head: $d_\text{head} = d / h$.
    2. Head `i`, with `i` from `range(h)`: the columns from `i * d_head` up to `(i + 1) * d_head` of $Q$, $K$ and $V$, through causal attention.
    3. The outputs of all heads side by side: `torch.cat(outputs, dim=1)`.
    4. The output projection: a linear layer from $d$ to $d$.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    3 positions with queries, keys and values of 4 numbers, in 2 heads of 2; the queries are also the keys here. `_causal_attention` is the causal attention of the last lesson. The comments with `#` say which step of the recipe each part is. Step 3 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_heads():
    import math
    import torch

    def _causal_attention(q, k, v):
        n, d_k = q.shape
        S = q @ k.T / math.sqrt(d_k)
        S = S.masked_fill(torch.tril(torch.ones(n, n)) == 0, float("-inf"))
        return torch.softmax(S, dim=1) @ v

    _Q = torch.tensor([[2.0, 0.0, 1.0, 0.0], [0.0, 2.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    _V = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.5, 0.5, 0.5, 0.5]])
    # 1. the width of a head
    _h = 2
    _d_head = int(_Q.shape[1] / _h)
    # 2. every head on its own columns
    _outputs = []
    for _i in range(_h):
        _a, _b = _i * _d_head, (_i + 1) * _d_head
        _outputs.append(_causal_attention(_Q[:, _a:_b], _Q[:, _a:_b], _V[:, _a:_b]))
    # 3. the outputs side by side
    _out = _outputs[0]  # complete: torch.cat(_outputs, dim=1)
    print(_out)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    As it is, the cell prints only the output of head 1, of shape (3, 2): the second position takes 0.944 of its value from itself, as in exercise 3 of [The causal mask](/course/open/attn-causal-mask?lang=en). Completed, it prints both heads side by side, of shape (3, 4). Head 2 sees other columns, `[1, 0]`, `[0, 1]` and `[1, 1]`, and gives the second position other weights: 0.330 from the first position and 0.670 from itself. One attention over all 4 columns would give it a single compromise, 0.076 and 0.924, for both halves.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_weights():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The parameters of the heads
    """)
    return


@app.cell(hide_code=True)
def choice_weights_cell():
    choice_weights = L.choice(
        "head-weights",
        question="Vectors of $d = 64$ numbers go into 4 heads of 16. How many parameters do the query matrices of all 4 heads have together?",
        options={
            "one": "1,024: 64 × 16, one head's matrix",
            "four": "16,384: 4 × 64 × 64, a full matrix per head",
            "right": "4,096: 64 × 64, as for one head of 64",
        },
        correct="right",
        why={
            "one": "That is one head. The 4 heads together take all 64 columns: 4 × 1,024.",
            "four": "Every head takes only its own 16 columns of one matrix of 64 by 64, not a full matrix of its own.",
            "right": "The 4 heads are 4 blocks of 16 columns of one matrix of 64 by 64. Splitting changes how the scores are computed, not how many parameters there are.",
        },
    )
    choice_weights
    return (choice_weights,)


@app.cell(hide_code=True)
def text_exercise_multi_head():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Heads side by side

        Complete `multi_head(Q, K, V, h)`: one query, key and value per row, with $d$ numbers each; it gives the outputs of $h$ causal heads side by side, without the output projection. `causal_attention` from the last lesson is given, and so are step 1 and the loop. Replace the two `...` following steps 2 and 3 of the recipe. The check calls `multi_head` with 3 positions of 4 numbers in 2 heads, which should give `[[1.0, 2.0, 3.0, 4.0], [3.679, 4.679, 6.7768, 7.7768], [6.0209, 7.0209, 7.0, 8.0]]`, and tries 1, 3 and 4 heads.
        """),
        L.task("Make `multi_head` run the heads on their own columns and put their outputs side by side."),
    ])
    return


@app.cell(hide_code=True)
def exercise_multi_head():
    box_multi_head = L.code_box("multi-head", lines=19)
    box_multi_head
    return (box_multi_head,)


@app.cell(hide_code=True)
def check_multi_head(box_multi_head):
    result_multi_head = L.check("multi-head", box_multi_head.value)
    result_multi_head
    return (result_multi_head,)


@app.cell(hide_code=True)
def hints_multi_head():
    L.hints(
        "multi-head",
        [
            "The loop of the worked example, with `Q`, `K`, `V` and the names of the function, and step 3 of the completed example.",
            "`Q[:, a:b]` keeps every row, every position, and takes columns `a` up to `b`: the block of one head.",
        ],
        why="Every head sees all positions, but only its own columns, so it compares other numbers and gives its own weights. `torch.cat(..., dim=1)` puts the blocks back next to each other, in the order of the heads.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_module():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Multi-head attention in PyTorch

        Complete the class `MultiHeadAttention(d, h)`: in `__init__` the learned layers `self.query`, `self.key` and `self.value`, each from `d` to `d` numbers without bias terms, and `self.out`, the output projection from `d` to `d`, made in that order; in `forward` the heads on their own columns, their outputs side by side, and `self.out` of that. `causal_attention` is given again, and so are `self.h`, `self.d_head` and the first line of `forward`. The check makes the module after `torch.manual_seed(0)`, with `d = 6` and 2 heads, and compares its output for 4 positions; then it tries 3 heads, and looks at the shapes of its parameters.
        """),
        L.task("Make `MultiHeadAttention` run the heads and mix their outputs with an output projection."),
    ])
    return


@app.cell(hide_code=True)
def exercise_module():
    box_module = L.code_box("multi-head-module", lines=26)
    box_module
    return (box_module,)


@app.cell(hide_code=True)
def check_module(box_module):
    result_module = L.check("multi-head-module", box_module.value)
    result_module
    return (result_module,)


@app.cell(hide_code=True)
def hints_module():
    L.hints(
        "multi-head-module",
        [
            "In `__init__`, four `nn.Linear` layers of `d` by `d`; the first three with `bias=False`, as in [Queries, keys and values](/course/open/attn-qkv?lang=en).",
            "In `forward`, the loop of exercise 2 with `q`, `k`, `v`, `self.h` and `self.d_head`, and `self.out` around `torch.cat`.",
        ],
        why="The three layers make $Q$, $K$ and $V$ for all heads at once; the loop cuts them into heads; `self.out` lets every number of the result use the outputs of all heads. For $d = 6$ that is $4 \\times 36 + 6 = 150$ parameters, with 2 heads or with 3.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The causal attention of [The causal mask](/course/open/attn-causal-mask?lang=en), with the scaling of [Scaling the scores](/course/open/attn-scale?lang=en), now in $h$ heads side by side, on the same matrices as one head. Something is still missing: attention compares contents, and apart from the mask nothing in it says where a word stands. The last word of "dog bites man" and of "bites dog man" gets the same new vector: it sees the same three words, and attention cannot tell in which order they came. In the next lesson, *Telling attention where things are*, every position gets a vector of its own place.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_weights, result_multi_head, result_module):
    L.wrap_up(choice_weights, result_multi_head, result_module)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
