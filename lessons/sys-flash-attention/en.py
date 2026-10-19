# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-flash-attention. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-flash-attention", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Memory or compute: what limits your GPU?](/course/open/sys-memory-roofline?lang=en) and [The causal mask](/course/open/attn-causal-mask?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_memory():
    quick_memory = L.choice(
        "quick-memory",
        question="Where on a GPU does a group of threads keep data that it reuses many times, close to its cores?",
        options={
            "gpu": "In GPU memory",
            "right": "In shared memory",
            "host": "In host memory",
        },
        correct="right",
        why={
            "gpu": "GPU memory is large, but every read from it costs bandwidth. See [Memory or compute: what limits your GPU?](/course/open/sys-memory-roofline?lang=en).",
            "host": "Host memory is the computer's own RAM, behind the slow PCIe connection. See [Memory or compute: what limits your GPU?](/course/open/sys-memory-roofline?lang=en).",
            "right": "Shared memory: on an RTX 4090, about 128 KB per streaming multiprocessor, small and fast. FlashAttention keeps its blocks there.",
        },
    )
    quick_memory
    return (quick_memory,)


@app.cell(hide_code=True)
def quick_check_causal():
    quick_causal = L.choice(
        "quick-causal",
        question="In causal attention, which scores does the softmax of row $i$ use?",
        options={
            "right": "The scores of positions 0 to $i$",
            "all": "All $n$ scores of the row",
            "self": "Only the score of position $i$ itself",
        },
        correct="right",
        why={
            "all": "The mask sets the scores of the positions after $i$ to $-\\infty$, so they get weight 0. See [The causal mask](/course/open/attn-causal-mask?lang=en).",
            "self": "A position may take from itself and from every position before it. See [The causal mask](/course/open/attn-causal-mask?lang=en).",
            "right": "The scores after $i$ are $-\\infty$ and get weight 0. In blocks, whole blocks of them can be skipped.",
        },
    )
    quick_causal
    return (quick_causal,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Attention compares every token with every token before it: for a text of $n$ tokens, $n \times n$ scores per head. Worked out and stored, that matrix grows with the square of the length, and reading and writing it costs more time than the multiplications. **FlashAttention** never stores it: it works through the scores in blocks that fit in shared memory, with a softmax that is updated block by block. It is why language models can read texts of tens of thousands of tokens, and it is what PyTorch uses when you call its attention.
    """)
    return


@app.cell(hide_code=True)
def predict_memory():
    guess_memory = L.guess(
        "score-memory",
        question="A model reads a text of 32,768 tokens with 32 heads. Stored in float32, how much memory do the score matrices of one layer take?",
        options={
            "one": "About 1 GB",
            "ten": "About 10 GB",
            "right": "More than 100 GB",
        },
        reveal=f"**More than 100 GB:** $32{{,}}768^2$ scores of 4 bytes is {C.size(32768 ** 2 * 4, 'en')} per head, and 32 heads make {C.size(32 * 32768 ** 2 * 4, 'en')}: more than five times the 24 GB of an RTX 4090. The text itself, with $d = 4{{,}}096$, is {C.size(32768 * 4096 * 4, 'en')}.",
    )
    guess_memory
    return (guess_memory,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Online softmax.** For one query, attention takes the scores $s_j$ with every key, their softmax, and the weighted sum of the values $\mathbf{v}_j$. Instead of all scores at once, take them a block at a time, and keep three things: the largest score so far $m$, the sum $\ell$ of $e^{s_j - m}$, and the sum $\mathbf{o}$ of $e^{s_j - m} \mathbf{v}_j$. When a block arrives, its maximum may be larger: the new maximum is
        """),
        L.formula(r"m' = \max(m, \max_j s_j)"),
        mo.md(r"""
        and every term added so far must now be taken relative to $m'$: since $e^{s - m'} = e^{s - m} \, e^{m - m'}$, that is one factor for everything at once. Then add the block:
        """),
        L.formula(r"\ell' = \ell \, e^{m - m'} + \sum_j e^{s_j - m'}"),
        L.formula(r"[[hidden:\mathbf{o}']] = [[hidden:\mathbf{o}]] \, e^{m - m'} + \sum_j e^{s_j - m'} [[hidden:\mathbf{v}_j]]"),
        mo.md(r"""
        After the last block, the output is $\mathbf{o} / \ell$: exactly the output of softmax attention, because in the end every term is taken relative to the same maximum. The factor $e^{m - m'}$ is at most 1, so nothing overflows.

        **FlashAttention** (Dao and others, 2022) does this for blocks of queries at once, in one fused kernel: it loads blocks of queries, keys and values into shared memory, works through them there, and writes back to GPU memory only the output, with $m$ and $\ell$. The $n \times n$ matrix never exists in GPU memory: memory grows with $n$, not $n^2$, and the traffic to GPU memory, which limits attention, drops by far. With the causal mask, the blocks above the diagonal are skipped altogether. FlashAttention-2 gives every block of queries its own program, which walks through all blocks of keys and values, and so shares the work out better over the GPU; later versions tune it to newer GPUs.

        **Two halves.** Two programs can each take half of the keys: each ends with its own $m$, $\ell$ and $\mathbf{o}$, and the two states are merged by rescaling both to the larger maximum. That is how long texts are split over more programs.

        The picture: the score matrix in blocks of keys and queries, with the causal mask. The blocks on and below the diagonal are worked out; the ones above are skipped. Query block 5 walks its row from left to right, updating $m$, $\ell$ and $\mathbf{o}$ after each block.
        """),
        mo.Html(C.blocks_picture({"title": "Causal attention in blocks", "keys": "blocks of keys and values →", "queries": "blocks of queries",
                                  "walk": "m, ℓ, o", "note": "dashed: skipped",
                                  "alt": "A grid of 8 by 8 squares. The squares on and below the diagonal are filled; those above are dashed outlines. Row 5 is filled darker, with an arrow running through its 5 squares from left to right, marked m, l, o."})),
        mo.md(r"""
        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **online softmax** | | a softmax worked out block by block, with a running maximum and a running sum that are rescaled when the maximum grows | |
        | **FlashAttention** | | attention in one fused kernel that walks through blocks of keys in shared memory and never stores the score matrix | |
        | $m$, $\ell$, $\mathbf{o}$ | m, ell, o | the running maximum, the running sum of $e^{s - m}$, and the running weighted sum of the values | `m`, `l`, `o` |
        """),
        L.watch_out("Without a mask, FlashAttention does the same multiplications as attention that stores the matrix, and in training a few more: the backward pass works the scores out again instead of reading them back. It is faster because it moves far less data to and from GPU memory. With the causal mask, it also skips the blocks above the diagonal, which stored attention works out and then masks."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Attention for one query, a block of keys at a time:

    1. Start with $m = -\infty$, $\ell = 0$ and $\mathbf{o} = \mathbf{0}$.
    2. For every block: its scores, and the new maximum $m'$.
    3. Multiply $\ell$ and $\mathbf{o}$ by $e^{m - m'}$; add the terms $e^{s_j - m'}$ to $\ell$, and $e^{s_j - m'} \mathbf{v}_j$ to $\mathbf{o}$.
    4. After the last block: the output is $\mathbf{o} / \ell$.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Steps 1 to 3 without the values: the sum $\ell$ that the softmax divides by, for the scores 1, 3, 2 and 5, in blocks of 2, against the sum taken all at once with the largest score taken off.
    """)
    return


@app.cell
def example_online_sum():
    import math

    _scores = [1.0, 3.0, 2.0, 5.0]
    _m, _l = float("-inf"), 0.0
    for _start in range(0, len(_scores), 2):
        _block = _scores[_start:_start + 2]
        _m_new = max(_m, max(_block))
        _l = _l * math.exp(_m - _m_new) + sum(math.exp(_s - _m_new) for _s in _block)
        _m = _m_new
        print("after the block", _block, "m =", _m, "l =", round(_l, 4))
    print("at once:", round(sum(math.exp(_s - max(_scores)) for _s in _scores), 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    After the first block, $m = 3$ and $\ell = e^{-2} + e^{0} \approx 1.1353$. The second block brings 5: $\ell$ is multiplied by $e^{3 - 5} \approx 0.1353$, and $e^{-3} + e^{0}$ is added, which gives 1.2034, the same as the sum taken all at once against 5. The weight of each score is $e^{s - m} / \ell$; with the values, the same steps give the output.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_online():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Online attention for one query

        Complete `online_attention(q, K, V, block)`: attention for the query `q` over the keys in the rows of `K` and the values in the rows of `V`, taking `block` keys at a time. The loop, the scores of a block and the new maximum are there; replace the `...` with step 3 of the recipe. The check calls it with a query of 2 numbers and 5 keys in blocks of 2, which should give about [1.6059, 1.3563], and tries blocks of 1, one block of all keys, and other sizes.
        """),
        L.task("Make `online_attention` rescale `l` and `o` to the new maximum, and add each block."),
    ])
    return


@app.cell(hide_code=True)
def exercise_online():
    box_online = L.code_box("online-attention", lines=16)
    box_online
    return (box_online,)


@app.cell(hide_code=True)
def check_online(box_online):
    result_online = L.check("online-attention", box_online.value)
    result_online
    return (result_online,)


@app.cell(hide_code=True)
def hints_online():
    L.hints(
        "online-attention",
        [
            "The factor `math.exp(m - m_new)` for what you have, and `torch.exp(s - m_new)` for the terms of the block; `p @ V[start:start + block]` adds the values weighted by those terms.",
        ],
        why="`scale = math.exp(m - m_new)`, `p = torch.exp(s - m_new)`, `l = l * scale + p.sum().item()` and `o = o * scale + p @ V[start:start + block]`. In the first block, `m` is minus infinity, and `math.exp` of minus infinity is 0: there is nothing yet to rescale.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_merge():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Two halves together

        Two programs each took half of the keys of one query, and ended with the states `(m1, l1, o1)` and `(m2, l2, o2)`, with `o1` and `o2` lists. Complete `merge`: give back the state `(m, l, o)` of all keys together. The check calls `merge(2.0, 1.5, [3.0, 1.0], 3.0, 1.0, [2.0, 0.0])`, which should give about (3, 1.5518, [3.1036, 0.3679]), and tries other states.
        """),
        L.task("Make `merge` rescale both states to the larger maximum and add them up."),
    ])
    return


@app.cell(hide_code=True)
def exercise_merge():
    box_merge = L.code_box("merge", lines=8)
    box_merge
    return (box_merge,)


@app.cell(hide_code=True)
def check_merge(box_merge):
    result_merge = L.check("merge", box_merge.value)
    result_merge
    return (result_merge,)


@app.cell(hide_code=True)
def hints_merge():
    L.hints(
        "merge",
        [
            "Each state was taken relative to its own maximum: its factor to the new maximum is $e$ to the power of its own maximum minus the new one.",
        ],
        why="`m = max(m1, m2)`, `a, b = math.exp(m1 - m), math.exp(m2 - m)`, `l = l1 * a + l2 * b` and `o = [o1[i] * a + o2[i] * b for i in range(len(o1))]`. In the example, part 1 is rescaled by $e^{2 - 3} \\approx 0.3679$, and part 2 stays as it is.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Stored or in blocks, timed

    The cell below runs causal attention on one text of 2,048 tokens with 8 heads of 64 numbers, on your GPU if you have one, else on your CPU: once with the score matrix worked out, masked and stored, and once with PyTorch's `F.scaled_dot_product_attention`, which works in blocks, as FlashAttention does, where the hardware allows it, and skips the blocks above the diagonal. It takes a few seconds.
    """)
    return


@app.cell(hide_code=True)
def demo_timed():
    _d = C.demo()
    _where = "your GPU" if _d["device"] == "cuda" else "your CPU"
    mo.md(
        f"On {_where}, with the matrix stored: {L.number(_d['stored'] * 1000, 0)} ms; in blocks: {L.number(_d['fused'] * 1000, 0)} ms, "
        f"{L.number(_d['stored'] / _d['fused'], 1)} times as fast. The stored score matrices take {C.size(_d['matrix'], 'en')}; "
        f"the two outputs differ by at most {L.number(_d['diff'] * 1e6, 1)} millionths, the rounding of float32."
    )
    return


@app.cell(hide_code=True)
def text_exercise_faster():
    mo.md(r"""
    ### Exercise 3. Why it is faster
    """)
    return


@app.cell(hide_code=True)
def choice_faster_cell():
    choice_faster = L.choice(
        "why-faster",
        question="Without a mask, FlashAttention does the same multiplications as attention that stores the score matrix. Why is it faster on a GPU?",
        options={
            "skip": "It leaves out multiplications whose weights would be small",
            "right": "It does not write the scores and weights to GPU memory and read them back",
            "precision": "It always computes in a lower precision",
        },
        correct="right",
        why={
            "skip": "Every score is worked out: the output is exactly that of softmax attention. Only with the causal mask does it skip blocks, the ones above the diagonal, and those are masked anyway.",
            "precision": "It computes in the same number format as other attention; the speed comes from the traffic it avoids.",
            "right": "Attention is limited by memory traffic, not by multiplications. Stored, the $n \\times n$ scores go to GPU memory and back several times; in blocks, they stay in shared memory.",
        },
    )
    choice_faster
    return (choice_faster,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The fused kernel of [Write your own GPU kernel](/course/open/sys-triton-kernel?lang=en), with the stable softmax taken one step further: a maximum that is updated along the way. The traffic it avoids is that of [Memory or compute: what limits your GPU?](/course/open/sys-memory-roofline?lang=en), and the blocks it skips are those of [The causal mask](/course/open/attn-causal-mask?lang=en). In PyTorch, `F.scaled_dot_product_attention` picks such a kernel for you.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_faster, result_merge, result_online):
    L.wrap_up(result_online, result_merge, choice_faster)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
