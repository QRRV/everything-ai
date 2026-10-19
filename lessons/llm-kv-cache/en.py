# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-kv-cache. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("llm-kv-cache", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Where the memory goes](/course/open/sys-memory-accounting?lang=en) and [Grouped-query attention](/course/open/llm-gqa?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_weights():
    quick_weights = L.choice(
        "quick-weights",
        question="You want to use a model of 13 billion parameters in BF16, not train it. How much memory do its weights take?",
        options={
            "one": "13 GB",
            "right": "26 GB",
            "fp32": "52 GB",
        },
        correct="right",
        why={
            "one": "That is one byte per parameter. BF16 takes 2 bytes per number: see [Where the memory goes](/course/open/sys-memory-accounting?lang=en).",
            "fp32": "That is FP32, 4 bytes per parameter. BF16 takes 2: see [Where the memory goes](/course/open/sys-memory-accounting?lang=en).",
            "right": "2 bytes per parameter. This lesson adds what a model keeps besides its weights while it writes.",
        },
    )
    quick_weights
    return (quick_weights,)


@app.cell(hide_code=True)
def quick_check_width():
    quick_width = L.choice(
        "quick-width",
        question="A layer has 32 query heads and 8 key and value heads, each 128 numbers wide. How many numbers does its key projection make per token?",
        options={
            "right": "1,024",
            "query": "4,096",
            "one": "128",
        },
        correct="right",
        why={
            "query": "That is the width of the queries, 32 heads of 128. There are only 8 key heads: see [Grouped-query attention](/course/open/llm-gqa?lang=en).",
            "one": "That is one head. The key projection makes all 8 key heads side by side: see [Grouped-query attention](/course/open/llm-gqa?lang=en).",
            "right": "8 key heads of 128 numbers, and as many values. This lesson keeps them, and counts what that costs.",
        },
    )
    quick_width
    return (quick_width,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Generate text with your GPT](/course/open/gpt-generate?lang=en) your GPT made a whole forward pass over the window for every new character. Most of that pass does again what the step before already did: because of the causal mask, a token never looks at the tokens after it, so in every layer its keys and values stay the same when the text grows. The **KV cache** keeps them, so that a step only works on the new token. Every transformer LLM uses one when it writes. It costs memory that grows with every token, and that memory decides how long the texts can be and how many people one GPU can serve at once.
    """)
    return


@app.cell(hide_code=True)
def predict_cache():
    guess_cache = L.guess(
        "llama-cache",
        question="Llama 2 7B has 32 layers, each with 32 heads of 128 numbers, and no grouped-query attention. It writes in BF16 and has 4,096 tokens of text so far. How much memory does its KV cache take?",
        options={
            "small": "About 20 MB",
            "right": "About 2 GB",
            "weights": "About 13 GB",
        },
        reveal="**About 2 GB.** Every token keeps, in each of the 32 layers, a key and a value of $32 \\times 128 = 4{,}096$ numbers of 2 bytes: $2 \\times 32 \\times 4{,}096 \\times 2$ bytes, 0.52 MB per token. Times 4,096 tokens: 2.1 GB, for one text. A GPU that writes 8 such texts at once keeps 17 GB of cache, more than the 13.5 GB of the weights.",
    )
    guess_cache
    return (guess_cache,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Step 1: what stays the same.** With the causal mask, the token at place $t$ only looks at places $1$ to $t$. So a token added at the end changes nothing at the places before it: in every layer, their vectors, and so their keys and values, are what they were. Without a cache, every step computes them all again.

        **Step 2: the cache.** A KV cache keeps, in every layer, the keys and values of all tokens so far: tensors of shape (B, t, d), one row per token. The first pass, over the start of the text, fills it. After that, a step sends only the new token through the model. In every layer it makes the new token's query, key and value, adds the key and value at the end of the cache, and compares the query with every key in the cache. No mask is needed: the cache holds only earlier tokens and the new one.

        **Step 3: what it saves.** Without a cache, the step with $t$ tokens sends all $t$ through every layer; with a cache, one. The new query still reads all $t$ keys and values, but the projections and the MLPs, most of the FLOPs, work on one token. The outcome is the same up to rounding: the cache saves work and changes nothing.

        The picture: a start of 3 tokens, then three steps that each add one. Boxes that stand out go through the model in that step; faint ones come from the cache.
        """),
        mo.Html(C.steps_picture({"title": "Four steps of writing", "without": "without a cache", "with": "with a KV cache", "step": "step {n}",
                                 "computed": "through the model", "cached": "from the cache",
                                 "alt": "Two grids of four rows, one row per step, with 3, 4, 5 and 6 boxes, one per token. Without a cache, every box of every row stands out: each step sends every token through the model, 18 in all. With a KV cache, the first row has 3 boxes that stand out, the start; in the other rows only the last box stands out and the others are faint, read from the cache: 6 in all."})),
        mo.md(r"""
        **Step 4: what it costs.** Every token keeps, in every layer, a key and a value for each of the $g$ key and value heads. In bytes:
        """),
        L.formula(r"M = 2 \cdot L \cdot g \cdot d_\text{head} \cdot n \cdot B \cdot b"),
        mo.md(r"""
        with $L$ layers, $n$ tokens, $B$ texts at once and $b$ bytes per number. It grows with every token and every text. And a step reads all of it, along with all the weights, for the work of one token per text: with few texts at once, writing is memory-bound. This is where grouped-query attention pays off: $h / g$ times less cache.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **KV cache** | the keys and values of every earlier token, kept in every layer while a model writes | `k_cache`, `v_cache` |
        | $L$ | the number of layers | `n_layers` |
        | $n$, $B$ | the tokens kept per text, and the texts at once | `n_tokens`, `batch` |
        | $b$ | the bytes per number: 2 in BF16, 4 in FP32 | `bytes_per_number` |
        """),
        L.watch_out("A cache only works while the text fits in the window. Your GPT has a table of `block` places: once the text is longer, the window moves on with every new token, every token gets another place, and all keys and values change. From then on every step needs a whole pass again, and the cache saves nothing."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To write with a KV cache:

    1. Send the start of the text through the model once, and keep in every layer the keys and values of all its tokens.
    2. For each new token, in every layer: make its query, key and value, and add the key and value at the end of the layer's cache, `torch.cat([k_cache, k], dim=1)`.
    3. Compare its query with every key in the cache: scaled, through softmax, times the values of the cache, without a mask.
    4. To count the memory: $2 \cdot L \cdot g \cdot d_\text{head} \cdot n \cdot B \cdot b$ bytes.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A tensor that grows is new: a cache that starts with no tokens and gets one key per step.
    """)
    return


@app.cell
def example_grow():
    import torch

    _k_cache = torch.zeros(1, 0, 2)  # one text, no tokens yet, keys of 2 numbers
    for _step in range(3):
        _k = torch.randn(1, 1, 2)  # the key of the new token
        _k_cache = torch.cat([_k_cache, _k], dim=1)
        print("after step", _step, "the cache has shape", tuple(_k_cache.shape))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `torch.zeros(1, 0, 2)` is a tensor with no tokens at all: the cache before the first step. `torch.cat` with `dim=1` puts the new key after the others, on the axis of the tokens: the cache grows by one row per step, and the axis of the texts and the axis of the numbers stay as they are.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_step():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. One step with a cache

        Complete `attend_step(q, k, v, k_cache, v_cache)`, one step of one attention head with a cache. `q`, `k` and `v` are the query, key and value of the new token, shape (B, 1, d); `k_cache` and `v_cache` hold the keys and values of the earlier tokens, (B, t, d). Add the new key and value at the end of the caches, compute the attention of the new token, and give back `out`, (B, 1, d), and the two new caches. The check calls `attend_step(torch.randn(1, 1, 4), torch.randn(1, 1, 4), torch.randn(1, 1, 4), torch.randn(1, 2, 4), torch.randn(1, 2, 4))`, a new token and a cache of 2 tokens, and tries an empty cache, more texts at once, a longer cache and other widths.
        """),
        L.task("Complete `attend_step`: add the new key and value to the caches, then compute the attention of the new token over every key in the cache."),
    ])
    return


@app.cell(hide_code=True)
def exercise_step():
    box_step = L.code_box("attend-step", lines=12)
    box_step
    return (box_step,)


@app.cell(hide_code=True)
def check_step(box_step):
    result_step = L.check("attend-step", box_step.value)
    result_step
    return (result_step,)


@app.cell(hide_code=True)
def hints_step():
    L.hints(
        "attend-step",
        [
            "Steps 2 and 3 of the recipe: the worked example shows how a cache grows, for the keys and for the values. Then attention as in causal attention, with `q` and the whole caches, but without a mask.",
        ],
        why="`k_cache = torch.cat([k_cache, k], dim=1)` and the same for `v_cache`; then `S = q @ k_cache.transpose(1, 2) / math.sqrt(q.shape[2])` and `out = torch.softmax(S, dim=2) @ v_cache`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_gb():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. What the cache costs

        Write `kv_cache_gb(n_layers, n_kv_heads, d_head, n_tokens, batch, bytes_per_number)`: the memory of a KV cache in GB, $10^9$ bytes. The check calls `kv_cache_gb(32, 8, 128, 8192, 1, 2)`, Llama 3 8B in BF16 with its window of 8,192 tokens full, which should give about 1.07, and tries other models, more texts at once, and FP32.
        """),
        L.task("Write `kv_cache_gb`: the bytes of every key and value the cache keeps, in GB."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gb():
    box_gb = L.code_box("kv-gb", lines=6)
    box_gb
    return (box_gb,)


@app.cell(hide_code=True)
def check_gb(box_gb):
    result_gb = L.check("kv-gb", box_gb.value)
    result_gb
    return (result_gb,)


@app.cell(hide_code=True)
def hints_gb():
    L.hints(
        "kv-gb",
        [
            "Step 4 of the idea: one token keeps a key and a value of `d_head` numbers for every key and value head, in every layer. The rest is multiplying, and dividing by `1e9`.",
        ],
        why="`return 2 * n_layers * n_kv_heads * d_head * n_tokens * batch * bytes_per_number / 1e9`: for Llama 3 8B, $2 \\times 32 \\times 8 \\times 128 \\times 2 = 131{,}072$ bytes per token.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fits():
    mo.md(r"""
    ### Exercise 3. How many texts fit
    """)
    return


@app.cell(hide_code=True)
def choice_fits_cell():
    choice_fits = L.choice(
        "fits",
        question="A GPU of 80 GB serves Llama 2 13B in BF16: 26 GB of weights, 40 layers with 40 heads of 128 numbers, no grouped-query attention. How many texts of 4,096 tokens can it write at once, if only the weights and the KV caches take memory? Your `kv_cache_gb` can help.",
        options={
            "all": "23",
            "keys": "32",
            "right": "16",
        },
        correct="right",
        why={
            "all": "That gives all 80 GB to the cache. The 26 GB of weights must fit too: 54 GB is left, and one text takes 3.36 GB.",
            "keys": "That counts only the keys, 1.68 GB per text. Every layer keeps a value per token too: 3.36 GB per text.",
            "right": "54 GB is left after the weights, and one text takes 3.36 GB. With 8 key and value heads, as Llama 2 70B has, one text would take 5 times less, and 80 texts would fit.",
        },
    )
    choice_fits
    return (choice_fits,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The loop of [Generate text with your GPT](/course/open/gpt-generate?lang=en), with the keys and values of earlier tokens kept instead of computed again, and the memory of [Where the memory goes](/course/open/sys-memory-accounting?lang=en) counted for them. The next lesson, *Project: upgrade your GPT to a modern LLM*, puts RMSNorm, SwiGLU, RoPE, grouped-query attention and a KV cache into your GPT.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fits, result_gb, result_step):
    L.wrap_up(result_step, result_gb, choice_fits)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
