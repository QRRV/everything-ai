# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-gqa. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("llm-gqa", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Multi-head attention](/course/open/attn-multi-head?lang=en) and [Assemble your GPT](/course/open/gpt-assemble?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_columns():
    quick_columns = L.choice(
        "quick-columns",
        question="Your attention has vectors of $d = 12$ numbers in 3 heads. Which columns of the keys does head 2 use, counting from head 0?",
        options={
            "first": "Columns 0 to 3",
            "right": "Columns 8 to 11",
            "all": "All 12 columns, like every head",
        },
        correct="right",
        why={
            "first": "Those are the columns of head 0: `i * d_head` to `(i + 1) * d_head`. See [Multi-head attention](/course/open/attn-multi-head?lang=en).",
            "all": "Every head works on its own columns, $d / h = 4$ of them. See [Multi-head attention](/course/open/attn-multi-head?lang=en).",
            "right": "Head $i$ has the columns $4i$ to $4i + 3$: for head 2, 8 to 11. This lesson lets several query heads use the same key columns.",
        },
    )
    quick_columns
    return (quick_columns,)


@app.cell(hide_code=True)
def quick_check_tokens():
    quick_tokens = L.choice(
        "quick-tokens",
        question="Your GPT reads a window of 64 tokens. How many keys does one head of one layer make?",
        options={
            "one": "One, for the last token",
            "square": "4,096, one for every pair of tokens",
            "right": "64, one for every token",
        },
        correct="right",
        why={
            "one": "Every token gets a query, a key and a value, so that later tokens can look at it. See [Assemble your GPT](/course/open/gpt-assemble?lang=en).",
            "square": "The scores come in pairs, $64 \\times 64$; the keys come one per token. See [Assemble your GPT](/course/open/gpt-assemble?lang=en).",
            "right": "And as many values. A GPT that generates can keep them for every earlier token instead of computing them again, as the next lesson shows: this lesson makes that cheaper.",
        },
    )
    quick_tokens
    return (quick_tokens,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A GPT that generates text can keep the keys and values of every token it has seen, in every head of every layer, instead of computing them again, and then reads all of them for every new token; the next lesson builds that. For a long text that memory, and reading it, are what slow generation down. **Grouped-query attention** lets several query heads share one key head and one value head. Llama 2 70B, Llama 3 and Mistral use it: the queries keep their variety, and the keys and values shrink.
    """)
    return


@app.cell(hide_code=True)
def predict_cache():
    guess_cache = L.guess(
        "fewer-keys",
        question="Llama 2 70B has 64 query heads in every layer, and only 8 key heads and 8 value heads. How much less memory do the keys and values it keeps take, compared with 64 of each?",
        options={
            "half": "Half",
            "right": "8 times less",
            "much": "64 times less",
        },
        reveal="**8 times less.** Each group of 8 query heads shares one key head and one value head: 8 of each instead of 64. The queries are not kept, so sharing them would save nothing.",
    )
    guess_cache
    return (guess_cache,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Step 1: one key and value head per query head.** In multi-head attention, every head has its own queries, keys and values: $h$ of each, every one $d_\text{head}$ columns wide.

        **Step 2: share them in groups.** **Multi-query attention** gives all $h$ query heads one key head and one value head. **Grouped-query attention** sits in between: $g$ key and value heads, each shared by a group of $h / g$ query heads next to each other; $g$ divides $h$, so every group is as large. Query head $i$ uses key and value head
        """),
        L.formula(r"j = \left\lfloor \frac{i}{h / g} \right\rfloor"),
        mo.md(r"""
        The brackets $\lfloor \, \rfloor$ round down to a whole number. With $h = 8$ and $g = 2$, query heads 0 to 3 use key head 0, and 4 to 7 key head 1. With $g = h$ this is multi-head attention again, with $g = 1$ multi-query attention.

        **Step 3: what it saves.** The key and value projections make $g \cdot d_\text{head}$ numbers per token instead of $h \cdot d_\text{head}$: fewer parameters, and above all $h / g$ times fewer keys and values to keep while generating. The scores and the weighted sums of the values take as many FLOPs as before: every query head still compares its queries with all keys.

        **Step 4: what it costs.** One shared head for all, as in multi-query attention, can cost some quality. The paper that introduced grouped-query attention found that 8 groups come close to the quality of multi-head attention, at close to the speed of multi-query attention.

        The picture: in every row 8 query heads, and under them, with a stronger outline, the key and value heads they use.
        """),
        mo.Html(C.heads_picture({"title": "Who shares keys and values", "mha": "multi-head", "gqa": "grouped, 2", "mqa": "multi-query",
                                 "lang": L.lang,
                                 "alt": "Three rows, each with 8 query heads above and their key and value heads below. Multi-head: 8 key and value heads, one for each query head. Grouped-query with 2 groups: 2 key and value heads, each shared by 4 query heads next to each other. Multi-query: 1 key and value head for all 8."})),
        mo.md(r"""
        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **multi-query attention** | all query heads share one key head and one value head | |
        | **grouped-query attention** | groups of query heads next to each other share one key head and one value head | |
        | $g$ | the number of key and value heads | `n_kv_heads` |
        | $\lfloor x \rfloor$ | floor of $x$: $x$ rounded down to a whole number, as `int(x)` does for $x$ not below 0 | `int(x)` |
        """),
        L.watch_out("A model is trained with its own number of key and value heads: the weights of a checkpoint fix it. To make $g$ smaller afterwards, you have to merge heads and train further: the paper averaged the key and value heads of every group and then trained for 5% of the original training."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To compute grouped-query attention with $h$ query heads and $g$ key and value heads:

    1. Make the queries $h \cdot d_\text{head}$ wide, the keys and values $g \cdot d_\text{head}$.
    2. For every query head $i$, find its key and value head: $j = \lfloor i / (h/g) \rfloor$, in Python `int(i / (n_heads / n_kv_heads))`.
    3. Run causal attention of the columns of query head $i$ with the columns of key and value head $j$.
    4. Put the outputs of the $h$ query heads side by side.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    What is new is which head each query head uses: 8 query heads, 2 key and value heads.
    """)
    return


@app.cell
def example_groups():
    _n_heads, _n_kv_heads = 8, 2
    _group = int(_n_heads / _n_kv_heads)
    for _i in range(_n_heads):
        print("query head", _i, "uses key and value head", int(_i / _group))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `_group` is 4: four query heads per key and value head. `int` drops everything after the decimal point, so `int(_i / _group)` sends query heads 0 to 3 to head 0 and 4 to 7 to head 1: groups of heads next to each other, as in Llama's code.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_gqa():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Grouped-query attention

        Complete `gqa(q, k, v, n_heads, n_kv_heads)`: `q` has the queries of `n_heads` heads side by side, shape (B, n, `n_heads * d_head`); `k` and `v` the keys and values of `n_kv_heads` heads, (B, n, `n_kv_heads * d_head`). Run causal attention for every query head with the key and value head of its group, as in [The transformer block](/course/open/attn-transformer-block?lang=en), and give back the outputs side by side, (B, n, `n_heads * d_head`). The check calls `gqa(torch.randn(1, 3, 8), torch.randn(1, 3, 4), torch.randn(1, 3, 4), 4, 2)`, 4 query heads of 2 numbers sharing 2 key heads, and tries 1 shared head, 1 head per query head, and more.
        """),
        L.task("Complete `gqa`: query head i uses the key and value head of its group, with causal attention as before."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gqa():
    box_gqa = L.code_box("gqa", lines=17)
    box_gqa
    return (box_gqa,)


@app.cell(hide_code=True)
def check_gqa(box_gqa):
    result_gqa = L.check("gqa", box_gqa.value)
    result_gqa
    return (result_gqa,)


@app.cell(hide_code=True)
def hints_gqa():
    L.hints(
        "gqa",
        [
            "Step 2 of the recipe, as in the worked example, gives the key and value head `j`; its columns run from `j * d_head` to `(j + 1) * d_head`. The rest is the loop body of your causal attention, with those columns for `k` and `v`.",
        ],
        why="Only the columns of the keys and values change: `j = int(i / (n_heads / n_kv_heads))`, and both `k` and `v` use the columns of head `j`, while `q` keeps the columns of head `i`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_share():
    mo.md(r"""
    ### Exercise 2. Who shares
    """)
    return


@app.cell(hide_code=True)
def choice_share_cell():
    choice_share = L.choice(
        "share",
        question="Mistral 7B has 32 query heads and 8 key and value heads in every layer. Which query heads use key and value head 1?",
        options={
            "right": "Heads 4, 5, 6 and 7",
            "turns": "Heads 1, 9, 17 and 25",
            "eight": "Heads 8 to 15",
        },
        correct="right",
        why={
            "right": "Groups of $32 / 8 = 4$ heads next to each other: `int(i / 4)` is 1 for heads 4 up to 7.",
            "turns": "That would be heads taking turns: 1, then 9, then 17. A group is heads next to each other: `int(i / 4)`.",
            "eight": "Those are 8 heads: a group has $32 / 8 = 4$ query heads.",
        },
    )
    choice_share
    return (choice_share,)


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ### Exercise 3. Why the keys and values
    """)
    return


@app.cell(hide_code=True)
def explain_why_keys():
    explain_keys_cell = L.explain(
        "why-keys",
        question="Grouped-query attention shares the keys and values between query heads, but gives every head its own queries. Why share those two and not the queries, and why does it cost little quality?",
        model_answer="**Why the keys and values:** while a GPT generates, it keeps the keys and values of every earlier token, in every head and layer, and reads them all for every new token; the queries are made for the new token only and thrown away. So sharing keys and values shrinks what is kept and read, by $h / g$, and sharing queries would save almost nothing. **Why little quality is lost:** every head still asks its own question with its own queries, so heads in a group can still look at different tokens; only what they find there is shared. With a few groups, as in Llama 2 70B and Mistral, the quality stays close to multi-head attention.",
        points=[
            "you say that keys and values of every earlier token are kept and read while generating, and queries are not",
            "you say that sharing them shrinks that memory by h / g",
            "you say that every head keeps its own queries, so heads in a group can still look at different tokens",
        ],
    )
    explain_keys_cell
    return (explain_keys_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The heads of [Multi-head attention](/course/open/attn-multi-head?lang=en), with groups of query heads sharing the columns of one key and one value head. The next lesson, *The KV cache*, keeps those keys and values while generating, and counts what that memory costs.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_share, explain_keys_cell, result_gqa):
    L.wrap_up(result_gqa, choice_share, explain_keys_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
