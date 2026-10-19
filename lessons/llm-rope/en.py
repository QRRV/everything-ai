# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-rope. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("llm-rope", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Rotating vectors and complex numbers](/course/open/math-complex-rotation?lang=en) and [Telling attention where things are](/course/open/attn-positions?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_turn():
    quick_turn = L.choice(
        "quick-turn",
        question="You turn the pair $(5, 12)$ by 90 degrees, counterclockwise. What do you get?",
        options={
            "right": "$(-12, 5)$",
            "clock": "$(12, -5)$",
            "same": "$(5, 12)$: a turn keeps the numbers",
        },
        correct="right",
        why={
            "right": "$(a\\cos\\theta - b\\sin\\theta,\\; a\\sin\\theta + b\\cos\\theta)$ with $\\cos 90° = 0$ and $\\sin 90° = 1$. Its length stays 13.",
            "clock": "That is a quarter turn clockwise, by $-90$ degrees. See [Rotating vectors and complex numbers](/course/open/math-complex-rotation?lang=en).",
            "same": "A turn keeps the length, 13, not the numbers themselves. See [Rotating vectors and complex numbers](/course/open/math-complex-rotation?lang=en).",
        },
    )
    quick_turn
    return (quick_turn,)


@app.cell(hide_code=True)
def quick_check_params():
    quick_params = L.choice(
        "quick-params",
        question="How many parameters does the sinusoidal positional encoding of [Telling attention where things are](/course/open/attn-positions?lang=en) have?",
        options={
            "places": "One per place: a number that says where the token is",
            "both": "One per place and per feature",
            "right": "None: it is computed",
        },
        correct="right",
        why={
            "places": "That would be a learned table, one number per place. The sinusoids come from a formula. See [Telling attention where things are](/course/open/attn-positions?lang=en).",
            "both": "That is GPT-2's learned position embedding. The sinusoids come from a formula, with nothing to learn.",
            "right": "Sines and cosines of $t / 10000^{2j/d}$: nothing to learn. RoPE uses the same speeds, and has no parameters either.",
        },
    )
    quick_params
    return (quick_params,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Your GPT adds a learned vector to every token for its place: one row per place, up to the longest text it trained on. Llama, Mistral, Qwen and most open models since then do something else: **rotary position embeddings**, RoPE. They add nothing to the tokens. Inside every attention layer, they turn each query and each key by an angle that depends on its place.
    """)
    return


@app.cell(hide_code=True)
def predict_same():
    guess_same = L.guess(
        "same-distance",
        question="With RoPE, you compare a query at place 10 with a key at place 7. Then you put the same two vectors, before turning, at places 110 and 107. What happens to their score?",
        options={
            "lower": "It is lower: the tokens are further from the start",
            "right": "It is the same",
            "other": "It is different, in a way that depends on the vectors",
        },
        reveal="**It is the same.** The query turns by its place and the key by its place, so their score depends only on the difference of their places: 3 in both cases. As [Rotating vectors and complex numbers](/course/open/math-complex-rotation?lang=en) showed, a dot product of turned pairs sees only the differences of their angles.",
    )
    guess_same
    return (guess_same,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Step 1: turn every pair by its place.** Read a query or key of $d$ numbers as $d/2$ pairs: entries 0 and 1, 2 and 3, and so on. At place $t$, pair $j$ turns by the angle
        """),
        L.formula(r"t \, \theta_j \qquad \theta_j = 10000^{-2j/d}"),
        mo.md(r"""
        These are the speeds of the sinusoids of [Telling attention where things are](/course/open/attn-positions?lang=en): pair 0 turns 1 radian per place, the last pair hardly at all. Code calls the 10000 `base`. Fast pairs tell tokens next to each other apart, slow pairs tokens far apart.

        **Step 2: only the distance counts.** A query at place $m$ turns by $m\theta_j$, a key at place $n$ by $n\theta_j$, and their dot product depends only on the differences, $(m - n)\theta_j$. So the score between the two depends on $m - n$: two tokens 3 places apart score alike anywhere in the text.

        **Step 3: where it goes.** In every attention layer, after the projections, on the queries and keys of every head, with $d$ the size of one head. The values are not turned. There are no position embeddings at the input, and RoPE has no parameters.

        **Step 4: what it brings.** Relative positions, without a table. When a query and a key point the same way, as the 64 ones of the picture do, their score tends to fall as the tokens get further apart. A model still knows only the distances it trained on: beyond them, the slow pairs reach angles it never saw. Ways to stretch the context, such as position interpolation, slow the turning down so that a longer text stays within the angles the model knows, usually with a short round of extra training on long texts.

        The picture: the score of a query and a key of 64 ones, both turned by RoPE and divided by $\sqrt{64}$ as in attention, against the distance between their places.
        """),
        mo.Html(C.distance_picture({"title": "Score against distance", "x": "distance between the places", "y": "score", "lang": L.lang,
                                    "alt": "A curve over distances 0 to 60. The score is 8 at distance 0, falls to about 5.3 at distance 10, and then swings up and down while it sinks slowly, between about 3.3 and 5.3, to about 4 at distance 60."})),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **RoPE** | rotary position embeddings: turning each pair of a query or key by its place times its speed | `rope(q)`, `rope(k)` |
        | $\theta_j$ | the speed of pair $j$: radians per place | `base ** (-2 * j / d)` |
        """),
        L.watch_out("Code uses two layouts. This lesson, the RoPE paper and Meta's Llama code pair entries that sit next to each other, 0 and 1; Hugging Face's Llama pairs the first half of a vector with the second half. A trained model's weights expect the layout it was trained with."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To apply RoPE to a tensor of queries or keys, with the places along the second-to-last axis:

    1. Make the speeds, $\theta_j = 10000^{-2j/d}$, one per pair.
    2. Make the angles: place times speed, a table of one row per place and one column per pair.
    3. Turn every pair by its angle: $(a\cos - b\sin,\; a\sin + b\cos)$.
    4. Do it to the queries and the keys, before their scores.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The table of angles is new: the speeds of 4 pairs, $d = 8$, and their angles at the places 0 to 3.
    """)
    return


@app.cell
def example_angles():
    import torch

    _d, _n = 8, 4
    _theta = 10000 ** (-torch.arange(0, _d, 2) / _d)
    _angles = torch.arange(_n)[:, None] * _theta[None, :]
    print("speeds:", [round(_s, 3) for _s in _theta.tolist()])
    for _t in range(_n):
        print("place", _t, [round(_a, 3) for _a in _angles[_t].tolist()])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `torch.arange(0, 8, 2)` gives the entries 0, 2, 4 and 6, the first of each pair, so the speeds are $10000^{-0/8}, 10000^{-2/8}, \ldots$: 1, 0.1, 0.01 and 0.001. `[:, None]` makes the places a column and `[None, :]` the speeds a row; their product is the table of angles, place times speed, with a row of zeros for place 0: the first token is not turned.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rope():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Your own rope

        Write `rope(x, base=10000.0)`: `x` has the places along its second-to-last axis and an even number of entries along its last, with any axes in front, such as (texts, heads, places, entries). Give back a tensor of the same shape with every pair turned by its angle. `x[..., 0::2]` takes the entries 0, 2, 4, … of the last axis, whatever axes come before it: `...` stands for all of them. `torch.cos` and `torch.sin` take the cosine and sine of every number of a tensor, as `np.cos` and `np.sin` do for an array. The check calls `rope(torch.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]]))`, which should give about `[[1, 0, 1, 0], [0.5403, 0.8415, 0.9999, 0.01]]`, and tries other shapes.
        """),
        L.task("Write `rope`: the angles of all places and pairs, and every pair of `x` turned by its angle."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rope():
    box_rope = L.code_box("rope", lines=11)
    box_rope
    return (box_rope,)


@app.cell(hide_code=True)
def check_rope(box_rope):
    result_rope = L.check("rope", box_rope.value)
    result_rope
    return (result_rope,)


@app.cell(hide_code=True)
def hints_rope():
    L.hints(
        "rope",
        [
            "The table of angles of the worked example, then the turn of [Rotating vectors and complex numbers](/course/open/math-complex-rotation?lang=en) on all pairs at once: `x[..., 0::2]` are the first entries of all pairs, `x[..., 1::2]` the second.",
        ],
        why="With `a = x[..., 0::2]` and `b = x[..., 1::2]`, the turned pairs are `a * cos - b * sin` and `a * sin + b * cos`, with `cos` and `sin` of the table of angles; written back into the even and the odd entries of a new tensor, they keep the layout of `x`.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### The same distance, other places

    The cell below turns one random query and one random key of 64 numbers with your `rope`, at five pairs of places, and gives their score, divided by $\sqrt{64}$ as in attention. It works once exercise 1 is right.
    """)
    return


@app.cell(hide_code=True)
def demo_places(box_rope, result_rope):
    if result_rope.state != "correct":
        _out = mo.md("*Finish exercise 1 first: this cell uses your `rope`.*")
    else:
        _rows = C.demo(box_rope.value)
        if _rows == "error":
            _out = mo.md("*Your `rope` stopped with an error on vectors of 64 numbers at 504 places.*")
        else:
            _table = "\n".join(f"| {_m} | {_k} | {_m - _k} | {C.number(_s, 3, L.lang)} |" for _m, _k, _s in _rows)
            _out = mo.md(f"""
    | query at | key at | distance | score |
    |---|---|---|---|
    {_table}

    At distance 3, the score is the same at places 3 and 0, at 10 and 7, and at 503 and 500. At other distances it is another score.
    """)
    _out
    return


@app.cell(hide_code=True)
def text_exercise_where():
    mo.md(r"""
    ### Exercise 2. Where the place comes in
    """)
    return


@app.cell(hide_code=True)
def choice_where_cell():
    choice_where = L.choice(
        "where",
        question="In a GPT with RoPE, where does the place of a token come in?",
        options={
            "input": "Added to the token embeddings, once, at the input",
            "right": "In every attention layer: queries and keys are turned by their place",
            "values": "In every attention layer: queries, keys and values are turned, so the output carries the place too",
        },
        correct="right",
        why={
            "input": "That is where GPT-2's learned position embeddings come in. RoPE adds nothing at the input.",
            "values": "RoPE leaves the values alone: the place matters for the scores, which decide where a token looks.",
            "right": "After the projections, in every head of every layer, and only on the queries and keys.",
        },
    )
    choice_where
    return (choice_where,)


@app.cell(hide_code=True)
def text_exercise_values():
    mo.md(r"""
    ### Exercise 3. Not the values
    """)
    return


@app.cell(hide_code=True)
def explain_values():
    explain_values_cell = L.explain(
        "not-the-values",
        question="RoPE turns the queries and the keys, but not the values. Why is that enough for a model to use where its tokens are, and what would turning the values as well do?",
        model_answer="**Why enough:** the scores, from queries and keys, decide how much each token takes from each other token. With RoPE, they depend on the distance between the two, so the attention weights already carry where things are: a head can learn to look at the token before, or at tokens far back. **The values:** they carry the content that is mixed. Turned by their own places, the values of different tokens would point in different directions for the same content, and the weighted sum would mix copies turned by different angles, which partly cancel; the layers after attention could not simply turn the sum back.",
        points=[
            "you say that the scores decide where a token looks, and that with RoPE they depend on the distance",
            "you say that the values carry the content that is mixed",
            "you say that turning the values would mix the same content in different directions, which the layers after attention cannot simply turn back",
        ],
    )
    explain_values_cell
    return (explain_values_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The turns of [Rotating vectors and complex numbers](/course/open/math-complex-rotation?lang=en), at the speeds of the sinusoids of [Telling attention where things are](/course/open/attn-positions?lang=en), now inside the attention of [Assemble your GPT](/course/open/gpt-assemble?lang=en). The next lesson, *Grouped-query attention*, changes how many keys and values attention keeps.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_where, explain_values_cell, result_rope):
    L.wrap_up(result_rope, choice_where, explain_values_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
