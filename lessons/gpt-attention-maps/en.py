# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson gpt-attention-maps. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("gpt-attention-maps", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Attention weights from dot products](/course/open/attn-scores?lang=en) and [Multi-head attention](/course/open/attn-multi-head?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sum():
    quick_sum = L.choice(
        "quick-sum",
        question="What do the attention weights in one row of a head add up to?",
        options={
            "positions": "The number of positions in the row",
            "right": "1",
            "free": "Nothing fixed: they are scores",
        },
        correct="right",
        why={
            "positions": "Softmax turns the scores of a row into weights that add up to 1. See [Attention weights from dot products](/course/open/attn-scores?lang=en).",
            "free": "The scores can be anything; softmax turns them into weights that add up to 1. See [Attention weights from dot products](/course/open/attn-scores?lang=en).",
            "right": "Each row is a weighted average: it shares out 1 over the positions it may see. This lesson reads those rows.",
        },
    )
    quick_sum
    return (quick_sum,)


@app.cell(hide_code=True)
def quick_check_count():
    quick_count = L.choice(
        "quick-count",
        question="A GPT with `n_layers` blocks of `h` heads each reads one text. How many matrices of attention weights does it make?",
        options={
            "heads": "`h`: all blocks share the same heads",
            "blocks": "`n_layers`: one per block",
            "right": "`n_layers` × `h`: one for every head",
        },
        correct="right",
        why={
            "heads": "Every block has its own attention with its own weights. See [Multi-head attention](/course/open/attn-multi-head?lang=en).",
            "blocks": "Every head of a block works out its own weights, side by side. See [Multi-head attention](/course/open/attn-multi-head?lang=en).",
            "right": "The course's GPT has 5 blocks of 4 heads: 20 matrices for every text, and this lesson looks at them.",
        },
    )
    quick_count
    return (quick_count,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Your GPT of [Project: train your GPT on a book](/course/open/gpt-train-chars?lang=en) writes text that looks like Shakespeare. But what does it look at, when it predicts the next character? Its attention weights are the one place where it says, for every position, how much it takes from every position before it. Drawn as a picture, they show that heads take on jobs of their own. Looking inside a model like this is the start of understanding why it does what it does.
    """)
    return


@app.cell(hide_code=True)
def predict_heads():
    guess_heads = L.guess(
        "previous-heads",
        question="The course's GPT has 5 blocks of 4 heads: 20 heads. How many of them put more than half of their weight, on average, on the character just before?",
        options={
            "none": "None",
            "right": "One",
            "many": "More than ten",
        },
        reveal=f"**One:** head 2 of block 0, which puts about {L.number(C.previous_score(C.course_maps(C.DEMO_TEXT)[0][2].tolist()), 2)} of its weight there, on any text from the plays. The other 19 stay below half: the next ones put about 0.4 there. One head is enough to bring the character before into every position; other heads do other jobs.",
    )
    guess_heads
    return (guess_heads,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **One matrix per head.** For a text of $n$ characters, a head gives an $n \times n$ matrix of attention weights $W$: row $t$ holds the weights position $t$ gives to the positions $0$ to $t$, and adds up to 1. Above the diagonal everything is 0: the causal mask.

        **An attention heatmap** draws that matrix with a square per weight, darker the larger: the rows are the positions that look, the columns the positions they look at. Patterns to look for:

        - a dark stripe just below the diagonal: the head looks at the character just before;
        - a dark diagonal: the head looks at the character itself;
        - dark columns: the head looks back at certain characters, such as new lines;
        - an even grey: the head takes an average of everything before.

        **Measure it.** With 20 heads and long texts, you want numbers too. How much a head looks at the character before, averaged over the rows that have one, is its score $p$:
        """),
        L.formula(r"p = \frac{1}{n - 1} \sum_{t=1}^{n-1} [[hidden:W_{t, t-1}]]"),
        mo.md(r"""
        And how much it looks at some characters, such as spaces and new lines: for every row, the sum of its weights on positions with such a character, averaged over the rows. Compare that with a head that spreads its weight evenly: it gets about the share of such characters among the positions it sees.

        The picture: two heads of the course's GPT reading a line of Juliet, with her name on the line before it, as in the plays. Head 2 of block 0 looks at the character just before; head 3 of block 3 looks back at the new line, where the line starts. Blocks and heads count from 0, as in Python.
        """),
        mo.Html(C.heatmap(C.course_maps()[0][2], C.TEXT, "Block 0, head 2", "A heatmap of 25 by 25 squares for the text JULIET:, a new line, and What's in a name? Almost all the weight lies on a stripe just below the diagonal: every character looks at the character just before it.",
                          {"cols": "looks at →", "rows": "↓ the character that looks"})),
        mo.Html(C.heatmap(C.course_maps()[3][3], C.TEXT, "Block 3, head 3", "A heatmap of 25 by 25 squares for the same text. On the second line, the darkest squares lie in the column of the new line: most characters of the line look back at the new line where it starts.",
                          {"cols": "looks at →", "rows": "↓ the character that looks"})),
        mo.md(r"""
        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **attention heatmap** | | a picture of a head's attention weights: a row for every position that looks, a column for every position it looks at, darker for a larger weight | |
        | **previous-token head** | | a head that puts most of its weight on the token just before | |
        | $W_{t, t-1}$ | W t, t minus 1 | the weight position $t$ gives to the position before it | `W[t][t - 1]` |
        | $p$ | p | a head's score on the character before: the average of $W_{t, t-1}$ | `previous_score(W)` |
        """),
        L.watch_out("A heatmap shows where a head takes from, not why the model predicts what it predicts. What a head takes is the value of a position, which the output projection, the MLP and the heads of later blocks still change. A large weight on a character does not mean the prediction depends on it: to find out, change the input or switch the head off, and see what the prediction does."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To find out what the heads of a GPT look at:

    1. Run a text through the GPT, and keep the matrix of attention weights of every head of every block.
    2. Draw each as a heatmap, and look for stripes, diagonals and columns.
    3. Measure every head: its score on the character before, and its share on the characters you are curious about.
    4. Compare with a head that spreads its weight evenly, and look closer at the heads that stand out.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Step 4: what share of its weight does a head that spreads it evenly give to spaces, in "to be or not"? Row $t$ gives $1 / (t + 1)$ to each of the $t + 1$ characters it sees, so its share on spaces is the number of spaces among them divided by $t + 1$. The comments with `#` say what each part is. The line in the loop is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_even():
    _text = "to be or not"
    _targets = " "
    _total = 0.0
    for _t in range(len(_text)):
        # the characters row t sees, and how many of them are targets
        _seen = _text[:_t + 1]
        _hits = sum([1 for _c in _seen if _c in _targets])
        _total = _total + 0.0  # complete: _hits / len(_seen)
    # averaged over the rows
    print(round(_total / len(_text), 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    As it is, the cell prints 0.0. Completed, it prints 0.234: a head that spreads evenly gives about a quarter of its weight to spaces here, because 3 of the 12 characters are spaces, and the first rows see none yet. A head that gives spaces much more than its even share has a job to do with them.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_previous():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The score on the character before

        Complete `previous_score(W)`: for the attention weights `W` of one head, a list of rows, give back the average weight on the character just before, over the rows that have one, as in the formula. The check calls `previous_score([[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.1, 0.6, 0.3]])`, which should give 0.7, and tries 4 positions and 2.
        """),
        L.task("Make `previous_score` give the average of `W[t][t - 1]` over the rows from 1 on."),
    ])
    return


@app.cell(hide_code=True)
def exercise_previous():
    box_previous = L.code_box("previous-score", lines=5)
    box_previous
    return (box_previous,)


@app.cell(hide_code=True)
def check_previous(box_previous):
    result_previous = L.check("previous-score", box_previous.value)
    result_previous
    return (result_previous,)


@app.cell(hide_code=True)
def hints_previous():
    L.hints(
        "previous-score",
        [
            "Row 0 has no character before it, so the rows run from 1 to `n - 1`: that is `n - 1` rows.",
            "In row `t`, the character before is in column `t - 1`. Add those up and divide by the number of rows.",
        ],
        why="`return sum([W[t][t - 1] for t in range(1, n)]) / (n - 1)`. For the example: (0.8 + 0.6) / 2 = 0.7.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_share():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The share on some characters

        Complete `share_on(W, text, targets)`: for every row `t`, add the weights on the positions `j` up to `t` whose character `text[j]` is one of `targets`, and give back the average over the rows. The check calls `share_on([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.2, 0.7, 0.1, 0.0], [0.1, 0.1, 0.6, 0.2]], "a b,", " ,")`, which should give 0.375, and tries other texts and targets.
        """),
        L.task("Make `share_on` give the average weight a row puts on the target characters."),
    ])
    return


@app.cell(hide_code=True)
def exercise_share():
    box_share = L.code_box("share-on", lines=10)
    box_share
    return (box_share,)


@app.cell(hide_code=True)
def check_share(box_share):
    result_share = L.check("share-on", box_share.value)
    result_share
    return (result_share,)


@app.cell(hide_code=True)
def hints_share():
    L.hints(
        "share-on",
        [
            "Inside the two loops: is the character at position `j` one of the targets? Then add the weight row `t` gives to position `j`.",
            "`if text[j] in targets:` and `W[t][j]`; at the end, divide `total` by the number of rows.",
        ],
        why="`if text[j] in targets: total = total + W[t][j]`, and `return total / n`. For the example, the targets are at positions 1 and 3: the rows give 0, 0.5, 0.7 and 0.1 + 0.2 = 0.3, and 1.5 / 4 = 0.375.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### The heads of your GPT

    The cell below runs Romeo's name and two of his lines, 97 characters, through your trained GPT of [Project: train your GPT on a book](/course/open/gpt-train-chars?lang=en), or the course's if you have not trained one. It measures all heads with your two functions, where they are right, and draws the head with the highest score on the character before and the one with the highest share on new lines, for the first 24 characters.
    """)
    return


@app.cell(hide_code=True)
def demo_heads(box_previous, box_share, result_previous, result_share):
    _r = C.demo(box_previous.value if result_previous.state == "correct" else None,
                box_share.value if result_share.state == "correct" else None)
    if _r["source"] is None:
        _out = mo.md("There is no trained GPT to look at: the course's checkpoint is missing from `data/`.")
    else:
        _gpt = "your GPT" if _r["source"] == "yours" else "the course's GPT"
        _p_who = "your `previous_score`" if _r["yours"][0] else "the course's `previous_score`"
        _s_who = "your `share_on`" if _r["yours"][1] else "the course's `share_on`"
        _pl, _ph, _pv, _ = _r["prev"]
        _ml, _mh, _, _mv = _r["marks"]
        _out = mo.vstack([
            mo.md(f"In {_gpt}, measured with {_p_who} and {_s_who}: head {_ph} of block {_pl} has the highest score on the character before, {L.number(_pv, 2)}, and head {_mh} of block {_ml} the highest share on new lines, {L.number(_mv, 2)}. A head that spreads its weight evenly would give them {L.number(_r['uniform'], 2)}."),
            mo.Html(C.heatmap(_r["maps"][_pl][_ph][:24, :24], C.DEMO_TEXT[:24], f"Block {_pl}, head {_ph}", "A heatmap of the head with the highest score on the character before.",
                              {"cols": "looks at →", "rows": "↓ the character that looks"})),
            mo.Html(C.heatmap(_r["maps"][_ml][_mh][:24, :24], C.DEMO_TEXT[:24], f"Block {_ml}, head {_mh}", "A heatmap of the head with the highest share on new lines.",
                              {"cols": "looks at →", "rows": "↓ the character that looks"})),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_explain():
    mo.md(r"""
    ### Exercise 3. What a heatmap says
    """)
    return


@app.cell(hide_code=True)
def choice_explain_cell():
    choice_explain = L.choice(
        "not-explanation",
        question="In the course's GPT, head 2 of block 0 puts about 0.9 of its weight on the character just before. What does that tell you about the prediction of the next character?",
        options={
            "right": "Only where this head takes from, not how much the prediction depends on it",
            "depends": "That the prediction of the next character depends mostly on the character just before",
            "nothing": "Nothing: attention weights are random numbers",
        },
        correct="right",
        why={
            "depends": "The head brings in the value of the character before, but 19 other heads, the MLPs and the output projections also shape the prediction. The weight alone does not say how much it counts.",
            "nothing": "The weights are not random: they come from the queries and keys the GPT learned, and the same head does the same on every text.",
            "right": "A heatmap is a good place to start looking. Whether it matters, you find out by changing the input or switching the head off, and seeing what the prediction does.",
        },
    )
    choice_explain
    return (choice_explain,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The attention weights of [Attention weights from dot products](/course/open/attn-scores?lang=en), with the causal mask of [The causal mask](/course/open/attn-causal-mask?lang=en), now read from the heads of [Multi-head attention](/course/open/attn-multi-head?lang=en) in your trained GPT, and drawn as heatmaps, as the bigram table of [The bigram table as a matrix of probabilities](/course/open/lm-bigram-matrix?lang=en) was. In larger models, measures like these, on text that repeats itself, helped find heads that look at what followed an earlier occurrence of the current token, and copy it into the prediction.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_explain, result_previous, result_share):
    L.wrap_up(result_previous, result_share, choice_explain)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
